"""端到端调用流程示例：模拟 vLLM 逐层调度 × 条带化存储 backend。

学习向脚本（不是 pytest 用例）。两部分教学要点：

1. **模拟 vLLM 的按层调度**：V1 的层间钩子（maybe_transfer_kv_layer）
   在每层 forward 前后调 wait_for_layer_load / save_kv_layer。本脚本
   用 VLLMForwardSim 复刻这个调用序：命中请求 start_load_kv 后，
   wait(L) 完成即"计算" L，同时 L+1 已在后台读取（预取重叠）；
   prefill 请求每层"计算"完立即 save_kv_layer 落盘该层。

2. **条带化 backend**：StripedSimStore 按 tutti 条带语义实现
   （逻辑 chunk 文件按 stripe unit 切条，轮询分布到多个"设备"
   分片 <dev>/striped/<chunk>.shard<i>）。它只实现 KVStore SPI，
   KVEngine 与调度模拟零改动——持久化形态是插件私有的。

真实 striped:// 后端（runtime 决定分片、跨物理盘）是 connector
store 的演进项；当前 TuttiKVStore 支持 file:// 单盘 target。

3. **staging 槽位生命周期**（阶段 5）：槽位可复用的充分条件是
   "DMA 完成 ∧ 消费方 scatter 已读完"（组合完成句柄 + 事件链，
   禁止全局同步）。阶段 5 用可控时序的句柄复现 wave-2 回绕场景，
   展示"DMA 完成但 scatter 未读完时，回绕波次被挡在槽外"。

Run:
    cd integration/vllm-connector && \
    /data/home/ryeqiu/tutti-env/bin/python tests/e2e_walkthrough.py
"""

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.core import KVEngine
from engine.nvtx import range as nvtx_range
from engine.staging import RingWindow
from index.chunk_index import derive_io_key  # noqa: F401  (教学展示用)
from stores.base import KVStore  # noqa: F401  (SPI 契约参照)

# ---- 几何（教学用小尺寸；生产为 256 token/chunk、80 层、512 chunk/波）----
NUM_LAYERS = 4          # 层数（≥3 才能看出 L+1 预取与 L 计算的重叠）
CHUNK_TOKENS = 4        # 每 chunk 的 token 数
SEGMENT_BYTES = 4096    # 层段字节 = chunk_kv_bytes // num_layers（对齐粒度）
CHUNK_KV_BYTES = SEGMENT_BYTES * NUM_LAYERS
MAX_WAVE = 4            # 单波最大 chunk 数（半窗容量）
POOL_CHUNKS = 2         # 存储池容量（chunk 数，小池便于演示淘汰）
# ---- 条带几何 ----
NUM_DEVICES = 4         # 条带"设备"数（对照真机 4 盘）
STRIPE_UNIT = 1024      # 条带单元字节：每层段 4 KiB → 4 条，恰跨 4 设备


class StripedSimStore:
    """条带化 KVStore 教学实现（KVStore SPI，同步完成）。

    语义对照 tutti 条带 target：逻辑 chunk 文件 = num_layers ×
    segment_bytes；每个 IO（一层段）按 stripe_unit 切条，第 k 条落
    设备 (chunk 起始设备 + k) % num_devices——分片分布私有，引擎只见
    (io_key, buffer_id, offset) 三元组。
    """

    def __init__(self, segment_bytes, num_chunks, num_devices, stripe_unit):
        if segment_bytes % stripe_unit:
            raise ValueError("segment_bytes 须为 stripe_unit 的整数倍")
        self._segment_bytes = segment_bytes
        self._capacity = num_chunks
        self._num_devices = num_devices
        self._stripe_unit = stripe_unit
        self._opened = False
        self._buffers: dict[int, memoryview] = {}
        self._next_buffer_id = 0
        # io_key → {stripe_idx: (device, bytes)}；chunk 起始设备由 key 定
        self._stripes: dict[bytes, dict[int, tuple[int, bytes]]] = {}

    # ---- SPI ----

    @property
    def capacity_chunks(self):
        return self._capacity

    def open(self):
        self._opened = True

    def close(self):
        self._opened = False
        self._stripes.clear()
        self._buffers.clear()

    def register_buffer(self, buffer, granularity):
        if granularity <= 0 or granularity % 4096 or granularity > self._segment_bytes:
            return None
        view = memoryview(buffer)
        if view.readonly:
            return None
        self._next_buffer_id += 1
        self._buffers[self._next_buffer_id] = view.cast("B")
        return self._next_buffer_id

    def put_batch(self, batch):
        for key, buffer_id, offset in batch:
            data = bytes(self._buffers[buffer_id][offset:offset + self._segment_bytes])
            self._stripes[key] = {
                k: (self._stripe_device(key, k),
                    data[k * self._stripe_unit:(k + 1) * self._stripe_unit])
                for k in range(self._segment_bytes // self._stripe_unit)
            }
        return _Done()

    def get_batch(self, batch):
        for key, buffer_id, offset in batch:
            stripes = self._stripes[key]
            data = b"".join(
                stripes[k][1] for k in range(self._segment_bytes // self._stripe_unit)
            )
            self._buffers[buffer_id][offset:offset + self._segment_bytes] = data
        return _Done()

    def drop(self, keys):
        for key in keys:
            self._stripes.pop(key, None)

    def scan(self):
        return list(self._stripes)

    # ---- 条带分布（教学展示）----

    def _stripe_device(self, io_key, stripe_idx):
        start = int.from_bytes(io_key[:16], "little") % self._num_devices
        return (start + stripe_idx) % self._num_devices

    def placement(self, io_key) -> list[int]:
        """返回该 io_key 各条带所在设备序（教学打印用）。"""
        return [self._stripes[io_key][k][0]
                for k in sorted(self._stripes[io_key])]


class _Done:
    """同步完成句柄（教学 store 无异步）。"""

    def wait(self):
        return None

    def query(self):
        return True


# ---- 阶段 5 专用：可控时序的句柄与事件 ----


class ManualHandle:
    """完成时机由外部控制的完成句柄（模拟在途 DMA）。"""

    def __init__(self, on_complete=None):
        self._ev = threading.Event()
        self._on_complete = on_complete

    def wait(self):
        self._ev.wait()

    def query(self):
        return self._ev.is_set()

    def complete(self):
        if self._on_complete is not None:
            self._on_complete()
        self._ev.set()


class FakeEvent:
    """消费方事件（模拟 scatter 的 CUDA event）。"""

    def __init__(self):
        self._ev = threading.Event()

    def synchronize(self):
        self._ev.wait()

    def query(self):
        return self._ev.is_set()

    def record(self):
        self._ev.set()


class ManualStore(StripedSimStore):
    """get_batch 不在调用时搬数，返回 ManualHandle；complete 时才落地。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inflight: list[ManualHandle] = []

    def get_batch(self, batch):
        def _land():
            super(ManualStore, self).get_batch(batch)

        handle = ManualHandle(on_complete=_land)
        self.inflight.append(handle)
        return handle


class DictHooks:
    """搬运钩子替身：staging 槽与"显存"之间的字节搬运（纯 Python）。

    真实部署中由 adapter 按池几何构造（GPU kernel / 块表索引拷贝）。
    """

    def __init__(self, window_buffer, segment_bytes):
        self._buf = window_buffer
        self._seg = segment_bytes
        self.src: dict[tuple[bytes, int], bytes] = {}
        self.dst: dict[tuple[bytes, int], bytes] = {}

    def gather(self, keys, layer_idx, first_blocks, slots):
        for key, slot in zip(keys, slots):
            off = slot * self._seg
            self._buf[off:off + self._seg] = self.src[(key, layer_idx)]

    def scatter(self, keys, layer_idx, first_blocks, slots):
        for key, slot in zip(keys, slots):
            off = slot * self._seg
            self.dst[(key, layer_idx)] = bytes(self._buf[off:off + self._seg])


class VLLMForwardSim:
    """模拟 vLLM V1 的逐层调度钩子序（adapter/worker 视角）。

    - prefill：每层"计算"产出该层 KV 后立即 save_kv_layer（vLLM 在
      层 forward 末尾调用）；save 间天然与引擎在途批次重叠。
    - 命中：start_load_kv 组批并发起第 0 层读取；wait_for_layer_load(L)
      等 L 落地后，先发起 L+1 读取再"计算" L——L+1 的 DMA 与 L 的
      计算重叠，这就是逐层流水（load_kv_async=False 契约的履行方式）。
    """

    def __init__(self, engine, num_layers):
        self._engine = engine
        self._num_layers = num_layers
        self.timeline: list[str] = []

    def prefill_save(self, keys, make_layer_kv):
        """逐层 save：make_layer_kv(layer) 模拟该层 forward 产出的 KV。"""
        self._engine.plan_store(keys)
        for layer in range(self._num_layers):
            with nvtx_range(f"tutti.sim.compute|phase=prefill|layer={layer}"):
                make_layer_kv(layer)  # "forward 计算"（写出源侧层段）
            self._engine.store_layer(keys, layer, src_first_blocks=None)
            self.timeline.append(f"save L{layer} 已发起（不等待，下层继续）")
        self._engine.wait_idle()
        self._engine.confirm_store(keys)
        self.timeline.append("wait_idle + confirm_store：全部层结算驻留")

    def hit_forward(self, keys, compute):
        """逐层 load + 计算：wait L → 预取 L+1 → compute L。"""
        self._engine.pin(keys)                      # start_load_kv 的读保护
        inflight = self._engine.load_layer(keys, 0, dst_first_blocks=None)
        self.timeline.append("load L0 发起（start_load_kv 预取首层）")
        for layer in range(self._num_layers):
            inflight.wait()                         # wait_for_layer_load(L)
            self.timeline.append(f"wait L{layer} 完成（KV 已在目的侧）")
            nxt = layer + 1
            if nxt < self._num_layers:
                inflight = self._engine.load_layer(keys, nxt, dst_first_blocks=None)
                self.timeline.append(f"load L{nxt} 发起（与 L{layer} 计算重叠）")
            with nvtx_range(f"tutti.sim.compute|phase=decode|layer={layer}"):
                compute(layer)                      # "attention 计算" L
        self._engine.unpin(keys)


def show(title, engine, store):
    keys = sorted(k.hex()[:8] for k in engine._index._resident)  # noqa: SLF001
    on_disk = sorted((k[:16].hex()[:8], int.from_bytes(k[16:18], "little"))
                     for k in store.scan())
    print(f"  [{title}] 索引驻留={keys}")
    print(f"  [{title}] 盘上 io_key(chunk, layer)={on_disk}")


def show_stripes(store, keys, num_layers):
    """打印条带分布：每行一个 io_key 的各条带所在设备。"""
    for key in keys:
        for layer in range(num_layers):
            io_key = derive_io_key(key, layer)
            devs = store.placement(io_key)
            print(f"    chunk {key.hex()[:8]} L{layer}: "
                  f"条带设备序 {devs}（shard<i> on dev{devs}）")


def main():
    # ================================================================
    # 阶段 0：初始化
    # ================================================================
    print("=" * 70)
    print("阶段 0：初始化 —— 条带 store + 语义索引 + staging 环窗绑定")
    print("=" * 70)

    store = StripedSimStore(SEGMENT_BYTES, POOL_CHUNKS, NUM_DEVICES, STRIPE_UNIT)
    engine = KVEngine(
        config={
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_kv_bytes": CHUNK_KV_BYTES,
            "max_chunks_per_wave": MAX_WAVE,
        },
        store=store,
    )
    window = RingWindow(bytearray(2 * MAX_WAVE * SEGMENT_BYTES),
                        2 * MAX_WAVE, SEGMENT_BYTES)
    hooks = DictHooks(window.buffer, SEGMENT_BYTES)
    engine.bind({}, window, NUM_LAYERS, 1,
                gather_fn=hooks.gather, scatter_fn=hooks.scatter)
    print(f"  bind 完成：staging buffer_id={engine._staging_buffer_id}，"  # noqa: SLF001
          f"条带 {NUM_DEVICES} 设备 × stripe {STRIPE_UNIT}B")
    sim = VLLMForwardSim(engine, NUM_LAYERS)

    # ================================================================
    # 阶段 1：prefill 落盘（vLLM 逐层调度模拟）
    # ================================================================
    print()
    print("=" * 70)
    print("阶段 1：prefill —— 每层 forward 产出 KV 后逐层 save_kv_layer")
    print("=" * 70)

    prompt_a = [10, 11, 12, 13, 14, 15, 16, 17]   # 2 个 chunk
    keys_a, _ = engine.hash_keys(prompt_a)
    print(f"  prompt_a → chunk keys：{[k.hex()[:8] for k in keys_a]}")

    def make_layer_kv(layer):
        for i, key in enumerate(keys_a):
            hooks.src[(key, layer)] = bytes([layer * 16 + i] * SEGMENT_BYTES)

    sim.prefill_save(keys_a, make_layer_kv)
    for line in sim.timeline:
        print(f"    {line}")
    sim.timeline.clear()
    show("prefill 落盘后", engine, store)

    print("  条带分布（引擎不可见，store 私有）：")
    show_stripes(store, keys_a, NUM_LAYERS)

    # ================================================================
    # 阶段 1.5：写入即时读回 —— get_batch 直读 store 独立校验写路径
    # （不经过 pin/load/scatter 的加载链路，专验 gather→put_batch→分片）
    # ================================================================
    print()
    print("=" * 70)
    print("阶段 1.5：写入即时读回 —— 绕开 load 路径直读，单验写路径")
    print("=" * 70)

    probe = bytearray(SEGMENT_BYTES)
    probe_id = store.register_buffer(probe, SEGMENT_BYTES)
    bad = 0
    for i, key in enumerate(keys_a):
        for layer in range(NUM_LAYERS):
            store.get_batch([(derive_io_key(key, layer), probe_id, 0)]).wait()
            if bytes(probe) != hooks.src[(key, layer)]:
                bad += 1
    total = NUM_LAYERS * len(keys_a)
    print(f"  直读校验：{total} 段，不一致 {bad} 段"
          f"{'（PASS：写路径独立成立）' if bad == 0 else '（FAIL）'}")
    assert bad == 0

    # ================================================================
    # 阶段 2：语义索引 —— scheduler 命中查询
    # ================================================================
    print()
    print("=" * 70)
    print("阶段 2：lookup_prefix —— waiting 期的纯索引查询（零数据搬运）")
    print("=" * 70)
    print(f"  同前缀        → {engine.lookup_prefix(prompt_a)} token（全命中）")
    print(f"  共享6后分叉   → {engine.lookup_prefix(prompt_a[:6] + [98, 99])} token")
    print(f"  全新前缀      → {engine.lookup_prefix([77, 78, 79, 80])} token")

    # ================================================================
    # 阶段 3：命中请求 forward（逐层 load，L+1 预取与 L 计算重叠）
    # ================================================================
    print()
    print("=" * 70)
    print("阶段 3：命中 forward —— wait L → 预取 L+1 → compute L 的流水")
    print("=" * 70)

    computed = []

    def compute(layer):
        computed.append(layer)

    sim.hit_forward(keys_a, compute)
    for line in sim.timeline:
        print(f"    {line}")
    assert computed == list(range(NUM_LAYERS))

    ok = all(
        hooks.dst[(key, layer)] == bytes([layer * 16 + i] * SEGMENT_BYTES)
        for i, key in enumerate(keys_a)
        for layer in range(NUM_LAYERS)
    )
    print(f"  数据一致性：{'PASS（盘→staging→目的侧，逐段一致）' if ok else 'FAIL'}")

    # Optional replay loop for Nsight: keep the same request resident and
    # immediately issue it again so cache reuse and layer overlap are visible.
    replay_rounds = max(1, int(os.environ.get("TUTTI_NVTX_REPLAY_ROUNDS", "1")))
    for replay in range(1, replay_rounds):
        with nvtx_range(f"tutti.replay.request|id=A|round={replay}"):
            sim.hit_forward(keys_a, lambda _layer: None)

    # ================================================================
    # 阶段 4：容量淘汰（LRU 驱逐 + 断链截断）
    # ================================================================
    print()
    print("=" * 70)
    print("阶段 4：容量淘汰 —— 池满驱逐最旧驻留，被汰分片物理消失")
    print("=" * 70)

    prompt_b = [50, 51, 52, 53]
    prompt_c = [60, 61, 62, 63]
    keys_b, _ = engine.hash_keys(prompt_b)
    keys_c, _ = engine.hash_keys(prompt_c)
    for layer in range(NUM_LAYERS):
        hooks.src[(keys_b[0], layer)] = bytes([0xB0 + layer] * SEGMENT_BYTES)
        hooks.src[(keys_c[0], layer)] = bytes([0xC0 + layer] * SEGMENT_BYTES)

    engine.plan_store(keys_b)
    for layer in range(NUM_LAYERS):
        engine.store_layer(keys_b, layer, src_first_blocks=None).wait()
    engine.confirm_store(keys_b)

    plan_c = engine.plan_store(keys_c)     # 池满：驱逐最旧驻留 chunk
    evicted = plan_c.evicted_keys
    print(f"  写入 C 触发驱逐：{[k.hex()[:8] for k in evicted]}")
    for layer in range(NUM_LAYERS):
        engine.store_layer(keys_c, layer, src_first_blocks=None).wait()
    engine.confirm_store(keys_c)
    show("写入 C 后", engine, store)
    print(f"  被汰 chunk 分片残留：{sum(1 for k in store.scan() if k[:16] in evicted)}"
          "（应为 0——drop 按全层 io_key 展开）")
    print(f"  驱逐后 lookup_prefix(prompt_a) → {engine.lookup_prefix(prompt_a)}"
          " token（断链截断）")

    # ================================================================
    # 阶段 5：staging 槽位生命周期（组合完成句柄，B1 修复的语义演示）
    # ================================================================
    print()
    print("=" * 70)
    print("阶段 5：槽位复用保护 —— DMA 完成 ∧ scatter 读完，回绕波次才放行")
    print("=" * 70)

    # 独立小环境：1 层、半窗 1 槽（num_slots=2）——第 3 次加载必然
    # 回绕复用第 1 次的槽位（wave-2 acquire 等 wave-0 组合句柄）。
    store5 = ManualStore(SEGMENT_BYTES, 16, NUM_DEVICES, STRIPE_UNIT)
    eng5 = KVEngine(
        config={"chunk_tokens": CHUNK_TOKENS,
                "chunk_kv_bytes": SEGMENT_BYTES,
                "max_chunks_per_wave": 1},
        store=store5,
    )
    window5 = RingWindow(bytearray(2 * SEGMENT_BYTES), 2, SEGMENT_BYTES)
    scatter_events: list[FakeEvent] = []
    dst5: dict = {}

    def scatter5(keys, layer_idx, first_blocks, slots):
        ev = FakeEvent()          # "scatter 已入流，但尚未读完槽位"
        scatter_events.append(ev)
        for key, slot in zip(keys, slots):
            off = slot * SEGMENT_BYTES
            dst5[key] = bytes(window5.buffer[off:off + SEGMENT_BYTES])
        return ev                 # 返回事件 → 进入组合句柄

    def gather5(keys, layer_idx, first_blocks, slots):
        raise AssertionError("阶段 5 不走 save 路径")

    eng5.bind({}, window5, 1, 1, gather_fn=gather5, scatter_fn=scatter5)

    # 预置盘上数据（直接写 store，绕过 save 路径）
    demo_keys = [bytes([0xD0 + i] * 16) for i in range(3)]
    for key in demo_keys:
        store5._stripes[derive_io_key(key, 0)] = {
            k: (0, bytes([0xE0] * STRIPE_UNIT))
            for k in range(SEGMENT_BYTES // STRIPE_UNIT)
        }

    def step(msg):
        time.sleep(0.4)
        print(f"    {msg}")

    # wave-0 / wave-1：两次加载各占一个槽。组合句柄在提交时即登记
    # 进环窗（wait_for_layer_load 与回绕 acquire 等的是同一个句柄；
    # 契约要求单等待方——本演示由回绕线程充当唯一等待者）。
    eng5.load_layer(demo_keys[:1], 0, dst_first_blocks=None)
    h1 = eng5.load_layer(demo_keys[1:2], 0, dst_first_blocks=None)
    step("wave-0/1 加载已提交（槽 0/1 各占一），DMA 均在途")

    store5.inflight[0].complete()   # wave-0 的 DMA 完成
    step("wave-0 DMA 完成 —— 但组合句柄未完成（scatter 尚未读槽）")

    # wave-2 回绕：需要复用槽 0 → acquire 等 wave-0 组合句柄
    # （= DMA 完成 ∧ scatter 事件完成）。此时 scatter 尚未 record。
    t2_submitted = threading.Event()
    wave2_handles: list = []

    def submit_wave2():
        wave2_handles.append(
            eng5.load_layer(demo_keys[2:3], 0, dst_first_blocks=None))
        t2_submitted.set()

    tt = threading.Thread(target=submit_wave2, daemon=True)
    tt.start()

    # 等回绕线程跑进等待（scatter 入流、事件出现），再断言其被挡
    deadline = time.time() + 5
    while len(scatter_events) < 1 and time.time() < deadline:
        time.sleep(0.05)
    step(f"wave-2 回绕请求槽 0：被挡住（submitted={t2_submitted.is_set()}）；"
         "等待中 scatter 已入流但事件未 record")
    assert not t2_submitted.is_set(), "scatter 未读完前 wave-2 不得复用槽位"
    assert len(scatter_events) == 1, "等待过程中 scatter 应已提交入流"

    # scatter 读完槽位（CUDA event record）→ 组合句柄完成 → wave-2 放行
    scatter_events[0].record()
    tt.join(timeout=5)
    step("scatter 事件 record（消费方读完槽）→ 组合句柄完成、"
         f"wave-2 获得槽位（submitted={t2_submitted.is_set()}）")
    assert t2_submitted.is_set()

    # 收尾：完成全部在途 DMA；逐个 wait 组合句柄并放行其 scatter 事件
    # （事件未放行前 wait 会挂——这正是组合句柄语义，close 依赖它）
    for h in store5.inflight[1:]:
        h.complete()

    def drain(handle):
        t = threading.Thread(target=handle.wait, daemon=True)
        t.start()
        deadline = time.time() + 5
        while len(scatter_events) == 0 and time.time() < deadline:
            time.sleep(0.05)          # 等该句柄的 scatter 入流
        ev = scatter_events.pop()
        ev.record()                   # 消费方读完槽
        t.join(timeout=5)

    drain(h1)
    drain(wave2_handles[0])
    print(f"  结论：槽位复用等待 = DMA 完成 ∧ scatter 读完（组合句柄）；"
          "全程无全局 synchronize")
    eng5.close()

    print()
    print("收尾：全流程结束。")


if __name__ == "__main__":
    main()

"""engine 契约测试：多层往返、波次与覆盖保护、驱逐接线、bind 与生命周期。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from engine.core import KVEngine
from engine.staging import RingWindow
from index.chunk_index import chunk_key_of, derive_io_key, layer_of
from stores.memory import MemoryKVStore

SEG = 4096            # 单层段字节数
NUM_LAYERS = 3
CHUNK_TOKENS = 4
MAX_WAVE = 2          # 单波最大 chunk 数
NUM_SLOTS = 2 * MAX_WAVE
NUM_CHUNKS = 8


def _tokens(base: int, n_chunks: int) -> list[int]:
    """生成 base 序列前 n_chunks 个 chunk 的 token 列表。"""
    return [base * 1000 + i for i in range(n_chunks * CHUNK_TOKENS)]


def _chunk_keys(engine: KVEngine, base: int, n_chunks: int) -> list[bytes]:
    keys, _ = engine.hash_keys(_tokens(base, n_chunks))
    return keys


def _segment_bytes(chunk_i: int, layer: int) -> bytes:
    """每个 (chunk, layer) 的层段内容：首字节作标记，其余填充。"""
    marker = (chunk_i * 37 + layer * 11) & 0xFF
    return bytes([marker]) + bytes([marker ^ 0xFF]) * (SEG - 1)


class MovingHooks:
    """真实搬运钩子：paged 侧以 dict 模拟，数据经 staging 缓冲中转。"""

    def __init__(self, buffer=None, seg: int = SEG):
        self._view = memoryview(buffer) if buffer is not None else None
        self._seg = seg
        self.source: dict[tuple[bytes, int], bytes] = {}
        self.sink: dict[tuple[bytes, int], bytes] = {}
        self.log: list[tuple] = []

    def gather(self, keys, layer_idx, first_blocks, slots):
        """store 方向：把源侧层段写入 staging 槽。"""
        self.log.append(("gather", tuple(keys), layer_idx, tuple(slots), first_blocks))
        for k, s in zip(keys, slots):
            self._view[s * self._seg:(s + 1) * self._seg] = self.source[(k, layer_idx)]

    def scatter(self, keys, layer_idx, first_blocks, slots):
        """load 方向：把 staging 槽内容读出到目的侧。"""
        self.log.append(("scatter", tuple(keys), layer_idx, tuple(slots), first_blocks))
        for k, s in zip(keys, slots):
            self.sink[(k, layer_idx)] = bytes(
                self._view[s * self._seg:(s + 1) * self._seg]
            )


class TailEvent:
    """模拟异步消费完成事件。"""

    def __init__(self):
        self.synchronize_count = 0
        self._done = False

    def synchronize(self):
        self.synchronize_count += 1
        self._done = True

    def query(self):
        return self._done


class EventHooks(MovingHooks):
    """搬运钩子：目的侧返回一个消费完成事件。"""

    def __init__(self, buffer=None, seg: int = SEG):
        super().__init__(buffer, seg)
        self.events: list[TailEvent] = []

    def scatter(self, keys, layer_idx, first_blocks, slots):
        super().scatter(keys, layer_idx, first_blocks, slots)
        event = TailEvent()
        self.events.append(event)
        return event


class StoreSpy:
    """store 包装：记录 drop 与 register_buffer 调用。"""

    def __init__(self, inner):
        self._inner = inner
        self.dropped: list[list[bytes]] = []
        self.register_calls: list[int] = []

    @property
    def capacity_chunks(self) -> int:
        return self._inner.capacity_chunks

    def open(self):
        self._inner.open()

    def close(self):
        self._inner.close()

    def register_buffer(self, buffer, granularity):
        self.register_calls.append(granularity)
        return self._inner.register_buffer(buffer, granularity)

    def put_batch(self, batch):
        return self._inner.put_batch(batch)

    def get_batch(self, batch):
        return self._inner.get_batch(batch)

    def drop(self, keys):
        self.dropped.append(list(keys))
        self._inner.drop(keys)

    def scan(self):
        return self._inner.scan()


class FakeEvent:
    """可数完成事件。"""

    def __init__(self):
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1

    def query(self):
        return True


def _make_engine(store, hooks=None, num_chunks=NUM_CHUNKS, max_wave=MAX_WAVE):
    """构造引擎 + 假显存窗口 + 绑定；返回 (engine, window, hooks)。"""
    config = {
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_kv_bytes": SEG * NUM_LAYERS,
        "max_chunks_per_wave": max_wave,
    }
    if hooks is not None:
        config["gather_fn"] = hooks.gather
        config["scatter_fn"] = hooks.scatter
    engine = KVEngine(config, store)
    buffer = bytearray(NUM_SLOTS * SEG)
    if hooks is not None:
        hooks._view = memoryview(buffer)
    window = RingWindow(buffer, NUM_SLOTS, SEG)
    engine.bind({}, window, NUM_LAYERS, blocks_per_chunk=2)
    return engine, window, hooks


class TestIoKeyHelpers:
    def test_roundtrip(self):
        ck = bytes(range(16))
        io_key = derive_io_key(ck, 7)
        assert len(io_key) == 18
        assert chunk_key_of(io_key) == ck
        assert layer_of(io_key) == 7

    def test_layer_encoding_is_little_endian(self):
        ck = b"\x11" * 16
        io_key = derive_io_key(ck, 0x0102)
        assert io_key[16:] == b"\x02\x01"

    def test_bad_arguments_raise(self):
        with pytest.raises(ValueError):
            derive_io_key(b"\x00" * 15, 0)
        with pytest.raises(ValueError):
            derive_io_key(b"\x00" * 16, 1 << 16)
        with pytest.raises(ValueError):
            chunk_key_of(b"\x00" * 17)
        with pytest.raises(ValueError):
            layer_of(b"\x00" * 18 + b"\x00")


class TestRoundtrip:
    def test_multilayer_store_load_no_cross_talk(self):
        store = MemoryKVStore(SEG, NUM_CHUNKS)
        hooks = MovingHooks(None, SEG)
        engine, _, _ = _make_engine(store, hooks)

        keys = _chunk_keys(engine, 1, 2)
        plan = engine.plan_store(keys)
        assert plan.new_keys == keys
        engine.confirm_store(keys)

        # 逐层写入：source 侧备好每层数据
        for layer in range(NUM_LAYERS):
            for i, k in enumerate(keys):
                hooks.source[(k, layer)] = _segment_bytes(i + 1, layer)
        for layer in range(NUM_LAYERS):
            engine.store_layer(keys, layer, src_first_blocks=10 + layer).wait()

        # 全新目的侧逐层读回
        for layer in range(NUM_LAYERS):
            engine.load_layer(keys, layer, dst_first_blocks=50 + layer).wait()
        for layer in range(NUM_LAYERS):
            for i, k in enumerate(keys):
                assert hooks.sink[(k, layer)] == _segment_bytes(i + 1, layer)

    def test_load_completion_runs_scatter_on_wait(self):
        store = MemoryKVStore(SEG, NUM_CHUNKS)
        hooks = MovingHooks(None, SEG)
        engine, _, _ = _make_engine(store, hooks)
        keys = _chunk_keys(engine, 1, 1)
        engine.plan_store(keys)
        engine.confirm_store(keys)
        hooks.source[(keys[0], 0)] = _segment_bytes(1, 0)
        engine.store_layer(keys, 0, 0).wait()
        assert engine.load_layer(keys, 0, 0).query() is True
        # scatter 只在 wait 后执行
        assert (keys[0], 0) not in hooks.sink
        engine.wait_idle()
        assert hooks.sink[(keys[0], 0)] == _segment_bytes(1, 0)

    def test_wave_reuse_waits_for_scatter_event(self):
        store = MemoryKVStore(SEG, NUM_CHUNKS)
        hooks = EventHooks(None, SEG)
        engine, _, _ = _make_engine(store, hooks)
        keys = _chunk_keys(engine, 1, 3)
        engine.plan_store(keys)
        engine.confirm_store(keys)
        for i, key in enumerate(keys):
            hooks.source[(key, 0)] = _segment_bytes(i + 1, 0)
        for key in keys:
            engine.store_layer([key], 0, 0).wait()

        first = engine.load_layer([keys[0]], 0, 0)
        engine.load_layer([keys[1]], 0, 0)
        # 第三波复用第一波槽位；acquire 必须等待底层搬运与 scatter 事件。
        engine.load_layer([keys[2]], 0, 0)
        assert len(hooks.events) == 1
        assert hooks.events[0].synchronize_count == 1
        assert first.query() is True

    def test_load_unknown_key_raises(self):
        store = MemoryKVStore(SEG, NUM_CHUNKS)
        engine, _, _ = _make_engine(store)
        keys = _chunk_keys(engine, 1, 1)
        with pytest.raises(ValueError):
            engine.load_layer(keys, 0, 0)


class TestWindowWaves:
    def test_wave_and_slot_rotation_at_engine_level(self):
        store = MemoryKVStore(SEG, NUM_CHUNKS)
        hooks = MovingHooks(None, SEG)
        engine, _, _ = _make_engine(store, hooks)
        keys = _chunk_keys(engine, 1, 6)
        engine.plan_store(keys)
        engine.confirm_store(keys)
        for layer in range(NUM_LAYERS):
            for i, k in enumerate(keys):
                hooks.source[(k, layer)] = _segment_bytes(i + 1, layer)
        gather_slots = []
        for start in range(0, 6, MAX_WAVE):
            batch = keys[start:start + MAX_WAVE]
            engine.store_layer(batch, 0, 0)
            gather_slots.append(hooks.log[-1][3])
        # 半窗交替：第 0 波前半窗、第 1 波后半窗、第 2 波复用前半窗
        assert gather_slots == [
            tuple(range(0, MAX_WAVE)),
            tuple(range(MAX_WAVE, 2 * MAX_WAVE)),
            tuple(range(0, MAX_WAVE)),
        ]

    def test_ring_window_wave_sequence_and_bounds(self):
        window = RingWindow(bytearray(NUM_SLOTS * SEG), NUM_SLOTS, SEG)
        wave0, slots0 = window.acquire(MAX_WAVE)
        wave1, slots1 = window.acquire(1)
        assert (wave0, wave1) == (0, 1)
        assert slots0 == [0, 1] and slots1 == [2]
        with pytest.raises(ValueError):
            window.acquire(0)
        with pytest.raises(ValueError):
            window.acquire(MAX_WAVE + 1)
        window.complete(wave0, FakeEvent())
        window.complete(wave1, FakeEvent())

    def test_overwrite_protection_waits_two_waves_back(self):
        window = RingWindow(bytearray(NUM_SLOTS * SEG), NUM_SLOTS, SEG)
        e0, e1, e2 = FakeEvent(), FakeEvent(), FakeEvent()
        wave0, _ = window.acquire(1)
        wave1, _ = window.acquire(1)
        window.complete(wave0, e0)
        window.complete(wave1, e1)
        wave2, _ = window.acquire(1)  # 与 wave0 同半窗
        window.complete(wave2, e2)
        assert (e0.wait_count, e1.wait_count, e2.wait_count) == (1, 0, 0)

    def test_unregistered_predecessor_does_not_block(self):
        window = RingWindow(bytearray(NUM_SLOTS * SEG), NUM_SLOTS, SEG)
        window.acquire(1)
        window.acquire(1)  # 均未登记完成事件
        wave, _ = window.acquire(1)  # 复用 wave0 的半窗，不阻塞
        assert wave == 2

    def test_constructor_validations(self):
        with pytest.raises(ValueError):
            RingWindow(bytearray(8 * SEG), 3, SEG)          # 奇数槽
        with pytest.raises(ValueError):
            RingWindow(bytearray(4 * SEG - 1), 4, SEG)      # 缓冲不足
        with pytest.raises(ValueError):
            RingWindow(bytearray(4 * SEG), 4, 0)            # 非法段宽


class TestEvictionWiring:
    def test_plan_store_drops_all_layers_of_evicted(self):
        spy = StoreSpy(MemoryKVStore(SEG, num_chunks=2))
        engine, _, _ = _make_engine(spy, MovingHooks(None, SEG), num_chunks=2)
        old_keys = _chunk_keys(engine, 1, 2)
        engine.plan_store(old_keys)
        engine.confirm_store(old_keys)

        new_keys = _chunk_keys(engine, 2, 1)
        plan = engine.plan_store(new_keys)
        assert plan is not None
        assert plan.evicted_keys == old_keys[:1]
        # 驱逐 chunk 的全部层 io_key 恰好一次 drop
        expected = {derive_io_key(old_keys[0], l) for l in range(NUM_LAYERS)}
        assert len(spy.dropped) == 1
        assert set(spy.dropped[0]) == expected
        assert len(spy.dropped[0]) == NUM_LAYERS
        engine.confirm_store(new_keys)

    def test_no_drop_before_bind(self):
        spy = StoreSpy(MemoryKVStore(SEG, 2))
        config = {
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_kv_bytes": SEG * NUM_LAYERS,
            "max_chunks_per_wave": MAX_WAVE,
        }
        engine = KVEngine(config, spy)  # 未 bind：无层宽，不做数据面删除
        keys = _chunk_keys(engine, 1, 2)
        engine.plan_store(keys)
        engine.confirm_store(keys)
        plan = engine.plan_store(_chunk_keys(engine, 2, 1))
        assert plan is not None
        assert spy.dropped == []


class TestBind:
    def test_registers_staging_buffer_exactly_once(self):
        spy = StoreSpy(MemoryKVStore(SEG, NUM_CHUNKS))
        _make_engine(spy)
        assert spy.register_calls == [SEG]

    def test_double_bind_rejected(self):
        engine, window, _ = _make_engine(MemoryKVStore(SEG, NUM_CHUNKS))
        with pytest.raises(RuntimeError):
            engine.bind({}, window, NUM_LAYERS, 2)

    def test_geometry_mismatch_rejected(self):
        store = MemoryKVStore(SEG, NUM_CHUNKS)
        config = {
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_kv_bytes": SEG * NUM_LAYERS,
            "max_chunks_per_wave": MAX_WAVE,
        }
        engine = KVEngine(config, store)
        bad_window = RingWindow(bytearray(NUM_SLOTS * (SEG * 2)), NUM_SLOTS, SEG * 2)
        with pytest.raises(ValueError):
            engine.bind({}, bad_window, NUM_LAYERS, 2)
        with pytest.raises(ValueError):
            engine.bind({}, RingWindow(bytearray(2 * SEG), 2, SEG), 7, 2)  # 不能整分
        small = RingWindow(bytearray(2 * SEG), 2, SEG)  # 单波容量 1 < max_wave 2
        with pytest.raises(ValueError):
            engine.bind({}, small, NUM_LAYERS, 2)

    def test_layer_calls_before_bind_rejected(self):
        store = MemoryKVStore(SEG, NUM_CHUNKS)
        config = {
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_kv_bytes": SEG * NUM_LAYERS,
            "max_chunks_per_wave": MAX_WAVE,
        }
        engine = KVEngine(config, store)
        keys = _chunk_keys(engine, 1, 1)
        with pytest.raises(RuntimeError):
            engine.store_layer(keys, 0, 0)
        with pytest.raises(RuntimeError):
            engine.load_layer(keys, 0, 0)

    def test_bad_config_rejected(self):
        store = MemoryKVStore(SEG, NUM_CHUNKS)
        base = {
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_kv_bytes": SEG * NUM_LAYERS,
            "max_chunks_per_wave": MAX_WAVE,
        }
        for key in base:
            bad = dict(base, **{key: 0})
            with pytest.raises(ValueError):
                KVEngine(bad, store)
        with pytest.raises(ValueError):
            KVEngine(dict(base, gather_fn=1), store)

    def test_layer_call_validation(self):
        engine, _, _ = _make_engine(MemoryKVStore(SEG, NUM_CHUNKS))
        keys = _chunk_keys(engine, 1, MAX_WAVE + 1)
        with pytest.raises(ValueError):
            engine.store_layer(keys, 0, 0)      # 超单波容量
        with pytest.raises(ValueError):
            engine.store_layer([], 0, 0)        # 空批
        with pytest.raises(ValueError):
            engine.store_layer(keys[:1], NUM_LAYERS, 0)  # 层号越界
        with pytest.raises(ValueError):
            engine.store_layer(keys[:1], -1, 0)


class TestLifecycle:
    def test_close_is_idempotent_and_guards_state(self):
        store = MemoryKVStore(SEG, NUM_CHUNKS)
        engine, _, _ = _make_engine(store)
        engine.close()
        engine.close()
        with pytest.raises(RuntimeError):
            engine.lookup_prefix(_tokens(1, 1))
        with pytest.raises(RuntimeError):
            engine.plan_store([])

    def test_wait_idle_drains_inflight(self):
        engine, _, _ = _make_engine(MemoryKVStore(SEG, NUM_CHUNKS))
        keys = _chunk_keys(engine, 1, 1)
        engine.plan_store(keys)
        engine.confirm_store(keys)
        handle = engine.store_layer(keys, 0, 0)
        engine.wait_idle()
        assert handle.query() is True
        assert engine.store_layer(keys, 1, 0).query() is True


class TestColdStartRestore:
    def test_incomplete_chunk_excluded_from_restore(self):
        store = MemoryKVStore(SEG, NUM_CHUNKS)
        hooks = MovingHooks(None, SEG)
        engine_a, _, _ = _make_engine(store, hooks)

        full_keys = _chunk_keys(engine_a, 1, 2)
        engine_a.plan_store(full_keys)
        engine_a.confirm_store(full_keys)
        for layer in range(NUM_LAYERS):
            for i, k in enumerate(full_keys):
                hooks.source[(k, layer)] = _segment_bytes(i + 1, layer)
            engine_a.store_layer(full_keys, layer, 0).wait()
        engine_a.wait_idle()  # 不 close：保留 store 内容模拟持久状态

        # 手工注入一个残缺 chunk（仅层 0 与层 2，缺层 1）
        partial_keys = _chunk_keys(engine_a, 2, 1)
        scratch = bytearray(SEG)
        buf_id = store.register_buffer(scratch, SEG)
        scratch[0] = 0xAB
        store.put_batch(
            [(derive_io_key(partial_keys[0], l), buf_id, 0) for l in (0, 2)]
        )

        engine_b, _, _ = _make_engine(store)
        # 完整 chunk 命中，残缺 chunk 视为缺失
        assert engine_b.lookup_prefix(_tokens(1, 2)) == 2 * CHUNK_TOKENS
        assert engine_b.lookup_prefix(_tokens(2, 1)) == 0
        engine_b.pin(full_keys)
        engine_b.unpin(full_keys)

    def test_restore_counts_toward_capacity(self):
        store = MemoryKVStore(SEG, num_chunks=2)
        hooks = MovingHooks(None, SEG)
        engine_a, _, _ = _make_engine(store, hooks, num_chunks=2)
        keys = _chunk_keys(engine_a, 1, 2)
        engine_a.plan_store(keys)
        engine_a.confirm_store(keys)
        for layer in range(NUM_LAYERS):
            for i, k in enumerate(keys):
                hooks.source[(k, layer)] = _segment_bytes(i + 1, layer)
            engine_a.store_layer(keys, layer, 0).wait()
        engine_a.wait_idle()

        engine_b, _, _ = _make_engine(store, MovingHooks(None, SEG), num_chunks=2)
        # 恢复占用全部容量：第三个 chunk 驱逐最旧者
        plan = engine_b.plan_store(_chunk_keys(engine_b, 2, 1))
        assert plan is not None
        assert plan.evicted_keys == keys[:1]


class TestIsolation:
    def test_import_pulls_no_heavy_dependencies(self):
        """子进程断言：import engine.core 后 sys.modules 无 vllm/torch/tutti_runtime。"""
        connector_root = Path(__file__).resolve().parents[2]
        code = (
            "import sys\n"
            "import engine.core\n"
            "leaked = {'vllm', 'torch', 'numpy', 'tutti_runtime'} & set(sys.modules)\n"
            "assert not leaked, f'unexpected modules: {sorted(leaked)}'\n"
            "print('clean')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=connector_root,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "clean" in proc.stdout

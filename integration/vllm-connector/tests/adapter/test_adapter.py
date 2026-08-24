"""适配层契约测试：真 vLLM 基类对接、双角色共享引擎、端到端读写往返。"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from adapter.connector import TuttiConnectorMetadata, TuttiConnectorV1
from engine.core import KVEngine
from index.chunk_index import derive_io_key
from stores.memory import MemoryKVStore
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)

SEG = 4096
NUM_LAYERS = 3
CHUNK_TOKENS = 8
MAX_WAVE = 4
BLOCK_SIZE = 16


def _segment(chunk_i: int, layer: int) -> bytes:
    """每个 (chunk, layer) 的层段内容：首字节标记。"""
    marker = (chunk_i * 29 + layer * 13) & 0xFF
    return bytes([marker]) + bytes([marker ^ 0x5A]) * (SEG - 1)


class PoolHooks:
    """两端搬运钩子：paged 侧以 dict 模拟，数据经 staging 缓冲中转。"""

    def __init__(self):
        self.source: dict[tuple[bytes, int], bytes] = {}
        self.sink: dict[tuple[bytes, int], bytes] = {}
        self.view = None
        self.gather_calls: list[tuple] = []
        self.scatter_calls: list[tuple] = []

    def bind_window(self, window) -> None:
        """把 staging 视图接到环窗显存上。"""
        self.view = memoryview(window.buffer.numpy()) if hasattr(window.buffer, "numpy") \
            else memoryview(window.buffer)

    def gather(self, keys, layer_idx, first_blocks, slots):
        self.gather_calls.append((tuple(keys), layer_idx, tuple(slots)))
        for k, s in zip(keys, slots):
            self.view[s * SEG:(s + 1) * SEG] = self.source[(k, layer_idx)]

    def scatter(self, keys, layer_idx, first_blocks, slots):
        self.scatter_calls.append((tuple(keys), layer_idx, tuple(slots)))
        for k, s in zip(keys, slots):
            self.sink[(k, layer_idx)] = bytes(self.view[s * SEG:(s + 1) * SEG])


@dataclass
class _Harness:
    store: MemoryKVStore
    engine: KVEngine
    hooks: PoolHooks
    vllm_config: SimpleNamespace
    scheduler: TuttiConnectorV1
    worker: TuttiConnectorV1
    layer_names: list[str] = field(default_factory=list)


def _make_vllm_config(extra: dict | None = None, block_size: int = BLOCK_SIZE):
    merged = {
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_kv_bytes": SEG * NUM_LAYERS,
        "max_chunks_per_wave": MAX_WAVE,
        "store": {"type": "memory", "options": {"segment_bytes": SEG, "num_chunks": 16}},
    }
    merged.update(extra or {})
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=merged),
        cache_config=SimpleNamespace(block_size=block_size),
    )


class _TensorAdaptingStore:
    """MemoryKVStore 的张量适配层。

    真机 store 走 data_ptr 协议直收张量；MemoryKVStore 走缓冲协议
    只收 numpy 视图。本层在注册时把 CPU 张量转成共享存储的 numpy
    视图，令测试替身与真机的注册接受面对齐。
    """

    def __init__(self, inner: MemoryKVStore):
        self._inner = inner

    @property
    def capacity_chunks(self) -> int:
        return self._inner.capacity_chunks

    def open(self) -> None:
        self._inner.open()

    def close(self) -> None:
        self._inner.close()

    def register_buffer(self, buffer, granularity: int):
        if torch.is_tensor(buffer) and not buffer.is_cuda:
            buffer = buffer.numpy()
        return self._inner.register_buffer(buffer, granularity)

    def put_batch(self, batch):
        return self._inner.put_batch(batch)

    def get_batch(self, batch):
        return self._inner.get_batch(batch)

    def drop(self, keys) -> None:
        self._inner.drop(keys)

    def scan(self):
        return self._inner.scan()


def _make_harness(extra: dict | None = None) -> _Harness:
    """手工构建 engine（注入搬运钩子），双角色挂载点共享同一实例。"""
    hooks = PoolHooks()
    store = _TensorAdaptingStore(MemoryKVStore(SEG, 16))
    engine = KVEngine(
        {
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_kv_bytes": SEG * NUM_LAYERS,
            "max_chunks_per_wave": MAX_WAVE,
            "gather_fn": hooks.gather,
            "scatter_fn": hooks.scatter,
        },
        store,
    )
    cfg = _make_vllm_config({**(extra or {}), "tutti_engine_instance": engine})
    kvcc = object()
    scheduler = TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, kvcc)
    worker = TuttiConnectorV1(cfg, KVConnectorRole.WORKER, kvcc)
    layer_names = [f"model.layers.{i}.self_attn.attn" for i in range(NUM_LAYERS)]
    kv_caches = {
        name: torch.zeros(8, BLOCK_SIZE, 2, 8, dtype=torch.float16)
        for name in layer_names
    }
    worker.register_kv_caches(kv_caches)
    return _Harness(
        store=store, engine=engine, hooks=hooks, vllm_config=cfg,
        scheduler=scheduler, worker=worker, layer_names=layer_names,
    )


def _fake_request(req_id: str, tokens: list[int]):
    return SimpleNamespace(
        request_id=req_id,
        prompt_token_ids=list(tokens),
        output_token_ids=[],
        all_token_ids=list(tokens),  # 活请求序列：decode 步持续追加
    )


def _sched_output(new_reqs=(), cached=None, finished=(), scheduled_tokens=None):
    return SimpleNamespace(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=cached,
        finished_req_ids=set(finished),
        num_scheduled_tokens=dict(scheduled_tokens or {}),
    )


class TestMounting:
    def test_isinstance_of_real_vllm_base(self):
        cfg = _make_vllm_config()
        conn = TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())
        assert isinstance(conn, KVConnectorBase_V1)

    def test_three_positional_args(self):
        cfg = _make_vllm_config()
        TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())

    def test_both_roles_share_engine(self):
        h = _make_harness()
        assert h.scheduler._engine is h.engine
        assert h.worker._engine is h.engine
        # 同 vllm_config 无注入时也共享（进程内注册表）
        cfg = _make_vllm_config()
        a = TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())._engine
        b = TuttiConnectorV1(cfg, KVConnectorRole.WORKER, object())._engine
        assert a is b

    def test_piecewise_overridden_true(self):
        assert TuttiConnectorV1.requires_piecewise_for_cudagraph({}) is True

    def test_prefer_cross_layer_blocks_true(self):
        h = _make_harness()
        assert h.worker.prefer_cross_layer_blocks is True

    def test_missing_engine_key_rejected(self):
        cfg = _make_vllm_config()
        del cfg.kv_transfer_config.kv_connector_extra_config["chunk_kv_bytes"]
        with pytest.raises(ValueError):
            TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())


class TestSchedulerSide:
    def test_matched_tokens_empty_store(self):
        h = _make_harness()
        req = _fake_request("r1", list(range(3 * CHUNK_TOKENS)))
        assert h.scheduler.get_num_new_matched_tokens(req, 0) == (0, False)

    def test_min_retrieve_threshold(self):
        h = _make_harness(extra={"min_retrieve_tokens": 2 * CHUNK_TOKENS})
        # 预置一个 chunk 驻留（写入并结算）
        keys, _ = h.engine.hash_keys(list(range(3 * CHUNK_TOKENS)))
        h.engine.plan_store(keys[:1])
        h.engine.confirm_store(keys[:1])
        req = _fake_request("r1", list(range(3 * CHUNK_TOKENS)))
        # 命中 1 chunk（8 tokens）< 阈值 16 → 0
        assert h.scheduler.get_num_new_matched_tokens(req, 0) == (0, False)
        assert h.scheduler.get_num_new_matched_tokens(req, 0)[0] == 0  # 无副作用

    def test_max_tokens_per_load_caps(self):
        h = _make_harness(extra={"max_tokens_per_load": CHUNK_TOKENS})
        keys, _ = h.engine.hash_keys(list(range(3 * CHUNK_TOKENS)))
        h.engine.plan_store(keys)
        h.engine.confirm_store(keys)
        req = _fake_request("r1", list(range(3 * CHUNK_TOKENS)))
        assert h.scheduler.get_num_new_matched_tokens(req, 0) == (CHUNK_TOKENS, False)

    def test_build_meta_records_save_plan(self):
        h = _make_harness()
        req = _fake_request("r1", list(range(2 * CHUNK_TOKENS + 3)))
        h.scheduler.get_num_new_matched_tokens(req, 0)
        h.scheduler.update_state_after_alloc(req, object(), 0)
        so = _sched_output(new_reqs=[SimpleNamespace(
            req_id="r1", prompt_token_ids=req.prompt_token_ids, block_ids=[0, 1]
        )])
        meta = h.scheduler.build_connector_meta(so)
        assert isinstance(meta, TuttiConnectorMetadata)
        assert len(meta.requests) == 1
        entry = meta.requests[0]
        # 尾部 3 token 舍弃：恰好 2 chunk
        assert entry.save_chunk_start == 0
        assert entry.save_chunk_count == 2
        assert entry.load_tokens == 0

    def test_partial_chunk_not_saved(self):
        h = _make_harness()
        req = _fake_request("r1", list(range(CHUNK_TOKENS - 1)))  # 不足一个 chunk
        h.scheduler.update_state_after_alloc(req, object(), 0)
        so = _sched_output(new_reqs=[SimpleNamespace(
            req_id="r1", prompt_token_ids=req.prompt_token_ids, block_ids=[0]
        )])
        meta = h.scheduler.build_connector_meta(so)
        assert meta.requests[0].save_chunk_count == 0

    def test_request_finished_cleans_up(self):
        h = _make_harness()
        req = _fake_request("r1", list(range(CHUNK_TOKENS)))
        h.scheduler.update_state_after_alloc(req, object(), CHUNK_TOKENS)
        assert h.scheduler.request_finished(req, [0]) == (False, None)
        so = _sched_output(new_reqs=[SimpleNamespace(
            req_id="r1", prompt_token_ids=req.prompt_token_ids, block_ids=[0]
        )])
        meta = h.scheduler.build_connector_meta(so)
        assert meta.requests[0].load_tokens == 0


class TestCachedStepForkSemantics:
    """fork 的 CachedRequestData 消费语义（PP 关闭时 new_token_ids 恒空）。

    回归背景：真机 IndexError——cached 步的 token 增量须从活请求对象
    切片（num_scheduled_tokens 对齐），块取 new_block_ids[i]（可 None）。
    """

    def test_decode_step_advances_tracker(self):
        h = _make_harness()
        prompt = list(range(3 * CHUNK_TOKENS))
        req = _fake_request("r1", prompt)
        h.scheduler.update_state_after_alloc(req, object(), 0)
        so = _sched_output(new_reqs=[SimpleNamespace(
            req_id="r1", prompt_token_ids=prompt, block_ids=[0, 1, 2]
        )])
        meta = h.scheduler.build_connector_meta(so)
        assert meta.requests[0].save_chunk_count == 3

        # ---- decode 步：fork 形态（new_token_ids 空、new_block_ids 嵌套）----
        req.all_token_ids.extend(range(1000, 1000 + CHUNK_TOKENS))  # 活序列增长
        cached = SimpleNamespace(
            req_ids=["r1"],
            new_token_ids=[],                       # PP 关闭：恒空
            new_block_ids=[([3],)],                 # tuple-of-lists，可 None
            all_token_ids={},
        )
        so2 = _sched_output(
            cached=cached, scheduled_tokens={"r1": CHUNK_TOKENS},
        )
        meta2 = h.scheduler.build_connector_meta(so2)
        assert len(meta2.requests) == 1
        # 记账推进：token 序列含 decode 增量、块表含新块
        assert meta2.requests[0].token_ids == prompt + list(
            range(1000, 1000 + CHUNK_TOKENS)
        )
        assert meta2.requests[0].block_ids == [0, 1, 2, 3]
        # 第 4 个 chunk 的保存计划在 decode 步成整后浮现
        assert meta2.requests[0].save_chunk_start == 3
        assert meta2.requests[0].save_chunk_count == 1

    def test_none_blocks_and_unscheduled_cached_req_tolerated(self):
        h = _make_harness()
        prompt = list(range(CHUNK_TOKENS))
        req = _fake_request("r1", prompt)
        h.scheduler.update_state_after_alloc(req, object(), 0)
        so = _sched_output(new_reqs=[SimpleNamespace(
            req_id="r1", prompt_token_ids=prompt, block_ids=[0]
        )])
        h.scheduler.build_connector_meta(so)

        cached = SimpleNamespace(
            req_ids=["r1", "ghost"],   # ghost：无 tracker（防御）
            new_token_ids=[],
            new_block_ids=[None, None],
            all_token_ids={},
        )
        meta = h.scheduler.build_connector_meta(_sched_output(
            cached=cached, scheduled_tokens={"r1": 1},
        ))
        assert [m.req_id for m in meta.requests] == ["r1"]
        assert meta.requests[0].block_ids == [0]

    def test_finished_req_state_released(self):
        h = _make_harness()
        prompt = list(range(CHUNK_TOKENS))
        req = _fake_request("r1", prompt)
        h.scheduler.update_state_after_alloc(req, object(), 0)
        h.scheduler.build_connector_meta(_sched_output(new_reqs=[
            SimpleNamespace(req_id="r1", prompt_token_ids=prompt, block_ids=[0])
        ]))
        assert "r1" in h.scheduler._trackers
        h.scheduler.build_connector_meta(_sched_output(finished=["r1"]))
        assert "r1" not in h.scheduler._trackers
        assert "r1" not in h.scheduler._live_requests


class TestEndToEnd:
    def test_store_then_load_roundtrip(self):
        h = _make_harness()
        # ---- 步 1：保存请求（16 tokens = 2 chunk）----
        prompt = list(range(2 * CHUNK_TOKENS))
        req = _fake_request("r1", prompt)
        h.scheduler.update_state_after_alloc(req, object(), 0)
        so = _sched_output(new_reqs=[SimpleNamespace(
            req_id="r1", prompt_token_ids=prompt, block_ids=[0, 1]
        )])
        meta = h.scheduler.build_connector_meta(so)
        assert meta.requests[0].save_chunk_count == 2

        h.worker.bind_connector_metadata(meta)
        h.worker.start_load_kv(None)  # 触发惰性绑定（本步无读取）
        h.hooks.bind_window(h.worker._impl.window)
        chunk_keys, _ = h.engine.hash_keys(prompt)
        for layer in range(NUM_LAYERS):
            for i in range(2):
                h.hooks.source[(chunk_keys[i], layer)] = _segment(i, layer)
        for name in h.layer_names:
            h.worker.save_kv_layer(name)
        h.worker.wait_for_save()
        # store 内 2 chunk × 全部层的 io_key
        assert set(h.store.scan()) == {
            derive_io_key(chunk_keys[i], layer)
            for i in range(2) for layer in range(NUM_LAYERS)
        }

        # ---- 步 2：新请求同前缀，命中并加载 ----
        req2 = _fake_request("r2", prompt)
        matched, async_load = h.scheduler.get_num_new_matched_tokens(req2, 0)
        assert (matched, async_load) == (2 * CHUNK_TOKENS, False)
        h.scheduler.update_state_after_alloc(req2, object(), matched)
        so2 = _sched_output(new_reqs=[SimpleNamespace(
            req_id="r2", prompt_token_ids=prompt, block_ids=[4, 5]
        )])
        meta2 = h.scheduler.build_connector_meta(so2)
        assert meta2.requests[0].load_tokens == 2 * CHUNK_TOKENS
        assert meta2.requests[0].save_chunk_count == 0  # 已驻留不重写

        h.worker.bind_connector_metadata(meta2)
        h.worker.start_load_kv(None)
        for name in h.layer_names:
            h.worker.wait_for_layer_load(name)
        for i in range(2):
            for layer in range(NUM_LAYERS):
                assert h.hooks.sink[(chunk_keys[i], layer)] == _segment(i, layer)
        # 每层恰好一次 scatter 调用（一层 × N chunk 一个批）
        assert len(h.hooks.scatter_calls) == NUM_LAYERS

    def test_stale_view_reports_load_errors(self):
        """近似视图命中而权威视图未遂：worker 上报重算块。"""
        h = _make_harness()
        prompt = list(range(CHUNK_TOKENS))
        # 调度侧视图命中（数据在），worker 侧另挂一个空引擎（数据不在）
        keys, _ = h.engine.hash_keys(prompt)
        h.engine.plan_store(keys)
        h.engine.confirm_store(keys)
        matched, _ = h.scheduler.get_num_new_matched_tokens(
            _fake_request("r1", prompt), 0
        )
        assert matched == CHUNK_TOKENS

        stale_store = _TensorAdaptingStore(MemoryKVStore(SEG, 16))
        stale_engine = KVEngine(
            {
                "chunk_tokens": CHUNK_TOKENS,
                "chunk_kv_bytes": SEG * NUM_LAYERS,
                "max_chunks_per_wave": MAX_WAVE,
            },
            stale_store,
        )
        cfg2 = _make_vllm_config({"tutti_engine_instance": stale_engine})
        worker2 = TuttiConnectorV1(cfg2, KVConnectorRole.WORKER, object())
        worker2.register_kv_caches({
            name: torch.zeros(8, BLOCK_SIZE, 2, 8, dtype=torch.float16)
            for name in h.layer_names
        })

        from adapter.connector import _ReqMeta
        meta = TuttiConnectorMetadata(requests=[
            _ReqMeta(
                req_id="r1", token_ids=prompt, block_ids=[0],
                load_tokens=CHUNK_TOKENS,
            )
        ])
        worker2.bind_connector_metadata(meta)
        worker2.start_load_kv(None)
        errors = worker2.get_block_ids_with_load_errors()
        assert errors == {0}
        assert worker2.get_block_ids_with_load_errors() == set()  # 读取即清

    def test_cross_layer_pool_binding(self):
        """单块跨层显存对象的登记与绑定路径（[块数, 层数, 块, K/V, 通道]）。"""
        h = _make_harness()
        pool = torch.zeros(8, NUM_LAYERS, BLOCK_SIZE, 2, 8, dtype=torch.float16)
        h.worker.register_cross_layers_kv_cache(pool, attn_backend=None)
        h.worker._impl._ensure_bound()
        assert h.worker._impl.window is not None
        assert h.worker._impl.window.num_slots == 2 * MAX_WAVE

    def test_max_in_flight_layers_caps(self):
        h = _make_harness(extra={"max_in_flight_layers": 1})
        prompt = list(range(CHUNK_TOKENS))
        req = _fake_request("r1", prompt)
        h.scheduler.update_state_after_alloc(req, object(), 0)
        so = _sched_output(new_reqs=[SimpleNamespace(
            req_id="r1", prompt_token_ids=prompt, block_ids=[0]
        )])
        meta = h.scheduler.build_connector_meta(so)
        h.worker.bind_connector_metadata(meta)
        h.worker.start_load_kv(None)
        h.hooks.bind_window(h.worker._impl.window)
        keys, _ = h.engine.hash_keys(prompt)
        for layer in range(NUM_LAYERS):
            h.hooks.source[(keys[0], layer)] = _segment(0, layer)
        h.worker.save_kv_layer(h.layer_names[0])
        h.worker.save_kv_layer(h.layer_names[1])  # 超限：最旧句柄被等待
        h.worker.save_kv_layer(h.layer_names[2])
        h.worker.wait_for_save()
        assert set(h.store.scan()) == {
            derive_io_key(keys[0], l) for l in range(NUM_LAYERS)
        }


# ----------------------------------------------------------------------
# 搬运钩子接线（worker 侧 bind 期构造）
# ----------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device not available"
)

# 接线用几何：chunk 对齐块边界（chunk_tokens=32，block_size=16 → 2 块/chunk）
WIRE_CHUNK_TOKENS = 32
WIRE_NH = 2
WIRE_HS = 64
WIRE_PTPL = 2 * 2 * WIRE_NH * WIRE_HS  # K/V × heads × dim × fp16 字节
WIRE_SEGMENT = WIRE_CHUNK_TOKENS * WIRE_PTPL


class _RecordingStore:
    """接线验证用 store 替身：接受任意缓冲，只记录注册粒度。"""

    def __init__(self):
        self.granularities: list[int] = []

    @property
    def capacity_chunks(self) -> int:
        return 16

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def register_buffer(self, buffer, granularity: int) -> int:
        self.granularities.append(granularity)
        return len(self.granularities)

    def put_batch(self, batch):
        return _ImmediateDone()

    def get_batch(self, batch):
        return _ImmediateDone()

    def drop(self, keys) -> None:
        return None

    def scan(self):
        return iter([])


class _ImmediateDone:
    def wait(self) -> None:
        return None

    def query(self) -> bool:
        return True


class TestTransferWiring:
    """加速侧池 → worker 构造搬运钩子并注入引擎；主存池 → 配置钩子保留。"""

    def _wired_engine(self, cross: bool) -> tuple[KVEngine, _RecordingStore]:
        store = _RecordingStore()
        engine = KVEngine(
            {
                "chunk_tokens": WIRE_CHUNK_TOKENS,
                "chunk_kv_bytes": WIRE_SEGMENT * NUM_LAYERS,
                "max_chunks_per_wave": MAX_WAVE,
            },
            store,
        )
        cfg = _make_vllm_config({
            "tutti_engine_instance": engine,
            "chunk_tokens": WIRE_CHUNK_TOKENS,
            "chunk_kv_bytes": WIRE_SEGMENT * NUM_LAYERS,
        })
        worker = TuttiConnectorV1(cfg, KVConnectorRole.WORKER, object())
        if cross:
            # 跨层交织池：[块数, 层数, 块大小, K/V, 通道]（fork 的
            # CROSS_LAYER 模式；逐层切片为 4-D [nb, bs, 2, kv]）
            pool = torch.zeros(
                8, NUM_LAYERS, BLOCK_SIZE, 2, WIRE_NH * WIRE_HS,
                dtype=torch.float16, device="cuda",
            )
            worker.register_cross_layers_kv_cache(pool, attn_backend=None)
        else:
            names = [f"model.layers.{i}.self_attn.attn" for i in range(NUM_LAYERS)]
            worker.register_kv_caches({
                name: torch.zeros(
                    2, 8, BLOCK_SIZE, WIRE_NH, WIRE_HS,
                    dtype=torch.float16, device="cuda",
                )
                for name in names
            })
        worker._impl._ensure_bound()
        return engine, store

    @requires_cuda
    def test_per_layer_pool_wires_hooks(self):
        engine, store = self._wired_engine(cross=False)
        assert callable(engine._transfer._gather_fn)
        assert callable(engine._transfer._scatter_fn)
        assert store.granularities == [WIRE_SEGMENT]  # 环窗注册恰一次

    @requires_cuda
    def test_cross_layer_pool_wires_hooks(self):
        engine, store = self._wired_engine(cross=True)
        assert callable(engine._transfer._gather_fn)
        assert callable(engine._transfer._scatter_fn)
        assert store.granularities == [WIRE_SEGMENT]

    def test_cpu_pools_keep_config_hooks(self):
        """主存池不接线：引擎配置中的钩子原样生效（无 GPU 环境行为不变）。"""
        h = _make_harness()
        h.worker._impl._ensure_bound()
        assert h.engine._transfer._gather_fn == h.hooks.gather
        assert h.engine._transfer._scatter_fn == h.hooks.scatter

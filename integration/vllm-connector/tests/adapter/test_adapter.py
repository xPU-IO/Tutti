"""适配层契约测试：真 vLLM 基类对接、双角色共享引擎、端到端读写往返。"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from adapter.connector import (
    TuttiConnectorMetadata,
    TuttiConnectorV1,
    _resolve_geometry,
)
from engine.core import KVEngine
from index.chunk_index import derive_io_key
from stores.tutti_nvme.layout import decode_io_key
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
        "num_layers": NUM_LAYERS,
        "store": {"type": "memory", "options": {"segment_bytes": SEG, "num_chunks": 16}},
    }
    merged.update(extra or {})
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=merged),
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
        ),
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


def _make_harness(extra: dict | None = None, num_layers: int | None = None,
                  register_per_layer: bool = True) -> _Harness:
    """手工构建 engine（注入搬运钩子），双角色挂载点共享同一实例。

    num_layers 覆盖层数几何（engine 的 chunk_kv_bytes 与逐层登记数）；
    register_per_layer=False 时不做逐层登记（cross 池模式由调用方
    自行 register_cross_layers_kv_cache）。
    """
    nl = num_layers or NUM_LAYERS
    extra = dict(extra or {})
    extra.setdefault("chunk_kv_bytes", SEG * nl)
    extra.setdefault("num_layers", nl)
    hooks = PoolHooks()
    store = _TensorAdaptingStore(MemoryKVStore(SEG, 16))
    engine = KVEngine(
        {
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_kv_bytes": SEG * nl,
            "max_chunks_per_wave": MAX_WAVE,
            "gather_fn": hooks.gather,
            "scatter_fn": hooks.scatter,
        },
        store,
    )
    cfg = _make_vllm_config({**extra, "tutti_engine_instance": engine})
    kvcc = object()
    scheduler = TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, kvcc)
    worker = TuttiConnectorV1(cfg, KVConnectorRole.WORKER, kvcc)
    layer_names = [f"model.layers.{i}.self_attn.attn" for i in range(nl)]
    if register_per_layer:
        kv_caches = {
            name: torch.zeros(8, BLOCK_SIZE, 2, 8, dtype=torch.float16)
            for name in layer_names
        }
        worker.register_kv_caches(kv_caches)
    return _Harness(
        store=store, engine=engine, hooks=hooks, vllm_config=cfg,
        scheduler=scheduler, worker=worker, layer_names=layer_names,
    )


def _fake_request(req_id: str, tokens: list[int], num_computed_tokens: int = 0):
    return SimpleNamespace(
        request_id=req_id,
        prompt_token_ids=list(tokens),
        output_token_ids=[],
        all_token_ids=list(tokens),  # 活请求序列：decode 步持续追加
        num_computed_tokens=num_computed_tokens,
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

    def test_roles_are_isolated_without_test_injection(self):
        h = _make_harness()
        # Explicit object injection remains available to the pure-Python test
        # harness, but production role construction is split.
        assert h.scheduler._engine is h.engine
        assert h.worker._engine is h.engine
        cfg = _make_vllm_config()
        a = TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())._engine
        b = TuttiConnectorV1(cfg, KVConnectorRole.WORKER, object())._engine
        assert a is not b
        assert type(a).__name__ == "SchedulerMetadataIndex"
        assert type(b).__name__ == "KVEngine"
        a.close()
        b.close()

    def test_piecewise_overridden_true(self):
        assert TuttiConnectorV1.requires_piecewise_for_cudagraph({}) is True

    def test_prefer_cross_layer_blocks_true(self):
        h = _make_harness()
        assert h.worker.prefer_cross_layer_blocks is True

    def test_requires_nhd_layout(self):
        assert TuttiConnectorV1.get_required_kvcache_layout(
            _make_vllm_config()
        ) == "NHD"

    def test_dcp_rejected_before_engine_open(self):
        cfg = _make_vllm_config()
        cfg.parallel_config.decode_context_parallel_size = 2
        with pytest.raises(ValueError, match="decode_context_parallel_size=2"):
            TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())

    def test_missing_engine_key_rejected(self):
        cfg = _make_vllm_config()
        del cfg.kv_transfer_config.kv_connector_extra_config["chunk_kv_bytes"]
        with pytest.raises(ValueError):
            TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())


class TestGeometryResolution:
    @staticmethod
    def _kv_cache_config(page_size=SEG, block_size=BLOCK_SIZE, layers=NUM_LAYERS):
        spec = SimpleNamespace(
            block_size=block_size,
            page_size_bytes=page_size,
        )
        group = SimpleNamespace(
            layer_names=[f"model.layers.{i}.attn" for i in range(layers)],
            kv_cache_spec=spec,
        )
        return SimpleNamespace(kv_cache_groups=[group])

    def test_derives_model_geometry(self):
        extra = {
            "chunk_tokens": 2 * BLOCK_SIZE,
            "max_chunks_per_wave": MAX_WAVE,
        }
        got = _resolve_geometry(extra, self._kv_cache_config())
        assert got["num_layers"] == NUM_LAYERS
        assert got["chunk_kv_bytes"] == 2 * SEG * NUM_LAYERS

    def test_connector_injects_derived_geometry_into_store(self):
        extra = {
            "chunk_tokens": 2 * BLOCK_SIZE,
            "max_chunks_per_wave": MAX_WAVE,
            "store": {"type": "memory", "options": {"num_chunks": 16}},
        }
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
            cache_config=SimpleNamespace(block_size=BLOCK_SIZE, cache_dtype="auto"),
            model_config=SimpleNamespace(model="fake-model"),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
        connector = TuttiConnectorV1(
            cfg,
            KVConnectorRole.SCHEDULER,
            self._kv_cache_config(),
        )
        assert connector._engine._num_layers == NUM_LAYERS
        assert connector._engine._chunk_kv_bytes == 2 * SEG * NUM_LAYERS
        assert connector._engine._store._segment_bytes == 2 * SEG

    def test_matching_explicit_geometry_is_accepted(self):
        extra = {
            "chunk_tokens": 2 * BLOCK_SIZE,
            "max_chunks_per_wave": MAX_WAVE,
            "num_layers": NUM_LAYERS,
            "chunk_kv_bytes": 2 * SEG * NUM_LAYERS,
        }
        assert _resolve_geometry(extra, self._kv_cache_config()) == extra

    def test_mismatched_explicit_geometry_is_rejected(self):
        extra = {
            "chunk_tokens": 2 * BLOCK_SIZE,
            "max_chunks_per_wave": MAX_WAVE,
            "num_layers": NUM_LAYERS + 1,
        }
        with pytest.raises(ValueError, match="推导值"):
            _resolve_geometry(extra, self._kv_cache_config())

    def test_multiple_cache_groups_are_rejected(self):
        config = self._kv_cache_config()
        config.kv_cache_groups.append(config.kv_cache_groups[0])
        extra = {
            "chunk_tokens": 2 * BLOCK_SIZE,
            "max_chunks_per_wave": MAX_WAVE,
        }
        with pytest.raises(ValueError, match="multi-group"):
            _resolve_geometry(extra, config)

    def test_nonuniform_layer_pages_are_rejected(self):
        names = ["layer.0", "layer.1"]
        spec = SimpleNamespace(
            block_size=BLOCK_SIZE,
            kv_cache_specs={
                names[0]: SimpleNamespace(page_size_bytes=SEG),
                names[1]: SimpleNamespace(page_size_bytes=2 * SEG),
            },
        )
        config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(
            layer_names=names,
            kv_cache_spec=spec,
        )])
        extra = {
            "chunk_tokens": 2 * BLOCK_SIZE,
            "max_chunks_per_wave": MAX_WAVE,
        }
        with pytest.raises(ValueError, match="每层 KV page"):
            _resolve_geometry(extra, config)


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
        # Planning reserves capacity only. Resident publication is gated by
        # the worker's successful save completion.
        assert h.engine.lookup_prefix(req.prompt_token_ids) == 0

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


class TestLayerResolution:
    """层名 → 层号解析（cross 池按名；禁止静默轮转）。

    回归背景：cross 池模式曾按调用序轮转（wait/save 共享计数），
    第 i 层 save 落 2i+1 (mod N)——盘上层号只余奇数、每槽双写覆盖。
    """

    NL = 80  # 仿真 Hy3 层数（几何独立于文件头常量）

    def _cross_harness(self):
        h = _make_harness(num_layers=self.NL, register_per_layer=False)
        pool = torch.zeros(4, self.NL, BLOCK_SIZE, 2, 8, dtype=torch.float16)
        h.worker.register_cross_layers_kv_cache(pool, attn_backend=None)
        return h

    def _run_prefill_step(self, h, prompt):
        """走一遍 prefill 编排（绑定 + 元数据 + 逐层 wait/save 交织）。"""
        req = _fake_request("r1", prompt)
        h.scheduler.update_state_after_alloc(req, object(), 0)
        meta = h.scheduler.build_connector_meta(_sched_output(new_reqs=[
            SimpleNamespace(
                req_id="r1", prompt_token_ids=prompt, block_ids=list(
                    range(len(prompt) // BLOCK_SIZE)
                )
            )
        ]))
        h.worker.bind_connector_metadata(meta)
        h.worker.start_load_kv(None)  # 触发绑定
        h.hooks.bind_window(h.worker._impl.window)
        names = [f"model.layers.{i}.self_attn.attn" for i in range(self.NL)]
        keys, _ = h.engine.hash_keys(prompt)
        for name in names:
            # 真实钩子序：层前 wait、层后 save——同名两次解析
            h.worker.wait_for_layer_load(name)
            for i in range(len(keys)):
                h.hooks.source[(keys[i], int(name.split(".")[2]))] = _segment(i, 0)
            h.worker.save_kv_layer(name)
        h.worker.wait_for_save()
        return keys

    def test_cross_pool_layer_indices_complete(self):
        """80 层 wait/save 交织：io_key 层号全集恰 0..79 各一次。"""
        h = self._cross_harness()
        keys = self._run_prefill_step(h, list(range(2 * CHUNK_TOKENS)))
        scanned = list(h.store.scan())
        # 每 chunk 每层恰一条（同名覆盖会令条目数减少）
        assert len(scanned) == 2 * self.NL
        layer_numbers = sorted(decode_io_key(k)[1] for k in scanned)
        assert layer_numbers == sorted(
            [layer for _ in range(2) for layer in range(self.NL)]
        )

    def test_same_name_resolves_consistently(self):
        """同名两次调用（wait 与 save）解析一致。"""
        h = self._cross_harness()
        h.worker.start_load_kv  # noqa: B018 - 不触发，仅示意
        impl = h.worker._impl
        impl._ensure_bound()
        name = "model.layers.17.self_attn.attn"
        assert impl._resolve_layer(name) == impl._resolve_layer(name) == 17

    def test_unnumbered_name_learns_stable_mapping(self):
        """无序号层名：首次出现序学习映射，同名恒定。"""
        h = self._cross_harness()
        impl = h.worker._impl
        impl._ensure_bound()
        assert impl._resolve_layer("attn_pool") == 0
        assert impl._resolve_layer("attn_pool") == 0  # 稳定
        assert impl._resolve_layer("other_block") == 1
        assert impl._resolve_layer("model.layers.3.self_attn.attn") == 3  # 正则优先

    def test_out_of_range_layer_name_raises(self):
        """越界层号直接抛错（禁止静默轮转/截断）。"""
        h = self._cross_harness()
        impl = h.worker._impl
        impl._ensure_bound()
        with pytest.raises(ValueError, match="越界"):
            impl._resolve_layer("model.layers.80.self_attn.attn")

    def test_per_layer_unknown_name_raises(self):
        """逐层登记模式：未收录层名抛错（原静默返回 0 已收紧）。"""
        h = _make_harness()
        impl = h.worker._impl
        impl._ensure_bound()
        with pytest.raises(ValueError, match="未在逐层登记"):
            impl._resolve_layer("model.layers.99.self_attn.attn")


class TestLoadSemantics:
    """加载区间 / resume 块表 / full-hit 契约（fork V1 语义对齐）。"""

    def _prefill_save(self, h, prompt, block_ids):
        """走一遍 prefill 保存（全 chunk 落盘驻留）。"""
        req = _fake_request("r1", prompt)
        h.scheduler.update_state_after_alloc(req, object(), 0)
        meta = h.scheduler.build_connector_meta(_sched_output(new_reqs=[
            SimpleNamespace(
                req_id="r1", prompt_token_ids=prompt, block_ids=block_ids
            )
        ]))
        h.worker.bind_connector_metadata(meta)
        h.worker.start_load_kv(None)
        h.hooks.bind_window(h.worker._impl.window)
        keys, _ = h.engine.hash_keys(prompt)
        for layer in range(NUM_LAYERS):
            for i, k in enumerate(keys):
                h.hooks.source[(k, layer)] = _segment(i, layer)
        for name in h.layer_names:
            h.worker.save_kv_layer(name)
        h.worker.wait_for_save()
        return keys

    def test_failed_layer_marks_invalid_and_stops_later_scatter(self):
        class Handle:
            def __init__(self, layer, fail=False):
                self.layer = layer
                self.fail = fail
                self.wait_count = 0
                self.abort_count = 0

            def wait(self):
                self.wait_count += 1
                if self.fail:
                    from engine.core import LoadGateError
                    raise LoadGateError("cq failed", whole_operation=True)

            def abort(self):
                self.abort_count += 1

        class Engine:
            def __init__(self):
                self.handles = {}
                self.calls = []
                self.unpinned = []

            def load_layer(self, keys, layer, blocks):
                self.calls.append(layer)
                handle = Handle(layer, fail=layer == 0)
                self.handles[layer] = handle
                return handle

            def abort(self):
                for handle in self.handles.values():
                    if not handle.fail:
                        handle.abort()

            def unpin(self, keys):
                self.unpinned.append(tuple(keys))

        from adapter.worker import WorkerImpl

        engine = Engine()
        worker = WorkerImpl(engine, lookahead_k=3)
        worker._num_layers = 4
        worker._load_keys = [b"k0", b"k1"]
        worker._load_block_tables = [[10], [11]]
        worker._pinned = True
        worker._prefetch_load_layers(0)
        assert engine.calls == [0, 1, 2]
        worker.wait_for_layer_load("model.layers.0.self_attn")
        assert worker.get_block_ids_with_load_errors() == {10, 11}
        assert engine.handles[1].abort_count == 1
        assert engine.handles[2].abort_count == 1
        worker.wait_for_layer_load("model.layers.1.self_attn")
        assert engine.calls == [0, 1, 2]
        assert engine.unpinned == [(b"k0", b"k1")]

    def test_abort_is_idempotent(self):
        class Engine:
            def __init__(self):
                self.abort_count = 0

            def abort(self):
                self.abort_count += 1

            def close(self):
                return None

        from adapter.worker import WorkerImpl

        engine = Engine()
        worker = WorkerImpl(engine)
        worker.abort()
        worker.abort()
        assert worker._save_inflight == []
        assert worker._load_handles == {}

    def test_save_failure_drains_and_never_publishes_resident(self):
        class FailedHandle:
            def wait(self):
                raise RuntimeError("write cq failed")

        class Engine:
            def __init__(self):
                self.wait_idle_count = 0
                self.confirm_calls = []

            def wait_idle(self):
                self.wait_idle_count += 1

            def confirm_store(self, keys, ok=True):
                self.confirm_calls.append((tuple(keys), ok))

        from adapter.worker import WorkerImpl

        engine = Engine()
        worker = WorkerImpl(engine)
        worker._save_keys = [b"k"]
        worker._save_inflight = [FailedHandle()]
        with pytest.raises(RuntimeError, match="write cq failed"):
            worker.wait_for_save()
        assert engine.wait_idle_count == 1
        assert engine.confirm_calls == [((b"k",), False)]
        assert worker._save_keys is None
        assert worker._save_inflight == []

    def test_mixed_prefix_loads_offset_window(self):
        """B3：本地命中 N 时只加载 [N, hit) 区间，不重写前 N 段。"""
        h = _make_harness()
        prompt = list(range(4 * CHUNK_TOKENS))
        blocks = list(range(4 * CHUNK_TOKENS // BLOCK_SIZE))
        keys = self._prefill_save(h, prompt, blocks)

        # 本地已有 2 chunk（computed=2CT）：外部补其后区间；上限
        # prompt-1（computed+new ≤ 4CT-1）→ new 对齐后 1 chunk
        req2 = _fake_request("r2", prompt, num_computed_tokens=2 * CHUNK_TOKENS)
        matched, _ = h.scheduler.get_num_new_matched_tokens(req2, 2 * CHUNK_TOKENS)
        assert matched == CHUNK_TOKENS
        h.scheduler.update_state_after_alloc(req2, object(), matched)
        meta2 = h.scheduler.build_connector_meta(_sched_output(new_reqs=[
            SimpleNamespace(req_id="r2", prompt_token_ids=prompt,
                            block_ids=list(range(10, 14)))  # 4 块覆盖全 prompt
        ]))
        m = meta2.requests[0]
        assert m.load_start_token == 2 * CHUNK_TOKENS
        assert m.load_tokens == CHUNK_TOKENS

        # worker 组批：恰为 chunk [2,3) 的 key 与块表（前 2 段不重写）
        h.worker.bind_connector_metadata(meta2)
        h.worker.start_load_kv(None)
        impl = h.worker._impl
        assert impl._load_keys == keys[2:3]
        assert impl._load_block_tables == [[12]]

    def test_resumed_request_replaces_block_table(self):
        """M2：preemption→resume 的 new_block_ids 为替换语义。"""
        h = _make_harness()
        prompt = list(range(CHUNK_TOKENS))
        h.scheduler.update_state_after_alloc(
            _fake_request("r1", prompt), object(), 0
        )
        h.scheduler.build_connector_meta(_sched_output(new_reqs=[
            SimpleNamespace(req_id="r1", prompt_token_ids=prompt,
                            block_ids=[0, 1])
        ]))
        # decode 增量：append 语义
        req = _fake_request("r1", prompt + [77])
        req.all_token_ids = prompt + [77]
        h.scheduler._live_requests["r1"] = req
        h.scheduler._trackers["r1"].token_ids = prompt  # 对齐切片基准
        h.scheduler.build_connector_meta(_sched_output(
            cached=SimpleNamespace(
                req_ids=["r1"], new_token_ids=[], new_block_ids=[[2]],
                all_token_ids={},
            ),
            scheduled_tokens={"r1": 1},
        ))
        assert h.scheduler._trackers["r1"].block_ids == [0, 1, 2]
        # resume：替换语义（旧块零残留）
        h.scheduler.build_connector_meta(_sched_output(
            cached=SimpleNamespace(
                req_ids=["r1"], new_token_ids=[], new_block_ids=[[20, 21, 22]],
                all_token_ids={},
                resumed_req_ids={"r1"},
            ),
            scheduled_tokens={"r1": 0},
        ))
        assert h.scheduler._trackers["r1"].block_ids == [20, 21, 22]

    def test_full_prompt_hit_capped(self):
        """M3：full-prompt 命中上限 prompt-1（对齐后）→ 调度可推进。"""
        h = _make_harness()
        prompt = list(range(4 * CHUNK_TOKENS))
        blocks = list(range(4 * CHUNK_TOKENS // BLOCK_SIZE))
        self._prefill_save(h, prompt, blocks)
        matched, _ = h.scheduler.get_num_new_matched_tokens(
            _fake_request("r2", prompt), 0
        )
        # 4 chunk 全驻留：上限 4CT-1 → 对齐 3CT（保留 ≥1 待算 token）
        assert matched == 3 * CHUNK_TOKENS


class TestEndToEnd:
    def test_store_then_load_roundtrip(self):
        h = _make_harness()
        # ---- 步 1：保存请求（24 tokens = 3 chunk）----
        prompt = list(range(3 * CHUNK_TOKENS))
        req = _fake_request("r1", prompt)
        h.scheduler.update_state_after_alloc(req, object(), 0)
        so = _sched_output(new_reqs=[SimpleNamespace(
            req_id="r1", prompt_token_ids=prompt, block_ids=[0, 1, 2]
        )])
        meta = h.scheduler.build_connector_meta(so)
        assert meta.requests[0].save_chunk_count == 3

        h.worker.bind_connector_metadata(meta)
        h.worker.start_load_kv(None)  # 触发惰性绑定（本步无读取）
        h.hooks.bind_window(h.worker._impl.window)
        chunk_keys, _ = h.engine.hash_keys(prompt)
        for layer in range(NUM_LAYERS):
            for i in range(3):
                h.hooks.source[(chunk_keys[i], layer)] = _segment(i, layer)
        for name in h.layer_names:
            h.worker.save_kv_layer(name)
        h.worker.wait_for_save()
        # store 内 3 chunk × 全部层的 io_key
        assert set(h.store.scan()) == {
            derive_io_key(chunk_keys[i], layer)
            for i in range(3) for layer in range(NUM_LAYERS)
        }

        # ---- 步 2：新请求同前缀，命中并加载（上限 prompt-1 → 2 chunk）----
        req2 = _fake_request("r2", prompt)
        matched, async_load = h.scheduler.get_num_new_matched_tokens(req2, 0)
        assert (matched, async_load) == (2 * CHUNK_TOKENS, False)
        h.scheduler.update_state_after_alloc(req2, object(), matched)
        so2 = _sched_output(new_reqs=[SimpleNamespace(
            req_id="r2", prompt_token_ids=prompt, block_ids=[4, 5, 6]
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
        # 2 chunk prompt：命中上限 prompt-1 → 对齐后 1 chunk（新契约）
        prompt = list(range(2 * CHUNK_TOKENS))
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
                req_id="r1", token_ids=prompt, block_ids=[0, 1],
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
        read = h.worker._impl.read_window
        write = h.worker._impl.write_window
        assert read.num_slots == write.num_slots == MAX_WAVE
        assert read.slot_base == 0
        assert write.slot_base == MAX_WAVE
        assert read.buffer is write.buffer
        assert len(read.buffer) == 2 * MAX_WAVE * (SEG * NUM_LAYERS // NUM_LAYERS)

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

"""vLLM 挂载点：调度侧回调与引擎计划态之间的纯翻译层。

只做协议翻译（vLLM 回调 ↔ 引擎调用），不做任何 KV 策略与存储决策；
存储实现经注册表按配置选择，选项原样透传构造。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from adapter.worker import WorkerImpl

# 模块以 adapter.* 顶层包导入，logger 名须落在 vllm 命名空间下
# 才能继承 vllm 根 logger 的 handler（否则输出被静默吞掉）。
logger = init_logger("vllm.tutti.connector")
from engine.core import KVEngine
from stores.registry import create_store

# 引擎构造所需的配置键（全部与硬件无关）
_ENGINE_KEYS = ("chunk_tokens", "chunk_kv_bytes", "max_chunks_per_wave")
# 策略键（调度取舍，引擎不消费）
_STRATEGY_KEYS = ("min_retrieve_tokens", "max_tokens_per_load")

# 同进程同配置的引擎实例表：键 = vllm_config 对象身份（值持有配置
# 强引用防 id 回收复用），二级键 = 规范化配置三元组。
_ENGINE_CACHE: dict[int, tuple[Any, dict[tuple, KVEngine]]] = {}


def _cdiv(a: int, b: int) -> int:
    """向上取整除法。"""
    return -(a // -b)


def _flatten_blocks(block_ids) -> list[int]:
    """把 vLLM 的 block id 形态（单层 list 或按缓存组嵌套）摊平为一维。"""
    if not block_ids:
        return []
    if isinstance(block_ids[0], (list, tuple)):
        return list(block_ids[0])
    return list(block_ids)


def _extra_config(vllm_config) -> dict:
    """读取 kv_connector_extra_config；缺省为空映射。"""
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    extra = getattr(transfer, "kv_connector_extra_config", None)
    return dict(extra) if extra else {}


def _expand_placeholders(value, vllm_config=None):
    """递归替换字符串值中的 {LOCAL_RANK} 占位符（多副本部署的按副本分叉）。

    rank 源优先级：vllm_config.parallel_config.rank（vLLM 各进程构造
    connector 时在手的权威副本号）→ LOCAL_RANK 环境变量 → 0。
    vLLM V1 多进程 worker 不设 LOCAL_RANK，仅靠环境变量时 4 个副本
    会全部展开为 0（真机实测），故配置对象优先。
    """
    rank = None
    parallel = getattr(vllm_config, "parallel_config", None)
    if parallel is not None:
        rank = getattr(parallel, "rank", None)
    if rank is None:
        rank = os.environ.get("LOCAL_RANK", "0")
    rank = str(rank)
    if isinstance(value, str):
        return value.replace("{LOCAL_RANK}", rank)
    if isinstance(value, dict):
        return {k: _expand_placeholders(v, vllm_config) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_placeholders(v, vllm_config) for v in value]
    return value


def _engine_for(vllm_config, extra: dict) -> KVEngine:
    """取同进程共享的引擎实例；extra 可直传实例绕过构造。

    实例表按配置对象身份索引并持有其强引用（配置存活期内 id 不
    复用；配置回收后条目随之失活——查询以 is 校验双保险，避免
    回收地址复用导致的假命中）。
    """
    injected = extra.get("tutti_engine_instance")
    if injected is not None:
        return injected
    keys = tuple(sorted((k, extra.get(k)) for k in _ENGINE_KEYS))
    entry = _ENGINE_CACHE.get(id(vllm_config))
    if entry is None or entry[0] is not vllm_config:
        entry = (vllm_config, {})
        _ENGINE_CACHE[id(vllm_config)] = entry
    engine = entry[1].get(keys)
    if engine is None:
        store_spec = extra.get("store") or {"type": "memory", "options": {}}
        options = _expand_placeholders(
            dict(store_spec.get("options") or {}), vllm_config
        )
        store = create_store(store_spec["type"], options)
        engine = KVEngine({k: extra[k] for k in _ENGINE_KEYS}, store)
        entry[1][keys] = engine
    return engine


@dataclass
class _RequestTracker:
    """调度侧的请求记账：token 序列、块分配与已保存边界。"""

    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    saved_tokens: int = 0

    def update(self, new_token_ids: list[int], new_block_ids: list[int]) -> None:
        """增量并入新调度的 token 与块。"""
        self.token_ids.extend(new_token_ids)
        self.block_ids.extend(_flatten_blocks(new_block_ids))

    def advance_save(self, chunk_tokens: int) -> tuple[int, int]:
        """推进可保存边界，返回 (起始 chunk 序号, 本次可保存 chunk 数)。

        不足一个完整 chunk 的尾部 token 舍弃；已保存过（含外部命中）
        的区间不重复保存。
        """
        token_len = len(self.token_ids)
        boundary = _cdiv(self.saved_tokens + 1, chunk_tokens) * chunk_tokens
        if token_len < boundary:
            return self.saved_tokens // chunk_tokens, 0
        target = token_len // chunk_tokens * chunk_tokens
        start = self.saved_tokens // chunk_tokens
        count = target // chunk_tokens - start
        self.saved_tokens = target
        return start, count


@dataclass
class _ReqMeta:
    """单个请求的传输计划（只用基本类型；worker 进程内自行展开）。"""

    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    load_tokens: int = 0
    save_chunk_start: int = 0
    save_chunk_count: int = 0


@dataclass
class TuttiConnectorMetadata(KVConnectorMetadata):
    """调度进程 → worker 进程的本步传输计划。"""

    requests: list[_ReqMeta]


class TuttiConnectorV1(KVConnectorBase_V1):
    """vLLM 挂载点：双角色壳。

    scheduler 角色承载调度侧回调（前缀命中、传输计划构建）；
    worker 角色的回调转发给 WorkerImpl（逐层编排）。
    """

    def __init__(self, vllm_config, role, kv_cache_config=None):
        """三参与 vLLM 工厂签名对齐；role 决定本实例承载的回调面。"""
        super().__init__(vllm_config, role, kv_cache_config)
        extra = _extra_config(vllm_config)
        for key in _ENGINE_KEYS:
            if extra.get(key) is None:
                raise ValueError(f"extra_config 缺少引擎键 {key!r}")
        self._engine = _engine_for(vllm_config, extra)
        self._chunk_tokens = extra["chunk_tokens"]
        self._min_retrieve_tokens = extra.get("min_retrieve_tokens", 0)
        self._max_tokens_per_load = extra.get("max_tokens_per_load", 0)
        self._block_size = getattr(
            getattr(vllm_config, "cache_config", None), "block_size", 16
        )
        # 调度侧请求记账
        self._trackers: dict[str, _RequestTracker] = {}
        self._pending_loads: dict[str, int] = {}
        # 调度侧活请求引用（fork 语义：cached 步的 token 增量须从活
        # 请求对象切片——scheduled_cached_reqs.new_token_ids 仅 PP 时
        # 非空，常规部署恒为空表）
        self._live_requests: dict[str, object] = {}
        # worker 角色实现
        self._impl: WorkerImpl | None = None
        if role is KVConnectorRole.WORKER:
            self._impl = WorkerImpl(
                self._engine,
                max_in_flight_layers=extra.get("max_in_flight_layers", 0),
            )
            self._impl.configure(
                chunk_tokens=self._chunk_tokens,
                chunk_kv_bytes=extra["chunk_kv_bytes"],
                max_chunks_per_wave=extra["max_chunks_per_wave"],
                block_size=self._block_size,
            )

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """声明偏好单块跨层池（统一层型模型的优化路径原料）。"""
        return True

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """逐层同步点无法进 CUDA 图整图捕获，要求分段捕获模式。"""
        return True

    # ---- worker 角色转发 ----

    def register_kv_caches(self, kv_caches: dict) -> None:
        """逐层显存对象登记（转发 worker 实现）。"""
        self._require_worker().register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(self, kv_cache, attn_backend) -> None:
        """单块跨层显存对象登记（转发 worker 实现）。"""
        self._require_worker().register_cross_layers_kv_cache(kv_cache, attn_backend)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """接收本步传输计划并同步给 worker 实现。"""
        super().bind_connector_metadata(connector_metadata)
        if self._impl is not None:
            self._impl.set_metadata(connector_metadata)

    def start_load_kv(self, forward_context=None, **kwargs) -> None:
        """发起本步读取（转发 worker 实现）。"""
        self._require_worker().start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """等待指定层读取完成（转发 worker 实现）。"""
        self._require_worker().wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer=None, attn_metadata=None, **kwargs) -> None:
        """发起指定层写入（转发 worker 实现）。"""
        self._require_worker().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        """等待本步全部写入完成（转发 worker 实现）。"""
        self._require_worker().wait_for_save()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """本实现的收发均在步内同步结算，无跨步异步完成集合。"""
        return set(), set()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """上报读取未遂的块（由上层重算兜底）。"""
        return self._require_worker().get_block_ids_with_load_errors()

    def shutdown(self) -> None:
        """收尾（转发 worker 实现）。"""
        if self._impl is not None:
            self._impl.shutdown()

    # ---- scheduler 角色回调 ----

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int | None, bool]:
        """按前缀命中报告外部可提供的 token 数（无副作用，可重复调用）。

        返回值：超出 num_computed_tokens 的可加载 token 数（chunk 对齐、
        受最小检索量与单步加载上限约束），读取为步内同步完成。

        查询前先对账持久层（多副本部署下命中查询方与落盘方
        可能分属不同进程，索引以持久层标记为准增量同步）。
        """
        self._engine.sync_from_store()
        tokens = list(getattr(request, "prompt_token_ids", None) or [])
        tokens += list(getattr(request, "output_token_ids", None) or [])
        hit = self._engine.lookup_prefix(tokens)
        new = max(0, hit - num_computed_tokens)
        new = new // self._chunk_tokens * self._chunk_tokens
        if self._min_retrieve_tokens and new < self._min_retrieve_tokens:
            new = 0
        if self._max_tokens_per_load:
            new = min(new, self._max_tokens_per_load)
        if new > 0:
            logger.info(
                "[tutti] external hit: req=%s tokens=%d computed=%d",
                request.request_id, new, num_computed_tokens,
            )
        return new, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int) -> None:
        """登记本步要加载的 token 数（块分配已完成）。

        同时保留活请求引用：请求对象的 token 序列随 decode 持续
        增长，cached 步的增量切片依赖该引用（见 build_connector_meta）。
        """
        self._live_requests[request.request_id] = request
        if num_external_tokens > 0:
            self._pending_loads[request.request_id] = num_external_tokens

    def build_connector_meta(self, scheduler_output) -> TuttiConnectorMetadata:
        """把本步调度结果折叠为传输计划；调用即重置调度侧记账。"""
        for req_id in scheduler_output.finished_req_ids:
            self._trackers.pop(req_id, None)
            self._live_requests.pop(req_id, None)
            self._pending_loads.pop(req_id, None)
        scheduled: list[str] = []
        for new_req in scheduler_output.scheduled_new_reqs:
            block_ids = _flatten_blocks(new_req.block_ids)
            cap = len(block_ids) * self._block_size
            tokens = list(new_req.prompt_token_ids)[:cap]
            self._trackers[new_req.req_id] = _RequestTracker(
                new_req.req_id, tokens, block_ids
            )
            scheduled.append(new_req.req_id)
        cached = scheduler_output.scheduled_cached_reqs
        if cached is not None:
            for i, req_id in enumerate(cached.req_ids):
                tracker = self._trackers.get(req_id)
                live = self._live_requests.get(req_id)
                if tracker is None or live is None:
                    continue
                # token 增量从活请求对象切片（new_token_ids 仅 PP 非空）
                num_new = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                have = len(tracker.token_ids)
                new_tokens = list(
                    live.all_token_ids[have : have + num_new]
                )
                blocks_i = (
                    cached.new_block_ids[i]
                    if i < len(cached.new_block_ids)
                    else None
                )
                tracker.update(new_tokens, blocks_i or [])
                scheduled.append(req_id)

        requests: list[_ReqMeta] = []
        save_specs: list[tuple[_ReqMeta, list[bytes]]] = []
        for req_id in scheduled:
            tracker = self._trackers.get(req_id)
            if tracker is None:
                continue
            start, count = tracker.advance_save(self._chunk_tokens)
            meta = _ReqMeta(
                req_id=req_id,
                token_ids=list(tracker.token_ids),
                block_ids=list(tracker.block_ids),
                load_tokens=self._pending_loads.pop(req_id, 0),
                save_chunk_start=start,
                save_chunk_count=count,
            )
            if count > 0:
                keys, _ = self._engine.hash_keys(tracker.token_ids)
                save_specs.append((meta, keys[start:start + count]))
            requests.append(meta)

        # 写入的乐观受理：近似视图立即标记驻留，读侧未遂由 worker 上报重算
        if save_specs:
            merged: list[bytes] = []
            for _meta, keys in save_specs:
                merged.extend(keys)
            plan = self._engine.plan_store(merged)
            if plan is None:
                for meta, _keys in save_specs:
                    meta.save_chunk_start, meta.save_chunk_count = 0, 0
            else:
                self._engine.confirm_store(plan.new_keys)
                if not plan.new_keys:
                    # 全部已驻留：本步无需重复写入
                    for meta, _keys in save_specs:
                        meta.save_chunk_start, meta.save_chunk_count = 0, 0
        return TuttiConnectorMetadata(requests=requests)

    def request_finished(self, request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        """请求终结：清理记账；块由上层同步释放（无跨步异步占用）。"""
        self._trackers.pop(request.request_id, None)
        self._pending_loads.pop(request.request_id, None)
        return False, None

    # ---- 内部 ----

    def _require_worker(self) -> WorkerImpl:
        if self._impl is None:
            raise RuntimeError("该回调只在 worker 角色实例上可用")
        return self._impl

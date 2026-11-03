"""vLLM 挂载点：调度侧回调与引擎计划态之间的纯翻译层。

只做协议翻译（vLLM 回调 ↔ 引擎调用），不做任何 KV 策略与存储决策；
存储实现经注册表按配置选择，选项原样透传构造。
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from tutti.integration.vllm.factory import scheduler_index_for, worker_engine_for
from tutti.integration.vllm.geometry import (
    cdiv,
    extra_config,
    flatten_blocks,
    resolve_geometry,
)
from tutti.integration.vllm.load_policy import (
    LoadModePolicy,
    resolve_free_ratio,
    resolve_load_mode,
)
from tutti.integration.vllm.worker_meta import TuttiWorkerMetadata

# 模块以 adapter.* 顶层包导入，logger 名须落在 vllm 命名空间下
# 才能继承 vllm 根 logger 的 handler（否则输出被静默吞掉）。
logger = init_logger("vllm.tutti.connector")


def _ensure_impl_logging_visible() -> None:
    """让 ``tutti.*`` 实现层日志与 vllm 日志一同可见。

    store / worker / engine 用 ``logging.getLogger(__name__)``（即 ``tutti.*``），
    而 vLLM 只配置 ``vllm.*`` 的 handler——实现层的 INFO/WARNING 会被整树过滤。
    实测（2026-09-21 长跑排查）：检查点落盘、容量裁剪、注册预热这些观测在
    生产日志里全部静默，导致"功能是否生效"只能靠盘上文件反推。

    这里给 ``tutti`` 树挂一个 handler 并放行 INFO（DEBUG 仍按开关走），已配置
    过就不重复挂；``propagate=False`` 避免与 root 重复输出。
    """
    impl = logging.getLogger("tutti")
    if impl.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    ))
    impl.addHandler(handler)
    impl.setLevel(logging.INFO)
    impl.propagate = False


_ensure_impl_logging_visible()

if TYPE_CHECKING:
    from tutti.integration.vllm.worker import WorkerImpl

# 在途写入预留的步龄上限（步）。TP rank 的完成可能落在不同步，窗口需
# 覆盖该偏差；过小会误回收正常预留（confirm 落空 → 驻留不发布 → 重复
# 重算），过大则容量泄漏回收不及时。3 步对"步内结算 + 聚合跨一步"的
# 常规节奏留有充足余量。
_PENDING_MAX_AGE_STEPS = 3


def _fmt(value, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"



@dataclass
class _RequestTracker:
    """调度侧的请求记账：token 序列、块分配与已保存边界。"""

    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    chunk_keys: list[bytes] = field(default_factory=list)
    canonical_keys: bool = False
    saved_tokens: int = 0

    def update(self, new_token_ids: list[int], new_block_ids: list[int],
               *, replace_blocks: bool = False) -> None:
        """增量并入新调度的 token 与块。

        replace_blocks=False：new_block_ids 追加到块表尾部（常规
        decode 增量）；True：整体替换块表（preemption→resume 契约，
        fork output.py：resumed 请求的 new_block_ids 是替换语义）。
        """
        self.token_ids.extend(new_token_ids)
        if replace_blocks:
            self.block_ids = flatten_blocks(new_block_ids)
        else:
            self.block_ids.extend(flatten_blocks(new_block_ids))

    def advance_save(self, chunk_tokens: int) -> tuple[int, int]:
        """推进可保存边界，返回 (起始 chunk 序号, 本次可保存 chunk 数)。

        不足一个完整 chunk 的尾部 token 舍弃；已保存过（含外部命中）
        的区间不重复保存。
        """
        token_len = len(self.token_ids)
        boundary = cdiv(self.saved_tokens + 1, chunk_tokens) * chunk_tokens
        if token_len < boundary:
            return self.saved_tokens // chunk_tokens, 0
        target = token_len // chunk_tokens * chunk_tokens
        start = self.saved_tokens // chunk_tokens
        count = target // chunk_tokens - start
        self.saved_tokens = target
        return start, count


@dataclass
class _PrefixHashCache:
    """One request's incrementally computed prefix key chain.

    vLLM keeps prompt tokens immutable and appends decode tokens to the same
    output list.  The list identities and lengths are used as the cheap
    version check; replacing either list invalidates the cache.
    """

    prompt_id: int
    output_id: int
    prompt_len: int
    output_len: int
    keys: list[bytes] = field(default_factory=list)
    parent: bytes | None = None


@dataclass
class _ReqMeta:
    """单个请求的传输计划（只用基本类型；worker 进程内自行展开）。"""

    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    # vLLM Request.block_hashes projected to Tutti chunk boundaries.  The
    # worker receives the exact scheduler keys and does not hash tokens again.
    chunk_keys: list[bytes] = field(default_factory=list)
    canonical_keys: bool = False
    load_tokens: int = 0
    load_start_token: int = 0   # 加载区间起点（本地前缀命中边界）
    save_chunk_start: int = 0
    save_chunk_count: int = 0
    save_generations: list[str] = field(default_factory=list)
    #: 异步加载：请求本步不参与计算（WAITING_FOR_REMOTE_KVS），worker 在读
    #: 流上读完后经 get_finished 回报；不进入本步的逐层 fence 与 TP 失败共识。
    load_async: bool = False


@dataclass
class TuttiConnectorMetadata(KVConnectorMetadata):
    """调度进程 → worker 进程的本步传输计划。"""

    requests: list[_ReqMeta]
    #: 调度侧本步为腾容量选定的驱逐集合（chunk key）。worker 据此执行
    #: 数据面删除 + 本 rank 索引对齐，使两侧独立 LRU 按构造收敛；worker
    #: 仍保留自身驱逐作为漂移兜底（见 KVEngine.apply_evictions）。
    evicted_keys: list[bytes] = field(default_factory=list)
    #: 本步调度的 token 总数：worker 用它把本步 GPU 计算耗时折成
    #: compute_ms(tokens) 样本（load_policy 的计算侧拟合）。
    step_tokens: int = 0
    prefill_work: bool = False


class TuttiConnectorV1(KVConnectorBase_V1):
    """vLLM 挂载点：双角色壳。

    scheduler 角色承载调度侧回调（前缀命中、传输计划构建）；
    worker 角色的回调转发给 WorkerImpl（逐层编排）。
    """

    def __init__(self, vllm_config, role, kv_cache_config=None):
        """三参与 vLLM 工厂签名对齐；role 决定本实例承载的回调面。"""
        super().__init__(vllm_config, role, kv_cache_config)
        dcp = int(getattr(
            getattr(vllm_config, "parallel_config", None),
            "decode_context_parallel_size", 1,
        ) or 1)
        extra = resolve_geometry(
            extra_config(vllm_config), kv_cache_config, dcp_size=dcp
        )
        from .replica_layout import validate_replica_io
        validate_replica_io(vllm_config, extra)
        self._vllm_config = vllm_config
        self._layerwise_kv = "kv_layer_page_bytes" in extra
        self._cache_hook_handles = ()
        if self._layerwise_kv:
            from .replica_layout import validate_layerwise_execution
            validate_layerwise_execution(vllm_config)
        if dcp not in (1, 2):
            raise ValueError(
                "TuttiConnectorV1 supports decode_context_parallel_size 1 or 2; "
                f"got {dcp}"
            )
        # 两个角色承载的对象类型不同，分开命名以免同一个字段承载两种语义：
        # WORKER 持有数据面引擎（KVEngine），SCHEDULER 持有仅元数据索引
        # （SchedulerMetadataIndex）。
        self._data_engine = None
        self._index = None
        if role is KVConnectorRole.WORKER:
            self._data_engine = worker_engine_for(vllm_config, extra)
        else:
            self._index = scheduler_index_for(vllm_config, extra)
        self._chunk_tokens = extra["chunk_tokens"]
        self._physical_chunk_tokens = extra.get(
            "physical_chunk_tokens", self._chunk_tokens
        )
        self._min_retrieve_tokens = extra.get("min_retrieve_tokens", 0)
        self._max_tokens_per_load = extra.get("max_tokens_per_load", 0)
        # 内存权威索引的发布门禁：全部 TP rank 报告持久化成功才驻留。
        self._tp_size = int(getattr(
            getattr(vllm_config, "parallel_config", None),
            "tensor_parallel_size", 1,
        ) or 1)
        self._block_size = extra.get(
            "scheduler_block_tokens",
            getattr(getattr(vllm_config, "cache_config", None), "block_size", 16),
        )
        self._kv_block_tokens = extra.get("kv_block_tokens", self._block_size)
        # 调度侧请求记账
        self._trackers: dict[str, _RequestTracker] = {}
        # Prefix key chains are request-local and bounded by the number of
        # complete chunks (typically 512 for a 128K prompt).  They avoid
        # re-hashing the same prefix on every scheduler callback.
        self._prefix_hash_cache: dict[str, _PrefixHashCache] = {}
        self._prefix_hash_calls = 0
        self._prefix_hash_cache_hits = 0
        self._prefix_hash_chunks = 0
        self._prefix_hash_ms = deque(maxlen=1024)
        self._pending_loads: dict[str, int] = {}
        # 调度侧活请求引用（fork 语义：cached 步的 token 增量须从活
        # 请求对象切片——scheduled_cached_reqs.new_token_ids 仅 PP 时
        # 非空，常规部署恒为空表）
        self._live_requests: dict[str, object] = {}
        # 外部加载区间起点 = get_num_new_matched_tokens 收到的本地命中边界。
        # 不能取 update_state_after_alloc 时的 request.num_computed_tokens：
        # vLLM 在该回调**之后**才给它赋值，此前恒为 0。
        self._match_starts: dict[str, int] = {}
        self._load_starts: dict[str, int] = {}
        # 同步/异步加载（见 load_policy）：
        #   _load_async      最近一次命中查询的判定（被调度时生效）
        #   _async_pending   已分配块、待随下一份 metadata 下发的异步加载
        #   _async_outstanding 已下发、尚未被全部 rank 报完成的异步加载
        #                    （req_id → 占用块数，供显存门槛计数）
        self._load_mode = resolve_load_mode(extra)
        self._load_async: dict[str, bool] = {}
        self._async_pending: dict[str, _ReqMeta] = {}
        self._async_outstanding: dict[str, int] = {}
        self._async_started_ns: dict[str, int] = {}
        self._async_ready_ns: dict[str, int] = {}
        self._prefetch_e2e_ms = deque(maxlen=512)
        self._prefetch_worker_ms = deque(maxlen=512)
        self._prefetch_resume_ms = deque(maxlen=512)
        self._prefetch_counters = [0, 0, 0, 0]
        self._profile_samples = deque(maxlen=1024)
        self._profile_window_bytes = 0
        self._profile_window_ops = 0
        # 显存门槛的分母：HBM KV 池总块数（单 KV 组）；未知时异步一律不放行。
        self._total_blocks = getattr(kv_cache_config, "num_blocks", None)
        self._policy: LoadModePolicy | None = None
        if role is not KVConnectorRole.WORKER:
            scheduler_config = getattr(vllm_config, "scheduler_config", None)
            self._policy = LoadModePolicy(
                mode=self._load_mode,
                free_ratio=resolve_free_ratio(extra),
                max_step_tokens=getattr(
                    scheduler_config, "max_num_batched_tokens", None
                ),
            )
        # 命中统计（原先是每请求一条 info 日志，改为累计计数 + debug）
        self._hit_tokens_total = 0
        self._hit_requests_total = 0
        # 长跑健康摘要（见 _maybe_log_health）：默认 INFO、按时间节流，与
        # 每步 debug 明细分工不同——长跑不需要每步细节，但需要"不开 DEBUG
        # 也能看到"的周期性体检（容量逼近、命中衰减、驱逐漂移、预留回收）。
        self._steps_total = 0
        self._evicted_total = 0
        try:
            self._health_interval_s = max(
                1.0, float(os.environ.get("TUTTI_HEALTH_INTERVAL_S", "30"))
            )
        except ValueError:
            self._health_interval_s = 30.0
        self._health_started_ns = time.monotonic_ns()
        self._health_last_ns = self._health_started_ns
        self._health_last = {
            "steps": 0, "hit_tokens": 0, "hit_requests": 0, "evicted": 0
        }
        # worker metadata 类型失配只告警一次（热路径，避免刷屏）
        self._meta_type_warned = False
        # worker 角色实现
        self._impl: WorkerImpl | None = None
        if role is KVConnectorRole.WORKER:
            from tutti.integration.vllm.worker import WorkerImpl

            self._impl = WorkerImpl(
                self._data_engine,
                lookahead_k=extra.get("lookahead_k", 2),
                failure_collective_timeout_s=extra.get(
                    "failure_collective_timeout_s", 30.0
                ),
            )
            self._impl.configure(
                chunk_tokens=self._chunk_tokens,
                chunk_kv_bytes=extra["chunk_kv_bytes"],
                max_chunks_per_wave=extra["max_chunks_per_wave"],
                block_size=self._kv_block_tokens,
                physical_chunk_tokens=self._physical_chunk_tokens,
                blocks_per_chunk=extra.get("blocks_per_chunk"),
                lookahead_k=extra.get("lookahead_k", 2),
                kv_group_layer_names=extra.get("kv_group_layer_names"),
            )

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """声明偏好单块跨层池（统一层型模型的优化路径原料）。"""
        return not self._layerwise_kv

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config) -> str:
        """Require the NHD cross-layer layout implemented by WorkerImpl."""
        return "NHD"

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """逐层同步点无法进 CUDA 图整图捕获，要求分段捕获模式。"""
        return True

    # ---- worker 角色转发 ----

    def register_kv_caches(self, kv_caches: dict) -> None:
        """逐层显存对象登记（转发 worker 实现）。"""
        if self._layerwise_kv and self._cache_hook_handles:
            raise RuntimeError("layerwise caches cannot be registered twice")
        self._require_worker().register_kv_caches(kv_caches)
        if self._layerwise_kv:
            try:
                from .minimax import install_cache_callbacks
                self._cache_hook_handles = install_cache_callbacks(self._vllm_config, self, kv_caches)
            except Exception:
                try:
                    self._require_worker().shutdown()
                except Exception:
                    _LOG.exception("Failed to close worker after cache callback installation error")
                raise

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
        """写入步内结算；异步加载读完的请求经此回报（finished_recving）。"""
        return self._require_worker().get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """上报读取未遂的块（由上层重算兜底）。"""
        return self._require_worker().get_block_ids_with_load_errors()

    def build_connector_worker_meta(self):
        """交出本 rank 本步的索引增量（worker → scheduler）。"""
        return self._require_worker().build_connector_worker_meta()

    def get_request_ids_with_load_errors(self) -> set[str]:
        """Return request IDs whose sampled output must be discarded."""
        return self._require_worker().get_request_ids_with_load_errors()

    def shutdown(self) -> None:
        """收尾（worker 转发给实现，scheduler 关闭元数据索引）。"""
        if self._impl is not None:
            try:
                self._impl.shutdown()
            finally:
                for handle in self._cache_hook_handles:
                    handle.remove()
                self._cache_hook_handles = ()
        elif self._index is not None:
            self._index.close()

    def abort(self) -> None:
        """Fail-closed worker abort hook used by preemption/error paths."""
        if self._impl is not None:
            self._impl.abort()

    # ---- scheduler 角色回调 ----

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int | None, bool]:
        """按前缀命中报告外部可提供的 token 数（无副作用，可重复调用）。

        返回值：超出 num_computed_tokens 的可加载 token 数（chunk 对齐、
        受最小检索量与单步加载上限约束），读取为步内同步完成。

        索引以调度进程内存为权威：驻留由 worker 落盘成功后经
        update_connector_output 增量发布，冷启动时才从盘上层标记
        重建。本回调不做任何持久层扫描——扫盘是 O(marker 数) 的
        Python 工作，放在调度热路径会直接变成调度空窗。
        """
        tokens = list(getattr(request, "prompt_token_ids", None) or [])
        tokens += list(getattr(request, "output_token_ids", None) or [])
        hit = self._lookup_request_prefix(request, tokens)
        new = max(0, hit - num_computed_tokens)
        # 命中上限：为生成首 token 保留至少一个待计算 token
        # （num_new_tokens ≥ 1，vLLM 调度推进前提；legacy 同语义）。
        # 先 clamp 后对齐——上限内取最大 chunk 对齐值。
        new = min(new, max(0, len(tokens) - 1 - num_computed_tokens))
        new = new // self._chunk_tokens * self._chunk_tokens
        if self._min_retrieve_tokens and new < self._min_retrieve_tokens:
            new = 0
        if self._max_tokens_per_load:
            new = min(new, self._max_tokens_per_load)
        req_id = request.request_id
        ready_ns = self._async_ready_ns.pop(req_id, None)
        if ready_ns is not None:
            self._prefetch_resume_ms.append(
                max(0.0, (time.perf_counter_ns() - ready_ns) / 1e6)
            )
        if new <= 0:
            self._load_async.pop(req_id, None)
            return new, False
        # 每请求一条、且是热路径：降为 debug 并聚合计数（见阶段 D 指标）。
        self._hit_tokens_total += new
        self._hit_requests_total += 1
        self._match_starts[req_id] = int(num_computed_tokens)
        is_async = False
        if self._policy is not None:
            others, held, free = self._block_usage(req_id)
            decision = self._policy.decide(
                chunks=new // self._chunk_tokens,
                compute_tokens=len(tokens) - num_computed_tokens - new,
                others=others,
                held_blocks=held,
                need_blocks=-(-(int(num_computed_tokens) + new) // self._block_size),
                free_blocks=free,
            )
            is_async = decision.is_async
            logger.debug(
                "[tutti] external hit: req=%s tokens=%d computed=%d async=%s "
                "reason=%s read_ms=%.1f compute_ms=%.1f step_ms=%.1f",
                req_id, new, num_computed_tokens, is_async, decision.reason,
                decision.read_ms, decision.compute_ms, decision.step_ms,
            )
        self._load_async[req_id] = is_async
        return new, is_async

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int) -> None:
        """登记本步要加载的 token 数（块分配已完成）。

        同时保留活请求引用：请求对象的 token 序列随 decode 持续
        增长，cached 步的增量切片依赖该引用（见 build_connector_meta）。
        外部加载区间 = [本地命中边界, 本地命中边界 + num_external_tokens)；
        边界取自 get_num_new_matched_tokens 的入参（见 _match_starts）。

        异步加载的请求本步不会出现在 scheduled_new_reqs 里，所以在这里就
        把块表与 token 序列取全，留到 build_connector_meta 下发。
        """
        req_id = request.request_id
        self._live_requests[req_id] = request
        start = self._match_starts.pop(req_id, None)
        is_async = self._load_async.pop(req_id, False)
        if num_external_tokens <= 0:
            return
        if start is None:
            start = int(getattr(request, "num_computed_tokens", 0))
        if is_async:
            tokens = list(getattr(request, "prompt_token_ids", None) or [])
            tokens += list(getattr(request, "output_token_ids", None) or [])
            get_ids = getattr(blocks, "get_block_ids", None)
            block_ids = flatten_blocks(get_ids() if callable(get_ids) else blocks)
            chunk_keys = self._vllm_chunk_keys(request, len(tokens))
            self._async_pending[req_id] = _ReqMeta(
                req_id=req_id,
                token_ids=tokens,
                block_ids=block_ids,
                chunk_keys=list(chunk_keys or []),
                canonical_keys=chunk_keys is not None,
                load_tokens=int(num_external_tokens),
                load_start_token=start,
                load_async=True,
            )
            self._async_started_ns[req_id] = time.perf_counter_ns()
            return
        self._pending_loads[req_id] = num_external_tokens
        self._load_starts[req_id] = start

    def _log_step_summary(self, scheduler_output) -> None:
        """每步容量/命中汇总：debug 明细 + info 级长跑体检。

        长稳问题（容量泄漏、假命中、驱逐漂移）在线上都表现为"命中率缓慢
        下降"，没有这行日志只能靠猜。分两级输出：
        - debug：每步明细（需 VLLM_LOGGING_LEVEL=DEBUG）；
        - info：按时间节流的体检摘要（默认可见，见 _maybe_log_health）。
        """
        self._steps_total += 1
        # 体检摘要在 debug 门控之前：长跑观测恰恰要在"不开 DEBUG"时可用。
        # 每步成本只有一次 monotonic_ns 比较，快照仅在窗口到达时才取。
        if (time.monotonic_ns() - self._health_last_ns
                >= int(self._health_interval_s * 1e9)):
            _stats = getattr(self._index, "stats", None)
            self._maybe_log_health(_stats() if callable(_stats) else {})
        if not logger.isEnabledFor(logging.DEBUG):
            return
        stats = getattr(self._index, "stats", None)
        snapshot = stats() if callable(stats) else {}
        # vLLM 的 scheduled_cached_reqs 是 CachedRequestData（无 __len__），
        # 计数要走它的 req_ids 列表。
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        logger.debug(
            "[tutti] step: new=%d cached=%d finished=%d "
            "hit_tokens_total=%d hit_requests=%d "
            "resident=%s/%s pending=%s pinned=%s drift=%s reclaimed=%s",
            len(getattr(scheduler_output, "scheduled_new_reqs", ()) or ()),
            len(getattr(cached, "req_ids", ()) or ()),
            len(getattr(scheduler_output, "finished_req_ids", ()) or ()),
            self._hit_tokens_total, self._hit_requests_total,
            snapshot.get("resident"), snapshot.get("capacity"),
            snapshot.get("pending"), snapshot.get("pinned"),
            snapshot.get("eviction_drift_total"),
            snapshot.get("pending_reclaimed_total"),
        )

    def _maybe_log_health(self, snapshot: dict) -> None:
        """长跑体检：默认 INFO、按时间节流（TUTTI_HEALTH_INTERVAL_S，默认 30s）。

        只关心"长时间运行会不会坏"：容量是否逼近上限（resident/capacity）、
        命中是否衰减、是否出现驱逐漂移（两侧 LRU 牺牲者选择不一致）或预留
        回收（某 rank 未回报）。累计值与窗口增量一起打印，趋势一眼可见。

        与 _log_step_summary 的 debug 明细分工：那条是每步事实，这条是
        周期趋势；长跑默认只看这条，无须开 DEBUG。
        """
        now = time.monotonic_ns()
        prev = self._health_last
        resident = snapshot.get("resident")
        capacity = snapshot.get("capacity")
        used_pct = (
            100.0 * resident / capacity if resident and capacity else 0.0
        )
        logger.info(
            "[tutti] health uptime=%.0fs steps=%d(+%d) "
            "resident=%s/%s(%.1f%%) pending=%s pinned=%s "
            "hit_tokens=%d(+%d) hit_reqs=%d(+%d) evicted=%d(+%d) "
            "drift=%s reclaimed=%s",
            (now - self._health_started_ns) / 1e9,
            self._steps_total, self._steps_total - prev["steps"],
            resident, capacity, used_pct,
            snapshot.get("pending"), snapshot.get("pinned"),
            self._hit_tokens_total,
            self._hit_tokens_total - prev["hit_tokens"],
            self._hit_requests_total,
            self._hit_requests_total - prev["hit_requests"],
            self._evicted_total, self._evicted_total - prev["evicted"],
            snapshot.get("eviction_drift_total"),
            snapshot.get("pending_reclaimed_total"),
        )
        hash_p50 = self._percentile(list(self._prefix_hash_ms), .50)
        hash_p95 = self._percentile(list(self._prefix_hash_ms), .95)
        hash_rate = (
            self._prefix_hash_cache_hits / self._prefix_hash_calls
            if self._prefix_hash_calls else 0.0
        )
        logger.info(
            "[tutti] index-profile hash_calls=%d cache_reuse=%.3f "
            "hashed_chunks=%d hash_ms_p50=%s hash_ms_p95=%s",
            self._prefix_hash_calls, hash_rate, self._prefix_hash_chunks,
            _fmt(hash_p50, 3), _fmt(hash_p95, 3),
        )
        if self._policy is not None:
            p = self._policy.snapshot()
            logger.info(
                "[tutti] load-policy mode=%s free_ratio=%.2f decisions=%s "
                "outstanding=%d read_ms/chunk=%s compute_ms/token=%s step_ms=%s",
                p["mode"], p["free_ratio"], p["decisions"],
                len(self._async_outstanding) + len(self._async_pending),
                _fmt(p["read_ms_per_chunk"], 3), _fmt(p["compute_ms_per_token"], 4),
                _fmt(p["step_ms"], 1),
            )
        self._log_io_profile(now)
        self._health_last_ns = now
        self._health_last = {
            "steps": self._steps_total,
            "hit_tokens": self._hit_tokens_total,
            "hit_requests": self._hit_requests_total,
            "evicted": self._evicted_total,
        }

    @staticmethod
    def _percentile(values, fraction):
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

    def _log_io_profile(self, now_ns: int) -> None:
        samples = list(self._profile_samples)
        read = [sample for sample in samples if sample[4] > 0]
        read_stream_ms = [sample[4] for sample in read]
        wall_ms = [sample[5] for sample in read]
        queue_ms = [sample[6] for sample in read]
        sync = [sample for sample in read if sample[0] == 0]
        overlap = [sample[8] for sample in sync if sample[8] >= 0]
        exposed_wait = [sample[9] for sample in sync if sample[9] >= 0]
        ranks = max(1, self._tp_size)
        duration_s = max(0.0, (now_ns - self._health_last_ns) / 1e9)
        effective_gbps = (
            self._profile_window_bytes / (duration_s * 1e9)
            if duration_s > 0 else 0.0
        )
        logger.info(
            "[tutti] io-profile window_s=%.1f samples=%d sync_plans=%d "
            "prefetch_plans=%d logical_io_ops=%d bytes=%d "
            "effective_gbps=%.3f read_stream_ms_p50=%s read_stream_ms_p95=%s "
            "plan_wall_ms_p50=%s prefetch_queue_ms_p50=%s "
            "prefetch_queue_ms_p95=%s overlap_plan_rate=%s "
            "overlap_layer_ratio=%s exposed_wait_ms_p50=%s "
            "exposed_wait_ms_p95=%s prefetch_request_totals=%s "
            "prefetch_worker_ms_p50=%s prefetch_worker_ms_p95=%s "
            "prefetch_e2e_ms_p50=%s prefetch_e2e_ms_p95=%s "
            "resume_delay_ms_p50=%s io_kernel_time=nsys_required",
            duration_s, len(read) / ranks, len(sync) / ranks,
            sum(s[0] == 1 for s in read) / ranks,
            self._profile_window_ops, self._profile_window_bytes,
            effective_gbps,
            _fmt(self._percentile(read_stream_ms, .50), 1),
            _fmt(self._percentile(read_stream_ms, .95), 1),
            _fmt(self._percentile(wall_ms, .50), 1),
            _fmt(self._percentile(queue_ms, .50), 1),
            _fmt(self._percentile(queue_ms, .95), 1),
            _fmt(sum(value > 0 for value in overlap) / len(overlap), 3)
            if overlap else "n/a",
            _fmt(sum(overlap) / len(overlap), 3) if overlap else "n/a",
            _fmt(self._percentile(exposed_wait, .50), 1),
            _fmt(self._percentile(exposed_wait, .95), 1),
            tuple(round(value / ranks, 3) for value in self._prefetch_counters),
            _fmt(self._percentile(list(self._prefetch_worker_ms), .50), 1),
            _fmt(self._percentile(list(self._prefetch_worker_ms), .95), 1),
            _fmt(self._percentile(list(self._prefetch_e2e_ms), .50), 1),
            _fmt(self._percentile(list(self._prefetch_e2e_ms), .95), 1),
            _fmt(self._percentile(list(self._prefetch_resume_ms), .50), 1),
        )
        self._profile_samples.clear()
        self._profile_window_bytes = 0
        self._profile_window_ops = 0

    def build_connector_meta(self, scheduler_output) -> TuttiConnectorMetadata:
        """把本步调度结果折叠为传输计划；调用即重置调度侧记账。"""
        # 每步推进一次索引步进，并回收长期未结算的在途预留（容量泄漏兜底）。
        begin_step = getattr(self._index, "begin_step", None)
        if callable(begin_step):
            begin_step(_PENDING_MAX_AGE_STEPS)
        self._log_step_summary(scheduler_output)
        for req_id in scheduler_output.finished_req_ids:
            self._trackers.pop(req_id, None)
            self._live_requests.pop(req_id, None)
            self._pending_loads.pop(req_id, None)
            self._load_starts.pop(req_id, None)
            self._prefix_hash_cache.pop(str(req_id), None)
        prefill_work = False
        scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        for req_id, count in scheduled_tokens.items():
            live = self._live_requests.get(req_id)
            if live is not None and count > 0:
                prefill_work |= live.num_computed_tokens < len(live.prompt_token_ids)
        scheduled: list[str] = []
        for new_req in scheduler_output.scheduled_new_reqs:
            block_ids = flatten_blocks(new_req.block_ids)
            cap = len(block_ids) * self._block_size
            tokens = list(new_req.prompt_token_ids)[:cap]
            live = self._live_requests.get(new_req.req_id)
            chunk_keys = self._vllm_chunk_keys(live, len(tokens))
            self._trackers[new_req.req_id] = _RequestTracker(
                new_req.req_id, tokens, block_ids,
                chunk_keys or [], chunk_keys is not None,
            )
            scheduled.append(new_req.req_id)
        cached = scheduler_output.scheduled_cached_reqs
        resumed = getattr(cached, "resumed_req_ids", None) or set()
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
                tracker.update(
                    new_tokens, blocks_i or [],
                    replace_blocks=req_id in resumed,
                )
                chunk_keys = self._vllm_chunk_keys(live, len(tracker.token_ids))
                if chunk_keys is not None:
                    tracker.chunk_keys = chunk_keys
                    tracker.canonical_keys = True
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
                chunk_keys=list(tracker.chunk_keys),
                canonical_keys=tracker.canonical_keys,
                load_tokens=self._pending_loads.pop(req_id, 0),
                load_start_token=self._load_starts.pop(req_id, 0),
                save_chunk_start=start,
                save_chunk_count=count,
            )
            if count > 0:
                keys = self._tracker_keys(tracker)
                save_specs.append((meta, keys[start:start + count]))
            requests.append(meta)

        # 写入只做容量计划，不在 scheduler 侧发布 resident。worker 在
        # 对应 save completion 成功后 confirm_store(ok=True)；失败路径
        # confirm_store(ok=False)，避免底层失败时产生假命中。
        evicted_keys: list[bytes] = []
        if save_specs:
            merged: list[bytes] = []
            for _meta, keys in save_specs:
                merged.extend(keys)
            plan = self._index.plan_store(merged)
            if plan is None:
                for meta, _keys in save_specs:
                    meta.save_chunk_start, meta.save_chunk_count = 0, 0
                    meta.save_generations = []
            else:
                # 调度侧选定的牺牲者下发给 worker 执行（两侧 LRU 收敛）。
                evicted_keys = list(plan.evicted_keys)
                self._evicted_total += len(evicted_keys)
                if not plan.new_keys:
                    # 全部已驻留：本步无需重复写入
                    for meta, _keys in save_specs:
                        meta.save_chunk_start, meta.save_chunk_count = 0, 0
                        meta.save_generations = []
                else:
                    generation_by_key = {
                        key: uuid.uuid4().hex for key in dict.fromkeys(merged)
                    }
                    for meta, keys in save_specs:
                        meta.save_generations = [
                            generation_by_key[key] for key in keys
                        ]
        # 异步加载随本步 metadata 下发：请求本步不计算，worker 只读不写。
        # 即使请求已被终止也照发——vLLM 对 WAITING_FOR_REMOTE_KVS 请求延迟
        # 释放块，直到全部 rank 报 finished_recving（见 scheduler.finish_requests）。
        for req_id, meta in self._async_pending.items():
            requests.append(meta)
            self._async_outstanding[req_id] = len(meta.block_ids)
        self._async_pending = {}
        step_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", None)
        if step_tokens is None:
            step_tokens = sum(
                (getattr(scheduler_output, "num_scheduled_tokens", None) or {}).values()
            )
        return TuttiConnectorMetadata(
            requests=requests, evicted_keys=evicted_keys,
            step_tokens=int(step_tokens or 0), prefill_work=prefill_work,
        )

    def update_connector_output(self, connector_output) -> None:
        """应用 worker 回传的持久化增量（内存权威索引的推进口）。

        vLLM 每步在 scheduler 侧调用一次，携带 KVOutputAggregator
        跨 TP rank 聚合后的 worker metadata。全部 rank 报告成功的
        chunk 才发布驻留；任一 rank 失败即 fail-closed 回收预留。

        异步加载的完成（finished_recving）也在这里结算：聚合器只在全部
        rank 都报完成时才放进集合，此后它不再算作在途。
        """
        for req_id in getattr(connector_output, "finished_recving", None) or ():
            self._async_outstanding.pop(req_id, None)
            req_id = str(req_id)
            now = time.perf_counter_ns()
            self._async_ready_ns[req_id] = now
            started = self._async_started_ns.pop(req_id, None)
            if started is not None:
                self._prefetch_e2e_ms.append(max(0.0, (now - started) / 1e6))
        meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if meta is None:
            return
        if not isinstance(meta, TuttiWorkerMetadata):
            # 传输层序列化格式变更（如上游从 pickle 切到 msgspec）会让类型
            # 静默失配，此时索引永远推不动、命中率归零。这里告警一次并
            # 尝试按映射重建，避免"无日志的静默失效"。
            rebuilt = self._coerce_worker_meta(meta)
            if rebuilt is None:
                self._meta_type_warned = True
                return
            meta = rebuilt
        if self._policy is not None:
            # 各 rank 样本已被聚合器拼在一起；TP 下一步跟最慢的 rank 走，
            # 所以每步只取最慢的一条（读按每 chunk 耗时、步按耗时）。
            reads = [(int(c), float(ms)) for c, ms in
                     getattr(meta, "read_samples", None) or () if c > 0 and ms > 0]
            if reads:
                self._policy.observe_read(*max(reads, key=lambda s: s[1] / s[0]))
            steps = [(int(t), float(ms)) for t, ms in
                     getattr(meta, "step_samples", None) or () if t > 0 and ms > 0]
            if steps:
                self._policy.observe_step(*max(steps, key=lambda s: s[1]))
        for sample in getattr(meta, "profile_samples", None) or ():
            if len(sample) != 10:
                continue
            self._profile_samples.append(tuple(sample))
            self._profile_window_bytes += int(sample[7])
            self._profile_window_ops += int(sample[3])
        stats = getattr(meta, "prefetch_stats", (0, 0, 0, 0))
        for i, value in enumerate(stats):
            self._prefetch_counters[i] += int(value)
        self._prefetch_worker_ms.extend(
            float(value) for value in getattr(meta, "prefetch_wait_ms", None) or ()
            if float(value) >= 0
        )
        apply_commits = getattr(self._index, "apply_worker_commits", None)
        if not callable(apply_commits):
            return
        apply_commits(meta.committed, meta.failed, self._tp_size,
                      getattr(meta, "forgotten", None),
                      getattr(meta, "evicted", None))

    def _coerce_worker_meta(self, raw) -> TuttiWorkerMetadata | None:
        """把非预期类型（如 dict）的 worker meta 重建为 dataclass。

        只在首次遇到时告警，避免热路径刷屏。
        """
        if not isinstance(raw, dict):
            if not self._meta_type_warned:
                self._meta_type_warned = True
                logger.warning(
                    "[tutti] worker metadata 类型非预期：%s（期望 %s）；"
                    "索引将停止推进，请检查 vLLM 的 connector metadata 序列化",
                    type(raw).__name__, TuttiWorkerMetadata.__name__,
                )
            return None
        committed = raw.get("committed") or {}
        failed = raw.get("failed") or ()
        forgotten = raw.get("forgotten") or ()
        evicted = raw.get("evicted") or ()
        profile_samples = raw.get("profile_samples") or ()
        prefetch_stats = raw.get("prefetch_stats") or (0, 0, 0, 0)
        prefetch_wait_ms = raw.get("prefetch_wait_ms") or ()
        try:
            rebuilt = TuttiWorkerMetadata(
                committed={bytes(k): int(v) for k, v in dict(committed).items()},
                failed={bytes(k) for k in failed},
                forgotten={bytes(k) for k in forgotten},
                evicted={bytes(k) for k in evicted},
                profile_samples=[tuple(item) for item in profile_samples],
                prefetch_stats=tuple(int(item) for item in prefetch_stats),
                prefetch_wait_ms=[float(item) for item in prefetch_wait_ms],
            )
        except Exception:
            return None
        if not self._meta_type_warned:
            self._meta_type_warned = True
            logger.warning(
                "[tutti] worker metadata 以映射形式到达（序列化格式变更），"
                "已按兼容路径重建；建议对齐 TuttiWorkerMetadata 的编解码契约",
            )
        return rebuilt

    def request_finished(self, request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        """请求终结：清理全部按请求记账的状态；块由上层同步释放。

        除 _trackers/_pending_loads 外，_live_requests/_load_starts 也必须
        在此清除：两者虽然由 build_connector_meta 的 finished_req_ids 路径
        兜底，但请求若在该兜底之前终结（异常终止/提前 abort），条目会滞留
        并持有请求对象引用。
        """
        self._trackers.pop(request.request_id, None)
        self._pending_loads.pop(request.request_id, None)
        self._live_requests.pop(request.request_id, None)
        self._load_starts.pop(request.request_id, None)
        self._match_starts.pop(request.request_id, None)
        self._prefix_hash_cache.pop(str(request.request_id), None)
        self._load_async.pop(request.request_id, None)
        self._async_started_ns.pop(request.request_id, None)
        self._async_ready_ns.pop(request.request_id, None)
        # _async_pending / _async_outstanding 不清：读仍要完成并回报
        # finished_recving，vLLM 才会释放这些块（见 build_connector_meta）。
        return False, None

    # ---- 内部 ----

    def _lookup_request_prefix(self, request, tokens: list[int]) -> int:
        """Incrementally hash and query one request's prefix.

        Only one new chunk is hashed per call, and only while all previously
        computed chunks are resident.  This preserves the old early-stop
        behavior on a cold cache while making repeated scheduler calls O(1)
        until the request grows across a chunk boundary.
        """
        req_id = str(request.request_id)
        started_ns = time.perf_counter_ns()
        self._prefix_hash_calls += 1
        prompt = getattr(request, "prompt_token_ids", None)
        output = getattr(request, "output_token_ids", None)
        prompt = prompt if prompt is not None else ()
        output = output if output is not None else ()
        prompt_len = len(prompt)
        output_len = len(output)
        cached = self._prefix_hash_cache.get(req_id)
        if (cached is None or cached.prompt_id != id(prompt)
                or cached.output_id != id(output)
                or cached.prompt_len != prompt_len
                or output_len < cached.output_len):
            cached = _PrefixHashCache(
                prompt_id=id(prompt), output_id=id(output),
                prompt_len=prompt_len, output_len=output_len,
            )
            self._prefix_hash_cache[req_id] = cached
        else:
            cached.output_len = output_len
            self._prefix_hash_cache_hits += 1

        total_chunks = len(tokens) // self._chunk_tokens
        external_keys = self._vllm_chunk_keys(request, len(tokens))
        if external_keys is not None:
            return self._index.lookup_prefix_keys(external_keys[:total_chunks])
        old_chunks = len(cached.keys)
        while len(cached.keys) < total_chunks:
            # If the already computed chain stops at a cold chunk, do not
            # spend CPU hashing later chunks that cannot be used yet.  Once
            # the chunk becomes resident, the next callback resumes here.
            if (self._index.lookup_prefix_keys(cached.keys)
                    < len(cached.keys) * self._chunk_tokens):
                break
            start = len(cached.keys) * self._chunk_tokens
            new_keys, parent = self._index.hash_keys(
                tokens, start=start, parent=cached.parent, max_keys=1
            )
            if not new_keys:
                break
            cached.keys.append(new_keys[0])
            cached.parent = parent
        self._prefix_hash_chunks += len(cached.keys) - old_chunks
        self._prefix_hash_ms.append(
            max(0.0, (time.perf_counter_ns() - started_ns) / 1e6)
        )
        return self._index.lookup_prefix_keys(cached.keys)

    def _vllm_chunk_keys(self, request, token_count: int) -> list[bytes] | None:
        """Project vLLM's canonical block hashes onto Tutti chunks.

        Tutti's on-disk key remains 16 bytes for compatibility with existing
        objects; using the prefix of vLLM's SHA-256 block hash preserves the
        upstream hash chain and avoids a second token-hashing implementation.
        Return ``None`` when the request or configured chunk geometry does
        not expose compatible upstream hashes.
        """
        if request is None:
            return None
        hashes = list(getattr(request, "block_hashes", ()) or ())
        if not hashes:
            return None
        block_tokens = int(self._block_size)
        if block_tokens <= 0 or self._chunk_tokens % block_tokens:
            return None
        per_chunk = self._chunk_tokens // block_tokens
        complete_chunks = min(token_count // self._chunk_tokens,
                              len(hashes) // per_chunk)
        if complete_chunks <= 0:
            return []
        keys = []
        for idx in range(complete_chunks):
            raw = bytes(hashes[(idx + 1) * per_chunk - 1])
            if len(raw) < 16:
                return None
            keys.append(raw[:16])
        return keys

    def _tracker_keys(self, tracker: _RequestTracker) -> list[bytes]:
        keys = list(tracker.chunk_keys)
        expected = len(tracker.token_ids) // self._chunk_tokens
        if tracker.canonical_keys:
            if len(keys) < expected:
                raise RuntimeError(
                    "vLLM block hashes do not cover complete Tutti chunks"
                )
            return keys
        # Compatibility for unit-test request doubles without block_hashes.
        return self._index.hash_keys(tracker.token_ids)[0]

    def _block_usage(self, req_id: str) -> tuple[int, int, int | None]:
        """(在算的其他请求数, 异步加载占着的块数, 估计空闲块数)。

        connector 看不到 vLLM 的块池，按活请求估：在算的请求每个占
        ceil(已计 token / block_size) 块，异步加载按已分配块数计。被抢占的
        请求仍在 _live_requests 里、前缀共享块会被重复计——都让空闲块偏少，
        即偏向同步，不会多放异步。
        """
        held = sum(self._async_outstanding.values()) + sum(
            len(meta.block_ids) for meta in self._async_pending.values()
        )
        others = 0
        used = 0
        for rid, request in self._live_requests.items():
            if (rid == req_id or rid in self._async_outstanding
                    or rid in self._async_pending):
                continue
            # Only a request currently in vLLM's RUNNING batch can hide a
            # detached read. WAITING/PREEMPTED requests still live in the
            # connector map, but counting them as ``others`` admits async
            # loads with no foreground compute heartbeat and can leave the
            # detached feeder parked indefinitely.
            status = getattr(request, "status", None)
            if status is not None and getattr(status, "name", str(status)) != "RUNNING":
                continue
            others += 1
            computed = int(getattr(request, "num_computed_tokens", 0) or 0)
            used += -(-computed // self._block_size)
        if not self._total_blocks:
            return others, held, None
        return others, held, int(self._total_blocks) - used - held

    def _require_worker(self) -> WorkerImpl:
        if self._impl is None:
            raise RuntimeError("该回调只在 worker 角色实例上可用")
        return self._impl

"""vLLM 挂载点：调度侧回调与引擎计划态之间的纯翻译层。

只做协议翻译（vLLM 回调 ↔ 引擎调用），不做任何 KV 策略与存储决策；
存储实现经注册表按配置选择，选项原样透传构造。
"""

from __future__ import annotations

import logging
import os
import time
import uuid
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
    apply_capacity_bytes,
    apply_device_groups,
    cdiv,
    deployment_rank,
    expand_placeholders,
    extra_config,
    flatten_blocks,
    resolve_geometry,
)
from tutti.integration.vllm.worker_meta import TuttiWorkerMetadata

# 模块以 adapter.* 顶层包导入，logger 名须落在 vllm 命名空间下
# 才能继承 vllm 根 logger 的 handler（否则输出被静默吞掉）。
logger = init_logger("vllm.tutti.connector")

if TYPE_CHECKING:
    from tutti.integration.vllm.worker import WorkerImpl

# 兼容别名：配置解析已迁至 adapter.geometry / adapter.factory，以下同名
# 私有符号保留供既有调用点与测试 import 使用（行为完全一致）。
_resolve_geometry = resolve_geometry
_deployment_rank = deployment_rank
_expand_placeholders = expand_placeholders
_apply_device_groups = apply_device_groups
_apply_capacity_bytes = apply_capacity_bytes
_worker_engine_for = worker_engine_for
_scheduler_index_for = scheduler_index_for

# 在途写入预留的步龄上限（步）。TP rank 的完成可能落在不同步，窗口需
# 覆盖该偏差；过小会误回收正常预留（confirm 落空 → 驻留不发布 → 重复
# 重算），过大则容量泄漏回收不及时。3 步对"步内结算 + 聚合跨一步"的
# 常规节奏留有充足余量。
_PENDING_MAX_AGE_STEPS = 3



@dataclass
class _RequestTracker:
    """调度侧的请求记账：token 序列、块分配与已保存边界。"""

    req_id: str
    token_ids: list[int]
    block_ids: list[int]
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
class _ReqMeta:
    """单个请求的传输计划（只用基本类型；worker 进程内自行展开）。"""

    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    load_tokens: int = 0
    load_start_token: int = 0   # 加载区间起点（vLLM 已计 token 数）
    save_chunk_start: int = 0
    save_chunk_count: int = 0
    save_generations: list[str] = field(default_factory=list)


@dataclass
class TuttiConnectorMetadata(KVConnectorMetadata):
    """调度进程 → worker 进程的本步传输计划。"""

    requests: list[_ReqMeta]
    #: 调度侧本步为腾容量选定的驱逐集合（chunk key）。worker 据此执行
    #: 数据面删除 + 本 rank 索引对齐，使两侧独立 LRU 按构造收敛；worker
    #: 仍保留自身驱逐作为漂移兜底（见 KVEngine.apply_evictions）。
    evicted_keys: list[bytes] = field(default_factory=list)


class TuttiConnectorV1(KVConnectorBase_V1):
    """vLLM 挂载点：双角色壳。

    scheduler 角色承载调度侧回调（前缀命中、传输计划构建）；
    worker 角色的回调转发给 WorkerImpl（逐层编排）。
    """

    def __init__(self, vllm_config, role, kv_cache_config=None):
        """三参与 vLLM 工厂签名对齐；role 决定本实例承载的回调面。"""
        super().__init__(vllm_config, role, kv_cache_config)
        extra = _resolve_geometry(extra_config(vllm_config), kv_cache_config)
        dcp = getattr(
            getattr(vllm_config, "parallel_config", None),
            "decode_context_parallel_size",
            1,
        )
        if dcp != 1:
            # Scheduler blocks and worker slot mappings are DCP-rank/interleave
            # dependent. Fail before either role constructs its store/index.
            raise ValueError(
                "TuttiConnectorV1 当前不支持 decode context parallelism；"
                f"decode_context_parallel_size={dcp}，必须为 1"
            )
        # 两个角色承载的对象类型不同，分开命名以免同一个字段承载两种语义：
        # WORKER 持有数据面引擎（KVEngine），SCHEDULER 持有仅元数据索引
        # （SchedulerMetadataIndex）。`_engine` 属性保留为兼容别名。
        self._data_engine = None
        self._index = None
        if role is KVConnectorRole.WORKER:
            self._data_engine = _worker_engine_for(vllm_config, extra)
        else:
            self._index = _scheduler_index_for(vllm_config, extra)
        self._chunk_tokens = extra["chunk_tokens"]
        self._min_retrieve_tokens = extra.get("min_retrieve_tokens", 0)
        self._max_tokens_per_load = extra.get("max_tokens_per_load", 0)
        # 内存权威索引的发布门禁：全部 TP rank 报告持久化成功才驻留。
        self._tp_size = int(getattr(
            getattr(vllm_config, "parallel_config", None),
            "tensor_parallel_size", 1,
        ) or 1)
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
        # 外部加载区间起点（update_state_after_alloc 时的已计 token 数）
        self._load_starts: dict[str, int] = {}
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
                lookahead_k=extra.get(
                    "lookahead_k", extra.get("prefetch_k", 2)
                ),
                failure_collective_timeout_s=extra.get(
                    "failure_collective_timeout_s", 30.0
                ),
            )
            self._impl.configure(
                chunk_tokens=self._chunk_tokens,
                chunk_kv_bytes=extra["chunk_kv_bytes"],
                max_chunks_per_wave=extra["max_chunks_per_wave"],
                block_size=self._block_size,
                lookahead_k=extra.get(
                    "lookahead_k", extra.get("prefetch_k", 2)
                ),
                kv_group_layer_names=extra.get("kv_group_layer_names"),
            )

    @property
    def _engine(self):
        """兼容别名：返回本角色实际承载的对象（scheduler→索引，worker→引擎）。

        历史上同一个字段承载两种类型（评审 N1 指出的命名歧义）；现已拆为
        ``_index`` / ``_data_engine``，此属性仅为既有调用点与测试保留。
        """
        return self._index if self._index is not None else self._data_engine

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """声明偏好单块跨层池（统一层型模型的优化路径原料）。"""
        return True

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

    def build_connector_worker_meta(self):
        """交出本 rank 本步的索引增量（worker → scheduler）。"""
        return self._require_worker().build_connector_worker_meta()

    def get_request_ids_with_load_errors(self) -> set[str]:
        """Return request IDs whose sampled output must be discarded."""
        return self._require_worker().get_request_ids_with_load_errors()

    def shutdown(self) -> None:
        """收尾（worker 转发给实现，scheduler 关闭元数据索引）。"""
        if self._impl is not None:
            self._impl.shutdown()
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
        hit = self._index.lookup_prefix(tokens)
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
        if new > 0:
            # 每请求一条、且是热路径：降为 debug 并聚合计数（见阶段 D 指标）。
            self._hit_tokens_total += new
            self._hit_requests_total += 1
            logger.debug(
                "[tutti] external hit: req=%s tokens=%d computed=%d",
                request.request_id, new, num_computed_tokens,
            )
        return new, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int) -> None:
        """登记本步要加载的 token 数（块分配已完成）。

        同时保留活请求引用：请求对象的 token 序列随 decode 持续
        增长，cached 步的增量切片依赖该引用（见 build_connector_meta）。
        外部加载区间起点 = 请求当前已计 token 数（vLLM 本地前缀
        命中计入其中，connector 只补其后区间）。
        """
        self._live_requests[request.request_id] = request
        if num_external_tokens > 0:
            self._pending_loads[request.request_id] = num_external_tokens
            self._load_starts[request.request_id] = int(
                getattr(request, "num_computed_tokens", 0)
            )

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
        self._health_last_ns = now
        self._health_last = {
            "steps": self._steps_total,
            "hit_tokens": self._hit_tokens_total,
            "hit_requests": self._hit_requests_total,
            "evicted": self._evicted_total,
        }

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
        scheduled: list[str] = []
        for new_req in scheduler_output.scheduled_new_reqs:
            block_ids = flatten_blocks(new_req.block_ids)
            cap = len(block_ids) * self._block_size
            tokens = list(new_req.prompt_token_ids)[:cap]
            self._trackers[new_req.req_id] = _RequestTracker(
                new_req.req_id, tokens, block_ids
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
                load_start_token=self._load_starts.pop(req_id, 0),
                save_chunk_start=start,
                save_chunk_count=count,
            )
            if count > 0:
                keys, _ = self._index.hash_keys(tracker.token_ids)
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
        return TuttiConnectorMetadata(
            requests=requests, evicted_keys=evicted_keys
        )

    def update_connector_output(self, connector_output) -> None:
        """应用 worker 回传的持久化增量（内存权威索引的推进口）。

        vLLM 每步在 scheduler 侧调用一次，携带 KVOutputAggregator
        跨 TP rank 聚合后的 worker metadata。全部 rank 报告成功的
        chunk 才发布驻留；任一 rank 失败即 fail-closed 回收预留。
        """
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
        try:
            rebuilt = TuttiWorkerMetadata(
                committed={bytes(k): int(v) for k, v in dict(committed).items()},
                failed={bytes(k) for k in failed},
                forgotten={bytes(k) for k in forgotten},
                evicted={bytes(k) for k in evicted},
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
        return False, None

    # ---- 内部 ----

    def _require_worker(self) -> WorkerImpl:
        if self._impl is None:
            raise RuntimeError("该回调只在 worker 角色实例上可用")
        return self._impl

"""worker 侧逐层编排：把 vLLM 的层回调翻译为引擎执行态调用。

读取按层流水（等待第 L 层时预取第 L+1 层）；写入逐层发起并以在途
句柄上限背压；staging 环窗显存在此分配并注入引擎。
"""

from __future__ import annotations

import logging
import re

import torch
from tutti_kv_transfer import (
    batched_layer_transfer,
    EngineKVFormat,
    discover_engine_format,
    single_layer_transfer,
)

from engine.staging import RingWindow
from engine.core import LoadGateError
from engine.nvtx import range as nvtx_range

#: 单步内块表缓存上限（超出即整体重建，防无界增长）。
_SLOT_CACHE_LIMIT = 1024

#: 层名序号提取（vLLM 层名约定 model.layers.{i}....）。
_LAYER_NAME_RE = re.compile(r"layers\.(\d+)")

#: 运行日志（准入失败等容量配置问题的非静默说明）。
_LOG = logging.getLogger(__name__)


def _load_chunk_span(start_token: int, token_count: int,
                     chunk_tokens: int) -> tuple[int, int]:
    """加载区间 [start, start+count) 覆盖的完整 chunk 区间。

    返回 (首个 chunk 序号, chunk 数)：起点向上取整到 chunk 边界
    （首个不完整 chunk 让渡给重算），终点向下取整。区间非法或
    无完整 chunk → (0, 0)。
    """
    if start_token < 0 or token_count <= 0:
        return 0, 0
    first = (start_token + chunk_tokens - 1) // chunk_tokens
    last = (start_token + token_count) // chunk_tokens
    return first, max(0, last - first)


class PagedTransferHooks:
    """staged 路径的两端搬运：paged 池 ↔ staging 槽。

    支持两类池布局：
    - 逐层 5-D/3-D 池（[2, nb, bs, nh, hs] 等常见形态）：经布局发现
      例程定格式，走搬运函数（single_layer_transfer）。
    - 跨层交织池 [nb, nl, bs, 2, kv]（CROSS_LAYER 模式，K/V 在 token
      行内交织）：逐层切片 [nb, bs, 2, kv] 与槽段 [tokens, 2, kv]
      字节同构，按块表索引直接拷贝（搬运函数的格式分支尚未覆盖
      该布局，见 results/ 的 cpp-gap 记录）。

    契约：
    - 构造注入：layer_view 为层序号 → 该层 paged 张量的访问器；staging
      为一维 uint8 字节张量（槽数 × 段长，加速侧或页锁定主存均可）；
      segment_bytes / chunk_tokens / block_size 为几何；fmt 为池布局
      格式整数（交织布局下忽略，可传 None）。chunk_tokens 须为
      block_size 的正整数倍，否则 ValueError。
    - 钩子签名对齐引擎侧调用约定 (keys, layer_idx, block_tables, slots)：
      block_tables 为逐 chunk 的块号列表（每 chunk 恰 chunk_tokens //
      block_size 个，块号可任意分布）；slots 为 staging 槽号列表。
      gather 在写入批发起前执行（源侧 → 槽）；scatter 在完成句柄
      wait 后执行（槽 → 目的侧）。
    - 层段按 token 主序打包（[tokens, 2, H]；单张量几何为 [tokens, H]）。
    - gather 返回前同步全部设备（staging 落 0 号加速器，池可能异卡，
      提交批次读槽前须确保跨卡拷贝全部落地）。
    """

    _INTERLEAVED = "interleaved"
    _KERNEL = "kernel"

    def __init__(self, layer_view, staging, segment_bytes: int,
                 chunk_tokens: int, block_size: int, fmt):
        self._layer_view = layer_view
        self._staging = staging
        self._segment_bytes = segment_bytes
        self._chunk_tokens = chunk_tokens
        self._block_size = block_size
        self._fmt = fmt
        if chunk_tokens % block_size != 0:
            raise ValueError(
                f"chunk_tokens({chunk_tokens}) 须为 block_size({block_size}) "
                "的正整数倍"
            )
        reference = layer_view(0)
        self._dtype = reference.dtype
        self._device = reference.device
        if reference.dim() == 4 and reference.shape[2] == 2:
            # 跨层交织池的逐层切片 [nb, bs, 2, kv]
            self._mode = self._INTERLEAVED
            self._kv_channels = int(reference.shape[3])
            self._staging_shape = (chunk_tokens, 2, self._kv_channels)
            if reference.shape[1] != block_size:
                raise ValueError(
                    f"交织池块维 {reference.shape[1]} 与 block_size"
                    f"({block_size}) 不符"
                )
        else:
            self._mode = self._KERNEL
            if (
                reference.device.type == "cuda"
                and staging.device.type == "cuda"
                and reference.device != staging.device
            ):
                raise ValueError(
                    "搬运函数路径要求 staging 与池同卡（跨卡池当前仅"
                    "支持交织布局的索引拷贝路径；主存 staging 不受限）"
                )
            row_bytes = segment_bytes // chunk_tokens
            element = reference.element_size()
            if segment_bytes % chunk_tokens or row_bytes % element:
                raise ValueError("层段字节数不能按 token 行整分")
            if fmt == EngineKVFormat.MLA:
                self._staging_shape = (chunk_tokens, row_bytes // element)
            else:
                if row_bytes % (2 * element):
                    raise ValueError("token 行字节不能按 K/V 两份整分")
                self._staging_shape = (chunk_tokens, 2, row_bytes // (2 * element))
        self._slot_cache: dict[tuple[int, ...], torch.Tensor] = {}

    def gather(self, keys, layer_idx: int, block_tables, slots):
        """源侧搬运：paged 池一层段 → staging 槽（写入批发起前调用）。"""
        self._transfer(block_tables, slots, layer_idx, "to_staging")
        if self._device.type != "cuda":
            return None
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device=self._device))
        return event

    def scatter(self, keys, layer_idx: int, block_tables, slots):
        """目的侧搬运并返回消费完成事件（无需事件时返回 None）。"""
        self._transfer(block_tables, slots, layer_idx, "to_paged")
        if self._device.type != "cuda":
            return None
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device=self._device))
        return event

    # ---- 内部 ----

    def _transfer(self, block_tables, slots, layer_idx: int,
                  direction: str) -> None:
        """搬运一层；连续槽位的多个 chunk 合并为一次 kernel。"""
        paged = self._layer_view(layer_idx)
        if (self._mode == self._KERNEL
                and len(block_tables) == len(slots)
                and self._can_batch_slots(slots)):
            mappings = [self._slot_mapping(blocks) for blocks in block_tables]
            if mappings:
                slot_mapping = torch.cat(mappings, dim=0)
                first = slots[0]
                staging_view = (
                    self._staging.narrow(
                        0, first * self._segment_bytes,
                        len(slots) * self._segment_bytes,
                    )
                    .view(self._dtype)
                    .view(len(slots) * self._chunk_tokens, *self._staging_shape[1:])
                )
                batched_layer_transfer(
                    staging_view, paged, slot_mapping, self._fmt, direction
                )
                return
        for blocks, slot in zip(block_tables, slots):
            staging_view = (
                self._staging
                .narrow(0, slot * self._segment_bytes, self._segment_bytes)
                .view(self._dtype)
                .view(*self._staging_shape)
            )
            slot_mapping = self._slot_mapping(blocks)
            if self._mode == self._INTERLEAVED:
                self._transfer_interleaved(staging_view, paged, slot_mapping,
                                           direction)
            else:
                single_layer_transfer(staging_view, paged, slot_mapping,
                                      self._fmt, direction)

    def _can_batch_slots(self, slots) -> bool:
        """Return true when slot segments form one contiguous staging view."""
        if not slots:
            return False
        return all(next_slot == slot + 1
                   for slot, next_slot in zip(slots, slots[1:]))
    def _transfer_interleaved(self, staging_view, paged, slot_mapping,
                              direction: str) -> None:
        """交织池搬运：块表索引直达（池段与槽段字节同构）。"""
        block_ids = torch.div(slot_mapping, self._block_size, rounding_mode="floor")
        offsets = slot_mapping % self._block_size
        if direction == "to_staging":
            staging_view.copy_(paged[block_ids, offsets])
        else:
            paged.index_put_(
                (block_ids, offsets), staging_view.to(paged.device)
            )

    def _slot_mapping(self, blocks) -> torch.Tensor:
        """块表 → 逐 token 平铺槽号（块号 × 块大小 + 块内偏移）。"""
        key = tuple(blocks)
        cached = self._slot_cache.get(key)
        if cached is None:
            ids = torch.tensor(key, dtype=torch.int64, device=self._device)
            offsets = torch.arange(
                self._block_size, dtype=torch.int64, device=self._device
            )
            cached = (ids[:, None] * self._block_size
                      + offsets[None, :]).reshape(-1)
            if len(self._slot_cache) >= _SLOT_CACHE_LIMIT:
                self._slot_cache.clear()
            self._slot_cache[key] = cached
        return cached


class WorkerImpl:
    """worker 角色的执行编排器。

    契约：
    - 构造注入引擎与在途句柄上限（max_in_flight_layers，0 = 不限）。
    - 显存对象经两个登记回调进入（单块跨层池或逐层映射，二选一）；
      首个执行回调触发惰性绑定（分配 staging 环窗显存并接入引擎）。
    - 读取流程：start_load_kv 组批并预取前 K 层；wait_for_layer_load
      等待当前层后补交第 K 层；末层等待返回即释放读保护。K=1
      时退化为原有逐层流水。
    - 写入流程：save_kv_layer 逐层发起一批；wait_for_save 等待全部
      完成并结算写入（在途句柄超限时先行等待最旧一批）。
    - 近似视图未遂（读取保护失败）的块经 get_block_ids_with_load_errors
      上报，由上层重算兜底。
    """

    def __init__(self, engine, max_in_flight_layers: int | None = None,
                 lookahead_k: int | None = None):
        """构造 worker；lookahead_k 为有界读取预取层数。"""
        if max_in_flight_layers is None:
            max_in_flight_layers = int(
                getattr(engine, "max_in_flight_operations", 0) or 0
            )
        if not isinstance(max_in_flight_layers, int) or max_in_flight_layers < 0:
            raise ValueError(
                f"max_in_flight_layers 须为非负整数，got {max_in_flight_layers!r}"
            )
        self._engine = engine
        self._max_in_flight = max_in_flight_layers
        if lookahead_k is None:
            engine_config = getattr(engine, "_config", {})
            lookahead_k = engine_config.get(
                "lookahead_k", engine_config.get("prefetch_k", 1)
            )
        if not isinstance(lookahead_k, int) or isinstance(lookahead_k, bool) \
                or lookahead_k <= 0:
            raise ValueError(
                f"lookahead_k 须为正整数，got {lookahead_k!r}"
            )
        self._lookahead_k = lookahead_k
        # configure 注入
        self._chunk_tokens = 0
        self._chunk_kv_bytes = 0
        self._max_chunks_per_wave = 0
        self._block_size = 16
        # 显存对象登记
        self._kv_caches: dict | None = None
        self._cross_pool = None
        self._layer_names: list[str] = []
        # 绑定态
        self._num_layers = 0
        self._bound = False
        self._window: RingWindow | None = None  # read-window compatibility alias
        self._read_window: RingWindow | None = None
        self._write_window: RingWindow | None = None
        # 本步计划与编排状态
        self._metadata = None
        self._load_keys: list[bytes] = []
        self._load_block_tables: list[list[int]] = []
        self._load_handles: dict[int, object] = {}
        self._pinned = False
        self._load_failed = False
        self._load_error_blocks: set[int] = set()
        # 无序号层名的学习映射（层名 → 层号，首次出现序；同一步内
        # wait/save 两次调用凭此解析一致）
        self._name_to_idx: dict[str, int] = {}
        self._save_keys: list[bytes] | None = None
        self._save_block_tables: list[list[int]] = []
        self._save_inflight: list = []

    # ---- 配置与登记 ----

    def configure(
        self,
        chunk_tokens: int,
        chunk_kv_bytes: int,
        max_chunks_per_wave: int,
        block_size: int,
        lookahead_k: int | None = None,
    ) -> None:
        """注入编排所需的几何参数（由挂载点从配置读出）。

        ``lookahead_k`` 可在配置阶段覆盖构造默认值，便于 connector
        统一透传策略参数；缺省保持构造时的值。
        """
        self._chunk_tokens = chunk_tokens
        self._chunk_kv_bytes = chunk_kv_bytes
        self._max_chunks_per_wave = max_chunks_per_wave
        if lookahead_k is not None:
            if (not isinstance(lookahead_k, int)
                    or isinstance(lookahead_k, bool) or lookahead_k <= 0):
                raise ValueError(
                    f"lookahead_k 须为正整数，got {lookahead_k!r}"
                )
            self._lookahead_k = lookahead_k
        self._block_size = block_size
        # 已有登记且尚未绑定 → 立即补绑（见 register_kv_caches 的时序说明）
        if (self._kv_caches or self._cross_pool is not None) and not self._bound:
            self._ensure_bound()

    def register_kv_caches(self, kv_caches: dict) -> None:
        """登记逐层显存对象映射（层名顺序即层序）。

        登记即绑定（幂等）：冷启动恢复（scan → restore）须在首个
        请求调度查询之前完成——惰性绑定会把恢复推迟到首请求执行期，
        调度器的命中查询先于执行，重启后的首请求会错过恢复窗口。
        """
        self._kv_caches = dict(kv_caches)
        self._layer_names = list(kv_caches.keys())
        if self._config_ready():
            self._ensure_bound()

    def register_cross_layers_kv_cache(self, kv_cache, attn_backend) -> None:
        """登记单块跨层显存对象（第 0 维为层数）。"""
        self._cross_pool = kv_cache
        if self._config_ready():
            self._ensure_bound()

    def _config_ready(self) -> bool:
        """编排几何是否已注入（未注入时保持惰性，configure 后补绑）。"""
        return (
            self._chunk_tokens is not None
            and self._chunk_kv_bytes is not None
            and self._max_chunks_per_wave is not None
            and self._block_size is not None
        )

    def set_metadata(self, metadata) -> None:
        """接收本步传输计划。"""
        self._metadata = metadata

    # ---- 读取编排 ----

    def start_load_kv(self, forward_context=None, **kwargs) -> None:
        """组读取批并预取第 0 层。

        加载区间 = [load_start_token, load_start_token + load_tokens)：
        vLLM 本地前缀命中的 token 已计入请求 computed，connector 只
        补其后区间——起点非 chunk 边界时首个不完整 chunk 让渡
        （重算兜底），key 与块表按同一 chunk 区间切片。
        """
        self._ensure_bound()
        self._finalize_load_state()
        self._load_failed = False
        self._load_keys = []
        self._load_block_tables = []
        self._load_handles = {}
        self._pinned = False
        keys: list[bytes] = []
        block_tables: list[list[int]] = []
        for meta in getattr(self._metadata, "requests", []) or []:
            if meta.load_tokens <= 0:
                continue
            first_chunk, n_chunks = _load_chunk_span(
                meta.load_start_token, meta.load_tokens, self._chunk_tokens
            )
            if n_chunks <= 0:
                continue
            req_keys, _ = self._engine.hash_keys(meta.token_ids)
            req_keys = req_keys[first_chunk : first_chunk + n_chunks]
            try:
                self._engine.pin(req_keys)
            except KeyError:
                # 近似视图未遂：该区间块上报重算
                self._report_load_errors(meta, first_chunk, n_chunks)
                continue
            keys.extend(req_keys)
            block_tables.extend(
                self._chunk_block_tables(meta, first_chunk + n_chunks)[first_chunk:]
            )
        if keys:
            self._load_keys = keys
        if keys:
            self._load_keys = keys
            self._load_block_tables = block_tables
            self._pinned = True
            with nvtx_range(
                f"tutti.request.load|requests="
                f"{len(getattr(self._metadata, 'requests', []) or [])}"
                f"|chunks={len(keys)}|K={self._lookahead_k}"
            ):
                try:
                    self._prefetch_load_layers(0)
                except Exception:
                    self._mark_load_failure()
                    self._abort_step()
                    raise

    def wait_for_layer_load(self, layer_name: str) -> None:
        """等待指定层读取完成，并预取下一层。"""
        idx = self._resolve_layer(layer_name)
        handle = self._load_handles.pop(idx, None)
        if handle is not None:
            try:
                with nvtx_range(f"tutti.request.wait_load|layer={idx}"):
                    handle.wait()
            except LoadGateError as exc:
                self._mark_load_failure(exc)
                self._abort_step()
                # vLLM's KV load recovery contract is invalid_block_ids, not a
                # connector exception. The current forward output is discarded
                # by the scheduler and either failed or recomputed per policy.
                return
            except Exception:
                self._abort_step()
                raise
        if self._load_failed:
            return
        # 维持固定的 K 层在途窗口：首次调用已提交 [0, K)，每次
        # 消费一层后只补交尾部一层。按 idx+K 定位可避免重复提交，
        # 也让乱序/重复回调保持幂等。
        nxt = idx + self._lookahead_k
        if (nxt < self._num_layers and self._load_keys
                and nxt not in self._load_handles):
            with nvtx_range(f"tutti.request.prefetch_load|layer={nxt}"):
                self._start_load_layer(nxt)
        if idx == self._num_layers - 1 and self._pinned:
            self._engine.unpin(self._load_keys)
            self._pinned = False

    # ---- 写入编排 ----

    def save_kv_layer(self, layer_name: str, kv_layer=None, attn_metadata=None, **kwargs) -> None:
        """发起指定层的写入批（本步首个写入层时组批）。

        组批时执行写入准入（plan_store）：为腾容量驱逐的 chunk 由
        engine 展开全层 io_key 实际删除（盘与权威索引）。数据面写
        全批（不按受理子集切片）；resident 只在所有层 save completion
        成功后发布，失败则回收 pending 预留。准入未被受理（容量不足且无可
        驱逐）时记录运行日志并跳过本批数据面——该路径正常部署
        不可达，出现即容量配置问题。
        """
        if self._load_failed:
            return
        self._ensure_bound()
        idx = self._resolve_layer(layer_name)
        if self._save_keys is None:
            keys: list[bytes] = []
            block_tables: list[list[int]] = []
            for meta in getattr(self._metadata, "requests", []) or []:
                if meta.save_chunk_count <= 0:
                    continue
                req_keys, _ = self._engine.hash_keys(meta.token_ids)
                start = meta.save_chunk_start
                end = start + meta.save_chunk_count
                keys.extend(req_keys[start:end])
                block_tables.extend(self._chunk_block_tables(meta, end)[start:end])
            if keys:
                plan = self._engine.plan_store(keys)
                if plan is None:
                    pending = getattr(self._engine, "store_plan_pending", None)
                    if not callable(pending) or not pending(keys):
                        _LOG.warning(
                            "写入计划未被受理（容量 %d 不足以容纳本批 %d "
                            "chunk 且无可驱逐驻留）——容量配置问题，跳过"
                            "本批数据面写入",
                            self._engine.capacity_chunks, len(keys),
                        )
                        keys = []
                        block_tables = []
            self._save_keys = keys
            self._save_block_tables = block_tables
            if not keys:
                return
        if not self._save_keys:
            return
        try:
            # The DataPath counts an operation as in-flight until progress/wait
            # observes its terminal event. Reap the oldest write before the
            # next submit reaches the runtime admission boundary. The write
            # stream itself preserves FIFO execution.
            if (self._max_in_flight and
                    len(self._save_inflight) >= self._max_in_flight):
                self._save_inflight.pop(0).wait()
            with nvtx_range(
                f"tutti.request.save|layer={idx}|chunks={len(self._save_keys)}"
            ):
                handle = self._engine.store_layer(
                    self._save_keys, idx, self._save_block_tables
                )
        except Exception:
            self._abort_step()
            raise
        self._save_inflight.append(handle)

    def wait_for_save(self) -> None:
        """等待全部写入完成并结算。"""
        first_error = None
        for handle in self._save_inflight:
            try:
                handle.wait()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        try:
            self._engine.wait_idle()
        except Exception as exc:
            if first_error is None:
                first_error = exc
        try:
            self._finalize_load_state()
        except Exception as exc:
            if first_error is None:
                first_error = exc
        if self._save_keys:
            self._engine.confirm_store(self._save_keys, ok=first_error is None)
        self._save_inflight = []
        self._save_keys = None
        self._save_block_tables = []
        if first_error is not None:
            raise first_error

    # ---- 状态与收尾 ----

    @property
    def window(self) -> RingWindow | None:
        """兼容属性：返回 read bank（其 buffer 仍是完整共享 staging）。"""
        return self._window

    @property
    def read_window(self) -> RingWindow | None:
        return self._read_window

    @property
    def write_window(self) -> RingWindow | None:
        return self._write_window

    def get_block_ids_with_load_errors(self) -> set[int]:
        """上报读取未遂的块集合（读取后清空）。"""
        errors = self._load_error_blocks
        self._load_error_blocks = set()
        return errors

    def shutdown(self) -> None:
        """等待在途并关闭引擎（幂等由引擎保证）。"""
        try:
            self._abort_step()
        finally:
            self._engine.close()

    def abort(self) -> None:
        """请求失败收尾：drain 双 bank、回滚写计划并释放读 pin。"""
        self._abort_step()

    # ---- 内部 ----

    def _start_load_layer(self, layer_idx: int) -> None:
        """发起一层的读取批。"""
        if self._load_failed:
            return
        self._load_handles[layer_idx] = self._engine.load_layer(
            self._load_keys, layer_idx, self._load_block_tables
        )

    def _prefetch_load_layers(self, first_idx: int) -> None:
        """提交从 ``first_idx`` 起的有限层窗口。

        staging 环窗本身对波次覆盖负责背压；这里仅限制层级在途数，
        因而不会创建无界 async bulk 队列。若环窗容量不足，底层
        ``acquire`` 会在复用前驱槽位时等待其完成，保持既有安全契约。
        """
        end = min(self._num_layers, first_idx + self._lookahead_k)
        for layer_idx in range(first_idx, end):
            if layer_idx not in self._load_handles:
                self._start_load_layer(layer_idx)

    def _finalize_load_state(self, suppress_errors: bool = False) -> None:
        """Drain remaining read callbacks and release any outstanding pin."""
        first_error = None
        handles = list(self._load_handles.values())
        self._load_handles = {}
        for handle in handles:
            try:
                handle.wait()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if self._pinned:
            try:
                self._engine.unpin(self._load_keys)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            self._pinned = False
        self._load_keys = []
        self._load_block_tables = []
        if first_error is not None and not suppress_errors:
            raise first_error

    def _abort_step(self) -> None:
        """Best-effort full-direction drain while preserving the first error."""
        try:
            self._engine.abort()
        except Exception:
            pass
        self._finalize_load_state(suppress_errors=True)
        if self._save_keys:
            try:
                self._engine.confirm_store(self._save_keys, ok=False)
            except Exception:
                pass
        self._save_inflight = []
        self._save_keys = None
        self._save_block_tables = []

    def _mark_load_failure(self, error=None) -> None:
        """Fail closed: all blocks in the gated load become invalid."""
        self._load_failed = True
        if isinstance(error, LoadGateError) and error.invalid_block_ids:
            self._load_error_blocks.update(error.invalid_block_ids)
            return
        selected = self._load_block_tables
        if (isinstance(error, LoadGateError) and not error.whole_operation
                and error.failed_batch_indices):
            selected = [
                self._load_block_tables[index]
                for index in error.failed_batch_indices
                if 0 <= index < len(self._load_block_tables)
            ]
        for blocks in selected:
            if isinstance(blocks, (list, tuple, set)):
                self._load_error_blocks.update(blocks)
            elif blocks is not None:
                self._load_error_blocks.add(blocks)

    def _chunk_block_tables(self, meta, count: int) -> list[list[int]]:
        """计算前 count 个 chunk 各自的块号表（块号可任意分布）。"""
        blocks = meta.block_ids
        per_chunk = -(-self._chunk_tokens // self._block_size)
        return [
            list(blocks[i * per_chunk:(i + 1) * per_chunk])
            for i in range(count)
        ]

    def _report_load_errors(self, meta, first_chunk: int, n_chunks: int) -> None:
        """把读取未遂区间覆盖的块上报为重算。"""
        per_chunk = -(-self._chunk_tokens // self._block_size)
        blocks = meta.block_ids
        start_blk = first_chunk * per_chunk
        end_blk = min((first_chunk + n_chunks) * per_chunk, len(blocks))
        if end_blk > start_blk:
            self._load_error_blocks.update(blocks[start_blk:end_blk])

    def _resolve_layer(self, layer_name: str) -> int:
        """把层名解析为层序号。

        两种登记模式统一按名解析：优先提取层名中的序号
        （layers.{i} 约定）；无序号时查学习映射（首次出现序），
        未见过则按已学习层数顺延登记。解析结果校验层号范围；
        登记模式下列表未收录的层名、任何越界结果均抛错——
        层号错位会静默写错层，宁可失败。
        """
        if self._kv_caches:
            try:
                return self._layer_names.index(layer_name)
            except ValueError:
                raise ValueError(
                    f"层名未在逐层登记中：{layer_name!r}"
                    f"（已登记 {len(self._layer_names)} 层）"
                )
        match = _LAYER_NAME_RE.search(layer_name)
        if match:
            idx = int(match.group(1))
        else:
            idx = self._name_to_idx.get(layer_name)
            if idx is None:
                idx = len(self._name_to_idx)
                self._name_to_idx[layer_name] = idx
        if not 0 <= idx < self._num_layers:
            raise ValueError(
                f"层名 {layer_name!r} 解析层号 {idx} 越界"
                f"（层数 {self._num_layers}）"
            )
        return idx

    def _ensure_bound(self) -> None:
        """惰性绑定：分配 staging 环窗显存并接入引擎（恰一次）。

        池为加速侧内存时构造两端搬运钩子并经 bind 注入（bind 参数
        优先于引擎配置中的同名钩子；逐层常见布局走布局发现 + 搬运
        函数，跨层交织布局走块表索引拷贝）；池为主存或布局无法识别
        时不注入，保持配置钩子或缺省 no-op 行为。
        """
        if self._bound:
            return
        if self._cross_pool is not None:
            if self._cross_pool.dim() < 2:
                raise ValueError(
                    "跨层池至少 2 维（[块数, 层数, ...]），got shape="
                    f"{tuple(self._cross_pool.shape)}"
                )
            # 跨层池布局为 [块数, 层数, 块大小, K/V, 通道]（CROSS_LAYER
            # 模式实测；层数在第 1 维，块数在第 0 维）。
            num_layers = int(self._cross_pool.shape[1])
        elif self._kv_caches:
            num_layers = len(self._kv_caches)
        else:
            raise RuntimeError("尚未登记任何显存对象，无法绑定")
        if num_layers <= 0:
            raise RuntimeError("登记的层数为空，无法绑定")
        if self._chunk_kv_bytes % num_layers != 0:
            raise ValueError(
                f"chunk_kv_bytes({self._chunk_kv_bytes}) 不能按层数"
                f"({num_layers}) 整分"
            )
        segment_bytes = self._chunk_kv_bytes // num_layers
        # 原 staging 总槽数不变，但静态拆为 read/write 两个 bank。
        # 每个 bank 各容纳 K 个满波；K=1 时同方向下一波在复用前
        # 等待，K>1 保留有界 lookahead，不扩展为全层预排。
        window_layers = min(self._lookahead_k, num_layers)
        bank_slots = self._max_chunks_per_wave * window_layers
        slots = 2 * bank_slots
        # Staging must be local to the registered KV pool. The deployment
        # preset binds the matching rank-local DataPath/IO streams to this
        # accelerator; cross-device staging would add P2P traffic and can
        # exhaust one GPU's queue group during TP initialization.
        if self._cross_pool is not None:
            pool_device = self._cross_pool.device
        elif self._kv_caches:
            pool_device = getattr(
                next(iter(self._kv_caches.values())), "device", torch.device("cpu")
            )
        else:
            pool_device = torch.device("cpu")
        if pool_device.type == "cuda":
            # 64 KiB 对齐分配：真机注册 DEVICE 内存要求基址按 snvme
            # GPU 页粒度（64 KiB）对齐，torch 分配器不保证——超量
            # 分配后取对齐切片（共享存储，无拷贝）。
            slack = 65536
            block = torch.empty(
                slots * segment_bytes + slack, dtype=torch.uint8,
                device=pool_device,
            )
            base = block.data_ptr()
            skip = (slack - base % slack) % slack
            staging = block[skip : skip + slots * segment_bytes]
            raw = staging
        else:
            staging = torch.empty(
                slots * segment_bytes, dtype=torch.uint8, pin_memory=True
            )
            raw = staging.numpy()
        read_window = RingWindow(
            raw, bank_slots, segment_bytes,
            capacity_per_wave=self._max_chunks_per_wave,
            slot_base=0,
        )
        write_window = RingWindow(
            raw, bank_slots, segment_bytes,
            capacity_per_wave=self._max_chunks_per_wave,
            slot_base=bank_slots,
        )
        blocks_per_chunk = -(-self._chunk_tokens // self._block_size)
        gather_fn = scatter_fn = None
        layer_view = self._layer_view()
        if layer_view is not None and getattr(layer_view(0), "is_cuda", False):
            reference = layer_view(0)
            interleaved = reference.dim() == 4 and reference.shape[2] == 2
            fmt = (
                None if interleaved else discover_engine_format(
                    reference, use_mla=reference.dim() == 3
                )
            )
            hooks = PagedTransferHooks(
                layer_view, staging, segment_bytes,
                self._chunk_tokens, self._block_size, fmt,
            )
            gather_fn, scatter_fn = hooks.gather, hooks.scatter
        bind_caches = self._kv_caches if self._kv_caches else (
            self._cross_pool if self._cross_pool is not None else {}
        )
        self._engine.bind(
            bind_caches,
            read_window, num_layers, blocks_per_chunk,
            write_window=write_window,
            gather_fn=gather_fn, scatter_fn=scatter_fn,
        )
        self._num_layers = num_layers
        self._window = read_window
        self._read_window = read_window
        self._write_window = write_window
        self._bound = True

    def _layer_view(self):
        """返回层序号 → paged 张量的访问器；未登记任何池时为 None。

        跨层池布局为 [块数, 层数, ...]（CROSS_LAYER 模式），逐层切片
        取第 1 维。
        """
        if self._cross_pool is not None:
            pool = self._cross_pool
            return lambda idx: pool[:, idx]
        if self._kv_caches:
            names = self._layer_names
            caches = self._kv_caches
            return lambda idx: caches[names[idx]]
        return None

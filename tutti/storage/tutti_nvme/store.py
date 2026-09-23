"""TuttiKVStore——tutti IO runtime 之上的层盲 KV 存储。
store 对 io_key 是纯映射（私有解读见 layout.py），数据面全部经由
runtime.submit 的 DMA 请求表达：put = memory→target（write），
get = target→memory（read）。**批是一等公民**：一次 put/get 对应
一批请求、一个 Completion；partial-commit 批次在内部窗口重发
（连续两轮零接受视为失败上抛）。

runtime 注入（测试/fake）或由 TUTTI_NVME_PRESET 构造（真机）：
环境变量值为 yaml/json 内联文本或文件路径；preset 可携带
daemon_config + device_id，设备事实（pci_bdf/mount_path/namespace_id）
从 daemon 配置推导——硬件信息单一来源，preset 只写用户级参数。
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from .object_layout import ObjectLayout
from .preset_derive import derive_device_fields
from .runtime_factory import (
    build_runtime,
    build_runtime_from_env,
    normalize_preset,
    preset_mounts,
)

from tutti.common.errors import DirectTransferUnavailable
from tutti.common.nvtx import range as nvtx_range
from tutti.index.chunk_index import decode_io_key, derive_io_key

#: 运行日志（池归属校验等部署问题的非静默说明）。
_LOG = logging.getLogger(__name__)

# runtime 在"这一批太宽"时给出的 RESOURCE_EXHAUSTED 文案（见 datapath 的
# max_batch_entries / max_batch_bytes 检查）。这类拒绝可以用拆批解决，不该
# 升级成 worker 崩溃：一批多宽由本步要搬多少 chunk 决定（长 prompt + 高并发
# 时实测 9248 条，当时上限 8192；上限已提到 16384 = 1M token/rank 的单层
# 理论最大宽度），不是调用方能先验保证的。
_BATCH_WIDTH_REJECTION_MARKERS = (
    "too many sub-IOs",
    "batch bytes exceed limit",
)


def _is_batch_width_rejection(status_msg: str | None) -> bool:
    text = status_msg or ""
    return any(marker in text for marker in _BATCH_WIDTH_REJECTION_MARKERS)

#: KV IO 页基：register_buffer 粒度必须为其正倍数（对齐 NVMe/DMA 路径）。
_IO_PAGE_BYTES = 4096

#: 后台完成观察者每次等待的上限。runtime 本身以条件变量唤醒；
#: 有上限是为了在 runtime shutdown/异常实现下也能及时退出。
_COMPLETION_WAIT_MS = 1000


def _buffer_info(buffer) -> tuple[int, int, str, int] | None:
    """提取 (地址, 字节数, memory 类型, accel_id)；无法定位稳定地址 → None。

    支持：torch tensor（data_ptr；CUDA → device）、bytearray、ctypes 数组。
    """
    data_ptr = getattr(buffer, "data_ptr", None)
    if callable(data_ptr):
        try:
            addr = int(data_ptr())
            if hasattr(buffer, "numel") and hasattr(buffer, "element_size"):
                size = int(buffer.numel()) * int(buffer.element_size())
            else:
                size = int(getattr(buffer, "nbytes", 0))
        except (TypeError, ValueError, RuntimeError):
            return None
        if addr == 0 or size <= 0:
            return None
        if getattr(buffer, "is_cuda", False):
            return addr, size, "device", int(buffer.get_device())
        return addr, size, "host", -1
    if isinstance(buffer, bytearray):
        if not buffer:
            return None
        keeper = (ctypes.c_char * len(buffer)).from_buffer(buffer)
        return ctypes.addressof(keeper), len(buffer), "host", -1
    try:
        addr = ctypes.addressof(buffer)
        size = ctypes.sizeof(buffer)
    except (TypeError, ValueError):
        return None
    if addr == 0 or size <= 0:
        return None
    return addr, size, "host", -1


@dataclass(frozen=True)
class _SubmittedHandle:
    handle: int
    batch_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _TargetCacheEntry:
    ticket: int
    size: int
    generation: int


@dataclass(frozen=True)
class DirectTargetPlanEntry:
    chunk_id: bytes
    target_ticket: int
    target_uri: str
    target_size: int
    target_generation: int
    # 段 0 在对象逻辑地址空间中的起点（对象头之后）。层内偏移在此之上叠加。
    target_offset: int = 0


@dataclass(frozen=True)
class DirectTargetPlan:
    direction: str
    memory_ticket: int
    plan_token: int
    entries: tuple[DirectTargetPlanEntry, ...]
    _entry_by_chunk: dict[bytes, DirectTargetPlanEntry] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        entry_by_chunk = {entry.chunk_id: entry for entry in self.entries}
        if len(entry_by_chunk) != len(self.entries):
            raise ValueError("direct target plan contains duplicate chunks")
        object.__setattr__(self, "_entry_by_chunk", entry_by_chunk)

    def entry_for(self, chunk_id: bytes) -> DirectTargetPlanEntry:
        chunk_id = bytes(chunk_id)
        try:
            return self._entry_by_chunk[chunk_id]
        except KeyError as exc:
            raise RuntimeError(
                f"direct {self.direction} target plan lacks chunk {chunk_id!r}"
            ) from exc


@dataclass(frozen=True)
class TuttiTerminalResult:
    handle: int
    observation: str
    state: str
    confirmed_bytes: int = 0
    timeout_seen: bool = False
    failed_request_indices: tuple[int, ...] = ()
    failure_scope: str = "NONE"
    failure_kind: str | None = None
    first_failed_entry: int | None = None
    raw_cq_status: int | None = None
    message: str = ""
    batch_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class TuttiBatchResult:
    ok: bool
    failed_batch_indices: tuple[int, ...]
    timeout_seen: bool
    failures: tuple[TuttiTerminalResult, ...]
    results: tuple[TuttiTerminalResult, ...]


class DirectAdmissionError(DirectTransferUnavailable):
    """A concrete reason why the cross-layer pool cannot use direct I/O."""


@dataclass(frozen=True)
class DirectPoolGeometry:
    pool_base: int
    pool_size: int
    block_stride_bytes: int
    layer_stride_bytes: int
    page_bytes: int
    num_blocks: int
    num_layers: int
    block_size: int
    blocks_per_chunk: int
    segment_bytes: int
    accel_id: int


class TuttiDirectBackend:
    """Direct byte-range I/O against one uniform cross-layer CUDA pool."""

    def __init__(self, store):
        self._store = store
        self._pool = None
        self._memory_ticket = None
        self._geometry: DirectPoolGeometry | None = None
        self._closed = False
        self._target_plans: dict[str, DirectTargetPlan] = {}
        self._prepared_write_request: tuple[bytes, ...] | None = None
        self._prepared_write_chunks: tuple[bytes, ...] | None = None
        # 绑定期预开的槽位 URI（registration 预热的票据来源）。
        self._warm_uris: list[str] = []
        self._invalid_plan_tokens: set[int] = set()
        self._next_plan_token = 0
        # Target-cache invalidation is the generation-change signal used by
        # layer submit. It avoids filesystem/object-pool queries on that path.
        self._store._direct_backend = self

    @property
    def geometry(self) -> DirectPoolGeometry:
        if self._geometry is None:
            raise RuntimeError("direct backend is not registered")
        return self._geometry

    def register_paged_caches(
        self,
        pool,
        *,
        num_layers: int,
        blocks_per_chunk: int,
        chunk_tokens: int,
        segment_bytes: int,
        max_chunks_per_wave: int | None = None,
    ) -> bool:
        if self._memory_ticket is not None:
            raise DirectAdmissionError("KV pool is already registered")
        geometry = self._derive_geometry(
            pool,
            num_layers=num_layers,
            blocks_per_chunk=blocks_per_chunk,
            chunk_tokens=chunk_tokens,
            segment_bytes=segment_bytes,
        )
        self._check_runtime_capacity(
            geometry, max_chunks_per_wave=max_chunks_per_wave
        )
        try:
            ticket = self._store._runtime.register_memory(
                geometry.pool_base,
                geometry.pool_size,
                "device",
                accel_id=geometry.accel_id,
                io_granularity=0,
            )
        except Exception as exc:
            raise DirectAdmissionError(
                f"Runtime.register_memory(io_granularity=0) failed: {exc}"
            ) from exc
        self._pool = pool
        self._memory_ticket = int(ticket)
        self._geometry = geometry
        _LOG.warning(
            "DIRECT_ADMISSION_ACCEPTED accel=%d pool_base=0x%x "
            "pool_size=%d num_blocks=%d num_layers=%d block_size=%d "
            "page_bytes=%d block_stride_bytes=%d layer_stride_bytes=%d "
            "blocks_per_chunk=%d segment_bytes=%d read_stream=%s "
            "write_stream=%s io_granularity=0",
            geometry.accel_id, geometry.pool_base, geometry.pool_size,
            geometry.num_blocks, geometry.num_layers, geometry.block_size,
            geometry.page_bytes, geometry.block_stride_bytes,
            geometry.layer_stride_bytes, geometry.blocks_per_chunk,
            geometry.segment_bytes, self._store._read_stream,
            self._store._write_stream,
        )
        return True

    def warm_up_registration(self) -> bool:
        """绑定期强制完成每个 DataPath 的 peer-memory 注册（一次性）。

        注册是惰性的：``StorageRuntime::submit`` 首次遇到一个未注册的
        (data_path, registration_domain) 才会把 KV 池显存映射成该 NVMe
        设备的 peer 内存（``nvm_dma_map_data_device``，走 NVIDIA 驱动 +
        phxfs 的 p2p 注册，对 48.7GB 的池逐页建表）。实测单次 200~275ms，
        且全程持 runtime registry 锁——8 卡首轮 476ms 期间所有前向线程的
        ``submit`` 都被阻塞、GPU 全空转（nsys: ioctl 275+201ms）。

        在 bind 期（KV 池已注册、槽位句柄已就绪）用一个 1-page **读**
        把它做掉，代价是启动多 ~0.5s、首个请求少 ~0.5s。读方向不写盘，
        失败也不影响正确性，故这里吞掉异常并返回 False。
        """
        store = self._store
        geometry = self._geometry
        if self._closed or geometry is None or self._memory_ticket is None:
            return False
        # 用绑定期预开的第一个槽位票据：open_batch 已完成 resolve +
        # peer-memory 注册，正是要挪出请求路径的那部分。冷池同样有票据
        # （预热槽位在 open 时已物化），所以这一路径不依赖已有驻留。
        target_ticket = 0
        for uri in getattr(store, "_warm_uris", ()) or ():
            entry = store._targets.get(uri)
            if entry is not None and entry.ticket:
                target_ticket = int(entry.ticket)
                break
        if not target_ticket:
            return False
        length = int(geometry.page_bytes)
        if not 0 < length <= int(geometry.segment_bytes):
            return False
        started_ns = time.perf_counter_ns()
        try:
            with nvtx_range("tutti.direct.warm_up_registration"):
                handles = store._submit_retry(
                    [(target_ticket, 0, int(self._memory_ticket), 0,
                      length, "read")],
                    "read",
                )
                completion = _TuttiCompletion(
                    store._runtime, handles, lambda _ok: None,
                    auto_watch=False, direction="read",
                )
                completion.wait_result()
        except Exception as exc:
            _LOG.warning("DIRECT_REGISTRATION_WARMUP_FAILED err=%r", exc)
            return False
        _LOG.warning(
            "DIRECT_REGISTRATION_WARMUP_DONE elapsed_ms=%.1f",
            (time.perf_counter_ns() - started_ns) / 1_000_000,
        )
        return True

    @staticmethod
    def _derive_geometry(pool, *, num_layers, blocks_per_chunk,
                         chunk_tokens, segment_bytes) -> DirectPoolGeometry:
        if not bool(getattr(pool, "is_cuda", False)):
            raise DirectAdmissionError("KV pool must be a CUDA device tensor")
        dim = getattr(pool, "dim", None)
        if not callable(dim) or int(dim()) != 5:
            raise DirectAdmissionError(
                "KV pool must have rank 5: "
                "[num_blocks, num_layers, block_size, kv_heads, kv_channels]"
            )
        shape = tuple(int(value) for value in pool.shape)
        if any(value <= 0 for value in shape):
            raise DirectAdmissionError(f"KV pool has invalid shape {shape}")
        if shape[1] != int(num_layers):
            raise DirectAdmissionError(
                f"KV pool layer axis is {shape[1]}, expected {num_layers}"
            )
        # shape[3] 是**每 rank 的 KV head 数**（TP 分片后），不是 K/V 轴：
        # 实测 TP4 [nb, 80, 64, 2, 256]、TP8 [nb, 80, 64, 1, 256]——K 与 V
        # 拼接在最后一维（2 × head_dim），head 数随 TP 变化。IO 按整页
        # （block × 层）搬运，不区分 K/V，故只校验其非零。
        if shape[3] < 1:
            raise DirectAdmissionError(
                f"KV pool head axis must be >= 1, got {shape[3]}"
            )
        if shape[2] != int(chunk_tokens) // int(blocks_per_chunk):
            raise DirectAdmissionError(
                f"KV pool block axis is {shape[2]}, inconsistent with "
                f"chunk_tokens={chunk_tokens} and "
                f"blocks_per_chunk={blocks_per_chunk}"
            )
        if int(chunk_tokens) % shape[2]:
            raise DirectAdmissionError(
                f"chunk_tokens({chunk_tokens}) is not divisible by "
                f"block_size({shape[2]})"
            )
        contiguous = getattr(pool, "is_contiguous", None)
        if not callable(contiguous) or not bool(contiguous()):
            raise DirectAdmissionError(
                "KV pool backing storage is not contiguous (padded/strided "
                "layouts are unsupported)"
            )
        stride = tuple(int(value) for value in pool.stride())
        expected_stride = (
            shape[1] * shape[2] * shape[3] * shape[4],
            shape[2] * shape[3] * shape[4],
            shape[3] * shape[4],
            shape[4],
            1,
        )
        if stride != expected_stride:
            raise DirectAdmissionError(
                f"KV pool stride {stride} is not uniform NHD cross-layer "
                f"stride {expected_stride}"
            )
        element_size = int(pool.element_size())
        pool_base = int(pool.data_ptr())
        pool_size = int(pool.numel()) * element_size
        block_stride_bytes = stride[0] * element_size
        layer_stride_bytes = stride[1] * element_size
        page_bytes = shape[2] * shape[3] * shape[4] * element_size
        derived_blocks = int(chunk_tokens) // shape[2]
        if derived_blocks != int(blocks_per_chunk):
            raise DirectAdmissionError(
                f"blocks_per_chunk mismatch: {blocks_per_chunk} != "
                f"{derived_blocks}"
            )
        if layer_stride_bytes != page_bytes:
            raise DirectAdmissionError(
                f"one layer page is not contiguous: layer_stride_bytes="
                f"{layer_stride_bytes}, page_bytes={page_bytes}"
            )
        if int(segment_bytes) != derived_blocks * page_bytes:
            raise DirectAdmissionError(
                f"padded or mismatched page geometry: segment_bytes="
                f"{segment_bytes}, expected {derived_blocks * page_bytes}"
            )
        if pool_base % 65536:
            raise DirectAdmissionError(
                f"KV pool base 0x{pool_base:x} is not 64 KiB aligned"
            )
        for name, value in (
            ("pool_size", pool_size),
            ("block_stride_bytes", block_stride_bytes),
            ("layer_stride_bytes", layer_stride_bytes),
            ("page_bytes", page_bytes),
            ("segment_bytes", int(segment_bytes)),
        ):
            if value % _IO_PAGE_BYTES:
                raise DirectAdmissionError(
                    f"{name}={value} is not 4 KiB I/O aligned"
                )
        get_device = getattr(pool, "get_device", None)
        if not callable(get_device):
            raise DirectAdmissionError("KV pool does not expose a CUDA device")
        return DirectPoolGeometry(
            pool_base=pool_base,
            pool_size=pool_size,
            block_stride_bytes=block_stride_bytes,
            layer_stride_bytes=layer_stride_bytes,
            page_bytes=page_bytes,
            num_blocks=shape[0],
            num_layers=shape[1],
            block_size=shape[2],
            blocks_per_chunk=derived_blocks,
            segment_bytes=int(segment_bytes),
            accel_id=int(get_device()),
        )

    def _check_runtime_capacity(self, geometry, *, max_chunks_per_wave) -> None:
        store = self._store
        if not callable(getattr(store._runtime, "unregister_memory", None)):
            raise DirectAdmissionError(
                "Runtime does not expose unregister_memory for direct KV pool "
                "lifecycle"
            )
        if not store._runtime_supports_multi_stream():
            raise DirectAdmissionError(
                "Runtime does not support independent read/write streams"
            )
        if (store._read_stream is None or store._write_stream is None
                or store._read_stream == store._write_stream):
            raise DirectAdmissionError(
                "read/write CUDA stream handles are not independent"
            )
        stream_accel = getattr(store, "_stream_accel_id", None)
        if stream_accel is not None and int(stream_accel) != geometry.accel_id:
            raise DirectAdmissionError(
                f"KV pool CUDA device {geometry.accel_id} does not match "
                f"Runtime stream device {stream_accel}"
            )
        try:
            caps = dict(store._runtime.caps())
        except Exception as exc:
            raise DirectAdmissionError(f"Runtime capability query failed: {exc}") from exc
        in_flight = int(caps.get("max_in_flight_operations", 0) or 0)
        required_ops = 2 * geometry.num_layers
        if in_flight and in_flight < required_ops:
            raise DirectAdmissionError(
                "direct operation capacity is insufficient: "
                f"configured={in_flight}, required={required_ops}, "
                f"num_layers={geometry.num_layers}"
            )
        max_batch = int(
            caps.get("max_batch_entries", caps.get("max_batch_requests", 0)) or 0
        )
        required_batch = geometry.blocks_per_chunk * int(
            max_chunks_per_wave or 1
        )
        if max_batch and max_batch < required_batch:
            raise DirectAdmissionError(
                f"max_batch_entries={max_batch} is below one-layer request "
                f"bound {required_batch}"
            )

    def get_paged_batch(self, keys, layer_idx: int, block_tables):
        return self._submit(keys, layer_idx, block_tables, "read")

    def put_paged_batch(self, keys, layer_idx: int, block_tables):
        return self._submit(keys, layer_idx, block_tables, "write")

    @staticmethod
    def _chunk_ids(keys) -> tuple[bytes, ...]:
        return tuple(dict.fromkeys(
            decode_io_key(derive_io_key(bytes(key), 0))[0] for key in keys
        ))

    def prepare_write_targets(self, keys) -> tuple[bytes, ...]:
        """Allocate every write target before Runtime tickets are opened.

        返回本批**容量受理**的 chunk（顺序同入参、已去重）。对象层契约是
        "部分受理、绝不阻塞"：容量耗尽时未受理的 chunk 不进写计划（见
        begin_target_plan 的注释），**本步不写**，调用方必须据此裁剪本批；
        否则 _submit 会以 "plan lacks chunk" fail-closed 并打死 worker
        （2026-09-20 在线驱逐压测实证：12 请求 × 79 chunk 远超容量时必然
        触发，EngineDeadError 整个 server 一起死）。
        """
        chunk_ids = self._chunk_ids(keys)
        active = self._target_plans.get("write")
        if active is not None:
            planned = tuple(entry.chunk_id for entry in active.entries)
            if planned != chunk_ids:
                raise RuntimeError(
                    "direct write target preparation changed within one step"
                )
            return self._prepared_write_chunks
        if self._prepared_write_request is not None:
            if self._prepared_write_request != chunk_ids:
                raise RuntimeError(
                    "direct write target preparation changed within one step"
                )
            return self._prepared_write_chunks
        # The highest layer is sufficient to request the complete configured
        # object: reservation materialises the slot and fixes its generation
        # before begin_target_plan calls open_batch.
        last_layer = self.geometry.num_layers - 1
        io_keys = [derive_io_key(chunk_id, last_layer)
                   for chunk_id in chunk_ids]
        # 槽位分配耗时（判据：动态分配是否落进前向计算的关键路径）。
        # 异步预建生效时 reserve 只做内存分配：命中就绪槽位，未命中即拒绝
        # （不建文件、不阻塞）；只有预建未启动才退化成现场 create+写实零+
        # fsync（~44ms/槽），那一刻 alloc_ms 会量级抬升并阻塞前向线程。
        alloc_t0 = time.perf_counter()
        admitted, rejected = self._store._layout.prepare_put(
            io_keys, self._store._num_chunks
        )
        alloc_ms = (time.perf_counter() - alloc_t0) * 1e3
        self._log_slot_alloc(
            len(chunk_ids), len(admitted), rejected, alloc_ms
        )
        if rejected:
            _LOG.warning(
                "DIRECT_WRITE_ADMISSION_SHORTFALL requested=%d admitted=%d "
                "rejected=%d capacity=%d alloc_ms=%.3f",
                len(chunk_ids), len(admitted), rejected,
                self._store._num_chunks, alloc_ms,
            )
        self._prepared_write_request = chunk_ids
        self._prepared_write_chunks = tuple(
            chunk_id for chunk_id in chunk_ids
            if derive_io_key(chunk_id, last_layer) in admitted
        )
        return self._prepared_write_chunks

    def _log_slot_alloc(self, requested: int, admitted: int, rejected: int,
                        alloc_ms: float) -> None:
        """动态槽位分配的时间线（每次写批前移一条；纯观测，失败静默）。

        ``alloc_ms`` 是对象层 reserve 的墙钟耗时，``rejected`` 是"就绪槽位
        不足被裁剪"的数量。``precreated/target`` 是后台预建的就绪水位与
        目标：两者相近说明分配吃的是预建余量（快路径，纯内存）；rejected
        上升说明余量跟不上分配前沿——此时写入被裁剪而**不阻塞计算**，是
        可接受的降级，但需要在日志里与"现场建槽的慢路径"区分开。
        """
        precreated = target = -1
        try:
            native = self._store._layout._store
            precreated = int(native.precreated_slots())
            target = int(native.precreate_target())
        except Exception:
            pass
        _LOG.log(
            logging.WARNING if rejected else logging.INFO,
            "DIRECT_SLOT_ALLOC requested=%d admitted=%d rejected=%d "
            "alloc_ms=%.3f precreated=%d target=%d",
            requested, admitted, rejected, alloc_ms, precreated, target,
        )

    def begin_target_plan(self, keys, direction: str) -> DirectTargetPlan:
        if direction not in ("read", "write"):
            raise ValueError(f"invalid direct target plan direction: {direction}")
        if direction in self._target_plans:
            raise RuntimeError(f"direct {direction} target plan already exists")
        if self._memory_ticket is None:
            raise RuntimeError("direct target plan requires registered memory")
        chunk_ids = self._chunk_ids(keys)
        if direction == "write":
            if self._prepared_write_request != chunk_ids:
                raise RuntimeError(
                    "direct write target plan requires prepare_write_targets"
                )
            # Unadmitted chunks (capacity exhausted) are not written this step:
            # the write plan simply does not contain them.
            chunk_ids = self._prepared_write_chunks
        entries = [(chunk_id + (0).to_bytes(2, "little"), 0, 0)
                   for chunk_id in chunk_ids]
        with nvtx_range(
            f"tutti.direct.target_plan_build|direction={direction}"
        ):
            target_tickets = self._store._ensure_targets(entries)
        plan_entries = []
        for chunk_id in chunk_ids:
            uri = self._store._layout.target_uri(chunk_id)
            cached = getattr(self._store, "_targets", {}).get(uri)
            target_size = getattr(cached, "size", None)
            target_generation = getattr(cached, "generation", None)
            if target_size is None:
                size = getattr(self._store._layout, "target_size", None)
                target_size = int(size(chunk_id)) if callable(size) else 0
            if target_generation is None:
                generation = getattr(
                    self._store._layout, "target_generation", None
                )
                target_generation = (
                    int(generation(chunk_id)) if callable(generation) else 0
                )
            plan_entries.append(DirectTargetPlanEntry(
                chunk_id=chunk_id,
                target_ticket=int(target_tickets[uri]),
                target_uri=uri,
                target_size=int(target_size),
                target_generation=int(target_generation),
                target_offset=int(self._store._layout.target_offset(chunk_id)),
            ))
        self._next_plan_token += 1
        plan = DirectTargetPlan(
            direction=direction,
            memory_ticket=int(self._memory_ticket),
            plan_token=self._next_plan_token,
            entries=tuple(plan_entries),
        )
        self._target_plans[direction] = plan
        # 正常路径的时间线诊断：debug（每个方向每次构建一条）。
        _LOG.debug(
            "DIRECT_TARGET_PLAN_BUILD direction=%s chunks=%d token=%d",
            direction, len(plan.entries), plan.plan_token,
        )
        return plan

    def end_target_plan(self, direction: str) -> None:
        plan = self._target_plans.pop(direction, None)
        if plan is not None:
            self._invalid_plan_tokens.discard(plan.plan_token)
        if direction == "write":
            self._prepared_write_chunks = None
            self._prepared_write_request = None

    def has_write_plan(self) -> bool:
        """是否已存在写方向的 target 计划（供执行引擎查询，避免窥探私有表）。"""
        return "write" in self._target_plans

    def planned_directions(self) -> tuple[str, ...]:
        """当前活跃的 target 计划方向（含尚未提交的预备写计划）。"""
        directions = list(self._target_plans)
        if (getattr(self, "_prepared_write_request", None) is not None
                and "write" not in directions):
            directions.append("write")
        return tuple(directions)

    def _invalidate_target_uris(self, uris) -> None:
        invalid = set(uris)
        if not invalid:
            return
        for plan in self._target_plans.values():
            if any(entry.target_uri in invalid for entry in plan.entries):
                self._invalid_plan_tokens.add(plan.plan_token)

    def _validate_target_plan(self, plan: DirectTargetPlan) -> None:
        """Validate the plan token without per-layer target metadata lookup."""
        if plan.memory_ticket != self._memory_ticket:
            raise RuntimeError(
                f"direct {plan.direction} target plan memory ticket is stale"
            )
        if plan.plan_token in self._invalid_plan_tokens:
            raise RuntimeError(
                f"direct {plan.direction} target generation mismatch; "
                f"plan token {plan.plan_token} is invalid"
            )

    def _validate_plan_entry(self, entry: DirectTargetPlanEntry) -> None:
        """Check the current cache record for one submitted chunk.

        槽位路径稳定 ⇒ 票据按 URI 长驻；槽位被回收再分配时对象层会换
        generation，`_ensure_targets` 据此关闭旧票据并重开，所以这里的
        「票据 + 大小 + generation 三者一致」就是完整判据。
        """
        cached = getattr(self._store, "_targets", {}).get(entry.target_uri)
        if cached is None:
            raise RuntimeError(
                f"direct {entry.chunk_id!r} target ticket is no longer cached"
            )
        if int(getattr(cached, "ticket", -1)) != entry.target_ticket:
            raise RuntimeError(
                f"direct target ticket invalid for chunk {entry.chunk_id!r}"
            )
        if (int(getattr(cached, "size", entry.target_size)) != entry.target_size
                or int(getattr(cached, "generation", entry.target_generation))
                != entry.target_generation):
            raise RuntimeError(
                f"direct target generation mismatch for chunk {entry.chunk_id!r}"
            )

    def validate_block_tables(self, block_tables):
        geometry = self.geometry
        validated_tables = []
        for table in block_tables:
            table = list(table)
            if len(table) != geometry.blocks_per_chunk:
                raise DirectAdmissionError(
                    f"direct block table length {len(table)} does not match "
                    f"blocks_per_chunk={geometry.blocks_per_chunk}"
                )
            validated = []
            for raw_block_id in table:
                if (not isinstance(raw_block_id, int)
                        or isinstance(raw_block_id, bool)):
                    raise DirectAdmissionError(
                        f"direct block id must be int, got {raw_block_id!r}"
                    )
                block_id = int(raw_block_id)
                if not 0 <= block_id < geometry.num_blocks:
                    raise DirectAdmissionError(
                        f"direct block id {block_id} is outside "
                        f"[0, {geometry.num_blocks})"
                    )
                validated.append(block_id)
            validated_tables.append(validated)
        try:
            caps = self._store._runtime.caps()
        except Exception as exc:
            raise DirectAdmissionError(
                f"Runtime capability query failed for direct batch: {exc}"
            ) from exc
        max_batch = int(
            caps.get("max_batch_entries", caps.get("max_batch_requests", 0)) or 0
        )
        # 单次 submit 的宽度**不是准入条件**：超宽批由 store._submit_retry 按
        # runtime 上限切成多次提交（预切 + 被拒拆半）。这里只观测。
        # 曾经在这抛 DirectAdmissionError：在线实测（TP8 + 2 盘条带，
        # num_layers=80）一层展开 8632 > 8192 就被判越界，经
        # fallback_from_direct 关掉整个直连绑定（close_batch BUSY），最终以
        # EngineDeadError 终止服务。批宽由本步要搬多少 chunk 决定，调用方无法
        # 先验保证，切分才是正确处理。
        request_count = len(validated_tables) * geometry.blocks_per_chunk
        if max_batch and request_count > max_batch:
            _LOG.debug(
                "DIRECT_LAYER_WIDE requests=%d limit=%d（按上限切成多次提交）",
                request_count, max_batch,
            )
        return validated_tables

    def _submit(self, keys, layer_idx: int, block_tables, direction: str):
        store = self._store
        geometry = self.geometry
        started_ns = time.perf_counter_ns()
        keys = [bytes(key) for key in keys]
        block_tables = [list(table) for table in block_tables]
        if len(keys) != len(block_tables):
            raise ValueError(
                f"direct {direction} block table count {len(block_tables)} "
                f"does not match chunk count {len(keys)}"
            )
        if not 0 <= int(layer_idx) < geometry.num_layers:
            raise ValueError(f"direct layer {layer_idx} is out of range")
        with nvtx_range(
            f"tutti.direct.prepare|direction={direction}|layer={layer_idx}"
        ):
            validated_tables = self.validate_block_tables(block_tables)
        io_keys = [derive_io_key(key, int(layer_idx)) for key in keys]
        if direction == "read" and any(io_key not in store._live for io_key in io_keys):
            missing = next(io_key for io_key in io_keys if io_key not in store._live)
            raise ValueError(f"direct get has non-resident key: {missing!r}")
        plan = self._target_plans.get(direction)
        if plan is None:
            if direction == "write":
                self.prepare_write_targets(keys)
            plan = self.begin_target_plan(keys, direction)
        self._validate_target_plan(plan)
        with nvtx_range(
            f"tutti.direct.request_build|direction={direction}|layer={layer_idx}"
        ):
            requests = []
            for io_key, table in zip(io_keys, validated_tables):
                chunk_id, _ = decode_io_key(io_key)
                plan_entry = plan.entry_for(chunk_id)
                self._validate_plan_entry(plan_entry)
                target = plan_entry.target_ticket
                for block_ordinal, block_id in enumerate(table):
                    memory_offset = (
                        block_id * geometry.block_stride_bytes
                        + int(layer_idx) * geometry.layer_stride_bytes
                    )
                    target_offset = (
                        plan_entry.target_offset
                        + int(layer_idx) * geometry.segment_bytes
                        + block_ordinal * geometry.page_bytes
                    )
                    requests.append((
                        target,
                        target_offset,
                        plan.memory_ticket,
                        memory_offset,
                        geometry.page_bytes,
                        direction,
                    ))
        submit_started_ns = time.perf_counter_ns()
        with nvtx_range(
            f"tutti.direct.runtime_submit|op={direction}|layer={layer_idx}"
            f"|chunks={len(keys)}|requests={len(requests)}"
        ):
            handles = store._submit_retry(requests, direction)
        # 逐层计时：debug 级。原先用 info 会在每层每方向打一行——实测高负载下
        # 占日志总行数 97%（4.26M/4.37M 行、867 MB），既淹没真问题（排查 OOM
        # 时被冲掉），又让格式化本身成为可观开销。要看时开
        # VLLM_LOGGING_LEVEL=DEBUG。
        _LOG.debug(
            "DIRECT_SUBMIT_TIMING direction=%s layer=%d chunks=%d requests=%d "
            "python_request_build_ms=%.3f runtime_submit_ms=%.3f",
            direction, layer_idx, len(keys), len(requests),
            (submit_started_ns - started_ns) / 1_000_000,
            (time.perf_counter_ns() - submit_started_ns) / 1_000_000,
        )
        completion_holder = {}
        if direction == "write":
            direct_settle = getattr(store, "_on_direct_put_settled", None)
            if callable(direct_settle):
                settled = lambda ok: direct_settle(
                    ok, io_keys, completion_holder.get("completion")
                )
            else:
                settled = lambda ok: store._on_put_settled(ok, io_keys)
        else:
            settled = lambda _ok: None
        completion = _TuttiCompletion(
            store._runtime, handles, settled,
            auto_watch=False, direction=direction,
        )
        completion_holder["completion"] = completion
        store._track_completion(completion, keys)
        return completion

    def close(self) -> None:
        if self._closed:
            return
        ticket = self._memory_ticket
        unregister = getattr(self._store._runtime, "unregister_memory", None)
        first_error = None
        wait_chunk_io = getattr(self._store, "_wait_chunk_io", None)
        if callable(wait_chunk_io):
            try:
                wait_chunk_io(set(self._store._inflight_by_chunk))
            except Exception as exc:
                first_error = exc
        for direction in tuple(self._target_plans):
            self.end_target_plan(direction)
        finalize_failures = getattr(
            self._store, "finalize_direct_failures", None
        )
        if callable(finalize_failures):
            try:
                finalize_failures()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        close_targets = getattr(self._store, "_close_cached_targets", None)
        if callable(close_targets):
            try:
                close_targets(list(self._store._targets))
            except Exception as exc:
                raise first_error or exc
        if ticket is not None:
            if not callable(unregister):
                raise RuntimeError(
                    "Runtime lost unregister_memory before direct KV pool close"
                )
            unregister(ticket)
        # Do not clear ownership until Runtime confirms success. A BUSY or
        # backend failure leaves the backend retryable after the caller drains.
        self._memory_ticket = None
        self._pool = None
        self._closed = True
        if first_error is not None:
            raise first_error


class _TuttiCompletion:
    """一批 runtime IO 的完成句柄，terminal 详情在 release 后仍保留。"""

    def __init__(self, runtime, handles, on_settled, *, auto_watch: bool = True,
                 direction: str | None = None):
        self._runtime = runtime
        self._submitted = [
            item if isinstance(item, _SubmittedHandle)
            else _SubmittedHandle(int(item))
            for item in handles
        ]
        self._handles = [item.handle for item in self._submitted]
        self._on_settled = on_settled
        self._settled = False
        self._failed = False
        self._failure_message = "tutti IO 批失败"
        self._terminal: bool | None = None
        self._terminal_results: dict[int, TuttiTerminalResult] = {}
        self._batch_result: TuttiBatchResult | None = None
        self._handles_released = False
        self._terminal_lock = threading.Lock()
        self._drain_lock = threading.Lock()
        self._done_callbacks = []
        self._terminal_callbacks = []
        self._ready = threading.Event()
        self._watcher = None
        self._auto_watch = bool(auto_watch)
        self._direction = direction
        self._runtime_wait_calls = 0
        self._release_io_calls = 0
        if auto_watch:
            self._start_watcher()

    def query(self) -> bool:
        if self._settled:
            return not self._failed
        if not self._auto_watch:
            return self._finish() if self._ready.is_set() else False
        if not self._ready.is_set():
            self._probe_runtime()
        if not self._ready.is_set():
            return False
        return self._finish()

    def wait(self, timeout: float | None = None) -> None:
        result = self.wait_result(timeout)
        if not result.ok:
            raise RuntimeError(self._failure_message)

    def wait_result(self, timeout: float | None = None) -> TuttiBatchResult:
        if self._settled:
            assert self._batch_result is not None
            return self._batch_result
        if self._auto_watch:
            self._start_watcher()
        else:
            # Direct completions are deliberately drained synchronously by
            # KVEngine finalization. Keep ``_watcher`` unset: auto_watch=False
            # must never create an observer, including a pseudo-watcher marker.
            with self._drain_lock:
                if self._terminal is None:
                    self._watch_runtime()
        with nvtx_range("tutti.runtime.wait"):
            if not self._ready.wait(timeout):
                raise TimeoutError("等待 tutti IO 批超时")
        self._finish()
        assert self._batch_result is not None
        return self._batch_result

    def wait_detail(self, timeout: float | None = None) -> TuttiBatchResult:
        return self.wait_result(timeout)

    def add_done_callback(self, callback) -> None:
        with self._terminal_lock:
            if self._settled:
                result = self._batch_result
            else:
                self._done_callbacks.append(callback)
                return
        callback(result)

    def add_terminal_callback(self, callback) -> None:
        """Observe structured terminal status without waiting for host drain."""
        with self._terminal_lock:
            if self._terminal is None:
                self._terminal_callbacks.append(callback)
                return
            result = self._build_batch_result()
        callback(result)

    def _start_watcher(self) -> None:
        with self._terminal_lock:
            if self._watcher is not None or self._terminal is not None:
                return
            self._watcher = threading.Thread(
                target=self._watch_runtime,
                name="tutti-io-completion",
                daemon=True,
            )
            self._watcher.start()

    def _observe(self, submitted: _SubmittedHandle,
                 timeout_ms: int) -> TuttiTerminalResult:
        self._runtime_wait_calls += 1
        structured_wait = getattr(self._runtime, "wait_result", None)
        if callable(structured_wait):
            raw = structured_wait(submitted.handle, timeout_ms)
            return TuttiTerminalResult(
                handle=submitted.handle,
                observation=str(raw.observation),
                state=str(raw.state),
                confirmed_bytes=int(raw.confirmed_bytes),
                timeout_seen=bool(raw.timeout_seen),
                failed_request_indices=tuple(raw.failed_request_indices),
                failure_scope=str(raw.failure_scope),
                failure_kind=raw.failure_kind,
                first_failed_entry=raw.first_failed_entry,
                raw_cq_status=raw.raw_cq_status,
                message=str(raw.message),
                batch_indices=submitted.batch_indices,
            )
        observation, state = self._runtime.wait(submitted.handle, timeout_ms)
        failed = state == "FAILED"
        return TuttiTerminalResult(
            handle=submitted.handle,
            observation=observation,
            state=state,
            failure_scope="WHOLE_OPERATION" if failed else "NONE",
            failure_kind="UNKNOWN" if failed else None,
            batch_indices=submitted.batch_indices,
        )

    def _record_result(self, result: TuttiTerminalResult) -> None:
        with self._terminal_lock:
            self._terminal_results.setdefault(result.handle, result)

    def _probe_runtime(self) -> None:
        """做一次非阻塞观察，避免 query 对 watcher 启动存在竞态。"""
        try:
            observed = []
            for submitted in self._submitted:
                result = self._observe(submitted, 0)
                if (result.observation == "TIMEOUT" or
                        result.state not in ("COMPLETED", "FAILED")):
                    return
                observed.append(result)
            for result in observed:
                self._record_result(result)
            failures = [result for result in observed
                        if result.observation != "OK" or result.state == "FAILED"]
            self._mark_terminal(not failures,
                                self._format_failure(failures[0])
                                if failures else None)
        except Exception as exc:
            self._mark_terminal(False, f"tutti IO 等待异常：{exc}")

    def _watch_runtime(self) -> None:
        """阻塞等待并 drain 每个 partial-commit handle，恰好记录一次。"""
        try:
            failures = []
            for submitted in self._submitted:
                while True:
                    result = self._observe(submitted, _COMPLETION_WAIT_MS)
                    if result.observation == "TIMEOUT":
                        continue
                    if (result.observation != "OK" or
                            result.state in ("COMPLETED", "FAILED")):
                        self._record_result(result)
                        if (result.observation != "OK" or
                                result.state == "FAILED"):
                            failures.append(result)
                        break
            self._mark_terminal(not failures,
                                self._format_failure(failures[0])
                                if failures else None)
            self._release_handles()
        except Exception as exc:
            self._mark_terminal(False, f"tutti IO 等待异常：{exc}")
            self._release_handles()

    def _mark_terminal(self, ok: bool, message: str | None) -> None:
        with self._terminal_lock:
            if self._terminal is not None:
                return
            self._terminal = ok
            if message:
                self._failure_message = message
            self._ready.set()
            callbacks = self._terminal_callbacks
            self._terminal_callbacks = []
            result = self._build_batch_result()
        for callback in callbacks:
            callback(result)

    @staticmethod
    def _format_failure(result: TuttiTerminalResult) -> str:
        kind = f"，kind={result.failure_kind}" if result.failure_kind else ""
        message = f"：{result.message}" if result.message else ""
        return f"tutti IO 失败（io handle={result.handle}{kind}）{message}"

    def _build_batch_result(self) -> TuttiBatchResult:
        results = tuple(
            self._terminal_results[item.handle] for item in self._submitted
            if item.handle in self._terminal_results
        )
        failures = tuple(
            result for result in results
            if result.observation != "OK" or result.state == "FAILED"
        )
        failed_batch_indices: set[int] = set()
        for result in failures:
            if (result.failure_scope == "REQUEST_INDICES" and
                    result.failed_request_indices):
                for index in result.failed_request_indices:
                    if 0 <= index < len(result.batch_indices):
                        failed_batch_indices.add(result.batch_indices[index])
            else:
                failed_batch_indices.update(result.batch_indices)
        return TuttiBatchResult(
            ok=not failures,
            failed_batch_indices=tuple(sorted(failed_batch_indices)),
            timeout_seen=any(result.timeout_seen for result in failures),
            failures=failures,
            results=results,
        )

    def _release_handles(self) -> None:
        """Release terminal Runtime handles once, without semantic settle."""
        with self._terminal_lock:
            if self._handles_released:
                return
            self._handles_released = True
            handles = tuple(self._handles)
        for handle in handles:
            try:
                self._release_io_calls += 1
                self._runtime.release_io(handle)
            except Exception:
                pass

    def _finish(self) -> bool:
        with self._terminal_lock:
            terminal = self._terminal
            if terminal is None:
                return False
            if self._settled:
                return not self._failed
            self._settled = True
            self._failed = not terminal
            self._batch_result = self._build_batch_result()
        self._release_handles()
        self._on_settled(terminal)
        with self._terminal_lock:
            callbacks = self._done_callbacks
            self._done_callbacks = []
            result = self._batch_result
        for callback in callbacks:
            callback(result)
        return terminal

    @property
    def drain_stats(self) -> dict:
        return {
            "direction": self._direction or "unknown",
            "wait_result_calls": self._runtime_wait_calls,
            "release_io_calls": self._release_io_calls,
            "failed": bool(self._settled and self._failed),
        }

    def _settle(self, ok: bool) -> None:
        self._mark_terminal(ok, None)
        self._finish()

    @classmethod
    def settled(cls, runtime, on_settled, ok: bool = True) -> "_TuttiCompletion":
        """已成终态的空批：没有提交任何请求（如容量耗尽时不写任何东西）。"""
        completion = cls(runtime, [], on_settled, auto_watch=False)
        completion._settle(ok)
        return completion


class TuttiKVStore:
    """tutti runtime 之上的 KVStore SPI 实现（层盲，io_key 纯映射）。"""

    def __init__(self, root, num_chunks: int, segment_bytes: int,
                 runtime=None, io_stream=None, preset=None,
                 layout="file_per_chunk", mounts=None,
                 initial_slots=None, low_watermark=None,
                 high_watermark=None, max_slots=None,
                 pool_wait_timeout_s: float = 5.0,
                 allocator_enabled: bool = True,
                 rank_id: int = 0, tp_size: int = 1,
                 defer_writes_after_reads: bool | None = None,
                 precreate_threads: int | None = None):
        """preset 为 dict 时优先于 TUTTI_NVME_PRESET 环境变量构造 runtime。

        preset 的字符串值恰为纯十进制整数时转为 int（配置占位符替换后
        的数字字符串由此归一，如 device_id / gpu_id）。

        ``layout="striped"`` 选择多盘布局：一个槽位一个文件，槽位号在
        ``mounts`` 间轮转；默认 file_per_chunk（单 mount）不变。
        """
        if num_chunks <= 0:
            raise ValueError(f"num_chunks 必须为正数，得到 {num_chunks}")
        if segment_bytes <= 0:
            raise ValueError(f"segment_bytes 必须为正数，得到 {segment_bytes}")
        if (not isinstance(rank_id, int) or isinstance(rank_id, bool)
                or rank_id < 0):
            raise ValueError(f"rank_id must be a non-negative integer: {rank_id!r}")
        if (not isinstance(tp_size, int) or isinstance(tp_size, bool)
                or tp_size <= rank_id):
            raise ValueError(
                f"tp_size must be greater than rank_id: {tp_size!r} <= {rank_id!r}"
            )
        self._root = Path(root)
        self._rank_id = rank_id
        self._tp_size = tp_size
        # 容量与预热都以 chunk（对象）为单位，对象层在打开时按对象几何折算成
        # 字节。旧的池水位（initial/low/high/max）没有对应物：物化按需发生、
        # 回收在后台进行，不再需要水位驱动的扩容，也不会阻塞前向线程。
        self._num_chunks = (
            min(num_chunks, max_slots) if max_slots is not None else num_chunks
        )
        warm_chunks = (
            min(32, self._num_chunks) if initial_slots is None
            else min(int(initial_slots), self._num_chunks)
        )
        if high_watermark is not None:
            warm_chunks = min(int(high_watermark), self._num_chunks)
        self._segment_bytes = segment_bytes
        self._runtime = runtime
        self._own_runtime = runtime is None
        self._preset = _normalize_preset(preset) if preset is not None else None
        if self._preset is not None and "daemon_config" in self._preset:
            # layout 构造需要设备字段（striped 的 mounts 来自 devices[].mount_path），
            # 在构造 layout 前先按 daemon 配置推导一次；_build_runtime 的推导
            # 幂等，重复调用不改变结果。
            import yaml
            self._preset = derive_device_fields(self._preset, yaml)
        self._key_namespace: bytes | None = None
        if layout in (None, "file_per_chunk", "file"):
            layout_mounts = [str(self._root)]
        elif layout == "striped":
            if mounts is None:
                mounts = _preset_mounts(self._preset)
            layout_mounts = [str(mount) for mount in mounts]
        else:
            raise ValueError(f"未知 tutti_nvme layout：{layout!r}")
        # 预建/预热口径必须在对象层打开（set_layer_span）之前定下来：它决定
        # open 是"建出缺失的预热槽位"（旧同步语义）还是"只走查已存在的槽位、
        # 把差量交给后台"（异步增长）。
        self._parse_precreate_options(precreate_threads)
        # 文件系统的全部职责（槽位分配、对象头、检查点、恢复、容量）都在
        # 这一层之下：本类只保留内存簿记与 runtime 票据缓存。
        self._layout = ObjectLayout(
            self._root,
            segment_bytes,
            mounts=layout_mounts,
            capacity_chunks=self._num_chunks,
            prewarm_chunks=warm_chunks,
            rank_id=rank_id,
            background_reclaim=bool(allocator_enabled),
            warmup_probe_only=self._precreate_threads > 0,
        )
        self._opened = False
        self._live: set[bytes] = set()
        self._buffers: dict[int, tuple[int, int]] = {}
        self._mem_cache: dict[tuple[int, int], int] = {}
        self._targets: dict[str, _TargetCacheEntry] = {}
        # _targets 由两个线程访问：请求线程（_ensure_targets 按需补开）
        # 与对象池的初始化/后台线程（槽位创建钩子批量预开）。Python
        # dict 的"检查再写"不是原子操作，无锁时并发会丢条目——实测
        # 会让预开的票据凭空消失，写路径反而退化成逐个 open。
        self._targets_lock = threading.Lock()
        self._inflight_by_chunk: dict[bytes, set[_TuttiCompletion]] = {}
        self._inflight_lock = threading.Lock()
        self._deferred_completions: list[_TuttiCompletion] = []
        self._direct_backend = None
        self._direct_failed_chunks: set[bytes] = set()
        self._keepers: list = []  # 持有 ctypes 视图防 GC
        self._next_buffer_id = 0
        self._accel_id = -1
        # runtime 广告的单次提交条目上限（caps.max_batch_requests）；
        # None = 尚未查询，0 = 未知/无限。见 _batch_width_limit。
        self._submit_width_limit: int | None = None
        # 'auto' 在 open() 惰性解析为专用 IO 流（见 _resolve_auto_stream）；
        # 其余取值（int 句柄 / None）原样使用。
        self._io_stream_raw = io_stream
        self._io_stream = None if io_stream == "auto" else io_stream
        # Keep direction-specific handles private.  Workers use the context
        # and fence helpers below instead of replacing these values.
        self._read_stream = None
        self._write_stream = None
        self._read_copy_stream = None
        self._read_stream_obj = None
        self._write_stream_obj = None
        self._read_copy_stream_obj = None
        self._stream_mode = "host"
        self._stream_accel_id = None
        self._execution = "device"
        # 写排在读之后（默认开）。NVMe 读写混跑会互相拖慢：同一 rank 实测并发
        # 时读 kernel +52%（2.08→3.16ms）、写 +31%（0.71→0.93ms），带宽利用率
        # 明显下降。开启后 worker 在写流上插一次 wait_event(最新读层 fence)，
        # 由设备侧保证"先读后写"（Python 不缓冲、不轮询）。实测：读写重叠
        # 40/80 → 7/80，写 kernel 0.81→0.61ms、读 2.62→2.38ms，墙钟中性。
        # 配置优先级：store options 的 defer_writes_after_reads（显式 bool）
        # > TUTTI_DEFER_WRITES_AFTER_READS 环境变量（测试后门）> 默认 True。
        # 读写持续并发的负载若出现写饥饿，显式关掉它回退。
        if defer_writes_after_reads is None:
            env = os.environ.get("TUTTI_DEFER_WRITES_AFTER_READS")
            self._defer_writes_after_reads = True if env is None else env != "0"
        elif isinstance(defer_writes_after_reads, bool):
            self._defer_writes_after_reads = defer_writes_after_reads
        else:
            raise ValueError(
                "defer_writes_after_reads 须为 bool，"
                f"got {defer_writes_after_reads!r}"
            )
        self._defer_warned = False
        # 周期落盘提交索引（见 checkpoint_if_due）；0 = 关闭。
        try:
            self._checkpoint_interval_s = max(
                0.0, float(os.environ.get("TUTTI_CHECKPOINT_INTERVAL_S", "60"))
            )
        except ValueError:
            self._checkpoint_interval_s = 60.0
        self._checkpoint_last_ns = time.monotonic_ns()

    def _parse_precreate_options(self, precreate_threads: int | None) -> None:
        # 后台预建线程数。**默认 0（关闭）**：一旦启用，open 就只走查已存在的
        # 槽位，写路径也不再自己 create+fsync（拿不到就绪槽位时裁剪，与容量
        # 耗尽同一契约）——这是行为变更，必须由部署显式选择。大容量冷启动/
        # 扩容的部署在 serve 脚本里设 TUTTI_PRECREATE_THREADS（如 4）；测试与
        # 小池保持旧的同步建槽语义。
        if precreate_threads is None:
            raw = os.environ.get("TUTTI_PRECREATE_THREADS")
            precreate_threads = int(raw) if raw and raw.isdigit() else 0
        if not isinstance(precreate_threads, int) or precreate_threads < 0:
            raise ValueError(
                "precreate_threads 须为非负整数，"
                f"got {precreate_threads!r}"
            )
        self._precreate_threads = precreate_threads
        # 就绪余量（槽位）：预建只跟随分配前沿保持这一段就绪，容量只是上限
        # （曾有一个"一路补到容量上限"的口径，已删除——盘占用 ∝ 容量而非
        # 工作集，且会与在线 KV 争带宽/写满盘）。
        raw_headroom = os.environ.get("TUTTI_PRECREATE_HEADROOM")
        self._precreate_headroom = (
            int(raw_headroom) if raw_headroom and raw_headroom.isdigit() else 0
        )

    # ---------- 生命周期 ----------

    @property
    def capacity_chunks(self) -> int:
        return self._num_chunks

    @property
    def defer_writes_after_reads(self) -> bool:
        """写批是否应排在读批之后（配置键 defer_writes_after_reads）。"""
        return self._defer_writes_after_reads

    def wait_write_stream_event(self, event) -> None:
        """让后续写 IO 在设备侧排在 ``event`` 之后（主机不阻塞）。

        只加设备侧依赖：写批照原节奏下发，fuse kernel 在 GPU 上等到该事件触发
        才执行。用于把 NVMe 读与写错开（混跑实测读 +52%、写 +31%）。
        """
        stream = self._write_stream_obj
        wait = getattr(stream, "wait_event", None)
        if stream is None or not callable(wait):
            # 写排序静默失效：只观测一次，防止将来流配置变化让性能悄悄
            # 回退（对齐读侧的 DIRECT_THREAD_DEVICE_BIND_FAILED）。
            if not self._defer_warned:
                self._defer_warned = True
                _LOG.debug(
                    "DIRECT_WRITE_DEFER_UNAVAILABLE reason=%s"
                    "（写排序未生效，写与读可能重新并发）",
                    "write stream 缺失" if stream is None
                    else "write stream 无 wait_event",
                )
            return
        wait(event)

    @property
    def max_in_flight_operations(self) -> int:
        """Runtime admission window; 0 means unknown/unbounded."""
        if self._runtime is None:
            return 0
        try:
            value = self._runtime.caps().get("max_in_flight_operations", 0)
            value = int(value or 0)
        except (AttributeError, TypeError, ValueError):
            return 0
        return max(value, 0)

    def _batch_width_limit(self) -> int:
        """单次 submit 的条目上限（0 = 未知/不限）。

        只在首次提交时查一次并缓存：caps() 会跨 pybind 取一遍全部能力字段，
        放在每层每方向的提交热路径上会给前向线程加无谓的 CPU 开销。
        """
        if self._submit_width_limit is None:
            try:
                caps = self._runtime.caps()
                value = int(
                    caps.get(
                        "max_batch_requests", caps.get("max_batch_entries", 0)
                    ) or 0
                )
            except Exception:
                value = 0
            self._submit_width_limit = max(value, 0)
        return self._submit_width_limit

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("tutti store 已 open")
        if self._runtime is None:
            if self._preset is not None:
                self._runtime = _build_runtime(self._preset)
            else:
                self._runtime = _build_runtime_from_env()
        # 对象层在 set_layer_span（层宽定案）时才打开：槽位几何必须先确定。
        # 命名空间不一致由对象层在打开时 fail-closed。
        if self._io_stream_raw == "auto":
            self._resolve_auto_stream()
        else:
            self._configure_shared_stream(self._io_stream)
        self._sync_execution_mode()
        _LOG.warning(
            "TUTTI_STREAM_ROUTING rank=%d accel=%s mode=%s read_io=%s "
            "read_copy=%s write_io=%s",
            self._rank_id, self._stream_accel_id, self._stream_mode,
            self._read_stream, self._read_copy_stream, self._write_stream,
        )
        self._opened = True

    def _runtime_supports_multi_stream(self) -> bool:
        """Return the advertised two-stream capability, conservatively.

        Current bindings expose both fields from the assembled DataPath.
        Missing fields mean an older/unknown runtime, which conservatively
        selects the shared-stream compatibility path.
        """
        try:
            caps = self._runtime.caps()
        except Exception:
            return False
        try:
            return bool(caps.get("supports_multi_stream", False)) and int(
                caps.get("max_concurrent_streams", 0)
            ) >= 2
        except (TypeError, ValueError):
            return False

    def _runtime_accel(self) -> int:
        """Resolve one accelerator shared by runtime, read, and write streams."""
        preset_accel = None
        if isinstance(self._preset, dict) and "gpu_id" in self._preset:
            try:
                preset_accel = int(self._preset["gpu_id"])
            except (TypeError, ValueError):
                preset_accel = None
        bound_accel = None
        try:
            raw = self._runtime.caps().get("bound_accel_id")
            if raw is not None and int(raw) >= 0:
                bound_accel = int(raw)
        except (AttributeError, TypeError, ValueError):
            bound_accel = None
        if (bound_accel is not None and preset_accel is not None
                and bound_accel != preset_accel):
            raise RuntimeError(
                "runtime bound_accel_id 与 preset gpu_id 不一致："
                f"{bound_accel} != {preset_accel}"
            )
        return bound_accel if bound_accel is not None else (preset_accel or 0)

    def _configure_shared_stream(self, stream) -> None:
        self._read_stream = stream
        self._write_stream = stream
        self._read_copy_stream = stream
        self._read_stream_obj = None
        self._write_stream_obj = None
        self._read_copy_stream_obj = None
        self._stream_mode = "shared" if stream is not None else "host"
        if stream is None:
            return
        try:
            import torch

            if not torch.cuda.is_available():
                return
            accel = self._runtime_accel()
            copy_obj = torch.cuda.Stream(device=f"cuda:{accel}")
            self._read_copy_stream_obj = copy_obj
            self._read_copy_stream = int(copy_obj.cuda_stream)
            self._stream_accel_id = accel
        except (ImportError, RuntimeError):
            # Old host/fake runtimes may carry an opaque stream integer without
            # a usable CUDA device. Their compatibility behavior is unchanged.
            return

    def _resolve_auto_stream(self) -> None:
        """在 runtime 加速器上建方向化 IO 流并取其句柄。

        默认流句柄为 0，与绑定的空指针语义冲突（submit 会当作未给流
        拒绝）。落在 preset 的 gpu_id 设备上（runtime 校验流须属于自身
        加速器）。旧 runtime 或不支持 multi-stream 的 target 明确回退
        到共享专用流。
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("io_stream='auto' 需要 CUDA 可用")
        accel = self._runtime_accel()
        self._accel_id = accel
        self._stream_accel_id = accel
        read_obj = torch.cuda.Stream(device=f"cuda:{accel}")
        # Copy/reshape is not a Runtime/DataPath submit stream, so it does not
        # consume a backend concurrent-stream capability slot. It is always a
        # separate CUDA stream in auto mode, including a shared-IO fallback.
        read_copy_obj = torch.cuda.Stream(device=f"cuda:{accel}")
        if not self._runtime_supports_multi_stream():
            self._read_stream_obj = read_obj
            self._write_stream_obj = read_obj
            self._read_copy_stream_obj = read_copy_obj
            self._read_stream = int(read_obj.cuda_stream)
            self._write_stream = self._read_stream
            self._read_copy_stream = int(read_copy_obj.cuda_stream)
            self._io_stream = self._read_stream
            self._stream_mode = "shared"
            return

        write_obj = torch.cuda.Stream(device=f"cuda:{accel}")
        # Both streams are constructed on the same explicit CUDA device.  A
        # custom torch stream implementation must still expose a compatible
        # device, otherwise fail before any IO is submitted.
        for stream_obj in (read_obj, write_obj, read_copy_obj):
            device = getattr(stream_obj, "device", None)
            index = getattr(device, "index", None)
            if index is not None and int(index) != accel:
                raise RuntimeError(
                    "read/write CUDA stream 必须属于 runtime accel device "
                    f"{accel}，得到 {index}"
                )
        self._read_stream_obj = read_obj
        self._write_stream_obj = write_obj
        self._read_copy_stream_obj = read_copy_obj
        self._read_stream = int(read_obj.cuda_stream)
        self._write_stream = int(write_obj.cuda_stream)
        self._read_copy_stream = int(read_copy_obj.cuda_stream)
        self._io_stream = None
        self._stream_mode = "dual"

    def close(self) -> None:
        if not self._opened:
            return
        backend_error = None
        backend = self._direct_backend
        if backend is not None and not backend._closed:
            try:
                backend.close()
            except Exception as exc:
                if not backend._closed:
                    raise
                backend_error = exc
        try:
            self._wait_chunk_io(set(self._inflight_by_chunk))
        except Exception:
            _LOG.exception("等待对象池 target IO 完成失败；继续关闭句柄")
        # 先让后台预建停手，再关对象层：避免线程在池已关闭后继续建文件。
        self._layout.stop_background_precreate()
        try:
            self._layout.close_object_pool()
        except Exception:
            _LOG.exception("关闭对象池失败；继续关闭 target 句柄")
        try:
            self._close_cached_targets(list(self._targets))
        finally:
            self._opened = False
        self._live = set()
        self._buffers = {}
        self._mem_cache = {}
        self._targets = {}
        self._inflight_by_chunk = {}
        self._direct_backend = None
        self._direct_failed_chunks = set()
        self._keepers = []
        if self._own_runtime and self._runtime is not None:
            try:
                self._runtime.shutdown(5000)
            except Exception:
                pass
            self._runtime = None
        # Keep all CUDA stream objects alive through runtime shutdown, then
        # release them together so no handle can outlive its owner.
        self._read_stream = None
        self._write_stream = None
        self._read_copy_stream = None
        self._read_stream_obj = None
        self._write_stream_obj = None
        self._read_copy_stream_obj = None
        self._stream_mode = "host"
        self._stream_accel_id = None
        self._io_stream = None
        self._own_runtime = self._runtime is None
        if backend_error is not None:
            raise backend_error

    def _sync_execution_mode(self) -> None:
        """推导 submit 执行模式：无 io_stream → host 路径；有则按 runtime
        能力（device 优先，host-only 回退）。"""
        if self._read_stream is None and self._write_stream is None:
            self._execution = "host"
            return
        try:
            caps = self._runtime.caps()
            memories = caps.get("memory", [])
        except Exception:
            return
        if "device" in memories:
            self._execution = "device"
        elif "host" in memories:
            self._execution = "host"

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("tutti store 未 open")

    # ---------- SPI ----------

    def register_buffer(self, buffer, granularity: int) -> int | None:
        """注册批量 IO 的内存缓冲，返回 store 内部 buffer_id。

        具体注册的是哪块内存：engine.bind 传入的 staging 环窗缓冲——
        一段连续 device 显存（或 host 内存），按"槽"等分，布局为
        slots × segment_bytes，槽 i 占据
        [i×segment_bytes, (i+1)×segment_bytes)。本方法的 buffer 即
        该段内存的起始视图；size 取自对象属性（numel×element_size）。

        granularity：单条批量 IO 的字节粒度 = 层段宽（segment_bytes）。
        DMA 以此为对齐与切分单位；必须是 _IO_PAGE_BYTES 的正倍数。

        返回 buffer_id：store 局部编号，映射到底层 (addr, size)。
        之后 put_batch/get_batch 的每条 IO 以
        (io_key, buffer_id, offset) 寻址，offset 即槽内字节偏移。

        语义：同一 (addr, size) 只向 runtime 注册一次（ticket 复用）；
        注册期间持有 buffer 引用防止回收；注册失败 → None（调用方
        视为该缓冲不可用于 DMA）。
        """
        self._require_open()
        if granularity is None or granularity <= 0:
            return None
        if granularity % _IO_PAGE_BYTES != 0:
            return None
        info = _buffer_info(buffer)
        if info is None:
            return None
        addr, size, kind, accel_id = info
        mem_ticket = self._mem_cache.get((addr, size))
        if mem_ticket is None:
            try:
                mem_ticket = self._runtime.register_memory(
                    addr, size, kind, accel_id=accel_id, io_granularity=granularity
                )
            except Exception:
                return None
            self._mem_cache[(addr, size)] = mem_ticket
            self._keepers.append(buffer)
        self._next_buffer_id += 1
        self._buffers[self._next_buffer_id] = (addr, size)
        return self._next_buffer_id

    def create_direct_transfer(
        self,
        kv_caches,
        *,
        num_layers: int,
        blocks_per_chunk: int,
        chunk_tokens: int,
        segment_bytes: int,
    ):
        """Create the Python byte-range direct backend.

        The backend uses only Runtime.register_memory/submit and keeps model,
        layer, and block-table semantics above Runtime/DataPath.
        """
        self._require_open()
        return TuttiDirectBackend(self)

    def _stream_for(self, direction: str):
        if direction == "read":
            return self._read_stream, self._read_stream_obj
        if direction == "read_copy":
            return self._read_copy_stream, self._read_copy_stream_obj
        if direction == "write":
            return self._write_stream, self._write_stream_obj
        raise ValueError(f"未知 IO 方向：{direction!r}")

    def stream_context(self, direction: str):
        """Return a context for worker scatter/gather enqueue.

        The returned context selects the store-owned stream and does not
        expose a setter for replacing it.  Host/shared-stream fallbacks are a
        no-op context.
        """
        _, stream_obj = self._stream_for(direction)
        if stream_obj is None:
            return nullcontext()
        import torch

        return torch.cuda.stream(stream_obj)

    def _wait_event(self, direction: str, event) -> None:
        _, stream_obj = self._stream_for(direction)
        if event is None:
            return
        if stream_obj is not None:
            stream_obj.wait_event(event)
        # Host/fake stores have no stream to wait on.  Do not turn an event
        # dependency into a host synchronize; production reuse is guarded by
        # CUDA stream wait_event above, and completion draining owns host waits.

    def wait_read_event(self, event) -> None:
        """Make the read stream wait for a worker producer fence."""
        self._wait_event("read", event)

    def wait_read_copy_event(self, event) -> None:
        """Make the read-copy stream wait for a read completion fence."""
        self._wait_event("read_copy", event)

    def wait_write_event(self, event) -> None:
        """Make the write stream wait for a worker gather fence."""
        self._wait_event("write", event)

    def drain_deferred_completions(self) -> None:
        """Drain submit-error handles during engine-level abort/final drain."""
        pending = self._deferred_completions
        self._deferred_completions = []
        first_error = None
        for completion in pending:
            try:
                completion.wait_result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def wait_event(self, event) -> None:
        """Backward-compatible alias for the write-side fence."""
        self.wait_write_event(event)

    def _record_event(self, direction: str, event=None):
        _, stream_obj = self._stream_for(direction)
        if stream_obj is None:
            # An event without an owning CUDA stream is not recorded and must
            # never be handed to a downstream wait_event call.
            return None
        if event is None:
            import torch

            event = torch.cuda.Event()
        event.record(stream_obj)
        return event

    def record_read_event(self, event=None):
        """Record and return a fence on the store-owned read stream."""
        return self._record_event("read", event)

    def record_read_copy_event(self, event=None):
        """Record and return a fence after scatter on the read-copy stream."""
        return self._record_event("read_copy", event)

    def record_write_event(self, event=None):
        """Record and return a fence on the store-owned write stream."""
        return self._record_event("write", event)

    def record_compute_event(self, event=None):
        """Record a fence on vLLM's current compute stream."""
        if self._stream_accel_id is None:
            return None
        import torch

        stream = torch.cuda.current_stream()
        if event is None:
            event = torch.cuda.Event()
        event.record(stream)
        return event

    def wait_compute_event(self, event) -> None:
        """Enqueue a device-side wait on vLLM's restored current stream."""
        if event is None:
            return
        import torch

        torch.cuda.current_stream().wait_event(event)

    def read_copy_stream_handle(self) -> int:
        """Return the controlled stream handle used for scatter release fences."""
        return int(self._read_copy_stream or 0)

    def put_batch(self, batch) -> _TuttiCompletion:
        self._require_open()
        entries = self._normalize(batch, require_live=False)
        io_keys = [io_key for io_key, _, _ in entries]
        chunk_ids = tuple(dict.fromkeys(
            decode_io_key(io_key)[0] for io_key in io_keys
        ))
        admitted, rejected = self._layout.prepare_put(io_keys, self._num_chunks)
        if rejected:
            # 容量耗尽：只写被受理的 chunk。缓存装不下不是请求的错误，本层
            # 按对象层契约裁剪本批（绝不阻塞、绝不抛错）。
            _LOG.warning(
                "DIRECT_WRITE_ADMISSION_SHORTFALL admitted=%d rejected=%d "
                "capacity=%d",
                len(admitted), rejected, self._num_chunks,
            )
            entries = [item for item in entries if bytes(item[0]) in admitted]
            io_keys = [io_key for io_key, _, _ in entries]
            chunk_ids = tuple(dict.fromkeys(
                decode_io_key(io_key)[0] for io_key in io_keys
            ))
            if not entries:
                return _TuttiCompletion.settled(
                    self._runtime, lambda _ok: None
                )
        targets = self._ensure_targets(entries)
        requests = []
        for io_key, buffer_id, offset in entries:
            chunk_id, layer = decode_io_key(io_key)
            uri = self._layout.target_uri(chunk_id)
            requests.append(
                (
                    targets[uri],
                    self._layout.target_offset(chunk_id)
                    + layer * self._segment_bytes,
                    self._mem_for(buffer_id),
                    offset,
                    self._segment_bytes,
                    "write",
                )
            )
        with nvtx_range(f"tutti.runtime.submit|op=write|requests={len(requests)}"):
            handles = self._submit_retry(requests, "write")
        completion = _TuttiCompletion(
            self._runtime, handles, lambda ok: self._on_put_settled(ok, io_keys)
        )
        self._track_completion(completion, chunk_ids)
        return completion

    def get_batch(self, batch) -> _TuttiCompletion:
        self._require_open()
        entries = self._normalize(batch, require_live=True)
        targets = self._ensure_targets(entries)
        chunk_ids = tuple(dict.fromkeys(
            decode_io_key(io_key)[0] for io_key, _, _ in entries
        ))
        requests = []
        for io_key, buffer_id, offset in entries:
            chunk_id, layer = decode_io_key(io_key)
            uri = self._layout.target_uri(chunk_id)
            requests.append(
                (
                    targets[uri],
                    self._layout.target_offset(chunk_id)
                    + layer * self._segment_bytes,
                    self._mem_for(buffer_id),
                    offset,
                    self._segment_bytes,
                    "read",
                )
            )
        with nvtx_range(f"tutti.runtime.submit|op=read|requests={len(requests)}"):
            handles = self._submit_retry(requests, "read")
        # Read completions are drained by KVEngine.wait_idle at host
        # completion time.  Do not start a background watcher while
        # start_load_kv is enqueueing the complete layer plan.
        completion = _TuttiCompletion(
            self._runtime, handles, lambda _ok: None
        )
        self._track_completion(completion, chunk_ids)
        return completion

    def drop(self, keys) -> None:
        self._require_open()
        io_keys = []
        chunk_ids = set()
        for key in keys:
            chunk_id, _ = decode_io_key(key)  # 类型/非空校验
            io_keys.append(bytes(key))
            chunk_ids.add(chunk_id)
        released = self._layout.releasable_chunks(io_keys)
        self._wait_chunk_io(released)
        self._close_cached_targets(self._chunk_target_uris_to_close(released))
        self._layout.drop(io_keys)
        self._live.difference_update(io_keys)
        # 对象被回收 ⇒ 该 chunk 的全部层一起离开在场集合（按对象提交 ⇒
        # 不存在"半删的 chunk"）。
        span = self._layout.layer_span or 0
        for chunk_id in released:
            if len(chunk_id) == 16:
                self._live.difference_update(
                    chunk_id + layer.to_bytes(2, "little")
                    for layer in range(span)
                )

    def checkpoint_if_due(self) -> bool:
        """按时间节流落盘提交索引（TUTTI_CHECKPOINT_INTERVAL_S，默认 60s）。

        为什么必须周期落盘：commit() 只写对象头并把检查点标脏，而脏检查点
        原先只在 close() 落盘（见对象层 load_checkpoint_locked 的契约注释）。
        长跑进程被强杀/崩溃/关闭链中断时检查点从未写过——重启后索引为空，
        盘上数据虽完好却全部失去复用性（实测 4 TiB 池重启后 hit_tokens=0，
        等效冷池）。

        为什么安全：检查点是多容器轮转写，落盘中途崩溃不会写坏上一个容器；
        且**对象头才是权威**——未进检查点的对象重启后其槽位保持 free，会被
        后续分配复用，不泄漏空间（代价仅是那部分 KV 不可复用）。

        单次开销 = 检查点状态表（209715 槽约 122 MB），默认 60s 一次。

        返回 True 表示本次真的落盘了。
        """
        interval = self._checkpoint_interval_s
        if interval <= 0:
            return False
        now = time.monotonic_ns()
        elapsed = (now - self._checkpoint_last_ns) / 1e9
        if elapsed < interval:
            _LOG.debug(
                "DIRECT_CHECKPOINT_SKIP elapsed=%.1fs interval=%.1fs",
                elapsed, interval,
            )
            return False
        # 先推时间戳：落盘失败也不该每步重试（避免放大故障）。
        self._checkpoint_last_ns = now
        try:
            self._layout.checkpoint()
        except Exception:
            _LOG.warning(
                "DIRECT_CHECKPOINT_FAILED 索引落盘失败；复用收益可能退化到"
                "上一次成功的检查点", exc_info=True,
            )
            return False
        _LOG.info(
            "DIRECT_CHECKPOINT_WRITTEN elapsed=%.1fs（提交索引已落盘，重启可复用）",
            elapsed,
        )
        return True

    def scan(self):
        """已驻留 io_key 快照（升序）。

        内存视图是权威（提交成功即入内存，不再扫盘）；并上对象层的恢复集合
        只为纳入本进程之外（上一代进程）已提交的对象——那部分只在冷启动时
        非空。
        """
        self._require_open()
        self._live |= self._layout.scan()
        return sorted(self._live)

    def has(self, io_key) -> bool:
        """存活查询（O(1)）：读取侧跳过无数据层（混合注意力模型的
        线性注意力层从未落盘，属正常状态而非错误）。"""
        return io_key in self._live

    def set_key_namespace(self, namespace: bytes) -> None:
        """声明 key 命名空间（engine 构造期注入，打开对象层前生效）。

        对象层用它做归属校验（对象头 + 检查点记录），不一致时 fail-closed，
        禁止静默复用异构数据；字节串对本层不透明。
        """
        if self._opened:
            raise RuntimeError("命名空间须在 open 之前注入")
        self._key_namespace = bytes(namespace)
        self._layout.set_namespace(self._key_namespace)

    def set_layer_span(self, num_layers: int) -> None:
        """定层宽并打开对象层：对象几何（段数 × 段大小 + 对象头）由此确定。

        对象层在此打开——它承担槽位物化（含预热）、检查点加载与崩溃恢复，
        恢复出的已提交 chunk 立即进入内存视图（`_live`）。本调用发生在
        worker 初始化期，不在请求路径上。
        """
        self._layout.set_layer_span(num_layers)
        self._live = self._layout.scan()
        self._preopen_ready_targets()
        # 对象层已打开、几何已定：把"容量增长"整体交给后台线程——既避免大容量
        # 冷启动在 open 期写实零，也避免写路径按需 create+fsync（详见
        # object_layout.start_background_precreate 的契约说明）。
        if self._precreate_threads > 0:
            self._layout.start_background_precreate(
                self._precreate_threads,
                headroom=self._precreate_headroom,
            )

    def object_pool_snapshot(self) -> dict | None:
        return self._layout.object_pool_snapshot()

    def abort_chunks(self, chunk_ids) -> None:
        """Rollback incomplete chunks after the step has drained all IO.

        容忍"已被驱逐/回收"的 chunk：驱逐（apply_evictions → drop）会把
        chunk 移出预留表，而本步的 _save_keys 仍可能含它——调度侧的驱逐决策
        与本步的写受理会交叉。对这类 chunk 没有可回滚的预留，跳过即可；
        否则 target_uri 的 fail-fast 会把一次写失败升级为 worker 崩溃
        （2026-09-20 在线驱逐压测复现：KeyError → EngineDeadError）。
        """
        chunks = tuple(dict.fromkeys(bytes(chunk_id) for chunk_id in chunk_ids))
        self._wait_chunk_io(chunks)
        self._close_cached_targets(self._layout.reserved_uris(chunks))
        self._layout.abort_uncommitted(chunks)

    # ---------- 内部 ----------

    def _normalize(self, batch, require_live: bool):
        entries = []
        for item in batch:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError(f"批条目须为 (io_key, buffer_id, offset)，得到 {item!r}")
            io_key, buffer_id, offset = item
            decode_io_key(io_key)
            if buffer_id not in self._buffers:
                raise ValueError(f"未注册的 buffer id：{buffer_id}")
            if not isinstance(offset, int):
                raise ValueError(f"offset 须为 int，得到 {type(offset).__name__}")
            _, size = self._buffers[buffer_id]
            if offset < 0 or offset + self._segment_bytes > size:
                raise ValueError(
                    f"buffer 偏移越界：offset={offset} + 段长 "
                    f"{self._segment_bytes} > buffer {size} 字节"
                )
            if require_live and bytes(io_key) not in self._live:
                raise ValueError(f"get 未驻留 key：{bytes(io_key)!r}")
            entries.append((bytes(io_key), buffer_id, offset))
        return entries

    def _preopen_ready_targets(self) -> None:
        """启动期把对象层已物化槽位的目标票据一次性开好。

        推理路径的 `_ensure_targets` 是"首次接触某 chunk 才 open"，而
        open 的成本是 resolve（open+fstat+fsync+FIEMAP）加句柄构建
        （cudaMalloc 192B + H2D + D2H），`open_batch` 还会为每个 URI
        现建一个线程；首次 resolve 同时触发 peer-memory 注册（实测单次
        200~275ms，且全程持 runtime registry 锁）。

        槽位路径稳定（分配/回收都不改名），所以这些票据可以在启动时
        一次开好，运行时只做内存查找。

        失败不致命：记录告警后回退到按需 open 的老路径。
        """
        if not callable(getattr(self._runtime, "open_batch", None)):
            return
        ready = self._layout.ready_slots()
        if not ready:
            return
        size = self._layout.slot_payload_bytes
        try:
            with nvtx_range(
                f"tutti.direct.preopen_ready|slots={len(ready)}"
            ):
                self._preopen_uris(ready, size)
        except Exception as exc:
            _LOG.warning("DIRECT_PREOPEN_READY_FAILED err=%r", exc)
            return
        self._warm_uris = [uri for uri, _ in ready]
        _LOG.info(
            "DIRECT_PREOPEN_READY slots=%d targets=%d",
            len(self._warm_uris), len(self._targets),
        )

    def _preopen_uris(self, entries, size: int) -> None:
        """把一批 ``(uri, generation)`` 打开成运行时票据并缓存。

        槽位路径稳定（分配/回收都不改名），因此票据可长驻；这里写入的
        size/generation 来自对象层，与后续 `_ensure_targets` 的校验口径一致，
        不会被误判为失效而重开。
        """
        with self._targets_lock:
            missing = [
                (uri, generation)
                for uri, generation in entries
                if uri not in self._targets
            ]
        if not missing:
            return
        tickets = self._runtime.open_batch([uri for uri, _ in missing])
        if len(tickets) != len(missing):
            raise RuntimeError("Runtime.open_batch returned wrong handle count")
        with self._targets_lock:
            for (uri, generation), ticket in zip(missing, tickets):
                self._targets.setdefault(
                    uri,
                    _TargetCacheEntry(int(ticket), int(size), int(generation)),
                )

    def _ensure_targets(self, entries) -> dict[str, int]:
        """解析一批 io_key 的运行时目标句柄。

        对象池模式下的快速路径：chunk 已绑定槽位且该槽位已有就绪
        GpuFile 时，句柄是**纯内存**取用（槽位路径稳定，句柄随槽位
        常驻）。只有池未覆盖的路径（无池、槽位未就绪）才回落到按需
        open。
        """
        descriptors = []
        seen = set()
        stale = []
        for io_key, _, _ in entries:
            chunk_id, _ = decode_io_key(io_key)
            uri = self._layout.target_uri(chunk_id)
            if uri in seen:
                continue
            seen.add(uri)
            size = self._layout.target_size(chunk_id)
            generation = self._layout.target_generation(chunk_id)
            cached = self._targets.get(uri)
            if cached is not None and (
                cached.size != size or cached.generation != generation
            ):
                stale.append(uri)
                cached = None
            descriptors.append((uri, size, generation))
        if stale:
            self._close_cached_targets(stale)
        with self._targets_lock:
            missing = [item for item in descriptors if item[0] not in self._targets]
        if missing:
            uris = [uri for uri, _, _ in missing]
            tickets = self._runtime.open_batch(uris)
            if len(tickets) != len(uris):
                raise RuntimeError("Runtime.open_batch returned wrong handle count")
            with self._targets_lock:
                for (uri, size, generation), ticket in zip(missing, tickets):
                    self._targets.setdefault(
                        uri,
                        _TargetCacheEntry(int(ticket), int(size), int(generation)),
                    )
        with self._targets_lock:
            return {uri: self._targets[uri].ticket for uri, _, _ in descriptors}

    def _chunk_target_uris_to_close(self, chunk_ids) -> list[str]:
        """解绑/中止/失败时应当关闭票据的 chunk 目标 URI。

        对象池模式下票据是**槽位级**的：槽位路径稳定、文件不随解绑消失
        （回收只是就地零化，extent 不变），因此解绑不应关闭票据——否则
        每次回收后再次使用都要重跑 open（resolve + 句柄构建 + 每 URI 一
        个线程），正是推理路径上要消除的开销。槽位文件真被重建时，
        `target_generation` 会在 `_ensure_targets` 里判定失效并关闭。
        """
        return []

    def _close_cached_targets(self, uris) -> None:
        records = [(uri, self._targets[uri]) for uri in dict.fromkeys(uris)
                   if uri in self._targets]
        if not records:
            return
        invalidate = getattr(self._direct_backend, "_invalidate_target_uris", None)
        if callable(invalidate):
            invalidate(uri for uri, _ in records)
        tickets = [record.ticket for _, record in records]
        close_batch = getattr(self._runtime, "close_batch", None)
        if callable(close_batch):
            close_batch(tickets)
        else:
            close_target = getattr(self._runtime, "close_target", None)
            if not callable(close_target):
                raise RuntimeError(
                    "Runtime must expose close_batch or close_target for target lifecycle"
                )
            for ticket in tickets:
                close_target(ticket)
        for uri, _ in records:
            self._targets.pop(uri, None)

    def _track_completion(self, completion, chunk_ids) -> None:
        chunks = tuple(dict.fromkeys(bytes(chunk_id) for chunk_id in chunk_ids))
        with self._inflight_lock:
            for chunk_id in chunks:
                self._inflight_by_chunk.setdefault(chunk_id, set()).add(completion)

        def done(_result):
            with self._inflight_lock:
                for chunk_id in chunks:
                    active = self._inflight_by_chunk.get(chunk_id)
                    if active is None:
                        continue
                    active.discard(completion)
                    if not active:
                        self._inflight_by_chunk.pop(chunk_id, None)

        completion.add_done_callback(done)

    def _wait_chunk_io(self, chunk_ids) -> None:
        with self._inflight_lock:
            completions = {
                completion
                for chunk_id in chunk_ids
                for completion in self._inflight_by_chunk.get(chunk_id, ())
            }
        for completion in completions:
            completion.wait_result()

    def _mem_for(self, buffer_id: int) -> int:
        addr, size = self._buffers[buffer_id]
        return self._mem_cache[(addr, size)]

    def _on_put_settled(self, ok: bool, io_keys) -> None:
        """put 批 settle：数据确认落盘后才提交对象并更新在场集（崩溃安全）。

        只有**对象已提交**（chunk 的全部段都写过）的那批层才进入在场集：
        半截对象按"没写过"处理（一层没写完就当整个 chunk 没写），否则读到
        的是从未校验过的字节。
        """
        if ok:
            committed = self._layout.commit_layers(io_keys)
            if not committed:
                return
            span = self._layout.layer_span or 0
            for chunk_id in committed:
                if len(chunk_id) == 16:
                    # 标准 io_key：对象有效 ⇔ 全部层段都写过，因此每一层
                    # 都可读——本批只带了触发提交的那一层，其余层必须一起
                    # 进入在场集合（否则复用只能命中最后一层）。
                    self._live.update(
                        chunk_id + layer.to_bytes(2, "little")
                        for layer in range(span)
                    )
                else:
                    # 通用短 key（通用 KV 契约）：形态不规范化，只记本批原样
                    self._live.update(
                        io_key for io_key in io_keys
                        if decode_io_key(io_key)[0] == chunk_id
                    )
            return
        chunk_ids = tuple(dict.fromkeys(
            decode_io_key(io_key)[0] for io_key in io_keys
        ))
        self._close_cached_targets(self._chunk_target_uris_to_close(chunk_ids))
        self._layout.abort_uncommitted(chunk_ids)

    def _on_direct_put_settled(self, ok: bool, io_keys, completion) -> None:
        """Settle direct writes without closing targets mid-drain."""
        if ok:
            self._on_put_settled(True, io_keys)
            return
        chunk_ids = tuple(dict.fromkeys(
            decode_io_key(io_key)[0] for io_key in io_keys
        ))
        self._direct_failed_chunks.update(chunk_ids)
        if completion is None:
            return
        with self._inflight_lock:
            active = {
                item
                for chunk_id in chunk_ids
                for item in self._inflight_by_chunk.get(chunk_id, ())
            }
        # A lone direct completion can be settled immediately. When another
        # layer still uses the target, defer recycle/close until engine drain.
        if active == {completion}:
            with self._inflight_lock:
                active_writes = {
                    item
                    for pending in self._inflight_by_chunk.values()
                    for item in pending
                    if getattr(item, "_direction", None) == "write"
                }
            if active_writes != {completion}:
                return
            backend = self._direct_backend
            end_plan = getattr(backend, "end_target_plan", None)
            if callable(end_plan):
                end_plan("write")
            self.finalize_direct_failures()

    def finalize_direct_failures(self) -> None:
        """Close failed write targets after all direct completions drain."""
        chunks = tuple(self._direct_failed_chunks)
        if not chunks:
            return
        self._direct_failed_chunks.difference_update(chunks)
        try:
            self._close_cached_targets(
                self._chunk_target_uris_to_close(chunks)
            )
            self._layout.abort_uncommitted(chunks)
            self._live = {
                io_key for io_key in self._live
                if decode_io_key(io_key)[0] not in chunks
            }
        except Exception:
            self._direct_failed_chunks.update(chunks)
            raise

    def _submit_retry(self, requests, direction: str):
        """提交整批；超宽先按上限切段，partial-commit 的被拒请求窗口重发。

        宽度处理分两层：
          * **提交前预切**：批宽超过 runtime 广告的 max_batch_requests 时，
            直接按上限切成多次 submit（多次 kernel）。宽度由本步要搬多少
            chunk 决定（长 prompt + 高并发时可达上万条），不是调用方能先验
            保证的；预切把它变成常规分批，不再浪费一次必然失败的提交。
          * **被拒后拆半**：预切用的是"请求条数"，而 runtime 的上限算的是
            展开后的 sub-IO 条目数（一个请求跨分片/超 MDTS 时会展开成多条）。
            若仍撞上限就拆半重投；拆到单笔仍被拒才说明是那一笔自身的问题。
        窗口耗尽（在飞配额/容量）另算：重投，连续两轮零接受才 RuntimeError。
        """
        handles = []
        indexed = list(enumerate(requests))
        width_limit = self._batch_width_limit()
        if width_limit and len(indexed) > width_limit:
            # 逆序入栈：循环取 segments[-1]，倒着压保证按原顺序提交。
            segments = [
                indexed[start:start + width_limit]
                for start in range(0, len(indexed), width_limit)
            ][::-1]
            _LOG.debug(
                "TUTTI_SUBMIT_PRESPLIT direction=%s requests=%d limit=%d "
                "segments=%d",
                direction, len(indexed), width_limit, len(segments),
            )
        else:
            segments = [indexed]
        all_rejected_rounds = 0
        rejection_diagnostics = []
        try:
            while segments:
                pending = segments[-1]
                result = self._runtime.submit(
                    [request for _, request in pending],
                    accel_id=self._accel_id,
                    stream=self._stream_for(direction)[0],
                    execution=self._execution,
                )
                rejected = list(result.rejected or [])
                # Runtime partial-commit deliberately returns a non-OK status
                # together with a valid handle for the accepted prefix/subset.
                # That handle must be drained exactly once; only rejected indices
                # are retried.  Treating status_ok as all-or-nothing loses issued IO.
                if result.io_handle is not None:
                    accepted_indices = tuple(
                        pending[i][0]
                        for i, accepted in enumerate(result.initial_states or [])
                        if accepted
                    )
                    handles.append(_SubmittedHandle(result.io_handle,
                                                    accepted_indices))
                elif len(rejected) != len(pending):
                    raise RuntimeError(
                        "tutti submit 返回部分受理状态但缺少 IO handle："
                        f"{result.status_msg}"
                    )
                if not rejected:
                    if result.io_handle is None or not result.status_ok:
                        raise RuntimeError(f"tutti submit 失败：{result.status_msg}")
                    segments.pop()
                    all_rejected_rounds = 0
                    continue
                if len(rejected) == len(pending):
                    rejection_diagnostics.append(
                        f"round={all_rejected_rounds + 1} pending={len(pending)} "
                        f"status={result.status_msg}"
                    )
                    # 整批被拒且原因是"这一批太宽"：拆半重投（见 docstring）。
                    if len(pending) > 1 and _is_batch_width_rejection(
                        result.status_msg
                    ):
                        mid = len(pending) // 2
                        segments[-1] = pending[:mid]
                        segments.append(pending[mid:])
                        _LOG.warning(
                            "TUTTI_SUBMIT_SPLIT direction=%s wide=%d -> %d+%d "
                            "status=%s",
                            direction, len(pending), mid, len(pending) - mid,
                            result.status_msg,
                        )
                        continue
                    all_rejected_rounds += 1
                    if all_rejected_rounds >= 2:
                        raise RuntimeError(
                            "partial-commit 连续两轮零接受："
                            + "; ".join(rejection_diagnostics[-2:])
                        )
                else:
                    all_rejected_rounds = 0
                _LOG.warning(
                    "TUTTI_PARTIAL_REJECT direction=%s rejected=%s "
                    "retry_round=%d status=%s",
                    direction, rejected, all_rejected_rounds,
                    result.status_msg,
                )
                segments[-1] = [pending[i] for i in rejected]
        except Exception:
            if handles:
                self._deferred_completions.append(
                    _TuttiCompletion(
                        self._runtime, handles, lambda _ok: None,
                        auto_watch=False,
                    )
                )
            raise
        return handles


# ---------- 真机 runtime 构造（TUTTI_NVME_PRESET） ----------


# 兼容别名：装配逻辑已迁至 stores.tutti_nvme.runtime_factory，这里保留
# 同名私有符号供既有调用点与测试使用（行为完全一致）。
_normalize_preset = normalize_preset
_preset_mounts = preset_mounts
_build_runtime = build_runtime
_build_runtime_from_env = build_runtime_from_env
_derive_device_fields = derive_device_fields

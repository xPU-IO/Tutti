"""Rank-local fixed-size storage object pool.

The request path only renames already materialized slot files. Slot creation,
real-zero writes, fsync, and FIEMAP verification happen during bind or on the
background allocator thread.
"""

from __future__ import annotations

import fcntl
import json
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path


_FIEMAP_IOCTL = 0xC020660B
_FIEMAP_HEADER = struct.Struct("<QQIIII")
# fiemap_extent: logical, physical, length, reserved64, device, flags,
# reserved[3], reserved2 (4x u64 + 6x u32 = 56 bytes).
_FIEMAP_EXTENT = struct.Struct("<QQQQIIIIII")
_FIEMAP_EXTENT_COUNT = 4096
_FIEMAP_EXTENT_LAST = 0x00000001
_FIEMAP_EXTENT_DELALLOC = 0x00000004
_FIEMAP_EXTENT_UNWRITTEN = 0x00000800
_MANIFEST_VERSION = 1


class PoolResourceExhausted(RuntimeError):
    """Bounded object-pool allocation failed without synchronous fallback."""

    code = "RESOURCE_EXHAUSTED"

    def __init__(self, requested: int, available: int, timeout_s: float):
        self.requested = requested
        self.available = available
        self.timeout_s = timeout_s
        super().__init__(
            "rank-local object pool exhausted: "
            f"requested={requested}, available={available}, "
            f"timeout_s={timeout_s}"
        )


@dataclass(frozen=True)
class PoolConfig:
    initial_slots: int
    low_watermark: int
    high_watermark: int
    max_slots: int
    wait_timeout_s: float = 5.0

    def validate(self) -> None:
        values = (
            self.initial_slots,
            self.low_watermark,
            self.high_watermark,
            self.max_slots,
        )
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values):
            raise ValueError("object-pool slot configuration must be integers")
        if self.max_slots <= 0:
            raise ValueError("max_slots must be positive")
        if not 0 <= self.initial_slots <= self.max_slots:
            raise ValueError("initial_slots must be in [0, max_slots]")
        if not 0 <= self.low_watermark <= self.high_watermark <= self.max_slots:
            raise ValueError(
                "watermarks must satisfy 0 <= low <= high <= max_slots"
            )
        if self.wait_timeout_s < 0:
            raise ValueError("wait_timeout_s must be non-negative")


@dataclass(frozen=True)
class SlotRecord:
    """一个 chunk 的绑定记录。

    ``generation`` 是**分配**世代（每次绑定递增，提交记录校验用）；
    目标票据的失效判定走槽位**物理**世代（``chunk_slot_generation``，
    只在槽位文件被重建时变化），二者不可混用。
    """

    slot: int
    generation: int


@dataclass
class GpuFile:
    """一个**就绪**的 KV 存储对象：槽位文件 + 已打开的运行时句柄。

    槽位是稳定身份（分配/回收都不改名），因此句柄可以随槽位常驻。
    运行时把 chunk 绑定到 GpuFile 只是内存操作——不再 open（resolve +
    句柄构建 + 每 URI 一个线程）。文件被重建时 `generation` 递增，
    旧句柄随之作废。
    """

    slot: int
    uri: str
    size: int
    generation: int
    ticket: int = 0        # 0 = 尚未打开


class ObjectPool:
    """Filesystem-authoritative pool with an atomically replaced manifest."""

    def __init__(self, backend, config: PoolConfig, *, allocator_enabled=True):
        config.validate()
        self._backend = backend
        self.config = config
        self._allocator_enabled = bool(allocator_enabled)
        self._namespace = b""
        self._geometry: dict | None = None
        self._free: dict[int, int] = {}
        self._allocated: dict[bytes, SlotRecord] = {}
        self._next_slot = 0
        self._next_generation = 0
        self._manifest_dirty = False
        self._creating: set[int] = set()
        self._scrubbing: dict[int, int] = {}
        # 槽位物理世代（文件重建才递增）——目标票据的失效判定依据
        self._slot_generations: dict[int, int] = {}
        # 就绪 GpuFile 表：槽位号 → GpuFile（含已打开的运行时句柄）。
        # 句柄随槽位常驻，分配/回收都不动它——请求路径因此零 open。
        self._gpu_files: dict[int, GpuFile] = {}
        self._gpu_file_opener = None
        self._gpu_file_closer = None
        # 分配侧发现"就绪槽位不足"时置位，用于唤醒分配器去就绪化
        self._ready_shortfall = False
        self._configured = False
        self._stop = False
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition(threading.RLock())

    @property
    def configured(self) -> bool:
        return self._configured

    @property
    def manifest_path(self) -> Path:
        return self._backend.pool_manifest_path()

    def set_namespace(self, namespace: bytes) -> None:
        self._namespace = bytes(namespace)

    def configure(self, num_layers: int, segment_bytes: int) -> None:
        slot_bytes = num_layers * segment_bytes
        geometry = self._backend.pool_geometry(
            num_layers=num_layers,
            segment_bytes=segment_bytes,
            slot_bytes=slot_bytes,
        )
        with self._condition:
            if self._configured:
                if geometry != self._geometry:
                    raise RuntimeError("rank-local object-pool geometry changed")
                return
            self._geometry = geometry
            self._recover_locked(num_layers)
            target = min(
                self.config.initial_slots,
                self.config.max_slots - len(self._allocated),
            )
        while self.free_count < target:
            if self._create_one_sync() is None:
                raise PoolResourceExhausted(
                    target - self.free_count,
                    self.free_count,
                    0,
                )
        # 初始槽位在此刻一次就绪化（本调用在 worker 初始化期，不在
        # 请求路径上）。
        self._ready_gpu_files()
        with self._condition:
            self._configured = True
            self._persist_locked()
            if self._allocator_enabled:
                self._thread = threading.Thread(
                    target=self._allocator_loop,
                    name="tutti-object-pool-allocator",
                    daemon=True,
                )
                self._thread.start()
            self._request_refill_locked()

    @property
    def free_count(self) -> int:
        with self._condition:
            return len(self._free)

    @property
    def total_count(self) -> int:
        with self._condition:
            return len(self._free) + len(self._allocated) + len(self._creating)

    def generation(self, chunk_id: bytes) -> int | None:
        with self._condition:
            record = self._allocated.get(bytes(chunk_id))
            return None if record is None else record.generation

    def allocate(self, chunk_ids) -> tuple[bytes, ...]:
        missing = []
        seen = set()
        with self._condition:
            for value in chunk_ids:
                chunk_id = bytes(value)
                if chunk_id in seen or chunk_id in self._allocated:
                    continue
                seen.add(chunk_id)
                missing.append(chunk_id)
            if not missing:
                return ()
            deadline = time.monotonic() + self.config.wait_timeout_s
            self._request_refill_locked(force=True)
            while len(self._free) < len(missing):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PoolResourceExhausted(
                        len(missing), len(self._free), self.config.wait_timeout_s
                    )
                self._condition.wait(remaining)
                self._request_refill_locked(force=True)

            # 只从**已就绪**的 GpuFile 里取槽位：未就绪的槽位留给分配器
            # （它建完就就绪化），否则请求线程要替它 open——resolve +
            # 句柄构建 + 每 URI 一个线程，实测每个新绑定的 chunk 一次
            # 192B 分配与一对 pthread，压在层 0→1 的计算下发路径上。
            slots = self._ready_free_slots_locked(len(missing))
            if slots is None:
                # 只有分配器在跑时才值得等它就绪化；否则直接走兜底，
                # 避免在无分配器的部署（测试/单线程）里空等超时。
                if self._thread is not None:
                    self._ready_shortfall = True
                    wait_deadline = time.monotonic() + self.config.wait_timeout_s
                    while True:
                        self._condition.notify_all()
                        self._request_refill_locked(force=True)
                        slots = self._ready_free_slots_locked(len(missing))
                        if slots is not None:
                            break
                        remaining = wait_deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(min(remaining, 0.01))
                    self._ready_shortfall = False
                if slots is None:
                    # 兜底：就绪队列始终不足时退回未就绪槽位，正确性
                    # 不受影响（请求线程会补 open），只是慢。
                    slots = sorted(self._free)[:len(missing)]
            completed: list[tuple[bytes, int, int]] = []
            try:
                for chunk_id, slot in zip(missing, slots):
                    old_generation = self._free.pop(slot)
                    self._backend.pool_rename_group(
                        self._backend.pool_slot_paths(slot),
                        self._backend.pool_bound_paths(chunk_id, slot),
                    )
                    self._next_generation += 1
                    self._allocated[chunk_id] = SlotRecord(
                        slot, self._next_generation
                    )
                    completed.append((chunk_id, slot, old_generation))
            except Exception:
                for chunk_id, slot, old_generation in reversed(completed):
                    self._backend.pool_rename_group(
                        self._backend.pool_bound_paths(chunk_id, slot),
                        self._backend.pool_slot_paths(slot),
                    )
                    self._allocated.pop(chunk_id, None)
                    self._free[slot] = old_generation
                raise
            self._manifest_dirty = True
            # Commit validation is filesystem-authoritative. Persist the new
            # chunk -> slot/pool-generation mapping before payload IO can
            # publish a rank commit record.
            self._persist_locked()
            self._request_refill_locked()
            return tuple(missing)

    def recycle(self, chunk_ids) -> tuple[bytes, ...]:
        recycled = []
        with self._condition:
            for value in chunk_ids:
                chunk_id = bytes(value)
                record = self._allocated.get(chunk_id)
                if record is None:
                    continue
                self._backend.pool_remove_markers(chunk_id)
                self._backend.pool_rename_group(
                    self._backend.pool_bound_paths(chunk_id, record.slot),
                    self._backend.pool_slot_paths(record.slot),
                )
                self._allocated.pop(chunk_id, None)
                # The slot file is not free until its old payload has been
                # overwritten with real zeros and FIEMAP-validated by the
                # background allocator. This keeps zeroing/fsync off the
                # request thread while preventing stale bytes from allocation.
                self._scrubbing[record.slot] = record.generation
                recycled.append(chunk_id)
            if recycled:
                self._manifest_dirty = True
                self._condition.notify_all()
        return tuple(recycled)

    def abort_uncommitted(self, chunk_ids) -> None:
        pending = [
            bytes(chunk_id) for chunk_id in chunk_ids
            if not self._backend.pool_chunk_complete(bytes(chunk_id))
        ]
        self.recycle(pending)

    def slot_of(self, chunk_id: bytes) -> int | None:
        """已分配 chunk 的槽位号；未分配返回 None。

        纯内存查询，供布局把 chunk 解析为**稳定**的槽位文件路径。
        调用方是分配该 chunk 的同一请求线程，无需加锁。
        """
        record = self._allocated.get(bytes(chunk_id))
        return None if record is None else record.slot

    def allocation_count(self) -> int:
        """当前已分配的槽位数（容量检查用）。"""
        return len(self._allocated)

    def slot_generation(self, slot: int) -> int:
        """槽位物理世代（文件重建才递增）。"""
        return self._slot_generations.get(slot, 0)

    def chunk_slot_generation(self, chunk_id: bytes) -> int | None:
        """已分配 chunk 所在槽位的**当前**物理世代；未分配返回 None。

        读实时值而非绑定时快照：槽位若在绑定后被重建（文件换 inode），
        票据必须随之失效并重新 open。
        """
        record = self._allocated.get(bytes(chunk_id))
        if record is None:
            return None
        return self._slot_generations.get(record.slot, 0)

    def slot_ids(self) -> list[int]:
        """当前存在的槽位号（自由 + 已分配 + 在途），供启动期批量预开。"""
        with self._condition:
            return sorted(
                set(self._free) | set(self._allocated_slots())
                | self._creating | set(self._scrubbing)
            )

    def _allocated_slots(self) -> set[int]:
        return {record.slot for record in self._allocated.values()}

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join()
        with self._condition:
            self._thread = None
            if self._manifest_dirty:
                self._persist_locked()
        self._release_gpu_files()

    def snapshot(self) -> dict:
        with self._condition:
            return {
                "configured": self._configured,
                "free": len(self._free),
                "allocated": len(self._allocated),
                "creating": len(self._creating),
                "scrubbing": len(self._scrubbing),
                "thread_alive": bool(self._thread and self._thread.is_alive()),
                "slot_bytes": 0 if self._geometry is None
                else self._geometry["slot_bytes"],
            }

    def _recover_locked(self, num_layers: int) -> None:
        manifest = self._read_manifest()
        if manifest:
            if manifest.get("layout_version") != _MANIFEST_VERSION:
                raise RuntimeError("object-pool layout_version mismatch")
            if manifest.get("namespace") != self._namespace.hex():
                raise RuntimeError("object-pool namespace mismatch")
            if manifest.get("rank_geometry") != self._geometry:
                raise RuntimeError("object-pool rank geometry mismatch")
            if manifest.get("slot_bytes") != self._geometry["slot_bytes"]:
                raise RuntimeError("object-pool slot_bytes mismatch")
            if manifest.get("max_slots") != self.config.max_slots:
                raise RuntimeError("object-pool max_slots mismatch")
        manifest_free = {
            int(slot): int(generation)
            for slot, generation in (manifest.get("free", {}) if manifest else {}).items()
        }
        manifest_allocated = manifest.get("allocated", {}) if manifest else {}
        self._next_slot = int(manifest.get("next_slot", 0)) if manifest else 0
        self._next_generation = int(manifest.get("next_generation", 0)) if manifest else 0

        # 归属由 manifest 的 allocated 段裁决，文件系统只裁决"存在与
        # 完整"。候选槽位 = free/ 下实际存在的文件 ∪ manifest 记录过的
        # 槽位——稳定路径后端（file_per_chunk）二者重叠，改名后端
        # （striped）已绑定槽位的文件不在 free/ 下，靠 manifest 补齐。
        present_slots = set(self._backend.pool_discover_free_slots())
        by_slot: dict[int, tuple[bytes, int]] = {}
        for chunk_hex, record in (manifest_allocated or {}).items():
            try:
                chunk_id = bytes.fromhex(chunk_hex)
                slot = int(record["slot"])
                generation = int(record["generation"])
            except (KeyError, TypeError, ValueError):
                continue
            by_slot[slot] = (chunk_id, generation)
        self._free = {}
        self._allocated = {}
        for slot in sorted(present_slots | set(by_slot)):
            bound = by_slot.get(slot)
            if bound is not None:
                chunk_id, generation = bound
                bound_paths = self._backend.pool_bound_paths(chunk_id, slot)
                complete = self._validate_group(
                    bound_paths
                ) and self._backend.pool_chunk_complete(chunk_id, num_layers)
                if complete:
                    if generation <= 0:
                        self._next_generation += 1
                        generation = self._next_generation
                    self._allocated[chunk_id] = SlotRecord(slot, generation)
                    self._next_generation = max(self._next_generation, generation)
                    self._next_slot = max(self._next_slot, slot + 1)
                    continue
                # 分配记录存在但数据/层标记残缺：收回为自由槽位
                self._backend.pool_remove_markers(chunk_id)
                self._backend.pool_rename_group(
                    bound_paths, self._backend.pool_slot_paths(slot)
                )
            if slot not in present_slots:
                continue
            # 自由槽位（含崩溃时正在擦除的槽位）：启动期统一零化后再
            # 暴露给分配，杜绝陈旧字节被复用。
            try:
                self._backend.pool_zero_slot(slot, self._geometry["slot_bytes"])
            except OSError:
                continue
            if self._validate_group(self._backend.pool_slot_paths(slot)):
                self._free[slot] = manifest_free.get(slot, 0)
                self._next_slot = max(self._next_slot, slot + 1)
            else:
                self._backend.pool_remove_group(
                    self._backend.pool_slot_paths(slot)
                )
        if len(self._free) + len(self._allocated) > self.config.max_slots:
            raise RuntimeError("existing object pool exceeds max_slots")

    def _create_one_sync(self) -> int | None:
        """物化一个槽位，成功返回槽位号（失败返回 None）。"""
        with self._condition:
            if self._total_locked() >= self.config.max_slots:
                return None
            slot = self._next_slot
            self._next_slot += 1
            self._creating.add(slot)
        ok = False
        try:
            self._backend.pool_create_slot(slot, self._geometry["slot_bytes"])
            ok = self._validate_group(self._backend.pool_slot_paths(slot))
        finally:
            with self._condition:
                self._creating.discard(slot)
                if ok:
                    self._free[slot] = 0
                    # 新文件 = 新物理身份，此前针对该路径的目标票据作废
                    self._slot_generations[slot] = (
                        self._slot_generations.get(slot, 0) + 1
                    )
                    self._persist_locked()
                else:
                    self._backend.pool_remove_group(
                        self._backend.pool_slot_paths(slot)
                    )
                self._condition.notify_all()
        return slot if ok else None

    def set_gpu_file_opener(self, opener) -> None:
        """注册"就绪化"回调：``fn(uris) -> tickets``。

        池在建出槽位后调用它把文件打开（resolve + 句柄构建），把
        GpuFile 变成就绪态。调用点在初始化与后台分配器线程上，**不在
        请求路径**。
        """
        self._gpu_file_opener = opener

    def set_gpu_file_closer(self, closer) -> None:
        """注册关闭回调：``fn(tickets)``（池关闭时释放句柄）。"""
        self._gpu_file_closer = closer

    def _ready_gpu_files(self) -> None:
        """把尚无句柄的槽位文件批量就绪化（调用方：初始化 / 分配器）。"""
        opener = self._gpu_file_opener
        if opener is None:
            return
        with self._condition:
            slots = sorted(
                set(self._free)
                | {record.slot for record in self._allocated.values()}
            )
            pending = [slot for slot in slots if slot not in self._gpu_files]
            if not pending:
                return
            planned = [
                GpuFile(
                    slot=slot,
                    uri=str(self._backend.pool_slot_uri(slot)),
                    size=int(self._geometry["slot_bytes"]),
                    generation=self._slot_generations.get(slot, 0),
                )
                for slot in pending
            ]
        try:
            tickets = opener([item.uri for item in planned])
        except Exception:
            return
        if len(tickets) != len(planned):
            return
        with self._condition:
            for item, ticket in zip(planned, tickets):
                item.ticket = int(ticket)
                self._gpu_files[item.slot] = item
            self._condition.notify_all()

    def mark_gpu_files_ready(self) -> None:
        """就绪化当前全部槽位（宿主注册 opener 后调用一次）。

        池的 configure 早于宿主注册 opener，因此初始槽位需要宿主显式
        触发一次就绪化；此后新增槽位由分配器自动就绪化。
        """
        self._ready_gpu_files()

    def _ready_free_slots_locked(self, count: int) -> list[int] | None:
        """最小的 ``count`` 个**已就绪**自由槽位；不足返回 None。"""
        ready = [
            slot for slot in sorted(self._free)
            if self._gpu_files.get(slot) is not None
            and self._gpu_files[slot].ticket
        ]
        if len(ready) < count:
            return None
        return ready[:count]

    def ticket_of_slot(self, slot: int) -> int:
        """槽位的运行时句柄（0 = 尚未就绪）。纯内存查询。"""
        entry = self._gpu_files.get(slot)
        return 0 if entry is None else entry.ticket

    def gpu_files(self) -> list[GpuFile]:
        with self._condition:
            return list(self._gpu_files.values())

    def _release_gpu_files(self) -> None:
        closer = self._gpu_file_closer
        with self._condition:
            tickets = [f.ticket for f in self._gpu_files.values() if f.ticket]
            self._gpu_files = {}
        if closer is not None and tickets:
            try:
                closer(tickets)
            except Exception:
                pass

    def _allocator_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or (
                        self._scrubbing or
                        # 分配侧发现"就绪槽位不足"时也要醒来去就绪化
                        self._ready_shortfall or
                        len(self._free) <= self.config.low_watermark and
                        self._total_locked() < self.config.max_slots
                    )
                )
                if self._scrubbing:
                    scrub_slots = list(self._scrubbing.items())
                else:
                    scrub_slots = []
                if self._stop and not scrub_slots:
                    return
            for slot, generation in scrub_slots:
                if self._scrub_one(slot, generation):
                    with self._condition:
                        if self._scrubbing.get(slot) == generation:
                            self._scrubbing.pop(slot, None)
                            self._free[slot] = generation
                            self._manifest_dirty = True
                            self._persist_locked()
                            self._condition.notify_all()
                else:
                    with self._condition:
                        self._scrubbing.pop(slot, None)
                        self._condition.notify_all()
            with self._condition:
                if self._stop:
                    return
                target = min(
                    self.config.high_watermark,
                    self.config.max_slots - len(self._allocated),
                )
            while self.free_count < target:
                with self._condition:
                    if self._stop or self._total_locked() >= self.config.max_slots:
                        break
                if self._create_one_sync() is None:
                    break
            # 新槽位在**本后台线程**就绪化：池扩展出的槽位因此不会在
            # 请求路径上首次 open（这正是 r1+ 仍有上百次句柄分配的
            # 原因）。就绪化后再清短标志，避免分配侧空等。
            self._ready_gpu_files()
            with self._condition:
                self._ready_shortfall = False
                self._condition.notify_all()

    def _scrub_one(self, slot: int, generation: int) -> bool:
        try:
            self._backend.pool_zero_slot(slot, self._geometry["slot_bytes"])
            return self._validate_group(self._backend.pool_slot_paths(slot))
        except Exception:
            return False

    def _request_refill_locked(self, force=False) -> None:
        if not self._allocator_enabled:
            return
        if force or len(self._free) <= self.config.low_watermark:
            self._condition.notify_all()

    def _total_locked(self) -> int:
        return (
            len(self._free) + len(self._allocated) +
            len(self._creating) + len(self._scrubbing)
        )

    def _validate_group(self, paths) -> bool:
        sizes = self._backend.pool_physical_sizes(self._geometry["slot_bytes"])
        return len(paths) == len(sizes) and all(
            _fiemap_covers(path, size) for path, size in zip(paths, sizes)
        )

    def _read_manifest(self) -> dict:
        try:
            return json.loads(self.manifest_path.read_text("utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid object-pool manifest: {exc}") from exc

    def _persist_locked(self) -> None:
        if self._geometry is None:
            return
        payload = {
            "layout_version": _MANIFEST_VERSION,
            "namespace": self._namespace.hex(),
            "rank_geometry": self._geometry,
            "slot_bytes": self._geometry["slot_bytes"],
            "max_slots": self.config.max_slots,
            "next_slot": self._next_slot,
            "next_generation": self._next_generation,
            "free": {str(slot): generation
                     for slot, generation in sorted(self._free.items())},
            "allocated": {
                chunk_id.hex(): {
                    "slot": record.slot,
                    "generation": record.generation,
                }
                for chunk_id, record in sorted(self._allocated.items())
            },
        }
        path = self.manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}"
        )
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        self._manifest_dirty = False


def _fiemap_covers(path: Path, expected_size: int) -> bool:
    try:
        stat = path.stat()
        if stat.st_size != expected_size or expected_size <= 0:
            return False
        buffer = bytearray(
            _FIEMAP_HEADER.size + _FIEMAP_EXTENT.size * _FIEMAP_EXTENT_COUNT
        )
        _FIEMAP_HEADER.pack_into(
            buffer, 0, 0, expected_size, 0, 0, _FIEMAP_EXTENT_COUNT, 0
        )
        with path.open("rb", buffering=0) as handle:
            fcntl.ioctl(handle.fileno(), _FIEMAP_IOCTL, buffer, True)
        _, _, _, mapped, _, _ = _FIEMAP_HEADER.unpack_from(buffer, 0)
        if mapped <= 0:
            return False
        cursor = 0
        for index in range(mapped):
            values = _FIEMAP_EXTENT.unpack_from(
                buffer, _FIEMAP_HEADER.size + index * _FIEMAP_EXTENT.size
            )
            logical, _, length = values[:3]
            flags = values[5]
            if flags & (_FIEMAP_EXTENT_DELALLOC | _FIEMAP_EXTENT_UNWRITTEN):
                return False
            if logical > cursor:
                return False
            cursor = max(cursor, logical + length)
            if cursor >= expected_size:
                return True
            if flags & _FIEMAP_EXTENT_LAST:
                break
        return cursor >= expected_size
    except OSError:
        return False

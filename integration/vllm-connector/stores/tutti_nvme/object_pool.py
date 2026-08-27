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
    slot: int
    generation: int


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
            if not self._create_one_sync():
                raise PoolResourceExhausted(
                    target - self.free_count,
                    self.free_count,
                    0,
                )
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

            reservations = [
                (chunk_id, min(self._free)) for chunk_id in missing
            ]
            # Consume distinct slots deterministically.
            slots = sorted(self._free)[:len(missing)]
            reservations = list(zip(missing, slots))
            completed: list[tuple[bytes, int, int]] = []
            try:
                for chunk_id, slot in reservations:
                    old_generation = self._free.pop(slot)
                    self._backend.pool_rename_group(
                        self._backend.pool_slot_paths(slot),
                        self._backend.pool_chunk_paths(chunk_id),
                    )
                    self._next_generation += 1
                    generation = self._next_generation
                    self._allocated[chunk_id] = SlotRecord(slot, generation)
                    completed.append((chunk_id, slot, old_generation))
            except Exception:
                for chunk_id, slot, old_generation in reversed(completed):
                    self._backend.pool_rename_group(
                        self._backend.pool_chunk_paths(chunk_id),
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
                    self._backend.pool_chunk_paths(chunk_id),
                    self._backend.pool_slot_paths(record.slot),
                )
                self._allocated.pop(chunk_id, None)
                # The renamed file is not free until its old payload has been
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

        free_slots = self._backend.pool_discover_free_slots()
        chunks = self._backend.pool_discover_chunks()
        self._free = {}
        self._allocated = {}
        for slot in sorted(free_slots):
            # A crash may leave a just-renamed victim in free/ before the
            # asynchronous scrub completed. Re-zero every discovered free
            # slot during startup before exposing it to allocation.
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

        used_slots = set(self._free)
        for chunk_id in sorted(chunks):
            record = manifest_allocated.get(chunk_id.hex(), {})
            slot = int(record.get("slot", self._next_slot))
            generation = int(record.get("generation", 0))
            if slot in used_slots:
                slot = self._next_slot
            self._next_slot = max(self._next_slot, slot + 1)
            used_slots.add(slot)
            paths = self._backend.pool_chunk_paths(chunk_id)
            valid = self._validate_group(paths)
            complete = valid and self._backend.pool_chunk_complete(
                chunk_id, num_layers
            )
            if complete:
                if generation <= 0:
                    self._next_generation += 1
                    generation = self._next_generation
                self._allocated[chunk_id] = SlotRecord(slot, generation)
                continue
            self._backend.pool_remove_markers(chunk_id)
            if valid:
                self._backend.pool_rename_group(
                    paths, self._backend.pool_slot_paths(slot)
                )
                self._free[slot] = max(generation, 0)
            else:
                self._backend.pool_remove_group(paths)
        if len(self._free) + len(self._allocated) > self.config.max_slots:
            raise RuntimeError("existing object pool exceeds max_slots")

    def _create_one_sync(self) -> bool:
        with self._condition:
            if self._total_locked() >= self.config.max_slots:
                return False
            slot = self._next_slot
            self._next_slot += 1
            self._creating.add(slot)
        ok = False
        try:
            self._backend.pool_create_slot(slot, self._geometry["slot_bytes"])
            ok = self._validate_group(self._backend.pool_slot_paths(slot))
            return ok
        finally:
            with self._condition:
                self._creating.discard(slot)
                if ok:
                    self._free[slot] = 0
                    self._persist_locked()
                else:
                    self._backend.pool_remove_group(
                        self._backend.pool_slot_paths(slot)
                    )
                self._condition.notify_all()

    def _allocator_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or (
                        self._scrubbing or
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
                if not self._create_one_sync():
                    break

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

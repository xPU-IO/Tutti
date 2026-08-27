"""Striped file-per-chunk layout for the optional NVMe target.

Markers stay in the store root.  Data for chunk ``<name>`` is represented by
one real, fully written shard file per mount::

    <mount_i>/striped/<name>.shard<i>

The runtime exposes those files as one logical ``striped://`` target.  A
chunk's first four identity bytes select its deterministic shard rotation.
When ``stripe_unit`` is larger than a segment, the segment maps to one shard
and the backing files are rounded up to the runtime's complete stripe round.
"""

from __future__ import annotations

import os
from pathlib import Path

from .layout import (
    _MANIFEST_NAME,
    _OBJECT_POOL_MANIFEST,
    _append_real_zeros,
    _bump_scan_generation,
    _read_scan_generation,
    _rewrite_real_zeros,
    decode_io_key,
)
from .commit import marker_generation, remove_rank_commit

_IO_PAGE_BYTES = 4096
_MARKER_SUFFIX = ".ok"
_SHARD_SEPARATOR = ".shard"


class StripedLayout:
    """Layout-compatible filesystem policy for a striped logical target."""

    def __init__(self, root: str | os.PathLike, segment_bytes: int,
                 mounts, stripe_unit: int):
        if segment_bytes <= 0:
            raise ValueError(f"segment_bytes 必须为正数，得到 {segment_bytes}")
        self.root = Path(root)
        self.segment_bytes = segment_bytes
        self.layer_span: int | None = None

        if isinstance(mounts, (str, bytes, os.PathLike)):
            raise ValueError("striped mounts 必须是至少两个挂载点的序列")
        try:
            normalized_mounts = [str(Path(mount).resolve()) for mount in mounts]
        except TypeError as exc:
            raise ValueError("striped mounts 必须是至少两个挂载点的序列") from exc
        if len(normalized_mounts) < 2:
            raise ValueError("striped target 至少需要两个挂载点")
        if any(not mount for mount in normalized_mounts):
            raise ValueError("striped mounts 不能包含空路径")
        if any(any(ch in mount for ch in ",?&") for mount in normalized_mounts):
            raise ValueError("striped mount 路径不能包含 URI 分隔符 , ? &")
        if stripe_unit <= 0:
            raise ValueError(f"stripe_unit 必须为正数，得到 {stripe_unit}")
        if stripe_unit % _IO_PAGE_BYTES != 0:
            raise ValueError("stripe_unit 必须是 4 KiB 的正倍数")

        self.mounts = tuple(normalized_mounts)
        self.stripe_unit = stripe_unit
        self._meta_dir = self.root / "meta"
        self._object_pool = None
        self._scan_signature: tuple[int, int, bytes] | None = None
        self._scan_cache: set[bytes] = set()

    @property
    def num_shards(self) -> int:
        return len(self.mounts)

    def set_layer_span(self, num_layers: int) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers 必须为正数，得到 {num_layers}")
        self.layer_span = num_layers
        if self._object_pool is not None:
            self._object_pool.configure(num_layers, self.segment_bytes)

    def attach_object_pool(self, pool) -> None:
        self._object_pool = pool

    def check_namespace(self, namespace: bytes) -> bool:
        path = self.root / _MANIFEST_NAME
        try:
            stored = path.read_bytes()
        except FileNotFoundError:
            self.root.mkdir(parents=True, exist_ok=True)
            path.write_bytes(namespace)
            if self._object_pool is not None:
                self._object_pool.set_namespace(namespace)
            return True
        matches = stored == namespace
        if matches and self._object_pool is not None:
            self._object_pool.set_namespace(namespace)
        return matches

    # ---------- naming ----------

    def marker_file(self, io_key: bytes) -> Path:
        return self._meta_dir / (bytes(io_key).hex() + _MARKER_SUFFIX)

    def marker_generation(self) -> str:
        return marker_generation(self._meta_dir)

    def remove_rank_commit(self, chunk_id: bytes) -> None:
        remove_rank_commit(self.root, chunk_id)

    def shard_rotation(self, chunk_id: bytes) -> int:
        return int.from_bytes(bytes(chunk_id)[:4], "little") % self.num_shards

    def shard_file(self, chunk_id: bytes, shard: int) -> Path:
        if not 0 <= shard < self.num_shards:
            raise ValueError(f"shard 索引越界：{shard}")
        name = bytes(chunk_id).hex()
        return Path(self.mounts[shard]) / "striped" / (
            name + _SHARD_SEPARATOR + str(shard)
        )

    def target_uri(self, chunk_id: bytes) -> str:
        name = bytes(chunk_id).hex()
        mounts = ",".join(self.mounts)
        rotation = self.shard_rotation(chunk_id)
        return (
            f"striped://{name}?devs={mounts}&unit={self.stripe_unit}"
            f"&rot={rotation}"
        )

    def target_size(self, chunk_id: bytes) -> int:
        """Match StripedLocalNvmePayload's min-shard logical-size rule."""
        sizes = []
        for shard in range(self.num_shards):
            try:
                sizes.append(self.shard_file(chunk_id, shard).stat().st_size)
            except OSError:
                return 0
        if not sizes:
            return 0
        complete_units = min(sizes) // self.stripe_unit
        return complete_units * self.stripe_unit * self.num_shards

    # ---------- lifecycle ----------

    def ensure_dirs(self) -> None:
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        for mount in self.mounts:
            (Path(mount) / "striped").mkdir(parents=True, exist_ok=True)
            if self._object_pool is not None:
                (Path(mount) / "striped" / "free").mkdir(
                    parents=True, exist_ok=True
                )

    def scan(self) -> set[bytes]:
        live: set[bytes] = set()
        if not self._meta_dir.is_dir():
            self._scan_signature = None
            self._scan_cache = set()
            return live
        try:
            stat = self._meta_dir.stat()
            signature = (
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                _read_scan_generation(self._meta_dir),
            )
        except OSError:
            self._scan_signature = None
            self._scan_cache = set()
            return live
        if signature == self._scan_signature:
            return set(self._scan_cache)
        for entry in self._meta_dir.iterdir():
            if not entry.name.endswith(_MARKER_SUFFIX):
                continue
            try:
                live.add(bytes.fromhex(entry.name[:-len(_MARKER_SUFFIX)]))
            except ValueError:
                continue
        self._scan_signature = signature
        self._scan_cache = live
        return set(live)

    def chunk_file_count(self) -> int:
        """Count logical chunks, not the N physical shard files."""
        names: set[str] = set()
        for mount in self.mounts:
            directory = Path(mount) / "striped"
            if not directory.is_dir():
                continue
            for entry in directory.iterdir():
                if not entry.is_file() or _SHARD_SEPARATOR not in entry.name:
                    continue
                name, shard = entry.name.rsplit(_SHARD_SEPARATOR, 1)
                if shard.isdigit() and name:
                    names.add(name)
        return len(names)

    # ---------- data plane ----------

    def prepare_put(self, io_keys, capacity_chunks: int):
        grouped: dict[bytes, int] = {}
        decoded: dict[bytes, tuple[bytes, int]] = {}
        for io_key in io_keys:
            chunk_id, layer = decode_io_key(io_key)
            decoded[bytes(io_key)] = (chunk_id, layer)
            grouped[chunk_id] = max(grouped.get(chunk_id, 0), layer + 1)

        existing = self.chunk_file_count()
        new_chunks = sum(
            1 for chunk_id in grouped
            if not any(self.shard_file(chunk_id, shard).exists()
                       for shard in range(self.num_shards))
        )
        if existing + new_chunks > capacity_chunks:
            raise ValueError(
                f"chunk 容量不足：现有 {existing} + 新增 {new_chunks}"
                f" > 上限 {capacity_chunks}"
            )

        if self._object_pool is not None and self._object_pool.configured:
            if self.layer_span is None:
                raise RuntimeError("object pool requires complete layer geometry")
            if any(layer_count > self.layer_span
                   for layer_count in grouped.values()):
                raise ValueError("io_key layer exceeds configured layer span")
            self._object_pool.allocate(grouped)
            return decoded

        for chunk_id, layer_count in grouped.items():
            logical_need = layer_count * self.segment_bytes
            if self.layer_span is not None:
                logical_need = max(
                    logical_need, self.layer_span * self.segment_bytes
                )
            # Runtime derives logical size from complete rounds common to all
            # shards, so round each backing file up to one complete round.
            rounds = (logical_need + self.num_shards * self.stripe_unit - 1) // (
                self.num_shards * self.stripe_unit
            )
            physical_need = max(self.stripe_unit, rounds * self.stripe_unit)
            for shard in range(self.num_shards):
                path = self.shard_file(chunk_id, shard)
                current = path.stat().st_size if path.exists() else 0
                if current < physical_need:
                    _append_real_zeros(path, current, physical_need)
        return decoded

    def commit_layers(self, io_keys) -> None:
        for io_key in io_keys:
            self.marker_file(io_key).touch(exist_ok=True)
        _bump_scan_generation(self._meta_dir)
        self._scan_signature = None

    def drop(self, io_keys) -> None:
        touched_chunks: set[bytes] = set()
        for io_key in io_keys:
            chunk_id, _ = decode_io_key(io_key)
            self.marker_file(io_key).unlink(missing_ok=True)
            touched_chunks.add(chunk_id)
        released = []
        for chunk_id in touched_chunks:
            if any(self._meta_dir.glob(chunk_id.hex() + "*" + _MARKER_SUFFIX)):
                continue
            released.append(chunk_id)
        if self._object_pool is not None and self._object_pool.configured:
            self._object_pool.recycle(released)
        else:
            for chunk_id in released:
                for shard in range(self.num_shards):
                    self.shard_file(chunk_id, shard).unlink(missing_ok=True)
        for chunk_id in released:
            self.remove_rank_commit(chunk_id)
        _bump_scan_generation(self._meta_dir)
        self._scan_signature = None

    def releasable_chunks(self, io_keys) -> set[bytes]:
        dropping: dict[bytes, set[bytes]] = {}
        for io_key in io_keys:
            chunk_id, _ = decode_io_key(io_key)
            dropping.setdefault(chunk_id, set()).add(bytes(io_key))
        released = set()
        for chunk_id, removed in dropping.items():
            remaining = {
                bytes.fromhex(path.name[:-len(_MARKER_SUFFIX)])
                for path in self._meta_dir.glob(
                    chunk_id.hex() + "*" + _MARKER_SUFFIX
                )
            } - removed
            if not remaining:
                released.add(chunk_id)
        return released

    def target_generation(self, chunk_id: bytes) -> int:
        if self._object_pool is None or not self._object_pool.configured:
            return 0
        generation = self._object_pool.generation(chunk_id)
        return -1 if generation is None else generation

    def abort_uncommitted(self, chunk_ids) -> None:
        if self._object_pool is not None and self._object_pool.configured:
            self._object_pool.abort_uncommitted(chunk_ids)

    def close_object_pool(self) -> None:
        if self._object_pool is not None:
            self._object_pool.close()

    def object_pool_snapshot(self) -> dict | None:
        return None if self._object_pool is None else self._object_pool.snapshot()

    # ---------- object-pool backend ----------

    def pool_manifest_path(self) -> Path:
        return self.root / _OBJECT_POOL_MANIFEST

    def pool_geometry(self, *, num_layers, segment_bytes, slot_bytes) -> dict:
        physical = self.pool_physical_sizes(slot_bytes)[0]
        return {
            "layout": "striped",
            "num_layers": num_layers,
            "segment_bytes": segment_bytes,
            "slot_bytes": slot_bytes,
            "num_shards": self.num_shards,
            "stripe_unit": self.stripe_unit,
            "physical_slot_bytes_per_shard": physical,
        }

    def pool_physical_sizes(self, slot_bytes: int) -> tuple[int, ...]:
        rounds = (
            slot_bytes + self.num_shards * self.stripe_unit - 1
        ) // (self.num_shards * self.stripe_unit)
        physical = max(self.stripe_unit, rounds * self.stripe_unit)
        return (physical,) * self.num_shards

    def pool_slot_paths(self, slot: int) -> tuple[Path, ...]:
        return tuple(
            Path(mount) / "striped" / "free" /
            f"{slot:08d}{_SHARD_SEPARATOR}{shard}"
            for shard, mount in enumerate(self.mounts)
        )

    def pool_chunk_paths(self, chunk_id: bytes) -> tuple[Path, ...]:
        return tuple(
            self.shard_file(chunk_id, shard)
            for shard in range(self.num_shards)
        )

    def pool_create_slot(self, slot: int, slot_bytes: int) -> None:
        paths = self.pool_slot_paths(slot)
        sizes = self.pool_physical_sizes(slot_bytes)
        try:
            for path, size in zip(paths, sizes):
                path.parent.mkdir(parents=True, exist_ok=True)
                _append_real_zeros(path, 0, size)
        except Exception:
            self.pool_remove_group(paths)
            raise

    def pool_zero_slot(self, slot: int, slot_bytes: int) -> None:
        paths = self.pool_slot_paths(slot)
        sizes = self.pool_physical_sizes(slot_bytes)
        try:
            for path, size in zip(paths, sizes):
                _rewrite_real_zeros(path, size)
        except Exception:
            raise

    def pool_discover_free_slots(self) -> set[int]:
        found: dict[int, set[int]] = {}
        for shard, mount in enumerate(self.mounts):
            directory = Path(mount) / "striped" / "free"
            if not directory.is_dir():
                continue
            for path in directory.glob("*" + _SHARD_SEPARATOR + str(shard)):
                prefix = path.name[:-len(_SHARD_SEPARATOR + str(shard))]
                try:
                    slot = int(prefix)
                except ValueError:
                    continue
                found.setdefault(slot, set()).add(shard)
        return {
            slot for slot, shards in found.items()
            if len(shards) == self.num_shards
        }

    def pool_discover_chunks(self) -> set[bytes]:
        chunks = set()
        for shard, mount in enumerate(self.mounts):
            directory = Path(mount) / "striped"
            if not directory.is_dir():
                continue
            for path in directory.glob("*" + _SHARD_SEPARATOR + str(shard)):
                name = path.name[:-len(_SHARD_SEPARATOR + str(shard))]
                try:
                    chunks.add(bytes.fromhex(name))
                except ValueError:
                    continue
        return chunks

    def pool_chunk_complete(self, chunk_id: bytes,
                            num_layers: int | None = None) -> bool:
        layers = self.layer_span if num_layers is None else num_layers
        if not layers:
            return False
        return all(
            self.marker_file(bytes(chunk_id) + layer.to_bytes(2, "little")).exists()
            for layer in range(layers)
        )

    def pool_remove_markers(self, chunk_id: bytes) -> None:
        for path in self._meta_dir.glob(
            bytes(chunk_id).hex() + "*" + _MARKER_SUFFIX
        ):
            path.unlink(missing_ok=True)
        self.remove_rank_commit(chunk_id)

    @staticmethod
    def pool_rename_group(sources, destinations) -> None:
        if len(sources) != len(destinations):
            raise RuntimeError("striped object-pool group shape mismatch")
        if not all(path.exists() for path in sources):
            raise RuntimeError("striped object-pool source group incomplete")
        if any(path.exists() for path in destinations):
            raise RuntimeError("striped object-pool destination already exists")
        completed = []
        try:
            for source, destination in zip(sources, destinations):
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
                completed.append((source, destination))
        except Exception:
            for source, destination in reversed(completed):
                os.replace(destination, source)
            raise

    @staticmethod
    def pool_remove_group(paths) -> None:
        for path in paths:
            path.unlink(missing_ok=True)

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

from .layout import _append_real_zeros, decode_io_key

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

    @property
    def num_shards(self) -> int:
        return len(self.mounts)

    def set_layer_span(self, num_layers: int) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers 必须为正数，得到 {num_layers}")
        self.layer_span = num_layers

    # ---------- naming ----------

    def marker_file(self, io_key: bytes) -> Path:
        return self._meta_dir / (bytes(io_key).hex() + _MARKER_SUFFIX)

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

    def scan(self) -> set[bytes]:
        live: set[bytes] = set()
        if not self._meta_dir.is_dir():
            return live
        for entry in self._meta_dir.iterdir():
            if not entry.name.endswith(_MARKER_SUFFIX):
                continue
            try:
                live.add(bytes.fromhex(entry.name[:-len(_MARKER_SUFFIX)]))
            except ValueError:
                continue
        return live

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

    def drop(self, io_keys) -> None:
        touched_chunks: set[bytes] = set()
        for io_key in io_keys:
            chunk_id, _ = decode_io_key(io_key)
            self.marker_file(io_key).unlink(missing_ok=True)
            touched_chunks.add(chunk_id)
        for chunk_id in touched_chunks:
            if any(self._meta_dir.glob(chunk_id.hex() + "*" + _MARKER_SUFFIX)):
                continue
            for shard in range(self.num_shards):
                self.shard_file(chunk_id, shard).unlink(missing_ok=True)

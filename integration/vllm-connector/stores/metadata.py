"""Scheduler-only metadata stores.

These stores expose only capacity, namespace ownership, marker scanning, and
eviction.  They intentionally have no runtime, buffer registration, stream,
target, or data-transfer API.
"""

from __future__ import annotations

import json
import logging

from .tutti_nvme.layout import Layout
from .tutti_nvme.striped_layout import StripedLayout
from .tutti_nvme.commit import remove_rank_commit, scan_rank_commits

_LOG = logging.getLogger(__name__)


class MemoryMetadataStore:
    """Process-local metadata store used by non-persistent test deployments."""

    def __init__(self, num_chunks: int, segment_bytes: int, **_ignored) -> None:
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks!r}")
        if segment_bytes <= 0:
            raise ValueError(
                f"segment_bytes must be positive, got {segment_bytes!r}"
            )
        self._num_chunks = num_chunks
        self._segment_bytes = segment_bytes
        self._live: set[bytes] = set()
        self._opened = False

    @property
    def capacity_chunks(self) -> int:
        return self._num_chunks

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False
        self._live.clear()

    def scan(self):
        self._require_open()
        return sorted(self._live)

    def drop(self, keys) -> None:
        self._require_open()
        self._live.difference_update(bytes(key) for key in keys)

    def set_key_namespace(self, _namespace: bytes) -> None:
        if self._opened:
            raise RuntimeError("namespace must be configured before open")

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("metadata store is not open")


class TuttiMetadataStore:
    """Marker/manifest-only view of a Tutti NVMe pool."""

    def __init__(
        self,
        root,
        num_chunks: int,
        segment_bytes: int,
        *,
        layout="file_per_chunk",
        mounts=None,
        stripe_unit=None,
        preset=None,
        rank_options=None,
        tp_size: int = 1,
        **_data_plane_options,
    ) -> None:
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks!r}")
        if segment_bytes <= 0:
            raise ValueError(
                f"segment_bytes must be positive, got {segment_bytes!r}"
            )
        self._num_chunks = num_chunks
        self._segment_bytes = segment_bytes
        self._key_namespace: bytes | None = None
        self._opened = False
        self._tp_size = int(tp_size)
        if self._tp_size <= 0:
            raise ValueError("tp_size must be positive")
        if rank_options is None:
            rank_options = [{
                "root": root,
                "layout": layout,
                "mounts": mounts,
                "stripe_unit": stripe_unit,
                "preset": preset,
            }]
        if len(rank_options) != self._tp_size:
            raise ValueError("rank_options must contain one entry per TP rank")
        self._layouts = {
            rank: _metadata_layout(options, segment_bytes)
            for rank, options in enumerate(rank_options)
        }
        roots = [str(item.root) for item in self._layouts.values()]
        if self._tp_size > 1 and len(set(roots)) != self._tp_size:
            raise ValueError("every TP rank requires a distinct metadata root")
        self._layout = self._layouts[0]
        self._namespace_matches = {rank: True for rank in self._layouts}

    @property
    def capacity_chunks(self) -> int:
        return self._num_chunks

    def set_key_namespace(self, namespace: bytes) -> None:
        if self._opened:
            raise RuntimeError("namespace must be configured before open")
        self._key_namespace = bytes(namespace)

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("metadata store is already open")
        for rank, layout in self._layouts.items():
            layout.ensure_dirs()
            if self._key_namespace is not None:
                self._namespace_matches[rank] = layout.check_namespace(
                    self._key_namespace
                )
                if not self._namespace_matches[rank]:
                    _LOG.warning(
                        "rank %d pool %s namespace mismatch; all-rank lookup "
                        "fails closed",
                        rank, layout.root,
                    )
        self._opened = True
        self._cleanup_incomplete_commits()

    def close(self) -> None:
        self._opened = False

    def scan(self):
        self._require_open()
        if not all(self._namespace_matches.values()):
            return []
        marker_sets = {
            rank: set(layout.scan()) for rank, layout in self._layouts.items()
        }
        valid = self._valid_all_rank_chunks(marker_sets)
        return sorted(
            chunk_key + layer.to_bytes(2, "little")
            for chunk_key in valid
            for layer in range(self._record_num_layers(chunk_key))
        )

    def drop(self, keys) -> None:
        self._require_open()
        self._layout.drop(keys)

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("metadata store is not open")

    def _valid_all_rank_chunks(self, marker_sets) -> set[bytes]:
        commits = {
            rank: scan_rank_commits(layout.root)
            for rank, layout in self._layouts.items()
        }
        candidates = set().union(*(set(items) for items in commits.values()))
        valid = set()
        namespace = "" if self._key_namespace is None else self._key_namespace.hex()
        for chunk_key in candidates:
            records = [commits[rank].get(chunk_key) for rank in range(self._tp_size)]
            if any(record is None for record in records):
                continue
            generations = {record.generation for record in records}
            if len(generations) != 1:
                continue
            first = records[0]
            expected_layers = set(range(first.num_layers))
            if first.slot_bytes != first.num_layers * self._segment_bytes:
                continue
            accepted = True
            for rank, record in enumerate(records):
                if (
                    record.namespace != namespace
                    or record.chunk_key != chunk_key.hex()
                    or record.rank_id != rank
                    or record.num_layers != first.num_layers
                    or record.slot_bytes != first.slot_bytes
                    or not record.marker_generation
                    or not self._pool_manifest_matches(
                        rank, chunk_key, record.pool_generation,
                        record.num_layers, record.slot_bytes,
                    )
                ):
                    accepted = False
                    break
                layers = {
                    int.from_bytes(io_key[16:], "little")
                    for io_key in marker_sets[rank]
                    if io_key[:16] == chunk_key and len(io_key) == 18
                }
                if not layers >= expected_layers:
                    accepted = False
                    break
            if accepted:
                valid.add(chunk_key)
        return valid

    def _record_num_layers(self, chunk_key: bytes) -> int:
        record = scan_rank_commits(self._layouts[0].root).get(chunk_key)
        return 0 if record is None else record.num_layers

    def _pool_manifest_matches(self, rank, chunk_key, pool_generation,
                               num_layers, slot_bytes) -> bool:
        path = self._layouts[rank].pool_manifest_path()
        try:
            manifest = json.loads(path.read_text("utf-8"))
            allocation = manifest["allocated"][chunk_key.hex()]
            geometry = manifest["rank_geometry"]
            return (
                manifest.get("namespace") == self._key_namespace.hex()
                and int(manifest.get("slot_bytes", -1)) == slot_bytes
                and int(geometry.get("num_layers", -1)) == num_layers
                and int(geometry.get("slot_bytes", -1)) == slot_bytes
                and int(allocation.get("generation", -1)) == pool_generation
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def _cleanup_incomplete_commits(self) -> None:
        """Restart cleanup removes commit records only, never rank payload."""
        marker_sets = {
            rank: set(layout.scan()) for rank, layout in self._layouts.items()
        }
        valid = self._valid_all_rank_chunks(marker_sets)
        commits = {
            rank: scan_rank_commits(layout.root)
            for rank, layout in self._layouts.items()
        }
        candidates = set().union(*(set(items) for items in commits.values()))
        for chunk_key in candidates - valid:
            for layout in self._layouts.values():
                remove_rank_commit(layout.root, chunk_key)


def create_metadata_store(type_name: str, options: dict):
    """Construct a scheduler store without importing a data-plane store."""
    if type_name == "memory":
        return MemoryMetadataStore(**options)
    if type_name == "tutti_nvme":
        return TuttiMetadataStore(**options)
    raise ValueError(f"store type {type_name!r} has no metadata-only client")


def _preset_mounts(preset):
    if not isinstance(preset, dict):
        return None
    devices = preset.get("devices")
    if not isinstance(devices, (list, tuple)):
        return None
    mounts = []
    for device in devices:
        if not isinstance(device, dict) or not device.get("mount_path"):
            return None
        mounts.append(device["mount_path"])
    return mounts or None


def _metadata_layout(options, segment_bytes):
    root = options.get("root")
    layout = options.get("layout", "file_per_chunk")
    if layout in (None, "file_per_chunk", "file"):
        return Layout(root, segment_bytes)
    if layout == "striped":
        mounts = options.get("mounts")
        if mounts is None:
            mounts = _preset_mounts(options.get("preset"))
        stripe_unit = options.get("stripe_unit")
        if stripe_unit is None:
            raise ValueError("striped metadata store requires stripe_unit")
        return StripedLayout(root, segment_bytes, mounts, stripe_unit)
    raise ValueError(f"unknown tutti_nvme layout: {layout!r}")

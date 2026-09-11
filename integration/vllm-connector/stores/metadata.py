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
        """全 rank 层标记齐备、且池 manifest 认可归属的 chunk。

        运行时驻留由内存权威索引门禁（worker→scheduler 的
        TuttiWorkerMetadata），盘上只保留两样东西：层标记（数据完整性）
        与池 manifest（槽位归属 + 几何 + 命名空间）。此前的 rank 提交
        凭证（commits/*.commit.json）与 manifest 信息高度重叠且每文件
        fsync，已整体删除。
        """
        candidates: set[bytes] = set()
        for rank in range(self._tp_size):
            for io_key in marker_sets[rank]:
                if len(io_key) == 18:
                    candidates.add(bytes(io_key[:16]))
        num_layers = self._layout.layer_span
        if not num_layers:
            return set()
        expected_layers = set(range(num_layers))
        slot_bytes = num_layers * self._segment_bytes
        valid = set()
        for chunk_key in candidates:
            accepted = True
            for rank in range(self._tp_size):
                if not self._pool_manifest_matches(rank, chunk_key, slot_bytes):
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
        del chunk_key
        return self._layout.layer_span

    def _pool_manifest_matches(self, rank, chunk_key, slot_bytes) -> bool:
        """池 manifest 是否认可该 chunk 的归属与几何。"""
        path = self._layouts[rank].pool_manifest_path()
        try:
            manifest = json.loads(path.read_text("utf-8"))
            allocation = manifest["allocated"][chunk_key.hex()]
            geometry = manifest["rank_geometry"]
            return (
                manifest.get("namespace") == self._key_namespace.hex()
                and int(manifest.get("slot_bytes", -1)) == slot_bytes
                and int(geometry.get("num_layers", -1)) * self._segment_bytes
                == slot_bytes
                and int(allocation.get("generation", -1)) > 0
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False


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

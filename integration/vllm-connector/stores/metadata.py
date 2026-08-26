"""Scheduler-only metadata stores.

These stores expose only capacity, namespace ownership, marker scanning, and
eviction.  They intentionally have no runtime, buffer registration, stream,
target, or data-transfer API.
"""

from __future__ import annotations

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
        self._namespace_matches = True
        if layout in (None, "file_per_chunk", "file"):
            self._layout = Layout(root, segment_bytes)
        elif layout == "striped":
            if mounts is None:
                mounts = _preset_mounts(preset)
            if stripe_unit is None:
                raise ValueError("striped metadata store requires stripe_unit")
            self._layout = StripedLayout(
                root, segment_bytes, mounts, stripe_unit
            )
        else:
            raise ValueError(f"unknown tutti_nvme layout: {layout!r}")

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
        self._layout.ensure_dirs()
        if self._key_namespace is not None:
            self._namespace_matches = self._layout.check_namespace(
                self._key_namespace
            )
            if not self._namespace_matches:
                _LOG.warning(
                    "pool %s namespace manifest mismatch; scheduler treats it "
                    "as empty",
                    self._layout.root,
                )
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def scan(self):
        self._require_open()
        if not self._namespace_matches:
            return []
        return sorted(self._layout.scan())

    def drop(self, keys) -> None:
        self._require_open()
        self._layout.drop(keys)

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("metadata store is not open")


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

"""Scheduler-only metadata stores.

These stores expose only capacity, namespace ownership, residency scanning, and
eviction. They intentionally have no runtime, buffer registration, stream,
target, or data-transfer API.

The persistent one is a **read-only view of the object layer**: the worker is
the sole writer of a namespace, the scheduler opens the same namespace read-only
(no directory creation, no prewarming, no materialisation) and reads the
committed key set from the checkpoint. Everything after that first read is
in-memory: residency is published by the worker's per-step increments, so the
scheduler never touches the filesystem on the request path.
"""

from __future__ import annotations

import logging

from .object_store import ObjectStore, _fingerprint_bytes
from .registry import (
    create_metadata_store as _create_metadata_store,
    register_metadata_store_type,
)
from .tutti_nvme.preset_derive import derive_device_fields


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

    @property
    def layer_span(self) -> int | None:
        return None

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
    """调度侧的驻留索引：对象层的**只读**视图 + 内存权威集合。

    盘上不再有 marker/manifest：worker 是命名空间唯一写者，调度侧以只读视图
    读一次已提交集合（冷启动对账），此后驻留集合由 worker 的 committed 增量
    维护，查询热路径完全不碰文件系统。
    """

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
        layer_span: int | None = None,
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
        self._roots = [
            str(options.get("root")) for options in rank_options
        ]
        if self._tp_size > 1 and len(set(self._roots)) != self._tp_size:
            raise ValueError("every TP rank requires a distinct metadata root")
        self._views: list[ObjectStore] = []
        self._live: set[bytes] = set()
        # 条带几何（挂载点 + 条带粒度）与数据面必须完全一致：对象层的槽位
        # 路径由它推导，不一致就会指向别的文件。
        self._mounts: list[str] | None = None
        self._stripe_unit = 0
        if rank_options[0].get("layout") == "striped":
            stripe_unit = rank_options[0].get("stripe_unit")
            if stripe_unit is None:
                raise ValueError("striped metadata store requires stripe_unit")
            mounts = rank_options[0].get("mounts")
            preset = rank_options[0].get("preset")
            if mounts is None and isinstance(preset, dict):
                if "daemon_config" in preset:
                    import yaml
                    preset = derive_device_fields(preset, yaml)
                mounts = _preset_mounts(preset)
            if not mounts:
                raise ValueError("striped metadata store requires mounts")
            self._mounts = [str(mount) for mount in mounts]
            self._stripe_unit = int(stripe_unit)
        # 冷启动对账需要层宽（对象几何 = 段数 × 段大小 + 对象头），未声明时
        # scan() 返回空（fail-closed）。worker 侧在 bind 后由引擎注入；调度侧
        # 没有 bind 阶段，必须在构造时给出——否则复用已有池时永远恢复不到任何
        # 驻留项，命中率静默归零（索引此后以内存为权威，不会再扫盘）。
        self._layer_span = int(layer_span) if layer_span else None

    @property
    def capacity_chunks(self) -> int:
        return self._num_chunks

    @property
    def layer_span(self) -> int | None:
        """层宽（对象几何的一半）；未声明时 None（scan 一律返回空）。"""
        return self._layer_span

    def set_key_namespace(self, namespace: bytes) -> None:
        if self._opened:
            raise RuntimeError("namespace must be configured before open")
        self._key_namespace = bytes(namespace)

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("metadata store is already open")
        self._opened = True
        if self._layer_span:
            self._refresh()

    def close(self) -> None:
        for view in self._views:
            try:
                view.close()
            except Exception:  # pragma: no cover - 关闭失败不影响调度
                _LOG.exception("关闭对象层只读视图失败")
        self._views = []
        self._live = set()
        self._opened = False

    def scan(self):
        """已驻留的 io_key（升序）。

        首次调用（或 open 时）读一次对象层；之后以内存集合为准——worker 的
        提交会经 ``TuttiWorkerMetadata`` 增量补进来。
        """
        self._require_open()
        if not self._live and self._layer_span:
            self._refresh()
        return sorted(self._live)

    def drop(self, keys) -> None:
        self._require_open()
        self._live.difference_update(bytes(key) for key in keys)

    # ---- 内部 ----

    def _refresh(self) -> None:
        """读一次各 rank 的已提交集合，取交集（全 rank 齐备才可读）。"""
        try:
            views = self._views or self._open_views()
        except Exception:
            _LOG.warning(
                "对象层只读视图打开失败；本轮冷启动对账按空池处理（后续由 "
                "worker 增量恢复）", exc_info=True,
            )
            return
        committed = [view.recover() for view in views]
        if not committed:
            return
        common = set.intersection(*committed)
        span = self._layer_span or 0
        self._live = {
            chunk + layer.to_bytes(2, "little")
            for chunk in common
            for layer in range(span)
        }
        if common:
            _LOG.info(
                "METADATA_RECOVERED chunks=%d io_keys=%d ranks=%d",
                len(common), len(self._live), len(committed),
            )

    def _open_views(self) -> list[ObjectStore]:
        stores = []
        for rank, root in enumerate(self._roots):
            options = {
                "scheme": (
                    "striped_local_nvme_file"
                    if self._stripe_unit else "local_nvme_file"
                ),
                "uri": root,
                "capacity_slots": self._num_chunks,
                "segment_bytes": self._segment_bytes,
                "segment_count": self._layer_span,
                "namespace_fingerprint": _fingerprint_bytes(self._key_namespace),
                "devices": self._devices_for(rank),
                "stripe_unit": self._stripe_unit,
                "prewarm_bytes": 0,
                "background_reclaim": False,
                "rank_id": 0,
                "rank_count": 1,
                "read_only": True,
            }
            store = ObjectStore(options)
            store.open()
            stores.append(store)
        self._views = stores
        return stores

    def _devices_for(self, rank: int) -> list[dict]:
        del rank
        mounts = self._mounts or [self._roots[0]]
        return [{"mount_path": mount} for mount in mounts]

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("metadata store is not open")


# 调度侧 store 的注册：与数据面共用 stores.registry 的注册面，目标以
# "module:Class" 惰性给出，因此导入本模块不会拉起数据面实现。
register_metadata_store_type("memory", "tutti.storage.metadata:MemoryMetadataStore")
register_metadata_store_type("tutti_nvme", "tutti.storage.metadata:TuttiMetadataStore")

#: 兼容别名：实现已上移到 stores.registry.create_metadata_store，这里保留
#: 同名模块属性，使既有导入路径与测试打桩（monkeypatch 本模块属性）不变。
create_metadata_store = _create_metadata_store


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

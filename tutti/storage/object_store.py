"""对象存储层（C++ SPI）的 Python 唯一入口。

分层约定（评审定案）：
  * 文件系统的一切职责——槽位分配、对象头、检查点、跨 rank 位图、崩溃恢复
    ——全部由 C++ 的 ``StorageObjectStore`` 承担；
  * Python 只做**内存簿记**：key 进，placement 出。稳态不产生任何 POSIX 调用，
    也不存在 marker / manifest 这类 Python 侧元数据文件；
  * 数据面不经过这里：``placement.uri`` 直接交给 ``StorageRuntime.open_batch``。

提交语义是**按对象（chunk）一次**，不是按层：一个 chunk 的所有层段写在同一个
对象里，``commit`` 只写一次对象头。因此"某层没写完"在结构上就等于"整个对象
未提交"，读侧永远不会看到半截 chunk。段序由调用方决定，未提交前对象一律无效。

    ticket        = runtime.open(placement.uri)
    target_offset = placement.offset + segment_index * segment_bytes
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ObjectPlacement", "ObjectStore", "ObjectStoreUnavailable"]


class ObjectStoreUnavailable(RuntimeError):
    """tutti_runtime 绑定不可用（未构建或不在 PYTHONPATH 上）。"""


@dataclass(frozen=True)
class ObjectPlacement:
    """一个对象在介质上的位置。``uri`` 交给 runtime 解析，本层不解析。"""

    key: bytes
    slot: int
    generation: int
    uri: str
    offset: int
    payload_bytes: int


def _core():
    try:
        from tutti_runtime import _core
    except ImportError as exc:  # pragma: no cover - 环境缺失路径
        raise ObjectStoreUnavailable(
            "tutti_runtime._core 不可用：先构建 csrc/python 并把 "
            "csrc/python/src 加入 PYTHONPATH"
        ) from exc
    return _core


def _fingerprint_bytes(value) -> bytes:
    """把命名空间指纹归一为 bytes（str/bytes/None 都接受）。"""
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


class ObjectStore:
    """``tutti_runtime._core.ObjectStore`` 的薄封装。

    只做三件事：把配置字典讲清楚、"key → placement" 的内存缓存、把 C++ 侧
    的返回结构转成 Python 对象。任何文件系统语义都不在这里实现。
    """

    def __init__(self, config: dict):
        self._config = dict(config)
        self._core_store = None
        self._placements: dict[bytes, ObjectPlacement] = {}

    # ---- 生命周期 ----

    def open(self) -> None:
        core = _core()
        config = dict(self._config)
        config["namespace_fingerprint"] = _fingerprint_bytes(
            config.get("namespace_fingerprint")
        )
        store = core.ObjectStore()
        store.open(config.pop("scheme", "local_nvme_file"), config)
        self._core_store = store

    def close(self) -> None:
        if self._core_store is None:
            return
        try:
            self._core_store.close()
        finally:
            self._core_store = None
            self._placements.clear()

    @property
    def is_open(self) -> bool:
        return self._core_store is not None

    # ---- 查询（全部命中内存缓存或 C++ 内存索引） ----

    def contains(self, key: bytes) -> bool:
        return self._required().contains(key)

    def contains_prefix(self, keys) -> int:
        return int(self._required().contains_prefix(list(keys)))

    def contains_prefix_all_ranks(self, keys) -> int:
        return int(self._required().contains_prefix_all_ranks(list(keys)))

    def placement(self, key: bytes) -> ObjectPlacement | None:
        """内存缓存的 placement；未命中时回落到 C++ 的 lookup。"""
        cached = self._placements.get(key)
        if cached is not None:
            return cached
        found = self._required().lookup(key)
        if found is None:
            return None
        placement = ObjectPlacement(
            key=key,
            slot=int(found["slot"]),
            generation=int(found["generation"]),
            uri=str(found["uri"]),
            offset=int(found["offset"]),
            payload_bytes=int(found["payload_bytes"]),
        )
        self._placements[key] = placement
        return placement

    def usage(self) -> dict:
        return dict(self._required().usage())

    def ready_slots(self) -> int:
        """已物化（盘上存在）的槽位数——绑定期预热票据的取值范围。"""
        return int(self._required().ready_slots())

    def slot_uri(self, slot: int) -> str | None:
        return self._required().slot_uri(int(slot))

    def slot_generation(self, slot: int) -> int:
        return int(self._required().slot_generation(int(slot)))

    # ---- 写路径 ----

    def reserve(self, keys) -> tuple[dict[bytes, ObjectPlacement], int]:
        """预留空间。返回 (accepted, rejected_count)。

        部分接受是合法结果（容量耗尽），不是异常：缓存未命中不该让产生它的
        请求失败。已提交的 key 不会被重复预留，直接带着原 placement 返回。
        """
        unique = list(dict.fromkeys(bytes(key) for key in keys))
        accepted, rejected = self._required().reserve(unique)
        placements: dict[bytes, ObjectPlacement] = {}
        for item in accepted:
            key = bytes(item["key"])
            placement = ObjectPlacement(
                key=key,
                slot=int(item["slot"]),
                generation=int(item["generation"]),
                uri=str(item["uri"]),
                offset=int(item["offset"]),
                payload_bytes=int(item["payload_bytes"]),
            )
            placements[key] = placement
            self._placements[key] = placement
        return placements, int(rejected)

    def commit(self, keys) -> None:
        keys = list(dict.fromkeys(bytes(key) for key in keys))
        if not keys:
            return
        self._required().commit(keys)

    def abort(self, keys) -> None:
        keys = list(dict.fromkeys(bytes(key) for key in keys))
        if not keys:
            return
        self._required().abort(keys)
        for key in keys:
            self._placements.pop(key, None)

    def release(self, keys) -> int:
        keys = list(dict.fromkeys(bytes(key) for key in keys))
        if not keys:
            return 0
        released = int(self._required().release(keys))
        for key in keys:
            self._placements.pop(key, None)
        return released

    # ---- 读保护 ----

    def pin(self, keys) -> None:
        keys = list(dict.fromkeys(bytes(key) for key in keys))
        if keys:
            self._required().pin(keys)

    def unpin(self, keys) -> None:
        keys = list(dict.fromkeys(bytes(key) for key in keys))
        if keys:
            self._required().unpin(keys)

    # ---- 恢复 ----

    def recover(self) -> set[bytes]:
        """open() 后确认有效的对象集合（冷启动对账，之后由内存索引接管）。"""
        return {bytes(key) for key in self._required().recover()}

    def checkpoint(self) -> None:
        self._required().checkpoint()

    # ---- 内部 ----

    def _required(self):
        if self._core_store is None:
            raise RuntimeError("object store 未打开")
        return self._core_store

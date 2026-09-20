"""对象层支撑的 KV 布局：Python 侧不做任何文件系统操作。

`Layout` / `StripedLayout` 的公开面（store.py 与调度侧只依赖这些方法）在这里
全部由 C++ ``StorageObjectStore`` 承担：

    槽位分配  → reserve()          对象有效性 → commit()（每对象一次）
    槽位路径  → placement.uri      崩溃恢复   → recover()
    容量      → usage()            跨 rank    → contains_prefix_all_ranks()

与旧实现的关键语义差异（评审定案）：

* **按对象提交，不按层**。一个 chunk 的全部层段共用一个对象，``commit`` 只写
  一次对象头；某层没写完在结构上就等于"整个对象未提交"，读侧永远看不到半截
  chunk。Python 侧只记 ``{chunk: 已写层集合}``，齐了才调一次 commit。
* **槽位路径稳定且不含 chunk 身份**（``<root>/slots/<slot>.obj`` 或
  ``<mount>/striped/<slot>.shard<i>``），因此票据缓存可以按 URI 长驻，分配与
  回收都不需要 rename。
* **零 marker 文件**。旧实现的 ``meta/<chunk>.<layer>.ok`` 与 manifest JSON 全
  部消失，冷启动驻留集合来自 ``recover()``。

层宽（``num_layers`` → ``segment_count``）在建布局时未知，因此对象的 open 推迟
到 :meth:`set_layer_span`（引擎在此之后立刻 ``_deferred_restore``，见
``engine/core.py``）；在那之前的 :meth:`scan` 返回空集。
"""

from __future__ import annotations

import os
from pathlib import Path

from tutti.index.chunk_index import decode_io_key as _decode
from tutti.storage.object_store import ObjectPlacement, ObjectStore

__all__ = ["ObjectLayout"]

class ObjectLayout:
    """由对象层驱动的布局。

    ``mounts`` 是数据盘目录（条带顺序）；``stripe_unit`` 为 0 时走单文件布局。
    ``devices`` 是给 runtime resolver 用的设备事实（controller/namespace），
    对象层只用其中的 ``mount_path``。
    """

    def __init__(
        self,
        root: str | os.PathLike,
        segment_bytes: int,
        *,
        mounts=None,
        stripe_unit: int = 0,
        devices=None,
        capacity_chunks: int = 0,
        prewarm_chunks: int = 0,
        rank_id: int = 0,
        rank_count: int = 1,
        background_reclaim: bool = True,
        namespace: bytes | str | None = None,
    ):
        self._root = Path(root)
        self._segment_bytes = int(segment_bytes)
        self._mounts = [str(m) for m in (mounts if mounts else [self._root])]
        self._stripe_unit = int(stripe_unit or 0)
        self._devices = list(devices or [])
        # 条带几何的本地快速校验：单个挂载点无条带可言；条带粒度必须容纳
        # 一个 4KiB IO（段本身也要求 4096 对齐），否则 C++ 侧要到首次写才报错。
        if self._stripe_unit:
            if len(self._mounts) < 2:
                raise ValueError(
                    f"striped layout requires at least two mounts, got "
                    f"{len(self._mounts)}"
                )
            if self._stripe_unit % 4096:
                raise ValueError(
                    f"stripe_unit must be 4096-aligned, got {self._stripe_unit}"
                )
        # 容量/预热按 chunk 计（对象层按 slot_bytes 折算为字节），避免上层
        # 重复推导对象几何——对象头 4096B 由本模块在打开时加进去。
        self._capacity_chunks = int(capacity_chunks or 0)
        self._prewarm_chunks = int(prewarm_chunks or 0)
        self._rank_id = int(rank_id)
        self._rank_count = int(rank_count)
        self._background_reclaim = bool(background_reclaim)
        self._namespace = (
            namespace.encode("utf-8") if isinstance(namespace, str) else namespace
        )

        self._layer_span: int | None = None
        self._store: ObjectStore | None = None
        self._committed: set[bytes] = set()
        self._reserved: dict[bytes, ObjectPlacement] = {}
        self._written_layers: dict[bytes, set[int]] = {}

    # ---------- 几何 / 生命周期 ----------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def mounts(self) -> tuple[str, ...]:
        """数据盘挂载点（单文件布局时即 root 本身）。"""
        return tuple(self._mounts)

    @property
    def stripe_unit(self) -> int:
        return self._stripe_unit

    @property
    def segment_bytes(self) -> int:
        return self._segment_bytes

    @property
    def layer_span(self) -> int | None:
        return self._layer_span

    def set_namespace(self, namespace) -> None:
        """声明 key 命名空间（打开前生效）；不一致时对象层 fail-closed。"""
        if self._store is not None:
            raise RuntimeError("命名空间须在打开对象层之前声明")
        self._namespace = (
            namespace.encode("utf-8") if isinstance(namespace, str) else namespace
        )

    def set_layer_span(self, num_layers: int) -> None:
        """定层宽并打开对象层（此后才可能做冷启动恢复）。"""
        num_layers = int(num_layers)
        if num_layers <= 0:
            raise ValueError(f"层数须为正整数，got {num_layers}")
        if self._layer_span is not None and self._layer_span != num_layers:
            raise ValueError(
                f"层宽已定案 {self._layer_span}，不能改为 {num_layers}"
            )
        self._layer_span = num_layers
        if self._store is None:
            self._open_store()

    def _config(self) -> dict:
        assert self._layer_span is not None
        devices = []
        if self._devices:
            for index, device in enumerate(self._devices):
                entry = dict(device)
                entry.setdefault(
                    "mount_path",
                    self._mounts[index] if index < len(self._mounts) else self._mounts[0],
                )
                devices.append(entry)
        else:
            devices = [{"mount_path": mount} for mount in self._mounts]
        prewarm_chunks = min(self._prewarm_chunks, self._capacity_chunks)
        return {
            "scheme": "striped_local_nvme_file" if self._stripe_unit else "local_nvme_file",
            "uri": str(self._root),
            # 容量/预热按槽位声明：每槽位字节数由对象层按几何算（条带布局的
            # 对象头前缀会向上取整到整个条带轮，本层不该重算）。
            "capacity_slots": self._capacity_chunks,
            "prewarm_slots": prewarm_chunks,
            "segment_bytes": self._segment_bytes,
            "segment_count": self._layer_span,
            "namespace_fingerprint": self._namespace or b"",
            "devices": devices,
            "stripe_unit": self._stripe_unit,
            "background_reclaim": self._background_reclaim,
            "rank_id": self._rank_id,
            "rank_count": self._rank_count,
        }

    def _open_store(self) -> None:
        store = ObjectStore(self._config())
        # 命名空间/几何不一致时对象层 fail-closed（不覆盖旧数据），异常直接上抛：
        # 异构池必须由人处理，不能被当成"空池"静默复用。
        store.open()
        self._store = store
        self._committed = store.recover()

    def ensure_dirs(self) -> None:
        """对象层自行物化目录与槽位文件，这里无事可做。"""

    def close_object_pool(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
        self._committed.clear()
        self._reserved.clear()
        self._written_layers.clear()

    def attach_object_pool(self, pool) -> None:
        """兼容旧装配签名：对象布局不需要外部池。"""

    def check_namespace(self, namespace) -> bool:
        """命名空间校验已下沉到对象层（open 时 fail-closed）。"""
        return True

    # ---------- 恢复 ----------

    def ready_slots(self) -> list[tuple[str, int]]:
        """已物化槽位的 ``(uri, generation)`` 列表（绑定期预热用）。

        冷池与热池都非空：对象层在 open() 时已把预热槽位物化，恢复出的槽位
        同样在盘上。绑定期拿这些 URI 先把运行时票据开好，请求路径就只剩内存
        查找——首轮 resolve + peer-memory 注册的几百毫秒因此移出请求路径。
        """
        store = self._store
        if store is None:
            return []
        out: list[tuple[str, int]] = []
        for slot in range(store.ready_slots()):
            uri = store.slot_uri(slot)
            if uri:
                out.append((uri, store.slot_generation(slot)))
        return out

    @property
    def slot_payload_bytes(self) -> int:
        """每个对象的 payload 字节数（各槽位一致，用于预热票据的尺寸校验）。"""
        return self._segment_bytes * int(self._layer_span or 0)

    def committed_chunks(self) -> set[bytes]:
        """已提交（= 层齐全）的 chunk 集合；层宽未定案时为空集。"""
        return set(self._committed)

    def checkpoint(self) -> None:
        """把对象层内存状态（已提交 key 集合）落到检查点。

        close() 会自动落盘；显式调用用于"希望下一次冷启动立刻看到"的时点
        （例如回收/驱逐之后）。
        """
        if self._store is not None:
            self._store.checkpoint()

    def scan(self) -> set[bytes]:
        """已提交对象的 io_key 集合。

        对象有效 ⇔ 全部段都写过，因此每个已提交 chunk 展开成 layer_span
        个 io_key——与旧实现"全层标记齐备"的判定等价。
        """
        span = self._layer_span or 0
        keys: set[bytes] = set()
        for chunk in self._committed:
            if len(chunk) == 16:
                # 标准 chunk：对象有效 ⇒ 全部层段都可读
                keys.update(
                    chunk + layer.to_bytes(2, "little")
                    for layer in range(span)
                )
            else:
                # 通用短 key（通用 KV 契约）：key 本身没有层后缀，形态即原样
                keys.add(chunk)
        return keys

    # ---------- 写路径 ----------

    def prepare_put(self, io_keys, capacity_chunks: int):
        """预留对象，返回 ``(admitted, rejected_count)``。

        ``admitted`` 形如 ``{io_key: (chunk, layer)}``，只含**受理**的 key；
        容量耗尽是稳态而非故障（对象层契约：部分受理、绝不阻塞），因此这里
        不抛错——调用方按返回的集合裁剪本批写入即可，缓存未命中不该让产生它
        的请求失败。
        """
        decoded = {}
        chunks: list[bytes] = []
        for io_key in io_keys:
            chunk_id, layer = _decode(io_key)
            decoded[bytes(io_key)] = (chunk_id, layer)
            if chunk_id not in chunks:
                chunks.append(chunk_id)
        placements, rejected = self._store_required().reserve(chunks)
        self._reserved.update(placements)
        admitted = {
            io_key: value for io_key, value in decoded.items()
            if value[0] in placements
        }
        return admitted, int(rejected)

    def commit_layers(self, io_keys) -> set[bytes]:
        """逐层记账；某 chunk 的层写齐后按对象提交一次。

        返回**本次新提交**的 chunk 集合——调用方据此把"对象有效"的那批
        层记为驻留（对象有效 ⇔ 全部段都写过）。
        """
        pending: dict[bytes, set[int]] = {}
        for io_key in io_keys:
            chunk_id, layer = _decode(io_key)
            pending.setdefault(chunk_id, set()).add(layer)
        ready: list[bytes] = []
        span = self._layer_span
        for chunk_id, layers in pending.items():
            written = self._written_layers.setdefault(chunk_id, set())
            written |= layers
            # 层宽未知（未 set_layer_span）时不允许提交：全对象语义要求
            # "所有段都写过"才有效，无法判定就必须等。
            if span is not None and len(written) >= span:
                ready.append(chunk_id)
        if not ready:
            return set()
        self._store_required().commit(ready)
        for chunk_id in ready:
            self._committed.add(chunk_id)
            self._written_layers.pop(chunk_id, None)
        return set(ready)

    def abort_uncommitted(self, chunk_ids) -> None:
        """丢弃未提交的预留（已提交的对象不受影响）。"""
        pending = [
            chunk for chunk in dict.fromkeys(bytes(c) for c in chunk_ids)
            if chunk not in self._committed
        ]
        if pending:
            self._store_required().abort(pending)
        for chunk in pending:
            self._reserved.pop(chunk, None)
            self._written_layers.pop(chunk, None)

    def release_chunks(self, chunk_ids) -> int:
        chunks = list(dict.fromkeys(bytes(c) for c in chunk_ids))
        released = self._store_required().release(chunks)
        for chunk in chunks:
            self._committed.discard(chunk)
            self._reserved.pop(chunk, None)
            self._written_layers.pop(chunk, None)
        return released

    def drop(self, io_keys) -> None:
        self.release_chunks(_decode(c)[0] for c in io_keys)

    def releasable_chunks(self, io_keys) -> set[bytes]:
        """这些 key 里可回收的 chunk：只回收已提交（层齐全）的对象。"""
        return {
            chunk for chunk in (_decode(c)[0] for c in io_keys)
            if chunk in self._committed
        }

    # ---------- 定位 ----------

    def is_committed(self, chunk_id: bytes) -> bool:
        """对象是否有效（全部段都写过）。半截对象一律不可读。"""
        return bytes(chunk_id) in self._committed

    def _placement(self, chunk_id: bytes) -> ObjectPlacement | None:
        placement = self._reserved.get(chunk_id)
        if placement is not None:
            return placement
        store = self._store
        if store is None:
            return None
        placement = store.placement(bytes(chunk_id))
        if placement is not None:
            self._reserved[chunk_id] = placement
        return placement

    def target_uri(self, chunk_id: bytes) -> str:
        placement = self._placement(chunk_id)
        if placement is None:
            raise KeyError(f"chunk 未预留：{bytes(chunk_id)!r}")
        return placement.uri

    def target_offset(self, chunk_id: bytes) -> int:
        """段 0 在对象逻辑地址空间中的起点（对象头之后）。"""
        placement = self._placement(chunk_id)
        if placement is None:
            raise KeyError(f"chunk 未预留：{bytes(chunk_id)!r}")
        return int(placement.offset)

    def target_size(self, chunk_id: bytes) -> int:
        placement = self._placement(chunk_id)
        return int(placement.payload_bytes) if placement is not None else 0

    def target_generation(self, chunk_id: bytes) -> int:
        placement = self._placement(chunk_id)
        return int(placement.generation) if placement is not None else 0

    def object_pool_snapshot(self) -> dict | None:
        if self._store is None:
            return None
        snapshot = self._store.usage()
        snapshot["committed_chunks"] = len(self._committed)
        return snapshot

    # ---------- 内部 ----------

    def _store_required(self) -> ObjectStore:
        if self._store is None:
            raise RuntimeError(
                "对象层未打开：先 set_layer_span(num_layers)（层宽定案后才能定"
                "义对象几何）"
            )
        return self._store




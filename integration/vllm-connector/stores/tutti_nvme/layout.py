"""file_per_chunk 布局——TuttiKVStore 的私有实现。

目录结构（root 之下）::

    chunks/<hex>.bin   每个 chunk 一个数据文件；<hex> = chunk 身份的 hex
    meta/<hex>.ok      层完成标记（空文件）；<hex> = 完整 io_key 的 hex

- io_key 私有解读（store 层盲，仅做纯字节映射）：前 16 字节为 chunk
  身份（文件分组键），其余字节按小端解释为层号——18B 标准 io_key 恰为
  chunk_key 16B + layer 2B；长度不足 16B 的 key 整体即 chunk 身份、
  层号为 0。
- 层段 offset = 层号 × segment_bytes（数据文件内字节偏移，直接作为
  runtime 的 target_offset）。
- 数据文件按需增长：扩展只做实写（1MiB 零块循环 + fsync），禁
  fallocate 与稀疏空洞——LocalFileResolver 依赖 FIEMAP 解析物理
  extent，稀疏区域无法承载 DMA。
- 崩溃安全：层标记在数据真正落盘（runtime wait 至 COMPLETED）之后
  创建；残缺 chunk 由 engine 侧按 io_key 完整性规则判缺。
- placement 表即目录本身：数据文件名承载 chunk 身份，标记文件名承载
  完整 io_key，scan() 列目录即可无损重组，无需独立 sidecar。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .marker_generation import marker_generation

#: chunk_key 的固定字节长度。
CHUNK_PREFIX_BYTES = 16

#: 实写预分配的零块粒度。
_EXTEND_BLOCK_BYTES = 1024 * 1024

_ZERO_BLOCK = b"\x00" * _EXTEND_BLOCK_BYTES

_DATA_SUFFIX = ".bin"
_MARKER_SUFFIX = ".ok"
_MANIFEST_NAME = "namespace.manifest"
_OBJECT_POOL_MANIFEST = "object-pool.manifest.json"
_SCAN_GENERATION_NAME = ".scan-generation"


def decode_io_key(io_key: bytes) -> tuple[bytes, int]:
    """io_key → (chunk 身份, 层号)。任意长度 key 均可解读。"""
    if not isinstance(io_key, (bytes, bytearray, memoryview)):
        raise ValueError(f"io_key 必须为 bytes，得到 {type(io_key).__name__}")
    io_key = bytes(io_key)
    if not io_key:
        raise ValueError("io_key 不能为空")
    chunk_id = io_key[:CHUNK_PREFIX_BYTES]
    layer = int.from_bytes(io_key[CHUNK_PREFIX_BYTES:], "little")
    return chunk_id, layer


class Layout:
    """file_per_chunk 布局的文件系统面（全部为同步本地文件系统调用）。"""

    def __init__(self, root: str | os.PathLike, segment_bytes: int):
        if segment_bytes <= 0:
            raise ValueError(f"segment_bytes 必须为正数，得到 {segment_bytes}")
        self.root = Path(root)
        self.segment_bytes = segment_bytes
        #: 已知层宽时数据文件首写即全尺寸（层宽未定则按需增长）。
        self.layer_span: int | None = None
        self._chunks_dir = self.root / "chunks"
        self._free_dir = self.root / "free"
        self._meta_dir = self.root / "meta"
        self._object_pool = None
        self._scan_signature: tuple[int, int, bytes] | None = None
        self._scan_cache: set[bytes] = set()

    def set_layer_span(self, num_layers: int) -> None:
        """声明层宽（bind 后由引擎注入）：数据文件按层宽全尺寸预分配。

        文件逐层增长会使传输层票据随文件尺寸失效重开，层多时
        票据池被耗尽；首写即全尺寸令每文件恰一张稳定票据。
        """
        if num_layers <= 0:
            raise ValueError(f"num_layers 必须为正数，得到 {num_layers}")
        self.layer_span = num_layers
        if self._object_pool is not None:
            self._object_pool.configure(num_layers, self.segment_bytes)

    def attach_object_pool(self, pool) -> None:
        self._object_pool = pool

    # ---------- 池归属 ----------

    def check_namespace(self, namespace: bytes) -> bool:
        """校验池归属：manifest 与给定命名空间一致才可复用。

        首次建池（无 manifest）写入当前命名空间并返回 True；已有
        manifest 且一致 → True；不一致 → False（调用方按空池语义
        处理，禁止静默复用）。命名空间为不透明字节串（本层不解读）。
        """
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

    # ---------- 命名 ----------

    def chunk_file(self, chunk_id: bytes) -> Path:
        """chunk 数据文件路径。

        对象池模式下返回其**槽位文件**路径：槽位是稳定身份，分配/回收
        只改内存映射与 manifest、不改名，从而让上层按路径缓存的目标
        票据与底层按 extent 缓存的 handle 在槽位复用后依然命中（否则
        每次复用都要重跑 resolve：open+fstat+fsync+FIEMAP）。
        """
        pool = self._object_pool
        if pool is not None and pool.configured:
            slot = pool.slot_of(chunk_id)
            if slot is not None:
                return self.pool_slot_paths(slot)[0]
        return self._chunks_dir / (chunk_id.hex() + _DATA_SUFFIX)

    def target_uri(self, chunk_id: bytes) -> str:
        """Return the runtime target URI for a chunk."""
        return "file://" + str(self.chunk_file(chunk_id).resolve())

    def target_size(self, chunk_id: bytes) -> int:
        """Return the logical size currently visible to a runtime target."""
        try:
            return self.chunk_file(chunk_id).stat().st_size
        except OSError:
            return 0

    def marker_file(self, io_key: bytes) -> Path:
        return self._meta_dir / (bytes(io_key).hex() + _MARKER_SUFFIX)

    def marker_generation(self) -> str:
        return marker_generation(self._meta_dir)

    # ---------- 生命周期 ----------

    def ensure_dirs(self) -> None:
        self._chunks_dir.mkdir(parents=True, exist_ok=True)
        if self._object_pool is not None:
            self._free_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)

    def scan(self) -> set[bytes]:
        """列 meta/ 目录重组在场 io_key。

        The directory signature makes repeated scheduler-side refreshes cheap
        while still observing markers committed by a different process.
        """
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
            name = entry.name
            if not name.endswith(_MARKER_SUFFIX):
                continue
            try:
                live.add(bytes.fromhex(name[: -len(_MARKER_SUFFIX)]))
            except ValueError:
                continue  # 非 marker 命名的杂散文件忽略
        self._scan_signature = signature
        self._scan_cache = live
        return set(live)

    def chunk_file_count(self) -> int:
        if not self._chunks_dir.is_dir():
            return 0
        return sum(1 for _ in self._chunks_dir.glob("*" + _DATA_SUFFIX))

    # ---------- 数据面 ----------

    def prepare_put(self, io_keys, capacity_chunks: int) -> dict[bytes, tuple[bytes, int]]:
        """put 前的布局准备：容量检查 + 建缺失数据文件 + 按需实写扩展。

        返回 {io_key: (chunk 身份, 层号)}；容量不足抛 ValueError 且
        不触碰文件系统。
        """
        grouped: dict[bytes, int] = {}  # chunk 身份 -> 需容纳的最大层数
        decoded: dict[bytes, tuple[bytes, int]] = {}
        for io_key in io_keys:
            chunk_id, layer = decode_io_key(io_key)
            decoded[io_key] = (chunk_id, layer)
            if grouped.get(chunk_id, 0) < layer + 1:
                grouped[chunk_id] = layer + 1

        pool_active = (
            self._object_pool is not None and self._object_pool.configured
        )
        if pool_active:
            # 池模式下数据文件以槽位为单位（chunks/ 目录不再承载），
            # 已占用数由池给出，容量即槽位上限。
            held = self._object_pool.allocation_count()
            new_chunks = sum(
                1 for chunk_id in grouped
                if self._object_pool.slot_of(chunk_id) is None
            )
        else:
            held = self.chunk_file_count()
            new_chunks = sum(
                1 for chunk_id in grouped
                if not self.chunk_file(chunk_id).exists()
            )
        if held + new_chunks > capacity_chunks:
            raise ValueError(
                f"chunk 容量不足：现有 {held} + 新增 "
                f"{new_chunks} > 上限 {capacity_chunks}"
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
            path = self.chunk_file(chunk_id)
            need = layer_count * self.segment_bytes
            if self.layer_span is not None:
                # 层宽已知：首写即全尺寸（票据稳定，见 set_layer_span）
                need = max(need, self.layer_span * self.segment_bytes)
            current = path.stat().st_size if path.exists() else 0
            if current < need:
                _append_real_zeros(path, current, need)
        return decoded

    def commit_layers(self, io_keys) -> None:
        """数据落盘后建层标记（崩溃安全窗口：先数据后标记）。"""
        for io_key in io_keys:
            self.marker_file(io_key).touch(exist_ok=True)
        _bump_scan_generation(self._meta_dir)
        self._scan_signature = None

    def drop(self, io_keys) -> None:
        """逐 io_key 删层标记；某 chunk 的标记删光后回收其数据文件。"""
        touched_chunks: set[bytes] = set()
        for io_key in io_keys:
            chunk_id, _ = decode_io_key(io_key)
            self.marker_file(io_key).unlink(missing_ok=True)
            touched_chunks.add(chunk_id)
        released = []
        for chunk_id in touched_chunks:
            if any(self._meta_dir.glob(chunk_id.hex() + "*" + _MARKER_SUFFIX)):
                continue  # 该 chunk 仍有存活层段
            released.append(chunk_id)
        if self._object_pool is not None and self._object_pool.configured:
            self._object_pool.recycle(released)
        else:
            for chunk_id in released:
                self.chunk_file(chunk_id).unlink(missing_ok=True)
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
                if path.name.endswith(_MARKER_SUFFIX)
            } - removed
            if not remaining:
                released.add(chunk_id)
        return released

    def target_generation(self, chunk_id: bytes) -> int:
        """目标票据的失效判定世代 = 槽位**物理**世代。

        只在槽位文件被重建时变化；分配 / 回收 / 就地零化都不变。稳定
        槽位路径下，同一槽位的票据因此可跨"回收→再分配"持续命中——
        推理路径不再重跑 open（resolve + 句柄构建）。
        """
        if self._object_pool is None or not self._object_pool.configured:
            return 0
        generation = self._object_pool.chunk_slot_generation(chunk_id)
        return -1 if generation is None else generation

    def allocation_generation(self, chunk_id: bytes) -> int:
        """chunk 的**分配**世代（每次绑定递增）——提交记录校验用。"""
        if self._object_pool is None or not self._object_pool.configured:
            return 0
        generation = self._object_pool.generation(chunk_id)
        return -1 if generation is None else generation

    def pool_slot_uri(self, slot: int) -> str:
        """槽位文件的运行时目标 URI（启动期批量预开用）。"""
        return "file://" + str(self.pool_slot_paths(slot)[0].resolve())

    def pool_slot_size(self, slot: int) -> int:
        try:
            return self.pool_slot_paths(slot)[0].stat().st_size
        except OSError:
            return 0

    def pool_slot_generation(self, slot: int) -> int:
        """槽位物理世代（启动期预开时登记，与 target_generation 同源）。"""
        if self._object_pool is None or not self._object_pool.configured:
            return 0
        return self._object_pool.slot_generation(slot)

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
        return {
            "layout": "file_per_chunk",
            "num_layers": num_layers,
            "segment_bytes": segment_bytes,
            "slot_bytes": slot_bytes,
            "physical_slot_bytes": slot_bytes,
        }

    def pool_physical_sizes(self, slot_bytes: int) -> tuple[int]:
        return (slot_bytes,)

    def pool_slot_paths(self, slot: int) -> tuple[Path]:
        return (self._free_dir / f"{slot:08d}{_DATA_SUFFIX}",)

    def pool_bound_paths(self, chunk_id: bytes, slot: int) -> tuple[Path]:
        """chunk 绑定到槽位后其数据文件的路径。

        file_per_chunk 声明**稳定槽位路径**：绑定不改名，槽位文件就地
        复用。这样上层按路径缓存的目标票据（store._targets）与底层按
        extent 缓存的 handle（HandleWorkspaceCache）在槽位复用后继续
        命中；一旦改名，两者都会失效并重跑 resolve（open+fstat+fsync+
        FIEMAP），这正是首个写入波次那 ~19ms 的主要来源。
        """
        return self.pool_slot_paths(slot)

    def pool_create_slot(self, slot: int, slot_bytes: int) -> None:
        path = self.pool_slot_paths(slot)[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        _append_real_zeros(path, 0, slot_bytes)

    def pool_zero_slot(self, slot: int, slot_bytes: int) -> None:
        path = self.pool_slot_paths(slot)[0]
        _rewrite_real_zeros(path, slot_bytes)

    def pool_discover_free_slots(self) -> set[int]:
        slots = set()
        if not self._free_dir.is_dir():
            return slots
        for path in self._free_dir.glob("*" + _DATA_SUFFIX):
            try:
                slots.add(int(path.name[:-len(_DATA_SUFFIX)]))
            except ValueError:
                continue
        return slots

    def pool_chunk_complete(self, chunk_id: bytes,
                            num_layers: int | None = None) -> bool:
        layers = self.layer_span if num_layers is None else num_layers
        if not layers:
            return False
        expected = {
            self.marker_file(bytes(chunk_id) + layer.to_bytes(2, "little"))
            for layer in range(layers)
        }
        return all(path.exists() for path in expected)

    def pool_remove_markers(self, chunk_id: bytes) -> None:
        for path in self._meta_dir.glob(
            bytes(chunk_id).hex() + "*" + _MARKER_SUFFIX
        ):
            path.unlink(missing_ok=True)

    @staticmethod
    def pool_rename_group(sources, destinations) -> None:
        # 稳定槽位路径下 source 与 destination 相同（绑定/回收都不需要
        # 移动文件），此时整组退化为空操作。
        pending = [
            (source, destination)
            for source, destination in zip(sources, destinations)
            if source != destination
        ]
        if not pending:
            return
        if len(sources) != len(destinations):
            raise RuntimeError("object-pool rename group shape mismatch")
        if not all(path.exists() for path, _ in pending):
            raise RuntimeError("object-pool source group is incomplete")
        if any(destination.exists() for _, destination in pending):
            raise RuntimeError("object-pool destination already exists")
        completed = []
        try:
            for source, destination in pending:
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


def _append_real_zeros(path: Path, start: int, end: int) -> None:
    """以 1MiB 零块实写扩展文件至 [start, end)，禁 fallocate/稀疏，末尾 fsync。

    FIEMAP 只对已分配物理块的区域返回 extent，稀疏/预分配空洞会让
    resolver 的 DMA 映射缺段，因此扩展必须真实写入零数据。
    """
    with open(path, "ab") as handle:
        position = start
        while position < end:
            block = min(_EXTEND_BLOCK_BYTES, end - position)
            handle.write(_ZERO_BLOCK[:block])
            position += block
        handle.flush()
        os.fsync(handle.fileno())


def _rewrite_real_zeros(path: Path, size: int) -> None:
    """Overwrite an existing regular file with real zeros and fsync."""
    with open(path, "wb") as handle:
        position = 0
        while position < size:
            block = min(_EXTEND_BLOCK_BYTES, size - position)
            handle.write(_ZERO_BLOCK[:block])
            position += block
        handle.flush()
        os.fsync(handle.fileno())


def _read_scan_generation(meta_dir: Path) -> bytes:
    try:
        return (meta_dir / _SCAN_GENERATION_NAME).read_bytes()
    except OSError:
        return b""


def _bump_scan_generation(meta_dir: Path) -> None:
    """Publish a cross-process marker-generation token with atomic replace."""
    now = time.time_ns()
    token = f"{os.getpid()}:{threading.get_ident()}:{now}".encode("ascii")
    target = meta_dir / _SCAN_GENERATION_NAME
    temporary = meta_dir / (
        f"{_SCAN_GENERATION_NAME}.{os.getpid()}.{threading.get_ident()}.{now}"
    )
    try:
        temporary.write_bytes(token)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

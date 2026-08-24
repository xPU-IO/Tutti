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
from pathlib import Path

#: chunk_key 的固定字节长度。
CHUNK_PREFIX_BYTES = 16

#: 实写预分配的零块粒度。
_EXTEND_BLOCK_BYTES = 1024 * 1024

_ZERO_BLOCK = b"\x00" * _EXTEND_BLOCK_BYTES

_DATA_SUFFIX = ".bin"
_MARKER_SUFFIX = ".ok"


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
        self._meta_dir = self.root / "meta"

    def set_layer_span(self, num_layers: int) -> None:
        """声明层宽（bind 后由引擎注入）：数据文件按层宽全尺寸预分配。

        文件逐层增长会使传输层票据随文件尺寸失效重开，层多时
        票据池被耗尽；首写即全尺寸令每文件恰一张稳定票据。
        """
        if num_layers <= 0:
            raise ValueError(f"num_layers 必须为正数，得到 {num_layers}")
        self.layer_span = num_layers

    # ---------- 命名 ----------

    def chunk_file(self, chunk_id: bytes) -> Path:
        return self._chunks_dir / (chunk_id.hex() + _DATA_SUFFIX)

    def marker_file(self, io_key: bytes) -> Path:
        return self._meta_dir / (bytes(io_key).hex() + _MARKER_SUFFIX)

    # ---------- 生命周期 ----------

    def ensure_dirs(self) -> None:
        self._chunks_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)

    def scan(self) -> set[bytes]:
        """列 meta/ 目录重组在场 io_key（标记文件名即完整 io_key 的 hex）。"""
        live: set[bytes] = set()
        if not self._meta_dir.is_dir():
            return live
        for entry in self._meta_dir.iterdir():
            name = entry.name
            if not name.endswith(_MARKER_SUFFIX):
                continue
            try:
                live.add(bytes.fromhex(name[: -len(_MARKER_SUFFIX)]))
            except ValueError:
                continue  # 非 marker 命名的杂散文件忽略
        return live

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

        new_chunks = sum(
            1 for chunk_id in grouped if not self.chunk_file(chunk_id).exists()
        )
        if self.chunk_file_count() + new_chunks > capacity_chunks:
            raise ValueError(
                f"chunk 容量不足：现有 {self.chunk_file_count()} + 新增 "
                f"{new_chunks} > 上限 {capacity_chunks}"
            )

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

    def drop(self, io_keys) -> None:
        """逐 io_key 删层标记；某 chunk 的标记删光后回收其数据文件。"""
        touched_chunks: set[bytes] = set()
        for io_key in io_keys:
            chunk_id, _ = decode_io_key(io_key)
            self.marker_file(io_key).unlink(missing_ok=True)
            touched_chunks.add(chunk_id)
        for chunk_id in touched_chunks:
            if any(self._meta_dir.glob(chunk_id.hex() + "*" + _MARKER_SUFFIX)):
                continue  # 该 chunk 仍有存活层段
            self.chunk_file(chunk_id).unlink(missing_ok=True)


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

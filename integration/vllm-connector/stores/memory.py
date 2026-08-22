"""MemoryKVStore：KVStore 的进程内内存实现（全部测试的基座）。"""

from collections.abc import Iterable, Sequence

from .base import Completion


class _ImmediateCompletion:
    """即时完成的完成句柄：wait 立即返回，query 恒为 True。"""

    __slots__ = ()

    def wait(self) -> None:
        """阻塞至完成；本实现同步完成，立即返回。"""
        return None

    def query(self) -> bool:
        """非阻塞查询是否完成；本实现恒为 True。"""
        return True


class MemoryKVStore:
    """进程内内存实现：整批同步搬运，Completion 即时完成。

    契约：
    - 构造：segment_bytes 为单次搬运的字节长度（正整数），
      num_chunks 为容量 chunk 数（非负整数）；非法 → ValueError。
    - 生命周期：register_buffer / put_batch / get_batch / drop / scan
      仅在 open 之后、close 之前可用，否则 RuntimeError；
      open 幂等；close 释放全部数据与注册状态且幂等。
    - register_buffer：接受可写缓冲（bytearray、ctypes 数组、可写
      memoryview）；granularity 须为 4096 的正倍数且 ≤ segment_bytes，
      否则返回 None；成功返回自 1 递增的 buffer id。
    - put_batch / get_batch：整批同步搬运并返回即时完成的 Completion；
      条目为 (key, buffer_id, offset)，offset 起一段 segment_bytes 须
      完整位于 buffer 内，越界或 buffer 未注册 → ValueError；整批参数
      校验先行，任一非法则整批拒绝；get_batch 遇未驻留 key →
      ValueError；put_batch 对已驻留 key 覆盖旧值。
    - drop：批量删除；未驻留的 key 静默忽略（驱逐幂等）。
    - scan：枚举当前存活 key；capacity_chunks 恒等于构造的 num_chunks。
    """

    _GRANULARITY_UNIT = 4096

    def __init__(self, segment_bytes: int, num_chunks: int) -> None:
        if not _is_int(segment_bytes) or segment_bytes <= 0:
            raise ValueError(f"segment_bytes 须为正整数，got {segment_bytes!r}")
        if not _is_int(num_chunks) or num_chunks < 0:
            raise ValueError(f"num_chunks 须为非负整数，got {num_chunks!r}")
        self._segment_bytes = segment_bytes
        self._num_chunks = num_chunks
        self._pool: dict[bytes, bytearray] = {}
        self._buffers: dict[int, memoryview] = {}
        self._next_buffer_id = 0
        self._opened = False

    @property
    def capacity_chunks(self) -> int:
        """容量 chunk 数（构造入参，恒定）。"""
        return self._num_chunks

    def open(self) -> None:
        """进入可用状态；幂等。"""
        self._opened = True

    def close(self) -> None:
        """释放全部数据与注册状态；幂等。"""
        self._opened = False
        self._pool.clear()
        self._buffers.clear()

    def register_buffer(self, buffer, granularity: int) -> int | None:
        """注册可写缓冲为 IO buffer，返回递增 buffer id。

        granularity 非 4096 正倍数、大于 segment_bytes，或 buffer
        不可写（bytes / 非缓冲对象 / 只读视图）→ None。
        """
        self._require_open()
        if not _is_int(granularity):
            return None
        if (
            granularity <= 0
            or granularity % self._GRANULARITY_UNIT != 0
            or granularity > self._segment_bytes
        ):
            return None
        try:
            view = memoryview(buffer)
        except TypeError:
            return None
        if view.readonly:
            return None
        try:
            view = view.cast("B")
        except (TypeError, ValueError):
            return None
        self._next_buffer_id += 1
        self._buffers[self._next_buffer_id] = view
        return self._next_buffer_id

    def put_batch(self, batch: Sequence[tuple[bytes, int, int]]) -> Completion:
        """整批同步写入：每条 (key, buffer_id, offset) 的
        [offset, offset+segment_bytes) 字节拷贝为该 key 的驻留值。"""
        self._require_open()
        entries = list(batch)
        views = [self._view(buffer_id, offset) for _key, buffer_id, offset in entries]
        for (key, _buffer_id, _offset), view in zip(entries, views):
            self._pool[key] = bytearray(view)
        return _ImmediateCompletion()

    def get_batch(self, batch: Sequence[tuple[bytes, int, int]]) -> Completion:
        """整批同步读出：每条 (key, buffer_id, offset) 的驻留值
        拷贝到 buffer 的 [offset, offset+segment_bytes)。"""
        self._require_open()
        entries = list(batch)
        for key, _buffer_id, _offset in entries:
            if key not in self._pool:
                raise ValueError(f"未知 key：{key!r}")
        views = [self._view(buffer_id, offset) for _key, buffer_id, offset in entries]
        for (key, _buffer_id, _offset), view in zip(entries, views):
            view[:] = self._pool[key]
        return _ImmediateCompletion()

    def drop(self, keys: Sequence[bytes]) -> None:
        """批量删除；未驻留的 key 静默忽略（驱逐幂等）。"""
        self._require_open()
        for key in keys:
            self._pool.pop(key, None)

    def scan(self) -> Iterable[bytes]:
        """枚举当前存活 key（快照迭代）。"""
        self._require_open()
        return iter(list(self._pool.keys()))

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("store 未 open（或已 close），数据面操作不可用")

    def _view(self, buffer_id: int, offset: int) -> memoryview:
        """校验并返回 buffer 内 [offset, offset+segment_bytes) 的视图。

        buffer 未注册、offset 非整数或越界 → ValueError。
        """
        view = self._buffers.get(buffer_id)
        if view is None:
            raise ValueError(f"buffer {buffer_id!r} 未注册")
        if (
            not _is_int(offset)
            or offset < 0
            or offset + self._segment_bytes > view.nbytes
        ):
            raise ValueError(
                f"offset {offset!r} 非法：须满足 0 <= offset 且 "
                f"offset+segment_bytes({self._segment_bytes}) <= buffer 长度({view.nbytes})"
            )
        return view[offset : offset + self._segment_bytes]


def _is_int(value) -> bool:
    """判断是否为真 int（排除 bool）。"""
    return isinstance(value, int) and not isinstance(value, bool)

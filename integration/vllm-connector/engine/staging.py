"""环形窗口：staging 槽位的波次轮转与覆盖保护。"""

from __future__ import annotations


class RingWindow:
    """staging 环形窗口：双半窗交替流水，波次单调递增。

    契约：
    - 构造注入字节缓冲（部署层分配），窗口只做槽位几何，不搬运字节。
    - num_slots 须为正偶数（两个半窗）；buffer 字节数须不少于
      num_slots × segment_bytes；segment_bytes 须为正整数。
      任一不满足 → ValueError。
    - acquire(n)：领取下一波次的 n 个连续槽，返回 (wave, slots)。
      n 须满足 1 ≤ n ≤ num_slots // 2，否则 ValueError。wave 自 0
      单调递增；第 wave 波固定使用第 wave % 2 个半窗的前 n 个槽。
      与当前波相隔两个波次的前驱若已登记组合完成事件，先等待其完成
      （底层搬运完成且消费方已读完，覆盖保护），然后释放登记；未登记
      视为已完成，不阻塞。
    - complete(wave, event)：登记波次的完成事件。须在该波槽位可能
      被复用（下一次同半窗 acquire）之前调用；event 只需满足完成
      句柄协议（wait / query）。
    """

    def __init__(self, buffer, num_slots: int, segment_bytes: int):
        """buffer 为注入的字节缓冲；几何参数见类契约。"""
        if not _is_int(num_slots) or num_slots < 2 or num_slots % 2 != 0:
            raise ValueError(f"num_slots 须为 ≥2 的偶数，got {num_slots!r}")
        if not _is_int(segment_bytes) or segment_bytes <= 0:
            raise ValueError(f"segment_bytes 须为正整数，got {segment_bytes!r}")
        try:
            nbytes = len(buffer)
        except TypeError as exc:
            raise ValueError(f"buffer 须为带长度的缓冲对象: {buffer!r}") from exc
        if nbytes < num_slots * segment_bytes:
            raise ValueError(
                f"buffer 字节数 {nbytes} 不足：须 ≥ num_slots({num_slots}) "
                f"× segment_bytes({segment_bytes})"
            )
        self.buffer = buffer
        self.num_slots = num_slots
        self.segment_bytes = segment_bytes
        self._half = num_slots // 2
        self._wave = -1
        self._events: dict[int, object] = {}

    @property
    def capacity_per_wave(self) -> int:
        """单波最大槽位需求数（半窗容量）。"""
        return self._half

    def slot_offset(self, slot: int) -> int:
        """返回槽位在 buffer 内的起始字节偏移；槽号越界 → ValueError。"""
        if not _is_int(slot) or not 0 <= slot < self.num_slots:
            raise ValueError(f"slot 须在 [0, {self.num_slots}) 内，got {slot!r}")
        return slot * self.segment_bytes

    def acquire(self, n: int) -> tuple[int, list[int]]:
        """领取下一波次的 n 个槽；语义见类契约。"""
        if not _is_int(n) or not 1 <= n <= self._half:
            raise ValueError(f"n 须在 [1, {self._half}] 内，got {n!r}")
        self._wave += 1
        wave = self._wave
        stale = self._events.pop(wave - 2, None)
        if stale is not None:
            stale.wait()
        base = (wave % 2) * self._half
        return wave, list(range(base, base + n))

    def complete(self, wave: int, event) -> None:
        """登记波次完成事件；wave 非法 → ValueError。"""
        if not _is_int(wave) or wave < 0 or wave > self._wave:
            raise ValueError(f"wave 须在 [0, {self._wave}] 内，got {wave!r}")
        self._events[wave] = event


def _is_int(value) -> bool:
    """判断是否为真 int（排除 bool）。"""
    return isinstance(value, int) and not isinstance(value, bool)

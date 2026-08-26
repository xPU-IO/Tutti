"""环形窗口：staging 槽位的波次轮转与覆盖保护。"""

from __future__ import annotations


class RingWindow:
    """staging 环形窗口：有界槽位轮转，波次单调递增。

    契约：
    - 构造注入字节缓冲（部署层分配），窗口只做槽位几何，不搬运字节。
    - 兼容构造（未给 capacity_per_wave）要求 num_slots 为正偶数并保持
      双半窗语义。方向 bank 构造显式给 capacity_per_wave，可使用任意
      正槽数及 slot_base；buffer 须覆盖该物理槽区间。
    - acquire(n)：领取下一波次的 n 个连续（环绕时分段）槽，返回
      (wave, slots)。n 须满足 1 ≤ n ≤ capacity_per_wave，否则 ValueError。
      槽位在整个缓冲区内轮转；候选槽与尚未释放的旧波次重叠时，若已
      登记组合完成事件则先等待其完成（底层搬运完成且消费方已读完，
      覆盖保护），未登记视为已完成。num_slots=2×单波容量时退化为
      原有双半窗 wave-2 语义；更大的有界窗口允许 K>1 层预取。
    - complete(wave, event)：登记波次的完成事件。须在该波槽位可能
      被复用（下一次同半窗 acquire）之前调用；event 只需满足完成
      句柄协议（wait / query）。
    """

    def __init__(self, buffer, num_slots: int, segment_bytes: int,
                 capacity_per_wave: int | None = None, *, slot_base: int = 0):
        """buffer 为注入的字节缓冲；几何参数见类契约。"""
        if not _is_int(num_slots) or num_slots <= 0:
            raise ValueError(f"num_slots 须为正整数，got {num_slots!r}")
        if not _is_int(slot_base) or slot_base < 0:
            raise ValueError(f"slot_base 须为非负整数，got {slot_base!r}")
        if not _is_int(segment_bytes) or segment_bytes <= 0:
            raise ValueError(f"segment_bytes 须为正整数，got {segment_bytes!r}")
        try:
            nbytes = len(buffer)
        except TypeError as exc:
            raise ValueError(f"buffer 须为带长度的缓冲对象: {buffer!r}") from exc
        required = (slot_base + num_slots) * segment_bytes
        if nbytes < required:
            raise ValueError(
                f"buffer 字节数 {nbytes} 不足：须覆盖物理槽区间 "
                f"[0, {slot_base + num_slots}) × segment_bytes({segment_bytes})"
            )
        self.buffer = buffer
        self.num_slots = num_slots
        self.segment_bytes = segment_bytes
        self.slot_base = slot_base
        if capacity_per_wave is None:
            if num_slots < 2 or num_slots % 2 != 0:
                raise ValueError(
                    "兼容双半窗的 num_slots 须为 ≥2 的偶数，"
                    f"got {num_slots!r}"
                )
            capacity_per_wave = num_slots // 2
        if (not _is_int(capacity_per_wave) or capacity_per_wave <= 0
                or capacity_per_wave > num_slots):
            raise ValueError(
                "capacity_per_wave 须为正整数且不超过 num_slots，"
                f"got {capacity_per_wave!r}"
            )
        self._half = capacity_per_wave
        self._legacy_halves = num_slots == 2 * self._half
        self._wave = -1
        self._cursor = 0
        self._events: dict[int, object] = {}
        self._allocations: dict[int, tuple[int, ...]] = {}

    @property
    def capacity_per_wave(self) -> int:
        """单波最大槽位需求数（半窗容量）。"""
        return self._half

    def slot_offset(self, slot: int) -> int:
        """返回槽位在 buffer 内的起始字节偏移；槽号越界 → ValueError。"""
        end = self.slot_base + self.num_slots
        if not _is_int(slot) or not self.slot_base <= slot < end:
            raise ValueError(
                f"slot 须在 [{self.slot_base}, {end}) 内，got {slot!r}"
            )
        return slot * self.segment_bytes

    def acquire(self, n: int) -> tuple[int, list[int]]:
        """领取下一波次的 n 个槽；语义见类契约。"""
        if not _is_int(n) or not 1 <= n <= self._half:
            raise ValueError(f"n 须在 [1, {self._half}] 内，got {n!r}")
        self._wave += 1
        wave = self._wave
        if self._legacy_halves:
            base = (wave % 2) * self._half
            slots = tuple(
                self.slot_base + slot for slot in range(base, base + n)
            )
        else:
            slots = tuple(
                self.slot_base + (self._cursor + i) % self.num_slots
                for i in range(n)
            )
            self._cursor = (self._cursor + n) % self.num_slots
        occupied = set(slots)
        for old_wave, old_slots in list(self._allocations.items()):
            if not occupied.intersection(old_slots):
                continue
            stale = self._events.pop(old_wave, None)
            if stale is not None:
                stale.wait()
            self._allocations.pop(old_wave, None)
        self._allocations[wave] = slots
        return wave, list(slots)

    def complete(self, wave: int, event) -> None:
        """登记波次完成事件；wave 非法 → ValueError。"""
        if not _is_int(wave) or wave < 0 or wave > self._wave:
            raise ValueError(f"wave 须在 [0, {self._wave}] 内，got {wave!r}")
        if wave not in self._allocations:
            raise ValueError(f"wave {wave} 无有效槽位分配")
        self._events[wave] = event

    def drain(self) -> None:
        """等待并释放所有已登记波次；首个错误在两边清空后由调用方处理。"""
        first_error = None
        for wave, event in list(self._events.items()):
            try:
                if event is not None:
                    event.wait()
            except Exception as exc:  # drain 必须继续覆盖其余波次
                if first_error is None:
                    first_error = exc
            finally:
                self._events.pop(wave, None)
                self._allocations.pop(wave, None)
        # 未登记 event 的保留分配在 abort/finalize 时也必须作废。
        self._allocations.clear()
        if first_error is not None:
            raise first_error


def _is_int(value) -> bool:
    """判断是否为真 int（排除 bool）。"""
    return isinstance(value, int) and not isinstance(value, bool)

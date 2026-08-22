"""语义索引：chunk key 的驻留 / pin / pending 状态、前缀命中与 LRU 驱逐决策。

纯逻辑组件，单线程使用，不加锁；只维护 key 与状态，不接触数据本体。
token 序列按 chunk_tokens 切块并链式哈希为 key 序列，key 即跨进程
一致的 chunk 身份。
"""

from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

# chunk key 的字节长度（blake2b 摘要宽度）
_KEY_BYTES = 16
# 单个 token id 的编码宽度（字节，小端）
_TOKEN_BYTES = 8


@dataclass
class StorePlan:
    """一次写入计划的受理结果。

    字段：
        new_keys：本批需要写入的 key（受理后处于在途状态，等待结算）。
        evicted_keys：为腾出容量而驱逐的驻留 key，按从旧到新排列，
            善后由调用方负责。
    """

    new_keys: list[bytes]
    evicted_keys: list[bytes]


def _chunk_digest(parent: bytes, chunk: Sequence[int]) -> bytes:
    """计算一个 chunk 的 key：blake2b(parent + 逐 token 8 字节小端拼接)。"""
    payload = b"".join(t.to_bytes(_TOKEN_BYTES, "little") for t in chunk)
    return hashlib.blake2b(parent + payload, digest_size=_KEY_BYTES).digest()


class ChunkIndex:
    """语义索引：key → 驻留 / pin / pending。

    契约：
    - key 链：key_i = blake2b(parent_{i-1} + tokens_i)，parent_0 为空
      bytes，parent_i = key_i；token 以 8 字节小端编码。同一 token
      序列在任何进程得到相同 key。尾部不足一个 chunk 的 token 舍弃。
    - 状态三种：resident（驻留，参与命中与 LRU）、pending（写入已
      受理未结算：不参与命中、不参与驱逐、占用容量）、pin 计数
      （大于 0 即受驱逐保护）。
    - 驱逐只发生在 plan_store；pin 计数大于 0 的 key 永不驱逐。
    """

    def __init__(self, capacity: int, chunk_tokens: int):
        """capacity 为可容纳的 chunk 总数，chunk_tokens 为每 chunk 的 token 数。

        参数非正 → ValueError。
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity!r}")
        if chunk_tokens <= 0:
            raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens!r}")
        self._capacity = capacity
        self._chunk_tokens = chunk_tokens
        # 驻留表：插入序即 LRU 序（最旧在前）。
        self._resident: OrderedDict[bytes, None] = OrderedDict()
        # 在途集合：plan_store 受理、confirm_store 尚未结算。
        self._pending: set[bytes] = set()
        # pin 计数：条目存在即计数 ≥ 1。
        self._pins: Counter[bytes] = Counter()

    @property
    def capacity(self) -> int:
        """可容纳的 chunk 总数。"""
        return self._capacity

    @property
    def chunk_tokens(self) -> int:
        """每 chunk 的 token 数。"""
        return self._chunk_tokens

    # ---- 查询 ----

    def lookup_prefix(self, token_ids: Sequence[int]) -> int:
        """返回 token_ids 前缀中连续命中的 token 数（链式滚动）。

        从空 parent 起逐 chunk 计算 key：驻留则推进并累计 token 数，
        首个未驻留的 chunk 即停止。在途（pending）chunk 不算命中。
        """
        matched = 0
        parent = b""
        step = self._chunk_tokens
        for i in range(0, len(token_ids) - step + 1, step):
            parent = _chunk_digest(parent, token_ids[i:i + step])
            if parent not in self._resident:
                break
            matched += step
        return matched

    def hash_keys(
        self,
        token_ids: Sequence[int],
        start: int = 0,
        parent: bytes = b"",
    ) -> tuple[list[bytes], bytes]:
        """把 token_ids 从下标 start 起折叠为 key 序列。

        start 之前的 token 视为已折叠进 parent（增量续算入口），
        start 为负 → ValueError。返回 (keys, last_parent)：尾部不足
        一个 chunk 的 token 舍弃；无完整 chunk 时 keys 为空、
        last_parent 原样返回 parent。
        """
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start!r}")
        keys: list[bytes] = []
        step = self._chunk_tokens
        for i in range(start, len(token_ids) - step + 1, step):
            parent = _chunk_digest(parent, token_ids[i:i + step])
            keys.append(parent)
        return keys, parent

    # ---- 写入计划（两阶段）----

    def plan_store(self, keys: Iterable[bytes]) -> StorePlan | None:
        """受理一批写入并预留容量（两阶段的第一阶段）。

        返回 StorePlan：new_keys 为需要写入的 key（批内已驻留的除外，
        批内重复 key 去重），evicted_keys 为被驱逐腾位的 key；受理的
        new_keys 进入在途状态，由 confirm_store 结算。
        返回 None（不改变任何状态）当且仅当：
        - 批内任一 key 已处于在途状态（在途重复计划，整批不受理）；
        - 容量不足且可驱逐的驻留 key 不够（其余全部受 pin 保护）。
        """
        batch = list(dict.fromkeys(keys))
        if any(k in self._pending for k in batch):
            return None
        new_keys = [k for k in batch if k not in self._resident]
        free = self._capacity - len(self._resident) - len(self._pending)
        deficit = len(new_keys) - free
        evicted: list[bytes] = []
        if deficit > 0:
            for k in self._resident:  # 最旧在前
                if len(evicted) == deficit:
                    break
                if self._pins.get(k, 0) == 0:
                    evicted.append(k)
            if len(evicted) < deficit:
                return None
            for k in evicted:
                del self._resident[k]
        self._pending.update(new_keys)
        return StorePlan(new_keys=new_keys, evicted_keys=evicted)

    def confirm_store(self, keys: Iterable[bytes], ok: bool = True) -> None:
        """结算一批写入（两阶段的第二阶段）。

        ok=True：key 结算为驻留并置为最近使用；ok=False：回收容量
        预留、不驻留。不在途的 key 按同样规则处理（幂等，重复结算
        无副作用）；调用方应只结算自己受理的批次。
        """
        for k in keys:
            self._pending.discard(k)
            if ok:
                if k in self._resident:
                    self._resident.move_to_end(k)
                else:
                    self._resident[k] = None

    # ---- 读保护 ----

    def pin(self, keys: Iterable[bytes]) -> None:
        """对一批 key 加读保护（计数制，同一 key 可重复 pin）。

        任一 key 不驻留 → KeyError，整批不生效。
        """
        batch = list(keys)
        missing = [k for k in batch if k not in self._resident]
        if missing:
            raise KeyError(missing)
        for k in batch:
            self._pins[k] += 1

    def unpin(self, keys: Iterable[bytes]) -> None:
        """解除读保护，与 pin 配对使用；计数归零即失去驱逐保护。

        对当前无 pin 计数的 key 调用 → KeyError。
        """
        for k in keys:
            if self._pins.get(k, 0) == 0:
                raise KeyError(k)
            self._pins[k] -= 1
            if self._pins[k] == 0:
                del self._pins[k]

    # ---- LRU ----

    def mark_recent(self, keys: Iterable[bytes]) -> None:
        """把一批 key 刷新为最近使用（调整 LRU 顺序）。

        非驻留的 key 静默忽略。
        """
        for k in keys:
            if k in self._resident:
                self._resident.move_to_end(k)

    # ---- 冷启动 ----

    def restore(self, keys: Iterable[bytes]) -> None:
        """灌入一批存活 key，全部置为驻留（冷启动重建入口）。

        幂等：重复灌入不重复计数、不改变既有 LRU 顺序；首次灌入按
        给定顺序建立初始 LRU 序（排前者更旧）。恰处于在途状态的 key
        一并结算为驻留。
        """
        for k in keys:
            self._pending.discard(k)
            if k not in self._resident:
                self._resident[k] = None

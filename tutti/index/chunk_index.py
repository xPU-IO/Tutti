"""语义索引：chunk key 的驻留 / pin / pending 状态、前缀命中与 LRU 驱逐决策。

纯逻辑组件，单线程使用，不加锁；只维护 key 与状态，不接触数据本体。
token 序列按 chunk_tokens 切块并链式哈希为 key 序列，key 即跨进程
一致的 chunk 身份。

术语表（评审 N3：chunk / block / page 三个词此前混用）
-----------------------------------------------------
- **token**：模型输入的最小单位，也是 chunk key 的派生依据。
- **block**：vLLM 的 KV 分配单位，``block_size`` 个 token（生产为 64）；
  vLLM 的 block table 决定请求的 KV 落在哪些物理块上。
- **chunk**：**本 connector 的落盘/复用单位**，``chunk_tokens`` 个 token
  （生产为 256 = 4 个 block）。chunk key 由 token 前缀链式哈希派生，
  故相同前缀的请求命中同一个 chunk。
- **page**：物理页，指一个 block 在**单层**上的 KV 字节
  （``page_size_bytes``）；``segment_bytes = blocks_per_chunk × page_size_bytes``
  是 chunk 在单层的字节数，``chunk_kv_bytes = num_layers × segment_bytes``。
- **io_key**：``chunk_key(16B) + layer(2B)`` 的线格式（``IO_KEY_BYTES``），
  数据面最小的可寻址/可删除单位——按层落盘、按层校验。

命名约定：容量类字段以 chunk 计数（capacity / resident / pending / pinned），
字节类字段一律带 ``_bytes`` 后缀。
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


# io_key 中层编号的编码宽度（字节，小端）
_IO_LAYER_BYTES = 2

#: io_key 的线格式长度（chunk key 16B + 层编号 2B）。对外公开，
#: 供扫描/校验等工具共用，避免各处重复字面量 18。
IO_KEY_BYTES = _KEY_BYTES + _IO_LAYER_BYTES


def decode_io_key(io_key: bytes) -> tuple[bytes, int]:
    """io_key → (chunk 身份, 层号)。

    任意长度 key 均可解读（chunk 取前 16 字节，其余按小端表示层号），与
    ``derive_io_key`` 互逆：短 key 只有 chunk、层号为 0（通用 KV 存储契约
    允许这种 key）。层号宽度超出 int 的 key 会 ValueError。
    """
    if not isinstance(io_key, (bytes, bytearray, memoryview)):
        raise ValueError(f"io_key 必须为 bytes，得到 {type(io_key).__name__}")
    io_key = bytes(io_key)
    if not io_key:
        raise ValueError("io_key 不能为空")
    chunk_id = io_key[:_KEY_BYTES]
    layer = int.from_bytes(io_key[_KEY_BYTES:], "little")
    return chunk_id, layer


def derive_io_key(chunk_key: bytes, layer_idx: int) -> bytes:
    """把层编号编入 chunk key，得到存储层使用的 io_key（18 字节）。

    chunk_key 长度须为 16 字节、layer_idx 须落在 2 字节无符号整数
    范围内，否则 ValueError。
    """
    if not isinstance(chunk_key, (bytes, bytearray)) or len(chunk_key) != _KEY_BYTES:
        raise ValueError(f"chunk_key 须为 16 字节，got {chunk_key!r}")
    if not isinstance(layer_idx, int) or isinstance(layer_idx, bool):
        raise ValueError(f"layer_idx 须为整数，got {layer_idx!r}")
    if not 0 <= layer_idx < (1 << (8 * _IO_LAYER_BYTES)):
        raise ValueError(f"layer_idx 超出编码范围，got {layer_idx!r}")
    return bytes(chunk_key) + layer_idx.to_bytes(_IO_LAYER_BYTES, "little")


def chunk_key_of(io_key: bytes) -> bytes:
    """从 io_key 取回 chunk key（前 16 字节）；长度非法 → ValueError。"""
    _require_io_key(io_key)
    return bytes(io_key[:_KEY_BYTES])


def layer_of(io_key: bytes) -> int:
    """从 io_key 取回层编号（末 2 字节小端整数）；长度非法 → ValueError。"""
    _require_io_key(io_key)
    return int.from_bytes(io_key[_KEY_BYTES:], "little")


def _require_io_key(io_key) -> None:
    """校验 io_key 形态：18 字节的字节串。"""
    if not isinstance(io_key, (bytes, bytearray)):
        raise ValueError(f"io_key 须为字节串，got {io_key!r}")
    if len(io_key) != _KEY_BYTES + _IO_LAYER_BYTES:
        raise ValueError(
            f"io_key 须为 {_KEY_BYTES + _IO_LAYER_BYTES} 字节，got {len(io_key)}"
        )


class ChunkIndex:
    """语义索引：key → 驻留 / pin / pending。

    契约：
    - key 链：key_i = blake2b(parent_{i-1} + tokens_i)，parent_0 为
      namespace（缺省空 bytes），parent_i = key_i；token 以 8 字节
      小端编码。同一 token 序列在同一 namespace 的任何进程得到
      相同 key；namespace 由部署层按模型/dtype/TP/几何组装为不
      透明串（本层只哈希、不解读），不同 namespace 的 key 天然
      隔离。尾部不足一个 chunk 的 token 舍弃。
    - 状态三种：resident（驻留，参与命中与 LRU）、pending（写入已
      受理未结算：不参与命中、不参与驱逐、占用容量）、pin 计数
      （大于 0 即受驱逐保护）。
    - 驱逐只发生在 plan_store；pin 计数大于 0 的 key 永不驱逐。
    """

    def __init__(self, capacity: int, chunk_tokens: int,
                 namespace: bytes = b""):
        """capacity 为可容纳的 chunk 总数，chunk_tokens 为每 chunk 的 token 数。

        namespace 为 key 链的不透明前缀（字节串；空 = 无命名空间，
        兼容无部署上下文的进程内使用）。参数非法 → ValueError。
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity!r}")
        if chunk_tokens <= 0:
            raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens!r}")
        if not isinstance(namespace, (bytes, bytearray)):
            raise ValueError(f"namespace 须为字节串，got {namespace!r}")
        self._capacity = capacity
        self._chunk_tokens = chunk_tokens
        self._namespace = bytes(namespace)
        # 驻留表：插入序即 LRU 序（最旧在前）。
        self._resident: OrderedDict[bytes, None] = OrderedDict()
        # 在途表：plan_store 受理、confirm_store 尚未结算；值为受理时的
        # epoch，用于回收"永远等不到结算"的预留（见 reclaim_stale_pending）。
        self._pending: dict[bytes, int] = {}
        # pin 计数：条目存在即计数 ≥ 1。
        self._pins: Counter[bytes] = Counter()
        # 步进计数：由调用方（调度侧每步一次）推进，是 pending 步龄的时基。
        self._epoch = 0

    @property
    def capacity(self) -> int:
        """可容纳的 chunk 总数。"""
        return self._capacity

    @property
    def chunk_tokens(self) -> int:
        """每 chunk 的 token 数。"""
        return self._chunk_tokens

    def is_resident(self, key: bytes) -> bool:
        """该 key 是否已发布驻留（不含在途/pending）。"""
        return key in self._resident

    def stats(self) -> dict[str, int]:
        """容量快照（可观测性：每步汇总与压测断言共用）。"""
        return {
            "capacity": self.capacity,
            "resident": len(self._resident),
            "pending": len(self._pending),
            "pinned": len(self._pins),
            "epoch": self._epoch,
        }

    def is_pinned(self, key: bytes) -> bool:
        """该 key 是否处于读保护中（在途读取期间不得删除数据）。"""
        return self._pins.get(key, 0) > 0

    @property
    def epoch(self) -> int:
        """当前步进计数（pending 步龄的时基）。"""
        return self._epoch

    def advance_epoch(self) -> int:
        """推进一个步进（调度侧每步调用一次）。"""
        self._epoch += 1
        return self._epoch

    def reclaim_stale_pending(self, max_age: int) -> list[bytes]:
        """回收在途超过 max_age 个步进仍未结算的预留，返回被回收的 key。

        正常路径下 pending 会在同一批的 confirm_store 结算（TP rank 的
        完成可能跨步，故 max_age 需覆盖该窗口）。但若某个 rank 从未回报
        （进程崩溃/重启、请求被抢占后不再 save），该项会永久占用容量：
        ``free = capacity - len(resident) - len(pending)`` 单调下降，最终
        写入被静默拒绝。这里按步龄兜底回收。

        **max_age 必须足够大**：过小会误回收正常跨步结算的预留，随后该
        key 的 confirm_store 找不到在途项，驻留发布落空——那比容量泄漏
        更糟（会造成调度侧认为未驻留而重复重算）。
        """
        if max_age <= 0 or not self._pending:
            return []
        cutoff = self._epoch - max_age
        stale = [key for key, planned in self._pending.items()
                 if planned <= cutoff]
        for key in stale:
            del self._pending[key]
        return stale

    # ---- 查询 ----

    def lookup_prefix(self, token_ids: Sequence[int]) -> int:
        """返回 token_ids 前缀中连续命中的 token 数（链式滚动）。

        从 namespace 起逐 chunk 计算 key：驻留则推进并累计 token 数，
        首个未驻留的 chunk 即停止。在途（pending）chunk 不算命中。
        命中即刷新 LRU（读热者存活，退化插入序）。
        """
        matched = 0
        parent = self._namespace
        step = self._chunk_tokens
        for i in range(0, len(token_ids) - step + 1, step):
            parent = _chunk_digest(parent, token_ids[i:i + step])
            if parent not in self._resident:
                break
            matched += step
            self._resident.move_to_end(parent)
        return matched

    def hash_keys(
        self,
        token_ids: Sequence[int],
        start: int = 0,
        parent: bytes | None = None,
    ) -> tuple[list[bytes], bytes]:
        """把 token_ids 从下标 start 起折叠为 key 序列。

        start 之前的 token 视为已折叠进 parent（增量续算入口），
        start 为负 → ValueError。返回 (keys, last_parent)：尾部不足
        一个 chunk 的 token 舍弃；无完整 chunk 时 keys 为空、
        last_parent 原样返回 parent（未给 parent 时自 namespace 起）。
        """
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start!r}")
        if parent is None:
            parent = self._namespace
        keys: list[bytes] = []
        step = self._chunk_tokens
        for i in range(start, len(token_ids) - step + 1, step):
            parent = _chunk_digest(parent, token_ids[i:i + step])
            keys.append(parent)
        return keys, parent

    @property
    def namespace(self) -> bytes:
        """key 链的不透明前缀（部署层组装，本层不解读）。"""
        return self._namespace

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
        for key in new_keys:
            self._pending[key] = self._epoch
        return StorePlan(new_keys=new_keys, evicted_keys=evicted)

    def confirm_store(self, keys: Iterable[bytes], ok: bool = True) -> None:
        """结算一批写入（两阶段的第二阶段）。

        ok=True：key 结算为驻留并置为最近使用；ok=False：回收容量
        预留、不驻留。不在途的 key 按同样规则处理（幂等，重复结算
        无副作用）；调用方应只结算自己受理的批次。
        """
        for k in keys:
            self._pending.pop(k, None)
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
            self._pending.pop(k, None)
            if k not in self._resident:
                self._resident[k] = None

    def forget(self, keys: Iterable[bytes]) -> list[bytes]:
        """按完整性翻转移除一批近似驻留项。

        适用于"曾判完整、现已消失"的组（对账方持有前后两次的完整
        组差集）；读保护（pin）中的 key 跳过——在途读取由 miss 降级
        兜底，强制移除会破坏 unpin 配对。非驻留 key 静默忽略。
        返回因 pin 保护而未能移除的 key（调用方应留存至下次对账
        重试，避免翻转事实随基准推进丢失）。
        """
        kept: list[bytes] = []
        for k in keys:
            if self._pins.get(k, 0) > 0:
                kept.append(k)
            else:
                self._resident.pop(k, None)
        return kept

"""KVStore SPI：存储插件协议（KVStore + Completion）。

批条目形如 (key, buffer, offset)：key 是存储身份，buffer 是
register_buffer 返回的 buffer id，offset 是 buffer 内字节偏移。
一次搬运的字节长度由实现与调用方在构造期对齐（实现的入参），
协议本身不另行表达。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol


class KVStore(Protocol):
    """注册 buffer ↔ 持久化 之间的 chunk 搬运契约。

    key 即存储身份；位置解释权归实现私有。
    IO 端点以 (buffer, offset) 表达——buffer 由实现注册并解释。
    **批是一等公民**：一次调用 = 一批条目 = 底层一次提交
    （tutti 的 fused launch 模型），逐条调用形态不存在。
    """

    @property
    def capacity_chunks(self) -> int:
        """可容纳的 chunk 总数（engine 据此建索引容量）。"""

    def open(self) -> None: ...

    def close(self) -> None: ...  # 幂等

    def register_buffer(self, buffer, granularity: int) -> int | None:
        """注册一块内存为 IO buffer（staging 环窗 / paged 池皆是）。
        返回 buffer id；不接受该粒度 → None（engine 回退 staged）。
        注册代价与描述符策略为实现私有。"""

    def put_batch(self, batch: Sequence[tuple[bytes, int, int]]) -> Completion:
        """异步一批：[(key, buffer, offset)] → 持久化。
        一个 Completion 覆盖整批；部分拒收由实现内部重发消化。"""

    def get_batch(self, batch: Sequence[tuple[bytes, int, int]]) -> Completion:
        """异步一批：持久化 → buffer 内 offset。一个 Completion 覆盖整批。"""

    def drop(self, keys: Sequence[bytes]) -> None:
        """批量驱逐执行（index 决策，store 只执行）。"""

    def scan(self) -> Iterable[bytes]:
        """枚举当前存活的 key（冷启动恢复用；MemoryKVStore 返回空）。"""


class Completion(Protocol):
    """异步批次的完成句柄：wait 阻塞至整批完成；query 非阻塞查询完成与否。"""

    def wait(self) -> None: ...

    def query(self) -> bool: ...

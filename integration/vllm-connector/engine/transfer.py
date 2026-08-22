"""传输路径：staged 兜底路径与 bind 期的路径定案。"""

from __future__ import annotations

from typing import Sequence


class StagedTransfer:
    """兜底传输路径：数据经 staging 槽中转，两端搬运以钩子注入。

    钩子契约（均为可调用或 None，None 时该侧搬运为 no-op）：
    - gather_fn(keys, layer_idx, first_blocks, slots)：store 方向提交前，
      把源侧一层段搬入给定 staging 槽。
    - scatter_fn(keys, layer_idx, first_blocks, slots)：load 方向完成句柄
      wait 后，把 staging 槽内容搬往目的侧。
    """

    def __init__(self, gather_fn=None, scatter_fn=None):
        """注入两侧搬运钩子；钩子须为可调用，否则 ValueError。"""
        for name, fn in (("gather_fn", gather_fn), ("scatter_fn", scatter_fn)):
            if fn is not None and not callable(fn):
                raise ValueError(f"{name} 须为可调用或 None，got {fn!r}")
        self._gather_fn = gather_fn
        self._scatter_fn = scatter_fn

    def gather(
        self,
        keys: Sequence[bytes],
        layer_idx: int,
        first_blocks,
        slots: Sequence[int],
    ) -> None:
        """执行 store 方向的源侧搬运（钩子缺省为 no-op）。"""
        if self._gather_fn is not None:
            self._gather_fn(list(keys), layer_idx, first_blocks, list(slots))

    def scatter(
        self,
        keys: Sequence[bytes],
        layer_idx: int,
        first_blocks,
        slots: Sequence[int],
    ) -> None:
        """执行 load 方向的目的侧搬运（钩子缺省为 no-op）。"""
        if self._scatter_fn is not None:
            self._scatter_fn(list(keys), layer_idx, first_blocks, list(slots))


def select_transfer(kv_caches, store, config: dict) -> StagedTransfer:
    """bind 期一次性定案传输路径。

    契约：接受布局对象、store 实例与引擎配置，返回选定路径且此后
    固定不变。当前恒返回 staged 兜底路径；config 中的 gather_fn /
    scatter_fn 原样注入为搬运钩子。
    """
    return StagedTransfer(config.get("gather_fn"), config.get("scatter_fn"))

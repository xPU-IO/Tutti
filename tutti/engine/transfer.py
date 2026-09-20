"""传输路径：direct paged backend 与 staged 兜底路径。"""

from __future__ import annotations

import logging
from typing import Sequence

from tutti.common.errors import DirectTransferUnavailable

_LOG = logging.getLogger(__name__)


class DirectTransfer:
    """Adapter for a store-native paged KV DMA implementation.

    The legacy GeminiFS path registers the vLLM paged tensors once and submits
    block-table aware IO directly against those tensors.  The generic KVStore
    SPI cannot express that addressing with ``(buffer, offset)`` entries, so
    stores opt into this path by returning a backend from
    ``create_direct_transfer``.  The backend must implement
    ``register_paged_caches``, ``get_paged_batch`` and ``put_paged_batch``.
    """

    direct = True

    def __init__(
        self,
        backend,
        kv_caches,
        *,
        num_layers: int,
        blocks_per_chunk: int,
        chunk_tokens: int,
        segment_bytes: int,
        max_chunks_per_wave: int | None = None,
    ):
        self._backend = backend
        register = getattr(backend, "register_paged_caches", None)
        if not callable(register):
            raise DirectTransferUnavailable(
                "direct backend lacks register_paged_caches"
            )
        kwargs = dict(
            num_layers=num_layers,
            blocks_per_chunk=blocks_per_chunk,
            chunk_tokens=chunk_tokens,
            segment_bytes=segment_bytes,
            max_chunks_per_wave=max_chunks_per_wave,
        )
        accepted = register(kv_caches, **kwargs)
        if accepted is False:
            raise DirectTransferUnavailable(
                "direct backend rejected paged cache registration"
            )

    def warm_up_registration(self) -> bool:
        """Force the backend's lazy per-DataPath memory registration now.

        Registration normally happens inside the first ``submit``; doing it
        at bind time keeps the multi-hundred-millisecond peer-memory map
        (which holds the runtime registry lock) out of the first request.
        """
        method = getattr(self._backend, "warm_up_registration", None)
        if not callable(method):
            return False
        return bool(method())

    def load_layer(self, keys, layer_idx: int, block_tables):
        method = getattr(self._backend, "get_paged_batch", None)
        if not callable(method):
            raise DirectTransferUnavailable(
                "direct backend lacks get_paged_batch"
            )
        return method(list(keys), layer_idx, list(block_tables))

    def store_layer(self, keys, layer_idx: int, block_tables):
        method = getattr(self._backend, "put_paged_batch", None)
        if not callable(method):
            raise DirectTransferUnavailable(
                "direct backend lacks put_paged_batch"
            )
        return method(list(keys), layer_idx, list(block_tables))

    def validate_block_tables(self, block_tables) -> None:
        method = getattr(self._backend, "validate_block_tables", None)
        if callable(method):
            method(list(block_tables))

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()


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
    ):
        """执行 store 方向的源侧搬运（钩子缺省为 no-op）。"""
        if self._gather_fn is not None:
            return self._gather_fn(
                list(keys), layer_idx, first_blocks, list(slots)
            )
        return None

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


def select_transfer(
    kv_caches,
    store,
    config: dict,
    *,
    num_layers: int | None = None,
    blocks_per_chunk: int | None = None,
    chunk_tokens: int | None = None,
    segment_bytes: int | None = None,
    max_chunks_per_wave: int | None = None,
):
    """bind 期一次性定案传输路径。

    **准入失败不再静默回退到 staged。** 这是本函数唯一的行为变更，也是这次要
    解决的真正问题：以前 store 声明了直连能力却准入失败时，会退到 staged 暂存
    路径（KV 数据经主机内存或 CUDA staging 缓冲往返），既与 GPU-direct 架构方向
    相反，又把"直连其实没生效"伪装成"跑得有点慢"，在生产里极难发现。staged 自身
    还持续制造缺陷——staging 张量形状、``shape[2] == 2`` 的巧合判定、packed 与
    split K/V 两种布局，三处问题都源于它。

    两种"用 staged"的情形要分开看：

    - **store 未提供 create_direct_transfer**：它没有直连能力。这是能力协商的
      结果，不是降级——测试替身、以及将来可能的其他后端本就不提供它。
      ``direct_transfer=true`` 可把这种情形变成显式错误。
    - **store 提供了直连但准入失败**：一律抛出。部署中的 store 恒属此类，故
      staged 在生产路径上不可达。
    """
    factory = getattr(store, "create_direct_transfer", None)
    if not callable(factory):
        if config.get("direct_transfer") is True:
            raise DirectTransferUnavailable(
                "direct_transfer=true 但 store 未提供 create_direct_transfer"
            )
        return StagedTransfer(config.get("gather_fn"), config.get("scatter_fn"))

    backend = factory(
        kv_caches,
        num_layers=num_layers,
        blocks_per_chunk=blocks_per_chunk,
        chunk_tokens=chunk_tokens,
        segment_bytes=segment_bytes,
    )
    if backend is None:
        # store 有能力但对这组 caches 拒绝受理，同属能力协商范畴。部署中的
        # store 恒返回 backend（store.py 的 create_direct_transfer 不返回 None），
        # 故生产路径上不会走到这里。
        if config.get("direct_transfer") is True:
            raise DirectTransferUnavailable(
                "direct_transfer=true 但 store 未受理直连注册"
            )
        return StagedTransfer(config.get("gather_fn"), config.get("scatter_fn"))

    try:
        return DirectTransfer(
            backend,
            kv_caches,
            num_layers=num_layers,
            blocks_per_chunk=blocks_per_chunk,
            chunk_tokens=chunk_tokens,
            segment_bytes=segment_bytes,
            max_chunks_per_wave=max_chunks_per_wave,
        )
    except DirectTransferUnavailable:
        # 后端已构造，DirectTransfer 的准入校验失败：释放它再抛，避免泄漏。
        close = getattr(backend, "close", None)
        if callable(close):
            close()
        raise

"""传输路径：direct paged backend 与 staged 兜底路径。"""

from __future__ import annotations

import logging
from typing import Sequence


_LOG = logging.getLogger(__name__)


class DirectTransferUnavailable(RuntimeError):
    """Raised when a store cannot expose a direct paged-memory backend."""


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

    契约：接受布局对象、store 实例与引擎配置，返回选定路径且此后
    固定不变。store 暴露 ``create_direct_transfer`` 时默认先尝试 direct；
    ``direct_transfer=false`` 可显式关闭。声明的准入失败自动回退 staged，
    ``direct_transfer_strict=true`` 则保留具体失败原因并抛出。
    """
    if config.get("direct_transfer", True):
        factory = getattr(store, "create_direct_transfer", None)
        if callable(factory):
            backend = factory(
                kv_caches,
                num_layers=num_layers,
                blocks_per_chunk=blocks_per_chunk,
                chunk_tokens=chunk_tokens,
                segment_bytes=segment_bytes,
            )
            if backend is not None:
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
                except DirectTransferUnavailable as exc:
                    close = getattr(backend, "close", None)
                    if callable(close):
                        close()
                    if config.get("direct_transfer_strict"):
                        raise
                    _LOG.warning(
                        "DIRECT_ADMISSION_FALLBACK reason=%s",
                        exc,
                    )
        elif config.get("direct_transfer_strict"):
            raise DirectTransferUnavailable(
                "store lacks create_direct_transfer"
            )
        elif config.get("direct_transfer") is True:
            _LOG.warning(
                "DIRECT_ADMISSION_FALLBACK reason=store lacks "
                "create_direct_transfer"
            )
        if config.get("direct_transfer_strict"):
            raise DirectTransferUnavailable(
                "store did not accept direct paged cache registration"
            )
    elif config.get("direct_transfer_strict"):
        raise DirectTransferUnavailable(
            "direct_transfer_strict requires direct transfer to be enabled"
        )
    return StagedTransfer(config.get("gather_fn"), config.get("scatter_fn"))

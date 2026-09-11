# SPDX-License-Identifier: Apache-2.0
"""gather/scatter CUDA kernels for the Tutti vLLM connector.

Minimal port of LMCache's ``single_layer_kv_transfer``
(third_pkgs/LMCache/csrc/cuda/mem_kernels.cu, Apache-2.0) as a standalone
torch cpp_extension. Depends only on torch — no vllm, no tutti_runtime.

Public API:
- :func:`single_layer_transfer` — one packed layer segment <-> one vLLM
  paged layer tensor, per-token via ``slot_mapping`` (``-1`` skips).
- :func:`discover_engine_format` — shape/stride based layout inference.
- :class:`EngineKVFormat` — format int constants (csrc/engine_kv_format.h).
"""

import torch

from ._discovery import EngineKVFormat, discover_engine_format
from . import _native

__all__ = [
    "EngineKVFormat",
    "discover_engine_format",
    "single_layer_transfer",
    "batched_layer_transfer",
]

# TransferDirection (csrc/kv_transfer_types.h)
_D2H = 1  # paged -> staging (gather)
_H2D = 0  # staging -> paged (scatter)

_DIRECTIONS = {"to_staging": _D2H, "to_paged": _H2D}

_SUPPORTED_FORMATS = (
    EngineKVFormat.FA_NHD,
    EngineKVFormat.FI_NHD,
    EngineKVFormat.MLA,
    EngineKVFormat.FA_HND,
    EngineKVFormat.FI_HND,
)


def _infer_token_major(staging_view: torch.Tensor, fmt: int) -> bool:
    """[tokens, 2, H] -> True; [2, tokens, H] -> False; MLA ignores it."""
    if fmt == EngineKVFormat.MLA:
        return True
    if staging_view.dim() == 3:
        if staging_view.shape[1] == 2:
            return True  # [num_tokens, 2, num_heads*head_size]
        if staging_view.shape[0] == 2:
            return False  # [2, num_tokens, num_heads*head_size]
    raise ValueError(
        "cannot infer token_major from staging_view with shape "
        f"{tuple(staging_view.shape)} (expected [tokens, 2, H], "
        "[2, tokens, H], or MLA [tokens, H])"
    )


def single_layer_transfer(
    staging_view: torch.Tensor,
    paged_view: torch.Tensor,
    slot_mapping: torch.Tensor,
    engine_kv_format: int,
    direction: str,
) -> None:
    """Copy one layer between a packed staging segment and a paged KV tensor.

    Args:
        staging_view: packed single-layer segment — ``[tokens, 2, H]``
            (token-major K/V) or ``[2, tokens, H]`` (K/V-major) with
            ``H = num_heads * head_size``; MLA uses ``[tokens, H]``.
            Must be CUDA (or pinned-CPU) and contiguous along the packed
            token row (row stride is honoured).
        paged_view: the engine's single-layer paged tensor matching
            ``engine_kv_format`` (physical layout; for HND formats pass
            the physically head-major tensor, not a logically-NHD
            permuted view — see :func:`discover_engine_format`).
        slot_mapping: int64 ``[tokens]``; flat slot id
            (``block_id * block_size + offset``) per token, ``-1`` skips
            the token in both directions.
        engine_kv_format: one of the :class:`EngineKVFormat` ints.
        direction: ``"to_staging"`` (gather, paged -> staging) or
            ``"to_paged"`` (scatter, staging -> paged).
    """
    if not isinstance(direction, str) or direction not in _DIRECTIONS:
        raise ValueError(
            f"direction must be 'to_staging' or 'to_paged', got {direction!r}"
        )
    if engine_kv_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"unsupported engine_kv_format {engine_kv_format!r}; "
            f"supported: {_SUPPORTED_FORMATS}"
        )
    if slot_mapping.dtype != torch.int64 or slot_mapping.dim() != 1:
        raise ValueError(
            "slot_mapping must be a 1-D int64 tensor, got "
            f"{slot_mapping.dtype} with shape {tuple(slot_mapping.shape)}"
        )
    if paged_view.dtype != staging_view.dtype:
        raise ValueError(
            f"dtype mismatch: staging {staging_view.dtype} vs "
            f"paged {paged_view.dtype}"
        )
    if paged_view.element_size() not in (1, 2, 4, 8):
        raise ValueError(
            f"unsupported element size {paged_view.element_size()}"
        )
    # The kernel addresses in 8-byte units; a partial trailing qword would
    # silently drop data.
    head_dim = paged_view.shape[-1]
    if head_dim * paged_view.element_size() % 8 != 0:
        raise ValueError(
            f"last dim {head_dim} x {paged_view.element_size()}B is not a "
            "multiple of the 8-byte kernel granularity"
        )

    _native.single_layer_kv_transfer(
        staging_view,
        paged_view,
        slot_mapping,
        _DIRECTIONS[direction],
        int(engine_kv_format),
        _infer_token_major(staging_view, engine_kv_format),
    )


def batched_layer_transfer(
    staging_view: torch.Tensor,
    paged_view: torch.Tensor,
    slot_mapping: torch.Tensor,
    engine_kv_format: int,
    direction: str,
) -> None:
    """Copy multiple chunks for one layer with one CUDA kernel launch.

    ``staging_view`` and ``slot_mapping`` contain the concatenated token rows
    of all chunks in the batch.  The native kernel is already token-count
    agnostic, so this entry point makes the batching contract explicit while
    retaining all validation and layout handling of :func:`single_layer_transfer`.
    """
    single_layer_transfer(
        staging_view, paged_view, slot_mapping, engine_kv_format, direction
    )

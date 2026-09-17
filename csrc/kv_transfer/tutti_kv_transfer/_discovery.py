# SPDX-License-Identifier: Apache-2.0
"""Layout discovery for single-layer vLLM paged KV tensors.

Minimal port of the LMCache detection idea
(lmcache/v1/gpu_connector/utils.py:186 normalize_kv_and_discover_format ->
kv_format/detection.py + kv_format/contiguity.py), covering the four
commonly used layouts:

- FA NHD  (NL_X_TWO_NB_BS_NH_HS, 1): [2, nb, bs, nh, hs] contiguous
- FI NHD  (NL_X_NB_TWO_BS_NH_HS, 2): [nb, 2, bs, nh, hs] contiguous
- MLA     (NL_X_NB_BS_HS,       3): [nb, bs, hs]
- HND     (NL_X_TWO_NB_NH_BS_HS, 6 / NL_X_NB_TWO_NH_BS_HS, 7):
  physically [2, nb, nh, bs, hs] / [nb, 2, nh, bs, hs]. vLLM hands these
  out as a *permuted view* with logical NHD shape, so a non-monotonic
  stride that permutes back to a contiguous view identifies them.

Everything else raises ``AssertionError``.
"""

import torch


class EngineKVFormat:
    """int values of csrc/engine_kv_format.h's EngineKVFormat enum (subset
    supported by single_layer_kv_transfer)."""

    FA_NHD = 1  # NL_X_TWO_NB_BS_NH_HS — vLLM flash-attn  [2, nb, bs, nh, hs]
    FI_NHD = 2  # NL_X_NB_TWO_BS_NH_HS — vLLM flash-infer [nb, 2, bs, nh, hs]
    MLA = 3     # NL_X_NB_BS_HS        — vLLM MLA          [nb, bs, hs]
    FA_HND = 6  # NL_X_TWO_NB_NH_BS_HS — vLLM flash-attn HND  [2, nb, nh, bs, hs]
    FI_HND = 7  # NL_X_NB_TWO_NH_BS_HS — vLLM flash-infer HND [nb, 2, nh, bs, hs]


def _stride_sort_perm(tensor: torch.Tensor) -> list[int]:
    """Dims ordered by descending stride (LMCache contiguity.py idea)."""
    stride = tensor.stride()
    return sorted(range(tensor.dim()), key=lambda i: stride[i], reverse=True)


def _permute_to_contiguous_view(tensor: torch.Tensor):
    """Zero-copy recovery of the physical layout (LMCache
    attempt_permute_to_contiguous_view, minimal form).

    Returns ``(view, permuted)``: if the stride order differs from the
    logical dim order and re-permuting yields a contiguous tensor, the
    physical view is returned with ``permuted=True``; otherwise the
    original tensor is returned with ``permuted=False``.
    """
    perm = _stride_sort_perm(tensor)
    if perm == list(range(tensor.dim())):
        return tensor, False
    candidate = tensor.permute(perm)
    if candidate.is_contiguous():
        return candidate, True
    # Non-recoverable strided view (e.g. dim-0 padded pools): keep as-is.
    return tensor, False


def discover_engine_format(paged_tensor: torch.Tensor, use_mla: bool) -> int:
    """Infer the EngineKVFormat int for one layer's vLLM paged KV tensor.

    Args:
        paged_tensor: a single layer's paged KV tensor as registered by the
            engine (shape/stride are inspected; storage is untouched).
        use_mla: caller-side declaration that the model uses MLA. Must be
            consistent with what the shape implies (3-D => MLA, 5-D => not).

    Returns:
        EngineKVFormat int (see :class:`EngineKVFormat`).

    Raises:
        AssertionError: on any structure outside the four covered layouts,
            or when ``use_mla`` contradicts the tensor rank.

    Note:
        A physically-contiguous HND tensor and an NHD tensor have identical
        shapes/strides (the two middle axes are indistinguishable), exactly
        as in LMCache's detector — that case needs an engine-side hint and
        is reported as NHD here. The permuted-view HND form vLLM actually
        hands out is detected unambiguously via the stride order.
    """
    tensor, permuted = _permute_to_contiguous_view(paged_tensor)

    if tensor.dim() == 3:
        assert use_mla, (
            "3-D paged KV tensor [nb, bs, hs] implies MLA; pass use_mla=True "
            "(other 3-D layouts, e.g. DSA indexer caches, are unsupported)"
        )
        return EngineKVFormat.MLA

    assert tensor.dim() == 5, (
        f"unsupported paged KV tensor rank {tensor.dim()} "
        f"(shape={tuple(paged_tensor.shape)})"
    )
    assert not use_mla, (
        "use_mla=True but paged tensor is 5-D "
        "(MLA paged cache is 3-D [nb, bs, hs])"
    )

    if permuted:
        # vLLM exposes a physically HND allocation as a logically-NHD
        # permuted view; the stride order betrays it (LMCache
        # contiguity.py documents this exact scheme).
        hnd = True
    else:
        hnd = False

    if tensor.shape[0] == 2:
        # K/V axis first: flash-attn family
        return EngineKVFormat.FA_HND if hnd else EngineKVFormat.FA_NHD
    if tensor.shape[1] == 2:
        # K/V axis second: flash-infer family
        return EngineKVFormat.FI_HND if hnd else EngineKVFormat.FI_NHD

    raise AssertionError(
        f"unrecognized 5-D paged KV layout: shape={tuple(paged_tensor.shape)}, "
        f"stride={tuple(paged_tensor.stride())} "
        "(expected size-2 K/V axis at position 0 (flash-attn) or 1 "
        "(flash-infer))"
    )

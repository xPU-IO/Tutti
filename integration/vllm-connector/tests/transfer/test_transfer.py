# SPDX-License-Identifier: Apache-2.0
"""Tests: LMCache single_layer_kv_transfer port (tutti_kv_transfer).

Run (GPU required for the kernel tests; they skip visibly without one):
    cd integration/vllm-connector && \
    /data/home/ryeqiu/tutti-env/bin/python -m pytest tests/transfer -v
"""

import sys
from pathlib import Path

import pytest
import torch

from tutti_kv_transfer import (
    EngineKVFormat,
    discover_engine_format,
    single_layer_transfer,
)

CSRC = Path(__file__).resolve().parents[2] / "transfer" / "csrc"
SENTINEL = 777.0

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device not available"
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def make_slot_mapping(tokens: int, num_slots: int, invalid: list[int]) -> torch.Tensor:
    """Random injective slot ids with the given token indices set to -1."""
    assert tokens <= num_slots
    slots = torch.randperm(num_slots)[:tokens].to(torch.int64)
    for i in invalid:
        slots[i] = -1
    return slots.cuda()


def valid_slots(slot_mapping: torch.Tensor) -> list[int]:
    return sorted({int(s) for s in slot_mapping.tolist() if s >= 0})


def _roundtrip(staging, paged, slot_mapping, fmt, tokens, invalid):
    """gather paged -> staging, then scatter staging -> fresh paged2.

    Returns (staging, paged2). Asserts the -1 skip semantics along the way.
    """
    # gather: paged -> staging
    single_layer_transfer(staging, paged, slot_mapping, fmt, "to_staging")
    torch.cuda.synchronize()
    # tokens with slot -1 are not read into staging: rows stay sentinel
    for i in invalid:
        assert torch.equal(
            staging[i], torch.full_like(staging[i], SENTINEL)
        ), f"token {i} (slot -1) must not be gathered"
    # valid rows actually received data
    ok = [i for i in range(tokens) if i not in invalid]
    assert not torch.equal(
        staging[ok], torch.full_like(staging[ok], SENTINEL)
    ), "valid rows must be filled by gather"

    # scatter: staging -> fresh paged2 (sentinel-filled)
    paged2 = torch.full_like(paged, SENTINEL)
    single_layer_transfer(staging, paged2, slot_mapping, fmt, "to_paged")
    torch.cuda.synchronize()
    return staging, paged2


def assert_bytes_equal(a: torch.Tensor, b: torch.Tensor, msg: str = "") -> None:
    """Bitwise (byte-level) equality of two same-shape/dtype tensors."""
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), msg


def check_roundtrip(paged, paged2, slot_mapping, slot_view, num_slots):
    """Valid slots round-trip byte-identically; untouched slots stay sentinel.

    ``slot_view(t, s)`` returns the view of one slot's data (shape identical
    across paged/paged2, e.g. ``[2, nh, hs]`` or ``[hs]``).
    """
    vs = valid_slots(slot_mapping)
    assert vs, "test bug: no valid slots"
    for slot in vs:
        assert_bytes_equal(
            slot_view(paged2, slot), slot_view(paged, slot),
            f"slot {slot} did not round-trip",
        )
    for slot in sorted(set(range(num_slots)) - set(vs)):
        got = slot_view(paged2, slot)
        assert torch.equal(got, torch.full_like(got, SENTINEL)), (
            f"untouched slot {slot} was written by scatter"
        )


# ----------------------------------------------------------------------
# import / license / hygiene
# ----------------------------------------------------------------------

def test_import_smoke():
    """Acceptance: `from tutti_kv_transfer import single_layer_transfer`."""
    from tutti_kv_transfer import single_layer_transfer as f

    assert callable(f)
    import tutti_kv_transfer._native as native

    assert hasattr(native, "single_layer_kv_transfer")


def test_apache_headers_and_notice():
    """csrc files keep the Apache-2.0 header; NOTICE present."""
    for name in ("mem_kernels.cu", "engine_kv_format.h", "kv_transfer_types.h"):
        src = (CSRC / name).read_text(encoding="utf-8")
        assert "SPDX-License-Identifier: Apache-2.0" in src, name
        assert "LMCache" in src, name
    notice = CSRC.parent / "NOTICE"
    assert notice.is_file()
    assert "LMCache" in notice.read_text(encoding="utf-8")


def test_no_vllm_import():
    """transfer 包自净：子进程隔离验证（同进程 adapter 测试会 import vllm）。"""
    import subprocess

    code = (
        "import sys; import tutti_kv_transfer; "
        "bad = [m for m in ('vllm', 'tutti_runtime') if m in sys.modules]; "
        "assert not bad, f'leaked imports: {bad}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ----------------------------------------------------------------------
# discover_engine_format
# ----------------------------------------------------------------------

def test_discover_fa_nhd_contiguous():
    t = torch.zeros(2, 4, 8, 2, 64)
    assert discover_engine_format(t, use_mla=False) == EngineKVFormat.FA_NHD


def test_discover_fi_nhd_contiguous():
    t = torch.zeros(4, 2, 8, 2, 64)
    assert discover_engine_format(t, use_mla=False) == EngineKVFormat.FI_NHD


def test_discover_mla():
    t = torch.zeros(4, 8, 576)
    assert discover_engine_format(t, use_mla=True) == EngineKVFormat.MLA


def test_discover_fa_hnd_permuted_view():
    """vLLM hands out physically-HND as a logically-NHD permuted view."""
    phys = torch.zeros(2, 4, 2, 8, 64)          # [2, nb, nh, bs, hs]
    view = phys.permute(0, 1, 3, 2, 4)          # logical [2, nb, bs, nh, hs]
    assert view.shape == (2, 4, 8, 2, 64)
    assert not view.is_contiguous()
    assert discover_engine_format(view, use_mla=False) == EngineKVFormat.FA_HND


def test_discover_fi_hnd_permuted_view():
    phys = torch.zeros(4, 2, 2, 8, 64)          # [nb, 2, nh, bs, hs]
    view = phys.permute(0, 1, 3, 2, 4)          # logical [nb, 2, bs, nh, hs]
    assert discover_engine_format(view, use_mla=False) == EngineKVFormat.FI_HND


def test_discover_rejects_bad_inputs():
    with pytest.raises(AssertionError):  # 3-D but use_mla=False
        discover_engine_format(torch.zeros(4, 8, 576), use_mla=False)
    with pytest.raises(AssertionError):  # 5-D but use_mla=True
        discover_engine_format(torch.zeros(2, 4, 8, 2, 64), use_mla=True)
    with pytest.raises(AssertionError):  # rank 4 unsupported
        discover_engine_format(torch.zeros(4, 8, 2, 64), use_mla=False)
    with pytest.raises(AssertionError):  # no size-2 K/V axis
        discover_engine_format(torch.zeros(3, 4, 8, 2, 64), use_mla=False)


# ----------------------------------------------------------------------
# kernel round-trips (real GPU)
# ----------------------------------------------------------------------

@requires_cuda
def test_roundtrip_fa_nhd_bf16():
    nb, bs, nh, hs = 4, 8, 2, 64
    tokens, invalid = 13, [1, 5]
    torch.manual_seed(42)
    paged = torch.randn(2, nb, bs, nh, hs, device="cuda", dtype=torch.bfloat16)
    slot_mapping = make_slot_mapping(tokens, nb * bs, invalid)
    staging = torch.full((tokens, 2, nh * hs), SENTINEL,
                         device="cuda", dtype=torch.bfloat16)

    staging, paged2 = _roundtrip(
        staging, paged, slot_mapping, EngineKVFormat.FA_NHD, tokens, invalid)

    # gather math: staging[t] == cat(paged[k][blk, off].flatten())
    for t_i in range(tokens):
        slot = int(slot_mapping[t_i])
        if slot < 0:
            continue
        blk, off = slot // bs, slot % bs
        expected = torch.cat(
            [paged[0, blk, off].flatten(), paged[1, blk, off].flatten()]
        ).view(2, nh * hs)
        assert_bytes_equal(staging[t_i], expected, f"token {t_i} gather")

    check_roundtrip(
        paged, paged2, slot_mapping,
        lambda t, s: t[:, s // bs, s % bs], nb * bs)


@requires_cuda
def test_roundtrip_fi_nhd_bf16():
    nb, bs, nh, hs = 4, 8, 2, 64
    tokens, invalid = 11, [3]
    torch.manual_seed(1)
    paged = torch.randn(nb, 2, bs, nh, hs, device="cuda", dtype=torch.bfloat16)
    slot_mapping = make_slot_mapping(tokens, nb * bs, invalid)
    staging = torch.full((tokens, 2, nh * hs), SENTINEL,
                         device="cuda", dtype=torch.bfloat16)

    staging, paged2 = _roundtrip(
        staging, paged, slot_mapping, EngineKVFormat.FI_NHD, tokens, invalid)

    for t_i in range(tokens):
        slot = int(slot_mapping[t_i])
        if slot < 0:
            continue
        blk, off = slot // bs, slot % bs
        expected = torch.cat(
            [paged[blk, 0, off].flatten(), paged[blk, 1, off].flatten()]
        ).view(2, nh * hs)
        assert_bytes_equal(staging[t_i], expected, f"token {t_i} gather")

    check_roundtrip(
        paged, paged2, slot_mapping,
        lambda t, s: t[s // bs, :, s % bs], nb * bs)


@requires_cuda
def test_roundtrip_fa_hnd_bf16():
    """Physically-contiguous HND tensor [2, nb, nh, bs, hs] + format FA_HND."""
    nb, bs, nh, hs = 4, 8, 2, 64
    tokens, invalid = 10, [2, 7]
    torch.manual_seed(2)
    paged = torch.randn(2, nb, nh, bs, hs, device="cuda", dtype=torch.bfloat16)
    slot_mapping = make_slot_mapping(tokens, nb * bs, invalid)
    staging = torch.full((tokens, 2, nh * hs), SENTINEL,
                         device="cuda", dtype=torch.bfloat16)

    staging, paged2 = _roundtrip(
        staging, paged, slot_mapping, EngineKVFormat.FA_HND, tokens, invalid)

    # gather math: staging[t] == cat(paged[k][blk, :, off, :].flatten())
    for t_i in range(tokens):
        slot = int(slot_mapping[t_i])
        if slot < 0:
            continue
        blk, off = slot // bs, slot % bs
        expected = torch.cat(
            [paged[0, blk, :, off, :].flatten(), paged[1, blk, :, off, :].flatten()]
        ).view(2, nh * hs)
        assert_bytes_equal(staging[t_i], expected, f"token {t_i} gather")

    check_roundtrip(
        paged, paged2, slot_mapping,
        lambda t, s: t[:, s // bs, :, s % bs, :], nb * bs)


@requires_cuda
def test_roundtrip_mla_bf16():
    """MLA: paged [nb, bs, hs], staging [tokens, hs] (kv_lora_rank+rope=576)."""
    nb, bs, hs = 4, 8, 576
    tokens, invalid = 12, [0, 4, 11]
    torch.manual_seed(3)
    paged = torch.randn(nb, bs, hs, device="cuda", dtype=torch.bfloat16)
    slot_mapping = make_slot_mapping(tokens, nb * bs, invalid)
    staging = torch.full((tokens, hs), SENTINEL,
                         device="cuda", dtype=torch.bfloat16)

    staging, paged2 = _roundtrip(
        staging, paged, slot_mapping, EngineKVFormat.MLA, tokens, invalid)

    for t_i in range(tokens):
        slot = int(slot_mapping[t_i])
        if slot < 0:
            continue
        blk, off = slot // bs, slot % bs
        assert_bytes_equal(staging[t_i], paged[blk, off], f"token {t_i} gather")

    check_roundtrip(
        paged, paged2, slot_mapping,
        lambda t, s: t[s // bs, s % bs], nb * bs)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_roundtrip_fa_nhd_dtypes(dtype):
    """8-byte kernel granularity across element sizes 2 and 4."""
    nb, bs, nh, hs = 2, 8, 4, 32
    tokens, invalid = 6, [1]
    torch.manual_seed(4)
    paged = torch.randn(2, nb, bs, nh, hs, device="cuda", dtype=dtype)
    slot_mapping = make_slot_mapping(tokens, nb * bs, invalid)
    staging = torch.full((tokens, 2, nh * hs), SENTINEL,
                         device="cuda", dtype=dtype)
    staging, paged2 = _roundtrip(
        staging, paged, slot_mapping, EngineKVFormat.FA_NHD, tokens, invalid)
    check_roundtrip(
        paged, paged2, slot_mapping,
        lambda t, s: t[:, s // bs, s % bs], nb * bs)


@requires_cuda
def test_roundtrip_kv_major_staging():
    """[2, tokens, H] staging layout (token_major=False path)."""
    nb, bs, nh, hs = 2, 8, 2, 64
    tokens, invalid = 9, [5]
    torch.manual_seed(5)
    paged = torch.randn(2, nb, bs, nh, hs, device="cuda", dtype=torch.bfloat16)
    slot_mapping = make_slot_mapping(tokens, nb * bs, invalid)
    staging = torch.full((2, tokens, nh * hs), SENTINEL,
                         device="cuda", dtype=torch.bfloat16)

    single_layer_transfer(
        staging, paged, slot_mapping, EngineKVFormat.FA_NHD, "to_staging")
    torch.cuda.synchronize()
    for t_i in range(tokens):
        slot = int(slot_mapping[t_i])
        if slot < 0:
            assert torch.equal(
                staging[:, t_i], torch.full_like(staging[:, t_i], SENTINEL))
            continue
        blk, off = slot // bs, slot % bs
        assert_bytes_equal(staging[0, t_i], paged[0, blk, off].flatten())
        assert_bytes_equal(staging[1, t_i], paged[1, blk, off].flatten())

    paged2 = torch.full_like(paged, SENTINEL)
    single_layer_transfer(
        staging, paged2, slot_mapping, EngineKVFormat.FA_NHD, "to_paged")
    torch.cuda.synchronize()
    check_roundtrip(
        paged, paged2, slot_mapping,
        lambda t, s: t[:, s // bs, s % bs], nb * bs)


@requires_cuda
def test_hnd_permuted_view_end_to_end():
    """discover on vLLM's permuted view, transfer on the physical tensor."""
    nb, bs, nh, hs = 2, 8, 2, 64
    tokens = 8
    torch.manual_seed(6)
    phys = torch.randn(2, nb, nh, bs, hs, device="cuda", dtype=torch.bfloat16)
    view = phys.permute(0, 1, 3, 2, 4)  # what vLLM hands out (logical NHD)
    fmt = discover_engine_format(view, use_mla=False)
    assert fmt == EngineKVFormat.FA_HND

    slot_mapping = make_slot_mapping(tokens, nb * bs, [])
    staging = torch.full((tokens, 2, nh * hs), SENTINEL,
                         device="cuda", dtype=torch.bfloat16)
    single_layer_transfer(staging, phys, slot_mapping, fmt, "to_staging")
    paged2 = torch.full_like(phys, SENTINEL)
    single_layer_transfer(staging, paged2, slot_mapping, fmt, "to_paged")
    torch.cuda.synchronize()
    check_roundtrip(
        phys, paged2, slot_mapping,
        lambda t, s: t[:, s // bs, :, s % bs, :], nb * bs)


# ----------------------------------------------------------------------
# python-side validation
# ----------------------------------------------------------------------

def test_validation_errors():
    paged = torch.zeros(2, 4, 8, 2, 64, dtype=torch.bfloat16)
    staging = torch.zeros(9, 2, 128, dtype=torch.bfloat16)
    slots = torch.zeros(9, dtype=torch.int64)
    with pytest.raises(ValueError):  # bad direction
        single_layer_transfer(staging, paged, slots, 1, "sideways")
    with pytest.raises(ValueError):  # unsupported format
        single_layer_transfer(staging, paged, slots, 0, "to_staging")
    with pytest.raises(ValueError):  # wrong slot dtype
        single_layer_transfer(staging, paged, slots.to(torch.int32), 1,
                              "to_staging")
    with pytest.raises(ValueError):  # dtype mismatch
        single_layer_transfer(staging.to(torch.float32), paged, slots, 1,
                              "to_staging")
    with pytest.raises(ValueError):  # head dim not 8B-aligned (fp16 x 30B)
        paged_odd = torch.zeros(2, 4, 8, 2, 30, dtype=torch.float16)
        staging_odd = torch.zeros(9, 2, 60, dtype=torch.float16)
        single_layer_transfer(staging_odd, paged_odd, slots, 1, "to_staging")
    with pytest.raises(ValueError):  # staging rank unrecognizable
        single_layer_transfer(torch.zeros(9, 128, dtype=torch.bfloat16),
                              paged, slots, 1, "to_staging")

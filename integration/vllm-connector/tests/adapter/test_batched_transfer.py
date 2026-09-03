# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the batched paged-transfer dispatch."""

import torch

from adapter import worker as worker_mod


def _hooks(slots=4):
    block_size = 2
    chunk_tokens = 2
    nh, hs = 1, 4
    paged = torch.zeros(2, slots, block_size, nh, hs, dtype=torch.float32)
    staging = torch.empty(
        slots * chunk_tokens * 2 * nh * hs * paged.element_size(),
        dtype=torch.uint8,
    )
    hooks = worker_mod.PagedTransferHooks(
        lambda _idx: paged,
        staging,
        chunk_tokens * 2 * nh * hs * paged.element_size(),
        chunk_tokens,
        block_size,
        worker_mod.EngineKVFormat.FA_NHD,
    )
    return hooks


def _interleaved_hooks(slots=4):
    block_size = 2
    chunk_tokens = 2
    channels = 4
    paged = torch.zeros(
        slots, block_size, 2, channels, dtype=torch.float32
    )
    staging = torch.empty(
        slots * chunk_tokens * 2 * channels * paged.element_size(),
        dtype=torch.uint8,
    )
    return worker_mod.PagedTransferHooks(
        lambda _idx: paged,
        staging,
        chunk_tokens * 2 * channels * paged.element_size(),
        chunk_tokens,
        block_size,
        None,
    )


def test_contiguous_slots_use_one_batched_dispatch(monkeypatch):
    hooks = _hooks()
    calls = []

    def batched(*args):
        calls.append(("batched", tuple(args[2].shape)))

    def single(*args):
        calls.append(("single", tuple(args[2].shape)))

    monkeypatch.setattr(worker_mod, "batched_layer_transfer", batched)
    monkeypatch.setattr(worker_mod, "single_layer_transfer", single)

    hooks.gather([], 0, [[0], [1]], [0, 1])

    assert calls == [("batched", (4,))]


def test_ring_wrap_falls_back_to_per_chunk_dispatch(monkeypatch):
    hooks = _hooks()
    calls = []
    monkeypatch.setattr(
        worker_mod,
        "batched_layer_transfer",
        lambda *args: calls.append("batched"),
    )
    monkeypatch.setattr(
        worker_mod,
        "single_layer_transfer",
        lambda *args: calls.append("single"),
    )

    hooks.gather([], 0, [[0], [1]], [1, 0])

    assert calls == ["single", "single"]


def test_contiguous_interleaved_slots_use_one_dispatch(monkeypatch):
    hooks = _interleaved_hooks()
    calls = []

    def interleaved(staging, _paged, mapping, direction):
        calls.append((tuple(staging.shape), tuple(mapping.shape), direction))

    monkeypatch.setattr(hooks, "_transfer_interleaved", interleaved)
    hooks.gather([], 0, [[0], [1]], [0, 1])

    assert calls == [((4, 2, 4), (4,), "to_staging")]

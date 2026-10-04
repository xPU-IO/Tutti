"""External loads must cover every advertised token without rewriting APC."""

from types import SimpleNamespace

import pytest
import torch

from tutti.integration.vllm import connector as connector_module
from tutti.integration.vllm.connector import TuttiConnectorV1
from tutti.integration.vllm.worker import PagedTransferHooks, _load_chunk_span
from tutti.common.utils import flatten_block_ids
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole


@pytest.fixture
def scheduler(monkeypatch):
    index = SimpleNamespace(lookup_prefix=lambda tokens: 1024, close=lambda: None)
    monkeypatch.setattr(connector_module, "_scheduler_index_for",
                        lambda *args: index)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
            "chunk_tokens": 256, "chunk_kv_bytes": 8192,
            "num_layers": 1, "max_chunks_per_wave": 4,
        }),
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    result = TuttiConnectorV1(config, KVConnectorRole.SCHEDULER, object())
    yield result
    result.shutdown()


def request():
    # vLLM has not written the APC hit count back to the Request at either
    # get_num_new_matched_tokens or update_state_after_alloc.
    return SimpleNamespace(request_id="r", prompt_token_ids=list(range(1056)),
                           output_token_ids=[], num_computed_tokens=0)


@pytest.mark.parametrize("local", [16, 64, 128, 240, 256, 272, 512])
def test_external_end_is_chunk_aligned(scheduler, local):
    req = request()
    matched, asynchronous = scheduler.get_num_new_matched_tokens(req, local)
    assert asynchronous is False
    assert matched == 1024 - local
    assert (local + matched) % 256 == 0


@pytest.mark.parametrize("local", [16, 256, 512])
def test_allocation_uses_lookup_prefix_not_stale_request(scheduler, local):
    req = request()
    matched, _ = scheduler.get_num_new_matched_tokens(req, local)
    assert req.num_computed_tokens == 0
    scheduler.update_state_after_alloc(req, object(), matched)
    assert scheduler._load_starts[req.request_id] == local


@pytest.mark.parametrize("start,count,expected", [
    (16, 496, (0, 2)), (272, 496, (1, 2)), (256, 512, (1, 2)),
])
def test_load_includes_partial_first_chunk(start, count, expected):
    assert _load_chunk_span(start, count, 256) == expected


def test_capped_load_preserves_aligned_end(scheduler):
    scheduler._max_tokens_per_load = 512
    matched, _ = scheduler.get_num_new_matched_tokens(request(), 16)
    assert matched == 496


def test_latest_lookup_replaces_previous_start(scheduler):
    req = request()
    scheduler.get_num_new_matched_tokens(req, 16)
    matched, _ = scheduler.get_num_new_matched_tokens(req, 272)
    scheduler.update_state_after_alloc(req, object(), matched)
    assert scheduler._load_starts[req.request_id] == 272


def test_cancelled_request_discards_lookup_boundary(scheduler):
    req = request()
    scheduler.get_num_new_matched_tokens(req, 16)
    assert scheduler._lookup_starts == {"r": 16}
    scheduler.request_finished(req, [])
    assert scheduler._lookup_starts == {}


def test_zero_allocation_drops_unused_lookup_boundary(scheduler):
    req = request()
    scheduler.get_num_new_matched_tokens(req, 16)
    scheduler.update_state_after_alloc(req, object(), 0)
    assert scheduler._lookup_starts == {}
    assert scheduler._load_starts == {}
    assert scheduler._pending_loads == {}


def test_unaligned_end_is_not_silently_dropped():
    with pytest.raises(ValueError, match="end must be chunk-aligned"):
        _load_chunk_span(16, 512, 256)


def test_staged_scatter_leaves_skipped_local_blocks_untouched():
    hooks = PagedTransferHooks.__new__(PagedTransferHooks)
    hooks._block_size = 2
    paged = torch.full((3, 2, 1), -99, dtype=torch.int64)
    staging = torch.arange(4).reshape(4, 1)
    hooks._transfer_interleaved(
        staging, paged, torch.tensor([-2, -1, 4, 5]), "to_paged"
    )
    assert paged[:2].tolist() == [[[-99], [-99]], [[-99], [-99]]]
    assert paged[2].tolist() == [[2], [3]]
    with pytest.raises(ValueError, match="read-skip"):
        hooks._transfer([[-1, 2]], [0], 0, "to_staging")


def test_failure_block_ids_exclude_read_skip_sentinels():
    assert flatten_block_ids([[-1, 3], [7, 0]]) == (3, 7, 0)
    assert flatten_block_ids([-1, 3, 7]) == (3, 7)
    assert flatten_block_ids(-1) == ()
    assert flatten_block_ids(0) == (0,)

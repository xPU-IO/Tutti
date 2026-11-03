from types import SimpleNamespace as NS

import pytest

from tutti.integration.vllm.connector import (
    TuttiConnectorV1,
    _RequestTracker,
)


def connector(*, chunk_tokens=256, block_tokens=128):
    instance = object.__new__(TuttiConnectorV1)
    instance._chunk_tokens = chunk_tokens
    instance._block_size = block_tokens
    return instance


def test_vllm_hash_chain_projects_to_tutti_chunk_boundaries():
    hashes = [bytes([i]) * 32 for i in range(6)]
    request = NS(block_hashes=hashes)
    keys = connector()._vllm_chunk_keys(request, token_count=6 * 128)
    assert keys == [hashes[1][:16], hashes[3][:16], hashes[5][:16]]


def test_vllm_projection_uses_only_complete_hashes_and_chunks():
    hashes = [bytes([i]) * 32 for i in range(3)]
    request = NS(block_hashes=hashes)
    keys = connector()._vllm_chunk_keys(request, token_count=5 * 128)
    assert keys == [hashes[1][:16]]


def test_vllm_projection_rejects_incompatible_chunk_geometry():
    request = NS(block_hashes=[b"h" * 32] * 4)
    assert connector(chunk_tokens=250)._vllm_chunk_keys(request, 512) is None
    assert connector()._vllm_chunk_keys(NS(), 512) is None


def test_canonical_tracker_never_falls_back_to_token_hashing():
    instance = connector()
    instance._index = NS(hash_keys=lambda _tokens: pytest.fail("rehash tokens"))
    tracker = _RequestTracker("r", [1] * 512, [0, 1], [b"k" * 16], True)
    with pytest.raises(RuntimeError, match="do not cover"):
        instance._tracker_keys(tracker)

"""Scheduler metadata-only role contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

from adapter.connector import TuttiConnectorV1
from engine.metadata import SchedulerMetadataIndex
from stores.tutti_nvme.layout import Layout
from stores.tutti_nvme.commit import (
    RankCommitRecord,
    commit_path,
    write_rank_commit,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole


CHUNK_TOKENS = 256
NUM_LAYERS = 80
SEGMENT_BYTES = 4096


def _config(root, *, num_chunks=64):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_kv_bytes": NUM_LAYERS * SEGMENT_BYTES,
            "max_chunks_per_wave": 512,
            "num_layers": NUM_LAYERS,
            "store": {
                "type": "tutti_nvme",
                "options": {
                    "root": str(root),
                    "num_chunks": num_chunks,
                    "io_stream": "auto",
                    "preset": {
                        "type": "local",
                        "device_id": 0,
                        "gpu_id": 0,
                    },
                },
            },
        }),
        cache_config=SimpleNamespace(block_size=64, cache_dtype="auto"),
        model_config=SimpleNamespace(model="metadata-only-test"),
        parallel_config=SimpleNamespace(
            rank=0,
            tensor_parallel_size=4,
            decode_context_parallel_size=1,
        ),
    )


def _root_template(tmp_path):
    return tmp_path / "pool-{LOCAL_RANK}"


def _rank_root(tmp_path, rank):
    return tmp_path / f"pool-{rank}"


def _publish_rank(tmp_path, namespace, keys, rank, generation,
                  *, num_layers=NUM_LAYERS, pool_generation=None):
    root = _rank_root(tmp_path, rank)
    layout = Layout(root, SEGMENT_BYTES)
    layout.ensure_dirs()
    if pool_generation is None:
        pool_generation = rank + 1
    layout.commit_layers(
        key + layer.to_bytes(2, "little")
        for key in keys
        for layer in range(num_layers)
    )
    allocated = {}
    for slot, key in enumerate(keys):
        allocated[key.hex()] = {
            "slot": slot,
            "generation": pool_generation,
        }
    layout.pool_manifest_path().write_text(json.dumps({
        "layout_version": 1,
        "namespace": namespace.hex(),
        "rank_geometry": {
            "layout": "file_per_chunk",
            "num_layers": NUM_LAYERS,
            "segment_bytes": SEGMENT_BYTES,
            "slot_bytes": NUM_LAYERS * SEGMENT_BYTES,
            "physical_slot_bytes": NUM_LAYERS * SEGMENT_BYTES,
        },
        "slot_bytes": NUM_LAYERS * SEGMENT_BYTES,
        "max_slots": 64,
        "allocated": allocated,
    }), encoding="utf-8")
    for key in keys:
        write_rank_commit(root, RankCommitRecord(
            version=1,
            namespace=namespace.hex(),
            chunk_key=key.hex(),
            generation=generation,
            num_layers=NUM_LAYERS,
            slot_bytes=NUM_LAYERS * SEGMENT_BYTES,
            rank_id=rank,
            marker_generation=layout.marker_generation(),
            pool_generation=pool_generation,
        ))
    return layout


def test_scheduler_never_calls_worker_or_runtime_factory(tmp_path, monkeypatch):
    import adapter.connector as connector_mod
    import stores.tutti_nvme.store as data_store_mod

    worker_calls = []
    runtime_calls = []

    def reject_worker(*args, **kwargs):
        worker_calls.append((args, kwargs))
        raise AssertionError("scheduler called worker engine factory")

    def reject_runtime(*args, **kwargs):
        runtime_calls.append((args, kwargs))
        raise AssertionError("scheduler called StorageRuntime factory")

    monkeypatch.setattr(connector_mod, "_worker_engine_for", reject_worker)
    monkeypatch.setattr(data_store_mod, "_build_runtime", reject_runtime)
    monkeypatch.setattr(data_store_mod, "_build_runtime_from_env", reject_runtime)

    connector = TuttiConnectorV1(
        _config(_root_template(tmp_path)), KVConnectorRole.SCHEDULER, object()
    )
    try:
        assert isinstance(connector._engine, SchedulerMetadataIndex)
        assert worker_calls == []
        assert runtime_calls == []
        assert not hasattr(connector._engine._store, "register_buffer")
        assert not hasattr(connector._engine._store, "get_batch")
        assert not hasattr(connector._engine._store, "put_batch")
        assert not hasattr(connector._engine._store, "_runtime")
    finally:
        connector.shutdown()


def test_all_rank_commit_gates_6400_and_marker_loss(tmp_path):
    connector = TuttiConnectorV1(
        _config(_root_template(tmp_path)), KVConnectorRole.SCHEDULER, object()
    )
    tokens = list(range(25 * CHUNK_TOKENS))
    keys, _ = connector._engine.hash_keys(tokens)
    namespace = connector._engine._index.namespace

    for rank in range(3):
        _publish_rank(tmp_path, namespace, keys, rank, "generation-a")
    connector._engine.sync_from_store()
    assert connector._engine.lookup_prefix(tokens) == 0

    _publish_rank(tmp_path, namespace, keys, 3, "generation-a")
    connector._engine.sync_from_store()
    assert connector._engine.lookup_prefix(tokens) == 6400

    rank2 = Layout(_rank_root(tmp_path, 2), SEGMENT_BYTES)
    rank2.marker_file(keys[0] + b"\x00\x00").unlink()
    connector._engine.sync_from_store()
    assert connector._engine.lookup_prefix(tokens) == 0
    connector.shutdown()


def test_generation_mismatch_fails_closed(tmp_path):
    connector = TuttiConnectorV1(
        _config(_root_template(tmp_path)), KVConnectorRole.SCHEDULER, object()
    )
    tokens = list(range(CHUNK_TOKENS))
    keys, _ = connector._engine.hash_keys(tokens)
    namespace = connector._engine._index.namespace
    for rank in range(3):
        _publish_rank(tmp_path, namespace, keys, rank, "generation-a")
    _publish_rank(tmp_path, namespace, keys, 3, "generation-b")
    connector._engine.sync_from_store()
    assert connector._engine.lookup_prefix(tokens) == 0
    connector.shutdown()

    restarted = TuttiConnectorV1(
        _config(_root_template(tmp_path)), KVConnectorRole.SCHEDULER, object()
    )
    assert all(
        not commit_path(_rank_root(tmp_path, rank), keys[0]).exists()
        for rank in range(4)
    )
    # Restart cleanup removes only visibility records, not layer markers.
    assert Layout(_rank_root(tmp_path, 0), SEGMENT_BYTES).marker_file(
        keys[0] + b"\x00\x00"
    ).exists()
    restarted.shutdown()


def test_restart_cleans_partial_commit_only(tmp_path):
    cfg = _config(_root_template(tmp_path))
    connector = TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())
    tokens = list(range(CHUNK_TOKENS))
    keys, _ = connector._engine.hash_keys(tokens)
    namespace = connector._engine._index.namespace
    layout = _publish_rank(tmp_path, namespace, keys, 0, "partial")
    payload = layout.chunk_file(keys[0])
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"valid-rank-data")
    marker = layout.marker_file(keys[0] + b"\x00\x00")
    connector.shutdown()

    restarted = TuttiConnectorV1(
        _config(_root_template(tmp_path)), KVConnectorRole.SCHEDULER, object()
    )
    assert not commit_path(_rank_root(tmp_path, 0), keys[0]).exists()
    assert payload.read_bytes() == b"valid-rank-data"
    assert marker.exists()
    restarted.shutdown()


def test_scheduler_lru_does_not_remove_worker_owned_pool_object(tmp_path):
    connector = TuttiConnectorV1(
        _config(_root_template(tmp_path), num_chunks=1),
        KVConnectorRole.SCHEDULER, object(),
    )
    tokens = list(range(2 * CHUNK_TOKENS))
    keys, _ = connector._engine.hash_keys(tokens)
    old_key, new_key = keys
    namespace = connector._engine._index.namespace
    for rank in range(4):
        _publish_rank(tmp_path, namespace, [old_key], rank, "generation-a")
    connector._engine.sync_from_store()
    layout = Layout(_rank_root(tmp_path, 0), SEGMENT_BYTES)
    chunk_file = layout.chunk_file(old_key)
    chunk_file.parent.mkdir(parents=True, exist_ok=True)
    chunk_file.write_bytes(b"x" * (NUM_LAYERS * SEGMENT_BYTES))
    markers = [
        old_key + layer.to_bytes(2, "little")
        for layer in range(NUM_LAYERS)
    ]
    plan = connector._engine.plan_store([new_key])
    assert plan is not None and plan.evicted_keys == [old_key]
    assert chunk_file.exists()
    assert all(layout.marker_file(key).exists() for key in markers)
    connector.shutdown()

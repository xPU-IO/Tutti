"""Scheduler metadata-only role contracts."""

from __future__ import annotations

from types import SimpleNamespace

from adapter.connector import TuttiConnectorV1
from engine.metadata import SchedulerMetadataIndex
from stores.tutti_nvme.layout import Layout
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole


CHUNK_TOKENS = 256
NUM_LAYERS = 80
SEGMENT_BYTES = 4096


def _config(root):
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
                    "num_chunks": 64,
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
        _config(tmp_path / "pool"), KVConnectorRole.SCHEDULER, object()
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


def test_complete_worker_markers_become_scheduler_hit(tmp_path):
    root = tmp_path / "pool"
    connector = TuttiConnectorV1(
        _config(root), KVConnectorRole.SCHEDULER, object()
    )
    tokens = list(range(25 * CHUNK_TOKENS))
    keys, _ = connector._engine.hash_keys(tokens)
    layout = Layout(root, SEGMENT_BYTES)
    layout.ensure_dirs()

    # Incomplete chunks must remain misses even though most layer markers exist.
    layout.commit_layers(
        key + layer.to_bytes(2, "little")
        for key in keys
        for layer in range(NUM_LAYERS - 1)
    )
    connector._engine.sync_from_store()
    assert connector._engine.lookup_prefix(tokens) == 0

    # The final marker of every chunk makes the same 6,400-token prefix visible.
    layout.commit_layers(
        key + (NUM_LAYERS - 1).to_bytes(2, "little")
        for key in keys
    )
    connector._engine.sync_from_store()
    assert connector._engine.lookup_prefix(tokens) == 6400
    connector.shutdown()

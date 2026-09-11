"""Scheduler metadata-only role contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

from adapter.connector import TuttiConnectorV1
from engine.metadata import SchedulerMetadataIndex
from stores.tutti_nvme.layout import Layout
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
    # 提交凭证已删除：冷启动可见性只由池 manifest（上面的 allocated 段）
    # 与层标记决定，运行时驻留由内存权威索引门禁。
    del keys, generation, pool_generation
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



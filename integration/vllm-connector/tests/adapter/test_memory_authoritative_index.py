"""内存权威索引契约：worker 落盘增量经 IPC 发布，查询热路径不扫盘。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from adapter.connector import TuttiConnectorV1
from adapter.worker_meta import TuttiWorkerMetadata
from engine.metadata import SchedulerMetadataIndex
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole


CHUNK_TOKENS = 256
NUM_LAYERS = 4
SEGMENT_BYTES = 4096
TP_SIZE = 4


def _config(root, *, tp_size=TP_SIZE):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_kv_bytes": NUM_LAYERS * SEGMENT_BYTES,
            "max_chunks_per_wave": 64,
            "num_layers": NUM_LAYERS,
            "store": {
                "type": "tutti_nvme",
                "options": {
                    "root": str(root),
                    "num_chunks": 64,
                    "io_stream": "auto",
                    "preset": {"type": "local", "device_id": 0, "gpu_id": 0},
                },
            },
        }),
        cache_config=SimpleNamespace(block_size=64, cache_dtype="auto"),
        model_config=SimpleNamespace(model="memory-authoritative-test"),
        parallel_config=SimpleNamespace(
            rank=0,
            tensor_parallel_size=tp_size,
            decode_context_parallel_size=1,
        ),
    )


def _request(req_id, tokens, computed=0):
    return SimpleNamespace(
        request_id=req_id,
        prompt_token_ids=list(tokens),
        output_token_ids=[],
        num_computed_tokens=computed,
    )


@pytest.fixture
def scheduler(tmp_path):
    connector = TuttiConnectorV1(
        _config(tmp_path / "pool-{LOCAL_RANK}"),
        KVConnectorRole.SCHEDULER,
        object(),
    )
    yield connector
    connector.shutdown()


def _output(meta):
    return SimpleNamespace(kv_connector_worker_meta=meta)


def test_lookup_never_scans_the_store(scheduler, monkeypatch):
    """查询热路径不得触发任何持久层枚举（92ms 空窗的根因）。"""
    index = scheduler._engine
    assert isinstance(index, SchedulerMetadataIndex)

    scans = []
    original = index._store.scan

    def counting_scan(*args, **kwargs):
        scans.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(index._store, "scan", counting_scan)

    tokens = list(range(4 * CHUNK_TOKENS))
    for _ in range(20):
        scheduler.get_num_new_matched_tokens(_request("r1", tokens), 0)
    assert scans == []


def test_all_rank_commits_publish_residency(scheduler):
    """全 TP rank 报告成功才发布驻留（与旧扫盘门禁同语义）。"""
    index = scheduler._engine
    tokens = list(range(4 * CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)

    plan = index.plan_store(keys)
    assert plan is not None and plan.new_keys == keys
    assert index.lookup_prefix(tokens) == 0

    for rank in range(TP_SIZE - 1):
        scheduler.update_connector_output(
            _output(TuttiWorkerMetadata(committed={k: 1 for k in keys}))
        )
        assert index.lookup_prefix(tokens) == 0, f"rank {rank} 就发布了驻留"

    scheduler.update_connector_output(
        _output(TuttiWorkerMetadata(committed={k: 1 for k in keys}))
    )
    assert index.lookup_prefix(tokens) == 4 * CHUNK_TOKENS


def test_single_step_aggregate_counts_all_ranks(scheduler):
    """KVOutputAggregator 在步内聚合时一次性达标即发布。"""
    index = scheduler._engine
    tokens = list(range(2 * CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None

    scheduler.update_connector_output(
        _output(TuttiWorkerMetadata(committed={k: TP_SIZE for k in keys}))
    )
    assert index.lookup_prefix(tokens) == 2 * CHUNK_TOKENS


def test_any_rank_failure_fails_closed(scheduler):
    """任一 rank 失败即回收预留、不驻留，且容量可再次受理。"""
    index = scheduler._engine
    tokens = list(range(2 * CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None

    scheduler.update_connector_output(
        _output(TuttiWorkerMetadata(committed={k: TP_SIZE - 1 for k in keys}))
    )
    scheduler.update_connector_output(
        _output(TuttiWorkerMetadata(failed=set(keys)))
    )
    assert index.lookup_prefix(tokens) == 0
    # 预留已释放：同批可再次受理（否则容量永久泄漏）
    assert index.plan_store(keys) is not None


def test_worker_meta_aggregate_merges_ranks():
    """跨 rank 聚合：计数相加、失败取并。"""
    a = TuttiWorkerMetadata(committed={b"k1": 1, b"k2": 1}, failed=set())
    b = TuttiWorkerMetadata(committed={b"k1": 1}, failed={b"k3"})
    merged = a.aggregate(b)
    assert merged.committed == {b"k1": 2, b"k2": 1}
    assert merged.failed == {b"k3"}


def test_unknown_worker_meta_is_ignored(scheduler):
    """非本 connector 的 worker meta 不得影响索引（防串扰）。"""
    index = scheduler._engine
    tokens = list(range(CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None
    scheduler.update_connector_output(_output(object()))
    scheduler.update_connector_output(_output(None))
    assert index.lookup_prefix(tokens) == 0

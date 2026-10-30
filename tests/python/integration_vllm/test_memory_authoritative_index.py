"""内存权威索引契约：worker 落盘增量经 IPC 发布，查询热路径不扫盘。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tutti.integration.vllm.connector import TuttiConnectorV1
from tutti.integration.vllm.worker_meta import TuttiWorkerMetadata
from tutti.engine.metadata import SchedulerMetadataIndex
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
                    "capacity": "1 MiB",
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
    index = scheduler._index
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


def test_lookup_reuses_request_prefix_hashes(scheduler, monkeypatch):
    """Repeated scheduler callbacks do not re-hash a 128K-style prefix."""
    tokens = list(range(4 * CHUNK_TOKENS))
    req = _request("hash-cache", tokens)
    index = scheduler._index
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None
    index.confirm_store(keys)

    calls = []
    original = index.hash_keys

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(index, "hash_keys", counted)
    for _ in range(8):
        scheduler.get_num_new_matched_tokens(req, 0)
    assert len(calls) == len(keys)

    # Decode appends to the same output list: only the newly completed chunk
    # needs a hash, and an incomplete tail needs none.
    req.output_token_ids.extend(range(4 * CHUNK_TOKENS, 5 * CHUNK_TOKENS))
    scheduler.get_num_new_matched_tokens(req, 0)
    assert len(calls) == len(keys) + 1


def test_all_rank_commits_publish_residency(scheduler):
    """全 TP rank 报告成功才发布驻留（与旧扫盘门禁同语义）。"""
    index = scheduler._index
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
    index = scheduler._index
    tokens = list(range(2 * CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None

    scheduler.update_connector_output(
        _output(TuttiWorkerMetadata(committed={k: TP_SIZE for k in keys}))
    )
    assert index.lookup_prefix(tokens) == 2 * CHUNK_TOKENS


def test_any_rank_failure_fails_closed(scheduler):
    """任一 rank 失败即回收预留、不驻留，且容量可再次受理。"""
    index = scheduler._index
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
    a = TuttiWorkerMetadata(
        committed={b"k1": 1, b"k2": 1}, failed=set(),
        profile_samples=[(1, 2, 80, 160, 10.0, 20.0, 3.0, 100, -1.0, -1.0)],
        prefetch_stats=(1, 1, 1, 0), prefetch_wait_ms=[21.0],
    )
    b = TuttiWorkerMetadata(
        committed={b"k1": 1}, failed={b"k3"},
        profile_samples=[(0, 1, 80, 80, 5.0, 8.0, 0.0, 50, 0.25, 4.0)],
        prefetch_stats=(0, 0, 0, 1), prefetch_wait_ms=[15.0],
    )
    merged = a.aggregate(b)
    assert merged.committed == {b"k1": 2, b"k2": 1}
    assert merged.failed == {b"k3"}
    assert len(merged.profile_samples) == 2
    assert merged.prefetch_stats == (1, 1, 1, 1)
    assert merged.prefetch_wait_ms == [21.0, 15.0]


def test_io_profile_reports_aggregate_bandwidth_and_overlap(caplog, scheduler):
    sample = (0, 10, 80, 800, 100.0, 200.0, 0.0,
              1_000_000_000, 0.5, 10.0)
    scheduler._profile_samples.extend([sample] * TP_SIZE)
    scheduler._profile_window_bytes = TP_SIZE * 1_000_000_000
    scheduler._profile_window_ops = TP_SIZE * 800
    scheduler._health_last_ns = 9_000_000_000
    scheduler._log_io_profile(10_000_000_000)
    line = next(record.message for record in caplog.records
                if "[tutti] io-profile" in record.message)
    assert "effective_gbps=4.000" in line
    assert "overlap_plan_rate=1.000" in line
    assert "overlap_layer_ratio=0.500" in line
    assert "exposed_wait_ms_p50=10.0" in line


def test_unknown_worker_meta_is_ignored(scheduler):
    """非本 connector 的 worker meta 不得影响索引（防串扰）。"""
    index = scheduler._index
    tokens = list(range(CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None
    scheduler.update_connector_output(_output(object()))
    scheduler.update_connector_output(_output(None))
    assert index.lookup_prefix(tokens) == 0


def test_worker_meta_mapping_is_coerced(scheduler):
    """序列化格式变更（dataclass → mapping）走兼容路径，索引仍能推进。"""
    index = scheduler._index
    tokens = list(range(CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None
    assert index.lookup_prefix(tokens) == 0

    # 同一份增量以 mapping 形态到达（模拟上游换成 msgspec 之类的编解码）
    payload = {"committed": {key: TP_SIZE for key in keys}, "failed": set()}
    scheduler.update_connector_output(_output(payload))
    assert scheduler._meta_type_warned is True
    assert index.lookup_prefix(tokens) == len(tokens)


def test_worker_meta_mapping_reports_failed_keys(scheduler):
    """mapping 兼容路径同样遵守 fail-closed：失败 key 不发布驻留。"""
    index = scheduler._index
    tokens = list(range(CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None
    scheduler.update_connector_output(_output({
        "committed": {key: TP_SIZE for key in keys},
        "failed": {keys[0]},
    }))
    assert index.lookup_prefix(tokens) == 0


def test_forgotten_keys_heal_ghost_hits(scheduler):
    """worker 回传 forgotten → 调度侧移除假驻留（幽灵命中自愈）。

    两侧是独立 LRU：worker 驱逐并删除数据后，调度侧仍可能报命中，
    于是每次都"假命中 → pin 失败 → 重算"。自愈后查询不再报命中。
    """
    index = scheduler._index
    tokens = list(range(CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None
    scheduler.update_connector_output(_output(TuttiWorkerMetadata(
        committed={key: TP_SIZE for key in keys}, failed=set(),
    )))
    assert index.lookup_prefix(tokens) == len(tokens)   # 已驻留

    # worker 侧索引判定缺失并回传
    scheduler.update_connector_output(_output(TuttiWorkerMetadata(
        committed={}, failed=set(), forgotten=set(keys),
    )))
    assert index.lookup_prefix(tokens) == 0              # 假命中已清除


def test_forgotten_does_not_touch_write_reservations(scheduler):
    """forgotten 只对齐"驻留视图"，不得动两阶段写入的容量预留。"""
    index = scheduler._index
    tokens = list(range(CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None
    reserved = set(index._planned_store_keys)
    assert reserved == set(keys)

    scheduler.update_connector_output(_output(TuttiWorkerMetadata(
        committed={}, failed=set(), forgotten=set(keys),
    )))
    # 预留仍在（否则在途写入的结算会落空）
    assert index._planned_store_keys == reserved


def test_eviction_drift_is_observed_not_acted_on(scheduler, caplog):
    """漂移观测：只统计与告警，不改变任何决策（切换前先量化）。"""
    index = scheduler._index
    tokens = list(range(CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None
    scheduler.update_connector_output(_output(TuttiWorkerMetadata(
        committed={key: TP_SIZE for key in keys}, failed=set(),
    )))
    assert index.lookup_prefix(tokens) == len(tokens)

    with caplog.at_level("WARNING", logger="tutti.engine.metadata"):
        scheduler.update_connector_output(_output(TuttiWorkerMetadata(
            committed={}, failed=set(), evicted=set(keys),
        )))
    assert index._eviction_drift_total == len(keys)
    assert index._eviction_drift_steps == 1
    assert any(r.message.startswith("INDEX_DRIFT_DETECTED")
               for r in caplog.records)
    # 观测不改变决策：驻留视图保持不变（仍报命中），行为与观测前一致
    assert index.lookup_prefix(tokens) == len(tokens)


def test_eviction_drift_ignores_keys_not_resident(scheduler):
    """worker 驱逐了本索引本就不驻留的 key，不算漂移。"""
    index = scheduler._index
    scheduler.update_connector_output(_output(TuttiWorkerMetadata(
        committed={}, failed=set(), evicted={b"never-seen-000001"},
    )))
    assert index._eviction_drift_total == 0
    assert index._eviction_drift_steps == 0


def test_independent_lru_orders_select_different_victims():
    """漂移机理：worker 与 scheduler 是两套独立 LRU。

    命中刷新（lookup_prefix → move_to_end）只发生在调度侧——worker
    从不查询自己的索引。于是同一个容量压力下，两侧选出的牺牲者不同：
    worker 删掉 K、调度侧删掉 J。之后调度侧仍报 K 命中 ⇒ 幽灵命中。
    """
    from tutti.index.chunk_index import ChunkIndex

    capacity = 4
    worker = ChunkIndex(capacity, CHUNK_TOKENS)
    scheduler = ChunkIndex(capacity, CHUNK_TOKENS)
    tokens = list(range(capacity * CHUNK_TOKENS))
    keys, _ = scheduler.hash_keys(tokens)
    assert len(keys) == capacity

    for index in (worker, scheduler):
        assert index.plan_store(keys) is not None
        index.confirm_store(keys, ok=True)

    # 调度侧发生一次命中：keys[0] 被刷新为最近使用（worker 侧不动）
    assert scheduler.lookup_prefix(tokens[:CHUNK_TOKENS]) == CHUNK_TOKENS

    # 容量压力：两侧各受理一个新 chunk，各自驱逐自己最旧的
    fresh = b"\xff" * 16
    worker_plan = worker.plan_store([fresh])
    scheduler_plan = scheduler.plan_store([fresh])
    assert worker_plan is not None and scheduler_plan is not None
    assert worker_plan.evicted_keys == [keys[0]]      # worker：最旧的
    assert scheduler_plan.evicted_keys == [keys[1]]   # 调度侧：keys[0] 已刷新

    # 漂移的两个方向：worker 删了 keys[0] 但调度侧仍驻留（幽灵命中），
    # 调度侧删了 keys[1] 但 worker 仍持有（盘上容量泄漏）。
    assert not worker.is_resident(keys[0])
    assert scheduler.is_resident(keys[0])
    assert worker.is_resident(keys[1])
    assert not scheduler.is_resident(keys[1])


def test_stale_pending_is_reclaimed_after_age_limit(scheduler):
    """某 rank 从未回报 ⇒ 预留永久占容量；超龄后必须回收。"""
    index = scheduler._index
    keys, _ = index.hash_keys(list(range(CHUNK_TOKENS)))
    assert index.plan_store(keys) is not None
    assert len(index._index._pending) == len(keys)

    # 年龄未到：不回收（覆盖 TP rank 跨步结算的正常窗口）
    for _ in range(2):
        assert index.begin_step(max_pending_age=3) == 0
    assert len(index._index._pending) == len(keys)

    # 超过 3 步：回收
    assert index.begin_step(max_pending_age=3) == len(keys)
    assert len(index._index._pending) == 0
    assert index._pending_reclaimed_total == len(keys)


def test_reclaimed_reservation_frees_capacity(scheduler):
    """回收后容量可用（否则容量会因泄漏单调下降直至写入被拒）。"""
    index = scheduler._index
    capacity = index.capacity_chunks
    # 灌满容量
    tokens = list(range(capacity * CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None
    assert index.capacity_chunks == capacity

    # 全部在途时再受理一批：容量不足且无可驱逐 → 不受理
    extra, _ = index.hash_keys(list(range(1000, 1000 + CHUNK_TOKENS)))
    assert index.plan_store(extra) is None

    for _ in range(3):
        index.begin_step(max_pending_age=3)
    # 回收后容量释放，新批可受理
    assert index.plan_store(extra) is not None


def test_pending_age_counts_from_plan_not_from_confirm(scheduler):
    """步龄从受理时刻起算；刚受理的项不应在下一步就被回收。"""
    index = scheduler._index
    keys, _ = index.hash_keys(list(range(CHUNK_TOKENS)))
    assert index.plan_store(keys) is not None
    # 下一步（age=1）不回收
    assert index.begin_step(max_pending_age=3) == 0
    # 但若上限设为 1，则应回收（确认 age 语义是"受理后经过的步数"）
    assert index.begin_step(max_pending_age=1) == len(keys)


def test_index_stats_snapshot(scheduler):
    """容量快照字段齐备且与实际状态一致（可观测性/压测断言共用）。"""
    index = scheduler._index
    tokens = list(range(CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None

    snapshot = index.stats()
    for field in ("capacity", "resident", "pending", "pinned", "epoch",
                  "rank_commit_counts", "planned_store_keys",
                  "eviction_drift_total", "pending_reclaimed_total"):
        assert field in snapshot, field
    assert snapshot["pending"] == len(keys)
    assert snapshot["resident"] == 0
    assert snapshot["planned_store_keys"] == len(keys)


def test_step_summary_logs_capacity_at_debug(scheduler, caplog):
    """每步汇总：debug 级输出容量/命中计数（原先完全静默）。"""
    from types import SimpleNamespace

    index = scheduler._index
    tokens = list(range(CHUNK_TOKENS))
    keys, _ = index.hash_keys(tokens)
    assert index.plan_store(keys) is not None

    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], resumed_req_ids=set()
        ),
        finished_req_ids=[],
        num_scheduled_tokens={},
    )
    with caplog.at_level("DEBUG", logger="vllm.tutti.connector"):
        scheduler.build_connector_meta(output)
    records = [r for r in caplog.records if r.message.startswith("[tutti] step:")]
    assert records, "每步应有一行汇总"
    message = records[0].message
    assert "pending=" in message and "resident=" in message
    assert records[0].levelname == "DEBUG"


def test_request_finished_clears_every_per_request_map(scheduler):
    """请求终结必须清空全部按请求记账（含 _live_requests/_load_starts）。"""
    request = _request("r-done", list(range(CHUNK_TOKENS)))
    # _load_starts 只在确有外部加载时登记，故此处给正数。
    scheduler.update_state_after_alloc(request, [0], num_external_tokens=CHUNK_TOKENS)
    scheduler._trackers[request.request_id] = object()
    scheduler._pending_loads[request.request_id] = 1
    assert request.request_id in scheduler._live_requests
    assert request.request_id in scheduler._load_starts

    keep = "r-alive"
    scheduler._live_requests[keep] = object()
    scheduler._load_starts[keep] = 0

    finished, _ = scheduler.request_finished(request, [0])
    assert finished is False
    for mapping in (scheduler._trackers, scheduler._pending_loads,
                    scheduler._live_requests, scheduler._load_starts):
        assert request.request_id not in mapping
    # 其它请求的记账不受影响
    assert keep in scheduler._live_requests

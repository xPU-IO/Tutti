from __future__ import annotations

from dataclasses import dataclass

import pytest

from adapter.worker import (
    WorkerImpl,
    _FailureConsensus,
    _LoadRequestSpan,
    _LogicalFailure,
    _TpLoadFailureCoordinator,
)


@dataclass
class _Coordinator:
    consensus: _FailureConsensus
    drain_ok: bool = True

    def __post_init__(self):
        self.calls = []

    def coordinate(self, generation, request_count, logical_failures):
        self.calls.append(("coordinate", generation, request_count,
                           tuple(logical_failures)))
        return self.consensus

    def drain_barrier(self, generation, local_ok):
        self.calls.append(("drain", generation, local_ok))
        return self.drain_ok and local_ok


class _Engine:
    max_in_flight_operations = 0

    def __init__(self, order=None, abort_error=None):
        self.order = order if order is not None else []
        self.abort_error = abort_error
        self.abort_count = 0
        self.wait_idle_count = 0
        self.confirm_calls = []
        self.rank_abort_calls = []

    def abort(self, timeout=None):
        self.order.append(("abort", timeout))
        self.abort_count += 1
        if self.abort_error is not None:
            raise self.abort_error

    def wait_idle(self):
        self.wait_idle_count += 1

    def abort_rank_commit(self, keys, generations=None):
        self.rank_abort_calls.append((tuple(keys), tuple(generations or ())))

    def confirm_store(self, keys, ok=True):
        self.confirm_calls.append((tuple(keys), ok))


def _worker(consensus, *, local_failed=False, local_failures=(), engine=None,
            drain_ok=True):
    coordinator = _Coordinator(consensus, drain_ok=drain_ok)
    engine = engine or _Engine()
    worker = WorkerImpl(engine, failure_coordinator=coordinator)
    worker._external_load_step = True
    worker._load_generation = consensus.generation
    worker._load_failed = local_failed
    worker._load_block_tables = [[10, 11], [12, 13]]
    worker._load_request_spans = [
        _LoadRequestSpan(0, "r0", 0, 0, 2)
    ]
    worker._load_logical_failures = {
        item.request_ordinal: item for item in local_failures
    }
    worker._metadata = type("Meta", (), {"requests": [object()]})()
    return worker, engine, coordinator


def test_single_rank_failure_propagates_same_range_to_every_rank():
    logical = (_LogicalFailure(0, 0, 2),)
    consensus = _FailureConsensus(True, 7, True, logical, 0.25)
    workers = [
        _worker(consensus, local_failed=rank == 1,
                local_failures=logical if rank == 1 else ())[0]
        for rank in range(4)
    ]
    for worker in workers:
        worker.wait_for_save()
    assert [worker.get_block_ids_with_load_errors() for worker in workers] == [
        {10, 11, 12, 13}, {10, 11, 12, 13},
        {10, 11, 12, 13}, {10, 11, 12, 13}
    ]


def test_nonfailed_rank_collects_before_it_drains_or_enters_next_step():
    order = []
    consensus = _FailureConsensus(
        True, 8, True, (_LogicalFailure(0, 0, 1),), 0.1
    )
    engine = _Engine(order=order)

    class OrderedCoordinator(_Coordinator):
        def coordinate(self, *args):
            order.append("collective")
            return super().coordinate(*args)

        def drain_barrier(self, *args):
            order.append("drain_barrier")
            return super().drain_barrier(*args)

    coordinator = OrderedCoordinator(consensus)
    worker = WorkerImpl(engine, failure_coordinator=coordinator)
    worker._external_load_step = True
    worker._load_generation = 8
    worker._load_block_tables = [[20, 21]]
    worker._load_request_spans = [_LoadRequestSpan(0, "r0", 0, 0, 1)]
    worker._metadata = type("Meta", (), {"requests": [object()]})()
    worker.wait_for_save()
    assert order == ["collective", ("abort", 30.0), "drain_barrier"]


def test_load_failure_forbids_rank_commit_and_resident_publication():
    consensus = _FailureConsensus(
        True, 9, True, (_LogicalFailure(0, 0, 1),), 0.1
    )
    worker, engine, _ = _worker(consensus)
    worker._save_keys = [b"chunk"]
    worker._save_generations = ["g9"]
    worker.wait_for_save()
    assert engine.rank_abort_calls == [((b"chunk",), ("g9",))]
    assert engine.confirm_calls == [((b"chunk",), False)]


def test_multiple_rank_failures_merge_conservatively():
    consensus = _TpLoadFailureCoordinator.merge_rows(12, 2, [
        [12, -1, -1, 0, -1, -1, 0],
        [12, 0, 1, 2, -1, -1, 0],
        [12, -1, -1, 0, 1, 7, 1],
        [12, 0, 3, 1, 1, 5, 1],
    ])
    assert consensus.failed
    assert consensus.generation_consistent
    assert consensus.logical_failures == (
        _LogicalFailure(0, 1, 3),
        _LogicalFailure(1, 5, 3),
    )


def test_generation_mismatch_fails_closed():
    consensus = _TpLoadFailureCoordinator.merge_rows(4, 1, [
        [4, -1, -1, 0], [4, -1, -1, 0],
        [5, -1, -1, 0], [4, -1, -1, 0],
    ])
    assert consensus.failed
    assert not consensus.generation_consistent


def test_normal_external_load_uses_one_collective_and_no_abort():
    consensus = _FailureConsensus(False, 13, True, (), 0.08)
    worker, engine, coordinator = _worker(consensus)
    worker.wait_for_save()
    assert [call[0] for call in coordinator.calls] == ["coordinate"]
    assert engine.abort_count == 0
    assert engine.wait_idle_count == 1


def test_abort_drain_timeout_fails_closed_after_collective():
    consensus = _FailureConsensus(
        True, 14, True, (_LogicalFailure(0, 0, 1),), 0.1
    )
    engine = _Engine(abort_error=TimeoutError("drain timeout"))
    worker, _engine, coordinator = _worker(
        consensus, engine=engine, drain_ok=False
    )
    with pytest.raises(RuntimeError, match="drain failed"):
        worker.wait_for_save()
    assert [call[0] for call in coordinator.calls] == [
        "coordinate", "drain"
    ]
    assert coordinator.calls[-1][-1] is False


def test_successful_failure_drain_allows_next_generation_reuse():
    failed = _FailureConsensus(
        True, 15, True, (_LogicalFailure(0, 0, 1),), 0.1
    )
    worker, engine, coordinator = _worker(failed)
    worker.wait_for_save()
    assert engine.abort_count == 1

    coordinator.consensus = _FailureConsensus(False, 16, True, (), 0.1)
    worker._external_load_step = True
    worker._load_generation = 16
    worker._load_failed = False
    worker._load_error_blocks = set()
    worker._failure_collective_done = False
    worker._failure_consensus = None
    worker.wait_for_save()
    assert engine.abort_count == 1
    assert engine.wait_idle_count == 1


def test_discrete_block_ids_rebuild_only_logical_failed_chunks():
    consensus = _FailureConsensus(
        True, 20, True, (_LogicalFailure(0, 1, 2),), 0.1
    )
    worker, _engine, _coordinator = _worker(consensus)
    worker._load_block_tables = [[7], [100], [9], [400]]
    worker._load_request_spans = [
        _LoadRequestSpan(0, "r0", 0, 0, 4)
    ]
    worker.wait_for_save()
    assert worker.get_block_ids_with_load_errors() == {100, 9}


def test_two_requests_failure_does_not_mark_other_request_blocks():
    consensus = _FailureConsensus(
        True, 21, True, (_LogicalFailure(1, 4, 1),), 0.1
    )
    worker, _engine, _coordinator = _worker(consensus)
    worker._metadata = type(
        "Meta", (), {"requests": [object(), object()]}
    )()
    worker._load_block_tables = [[7], [100], [9], [400]]
    worker._load_request_spans = [
        _LoadRequestSpan(0, "r0", 0, 0, 2),
        _LoadRequestSpan(1, "r1", 3, 2, 2),
    ]
    worker.wait_for_save()
    assert worker.get_block_ids_with_load_errors() == {400}
    assert worker.get_request_ids_with_load_errors() == {"r1"}

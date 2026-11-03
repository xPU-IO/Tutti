"""TuttiKVStore 专项测试与契约套件支架（不进共享套件的私有契约）。

FakeRuntime 复刻 tutti_runtime 的 Python API 形态（caps/open_batch/
register_memory/submit/release_io/wait/shutdown），接受的请求在本地
文件系统上真搬运数据（os.pwrite/pread + ctypes），可注入按轮次的
部分拒收计划以驱动 store 的窗口重发逻辑。

专项覆盖：partial-commit 窗口重发、层段布局（18B 标准 io_key）、
容量拒绝、整 chunk 回收、持久化重开恢复、崩溃安全时序（先数据后
marker）、buffer 恰一次注册、TUTTI_NVME_PRESET 推导。
"""

from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
import weakref
from collections import namedtuple
from pathlib import Path

import pytest

from tutti.index.chunk_index import decode_io_key
from tutti.storage.tutti_nvme.object_layout import DEFAULT_SEGMENT_HEADER_BYTES
from tutti.storage.tutti_nvme.store import (
    TuttiKVStore,
    _TuttiCompletion,
    _derive_device_fields,
)
import json

SEG = 4096
# 小段文件：每文件 8 槽，避免每个用例零填充一个默认 2048 槽的大文件。
SLOTS_PER_FILE = 8
HEADER = DEFAULT_SEGMENT_HEADER_BYTES


def io_key(chunk: bytes, layer: int) -> bytes:
    """构造 18B 标准 io_key：chunk_key 16B + layer 2B 小端。"""
    assert len(chunk) == 16
    return chunk + layer.to_bytes(2, "little")


FakeSubmitResult = namedtuple(
    "FakeSubmitResult",
    ["status_ok", "status_msg", "io_handle", "initial_states", "rejected"],
)

FakeWaitResult = namedtuple(
    "FakeWaitResult",
    [
        "observation", "state", "confirmed_bytes", "timeout_seen",
        "failed_request_indices", "failure_scope", "failure_kind",
        "first_failed_entry", "raw_cq_status", "message",
    ],
)


def _direct_read_cell(monkeypatch, *, width=None, detached=False):
    import threading
    from contextlib import contextmanager
    from types import SimpleNamespace as NS
    import tutti.storage.tutti_nvme.store as module
    monkeypatch.setattr(module, "_ASYNC_COMPLETION", False)
    active, submitted, released, observations, events = {}, [], [], [], []
    ready, generation, stream = [False], [1], [None]
    def submit(requests, **kwargs):
        submitted.append((tuple(requests), kwargs["stream"]))
        if active:
            return FakeSubmitResult(False, "in-flight operation capacity exhausted", None,
                                    (False,) * len(requests), tuple(range(len(requests))))
        handle = 10 + len(submitted)
        active[handle] = tuple(requests[:1])
        return FakeSubmitResult(len(requests) == 1, "partial", handle,
                                (True,) + (False,) * (len(requests) - 1),
                                tuple(range(1, len(requests))))
    def query(handle):
        assert handle in active, "query after release"
        return "COMPLETED" if ready[0] else "IN_FLIGHT"
    def observe(handle, timeout):
        assert handle in active
        observations.append(timeout)
        return FakeWaitResult("OK", "COMPLETED", 4096, False, (), "NONE", None, None, None, "")
    def release(handle):
        assert handle in active, "double release"
        del active[handle]
        released.append(handle)
    @contextmanager
    def context(direction):
        previous, stream[0] = stream[0], direction
        try:
            yield
        finally:
            stream[0] = previous
    def fence(direction):
        assert stream[0] == direction
        event = NS(query=lambda: ready[0])
        events.append(("fence", direction, event))
        return event
    runtime = NS(submit=submit, query_io=query, wait_result=observe, release_io=release,
                 caps=lambda: {"max_batch_requests": width})
    store = NS(_runtime=runtime, _batch_width_limit=lambda: width, _accel_id=0, _execution="device",
               _inflight_lock=threading.Lock(), _inflight_by_chunk={},
               _stream_for=lambda direction: (123 if direction == "read" else 456, None),
               stream_context=context, _record_event=fence,
               _layout=NS(target_generation=lambda chunk: generation[0]),
               _targets={"file://cell": NS(ticket=7)}, _read_async_stream=456)
    store._track_completion = lambda part, keys: TuttiKVStore._track_completion(store, part, keys)
    backend = module.TuttiDirectBackend(store)
    backend._memory_ticket = 3
    backend._geometry = module.DirectPoolGeometry(65536, 32768, 4096, 4096, 4096,
                                                 8, 1, 1, 2, 8192, 0)
    key = b"k" * 16
    store._live = {io_key(key, 0)}
    entry = module.DirectTargetPlanEntry(key, 7, "file://cell", 8192, 1, 0)
    plan = module.DirectTargetPlan("read", 3, 1, (entry,))
    if detached:
        backend._detached_plans[1] = plan
    else:
        backend._target_plans["read"] = plan
    return NS(backend=backend, plan=plan, keys=(key,), tables=((1, 7),), active=active,
              submitted=submitted, released=released, observations=observations,
              ready=ready, generation=generation, events=events, stream=stream)


def _direct_write_cell(monkeypatch, *, width=None):
    from tutti.storage.tutti_nvme.store import DirectTargetPlan
    cell = _direct_read_cell(monkeypatch, width=width)
    cell.plan = DirectTargetPlan("write", 3, 1, cell.plan.entries)
    cell.backend._target_plans = {"write": cell.plan}
    cell.settled = []
    cell.backend._store._on_direct_put_settled = (
        lambda ok, keys, completion: cell.settled.append((ok, tuple(keys), completion)))
    return cell


@pytest.mark.parametrize("width", [None, 1])
def test_direct_try_write_cap_one_partial_settles_whole_layer_once(monkeypatch, width):
    cell = _direct_write_cell(monkeypatch, width=width)
    def advance():
        before = len(cell.submitted)
        result = cell.backend.try_put_paged_batch(cell.keys, 0, cell.tables, cell.plan)
        assert len(cell.submitted) - before <= 1
        return result
    assert not cell.backend.has_pending_write_io()
    assert advance() is None
    assert cell.backend.has_pending_write_io() and cell.settled == []
    assert advance() is None
    assert len(cell.active) == 1 and cell.observations == []
    cell.ready[0] = True
    cell.backend.poll_write_plan()
    assert cell.released == [11] and cell.settled == []
    assert not cell.backend.has_pending_write_io() and len(cell.submitted) == 2
    completion = advance()
    assert completion._direction == "write" and cell.settled == []
    assert cell.backend._target_plans["write"] is cell.plan
    cell.backend.poll_write_plan()
    result = completion.wait_result(0)
    assert result.ok and sum(item.confirmed_bytes for item in result.results) == 8192
    assert {item.batch_indices for item in result.results} == {(0,), (1,)}
    assert cell.settled == [(True, (io_key(cell.keys[0], 0),), completion)]
    assert advance() is completion
    assert len(cell.submitted) == 3 and len(cell.settled) == 1
    assert cell.released == [11, 13] and not cell.backend.has_pending_write_io()
    assert all(request[-1] == "write" for batch, _ in cell.submitted for request in batch)
    assert cell.events[-1][1] == "write" and completion.fence_event is cell.events[-1][2]
    cell.backend.end_target_plan("write")
    assert not cell.backend._write_batches and not cell.backend._target_plans


def test_direct_write_layer_poll_does_not_scan_or_observe_other_cells(monkeypatch):
    from types import SimpleNamespace as NS
    cell = _direct_write_cell(monkeypatch, width=1)
    assert cell.backend.try_put_paged_batch(cell.keys, 0, cell.tables) is None
    def forbidden(*args, **kwargs):
        raise AssertionError("layer poll scanned or advanced another write cell")
    class ExactCells(dict):
        items = forbidden
    foreign = dict(parts=[NS(_settled=False, poll=forbidden)], completion=NS(poll=forbidden), error=None)
    cell.backend._write_batches = ExactCells(cell.backend._write_batches)
    cell.backend._write_batches[1, 1] = foreign
    assert cell.backend.has_pending_write_io(0)
    assert cell.backend.has_pending_write_io(1)
    assert not cell.backend.has_pending_write_io(2)
    cell.backend.poll_write_plan(2)
    assert cell.observations == [] and len(cell.submitted) == 1
    cell.ready[0] = True
    cell.backend.poll_write_plan(0)
    assert cell.released == [11] and cell.observations == [0]
    assert not cell.backend.has_pending_write_io(0) and cell.backend.has_pending_write_io(1)
    assert cell.settled == [] and len(cell.submitted) == 1
    completion = cell.backend.try_put_paged_batch(cell.keys, 0, cell.tables)
    cell.backend.poll_write_plan(0)
    assert completion.poll() is True and len(cell.settled) == 1
    assert cell.observations == [0, 0] and len(cell.submitted) == 2
    cell.backend._write_batches = {(1, 0): cell.backend._write_batches[1, 0]}
    cell.backend.end_target_plan("write")


def test_write_layer_cleanup_drains_only_accepted_own_io_without_retry(monkeypatch):
    from types import SimpleNamespace as NS
    cell = _direct_write_cell(monkeypatch, width=1)
    assert cell.backend.try_put_paged_batch(cell.keys, 0, cell.tables) is None
    foreign = NS(wait_result=lambda: pytest.fail("cleanup waited another write layer"))
    cell.backend._write_batches[1, 1] = dict(parts=[foreign], completion=None, error=None)
    cell.backend.drain_write_layer(0)
    assert len(cell.submitted) == 1 and cell.released == [11]
    assert cell.settled == [] and cell.backend._target_plans["write"] is cell.plan
    cell.backend._write_batches.pop((1, 1))
    cell.backend.end_target_plan("write")


def test_write_layer_cleanup_retains_unconfirmed_dma_until_safe_retry(monkeypatch):
    cell = _direct_write_cell(monkeypatch, width=1)
    assert cell.backend.try_put_paged_batch(cell.keys, 0, cell.tables) is None
    wait = cell.backend._store._runtime.wait_result
    def failed(*args):
        raise RuntimeError("write completion unknown")
    cell.backend._store._runtime.wait_result = failed
    with pytest.raises(RuntimeError, match="write completion unknown"):
        cell.backend.drain_write_layer(0)
    assert cell.released == [] and cell.backend._target_plans["write"] is cell.plan
    assert cell.backend.has_pending_write_io(layer_idx=0)
    cell.backend._store._runtime.wait_result = wait
    cell.backend.drain_write_layer(0)
    assert cell.released == [11] and cell.settled == []
    cell.backend.end_target_plan("write")


@pytest.mark.parametrize("failure", ["keys", "tables", "generation", "invalid_token", "submit"])
def test_direct_try_write_retry_failure_drains_without_partial_commit(monkeypatch, failure):
    cell = _direct_write_cell(monkeypatch)
    assert cell.backend.try_put_paged_batch(cell.keys, 0, cell.tables) is None
    keys, tables = cell.keys, cell.tables
    if failure == "keys":
        keys = (b"z" * 16,)
    elif failure == "tables":
        tables = ((2, 7),)
    elif failure == "generation":
        cell.generation[0] = 2
    elif failure == "invalid_token":
        cell.backend._invalid_plan_tokens.add(1)
    else:
        def failed(*args, **kwargs):
            raise RuntimeError("write retry submit failed")
        cell.backend._store._runtime.submit = failed
    with pytest.raises(RuntimeError) as error:
        cell.backend.try_put_paged_batch(keys, 0, tables)
    assert len(error.value.tutti_submitted_completions) == 1
    assert cell.backend.has_pending_write_io() and cell.released == []
    cell.backend.end_target_plan("write")
    assert cell.released == [11] and cell.settled == []
    assert not cell.backend._write_batches and not cell.backend.has_pending_write_io()


@pytest.mark.parametrize("failure", ["cq", "observation"])
def test_direct_try_write_closed_window_failure_retains_unsafe_plan(monkeypatch, failure):
    cell = _direct_write_cell(monkeypatch, width=1)
    assert cell.backend.try_put_paged_batch(cell.keys, 0, cell.tables) is None
    observe = cell.backend._store._runtime.wait_result
    def result(handle, timeout):
        if failure == "observation":
            raise RuntimeError("write observation failed")
        return observe(handle, timeout)._replace(state="FAILED", failure_kind="NVME_CQ")
    cell.backend._store._runtime.wait_result = result
    cell.ready[0] = True
    with pytest.raises(RuntimeError) as error:
        cell.backend.poll_write_plan()
    assert len(error.value.tutti_submitted_completions) == 1
    assert len(cell.submitted) == 1 and cell.settled == []
    if failure == "observation":
        assert cell.backend.has_pending_write_io() and cell.released == []
        with pytest.raises(RuntimeError, match="write observation failed"):
            cell.backend.end_target_plan("write")
        assert cell.backend._target_plans["write"] is cell.plan
        assert cell.backend._write_batches
        cell.backend._store._runtime.wait_result = observe
    else:
        assert not cell.backend.has_pending_write_io() and cell.released == [11]
    cell.backend.end_target_plan("write")
    assert not cell.active and cell.settled == [] and not cell.backend._write_batches


@pytest.mark.parametrize("failure", ["cq", "observation"])
def test_direct_try_write_whole_completion_preserves_late_failure(monkeypatch, failure):
    cell = _direct_write_cell(monkeypatch)
    assert cell.backend.try_put_paged_batch(cell.keys, 0, cell.tables) is None
    cell.ready[0] = True
    completion = cell.backend.try_put_paged_batch(cell.keys, 0, cell.tables)
    assert cell.released == [11] and cell.settled == []
    observe = cell.backend._store._runtime.wait_result
    def result(handle, timeout):
        if failure == "observation":
            raise RuntimeError("late write observation failed")
        return observe(handle, timeout)._replace(state="FAILED", failed_request_indices=(0,),
                                                failure_scope="REQUEST_INDICES", failure_kind="NVME_CQ")
    cell.backend._store._runtime.wait_result = result
    if failure == "observation":
        assert completion.poll() is None and cell.settled == []
        with pytest.raises(RuntimeError, match="late write observation failed"):
            cell.backend.end_target_plan("write")
        assert cell.backend.has_pending_write_io() and cell.backend._write_batches
        cell.backend._store._runtime.wait_result = observe
    else:
        assert completion.poll() is False
        assert completion.wait_result(0).failed_batch_indices == (1,)
        assert cell.settled == [(False, (io_key(cell.keys[0], 0),), completion)]
        assert cell.backend._target_plans["write"] is cell.plan
    cell.backend.end_target_plan("write")
    assert len(cell.settled) == 1 and not cell.active and cell.released == [11, 12]


@pytest.mark.parametrize("empty", [False, True])
def test_direct_try_write_final_empty_noop_and_stale_replay(monkeypatch, empty):
    cell = _direct_write_cell(monkeypatch)
    keys, tables = ((), ()) if empty else (cell.keys, cell.tables)
    if not empty:
        cell.backend.try_put_paged_batch(keys, 0, tables)
        cell.ready[0] = True
    completion = cell.backend.try_put_paged_batch(keys, 0, tables)
    assert completion.poll() is True and len(cell.settled) == 1
    if empty:
        assert not cell.submitted and completion.wait_result(0).results == ()
    accepted = len(cell.submitted)
    cell.backend.end_target_plan("write")
    with pytest.raises(RuntimeError, match="no longer active"):
        cell.backend.try_put_paged_batch(keys, 0, tables, cell.plan)
    assert len(cell.submitted) == accepted and len(cell.settled) == 1


def test_direct_async_try_write_enqueues_partials_before_write_fence(monkeypatch):
    import tutti.storage.tutti_nvme.store as module
    cell = _direct_write_cell(monkeypatch)
    monkeypatch.setattr(module, "_ASYNC_COMPLETION", True)
    def enqueue(parts):
        assert cell.stream[0] == "write"
        cell.events.append(("wait", tuple(handle for part in parts for handle in part._handles)))
    cell.backend._store.enqueue_compute_wait = enqueue
    assert cell.backend.try_put_paged_batch(cell.keys, 0, cell.tables) is None
    cell.ready[0] = True
    completion = cell.backend.try_put_paged_batch(cell.keys, 0, cell.tables)
    assert [event[0] for event in cell.events] == ["wait", "wait", "fence"]
    assert completion.fence_event is cell.events[-1][2] and cell.events[-1][1] == "write"
    cell.backend.end_target_plan("write")


@pytest.mark.parametrize("width", [None, 1])
def test_direct_try_read_yields_once_and_releases_partial_handles_before_retry(monkeypatch, width):
    cell = _direct_read_cell(monkeypatch, width=width)
    def advance():
        before = len(cell.submitted)
        completion = cell.backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan)
        assert len(cell.submitted) - before <= 1
        return completion
    assert advance() is None
    assert len(cell.active) == 1 and len(cell.submitted) == 1
    assert cell.observations == []
    assert advance() is None
    assert len(cell.active) == 1 and len(cell.submitted) == 2
    cell.ready[0] = True
    completion = advance()
    assert completion is not None and len(cell.active) == 1
    assert cell.released == [11] and cell.observations == [0]
    callbacks = []
    completion.add_done_callback(callbacks.append)
    assert completion.poll() is True
    result = completion.wait_result(0)
    assert result.ok and sum(item.confirmed_bytes for item in result.results) == 8192
    assert {item.batch_indices for item in result.results} == {(0,), (1,)}
    assert callbacks == [result] and cell.released == [11, 13]
    assert completion.poll() is True and callbacks == [result]
    assert cell.backend._store._inflight_by_chunk == {}
    assert all(request[-1] == "read" for batch, _ in cell.submitted for request in batch)
    assert cell.events[-1][2] is completion.fence_event
    cell.backend.end_target_plan("read")
    assert cell.backend._read_batches == {} and cell.active == {}


@pytest.mark.parametrize("failure", ["keys", "tables", "generation", "invalid_token", "submit"])
@pytest.mark.parametrize("detached", [False, True])
def test_direct_try_read_retry_failure_retains_accepted_dma_until_cleanup(monkeypatch, failure, detached):
    cell = _direct_read_cell(monkeypatch, detached=detached)
    assert cell.backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan) is None
    keys, tables = cell.keys, cell.tables
    if failure == "keys":
        keys = (b"z" * 16,)
    elif failure == "tables":
        tables = ((2, 7),)
    elif failure == "generation":
        cell.generation[0] = 2
    elif failure == "invalid_token":
        cell.backend._invalid_plan_tokens.add(1)
    else:
        def failed(*a, **k):
            raise RuntimeError("retry submit failed")
        cell.backend._store._runtime.submit = failed
    with pytest.raises(RuntimeError) as error:
        cell.backend.try_get_paged_batch(keys, 0, tables, cell.plan)
    parts = error.value.tutti_submitted_completions
    assert len(parts) == 1 and parts[0]._handles == [11]
    assert len(cell.active) == 1 and cell.released == []
    if detached:
        cell.backend.end_detached_read_plan(cell.plan)
        assert cell.backend._detached_plans == {}
    else:
        cell.backend.end_target_plan("read")
        assert cell.backend._target_plans == {}
    assert cell.active == {} and cell.released == [11]
    assert cell.backend._read_batches == {}
    assert parts[0].wait_result(0).ok


def test_direct_partial_closed_window_reaps_cap_one_without_submit_or_spurious_wake(monkeypatch):
    from types import SimpleNamespace as NS
    cell = _direct_read_cell(monkeypatch, width=1)
    wakes = []
    cell.backend._store._read_service = NS(wake=lambda: wakes.append(True))
    assert cell.backend.has_pending_read_io(cell.plan) is False
    cell.backend.poll_read_plan(cell.plan)
    assert not wakes and not cell.submitted
    assert cell.backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan) is None
    assert len(wakes) == 1 and cell.backend.has_pending_read_io(cell.plan)
    cell.backend.poll_read_plan(cell.plan)
    cell.backend.poll_read_plan(cell.plan)
    assert len(wakes) == 1 and not cell.observations and len(cell.submitted) == 1
    cell.ready[0] = True
    cell.backend.poll_read_plan(cell.plan)
    assert cell.released == [11] and cell.observations == [0]
    assert len(wakes) == 2 and not cell.backend.has_pending_read_io(cell.plan)
    cell.backend.poll_read_plan(cell.plan)
    assert len(wakes) == 2 and len(cell.submitted) == 1
    completion = cell.backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan)
    assert completion is not None and len(wakes) == 3
    assert cell.backend.has_pending_read_io(cell.plan)
    cell.backend.poll_read_plan(cell.plan)
    assert len(wakes) == 4 and not cell.backend.has_pending_read_io(cell.plan)
    assert completion.poll() is True and cell.released == [11, 12]


@pytest.mark.parametrize("failure", ["cq", "observation"])
def test_direct_partial_closed_window_failure_keeps_accepted_ownership_for_cancel(monkeypatch, failure):
    cell = _direct_read_cell(monkeypatch, width=1, detached=True)
    assert cell.backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan) is None
    observe = cell.backend._store._runtime.wait_result
    late = RuntimeError("closed window observation failure")
    def result(handle, timeout):
        if failure == "observation":
            raise late
        return observe(handle, timeout)._replace(state="FAILED", failed_request_indices=(0,),
                                                failure_scope="REQUEST_INDICES", failure_kind="NVME_CQ",
                                                message="closed window CQ failure")
    cell.backend._store._runtime.wait_result = result
    cell.ready[0] = True
    with pytest.raises(RuntimeError) as error:
        cell.backend.poll_read_plan(cell.plan)
    assert len(cell.submitted) == 1
    parts = error.value.tutti_submitted_completions
    assert len(parts) == 1
    if failure == "cq":
        assert not cell.backend.has_pending_read_io(cell.plan)
        assert cell.released == [11] and not parts[0].wait_result(0).ok
    else:
        assert cell.backend.has_pending_read_io(cell.plan)
        assert not cell.released
        with pytest.raises(RuntimeError, match="closed window observation failure"):
            cell.backend.end_detached_read_plan(cell.plan)
        assert cell.plan.plan_token in cell.backend._detached_plans
        cell.backend._store._runtime.wait_result = observe
    cell.backend.end_detached_read_plan(cell.plan)
    assert cell.released == [11] and not cell.active and not cell.backend._read_batches
    assert not cell.backend.has_pending_read_io(cell.plan)


def test_direct_partial_invalidated_plan_can_reap_before_cancel_without_submitting(monkeypatch):
    cell = _direct_read_cell(monkeypatch, width=1)
    assert cell.backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan) is None
    cell.backend._invalid_plan_tokens.add(cell.plan.plan_token)
    cell.ready[0] = True
    cell.backend.poll_read_plan(cell.plan)
    assert cell.released == [11] and not cell.backend.has_pending_read_io(cell.plan)
    with pytest.raises(RuntimeError, match="plan token 1 is invalid") as error:
        cell.backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan)
    assert error.value.tutti_submitted_completions[0].wait_result(0).ok
    cell.backend.end_target_plan("read")
    assert not cell.backend._read_batches and len(cell.submitted) == 1


def test_direct_empty_try_read_needs_no_submit_and_keeps_empty_completion_contract(monkeypatch):
    cell = _direct_read_cell(monkeypatch)
    completion = cell.backend.try_get_paged_batch((), 0, (), cell.plan)
    assert cell.submitted == [] and completion.poll() is True
    assert completion.wait_result(0).results == ()


def test_direct_async_try_read_enqueues_each_partial_on_submission_stream_before_fence(monkeypatch):
    import tutti.storage.tutti_nvme.store as module
    cell = _direct_read_cell(monkeypatch, detached=True)
    monkeypatch.setattr(module, "_ASYNC_COMPLETION", True)
    def enqueue(parts):
        assert cell.stream[0] == "read_async"
        cell.events.append(("wait", tuple(handle for part in parts for handle in part._handles)))
        return True
    cell.backend._store.enqueue_compute_wait = enqueue
    assert cell.backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan) is None
    cell.ready[0] = True
    completion = cell.backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan)
    assert [event[0] for event in cell.events] == ["wait", "wait", "fence"]
    assert [batch[1] for batch in cell.submitted] == [456, 456]
    assert cell.events[-1][1] == "read_async" and completion.fence_event is cell.events[-1][2]


@pytest.mark.parametrize("failure", ["cq", "observation"])
def test_direct_try_batch_preserves_late_failure_after_prior_segment_release(monkeypatch, failure):
    cell = _direct_read_cell(monkeypatch)
    cell.backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan)
    cell.ready[0] = True
    completion = cell.backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan)
    assert cell.released == [11]
    observe = cell.backend._store._runtime.wait_result
    late = RuntimeError("late read observation failed")
    def result(handle, timeout):
        if failure == "observation":
            raise late
        return observe(handle, timeout)._replace(state="FAILED", failed_request_indices=(0,),
                                                failure_scope="REQUEST_INDICES", failure_kind="NVME_CQ",
                                                message="late read CQ failed")
    cell.backend._store._runtime.wait_result = result
    callbacks = []
    completion.add_done_callback(callbacks.append)
    if failure == "cq":
        assert completion.poll() is False
        outcome = completion.wait_result(0)
        assert outcome.failed_batch_indices == (1,) and len(outcome.results) == 2
        assert callbacks == [outcome] and cell.released == [11, 12]
        cell.backend.end_target_plan("read")
    else:
        assert completion.poll() is None and completion.observation_error is late
        assert cell.released == [11] and callbacks == []
        with pytest.raises(RuntimeError, match="late read observation failed"):
            cell.backend.end_target_plan("read")
        assert cell.backend._target_plans["read"] is cell.plan
        assert cell.backend._read_batches and cell.active
        cell.backend._store._runtime.wait_result = observe
        cell.backend.end_target_plan("read")
        assert cell.released == [11, 12] and not cell.active
        with pytest.raises(RuntimeError, match="late read observation failed"):
            completion.wait_result(0)


def test_layerwise_try_read_reuses_nonblocking_submission_with_entity_memory_ticket(monkeypatch):
    from tutti.storage.tutti_nvme.layerwise_direct import TuttiLayerwiseDirectBackend, LayerPageGeometry
    cell = _direct_read_cell(monkeypatch)
    backend = TuttiLayerwiseDirectBackend(cell.backend._store)
    backend._geometry = cell.backend._geometry
    backend._memory_ticket = 3
    backend._layer_tickets = [22]
    backend._layer_pages = (LayerPageGeometry(4096, 4096, 8),)
    backend._target_plans["read"] = cell.plan
    assert backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan) is None
    cell.ready[0] = True
    completion = backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan)
    assert completion.poll() is True
    assert all(request[2] == 22 and request[4:] == (4096, "read")
               for batch, _ in cell.submitted for request in batch)
    assert cell.released == [11, 12]


def test_ordinary_and_replica_share_submit_step_and_read_batch_completion():
    from tutti.storage.tutti_nvme.replica_direct import _ReplicaSubmit, _ReplicaBatchCompletion
    from tutti.storage.tutti_nvme.store import _SubmitStep, _ReadBatchCompletion
    assert _ReplicaSubmit is _SubmitStep and _ReplicaBatchCompletion is _ReadBatchCompletion


def test_ordinary_sync_read_cannot_duplicate_pending_nonblocking_cell(monkeypatch):
    cell = _direct_read_cell(monkeypatch, detached=True)
    backend = cell.backend
    assert backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan) is None
    accepted = len(cell.submitted)
    with pytest.raises(RuntimeError, match="already owned"):
        backend.get_paged_batch(cell.keys, 0, cell.tables, cell.plan)
    assert len(cell.submitted) == accepted
    cell.ready[0] = True
    backend.end_detached_read_plan(cell.plan)
    with pytest.raises(RuntimeError, match="no longer active"):
        backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan)
    assert len(cell.submitted) == accepted


def test_ordinary_terminal_io_failure_does_not_permanently_retain_target_plan(monkeypatch):
    cell = _direct_read_cell(monkeypatch, detached=True)
    backend = cell.backend
    assert backend.try_get_paged_batch(cell.keys, 0, cell.tables, cell.plan) is None
    cell.ready[0] = True
    observe = backend._store._runtime.wait_result
    backend._store._runtime.wait_result = lambda handle, timeout: observe(handle, timeout)._replace(
        state="FAILED", failed_request_indices=(0,), failure_scope="REQUEST_INDICES",
        failure_kind="NVME_CQ", message="read CQ failure")
    with pytest.raises(RuntimeError, match="read CQ failure"):
        backend.poll_read_plan(cell.plan)
    backend.end_detached_read_plan(cell.plan)
    assert cell.released == [11]
    assert backend._read_batches == {}
    assert backend._detached_plans == {}


def test_segment_failure_notifies_parent_before_other_accepted_dma_finishes():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.store import _ReadBatchCompletion, _SubmittedHandle, _TuttiCompletion
    ready, released = {11: True, 12: False}, []
    def result(handle, timeout):
        failed = handle == 11
        return NS(observation="OK", state="FAILED" if failed else "COMPLETED",
                  confirmed_bytes=4096, timeout_seen=False,
                  failed_request_indices=(0,) if failed else (),
                  failure_scope="REQUEST_INDICES" if failed else "NONE",
                  failure_kind="NVME_CQ" if failed else None, first_failed_entry=None,
                  raw_cq_status=None, message="first segment failed" if failed else "")
    runtime = NS(query_io=lambda handle: "COMPLETED" if ready[handle] else "IN_FLIGHT",
                 wait_result=result, release_io=released.append)
    parts = [_TuttiCompletion(runtime, [_SubmittedHandle(handle, (index,))], lambda ok: None,
                              auto_watch=False, retain_observation_errors=True)
             for handle, index in ((11, 3), (12, 7))]
    batch = _ReadBatchCompletion(runtime, parts)
    terminal, done = [], []
    batch.add_terminal_callback(terminal.append)
    batch.add_done_callback(done.append)
    assert parts[0].poll() is False
    assert len(terminal) == 1 and terminal[0].failed_batch_indices == (3,)
    assert done == [] and not batch.io_terminal
    assert released == [11] and parts[1]._handles == [12]
    late = []
    batch.add_terminal_callback(late.append)
    assert late == terminal
    ready[12] = True
    assert batch.poll() is False
    assert len(terminal) == len(done) == 1 and released == [11, 12]
    assert {result.batch_indices for result in done[0].results} == {(3,), (7,)}


def _reject_none(round_idx: int, count: int) -> set[int]:
    return set()


class FakeRuntime:
    """tutti_runtime 的文件系统级 fake（数据真搬运，提交即完成）。"""

    def __init__(self, reject_plan=None, supports_multi_stream=None,
                 bound_accel_id=None, partial_status_non_ok=False):
        self._next_ticket = 0
        self._targets: dict[int, str] = {}
        self._memories: dict[int, tuple[int, int]] = {}
        self._io_done: set[int] = set()
        self._released: set[int] = set()
        self._closed_targets: set[int] = set()
        self.register_calls: list[tuple] = []
        self.submit_rounds = 0
        self.shutdown_called = False
        self.submit_streams: list[tuple[str, int | None]] = []
        self._supports_multi_stream = supports_multi_stream
        self._bound_accel_id = bound_accel_id
        self._partial_status_non_ok = partial_status_non_ok
        self._reject_plan = reject_plan or _reject_none

    def caps(self):
        caps = {
            "target": ["stub"],
            "memory": ["host"],
            "length_alignment_bytes": 1,
            "max_single_io_bytes": None,
            "max_batch_requests": None,
            "max_in_flight_operations": None,
        }
        if self._supports_multi_stream is not None:
            caps["supports_multi_stream"] = self._supports_multi_stream
            caps["max_concurrent_streams"] = 2 if self._supports_multi_stream else 1
        if self._bound_accel_id is not None:
            caps["bound_accel_id"] = self._bound_accel_id
        return caps

    def open_batch(self, uris):
        tickets = []
        for uri in uris:
            if not uri.startswith("file://"):
                raise ValueError(f"FakeRuntime 仅支持 file:// uri：{uri!r}")
            self._next_ticket += 1
            self._targets[self._next_ticket] = uri[len("file://"):]
            tickets.append(self._next_ticket)
        return tickets

    def close_target(self, ticket):
        self._closed_targets.add(ticket)

    def close_batch(self, tickets):
        self._closed_targets.update(tickets)

    def register_memory(self, addr, size, kind, accel_id=-1, io_granularity=0):
        self.register_calls.append((addr, size, kind, accel_id, io_granularity))
        self._next_ticket += 1
        self._memories[self._next_ticket] = (addr, size)
        return self._next_ticket

    def submit(self, requests, accel_id=-1, stream=None, execution="device"):
        self.submit_streams.append((requests[0][-1] if requests else "", stream))
        rejected = set(self._reject_plan(self.submit_rounds, len(requests)))
        self.submit_rounds += 1
        for idx, req in enumerate(requests):
            if idx not in rejected:
                self._execute(req)
        self._next_ticket += 1
        handle = self._next_ticket
        self._io_done.add(handle)
        return FakeSubmitResult(
            status_ok=not (rejected and self._partial_status_non_ok),
            status_msg="partial submit" if rejected else "",
            io_handle=handle,
            initial_states=[idx not in rejected for idx in range(len(requests))],
            rejected=sorted(rejected),
        )

    def release_io(self, io_handle):
        self._released.add(io_handle)

    def wait(self, io_handle, timeout_ms=0):
        if io_handle in self._io_done:
            return ("OK", "COMPLETED")
        return ("OK", "")

    def shutdown(self, timeout_ms):
        self.shutdown_called = True

    # ---------- 内部 ----------

    def _execute(self, req):
        target, target_offset, memory, memory_offset, length, direction = req
        path = self._targets[target]
        addr, _size = self._memories[memory]
        fd = os.open(path, os.O_RDWR)
        try:
            if direction == "write":
                os.pwrite(fd, ctypes.string_at(addr + memory_offset, length), target_offset)
            elif direction == "read":
                data = os.pread(fd, length, target_offset)
                ctypes.memmove(addr + memory_offset, data, len(data))
            else:
                raise ValueError(f"未知 direction：{direction!r}")
        finally:
            os.close(fd)


def register_factory(register) -> None:
    """向契约套件注册 tutti_nvme 实现（fake runtime 支架）。"""

    def make_store() -> TuttiKVStore:
        root = tempfile.mkdtemp(prefix="tutti-nvme-contract-")
        store = TuttiKVStore(
            root=root, num_chunks=8, segment_bytes=4096, runtime=FakeRuntime(),
            segment_file_slots=SLOTS_PER_FILE,
        )
        # 对象几何（段数 × 段大小 + 对象头）必须定案才能落盘。通用契约的
        # key 是不带层号的短 key（每 key 自成一个对象），故层宽取 1。
        store.set_layer_span(1)
        weakref.finalize(store, shutil.rmtree, root, True)
        return store

    register(
        "tutti_nvme",
        make_store,
        segment_bytes=4096,
        granularity=4096,
        capacity_chunks=8,
    )


def make_store(tmp_path, num_chunks=8, runtime=None, layers=4) -> TuttiKVStore:
    store = TuttiKVStore(
        root=tmp_path / "pool",
        num_chunks=num_chunks,
        segment_bytes=SEG,
        runtime=runtime or FakeRuntime(),
        segment_file_slots=SLOTS_PER_FILE,
    )
    # 对象几何必须定案（对象头 + 每层一个段）才能分配槽位。
    store.set_layer_span(layers)
    return store


def test_sibling_store_recovers_after_checkpoint(tmp_path):
    """同命名空间的另一个实例在对象层落检查点后恢复出已提交对象。

    生产语义：worker 是命名空间唯一写者（close 落检查点，也可显式落），
    调度侧只读视图据此重建驻留集合；热路径由 worker 的增量发布维护，不再
    扫盘，因此这里验证的是冷启动恢复这一条通道。
    """
    root = tmp_path / "pool"
    worker = TuttiKVStore(
        root=root, num_chunks=8, segment_bytes=SEG, runtime=FakeRuntime()
    )
    worker.set_layer_span(1)
    worker.open()
    src = bytearray(b"x" * SEG)
    buffer_id = worker.register_buffer(src, SEG)
    key = io_key(b"sibling".ljust(16, b"_"), 0)
    worker.put_batch([(key, buffer_id, 0)]).wait()

    # 未落检查点前，另一个实例看不到（内存视图是本进程权威）
    reader = TuttiKVStore(
        root=root, num_chunks=8, segment_bytes=SEG, runtime=FakeRuntime()
    )
    reader.set_layer_span(1)
    reader.open()
    assert reader.scan() == []

    worker._layout.checkpoint()
    recovered = TuttiKVStore(
        root=root, num_chunks=8, segment_bytes=SEG, runtime=FakeRuntime()
    )
    recovered.set_layer_span(1)
    recovered.open()
    assert recovered.scan() == [key]
    worker.close()
    reader.close()
    recovered.close()


# ---------- partial-commit 窗口重发 ----------


def test_partial_commit_retried(tmp_path):
    """第一轮拒一半 → 重发后整批完成，数据完整落盘。"""
    runtime = FakeRuntime(
        reject_plan=lambda rnd, n: set(range(0, n, 2)) if rnd == 0 else set()
    )
    # 每个 key 自成一个对象（层宽 1）：这里验证的是运行时部分拒绝后的窗口
    # 重发，与层数无关；层宽 >1 时对象要等各层写齐才提交。
    store = make_store(tmp_path, runtime=runtime, layers=1)
    store.open()
    src = bytearray(4 * SEG)
    src_id = store.register_buffer(src, SEG)
    keys = [io_key(bytes([i]) * 16, 0) for i in range(4)]
    for i, key in enumerate(keys):
        src[i * SEG:(i + 1) * SEG] = bytes([0x30 + i]) * SEG
    store.put_batch([(k, src_id, i * SEG) for i, k in enumerate(keys)]).wait()
    assert runtime.submit_rounds == 2  # 一轮部分拒 + 一轮重发
    dst = bytearray(4 * SEG)
    dst_id = store.register_buffer(dst, SEG)
    store.get_batch([(k, dst_id, i * SEG) for i, k in enumerate(keys)]).wait()
    for i in range(4):
        assert dst[i * SEG:(i + 1) * SEG] == bytes([0x30 + i]) * SEG


def test_partial_commit_non_ok_status_still_drains_handle(tmp_path):
    """Non-OK partial status carries issued IO that must not be discarded."""
    runtime = FakeRuntime(
        reject_plan=lambda rnd, n: {0} if rnd == 0 else set(),
        partial_status_non_ok=True,
    )
    store = make_store(tmp_path, runtime=runtime, layers=1)
    store.open()
    src = bytearray(2 * SEG)
    src[:SEG] = b"a" * SEG
    src[SEG:] = b"b" * SEG
    src_id = store.register_buffer(src, SEG)
    keys = [io_key(b"a" * 16, 0), io_key(b"b" * 16, 0)]

    completion = store.put_batch(
        [(keys[0], src_id, 0), (keys[1], src_id, SEG)]
    )
    completion.wait()

    assert runtime.submit_rounds == 2
    assert len(runtime._released) == 2
    assert store.scan() == keys


class WidthLimitedRuntime(FakeRuntime):
    """一批请求数超过 width 就整批拒绝，模拟 runtime 的单次提交条目上限。

    真机形态（2026-09-21 在线）：一个读计划覆盖整步所有 chunk，长 prompt +
    高并发时单层提交 9248 条 > max_batch_entries=8192，被整批拒绝 →
    旧逻辑"连续两轮零接受"直接 RuntimeError，打死 worker 并终止服务。
    """

    def __init__(self, width, advertise=False, **kwargs):
        super().__init__(**kwargs)
        self._width = width
        self._advertise = advertise
        self.wide_rejections = 0
        self.accepted_counts: list[int] = []

    def caps(self):
        caps = super().caps()
        if self._advertise:
            caps["max_batch_requests"] = self._width
        return caps

    def submit(self, requests, accel_id=-1, stream=None, execution="device"):
        if len(requests) > self._width:
            self.wide_rejections += 1
            self.submit_rounds += 1
            self._next_ticket += 1
            return FakeSubmitResult(
                status_ok=False,
                status_msg="RESOURCE_EXHAUSTED: too many sub-IOs (entries)",
                io_handle=None,
                initial_states=[False] * len(requests),
                rejected=list(range(len(requests))),
            )
        self.accepted_counts.append(len(requests))
        return super().submit(requests, accel_id=accel_id, stream=stream,
                              execution=execution)


def test_wide_batch_is_split_instead_of_killing_the_worker(tmp_path):
    """整批因"太宽"被拒 → 拆半重投并全部完成，而不是致命升级。"""
    runtime = WidthLimitedRuntime(width=3)
    store = make_store(tmp_path, runtime=runtime, layers=1)
    store.open()
    src = bytearray(8 * SEG)
    keys = []
    for i in range(8):
        src[i * SEG:(i + 1) * SEG] = bytes([0x40 + i]) * SEG
        keys.append(io_key(bytes([i]) * 16, 0))
    src_id = store.register_buffer(src, SEG)

    store.put_batch([(k, src_id, i * SEG) for i, k in enumerate(keys)]).wait()

    assert runtime.wide_rejections > 0, "必须真的撞上过上限（否则用例没覆盖）"
    # 8 → 4+4（仍超限）→ 2+2+2+2，每段都在上限内。
    assert runtime.accepted_counts == [2, 2, 2, 2]
    dst = bytearray(8 * SEG)
    dst_id = store.register_buffer(dst, SEG)
    store.get_batch([(k, dst_id, i * SEG) for i, k in enumerate(keys)]).wait()
    for i in range(8):
        assert dst[i * SEG:(i + 1) * SEG] == bytes([0x40 + i]) * SEG


def test_single_request_over_the_limit_still_fails(tmp_path):
    """拆到单笔仍被拒 → 说明是那一笔自身的问题，保持 fail-closed。"""
    runtime = WidthLimitedRuntime(width=0)
    store = make_store(tmp_path, runtime=runtime, layers=1)
    store.open()
    src = bytearray(SEG)
    src_id = store.register_buffer(src, SEG)
    keys = [io_key(b"z" * 16, 0)]

    with pytest.raises(RuntimeError, match="连续两轮零接受"):
        store.put_batch([(keys[0], src_id, 0)]).wait()


def test_advertised_width_limit_splits_before_submitting(tmp_path):
    """runtime 广告了上限 → 提交前按上限切段，不浪费必然失败的那次提交。"""
    runtime = WidthLimitedRuntime(width=3, advertise=True)
    store = make_store(tmp_path, runtime=runtime, layers=1)
    store.open()
    src = bytearray(8 * SEG)
    keys = []
    for i in range(8):
        src[i * SEG:(i + 1) * SEG] = bytes([0x50 + i]) * SEG
        keys.append(io_key(bytes([i]) * 16, 0))
    src_id = store.register_buffer(src, SEG)

    store.put_batch([(k, src_id, i * SEG) for i, k in enumerate(keys)]).wait()

    assert runtime.wide_rejections == 0, "预切后不应再撞上限"
    # 8 → 3+3+2，按原顺序提交。
    assert runtime.accepted_counts == [3, 3, 2]
    dst = bytearray(8 * SEG)
    dst_id = store.register_buffer(dst, SEG)
    store.get_batch([(k, dst_id, i * SEG) for i, k in enumerate(keys)]).wait()
    for i in range(8):
        assert dst[i * SEG:(i + 1) * SEG] == bytes([0x50 + i]) * SEG


def test_auto_routes_read_and_write_to_distinct_streams(tmp_path, monkeypatch):
    """auto creates distinct read/write submit and worker stream routes."""
    import types
    import torch

    class FakeStream:
        next_handle = 100

        def __init__(self, device):
            self.device = types.SimpleNamespace(index=int(str(device).split(":")[-1]))
            FakeStream.next_handle += 1
            self.cuda_stream = FakeStream.next_handle
            self.waited = []
            self.recorded = []

        def wait_event(self, event):
            self.waited.append(event)

    class FakeEvent:
        def __init__(self):
            self.recorded_on = None

        def record(self, stream):
            self.recorded_on = stream

    class FakeStreamContext:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", FakeStreamContext)
    compute_stream = FakeStream("cuda:0")
    next_compute_stream = FakeStream("cuda:0")
    current_compute_stream = [compute_stream]
    current_stream_calls = []
    monkeypatch.setattr(
        torch.cuda, "current_stream",
        lambda device=None: (
            current_stream_calls.append(device), current_compute_stream[0]
        )[1],
    )

    runtime = FakeRuntime(supports_multi_stream=True, bound_accel_id=0)
    store = TuttiKVStore(
        root=tmp_path / "pool", num_chunks=2, segment_bytes=SEG,
        runtime=runtime, io_stream="auto", preset={"type": "local", "gpu_id": 0},
    )
    store.open()
    store.set_layer_span(1)   # 对象几何必须定案（单层 key 各成一个对象）
    assert store._stream_mode == "dual"
    assert store._read_stream != store._write_stream
    assert store._read_stream_obj is not store._write_stream_obj
    assert store._read_copy_stream not in (
        store._read_stream, store._write_stream
    )
    assert store._read_copy_stream_obj not in (
        store._read_stream_obj, store._write_stream_obj
    )
    with store.stream_context("read") as selected:
        assert selected is store._read_stream_obj
    with store.stream_context("write") as selected:
        assert selected is store._write_stream_obj
    with store.stream_context("read_copy") as selected:
        assert selected is store._read_copy_stream_obj

    src = bytearray(b"r" * SEG)
    src_id = store.register_buffer(src, SEG)
    key = io_key(b"r" * 16, 0)
    store.put_batch([(key, src_id, 0)]).wait()
    dst = bytearray(SEG)
    dst_id = store.register_buffer(dst, SEG)
    store.get_batch([(key, dst_id, 0)]).wait()

    assert runtime.submit_streams == [
        ("write", store._write_stream),
        ("read", store._read_stream),
    ]
    event = FakeEvent()
    store.wait_read_event(event)
    store.wait_write_event(event)
    store.wait_compute_event(event)
    current_compute_stream[0] = next_compute_stream
    store.wait_compute_event(event)
    assert store._read_stream_obj.waited == [event]
    assert store._write_stream_obj.waited == [event]
    assert compute_stream.waited == [event]
    assert next_compute_stream.waited == [event]
    compute_event = store.record_compute_event()
    assert compute_event.recorded_on is next_compute_stream
    assert current_stream_calls == [None, None, None]
    assert store.record_read_event().recorded_on is store._read_stream_obj
    assert store.record_write_event().recorded_on is store._write_stream_obj
    assert (store.record_read_copy_event().recorded_on is
            store._read_copy_stream_obj)
    assert store.read_copy_stream_handle() == store._read_copy_stream
    store.close()
    assert store._read_stream_obj is None
    assert store._write_stream_obj is None
    assert store._read_copy_stream_obj is None


def test_auto_falls_back_when_caps_reject_multi_stream(tmp_path, monkeypatch):
    """An explicit no-multi-stream capability selects the shared fallback."""
    import types
    import torch

    class FakeStream:
        def __init__(self, device):
            self.device = types.SimpleNamespace(index=0)
            self.cuda_stream = 777

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    runtime = FakeRuntime(supports_multi_stream=False)
    store = TuttiKVStore(
        root=tmp_path / "pool", num_chunks=2, segment_bytes=SEG,
        runtime=runtime, io_stream="auto", preset={"type": "local", "gpu_id": 0},
    )
    store.open()
    assert store._stream_mode == "shared"
    assert store._read_stream == store._write_stream == 777
    assert store._read_stream_obj is store._write_stream_obj


def test_auto_falls_back_when_multi_stream_caps_are_missing(tmp_path, monkeypatch):
    """An older binding without stream caps uses one shared stream."""
    import types
    import torch

    class FakeStream:
        def __init__(self, device):
            self.device = types.SimpleNamespace(index=0)
            self.cuda_stream = 778

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    runtime = FakeRuntime()
    store = TuttiKVStore(
        root=tmp_path / "pool", num_chunks=2, segment_bytes=SEG,
        runtime=runtime, io_stream="auto", preset={"type": "local", "gpu_id": 0},
    )
    store.open()
    assert store._stream_mode == "shared"
    assert store._read_stream == store._write_stream == 778


def test_auto_rejects_runtime_and_preset_device_mismatch(tmp_path, monkeypatch):
    """Read/write streams cannot be created away from the runtime accelerator."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runtime = FakeRuntime(supports_multi_stream=True, bound_accel_id=1)
    store = TuttiKVStore(
        root=tmp_path / "pool", num_chunks=2, segment_bytes=SEG,
        runtime=runtime, io_stream="auto", preset={"type": "local", "gpu_id": 0},
    )
    with pytest.raises(RuntimeError, match="bound_accel_id.*gpu_id"):
        store.open()


def test_explicit_legacy_stream_shares_io_but_keeps_copy_separate(
        tmp_path, monkeypatch):
    """An explicit io_stream keeps the pre-existing single-stream contract."""
    import types
    import torch

    class FakeStream:
        def __init__(self, device):
            self.device = types.SimpleNamespace(index=0)
            self.cuda_stream = 5678

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)

    runtime = FakeRuntime(supports_multi_stream=True)
    store = TuttiKVStore(
        root=tmp_path / "pool", num_chunks=2, segment_bytes=SEG,
        runtime=runtime, io_stream=1234,
    )
    store.open()
    assert store._stream_mode == "shared"
    assert store._read_stream == store._write_stream == 1234
    assert store._read_copy_stream == 5678
    event = types.SimpleNamespace(synchronize=lambda: None)
    store.wait_read_event(event)
    store.wait_write_event(event)


def test_owned_runtime_shutdown_precedes_stream_release(tmp_path, monkeypatch):
    """close keeps both stream owners alive through runtime shutdown."""
    import types
    import torch
    import tutti.storage.tutti_nvme.store as store_module

    class FakeStream:
        next_handle = 900

        def __init__(self, device):
            self.device = types.SimpleNamespace(index=0)
            FakeStream.next_handle += 1
            self.cuda_stream = FakeStream.next_handle

    class OwnedRuntime(FakeRuntime):
        owner = None

        def shutdown(self, timeout_ms):
            assert self.owner._read_stream_obj is not None
            assert self.owner._write_stream_obj is not None
            assert self.owner._read_copy_stream_obj is not None
            super().shutdown(timeout_ms)

    runtime = OwnedRuntime()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(store_module, "_build_runtime", lambda _preset: runtime)
    store = TuttiKVStore(
        root=tmp_path / "pool", num_chunks=2, segment_bytes=SEG,
        runtime=None, io_stream="auto", preset={"type": "local", "gpu_id": 0},
    )
    runtime.owner = store
    store.open()
    store.close()
    assert runtime.shutdown_called is True
    assert store._runtime is None
    assert store._read_stream_obj is None
    assert store._write_stream_obj is None
    assert store._read_copy_stream_obj is None


def test_double_stall_raises(tmp_path):
    """连续两轮零接受 → RuntimeError。"""
    runtime = FakeRuntime(reject_plan=lambda rnd, n: set(range(n)))
    store = make_store(tmp_path, runtime=runtime)
    store.open()
    src = bytearray(SEG)
    src_id = store.register_buffer(src, SEG)
    with pytest.raises(RuntimeError, match="连续两轮零接受"):
        store.put_batch([(io_key(b"a" * 16, 0), src_id, 0)])
    assert runtime.submit_rounds == 2  # 恰好两轮后放弃


# ---------- 层段布局（18B 标准 io_key） ----------


def _slot_path(store, chunk_id) -> Path:
    uri = store._layout.target_uri(chunk_id)
    return Path(uri[len("file://"):] if uri.startswith("file://") else uri)


def test_layer_segments_share_one_object(tmp_path):
    """同 chunk 各层共用一个槽位；段 offset = 槽位前缀 + layer × segment。"""
    store = make_store(tmp_path, layers=6)
    store.open()
    src = bytearray(SEG)
    src_id = store.register_buffer(src, SEG)
    chunk = b"\x11" * 16
    layers = [0, 2, 5]
    for layer in layers:
        src[:] = bytes([0x40 + layer]) * SEG
        store.put_batch([(io_key(chunk, layer), src_id, 0)]).wait()

    path = _slot_path(store, chunk)
    data = path.read_bytes()
    payload = store._layout.target_offset(chunk)
    assert payload == HEADER                    # 槽位 0：前缀后即段 0
    # 段文件按几何整体定型：每文件 SLOTS_PER_FILE 个 (前缀 + 6 段) 槽位。
    assert len(data) == SLOTS_PER_FILE * (HEADER + 6 * SEG)
    for layer in layers:
        segment = data[payload + layer * SEG:payload + (layer + 1) * SEG]
        assert segment == bytes([0x40 + layer]) * SEG


def test_slot_materialisation_is_physical(tmp_path):
    """段文件整体实写零（物理块就位），非稀疏文件。"""
    store = make_store(tmp_path, layers=4)
    store.open()
    src_id = store.register_buffer(bytearray(SEG), SEG)
    chunk = b"\x22" * 16
    store.put_batch([(io_key(chunk, 3), src_id, 0)]).wait()
    path = _slot_path(store, chunk)
    file_bytes = SLOTS_PER_FILE * (HEADER + 4 * SEG)
    assert path.stat().st_size == file_bytes
    assert path.stat().st_blocks * 512 >= file_bytes


def test_incomplete_object_is_unreadable(tmp_path):
    """半截对象（层没写齐）读不得：一层没写完就当整个 chunk 没写。"""
    store = make_store(tmp_path, layers=3)
    store.open()
    src = bytearray(SEG)
    src_id = store.register_buffer(src, SEG)
    chunk = b"\x33" * 16
    for layer in (0, 2):
        store.put_batch([(io_key(chunk, layer), src_id, 0)]).wait()
    assert store.scan() == []  # 未提交 ⇒ 不在驻留集合
    with pytest.raises(ValueError):
        store.get_batch([(io_key(chunk, 1), src_id, 0)])

    store.put_batch([(io_key(chunk, 1), src_id, 0)]).wait()   # 补齐最后一层
    assert store.scan() == sorted(
        io_key(chunk, layer) for layer in range(3)
    )


# ---------- 容量 ----------


def test_capacity_exhaustion_trims_batch(tmp_path, caplog):
    """容量耗尽不失败：只写被受理的 chunk（对象层部分受理契约）。"""
    store = make_store(tmp_path, num_chunks=2, layers=1)
    store.open()
    src = bytearray(3 * SEG)
    src_id = store.register_buffer(src, SEG)
    keys = [io_key(bytes([i]) * 16, 0) for i in range(3)]
    with caplog.at_level("WARNING", logger="tutti.storage.tutti_nvme.store"):
        completion = store.put_batch(
            [(k, src_id, i * SEG) for i, k in enumerate(keys)]
        )
        completion.wait()
    resident = store.scan()
    assert len(resident) == 2                  # 容量只允许两个对象
    assert set(resident) <= set(keys)
    assert any("ADMISSION_SHORTFALL" in r.message for r in caplog.records)
    # 未受理的 chunk 不进驻留集合；盘上只有一个段文件（容量 2 < 一个文件组）
    segments = _slot_path(store, bytes(resident[0][:16])).parent
    assert sorted(p.name for p in segments.glob("*.seg")) == ["0.seg"]
    rejected = [key for key in keys if key not in set(resident)]
    for key in rejected:
        assert not store._layout.is_committed(bytes(key[:16]))


# ---------- drop 与槽位回收 ----------


def test_drop_releases_whole_object(tmp_path):
    """drop 命中对象任一 key 即回收整个对象：按对象提交 ⇒ 不存在半删的 chunk。"""
    store = make_store(tmp_path, num_chunks=2, layers=2)
    store.open()
    src_id = store.register_buffer(bytearray(2 * SEG), SEG)
    chunk = b"\x44" * 16
    keys = [io_key(chunk, layer) for layer in (0, 1)]
    store.put_batch(
        [(key, src_id, index * SEG) for index, key in enumerate(keys)]
    ).wait()
    path = _slot_path(store, chunk)
    assert path.exists() and store.scan() == sorted(keys)

    store.drop([keys[0]])  # 命中一次 → 整个对象回收（各层一起）
    assert store.scan() == []
    assert not store._layout.is_committed(chunk)
    assert not store.has(keys[1])       # 各层一起失效，不存在半删 chunk
    assert not store.has(keys[0])
    # 槽位文件由对象层决定何时回收/复用（可能保留给下一次分配），但对象
    # 已经不可读：是否还存在文件不属于本层的契约。

    # 回收后的容量可复用
    other = b"\x55" * 16
    store.put_batch([(io_key(other, 0), src_id, 0)]).wait()
    assert _slot_path(store, other).exists()


def test_abort_chunks_tolerates_recycled_chunks(tmp_path):
    """清理路径容忍已被驱逐回收的 chunk。

    回归（2026-09-20 在线驱逐压测）：调度侧驱逐决策与本步写受理交叉时，
    本步 _save_keys 可能含一个已被 store.drop() 移出预留表的 chunk；回滚
    路径对它调 target_uri 的 fail-fast 会把"本轮写失败"升级成 worker 崩溃
    （KeyError → EngineDeadError，整个 server 一起死）。
    """
    store = make_store(tmp_path, num_chunks=4, layers=2)
    store.open()
    kept = b"\x11" * 16
    recycled = b"\x22" * 16

    # 两个 chunk 同批受理：预留但未提交（= 写受理后、提交前的窗口）
    admitted, _ = store._layout.prepare_put(
        [io_key(kept, 0), io_key(recycled, 0)], store.capacity_chunks
    )
    assert len(admitted) == 2
    # 调度侧驱逐 recycled：drop 命中对象任一 key 即回收整个对象、移出预留表
    store.drop([io_key(recycled, 0)])

    # 本步写失败后的回滚：必须容忍 recycled 已不在预留表（原先抛 KeyError）
    store.abort_chunks([io_key(kept, 0), io_key(recycled, 0)])
    # kept 的预留被回滚、recycled 保持已回收状态，两者都不可读
    assert not store.has(io_key(kept, 0))
    assert not store.has(io_key(recycled, 0))


# ---------- 持久化与重开恢复 ----------


def test_reopen_recovers_scan(tmp_path):
    """close 后新实例 open 同一 root → scan 恢复在场 io_key 且数据可读。"""
    store = make_store(tmp_path, layers=2)
    store.open()
    src = bytearray(4 * SEG)
    src_id = store.register_buffer(src, SEG)
    first, second = b"\x66" * 16, b"\x77" * 16
    src[:2 * SEG] = b"\x0A" * (2 * SEG)
    src[2 * SEG:] = b"\x0B" * (2 * SEG)
    keys = [
        io_key(first, 0), io_key(first, 1),
        io_key(second, 0), io_key(second, 1),
    ]
    store.put_batch([
        (io_key(first, 0), src_id, 0), (io_key(first, 1), src_id, SEG),
        (io_key(second, 0), src_id, 2 * SEG),
        (io_key(second, 1), src_id, 3 * SEG),
    ]).wait()
    store.close()

    reopened = make_store(tmp_path, layers=2)
    reopened.open()
    assert reopened.scan() == sorted(keys)
    dst = bytearray(4 * SEG)
    dst_id = reopened.register_buffer(dst, SEG)
    reopened.get_batch([
        (io_key(first, 0), dst_id, 0), (io_key(first, 1), dst_id, SEG),
        (io_key(second, 0), dst_id, 2 * SEG),
        (io_key(second, 1), dst_id, 3 * SEG),
    ]).wait()
    assert dst[:SEG] == b"\x0A" * SEG
    assert dst[3 * SEG:] == b"\x0B" * SEG


def test_object_committed_only_after_completion(tmp_path):
    """先数据后提交：put_batch 返回时对象尚未提交（崩溃安全时序）。"""
    store = make_store(tmp_path, layers=2)
    store.open()
    src = bytearray(2 * SEG)
    src_id = store.register_buffer(src, SEG)
    chunk = b"\x88" * 16
    keys = [io_key(chunk, layer) for layer in range(2)]
    completion = store.put_batch([
        (keys[0], src_id, 0), (keys[1], src_id, SEG),
    ])
    assert store.scan() == []  # 数据已下发但对象未提交
    completion.wait()
    assert store.scan() == sorted(keys)


def test_write_leaves_object_files_only(tmp_path):
    """写入的盘上痕迹只有对象文件（对象头 + 段）；无 marker / manifest。

    Python 侧不再产生任何元数据文件：层标记、manifest、提交凭证都已下沉到
    对象层（对象头 + 检查点），冷启动恢复由对象层给出。
    """
    root = tmp_path / "pool"
    namespace = b"all-rank-commit-test"
    store = TuttiKVStore(
        root=root,
        num_chunks=2,
        segment_bytes=SEG,
        runtime=FakeRuntime(),
        rank_id=2,
        tp_size=4,
        segment_file_slots=SLOTS_PER_FILE,
    )
    store.set_key_namespace(namespace)
    store.open()
    store.set_layer_span(2)
    chunk = b"rank-commit-key!"
    src = bytearray(2 * SEG)
    src_id = store.register_buffer(src, SEG)
    store.put_batch([
        (io_key(chunk, 0), src_id, 0),
        (io_key(chunk, 1), src_id, SEG),
    ]).wait()

    # 对象有效 ⇔ 全部段都写过；归属与几何都在对象头里
    assert store._layout.is_committed(chunk)
    assert store._layout.target_offset(chunk) == HEADER
    assert store._layout.target_size(chunk) == 2 * SEG

    names = {path.name for path in root.rglob("*")}
    assert not [name for name in names if name.endswith(".ok")]
    assert not [name for name in names if "manifest" in name]
    assert not (root / "commits").exists()
    # 元数据只剩对象层自己的（检查点 / 跨 rank 位图）
    assert (root / "meta").is_dir()
    store.close()


def test_completion_wait_uses_runtime_notification_without_polling():
    """Completion waiting blocks in the runtime watcher, not 100ms polling."""
    import threading

    class BlockingRuntime:
        def __init__(self):
            self.done = threading.Event()
            self.released_event = threading.Event()
            self.timeouts = []
            self.released = []

        def wait(self, handle, timeout_ms=0):
            self.timeouts.append(timeout_ms)
            if timeout_ms == 0:
                return ("TIMEOUT", "")
            self.done.wait(timeout_ms / 1000)
            return ("OK", "COMPLETED") if self.done.is_set() else ("TIMEOUT", "")

        def release_io(self, handle):
            self.released.append(handle)
            self.released_event.set()

    runtime = BlockingRuntime()
    settled = []
    completion = _TuttiCompletion(runtime, [7], settled.append)
    assert completion.query() is False
    runtime.done.set()
    # The watcher owns terminal handle reclamation; a long layer sequence
    # must not retain every completed handle until wait_for_save().
    assert runtime.released_event.wait(1.0)
    assert settled == []
    assert runtime.released == [7]
    completion.wait(timeout=1.0)
    assert settled == [True]
    assert runtime.released == [7]
    # The first blocking observation is one long runtime wait, rather than
    # repeated 100ms probes from the model thread.
    assert 1000 in runtime.timeouts


def test_completion_wait_detail_is_retained_after_release():
    class StructuredRuntime:
        def __init__(self):
            self.released = []

        def wait_result(self, handle, timeout_ms=0):
            return FakeWaitResult(
                "OK", "FAILED", 4096, True, (), "WHOLE_OPERATION",
                "CQ_TIMEOUT", 7, 0x1234, "controller timeout",
            )

        def release_io(self, handle):
            self.released.append(handle)

    runtime = StructuredRuntime()
    completion = _TuttiCompletion(runtime, [11], lambda _ok: None)
    result = completion.wait_result(timeout=1.0)
    assert not result.ok
    assert result.timeout_seen
    assert result.failures[0].failure_kind == "CQ_TIMEOUT"
    assert result.failures[0].first_failed_entry == 7
    assert result.failures[0].raw_cq_status == 0x1234
    assert runtime.released == [11]
    retained = completion.wait_detail(timeout=0)
    assert retained == result


@pytest.mark.parametrize("auto_watch", [False, True])
@pytest.mark.parametrize("outcome", ["failed", "observation_error", "exception"])
def test_completion_failure_never_reports_batch_success(auto_watch, outcome):
    class FailingRuntime:
        def __init__(self):
            self.released = []

        def wait_result(self, handle, timeout_ms=0):
            if outcome == "exception":
                raise RuntimeError("injected wait failure")
            return FakeWaitResult(
                "OK" if outcome == "failed" else "INVALID_HANDLE",
                "FAILED" if outcome == "failed" else "",
                0, False, (), "WHOLE_OPERATION", "UNKNOWN", None, None,
                "injected completion failure",
            )

        def release_io(self, handle):
            self.released.append(handle)

    runtime = FailingRuntime()
    settled = []
    notified = []
    completion = _TuttiCompletion(runtime, [17], settled.append,
                                  auto_watch=auto_watch)
    completion.add_terminal_callback(notified.append)
    result = completion.wait_result(timeout=1.0)
    assert not result.ok
    assert settled == [False]
    assert len(notified) == 1
    assert not notified[0].ok
    assert runtime.released == [17]
    with pytest.raises(RuntimeError, match="tutti IO"):
        completion.wait(timeout=0)


def test_completion_poll_settles_invalid_observation():
    class InvalidRuntime:
        def wait_result(self, handle, timeout_ms=0):
            return FakeWaitResult(
                "INVALID_HANDLE", "", 0, False, (), "WHOLE_OPERATION",
                "UNKNOWN", None, None, "invalid handle",
            )

        def release_io(self, handle):
            pass

    completion = _TuttiCompletion(InvalidRuntime(), [19], lambda _ok: None,
                                  auto_watch=False)
    assert completion.poll() is False
    assert completion._settled
    assert not completion.wait_result(timeout=0).ok


@pytest.mark.parametrize("error", ["exception", "invalid"])
def test_async_completion_retains_unknown_observation_without_releasing_dma(error):
    class Runtime:
        def __init__(self):
            self.done = False
            self.released = []
        def wait_result(self, handle, timeout_ms=0):
            if not self.done:
                if error == "exception":
                    raise RuntimeError("observation unavailable")
                return FakeWaitResult("INVALID_HANDLE", "", 0, False, (), "WHOLE_OPERATION",
                                      "UNKNOWN", None, None, "observation unavailable")
            return FakeWaitResult("OK", "COMPLETED", 0, False, (), "NONE", None, None, None, "")
        def release_io(self, handle):
            self.released.append(handle)
    runtime = Runtime()
    completion = _TuttiCompletion(runtime, [19], lambda ok: None, auto_watch=False,
                                  retain_observation_errors=True)
    assert completion.poll() is None
    assert completion.observation_error is not None
    assert not completion.io_terminal and runtime.released == []
    with pytest.raises(RuntimeError, match="observation unavailable"):
        completion.wait()
    assert runtime.released == []
    runtime.done = True
    assert completion.poll() is True and completion.io_terminal
    assert runtime.released == [19]


def test_get_batch_auto_releases_terminal_handle(tmp_path):
    """Pre-enqueued reads cannot retain all handles until layer callbacks."""
    runtime = FakeRuntime()
    store = make_store(tmp_path, runtime=runtime, layers=1)
    store.open()
    key = io_key(b"read-auto-free".ljust(16, b"_"), 0)
    src = bytearray(b"r" * SEG)
    src_id = store.register_buffer(src, SEG)
    store.put_batch([(key, src_id, 0)]).wait()

    released_before = len(runtime._released)
    dst = bytearray(SEG)
    dst_id = store.register_buffer(dst, SEG)
    completion = store.get_batch([(key, dst_id, 0)])
    assert completion._ready.wait(1.0)
    assert len(runtime._released) == released_before + 1
    completion.wait(timeout=1.0)
    assert len(runtime._released) == released_before + 1
    assert dst == src
    store.close()


# ---------- register_buffer ----------


def test_register_buffer_called_once_per_buffer(tmp_path):
    """同一 buffer 重复注册 → runtime.register_memory 恰一次，id 各异。"""
    runtime = FakeRuntime()
    store = make_store(tmp_path, runtime=runtime)
    store.open()
    buf = bytearray(2 * SEG)
    first = store.register_buffer(buf, SEG)
    second = store.register_buffer(buf, SEG)
    assert isinstance(first, int) and isinstance(second, int)
    assert first != second
    assert len(runtime.register_calls) == 1  # 恰一次
    kind = runtime.register_calls[0][2]
    granularity = runtime.register_calls[0][4]
    assert kind == "host"
    assert granularity == SEG


def test_register_buffer_rejects_bad_inputs(tmp_path):
    """非 4096 对齐粒度 / 不可定位地址 → None。"""
    store = make_store(tmp_path)
    store.open()
    buf = bytearray(2 * SEG)
    assert store.register_buffer(buf, SEG + 1) is None
    assert store.register_buffer(buf, 0) is None
    assert store.register_buffer(b"read-only", SEG) is None
    assert store.register_buffer(12345, SEG) is None


def test_unregistered_buffer_and_bad_offset(tmp_path):
    store = make_store(tmp_path)
    store.open()
    buf = bytearray(SEG)
    buf_id = store.register_buffer(buf, SEG)
    with pytest.raises(ValueError):
        store.put_batch([(b"k", buf_id + 100, 0)])
    with pytest.raises(ValueError):
        store.put_batch([(b"k", buf_id, SEG)])  # 越界（buffer 恰一段长）
    store.put_batch([(b"k", buf_id, 0)]).wait()


def test_ctypes_buffer_supported(tmp_path):
    store = make_store(tmp_path, layers=1)
    store.open()
    src = (ctypes.c_char * SEG)()
    dst = bytearray(SEG)
    src_id = store.register_buffer(src, SEG)
    dst_id = store.register_buffer(dst, SEG)
    ctypes.memset(src, 0xCD, SEG)
    store.put_batch([(b"ct", src_id, 0)]).wait()
    store.get_batch([(b"ct", dst_id, 0)]).wait()
    assert bytes(dst) == b"\xCD" * SEG


# ---------- 构造与注册表 ----------


def test_constructor_rejects_bad_args(tmp_path):
    with pytest.raises(ValueError):
        make_store(tmp_path, num_chunks=0)
    with pytest.raises(ValueError):
        TuttiKVStore(root=tmp_path, num_chunks=1, segment_bytes=0, runtime=FakeRuntime())


def test_registered_in_store_registry():
    from tutti.storage.registry import create_store

    store = create_store(
        "tutti_nvme",
        {
            "root": "/tmp/should-not-be-opened",
            "num_chunks": 3,
            "segment_bytes": 8192,
            "runtime": FakeRuntime(),
        },
    )
    assert isinstance(store, TuttiKVStore)
    assert store.capacity_chunks == 3
    store.close()


def test_runtime_built_from_env_requires_preset(tmp_path, monkeypatch):
    """runtime=None 且 TUTTI_NVME_PRESET 缺失 → open 抛 RuntimeError。"""
    monkeypatch.delenv("TUTTI_NVME_PRESET", raising=False)
    store = TuttiKVStore(root=tmp_path, num_chunks=1, segment_bytes=SEG)
    with pytest.raises(RuntimeError, match="TUTTI_NVME_PRESET"):
        store.open()


def _load_bindings_runtime():
    """导入 bindings 构建产物（需先 build_ext --inplace）。"""
    import sys

    # csrc/python 是自带 setup.py 的独立 distribution（pybind 扩展 tutti_runtime），
    # 不由根 pyproject 打包，故这里显式定位其源码树。
    bindings = Path(__file__).resolve().parents[3] / "csrc" / "python"
    if str(bindings) not in sys.path:
        sys.path.insert(0, str(bindings))
    return pytest.importorskip("tutti_runtime")


@pytest.mark.parametrize("key", ["handle_cache_capacity", "handle_cache_l2_capacity", "prp_cache_capacity"])
@pytest.mark.parametrize("factory", ["make_local_nvme_runtime", "make_raid0_nvme_runtime"])
def test_cache_capacity_is_not_a_user_preset_option(tmp_path, key, factory):
    runtime = _load_bindings_runtime()
    device = {"pci_bdf": "ffff:00:00.0", "mount_path": str(tmp_path)}
    preset = {"device": device, key: 32} if factory == "make_local_nvme_runtime" else {"devices": [device, device], key: 32}
    with pytest.raises(Exception) as error:
        getattr(runtime, factory)(preset)
    assert "unknown preset key" in str(error.value)
    assert key in str(error.value)


def test_unused_allocator_option_is_rejected(tmp_path):
    with pytest.raises(TypeError, match="allocator_enabled"):
        TuttiKVStore(tmp_path, 4, SEG, runtime=FakeRuntime(), allocator_enabled=False)


def test_preset_optional_fields_parse(tmp_path):
    """preset 回收：硬件几何/预算字段全部可省略，解析不被 missing key 卡住。

    本机无 NVMe 设备，runtime 构造必然在设备打开处失败；断言点在于
    失败原因不是 'missing preset key'（即解析已按 C++ 默认兜底通过）。
    """
    tutti_runtime = _load_bindings_runtime()
    preset = {
        "device": {
            "pci_bdf": "0000:00:00.0",
            "mount_path": str(tmp_path),
        }
        # backing_device/namespace_id/block_size 与全部预算字段均省略
    }
    with pytest.raises(Exception) as excinfo:
        tutti_runtime.make_local_nvme_runtime(preset)
    assert "missing preset key" not in str(excinfo.value)
    assert "unknown preset key" not in str(excinfo.value)


@pytest.mark.parametrize(
    "key",
    [
        # gpu_id is deliberately absent: it is std::int32_t, so -1 is a legal
        # value rather than a wrap, and there is a separate assertion below
        # pinning that. Only the unsigned fields belong here.
        "num_queues",
        "max_batch_entries",
        "max_in_flight_operations",
    ],
)
def test_preset_rejects_negative_unsigned_fields(tmp_path, key):
    """负数必须被拒绝，而不是静默 wrap 成超大无符号值。

    `static_cast<T>(int64)` 对负数是回绕的：num_queues=-1 曾变成 4294967295，
    该值随后进入队列预算与 arena 尺寸计算。回绕在调用点完全不可见，后续报错也
    不会提到原始输入，所以拒绝必须发生在解析处——那里还知道 preset key 的名字。

    断言包含 key 名与 "out of range"：只断言"抛异常"是不够的，因为无硬件环境下
    构造本就会因设备打不开而抛，那样这条测试会永远通过。
    """
    tutti_runtime = _load_bindings_runtime()
    preset = {
        "device": {"pci_bdf": "0000:00:00.0", "mount_path": str(tmp_path)},
        key: -1,
    }
    with pytest.raises(Exception) as excinfo:
        tutti_runtime.make_local_nvme_runtime(preset)
    message = str(excinfo.value)
    assert key in message, message
    assert "out of range" in message, message


@pytest.mark.parametrize("key", ["namespace_id", "block_size"])
def test_preset_rejects_negative_device_fields(tmp_path, key):
    """设备子字典里的无符号字段同样拒绝负数（它们走同一个转换函数）。"""
    tutti_runtime = _load_bindings_runtime()
    preset = {
        "device": {
            "pci_bdf": "0000:00:00.0",
            "mount_path": str(tmp_path),
            key: -1,
        }
    }
    with pytest.raises(Exception) as excinfo:
        tutti_runtime.make_local_nvme_runtime(preset)
    message = str(excinfo.value)
    assert key in message, message
    assert "out of range" in message, message


def test_preset_rejects_out_of_range_unsigned_field(tmp_path):
    """超出目标类型上限的值也要拒绝，而不只是负数。

    2**63 落在 int64 表示范围内（Python int 无界，取到 int64 后仍为正），
    但远超 uint32 字段的容量，必须在这里挡住。
    """
    tutti_runtime = _load_bindings_runtime()
    preset = {
        "device": {"pci_bdf": "0000:00:00.0", "mount_path": str(tmp_path)},
        "num_queues": 2**40,
    }
    with pytest.raises(Exception) as excinfo:
        tutti_runtime.make_local_nvme_runtime(preset)
    message = str(excinfo.value)
    assert "num_queues" in message, message
    assert "out of range" in message, message


# 2026-09-22：原 test_preset_accepts_wide_unsigned_field 已删除——条带 preset
# 的唯一 uint64 字段 stripe_unit 随条带化一起移除后，preset 不再有 uint64 字段，
# 该回归（有符号域比较吞掉合法值）失去了覆盖对象。


def test_preset_accepts_negative_gpu_id(tmp_path):
    """gpu_id 是有符号的，-1 必须照常通过解析。

    与上一组测试成对：把校验做成"拒绝一切负数"就会误伤它。gpu_id 是下标型字段，
    负值在某些部署里表示"未绑定"，所以它不该被无符号规则覆盖。

    失败（若有）应来自设备打开，而不是解析层。
    """
    tutti_runtime = _load_bindings_runtime()
    preset = {
        "device": {"pci_bdf": "0000:00:00.0", "mount_path": str(tmp_path)},
        "gpu_id": -1,
    }
    with pytest.raises(Exception) as excinfo:
        tutti_runtime.make_local_nvme_runtime(preset)
    message = str(excinfo.value)
    assert "gpu_id" not in message, message
    assert "out of range" not in message, message


def test_preset_still_accepts_valid_bounds(tmp_path):
    """边界内的合法值不受影响——校验不能过严。

    0 与 1 都必须通过解析；失败（若有）应来自设备打开，而不是解析层。
    这条同时防止"把校验写成拒绝一切"这种过度收紧。
    """
    tutti_runtime = _load_bindings_runtime()
    for value in (0, 1):
        preset = {
            "device": {"pci_bdf": "0000:00:00.0", "mount_path": str(tmp_path)},
            "num_queues": value,
            "max_in_flight_operations": value,
        }
        with pytest.raises(Exception) as excinfo:
            tutti_runtime.make_local_nvme_runtime(preset)
        message = str(excinfo.value)
        assert "out of range" not in message, message


# ---------- preset 推导（daemon_config + device_id） ----------


def test_derive_device_fields_from_daemon(tmp_path):
    """设备事实从 daemon 配置推导；preset 显式键优先。"""
    import yaml

    daemon = {
        "nvmes": [
            {"device_id": 0, "pci_addr": "0000:08:00.0",
             "backing_mount_path": "/mnt/nvme0", "namespace_id": 1},
            {"device_id": 1, "pci_addr": "0000:4b:00.0",
             "backing_mount_path": "/mnt/nvme1", "namespace_id": 2},
        ]
    }
    daemon_path = tmp_path / "daemon.yaml"
    daemon_path.write_text(yaml.safe_dump(daemon))

    preset = {"type": "local", "daemon_config": str(daemon_path), "device_id": 1, "gpu_id": 0}
    derived = _derive_device_fields(preset, yaml)
    device = derived["device"]
    assert device["pci_bdf"] == "0000:4b:00.0"       # daemon 推导
    assert device["mount_path"] == "/mnt/nvme1"      # daemon 推导
    assert device["namespace_id"] == 2               # daemon 推导
    assert device["backing_device"] == "/dev/snvme1n2"
    assert derived["type"] == "local"               # 保留明确的配置拓扑

    override = dict(preset, device={"mount_path": "/custom/mount"})
    device = _derive_device_fields(override, yaml)["device"]
    assert device["mount_path"] == "/custom/mount"   # 显式优先
    assert device["pci_bdf"] == "0000:4b:00.0"


def test_derive_device_fields_unknown_device(tmp_path):
    import yaml

    daemon_path = tmp_path / "daemon.yaml"
    daemon_path.write_text(yaml.safe_dump({"nvmes": []}))
    with pytest.raises(RuntimeError, match="device_id"):
        _derive_device_fields(
            {"type": "local", "daemon_config": str(daemon_path), "device_id": 9}, yaml
        )


def test_derive_devices_from_several_daemon_configs(tmp_path):
    """每个盘组一个 daemon：多份配置合并成一份设备清单；device_id 不得重复。"""
    import yaml

    kv0 = tmp_path / "kv0.yaml"
    md0 = tmp_path / "md0.yaml"
    kv0.write_text(yaml.safe_dump({"nvmes": [
        {"device_id": 0, "pci_addr": "ffff:00:00.0", "backing_mount_path": "/kv"}]}))
    md0.write_text(yaml.safe_dump({"nvmes": [
        {"device_id": 4, "pci_addr": "ffff:00:04.0", "backing_mount_path": "/md"}]}))
    for daemon_config in ([str(kv0), str(md0)], f"{kv0},{md0}"):
        preset = {"type": "nvme-device-group", "daemon_config": daemon_config,
                  "devices": [{"device_id": 0}, {"device_id": 4}]}
        devices = _derive_device_fields(preset, yaml)["devices"]
        assert [d["mount_path"] for d in devices] == ["/kv", "/md"]
        assert [d["pci_bdf"] for d in devices] == ["ffff:00:00.0", "ffff:00:04.0"]

    md0.write_text(yaml.safe_dump({"nvmes": [
        {"device_id": 0, "pci_addr": "ffff:00:04.0", "backing_mount_path": "/md"}]}))
    with pytest.raises(RuntimeError, match="device_id=0"):
        _derive_device_fields(
            {"type": "nvme-device-group", "daemon_config": [str(kv0), str(md0)],
             "devices": [{"device_id": 0}]}, yaml)


def test_derive_backing_device_by_pci_address(tmp_path, monkeypatch):
    """块设备按 PCI 地址从 sysfs 查：snvme 编号不一定等于 device_id。"""
    import yaml

    from tutti.storage.tutti_nvme import preset_derive

    (tmp_path / "pci" / "ffff:00:04.0" / "snvme" / "snvme0" / "snvme0n1").mkdir(
        parents=True)
    monkeypatch.setattr(preset_derive, "_PCI_SYSFS", tmp_path / "pci")
    daemon = tmp_path / "md0.yaml"
    daemon.write_text(yaml.safe_dump({"nvmes": [
        {"device_id": 4, "pci_addr": "ffff:00:04.0", "backing_mount_path": "/md"},
        {"device_id": 5, "pci_addr": "ffff:00:05.0", "backing_mount_path": "/md"}]}))
    devices = _derive_device_fields(
        {"type": "nvme-device-group", "daemon_config": str(daemon),
         "devices": [{"device_id": 4}, {"device_id": 5}]}, yaml)["devices"]
    assert devices[0]["backing_device"] == "/dev/snvme0n1"   # sysfs 实际编号
    assert devices[1]["backing_device"] == "/dev/snvme5n1"   # 未接管：旧约定


def test_derive_striped_devices_from_daemon(tmp_path, monkeypatch):
    """striped preset：devices 的 device_id 列表按 daemon 配置逐个推导。"""
    from tutti.storage.tutti_nvme import preset_derive

    # 不读本机 sysfs：块设备走 device_id 旧约定。
    monkeypatch.setattr(preset_derive, "_PCI_SYSFS", tmp_path / "no-sysfs")
    import yaml

    daemon = {
        "nvmes": [
            {"device_id": 0, "pci_addr": "0000:08:00.0",
             "backing_mount_path": "/mnt/nvme0", "namespace_id": 1},
            {"device_id": 2, "pci_addr": "0000:57:00.0",
             "backing_mount_path": "/mnt/nvme2", "namespace_id": 1},
        ]
    }
    daemon_path = tmp_path / "daemon.yaml"
    daemon_path.write_text(yaml.safe_dump(daemon))

    preset = {
        "type": "nvme-device-group",
        "daemon_config": str(daemon_path),
        "devices": [{"device_id": 0}, {"device_id": 2}],
    }
    derived = _derive_device_fields(preset, yaml)
    devices = derived["devices"]
    assert [d["pci_bdf"] for d in devices] == [
        "0000:08:00.0", "0000:57:00.0"
    ]
    assert [d["mount_path"] for d in devices] == ["/mnt/nvme0", "/mnt/nvme2"]
    assert [d["backing_device"] for d in devices] == [
        "/dev/snvme0n1", "/dev/snvme2n1"
    ]
    assert all("device_id" not in d for d in devices)
    # 幂等：store 构造与 runtime 构造各推导一次，结果必须一致
    assert _derive_device_fields(derived, yaml)["devices"] == devices
    # 显式字段优先于 daemon 事实
    override = dict(preset, devices=[{"device_id": 0, "mount_path": "/custom"}])
    assert (_derive_device_fields(override, yaml)["devices"][0]["mount_path"]
            == "/custom")


def test_store_derives_mounts_from_preset(tmp_path):
    """未显式给 mounts 时，mounts 从 preset.devices 的 daemon 推导结果取得。"""
    import yaml

    nvme0 = tmp_path / "nvme0"
    nvme2 = tmp_path / "nvme2"
    daemon_path = tmp_path / "daemon.yaml"
    daemon_path.write_text(yaml.safe_dump({"nvmes": [
        {"device_id": 0, "pci_addr": "0000:08:00.0",
         "backing_mount_path": str(nvme0), "namespace_id": 1},
        {"device_id": 2, "pci_addr": "0000:57:00.0",
         "backing_mount_path": str(nvme2), "namespace_id": 1},
    ]}))
    preset = {
        "type": "nvme-device-group",
        "daemon_config": str(daemon_path),
        "devices": [{"device_id": 0}, {"device_id": 2}],
    }
    store = TuttiKVStore(
        tmp_path / "meta-root", 8, SEG, preset=preset,
    )
    assert store._layout.mounts == (
        str(nvme0.resolve()), str(nvme2.resolve())
    )


def test_store_payload_tag_scopes_segment_mounts(tmp_path):
    """Benchmark payload tags isolate segment files without changing runtime mounts."""
    preset = {
        "type": "nvme-device-group",
        "devices": [{"mount_path": str(tmp_path)}],
    }
    store = TuttiKVStore(
        tmp_path / "meta-root", 8, SEG, preset=preset,
        payload_tag="block256",
    )
    assert store._layout.mounts == (
        str((tmp_path / "tutti-payload-block256").resolve()),
    )
    with pytest.raises(ValueError, match="payload_tag"):
        TuttiKVStore(tmp_path / "bad", 8, SEG, payload_tag="../escape")


@pytest.mark.parametrize("preset_type", ["raid0", None])
def test_raid0_preset_requires_one_mount_and_defaults_to_raid0(preset_type, tmp_path):
    from tutti.storage.tutti_nvme.runtime_factory import prepare_preset

    preset = {"devices": [{"mount_path": str(tmp_path)}, {"mount_path": str(tmp_path)}]}
    if preset_type is not None:
        preset["type"] = preset_type
    assert prepare_preset(preset)["type"] == "raid0"
    assert ("type" in preset) == (preset_type is not None)


@pytest.mark.parametrize("devices", [
    [], [{"mount_path": "/one"}],
    [{"mount_path": "/one"}, {"mount_path": "/two"}],
    [{"mount_path": "/one"}, {}],
])
def test_raid0_preset_rejects_non_array_layout(devices):
    from tutti.storage.tutti_nvme.runtime_factory import prepare_preset

    with pytest.raises(RuntimeError, match="raid0"):
        prepare_preset({"type": "raid0", "devices": devices})


def test_raid0_rejects_payload_mount_override(tmp_path):
    from tutti.storage.tutti_nvme.runtime_factory import prepare_preset

    preset = {"type": "raid0", "devices": [{"mount_path": "/array"}] * 2}
    with pytest.raises(RuntimeError, match="mounts"):
        prepare_preset(preset, mounts=[str(tmp_path)])


@pytest.mark.parametrize("preset_type", ["striped", "stroped", "multi", "multi-nvme"])
def test_removed_preset_type_is_not_an_alias(preset_type):
    from tutti.storage.tutti_nvme.runtime_factory import prepare_preset

    with pytest.raises(RuntimeError, match="未知 preset"):
        prepare_preset({"type": preset_type})


def test_raid0_runtime_rejects_non_md_mount(tmp_path):
    runtime = _load_bindings_runtime()
    preset = {"devices": [
        {"pci_bdf": "ffff:00:00.0", "mount_path": str(tmp_path), "backing_device": "/dev/nonexistent-a"},
        {"pci_bdf": "ffff:00:01.0", "mount_path": str(tmp_path), "backing_device": "/dev/nonexistent-b"},
    ]}
    with pytest.raises(Exception, match="md|raid0"):
        runtime.make_raid0_nvme_runtime(preset)


# ---------- layout 单元 ----------


def test_decode_io_key_rules():
    assert decode_io_key(b"a" * 16 + (3).to_bytes(2, "little")) == (b"a" * 16, 3)
    assert decode_io_key(b"short") == (b"short", 0)  # <16B：整体即 chunk 身份
    assert decode_io_key(b"a" * 16 + b"\xff\x00") == (b"a" * 16, 255)
    with pytest.raises(ValueError):
        decode_io_key(b"")
    with pytest.raises(ValueError):
        decode_io_key("not-bytes")


def test_scan_reports_committed_objects_only(tmp_path):
    """scan 只报已提交对象（对象层职责）；目录里的杂散文件不参与判定。"""
    store = make_store(tmp_path, layers=1)
    store.open()
    src_id = store.register_buffer(bytearray(SEG), SEG)
    key = io_key(b"\xab" * 16, 0)
    store.put_batch([(key, src_id, 0)]).wait()
    (tmp_path / "pool" / "stray.bin").write_text("junk")
    assert store.scan() == [key]


# ---------- 写排序开关（store options / env 后门，§11.1-③） ----------


def test_defer_writes_after_reads_option_and_env_backdoor(
    tmp_path, monkeypatch
):
    """显式选项优先于 env 后门；缺省为 True；非 bool 拒绝。"""
    from tutti.storage.tutti_nvme.store import TuttiKVStore

    monkeypatch.setenv("TUTTI_DEFER_WRITES_AFTER_READS", "0")
    explicit = TuttiKVStore(
        tmp_path / "explicit", 4, SEG, defer_writes_after_reads=True,
    )
    assert explicit.defer_writes_after_reads is True

    from_env = TuttiKVStore(tmp_path / "env", 4, SEG)
    assert from_env.defer_writes_after_reads is False

    monkeypatch.delenv("TUTTI_DEFER_WRITES_AFTER_READS", raising=False)
    default = TuttiKVStore(tmp_path / "default", 4, SEG)
    assert default.defer_writes_after_reads is True

    with pytest.raises(ValueError, match="defer_writes_after_reads"):
        TuttiKVStore(
            tmp_path / "bad", 4, SEG, defer_writes_after_reads=1,
        )


# ---------- 周期落盘（checkpoint_if_due）----------


def test_checkpoint_if_due_throttles_and_can_be_disabled(tmp_path, monkeypatch):
    """提交索引周期落盘：按时间节流，间隔 0 时关闭。

    回归（2026-09-21 长跑复盘）：commit() 只写对象头并把检查点标脏，脏检查点
    原先只在 close() 落盘；长跑进程被强杀/崩溃时索引从未持久化——重启后
    盘上 4 TiB KV 完好却查不到（hit_tokens=0，等效冷池）。
    """
    from tutti.storage.tutti_nvme.store import TuttiKVStore

    monkeypatch.setenv("TUTTI_CHECKPOINT_INTERVAL_S", "0")
    disabled = TuttiKVStore(tmp_path / "off", 4, SEG)
    assert disabled.checkpoint_if_due() is False

    monkeypatch.setenv("TUTTI_CHECKPOINT_INTERVAL_S", "3600")
    store = TuttiKVStore(tmp_path / "on", 4, SEG)
    # 刚构造：距上次落盘不足一个间隔 → 跳过
    assert store.checkpoint_if_due() is False
    # 模拟间隔已过（把上次落盘时刻推回原点）→ 触发
    store._checkpoint_last_ns = 0
    assert store.checkpoint_if_due() is True
    # 落盘后重新计时 → 紧接着的下一步不再落盘
    assert store.checkpoint_if_due() is False


def test_checkpoint_if_due_rejects_bad_env(tmp_path, monkeypatch):
    """非法间隔值回退默认（不因环境变量写错而崩或静默关闭）。"""
    from tutti.storage.tutti_nvme.store import TuttiKVStore

    monkeypatch.setenv("TUTTI_CHECKPOINT_INTERVAL_S", "not-a-number")
    store = TuttiKVStore(tmp_path / "bad", 4, SEG)
    assert store._checkpoint_interval_s == 60.0
def test_drain_deferred_reads_retains_writes_and_unobserved_reads():
    from types import SimpleNamespace
    from tutti.storage.tutti_nvme.store import TuttiKVStore

    store = object.__new__(TuttiKVStore)
    calls = []
    error = RuntimeError("read observation failed")
    def failed_read():
        calls.append("read")
        raise error
    read = SimpleNamespace(_direction="read", wait_result=failed_read)
    write = SimpleNamespace(_direction="write", wait_result=lambda: calls.append("write"))
    store._deferred_completions = [read, write]
    with pytest.raises(RuntimeError, match="read observation failed"):
        store.drain_deferred_completions(direction="read")
    assert calls == ["read"]
    assert store._deferred_completions == [write, read]
    read.wait_result = lambda: calls.append("read-recovered")
    store.drain_deferred_completions()
    assert calls == ["read", "write", "read-recovered"]
    assert store._deferred_completions == []

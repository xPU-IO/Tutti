from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tutti.integration.vllm import worker as worker_module
from tutti.integration.vllm.worker import WorkerImpl
from tutti.engine.core import KVEngine, _DirectAllLayerReadPlan, _ReadPlan
from tutti.engine.transfer import (
    DirectTransfer,
    DirectTransferUnavailable,
    StagedTransfer,
    select_transfer,
)
from tutti.index.chunk_index import derive_io_key
from tutti.storage.tutti_nvme.object_layout import DEFAULT_SEGMENT_HEADER_BYTES
from tutti.storage.tutti_nvme.store import (
    DirectAdmissionError,
    TuttiDirectBackend,
    TuttiKVStore,
)


class FakePool:
    is_cuda = True

    def __init__(self, block_size, *, blocks=8, layers=3, base=0x20000,
                 kv_channels=None, stride=None, contiguous=True):
        kv_channels = kv_channels or 4096 // (block_size * 4)
        self.shape = (blocks, layers, block_size, 2, kv_channels)
        expected = (
            layers * block_size * 2 * kv_channels,
            block_size * 2 * kv_channels,
            2 * kv_channels,
            kv_channels,
            1,
        )
        self._stride = tuple(stride or expected)
        self._base = base
        self._contiguous = contiguous

    def dim(self):
        return len(self.shape)

    def stride(self):
        return self._stride

    def element_size(self):
        return 2

    def numel(self):
        result = 1
        for value in self.shape:
            result *= value
        return result

    def data_ptr(self):
        return self._base

    def is_contiguous(self):
        return self._contiguous

    def get_device(self):
        return 0


class FakeLayout:
    """对象层布局的最小替身：容量按对象、段 0 跳过对象头。

    ``payload_offset`` 与 C++ 对象层一致（4096B 自描述头），因此这里验证的
    地址公式等同于真机：``payload_offset + layer * segment_bytes + 块内偏移``。
    """

    payload_offset = 4096
    layer_span = 3

    def __init__(self, events=None):
        self.prepared = []
        self.events = events if events is not None else []
        self.generations = {}

    def prepare_put(self, io_keys, capacity):
        """预留对象：返回 ``(admitted, rejected)``（对象层契约）。"""
        self.prepared.append((tuple(io_keys), capacity))
        self.events.append("prepare_put")
        admitted = {}
        for io_key in io_keys:
            chunk_id = bytes(io_key[:16])
            self.generations.setdefault(chunk_id, 1)
            admitted[bytes(io_key)] = (
                chunk_id, int.from_bytes(bytes(io_key[16:]), "little")
            )
        return admitted, 0

    @staticmethod
    def target_uri(chunk_id):
        return f"file:///{bytes(chunk_id).hex()}"

    @staticmethod
    def target_size(chunk_id):
        return 3 * 8192

    @staticmethod
    def target_offset(chunk_id):
        return FakeLayout.payload_offset

    def target_generation(self, chunk_id):
        return self.generations.get(bytes(chunk_id), 1)


class FakeRuntime:
    def __init__(self, *, layers=3, in_flight=None, max_batch=64,
                 fail=False, unregister_error=None):
        self._layers = layers
        self._in_flight = 2 * layers if in_flight is None else in_flight
        self._max_batch = max_batch
        self._fail = fail
        self._unregister_error = unregister_error
        self.register_calls = []
        self.unregister_calls = []
        self.submit_calls = []
        self.wait_calls = []
        self.release_calls = []
        self.open_batch_calls = []
        self.close_batch_calls = []
        self.events = []
        self._next = 100

    def caps(self):
        return {
            "supports_multi_stream": True,
            "max_concurrent_streams": 2,
            "max_in_flight_operations": self._in_flight,
            "max_batch_entries": self._max_batch,
            "memory": ["device"],
        }

    def register_memory(self, addr, size, kind, accel_id=-1,
                        io_granularity=0):
        self.register_calls.append(
            (addr, size, kind, accel_id, io_granularity)
        )
        return 41

    def unregister_memory(self, ticket):
        if self._unregister_error is not None:
            raise self._unregister_error
        self.events.append("unregister_memory")
        self.unregister_calls.append(ticket)

    def submit(self, requests, **kwargs):
        self.events.append("submit")
        self.submit_calls.append((tuple(requests), kwargs))
        self._next += 1
        return SimpleNamespace(
            status_ok=True,
            status_msg="",
            io_handle=self._next,
            initial_states=[True] * len(requests),
            rejected=[],
        )

    def wait(self, handle, timeout_ms=0):
        self.events.append("wait")
        self.wait_calls.append((handle, timeout_ms))
        return ("OK", "FAILED") if self._fail else ("OK", "COMPLETED")

    def release_io(self, handle):
        self.events.append("release_io")
        self.release_calls.append(handle)

    def open_batch(self, uris):
        uris = tuple(uris)
        self.events.append("open_batch")
        self.open_batch_calls.append(uris)
        return [1000 + len(self.open_batch_calls) * 100 + index
                for index in range(len(uris))]

    def close_batch(self, tickets):
        self.events.append("close_batch")
        self.close_batch_calls.append(tuple(tickets))


class FakeStoreOwner:
    def __init__(self, runtime):
        self._runtime = runtime
        self._read_stream = 11
        self._write_stream = 22
        self._read_async_stream = None
        self._accel_id = 0
        self._execution = "device"
        self._layout = FakeLayout(runtime.events)
        self._num_chunks = 16
        self._live = set()
        self._targets = {}
        self._submit_width_limit = None
        self.put_results = []
        self.ensure_targets_calls = 0

    def _runtime_supports_multi_stream(self):
        return True

    def create_direct_transfer(self, pool, **kwargs):
        return TuttiDirectBackend(self)

    def _stream_for(self, direction):
        return {"read": 11, "write": 22,
                "read_async": self._read_async_stream}[direction], None

    def _ensure_targets(self, entries):
        self.ensure_targets_calls += 1
        uris = tuple(dict.fromkeys(
            self._layout.target_uri(io_key[:16])
            for io_key, _, _ in entries
        ))
        missing = [uri for uri in uris if uri not in self._targets]
        if missing:
            tickets = self._runtime.open_batch(missing)
            for uri, ticket in zip(missing, tickets):
                self._targets[uri] = SimpleNamespace(
                    ticket=ticket, size=3 * 8192, generation=1,
                )
        return {uri: self._targets[uri].ticket for uri in uris}

    def _submit_retry(self, requests, direction, stream_direction=None):
        return TuttiKVStore._submit_retry(
            self, requests, direction, stream_direction=stream_direction
        )

    def _batch_width_limit(self):
        return TuttiKVStore._batch_width_limit(self)

    def _on_put_settled(self, ok, io_keys):
        self.put_results.append((ok, tuple(io_keys)))

    def _track_completion(self, completion, chunk_ids):
        return None

    def _close_cached_targets(self, uris):
        return TuttiKVStore._close_cached_targets(self, uris)


class FakeEngineStore(FakeStoreOwner):
    capacity_chunks = 16
    max_in_flight_operations = 6

    def __init__(self, runtime):
        super().__init__(runtime)
        self.layer_span = None
        self.closed = False

    def open(self):
        return None

    def close(self):
        self.closed = True

    def scan(self):
        return []

    def set_key_namespace(self, namespace):
        return None

    def set_layer_span(self, num_layers):
        self.layer_span = num_layers

    def drop(self, keys):
        return None


def _warm_up_backend(tickets):
    """绑定期预热的替身：对象层已物化槽位的 URI 与票据。

    ``tickets=None`` 模拟非池部署（对象层没有可预开的槽位）。
    """
    runtime = FakeRuntime()
    store = FakeStoreOwner(runtime)
    if tickets is not None:
        store._warm_uris = [f"file:///slot{slot}.obj" for slot in range(len(tickets))]
        store._targets = {
            uri: SimpleNamespace(ticket=ticket, size=3 * 8192, generation=1)
            for uri, ticket in zip(store._warm_uris, tickets)
        }
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    return runtime, backend


def test_direct_warm_up_registration_issues_one_page_read():
    """绑定期预热：一个 1-page 读触发 peer-memory 注册（首轮成本前移）。

    注册本身在 C++ 侧是惰性的（首次 submit 触发 nvm_dma_map_data_device，
    8 卡实测 476ms 且持 registry 锁），这里只验证 Python 侧正确地用
    一个最小的读把该路径走通，且不写盘、不动 live 集合。
    """
    runtime, backend = _warm_up_backend([777])
    try:
        assert backend.warm_up_registration() is True
        requests, kwargs = runtime.submit_calls[-1]
        assert len(requests) == 1
        target, target_offset, memory, memory_offset, length, direction = (
            requests[0]
        )
        assert target == 777
        assert target_offset == 0
        assert memory == 41          # register_paged_caches 的 memory ticket
        assert memory_offset == 0
        assert length == backend.geometry.page_bytes
        assert direction == "read"
        assert kwargs.get("stream") == 11     # 读流
        # 读方向不产生落盘回调，也不改 live 集合
        assert backend._store.put_results == []
        assert backend._store._live == set()
    finally:
        backend.close()


def test_direct_warm_up_registration_skips_without_ready_targets():
    """无对象池（非池部署）或槽位句柄未就绪时预热直接跳过，不抛异常。"""
    runtime, backend = _warm_up_backend(None)
    try:
        assert backend.warm_up_registration() is False
    finally:
        backend.close()

    runtime, backend = _warm_up_backend([0])   # ticket=0 → 尚未 open
    try:
        assert backend.warm_up_registration() is False
        assert runtime.submit_calls == []
    finally:
        backend.close()


@pytest.mark.parametrize("block_size", [64, 128, 256])
def test_direct_address_formula_and_one_submit_per_layer(block_size):
    runtime = FakeRuntime()
    store = FakeStoreOwner(runtime)
    pool = FakePool(block_size)
    backend = TuttiDirectBackend(store)
    chunk_tokens = block_size * 2
    backend.register_paged_caches(
        pool,
        num_layers=3,
        blocks_per_chunk=2,
        chunk_tokens=chunk_tokens,
        segment_bytes=8192,
        max_chunks_per_wave=2,
    )
    assert runtime.register_calls == [
        (pool.data_ptr(), pool.numel() * 2, "device", 0, 0)
    ]

    keys = [b"a" * 16, b"b" * 16]
    tables = [[3, 1], [7, 0]]
    store._live.update(derive_io_key(key, 2) for key in keys)
    completion = backend.get_paged_batch(keys, 2, tables)
    completion.wait()

    assert len(runtime.submit_calls) == 1
    requests, kwargs = runtime.submit_calls[0]
    geometry = backend.geometry
    assert kwargs["stream"] == store._read_stream
    assert [request[3] for request in requests] == [
        block_id * geometry.block_stride_bytes
        + 2 * geometry.layer_stride_bytes
        for block_id in (3, 1, 7, 0)
    ]
    assert [request[1] for request in requests] == [
        FakeLayout.payload_offset
        + 2 * geometry.segment_bytes
        + ordinal * geometry.page_bytes
        for ordinal in (0, 1, 0, 1)
    ]
    assert {request[4] for request in requests} == {geometry.page_bytes}
    assert {request[5] for request in requests} == {"read"}

    backend.close()
    backend.close()
    assert runtime.unregister_calls == [41]


def test_direct_write_is_one_submit_for_multiple_chunks_and_blocks():
    runtime = FakeRuntime()
    store = FakeStoreOwner(runtime)
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    keys = [b"c" * 16, b"d" * 16]
    completion = backend.put_paged_batch(keys, 1, [[6, 2], [5, 0]])
    completion.wait()
    assert len(runtime.submit_calls) == 1
    assert len(runtime.submit_calls[0][0]) == 4
    assert store.put_results == [
        (True, tuple(derive_io_key(key, 1) for key in keys))
    ]


def test_direct_target_plan_is_built_once_per_direction():
    runtime = FakeRuntime()
    store = FakeStoreOwner(runtime)
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    keys = [b"a" * 16, b"b" * 16]
    store._live.update(
        derive_io_key(key, layer) for key in keys for layer in range(3)
    )
    backend.get_paged_batch(keys, 0, [[3, 1], [7, 0]])
    backend.get_paged_batch(keys, 1, [[3, 1], [7, 0]])
    backend.get_paged_batch(keys, 2, [[3, 1], [7, 0]])
    assert store.ensure_targets_calls == 1
    assert backend._target_plans["read"].plan_token == 1
    backend.end_target_plan("read")
    backend.put_paged_batch(keys, 0, [[3, 1], [7, 0]])
    backend.put_paged_batch(keys, 1, [[3, 1], [7, 0]])
    assert store.ensure_targets_calls == 2
    assert backend._target_plans["write"].plan_token == 2
    backend.close()


def test_detached_read_plan_submits_on_async_stream():
    """异步加载的读走独立流：submit kernel 在流上跑到 IO 完成，共用读流会让
    同一 rank 本步的同步读排在它后面。与同步读计划并存、互不占槽位。"""
    runtime = FakeRuntime()
    store = FakeStoreOwner(runtime)
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    keys = [b"a" * 16, b"b" * 16]
    store._live.update(
        derive_io_key(key, layer) for key in keys for layer in range(3)
    )
    with pytest.raises(RuntimeError, match="dedicated read stream"):
        backend.begin_detached_read_plan(keys[:1])
    store._read_async_stream = 33
    detached = backend.begin_detached_read_plan(keys[:1])
    backend.get_paged_batch(keys[:1], 0, [[3, 1]], target_plan=detached)
    assert runtime.submit_calls[-1][1]["stream"] == 33
    backend.get_paged_batch(keys[1:], 0, [[7, 0]])
    assert runtime.submit_calls[-1][1]["stream"] == 11
    assert backend._target_plans["read"].plan_token != detached.plan_token
    backend.end_detached_read_plan(detached)
    assert detached.plan_token not in backend._detached_plans
    backend.close()


def _async_layer_engine(monkeypatch, layers=3):
    from types import SimpleNamespace as NS
    from tutti.engine.core import _AsyncReadPlan
    monkeypatch.setattr(_AsyncReadPlan, "_timing_event", staticmethod(lambda: None))
    engine = object.__new__(KVEngine)
    engine._closed = False
    engine._direct_submit_lock = threading.RLock()
    engine._active_read_plan = None
    engine._foreground_read_submit_pending = 0
    engine._async_read_plans = set()
    engine.begin_prefill_feeder(True)
    engine.advance_prefill_feeder(0)
    engine._store = NS(_accel_id=None)
    calls, completions, ended = [], {}, []
    class Completion:
        def __init__(self):
            self.done = False
            self.wait_count = 0
        def poll(self):
            return True if self.done else None
        def wait(self):
            self.wait_count += 1
            self.done = True
    backend = NS(begin_detached_read_plan=lambda keys: object(),
                 end_detached_read_plan=lambda plan: ended.append(plan))
    def load(keys, layer, tables, target_plan):
        calls.append((layer, tuple(keys), tuple(tuple(t) for t in tables)))
        completion = Completion()
        completions[layer] = completion
        return completion
    engine._transfer = NS(_backend=backend, load_layer=load)
    return engine, calls, completions, ended


def _wait_until(predicate, seconds=2):
    import time
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "background feeder did not progress"
        time.sleep(.001)


def test_async_layer_list_pauses_between_prefills_without_host_sync(monkeypatch):
    from types import SimpleNamespace as NS
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch)
    engine.end_prefill_feeder()
    engine._active_read_plan = NS(failed=None, all_io_done=lambda: True,
        fence_event=lambda callback: pytest.fail("gate must not use only the last fence"))
    plan = _AsyncReadPlan(engine, [b"queued"], [[7]], (0, 1, 2))
    try:
        assert not plan._feeder_done.wait(.05)
        assert calls == []
        engine.begin_prefill_feeder(True)
        engine.advance_prefill_feeder(0)
        _wait_until(lambda: len(calls) == 1)
        assert plan.submitted_layers == (0,)
        assert plan.poll() is None
        assert len(calls) == 1
        engine.end_prefill_feeder()
        completed[0].done = True
        assert not plan._feeder_done.wait(.05)
        assert len(calls) == 1
        engine.begin_prefill_feeder(False)
        engine.advance_prefill_feeder(1)
        assert not plan._feeder_done.wait(.05)
        assert len(calls) == 1
        engine.begin_prefill_feeder(True)
        engine.advance_prefill_feeder(0)
        _wait_until(lambda: len(calls) == 2)
        completed[1].done = True
        engine.advance_prefill_feeder(1)
        _wait_until(lambda: len(calls) == 3)
        completed[2].done = True
        assert plan._feeder_done.wait(2)
        assert plan.poll() is True
        assert [c[0] for c in calls] == [0, 1, 2]
        assert all(c.wait_count == 0 for c in completed.values())
        assert len(ended) == 1
    finally:
        plan.abort()


def test_same_prefill_advances_one_batch_at_a_time_until_compute_event_finishes(monkeypatch):
    from types import SimpleNamespace as NS
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch)
    compute = NS(done=False)
    records = []
    engine._store.record_compute_event = lambda: records.append(compute) or NS(query=lambda: compute.done)
    plan = _AsyncReadPlan(engine, [b"queued"], [[7]], (0, 1, 2))
    try:
        _wait_until(lambda: len(calls) == 1)
        engine.finish_prefill_feeder()
        engine.finish_prefill_feeder()
        assert len(records) == 1
        completed[0].done = True
        _wait_until(lambda: len(calls) == 2)
        assert not completed[1].done
        compute.done = True
        completed[1].done = True
        engine._wake_read_service()
        assert not plan._feeder_done.wait(.03)
        assert len(calls) == 2 and not engine._background_read_allowed()
        engine.begin_prefill_feeder(True)
        assert not engine._background_read_allowed()
        engine.advance_prefill_feeder(0)
        _wait_until(lambda: len(calls) == 3)
        completed[2].done = True
        plan._feeder.join(2)
        assert plan.poll() is True
    finally:
        plan.abort()
        engine._get_read_service().close()


def test_queued_request_starts_after_fifo_head_finishes_without_scheduler_poll(monkeypatch):
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch, layers=1)
    first = _AsyncReadPlan(engine, [b"first"], [[7]], (0,))
    _wait_until(lambda: len(calls) == 1)
    first_io = completed[0]
    second = _AsyncReadPlan(engine, [b"second"], [[8]], (0,))
    try:
        assert not second._feeder_done.wait(.03)
        assert len(calls) == 1
        first_io.done = True
        _wait_until(lambda: len(calls) == 2)
        assert calls[0][1] == (b"first",) and calls[1][1] == (b"second",)
        assert len(ended) == 1
        completed[0].done = True
        second._feeder.join(2)
        first._feeder.join(2)
        assert first.poll() is True and second.poll() is True
        assert len(ended) == 2
    finally:
        first.abort()
        second.abort()
        engine._get_read_service().close()


@pytest.mark.parametrize("main_done,index_done", [(False, False), (True, False), (False, True), (True, True)])
def test_all_io_done_checks_both_frontend_streams(main_done, index_done):
    plan = object.__new__(_DirectAllLayerReadPlan)
    plan.physical_layers = (2, 0)
    plan.terminal_failure = None
    plan._all_io_complete = False
    plan._ready_cond = threading.Condition()
    states = [index_done, main_done]
    events = [SimpleNamespace(query=lambda i=i: states[i],
                synchronize=lambda: pytest.fail("host synchronization is forbidden"))
              for i in range(2)]
    plan.read_ready_events = {0: events[0]}
    assert not plan.all_io_done(), "a partial submission cannot open the gate"
    plan.read_ready_events[1] = events[1]
    assert plan.all_io_done() is (main_done and index_done)
    states[:] = [True, True]
    assert plan.all_io_done()
    plan.terminal_failure = RuntimeError("late IO failure")
    assert not plan.all_io_done()


def test_all_io_done_query_error_fails_closed():
    plan = object.__new__(_DirectAllLayerReadPlan)
    plan.physical_layers = (0,)
    plan.terminal_failure = None
    plan._all_io_complete = False
    plan._ready_cond = threading.Condition()
    plan.on_failure = None
    error = RuntimeError("CUDA event query failed")
    def query():
        raise error
    plan.read_ready_events = {0: SimpleNamespace(query=query)}
    assert not plan.all_io_done()
    assert plan.failed is error


def test_compute_wait_profile_reports_pending_layer_fraction_and_exposed_wait():
    plan = object.__new__(_DirectAllLayerReadPlan)
    plan.physical_layers = (0, 1, 2, 3)
    plan.read_pending_callbacks = {1, 3}
    class End:
        def query(self):
            return True
    class Start:
        def elapsed_time(self, _end):
            return 2.5
    plan.compute_wait_events = {1: (Start(), End()), 3: (Start(), End())}
    assert plan.compute_wait_profile() == (0.5, 5.0)

    plan.compute_wait_events[3][1].query = lambda: False
    assert plan.compute_wait_profile() is None


def test_detached_background_does_not_wait_for_front_read_tail(monkeypatch):
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch, layers=1)
    plan = object.__new__(_DirectAllLayerReadPlan)
    plan.physical_layers = (2, 0)
    plan.terminal_failure = None
    plan._all_io_complete = False
    plan._ready_cond = threading.Condition()
    state = [False, True]
    plan.read_ready_events = {i: SimpleNamespace(query=lambda i=i: state[i]) for i in range(2)}
    engine._active_read_plan = plan
    background = _AsyncReadPlan(engine, [b"queued"], [[7]], (0,))
    try:
        _wait_until(lambda: len(calls) == 1)
        assert calls == [(0, (b"queued",), ((7,),))]
        completed[0].done = True
        assert background._feeder_done.wait(2)
        assert background.poll() is True
        engine.end_prefill_feeder()
        assert not engine._background_read_allowed()
        engine.begin_prefill_feeder(False)
        engine.advance_prefill_feeder(0)
        assert not engine._background_read_allowed()
    finally:
        background.abort()


def test_new_foreground_plan_takes_priority_over_unsubmitted_background_layers(monkeypatch):
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch)
    plan = _AsyncReadPlan(engine, [b"queued"], [[7]], (0, 1, 2))
    try:
        _wait_until(lambda: len(calls) == 1)
        engine._foreground_read_submit_pending = 1
        completed[0].done = True
        assert not plan._feeder_done.wait(.05)
        assert len(calls) == 1
        assert plan.submitted_layers == (0,)
        engine._foreground_read_submit_pending = 0
        engine.advance_prefill_feeder(1)
        _wait_until(lambda: len(calls) == 2)
        completed[1].done = True
        engine.advance_prefill_feeder(2)
        _wait_until(lambda: len(calls) == 3)
        completed[2].done = True
        assert plan._feeder_done.wait(2)
        assert plan.poll() is True
    finally:
        engine._foreground_read_submit_pending = 0
        plan.abort()


def test_cancel_partial_async_read_keeps_submitted_io_until_drained(monkeypatch):
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch)
    plan = _AsyncReadPlan(engine, [b"queued"], [[7]], (0, 1, 2))
    _wait_until(lambda: len(calls) == 1)
    plan.abort()
    assert len(calls) == 1
    assert completed[0].wait_count == 1
    assert len(ended) == 1


@pytest.mark.parametrize("late_failure", [False, True])
def test_managed_async_cleanup_never_blocks_poll_or_releases_early(monkeypatch, late_failure):
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch, layers=1)
    backend = engine._transfer._backend
    backend.manages_background_reads = True
    owner = threading.get_ident()
    cleanup_entered, release = threading.Event(), threading.Event()
    backend.prepare_detached_read = lambda *args: None
    backend.start_detached_read = lambda *args: None
    def load(*args, **kwargs):
        return type("Ready", (), {"poll": lambda self: True})()
    backend.try_get_paged_batch = load
    def cleanup(target):
        if not ended:
            assert threading.get_ident() != owner
        cleanup_entered.set()
        assert release.wait(3)
        if late_failure and not ended:
            ended.append("failed")
            raise RuntimeError("late owned IO failure")
        ended.append(target)
    backend.end_detached_read_plan = cleanup
    plan = _AsyncReadPlan(engine, [b"queued"], [[7]], (0,))
    try:
        assert cleanup_entered.wait(2)
        assert plan.poll() is None
        assert not plan._ended
    finally:
        release.set()
        plan._feeder.join(3)
    if late_failure:
        assert plan.failure is not None
        assert plan.poll() is None
        assert plan in engine._async_read_plans and not plan._ended
        assert plan._cleanup_job.done.is_set()
        plan.poll()
        assert plan.poll() is None
        plan.abort()
        assert plan.poll() is False
    else:
        assert plan.poll() is True


def test_managed_async_cleanup_returned_error_stays_failed(monkeypatch):
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch, layers=1)
    backend = engine._transfer._backend
    backend.manages_background_reads = True
    backend.prepare_detached_read = lambda *args: None
    backend.start_detached_read = lambda *args: None
    backend.try_get_paged_batch = lambda *args, **kwargs: type(
        "Ready", (), {"poll": lambda self: True})()
    error = RuntimeError("sticky owned IO failure")
    backend.end_detached_read_plan = lambda target: error
    plan = _AsyncReadPlan(engine, [b"queued"], [[7]], (0,))
    plan._feeder.join(3)
    assert plan.poll() is False and plan.failure is error
    assert plan not in engine._async_read_plans


def test_prefill_close_serializes_with_actual_background_enqueue(monkeypatch):
    engine, _, _, _ = _async_layer_engine(monkeypatch, layers=1)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    def submit():
        with engine._background_submit_scope() as allowed:
            assert allowed
            entered.set()
            assert release.wait(3)
    def close():
        engine.end_prefill_feeder()
        closed.set()
    submitting = threading.Thread(target=submit)
    submitting.start()
    assert entered.wait(2)
    closing = threading.Thread(target=close)
    closing.start()
    try:
        assert not closed.wait(.03)
    finally:
        release.set()
        submitting.join(3)
        closing.join(3)
    assert closed.is_set()
    with engine._background_submit_scope() as allowed:
        assert not allowed


def test_async_partial_submit_failure_does_not_release_blocks_before_accepted_io_finishes(monkeypatch):
    from types import SimpleNamespace as NS
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch)
    accepted_done = threading.Event()
    waiting = threading.Event()
    def wait():
        waiting.set()
        assert accepted_done.wait(3)
    completion = NS(poll=lambda: True if accepted_done.is_set() else None,
                    wait=wait)
    engine._store._deferred_completions = [completion]
    def load(*args, **kwargs):
        error = RuntimeError("partial read submit failed")
        error.tutti_submitted_completions = (completion,)
        raise error
    engine._transfer.load_layer = load
    plan = _AsyncReadPlan(engine, [b"queued"], [[7]], (0, 1, 2))
    assert waiting.wait(2)
    assert engine._store._deferred_completions == []
    assert plan.poll() is None and ended == []
    assert plan in engine._async_read_plans
    accepted_done.set()
    plan._feeder.join(3)
    assert plan.poll() is False and len(ended) == 1
    assert str(plan.failure) == "partial read submit failed"


def test_background_failure_stops_before_next_entity_and_optional_timing_does_not_leak(monkeypatch):
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch)
    plan = _AsyncReadPlan(engine, [b"queued"], [[7]], (0, 1, 2))
    _wait_until(lambda: len(calls) == 1)
    def failed():
        raise RuntimeError("device read failed")
    completed[0].poll = failed
    assert plan._feeder_done.wait(2)
    plan._feeder.join(3)
    assert plan.poll() is False and len(ended) == 1
    assert [c[0] for c in calls] == [0]
    assert completed[0].done
    assert plan.poll() is False and len(ended) == 1
    assert str(plan.failure) == "device read failed"


@pytest.mark.parametrize("fail_at", ["timing", "thread"])
def test_async_constructor_failure_closes_unsubmitted_detached_plan(monkeypatch, fail_at):
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError("start failed")
    if fail_at == "timing":
        monkeypatch.setattr(_AsyncReadPlan, "_timing_event", staticmethod(lambda: object()))
        engine._store.record_read_async_event = fail
    else:
        monkeypatch.setattr(threading.Thread, "start", fail)
    with pytest.raises(RuntimeError, match="start failed"):
        _AsyncReadPlan(engine, [b"queued"], [[7]], (0, 1, 2))
    assert calls == [] and len(ended) == 1


def test_async_abort_observation_error_retains_plan_until_io_is_terminal(monkeypatch):
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch)
    plan = _AsyncReadPlan(engine, [b"queued"], [[7]], (0, 1, 2))
    engine._async_read_plans.add(plan)
    _wait_until(lambda: len(calls) == 1)
    def fail():
        raise RuntimeError("observation unavailable")
    completed[0].wait = fail
    error = engine._abort_async_read_plans()
    assert str(error) == "observation unavailable"
    assert plan in engine._async_read_plans and ended == []
    assert not plan._ended
    with pytest.raises(RuntimeError, match="observation unavailable"):
        engine.close()
    assert not engine._closed
    completed[0].wait = lambda: None
    completed[0].io_terminal = True
    assert engine._abort_async_read_plans() is None
    assert not engine._async_read_plans and len(ended) == 1


def test_direct_target_plan_rejects_chunk_not_in_immutable_plan():
    runtime = FakeRuntime()
    store = FakeStoreOwner(runtime)
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    key = b"c" * 16
    store._live.update(derive_io_key(key, layer) for layer in range(3))
    store._live.update(derive_io_key(b"d" * 16, layer) for layer in range(3))
    backend.begin_target_plan([key], "read")
    with pytest.raises(RuntimeError, match="target plan lacks chunk"):
        backend.get_paged_batch([b"d" * 16], 0, [[3, 1]])
    backend.close()


def test_direct_write_plan_prepares_before_open_and_reuses_generation():
    runtime = FakeRuntime()
    store = FakeStoreOwner(runtime)
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    keys = [b"w" * 16, b"x" * 16]
    first = backend.put_paged_batch(keys, 0, [[3, 1], [7, 0]])
    second = backend.put_paged_batch(keys, 1, [[3, 1], [7, 0]])
    assert runtime.events[:3] == ["prepare_put", "open_batch", "submit"]
    assert store._layout.prepared == [
        (tuple(derive_io_key(key, 2) for key in keys), store._num_chunks)
    ]
    assert len(runtime.open_batch_calls) == 1
    assert len(runtime.submit_calls) == 2
    assert first._watcher is None and second._watcher is None
    backend.close()


def test_direct_target_plan_generation_change_fails_closed():
    runtime = FakeRuntime()
    store = FakeStoreOwner(runtime)
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    key = b"g" * 16
    store._live.update(derive_io_key(key, layer) for layer in range(3))
    completion = backend.get_paged_batch([key], 0, [[3, 1]])
    # 槽位被回收再分配：对象层换了 generation，计划里的旧记录必须失效。
    store._layout.generations[key] = store._layout.target_generation(key) + 1
    with pytest.raises(RuntimeError, match="generation mismatch"):
        backend.get_paged_batch([key], 1, [[3, 1]])
    completion.wait()
    backend.close()


def test_direct_shared_segment_file_reuses_ticket_and_checks_each_chunk():
    runtime = FakeRuntime()
    store = FakeStoreOwner(runtime)
    store._layout.target_uri = lambda _chunk: "file:///shared.seg"
    store._layout.target_offset = lambda chunk: (
        32768 if chunk == b"a" * 16 else 32768 + 3 * 8192 + 32768
    )
    store._layout.generations[b"a" * 16] = 1
    store._layout.generations[b"b" * 16] = 2
    store._ensure_targets = lambda entries: TuttiKVStore._ensure_targets(store, entries)
    store._close_cached_targets = lambda uris: TuttiKVStore._close_cached_targets(store, uris)
    store._targets_lock = threading.RLock()

    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    keys = [b"a" * 16, b"b" * 16]
    store._live.update(derive_io_key(key, layer) for key in keys for layer in range(3))
    backend.begin_target_plan(keys, "read")
    assert len(runtime.open_batch_calls) == 1
    assert runtime.open_batch_calls[0] == ("file:///shared.seg",)
    plan = backend._target_plans["read"]
    assert [entry.target_generation for entry in plan.entries] == [1, 2]
    assert plan.entries[0].target_ticket == plan.entries[1].target_ticket
    backend.get_paged_batch(keys, 0, [[3, 1], [7, 0]]).wait()
    store._layout.generations[keys[1]] = 3
    with pytest.raises(RuntimeError, match="generation mismatch"):
        backend.get_paged_batch(keys, 1, [[3, 1], [7, 0]])
    assert len(runtime.open_batch_calls) == 1
    backend.close()


def test_direct_read_write_plans_are_isolated():
    runtime = FakeRuntime()
    store = FakeStoreOwner(runtime)
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    read_key = b"r" * 16
    write_key = b"s" * 16
    store._live.update(
        derive_io_key(read_key, layer) for layer in range(3)
    )
    store._live.update(
        derive_io_key(write_key, layer) for layer in range(3)
    )
    backend.begin_target_plan([read_key], "read")
    backend.prepare_write_targets([write_key])
    backend.begin_target_plan([write_key], "write")
    read_plan = backend._target_plans["read"]
    write_plan = backend._target_plans["write"]
    assert read_plan.plan_token != write_plan.plan_token
    with pytest.raises(RuntimeError, match="target plan lacks chunk"):
        backend.get_paged_batch([write_key], 0, [[3, 1]])
    backend.end_target_plan("read")
    assert "write" in backend._target_plans
    backend.close()


def test_direct_target_plan_scales_to_eighty_layers():
    runtime = FakeRuntime(layers=80, in_flight=160)
    store = FakeStoreOwner(runtime)
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128, layers=80), num_layers=80, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    keys = [b"8" * 16, b"9" * 16]
    store._live.update(
        derive_io_key(key, layer) for key in keys for layer in range(80)
    )
    for layer in range(80):
        backend.get_paged_batch(keys, layer, [[3, 1], [7, 0]])
    assert store.ensure_targets_calls == 1
    assert len(runtime.open_batch_calls) == 1
    backend.close()


def test_direct_clean_root_first_write_materializes_target(tmp_path):
    runtime = FakeRuntime()
    store = TuttiKVStore(
        root=tmp_path / "clean-root", num_chunks=2, segment_bytes=8192,
        runtime=runtime, segment_file_slots=4,
    )
    store.open()
    store._read_stream = 11
    store._write_stream = 22
    store.set_layer_span(3)
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    key = b"clean-root-key!".ljust(16, b"_")
    completion = backend.put_paged_batch([key], 0, [[3, 1]])
    uri = store._layout.target_uri(key)
    target_path = Path(uri[len("file://"):])
    # 段文件在 open 时整体建好：每槽位 = 前缀 + 全部层的段（层数 × 段大小）。
    assert target_path.exists()
    assert target_path.stat().st_size == 4 * (DEFAULT_SEGMENT_HEADER_BYTES + 3 * 8192)
    assert len(runtime.open_batch_calls) == 1
    assert completion._watcher is None
    completion.wait()
    backend.close()
    store.close()


def test_direct_real_store_failure_rolls_back_live_and_layout(tmp_path):
    runtime = FakeRuntime(fail=True)
    store = TuttiKVStore(
        root=tmp_path / "failed-write", num_chunks=2, segment_bytes=8192,
        runtime=runtime,
    )
    store.open()
    store._read_stream = 11
    store._write_stream = 22
    store.set_layer_span(3)
    # 槽位目标在启动期（set_layer_span）预开，失败写入不得再新增票据：
    # 槽位是稳定身份，失败的是该 chunk 的绑定而非槽位文件本身。
    ready = dict(store._targets)
    assert ready and all(entry.ticket for entry in ready.values()), (
        "槽位应在 set_layer_span 时就绪化"
    )
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    key = b"failed-write-key"[:16]
    completion = backend.put_paged_batch([key], 0, [[3, 1]])
    with pytest.raises(RuntimeError, match="失败"):
        completion.wait()
    assert store._live == set()
    # 已就绪的槽位票据不得因一次失败的写入而变化（票据随槽位常驻）
    assert store._targets == ready
    # 失败的写入不得提交对象：按对象提交 ⇒ 半截对象读不得
    assert not store._layout.is_committed(key)
    assert "write" not in backend._target_plans
    backend.close()
    store.close()


def test_engine_plan_store_prepares_direct_write_before_first_layer():
    runtime = FakeRuntime()
    store = FakeEngineStore(runtime)
    engine = KVEngine(
        {
            "chunk_tokens": 256,
            "chunk_kv_bytes": 3 * 8192,
            "max_chunks_per_wave": 2,
            "num_layers": 3,
        },
        store,
    )
    assert engine.try_bind_direct(FakePool(128), 3, 2)
    key = b"plan-store-key!".ljust(16, b"_")
    plan = engine.plan_store([key])
    assert plan is not None and plan.new_keys == [key]
    assert runtime.events == ["prepare_put"]
    engine.store_layer([key], 0, [[3, 1]])
    assert runtime.events[:3] == ["prepare_put", "open_batch", "submit"]
    engine.wait_idle()
    engine.confirm_store([key])
    engine.close()


def test_direct_completion_has_no_watcher_until_request_drain():
    runtime = FakeRuntime()
    store = FakeStoreOwner(runtime)
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    completion = backend.put_paged_batch(
        [b"n" * 16], 1, [[6, 2]]
    )
    assert completion._watcher is None
    assert runtime.wait_calls == []
    settled = []
    completion.add_done_callback(settled.append)
    completion.wait()
    assert len(runtime.wait_calls) == 1
    assert len(settled) == 1 and settled[0].ok


def test_wait_idle_drains_direct_completion_and_runs_callback():
    runtime = FakeRuntime()
    store = FakeStoreOwner(runtime)
    backend = TuttiDirectBackend(store)
    transfer = DirectTransfer(
        backend, FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    completion = transfer.store_layer([b"o" * 16], 1, [[6, 2]])
    callbacks = []
    completion.add_done_callback(callbacks.append)
    engine = object.__new__(KVEngine)
    engine._transfer = transfer
    engine._inflight = [completion]
    engine._active_read_plan = None
    engine._read_window = None
    engine._write_window = None
    engine._write_reuse_event = None
    engine._store = store
    assert runtime.wait_calls == []
    engine.wait_idle()
    assert len(runtime.wait_calls) == 1
    assert len(callbacks) == 1 and callbacks[0].ok


def test_direct_completion_failure_drain_is_idempotent():
    runtime = FakeRuntime(fail=True)
    store = FakeStoreOwner(runtime)
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    completion = backend.put_paged_batch([b"f" * 16], 0, [[3, 1]])
    with pytest.raises(RuntimeError, match="失败"):
        completion.wait()
    assert len(runtime.wait_calls) == 1
    assert runtime.release_calls == [101]
    with pytest.raises(RuntimeError, match="失败"):
        completion.wait()
    assert len(runtime.wait_calls) == 1
    assert runtime.release_calls == [101]
    assert completion.drain_stats["failed"] is True
    backend.close()


def test_direct_write_failure_never_confirms_markers():
    store = FakeStoreOwner(FakeRuntime(fail=True))
    backend = TuttiDirectBackend(store)
    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    completion = backend.put_paged_batch(
        [b"z" * 16], 1, [[2, 4]]
    )
    with pytest.raises(RuntimeError, match="失败"):
        completion.wait()
    assert store.put_results == [
        (False, (derive_io_key(b"z" * 16, 1),))
    ]


@pytest.mark.parametrize(
    ("pool", "reason"),
    [
        (SimpleNamespace(is_cuda=False), "CUDA device tensor"),
        (FakePool(128, base=0x21000), "64 KiB aligned"),
        (FakePool(128, contiguous=False), "not contiguous"),
        (FakePool(128, stride=(12288, 4096, 32, 8, 1)), "stride"),
    ],
)
def test_direct_layout_admission_reasons(pool, reason):
    backend = TuttiDirectBackend(FakeStoreOwner(FakeRuntime()))
    with pytest.raises(DirectAdmissionError, match=reason):
        backend.register_paged_caches(
            pool, num_layers=3, blocks_per_chunk=2,
            chunk_tokens=256, segment_bytes=8192,
            max_chunks_per_wave=2,
        )


@pytest.mark.parametrize("capacity", [0, 6, 7])
def test_direct_capacity_accepts_zero_equal_and_greater(capacity):
    pool = FakePool(128)
    backend = TuttiDirectBackend(
        FakeStoreOwner(FakeRuntime(in_flight=capacity))
    )
    assert backend.register_paged_caches(
        pool, num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192,
        max_chunks_per_wave=2,
    )


def test_direct_capacity_five_rejects_required_six():
    pool = FakePool(128)
    with pytest.raises(
        DirectAdmissionError,
        match=r"configured=5, required=6, num_layers=3",
    ):
        TuttiDirectBackend(
            FakeStoreOwner(FakeRuntime(in_flight=5))
        ).register_paged_caches(
            pool, num_layers=3, blocks_per_chunk=2,
            chunk_tokens=256, segment_bytes=8192,
            max_chunks_per_wave=2,
        )


def test_direct_capacity_shortfall_is_always_loud():
    """容量不足属于准入失败，必须以异常终止——不再静默退回 staged。

    这里曾经是"默认回退、strict 才抛"的分叉，而分叉本身就是缺陷：store 已声明
    直连能力，却因为运行期配额不足悄悄改用 staged（KV 数据经主机内存往返），
    把"直连没生效"伪装成"跑得有点慢"。现在只有一种出口。

    实测触发过：8 卡 TP8 下 in-flight 配额默认 4，而准入要求 2 x 80 层 = 160，
    整个基准跑在 staged 上却只表现为"稍慢"。
    """
    kwargs = dict(
        num_layers=3, blocks_per_chunk=2, chunk_tokens=256,
        segment_bytes=8192, max_chunks_per_wave=2,
    )
    # 保持默认与显式要求 direct，两种配置下的行为必须一致。
    for config in ({}, {"direct_transfer": True}):
        store = FakeStoreOwner(FakeRuntime(in_flight=5))
        with pytest.raises(
            DirectTransferUnavailable,
            match=r"configured=5, required=6, num_layers=3",
        ):
            select_transfer(FakePool(128), store, config, **kwargs)
        # 失败路径不得留下注册痕迹。
        assert store._runtime.register_calls == []


def test_direct_capacity_is_distinct_from_batch_width():
    pool = FakePool(128)
    with pytest.raises(DirectAdmissionError, match="max_batch_entries"):
        TuttiDirectBackend(FakeStoreOwner(FakeRuntime(max_batch=3))).register_paged_caches(
            pool, num_layers=3, blocks_per_chunk=2,
            chunk_tokens=256, segment_bytes=8192,
            max_chunks_per_wave=2,
        )


def test_direct_batch_wider_than_limit_is_not_rejected():
    """批宽超过 runtime 单次提交上限不是准入错误——由提交层切成多次 submit。

    在线实测（TP8 + 2 盘条带 / 每 chunk 4 block / max_batch_entries=8192）：
    某一保存步攒出 2158 个 chunk（8632 entries）、某一读层展开 9248 entries。
    批宽由本步要搬多少 chunk 决定，调用方无法先验保证；曾经在这里抛
    DirectAdmissionError，经 fallback_from_direct 关闭直连绑定后把整个 engine
    拖死。切分见 test_tutti_store.py 的预切/拆半用例。
    """
    backend = TuttiDirectBackend(FakeStoreOwner(FakeRuntime(max_batch=4)))
    assert backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192,
        max_chunks_per_wave=2,
    )
    # 5 张表 = 10 entries > max_batch=4：照常受理。
    tables = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 1]]
    assert len(backend.validate_block_tables(tables)) == len(tables)

    # 登记时未声明波宽也一样（宽度不再参与准入判定）。
    no_wave = TuttiDirectBackend(FakeStoreOwner(FakeRuntime(max_batch=4)))
    assert no_wave.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192,
    )
    assert len(no_wave.validate_block_tables([[0, 1], [2, 3], [4, 5]])) == 3


def test_direct_admission_failure_reason_is_deterministic():
    """准入失败的报错必须逐字稳定——它是排查"直连为何没生效"的唯一线索。

    与容量不足同理：store 有直连能力即只有抛出一条路，不再因配置不同而分叉。
    """
    pool = FakePool(128, base=0x21000)
    kwargs = dict(
        num_layers=3,
        blocks_per_chunk=2,
        chunk_tokens=256,
        segment_bytes=8192,
        max_chunks_per_wave=2,
    )
    for config in ({}, {"direct_transfer": True}):
        with pytest.raises(DirectTransferUnavailable, match="64 KiB aligned"):
            select_transfer(
                pool, FakeStoreOwner(FakeRuntime()), config, **kwargs
            )


def test_direct_is_default_for_an_eligible_tutti_store():
    store = FakeStoreOwner(FakeRuntime())
    transfer = select_transfer(
        FakePool(128), store, {}, num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192, max_chunks_per_wave=2,
    )
    assert isinstance(transfer, DirectTransfer)
    assert len(store._runtime.register_calls) == 1
    transfer.close()


def test_engine_direct_bind_and_close_have_no_staging_lifecycle():
    runtime = FakeRuntime()
    store = FakeEngineStore(runtime)
    engine = KVEngine(
        {
            "chunk_tokens": 256,
            "chunk_kv_bytes": 3 * 8192,
            "max_chunks_per_wave": 2,
            "num_layers": 3,
        },
        store,
    )
    assert engine.try_bind_direct(FakePool(128), 3, 2)
    assert engine.direct
    assert engine._read_window is engine._write_window is None
    assert engine._staging_buffer_id is None
    assert store.layer_span == 3
    engine.close()
    assert runtime.unregister_calls == [41]
    assert store.closed


def test_direct_engine_close_drains_before_target_and_memory_close():
    runtime = FakeRuntime()
    store = FakeEngineStore(runtime)
    engine = KVEngine(
        {
            "chunk_tokens": 256,
            "chunk_kv_bytes": 3 * 8192,
            "max_chunks_per_wave": 2,
            "num_layers": 3,
        },
        store,
    )
    assert engine.try_bind_direct(FakePool(128), 3, 2)
    completion = engine.store_layer([b"t" * 16], 0, [[3, 1]])
    assert runtime.wait_calls == []
    engine.close()
    assert runtime.wait_calls == [(101, 1000)]
    assert runtime.release_calls == [101]
    assert runtime.events.index("release_io") < runtime.events.index("close_batch")
    assert runtime.events.index("close_batch") < runtime.events.index("unregister_memory")


def test_engine_direct_strict_preserves_admission_reason():
    store = FakeEngineStore(FakeRuntime())
    engine = KVEngine(
        {
            "chunk_tokens": 256,
            "chunk_kv_bytes": 3 * 8192,
            "max_chunks_per_wave": 2,
            "num_layers": 3,
        },
        store,
    )
    with pytest.raises(DirectTransferUnavailable, match="64 KiB aligned"):
        engine.try_bind_direct(FakePool(128, base=0x21000), 3, 2)


def test_direct_close_failure_preserves_registration_for_retry():
    runtime = FakeRuntime(unregister_error=RuntimeError("memory busy"))
    backend = TuttiDirectBackend(FakeStoreOwner(runtime))
    pool = FakePool(128)
    backend.register_paged_caches(
        pool, num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192,
        max_chunks_per_wave=2,
    )
    with pytest.raises(RuntimeError, match="memory busy"):
        backend.close()
    assert backend._memory_ticket == 41
    assert backend._pool is pool
    assert not backend._closed
    runtime._unregister_error = None
    backend.close()
    backend.close()
    assert runtime.unregister_calls == [41]


def test_direct_rejection_releases_registered_pool_before_raising():
    """拒绝直连后先解除 runtime 注册、再抛出——不再静默换路径。

    解除注册与"是否回退到 staged"无关：直连绑定已建立、池已注册，放弃时若不解除
    就会留下悬挂注册。以前它配合 staged 回退，现在只配合异常。
    """
    runtime = FakeRuntime()
    store = FakeEngineStore(runtime)
    engine = KVEngine(
        {
            "chunk_tokens": 256,
            "chunk_kv_bytes": 3 * 8192,
            "max_chunks_per_wave": 2,
            "num_layers": 3,
        },
        store,
    )
    assert engine.try_bind_direct(FakePool(128), 3, 2)
    with pytest.raises(DirectTransferUnavailable, match="late plan rejection"):
        engine.fallback_from_direct(DirectAdmissionError("late plan rejection"))
    # 无悬挂注册，且引擎不再声称自己是直连。
    assert runtime.unregister_calls == [41]
    assert not engine.direct


def test_direct_rejects_padded_geometry_and_invalid_block_tables():
    store = FakeStoreOwner(FakeRuntime())
    backend = TuttiDirectBackend(store)
    with pytest.raises(DirectAdmissionError, match="padded or mismatched"):
        backend.register_paged_caches(
            FakePool(128), num_layers=3, blocks_per_chunk=2,
            chunk_tokens=256, segment_bytes=12288,
            max_chunks_per_wave=2,
        )

    backend.register_paged_caches(
        FakePool(128), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192,
        max_chunks_per_wave=2,
    )
    key = b"g" * 16
    store._live.add(derive_io_key(key, 0))
    with pytest.raises(DirectAdmissionError, match="block table length"):
        backend.get_paged_batch([key], 0, [[1]])
    with pytest.raises(DirectAdmissionError, match="outside"):
        backend.get_paged_batch([key], 0, [[1, 99]])
    assert runtime_submit_count(store) == 0


def runtime_submit_count(store):
    return len(store._runtime.submit_calls)


class FakeCompletion:
    def __init__(self, log, layer):
        self.log = log
        self.layer = layer
        self.wait_count = 0
        self._callbacks = []

    def wait(self):
        self.wait_count += 1

    def query(self):
        return True

    def add_terminal_callback(self, callback):
        self._callbacks.append(callback)

    def finish(self, ok):
        result = SimpleNamespace(ok=ok)
        for callback in self._callbacks:
            callback(result)


class EventBackend:
    def __init__(self, log):
        self.log = log

    def register_paged_caches(self, pool, **kwargs):
        return True

    def get_paged_batch(self, keys, layer, blocks):
        self.log.append(("read_submit", layer))
        return FakeCompletion(self.log, layer)

    def put_paged_batch(self, keys, layer, blocks):
        self.log.append(("write_submit", layer))
        return FakeCompletion(self.log, layer)


class EventStore:
    def __init__(self, log):
        self.log = log
        self._read_stream_obj = object()

    def record_read_event(self, event=None):
        self.log.append(("read_record", event))
        return event

    def record_compute_event(self, event=None):
        event = event or object()
        self.log.append(("compute_record", event))
        return event

    def wait_write_event(self, event):
        self.log.append(("write_wait", event))

    def wait_compute_event(self, event):
        self.log.append(("compute_wait", event))


def test_direct_plan_keeps_backend_entity_fence(monkeypatch):
    log = []
    engine = _direct_engine(log)
    ready = object()
    def load(keys, layer, tables):
        completion = FakeCompletion(log, layer)
        completion.fence_event = ready
        return completion
    monkeypatch.setattr(engine._transfer._backend, "get_paged_batch", load)
    plan = _DirectAllLayerReadPlan(engine, [b"f" * 16], [[1, 2]], (0,))
    assert plan.fence_event(0) is ready
    assert plan.handles[0].fence_event is ready
    assert not any(item[0] == "read_record" for item in log)


def _direct_engine(log):
    backend = EventBackend(log)
    transfer = DirectTransfer(
        backend, object(), num_layers=3, blocks_per_chunk=2,
        chunk_tokens=256, segment_bytes=8192,
    )
    engine = object.__new__(KVEngine)
    engine._closed = False
    engine._num_layers = 3
    engine._transfer = transfer
    engine._store = EventStore(log)
    engine._inflight = []
    engine._max_in_flight_operations = 0
    engine._direct_submit_lock = threading.RLock()
    engine._foreground_read_submit_pending = 0
    engine._active_read_plan = None
    engine.begin_prefill_feeder(False)
    return engine


def _finish_direct_plan(plan):
    plan.join_feeder()


def _gated_load_layer(engine, layer_idx, gate, entered=None):
    """包装 engine.load_layer：命中 layer_idx 时先置 entered 再等 gate。"""
    original = engine.load_layer

    def gated(keys, layer, blocks, **kwargs):
        if layer == layer_idx:
            if entered is not None:
                entered.set()
            gate.wait(30.0)
        return original(keys, layer, blocks, **kwargs)

    return gated


def test_staged_read_plan_submission_behavior_is_unchanged(monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _ReadPlan(
        engine, [b"e" * 16], [[2, 7]], (0, 1, 2), 1
    )
    assert [item[0] for item in log] == [
        "read_submit", "read_record",
        "read_submit", "read_record",
        "read_submit", "read_record",
    ]
    before = list(log)
    assert plan.wait_layer(0) is not None
    assert log == before
    assert all(handle.wait_count == 0 for handle in plan._handles.values())


@pytest.mark.parametrize("independent", [False, True])
def test_direct_read_lock_does_not_block_write_for_independent_backend(independent):
    import threading
    log = []
    engine = _direct_engine(log)
    engine._direct_submit_lock = threading.RLock()
    engine._direct_read_lock = threading.RLock()
    backend = engine._transfer._backend
    backend.independent_read_write = independent
    entered, release, written = threading.Event(), threading.Event(), threading.Event()
    errors = []
    original = backend.get_paged_batch

    def read(keys, layer, blocks):
        entered.set()
        assert release.wait(3)
        return original(keys, layer, blocks)

    def run(call):
        try:
            call()
        except Exception as exc:
            errors.append(exc)

    backend.get_paged_batch = read
    reader = threading.Thread(target=run, args=(lambda: engine.load_layer([b"r" * 16], 0, [[2, 7]]),))
    writer = threading.Thread(target=run, args=(lambda: (
        engine.store_layer([b"w" * 16], 0, [[3, 8]]), written.set()),))
    reader.start()
    try:
        assert entered.wait(2)
        writer.start()
        assert written.wait(0.2) is independent
    finally:
        release.set()
        reader.join(3)
        if writer.ident is not None:
            writer.join(3)
    assert not errors and not reader.is_alive() and not writer.is_alive()
    assert written.is_set()


def test_feeder_outer_lock_is_independent_for_replica_backend(monkeypatch):
    import threading
    engine = _direct_engine([])
    engine._transfer._backend.independent_read_write = True
    engine._direct_submit_lock = threading.RLock()
    engine._direct_read_lock = threading.RLock()
    entered, release, written = threading.Event(), threading.Event(), threading.Event()
    original = engine._transfer._backend.get_paged_batch

    def read(keys, layer, blocks):
        if layer == 1:
            entered.set()
            assert release.wait(3)
        return original(keys, layer, blocks)

    engine._transfer._backend.get_paged_batch = read
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(engine, [b"r" * 16], [[2, 7]], (0, 1, 2))
    writer = threading.Thread(target=lambda: (
        engine.store_layer([b"w" * 16], 0, [[3, 8]]), written.set()))
    try:
        assert entered.wait(2)
        writer.start()
        assert written.wait(1), "feeder peer wait still holds the write lock"
    finally:
        release.set()
        plan.join_feeder()
        if writer.ident is not None:
            writer.join(3)
    assert plan.failed is None


def test_latest_read_event_does_not_wait_for_feeder_submission(monkeypatch):
    import threading
    engine = _direct_engine([])
    entered, release, queried = threading.Event(), threading.Event(), threading.Event()
    original = engine._transfer._backend.get_paged_batch

    def read(keys, layer, blocks):
        if layer == 1:
            entered.set()
            assert release.wait(3)
        return original(keys, layer, blocks)

    engine._transfer._backend.get_paged_batch = read
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(engine, [b"r" * 16], [[2, 7]], (0, 1, 2))
    result = []
    reader = threading.Thread(target=lambda: (
        result.append(plan.latest_read_event()), result.append(plan.all_io_done()), queried.set()))
    try:
        assert entered.wait(2)
        reader.start()
        assert queried.wait(1), "read-tail snapshot blocked on the feeder's long submission lock"
        assert result == [plan.read_ready_events[0], False]
    finally:
        release.set()
        plan.join_feeder()
        if reader.ident is not None:
            reader.join(3)


def test_direct_all_layer_constructor_submits_complete_order(monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"r" * 16], [[2, 7]], (0, 1, 2)
    )
    # 第 0 层必须在构造函数返回前提交完成（start_load_kv 快速返回的前提）；
    # 余下层由 feeder 并发补上，因此这里只校验"首个提交是第 0 层"。
    submits = [item for item in log if item[0] == "read_submit"]
    assert submits[0] == ("read_submit", 0)
    assert plan.read_ready_events.get(0) is not None
    _finish_direct_plan(plan)
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0), ("read_submit", 1), ("read_submit", 2)
    ]
    assert [item[0] for item in log] == [
        "read_submit", "read_record",
        "read_submit", "read_record",
        "read_submit", "read_record",
    ]
    assert plan.next_read_to_submit == 3
    assert set(plan.handles) == {0, 1, 2}
    assert set(plan.read_ready_events) == {0, 1, 2}


def test_replica_collectives_use_persistent_service_and_callbacks_only_wait(monkeypatch):
    log = []
    engine = _direct_engine(log)
    engine._transfer._backend.requires_ordered_collectives = True
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"replica" * 4], [[2, 7]], (0, 1, 2)
    )
    assert plan._feeder_thread is not None
    plan._feeder_thread.join(3)
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0), ("read_submit", 1), ("read_submit", 2)
    ]
    plan.wait_layer(0, 0)
    plan.wait_layer(1, 1)
    before = list(log)
    plan.wait_layer(1, 1)
    assert log == before
    plan.wait_layer(2, 2)
    plan.require_complete()
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0), ("read_submit", 1), ("read_submit", 2)
    ]


def test_owned_prefetch_starts_before_first_collective_and_abort_stops_it(monkeypatch):
    log = []
    engine = _direct_engine(log)
    backend = engine._transfer._backend
    backend.requires_ordered_collectives = True
    backend.begin_prefetch = lambda keys, blocks, layers: log.append(("prefetch", tuple(layers)))
    backend.join_prefetch = lambda: log.append(("join_prefetch",))
    backend.stop_prefetch = lambda: log.append(("stop_prefetch",))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(engine, [b"r" * 32], [[2, 7]], (0, 1, 2))
    assert log[:2] == [("prefetch", (0, 1, 2)), ("read_submit", 0)]
    plan.abort()
    assert ("stop_prefetch",) in log
    assert ("join_prefetch",) in log
    assert not any(item == ("read_submit", 1) for item in log)


def test_ordinary_nonblocking_foreground_first_layer_runs_on_persistent_service(monkeypatch):
    from types import SimpleNamespace as NS
    log = []
    engine = _direct_engine(log)
    backend = engine._transfer._backend
    caller = threading.get_ident()
    attempts = []
    release = threading.Event()
    def step(keys, layer, tables):
        attempts.append((layer, threading.get_ident()))
        if not release.is_set():
            return None
        completion = backend.get_paged_batch(keys, layer, tables)
        completion.fence_event = NS(query=lambda: True)
        return completion
    monkeypatch.setattr(backend, "try_get_paged_batch", step, raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    plan = _DirectAllLayerReadPlan(engine, [b"ordinary" * 2], [[2, 7]], (0, 1, 2))
    try:
        _wait_until(lambda: bool(attempts))
        assert all(tid != caller for _, tid in attempts)
        release.set()
        engine._get_read_service().wake()
        plan.join_feeder()
        assert plan.failed is None
        assert [item[1] for item in log if item[0] == "read_submit"] == [0, 1, 2]
    finally:
        release.set()
        plan.abort()
        engine._get_read_service().close()


def test_ordinary_async_capacity_retry_yields_and_waits_for_gpu_batch_fence(monkeypatch):
    from types import SimpleNamespace as NS
    from tutti.engine.core import _AsyncReadPlan
    engine, calls, completed, ended = _async_layer_engine(monkeypatch, layers=2)
    backend = engine._transfer._backend
    original = engine._transfer.load_layer
    capacity = threading.Event()
    cq = NS(active=False)
    polls = []
    fence = NS(done=False)
    def step(keys, layer, tables, target_plan=None):
        if layer == 0 and not capacity.is_set():
            cq.active = True
            return None
        result = original(keys, layer, tables, target_plan)
        result.fence_event = NS(query=lambda: fence.done)
        return result
    def partial(plan):
        polls.append(plan)
        cq.active = False
    backend.try_get_paged_batch = step
    backend.poll_read_plan = partial
    backend.has_pending_read_io = lambda plan: cq.active
    engine._transfer.try_load_layer = step
    plan = _AsyncReadPlan(engine, [b"queued"], [[7]], (0, 1))
    ran = threading.Event()
    class Foreground:
        def advance(self):
            ran.set()
            return True
    try:
        _wait_until(lambda: cq.active)
        job = engine._get_read_service().submit(Foreground(), priority=0)
        assert ran.wait(2)
        engine.end_prefill_feeder()
        engine._get_read_service().wake()
        _wait_until(lambda: bool(polls) and not cq.active)
        assert calls == []
        capacity.set()
        engine.begin_prefill_feeder(True)
        engine.advance_prefill_feeder(0)
        _wait_until(lambda: len(calls) == 1)
        completed[0].done = True
        assert not plan._feeder_done.wait(.03)
        assert len(calls) == 1
        fence.done = True
        engine._get_read_service().wake()
        _wait_until(lambda: len(calls) == 2)
        completed[1].done = True
        plan._feeder.join(2)
        assert plan.poll() is True
    finally:
        plan.abort()
        engine._get_read_service().close()


def _queued_write_engine(monkeypatch):
    from types import SimpleNamespace as NS
    engine = _direct_engine([])
    backend = engine._transfer._backend
    submitted, records, waits = [], [], []
    ready = NS(front=False, write=False)
    front = NS(failed=None, all_io_done=lambda: ready.front)
    engine._active_read_plan = front
    compute = object()
    engine._store.record_compute_event = lambda: records.append(threading.get_ident()) or compute
    engine._store.wait_write_event = lambda event: waits.append((event, threading.get_ident()))
    class Completion:
        def poll(self):
            return True if ready.write else None
    def submit(keys, layer, tables):
        submitted.append((layer, tuple(keys), tuple(tuple(t) for t in tables), threading.get_ident()))
        return Completion()
    backend.try_put_paged_batch = submit
    backend.poll_write_plan = lambda layer_idx=None: None
    backend.has_pending_write_io = lambda layer_idx=None: False
    backend.drain_write_layer = lambda layer: None
    return NS(engine=engine, backend=backend, submitted=submitted, records=records,
              waits=waits, ready=ready, front=front, compute=compute)


def test_high_priority_reads_finish_before_writes_but_low_priority_reads_can_mix(monkeypatch):
    cell = _queued_write_engine(monkeypatch)
    main = threading.get_ident()
    keys, tables = [b"write" * 4], [[2, 7]]
    handle = cell.engine.store_layer(keys, 0, tables)
    keys[0] = b"changed"
    tables[0][0] = 99
    low_started = threading.Event()
    low_release = threading.Event()
    class LowRead:
        def advance(self):
            low_started.set()
            return low_release.is_set()
    low_job = cell.engine._get_read_service().submit(LowRead(), priority=1)
    try:
        assert low_started.wait(2)
        assert cell.records == [main]
        assert cell.submitted == [] and cell.waits == []
        assert handle.poll() is None
        cell.ready.front = True
        _wait_until(lambda: bool(cell.submitted))
        assert low_job.is_alive(), "write must not wait for low priority read to finish"
        assert cell.submitted[0][:3] == (0, (b"write" * 4,), ((2, 7),))
        assert cell.submitted[0][3] != main
        assert cell.waits == [(cell.compute, cell.submitted[0][3])]
        cell.ready.write = True
        handle.wait(2)
        assert handle.poll() is True
    finally:
        low_release.set()
        cell.ready.front = cell.ready.write = True
        cell.engine._get_read_service().wake()
        handle.wait(2)
        low_job.join(2)
        cell.engine._get_read_service().close()


def test_wait_reads_idle_settles_reads_without_waiting_for_queued_writes(monkeypatch):
    cell = _queued_write_engine(monkeypatch)
    read_waited = threading.Event()
    class ReadCompletion:
        def wait(self):
            read_waited.set()
    cell.front.join_feeder = lambda: None
    cell.engine._read_window = None
    handle = cell.engine.store_layer([b"queued" * 3], 0, [[2, 7]])
    cell.engine._inflight.insert(0, ReadCompletion())
    try:
        cell.engine.wait_reads_idle()
        assert read_waited.is_set()
        assert cell.engine._inflight == [handle]
        assert handle.job.is_alive()
        assert cell.submitted == []
        cell.ready.front = True
        _wait_until(lambda: bool(cell.submitted))
        cell.ready.write = True
        handle.wait(2)
    finally:
        cell.ready.front = cell.ready.write = True
        cell.engine._get_read_service().wake()
        handle.wait(2)
        cell.engine._get_read_service().close()


def test_wait_reads_idle_preserves_direct_writes_and_reports_partial_read_failure(monkeypatch):
    cell = _queued_write_engine(monkeypatch)
    calls = []
    write = SimpleNamespace(drain_stats={"direction": "write"},
                            wait=lambda: pytest.fail("write drain must follow consensus"))
    error = RuntimeError("partial read CQ failed")
    def fail_read():
        calls.append("read")
        raise error
    read = SimpleNamespace(wait=fail_read)
    cell.front.join_feeder = lambda: None
    cell.engine._read_window = None
    cell.engine._inflight = [read, write]
    cell.engine._store.drain_deferred_completions = lambda direction: calls.append(direction)
    cell.backend.end_target_plan = lambda direction: calls.append(("end", direction))
    with pytest.raises(RuntimeError, match="partial read CQ failed"):
        cell.engine.wait_reads_idle()
    assert calls == ["read", "read", ("end", "read")]
    assert cell.engine._inflight == [write, read]


def test_wait_idle_does_not_hold_submission_lock_while_queued_writes_finish(monkeypatch):
    cell = _queued_write_engine(monkeypatch)
    joined = threading.Event()
    errors = []
    cell.front.join_feeder = lambda: joined.set()
    cell.engine._read_window = cell.engine._write_window = None
    cell.engine.store_layer([b"queued" * 3], 0, [[2, 7]])
    def drain():
        try:
            cell.engine.wait_idle()
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    try:
        assert joined.wait(2)
        assert cell.submitted == []
        cell.ready.front = cell.ready.write = True
        cell.engine._get_read_service().wake()
        thread.join(2)
        assert not thread.is_alive() and errors == []
        assert len(cell.submitted) == 1 and cell.engine._inflight == []
    finally:
        cell.ready.front = cell.ready.write = True
        cell.engine._get_read_service().wake()
        thread.join(2)
        cell.engine._get_read_service().close()


def test_write_capacity_retry_yields_and_each_retry_obeys_foreground_gate(monkeypatch):
    cell = _queued_write_engine(monkeypatch)
    capacity = threading.Event()
    attempts = []
    original = cell.backend.try_put_paged_batch
    def submit(keys, layer, tables):
        attempts.append(layer)
        return original(keys, layer, tables) if capacity.is_set() else None
    cell.backend.try_put_paged_batch = submit
    cell.ready.front = True
    handle = cell.engine.store_layer([b"queued" * 3], 1, [[2, 7]])
    foreground_ran = threading.Event()
    class Foreground:
        def advance(self):
            foreground_ran.set()
            return True
    try:
        _wait_until(lambda: bool(attempts))
        cell.ready.front = False
        job = cell.engine._get_read_service().submit(Foreground(), priority=0)
        assert foreground_ran.wait(2)
        count = len(attempts)
        handle.job.join(.03)
        assert len(attempts) == count and cell.submitted == []
        capacity.set()
        cell.ready.front = True
        cell.engine._get_read_service().wake()
        _wait_until(lambda: bool(cell.submitted))
        cell.ready.write = True
        handle.wait(2)
        assert len(cell.waits) == 1
    finally:
        cell.ready.front = cell.ready.write = True
        capacity.set()
        cell.engine._get_read_service().wake()
        handle.wait(2)
        cell.engine._get_read_service().close()


def test_read_failure_cancels_unsubmitted_write_and_cleanup_never_blocks_feeder(monkeypatch):
    cell = _queued_write_engine(monkeypatch)
    entered, release, foreground_ran = threading.Event(), threading.Event(), threading.Event()
    def drain(layer):
        entered.set()
        assert release.wait(2)
    cell.backend.drain_write_layer = drain
    handle = cell.engine.store_layer([b"failed" * 3], 0, [[2, 7]])
    class Foreground:
        def advance(self):
            foreground_ran.set()
            return True
    try:
        cell.front.failed = RuntimeError("front IO failed")
        cell.engine._get_read_service().wake()
        assert entered.wait(2)
        cell.engine._get_read_service().submit(Foreground(), priority=0)
        assert foreground_ran.wait(2)
        assert handle.poll() is None and cell.submitted == []
        release.set()
        with pytest.raises(RuntimeError, match="front IO failed"):
            handle.wait(2)
        assert handle.poll() is False
    finally:
        release.set()
        handle.job.join(2)
        cell.engine._get_read_service().close()


def test_direct_foreground_gate_joins_every_exact_layer_event(monkeypatch):
    from types import SimpleNamespace as NS
    log = []
    engine = _direct_engine(log)
    service = engine._get_read_service()
    joined = []
    done = NS(ready=False, queries=0)
    def query():
        done.queries += 1
        return done.ready
    done.query = query
    monkeypatch.setattr(service, "join_events", lambda events: joined.append(events) or done)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(engine, [b"gate" * 4], [[2, 7]], (0, 1, 2))
    plan.join_feeder()
    assert joined == [tuple(plan.read_ready_events[index] for index in range(3))]
    assert not plan.all_io_done() and done.queries == 1
    done.ready = True
    assert plan.all_io_done() and done.queries == 2
    assert plan.all_io_done() and done.queries == 2
    plan.abort()


def test_direct_feeder_submits_without_kick_and_join_is_idempotent(monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"feed" * 4], [[2, 7]], (0, 1, 2)
    )
    # feeder 不等 compute 回调：构造函数返回后即可自行把余下层提交完。
    plan.join_feeder()
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0), ("read_submit", 1), ("read_submit", 2)
    ]
    plan.join_feeder()
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0), ("read_submit", 1), ("read_submit", 2)
    ]
    plan.abort()


def test_direct_feeder_failure_is_recorded_asynchronously(monkeypatch):
    log = []
    engine = _direct_engine(log)
    original = engine.load_layer

    def fail_second(keys, layer_idx, blocks, **kwargs):
        if layer_idx == 1:
            raise RuntimeError("fake feeder admission failure")
        return original(keys, layer_idx, blocks, **kwargs)

    engine.load_layer = fail_second
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"fail" * 4], [[2, 7]], (0, 1, 2)
    )
    assert [item for item in log if item[0] == "read_submit"][0] == (
        "read_submit", 0
    )
    plan.join_feeder()
    assert isinstance(plan.failed, RuntimeError)
    assert "fake feeder admission failure" in str(plan.failed)
    plan.abort()


def test_direct_all_layer_plan_logs_submit_timing(caplog, monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    # 逐层提交时间线是 debug 级诊断，需显式放开该 logger 的级别。
    with caplog.at_level("DEBUG", logger="tutti.engine.core"):
        plan = _DirectAllLayerReadPlan(
            engine, [b"m" * 16], [[2, 7]], (0, 1, 2)
        )
        _finish_direct_plan(plan)
    assert plan._submit_all_total_ms >= 0
    assert len(plan._layer_submit_ms) == 3
    assert plan._first_read_submit_completed_ns is not None
    ready = [
        record.message for record in caplog.records
        if record.message.startswith("DIRECT_READ_PLAN_READY")
    ]
    layer_records = [
        record.message for record in caplog.records
        if record.message.startswith("DIRECT_READ_LAYER_SUBMIT")
    ]
    assert len(ready) == 1
    assert "layers=3" in ready[0]
    assert "total_ms=" in ready[0] and "max_layer_ms=" in ready[0]
    assert len(layer_records) == 3


def test_engine_selects_all_layer_plan_only_for_direct(monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = engine.start_read_plan(
        [b"q" * 16], [[2, 7]], (0, 1, 2), depth=99
    )
    assert isinstance(plan, _DirectAllLayerReadPlan)
    assert plan.read_ready_events.get(0) is not None
    _finish_direct_plan(plan)
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0), ("read_submit", 1), ("read_submit", 2)
    ]


def test_direct_all_layer_wait_is_submit_and_host_wait_free(monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"s" * 16], [[2, 7]], (0, 1, 2)
    )
    _finish_direct_plan(plan)
    before = list(log)
    assert plan.wait_layer(0, 0) is plan.read_ready_events[0]
    assert log == before
    assert plan.handles[0].wait_count == 0
    plan.abort()


def test_direct_all_layer_after_layer_does_not_submit_or_duplicate(monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"t" * 16], [[2, 7]], (0, 1, 2)
    )
    _finish_direct_plan(plan)
    assert len([item for item in log if item[0] == "read_submit"]) == 3
    plan.after_layer(0, 0)
    assert len([item for item in log if item[0] == "read_submit"]) == 3
    plan.after_layer(0, 0)
    assert len([item for item in log if item[0] == "read_submit"]) == 3
    plan.wait_layer(0, 0)
    plan.wait_layer(1, 1)
    plan.after_layer(1, 1)
    plan.wait_layer(2, 2)
    plan.after_layer(2, 2)
    plan.after_layer(2, 2)
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0), ("read_submit", 1), ("read_submit", 2)
    ]
    plan.require_complete()


def test_direct_all_layer_wait_blocks_until_feeder_registers(monkeypatch):
    """feeder 尚未提交该层时，compute 必须等 fence，而不是判定失败。"""
    log = []
    engine = _direct_engine(log)
    gate = threading.Event()
    entered = threading.Event()
    engine.load_layer = _gated_load_layer(engine, 1, gate, entered)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"w" * 16], [[2, 7]], (0, 1, 2)
    )
    assert plan.wait_layer(0, 0) is plan.read_ready_events[0]
    assert entered.wait(5.0), "feeder 应已进入第 1 层提交"
    outcome = {}

    def waiter():
        outcome["event"] = plan.wait_layer(1, 1)

    thread = threading.Thread(target=waiter)
    thread.start()
    thread.join(0.3)
    assert thread.is_alive(), "feeder 未登记时 wait_layer 应等待而非失败"
    assert plan.failed is None
    gate.set()
    thread.join(5.0)
    assert not thread.is_alive()
    assert outcome["event"] is plan.read_ready_events[1]
    plan.abort()


def test_direct_all_layer_wait_surfaces_feeder_failure(monkeypatch):
    """feeder 真失败时，等待中的 compute 立刻拿到该错误。"""
    log = []
    engine = _direct_engine(log)
    original = engine.load_layer

    def fail_second(keys, layer, blocks, **kwargs):
        if layer == 1:
            raise RuntimeError("fake feeder admission failure")
        return original(keys, layer, blocks, **kwargs)

    engine.load_layer = fail_second
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"x" * 16], [[2, 7]], (0, 1, 2)
    )
    assert plan.wait_layer(0, 0) is plan.read_ready_events[0]
    with pytest.raises(RuntimeError, match="fake feeder admission failure"):
        plan.wait_layer(1, 1)
    plan.abort()


def test_direct_all_layer_wait_times_out_when_feeder_stalls(monkeypatch):
    """feeder 卡死（既未登记也未报错）时，等待以显式错误收敛，不永久挂起。"""
    log = []
    engine = _direct_engine(log)
    gate = threading.Event()
    engine.load_layer = _gated_load_layer(engine, 1, gate)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    monkeypatch.setattr("tutti.engine.core._DIRECT_FEEDER_WAIT_SECONDS", 0.5)
    plan = _DirectAllLayerReadPlan(
        engine, [b"y" * 16], [[2, 7]], (0, 1, 2)
    )
    assert plan.wait_layer(0, 0) is plan.read_ready_events[0]
    with pytest.raises(RuntimeError, match="did not submit layer 1"):
        plan.wait_layer(1, 1)
    gate.set()
    plan.abort()


def test_direct_all_layer_out_of_order_callbacks_fail_closed(monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"u" * 16], [[2, 7]], (0, 1, 2)
    )
    _finish_direct_plan(plan)
    with pytest.raises(RuntimeError, match="out-of-order direct wait callback"):
        plan.wait_layer(1, 1)
    plan.after_layer(0, 0)
    with pytest.raises(RuntimeError, match="incomplete direct all-layer"):
        plan.require_complete()
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0), ("read_submit", 1), ("read_submit", 2)
    ]


def test_direct_all_layer_rejects_wrong_physical_mapping(monkeypatch):
    engine = _direct_engine([])
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"p" * 16], [[2, 7]], (0, 1, 2)
    )
    with pytest.raises(RuntimeError, match="maps to physical 0, got 99"):
        plan.wait_layer(0, 99)
    assert plan.terminal_failure is not None
    plan.abort()


def test_direct_all_layer_read_failure_marks_whole_plan_without_host_wait(
        monkeypatch):
    log = []
    failures = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"h" * 16], [[3, 1]], (0, 1, 2),
        on_failure=failures.append,
    )
    plan.handles[0].finish(False)
    assert plan.failed is failures[0]
    assert plan.failed.whole_operation
    assert plan.failed.invalid_block_ids == (3, 1)
    assert all(handle.wait_count == 0 for handle in plan.handles.values())
    with pytest.raises(RuntimeError, match="direct read failed"):
        plan.wait_layer(0, 0)
    plan.abort()


def test_direct_write_records_compute_then_waits_write_then_submits():
    log = []
    engine = _direct_engine(log)
    engine.store_layer([b"f" * 16], 2, [[4, 1]])
    assert [item[0] for item in log] == [
        "compute_record", "write_wait", "write_submit"
    ]


def _all_layer_worker(engine, plan, *, save_keys):
    worker = WorkerImpl(engine)
    worker._bound = True
    worker._num_layers = 3
    worker._callback_to_physical = (0, 1, 2)
    worker._callback_by_name = {
        f"model.layers.{layer}.self_attn": layer for layer in range(3)
    }
    worker._load_keys = [b"v" * 16]
    worker._load_block_tables = [[2, 7]]
    worker._read_plan = plan
    worker._save_keys = save_keys
    worker._save_block_tables = [[2, 7]] if save_keys else []
    worker._save_generations = ["g"] if save_keys else []
    return worker


def test_worker_defers_write_stream_behind_latest_read_event(monkeypatch):
    """开启写延后时，写流被设备侧排到"最新已提交读层"之后。

    机制：worker 每次 save 前取读计划最新读层的 fence 事件，在写流上插一次
    ``wait_event``（同一事件不重复插）——读与写因此不混跑（NVMe 混跑实测读
    +52%、写 +31%）。Python 不做缓冲、不轮询，顺序完全由设备侧保证。
    """
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())

    waits = []
    store = engine._store          # 复用假 store，只加延后相关的钩子
    store.defer_writes_after_reads = True
    store.wait_write_stream_event = waits.append
    plan = _DirectAllLayerReadPlan(
        engine, [b"v" * 16], [[2, 7]], (0, 1, 2)
    )
    worker = _all_layer_worker(engine, plan, save_keys=[b"w" * 16])
    plan.join_feeder()
    latest = plan.latest_read_event()
    assert latest is not None

    worker.save_kv_layer("model.layers.0.self_attn")
    assert waits == [latest], "写流应排到最新读层之后"

    # 同一个事件不重复插入（每层一次 wait_event 就够，多插无意义）
    worker.save_kv_layer("model.layers.1.self_attn")
    assert waits == [latest]

    # 关掉开关：不再插入任何设备侧等待
    store.defer_writes_after_reads = False
    worker.save_kv_layer("model.layers.2.self_attn")
    assert waits == [latest]


def test_worker_host_enqueue_order_is_all_read_then_compute_then_write(
        monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"v" * 16], [[2, 7]], (0, 1, 2)
    )
    worker = _all_layer_worker(engine, plan, save_keys=[b"w" * 16])

    worker.wait_for_layer_load("model.layers.0.self_attn")
    log.append(("compute_enqueue", 0))
    worker.save_kv_layer("model.layers.0.self_attn")
    plan.join_feeder()

    names = [item[0] for item in log]
    assert names.index("read_submit") < names.index("compute_wait")
    assert names.index("compute_wait") < names.index("compute_enqueue")
    assert names.index("compute_enqueue") < names.index("compute_record")
    assert names.index("compute_record") < names.index("write_wait") < names.index("write_submit")
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0), ("read_submit", 1), ("read_submit", 2)
    ]


def test_worker_save_without_save_plan_does_not_submit_read(monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"x" * 16], [[2, 7]], (0, 1, 2)
    )
    worker = _all_layer_worker(engine, plan, save_keys=None)
    worker.wait_for_layer_load("model.layers.0.self_attn")
    worker.save_kv_layer("model.layers.0.self_attn")
    worker.save_kv_layer("model.layers.0.self_attn")
    plan.join_feeder()
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0), ("read_submit", 1), ("read_submit", 2)
    ]
    assert not [item for item in log if item[0] == "write_submit"]


def test_select_new_chunks_trims_resident_and_duplicates():
    """写入批裁剪：排除已驻留 chunk 与批内重复，三数组同步切片。"""
    a, b, c = b"a" * 16, b"b" * 16, b"c" * 16
    keys = [a, b, c, a]
    gens = [1, 2, 3, 4]
    tables = [[10], [20], [30], [40]]

    # 全受理（批内重复 a 仍需去重）
    trimmed = WorkerImpl._select_new_chunks(keys, gens, tables, [a, b, c])
    assert trimmed[0] == [a, b, c]
    assert trimmed[1] == [1, 2, 3]          # 取首次出现位置
    assert trimmed[2] == [[10], [20], [30]]

    # 部分已驻留（b 已在盘上 → 不在 new_keys 中）
    trimmed = WorkerImpl._select_new_chunks(keys, gens, tables, [a, c])
    assert trimmed[0] == [a, c]
    assert trimmed[1] == [1, 3]
    assert trimmed[2] == [[10], [30]]

    # 全部已驻留 → 裁剪为空（不写任何 chunk）
    trimmed = WorkerImpl._select_new_chunks(keys, gens, tables, [])
    assert trimmed[0] == []
    assert trimmed[1] == []
    assert trimmed[2] == []

    # 无冗余 → 原样返回（保持原对象，避免热路径无谓拷贝）
    plain = [a, b, c]
    assert WorkerImpl._select_new_chunks(
        plain, gens[:3], tables[:3], plain
    )[0] is plain


def test_worker_write_batch_excludes_resident_chunks():
    """端到端：save span 内的已驻留 chunk 不得进入数据面写入。

    B 请求（外部命中）的 save span 由调度侧按"未保存 token 边界"切出，
    含已驻留 chunk；若不过滤，实测 39 个 chunk 里有 30 个是重复写。
    """
    runtime = FakeRuntime()
    store = FakeEngineStore(runtime)
    engine = KVEngine(
        {
            "chunk_tokens": 256,
            "chunk_kv_bytes": 3 * 8192,
            "max_chunks_per_wave": 4,
            "num_layers": 3,
        },
        store,
    )
    assert engine.try_bind_direct(FakePool(128), 3, 2)

    resident_tokens = list(range(256))
    fresh_tokens = list(range(256, 512))
    resident_key, _ = engine.hash_keys(resident_tokens)

    # 先把第一个 chunk 走完两阶段，使其成为"已驻留"
    plan = engine.plan_store(resident_key)
    assert plan is not None and plan.new_keys == resident_key
    engine.store_layer(resident_key, 0, [[3, 1]])
    engine.wait_idle()
    engine.confirm_store(resident_key)

    worker = WorkerImpl(engine)
    worker._ensure_bound = lambda: None
    worker._chunk_tokens = 256
    worker._block_size = 128
    # block_size=128 + chunk_tokens=256 ⇒ 每 chunk 2 个块（准入会校验）
    worker._metadata = SimpleNamespace(requests=[
        SimpleNamespace(
            token_ids=resident_tokens, save_chunk_start=0, save_chunk_count=1,
            block_ids=[0, 1], save_generations=["g-resident"],
        ),
        SimpleNamespace(
            token_ids=fresh_tokens, save_chunk_start=0, save_chunk_count=1,
            block_ids=[2, 3], save_generations=["g-fresh"],
        ),
    ])

    worker._prepare_write_batch()

    # 只保留未驻留的那个 chunk（已驻留的 39→30 冗余在此消除）
    assert len(worker._save_keys) == 1
    assert worker._save_keys == engine.hash_keys(fresh_tokens)[0]
    assert worker._save_generations == ["g-fresh"]
    assert worker._save_block_tables == [[2, 3]]
    engine.close()


def test_worker_write_capacity_trim_rolls_back_pending_keys(monkeypatch):
    """Physical slot shortfall must not leak semantic pending reservations."""
    engine = _bound_engine()
    tokens = list(range(256))
    key = engine.hash_keys(tokens)[0][0]
    worker = WorkerImpl(engine)
    worker._ensure_bound = lambda: None
    worker._chunk_tokens = 256
    worker._block_size = 128
    worker._metadata = SimpleNamespace(requests=[SimpleNamespace(
        token_ids=tokens, save_chunk_start=0, save_chunk_count=1,
        block_ids=[0, 1], save_generations=["g"],
    )])
    monkeypatch.setattr(engine, "prepare_write_targets", lambda _keys: [])

    worker._prepare_write_batch()

    assert worker._save_keys == []
    assert engine._index.stats()["pending"] == 0
    assert key in worker._store_failed
    engine.close()


def _bound_engine(capacity=4):
    runtime = FakeRuntime()
    store = FakeEngineStore(runtime)
    engine = KVEngine(
        {
            "chunk_tokens": 256,
            "chunk_kv_bytes": 3 * 8192,
            "max_chunks_per_wave": 8,
            "num_layers": 3,
        },
        store,
    )
    assert engine.try_bind_direct(FakePool(128), 3, 2)
    return engine


def test_apply_evictions_drops_data_and_forgets():
    """调度侧下发的驱逐：数据面删除 + 本 rank 索引对齐。"""
    engine = _bound_engine()
    tokens = list(range(256))
    keys, _ = engine.hash_keys(tokens)
    plan = engine.plan_store(keys)
    assert plan is not None and plan.new_keys == keys
    engine.store_layer(keys, 0, [[0, 1]])
    engine.wait_idle()
    engine.confirm_store(keys)
    assert engine._index.is_resident(keys[0])

    assert engine.apply_evictions(keys) == 1
    assert not engine._index.is_resident(keys[0])
    engine.close()


def test_apply_evictions_skips_pinned_keys():
    """在途读取（pin）中的数据不得被删除。"""
    engine = _bound_engine()
    tokens = list(range(256))
    keys, _ = engine.hash_keys(tokens)
    plan = engine.plan_store(keys)
    assert plan is not None
    engine.store_layer(keys, 0, [[0, 1]])
    engine.wait_idle()
    engine.confirm_store(keys)

    engine.pin(keys)
    assert engine.apply_evictions(keys) == 0      # 受保护，跳过
    assert engine._index.is_resident(keys[0])
    engine.unpin(keys)
    assert engine.apply_evictions(keys) == 1      # 解除后即可驱逐
    engine.close()


def test_apply_evictions_is_noop_for_unknown_keys():
    """漂移兜底：调度侧选中的 key 在本 rank 不存在时不得报错。

    此时 worker 仍会自行驱逐腾容量（plan_store 的兜底路径），
    因此不会出现"容量不足导致写入被拒"。
    """
    engine = _bound_engine()
    assert engine.apply_evictions([b"never-here-key!1"]) == 0
    engine.close()


def test_worker_applies_scheduler_evictions_before_planning():
    """worker 在自身写入计划之前执行调度侧驱逐，使两侧索引收敛。"""
    engine = _bound_engine()
    tokens = list(range(256))
    keys, _ = engine.hash_keys(tokens)
    plan = engine.plan_store(keys)
    assert plan is not None
    engine.store_layer(keys, 0, [[0, 1]])
    engine.wait_idle()
    engine.confirm_store(keys)
    assert engine._index.is_resident(keys[0])

    worker = WorkerImpl(engine)
    worker._metadata = SimpleNamespace(
        requests=[], evicted_keys=list(keys),
    )
    assert worker._apply_scheduler_evictions() == 1
    assert not engine._index.is_resident(keys[0])
    engine.close()


def test_worker_eviction_apply_failure_is_non_fatal():
    """驱逐执行失败必须降级（worker 仍会自行驱逐），不能让写路径崩。"""
    class Exploding:
        def apply_evictions(self, keys):
            raise RuntimeError("boom")

    worker = WorkerImpl(Exploding())
    worker._metadata = SimpleNamespace(requests=[], evicted_keys=[b"k" * 16])
    assert worker._apply_scheduler_evictions() == 0


def test_worker_reports_pin_missing_keys_as_forgotten():
    """pin 未遂的 key 必须回传调度侧（幽灵命中自愈的入口）。

    两侧索引是独立 LRU，worker 驱逐后调度侧仍可能报命中；worker 在此
    把本进程判定缺失的 key 记入 forgotten，由 build_connector_worker_meta
    交给调度侧对齐。
    """

    class Engine:
        direct = True
        read_plan_supported = True

        def hash_keys(self, token_ids):
            return [b"missing-key-0001"], b"parent"

        def pin(self, keys):
            raise KeyError([b"missing-key-0001"])

        def start_read_plan(self, *args, **kwargs):
            raise AssertionError("读取未遂不应进入读取计划")

    meta = SimpleNamespace(
        load_tokens=256,
        load_start_token=0,
        token_ids=list(range(256)),
        req_id="r-ghost",
        block_ids=[4, 5],
        save_chunk_start=0,
        save_chunk_count=0,
    )
    worker = WorkerImpl(Engine())
    worker._metadata = SimpleNamespace(requests=[meta])
    worker._chunk_tokens = 256
    worker._max_chunks_per_wave = 2
    worker._block_size = 128
    worker._callback_to_physical = (0, 1, 2)
    worker._ensure_bound = lambda: None
    worker._finalize_load_state = lambda: None
    worker.start_load_kv(None)

    assert worker._index_forgotten == {b"missing-key-0001"}
    worker._settle_unreported_plans = lambda: None
    payload = worker.build_connector_worker_meta()
    assert payload is not None
    assert payload.forgotten == {b"missing-key-0001"}
    # 交出即清空（避免重复上报）
    assert worker._index_forgotten == set()


def test_worker_logs_direct_start_load_return(caplog):
    class Plan:
        failed = None

    class Engine:
        direct = True
        read_plan_supported = True

        def hash_keys(self, token_ids):
            return [b"k" * 16], b"parent"

        def pin(self, keys):
            return None

        def start_read_plan(self, *args, **kwargs):
            return Plan()

    meta = SimpleNamespace(
        load_tokens=256,
        load_start_token=0,
        token_ids=list(range(256)),
        req_id="r0",
        block_ids=[4, 5],
        # start_load_kv 现在同时准备写批（写侧字段是元数据契约的一部分）。
        save_chunk_start=0,
        save_chunk_count=0,
    )
    worker = WorkerImpl(Engine())
    worker._metadata = SimpleNamespace(requests=[meta])
    worker._chunk_tokens = 256
    worker._max_chunks_per_wave = 2
    worker._block_size = 128
    worker._callback_to_physical = (0, 1, 2)
    worker._ensure_bound = lambda: None
    worker._finalize_load_state = lambda: None
    # 该诊断是正常路径日志，级别为 debug（避免每步刷 warning）。
    with caplog.at_level("DEBUG", logger="tutti.integration.vllm.worker"):
        worker.start_load_kv(None)
    records = [
        record for record in caplog.records
        if record.message.startswith("DIRECT_START_LOAD_RETURN")
    ]
    assert len(records) == 1
    assert records[0].levelname == "DEBUG"
    assert "total_ms=" in records[0].message


def test_worker_direct_bind_allocates_no_staging(monkeypatch):
    class Engine:
        max_in_flight_operations = 8

        def try_bind_direct(self, pool, num_layers, blocks_per_chunk):
            self.args = pool, num_layers, blocks_per_chunk
            return True

    engine = Engine()
    worker = WorkerImpl(engine)
    worker.configure(256, 3 * 8192, 2, 128)
    monkeypatch.setattr(
        worker_module.torch, "empty",
        lambda *args, **kwargs: pytest.fail("staging allocation attempted"),
    )
    monkeypatch.setattr(
        worker_module, "RingWindow",
        lambda *args, **kwargs: pytest.fail("RingWindow created"),
    )
    pool = FakePool(128)
    worker.register_cross_layers_kv_cache(pool, attn_backend=object())
    assert engine.args == (pool, 3, 2)
    assert worker.window is None
    assert worker.read_window is None
    assert worker.write_window is None


def test_worker_direct_compute_callback_only_waits_recorded_fence():
    fence = object()

    class Plan:
        failed = None

        def wait_layer(self, callback, physical):
            assert callback == 0
            assert physical == 0
            return fence

    class Store:
        def __init__(self):
            self.waited = []

        def wait_compute_event(self, event):
            self.waited.append(event)

    class Engine:
        direct = True
        max_in_flight_operations = 8

        def __init__(self):
            self._store = Store()

    engine = Engine()
    worker = WorkerImpl(engine)
    worker._num_layers = 1
    worker._callback_to_physical = (0,)
    worker._load_keys = [b"i" * 16]
    worker._read_plan = Plan()
    worker.wait_for_layer_load("model.layers.0.self_attn")
    worker.wait_for_layer_load("model.layers.0.self_attn")
    assert engine._store.waited == [fence]


def test_worker_logs_first_direct_compute_callback(caplog):
    class Plan:
        failed = None

        def wait_layer(self, callback, physical):
            return object()

    class Store:
        def wait_compute_event(self, event):
            return None

    class Engine:
        direct = True

        def __init__(self):
            self._store = Store()

    worker = WorkerImpl(Engine())
    worker._num_layers = 1
    worker._callback_to_physical = (0,)
    worker._load_keys = [b"j" * 16]
    worker._read_plan = Plan()
    worker._direct_start_load_started_ns = 1
    with caplog.at_level("DEBUG", logger="tutti.integration.vllm.worker"):
        worker.wait_for_layer_load("model.layers.0.self_attn")
        worker.wait_for_layer_load("model.layers.0.self_attn")
    messages = [
        record.message for record in caplog.records
        if record.message.startswith("DIRECT_FIRST_COMPUTE_CALLBACK")
    ]
    assert len(messages) == 1
    assert "callback=0" in messages[0]
    assert "since_start_ms=" in messages[0]


def test_worker_fails_before_first_io_for_invalid_direct_plan():
    """block tables 不合规时在提交前即失败——不再改绑 staged。

    要点是"第一次 I/O 之前"：一旦 IO 已经发出再改路径，盘上就会留下半写数据，
    而调用方以为走的是另一条路。所以校验必须前置，且失败只有一种出口。
    """
    class Engine:
        direct = True
        max_in_flight_operations = 8

        def __init__(self):
            self.reason = None

        def validate_direct_block_tables(self, block_tables):
            raise DirectAdmissionError("direct block table length mismatch")

        def fallback_from_direct(self, reason):
            # 与真实 Engine 一致：记录原因后抛出（真实实现还会先解除池注册）。
            self.reason = str(reason)
            raise DirectTransferUnavailable(str(reason)) from reason

    engine = Engine()
    worker = WorkerImpl(engine)
    worker._num_layers = 3
    worker._chunk_tokens = 256
    worker._chunk_kv_bytes = 3 * 8192
    worker._block_size = 128
    # 若 Worker 仍在失败路径上补绑 staged，这里会记录到而不被发现。
    bound = []
    worker._bind_staged = lambda *args: bound.append(args)

    with pytest.raises(DirectTransferUnavailable, match="block table length"):
        worker._validate_direct_or_fail([[1]])
    assert "block table length" in engine.reason
    assert bound == []

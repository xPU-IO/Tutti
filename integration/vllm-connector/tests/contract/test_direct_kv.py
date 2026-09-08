from __future__ import annotations

from types import SimpleNamespace

import pytest

from adapter import worker as worker_module
from adapter.worker import WorkerImpl
from engine.core import KVEngine, _DirectAllLayerReadPlan, _ReadPlan
from engine.transfer import (
    DirectTransfer,
    DirectTransferUnavailable,
    StagedTransfer,
    select_transfer,
)
from index.chunk_index import derive_io_key
from stores.tutti_nvme.store import (
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
    def __init__(self, events=None):
        self.prepared = []
        self.events = events if events is not None else []
        self.generations = {}

    def prepare_put(self, io_keys, capacity):
        self.prepared.append((tuple(io_keys), capacity))
        self.events.append("prepare_put")
        for io_key in io_keys:
            self.generations.setdefault(bytes(io_key[:16]), 1)

    @staticmethod
    def target_uri(chunk_id):
        return f"file:///{bytes(chunk_id).hex()}"

    @staticmethod
    def target_size(chunk_id):
        return 3 * 8192

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
        self._accel_id = 0
        self._execution = "device"
        self._layout = FakeLayout(runtime.events)
        self._num_chunks = 16
        self._live = set()
        self._targets = {}
        self.put_results = []
        self.ensure_targets_calls = 0

    def _runtime_supports_multi_stream(self):
        return True

    def create_direct_transfer(self, pool, **kwargs):
        return TuttiDirectBackend(self)

    def _stream_for(self, direction):
        return {"read": 11, "write": 22}[direction], None

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

    def _submit_retry(self, requests, direction):
        return TuttiKVStore._submit_retry(self, requests, direction)

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
        2 * geometry.segment_bytes + ordinal * geometry.page_bytes
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
    uri = store._layout.target_uri(key)
    store._targets[uri].generation += 1
    with pytest.raises(RuntimeError, match="generation mismatch"):
        backend.get_paged_batch([key], 1, [[3, 1]])
    completion.wait()
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
        runtime=runtime, allocator_enabled=False,
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
    target_path = store._layout.chunk_file(key)
    assert target_path.exists()
    assert target_path.stat().st_size == 3 * 8192
    assert len(runtime.open_batch_calls) == 1
    assert completion._watcher is None
    completion.wait()
    backend.close()
    store.close()


def test_direct_real_store_failure_rolls_back_live_and_layout(tmp_path):
    runtime = FakeRuntime(fail=True)
    store = TuttiKVStore(
        root=tmp_path / "failed-write", num_chunks=2, segment_bytes=8192,
        runtime=runtime, allocator_enabled=False,
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
    key = b"failed-write-key"[:16]
    completion = backend.put_paged_batch([key], 0, [[3, 1]])
    with pytest.raises(RuntimeError, match="失败"):
        completion.wait()
    assert store._live == set()
    assert store._targets == {}
    assert not store._layout.chunk_file(key).exists()
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


def test_direct_capacity_five_falls_back_or_reports_strict_boundary():
    kwargs = dict(
        num_layers=3, blocks_per_chunk=2, chunk_tokens=256,
        segment_bytes=8192, max_chunks_per_wave=2,
    )
    normal_store = FakeStoreOwner(FakeRuntime(in_flight=5))
    transfer = select_transfer(FakePool(128), normal_store, {}, **kwargs)
    assert isinstance(transfer, StagedTransfer)
    assert normal_store._runtime.register_calls == []

    strict_store = FakeStoreOwner(FakeRuntime(in_flight=5))
    with pytest.raises(
        DirectTransferUnavailable,
        match=r"configured=5, required=6, num_layers=3",
    ):
        select_transfer(
            FakePool(128), strict_store,
            {"direct_transfer_strict": True}, **kwargs,
        )
    assert strict_store._runtime.register_calls == []


def test_direct_capacity_is_distinct_from_batch_width():
    pool = FakePool(128)
    with pytest.raises(DirectAdmissionError, match="max_batch_entries"):
        TuttiDirectBackend(FakeStoreOwner(FakeRuntime(max_batch=3))).register_paged_caches(
            pool, num_layers=3, blocks_per_chunk=2,
            chunk_tokens=256, segment_bytes=8192,
            max_chunks_per_wave=2,
        )


def test_direct_normal_fallback_and_strict_reason_are_deterministic():
    pool = FakePool(128, base=0x21000)
    kwargs = dict(
        num_layers=3,
        blocks_per_chunk=2,
        chunk_tokens=256,
        segment_bytes=8192,
        max_chunks_per_wave=2,
    )
    transfer = select_transfer(
        pool, FakeStoreOwner(FakeRuntime()), {}, **kwargs
    )
    assert isinstance(transfer, StagedTransfer)
    with pytest.raises(DirectTransferUnavailable, match="64 KiB aligned"):
        select_transfer(
            pool,
            FakeStoreOwner(FakeRuntime()),
            {"direct_transfer_strict": True},
            **kwargs,
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
            "direct_transfer_strict": True,
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


def test_direct_to_staged_fallback_unregisters_registered_pool():
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
    engine.fallback_from_direct(DirectAdmissionError("late plan rejection"))
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
    return engine


def _finish_direct_plan(plan):
    plan.kick_feeder()
    plan.join_feeder()


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


def test_direct_all_layer_constructor_submits_complete_order(monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"r" * 16], [[2, 7]], (0, 1, 2)
    )
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0)
    ]
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


def test_direct_feeder_starts_after_explicit_kick_and_is_idempotent(monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"feed" * 4], [[2, 7]], (0, 1, 2)
    )
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0)
    ]
    plan.kick_feeder()
    plan.join_feeder()
    plan.kick_feeder()
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
    assert [item for item in log if item[0] == "read_submit"] == [
        ("read_submit", 0)
    ]
    plan.kick_feeder()
    plan.join_feeder()
    assert isinstance(plan.failed, RuntimeError)
    assert "fake feeder admission failure" in str(plan.failed)
    plan.abort()


def test_direct_all_layer_plan_logs_submit_timing(caplog, monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    with caplog.at_level("WARNING", logger="engine.core"):
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
    assert [item for item in log if item[0] == "read_submit"] == [("read_submit", 0)]
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
    before = list(log)
    assert plan.wait_layer(0, 0) is plan.read_ready_events[0]
    assert log == before
    assert plan.handles[0].wait_count == 0
    _finish_direct_plan(plan)


def test_direct_all_layer_after_layer_does_not_submit_or_duplicate(monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"t" * 16], [[2, 7]], (0, 1, 2)
    )
    assert len([item for item in log if item[0] == "read_submit"]) == 1
    _finish_direct_plan(plan)
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


def test_direct_all_layer_out_of_order_and_missing_callbacks_fail_closed(
        monkeypatch):
    log = []
    engine = _direct_engine(log)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.Event", lambda enable_timing=False: object())
    plan = _DirectAllLayerReadPlan(
        engine, [b"u" * 16], [[2, 7]], (0, 1, 2)
    )
    with pytest.raises(RuntimeError, match="has no recorded ready event"):
        plan.wait_layer(1, 1)
    plan.after_layer(0, 0)
    with pytest.raises(RuntimeError, match="incomplete direct all-layer"):
        plan.require_complete()
    assert [item for item in log if item[0] == "read_submit"] == [("read_submit", 0)]


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
    )
    worker = WorkerImpl(Engine())
    worker._metadata = SimpleNamespace(requests=[meta])
    worker._chunk_tokens = 256
    worker._max_chunks_per_wave = 2
    worker._block_size = 128
    worker._callback_to_physical = (0, 1, 2)
    worker._ensure_bound = lambda: None
    worker._finalize_load_state = lambda: None
    with caplog.at_level("WARNING", logger="adapter.worker"):
        worker.start_load_kv(None)
    messages = [
        record.message for record in caplog.records
        if record.message.startswith("DIRECT_START_LOAD_RETURN")
    ]
    assert len(messages) == 1
    assert "total_ms=" in messages[0]


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
    with caplog.at_level("WARNING", logger="adapter.worker"):
        worker.wait_for_layer_load("model.layers.0.self_attn")
        worker.wait_for_layer_load("model.layers.0.self_attn")
    messages = [
        record.message for record in caplog.records
        if record.message.startswith("DIRECT_FIRST_COMPUTE_CALLBACK")
    ]
    assert len(messages) == 1
    assert "callback=0" in messages[0]
    assert "since_start_ms=" in messages[0]


def test_worker_falls_back_before_first_io_for_invalid_direct_plan():
    class Engine:
        direct = True
        max_in_flight_operations = 8

        def validate_direct_block_tables(self, block_tables):
            raise DirectAdmissionError("direct block table length mismatch")

        def fallback_from_direct(self, reason):
            self.direct = False
            self.reason = str(reason)

    engine = Engine()
    worker = WorkerImpl(engine)
    worker._num_layers = 3
    worker._chunk_tokens = 256
    worker._chunk_kv_bytes = 3 * 8192
    worker._block_size = 128
    bound = []
    worker._bind_staged = lambda *args: bound.append(args)
    worker._validate_direct_or_fallback([[1]])
    assert "block table length" in engine.reason
    assert bound == [(3, 8192, 2)]

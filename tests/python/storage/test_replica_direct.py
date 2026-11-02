import pytest

from tutti.storage.tutti_nvme.replica_direct import owner_index, peer_page_rows, shard_indices


def keys(count):
    return [n.to_bytes(8, "little") + b"x" * 24 for n in range(count)]


def test_stable_owners_are_independent_of_request_order():
    original = keys(100)
    for size in (2, 8):
        a = {key: owner_index(key, size) for key in original}
        b = {key: owner_index(key, size) for key in reversed(original)}
        assert a == b
        assert set(a.values()) == set(range(size))


@pytest.mark.parametrize("size", [2, 8])
@pytest.mark.parametrize("count", [0, 1, 7, 17, 1031, 10000])
def test_shards_cover_each_chunk_once_without_capacity_or_batches(size, count):
    items = keys(count)
    shards = shard_indices(items, size)
    assert isinstance(shards, tuple)
    assert len(shards) == size
    seen = []
    for owner, indices in enumerate(shards):
        assert list(indices) == list(range(owner, count, size))
        assert all(owner_index(items[index], size) == owner for index in indices)
        seen.extend(indices)
    assert sorted(seen) == list(range(count))
    if count == 10000:
        assert max(map(len, shards)) > 1031


def test_peer_rows_use_owner_rank_source_and_local_noncontiguous_destination():
    members = (4, 2, 7)
    tables = {
        4: ((1, 9), (4, 17), (6, 23), (12, 28)),
        2: ((11, 2), (5, 17), (9, 23), (18, 30)),
        7: ((19, 20), (21, 22), (25, 31), (32, 37)),
    }
    assert peer_page_rows(keys(4), members, 2, tables) == [
        [0, 1, 11], [0, 9, 2],
        [1, 25, 9], [1, 31, 23],
        [0, 12, 18], [0, 28, 30],
    ]


def test_local_owner_is_not_copied_and_empty_owners_need_no_rows():
    members = (4, 2, 7, 6, 3, 5, 0, 1)
    tables = {rank: ((rank + 10, rank + 30),) for rank in members}
    assert shard_indices(keys(1), 8) == ((0,), (), (), (), (), (), (), ())
    assert peer_page_rows(keys(1), members, 4, tables) == []
    assert peer_page_rows(keys(1), members, 7, tables) == [[0, 14, 17], [0, 34, 37]]
    assert peer_page_rows([], members, 4, {rank: () for rank in members}) == []


def test_overlapping_destination_rejected_even_for_local_owned_chunk():
    with pytest.raises(ValueError, match="overlap"):
        peer_page_rows(keys(2), (0, 1), 0,
                       {0: ((2, 7), (7, 9)), 1: ((12, 17), (18, 19))})


@pytest.mark.parametrize("tables", [
    {0: ((2, 7), (8, 9)), 1: ((12,), (18, 19))},
    {0: ((2,), (8, 9)), 1: ((12, 17), (18, 19))},
    {0: ((2, 7), (8, 9)), 1: ((12, 17),)},
    {0: ((2, 7), (8, 9)), 1: ((12, 17), (17, 19))},
    {0: ((2, 7), (8, 9)), 1: ((-1, 17), (18, 19))},
])
def test_invalid_peer_tables_are_rejected(tables):
    with pytest.raises(ValueError):
        peer_page_rows(keys(2), (0, 1), 0, tables)


def test_plan_exchange_accepts_distinct_rank_block_ids_and_keeps_rank_tables():
    import struct
    from types import SimpleNamespace as NS
    items = keys(2)
    tables = {0: ((11, 2), (5, 17)), 1: ((6, 23), (9, 31))}
    descriptors = []
    for rank in (0, 1):
        control = signals(rank=rank)
        control.begin = lambda epoch, descriptor: descriptors.append(descriptor)
        peer = 1 - rank
        flat = [block for table in tables[peer] for block in table]
        control.store.set(f"3/blocks/{peer}", struct.pack("<4q", *flat))
        backend = _p2p_backend(NS(_replica_groups=((0, 1),)), _rank=rank)
        backend._geometry = NS(blocks_per_chunk=2, segment_bytes=65536)
        backend._signals, backend._epoch = control, 3
        backend._check_plan(items, tables[rank], (0,))
        assert backend._peer_tables == tables
    assert descriptors[0] == descriptors[1]


@pytest.mark.parametrize("peer_blob", [
    b"short", (-1, 23, 9, 31), (6, 23, 23, 31),
])
def test_plan_exchange_rejects_invalid_peer_tables(peer_blob):
    import struct
    from types import SimpleNamespace as NS
    control = signals()
    control.begin = lambda *args: None
    if not isinstance(peer_blob, bytes):
        peer_blob = struct.pack("<4q", *peer_blob)
    control.store.set("3/blocks/1", peer_blob)
    backend = _p2p_backend(NS(_replica_groups=((0, 1),)))
    backend._geometry = NS(blocks_per_chunk=2, segment_bytes=65536)
    backend._signals, backend._epoch = control, 3
    with pytest.raises(ValueError):
        backend._check_plan(keys(2), ((11, 2), (5, 17)), (0,))


def _shared_view_backend(monkeypatch, peer_groups=None, peer_names=None):
    import json
    import sys
    from types import ModuleType, SimpleNamespace as NS
    calls = []
    def mapping(*args):
        calls.append(args)
        return NS(close=lambda: None)
    native = NS(ReplicaIpcMapping=mapping)
    module = ModuleType("tutti_kv_transfer")
    module._native = native
    monkeypatch.setitem(sys.modules, "tutti_kv_transfer", module)
    backend = _p2p_backend(NS(_replica_groups=((0, 1),)))
    backend._entity_names = ("layer.kv",)
    backend._raw_layers = (NS(numel=lambda: 32),)
    backend._exports = [NS(handle=lambda: b"local", offset=0, allocation_bytes=32)]
    backend._signals = signals(ranks=range(8))
    for rank in range(1, 8):
        members = (peer_groups or {}).get(rank, (rank // 2 * 2, rank // 2 * 2 + 1))
        name = (peer_names or {}).get(rank, "layer.kv")
        view = [(bytes([rank]).hex(), 0, 32, 32, 4, members, name)]
        backend._signals.store.set(f"handle/{rank}", json.dumps(view).encode())
    return backend, calls


def test_shared_views_allow_distinct_consistent_replica_pairs(monkeypatch):
    backend, calls = _shared_view_backend(monkeypatch)
    backend._exchange_shared_views()
    assert len(calls) == 7
    assert set(backend._peer_layers) == {(rank, 0) for rank in range(1, 8)}


def test_main_views_are_not_exported_or_imported(monkeypatch):
    import json
    from types import SimpleNamespace as NS
    backend, calls = _shared_view_backend(monkeypatch)
    backend._store._replica_groups = ((0,),)
    backend._exports = [None]
    for rank in range(1, 8):
        backend._signals.store.set(f"handle/{rank}", json.dumps([
            ("", 0, 32, 32, 4, (rank,), "layer.kv")]).encode())
    backend._exchange_shared_views()
    assert calls == [] and backend._peer_layers == {}
    assert json.loads(backend._signals.store.get("handle/0"))[0][0] == ""


def test_shared_allocation_handle_imported_once_for_two_entity_views(monkeypatch):
    import json
    from types import SimpleNamespace as NS
    backend, calls = _shared_view_backend(monkeypatch)
    backend._store._replica_groups = ((0, 1), (0, 1))
    backend._entity_names = ("layer.key", "layer.value")
    backend._raw_layers = (NS(numel=lambda: 16), NS(numel=lambda: 16))
    backend._layer_pages = (NS(page_bytes=4), NS(page_bytes=4))
    backend._exports = [NS(handle=lambda: b"local", offset=offset, allocation_bytes=32)
                        for offset in (0, 16)]
    for rank in range(1, 8):
        members = (rank // 2 * 2, rank // 2 * 2 + 1)
        views = [(bytes([rank]).hex(), offset, 16, 32, 4, members, name)
                 for offset, name in zip((0, 16), backend._entity_names)]
        backend._signals.store.set(f"handle/{rank}", json.dumps(views).encode())
    backend._exchange_shared_views()
    assert len(calls) == len(backend._peer_mappings) == 7
    for rank in range(1, 8):
        first, second = backend._peer_layers[rank, 0], backend._peer_layers[rank, 1]
        assert first[0] is second[0]
        assert first[1:] == (0, 16) and second[1:] == (16, 16)


@pytest.mark.parametrize("peer_groups", [
    {1: (1, 0)}, {1: (0, 1, 1)}, {2: (0, 2)},
])
def test_shared_views_reject_member_order_duplicate_or_inconsistent_groups(monkeypatch, peer_groups):
    from tutti.storage.tutti_nvme.store import DirectAdmissionError
    backend, calls = _shared_view_backend(monkeypatch, peer_groups=peer_groups)
    with pytest.raises(DirectAdmissionError, match="source members"):
        backend._exchange_shared_views()


def test_shared_views_reject_changed_entity_name(monkeypatch):
    from tutti.storage.tutti_nvme.store import DirectAdmissionError
    backend, calls = _shared_view_backend(monkeypatch, peer_names={1: "different.kv"})
    with pytest.raises(DirectAdmissionError, match="entity"):
        backend._exchange_shared_views()


def test_common_plan_requires_entity_name_order_consensus():
    import struct
    from types import SimpleNamespace as NS
    digests = []
    for names in (("key", "value"), ("value", "key")):
        backend = _p2p_backend(NS(_replica_groups=((0, 1), (0, 1))))
        backend._geometry = NS(blocks_per_chunk=1, segment_bytes=4096)
        backend._layer_pages = (NS(page_bytes=4), NS(page_bytes=4))
        backend._entity_names, backend._epoch = names, 3
        backend._signals = signals()
        backend._signals.begin = lambda epoch, digest: digests.append(digest)
        backend._signals.store.set("3/blocks/1", struct.pack("<q", 5))
        backend._check_plan(keys(1), ((7,),), (0, 1))
    assert digests[0] != digests[1]


@pytest.mark.parametrize("members", [(0, 0), (), (1, 2), (0, 8)])
def test_binding_rejects_invalid_replica_members_before_export(monkeypatch, members):
    import sys
    from types import ModuleType, SimpleNamespace as NS
    import torch
    from tutti.storage.tutti_nvme.layerwise_direct import TuttiLayerwiseDirectBackend
    from tutti.storage.tutti_nvme.replica_direct import TuttiReplicaDirectBackend
    from tutti.storage.tutti_nvme.store import DirectAdmissionError
    module = ModuleType("tutti_kv_transfer")
    module._native = NS()
    monkeypatch.setitem(sys.modules, "tutti_kv_transfer", module)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "PrefixStore", lambda *a: FakeSignalStore())
    monkeypatch.setattr(torch.distributed.distributed_c10d, "_get_default_store", lambda: None)
    def registered(self, pool, **kwargs):
        self._layer_pages = (NS(page_bytes=4),)
        return True
    monkeypatch.setattr(TuttiLayerwiseDirectBackend, "register_paged_caches", registered)
    backend = TuttiReplicaDirectBackend(NS(_rank_id=0, _replica_groups=(members,)))
    with pytest.raises(DirectAdmissionError, match="source members"):
        backend.register_paged_caches({"layer.kv": object()})
    assert backend._exports == [] and backend._poisoned


def test_owned_write_filter_keeps_original_page_offsets_and_empty_peers():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.layerwise_direct import LayerPageGeometry
    from tutti.storage.tutti_nvme.replica_direct import TuttiReplicaDirectBackend
    from tutti.storage.tutti_nvme.store import DirectPoolGeometry
    store = NS(_replica_groups=((0, 1),))
    backend = TuttiReplicaDirectBackend(store)
    backend._rank = 0
    backend._geometry = DirectPoolGeometry(4096, 8 * 65536, 65536, 0, 65536,
                                           8, 1, 128, 2, 131072, 0)
    backend._layer_pages = (LayerPageGeometry(65536, 65536, 8),)
    backend._layer_tickets = [22]
    entry = NS(chunk_id=keys(1)[0], target_offset=262144, target_ticket=7)
    plan = NS(direction="write")
    assert backend._page_requests(0, 3, 1, entry, plan) == [
        (7, 327680, 22, 196608, 65536, "write")]
    backend._rank = 1
    assert backend._page_requests(0, 3, 1, entry, plan) == []


def test_write_statistics_count_confirmed_bytes_even_for_partial_failure():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import TuttiReplicaDirectBackend
    backend = TuttiReplicaDirectBackend(NS())
    result = NS(ok=False, results=(NS(confirmed_bytes=65536), NS(confirmed_bytes=0)))
    backend._record_write_result(result)
    assert backend._stats["write_bytes"] == 65536


def test_nonowner_empty_submit_does_not_touch_runtime():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.store import TuttiKVStore
    def forbidden():
        raise AssertionError("empty replica owner reached the runtime")
    store = NS(_batch_width_limit=forbidden)
    assert TuttiKVStore._submit_retry(store, [], "read") == []
    assert TuttiKVStore._submit_retry(store, [], "write") == []


@pytest.mark.parametrize("owner", [False, True])
def test_replica_nonblocking_write_inherits_owner_filter_without_scatter(monkeypatch, owner):
    from contextlib import nullcontext
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.layerwise_direct import LayerPageGeometry
    from tutti.storage.tutti_nvme.replica_direct import TuttiReplicaDirectBackend
    from tutti.storage.tutti_nvme.store import DirectPoolGeometry, DirectTargetPlan, DirectTargetPlanEntry, TuttiDirectBackend
    assert TuttiReplicaDirectBackend.try_put_paged_batch is TuttiDirectBackend.try_put_paged_batch
    items = (int(owner).to_bytes(8, "little") + b"k" * 8,)
    submissions, releases, settled = [], [], []
    def submit(requests, **kwargs):
        submissions.append(tuple(requests))
        return NS(io_handle=17, initial_states=(True,), rejected=(), status_ok=True, status_msg="")
    runtime = NS(caps=lambda: {}, submit=submit, query_io=lambda handle: "COMPLETED",
                 wait=lambda handle, timeout: ("OK", "COMPLETED"), release_io=releases.append)
    store = NS(_replica_groups=((0, 1),), _runtime=runtime,
               _stream_for=lambda direction: (123, None), stream_context=lambda direction: nullcontext(),
               _record_event=lambda direction: NS(query=lambda: True), _batch_width_limit=lambda: None,
               _targets={"file://cell": NS(ticket=7)}, _layout=NS(target_generation=lambda key: 1),
               _track_completion=lambda *args: None, _accel_id=0, _execution="device",
               _on_put_settled=lambda ok, keys: settled.append((ok, tuple(keys))))
    backend = TuttiReplicaDirectBackend(store)
    backend._rank = 1
    backend._geometry = DirectPoolGeometry(65536, 32768, 4096, 4096, 4096, 8, 1, 1, 1, 4096, 0)
    backend._memory_ticket = 3
    backend._layer_pages = (LayerPageGeometry(4096, 4096, 8),)
    backend._layer_tickets = [3]
    entry = DirectTargetPlanEntry(items[0], 7, "file://cell", 4096, 1, 0)
    plan = DirectTargetPlan("write", 3, 1, (entry,))
    backend._target_plans["write"] = plan
    import tutti.storage.tutti_nvme.store as module
    monkeypatch.setattr(module, "_ASYNC_COMPLETION", False)
    completion = backend.try_put_paged_batch(items, 0, ((7,),), plan)
    assert completion._direction == "write" and not hasattr(completion, "done_event")
    assert completion.poll() is True and len(settled) == 1
    assert backend.try_put_paged_batch(items, 0, ((7,),), plan) is completion
    assert len(settled) == 1 and len(submissions) == int(owner)
    if owner:
        assert submissions[0][0][2:] == (3, 7 * 4096, 4096, "write")
        assert releases == [17]
    else:
        assert completion.wait_result(0).results == () and releases == []
    backend.end_target_plan("write")
    assert not backend._write_batches


def test_scatter_completion_does_not_finish_before_gpu_event():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import _ScatterCompletion
    ready = [False]
    completion = _ScatterCompletion(NS(), NS(query=lambda: ready[0]))
    assert completion.poll() is None
    with pytest.raises(TimeoutError):
        completion.wait_result(0)
    ready[0] = True
    assert completion.wait_result().ok
    def repeated_query():
        raise AssertionError("settled fence must not be queried again")
    completion.fence_event.query = repeated_query
    assert completion.poll() is True


@pytest.mark.parametrize("mixed", [False, True])
def test_async_compute_wait_preserves_scatter_fence_without_runtime_shortcut(monkeypatch, mixed):
    from types import SimpleNamespace as NS
    import torch
    import tutti.storage.tutti_nvme.store as module
    from tutti.storage.tutti_nvme.replica_direct import _ScatterCompletion
    monkeypatch.setattr(module, "_ASYNC_COMPLETION", True)
    def forbidden(*args, **kwargs):
        raise AssertionError("scatter fence must not be bypassed by async IO completion")
    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    runtime = NS(enqueue_wait=forbidden)
    event = NS(query=forbidden, synchronize=forbidden)
    scatter = _ScatterCompletion(runtime, event)
    completions = [NS(_settled=False, _handles=[11]), scatter] if mixed else [scatter]
    assert module.TuttiKVStore.enqueue_compute_wait(NS(_runtime=runtime), completions) is False
    assert scatter.fence_event is event


def test_async_compute_wait_still_enqueues_plain_owned_io_handles(monkeypatch):
    from types import SimpleNamespace as NS
    import torch
    import tutti.storage.tutti_nvme.store as module
    monkeypatch.setattr(module, "_ASYNC_COMPLETION", True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: NS(cuda_stream=123))
    calls = []
    store = NS(_runtime=NS(enqueue_wait=lambda handles, stream: calls.append((handles, stream))))
    completions = [NS(_settled=False, _handles=[11, 12]), NS(_settled=True, _handles=[13])]
    assert module.TuttiKVStore.enqueue_compute_wait(store, completions) is True
    assert calls == [([11, 12], 123)]


def test_async_replica_wait_uses_the_owned_io_stream(monkeypatch):
    from types import SimpleNamespace as NS
    import tutti.storage.tutti_nvme.store as module
    monkeypatch.setattr(module, "_ASYNC_COMPLETION", True)
    calls = []
    store = NS(
        _runtime=NS(enqueue_wait=lambda handles, stream: calls.append((handles, stream))),
        _stream_for=lambda direction: (456, NS(cuda_stream=456)),
    )
    completion = NS(_settled=False, _handles=[21, 22])
    assert module.TuttiKVStore.enqueue_io_wait(store, [completion], "read_async") is True
    assert calls == [([21, 22], 456)]


def test_busy_close_keeps_final_registered_backing_and_readonly_exports():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import TuttiReplicaDirectBackend
    store = NS(_read_copy_stream_obj=NS(query=lambda: False), _deferred_completions=[])
    backend = TuttiReplicaDirectBackend(store)
    backend._layer_tickets = [7]
    backing, exported, mapping = object(), object(), object()
    backend._layer_tensors = (backing,)
    backend._exports = [exported]
    backend._peer_mappings = {b"handle": mapping}
    ipc_event, peer_event = object(), object()
    backend._ipc_events = [ipc_event]
    backend._peer_events = {("ack", 2, 1): peer_event}
    def pending(stream, cancellable=True):
        raise TimeoutError("device work pending; backing retained")
    backend._store._read_stream_obj = store._read_copy_stream_obj
    store._stream_for = lambda direction: (None, getattr(store, f"_{direction}_stream_obj"))
    backend._wait_event = pending
    with pytest.raises(TimeoutError, match="backing retained"):
        backend.close()
    assert backend._layer_tickets == [7]
    assert backend._layer_tensors == (backing,)
    assert backend._exports == [exported]
    assert backend._peer_mappings[b"handle"] is mapping
    assert backend._ipc_events == [ipc_event]
    assert backend._peer_events == {("ack", 2, 1): peer_event}
    assert not backend._closed


def test_close_releases_scatter_plans_before_mappings_exports_and_final_tickets():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import TuttiReplicaDirectBackend
    calls = []
    store = NS(_runtime=NS(unregister_memory=lambda ticket: calls.append(("unregister", ticket))))
    backend = TuttiReplicaDirectBackend(store)
    backend._quiesce = lambda: calls.append("quiesce")
    backend._signals = NS(closing=lambda: calls.append("closing"),
                          release=lambda: calls.append("release"))
    backend._scatter_plans = {3: NS(close=lambda: calls.append("plan_close"))}
    backend._peer_mappings = {b"handle": NS(close=lambda: calls.append("mapping_close"))}
    backend._exports = [NS(close=lambda: calls.append("export_close"))]
    backend._layer_tickets = [7, 8]
    backend._memory_ticket = 7
    backend._layer_tensors = (object(), object())
    backend.close()
    assert calls == ["closing", "quiesce", "plan_close", "mapping_close", "release",
                     "export_close", ("unregister", 7), ("unregister", 8)]
    assert backend._closed
    assert backend._layer_tickets == []


def test_prefetch_task_reports_failure_and_stop_without_waiting():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import _ReplicaReadTask
    backend = _p2p_backend(NS())
    backend._signals = signals()
    task = _ReplicaReadTask(backend, NS())
    task.fail(RuntimeError("owned IO failed"))
    assert task.advance() is True
    assert backend._poisoned
    assert "owned IO failed" in str(backend._prefetch_error)
    backend._prefetch_stop.set()
    assert _ReplicaReadTask(backend, NS()).advance() is True


def test_replica_backend_has_no_extra_send_or_receive_payload():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import TuttiReplicaDirectBackend
    backend = TuttiReplicaDirectBackend(NS())
    assert backend._entity_sequence == ()
    assert backend._next_entity == 0
    for name in ("_send", "_send_ticket", "_send_allocation", "_receive", "_slot_count"):
        assert not hasattr(backend, name)


def test_replica_refuses_detached_read():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import TuttiReplicaDirectBackend
    backend = TuttiReplicaDirectBackend(NS())
    assert backend.supports_async_read is False
    backend._memory_ticket = 3
    with pytest.raises(RuntimeError, match="async read"):
        backend.begin_detached_read_plan(keys(1))


def test_recovery_runs_on_the_feeder_without_ordered_collectives():
    from tutti.storage.tutti_nvme.replica_direct import TuttiReplicaDirectBackend
    assert TuttiReplicaDirectBackend.requires_ordered_collectives is False


def _p2p_backend(store, **attrs):
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import TuttiReplicaDirectBackend
    if not hasattr(store, "_stream_for"):
        store._stream_for = lambda direction: (None, getattr(store, f"_{direction}_stream_obj"))
    backend = TuttiReplicaDirectBackend(store)
    backend._geometry = NS(accel_id=0, num_layers=1, blocks_per_chunk=1)
    backend._layer_pages = (NS(page_bytes=4),)
    backend._rank = 0
    for name, value in attrs.items():
        setattr(backend, name, value)
    return backend


class FakeSignalStore:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value.encode() if isinstance(value, str) else bytes(value)

    def get(self, key):
        return self.data[key]

    def check(self, keys):
        return all(k in self.data for k in keys)

    def compare_set(self, key, expected, desired):
        old = self.data.get(key, b"")
        if old == expected:
            self.data[key] = desired
        return self.data.get(key, b"")


def signals(rank=0, ranks=(0, 1), epoch=1, store=None):
    from tutti.storage.tutti_nvme.replica_direct import _ReplicaSignals
    result = _ReplicaSignals(store or FakeSignalStore(), rank, ranks)
    result.epoch = epoch
    return result


def test_detached_replica_borrows_memory_without_replacing_store_owner(monkeypatch):
    from types import SimpleNamespace as NS
    import torch
    import torch.distributed as dist
    prefixes = []
    def prefix(name, store):
        prefixes.append(name)
        return store
    monkeypatch.setattr(dist, "PrefixStore", prefix)
    event = NS(record=lambda stream: None)
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
    store = NS(_read_async_stream=1)
    backend = _p2p_backend(store)
    backend._signals = signals()
    first, second = NS(plan_token=1), NS(plan_token=2)
    backend.prepare_detached_read(first, [b"a"], [[1]], (0,), lambda: False)
    backend.prepare_detached_read(second, [b"b"], [[2]], (0,), lambda: False)
    assert store._direct_backend is backend
    a, b = backend._detached_readers.values()
    assert a._layer_tickets is backend._layer_tickets
    assert a._raw_layers is backend._raw_layers
    assert a._invalid_plan_tokens is backend._invalid_plan_tokens
    assert a._parent is backend and b._parent is backend
    assert a._read_direction == a._io_direction == "read_async"
    assert a._prefetch_job is None and b._prefetch_job is None
    assert a._compute_prefix is event
    assert prefixes == ["detached/1/", "detached/2/"]
    a._fail(RuntimeError("detached failed"))
    assert backend._poisoned and b._prefetch_error is None


def test_background_admission_checks_new_window_without_waiting():
    from types import SimpleNamespace as NS
    backend = _p2p_backend(NS())
    backend._signals = signals()
    state = [None]
    backend._background_allowed = lambda: state[0]
    assert backend._admit_background_entity(0) is False
    state[0] = (1, 0)
    assert backend._admit_background_entity(0) is True
    assert backend._admit_background_entity(1) is True
    state[0] = None
    assert backend._admit_background_entity(1) is False
    state[0] = (2, 0)
    assert backend._admit_background_entity(1) is True
    assert not any("admit" in key for key in backend._signals.store.data)
    assert not backend._poisoned


def test_paused_peer_ready_does_not_expire_notification_timeout(monkeypatch):
    from tutti.storage.tutti_nvme import replica_direct
    control = signals()
    control._pause_waits = True
    control.epoch = 1
    control.store.set("1/ready/0/0", b"ready")
    now = [0.0]
    monkeypatch.setattr(replica_direct.time, "monotonic", lambda: now[0])
    def resume(delay):
        now[0] = 1000.0
        control.store.set("1/ready/0/1", b"ready")
    monkeypatch.setattr(replica_direct.time, "sleep", resume)
    control.wait_ready(0, (0, 1), None)
    assert control.error is None


def test_partial_submit_yields_and_retries_only_rejected_request():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import _ReplicaSubmit
    calls = []
    def submit(batch, **kwargs):
        calls.append(list(batch))
        return (NS(io_handle=17, initial_states=(True, False), rejected=(1,), status_ok=False,
                   status_msg="partial") if len(calls) == 1 else
                NS(io_handle=18, initial_states=(True,), rejected=(), status_ok=True, status_msg=""))
    store = NS(_runtime=NS(submit=submit), _batch_width_limit=lambda: None, _accel_id=0,
               _execution="device", _stream_for=lambda direction: (123, None))
    task = _ReplicaSubmit(store, [(1,), (2,)], "read_async")
    assert task.advance() is False
    assert calls == [[(1,), (2,)]]
    assert task.advance() is True
    assert calls == [[(1,), (2,)], [(2,)]]
    assert [(h.handle, h.batch_indices) for h in task.handles] == [(17, (0,)), (18, (1,))]


def test_open_batch_capacity_retries_do_not_refresh_deadline(monkeypatch):
    backend, plan, calls = _producer_backend(monkeypatch, count=2)
    state = backend._new_entity(plan, 2)
    deadline = state["deadline"]
    backend._store._runtime.submit = lambda *a, **k: type("Result", (), dict(
        io_handle=None, initial_states=(False, False), rejected=(0, 1), status_ok=False,
        status_msg="in-flight operation capacity exhausted"))()
    for _ in range(3):
        backend._advance_submit(state)
        assert state["deadline"] == deadline and state["phase"] == "submit"


def test_ready_and_ack_publish_binary_ipc_handles_without_gpu_calls():
    import threading
    store = FakeSignalStore()
    local, peer = signals(store=store), signals(rank=1, store=store)
    handle = bytes(range(64))
    local.ready(3, handle)
    local.ack(3, handle[::-1])
    peer.ready(3, handle[::-1])
    peer.ack(3, handle)
    local.wait_ready(3, (0, 1), threading.Event())
    local.wait_acks(3, threading.Event())
    assert store.get("1/ready/3/0") == handle
    assert store.get("1/ack/3/0") == handle[::-1]
    assert store.get("1/ready/3/1") == handle[::-1]
    assert store.get("1/ack/3/1") == handle


def test_old_epoch_cannot_satisfy_new_ready_or_ack(monkeypatch):
    import threading
    import tutti.storage.tutti_nvme.replica_direct as module
    control = signals()
    control.store.set("1/ready/0/1", b"1")
    control.store.set("1/ack/0/1", b"1")
    control.epoch = 2
    monkeypatch.setattr(module, "_POLL_SECONDS", 0)
    for wait in (lambda: control.wait_ready(0, (1,), threading.Event()),
                 lambda: control.wait_acks(0, threading.Event())):
        with pytest.raises(TimeoutError, match="notification timed out"):
            wait()


def test_peer_failure_is_sticky_and_stops_every_wait():
    import threading
    store = FakeSignalStore()
    a, b = signals(store=store), signals(rank=1, store=store)
    b.fail(RuntimeError("owned IO failed"))
    for call in (lambda: a.wait_ready(7, (1,), threading.Event()),
                 lambda: a.wait_acks(3, threading.Event()), lambda: a.ready(8)):
        with pytest.raises(RuntimeError, match="owned IO failed"):
            call()
    with pytest.raises(RuntimeError, match="poisoned"):
        a.begin(2, b"plan")


def test_ack_requires_every_consumer_and_cancellation_is_bounded():
    import threading
    control = signals(ranks=(0, 1, 2))
    control.ack(3)
    control.store.set("1/ack/3/1", b"1")
    stop = threading.Event()
    stop.set()
    with pytest.raises(RuntimeError, match="stopped"):
        control.wait_acks(3, stop)
    control.store.set("1/ack/3/2", b"1")
    control.wait_acks(3, threading.Event())


def test_plan_mismatch_fails_before_any_device_transfer():
    control = signals(epoch=0)
    control.store.set("1/plan/1", b"different geometry")
    with pytest.raises(RuntimeError, match="geometry differ"):
        control.begin(1, b"plan")
    assert control.store.check(["1/failed/0"])


def test_release_requires_all_importers(monkeypatch):
    import tutti.storage.tutti_nvme.replica_direct as module
    control = signals()
    monkeypatch.setattr(module, "_POLL_SECONDS", 0)
    with pytest.raises(TimeoutError):
        control.release()
    control.store.set("released/1", b"1")
    control.release()


def test_closing_peer_prevents_next_request():
    control = signals()
    control.store.set("terminal", b"peer 1 is closing")
    with pytest.raises(RuntimeError, match="closing"):
        control.begin(2, b"plan")
    assert not control.store.check(["2/plan/0"])


def test_binding_failure_is_visible_in_next_epoch():
    store = FakeSignalStore()
    a = signals(epoch=0, store=store)
    b = signals(rank=1, epoch=1, store=store)
    a.fail(RuntimeError("binding allocation failed"))
    with pytest.raises(RuntimeError, match="binding allocation failed"):
        b.check()


@pytest.mark.parametrize("alive", [False, True])
def test_stop_prefetch_publishes_abort_until_all_entities_recovered(alive):
    from types import SimpleNamespace as NS
    backend = _p2p_backend(NS())
    store = FakeSignalStore()
    backend._signals = signals(store=store)
    backend._active_token = 1
    backend._entity_sequence = (2, 0)
    backend._next_entity = 1
    state, calls = [alive], []
    def join(timeout):
        calls.append("join")
        state[0] = False
    backend._prefetch_job = NS(is_alive=lambda: state[0], join=join)
    backend.stop_prefetch()
    assert backend._poisoned
    assert "aborted" in str(backend._prefetch_error)
    assert calls == ["join"] and backend._prefetch_job is None
    peer = signals(rank=1, store=store)
    with pytest.raises(RuntimeError, match="aborted"):
        peer.check()


@pytest.mark.parametrize("alive", [False, True])
def test_stop_prefetch_preserves_first_io_failure(alive):
    from types import SimpleNamespace as NS
    backend = _p2p_backend(NS())
    backend._signals = signals()
    backend._active_token = 1
    backend._entity_sequence = (2, 0)
    backend._next_entity = 1
    original = RuntimeError("original owned IO failure")
    backend._fail(original)
    terminal = backend._signals.store.data["terminal"]
    state = [alive]
    backend._prefetch_job = NS(is_alive=lambda: state[0],
                                  join=lambda timeout: state.__setitem__(0, False))
    backend.stop_prefetch()
    assert backend._prefetch_error is original
    assert backend._signals.error is original
    assert backend._signals.store.data["terminal"] == terminal


def test_stop_prefetch_does_not_abort_completed_recovery():
    from types import SimpleNamespace as NS
    backend = _p2p_backend(NS())
    backend._signals = signals()
    backend._active_token = 1
    backend._entity_sequence = (2, 0)
    backend._next_entity = 2
    backend._prefetch_job = NS(is_alive=lambda: False, join=lambda timeout: None)
    backend.stop_prefetch()
    assert not backend._poisoned and backend._prefetch_error is None
    assert "terminal" not in backend._signals.store.data


def test_completed_request_joins_before_stopping_producer():
    from types import SimpleNamespace as NS
    calls = []
    store = NS(_deferred_completions=[object()], _read_stream_obj="read",
               _read_copy_stream_obj="read_copy", _inflight_by_chunk={b"chunk": {object()}},
               _wait_chunk_io=lambda keys: calls.append(("drain_inflight", keys)),
               drain_deferred_completions=lambda: calls.append("drain_deferred"))
    backend = _p2p_backend(store)
    backend._layer_tickets = [7]
    backend._prefetch_keys = {b"chunk"}
    backend._active_token = 1
    backend._entity_sequence = (0,)
    backend._next_entity = 1
    ipc_event, peer_event = object(), object()
    backend._ipc_events = [ipc_event]
    backend._peer_events = {("ack", 0, 1): peer_event}
    def quiescent():
        assert backend._ipc_events == [ipc_event]
        assert backend._peer_events == {("ack", 0, 1): peer_event}
        assert backend._retained and backend._pending_events
        calls.append("quiescent")
    backend._signals = NS(quiescent=quiescent)
    backend.join_prefetch = lambda: calls.append("join")
    backend.stop_prefetch = lambda: calls.append("stop")
    backend._wait_event = lambda stream, cancellable: calls.append((stream, cancellable))
    backend._retained = [object()]
    backend._pending_events = [object()]
    backend._quiesce()
    assert calls == ["join", "stop", ("read", False), ("read_copy", False),
                     ("drain_inflight", {b"chunk"}), "drain_deferred", "quiescent"]
    assert not backend._poisoned
    assert backend._retained == [] and backend._pending_events == []
    assert backend._ipc_events == [] and backend._peer_events == {}


@pytest.mark.parametrize("started_poisoned", [False, True])
def test_quiesce_drains_then_reports_observer_failure_during_join(started_poisoned):
    from types import SimpleNamespace as NS
    calls = []
    store = NS(_deferred_completions=[object()], _read_stream_obj="read",
               _read_copy_stream_obj="read_copy", _inflight_by_chunk={b"chunk": {object()}},
               _wait_chunk_io=lambda keys: calls.append("drain_inflight"),
               drain_deferred_completions=lambda: calls.append("drain_deferred"))
    backend = _p2p_backend(store)
    backend._active_token = 1
    backend._entity_sequence = (0,)
    backend._next_entity = 1
    backend._layer_tickets = [7]
    backend._signals = signals(ranks=(0,))
    original = RuntimeError("original owned IO failure")
    late = RuntimeError("late observer IO failure")
    if started_poisoned:
        backend._fail(original)
    def observer_join(timeout):
        calls.append("job")
        backend._fail(late)
    backend._prefetch_job = NS(join=observer_join, is_alive=lambda: False)
    backend._wait_event = lambda stream, cancellable: calls.append((stream, cancellable))
    ipc_event, peer_event, plan = object(), object(), object()
    backend._ipc_events = [ipc_event]
    backend._peer_events = {("ack", 0, 1): peer_event}
    backend._retained = [plan]
    backend._pending_events = [ipc_event]
    def quiescent():
        assert backend._ipc_events == [ipc_event]
        assert backend._peer_events == {("ack", 0, 1): peer_event}
        calls.append("quiescent")
    backend._signals.quiescent = quiescent
    if started_poisoned:
        backend._quiesce()
    else:
        with pytest.raises(RuntimeError, match="late observer IO failure"):
            backend._quiesce()
    assert backend._quiesced and backend._poisoned
    assert backend._prefetch_error is (original if started_poisoned else late)
    assert calls[-5:] == [("read", False), ("read_copy", False),
                           "drain_inflight", "drain_deferred", "quiescent"]
    assert backend._ipc_events == [] and backend._peer_events == {}
    assert backend._retained == [] and backend._pending_events == []
    backend._quiesce()


def test_peer_quiescence_timeout_retains_imported_and_exported_events():
    from types import SimpleNamespace as NS
    backend = _p2p_backend(NS(_deferred_completions=[]))
    ipc_event, peer_event, plan = object(), object(), object()
    backend._ipc_events = [ipc_event]
    backend._peer_events = {("ack", 2, 1): peer_event}
    backend._retained = [plan]
    backend._pending_events = [ipc_event]
    def quiescent():
        raise TimeoutError("peer device work pending")
    backend._signals = NS(quiescent=quiescent)
    with pytest.raises(TimeoutError, match="peer device work pending"):
        backend._quiesce()
    assert not backend._quiesced
    assert backend._ipc_events == [ipc_event]
    assert backend._peer_events == {("ack", 2, 1): peer_event}
    assert backend._retained == [plan] and backend._pending_events == [ipc_event]


def _event_handle(phase, entity, rank):
    return f"{phase}/{entity}/{rank}".encode().ljust(64, b"\0")


def _fake_cuda_events(monkeypatch, calls, record_label="record_device_event"):
    import torch
    events, imports = [], []
    class Event:
        def __init__(self, *args, **kwargs):
            self.options = kwargs
            self.handle = _event_handle("local", len(events), 0)
            self.recorded = False
            self.record_failure = None
            self.ready = None
            events.append(self)
        def record(self, stream):
            calls.append(record_label)
            if self.record_failure is not None:
                raise self.record_failure
            self.recorded = True
            self.stream = stream
        def ipc_handle(self):
            assert self.recorded and self.options.get("interprocess") is True
            return self.handle
        def query(self):
            if self.ready is None:
                raise AssertionError("GPU query is forbidden before compute enqueue")
            return self.ready
        def synchronize(self):
            raise AssertionError("GPU host synchronization is forbidden")
        @classmethod
        def from_ipc_handle(cls, device, handle):
            assert device == 0
            if not isinstance(handle, bytes) or len(handle) != 64:
                raise RuntimeError("invalid CUDA IPC event handle")
            event = cls(interprocess=True)
            event.handle, event.recorded = handle, True
            imports.append(event)
            calls.append("import_" + handle.rstrip(b"\0").decode())
            return event
    monkeypatch.setattr(torch.cuda, "Event", Event)
    return events, imports


def _recovery_backend(monkeypatch, physical=2, members=(0, 1)):
    import queue
    from contextlib import nullcontext
    from types import SimpleNamespace as NS
    import torch
    from tutti.index.chunk_index import derive_io_key

    calls = []
    events, imports = _fake_cuda_events(monkeypatch, calls)
    event = torch.cuda.Event(interprocess=True)
    event.record(object())
    calls.clear()
    items, tables = [key[:16] for key in keys(2)], ((7,), (3,))
    def record_read():
        calls.append("compute_handover")
        ready = torch.cuda.Event()
        ready.record(store._read_stream_obj)
        return ready
    def wait_event(done):
        calls.append("wait_" + done.handle.rstrip(b"\0").decode())
    store = NS(_rank_id=0, _replica_groups=(members,) * (physical + 1), _runtime=NS(),
               _live={derive_io_key(k, physical) for k in items},
               stream_context=lambda direction: nullcontext(),
               _read_stream_obj=NS(wait_event=wait_event), record_read_event=record_read)
    backend = _p2p_backend(store, _epoch=1)
    backend._test_events, backend._test_imports = events, imports
    backend._geometry = NS(accel_id=0, num_layers=physical + 1, blocks_per_chunk=1)
    backend._layer_pages = (NS(page_bytes=4, block_stride=4),) * (physical + 1)
    backend._prefetch_job = object()
    backend._prefetch_keys, backend._prefetch_tables = tuple(items), tuple(tables)
    backend._target_plans["read"] = NS(plan_token=1)
    backend._active_token = 1
    backend._validate_target_plan = lambda plan: None
    backend.validate_block_tables = lambda value: value
    backend._entity_sequence = (physical,)
    backend._ready_entities = queue.Queue()
    backend._ready_entities.put((physical, event, None))
    backend._raw_layers = tuple(torch.zeros((8, 4), dtype=torch.uint8)
                                for _ in range(physical + 1))
    backend._raw_layers[physical][7] = torch.tensor([10, 11, 12, 13], dtype=torch.uint8)
    peer_data = torch.zeros((8, 4), dtype=torch.uint8)
    peer_data[5] = torch.tensor([20, 21, 22, 23], dtype=torch.uint8)
    backend._peer_tables = {0: tables, members[1]: ((6,), (5,))}
    backend._signals = signals(ranks=tuple(range(8)))
    for rank in members:
        backend._signals.store.set(f"1/ready/{physical}/{rank}",
                                   _event_handle("ready", physical, rank))
    def launch():
        calls.append("launch")
        backend._raw_layers[physical][3].copy_(peer_data[5])
    backend._scatter_plans = {physical: NS(launch=launch)}
    return backend, items, tables, calls


@pytest.mark.parametrize("mode", ["pending", "launch_failure", "record_failure"])
def test_get_paged_batch_copies_peer_pages_without_waiting_gpu_completion(monkeypatch, mode):
    backend, items, tables, calls = _recovery_backend(monkeypatch)
    physical = 2
    scatter_plan = backend._scatter_plans[physical]
    backend._signals.store.set(f"1/ack/{physical}/1", _event_handle("ack", physical, 1))
    def forbidden(*args, **kwargs):
        raise AssertionError("recovery must not wait for GPU completion on the host")
    backend._wait_event = forbidden
    if mode == "launch_failure":
        def launch():
            calls.append("launch")
            raise RuntimeError("launch failed")
        scatter_plan.launch = launch
    elif mode == "record_failure":
        def record(self, stream):
            raise RuntimeError("scatter event record failed")
        monkeypatch.setattr(type(backend._test_events[0]), "record", record)
    if mode != "pending":
        with pytest.raises(RuntimeError, match="failed"):
            _advance_test_scatter(backend)
            backend.get_paged_batch(items, physical, tables)
        assert calls.count("launch") == 1 and backend._poisoned
        assert not backend._signals.store.check([f"1/ack/{physical}/0"])
        assert "compute_handover" not in calls
        assert scatter_plan in backend._retained
        assert backend._scatter_plans[physical] is scatter_plan
        if mode == "record_failure":
            assert backend._pending_events and backend._ipc_events
        return
    _advance_test_scatter(backend)
    completion = backend.get_paged_batch(items, physical, tables)
    assert completion.fence_event is backend._test_events[-1]
    assert backend._raw_layers[physical][7].tolist() == [10, 11, 12, 13]
    assert backend._raw_layers[physical][3].tolist() == [20, 21, 22, 23]
    local_ack = backend._signals.store.get(f"1/ack/{physical}/0")
    assert len(local_ack) == 64 and local_ack != b"1"
    assert backend._stats["gather_bytes"] == 4
    assert backend._next_entity == 1
    assert calls.count("launch") == 1
    assert calls.index("wait_ready/2/1") < calls.index("launch")
    assert calls.index("launch") < calls.index("wait_ack/2/1") < calls.index("compute_handover")
    assert scatter_plan in backend._retained
    assert backend._pending_events and backend._ipc_events
    assert set(backend._peer_events) == {("ready", physical, 1), ("ack", physical, 1)}
    assert [event.handle for event in backend._test_imports] == [
        _event_handle("ready", physical, 1), _event_handle("ack", physical, 1)]
    assert all(event.ready is None for event in backend._test_events)


@pytest.mark.parametrize("mode", ["peer", "handover", "missing_ready", "missing_ack"])
def test_get_paged_batch_fails_closed_before_compute_handover(monkeypatch, mode):
    import tutti.storage.tutti_nvme.replica_direct as module
    backend, items, tables, calls = _recovery_backend(monkeypatch)
    physical = 2
    monkeypatch.setattr(module, "_POLL_SECONDS", 0)
    if mode == "peer":
        backend._signals.store.set("terminal", b"peer 1 failed: owned IO failed")
    elif mode == "handover":
        backend._ready_entities.get_nowait()
        backend._ready_entities.put((physical, None, RuntimeError("owned IO failed")))
    elif mode == "missing_ready":
        del backend._signals.store.data[f"1/ready/{physical}/1"]
    error = RuntimeError if mode in ("peer", "handover") else TimeoutError
    with pytest.raises(error):
        _advance_test_scatter(backend)
        backend.get_paged_batch(items, physical, tables)
    assert "compute_handover" not in calls
    assert backend._poisoned
    if mode == "missing_ack":
        assert calls.count("launch") == 1
        assert backend._signals.store.check([f"1/ack/{physical}/0"])
        assert backend._scatter_plans[physical] in backend._retained
    else:
        assert "launch" not in calls
        assert not backend._signals.store.check([f"1/ack/{physical}/0"])


@pytest.mark.parametrize("missing_peer", [None, 2])
def test_compute_handover_requires_every_peer_scatter_record_not_gpu_done(monkeypatch, missing_peer):
    import tutti.storage.tutti_nvme.replica_direct as module
    backend, items, tables, calls = _recovery_backend(monkeypatch, members=(0, 1, 2))
    monkeypatch.setattr(module, "_POLL_SECONDS", 0)
    for peer in (1, 2):
        if peer != missing_peer:
            backend._signals.store.set(f"1/ack/2/{peer}", _event_handle("ack", 2, peer))
    if missing_peer is not None:
        with pytest.raises(TimeoutError, match="notification timed out"):
            _advance_test_scatter(backend)
            backend.get_paged_batch(items, 2, tables)
        assert "compute_handover" not in calls
        assert backend._scatter_plans[2] in backend._retained
        assert backend._pending_events and backend._ipc_events
        assert backend._signals.store.check(["1/ack/2/0"])
        assert backend._next_entity == 0
        return
    _advance_test_scatter(backend)
    completion = backend.get_paged_batch(items, 2, tables)
    assert completion.fence_event.recorded
    for peer in (1, 2):
        assert calls.index(f"wait_ready/2/{peer}") < calls.index("launch")
        assert calls.index("launch") < calls.index(f"wait_ack/2/{peer}") < calls.index("compute_handover")
    assert set(backend._peer_events) == {
        (phase, 2, peer) for phase in ("ready", "ack") for peer in (1, 2)}
    assert all(event.ready is None for event in backend._test_events)


@pytest.mark.parametrize("method", ["poll", "wait_result"])
def test_late_owned_failure_rejects_scatter_success_after_compute_enqueue(monkeypatch, method):
    backend, items, tables, calls = _recovery_backend(monkeypatch)
    backend._signals.store.set("1/ack/2/1", _event_handle("ack", 2, 1))
    _advance_test_scatter(backend)
    completion = backend.get_paged_batch(items, 2, tables)
    assert "compute_handover" in calls
    backend._fail(RuntimeError("background owned IO failed after ready"))
    with pytest.raises(RuntimeError, match="background owned IO failed after ready"):
        getattr(completion, method)()
    assert backend._poisoned and backend._retained
    assert backend._ipc_events and backend._peer_events


def test_peer_signal_failure_at_scatter_poll_poisoned_backend_before_safe_quiesce(monkeypatch):
    from types import SimpleNamespace as NS
    backend, items, tables, calls = _recovery_backend(monkeypatch)
    backend._signals.store.set("1/ack/2/1", _event_handle("ack", 2, 1))
    _advance_test_scatter(backend)
    completion = backend.get_paged_batch(items, 2, tables)
    backend._entity_sequence = (2, 0)
    backend._prefetch_job = NS(join=lambda timeout: calls.append("producer_join"),
                                  is_alive=lambda: False)
    backend._store._deferred_completions = []
    peer = signals(rank=1, ranks=range(8), store=backend._signals.store)
    peer.fail(RuntimeError("peer owned IO failed after scatter enqueue"))
    terminal = backend._signals.store.get("terminal")
    assert not backend._poisoned and backend._prefetch_error is None
    with pytest.raises(RuntimeError, match="peer owned IO failed after scatter enqueue") as failed:
        completion.poll()
    assert backend._poisoned
    assert backend._prefetch_error is failed.value
    assert backend._signals.error is failed.value
    for rank in range(1, 8):
        backend._signals.store.set(f"1/quiescent/{rank}", b"1")
    backend._quiesce()
    assert backend._quiesced
    assert backend._prefetch_error is failed.value
    assert backend._signals.store.get("terminal") == terminal
    assert "aborted" not in str(backend._prefetch_error)
    assert backend._prefetch_job is None
    assert backend._ipc_events == [] and backend._peer_events == {}
    assert backend._retained == [] and backend._pending_events == []


@pytest.mark.parametrize("phase", ["ready", "ack"])
def test_invalid_peer_event_handle_fails_closed_in_cuda_import(monkeypatch, phase):
    backend, items, tables, calls = _recovery_backend(monkeypatch)
    backend._signals.store.set("1/ack/2/1", _event_handle("ack", 2, 1))
    backend._signals.store.set(f"1/{phase}/2/1", b"1")
    with pytest.raises(RuntimeError, match="invalid CUDA IPC event handle"):
        _advance_test_scatter(backend)
        backend.get_paged_batch(items, 2, tables)
    assert backend._poisoned and "compute_handover" not in calls
    if phase == "ready":
        assert "launch" not in calls
    else:
        assert "launch" in calls
        assert backend._scatter_plans[2] in backend._retained
        assert backend._pending_events and backend._ipc_events


def test_peer_event_import_is_cached_by_phase_entity_and_peer(monkeypatch):
    backend, items, tables, calls = _recovery_backend(monkeypatch)
    backend._signals.store.set("1/ack/2/1", _event_handle("ack", 2, 1))
    for phase in ("ready", "ack"):
        backend._wait_peer_events(2, (0, 1), phase)
        backend._wait_peer_events(2, (0, 1), phase)
    assert len(backend._test_imports) == 2
    assert set(backend._peer_events) == {("ready", 2, 1), ("ack", 2, 1)}
    assert calls.count("wait_ready/2/1") == calls.count("wait_ack/2/1") == 2
    assert not any(key[2] == 0 for key in backend._peer_events)


@pytest.mark.parametrize("mode", ["duplicate", "out_of_order", "wrong_handover"])
def test_entity_recovery_sequence_rejects_invalid_callbacks(monkeypatch, mode):
    backend, items, tables, calls = _recovery_backend(monkeypatch)
    if mode == "duplicate":
        backend._next_entity = 1
    elif mode == "out_of_order":
        backend._entity_sequence = (1, 2)
    else:
        backend._ready_entities.get_nowait()
        backend._ready_entities.put((1, None, None))
    with pytest.raises(RuntimeError):
        _advance_test_scatter(backend)
        backend.get_paged_batch(items, 2, tables)
    assert "launch" not in calls and "compute_handover" not in calls


def test_ack_is_scoped_to_entity_replica_members_not_global_world(monkeypatch):
    import threading
    import tutti.storage.tutti_nvme.replica_direct as module
    control = signals(rank=4, ranks=range(8))
    control.ack(7)
    control.store.set("1/ack/7/6", b"1")
    control.wait_acks(7, threading.Event(), (4, 6))
    monkeypatch.setattr(module, "_POLL_SECONDS", 0)
    with pytest.raises(TimeoutError):
        control.wait_acks(7, threading.Event(), (4, 6, 7))


def _advance_test_scatter(backend):
    from types import SimpleNamespace as NS
    import time
    from tutti.storage.tutti_nvme.replica_direct import _POLL_SECONDS
    try:
        backend._check_read_failure()
        physical = backend._entity_sequence[backend._next_entity]
        item = backend._ready_entities.get_nowait()
        if item[0] != physical:
            raise RuntimeError("replica prefetch entity order differs")
        if item[2] is not None:
            raise item[2]
        completion = NS(poll=lambda: True, observation_error=None,
                        wait_result=lambda *a: NS(ok=True, results=()))
        state = dict(physical=physical, members=tuple(backend._store._replica_groups[physical]),
                     phase="ready", io_done=item[1], completion=completion,
                     submit=NS(handles=[], segments=[]),
                     deadline=time.monotonic() + _POLL_SECONDS)
        backend._prefetch_task = NS(states=[state])
        backend._advance_scatter(state)
    except IndexError as exc:
        backend._fail(RuntimeError("replica invalid entity callback"))
        raise backend._prefetch_error from exc
    except Exception as exc:
        backend._fail(exc)
        raise


def _produce_test_owned(backend, plan):
    try:
        for key in backend._prefetch_keys:
            backend._validate_plan_entry(plan.entry_for(key))
        backend._test_states = []
        for physical in backend._entity_sequence:
            state = backend._new_entity(plan, physical)
            backend._test_states.append(state)
            while state["phase"] == "submit":
                backend._advance_submit(state)
            backend._ready_entities.put((physical, state["io_done"], None))
    except Exception as exc:
        backend._fail(exc)


def _complete_test_owned(backend):
    try:
        for state in getattr(backend, "_test_states", ()):
            backend._advance_owned(state)
    except Exception as exc:
        backend._fail(exc)


def _producer_backend(monkeypatch, count=1000, mode="ok", rank=0):
    import queue
    from contextlib import nullcontext
    from types import SimpleNamespace as NS
    import torch
    import tutti.storage.tutti_nvme.replica_direct as module

    physical, calls = 2, []
    items = keys(count)
    tables = tuple((3 * i + 1, 3 * i + 2) for i in range(count))
    entries = {key: NS(chunk_id=key, target_ticket=i + 40, target_offset=8192)
               for i, key in enumerate(items)}
    events, imports = _fake_cuda_events(monkeypatch, calls, record_label="event_record")
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    def submit(requests, **kwargs):
        calls.append(("submit", list(requests), "read", backend._io_direction))
        return NS(io_handle=101, initial_states=(True,) * len(requests), rejected=(),
                  status_ok=True, status_msg="")
    store = NS(_replica_groups=((0, 1),) * 3, _runtime=NS(submit=submit), _deferred_completions=[],
               stream_context=lambda direction: nullcontext(),
               _read_copy_stream_obj=NS(), _batch_width_limit=lambda: None,
               _accel_id=0, _execution="device",
               _track_completion=lambda completion, chunk_ids: calls.append("track"))
    backend = _p2p_backend(store, _rank=rank, _epoch=1)
    backend._test_events, backend._test_imports = events, imports
    backend._geometry = NS(accel_id=0, num_layers=3, blocks_per_chunk=2, segment_bytes=65536)
    backend._layer_pages = (NS(page_bytes=4096, block_stride=8192),) * 3
    backend._layer_tickets = [11, 22, 33]
    backend._prefetch_keys, backend._prefetch_tables = tuple(items), tables
    backend._entity_sequence = (physical,)
    backend._ready_entities = queue.Queue()
    backend._signals = signals(rank=rank)
    backend._validate_plan_entry = lambda entry: None
    class Completion:
        observation_error = None
        def __init__(self, runtime, handles, on_settled, **kwargs):
            assert kwargs == {"auto_watch": False, "direction": "read",
                              "retain_observation_errors": True}
        def poll(self):
            calls.append("poll")
            if mode == "pending":
                backend._prefetch_stop.set()
                return None
            return mode != "failure"
        def wait_result(self, timeout=None):
            calls.append("outcome")
            return NS(ok=mode != "failure", results=(NS(confirmed_bytes=4096),))
    monkeypatch.setattr(module, "_TuttiCompletion", Completion)
    def wait_event(done, *args, **kwargs):
        calls.append("event_wait")
        assert done.recorded
        assert any(value == done.handle and "/ready/" in key
                   for key, value in backend._signals.store.data.items())
    backend._wait_event = wait_event
    return backend, NS(direction="read", entry_for=lambda key: entries[key]), calls


def test_producer_reads_whole_entity_once_into_final_tickets_without_send(monkeypatch):
    backend, plan, calls = _producer_backend(monkeypatch)
    _produce_test_owned(backend, plan)
    _complete_test_owned(backend)
    assert not backend._poisoned
    submissions = [call for call in calls if isinstance(call, tuple)]
    assert len(submissions) == 1
    _, requests, direction, stream_direction = submissions[0]
    assert len(requests) == 1000
    assert (direction, stream_direction) == ("read", "read_copy")
    for i in range(0, 1000, 2):
        assert requests[i:i + 2] == [
            (i + 40, 8192 + 2 * 65536, 33, (3 * i + 1) * 8192, 4096, "read"),
            (i + 40, 8192 + 2 * 65536 + 4096, 33, (3 * i + 2) * 8192, 4096, "read"),
        ]
    assert backend._signals.store.check(["1/ready/2/0"])
    physical, done, error = backend._ready_entities.get_nowait()
    assert physical == 2 and done.recorded and error is None
    assert backend._signals.store.get("1/ready/2/0") == done.ipc_handle()
    assert backend._ready_entities.empty() and backend._ready_entities.maxsize == 0
    assert calls.index("event_record") < calls.index("poll")
    assert not hasattr(backend, "_send_ticket")


@pytest.mark.parametrize("mode", ["pending", "failure"])
def test_producer_ready_is_record_notification_and_late_io_failure_stays_sticky(monkeypatch, mode):
    backend, plan, calls = _producer_backend(monkeypatch, count=4, mode=mode)
    _produce_test_owned(backend, plan)
    assert "poll" not in calls and "outcome" not in calls
    physical, done, error = backend._ready_entities.get_nowait()
    assert physical == 2 and done.recorded and error is None
    handle = backend._signals.store.get("1/ready/2/0")
    assert handle == done.ipc_handle()
    _complete_test_owned(backend)
    assert backend._signals.store.get("1/ready/2/0") == handle
    assert backend._ready_entities.empty()
    if mode == "failure":
        assert backend._poisoned and backend._prefetch_error is not None
        with pytest.raises(RuntimeError):
            backend._signals.check()
        with pytest.raises(RuntimeError):
            backend.join_prefetch()
    else:
        assert not backend._poisoned
    assert done in backend._ipc_events
    assert backend._owned_work
    assert calls.index("event_record") < calls.index("poll")


@pytest.mark.parametrize("stage", ["preflight", "submit", "record"])
def test_producer_failure_before_record_notification_never_publishes_ready(monkeypatch, stage):
    import torch
    backend, plan, calls = _producer_backend(monkeypatch, count=4)
    def failed(*args, **kwargs):
        raise RuntimeError(f"owned {stage} failed")
    if stage == "preflight":
        backend._validate_plan_entry = failed
    elif stage == "submit":
        backend._store._runtime.submit = failed
    else:
        monkeypatch.setattr(torch.cuda.Event, "record", failed)
    _produce_test_owned(backend, plan)
    assert backend._poisoned
    assert stage in str(backend._prefetch_error)
    assert not backend._signals.store.check(["1/ready/2/0"])
    assert backend._ready_entities.empty()
    assert "poll" not in calls and "outcome" not in calls
    if stage == "record":
        assert backend._owned_work and backend._ipc_events
        done = backend._owned_work[0][1]
        assert done in backend._ipc_events and not done.recorded
    else:
        assert backend._ipc_events == []


def test_empty_owner_publishes_entity_metadata_without_any_read_payload(monkeypatch):
    backend, plan, calls = _producer_backend(monkeypatch, count=1, rank=1)
    _produce_test_owned(backend, plan)
    _complete_test_owned(backend)
    assert not backend._poisoned
    assert not any(isinstance(call, tuple) for call in calls)
    assert backend._signals.store.check(["1/ready/2/1"])
    assert backend._ready_entities.get_nowait()[0] == 2


@pytest.mark.parametrize("detached", [False, True])
def test_begin_prefetch_fences_forward_prefix_and_submits_one_service_job(monkeypatch, detached):
    from contextlib import nullcontext
    from types import SimpleNamespace as NS
    import torch
    import tutti.storage.tutti_nvme.replica_direct as module

    calls, compute_stream = [], object()
    event = NS(record=lambda stream: calls.append(("record_prefix", stream)))
    monkeypatch.setattr(torch.cuda, "Event", lambda *a, **k: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: compute_stream)
    monkeypatch.setattr(module.threading, "Thread", lambda *a, **k: pytest.fail("per-plan thread"))
    job = object()
    def submit(task, priority):
        calls.append(("submit", priority))
        return job
    store = NS(stream_context=lambda direction: nullcontext(), _read_service=NS(submit=submit),
               _read_copy_stream_obj=NS(wait_event=lambda e: calls.append(("wait_prefix", e))))
    backend = _p2p_backend(store)
    backend._geometry = NS(accel_id=0, num_layers=3, blocks_per_chunk=1)
    backend._target_plans["read"] = NS(plan_token=10, entry_for=lambda key: object())
    backend._validate_target_plan = backend._validate_plan_entry = lambda plan: None
    backend.validate_block_tables = lambda tables: tables
    backend._check_plan = lambda *args: calls.append("check_plan")
    backend._build_scatter_plans = lambda layers: calls.append(("plans", layers))
    if detached:
        backend._parent = object()
        backend._compute_prefix = event
    backend.begin_prefetch(keys(2), ((7,), (3,)), (2, 0))
    expected = [] if detached else [("record_prefix", compute_stream)]
    assert calls == expected + ["check_plan", ("wait_prefix", event),
                                ("plans", (2, 0)), ("submit", 1 if detached else 0)]
    assert backend._entity_sequence == (2, 0) and backend._next_entity == 0
    assert backend._prefetch_job is job
    assert backend._prefetch_task.backend is backend


@pytest.mark.parametrize("retry_fails", [False, True])
def test_partial_submit_retries_only_rejected_final_page_and_retains_accepted_io(retry_fails):
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.store import TuttiKVStore

    requests = [(40, 0, 33, 8192 * block, 4096, "read") for block in (1, 7)]
    calls = []
    def submit(batch, **kwargs):
        calls.append(list(batch))
        if len(calls) == 1:
            return NS(io_handle=101, initial_states=(True, False), rejected=(1,),
                      status_ok=False, status_msg="partial submit")
        if retry_fails:
            raise RuntimeError("retry failed")
        return NS(io_handle=102, initial_states=(True,), rejected=(),
                  status_ok=True, status_msg="")
    store = NS(_runtime=NS(submit=submit), _batch_width_limit=lambda: None,
               _accel_id=0, _stream_for=lambda direction: (123, None), _execution="device",
               _deferred_completions=[])
    if retry_fails:
        with pytest.raises(RuntimeError, match="retry failed"):
            TuttiKVStore._submit_retry(store, requests, "read", stream_direction="read_copy")
        assert len(store._deferred_completions) == 1
        completion = store._deferred_completions[0]
        assert completion._handles == [101]
        assert completion._submitted[0].batch_indices == (0,)
    else:
        handles = TuttiKVStore._submit_retry(store, requests, "read", stream_direction="read_copy")
        assert [(h.handle, h.batch_indices) for h in handles] == [(101, (0,)), (102, (1,))]
        assert store._deferred_completions == []
    assert calls == [requests, requests[1:]]


def _scatter_plan_backend(monkeypatch, groups):
    import sys
    from types import ModuleType, SimpleNamespace as NS
    import torch
    import tutti.storage.tutti_nvme.replica_direct as module
    calls, builds = [], []
    original = module.peer_page_rows
    def rows(*args):
        builds.append(args[1])
        return original(*args)
    monkeypatch.setattr(module, "peer_page_rows", rows)
    native = ModuleType("tutti_kv_transfer")
    def plan(peers, offsets, sizes, destination, descriptors):
        calls.append((peers, offsets, sizes, destination, descriptors))
        return NS(close=lambda: None)
    native._native = NS(ReplicaPeerPagePlan=plan)
    monkeypatch.setitem(sys.modules, "tutti_kv_transfer", native)
    backend = _p2p_backend(NS(_replica_groups=groups))
    backend._prefetch_keys = tuple(keys(4))
    backend._peer_tables = {r: ((0, 1), (2, 3), (4, 5), (6, 7)) for r in range(8)}
    backend._raw_layers = tuple(torch.zeros((8, 4 if p % 2 else 8), dtype=torch.uint8)
                                for p in range(len(groups)))
    backend._layer_pages = tuple(NS(page_bytes=t.size(1)) for t in backend._raw_layers)
    backend._peer_layers = {(r, p): (object(), p * 100, t.numel())
                            for p, t in enumerate(backend._raw_layers) for r in range(1, 8)}
    return backend, calls, builds


def test_scatter_descriptors_built_once_per_group_and_bound_to_each_entity(monkeypatch):
    groups = tuple((0, 1) if p % 2 else tuple(range(8)) for p in range(117))
    backend, calls, builds = _scatter_plan_backend(monkeypatch, groups)
    backend._build_scatter_plans(tuple(range(117)))
    assert len(calls) == len(backend._scatter_plans) == 117
    assert len(builds) == 2
    for physical, call in enumerate(calls):
        assert call[3] is backend._raw_layers[physical]
        assert call[1] == [physical * 100] * len(call[0])
        assert call[4] is calls[physical % 2][4]
    assert calls[0][4] is not calls[1][4]
    first = calls[0][4]
    backend._peer_tables = {r: tuple(reversed(t)) for r, t in backend._peer_tables.items()}
    backend._build_scatter_plans(tuple(range(117)))
    assert len(builds) == 4 and calls[117][4] is not first
    assert not first.equal(calls[117][4])


def test_main_entities_do_not_build_peer_scatter_plans(monkeypatch):
    groups = ((0,),) * 60 + (tuple(range(8)),) * 57
    backend, calls, builds = _scatter_plan_backend(monkeypatch, groups)
    backend._build_scatter_plans(tuple(range(117)))
    assert set(backend._scatter_plans) == set(range(60, 117))
    assert len(calls) == 57 and builds == [tuple(range(8))]


def test_main_local_read_returns_plain_completion_without_peer_signals(monkeypatch):
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import _ScatterCompletion
    backend, items, tables, calls = _recovery_backend(monkeypatch)
    backend._store._replica_groups = ((0,),) * 3
    local = NS(fence_event=object())
    backend._local_completions[2] = local
    backend._signals.wait_ready = lambda *args: pytest.fail("main peer ready")
    backend._signals.ack = lambda *args: pytest.fail("main peer ACK")
    backend._signals.wait_acks = lambda *args: pytest.fail("main peer ACK wait")
    backend._scatter_plans.clear()
    _advance_test_scatter(backend)
    assert backend.get_paged_batch(items, 2, tables) is local
    assert not isinstance(local, _ScatterCompletion)
    assert "launch" not in calls and backend._stats["gather_bytes"] == 0


def test_main_owner_filter_keeps_every_rank_local_page():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.layerwise_direct import LayerPageGeometry
    backend = _p2p_backend(NS(_replica_groups=((0,),)))
    backend._layer_pages = (LayerPageGeometry(4096, 4096, 8),)
    backend._layer_tickets = [3]
    backend._geometry.segment_bytes = 8192
    for key in keys(8):
        entry = NS(chunk_id=key, target_offset=4096, target_ticket=7)
        assert backend._page_requests(0, 2, 0, entry, NS(direction="write")) == [
            (7, 4096, 3, 8192, 4096, "write")]


def test_cancelled_service_task_does_not_poison_parent():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import _ReplicaReadTask
    backend = _p2p_backend(NS())
    parent = _p2p_backend(NS())
    backend._parent = parent
    backend._signals = signals()
    backend._cancel_requested.set()
    backend._prefetch_stop.set()
    task = _ReplicaReadTask(backend, NS())
    assert task.advance() is True
    backend._fail(RuntimeError("replica prefetch stopped"))
    assert not backend._poisoned and not parent._poisoned


def test_descriptor_cache_keeps_ordered_members_not_group_cardinality(monkeypatch):
    backend, calls, builds = _scatter_plan_backend(monkeypatch, ((0, 1), (1, 0), (0, 2)))
    backend._build_scatter_plans((0, 1, 2))
    assert len(builds) == 3 and len({id(call[4]) for call in calls}) == 3
    assert not calls[0][4].equal(calls[1][4])


@pytest.mark.parametrize("empty_rows", [False, True])
def test_descriptor_cache_hit_still_validates_entity_source_capacity(monkeypatch, empty_rows):
    backend, calls, builds = _scatter_plan_backend(monkeypatch, ((0, 1), (0, 1)))
    if empty_rows:
        backend._prefetch_keys = (keys(1)[0],)
        backend._peer_tables = {r: ((6, 7),) for r in range(8)}
    mapping, offset, _ = backend._peer_layers[1, 1]
    backend._peer_layers[1, 1] = (mapping, offset, 8)
    with pytest.raises(ValueError, match="out of bounds"):
        backend._build_scatter_plans((0, 1))
    assert len(calls) == len(builds) == 1


def test_producer_preflights_all_entries_but_builds_only_owned_requests(monkeypatch):
    backend, plan, calls = _producer_backend(monkeypatch, count=8)
    backend._entity_sequence = (0, 1, 2)
    backend._wait_event = lambda event: calls.append("event_wait")
    seen = []
    lookup = plan.entry_for
    plan.entry_for = lambda key: (seen.append(key), lookup(key))[1]
    _produce_test_owned(backend, plan)
    assert not backend._poisoned
    assert len(seen) == 8 + 3 * 4
    assert backend._owned_indices[(0, 1)] == (0, 2, 4, 6)
    assert len([call for call in calls if isinstance(call, tuple)]) == 3


def test_main_producer_reads_all_pages_and_never_publishes_peer_ready(monkeypatch):
    backend, plan, calls = _producer_backend(monkeypatch, count=8)
    backend._store._replica_groups = ((0,),) * 3
    backend._signals.ready = lambda *args: pytest.fail("main must not publish peer ready")
    _produce_test_owned(backend, plan)
    assert not backend._poisoned
    batches = [call for call in calls if isinstance(call, tuple)]
    assert len(batches) == 1 and len(batches[0][1]) == 16
    assert backend._owned_indices[(0,)] == tuple(range(8))
    assert backend._local_completions[2].fence_event is backend._test_events[0]
    assert backend._test_events[0].options["interprocess"] is False


def test_producer_publishes_all_entity_events_before_polling_prior_completion(monkeypatch):
    backend, plan, calls = _producer_backend(monkeypatch, count=8)
    backend._entity_sequence = (2, 0, 1)
    _produce_test_owned(backend, plan)
    assert len([c for c in calls if isinstance(c, tuple)]) == 3
    assert "poll" not in calls and "outcome" not in calls
    events = backend._test_events
    assert len(events) == 3 and len({id(e) for e in events}) == 3
    ready = [backend._ready_entities.get_nowait() for _ in range(3)]
    assert [item[0] for item in ready] == [2, 0, 1]
    assert [item[1] for item in ready] == events
    assert all(item[2] is None for item in ready)
    for physical, event, _ in ready:
        assert backend._signals.store.get(f"1/ready/{physical}/0") == event.ipc_handle()
    assert all(event.ready is None for event in events)
    assert events == backend._ipc_events
    _complete_test_owned(backend)
    assert backend._ready_entities.empty()
    assert backend._stats["read_bytes"] == 3 * 4096
    assert calls.count("poll") == calls.count("outcome") == 3


def test_submit_backpressure_progresses_accepted_handles_inside_entity():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.store import TuttiKVStore
    requests = [(1, 0, 2, block * 4096, 4096, "read") for block in range(3)]
    calls, progress = [], []
    def submit(batch, **kwargs):
        calls.append(list(batch))
        if len(calls) == 1:
            return NS(io_handle=11, initial_states=(True,), rejected=(), status_ok=True, status_msg="")
        if len(calls) in (2, 3, 4):
            return NS(io_handle=None, initial_states=(False,), rejected=(0,), status_ok=False,
                      status_msg="RESOURCE_EXHAUSTED: in-flight operation capacity exhausted")
        return NS(io_handle=10 + len(calls), initial_states=(True,), rejected=(), status_ok=True, status_msg="")
    store = NS(_runtime=NS(submit=submit), _batch_width_limit=lambda: 1, _accel_id=0,
               _stream_for=lambda direction: (123, None), _execution="device", _deferred_completions=[])
    handles = TuttiKVStore._submit_retry(store, requests, "read", stream_direction="read_copy",
        on_backpressure=lambda accepted: progress.append(tuple(h.handle for h in accepted)))
    assert progress == [(11,)] * 3
    assert [h.batch_indices for h in handles] == [(0,), (1,), (2,)]
    assert calls == [requests[:1], requests[1:2], requests[1:2], requests[1:2], requests[1:2], requests[2:]]
    assert not store._deferred_completions


def test_submit_backpressure_exception_retains_single_owner_of_accepted_handles():
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.store import TuttiKVStore
    calls = []
    def submit(batch, **kwargs):
        calls.append(1)
        return (NS(io_handle=11, initial_states=(True,), rejected=(), status_ok=True, status_msg="")
                if len(calls) == 1 else
                NS(io_handle=None, initial_states=(False,), rejected=(0,), status_ok=False,
                   status_msg="in-flight operation capacity exhausted"))
    store = NS(_runtime=NS(submit=submit), _batch_width_limit=lambda: 1, _accel_id=0,
               _stream_for=lambda direction: (123, None), _execution="device", _deferred_completions=[])
    def stopped(handles):
        raise TimeoutError("submit capacity timed out")
    with pytest.raises(TimeoutError, match="capacity timed out"):
        TuttiKVStore._submit_retry(store, [(1,), (2,)], "read", on_backpressure=stopped)
    assert len(store._deferred_completions) == 1
    assert store._deferred_completions[0]._handles == [11]


def test_capacity_yields_after_progressing_accepted_io_and_tracked_write():
    import threading
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import _ReplicaSubmit
    from tutti.storage.tutti_nvme.store import _SubmittedHandle
    calls = []
    def submit(*args, **kwargs):
        calls.append("submit")
        return NS(io_handle=None, initial_states=(False,), rejected=(0,), status_ok=False,
                  status_msg="in-flight operation capacity exhausted")
    store = NS(_runtime=NS(submit=submit, query_io=lambda h: calls.append(("query", h))),
               _batch_width_limit=lambda: None, _accel_id=0, _execution="device",
               _stream_for=lambda direction: (123, None),
               _inflight_lock=threading.Lock())
    class PendingWrite:
        def poll(self):
            calls.append("write_progress")
    store._inflight_by_chunk = {b"x": {PendingWrite()}}
    task = _ReplicaSubmit(store, [(1,)], "read_copy")
    task.handles = [_SubmittedHandle(11)]
    assert task.advance() is False
    task.handles.clear()
    assert task.advance() is False
    assert calls == ["submit", ("query", 11), "write_progress", "submit", "write_progress"]


def test_join_and_stop_wait_single_service_job():
    from types import SimpleNamespace as NS
    calls = []
    backend = _p2p_backend(NS(_read_service=NS(wake=lambda: calls.append("wake"))))
    backend._active_token = 1
    backend._entity_sequence = (0,)
    backend._next_entity = 1
    backend._prefetch_job = NS(join=lambda timeout: calls.append("job"), is_alive=lambda: False)
    backend.join_prefetch()
    assert calls == ["job"]
    backend.stop_prefetch()
    assert calls == ["job", "wake", "job"]
    assert backend._prefetch_job is None
    assert not backend._poisoned


def _service_task_backend(monkeypatch, background=False):
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import _ReplicaReadTask
    from tutti.index.chunk_index import derive_io_key
    backend, plan, calls = _producer_backend(monkeypatch, count=2)
    backend._prefetch_keys = tuple(key[:16] for key in backend._prefetch_keys)
    lookup = plan.entry_for
    plan.entry_for = lambda key: lookup(key + b"x" * 16)
    backend._store._replica_groups = ((0,),) * 3
    backend._store._read_async_stream_obj = backend._store._read_copy_stream_obj
    backend._entity_sequence = (2, 0, 1)
    backend._signals = signals(ranks=(0,))
    plan.plan_token = backend._active_token = 1
    backend._target_plans["read"] = plan
    backend._validate_target_plan = lambda plan: None
    backend.validate_block_tables = lambda tables: tables
    backend._store._live = {derive_io_key(key, p) for key in backend._prefetch_keys for p in range(3)}
    backend._prefetch_job = NS(join=lambda **k: None, is_alive=lambda: False)
    cq_ready, window = [False], [(1, 0)]
    own = backend._own_completion
    def completion(handles):
        result = own(handles)
        result.poll = lambda: True if cq_ready[0] else None
        return result
    backend._own_completion = completion
    submit = backend._advance_submit
    def advance(state):
        submit(state)
        if "io_done" in state:
            state["io_done"].ready = False
    backend._advance_submit = advance
    if background:
        backend._parent = NS(_signals=signals(ranks=(0,)))
        backend._read_direction = backend._io_direction = "read_async"
        backend._background_allowed = lambda: window[0]
    task = backend._prefetch_task = _ReplicaReadTask(backend, plan)
    return backend, task, calls, cq_ready, window


def test_foreground_task_submits_every_entity_while_owned_cq_is_pending(monkeypatch):
    backend, task, calls, cq, window = _service_task_backend(monkeypatch)
    assert [task.advance() for _ in range(3)] == [False, False, False]
    assert task.produced == 3 and backend._next_entity == 3
    assert len([item for item in calls if isinstance(item, tuple)]) == 3
    assert all(state["phase"] == "fence" for state in task.states)
    cq[0] = True
    assert task.advance() is True
    assert backend._stats["read_bytes"] == 3 * 4096


def test_background_task_finishes_batch_cq_and_fence_before_next_window(monkeypatch):
    backend, task, calls, cq, window = _service_task_backend(monkeypatch, background=True)
    assert task.advance() is False
    assert task.produced == 1
    window[0] = None
    assert task.advance() is False
    assert task.produced == 1
    cq[0] = True
    window[0] = (2, 0)
    assert task.advance() is False
    assert task.produced == 1 and not task.states[0].get("finished", False)
    task.states[0]["ready"].ready = True
    assert task.advance() is False
    assert task.states[0]["finished"] and task.produced == 1
    def repeated_query():
        raise AssertionError("finished historical batch must not be queried again")
    task.states[0]["ready"].query = repeated_query
    assert task.advance() is False
    assert task.produced == 2
    assert len([item for item in calls if isinstance(item, tuple)]) == 2


def test_try_get_returns_none_until_fence_registered_and_does_not_consume_callback(monkeypatch):
    backend, task, calls, cq, window = _service_task_backend(monkeypatch)
    args = backend._prefetch_keys, 2, backend._prefetch_tables
    assert backend.try_get_paged_batch(*args) is None
    assert backend.try_get_paged_batch(*args) is None
    assert backend._callback_entity == 0
    task.advance()
    assert backend.try_get_paged_batch(*args) is backend._local_completions[2]
    assert backend._callback_entity == 1
    with pytest.raises(RuntimeError, match="duplicated or out of order"):
        backend.try_get_paged_batch(*args)


def test_started_background_scatter_finishes_after_prefill_window_closes(monkeypatch):
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import _ReplicaReadTask
    backend, items, tables, calls = _recovery_backend(monkeypatch)
    _advance_test_scatter(backend)
    state = backend._prefetch_task.states[0]
    assert state["phase"] == "ack"
    backend._signals.store.set("1/ack/2/1", _event_handle("ack", 2, 1))
    backend._parent = NS(_signals=signals(ranks=(0,)))
    backend._store.record_read_async_event = backend._store.record_read_event
    backend._background_allowed = lambda: None
    monkeypatch.setattr(type(backend._test_events[0]), "query", lambda self: False)
    task = backend._prefetch_task = _ReplicaReadTask(backend, backend._target_plans["read"])
    task.states, task.produced = [state], 1
    assert task.advance() is False
    assert state["phase"] == "fence" and backend._next_entity == 1
    assert calls.count("launch") == 1
    assert backend.try_get_paged_batch(items, 2, tables).fence_event is state["ready"]


@pytest.mark.parametrize("cancel", [False, True])
def test_partial_task_failure_or_cancellation_retains_accepted_dma(monkeypatch, cancel):
    from tutti.storage.tutti_nvme.replica_direct import _ReplicaReadTask
    from tutti.storage.tutti_nvme.store import _SubmittedHandle
    backend, plan, calls = _producer_backend(monkeypatch, count=2)
    task = _ReplicaReadTask(backend, plan)
    state = backend._new_entity(plan, 2)
    state["submit"].handles = [_SubmittedHandle(17, (0,))]
    task.states = [state]
    if cancel:
        backend._cancel_requested.set()
        backend._prefetch_stop.set()
        assert task.advance() is True
        assert not backend._poisoned
    else:
        task.fail(RuntimeError("retry failed"))
        assert task.advance() is True
        assert backend._poisoned
    assert len(backend._owned_work) == 1
    assert state["completion"] is backend._owned_work[0][0]
    assert state["submit"].handles[0].handle == 17
    assert "poll" not in calls


@pytest.mark.parametrize("status", [None, True, False, "observation_error"])
def test_scatter_completion_confirms_owned_io_and_records_late_failure(status):
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import _ScatterCompletion
    failed = []
    error = RuntimeError("owned observation failed") if status == "observation_error" else None
    owned = NS(poll=lambda: None if error else status, observation_error=error,
               wait_result=lambda *args: NS(ok=False, message="owned failed"))
    completion = _ScatterCompletion(NS(), NS(query=lambda: True), owned_io=owned,
                                    on_failure=failed.append)
    if status is False or error:
        with pytest.raises(RuntimeError):
            completion.poll()
        assert len(failed) == 1
    else:
        assert completion.poll() is status
        assert failed == []


def _initializing_backend():
    from contextlib import nullcontext
    from types import SimpleNamespace as NS
    backend = _p2p_backend(NS(_replica_groups=((0, 1),), stream_context=lambda direction: nullcontext()))
    backend._signals = signals(epoch=0)
    backend._epoch = 1
    backend._geometry.segment_bytes = 4096
    backend._prefetch_keys = (keys(1)[0][:16],)
    backend._prefetch_tables = ((7,),)
    backend._entity_sequence = (0,)
    backend._signals.wait = lambda *a, **k: pytest.fail("service must not wait notifications")
    backend._signals.begin = lambda *a, **k: pytest.fail("service must not use blocking begin")
    built = []
    backend._build_scatter_plans = lambda layers: built.append(tuple(layers))
    backend._check_plan(backend._prefetch_keys, backend._prefetch_tables, (0,), deferred=True)
    return backend, built


def test_detached_init_publishes_then_checks_peer_plan_and_tables_without_wait():
    import struct
    backend, built = _initializing_backend()
    assert backend._initializing
    assert backend._signals.epoch == 1
    assert backend._advance_plan_init() is False
    assert built == []
    backend._signals.store.set("1/plan/1", backend._init_descriptor)
    assert backend._advance_plan_init() is False
    backend._signals.store.set("1/blocks/1", struct.pack("<q", 3))
    assert backend._advance_plan_init() is True
    assert not backend._initializing
    assert backend._peer_tables == {0: ((7,),), 1: ((3,),)}
    assert built == [(0,)]


@pytest.mark.parametrize("mode", ["descriptor", "short_table", "negative_table"])
def test_detached_init_rejects_peer_mismatch_before_scatter_or_dma(mode):
    import struct
    backend, built = _initializing_backend()
    backend._signals.store.set("1/plan/1", b"different" if mode == "descriptor" else backend._init_descriptor)
    blob = b"short" if mode == "short_table" else struct.pack("<q", -1 if mode == "negative_table" else 3)
    backend._signals.store.set("1/blocks/1", blob)
    with pytest.raises((ValueError, RuntimeError)):
        backend._advance_plan_init()
    assert built == [] and backend._owned_work == []


def test_start_detached_read_uses_nonblocking_init_and_keeps_prepared_prefix():
    from types import SimpleNamespace as NS
    backend = _p2p_backend(NS())
    calls, prefix = [], object()
    reader = NS(_detached_args=((b"key",), ((7,),), (0,)), _compute_prefix=prefix,
                begin_prefetch=lambda *a, **k: calls.append((a, k)))
    plan = NS(plan_token=1)
    backend._detached_readers[1] = reader
    backend.start_detached_read(plan)
    assert calls == [(reader._detached_args, {"deferred_init": True})]
    assert reader._compute_prefix is prefix


def test_background_same_epoch_can_admit_next_batch_after_full_completion(monkeypatch):
    backend, task, calls, cq, window = _service_task_backend(monkeypatch, background=True)
    cq[0] = True
    assert task.advance() is False
    task.states[0]["ready"].ready = True
    assert task.advance() is False
    assert task.states[0]["finished"]
    assert task.advance() is False
    assert task.produced == 2
    assert window[0] == (1, 0)


def test_background_ready_waits_next_window_before_launching_new_scatter(monkeypatch):
    from types import SimpleNamespace as NS
    backend, items, tables, calls = _recovery_backend(monkeypatch)
    window = [None]
    backend._parent = NS(_signals=signals(ranks=(0,)))
    backend._background_allowed = lambda: window[0]
    backend._store.record_read_async_event = backend._store.record_read_event
    _advance_test_scatter(backend)
    state = backend._prefetch_task.states[0]
    assert state["phase"] == "ready" and "launch" not in calls
    backend._signals.store.set("1/ready/2/1", _event_handle("ready", 2, 1))
    backend._advance_scatter(state)
    assert state["phase"] == "ready" and "launch" not in calls
    window[0] = (1, 0)
    backend._advance_scatter(state)
    assert state["phase"] == "ack" and calls.count("launch") == 1
    window[0] = None
    backend._signals.store.set("1/ack/2/1", _event_handle("ack", 2, 1))
    backend._advance_scatter(state)
    assert state["phase"] == "fence" and calls.count("launch") == 1


def test_background_scatter_checks_scope_even_when_token_still_visible(monkeypatch):
    from contextlib import nullcontext
    backend, items, tables, calls = _recovery_backend(monkeypatch)
    backend._background_allowed = lambda: (1, 0)
    backend._background_scope = lambda: nullcontext(False)
    backend._signals.store.set("1/ready/2/1", _event_handle("ready", 2, 1))
    _advance_test_scatter(backend)
    assert backend._prefetch_task.states[0]["phase"] == "ready"
    assert "launch" not in calls


def _real_segment_task(monkeypatch, *, partial=False):
    from types import SimpleNamespace as NS
    import tutti.storage.tutti_nvme.replica_direct as module
    from tutti.storage.tutti_nvme.store import _TuttiCompletion
    backend, task, calls, cq, window = _service_task_backend(monkeypatch, background=True)
    monkeypatch.setattr(module, "_TuttiCompletion", _TuttiCompletion)
    backend._prefetch_keys = backend._prefetch_keys[:1]
    backend._prefetch_tables = backend._prefetch_tables[:1]
    backend._own_completion = lambda handles: module.TuttiReplicaDirectBackend._own_completion(backend, handles)
    backend._entity_sequence = (2,)
    if not partial:
        backend._store._batch_width_limit = lambda: 1
    active, submitted, released = {}, [], []
    ready = [False]
    serial = [10]
    def submit(requests, **kwargs):
        submitted.append(tuple(requests))
        if active:
            return NS(io_handle=None, initial_states=(False,) * len(requests),
                      rejected=tuple(range(len(requests))), status_ok=False,
                      status_msg="in-flight operation capacity exhausted")
        serial[0] += 1
        handle = serial[0]
        active[handle] = tuple(requests[:1])
        return NS(io_handle=handle, initial_states=(True,) + (False,) * (len(requests) - 1),
                  rejected=tuple(range(1, len(requests))), status_ok=len(requests) == 1,
                  status_msg="partial" if len(requests) > 1 else "")
    def query(handle):
        assert handle in active, "released handle was queried"
        return "COMPLETED" if ready[0] else "IN_FLIGHT"
    def observe(handle, timeout):
        assert timeout == 0 and handle in active
        return NS(observation="OK", state="COMPLETED", confirmed_bytes=4096,
                  timeout_seen=False, failed_request_indices=(), failure_scope="NONE",
                  failure_kind=None, first_failed_entry=None, raw_cq_status=None, message="")
    def release(handle):
        assert handle in active, "double release"
        released.append(handle)
        del active[handle]
    backend._store._runtime = NS(submit=submit, query_io=query, wait_result=observe, release_io=release)
    return backend, task, window, ready, submitted, released, active


@pytest.mark.parametrize("partial", [False, True])
def test_partial_or_segment_retry_pauses_window_and_releases_accepted_before_resume(monkeypatch, partial):
    backend, task, window, ready, submitted, released, active = _real_segment_task(monkeypatch, partial=partial)
    assert task.advance() is False
    state = task.states[0]
    assert state["phase"] == "submit" and len(state["parts"]) == 1
    assert len(active) == 1 and task.needs_poll() is True
    window[0] = None
    assert task.advance() is False
    assert len(submitted) == 1 and released == []
    ready[0] = True
    assert task.advance() is False
    assert len(submitted) == 1 and released == [11] and active == {}
    assert task.needs_poll() is False
    window[0] = (1, 0)
    assert task.advance() is False
    assert state["phase"] == "fence" and released == [11, 12] and active == {}
    state["ready"].ready = True
    assert task.advance() is True
    outcome = state["completion"].wait_result(0)
    assert outcome.ok and sum(item.confirmed_bytes for item in outcome.results) == 8192
    assert {item.batch_indices for item in outcome.results} == {(0,), (1,)}
    assert released == [11, 12]
    assert backend._stats["read_bytes"] == 8192


def test_segmented_batch_with_one_terminal_slot_releases_before_next_submission(monkeypatch):
    backend, task, window, ready, submitted, released, active = _real_segment_task(monkeypatch)
    ready[0] = True
    assert task.advance() is False
    assert released == [11] and active == {}
    assert task.advance() is False
    assert len(submitted) == 2 and released == [11, 12] and active == {}
    task.states[0]["ready"].ready = True
    assert task.advance() is True
    assert task.states[0]["completion"].poll() is True
    assert released == [11, 12]


def test_background_submit_scope_blocks_partial_retry_with_visible_window(monkeypatch):
    from contextlib import nullcontext
    backend, task, window, ready, submitted, released, active = _real_segment_task(monkeypatch, partial=True)
    task.advance()
    backend._background_scope = lambda: nullcontext(False)
    ready[0] = True
    assert task.advance() is False
    assert len(submitted) == 1 and released == [11]
    backend._background_scope = lambda: nullcontext(True)
    task.advance()
    assert len(submitted) == 2 and released == [11, 12]


@pytest.mark.parametrize("phase", ["submit", "ready", "ack", "init"])
def test_background_long_pause_excluded_from_timeout_budget(monkeypatch, phase):
    from types import SimpleNamespace as NS
    import tutti.storage.tutti_nvme.replica_direct as module
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    backend, task, calls, cq, window = _service_task_backend(monkeypatch, background=True)
    state = backend._new_entity(task.plan, 2)
    task.states = [state]
    state["phase"] = phase
    if phase == "init":
        backend._initializing = True
        backend._init_deadline = module._POLL_SECONDS
    cq[0] = True
    now[0] = 1.0
    window[0] = None
    task.needs_poll()
    now[0] = 1000.0
    window[0] = (1, 0)
    task._tick_window()
    deadline = backend._init_deadline if phase == "init" else state["deadline"]
    assert deadline > now[0]
    assert not backend._poisoned
    now[0] += module._POLL_SECONDS + 1
    task._tick_window()
    deadline = backend._init_deadline if phase == "init" else state["deadline"]
    assert deadline < now[0]


def test_paused_peer_ready_without_cq_work_sleeps_but_ack_and_fence_are_polled(monkeypatch):
    backend, task, calls, cq, window = _service_task_backend(monkeypatch, background=True)
    window[0] = None
    assert task.needs_poll() is False
    state = backend._new_entity(task.plan, 2)
    state["phase"], state["cq_done"] = "ready", True
    task.states = [state]
    assert task.needs_poll() is False
    state["phase"] = "ack"
    assert task.needs_poll() is True
    state["phase"] = "fence"
    assert task.needs_poll() is True
    state["finished"] = True
    assert task.needs_poll() is False


def test_real_progress_wakes_service_but_inflight_poll_does_not(monkeypatch):
    from types import SimpleNamespace as NS
    backend, task, calls, cq, window = _service_task_backend(monkeypatch, background=True)
    wakes = []
    backend._store._read_service = NS(wake=lambda: wakes.append(True))
    task.advance()
    assert wakes == [True]
    task.advance()
    assert wakes == [True]
    cq[0] = True
    task.states[0]["ready"].ready = True
    task.advance()
    assert wakes == [True, True]


def test_read_service_foreground_progresses_while_replica_background_waits_cq(monkeypatch):
    from types import SimpleNamespace as NS
    from tutti.engine.read_service import ReadService
    monkeypatch.setattr(ReadService, "_bind_device", lambda self: None)
    backend, task, calls, cq, window = _service_task_backend(monkeypatch, background=True)
    task.advance()
    assert task.produced == 1 and not task.states[0].get("cq_done")
    service = ReadService(backend._store)
    backend._store._read_service = service
    background = service.submit(task, priority=1)
    seen = []
    foreground = service.submit(NS(advance=lambda: seen.append("front") or True,
                                   fail=lambda exc: pytest.fail(str(exc))), priority=0)
    try:
        foreground.join(1)
        assert not foreground.is_alive() and foreground.error is None
        assert seen == ["front"] and background.is_alive()
        assert task.produced == 1
    finally:
        backend._cancel_requested.set()
        backend._prefetch_stop.set()
        service.wake()
        background.join(1)
        service.close(timeout=1)
    assert background.error is None and not backend._poisoned


@pytest.mark.parametrize("failed", [False, True])
def test_cancel_retains_accepted_dma_but_real_failed_cq_still_poisons_parent(monkeypatch, failed):
    from types import SimpleNamespace as NS
    backend, task, window, ready, submitted, released, active = _real_segment_task(monkeypatch, partial=True)
    parent = _p2p_backend(NS())
    parent._signals = signals(ranks=(0,))
    backend._parent = parent
    task.advance()
    part = task.states[0]["parts"][0]
    backend._cancel_requested.set()
    backend._prefetch_stop.set()
    assert task.advance() is True
    assert not backend._poisoned and len(active) == 1 and released == []
    observe = backend._store._runtime.wait_result
    def result(handle, timeout):
        outcome = observe(handle, 0)
        if failed:
            outcome.state, outcome.failure_scope = "FAILED", "WHOLE_OPERATION"
            outcome.failure_kind, outcome.message = "NVME_CQ", "device CQ failed"
        return outcome
    backend._store._runtime.wait_result = result
    ready[0] = True
    backend._store.record_read_async_event = lambda: None
    backend._wait_event = lambda *a, **k: None
    if failed:
        with pytest.raises(RuntimeError, match="device CQ failed"):
            backend._quiesce()
    else:
        backend._quiesce()
    assert backend._poisoned is failed and parent._poisoned is failed
    assert released == [11] and active == {}
    if failed:
        assert "device CQ failed" in str(backend._prefetch_error)
    else:
        assert part.wait_result(0).ok


@pytest.mark.parametrize("phase", ["ready", "ack"])
def test_peer_notification_resumes_after_long_closed_window_without_poison(monkeypatch, phase):
    from types import SimpleNamespace as NS
    import tutti.storage.tutti_nvme.replica_direct as module
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    backend, items, tables, calls = _recovery_backend(monkeypatch)
    window = [(1, 0)]
    backend._parent = NS(_signals=signals(ranks=(0,)))
    backend._background_allowed = lambda: window[0]
    backend._store.record_read_async_event = backend._store.record_read_event
    _advance_test_scatter(backend)
    state = backend._prefetch_task.states[0]
    assert state["phase"] == "ack"
    monkeypatch.setattr(type(backend._test_events[0]), "query", lambda self: False)
    if phase == "ready":
        state["phase"] = "ready"
        del backend._signals.store.data["1/ready/2/1"]
    task = module._ReplicaReadTask(backend, backend._target_plans["read"])
    task.states, task.produced = [state], 1
    window[0] = None
    task.needs_poll()
    now[0] = 1000.0
    assert task.advance() is False
    assert state["deadline"] > now[0] and not backend._poisoned
    backend._signals.store.set("1/ready/2/1", _event_handle("ready", 2, 1))
    backend._signals.store.set("1/ack/2/1", _event_handle("ack", 2, 1))
    window[0] = (1, 0)
    task.advance()
    assert state["phase"] == "fence" and not backend._poisoned


@pytest.mark.parametrize("failed", [False, True])
def test_segment_batch_preserves_terminal_and_done_callbacks_after_child_release(failed):
    from types import SimpleNamespace as NS
    from tutti.storage.tutti_nvme.replica_direct import _ReplicaBatchCompletion
    from tutti.storage.tutti_nvme.store import _SubmittedHandle, _TuttiCompletion
    second_ready, released = [False], []
    def query(handle):
        return "IN_FLIGHT" if handle == 12 and not second_ready[0] else "COMPLETED"
    def observe(handle, timeout):
        bad = failed and handle == 12
        return NS(observation="OK", state="FAILED" if bad else "COMPLETED", confirmed_bytes=4096,
                  timeout_seen=False, failed_request_indices=(0,) if bad else (),
                  failure_scope="REQUEST_INDICES" if bad else "NONE",
                  failure_kind="NVME_CQ" if bad else None, first_failed_entry=None,
                  raw_cq_status=None, message="failed segment" if bad else "")
    runtime = NS(query_io=query, wait_result=observe, release_io=released.append)
    parts = [_TuttiCompletion(runtime, [_SubmittedHandle(handle, (index,))], lambda ok: None,
                              auto_watch=False, retain_observation_errors=True)
             for handle, index in ((11, 0), (12, 3))]
    batch = _ReplicaBatchCompletion(runtime, parts)
    terminal, done = [], []
    batch.add_terminal_callback(terminal.append)
    batch.add_done_callback(done.append)
    assert batch.poll() is None
    assert released == [11] and terminal == done == []
    second_ready[0] = True
    assert batch.poll() is (not failed)
    assert len(terminal) == len(done) == 1
    assert done[0].ok is (not failed)
    assert done[0].failed_batch_indices == ((3,) if failed else ())
    assert {result.batch_indices for result in done[0].results} == {(0,), (3,)}
    assert batch.poll() is (not failed)
    assert batch.wait_result(0) is done[0]
    assert released == [11, 12] and len(terminal) == len(done) == 1


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_group_rejected(size):
    with pytest.raises(ValueError):
        owner_index(keys(1)[0], size)

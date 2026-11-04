"""Opt-in CUDA IPC protocol tests without models, NVMe or NCCL."""
import os
import queue
import threading
import time
from datetime import timedelta

import pytest


def _ipc_worker(rank, world, port, output):
    import torch
    import torch.distributed as dist
    import tutti.storage.tutti_nvme.replica_direct as replica
    from tutti_kv_transfer import _native

    replica._POLL_SECONDS = 10.0
    torch.cuda.set_device(rank)
    store = dist.TCPStore("127.0.0.1", port, world + 1, False,
                          timeout=timedelta(seconds=15))
    signals = replica._ReplicaSignals(store, rank, range(world))
    import json
    page_bytes = (4096, 17, 8192)
    pool = torch.empty(65539 + sum(12 * p for p in page_bytes), dtype=torch.uint8,
                       device=f"cuda:{rank}")
    tensors, exports, start = [], [], 65539
    for size in page_bytes:
        tensor = pool[start:start + 12 * size].view(12, size)
        tensors.append(tensor)
        exports.append(_native.ReplicaIpcExport(tensor))
        start += tensor.numel()
    assert len({export.handle() for export in exports}) == 1
    assert exports[0].offset > 65536
    producer_stream = torch.cuda.Stream(device=rank)
    copy_stream = torch.cuda.Stream(device=rank)
    compute_stream = torch.cuda.default_stream(rank)
    keys = [i.to_bytes(16, "little") for i in range(4)]
    tables = {r: tuple((4 + (2 * i + r) % 8, 4 + (2 * i + 1 + r) % 8)
                       for i in range(4)) for r in range(world)}
    pair = (rank // 2 * 2, rank // 2 * 2 + 1)
    groups = (pair, pair, tuple(range(world)))
    store.set(f"handle/{rank}", exports[0].handle())
    store.set(f"views/{rank}", json.dumps([(e.offset, t.numel(), e.allocation_bytes)
                                          for e, t in zip(exports, tensors)]))
    signals.wait([f"handle/{r}" for r in range(world)])
    views = {r: json.loads(store.get(f"views/{r}")) for r in range(world)}
    mappings = {r: _native.ReplicaIpcMapping(store.get(f"handle/{r}"), 0,
                                          views[r][0][2], rank)
                for r in range(world) if r != rank}
    plans = []
    for entity, members in enumerate(groups):
        peers = [r for r in members if r != rank]
        rows = replica.peer_page_rows(keys, members, rank, tables)
        with torch.cuda.stream(copy_stream):
            plans.append(_native.ReplicaPeerPagePlan(
                [mappings[r] for r in peers], [views[r][entity][0] for r in peers],
                [views[r][entity][1] for r in peers], tensors[entity],
                torch.tensor(rows, dtype=torch.int64, device="cpu").reshape(-1, 3)))
    verified, aborted = 0, False

    def wait_event(event):
        deadline = time.monotonic() + 10.0
        while not event.query():
            if time.monotonic() > deadline:
                raise TimeoutError("IPC device fence did not complete")
            time.sleep(0.0001)

    try:
        for epoch in range(1, 4):
            for tensor in tensors:
                tensor.fill_(255)
            compute_stream.synchronize()
            signals.begin(epoch, b"fixed-direct-pages-geometry")
            stop, errors = threading.Event(), []

            def producer():
                try:
                    torch.cuda.set_device(rank)
                    for entity, members in enumerate(groups):
                        if stop.is_set():
                            return
                        if epoch == 3 and rank == world - 1 and entity == 2:
                            raise RuntimeError("injected owned IO failure")
                        if epoch == 2 and rank == 1 and entity == 1:
                            time.sleep(0.01)
                        with torch.cuda.stream(producer_stream):
                            head = rank // 2 if entity != 2 else 0
                            for i, key in enumerate(keys):
                                if members[replica.owner_index(key, len(members))] == rank:
                                    for page, block in enumerate(tables[rank][i]):
                                        tensors[entity][block].fill_(
                                            (epoch * 41 + entity * 53 + i * 7 + page * 11 + head) % 256)
                            event = torch.cuda.Event()
                            event.record(producer_stream)
                        wait_event(event)
                        signals.ready(entity)
                except Exception as exc:
                    errors.append(exc)
                    signals.fail(exc)

            thread = threading.Thread(target=producer)
            thread.start()
            try:
                for entity, members in enumerate(groups):
                    signals.wait_ready(entity, members, stop)
                    with torch.cuda.stream(copy_stream):
                        plans[entity].launch()
                        done = torch.cuda.Event()
                        done.record(copy_stream)
                    wait_event(done)
                    signals.ack(entity)
                    signals.wait_acks(entity, stop, members)
                    head = rank // 2 if entity != 2 else 0
                    for i, blocks in enumerate(tables[rank]):
                        for page, block in enumerate(blocks):
                            expected = (epoch * 41 + entity * 53 + i * 7 + page * 11 + head) % 256
                            assert bool((tensors[entity][block] == expected).all()), (
                                rank, epoch, entity, i, page)
                    assert bool((tensors[entity][:4] == 255).all())
                    verified += 1
            except RuntimeError as exc:
                if epoch != 3 or "injected owned IO failure" not in str(exc):
                    raise
                aborted = True
            finally:
                stop.set()
                thread.join(timeout=12)
                assert not thread.is_alive(), "producer did not stop"
                producer_stream.synchronize()
                copy_stream.synchronize()
                signals.quiescent()
            if epoch < 3:
                assert not errors and not aborted
        assert aborted and verified >= 6
        with pytest.raises(RuntimeError, match="live scatter plans"):
            mappings[(rank + 1) % world].close()
        peers = [r for r in groups[0] if r != rank]
        for bad in ([[0, -1, 0]], [[0, 12, 0]], [[len(peers), 0, 0]], [[0, 0, 12]],
                    [[0, 0, 0], [0, 1, 0]]):
            with pytest.raises(RuntimeError, match="out of bounds|overlap"):
                _native.ReplicaPeerPagePlan([mappings[r] for r in peers],
                    [views[r][0][0] for r in peers], [views[r][0][1] for r in peers],
                    tensors[0], torch.tensor(bad, dtype=torch.int64, device="cpu"))
        with torch.cuda.stream(compute_stream):
            torch.cuda._sleep(100_000_000)
        with torch.cuda.stream(copy_stream):
            plans[0].launch()
            done = torch.cuda.Event()
            done.record(copy_stream)
        wait_event(done)
        compute_stream.synchronize()
        peer = peers[0]
        small = torch.full((8, 17), 93, dtype=torch.uint8, device=f"cuda:{rank}")
        compute_stream.synchronize()
        with torch.cuda.stream(copy_stream):
            tail = _native.ReplicaPeerPagePlan([mappings[peer]], [views[peer][1][0]],
                [views[peer][1][1]], small[1:], torch.tensor([[0, 0, 5]], dtype=torch.int64))
            torch.cuda._sleep(100_000_000)
            tail.launch()
        with pytest.raises(RuntimeError, match="busy"):
            tail.close()
        copy_stream.synchronize()
        assert bool((small[6] == 255).all())
        assert bool((small[:6] == 93).all()) and bool((small[7] == 93).all())
        tail.close()
        signals.closing()
        for plan in plans:
            plan.close()
            plan.close()
            with pytest.raises(RuntimeError, match="closed"):
                plan.launch()
        for mapping in mappings.values():
            mapping.close()
        mappings.clear()
        signals.release()
        for export in exports:
            export.close()
            export.close()
            with pytest.raises(RuntimeError, match="closed"):
                export.handle()
        output.put((rank, verified, aborted, None))
    except Exception as exc:
        signals.fail(exc)
        output.put((rank, verified, aborted, repr(exc)))
        raise


def _nvme_worker(rank, world, port, output):
    from pathlib import Path
    import torch
    import torch.distributed as dist
    import tutti.storage.tutti_nvme.replica_direct as replica
    from tutti.storage.tutti_nvme.store import TuttiKVStore
    from tutti.engine.read_service import ReadService

    replica._POLL_SECONDS = 15.0
    torch.cuda.set_device(rank)
    rendezvous = dist.TCPStore("127.0.0.1", port, world + 1, False,
                               timeout=timedelta(seconds=20))
    dist.init_process_group("gloo", store=dist.PrefixStore("world/", rendezvous),
                            rank=rank, world_size=world, timeout=timedelta(seconds=20))
    root = Path(os.environ["TUTTI_TEST_REPLICA_NVME_ROOT"])
    mounts = (Path("/mnt/tutti_md0"), Path("/mnt/nvme4"))
    ids = range(4) if rank < 4 else range(4, 8)
    mount = mounts[rank // 4]
    preset = dict(type="raid0", gpu_id=rank, num_queues=2,
                  max_batch_entries=4096, max_in_flight_operations=8,
                  daemon_config=["/data/home/ryeqiu/Tutti/config/local/tutti_daemon_kv0.yaml",
                                 "/data/home/ryeqiu/Tutti/config/local/tutti_daemon_md0.yaml"],
                  devices=[dict(device_id=i, mount_path=str(mount / root.name)) for i in ids])
    pair = (rank // 2 * 2, rank // 2 * 2 + 1)
    pages = (4096, 8192, 4096)
    store = TuttiKVStore(root / f"r{rank}", 160, 16384, preset=preset,
                         io_stream="auto", segment_file_slots=160,
                         rank_id=rank, tp_size=world, layer_page_bytes=pages,
                         replica_groups=(pair, pair, tuple(range(world))))
    kernel_group = int(os.environ.get("TUTTI_TEST_REPLICA_KERNEL_GROUP", "1"))
    assert kernel_group in (1, 2, 4)
    caches = {f"layer{l}": torch.empty((384 * kernel_group, 128, p // (128 * kernel_group)),
                                      dtype=torch.uint8, device=f"cuda:{rank}")
              for l, p in enumerate(pages)}
    keys = [((i * 6364136223846793005) % (1 << 64)).to_bytes(8, "little") +
            i.to_bytes(8, "little") for i in range(140)]
    tables = tuple((8 + (279 - i * 2 + rank * 13) % 280,
                    8 + (278 - i * 2 + rank * 13) % 280) for i in range(140))
    logical_views = tuple(tensor.view(384, page_bytes)
                          for tensor, page_bytes in zip(caches.values(), pages))
    verified = 0
    try:
        store.open()
        store.set_layer_span(3)
        backend = store.create_direct_transfer(caches, num_layers=3, blocks_per_chunk=2,
                                               chunk_tokens=256, segment_bytes=16384)
        store._direct_backend = backend
        backend.register_paged_caches(caches, num_layers=3, blocks_per_chunk=2,
                                       chunk_tokens=256, segment_bytes=16384,
                                       max_chunks_per_wave=140)
        store._read_service = ReadService(store)
        for layer, tensor in enumerate(logical_views):
            head = rank // 2 if layer != 2 else 0
            for i, blocks in enumerate(tables):
                for page, block in enumerate(blocks):
                    tensor[block].fill_((layer * 53 + i * 7 + page * 11 + head) % 256)
        torch.cuda.current_stream(rank).synchronize()
        assert backend.prepare_write_targets(keys) == tuple(keys)
        for layer in range(3):
            assert backend.put_paged_batch(keys, layer, tables).wait_result(timeout=15).ok
        backend.end_target_plan("write")
        backend.warm_up_registration()
        rendezvous.set(f"written/{rank}", b"1")
        rendezvous.wait([f"written/{r}" for r in range(world)], timedelta(seconds=20))

        submit = backend._advance_submit
        reads = []
        def checked_submit(state):
            physical = state["physical"]
            if physical == len(reads):
                requests = [request for segment in reversed(state["submit"].segments)
                            for _, request in segment]
                assert physical < 3, "cache entity was split into extra submits"
                members = store._replica_groups[physical]
                expected = [block * pages[physical] for key, blocks in zip(keys, tables)
                            if members[replica.owner_index(key, len(members))] == rank
                            for block in blocks]
                assert [r[3] for r in requests] == expected
                assert all(r[2] == backend._layer_tickets[physical]
                           and r[4] == pages[physical] and r[5] == "read" for r in requests)
                reads.append(len(requests))
                if epoch == 3 and rank == world - 1 and physical == 2:
                    raise RuntimeError("injected NVMe read failure before submit")
            return submit(state)
        backend._advance_submit = checked_submit
        for epoch in range(1, 4):
            reads.clear()
            for tensor in caches.values():
                tensor.fill_(255)
            torch.cuda.current_stream(rank).synchronize()
            try:
                backend.begin_target_plan(keys, "read")
                if epoch == 2:
                    with store.stream_context("read"):
                        torch.cuda._sleep(4_000_000_000)
                backend.begin_prefetch(keys, tables, (0, 1, 2))
                for layer, tensor in enumerate(logical_views):
                    completion = backend.get_paged_batch(keys, layer, tables)
                    if epoch == 2 and layer == 0:
                        assert not completion.fence_event.query(), "recovery waited for GPU before returning"
                    assert completion.wait_result(timeout=15).ok
                    head = rank // 2 if layer != 2 else 0
                    for i, blocks in enumerate(tables):
                        for page, block in enumerate(blocks):
                            expected = (layer * 53 + i * 7 + page * 11 + head) % 256
                            assert bool((tensor[block] == expected).all()), (rank, epoch, layer, i, page)
                    assert bool((tensor[:8] == 255).all()), "unrelated KV pages overwritten"
                    verified += 1
                backend.join_prefetch()
                if epoch == 3:
                    raise AssertionError("injected failure was not propagated")
                assert len(reads) == 3 and reads[0] == reads[1] == 140
                assert not hasattr(backend, "_send_ticket")
            except RuntimeError as exc:
                if epoch != 3 or "injected NVMe read failure" not in str(exc):
                    raise
            finally:
                backend.end_target_plan("read")
        assert verified >= 6
        store._read_service.close()
        store.close()
        dist.destroy_process_group()
        output.put((rank, verified, True, None))
    except Exception as exc:
        output.put((rank, verified, False, repr(exc)))
        raise


def _run_workers(worker, world):
    import torch.distributed as dist
    import torch.multiprocessing as mp
    server = dist.TCPStore("127.0.0.1", 0, world + 1, True,
                           timeout=timedelta(seconds=20), wait_for_workers=False)
    context = mp.get_context("spawn")
    output = context.Queue()
    processes = [context.Process(target=worker, args=(r, world, server.port, output))
                 for r in range(world)]
    for process in processes:
        process.start()
    reports = []
    try:
        for _ in processes:
            reports.append(output.get(timeout=90))
    except queue.Empty:
        pytest.fail("IPC test stalled; stop hardware escalation and inspect workers")
    for process in processes:
        process.join(timeout=20)
    assert all(p.exitcode == 0 for p in processes), [p.exitcode for p in processes]
    assert len(reports) == world and all(r[2] and r[3] is None for r in reports), reports


def test_nvme_peer_recovery_repeated_epochs_and_failed_read():
    from pathlib import Path
    if not os.environ.get("TUTTI_TEST_REPLICA_NVME_ROOT"):
        pytest.skip("set TUTTI_TEST_REPLICA_NVME_ROOT to a new isolated directory")
    import torch
    if torch.cuda.device_count() < 8:
        pytest.skip("not enough GPUs")
    root = Path(os.environ["TUTTI_TEST_REPLICA_NVME_ROOT"])
    assert root.is_absolute() and root.parent == Path("/mnt/nvme4/tutti-profile/replica-pages-20261020")
    assert root.name.startswith("nvme-") and not root.exists()
    root.mkdir()
    for mount in (Path("/mnt/tutti_md0"), Path("/mnt/nvme4")):
        assert os.path.ismount(mount), f"not mounted: {mount}"
        (mount / root.name).mkdir(exist_ok=False)
    _run_workers(_nvme_worker, 8)


@pytest.mark.parametrize("world", [2, 8])
def test_ipc_repeated_epochs_delayed_peer_failure_and_safe_close(world):
    if os.environ.get("TUTTI_TEST_REPLICA_IPC") != "1":
        pytest.skip("set TUTTI_TEST_REPLICA_IPC=1 to run isolated GPU IPC tests")
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < world:
        pytest.skip("not enough GPUs")
    server = dist.TCPStore("127.0.0.1", 0, world + 1, True,
                           timeout=timedelta(seconds=15), wait_for_workers=False)
    context = mp.get_context("spawn")
    output = context.Queue()
    processes = [context.Process(target=_ipc_worker, args=(r, world, server.port, output))
                 for r in range(world)]
    for process in processes:
        process.start()
    reports = []
    try:
        for _ in processes:
            reports.append(output.get(timeout=45))
    except queue.Empty:
        pytest.fail("IPC test stalled; stop hardware escalation and inspect workers")
    for process in processes:
        process.join(timeout=15)
    assert all(p.exitcode == 0 for p in processes), [p.exitcode for p in processes]
    assert len(reports) == world and all(r[2] and r[3] is None for r in reports), reports

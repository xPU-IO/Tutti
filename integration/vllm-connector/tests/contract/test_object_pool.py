from __future__ import annotations

import os
import time
from collections import namedtuple

import pytest

from stores.tutti_nvme.layout import Layout
from stores.tutti_nvme.object_pool import ObjectPool, PoolConfig, PoolResourceExhausted
from stores.tutti_nvme.store import TuttiKVStore


SEG = 4096


class _Runtime:
    def __init__(self):
        self.open_batches = []
        self.close_batches = []
        self.submit_calls = []
        self.released = []
        self.next_ticket = 1

    def caps(self):
        return {
            "memory": ["host"],
            "max_in_flight_operations": 64,
            "supports_multi_stream": False,
            "max_concurrent_streams": 1,
        }

    def open_batch(self, uris):
        self.open_batches.append(list(uris))
        result = list(range(self.next_ticket, self.next_ticket + len(uris)))
        self.next_ticket += len(uris)
        return result

    def close_batch(self, tickets):
        self.close_batches.append(list(tickets))

    def register_memory(self, addr, size, kind, accel_id=-1, io_granularity=0):
        return 9000

    def submit(self, requests, **_kwargs):
        self.submit_calls.append(list(requests))
        return namedtuple("Submit", "status_ok status_msg io_handle initial_states rejected")(
            True, "", self.next_ticket,
            [True] * len(requests), [],
        )

    def wait(self, handle, timeout_ms):
        return "OK", "COMPLETED"

    def release_io(self, handle):
        self.released.append(handle)



def _key(value: int, layer: int = 0) -> bytes:
    return bytes([value]) * 16 + layer.to_bytes(2, "little")


def _configured_pool(tmp_path, *, initial=2, low=1, high=2, maximum=4,
                     allocator_enabled=False):
    layout = Layout(tmp_path, SEG)
    pool = ObjectPool(
        layout,
        PoolConfig(initial, low, high, maximum, wait_timeout_s=0.05),
        allocator_enabled=allocator_enabled,
    )
    layout.attach_object_pool(pool)
    layout.ensure_dirs()
    layout.set_layer_span(2)
    return layout, pool


def test_bind_creates_real_fiemap_slots_before_request(tmp_path):
    layout, pool = _configured_pool(tmp_path)
    assert pool.snapshot() == {
        "configured": True,
        "free": 2,
        "allocated": 0,
        "creating": 0,
        "scrubbing": 0,
        "thread_alive": False,
        "slot_bytes": 2 * SEG,
    }
    paths = layout.pool_slot_paths(0)
    assert paths[0].stat().st_size == 2 * SEG
    assert paths[0].stat().st_blocks > 0
    assert pool._validate_group(paths)
    pool.close()


def test_put_does_not_write_zero_or_fsync_after_bind(tmp_path, monkeypatch):
    layout, pool = _configured_pool(tmp_path)
    calls = []
    monkeypatch.setattr(
        "stores.tutti_nvme.layout._append_real_zeros",
        lambda *args, **kwargs: calls.append(args),
    )
    runtime = _Runtime()
    store = TuttiKVStore(tmp_path, 2, SEG, runtime=runtime)
    store._layout = layout
    store.open()
    store.set_layer_span(2)
    buffer = bytearray(SEG)
    buffer_id = store.register_buffer(buffer, SEG)
    completion = store.put_batch([(_key(1), buffer_id, 0)])
    completion.wait()
    assert calls == []
    assert runtime.open_batches == [[layout.target_uri(bytes([1]) * 16)]]
    store.close()
    pool.close()


def test_32_missing_targets_use_one_open_batch(tmp_path):
    runtime = _Runtime()
    store = TuttiKVStore(
        tmp_path, 32, SEG, runtime=runtime,
        initial_slots=32, low_watermark=0, high_watermark=32, max_slots=32,
    )
    store.open()
    store.set_layer_span(1)
    buffer_id = store.register_buffer(bytearray(32 * SEG), SEG)
    keys = [_key(i) for i in range(32)]
    store.put_batch([(key, buffer_id, i * SEG) for i, key in enumerate(keys)]).wait()
    assert len(runtime.open_batches) == 1
    assert len(runtime.open_batches[0]) == 32
    store.close()


def test_eviction_closes_targets_before_rename(tmp_path):
    runtime = _Runtime()
    store = TuttiKVStore(
        tmp_path, 1, SEG, runtime=runtime,
        initial_slots=1, low_watermark=0, high_watermark=1, max_slots=1,
    )
    store.open()
    store.set_layer_span(1)
    buffer_id = store.register_buffer(bytearray(SEG), SEG)
    key = _key(7)
    store.put_batch([(key, buffer_id, 0)]).wait()
    events = []
    original = store._layout.pool_rename_group
    store._layout.pool_rename_group = lambda src, dst: (
        events.append("rename"), original(src, dst)
    )[1]
    original_close = runtime.close_batch
    runtime.close_batch = lambda tickets: (events.append("close"), original_close(tickets))[1]
    store.drop([key])
    assert events.index("close") < events.index("rename")
    store.close()


def test_inode_stable_across_free_alloc_free(tmp_path):
    layout, pool = _configured_pool(
        tmp_path, initial=1, low=0, high=1, maximum=1,
        allocator_enabled=True,
    )
    free_path = layout.pool_slot_paths(0)[0]
    inode = free_path.stat().st_ino
    key = bytes([8]) * 16
    layout.prepare_put([key + b"\x00\x00"], 1)
    chunk_path = layout.chunk_file(key)
    assert chunk_path.stat().st_ino == inode
    layout.commit_layers([key + b"\x00\x00", key + b"\x01\x00"])
    layout.drop([key + b"\x00\x00", key + b"\x01\x00"])
    deadline = time.monotonic() + 2.0
    while pool.free_count < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert free_path.stat().st_ino == inode
    assert pool._validate_group((free_path,))
    assert free_path.read_bytes() == b"\x00" * (2 * SEG)
    pool.close()


def test_multiple_chunks_recycle_every_slot_and_manifest_entry(tmp_path):
    layout, pool = _configured_pool(
        tmp_path, initial=2, low=0, high=2, maximum=2,
        allocator_enabled=True,
    )
    chunks = [bytes([20]) * 16, bytes([21]) * 16]
    io_keys = [
        chunk + layer.to_bytes(2, "little")
        for chunk in chunks
        for layer in range(2)
    ]
    layout.prepare_put(io_keys, 2)
    layout.commit_layers(io_keys)
    layout.drop(io_keys)

    deadline = time.monotonic() + 2.0
    while pool.free_count < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    snapshot = pool.snapshot()
    assert snapshot["allocated"] == 0
    assert snapshot["scrubbing"] == 0
    assert snapshot["free"] == 2
    assert all(not layout.chunk_file(chunk).exists() for chunk in chunks)
    assert all(layout.pool_slot_paths(slot)[0].exists() for slot in (0, 1))
    pool.close()


def test_low_watermark_allocator_refills_to_high(tmp_path):
    layout, pool = _configured_pool(
        tmp_path, initial=1, low=1, high=2, maximum=3,
        allocator_enabled=True,
    )
    pool.allocate([bytes([9]) * 16])
    deadline = time.monotonic() + 2.0
    while pool.free_count < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pool.free_count == 2
    assert pool.total_count <= 3
    pool.close()
    assert not pool.snapshot()["thread_alive"]


def test_pool_exhaustion_is_structured_and_bounded(tmp_path):
    layout, pool = _configured_pool(
        tmp_path, initial=1, low=0, high=1, maximum=1,
        allocator_enabled=False,
    )
    pool.allocate([bytes([10]) * 16])
    with pytest.raises(PoolResourceExhausted) as exc:
        pool.allocate([bytes([11]) * 16])
    assert exc.value.code == "RESOURCE_EXHAUSTED"
    pool.close()


def test_restart_reclaims_orphan_without_markers(tmp_path):
    layout, pool = _configured_pool(tmp_path, initial=1, low=0, high=1, maximum=1)
    key = bytes([12]) * 16
    layout.prepare_put([key + b"\x00\x00"], 1)
    pool.close()

    layout2 = Layout(tmp_path, SEG)
    pool2 = ObjectPool(
        layout2, PoolConfig(1, 0, 1, 1), allocator_enabled=False
    )
    layout2.attach_object_pool(pool2)
    layout2.ensure_dirs()
    layout2.set_layer_span(2)
    assert pool2.free_count == 1
    assert not layout2.chunk_file(key).exists()
    pool2.close()

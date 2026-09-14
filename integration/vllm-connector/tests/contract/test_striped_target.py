"""Striped target layout and store contract tests.

The fake runtime performs the same logical-to-shard mapping as the existing
runtime binding, so these tests exercise actual files and marker ordering
without requiring a daemon or NVMe hardware.
"""

from __future__ import annotations

import ctypes
import os
import time
from collections import namedtuple
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from stores.tutti_nvme.store import TuttiKVStore
from stores.tutti_nvme.striped_layout import StripedLayout


SEGMENT = 16 * 1024
UNIT = 8 * 1024
MOUNTS = 3


def _key(value, layer=0):
    return bytes([value]) * 16 + layer.to_bytes(2, "little")


SubmitResult = namedtuple(
    "SubmitResult", "status_ok status_msg io_handle initial_states rejected"
)


class StripedFakeRuntime:
    """Host fake that maps every request byte into real shard files."""

    def __init__(self):
        self._next = 0
        self._targets = {}
        self._memories = {}

    def caps(self):
        return {"target": ["stub"], "memory": ["host"]}

    def open_batch(self, uris):
        tickets = []
        for uri in uris:
            assert uri.startswith("striped://")
            parsed = urlsplit(uri)
            query = parse_qs(parsed.query)
            mounts = query["devs"][0].split(",")
            unit = int(query["unit"][0])
            rotation = int(query["rot"][0])
            self._next += 1
            self._targets[self._next] = (
                parsed.netloc + parsed.path, mounts, unit, rotation
            )
            tickets.append(self._next)
        return tickets

    def close_target(self, ticket):
        self._targets.pop(ticket, None)

    def close_batch(self, tickets):
        for ticket in tickets:
            self.close_target(ticket)

    def register_memory(self, addr, size, kind, accel_id=-1, io_granularity=0):
        self._next += 1
        self._memories[self._next] = (addr, size)
        return self._next

    def submit(self, requests, **_kwargs):
        for request in requests:
            self._execute(request)
        self._next += 1
        return SubmitResult(True, "", self._next, [], [])

    def wait(self, _handle, _timeout_ms=0):
        return "OK", "COMPLETED"

    def release_io(self, _handle):
        return None

    def shutdown(self, _timeout_ms):
        return None

    def _execute(self, request):
        target, target_offset, memory, memory_offset, length, direction = request
        name, mounts, unit, rotation = self._targets[target]
        addr, _ = self._memories[memory]
        remaining = length
        logical = target_offset
        copied = memory_offset
        while remaining:
            shard = ((logical // unit) + rotation) % len(mounts)
            shard_offset = (logical // (unit * len(mounts))) * unit + logical % unit
            count = min(remaining, unit - logical % unit)
            path = Path(mounts[shard]) / "striped" / (
                name + ".shard" + str(shard)
            )
            fd = os.open(str(path), os.O_RDWR)
            try:
                if direction == "write":
                    data = ctypes.string_at(addr + copied, count)
                    os.pwrite(fd, data, shard_offset)
                else:
                    data = os.pread(fd, count, shard_offset)
                    ctypes.memmove(addr + copied, data, len(data))
            finally:
                os.close(fd)
            logical += count
            copied += count
            remaining -= count


def test_striped_layout_uri_rotation_and_capacity(tmp_path):
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(MOUNTS)]
    layout = StripedLayout(tmp_path / "meta-root", SEGMENT, mounts, UNIT)
    layout.ensure_dirs()
    key = _key(5, 2)
    decoded = layout.prepare_put([key], capacity_chunks=1)
    chunk, layer = decoded[key]

    assert layer == 2
    assert layout.shard_rotation(chunk) == int.from_bytes(b"\x05" * 4, "little") % MOUNTS
    # URI 的 name 带 rank 隔离段：<rank_segment>/<chunk hex>
    assert (f"striped://{layout.rank_segment}/" + chunk.hex()
            in layout.target_uri(chunk))
    assert "unit=8192" in layout.target_uri(chunk)
    assert "rot=2" in layout.target_uri(chunk)
    assert layout.target_size(chunk) == 6 * UNIT
    assert layout.chunk_file_count() == 1
    assert all(layout.shard_file(chunk, i).stat().st_size == 2 * UNIT
               for i in range(MOUNTS))


def test_striped_store_roundtrip_marker_and_drop(tmp_path):
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(MOUNTS)]
    store = TuttiKVStore(
        tmp_path / "meta-root", 2, SEGMENT, runtime=StripedFakeRuntime(),
        layout="striped", mounts=mounts, stripe_unit=UNIT,
    )
    store.open()
    source = bytearray(SEGMENT * 2)
    source[:SEGMENT] = bytes(range(256)) * (SEGMENT // 256)
    source[SEGMENT:] = b"z" * SEGMENT
    source_id = store.register_buffer(source, UNIT)
    key0, key1 = _key(7, 0), _key(7, 1)
    store.put_batch([(key0, source_id, 0), (key1, source_id, SEGMENT)]).wait()

    assert store.scan() == sorted([key0, key1])
    destination = bytearray(SEGMENT * 2)
    destination_id = store.register_buffer(destination, UNIT)
    store.get_batch([(key0, destination_id, 0), (key1, destination_id, SEGMENT)]).wait()
    assert destination == source

    store.drop([key0])
    assert store.has(key1)
    assert all(store._layout.shard_file(key1[:16], i).exists()
               for i in range(MOUNTS))
    store.drop([key1])
    assert store.scan() == []
    assert store._targets == {}
    assert all(not store._layout.shard_file(key1[:16], i).exists()
               for i in range(MOUNTS))
    store.close()


def test_striped_rank_isolation(tmp_path):
    """多 rank 共用同一组盘：分片/槽位路径与 URI 必须按 rank 隔离。

    否则不同 rank 会写同名文件（互相追加/覆盖），槽位校验失败且数据
    互相污染——8 卡每 4 rank 共用一组盘时的实测故障。
    """
    mounts = [tmp_path / "nvme0", tmp_path / "nvme1"]
    rank0 = StripedLayout(tmp_path / "root-a", SEGMENT, mounts, UNIT,
                          rank_id=0)
    rank4 = StripedLayout(tmp_path / "root-b", SEGMENT, mounts, UNIT,
                          rank_id=4)
    chunk = _key(7)[:16]
    assert rank0.shard_file(chunk, 0) != rank4.shard_file(chunk, 0)
    assert rank0.pool_slot_paths(3) != rank4.pool_slot_paths(3)
    assert rank0.pool_slot_uri(3) != rank4.pool_slot_uri(3)
    # URI 的 name 与物理路径严格对应（C++ 按 <mount>/striped/<name>.shard<i> 拼）
    for layout in (rank0, rank4):
        uri = layout.pool_slot_uri(3)
        name = uri[len("striped://"):uri.index("?")]
        assert name.startswith(layout.rank_segment + "/")
        for shard, path in enumerate(layout.pool_slot_paths(3)):
            expected = (Path(mounts[shard]).resolve() / "striped"
                        / (name + ".shard" + str(shard)))
            assert path == expected


def test_pool_create_slot_is_idempotent(tmp_path):
    """同名槽位文件重复创建必须覆盖而非追加（崩溃残留场景）。"""
    mounts = [tmp_path / "nvme0", tmp_path / "nvme1"]
    layout = StripedLayout(tmp_path / "root", SEGMENT, mounts, UNIT)
    slot_bytes = SEGMENT * 2
    expected = layout.pool_physical_sizes(slot_bytes)
    layout.pool_create_slot(2, slot_bytes)
    layout.pool_create_slot(2, slot_bytes)   # 第二次必须仍是同一大小
    for path, size in zip(layout.pool_slot_paths(2), expected):
        assert path.stat().st_size == size


def test_striped_pool_slots_are_stable_identities(tmp_path):
    """池模式下槽位是稳定身份：绑定/回收不改名、URI 与预开票据恒定。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(2)]
    store = TuttiKVStore(
        tmp_path / "meta-root", 8, SEGMENT, runtime=StripedFakeRuntime(),
        layout="striped", mounts=mounts, stripe_unit=UNIT,
        initial_slots=4,
    )
    store.open()
    store.set_layer_span(2)
    source = bytearray(SEGMENT * 2)
    source[:SEGMENT] = b"q" * SEGMENT
    source[SEGMENT:] = b"w" * SEGMENT
    source_id = store.register_buffer(source, UNIT)
    key = _key(9, 0)
    store.put_batch([(key, source_id, 0)]).wait()

    layout = store._layout
    pool = store._object_pool
    chunk = key[:16]
    slot = pool.slot_of(chunk)
    assert slot is not None
    # 绑定不改名：数据分片仍在 free/ 槽位路径下，chunk 命名文件不存在
    assert all(path.exists() for path in layout.pool_slot_paths(slot))
    assert not any(layout.shard_file(chunk, i).exists()
                   for i in range(layout.num_shards))
    # 请求路径 URI == 槽位 URI（预开票据与提交共用同一目标）
    assert layout.target_uri(chunk) == layout.pool_slot_uri(slot)
    assert pool.ticket_of_slot(slot) > 0
    # rotation 由槽位号派生（跨"回收→再分配"恒定）
    assert layout.pool_slot_rotation(slot) == slot % layout.num_shards
    # 槽位逻辑大小 = 各分片 min-shard 完整 stripe round
    assert layout.target_size(chunk) == 2 * SEGMENT
    store.close()


def test_striped_pool_recycled_slot_keeps_uri(tmp_path):
    """回收再分配后槽位 URI 不变：预开票据跨复用继续命中。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(2)]
    store = TuttiKVStore(
        tmp_path / "meta-root", 8, SEGMENT, runtime=StripedFakeRuntime(),
        layout="striped", mounts=mounts, stripe_unit=UNIT,
        initial_slots=1, max_slots=1,   # 单槽位：禁止扩容，回收必复用
    )
    store.open()
    store.set_layer_span(2)
    source = bytearray(SEGMENT * 2)
    source[:SEGMENT] = b"a" * SEGMENT
    source[SEGMENT:] = b"b" * SEGMENT
    source_id = store.register_buffer(source, UNIT)
    layout = store._layout
    pool = store._object_pool

    first = _key(3, 0)
    store.put_batch([(first, source_id, 0)]).wait()
    slot = pool.slot_of(first[:16])
    assert slot is not None
    first_uri = layout.target_uri(first[:16])
    first_ticket = pool.ticket_of_slot(slot)

    store.drop([first])
    deadline = time.monotonic() + 5.0
    while pool.free_count == 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    second = _key(4, 0)
    store.put_batch([(second, source_id, 0)]).wait()
    assert pool.slot_of(second[:16]) == slot
    assert layout.target_uri(second[:16]) == first_uri
    assert pool.ticket_of_slot(slot) == first_ticket
    store.close()


def test_striped_unit_larger_than_segment_maps_one_shard(tmp_path):
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(2)]
    layout = StripedLayout(tmp_path / "meta-root", 4096, mounts, 8192)
    layout.ensure_dirs()
    key = _key(1)
    layout.prepare_put([key], 1)
    assert layout.target_size(key[:16]) == 2 * 8192
    assert "unit=8192" in layout.target_uri(key[:16])


@pytest.mark.parametrize("kwargs", [
    {"mounts": ["/one"], "stripe_unit": UNIT},
    {"mounts": ["/one", "/two"], "stripe_unit": 2048},
    {"mounts": ["/one", "/two"], "stripe_unit": UNIT, "layout": "other"},
])
def test_invalid_striped_options(tmp_path, kwargs):
    options = dict(
        root=tmp_path, num_chunks=1, segment_bytes=SEGMENT,
        runtime=StripedFakeRuntime(), layout="striped",
        mounts=["/one", "/two"], stripe_unit=UNIT,
    )
    options.update(kwargs)
    with pytest.raises(ValueError):
        TuttiKVStore(**options)

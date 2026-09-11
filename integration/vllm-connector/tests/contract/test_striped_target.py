"""Striped target layout and store contract tests.

The fake runtime performs the same logical-to-shard mapping as the existing
runtime binding, so these tests exercise actual files and marker ordering
without requiring a daemon or NVMe hardware.
"""

from __future__ import annotations

import ctypes
import os
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
    assert "striped://" + chunk.hex() in layout.target_uri(chunk)
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

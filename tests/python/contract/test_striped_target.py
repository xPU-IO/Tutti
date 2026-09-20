"""条带布局契约：对象层 + 条带 resolver 替身。

替身按 C++ 条带 resolver 的同一映射（``shard = (offset/unit + rot) % N``、
``shard_offset = (offset/(unit*N))*unit + offset%unit``）把每个请求落到真实
分片文件上，因此这些用例覆盖真实文件读写，却不需要 daemon 或 NVMe 硬件。

条带的对象几何、槽位路径与 rotation 都在对象层（C++）：
槽位以 ``<mount_i>/striped/<slot>.shard<i>`` 命名，段 0 从对象头之后开始。
"""

from __future__ import annotations

import ctypes
import os
from collections import namedtuple
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from tutti.storage.tutti_nvme.object_layout import ObjectLayout
from tutti.storage.tutti_nvme.store import TuttiKVStore


SEGMENT = 16 * 1024
UNIT = 16 * 1024
MOUNTS = 3
HEADER = 4096
# 对象层的条带几何要求 payload 能被 unit × 分片数整除（否则末轮条带不满，
# 段尾会越过分片边界）。因此各用例按分片数取层宽：
#   3 分片 → span 3（payload 48KiB = unit × 3）
#   2 分片 → span 2（payload 32KiB = unit × 2）


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
            # 对象层不写 rot：缺省 0（rotation 由槽位号在 C++ 侧派生）
            rotation = int(query["rot"][0]) if "rot" in query else 0
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


def _object_layout(tmp_path, *, mounts, layers=2, segment=SEGMENT, unit=UNIT,
                   capacity=8, rank_id=0, rank_count=1):
    root = tmp_path / "meta-root"
    root.mkdir(parents=True, exist_ok=True)
    for mount in mounts:
        Path(mount).mkdir(parents=True, exist_ok=True)
    layout = ObjectLayout(
        root, segment, mounts=[str(mount) for mount in mounts],
        stripe_unit=unit, capacity_chunks=capacity,
        prewarm_chunks=min(capacity, 4), rank_id=rank_id, rank_count=rank_count,
        namespace=b"striped-contract",
    )
    layout.set_layer_span(layers)
    return layout


def test_striped_store_roundtrip_and_drop(tmp_path):
    """条带 store 端到端：写读回、按对象提交、drop 使整个对象失效。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(2)]
    store = TuttiKVStore(
        tmp_path / "meta-root", 2, SEGMENT, runtime=StripedFakeRuntime(),
        layout="striped", mounts=mounts, stripe_unit=UNIT,
    )
    store.open()
    store.set_layer_span(2)
    source = bytearray(SEGMENT * 2)
    source[:SEGMENT] = bytes(range(256)) * (SEGMENT // 256)
    source[SEGMENT:] = b"z" * SEGMENT
    source_id = store.register_buffer(source, UNIT)
    key0, key1 = _key(7, 0), _key(7, 1)
    store.put_batch([(key0, source_id, 0), (key1, source_id, SEGMENT)]).wait()

    assert store.scan() == sorted([key0, key1])
    destination = bytearray(SEGMENT * 2)
    destination_id = store.register_buffer(destination, UNIT)
    store.get_batch(
        [(key0, destination_id, 0), (key1, destination_id, SEGMENT)]
    ).wait()
    assert destination == source

    # drop 命中对象任一 key ⇒ 整个对象（各层）一起失效
    store.drop([key0])
    assert not store.has(key0) and not store.has(key1)
    assert store.scan() == []
    store.close()


def test_striped_uri_and_shard_paths(tmp_path):
    """URI 携带 devs/unit；分片文件落在 <mount>/striped/<slot>.shard<i>。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(MOUNTS)]
    layout = _object_layout(tmp_path, mounts=mounts, layers=MOUNTS)
    chunk = _key(5)[:16]
    admitted, rejected = layout.prepare_put([_key(5, 1)], capacity_chunks=8)
    assert admitted and rejected == 0

    uri = layout.target_uri(chunk)
    parsed = urlsplit(uri)
    query = parse_qs(parsed.query)
    assert query["devs"][0].split(",") == [str(m) for m in mounts]
    assert query["unit"] == [str(UNIT)]
    # 对象头前缀向上取整到整个条带轮：段（因而每个请求）都与条带单元对齐，
    # 不会因 4096 的偏移让 128KiB 请求多跨一个分片（实测多出的 NVMe 命令把
    # 写 IO 设备时间抬高 64%）。
    wheel = UNIT * MOUNTS
    offset = layout.target_offset(chunk)
    assert offset % wheel == 0
    assert offset == -(-HEADER // wheel) * wheel
    assert layout.target_size(chunk) == MOUNTS * SEGMENT

    # 分片文件按 URI 的 name 命名；每分片是整数个条带轮，且总逻辑空间
    # （N × floor(分片尺寸/unit) × unit）覆盖前缀 + payload。
    slot = int(parsed.netloc + parsed.path)
    for index, mount in enumerate(mounts):
        path = Path(mount) / "striped" / f"{slot}.shard{index}"
        assert path.exists()
        size = path.stat().st_size
        assert size % UNIT == 0
        assert MOUNTS * (size // UNIT) * UNIT >= offset + MOUNTS * SEGMENT


def test_striped_slot_identity_is_stable(tmp_path):
    """同一 chunk 的槽位 URI 恒定；回收后槽位可被新 chunk 复用。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(2)]
    layout = _object_layout(tmp_path, mounts=mounts, capacity=2)
    first = _key(3)[:16]
    layout.prepare_put([_key(3, 0), _key(3, 1)], capacity_chunks=2)
    uri = layout.target_uri(first)
    # 重复预留（同一步重放）不改 URI
    layout.prepare_put([_key(3, 0)], capacity_chunks=2)
    assert layout.target_uri(first) == uri

    layout.commit_layers([_key(3, 0), _key(3, 1)])
    assert layout.is_committed(first)
    layout.release_chunks([first])
    assert not layout.is_committed(first)
    assert layout.committed_chunks() == set()

    # 回收后的槽位可再次分配并完成提交
    second = _key(4)[:16]
    admitted, rejected = layout.prepare_put(
        [_key(4, 0), _key(4, 1)], capacity_chunks=2
    )
    assert len(admitted) == 2 and rejected == 0
    layout.commit_layers([_key(4, 0), _key(4, 1)])
    assert layout.is_committed(second)
    layout.close_object_pool()


def test_striped_capacity_trims_unadmitted_chunks(tmp_path):
    """容量耗尽按对象层契约裁剪：被拒 chunk 没有槽位、也不会被提交。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(2)]
    layout = _object_layout(tmp_path, mounts=mounts, capacity=1)
    admitted, rejected = layout.prepare_put(
        [_key(9, 0), _key(10, 0)], capacity_chunks=1
    )
    assert len(admitted) == 1 and rejected == 1
    kept = next(iter(admitted.values()))[0]
    dropped = _key(10)[:16] if kept == _key(9)[:16] else _key(9)[:16]
    with pytest.raises(KeyError):
        layout.target_uri(dropped)
    assert not layout.is_committed(dropped)
    layout.close_object_pool()


def test_striped_rank_roots_are_isolated(tmp_path):
    """多 rank 共用同一组盘：槽位路径与 URI 必须按 rank 的 root 隔离。

    否则不同 rank 会写同名分片（互相追加/覆盖），槽位校验失败且数据互相
    污染——8 卡每 4 rank 共用一组盘时的实测故障。分片名由对象的 uri 派生，
    故隔离由「每个 rank 独立 root」保证，这里把它钉住。
    """
    mounts = [tmp_path / "nvme0", tmp_path / "nvme1"]
    rank0 = _object_layout(tmp_path / "rank0", mounts=mounts, rank_id=0)
    rank4 = _object_layout(tmp_path / "rank4", mounts=mounts, rank_id=4)
    rank0.prepare_put([_key(7, 0)], capacity_chunks=8)
    rank4.prepare_put([_key(7, 0)], capacity_chunks=8)
    uri0 = urlsplit(rank0.target_uri(_key(7)[:16]))
    uri4 = urlsplit(rank4.target_uri(_key(7)[:16]))
    # 命名空间不同 ⇒ 分片文件名不同（同一组盘上互不覆盖）
    assert rank0.root != rank4.root
    assert (uri0.netloc, uri0.path) == (uri4.netloc, uri4.path)  # 槽位号可相同
    assert uri0.query == uri4.query
    rank0.close_object_pool()
    rank4.close_object_pool()


def test_striped_unit_larger_than_segment(tmp_path):
    """条带粒度大于段大小仍是合法几何（一个分片承载多段）。"""
    mounts = [tmp_path / "nvme0", tmp_path / "nvme1"]
    layout = _object_layout(
        tmp_path, mounts=mounts, layers=4, segment=8 * 1024,
        unit=16 * 1024, capacity=4,
    )
    admitted, _ = layout.prepare_put([_key(1, 0)], capacity_chunks=4)
    assert admitted
    assert layout.target_size(_key(1)[:16]) == 4 * 8 * 1024
    assert "unit=16384" in layout.target_uri(_key(1)[:16])
    layout.close_object_pool()


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

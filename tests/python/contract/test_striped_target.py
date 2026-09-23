"""多盘轮转布局契约：对象层 + 文件 resolver 替身。

替身按 C++ local-file resolver 的同一语义（``file://<绝对路径>``，路径即
文件、文件即对象）把每个请求落到真实文件上，因此这些用例覆盖真实文件
读写，却不需要 daemon 或 NVMe 硬件。

多盘放置的槽位路径在对象层（C++）：一个槽位是一个文件
``<mount>/<r<rank_id>>/<slot>.obj``，槽位号在 mounts 间轮转，段 0 从对象
头之后开始。
"""

from __future__ import annotations

import ctypes
import os
from collections import namedtuple
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from tutti.storage.tutti_nvme.object_layout import ObjectLayout
from tutti.storage.tutti_nvme.store import TuttiKVStore


SEGMENT = 16 * 1024
MOUNTS = 3
HEADER = 4096
# 多盘几何不再有 unit 整除约束（无条带）：payload 只需 4096 对齐。
SPAN = 2


def _key(value, layer=0):
    return bytes([value]) * 16 + layer.to_bytes(2, "little")


SubmitResult = namedtuple(
    "SubmitResult", "status_ok status_msg io_handle initial_states rejected"
)


class RotatingFakeRuntime:
    """Host fake that maps every request onto the real slot file."""

    def __init__(self):
        self._next = 0
        self._targets = {}
        self._memories = {}

    def caps(self):
        return {"target": ["stub"], "memory": ["host"]}

    def open_batch(self, uris):
        tickets = []
        for uri in uris:
            assert uri.startswith("file://")
            path = unquote(urlsplit(uri).netloc + urlsplit(uri).path)
            self._next += 1
            self._targets[self._next] = path
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
        path = self._targets[target]
        addr, _ = self._memories[memory]
        fd = os.open(path, os.O_RDWR)
        try:
            if direction == "write":
                data = ctypes.string_at(addr + memory_offset, length)
                os.pwrite(fd, data, target_offset)
            else:
                data = os.pread(fd, length, target_offset)
                ctypes.memmove(addr + memory_offset, data, len(data))
        finally:
            os.close(fd)


def _object_layout(tmp_path, *, mounts, layers=SPAN, segment=SEGMENT,
                   capacity=8, rank_id=0, rank_count=1):
    root = tmp_path / "meta-root"
    root.mkdir(parents=True, exist_ok=True)
    for mount in mounts:
        Path(mount).mkdir(parents=True, exist_ok=True)
    layout = ObjectLayout(
        root, segment, mounts=[str(mount) for mount in mounts],
        capacity_chunks=capacity,
        prewarm_chunks=min(capacity, 4), rank_id=rank_id, rank_count=rank_count,
        namespace=b"rotating-contract",
    )
    layout.set_layer_span(layers)
    return layout


def _slot_path(mount, rank_id, slot):
    return Path(mount) / f"r{rank_id}" / f"{slot}.obj"


def test_rotating_store_roundtrip_and_drop(tmp_path):
    """多盘 store 端到端：写读回、按对象提交、drop 使整个对象失效。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(2)]
    store = TuttiKVStore(
        tmp_path / "meta-root", 2, SEGMENT, runtime=RotatingFakeRuntime(),
        layout="striped", mounts=mounts,
    )
    store.open()
    store.set_layer_span(SPAN)
    source = bytearray(SEGMENT * SPAN)
    source[:SEGMENT] = bytes(range(256)) * (SEGMENT // 256)
    source[SEGMENT:] = b"z" * SEGMENT
    source_id = store.register_buffer(source, SEGMENT)
    key0, key1 = _key(7, 0), _key(7, 1)
    store.put_batch([(key0, source_id, 0), (key1, source_id, SEGMENT)]).wait()

    assert store.scan() == sorted([key0, key1])
    destination = bytearray(SEGMENT * SPAN)
    destination_id = store.register_buffer(destination, SEGMENT)
    store.get_batch(
        [(key0, destination_id, 0), (key1, destination_id, SEGMENT)]
    ).wait()
    assert destination == source

    # drop 命中对象任一 key ⇒ 整个对象（各层）一起失效
    store.drop([key0])
    assert not store.has(key0) and not store.has(key1)
    assert store.scan() == []
    store.close()


def test_rotating_uri_and_slot_files(tmp_path):
    """URI 是 file://<mount>/r<rank>/<slot>.obj；槽位文件是整个对象。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(MOUNTS)]
    layout = _object_layout(tmp_path, mounts=mounts, layers=MOUNTS)
    chunk = _key(5)[:16]
    admitted, rejected = layout.prepare_put([_key(5, 1)], capacity_chunks=8)
    assert admitted and rejected == 0

    uri = layout.target_uri(chunk)
    parsed = urlsplit(uri)
    assert uri.startswith("file://")
    # 对象头紧贴 payload：无条带轮对齐预留。
    offset = layout.target_offset(chunk)
    assert offset == HEADER
    assert layout.target_size(chunk) == MOUNTS * SEGMENT

    # 槽位文件按 URI 的路径命名，恰好一个文件承载整个对象。
    slot = int(Path(unquote(parsed.netloc + parsed.path)).stem)
    mount = mounts[slot % MOUNTS]
    path = _slot_path(mount, 0, slot)
    assert path.exists()
    size = path.stat().st_size
    assert size >= HEADER + MOUNTS * SEGMENT


def test_rotating_slots_spread_over_mounts(tmp_path):
    """槽位号在 mounts 间轮转：连续槽位把请求摊到每块盘。"""
    mounts = [tmp_path / ("nvme" + str(i)) for i in range(2)]
    layout = _object_layout(tmp_path, mounts=mounts, capacity=8)
    per_mount = {0: 0, 1: 0}
    for value in range(8):
        layout.prepare_put([_key(value, 0)], capacity_chunks=8)
        uri = layout.target_uri(_key(value)[:16])
        path = Path(unquote(urlsplit(uri).netloc + urlsplit(uri).path))
        mount_index = next(
            i for i, mount in enumerate(mounts)
            if str(path).startswith(str(mount))
        )
        per_mount[mount_index] += 1
    # 8 槽 × 2 盘 = 每盘 4：轮转必须均匀，不能退化成常量映射。
    assert per_mount == {0: 4, 1: 4}
    layout.close_object_pool()


def test_rotating_slot_identity_is_stable(tmp_path):
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


def test_rotating_capacity_trims_unadmitted_chunks(tmp_path):
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


def test_rotating_rank_slots_are_isolated(tmp_path):
    """多 rank 共用同一组盘：槽位文件按 rank 子目录隔离。

    否则不同 rank 会写同名文件（互相覆盖），槽位校验失败且数据互相污染
    ——8 卡共享一组盘时的实测故障。分片文件在 ``<mount>/r<rank_id>/`` 下，
    这里把隔离钉住。
    """
    mounts = [tmp_path / "nvme0", tmp_path / "nvme1"]
    rank0 = _object_layout(tmp_path / "rank0", mounts=mounts, rank_id=0)
    rank4 = _object_layout(tmp_path / "rank4", mounts=mounts, rank_id=4)
    rank0.prepare_put([_key(7, 0)], capacity_chunks=8)
    rank4.prepare_put([_key(7, 0)], capacity_chunks=8)
    path0 = Path(unquote(urlsplit(
        rank0.target_uri(_key(7)[:16])).netloc +
        urlsplit(rank0.target_uri(_key(7)[:16])).path))
    path4 = Path(unquote(urlsplit(
        rank4.target_uri(_key(7)[:16])).netloc +
        urlsplit(rank4.target_uri(_key(7)[:16])).path))
    # 同一槽位号、同一组盘，但 rank 子目录不同 ⇒ 互不覆盖
    assert path0.parent != path4.parent
    assert path0.parent.name == "r0"
    assert path4.parent.name == "r4"
    assert path0.exists() and path4.exists()
    assert path0 != path4
    rank0.close_object_pool()
    rank4.close_object_pool()


def test_invalid_rotating_options(tmp_path):
    """未知 layout 拒绝；多盘无其它几何约束（unit 已不存在）。"""
    with pytest.raises(ValueError):
        TuttiKVStore(
            tmp_path, 1, SEGMENT, runtime=RotatingFakeRuntime(),
            layout="other",
        )

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
from tutti.storage.tutti_nvme.store import (
    TuttiKVStore,
    _TuttiCompletion,
    _derive_device_fields,
)
import json

SEG = 4096


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
            root=root, num_chunks=8, segment_bytes=4096, runtime=FakeRuntime()
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
        runtime=runtime, io_stream="auto", preset={"gpu_id": 0},
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
        runtime=runtime, io_stream="auto", preset={"gpu_id": 0},
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
        runtime=runtime, io_stream="auto", preset={"gpu_id": 0},
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
        runtime=runtime, io_stream="auto", preset={"gpu_id": 0},
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
        runtime=None, io_stream="auto", preset={"gpu_id": 0},
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
    """同 chunk 各层共用一个对象文件；段 offset = 对象头 + layer × segment。"""
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
    assert payload == 4096                      # 自描述对象头
    assert len(data) == payload + 6 * SEG       # 几何按层宽定型，不是"用多少长多少"
    for layer in layers:
        segment = data[payload + layer * SEG:payload + (layer + 1) * SEG]
        assert segment == bytes([0x40 + layer]) * SEG


def test_slot_materialisation_is_physical(tmp_path):
    """预留即物化：整槽位实写零（物理块就位），非稀疏跳写。"""
    store = make_store(tmp_path, layers=4)
    store.open()
    src_id = store.register_buffer(bytearray(SEG), SEG)
    chunk = b"\x22" * 16
    store.put_batch([(io_key(chunk, 3), src_id, 0)]).wait()
    path = _slot_path(store, chunk)
    assert path.stat().st_size == 4096 + 4 * SEG
    assert path.stat().st_blocks * 512 >= 4096 + 4 * SEG


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
    # 未受理的 chunk 不进驻留集合、也不产出对象文件
    slots = _slot_path(store, bytes(resident[0][:16])).parent
    assert len(list(slots.glob("*.obj"))) <= 2
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
        max_slots=2,
        allocator_enabled=False,
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
    assert store._layout.target_offset(chunk) == 4096
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
        "threads_per_block",
        "handle_cache_capacity",
        "prp_cache_capacity",
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


@pytest.mark.parametrize("value", [65536, 2**40])
def test_preset_accepts_wide_unsigned_field(tmp_path, value):
    """uint64 字段（stripe_unit）必须接受正常正数，包括超出 uint32 的值。

    回归：范围检查曾在有符号域里比较，而 ``uint64_t`` 的 max() 转回 int64 是
    -1 —— 于是 ``value > kMax`` 对任何正数都成立，stripe_unit=65536 被解析层
    拒绝，8 卡条带真机运行直接起不来。只有真机跑才会暴露（单测当时没有覆盖
    uint64 字段），所以这条测试按"解析层不得报 out of range"断言。
    """
    tutti_runtime = _load_bindings_runtime()
    # stripe_unit 只存在于条带 preset（单盘 preset 没有这个键）。
    preset = {
        "devices": [{"pci_bdf": "0000:00:00.0", "mount_path": str(tmp_path)}],
        "stripe_unit": value,
    }
    with pytest.raises(Exception) as excinfo:
        tutti_runtime.make_striped_nvme_runtime(preset)
    message = str(excinfo.value)
    assert "stripe_unit" not in message, message
    assert "out of range" not in message, message


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

    preset = {"daemon_config": str(daemon_path), "device_id": 1, "gpu_id": 0}
    derived = _derive_device_fields(preset, yaml)
    device = derived["device"]
    assert device["pci_bdf"] == "0000:4b:00.0"       # daemon 推导
    assert device["mount_path"] == "/mnt/nvme1"      # daemon 推导
    assert device["namespace_id"] == 2               # daemon 推导
    assert device["backing_device"] == "/dev/snvme1n2"
    assert "type" not in derived                      # 元键不注入 runtime preset

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
            {"daemon_config": str(daemon_path), "device_id": 9}, yaml
        )


def test_derive_striped_devices_from_daemon(tmp_path):
    """striped preset：devices 的 device_id 列表按 daemon 配置逐个推导。"""
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
        "type": "striped",
        "daemon_config": str(daemon_path),
        "devices": [{"device_id": 0}, {"device_id": 2}],
        "stripe_unit": 65536,
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


def test_striped_store_derives_mounts_from_preset(tmp_path):
    """striped store：mounts 从 preset.devices 的 daemon 推导结果取得。"""
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
        "type": "striped",
        "daemon_config": str(daemon_path),
        "devices": [{"device_id": 0}, {"device_id": 2}],
        "stripe_unit": 4096,
    }
    store = TuttiKVStore(
        tmp_path / "meta-root", 8, SEG,
        layout="striped", preset=preset, stripe_unit=4096,
    )
    assert store._layout.mounts == (
        str(nvme0.resolve()), str(nvme2.resolve())
    )


def test_striped_store_rejects_stripe_unit_mismatch(tmp_path):
    """options 与 preset 的 stripe_unit 不一致必须 fail-closed。"""
    import yaml

    daemon_path = tmp_path / "daemon.yaml"
    daemon_path.write_text(yaml.safe_dump({"nvmes": [
        {"device_id": 0, "pci_addr": "0000:08:00.0",
         "backing_mount_path": str(tmp_path / "nvme0"), "namespace_id": 1},
    ]}))
    preset = {
        "type": "striped",
        "daemon_config": str(daemon_path),
        "devices": [{"device_id": 0}],
        "stripe_unit": 8192,
    }
    with pytest.raises(RuntimeError, match="stripe_unit 不一致"):
        TuttiKVStore(
            tmp_path / "meta-root", 8, SEG,
            layout="striped", preset=preset, stripe_unit=4096,
        )


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

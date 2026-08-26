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

from stores.tutti_nvme.layout import Layout, decode_io_key
from stores.tutti_nvme.store import (
    TuttiKVStore,
    _TuttiCompletion,
    _derive_device_fields,
)

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
        weakref.finalize(store, shutil.rmtree, root, True)
        return store

    register(
        "tutti_nvme",
        make_store,
        segment_bytes=4096,
        granularity=4096,
        capacity_chunks=8,
    )


def make_store(tmp_path, num_chunks=8, runtime=None) -> TuttiKVStore:
    return TuttiKVStore(
        root=tmp_path / "pool",
        num_chunks=num_chunks,
        segment_bytes=SEG,
        runtime=runtime or FakeRuntime(),
    )


def test_scan_refreshes_markers_from_sibling_store(tmp_path):
    """A scheduler-side store observes a worker's newly committed marker."""
    root = tmp_path / "pool"
    worker = TuttiKVStore(
        root=root, num_chunks=8, segment_bytes=SEG, runtime=FakeRuntime()
    )
    scheduler = TuttiKVStore(
        root=root, num_chunks=8, segment_bytes=SEG, runtime=FakeRuntime()
    )
    worker.open()
    scheduler.open()
    assert scheduler.scan() == []

    src = bytearray(b"x" * SEG)
    buffer_id = worker.register_buffer(src, SEG)
    key = io_key(b"sibling".ljust(16, b"_"), 0)
    worker.put_batch([(key, buffer_id, 0)]).wait()

    assert scheduler.scan() == [key]
    worker.close()
    scheduler.close()


# ---------- partial-commit 窗口重发 ----------


def test_partial_commit_retried(tmp_path):
    """第一轮拒一半 → 重发后整批完成，数据完整落盘。"""
    runtime = FakeRuntime(
        reject_plan=lambda rnd, n: set(range(0, n, 2)) if rnd == 0 else set()
    )
    store = make_store(tmp_path, runtime=runtime)
    store.open()
    src = bytearray(4 * SEG)
    src_id = store.register_buffer(src, SEG)
    keys = [io_key(bytes([i]) * 16, i % 3) for i in range(4)]
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
    store = make_store(tmp_path, runtime=runtime)
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

    runtime = FakeRuntime(supports_multi_stream=True, bound_accel_id=0)
    store = TuttiKVStore(
        root=tmp_path / "pool", num_chunks=2, segment_bytes=SEG,
        runtime=runtime, io_stream="auto", preset={"gpu_id": 0},
    )
    store.open()
    assert store._stream_mode == "dual"
    assert store._read_stream != store._write_stream
    assert store._read_stream_obj is not store._write_stream_obj
    with store.stream_context("read") as selected:
        assert selected is store._read_stream_obj
    with store.stream_context("write") as selected:
        assert selected is store._write_stream_obj

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
    assert store._read_stream_obj.waited == [event]
    assert store._write_stream_obj.waited == [event]
    assert store.record_read_event().recorded_on is store._read_stream_obj
    assert store.record_write_event().recorded_on is store._write_stream_obj
    store.close()
    assert store._read_stream_obj is None
    assert store._write_stream_obj is None


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


def test_explicit_legacy_stream_is_shared_for_both_directions(tmp_path):
    """An explicit io_stream keeps the pre-existing single-stream contract."""
    import types

    runtime = FakeRuntime(supports_multi_stream=True)
    store = TuttiKVStore(
        root=tmp_path / "pool", num_chunks=2, segment_bytes=SEG,
        runtime=runtime, io_stream=1234,
    )
    store.open()
    assert store._stream_mode == "shared"
    assert store._read_stream == store._write_stream == 1234
    event = types.SimpleNamespace(synchronize=lambda: None)
    store.wait_read_event(event)
    store.wait_write_event(event)


def test_owned_runtime_shutdown_precedes_stream_release(tmp_path, monkeypatch):
    """close keeps both stream owners alive through runtime shutdown."""
    import types
    import torch
    import stores.tutti_nvme.store as store_module

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


def test_layer_segments_share_chunk_file(tmp_path):
    """同 chunk 各层共用一个数据文件；层段 offset = layer × segment。"""
    store = make_store(tmp_path)
    store.open()
    src = bytearray(SEG)
    src_id = store.register_buffer(src, SEG)
    chunk = b"\x11" * 16
    layers = [0, 2, 5]
    for layer in layers:
        src[:] = bytes([0x40 + layer]) * SEG
        store.put_batch([(io_key(chunk, layer), src_id, 0)]).wait()

    data_files = list((Path(store._layout.root) / "chunks").glob("*.bin"))
    assert len(data_files) == 1  # 一个 chunk 一个文件
    data = data_files[0].read_bytes()
    assert len(data) == 6 * SEG  # 最大层 5 → 实写扩展到 6 段
    for layer in layers:
        segment = data[layer * SEG:(layer + 1) * SEG]
        assert segment == bytes([0x40 + layer]) * SEG
        assert data_files[0].name == chunk.hex() + ".bin"


def test_real_write_extension(tmp_path):
    """按需扩展是实写（物理块就位），非稀疏跳写。"""
    store = make_store(tmp_path)
    store.open()
    src = bytearray(SEG)
    src_id = store.register_buffer(src, SEG)
    store.put_batch([(io_key(b"\x22" * 16, 3), src_id, 0)]).wait()
    path = Path(store._layout.root) / "chunks" / ((b"\x22" * 16).hex() + ".bin")
    assert path.stat().st_size == 4 * SEG
    assert path.stat().st_blocks * 512 >= 4 * SEG  # 物理块已分配（非稀疏）


def test_get_missing_layer_of_live_chunk_raises(tmp_path):
    """同 chunk 已写层 0/2，get 层 1（无 marker）→ ValueError。"""
    store = make_store(tmp_path)
    store.open()
    src = bytearray(SEG)
    src_id = store.register_buffer(src, SEG)
    chunk = b"\x33" * 16
    for layer in (0, 2):
        store.put_batch([(io_key(chunk, layer), src_id, 0)]).wait()
    with pytest.raises(ValueError):
        store.get_batch([(io_key(chunk, 1), src_id, 0)])


# ---------- 容量 ----------


def test_capacity_rejected_without_side_effects(tmp_path):
    """超容量的批整体拒绝，且不触碰文件系统。"""
    store = make_store(tmp_path, num_chunks=2)
    store.open()
    src = bytearray(3 * SEG)
    src_id = store.register_buffer(src, SEG)
    keys = [io_key(bytes([i]) * 16, 0) for i in range(3)]
    with pytest.raises(ValueError, match="容量"):
        store.put_batch([(k, src_id, i * SEG) for i, k in enumerate(keys)])
    chunks_dir = Path(store._layout.root) / "chunks"
    assert list(chunks_dir.iterdir()) == []  # 未建任何文件


# ---------- drop 与槽位回收 ----------


def test_drop_all_layers_recycles_chunk_file(tmp_path):
    """一个 chunk 的全部层被 drop → 数据文件删除（槽位回收）。"""
    store = make_store(tmp_path, num_chunks=2)
    store.open()
    src = bytearray(SEG)
    src_id = store.register_buffer(src, SEG)
    chunk = b"\x44" * 16
    keys = [io_key(chunk, layer) for layer in (0, 1)]
    for layer in (0, 1):
        store.put_batch([(io_key(chunk, layer), src_id, 0)]).wait()
    chunk_files = list((Path(store._layout.root) / "chunks").glob("*.bin"))
    assert len(chunk_files) == 1

    store.drop([keys[0]])  # 剩一层 → 文件保留
    assert chunk_files[0].exists()
    store.drop([keys[1]])  # 全部层删光 → 文件回收
    assert not chunk_files[0].exists()
    assert store.scan() == []

    # 回收后的容量可复用
    store.put_batch([(io_key(b"\x55" * 16, 0), src_id, 0)]).wait()
    assert len(list((Path(store._layout.root) / "chunks").glob("*.bin"))) == 1


# ---------- 持久化与重开恢复 ----------


def test_reopen_recovers_scan(tmp_path):
    """close 后新实例 open 同一 root → scan 恢复在场 io_key 且数据可读。"""
    store = make_store(tmp_path)
    store.open()
    src = bytearray(2 * SEG)
    src_id = store.register_buffer(src, SEG)
    keys = [io_key(b"\x66" * 16, 0), io_key(b"\x77" * 16, 4)]
    src[:SEG] = b"\x0A" * SEG
    src[SEG:] = b"\x0B" * SEG
    store.put_batch([(keys[0], src_id, 0), (keys[1], src_id, SEG)]).wait()
    store.close()

    reopened = make_store(tmp_path)
    reopened.open()
    assert reopened.scan() == sorted(keys)
    dst = bytearray(2 * SEG)
    dst_id = reopened.register_buffer(dst, SEG)
    reopened.get_batch([(keys[0], dst_id, 0), (keys[1], dst_id, SEG)]).wait()
    assert dst[:SEG] == b"\x0A" * SEG
    assert dst[SEG:] == b"\x0B" * SEG


def test_marker_only_after_completion(tmp_path):
    """put_batch 返回时 marker 尚未创建（先数据后 marker 的崩溃安全时序）。"""
    store = make_store(tmp_path)
    store.open()
    src = bytearray(SEG)
    src_id = store.register_buffer(src, SEG)
    key = io_key(b"\x88" * 16, 1)
    completion = store.put_batch([(key, src_id, 0)])
    assert store.scan() == []  # 数据已落盘但 marker 未建
    completion.wait()
    assert store.scan() == [key]


def test_completion_wait_uses_runtime_notification_without_polling():
    """Completion waiting blocks in the runtime watcher, not 100ms polling."""
    import threading

    class BlockingRuntime:
        def __init__(self):
            self.done = threading.Event()
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

    runtime = BlockingRuntime()
    settled = []
    completion = _TuttiCompletion(runtime, [7], settled.append)
    assert completion.query() is False
    runtime.done.set()
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
    store = make_store(tmp_path)
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
    from stores.registry import create_store

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

    bindings = Path(__file__).resolve().parents[2] / "bindings" / "python"
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


# ---------- layout 单元 ----------


def test_decode_io_key_rules():
    assert decode_io_key(b"a" * 16 + (3).to_bytes(2, "little")) == (b"a" * 16, 3)
    assert decode_io_key(b"short") == (b"short", 0)  # <16B：整体即 chunk 身份
    assert decode_io_key(b"a" * 16 + b"\xff\x00") == (b"a" * 16, 255)
    with pytest.raises(ValueError):
        decode_io_key(b"")
    with pytest.raises(ValueError):
        decode_io_key("not-bytes")


def test_scan_ignores_stray_files(tmp_path):
    layout = Layout(tmp_path, SEG)
    layout.ensure_dirs()
    (tmp_path / "meta" / "stray.txt").write_text("junk")
    (tmp_path / "meta" / ("ab" + ".ok")).write_text("")
    assert layout.scan() == {bytes.fromhex("ab")}

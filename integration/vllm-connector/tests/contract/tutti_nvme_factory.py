"""tutti_nvme 契约套件支架：FakeRuntime + 工厂注册。

FakeRuntime 复刻 tutti_runtime 的 Python API 形态（caps/open_batch/
register_memory/submit/release_io/wait/shutdown），接受的请求在本地
文件系统上真搬运数据（os.pwrite/pread + ctypes），可注入按轮次的部分
拒收计划以驱动 store 的窗口重发逻辑。
"""

from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
import weakref
from collections import namedtuple

from stores.tutti_nvme.store import TuttiKVStore

FakeSubmitResult = namedtuple(
    "FakeSubmitResult",
    ["status_ok", "status_msg", "io_handle", "initial_states", "rejected"],
)


def _reject_none(round_idx: int, count: int) -> set[int]:
    return set()


class FakeRuntime:
    """tutti_runtime 的文件系统级 fake（数据真搬运，提交即完成）。"""

    def __init__(self, reject_plan=None):
        self._next_ticket = 0
        self._targets: dict[int, str] = {}
        self._memories: dict[int, tuple[int, int]] = {}
        self._io_done: set[int] = set()
        self._released: set[int] = set()
        self.register_calls: list[tuple] = []
        self.submit_rounds = 0
        self.shutdown_called = False
        self._reject_plan = reject_plan or _reject_none

    def caps(self):
        return {
            "target": ["stub"],
            "memory": ["host"],
            "length_alignment_bytes": 1,
            "max_single_io_bytes": None,
            "max_batch_requests": None,
            "max_in_flight_operations": None,
        }

    def open_batch(self, uris):
        tickets = []
        for uri in uris:
            if not uri.startswith("file://"):
                raise ValueError(f"FakeRuntime 仅支持 file:// uri：{uri!r}")
            self._next_ticket += 1
            self._targets[self._next_ticket] = uri[len("file://") :]
            tickets.append(self._next_ticket)
        return tickets

    def register_memory(self, addr, size, kind, accel_id=-1, io_granularity=0):
        self.register_calls.append((addr, size, kind, accel_id, io_granularity))
        self._next_ticket += 1
        self._memories[self._next_ticket] = (addr, size)
        return self._next_ticket

    def submit(self, requests, accel_id=-1, stream=None, execution="device"):
        rejected = set(self._reject_plan(self.submit_rounds, len(requests)))
        self.submit_rounds += 1
        for idx, req in enumerate(requests):
            if idx not in rejected:
                self._execute(req)
        self._next_ticket += 1
        handle = self._next_ticket
        self._io_done.add(handle)
        return FakeSubmitResult(
            status_ok=True,
            status_msg="",
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

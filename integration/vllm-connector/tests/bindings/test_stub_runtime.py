"""Stub-mode contract tests for tutti_runtime.

All tests run hardware-free via make_stub_runtime:
assembly -> open_batch -> register_memory(host) -> submit -> force-complete
-> wait -> release_io -> shutdown.
"""

import ctypes

import pytest

import tutti_runtime as trt


@pytest.fixture
def stub_rt():
    return trt.make_stub_runtime()


def _host_buffer(size):
    return ctypes.create_string_buffer(size)


class TestCaps:
    def test_stub_caps_alignments_are_one_and_limits_unlimited(self, stub_rt):
        assert stub_rt.caps() == {
            "target_alignment_bytes": 1,
            "memory_alignment_bytes": 1,
            "length_alignment_bytes": 1,
            "max_single_io_bytes": 0,
            "max_batch_requests": 0,
            "max_in_flight_operations": 0,
        }


class TestOpenBatch:
    def test_empty_uri_list_returns_empty_list(self, stub_rt):
        assert stub_rt.open_batch([]) == []

    def test_open_batch_returns_int_handles(self, stub_rt):
        handles = stub_rt.open_batch(["file:///tmp/a.bin", "file:///tmp/b.bin"])
        assert len(handles) == 2
        assert all(isinstance(h, int) for h in handles)
        assert len(set(handles)) == 2


class TestRegisterMemory:
    def test_register_host_memory_returns_int(self, stub_rt):
        buf = _host_buffer(4096)
        h = stub_rt.register_memory(
            ctypes.addressof(buf), 4096, "host"
        )
        assert isinstance(h, int)

    def test_invalid_kind_raises_value_error(self, stub_rt):
        buf = _host_buffer(4096)
        with pytest.raises(ValueError):
            stub_rt.register_memory(ctypes.addressof(buf), 4096, "pinned")


class TestSubmitValidation:
    def test_invalid_direction_raises_value_error(self, stub_rt):
        targets = stub_rt.open_batch(["file:///tmp/a.bin"])
        buf = _host_buffer(4096)
        mem = stub_rt.register_memory(ctypes.addressof(buf), 4096, "host")
        req = (targets[0], 0, mem, 0, 512, "erase")
        with pytest.raises(ValueError):
            stub_rt.submit([req], accel_id=-1, stream=None, execution="host")

    def test_invalid_execution_raises_value_error(self, stub_rt):
        targets = stub_rt.open_batch(["file:///tmp/a.bin"])
        buf = _host_buffer(4096)
        mem = stub_rt.register_memory(ctypes.addressof(buf), 4096, "host")
        req = (targets[0], 0, mem, 0, 512, "read")
        with pytest.raises(ValueError):
            stub_rt.submit([req], accel_id=-1, stream=None, execution="both")

    def test_device_execution_requires_stream(self, stub_rt):
        targets = stub_rt.open_batch(["file:///tmp/a.bin"])
        buf = _host_buffer(4096)
        mem = stub_rt.register_memory(ctypes.addressof(buf), 4096, "host")
        req = (targets[0], 0, mem, 0, 512, "read")
        with pytest.raises(ValueError):
            stub_rt.submit([req], accel_id=-1, stream=None, execution="device")


class TestStubEndToEnd:
    def test_full_stub_lifecycle(self, stub_rt):
        # 1. open 2 uris
        targets = stub_rt.open_batch(
            ["file:///tmp/t101-a.bin", "file:///tmp/t101-b.bin"]
        )
        assert len(targets) == 2

        # 2. register host memory
        buf = _host_buffer(8192)
        mem = stub_rt.register_memory(ctypes.addressof(buf), 8192, "host")

        # 3. submit 3 requests, all accepted
        requests = [
            (targets[0], 0, mem, 0, 1024, "read"),
            (targets[0], 4096, mem, 2048, 512, "write"),
            (targets[1], 0, mem, 4096, 1024, "read"),
        ]
        result = stub_rt.submit(requests, accel_id=-1, stream=None,
                                execution="host")
        assert result.status_ok, result.status_msg
        assert result.initial_states == [True, True, True]
        assert result.rejected == []
        assert result.io_handle is not None

        # 4. force-complete (stub only), then wait
        stub_rt.testing_force_complete(result.io_handle)
        observation, state = stub_rt.wait(result.io_handle, timeout_ms=1000)
        assert observation == "OK"
        assert state == "COMPLETED"

        # 5. release + shutdown
        stub_rt.release_io(result.io_handle)
        stub_rt.shutdown(timeout_ms=1000)

    def test_wait_before_terminal_times_out(self, stub_rt):
        targets = stub_rt.open_batch(["file:///tmp/t101-tmo.bin"])
        buf = _host_buffer(4096)
        mem = stub_rt.register_memory(ctypes.addressof(buf), 4096, "host")
        result = stub_rt.submit(
            [(targets[0], 0, mem, 0, 512, "read")],
            accel_id=-1, stream=None, execution="host",
        )
        assert result.io_handle is not None
        observation, state = stub_rt.wait(result.io_handle, timeout_ms=50)
        assert observation == "TIMEOUT"
        assert state == ""
        stub_rt.testing_force_complete(result.io_handle, "COMPLETED")
        stub_rt.wait(result.io_handle, timeout_ms=1000)
        stub_rt.release_io(result.io_handle)
        stub_rt.shutdown(timeout_ms=1000)

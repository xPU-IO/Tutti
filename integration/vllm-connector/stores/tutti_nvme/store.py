"""TuttiKVStore——tutti IO runtime 之上的层盲 KV 存储。
store 对 io_key 是纯映射（私有解读见 layout.py），数据面全部经由
runtime.submit 的 DMA 请求表达：put = memory→target（write），
get = target→memory（read）。**批是一等公民**：一次 put/get 对应
一批请求、一个 Completion；partial-commit 批次在内部窗口重发
（连续两轮零接受视为失败上抛）。

runtime 注入（测试/fake）或由 TUTTI_NVME_PRESET 构造（真机）：
环境变量值为 yaml/json 内联文本或文件路径；preset 可携带
daemon_config + device_id，设备事实（pci_bdf/mount_path/namespace_id）
从 daemon 配置推导——硬件信息单一来源，preset 只写用户级参数。
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from .layout import Layout, decode_io_key
from .object_pool import ObjectPool, PoolConfig
from .commit import RankCommitRecord, remove_rank_commit, write_rank_commit
from .striped_layout import StripedLayout
from engine.nvtx import range as nvtx_range

#: 运行日志（池归属校验等部署问题的非静默说明）。
_LOG = logging.getLogger(__name__)

#: KV IO 页基：register_buffer 粒度必须为其正倍数（对齐 NVMe/DMA 路径）。
_IO_PAGE_BYTES = 4096

#: 后台完成观察者每次等待的上限。runtime 本身以条件变量唤醒；
#: 有上限是为了在 runtime shutdown/异常实现下也能及时退出。
_COMPLETION_WAIT_MS = 1000


def _buffer_info(buffer) -> tuple[int, int, str, int] | None:
    """提取 (地址, 字节数, memory 类型, accel_id)；无法定位稳定地址 → None。

    支持：torch tensor（data_ptr；CUDA → device）、bytearray、ctypes 数组。
    """
    data_ptr = getattr(buffer, "data_ptr", None)
    if callable(data_ptr):
        try:
            addr = int(data_ptr())
            if hasattr(buffer, "numel") and hasattr(buffer, "element_size"):
                size = int(buffer.numel()) * int(buffer.element_size())
            else:
                size = int(getattr(buffer, "nbytes", 0))
        except (TypeError, ValueError, RuntimeError):
            return None
        if addr == 0 or size <= 0:
            return None
        if getattr(buffer, "is_cuda", False):
            return addr, size, "device", int(buffer.get_device())
        return addr, size, "host", -1
    if isinstance(buffer, bytearray):
        if not buffer:
            return None
        keeper = (ctypes.c_char * len(buffer)).from_buffer(buffer)
        return ctypes.addressof(keeper), len(buffer), "host", -1
    try:
        addr = ctypes.addressof(buffer)
        size = ctypes.sizeof(buffer)
    except (TypeError, ValueError):
        return None
    if addr == 0 or size <= 0:
        return None
    return addr, size, "host", -1


@dataclass(frozen=True)
class _SubmittedHandle:
    handle: int
    batch_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _TargetCacheEntry:
    ticket: int
    size: int
    generation: int


@dataclass(frozen=True)
class TuttiTerminalResult:
    handle: int
    observation: str
    state: str
    confirmed_bytes: int = 0
    timeout_seen: bool = False
    failed_request_indices: tuple[int, ...] = ()
    failure_scope: str = "NONE"
    failure_kind: str | None = None
    first_failed_entry: int | None = None
    raw_cq_status: int | None = None
    message: str = ""
    batch_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class TuttiBatchResult:
    ok: bool
    failed_batch_indices: tuple[int, ...]
    timeout_seen: bool
    failures: tuple[TuttiTerminalResult, ...]
    results: tuple[TuttiTerminalResult, ...]


class _TuttiCompletion:
    """一批 runtime IO 的完成句柄，terminal 详情在 release 后仍保留。"""

    def __init__(self, runtime, handles, on_settled, *, auto_watch: bool = True):
        self._runtime = runtime
        self._submitted = [
            item if isinstance(item, _SubmittedHandle)
            else _SubmittedHandle(int(item))
            for item in handles
        ]
        self._handles = [item.handle for item in self._submitted]
        self._on_settled = on_settled
        self._settled = False
        self._failed = False
        self._failure_message = "tutti IO 批失败"
        self._terminal: bool | None = None
        self._terminal_results: dict[int, TuttiTerminalResult] = {}
        self._batch_result: TuttiBatchResult | None = None
        self._handles_released = False
        self._terminal_lock = threading.Lock()
        self._done_callbacks = []
        self._ready = threading.Event()
        self._watcher = None
        if auto_watch:
            self._start_watcher()

    def query(self) -> bool:
        if self._settled:
            return not self._failed
        if not self._ready.is_set():
            self._probe_runtime()
        if not self._ready.is_set():
            return False
        return self._finish()

    def wait(self, timeout: float | None = None) -> None:
        result = self.wait_result(timeout)
        if not result.ok:
            raise RuntimeError(self._failure_message)

    def wait_result(self, timeout: float | None = None) -> TuttiBatchResult:
        if self._settled:
            assert self._batch_result is not None
            return self._batch_result
        self._start_watcher()
        with nvtx_range("tutti.runtime.wait"):
            if not self._ready.wait(timeout):
                raise TimeoutError("等待 tutti IO 批超时")
        self._finish()
        assert self._batch_result is not None
        return self._batch_result

    def wait_detail(self, timeout: float | None = None) -> TuttiBatchResult:
        return self.wait_result(timeout)

    def add_done_callback(self, callback) -> None:
        with self._terminal_lock:
            if self._settled:
                result = self._batch_result
            else:
                self._done_callbacks.append(callback)
                return
        callback(result)

    def _start_watcher(self) -> None:
        with self._terminal_lock:
            if self._watcher is not None or self._terminal is not None:
                return
            self._watcher = threading.Thread(
                target=self._watch_runtime,
                name="tutti-io-completion",
                daemon=True,
            )
            self._watcher.start()

    def _observe(self, submitted: _SubmittedHandle,
                 timeout_ms: int) -> TuttiTerminalResult:
        structured_wait = getattr(self._runtime, "wait_result", None)
        if callable(structured_wait):
            raw = structured_wait(submitted.handle, timeout_ms)
            return TuttiTerminalResult(
                handle=submitted.handle,
                observation=str(raw.observation),
                state=str(raw.state),
                confirmed_bytes=int(raw.confirmed_bytes),
                timeout_seen=bool(raw.timeout_seen),
                failed_request_indices=tuple(raw.failed_request_indices),
                failure_scope=str(raw.failure_scope),
                failure_kind=raw.failure_kind,
                first_failed_entry=raw.first_failed_entry,
                raw_cq_status=raw.raw_cq_status,
                message=str(raw.message),
                batch_indices=submitted.batch_indices,
            )
        observation, state = self._runtime.wait(submitted.handle, timeout_ms)
        failed = state == "FAILED"
        return TuttiTerminalResult(
            handle=submitted.handle,
            observation=observation,
            state=state,
            failure_scope="WHOLE_OPERATION" if failed else "NONE",
            failure_kind="UNKNOWN" if failed else None,
            batch_indices=submitted.batch_indices,
        )

    def _record_result(self, result: TuttiTerminalResult) -> None:
        with self._terminal_lock:
            self._terminal_results.setdefault(result.handle, result)

    def _probe_runtime(self) -> None:
        """做一次非阻塞观察，避免 query 对 watcher 启动存在竞态。"""
        try:
            observed = []
            for submitted in self._submitted:
                result = self._observe(submitted, 0)
                if (result.observation == "TIMEOUT" or
                        result.state not in ("COMPLETED", "FAILED")):
                    return
                observed.append(result)
            for result in observed:
                self._record_result(result)
            failures = [result for result in observed
                        if result.observation != "OK" or result.state == "FAILED"]
            self._mark_terminal(not failures,
                                self._format_failure(failures[0])
                                if failures else None)
        except Exception as exc:
            self._mark_terminal(False, f"tutti IO 等待异常：{exc}")

    def _watch_runtime(self) -> None:
        """阻塞等待并 drain 每个 partial-commit handle，恰好记录一次。"""
        try:
            failures = []
            for submitted in self._submitted:
                while True:
                    result = self._observe(submitted, _COMPLETION_WAIT_MS)
                    if result.observation == "TIMEOUT":
                        continue
                    if (result.observation != "OK" or
                            result.state in ("COMPLETED", "FAILED")):
                        self._record_result(result)
                        if (result.observation != "OK" or
                                result.state == "FAILED"):
                            failures.append(result)
                        break
            self._mark_terminal(not failures,
                                self._format_failure(failures[0])
                                if failures else None)
            self._release_handles()
        except Exception as exc:
            self._mark_terminal(False, f"tutti IO 等待异常：{exc}")
            self._release_handles()

    def _mark_terminal(self, ok: bool, message: str | None) -> None:
        with self._terminal_lock:
            if self._terminal is not None:
                return
            self._terminal = ok
            if message:
                self._failure_message = message
            self._ready.set()

    @staticmethod
    def _format_failure(result: TuttiTerminalResult) -> str:
        kind = f"，kind={result.failure_kind}" if result.failure_kind else ""
        message = f"：{result.message}" if result.message else ""
        return f"tutti IO 失败（io handle={result.handle}{kind}）{message}"

    def _build_batch_result(self) -> TuttiBatchResult:
        results = tuple(
            self._terminal_results[item.handle] for item in self._submitted
            if item.handle in self._terminal_results
        )
        failures = tuple(
            result for result in results
            if result.observation != "OK" or result.state == "FAILED"
        )
        failed_batch_indices: set[int] = set()
        for result in failures:
            if (result.failure_scope == "REQUEST_INDICES" and
                    result.failed_request_indices):
                for index in result.failed_request_indices:
                    if 0 <= index < len(result.batch_indices):
                        failed_batch_indices.add(result.batch_indices[index])
            else:
                failed_batch_indices.update(result.batch_indices)
        return TuttiBatchResult(
            ok=not failures,
            failed_batch_indices=tuple(sorted(failed_batch_indices)),
            timeout_seen=any(result.timeout_seen for result in failures),
            failures=failures,
            results=results,
        )

    def _release_handles(self) -> None:
        """Release terminal Runtime handles once, without semantic settle."""
        with self._terminal_lock:
            if self._handles_released:
                return
            self._handles_released = True
            handles = tuple(self._handles)
        for handle in handles:
            try:
                self._runtime.release_io(handle)
            except Exception:
                pass

    def _finish(self) -> bool:
        with self._terminal_lock:
            terminal = self._terminal
            if terminal is None:
                return False
            if self._settled:
                return not self._failed
            self._settled = True
            self._failed = not terminal
            self._batch_result = self._build_batch_result()
        self._release_handles()
        self._on_settled(terminal)
        with self._terminal_lock:
            callbacks = self._done_callbacks
            self._done_callbacks = []
            result = self._batch_result
        for callback in callbacks:
            callback(result)
        return terminal

    def _settle(self, ok: bool) -> None:
        self._mark_terminal(ok, None)
        self._finish()


class TuttiKVStore:
    """tutti runtime 之上的 KVStore SPI 实现（层盲，io_key 纯映射）。"""

    def __init__(self, root, num_chunks: int, segment_bytes: int,
                 runtime=None, io_stream=None, preset=None,
                 layout="file_per_chunk", mounts=None, stripe_unit=None,
                 initial_slots=None, low_watermark=None,
                 high_watermark=None, max_slots=None,
                 pool_wait_timeout_s: float = 5.0,
                 allocator_enabled: bool = True,
                 rank_id: int = 0, tp_size: int = 1):
        """preset 为 dict 时优先于 TUTTI_NVME_PRESET 环境变量构造 runtime。

        preset 的字符串值恰为纯十进制整数时转为 int（配置占位符替换后
        的数字字符串由此归一，如 device_id / gpu_id）。

        ``layout="striped"`` 选择条带逻辑 target；其 ``mounts`` 与
        ``stripe_unit`` 仅作用于该布局，默认 file_per_chunk 不变。
        """
        if num_chunks <= 0:
            raise ValueError(f"num_chunks 必须为正数，得到 {num_chunks}")
        if segment_bytes <= 0:
            raise ValueError(f"segment_bytes 必须为正数，得到 {segment_bytes}")
        if (not isinstance(rank_id, int) or isinstance(rank_id, bool)
                or rank_id < 0):
            raise ValueError(f"rank_id must be a non-negative integer: {rank_id!r}")
        if (not isinstance(tp_size, int) or isinstance(tp_size, bool)
                or tp_size <= rank_id):
            raise ValueError(
                f"tp_size must be greater than rank_id: {tp_size!r} <= {rank_id!r}"
            )
        self._root = Path(root)
        self._rank_id = rank_id
        self._tp_size = tp_size
        max_slots = num_chunks if max_slots is None else max_slots
        initial_slots = min(32, max_slots) if initial_slots is None else initial_slots
        high_watermark = initial_slots if high_watermark is None else high_watermark
        low_watermark = high_watermark // 2 if low_watermark is None else low_watermark
        pool_config = PoolConfig(
            initial_slots=initial_slots,
            low_watermark=low_watermark,
            high_watermark=high_watermark,
            max_slots=max_slots,
            wait_timeout_s=float(pool_wait_timeout_s),
        )
        pool_config.validate()
        self._num_chunks = min(num_chunks, max_slots)
        self._segment_bytes = segment_bytes
        self._runtime = runtime
        self._own_runtime = runtime is None
        self._preset = _normalize_preset(preset) if preset is not None else None
        self._key_namespace: bytes | None = None
        if layout in (None, "file_per_chunk", "file"):
            self._layout = Layout(self._root, segment_bytes)
        elif layout == "striped":
            if mounts is None:
                mounts = _preset_mounts(self._preset)
            if stripe_unit is None:
                raise ValueError("striped target 必须提供 stripe_unit")
            self._layout = StripedLayout(
                self._root, segment_bytes, mounts, stripe_unit
            )
        else:
            raise ValueError(f"未知 tutti_nvme layout：{layout!r}")
        self._object_pool = ObjectPool(
            self._layout, pool_config, allocator_enabled=allocator_enabled
        )
        self._layout.attach_object_pool(self._object_pool)
        self._opened = False
        self._live: set[bytes] = set()
        self._buffers: dict[int, tuple[int, int]] = {}
        self._mem_cache: dict[tuple[int, int], int] = {}
        self._targets: dict[str, _TargetCacheEntry] = {}
        self._inflight_by_chunk: dict[bytes, set[_TuttiCompletion]] = {}
        self._inflight_lock = threading.Lock()
        self._deferred_completions: list[_TuttiCompletion] = []
        self._keepers: list = []  # 持有 ctypes 视图防 GC
        self._next_buffer_id = 0
        self._accel_id = -1
        # 'auto' 在 open() 惰性解析为专用 IO 流（见 _resolve_auto_stream）；
        # 其余取值（int 句柄 / None）原样使用。
        self._io_stream_raw = io_stream
        self._io_stream = None if io_stream == "auto" else io_stream
        # Keep direction-specific handles private.  Workers use the context
        # and fence helpers below instead of replacing these values.
        self._read_stream = None
        self._write_stream = None
        self._read_copy_stream = None
        self._read_stream_obj = None
        self._write_stream_obj = None
        self._read_copy_stream_obj = None
        self._stream_mode = "host"
        self._stream_accel_id = None
        self._execution = "device"

    def begin_rank_commit(self, keys, generations) -> None:
        """Invalidate this rank's old visibility before overwriting payload."""
        self._require_open()
        keys = [bytes(key) for key in keys]
        if len(keys) != len(list(generations)):
            raise ValueError("rank commit keys/generations length mismatch")
        for key in keys:
            remove_rank_commit(self._root, key)

    def commit_rank_chunks(self, keys, generations) -> None:
        """Publish one rank record only after all local layer writes succeed."""
        self._require_open()
        keys = [bytes(key) for key in keys]
        generations = list(generations)
        if len(keys) != len(generations):
            raise ValueError("rank commit keys/generations length mismatch")
        if self._key_namespace is None:
            raise RuntimeError("rank commit requires a key namespace")
        num_layers = self._layout.layer_span
        if not num_layers:
            raise RuntimeError("rank commit requires complete layer geometry")
        slot_bytes = num_layers * self._segment_bytes
        marker_token = self._layout.marker_generation()
        if not marker_token:
            raise RuntimeError("rank commit requires a marker generation")
        for key, generation in zip(keys, generations):
            if not self._layout.pool_chunk_complete(key, num_layers):
                raise RuntimeError(
                    f"rank {self._rank_id} chunk {key.hex()} lacks complete markers"
                )
            pool_generation = self._layout.target_generation(key)
            if pool_generation <= 0:
                raise RuntimeError(
                    f"rank {self._rank_id} chunk {key.hex()} lacks pool generation"
                )
            record = RankCommitRecord(
                version=1,
                namespace=self._key_namespace.hex(),
                chunk_key=key.hex(),
                generation=str(generation),
                num_layers=num_layers,
                slot_bytes=slot_bytes,
                rank_id=self._rank_id,
                marker_generation=marker_token,
                pool_generation=pool_generation,
            )
            write_rank_commit(self._root, record)

    def abort_rank_commit(self, keys, generations=None) -> None:
        """Fail closed for this rank without deleting payload or peer data."""
        for key in keys:
            remove_rank_commit(self._root, bytes(key))

    # ---------- 生命周期 ----------

    @property
    def capacity_chunks(self) -> int:
        return self._num_chunks

    @property
    def max_in_flight_operations(self) -> int:
        """Runtime admission window; 0 means unknown/unbounded."""
        if self._runtime is None:
            return 0
        try:
            value = self._runtime.caps().get("max_in_flight_operations", 0)
            value = int(value or 0)
        except (AttributeError, TypeError, ValueError):
            return 0
        return max(value, 0)

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("tutti store 已 open")
        if self._runtime is None:
            if self._preset is not None:
                self._runtime = _build_runtime(self._preset)
            else:
                self._runtime = _build_runtime_from_env()
        self._layout.ensure_dirs()
        self._live = self._layout.scan()
        # 池归属校验：manifest 与命名空间不一致 → 空池语义（miss），
        # 禁止静默复用异构数据（不同模型/几何的旧池）。
        if self._key_namespace is not None:
            if not self._layout.check_namespace(self._key_namespace):
                _LOG.warning(
                    "池 %s 的命名空间 manifest 与当前配置不一致——按空池"
                    "处理（不读旧数据）；如需腾挪请人工清理",
                    self._root,
                )
                self._live = set()
        if self._io_stream_raw == "auto":
            self._resolve_auto_stream()
        else:
            self._configure_shared_stream(self._io_stream)
        self._sync_execution_mode()
        _LOG.warning(
            "TUTTI_STREAM_ROUTING rank=%d accel=%s mode=%s read_io=%s "
            "read_copy=%s write_io=%s",
            self._rank_id, self._stream_accel_id, self._stream_mode,
            self._read_stream, self._read_copy_stream, self._write_stream,
        )
        self._opened = True

    def _runtime_supports_multi_stream(self) -> bool:
        """Return the advertised two-stream capability, conservatively.

        Current bindings expose both fields from the assembled DataPath.
        Missing fields mean an older/unknown runtime, which conservatively
        selects the shared-stream compatibility path.
        """
        try:
            caps = self._runtime.caps()
        except Exception:
            return False
        try:
            return bool(caps.get("supports_multi_stream", False)) and int(
                caps.get("max_concurrent_streams", 0)
            ) >= 2
        except (TypeError, ValueError):
            return False

    def _runtime_accel(self) -> int:
        """Resolve one accelerator shared by runtime, read, and write streams."""
        preset_accel = None
        if isinstance(self._preset, dict) and "gpu_id" in self._preset:
            try:
                preset_accel = int(self._preset["gpu_id"])
            except (TypeError, ValueError):
                preset_accel = None
        bound_accel = None
        try:
            raw = self._runtime.caps().get("bound_accel_id")
            if raw is not None and int(raw) >= 0:
                bound_accel = int(raw)
        except (AttributeError, TypeError, ValueError):
            bound_accel = None
        if (bound_accel is not None and preset_accel is not None
                and bound_accel != preset_accel):
            raise RuntimeError(
                "runtime bound_accel_id 与 preset gpu_id 不一致："
                f"{bound_accel} != {preset_accel}"
            )
        return bound_accel if bound_accel is not None else (preset_accel or 0)

    def _configure_shared_stream(self, stream) -> None:
        self._read_stream = stream
        self._write_stream = stream
        self._read_copy_stream = stream
        self._read_stream_obj = None
        self._write_stream_obj = None
        self._read_copy_stream_obj = None
        self._stream_mode = "shared" if stream is not None else "host"
        if stream is None:
            return
        try:
            import torch

            if not torch.cuda.is_available():
                return
            accel = self._runtime_accel()
            copy_obj = torch.cuda.Stream(device=f"cuda:{accel}")
            self._read_copy_stream_obj = copy_obj
            self._read_copy_stream = int(copy_obj.cuda_stream)
            self._stream_accel_id = accel
        except (ImportError, RuntimeError):
            # Old host/fake runtimes may carry an opaque stream integer without
            # a usable CUDA device. Their compatibility behavior is unchanged.
            return

    def _resolve_auto_stream(self) -> None:
        """在 runtime 加速器上建方向化 IO 流并取其句柄。

        默认流句柄为 0，与绑定的空指针语义冲突（submit 会当作未给流
        拒绝）。落在 preset 的 gpu_id 设备上（runtime 校验流须属于自身
        加速器）。旧 runtime 或不支持 multi-stream 的 target 明确回退
        到共享专用流。
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("io_stream='auto' 需要 CUDA 可用")
        accel = self._runtime_accel()
        self._accel_id = accel
        self._stream_accel_id = accel
        read_obj = torch.cuda.Stream(device=f"cuda:{accel}")
        # Copy/reshape is not a Runtime/DataPath submit stream, so it does not
        # consume a backend concurrent-stream capability slot. It is always a
        # separate CUDA stream in auto mode, including a shared-IO fallback.
        read_copy_obj = torch.cuda.Stream(device=f"cuda:{accel}")
        if not self._runtime_supports_multi_stream():
            self._read_stream_obj = read_obj
            self._write_stream_obj = read_obj
            self._read_copy_stream_obj = read_copy_obj
            self._read_stream = int(read_obj.cuda_stream)
            self._write_stream = self._read_stream
            self._read_copy_stream = int(read_copy_obj.cuda_stream)
            self._io_stream = self._read_stream
            self._stream_mode = "shared"
            return

        write_obj = torch.cuda.Stream(device=f"cuda:{accel}")
        # Both streams are constructed on the same explicit CUDA device.  A
        # custom torch stream implementation must still expose a compatible
        # device, otherwise fail before any IO is submitted.
        for stream_obj in (read_obj, write_obj, read_copy_obj):
            device = getattr(stream_obj, "device", None)
            index = getattr(device, "index", None)
            if index is not None and int(index) != accel:
                raise RuntimeError(
                    "read/write CUDA stream 必须属于 runtime accel device "
                    f"{accel}，得到 {index}"
                )
        self._read_stream_obj = read_obj
        self._write_stream_obj = write_obj
        self._read_copy_stream_obj = read_copy_obj
        self._read_stream = int(read_obj.cuda_stream)
        self._write_stream = int(write_obj.cuda_stream)
        self._read_copy_stream = int(read_copy_obj.cuda_stream)
        self._io_stream = None
        self._stream_mode = "dual"

    def close(self) -> None:
        if not self._opened:
            return
        self._layout.close_object_pool()
        try:
            self._wait_chunk_io(set(self._inflight_by_chunk))
        except Exception:
            _LOG.exception("等待对象池 target IO 完成失败；继续关闭句柄")
        try:
            self._close_cached_targets(list(self._targets))
        finally:
            self._opened = False
        self._live = set()
        self._buffers = {}
        self._mem_cache = {}
        self._targets = {}
        self._inflight_by_chunk = {}
        self._keepers = []
        if self._own_runtime and self._runtime is not None:
            try:
                self._runtime.shutdown(5000)
            except Exception:
                pass
            self._runtime = None
        # Keep all CUDA stream objects alive through runtime shutdown, then
        # release them together so no handle can outlive its owner.
        self._read_stream = None
        self._write_stream = None
        self._read_copy_stream = None
        self._read_stream_obj = None
        self._write_stream_obj = None
        self._read_copy_stream_obj = None
        self._stream_mode = "host"
        self._stream_accel_id = None
        self._io_stream = None
        self._own_runtime = self._runtime is None

    def _sync_execution_mode(self) -> None:
        """推导 submit 执行模式：无 io_stream → host 路径；有则按 runtime
        能力（device 优先，host-only 回退）。"""
        if self._read_stream is None and self._write_stream is None:
            self._execution = "host"
            return
        try:
            caps = self._runtime.caps()
            memories = caps.get("memory", [])
        except Exception:
            return
        if "device" in memories:
            self._execution = "device"
        elif "host" in memories:
            self._execution = "host"

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("tutti store 未 open")

    # ---------- SPI ----------

    def register_buffer(self, buffer, granularity: int) -> int | None:
        """注册批量 IO 的内存缓冲，返回 store 内部 buffer_id。

        具体注册的是哪块内存：engine.bind 传入的 staging 环窗缓冲——
        一段连续 device 显存（或 host 内存），按"槽"等分，布局为
        slots × segment_bytes，槽 i 占据
        [i×segment_bytes, (i+1)×segment_bytes)。本方法的 buffer 即
        该段内存的起始视图；size 取自对象属性（numel×element_size）。

        granularity：单条批量 IO 的字节粒度 = 层段宽（segment_bytes）。
        DMA 以此为对齐与切分单位；必须是 _IO_PAGE_BYTES 的正倍数。

        返回 buffer_id：store 局部编号，映射到底层 (addr, size)。
        之后 put_batch/get_batch 的每条 IO 以
        (io_key, buffer_id, offset) 寻址，offset 即槽内字节偏移。

        语义：同一 (addr, size) 只向 runtime 注册一次（ticket 复用）；
        注册期间持有 buffer 引用防止回收；注册失败 → None（调用方
        视为该缓冲不可用于 DMA）。
        """
        self._require_open()
        if granularity is None or granularity <= 0:
            return None
        if granularity % _IO_PAGE_BYTES != 0:
            return None
        info = _buffer_info(buffer)
        if info is None:
            return None
        addr, size, kind, accel_id = info
        mem_ticket = self._mem_cache.get((addr, size))
        if mem_ticket is None:
            try:
                mem_ticket = self._runtime.register_memory(
                    addr, size, kind, accel_id=accel_id, io_granularity=granularity
                )
            except Exception:
                return None
            self._mem_cache[(addr, size)] = mem_ticket
            self._keepers.append(buffer)
        self._next_buffer_id += 1
        self._buffers[self._next_buffer_id] = (addr, size)
        return self._next_buffer_id

    def create_direct_transfer(
        self,
        kv_caches,
        *,
        num_layers: int,
        blocks_per_chunk: int,
        chunk_tokens: int,
        segment_bytes: int,
    ):
        """Return a runtime-native paged backend when one is available.

        The current generic ``tutti`` runtime exposes only contiguous
        ``register_memory`` buffers, so this returns ``None`` on existing
        deployments and the engine keeps the staged path.  A runtime that
        implements legacy-style paged registration can expose
        ``create_direct_transfer``; no store-side copy or staging buffer is
        then required.  Keeping the capability probe here makes the direct
        path explicit and prevents silently treating contiguous IO as paged.
        """
        factory = getattr(self._runtime, "create_direct_transfer", None)
        if not callable(factory):
            return None
        return factory(
            kv_caches,
            num_layers=num_layers,
            blocks_per_chunk=blocks_per_chunk,
            chunk_tokens=chunk_tokens,
            segment_bytes=segment_bytes,
        )

    def _stream_for(self, direction: str):
        if direction == "read":
            return self._read_stream, self._read_stream_obj
        if direction == "read_copy":
            return self._read_copy_stream, self._read_copy_stream_obj
        if direction == "write":
            return self._write_stream, self._write_stream_obj
        raise ValueError(f"未知 IO 方向：{direction!r}")

    def stream_context(self, direction: str):
        """Return a context for worker scatter/gather enqueue.

        The returned context selects the store-owned stream and does not
        expose a setter for replacing it.  Host/shared-stream fallbacks are a
        no-op context.
        """
        _, stream_obj = self._stream_for(direction)
        if stream_obj is None:
            return nullcontext()
        import torch

        return torch.cuda.stream(stream_obj)

    def _wait_event(self, direction: str, event) -> None:
        _, stream_obj = self._stream_for(direction)
        if event is None:
            return
        if stream_obj is not None:
            stream_obj.wait_event(event)
        # Host/fake stores have no stream to wait on.  Do not turn an event
        # dependency into a host synchronize; production reuse is guarded by
        # CUDA stream wait_event above, and completion draining owns host waits.

    def wait_read_event(self, event) -> None:
        """Make the read stream wait for a worker producer fence."""
        self._wait_event("read", event)

    def wait_read_copy_event(self, event) -> None:
        """Make the read-copy stream wait for a read completion fence."""
        self._wait_event("read_copy", event)

    def wait_write_event(self, event) -> None:
        """Make the write stream wait for a worker gather fence."""
        self._wait_event("write", event)

    def drain_deferred_completions(self) -> None:
        """Drain submit-error handles during engine-level abort/final drain."""
        pending = self._deferred_completions
        self._deferred_completions = []
        first_error = None
        for completion in pending:
            try:
                completion.wait_result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def wait_event(self, event) -> None:
        """Backward-compatible alias for the write-side fence."""
        self.wait_write_event(event)

    def _record_event(self, direction: str, event=None):
        _, stream_obj = self._stream_for(direction)
        if stream_obj is None:
            # An event without an owning CUDA stream is not recorded and must
            # never be handed to a downstream wait_event call.
            return None
        if event is None:
            import torch

            event = torch.cuda.Event()
        event.record(stream_obj)
        return event

    def record_read_event(self, event=None):
        """Record and return a fence on the store-owned read stream."""
        return self._record_event("read", event)

    def record_read_copy_event(self, event=None):
        """Record and return a fence after scatter on the read-copy stream."""
        return self._record_event("read_copy", event)

    def record_write_event(self, event=None):
        """Record and return a fence on the store-owned write stream."""
        return self._record_event("write", event)

    def wait_compute_event(self, event) -> None:
        """Enqueue a device-side wait on vLLM's restored current stream."""
        if event is None:
            return
        import torch

        torch.cuda.current_stream(device=self._stream_accel_id).wait_event(event)

    def read_copy_stream_handle(self) -> int:
        """Return the controlled stream handle used for scatter release fences."""
        return int(self._read_copy_stream or 0)

    def put_batch(self, batch) -> _TuttiCompletion:
        self._require_open()
        entries = self._normalize(batch, require_live=False)
        io_keys = [io_key for io_key, _, _ in entries]
        chunk_ids = tuple(dict.fromkeys(
            decode_io_key(io_key)[0] for io_key in io_keys
        ))
        self._layout.prepare_put(io_keys, self._num_chunks)
        targets = self._ensure_targets(entries)
        requests = []
        for io_key, buffer_id, offset in entries:
            chunk_id, layer = decode_io_key(io_key)
            uri = self._layout.target_uri(chunk_id)
            requests.append(
                (
                    targets[uri],
                    layer * self._segment_bytes,
                    self._mem_for(buffer_id),
                    offset,
                    self._segment_bytes,
                    "write",
                )
            )
        with nvtx_range(f"tutti.runtime.submit|op=write|requests={len(requests)}"):
            handles = self._submit_retry(requests, "write")
        completion = _TuttiCompletion(
            self._runtime, handles, lambda ok: self._on_put_settled(ok, io_keys)
        )
        self._track_completion(completion, chunk_ids)
        return completion

    def get_batch(self, batch) -> _TuttiCompletion:
        self._require_open()
        entries = self._normalize(batch, require_live=True)
        targets = self._ensure_targets(entries)
        chunk_ids = tuple(dict.fromkeys(
            decode_io_key(io_key)[0] for io_key, _, _ in entries
        ))
        requests = []
        for io_key, buffer_id, offset in entries:
            chunk_id, layer = decode_io_key(io_key)
            uri = self._layout.target_uri(chunk_id)
            requests.append(
                (
                    targets[uri],
                    layer * self._segment_bytes,
                    self._mem_for(buffer_id),
                    offset,
                    self._segment_bytes,
                    "read",
                )
            )
        with nvtx_range(f"tutti.runtime.submit|op=read|requests={len(requests)}"):
            handles = self._submit_retry(requests, "read")
        # Read completions are drained by KVEngine.wait_idle at host
        # completion time.  Do not start a background watcher while
        # start_load_kv is enqueueing the complete layer plan.
        completion = _TuttiCompletion(
            self._runtime, handles, lambda _ok: None
        )
        self._track_completion(completion, chunk_ids)
        return completion

    def drop(self, keys) -> None:
        self._require_open()
        io_keys = []
        chunk_ids = set()
        for key in keys:
            chunk_id, _ = decode_io_key(key)  # 类型/非空校验
            io_keys.append(bytes(key))
            chunk_ids.add(chunk_id)
        released = self._layout.releasable_chunks(io_keys)
        self._wait_chunk_io(released)
        self._close_cached_targets(
            [self._layout.target_uri(chunk_id) for chunk_id in released]
        )
        self._layout.drop(io_keys)
        self._live.difference_update(io_keys)

    def scan(self):
        self._require_open()
        # Refresh the local view from the marker directory. Layout.scan uses
        # a directory-generation cache, so this observes commits from a
        # sibling scheduler/worker process without rescanning unchanged pools.
        self._live = self._layout.scan()
        return sorted(self._live)

    def has(self, io_key) -> bool:
        """存活查询（O(1)）：读取侧跳过无数据层（混合注意力模型的
        线性注意力层从未落盘，属正常状态而非错误）。"""
        return io_key in self._live

    def set_key_namespace(self, namespace: bytes) -> None:
        """声明 key 命名空间（engine 构造期注入，open 前生效）。

        用于池归属 manifest 校验：不透明字节串，本层不解读字段。
        """
        if self._opened:
            raise RuntimeError("命名空间须在 open 之前注入")
        self._key_namespace = bytes(namespace)

    def set_layer_span(self, num_layers: int) -> None:
        """Bind rank-local geometry and synchronously create initial slots."""
        self._layout.set_layer_span(num_layers)

    def object_pool_snapshot(self) -> dict | None:
        return self._layout.object_pool_snapshot()

    def abort_chunks(self, chunk_ids) -> None:
        """Rollback incomplete chunks after the step has drained all IO."""
        chunks = tuple(dict.fromkeys(bytes(chunk_id) for chunk_id in chunk_ids))
        self._wait_chunk_io(chunks)
        self._close_cached_targets(
            [self._layout.target_uri(chunk_id) for chunk_id in chunks]
        )
        self._layout.abort_uncommitted(chunks)

    # ---------- 内部 ----------

    def _normalize(self, batch, require_live: bool):
        entries = []
        for item in batch:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError(f"批条目须为 (io_key, buffer_id, offset)，得到 {item!r}")
            io_key, buffer_id, offset = item
            decode_io_key(io_key)
            if buffer_id not in self._buffers:
                raise ValueError(f"未注册的 buffer id：{buffer_id}")
            if not isinstance(offset, int):
                raise ValueError(f"offset 须为 int，得到 {type(offset).__name__}")
            _, size = self._buffers[buffer_id]
            if offset < 0 or offset + self._segment_bytes > size:
                raise ValueError(
                    f"buffer 偏移越界：offset={offset} + 段长 "
                    f"{self._segment_bytes} > buffer {size} 字节"
                )
            if require_live and bytes(io_key) not in self._live:
                raise ValueError(f"get 未驻留 key：{bytes(io_key)!r}")
            entries.append((bytes(io_key), buffer_id, offset))
        return entries

    def _ensure_targets(self, entries) -> dict[str, int]:
        descriptors = []
        seen = set()
        stale = []
        for io_key, _, _ in entries:
            chunk_id, _ = decode_io_key(io_key)
            uri = self._layout.target_uri(chunk_id)
            if uri in seen:
                continue
            seen.add(uri)
            size = self._layout.target_size(chunk_id)
            generation = self._layout.target_generation(chunk_id)
            cached = self._targets.get(uri)
            if cached is not None and (
                cached.size != size or cached.generation != generation
            ):
                stale.append(uri)
                cached = None
            descriptors.append((uri, size, generation))
        if stale:
            self._close_cached_targets(stale)
        missing = [item for item in descriptors if item[0] not in self._targets]
        if missing:
            uris = [uri for uri, _, _ in missing]
            tickets = self._runtime.open_batch(uris)
            if len(tickets) != len(uris):
                raise RuntimeError("Runtime.open_batch returned wrong handle count")
            for (uri, size, generation), ticket in zip(missing, tickets):
                self._targets[uri] = _TargetCacheEntry(
                    int(ticket), int(size), int(generation)
                )
        return {uri: self._targets[uri].ticket for uri, _, _ in descriptors}

    def _close_cached_targets(self, uris) -> None:
        records = [(uri, self._targets[uri]) for uri in dict.fromkeys(uris)
                   if uri in self._targets]
        if not records:
            return
        tickets = [record.ticket for _, record in records]
        close_batch = getattr(self._runtime, "close_batch", None)
        if callable(close_batch):
            close_batch(tickets)
        else:
            close_target = getattr(self._runtime, "close_target", None)
            if not callable(close_target):
                raise RuntimeError(
                    "Runtime must expose close_batch or close_target for target lifecycle"
                )
            for ticket in tickets:
                close_target(ticket)
        for uri, _ in records:
            self._targets.pop(uri, None)

    def _track_completion(self, completion, chunk_ids) -> None:
        chunks = tuple(dict.fromkeys(bytes(chunk_id) for chunk_id in chunk_ids))
        with self._inflight_lock:
            for chunk_id in chunks:
                self._inflight_by_chunk.setdefault(chunk_id, set()).add(completion)

        def done(_result):
            with self._inflight_lock:
                for chunk_id in chunks:
                    active = self._inflight_by_chunk.get(chunk_id)
                    if active is None:
                        continue
                    active.discard(completion)
                    if not active:
                        self._inflight_by_chunk.pop(chunk_id, None)

        completion.add_done_callback(done)

    def _wait_chunk_io(self, chunk_ids) -> None:
        with self._inflight_lock:
            completions = {
                completion
                for chunk_id in chunk_ids
                for completion in self._inflight_by_chunk.get(chunk_id, ())
            }
        for completion in completions:
            completion.wait_result()

    def _mem_for(self, buffer_id: int) -> int:
        addr, size = self._buffers[buffer_id]
        return self._mem_cache[(addr, size)]

    def _on_put_settled(self, ok: bool, io_keys) -> None:
        """put 批 settle：数据确认落盘后才建层标记并更新在场集（崩溃安全）。"""
        if ok:
            self._layout.commit_layers(io_keys)
            self._live.update(io_keys)
            return
        chunk_ids = tuple(dict.fromkeys(
            decode_io_key(io_key)[0] for io_key in io_keys
        ))
        self._close_cached_targets([
            self._layout.target_uri(chunk_id) for chunk_id in chunk_ids
        ])
        self._layout.abort_uncommitted(chunk_ids)

    def _submit_retry(self, requests, direction: str):
        """提交整批；partial-commit 的被拒请求窗口重发。

        连续两轮零接受（runtime 窗口彻底耗尽）→ RuntimeError。
        """
        handles = []
        pending = list(enumerate(requests))
        all_rejected_rounds = 0
        rejection_diagnostics = []
        try:
            while pending:
                result = self._runtime.submit(
                    [request for _, request in pending],
                    accel_id=self._accel_id,
                    stream=self._stream_for(direction)[0],
                    execution=self._execution,
                )
                rejected = list(result.rejected or [])
                # Runtime partial-commit deliberately returns a non-OK status
                # together with a valid handle for the accepted prefix/subset.
                # That handle must be drained exactly once; only rejected indices
                # are retried.  Treating status_ok as all-or-nothing loses issued IO.
                if result.io_handle is not None:
                    accepted_indices = tuple(
                        pending[i][0]
                        for i, accepted in enumerate(result.initial_states or [])
                        if accepted
                    )
                    handles.append(_SubmittedHandle(result.io_handle,
                                                    accepted_indices))
                elif len(rejected) != len(pending):
                    raise RuntimeError(
                        "tutti submit 返回部分受理状态但缺少 IO handle："
                        f"{result.status_msg}"
                    )
                if not rejected:
                    if result.io_handle is None or not result.status_ok:
                        raise RuntimeError(f"tutti submit 失败：{result.status_msg}")
                    break
                if len(rejected) == len(pending):
                    all_rejected_rounds += 1
                    rejection_diagnostics.append(
                        f"round={all_rejected_rounds} pending={len(pending)} "
                        f"status={result.status_msg}"
                    )
                    if all_rejected_rounds >= 2:
                        raise RuntimeError(
                            "partial-commit 连续两轮零接受："
                            + "; ".join(rejection_diagnostics[-2:])
                        )
                else:
                    all_rejected_rounds = 0
                _LOG.warning(
                    "TUTTI_PARTIAL_REJECT direction=%s rejected=%s "
                    "retry_round=%d status=%s",
                    direction, rejected, all_rejected_rounds,
                    result.status_msg,
                )
                pending = [pending[i] for i in rejected]
        except Exception:
            if handles:
                self._deferred_completions.append(
                    _TuttiCompletion(
                        self._runtime, handles, lambda _ok: None,
                        auto_watch=False,
                    )
                )
            raise
        return handles


# ---------- 真机 runtime 构造（TUTTI_NVME_PRESET） ----------


def _normalize_preset(preset) -> dict:
    """递归归一 preset：字符串值恰为纯十进制整数时转 int。"""
    if isinstance(preset, dict):
        return {k: _normalize_preset(v) for k, v in preset.items()}
    if isinstance(preset, list):
        return [_normalize_preset(v) for v in preset]
    if isinstance(preset, str) and preset.strip().isdigit():
        return int(preset)
    return preset


def _preset_mounts(preset):
    """Derive striped layout mounts from a striped preset when available."""
    if not isinstance(preset, dict):
        return None
    devices = preset.get("devices")
    if not isinstance(devices, (list, tuple)):
        return None
    mounts = []
    for device in devices:
        if not isinstance(device, dict) or not device.get("mount_path"):
            return None
        mounts.append(device["mount_path"])
    return mounts or None


def _build_runtime(preset: dict):
    """按 preset dict 构造真机 runtime（daemon_config 推导与归一同环境变量路径）。"""
    import yaml

    if not isinstance(preset, dict):
        raise RuntimeError("preset 必须是映射")
    if "daemon_config" in preset:
        preset = _derive_device_fields(preset, yaml)

    try:
        import tutti_runtime  # bindings 构建产物（需在 sys.path/PYTHONPATH）
    except ImportError as exc:
        raise RuntimeError(
            "tutti_runtime 绑定不可用：先构建 integration/vllm-connector/"
            "bindings/python 并将其加入 PYTHONPATH"
        ) from exc

    preset = dict(preset)
    preset_type = preset.pop("type", "local")
    preset.pop("daemon_config", None)  # 推导元键不进 runtime preset
    preset.pop("device_id", None)
    if preset_type == "striped":
        return tutti_runtime.make_striped_nvme_runtime(preset)
    if preset_type == "local":
        return tutti_runtime.make_local_nvme_runtime(preset)
    raise RuntimeError(f"未知 preset type：{preset_type}")


def _build_runtime_from_env():
    """按 TUTTI_NVME_PRESET 构造真机 runtime（本包私有推导）。"""
    import yaml

    raw = os.environ.get("TUTTI_NVME_PRESET", "").strip()
    if not raw:
        raise RuntimeError(
            "runtime=None 需要 TUTTI_NVME_PRESET（yaml/json 内联或文件路径）"
        )
    text = Path(raw).read_text() if os.path.isfile(raw) else raw
    preset = yaml.safe_load(text)
    if not isinstance(preset, dict):
        raise RuntimeError("TUTTI_NVME_PRESET 解析结果必须是映射")
    return _build_runtime(_normalize_preset(preset))


def _derive_device_fields(preset: dict, yaml) -> dict:
    """daemon_config + device_id → 设备字段（硬件信息单一来源）。

    daemon 配置（yaml）的 nvmes 列表按 device_id 给出权威事实：
    pci_addr、backing_mount_path、namespace_id；preset 可显式覆盖。
    SNVMe 字符设备由 C++ preset 组装器按 BDF 查询 sysfs，Python 不拼接
    `/dev/ssnvme<N>`；backing block device 仍可按 namespace 约定回退。
    """
    daemon_path = preset.get("daemon_config")
    device_id = preset.get("device_id")
    if not daemon_path or device_id is None:
        raise RuntimeError("preset 携带 daemon_config 时必须同时给出 device_id")
    daemon = yaml.safe_load(Path(daemon_path).read_text())
    entry = next(
        (n for n in daemon.get("nvmes", []) if n.get("device_id") == device_id), None
    )
    if entry is None:
        raise RuntimeError(f"daemon 配置无 device_id={device_id} 的 NVMe 条目")
    device = dict(preset.get("device") or {})
    namespace_id = device.get("namespace_id", entry.get("namespace_id", 1))
    device.setdefault("pci_bdf", entry["pci_addr"])
    device.setdefault("mount_path", entry["backing_mount_path"])
    device.setdefault("namespace_id", namespace_id)
    device.setdefault(
        "backing_device", f"/dev/snvme{device_id}n{device.get('namespace_id', 1)}"
    )
    derived = dict(preset)
    derived["device"] = device
    return derived

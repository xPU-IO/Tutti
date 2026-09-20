"""KVEngine：计划态转发语义索引，执行态编排 staging 环窗、store 与传输路径。"""

from __future__ import annotations

from contextlib import nullcontext
import logging
import os
import threading
import time
from typing import Sequence

from tutti.engine.staging import RingWindow
from tutti.engine.transfer import (
    DirectTransfer,
    DirectTransferUnavailable,
    StagedTransfer,
    select_transfer,
)
from tutti.engine.nvtx import range as nvtx_range
from tutti.common.utils import (
    flatten_block_ids as _flatten_block_ids,
    group_scan,
    is_int,
    positive_int,
)
# 完成句柄与门禁异常在 engine.completion 中实现；此处再导出以保持
# 既有导入路径（adapter.worker、契约测试）不变。
from tutti.engine.completion import (
    LoadGateError,
    _AggregateCompletion,
    _PostCompletion,
)
from tutti.index.chunk_index import (
    ChunkIndex,
    IO_KEY_BYTES,
    StorePlan,
    derive_io_key,
)
_LOG = logging.getLogger(__name__)
_FEEDER_DIAG = os.environ.get("TUTTI_FEEDER_DIAGNOSTICS") == "1"
# feeder 滞后时 compute 回调等待其登记的时长上限。等待是正常路径（feeder 只是
# 落后于 compute），超时仅用于把"feeder 卡死"这类真故障转成显式错误。
_DIRECT_FEEDER_WAIT_SECONDS = 120.0
# 一次等待超过该阈值（毫秒）就记一行，用于观测 compute 与 feeder 的竞速余量。
_DIRECT_FEEDER_WAIT_LOG_MS = 1.0


class _ReadPlan:
    """Python-owned eager layer plan.

    Construction submits every layer immediately.  The engine keeps staging
    reuse asynchronous by inserting read-stream waits on the previous layer's
    already-recorded scatter fence; no host observer or callback submission is
    involved.
    """

    def __init__(self, engine, keys, block_tables, physical_layers, depth,
                 on_failure=None):
        self.engine = engine
        self.keys = list(keys)
        self.block_tables = list(block_tables)
        self.physical_layers = tuple(physical_layers)
        self.depth = max(1, int(depth))
        self.on_failure = on_failure
        self._handles = {}
        self._fence_slots = {}
        self._failed = None
        self._submit_all()

    @property
    def layer_count(self):
        return len(self.physical_layers)

    @property
    def failed(self):
        return self._failed

    def fence_event(self, layer):
        return self._fence_slots.get(layer)

    def _submit_all(self):
        for layer in range(len(self.physical_layers)):
            physical = self.physical_layers[layer]
            wait_for_reuse = layer >= self.depth
            fence_event = self._new_fence_event()
            self._fence_slots[layer] = fence_event
            reuse_event = (
                self._fence_slots.get(layer - self.depth)
                if wait_for_reuse else None
            )
            try:
                self._handles[layer] = self.engine.load_layer(
                    self.keys, physical, self.block_tables,
                    # Every eager-plan wave must avoid RingWindow's host
                    # reuse wait, including a second wave within one layer.
                    # The optional predecessor fence is inserted on the read
                    # stream only when this layer wraps the staging bank.
                    async_reuse=True,
                    fence_event=fence_event,
                    reuse_event=reuse_event,
                    bridge_compute=False,
                )
                add_terminal = getattr(
                    self._handles[layer], "add_terminal_callback", None
                )
                if not callable(add_terminal):
                    add_terminal = getattr(
                        self._handles[layer], "add_done_callback", None
                    )
                if callable(add_terminal):
                    add_terminal(self._on_completion)
            except Exception as exc:
                self._failed = exc
                if callable(self.on_failure):
                    self.on_failure(exc)
                raise

    def _on_completion(self, result) -> None:
        if getattr(result, "ok", True) or self._failed is not None:
            return
        self._failed = LoadGateError(
            "direct read failed; recompute the complete step",
            whole_operation=True,
            invalid_block_ids=_flatten_block_ids(self.block_tables),
        )
        if callable(self.on_failure):
            self.on_failure(self._failed)

    def _new_fence_event(self):
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.Event(enable_timing=False)
        except Exception:
            pass
        store = getattr(self.engine, "_store", None)
        factory = getattr(store, "new_event", None)
        if callable(factory):
            return factory()
        # Host/fake stores used by contract tests can provide their own event
        # object through record_* methods; None keeps that compatibility path.
        return None

    def wait_layer(self, layer):
        if self._failed is not None:
            raise self._failed
        return self._fence_slots.get(layer)

    def abort(self):
        for handle in self._handles.values():
            try:
                abort = getattr(handle, "abort", None)
                if callable(abort):
                    abort()
                else:
                    handle.wait()
            except Exception:
                pass


def _bind_thread_cuda_device(store) -> None:
    """把当前线程的 CUDA 当前设备绑定到本 rank 的 GPU。

    新线程的当前设备是进程默认值 0。不绑定的话，运行时 `DeviceGuard` 每次
    submit 都要先切到本 rank、restore 再切回 0（两次 ``cudaSetDevice``），而
    8 rank 并发时其中一次实测约 950ms——该调用不进入 nsys 的 CUDA API 表，只在
    runtime 内埋点可见——表现为 `tutti.direct.runtime_submit|op=read` 上 1 秒的
    气泡（提交自身只有 0.2ms）。绑定后 guard 的 enter/restore 都是 no-op。

    失败不致命：guard 仍会切设备，只是回到"每次两次 cudaSetDevice"的老行为。
    """
    accel = getattr(store, "_accel_id", -1)
    if not isinstance(accel, int) or accel < 0:
        return
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.current_device() != accel:
            torch.cuda.set_device(accel)
    except Exception:  # pragma: no cover - 环境异常时保持老行为
        _LOG.debug("DIRECT_THREAD_DEVICE_BIND_FAILED accel=%d", accel,
                   exc_info=True)


class _DirectAllLayerReadPlan:
    """Direct-only all-layer read plan.

    Layer zero is submitted synchronously so ``start_load_kv`` can return
    without waiting for the complete plan. Remaining layers are submitted by
    an independent host feeder that starts as soon as the plan is constructed.
    The callback never invokes Runtime I/O itself.
    """

    def __init__(self, engine, keys, block_tables, physical_layers,
                 on_failure=None):
        self.engine = engine
        self.keys = list(keys)
        self.block_tables = list(block_tables)
        self.physical_layers = tuple(physical_layers)
        self.handles = {}
        self.read_ready_events = {}
        self.next_read_to_submit = 0
        self.waited_callbacks = set()
        self.advanced_callbacks = set()
        self.terminal_failure = None
        self.on_failure = on_failure
        self._layer_submit_ms = {}
        self._layer_submit_completed_ns = {}
        self._submit_all_total_ms = 0.0
        self._first_read_submit_completed_ns = None
        self._state_lock = threading.RLock()
        # fence 可用性的专用条件变量：**必须与 _state_lock 分离**——feeder 在
        # 提交单层期间持有 _state_lock（load_layer 可达数百毫秒），若等待者共用
        # 该锁，等待会退化成互斥排队，超时与失败检查都会失效。
        self._ready_cond = threading.Condition()
        self._feeder_stop = threading.Event()
        self._feeder_done = threading.Event()
        self._feeder_thread = None
        if not self.physical_layers:
            raise ValueError("direct all-layer read plan requires at least one layer")
        self._submit_layer(0)
        if self.layer_count > 1:
            # feeder 立即起跑：余下层不等首个 compute 回调。提前提交让读尽早占满
            # 盘侧带宽，并把"某层尚未提交"的窗口压到最小；并发由 in-flight 配额
            # 约束，不需要靠 compute 节奏限流。
            self._feeder_thread = threading.Thread(
                target=self._run_feeder,
                name="tutti-direct-read-feeder",
                daemon=True,
            )
            self._feeder_thread.start()

    @property
    def layer_count(self):
        return len(self.physical_layers)

    @property
    def failed(self):
        return self.terminal_failure

    def fence_event(self, callback):
        return self.read_ready_events.get(callback)

    def wait_layer(self, callback, physical=None):
        self._validate_callback(callback, physical)
        event = self._await_ready_event(callback)
        if callback in self.waited_callbacks:
            return event
        expected = len(self.waited_callbacks)
        if callback != expected:
            error = RuntimeError(
                f"out-of-order direct wait callback: expected {expected}, "
                f"got {callback}"
            )
            self._record_failure(error)
            raise error
        self.waited_callbacks.add(callback)
        return event

    def _await_ready_event(self, callback: int):
        """等到 feeder 为 callback 层登记 fence event，再返回该 event。

        compute 回调可能早于 feeder 提交该层（feeder 是主机线程、逐层提交，
        单层耗时与 compute 同量级），此时**必须等**而不是判定失败：feeder 的
        推进只受 in-flight 配额约束，配额随 IO 完成释放、不依赖 compute 进度，
        所以等待必然收敛。等待期间若发现计划已失败或 feeder 已停止，立即抛出
        对应错误，避免把真故障变成挂起。
        """
        deadline = time.monotonic() + _DIRECT_FEEDER_WAIT_SECONDS
        waited_started_ns = None
        error = None
        event = None
        with self._ready_cond:
            while True:
                event = self.read_ready_events.get(callback)
                if event is not None:
                    break
                failure = self.terminal_failure
                if failure is not None:
                    error = failure
                    break
                if self._feeder_stop.is_set():
                    error = RuntimeError(
                        f"direct read feeder stopped before layer {callback}"
                    )
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = RuntimeError(
                        f"direct read feeder did not submit layer {callback} "
                        f"within {_DIRECT_FEEDER_WAIT_SECONDS:.0f}s"
                    )
                    break
                if waited_started_ns is None:
                    waited_started_ns = time.perf_counter_ns()
                self._ready_cond.wait(timeout=min(remaining, 0.5))
        if error is not None:
            # _record_failure 会通知同一条件变量，必须在释放后调用。
            if error is not self.terminal_failure:
                self._record_failure(error)
            raise error
        if waited_started_ns is not None:
            waited_ms = (time.perf_counter_ns() - waited_started_ns) / 1_000_000
            if waited_ms >= _DIRECT_FEEDER_WAIT_LOG_MS:
                _LOG.info(
                    "DIRECT_READ_LAYER_WAIT layer=%d waited_ms=%.3f",
                    callback, waited_ms,
                )
        return event

    def after_layer(self, callback, physical=None) -> None:
        """Validate a legacy after-layer notification without submitting I/O."""
        self._validate_callback(callback, physical)
        if callback in self.advanced_callbacks:
            return
        expected = len(self.advanced_callbacks)
        if callback != expected:
            error = RuntimeError(
                f"out-of-order direct after-layer callback: expected "
                f"{expected}, got {callback}"
            )
            self._record_failure(error)
            raise error
        self.advanced_callbacks.add(callback)

    def require_complete(self) -> None:
        self.join_feeder()
        expected = set(range(self.layer_count))
        missing_waits = tuple(sorted(expected - self.waited_callbacks))
        if self.next_read_to_submit != self.layer_count or missing_waits:
            error = RuntimeError(
                "incomplete direct all-layer callbacks: "
                f"next_read={self.next_read_to_submit}, "
                f"missing_waits={missing_waits}"
            )
            self._record_failure(error)
            raise error

    def _submit_layer(self, callback: int) -> None:
        physical = self.physical_layers[callback]
        with self._state_lock:
            if callback in self.handles or self._feeder_stop.is_set():
                return
        fence_event = self._new_fence_event()
        layer_started_ns = time.perf_counter_ns()
        try:
            with self._state_lock:
                if callback in self.handles or self._feeder_stop.is_set():
                    return
                submit_lock = getattr(
                    self.engine, "_direct_submit_lock", self._state_lock
                )
                with submit_lock:
                    handle = self.engine.load_layer(
                        self.keys,
                        physical,
                        self.block_tables,
                        fence_event=fence_event,
                        bridge_compute=False,
                    )
        except Exception as exc:
            self._record_failure(exc)
            self._feeder_stop.set()
            if callback == 0:
                self._abort_submitted()
                raise
            return
        if fence_event is None:
            fence_event = getattr(handle, "fence_event", None)
        if fence_event is None:
            error = RuntimeError(
                f"direct read layer {callback} did not record a ready event"
            )
            self._record_failure(error)
            self._feeder_stop.set()
            if callback == 0:
                self._abort_submitted()
                raise error
            return
        completed_ns = time.perf_counter_ns()
        elapsed_ms = (completed_ns - layer_started_ns) / 1_000_000
        with self._state_lock:
            self.handles[callback] = handle
            self.next_read_to_submit = max(
                self.next_read_to_submit, callback + 1
            )
            self._layer_submit_ms[callback] = elapsed_ms
            self._layer_submit_completed_ns[callback] = completed_ns
            if callback == 0:
                self._first_read_submit_completed_ns = completed_ns
        with self._ready_cond:
            self.read_ready_events[callback] = fence_event
            self._ready_cond.notify_all()
        # 正常路径的逐层时间线诊断：debug（每层一条，warning 会淹没真实告警）。
        _LOG.debug(
            "DIRECT_READ_LAYER_SUBMIT layer=%d physical=%d "
            "elapsed_ms=%.3f completed_ns=%d",
            callback, physical, elapsed_ms, completed_ns,
        )
        add_terminal = getattr(handle, "add_terminal_callback", None)
        if not callable(add_terminal):
            add_terminal = getattr(handle, "add_done_callback", None)
        if callable(add_terminal):
            add_terminal(self._on_completion)

    def _run_feeder(self) -> None:
        _bind_thread_cuda_device(getattr(self.engine, "_store", None))
        try:
            started_ns = time.perf_counter_ns()
            for callback in range(1, self.layer_count):
                if self._feeder_stop.is_set() or self.terminal_failure is not None:
                    break
                self._submit_layer(callback)
        except Exception as exc:
            self._record_failure(exc)
            self._feeder_stop.set()
        finally:
            started_ns = locals().get("started_ns", time.perf_counter_ns())
            self._submit_all_total_ms = (
                time.perf_counter_ns() - started_ns
            ) / 1_000_000
            max_layer_ms = max(self._layer_submit_ms.values(), default=0.0)
            _LOG.warning(
                "DIRECT_READ_PLAN_READY layers=%d total_ms=%.3f "
                "max_layer_ms=%.3f first_read_submit_completed_ns=%s",
                self.layer_count,
                self._submit_all_total_ms,
                max_layer_ms,
                self._first_read_submit_completed_ns,
            )
            self._publish_ready()
            self._feeder_done.set()

    def _publish_ready(self) -> None:
        """唤醒等待 fence 的 compute 回调（登记新层 / 失败 / feeder 结束）。"""
        with self._ready_cond:
            self._ready_cond.notify_all()

    def join_feeder(self) -> None:
        thread = self._feeder_thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join()

    def _on_completion(self, result) -> None:
        if getattr(result, "ok", True) or self.terminal_failure is not None:
            return
        self._record_failure(LoadGateError(
            "direct read failed; recompute the complete step",
            whole_operation=True,
            invalid_block_ids=_flatten_block_ids(self.block_tables),
        ))

    def _record_failure(self, error) -> None:
        if self.terminal_failure is not None:
            return
        self.terminal_failure = error
        # 唤醒可能在 _await_ready_event 里等待 fence 的 compute 回调，让它们
        # 立刻看到失败而不是睡到超时。
        self._publish_ready()
        if callable(self.on_failure):
            self.on_failure(error)

    def _validate_callback(self, callback, physical=None) -> None:
        if (not isinstance(callback, int) or isinstance(callback, bool)
                or not 0 <= callback < self.layer_count):
            error = RuntimeError(
                f"direct callback ordinal out of range: {callback}"
            )
            self._record_failure(error)
            raise error
        expected_physical = self.physical_layers[callback]
        if physical is not None and physical != expected_physical:
            error = RuntimeError(
                f"direct callback {callback} maps to physical "
                f"{expected_physical}, got {physical}"
            )
            self._record_failure(error)
            raise error

    def _new_fence_event(self):
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.Event(enable_timing=False)
        except Exception:
            pass
        store = getattr(self.engine, "_store", None)
        factory = getattr(store, "new_event", None)
        return factory() if callable(factory) else None

    def abort(self):
        self._feeder_stop.set()
        self._publish_ready()
        self.join_feeder()
        self._abort_submitted()

    def _abort_submitted(self):
        for handle in self.handles.values():
            try:
                abort = getattr(handle, "abort", None)
                if callable(abort):
                    abort()
                else:
                    handle.wait()
            except Exception:
                pass


class _EngineStepIO:
    """Compatibility layer callback helper for contract tests.

    Production WorkerImpl uses ordinary per-layer handles directly.  This
    helper remains isolated in the engine test surface and does not call any
    lower-layer feeder API.
    """

    def __init__(self, inner, engine, keys, block_tables, slots_by_callback,
                 direction, physical_layers):
        self._inner = inner
        self._engine = engine
        self._keys = list(keys)
        self._block_tables = block_tables
        self._slots = slots_by_callback
        self._direction = direction
        self._physical_layers = tuple(physical_layers)
        self._seen: set[int] = set()
        self._next_callback = 0
        self._released_after_failure: set[int] = set()
        self._drained = False
        self._drain_result_value = None

    def wait_layer(self, callback: int, physical: int) -> None:
        if not self._enter_callback(callback, physical):
            return
        if _FEEDER_DIAG:
            _LOG.warning("FEEDER_DIAG engine gate_wait_begin t_ns=%d direction=read callback=%d physical=%d",
                         time.monotonic_ns(), callback, physical)
        wait_detail = getattr(self._inner, "wait_layer_detail", None)
        if callable(wait_detail):
            state, failed_flat_index = wait_detail(callback)
            layer_ready = state == "READY"
        else:
            layer_ready = self._inner.wait_layer(callback)
            failed_flat_index = None
        if _FEEDER_DIAG:
            _LOG.warning(
                "FEEDER_DIAG layer_event t_ns=%d phase=read_ready "
                "callback=%d physical=%d ready=%s",
                time.monotonic_ns(), callback, physical, layer_ready,
            )
        if not layer_ready:
            # The callback must return quickly so this TP rank can execute the
            # old forward's remaining model collectives. Structured CQ detail
            # is harvested after the all-rank failure collective at finalize.
            chunk_count = len(self._keys)
            failed_chunks = () if failed_flat_index is None else (
                int(failed_flat_index) % chunk_count,
            )
            selected = self._block_tables if not failed_chunks else [
                self._block_tables[index] for index in failed_chunks
            ]
            invalid_blocks = _flatten_block_ids(selected)
            _LOG.error(
                "REAL_LOAD_FAILURE_GATE callback=%d physical=%d "
                "scope=%s deferred_drain=true failed_flat_index=%s "
                "failed_chunks=%s invalid_block_ids=%s",
                callback, physical,
                "REQUEST_INDICES" if failed_chunks else "WHOLE_OPERATION",
                failed_flat_index, failed_chunks, invalid_blocks,
            )
            raise LoadGateError(
                f"step read feeder callback {callback} physical {physical} failed",
                failed_batch_indices=failed_chunks,
                whole_operation=not bool(failed_chunks),
                invalid_block_ids=invalid_blocks,
            )
        slots = self._slots[callback]
        if _FEEDER_DIAG:
            _LOG.warning(
                "FEEDER_DIAG layer_event t_ns=%d phase=scatter_enqueue "
                "callback=%d physical=%d stream=%d",
                time.monotonic_ns(), callback, physical,
                self._read_copy_stream_handle(),
            )
        with nvtx_range(
            f"tutti.load.scatter|mode=feeder|layer={physical}"
            f"|chunks={len(self._keys)}"
        ):
            event = self._scatter_on_read_copy(
                self._keys, physical, self._block_tables, slots
            )
        if _FEEDER_DIAG:
            _LOG.warning(
                "FEEDER_DIAG layer_event t_ns=%d phase=scatter_done_record "
                "callback=%d physical=%d event=%s",
                time.monotonic_ns(), callback, physical, event is not None,
            )
        self._bridge_read_copy_event(event)
        if _FEEDER_DIAG:
            _LOG.warning(
                "FEEDER_DIAG layer_event t_ns=%d phase=compute_wait_event "
                "callback=%d physical=%d",
                time.monotonic_ns(), callback, physical,
            )
        if callback + self._inner.staging_depth < self._inner.layer_count:
            # The release kernel is enqueued after scatter/event-record on the
            # same read-copy stream, so K=2 staging cannot be reused early.
            self._inner.signal_layer(
                callback, self._read_copy_stream_handle()
            )
            if _FEEDER_DIAG:
                _LOG.warning(
                    "FEEDER_DIAG layer_event t_ns=%d phase=staging_release "
                    "callback=%d physical=%d stream=%d",
                    time.monotonic_ns(), callback, physical,
                    self._read_copy_stream_handle(),
                )
            if _FEEDER_DIAG:
                _LOG.warning("FEEDER_DIAG engine gate_release_publish t_ns=%d direction=read callback=%d physical=%d",
                             time.monotonic_ns(), callback, physical)
        # Windowed reads use this notification only to wake the detached
        # feeder.  The feeder enqueues the read-stream wait before it reuses
        # the bank; the attention callback never submits a window and never
        # performs a host CUDA wait.  Mark the final window as consumed too,
        # even though it has no successor bank to protect.
        release = getattr(self._inner, "release_layer", None)
        if callable(release):
            release(callback, event)
        if _FEEDER_DIAG:
            _LOG.warning("FEEDER_DIAG engine gate_wait_end t_ns=%d direction=read callback=%d physical=%d",
                         time.monotonic_ns(), callback, physical)

    def publish_layer(self, callback: int, physical: int) -> None:
        if not self._enter_callback(callback, physical):
            return
        # Before overwriting bank layer%K, wait until the feeder consumed L-K.
        if _FEEDER_DIAG:
            _LOG.warning("FEEDER_DIAG engine gate_release_wait_begin t_ns=%d direction=write callback=%d physical=%d",
                         time.monotonic_ns(), callback, physical)
        self._inner.wait_layer(callback)
        if _FEEDER_DIAG:
            _LOG.warning(
                "FEEDER_DIAG layer_event t_ns=%d phase=write_gather "
                "callback=%d physical=%d stream=%d",
                time.monotonic_ns(), callback, physical, self._current_stream(),
            )
        slots = self._slots[callback]
        event = None
        transfer = self._engine._transfer
        if transfer is not None:
            event = transfer.gather(
                self._keys, physical, self._block_tables, slots
            )
        if event is not None:
            # The ready signal is enqueued on the same compute stream after
            # gather; no host synchronization and no feeder-stream deadlock.
            pass
        self._inner.signal_layer(callback, self._current_stream())
        if _FEEDER_DIAG:
            _LOG.warning(
                "FEEDER_DIAG layer_event t_ns=%d phase=write_ready "
                "callback=%d physical=%d stream=%d",
                time.monotonic_ns(), callback, physical, self._current_stream(),
            )
        if _FEEDER_DIAG:
            _LOG.warning("FEEDER_DIAG engine gate_ready_publish t_ns=%d direction=write callback=%d physical=%d",
                         time.monotonic_ns(), callback, physical)

    def wait(self):
        self._require_complete()
        return self._inner.wait()

    def wait_result(self):
        if self._drained:
            return self._drain_result_value
        self._require_complete()
        return self._inner.wait_result()

    def drain(self, timeout=None):
        return self._drain_result(timeout)

    def _drain_result(self, timeout=None):
        if self._drained:
            return self._drain_result_value
        self._drained = True
        if self._direction == "read":
            # Abort/shutdown can happen before attention consumes every
            # callback. Release all outstanding future waits so the
            # pre-enqueued read plan reaches its terminal event.
            stream = self._read_copy_stream_handle()
            for callback in range(len(self._physical_layers)):
                if (callback in self._seen or
                        callback in self._released_after_failure):
                    continue
                self._inner.signal_layer(callback, stream)
                self._released_after_failure.add(callback)
        if timeout is None:
            self._drain_result_value = self._inner.drain(
                self._current_stream()
            )
        else:
            self._drain_result_value = self._inner.drain(
                self._current_stream(), timeout=timeout
            )
        return self._drain_result_value

    def abort(self, timeout=None):
        return self.drain(timeout)

    def release_after_failure(self, callback: int) -> None:
        """Unblock pre-enqueued read windows without submitting new I/O.

        A failed layer has no scatter fence.  Recording release events for the
        failed and later layers lets already-enqueued kernels observe the
        fail-closed ready flag and exit without issuing NVMe commands.
        """
        if self._direction != "read":
            return
        stream = self._read_copy_stream_handle()
        release = getattr(self._inner, "release_after_failure", None)
        if callable(release):
            # Stop the detached feeder before walking the remaining callback
            # ordinals; unsent windows must not be awaited or signalled.
            release(callback)
        for layer in range(callback, len(self._physical_layers)):
            if layer in self._released_after_failure:
                continue
            self._inner.signal_layer(layer, stream)
            self._released_after_failure.add(layer)

    @property
    def pending_callbacks(self) -> tuple[int, ...]:
        return tuple(i for i in range(len(self._physical_layers))
                     if i not in self._seen)

    def _enter_callback(self, callback: int, physical: int) -> bool:
        if callback < 0 or callback >= len(self._physical_layers):
            self.drain()
            raise RuntimeError(f"callback ordinal out of range: {callback}")
        expected_physical = self._physical_layers[callback]
        if physical != expected_physical:
            self.drain()
            raise RuntimeError(
                f"callback {callback} maps to physical {expected_physical}, "
                f"got {physical}"
            )
        if callback in self._seen:
            return False
        if callback != self._next_callback:
            self.drain()
            raise RuntimeError(
                f"out-of-order callback: expected {self._next_callback}, "
                f"got {callback}"
            )
        self._seen.add(callback)
        self._next_callback += 1
        return True

    def _require_complete(self) -> None:
        missing = self.pending_callbacks
        if missing:
            self.drain()
            raise RuntimeError(f"missing feeder callbacks: {missing}")

    def _scatter_on_read_copy(self, keys, physical, block_tables, slots):
        method = getattr(self._engine, "_scatter_on_read_copy", None)
        if callable(method):
            return method(keys, physical, block_tables, slots)
        hook = getattr(self._engine, "_scatter_hook", None)
        if not callable(hook):
            return None
        store = getattr(self._engine, "_store", None)
        context = getattr(store, "stream_context", None)
        with (context("read_copy") if callable(context) else nullcontext()):
            event = hook(keys, physical, block_tables, slots)
            record = getattr(store, "record_read_copy_event", None)
            return record(event) if callable(record) else event

    def _bridge_read_copy_event(self, event) -> None:
        method = getattr(self._engine, "_bridge_read_copy_event", None)
        if callable(method):
            if _FEEDER_DIAG:
                _LOG.warning(
                    "FEEDER_DIAG layer_event t_ns=%d phase=compute_wait_enqueue",
                    time.monotonic_ns(),
                )
            method(event, protect_read_io=False)
            return
        store = getattr(self._engine, "_store", None)
        wait_compute = getattr(store, "wait_compute_event", None)
        if event is not None and callable(wait_compute):
            if _FEEDER_DIAG:
                _LOG.warning(
                    "FEEDER_DIAG layer_event t_ns=%d phase=compute_wait_enqueue",
                    time.monotonic_ns(),
                )
            wait_compute(event)

    def _read_copy_stream_handle(self) -> int:
        method = getattr(self._engine, "_read_copy_stream_handle", None)
        if callable(method):
            return int(method())
        store = getattr(self._engine, "_store", None)
        handle = getattr(store, "read_copy_stream_handle", None)
        return int(handle()) if callable(handle) else self._current_stream()

    @staticmethod
    def _current_stream() -> int:
        try:
            import torch
            return int(torch.cuda.current_stream().cuda_stream)
        except Exception:
            return 0


class KVEngine:
    """编排核心：构造注入 store，语义索引自建并做冷启动恢复。

    config 契约（键全部与硬件无关）：
    - chunk_tokens：每 chunk 的 token 数（正整数）。
    - chunk_kv_bytes：单 chunk 的 KV 字节数（正整数）。
    - max_chunks_per_wave：单波最大 chunk 数（正整数）。
    - num_layers：可选层数预告（正整数）。查询侧（不做 bind）注入后，
      驱逐展开与冷启动完整性判定即可在 bind 之前正确工作；bind 时
      以实测定案并与预告校验一致（不一致 → ValueError）。缺省 None。
    - key_namespace：可选 key 命名空间（str 或 bytes，部署层按模型/
      dtype/TP/几何组装的不透明串，含格式版本号）。注入后 chunk key
      链以其为前缀派生——不同命名空间（不同模型/几何复用同一池）
      的 key 天然隔离；支持持久层 manifest 的 store（可选实现）以之
      校验池归属。缺省无命名空间。
    - direct_transfer：可选 bool。缺省（True）尝试 Python byte-range direct
      backend。staged 暂存路径已退役，故显式 false 不再表示"改用 staged"，
      而是配置错误（select_transfer 抛出）。
    - direct_transfer_strict：**已冗余，不再被读取**。直连准入失败一律抛出，
      恒等于原先的 strict 行为；保留该键只为兼容既有配置文件，设它无任何效果。
    - gather_fn / scatter_fn：可选搬运钩子（缺省 None，搬运为 no-op），
      语义见传输路径。

    构造期：打开 store、建立语义索引、对 store 的存活枚举按 chunk key
    分组；层数定案（bind）后才把层完整的 chunk 灌入索引（见 io_key
    线格式约定）。
    """

    def __init__(self, config: dict, store):
        """config 见类契约；store 为 KVStore 实现。参数非法 → ValueError。"""
        self._chunk_tokens = positive_int(config, "chunk_tokens")
        self._chunk_kv_bytes = positive_int(config, "chunk_kv_bytes")
        self._max_chunks_per_wave = positive_int(config, "max_chunks_per_wave")
        for name in ("gather_fn", "scatter_fn"):
            fn = config.get(name)
            if fn is not None and not callable(fn):
                raise ValueError(f"config[{name!r}] 须为可调用或 None，got {fn!r}")
        for name in ("direct_transfer", "direct_transfer_strict"):
            value = config.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f"config[{name!r}] 须为 bool，got {value!r}")
        self._config = dict(config)
        self._store = store
        self._closed = False
        layers_hint = config.get("num_layers")
        if layers_hint is not None and (not is_int(layers_hint) or layers_hint <= 0):
            raise ValueError(f"config['num_layers'] 须为正整数或 None，got {layers_hint!r}")
        raw_ns = config.get("key_namespace")
        if raw_ns is None:
            namespace = b""
        elif isinstance(raw_ns, str):
            namespace = raw_ns.encode("utf-8")
        elif isinstance(raw_ns, (bytes, bytearray)):
            namespace = bytes(raw_ns)
        else:
            raise ValueError(
                f"config['key_namespace'] 须为 str/bytes/None，got {raw_ns!r}"
            )
        # 命名空间注入持久层（可选实现）：manifest 校验池归属。
        # 注入须在 open 之前（持久层契约），open 时按 manifest 校验归属。
        setter = getattr(store, "set_key_namespace", None)
        if setter is not None and namespace:
            setter(namespace)
        store.open()
        self._max_in_flight_operations = int(
            getattr(store, "max_in_flight_operations", 0) or 0
        )
        self._index = ChunkIndex(store.capacity_chunks, self._chunk_tokens,
                                 namespace=namespace)
        # 冷启动分组：层数定案前暂存；层集合不完整的 chunk 视为缺失。
        self._scan_groups = group_scan(store)
        self._restored = False
        # 上次对账判完整的组（完整性翻转修正的基准）与因 pin 保护
        # 未遂的移除项（下次对账重试）。
        self._synced_full: set[bytes] = set()
        self._pending_forget: set[bytes] = set()
        # 执行态（bind 后可用；num_layers 预告可先行）
        self._num_layers: int | None = layers_hint
        self._segment_bytes: int | None = None
        self._window: RingWindow | None = None  # read-window compatibility alias
        self._read_window: RingWindow | None = None
        self._write_window: RingWindow | None = None
        self._transfer: DirectTransfer | StagedTransfer | None = None
        self._scatter_hook = None
        self._staging_buffer_id: int | None = None
        self._read_staging_buffer_id: int | None = None
        self._write_staging_buffer_id: int | None = None
        self._inflight: list = []
        self._planned_store_keys: set[bytes] = set()
        self._write_reuse_event = None
        self._direct_submit_lock = threading.RLock()
        self._active_read_plan = None

    @property
    def max_in_flight_operations(self) -> int:
        return self._max_in_flight_operations

    # ---- 计划态 ----

    def lookup_prefix(self, token_ids: Sequence[int]) -> int:
        """查询前缀命中的 token 数（转发语义索引）。"""
        self._require_open()
        return self._index.lookup_prefix(token_ids)

    def hash_keys(
        self,
        token_ids: Sequence[int],
        start: int = 0,
        parent: bytes | None = None,
    ) -> tuple[list[bytes], bytes]:
        """把 token 序列折叠为 chunk key 链（转发语义索引）。

        parent 未给时自命名空间起（见 ChunkIndex.hash_keys）。
        """
        self._require_open()
        return self._index.hash_keys(token_ids, start, parent)

    def plan_store(self, keys) -> StorePlan | None:
        """受理写入计划并预留容量（转发语义索引）。

        层数已定案时，驱逐的 chunk 展开为其全部层的 io_key 交给
        store 执行删除；层数未定案（bind 之前）时不执行数据面删除，
        驱逐善后由层数定案后的权威进程结算。
        """
        self._require_open()
        keys = list(keys)
        plan = self._index.plan_store(keys)
        if plan is None:
            return None
        self._planned_store_keys.update(plan.new_keys)
        if plan.evicted_keys and self._num_layers is not None:
            self._store.drop(_expand_io_keys(plan.evicted_keys, self._num_layers))
        # 只预置**本次受理的** chunk（plan.new_keys，已去重且排除驻留）：
        # 已驻留的 chunk 不会写，不该占用对象池槽位；且预置集合必须与
        # 后续 store_layer/_prepare_write_batch 实际提交的集合一致，否则
        # backend 会以"同一步内写目标变更"拒绝（fail-closed 守卫）。
        if plan.new_keys and self.direct:
            backend = getattr(self._transfer, "_backend", None)
            prepare = getattr(backend, "prepare_write_targets", None)
            if callable(prepare):
                try:
                    prepare(plan.new_keys)
                except Exception:
                    abort_chunks = getattr(self._store, "abort_chunks", None)
                    if callable(abort_chunks):
                        try:
                            abort_chunks(plan.new_keys)
                        except Exception:
                            pass
                    self._index.confirm_store(plan.new_keys, ok=False)
                    self._planned_store_keys.difference_update(plan.new_keys)
                    raise
        return plan

    def apply_evictions(self, keys) -> int:
        """执行调度侧下发的驱逐决策（数据面删除 + 本 rank 索引对齐）。

        动机：两侧索引是独立 LRU（只有调度侧在命中时刷新访问序），容量
        压力下选出的牺牲者可能不同——worker 删了 K 而调度侧仍报驻留
        （幽灵命中，由 forgotten 通道自愈），或调度侧删了 J 而 worker
        仍持有（盘上容量泄漏）。让调度侧把决策下发、worker 执行，可让
        两侧在常规情况下**按构造收敛**，不必等到事后自愈。

        保留 worker 自身的 plan_store 驱逐作为兜底：索引漂移时（调度侧
        选中的 key 在本 rank 并不驻留）这里是无操作，worker 仍会自行
        腾容量，不会出现"容量不足导致写入被拒"。

        过滤两条，其余跳过：
        - **本 rank 未驻留**——索引漂移时调度侧选中的 key 可能不在本 rank，
          删它既无对象也无意义（避免无谓的删除 IO）；
        - **读保护（pin）中**——在途读取的数据不能被删。

        反过来，"调度侧已删、本 rank 仍驻留"（容量泄漏方向）会被正常
        删除，因为该 key 在本 rank 是驻留的。
        """
        self._require_open()
        batch = [bytes(key) for key in keys or ()]
        if not batch:
            return 0
        index = self._index
        droppable = [
            key for key in batch
            if index.is_resident(key) and not index.is_pinned(key)
        ]
        if not droppable:
            return 0
        if self._num_layers is not None:
            self._store.drop(_expand_io_keys(droppable, self._num_layers))
        index.forget(droppable)
        self._planned_store_keys.difference_update(droppable)
        return len(droppable)

    def begin_step(self, max_pending_age: int) -> int:
        """推进一个步进并回收超龄的在途写入预留，返回回收数量。

        与调度侧同名方法对称：worker 侧的在途项正常由 wait_for_save 的
        confirm_store 结算，但该回调未必总被调用（save 抛错后上层跳过、
        请求被抢占），漏结算的项会永久占用容量并让同 key 的后续计划
        被拒（"在途重复计划"）。按步龄兜底回收。
        """
        self._require_open()
        self._index.advance_epoch()
        reclaimed = self._index.reclaim_stale_pending(max_pending_age)
        if reclaimed:
            self._planned_store_keys.difference_update(reclaimed)
            _LOG.warning(
                "PENDING_RECLAIMED_WORKER count=%d age>%d steps："
                "写入预留未在预期步数内结算（save 回调缺失），已回收",
                len(reclaimed), max_pending_age,
            )
        return len(reclaimed)

    def confirm_store(self, keys, ok: bool = True) -> None:
        """结算写入计划（转发语义索引）。"""
        self._require_open()
        if not ok:
            abort_chunks = getattr(self._store, "abort_chunks", None)
            if callable(abort_chunks):
                abort_chunks(keys)
        self._index.confirm_store(keys, ok)
        self._planned_store_keys.difference_update(keys)

    def store_plan_pending(self, keys) -> bool:
        """Return whether every key is already reserved by this engine's plan."""
        return all(bytes(key) in self._planned_store_keys for key in keys)

    def pin(self, keys) -> None:
        """对一批 chunk key 加读保护；任一未驻留 → KeyError。"""
        self._require_open()
        self._index.pin(keys)

    def unpin(self, keys) -> None:
        """解除读保护；无保护计数的 key → KeyError。"""
        self._require_open()
        self._index.unpin(keys)

    @property
    def capacity_chunks(self) -> int:
        """可容纳的 chunk 总数（转发 store 容量）。"""
        return self._store.capacity_chunks

    # ---- 执行态 ----

    @property
    def direct(self) -> bool:
        return isinstance(self._transfer, DirectTransfer)

    def try_bind_direct(self, kv_pool, num_layers: int,
                        blocks_per_chunk: int) -> bool:
        """Attempt direct registration before any staging object is created."""
        self._require_open()
        if self._transfer is not None:
            raise RuntimeError("bind 恰允许一次")
        if not is_int(num_layers) or num_layers <= 0:
            raise ValueError(f"num_layers 须为正整数，got {num_layers!r}")
        if self._num_layers is not None and self._num_layers != num_layers:
            raise ValueError(
                f"bind 层数 {num_layers} 与构造预告 num_layers"
                f"({self._num_layers}) 不一致"
            )
        if not is_int(blocks_per_chunk) or blocks_per_chunk <= 0:
            raise ValueError(
                f"blocks_per_chunk 须为正整数，got {blocks_per_chunk!r}"
            )
        if self._chunk_kv_bytes % num_layers:
            raise ValueError(
                f"chunk_kv_bytes({self._chunk_kv_bytes}) 不能被 "
                f"num_layers({num_layers}) 整分"
            )
        segment_bytes = self._chunk_kv_bytes // num_layers
        transfer = select_transfer(
            kv_pool,
            self._store,
            self._config,
            num_layers=num_layers,
            blocks_per_chunk=blocks_per_chunk,
            chunk_tokens=self._chunk_tokens,
            segment_bytes=segment_bytes,
            max_chunks_per_wave=self._max_chunks_per_wave,
        )
        if not isinstance(transfer, DirectTransfer):
            return False
        try:
            setter = getattr(self._store, "set_layer_span", None)
            if callable(setter):
                setter(num_layers)
            # 池就绪后立刻把每个 DataPath 的 peer-memory 注册做掉：注册是
            # 惰性的（首次 submit 触发），对 48.7GB 的 KV 池是 200~275ms/盘
            # 且持 runtime registry 锁，8 卡首轮实测 476ms 全落在首个请求里。
            warmer = getattr(transfer, "warm_up_registration", None)
            if callable(warmer):
                warmer()
        except Exception:
            transfer.close()
            raise
        self._num_layers = num_layers
        self._segment_bytes = segment_bytes
        self._transfer = transfer
        self._scatter_hook = None
        self._window = None
        self._read_window = None
        self._write_window = None
        self._staging_buffer_id = None
        self._read_staging_buffer_id = None
        self._write_staging_buffer_id = None
        self._deferred_restore()
        return True

    def validate_direct_block_tables(self, block_tables) -> None:
        if isinstance(self._transfer, DirectTransfer):
            self._transfer.validate_block_tables(block_tables)

    def fallback_from_direct(self, reason: Exception) -> None:
        """直连准入失败时的处置：解除绑定后抛出，不再降级到 staged。

        方法名保留为 ``fallback_...`` 是因为调用方（Worker 在提交前校验
        block tables）的语义仍是"请求回退"；但自回退取消后，唯一的处置就是失败。
        把这条契约留在 Engine 上而不是让 Worker 直接 raise，是为了让"直连一旦
        绑定就不再降级"只在一处定义。

        先 close 再抛：直连绑定已建立、池已注册，放弃时必须解除，否则会在 runtime
        侧留下悬挂注册。这件事与"是否回退到 staged"无关，因此保留。

        运行期触发条件是 block table 长度与该 chunk 的块数不符，即 chunk 不满。
        而保存计划按包络推进、只统计完整 chunk，故该条件不应成立——真成立就意味着
        上层契约被破坏，此时静默换路径只会掩盖缺陷。
        """
        if not isinstance(self._transfer, DirectTransfer):
            return
        self._transfer.close()
        self._transfer = None
        raise DirectTransferUnavailable(str(reason)) from reason

    def bind(
        self,
        kv_caches: dict,
        window: RingWindow,
        num_layers: int,
        blocks_per_chunk: int,
        *,
        write_window: RingWindow | None = None,
        gather_fn=None,
        scatter_fn=None,
        force_staged: bool = False,
    ) -> None:
        """绑定执行态：层数定案、选定传输路径、staging 环窗接入 store。

        kv_caches 为布局侧的层缓存映射（staged 路径不使用）；window 为
        read 环窗，write_window 为 write 环窗。write_window 缺省时两方向
        共享旧环窗；num_layers、blocks_per_chunk 为正整数。
        gather_fn / scatter_fn 为可选搬运钩子，给出时优先于 config 中
        的同名项（部署层在绑定期按池几何构造后注入）。
        重复 bind、chunk_kv_bytes 不能按层数整分、窗口几何不匹配、
        staging 环窗被 store 拒收 → RuntimeError / ValueError。
        """
        self._require_open()
        if self._transfer is not None:
            raise RuntimeError("bind 恰允许一次")
        if not is_int(num_layers) or num_layers <= 0:
            raise ValueError(f"num_layers 须为正整数，got {num_layers!r}")
        if self._num_layers is not None and self._num_layers != num_layers:
            raise ValueError(
                f"bind 层数 {num_layers} 与构造预告 num_layers"
                f"({self._num_layers}) 不一致"
            )
        if not is_int(blocks_per_chunk) or blocks_per_chunk <= 0:
            raise ValueError(f"blocks_per_chunk 须为正整数，got {blocks_per_chunk!r}")
        if not isinstance(window, RingWindow):
            raise ValueError(f"window 须为 RingWindow 实例，got {window!r}")
        if write_window is None:
            write_window = window
        if not isinstance(write_window, RingWindow):
            raise ValueError(
                f"write_window 须为 RingWindow 实例，got {write_window!r}"
            )
        for name, fn in (("gather_fn", gather_fn), ("scatter_fn", scatter_fn)):
            if fn is not None and not callable(fn):
                raise ValueError(f"{name} 须为可调用或 None，got {fn!r}")
        if self._chunk_kv_bytes % num_layers != 0:
            raise ValueError(
                f"chunk_kv_bytes({self._chunk_kv_bytes}) 不能被 "
                f"num_layers({num_layers}) 整分"
            )
        segment_bytes = self._chunk_kv_bytes // num_layers
        for direction, bank in (("read", window), ("write", write_window)):
            if bank.segment_bytes != segment_bytes:
                raise ValueError(
                    f"{direction} 窗口槽宽 {bank.segment_bytes} 与层段宽 "
                    f"{segment_bytes} 不匹配"
                )
            if bank.capacity_per_wave < self._max_chunks_per_wave:
                raise ValueError(
                    f"{direction} 窗口单波容量 {bank.capacity_per_wave} 小于 "
                    f"max_chunks_per_wave({self._max_chunks_per_wave})"
                )
        if window is not write_window and window.buffer is write_window.buffer:
            read_slots = set(range(
                window.slot_base, window.slot_base + window.num_slots
            ))
            write_slots = set(range(
                write_window.slot_base,
                write_window.slot_base + write_window.num_slots,
            ))
            if read_slots.intersection(write_slots):
                raise ValueError("read/write RingWindow 物理槽区间不得重叠")
        self._num_layers = num_layers
        self._segment_bytes = segment_bytes
        self._window = window
        self._read_window = window
        self._write_window = write_window
        config = dict(self._config)
        if gather_fn is not None:
            config["gather_fn"] = gather_fn
        if scatter_fn is not None:
            config["scatter_fn"] = scatter_fn
        self._scatter_hook = config.get("scatter_fn")
        self._transfer = (
            StagedTransfer(config.get("gather_fn"), config.get("scatter_fn"))
            if force_staged else select_transfer(
                kv_caches,
                self._store,
                config,
                num_layers=num_layers,
                blocks_per_chunk=blocks_per_chunk,
                chunk_tokens=self._chunk_tokens,
                segment_bytes=segment_bytes,
                max_chunks_per_wave=self._max_chunks_per_wave,
            )
        )
        if isinstance(self._transfer, DirectTransfer):
            # Direct backends register the vLLM paged tensors themselves and
            # submit block-table aware IO.  No staging buffer id is valid for
            # this path; load/store delegate below and retain the same engine
            # completion contract.
            self._staging_buffer_id = None
            self._read_staging_buffer_id = None
            self._write_staging_buffer_id = None
            self._deferred_restore()
            return
        # 层宽注入布局（可选实现）：段数据首写即全尺寸，避免
        # 逐层增长令传输层票据失效重开、票据池耗尽。
        setter = getattr(self._store, "set_layer_span", None)
        if setter is not None:
            setter(num_layers)
        buffer_id = self._store.register_buffer(window.buffer, segment_bytes)
        if buffer_id is None:
            raise RuntimeError(
                f"staging 环窗被 store 拒收（granularity={segment_bytes}）"
            )
        self._staging_buffer_id = buffer_id
        self._read_staging_buffer_id = buffer_id
        if write_window.buffer is window.buffer:
            write_buffer_id = buffer_id
        else:
            write_buffer_id = self._store.register_buffer(
                write_window.buffer, segment_bytes
            )
            if write_buffer_id is None:
                raise RuntimeError(
                    f"write staging 环窗被 store 拒收"
                    f"（granularity={segment_bytes}）"
                )
        self._write_staging_buffer_id = write_buffer_id
        self._deferred_restore()

    def load_layer(self, keys, layer_idx: int, dst_first_blocks, *,
                   fence_event=None, reuse_event=None, async_reuse=False,
                   bridge_compute=True):
        """发起一批读取：一层 × N chunk，持久化 → staging 槽 → 目的侧。

        返回聚合完成句柄；wait 返回即全部波次的源侧搬运（scatter）
        已执行。层号越界、未 bind → ValueError / RuntimeError。store
        侧未知 key 的异常原样上抛。
        """
        keys = self._prepare_layer_call(keys, layer_idx)
        if isinstance(self._transfer, DirectTransfer):
            direct_lock = getattr(self, "_direct_submit_lock", None)
            with (direct_lock if direct_lock is not None else nullcontext()):
                completion = self._transfer.load_layer(
                    keys, layer_idx, dst_first_blocks
                )
                record_read = getattr(self._store, "record_read_event", None)
                if fence_event is not None and not callable(record_read):
                    raise RuntimeError("direct read event recorder is unavailable")
                if callable(record_read):
                    fence_started_ns = time.perf_counter_ns()
                    with nvtx_range(
                        f"tutti.direct.record_fence|direction=read|layer={layer_idx}"
                    ):
                        recorded = record_read(fence_event)
                    _LOG.info(
                        "DIRECT_EVENT_RECORD direction=read layer=%d elapsed_ms=%.3f",
                        layer_idx,
                        (time.perf_counter_ns() - fence_started_ns) / 1_000_000,
                    )
                    if fence_event is not None and recorded is None:
                        raise RuntimeError("direct read fence was not recorded")
                    completion.fence_event = recorded
                self._inflight.append(completion)
                return completion
        handles = []
        reuse_fence = reuse_event
        for start in range(0, len(keys), self._max_chunks_per_wave):
            end = min(start + self._max_chunks_per_wave, len(keys))
            wave_keys = keys[start:end]
            wave_blocks = _slice_first_blocks(
                dst_first_blocks, start, end, len(keys)
            )
            if async_reuse and reuse_fence is not None:
                wait_copy = getattr(self._store, "wait_read_event", None)
                if callable(wait_copy):
                    wait_copy(reuse_fence)
            wave, slots = self._read_window.acquire(
                len(wave_keys), wait_for_reuse=not async_reuse
            )
            batch = [
                (derive_io_key(k, layer_idx), self._read_staging_buffer_id,
                 self._read_window.slot_offset(s))
                for k, s in zip(wave_keys, slots)
            ]
            with nvtx_range(
                f"tutti.load.submit|layer={layer_idx}|wave={wave}"
                f"|chunks={len(wave_keys)}"
            ):
                try:
                    completion = self._store.get_batch(batch)
                except Exception as exc:
                    _LOG.error(
                        "READ_ADMISSION_REJECTED layer=%d wave=%d "
                        "chunks=%d reason=%s",
                        layer_idx, wave, len(wave_keys), exc,
                    )
                    raise
            if not async_reuse and fence_event is None:
                handles.append(
                    self._settle(
                        wave, wave_keys, layer_idx, wave_blocks, slots,
                        completion, is_load=True
                    )
                )
                continue
            # The IO submission itself is the only operation on the read
            # stream.  Record its completion before touching read-copy, then
            # enqueue exactly one batched scatter there.  The returned handle
            # only drains runtime completion; scatter is never host-triggered.
            read_done = None
            record_read = getattr(self._store, "record_read_event", None)
            if callable(record_read):
                read_done = record_read()
            wait_copy = getattr(self._store, "wait_read_copy_event", None)
            if callable(wait_copy) and read_done is not None:
                wait_copy(read_done)
            with nvtx_range(
                f"tutti.load.scatter|layer={layer_idx}|wave={wave}"
                f"|chunks={len(wave_keys)}"
            ):
                scatter_event = self._scatter_on_read_copy(
                    list(wave_keys), layer_idx, wave_blocks, list(slots),
                    record=False,
                )
            if fence_event is not None:
                record_copy = getattr(
                    self._store, "record_read_copy_event", None
                )
                if callable(record_copy):
                    record_copy(fence_event)
                # A layer may contain several waves.  The next wave can wrap
                # onto the same staging slots, so it waits on this wave's
                # scatter fence as well as on the predecessor layer fence.
                reuse_fence = fence_event
            elif scatter_event is not None:
                reuse_fence = scatter_event
            post = _PostCompletion(
                completion,
                lambda: None,
                block_tables=wave_blocks,
                fence_event=(
                    fence_event if fence_event is not None else scatter_event
                ),
            )
            self._read_window.complete(wave, post)
            handles.append(post)
        aggregate = _AggregateCompletion(handles)
        del self._inflight[-len(handles):]
        self._inflight.append(aggregate)
        return aggregate

    @property
    def read_plan_supported(self) -> bool:
        """Whether Python can bridge read and read-copy CUDA streams."""
        store = self._store
        if self.direct:
            return (
                getattr(store, "_read_stream_obj", None) is not None
                and callable(getattr(store, "record_read_event", None))
                and callable(getattr(store, "wait_compute_event", None))
            )
        return (
            getattr(store, "_read_stream_obj", None) is not None
            and getattr(store, "_read_copy_stream_obj", None) is not None
            and all(callable(getattr(store, name, None)) for name in (
                "record_read_event", "wait_read_copy_event", "stream_context",
            ))
        )

    def start_read_plan(self, keys, block_tables, physical_layers, depth,
                        on_failure=None):
        if not self.read_plan_supported:
            raise RuntimeError("read CUDA event bridge unavailable")
        if self.direct:
            backend = getattr(self._transfer, "_backend", None)
            begin_plan = getattr(backend, "begin_target_plan", None)
            if callable(begin_plan):
                begin_plan(keys, "read")
            plan = _DirectAllLayerReadPlan(
                self, keys, block_tables, physical_layers, on_failure
            )
            self._active_read_plan = plan
            return plan
        return _ReadPlan(
            self, keys, block_tables, physical_layers, depth, on_failure
        )

    def prepare_write_targets(self, keys) -> None:
        """预置写入目标（对象池分配 + 票据就绪）。

        与 store_layer 内部的首层惰性调用等价且幂等（重复调用直接
        返回）。提前调用的意义是把对象池分配/manifest 落盘从"首个
        写入层"挪到 start_load_kv——那里读计划已预提交、计算尚未
        开始，开销不与层 0→1 的计算下发争用前向线程。
        """
        if not isinstance(self._transfer, DirectTransfer):
            return
        backend = getattr(self._transfer, "_backend", None)
        prepare = getattr(backend, "prepare_write_targets", None)
        if not callable(prepare):
            return
        direct_lock = getattr(self, "_direct_submit_lock", None)
        with (direct_lock if direct_lock is not None else nullcontext()):
            prepare(keys)

    def store_layer(self, keys, layer_idx: int, src_first_blocks):
        """发起一批写入：一层 × N chunk，源侧 → staging 槽 → 持久化。

        返回聚合完成句柄；源侧搬运（gather）在各波提交前已执行。其余
        契约同 load_layer。
        """
        keys = self._prepare_layer_call(keys, layer_idx)
        if isinstance(self._transfer, DirectTransfer):
            direct_lock = getattr(self, "_direct_submit_lock", None)
            with (direct_lock if direct_lock is not None else nullcontext()):
                backend = getattr(self._transfer, "_backend", None)
                # 走公开查询接口，不窥探 backend 的私有计划表（见阶段 C）。
                has_plan = getattr(backend, "has_write_plan", None)
                planned = (bool(has_plan()) if callable(has_plan)
                           else False)
                if not planned:
                    prepare = getattr(backend, "prepare_write_targets", None)
                    begin = getattr(backend, "begin_target_plan", None)
                    if callable(prepare) != callable(begin):
                        raise RuntimeError(
                            "direct backend lacks write target planning"
                        )
                    if callable(prepare):
                        prepare(keys)
                        begin(keys, "write")
                record_compute = getattr(self._store, "record_compute_event", None)
                wait_write = getattr(self._store, "wait_write_event", None)
                # 两半能力的要求**不对称**，理由在语义而不在对称美观：
                #
                #   record_compute 缺失 → 无从产生事件 → 这条 fence 无法建立，走
                #     普通路径由 store 自行保证顺序（参考内存后端即是同步写）。
                #     此时 wait_write 有没有都无所谓：没有事件可等。
                #
                #   record_compute 存在但没有 wait_write → 能记录却无法让写等待，
                #     "compute 先完成、写后提交"这一顺序会被静默丢掉。这才是真正
                #     危险的一侧，必须报错。
                #
                # 曾经这里写成"非两者皆备即报错"，把无害的不对称（只有 wait）
                # 与危险的不对称混为一谈：参考内存后端补上 wait_write_event 后，
                # 立刻从"两者皆无"掉进这条错误分支。
                if not callable(record_compute):
                    completion = self._transfer.store_layer(
                        keys, layer_idx, src_first_blocks
                    )
                    self._inflight.append(completion)
                    return completion
                if not callable(wait_write):
                    raise RuntimeError(
                        "direct store can record a compute fence but cannot "
                        "make the write wait on it; refusing to drop the "
                        "compute-to-write ordering"
                    )
                fence_started_ns = time.perf_counter_ns()
                with nvtx_range(
                    f"tutti.direct.record_fence|direction=write|layer={layer_idx}"
                ):
                    compute_done = record_compute()
                _LOG.info(
                    "DIRECT_EVENT_RECORD direction=write layer=%d elapsed_ms=%.3f",
                    layer_idx,
                    (time.perf_counter_ns() - fence_started_ns) / 1_000_000,
                )
                if compute_done is None:
                    raise RuntimeError("direct compute fence was not recorded")
                wait_write(compute_done)
                completion = self._transfer.store_layer(
                    keys, layer_idx, src_first_blocks
                )
                self._inflight.append(completion)
                return completion
        handles = []
        reuse_event = self._write_reuse_event
        for start in range(0, len(keys), self._max_chunks_per_wave):
            end = min(start + self._max_chunks_per_wave, len(keys))
            wave_keys = keys[start:end]
            wave_blocks = _slice_first_blocks(
                src_first_blocks, start, end, len(keys)
            )
            if reuse_event is not None:
                wait_write = getattr(self._store, "wait_write_event", None)
                if not callable(wait_write):
                    wait_write = getattr(self._store, "wait_event", None)
                if not callable(wait_write):
                    raise RuntimeError(
                        "write staging reuse requires a write completion fence"
                    )
                wait_write(reuse_event)
            # Never let RingWindow perform its host-side reuse wait from a
            # save callback.  The preceding write fence, when available, is
            # consumed by the write stream above.
            wave, slots = self._write_window.acquire(
                len(wave_keys), wait_for_reuse=False
            )
            with nvtx_range(
                f"tutti.store.submit|layer={layer_idx}|wave={wave}"
                f"|chunks={len(wave_keys)}"
            ):
                # Gather is produced on vLLM's current compute stream. The
                # store fences its write stream on the returned event before
                # submitting NVMe IO, preserving compute -> gather -> write.
                gather_event = self._transfer.gather(
                    wave_keys, layer_idx, wave_blocks, slots
                )
            if gather_event is not None:
                wait_event = getattr(self._store, "wait_write_event", None)
                if not callable(wait_event):
                    wait_event = getattr(self._store, "wait_event", None)
                if callable(wait_event):
                    wait_event(gather_event)
                else:
                    raise RuntimeError(
                        "write stream fence bridge is unavailable; refusing "
                        "host synchronization in save callback"
                    )
            batch = [
                (derive_io_key(k, layer_idx), self._write_staging_buffer_id,
                 self._write_window.slot_offset(s))
                for k, s in zip(wave_keys, slots)
            ]
            with nvtx_range(
                f"tutti.store.io|layer={layer_idx}|wave={wave}"
                f"|chunks={len(wave_keys)}"
            ):
                try:
                    completion = self._store.put_batch(batch)
                except Exception as exc:
                    _LOG.error(
                        "WRITE_ADMISSION_REJECTED layer=%d wave=%d "
                        "chunks=%d reason=%s",
                        layer_idx, wave, len(wave_keys), exc,
                    )
                    raise
            # Keep a write-stream fence for the next wave/layer's staging
            # reuse.  Stores with a real CUDA write stream record this event
            # after the put enqueue; stores lacking that capability are
            # rejected on the first attempted reuse instead of overwriting a
            # live slot.
            record_write = getattr(self._store, "record_write_event", None)
            if callable(record_write):
                recorded = record_write()
                if recorded is not None:
                    reuse_event = recorded
            handles.append(
                self._settle(
                    wave, wave_keys, layer_idx, wave_blocks, slots,
                    completion, is_load=False
                )
            )
        self._write_reuse_event = reuse_event
        aggregate = _AggregateCompletion(handles)
        del self._inflight[-len(handles):]
        self._inflight.append(aggregate)
        return aggregate

    def wait_idle(self) -> None:
        """等待全部在途批次并 drain read/write 两个 bank。"""
        active_plan = getattr(self, "_active_read_plan", None)
        if active_plan is not None:
            active_plan.join_feeder()
        inflight = self._inflight
        self._inflight = []
        first_error = None
        drain_started_ns = time.perf_counter_ns()
        context = (
            nvtx_range(
                f"tutti.direct.completion_drain|completions={len(inflight)}"
            ) if self.direct else nullcontext()
        )
        with context:
            for completion in inflight:
                try:
                    completion.wait()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
        if self.direct and inflight:
            stats = [
                getattr(completion, "drain_stats", {})
                for completion in inflight
            ]
            # 正常路径的排空统计：debug（原先 warning，每步都刷）。
            _LOG.debug(
                "DIRECT_COMPLETION_DRAIN completions=%d wait_result_calls=%d "
                "release_io_calls=%d failed=%d read=%d write=%d elapsed_ms=%.3f",
                len(inflight),
                sum(item.get("wait_result_calls", 0) for item in stats),
                sum(item.get("release_io_calls", 0) for item in stats),
                sum(bool(item.get("failed")) for item in stats),
                sum(item.get("direction") == "read" for item in stats),
                sum(item.get("direction") == "write" for item in stats),
                (time.perf_counter_ns() - drain_started_ns) / 1_000_000,
            )
        for window in self._unique_windows():
            try:
                window.drain()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        drain_deferred = getattr(self._store, "drain_deferred_completions", None)
        if callable(drain_deferred):
            try:
                drain_deferred()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self._end_direct_target_plans()
        finalize_failures = getattr(self._store, "finalize_direct_failures", None)
        if callable(finalize_failures):
            try:
                finalize_failures()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self._write_reuse_event = None
        if first_error is not None:
            raise first_error

    def abort(self, timeout=None) -> None:
        """安全中间态 abort：不取消底层 DMA，drain 两边后再返回。"""
        active_plan = getattr(self, "_active_read_plan", None)
        if active_plan is not None:
            active_plan.abort()
            self._active_read_plan = None
        inflight = self._inflight
        self._inflight = []
        first_error = None
        context = (
            nvtx_range(
                f"tutti.direct.abort_drain|completions={len(inflight)}"
            ) if self.direct else nullcontext()
        )
        with context:
            for completion in inflight:
                abort = getattr(completion, "abort", None)
                try:
                    if callable(abort):
                        abort()
                    else:
                        completion.wait()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
        for window in self._unique_windows():
            try:
                window.drain()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        drain_deferred = getattr(self._store, "drain_deferred_completions", None)
        if callable(drain_deferred):
            try:
                drain_deferred()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self._end_direct_target_plans()
        finalize_failures = getattr(self._store, "finalize_direct_failures", None)
        if callable(finalize_failures):
            try:
                finalize_failures()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self._write_reuse_event = None
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        """收尾：等待在途批次并关闭 store；幂等。"""
        if self._closed:
            return
        first_error = None
        try:
            self.wait_idle()
        except Exception as exc:
            first_error = exc
        self._active_read_plan = None
        if isinstance(self._transfer, DirectTransfer):
            try:
                self._transfer.close()
            except Exception as exc:
                backend = getattr(self._transfer, "_backend", None)
                if not bool(getattr(backend, "_closed", False)):
                    raise
                if first_error is None:
                    first_error = exc
        self._store.close()
        self._closed = True
        if first_error is not None:
            raise first_error

    def _end_direct_target_plans(self) -> None:
        backend = getattr(self._transfer, "_backend", None)
        end_plan = getattr(backend, "end_target_plan", None)
        if not callable(end_plan):
            return
        # 走公开查询接口（含尚未提交的预备写计划），不窥探私有计划表。
        directions = getattr(backend, "planned_directions", None)
        if callable(directions):
            for direction in tuple(directions()):
                end_plan(direction)
            return
        # 兼容不含该接口的轻量替身：退化为仅结束预备写计划。
        if getattr(backend, "_prepared_write_chunks", None) is not None:
            end_plan("write")

    # ---- 内部 ----

    def _prepare_layer_call(self, keys, layer_idx: int) -> list[bytes]:
        """校验执行态前置条件，返回保持原序的 key 列表。"""
        self._require_open()
        if self._transfer is None:
            raise RuntimeError("执行态方法须在 bind 之后调用")
        if not is_int(layer_idx) or not 0 <= layer_idx < self._num_layers:
            raise ValueError(
                f"layer_idx 须在 [0, {self._num_layers}) 内，got {layer_idx!r}"
            )
        keys = list(keys)
        if not keys:
            raise ValueError("keys 不能为空")
        return keys

    def _settle(
        self,
        wave: int,
        keys: list[bytes],
        layer_idx: int,
        first_blocks,
        slots: list[int],
        completion,
        is_load: bool,
    ):
        """登记波次完成事件与在途句柄；load 方向追加目的侧搬运。"""
        handle = completion
        if is_load:
            def scatter_and_capture():
                if self._scatter_hook is None:
                    return None
                with nvtx_range(
                    f"tutti.load.scatter|layer={layer_idx}|wave={wave}"
                    f"|chunks={len(keys)}"
                ):
                    return self._scatter_on_read_copy(
                        list(keys), layer_idx, first_blocks, list(slots)
                    )

            handle = _PostCompletion(
                completion,
                scatter_and_capture,
                block_tables=first_blocks,
                after_event=lambda event: self._bridge_read_copy_event(
                    event, protect_read_io=True
                ),
            )
        window = self._read_window if is_load else self._write_window
        window.complete(wave, handle)
        self._inflight.append(handle)
        return handle

    def _store_stream_context(self, direction: str):
        context = getattr(self._store, "stream_context", None)
        return context(direction) if callable(context) else nullcontext()

    def _scatter_on_read_copy(self, keys, layer_idx, block_tables, slots,
                              *, record=True):
        """Enqueue scatter on the controlled copy stream and record its tail."""
        if self._scatter_hook is None:
            return None
        with self._store_stream_context("read_copy"):
            event = self._scatter_hook(
                list(keys), layer_idx, block_tables, list(slots)
            )
            recorder = getattr(self._store, "record_read_copy_event", None)
            return recorder(event) if record and callable(recorder) else event

    def _bridge_read_copy_event(self, event, *, protect_read_io: bool) -> None:
        """Bridge copy completion without a host CUDA synchronization."""
        if event is None:
            return
        wait_compute = getattr(self._store, "wait_compute_event", None)
        if callable(wait_compute):
            wait_compute(event)
        else:
            wait = getattr(event, "wait", None)
            if callable(wait):
                wait()
        if protect_read_io:
            wait_read = getattr(self._store, "wait_read_event", None)
            if callable(wait_read):
                wait_read(event)

    def _read_copy_stream_handle(self) -> int:
        handle = getattr(self._store, "read_copy_stream_handle", None)
        if callable(handle):
            return int(handle())
        try:
            import torch
            return int(torch.cuda.current_stream().cuda_stream)
        except Exception:
            return 0

    def _unique_windows(self):
        windows = []
        for window in (self._read_window, self._write_window):
            if window is not None and all(window is not item for item in windows):
                windows.append(window)
        return windows

    def sync_from_store(self) -> None:
        """从盘上持久层枚举对账近似索引（幂等，可重复调用）。

        多副本部署下索引属主与命中查询方可能分属不同进程（worker
        落盘、调度侧查询），查询方以盘上标记为准对账——仅扫持久层
        元数据目录，不触碰数据面。层集合不完整的 chunk 视为缺失
        （miss 语义，不驻留）；盘上完整层组消失时移除近似项
        （完整性翻转修正，见 ChunkIndex.reconcile）。
        """
        self._require_open()
        groups = group_scan(self._store)
        expected = set(range(self._num_layers or 0))
        # 灌入序 = 枚举序（确定；restore 的首次灌入序即 LRU 初始序）
        full_keys = [
            chunk_key for chunk_key, layers in groups.items()
            if layers >= expected
        ]
        full = set(full_keys)
        # Scheduler and worker may live in different processes.  Scheduler
        # plans reserve capacity but must not publish resident before the
        # worker's durable layer markers exist.  The next authoritative scan
        # settles those local reservations: complete marker groups become
        # resident; absent/incomplete groups release pending fail-closed.
        for key in list(self._planned_store_keys):
            self._index.confirm_store([key], ok=key in full)
            self._planned_store_keys.discard(key)
        # 完整性翻转修正：上次对账时完整、现已不完整/消失的组移除
        # 近似项；乐观受理项（从未在盘上判完整）不受影响——worker
        # 落盘前它们只在近似视图，miss 降级兜底。pin 保护而未移除
        # 的项留存至下次对账重试（防翻转事实随基准推进丢失）。
        stale = (self._synced_full - full) | self._pending_forget
        self._pending_forget = set(self._index.forget(stale))
        self._index.restore(full_keys)
        self._synced_full = full
        self._scan_groups = groups
        self._restored = True

    def _deferred_restore(self) -> None:
        """层数定案后执行冷启动灌入：层完整的 chunk 才驻留。

        层集合不完整的 chunk 视为缺失（miss 语义，不驻留索引——
        命中查询不会报告该 chunk）。
        """
        if self._restored:
            return
        self.sync_from_store()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("engine 已 close")


def _expand_io_keys(chunk_keys, num_layers: int) -> list[bytes]:
    """把一批 chunk key 展开为全部层的 io_key。"""
    return [
        derive_io_key(chunk_key, layer)
        for chunk_key in chunk_keys
        for layer in range(num_layers)
    ]


def _slice_first_blocks(first_blocks, start: int, end: int, total: int):
    """按 chunk 位置切片块表；标量/None 参数在各波保持原样。"""
    if first_blocks is None:
        return None
    try:
        size = len(first_blocks)
    except TypeError:
        return first_blocks
    if size != total or isinstance(first_blocks, (bytes, bytearray, str)):
        return first_blocks
    try:
        return first_blocks[start:end]
    except TypeError:
        return first_blocks





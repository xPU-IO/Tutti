"""KVEngine：计划态转发语义索引，执行态编排 staging 环窗、store 与传输路径。"""

from __future__ import annotations

from contextlib import nullcontext
import logging
import os
import time
from typing import Sequence

from engine.staging import RingWindow
from engine.transfer import DirectTransfer, StagedTransfer, select_transfer
from engine.nvtx import range as nvtx_range
from index.chunk_index import (
    ChunkIndex,
    StorePlan,
    chunk_key_of,
    derive_io_key,
    layer_of,
)

# 冷启动恢复时 io_key 分组的来源宽度（chunk key 16 字节 + 层编号 2 字节）
_IO_KEY_BYTES = 18
_LOG = logging.getLogger(__name__)
_FEEDER_DIAG = os.environ.get("TUTTI_FEEDER_DIAGNOSTICS") == "1"


class LoadGateError(RuntimeError):
    """A layer's runtime/CQ completion failed before scatter."""

    def __init__(self, message: str, failed_batch_indices=(),
                 whole_operation: bool = True, invalid_block_ids=()):
        super().__init__(message)
        self.failed_batch_indices = tuple(failed_batch_indices)
        self.whole_operation = bool(whole_operation)
        self.invalid_block_ids = tuple(invalid_block_ids)


class _PostCompletion:
    """底层完成句柄与消费侧事件组成的完成句柄。

    契约：wait 先等底层完成，再执行收尾动作并等待其返回的事件（恰一
    次）；query 在收尾尚未启动时反映底层状态，事件已产生后反映事件状态。
    """

    __slots__ = ("_inner", "_after", "_event", "_done", "_block_tables")

    def __init__(self, inner, after, block_tables=None):
        """inner 为底层完成句柄，after 为无参收尾可调用。"""
        self._inner = inner
        self._after = after
        self._event = None
        self._done = False
        self._block_tables = block_tables

    def wait(self) -> None:
        """阻塞至底层完成并执行收尾动作（恰一次）。"""
        if self._done:
            return
        wait_result = getattr(self._inner, "wait_result", None)
        if callable(wait_result):
            result = wait_result()
            if not getattr(result, "ok", True):
                failed = tuple(getattr(result, "failed_batch_indices", ()) or ())
                failures = tuple(getattr(result, "failures", ()) or ())
                whole = not failed or any(
                    getattr(item, "failure_scope", "WHOLE_OPERATION")
                    != "REQUEST_INDICES" for item in failures
                )
                selected = self._block_tables
                if not whole and failed and self._block_tables is not None:
                    selected = [
                        self._block_tables[index]
                        for index in failed
                        if 0 <= index < len(self._block_tables)
                    ]
                invalid_blocks = _flatten_block_ids(selected)
                raise LoadGateError(
                    "底层 IO 失败，禁止 scatter",
                    failed_batch_indices=failed,
                    whole_operation=whole,
                    invalid_block_ids=invalid_blocks,
                )
        else:
            # TODO(STRUCTURED-COMPLETION): remove this compatibility branch
            # once every KVStore completion exposes wait_result().  A legacy
            # wait exception has no request-index detail, so WorkerImpl maps
            # it to WHOLE_OPERATION fail-closed.
            self._inner.wait()
        self._event = self._after()
        if self._event is not None:
            synchronize = getattr(self._event, "synchronize", None)
            if callable(synchronize):
                synchronize()
            else:
                self._event.wait()
        self._done = True

    def abort(self) -> None:
        """Drain the inner operation without running consumer-side ``after``.

        A look-ahead load may have already submitted later layers when an
        earlier layer fails.  Those operations still need CQ draining and
        release, but their scatter callback must never publish data after the
        step has entered the failed state.
        """
        if self._done:
            return
        try:
            wait_result = getattr(self._inner, "wait_result", None)
            if callable(wait_result):
                wait_result()
            else:
                self._inner.wait()
        except Exception:
            pass
        self._done = True

    def query(self) -> bool:
        """非阻塞查询底层是否完成。"""
        if not self._inner.query():
            return False
        if self._event is None:
            return True
        query = getattr(self._event, "query", None)
        return bool(query()) if callable(query) else self._done


class _AggregateCompletion:
    """按提交顺序组合多个波次完成句柄。

    每个子句柄已经登记到环窗，因此这里只负责对外提供一个句柄：
    ``wait`` 顺序等待全部波次，``query`` 只有在全部波次完成时才返回
    true。顺序等待也保证同一批的 scatter/gather 收尾契约不被重排。
    """

    __slots__ = ("_handles", "_done")

    def __init__(self, handles):
        self._handles = tuple(handles)
        self._done = False

    def wait(self) -> None:
        if self._done:
            return
        for handle in self._handles:
            handle.wait()
        self._done = True

    def abort(self) -> None:
        """Drain child operations while suppressing all scatter callbacks."""
        if self._done:
            return
        for handle in self._handles:
            abort = getattr(handle, "abort", None)
            try:
                if callable(abort):
                    abort()
                else:
                    handle.wait()
            except Exception:
                pass
        self._done = True

    def query(self) -> bool:
        if self._done:
            return True
        # 子句柄的 query 可能在其 after/scatter 尚未执行时已为真；不要
        # 在这里标记聚合句柄完成，否则后续 wait 会跳过必要收尾。
        return all(handle.query() for handle in self._handles)


class _EngineStepIO:
    """Engine-side layer callbacks over one store feeder completion."""

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
        self._drained = False

    def wait_layer(self, callback: int, physical: int) -> None:
        if not self._enter_callback(callback, physical):
            return
        if _FEEDER_DIAG:
            _LOG.warning("FEEDER_DIAG engine gate_wait_begin t_ns=%d direction=read callback=%d physical=%d",
                         time.monotonic_ns(), callback, physical)
        if not self._inner.wait_layer(callback):
            raise LoadGateError(
                f"step read feeder callback {callback} physical {physical} failed",
                whole_operation=True,
            )
        slots = self._slots[callback]
        event = None
        if self._engine._scatter_hook is not None:
            event = self._engine._scatter_hook(
                self._keys, physical, self._block_tables, slots
            )
        if event is not None:
            synchronize = getattr(event, "synchronize", None)
            if callable(synchronize):
                synchronize()
        if callback + self._inner.staging_depth < self._inner.layer_count:
            self._inner.signal_layer(callback, self._current_stream())
            if _FEEDER_DIAG:
                _LOG.warning("FEEDER_DIAG engine gate_release_publish t_ns=%d direction=read callback=%d physical=%d",
                             time.monotonic_ns(), callback, physical)
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
            _LOG.warning("FEEDER_DIAG engine gate_ready_publish t_ns=%d direction=write callback=%d physical=%d",
                         time.monotonic_ns(), callback, physical)

    def wait(self):
        self._require_complete()
        return self._inner.wait()

    def wait_result(self):
        self._require_complete()
        return self._inner.wait_result()

    def drain(self):
        if self._drained:
            return None
        self._drained = True
        return self._inner.drain(self._current_stream())

    def abort(self):
        return self.drain()

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
    - direct_transfer：可选 bool。请求 store-native 的 paged KV DMA
      后端（legacy GeminiFS 风格）；当前 contiguous tutti runtime 不具备
      该能力时自动回退 staged。
    - direct_transfer_strict：可选 bool。direct_transfer 能力缺失时失败，
      用于硬件部署验收，避免误把 staged 当成 direct。
    - gather_fn / scatter_fn：可选搬运钩子（缺省 None，搬运为 no-op），
      语义见传输路径。

    构造期：打开 store、建立语义索引、对 store 的存活枚举按 chunk key
    分组；层数定案（bind）后才把层完整的 chunk 灌入索引（见 io_key
    线格式约定）。
    """

    def __init__(self, config: dict, store):
        """config 见类契约；store 为 KVStore 实现。参数非法 → ValueError。"""
        self._chunk_tokens = _positive_int(config, "chunk_tokens")
        self._chunk_kv_bytes = _positive_int(config, "chunk_kv_bytes")
        self._max_chunks_per_wave = _positive_int(config, "max_chunks_per_wave")
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
        if layers_hint is not None and (not _is_int(layers_hint) or layers_hint <= 0):
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
        self._scan_groups = _group_scan(store)
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
        self._transfer: StagedTransfer | None = None
        self._scatter_hook = None
        self._staging_buffer_id: int | None = None
        self._read_staging_buffer_id: int | None = None
        self._write_staging_buffer_id: int | None = None
        self._inflight: list = []
        self._planned_store_keys: set[bytes] = set()

    @property
    def max_in_flight_operations(self) -> int:
        return self._max_in_flight_operations

    @property
    def supports_step_io(self) -> bool:
        return (callable(getattr(self._store, "submit_step", None)) and
                not isinstance(self._transfer, DirectTransfer))

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
        plan = self._index.plan_store(keys)
        if plan is None:
            return None
        self._planned_store_keys.update(plan.new_keys)
        if plan.evicted_keys and self._num_layers is not None:
            self._store.drop(_expand_io_keys(plan.evicted_keys, self._num_layers))
        return plan

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

    def begin_rank_commit(self, keys, generations) -> None:
        begin = getattr(self._store, "begin_rank_commit", None)
        if callable(begin):
            begin(keys, generations)

    def commit_rank_chunks(self, keys, generations) -> None:
        commit = getattr(self._store, "commit_rank_chunks", None)
        if callable(commit):
            commit(keys, generations)

    def abort_rank_commit(self, keys, generations=None) -> None:
        abort = getattr(self._store, "abort_rank_commit", None)
        if callable(abort):
            abort(keys, generations)

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
        if self._window is not None:
            raise RuntimeError("bind 恰允许一次")
        if not _is_int(num_layers) or num_layers <= 0:
            raise ValueError(f"num_layers 须为正整数，got {num_layers!r}")
        if self._num_layers is not None and self._num_layers != num_layers:
            raise ValueError(
                f"bind 层数 {num_layers} 与构造预告 num_layers"
                f"({self._num_layers}) 不一致"
            )
        if not _is_int(blocks_per_chunk) or blocks_per_chunk <= 0:
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
        self._transfer = select_transfer(
            kv_caches,
            self._store,
            config,
            num_layers=num_layers,
            blocks_per_chunk=blocks_per_chunk,
            chunk_tokens=self._chunk_tokens,
            segment_bytes=segment_bytes,
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

    def load_layer(self, keys, layer_idx: int, dst_first_blocks):
        """发起一批读取：一层 × N chunk，持久化 → staging 槽 → 目的侧。

        返回聚合完成句柄；wait 返回即全部波次的源侧搬运（scatter）
        已执行。层号越界、未 bind → ValueError / RuntimeError。store
        侧未知 key 的异常原样上抛。
        """
        keys = self._prepare_layer_call(keys, layer_idx)
        if isinstance(self._transfer, DirectTransfer):
            completion = self._transfer.load_layer(
                keys, layer_idx, dst_first_blocks
            )
            self._inflight.append(completion)
            return completion
        handles = []
        for start in range(0, len(keys), self._max_chunks_per_wave):
            end = min(start + self._max_chunks_per_wave, len(keys))
            wave_keys = keys[start:end]
            wave_blocks = _slice_first_blocks(
                dst_first_blocks, start, end, len(keys)
            )
            wave, slots = self._read_window.acquire(len(wave_keys))
            batch = [
                (derive_io_key(k, layer_idx), self._read_staging_buffer_id,
                 self._read_window.slot_offset(s))
                for k, s in zip(wave_keys, slots)
            ]
            with nvtx_range(
                f"tutti.load.submit|layer={layer_idx}|wave={wave}"
                f"|chunks={len(wave_keys)}"
            ):
                completion = self._store.get_batch(batch)
            handles.append(
                self._settle(
                    wave, wave_keys, layer_idx, wave_blocks, slots,
                    completion, is_load=True
                )
            )
        aggregate = _AggregateCompletion(handles)
        del self._inflight[-len(handles):]
        self._inflight.append(aggregate)
        return aggregate

    def store_layer(self, keys, layer_idx: int, src_first_blocks):
        """发起一批写入：一层 × N chunk，源侧 → staging 槽 → 持久化。

        返回聚合完成句柄；源侧搬运（gather）在各波提交前已执行。其余
        契约同 load_layer。
        """
        keys = self._prepare_layer_call(keys, layer_idx)
        if isinstance(self._transfer, DirectTransfer):
            completion = self._transfer.store_layer(
                keys, layer_idx, src_first_blocks
            )
            self._inflight.append(completion)
            return completion
        handles = []
        for start in range(0, len(keys), self._max_chunks_per_wave):
            end = min(start + self._max_chunks_per_wave, len(keys))
            wave_keys = keys[start:end]
            wave_blocks = _slice_first_blocks(
                src_first_blocks, start, end, len(keys)
            )
            wave, slots = self._write_window.acquire(len(wave_keys))
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
                    # Test/fallback stores have no IO stream to fence. Keep
                    # correctness with a stream-local event wait only.
                    synchronize = getattr(gather_event, "synchronize", None)
                    if callable(synchronize):
                        synchronize()
            batch = [
                (derive_io_key(k, layer_idx), self._write_staging_buffer_id,
                 self._write_window.slot_offset(s))
                for k, s in zip(wave_keys, slots)
            ]
            with nvtx_range(
                f"tutti.store.io|layer={layer_idx}|wave={wave}"
                f"|chunks={len(wave_keys)}"
            ):
                completion = self._store.put_batch(batch)
            handles.append(
                self._settle(
                    wave, wave_keys, layer_idx, wave_blocks, slots,
                    completion, is_load=False
                )
            )
        aggregate = _AggregateCompletion(handles)
        del self._inflight[-len(handles):]
        self._inflight.append(aggregate)
        return aggregate

    def start_step_io(self, keys, block_tables, direction: str, depth: int,
                      physical_layers=None):
        """Create one Runtime/DataPath feeder submit for every model layer."""
        self._require_open()
        if self._num_layers is None or self._segment_bytes is None:
            raise RuntimeError("step IO requires a bound engine")
        keys = list(keys)
        if not keys:
            return None
        window = self._read_window if direction == "read" else self._write_window
        if len(keys) > window.capacity_per_wave:
            raise ValueError(
                "step feeder chunks exceed one staging wave; increase "
                "max_chunks_per_wave"
            )
        if physical_layers is None:
            physical_layers = tuple(range(self._num_layers))
        else:
            physical_layers = tuple(physical_layers)
        if not physical_layers:
            return None
        if (len(set(physical_layers)) != len(physical_layers)
                or any(not isinstance(layer, int)
                       or not 0 <= layer < self._num_layers
                       for layer in physical_layers)):
            raise ValueError("physical feeder layer plan is invalid")
        slots_by_layer = []
        layer_batches = []
        for callback, physical in enumerate(physical_layers):
            base = window.slot_base + (
                callback % depth
            ) * window.capacity_per_wave
            slots = [base + index for index in range(len(keys))]
            slots_by_layer.append(slots)
            layer_batches.append([
                (derive_io_key(key, physical),
                 self._read_staging_buffer_id if direction == "read"
                 else self._write_staging_buffer_id,
                 window.slot_offset(slot))
                for key, slot in zip(keys, slots)
            ])
        if _FEEDER_DIAG:
            _LOG.warning(
                "FEEDER_DIAG physical_plan t_ns=%d direction=%s "
                "physical_layer_count=%d callback_count=%d depth=%d "
                "callback_to_physical=%r",
                time.monotonic_ns(), direction, self._num_layers,
                len(physical_layers), depth, physical_layers,
            )
        inner = self._store.submit_step(layer_batches, direction, depth)
        step = _EngineStepIO(
            inner, self, keys, block_tables, slots_by_layer, direction,
            physical_layers,
        )
        self._inflight.append(step)
        return step

    def wait_idle(self) -> None:
        """等待全部在途批次并 drain read/write 两个 bank。"""
        inflight = self._inflight
        self._inflight = []
        first_error = None
        for completion in inflight:
            try:
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
        if first_error is not None:
            raise first_error

    def abort(self) -> None:
        """安全中间态 abort：不取消底层 DMA，drain 两边后再返回。"""
        inflight = self._inflight
        self._inflight = []
        first_error = None
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
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        """收尾：等待在途批次并关闭 store；幂等。"""
        if self._closed:
            return
        self._closed = True
        try:
            self.wait_idle()
        finally:
            if isinstance(self._transfer, DirectTransfer):
                self._transfer.close()
            self._store.close()

    # ---- 内部 ----

    def _prepare_layer_call(self, keys, layer_idx: int) -> list[bytes]:
        """校验执行态前置条件，返回保持原序的 key 列表。"""
        self._require_open()
        if self._window is None:
            raise RuntimeError("执行态方法须在 bind 之后调用")
        if not _is_int(layer_idx) or not 0 <= layer_idx < self._num_layers:
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
                    with self._store_stream_context("read"):
                        return self._scatter_hook(
                            list(keys), layer_idx, first_blocks, list(slots)
                        )

            handle = _PostCompletion(
                completion,
                scatter_and_capture,
                block_tables=first_blocks,
            )
        window = self._read_window if is_load else self._write_window
        window.complete(wave, handle)
        self._inflight.append(handle)
        return handle

    def _store_stream_context(self, direction: str):
        context = getattr(self._store, "stream_context", None)
        return context(direction) if callable(context) else nullcontext()

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
        groups = _group_scan(self._store)
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


def _group_scan(store) -> dict[bytes, set[int]]:
    """把 store 的存活枚举按 chunk key 分组为层集合；线格式非法的条目忽略。"""
    groups: dict[bytes, set[int]] = {}
    for io_key in store.scan():
        if not isinstance(io_key, (bytes, bytearray)) or len(io_key) != _IO_KEY_BYTES:
            continue
        chunk_key = chunk_key_of(bytes(io_key))
        groups.setdefault(chunk_key, set()).add(layer_of(bytes(io_key)))
    return groups


def _expand_io_keys(chunk_keys, num_layers: int) -> list[bytes]:
    """把一批 chunk key 展开为全部层的 io_key。"""
    return [
        derive_io_key(chunk_key, layer)
        for chunk_key in chunk_keys
        for layer in range(num_layers)
    ]


def _positive_int(config: dict, key: str) -> int:
    """从 config 读取正整数键；缺失或非法 → ValueError。"""
    value = config.get(key)
    if not _is_int(value) or value <= 0:
        raise ValueError(f"config[{key!r}] 须为正整数，got {value!r}")
    return value


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


def _flatten_block_ids(block_tables) -> tuple[int, ...]:
    """Flatten wave-local block tables for fail-closed error reporting."""
    if block_tables is None:
        return ()
    if isinstance(block_tables, (bytes, bytearray, str)):
        return ()
    try:
        values = list(block_tables)
    except TypeError:
        return (block_tables,) if isinstance(block_tables, int) else ()
    flattened = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            flattened.extend(item for item in value if isinstance(item, int))
        elif isinstance(value, int):
            flattened.append(value)
    return tuple(flattened)


def _is_int(value) -> bool:
    """判断是否为真 int（排除 bool）。"""
    return isinstance(value, int) and not isinstance(value, bool)

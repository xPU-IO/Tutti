"""完成句柄：底层 IO 完成与消费侧收尾的组合/聚合。

从 engine/core.py 搬出（评审意见：KVEngine 同时承担完成句柄、传输计划、
生命周期善后与语义索引转发四类职责）。本模块是**零逻辑搬迁**：类的实现
逐字保留，仅调整导入位置。对外仍经 ``engine.core`` 再导出，既有调用点
（含测试）不受影响。
"""

from __future__ import annotations

from common.utils import flatten_block_ids as _flatten_block_ids


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

    __slots__ = (
        "_inner", "_after", "_before", "_after_event", "_event", "_done",
        "_block_tables", "_fence_event",
    )

    def __init__(self, inner, after, block_tables=None, after_event=None,
                 before=None, fence_event=None):
        """inner 为底层完成句柄，after 为无参收尾可调用。"""
        self._inner = inner
        self._after = after
        self._before = before
        self._fence_event = fence_event
        self._after_event = after_event
        self._event = None
        self._done = False
        self._block_tables = block_tables

    def wait(self) -> None:
        """阻塞至底层完成并执行收尾动作（恰一次）。"""
        if self._done:
            return
        try:
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
                # Legacy completions have no request-index detail; fail closed.
                self._inner.wait()
            if self._before is not None:
                self._before()
            self._event = self._after()
            if self._event is not None:
                if callable(self._after_event):
                    self._after_event(self._event)
                else:
                    wait = getattr(self._event, "wait", None)
                    if callable(wait):
                        wait()
        finally:
            self._done = True

    def poll(self) -> bool:
        """Non-blocking terminal probe with exactly-once post processing."""
        if self._done:
            return True
        if not self._inner.query():
            return False
        self.wait()
        return True

    @property
    def fence_event(self):
        return self._fence_event if self._fence_event is not None else self._event

    def abort(self, timeout=None) -> None:
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
        first_error = None
        for handle in self._handles:
            try:
                handle.wait()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self._done = True
        if first_error is not None:
            raise first_error

    def abort(self, timeout=None) -> None:
        """Drain child operations while suppressing all scatter callbacks."""
        if self._done:
            return
        for handle in self._handles:
            abort = getattr(handle, "abort", None)
            try:
                if callable(abort):
                    if timeout is None:
                        abort()
                    else:
                        try:
                            abort(timeout=timeout)
                        except TypeError:
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

    @property
    def fence_event(self):
        if not self._handles:
            return None
        return getattr(self._handles[-1], "fence_event", None)



"""Scheduler-side semantic index backed only by durable store metadata."""

from __future__ import annotations

import logging
from typing import Sequence

from tutti.common.utils import group_scan, positive_int
from tutti.index.chunk_index import ChunkIndex, StorePlan
_LOG = logging.getLogger(__name__)


class SchedulerMetadataIndex:
    """ChunkIndex plus marker reconciliation, with no data-plane methods."""

    def __init__(self, config: dict, store) -> None:
        self._chunk_tokens = positive_int(config, "chunk_tokens")
        self._chunk_kv_bytes = positive_int(config, "chunk_kv_bytes")
        self._max_chunks_per_wave = positive_int(
            config, "max_chunks_per_wave"
        )
        self._num_layers = positive_int(config, "num_layers")
        raw_namespace = config.get("key_namespace")
        if raw_namespace is None:
            namespace = b""
        elif isinstance(raw_namespace, str):
            namespace = raw_namespace.encode("utf-8")
        elif isinstance(raw_namespace, (bytes, bytearray)):
            namespace = bytes(raw_namespace)
        else:
            raise ValueError("config['key_namespace'] must be str/bytes/None")
        self._store = store
        setter = getattr(store, "set_key_namespace", None)
        if callable(setter) and namespace:
            setter(namespace)
        store.open()
        self._index = ChunkIndex(
            store.capacity_chunks, self._chunk_tokens, namespace=namespace
        )
        self._planned_store_keys: set[bytes] = set()
        self._synced_full: set[bytes] = set()
        self._pending_forget: set[bytes] = set()
        # 内存权威索引：worker 回传的 per-key rank 完成计数（跨步累计，
        # 达到 tp_size 即发布驻留）。
        self._rank_commit_counts: dict[bytes, int] = {}
        # 驱逐漂移观测计数（见 observe_worker_evictions）
        self._eviction_drift_total = 0
        self._eviction_drift_steps = 0
        # 超龄在途预留回收计数（见 begin_step）
        self._pending_reclaimed_total = 0
        self._closed = False
        # 冷启动恢复：进程启动时从盘上层标记重建一次内存索引；此后
        # 索引以内存为权威，靠 worker 回传增量推进，查询不再扫盘。
        self.sync_from_store()

    @property
    def capacity_chunks(self) -> int:
        return self._store.capacity_chunks

    def stats(self) -> dict[str, int]:
        """容量与漂移快照（可观测性：每步汇总日志）。"""
        snapshot = self._index.stats()
        snapshot.update({
            "rank_commit_counts": len(self._rank_commit_counts),
            "planned_store_keys": len(self._planned_store_keys),
            "eviction_drift_total": self._eviction_drift_total,
            "pending_reclaimed_total": self._pending_reclaimed_total,
        })
        return snapshot

    def lookup_prefix(self, token_ids: Sequence[int]) -> int:
        self._require_open()
        return self._index.lookup_prefix(token_ids)

    def hash_keys(
        self,
        token_ids: Sequence[int],
        start: int = 0,
        parent: bytes | None = None,
    ) -> tuple[list[bytes], bytes]:
        self._require_open()
        return self._index.hash_keys(token_ids, start, parent)

    def plan_store(self, keys) -> StorePlan | None:
        self._require_open()
        plan = self._index.plan_store(keys)
        if plan is None:
            return None
        self._planned_store_keys.update(plan.new_keys)
        # Scheduler owns only semantic admission. Every TP worker runs the
        # same store plan against its rank-local index and performs the
        # physical target close/object recycle in KVEngine.plan_store(). A
        # metadata-only process must never unlink worker-owned pool objects.
        return plan

    def confirm_store(self, keys, ok: bool = True) -> None:
        self._require_open()
        self._index.confirm_store(keys, ok)
        self._planned_store_keys.difference_update(keys)

    def begin_step(self, max_pending_age: int) -> int:
        """推进一个步进并回收超龄的在途预留，返回回收数量。

        由调度侧每步调用一次（build_connector_meta）。pending 项正常会
        在自身批次的 confirm_store 结算，但某个 TP rank 从未回报时
        （进程崩溃/重启、请求被抢占后不再 save）会永久占用容量：
        ``free = capacity - resident - pending`` 单调下降，最终写入被
        静默拒绝。这里按步龄兜底。

        max_pending_age 必须覆盖"TP rank 完成落在不同步"的正常窗口，
        否则会误回收正常预留（后果比容量泄漏更糟：confirm 落空 → 驻留
        不发布 → 重复重算）。
        """
        self._require_open()
        self._index.advance_epoch()
        reclaimed = self._index.reclaim_stale_pending(max_pending_age)
        if reclaimed:
            for key in reclaimed:
                self._rank_commit_counts.pop(key, None)
                self._planned_store_keys.discard(key)
            self._pending_reclaimed_total += len(reclaimed)
            _LOG.warning(
                "PENDING_RECLAIMED count=%d total=%d age>%d steps："
                "这些写入预留长期未结算（某 rank 未回报），已回收容量",
                len(reclaimed), self._pending_reclaimed_total,
                max_pending_age,
            )
        return len(reclaimed)

    def forget(self, keys) -> None:
        """把一批 key 移出调度侧索引（worker 判定不驻留时的对齐口）。

        用于自愈幽灵命中：调度侧与 worker 各自维护独立 LRU，驱逐集合
        可能不同；worker 在 pin 失败时回传缺失 key，这里对齐视图，使
        后续查询不再报假命中。不在驻留集合内的 key 是空操作（幂等）。

        只动"驻留视图"（_index）与跨步累计计数，**不动** _planned_store_keys：
        那是两阶段写入的容量预留，属于另一套状态；误清会让在途写入的
        结算落空。
        """
        self._require_open()
        batch = [bytes(key) for key in keys or ()]
        if not batch:
            return
        for key in batch:
            self._rank_commit_counts.pop(key, None)
        self._index.forget(batch)

    def observe_worker_evictions(self, evicted) -> int:
        """漂移观测：worker 已驱逐（并物理删除）的 key 若仍被本索引视为
        驻留，说明两侧 LRU 的牺牲者选择不一致，后续会表现为假命中。

        只统计与告警，**不改变任何决策**——用于在切换"调度侧单一决策源"
        之前先量化问题规模（切换本身有回归风险，需要数据支撑）。
        返回本步观测到的漂移 key 数。
        """
        self._require_open()
        drift = 0
        for key in evicted or ():
            key = bytes(key)
            if self._index.is_resident(key):
                drift += 1
        if drift:
            self._eviction_drift_total += drift
            self._eviction_drift_steps += 1
            # 按步告警会刷屏，这里做节流：每 64 步或首次全量打印一次。
            if (self._eviction_drift_steps == 1
                    or self._eviction_drift_steps % 64 == 0):
                _LOG.warning(
                    "INDEX_DRIFT_DETECTED step_drift=%d total_drift=%d "
                    "steps=%d：worker 已驱逐但调度侧仍视为驻留，"
                    "这些 key 会产生假命中并重复重算",
                    drift, self._eviction_drift_total,
                    self._eviction_drift_steps,
                )
        return drift

    def apply_worker_commits(self, committed, failed, tp_size: int,
                             forgotten=None, evicted=None) -> None:
        """应用 worker 回传的持久化增量（内存权威索引的唯一发布口）。

        committed 是 chunk key → 报告成功的 rank 数；只有全部 TP rank
        都报告成功且不在 failed 中，才发布驻留——与此前"盘上全 rank
        层标记齐备"同门禁，只是判定依据改为回传，不再扫盘。

        跨步累计：TP rank 的完成可能落在不同步（各 rank 步内结算，
        但 vLLM 聚合按步进行），故未达 tp_size 的计数留存待后续步
        补齐。failed 立即 fail-closed 回收预留并丢弃累计。
        forgotten 是 worker 认定不驻留的 key，直接对齐移除。
        """
        self._require_open()
        self.observe_worker_evictions(evicted)
        self.forget(forgotten)
        for key in failed or ():
            key = bytes(key)
            self._rank_commit_counts.pop(key, None)
            if key in self._planned_store_keys:
                self.confirm_store([key], ok=False)
        resident: list[bytes] = []
        for key, count in (committed or {}).items():
            key = bytes(key)
            if key in (failed or ()):
                continue
            total = self._rank_commit_counts.get(key, 0) + int(count)
            if total >= tp_size:
                self._rank_commit_counts.pop(key, None)
                resident.append(key)
            else:
                self._rank_commit_counts[key] = total
        if resident:
            self.confirm_store(resident, ok=True)

    def sync_from_store(self) -> None:
        """Rebuild the in-memory index from durable layer markers.

        Cold-start / crash-recovery only.  Steady-state residency is
        published by worker commit deltas (``apply_worker_commits``); this
        full scan must never run on the lookup hot path — it is O(markers)
        Python work inside the vLLM scheduler thread.
        """
        self._require_open()
        groups = group_scan(self._store)
        expected = set(range(self._num_layers))
        full_keys = [
            chunk_key
            for chunk_key, layers in groups.items()
            if layers >= expected
        ]
        full = set(full_keys)
        for key in list(self._planned_store_keys):
            self._index.confirm_store([key], ok=key in full)
            self._planned_store_keys.discard(key)
        stale = (self._synced_full - full) | self._pending_forget
        self._pending_forget = set(self._index.forget(stale))
        self._index.restore(full_keys)
        self._synced_full = full

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._store.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("scheduler metadata index is closed")



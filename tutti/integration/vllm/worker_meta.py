"""worker → scheduler 的索引增量（内存权威索引的发布通道）。

调度侧索引以内存为权威：写入受理（plan_store）只预留容量，真正
的驻留发布由 worker 在本步持久化成功后经 vLLM 既有的
``KVConnectorOutput.kv_connector_worker_meta`` 通道回传。盘上层标记
退化为冷启动/崩溃恢复的依据，不再参与查询热路径。

放在独立模块是为了让 ``adapter.connector``（调度侧）与
``adapter.worker``（worker 侧）都能导入而不形成循环。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMetadata,
)


@dataclass
class TuttiWorkerMetadata(KVConnectorWorkerMetadata):
    """单步内各 TP rank 的持久化结算增量。

    字段：
        committed：chunk key → 已成功持久化该 key 的 rank 数。单个
            rank 每步对同一 key 至多计一次（发布前按 rank 内集合去
            重），因而聚合后的计数就是"报告成功的 rank 数"。
        failed：本步 rank 本地结算失败的 key（任一 rank 失败即整体
            失败，fail-closed）。
        forgotten：worker 本地索引已判定**不驻留**、请求调度侧一并
            遗忘的 key。用于自愈"幽灵命中"——调度侧索引与 worker 索引
            是两套独立的 LRU（各自在 plan_store 里按自身迭代序驱逐），
            序不同则驱逐集合不同：worker 已驱逐并删除数据、调度侧仍
            认为驻留，于是调度侧每次请求都报命中、worker 每次 pin 都
            失败并重算，且调度侧索引没有移除通道 ⇒ 该 chunk 被永久
            重复重算（纯命中批次下还会因 span 清零而永不重写）。
            与 failed 分开是因为语义不同：failed 是"写失败、fail-closed
            回收预留"，forgotten 是"worker 认定缺失、请调度侧对齐视图"。
        evicted：worker 本步为腾容量驱逐（并已物理删除）的 key，**仅用于
            漂移观测**——调度侧比对自己是否仍视其驻留，命中则记
            INDEX_DRIFT_DETECTED。不参与任何决策，因此可以安全地只观测
            一段时间后再决定是否切换为"调度侧单一决策源"。

    调度侧只在 ``committed[key] == tp_size`` 且该 key 不在 ``failed``
    中时才发布驻留——与此前盘上"全 rank 层标记齐备"的门禁同语义，
    只是把判定依据从扫盘换成了回传。
    """

    committed: dict[bytes, int] = field(default_factory=dict)
    failed: set[bytes] = field(default_factory=set)
    forgotten: set[bytes] = field(default_factory=set)
    evicted: set[bytes] = field(default_factory=set)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        """跨 rank 合并（vLLM KVOutputAggregator 在步内调用）。"""
        assert isinstance(other, TuttiWorkerMetadata)
        merged = dict(self.committed)
        for key, count in other.committed.items():
            merged[key] = merged.get(key, 0) + count
        return TuttiWorkerMetadata(
            committed=merged,
            failed=self.failed | other.failed,
            forgotten=self.forgotten | other.forgotten,
            evicted=self.evicted | other.evicted,
        )

    def is_empty(self) -> bool:
        return (not self.committed and not self.failed
                and not self.forgotten and not self.evicted)

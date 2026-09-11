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

    调度侧只在 ``committed[key] == tp_size`` 且该 key 不在 ``failed``
    中时才发布驻留——与此前盘上"全 rank 层标记齐备"的门禁同语义，
    只是把判定依据从扫盘换成了回传。
    """

    committed: dict[bytes, int] = field(default_factory=dict)
    failed: set[bytes] = field(default_factory=set)

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
        )

    def is_empty(self) -> bool:
        return not self.committed and not self.failed

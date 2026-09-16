"""容量压力下的驻留一致性不变量。

P0-1（驱逐决策双份独立）的暴露条件是 `resident ≈ capacity` 的长期运行：
两侧索引各自按自身迭代序挑牺牲者，序不同则驱逐集合不同，于是出现
"幽灵命中"（worker 已删、调度侧仍报驻留）与"容量泄漏"（调度侧已删、
worker 仍持有）。这些测试把该场景压成可复现的索引级不变量。

两侧用的都是生产同一个 `ChunkIndex` 类；调度侧的访问序刷新走真实入口
`lookup_prefix`（命中即 move_to_end），这正是两侧序分叉的根源。

覆盖：
1. 容量不变量：resident + pending 不得超过 capacity；
2. 两侧收敛：调度侧下发的驱逐被 worker 执行后，worker 不再持有被驱逐项；
3. 漂移可检出且可自愈：worker 走兜底自行驱逐造成的假驻留，经 forgotten 对齐后消失；
4. 超龄在途预留不泄漏容量。
"""

from __future__ import annotations

import pytest

from tutti.index.chunk_index import ChunkIndex

CHUNK_TOKENS = 8


def _tokens(prefix: int, count: int) -> list[int]:
    return list(range(prefix * CHUNK_TOKENS,
                      (prefix + count) * CHUNK_TOKENS))


class _Pair:
    """调度侧 + worker 侧两套独立索引（与生产同构）。"""

    def __init__(self, capacity: int):
        self.scheduler = ChunkIndex(capacity, CHUNK_TOKENS)
        self.worker = ChunkIndex(capacity, CHUNK_TOKENS)

    def keys(self, prefix: int, count: int) -> list[bytes]:
        return self.scheduler.hash_keys(_tokens(prefix, count))[0]

    def hit(self, prefix: int, count: int) -> int:
        """模拟调度侧命中：刷新访问序（worker 侧不感知，故序会分叉）。"""
        return self.scheduler.lookup_prefix(_tokens(prefix, count))

    def write(self, keys, converge: bool = True):
        """一步写入：调度侧计划 → 下发驱逐 → worker 执行并计划 → 双确认。

        converge=False 模拟 worker 走"自行驱逐"的历史兜底路径（不执行
        调度侧决策），用于复现漂移。
        """
        sched_plan = self.scheduler.plan_store(keys)
        if sched_plan is None:
            return None
        if converge:
            for key in sched_plan.evicted_keys:
                if self.worker.is_resident(key):
                    self.worker.forget([key])
        worker_plan = self.worker.plan_store(keys)
        assert worker_plan is not None, "worker 必须能受理同一批写入"
        self.scheduler.confirm_store(keys, ok=True)
        self.worker.confirm_store(keys, ok=True)
        return sched_plan, worker_plan

    def assert_capacity_invariant(self):
        for index in (self.scheduler, self.worker):
            assert len(index._resident) + len(index._pending) <= index.capacity


def test_capacity_invariant_holds_under_pressure():
    """持续满容量写入：resident + pending 不得越界。"""
    pair = _Pair(capacity=4)
    for step in range(40):
        pair.write(pair.keys(step, 2))
        pair.assert_capacity_invariant()
    assert len(pair.scheduler._resident) <= 4


def test_scheduler_eviction_converges_worker():
    """调度侧选定的牺牲者被 worker 执行后，两侧驻留集合收敛。"""
    pair = _Pair(capacity=4)
    first = pair.keys(0, 4)
    assert pair.write(first) is not None

    # 命中刷新：调度侧把 first[0] 置为最近使用，两侧序从此不同
    assert pair.hit(0, 1) == CHUNK_TOKENS

    plans = pair.write(pair.keys(1, 2))
    assert plans is not None
    sched_plan, _worker_plan = plans
    assert len(sched_plan.evicted_keys) == 2
    for key in sched_plan.evicted_keys:
        assert not pair.worker.is_resident(key), "worker 必须执行调度侧决策"
        assert not pair.scheduler.is_resident(key)
    pair.assert_capacity_invariant()


def test_divergence_is_detectable_and_healable():
    """不执行下发决策（漂移场景）：假驻留可被检出并经 forgotten 自愈。"""
    pair = _Pair(capacity=4)
    first = pair.keys(0, 4)
    assert pair.write(first) is not None
    assert pair.hit(0, 1) == CHUNK_TOKENS

    plans = pair.write(pair.keys(1, 2), converge=False)
    assert plans is not None
    _sched_plan, worker_plan = plans

    ghost = [key for key in worker_plan.evicted_keys
             if pair.scheduler.is_resident(key)]
    assert ghost, "本用例前提：两侧按各自 LRU 选出的牺牲者不同"

    # 自愈：worker 回传 forgotten → 调度侧对齐视图
    pair.scheduler.forget(ghost)
    for key in ghost:
        assert not pair.scheduler.is_resident(key)


def test_stale_pending_does_not_leak_capacity():
    """某 rank 从未回报：超龄预留被回收，容量不泄漏。"""
    index = ChunkIndex(capacity=2, chunk_tokens=CHUNK_TOKENS)
    keys = index.hash_keys(_tokens(0, 2))[0]
    assert index.plan_store(keys) is not None
    # 在途占满容量：新批不受理
    assert index.plan_store(index.hash_keys(_tokens(1, 2))[0]) is None

    index.advance_epoch()
    index.advance_epoch()
    assert index.reclaim_stale_pending(max_age=3) == []

    index.advance_epoch()
    assert len(index.reclaim_stale_pending(max_age=3)) == 2
    # 回收后容量可用
    assert index.plan_store(index.hash_keys(_tokens(1, 2))[0]) is not None


@pytest.mark.parametrize("capacity", [1, 2, 8])
def test_no_leak_across_many_cycles(capacity):
    """多轮循环后两侧驻留数不得超过容量（长期不变量）。"""
    pair = _Pair(capacity=capacity)
    for step in range(30):
        pair.write(pair.keys(step, 1))
        if step % 3 == 0:
            pair.hit(step, 1)
        pair.assert_capacity_invariant()
    assert len(pair.scheduler._resident) <= capacity
    assert len(pair.worker._resident) <= capacity

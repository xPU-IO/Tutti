"""ChunkIndex 单元测试：链式哈希、前缀命中、两阶段写入、LRU 与 pin 保护。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from index.chunk_index import ChunkIndex, StorePlan

CT = 4  # 每个 chunk 的 token 数


def _tokens(n: int, base: int) -> list[int]:
    """生成 n 个 token id；不同 base 保证序列彼此不同。"""
    return [base * 1000 + i for i in range(n)]


def _keys(idx: ChunkIndex, base: int, n_chunks: int = 1) -> list[bytes]:
    """取 base 序列前 n_chunks 个完整 chunk 的 key。"""
    keys, _ = idx.hash_keys(_tokens(n_chunks * CT, base))
    return keys


class TestConstructor:
    def test_rejects_nonpositive_arguments(self):
        with pytest.raises(ValueError):
            ChunkIndex(0, CT)
        with pytest.raises(ValueError):
            ChunkIndex(4, 0)
        with pytest.raises(ValueError):
            ChunkIndex(-1, CT)
        with pytest.raises(ValueError):
            ChunkIndex(4, -1)


class TestHashKeys:
    def test_empty_and_short_input_yield_no_keys(self):
        idx = ChunkIndex(4, CT)
        assert idx.hash_keys([]) == ([], b"")
        assert idx.hash_keys(_tokens(CT - 1, 1)) == ([], b"")

    def test_full_chunks_and_tail_discard(self):
        idx = ChunkIndex(4, CT)
        toks = _tokens(2 * CT + 2, 1)  # 2 个完整 chunk + 2 个尾部 token
        keys, parent = idx.hash_keys(toks)
        assert len(keys) == 2
        assert all(isinstance(k, bytes) and len(k) == 16 for k in keys)
        assert parent == keys[-1]
        keys_again, parent_again = idx.hash_keys(toks)
        assert keys_again == keys and parent_again == parent

    def test_key_chain_depends_on_parent(self):
        idx = ChunkIndex(4, CT)
        first, _ = idx.hash_keys(_tokens(CT, 1))
        chained, _ = idx.hash_keys(_tokens(2 * CT, 1))
        # 同一段 token 换了前缀（parent）→ key 不同
        assert chained[0] == first[0]
        assert chained[1] != first[0]

    @pytest.mark.parametrize("m", [0, 1, CT, 2 * CT, 2 * CT + 3, 5 * CT])
    def test_incremental_matches_full(self, m):
        idx = ChunkIndex(8, CT)
        toks = _tokens(5 * CT + 2, 1)
        full_keys, full_parent = idx.hash_keys(toks)
        head_keys, head_parent = idx.hash_keys(toks[:m])
        consumed = len(head_keys) * CT
        tail_keys, tail_parent = idx.hash_keys(toks, start=consumed, parent=head_parent)
        assert head_keys + tail_keys == full_keys
        assert tail_parent == full_parent

    def test_negative_start_raises(self):
        idx = ChunkIndex(4, CT)
        with pytest.raises(ValueError):
            idx.hash_keys(_tokens(CT, 1), start=-1)


class TestLookupPrefix:
    def test_empty_index_misses_everything(self):
        idx = ChunkIndex(4, CT)
        assert idx.lookup_prefix(_tokens(3 * CT, 1)) == 0
        assert idx.lookup_prefix([]) == 0
        assert idx.lookup_prefix(_tokens(CT - 1, 1)) == 0

    def test_store_confirm_then_hit(self):
        idx = ChunkIndex(4, CT)
        toks = _tokens(3 * CT, 1)
        keys = _keys(idx, 1, 3)
        plan = idx.plan_store(keys)
        assert isinstance(plan, StorePlan)
        assert plan.new_keys == keys
        assert plan.evicted_keys == []
        # 在途 chunk 不参与命中
        assert idx.lookup_prefix(toks) == 0
        idx.confirm_store(keys)
        assert idx.lookup_prefix(toks) == 3 * CT
        # 前缀部分命中
        assert idx.lookup_prefix(toks[:2 * CT]) == 2 * CT
        # 第三个 chunk 分叉 → 命中止于前两个
        divergent = toks[:2 * CT] + _tokens(CT, 9)
        assert idx.lookup_prefix(divergent) == 2 * CT


class TestPlanStore:
    def test_evicts_oldest_unpinned(self):
        idx = ChunkIndex(3, CT)
        for base in (1, 2, 3):
            keys = _keys(idx, base)
            idx.plan_store(keys)
            idx.confirm_store(keys)
        plan = idx.plan_store(_keys(idx, 4))
        assert plan.new_keys == _keys(idx, 4)
        assert plan.evicted_keys == _keys(idx, 1)
        assert idx.lookup_prefix(_tokens(CT, 1)) == 0
        assert idx.lookup_prefix(_tokens(CT, 2)) == CT

    def test_all_pinned_returns_none_and_state_intact(self):
        idx = ChunkIndex(2, CT)
        for base in (1, 2):
            keys = _keys(idx, base)
            idx.plan_store(keys)
            idx.confirm_store(keys)
            idx.pin(keys)
        assert idx.plan_store(_keys(idx, 3)) is None
        # None 不改变状态：仍全部命中
        assert idx.lookup_prefix(_tokens(CT, 1)) == CT
        assert idx.lookup_prefix(_tokens(CT, 2)) == CT

    def test_partial_pin_limits_eviction(self):
        idx = ChunkIndex(3, CT)
        for base in (1, 2, 3):
            keys = _keys(idx, base)
            idx.plan_store(keys)
            idx.confirm_store(keys)
        idx.pin(_keys(idx, 2))
        idx.pin(_keys(idx, 3))
        # 可驱逐者只剩 base=1：单新 key 腾位受理成功
        plan = idx.plan_store(_keys(idx, 4))
        assert plan.evicted_keys == _keys(idx, 1)
        assert plan.new_keys == _keys(idx, 4)
        idx.confirm_store(_keys(idx, 4))
        idx.pin(_keys(idx, 4))
        # 驻留者全部受保护：要 2 个空位 → None
        assert idx.plan_store(_keys(idx, 5) + _keys(idx, 6)) is None
        # 要 1 个空位同样 → None
        assert idx.plan_store(_keys(idx, 5)) is None
        # 受保护者全部仍命中
        for base in (2, 3, 4):
            assert idx.lookup_prefix(_tokens(CT, base)) == CT

    def test_pending_reentrant_returns_none(self):
        idx = ChunkIndex(2, CT)
        ka, kb = _keys(idx, 1), _keys(idx, 2)
        assert idx.plan_store(ka) is not None
        # 重复计划在途 key（单独或混批）→ 整批不受理
        assert idx.plan_store(ka) is None
        assert idx.plan_store(ka + kb) is None
        idx.confirm_store(ka)
        # 结算后再计划：已驻留 → 空计划（无可写内容、无驱逐）
        plan = idx.plan_store(ka)
        assert plan.new_keys == [] and plan.evicted_keys == []

    def test_pending_holds_capacity_and_is_never_evicted(self):
        idx = ChunkIndex(2, CT)
        ka, kb, kc = _keys(idx, 1), _keys(idx, 2), _keys(idx, 3)
        idx.plan_store(ka)
        idx.plan_store(kb)
        # 容量被两个在途 key 占满，无驻留者可驱逐 → None
        assert idx.plan_store(kc) is None
        idx.confirm_store(ka)
        # 在途 kb 不参与驱逐：腾位来自驻留者 ka
        plan = idx.plan_store(kc)
        assert plan.evicted_keys == ka
        assert kb[0] not in plan.evicted_keys
        idx.confirm_store(kb)
        idx.confirm_store(kc)
        assert idx.lookup_prefix(_tokens(CT, 2)) == CT
        assert idx.lookup_prefix(_tokens(CT, 3)) == CT

    def test_confirm_store_failure_recycles(self):
        idx = ChunkIndex(2, CT)
        ka = _keys(idx, 1)
        idx.plan_store(ka)
        idx.confirm_store(ka, ok=False)
        assert idx.lookup_prefix(_tokens(CT, 1)) == 0
        # 容量已回收：新 key 无需驱逐即可受理
        plan = idx.plan_store(_keys(idx, 2))
        assert plan.evicted_keys == []

    def test_empty_batch_and_duplicate_keys(self):
        idx = ChunkIndex(2, CT)
        plan = idx.plan_store([])
        assert plan.new_keys == [] and plan.evicted_keys == []
        ka = _keys(idx, 1)
        plan = idx.plan_store(ka + ka)
        assert plan.new_keys == ka


class TestPin:
    def test_pin_miss_raises_and_is_atomic(self):
        idx = ChunkIndex(2, CT)
        ka = _keys(idx, 1)
        idx.plan_store(ka)
        idx.confirm_store(ka)
        with pytest.raises(KeyError):
            idx.pin(ka + [b"\x00" * 16])
        # 整批不生效：ka 未被 pin（对其 unpin 应报 KeyError）
        with pytest.raises(KeyError):
            idx.unpin(ka)

    def test_unpin_without_pin_raises(self):
        idx = ChunkIndex(2, CT)
        ka = _keys(idx, 1)
        idx.plan_store(ka)
        idx.confirm_store(ka)
        with pytest.raises(KeyError):
            idx.unpin(ka)

    def test_unpin_makes_key_evictable_again(self):
        idx = ChunkIndex(1, CT)
        ka, kb = _keys(idx, 1), _keys(idx, 2)
        idx.plan_store(ka)
        idx.confirm_store(ka)
        idx.pin(ka)
        assert idx.plan_store(kb) is None
        idx.unpin(ka)
        plan = idx.plan_store(kb)
        assert plan.evicted_keys == ka

    def test_pin_counts_pair_up(self):
        idx = ChunkIndex(1, CT)
        ka, kb = _keys(idx, 1), _keys(idx, 2)
        idx.plan_store(ka)
        idx.confirm_store(ka)
        idx.pin(ka)
        idx.pin(ka)
        idx.unpin(ka)
        # 还剩一层 pin：仍不可驱逐
        assert idx.plan_store(kb) is None
        idx.unpin(ka)
        assert idx.plan_store(kb).evicted_keys == ka


class TestLru:
    def test_mark_recent_changes_eviction_order(self):
        idx = ChunkIndex(3, CT)
        for base in (1, 2, 3):
            keys = _keys(idx, base)
            idx.plan_store(keys)
            idx.confirm_store(keys)
        idx.mark_recent(_keys(idx, 1))  # 最旧者被刷新
        plan = idx.plan_store(_keys(idx, 4))
        assert plan.evicted_keys == _keys(idx, 2)
        # 非驻留 key 静默忽略
        idx.mark_recent([b"\xff" * 16])

    def test_confirm_refreshes_recency(self):
        idx = ChunkIndex(2, CT)
        ka, kb = _keys(idx, 1), _keys(idx, 2)
        for keys in (ka, kb):
            idx.plan_store(keys)
            idx.confirm_store(keys)
        # 重新写入并确认 ka → ka 变为最近使用
        idx.plan_store(ka)
        idx.confirm_store(ka)
        plan = idx.plan_store(_keys(idx, 3))
        assert plan.evicted_keys == kb


class TestRestore:
    def test_restore_is_idempotent(self):
        idx = ChunkIndex(4, CT)
        keys = _keys(idx, 1, 3)
        toks = _tokens(3 * CT, 1)
        idx.restore(keys)
        assert idx.lookup_prefix(toks) == 3 * CT
        idx.restore(keys)
        idx.restore(keys)
        assert idx.lookup_prefix(toks) == 3 * CT
        # 重复灌入不重复占容量：4 - 3 = 1 个空位，无需驱逐
        plan = idx.plan_store(_keys(idx, 2))
        assert plan.evicted_keys == []

    def test_restore_order_sets_initial_lru(self):
        idx = ChunkIndex(3, CT)
        idx.restore(_keys(idx, 1) + _keys(idx, 2) + _keys(idx, 3))
        plan = idx.plan_store(_keys(idx, 4))
        # 排前者更旧：先驱逐 base=1
        assert plan.evicted_keys == _keys(idx, 1)


class TestIsolation:
    def test_import_pulls_no_heavy_dependencies(self):
        """子进程断言：import 本模块后 sys.modules 无 vllm/torch/numpy。"""
        connector_root = Path(__file__).resolve().parents[2]
        code = (
            "import sys\n"
            "import index.chunk_index\n"
            "leaked = {'vllm', 'torch', 'numpy'} & set(sys.modules)\n"
            "assert not leaked, f'unexpected modules: {sorted(leaked)}'\n"
            "print('clean')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=connector_root,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "clean" in proc.stdout

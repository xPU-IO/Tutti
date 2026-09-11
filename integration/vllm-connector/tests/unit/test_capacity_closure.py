"""容量闭环：权威淘汰（物理执行）+ hit 刷新 + 镜像完整性修正。

Run:
    cd integration/vllm-connector && \
    /data/home/ryeqiu/tutti-env/bin/python -m pytest tests/unit/test_capacity_closure.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from engine.core import KVEngine
from engine.staging import RingWindow
from index.chunk_index import ChunkIndex, derive_io_key
from stores.memory import MemoryKVStore

SEG = 4096
NL = 3
CHUNK_TOKENS = 4
MAX_WAVE = 8
SLOTS = 2 * MAX_WAVE


class MovingHooks:
    """真实搬运钩子：paged 侧以 dict 模拟（与 contract harness 同构）。"""

    def __init__(self, buffer=None, seg: int = SEG):
        self._view = memoryview(buffer) if buffer is not None else None
        self._seg = seg
        self.source: dict[tuple[bytes, int], bytes] = {}

    def gather(self, keys, layer_idx, first_blocks, slots):
        for k, s in zip(keys, slots):
            self._view[s * self._seg:(s + 1) * self._seg] = self.source[(k, layer_idx)]


def _tokens(base: int, n_chunks: int) -> list[int]:
    return [base * 1000 + i for i in range(n_chunks * CHUNK_TOKENS)]


def _mk(capacity: int, num_layers: int | None = None):
    """构造已 bind 引擎；返回 (engine, store, hooks)。"""
    store = MemoryKVStore(SEG, num_chunks=capacity)
    hooks = MovingHooks()
    config = {
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_kv_bytes": SEG * NL,
        "max_chunks_per_wave": MAX_WAVE,
        "gather_fn": hooks.gather,
    }
    if num_layers is not None:
        config["num_layers"] = num_layers
    engine = KVEngine(config, store)
    buffer = bytearray(SLOTS * SEG)
    hooks._view = memoryview(buffer)
    window = RingWindow(buffer, SLOTS, SEG)
    engine.bind({}, window, NL, blocks_per_chunk=2)
    return engine, store, hooks


def _write(engine, hooks, base: int, n_chunks: int) -> list[bytes]:
    """全层写入 base 序列前 n_chunks 个 chunk 并结算（权威写入路径）。"""
    keys, _ = engine.hash_keys(_tokens(base, n_chunks))
    engine.plan_store(keys)
    for layer in range(NL):
        for i, k in enumerate(keys):
            hooks.source[(k, layer)] = bytes([(base + i + layer) & 0xFF]) * SEG
        engine.store_layer(keys, layer, [0] * len(keys)).wait()
    engine.confirm_store(keys, ok=True)
    return keys


class TestAuthoritativeEviction:
    """worker 权威淘汰：驱逐集按全层 io_key 物理删除，LRU 顺序正确。"""

    def test_eviction_removes_files_and_index(self):
        engine, store, hooks = _mk(capacity=4)
        a = _write(engine, hooks, 1, 2)
        _write(engine, hooks, 2, 2)   # 容量 4 恰满
        c = _write(engine, hooks, 3, 2)  # 驱逐最旧：a 的两个 chunk
        live = set(store.scan())
        for k in a:  # 被汰 chunk 全层文件消失
            assert {derive_io_key(k, l) for l in range(NL)}.isdisjoint(live)
        assert {derive_io_key(k, l) for k in c for l in range(NL)} <= live
        # 索引无残留
        assert engine.lookup_prefix(_tokens(1, 2)) == 0
        assert engine.lookup_prefix(_tokens(2, 2)) == 2 * CHUNK_TOKENS
        assert engine.lookup_prefix(_tokens(3, 2)) == 2 * CHUNK_TOKENS

    def test_lru_order_and_hit_refresh(self):
        """先写先汰；命中刷新者存活。"""
        engine, store, hooks = _mk(capacity=4)
        _write(engine, hooks, 1, 1)  # A
        _write(engine, hooks, 2, 1)  # B
        _write(engine, hooks, 3, 1)  # C
        _write(engine, hooks, 4, 1)  # D（满）
        assert engine.lookup_prefix(_tokens(1, 1)) == CHUNK_TOKENS  # A 命中即刷新
        _write(engine, hooks, 5, 1)  # E → 驱逐 B（A 已被刷新）
        assert engine.lookup_prefix(_tokens(1, 1)) == CHUNK_TOKENS  # A 存活
        assert engine.lookup_prefix(_tokens(2, 1)) == 0             # B 被汰
        assert engine.lookup_prefix(_tokens(5, 1)) == CHUNK_TOKENS

    def test_pin_protects_from_eviction(self):
        """pin 中的 key 不被驱逐；全部受保护时计划不受理。"""
        engine, store, hooks = _mk(capacity=2)
        _write(engine, hooks, 1, 1)
        _write(engine, hooks, 2, 1)   # 满
        engine.pin(engine.hash_keys(_tokens(1, 1))[0])
        engine.pin(engine.hash_keys(_tokens(2, 1))[0])
        # 全部 pin：无可驱逐 → 计划 None，状态不变
        assert engine.plan_store(engine.hash_keys(_tokens(3, 1))[0]) is None
        assert engine.lookup_prefix(_tokens(1, 1)) == CHUNK_TOKENS  # 原状保留


class TestMirrorIntegrity:
    """scheduler 镜像完整化：num_layers 预告 + 完整性翻转修正。"""

    def _mirror(self, capacity: int = 8):
        """未 bind 的镜像侧引擎（num_layers 预告生效）。"""
        store = MemoryKVStore(SEG, num_chunks=capacity)
        engine = KVEngine(
            {
                "chunk_tokens": CHUNK_TOKENS,
                "chunk_kv_bytes": SEG * NL,
                "max_chunks_per_wave": MAX_WAVE,
                "num_layers": NL,
            },
            store,
        )
        return engine, store

    def test_partial_layer_group_is_miss(self):
        """残缺层组判不完整（miss），盘上部分层不驻留镜像。"""
        engine, store = self._mirror()
        # 直接经 store 写两层（残缺）——镜像 sync 应判 miss
        src = bytearray(SEG)
        src_id = store.register_buffer(src, SEG)
        keys, _ = engine.hash_keys(_tokens(1, 1))
        store.put_batch([  # 只写 NL-1 层
            (derive_io_key(keys[0], layer), src_id, 0)
            for layer in range(NL - 1)
        ]).wait()
        engine.sync_from_store()
        assert engine.lookup_prefix(_tokens(1, 1)) == 0

    def test_completeness_flip_removes_stale_mirror(self):
        """完整性翻转修正：盘上完整层组消失 → 移除近似项。"""
        engine, store = self._mirror()
        src = bytearray(SEG)
        src_id = store.register_buffer(src, SEG)
        keys, _ = engine.hash_keys(_tokens(1, 1))
        store.put_batch([  # 全层
            (derive_io_key(keys[0], layer), src_id, 0) for layer in range(NL)
        ]).wait()
        engine.sync_from_store()
        assert engine.lookup_prefix(_tokens(1, 1)) == CHUNK_TOKENS

        store.drop([derive_io_key(keys[0], l) for l in range(NL)])
        engine.sync_from_store()
        assert engine.lookup_prefix(_tokens(1, 1)) == 0  # 近似项被移除

    def test_pinned_stale_entry_survives_mirror(self):
        """pin 中的过期近似项不移除（unpin 配对保护）。"""
        engine, store = self._mirror()
        src = bytearray(SEG)
        src_id = store.register_buffer(src, SEG)
        keys, _ = engine.hash_keys(_tokens(1, 1))
        store.put_batch([
            (derive_io_key(keys[0], layer), src_id, 0) for layer in range(NL)
        ]).wait()
        engine.sync_from_store()
        engine.pin(keys)
        store.drop([derive_io_key(keys[0], l) for l in range(NL)])
        engine.sync_from_store()
        # 仍在近似索引（pin 保护）；读时由 store miss 降级兜底
        assert engine.lookup_prefix(_tokens(1, 1)) == CHUNK_TOKENS
        engine.unpin(keys)
        engine.sync_from_store()
        assert engine.lookup_prefix(_tokens(1, 1)) == 0

    def test_bind_validates_layers_hint(self):
        """bind 层数与构造预告不一致 → ValueError（fail-fast）。"""
        engine, store = self._mirror()
        window = RingWindow(bytearray(SLOTS * SEG), SLOTS, SEG)
        with pytest.raises(ValueError, match="不一致"):
            engine.bind({}, window, NL + 1, 1)


class TestReconcileIndex:
    """ChunkIndex.forget 的纯逻辑契约（完整性翻转移除）。"""

    def test_forget_removes_unpinned(self):
        idx = ChunkIndex(capacity=4, chunk_tokens=CHUNK_TOKENS)
        k1, k2 = (bytes([i]) * 16 for i in (1, 2))
        idx.restore([k1, k2])
        idx.forget([k1])
        assert set(idx._resident) == {k2}

    def test_forget_keeps_pinned(self):
        idx = ChunkIndex(capacity=4, chunk_tokens=CHUNK_TOKENS)
        k1, k2 = (bytes([i]) * 16 for i in (1, 2))
        idx.restore([k1, k2])
        idx.pin([k1])
        idx.forget([k1])         # pin 保护：跳过移除
        assert set(idx._resident) == {k1, k2}

    def test_forget_ignores_absent(self):
        idx = ChunkIndex(capacity=4, chunk_tokens=CHUNK_TOKENS)
        idx.forget([bytes([9]) * 16])  # 非驻留静默忽略
        assert not idx._resident

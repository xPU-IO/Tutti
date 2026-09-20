"""key 命名空间：ns 前缀入链式哈希 + 池归属 manifest + v2 版本头。

Run:
    cd <repo-root> && \
    /data/home/ryeqiu/tutti-env/bin/python -m pytest tests/unit/test_key_namespace.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutti.engine.core import KVEngine
from tutti.index.chunk_index import ChunkIndex
from tutti.storage.memory import MemoryKVStore

SEG = 4096
NL = 3
CHUNK_TOKENS = 8

NS_A = b"v2|model=hy3|dtype=bfloat16|tp=1|chunk_kv_bytes=12288|chunk_tokens=8"
NS_B_MODEL = b"v2|model=other|dtype=bfloat16|tp=1|chunk_kv_bytes=12288|chunk_tokens=8"
NS_B_GEOM = b"v2|model=hy3|dtype=bfloat16|tp=1|chunk_kv_bytes=20971520|chunk_tokens=256"
NS_V1 = b"v1|model=hy3|dtype=bfloat16|tp=1|chunk_kv_bytes=12288|chunk_tokens=8"

TOKENS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444, 555, 666, 777]


def _idx(namespace: bytes | None = None) -> ChunkIndex:
    kw = {"namespace": namespace} if namespace is not None else {}
    return ChunkIndex(capacity=8, chunk_tokens=CHUNK_TOKENS, **kw)


class TestNamespaceKeyChain:
    """命名空间入链式哈希：同 ns 跨实例一致、异 ns 隔离。"""

    def test_same_namespace_same_keys(self):
        """同 ns 的两个独立实例（模拟跨进程）派生相同 key 链。"""
        a = _idx(NS_A)
        b = _idx(NS_A)
        ka, pa = a.hash_keys(TOKENS)
        kb, pb = b.hash_keys(TOKENS)
        assert ka == kb
        assert pa == pb
        assert len(ka) == 2  # 16 tokens / CT=8

    def test_different_namespaces_isolated(self):
        """模型不同 / 几何不同 / 版本不同 → key 全不同。"""
        base = _idx(NS_A).hash_keys(TOKENS)[0]
        for ns in (NS_B_MODEL, NS_B_GEOM, NS_V1):
            other = _idx(ns).hash_keys(TOKENS)[0]
            assert other != base, f"{ns!r} 未隔离"

    def test_empty_namespace_backward_compatible(self):
        """空 ns（缺省）与既有无 ns 行为一致：首 parent 为空串。"""
        idx = _idx()
        keys, _ = idx.hash_keys(TOKENS)
        legacy, _ = _idx(b"").hash_keys(TOKENS)
        assert keys == legacy

    def test_lookup_respects_namespace(self):
        """驻留判定按 ns：A 驻留的 key 对 B 不命中。"""
        a = _idx(NS_A)
        ka, _ = a.hash_keys(TOKENS)
        a.restore(ka)
        assert a.lookup_prefix(TOKENS) == 2 * CHUNK_TOKENS
        b = _idx(NS_B_MODEL)
        assert b.lookup_prefix(TOKENS) == 0  # B 视角全 miss

    def test_namespace_property_roundtrip(self):
        assert _idx(NS_A).namespace == NS_A


class TestEngineNamespace:
    """engine 可选 key_namespace 注入（config 契约）。"""

    def _mk(self, namespace=None):
        store = MemoryKVStore(SEG, num_chunks=8)
        config = {
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_kv_bytes": SEG * NL,
            "max_chunks_per_wave": 8,
        }
        if namespace is not None:
            config["key_namespace"] = namespace
        return KVEngine(config, store), store

    def test_engine_keys_namespaced(self):
        ea, _ = self._mk("v2|model=m|tp=1")
        eb, _ = self._mk("v2|model=other|tp=1")
        ka, _ = ea.hash_keys(TOKENS)
        kb, _ = eb.hash_keys(TOKENS)
        assert ka != kb
        # 同 ns 两个 engine（同进程共享语义）一致
        ec, _ = self._mk("v2|model=m|tp=1")
        assert ec.hash_keys(TOKENS)[0] == ka

    def test_engine_default_no_namespace(self):
        e, _ = self._mk()
        keys, _ = e.hash_keys(TOKENS)
        legacy, _ = _idx().hash_keys(TOKENS)
        assert keys == legacy

    def test_invalid_namespace_type_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="key_namespace"):
            self._mk(12345)


class TestNamespacePlumbing:
    """命名空间（对象身份）进对象层配置：时机与取值。

    Python 侧不再有 namespace.manifest：身份写进每个对象的对象头，由对象层
    在 lookup/读取时逐对象校验（C++ 契约测试覆盖）。这里只钉住 Python 到
    对象层的管道：命名空间必须进配置、且只能在打开对象层之前声明。
    """

    @staticmethod
    def _layout(root, namespace=None):
        from tutti.storage.tutti_nvme.object_layout import ObjectLayout

        layout = ObjectLayout(
            root, segment_bytes=SEG, capacity_chunks=8, prewarm_chunks=1,
            namespace=namespace,
        )
        return layout

    def test_namespace_reaches_object_layer_config(self, tmp_path):
        layout = self._layout(tmp_path / "pool", NS_A)
        layout.set_layer_span(NL)
        assert layout._config()["namespace_fingerprint"] == NS_A
        layout.close_object_pool()

    def test_default_namespace_is_empty(self, tmp_path):
        layout = self._layout(tmp_path / "pool")
        layout.set_layer_span(NL)
        assert layout._config()["namespace_fingerprint"] == b""
        layout.close_object_pool()

    def test_namespace_declared_after_open_is_rejected(self, tmp_path):
        import pytest

        layout = self._layout(tmp_path / "pool")
        layout.set_layer_span(NL)          # 打开对象层（几何定案）
        with pytest.raises(RuntimeError, match="命名空间"):
            layout.set_namespace(NS_A)
        layout.close_object_pool()

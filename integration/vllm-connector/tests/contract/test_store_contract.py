"""KVStore SPI 契约套件。

所有实现经 STORE_FACTORIES 注册后跑同一组断言：生命周期、批往返、
多 key 混合批、未知 key、容量、scan、drop、粒度拒绝。
实现私有的扩展契约写在各自的专项测试类里，不进共享套件。
"""

import ctypes
from collections import namedtuple

import pytest

from stores.memory import MemoryKVStore
from stores.registry import create_store, register_store_type

StoreSpec = namedtuple(
    "StoreSpec", ["make_store", "segment_bytes", "granularity", "capacity_chunks"]
)

# 契约套件工厂注册表：name -> StoreSpec。
# 新实现在此注册后自动进入下方全部参数化用例。
STORE_FACTORIES: dict[str, StoreSpec] = {}


def register_store_factory(
    name: str,
    make_store,
    *,
    segment_bytes: int,
    granularity: int,
    capacity_chunks: int,
) -> None:
    """注册契约套件工厂。

    make_store：零参调用，返回未 open 的新实例；
    segment_bytes / granularity / capacity_chunks：该实例契约参数
    （granularity 须是该实现可接受的注册粒度）。
    """
    STORE_FACTORIES[name] = StoreSpec(
        make_store, segment_bytes, granularity, capacity_chunks
    )


register_store_factory(
    "memory",
    lambda: MemoryKVStore(segment_bytes=4096, num_chunks=8),
    segment_bytes=4096,
    granularity=4096,
    capacity_chunks=8,
)

# tutti_nvme 双实现注册（FakeRuntime 支架与本文件同目录）。
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from tutti_nvme_factory import register_factory as _register_tutti_nvme  # noqa: E402

_register_tutti_nvme(register_store_factory)


def _store_params():
    return [pytest.param(spec, id=name) for name, spec in sorted(STORE_FACTORIES.items())]


@pytest.fixture(params=_store_params())
def spec(request):
    return request.param


@pytest.fixture
def store(spec):
    instance = spec.make_store()
    yield instance
    instance.close()


# ---------- 共享契约（所有实现同跑） ----------


def test_data_ops_require_open(store, spec):
    """open 之前数据面操作一律 RuntimeError。"""
    buf = bytearray(4 * spec.segment_bytes)
    with pytest.raises(RuntimeError):
        store.register_buffer(buf, spec.granularity)
    with pytest.raises(RuntimeError):
        store.put_batch([(b"k", 0, 0)])
    with pytest.raises(RuntimeError):
        store.get_batch([(b"k", 0, 0)])
    with pytest.raises(RuntimeError):
        store.drop([b"k"])
    with pytest.raises(RuntimeError):
        list(store.scan())


def test_close_is_idempotent(store):
    """close 二次调用不抛异常。"""
    store.open()
    store.close()
    store.close()


def test_capacity_chunks(store, spec):
    assert store.capacity_chunks == spec.capacity_chunks


def test_scan_initially_empty(store):
    store.open()
    assert list(store.scan()) == []


def test_batch_roundtrip(store, spec):
    """写后读回数据一致；Completion 可 wait 且完成后 query 为真。"""
    store.open()
    seg = spec.segment_bytes
    src = bytearray(4 * seg)
    dst = bytearray(4 * seg)
    src_id = store.register_buffer(src, spec.granularity)
    dst_id = store.register_buffer(dst, spec.granularity)
    assert isinstance(src_id, int)

    keys = [b"roundtrip-a", b"roundtrip-b", b"roundtrip-c"]
    offsets = [0, seg, 3 * seg]
    patterns = []
    for idx, (key, off) in enumerate(zip(keys, offsets)):
        pattern = bytes([0x11 * (idx + 1) & 0xFF]) * seg
        patterns.append(pattern)
        src[off : off + seg] = pattern

    put = store.put_batch(list(zip(keys, [src_id] * len(keys), offsets)))
    put.wait()
    assert put.query()

    get = store.get_batch(list(zip(keys, [dst_id] * len(keys), offsets)))
    get.wait()
    assert get.query()
    for pattern, off in zip(patterns, offsets):
        assert bytes(dst[off : off + seg]) == pattern


def test_multi_key_mixed_batch(store, spec):
    """多 key 混批：来源 offset 各异，读回重排到不同 offset 仍一一对应。"""
    store.open()
    seg = spec.segment_bytes
    src = bytearray(3 * seg)
    dst = bytearray(3 * seg)
    src_id = store.register_buffer(src, spec.granularity)
    dst_id = store.register_buffer(dst, spec.granularity)
    keys = [b"mix-0", b"mix-1", b"mix-2"]
    src_offsets = [0, seg, 2 * seg]
    dst_offsets = [2 * seg, 0, seg]
    for idx, off in enumerate(src_offsets):
        src[off : off + seg] = bytes([0x40 + idx]) * seg

    store.put_batch(list(zip(keys, [src_id] * 3, src_offsets))).wait()
    store.get_batch(list(zip(keys, [dst_id] * 3, dst_offsets))).wait()

    for idx, off in enumerate(dst_offsets):
        assert bytes(dst[off : off + seg]) == bytes([0x40 + idx]) * seg


def test_get_unknown_key_raises(store, spec):
    """get_batch 遇未驻留 key → ValueError。"""
    store.open()
    buf = bytearray(spec.segment_bytes)
    buf_id = store.register_buffer(buf, spec.granularity)
    with pytest.raises(ValueError):
        store.get_batch([(b"never-stored", buf_id, 0)])


def test_drop_removes_from_scan(store, spec):
    """put 后 scan 可见，drop 后从 scan 消失。"""
    store.open()
    seg = spec.segment_bytes
    buf = bytearray(2 * seg)
    buf_id = store.register_buffer(buf, spec.granularity)
    keys = [b"drop-0", b"drop-1"]
    store.put_batch([(keys[0], buf_id, 0), (keys[1], buf_id, seg)]).wait()
    assert sorted(store.scan()) == sorted(keys)
    store.drop([keys[0]])
    assert list(store.scan()) == [keys[1]]


def test_register_buffer_rejects_unaligned_granularity(store, spec):
    """非 4096 对齐的 granularity → None。"""
    store.open()
    buf = bytearray(2 * spec.segment_bytes)
    assert store.register_buffer(buf, spec.granularity + 1) is None


# ---------- MemoryKVStore 私有契约 ----------


class TestMemoryKVStorePrivateContract:
    """MemoryKVStore 文档化的私有行为（不要求其他实现跟随）。"""

    SEG = 4096

    def _open_store(self) -> MemoryKVStore:
        store = MemoryKVStore(segment_bytes=self.SEG, num_chunks=4)
        store.open()
        return store

    def test_register_buffer_rejects_oversized_granularity(self):
        """4096 对齐但大于 segment_bytes 的 granularity → None。"""
        store = self._open_store()
        buf = bytearray(2 * self.SEG)
        assert store.register_buffer(buf, 2 * self.SEG) is None

    def test_register_buffer_rejects_non_writable(self):
        """只读或非缓冲对象 → None。"""
        store = self._open_store()
        assert store.register_buffer(b"read-only", self.SEG) is None
        assert store.register_buffer(12345, self.SEG) is None

    def test_unregistered_buffer_raises(self):
        store = self._open_store()
        buf = bytearray(self.SEG)
        buf_id = store.register_buffer(buf, self.SEG)
        with pytest.raises(ValueError):
            store.put_batch([(b"k", buf_id + 1000, 0)])
        store.put_batch([(b"k", buf_id, 0)]).wait()
        with pytest.raises(ValueError):
            store.get_batch([(b"k", buf_id + 1000, 0)])

    def test_offset_out_of_range_raises(self):
        store = self._open_store()
        buf = bytearray(2 * self.SEG)
        buf_id = store.register_buffer(buf, self.SEG)
        with pytest.raises(ValueError):
            store.put_batch([(b"k", buf_id, 2 * self.SEG - 1)])  # 尾部不完整
        with pytest.raises(ValueError):
            store.put_batch([(b"k", buf_id, -self.SEG)])  # 负偏移
        store.put_batch([(b"k", buf_id, self.SEG)]).wait()  # 恰好末段合法
        with pytest.raises(ValueError):
            store.get_batch([(b"k", buf_id, self.SEG + 1)])

    def test_drop_missing_key_is_silent(self):
        store = self._open_store()
        store.drop([b"never-existed"])  # 驱逐幂等：未驻留 key 静默忽略

    def test_put_overwrites_existing_key(self):
        store = self._open_store()
        src = bytearray(self.SEG)
        dst = bytearray(self.SEG)
        src_id = store.register_buffer(src, self.SEG)
        dst_id = store.register_buffer(dst, self.SEG)
        src[:] = b"\x01" * self.SEG
        store.put_batch([(b"k", src_id, 0)]).wait()
        src[:] = b"\x02" * self.SEG
        store.put_batch([(b"k", src_id, 0)]).wait()
        store.get_batch([(b"k", dst_id, 0)]).wait()
        assert bytes(dst) == b"\x02" * self.SEG

    def test_ctypes_buffer_supported(self):
        store = self._open_store()
        src = (ctypes.c_char * self.SEG)()
        dst = bytearray(self.SEG)
        src_id = store.register_buffer(src, self.SEG)
        assert isinstance(src_id, int)
        dst_id = store.register_buffer(dst, self.SEG)
        ctypes.memset(src, 0xAB, self.SEG)
        store.put_batch([(b"ct", src_id, 0)]).wait()
        store.get_batch([(b"ct", dst_id, 0)]).wait()
        assert bytes(dst) == b"\xAB" * self.SEG

    def test_constructor_rejects_bad_args(self):
        with pytest.raises(ValueError):
            MemoryKVStore(segment_bytes=0, num_chunks=4)
        with pytest.raises(ValueError):
            MemoryKVStore(segment_bytes=4096, num_chunks=-1)

    def test_close_releases_state(self):
        store = self._open_store()
        buf = bytearray(self.SEG)
        buf_id = store.register_buffer(buf, self.SEG)
        store.put_batch([(b"k", buf_id, 0)]).wait()
        store.close()
        with pytest.raises(RuntimeError):
            store.get_batch([(b"k", buf_id, 0)])
        store.open()  # 重开后为空实例
        assert list(store.scan()) == []


# ---------- 注册表 ----------


class TestRegistry:
    """注册表：type 选择与 options 透传。"""

    def test_create_memory_store(self):
        store = create_store("memory", {"segment_bytes": 4096, "num_chunks": 8})
        assert isinstance(store, MemoryKVStore)
        assert store.capacity_chunks == 8
        store.close()

    def test_options_passed_through(self):
        store = create_store("memory", {"segment_bytes": 8192, "num_chunks": 2})
        assert store.capacity_chunks == 2
        store.close()

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            create_store("no-such-type", {})

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError):
            register_store_type("memory", MemoryKVStore)

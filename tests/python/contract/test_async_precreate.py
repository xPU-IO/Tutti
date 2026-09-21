"""异步容量增长契约：预建不在请求路径上，增长由后台驱动。

定案（2026-09-21，10 TB 级冷启动）：
  1. open() 只**走查**（probe，不写）调用方声明的初始槽位：盘上已有的登记
     为就绪（复用零 IO），缺的留给后台——大容量不再意味着几十分钟启动；
  2. 写路径永不 create+fsync 槽位（~44ms/槽会直接打在 GPU 计算/IO 流水线
     上）：拿不到就绪槽位时与容量耗尽同一契约——拒绝，由调用方裁剪；
  3. 增长由后台 ``precreate_step`` 驱动，默认按容量补齐差量（容量比已有
     的多多少就异步补多少），不超过容量上限；
  4. 后台失败时把按需预建交还写路径：宁慢，不静默停摆。

复用与容量因此合在同一条前沿上：容量 ≤ 已有 → 只走查、不建文件；容量 >
已有 → 只补差量，已存在的部分由 C++ 的幂等探测跳过。
"""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("tutti_runtime._core")

from tutti.storage.tutti_nvme.object_layout import ObjectLayout

SPAN = 2
SEGMENT = 4096


def _layout(tmp_path, *, capacity=64, prewarm=4, probe_only=True):
    """probe_only=True 即"异步增长"部署：open 只走查、不建缺失槽位。"""
    root = tmp_path / "ns"
    root.mkdir(parents=True, exist_ok=True)
    mount = tmp_path / "dev0"
    mount.mkdir(parents=True, exist_ok=True)
    layout = ObjectLayout(
        root,
        SEGMENT,
        mounts=[str(mount)],
        capacity_chunks=capacity,
        prewarm_chunks=prewarm,
        background_reclaim=False,
        warmup_probe_only=probe_only,
        namespace=b"test-async-precreate",
    )
    layout.set_layer_span(SPAN)
    return layout


def _keys(*chunks: bytes) -> list[bytes]:
    return [
        chunk + layer.to_bytes(2, "little")
        for chunk in chunks
        for layer in range(SPAN)
    ]


def _path(uri: str) -> str:
    """``slot_uri`` 返回的是 URI（file://…），落到文件系统要剥掉 scheme。"""
    return uri[len("file://"):] if uri.startswith("file://") else uri


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_large_capacity_does_not_pay_at_open(tmp_path):
    """冷池（盘上零文件）配 4096 槽容量：open 后盘上仍然零文件。

    "大容量不等于长启动"的结构性断言：容量只是分配器的上限，open 不落盘。
    """
    layout = _layout(tmp_path, capacity=4096, prewarm=8)
    store = layout._store
    assert store.precreated_slots() == 0
    slots_dir = os.path.dirname(_path(store.slot_uri(0)))
    assert not os.path.exists(slots_dir) or os.listdir(slots_dir) == []


def test_default_warmup_still_creates_the_declared_prefix(tmp_path):
    """未启用异步预建（默认）时保持旧语义：open 把缺失的预热槽位建出来。

    异步是显式的部署选择；默认路径的行为不该被改变（测试与小池依赖它）。
    """
    layout = _layout(tmp_path, capacity=64, prewarm=4, probe_only=False)
    store = layout._store
    assert store.precreated_slots() == 4
    assert os.path.exists(_path(store.slot_uri(3)))
    assert not os.path.exists(_path(store.slot_uri(4)))


def test_open_proves_reuse_instead_of_writing(tmp_path):
    """已有槽位（复用）：open 走查登记为就绪，一个字节都不写。"""
    layout = _layout(tmp_path, capacity=64, prewarm=4)
    store = layout._store
    store.set_precreate_on_write(True)
    # 先让写路径把 0..3 建出来（模拟一个已经用过的池子）。
    layout.prepare_put(
        _keys(b"a" * 16, b"b" * 16, b"c" * 16, b"d" * 16), capacity_chunks=4
    )
    slots_dir = os.path.dirname(_path(store.slot_uri(0)))
    before = {name: os.stat(os.path.join(slots_dir, name)).st_mtime_ns
              for name in os.listdir(slots_dir)}

    reopened = _layout(tmp_path, capacity=64, prewarm=4)
    reopened_store = reopened._store
    assert reopened_store.precreated_slots() == 4, "已有槽位应当被走查证明为就绪"
    after = {name: os.stat(os.path.join(slots_dir, name)).st_mtime_ns
             for name in os.listdir(slots_dir)}
    assert before == after, "复用路径不得重写槽位文件"


def test_capacity_below_reuse_needs_no_growth(tmp_path):
    """容量 ≤ 已有：后台增长一个文件都不建（复用立即生效，无异步工作）。"""
    seeded = _layout(tmp_path, capacity=64, prewarm=4)
    seeded._store.set_precreate_on_write(True)
    seeded.prepare_put(
        _keys(b"a" * 16, b"b" * 16, b"c" * 16, b"d" * 16), capacity_chunks=4
    )
    slots_dir = os.path.dirname(_path(seeded._store.slot_uri(0)))
    before = sorted(os.listdir(slots_dir))
    assert len(before) == 4

    # 新容量 4 ≤ 盘上已有的 4 槽 ⇒ 走查即全就绪，后台无事可做。
    smaller = _layout(tmp_path, capacity=4, prewarm=4)
    assert smaller.start_background_precreate(threads=2)
    time.sleep(0.4)
    assert smaller._store.precreated_slots() == 4
    assert sorted(os.listdir(slots_dir)) == before, "容量以内不应新建文件"
    smaller.stop_background_precreate()


def test_write_path_refuses_instead_of_precreating(tmp_path):
    """写路径没拿到就绪槽位时立即拒绝——绝不自己 create+fsync。"""
    layout = _layout(tmp_path, capacity=64, prewarm=4)
    store = layout._store
    store.set_precreate_on_write(False)
    started = time.monotonic()
    admitted, rejected = layout.prepare_put(_keys(b"a" * 16), capacity_chunks=8)
    elapsed = time.monotonic() - started
    assert rejected == 1 and admitted == {}
    assert elapsed < 0.05, f"拒绝耗时 {elapsed * 1000:.1f}ms，写路径被预建阻塞了"
    assert store.precreated_slots() == 0, "写路径不得偷偷预建"
    slots_dir = os.path.dirname(_path(store.slot_uri(0)))
    assert not os.path.exists(slots_dir) or os.listdir(slots_dir) == []


def test_background_growth_fills_capacity(tmp_path, caplog):
    """容量 > 已有：后台把差量异步补齐（写路径随后无需自己建槽）。

    同时断言两件与"可持续"有关的事：追上目标时要留下可观测信号
    （线程名不进 /proc，没有日志就完全看不见它）；批次之间不得空等，
    否则追赶速度会被睡眠周期拖慢几个数量级。
    """
    layout = _layout(tmp_path, capacity=64, prewarm=4)
    store = layout._store
    assert layout.start_background_precreate(threads=2)
    started = time.monotonic()
    assert _wait_for(lambda: store.precreated_slots() >= 64, timeout=10.0)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"预建 64 个槽位用了 {elapsed:.2f}s，说明批次间在空等"
    assert os.path.exists(_path(store.slot_uri(63)))
    admitted, rejected = layout.prepare_put(_keys(b"b" * 16), capacity_chunks=8)
    assert rejected == 0 and len(admitted) == SPAN
    assert _wait_for(
        lambda: any("BACKGROUND_PRECREATE_CAUGHT_UP" in r.message
                    for r in caplog.records), timeout=2.0
    ), "预建追上目标后必须留下可观测信号"
    # 信号必须只在"真的到 target"时打：认领在飞导致的 0 返回不得被当成
    # 追平（真机首跑就在 precreated=151049/209715 处误报过一次）。
    for record in caplog.records:
        if "BACKGROUND_PRECREATE_CAUGHT_UP" in record.message:
            assert "precreated=64" in record.message, record.message
    layout.stop_background_precreate()


def test_demand_scope_keeps_only_headroom(tmp_path):
    """full_capacity=False：只保持需求余量，不为容量预先建文件。"""
    layout = _layout(tmp_path, capacity=64, prewarm=4)
    store = layout._store
    assert layout.start_background_precreate(threads=1, headroom=8,
                                               full_capacity=False)
    assert _wait_for(lambda: store.precreated_slots() >= 8)
    assert store.precreated_slots() < 64
    layout.stop_background_precreate()


def test_growth_never_exceeds_capacity(tmp_path):
    """就绪余量再大也不越过容量上限。"""
    layout = _layout(tmp_path, capacity=8, prewarm=4)
    store = layout._store
    assert layout.start_background_precreate(threads=2, headroom=64)
    assert _wait_for(lambda: store.precreated_slots() >= 8)
    time.sleep(0.6)
    assert store.precreated_slots() == 8


def test_start_is_idempotent(tmp_path):
    layout = _layout(tmp_path, capacity=64, prewarm=4)
    assert layout.start_background_precreate(threads=1, headroom=8)
    assert layout.start_background_precreate(threads=1, headroom=8)
    layout.stop_background_precreate()


def test_on_write_precreate_is_the_default(tmp_path):
    """没开后台增长时保持旧行为：写路径按需预建。"""
    layout = _layout(tmp_path, capacity=64, prewarm=4)
    store = layout._store
    admitted, rejected = layout.prepare_put(
        _keys(b"c" * 16, b"d" * 16, b"e" * 16, b"f" * 16, b"g" * 16),
        capacity_chunks=8,
    )
    assert rejected == 0 and len(admitted) == 5 * SPAN
    assert store.precreated_slots() == 5, "写路径应当自己把槽位建出来"
    assert os.path.exists(_path(store.slot_uri(4)))


def test_grower_failure_hands_precreate_back_to_the_write_path(tmp_path):
    """后台增长出错时必须交还按需预建，否则池子会静默停止增长。"""
    layout = _layout(tmp_path, capacity=64, prewarm=4)
    store = layout._store

    def boom(*_args, **_kwargs):
        raise RuntimeError("injected grower failure")

    store.precreate_step = boom
    assert layout.start_background_precreate(threads=1, headroom=8)

    # 写路径恢复自建槽位是唯一能解释"随后这笔写成功"的原因。
    admitted, rejected = layout.prepare_put(_keys(b"d" * 16), capacity_chunks=8)
    if rejected:
        for _ in range(100):
            time.sleep(0.05)
            admitted, rejected = layout.prepare_put(
                _keys(b"d" * 16), capacity_chunks=8
            )
            if not rejected:
                break
    assert rejected == 0 and len(admitted) == SPAN

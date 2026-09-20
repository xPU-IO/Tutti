"""对象层布局契约：按对象提交、零 Python 元数据文件、可恢复。

这些用例覆盖评审定案的三条语义：
  1. 提交是按对象（chunk）一次，不是按层——层没写齐的对象永远不可见；
  2. Python 侧不产生任何元数据文件（没有 marker，也没有 manifest）；
  3. 冷启动驻留集合来自对象层恢复，而不是重新扫描目录。
"""

from __future__ import annotations

import pytest

pytest.importorskip("tutti_runtime._core")

from tutti.storage.tutti_nvme.object_layout import ObjectLayout

SPAN = 3
SEGMENT = 4096
SLOT_BYTES = 4096 + SEGMENT * SPAN


def _io_key(chunk: bytes, layer: int) -> bytes:
    return chunk + layer.to_bytes(2, "little")


def _layout(tmp_path, *, mounts=None, stripe_unit=0, rank_id=0):
    root = tmp_path / "ns"
    root.mkdir(parents=True, exist_ok=True)
    if mounts is None:
        mounts = [tmp_path / "dev0"]
    for mount in mounts:
        mount.mkdir(parents=True, exist_ok=True)
    layout = ObjectLayout(
        root,
        SEGMENT,
        mounts=[str(m) for m in mounts],
        stripe_unit=stripe_unit,
        capacity_chunks=8,
        prewarm_chunks=4,
        rank_id=rank_id,
        background_reclaim=False,
        namespace=b"test-namespace",
    )
    layout.set_layer_span(SPAN)
    return layout


def test_commit_is_per_object_not_per_layer(tmp_path):
    """层未写齐 = 整个对象未提交（读侧看不到半截 chunk）。"""
    layout = _layout(tmp_path)
    a, b = b"a" * 16, b"b" * 16
    layout.prepare_put(
        [_io_key(a, layer) for layer in range(SPAN)]
        + [_io_key(b, layer) for layer in range(SPAN)],
        capacity_chunks=8,
    )

    # 两个 chunk 都只写了前两层：一个都不该可见
    layout.commit_layers(
        [_io_key(a, 0), _io_key(a, 1), _io_key(b, 0), _io_key(b, 1)]
    )
    assert layout.committed_chunks() == set()

    # a 补齐最后一层 → 只有 a 提交
    layout.commit_layers([_io_key(a, 2)])
    assert layout.committed_chunks() == {a}
    assert layout.releasable_chunks([_io_key(b, 0)]) == set()
    assert layout.releasable_chunks([_io_key(a, 0)]) == {a}


def test_placement_is_stable_single_file(tmp_path):
    """槽位 URI 稳定、段 0 跳过对象头、payload 覆盖全部层。"""
    layout = _layout(tmp_path)
    chunk = b"c" * 16
    layout.prepare_put([_io_key(chunk, 0)], capacity_chunks=8)

    uri = layout.target_uri(chunk)
    assert uri.startswith("file://")
    assert layout.target_uri(chunk) == uri  # 稳定：不随分配改名
    assert layout.target_offset(chunk) == 4096
    assert layout.target_size(chunk) == SEGMENT * SPAN
    assert layout.target_generation(chunk) == 0


def test_python_side_writes_no_metadata_files(tmp_path):
    """Python 不落任何元数据：没有 marker 目录，也没有 manifest。"""
    layout = _layout(tmp_path)
    chunk = b"d" * 16
    layout.prepare_put([_io_key(chunk, layer) for layer in range(SPAN)],
                       capacity_chunks=8)
    layout.commit_layers([_io_key(chunk, layer) for layer in range(SPAN)])

    root = tmp_path / "ns"
    names = sorted(p.name for p in root.iterdir())
    assert not [n for n in names if n.endswith(".ok")], names
    assert not [n for n in names if "manifest" in n], names


def test_recover_survives_reopen_and_ignores_uncommitted(tmp_path):
    """提交过的对象重开后仍在；未提交的预留既不复活也不占位。"""
    layout = _layout(tmp_path)
    kept, dropped = b"e" * 16, b"f" * 16
    layout.prepare_put(
        [_io_key(kept, layer) for layer in range(SPAN)]
        + [_io_key(dropped, layer) for layer in range(SPAN)],
        capacity_chunks=8,
    )
    layout.commit_layers([_io_key(kept, layer) for layer in range(SPAN)])
    layout.commit_layers([_io_key(dropped, 0)])  # 只写了一层
    layout.close_object_pool()

    reopened = _layout(tmp_path)
    assert reopened.committed_chunks() == {kept}


def test_release_removes_object_from_recovery_set(tmp_path):
    """回收后对象不再出现在恢复集合里，槽位可被重新分配。"""
    layout = _layout(tmp_path)
    chunk = b"g" * 16
    layout.prepare_put([_io_key(chunk, layer) for layer in range(SPAN)],
                       capacity_chunks=8)
    layout.commit_layers([_io_key(chunk, layer) for layer in range(SPAN)])
    assert layout.release_chunks([chunk]) == 1
    assert layout.committed_chunks() == set()

    layout.close_object_pool()
    assert _layout(tmp_path).committed_chunks() == set()


def test_striped_uri_carries_geometry(tmp_path):
    """条带布局的 URI 直接由现有 striped resolver 消费。

    条带几何要求 payload 能被 ``stripe_unit × N`` 整除（否则末轮条带不满，
    段会跨到没有预留空间的盘上），所以这里单独取几何。
    """
    mounts = [tmp_path / "d0", tmp_path / "d1"]
    for mount in mounts:
        mount.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "ns"
    root.mkdir(parents=True, exist_ok=True)
    layout = ObjectLayout(
        root,
        8192,
        mounts=[str(m) for m in mounts],
        stripe_unit=4096,
        capacity_chunks=8,
        prewarm_chunks=4,
        background_reclaim=False,
        namespace=b"test-namespace",
    )
    layout.set_layer_span(4)
    chunk = b"h" * 16
    layout.prepare_put([_io_key(chunk, 0)], capacity_chunks=8)

    uri = layout.target_uri(chunk)
    assert uri.startswith("striped://")
    assert f"devs={mounts[0]},{mounts[1]}" in uri
    assert "unit=4096" in uri


def test_commit_before_layer_span_is_impossible(tmp_path):
    """层宽未定案时不允许提交：对象几何未知，无法判定"写齐"。"""
    root = tmp_path / "ns"
    root.mkdir(parents=True, exist_ok=True)
    layout = ObjectLayout(root, SEGMENT, mounts=[str(tmp_path)], stripe_unit=0)
    with pytest.raises(RuntimeError, match="set_layer_span"):
        layout.prepare_put([_io_key(b"i" * 16, 0)], capacity_chunks=4)

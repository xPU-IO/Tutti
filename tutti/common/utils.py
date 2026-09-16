"""跨模块共用的纯工具函数（唯一真源）。

原先 engine/core.py、engine/metadata.py、engine/staging.py、
stores/memory.py 各自复制了一份 `_is_int` / `_positive_int` / `_group_scan`；
评审意见指出这属于重复实现。这里收敛为唯一实现，各调用点改为导入。

本模块只放**无状态纯函数**，除 index 的 key 编解码外不依赖项目内部模块，
因此可被任意层安全导入。
"""

from __future__ import annotations

from tutti.index.chunk_index import IO_KEY_BYTES, chunk_key_of, layer_of


def is_int(value) -> bool:
    """是否为真 int（排除 bool：Python 里 bool 是 int 的子类）。"""
    return isinstance(value, int) and not isinstance(value, bool)


def positive_int(config: dict, key: str) -> int:
    """从 config 读取正整数键；缺失或非法 → ValueError。"""
    value = config.get(key)
    if not is_int(value) or value <= 0:
        raise ValueError(f"config[{key!r}] 须为正整数，got {value!r}")
    return value


def flatten_block_ids(block_tables) -> tuple[int, ...]:
    """Flatten wave-local block tables for fail-closed error reporting."""
    if block_tables is None:
        return ()
    if isinstance(block_tables, (bytes, bytearray, str)):
        return ()
    try:
        values = list(block_tables)
    except TypeError:
        return (block_tables,) if isinstance(block_tables, int) else ()
    flattened = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            flattened.extend(item for item in value if isinstance(item, int))
        elif isinstance(value, int):
            flattened.append(value)
    return tuple(flattened)


def group_scan(store) -> dict[bytes, set[int]]:
    """把 store 的存活枚举按 chunk key 分组为层集合。

    线格式非法的条目直接忽略（扫描是诊断/对账用途，不应因单条脏数据
    中断整轮）；返回 chunk key → 该 chunk 已落盘的层集合。
    """
    groups: dict[bytes, set[int]] = {}
    for io_key in store.scan():
        if (not isinstance(io_key, (bytes, bytearray))
                or len(io_key) != IO_KEY_BYTES):
            continue
        payload = bytes(io_key)
        groups.setdefault(chunk_key_of(payload), set()).add(layer_of(payload))
    return groups

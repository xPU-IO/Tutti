"""存储插件注册表：type 名 → 实现类。

调用方以 {"type": ..., "options": {...}} 描述所需存储；options 由
本模块原样透传实现构造函数，键义归各插件私有文档。
"""

from .base import KVStore
from .memory import MemoryKVStore
from .tutti_nvme.store import TuttiKVStore

_STORE_TYPES: dict[str, type[KVStore]] = {}


def register_store_type(name: str, cls: type[KVStore]) -> None:
    """注册实现类；重名 → ValueError。"""
    if name in _STORE_TYPES:
        raise ValueError(f"store type 已注册：{name!r}")
    _STORE_TYPES[name] = cls


def create_store(type_name: str, options: dict) -> KVStore:
    """按 type 名构造 store 实例。

    options 原样透传构造函数；未知 type → ValueError；
    options 与构造签名不符时由实现抛出相应异常。
    """
    cls = _STORE_TYPES.get(type_name)
    if cls is None:
        raise ValueError(
            f"未知 store type：{type_name!r}（已注册：{sorted(_STORE_TYPES)}）"
        )
    return cls(**options)


register_store_type("memory", MemoryKVStore)
register_store_type("tutti_nvme", TuttiKVStore)

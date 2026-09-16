"""存储插件注册表：type 名 → 实现。

调用方以 {"type": ..., "options": {...}} 描述所需存储；options 由
本模块原样透传实现构造函数，键义归各插件私有文档。

**两套注册面**（数据面 / 调度侧仅元数据），共用同一份 type 名空间语义：
- ``_STORE_TYPES``：类对象，worker 进程使用（数据面实现已在导入期就绪）；
- ``_METADATA_STORE_TYPES``：``"module:Class"`` 字符串，**惰性解析**——
  调度进程不得因注册表而导入数据面模块（绑定/驱动依赖），故推迟到
  真正构造时再 import。
"""

import importlib

from .base import KVStore
from .memory import MemoryKVStore
from .tutti_nvme.store import TuttiKVStore

_STORE_TYPES: dict[str, type[KVStore]] = {}
_METADATA_STORE_TYPES: dict[str, str] = {}


def register_store_type(name: str, cls: type[KVStore]) -> None:
    """注册数据面实现类；重名 → ValueError。"""
    if name in _STORE_TYPES:
        raise ValueError(f"store type 已注册：{name!r}")
    _STORE_TYPES[name] = cls


def register_metadata_store_type(name: str, target: str) -> None:
    """注册调度侧（仅元数据）实现：name → ``"module:Class"``；重名 → ValueError。"""
    if name in _METADATA_STORE_TYPES:
        raise ValueError(f"metadata store type 已注册：{name!r}")
    _METADATA_STORE_TYPES[name] = target


def create_metadata_store(type_name: str, options: dict):
    """按 type 名构造调度侧 store（惰性解析目标模块，不触碰数据面绑定）。"""
    target = _METADATA_STORE_TYPES.get(type_name)
    if target is None:
        raise ValueError(f"store type {type_name!r} has no metadata-only client")
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)(**options)


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

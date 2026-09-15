"""引擎构造：按配置取同进程共享的 KVEngine / SchedulerMetadataIndex。

从 adapter/connector.py 搬出（评审意见：壳内不应承载装配逻辑）。构造
需要按角色选择存储实现（stores.registry / stores.metadata），因此这里
持有进程级实例表——实例表按配置对象身份索引，避免同一进程内重复建库。
函数体自原处逐字搬迁，行为不变。
"""

from __future__ import annotations

from typing import Any

from adapter.geometry import (
    _ENGINE_KEYS,
    _ENGINE_OPTIONAL_KEYS,
    apply_device_groups,
    deployment_rank,
    expand_placeholders,
    key_namespace,
)

# 同进程同配置的引擎实例表：键 = vllm_config 对象身份（值持有配置
# 强引用防 id 回收复用），二级键 = 规范化配置三元组。
_ENGINE_CACHE: dict[int, tuple[Any, dict[tuple, Any]]] = {}
_SCHEDULER_CACHE: dict[int, tuple[Any, dict[tuple, Any]]] = {}


def worker_engine_for(vllm_config, extra: dict):
    """取同进程共享的引擎实例；extra 可直传实例绕过构造。

    实例表按配置对象身份索引并持有其强引用（配置存活期内 id 不
    复用；配置回收后条目随之失活——查询以 is 校验双保险，避免
    回收地址复用导致的假命中）。
    """
    injected = extra.get("tutti_engine_instance")
    if injected is not None:
        return injected
    keys = tuple(sorted(
        (k, extra.get(k))
        for k in (*_ENGINE_KEYS, *_ENGINE_OPTIONAL_KEYS, "num_layers")
    ))
    entry = _ENGINE_CACHE.get(id(vllm_config))
    if entry is None or entry[0] is not vllm_config:
        entry = (vllm_config, {})
        _ENGINE_CACHE[id(vllm_config)] = entry
    engine = entry[1].get(keys)
    if engine is None:
        from engine.core import KVEngine
        from stores.registry import create_store

        store_spec = extra.get("store") or {"type": "memory", "options": {}}
        worker_rank = deployment_rank(vllm_config, worker=True)
        options = expand_placeholders(
            dict(store_spec.get("options") or {}), vllm_config,
            rank=worker_rank,
        )
        if store_spec["type"] == "tutti_nvme":
            tp_size = int(getattr(
                getattr(vllm_config, "parallel_config", None),
                "tensor_parallel_size", 1,
            ))
            options = apply_device_groups(
                options, rank=worker_rank, tp_size=tp_size
            )
            options.setdefault("rank_id", int(worker_rank))
            options.setdefault("tp_size", tp_size)
        segment_bytes = extra["chunk_kv_bytes"] // extra["num_layers"]
        configured_segment = options.get("segment_bytes")
        if configured_segment is not None and configured_segment != segment_bytes:
            raise ValueError(
                f"store segment_bytes({configured_segment}) 与 KVCacheConfig "
                f"推导值 {segment_bytes} 不一致"
            )
        options["segment_bytes"] = segment_bytes
        store = create_store(store_spec["type"], options)
        # 可选层数预告：查询侧（不做 bind）的驱逐展开与冷启动完整性
        # 判定依赖层数；与缓存键无关（同配置实例共享同引擎）。
        config = {k: extra[k] for k in _ENGINE_KEYS}
        for key in _ENGINE_OPTIONAL_KEYS:
            if key in extra:
                config[key] = extra[key]
        if extra.get("num_layers") is not None:
            config["num_layers"] = extra["num_layers"]
        # key 命名空间：同 vllm_config 派生恒定，无需入缓存键
        config["key_namespace"] = key_namespace(vllm_config, extra)
        engine = KVEngine(config, store)
        entry[1][keys] = engine
    return engine


def scheduler_index_for(vllm_config, extra: dict):
    """Build/cache the scheduler metadata client without data-plane imports."""
    injected = extra.get("tutti_scheduler_index_instance")
    if injected is None:
        # Compatibility for existing pure-Python test harnesses. Production
        # configurations never carry an object in connector extra_config.
        injected = extra.get("tutti_engine_instance")
    if injected is not None:
        return injected
    keys = tuple(sorted(
        (k, extra.get(k))
        for k in (*_ENGINE_KEYS, "num_layers")
    ))
    entry = _SCHEDULER_CACHE.get(id(vllm_config))
    if entry is None or entry[0] is not vllm_config:
        entry = (vllm_config, {})
        _SCHEDULER_CACHE[id(vllm_config)] = entry
    index = entry[1].get(keys)
    if index is None:
        from engine.metadata import SchedulerMetadataIndex
        from stores.metadata import create_metadata_store

        store_spec = extra.get("store") or {"type": "memory", "options": {}}
        raw_options = dict(store_spec.get("options") or {})
        tp_size = int(getattr(
            getattr(vllm_config, "parallel_config", None),
            "tensor_parallel_size", 1,
        ))
        if store_spec["type"] == "tutti_nvme":
            metadata_ranks = (
                list(range(tp_size)) if tp_size > 1
                else [int(deployment_rank(vllm_config))]
            )
            rank_options = []
            for rank in metadata_ranks:
                item = dict(apply_device_groups(
                    expand_placeholders(
                        raw_options, vllm_config, rank=str(rank)
                    ),
                    rank=str(rank),
                    tp_size=tp_size,
                ))
                # striped 数据盘按 rank 分目录（<mount>/striped/r<rank>/...），
                # 调度侧的 target_size 探测必须用与 worker 相同的 rank。
                item.setdefault("rank_id", int(rank))
                rank_options.append(item)
            roots = [str(item.get("root", "")) for item in rank_options]
            if tp_size > 1 and len(set(roots)) != tp_size:
                raise ValueError(
                    "TP ranks require distinct metadata roots; include "
                    "{LOCAL_RANK} in the Tutti root"
                )
            options = dict(rank_options[0])
            options["rank_options"] = rank_options
            options["tp_size"] = tp_size
        else:
            options = expand_placeholders(
                raw_options, vllm_config, rank=deployment_rank(vllm_config)
            )
        segment_bytes = extra["chunk_kv_bytes"] // extra["num_layers"]
        configured_segment = options.get("segment_bytes")
        if configured_segment is not None and configured_segment != segment_bytes:
            raise ValueError(
                f"store segment_bytes({configured_segment}) 与 KVCacheConfig "
                f"推导值 {segment_bytes} 不一致"
            )
        options["segment_bytes"] = segment_bytes
        store = create_metadata_store(store_spec["type"], options)
        config = {k: extra[k] for k in _ENGINE_KEYS}
        config["num_layers"] = extra["num_layers"]
        config["key_namespace"] = key_namespace(vllm_config, extra)
        index = SchedulerMetadataIndex(config, store)
        entry[1][keys] = index
    return index

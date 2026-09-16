"""配置解析与几何推导：vLLM 配置 → connector 运行参数。

从 adapter/connector.py 搬出（评审意见：壳内不应承载配置解析）。这里是
**纯函数层**：只读配置、算几何、展开占位符，不构造任何存储/引擎对象
（构造在 adapter/factory.py）。函数体自原处逐字搬迁，行为不变。
"""

from __future__ import annotations

import os

# 引擎构造所需的配置键（全部与硬件无关）。chunk_kv_bytes 与
# num_layers 优先从 vLLM KVCacheConfig 推导，extra_config 仅作兼容覆盖。
_ENGINE_REQUIRED_KEYS = ("chunk_tokens", "max_chunks_per_wave")
_ENGINE_KEYS = (*_ENGINE_REQUIRED_KEYS, "chunk_kv_bytes")
_ENGINE_OPTIONAL_KEYS = ("direct_transfer", "direct_transfer_strict")

# vLLM 的 cdiv 惰性解析缓存：该函数在每请求路径上被调用，不能每次
# 都走 import 查找；同时纯 Python 单测环境没有 vLLM，需保留本地兜底。
_CDIV = None


def cdiv(a: int, b: int) -> int:
    """向上取整除法（优先复用 vLLM 实现，保证与上游分块语义一致）。"""
    global _CDIV
    if _CDIV is None:
        try:
            from vllm.utils.math_utils import cdiv as _resolved
        except Exception:  # pragma: no cover - 非 vLLM 环境
            def _resolved(x, y):
                return -(x // -y)
        _CDIV = _resolved
    return int(_CDIV(a, b))


def flatten_blocks(block_ids) -> list[int]:
    """把 vLLM 的 block id 形态（单层 list 或按缓存组嵌套）摊平为一维。"""
    if not block_ids:
        return []
    if isinstance(block_ids[0], (list, tuple)):
        return list(block_ids[0])
    return list(block_ids)


def extra_config(vllm_config) -> dict:
    """读取 kv_connector_extra_config；缺省为空映射。"""
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    extra = getattr(transfer, "kv_connector_extra_config", None)
    return dict(extra) if extra else {}


def resolve_geometry(extra: dict, kv_cache_config) -> dict:
    """从 vLLM cache spec 推导 Tutti 的单组、定宽逐层几何。

    旧测试/调用方没有 KVCacheConfig 时仍可显式提供 num_layers 与
    chunk_kv_bytes。生产路径有 KVCacheConfig 时，显式值只允许与权威
    spec 一致，避免 scheduler 与 worker 使用不同字节布局。
    """
    resolved = dict(extra)
    for key in _ENGINE_REQUIRED_KEYS:
        value = resolved.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"extra_config 缺少正整数引擎键 {key!r}")

    groups = list(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    if not groups:
        for key in ("num_layers", "chunk_kv_bytes"):
            value = resolved.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"无法从 KVCacheConfig 推导几何；extra_config 须提供"
                    f"正整数 {key!r}"
                )
        return resolved
    if len(groups) != 1:
        raise ValueError(
            "TuttiConnectorV1 当前仅支持单 KV cache group；"
            f"模型提供了 {len(groups)} groups，需要 HMA/multi-group 支持"
        )

    group = groups[0]
    layer_names = list(getattr(group, "layer_names", ()) or ())
    if not layer_names:
        raise ValueError("KVCacheConfig 的唯一 cache group 不含层")
    spec = getattr(group, "kv_cache_spec", None)
    block_size = getattr(spec, "block_size", None)
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("KV cache spec 缺少正整数 block_size")
    if resolved["chunk_tokens"] % block_size:
        raise ValueError(
            f"chunk_tokens({resolved['chunk_tokens']}) 必须是 KV cache "
            f"block_size({block_size}) 的整数倍"
        )

    child_specs = getattr(spec, "kv_cache_specs", None)
    if isinstance(child_specs, dict):
        try:
            page_sizes = [child_specs[name].page_size_bytes for name in layer_names]
        except KeyError as exc:
            raise ValueError(f"KV cache group 层缺少物理 spec：{exc.args[0]}") from exc
    else:
        page_size = getattr(spec, "page_size_bytes", None)
        page_sizes = [page_size] * len(layer_names)
    if any(not isinstance(size, int) or size <= 0 for size in page_sizes):
        raise ValueError("KV cache spec 含无效 page_size_bytes")
    if len(set(page_sizes)) != 1:
        raise ValueError(
            "TuttiConnectorV1 当前要求每层 KV page 字节数一致；"
            f"得到 {sorted(set(page_sizes))}"
        )

    num_layers = len(layer_names)
    blocks_per_chunk = resolved["chunk_tokens"] // block_size
    segment_bytes = blocks_per_chunk * page_sizes[0]
    chunk_kv_bytes = num_layers * segment_bytes
    for key, derived in (
        ("num_layers", num_layers),
        ("chunk_kv_bytes", chunk_kv_bytes),
    ):
        configured = resolved.get(key)
        if configured is not None and configured != derived:
            raise ValueError(
                f"extra_config[{key!r}]={configured} 与 KVCacheConfig "
                f"推导值 {derived} 不一致"
            )
        resolved[key] = derived
    resolved["kv_group_layer_names"] = tuple(layer_names)
    return resolved


def deployment_rank(vllm_config=None, *, worker: bool = False) -> str:
    """Resolve the process-local accelerator rank after vLLM distributed init."""
    if worker:
        try:
            from vllm.distributed.parallel_state import get_world_group
            return str(get_world_group().local_rank)
        except (AssertionError, RuntimeError):
            pass
        try:
            import torch
            if torch.cuda.is_initialized():
                return str(torch.cuda.current_device())
        except (ImportError, RuntimeError):
            pass
    parallel = getattr(vllm_config, "parallel_config", None)
    rank = getattr(parallel, "rank", None) if parallel is not None else None
    if rank is None:
        rank = os.environ.get("LOCAL_RANK", "0")
    return str(rank)


def expand_placeholders(value, vllm_config=None, *, rank: str | None = None):
    """递归替换字符串值中的 {LOCAL_RANK} 占位符（多副本部署的按副本分叉）。

    rank 源优先级：vllm_config.parallel_config.rank（vLLM 各进程构造
    connector 时在手的权威副本号）→ LOCAL_RANK 环境变量 → 0。
    vLLM V1 多进程 worker 不设 LOCAL_RANK，仅靠环境变量时 4 个副本
    会全部展开为 0（真机实测），故配置对象优先。
    """
    if rank is None:
        rank = deployment_rank(vllm_config)
    if isinstance(value, str):
        return value.replace("{LOCAL_RANK}", rank)
    if isinstance(value, dict):
        return {
            k: expand_placeholders(v, vllm_config, rank=rank)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [expand_placeholders(v, vllm_config, rank=rank) for v in value]
    return value


def apply_device_groups(options: dict, *, rank: str, tp_size: int) -> dict:
    """按 rank 把 preset.device_groups 展开为 preset.devices。

    多卡共用盘组的部署形态（如 8 卡、每 4 个 rank 共用一组 2 盘条带）：
    ``device_groups=[[0, 1], [2, 3]]`` 表示前一半 rank 用盘 0-1、后一半
    用盘 2-3；组数必须整除 tp_size，组内 rank 数 = tp_size / 组数。
    无 device_groups 时原样返回（单设备/单组部署不变）。
    """
    preset = options.get("preset")
    if not isinstance(preset, dict) or "device_groups" not in preset:
        return options
    groups = preset["device_groups"]
    if (not isinstance(groups, (list, tuple)) or not groups
            or any(not isinstance(group, (list, tuple)) or not group
                   for group in groups)):
        raise ValueError("preset.device_groups 必须是非空设备组列表")
    if tp_size % len(groups):
        raise ValueError(
            f"tensor_parallel_size({tp_size}) 必须能被 device_groups "
            f"组数({len(groups)}) 整除"
        )
    group_span = tp_size // len(groups)
    index = int(rank) // group_span
    if not 0 <= index < len(groups):
        raise ValueError(f"rank {rank} 越出 device_groups 分组范围")
    preset = dict(preset)
    preset.pop("device_groups")
    preset["devices"] = [{"device_id": int(device)} for device in groups[index]]
    expanded = dict(options)
    expanded["preset"] = preset
    return expanded


def key_namespace(vllm_config, extra: dict) -> str:
    """组装 chunk key 命名空间（影响字节布局的维度，v2 格式头）。

    字段取舍以"影响 KV 字节布局"为准：模型标识、KV dtype、TP world
    size、chunk KV 字节数（含 head/dim 几何）、chunk_tokens。字段序
    固定；worker_id 不入（per-rank 池已物理隔离）。旧池数据不兼容
    （key 变更），部署升级时清池。
    """
    model = getattr(getattr(vllm_config, "model_config", None), "model", "")
    cache_cfg = getattr(vllm_config, "cache_config", None)
    dtype = getattr(cache_cfg, "cache_dtype", "") or ""
    tp = getattr(getattr(vllm_config, "parallel_config", None),
                 "tensor_parallel_size", 1)
    return "|".join([
        "v2",
        f"model={model}",
        f"dtype={dtype}",
        f"tp={tp}",
        f"chunk_kv_bytes={extra.get('chunk_kv_bytes')}",
        f"chunk_tokens={extra.get('chunk_tokens')}",
    ])

"""配置解析与几何推导：vLLM 配置 → connector 运行参数。

从 adapter/connector.py 搬出（评审意见：壳内不应承载配置解析）。这里是
**纯函数层**：只读配置、算几何、展开占位符，不构造任何存储/引擎对象
（构造在 adapter/factory.py）。函数体自原处逐字搬迁，行为不变。
"""

from __future__ import annotations

import os

from tutti.common.utils import parse_capacity

# 引擎构造所需的配置键（全部与硬件无关）。chunk_kv_bytes 与
# num_layers 优先从 vLLM KVCacheConfig 推导，extra_config 仅作兼容覆盖。
_ENGINE_REQUIRED_KEYS = ("chunk_tokens", "max_chunks_per_wave")
_ENGINE_KEYS = (*_ENGINE_REQUIRED_KEYS, "chunk_kv_bytes")
_ENGINE_OPTIONAL_KEYS = ("direct_transfer",)

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


def apply_wave_default(extra: dict, *, blocks_per_chunk: int = 1) -> dict:
    """缺省波次最多512个chunk；已知preset批次额度时按block几何缩小。

    wave是引擎工作窗口，不是盘条带、NVMe队列深度或整个请求大小。
    显式配置原样保留，继续由原准入路径校验。
    """
    if "max_chunks_per_wave" in extra:
        return extra
    wave = 512
    store = extra.get("store") or {}
    preset = (store.get("options") or {}).get("preset")
    if store.get("type") == "tutti_nvme" and isinstance(preset, dict):
        max_batch = preset.get("max_batch_entries")
        if isinstance(max_batch, str) and max_batch.strip().isdigit():
            max_batch = int(max_batch)
        if max_batch is None:
            max_batch = 4096 if preset.get("type", "raid0") == "local" else 16384
        elif isinstance(max_batch, int) and not isinstance(max_batch, bool) and max_batch == 0:
            max_batch = 256
        if isinstance(max_batch, int) and not isinstance(max_batch, bool) and max_batch > 0:
            wave = min(wave, max_batch // blocks_per_chunk)
            if wave == 0:
                raise ValueError("preset.max_batch_entries 不足一个chunk，不能推导max_chunks_per_wave")
    resolved = dict(extra)
    resolved["max_chunks_per_wave"] = wave
    return resolved


def resolve_geometry(extra: dict, kv_cache_config, *, dcp_size: int = 1) -> dict:
    """从 vLLM cache spec 推导 Tutti 的单组、定宽逐层几何。

    旧测试/调用方没有 KVCacheConfig 时仍可显式提供 num_layers 与
    chunk_kv_bytes。生产路径有 KVCacheConfig 时，显式值只允许与权威
    spec 一致，避免 scheduler 与 worker 使用不同字节布局。
    """
    if not isinstance(dcp_size, int) or isinstance(dcp_size, bool) or dcp_size <= 0:
        raise ValueError(f"dcp_size must be a positive integer, got {dcp_size!r}")
    if "prefetch_k" in extra:
        raise ValueError("prefetch_k 已删除，请使用 lookahead_k")
    resolved = dict(extra)
    for key in _ENGINE_REQUIRED_KEYS:
        if key == "max_chunks_per_wave" and key not in resolved:
            continue
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
        return apply_wave_default(resolved)
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
    # vLLM's scheduler config simplifies UniformTypeKVCacheSpecs, but keeps
    # per-entity tensor allocations. Recover physical widths from those facts
    # rather than applying the arbitrary representative spec to every entity.
    num_blocks = getattr(kv_cache_config, "num_blocks", 0)
    allocations = list(getattr(kv_cache_config, "kv_cache_tensors", ()) or ())
    physical_pages = {}
    if isinstance(num_blocks, int) and num_blocks > 0:
        for allocation in allocations:
            shared = list(getattr(allocation, "shared_by", ()) or ())
            size = getattr(allocation, "size", 0)
            if len(shared) == 1 and isinstance(size, int) and size > 0 and size % num_blocks == 0:
                physical_pages[shared[0]] = size // num_blocks
    if all(name in physical_pages for name in layer_names):
        derived_pages = [physical_pages[name] for name in layer_names]
        if isinstance(child_specs, dict) and derived_pages != page_sizes:
            raise ValueError("KV tensor allocation与逐实体page spec不一致")
        page_sizes = derived_pages
    if any(not isinstance(size, int) or size <= 0 for size in page_sizes):
        raise ValueError("KV cache spec 含无效 page_size_bytes")
    resolved["kv_group_page_bytes"] = tuple(page_sizes)
    independent_entities = (resolved.get("replica_io", False) or
                            len(set(page_sizes)) != 1 or
                            isinstance(child_specs, dict) and
                            len({type(child_specs[name]) for name in layer_names}) > 1)
    if independent_entities:
        if any(size % 4096 for size in page_sizes):
            raise ValueError("异构 KV page 必须按4KiB对齐")
        # One logical group shares block IDs. Disk slots keep a fixed entity
        # width; each entity transfers only its own physical bytes.
        resolved["kv_layer_page_bytes"] = tuple(page_sizes)

    num_layers = len(layer_names)
    if resolved["chunk_tokens"] % dcp_size:
        raise ValueError(
            f"chunk_tokens({resolved['chunk_tokens']}) must be divisible by "
            f"dcp_size({dcp_size})"
        )
    # Keep hash chunks in global-token units. A DCP rank stores only its local
    # token shard, so its physical transfer chunk is smaller by dcp_size.
    physical_chunk_tokens = resolved["chunk_tokens"] // dcp_size
    if physical_chunk_tokens % block_size:
        raise ValueError(
            f"physical chunk tokens({physical_chunk_tokens}) must be divisible "
            f"by block_size({block_size})"
        )
    blocks_per_chunk = physical_chunk_tokens // block_size
    segment_bytes = blocks_per_chunk * max(page_sizes)
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
    resolved["kv_block_tokens"] = block_size
    resolved["scheduler_block_tokens"] = block_size * dcp_size
    resolved["dcp_size"] = dcp_size
    resolved["physical_chunk_tokens"] = physical_chunk_tokens
    resolved["blocks_per_chunk"] = blocks_per_chunk
    resolved["kv_page_bytes"] = max(page_sizes)
    return apply_wave_default(resolved, blocks_per_chunk=blocks_per_chunk)


def resolve_io_budget(options: dict, extra: dict, vllm_config) -> dict:
    """按已知模型工作量选择有限预算，不保证任意extent fan-out都无需拆批。

    小模型不沿用多盘16384的固定批宽。4KiB粒度用于保守工作量估算，
    不是覆盖namespace块大小；真实sub-IO宽度仍由native准入约束。
    显式高级预算保留；不根据瞬时空闲显存放大到最大值。
    """
    preset = options.get("preset")
    if not isinstance(preset, dict):
        return options
    preset = dict(preset)
    layers = int(extra["num_layers"])
    from tutti.integration.vllm.load_policy import resolve_load_mode

    load_mode = resolve_load_mode(extra)
    if preset.get("max_in_flight_operations") is None:
        preset["max_in_flight_operations"] = layers * (2 if load_mode == "sync" else 3)
    if preset.get("max_batch_entries") is None:
        block_tokens = extra.get("kv_block_tokens")
        page_bytes = extra.get("kv_page_bytes")
        model = getattr(vllm_config, "model_config", None)
        scheduler = getattr(vllm_config, "scheduler_config", None)
        context = getattr(model, "max_model_len", None)
        sequences = getattr(scheduler, "max_num_seqs", None)
        known = all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                    for value in (block_tokens, page_bytes, context, sequences))
        default_entries = 4096 if preset.get("type", "raid0") == "local" else 16384
        # Async reads can aggregate requests across scheduler steps. Shrinking
        # their batch by a single-step limit would multiply retained op handles.
        if known and load_mode == "sync":
            blocks_per_chunk = int(extra.get(
                "blocks_per_chunk", extra["chunk_tokens"] // block_tokens
            ))
            wave_blocks = extra["max_chunks_per_wave"] * blocks_per_chunk
            requests = cdiv(context, block_tokens) * sequences
            expansion = cdiv(page_bytes, 4096)
            # Keep one configured wave admissible, but don't grow beyond the
            # established workload default just because a context is huge.
            default_entries = max(wave_blocks, min(default_entries, requests * expansion))
        preset["max_batch_entries"] = default_entries
    resolved = dict(options)
    resolved["preset"] = preset
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


def apply_capacity(options: dict, *, segment_bytes: int,
                   num_layers: int) -> dict:
    """把每rank带单位的 ``capacity`` payload预算换算成内部 ``num_chunks``。

    当前公式只用层数和每层段字节数，不依赖设备组或盘数。不扣除对象层
    加入的槽位前缀（默认 128 KiB）、预建文件组取整或文件系统开销，因而
    不是严格的物理总占用上限。服务总预算应由调用者先除以 TP，再传入。
    """
    if "allocator_enabled" in options:
        raise ValueError("allocator_enabled 未被消费，已删除")
    preset = options.get("preset")
    if isinstance(preset, dict):
        removed = {"prp_cache_capacity", "handle_cache_capacity", "handle_cache_l2_capacity"}.intersection(preset)
        if removed:
            raise ValueError(f"cache容量由runtime管理，不支持preset参数 {sorted(removed)}")
    removed = {"capacity_bytes", "num_chunks"}.intersection(options)
    if removed:
        raise ValueError(f"vLLM store options 不支持 {sorted(removed)}；请设置 capacity，例如 '64 GiB'")
    if "capacity" not in options:
        return options
    options = dict(options)
    capacity_bytes = parse_capacity(options.pop("capacity"))
    if segment_bytes <= 0 or num_layers <= 0:
        raise ValueError(
            f"capacity 换算需要正几何，got segment_bytes="
            f"{segment_bytes}, num_layers={num_layers}"
        )

    per_slot = segment_bytes * num_layers
    num_chunks = capacity_bytes // per_slot
    if num_chunks < 1:
        raise ValueError(
            f"capacity({capacity_bytes} B) 小于单槽位payload"
            f"({per_slot} 字节 × 槽位)"
        )
    options["num_chunks"] = int(num_chunks)
    return options


def apply_device_groups(options: dict, *, rank: str, tp_size: int) -> dict:
    """按 rank 把 preset.device_groups 展开为 preset.devices。

    device_groups是部署映射，不是IO并行度：设备ID来自daemon配置。
    ``device_groups=[[0, 1], [2, 3]]`` 在TP4时让rank0-1使用设备0-1，
    rank2-3使用设备2-3；组数必须整除tp_size，每组rank数=tp_size/组数。
    P2P、ACL和已有阵列成员必须由部署者审核，不能从设备编号自动猜测。
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
    parallel = getattr(vllm_config, "parallel_config", None)
    tp = getattr(parallel, "tensor_parallel_size", 1)
    dcp = int(getattr(parallel, "decode_context_parallel_size", 1) or 1)
    interleave = int(getattr(parallel, "cp_kv_cache_interleave_size", 1) or 1)
    return "|".join([
        "v2",
        f"model={model}",
        f"dtype={dtype}",
        f"tp={tp}",
        f"dcp={dcp}",
        f"cp_interleave={interleave}",
        f"chunk_kv_bytes={extra.get('chunk_kv_bytes')}",
        f"chunk_tokens={extra.get('chunk_tokens')}",
        *([f"entity_pages={extra['kv_group_page_bytes']}", "kv_layout=NHD-v1"]
          if "kv_group_page_bytes" in extra else []),
        *(["replica_io=minimax-dcp-aware-v4"] if extra.get("replica_io", False) else []),
    ])

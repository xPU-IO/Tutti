"""Experimental replica IO layout, derived without importing the data plane."""
import os


def validate_layerwise_execution(vllm_config):
    if getattr(vllm_config.model_config, "enforce_eager", False):
        return
    compilation = getattr(vllm_config, "compilation_config", None)
    graph = getattr(compilation, "cudagraph_mode", None)
    mode = getattr(compilation, "mode", None)
    if getattr(graph, "name", graph) not in ("NONE", "PIECEWISE") or mode != 3:
        raise ValueError(
            "CUDA graph mode unsupported: 独立KV实体支持 enforce_eager=True 或 "
            "mode=3、cudagraph_mode=NONE/PIECEWISE"
        )
    if os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "1":
        raise ValueError("独立KV实体编译要求 VLLM_USE_BREAKABLE_CUDAGRAPH=0")


def validate_replica_io(vllm_config, extra):
    enabled = extra.get("replica_io", False)
    if not isinstance(enabled, bool):
        raise ValueError("replica_io must be a bool")
    if not enabled:
        return
    parallel = vllm_config.parallel_config
    dcp_size = int(getattr(parallel, "decode_context_parallel_size", 1) or 1)
    if (parallel.tensor_parallel_size != 8 or
            any(getattr(parallel, field, 1) != 1 for field in (
                "data_parallel_size", "pipeline_parallel_size",
                "prefill_context_parallel_size")) or
            dcp_size not in (1, 2)):
        raise ValueError("replica_io requires single-node TP8, DP/PP/PCP=1 and DCP=1/2")
    if dcp_size > 1:
        raise ValueError(
            "DCP=2 shards the replicated MiniMax KV heads by context; "
            "disable replica_io because those rank pages are no longer duplicates"
        )
    model = vllm_config.model_config
    hf = getattr(model, "hf_config", None)
    text = getattr(model, "hf_text_config", None)
    if text is None:
        text = getattr(hf, "text_config", hf)
    if (getattr(hf, "model_type", "") not in ("minimax_m3", "minimax_m3_vl") or
            getattr(text, "num_key_value_heads", None) != 4 or
            getattr(text, "head_dim", None) != 128):
        raise ValueError("replica_io initially supports MiniMax M3 with 4 KV heads, head_dim=128")
    validate_layerwise_execution(vllm_config)
    if extra.get("store", {}).get("type") != "tutti_nvme":
        raise ValueError("replica_io requires tutti_nvme")
    names = extra.get("kv_group_layer_names", ())
    if ("kv_layer_page_bytes" not in extra or len(names) != 117 or
            sum(name.endswith(".index_cache") for name in names) != 57):
        raise ValueError("replica_io requires the registered 60 main KV and 57 index entities")


def replica_groups_for(extra, rank):
    """Return the physical owner group for each MiniMax cache entity.

    TP8 with four global KV heads has two physical copies of each main KV
    head. vLLM's shard mapping duplicates adjacent ranks, so main entities
    use (0,1), (2,3), (4,5), (6,7). The index projection is replicated on all
    ranks and therefore uses one eight-rank owner group.
    """
    if not extra.get("replica_io", False):
        return ()
    rank = int(rank)
    if not 0 <= rank < 8:
        raise ValueError(f"replica rank must be in [0, 8), got {rank!r}")
    dcp_size = int(extra.get("dcp_size", 1))
    main_group = (rank // 2 * 2, rank // 2 * 2 + 1)
    return tuple(
        tuple(range(8)) if name.endswith(".index_cache") else main_group
        for name in extra["kv_group_layer_names"]
    )

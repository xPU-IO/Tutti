import sys
from types import SimpleNamespace as NS

import pytest

from tutti.integration.vllm.geometry import key_namespace
from tutti.integration.vllm.replica_layout import replica_groups_for, validate_replica_io


def config():
    text = NS(num_key_value_heads=4, head_dim=128)
    return NS(parallel_config=NS(tensor_parallel_size=8),
              model_config=NS(hf_config=NS(model_type="minimax_m3_vl", text_config=text),
                              enforce_eager=True, model="minimax"),
              cache_config=NS(cache_dtype="fp8_e4m3"))


def extra():
    names = tuple([f"layers.{i}.attn" for i in range(60)] +
                  [f"layers.{i}.attn.index_cache" for i in range(3, 60)])
    return dict(replica_io=True, load_mode="sync", store={"type": "tutti_nvme"},
                kv_group_layer_names=names, kv_layer_page_bytes=(32768,) * 117)


def test_disabled_does_not_require_model_or_cuda():
    validate_replica_io(NS(), {})
    assert replica_groups_for({}, 0) == ()


def test_minimax_replica_members_and_namespace(monkeypatch):
    monkeypatch.delenv("TUTTI_LOAD_MODE", raising=False)
    monkeypatch.delenv("TUTTI_ASYNC_COMPLETION", raising=False)
    c, e = config(), extra()
    validate_replica_io(c, e)
    for rank in range(8):
        groups = replica_groups_for(e, rank)
        assert len(groups) == 117
        expected_main = (rank // 2 * 2, rank // 2 * 2 + 1)
        assert all(group == expected_main for group in groups[:60])
        assert all(group == tuple(range(8)) for group in groups[60:])
    assert key_namespace(c, e) != key_namespace(c, dict(e, replica_io=False))


@pytest.mark.parametrize("mode", ["auto", "async"])
def test_replica_background_load_mode_allowed(monkeypatch, mode):
    monkeypatch.setenv("TUTTI_LOAD_MODE", mode)
    monkeypatch.delenv("TUTTI_ASYNC_COMPLETION", raising=False)
    validate_replica_io(config(), extra())


def test_async_completion_allowed(monkeypatch):
    monkeypatch.setenv("TUTTI_LOAD_MODE", "sync")
    monkeypatch.setenv("TUTTI_ASYNC_COMPLETION", "1")
    validate_replica_io(config(), extra())


@pytest.mark.parametrize("change", ["tp", "model", "heads", "eager", "entities", "type"])
def test_unknown_replica_layout_rejected(monkeypatch, change):
    monkeypatch.setenv("TUTTI_LOAD_MODE", "sync")
    monkeypatch.delenv("TUTTI_ASYNC_COMPLETION", raising=False)
    c, e = config(), extra()
    if change == "tp":
        c.parallel_config.tensor_parallel_size = 4
    elif change == "model":
        c.model_config.hf_config.model_type = "hy_v3"
    elif change == "heads":
        c.model_config.hf_config.text_config.num_key_value_heads = 8
    elif change == "eager":
        c.model_config.enforce_eager = False
    elif change == "entities":
        e["kv_group_layer_names"] = e["kv_group_layer_names"][:-1]
    else:
        e["replica_io"] = "true"
    with pytest.raises(ValueError):
        validate_replica_io(c, e)


def test_simplified_equal_page_specs_keep_independent_entities():
    from tutti.integration.vllm.geometry import resolve_geometry
    e = extra()
    names = e["kv_group_layer_names"]
    spec = NS(block_size=128, page_size_bytes=32768)
    cache = NS(kv_cache_groups=[NS(layer_names=names, kv_cache_spec=spec)])
    resolved = resolve_geometry(dict(e, chunk_tokens=256), cache)
    assert resolved["kv_layer_page_bytes"] == (32768,) * 117
    validate_replica_io(config(), resolved)


def test_replica_allows_explicit_compilation_without_graph(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    c = config()
    c.model_config.enforce_eager = False
    c.compilation_config = NS(mode=3, cudagraph_mode=NS(name="NONE"))
    validate_replica_io(c, extra())


@pytest.mark.parametrize("graph", ["FULL", None])
def test_replica_rejects_unsupported_graph_capture(monkeypatch, graph):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    c = config()
    c.model_config.enforce_eager = False
    c.compilation_config = NS(mode=3, cudagraph_mode=graph)
    with pytest.raises(ValueError, match="CUDA graph"):
        validate_replica_io(c, extra())


def test_replica_allows_piecewise_capture(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    c = config()
    c.model_config.enforce_eager = False
    c.compilation_config = NS(mode=3, cudagraph_mode=NS(name="PIECEWISE"))
    validate_replica_io(c, extra())


def test_dcp2_disables_replica_io_and_preserves_unique_head_shards():
    c, e = config(), extra()
    c.parallel_config.decode_context_parallel_size = 2
    with pytest.raises(ValueError, match="disable replica_io"):
        validate_replica_io(c, e)
    e["replica_io"] = False
    validate_replica_io(c, e)


def test_configuration_module_does_not_import_data_plane():
    import subprocess
    result = subprocess.run([sys.executable, "-c",
                             "import sys; import tutti.integration.vllm.replica_layout; "
                             "assert 'torch' not in sys.modules; "
                             "assert 'tutti_runtime' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

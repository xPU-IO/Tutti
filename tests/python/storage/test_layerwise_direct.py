from types import SimpleNamespace

import pytest
import torch

from tutti.storage.tutti_nvme.layerwise_direct import page_geometry, TuttiLayerwiseDirectBackend
from tutti.storage.tutti_nvme.store import DirectAdmissionError


def test_page_geometry_covers_packed_kv_and_key_only_index():
    main = torch.empty((8, 128, 1, 256), dtype=torch.bfloat16).permute(0, 2, 1, 3)
    index = torch.empty((8, 128, 128), dtype=torch.bfloat16)
    g = page_geometry(main, 65536)
    assert g.page_bytes == g.block_stride == 65536
    assert g.num_blocks == 8
    g = page_geometry(index, 32768)
    assert g.page_bytes == g.block_stride == 32768


@pytest.mark.parametrize("group", [2, 4])
def test_logical_page_groups_contiguous_kernel_blocks_without_copy(group):
    main = torch.empty((16, 128, 1, 256), dtype=torch.uint8).permute(0, 2, 1, 3)
    index = torch.empty((16, 128, 128), dtype=torch.bfloat16)
    for tensor in (main, index):
        geometry = page_geometry(tensor, 32768 * group)
        assert geometry.page_bytes == geometry.block_stride == 32768 * group
        assert geometry.num_blocks == 16 // group
        backend = TuttiLayerwiseDirectBackend(SimpleNamespace())
        backend._layer_pages = (geometry,)
        backend._layer_tickets = [11]
        backend._geometry = SimpleNamespace(segment_bytes=32768 * group * 2)
        entry = SimpleNamespace(target_ticket=7, target_offset=0)
        assert backend._page_requests(0, 2, 1, entry, SimpleNamespace(direction="read")) == [
            (7, 32768 * group, 11, 2 * 32768 * group, 32768 * group, "read")]


@pytest.mark.parametrize("page_bytes", [49152, 131072])
def test_logical_page_rejects_non_integral_or_incomplete_kernel_groups(page_bytes):
    with pytest.raises(DirectAdmissionError):
        page_geometry(torch.empty((6, 128, 256), dtype=torch.uint8), page_bytes)


def test_strided_or_unaligned_page_is_rejected():
    with pytest.raises(DirectAdmissionError):
        page_geometry(torch.empty((8, 128, 256))[:, :, ::2], 65536)
    with pytest.raises(DirectAdmissionError):
        page_geometry(torch.empty((8, 1024), dtype=torch.uint8), 1024)


def test_ambiguous_two_block_split_and_hnd_layouts_are_rejected():
    with pytest.raises(DirectAdmissionError, match="block-axis-0"):
        page_geometry(torch.empty((2, 2, 128, 1, 128), dtype=torch.bfloat16), 65536)
    with pytest.raises(DirectAdmissionError, match="packed NHD"):
        page_geometry(torch.empty((8, 2, 128, 128), dtype=torch.bfloat16), 65536)


def test_layerwise_requests_use_entity_ticket_and_fixed_disk_width():
    store = SimpleNamespace(_layer_page_bytes=(65536, 32768))
    backend = TuttiLayerwiseDirectBackend(store)
    backend._layer_pages = (page_geometry(torch.empty((8, 1, 128, 256), dtype=torch.bfloat16), 65536),
                            page_geometry(torch.empty((8, 128, 128), dtype=torch.bfloat16), 32768))
    backend._layer_tickets = [11, 22]
    backend._geometry = SimpleNamespace(segment_bytes=131072)
    entry = SimpleNamespace(target_ticket=7, target_offset=262144)
    plan = SimpleNamespace(direction="read")
    assert backend._page_requests(0, 3, 1, entry, plan) == [
        (7, 327680, 11, 196608, 65536, "read"),
    ]
    assert backend._page_requests(1, 3, 1, entry, plan) == [
        (7, 425984, 22, 98304, 32768, "read"),
    ]


def test_full_prp_prebuild_registers_each_physical_page(monkeypatch):
    from tutti.storage.tutti_nvme.layerwise_direct import TuttiLayerwiseDirectBackend

    class Tensor:
        shape = (4, 128, 128)
        stride = lambda self: (16384, 128, 1)
        is_cuda = True
        get_device = lambda self: 0
        data_ptr = lambda self: 65536
        numel = lambda self: 4 * 128 * 128
        element_size = lambda self: 2

    class Runtime:
        def __init__(self):
            self.calls = []

        def register_memory(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return len(self.calls)

    monkeypatch.setenv("TUTTI_PRP_MODE", "full_prebuild")
    runtime = Runtime()
    store = SimpleNamespace(
        _layer_page_bytes=(32768,), _runtime=runtime, _rank_id=0,
    )
    backend = TuttiLayerwiseDirectBackend(store)
    backend._check_runtime_capacity = lambda *args, **kwargs: None
    backend.register_paged_caches(
        {"index": Tensor()}, num_layers=1, blocks_per_chunk=1,
        chunk_tokens=128, segment_bytes=32768,
    )
    assert runtime.calls[0][1]["io_granularity"] == 32768


@pytest.mark.parametrize("fail_last", [False, True])
def test_layerwise_close_releases_all_tickets_after_drain_error(fail_last):
    calls = []

    def release(ticket):
        calls.append(ticket)
        if ticket == 22 and fail_last and calls.count(22) == 1:
            raise RuntimeError("BUSY")

    def drain(chunks):
        raise RuntimeError("drain failed")

    store = SimpleNamespace(_runtime=SimpleNamespace(unregister_memory=release),
                            _inflight_by_chunk={}, _wait_chunk_io=drain)
    backend = TuttiLayerwiseDirectBackend(store)
    backend._layer_tickets = [11, 22]
    backend._memory_ticket = 11
    backend._layer_tensors = (object(),)
    with pytest.raises(RuntimeError, match="BUSY" if fail_last else "drain failed"):
        backend.close()
    if fail_last:
        assert backend._layer_tensors
        assert not backend._closed
        store._wait_chunk_io = lambda _: None
        backend.close()
        assert calls == [11, 22, 22]
    else:
        assert calls == [11, 22]
    assert backend._layer_tickets == []
    assert backend._layer_tensors == ()
    backend.close()


def test_hook_install_rolls_back_if_second_registration_fails():
    from tutti.integration.vllm.minimax import install_cache_callbacks

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.indexer = SimpleNamespace(index_cache=SimpleNamespace(prefix="index"))

        def register_forward_hook(self, hook):
            raise RuntimeError("hook install failed")

    module = Attention()
    connector = SimpleNamespace(_impl=SimpleNamespace(_kv_group_layer_names=("index", "main")))
    config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={"main": module}))
    with pytest.raises(RuntimeError, match="hook install failed"):
        install_cache_callbacks(config, connector, {"index": object(), "main": object()})
    assert not module._forward_pre_hooks


def test_callback_hooks_wait_both_caches_then_save_both(monkeypatch):
    from tutti.integration.vllm.minimax import install_cache_callbacks
    import vllm.forward_context

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.indexer = SimpleNamespace(index_cache=SimpleNamespace(prefix="index"))

        def forward(self, x):
            events.append("compute")
            return x

    events = []
    module = Attention()
    caches = {"index": object(), "main": object()}
    connector = SimpleNamespace(_impl=SimpleNamespace(_kv_group_layer_names=tuple(caches)),
        has_connector_metadata=lambda: True,
        wait_for_layer_load=lambda key: events.append(("wait", key)),
        save_kv_layer=lambda key, tensor, meta: events.append(("save", key)))
    monkeypatch.setattr(vllm.forward_context, "get_forward_context", lambda: SimpleNamespace(attn_metadata={}))
    config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={"main": module}))
    handles = install_cache_callbacks(config, connector, caches)
    module(torch.tensor(1))
    assert events == [("wait", "index"), ("wait", "main"), "compute", ("save", "index"), ("save", "main")]
    for handle in handles:
        handle.remove()
    events.clear()
    module(torch.tensor(1))
    assert events == ["compute"]


def test_compiled_projection_replay_preserves_runtime_kv_callbacks(monkeypatch):
    import vllm.forward_context
    from tutti.integration.vllm.minimax import install_cache_callbacks
    from torch._dynamo.backends import registry
    events, graphs = [], []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    monkeypatch.setattr(registry, "lookup_backend", lambda name: backend)
    class Projection(torch.nn.Module):
        def forward(self, x):
            return x.sin() + 1
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.indexer = SimpleNamespace(index_cache=SimpleNamespace(prefix="index"))
            self.qkv_proj = Projection()
            self.o_proj = Projection()
        def forward(self, x):
            return self.o_proj(self.qkv_proj(x))
    module = Attention()
    originals = (module.qkv_proj.forward, module.o_proj.forward)
    connector = SimpleNamespace(_impl=SimpleNamespace(_kv_group_layer_names=("index", "main")),
        has_connector_metadata=lambda: True,
        wait_for_layer_load=lambda key: events.append(("wait", key)),
        save_kv_layer=lambda key, tensor, meta: events.append(("save", key)))
    config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(static_forward_context={"main": module}))
    monkeypatch.setattr(vllm.forward_context, "get_forward_context", lambda: SimpleNamespace(attn_metadata={}))
    handles = install_cache_callbacks(config, connector, {"index": object(), "main": object()})
    try:
        for size in (4, 7, 4):
            value = torch.arange(size, dtype=torch.float32)
            expected = (value.sin() + 1).sin() + 1
            torch.testing.assert_close(module(value), expected)
        assert graphs
        assert events == [("wait", "index"), ("wait", "main"),
                          ("save", "index"), ("save", "main")] * 3
    finally:
        for handle in handles:
            handle.remove()
    assert (module.qkv_proj.forward, module.o_proj.forward) == originals

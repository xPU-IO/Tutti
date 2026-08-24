"""80 层跨层池全链路往返（层序解析回归的集成面）。

背景：层名解析曾按调用序轮转，第 i 层落 2i+1 (mod N)——层错位使
save/load 数据流错层。本用例以 80 层（Hy3 几何）逐层数据流验证：
层号全集完整、逐层写入互不覆盖、读回逐层内容一致。

Run:
    cd integration/vllm-connector && \
    /data/home/ryeqiu/tutti-env/bin/python -m pytest tests/overlap -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from index.chunk_index import derive_io_key
from stores.tutti_nvme.layout import decode_io_key
from stores.memory import MemoryKVStore
from tests.adapter.test_adapter import (
    BLOCK_SIZE,
    CHUNK_TOKENS,
    MAX_WAVE,
    SEG,
    _TensorAdaptingStore,
    _fake_request,
    _make_harness,
    _sched_output,
)

NL = 80  # 仿真 Hy3 层数


def test_cross_pool_80_layer_roundtrip():
    h = _make_harness(num_layers=NL, register_per_layer=False)
    pool = torch.zeros(4, NL, BLOCK_SIZE, 2, 8, dtype=torch.float16)
    h.worker.register_cross_layers_kv_cache(pool, attn_backend=None)

    # ---- prefill：80 层 wait/save 交织写入 ----
    prompt = list(range(2 * CHUNK_TOKENS))
    req = _fake_request("r1", prompt)
    h.scheduler.update_state_after_alloc(req, object(), 0)
    meta = h.scheduler.build_connector_meta(_sched_output(new_reqs=[
        SimpleNamespace(
            req_id="r1", prompt_token_ids=prompt,
            block_ids=list(range(len(prompt) // BLOCK_SIZE)),
        )
    ]))
    h.worker.bind_connector_metadata(meta)
    h.worker.start_load_kv(None)
    h.hooks.bind_window(h.worker._impl.window)
    keys, _ = h.engine.hash_keys(prompt)
    names = [f"model.layers.{i}.self_attn.attn" for i in range(NL)]
    patterns = {}
    for idx, name in enumerate(names):
        h.worker.wait_for_layer_load(name)
        for chunk in range(2):
            # 每层不同内容：层号与 chunk 共同编码
            seg = _segment_bytes(chunk, idx)
            h.hooks.source[(keys[chunk], idx)] = seg
            patterns[(keys[chunk], idx)] = seg
        h.worker.save_kv_layer(name)
    h.worker.wait_for_save()

    # 层号全集恰 0..79（每 chunk 各一次 → 每层恰 2 条）
    scanned = list(h.store.scan())
    assert len(scanned) == 2 * NL
    layers = sorted(decode_io_key(k)[1] for k in scanned)
    assert layers == sorted(
        [layer for _ in range(2) for layer in range(NL)]
    )

    # ---- 命中加载：同前缀新请求，逐层读回 ----
    req2 = _fake_request("r2", prompt)
    matched, _ = h.scheduler.get_num_new_matched_tokens(req2, 0)
    assert matched == 2 * CHUNK_TOKENS
    h.scheduler.update_state_after_alloc(req2, object(), matched)
    meta2 = h.scheduler.build_connector_meta(_sched_output(new_reqs=[
        SimpleNamespace(
            req_id="r2", prompt_token_ids=prompt, block_ids=[8, 9]
        )
    ]))
    h.worker.bind_connector_metadata(meta2)
    h.worker.start_load_kv(None)
    for name in names:
        h.worker.wait_for_layer_load(name)

    # 逐层逐 chunk 读回内容与写入一致（层错位会在此暴露）
    for chunk in range(2):
        for layer in range(NL):
            assert h.hooks.sink[(keys[chunk], layer)] == patterns[(keys[chunk], layer)]
    assert len(h.hooks.scatter_calls) == NL  # 每层恰一批


def _segment_bytes(chunk: int, layer: int) -> bytes:
    """层与 chunk 共同编码的确定性段内容。"""
    return bytes([(layer * 7 + chunk * 13 + i) % 251 for i in range(SEG)])


def test_layer_span_predisposes_full_size(tmp_path):
    """bind 注入层宽后，数据文件首写即全尺寸（票据稳定不再重开）。

    回归背景：文件随层逐层增长时，传输层票据因尺寸过期被反复弃置
    重开，层多时票据池耗尽（open: handle cache pool exhausted）。
    """
    from stores.tutti_nvme.layout import Layout

    layout = Layout(tmp_path, segment_bytes=SEG)
    layout.ensure_dirs()
    layout.set_layer_span(80)
    io_keys = [derive_io_key(b"\x42" * 16, 0)]  # 首层写入
    layout.prepare_put(io_keys, capacity_chunks=8)
    size = layout.chunk_file(b"\x42" * 16).stat().st_size
    assert size == 80 * SEG  # 首写即全尺寸

    # 未设层宽时保持按需增长（兼容路径）
    layout2 = Layout(tmp_path / "spanless", segment_bytes=SEG)
    layout2.ensure_dirs()
    layout2.prepare_put([derive_io_key(b"\x43" * 16, 2)], 8)
    assert layout2.chunk_file(b"\x43" * 16).stat().st_size == 3 * SEG

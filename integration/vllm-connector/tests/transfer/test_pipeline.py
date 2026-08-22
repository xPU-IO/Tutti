# SPDX-License-Identifier: Apache-2.0
"""GPU 管线用例：真搬运函数 + MemoryKVStore + 真显存池的 engine 全链路往返。

覆盖 staged 路径完整数据面：paged 池 → staging 槽 → store → staging 槽
→ paged 池，逐字节一致。块号取非连续分布，验证块表语义的正确性。

Run（GPU required; skips visibly without one）:
    cd integration/vllm-connector && \
    /data/home/ryeqiu/tutti-env/bin/python -m pytest tests/transfer -v
"""

import pytest
import torch

from adapter.worker import PagedTransferHooks
from engine.core import KVEngine
from engine.staging import RingWindow
from index.chunk_index import derive_io_key
from stores.memory import MemoryKVStore
from tutti_kv_transfer import EngineKVFormat, discover_engine_format

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device not available"
)

NUM_LAYERS = 3
BLOCK_SIZE = 16
CHUNK_TOKENS = 32
BPC = CHUNK_TOKENS // BLOCK_SIZE
MAX_WAVE = 4
NUM_CHUNKS = 2
NUM_BLOCKS = 16
# 非连续块号：两个 chunk 各占两块，故意跳号分布
BLOCK_TABLES = [[5, 9], [2, 7]]
PROTECTED_BLOCKS = sorted({b for table in BLOCK_TABLES for b in table})
SENTINEL = 777.0


def _fill_deterministic(pool: torch.Tensor) -> None:
    """按展平坐标撒确定性图案（GQA 与 MLA 通用）。"""
    flat = torch.arange(pool.numel(), device=pool.device, dtype=torch.int64)
    pattern = (flat * 7 + 13) % 977
    pool.copy_(pattern.reshape(pool.shape).to(pool.dtype))


def _block_view(pool: torch.Tensor, block: int, use_mla: bool) -> torch.Tensor:
    """取全部层在指定块上的切片。"""
    if use_mla:
        return pool[:, block]
    return pool[:, :, block]


def _run_full_chain(pool: torch.Tensor, use_mla: bool,
                    per_token_bytes: int) -> None:
    """engine 全链路：store_layer 落盘 → 清池 → load_layer 读回比对。"""
    reference = pool[0]
    fmt = discover_engine_format(reference, use_mla=use_mla)
    segment_bytes = per_token_bytes * CHUNK_TOKENS
    assert segment_bytes % 4096 == 0  # store 注册粒度的对齐前提

    keys = [bytes([0xA0 + i]) * 16 for i in range(NUM_CHUNKS)]
    slots = 2 * MAX_WAVE
    # 页锁定主存 staging：搬运函数支持；store 侧经共享存储的数组视图登记
    staging = torch.empty(slots * segment_bytes, dtype=torch.uint8,
                          pin_memory=True)
    window = RingWindow(staging.numpy(), slots, segment_bytes)
    store = MemoryKVStore(segment_bytes, num_chunks=16)
    engine = KVEngine(
        {
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_kv_bytes": segment_bytes * NUM_LAYERS,
            "max_chunks_per_wave": MAX_WAVE,
        },
        store,
    )
    hooks = PagedTransferHooks(
        lambda idx: pool[idx], staging, segment_bytes,
        CHUNK_TOKENS, BLOCK_SIZE, fmt,
    )
    engine.bind({}, window, NUM_LAYERS, BPC,
                gather_fn=hooks.gather, scatter_fn=hooks.scatter)

    original = pool.clone()

    # ---- 写链：paged → staging 槽 → store ----
    for layer in range(NUM_LAYERS):
        engine.store_layer(keys, layer, BLOCK_TABLES).wait()
    expected_io_keys = {
        derive_io_key(k, layer)
        for k in keys for layer in range(NUM_LAYERS)
    }
    assert set(store.scan()) == expected_io_keys

    # ---- 读链：store → staging 槽 → paged ----
    pool.fill_(SENTINEL)
    for layer in range(NUM_LAYERS):
        engine.load_layer(keys, layer, BLOCK_TABLES).wait()

    # ---- 比对：受保护块逐字节还原，其余块保持哨兵 ----
    for block in range(NUM_BLOCKS):
        got = _block_view(pool, block, use_mla).reshape(-1)
        want = _block_view(original, block, use_mla).reshape(-1)
        if block not in PROTECTED_BLOCKS:
            want = torch.full_like(want, SENTINEL)
        assert torch.equal(
            got.view(torch.uint8).cpu(), want.view(torch.uint8).cpu()
        ), f"block {block} 内容不符"

    engine.close()


@requires_cuda
def test_full_chain_gqa_fa_nhd():
    """GQA 几何（FA_NHD [2, nb, bs, nh, hs]）全链路逐字节一致。"""
    pool = torch.zeros(
        NUM_LAYERS, 2, NUM_BLOCKS, BLOCK_SIZE, 2, 64,
        dtype=torch.float16, device="cuda",
    )
    _fill_deterministic(pool)
    fmt = discover_engine_format(pool[0], use_mla=False)
    assert fmt == EngineKVFormat.FA_NHD
    per_token = 2 * 2 * 2 * 64 * 2  # K/V × heads × dim × fp16
    _run_full_chain(pool, use_mla=False, per_token_bytes=per_token)


@requires_cuda
def test_full_chain_mla():
    """MLA 几何（[nb, bs, hs] 单张量）全链路逐字节一致。"""
    pool = torch.zeros(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, 576,
        dtype=torch.bfloat16, device="cuda",
    )
    _fill_deterministic(pool)
    fmt = discover_engine_format(pool[0], use_mla=True)
    assert fmt == EngineKVFormat.MLA
    per_token = 576 * 2  # hs × bf16
    _run_full_chain(pool, use_mla=True, per_token_bytes=per_token)

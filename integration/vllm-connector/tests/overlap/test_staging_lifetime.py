"""CUDA staging 生命周期探针：三层读取回绕时消费完成事件必须生效。"""

import pytest
import torch

from engine.core import KVEngine
from engine.staging import RingWindow
from index.chunk_index import derive_io_key
from stores.memory import MemoryKVStore


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device not available"
)

NUM_LAYERS = 3
CHUNK_TOKENS = 16
SEGMENT_BYTES = 4096


class GuardedMemoryStore(MemoryKVStore):
    """复用第一波槽位前检查其消费事件已完成。"""

    def __init__(self):
        super().__init__(SEGMENT_BYTES, num_chunks=8)
        self.get_count = 0
        self.protected_event = None

    def get_batch(self, batch):
        self.get_count += 1
        if self.get_count == 3:
            assert self.protected_event is not None
            assert self.protected_event.query(), "scatter 完成前发生槽位覆盖"
        return super().get_batch(batch)


class DelayedScatterHooks:
    """在目的侧消费前插入设备工作，放大槽位覆盖窗口。"""

    def __init__(self, staging, destinations, store):
        self._staging = staging
        self._destinations = destinations
        self._store = store

    def scatter(self, keys, layer_idx, block_tables, slots):
        torch.cuda._sleep(1_000_000_000)
        slot = slots[0]
        source = self._staging.narrow(
            0, slot * SEGMENT_BYTES, SEGMENT_BYTES
        )
        self._destinations[layer_idx].copy_(source, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._store.protected_event = event
        return event


@requires_cuda
def test_three_layer_load_wrap_keeps_staging_until_scatter_finishes():
    """三波跨层读取复用槽位时，第一波数据保持逐字节一致。"""
    destinations = torch.zeros(
        NUM_LAYERS, SEGMENT_BYTES, dtype=torch.uint8, device="cuda"
    )
    expected = []
    for layer in range(NUM_LAYERS):
        value = (layer + 1) * 29
        expected.append(torch.full(
            (SEGMENT_BYTES,), value, dtype=torch.uint8
        ))

    staging = torch.empty(
        2 * SEGMENT_BYTES, dtype=torch.uint8, pin_memory=True
    )
    window = RingWindow(staging.numpy(), 2, SEGMENT_BYTES)
    store = GuardedMemoryStore()
    engine = KVEngine(
        {
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_kv_bytes": SEGMENT_BYTES * NUM_LAYERS,
            "max_chunks_per_wave": 1,
        },
        store,
    )
    hooks = DelayedScatterHooks(staging, destinations, store)
    engine.bind({}, window, NUM_LAYERS, blocks_per_chunk=1,
                scatter_fn=hooks.scatter)

    keys = [bytes([0xA0 + i]) * 16 for i in range(NUM_LAYERS)]
    seed = bytearray(NUM_LAYERS * SEGMENT_BYTES)
    seed_id = store.register_buffer(seed, SEGMENT_BYTES)
    assert seed_id is not None
    for layer, key in enumerate(keys):
        begin = layer * SEGMENT_BYTES
        seed[begin:begin + SEGMENT_BYTES] = expected[layer].numpy().tobytes()
        store.put_batch([
            (derive_io_key(key, layer), seed_id, begin),
        ])

    block_tables = [[[0]], [[1]], [[2]]]
    engine.load_layer([keys[0]], 0, block_tables[0])
    engine.load_layer([keys[1]], 1, block_tables[1])
    engine.load_layer([keys[2]], 2, block_tables[2])
    engine.wait_idle()

    for layer in range(NUM_LAYERS):
        assert torch.equal(destinations[layer].cpu(), expected[layer])
    engine.close()

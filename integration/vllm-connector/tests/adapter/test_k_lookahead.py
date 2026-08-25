"""Bounded worker layer look-ahead scheduling contracts."""

from __future__ import annotations

from adapter.worker import WorkerImpl
from engine.staging import RingWindow


class _Handle:
    def __init__(self, layer, log):
        self.layer = layer
        self._log = log

    def wait(self):
        self._log.append(("wait", self.layer))


class _Engine:
    def __init__(self):
        self.calls = []
        self.waits = []

    def load_layer(self, keys, layer_idx, blocks):
        self.calls.append((layer_idx, tuple(keys), tuple(blocks)))
        return _Handle(layer_idx, self.waits)

    def unpin(self, keys):
        self.waits.append(("unpin", tuple(keys)))


def _worker(k=None):
    engine = _Engine()
    worker = WorkerImpl(engine, lookahead_k=k)
    worker._num_layers = 5
    worker._load_keys = [b"k0", b"k1"]
    worker._load_block_tables = [[10], [11]]
    worker._pinned = True
    return worker, engine


def test_default_k_is_one_and_preserves_stepwise_submission():
    worker, engine = _worker()

    worker._prefetch_load_layers(0)
    assert [call[0] for call in engine.calls] == [0]
    worker.wait_for_layer_load("model.layers.0.self_attn")
    assert [call[0] for call in engine.calls] == [0, 1]
    assert engine.waits[:1] == [("wait", 0)]


def test_k_three_keeps_bounded_window_and_refills_tail():
    worker, engine = _worker(3)

    worker._prefetch_load_layers(0)
    assert [call[0] for call in engine.calls] == [0, 1, 2]
    worker.wait_for_layer_load("model.layers.0.self_attn")
    assert [call[0] for call in engine.calls] == [0, 1, 2, 3]
    worker.wait_for_layer_load("model.layers.1.self_attn")
    assert [call[0] for call in engine.calls] == [0, 1, 2, 3, 4]
    assert [call[0] for call in engine.calls].count(2) == 1


def test_duplicate_wait_does_not_resubmit_or_unpin_early():
    worker, engine = _worker(3)
    worker._prefetch_load_layers(0)

    worker.wait_for_layer_load("model.layers.1.self_attn")
    worker.wait_for_layer_load("model.layers.1.self_attn")
    assert [call[0] for call in engine.calls] == [0, 1, 2, 4]
    assert not any(item[0] == "unpin" for item in engine.waits)


class _WindowEvent:
    def __init__(self):
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1


def test_extended_ring_window_holds_three_waves_before_reuse():
    window = RingWindow(
        bytearray(6), 6, 1, capacity_per_wave=1
    )
    events = [_WindowEvent() for _ in range(4)]
    waves = [window.acquire(1)[0] for _ in range(3)]
    for wave, event in zip(waves, events):
        window.complete(wave, event)
    fourth, slots = window.acquire(1)
    assert (waves, slots, fourth) == ([0, 1, 2], [3], 3)
    assert [event.wait_count for event in events[:3]] == [0, 0, 0]
    window.complete(fourth, events[3])
    for expected in (4, 5):
        wave, _ = window.acquire(1)
        assert wave == expected
    sixth, _ = window.acquire(1)
    assert sixth == 6
    assert events[0].wait_count == 1

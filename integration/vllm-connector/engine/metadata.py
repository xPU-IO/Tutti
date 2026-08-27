"""Scheduler-side semantic index backed only by durable store metadata."""

from __future__ import annotations

from typing import Sequence

from index.chunk_index import (
    ChunkIndex,
    StorePlan,
    chunk_key_of,
    layer_of,
)

_IO_KEY_BYTES = 18


class SchedulerMetadataIndex:
    """ChunkIndex plus marker reconciliation, with no data-plane methods."""

    def __init__(self, config: dict, store) -> None:
        self._chunk_tokens = _positive_int(config, "chunk_tokens")
        self._chunk_kv_bytes = _positive_int(config, "chunk_kv_bytes")
        self._max_chunks_per_wave = _positive_int(
            config, "max_chunks_per_wave"
        )
        self._num_layers = _positive_int(config, "num_layers")
        raw_namespace = config.get("key_namespace")
        if raw_namespace is None:
            namespace = b""
        elif isinstance(raw_namespace, str):
            namespace = raw_namespace.encode("utf-8")
        elif isinstance(raw_namespace, (bytes, bytearray)):
            namespace = bytes(raw_namespace)
        else:
            raise ValueError("config['key_namespace'] must be str/bytes/None")
        self._store = store
        setter = getattr(store, "set_key_namespace", None)
        if callable(setter) and namespace:
            setter(namespace)
        store.open()
        self._index = ChunkIndex(
            store.capacity_chunks, self._chunk_tokens, namespace=namespace
        )
        self._planned_store_keys: set[bytes] = set()
        self._synced_full: set[bytes] = set()
        self._pending_forget: set[bytes] = set()
        self._closed = False

    @property
    def capacity_chunks(self) -> int:
        return self._store.capacity_chunks

    def lookup_prefix(self, token_ids: Sequence[int]) -> int:
        self._require_open()
        return self._index.lookup_prefix(token_ids)

    def hash_keys(
        self,
        token_ids: Sequence[int],
        start: int = 0,
        parent: bytes | None = None,
    ) -> tuple[list[bytes], bytes]:
        self._require_open()
        return self._index.hash_keys(token_ids, start, parent)

    def plan_store(self, keys) -> StorePlan | None:
        self._require_open()
        plan = self._index.plan_store(keys)
        if plan is None:
            return None
        self._planned_store_keys.update(plan.new_keys)
        # Scheduler owns only semantic admission. Every TP worker runs the
        # same store plan against its rank-local index and performs the
        # physical target close/object recycle in KVEngine.plan_store(). A
        # metadata-only process must never unlink worker-owned pool objects.
        return plan

    def confirm_store(self, keys, ok: bool = True) -> None:
        self._require_open()
        self._index.confirm_store(keys, ok)
        self._planned_store_keys.difference_update(keys)

    def sync_from_store(self) -> None:
        """Reconcile complete layer-marker groups written by TP workers."""
        self._require_open()
        groups = _group_scan(self._store)
        expected = set(range(self._num_layers))
        full_keys = [
            chunk_key
            for chunk_key, layers in groups.items()
            if layers >= expected
        ]
        full = set(full_keys)
        for key in list(self._planned_store_keys):
            self._index.confirm_store([key], ok=key in full)
            self._planned_store_keys.discard(key)
        stale = (self._synced_full - full) | self._pending_forget
        self._pending_forget = set(self._index.forget(stale))
        self._index.restore(full_keys)
        self._synced_full = full

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._store.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("scheduler metadata index is closed")


def _group_scan(store) -> dict[bytes, set[int]]:
    groups: dict[bytes, set[int]] = {}
    for io_key in store.scan():
        if not isinstance(io_key, (bytes, bytearray)) or len(io_key) != _IO_KEY_BYTES:
            continue
        io_key = bytes(io_key)
        groups.setdefault(chunk_key_of(io_key), set()).add(layer_of(io_key))
    return groups
def _positive_int(config: dict, key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config[{key!r}] must be a positive integer")
    return value

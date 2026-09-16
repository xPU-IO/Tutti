# Tutti Architecture

## Why This Design

A prefill-heavy LLM workload recomputes the same prefix KV cache over and over.
Tutti stores that KV on local NVMe and reads it back **directly into GPU memory**:
NVMe submission/completion queues are driven from CUDA kernels, and DMA targets
GPU BAR memory. There is no host bounce buffer and no `cudaMemcpy` on the data
path — which is why the whole stack is built around "GPU memory is a peer DMA
target", not around a filesystem.

## Layers

```
vLLM scheduler process          vLLM worker processes (one per TP rank)
  KVConnector callbacks           KVConnector callbacks
        │                               │
        ▼                               ▼
tutti/integration/vllm/connector.py  (dual-role shell, same class)
   ├── scheduler role → engine/metadata.py  SchedulerMetadataIndex
   └── worker role    → integration/vllm/worker.py  WorkerImpl
                            │
                            ▼
                    tutti/engine/core.py  KVEngine
                    (read/write plan orchestration, layer pipelining)
                            │
                            ▼
                    tutti/storage/  KVStore SPI + tutti_nvme backend
                            │
                            ▼
                    csrc/  StorageRuntime → DataPath → NVMe queues
```

- `tutti/index/chunk_index.py` — pure logic, no IO. Chunk key derivation and the
  resident / pinned / pending state machine. Instantiated on **both** sides.
- `tutti/engine/` — framework-agnostic orchestration: `core.py` (plans),
  `completion.py` (handles), `transfer.py` (direct vs staged), `staging.py`,
  `metadata.py` (scheduler-side index).
- `tutti/storage/` — `base.py` defines the `KVStore` SPI; `tutti_nvme/` is the
  native backend (object pool, layout, striped layout, runtime factory).
- `tutti/integration/vllm/` — `connector.py` (entry point registered with vLLM),
  `worker.py` (per-layer orchestration), `worker_meta.py` (worker→scheduler
  increments), `geometry.py` (derive KV geometry from vLLM config),
  `factory.py` (process-level assembly).

## Control Flow of One Request

**Scheduler side**

1. `get_num_new_matched_tokens` — look up the in-memory index, report how many
   prefix tokens can be served from the store. Never scans the disk on this path.
2. `build_connector_meta` — decide what to load, what to store, what to evict;
   emit `evicted_keys` so eviction has a single decision source.
3. `update_connector_output` — fold per-step worker increments (`committed`,
   `failed`, `forgotten`, `evicted`) back into the index.

**Worker side**

1. `start_load_kv` — build the read plan. Layer 0 is submitted synchronously so
   this returns fast; a host feeder thread submits the remaining layers
   concurrently, starting immediately (it does not wait for a compute callback).
   The write batch is also prepared here (hashing, admission, target reservation).
2. `wait_for_layer_load(N)` — wait on the CUDA event fenced after layer N's
   reads. This is a **device-side** dependency (`stream.wait_event`), so compute
   overlaps IO. If the feeder has not registered layer N yet, the caller blocks
   until it does — being behind is a normal state, not an error.
3. `save_kv_layer` / `wait_for_save` — write the layers selected in the prepared
   batch (only `plan.new_keys`, i.e. keys not already resident on this rank).
4. `build_connector_worker_meta` — hand the accumulated increments to vLLM's
   aggregator for cross-rank folding.

## The Index Is Memory-Authoritative

The scheduler index is the authority during a process's lifetime. The disk is
scanned **once**, at construction (`sync_from_store()`), to recover what a
previous process left behind; after that all updates arrive as worker increments.

Consequences worth internalising:

- A bug in the cold reconciliation path produces **zero cache hits for the entire
  process lifetime, with no error and no warning** — the hot path never falls
  back to scanning. Any change near `sync_from_store` needs an explicit test.
- Commit counting is per-rank: a chunk is only confirmed when the commit count
  reaches `tp_size` and it is not in `failed`.
- `pending` entries carry an epoch and are reclaimed after a few steps, so a
  worker that never reports back cannot leak capacity reservations.

## KV Layouts

- **`file_per_chunk`** — one file per chunk. Simple; one device per rank.
- **`striped`** — a chunk's layer segments are distributed across N devices with
  `stripe_unit` granularity. Placement is per tensor/chunk so a single IO is
  never fragmented; splitting inside a tensor happens only at the MDTS layer
  (contiguous LBAs on one device). Fine-grained striping *within* a tensor is a
  known anti-pattern that collapses bandwidth.

## Object Pool

Slots (files) are pre-created at start-up with real zero-filled extents, then
allocated purely in memory at run time. Two properties matter:

- **Slot paths are stable** across recycle/reallocate for `file_per_chunk`, so
  the C++ handle cache (keyed on the extent signature: controller PCI address,
  namespace id, file size, LBA extents) stays warm. `striped` must rename shard
  files because their names derive from the chunk URI.
- **The manifest arbitrates ownership** on restart; a directory scan cannot,
  because allocated files and free slots share the same directory once paths are
  stable.

Eviction lives in the index (a semantic decision), not in the pool (which only
knows slot numbers).

## Registration Is Lazy — and Warmed Up Deliberately

Handing GPU memory to the NVMe device (`nvm_dma_map_data_device`) is expensive
(hundreds of ms per device) and serialised by a global driver lock. It happens on
first use of a `(data_path, domain)` pair. Two things follow:

- The registration **domain key must be per device set**, not per target.
  Otherwise every new chunk re-registers, and the cost lands inside the forward
  thread while holding a registry lock.
- `try_bind_direct` issues a one-page read after the pool is ready purely to pay
  this cost during binding instead of inside the first request.

## Extending to a New Framework

Create `tutti/integration/<framework>/`. The adapter's job is to translate
framework callbacks into engine calls; it must not push framework types down into
`engine/`, `storage/` or `index/`. Register stores via
`tutti/storage/registry.py` — it exposes concrete classes for the data plane and
lazy `"module:Class"` strings for the scheduler side, so naming a store never
imports device bindings into the scheduler process.

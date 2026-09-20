# Tutti System Architecture

> **Status**: v0.1.1. This document describes the architecture as
> implemented and verified in the current codebase.

## 1. Overview

Tutti (Italian for "all instruments together") is a **CPU/GPU companion
unified storage runtime**: applications face a single stable set of
memory, target, and asynchronous IO handles, while file resolution, GPU
vendor capabilities, local NVMe, GDS, RDMA, and other data-movement
implementations live behind replaceable internal boundaries.

The core idea: **the CPU launches IO kernels, the GPU executes them.**
The CPU prepares batch descriptors, stages them into GPU memory, and
launches a GPU kernel that rings NVMe doorbells directly — the CPU never
mediates an individual IO.

### 1.1 Design goals

| Goal | How |
|------|-----|
| GPU-direct NVMe access | `snvme` kernel module + `libnvm` user library; GPU kernels write SQ entries and tap doorbells directly |
| Batch IO throughput | One GPU kernel launch handles thousands of NVMe IOs; per-IO device/LBA resolution happens on-GPU |
| Multi-device striping | A striped target spans up to 4 NVMe devices in tensor-sized units; one fused kernel fans out the batch |
| Stable public API | `StorageRuntime` is the only surface applications touch; resolvers/bindings/DataPaths are replaceable behind the SPI |
| Vendor portability | `cuda_like` three-layer GPU framework; kernel P2P layer split into symmetric per-vendor backends |

### 1.2 Current scope (v0.1.1)

- **One DataPath family**: local NVMe (`local_nvme` single-device +
  `striped_local_nvme` multi-device), ext4 file-backed via FIEMAP
- **Accelerator profiles**: CUDA (production-proven), HOST (header-only
  shim for contract tests), MUSA/MACA (build profiles in place, no
  hardware validation yet)
- **Submission mode**: host-initiated / device-executed
  (`DEVICE_EXECUTION`); host-executed is reserved for the future
- **Runtime topology**: daemon-only — `tutti_daemon` owns the NVMe
  controllers; application processes attach as clients

## 2. Layered architecture

```text
┌──────────────────────────────────────────────────────────────┐
│ Application / Framework Adapter (examples/)                    │
│   thinks in: uri / offset / size (persistent)                  │
├──────────────────────────────────────────────────────────────┤
│ StorageRuntime  (tutti/include/tutti/storage_runtime.h)        │
│   stable public API: open/open_batch, register_memory,         │
│   submit/wait, close; per-request fail-closed status           │
├──────────────────────────────────────────────────────────────┤
│ SPI  (tutti/include/tutti/spi/)                                │
│   StorageTargetResolver · DataPath · opaque identity minting   │
├───────────────┬──────────────────────────────┬───────────────┤
│ Resolvers     │ Bindings                     │ DataPaths     │
│ local_file    │ ext4_local_nvme              │ local_nvme    │
│ striped_file  │ striped_local_nvme           │ striped_…     │
│ (tutti/       │ (pair-private payload,       │ (tutti/       │
│  resolvers/)  │  type-id + version checked)  │  data_paths/) │
├───────────────┴──────────────────────────────┴───────────────┤
│ Device Manager  (tutti/device_manager/nvme/)                   │
│   libnvm (userspace queue library) · nvmeservice / tutti_daemon│
│   (controller bring-up, queue budget, mount)                   │
├──────────────────────────────────────────────────────────────┤
│ snvme kernel module  (tutti/device_manager/nvme/kernel_modules/)│
│   queue alloc · DMA map · peer_memory backends (nvidia/metax)  │
├──────────────────────────────────────────────────────────────┤
│ cuda_like GPU framework  (tutti/include/tutti/cuda_like.h,     │
│   gpu_vendor/, cmake/accelerators/) — cross-cutting            │
└──────────────────────────────────────────────────────────────┘
```

Each layer depends only on layers below it. The public API
(`tutti/include/tutti/`) never names a resolver/binding/DataPath
implementation type; SPI implementations never touch public request
types beyond their contracts.

### 2.1 Layer responsibilities

#### StorageRuntime — the only public surface

Owns the target registry, memory registry, and IO lifecycle. Routes each
URI scheme to its resolver, each resolved target to its DataPath, and
groups one `submit()` call by DataPath so a multi-target batch becomes
**one** `DataPath::submit()` — one kernel launch. `open_batch()` pipelines
parallel FIEMAP resolution with serialized DataPath open. All public
operations are serialized internally (`registry_mutex_`,
`datapath_open_mutex_`), so SPI implementations may assume single-threaded
access per instance.

Key semantics: **partial-commit** — `submit()` returns a `SubmitOutcome`
whose `initial_states` must be walked item by item; a rejected request
(e.g. backpressure) was never executed and must be resubmitted by the
caller. **Fail-closed** — every fallible step reports per-item status;
no silent fallbacks.

#### Resolvers (`tutti/resolvers/`)

Parse a URI into an immutable `ResolvedTarget` (size + pair-private
payload). `local_file` walks FIEMAP exactly once per file (O_DIRECT,
pre-stat validation). `striped_file` composes N per-device file targets
into one logical striped target (tensor-unit placement: tensor index mod N,
packed within each per-device shard).

#### Bindings (`tutti/bindings/`)

The pair-private contract between one resolver family and one DataPath
family: payload type, identity constants (`type-id` + `api-version` +
recommended DataPath key — each string defined exactly once), and the
`make_resolved_target` / `view_payload` helpers that make resolver and
DataPath physically unable to diverge.

#### DataPaths (`tutti/data_paths/`) — private, replaceable

`local_nvme`: per-device NVMe IO engine. Registration pre-builds PRP
descriptors (see [key-designs.md](key-designs.md) §2); open builds a
~200 B GPU-resident file handle cached in a GPU-L1 / pinned-host-L2
two-tier cache (§3); submit stages one H2D copy and launches the
per-queue kernel (§1).

`striped_local_nvme`: multi-device fan-out. One fused kernel submits to
N devices through a device table (§5); per-device arenas pre-allocate
PRP space mapped once per device.

#### Device Manager (`tutti/device_manager/nvme/`)

`libnvm` — userspace NVMe queue library wrapping `snvme` ioctls: queue
pair creation, DMA window mapping, ring memory. `tutti_daemon`
(nvmeservice) — controller bring-up, cross-process queue budget, and
mount; `/dev/snvme*` block devices exist only after daemon bring-up.
Bring-up order is strict: kernel modules → daemon → mount.

#### snvme kernel module (`kernel_modules/`)

Modified NVMe driver: userspace queue allocation (16 queue pairs × 1024
entries per controller, depth fixed at install time), DMA mapping of GPU
memory, and GPU P2P via the `peer_memory/` backend layer (symmetric
per-vendor backends: nvidia upstream, metax symmetric). See
[../design/kernel-portability.md](../design/kernel-portability.md).

#### cuda_like GPU framework — cross-cutting

Every userspace layer above the kernel includes `<tutti/cuda_like.h>`
instead of vendor SDK headers. One profile is selected at configure time
(`-DTUTTI_ACCELERATOR=CUDA|HOST|MUSA|MACA`). See
[../gpu-porting-guide.md](../gpu-porting-guide.md).

## 3. IO walkthrough

**Open** (`rt.open("file:///mnt/nvme0/f")`): scheme → `LocalFileResolver`
FIEMAP → `ResolvedTarget` → `LocalNvmeDataPath::open` → GPU handle build
(or L1/L2 cache hit) → opaque `TargetHandle`.

**Register** (`rt.register_memory(buf)`): DataPath DMA-maps the buffer
once, pre-splits into MDTS slices, pre-builds descriptors from the GPU
pool and packed PRP lists in pinned host memory, returns a `MemoryHandle`.
Cost is paid once per buffer, never per IO.

**Submit** (`rt.submit(reqs)`): requests grouped by DataPath → entries
staged host-side → ONE `cudaMemcpyAsync` H2D → ONE kernel launch → each
GPU thread resolves LBA from the handle's extent list, picks a queue,
writes the SQE, rings the doorbell, busy-polls the CQ phase bit. The
returned IO handle tracks accepted requests only — rejected ones
(backpressure) are reported in `initial_states` and must be resubmitted.

**Wait** (`rt.wait(io)`): host-side progress polling; no internal stream
synchronization — callers that need GPU-side ordering wire their own
stream semantics.

## 4. Deployment topology

- **Daemon**: one `tutti_daemon` per host owns the controllers; the canonical
  deployment template is `config/local_nvme_config.yaml` (gRPC defaults to
  127.0.0.1:50051).
- **Devices**: daemon-published block and view paths are authoritative;
  striped targets span multiple backing mounts when configured.
- **Build**: one default CUDA hardware build includes the module, daemon,
  local-NVMe targets, and hardware tests in `build/`.
- **Contract suites**: SPI/API contracts plus hardware gates (datapath /
  runtime E2E / striped / resolver). Build and test commands are centralized
  in [../getting-started.md](../getting-started.md).

## 5. Source map

```text
tutti/include/tutti/          public API (storage_runtime.h, io_types.h, …)
tutti/include/tutti/spi/      SPI contracts (data_path.h, resolver, …)
tutti/include/tutti/gpu_vendor/  cuda_like vendor shims
tutti/resolvers/              local_file, striped_file
tutti/bindings/               ext4_local_nvme, striped_local_nvme, memfs (sample)
tutti/data_paths/             local_nvme, striped_local_nvme
tutti/device_manager/nvme/    libnvm, nvmeservice, kernel_modules
tutti/examples/               layerwise_kv_overlap (KV cache reference)
tests/                        contract suites + perf microbenchmarks
doc/                          architecture, design, porting, build docs
```

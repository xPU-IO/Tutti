# Refactor Architecture & Legacy Decomposition

> **Scope.**  This file is the single source of truth for the ongoing
> Tutti refactor.  It defines (1) the layer cake, (2) what each layer
> does and is forbidden to do, (3) the per-layer API surface at
> method-signature granularity, (4) where every chunk of legacy code
> ends up, and (5) the remaining refactor steps.
>
> **Audience.**  Anyone touching a layer during R5 onwards.  Read
> §2 + §3 before designing a new public type, §4 before opening a
> legacy `.cu/.cpp`, §5 before starting a new commit.
>
> **Lifetime.**  Deleted in the final cleanup step (currently R10),
> together with `filesystems/ext4/libgeminifs/`.

---

## 1. Why we re-architected mid-flight

The first attempt (R0–R4) had three layers at the bottom (memory,
device_manager, io_engine) sitting in parallel under a thin
`runtime/` facade.  That shape worked for SPI sketching but broke as
soon as we asked the question *"where do we put the moral equivalent
of `NVMeController` + `GPUFileManager`?"*  Those legacy classes
collectively do seven distinct jobs:

1.  Bring up an NVMe controller (init, queues, BAR0).
2.  Maintain an on-disk file directory with names + sizes.
3.  Translate "GPU file id + offset" into raw NVMe LBA + ctrl + queue.
4.  Stripe / replicate one GPU-visible file across multiple NVMe files.
5.  Register host or device tensors and turn them into NVMe-addressable
    descriptors (PRP today, SGL where supported).
6.  Schedule batched IO from a host-issued request down to a CUDA
    kernel that posts NVMe commands and rings doorbells.
7.  Provide host-side blocking read/write for bootstrap and tests.

Squeezing all seven into "device_manager + io_engine" forces every
class to leak across two layers.  The fix is to recognise that we
actually need **eight layers**, with strict single-direction edges,
each owning exactly one of those jobs (plus memory and the public
facade).

---

## 2. Layer cake

```
                                 ┌──────────────────────────┐
                                 │  api / coordinator       │  USER ENTRY
                                 │  (eventual: rename       │
                                 │   runtime/ -> coordinator/)│
                                 └─────────────┬────────────┘
                                               │
                                 ┌─────────────▼────────────┐
                                 │  io_engine               │  ORCHESTRATION
                                 │  BatchIoRequest scheduler│
                                 └──┬──────┬─────────┬──────┘
                                    │      │         │
                       ┌────────────▼─┐ ┌──▼──────┐ ┌▼──────────┐
                       │ block_storage│ │ memory  │ │nvme_storage│  ABSTRACTIONS
                       │ virtual file │ │ tensor→ │ │ named LBA  │
                       │ over many    │ │ PRP/SGL │ │ ranges +   │
                       │ NvmeFiles    │ │         │ │ device-side│
                       │              │ │         │ │ submit     │
                       └─────┬────────┘ └────┬────┘ └─────┬─────┘
                             │               │            │
                             └───────────────┼────────────┘
                                             │
                              ┌──────────────▼─────────────┐
                              │ device_manager             │  CONTROL PLANE
                              │ controller bring-up,       │
                              │ leases, heartbeats         │
                              └──────────────┬─────────────┘
                                             │
                              ┌──────────────▼─────────────┐
                              │ backends/local/nvme/       │  DRIVER
                              │ libnvm + NVMeService       │
                              │ + snvme kernel module      │
                              └────────────────────────────┘
```

### 2.1  One sentence per layer

| Layer | Sentence | Forbidden to … |
|-------|----------|----------------|
| **api/** | Public types only — `Status`, `Runtime/Coordinator`, `RuntimeConfig`, `error.h`. | depend on anything below. |
| **coordinator/** *(today: `runtime/`)* | The single entry point a GPU program holds; wires every other layer together at startup, owns the global capability handshake. | run any data-plane code itself; touch libnvm directly. |
| **io_engine/** | Accepts a `BatchIoRequest` (a list of "(tensor, BlockHandle, offset, length)"), looks up addresses via block_storage and memory, then drives the GPU kernel that submits NVMe commands through nvme_storage. | own queue pairs; talk to libnvm; know about ext4. |
| **block_storage/** | Virtual file abstraction visible to GPU kernels; one `BlockHandle` references *N* `NvmeFile`s and resolves "(block_handle, offset)" to "(NvmeFile, file_offset)". | own NvmeFiles (it only references them); know which controller a NvmeFile lives on; touch ext4. |
| **memory/** | Single source of truth for "this buffer is registered for IO"; produces NVMe address descriptors (PRP today, SGL when the cluster supports it) on a per-tensor / per-host-buffer basis. | submit IO; touch NvmeFile / BlockHandle / ext4; know about libnvm queue layout. |
| **nvme_storage/** | Owns named LBA ranges on top of an NVMe namespace, the device-side submission entry-points (`__device__ submit_read_one` / `submit_write_one`), and the queue-pair pool with its on-GPU acquire helper.  Also handles host FS allocation + FIEMAP translation when running on top of ext4/xfs. | care about tensors; care about virtual files; talk to multiple controllers in one call. |
| **device_manager/** | Brings controllers up and down, hands out `Device*` instances, runs leases and heartbeats.  Future home for non-PCIe transports (RDMA-over-fabric, etc). | own files; own buffers; own queue pairs after handing them off. |
| **backends/local/nvme/** | Concrete driver glue: `libnvm` user-space NVMe driver + `NVMeService` daemon + `snvme*` kernel module. | be visible above device_manager / nvme_storage. |

### 2.2  Hard invariants (memorise these)

1.  **Edges are single-direction.**  No layer ever calls *up*.  The
    diagram in §2 is a strict DAG; a back-edge is a bug.
2.  **`block_storage` is reference-only over `NvmeFile`.**  An
    `NvmeFile` is created and destroyed by `nvme_storage`.  A
    `BlockHandle` keeps a refcount-stable view; tearing down a
    `BlockHandle` does **not** delete its `NvmeFile`s.  This is what
    lets us recycle GPU-visible files and reassign them.
3.  **Address-descriptor format is cluster-wide, not per-device.**
    The runtime probes every controller it ever attaches; if any one
    of them lacks SGL support, the *whole runtime* falls back to
    PRP for all controllers.  Memory therefore stores at most one
    descriptor format per registration, never a mix.  Decided once
    at coordinator boot, frozen for the run.
4.  **Queue pairs are GPU-side concurrency primitives.**  The host
    side never selects a queue.  An io_engine kernel thread, having
    decided which `NvmeFile` to hit, calls into `nvme_storage`'s
    device-side submit function which uses the on-GPU
    `QueueAcquireHelper` to claim an SQ slot, write the SQE, ring
    the doorbell, and poll the CQ entry.  Host code never sees a
    queue-pair handle.
5.  **`memory` does not know about NVMe queues, but does know
    descriptor formats.**  Producing a PRP list is a memory job
    because it is "this buffer expressed in NVMe-readable form".
    Submitting that PRP list is an nvme_storage job.
6.  **`nvme_storage` is not bound to ext4.**  Today it is implemented
    on top of an ext4 / xfs mount via FIEMAP, but its public surface
    only mentions named LBA ranges.  A future
    `direct_namespace_storage` impl that owns a whole namespace
    must satisfy the same interface without changes above.
7.  **`device_manager` is the only place that holds a controller's
    lifecycle.**  Heartbeats, lease expiry, future RDMA-over-fabric
    re-attach, all live here.  No higher layer ever calls
    `nvm_ctrl_free`.

---

## 3. Per-layer API surface

This is method-signature level, not implementation level.  Field
shapes are illustrative — the real header may add detail, but
**adding a new top-level method requires editing this file first**.

### 3.1  api/

```cpp
namespace tutti {

enum class StatusCode : int { OK = 0, /* ... */ };
struct Status { StatusCode code; std::string message; };

class Runtime;             // renamed to Coordinator in R8
struct RuntimeConfig;      // renamed to CoordinatorConfig in R8

}
```

### 3.2  coordinator/  (today still `runtime/`)

```cpp
namespace tutti {

class Coordinator {                  // R8: rename from Runtime
public:
    static Status Create(const CoordinatorConfig&,
                         std::unique_ptr<Coordinator>*);

    // === wiring =========================================================
    // 1. Adds a controller to the registry (delegates to device_manager).
    // 2. Probes ctrl identify to learn descriptor capability.
    // 3. ANDs ctrl capabilities with the cluster-wide capability set.
    //    SGL is enabled iff every attached ctrl supports it.
    // 4. Tells nvme_storage to start tracking files on this device.
    // 5. Tells memory which descriptor format to produce from now on.
    Status RegisterNvmeDevice(Device* dev);

    // === user-facing data plane ========================================
    Status SubmitBatch(const BatchIoRequest& req);
    Status WaitBatch (BatchHandle h);

    // === user-facing files (thin pass-through to lower layers) =========
    Status CreateBlock (const BlockSpec&,  BlockHandle**  out);
    Status OpenBlock   (BlockId,           BlockHandle**  out);
    Status RegisterTensor(const TensorRegistrationSpec&, MemoryRegion** out);
};

}
```

### 3.3  io_engine/

```cpp
namespace tutti {

enum class IoOpcode : uint8_t { READ, WRITE };

struct BatchIoEntry {
    BlockHandle*  block;
    uint64_t      block_offset;
    uint64_t      length;
    void*         tensor_ptr;        // host pointer registered via memory
    IoOpcode      opcode;
    // legacy parity: K-cache + V-cache layered IO walks the same struct
    // by varying block_offset; no stride field at this stage.
};

struct BatchIoRequest {
    std::span<const BatchIoEntry> entries;
    cudaStream_t                  completion_stream;
};

class IIoEngine {
public:
    virtual Status submit_batch(const BatchIoRequest&) = 0;
    virtual Status wait_batch  (BatchHandle h)         = 0;
};

// Concrete kernel (lives in io_engine/src/), exported for tests.
__global__ void nvme_batch_xfer_kernel(BatchIoEntry*  d_entries,
                                       uint32_t       count,
                                       bool           is_read);

}
```

The kernel body is the legacy `nvme_batch_xfer_kernel` rewritten to
call `nvme_storage::submit_read_one / submit_write_one` for the
NVMe submission path, and to use `block_storage::resolve_on_device`
for the file-id-to-LBA translation.

### 3.4  block_storage/

```cpp
namespace tutti {

using BlockId = uint32_t;

struct BlockSpec {
    uint64_t total_size;             // bytes the user wants
    uint32_t stripe_count;           // number of NvmeFiles to spread over
    std::vector<uint64_t> shape;     // optional tensor shape hint
};

struct BlockHandle {
    BlockId            id;
    uint64_t           total_size;
    // Reference (not owned).  nvme_storage owns the lifetime.
    cuda::std::span<NvmeFile* const> nvme_files;
};

class IBlockStorage {
public:
    virtual Status create_block (const BlockSpec&,  BlockHandle**) = 0;
    virtual Status open_block   (BlockId,           BlockHandle**) = 0;
    virtual Status close_block  (BlockHandle*)                     = 0;
    virtual Status delete_block (BlockId)                          = 0;

    // Host-side resolution, used by tests / metadata paths.
    virtual Status resolve(const BlockHandle*, uint64_t block_offset,
                           NvmeFile** out_file,
                           uint64_t*  out_file_offset) const = 0;
};

// Device-side resolution, used inline from io_engine kernels.
__device__ NvmeFile* block_resolve_on_device(const BlockHandle* h,
                                             uint64_t           block_offset,
                                             uint64_t*          out_file_offset);

}
```

The striping logic is the legacy
```
fd_idx   = gpu_blk % nvme_files.size();
file_off = (gpu_blk / nvme_files.size()) * blk_size;
```
moved verbatim into `block_resolve_on_device`.

### 3.5  memory/

```cpp
namespace tutti {

enum class DescriptorFormat : uint8_t { PRP = 1, SGL = 2 };

// One row of an NVMe-readable description of part of a buffer.
struct AddressDescriptor {
    uint64_t prp1;          // valid iff format == PRP
    uint64_t prp2;
    // SGL fields collapsed for now; expand when R7 adds SGL builder.
    uint64_t data_length;
};

struct TensorRegistrationSpec {
    void*                   ptr;
    size_t                  size;
    std::vector<size_t>     shape;
    std::vector<Device*>    target_devices; // see invariant 3 in §2.2
};

class IMemorySubsystem {
public:
    // Cluster-wide format selection.  Coordinator sets this once at
    // boot, ANDed across every attached controller.
    virtual void set_descriptor_format(DescriptorFormat fmt) = 0;

    // Allocation + free
    virtual Status allocate_host  (size_t, MemoryKind, MemoryRegion**) = 0;
    virtual Status allocate_device(size_t, MemoryKind, int gpu_device_index,
                                   MemoryRegion**) = 0;
    virtual void   free(MemoryRegion*) = 0;

    // Caller-owned memory
    virtual Status register_host_buffer(void* ptr, size_t,
                                        MemoryRegion**) = 0;
    virtual Status register_tensor    (const TensorRegistrationSpec&,
                                        MemoryRegion**) = 0;
    virtual Status unregister(MemoryRegion*) = 0;

    // Address descriptors visible to io_engine.
    // The slice index addresses into the per-page descriptor table that
    // memory built at registration time.
    virtual Status descriptor_slice(MemoryRegion*,
                                    uint64_t byte_offset,
                                    uint64_t byte_length,
                                    cuda::std::span<AddressDescriptor>* out) = 0;
};

}
```

`registerTensorMemory + initializePRPList + addPRPMappingsToGPU +
performDMASlicing` from `gpu_controller.cu` collapse into
`register_tensor` + `descriptor_slice`.  The "PRP table on GPU" that
legacy parks next to a DMA context becomes the result of
`descriptor_slice`.

### 3.6  nvme_storage/

```cpp
namespace tutti {

using NvmeFileId = uint64_t;

struct LbaExtent { uint64_t start_lba; uint64_t length_blocks; };

struct NvmeFile {                 // host view (opaque to user)
    NvmeFileId id;
    uint64_t   size_bytes;
    Device*    device;            // home controller
    std::vector<LbaExtent> extents;
    // device-side mirror; populated by nvme_storage on attach.
    NvmeFileDeviceHandle* d_handle;
};

class INvmeStorage {
public:
    // === directory ====================================================
    virtual Status create_file(Device*, std::string_view name,
                               uint64_t size, NvmeFile** out) = 0;
    virtual Status open_file  (Device*, NvmeFileId,  NvmeFile** out) = 0;
    virtual Status delete_file(NvmeFile*)                              = 0;
    virtual std::vector<NvmeFile*> list_files(Device*) const           = 0;

    // === capacity =====================================================
    virtual uint64_t total_capacity(Device*)     const = 0;
    virtual uint64_t available_capacity(Device*) const = 0;

    // === host-side blocking IO (for bootstrap / tests / metadata) =====
    virtual Status read_blocking (NvmeFile*, uint64_t off, void* dst,  size_t len) = 0;
    virtual Status write_blocking(NvmeFile*, uint64_t off, const void* src, size_t len) = 0;

    // === queue-pair lifecycle (for io_engine kernels) =================
    // io_engine kernels never call these directly; nvme_storage hands a
    // device-side QueueAcquireHelper to coordinator at boot, kernels use
    // that helper plus the device submit functions below.
    virtual Status acquire_queue_pair(Device*, QueuePairHandle*) = 0;
    virtual Status release_queue_pair(QueuePairHandle)           = 0;
};

// Device-side submit entry-points called inline from io_engine kernels.
// One thread per IO; the helper takes care of SQ slot arbitration and
// CQ polling.  This is the new home of the legacy
// nvme_controller_g_read / nvme_controller_g_write functions.
__device__ Status submit_read_one (NvmeFileDeviceHandle* file,
                                   uint64_t prp1, uint64_t prp2,
                                   uint64_t file_offset,
                                   uint64_t length);
__device__ Status submit_write_one(NvmeFileDeviceHandle* file,
                                   uint64_t prp1, uint64_t prp2,
                                   uint64_t file_offset,
                                   uint64_t length);

}
```

`read_fiemap` and `host_refine_nvmeofst` are private helpers inside
the `Ext4FsBackedNvmeStorage` impl: they translate
`(host_path → ext4 inode → FIEMAP extents → LBA list)` into
`NvmeFile::extents`.  A future `DirectNamespaceNvmeStorage` impl
would skip them entirely.

### 3.7  device_manager/

```cpp
namespace tutti {

class IDeviceRegistry {
public:
    virtual Status open(/* impl-specific config */) = 0;
    virtual void   close()                          = 0;
    virtual size_t device_count() const             = 0;
    virtual Device* find_by_id(uint32_t id) const   = 0;
    virtual std::vector<Device*> list() const       = 0;
};

class ILeaseManager {
public:
    virtual Status start_heartbeat(Device*) = 0;
    virtual Status stop_heartbeat (Device*) = 0;
};

}
```

Everything that exists today (`LocalNvmeDirectRegistry`,
`NvmeServiceBackedRegistry`, `LocalNvmeDevice`) stays.  No
expansion.

### 3.8  backends/local/nvme/

Unchanged: libnvm B3/B6 ABI, NVMeService daemon + client, snvme*
kernel module.  Layers above only consume `Device*` and
`nvm_ctrl_t*` through device_manager.

---

## 4. Legacy migration map

The legacy tree under `filesystems/ext4/libgeminifs/` is restored
on disk for reference (not built).  Each chunk listed below moves
to the layer named in the third column.  R5–R9 are the steps that
do the actual moves; this table is the *what*, §5 is the *when*.

| Legacy file | Section | New home |
|-------------|---------|----------|
| `nvme_controller.cu` / `.cuh` | `NVMeController` ctor / dtor (`open_single_controller`, queue setup) | **device_manager/** *(already moved during R3-R4 as `LocalNvmeDevice` + the two registries)* |
| | `host_file_create_managed`, `host_file_open_managed`, `device_file_*_managed`, `cleanup_device_files`, `get_file_path`, `next_nvme_file_id` | **nvme_storage/** |
| | `init_queue_acquire_helper_kernel`, `d_queue_acquire_helper`, `nvme_controller_g_read/g_write` `__device__` | **nvme_storage/** *(the helper + the device submit functions)* |
| `nvme_file.cpp` / `nvme_file.h` | `FileManager`, `LogHeader`, `NVMeFileDesc`, `OpenFileHandle`, log persistence | **nvme_storage/** *(directory + capacity)* |
| | `geminiFS_hdr` on-disk header struct | **nvme_storage/** *(internal)* |
| `gpu_controller.cu` / `.cuh` | `GPUController` ctor / dtor / cleanup, `addNVMeController`, `getAllNVMeControllers`, `GPUControllerRegistry` singleton | partly **coordinator/** *(register_nvme_device flow)*, partly **block_storage/** *(multi-NvmeFile pool)* |
| | `registerTensorMemory`, `registerTensorList`, `unregisterTensorMemory`, `getDMAContext`, `validateTensor` | **memory/** |
| | `performDMASlicing`, `createDMAContext`, `createDMAContexts`, `initializePRPList`, `doInitializePRPList`, `initializePRPEntries`, `addPRPMappingsToGPU`, `geminifs_dma` | **memory/** *(descriptor builder)* |
| `gpu_file_manager.cu` / `.cuh` | `GPUFileManager` (GPU-visible file directory, `createGPUFile`, `openGPUFile`, `getNVMeFilesSpanById`) | **block_storage/** |
| | `BatchIoEntry`, `GPUIoContext`, `BatchIoPool` | **io_engine/** |
| `geminifs.cu` / `geminifs.cpp` | `geminifs_batched_xfer` (host-side orchestration + kernel launch) | **io_engine/** |
| | `nvme_batch_xfer_kernel` `__global__` | **io_engine/** |
| | Torch-specific facade (`save_kvcache`, `load_kvcache`, `get_prp_mappings`) | **adapters/torch_kvcache/** *(deferred to Phase 6; do not rewrite during this refactor)* |
| `geminifs_helper.cpp` | `read_fiemap`, `host_refine_nvmeofst` | **nvme_storage/** *(internal helpers in `Ext4FsBackedNvmeStorage`)* |
| | YAML topology parsing, error printers | scattered: yaml stays in **coordinator/**, error printers fold into `api/error.h` |
| `gemini_fiemap.h` | FIEMAP ABI wrapper | **nvme_storage/** *(internal)* |
| `prp_mapping_entry.h` | `PRPMappingEntry`, `PRPListPage`, `SubSliceInfo` | **memory/** *(used by descriptor builder)* |
| `utils.cuh` / `helper.cuh` | misc macros + small device utilities | already replaced by `memory/include/cuda_helpers.cuh`; remaining bits inlined where used. |
| `memory.cpp` | host malloc + alignment helpers | **memory/** *(absorbed into `HostDeviceMemorySubsystem`)* |
| `backtrace.cpp` | crash debug only | **drop** — not part of any layer; not load-bearing. |
| `ops.h` | Torch python bindings | **adapters/torch_kvcache/** *(deferred)* |

---

## 5. Refactor steps (R0 — R10)

R3–R4 are done; R5 onward is the new plan that reflects the
eight-layer cake above.

| Step | What lands | Smoke / acceptance |
|------|------------|--------------------|
| R0   | This file + delete `.bak`, `json.h`. | n/a |
| R3   | `memory/` first cut: `HostDeviceMemorySubsystem` (host alloc, device alloc, register external).  **No** descriptor builder yet. | `memory_smoke` |
| R4   | `device_manager/` first cut: `LocalNvmeDevice`, `LocalNvmeDirectRegistry`, `NvmeServiceBackedRegistry`. | `registry_smoke --mode=direct/service` |
| **R5** | **`nvme_storage/` lands.**  `Ext4FsBackedNvmeStorage` with FileManager-style directory, FIEMAP-based extent capture, host-side blocking read/write, queue-pair pool, the on-GPU `QueueAcquireHelper`, and the device-side `submit_read_one` / `submit_write_one`. | `nvme_storage_smoke` (CPU read/write) + `nvme_storage_gpu_smoke` (one device-side submit per thread, no batching). |
| **R6** | **`block_storage/` lands.**  `IBlockStorage` impl + `BlockHandle` + `block_resolve_on_device`.  References `NvmeFile`s by raw pointer; nvme_storage stays the owner. | `block_storage_smoke` (create block over 2 NvmeFiles, host-side resolve, GPU resolve via a tiny kernel that prints `(file_idx, file_off)`). |
| **R7** | **`memory/` PRP/SGL builder.**  Adds `register_tensor`, `set_descriptor_format`, `descriptor_slice`.  Implements PRP only at this stage; SGL is a feature flag that throws `Unimplemented` until R8 needs it. | `memory_descriptor_smoke` (register a 4 MB host buffer, slice into a 64-page descriptor span, print prp1/prp2). |
| **R8** | **`io_engine/` lands.**  `BatchIoEntry`, `BatchIoPool`, `nvme_batch_xfer_kernel` rebuilt on top of memory + block_storage + nvme_storage.  Lift `geminifs_batched_xfer` host-side orchestration into `IIoEngine::submit_batch`. | `io_engine_smoke` (one BatchIoRequest with 4 entries against a 2-stripe block, end-to-end through the kernel). |
| **R9** | **Coordinator wiring + examples reorganisation.**  Rename `runtime/` -> `coordinator/`, `Runtime` -> `Coordinator`, `RuntimeConfig` -> `CoordinatorConfig`.  Implement `RegisterNvmeDevice` (probe + AND capabilities + freeze descriptor format), `SubmitBatch`.  Move all smoke binaries from `*/test/` into a flat `examples/` tree; add `tutti_daemon` (wraps NVMeService daemon entry with the new high-level API). | `e2e_smoke` (Coordinator -> create block -> register tensor -> SubmitBatch -> verify roundtrip). |
| **R10** | Delete `filesystems/ext4/libgeminifs/`, `doc/refactor/`, any leftover `examples/test_*` legacy dirs.  Final pass to remove the word `geminifs` from the source tree. | grep -i geminifs returns nothing in code. |

### 5.1  Status checklist

- [x] R0 — this file (rewritten R5-prep edition).
- [x] R3 — memory v1 + smoke.
- [x] R4 — device_manager v1 + smoke (direct + service).
- [ ] R5 — nvme_storage.
- [ ] R6 — block_storage.
- [ ] R7 — memory descriptor builder.
- [ ] R8 — io_engine.
- [ ] R9 — coordinator + examples reorganisation + tutti_daemon.
- [ ] R10 — final cleanup.

---

## 6. Open implementation questions

These do not block R5; they will be revisited at the step that
needs them.

- **R5.q1**  Does `Ext4FsBackedNvmeStorage::create_file` use
  `fallocate` (legacy behaviour) or `posix_fallocate`?  Decide when
  writing the impl — only matters for which extents end up
  contiguous.
- **R5.q2**  How does the queue-pair pool allocate SQ/CQ rings —
  one ring per (cuda_stream, device) tuple, or a global pool?
  Legacy used per-stream; revisit if it shows up as contention in
  io_engine_smoke.
- **R6.q1**  Does `BlockHandle` keep the `NvmeFile*` array on host
  only, or also publish a device-side mirror?  Legacy publishes a
  device-side `NVMeFilesSpan`; replicate that.
- **R7.q1**  Does the descriptor table live in pinned host memory
  or device memory?  Legacy uses device.  Defer the call until R7.
- **R7.q2**  Where does the SGL builder land when SGL fallback is
  added?  Probably a sibling class to the PRP builder behind a
  common interface.
- **R8.q1**  Should `BatchIoEntry` grow a `stride` field for KV
  cache layered access, or do we keep the legacy "compute offset
  externally and pass it in" pattern?  Lean: keep legacy.
- **R9.q1**  Does `tutti_daemon` reuse `nvmeservice_daemon`'s
  `SessionBroker` directly or wrap it behind a thinner facade?
  Lean: reuse, wrap only the entry point in `coordinator/`.

# R5b Plan — GPU Device-Side NVMe Submit

**Status:** in-progress (companion to R5a `nvme_storage` host-side)
**Owner:** refactor/code-cleanup branch
**Companion docs:** `LegacyDecomposition.md` §2 (layer cake), §3.6
(`INvmeStorage`), §4 (legacy migration).

---

## 1  What R5b adds on top of R5a

R5a delivered the **host-side** half of `nvme_storage`:

```
INvmeStorage::create_file(dev, name, size)
                    ↓
                NvmeFile { host_fd, extents[], ... }
                    ↓
INvmeStorage::read_blocking / write_blocking      ← host fd path
```

R5b adds the **GPU-side** half — the path that lets a CUDA kernel
submit one NVMe SQE per thread directly against the controller's
SQ ring, bypassing the kernel block layer:

```
NvmeFile (R5a, host-only)
    ↓  acquire_device_handle()
NvmeFileDeviceHandle (R5b)  -- on GPU memory, holds extents[]
                                + namespace + block size +
                                back-pointer to the controller's
                                d_qps[] pool

inside a CUDA kernel:
    __device__ submit_read_one (file_handle, prp1, prp2,
                                file_offset, nbytes)
    __device__ submit_write_one(file_handle, prp1, prp2,
                                file_offset, nbytes)
```

The contract is verbatim from legacy `nvme_controller_g_read /
g_write`: caller gives PRP1/PRP2 (already produced by `memory/`,
which knows how to translate a tensor pointer into NVMe-compatible
descriptor entries) plus a logical (file_offset, nbytes) span; the
device function picks an SQ slot, writes the SQE, rings the door-
bell, and busy-polls the CQ.  Host gets back nothing — the kernel
itself is the synchronization point.

R5b does **not** introduce batching (R7 / `io_engine` does that)
and does **not** know about tensors, virtual blocks, or PRP mapping
(those live in `memory/` and `block_storage/`).

## 2  Why the current LocalNvmeDevice doesn't have what we need

`LocalNvmeDevice::ctrl` is the C-style `nvm_ctrl_t*`.  That handle
holds the BAR0 mapping, the queue group state, and the kernel
ioctl fd — but it does **not** carry `d_qps` (the GPU-resident
`QueuePair[]` array that kernels index into).

`d_qps` is a member of libnvm's C++ `Controller` wrapper, which
historically owns the *whole* bring-up:

```
Controller ctor
    -> nvm_controller_init_b3()                ← already done by
                                                  LocalNvmeDirectRegistry
    -> init_queues()                            ← what we actually want
        -> nvm_create_group()
        -> N x QueuePair(B3 ctor)
        -> batched NVM_ADD_USER_QUEUE
        -> cudaMalloc d_qps + memcpy each QueuePair into d_qps[i]
    -> cudaMalloc d_ctrl_ptr + memcpy *this
```

That ctor is monolithic — there is **no current way** to ask
`Controller` to "wrap an already-existing `nvm_ctrl_t*` and just
do the queue-pool half".  R5b therefore needs three coordinated
changes:

1. libnvm: add a `Controller` "wrap" ctor that skips
   `nvm_controller_init_b3` and assumes the caller passes in an
   already-brought-up `nvm_ctrl_t*` plus the disk struct.  Mark
   the resulting Controller as "ctrl is borrowed" so the dtor
   does not free it.
2. device_manager: extend `LocalNvmeDevice` with
   `std::shared_ptr<Controller> cpp_ctrl` and have
   `LocalNvmeDirectRegistry::open_one()` build it after init_b3.
   `LocalNvmeDevice::dma_ctrl()` becomes the canonical way for
   upper layers to reach `d_qps` / `d_ctrl_ptr`.
3. nvme_storage: introduce the device-side surface defined in §4
   below.

Service mode (`NvmeServiceBackedRegistry`) is **out of scope for
R5b** — it would require sharing `d_qps` across the daemon/client
process boundary, which is a separate refactor.  In R5b
`acquire_device_handle()` simply returns `nullptr` for SERVICE_CLIENT
attach mode and the GPU smoke runs only against direct-mode
controllers.

## 3  libnvm Controller wrap ctor

Smallest possible API surface, additive only:

```cpp
// ctrl.h additions

struct Controller {
    // Existing fields ...

    // R5b: "ctrl is borrowed -- caller owns it" flag.  When true,
    // ~Controller does not call nvm_ctrl_free()/nvm_ctrl_free_client().
    bool ctrl_borrowed = false;

    // Existing ctors ...

    // R5b: wrap an existing nvm_ctrl_t* that the caller already
    // brought up via nvm_controller_init_b3 (or attach_client).
    // We only do init_queues() and the GPU side allocations here.
    // existing_disk MUST be the disk struct populated by the
    // earlier init_b3 call.
    Controller(nvm_ctrl_t*    existing_ctrl,
               struct disk    existing_disk,
               std::string    mount_path,
               uint32_t       ns_id,
               const std::vector<QueueMemTarget>& queue_targets,
               uint64_t       queueDepth);
};

// Implementation sketch:
inline Controller::Controller(nvm_ctrl_t*    existing_ctrl, ...)
    : ctrl(existing_ctrl), ctrl_borrowed(true), deviceId(0)
{
    // Pick primary cudaDevice from queue_targets.
    // ... (same as monolithic ctor, but skip init_b3)
    this->disk = existing_disk;
    this->disk.ns_id = ns_id;

    int status = init_queues(ns_id, queue_targets, queueDepth);
    if (status != 0) nvm_throw_error("init_queues", status);

    page_size = ctrl->page_size;
    blk_size  = disk.block_size;
    blk_size_log = std::log2(blk_size);
    dev_path = strdup(("/dev/" + std::string(disk.disk_name)).c_str());
    dev_mount_path = std::move(mount_path);

    d_ctrl_buff = createBuffer(sizeof(Controller), deviceId);
    d_ctrl_ptr  = d_ctrl_buff.get();
    cuda_err_chk(cudaMemcpy(d_ctrl_ptr, this, sizeof(Controller),
                            cudaMemcpyHostToDevice));
}

inline Controller::~Controller() {
    // ... (existing d_qps + h_qps cleanup unchanged) ...

    if (!ctrl_borrowed && ctrl != nullptr) {
        nvm_ctrl_free(ctrl);     // existing path
        ctrl = nullptr;
    }
    // borrowed: caller owns ctrl; do nothing here.
}
```

**Risk:** `Controller::~Controller` currently does
`nvm_ctrl_free(ctrl)` unconditionally.  Tests that already use
the monolithic ctor must continue to work — `ctrl_borrowed` defaults
to `false`, so the existing path is unchanged.

## 4  nvme_storage device-side surface

### 4.1  Data structures (header only)

```cpp
// nvme_storage/include/nvme_file_device_handle.h

namespace tutti {

// On-GPU file handle.  Pure POD.  cudaMalloc'd once by
// acquire_device_handle(), copied verbatim from host, never mutated
// thereafter.  Kernels read it; nothing writes it.
struct NvmeFileDeviceHandle {
    uint64_t   file_id;
    uint64_t   logical_size_bytes;   // user view (excludes header)
    uint32_t   header_bytes;         // sizeof(NvmeFileHeader) = 4096
    uint32_t   nvme_block_size;      // typ. 4096
    uint32_t   nvme_block_size_log;  // 12
    uint32_t   namespace_id;

    uint32_t   num_extents;
    LbaExtent  extents[kNvmeFileHeaderMaxExtents];   // inline copy

    // Back-pointer to the d_qps[] pool of the controller this
    // file lives on.  Caller (nvme_storage::acquire_device_handle)
    // takes this from cpp_ctrl->d_qps.
    QueuePair* d_qps;
    uint32_t   num_d_qps;
};

}  // namespace tutti
```

### 4.2  QueueAcquireHelper (device-side concurrency)

Behavior is verbatim from legacy `helper.cuh`:

```cpp
// nvme_storage/include/queue_acquire_helper.cuh

namespace tutti {

class QueueAcquireHelper {
public:
    __device__ __forceinline__
    static uint32_t acquire_queue(uint32_t num_queues) {
        // legacy: (blockDim.x * 32 + threadIdx.x) % num_queues
        // pure hash, no actual locking; release_queue is a no-op.
        return (blockDim.x * 32u + threadIdx.x) % num_queues;
    }

    __device__ __forceinline__
    static void issue_nvme_cmd(QueuePair* qp,
                               uint64_t prp1, uint64_t prp2,
                               uint64_t n_blocks, uint64_t starting_lba,
                               uint8_t opcode, uint16_t* cid_out);

    __device__ __forceinline__
    static void poll(QueuePair* qp, uint16_t cid);
};

}  // namespace tutti
```

`issue_nvme_cmd` and `poll` are exact ports of legacy:

```
*cid = get_cid(&qp->sq);
nvm_cmd_header(&cmd, *cid, opcode, qp->nvmNamespace);
nvm_cmd_data_ptr(&cmd, prp1, prp2);
nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
sq_enqueue(&qp->sq, &cmd);

uint32_t cq_pos = cq_poll(&qp->cq, cid);
cq_dequeue(&qp->cq, cq_pos, &qp->sq);
put_cid(&qp->sq, cid);
```

### 4.3  Device-side submit functions

```cpp
// nvme_storage/include/nvme_storage_device.cuh

namespace tutti {

// Translate a logical file offset into physical LBA + length, by
// walking handle->extents[].  Inline so it is folded into the
// caller kernel.  Returns false if offset is out-of-range or
// crosses an extent boundary at non-aligned position.
__device__ __forceinline__
bool resolve_lba(const NvmeFileDeviceHandle* h,
                 uint64_t logical_off, uint64_t nbytes,
                 uint64_t* starting_lba_out, uint64_t* n_blocks_out);

// One-thread-per-IO submit + busy-poll completion.  Returns when
// CQE is observed; no host involvement.  prp1/prp2 must already
// be NVMe-DMA-resolved (the memory/ layer's job).
__device__ __forceinline__
void submit_read_one (const NvmeFileDeviceHandle* h,
                      uint64_t prp1, uint64_t prp2,
                      uint64_t logical_off, uint64_t nbytes);

__device__ __forceinline__
void submit_write_one(const NvmeFileDeviceHandle* h,
                      uint64_t prp1, uint64_t prp2,
                      uint64_t logical_off, uint64_t nbytes);

}  // namespace tutti
```

`logical_off` is in user view (0 = first user byte, header is hidden).
The implementation adds `header_bytes` before resolving.

### 4.4  Host-side facade additions on `INvmeStorage`

```cpp
// nvme_storage/include/nvme_storage.h additions

class INvmeStorage {
public:
    // Existing: bootstrap / shutdown / create_file / ... / read_blocking

    // R5b: open a NvmeFile on GPU memory.  Caller picks the
    // cuda device implicitly (current cudaSetDevice context);
    // future overload may accept an explicit device id.
    //
    // Returns nullptr if:
    //   - file's owning device is in SERVICE_CLIENT mode (R5b
    //     doesn't support that yet), or
    //   - cudaMalloc / cudaMemcpy fails.
    //
    // The returned pointer is a GPU-resident NvmeFileDeviceHandle*.
    // Caller MUST NOT dereference on the host; pass it to a kernel.
    virtual NvmeFileDeviceHandle*
        acquire_device_handle(NvmeFile* file) = 0;

    // R5b: free the GPU handle.  Idempotent on nullptr.
    virtual void release_device_handle(NvmeFileDeviceHandle* dh) = 0;
};
```

`HostFsBackedNvmeStorage` implementation:

```
acquire_device_handle(file):
    if file->device->backend_private->attach_mode != DIRECT
        return nullptr        // not yet supported in service mode

    cpp_ctrl = file->device->backend_private->cpp_ctrl
    if cpp_ctrl == nullptr
        return nullptr        // shouldn't happen if registry built it

    NvmeFileDeviceHandle h_host;
    h_host.file_id              = file->file_id;
    h_host.logical_size_bytes   = file->size_bytes;
    h_host.header_bytes         = sizeof(NvmeFileHeader);
    h_host.nvme_block_size      = kNvmeBlockSize;
    h_host.nvme_block_size_log  = 12;
    h_host.namespace_id         = device->namespace_id;
    h_host.num_extents          = file->extents.size();
    memcpy(h_host.extents, file->extents.data(),
           sizeof(LbaExtent) * file->extents.size());
    h_host.d_qps      = cpp_ctrl->d_qps;
    h_host.num_d_qps  = cpp_ctrl->n_qps;

    NvmeFileDeviceHandle* dh = nullptr;
    cudaMalloc(&dh, sizeof(NvmeFileDeviceHandle));
    cudaMemcpy(dh, &h_host, sizeof(NvmeFileDeviceHandle),
               cudaMemcpyHostToDevice);
    return dh;
```

## 5  Smoke test

`nvme_storage/test/nvme_storage_gpu_smoke.cu` — patterned after
R5a's `nvme_storage_smoke` plus a kernel launch:

```
[ 1] cudaSetDevice
[ 2] LocalNvmeDirectRegistry::Open(N controllers,
        request 4 user queues per ctrl)
[ 3] HostFsBackedNvmeStorage::bootstrap
[ 4] for each device i in 0..N:
        create_file("gpu_smoke_<i>", 1 MiB)
[ 5] for each device i:
        write known pattern via write_blocking         (host pwrite)
[ 6] cudaMalloc + nvm_dma_map_device a 1 MiB GPU buffer per device
        (so we have a known prp1)
[ 7] for each device i:
        dh = storage.acquire_device_handle(file_i)
[ 8] launch 1-block, 1-thread kernel per device:
        kernel<<<1,1>>>(dh, prp1, /*prp2=*/0, /*off=*/0,
                        /*nbytes=*/4096):
            submit_read_one(dh, prp1, 0, 0, 4096)
[ 9] cudaDeviceSynchronize
[10] cudaMemcpy GPU buffer -> host, byte-compare with the
        pattern written in [5] (first 4 KiB only)
[11] release_device_handle(dh)
[12] storage.shutdown
[13] registry.Close
```

This is the **minimum** GPU-side smoke; later in R7 we'll add
batched / multi-thread variants.

## 6  Files affected (sketch)

| File | R5b action |
|------|-----------|
| `backends/local/nvme/libnvm/include/ctrl.h` | + wrap ctor + ctrl_borrowed flag in dtor |
| `device_manager/include/local_nvme_device.h` | + `std::shared_ptr<Controller> cpp_ctrl` |
| `device_manager/src/local_nvme_direct_registry.cpp` | + build Controller after init_b3 |
| `device_manager/src/nvmeservice_backed_registry.cpp` | (unchanged; cpp_ctrl stays nullptr) |
| `device_manager/CMakeLists.txt` | + link CUDA::cudart (Controller pulls cudaMemcpy) |
| `nvme_storage/include/nvme_file_device_handle.h` | NEW |
| `nvme_storage/include/queue_acquire_helper.cuh` | NEW |
| `nvme_storage/include/nvme_storage_device.cuh` | NEW (submit_*, resolve_lba) |
| `nvme_storage/include/nvme_storage.h` | + acquire_device_handle / release_device_handle on `INvmeStorage` |
| `nvme_storage/include/host_fs_backed_nvme_storage.h` | + override new methods |
| `nvme_storage/src/host_fs_backed_nvme_storage.cpp` | + impl new methods (cudaMalloc + memcpy) |
| `nvme_storage/test/nvme_storage_gpu_smoke.cu` | NEW |
| `nvme_storage/CMakeLists.txt` | + new headers + new smoke binary |

## 7  Open follow-ups, deferred

* **Service mode.**  `NvmeServiceBackedRegistry` does not yet
  produce `cpp_ctrl`.  Need to share `d_qps` across the daemon
  process boundary first, which is a separate item (probably
  R5c).  In the meantime `acquire_device_handle` on a service-mode
  device returns `nullptr` and the GPU smoke skips it.

* **Multi-extent files.**  R5a allows up to
  `kNvmeFileHeaderMaxExtents` extents per file.  R5b's
  `resolve_lba` must walk that table per-thread; for highly
  fragmented files this is O(num_extents) per IO.  Acceptable
  at R5b scale (typical: 1-3 extents); R7 may add a binary
  search if measured to matter.

* **NVMe FLUSH on close.**  `close_file` currently does
  `fsync(host_fd)`, which goes through the kernel block path.
  Once R5b GPU writes start landing without going through the
  kernel block layer, `close_file` must additionally issue an
  NVMe FLUSH command.  The FLUSH path requires either a host-
  callable libnvm helper (preferred) or a one-shot kernel; will
  be picked up at R7 when batch IO actually exercises this.

* **Doorbell host-VA → GPU-VA.**  `cpp_ctrl->d_qps` already has
  doorbell GPU VAs baked in via `cudaHostGetDevicePointer`
  (libnvm's existing path).  Multi-GPU + cross-PCIe-switch
  scenarios are tracked separately in `Todolist.md`.

## 8  Acceptance

R5b is done when:

1. `nvme_storage_gpu_smoke` runs green on a 3-NVMe machine
   (4b/57/63), 16+ steps, with byte-compare against the host-
   written pattern showing **all bytes** match.
2. R5a's existing `nvme_storage_smoke` continues to pass.
3. `registry_smoke` and `memory_smoke` continue to pass (no
   regressions on the lower layers).

Then commit; R6 (`block_storage`) starts.

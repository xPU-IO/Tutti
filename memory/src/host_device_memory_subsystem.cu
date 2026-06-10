/**
 * host_device_memory_subsystem.cu -- v0.1 IMemorySubsystem
 * implementation backed by malloc / cudaMalloc + libnvm DMA mapping.
 *
 * See host_device_memory_subsystem.h for the design rationale.
 *
 * R4.5 changes vs the original R3 cut:
 *   - No nvm_ctrl_t* in the constructor; DMA mappings are tracked
 *     per-(region, Device*) and are created on demand by
 *     register_tensor() walking bound devices.
 *   - prepare_nvme_dma() / prepare_rdma_mr() are gone.
 *   - set_descriptor_format() and descriptor_slice() are wired up
 *     (the latter as an Unimplemented stub).
 */

#include "host_device_memory_subsystem.h"
#include "cuda_helpers.cuh"

#include "../../device_manager/include/local_nvme_device.h"
#include "../../runtime/include/device.h"

#include <cuda_runtime.h>
#include <nvm_dma.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <cstdio>

namespace tutti {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Build a fresh, zeroed MemoryRegion ready to be filled in by the caller.
std::unique_ptr<MemoryRegion> make_region(uint64_t id) {
    auto r = std::make_unique<MemoryRegion>();
    std::memset(r.get(), 0, sizeof(MemoryRegion));
    r->region_id   = id;
    r->cuda_device = -1;
    return r;
}

// Pull the libnvm ctrl out of a tutti::Device that carries a
// LocalNvmeDevice payload.  Returns nullptr if the device is the
// wrong shape (which means caller bound a non-LocalNvme Device
// via bind_devices).
nvm_ctrl_t* ctrl_for(const Device* dev) {
    if (dev == nullptr || dev->backend_private == nullptr) return nullptr;
    auto* lnd = static_cast<LocalNvmeDevice*>(dev->backend_private);
    return lnd->ctrl;
}

} // namespace

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

HostDeviceMemorySubsystem::HostDeviceMemorySubsystem() = default;

HostDeviceMemorySubsystem::~HostDeviceMemorySubsystem() {
    // Tear down whatever the user forgot to free.  We loop locally
    // because erase_locked() mutates regions_.
    std::lock_guard<std::mutex> lock(mtx_);
    while (!regions_.empty()) {
        auto it = regions_.begin();
        Slot& slot = it->second;

        // Free the cluster-wide IO-slice table (GPU AddressDescriptor[]
        // blob + PRP-list buffer + PRP DMA handle) before unmapping
        // the data DMA -- IO-slice table teardown does not depend on
        // slot.data_dma but consistency wins.
        free_io_slice_table_locked(slot.io_slice_table);

        // Unmap the single data-buffer DMA handle (one nvm_dma_unmap
        // releases all per-ctrl bookkeeping snvme.ko built behind it).
        if (slot.data_dma != nullptr) {
            nvm_dma_unmap(slot.data_dma);
            slot.data_dma = nullptr;
        }

        if (slot.owns_host_alloc && slot.region->host_ptr != nullptr) {
            if (slot.region->kind == MemoryKind::PINNED_HOST) {
                cudaFreeHost(slot.region->host_ptr);
            } else {
                std::free(slot.region->host_ptr);
            }
        }
        if (slot.owns_device_alloc && slot.region->device_ptr != nullptr) {
            cudaFree(slot.region->device_ptr);
        }
        regions_.erase(it);
    }
}

// ---------------------------------------------------------------------------
// Descriptor format handshake
// ---------------------------------------------------------------------------

void HostDeviceMemorySubsystem::set_descriptor_format(DescriptorFormat fmt) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (fmt_ != DescriptorFormat::UNSET && fmt_ != fmt) {
        std::fprintf(stderr,
            "[memory] set_descriptor_format: changing %u -> %u (logic error)\n",
            (unsigned)fmt_, (unsigned)fmt);
        // Don't assert; in v0.1 just take the new value and warn.
    }
    fmt_ = fmt;
}

DescriptorFormat HostDeviceMemorySubsystem::descriptor_format() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return fmt_;
}

// ---------------------------------------------------------------------------
// Cluster device binding (called once at coordinator bring-up).
//
// Caches the cluster-wide caps (page_size + min MDTS) so every
// subsequent register_tensor + build_io_slice_table can size IO
// slicing without re-walking devices.  Replaces any previous
// binding -- callers that already have granularity > 0 regions
// registered MUST not re-bind in v0.1 (cached caps would no
// longer match the IO-slice tables; future revisions may rebuild).
// ---------------------------------------------------------------------------

void HostDeviceMemorySubsystem::bind_devices(
    const std::vector<const Device*>& devices)
{
    std::lock_guard<std::mutex> lock(mtx_);

    if (devices.empty()) {
        std::fprintf(stderr,
            "[memory] bind_devices: empty device list (logic error)\n");
        return;
    }

    // Validate every device exposes initialised caps; derive
    // cluster-wide page_size + min MDTS.
    std::size_t page_size = 0;
    std::size_t min_mdts  = 0;
    for (const Device* d : devices) {
        if (d == nullptr) {
            std::fprintf(stderr,
                "[memory] bind_devices: null entry in devices\n");
            return;
        }
        const std::size_t ps = d->capabilities.page_size;
        const std::size_t mb = d->capabilities.max_io_bytes;
        if (ps == 0 || mb == 0) {
            std::fprintf(stderr,
                "[memory] bind_devices: device %d caps not initialised "
                "(page_size=%zu max_io_bytes=%zu)\n",
                d->device_id, ps, mb);
            return;
        }
        if (page_size == 0) {
            page_size = ps;
            min_mdts  = mb;
        } else {
            if (ps != page_size) {
                std::fprintf(stderr,
                    "[memory] bind_devices: device %d page_size=%zu != "
                    "cluster page_size=%zu (logic error)\n",
                    d->device_id, ps, page_size);
                return;
            }
            if (mb < min_mdts) min_mdts = mb;
        }
    }

    bound_devices_     = devices;
    cluster_page_size_ = page_size;
    cluster_min_mdts_  = min_mdts;
}

// ---------------------------------------------------------------------------
// register_into_table -- the common path that adds a new MemoryRegion
// ---------------------------------------------------------------------------

MemoryRegion* HostDeviceMemorySubsystem::register_into_table(
    std::unique_ptr<MemoryRegion> r,
    bool owns_host_alloc,
    bool owns_device_alloc)
{
    std::lock_guard<std::mutex> lock(mtx_);

    uint64_t id = r->region_id;
    Slot slot;
    slot.region            = std::move(r);
    slot.owns_host_alloc   = owns_host_alloc;
    slot.owns_device_alloc = owns_device_alloc;

    auto [it, ok] = regions_.emplace(id, std::move(slot));
    if (!ok) {
        std::fprintf(stderr, "[memory] register_into_table: id=%lu collision\n",
                     (unsigned long)id);
        return nullptr;
    }
    return it->second.region.get();
}

void HostDeviceMemorySubsystem::erase_locked(uint64_t region_id) {
    auto it = regions_.find(region_id);
    if (it == regions_.end()) return;

    Slot& slot = it->second;
    // R7: free the cluster-wide IO-slice table (descriptor blob +
    // PRP-list buffer + PRP DMA handle) first.
    free_io_slice_table_locked(slot.io_slice_table);

    // Drop the single data-buffer DMA handle (snvme.ko cleans up
    // every ctrl's mapping behind it on a single nvm_dma_unmap).
    if (slot.data_dma != nullptr) {
        nvm_dma_unmap(slot.data_dma);
        slot.data_dma = nullptr;
    }

    if (slot.owns_host_alloc && slot.region->host_ptr != nullptr) {
        if (slot.region->kind == MemoryKind::PINNED_HOST) {
            cudaFreeHost(slot.region->host_ptr);
        } else {
            std::free(slot.region->host_ptr);
        }
    }
    if (slot.owns_device_alloc && slot.region->device_ptr != nullptr) {
        cudaFree(slot.region->device_ptr);
    }
    regions_.erase(it);
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

MemoryRegion* HostDeviceMemorySubsystem::allocate_host(
    std::size_t size, MemoryKind kind)
{
    if (size == 0) return nullptr;
    if (kind != MemoryKind::HOST && kind != MemoryKind::PINNED_HOST) {
        std::fprintf(stderr, "[memory] allocate_host: unsupported kind=%u\n",
                     (unsigned)kind);
        return nullptr;
    }

    void* ptr = nullptr;
    if (kind == MemoryKind::PINNED_HOST) {
        if (cudaMallocHost(&ptr, size) != cudaSuccess) return nullptr;
    } else {
        ptr = std::malloc(size);
        if (ptr == nullptr) return nullptr;
    }

    uint64_t id;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        id = next_region_id_++;
    }
    auto r = make_region(id);
    r->kind     = kind;
    r->host_ptr = ptr;
    r->size     = size;

    return register_into_table(std::move(r),
                                /*owns_host_alloc=*/true,
                                /*owns_device_alloc=*/false);
}

MemoryRegion* HostDeviceMemorySubsystem::allocate_device(
    std::size_t size, MemoryKind kind, int device_id)
{
    if (size == 0) return nullptr;

    if (kind == MemoryKind::DEVICE) {
        if (device_id < 0) return nullptr;
        if (cudaSetDevice(device_id) != cudaSuccess) return nullptr;
        void* ptr = nullptr;
        if (cudaMalloc(&ptr, size) != cudaSuccess) return nullptr;

        uint64_t id;
        { std::lock_guard<std::mutex> lock(mtx_); id = next_region_id_++; }
        auto r = make_region(id);
        r->kind        = MemoryKind::DEVICE;
        r->cuda_device = device_id;
        r->device_ptr  = ptr;
        r->size        = size;
        return register_into_table(std::move(r), false, true);
    }

    if (kind == MemoryKind::MANAGED) {
        void* ptr = nullptr;
        if (cudaMallocManaged(&ptr, size) != cudaSuccess) return nullptr;

        uint64_t id;
        { std::lock_guard<std::mutex> lock(mtx_); id = next_region_id_++; }
        auto r = make_region(id);
        r->kind        = MemoryKind::MANAGED;
        r->cuda_device = device_id;
        r->host_ptr    = ptr;
        r->device_ptr  = ptr;
        r->size        = size;
        return register_into_table(std::move(r), false, true);
    }

    std::fprintf(stderr, "[memory] allocate_device: unsupported kind=%u\n",
                 (unsigned)kind);
    return nullptr;
}

void HostDeviceMemorySubsystem::free(MemoryRegion* region) {
    if (region == nullptr) return;
    std::lock_guard<std::mutex> lock(mtx_);
    erase_locked(region->region_id);
}

// ---------------------------------------------------------------------------
// Registration of caller-allocated buffers
// ---------------------------------------------------------------------------

MemoryRegion* HostDeviceMemorySubsystem::register_host(
    void* host_ptr, std::size_t size)
{
    if (host_ptr == nullptr || size == 0) return nullptr;
    uint64_t id;
    { std::lock_guard<std::mutex> lock(mtx_); id = next_region_id_++; }
    auto r = make_region(id);
    r->kind     = MemoryKind::HOST;
    r->host_ptr = host_ptr;
    r->size     = size;
    return register_into_table(std::move(r), false, false);
}

MemoryRegion* HostDeviceMemorySubsystem::register_device(
    void* device_ptr, std::size_t size, int device_id)
{
    if (device_ptr == nullptr || size == 0 || device_id < 0) return nullptr;
    uint64_t id;
    { std::lock_guard<std::mutex> lock(mtx_); id = next_region_id_++; }
    auto r = make_region(id);
    r->kind        = MemoryKind::DEVICE;
    r->cuda_device = device_id;
    r->device_ptr  = device_ptr;
    r->size        = size;
    return register_into_table(std::move(r), false, false);
}

MemoryRegion* HostDeviceMemorySubsystem::register_external(
    void* host_ptr, void* device_ptr,
    std::size_t size, const ExternalMemorySpec& spec)
{
    if (size == 0) return nullptr;
    if (host_ptr == nullptr && device_ptr == nullptr) return nullptr;

    if (spec.source != ExternalMemorySource::APP_MANAGED) {
        std::fprintf(stderr,
            "[memory] register_external: source=%u not implemented in v0.1\n",
            (unsigned)spec.source);
        return nullptr;
    }

    uint64_t id;
    { std::lock_guard<std::mutex> lock(mtx_); id = next_region_id_++; }
    auto r = make_region(id);
    r->kind       = MemoryKind::EXTERNAL;
    r->host_ptr   = host_ptr;
    r->device_ptr = device_ptr;
    r->size       = size;
    r->external   = spec;
    return register_into_table(std::move(r), false, false);
}

void HostDeviceMemorySubsystem::unregister(MemoryRegion* region) {
    if (region == nullptr) return;
    std::lock_guard<std::mutex> lock(mtx_);
    erase_locked(region->region_id);
}

// ---------------------------------------------------------------------------
// register_tensor -- high-level entry-point that drives DMA mapping
// ---------------------------------------------------------------------------

HostDeviceMemorySubsystem::Slot*
HostDeviceMemorySubsystem::slot_by_ptr_locked(const void* ptr) {
    if (ptr == nullptr) return nullptr;
    const auto* needle = static_cast<const uint8_t*>(ptr);
    for (auto& [id, slot] : regions_) {
        const auto* hbase = static_cast<const uint8_t*>(slot.region->host_ptr);
        if (hbase != nullptr &&
            needle >= hbase && needle < hbase + slot.region->size) {
            return &slot;
        }
        const auto* dbase = static_cast<const uint8_t*>(slot.region->device_ptr);
        if (dbase != nullptr &&
            needle >= dbase && needle < dbase + slot.region->size) {
            return &slot;
        }
    }
    return nullptr;
}

bool HostDeviceMemorySubsystem::ensure_mapping_locked(Slot& slot)
{
    // Idempotent: already mapped means no-op.
    if (slot.data_dma != nullptr) return true;

    if (bound_devices_.empty()) {
        std::fprintf(stderr,
            "[memory] ensure_mapping: bind_devices() not called\n");
        return false;
    }

    // Use any cluster-bound ctrl as the proxy.  snvme.ko's
    // NVM_MAP_DEVICE_MEMORY ioctl loops over every open ctrl
    // internally (see snvme/map.c map_gpu_memory line 506-532),
    // so a single nvm_dma_map_data_* call is sufficient and the
    // resulting nvm_dma_t.ioaddrs[] is usable from every
    // cluster-bound ctrl (IOMMU=pt deployment contract).
    const Device* ref_dev = bound_devices_.front();
    nvm_ctrl_t* ctrl = ctrl_for(ref_dev);
    if (ctrl == nullptr) {
        std::fprintf(stderr,
            "[memory] ensure_mapping: ref device %d has no libnvm ctrl\n",
            ref_dev->device_id);
        return false;
    }

    nvm_dma_t* dma = nullptr;
    int rc = -1;
    if (slot.region->device_ptr != nullptr) {
        rc = nvm_dma_map_data_device(&dma, ctrl,
                                      slot.region->device_ptr,
                                      slot.region->size);
    } else if (slot.region->host_ptr != nullptr) {
        rc = nvm_dma_map_data_host(&dma, ctrl,
                                    slot.region->host_ptr,
                                    slot.region->size);
    } else {
        return false;
    }

    if (rc != 0 || dma == nullptr) {
        std::fprintf(stderr,
            "[memory] ensure_mapping: nvm_dma_map_data_* rc=%d dma=%p\n",
            rc, (void*)dma);
        return false;
    }

    slot.data_dma = dma;
    return true;
}

MemoryRegion* HostDeviceMemorySubsystem::register_tensor(
    const TensorRegistrationSpec& spec)
{
    if (spec.ptr == nullptr || spec.size == 0) return nullptr;

    // Step 1: find or create a region for spec.ptr.
    MemoryRegion* region = nullptr;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (Slot* existing = slot_by_ptr_locked(spec.ptr)) {
            region = existing->region.get();
        }
    }

    if (region == nullptr) {
        // Not registered yet -- classify by pointer attributes.
        cudaPointerAttributes attr{};
        cudaError_t cerr = cudaPointerGetAttributes(&attr, spec.ptr);
        if (cerr != cudaSuccess) {
            // Plain host pointer that CUDA doesn't know about.
            // Treat as host buffer.
            (void)cudaGetLastError();   // clear sticky error
            region = register_host(spec.ptr, spec.size);
        } else if (attr.type == cudaMemoryTypeDevice) {
            int dev = attr.device;
            region = register_device(spec.ptr, spec.size, dev);
        } else if (attr.type == cudaMemoryTypeHost ||
                   attr.type == cudaMemoryTypeManaged) {
            region = register_host(spec.ptr, spec.size);
        } else {
            region = register_host(spec.ptr, spec.size);
        }
        if (region == nullptr) return nullptr;
    }

    // Step 2: ensure DMA mapping (single call, see deployment
    //         contract on Slot::data_dma -- snvme.ko handles
    //         multi-ctrl bookkeeping internally on the one ioctl
    //         we issue here).
    {
        std::lock_guard<std::mutex> lock(mtx_);
        Slot* slot = slot_by_ptr_locked(spec.ptr);
        if (slot == nullptr) return nullptr;
        if (bound_devices_.empty()) {
            std::fprintf(stderr,
                "[memory] register_tensor: bind_devices() must be called "
                "before register_tensor (no cluster device set)\n");
            return nullptr;
        }
        if (!ensure_mapping_locked(*slot)) {
            std::fprintf(stderr,
                "[memory] register_tensor: mapping failed\n");
            return nullptr;
        }

        // R7: pre-compute the cluster-wide IO-slice table when the
        // caller asks for it via spec.granularity > 0.  When
        // granularity == 0 we skip the slice build entirely (R3
        // raw-DMA-only path is source-compatible).
        //
        // ONE table per region (not per device): PCI bus addresses
        // are controller-agnostic on the host-NVMe direct path, so
        // a single PRP/SGL list + a single AddressDescriptor[] blob
        // serve every cluster-bound controller.
        //
        // Idempotency: if a slice table already exists for this
        // region (caller called register_tensor with granularity > 0
        // earlier), build_io_slice_table_locked is a no-op and
        // returns success.  Caller must unregister(region) before
        // re-registering with a different granularity.
        if (spec.granularity > 0) {
            if (!build_io_slice_table_locked(*slot, spec)) {
                std::fprintf(stderr,
                    "[memory] register_tensor: build_io_slice_table failed\n");
                return nullptr;
            }
        }
    }

    return region;
}

// ---------------------------------------------------------------------------
// R7: IO-slice table construction + lookup
//
// Cluster-wide design (one table per region, NOT per device): on the
// host-NVMe direct path a GPU physical page has the same PCI bus
// address in every NVMe controller's view, so one PRP/SGL list +
// one AddressDescriptor[] blob serves every controller in
// bound devices.  effective_io is therefore sized to the
// weakest controller (min over bound devices of caps.max_io_bytes).
//
// PRP-list buffer ownership: build_io_slice_table_locked() allocates
// the PRP-list scratch buffer INTERNALLY (cudaMalloc + per-ctrl
// nvm_dma_map_data_device) and stores it in IoSliceTable.  Callers
// do NOT supply the buffer (this matches legacy
// GPUController::initializePRPList() behaviour).
//
// build_io_slice_table_locked() is decomposed into 9 single-purpose
// stages so that each can be reasoned about (and unit-tested) on its
// own.  The stages are, top-down:
//
//   1. caps + ref_dev resolution    -- pick the weakest MDTS as the
//                                       cluster-wide bound; pick any
//                                       ctrl as the ioaddrs[] source.
//   2. validate_io_slice_alignment  -- spec.ptr/size/granularity
//                                       must satisfy 4 KiB alignment.
//   3. compute_io_slice_plan        -- pure derivation:
//                                       (granularity, min_mdts) -> plan.
//   4. idempotent skip              -- if an IO-slice table already
//                                       exists for this region,
//                                       return success (no rebuild).
//                                       Caller must unregister
//                                       first to change granularity.
//   5. allocate + DMA-map PRP buf   -- (only if needs_prp_list)
//                                       internal cudaMalloc + ONE
//                                       nvm_dma_map_data_device
//                                       call (ref_dev as proxy;
//                                       snvme handles multi-ctrl
//                                       internally).
//   6. fill_address_descriptors     -- pure: walk the (slice, io)
//                                       grid, populate host-side
//                                       AddressDescriptor[] +
//                                       PRP-list pages.
//   7. upload_descriptors_to_gpu    -- cudaMalloc + cudaMemcpy the
//                                       AddressDescriptor[] blob.
//   8. upload_prp_list_pages        -- cudaMemcpy the PRP-list
//                                       pages into our GPU scratch
//                                       buffer (ONE copy serves
//                                       every ctrl).
//   9. build_slice_views            -- pure: assemble the host-
//                                       resident slice_addr-sorted
//                                       lookup index.
// ---------------------------------------------------------------------------

namespace {

// Plan derived purely from (TensorRegistrationSpec, page_size, MDTS).
// Held by value so every stage below can be `const`-correct.
struct IoSliceBuildPlan {
    std::size_t   page_size       = 0;
    std::size_t   bytes_per_slice = 0;
    std::size_t   effective_io    = 0;
    std::size_t   pages_per_io    = 0;
    std::uint32_t ios_per_slice   = 0;
    std::uint32_t num_slices      = 0;
    std::uint32_t total_ios       = 0;
    bool          needs_prp_list  = false;
};

// Stage 1 -- spec alignment validation.
bool validate_io_slice_alignment(const TensorRegistrationSpec& spec,
                                 std::size_t page_size) {
    const auto base_addr = reinterpret_cast<std::uintptr_t>(spec.ptr);
    if ((base_addr % page_size) != 0) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: spec.ptr=%p not aligned to "
            "page_size=%zu\n", spec.ptr, page_size);
        return false;
    }
    if ((spec.size % page_size) != 0) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: spec.size=%zu not a multiple of "
            "page_size=%zu\n", spec.size, page_size);
        return false;
    }
    if (spec.granularity != 0) {
        if ((spec.granularity % page_size) != 0) {
            std::fprintf(stderr,
                "[memory] build_io_slice_table: granularity=%zu not a "
                "multiple of page_size=%zu\n", spec.granularity, page_size);
            return false;
        }
        if ((spec.size % spec.granularity) != 0) {
            std::fprintf(stderr,
                "[memory] build_io_slice_table: spec.size=%zu not a multiple "
                "of granularity=%zu\n", spec.size, spec.granularity);
            return false;
        }
    }
    return true;
}

// Stage 2 -- pure derivation.  Caller must have validated alignment.
IoSliceBuildPlan compute_io_slice_plan(const TensorRegistrationSpec& spec,
                                       std::size_t page_size,
                                       std::size_t mdts) {
    IoSliceBuildPlan p{};
    p.page_size       = page_size;
    p.bytes_per_slice = (spec.granularity == 0) ? spec.size : spec.granularity;
    p.effective_io    = std::min<std::size_t>(p.bytes_per_slice, mdts);
    p.pages_per_io    = (p.effective_io + page_size - 1) / page_size;
    p.ios_per_slice   = static_cast<std::uint32_t>(
        (p.bytes_per_slice + p.effective_io - 1) / p.effective_io);
    p.num_slices      = static_cast<std::uint32_t>(spec.size / p.bytes_per_slice);
    p.total_ios       = p.num_slices * p.ios_per_slice;
    p.needs_prp_list  = (p.pages_per_io > 2);
    return p;
}

// Stage 3 -- the data buffer's nvm_dma_t is what we'll read IOVAs
// from while filling AddressDescriptor[].  Verify it covers the
// whole region and reports the device's page size.
bool validate_data_dma(const nvm_dma_t* data_dma,
                       std::size_t      page_size,
                       std::size_t      region_size,
                       int              device_id) {
    if (data_dma == nullptr) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: data dma is null for device %d\n",
            device_id);
        return false;
    }
    if (data_dma->page_size != page_size) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: data dma page_size=%zu != "
            "device page_size=%zu\n",
            (std::size_t)data_dma->page_size, page_size);
        return false;
    }
    const std::size_t needed_data_pages = region_size / page_size;
    if ((std::size_t)data_dma->n_ioaddrs < needed_data_pages) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: data dma n_ioaddrs=%zu < "
            "required %zu\n",
            (std::size_t)data_dma->n_ioaddrs, needed_data_pages);
        return false;
    }
    return true;
}

// Stage 4b -- sanity-check the freshly-built PRP-list nvm_dma_t.
// We allocated/mapped this ourselves, so the only failure modes
// here would be libnvm/snvme returning something unexpected.
bool validate_prp_list_dma(const nvm_dma_t* prp_dma,
                           std::size_t      page_size,
                           std::uint32_t    total_ios) {
    if (prp_dma == nullptr) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: prp_list dma is null\n");
        return false;
    }
    if (prp_dma->page_size != page_size) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: prp_list dma page_size=%zu != "
            "device page_size=%zu\n",
            (std::size_t)prp_dma->page_size, page_size);
        return false;
    }
    if ((std::size_t)prp_dma->n_ioaddrs < total_ios) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: prp_list dma n_ioaddrs=%zu < "
            "total_ios %u\n",
            (std::size_t)prp_dma->n_ioaddrs, total_ios);
        return false;
    }
    return true;
}

// Stage 5 -- pure: walk every (slice, io) cell and emit a host-side
// AddressDescriptor + (when needed) a packed PRP-list page.
//
// Layout of h_prp_pages: total_ios pages laid out contiguously, each
// page_size bytes long; PRP page i starts at byte offset
// `i * page_size` (i.e. element offset `i * entries_per_page`).
void fill_address_descriptors(const IoSliceBuildPlan&         plan,
                              const nvm_dma_t*                data_dma,
                              const nvm_dma_t*                prp_dma,
                              std::vector<AddressDescriptor>& h_entries,
                              std::vector<std::uint64_t>&     h_prp_pages) {
    const std::size_t entries_per_page =
        plan.page_size / sizeof(std::uint64_t);
    h_entries.assign(plan.total_ios, AddressDescriptor{});
    if (plan.needs_prp_list) {
        h_prp_pages.assign(
            (std::size_t)plan.total_ios * entries_per_page, 0ull);
    }

    for (std::uint32_t g = 0; g < plan.num_slices; ++g) {
        for (std::uint32_t i = 0; i < plan.ios_per_slice; ++i) {
            const std::uint32_t io_idx      = g * plan.ios_per_slice + i;
            const std::uint64_t io_byte_off =
                (std::uint64_t)g * plan.bytes_per_slice
                + (std::uint64_t)i * plan.effective_io;

            // Last IO of a slice may be short when granularity is
            // not a multiple of effective_io.  R7 spec checks
            // currently rule that out, but stay conservative.
            std::uint64_t io_size = plan.effective_io;
            const std::uint64_t slice_end =
                (std::uint64_t)(g + 1) * plan.bytes_per_slice;
            if (io_byte_off + io_size > slice_end) {
                io_size = slice_end - io_byte_off;
            }
            const std::uint32_t pages_in_io =
                (std::uint32_t)((io_size + plan.page_size - 1)
                                / plan.page_size);
            const std::size_t start_page =
                (std::size_t)(io_byte_off / plan.page_size);

            AddressDescriptor& d = h_entries[io_idx];
            d.data_length = io_size;
            d.prp1        = data_dma->ioaddrs[start_page];
            if (pages_in_io == 1) {
                d.prp2 = 0;
            } else if (pages_in_io == 2) {
                d.prp2 = data_dma->ioaddrs[start_page + 1];
            } else {
                // PRP-list path.  prp2 = IOVA of the PRP-list page.
                // Pack data IOVAs[start_page+1 .. start_page+pages_in_io-1]
                // into the scratch page at element offset
                // io_idx * entries_per_page.  Remaining entries stay
                // 0 (the controller stops at pages_in_io - 1).
                d.prp2 = prp_dma->ioaddrs[io_idx];
                std::uint64_t* page = h_prp_pages.data()
                                      + (std::size_t)io_idx * entries_per_page;
                for (std::uint32_t p = 1; p < pages_in_io; ++p) {
                    page[p - 1] = data_dma->ioaddrs[start_page + p];
                }
            }
        }
    }
}

// Stage 6 -- cudaMalloc the contiguous AddressDescriptor[] blob and
// upload it.  On success, *out_d_descriptors owns a fresh GPU
// allocation and *out_total holds the entry count.
bool upload_descriptors_to_gpu(
    const std::vector<AddressDescriptor>& h_entries,
    AddressDescriptor*&                   out_d_descriptors,
    std::size_t&                          out_total) {
    const std::size_t bytes = h_entries.size() * sizeof(AddressDescriptor);
    cudaError_t cerr = cudaMalloc(&out_d_descriptors, bytes);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: cudaMalloc(%zu) failed: %s\n",
            bytes, cudaGetErrorString(cerr));
        out_d_descriptors = nullptr;
        return false;
    }
    cerr = cudaMemcpy(out_d_descriptors, h_entries.data(), bytes,
                      cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: cudaMemcpy(entries) failed: "
            "%s\n", cudaGetErrorString(cerr));
        cudaFree(out_d_descriptors);
        out_d_descriptors = nullptr;
        return false;
    }
    out_total = h_entries.size();
    return true;
}

// Stage 7 -- copy the packed PRP-list pages into the GPU scratch
// buffer that build_io_slice_table_locked just cudaMalloc'd.  Must
// be called only when plan.needs_prp_list.
bool upload_prp_list_pages(
    void*                                 prp_list_devptr,
    const std::vector<std::uint64_t>&     h_prp_pages,
    const IoSliceBuildPlan&               plan) {
    cudaError_t cerr = cudaMemcpy(prp_list_devptr,
                                  h_prp_pages.data(),
                                  (std::size_t)plan.total_ios * plan.page_size,
                                  cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: cudaMemcpy(prp_pages) failed: "
            "%s\n", cudaGetErrorString(cerr));
        return false;
    }
    return true;
}

// Stage 8 -- pure: build the host-side slice_addr-sorted index.
// slice_addr keys use the GPU-visible base when the region is device
// memory, host base otherwise (callers MUST use the same flavour to
// look up).  num_slices is small enough (today ~hundreds) that an
// O(num_slices) memcpy + later O(log N) bsearch is fine.
void build_slice_views(const MemoryRegion*       region,
                       const IoSliceBuildPlan&   plan,
                       AddressDescriptor*        d_all_descriptors,
                       std::vector<IoSliceView>& views) {
    const std::uint64_t base_ptr =
        (region->device_ptr != nullptr)
            ? reinterpret_cast<std::uint64_t>(region->device_ptr)
            : reinterpret_cast<std::uint64_t>(region->host_ptr);
    views.clear();
    views.reserve(plan.num_slices);
    for (std::uint32_t g = 0; g < plan.num_slices; ++g) {
        IoSliceView v{};
        v.slice_addr  = base_ptr + (std::uint64_t)g * plan.bytes_per_slice;
        v.num_ios     = plan.ios_per_slice;
        v.total_bytes = plan.bytes_per_slice;
        v.d_ios       = d_all_descriptors
                        + (std::size_t)g * plan.ios_per_slice;
        views.push_back(v);
    }
}

} // namespace

void HostDeviceMemorySubsystem::free_io_slice_table_locked(IoSliceTable& tab) {
    // Order matters: drop kernel DMA bookkeeping first, then GPU
    // memory allocations.  All idempotent (clear-on-empty).
    if (tab.prp_list_dma != nullptr) {
        nvm_dma_unmap(tab.prp_list_dma);
        tab.prp_list_dma = nullptr;
    }
    if (tab.prp_list_devptr != nullptr) {
        cudaFree(tab.prp_list_devptr);
        tab.prp_list_devptr = nullptr;
    }
    tab.prp_list_bytes = 0;

    if (tab.d_all_descriptors != nullptr) {
        cudaFree(tab.d_all_descriptors);
        tab.d_all_descriptors = nullptr;
    }
    tab.total_descriptors = 0;
    tab.views.clear();
}

bool HostDeviceMemorySubsystem::build_io_slice_table_locked(
    Slot& slot, const TensorRegistrationSpec& spec)
{
    // ---- 1. Resolve cluster-wide caps from bound devices.
    //          page_size: taken from the reference (first) device,
    //                     assumed uniform across NVMe controllers
    //                     in the host-direct path (all 4 KiB today).
    //          min_mdts:  the weakest controller's MDTS bounds the
    //                     L2 IO size for the whole cluster.
    //          ref_dev :  any device whose nvm_dma_t we can use as
    //                     an ioaddrs[] source (PCI bus addresses are
    //                     controller-agnostic, so picking the first
    //                     mapped controller is fine). ----------------
    // ---- 1. Cluster caps (cached by bind_devices()).
    //          page_size: device caps are required to be uniform
    //                     across cluster-bound NVMe controllers
    //                     (validated by bind_devices); all 4 KiB
    //                     today.
    //          min_mdts:  the weakest controller's MDTS bounds the
    //                     L2 IO size for the whole cluster.
    //          ref_dev :  any cluster-bound device serves as the
    //                     ioaddrs[] source -- PCI bus addresses are
    //                     controller-agnostic in the host-NVMe
    //                     direct path.  We pick bound_devices_[0]. -
    if (bound_devices_.empty() ||
        cluster_page_size_ == 0 || cluster_min_mdts_ == 0) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: bind_devices() not called or "
            "produced empty cluster caps (page_size=%zu min_mdts=%zu)\n",
            cluster_page_size_, cluster_min_mdts_);
        return false;
    }
    const Device*    ref_dev   = bound_devices_.front();
    const std::size_t page_size = cluster_page_size_;
    const std::size_t min_mdts  = cluster_min_mdts_;

    // ---- 2. Spec alignment + plan derivation ----------------------
    if (!validate_io_slice_alignment(spec, page_size)) return false;
    const IoSliceBuildPlan plan =
        compute_io_slice_plan(spec, page_size, min_mdts);
    if (plan.total_ios == 0) return false;

    // ---- 3. The data buffer's single DMA handle was created by
    //          ensure_mapping_locked at register_tensor time.
    //          Defensive check: should never fire under the
    //          register_tensor invariant. -----------------------
    nvm_dma_t* data_dma = slot.data_dma;
    if (data_dma == nullptr) {
        std::fprintf(stderr,
            "[memory] build_io_slice_table: data buffer not DMA-mapped "
            "(register_tensor invariant violated)\n");
        return false;
    }
    if (!validate_data_dma(data_dma, page_size, spec.size,
                            ref_dev->device_id)) {
        return false;
    }

    // ---- 4. Idempotent skip when an IO-slice table already exists.
    //          register_tensor() is strictly idempotent (matches
    //          legacy GPUController::registerTesnsorList semantics
    //          which warns + skips on duplicate registration);
    //          to rebuild with a different granularity caller must
    //          unregister(region) first. ---------------------------
    IoSliceTable& tab = slot.io_slice_table;
    if (tab.total_descriptors > 0) {
        return true;
    }

    // ---- 5. Allocate + DMA-map the PRP-list buffer (only when
    //          plan.needs_prp_list).  Owned by `tab` and freed by
    //          free_io_slice_table_locked().  Sized to
    //          plan.total_ios * page_size (one 4 KiB PRP page per
    //          IO).  ONE nvm_dma_map_data_device call here -- same
    //          deployment contract as Slot::data_dma above
    //          (snvme.ko handles the multi-ctrl bookkeeping
    //          internally on the single ioctl). ------------------
    nvm_dma_t* prp_dma = nullptr;
    if (plan.needs_prp_list) {
        tab.prp_list_bytes = (std::size_t)plan.total_ios * page_size;
        cudaError_t cerr =
            cudaMalloc(&tab.prp_list_devptr, tab.prp_list_bytes);
        if (cerr != cudaSuccess) {
            std::fprintf(stderr,
                "[memory] build_io_slice_table: cudaMalloc(prp_list, "
                "%zu B) failed: %s\n",
                tab.prp_list_bytes, cudaGetErrorString(cerr));
            tab.prp_list_devptr = nullptr;
            tab.prp_list_bytes  = 0;
            return false;
        }

        nvm_ctrl_t* ctrl = ctrl_for(ref_dev);
        if (ctrl == nullptr) {
            std::fprintf(stderr,
                "[memory] build_io_slice_table: ref device %d has no "
                "libnvm ctrl (PRP-list mapping)\n", ref_dev->device_id);
            free_io_slice_table_locked(tab);
            return false;
        }
        int rc = nvm_dma_map_data_device(&tab.prp_list_dma, ctrl,
                                          tab.prp_list_devptr,
                                          tab.prp_list_bytes);
        if (rc != 0 || tab.prp_list_dma == nullptr) {
            std::fprintf(stderr,
                "[memory] build_io_slice_table: nvm_dma_map_data_device"
                "(prp_list) rc=%d dma=%p\n",
                rc, (void*)tab.prp_list_dma);
            tab.prp_list_dma = nullptr;
            free_io_slice_table_locked(tab);
            return false;
        }
        prp_dma = tab.prp_list_dma;
        if (!validate_prp_list_dma(prp_dma, page_size, plan.total_ios)) {
            free_io_slice_table_locked(tab);
            return false;
        }
    }

    // ---- 6. Build host-side AddressDescriptor[] + PRP-list pages --
    std::vector<AddressDescriptor> h_entries;
    std::vector<std::uint64_t>     h_prp_pages;
    fill_address_descriptors(plan, data_dma, prp_dma, h_entries, h_prp_pages);

    // ---- 7. Upload the descriptor blob to GPU ---------------------
    if (!upload_descriptors_to_gpu(h_entries,
                                    tab.d_all_descriptors,
                                    tab.total_descriptors)) {
        free_io_slice_table_locked(tab);
        return false;
    }

    // ---- 8. Upload PRP-list pages (if any) into our own GPU
    //          scratch buffer.  ONE copy serves every controller
    //          across bound devices (PCI bus addresses are
    //          controller-agnostic). -----------------------------
    if (plan.needs_prp_list) {
        if (!upload_prp_list_pages(tab.prp_list_devptr, h_prp_pages, plan)) {
            free_io_slice_table_locked(tab);
            return false;
        }
    }

    // ---- 9. Build host-side slice_addr-sorted view index ----------
    build_slice_views(slot.region.get(), plan, tab.d_all_descriptors,
                      tab.views);

    return true;
}

const IoSliceView*
HostDeviceMemorySubsystem::lookup_io_slice(MemoryRegion* region,
                                          uint64_t      slice_addr) const
{
    if (region == nullptr) return nullptr;
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = regions_.find(region->region_id);
    if (it == regions_.end()) return nullptr;
    const Slot& slot = it->second;
    const auto& views = slot.io_slice_table.views;
    if (views.empty()) return nullptr;
    // views[] is sorted ascending by slice_addr (built that way).
    auto vit = std::lower_bound(views.begin(), views.end(), slice_addr,
        [](const IoSliceView& a, std::uint64_t b) {
            return a.slice_addr < b;
        });
    if (vit == views.end() || vit->slice_addr != slice_addr) return nullptr;
    return &(*vit);
}

std::vector<IoSliceView>
HostDeviceMemorySubsystem::list_io_slices(MemoryRegion* region) const
{
    if (region == nullptr) return {};
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = regions_.find(region->region_id);
    if (it == regions_.end()) return {};
    return it->second.io_slice_table.views;
}

// ---------------------------------------------------------------------------
// descriptor_slice -- v0.1 stub kept as a placeholder for the
// "ad-hoc slice without prior granularity registration" path.
// ---------------------------------------------------------------------------

bool HostDeviceMemorySubsystem::descriptor_slice(
    MemoryRegion*       /*region*/,
    const Device*       /*device*/,
    uint64_t            /*byte_offset*/,
    uint64_t            /*byte_length*/,
    AddressDescriptor*  /*out*/,
    std::size_t*        /*inout_count*/)
{
    // R7 lands the real PRP builder; R8 the SGL fallback.
    // Until then upper layers query the raw mapping via
    // query_nvme_mapping() (test-only) or through nvme_storage's
    // own NVMe-format helpers.
    return false;
}

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

MemoryRegion* HostDeviceMemorySubsystem::lookup(
    const MemoryLookupKey& key) const
{
    std::lock_guard<std::mutex> lock(mtx_);

    for (const auto& [id, slot] : regions_) {
        switch (key.by) {
        case MemoryLookupKey::By::REGION_ID:
            if (id == key.region_id) return slot.region.get();
            break;
        case MemoryLookupKey::By::HOST_PTR: {
            const auto* base = static_cast<const uint8_t*>(slot.region->host_ptr);
            if (base == nullptr) break;
            const auto* needle = static_cast<const uint8_t*>(key.ptr);
            if (needle >= base && needle < base + slot.region->size)
                return slot.region.get();
            break;
        }
        case MemoryLookupKey::By::DEVICE_PTR: {
            const auto* base = static_cast<const uint8_t*>(slot.region->device_ptr);
            if (base == nullptr) break;
            const auto* needle = static_cast<const uint8_t*>(key.ptr);
            if (needle >= base && needle < base + slot.region->size)
                return slot.region.get();
            break;
        }
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Test-only knobs
// ---------------------------------------------------------------------------

std::size_t HostDeviceMemorySubsystem::region_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return regions_.size();
}

bool HostDeviceMemorySubsystem::query_nvme_mapping(
    const MemoryRegion* region,
    const Device*       device,
    std::size_t*        out_count,
    std::size_t*        out_page_size,
    uint64_t*           out_first_ioaddr) const
{
    // `device` is retained for ABI compatibility with R4.5 callers
    // but ignored in R7+: each region has a single cluster-wide
    // nvm_dma_t whose ioaddrs[] are usable from every cluster-bound
    // ctrl (see Slot::data_dma deployment contract).
    (void)device;
    if (region == nullptr) return false;

    std::lock_guard<std::mutex> lock(mtx_);
    auto it = regions_.find(region->region_id);
    if (it == regions_.end()) return false;
    const Slot& slot = it->second;

    nvm_dma_t* dma = slot.data_dma;
    if (dma == nullptr) return false;

    if (out_count)        *out_count        = (std::size_t)dma->n_ioaddrs;
    if (out_page_size)    *out_page_size    = (std::size_t)dma->page_size;
    if (out_first_ioaddr) *out_first_ioaddr = (dma->n_ioaddrs > 0)
                                                ? (uint64_t)dma->ioaddrs[0]
                                                : 0ULL;
    return true;
}

} // namespace tutti

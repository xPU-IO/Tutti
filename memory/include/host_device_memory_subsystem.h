#ifndef __TUTTI_MEMORY_HOST_DEVICE_MEMORY_SUBSYSTEM_H__
#define __TUTTI_MEMORY_HOST_DEVICE_MEMORY_SUBSYSTEM_H__

/**
 * host_device_memory_subsystem.h -- the v0.1 IMemorySubsystem
 * implementation.
 *
 * Layer: Memory.
 *
 * R4.5 changes (vs the original R3 cut):
 *   - Constructor no longer takes a single nvm_ctrl_t*.  The
 *     subsystem now serves multiple devices; DMA mappings are
 *     tracked per (region_id, Device*).  Coordinator is expected
 *     to wire each Device in via register_tensor() (whose
 *     bound devices drives mapping creation).
 *   - prepare_nvme_dma() / prepare_rdma_mr() are gone.  DMA
 *     mapping happens implicitly during register_tensor().
 *   - set_descriptor_format() / descriptor_format() implemented.
 *     PRP/SGL builder lives in v0.1 as a Unimplemented stub
 *     (descriptor_slice returns false); the real builder lands
 *     in R7 (PRP) / R8 (SGL).
 *
 * Coverage in v0.1
 *   - allocate_host(HOST | PINNED_HOST)        => malloc / cudaMallocHost
 *   - allocate_device(DEVICE)                  => cudaMalloc on a target GPU
 *   - allocate_device(MANAGED)                 => cudaMallocManaged
 *   - free                                     => matching free / cudaFree*
 *   - register_host                            => no-copy view, MR id assigned
 *   - register_device                          => same, on a GPU buffer
 *   - register_external(APP_MANAGED)           => same, source recorded
 *   - register_external(CUDA_IPC | HOST_SHM | HOST_FD_MAP) => unimplemented;
 *                                                returns nullptr.
 *   - register_tensor                          => idempotent; finds (or
 *                                                creates) the region for
 *                                                spec.ptr and DMA-maps it
 *                                                to each target device
 *                                                via libnvm.
 *   - descriptor_slice                         => Unimplemented stub
 *                                                (returns false until R7).
 *   - lookup                                   => walks an internal table.
 *
 * Threading
 *   - All public methods are protected by a single std::mutex; v0.1
 *     is deliberately simple.  When tracing shows real contention
 *     we'll swap in a striped table.
 *
 * Lifetime
 *   - Caller-allocated regions: subsystem records metadata only.
 *     unregister() releases metadata + any nvm_dma_t handles
 *     created on its behalf.  The buffer stays the caller's.
 *   - Subsystem-allocated regions: free() releases metadata,
 *     nvm_dma_t handles, AND the underlying buffer.
 */

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "memory_kind.h"
#include "memory_region.h"
#include "memory_subsystem.h"

#include <nvm_types.h>   // nvm_ctrl_t, nvm_dma_t

namespace tutti {

struct Device;   // runtime/include/device.h -- forward-decl mirrored
                 // here so this header doesn't need to pull in the
                 // full Device definition.

class HostDeviceMemorySubsystem : public IMemorySubsystem {
public:
    HostDeviceMemorySubsystem();
    ~HostDeviceMemorySubsystem() override;

    HostDeviceMemorySubsystem(const HostDeviceMemorySubsystem&)            = delete;
    HostDeviceMemorySubsystem& operator=(const HostDeviceMemorySubsystem&) = delete;

    // --- IMemorySubsystem -----------------------------------------

    void              set_descriptor_format(DescriptorFormat fmt) override;
    DescriptorFormat  descriptor_format() const override;

    void              bind_devices(
        const std::vector<const Device*>& devices) override;

    MemoryRegion* allocate_host(std::size_t size, MemoryKind kind) override;
    MemoryRegion* allocate_device(std::size_t size,
                                   MemoryKind  kind,
                                   int         device_id) override;
    void          free(MemoryRegion* region) override;

    MemoryRegion* register_host(void* host_ptr, std::size_t size) override;
    MemoryRegion* register_device(void*       device_ptr,
                                   std::size_t size,
                                   int         device_id) override;
    MemoryRegion* register_external(void*                     host_ptr,
                                     void*                     device_ptr,
                                     std::size_t               size,
                                     const ExternalMemorySpec& spec) override;
    void          unregister(MemoryRegion* region) override;

    MemoryRegion* register_tensor(const TensorRegistrationSpec& spec) override;

    const IoSliceView*
                  lookup_io_slice(MemoryRegion* region,
                                  uint64_t      slice_addr) const override;

    std::vector<IoSliceView>
                  list_io_slices(MemoryRegion* region) const override;

    bool descriptor_slice(MemoryRegion*       region,
                          const Device*       device,
                          uint64_t            byte_offset,
                          uint64_t            byte_length,
                          AddressDescriptor*  out,
                          std::size_t*        inout_count) override;

    MemoryRegion* lookup(const MemoryLookupKey& key) const override;

    // --- Test-only knobs ------------------------------------------

    /// Number of regions currently tracked.
    std::size_t region_count() const;

    /// Test-friendly query: does `region` have a DMA mapping for
    /// `device`?  Returns the per-page IO address count via
    /// out_count, the page size via out_page_size, and the first
    /// IO address via out_first_ioaddr.  Returns false if no
    /// mapping exists.  Smokes use this to validate that
    /// register_tensor wired DMA mapping correctly without leaking
    /// nvm_dma_t* details.
    bool query_nvme_mapping(const MemoryRegion* region,
                            const Device*       device,
                            std::size_t*        out_count,
                            std::size_t*        out_page_size,
                            uint64_t*           out_first_ioaddr) const;

private:
    // R7: cluster-wide IO-slice table.  views[] is the host-resident
    // lookup index in slice_addr order; d_all_descriptors is the
    // GPU-resident contiguous AddressDescriptor[] block that every
    // view's d_ios pointer points into.  ONE table per region.
    //
    // Deployment contract: IOMMU=pt + bare metal (no vIOMMU,
    // no virtualization).  PCI bus addresses are therefore
    // controller-agnostic, and snvme.ko's NVM_MAP_DEVICE_MEMORY
    // ioctl already loops over every open ctrl on a single call
    // (see snvme/map.c map_gpu_memory() line 506-532).  We exploit
    // both:
    //   - ONE PRP buffer + ONE descriptor blob serve every
    //     cluster-bound NVMe ctrl;
    //   - we call nvm_dma_map_data_device EXACTLY ONCE per buffer
    //     (any ctrl works as the proxy; the kernel handles the
    //     rest internally; the resulting nvm_dma_t.ioaddrs[] is
    //     usable from every cluster-bound ctrl).
    //
    // PRP-list buffer (prp_list_devptr) is allocated INTERNALLY by
    // build_io_slice_table_locked() when an IO needs more than 2
    // pages.  Sized to total_ios * page_size, cudaMalloc'd once,
    // and DMA-mapped through ONE controller (prp_list_dma).
    // Released by free_io_slice_table_locked().
    struct IoSliceTable {
        std::vector<IoSliceView>    views;
        AddressDescriptor*          d_all_descriptors = nullptr;
        std::size_t                 total_descriptors = 0;

        // PRP-list buffer (GPU memory, owned by this table).
        // nullptr when no IO needs a PRP list.
        void*                       prp_list_devptr = nullptr;
        std::size_t                 prp_list_bytes  = 0;
        // Single DMA handle for prp_list_devptr (see deployment
        // contract above).  nullptr when no IO needs a PRP list.
        nvm_dma_t*                  prp_list_dma    = nullptr;
    };

    struct Slot {
        std::unique_ptr<MemoryRegion> region;
        bool        owns_host_alloc   = false;
        bool        owns_device_alloc = false;

        // Single kernel-side DMA registration handle for the data
        // buffer.  Created lazily by register_tensor (one
        // nvm_dma_map_data_* call per region, NOT per controller --
        // see deployment contract on IoSliceTable above).  The
        // subsystem owns the nvm_dma_t and unmaps on erase.
        //
        // HISTORICAL REMARK: earlier R7 cuts kept an
        // unordered_map<Device*, nvm_dma_t*> here (one handle per
        // ctrl).  That was redundant on every level: snvme.ko's
        // NVM_MAP_DEVICE_MEMORY ioctl already loops over every
        // open ctrl in a single call, and the resulting PCI bus
        // addresses are identical across ctrls under IOMMU=pt.
        // Calling nvm_dma_map_data_device once per ctrl was just
        // N GPU-pin refcount bumps and N copies of an identical
        // ioaddrs[].  We now call it exactly once per buffer.
        // R10+ TODO covers the upstream libnvm/snvme ABI cleanup
        // that would deprecate the multi-call shape entirely.
        nvm_dma_t*                    data_dma = nullptr;

        // Cluster-wide IO-slice table.  Populated by register_tensor
        // when spec.granularity > 0; freed by erase_locked / re-built
        // by build_io_slice_table_locked.  Strictly idempotent: a
        // second register_tensor call returns the existing table
        // unchanged.
        IoSliceTable io_slice_table;
    };

    MemoryRegion* register_into_table(std::unique_ptr<MemoryRegion> r,
                                       bool owns_host_alloc,
                                       bool owns_device_alloc);
    void          erase_locked(uint64_t region_id);

    /// Find the slot that owns `ptr` (host or device pointer).
    /// Returns nullptr if not registered.  Caller must hold mtx_.
    Slot*         slot_by_ptr_locked(const void* ptr);

    /// DMA-map `slot.region` if not already mapped.  Uses
    /// bound_devices_.front() as the proxy ctrl (one ioctl
    /// internally covers every cluster-bound ctrl -- see
    /// deployment contract on Slot::data_dma).  Caller must hold
    /// mtx_.  Returns true on success or already-mapped.
    bool          ensure_mapping_locked(Slot& slot);

    /// Build (or rebuild) the cluster-wide IO-slice table for `slot`
    /// using spec.granularity.  Caller must hold mtx_ and have
    /// already ensured DMA mappings exist on every device in
    /// bound devices for `slot`.  effective_io is sized to
    /// the weakest controller (min over bound devices of
    /// caps.max_io_bytes); ioaddrs are taken from any one
    /// controller (bound_devices_.front()) since they are
    /// controller-agnostic in the host-NVMe direct path.  The
    /// PRP-list buffer (when needed) is allocated INTERNALLY
    /// here -- callers do not provide it.  Returns false on
    /// validation failure or CUDA error.  On success,
    /// slot.io_slice_table is populated and any pre-existing
    /// entry has been freed.
    bool          build_io_slice_table_locked(Slot&                          slot,
                                               const TensorRegistrationSpec&  spec);

    /// Free a single IoSliceTable's GPU resources (descriptor
    /// blob + PRP-list buffer + per-ctrl DMA handles for the PRP
    /// buffer).  Caller must hold mtx_.  Idempotent.
    void          free_io_slice_table_locked(IoSliceTable& tab);

    DescriptorFormat                       fmt_ = DescriptorFormat::UNSET;
    mutable std::mutex                     mtx_;
    uint64_t                               next_region_id_ = 1;
    std::unordered_map<uint64_t, Slot>     regions_;

    // Cluster-wide device set + cached caps, populated by
    // bind_devices() and consumed by register_tensor() /
    // build_io_slice_table_locked().  Empty until first bind.
    std::vector<const Device*>             bound_devices_;
    std::size_t                            cluster_page_size_ = 0;
    std::size_t                            cluster_min_mdts_  = 0;
};

} // namespace tutti

#endif // __TUTTI_MEMORY_HOST_DEVICE_MEMORY_SUBSYSTEM_H__

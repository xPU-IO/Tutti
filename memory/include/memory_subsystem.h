#ifndef __TUTTI_MEMORY_MEMORY_SUBSYSTEM_H__
#define __TUTTI_MEMORY_MEMORY_SUBSYSTEM_H__

/**
 * memory_subsystem.h -- the IMemorySubsystem interface.
 *
 * Layer: Memory.  See doc/refactor/LegacyDecomposition.md §2 / §3.5
 * for the full layer cake.  This layer is the single source of truth
 * for "the runtime knows about this piece of memory" and for
 * translating tensors into NVMe-readable address descriptors
 * (PRP today, SGL when supported cluster-wide).
 *
 * Hard invariants (mirror of LegacyDecomposition.md §2.2):
 *   - The descriptor format (PRP vs SGL) is **cluster-wide**, set
 *     once at coordinator boot via set_descriptor_format().  If any
 *     attached controller lacks SGL support, the entire runtime
 *     falls back to PRP.  Memory therefore stores at most one
 *     descriptor format per registration, never a mix.
 *   - Memory is per-process, never per-controller.  A single
 *     IMemorySubsystem instance services all attached devices; DMA
 *     mappings are tracked internally as a (region, device) table.
 *   - Memory does not own queue pairs and does not submit IO.
 *     Producing a PRP/SGL list is a memory job; submitting it is
 *     nvme_storage's.
 *
 * Allocation vs registration:
 *   - allocate_X (X = host / device): convenience for callers that
 *     don't already own a buffer.  The runtime allocates with the
 *     appropriate primitive, records a MemoryRegion, returns it.
 *     free() releases both the registration and the underlying
 *     allocation.
 *   - register_X (X = host / device / external): the caller already
 *     has a buffer.  The runtime records metadata; the underlying
 *     buffer stays the caller's responsibility.
 *   - register_tensor: the high-level entry-point used by upper
 *     layers (block_storage, io_engine).  Implicitly DMA-maps the
 *     buffer to every device bound via bind_devices() at coordinator
 *     bring-up.  May be called more than once on the same buffer
 *     (idempotent on (region, device) pairs).
 *
 * Threading:
 *   - Implementations MUST be thread-safe.  Multiple data-plane
 *     threads concurrently look up MemoryRegions during IO submission.
 *
 * Lifetime:
 *   - The MemoryRegion pointer returned by allocate_X / register_X
 *     is valid until the matching free() / unregister().
 *   - After free() / unregister(), the pointer is invalid.  Callers
 *     MUST drop their copies.
 *
 * Out of scope for v0.1:
 *   - No reference counting on regions.
 *   - No region splitting / sub-allocation.
 *   - register_external CUDA_IPC / HOST_SHM / HOST_FD_MAP variants
 *     are stubbed (return nullptr).
 *   - descriptor_slice() ships an Unimplemented stub; the real PRP
 *     builder lands in R7, the SGL builder in R8.
 */

#include <cstdint>
#include <cstddef>
#include <vector>

#include "memory_kind.h"
#include "memory_region.h"

namespace tutti {

struct Device;   // runtime/include/device.h

// ---------------------------------------------------------------------------
// Cluster-wide descriptor format.
//
// Coordinator probes every controller it ever attaches and ANDs the
// capability bits.  If every ctrl supports SGL, the runtime can use
// SGL; otherwise it falls back to PRP for the whole run.
// ---------------------------------------------------------------------------
enum class DescriptorFormat : uint8_t {
    UNSET = 0,   // before set_descriptor_format() is called
    PRP   = 1,
    SGL   = 2,
};

// ---------------------------------------------------------------------------
// One row of an NVMe-readable description of part of a buffer.
//
// PRP path uses prp1 (and optional prp2 for cross-page IO); SGL path
// reinterprets the same struct.  The slice walker emits as many of
// these as needed to cover the requested byte range.
// ---------------------------------------------------------------------------
struct AddressDescriptor {
    uint64_t prp1;          // valid iff format == PRP
    uint64_t prp2;          // valid iff format == PRP, second page in pair
    uint64_t data_length;   // bytes covered by this descriptor row
    // SGL fields collapsed for now; expanded in R7.
};

// ---------------------------------------------------------------------------
// Caller intent at register_tensor() time.
//
// `granularity` is the only knob normal callers touch: it's the
// business-level transfer unit (the size of one logical slice the
// app reads/writes atomically).  Set 0 to skip IO-slice table
// pre-computation (R3 path: just DMA-map and return).
//
// Cluster device set is NOT in this spec -- it's a cluster-wide
// property bound once via IMemorySubsystem::bind_devices() at
// coordinator bring-up.  register_tensor() implicitly DMA-maps the
// buffer to every bound device.
//
// R7 NEW (granularity):
//   `granularity` (also called the "slice size") is the business-
//   level transfer unit: the size of one logical slice the
//   application reads/writes atomically.  For vLLM PageAttention
//   this matches the per-layer K (or V) tensor size for one page.
//   At register_tensor() time memory/ pre-computes one
//   AddressDescriptor[] block per slice and uploads it to GPU
//   memory; the IO engine then looks slices up in O(log N) via
//   `lookup_io_slice(slice_addr)` instead of recomputing PRPs on
//   the submit hot path.
//
//   Cluster-wide (NOT per-device):
//     The IO-slice table is a *single* table per registered tensor.
//     For the host-NVMe direct path, a GPU physical page has the
//     same PCI bus address in every NVMe controller's view (the
//     IOMMU passes through identity), so one PRP/SGL list serves
//     every controller bound via bind_devices().  We therefore
//     size the second-level slicing to the weakest controller and
//     store one AddressDescriptor[] blob per region.
//
//   Two-level slicing semantics:
//     L1 (business, set by caller) : granularity                (= slice size)
//     L2 (cluster, derived from caps cached in bind_devices()) :
//          min_mdts = min over bound_devices of caps.max_io_bytes
//     effective_io = (granularity == 0) ? min_mdts
//                                       : min(granularity, min_mdts)
//     ios_per_slice = (granularity == 0) ? 1
//                                        : ceil(granularity / effective_io)
//
//   Alignment requirements (NVMe block-size constraints; checked at
//   register_tensor and reject -> nullptr if violated):
//     - spec.ptr  must be page_size-aligned (4 KiB on every drive
//       we ship today; queried from cluster caps cached at
//       bind_devices() time)
//     - spec.size must be a multiple of page_size
//     - spec.granularity == 0  OR  granularity % page_size == 0
//     - spec.size must be a multiple of granularity (every slice
//       fully populated)
//
//   PRP-list buffer:
//     When an IO needs more than 2 pages (effective_io > 2 *
//     page_size), it requires a PRP-list page.  memory/
//     allocates this buffer **internally** at register_tensor
//     time (cudaMalloc + nvm_dma_map_data_device for every
//     bound device) and stores it inside the IO-slice table;
//     it is freed when the region is unregistered.  Callers do
//     NOT supply the buffer (this matches legacy
//     `GPUController::initializePRPList()` behaviour).
// ---------------------------------------------------------------------------
struct TensorRegistrationSpec {
    void*                       ptr;           // host or device pointer
    std::size_t                 size;          // bytes
    std::vector<std::size_t>    shape;         // optional tensor shape

    // R7 NEW (defaulted so v0.1 callers stay source-compatible).
    std::size_t                 granularity      = 0;   // a.k.a. slice size
};

// ---------------------------------------------------------------------------
// One pre-computed IO slice exported to upper layers (R7).
//
// register_tensor() splits a tensor into N slices of `granularity`
// bytes each (or one slice covering the whole tensor when
// granularity == 0).  Each slice in turn fans out into one or more
// NVMe IOs, depending on how the slice size compares to the
// cluster-wide effective IO (= min over bound devices of MDTS).
// IoSliceView exposes both the host-visible address that keys the
// slice and the GPU-resident AddressDescriptor[] block describing
// the IOs the slice will issue.  ONE table serves every controller
// across bound devices (PCI bus addresses are controller-agnostic for
// the host-NVMe direct path).
//
// Lifetime: valid until unregister(region) or free(region).
//
// Hot-path usage (R8 io_engine):
//   const IoSliceView* v = mem.lookup_io_slice(region, addr);
//   for (uint32_t i = 0; i < v->num_ios; ++i) {
//       submit_read_one(dh,
//                       v->d_ios[i].prp1,
//                       v->d_ios[i].prp2,
//                       /*off=*/(addr - region_base) + i*effective_io,
//                       v->d_ios[i].data_length);
//   }
// ---------------------------------------------------------------------------
struct IoSliceView {
    uint64_t                    slice_addr;    // = tensor.ptr + j * granularity
                                                //   (or tensor.ptr when
                                                //    granularity == 0)
    uint32_t                    num_ios;       // = 1 when granularity <= effective_io
                                                //   = ceil(granularity / effective_io)
                                                //     otherwise
    uint64_t                    total_bytes;   // bytes covered by this slice
                                                //   (== granularity, except
                                                //    last slice may be smaller
                                                //    -- though R7 currently
                                                //    requires size %
                                                //    granularity == 0)
    const AddressDescriptor*    d_ios;         // GPU-resident pointer to
                                                //   AddressDescriptor[num_ios];
                                                //   contiguous, never null
                                                //   for a valid slice.
};

// ---------------------------------------------------------------------------
// Look-up key for `lookup()`.
// ---------------------------------------------------------------------------
struct MemoryLookupKey {
    enum class By : uint32_t { HOST_PTR, DEVICE_PTR, REGION_ID };
    By       by;
    union {
        const void* ptr;        // for HOST_PTR / DEVICE_PTR
        uint64_t    region_id;  // for REGION_ID
    };
};

class IMemorySubsystem {
public:
    virtual ~IMemorySubsystem() = default;

    // ------------------------------------------------------------------
    // Cluster-wide capability handshake (called by coordinator at boot)
    // ------------------------------------------------------------------

    /// Freeze the descriptor format.  Coordinator probes every ctrl
    /// it attaches and calls this once with the AND of their caps.
    /// Any later register_tensor / descriptor_slice produces this
    /// format.  Calling twice with different values is a logic
    /// error in v0.1 (implementations may assert).
    virtual void set_descriptor_format(DescriptorFormat fmt) = 0;

    /// Inspect the currently-frozen format.
    virtual DescriptorFormat descriptor_format() const = 0;

    /// Bind the cluster of devices this subsystem will DMA-map all
    /// subsequently-registered tensors to.  Called ONCE by the
    /// coordinator after every backend has surfaced its devices.
    /// Caches the cluster-wide caps used by R7 IO slicing:
    ///   page_size  = devices.front().capabilities.page_size
    ///   min_mdts   = min over devices of capabilities.max_io_bytes
    /// Replaces any previous binding.  Calling with an empty list
    /// is a logic error in v0.1 (R7 paths require min_mdts > 0).
    ///
    /// Lifetime invariant: bind_devices() must be called BEFORE
    /// any register_tensor() with granularity > 0.  Re-binding
    /// with a different device set after granularity > 0 regions
    /// have been registered is undefined behaviour in v0.1
    /// (the cached caps would no longer match the IO-slice tables);
    /// future revisions may rebuild on rebind, see the R10+
    /// libnvm DMA ABI redesign in Todolist.md.
    virtual void bind_devices(const std::vector<const Device*>& devices) = 0;

    // ------------------------------------------------------------------
    // Allocation (runtime-driven)
    // ------------------------------------------------------------------

    /// Allocate `size` bytes of host memory of `kind` (HOST or PINNED_HOST).
    virtual MemoryRegion* allocate_host(std::size_t size, MemoryKind kind) = 0;

    /// Allocate `size` bytes of CUDA device memory on `device_id`
    /// (DEVICE) or CUDA managed memory (MANAGED).
    virtual MemoryRegion* allocate_device(std::size_t size,
                                           MemoryKind  kind,
                                           int         device_id) = 0;

    /// Release a region the runtime allocated.
    virtual void free(MemoryRegion* region) = 0;

    // ------------------------------------------------------------------
    // Low-level registration (caller-allocated, no DMA mapping)
    //
    // These are the thin entry-points used internally by
    // register_tensor() and by tests; upper layers should normally
    // call register_tensor() (which auto-DMA-maps to every
    // bind_devices()'d controller).
    // ------------------------------------------------------------------

    virtual MemoryRegion* register_host(void*       host_ptr,
                                         std::size_t size) = 0;
    virtual MemoryRegion* register_device(void*       device_ptr,
                                           std::size_t size,
                                           int         device_id) = 0;

    /// External-source (vLLM slab, CUDA IPC import, ...).
    virtual MemoryRegion* register_external(void*                       host_ptr,
                                             void*                       device_ptr,
                                             std::size_t                 size,
                                             const ExternalMemorySpec&   spec) = 0;

    /// Drop the runtime's metadata for `region`.
    virtual void unregister(MemoryRegion* region) = 0;

    // ------------------------------------------------------------------
    // High-level registration
    // ------------------------------------------------------------------

    /// Register a buffer that will be DMA'd to one or more NVMe
    /// devices.  Side effect: every device in bound devices
    /// gets a DMA mapping created for this buffer (if it doesn't
    /// already have one).  Calling repeatedly with the same ptr
    /// is fine; existing mappings are reused.
    ///
    /// Returns a stable MemoryRegion* tied to spec.ptr.  If spec.ptr
    /// was not previously registered, an internal register_host /
    /// register_device is performed first, classifying by whether
    /// the pointer maps to host or device memory.
    virtual MemoryRegion* register_tensor(const TensorRegistrationSpec& spec) = 0;

    // ------------------------------------------------------------------
    // Address descriptor extraction (consumed by io_engine kernel +
    // nvme_storage host-side IO).
    // ------------------------------------------------------------------

    /// Look up the pre-computed IoSliceView for a registered tensor
    /// (R7).  `slice_addr` MUST equal
    ///   tensor.ptr + j * granularity      for some j in [0, num_slices)
    /// when register_tensor was called with granularity > 0.  When
    /// granularity == 0 there is exactly one slice covering the
    /// whole tensor and `slice_addr == tensor.ptr` is the only
    /// valid key.
    ///
    /// The IO-slice table is cluster-wide: ONE table serves every
    /// controller the tensor was register_tensor'd on, since GPU
    /// physical pages have controller-agnostic PCI bus addresses.
    ///
    /// Returns nullptr if the region was never register_tensor'd
    /// with granularity > 0, the address is not a slice boundary,
    /// or the region is unknown.
    ///
    /// Complexity: O(log num_slices) (binary search over a
    /// slice_addr-sorted vector).  Thread-safe.  The returned
    /// pointer stays valid until unregister(region) or free(region).
    virtual const IoSliceView*
        lookup_io_slice(MemoryRegion*  region,
                        uint64_t       slice_addr) const = 0;

    /// Bulk export every (slice_addr -> IoSliceView) for a region
    /// in slice_addr order (R7).  Empty vector if the region has no
    /// IO-slice table.  Used by R8 io_engine for batched host-side
    /// IO context staging where the caller wants to walk every
    /// slice once.
    virtual std::vector<IoSliceView>
        list_io_slices(MemoryRegion* region) const = 0;

    /// Walk `region`'s DMA mapping for `device` and emit address
    /// descriptors covering byte_range [byte_offset, byte_offset + byte_length).
    ///
    /// On entry *inout_count tells the maximum number of descriptors
    /// the caller can accept; on success *inout_count is set to the
    /// number actually emitted.  Returns false if the format hasn't
    /// been set, the region has no mapping for `device`, or the
    /// caller's buffer is too small.
    ///
    /// v0.1 status: this entry-point remains a stub for the "ad-hoc
    /// slice without prior granularity registration" path.  The
    /// granularity-aware fast path is via lookup_io_slice /
    /// list_io_slices above; descriptor_slice() exists to leave
    /// room for a future "compute on demand" caller (e.g. write
    /// paths whose extent doesn't line up with a registered slice).
    virtual bool descriptor_slice(MemoryRegion*       region,
                                   const Device*       device,
                                   uint64_t            byte_offset,
                                   uint64_t            byte_length,
                                   AddressDescriptor*  out,
                                   std::size_t*        inout_count) = 0;

    // ------------------------------------------------------------------
    // Query
    // ------------------------------------------------------------------

    /// Find the MemoryRegion that contains `key`.
    virtual MemoryRegion* lookup(const MemoryLookupKey& key) const = 0;
};

} // namespace tutti

#endif // __TUTTI_MEMORY_MEMORY_SUBSYSTEM_H__

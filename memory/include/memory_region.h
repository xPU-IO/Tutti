#ifndef __TUTTI_MEMORY_MEMORY_REGION_H__
#define __TUTTI_MEMORY_MEMORY_REGION_H__

/**
 * memory_region.h -- the runtime-internal handle for a piece of memory.
 *
 * Layer: Memory Layer (independent, parallel to Device Manager per Roadmap §3).
 *
 * Role:
 *   - The application allocates memory by whatever means it likes
 *     (malloc, cudaMalloc, mmap shared memory, vLLM pool slice, CUDA
 *     IPC import, ...). It then registers that memory with the
 *     IMemorySubsystem (see memory_subsystem.h), which returns a
 *     MemoryRegion handle.
 *   - The MemoryRegion is what every other layer (IO Engine, Device
 *     Manager, Backend SPI, COOP channels) refers to when it needs
 *     memory: instead of carrying raw pointers + DMA addresses around,
 *     they carry MemoryRegion* and let the Memory Layer resolve the
 *     details.
 *   - This decoupling means the runtime can grow new memory sources
 *     (RDMA-registered, NVLink-shared, multi-process IPC, ...) without
 *     forcing every layer above to learn about them.
 *
 * Design rules:
 *   - Memory ownership stays with the APPLICATION. MemoryRegion never
 *     frees the underlying buffer on its own. unregister() drops
 *     runtime-side metadata but leaves the buffer untouched.
 *   - A MemoryRegion is immutable after registration. Re-mapping or
 *     resizing requires unregister + register again.
 *   - One MemoryRegion may carry multiple "views" of the same memory:
 *     a host pointer (for CPU code), a device pointer (for GPU
 *     kernels), and optional registration metadata (DMA bus addresses
 *     for NVMe DMA, IPC handles for cross-process sharing). Backends
 *     pick the view they need.
 *   - The struct itself is forward-declared in headers that only need
 *     to point at it (e.g. io_engine batch types). Code that actually
 *     reads the fields includes this header.
 *
 * Versioning:
 *   - The struct may grow. Adding fields at the end is non-breaking.
 *     Reordering or removing fields requires a coordinated bump.
 */

#include <cstdint>
#include <cstddef>

#include "memory_kind.h"

namespace tutti {

/**
 * Source descriptor for EXTERNAL memory. The application populates
 * exactly one of the unioned variants when calling
 * IMemorySubsystem::register_external() to tell the runtime where the
 * memory really lives.
 *
 * The runtime does NOT reach into the application's allocator; it
 * only records enough metadata to share the memory with backends and
 * (where applicable) other processes.
 */
enum class ExternalMemorySource : uint32_t {
    APP_MANAGED = 0,   // pointer into a slab the app allocated and registered
                       // (e.g. vLLM's pre-allocated tensor pool). Host or device.
    CUDA_IPC    = 1,   // pointer obtained via cudaIpcOpenMemHandle in this process.
                       // The runtime tracks the originating process / IPC handle
                       // so it can refresh on lease loss.
    HOST_SHM    = 2,   // host memory mapped from a POSIX shm segment (shm_open + mmap).
                       // Useful for cooperative submit between processes on the same host.
    HOST_FD_MAP = 3,   // host memory mmaped from a file descriptor (e.g. memfd, hugepage fd).
};

/**
 * IPC import metadata when source == CUDA_IPC. Filled in by the caller
 * before register_external(); the runtime stores a copy.
 *
 * Why we don't include cudaIpcMemHandle_t directly: keep this header
 * free of <cuda_runtime.h>. The runtime stores the 64 raw bytes; the
 * Memory Layer's .cu translation unit re-interprets them when it
 * actually calls cudaIpcOpenMemHandle.
 */
struct CudaIpcImport {
    unsigned char raw_handle[64];   // bit-for-bit cudaIpcMemHandle_t bytes
    int           cuda_device;       // CUDA device the originating allocation belongs to
};

/**
 * shm or fd-mapped host memory.
 */
struct HostFdMapping {
    int      fd;          // file descriptor backing the mapping (caller still owns it)
    uint64_t fd_offset;   // offset into the fd
    uint32_t mmap_prot;   // PROT_READ / PROT_WRITE / PROT_READ|PROT_WRITE
    uint32_t mmap_flags;  // MAP_SHARED / MAP_PRIVATE / MAP_HUGETLB / ...
};

/**
 * Caller-supplied ExternalMemorySource detail. Exactly one variant
 * applies, selected by `source`. Layered tagged union to keep the
 * shape stable across future sources.
 */
struct ExternalMemorySpec {
    ExternalMemorySource source;
    union {
        // APP_MANAGED has no extra metadata -- the (host_ptr, device_ptr,
        // size, kind) tuple already in MemoryRegion is enough.
        CudaIpcImport ipc;
        HostFdMapping fd_map;
    };
};

/**
 * Backend-facing registration metadata. Backends that need DMA-mapped
 * IO addresses (NVMe), RDMA keys (verbs), or process-shareable handles
 * read these fields. Filled in by the Memory Layer at registration
 * time; nullptr/zero means "this backend hasn't asked for it yet".
 *
 * The Memory Layer is the canonical owner of these arrays. Backends
 * consult them through a const view; they do not free or resize.
 */
struct RegistrationMetadata {
    // NVMe / GDS: per-page DMA bus addresses, one entry per `page_size`.
    // The Memory Layer obtained these by walking libnvm's DMA mapping or
    // GDS's cuFileBufRegister equivalent.
    const uint64_t* dma_ioaddrs;
    std::size_t     dma_ioaddr_count;
    std::size_t     page_size;          // bytes per dma_ioaddrs entry

    // RDMA: opaque rkey/lkey blob. Defined by the RDMA backend; other
    // backends ignore. Stays as raw bytes here so this header doesn't
    // pull in libibverbs.
    const void* rdma_keys;
    std::size_t rdma_keys_size;
};

/**
 * The runtime-internal handle. Every other layer that wants to refer
 * to a piece of registered memory carries a `const MemoryRegion*`.
 *
 * Either pointer can be nullptr if that view is not available:
 *   - HOST kind             : host_ptr set, device_ptr = nullptr.
 *   - DEVICE kind           : host_ptr = nullptr, device_ptr set,
 *                              cuda_device >= 0.
 *   - PINNED_HOST + GPU map : both set; device_ptr is the
 *                              cudaHostGetDevicePointer view.
 *   - MANAGED               : both set, both equal (unified VA).
 *   - EXTERNAL              : populated per the ExternalMemorySpec
 *                              recorded at registration time.
 */
struct MemoryRegion {
    // ---- Identity --------------------------------------------------
    uint64_t   region_id;       // unique within the IMemorySubsystem instance
    MemoryKind kind;
    int        cuda_device;     // -1 if not GPU-bound; otherwise the CUDA device index

    // ---- Views ------------------------------------------------------
    void*    host_ptr;          // CPU-visible base, or nullptr
    void*    device_ptr;        // GPU-visible base, or nullptr
    uint64_t size;              // bytes; identical for host and device views

    // ---- External-source metadata (populated only when kind == EXTERNAL) ----
    ExternalMemorySpec external;

    // ---- Backend registration metadata ----------------------------
    // Filled in after the Memory Layer has prepared backend-specific
    // mappings (NVMe DMA, RDMA keys, ...). May be partially populated
    // depending on which backends have asked for registration.
    RegistrationMetadata registration;
};

} // namespace tutti

#endif // __TUTTI_MEMORY_MEMORY_REGION_H__

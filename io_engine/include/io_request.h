#ifndef __TUTTI_IO_ENGINE_IO_REQUEST_H__
#define __TUTTI_IO_ENGINE_IO_REQUEST_H__

/**
 * io_request.h -- per-IO request, batched view, and completion record.
 *
 * Layer: io_engine (Backend SPI). Builds on memory/include/memory_region.h.
 *
 * Spec: doc/design/backend-spi.md §3.4 + Roadmap.md §3 (object model).
 *
 * Three types live here because they form one conceptual triple:
 *   - IORequest      : one IO op (offset, size, direction, descriptor pointer).
 *   - IORequestBatch : a contiguous array of IORequest tied to a MemoryRegion.
 *   - IOCompletion   : the per-request result record.
 *
 * Memory residency policy:
 *   IORequest is a plain POD. The same layout is valid in host memory,
 *   in CUDA device memory, in pinned-shared memory, or copied between
 *   them. The struct itself carries NO residency tag -- residency is a
 *   property of the MemoryRegion that backs the array (see
 *   memory/include/memory_region.h).
 *
 *   The SPI's submission methods establish the residency contract per
 *   call site. Examples:
 *     - launch_batch_gpu_stream() requires the array's MemoryRegion to
 *       expose a device_ptr accessible from `stream`'s CUDA context.
 *     - submit_batch_cpu_sync() requires the array's MemoryRegion to
 *       expose a host_ptr dereferenceable from the calling thread.
 *     - COOP channels live in MANAGED or PINNED_HOST regions readable
 *       by both sides.
 *
 * Memory ownership policy:
 *   Memory comes from the application. The Memory Layer wraps it in a
 *   MemoryRegion at registration time; nothing in io_engine allocates.
 *   IORequestBatch carries a `const MemoryRegion*` so backends can
 *   resolve registration metadata (DMA addresses, IPC handles, ...)
 *   without re-walking page tables.
 *
 * Address-space semantics for `file_offset`:
 *   - Local NVMe: bytes from the start of the target file.
 *   - RDMA: may carry a remote virtual address; the backend documents
 *     its own interpretation (see RDMABufferDesc).
 *
 * Lifetime:
 *   - The pointed-to BufferDescriptor and the request array MUST stay
 *     valid until the matching IO completes. Lifetime ownership lies
 *     with the application's MemoryRegion; the runtime keeps the
 *     handle alive across the kernel / completion poll but does not
 *     retain pointers beyond.
 */

#include <cstdint>

namespace tutti {

struct BufferDescriptor;  // backend-neutral; see buffer_descriptor.h
struct MemoryRegion;      // memory/include/memory_region.h

/**
 * One IO operation. Backend-agnostic: no PRP/SGL/RDMA bytes here. The
 * addressing payload lives in `descriptor`, which is a pointer into a
 * BufferDescriptor pool (typically the BufferDescriptorBatch paired
 * with this request's batch).
 *
 * `descriptor` MUST point to memory in the same residency class as the
 * request itself (host->host, device->device); otherwise the consuming
 * kernel/CPU thread will fault when dereferencing.
 */
struct IORequest {
    uint64_t file_offset;                 // bytes (or remote VA, backend-defined)
    uint32_t size;                        // transfer size in bytes
    bool     is_read;                     // direction
    uint32_t reserved;                    // pad / future flags
    const BufferDescriptor* descriptor;   // points to a matching BufferDescriptor
};

/**
 * Contiguous batch of IORequest. Carries a MemoryRegion handle that
 * names the app-allocated, runtime-registered region the array lives
 * in. The pointer's residency is established by that region (host /
 * device / pinned / managed / external).
 *
 * Pairs with BufferDescriptorBatch in buffer_descriptor.h. The two
 * batches MAY share one MemoryRegion (the app packed both into one
 * registered slab) or sit in two different regions; backends that
 * care MUST handle both shapes.
 *
 * The `requests` pointer must lie inside `region`'s host_ptr or
 * device_ptr extent; the Memory Layer may validate this at SPI entry.
 */
struct IORequestBatch {
    const IORequest*    requests;
    uint32_t            count;
    const MemoryRegion* region;   // app-registered home of `requests`
};

/**
 * Per-request completion record. Filled by the backend when a request
 * finishes. `request_index` matches the position the request held in
 * the submitted IORequestBatch (0..count-1).
 *
 * Same residency policy as IORequest: this struct is residency-neutral;
 * the buffer it's written into lives wherever the caller's contract
 * says (host poll buffer, device CQ ring, etc.).
 */
struct IOCompletion {
    uint32_t request_index;
    uint32_t status;          // 0 = success; non-zero = backend-defined error code
    uint64_t bytes_done;
};

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_IO_REQUEST_H__

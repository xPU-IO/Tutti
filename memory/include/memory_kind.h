#ifndef __TUTTI_MEMORY_MEMORY_KIND_H__
#define __TUTTI_MEMORY_MEMORY_KIND_H__

/**
 * memory_kind.h -- enumeration of memory variants the runtime understands.
 *
 * Layer: Memory Layer (independent, parallel to Device Manager per Roadmap §3).
 *
 * The runtime does not allocate memory on its own. Applications allocate
 * (malloc / cudaMalloc / vLLM-managed slab / IPC / shm) and hand the
 * runtime a pointer + length, which the Memory Layer wraps in a
 * MemoryRegion. `MemoryKind` records HOW that memory was obtained so
 * the Memory Layer can pick the right registration / IPC / DMA-mapping
 * path for downstream backends.
 *
 * The first five values mirror the canonical CUDA/Linux residency
 * classes. EXTERNAL is the catch-all for "the application brought its
 * own; tell us through ExternalMemorySource how to treat it".
 *
 * Numeric values are stable across versions (consumers may persist
 * them). New variants append.
 */

#include <cstdint>

namespace tutti {

enum class MemoryKind : uint32_t {
    HOST         = 0,   // plain heap memory (malloc / new). CPU-only.
    PINNED_HOST  = 1,   // page-locked host memory (cudaHostAlloc / cudaMallocHost).
                         // Visible to GPU via cudaHostGetDevicePointer when
                         // the cudaHostRegisterMapped flag is set.
    DEVICE       = 2,   // CUDA device memory (cudaMalloc). One CUDA device only.
    MANAGED      = 3,   // CUDA managed memory (cudaMallocManaged). Visible to
                         // host and device via the unified address space.
    EXTERNAL     = 4,   // App-supplied memory whose original allocator is
                         // outside the runtime's control. The caller must
                         // also pass an ExternalMemorySource record (see
                         // memory_region.h) describing how to access it
                         // (CUDA IPC handle, posix shm fd, vLLM-pool offset, ...).
};

} // namespace tutti

#endif // __TUTTI_MEMORY_MEMORY_KIND_H__

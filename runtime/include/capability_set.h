#ifndef __TUTTI_RUNTIME_CAPABILITY_SET_H__
#define __TUTTI_RUNTIME_CAPABILITY_SET_H__

/**
 * capability_set.h -- runtime-level capability advertisement.
 *
 * Layer: Core Runtime (Roadmap.md §3 object model).
 *
 * Role:
 *   - Backends and Devices expose what they can do via a single
 *     CapabilitySet structure. Upper layers (Runtime, adapters) read
 *     this to decide which submission mode to use, whether to register
 *     a buffer for GPU access, etc.
 *   - The SPI's IBackendProvider has its own per-method advertisement
 *     (max_io_size, queue_depth, ...). CapabilitySet is the
 *     runtime-level rollup: a bitmask of feature toggles plus a few
 *     numeric ceilings.
 *
 * Bit-layout rules:
 *   - Bits 0..15 reserved for submission-mode capability flags.
 *   - Bits 16..23 reserved for memory-source capabilities.
 *   - Bits 24..31 reserved for transport/topology capabilities.
 *   - New bits append; do not reorder existing bits.
 */

#include <cstdint>
#include <cstddef>

namespace tutti {

/**
 * Capability bits. AND with CapabilitySet::flags to test support.
 *
 * Submission modes (bits 0..15) MIRROR IOSubmitMode in
 * io_engine/include/io_submit_mode.h. A device that advertises e.g.
 * CAP_SUBMIT_BATCH_GPU_STREAM is contractually required to accept
 * launch_batch_gpu_stream() through its backend.
 */
enum CapabilityBit : uint64_t {
    // Submission modes ----------------------------------------------------
    CAP_SUBMIT_BATCH_GPU_STREAM      = 1ull << 0,
    CAP_SUBMIT_BATCH_CPU_SYNC        = 1ull << 1,
    CAP_SUBMIT_BATCH_CPU_ASYNC       = 1ull << 2,
    CAP_SUBMIT_COOP_GPU_PRODUCE_CPU  = 1ull << 3,   // GPU writes SQ; CPU drains
    CAP_SUBMIT_COOP_CPU_PRODUCE_GPU  = 1ull << 4,   // CPU writes SQ; GPU drains
    CAP_SUBMIT_BATCH_GPU_ASYNC       = 1ull << 5,   // future: persistent kernel / CUDA Graphs

    // Memory sources (bits 16..23) ---------------------------------------
    CAP_MEM_HOST_REGISTER            = 1ull << 16,  // accepts plain host memory
    CAP_MEM_PINNED_HOST_REGISTER     = 1ull << 17,  // accepts cudaHostAlloc memory
    CAP_MEM_DEVICE_REGISTER          = 1ull << 18,  // accepts cudaMalloc memory
    CAP_MEM_MANAGED                  = 1ull << 19,  // accepts cudaMallocManaged memory
    CAP_MEM_EXTERNAL_CUDA_IPC        = 1ull << 20,  // accepts CUDA IPC imported memory
    CAP_MEM_EXTERNAL_HOST_SHM        = 1ull << 21,  // accepts POSIX shm-mapped memory
    CAP_MEM_EXTERNAL_APP_MANAGED     = 1ull << 22,  // accepts vLLM-style app slabs

    // Transport / topology (bits 24..31) ---------------------------------
    CAP_TOPO_GPUDIRECT_DMA           = 1ull << 24,  // peer-to-peer GPU<->NVMe DMA
    CAP_TOPO_GPUDIRECT_RDMA          = 1ull << 25,  // peer-to-peer GPU<->NIC DMA
    CAP_TOPO_NUMA_AWARE              = 1ull << 26,  // honors NUMA-affine queue placement
};

/**
 * Capability rollup for one Device or one IBackendProvider. Read-only
 * after the device is brought up.
 *
 * `flags` is the bitmask. The numeric fields are hard ceilings: an
 * IO larger than `max_io_bytes`, a batch wider than
 * `max_batch_count`, etc. is a programmer error and the SPI MAY
 * reject it.
 */
struct CapabilitySet {
    uint64_t    flags;              // bitmask of CapabilityBit
    std::size_t max_io_bytes;       // largest single-IO transfer
    std::size_t max_batch_count;    // largest IORequestBatch.count
    std::size_t max_queue_depth;    // largest QueueConfig.sq_depth / cq_depth
    std::size_t max_queue_count;    // upper bound on concurrent queues
    std::size_t page_size;          // backend's natural transfer unit (often 4KiB)
};

// Convenience: bit-and shortcut so call sites don't have to cast.
constexpr inline bool has_capability(const CapabilitySet& s, CapabilityBit bit) {
    return (s.flags & static_cast<uint64_t>(bit)) != 0;
}

} // namespace tutti

#endif // __TUTTI_RUNTIME_CAPABILITY_SET_H__

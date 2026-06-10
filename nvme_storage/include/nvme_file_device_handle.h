#ifndef __TUTTI_NVME_STORAGE_NVME_FILE_DEVICE_HANDLE_H__
#define __TUTTI_NVME_STORAGE_NVME_FILE_DEVICE_HANDLE_H__

/**
 * nvme_file_device_handle.h -- on-GPU file handle.
 *
 * Layer: nvme_storage (R5b).
 *
 * Lives entirely in GPU memory.  Produced by
 * `INvmeStorage::acquire_device_handle(NvmeFile*)` (which cudaMallocs one
 * and cudaMemcpys a host-side template into it), consumed verbatim
 * by `__device__ submit_read_one / submit_write_one`.  Never
 * mutated after construction; freed by
 * `INvmeStorage::release_device_handle`.
 *
 * Why this struct is POD with inline extents:
 *
 *   1. Kernels read it in a single load - no chasing pointers
 *      into separately-allocated extent arrays.
 *   2. cudaMemcpy is one shot.
 *   3. The host-side `NvmeFile::extents` already has a hard cap
 *      (`kNvmeFileHeaderMaxExtents`); the GPU handle just reuses
 *      that cap.
 *
 * What this struct deliberately does NOT contain:
 *
 *   - tensor / IOVA / PRP descriptors (memory/'s job; passed in
 *     per-call as prp1/prp2 args)
 *   - file_offset/length state (per-call args, not file-state)
 *   - back-pointer to the host-side NvmeFile (the GPU has no
 *     reason to dereference host pointers)
 */

#include <cstdint>

#include "lba_extent.h"
#include "nvme_file_header.h"   // kNvmeFileHeaderMaxExtents

// Forward declaration of libnvm's QueuePair.  We don't include
// queue.h here because queue.h drags in <cuda_runtime.h> and a chunk
// of host-only libnvm machinery; this header is meant to be
// includable from both .cpp and .cu callers.  The .cu submit
// helpers that actually dereference QueuePair (queue_acquire_helper.cuh,
// nvme_storage_device.cuh) include queue.h directly.
struct QueuePair;

namespace tutti {

struct NvmeFileDeviceHandle {
    /// PersistentFileLog::Entry::file_id of the source NvmeFile.
    /// Useful for debugging; kernels usually don't read this.
    uint64_t   file_id;

    /// Logical (user-visible) size in bytes.  Excludes the 4 KiB
    /// NvmeFileHeader stored at the start of the host file.  IO
    /// requests with logical_off + nbytes > logical_size_bytes
    /// MUST be rejected by the caller; submit_*_one assumes the
    /// args are in-range.
    uint64_t   logical_size_bytes;

    /// Bytes consumed by the NvmeFileHeader at the start of the
    /// host file.  `submit_*_one` adds this to the user-supplied
    /// `logical_off` before walking extents, so users see a clean
    /// "offset 0 == first byte of payload".
    uint32_t   header_bytes;

    /// NVMe block size for the namespace this file lives on
    /// (typ. 4096).  All `nbytes` and `logical_off` arguments
    /// MUST be multiples of this; submit_*_one converts both
    /// to LBA / block-count by right-shifting by
    /// `nvme_block_size_log`.
    uint32_t   nvme_block_size;
    uint32_t   nvme_block_size_log;   // log2(nvme_block_size)

    /// NVMe namespace id used in every SQE we synthesize.  Mirrors
    /// `qp->nvmNamespace` for the queue pairs in `d_qps` below.
    uint32_t   namespace_id;

    /// Inline copy of the NvmeFile's LBA extent table, exactly as
    /// captured at create_file time via FIEMAP.  Total length is
    /// `num_extents`; entries beyond that are unused.
    uint32_t   num_extents;
    LbaExtent  extents[kNvmeFileHeaderMaxExtents];

    /// Back-pointer to the controller's d_qps[] pool.  Lives on
    /// the GPU memory of the same cuda device as this handle.
    /// Lifetime is tied to the LocalNvmeDevice's queue_group: as long
    /// as that Controller is alive, d_qps is valid.  Kernels MUST
    /// NOT release / cudaFree this pointer.
    QueuePair* d_qps;
    uint32_t   num_d_qps;

    /// Reserved padding to keep the struct's overall size stable
    /// across compilers / minor field changes.  Currently unused.
    uint32_t   reserved0[3];
};

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_NVME_FILE_DEVICE_HANDLE_H__

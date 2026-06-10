#ifndef __TUTTI_NVME_STORAGE_NVME_STORAGE_DEVICE_CUH__
#define __TUTTI_NVME_STORAGE_NVME_STORAGE_DEVICE_CUH__

/**
 * nvme_storage_device.cuh -- the entire on-GPU NVMe submit surface.
 *
 * Layer: nvme_storage (R5b).
 *
 *   resolve_lba       walks NvmeFileDeviceHandle::extents to map
 *                     a (logical_off, nbytes) request onto an
 *                     (LBA, n_blocks) pair.
 *   submit_read_one   one-thread-per-IO read (host buffer -> NVMe
 *                     read) followed by busy-poll completion.
 *   submit_write_one  one-thread-per-IO write (host data -> NVMe).
 *
 * All three are __device__ __forceinline__ so they fold into the
 * caller kernel.  None of them take grid / block ids implicitly:
 * any thread is allowed to call them, and concurrency is mediated
 * by QueueAcquireHelper + libnvm's per-queue cid pool.
 *
 * Argument semantics:
 *
 *   logical_off  in user view (0 = first user byte).  R5b adds
 *                handle->header_bytes internally, so kernels can
 *                think of the file as starting at byte 0.
 *
 *   nbytes       MUST be a multiple of handle->nvme_block_size and
 *                MUST be > 0.  Sub-block partial IO is not in scope
 *                for R5b (that's a memory/-side concern about
 *                tensor padding).
 *
 *   prp1, prp2   already-resolved NVMe PRP entries.  prp2 may be 0
 *                when nbytes <= one NVMe page; otherwise it points
 *                at a PRP list.  R5b doesn't build PRPs -- the
 *                memory/ layer hands these in.
 *
 * Sub-IO contiguity:
 *
 *   resolve_lba REJECTS requests that span multiple extents.  In
 *   practice fallocate on a fresh ext4 yields one extent per file,
 *   so this is rarely a problem in R5b smoke; R7's batched IO will
 *   pre-split per-extent before calling submit_*_one.
 *
 *   On rejection submit_*_one prints a single line via printf
 *   from the kernel and returns without issuing IO.  The caller
 *   thread observes nothing else -- there is no error channel
 *   back to the host yet.  R7 introduces a per-IO completion
 *   record where errors will be stamped.
 */

#include <cstdint>
#include <cstdio>

#include "nvme_file_device_handle.h"
#include "queue_acquire_helper.cuh"

#include <queue.h>          // QueuePair (already pulled by helper)
#include <nvm_io.h>         // NVM_IO_READ, NVM_IO_WRITE

namespace tutti {

/// Map a (logical_off_bytes, nbytes) request inside this file
/// onto a (starting_lba, n_blocks) span on the underlying NVMe
/// namespace.  Walks `handle->extents` and adds `header_bytes` to
/// the logical offset before searching.
///
/// Returns true on success.  Returns false (and leaves the out
/// values untouched) if:
///   - nbytes == 0
///   - nbytes / logical_off not aligned to nvme_block_size
///   - the request spills past file end
///   - the request crosses an extent boundary
__device__ __forceinline__
bool resolve_lba(const NvmeFileDeviceHandle* h,
                 uint64_t                    logical_off,
                 uint64_t                    nbytes,
                 uint64_t*                   starting_lba_out,
                 uint64_t*                   n_blocks_out)
{
    if (h == nullptr || h->num_extents == 0) return false;
    if (nbytes == 0)                          return false;

    const uint32_t bs     = h->nvme_block_size;
    const uint32_t bs_log = h->nvme_block_size_log;

    if ((logical_off & (bs - 1)) != 0) return false;
    if ((nbytes      & (bs - 1)) != 0) return false;

    if (logical_off + nbytes > h->logical_size_bytes) return false;

    // Convert to disk-side (post-header) byte offset, then to
    // a block index inside the file's allocated region.
    const uint64_t disk_off       = logical_off + (uint64_t)h->header_bytes;
    const uint64_t want_blk_first = disk_off >> bs_log;
    const uint64_t want_blk_count = nbytes   >> bs_log;
    const uint64_t want_blk_last  = want_blk_first + want_blk_count;  // exclusive

    uint64_t cursor = 0;
    for (uint32_t i = 0; i < h->num_extents; ++i) {
        const uint64_t ext_start = cursor;
        const uint64_t ext_end   = cursor + h->extents[i].length_blocks;
        if (want_blk_first >= ext_start && want_blk_last <= ext_end) {
            const uint64_t off_in_ext = want_blk_first - ext_start;
            *starting_lba_out =
                h->extents[i].start_lba + off_in_ext;
            *n_blocks_out     = want_blk_count;
            return true;
        }
        cursor = ext_end;
    }
    return false;  // not found OR spans extents
}

/// Submit one NVMe read against this file, polling synchronously
/// for completion.  prp1 / prp2 are NVMe-DMA-resolved IOVAs of
/// the buffer that should receive the data; the buffer must be
/// of size >= nbytes and reachable by the controller.
__device__ __forceinline__
void submit_read_one(const NvmeFileDeviceHandle* h,
                     uint64_t                    prp1,
                     uint64_t                    prp2,
                     uint64_t                    logical_off,
                     uint64_t                    nbytes)
{
    uint64_t starting_lba = 0;
    uint64_t n_blocks     = 0;
    if (!resolve_lba(h, logical_off, nbytes, &starting_lba, &n_blocks)) {
        // R5b: no error channel back to host yet.  Print + bail.
        printf("submit_read_one: resolve_lba failed "
               "(file=%llu off=%llu len=%llu)\n",
               (unsigned long long)h->file_id,
               (unsigned long long)logical_off,
               (unsigned long long)nbytes);
        return;
    }

    const uint32_t qidx = QueueAcquireHelper::acquire_queue(h->num_d_qps);
    QueuePair* qp = &h->d_qps[qidx];
    uint16_t cid = 0;
    QueueAcquireHelper::issue_nvme_cmd(
        qp, prp1, prp2, n_blocks, starting_lba,
        (uint8_t)NVM_IO_READ, &cid);
    QueueAcquireHelper::poll(qp, cid);
}

/// Submit one NVMe write against this file, polling synchronously
/// for completion.  Same buffer / sizing rules as submit_read_one.
__device__ __forceinline__
void submit_write_one(const NvmeFileDeviceHandle* h,
                      uint64_t                    prp1,
                      uint64_t                    prp2,
                      uint64_t                    logical_off,
                      uint64_t                    nbytes)
{
    uint64_t starting_lba = 0;
    uint64_t n_blocks     = 0;
    if (!resolve_lba(h, logical_off, nbytes, &starting_lba, &n_blocks)) {
        printf("submit_write_one: resolve_lba failed "
               "(file=%llu off=%llu len=%llu)\n",
               (unsigned long long)h->file_id,
               (unsigned long long)logical_off,
               (unsigned long long)nbytes);
        return;
    }

    const uint32_t qidx = QueueAcquireHelper::acquire_queue(h->num_d_qps);
    QueuePair* qp = &h->d_qps[qidx];
    uint16_t cid = 0;
    QueueAcquireHelper::issue_nvme_cmd(
        qp, prp1, prp2, n_blocks, starting_lba,
        (uint8_t)NVM_IO_WRITE, &cid);
    QueueAcquireHelper::poll(qp, cid);
}

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_NVME_STORAGE_DEVICE_CUH__

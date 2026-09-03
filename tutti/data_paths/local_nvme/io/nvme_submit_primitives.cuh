#pragma once

// tutti/data_paths/local_nvme/io/nvme_submit_primitives.cuh
//
// Shared NVMe submission primitives.
//
// Extracted from submit_one.cuh (Round 15 Session 5) so that both the
// single-device kernel (submit_one.cuh's submit_one_kernel) and the
// striped multi-device kernel (striped_local_nvme's fused_submit_kernel)
// can reuse the same resolve_lba + queue acquire + SQE issue + CQ bounded
// poll code without duplication.
//
// Layout:
//   - Host-visible section (no __CUDACC__ guard): EntryCompletionStatus.
//     Plain include-able from .cpp translation units.
//   - Device-only section (__CUDACC__ guard): QueueAcquireHelper,
//     resolve_lba, submit_read_one, submit_write_one.  Pulls in libnvm
//     device headers; must NOT be included from plain .cpp files.
//
// No behavioral change to LocalNvmeDataPath — every function below is
// byte-for-byte identical to what used to live directly in submit_one.cuh.

#include <cstdint>

#include "tutti/data_paths/local_nvme/io/tutti_gpu_primitives.cuh"

namespace tutti::data_paths::local_nvme {

// -------------------------------------------------------------------------
// EntryCompletionStatus -- per-entry completion status written by the
// device kernel and read back by the host after the stream event signals.
//
//   result: 0 = success, 1 = resolve_lba failure,
//           2 = CQ timeout, 3 = NVMe CQ error status (non-zero SCT/SC),
//           UINT32_MAX = pending / kernel did not execute this entry
//   nvme_status_dword3: raw CQE dword[3] for error diagnostics (0 if ok)
// -------------------------------------------------------------------------
constexpr std::uint32_t kEntryCompletionPending = UINT32_MAX;

struct EntryCompletionStatus {
    std::uint32_t result = kEntryCompletionPending;
    std::uint32_t nvme_status_dword3 = 0;
};

} // namespace tutti::data_paths::local_nvme

// =========================================================================
// Device-only code below — compiled only by nvcc.
// =========================================================================
#if defined(__CUDACC__)

#include "tutti/data_paths/local_nvme/io/device_target.h"

// libnvm device headers — same include ordering as main's
// queue_acquire_helper.cuh.
#include <ctrl.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_cmd.h>
#include <nvm_io.h>

#include <cstdio>

namespace tutti::data_paths::local_nvme {

// -------------------------------------------------------------------------
// QueueAcquireHelper — ported verbatim from main.
// -------------------------------------------------------------------------

class QueueAcquireHelper {
public:
    TUTTI_DEVICE TUTTI_FORCEINLINE
    static std::uint32_t acquire_queue(std::uint32_t num_queues) {
        return (TUTTI_BLOCK_DIM_X * 32u + TUTTI_THREAD_IDX_X) % num_queues;
    }

    TUTTI_DEVICE TUTTI_FORCEINLINE
    static void release_queue(std::uint32_t /*queue_idx*/) {}

    TUTTI_DEVICE TUTTI_FORCEINLINE
    static void issue_nvme_cmd(QueuePair* qp,
                               std::uint64_t  prp1,
                               std::uint64_t  prp2,
                               std::uint64_t  n_blocks,
                               std::uint64_t  starting_lba,
                               std::uint8_t   opcode,
                               std::uint16_t* cid_out)
    {
        nvm_cmd_t cmd;
        nvm_cmd_clear(&cmd);           // zero-init entire 64-byte SQE
        *cid_out = get_cid(&qp->sq);
        nvm_cmd_header  (&cmd, *cid_out, opcode, qp->nvmNamespace);
        nvm_cmd_data_ptr(&cmd, prp1, prp2);
        nvm_cmd_rw_blks (&cmd, starting_lba, n_blocks);
        sq_enqueue(&qp->sq, &cmd);
    }

    // Bounded poll: returns the real CQ slot on completion (whether or not
    // the CQE reports an NVMe error), or NVM_CQ_TIMEOUT on budget exhaustion.
    __device__ __forceinline__
    static uint32_t poll_bounded(QueuePair* qp, uint16_t cid,
                                  uint32_t cq_poll_budget,
                                  uint32_t* out_status_dword3)
    {
        uint32_t loc = cq_poll_bounded(&qp->cq, cid, cq_poll_budget);
        if (loc == NVM_CQ_TIMEOUT) {
            return NVM_CQ_TIMEOUT;
        }
        nvm_cpl_t* cpl = (nvm_cpl_t*)qp->cq.vaddr;
        uint32_t cpl_entry = cpl[loc].dword[3];
        *out_status_dword3 = cpl_entry;
        cq_dequeue(&qp->cq, loc, &qp->sq);
        put_cid(&qp->sq, cid);
        return loc;
    }
};

// -------------------------------------------------------------------------
// resolve_lba — maps (logical_off_bytes, nbytes) onto (starting_lba, n_blocks)
// by walking DeviceTargetHandle::extents.  Rejects requests that span
// multiple extents.
// -------------------------------------------------------------------------

TUTTI_DEVICE TUTTI_FORCEINLINE
bool try_lba_extent(const DeviceLbaExtent& ext,
                    std::uint64_t          ext_start,
                    std::uint64_t          want_blk_first,
                    std::uint64_t          want_blk_last,
                    std::uint64_t          want_blk_count,
                    std::uint64_t*         starting_lba_out,
                    std::uint64_t*         n_blocks_out)
{
    const std::uint64_t ext_end = ext_start + ext.length_blocks;
    if (want_blk_first >= ext_start && want_blk_last <= ext_end) {
        const std::uint64_t off_in_ext = want_blk_first - ext_start;
        *starting_lba_out = ext.start_lba + off_in_ext;
        *n_blocks_out     = want_blk_count;
        return true;
    }
    return false;
}

TUTTI_DEVICE TUTTI_FORCEINLINE
bool resolve_lba(const DeviceTargetHandle* h,
                 std::uint64_t             logical_off,
                 std::uint64_t             nbytes,
                 std::uint64_t*            starting_lba_out,
                 std::uint64_t*            n_blocks_out,
                 std::uint32_t             inject_flag = 0)
{
    if (inject_flag & 0x1u) return false;

    if (h == nullptr || h->num_extents == 0) return false;
    if (nbytes == 0)                         return false;

    const std::uint32_t bs     = h->nvme_block_size;
    const std::uint32_t bs_log = h->nvme_block_size_log;

    if ((logical_off & (bs - 1)) != 0) return false;
    if ((nbytes      & (bs - 1)) != 0) return false;

    if (logical_off + nbytes > h->logical_size_bytes) return false;

    const std::uint64_t disk_off       = logical_off + (std::uint64_t)h->header_bytes;
    const std::uint64_t want_blk_first = disk_off >> bs_log;
    const std::uint64_t want_blk_count = nbytes   >> bs_log;
    const std::uint64_t want_blk_last  = want_blk_first + want_blk_count;

    std::uint64_t cursor = 0;

    const std::uint32_t n_inline = h->num_extents < kDeviceTargetInlineExtents
                                 ? h->num_extents
                                 : kDeviceTargetInlineExtents;
    for (std::uint32_t i = 0; i < n_inline; ++i) {
        if (try_lba_extent(h->extents[i], cursor, want_blk_first, want_blk_last,
                           want_blk_count, starting_lba_out, n_blocks_out))
            return true;
        cursor += h->extents[i].length_blocks;
    }

    if (h->num_extents > kDeviceTargetInlineExtents &&
        h->extents_overflow != nullptr) {
        const std::uint32_t n_overflow = h->num_extents - kDeviceTargetInlineExtents;
        for (std::uint32_t j = 0; j < n_overflow; ++j) {
            if (try_lba_extent(h->extents_overflow[j], cursor, want_blk_first,
                               want_blk_last, want_blk_count,
                               starting_lba_out, n_blocks_out))
                return true;
            cursor += h->extents_overflow[j].length_blocks;
        }
    }

    return false;
}

// -------------------------------------------------------------------------
// submit_read_one / submit_write_one — resolve_lba + acquire queue +
// issue SQE + bounded CQ poll.  Writes EntryCompletionStatus.
// -------------------------------------------------------------------------

TUTTI_DEVICE TUTTI_FORCEINLINE
void submit_read_one(const DeviceTargetHandle* h,
                     std::uint64_t              prp1,
                     std::uint64_t              prp2,
                     std::uint64_t              logical_off,
                     std::uint64_t              nbytes,
                     EntryCompletionStatus*     status,
                     std::uint32_t              cq_poll_budget,
                     std::uint32_t              inject_flag)
{
    std::uint64_t starting_lba = 0;
    std::uint64_t n_blocks     = 0;
    if (!resolve_lba(h, logical_off, nbytes, &starting_lba, &n_blocks, inject_flag)) {
        if (status) { status->result = 1; status->nvme_status_dword3 = 0; }
        return;
    }

    const std::uint32_t qidx = QueueAcquireHelper::acquire_queue(h->num_d_qps);
    QueuePair* qp = &h->d_qps[qidx];
    std::uint16_t cid = 0;
    QueueAcquireHelper::issue_nvme_cmd(
        qp, prp1, prp2, n_blocks, starting_lba,
        (std::uint8_t)NVM_IO_READ, &cid);

    std::uint32_t status_dword3 = 0;
    uint32_t loc = QueueAcquireHelper::poll_bounded(
        qp, cid, cq_poll_budget, &status_dword3);
    if (loc == NVM_CQ_TIMEOUT) {
        if (status) { status->result = 2; status->nvme_status_dword3 = 0; }
        return;
    }
    if ((status_dword3 >> 17) != 0) {
        if (status) { status->result = 3; status->nvme_status_dword3 = status_dword3; }
        return;
    }
    if (inject_flag & 0x2u) {
        if (status) { status->result = 3; status->nvme_status_dword3 = status_dword3 | (1u << 17); }
        return;
    }
    if (status) { status->result = 0; status->nvme_status_dword3 = status_dword3; }
}

TUTTI_DEVICE TUTTI_FORCEINLINE
void submit_write_one(const DeviceTargetHandle* h,
                      std::uint64_t              prp1,
                      std::uint64_t              prp2,
                      std::uint64_t              logical_off,
                      std::uint64_t              nbytes,
                      EntryCompletionStatus*     status,
                      std::uint32_t              cq_poll_budget,
                      std::uint32_t              inject_flag)
{
    std::uint64_t starting_lba = 0;
    std::uint64_t n_blocks     = 0;
    if (!resolve_lba(h, logical_off, nbytes, &starting_lba, &n_blocks, inject_flag)) {
        if (status) { status->result = 1; status->nvme_status_dword3 = 0; }
        return;
    }

    const std::uint32_t qidx = QueueAcquireHelper::acquire_queue(h->num_d_qps);
    QueuePair* qp = &h->d_qps[qidx];
    std::uint16_t cid = 0;
    QueueAcquireHelper::issue_nvme_cmd(
        qp, prp1, prp2, n_blocks, starting_lba,
        (std::uint8_t)NVM_IO_WRITE, &cid);

    std::uint32_t status_dword3 = 0;
    uint32_t loc = QueueAcquireHelper::poll_bounded(
        qp, cid, cq_poll_budget, &status_dword3);
    if (loc == NVM_CQ_TIMEOUT) {
        if (status) { status->result = 2; status->nvme_status_dword3 = 0; }
        return;
    }
    if ((status_dword3 >> 17) != 0) {
        if (status) { status->result = 3; status->nvme_status_dword3 = status_dword3; }
        return;
    }
    if (inject_flag & 0x2u) {
        if (status) { status->result = 3; status->nvme_status_dword3 = status_dword3 | (1u << 17); }
        return;
    }
    if (status) { status->result = 0; status->nvme_status_dword3 = status_dword3; }
}

} // namespace tutti::data_paths::local_nvme

#endif // __CUDACC__

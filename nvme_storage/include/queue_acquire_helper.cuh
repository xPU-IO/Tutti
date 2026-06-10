#ifndef __TUTTI_NVME_STORAGE_QUEUE_ACQUIRE_HELPER_CUH__
#define __TUTTI_NVME_STORAGE_QUEUE_ACQUIRE_HELPER_CUH__

/**
 * queue_acquire_helper.cuh -- on-GPU queue selection + submit.
 *
 * Layer: nvme_storage (R5b).
 *
 * Verbatim port of legacy filesystems/ext4/libgeminifs/include/helper.cuh:
 *
 *   - acquire_queue:  hash-only.  No actual locking, no release path
 *                     needed.  We rely on libnvm's get_cid/put_cid
 *                     to serialize per-queue SQ slot allocation.
 *   - issue_nvme_cmd: compose SQE + nvm_cmd_*; sq_enqueue does the
 *                     actual ring write + doorbell tail update.
 *   - poll:           cq_poll busy-loops on the CQ phase bit, then
 *                     cq_dequeue + put_cid.  No interrupt path.
 *
 * Header-only because every method is either inline or trivially
 * forwarding.  Callable only from CUDA device code.
 *
 * Concurrency model
 *   - Many threads can call acquire_queue() and get the same
 *     queue index (it's a hash, not an exclusive lease).  That's
 *     fine: SQ slot allocation is serialized inside libnvm's
 *     get_cid via atomic CAS on the tail pointer.
 *   - poll() spins on the CQ phase bit for the cid we own.  If
 *     two threads share a queue and pick adjacent slots, each
 *     finds its own CQE -- cq_poll keys on cid, not position.
 *
 * Behavior assumed by callers:
 *   - submit_read_one / submit_write_one wrap acquire_queue +
 *     issue_nvme_cmd + poll into a single inline call so the
 *     three pieces are never used out of order.
 */

#include <cstdint>

// libnvm has a circular ctrl.h <-> queue.h include relationship.
// Both headers expect to be the "outer" include; entering through
// queue.h directly leaves Controller seeing QueuePair as incomplete.
// Always go through ctrl.h first; it pulls queue.h on its own.
#include <ctrl.h>            // libnvm: outer include, transitively brings queue.h
#include <queue.h>           // libnvm: QueuePair definition
#include <nvm_parallel_queue.h>  // libnvm: get_cid / put_cid / sq_enqueue / cq_poll / cq_dequeue
#include <nvm_cmd.h>         // libnvm: nvm_cmd_header / data_ptr / rw_blks
#include <nvm_io.h>          // libnvm: NVM_IO_READ / NVM_IO_WRITE opcodes

namespace tutti {

class QueueAcquireHelper {
public:
    /// Pick a queue index for this thread.  Pure hash; multiple
    /// threads in the same warp / block CAN end up on the same
    /// queue (libnvm's per-queue cid pool handles the rest).
    ///
    /// `num_queues` must be > 0.
    __device__ __forceinline__
    static uint32_t acquire_queue(uint32_t num_queues) {
        // legacy: (blockDim.x * 32 + threadIdx.x) % num_queues
        // The 32 isn't warp-related; it's just a stride that
        // spreads warps across queues.  Kept verbatim to match
        // legacy behavior bit-for-bit.
        return (blockDim.x * 32u + threadIdx.x) % num_queues;
    }

    /// No-op release.  Kept for symmetry with legacy callers and
    /// in case a future locked variant lands.
    __device__ __forceinline__
    static void release_queue(uint32_t /*queue_idx*/) {}

    /// Compose SQE + write to SQ ring + ring doorbell.  Caller
    /// must invoke `poll(qp, *cid_out)` afterwards (or batch
    /// across multiple SQEs and poll each).
    __device__ __forceinline__
    static void issue_nvme_cmd(QueuePair* qp,
                               uint64_t  prp1,
                               uint64_t  prp2,
                               uint64_t  n_blocks,
                               uint64_t  starting_lba,
                               uint8_t   opcode,
                               uint16_t* cid_out)
    {
        nvm_cmd_t cmd;
        *cid_out = get_cid(&qp->sq);
        nvm_cmd_header  (&cmd, *cid_out, opcode, qp->nvmNamespace);
        nvm_cmd_data_ptr(&cmd, prp1, prp2);
        nvm_cmd_rw_blks (&cmd, starting_lba, n_blocks);
        sq_enqueue(&qp->sq, &cmd);
    }

    /// Busy-poll the CQ for `cid`'s CQE, then dequeue + return
    /// the cid to the pool.  Returns when CQE is observed.
    __device__ __forceinline__
    static void poll(QueuePair* qp, uint16_t cid) {
        uint32_t cq_pos = cq_poll(&qp->cq, cid);
        cq_dequeue(&qp->cq, cq_pos, &qp->sq);
        put_cid(&qp->sq, cid);
    }
};

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_QUEUE_ACQUIRE_HELPER_CUH__

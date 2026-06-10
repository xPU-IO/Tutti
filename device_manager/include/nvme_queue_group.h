#ifndef __TUTTI_DEVICE_MANAGER_NVME_QUEUE_GROUP_H__
#define __TUTTI_DEVICE_MANAGER_NVME_QUEUE_GROUP_H__

/**
 * nvme_queue_group.h -- on-GPU NVMe user-queue pool.
 *
 * Layer: device_manager.
 *
 * What this is
 * ============
 * A self-contained class that creates one NVM_CREATE_QUEUE_GROUP +
 * N user-queue pairs around an *externally-owned* nvm_ctrl_t* and
 * stages the resulting QueuePair[] array on a GPU so kernels can
 * submit NVMe commands directly via SQ ring writes + doorbell taps.
 *
 * Equivalent to the GPU-side half of libnvm's monolithic
 * `Controller`, but stripped of:
 *   - bring-up (init_b3 / attach_client)
 *   - host filesystem mount/umount
 *   - ctrl ownership
 *
 * The caller (typically `LocalNvmeDirectRegistry::open_one` or
 * `NvmeServiceBackedRegistry::open_one`) has already produced the
 * `nvm_ctrl_t*` via init_b3 or attach_client; this class just adds
 * the queue group + d_qps[] on top.  The caller continues to own
 * `nvm_ctrl_t*` and is responsible for nvm_ctrl_free / free_client
 * AFTER ~NvmeQueueGroup() returns.
 *
 * Why not reuse libnvm's `Controller`
 * ===================================
 * `Controller`'s ctor is monolithic -- it does init_b3 + init_queues
 * + Host_file_system_int in one shot.  In the eight-layer
 * architecture, init_b3 is device_manager's job and mount is
 * nvme_storage's job, so plumbing them around `Controller` requires
 * either a "wrap" ctor + `ctrl_borrowed` flag (libnvm pollution) or
 * an entire reimplementation.  We chose the latter -- this class --
 * to keep libnvm headers untouched and to give us a clean place to
 * later add queue-group variants (RDMA queue group, GDS queue
 * group, ...).
 *
 * Future-proofing
 * ===============
 * The name `NvmeQueueGroup` is deliberate: it's NVMe-specific.
 * When/if RDMA or GDS backends arrive they get their own
 * `RdmaQueueGroup` / `GdsQueueGroup` types.  An `IQueueGroup`
 * abstract base may eventually crystallise; that's deferred to
 * when there is a second concrete backend to compare shapes
 * against (YAGNI).
 *
 * Lifetime
 * ========
 * - ctor (RAII):
 *     1. nvm_create_group(borrowed_ctrl) -> group_id
 *     2. for each queue: new QueuePair(B3 ctor) on its target GPU
 *     3. NVM_ADD_USER_QUEUE batch (one ioctl, all SQs+CQs)
 *     4. doorbell host-VA -> GPU-VA via cudaHostGetDevicePointer
 *     5. cudaMalloc d_qps[] on the primary cuda device, memcpy
 *        each populated QueuePair into d_qps[i]
 *   On any failure throws std::runtime_error; partial state
 *   (already-created QueuePairs / group) is rolled back before
 *   throwing.
 *
 * - dtor (no exceptions):
 *     1. nvm_destroy_group(group_id) cascades through Delete I/O
 *        SQ + Delete I/O CQ + ring-map cleanup; user QIDs return
 *        to the kernel pool
 *     2. cudaFree(d_qps)
 *     3. delete each h_qps[i] (frees its sq_mem / cq_mem DmaPtr +
 *        intra-queue tickets/marks)
 *   ctrl is left intact -- caller frees it.
 *
 * Threading
 * =========
 * Construction / destruction are not thread-safe (typical for
 * resource RAII).  Once constructed, `d_qps()` / `n_qps()` /
 * `group_id()` are read-only and may be queried concurrently.
 */

#include <cstdint>
#include <memory>

#include <nvm_types.h>      // nvm_ctrl_t, struct disk

// Forward-decl libnvm's QueuePair (queue.h is heavy, full of CUDA
// pulls; we only need the pointer type here).
struct QueuePair;

namespace tutti {

class NvmeQueueGroup {
public:
    /// Build a new queue group around `borrowed_ctrl`.  `borrowed_ctrl`
    /// MUST already have NVM_GET_DEV_INFO applied (q_depth > 0,
    /// max_user_qid > 0, page_size > 0); both nvm_controller_init_b3
    /// and nvm_ctrl_attach_client guarantee this.
    ///
    /// Throws std::runtime_error on any libnvm / CUDA failure.
    NvmeQueueGroup(nvm_ctrl_t*    borrowed_ctrl,
                   const struct disk& disk_info,    // .ns_id, .block_size, .disk_name
                   uint32_t       ns_id,
                   uint32_t       cuda_device,      // primary GPU for d_qps + per-queue rings
                   uint32_t       num_queues,       // count of user QPs to create
                   uint32_t       queue_depth);     // SQ/CQ ring depth

    ~NvmeQueueGroup();

    NvmeQueueGroup(const NvmeQueueGroup&)            = delete;
    NvmeQueueGroup& operator=(const NvmeQueueGroup&) = delete;

    /// GPU-resident QueuePair[] array.  Length == n_qps().  Lives on
    /// `cuda_device` (the one passed to ctor).  Caller MUST NOT free.
    QueuePair* d_qps()      const { return d_qps_; }
    uint32_t   n_qps()      const { return n_qps_; }

    /// Per-fd queue group id assigned by NVM_CREATE_QUEUE_GROUP.
    /// Mostly informational (dtor uses it internally).
    uint32_t   group_id()   const { return group_id_; }

    /// CUDA device on which `d_qps_` was allocated.
    uint32_t   cuda_device() const { return cuda_device_; }

private:
    void init_(uint32_t ns_id, uint32_t num_queues, uint32_t queue_depth);
    void destroy_locked_();

    nvm_ctrl_t*    ctrl_         = nullptr;       // borrowed; not freed
    QueuePair**    h_qps_        = nullptr;
    QueuePair*     d_qps_        = nullptr;
    uint32_t       n_qps_        = 0;
    uint32_t       group_id_     = 0;
    uint32_t       cuda_device_  = 0;

    // disk_info copy used during init (ns_id, block_size, disk_name);
    // not currently re-read after ctor.
    uint32_t       ns_id_        = 0;
    uint32_t       block_size_   = 0;
};

} // namespace tutti

#endif // __TUTTI_DEVICE_MANAGER_NVME_QUEUE_GROUP_H__

/**
 * nvme_queue_group.cu -- NvmeQueueGroup implementation.
 *
 * Layer: device_manager.
 *
 * Reimplements the GPU-side half of libnvm `Controller::init_queues`
 * + dtor, against an externally-owned `nvm_ctrl_t*`.  The flow is
 * verbatim from libnvm's `init_queues` (B3) -- this is intentional:
 * we want the bytes hitting the kernel ABI to be identical.
 */

#include "nvme_queue_group.h"

// libnvm public surface -- NOTE: ctrl.h *first* to avoid the
// circular ctrl.h<->queue.h include relationship (queue.h alone
// leaves QueuePair incomplete via a different include order).
#include <ctrl.h>             // pulls queue.h with the right ordering
#include <queue.h>            // QueuePair (full def)
#include <nvm_ctrl.h>         // nvm_create_group / nvm_destroy_group / nvm_add_user_queue
#include <nvm_error.h>        // nvm_strerror
#include <ioctl.h>            // nvm_ioctl_add_user_queue, NVM_MAX_QUEUES_PER_GROUP

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace tutti {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("NvmeQueueGroup: CUDA error in ") + what +
            ": " + cudaGetErrorString(e));
    }
}

} // namespace

NvmeQueueGroup::NvmeQueueGroup(nvm_ctrl_t*       borrowed_ctrl,
                               const struct disk& disk_info,
                               uint32_t           ns_id,
                               uint32_t           cuda_device,
                               uint32_t           num_queues,
                               uint32_t           queue_depth)
    : ctrl_(borrowed_ctrl),
      cuda_device_(cuda_device),
      ns_id_(ns_id),
      block_size_(static_cast<uint32_t>(disk_info.block_size))
{
    if (ctrl_ == nullptr) {
        throw std::runtime_error("NvmeQueueGroup: ctrl is null");
    }
    // BAR0 must already be mmap'd AND cudaHostRegister(IoMemory)'d by the
    // bring-up path (nvm_controller_init_b3 or nvm_ctrl_attach_client).
    // We rely on ctrl_->mm_ptr both for host-side doorbell pointer math
    // (mm_ptr + sq/cq_doorbell_offset, see init_ step 4) and for
    // cudaHostGetDevicePointer to translate those into GPU VAs.  If
    // mm_ptr is null here the bring-up was skipped or failed and we
    // would crash dereferencing it; fail loudly instead.
    if (ctrl_->mm_ptr == nullptr) {
        throw std::runtime_error(
            "NvmeQueueGroup: ctrl->mm_ptr is null (BAR0 not mapped; "
            "ctrl bring-up via nvm_controller_init_b3 / "
            "nvm_ctrl_attach_client did not complete)");
    }
    if (num_queues == 0) {
        throw std::runtime_error("NvmeQueueGroup: num_queues == 0");
    }
    if (queue_depth == 0) {
        throw std::runtime_error("NvmeQueueGroup: queue_depth == 0");
    }
    if (block_size_ == 0) {
        throw std::runtime_error("NvmeQueueGroup: block_size == 0");
    }

    try {
        init_(ns_id, num_queues, queue_depth);
    } catch (...) {
        destroy_locked_();   // best-effort rollback
        throw;
    }
}

void NvmeQueueGroup::init_(uint32_t ns_id,
                           uint32_t num_queues,
                           uint32_t queue_depth)
{
    int status = 0;

    // Step 1: per-fd queue group container.
    uint32_t max_q = 0;
    status = nvm_create_group(ctrl_, &group_id_, &max_q);
    if (status != 0) {
        throw std::runtime_error(
            std::string("nvm_create_group failed: ") + nvm_strerror(status));
    }
    if (group_id_ == 0 || max_q == 0) {
        throw std::runtime_error(
            "nvm_create_group returned bogus gid/max_q");
    }

    // Authoritative caps come from kernel: NVM_MAX_QUEUES_PER_GROUP
    // bounds per-group user-queue count, and the user QID pool
    // [start_cq_idx, max_user_qid] bounds the absolute queue count.
    uint32_t qpool_room = (ctrl_->max_user_qid >= ctrl_->start_cq_idx)
                            ? (ctrl_->max_user_qid - ctrl_->start_cq_idx + 1)
                            : 0;
    uint32_t kernel_cap = std::min<uint32_t>(max_q, qpool_room);
    if (kernel_cap == 0) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "NvmeQueueGroup: no user QID room "
            "(max_user_qid=%u start_cq_idx=%u max_q_per_grp=%u)",
            ctrl_->max_user_qid, ctrl_->start_cq_idx, max_q);
        throw std::runtime_error(buf);
    }
    n_qps_ = std::min<uint32_t>(kernel_cap, num_queues);

    // Mirror n_qps onto ctrl_-> bookkeeping (some libnvm internals
    // read these; preserving parity with monolithic Controller).
    ctrl_->cq_num = (uint16_t)n_qps_;
    ctrl_->sq_num = (uint16_t)n_qps_;

    // Step 2: per-queue ring memory.
    h_qps_ = (QueuePair**) std::malloc(sizeof(QueuePair*) * n_qps_);
    if (h_qps_ == nullptr) {
        throw std::runtime_error("NvmeQueueGroup: malloc h_qps failed");
    }
    std::memset(h_qps_, 0, sizeof(QueuePair*) * n_qps_);

    cuda_check(cudaMalloc((void**)&d_qps_, sizeof(QueuePair) * n_qps_),
               "cudaMalloc d_qps");

    for (uint32_t i = 0; i < n_qps_; ++i) {
        // defer_gpu_init=true: tickets/marks/cid are allocated AFTER
        // NVM_ADD_USER_QUEUE succeeds, so a partial failure doesn't
        // leak per-queue GPU buffers.
        h_qps_[i] = new QueuePair(ctrl_,
                                  cuda_device_,
                                  /*qp_id=*/(uint16_t)i,
                                  (uint64_t)queue_depth,
                                  group_id_,
                                  /*defer_gpu_init=*/true);
    }

    // Step 3: batch NVM_ADD_USER_QUEUE for all rings in one shot.
    if (n_qps_ > NVM_MAX_QUEUES_PER_GROUP) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "NvmeQueueGroup: n_qps=%u exceeds NVM_MAX_QUEUES_PER_GROUP=%u",
            n_qps_, (unsigned)NVM_MAX_QUEUES_PER_GROUP);
        throw std::runtime_error(buf);
    }
    struct nvm_ioctl_add_user_queue req;
    std::memset(&req, 0, sizeof(req));
    req.group_id = group_id_;
    req.nr_pairs = n_qps_;
    for (uint32_t i = 0; i < n_qps_; ++i) {
        req.pairs[i].sq_vaddr = (uint64_t)(uintptr_t) h_qps_[i]->sq_mem.get()->vaddr;
        req.pairs[i].cq_vaddr = (uint64_t)(uintptr_t) h_qps_[i]->cq_mem.get()->vaddr;
    }
    status = nvm_add_user_queue(ctrl_, &req);
    if (status != 0) {
        throw std::runtime_error(
            std::string("nvm_add_user_queue failed: ") + nvm_strerror(status));
    }

    // Step 4: bind doorbells + finish per-queue GPU init + cudaMemcpy
    // into d_qps[i].
    cuda_check(cudaSetDevice(cuda_device_), "cudaSetDevice(primary)");

    void* devicePtr = nullptr;
    for (uint32_t i = 0; i < n_qps_; ++i) {
        // Host-side doorbell pointers (within BAR0 mmap range).
        volatile uint32_t* sq_db_host = (volatile uint32_t*)
            ((uintptr_t) ctrl_->mm_ptr + req.out_pairs[i].sq_doorbell_offset);
        volatile uint32_t* cq_db_host = (volatile uint32_t*)
            ((uintptr_t) ctrl_->mm_ptr + req.out_pairs[i].cq_doorbell_offset);

        QueuePair* qp = h_qps_[i];
        qp->pageSize             = ctrl_->page_size;
        qp->block_size           = block_size_;
        qp->block_size_minus_1   = block_size_ - 1;
        qp->block_size_log       = (uint32_t) std::log2((double)block_size_);
        qp->nvmNamespace         = ns_id;

        qp->cq.no       = (uint16_t) req.out_pairs[i].qid;
        qp->cq.es       = sizeof(nvm_cpl_t);
        qp->cq.head     = 0;
        qp->cq.tail     = 0;
        qp->cq.last     = 0;
        qp->cq.phase    = 1;
        qp->cq.local    = 0;
        qp->cq.head_lock = 0;
        qp->cq.tail_lock = 0;
        qp->cq.in_ticket = 0;
        qp->cq.cid_ticket = 0;
        qp->cq.vaddr    = qp->cq_mem.get()->vaddr;
        qp->cq.ioaddr   = qp->cq_mem.get()->ioaddrs[0];
        qp->cq.db       = cq_db_host;

        qp->sq.no       = (uint16_t) req.out_pairs[i].qid;
        qp->sq.es       = sizeof(nvm_cmd_t);
        qp->sq.head     = 0;
        qp->sq.tail     = 0;
        qp->sq.last     = 0;
        qp->sq.phase    = 1;
        qp->sq.local    = 0;
        qp->sq.head_lock = 0;
        qp->sq.tail_lock = 0;
        qp->sq.in_ticket = 0;
        qp->sq.cid_ticket = 0;
        qp->sq.vaddr    = qp->sq_mem.get()->vaddr;
        qp->sq.ioaddr   = qp->sq_mem.get()->ioaddrs[0];
        qp->sq.db       = sq_db_host;

        // Allocate intra-queue tickets/marks/cid/pos_locks now that
        // the ring is alive on the controller.
        qp->init_gpu_specific_struct(cuda_device_);

        // Translate doorbell host VA -> GPU VA (BAR0 was already
        // cudaHostRegister'd by the bring-up path -- init_b3 or
        // attach_client).
        cuda_check(cudaHostGetDevicePointer(&devicePtr, (void*) qp->cq.db, 0),
                   "cudaHostGetDevicePointer(cq.db)");
        qp->cq.db = (volatile uint32_t*) devicePtr;
        cuda_check(cudaHostGetDevicePointer(&devicePtr, (void*) qp->sq.db, 0),
                   "cudaHostGetDevicePointer(sq.db)");
        qp->sq.db = (volatile uint32_t*) devicePtr;

        cuda_check(cudaMemcpy(d_qps_ + i, qp, sizeof(QueuePair),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy QueuePair -> d_qps[i]");
    }
}

NvmeQueueGroup::~NvmeQueueGroup() {
    destroy_locked_();
}

void NvmeQueueGroup::destroy_locked_() {
    // Mirrors libnvm `Controller::~Controller` minus the
    // umount + nvm_ctrl_free (caller owns those).
    //
    //   1. nvm_destroy_group cascades Delete I/O SQ+CQ + ring-map
    //      cleanup; user QIDs return to kernel pool.
    //   2. cudaFree(d_qps).
    //   3. delete each h_qps[i] (DmaPtr ring members + cudaFree of
    //      per-queue GPU buffers).

    if (group_id_ != 0 && ctrl_ != nullptr) {
        int rc = nvm_destroy_group(ctrl_, group_id_);
        if (rc != 0) {
            // Logged but not fatal: kernel falls back to fd-close
            // cascade (snvm_dev_release).
            std::fprintf(stderr,
                "[nvme_queue_group] nvm_destroy_group(gid=%u) failed: %s\n",
                group_id_, nvm_strerror(rc));
        }
        group_id_ = 0;
    }

    if (d_qps_ != nullptr) {
        cudaError_t e = cudaFree(d_qps_);
        if (e != cudaSuccess) {
            std::fprintf(stderr,
                "[nvme_queue_group] cudaFree(d_qps): %s\n",
                cudaGetErrorString(e));
        }
        d_qps_ = nullptr;
    }

    if (h_qps_ != nullptr) {
        for (uint32_t i = 0; i < n_qps_; ++i) {
            delete h_qps_[i];
        }
        std::free(h_qps_);
        h_qps_ = nullptr;
    }
    n_qps_ = 0;
}

} // namespace tutti

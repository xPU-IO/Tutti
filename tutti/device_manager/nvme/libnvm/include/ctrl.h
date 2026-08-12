#ifndef __BENCHMARK_CTRL_H__
#define __BENCHMARK_CTRL_H__

// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <cstdint>
#include "buffer.h"
#include "ioctl.h"
#include "nvm_types.h"
#include "nvm_ctrl.h"
#include "nvm_error.h"
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda/atomic>
#include "file.h"
#include "queue.h"

// Per-queue memory target. Each IO queue's SQ/CQ ring memory either
// lives on a specific GPU (on_host=false, cuda_device=<id>) or in host
// memory (on_host=true). The host-memory path is reserved for future
// CPU_SUBMIT support: init_queues currently rejects on_host=true with
// ENOTSUP. cuda_device values can be arbitrary, non-contiguous CUDA
// device indices (e.g. {0, 2}); they are NOT treated as positional
// indices into anything.
struct QueueMemTarget {
    bool     on_host     = false;
    uint32_t cuda_device = 0;
};

#define MAX_QUEUES 1024
#define NVM_CTRL_IOQ_MINNUM    64

struct Controller
{
    cuda::atomic<uint64_t, cuda::thread_scope_device> access_counter;
    nvm_ctrl_t*             ctrl;
    struct nvm_ctrl_info    info;
    struct nvm_ns_info      ns;
    struct disk             disk;
    uint16_t                n_sqs;
    uint16_t                n_cqs;
    uint16_t                n_qps;
    uint16_t                n_user_qps;

    uint32_t                deviceId;
    QueuePair**             h_qps;
    QueuePair*              d_qps;


    cuda::atomic<uint64_t, cuda::thread_scope_device> queue_counter;

    uint32_t page_size;
    uint32_t blk_size;
    uint32_t blk_size_log;
    char* dev_path;
    // char* dev_mount_path;
    std::string dev_mount_path;
    void* d_ctrl_ptr;
    BufferPtr d_ctrl_buff;

    // Queue group id assigned by NVM_CREATE_QUEUE_GROUP during
    // init_queues().  ~Controller() destroys the group when non-zero.
    uint32_t                group_id = 0;

    Controller(const char* snvme_control_path,
                const char* pci_addr,
                std::string mount_path,
                uint32_t ns_id,
                uint32_t cudaDevice,
                uint64_t queueDepth,
                uint64_t numQueues);

    // Per-queue-target constructor. queue_targets[i] specifies where
    // queue i's SQ/CQ memory lives. The "primary" cuda device used for
    // the controller-wide d_qps and d_ctrl_buff allocations is the
    // first GPU target found in queue_targets. queue_targets must
    // contain at least one !on_host entry until host-mem queues are
    // implemented.
    Controller(const char* snvme_control_path,
                const char* pci_addr,
                std::string mount_path,
                uint32_t ns_id,
                const std::vector<QueueMemTarget>& queue_targets,
                uint64_t queueDepth);

    void print_reset_stats(void);
    int init_queues(uint32_t ns_id,  uint32_t cudaDevice,
                     uint64_t numQueues, uint64_t queueDepth);

    // Per-queue-target init_queues. Returns ENOTSUP if any
    // queue_targets[i].on_host is true (CPU-resident queues are
    // reserved for a later round). The legacy single-cudaDevice
    // init_queues delegates to this with a uniform GPU vector.
    int init_queues(uint32_t ns_id,
                     const std::vector<QueueMemTarget>& queue_targets,
                     uint64_t queueDepth);

    ~Controller();
};



using error = std::runtime_error;
using std::string;


inline void Controller::print_reset_stats(void) {
    cuda_err_chk(cudaMemcpy(&access_counter, d_ctrl_ptr, sizeof(cuda::atomic<uint64_t, cuda::thread_scope_device>), cudaMemcpyDeviceToHost));
    std::cout << "------------------------------------" << std::endl;
    std::cout << std::dec << "#SSDAccesses:\t" << access_counter << std::endl;

    cuda_err_chk(cudaMemset(d_ctrl_ptr, 0, sizeof(cuda::atomic<uint64_t, cuda::thread_scope_device>)));
}





inline Controller::Controller(const char* snvme_control_path,
                                const char* pci_addr,
                                std::string mount_path,
                                uint32_t ns_id,
                                uint32_t cudaDevice,
                                uint64_t queueDepth,
                                uint64_t numQueues)
    : Controller(snvme_control_path, pci_addr, std::move(mount_path),
                 ns_id,
                 std::vector<QueueMemTarget>(numQueues,
                     QueueMemTarget{/*on_host=*/false, cudaDevice}),
                 queueDepth)
{
    // Delegating constructor: builds a uniform GPU target vector and
    // forwards to the per-queue-target constructor below.
}

inline Controller::Controller(const char* snvme_control_path,
                                const char* pci_addr,
                                std::string mount_path,
                                uint32_t ns_id,
                                const std::vector<QueueMemTarget>& queue_targets,
                                uint64_t queueDepth)
    : ctrl(nullptr), deviceId(0)
{
    if (queue_targets.empty()) {
        nvm_throw_error("Controller: queue_targets is empty", EINVAL);
    }
    // Pick the first GPU-backed entry as the "primary" device used for
    // d_qps / d_ctrl_buff. CPU entries are reserved for a later round.
    bool primary_set = false;
    for (const auto& t : queue_targets) {
        if (!t.on_host) { deviceId = t.cuda_device; primary_set = true; break; }
    }
    if (!primary_set) {
        nvm_throw_error("Controller: no GPU-backed entry in queue_targets "
                        "(host-only queue pools are not yet supported)", ENOTSUP);
    }

    int status;

    /* GPU-capable bring-up: chrdev_create -> SET_KERNEL_IOQ_CAP -> BIND ->
     * wait probe -> GET_DEV_INFO -> ctrl_init -> cudaHostRegister.
     *
     * kernel_ioq_cap=0 means "no cap, use upstream nvme default of
     * num_possible_cpus()".  Callers that need to pin a smaller
     * kernel-side IOQ count (e.g. Intel DC SSD: MSI-X=136 vs 192
     * vCPUs) should expose a separate Controller ctor / param;
     * for now the cap is left at 0 to match the upstream default. */
    status = nvm_controller_init_gpu(&ctrl,
                                     snvme_control_path,
                                     pci_addr,
                                     /* kernel_ioq_cap */ 0,
                                     &this->disk);
    if (status != 0){
        nvm_throw_error("Failed to nvm_controller_init_gpu", status);
    }
    this->disk.ns_id = ns_id;

    status = init_queues(ns_id, queue_targets, queueDepth);
    if (status != 0){
        nvm_throw_error("Failed to init_queues", status);
    }

    page_size = ctrl->page_size;
    blk_size = disk.block_size;
    blk_size_log = std::log2(blk_size);

    dev_path = (char *)malloc(40 * sizeof(char));
    snprintf(dev_path, 40, "/dev/%s", disk.disk_name);

    dev_mount_path = mount_path;

    d_ctrl_buff = createBuffer(sizeof(Controller), deviceId);
    d_ctrl_ptr = d_ctrl_buff.get();
    cuda_err_chk(cudaMemcpy(d_ctrl_ptr, this, sizeof(Controller), cudaMemcpyHostToDevice));

    Host_file_system_int(dev_path, dev_mount_path.c_str());
}


inline int Controller::init_queues(uint32_t ns_id,  uint32_t cudaDevice,
                                    uint64_t numQueues, uint64_t queueDepth){
    // Legacy single-cudaDevice path: delegate to the per-queue-target
    // overload with a uniform vector.
    return init_queues(ns_id,
                       std::vector<QueueMemTarget>(numQueues,
                           QueueMemTarget{/*on_host=*/false, cudaDevice}),
                       queueDepth);
}

inline int Controller::init_queues(uint32_t ns_id,
                                    const std::vector<QueueMemTarget>& queue_targets,
                                    uint64_t queueDepth){
    /*
     * Queue-group based init_queues.
     *
     * Replaces the legacy:
     *   nvm_queue_set(n_sq+n_cq) -> per-q create_queue_Dma rings
     *   -> nvm_device_init (BIND) -> init_userioq_device
     *
     * with the explicit sequence (BIND already happened in the
     * Controller ctor via nvm_controller_init_gpu):
     *
     *   1. nvm_create_group()                   -> group_id
     *   2. for each requested queue:
     *        new QueuePair(queue-group ctor) which calls
     *          create_ring_Dma(group_id, RING_SQ)
     *          create_ring_Dma(group_id, RING_CQ)
     *          (defers init_gpu_specific_struct)
     *   3. NVM_ADD_USER_QUEUE batch (1 ioctl, all (sq_vaddr, cq_vaddr)
     *      pairs at once).  Kernel issues Create I/O CQ + Create I/O SQ
     *      admin commands; on partial failure it unwinds the
     *      already-created pairs before returning.
     *   4. for each pair: write the kernel-returned doorbell offsets
     *      into nvm_queue_t.db, init_gpu_specific_struct, cudaMemcpy
     *      QueuePair -> d_qps[i].
     *
     * Doorbell handling differs from the legacy path: the kernel
     * returns BAR0-relative byte offsets in out_pairs[].sq/cq_doorbell_offset
     * rather than us re-deriving the doorbell address from QID via the
     * SQ_DBL/CQ_DBL macros.  This is the kernel's documented invariant
     * for the queue-group API.  Doorbell GPU VAs are still obtained per-queue via
     * cudaHostGetDevicePointer on the BAR0 host mapping (with cudaSetDevice
     * to the per-queue device first), preserving multi-GPU semantics.
     */

    void* devicePtr = nullptr;
    int status;

    // Reject host-resident queues for now -- API accepts them so the
    // shape is stable, but the implementation only handles GPU memory.
    for (size_t k = 0; k < queue_targets.size(); ++k) {
        if (queue_targets[k].on_host) {
            printf("init_queues: host-resident queue %zu not yet supported\n", k);
            return ENOTSUP;
        }
    }

    const uint64_t numQueues = static_cast<uint64_t>(queue_targets.size());

    // Step 1: per-fd queue group container.
    uint32_t max_q = 0;
    status = nvm_create_group(this->ctrl, &this->group_id, &max_q);
    if (status != 0) {
        printf("nvm_create_group failed: %s\n", nvm_strerror(status));
        return EFAULT;
    }
    if (this->group_id == 0 || max_q == 0) {
        printf("nvm_create_group returned bogus gid=%u max_q=%u\n",
               this->group_id, max_q);
        return EFAULT;
    }

    // Authoritative caps come from kernel: NVM_MAX_QUEUES_PER_GROUP
    // (mirrored into ctrl->max_queues_per_group via NVM_GET_DEV_INFO)
    // bounds the per-group user-queue count, and the user QID pool
    // [start_cq_idx, max_user_qid] bounds the absolute queue count.
    uint32_t qpool_room = (this->ctrl->max_user_qid >= this->ctrl->start_cq_idx)
                            ? (this->ctrl->max_user_qid - this->ctrl->start_cq_idx + 1)
                            : 0;
    uint16_t kernel_cap = static_cast<uint16_t>(std::min<uint32_t>(max_q, qpool_room));
    if (kernel_cap == 0) {
        printf("init_queues: no user QID room (max_user_qid=%u start_cq_idx=%u max_q_per_grp=%u)\n",
               this->ctrl->max_user_qid, this->ctrl->start_cq_idx, max_q);
        nvm_destroy_group(this->ctrl, this->group_id);
        this->group_id = 0;
        return EBUSY;
    }
    this->n_qps = std::min(kernel_cap, (uint16_t)numQueues);
    this->n_sqs = n_qps;
    this->n_cqs = n_qps;

    this->ctrl->cq_num = this->n_cqs;
    this->ctrl->sq_num = this->n_sqs;

    this->h_qps = (QueuePair**) malloc(sizeof(QueuePair*) * n_qps);
    cuda_err_chk(cudaMalloc((void**)&this->d_qps, sizeof(QueuePair) * this->n_qps));

    // Step 2: allocate ring memory per queue pair.
    for (size_t i = 0; i < n_qps; i++) {
        const uint32_t per_queue_dev = queue_targets[i].cuda_device;
        // defer_gpu_init = true: tickets/marks/cid arrays get allocated
        // AFTER NVM_ADD_USER_QUEUE succeeds, so a partial failure
        // doesn't leak per-queue GPU buffers.
        h_qps[i] = new QueuePair(ctrl, per_queue_dev,
                                 (uint16_t)i, queueDepth,
                                 this->group_id,
                                 /*defer_gpu_init=*/true);
    }

    // Step 3: batch NVM_ADD_USER_QUEUE for all rings in one shot.
    if (n_qps > NVM_MAX_QUEUES_PER_GROUP) {
        // Should not happen given the cap above, but guard anyway --
        // the kernel ABI hard-limits nr_pairs to NVM_MAX_QUEUES_PER_GROUP.
        printf("init_queues: n_qps=%u exceeds NVM_MAX_QUEUES_PER_GROUP=%u\n",
               this->n_qps, NVM_MAX_QUEUES_PER_GROUP);
        return EINVAL;
    }
    struct nvm_ioctl_add_user_queue req;
    memset(&req, 0, sizeof(req));
    req.group_id = this->group_id;
    req.nr_pairs = (uint32_t) n_qps;
    for (size_t i = 0; i < n_qps; i++) {
        req.pairs[i].sq_vaddr = (uint64_t)(uintptr_t) h_qps[i]->sq_mem.get()->vaddr;
        req.pairs[i].cq_vaddr = (uint64_t)(uintptr_t) h_qps[i]->cq_mem.get()->vaddr;
    }
    status = nvm_add_user_queue(this->ctrl, &req);
    if (status != 0) {
        printf("nvm_add_user_queue failed: %s\n", nvm_strerror(status));
        // QueuePair dtors + nvm_destroy_group cascade clean up.
        return EFAULT;
    }

    // Step 4: bind doorbells + finish per-queue GPU init.
    //
    // The kernel's NVM_ADD_USER_QUEUE returns BAR0-relative offsets in
    // out_pairs[i].sq/cq_doorbell_offset.  We add those to the BAR0
    // host VA (ctrl->mm_ptr) to get the host-side doorbell address,
    // then translate to a GPU VA via cudaHostGetDevicePointer on the
    // current cudaDevice (which was already cudaHostRegister'd in
    // nvm_controller_init_gpu via cudaHostRegisterIoMemory).
    for (size_t i = 0; i < n_qps; i++) {
        const uint32_t per_queue_dev = queue_targets[i].cuda_device;
        cuda_err_chk(cudaSetDevice(per_queue_dev));

        // Host-side doorbell pointers (within BAR0 mmap range).
        volatile uint32_t* sq_db_host = (volatile uint32_t*)
            ((uintptr_t) this->ctrl->mm_ptr + req.out_pairs[i].sq_doorbell_offset);
        volatile uint32_t* cq_db_host = (volatile uint32_t*)
            ((uintptr_t) this->ctrl->mm_ptr + req.out_pairs[i].cq_doorbell_offset);

        // Populate the host-side nvm_queue_t skeleton.  This replaces
        // nvm_queue_clear's macro-derived db with the kernel-returned
        // offset; the rest of the fields match nvm_queue_clear's
        // semantics).  qid uses the kernel-assigned value, NOT
        // i + start_cq_idx -- the kernel may not always hand them out
        // contiguously (e.g. after recycle).
        QueuePair* qp = h_qps[i];
        qp->pageSize = this->ctrl->page_size;
        qp->block_size = this->disk.block_size;
        qp->block_size_minus_1 = this->disk.block_size - 1;
        qp->block_size_log = std::log2(this->disk.block_size);
        qp->nvmNamespace = ns_id;

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

        // Allocate intra-queue tickets/marks/cid/pos_locks now that the
        // ring is alive on the controller.  This is the half of the old
        // QueuePair ctor that we deferred.
        qp->init_gpu_specific_struct(per_queue_dev);

        // Translate doorbell host VA -> GPU VA for the current device.
        // Done AFTER init_gpu_specific_struct since that path doesn't
        // touch qp->{sq,cq}.db.
        cudaError_t err = cudaHostGetDevicePointer(&devicePtr, (void*) qp->cq.db, 0);
        if (err != cudaSuccess) {
            printf("Failed to get device pointer (cq db): %s\n", cudaGetErrorString(err));
            return -1;
        }
        qp->cq.db = (volatile uint32_t*) devicePtr;

        err = cudaHostGetDevicePointer(&devicePtr, (void*) qp->sq.db, 0);
        if (err != cudaSuccess) {
            printf("Failed to get device pointer (sq db): %s\n", cudaGetErrorString(err));
            return -1;
        }
        qp->sq.db = (volatile uint32_t*) devicePtr;

        // d_qps lives on the primary GPU; cudaMemcpy from any host
        // source to that primary works regardless of per_queue_dev.
        cuda_err_chk(cudaMemcpy(d_qps + i, qp, sizeof(QueuePair),
                                cudaMemcpyHostToDevice));
    }
    return 0;
}

inline Controller::~Controller()
{
    /*
     * Cleanup order matters under B6:
     *
     *   1. NVM_DESTROY_QUEUE_GROUP first.  This cascades through
     *      Delete I/O SQ + Delete I/O CQ on the controller and
     *      releases every RING_SQ/RING_CQ map attached to the group.
     *      User QIDs go back into the kernel's pool.
     *
     *   2. Then drop d_qps (cudaFree on primary GPU) and h_qps
     *      (each delete h_qps[i] runs the QueuePair dtor, which drops
     *      the DmaPtr ring members and cudaFree's the GPU buffers).
     *
     *   3. Standalone teardown: umount + nvm_ctrl_free (which still
     *      cascades unbind+chrdev_remove on owner-built controllers).
     */
    if (this->group_id != 0 && this->ctrl != nullptr) {
        int rc = nvm_destroy_group(this->ctrl, this->group_id);
        if (rc != 0) {
            // Logged but not fatal: the kernel falls back to the
            // fd-close cascade in snvm_dev_release, so the worst
            // case is a noisy dmesg, not a leak.
            printf("Controller dtor: nvm_destroy_group(gid=%u) failed: %s\n",
                   this->group_id, nvm_strerror(rc));
        }
        this->group_id = 0;
    }

    if (d_qps != nullptr) {
        cudaFree(d_qps);
        d_qps = nullptr;
    }
    if (h_qps != nullptr) {
        for (size_t i = 0; i < n_qps; i++) {
            delete h_qps[i];
        }
        free(h_qps);
        h_qps = nullptr;
    }

    int ret = Host_file_system_exit(dev_path);
    if(ret < 0)
        exit(-1);
    printf("Controller release\n");
    nvm_ctrl_free(ctrl);
}


#endif

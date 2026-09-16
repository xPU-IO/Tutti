#pragma once

// tutti/data_paths/local_nvme/io/nvme_queue_group.h
//
// On-GPU NVMe user-queue pool — ported from main's device_manager.
//
// Namespace adapted to tutti::data_paths::local_nvme; structure, creation
// order, doorbell mapping, and destruction order are faithful to source.
//
// The caller (LocalNvmeDataPath) owns the nvm_ctrl_t* and must free it
// AFTER ~NvmeQueueGroup() returns.

#include <cstdint>

#include <nvm_types.h>      // nvm_ctrl_t, struct disk

// Forward-decl libnvm's QueuePair (queue.h is heavy, full of CUDA
// pulls; we only need the pointer type here).
struct QueuePair;

namespace tutti::data_paths::local_nvme {

class NvmeQueueGroup {
public:
    NvmeQueueGroup(nvm_ctrl_t*    borrowed_ctrl,
                   const struct disk& disk_info,
                   uint32_t       ns_id,
                   uint32_t       cuda_device,
                   uint32_t       num_queues,
                   uint32_t       queue_depth);

    ~NvmeQueueGroup();

    NvmeQueueGroup(const NvmeQueueGroup&)            = delete;
    NvmeQueueGroup& operator=(const NvmeQueueGroup&) = delete;

    QueuePair* d_qps()       const { return d_qps_; }
    uint32_t   n_qps()       const { return n_qps_; }
    uint32_t   group_id()    const { return group_id_; }
    uint32_t   cuda_device() const { return cuda_device_; }

private:
    void init_(uint32_t ns_id, uint32_t num_queues, uint32_t queue_depth);
    void destroy_locked_();

    nvm_ctrl_t*    ctrl_         = nullptr;
    QueuePair**    h_qps_        = nullptr;
    QueuePair*     d_qps_        = nullptr;
    uint32_t       n_qps_        = 0;
    uint32_t       group_id_     = 0;
    uint32_t       cuda_device_  = 0;
    uint32_t       ns_id_        = 0;
    uint32_t       block_size_   = 0;
};

} // namespace tutti::data_paths::local_nvme

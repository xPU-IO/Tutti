/**
 * local_nvme_direct_registry.cpp -- IN_PROCESS bootstrap path.
 *
 * Brings up each NVMe controller via nvm_controller_init_b3() with
 * this process as the chrdev / bind owner.  Tear-down goes through
 * nvm_ctrl_free which cascades unbind + chrdev_remove, so a
 * LocalNvmeDirectRegistry MUST be the only owner of each controller
 * it brings up.
 */

#include "local_nvme_direct_registry.h"
#include "nvme_queue_group.h"   // R5b

#include "../../io_engine/include/backend_type.h"

#include <nvm_ctrl.h>
#include <nvm_types.h>

#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace tutti {

LocalNvmeDirectRegistry::LocalNvmeDirectRegistry(
    std::vector<LocalNvmeDirectConfig> cfgs)
    : cfgs_(std::move(cfgs))
{}

LocalNvmeDirectRegistry::~LocalNvmeDirectRegistry() {
    Close();
}

bool LocalNvmeDirectRegistry::Open() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (is_open_) return true;

    slots_.reserve(cfgs_.size());

    for (size_t i = 0; i < cfgs_.size(); ++i) {
        auto slot = std::make_unique<Slot>();
        if (!open_one(cfgs_[i], (int32_t)i, *slot)) {
            std::fprintf(stderr,
                "[direct-registry] open_one(pci=%s) failed; rolling back\n",
                cfgs_[i].pci_addr.c_str());
            close_locked();
            return false;
        }
        slots_.push_back(std::move(slot));
    }

    is_open_ = true;
    return true;
}

bool LocalNvmeDirectRegistry::open_one(const LocalNvmeDirectConfig& cfg,
                                       int32_t                       device_id,
                                       Slot&                         out)
{
    auto bp = std::make_unique<LocalNvmeDevice>();
    bp->device_id      = device_id;
    bp->pci_addr       = cfg.pci_addr;
    bp->attach_mode    = LocalNvmeAttachMode::DIRECT;
    bp->display_name   = cfg.display_name.empty()
                        ? std::string("local_nvme @ ") + cfg.pci_addr
                        : cfg.display_name;

    struct disk d;
    std::memset(&d, 0, sizeof(d));

    int rc = nvm_controller_init_b3(&bp->ctrl,
                                     "/dev/snvm_control",
                                     bp->pci_addr.c_str(),
                                     cfg.kernel_ioq_cap,
                                     &d);
    if (rc != 0 || bp->ctrl == nullptr) {
        std::fprintf(stderr,
            "[direct-registry] nvm_controller_init_b3(pci=%s) rc=%d\n",
            bp->pci_addr.c_str(), rc);
        return false;
    }

    // Persist `disk` so we can hand it to the C++ Controller below
    // (its wrap ctor needs the GET_DEV_INFO data we just got).
    struct disk d_for_wrap = d;

    // /dev/ssnvme<minor> -- nvm_controller_init_b3 doesn't surface
    // the minor explicitly today; in v0.1 we rely on disk_name like
    // "snvme0n1" to derive it (legacy convention).  Fallback to
    // "ssnvme0".
    char path[64];
    int minor = 0;
    if (std::strncmp(d.disk_name, "snvme", 5) == 0) {
        const char* p = d.disk_name + 5;
        char* endp = nullptr;
        long v = std::strtol(p, &endp, 10);
        if (endp != p && v >= 0) minor = (int)v;
    }
    std::snprintf(path, sizeof(path), "/dev/ssnvme%d", minor);
    bp->snvme_dev_path  = path;

    bp->namespace_id    = (d.ns_id != 0) ? d.ns_id : 1u;
    bp->bar0_size       = bp->ctrl->bar0_size;
    bp->dstrd           = bp->ctrl->dstrd;
    bp->blk_size        = (uint32_t)d.block_size;
    // blk_size_log = log2(block_size) for power-of-two block sizes.
    // NVMe block sizes are always 2^k (CAP.MPS-derived).
    {
        uint32_t bs = bp->blk_size;
        uint32_t lg = 0;
        while (bs > 1) { bs >>= 1; ++lg; }
        bp->blk_size_log = lg;
    }
    bp->queue_depth     = bp->ctrl->q_depth;
    bp->max_user_qid    = bp->ctrl->max_user_qid;
    bp->max_queues_per_group = bp->ctrl->max_queues_per_group;
    bp->page_size       = d.page_size;
    bp->max_data_size   = d.max_data_size;     // CTRL.MDTS in bytes
    bp->mount_path.clear();   // direct mode: no daemon, no symlink

    // Fill the runtime-visible Device shell.
    out.device.device_id        = device_id;
    out.device.backend_type     = BackendType::LOCAL_NVME;
    out.device.pci_addr         = bp->pci_addr;
    out.device.display_name     = bp->display_name;
    out.device.backend          = nullptr;   // SPI impl wires this up later (R6)
    out.device.queues           = nullptr;
    out.device.backend_private  = bp.get();

    // CapabilitySet shape for v0.1 -- bits we know are true today.
    auto& caps = out.device.capabilities;
    caps.flags = 0;
    caps.flags |= CAP_SUBMIT_BATCH_GPU_STREAM;
    caps.flags |= CAP_SUBMIT_BATCH_CPU_SYNC;
    caps.flags |= CAP_MEM_HOST_REGISTER;
    caps.flags |= CAP_MEM_PINNED_HOST_REGISTER;
    caps.flags |= CAP_MEM_DEVICE_REGISTER;
    caps.flags |= CAP_TOPO_GPUDIRECT_DMA;
    caps.max_io_bytes     = bp->max_data_size;     // CTRL.MDTS in bytes
    caps.max_batch_count  = 4096;
    caps.max_queue_depth  = bp->queue_depth;
    caps.max_queue_count  = bp->max_queues_per_group;
    caps.page_size        = bp->page_size;

    // R5b: optional NvmeQueueGroup so upper layers can reach
    // d_qps[] for on-GPU NVMe submit kernels.  Skipped by default
    // to avoid forcing every caller to allocate user queues / GPU
    // queue rings; nvme_storage's GPU-side smoke flips this on.
    if (cfg.build_queue_group) {
        const uint32_t nq = cfg.num_user_queues > 0 ? cfg.num_user_queues : 4;
        const uint32_t qd = cfg.queue_depth > 0 ? cfg.queue_depth
                                                : bp->queue_depth;
        if (qd == 0) {
            std::fprintf(stderr,
                "[direct-registry] build_queue_group: q_depth=0 (ctrl "
                "didn't surface q_depth via GET_DEV_INFO?)\n");
            nvm_ctrl_free(bp->ctrl);
            bp->ctrl = nullptr;
            return false;
        }
        try {
            bp->queue_group = std::make_shared<NvmeQueueGroup>(
                bp->ctrl,
                d_for_wrap,
                cfg.namespace_id,
                (uint32_t)cfg.cuda_device,
                nq,
                qd);
        } catch (const std::exception& ex) {
            std::fprintf(stderr,
                "[direct-registry] NvmeQueueGroup on pci=%s threw: %s\n",
                bp->pci_addr.c_str(), ex.what());
            nvm_ctrl_free(bp->ctrl);
            bp->ctrl = nullptr;
            return false;
        }
    }

    out.backend_private = std::move(bp);
    return true;
}

void LocalNvmeDirectRegistry::Close() {
    std::lock_guard<std::mutex> lock(mtx_);
    close_locked();
}

void LocalNvmeDirectRegistry::close_locked() {
    // Reverse-order teardown so chrdev_create/remove pair stays nested.
    for (auto it = slots_.rbegin(); it != slots_.rend(); ++it) {
        auto& slot = **it;
        auto& bp   = slot.backend_private;
        if (bp) {
            // R5b: drop NvmeQueueGroup BEFORE nvm_ctrl_free.  The
            // queue group's dtor cascades nvm_destroy_group +
            // cudaFree on d_qps[]; those need the live ctrl handle.
            // After .reset() returns, we own ctrl exclusively, so
            // nvm_ctrl_free below still does the unbind+chrdev_remove
            // cascade.
            bp->queue_group.reset();

            if (bp->ctrl != nullptr) {
                nvm_ctrl_free(bp->ctrl);   // owner path: unbind + chrdev_remove
                bp->ctrl = nullptr;
            }
        }
    }
    slots_.clear();
    is_open_ = false;
}

std::size_t LocalNvmeDirectRegistry::device_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return slots_.size();
}

const Device* LocalNvmeDirectRegistry::device_at(std::size_t i) const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (i >= slots_.size()) return nullptr;
    return &slots_[i]->device;
}

const Device* LocalNvmeDirectRegistry::find_by_id(int32_t device_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    for (const auto& s : slots_) {
        if (s->device.device_id == device_id) return &s->device;
    }
    return nullptr;
}

std::vector<const Device*> LocalNvmeDirectRegistry::list() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<const Device*> out;
    out.reserve(slots_.size());
    for (const auto& s : slots_) out.push_back(&s->device);
    return out;
}

} // namespace tutti

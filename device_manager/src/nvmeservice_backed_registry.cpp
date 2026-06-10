/**
 * nvmeservice_backed_registry.cpp -- SERVICE_CLIENT bootstrap path.
 *
 * Talks to a running nvmeservice_daemon via gRPC.  This translation
 * unit pulls in protobuf / gRPC headers, so it must be compiled by
 * g++ (not nvcc) -- protobuf's C++17 inline-static enum traits trip
 * up nvcc's frontend.  The only CUDA-touching part of this file is
 * the libnvm attach (CUDA-runtime calls inside libnvm), which is
 * fine because libnvm exposes a plain C ABI to us.
 */

#include "nvmeservice_backed_registry.h"
#include "nvme_queue_group.h"            // R5b
#include "../../io_engine/include/backend_type.h"

#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <ioctl.h>                  // struct nvm_ioctl_dev for max_data_size

#include "nvmeservice_client.h"   // backends/local/NVMeService/src/

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <utility>
#include <vector>

namespace tutti {

namespace {
inline nvmeservice::NvmeServiceClient::Session* as_session(void* p) {
    return reinterpret_cast<nvmeservice::NvmeServiceClient::Session*>(p);
}
} // namespace

NvmeServiceBackedRegistry::NvmeServiceBackedRegistry(
    std::string                            endpoint,
    std::vector<NvmeServiceBackedRequest>  requests)
    : endpoint_(std::move(endpoint)), requests_(std::move(requests))
{}

NvmeServiceBackedRegistry::~NvmeServiceBackedRegistry() {
    Close();
}

bool NvmeServiceBackedRegistry::Open() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (is_open_) return true;

    client_ = std::make_unique<nvmeservice::NvmeServiceClient>(endpoint_);

    slots_.reserve(requests_.size());

    for (size_t i = 0; i < requests_.size(); ++i) {
        auto slot = std::make_unique<Slot>();
        if (!open_one(requests_[i], (int32_t)i, *slot)) {
            std::fprintf(stderr,
                "[svc-registry] open_one(daemon_dev=%d, cuda=%d) failed; rolling back\n",
                requests_[i].daemon_device_id, requests_[i].cuda_device);
            close_locked();
            return false;
        }
        slots_.push_back(std::move(slot));
    }

    is_open_ = true;
    return true;
}

bool NvmeServiceBackedRegistry::open_one(const NvmeServiceBackedRequest& req,
                                          int32_t                          device_id,
                                          Slot&                            out)
{
    auto sess = client_->connect(req.daemon_device_id,
                                  req.cuda_device,
                                  req.num_queues);
    if (!sess) {
        std::fprintf(stderr, "[svc-registry] Connect rejected\n");
        return false;
    }

    auto bp = std::make_unique<LocalNvmeDevice>();
    bp->device_id        = device_id;
    bp->pci_addr         = sess->pci_addr;
    bp->snvme_dev_path   = sess->snvme_dev_path;
    bp->attach_mode      = LocalNvmeAttachMode::SERVICE_CLIENT;
    bp->display_name     = req.display_name.empty()
                          ? std::string("local_nvme @ ") + sess->pci_addr +
                              " (svc)"
                          : req.display_name;
    bp->namespace_id     = sess->namespace_id;
    bp->bar0_size        = (uint32_t)sess->bar0_size;
    bp->dstrd            = sess->dstrd;
    bp->blk_size         = sess->blk_size;
    bp->blk_size_log     = sess->blk_size_log;
    bp->queue_depth      = sess->queue_depth;
    bp->page_size        = sess->page_size;
    bp->mount_path       = sess->mount_path;

    int rc = nvm_ctrl_attach_client(&bp->ctrl,
                                     bp->snvme_dev_path.c_str(),
                                     bp->bar0_size);
    if (rc != 0 || bp->ctrl == nullptr) {
        std::fprintf(stderr,
            "[svc-registry] nvm_ctrl_attach_client(%s) rc=%d\n",
            bp->snvme_dev_path.c_str(), rc);
        // sess goes out of scope and Disconnects automatically.
        return false;
    }

    // Pull max_user_qid / max_queues_per_group from nvm_ctrl_t (these
    // were populated during attach via the kernel's reflection of
    // NVM_GET_DEV_INFO).
    bp->max_user_qid         = bp->ctrl->max_user_qid;
    bp->max_queues_per_group = bp->ctrl->max_queues_per_group;

    // CTRL.MDTS (max single-IO transfer size) is NOT carried on the
    // gRPC session payload today, and nvm_ctrl_attach_client doesn't
    // surface it on the bare nvm_ctrl_t.  Fetch it directly from the
    // kernel via NVM_GET_DEV_INFO on our own attached fd; the bind
    // is already complete on the daemon side, so this is a single
    // ioctl with no real polling.  Failure is non-fatal: leaves
    // bp->max_data_size at 0 and the caller's caps.max_io_bytes at
    // 0, which downstream code (memory/, block_storage) will treat
    // as "unknown, use a conservative default".
    {
        struct nvm_ioctl_dev info;
        std::memset(&info, 0, sizeof(info));
        int rc_info = nvm_wait_dev_info(bp->ctrl, &info, /*timeout_ms=*/100);
        if (rc_info != 0) {
            std::fprintf(stderr,
                "[svc-registry] nvm_wait_dev_info(daemon_dev=%d) rc=%d; "
                "max_data_size will be 0\n",
                req.daemon_device_id, rc_info);
        } else {
            bp->max_data_size = info.max_data_size;
        }
    }

    // Service-mode bookkeeping: store the session so dtor knows to
    // Disconnect it.
    bp->service_session_opaque = sess.get();
    out.session                = sess.release();   // ownership now in Slot

    // Runtime-visible Device shell.
    out.device.device_id       = device_id;
    out.device.backend_type    = BackendType::LOCAL_NVME;
    out.device.pci_addr        = bp->pci_addr;
    out.device.display_name    = bp->display_name;
    out.device.backend         = nullptr;
    out.device.queues          = nullptr;
    out.device.backend_private = bp.get();

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
    caps.max_queue_count  = (size_t)bp->max_queues_per_group;
    caps.page_size        = bp->page_size;

    // R5b: optional NvmeQueueGroup so upper layers can reach
    // d_qps[] for on-GPU NVMe submit kernels.  In service mode the
    // CLIENT itself runs nvm_create_group + nvm_add_user_queue
    // against its own attach_client fd; the daemon does NOT share
    // any GPU memory or queue handles -- it only handed out the
    // chrdev/bind lease at Connect time.
    if (req.build_queue_group) {
        const uint32_t granted = (uint32_t)(sess->granted_queues > 0
                                            ? sess->granted_queues
                                            : (req.num_queues > 0 ? req.num_queues : 4));
        const uint32_t nq = req.num_user_queues > 0
                            ? std::min<uint32_t>(req.num_user_queues, granted)
                            : granted;
        const uint32_t qd = req.queue_depth > 0
                            ? req.queue_depth
                            : bp->queue_depth;
        if (qd == 0) {
            std::fprintf(stderr,
                "[svc-registry] build_queue_group: q_depth=0 (attach_client "
                "didn't surface q_depth?)\n");
            nvm_ctrl_free_client(bp->ctrl);
            bp->ctrl = nullptr;
            return false;
        }
        if (nq == 0) {
            std::fprintf(stderr,
                "[svc-registry] build_queue_group: num_user_queues=0 "
                "(daemon granted=%u, req.num_user_queues=%u)\n",
                granted, req.num_user_queues);
            nvm_ctrl_free_client(bp->ctrl);
            bp->ctrl = nullptr;
            return false;
        }
        const uint32_t ns_id = req.namespace_id > 0 ? req.namespace_id
                                                    : bp->namespace_id;

        // Synthesise a `disk` struct from the session payload.  The
        // queue group only reads disk.{ns_id,block_size,disk_name,page_size}
        // -- max_data_size is unused inside init.  disk_name is derived
        // from the chrdev path ("/dev/ssnvme0" -> "snvme0n1") since
        // attach_client never surfaced the block-device name.
        struct disk d;
        std::memset(&d, 0, sizeof(d));
        d.page_size  = bp->page_size;
        d.ns_id      = ns_id;
        d.block_size = bp->blk_size;
        const char* tail = bp->snvme_dev_path.c_str();
        if (std::strncmp(tail, "/dev/s", 6) == 0) tail += 6;
        std::snprintf(d.disk_name, sizeof(d.disk_name),
                      "%sn%u", tail, ns_id);

        try {
            bp->queue_group = std::make_shared<NvmeQueueGroup>(
                bp->ctrl,
                d,
                ns_id,
                (uint32_t)req.cuda_device,
                nq,
                qd);
        } catch (const std::exception& ex) {
            std::fprintf(stderr,
                "[svc-registry] NvmeQueueGroup on dev=%d threw: %s\n",
                req.daemon_device_id, ex.what());
            nvm_ctrl_free_client(bp->ctrl);
            bp->ctrl = nullptr;
            return false;
        }
    }

    out.backend_private = std::move(bp);
    return true;
}

void NvmeServiceBackedRegistry::Close() {
    std::lock_guard<std::mutex> lock(mtx_);
    close_locked();
}

void NvmeServiceBackedRegistry::close_locked() {
    // Reverse-order teardown.
    for (auto it = slots_.rbegin(); it != slots_.rend(); ++it) {
        auto& slot = **it;
        auto& bp   = slot.backend_private;

        if (bp) {
            // R5b: drop NvmeQueueGroup BEFORE nvm_ctrl_free_client.
            // The queue group's dtor cascades nvm_destroy_group +
            // cudaFree on d_qps[]; those need the live ctrl handle.
            bp->queue_group.reset();

            if (bp->ctrl != nullptr) {
                // Client path: drop the libnvm attach (no unbind, no
                // chrdev_remove -- that's the daemon's job).
                nvm_ctrl_free_client(bp->ctrl);
                bp->ctrl = nullptr;
            }
        }
        if (slot.session != nullptr) {
            // Session dtor sends Disconnect RPC to the daemon.
            delete as_session(slot.session);
            slot.session = nullptr;
        }
    }
    slots_.clear();
    client_.reset();
    is_open_ = false;
}

std::size_t NvmeServiceBackedRegistry::device_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return slots_.size();
}

const Device* NvmeServiceBackedRegistry::device_at(std::size_t i) const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (i >= slots_.size()) return nullptr;
    return &slots_[i]->device;
}

const Device* NvmeServiceBackedRegistry::find_by_id(int32_t device_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    for (const auto& s : slots_) {
        if (s->device.device_id == device_id) return &s->device;
    }
    return nullptr;
}

std::vector<const Device*> NvmeServiceBackedRegistry::list() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<const Device*> out;
    out.reserve(slots_.size());
    for (const auto& s : slots_) out.push_back(&s->device);
    return out;
}

} // namespace tutti

#ifndef __TUTTI_DEVICE_MANAGER_LOCAL_NVME_DIRECT_REGISTRY_H__
#define __TUTTI_DEVICE_MANAGER_LOCAL_NVME_DIRECT_REGISTRY_H__

/**
 * local_nvme_direct_registry.h -- IDeviceRegistry impl for the
 * IN_PROCESS bootstrap mode (no NVMeService daemon).
 *
 * Layer: Device Manager.
 *
 * Brings up each configured NVMe controller itself via
 * nvm_controller_init_b3() (chrdev_create + cap + bind + probe).
 * Best for single-tenant deployments / smoke testing / micro-benchmarks
 * where the runtime process is the sole owner of every NVMe.
 *
 * Construction takes a vector of (pci_addr, kernel_ioq_cap) pairs.
 * Each pair becomes one Device on success; failures are logged to
 * stderr and the entry is skipped.  Use Open() to drive bring-up
 * AFTER any fork() the application might do; the constructor is
 * cheap and does not touch CUDA or libnvm.
 *
 * Lifetime
 *   - Construction: validates inputs only.
 *   - Open(): opens every controller; throws on the first failure.
 *   - Destruction: closes every controller via nvm_ctrl_free
 *     (cascades unbind + chrdev_remove for direct-owned controllers).
 *
 * Thread safety
 *   - Open() / Close() are NOT thread-safe and must be sequenced by
 *     the caller.  device_count() / device_at() / find_by_id() ARE
 *     safe to call concurrently.
 *
 * Hardware path
 *   - The caller must ensure /dev/snvm_control exists (snvme module
 *     loaded) before calling Open().  Failure surfaces as ENOENT.
 *   - The same NVMe MUST NOT already be bound to the stock nvme
 *     driver (echo it out of nvme0 first).
 */

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "device_registry.h"
#include "local_nvme_device.h"
#include "../../runtime/include/device.h"

namespace tutti {

struct LocalNvmeDirectConfig {
    /// PCI BDF, e.g. "0000:08:00.0".
    std::string pci_addr;

    /// Passed to NVM_SET_KERNEL_IOQ_CAP at bring-up.  Reserves this
    /// many queue pairs for the kernel-side blk-mq path; everything
    /// above goes into the user QID pool the runtime hands out.
    /// 0 means "kernel default" (typically all but a few).
    uint32_t    kernel_ioq_cap = 0;

    /// Optional label that surfaces as Device::display_name.  If
    /// empty the registry synthesises something like
    /// "local_nvme @ 0000:08:00.0".
    std::string display_name;

    /// R5b: build a NvmeQueueGroup around the brought-up ctrl so
    /// upper layers can reach `d_qps[]` for on-GPU NVMe submit
    /// kernels.
    ///
    /// When false, `LocalNvmeDevice::queue_group` stays null and
    /// only the host-side blocking IO path through `nvme_storage`
    /// is available.  Set to true when GPU-direct submit is needed.
    bool        build_queue_group = false;

    /// R5b: primary CUDA device for the queue group's d_qps /
    /// per-queue ring allocations.  Ignored when
    /// `build_queue_group=false`.
    int32_t     cuda_device = 0;

    /// R5b: number of user-queue pairs to allocate for this NVMe.
    /// Each pair gives one extra concurrent IO slot for GPU
    /// kernels.  Must be <= the user QID pool the kernel sized for
    /// us (max_user_qid - start_cq_idx + 1) and <= max_queues_per_group.
    /// Ignored when `build_queue_group=false`.
    uint32_t    num_user_queues = 4;

    /// R5b: SQ/CQ ring depth (power of 2).  0 means "use the
    /// controller's reported max q_depth".
    /// Ignored when `build_queue_group=false`.
    uint32_t    queue_depth = 0;

    /// R5b: namespace id used by user queue commands.  1 is the
    /// usual default for single-namespace devices.
    /// Ignored when `build_queue_group=false`.
    uint32_t    namespace_id = 1;
};

class LocalNvmeDirectRegistry : public IDeviceRegistry {
public:
    explicit LocalNvmeDirectRegistry(std::vector<LocalNvmeDirectConfig> cfgs);
    ~LocalNvmeDirectRegistry() override;

    LocalNvmeDirectRegistry(const LocalNvmeDirectRegistry&)            = delete;
    LocalNvmeDirectRegistry& operator=(const LocalNvmeDirectRegistry&) = delete;

    /// Bring up every configured NVMe.  Must be called AFTER fork()
    /// if the host process forks at all (libnvm pulls in CUDA, which
    /// is fork-hostile).  Returns false on first failure; partial
    /// state is rolled back before returning.
    bool Open();

    /// Close every controller in reverse-open order.  Idempotent.
    void Close();

    // ---- IDeviceRegistry ----------------------------------------

    std::size_t   device_count() const override;
    const Device* device_at(std::size_t i) const override;
    const Device* find_by_id(int32_t device_id) const override;
    std::vector<const Device*> list() const override;

private:
    struct Slot {
        Device                                device;            // exposed view
        std::unique_ptr<LocalNvmeDevice>      backend_private;   // owned payload
    };

    bool open_one(const LocalNvmeDirectConfig& cfg, int32_t device_id, Slot& out);
    void close_locked();

    std::vector<LocalNvmeDirectConfig> cfgs_;
    mutable std::mutex                 mtx_;
    std::vector<std::unique_ptr<Slot>> slots_;
    bool                               is_open_ = false;
};

} // namespace tutti

#endif // __TUTTI_DEVICE_MANAGER_LOCAL_NVME_DIRECT_REGISTRY_H__

#ifndef __TUTTI_DEVICE_MANAGER_NVMESERVICE_BACKED_REGISTRY_H__
#define __TUTTI_DEVICE_MANAGER_NVMESERVICE_BACKED_REGISTRY_H__

/**
 * nvmeservice_backed_registry.h -- IDeviceRegistry impl for the
 * SERVICE_CLIENT bootstrap mode.
 *
 * Layer: Device Manager.
 *
 * Connects to a running NVMeService daemon, calls list_devices() to
 * enumerate the fleet, and Connect()s a session per device the user
 * asks to open.  The daemon owns chrdev / bind; this registry is a
 * thin client.
 *
 * The constructor only takes the daemon endpoint and a list of
 * (device_id, cuda_device, num_queues) requests.  Open() then runs
 * the gRPC dance and produces one Device per successful Connect.
 *
 * Lifetime
 *   - Open(): connects gRPC channel, calls list_devices() once,
 *     Connect()s a session per request, attaches a libnvm controller
 *     via nvm_ctrl_attach_client.  On failure: rolls back (closes
 *     ctrls, disconnects sessions).
 *   - Destruction: closes every controller via nvm_ctrl_free_client
 *     and Disconnect()s each session (drops the daemon-side lease
 *     record cleanly).
 *
 * Thread safety
 *   - Open() / Close() are NOT thread-safe and must be sequenced by
 *     the caller.  device_count() / device_at() / find_by_id() ARE
 *     safe to call concurrently.
 *
 * Hardware path
 *   - The caller MUST start nvmeservice_daemon before constructing
 *     this registry; we don't auto-launch it.
 */

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "device_registry.h"
#include "local_nvme_device.h"
#include "../../runtime/include/device.h"

namespace nvmeservice {
class NvmeServiceClient;
}

namespace tutti {

struct NvmeServiceBackedRequest {
    /// Daemon-local device id (from ListDevices).  In v0.1 this is
    /// usually 0 for single-NVMe deployments.
    int32_t  daemon_device_id = 0;

    /// CUDA device that will drive IO; daemon enforces ACL.
    int32_t  cuda_device      = 0;

    /// Lease quota request; 0 lets the daemon pick its default.
    int32_t  num_queues       = 0;

    /// Optional label that surfaces as Device::display_name.
    std::string display_name;

    /// R5b: build a NvmeQueueGroup around the attach_client'd ctrl
    /// so upper layers can reach `d_qps[]` for on-GPU NVMe submit
    /// kernels.  In service mode this means the *client* itself runs
    /// nvm_create_group + nvm_add_user_queue against its own
    /// attach_client fd; the daemon does NOT share any GPU memory
    /// or queue handles -- it only handed out the chrdev/bind lease.
    ///
    /// When false, `LocalNvmeDevice::queue_group` stays null and only
    /// host-side blocking IO through `nvme_storage` is available.
    bool        build_queue_group = false;

    /// R5b: number of user-queue pairs the client will create on
    /// its own attach_client fd via nvm_add_user_queue.  MUST be
    /// <= the daemon-granted lease quota (`num_queues` above) and
    /// <= max_queues_per_group.  If 0, defaults to `num_queues`
    /// (use the full quota).
    /// Ignored when `build_queue_group=false`.
    uint32_t    num_user_queues = 0;

    /// R5b: SQ/CQ ring depth (power of 2).  0 means "use the
    /// controller's reported max q_depth from attach_client".
    /// Ignored when `build_queue_group=false`.
    uint32_t    queue_depth = 0;

    /// R5b: namespace id used by user queue commands.  0 means
    /// "use whatever attach_client surfaced".
    /// Ignored when `build_queue_group=false`.
    uint32_t    namespace_id = 0;
};

class NvmeServiceBackedRegistry : public IDeviceRegistry {
public:
    NvmeServiceBackedRegistry(std::string                            endpoint,
                              std::vector<NvmeServiceBackedRequest>  requests);
    ~NvmeServiceBackedRegistry() override;

    NvmeServiceBackedRegistry(const NvmeServiceBackedRegistry&)            = delete;
    NvmeServiceBackedRegistry& operator=(const NvmeServiceBackedRegistry&) = delete;

    /// gRPC-connect, ListDevices, Connect each request, attach libnvm.
    /// Returns false on failure.  Partial state is rolled back.
    /// MUST be called after any fork(): NvmeServiceClient internally
    /// stands up a heartbeat thread via std::thread which doesn't
    /// survive fork-without-exec (and CUDA fork-hostility applies if
    /// libnvm bring-up has run).
    bool Open();

    /// Close all sessions + drop ctrl handles.  Idempotent.
    void Close();

    // ---- IDeviceRegistry ----------------------------------------

    std::size_t   device_count() const override;
    const Device* device_at(std::size_t i) const override;
    const Device* find_by_id(int32_t device_id) const override;
    std::vector<const Device*> list() const override;

private:
    struct Slot {
        Device                                device;
        std::unique_ptr<LocalNvmeDevice>      backend_private;
        // Type erased here to keep client.h out of this header.
        // Cast back to nvmeservice::NvmeServiceClient::Session* in .cu/.cpp.
        void*                                 session = nullptr;
    };

    bool open_one(const NvmeServiceBackedRequest& req,
                  int32_t                         device_id,
                  Slot&                           out);
    void close_locked();

    std::string                                endpoint_;
    std::vector<NvmeServiceBackedRequest>      requests_;
    std::unique_ptr<nvmeservice::NvmeServiceClient> client_;
    mutable std::mutex                         mtx_;
    std::vector<std::unique_ptr<Slot>>         slots_;
    bool                                       is_open_ = false;
};

} // namespace tutti

#endif // __TUTTI_DEVICE_MANAGER_NVMESERVICE_BACKED_REGISTRY_H__

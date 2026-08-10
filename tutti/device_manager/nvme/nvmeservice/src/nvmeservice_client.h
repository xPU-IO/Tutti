#ifndef __NVMESERVICE_CLIENT_H__
#define __NVMESERVICE_CLIENT_H__

/**
 * nvmeservice_client.h -- thin gRPC session wrapper.
 *
 * Post-L1-Commit-4b the client library hands the caller a *session*
 * (allocation_id + metadata + heartbeat thread).  Caller is
 * responsible for the libnvm bring-up:
 *
 *   auto sess = client.connect(device_id=0, cuda_device=0, num_queues=4);
 *   if (!sess) ...;
 *
 *   nvm_ctrl_t* ctrl = nullptr;
 *   nvm_ctrl_attach_client(&ctrl, sess->snvme_dev_path.c_str(),
 *                          (uint32_t)sess->bar0_size);
 *   uint32_t group_id = 0, max_q = 0;
 *   nvm_create_group(ctrl, &group_id, &max_q);
 *   // map rings, add user queues up to sess->granted_queues, drive IO
 *   ...
 *   nvm_destroy_group(ctrl, group_id);
 *   nvm_ctrl_free_client(ctrl);
 *   sess.reset();   // -> Disconnect RPC + heartbeat thread joins
 *
 * Heartbeat is maintained by an internal thread as long as any
 * Session is live; interval/timeout come from the daemon's
 * ConnectResponse.
 *
 * Canonical callers use list_accelerators(), list_nvme_resources(), and
 * acquire_nvme_slices(). The returned Allocation owns one logical Release;
 * its slices contain the actual owner-returned chrdev, block, and view paths.
 */

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "nvmeservice.grpc.pb.h"

namespace nvmeservice {

// Compatibility ACL row from DeviceInfo.allowed_gpus[]. mount_path is the
// accelerator-view symlink the daemon published; empty means the mount or
// publication was not ready.
struct ClientAllowedGpu {
    int32_t     cuda_device = -1;
    std::string mount_path;
};

// Mirror of DeviceInfo for callers that want to enumerate.
struct ClientDeviceInfo {
    int32_t     device_id            = -1;
    std::string pci_addr;
    std::string snvme_dev_path;
    uint32_t    namespace_id         = 0;
    uint32_t    page_size            = 0;
    uint32_t    blk_size             = 0;
    uint32_t    blk_size_log         = 0;
    uint32_t    queue_depth          = 0;
    uint32_t    dstrd                = 0;
    uint64_t    bar0_size            = 0;
    uint32_t    max_user_qid         = 0;
    uint32_t    max_queues_per_group = 0;
    std::vector<ClientAllowedGpu> allowed_gpus;
};

struct ClientAcceleratorInfo {
    int32_t accel_id = -1;
    std::string view_root;
};

// Canonical ListNvmeResources snapshot. Paths are owner/RPC facts, not
// client-side path templates.
struct ClientNvmeResource {
    int32_t device_id = -1;
    std::string pci_bdf;
    int32_t chrdev_minor = -1;
    std::string chrdev_path;
    std::string block_path;
    std::string backing_mount_path;
    uint32_t namespace_id = 0;
    uint32_t page_size = 0;
    uint32_t logical_block_size = 0;
    uint32_t logical_block_size_log = 0;
    uint32_t queue_depth = 0;
    uint32_t dstrd = 0;
    uint64_t bar0_size = 0;
    uint64_t max_data_size = 0;
    uint32_t max_user_qid = 0;
    uint32_t kernel_io_qps = 0;
    uint32_t controller_queue_capacity = 0;
    uint32_t max_queues_per_group = 0;
    uint32_t reserved_queues = 0;
    uint32_t available_queues = 0;
    std::vector<int32_t> allowed_accel_ids;
    bool available = false;
    std::string diagnostic;
    uint32_t heartbeat_interval_sec = 0;
    uint32_t lease_timeout_sec = 0;
};

// Canonical allocation slice. One Allocation may contain multiple ordered
// slices for striped selection and releases them together.
struct ClientNvmeSlice {
    int32_t device_id = -1;
    int32_t accel_id = -1;
    std::string pci_bdf;
    int32_t chrdev_minor = -1;
    std::string chrdev_path;
    std::string block_path;
    std::string backing_mount_path;
    std::string view_path;
    uint32_t namespace_id = 0;
    uint32_t page_size = 0;
    uint32_t logical_block_size = 0;
    uint32_t logical_block_size_log = 0;
    uint32_t queue_depth = 0;
    uint32_t dstrd = 0;
    uint64_t bar0_size = 0;
    uint64_t max_data_size = 0;
    uint32_t controller_queue_capacity = 0;
    uint32_t granted_queues = 0;
    uint32_t max_queues_per_group = 0;
    std::vector<int32_t> allowed_accel_ids;
    uint32_t heartbeat_interval_sec = 0;
    uint32_t lease_timeout_sec = 0;
};

// Selection policy passed to AcquireNvmeSlices.
enum class ClientSelectionMode { Allowed, Explicit, Striped };

class NvmeServiceClient {
public:
    struct Allocation {
        // Move-only RAII handle for one logical daemon allocation.
        std::string allocation_id;
        std::vector<ClientNvmeSlice> slices;

        Allocation() = default;
        ~Allocation();
        Allocation(const Allocation&) = delete;
        Allocation& operator=(const Allocation&) = delete;

    private:
        friend class NvmeServiceClient;
        NvmeServiceClient* owner = nullptr;
    };

    /**
     * An active session with the daemon.  Pure metadata + lease
     * bookkeeping; the libnvm controller / queue group / DATA buffers
     * are NOT owned here -- the caller drives those itself.
     *
     * Destructor sends a Disconnect RPC and stops the heartbeat
     * thread (when this is the last live session).
     */
    struct Session {
        std::string allocation_id;

        // Echo of the request so callers can pass it to libnvm.
        int32_t     device_id     = -1;
        int32_t     cuda_device   = -1;

        // Daemon's per-client grant. The canonical allocator reserves this
        // budget before returning; the kernel remains authoritative when the
        // client creates the actual queue group.
        int32_t     granted_queues = 0;

        // Device metadata for nvm_ctrl_attach_client + IO submit.
        std::string pci_addr;
        std::string snvme_dev_path;
        uint64_t    bar0_size     = 0;
        uint32_t    dstrd         = 0;
        uint32_t    namespace_id  = 0;
        uint32_t    page_size     = 0;
        uint32_t    blk_size      = 0;
        uint32_t    blk_size_log  = 0;
        uint32_t    queue_depth   = 0;

        // CTRL.MDTS in bytes (per-IO transfer cap).  The daemon read
        // this from NVM_GET_DEV_INFO; clients can't (attach_client
        // skips GET_DEV_INFO), so it arrives via RPC.
        uint64_t    max_data_size = 0;

        // Compatibility view path for (cuda_device, NVMe). Empty if the mount
        // was not ready or publication failed.
        std::string mount_path;

        // Lease parameters (informational; heartbeat thread uses these).
        uint32_t    heartbeat_interval_sec = 0;
        uint32_t    lease_timeout_sec      = 0;

        uint32_t    client_pid    = 0;

        Session() = default;
        ~Session();

        Session(const Session&)            = delete;
        Session& operator=(const Session&) = delete;

    private:
        friend class NvmeServiceClient;
        NvmeServiceClient* owner = nullptr;
    };

    explicit NvmeServiceClient(const std::string& endpoint);
    ~NvmeServiceClient();

    NvmeServiceClient(const NvmeServiceClient&)            = delete;
    NvmeServiceClient& operator=(const NvmeServiceClient&) = delete;

    std::vector<ClientDeviceInfo> list_devices();
    std::vector<ClientAcceleratorInfo> list_accelerators();
    std::vector<ClientNvmeResource> list_nvme_resources();

    std::unique_ptr<Allocation> acquire_nvme_slices(
        int32_t accel_id,
        ClientSelectionMode selection,
        const std::vector<int32_t>& device_ids,
        int32_t queues_per_controller);

    /**
     * Open a session.  num_queues == 0 -> use daemon default
     * (capped by config).  Returns nullptr on failure; error details
     * go to stderr.
     */
    std::unique_ptr<Session> connect(int32_t device_id,
                                      int32_t cuda_device,
                                      int32_t num_queues);

private:
    friend struct Session;

    void release_session(Session* sess);
    void release_allocation(Allocation* allocation);

    void ensure_heartbeat_started();
    void stop_heartbeat();
    void heartbeat_loop();

    std::string                                          endpoint_;
    std::shared_ptr<grpc::Channel>                       channel_;
    std::unique_ptr<NvmeService::Stub>                   stub_;

    struct LiveSession {
        std::string allocation_id;
        uint32_t    heartbeat_interval_sec = 10;
    };
    std::mutex                                           live_mtx_;
    std::unordered_map<std::string, LiveSession>         live_sessions_;

    std::thread                                          hb_thread_;
    std::atomic<bool>                                    hb_running_{false};
};

} // namespace nvmeservice

#endif // __NVMESERVICE_CLIENT_H__

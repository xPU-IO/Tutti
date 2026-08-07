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

// One ACL row from DeviceInfo.allowed_gpus[].  mount_path is the
// GPU-view symlink the daemon published; empty == mount not ready or
// publication failed.
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

class NvmeServiceClient {
public:
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

        // Daemon's recommendation: caller must not exceed this when
        // calling nvm_add_user_queue.  Authoritative kernel cap is
        // also available via max_queues_per_group on ListDevices
        // output.  This is policy only -- daemon does not maintain
        // a quota ledger.
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

        // GPU-view symlink path the daemon published for
        // (cuda_device, NVMe).  Empty if the mount was not ready or
        // publication failed.
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

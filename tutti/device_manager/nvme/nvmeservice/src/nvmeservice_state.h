#ifndef __NVMESERVICE_STATE_H__
#define __NVMESERVICE_STATE_H__

/**
 * nvmeservice_state.h -- daemon-side service state.
 *
 * Post-L1-Commit-4b (no per-GPU queue accounting in daemon)
 * --------------------------------------------------------
 *
 * The daemon owns the chrdev/bind for each NVMe and keeps the
 * controller alive for clients.  It does NOT track any per-GPU
 * queue ledger -- the kernel's user QID pool is the single source
 * of truth, and the per-fd queue group model means each client
 * brings up its own queues independently.
 *
 * What ServiceState tracks:
 *
 *   * One nvm_ctrl_t* per configured NVMe (owner-side B3 bring-up;
 *     freed at daemon shutdown).
 *   * The kernel-reported metadata (max_user_qid, max_queues_per_group,
 *     queue_depth, dstrd, bar0_size) -- read once from NVM_GET_DEV_INFO.
 *   * Per-GPU mount-path symlinks published after the backing filesystems
 *     are mounted.
 *   * Per-allocation lease (PID + starttime + last_heartbeat) for the
 *     reaper to clean up dead clients' bookkeeping.  No quota refund
 *     is needed -- the kernel's snvm_dev_release fd-cascade reclaims
 *     the actual queues when the dead client's fd closes.
 *
 * Thread safety: all public operations are serialised by an internal
 * mutex.  No protobuf dependency -- the server layer translates
 * to/from proto.
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nvmeservice_config.h"

// libnvm types: nvm_ctrl_t is a typedef of an anonymous struct in
// nvm_types.h, so a forward decl wouldn't match.  Pull the header.
#include <nvm_types.h>

namespace nvmeservice {

// -----------------------------------------------------------------
// Snapshots returned to the gRPC layer (proto-free DTOs)
// -----------------------------------------------------------------

// One row per allowed GPU on a device, telling clients where the
// per-GPU view directory lives.  Empty mount_path means symlink
// publication failed after the backing filesystem was mounted.
struct AllowedGpuView {
    int32_t     cuda_device = -1;
    std::string mount_path;
};

struct DeviceSnapshot {
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

    // ACL: which cuda_device values may Connect to this NVMe.
    // Empty == any GPU in the daemon's gpus[] list.  Mirrors
    // nvmes[].allowed_gpus from YAML, with mount_paths populated.
    std::vector<AllowedGpuView> allowed_gpus;
};

// What the client gets after Connect.
struct ConnectGrant {
    std::string allocation_id;
    int32_t     device_id              = -1;
    std::string pci_addr;
    std::string snvme_dev_path;
    uint64_t    bar0_size              = 0;
    uint32_t    dstrd                  = 0;

    // Daemon's recommendation.  The client must not exceed this when
    // calling nvm_add_user_queue.  This is policy only -- the kernel
    // separately enforces NVM_MAX_QUEUES_PER_GROUP per client fd.
    int32_t     granted_queues         = 0;

    uint32_t    namespace_id           = 0;
    uint32_t    page_size              = 0;
    uint32_t    blk_size               = 0;
    uint32_t    blk_size_log           = 0;
    uint32_t    queue_depth            = 0;
    std::string mount_path;
    uint32_t    heartbeat_interval_sec = 0;
    uint32_t    lease_timeout_sec      = 0;
    uint64_t    max_data_size          = 0;   // CTRL.MDTS in bytes
};

// -----------------------------------------------------------------
// Internal allocation record (pure lease bookkeeping; no quota)
// -----------------------------------------------------------------

struct Allocation {
    std::string                              allocation_id;
    int32_t                                  device_id            = -1;
    int32_t                                  cuda_device          = -1;
    int32_t                                  granted_queues       = 0;
    uint32_t                                 client_pid           = 0;
    uint64_t                                 client_pid_starttime = 0;
    std::chrono::steady_clock::time_point    last_heartbeat;
};

// Per-NVMe device state.  No queue ledger; the kernel owns that.
struct DeviceState {
    int32_t                  device_id            = -1;
    std::string              pci_addr;
    std::string              snvme_dev_path;
    std::string              mount_path;          // real NVMe mount
    uint64_t                 bar0_size            = 0;
    uint32_t                 dstrd                = 0;
    uint32_t                 namespace_id         = 0;
    uint32_t                 page_size            = 0;
    uint32_t                 blk_size             = 0;
    uint32_t                 blk_size_log         = 0;
    uint32_t                 queue_depth          = 0;
    uint32_t                 max_user_qid         = 0;
    uint32_t                 max_queues_per_group = 0;
    size_t                   max_data_size        = 0;   // CTRL.MDTS in bytes

    // Owner-side libnvm handle.  Held for the daemon's lifetime to
    // keep the chrdev / bind alive.  Freed via nvm_ctrl_free in
    // ~ServiceState().
    nvm_ctrl_t*              ctrl                 = nullptr;

    // ACL: empty == any GPU in cfg_.gpus.
    std::unordered_set<int>  allowed_gpus;

    // cuda_device -> per-GPU symlink path (e.g. "/mnt/gpu0/ssnvme0").
    // Populated during publish_gpu_views(); a missing entry means
    // publication failed or was skipped because the mount was not ready.
    std::unordered_map<int, std::string> gpu_view_paths;

    // Paths managed by publish_gpu_views(), removed by
    // unpublish_gpu_views() (and defensively by the destructor).
    std::vector<std::string> created_symlinks;
    std::vector<std::string> created_nvme_subdirs;
};

// -----------------------------------------------------------------
// Public service state
// -----------------------------------------------------------------

class ServiceState {
public:
    /**
     * Initialise all devices from config: bring up each libnvm controller
     * (owner role: chrdev_create + cap + bind + probe).  GPU filesystem
     * views are deliberately NOT created here: the block devices only exist
     * after bring-up, so the daemon must mount them first and then call
     * publish_gpu_views().  Throws std::runtime_error on bring-up failure.
     */
    explicit ServiceState(const ServiceConfig& cfg);
    ~ServiceState();

    ServiceState(const ServiceState&)            = delete;
    ServiceState& operator=(const ServiceState&) = delete;

    void start_reaper();
    void stop_reaper();

    /**
     * Publish the per-GPU filesystem views for one ready device.
     *
     * The caller must first verify that DeviceState::mount_path is a mounted
     * filesystem.  Keeping that mount decision in the daemon prevents this
     * class from creating GPU<n> directories on the host filesystem and then
     * having a later mount hide them.
     *
     * Returns false for an invalid device id or if any directory/symlink
     * could not be installed.  Successful views remain available when a
     * different GPU view fails.
     */
    bool publish_gpu_views(int32_t device_id);

    // Remove published GPU-view symlinks and empty subdirectories.
    // Idempotent; the production daemon calls this before unmounting.
    void unpublish_gpu_views();

    // --- Query ---
    std::vector<DeviceSnapshot> list_devices() const;

    // --- Session lifecycle ---

    struct ConnectResult {
        bool         success = false;
        std::string  error;
        ConnectGrant grant;
    };

    /**
     * Validate (device_id, cuda_device) against allowed_gpus, clamp
     * num_queues to per-client policy + kernel max_queues_per_group,
     * record the lease, return the grant.  No quota account is
     * decremented -- the kernel's user QID pool is the only ledger.
     */
    ConnectResult connect(int32_t device_id,
                          int32_t cuda_device,
                          int32_t num_queues,
                          uint32_t client_pid);

    bool disconnect(const std::string& allocation_id,
                    uint32_t client_pid,
                    std::string* error);

    bool update_heartbeat(const std::string& allocation_id, std::string* error);

    bool has_allocation(const std::string& allocation_id) const;

private:
    // --- Init helpers ---
    void init_device(const NvmeEntry& nvme, int32_t device_id);

    bool install_gpu_symlinks(DeviceState& dev);
    void remove_gpu_symlinks(DeviceState& dev);

    // --- Reaper ---
    void reaper_loop();
    bool is_pid_dead(uint32_t pid, uint64_t recorded_starttime) const;

    // --- Helpers ---
    static std::string generate_allocation_id();
    static std::optional<uint64_t> read_pid_starttime(uint32_t pid);

    // GPU id -> mount path (drawn from cfg_.gpus at construction;
    // immutable thereafter).
    const std::string* gpu_mount_for(int gpu_id) const;

    // --- Data members ---
    ServiceConfig                                cfg_;
    std::vector<DeviceState>                     devices_;
    std::unordered_map<std::string, Allocation>  allocations_;
    mutable std::mutex                           state_mtx_;

    std::thread                                  reaper_thread_;
    std::atomic<bool>                            reaper_running_{false};
};

} // namespace nvmeservice

#endif // __NVMESERVICE_STATE_H__

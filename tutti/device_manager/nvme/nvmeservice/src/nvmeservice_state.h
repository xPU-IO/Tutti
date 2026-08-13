#ifndef __NVMESERVICE_STATE_H__
#define __NVMESERVICE_STATE_H__

/**
 * nvmeservice_state.h -- daemon-side resource, allocation, and lease state.
 *
 * ServiceState owns one libnvm controller per configured NVMe. Owner bring-up
 * creates/binds/probes the controller and returns the actual chrdev minor/path
 * and namespace disk name/block path. The daemon validates those facts against
 * the configured PCI BDF before a resource can become available.
 *
 * What ServiceState tracks:
 *
 *   * Canonical accelerator and explicit NVMe resource identities.
 *   * Owner-reported device paths plus controller/namespace metadata.
 *   * Published per-accelerator view paths under configured view roots.
 *   * Per-controller reserved queue totals derived from successful logical
 *     allocations.
 *   * Per-allocation slice reservations, PID/starttime, and heartbeat time.
 *
 * The reservation ledger is admission control. The kernel user QID pool stays
 * authoritative when a client attaches, creates a group, and adds real queues.
 * Selection, ACL/view checks, all striped reservations, allocation insertion,
 * release, and reaping are serialized by state_mtx_. The state layer remains
 * protobuf-free; the server translates snapshots and grants to wire messages.
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nvmeservice_config.h"

// nvm_ctrl_t is a typedef of an anonymous struct in nvm_types.h, so a forward
// declaration would not match the libnvm ABI.
#include <nvm_types.h>

namespace nvmeservice {

struct AcceleratorSnapshot {
    // Configured accelerator identity and its view root.
    int32_t accel_id = -1;
    std::string view_root;
};

struct NvmeResourceSnapshot {
    // Static identity and owner bring-up facts returned by ListNvmeResources.
    int32_t device_id = -1;
    std::string pci_bdf;
    int32_t chrdev_minor = -1;
    std::string chrdev_path;
    std::string disk_name;
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
    // Actual kernel I/O QPs, excluding QID 0's admin queue.
    uint32_t kernel_io_qps = 0;
    // User-QID pool capacity reported at bring-up, not a YAML-derived limit.
    uint32_t controller_queue_capacity = 0;
    uint32_t max_queues_per_group = 0;
    uint32_t reserved_queues = 0;
    // Derived as capacity - reserved while holding the state mutex.
    uint32_t available_queues = 0;
    // Expanded and sorted ACL; empty input is never exposed internally.
    std::vector<int32_t> allowed_accel_ids;
    // Published accelerator ID -> view path. Missing entries are diagnostics.
    std::unordered_map<int32_t, std::string> view_paths;
    bool available = false;
    std::string diagnostic;
    uint32_t heartbeat_interval_sec = 0;
    uint32_t lease_timeout_sec = 0;
};

struct BringupMetadata {
    // Facts used for fail-closed association checks after owner bring-up.
    std::string configured_pci_bdf;
    std::string observed_pci_bdf;
    int32_t chrdev_minor = -1;
    std::string chrdev_path;
    std::string disk_name;
    std::string block_path;
};

struct BringupValidationProbe {
    // Injectable filesystem/sysfs probes keep metadata validation hardware-free.
    std::function<bool(const std::string&)> is_character_device;
    std::function<bool(const std::string&)> is_block_device;
    std::function<std::optional<uint32_t>(const std::string&)> character_minor;
    std::function<bool(const std::string&, const std::string&)> block_matches_pci_bdf;
};

bool validate_bringup_metadata(const BringupMetadata& metadata,
                               const BringupValidationProbe& probe,
                               std::string* error = nullptr);

enum class SelectionMode {
    // Selection policy used by the canonical AcquireNvmeSlices RPC.
    Allowed,
    Explicit,
    Striped,
};

struct AcquireRequest {
    // device_ids is empty for allowed, one item for explicit, and ordered
    // unique items (two or more) for striped selection.
    int32_t accel_id = -1;
    SelectionMode selection = SelectionMode::Allowed;
    std::vector<int32_t> device_ids;
    int32_t queues_per_controller = 0;
    uint32_t client_pid = 0;
};

struct NvmeSliceGrant {
    // One atomically reserved slice returned as part of an allocation.
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

struct AllocationGrant {
    // A single logical allocation may contain multiple striped slices.
    std::string allocation_id;
    std::vector<NvmeSliceGrant> slices;
};

struct AcquireResult {
    bool success = false;
    std::string error;
    AllocationGrant grant;
};

struct ReleaseResult {
    bool success = false;
    bool already_released = false;
    std::string error;
};

struct ConnectGrant {
    // Legacy Connect adapter DTO. New callers use NvmeSliceGrant instead.
    std::string allocation_id;
    int32_t device_id = -1;
    std::string pci_addr;
    std::string snvme_dev_path;
    uint64_t bar0_size = 0;
    uint32_t dstrd = 0;
    int32_t granted_queues = 0;
    uint32_t namespace_id = 0;
    uint32_t page_size = 0;
    uint32_t blk_size = 0;
    uint32_t blk_size_log = 0;
    uint32_t queue_depth = 0;
    uint64_t max_data_size = 0;
    std::string mount_path;
    uint32_t heartbeat_interval_sec = 0;
    uint32_t lease_timeout_sec = 0;
};

class ServiceState {
public:
    using ProcessDeadProbe = std::function<bool(uint32_t, uint64_t)>;
    using AllocationIdGenerator = std::function<std::string()>;

    /**
     * Production constructor. Brings up every configured controller in the
     * owner role. Views are published separately after backing mounts exist.
     * Mixed namespace block sizes leave the affected resources unavailable.
     */
    explicit ServiceState(const ServiceConfig& config);

    /**
     * Hardware-free constructor for allocator/parser contract tests. Injected
     * snapshots still use the production selection, ledger, release, and
     * reaper logic.
     */
    ServiceState(const ServiceConfig& config,
                 std::vector<NvmeResourceSnapshot> resources,
                 ProcessDeadProbe process_dead_probe = {},
                 AllocationIdGenerator allocation_id_generator = {});
    ~ServiceState();

    ServiceState(const ServiceState&) = delete;
    ServiceState& operator=(const ServiceState&) = delete;

    // Snapshots are ordered by explicit ID, independent of YAML/vector order.
    std::vector<AcceleratorSnapshot> list_accelerators() const;
    std::vector<NvmeResourceSnapshot> list_nvme_resources() const;

    /**
     * Publish view symlinks for one resource whose backing mount is ready.
     * Successful views remain usable if another accelerator view fails; the
     * resource diagnostic records incomplete publication.
     */
    bool publish_accelerator_views(int32_t device_id);
    bool publish_gpu_views(int32_t device_id) {
        return publish_accelerator_views(device_id);
    }
    void unpublish_accelerator_views();
    void unpublish_gpu_views() { unpublish_accelerator_views(); }

    // Atomically validate, reserve, and create one logical allocation. Allowed
    // selection is ordered by device_id; striped selection preserves request
    // order and never exposes partial reservations.
    AcquireResult acquire(const AcquireRequest& request);
    // Idempotently refund every reservation recorded by the allocation.
    ReleaseResult release(const std::string& allocation_id);

    struct ConnectResult {
        bool success = false;
        std::string error;
        ConnectGrant grant;
    };

    // Legacy Connect enters the same canonical allocator as one explicit slice.
    ConnectResult connect(int32_t device_id, int32_t cuda_device,
                          int32_t num_queues, uint32_t client_pid);
    bool disconnect(const std::string& allocation_id, uint32_t client_pid,
                    std::string* error);

    // Heartbeat refreshes the allocation lease; the reaper owns timeout cleanup.
    bool update_heartbeat(const std::string& allocation_id,
                          std::string* error);
    bool has_allocation(const std::string& allocation_id) const;

    void start_reaper();
    void stop_reaper();
    size_t reap_once(std::chrono::steady_clock::time_point now);

private:
    struct DeviceState {
        // Per-controller owner handle plus canonical snapshot and published paths.
        NvmeResourceSnapshot resource;
        // Owner handle lives until daemon shutdown; clients open independent fds.
        nvm_ctrl_t* ctrl = nullptr;
        // True only after BDF/chrdev/block association validation succeeds.
        bool control_ready = false;
        // Only paths created by this instance are removed during cleanup.
        std::vector<std::string> created_symlinks;
        std::vector<std::string> created_backing_subdirs;
    };

    struct Reservation {
        // Queue budget refunded as one unit when its allocation is released.
        int32_t device_id = -1;
        uint32_t queues = 0;
    };

    struct Allocation {
        // Lease record for one logical allocation, including all striped slices.
        std::string allocation_id;
        int32_t accel_id = -1;
        std::vector<Reservation> reservations;
        uint32_t client_pid = 0;
        uint64_t client_pid_starttime = 0;
        std::chrono::steady_clock::time_point last_heartbeat;
    };

    void initialize_hardware_resources();
    void add_hardware_resource(const NvmeEntry& entry);
    void rebuild_device_index();
    DeviceState* find_device_locked(int32_t device_id);
    const DeviceState* find_device_locked(int32_t device_id) const;
    const AcceleratorEntry* find_accelerator(int32_t accel_id) const;
    bool install_accelerator_views_locked(DeviceState& device);
    void remove_accelerator_views_locked(DeviceState& device);
    bool can_allocate_locked(const DeviceState& device, int32_t accel_id,
                             uint32_t grant, std::string* error) const;
    uint32_t compute_grant_locked(const DeviceState& device,
                                  int32_t requested) const;
    NvmeSliceGrant make_slice_locked(const DeviceState& device,
                                     int32_t accel_id,
                                     uint32_t grant) const;
    ReleaseResult release_locked(const std::string& allocation_id);
    static std::string generate_allocation_id();
    static std::optional<uint64_t> read_pid_starttime(uint32_t pid);
    bool default_process_dead(uint32_t pid, uint64_t starttime) const;
    void reaper_loop();

    ServiceConfig config_;
    std::vector<DeviceState> devices_;
    std::unordered_map<int32_t, size_t> device_index_;
    // Both canonical and compatibility RPCs share this single allocation table.
    std::unordered_map<std::string, Allocation> allocations_;
    // Retains deterministic idempotent Release semantics without refunding twice.
    std::unordered_set<std::string> released_allocation_ids_;
    ProcessDeadProbe process_dead_probe_;
    AllocationIdGenerator allocation_id_generator_;
    mutable std::mutex state_mtx_;
    std::thread reaper_thread_;
    std::atomic<bool> reaper_running_{false};
};

} // namespace nvmeservice

#endif // __NVMESERVICE_STATE_H__

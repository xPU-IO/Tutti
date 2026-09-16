#ifndef __NVMESERVICE_MOUNT_MANAGER_H__
#define __NVMESERVICE_MOUNT_MANAGER_H__

/**
 * mount_manager.h -- Round 17 S1: auto-mount/auto-unmount with EBUSY
 * diagnostics.
 *
 * Lifecycle:
 *   - mount_all(): called at daemon startup.  For every NVMe entry with
 *     auto_mount=true, mount(2) the block device at mount_path (ext4).
 *     Failures (already mounted, fs dirty, device absent) are logged
 *     and the device continues WITHOUT a mount; it is NOT recorded as
 *     daemon-owned.  Devices already mounted before the daemon started
 *     are also NOT recorded as daemon-owned (the daemon never umounts
 *     a mount it did not create).
 *
 *   - unmount_all(): called at daemon shutdown.  For every device the
 *     daemon recorded as "I mounted this", umount2(2).  On EBUSY,
 *     scan /proc to report holders (PID + comm + fd/maps/cwd), retry
 *     up to UnmountRetryConfig.max times with interval_ms sleep.
 *
 *   - force_exit_requested(): returns true if a second SIGTERM/SIGINT
 *     arrived during the retry loop; the caller should exit immediately
 *     (leaving mounts in place) and report which mounts are still up.
 *
 * Signal-safety: unmount_all() checks force_exit_requested() at the
 * top of each retry iteration; the second signal handler just flips
 * an atomic flag (async-signal-safe).
 */

#include "nvmeservice_config.h"

#include <atomic>
#include <string>
#include <vector>

namespace nvmeservice {

// One row of "who is holding this mount busy".
struct MountHolder {
    uint32_t    pid = 0;
    std::string comm;
    std::string holder_type;  // "fd", "maps", or "cwd"
    std::string detail;       // e.g. fd=N, or maps line
};

struct MountResult {
    bool        mounted_by_daemon = false;  // true => daemon owns it
    bool        already_mounted   = false;  // true => pre-existing mount
    std::string error;                       // non-empty on failure
    std::string block_device;                // e.g. "/dev/snvme0n1"
};

class MountManager {
public:
    explicit MountManager(const UnmountRetryConfig& retry_cfg);

    // Mount a single device.  block_device e.g. "/dev/snvme0n1".
    // mount_path e.g. "/mnt/nvme1".  Records ownership if successful.
    // Returns MountResult describing the outcome.
    MountResult mount_one(const std::string& block_device,
                          const std::string& mount_path);

    // Attempt to unmount every device recorded as daemon-owned.
    // Returns the number of mounts still busy (0 = all clean).
    // If force_exit_requested() becomes true mid-loop, stops retrying
    // and returns the remaining count immediately.
    int unmount_all();

    // Called from the second-signal handler (async-signal-safe:
    // just flips an atomic).
    void request_force_exit() { force_exit_.store(1, std::memory_order_relaxed); }
    bool force_exit_requested() const { return force_exit_.load(std::memory_order_relaxed); }

    // Diagnostic: scan /proc for processes holding `mount_path` busy.
    // Used internally by unmount_all() but exposed for testing.
    static std::vector<MountHolder> scan_holders(const std::string& mount_path);

    // Check if a path is already a mount point (uses /proc/self/mountinfo).
    static bool is_mounted(const std::string& mount_path);

private:
    struct OwnedMount {
        std::string block_device;
        std::string mount_path;
    };

    UnmountRetryConfig          retry_cfg_;
    std::vector<OwnedMount>     owned_mounts_;
    std::atomic<int>            force_exit_{0};

    // Internal: try umount2 once.  Returns 0 on success, errno on failure.
    static int try_umount_(const std::string& mount_path);

    // Internal: report holders to stdout/stderr.
    static void report_holders_(const std::string& mount_path,
                                const std::vector<MountHolder>& holders);
};

} // namespace nvmeservice

#endif // __NVMESERVICE_MOUNT_MANAGER_H__

#ifndef __NVMESERVICE_CONFIG_H__
#define __NVMESERVICE_CONFIG_H__

/**
 * nvmeservice_config.h -- canonical daemon configuration and diagnostics.
 *
 * Accelerator and NVMe identities are explicit: accel_id is the compiled
 * backend ordinal and device_id identifies an NVMe resource. Array order is
 * not identity and cannot be used to construct device paths.
 *
 * Canonical schema:
 *
 *   grpc:
 *     endpoint: "127.0.0.1:50051"
 *
 *   accelerators:
 *     - accel_id: 0
 *       view_root: "/mnt/snvme/gpu0"
 *
 *   nvmes:
 *     - device_id: 0
 *       pci_addr: "0000:41:00.0"
 *       backing_mount_path: "/mnt/snvme/nvme1"
 *       namespace_id: 1
 *       kernel_ioq_cap: 32
 *       allowed_accel_ids: [0, 1]
 *       auto_mount: true
 *
 *   queue_pool:
 *     default_per_client: 4
 *     max_per_client: 32
 *
 *   lease:
 *     heartbeat_interval_sec: 10
 *     timeout_sec: 30
 *
 * The parser accepts one completely legacy-only schema during the transition:
 * gpus[].id/mount_path and nvmes[].mount_path/allowed_gpus are normalized at
 * the parser boundary. Canonical and legacy fields cannot be mixed. Legacy
 * deprecation messages are returned through ConfigDiagnostics so tests and
 * embedders do not depend on stderr.
 *
 * The kernel-reported user QID range supplies controller capacity. queue_pool
 * is only the per-client admission policy; ServiceState keeps the runtime
 * reservation ledger after owner bring-up.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nvmeservice {

// The striped data path currently assumes 4 KiB namespace logical blocks.
// The controller reports the value for each namespace during bring-up; this
// constant is only used for the daemon's operator-facing warning.  It is not
// a controller-wide format setting, filesystem block size, or physical block
// size.  Values are bytes (1 << Identify Namespace LBA shift).
constexpr uint32_t kExpectedNvmeBlockSize = 4096;

struct GrpcConfig {
    std::string endpoint = "127.0.0.1:50051";
};

struct AcceleratorEntry {
    // Compiled backend ordinal; never an array index inferred by the daemon.
    int32_t     accel_id = -1;
    // Root under which this accelerator's published view paths live.
    std::string view_root;
};

struct NvmeEntry {
    // Explicit resource identity; array order is not identity.
    int32_t              device_id = -1;
    // Canonical PCI BDF after parser normalization.
    std::string          pci_addr;
    // Real filesystem mount target for the owner-returned block device.
    std::string          backing_mount_path;
    uint32_t             namespace_id = 1;
    // NVM_SET_KERNEL_IOQ_CAP pre-bind hint in queue-pair units. Zero leaves the
    // cap to the kernel. Set it when controller MSI-X capacity is smaller than
    // the host CPU count so the user-share pool remains usable.
    uint32_t             kernel_ioq_cap = 0;
    // Accelerator ACL used both for acquisition checks and view publication.
    // Empty input is expanded to all configured accelerator IDs before the
    // internal model or RPC snapshot is exposed.
    std::vector<int32_t> allowed_accel_ids;
    // When true the daemon mounts block_path at backing_mount_path, records
    // ownership, and unmounts during graceful shutdown. When false it only
    // consumes an operator-prepared mount.
    bool                 auto_mount = true;
};

struct QueuePoolConfig {
    // Requested zero means default_per_client. The effective grant is further
    // clamped by max_per_client, max_queues_per_group, and available capacity.
    int32_t default_per_client = 4;
    int32_t max_per_client = 32;
};

struct LeaseConfig {
    uint32_t heartbeat_interval_sec = 10;
    uint32_t timeout_sec = 30;
};

struct UnmountRetryConfig {
    // EBUSY retry policy during graceful daemon shutdown. A second signal can
    // stop the retry loop and leave the mount for explicit operator cleanup.
    uint32_t interval_ms = 1000;
    uint32_t max = 30;
};

struct ServiceConfig {
    GrpcConfig                   grpc;
    std::vector<AcceleratorEntry> accelerators;
    std::vector<NvmeEntry>       nvmes;
    QueuePoolConfig              queue_pool;
    LeaseConfig                  lease;
    UnmountRetryConfig           unmount_retry;
};

struct ConfigDiagnostics {
    // Injectable warnings, primarily for legacy-schema deprecation assertions.
    std::vector<std::string> warnings;
};

std::optional<ServiceConfig> parse_config_file(
    const std::string& path,
    std::string* error = nullptr,
    ConfigDiagnostics* diagnostics = nullptr);

// Cross-reference validator for IDs, paths, PCI BDFs, ACLs, policy, and lease
// cadence.  It does not probe hardware; bring-up validates device facts later.
bool validate_config(const ServiceConfig& cfg, std::string* error = nullptr);

// Validate the invariant required by striped operation: every namespace
// brought up by one daemon must report the same logical block size in bytes.
// A non-4 KiB value is intentionally not rejected here; callers warn about it
// separately so a uniform non-default namespace remains usable for
// non-striped workloads.
bool validate_uniform_block_size(const std::vector<uint32_t>& block_sizes,
                                 std::string* error = nullptr);

} // namespace nvmeservice

#endif // __NVMESERVICE_CONFIG_H__

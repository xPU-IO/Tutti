#ifndef __NVMESERVICE_CONFIG_H__
#define __NVMESERVICE_CONFIG_H__

/**
 * nvmeservice_config.h -- sys_config.yaml parser for NVMeService daemon.
 *
 * Post-L1-Commit-4b schema (no per-GPU queue accounting in YAML)
 * -------------------------------------------------------------
 *
 * The kernel's user QID pool [start_cq_idx, max_user_qid] is the
 * single source of truth for how many user queues exist on a given
 * NVMe.  The daemon learns those values via NVM_GET_DEV_INFO at
 * bring-up and does NOT track any shadow accounting in YAML.
 *
 * What's left for YAML to express:
 *
 *   * GPU enumeration + per-GPU mount paths (symlink targets).
 *   * NVMe enumeration + which GPUs are allowed to attach to each
 *     NVMe (NUMA / PCIe-switch affinity ACL).
 *   * kernel_ioq_cap pre-bind hint (NVM_SET_KERNEL_IOQ_CAP).
 *   * Per-client allocation policy (default_per_client / max_per_client).
 *   * Lease cadence.
 *
 * YAML schema:
 *
 *   grpc:
 *     endpoint: "127.0.0.1:50051"
 *
 *   gpus:
 *     - id: 0
 *       mount_path: "/mnt/gpu0"
 *
 *   nvmes:
 *     - pci_addr: "0000:50:00.0"
 *       mount_path: "/mnt/nvme0"
 *       namespace_id: 1                # libnvm doesn't get this from the
 *                                      # kernel today; YAML must supply it.
 *       kernel_ioq_cap: 32             # OPTIONAL: NVM_SET_KERNEL_IOQ_CAP
 *                                      # value passed pre-bind.  0 = let the
 *                                      # kernel pick num_possible_cpus().
 *       allowed_gpus: [0, 1]           # OPTIONAL: NUMA / PCIe-switch
 *                                      # affinity ACL.  When omitted,
 *                                      # every GPU in `gpus` may attach.
 *
 *   queue_pool:
 *     default_per_client: 4            # ConnectRequest.num_queues=0 -> use this
 *     max_per_client: 16               # hard cap; further clamped by
 *                                      # kernel max_queues_per_group at runtime
 *
 *   lease:
 *     heartbeat_interval_sec: 10
 *     timeout_sec: 30
 */

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nvmeservice {

struct GrpcConfig {
    std::string endpoint = "127.0.0.1:50051";
};

struct GpuEntry {
    int         id = -1;
    std::string mount_path;
};

struct NvmeEntry {
    std::string         pci_addr;
    std::string         mount_path;          // real NVMe mount, e.g. "/mnt/nvme0"
    uint32_t            namespace_id   = 1;

    // NVM_SET_KERNEL_IOQ_CAP hint (QueuePair units).  0 == "no cap,
    // kernel uses num_possible_cpus()".  Set explicitly when the
    // controller's MSI-X budget is smaller than the host vCPU count
    // (otherwise s_nvme_setup_io_queues falls back to
    // dma_alloc_coherent and the user share path silently misses GPU
    // memory).
    uint32_t            kernel_ioq_cap = 0;

    // Optional ACL: which gpus[].id values may Connect to this NVMe.
    // Empty == "any GPU in gpus[]".  Used both for ACL enforcement
    // in ConnectRequest and to decide which GPU view paths get a
    // symlink installed.  Validator cross-checks each id against
    // gpus[].
    std::vector<int>    allowed_gpus;
};

struct QueuePoolConfig {
    int default_per_client = 4;
    int max_per_client     = 16;
};

struct LeaseConfig {
    uint32_t heartbeat_interval_sec = 10;
    uint32_t timeout_sec            = 30;
};

struct ServiceConfig {
    GrpcConfig             grpc;
    std::vector<GpuEntry>  gpus;
    std::vector<NvmeEntry> nvmes;
    QueuePoolConfig        queue_pool;
    LeaseConfig            lease;
};

/**
 * Parse a YAML file into ServiceConfig.  Returns std::nullopt on
 * parse / validation error; the error message is written to *error
 * when non-null.
 */
std::optional<ServiceConfig> parse_config_file(const std::string& path,
                                                std::string* error = nullptr);

/**
 * Cross-reference validator: every nvmes[].allowed_gpus id must be
 * present in gpus[]; no duplicate pci_addr / mount_path; pool sizes
 * sane; lease cadence sane.  No "queue accounting" check -- that
 * lives in the kernel now.
 */
bool validate_config(const ServiceConfig& cfg, std::string* error = nullptr);

} // namespace nvmeservice

#endif // __NVMESERVICE_CONFIG_H__

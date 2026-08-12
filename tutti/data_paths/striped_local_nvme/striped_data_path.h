#pragma once

// tutti/data_paths/striped_local_nvme/striped_data_path.h
//
// StripedDataPath — single-kernel fused submission across N NVMe devices.
//
// Implements the DataPath SPI for striped:// targets backed by N local NVMe
// devices.  The key property (maintainer-mandated design, Round 15 S5):
// ONE cudaLaunchKernel dispatches IO entries to N devices' queues, using a
// per-op device table of N DeviceTargetHandle pointers.  Workspace (entries,
// status, PRP-list pages, device table, event) is leased from a bounded
// StripedArena — zero per-op cudaMalloc.
//
// Scope constraint (documented, not a Runtime/SPI limitation): the SPI
// (spi/data_path.h) permits a submit() batch to span multiple targets
// within one DataPath. This DataPath honors that contract but has a
// device-table CAPACITY of exactly N slots per op (one striped target's
// full shard set). A batch whose requests span more than one striped
// target therefore hits RESOURCE_EXHAUSTED for the requests beyond the
// first target -- an explicit per-request capacity rejection (partial
// commit), the same mechanism used for over-large batches elsewhere in
// submit(), not a silent single-target assumption.
//
// Lifecycle:
//   initialize()  — attach N controllers, create N queue groups, arena init
//   open()        — extract StripedLocalNvmePayload, build N DeviceTargetHandles
//   register_memory() — nvm_dma_map_data_device × N (same buffer, N IOVA tables)
//   submit()      — stripe-split -> entries with dev_idx -> 1 H2D -> 1 launch -> 1 event
//   progress()    — poll the op's event; D2H + aggregate on signal
//   query()       — return aggregated snapshot
//   release()     — return op's arena lease
//   close()       — release N target handles
//   shutdown()    — release N controllers + queue groups + arena

#include <tutti/spi/data_path.h>
#include <tutti/status.h>
#include <tutti/io_types.h>
#include "tutti/bindings/striped_local_nvme/binding.h"
#include "tutti/data_paths/striped_local_nvme/striped_arena.h"
#include "tutti/data_paths/local_nvme/metadata/prp_page_cache.h"  // Round 16 S5

#include <nvm_ctrl.h>
#include <nvm_types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tutti::data_paths::local_nvme {
class NvmeQueueGroup;
struct DeviceTargetHandle;
struct EntryCompletionStatus;
} // namespace tutti::data_paths::local_nvme

namespace tutti::data_paths::striped_local_nvme {

struct StripedDeviceSubmitEntry;  // defined in fused_submit_kernel.cuh

using tutti::DataPath;
using tutti::DataPathCapabilities;
using tutti::DataPathConfig;
using tutti::DataPathMemory;
using tutti::DataPathMemoryView;
using tutti::DataPathOp;
using tutti::DataPathRequest;
using tutti::DataPathSnapshot;
using tutti::DataPathTarget;
using tutti::HostSubmitContext;
using tutti::ProgressBudget;
using tutti::ProgressResult;
using tutti::RegistrationDomainKey;
using tutti::ResourceProvider;
using tutti::SubmitOutcome;
using tutti::OpState;
using tutti::RequestState;
using tutti::StatusCode;
using tutti::Status;
using tutti::Result;

// -------------------------------------------------------------------------
// DeviceDescriptor — one NVMe device's configuration.
//
// All N namespaces must share the same logical block size in bytes
// (block_size, validated in initialize()); this is the NVMe LBA data size, not a
// filesystem or physical block size.
// this is a documented assumption, not an SPI constraint.
// -------------------------------------------------------------------------
struct DeviceDescriptor {
    std::string snvme_dev_path;  // e.g. "/dev/ssnvme0"
    std::uint64_t bar0_size = 0;
    std::uint32_t namespace_id = 1;
    std::uint32_t cuda_device = 0;
    std::uint32_t num_user_queues = 1;
    // (no queue_depth field: ring depth is fixed by the kernel module's
    //  io_queue_depth and obtained via NVM_GET_DEV_INFO at initialize())
    std::uint32_t block_size = 4096;
    std::string controller_pci_addr;
};

// -------------------------------------------------------------------------
// StripedDataPath — fused multi-device DataPath.
// -------------------------------------------------------------------------
class StripedDataPath : public DataPath {
public:
    // devices: N device descriptors (order defines shard index [0, N)).
    // cuda_device: the single GPU the fused kernel launches on; every
    //   device's queue group's d_qps must be resident/accessible from this
    //   GPU (same GPU as devices[i].cuda_device for all i, in the tested
    //   single-GPU-multi-NVMe topology).
    // mdts_override: 0 = use hardware min(); else min(override, hardware min()).
    // max_batch_entries: max fan-out entries per op (bounds StripedArena sizing).
    // max_in_flight_operations: cap on concurrent IN_FLIGHT ops.
    // handle_cache_capacity: 0 = OFF (default); >0 = GPU LRU handle cache slots.
    // prp_cache_capacity: 0 = OFF (default); >0 = PRP LIST page cache slots.
    //   (Round 16 S5: aligned to LocalNvmeDataPath's prp_cache_capacity.)
    // threads_per_block: fused submit kernel block size (1..1024, default 16).
    //   Must not exceed any device's actual queue count.
    StripedDataPath(std::vector<DeviceDescriptor> devices,
                    std::uint32_t cuda_device = 0,
                    std::uint64_t mdts_override = 0,
                    std::uint32_t cq_poll_budget = 2000000,
                    std::uint32_t max_batch_entries = 256,
                    std::uint32_t max_in_flight_operations = 16,
                    std::uint32_t handle_cache_capacity = 0,
                    std::uint32_t prp_cache_capacity = 0,
                    std::uint32_t threads_per_block = 16);
    ~StripedDataPath() override;

    StripedDataPath(const StripedDataPath&) = delete;
    StripedDataPath& operator=(const StripedDataPath&) = delete;

    // ---- DataPath SPI ----
    const DataPathCapabilities& capabilities() const override;
    Status initialize(const DataPathConfig& config,
                      ResourceProvider& resources) override;
    Status shutdown(std::uint64_t timeout_ns) override;

    Result<DataPathTarget> open(const ResolvedTarget& target) override;
    Status close(DataPathTarget target) override;
    Result<RegistrationDomainKey> registration_domain(
        DataPathTarget target) const override;

    Result<DataPathMemory> register_memory(
        const DataPathMemoryView& view,
        const RegistrationDomainKey& domain) override;
    Status unregister_memory(DataPathMemory memory) override;

    SubmitOutcome submit(const DataPathRequest* requests,
                         std::size_t count,
                         const HostSubmitContext& ctx) override;
    Result<ProgressResult> progress(ProgressBudget budget) override;
    Result<DataPathSnapshot> query(DataPathOp op) const override;
    Status release(DataPathOp op) override;

    // ---- test-only accessors ----
    std::uint32_t test_num_devices() const {
        return static_cast<std::uint32_t>(devices_.size());
    }
    const std::vector<DeviceDescriptor>& test_device_descriptors() const {
        return device_descs_;
    }
    std::uint64_t test_effective_mdts() const { return effective_mdts_bytes_; }
    std::uint32_t test_threads_per_block() const {
        return threads_per_block_;
    }
    std::uint64_t test_submit_call_count() const { return test_submit_call_count_; }
    std::uint64_t test_kernel_launch_count() const { return test_kernel_launch_count_; }
    void test_reset_submit_counters() {
        test_submit_call_count_ = 0;
        test_kernel_launch_count_ = 0;
    }
    std::uint32_t test_arena_capacity() const { return arena_.capacity(); }
    std::uint32_t test_arena_available() const { return arena_.available(); }
    bool test_op_has_timeout(DataPathOp op) const;
    // Number of fan-out entries the given op produced (0 if op not found).
    std::uint32_t test_entry_count(DataPathOp op) const;
    // Per-entry dev_idx (shard) the op's fan-out assigned, D2H'd on demand.
    // Returns false if op not found or D2H failed; out resized to entry count.
    bool test_copy_entry_dev_idx(DataPathOp op,
                                 std::vector<std::uint32_t>& out) const;

private:
    // Public SPI entry points are thin device-guarded wrappers; impl methods
    // keep resource/error paths free of duplicated current-device plumbing.
    Status initialize_impl_(const DataPathConfig& config,
                            ResourceProvider& resources);
    Status shutdown_impl_(std::uint64_t timeout_ns);
    Result<DataPathTarget> open_impl_(const ResolvedTarget& target);
    Status close_impl_(DataPathTarget target);
    Result<RegistrationDomainKey> registration_domain_impl_(
        DataPathTarget target) const;
    Result<DataPathMemory> register_memory_impl_(
        const DataPathMemoryView& view,
        const RegistrationDomainKey& domain);
    Status unregister_memory_impl_(DataPathMemory memory);
    SubmitOutcome submit_impl_(const DataPathRequest* requests,
                               std::size_t count,
                               const HostSubmitContext& ctx);
    Result<ProgressResult> progress_impl_(ProgressBudget budget);
    Result<DataPathSnapshot> query_impl_(DataPathOp op) const;
    Status release_impl_(DataPathOp op);

    struct DeviceSlot {
        DeviceDescriptor desc;
        nvm_ctrl_t* ctrl = nullptr;
        std::unique_ptr<tutti::data_paths::local_nvme::NvmeQueueGroup> queue_group;
        std::uint64_t hardware_mdts = 0;
        std::uint64_t page_size = 0;
    };

    // Byte-unit host-side extent (mirrors LocalNvmeDataPath::LbaExtent's
    // logical_offset_bytes/length role, kept in bytes here since the fused
    // kernel's device handle already carries LBA-unit extents; the host
    // side only needs byte extents to clamp stripe sub-IOs at extent
    // boundaries before they reach resolve_lba on the device).
    struct HostExtent {
        std::uint64_t logical_offset_bytes = 0;
        std::uint64_t length_bytes = 0;
    };

    struct StripedTarget {
        std::uint32_t num_shards = 0;
        std::uint64_t stripe_unit = 0;
        // Round 16 S7: per-target shard rotation, mirrored from the
        // payload (legacy shard_placement equivalent).  0 = no rotation.
        std::uint32_t shard_rotation = 0;
        std::uint64_t logical_size = 0;
        // N DeviceTargetHandle* (GPU pointers), one per shard.
        std::vector<tutti::data_paths::local_nvme::DeviceTargetHandle*> dev_handles;
        // N overflow extents buffers (owned, freed on close).
        std::vector<void*> overflow_allocs;
        // Per-shard host-side extents (for stripe-split boundary clamping).
        std::vector<std::vector<HostExtent>> shard_extents;
        std::string domain_key;
        std::uint64_t generation = 0;
    };

    struct StripedMemory {
        void* base = nullptr;
        std::uint64_t size = 0;
        // N nvm_dma_t* (one per device), each with its own IOVA table for
        // the SAME GPU buffer.
        std::vector<nvm_dma_t*> dmas;
        std::uint64_t generation = 0;

        // Round 16 S5 (V3): per-device pre-built AddressDescriptor[].
        // Populated when io_granularity > 0 at registration time.
        // d_descs[d] = GPU-resident AddressDescriptor[] for device d.
        // Each descriptor covers one sub-IO of bytes_per_slice.
        struct Prebuilt {
            std::vector<void*> d_descs_per_dev;   // N GPU pointers
            std::uint64_t bytes_per_slice = 0;
            std::uint64_t num_descs = 0;           // same per device (same IOVA layout)
            bool valid = false;
        };
        Prebuilt prebuilt;
    };

    struct OpEntry {
        OpState state = OpState::IN_FLIGHT;
        Status status;
        std::uint64_t bytes_transferred = 0;
        std::uint64_t total_bytes = 0;

        std::uint32_t arena_slot = UINT32_MAX;
        StripedDeviceSubmitEntry* d_entries = nullptr;
        local_nvme::EntryCompletionStatus* d_status = nullptr;
        std::uint32_t entry_count = 0;
        // Round 16 S6 (REQUIRED 0): entry lengths (was inline in
        // StripedDeviceSubmitEntry::length; now in descriptor on GPU).
        std::vector<std::uint64_t> entry_lengths;
        void* event = nullptr;   // cudaEvent_t
        void* stream = nullptr;  // borrowed cudaStream_t

        // PRP-list workspace (borrowed from arena's pre-allocated pool).
        std::uint32_t prp_ioaddrs_base = 0;
        void* prp_pages_devptr = nullptr;
        std::uint32_t prp_list_page_count = 0;

        bool has_timeout = false;

        std::uint64_t target_token = 0;
        // P0-2 fix: collect ALL accepted requests' memory tokens so
        // memory_has_inflight_ops_() correctly prevents unregister during
        // in-flight ops.  A batch may span multiple memory registrations.
        std::vector<std::uint64_t> memory_tokens;

        std::uint64_t op_token = 0;
        std::uint64_t op_generation = 0;
    };

    const StripedTarget* find_target_(DataPathTarget target) const;
    StripedTarget* find_target_(DataPathTarget target);
    const StripedMemory* find_memory_(DataPathMemory memory) const;
    StripedMemory* find_memory_(DataPathMemory memory);
    const OpEntry* find_op_(DataPathOp op) const;
    OpEntry* find_op_(DataPathOp op);
    bool target_has_inflight_ops_(std::uint64_t token) const;
    bool memory_has_inflight_ops_(std::uint64_t token) const;

    // Build a DeviceTargetHandle + host extents for one shard.
    bool build_shard_handle_(
        std::uint32_t dev_idx,
        const ResolvedTarget& shard_target,
        StripedTarget& out);

    // D2H the op's status array, aggregate into op.state/status/bytes.
    void aggregate_completion_status_(OpEntry& op);

    // ---- Members ----
    std::vector<DeviceDescriptor> device_descs_;
    std::vector<DeviceSlot> devices_;
    std::uint32_t cuda_device_ = 0;
    std::uint64_t mdts_override_ = 0;
    std::uint64_t effective_mdts_bytes_ = 0;
    std::uint32_t cq_poll_budget_ = 0;
    std::uint32_t max_batch_entries_ = 0;
    std::uint64_t max_in_flight_operations_ = 0;
    std::uint32_t threads_per_block_ = 16;
    // Round 16 S5: cache capacities (default OFF, aligned to LocalNvme).
    std::uint32_t handle_cache_capacity_ = 0;
    std::uint32_t prp_cache_capacity_ = 0;
    std::uint32_t block_size_ = 0;         // uniform across all shards
    std::uint64_t max_request_bytes_ = 0;  // max_batch_entries_ * effective_mdts_bytes_
    bool initialized_ = false;

    DataPathCapabilities caps_{};

    StripedArena arena_;

    // Round 16 S5: per-device PRP page cache (one per controller).
    std::vector<std::unique_ptr<tutti::data_paths::local_nvme::PrpPageCache>> prp_caches_;

    std::uint64_t next_target_ = 1;
    std::uint64_t next_memory_ = 1;
    std::uint64_t next_op_token_ = 1;
    std::unordered_map<std::uint64_t, StripedTarget> targets_;
    std::unordered_map<std::uint64_t, StripedMemory> memory_regs_;
    std::unordered_map<std::uint64_t, OpEntry> ops_;

    // Round 16 S5 (V3): registration-time pre-build for striped.
    bool build_striped_prebuilt_(StripedMemory& mem,
                                  std::uint64_t io_granularity,
                                  std::string& status_msg);
    void destroy_striped_prebuilt_(StripedMemory& mem);

    std::uint64_t test_submit_call_count_ = 0;
    std::uint64_t test_kernel_launch_count_ = 0;
};

} // namespace tutti::data_paths::striped_local_nvme

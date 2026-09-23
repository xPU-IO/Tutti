#pragma once

// csrc/data_paths/striped_local_nvme/striped_data_path.h
//
// StripedDataPath — single-kernel fused submission across N NVMe devices.
//
// Implements the DataPath SPI for file:// targets backed by N local NVMe
// devices.  The key property (maintainer-mandated design, Round 15 S5):
// ONE cudaLaunchKernel dispatches IO entries to N devices' queues, using a
// per-op device table of DeviceTargetHandle pointers.  Workspace (entries,
// status, PRP-list pages, device table, event) is leased from a bounded
// StripedArena — zero per-op cudaMalloc.
//
// Placement model (2026-09-22): every target is ONE file living entirely on
// ONE of the N devices; the placement layer (RotatingFilePlacement) decides
// which device by the slot number, and this DataPath finds the device by
// matching the payload's controller PCI address.  A request is NEVER split
// across devices -- long prompts get their N-way parallelism from having many
// chunks, whose files rotate across the devices, in one submit.  Sub-IO
// splitting happens only at MDTS and file-extent boundaries, exactly like
// LocalNvmeDataPath.
//
// Scope constraint (documented, not a Runtime/SPI limitation): the SPI
// (spi/data_path.h) permits a submit() batch to span multiple targets
// within one DataPath. This DataPath honors that contract with a device-table
// CAPACITY of N slots per op (one device table row per target). A batch whose
// targets' devices exceed that capacity hits RESOURCE_EXHAUSTED for the
// requests beyond it -- an explicit per-request capacity rejection (partial
// commit), the same mechanism used for over-large batches elsewhere in
// submit(), not a silent single-target assumption.
//
// Lifecycle:
//   initialize()  — attach N controllers, create N queue groups, arena init
//   open()        — extract Ext4LocalNvmePayload, match PCI -> ONE device handle
//   register_memory() — nvm_dma_map_data_device × N (same buffer, N IOVA tables)
//   submit()      — clamp at MDTS/extents -> entries with dev_idx -> 1 H2D -> 1 launch
//   progress()    — poll the op's event; D2H + aggregate on signal
//   query()       — return aggregated snapshot
//   release()     — return op's arena lease
//   close()       — release target handle
//   shutdown()    — release N controllers + queue groups + arena

#include <tutti/spi/data_path.h>
#include <tutti/status.h>
#include <tutti/io_types.h>
#include "csrc/payloads/ext4_local_nvme/payload.h"
#include "csrc/data_paths/striped_local_nvme/striped_arena.h"
#include "csrc/data_paths/local_nvme/metadata/prp_page_cache.h"  // Round 16 S5
#include "csrc/data_paths/local_nvme/metadata/prp_buf_pool.h"

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
using tutti::IoState;
using tutti::IoRequestState;
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
    // prp_cache_capacity: 0 = OFF (default); >0 = PRP LIST page cache slots.
    //   (Round 16 S5: aligned to LocalNvmeDataPath's prp_cache_capacity.)
    // threads_per_block: fused submit kernel block size (1..1024, default 16).
    //   Must not exceed any device's actual queue count.
    //
    // 注：本 DataPath 没有 HandleWorkspaceCache 实例，故不接受
    // handle_cache_capacity（S1）——同一配置键只对 LocalNvmeDataPath 生效。
    StripedDataPath(std::vector<DeviceDescriptor> devices,
                    std::uint32_t cuda_device = 0,
                    std::uint64_t mdts_override = 0,
                    std::uint32_t cq_poll_budget = 2000000,
                    std::uint32_t max_batch_entries = 256,
                    std::uint32_t max_in_flight_operations = 16,
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
    std::uint64_t test_device_hardware_mdts(std::uint32_t device) const {
        return device < devices_.size() ? devices_[device].hardware_mdts : 0;
    }
    std::uint64_t test_device_effective_mdts(std::uint32_t device) const {
        return device_effective_mdts_(device);
    }
    std::uint32_t test_threads_per_block() const {
        return threads_per_block_;
    }
    std::uint64_t test_submit_call_count() const { return test_submit_call_count_; }
    std::uint64_t test_kernel_launch_count() const { return test_kernel_launch_count_; }
    void test_reset_submit_counters() {
        test_submit_call_count_ = 0;
        test_kernel_launch_count_ = 0;
        test_last_prebuilt_entry_count_ = 0;
        test_last_dynamic_entry_count_ = 0;
    }
    std::uint64_t test_last_prebuilt_entry_count() const {
        return test_last_prebuilt_entry_count_;
    }
    std::uint64_t test_last_dynamic_entry_count() const {
        return test_last_dynamic_entry_count_;
    }
    std::uint32_t test_arena_capacity() const { return arena_.capacity(); }
    std::uint32_t test_arena_available() const { return arena_.available(); }
    StripedArena::AllocCounts test_arena_alloc_counts() const {
        return arena_.alloc_counts();
    }
    void test_arena_reset_alloc_counts() { arena_.reset_alloc_counts(); }
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
    // side only needs byte extents to clamp sub-IOs at extent boundaries
    // before they reach resolve_lba on the device).
    struct HostExtent {
        std::uint64_t logical_offset_bytes = 0;
        std::uint64_t length_bytes = 0;
    };

    // One target = one file on one device. The device is matched at open()
    // from the payload's namespace identity (controller PCI address), so
    // placement -- not this DataPath -- decides where a file lives.
    struct StripedTarget {
        // Index into devices_ of the device this file lives on.
        std::uint32_t dev_idx = 0;
        std::uint64_t logical_size = 0;
        // 1 DeviceTargetHandle* (GPU pointer), for dev_idx's device.
        std::vector<tutti::data_paths::local_nvme::DeviceTargetHandle*> dev_handles;
        // overflow extents buffer (owned, freed on close).
        std::vector<void*> overflow_allocs;
        // Host-side extents of the file (for sub-IO boundary clamping).
        std::vector<HostExtent> extents;
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

        // Per-controller pre-built AddressDescriptor tables. io_granularity
        // remains the logical block size; every device splits that block by
        // its own driver-reported MDTS.
        struct Prebuilt {
            struct DeviceTable {
                void* d_descs = nullptr;
                std::uint64_t mdts_bytes = 0;
                std::uint64_t ios_per_slice = 0;
                std::uint64_t num_descs = 0;
                tutti::data_paths::local_nvme::PrpBufRef prp_buf_ref;
                std::uint64_t num_prp_pages = 0;
            };
            std::vector<DeviceTable> devices;
            std::uint64_t bytes_per_slice = 0;  // logical io_granularity
            std::uint64_t num_slices = 0;
            bool valid = false;
        };
        Prebuilt prebuilt;
    };

    struct OpEntry {
        IoState state = IoState::IN_FLIGHT;
        Status status;
        std::uint64_t bytes_transferred = 0;
        std::uint64_t total_bytes = 0;
        IoCompletionDetail completion_detail;

        std::uint32_t arena_slot = UINT32_MAX;
        StripedDeviceSubmitEntry* d_entries = nullptr;
        local_nvme::EntryCompletionStatus* d_status = nullptr;
        std::uint32_t entry_count = 0;
        // Round 16 S6 (REQUIRED 0): entry lengths (was inline in
        // StripedDeviceSubmitEntry::length; now in descriptor on GPU).
        std::vector<std::uint64_t> entry_lengths;
        void* event = nullptr;   // cudaEvent_t
        void* stream = nullptr;  // borrowed cudaStream_t

        // Host-pinned PRP-list leases, one or more per controller.
        // 每设备一段 host 池页面 + 其归属池：完成路径按池归还，
        // 否则 cache-miss 提交会把池无限撑大（P0，见 prp_buf_pool.h）。
        std::vector<tutti::data_paths::local_nvme::PrpBufLeaseRef>
            prp_buf_refs;

        bool has_timeout = false;
        // Per-layer read-copy/reuse fences; see LocalNvmeDataPath::OpEntry.

        std::uint64_t target_token = 0;
        // P0-2 fix: collect ALL accepted requests' memory tokens so
        // memory_has_inflight_ops_() correctly prevents unregister during
        // in-flight ops.  A batch may span multiple memory registrations.
        std::vector<std::uint64_t> memory_tokens;

        // PRP cache entries pinned for this op's lifetime (pin at submit,
        // unpin at release).  Without this the striped path used to leak
        // every cache entry it touched (checkout never released).
        struct PrpCacheRef {
            tutti::data_paths::local_nvme::PrpPageCache* cache = nullptr;
            tutti::data_paths::local_nvme::PrpPageCache::Entry* entry = nullptr;
        };
        std::vector<PrpCacheRef> prp_cache_refs;

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

    // Build the DeviceTargetHandle + host extents for the target's file.
    // Returns the matched device index through out.dev_idx.
    bool build_file_handle_(const ResolvedTarget& target,
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
    // Round 16 S5: cache capacity (default OFF, aligned to LocalNvme).
    std::uint32_t prp_cache_capacity_ = 0;
    std::uint32_t block_size_ = 0;         // uniform across all shards
    std::uint64_t max_request_bytes_ = 0;  // max_batch_entries_ * effective_mdts_bytes_
    bool initialized_ = false;

    DataPathCapabilities caps_{};

    StripedArena arena_;

    // Round 16 S5: per-device PRP page cache (one per controller).
    std::vector<std::unique_ptr<tutti::data_paths::local_nvme::PrpPageCache>> prp_caches_;
    // Growing host-pinned miss/exhaustion pool, one per controller/IOMMU domain.
    std::vector<std::unique_ptr<tutti::data_paths::local_nvme::PrpBufPool>> prp_buf_pools_;

    // ---- test-only: host PRP pool accounting (summed over devices) ----
    // Pages handed out to ops. Must return to 0 after every op is released and
    // must not grow across repeated identical submits; see
    // LocalNvmeDataPath's accessors for the leak this guards.
public:
    std::uint64_t test_prp_pool_leased_pages() const;
    std::uint64_t test_prp_pool_total_pages() const;

private:
    bool timeout_prp_retained_ = false;

    std::uint64_t next_target_token_ = 1;
    std::uint64_t next_memory_token_ = 1;
    std::uint64_t next_op_token_ = 1;
    // Registration domain shared by every target of this DataPath: the
    // memory registration maps the buffer for ALL devices_ regardless of
    // the target, so the domain must be a function of the device set --
    // NOT of the target token.  A per-target key makes the runtime
    // re-run nvm_dma_map_data_device for every freshly opened target
    // (measured 2026-09-14: 39 new chunk targets x 2 shards = 19.3s in
    // submit, each map taking the NVIDIA RM global lock), while the
    // device-keyed key registers once and reuses.
    std::string device_domain_key_;
    std::unordered_map<std::uint64_t, StripedTarget> targets_;
    std::unordered_map<std::uint64_t, StripedMemory> memory_regs_;
    std::unordered_map<std::uint64_t, OpEntry> ops_;

    // Round 16 S5 (V3): registration-time pre-build for striped.
    bool build_striped_prebuilt_(StripedMemory& mem,
                                  std::uint64_t io_granularity,
                                  std::string& status_msg);
    void destroy_striped_prebuilt_(StripedMemory& mem);
    std::uint64_t device_effective_mdts_(std::uint32_t device) const;

    std::uint64_t test_submit_call_count_ = 0;
    std::uint64_t test_kernel_launch_count_ = 0;
    std::uint64_t test_last_prebuilt_entry_count_ = 0;
    std::uint64_t test_last_dynamic_entry_count_ = 0;

    // ---- Optional per-entry IO timing (env TUTTI_IO_TIMING=1) ----
    // The fused kernel stamps 4 globaltimer values per entry; the wait path
    // copies them back and periodically prints per-stage average/max.  Pure
    // diagnostic: disabled (nullptr) unless the env var is set.
    bool io_timing_enabled_ = false;
    unsigned long long* d_timing_ = nullptr;
    std::uint32_t timing_capacity_entries_ = 0;
    std::vector<unsigned long long> timing_host_;
    std::uint64_t timing_acc_batches_ = 0;
    std::uint64_t timing_acc_entries_ = 0;
    // Stages: [0] submit setup, [1] command execution (doorbell->CQE visible),
    //         [2] completion reclaim, [3] per-entry lifetime.
    std::uint64_t timing_acc_ns_[4] = {0, 0, 0, 0};
    std::uint64_t timing_max_ns_[4] = {0, 0, 0, 0};
    void io_timing_accumulate_(OpEntry& op);

    // ---- Worker-pool kernel model (default: 2048 workers) ----
    // 0 = legacy one-thread-per-entry kernel.  >0 = fixed pool of N worker
    // threads pulling entries from a per-batch task cursor (the arena slot's
    // trailing status element — per-slot, so concurrent submits on different
    // CUDA streams never share a counter).  See fused_submit_kernel.cuh.
    std::uint32_t pool_workers_ = 0;
};

} // namespace tutti::data_paths::striped_local_nvme

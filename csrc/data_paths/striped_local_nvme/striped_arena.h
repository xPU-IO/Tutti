#pragma once

// tutti/data_paths/striped_local_nvme/striped_arena.h
//
// Per-DataPath, bounded StripedArena for StripedDataPath.
//
// Same design as LocalNvmeDataPath's MetadataArena (see
// tutti/data_paths/local_nvme/metadata/metadata_arena.h): pre-allocates all
// per-op GPU workspace at initialize() time so submit() performs zero
// cudaMalloc/cudaEventCreate calls (Round 15 Session 5 boundary: "禁止 per-op
// cudaMalloc 简化交付").
//
// PRP-list pages are host-pinned per-controller resources owned by
// StripedDataPath's PrpBufPool/PrpPageCache, never by this GPU arena.
//
// submit() calls acquire() to lease a slot; release() returns it.  Arena
// exhaustion -> submit() returns RESOURCE_EXHAUSTED (no cudaMalloc fallback).
// Timeout only retains the separate host PRP lease; GPU metadata slots remain
// reusable.

#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

#include <nvm_types.h>

namespace tutti::data_paths::local_nvme {
struct DeviceTargetHandle;
struct EntryCompletionStatus;  // defined in nvme_submit_primitives.cuh
struct AddressDescriptor;       // defined in io/prp_builder.h (24 bytes)
} // namespace tutti::data_paths::local_nvme

namespace tutti::data_paths::striped_local_nvme {

// Forward declaration (defined in fused_submit_kernel.cuh).
struct StripedDeviceSubmitEntry;

// Reused verbatim from local_nvme (same struct, same layout) -- NOT a
// redeclaration: this is the SAME type as
// tutti::data_paths::local_nvme::EntryCompletionStatus, just visible under
// this namespace too so callers don't need to spell the full local_nvme::
// qualifier everywhere in this file.
using EntryCompletionStatus = tutti::data_paths::local_nvme::EntryCompletionStatus;

class StripedArena {
public:
    struct Config {
        std::uint32_t num_slots = 0;             // = 2 * max_in_flight_operations
        std::uint32_t max_entries_per_slot = 0;  // = max_batch_entries
        std::uint32_t page_size = 4096;          // NVMe page size (assumed uniform across devices)
        std::uint32_t cuda_device = 0;
        // Device table capacity per slot: max distinct (target,shard) handles
        // referenced by one submit() call.  Bounded at num_devices (N) --
        // one submit call fans a single striped target out across at most
        // its N shards.  Batches spanning multiple striped targets that
        // together reference more than N distinct handles are rejected by
        // submit() (RESOURCE_EXHAUSTED), not supported by this arena.
        std::uint32_t dev_table_capacity_per_slot = 0;
    };

    // A lease grants exclusive use of one arena slot's workspace.
    // All pointers are pre-computed at init; acquire() is O(1) with
    // zero CUDA API calls.
    struct Lease {
        std::uint32_t slot_index = UINT32_MAX;
        void* event = nullptr;                          // cudaEvent_t
        StripedDeviceSubmitEntry* d_entries = nullptr;   // GPU: entry array base
        EntryCompletionStatus* d_status = nullptr;       // GPU: status array base
        // Device table workspace (pre-allocated).  Host fills up to
        // dev_table_capacity entries with DeviceTargetHandle* pointers,
        // H2D-copies to d_dev_table, then the kernel indexes it by dev_idx.
        const void** d_dev_table = nullptr;  // GPU: this slot's device table base
        std::uint32_t dev_table_capacity = 0;
        // Round 16 S6 (REQUIRED 0): per-slot descriptor pool for dynamic-path entries.
        const tutti::data_paths::local_nvme::AddressDescriptor* d_desc_pool = nullptr;
    };

    struct AllocCounts {
        std::uint64_t cuda_malloc = 0;
        std::uint64_t cuda_event_create = 0;
        std::uint64_t cuda_free = 0;
        std::uint64_t cuda_event_destroy = 0;
        std::uint64_t nvm_dma_map = 0;
        std::uint64_t nvm_dma_unmap = 0;
        std::uint64_t gpu_prp_cuda_malloc = 0;
    };

    StripedArena() = default;
    ~StripedArena();

    StripedArena(const StripedArena&) = delete;
    StripedArena& operator=(const StripedArena&) = delete;

    // Pre-allocate GPU metadata memory and events. `ctrls` remains for source
    // compatibility; no PRP DMA mapping is created. Must be called after all N
    // controllers are attached.  Returns false on any CUDA/DMA failure
    // (rolls back partial allocations).
    bool init(const Config& cfg, const std::vector<nvm_ctrl_t*>& ctrls);

    // Free all resources. Idempotent. Caller must ensure no in-flight GPU
    // work touches arena memory (sync all streams first).
    // `skip_prp` is a compatibility no-op: this arena owns no PRP backing.
    void shutdown(bool skip_prp = false);

    bool initialized() const { return initialized_; }
    std::uint32_t capacity() const { return cfg_.num_slots; }
    std::uint32_t available() const;

    bool acquire(Lease& out);
    void release(std::uint32_t slot_index);
    void release_with_timeout_leak(std::uint32_t slot_index);

    const AllocCounts& alloc_counts() const { return alloc_counts_; }
    void reset_alloc_counts() { alloc_counts_ = {}; }

private:
    Config cfg_{};
    std::vector<nvm_ctrl_t*> ctrls_;  // borrowed, one per device
    bool initialized_ = false;

    std::vector<void*> events_;  // cudaEvent_t stored as void*

    StripedDeviceSubmitEntry* d_entries_pool_ = nullptr;
    EntryCompletionStatus* d_status_pool_ = nullptr;
    void* d_dev_table_pool_ = nullptr;  // GPU: const void*[num_slots * dev_table_capacity_per_slot]
    const tutti::data_paths::local_nvme::AddressDescriptor* d_desc_pool_ = nullptr;  // Round 16 S6: dynamic-path descriptors

    std::deque<std::uint32_t> free_list_;
    mutable std::mutex mtx_;

    AllocCounts alloc_counts_;
};

} // namespace tutti::data_paths::striped_local_nvme

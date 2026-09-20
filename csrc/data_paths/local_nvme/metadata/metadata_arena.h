#pragma once

// tutti/data_paths/local_nvme/metadata/metadata_arena.h
//
// Per-device, bounded MetadataArena for LocalNvmeDataPath.
//
// Replaces per-op cudaMalloc/cudaEventCreate in the submit hot path.
// At init time, pre-allocates:
//   - N cudaEvent_t (one per slot, cudaEventDisableTiming)
//   - One contiguous GPU buffer for all DeviceSubmitEntry arrays
//   - One contiguous GPU buffer for all EntryCompletionStatus arrays
// PRP-list pages are deliberately NOT arena resources. They come from the
// host-pinned PrpBufPool/PrpPageCache.
//
// submit() calls acquire() to lease a slot; release() returns it.
// Arena exhaustion → submit() returns RESOURCE_EXHAUSTED (no fallback
// to cudaMalloc). Timeout only retains the separate host PRP lease; GPU
// metadata slots remain reusable.

#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

#include <nvm_types.h>

namespace tutti::data_paths::local_nvme {

// Forward declarations (defined in io/submit_one.cuh / nvme_submit_primitives.cuh).
struct DeviceSubmitEntry;
struct EntryCompletionStatus;
struct AddressDescriptor;  // defined in io/prp_builder.h (24 bytes: prp1/prp2/data_length)

class MetadataArena {
public:
    struct Config {
        std::uint32_t num_slots = 16;             // = max_in_flight_operations
        std::uint32_t max_entries_per_slot = 256;  // = max_batch_entries
        std::uint32_t page_size = 4096;            // NVMe page size
        std::uint32_t cuda_device = 0;
    };

    // A lease grants exclusive use of one arena slot's workspace.
    // All pointers are pre-computed at init; acquire() is O(1) with
    // zero CUDA API calls.
    struct Lease {
        std::uint32_t slot_index = UINT32_MAX;
        void* event = nullptr;                        // cudaEvent_t
        DeviceSubmitEntry* d_entries = nullptr;      // GPU: entry array base
        EntryCompletionStatus* d_status = nullptr;   // GPU: status array base
        // Round 16 S6 (REQUIRED 0): per-slot descriptor pool for the
        // dynamic (non-pre-built) submit path.  The kernel ALWAYS reads
        // prp1/prp2/data_length from e.prp_entry; for dynamic-path
        // entries the host writes the computed descriptor here and
        // H2D-copies it before launch.  Capacity = max_entries_per_slot.
        AddressDescriptor* d_desc_pool = nullptr;    // GPU: this slot's descriptor base
    };

    // Allocation counters — test seam to prove zero hot-path allocation.
    // Incremented only in init()/shutdown(), never in acquire()/release().
    struct AllocCounts {
        std::uint64_t cuda_malloc = 0;
        std::uint64_t cuda_event_create = 0;
        std::uint64_t cuda_free = 0;
        std::uint64_t cuda_event_destroy = 0;
        std::uint64_t nvm_dma_map = 0;
        std::uint64_t nvm_dma_unmap = 0;
        std::uint64_t gpu_prp_cuda_malloc = 0;
    };

    MetadataArena() = default;
    ~MetadataArena();

    MetadataArena(const MetadataArena&) = delete;
    MetadataArena& operator=(const MetadataArena&) = delete;

    // Pre-allocate GPU entry/status/descriptor memory and events. `ctrl` is
    // retained in the signature for source compatibility and is not mapped.
    // Returns false on any CUDA/DMA failure.
    bool init(const Config& cfg, nvm_ctrl_t* ctrl);

    // Free all resources. Idempotent. Caller must ensure no in-flight
    // GPU work touches arena memory (sync all streams first).
    // `skip_prp` is a compatibility no-op: the arena owns no PRP backing.
    void shutdown(bool skip_prp = false);

    bool initialized() const { return initialized_; }
    std::uint32_t capacity() const { return cfg_.num_slots; }
    std::uint32_t available() const;

    // Lease a slot. Returns false if arena is exhausted.
    // Zero CUDA API calls — pure CPU free-list pop.
    bool acquire(Lease& out);

    // Return a slot for reuse (normal completion).
    void release(std::uint32_t slot_index);

    // Compatibility alias. With host PRP leases outside the arena, timeout
    // no longer consumes GPU metadata slots.
    void release_with_timeout_leak(std::uint32_t slot_index);

    // Test seam: allocation counters.
    const AllocCounts& alloc_counts() const { return alloc_counts_; }
    void reset_alloc_counts() { alloc_counts_ = {}; }

private:
    Config cfg_{};
    nvm_ctrl_t* ctrl_ = nullptr;
    bool initialized_ = false;

    // Events: pre-created, one per slot.
    std::vector<void*> events_;  // cudaEvent_t stored as void*

    // Entry pool: one contiguous GPU buffer for all slots.
    // Slot i's entries: d_entries_pool_ + i * max_entries_per_slot
    DeviceSubmitEntry* d_entries_pool_ = nullptr;

    // Status pool: one contiguous GPU buffer for all slots.
    EntryCompletionStatus* d_status_pool_ = nullptr;

    // Round 16 S6 (REQUIRED 0): descriptor pool for dynamic-path entries.
    // One contiguous GPU buffer: num_slots * max_entries_per_slot * sizeof(AddressDescriptor).
    AddressDescriptor* d_desc_pool_ = nullptr;

    // Free-list of available slots.
    std::deque<std::uint32_t> free_list_;
    mutable std::mutex mtx_;

    AllocCounts alloc_counts_;
};

} // namespace tutti::data_paths::local_nvme

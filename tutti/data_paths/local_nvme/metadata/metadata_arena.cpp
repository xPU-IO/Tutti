// tutti/data_paths/local_nvme/metadata/metadata_arena.cpp

#include "tutti/data_paths/local_nvme/metadata/metadata_arena.h"

#include <tutti/cuda_like.h>
#include <nvm_types.h>
#include <nvm_dma.h>

#include "tutti/data_paths/local_nvme/io/submit_one.cuh"

#include <cstdio>
#include <cstring>

namespace tutti::data_paths::local_nvme {

// -----------------------------------------------------------------------
// Construction / destruction
// -----------------------------------------------------------------------

MetadataArena::~MetadataArena() {
    shutdown();
}

// -----------------------------------------------------------------------
// init: pre-allocate all workspace
// -----------------------------------------------------------------------

bool MetadataArena::init(const Config& cfg, nvm_ctrl_t* ctrl) {
    if (initialized_) return false;
    if (cfg.num_slots == 0 || cfg.max_entries_per_slot == 0 || ctrl == nullptr) {
        return false;
    }

    cfg_ = cfg;
    ctrl_ = ctrl;

    int prev_dev = -1;
    cudaError_t ce = cudaGetDevice(&prev_dev);
    if (ce != cudaSuccess) return false;
    ce = cudaSetDevice(cfg_.cuda_device);
    if (ce != cudaSuccess) return false;

    // 1. Pre-create events (one per slot).
    events_.resize(cfg_.num_slots, nullptr);
    for (std::uint32_t i = 0; i < cfg_.num_slots; ++i) {
        cudaEvent_t ev;
        ce = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (ce != cudaSuccess) {
            // Clean up events created so far.
            for (std::uint32_t j = 0; j < i; ++j) {
                cudaEventDestroy(static_cast<cudaEvent_t>(events_[j]));
                --alloc_counts_.cuda_event_destroy;
            }
            events_.clear();
            cudaSetDevice(prev_dev);
            return false;
        }
        events_[i] = ev;
        ++alloc_counts_.cuda_event_create;
    }

    // 2. Allocate entry pool (contiguous GPU buffer for all slots).
    std::size_t entry_pool_bytes = static_cast<std::size_t>(cfg_.num_slots) *
                                   cfg_.max_entries_per_slot *
                                   sizeof(DeviceSubmitEntry);
    ce = cudaMalloc(reinterpret_cast<void**>(&d_entries_pool_), entry_pool_bytes);
    if (ce != cudaSuccess) {
        for (auto& ev : events_) {
            cudaEventDestroy(static_cast<cudaEvent_t>(ev));
            ++alloc_counts_.cuda_event_destroy;
        }
        events_.clear();
        cudaSetDevice(prev_dev);
        return false;
    }
    ++alloc_counts_.cuda_malloc;

    // 3. Allocate status pool (contiguous GPU buffer for all slots).
    std::size_t status_pool_bytes = static_cast<std::size_t>(cfg_.num_slots) *
                                    cfg_.max_entries_per_slot *
                                    sizeof(EntryCompletionStatus);
    ce = cudaMalloc(reinterpret_cast<void**>(&d_status_pool_), status_pool_bytes);
    if (ce != cudaSuccess) {
        cudaFree(d_entries_pool_);
        d_entries_pool_ = nullptr;
        ++alloc_counts_.cuda_free;
        for (auto& ev : events_) {
            cudaEventDestroy(static_cast<cudaEvent_t>(ev));
            ++alloc_counts_.cuda_event_destroy;
        }
        events_.clear();
        cudaSetDevice(prev_dev);
        return false;
    }
    ++alloc_counts_.cuda_malloc;

    // 4. Allocate the GPU AddressDescriptor pool. PRP list backing is
    // host-pinned and owned outside the arena.
    //    entries.  The kernel ALWAYS reads prp1/prp2/data_length from
    //    e.prp_entry; for entries without a pre-built descriptor, the host
    //    writes the computed descriptor into this pool + H2D before launch.
    std::size_t desc_pool_bytes = static_cast<std::size_t>(cfg_.num_slots) *
                                   cfg_.max_entries_per_slot *
                                   sizeof(AddressDescriptor);
    ce = cudaMalloc(reinterpret_cast<void**>(&d_desc_pool_), desc_pool_bytes);
    if (ce != cudaSuccess) {
        cudaFree(d_status_pool_);
        d_status_pool_ = nullptr;
        ++alloc_counts_.cuda_free;
        cudaFree(d_entries_pool_);
        d_entries_pool_ = nullptr;
        ++alloc_counts_.cuda_free;
        for (auto& ev : events_) {
            cudaEventDestroy(static_cast<cudaEvent_t>(ev));
            ++alloc_counts_.cuda_event_destroy;
        }
        events_.clear();
        cudaSetDevice(prev_dev);
        return false;
    }
    ++alloc_counts_.cuda_malloc;

    cudaSetDevice(prev_dev);

    // 5. Populate free-list: all slots available.
    for (std::uint32_t i = 0; i < cfg_.num_slots; ++i) {
        free_list_.push_back(i);
    }

    initialized_ = true;
    return true;
}

// -----------------------------------------------------------------------
// shutdown: free all resources
// -----------------------------------------------------------------------

void MetadataArena::shutdown(bool skip_prp) {
    if (!initialized_) return;

    int prev_dev = -1;
    cudaGetDevice(&prev_dev);
    cudaSetDevice(cfg_.cuda_device);

    (void)skip_prp;
    // Events and entry/status pools are always safe to free — the kernel
    // has returned (caller synced all streams before calling shutdown).
    if (d_status_pool_) {
        cudaFree(d_status_pool_);
        d_status_pool_ = nullptr;
        ++alloc_counts_.cuda_free;
    }
    if (d_entries_pool_) {
        cudaFree(d_entries_pool_);
        d_entries_pool_ = nullptr;
        ++alloc_counts_.cuda_free;
    }
    if (d_desc_pool_) {
        cudaFree(d_desc_pool_);
        d_desc_pool_ = nullptr;
        ++alloc_counts_.cuda_free;
    }
    for (auto& ev : events_) {
        if (ev) {
            cudaEventDestroy(static_cast<cudaEvent_t>(ev));
            ++alloc_counts_.cuda_event_destroy;
        }
    }
    events_.clear();

    cudaSetDevice(prev_dev);

    free_list_.clear();
    initialized_ = false;
}

// -----------------------------------------------------------------------
// acquire: lease a slot (zero CUDA calls)
// -----------------------------------------------------------------------

bool MetadataArena::acquire(Lease& out) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (free_list_.empty()) {
        return false;
    }

    std::uint32_t slot = free_list_.front();
    free_list_.pop_front();

    out.slot_index = slot;
    out.event = events_[slot];
    out.d_entries = d_entries_pool_ + static_cast<std::size_t>(slot) * cfg_.max_entries_per_slot;
    out.d_status = d_status_pool_ + static_cast<std::size_t>(slot) * cfg_.max_entries_per_slot;
    out.d_desc_pool = d_desc_pool_ + static_cast<std::size_t>(slot) * cfg_.max_entries_per_slot;

    return true;
}

// -----------------------------------------------------------------------
// release: return a slot for reuse
// -----------------------------------------------------------------------

void MetadataArena::release(std::uint32_t slot_index) {
    std::lock_guard<std::mutex> lock(mtx_);
    free_list_.push_back(slot_index);
}

// -----------------------------------------------------------------------
// release_with_timeout_leak: permanently consume a slot
// -----------------------------------------------------------------------

void MetadataArena::release_with_timeout_leak(std::uint32_t slot_index) {
    // PRP list backing is now a host-pinned pool lease whose lifetime is
    // independent of this GPU metadata slot. The descriptor/status/event
    // workspace is safe to reuse after the submit kernel has returned.
    release(slot_index);
}

// -----------------------------------------------------------------------
// available: count free slots
// -----------------------------------------------------------------------

std::uint32_t MetadataArena::available() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return static_cast<std::uint32_t>(free_list_.size());
}

} // namespace tutti::data_paths::local_nvme

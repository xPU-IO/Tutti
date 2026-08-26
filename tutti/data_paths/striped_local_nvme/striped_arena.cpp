// tutti/data_paths/striped_local_nvme/striped_arena.cpp

#include "tutti/data_paths/striped_local_nvme/striped_arena.h"

#include <tutti/cuda_like.h>
#include <nvm_types.h>
#include <nvm_dma.h>

#include "tutti/data_paths/local_nvme/io/nvme_submit_primitives.cuh"
#include "tutti/data_paths/striped_local_nvme/fused_submit_kernel.cuh"

#include <cstdio>
#include <cstring>

namespace tutti::data_paths::striped_local_nvme {

using tutti::data_paths::local_nvme::EntryCompletionStatus;

StripedArena::~StripedArena() {
    shutdown();
}

bool StripedArena::init(const Config& cfg, const std::vector<nvm_ctrl_t*>& ctrls) {
    if (initialized_) return false;
    if (cfg.num_slots == 0 || cfg.max_entries_per_slot == 0 || ctrls.empty() ||
        cfg.dev_table_capacity_per_slot == 0) {
        return false;
    }
    for (auto* c : ctrls) {
        if (c == nullptr) return false;
    }

    cfg_ = cfg;
    ctrls_ = ctrls;

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
            for (std::uint32_t j = 0; j < i; ++j) {
                cudaEventDestroy(static_cast<cudaEvent_t>(events_[j]));
                ++alloc_counts_.cuda_event_destroy;
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
                                   sizeof(StripedDeviceSubmitEntry);
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

    // 4. Allocate device-table pool (contiguous GPU buffer for all slots).
    //    Each slot holds up to dev_table_capacity_per_slot DeviceTargetHandle*
    //    pointers; the kernel indexes it by entry.dev_idx.
    std::size_t dev_table_pool_bytes = static_cast<std::size_t>(cfg_.num_slots) *
                                       cfg_.dev_table_capacity_per_slot *
                                       sizeof(void*);
    ce = cudaMalloc(reinterpret_cast<void**>(&d_dev_table_pool_), dev_table_pool_bytes);
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

    // 5. GPU AddressDescriptor pool for dynamic-path entries.
    std::size_t desc_pool_bytes = static_cast<std::size_t>(cfg_.num_slots) *
                                   cfg_.max_entries_per_slot *
                                   sizeof(tutti::data_paths::local_nvme::AddressDescriptor);
    ce = cudaMalloc(reinterpret_cast<void**>(const_cast<tutti::data_paths::local_nvme::AddressDescriptor**>(&d_desc_pool_)), desc_pool_bytes);
    if (ce != cudaSuccess) {
        cudaFree(d_dev_table_pool_);
        d_dev_table_pool_ = nullptr;
        ++alloc_counts_.cuda_free;
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

    for (std::uint32_t i = 0; i < cfg_.num_slots; ++i) {
        free_list_.push_back(i);
    }

    initialized_ = true;
    return true;
}

void StripedArena::shutdown(bool skip_prp) {
    if (!initialized_) return;

    int prev_dev = -1;
    cudaGetDevice(&prev_dev);
    cudaSetDevice(cfg_.cuda_device);

    (void)skip_prp;
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
    if (d_dev_table_pool_) {
        cudaFree(d_dev_table_pool_);
        d_dev_table_pool_ = nullptr;
        ++alloc_counts_.cuda_free;
    }
    if (d_desc_pool_) {
        cudaFree(const_cast<tutti::data_paths::local_nvme::AddressDescriptor*>(d_desc_pool_));
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
    ctrls_.clear();
    initialized_ = false;
}

bool StripedArena::acquire(Lease& out) {
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
    out.d_dev_table = reinterpret_cast<const void**>(d_dev_table_pool_) +
                      static_cast<std::size_t>(slot) * cfg_.dev_table_capacity_per_slot;
    out.dev_table_capacity = cfg_.dev_table_capacity_per_slot;
    out.d_desc_pool = d_desc_pool_ +
                      static_cast<std::size_t>(slot) * cfg_.max_entries_per_slot;

    return true;
}

void StripedArena::release(std::uint32_t slot_index) {
    std::lock_guard<std::mutex> lock(mtx_);
    free_list_.push_back(slot_index);
}

void StripedArena::release_with_timeout_leak(std::uint32_t slot_index) {
    // Host-pinned PRP leases live outside the GPU metadata arena.
    release(slot_index);
}

std::uint32_t StripedArena::available() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return static_cast<std::uint32_t>(free_list_.size());
}

} // namespace tutti::data_paths::striped_local_nvme

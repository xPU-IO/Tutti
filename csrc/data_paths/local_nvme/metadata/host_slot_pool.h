#ifndef __TUTTI_MEMORY_HOST_SLOT_POOL_H__
#define __TUTTI_MEMORY_HOST_SLOT_POOL_H__

/**
 * host_slot_pool.h -- fixed-capacity pinned-host object pool (L2 tier).
 *
 * Layer: memory.  The CPU-side half of the two-tier handle cache (see
 * tiered_handle_cache.h): a large pinned-host array that holds fully-
 * built T templates that are NOT currently GPU-resident.  Backing
 * store for GpuSlotPool (L1, gpu_slot_pool.h) -- when a value needs
 * to move to the GPU, its slot here is the pinned SOURCE for the
 * H2D `cudaMemcpyAsync`, so L1 needs no staging buffer of its own.
 *
 * Why a bump allocator, not a general free-list:
 *
 *   L2 is sized to be large (e.g. 2 GiB / sizeof(NvmeFileDeviceHandle)
 *   ~= 11M slots) and, in the steady state, mostly monotonically
 *   growing -- entries are built once on COLD miss and then almost
 *   always just move between L1/L2, not actually evicted from L2.
 *   A general free-list needs a fast way to find a *contiguous* run
 *   of `count` free slots for batch acquisition (see
 *   acquire_batch()); scanning a fragmented free-list for that is
 *   O(capacity) and would defeat the point of batching.
 *
 *   So: new slots are always cut from the untouched "bump" region
 *   (O(1), trivially contiguous).  Only once the bump region is
 *   exhausted does a genuinely-freed slot (evicted L2 entry) get
 *   reused, and only for SINGLE-slot acquisition -- batch
 *   acquisition after the bump region is exhausted falls back to
 *   the caller doing multiple single acquires (rare path, since L2
 *   eviction itself should be rare given the large budget).
 *
 * Thread-safety: acquire/acquire_batch/release lock an internal
 * mutex; this is a CPU-only pool (no CUDA calls except the one-time
 * cudaMallocHost/cudaFreeHost in init/shutdown), so no async
 * complexity here -- unlike L1, an L2 slot is never being read by a
 * GPU kernel, so release() can free it for reuse immediately.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <vector>

#include <tutti/cuda_like.h>   // cudaMallocHost / cudaFreeHost

namespace tutti::data_paths::local_nvme {

template <typename T>
class HostSlotPool {
public:
    HostSlotPool() = default;
    ~HostSlotPool() { shutdown(); }

    HostSlotPool(const HostSlotPool&)            = delete;
    HostSlotPool& operator=(const HostSlotPool&) = delete;

    /// One-time setup: cudaMallocHost(capacity * sizeof(T)).  Returns
    /// false on failure.
    bool init(uint32_t capacity) {
        if (capacity == 0) {
            std::fprintf(stderr, "[host_slot_pool] init: capacity == 0\n");
            return false;
        }
        cudaError_t cerr = cudaMallocHost((void**)&pool_, (std::size_t)capacity * sizeof(T));
        if (cerr != cudaSuccess) {
            std::fprintf(stderr,
                "[host_slot_pool] init: cudaMallocHost(%zu B) failed: %s\n",
                (std::size_t)capacity * sizeof(T), cudaGetErrorString(cerr));
            return false;
        }
        capacity_  = capacity;
        next_bump_ = 0;
        free_list_.clear();
        return true;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (pool_ != nullptr) {
            cudaFreeHost(pool_);
            pool_ = nullptr;
        }
        capacity_ = 0;
        next_bump_ = 0;
        free_list_.clear();
    }

    bool initialized() const { return pool_ != nullptr; }
    uint32_t capacity() const { return capacity_; }

    /// Slots currently occupied (bump region used + reused free-list
    /// entries).  Diagnostic / memory-accounting only.
    uint32_t in_use() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return next_bump_ - (uint32_t)free_list_.size();
    }

    /// Acquire ONE slot, copy `value` into it, return its index.
    /// Returns UINT32_MAX if the pool is full (bump region exhausted
    /// AND free-list empty).
    uint32_t acquire(const T& value) {
        std::lock_guard<std::mutex> lock(mtx_);
        uint32_t idx;
        if (!free_list_.empty()) {
            idx = free_list_.front();
            free_list_.pop_front();
        } else if (next_bump_ < capacity_) {
            idx = next_bump_++;
        } else {
            return UINT32_MAX;
        }
        pool_[idx] = value;
        return idx;
    }

    /// Acquire a CONTIGUOUS run of `count` slots from the untouched
    /// bump region and fill them with `values[0..count)`.  Fast path
    /// (O(1)): only succeeds while the bump region has room; once
    /// exhausted, returns false and the caller should fall back to
    /// `count` individual `acquire()` calls (each of which may land
    /// anywhere, including reused free-list slots).
    ///
    /// This asymmetry (contiguous only from bump, not from
    /// fragmented free space) is deliberate -- see the file comment.
    bool acquire_batch_contig(const T* values, uint32_t count, uint32_t* out_start) {
        if (count == 0) { *out_start = 0; return true; }
        std::lock_guard<std::mutex> lock(mtx_);
        if (next_bump_ + count > capacity_) return false;
        const uint32_t start = next_bump_;
        for (uint32_t i = 0; i < count; ++i) pool_[start + i] = values[i];
        next_bump_ += count;
        *out_start = start;
        return true;
    }

    /// Release a single slot for reuse.  Safe to call any time --
    /// this pool is never touched by a GPU kernel, so no stream
    /// ordering is needed (unlike GpuSlotPool::release_async).
    void release(uint32_t idx) {
        std::lock_guard<std::mutex> lock(mtx_);
        free_list_.push_back(idx);
    }

    T*       slot_ptr(uint32_t idx)       { return pool_ + idx; }
    const T* slot_ptr(uint32_t idx) const { return pool_ + idx; }

private:
    mutable std::mutex   mtx_;
    T*                    pool_      = nullptr;
    uint32_t              capacity_  = 0;
    uint32_t              next_bump_ = 0;    // first never-yet-used index
    std::deque<uint32_t>  free_list_;         // reclaimed (evicted) indices
};

} // namespace tutti

#endif // __TUTTI_MEMORY_HOST_SLOT_POOL_H__

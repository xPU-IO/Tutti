#ifndef __TUTTI_MEMORY_GPU_SLOT_POOL_H__
#define __TUTTI_MEMORY_GPU_SLOT_POOL_H__

/**
 * gpu_slot_pool.h -- fixed-capacity, stream-ordered GPU object pool
 * (L1 tier of the two-tier handle cache -- see tiered_handle_cache.h).
 *
 * Layer: memory.
 *
 * Release semantics: release() is CPU-side immediate -- a freed slot
 * reappears in the free list synchronously, so a subsequent acquire
 * in the SAME call can reuse it right away.  This is required by
 * TieredHandleCache's "evict one LRU slot, then immediately reuse it"
 * pattern, which has no intervening sync point to hang a deferred
 * (host-callback-based) release off of.  GPU-side correctness is
 * guaranteed separately, by an event fence, so a slot's previous
 * contents are never overwritten by a new fill before the GPU has
 * actually finished reading them.
 *
 * Event granularity: ONE event per PRODUCER STREAM, not one per slot.
 * This pool is sized to a memory BUDGET (see coordinator_config.h's
 * handle_l1_gpu_budget_bytes), which at the default 512 MiB / ~200
 * B-per-slot payload works out to capacity in the HUNDREDS OF
 * THOUSANDS to LOW MILLIONS -- creating one `cudaEvent` per slot
 * would be a real, measurable startup cost, and recording one per
 * slot on every batch fill/release would be pure waste, since a
 * whole batch is filled by ONE `cudaMemcpyAsync` and typically
 * consumed by ONE downstream kernel/transfer call: there is only
 * ever ONE fence point that matters per batch operation, not one per
 * slot in it.
 *
 *   Each slot just remembers WHICH STREAM last touched it
 *   (`last_touch_stream_[idx]`); a small `stream -> cudaEvent_t` map
 *   (`stream_events_`, lazily populated, one entry per DISTINCT
 *   stream actually used -- typically a handful in real deployments,
 *   never anywhere near `capacity`) supplies the shared event to
 *   record/wait on.  A batch fill/release does exactly ONE
 *   `cudaEventRecord` (on the batch's single `stream` parameter) no
 *   matter how many slots are in the batch.
 *
 *   Correctness of sharing one event across many unrelated touches to
 *   the SAME stream: `cudaEventRecord` on an already-used event just
 *   re-targets it to a LATER point in that stream's queue.  Any
 *   `cudaStreamWaitEvent` issued afterwards (in host call order) waits
 *   for that (possibly advanced) later point -- which, on the SAME
 *   producer stream, strictly implies every earlier point (including
 *   whatever this slot's own last touch was) has already retired.  So
 *   sharing can only make a wait MORE conservative (wait for a bit
 *   more of that stream's later work than strictly necessary), never
 *   LESS correct -- under-synchronization is the only real bug class
 *   here, and this design cannot produce one.
 *
 *   Trade-off worth knowing: if ONE producer stream is reused very
 *   heavily by many logically-unrelated fills, a cross-stream
 *   `wait_ready()` for a slot filled long ago on that stream will wait
 *   for everything queued on it since, not just that slot's own fill.
 *   This matches the codebase's existing convention (see
 *   coordinator.h/tiered_handle_cache.h): reuse the SAME stream for
 *   producing and consuming a given piece of data; cross-stream
 *   consumption of data from a stream that's ALSO busy with a lot of
 *   unrelated work is the case this is conservative about. Callers
 *   with that shape should use dedicated streams per logical
 *   producer, same advice CUDA itself gives for any shared-resource
 *   cross-stream reuse.
 *
 *   `stream_events_` grows by one entry per NEW distinct stream ever
 *   passed to this pool and is never shrunk before shutdown() --
 *   fine for the expected usage (a bounded, long-lived set of
 *   caller-owned streams); a caller that created a fresh,
 *   never-reused stream per call would leak one event per such call
 *   until shutdown().
 *
 * Batch acquire (`acquire_batch_from_host_async`): ONE
 * `cudaMemcpyAsync` for a whole contiguous run of slots instead of
 * one call per item -- the fix for "one IO batch launching thousands
 * of copies."  Falls back to the caller doing single acquires when
 * the bump region can't satisfy a contiguous run.
 *
 * Thread-safety: acquire/release lock an internal mutex around
 * free-list/bump bookkeeping only (O(1) each, or O(1) amortised for
 * the batch path); a separate small mutex guards `stream_events_`
 * lookups/inserts.  The CUDA calls themselves are not under either
 * lock.
 *
 * Lifetime: init() once at bootstrap; shutdown() once at teardown.
 * shutdown() assumes the caller has already drained/synced any
 * outstanding stream work touching pool slots.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <tutti/cuda_like.h>

namespace tutti::data_paths::local_nvme {

template <typename T>
class GpuSlotPool {
public:
    GpuSlotPool() = default;
    ~GpuSlotPool() { shutdown(); }

    GpuSlotPool(const GpuSlotPool&)            = delete;
    GpuSlotPool& operator=(const GpuSlotPool&) = delete;

    /// One-time setup: cudaMalloc(capacity * sizeof(T)) on
    /// `cuda_device`.  Events are created lazily, one per DISTINCT
    /// stream actually used (see the file comment above) -- none are
    /// created here.  Returns false on any CUDA failure.
    bool init(uint32_t capacity, int cuda_device) {
        if (capacity == 0) {
            std::fprintf(stderr, "[gpu_slot_pool] init: capacity == 0\n");
            return false;
        }
        int prev_dev = -1;
        cudaError_t cerr = cudaGetDevice(&prev_dev);
        if (cerr != cudaSuccess) {
            std::fprintf(stderr, "[gpu_slot_pool] init: cudaGetDevice: %s\n",
                        cudaGetErrorString(cerr));
            return false;
        }
        cerr = cudaSetDevice(cuda_device);
        if (cerr != cudaSuccess) {
            std::fprintf(stderr, "[gpu_slot_pool] init: cudaSetDevice(%d): %s\n",
                        cuda_device, cudaGetErrorString(cerr));
            return false;
        }

        cerr = cudaMalloc((void**)&d_pool_, (std::size_t)capacity * sizeof(T));
        if (cerr != cudaSuccess) {
            std::fprintf(stderr,
                "[gpu_slot_pool] init: cudaMalloc(%zu B) failed: %s\n",
                (std::size_t)capacity * sizeof(T), cudaGetErrorString(cerr));
            cudaSetDevice(prev_dev);
            return false;
        }

        // Per-slot bookkeeping only -- NO events allocated here (see
        // the file comment above); `last_touch_stream_[i]` is
        // meaningless until `ever_recorded_[i]` is set.
        last_touch_stream_.assign(capacity, nullptr);
        ever_recorded_.assign(capacity, 0);

        cuda_device_ = cuda_device;
        capacity_    = capacity;
        next_bump_   = 0;
        free_list_.clear();

        cudaSetDevice(prev_dev);
        return true;
    }

    /// Release the pool's own GPU memory + every lazily-created
    /// per-stream event.  Caller MUST have already ensured no
    /// in-flight stream work still touches any slot (e.g. via
    /// cudaStreamSynchronize on every stream ever passed to this
    /// pool).
    void shutdown() {
        std::lock_guard<std::mutex> lock(mtx_);
        int prev_dev = -1;
        const bool have_device_state = d_pool_ != nullptr || !stream_events_.empty();
        if (have_device_state) {
            cudaGetDevice(&prev_dev);
            cudaSetDevice(cuda_device_);
        }
        {
            std::lock_guard<std::mutex> elock(stream_events_mtx_);
            for (auto& kv : stream_events_) cudaEventDestroy(kv.second);
            stream_events_.clear();
        }
        last_touch_stream_.clear();
        ever_recorded_.clear();
        if (d_pool_ != nullptr) {
            cudaFree(d_pool_);
            d_pool_ = nullptr;
        }
        if (have_device_state) cudaSetDevice(prev_dev);
        free_list_.clear();
        capacity_  = 0;
        next_bump_ = 0;
    }

    bool initialized() const { return d_pool_ != nullptr; }
    uint32_t capacity() const { return capacity_; }

    /// Raw slot access by index -- used by TieredHandleCache to
    /// recover a pointer from a stored index without re-deriving it
    /// from a stashed T* every time.
    T*       slot_ptr(uint32_t idx)       { return d_pool_ + idx; }
    const T* slot_ptr(uint32_t idx) const { return d_pool_ + idx; }

    /// True iff `ptr` falls inside this pool's device-resident array.
    bool owns(const T* ptr) const {
        return d_pool_ != nullptr && ptr >= d_pool_ && ptr < d_pool_ + capacity_;
    }

    /// Slots currently checked out (capacity - free).  Diagnostic /
    /// memory-accounting only.
    uint32_t in_use() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return (capacity_ - next_bump_) == 0 && free_list_.empty()
             ? capacity_
             : capacity_ - (uint32_t)free_list_.size() - (capacity_ - next_bump_);
    }

    /// Reserve ONE free slot WITHOUT filling it -- for callers that
    /// need to write their own pinned staging in between (e.g.
    /// block_storage's ShardPtrSlot pool, which has no L2 tier of its
    /// own to serve as the pinned source).  If the slot is being
    /// REUSED, fences the caller's future writes to it behind that
    /// content's last-touch stream's shared event (same-stream: free
    /// no-op; cross-stream: a real fence) -- call this BEFORE writing
    /// your pinned staging buffer's content for the upcoming
    /// cudaMemcpyAsync, so the fence precedes it in submission order.
    /// Pair with `mark_filled()` once your own cudaMemcpyAsync is
    /// queued.  Returns nullptr if the pool is exhausted.
    T* reserve_slot(cudaStream_t stream, uint32_t* out_idx) {
        uint32_t idx; bool reused;
        if (!take_free_slot_(&idx, &reused)) return nullptr;
        if (reused && ever_recorded_[idx]) {
            cudaEvent_t e = event_for_stream_(last_touch_stream_[idx]);
            if (e != nullptr) cudaStreamWaitEvent(stream, e, 0);
        }
        if (out_idx != nullptr) *out_idx = idx;
        return d_pool_ + idx;
    }

    /// Re-records slot `idx`'s producer-stream event on `stream` --
    /// call this right after queuing your own `cudaMemcpyAsync` into
    /// a slot obtained from `reserve_slot()`, so the NEXT reuse of
    /// this slot fences correctly against it (mirrors what
    /// `acquire_one_from_host_async` does internally after its own
    /// copy).
    void mark_filled(uint32_t idx, cudaStream_t stream) {
        cudaEvent_t e = event_for_stream_(stream);
        if (e != nullptr) cudaEventRecord(e, stream);
        last_touch_stream_[idx] = stream;
        ever_recorded_[idx] = 1;
    }

    /// Borrow ONE free slot and queue an async H2D copy of
    /// `*host_pinned_src` (must be pinned -- typically a
    /// HostSlotPool<T> slot) on `stream`.  If the slot is being
    /// REUSED (previously held content), fences the copy behind that
    /// content's last-touch stream's shared event first (see the file
    /// comment) -- a same-stream fence is a free no-op; a cross-stream
    /// one is a real, correct synchronization.  Re-records the shared
    /// event for `stream` after the copy.  Returns the device pointer
    /// immediately; still call `wait_ready()` before any GPU work that
    /// reads it queues on a DIFFERENT stream than `stream`.
    ///
    /// Returns nullptr if the pool is exhausted.
    T* acquire_one_from_host_async(const T* host_pinned_src, cudaStream_t stream) {
        uint32_t idx; bool reused;
        if (!take_free_slot_(&idx, &reused)) return nullptr;

        T* slot = d_pool_ + idx;
        if (reused && ever_recorded_[idx]) {
            cudaEvent_t e = event_for_stream_(last_touch_stream_[idx]);
            if (e != nullptr) cudaStreamWaitEvent(stream, e, 0);
        }
        cudaError_t cerr = cudaMemcpyAsync(slot, host_pinned_src, sizeof(T),
                                          cudaMemcpyHostToDevice, stream);
        if (cerr != cudaSuccess) {
            std::fprintf(stderr,
                "[gpu_slot_pool] acquire_one_from_host_async: cudaMemcpyAsync "
                "failed: %s (returning slot %u)\n", cudaGetErrorString(cerr), idx);
            return_free_slot_(idx);
            return nullptr;
        }
        cudaEvent_t e = event_for_stream_(stream);
        if (e != nullptr) cudaEventRecord(e, stream);
        last_touch_stream_[idx] = stream;
        ever_recorded_[idx] = 1;
        return slot;
    }

    /// Borrow a CONTIGUOUS run of `count` free slots (fast path: only
    /// succeeds while the bump region has room) and queue ONE async
    /// H2D copy of `host_pinned_src_contig[0..count)` on `stream`.
    /// Bump-region slots are, by definition, never reused, so no
    /// per-slot fence is needed here (only `acquire_one_from_host_async`
    /// deals with reused slots -- batch callers that need to reuse
    /// freed slots should fall back to individual acquires, same as
    /// the "bump exhausted" fallback already documented at the call
    /// sites in tiered_handle_cache.h).  Records the shared event for
    /// `stream` exactly ONCE for the whole batch (see the file
    /// comment above), then stamps every slot with `stream` as its
    /// last-touch.
    ///
    /// Returns false (no slots taken) if the bump region can't
    /// satisfy `count` contiguously.
    bool acquire_batch_from_host_async(const T*      host_pinned_src_contig,
                                       uint32_t      count,
                                       cudaStream_t  stream,
                                       T**           out_start) {
        if (count == 0) { *out_start = d_pool_; return true; }
        uint32_t start_idx;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (next_bump_ + count > capacity_) return false;
            start_idx = next_bump_;
            next_bump_ += count;
        }
        T* start_ptr = d_pool_ + start_idx;
        cudaError_t cerr = cudaMemcpyAsync(start_ptr, host_pinned_src_contig,
                                          (std::size_t)count * sizeof(T),
                                          cudaMemcpyHostToDevice, stream);
        if (cerr != cudaSuccess) {
            std::fprintf(stderr,
                "[gpu_slot_pool] acquire_batch_from_host_async: cudaMemcpyAsync "
                "(%u items) failed: %s\n", count, cudaGetErrorString(cerr));
            std::lock_guard<std::mutex> lock(mtx_);
            for (uint32_t i = 0; i < count; ++i) free_list_.push_back(start_idx + i);
            return false;
        }
        cudaEvent_t e = event_for_stream_(stream);
        if (e != nullptr) cudaEventRecord(e, stream);
        for (uint32_t i = 0; i < count; ++i) {
            last_touch_stream_[start_idx + i] = stream;
            ever_recorded_[start_idx + i] = 1;
        }
        *out_start = start_ptr;
        return true;
    }

    /// Insert a `cudaStreamWaitEvent` so `consumer_stream` won't run
    /// anything queued after this call until `slot`'s most recent
    /// touch (fill or release) has completed.  Safe/cheap to call
    /// unconditionally -- including when `consumer_stream` IS the
    /// stream that last touched the slot (a redundant but harmless
    /// wait on an already-queued event in the same stream).  No-op
    /// (with a diagnostic) if `slot` doesn't belong to this pool.
    void wait_ready(const T* slot, cudaStream_t consumer_stream) {
        if (slot == nullptr) return;
        if (!owns(slot)) {
            std::fprintf(stderr,
                "[gpu_slot_pool] wait_ready: slot %p not owned by this pool\n",
                (const void*)slot);
            return;
        }
        const uint32_t idx = (uint32_t)(slot - d_pool_);
        if (!ever_recorded_[idx]) return;
        cudaEvent_t e = event_for_stream_(last_touch_stream_[idx]);
        if (e != nullptr) cudaStreamWaitEvent(consumer_stream, e, 0);
    }

    /// Release `slot` for reuse.  CPU-side IMMEDIATE (the slot
    /// reappears in the free list synchronously -- safe to
    /// evict-then-reacquire in a tight CPU loop with no intervening
    /// GPU sync).  Re-records the shared event for `stream` so the
    /// NEXT acquire that reuses this slot fences correctly against
    /// whatever was queued on `stream` up to and including this call
    /// (see the file comment for why this is both correct and, in
    /// the common same-stream case, free).  No-op (with a
    /// diagnostic) if `slot` doesn't belong to this pool.
    void release_async(T* slot, cudaStream_t stream) {
        if (slot == nullptr) return;
        if (!owns(slot)) {
            std::fprintf(stderr,
                "[gpu_slot_pool] release_async: slot %p not owned by this "
                "pool (device pool range invalid)\n", (const void*)slot);
            return;
        }
        const uint32_t idx = (uint32_t)(slot - d_pool_);
        cudaEvent_t e = event_for_stream_(stream);
        if (e != nullptr) cudaEventRecord(e, stream);
        last_touch_stream_[idx] = stream;
        ever_recorded_[idx] = 1;
        std::lock_guard<std::mutex> lock(mtx_);
        free_list_.push_back(idx);
    }

    /// Batch release: same immediate-CPU-side-effect, event-fenced
    /// semantics as `release_async`, for `count` slots -- but ONE
    /// shared-event record for the whole batch (see the file comment
    /// above) instead of `count`, plus ONE lock acquisition for the
    /// free-list update.
    void release_batch_async(T* const* slots, uint32_t count, cudaStream_t stream) {
        if (count == 0) return;
        cudaEvent_t e = event_for_stream_(stream);
        if (e != nullptr) cudaEventRecord(e, stream);
        std::vector<uint32_t> indices;
        indices.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            if (slots[i] == nullptr || !owns(slots[i])) continue;
            const uint32_t idx = (uint32_t)(slots[i] - d_pool_);
            last_touch_stream_[idx] = stream;
            ever_recorded_[idx] = 1;
            indices.push_back(idx);
        }
        if (indices.empty()) return;
        std::lock_guard<std::mutex> lock(mtx_);
        for (uint32_t idx : indices) free_list_.push_back(idx);
    }

private:
    // Free-list first (reclaimed slots -- these are "reused"), else
    // cut a fresh index from the untouched bump region (never
    // "reused" -- no fence needed for these).
    bool take_free_slot_(uint32_t* out_idx, bool* out_reused) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!free_list_.empty()) {
            *out_idx = free_list_.front();
            free_list_.pop_front();
            *out_reused = true;
            return true;
        }
        if (next_bump_ < capacity_) {
            *out_idx = next_bump_++;
            *out_reused = false;
            return true;
        }
        return false;
    }
    void return_free_slot_(uint32_t idx) {
        std::lock_guard<std::mutex> lock(mtx_);
        free_list_.push_back(idx);
    }

    // Get-or-create the shared event for `stream` (one event
    // per DISTINCT stream ever used by this pool, not one per slot --
    // see the file comment above).  Returns nullptr (with a diagnostic) on
    // CUDA failure; callers treat that as "skip the record/wait"
    // (matches this pool's existing best-effort diagnostic style for
    // CUDA call failures elsewhere).
    cudaEvent_t event_for_stream_(cudaStream_t stream) {
        {
            std::lock_guard<std::mutex> lock(stream_events_mtx_);
            auto it = stream_events_.find(stream);
            if (it != stream_events_.end()) return it->second;
        }
        cudaEvent_t e = nullptr;
        cudaError_t cerr = cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        if (cerr != cudaSuccess) {
            std::fprintf(stderr,
                "[gpu_slot_pool] event_for_stream_: cudaEventCreateWithFlags "
                "failed: %s\n", cudaGetErrorString(cerr));
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(stream_events_mtx_);
        auto ins = stream_events_.emplace(stream, e);
        if (!ins.second) {
            // Another thread created it between our check above and
            // now -- drop ours, use theirs.
            cudaEventDestroy(e);
            return ins.first->second;
        }
        return e;
    }

    mutable std::mutex          mtx_;
    T*                          d_pool_      = nullptr;
    // Per-slot: which stream last touched it (fill or release), and
    // whether it's ever been touched at all.  `last_touch_stream_[i]`
    // is only meaningful when `ever_recorded_[i]` is set -- a stream
    // value of nullptr is itself a legitimate stream (the default/
    // legacy stream), so it cannot double as an "unset" sentinel.
    std::vector<cudaStream_t>   last_touch_stream_;  // capacity_ entries
    std::vector<uint8_t>        ever_recorded_;       // capacity_ entries
    int                         cuda_device_  = -1;
    uint32_t                    capacity_     = 0;
    uint32_t                    next_bump_    = 0;   // first never-yet-used index
    std::deque<uint32_t>        free_list_;           // reclaimed (released) indices

    // Shared per-producer-stream events, lazily created --
    // bounded by the number of DISTINCT streams actually used, not by
    // `capacity_`.  Separate mutex from `mtx_` (guards unrelated
    // free-list/bump bookkeeping) so a stream_events_ lookup never
    // contends with that hot path.
    std::mutex                                     stream_events_mtx_;
    std::unordered_map<cudaStream_t, cudaEvent_t>  stream_events_;
};

} // namespace tutti

#endif // __TUTTI_MEMORY_GPU_SLOT_POOL_H__

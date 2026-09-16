#pragma once

#include "csrc/data_paths/local_nvme/io/device_target.h"

// tutti/data_paths/local_nvme/metadata/handle_workspace_cache.h
//
// Two-tier GPU LRU cache for device target handles (DeviceTargetHandle +
// overflow extents).  Round 16 S6b: L2 host-pinned content backup added
// (aligned to legacy TieredHandleCache semantics).
//
// L1 = GPU-resident entry (DeviceTargetHandle* + overflow*).
// L2 = host-pinned content copy (handle struct bytes + overflow bytes).
//
// Eviction (downgrade): copy handle+overflow content to L2 host buffer,
//   free GPU allocations.  The L2 entry is retained.
// Miss with L2 hit (promote): allocate new GPU slot, memcpy content from
//   L2 host buffer back to GPU.  One memcpy, NOT a full rebuild.
// Miss with L2 miss (cold): call build_fn to create from scratch.
//
// Design rationale (full evaluation in chat/round11/result2.md):
//   The legacy TieredHandleCache (memory/include/tiered_handle_cache.h, 644
//   lines) provides two tiers (L1 GPU / L2 host-pinned), inclusive LRU, batch
//   promote, and stream-fenced slot reuse via GpuSlotPool.  For the current
//   DataPath this is over-engineered:
//     - L2 host template is already LocalNvmeTargetState in the targets_ map.
//     - Batch promote targets "thousands of new files per IO batch" — the
//       current DataPath's batch is per-op (all entries target the same file).
//     - Stream-fenced slot reuse assumes multi-stream concurrent access to
//       the same cache slot — the DataPath is single-threaded for open/close.
//   A simple single-tier (GPU only) LRU with explicit pin/unpin is sufficient.
//
// Pin semantics:
//   submit() pins the target's cache entry; release() unpins.
//   Pin count > 0 ⇒ entry is protected from eviction.
//   Pin count == 0 ⇒ entry is evictable (LRU candidate).
//
// Open-refcount semantics (Round 16 P0-1 fix):
//   open() increments open_refcount on hit AND miss; close() decrements.
//   An entry is evictable only when pin_count == 0 AND open_refcount == 0.
//   This prevents the reopen→eviction UAF: after close(A)→open(A), the
//   reopened entry has open_refcount > 0, so a subsequent open(B) cannot
//   evict it even though the old close already set in_use=false.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "csrc/data_paths/local_nvme/metadata/host_slot_pool.h"

namespace tutti::data_paths::local_nvme {

// Round 16 S6b: L2 record inline overflow blob capacity.  Records whose
// overflow exceeds this are not saved to L2 (eviction = genuine delete,
// next open = cold rebuild).  4096 B = 256 overflow extents — far beyond
// the chunked-tensor working set (typically 1 extent per file).
inline constexpr std::uint32_t kL2OverflowBlobBytes = 4096;

class HandleWorkspaceCache {
public:
    struct Config {
        std::uint32_t capacity = 0;     // L1 capacity (0 = disabled)
        std::uint32_t l2_capacity = 0;  // L2 slots (0 = single-tier)
        std::uint32_t cuda_device = 0;
    };

    struct Stats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t evictions = 0;    // L1 evictions (downgrades when L2 on)
        std::uint32_t pinned = 0;       // current total pin count
        std::uint32_t entries = 0;      // current L1 entry count
        // Round 16 S6b: L2 stats (legacy tiered_handle_cache naming)
        std::uint64_t l2_hits = 0;      // L2-resident, promoted to L1 (memcpy restore)
        std::uint64_t l2_misses = 0;    // not in L2 either (cold, build_fn called)
        std::uint64_t l2_evictions = 0; // L2 genuine delete (content dropped)
    };

    // Entry: owns DeviceTargetHandle* + overflow GPU buffer.
    // Returned as a borrowed pointer; caller must NOT free handle/overflow.
    struct Entry {
        std::uint64_t key = 0;
        DeviceTargetHandle* handle = nullptr;
        void* overflow = nullptr;
        std::uint64_t overflow_bytes = 0;  // Round 16 S6b: for L2 save/restore
        std::uint32_t pin_count = 0;
        std::uint32_t open_refcount = 0;  // >0 while one or more targets reference this entry
        bool in_use = false;  // deprecated: kept for backward compat, superseded by open_refcount
    };

    HandleWorkspaceCache() = default;
    ~HandleWorkspaceCache() { shutdown(); }

    HandleWorkspaceCache(const HandleWorkspaceCache&) = delete;
    HandleWorkspaceCache& operator=(const HandleWorkspaceCache&) = delete;

    // Pre-allocate the entry pool.  Does NOT allocate GPU memory per entry —
    // GPU allocations happen lazily on get_or_build() misses.
    bool init(const Config& cfg) {
        if (cfg.capacity == 0) { cfg_ = cfg; return true; }
        if (cfg.l2_capacity > 0) {
            if (!l2_pool_.init(cfg.l2_capacity)) return false;
            l2_capacity_ = cfg.l2_capacity;
        }
        entries_.resize(cfg.capacity);
        free_list_.clear();
        for (std::uint32_t i = 0; i < cfg.capacity; ++i)
            free_list_.push_back(i);
        index_.clear();
        lru_.clear();
        lru_pos_.clear();
        stats_ = {};
        cfg_ = cfg;
        initialized_ = true;
        return true;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!initialized_) return;
        // Free all GPU allocations owned by entries.
        // P0-1: entries with open_refcount > 0 are conservatively retained
        // (handle not destroyed), matching the timeout-leak semantics —
        // a target may still be in use by an open handle.
        for (auto& e : entries_) {
            if (e.open_refcount > 0) {
                // Leak: don't destroy handle, just clear bookkeeping.
                continue;
            }
            if (e.handle != nullptr || e.overflow != nullptr) {
                free_entry_gpu_(e);
            }
            e = {};
        }
        entries_.clear();
        free_list_.clear();
        index_.clear();
        lru_.clear();
        lru_pos_.clear();
        l2_index_.clear();
        l2_lru_.clear();
        l2_pos_.clear();
        l2_pool_.shutdown();
        l2_capacity_ = 0;
        stats_ = {};
        initialized_ = false;
    }

    bool enabled() const { return cfg_.capacity > 0 && initialized_; }
    std::uint32_t capacity() const { return cfg_.capacity; }

    // Get or build a handle entry.
    // On L1 hit: returns cached Entry* (no GPU alloc / H2D).
    // On L1 miss + L2 hit: promotes from L2 — memcpy restore of the saved
    //   handle struct + overflow content (restore_fn), NOT a rebuild.
    // On cold miss: calls build_fn(handle_out, overflow_out,
    //   overflow_bytes_out) to create the GPU handle, then admits a
    //   content snapshot to L2 (downgrade backing store).
    // If the L1 pool is full, evicts the LRU unpinned entry first —
    // eviction is a DOWNGRADE when L2 is enabled (content retained in
    // L2), a genuine delete otherwise.
    // Returns nullptr on failure (build_fn failed or pool exhausted with
    //   nothing evictable).
    template <typename BuildFn>
    Entry* get_or_build(std::uint64_t key, BuildFn&& build_fn) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!enabled()) return nullptr;

        auto it = index_.find(key);
        if (it != index_.end()) {
            std::uint32_t slot = it->second;
            Entry& hit = entries_[slot];
            ++hit.open_refcount;  // P0-1: reopen increments refcount
            hit.in_use = true;    // backward compat
            // Remove from LRU if present — open entries are not evictable.
            remove_from_lru_(slot);
            ++stats_.hits;
            return &hit;
        }

        ++stats_.misses;

        std::uint32_t slot = acquire_slot_();
        if (slot == UINT32_MAX) return nullptr;

        Entry& e = entries_[slot];
        e.key = key;
        e.pin_count = 0;
        e.open_refcount = 1;  // P0-1: new entry starts with one open reference
        e.in_use = true;  // backward compat
        e.overflow_bytes = 0;

        // L2 hit → promote (memcpy restore).  The L2 record is RETAINED
        // (inclusive tiering: it is L1's backing store) and leaves the
        // L2-eviction LRU while L1-resident.
        if (l2_enabled()) {
            auto l2it = l2_index_.find(key);
            if (l2it != l2_index_.end()) {
                L2RecordSlot* rec = l2_pool_.slot_ptr(l2it->second);
                if (restore_fn_ != nullptr &&
                    restore_fn_(&rec->handle_image, rec->overflow_blob,
                                rec->overflow_bytes, cfg_.cuda_device,
                                &e.handle, &e.overflow)) {
                    e.overflow_bytes = rec->overflow_bytes;
                    remove_from_l2_lru_(key);
                    index_[key] = slot;
                    ++stats_.l2_hits;
                    ++stats_.entries;
                    return &e;
                }
                // Restore failed — drop the stale record, fall through
                // to cold build.
                l2_pool_.release(l2it->second);
                remove_from_l2_lru_(key);
                l2_index_.erase(l2it);
            }
            ++stats_.l2_misses;
        }

        if (!build_fn(&e.handle, &e.overflow, &e.overflow_bytes)) {
            // Build failed: return slot to free list.
            free_list_.push_back(slot);
            e = {};
            return nullptr;
        }
        // Cold build succeeded — admit content snapshot to L2 (this is
        // the ONLY place content enters L2 besides a first-time
        // downgrade of an entry that skipped admission).
        if (l2_enabled()) admit_to_l2_(key, e);
        index_[key] = slot;
        // Do NOT add to LRU yet — entry is in_use (target open).
        // release_entry() adds it to LRU when the target is closed.
        ++stats_.entries;
        return &e;
    }

    // Release an entry: decrement open_refcount (called by close()).
    // The entry becomes evictable (added to LRU) when open_refcount reaches 0
    // AND pin_count is 0.
    void release_entry(Entry* e) {
        if (!e) return;
        std::lock_guard<std::mutex> lock(mtx_);
        if (e->open_refcount > 0) --e->open_refcount;
        e->in_use = (e->open_refcount > 0);  // backward compat
        if (e->open_refcount == 0 && e->pin_count == 0) {
            std::uint32_t slot = static_cast<std::uint32_t>(e - entries_.data());
            if (index_.count(e->key) && !lru_pos_.count(slot)) {
                lru_.push_front(slot);
                lru_pos_[slot] = lru_.begin();
            }
        }
    }

    // Pin: protects the entry from eviction.  Called by submit().
    void pin(Entry* e) {
        if (!e) return;
        std::lock_guard<std::mutex> lock(mtx_);
        ++e->pin_count;
        ++stats_.pinned;
        // Remove from LRU (pinned entries are not evictable).
        auto it = lru_pos_.find(static_cast<std::uint32_t>(e - entries_.data()));
        if (it != lru_pos_.end()) {
            lru_.erase(it->second);
            lru_pos_.erase(it);
        }
    }

    // Unpin: makes the entry evictable again (if open_refcount is also 0).
    // Called by release().
    void unpin(Entry* e) {
        if (!e) return;
        std::lock_guard<std::mutex> lock(mtx_);
        if (e->pin_count > 0) {
            --e->pin_count;
            --stats_.pinned;
        }
        if (e->pin_count == 0 && e->open_refcount == 0) {
            std::uint32_t slot = static_cast<std::uint32_t>(e - entries_.data());
            if (index_.count(e->key) && !lru_pos_.count(slot)) {
                lru_.push_front(slot);
                lru_pos_[slot] = lru_.begin();
            }
        }
    }

    // Erase: explicitly remove an entry (frees GPU memory).
    // Used when the caller knows the handle is no longer needed
    // (e.g., DataPath shutdown or explicit invalidation).
    void erase(std::uint64_t key) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = index_.find(key);
        if (it == index_.end()) return;
        std::uint32_t slot = it->second;
        Entry& e = entries_[slot];
        if (e.pin_count > 0 || e.open_refcount > 0) return;  // in use: cannot erase
        free_entry_gpu_(e);
        remove_from_lru_(slot);
        index_.erase(it);
        e = {};
        free_list_.push_back(slot);
        if (stats_.entries > 0) --stats_.entries;
        // Explicit invalidation also drops the L2 record (content may
        // be stale — e.g. the file was rewritten).
        l2_erase_(key);
    }

    Stats stats() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return stats_;
    }

private:
    Config cfg_{};
    bool initialized_ = false;
    std::vector<Entry> entries_;
    std::unordered_map<std::uint64_t, std::uint32_t> index_;
    std::list<std::uint32_t> lru_;  // unpinned, in-index entries; MRU front
    std::unordered_map<std::uint32_t, std::list<std::uint32_t>::iterator> lru_pos_;
    std::list<std::uint32_t> free_list_;

    // Round 16 S6b: L2 host-pinned content tier (design ruling
    // 2026-08-04: T = {handle POD + overflow blob}; d_qps copied by
    // value — queue structures never move; overflow travels as CONTENT
    // bytes; promote = memcpy restore, not rebuild).
    //
    // Record layout per L2 slot: handle struct image + inline overflow
    // blob.  Inclusive tiering: an L1-resident entry's L2 record is
    // pinned (absent from l2_lru_); it becomes L2-evictable only when
    // the L1 entry is downgraded or erased.
    struct L2RecordSlot {
        DeviceTargetHandle handle_image{};  // extents_overflow is stale; patched on restore
        std::uint64_t      overflow_bytes = 0;
        std::uint8_t       overflow_blob[kL2OverflowBlobBytes];
    };
    HostSlotPool<L2RecordSlot> l2_pool_;
    std::uint32_t l2_capacity_ = 0;
    std::unordered_map<std::uint64_t, std::uint32_t> l2_index_;  // key -> pool slot
    std::list<std::uint64_t> l2_lru_;   // L2-evictable keys (NOT L1-resident); MRU front
    std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator> l2_pos_;
    mutable std::mutex mtx_;
    Stats stats_{};

    bool l2_enabled() const { return l2_capacity_ > 0; }

    // Free GPU allocations for an entry (called under lock).
    void free_entry_gpu_(Entry& e) {
        if (free_fn_ && (e.handle != nullptr || e.overflow != nullptr)) {
            free_fn_(e.handle, e.overflow, cfg_.cuda_device);
        }
        e.handle = nullptr;
        e.overflow = nullptr;
        e.overflow_bytes = 0;
    }

    // Admit an L1-resident entry's content to L2 (called under lock
    // after a cold build).  The record is PINNED (not in l2_lru_) while
    // the entry stays L1-resident.  Snapshot failure or oversized
    // overflow simply skips L2 (a later downgrade retries; worst case
    // the entry degrades to single-tier behaviour).
    void admit_to_l2_(std::uint64_t key, const Entry& e) {
        if (!l2_enabled() || snapshot_fn_ == nullptr) return;
        if (l2_index_.count(key)) return;  // already backed
        save_to_l2_(key, e, /*l2_evictable=*/false);
    }

    // Save entry content into an L2 record (called under lock).
    // l2_evictable=true inserts the record into the L2 LRU (used on
    // downgrade); false keeps it pinned (used on admit).
    void save_to_l2_(std::uint64_t key, const Entry& e, bool l2_evictable) {
        if (e.handle == nullptr) return;
        if (e.overflow_bytes > kL2OverflowBlobBytes) return;  // too big for a slot
        if (l2_index_.count(key)) return;  // content already backed (immutable handle)

        std::uint32_t l2idx = l2_pool_.acquire(L2RecordSlot{});
        if (l2idx == UINT32_MAX) {
            // L2 full: genuine-delete the LRU record that is not
            // L1-resident, then retry once.
            evict_l2_lru_();
            l2idx = l2_pool_.acquire(L2RecordSlot{});
            if (l2idx == UINT32_MAX) return;  // still full: skip save
        }
        L2RecordSlot* rec = l2_pool_.slot_ptr(l2idx);
        if (!snapshot_fn_(e.handle, e.overflow, e.overflow_bytes,
                          cfg_.cuda_device, &rec->handle_image,
                          rec->overflow_blob)) {
            l2_pool_.release(l2idx);
            return;
        }
        rec->overflow_bytes = e.overflow_bytes;
        l2_index_[key] = l2idx;
        if (l2_evictable) {
            l2_lru_.push_front(key);
            l2_pos_[key] = l2_lru_.begin();
        }
    }

    // Genuine-delete the L2-LRU-back record (called under lock).
    void evict_l2_lru_() {
        while (!l2_lru_.empty()) {
            std::uint64_t victim_key = l2_lru_.back();
            l2_lru_.pop_back();
            l2_pos_.erase(victim_key);
            auto it = l2_index_.find(victim_key);
            if (it == l2_index_.end()) continue;
            l2_pool_.release(it->second);
            l2_index_.erase(it);
            ++stats_.l2_evictions;
            return;
        }
    }

    void l2_erase_(std::uint64_t key) {
        auto it = l2_index_.find(key);
        if (it == l2_index_.end()) return;
        l2_pool_.release(it->second);
        l2_index_.erase(it);
        remove_from_l2_lru_(key);
    }

    void remove_from_l2_lru_(std::uint64_t key) {
        auto it = l2_pos_.find(key);
        if (it != l2_pos_.end()) {
            l2_lru_.erase(it->second);
            l2_pos_.erase(it);
        }
    }

    std::uint32_t acquire_slot_() {
        if (!free_list_.empty()) {
            std::uint32_t s = free_list_.front();
            free_list_.pop_front();
            return s;
        }
        // Evict LRU entry that is both unpinned AND has no open references.
        // P0-1: previously only checked pin_count==0, missing the
        // reopen→eviction UAF (open_refcount > 0 but in_use was false
        // after close, making the entry a false LRU candidate).
        // LRU only contains entries with open_refcount==0 (release_entry
        // and unpin gate LRU insertion on open_refcount==0), so any LRU
        // entry is safe to evict.
        if (lru_.empty()) return UINT32_MAX;  // all in use or pinned
        std::uint32_t victim = lru_.back();
        lru_.pop_back();
        lru_pos_.erase(victim);
        Entry& ve = entries_[victim];
        // Round 16 S6b: eviction is a DOWNGRADE when L2 is enabled —
        // ensure the content is backed in L2 before freeing the GPU
        // allocations.  (Normally the record already exists from admit;
        // this also covers entries whose admit was skipped.)
        if (l2_enabled()) save_to_l2_(ve.key, ve, /*l2_evictable=*/true);
        free_entry_gpu_(ve);
        index_.erase(ve.key);
        ve = {};
        if (stats_.entries > 0) --stats_.entries;
        ++stats_.evictions;
        return victim;
    }

    void touch_lru_(std::uint32_t slot) {
        auto it = lru_pos_.find(slot);
        if (it != lru_pos_.end()) lru_.erase(it->second);
        lru_.push_front(slot);
        lru_pos_[slot] = lru_.begin();
    }

    void remove_from_lru_(std::uint32_t slot) {
        auto it = lru_pos_.find(slot);
        if (it != lru_pos_.end()) {
            lru_.erase(it->second);
            lru_pos_.erase(it);
        }
    }

public:
    // Function pointers for GPU allocation management, set by
    // LocalNvmeDataPath::initialize().  They keep CUDA calls out of
    // this header (the implementations live in io/device_target.cu).
    using FreeFn = void(*)(DeviceTargetHandle*, void*, std::uint32_t);
    void set_free_fn(FreeFn fn) { free_fn_ = fn; }

    // Round 16 S6b: L2 downgrade/restore hooks.
    // SnapshotFn: GPU handle+overflow → host image (2 D2H memcpys).
    // RestoreFn: host image → fresh GPU handle+overflow (2 H2D memcpys,
    //   extents_overflow patched to the fresh allocation).
    using SnapshotFn = bool(*)(const DeviceTargetHandle*, const void*,
                               std::uint64_t, std::uint32_t,
                               DeviceTargetHandle*, void*);
    using RestoreFn = bool(*)(const DeviceTargetHandle*, const void*,
                              std::uint64_t, std::uint32_t,
                              DeviceTargetHandle**, void**);
    void set_snapshot_fn(SnapshotFn fn) { snapshot_fn_ = fn; }
    void set_restore_fn(RestoreFn fn) { restore_fn_ = fn; }

private:
    FreeFn free_fn_ = nullptr;
    SnapshotFn snapshot_fn_ = nullptr;
    RestoreFn restore_fn_ = nullptr;
};

} // namespace tutti::data_paths::local_nvme

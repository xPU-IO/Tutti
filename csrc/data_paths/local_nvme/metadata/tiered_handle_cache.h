#ifndef __TUTTI_MEMORY_TIERED_HANDLE_CACHE_H__
#define __TUTTI_MEMORY_TIERED_HANDLE_CACHE_H__

/**
 * tiered_handle_cache.h -- two-tier (CPU pinned L2 + GPU L1) cache
 * with a location index, inclusive LRU, and batch build/promote.
 *
 * Layer: memory.
 *
 * Why two tiers:
 *
 *   A single GPU-resident pool sized to hold every file's handle
 *   (millions, at LMCache scale) would cost tens of GB of scarce GPU
 *   memory that has to compete with model weights and the KV cache
 *   tensors themselves -- unacceptable (~1 GB is the realistic
 *   budget).  But a small GPU pool alone means constant eviction +
 *   rebuild, and "rebuild" for a file handle means re-walking FIEMAP
 *   extents on the host -- expensive to repeat on every miss.
 *
 *   The fix (same shape as an OS page cache, or a classic L1/L2):
 *     L2 (CPU pinned, large: e.g. 2 GiB)  -- holds a fully-built T
 *          template for every file that's been touched at least
 *          once.  16M-slot scale (2 GiB / ~192 B) comfortably covers
 *          multi-ten-million-file deployments.  Building here is
 *          the ONLY place FIEMAP/extent work happens.
 *     L1 (GPU, small: e.g. 512 MiB) -- holds only the current
 *          working set.  Filled/evicted by plain memcpy against L2 --
 *          NEVER re-walks FIEMAP.
 *
 *   Eviction asymmetry (the key correctness/performance property):
 *     - L1 eviction is a DOWNGRADE, not a delete: the L2 copy is
 *       untouched, so the next access is one cudaMemcpyAsync away,
 *       not a full rebuild.
 *     - L2 eviction is a genuine delete (rare -- L2's budget is
 *       chosen to be large enough that this should almost never
 *       trigger in practice) and only ever targets entries that are
 *       NOT currently promoted to L1 (an L1-resident entry's L2 copy
 *       is pinned in place as L1's backing store).
 *
 * Batch acquisition (the "one IO batch must not launch thousands of
 * copies" fix): `get_or_build_batch` handles the COLD (never-seen)
 * subset of a batch with ONE host-side build pass + ONE contiguous
 * L2 write + ONE contiguous L1 cudaMemcpyAsync, regardless of how
 * many files in the batch are new.  L2-resident-but-not-L1 entries
 * are promoted individually (expected to be a small minority once
 * the working set has warmed up); L1-resident hits cost nothing but
 * an LRU touch + a (cheap, often-redundant) `wait_ready`.
 *
 * Streams: `get_or_build*` never assumes "the stream that filled a
 * slot" and "the stream that reads it" are the same -- every
 * returned pointer has already been passed through
 * GpuSlotPool::wait_ready(ptr, stream) against the CALLER's stream,
 * so the caller's stream is always safe to launch IO/consumer
 * kernels on immediately after, with zero additional synchronization
 * on the caller's part (see gpu_slot_pool.h).
 *
 * Thread-safety: one mutex guards the location index + both LRU
 * lists + the calls into L1/L2 for the duration of one
 * get_or_build[_batch] / evict_* / erase call.  Fine for the current
 * call pattern (a single control-plane thread per Coordinator); a
 * striped/sharded lock is a future option if profiling ever shows
 * contention here.
 */

#include <cstdint>
#include <cstdio>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tutti/cuda_like.h>

#include "csrc/data_paths/local_nvme/metadata/gpu_slot_pool.h"
#include "csrc/data_paths/local_nvme/metadata/host_slot_pool.h"

namespace tutti::data_paths::local_nvme {

template <typename T, typename KeyType>
class TieredHandleCache {
public:
    /// Builds the T template for a COLD key (e.g. walks FIEMAP,
    /// reads controller state).  Returns false to fail the whole
    /// containing get_or_build[_batch] call.
    using BuilderFn = std::function<bool(const KeyType& key, T* out)>;

    struct Config {
        uint32_t l1_capacity  = 0;   // GPU-resident slots
        uint32_t l2_capacity  = 0;   // CPU-pinned slots
        int      cuda_device  = 0;
    };

    /// Cumulative counters (since init), for observability / stress
    /// tests that want to prove eviction actually happened and measure
    /// its cost.  Plain uint64_t under mtx_ (every mutation already
    /// happens while the cache mutex is held), read via stats().
    struct Stats {
        uint64_t cold_builds   = 0;  // COLD miss -> builder ran (FIEMAP etc.)
        uint64_t l1_hits       = 0;  // already GPU-resident on lookup
        uint64_t l2_hits       = 0;  // L2-resident, promoted to L1
        uint64_t l1_promotions = 0;  // L2 -> L1 (memcpy in)  [cold + l2_hit]
        uint64_t l1_evictions  = 0;  // L1 -> L2 downgrade (GPU slot freed)
        uint64_t l2_evictions  = 0;  // L2 genuine delete (needs rebuild next)
    };

    bool init(const Config& cfg) {
        if (cfg.l1_capacity == 0 || cfg.l2_capacity == 0) {
            std::fprintf(stderr,
                "[tiered_handle_cache] init: l1_capacity=%u l2_capacity=%u "
                "(both must be > 0)\n", cfg.l1_capacity, cfg.l2_capacity);
            return false;
        }
        if (cfg.l1_capacity > cfg.l2_capacity) {
            std::fprintf(stderr,
                "[tiered_handle_cache] init: l1_capacity=%u > l2_capacity=%u "
                "-- L1 must fit inside L2's backing store\n",
                cfg.l1_capacity, cfg.l2_capacity);
            return false;
        }
        if (!l2_.init(cfg.l2_capacity)) return false;
        if (!l1_.init(cfg.l1_capacity, cfg.cuda_device)) { l2_.shutdown(); return false; }
        cfg_ = cfg;
        return true;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mtx_);
        l1_.shutdown();
        l2_.shutdown();
        index_.clear();
        l1_lru_.clear();
        l1_lru_pos_.clear();
        l2_evict_lru_.clear();
        l2_evict_pos_.clear();
    }

    // ------------------------------------------------------------------
    // Memory accounting
    // ------------------------------------------------------------------

    std::size_t l1_gpu_bytes()  const { return (std::size_t)cfg_.l1_capacity * sizeof(T); }
    std::size_t l2_host_bytes() const { return (std::size_t)cfg_.l2_capacity * sizeof(T); }
    uint32_t    l1_capacity()   const { return cfg_.l1_capacity; }
    uint32_t    l2_capacity()   const { return cfg_.l2_capacity; }

    /// Snapshot of the cumulative counters.  Taken under mtx_.
    Stats stats() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return stats_;
    }

    // ------------------------------------------------------------------
    // Single-key resolve
    // ------------------------------------------------------------------

    /// Ensure `key`'s T is GPU-resident, building it via `builder` on
    /// a COLD miss.  Returns nullptr on any failure (builder failure,
    /// or both tiers exhausted with nothing evictable).  The
    /// returned pointer is already `wait_ready`'d against `stream` --
    /// safe to consume from GPU work queued on `stream` immediately
    /// after this call, no matter which stream last filled it.
    T* get_or_build(const KeyType& key, const BuilderFn& builder, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mtx_);
        T* ptr = get_or_build_locked_(key, builder, stream);
        if (ptr != nullptr) l1_.wait_ready(ptr, stream);
        return ptr;
    }

    /// Batch resolve.  See the file comment for the COLD-batching
    /// strategy.  `out_ptrs[i]` corresponds to `keys[i]`; on any
    /// per-key failure (builder or exhaustion) this returns false and
    /// `out_ptrs` is left partially filled (undefined past the
    /// failure point) -- callers should treat any false return as
    /// "the whole batch failed", matching the existing
    /// max_entries_per_batch / handle_cache_capacity error contracts.
    ///
    /// Duplicate keys in `keys` are fine: they resolve to the same
    /// cache entry and the same output pointer (R12 A-5 / 隐患-4:
    /// without dedup, a repeated COLD key would overwrite index_[key]
    /// in insert_l2_batch_locked_ and leak the first L2 slot).
    ///
    /// Hard requirement: the number of DISTINCT keys in one call
    /// MUST be <= l1_capacity().  Every key resolved by one batch
    /// call must stay L1-resident for the DURATION of that call (the
    /// caller is about to launch IO/consumer kernels against all of
    /// them); eviction during this call therefore protects every key
    /// in `keys` and only evicts entries OUTSIDE the batch.  If the
    /// batch itself doesn't fit in L1, there is nothing eviction can
    /// do -- this is rejected up front rather than silently
    /// corrupting an early result by evicting-and-overwriting it
    /// later in the same call (a real bug caught by
    /// tiered_handle_cache_smoke, not by inspection).
    bool get_or_build_batch(const KeyType*    keys,
                            uint32_t          count,
                            const BuilderFn&  builder,
                            cudaStream_t      stream,
                            T**               out_ptrs) {
        if (count == 0) return true;

        // Deduplicate keys: duplicate keys are valid (callers may pass
        // the same id multiple times) but must resolve to the same
        // cache entry.  Without dedup, a repeated COLD key would cause
        // insert_l2_batch_locked_ to overwrite index_[key] and leak
        // the first L2 slot.
        std::unordered_map<KeyType, uint32_t> uniq_idx;
        uniq_idx.reserve(count);
        std::vector<KeyType> unique_keys;
        unique_keys.reserve(count);
        std::vector<uint32_t> orig_to_uniq(count, 0);
        for (uint32_t i = 0; i < count; ++i) {
            auto r = uniq_idx.emplace(keys[i], (uint32_t)unique_keys.size());
            if (r.second) unique_keys.push_back(keys[i]);
            orig_to_uniq[i] = r.first->second;
        }
        const uint32_t ucount = (uint32_t)unique_keys.size();

        if (ucount > l1_.capacity()) {
            std::fprintf(stderr,
                "[tiered_handle_cache] get_or_build_batch: distinct_count=%u > "
                "l1_capacity=%u -- a single batch's distinct keys must fit "
                "in L1 for the duration of the call (raise l1_capacity)\n",
                ucount, l1_.capacity());
            return false;
        }
        if (ucount > l2_.capacity()) {
            std::fprintf(stderr,
                "[tiered_handle_cache] get_or_build_batch: distinct_count=%u > "
                "l2_capacity=%u -- a single batch's distinct keys must fit "
                "in L2 too (raise l2_capacity)\n",
                ucount, l2_.capacity());
            return false;
        }
        std::lock_guard<std::mutex> lock(mtx_);

        // Every key in this batch is protected from eviction by any
        // of the promote/evict calls below -- see the doc comment.
        std::unordered_set<KeyType> protect(unique_keys.begin(), unique_keys.end());

        std::vector<uint32_t> hit_l1, hit_l2, cold;
        hit_l1.reserve(ucount); hit_l2.reserve(ucount); cold.reserve(ucount);
        for (uint32_t i = 0; i < ucount; ++i) {
            auto it = index_.find(unique_keys[i]);
            if (it == index_.end())                    cold.push_back(i);
            else if (it->second.state == Loc::L1_GPU)  hit_l1.push_back(i);
            else                                        hit_l2.push_back(i);
        }
        // Stats: classify each key.  cold -> builder + L2 insert + L1
        // promote; hit_l2 -> L1 promote; hit_l1 -> pure hit.
        stats_.l1_hits       += hit_l1.size();
        stats_.l2_hits       += hit_l2.size();
        stats_.cold_builds   += cold.size();
        stats_.l1_promotions += cold.size() + hit_l2.size();

        // Results for unique keys; scattered back to original indices
        // at the end so duplicate keys share the same pointer.
        std::vector<T*> uniq_out(ucount, nullptr);

        // (1) L1 hits: touch LRU, collect pointer.
        for (uint32_t i : hit_l1) {
            Entry& e = index_[unique_keys[i]];
            touch_l1_lru_locked_(unique_keys[i]);
            uniq_out[i] = l1_.slot_ptr(e.l1_idx);
        }

        // (2) COLD: batch-build host templates, batch-insert into L2
        //     (contiguous when possible), batch-promote to L1 with
        //     ONE cudaMemcpyAsync.  This is the path that matters for
        //     "one IO batch, thousands of new files".
        if (!cold.empty()) {
            std::vector<T> built(cold.size());
            for (std::size_t i = 0; i < cold.size(); ++i) {
                if (!builder(unique_keys[cold[i]], &built[i])) {
                    std::fprintf(stderr,
                        "[tiered_handle_cache] get_or_build_batch: builder "
                        "failed for cold key (batch pos %u)\n", cold[i]);
                    return false;
                }
            }
            if (!insert_l2_batch_locked_(unique_keys.data(), cold, built, protect)) return false;
            if (!promote_batch_to_l1_locked_(unique_keys.data(), cold, stream, uniq_out.data(), &protect)) return false;
        }

        // (3) L2 hits (not yet on L1): promote individually.  Expected
        //     to be a small minority once the working set is warm.
        for (uint32_t i : hit_l2) {
            T* ptr = promote_one_to_l1_locked_(unique_keys[i], stream, &protect);
            if (ptr == nullptr) {
                std::fprintf(stderr,
                    "[tiered_handle_cache] get_or_build_batch: L2->L1 "
                    "promote failed (batch pos %u)\n", i);
                return false;
            }
            uniq_out[i] = ptr;
        }

        // Scatter unique-key results back to original (possibly
        // duplicated) indices -- duplicate keys share the same pointer.
        for (uint32_t i = 0; i < count; ++i)
            out_ptrs[i] = uniq_out[orig_to_uniq[i]];

        for (uint32_t i = 0; i < ucount; ++i) l1_.wait_ready(uniq_out[i], stream);
        return true;
    }

    // ------------------------------------------------------------------
    // Explicit eviction / cleanup
    // ------------------------------------------------------------------

    /// Admit `key` into L2 (the CPU-pinned tier) ONLY -- build its T
    /// template via `builder` on a COLD miss, but do NOT promote to L1
    /// (no GPU residency, hence no stream parameter and no GPU work).
    /// Idempotent: if `key` is already present in EITHER tier this is a
    /// no-op (an L1-resident entry is by definition already L2-backed).
    ///
    /// Purpose: bind cache membership to an external open/close
    /// lifecycle.  A caller that admits on "open" and `erase`s on
    /// "close" gets the invariant "present in this cache  <=>  the
    /// underlying object is currently open" -- so a cache miss
    /// unambiguously means "not open", which a plain lazy
    /// get_or_build cache cannot guarantee (a live-but-idle entry may
    /// have been silently L2-evicted).  See nvme_storage's
    /// open_file/close_file.
    ///
    /// Returns false only on builder failure or L2 exhaustion with
    /// nothing evictable (every L2 entry currently pinned in L1).
    bool admit(const KeyType& key, const BuilderFn& builder) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (index_.find(key) != index_.end()) return true;   // already L2 or L1

        T tmp{};
        if (!builder(key, &tmp)) {
            std::fprintf(stderr, "[tiered_handle_cache] admit: builder failed\n");
            return false;
        }
        uint32_t l2_idx = l2_.acquire(tmp);
        if (l2_idx == UINT32_MAX) {
            if (!evict_l2_lru_one_locked_()) {
                std::fprintf(stderr,
                    "[tiered_handle_cache] admit: L2 exhausted and nothing "
                    "evictable (every entry pinned in L1)\n");
                return false;
            }
            l2_idx = l2_.acquire(tmp);
            if (l2_idx == UINT32_MAX) return false;
        }
        Entry e; e.state = Loc::L2_HOST; e.l2_idx = l2_idx;
        index_[key] = e;
        l2_evict_lru_.push_front(key);
        l2_evict_pos_[key] = l2_evict_lru_.begin();
        return true;
    }

    /// Downgrade `key` from L1 to L2 (content preserved, GPU slot
    /// freed).  No-op if `key` isn't currently L1-resident.
    void evict_from_l1(const KeyType& key, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = index_.find(key);
        if (it == index_.end() || it->second.state != Loc::L1_GPU) return;
        downgrade_l1_entry_locked_(key, it->second, stream);
    }

    /// Downgrade every L1-resident entry to L2.  Useful under GPU
    /// memory pressure, or to prove content survives purely from L2
    /// (see kv_cache_adapter_smoke's drop_cached_handles-equivalent).
    void evict_all_from_l1(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::vector<KeyType> victims(l1_lru_.begin(), l1_lru_.end());
        for (const KeyType& k : victims) {
            auto it = index_.find(k);
            if (it != index_.end() && it->second.state == Loc::L1_GPU)
                downgrade_l1_entry_locked_(k, it->second, stream);
        }
    }

    /// Fully remove `key` from both tiers (used when the underlying
    /// file is deleted).  If it's currently L1-resident, downgrades
    /// first (frees the GPU slot) then deletes the L2 copy.
    void erase(const KeyType& key, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = index_.find(key);
        if (it == index_.end()) return;
        if (it->second.state == Loc::L1_GPU) downgrade_l1_entry_locked_(key, it->second, stream);
        // Now guaranteed L2_HOST (or was already) -- remove from the
        // L2-evictable LRU and the pool.
        auto pit = l2_evict_pos_.find(key);
        if (pit != l2_evict_pos_.end()) { l2_evict_lru_.erase(pit->second); l2_evict_pos_.erase(pit); }
        l2_.release(it->second.l2_idx);
        index_.erase(it);
    }

private:
    enum class Loc : uint8_t { L2_HOST, L1_GPU };   // no explicit COLD state --
                                                     // absence from index_ IS cold.
    struct Entry {
        Loc      state;
        uint32_t l2_idx = UINT32_MAX;
        uint32_t l1_idx = UINT32_MAX;   // valid iff state == L1_GPU
    };

    // ---- L1 <-> L2 transition helpers (caller holds mtx_) ----

    void downgrade_l1_entry_locked_(const KeyType& key, Entry& e, cudaStream_t stream) {
        T* gpu_ptr = l1_.slot_ptr(e.l1_idx);
        l1_.release_async(gpu_ptr, stream);
        e.state  = Loc::L2_HOST;
        e.l1_idx = UINT32_MAX;
        auto lit = l1_lru_pos_.find(key);
        if (lit != l1_lru_pos_.end()) { l1_lru_.erase(lit->second); l1_lru_pos_.erase(lit); }
        l2_evict_lru_.push_front(key);
        l2_evict_pos_[key] = l2_evict_lru_.begin();
        ++stats_.l1_evictions;
    }

    void touch_l1_lru_locked_(const KeyType& key) {
        auto it = l1_lru_pos_.find(key);
        if (it != l1_lru_pos_.end()) l1_lru_.erase(it->second);
        l1_lru_.push_front(key);
        l1_lru_pos_[key] = l1_lru_.begin();
    }

    bool evict_l1_lru_one_locked_(cudaStream_t stream,
                                  const std::unordered_set<KeyType>* protect = nullptr) {
        // Scan from the LRU tail (oldest) for the first entry NOT in
        // `protect`.  `protect` is small (one batch's worth of keys)
        // and evictions are rare relative to hits, so a linear scan
        // here is fine; erase-and-return immediately once found so we
        // never touch an iterator invalidated by that erase.
        for (auto it = l1_lru_.rbegin(); it != l1_lru_.rend(); ++it) {
            if (protect != nullptr && protect->count(*it) != 0) continue;
            const KeyType victim = *it;
            Entry& e = index_.at(victim);
            downgrade_l1_entry_locked_(victim, e, stream);
            return true;
        }
        return false;   // nothing evictable (everything is protected)
    }

    bool evict_l2_lru_one_locked_(const std::unordered_set<KeyType>* protect = nullptr) {
        for (auto it = l2_evict_lru_.rbegin(); it != l2_evict_lru_.rend(); ++it) {
            if (protect != nullptr && protect->count(*it) != 0) continue;
            const KeyType victim = *it;
            // erase via the forward-iterator stashed in l2_evict_pos_
            // (safe: we return immediately after, never touching the
            // now-invalidated reverse iterator again).
            auto pit = l2_evict_pos_.find(victim);
            l2_evict_lru_.erase(pit->second);
            l2_evict_pos_.erase(pit);
            auto it2 = index_.find(victim);
            if (it2 != index_.end()) {
                l2_.release(it2->second.l2_idx);
                index_.erase(it2);
            }
            ++stats_.l2_evictions;
            return true;
        }
        return false;   // nothing evictable (all pinned by L1 or protected)
    }

    // Single get-or-build, no wait_ready (caller adds it once, for
    // both the single- and batch-key paths, at the outer call site).
    T* get_or_build_locked_(const KeyType& key, const BuilderFn& builder, cudaStream_t stream) {
        auto it = index_.find(key);
        if (it != index_.end() && it->second.state == Loc::L1_GPU) {
            touch_l1_lru_locked_(key);
            ++stats_.l1_hits;
            return l1_.slot_ptr(it->second.l1_idx);
        }
        if (it != index_.end() && it->second.state == Loc::L2_HOST) {
            ++stats_.l2_hits;
            ++stats_.l1_promotions;
            return promote_one_to_l1_locked_(key, stream, /*protect=*/nullptr);
        }
        // COLD.
        ++stats_.cold_builds;
        ++stats_.l1_promotions;
        T tmp{};
        if (!builder(key, &tmp)) {
            std::fprintf(stderr, "[tiered_handle_cache] get_or_build: builder failed\n");
            return nullptr;
        }
        uint32_t l2_idx = l2_.acquire(tmp);
        if (l2_idx == UINT32_MAX) {
            if (!evict_l2_lru_one_locked_()) {
                std::fprintf(stderr,
                    "[tiered_handle_cache] get_or_build: L2 exhausted and "
                    "nothing evictable (every entry pinned in L1)\n");
                return nullptr;
            }
            l2_idx = l2_.acquire(tmp);
            if (l2_idx == UINT32_MAX) return nullptr;
        }
        Entry e; e.state = Loc::L2_HOST; e.l2_idx = l2_idx;
        index_[key] = e;
        l2_evict_lru_.push_front(key);
        l2_evict_pos_[key] = l2_evict_lru_.begin();
        return promote_one_to_l1_locked_(key, stream, /*protect=*/nullptr);
    }

    T* promote_one_to_l1_locked_(const KeyType& key, cudaStream_t stream,
                                 const std::unordered_set<KeyType>* protect) {
        Entry& e = index_.at(key);
        const T* host_src = l2_.slot_ptr(e.l2_idx);
        T* gpu_ptr = l1_.acquire_one_from_host_async(host_src, stream);
        if (gpu_ptr == nullptr) {
            if (!evict_l1_lru_one_locked_(stream, protect)) return nullptr;
            gpu_ptr = l1_.acquire_one_from_host_async(host_src, stream);
            if (gpu_ptr == nullptr) return nullptr;
        }
        e.state  = Loc::L1_GPU;
        e.l1_idx = (uint32_t)(gpu_ptr - l1_.slot_ptr(0));
        // remove from L2-evictable LRU (now pinned by L1), add to L1 LRU.
        auto pit = l2_evict_pos_.find(key);
        if (pit != l2_evict_pos_.end()) { l2_evict_lru_.erase(pit->second); l2_evict_pos_.erase(pit); }
        touch_l1_lru_locked_(key);
        return gpu_ptr;
    }

    // ---- batch helpers (caller holds mtx_) ----

    bool insert_l2_batch_locked_(const KeyType* keys, const std::vector<uint32_t>& cold_pos,
                                 const std::vector<T>& built,
                                 const std::unordered_set<KeyType>& protect) {
        uint32_t l2_start = 0;
        if (l2_.acquire_batch_contig(built.data(), (uint32_t)built.size(), &l2_start)) {
            for (std::size_t i = 0; i < cold_pos.size(); ++i) {
                Entry e; e.state = Loc::L2_HOST; e.l2_idx = l2_start + (uint32_t)i;
                const KeyType& key = keys[cold_pos[i]];
                index_[key] = e;
                l2_evict_lru_.push_front(key);
                l2_evict_pos_[key] = l2_evict_lru_.begin();
            }
            return true;
        }
        // Fallback: L2 bump region can't satisfy a contiguous run
        // (rare -- only once L2 itself has been evicted from at least
        // once).  Insert one at a time, evicting L2 LRU as needed --
        // excluding this batch's own keys (see get_or_build_batch's
        // doc comment; same "don't evict-then-overwrite yourself"
        // hazard applies at the L2 tier).
        for (std::size_t i = 0; i < cold_pos.size(); ++i) {
            uint32_t idx = l2_.acquire(built[i]);
            if (idx == UINT32_MAX) {
                if (!evict_l2_lru_one_locked_(&protect)) return false;
                idx = l2_.acquire(built[i]);
                if (idx == UINT32_MAX) return false;
            }
            Entry e; e.state = Loc::L2_HOST; e.l2_idx = idx;
            const KeyType& key = keys[cold_pos[i]];
            index_[key] = e;
            l2_evict_lru_.push_front(key);
            l2_evict_pos_[key] = l2_evict_lru_.begin();
        }
        return true;
    }

    bool promote_batch_to_l1_locked_(const KeyType* keys, const std::vector<uint32_t>& cold_pos,
                                     cudaStream_t stream, T** out_ptrs,
                                     const std::unordered_set<KeyType>* protect) {
        // The L2 entries we just inserted may or may not be
        // contiguous there (see insert_l2_batch_locked_'s fallback),
        // so re-derive contiguity from the actual l2_idx values
        // rather than assuming it.
        bool l2_contig = true;
        uint32_t base_l2_idx = index_.at(keys[cold_pos[0]]).l2_idx;
        for (std::size_t i = 0; i < cold_pos.size(); ++i) {
            if (index_.at(keys[cold_pos[i]]).l2_idx != base_l2_idx + i) { l2_contig = false; break; }
        }

        if (l2_contig) {
            const T* host_src = l2_.slot_ptr(base_l2_idx);
            T* l1_start = nullptr;
            if (l1_.acquire_batch_from_host_async(host_src, (uint32_t)cold_pos.size(),
                                                  stream, &l1_start)) {
                for (std::size_t i = 0; i < cold_pos.size(); ++i) {
                    const KeyType& key = keys[cold_pos[i]];
                    Entry& e = index_.at(key);
                    e.state  = Loc::L1_GPU;
                    e.l1_idx = (uint32_t)((l1_start + i) - l1_.slot_ptr(0));
                    auto pit = l2_evict_pos_.find(key);
                    if (pit != l2_evict_pos_.end()) { l2_evict_lru_.erase(pit->second); l2_evict_pos_.erase(pit); }
                    touch_l1_lru_locked_(key);
                    out_ptrs[cold_pos[i]] = l1_start + i;
                }
                return true;
            }
            // L1 bump exhausted for a contiguous run of this size --
            // evict enough LRU-tail entries (excluding this batch's
            // own keys) and retry once, then fall through to the
            // per-item path below if still short.
            for (std::size_t tries = 0; tries < cold_pos.size(); ++tries) {
                if (!evict_l1_lru_one_locked_(stream, protect)) break;
                if (l1_.acquire_batch_from_host_async(host_src, (uint32_t)cold_pos.size(),
                                                      stream, &l1_start)) {
                    for (std::size_t i = 0; i < cold_pos.size(); ++i) {
                        const KeyType& key = keys[cold_pos[i]];
                        Entry& e = index_.at(key);
                        e.state  = Loc::L1_GPU;
                        e.l1_idx = (uint32_t)((l1_start + i) - l1_.slot_ptr(0));
                        auto pit = l2_evict_pos_.find(key);
                        if (pit != l2_evict_pos_.end()) { l2_evict_lru_.erase(pit->second); l2_evict_pos_.erase(pit); }
                        touch_l1_lru_locked_(key);
                        out_ptrs[cold_pos[i]] = l1_start + i;
                    }
                    return true;
                }
            }
        }

        // Fallback: per-item promotion (still correct, just not the
        // single-cudaMemcpyAsync fast path).  Still excludes this
        // batch's own keys from eviction.
        for (std::size_t i = 0; i < cold_pos.size(); ++i) {
            T* ptr = promote_one_to_l1_locked_(keys[cold_pos[i]], stream, protect);
            if (ptr == nullptr) return false;
            out_ptrs[cold_pos[i]] = ptr;
        }
        return true;
    }

    mutable std::mutex mtx_;
    Config     cfg_{};
    Stats      stats_{};
    HostSlotPool<T> l2_;
    GpuSlotPool<T>  l1_;

    std::unordered_map<KeyType, Entry> index_;

    // L1 LRU: only L1_GPU-state keys.  MRU at front.
    std::list<KeyType> l1_lru_;
    std::unordered_map<KeyType, typename std::list<KeyType>::iterator> l1_lru_pos_;

    // L2-evictable LRU: only keys that are L2_HOST and NOT L1_GPU
    // (an L1-resident entry's L2 copy is pinned -- it's L1's backing
    // store and must not be reclaimed out from under it).  MRU at
    // front (most-recently-downgraded-from-L1, or most-recently-built).
    std::list<KeyType> l2_evict_lru_;
    std::unordered_map<KeyType, typename std::list<KeyType>::iterator> l2_evict_pos_;
};

} // namespace tutti

#endif // __TUTTI_MEMORY_TIERED_HANDLE_CACHE_H__

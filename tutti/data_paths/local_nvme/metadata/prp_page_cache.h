#pragma once

// tutti/data_paths/local_nvme/metadata/prp_page_cache.h
//
// Single-tier content-addressed LRU cache for PRP-list pages.
//
// Design (2026-08-20 rework, replaces the GPU-resident pool):
//   The pool is page-aligned host memory, pinned and DMA-mapped with
//   nvm_dma_map_data_host once at init().  This makes the miss path a plain
//   host memcpy — no cudaMemcpy, no stream, no event fencing:
//     - fill_prp_list_page() writes the page content directly into the pool
//       slot (~1us), versus ~150us per page for a pageable-source
//       cudaMemcpyAsync (the measured 9.4ms cold-write stall).
//     - The NVMe controller DMAs the PRP list from host DRAM per command —
//       one PCIe fetch either way, equivalent to fetching from GPU memory.
//     - Host RAM is the right tier for page tables at GB-scale working
//       sets; HBM is left for payload data.
//   Sizing requirement: the pool MUST hold the whole deployment working set
//   (all LIST keys of all registered memories).  Exhaustion falls back to
//   the arena path (correct, slower) — size prp_cache_capacity accordingly
//   (~1-2 GB per device for KV-cache-scale deployments).
//
// DMA lifecycle:
//   The cache owns one contiguous host-pinned allocation and one shared
//   DMA mapping (nvm_dma_map_data_host) covering all capacity pages.
//   - Born together: init() allocates both.
//   - Die together: shutdown() unmaps DMA first, then frees host memory.
//   - Per-page eviction: marks slot reusable (no unmap/free).  The pin
//     mechanism ensures no in-flight op references an evicted slot.
//   - This satisfies "DMA mapping lives/dies with backing page": the shared
//     mapping and backing allocation have identical lifetimes.
//
// Content key:
//   {memory_token, start_page, pages_in_io} uniquely determines the PRP-list
//   page content because content = ioaddrs[start_page+1..start_page+pages_in_io-1].
//   If memory is re-registered (new token), old entries are never hit again.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <nvm_types.h>

namespace tutti::data_paths::local_nvme {

class PrpPageCache {
public:
    struct Config {
        std::uint32_t capacity = 0;     // 0 = disabled; number of PRP-list pages
        std::uint32_t page_size = 4096; // NVMe page size
        std::uint32_t cuda_device = 0;  // legacy field, unused (host-pinned pool)
    };

    struct Stats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t evictions = 0;
        std::uint32_t pinned = 0;
        std::uint32_t entries = 0;
    };

    // Key: identifies a unique PRP-list page content.
    struct Key {
        std::uint64_t memory_token = 0;
        std::uint32_t start_page = 0;
        std::uint32_t pages_in_io = 0;
        bool operator==(const Key& o) const {
            return memory_token == o.memory_token &&
                   start_page == o.start_page &&
                   pages_in_io == o.pages_in_io;
        }
    };

    struct KeyHash {
        std::size_t operator()(const Key& k) const {
            return std::hash<std::uint64_t>()(k.memory_token) ^
                   (std::hash<std::uint32_t>()(k.start_page) << 1) ^
                   (std::hash<std::uint32_t>()(k.pages_in_io) << 2);
        }
    };

    // Entry: a cached PRP-list page.
    struct Entry {
        Key key{};
        void* vaddr = nullptr;        // host-pinned VA of the page (within pool)
        std::uint64_t ioaddr = 0;     // DMA IOVA of the page (host-side)
        std::uint32_t pin_count = 0;
        std::uint32_t checkout_refcount = 0;  // P0-1: >0 while checked out by submit
        bool in_use = false;  // deprecated: superseded by checkout_refcount
    };

    PrpPageCache() = default;
    ~PrpPageCache() { shutdown(); }

    PrpPageCache(const PrpPageCache&) = delete;
    PrpPageCache& operator=(const PrpPageCache&) = delete;

    // Pre-allocate the pool: one aligned allocation + one DMA map/pin.
    bool init(const Config& cfg, nvm_ctrl_t* ctrl);
    void shutdown();

    bool enabled() const { return cfg_.capacity > 0 && initialized_; }
    std::uint32_t capacity() const { return cfg_.capacity; }

    // Get or build a PRP-list page.
    // On hit: returns cached Entry* (no work).
    // On miss: fills the page content from data_dma's IOVA table directly
    // into the pool slot (host memcpy, ~1us) and returns the new entry.
    // Returns nullptr when disabled, out of slots, or on failure.
    //
    // The data path passes only (Key, data_dma); how the page content is
    // produced is the cache's business — no PRP staging machinery in the
    // data path layer.
    Entry* get_or_build(const Key& key, const nvm_dma_t* data_dma);

    // Batch variant: ONE locked pass over `count` items (a 921-target
    // 256K-context batch used to take this mutex 1842 times per submit).
    // On hit: result = cached entry (checkout++, not evictable).
    // On miss: builds the page and result = the new entry.
    // Items whose result stays nullptr (disabled / out of slots / null
    // dma) must be handled by the caller via the arena fallback.
    struct BatchItem {
        Key key{};
        const nvm_dma_t* data_dma = nullptr;
        std::uint32_t user = 0;       // caller payload (e.g. list index)
        Entry* result = nullptr;      // out
    };
    std::uint32_t get_or_build_batch(BatchItem* items, std::size_t count);

    // Release one checkout WITHOUT pinning — for error paths before op
    // registration (pin() already handles the success path by consuming
    // the checkout).  When both counts reach zero the entry re-enters LRU.
    void release_checkout(Entry* e) {
        if (!e) return;
        std::lock_guard<std::mutex> lock(mtx_);
        if (e->checkout_refcount > 0) --e->checkout_refcount;
        e->in_use = (e->checkout_refcount > 0) || (e->pin_count > 0);
        if (e->pin_count == 0 && e->checkout_refcount == 0) {
            std::uint32_t slot = static_cast<std::uint32_t>(e - entries_.data());
            if (index_.count(e->key) && !lru_pos_.count(slot)) {
                lru_.push_front(slot);
                lru_pos_[slot] = lru_.begin();
            }
        }
    }

    void pin(Entry* e) {
        if (!e) return;
        std::lock_guard<std::mutex> lock(mtx_);
        // P0-1: decrement checkout_refcount (the submit path checked it out,
        // now it's being pinned for the op's lifetime).
        if (e->checkout_refcount > 0) --e->checkout_refcount;
        e->in_use = false;  // no longer just "checked out" — now pinned
        ++e->pin_count;
        ++stats_.pinned;
        auto it = lru_pos_.find(static_cast<std::uint32_t>(e - entries_.data()));
        if (it != lru_pos_.end()) {
            lru_.erase(it->second);
            lru_pos_.erase(it);
        }
    }

    void unpin(Entry* e) {
        if (!e) return;
        std::lock_guard<std::mutex> lock(mtx_);
        if (e->pin_count > 0) {
            --e->pin_count;
            --stats_.pinned;
        }
        if (e->pin_count == 0 && e->checkout_refcount == 0) {
            std::uint32_t slot = static_cast<std::uint32_t>(e - entries_.data());
            if (index_.count(e->key) && !lru_pos_.count(slot)) {
                lru_.push_front(slot);
                lru_pos_[slot] = lru_.begin();
            }
        }
    }

    // Invalidate all entries for a given memory token (called on unregister).
    // Pinned entries are skipped (in-flight op still references them).
    void invalidate_memory(std::uint64_t memory_token) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::vector<Key> to_erase;
        for (auto& [k, slot] : index_) {
            if (k.memory_token == memory_token && entries_[slot].pin_count == 0 &&
                entries_[slot].checkout_refcount == 0) {
                to_erase.push_back(k);
            }
        }
        for (const auto& k : to_erase) {
            auto it = index_.find(k);
            if (it == index_.end()) continue;
            std::uint32_t slot = it->second;
            remove_from_lru_(slot);
            entries_[slot] = {};
            free_list_.push_back(slot);
            index_.erase(it);
            if (stats_.entries > 0) --stats_.entries;
        }
    }

    Stats stats() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return stats_;
    }

    // Test accessor: check if a page's DMA mapping is active.
    // Returns true if the pool DMA mapping exists (cache initialized).
    bool test_dma_active() const { return pool_dma_ != nullptr; }

private:
    Config cfg_{};
    nvm_ctrl_t* ctrl_ = nullptr;
    bool initialized_ = false;

    // Pool: one contiguous host-pinned allocation, DMA-mapped as a whole.
    void* pool_host_ = nullptr;
    nvm_dma_t* pool_dma_ = nullptr;

    std::vector<Entry> entries_;
    std::unordered_map<Key, std::uint32_t, KeyHash> index_;
    std::list<std::uint32_t> lru_;
    std::unordered_map<std::uint32_t, std::list<std::uint32_t>::iterator> lru_pos_;
    std::list<std::uint32_t> free_list_;
    mutable std::mutex mtx_;
    Stats stats_{};

    std::uint32_t acquire_slot_() {
        if (!free_list_.empty()) {
            std::uint32_t s = free_list_.front();
            free_list_.pop_front();
            return s;
        }
        if (lru_.empty()) return UINT32_MAX;
        std::uint32_t victim = lru_.back();
        lru_.pop_back();
        lru_pos_.erase(victim);
        index_.erase(entries_[victim].key);
        entries_[victim] = {};
        if (stats_.entries > 0) --stats_.entries;
        ++stats_.evictions;
        return victim;
    }

    void remove_from_lru_(std::uint32_t slot) {
        auto it = lru_pos_.find(slot);
        if (it != lru_pos_.end()) {
            lru_.erase(it->second);
            lru_pos_.erase(it);
        }
    }
};

} // namespace tutti::data_paths::local_nvme

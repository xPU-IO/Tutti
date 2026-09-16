// tutti/data_paths/local_nvme/metadata/tiered_handle_adapter.h
// Round 16 S6b: adapter wrapping TieredHandleCache<T, KeyType> to provide
// a HandleWorkspaceCache-compatible API, so LocalNvmeDataPath can use the
// tiered (L1 GPU + L2 host-pinned) cache without rewriting its call sites.
//
// T = HandleCacheEntry (DeviceTargetHandle* + overflow* + open_refcount).
// Per the design ruling (2026-08-04): these are stable GPU pointers that
// can be memcpy'd between L1 and L2 — restore, not rebuild.

#pragma once

#include "csrc/data_paths/local_nvme/metadata/tiered_handle_cache.h"
#include "csrc/data_paths/local_nvme/io/device_target.h"

#include <cstdint>
#include <functional>
#include <mutex>

namespace tutti::data_paths::local_nvme {

// The T type for TieredHandleCache — a small wrapper around
// DeviceTargetHandle* + overflow blob, both stable GPU pointers.
struct HandleCacheEntry {
    DeviceTargetHandle* handle = nullptr;
    void* overflow = nullptr;
    int open_refcount = 0;  // >0 while in use (cannot downgrade)
};

// Adapter: provides HandleWorkspaceCache-compatible get_or_build() API,
// backed by TieredHandleCache<HandleCacheEntry, uint64_t>.
class TieredHandleAdapter {
public:
    struct Config {
        std::uint32_t capacity = 0;       // L1 capacity (0 = disabled)
        std::uint32_t l2_capacity = 0;    // L2 capacity (0 = 4×L1)
        std::uint32_t cuda_device = 0;
    };

    struct Entry {
        std::uint64_t key = 0;
        DeviceTargetHandle* handle = nullptr;
        void* overflow = nullptr;
        std::uint32_t open_refcount = 0;
        // Internal: points to the TieredHandleCache entry for release
        HandleCacheEntry* tc_entry_ = nullptr;
    };

    TieredHandleAdapter() = default;
    ~TieredHandleAdapter() { shutdown(); }

    bool init(const Config& cfg) {
        if (cfg.capacity == 0) { cfg_ = cfg; return true; }
        TieredHandleCache<HandleCacheEntry, uint64_t>::Config tc_cfg;
        tc_cfg.l1_capacity = cfg.capacity;
        tc_cfg.l2_capacity = cfg.l2_capacity > 0 ? cfg.l2_capacity : cfg.capacity * 4;
        tc_cfg.cuda_device = cfg.cuda_device;
        if (!cache_.init(tc_cfg)) return false;
        cfg_ = cfg;
        initialized_ = true;
        return true;
    }

    void shutdown() {
        if (initialized_) {
            cache_.shutdown();
            initialized_ = false;
        }
    }

    bool enabled() const { return cfg_.capacity > 0 && initialized_; }

    // Compatible with HandleWorkspaceCache::get_or_build.
    // build_fn(handle_out, overflow_out) creates the GPU handle.
    template <typename BuildFn>
    Entry* get_or_build(std::uint64_t key, BuildFn&& build_fn, cudaStream_t stream = 0) {
        if (!enabled()) return nullptr;
        auto builder = [&](const uint64_t& k, HandleCacheEntry* out) -> bool {
            return build_fn(out->handle, out->overflow);
        };
        HandleCacheEntry* tc = cache_.get_or_build(key, builder, stream);
        if (!tc) return nullptr;
        // Return a borrowed Entry view — tc_entry_ is the real storage.
        borrowed_.key = key;
        borrowed_.handle = tc->handle;
        borrowed_.overflow = tc->overflow;
        borrowed_.open_refcount = ++tc->open_refcount;
        borrowed_.tc_entry_ = tc;
        return &borrowed_;
    }

    // Release (decrement open_refcount).  When refcount hits 0, the entry
    // becomes evictable (can be downgraded from L1 to L2).
    void release(Entry* e) {
        if (!e || !e->tc_entry_) return;
        --e->tc_entry_->open_refcount;
        e->tc_entry_ = nullptr;
    }

    void erase(std::uint64_t key) {
        cache_.erase(key);
    }

    void get_stats(uint64_t* l1_hits, uint64_t* l2_hits, uint64_t* evictions) const {
        if (l1_hits) *l1_hits = cache_.stats().l1_hits;
        if (l2_hits) *l2_hits = cache_.stats().l2_hits;
        if (evictions) *evictions = cache_.stats().l1_evictions;
    }

private:
    TieredHandleCache<HandleCacheEntry, uint64_t> cache_;
    Config cfg_{};
    bool initialized_ = false;
    Entry borrowed_{};  // single-threaded get_or_build → single borrowed slot
};

} // namespace tutti::data_paths::local_nvme

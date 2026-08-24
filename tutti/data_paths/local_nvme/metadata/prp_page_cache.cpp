// tutti/data_paths/local_nvme/metadata/prp_page_cache.cpp
//
// PrpPageCache init/shutdown + get_or_build (host-pinned pool; the miss
// path is a plain host memcpy — no CUDA, no stream, no events).

#include "tutti/data_paths/local_nvme/metadata/prp_page_cache.h"

#include "tutti/data_paths/local_nvme/io/prp_builder.h"

#include <nvm_dma.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

namespace tutti::data_paths::local_nvme {

bool PrpPageCache::init(const Config& cfg, nvm_ctrl_t* ctrl) {
    if (initialized_) return false;
    if (cfg.capacity == 0) { cfg_ = cfg; return true; }
    if (ctrl == nullptr) return false;

    cfg_ = cfg;
    ctrl_ = ctrl;

    // Allocate page-aligned host memory, then let nvm_dma_map_data_host pin
    // and DMA-map it.  CUDA host allocations are not valid backing for this
    // kernel get_user_pages path on every deployment: the mapping can succeed
    // while the controller observes stale/zero PRP-list entries.
    const std::size_t bytes =
        static_cast<std::size_t>(cfg_.capacity) * cfg_.page_size;
    const int alloc_rc = posix_memalign(&pool_host_, cfg_.page_size, bytes);
    if (alloc_rc != 0 || pool_host_ == nullptr) {
        std::fprintf(stderr,
                     "[prp_cache] init: posix_memalign(%zu bytes) failed: rc=%d\n",
                     bytes, alloc_rc);
        return false;
    }

    int rc = nvm_dma_map_data_host(&pool_dma_, ctrl, pool_host_, bytes);
    if (rc != 0 || pool_dma_ == nullptr) {
        std::fprintf(stderr,
                     "[prp_cache] init: nvm_dma_map_data_host failed: rc=%d\n",
                     rc);
        std::free(pool_host_);
        pool_host_ = nullptr;
        return false;
    }

    // Initialize entries + free list.
    entries_.resize(cfg_.capacity);
    free_list_.clear();
    for (std::uint32_t i = 0; i < cfg_.capacity; ++i)
        free_list_.push_back(i);
    index_.clear();
    lru_.clear();
    lru_pos_.clear();
    stats_ = {};

    initialized_ = true;
    return true;
}

void PrpPageCache::shutdown() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!initialized_) return;

    // DMA unmap FIRST, then free host memory (lives/dies together).
    if (pool_dma_) {
        nvm_dma_unmap(pool_dma_);
        pool_dma_ = nullptr;
    }
    if (pool_host_) {
        std::free(pool_host_);
        pool_host_ = nullptr;
    }

    entries_.clear();
    free_list_.clear();
    index_.clear();
    lru_.clear();
    lru_pos_.clear();
    stats_ = {};
    initialized_ = false;
}

PrpPageCache::Entry* PrpPageCache::get_or_build(const Key& key,
                                                const nvm_dma_t* data_dma) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!enabled()) return nullptr;
    if (data_dma == nullptr) return nullptr;

    auto it = index_.find(key);
    if (it != index_.end()) {
        std::uint32_t slot = it->second;
        Entry& hit = entries_[slot];
        ++hit.checkout_refcount;  // P0-1: re-checkout increments refcount
        hit.in_use = true;        // backward compat
        remove_from_lru_(slot);   // checked-out entries are not evictable
        ++stats_.hits;
        return &hit;
    }

    ++stats_.misses;

    std::uint32_t slot = acquire_slot_();
    if (slot == UINT32_MAX) return nullptr;

    Entry& e = entries_[slot];
    e.key = key;
    e.pin_count = 0;
    e.checkout_refcount = 1;  // P0-1: new entry starts with one checkout
    e.in_use = true;  // backward compat
    e.vaddr = static_cast<char*>(pool_host_) +
              static_cast<std::size_t>(slot) * cfg_.page_size;
    e.ioaddr = pool_dma_->ioaddrs[slot];

    // Fill: pure host memcpy of the IOVA table into the pool page.
    fill_prp_list_page(static_cast<std::uint64_t*>(e.vaddr), data_dma,
                       key.start_page, key.pages_in_io, cfg_.page_size);

    index_[key] = slot;
    // Do NOT add to LRU — entry is in_use (checked out by submit).
    // unpin() adds it to LRU when the op releases it.
    ++stats_.entries;
    return &e;
}

std::uint32_t PrpPageCache::get_or_build_batch(BatchItem* items,
                                               std::size_t count) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!enabled() || items == nullptr) return 0;

    std::uint32_t resolved = 0;
    for (std::size_t i = 0; i < count; ++i) {
        BatchItem& item = items[i];
        if (item.data_dma == nullptr) continue;

        auto it = index_.find(item.key);
        if (it != index_.end()) {
            std::uint32_t slot = it->second;
            Entry& hit = entries_[slot];
            ++hit.checkout_refcount;
            hit.in_use = true;
            remove_from_lru_(slot);
            ++stats_.hits;
            item.result = &hit;
            ++resolved;
            continue;
        }

        ++stats_.misses;
        std::uint32_t slot = acquire_slot_();
        if (slot == UINT32_MAX) continue;  // caller falls back to arena

        Entry& e = entries_[slot];
        e.key = item.key;
        e.pin_count = 0;
        e.checkout_refcount = 1;
        e.in_use = true;
        e.vaddr = static_cast<char*>(pool_host_) +
                  static_cast<std::size_t>(slot) * cfg_.page_size;
        e.ioaddr = pool_dma_->ioaddrs[slot];

        fill_prp_list_page(static_cast<std::uint64_t*>(e.vaddr),
                           item.data_dma, item.key.start_page,
                           item.key.pages_in_io, cfg_.page_size);

        index_[item.key] = slot;
        ++stats_.entries;
        item.result = &e;
        ++resolved;
    }
    return resolved;
}

} // namespace tutti::data_paths::local_nvme

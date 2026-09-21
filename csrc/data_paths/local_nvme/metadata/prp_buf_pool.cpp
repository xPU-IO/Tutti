// csrc/data_paths/local_nvme/metadata/prp_buf_pool.cpp
//
// R19 S3b REQUIRED 1: host-pinned PRP-list buffer pool implementation.

#include "csrc/data_paths/local_nvme/metadata/prp_buf_pool.h"

#include <nvm_dma.h>   // nvm_dma_map_data_host, nvm_dma_unmap

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace tutti::data_paths::local_nvme {

PrpBufPool::~PrpBufPool() {
    shutdown();
}

void PrpBufPool::shutdown(bool retain) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!retain) {
        for (auto& seg : segments_) {
            if (seg.dma) nvm_dma_unmap(seg.dma);
            if (seg.backing) std::free(seg.backing);
        }
    }
    // nvm_dma_t has no owning C++ destructor. Clearing with retain=true
    // deliberately leaves the DMA mapping and pinned host backing alive.
    segments_.clear();
    ctrl_ = nullptr;
    total_pages_ = 0;
    leased_pages_ = 0;
}

void PrpBufPool::init(nvm_ctrl_t* ctrl, std::uint64_t page_size) {
    std::lock_guard<std::mutex> lock(mtx_);
    ctrl_ = ctrl;
    page_size_ = page_size;
}

PrpBufRef PrpBufPool::alloc_pages(std::uint64_t n_pages) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (n_pages == 0 || !ctrl_) return {};

    // Reuse a released range first (first fit). Without this the pool grew by
    // a segment per cache-missing submit -- see the header's lifecycle note.
    for (auto& seg : segments_) {
        for (std::size_t i = 0; i < seg.free_ranges.size(); ++i) {
            Range& r = seg.free_ranges[i];
            if (r.count < n_pages) continue;
            PrpBufRef ref;
            ref.segment = seg.dma;
            ref.base_page = r.base;
            ref.num_pages = n_pages;
            ref.valid = true;
            if (r.count == n_pages) {
                seg.free_ranges.erase(seg.free_ranges.begin() +
                                      static_cast<std::ptrdiff_t>(i));
            } else {
                r.base += n_pages;
                r.count -= n_pages;
            }
            seg.leased_pages += n_pages;
            leased_pages_ += n_pages;
            return ref;
        }
    }

    // Try the tail of the segment with the most room left. Segments are
    // appended in order, but a lease may have been returned to an older one,
    // so scan for a segment whose bump region still has space.
    for (auto& seg : segments_) {
        if (seg.used_pages + n_pages <= seg.capacity_pages) {
            PrpBufRef ref;
            ref.segment = seg.dma;
            ref.base_page = seg.used_pages;
            ref.num_pages = n_pages;
            ref.valid = true;
            seg.used_pages += n_pages;
            seg.leased_pages += n_pages;
            leased_pages_ += n_pages;
            return ref;
        }
    }

    // Need a new segment. Size = max(segment_pages_, n_pages rounded up).
    std::uint64_t seg_pages = segment_pages_;
    if (n_pages > seg_pages) {
        seg_pages = ((n_pages + segment_pages_ - 1) / segment_pages_) * segment_pages_;
    }

    const std::uint64_t seg_bytes = seg_pages * page_size_;
    void* backing = nullptr;
    const int alloc_rc = posix_memalign(
        &backing, static_cast<std::size_t>(page_size_),
        static_cast<std::size_t>(seg_bytes));
    if (alloc_rc != 0 || backing == nullptr) return {};
    std::memset(backing, 0, static_cast<std::size_t>(seg_bytes));
    nvm_dma_t* dma = nullptr;
    int rc = nvm_dma_map_data_host(&dma, ctrl_, backing,
                                   static_cast<size_t>(seg_bytes));
    if (rc != 0 || !dma) {
        std::free(backing);
        return {};
    }

    Segment seg;
    seg.dma = dma;
    seg.backing = backing;
    seg.capacity_pages = seg_pages;
    seg.used_pages = n_pages;
    seg.leased_pages = n_pages;
    segments_.push_back(std::move(seg));
    total_pages_ += seg_pages;
    leased_pages_ += n_pages;

    PrpBufRef ref;
    ref.segment = dma;
    ref.base_page = 0;
    ref.num_pages = n_pages;
    ref.valid = true;
    return ref;
}

void PrpBufPool::release_pages(nvm_dma_t* segment, std::uint64_t base_page,
                               std::uint64_t num_pages) {
    if (segment == nullptr || num_pages == 0) return;
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& seg : segments_) {
        if (seg.dma != segment) continue;
        if (base_page + num_pages > seg.capacity_pages) return;  // bogus range
        seg.leased_pages = (seg.leased_pages >= num_pages)
                               ? seg.leased_pages - num_pages
                               : 0;
        leased_pages_ = (leased_pages_ >= num_pages) ? leased_pages_ - num_pages
                                                     : 0;
        auto it = std::lower_bound(
            seg.free_ranges.begin(), seg.free_ranges.end(), base_page,
            [](const Range& r, std::uint64_t value) { return r.base < value; });
        seg.free_ranges.insert(it, Range{base_page, num_pages});
        // Merge with the neighbours so repeated recycles do not fragment the
        // free list into unbounded single-page entries.
        std::sort(seg.free_ranges.begin(), seg.free_ranges.end(),
                  [](const Range& a, const Range& b) { return a.base < b.base; });
        std::vector<Range> merged;
        merged.reserve(seg.free_ranges.size());
        for (const Range& r : seg.free_ranges) {
            if (!merged.empty() &&
                merged.back().base + merged.back().count == r.base) {
                merged.back().count += r.count;
            } else {
                merged.push_back(r);
            }
        }
        seg.free_ranges.swap(merged);
        return;
    }
}

} // namespace tutti::data_paths::local_nvme

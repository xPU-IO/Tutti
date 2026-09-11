// tutti/data_paths/local_nvme/metadata/prp_buf_pool.cpp
//
// R19 S3b REQUIRED 1: host-pinned PRP-list buffer pool implementation.

#include "tutti/data_paths/local_nvme/metadata/prp_buf_pool.h"

#include <nvm_dma.h>   // nvm_dma_map_data_host, nvm_dma_unmap

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
}

void PrpBufPool::init(nvm_ctrl_t* ctrl, std::uint64_t page_size) {
    std::lock_guard<std::mutex> lock(mtx_);
    ctrl_ = ctrl;
    page_size_ = page_size;
}

PrpBufRef PrpBufPool::alloc_pages(std::uint64_t n_pages) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (n_pages == 0 || !ctrl_) return {};

    // Try current segment first.
    if (!segments_.empty()) {
        Segment& cur = segments_.back();
        if (cur.used_pages + n_pages <= cur.capacity_pages) {
            PrpBufRef ref;
            ref.segment = cur.dma;
            ref.base_page = cur.used_pages;
            ref.num_pages = n_pages;
            ref.valid = true;
            cur.used_pages += n_pages;
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
    segments_.push_back(seg);
    total_pages_ += seg_pages;

    PrpBufRef ref;
    ref.segment = dma;
    ref.base_page = 0;
    ref.num_pages = n_pages;
    ref.valid = true;
    return ref;
}

} // namespace tutti::data_paths::local_nvme

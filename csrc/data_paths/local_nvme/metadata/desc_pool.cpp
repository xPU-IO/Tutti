// tutti/data_paths/local_nvme/metadata/desc_pool.cpp
//
// R19 S3 REQUIRED 3: GPU descriptor pool implementation.

#include "csrc/data_paths/local_nvme/metadata/desc_pool.h"

#include <tutti/cuda_like.h>

namespace tutti::data_paths::local_nvme {

DescPool::~DescPool() {
    for (auto& seg : segments_) {
        if (seg.base) cudaFree(seg.base);
    }
}

void* DescPool::alloc(std::uint64_t n_bytes) {
    if (n_bytes == 0) return nullptr;

    // Try current segment first.
    if (!segments_.empty()) {
        Segment& cur = segments_.back();
        if (cur.used + n_bytes <= cur.capacity) {
            void* ptr = static_cast<char*>(cur.base) + cur.used;
            cur.used += n_bytes;
            return ptr;
        }
    }

    // Need a new segment. Size = max(segment_size_, n_bytes rounded up to
    // segment_size_ granularity).
    std::uint64_t seg_size = segment_size_;
    if (n_bytes > seg_size) {
        // Round up to segment_size_ boundary.
        seg_size = ((n_bytes + segment_size_ - 1) / segment_size_) * segment_size_;
    }

    void* base = nullptr;
    cudaError_t ce = cudaMalloc(&base, seg_size);
    if (ce != cudaSuccess) return nullptr;

    segments_.push_back({base, seg_size, n_bytes});
    total_bytes_ += seg_size;
    return base;
}

} // namespace tutti::data_paths::local_nvme

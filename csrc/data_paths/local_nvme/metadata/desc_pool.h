#pragma once

// tutti/data_paths/local_nvme/metadata/desc_pool.h
//
// GPU descriptor pool — replaces per-registration cudaMalloc for
// pre-built AddressDescriptor[] arrays.
//
// R19 S3 REQUIRED 3: at 180GB KV / 128KiB tensor = 1.47M registrations,
// per-registration cudaMalloc causes minute-level latency and millions
// of small allocations. This pool pre-allocates a large GPU buffer and
// sub-allocates from it via a bump allocator with segment growth.
//
// Lifecycle:
//   - alloc(n) returns a GPU ptr from the current segment; grows if needed.
//   - free(ptr) is a no-op (segments are freed on pool shutdown).
//   - shutdown() cudaFree's all segments.
//
// This is a simple bump allocator — no per-free recycling. Registrations
// are permanent (KV cache lifetime), so free is never called in production.
// Unregister reclaims nothing (the segment is freed on DataPath shutdown).

#include <cstdint>
#include <vector>

namespace tutti::data_paths::local_nvme {

class DescPool {
public:
    // Default segment size: 256 MiB. Each segment is one cudaMalloc.
    // At 24 bytes per descriptor, one segment holds ~11.2M descriptors.
    static constexpr std::uint64_t DEFAULT_SEGMENT_BYTES = 256ULL * 1024 * 1024;

    DescPool() = default;
    ~DescPool();

    DescPool(const DescPool&) = delete;
    DescPool& operator=(const DescPool&) = delete;

    // Allocate n bytes from the pool. Returns nullptr on failure.
    // The returned pointer is GPU-accessible (cudaMalloc'd).
    void* alloc(std::uint64_t n_bytes);

    // Total bytes allocated across all segments.
    std::uint64_t total_capacity() const { return total_bytes_; }

    // Number of segments allocated (for diagnostics).
    std::size_t num_segments() const { return segments_.size(); }

private:
    struct Segment {
        void* base = nullptr;        // cudaMalloc'd GPU buffer
        std::uint64_t capacity = 0;  // total bytes in this segment
        std::uint64_t used = 0;     // bytes already sub-allocated
    };

    std::vector<Segment> segments_;
    std::uint64_t total_bytes_ = 0;
    std::uint64_t segment_size_ = DEFAULT_SEGMENT_BYTES;
};

} // namespace tutti::data_paths::local_nvme

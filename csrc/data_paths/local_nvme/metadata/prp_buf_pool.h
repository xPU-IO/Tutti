#pragma once

// csrc/data_paths/local_nvme/metadata/prp_buf_pool.h
//
// Host-pinned PRP-list buffer pool — replaces per-registration
// nvm_dma_map_data_host for pre-built PRP-list pages.
//
// R19 S3b REQUIRED 1: at 180GB KV / 128KiB tensor = 1.47M registrations,
// per-registration nvm_dma_map_data_host causes minute-level latency
// (each call is a kernel ioctl). This pool pre-maps a large host-pinned
// DMA buffer via one nvm_dma_map_data_host per segment, then sub-allocates
// PRP pages from it.
//
// NOTE: This is NOT the legacy "prp_list_pool" that was cleaned up in
// R16 S6b. That was a fixed-IOVA scatter-patch mechanism (deleted).
// This pool is a host-pinned PRP buffer allocator — different purpose,
// different implementation. The name "prp_buf_pool" avoids confusion.
//
// Lifecycle:
//   - alloc_pages(n) returns a PrpBufRef (segment ptr + base page index);
//     reuses a released range when one fits, else grows by a segment (new
//     nvm_dma_map_data_host per segment).
//   - release_pages() returns a range for reuse. Pages MUST be released once
//     the command that referenced them has completed: the fallback path takes
//     pages on every cache miss, so a pool without reuse grows by a segment
//     per submit (measured: 250 KiB per submit ≈ 48 GB in 40 minutes on the
//     8-GPU online deployment, i.e. an OOM).
//   - Use PrpBufLease to hold a range across the submit function: it releases
//     on every early exit unless the op takes ownership (disown()).
//   - shutdown nvm_dma_unmap's all segments.
//
// Thread safety: internal mutex covers init/growth/release/shutdown because
// dynamic submit misses may arrive from multiple host threads.

#include <cstdint>
#include <mutex>
#include <vector>

// nvm_dma.h pulls in nvm_types.h which defines nvm_dma_t and nvm_ctrl_t
// as anonymous-struct typedefs (no tag name). Include the full header so
// all types are available; cannot forward-declare anonymous typedefs.
#include <nvm_dma.h>

namespace tutti::data_paths::local_nvme {

// Opaque reference to a sub-allocated range within the pool.
struct PrpBufRef {
    nvm_dma_t* segment = nullptr;     // the nvm_dma segment (for ioaddrs/vaddr)
    std::uint64_t base_page = 0;     // page index within this segment
    std::uint64_t num_pages = 0;     // pages allocated
    bool valid = false;
};

class PrpBufPool {
public:
    // Default segment size: 16 MiB = 4096 pages. The pool grows by adding
    // segments; a small dynamic LIST miss must not require pinning 256 MiB.
    // Requests larger than one segment are rounded up to a segment multiple.
    static constexpr std::uint64_t DEFAULT_SEGMENT_PAGES = 4096ULL;

    PrpBufPool() = default;
    ~PrpBufPool();

    PrpBufPool(const PrpBufPool&) = delete;
    PrpBufPool& operator=(const PrpBufPool&) = delete;

    // Initialize with the controller handle (needed for nvm_dma_map_data_host).
    void init(nvm_ctrl_t* ctrl, std::uint64_t page_size);

    // Allocate n_pages from the pool. Returns a PrpBufRef; .valid=false on
    // failure. The caller uses segment->ioaddrs[base_page + i] and
    // segment->vaddr + (base_page + i) * page_size for IOVA/virtual access.
    PrpBufRef alloc_pages(std::uint64_t n_pages);

    // Return a range obtained from alloc_pages() for reuse. Must not be called
    // while a command that references the pages is in flight.
    void release_pages(nvm_dma_t* segment, std::uint64_t base_page,
                       std::uint64_t num_pages);

    // Unmap all host-pinned segments.  retain=true intentionally leaks the
    // mappings/backing after a controller timeout; the controller may still
    // fetch a PRP list.  Idempotent.
    void shutdown(bool retain = false);

    // Total pages mapped across all segments (capacity, not usage).
    std::uint64_t total_pages() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return total_pages_;
    }

    // Pages currently handed out. This is what must stay bounded by the
    // in-flight working set; total_pages() only follows it.
    std::uint64_t leased_pages() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return leased_pages_;
    }

    // Number of segments allocated (for diagnostics / dma_map count).
    std::size_t num_segments() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return segments_.size();
    }

private:
    struct Range {
        std::uint64_t base = 0;
        std::uint64_t count = 0;
    };

    struct Segment {
        nvm_dma_t* dma = nullptr;      // nvm_dma_map_data_host'd
        void* backing = nullptr;       // page-aligned caller-owned host memory
        std::uint64_t capacity_pages = 0;  // total pages in this segment
        std::uint64_t used_pages = 0;     // bump high-water within this segment
        std::uint64_t leased_pages = 0;   // handed out right now
        // Ranges handed back by release_pages(), address-ordered and merged.
        std::vector<Range> free_ranges;
    };

    std::vector<Segment> segments_;
    nvm_ctrl_t* ctrl_ = nullptr;
    std::uint64_t page_size_ = 4096;
    std::uint64_t total_pages_ = 0;
    std::uint64_t leased_pages_ = 0;
    std::uint64_t segment_pages_ = DEFAULT_SEGMENT_PAGES;
    mutable std::mutex mtx_;
};

// A pool sub-allocation paired with the pool that owns it. Multi-device
// callers (striped) hold one per device, so the completion path can return
// pages without reconstructing which pool served which range.
struct PrpBufLeaseRef {
    PrpBufPool* pool = nullptr;
    nvm_dma_t* segment = nullptr;
    std::uint64_t base_page = 0;
    std::uint64_t num_pages = 0;
};

// A range owned by a lease: returns the pages to its pool on destruction
// unless disown() was called, i.e. an op took ownership. Every early exit
// between alloc_pages() and "the op owns the pages" is then covered without
// having to enumerate the error paths by hand.
class PrpBufLease {
public:
    PrpBufLease() = default;
    PrpBufLease(PrpBufPool* pool, const PrpBufRef& ref)
        : pool_(ref.valid ? pool : nullptr),
          dma_(ref.segment),
          base_(ref.base_page),
          pages_(ref.num_pages) {}
    ~PrpBufLease() { release(); }

    PrpBufLease(const PrpBufLease&) = delete;
    PrpBufLease& operator=(const PrpBufLease&) = delete;
    // Move-only: striped holds one lease per device in a vector.
    PrpBufLease(PrpBufLease&& other) noexcept
        : pool_(other.pool_), dma_(other.dma_), base_(other.base_),
          pages_(other.pages_) {
        other.pool_ = nullptr;
        other.pages_ = 0;
    }
    PrpBufLease& operator=(PrpBufLease&& other) noexcept {
        if (this != &other) {
            release();
            pool_ = other.pool_;
            dma_ = other.dma_;
            base_ = other.base_;
            pages_ = other.pages_;
            other.pool_ = nullptr;
            other.pages_ = 0;
        }
        return *this;
    }

    PrpBufPool* pool() const { return pool_; }

    void release() {
        if (pool_ != nullptr && pages_ != 0) {
            pool_->release_pages(dma_, base_, pages_);
        }
        pool_ = nullptr;
        pages_ = 0;
    }

    // The op now owns the pages; the release path will return them.
    void disown() {
        pool_ = nullptr;
        pages_ = 0;
    }

    nvm_dma_t* segment() const { return dma_; }
    std::uint64_t base_page() const { return base_; }
    std::uint64_t num_pages() const { return pages_; }
    bool owns() const { return pool_ != nullptr && pages_ != 0; }

private:
    PrpBufPool* pool_ = nullptr;
    nvm_dma_t* dma_ = nullptr;
    std::uint64_t base_ = 0;
    std::uint64_t pages_ = 0;
};

} // namespace tutti::data_paths::local_nvme

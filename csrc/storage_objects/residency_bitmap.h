#pragma once

// csrc/storage_objects/residency_bitmap.h -- Cross-rank residency, mmap-backed.
//
// IMPLEMENTATION DETAIL. Reached only through the SPI.
//
// One file per rank. Each rank mmaps its own file for writing; bit s means
// "slot s is committed on this rank". Setting and clearing touch memory only.
// Durability comes from kernel writeback plus a periodic msync, so the content
// is EVENTUALLY consistent by construction.
//
// One file per rank rather than one shared file: writers never conflict, there
// is no cross-process false sharing, and no atomic bit operations are needed. A
// reader maps all ranks' files and ANDs them to get "committed everywhere".
// This shape also survives ranks moving to separate hosts -- each keeps a local
// file and the AND becomes an aggregation.
//
// THE BITMAP IS NEVER AUTHORITATIVE, AND ITS SKEW DIRECTION IS PINNED.
//
//   bitmap says absent, actually present -> recomputation. Acceptable.
//   bitmap says present, actually absent -> truncated KV loaded silently.
//                                          Forbidden.
//
// Dirty mmap pages are written back on the kernel's schedule, so after a crash
// it is unknowable which pages landed. "The bitmap says present" is therefore
// untrustworthy on its own, and two mechanisms keep the error one-sided:
//
//   1. Recovery treats the bitmap as candidates only. Authority is the
//      object-header cross-check, and the bitmap is REBUILT from the verified
//      set rather than trusted as found on media.
//   2. If the bitmap is unusable -- missing, bad magic, fingerprint mismatch,
//      slot_count mismatch -- the query degrades to this rank's own view.
//      Fewer hits, never wrong hits.
//
// Because the error is one-sided, the msync period is a pure performance knob.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <tutti/status.h>
#include <tutti/spi/storage_object_store.h>

namespace tutti::storage_objects {

// Total file size for `slot_count` slots, including the header and rounded to a
// page so the body starts and ends page-aligned.
std::uint64_t residency_file_bytes(std::uint64_t slot_count) noexcept;

// Byte offset of the bit for `slot` within the file, and its bit position.
// Exposed for tests that need to corrupt a specific bit.
std::uint64_t residency_bit_offset(std::uint64_t slot) noexcept;
std::uint32_t residency_bit_shift(std::uint64_t slot) noexcept;

// Encode the 4096-byte header. Kept separate from the mapping code so it can be
// unit-tested without touching a filesystem.
void encode_residency_header(std::uint8_t* out, std::uint32_t rank_id,
                            std::uint32_t rank_count, std::uint64_t slot_count,
                            std::uint64_t fingerprint_digest) noexcept;

enum class ResidencyRejection {
    kNone = 0,
    kEmptyFile,            // never initialised
    kBadMagic,
    kUnsupportedVersion,
    kTruncatedBuffer,
    kSlotCountMismatch,    // geometry changed under us
    kFingerprintMismatch,  // different namespace
    kRankCountMismatch,    // topology changed
};

const char* to_string(ResidencyRejection rejection) noexcept;

// Validate a mapped header against what this process expects. Any rejection
// means "treat the bitmap as unavailable", which is always safe.
ResidencyRejection validate_residency_header(
    const std::uint8_t* data, std::size_t data_bytes, std::uint32_t rank_count,
    std::uint64_t slot_count, std::uint64_t fingerprint_digest) noexcept;

// -------------------------------------------------------------------------
// ResidencyBitmap -- an mmapped view over one rank's file.
//
// Not copyable; movable. Destruction unmaps but does NOT msync: an explicit
// sync() is the only way to force durability, keeping the cost visible at the
// call site rather than hidden in a destructor.
// -------------------------------------------------------------------------
class ResidencyBitmap {
public:
    ResidencyBitmap() = default;
    ~ResidencyBitmap();

    ResidencyBitmap(const ResidencyBitmap&) = delete;
    ResidencyBitmap& operator=(const ResidencyBitmap&) = delete;
    ResidencyBitmap(ResidencyBitmap&& other) noexcept;
    ResidencyBitmap& operator=(ResidencyBitmap&& other) noexcept;

    // Create or open `path` for this rank and map it writable. Creates and
    // initialises the file when absent; validates the header when present.
    //
    // On validation failure the mapping is NOT established and the object stays
    // !usable(). The caller must then fall back to the single-rank path rather
    // than reinitialising, since silently zeroing a peer's bitmap would destroy
    // information the peer still needs.
    Status open_writable(const std::string& path, std::uint32_t rank_id,
                         std::uint32_t rank_count, std::uint64_t slot_count,
                         std::uint64_t fingerprint_digest);

    // Map another rank's file read-only. Never creates.
    Status open_readonly(const std::string& path, std::uint32_t rank_count,
                         std::uint64_t slot_count,
                         std::uint64_t fingerprint_digest);

    void close();

    bool usable() const noexcept { return base_ != nullptr; }
    std::uint64_t slot_count() const noexcept { return slot_count_; }

    // Bit access. Out-of-range slots read false and ignore writes: a geometry
    // disagreement must not corrupt neighbouring bits.
    bool test(std::uint64_t slot) const noexcept;
    void set(std::uint64_t slot) noexcept;
    void clear(std::uint64_t slot) noexcept;

    // Clear every bit. Used by recovery, which rebuilds the bitmap from the
    // header-verified set instead of trusting what it found.
    void clear_all() noexcept;

    // Flush dirty pages. Cost is visible here by design.
    Status sync();

private:
    void* base_ = nullptr;
    std::size_t mapped_bytes_ = 0;
    std::uint64_t slot_count_ = 0;
    bool writable_ = false;
};

// -------------------------------------------------------------------------
// all_ranks_committed
//
// AND of every rank's bit for `slot`. Returns false if ANY rank's bitmap is
// unusable -- the safe direction, since claiming cross-rank knowledge we do not
// have is precisely the forbidden failure.
// -------------------------------------------------------------------------
bool all_ranks_committed(const std::vector<const ResidencyBitmap*>& ranks,
                        std::uint64_t slot) noexcept;

} // namespace tutti::storage_objects

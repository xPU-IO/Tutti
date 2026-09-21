#pragma once

// csrc/storage_objects/slot_media.h -- physical slot operations.
//
// IMPLEMENTATION DETAIL. Reached only through the SPI.
//
// Everything here uses O_DIRECT, per project policy for data files. On a
// GPU-direct path buffered IO is actively harmful, not merely slower:
//   * page cache pollution competes for memory with the KV pool;
//   * writeback competes with GPU DMA for device bandwidth;
//   * after a GPU DMA write the page cache holds STALE data, so a later
//     buffered read can return the old contents -- a correctness bug, not a
//     performance one.
// O_DIRECT therefore requires 4096-alignment of buffer, offset and length,
// which is why every structure in this layer is sized in whole 4096-byte units
// and why buffers come from posix_memalign rather than std::vector.
//
// This layer performs HOST-side metadata IO only: object headers, precreating
// space, re-zeroing on reclaim. Payload IO is the caller's job via the
// GPU-direct DataPath, using an ObjectPlacement. Keeping payload out of here is
// what preserves the premise that KV data never passes through host memory.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <tutti/status.h>

namespace tutti::storage_objects {

// -------------------------------------------------------------------------
// AlignedBuffer -- posix_memalign-backed, for O_DIRECT.
//
// std::vector cannot be used: its data has no alignment guarantee beyond the
// element type, and an unaligned O_DIRECT buffer fails with EINVAL.
// -------------------------------------------------------------------------
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t bytes);
    ~AlignedBuffer();

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

    std::uint8_t* data() noexcept { return data_; }
    const std::uint8_t* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool valid() const noexcept { return data_ != nullptr; }

    void zero() noexcept;

private:
    std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
};

// -------------------------------------------------------------------------
// materialise_slot
//
// Creates each shard file at exactly `bytes_per_shard` and writes REAL ZEROS
// over the whole extent, then fsyncs the file and its directory.
//
// Real zeros, not fallocate or a sparse hole: the resolver maps a file to
// physical extents via FIEMAP and fail-closed rejects UNWRITTEN and DELALLOC
// extents, because DMA cannot target blocks the filesystem has not actually
// allocated. This is the single reason materialisation costs real time --
// measured at roughly 225 MB/s per rank, so a terabyte of prewarm is an hour,
// not a moment. Callers must treat it as such.
//
// Idempotent: a file already at the right size with allocated extents is left
// alone, so restarting into an existing pool does not rewrite it.
Status materialise_slot(const std::vector<std::string>& paths,
                        std::uint64_t bytes_per_shard);

// Probe only: whether every shard already occupies `bytes_per_shard` on media.
// Creates nothing -- unlike materialise_slot this does not even O_CREAT, so it
// can be used to prove that a slot is reusable without touching the media.
// A missing shard is (*out = false), not an error: the caller leaves the slot
// to the grower. Real IO errors are reported, so an unreadable pool fails
// loudly instead of looking like an empty one.
Status slot_is_precreated(const std::vector<std::string>& paths,
                            std::uint64_t bytes_per_shard, bool* out);

// Rewrite zeros over a slot's payload region, for reclamation. Does NOT resize.
// The header region is zeroed too, which is what actually invalidates the
// object: a zero magic decodes as "never written" rather than as corruption.
Status zero_slot(const std::vector<std::string>& paths,
                 std::uint64_t bytes_per_shard);

// -------------------------------------------------------------------------
// Header IO
//
// The header is exactly one 4096-byte block, so both directions are a single
// aligned operation.
//
// write_object_header fsyncs before returning. This is THE commit point: the
// caller must have already made the payload durable, because the ordering
// "payload fsync, then header fsync" is what makes a valid header mean the
// payload was durable. Reversing it would permit a header that describes data
// which never landed -- the one failure this layer must not have.
Status write_object_header(const std::string& path, std::uint64_t offset,
                           const std::uint8_t* header, std::size_t header_bytes);

// Reads one header block. A short read at EOF is reported as NOT_FOUND rather
// than an error: a file shorter than its header simply has no object.
Status read_object_header(const std::string& path, std::uint64_t offset,
                          std::uint8_t* out, std::size_t out_bytes);

// -------------------------------------------------------------------------
// Checkpoint and bitmap IO
//
// Checkpoint containers are whole 4096-byte multiples by construction, so they
// are written with O_DIRECT like everything else. The residency bitmap is
// deliberately NOT here: it is mmapped, where O_DIRECT is meaningless.
// -------------------------------------------------------------------------
Status write_checkpoint_container(const std::string& path, std::uint64_t offset,
                                  const std::uint8_t* image,
                                  std::size_t image_bytes);

Status read_checkpoint_container(const std::string& path, std::uint64_t offset,
                                 std::uint8_t* out, std::size_t out_bytes);

// Create the file if absent and ensure it is at least `bytes`, precreating
// with real zeros. Used for the checkpoint region.
Status ensure_metadata_file(const std::string& path, std::uint64_t bytes);

// Create a directory and every missing parent. Returns OK if it already exists.
Status ensure_directory(const std::string& path);

// fsync a directory, so a file creation or rename inside it is durable. A file's
// own fsync does not make its directory entry durable.
Status sync_directory(const std::string& path);

} // namespace tutti::storage_objects

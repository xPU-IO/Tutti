#pragma once

// csrc/storage_objects/checkpoint_region.h -- Metadata checkpoint codec.
//
// IMPLEMENTATION DETAIL. Reached only through the SPI; see
// object_header_codec.h for the include-prefix rationale.
//
// The checkpoint records "which key lives at which offset" so a cold start
// does not have to read every object header. Mirrored containers are written
// alternately, each carrying a monotonic sequence number and a CRC over its
// body; recovery picks the highest sequence that passes CRC.
//
// Atomicity without rename or temp files: the container being written is never
// the one currently holding the newest valid state, so a crash mid-write
// leaves the previous container intact and selectable.
//
// THE CHECKPOINT IS AN ACCELERATOR, NOT THE TRUTH. Losing every container is
// survivable by scanning object headers, at the cost of reading them all. This
// is a deliberate departure from designs where a manifest is the sole record
// of what exists -- there, losing it makes all data unusable, which is a
// failure mode this layer refuses to have.
//
// Consequence for this codec: decode failures are never fatal. They degrade to
// "no checkpoint available", and the caller falls back to a header scan.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <tutti/spi/object_digest.h>
#include <tutti/spi/storage_object_store.h>

namespace tutti::storage_objects {

// One recorded object placement. Deliberately not ObjectPlacement: that type
// carries a borrowed ResolvedTarget pointer, which is a runtime concept with no
// meaning on media. Persisting only slot + geometry keeps the format
// independent of how targets happen to be resolved in a given process.
struct CheckpointEntry {
    ObjectKey key;
    std::uint64_t slot = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t generation = 0;
    std::uint64_t commit_seq = 0;
};

enum class CheckpointRejection {
    kNone = 0,
    kEmptyContainer,      // all-zero: never written
    kBadMagic,
    kUnsupportedVersion,
    kTruncatedBuffer,     // container header or body incomplete
    kBadBodyCrc,          // torn write
    kMalformedBody,       // internally inconsistent entry stream
};

const char* to_string(CheckpointRejection rejection) noexcept;

// -------------------------------------------------------------------------
// encode_checkpoint
//
// Produces a full container image: kContainerHeaderBytes of header followed by
// the entry stream, padded so the total is a multiple of 4096 (O_DIRECT).
//
// Body format, all little-endian:
//   u64 entry_count
//   then per entry:
//     u32 key_len, u64 slot, u64 payload_bytes, u64 generation,
//     u64 commit_seq, key_len bytes of key
//
// Variable-length keys are length-prefixed rather than fixed-width because the
// store treats keys as opaque byte strings of caller-chosen length; assuming a
// width here would silently truncate a longer key.
// -------------------------------------------------------------------------
std::vector<std::uint8_t> encode_checkpoint(
    const std::vector<CheckpointEntry>& entries, std::uint64_t sequence);

// -------------------------------------------------------------------------
// decode_checkpoint
//
// Validates magic, version, declared body length and body CRC, then parses
// entries. Bounds are re-checked at every step: a corrupted length field must
// not be able to walk the parser off the buffer, and returns kMalformedBody
// rather than reading out of range.
//
// `sequence` and `entries` are populated only on success.
// -------------------------------------------------------------------------
CheckpointRejection decode_checkpoint(const std::uint8_t* data,
                                      std::size_t data_bytes,
                                      std::uint64_t* sequence,
                                      std::vector<CheckpointEntry>* entries);

// -------------------------------------------------------------------------
// Container placement
//
// Containers are laid out back to back at the start of the metadata region.
// Each is sized to hold the largest checkpoint the store can produce, so a
// write never spills into its neighbour.
// -------------------------------------------------------------------------

// Upper bound on the encoded size for `max_entries` entries whose keys are at
// most `max_key_bytes` each, rounded up to 4096.
std::uint64_t checkpoint_container_bytes(std::uint64_t max_entries,
                                         std::uint32_t max_key_bytes) noexcept;

// Byte offset of container `index` within the metadata region.
std::uint64_t checkpoint_container_offset(std::uint32_t index,
                                          std::uint64_t container_bytes) noexcept;

// Which container to write next, given the sequence numbers currently found in
// each. Chooses the container NOT holding the newest valid state, which is what
// makes the scheme crash-atomic.
//
// `sequences[i] == 0` means container i is empty or unreadable. With all
// containers empty, returns 0.
std::uint32_t next_checkpoint_container(const std::uint64_t* sequences,
                                        std::uint32_t count) noexcept;

// Which container to read, i.e. the one with the highest nonzero sequence.
// Returns false when no container holds a valid checkpoint -- the caller must
// then fall back to a header scan rather than treating the store as empty.
bool select_checkpoint_container(const std::uint64_t* sequences,
                                std::uint32_t count,
                                std::uint32_t* chosen) noexcept;

} // namespace tutti::storage_objects

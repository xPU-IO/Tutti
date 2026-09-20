#pragma once

// csrc/storage_objects/object_header_codec.h -- Object header encode/decode.
//
// IMPLEMENTATION DETAIL. Not reachable as <tutti/...>; consumers go through
// <tutti/spi/storage_object_store.h>. The repo-root-relative include prefix
// ("csrc/...") is what keeps this out of the public namespace, and
// tests/header_hygiene asserts it stays that way.
//
// The header is the sole record that an object's payload is valid; there are
// no marker files. A valid header IS the validity statement. This file owns
// the byte-level format described by tutti::ObjectHeaderLayout.
//
// Design consequences that this codec must not violate:
//
//   * One header write per object, not per segment. Callers may write
//     segments in any order or only some of them: before the header lands the
//     object is uniformly invalid. "If the write did not finish, treat it as
//     never written" is therefore structural, not a convention.
//
//   * Metadata only. The CRC covers the header's own leading bytes and the
//     key bytes. The payload is never checksummed -- that would pull KV data
//     through the host CPU and defeat the GPU-direct premise.
//
//   * Fail-closed decode. Any inconsistency yields a typed rejection rather
//     than a partially trusted header. A header that cannot be fully
//     validated is treated as absent, which is the safe direction: the object
//     gets recomputed.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <tutti/spi/object_digest.h>
#include <tutti/spi/storage_object_store.h>

namespace tutti::storage_objects {

// Why a header is rejected. Distinguished because they mean different things
// operationally: kEmptySlot is the expected state of never-written space,
// while kBadHeaderCrc indicates a torn write, and kIdentityMismatch indicates
// stale space reuse. Only the latter two are worth reporting as anomalies.
enum class HeaderRejection {
    kNone = 0,
    kEmptySlot,          // all-zero magic: space was never committed
    kBadMagic,           // non-zero but not ours: foreign or corrupt data
    kUnsupportedVersion, // written by a newer format
    kBadHeaderCrc,       // torn or corrupted header write
    kIdentityMismatch,   // header holds a different key than expected
    kKeyCrcMismatch,     // identity digest collided; full key disagrees
    kPayloadBytesMismatch,
    kGenerationMismatch, // ABA: space was recycled since the record was made
    kTruncatedBuffer,    // caller supplied fewer than kHeaderBytes
};

const char* to_string(HeaderRejection rejection) noexcept;

// Decoded header contents. Only produced when validation fully succeeded.
struct ObjectHeaderFields {
    std::uint64_t identity = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t generation = 0;
    std::uint64_t commit_seq = 0;
    std::uint32_t key_len = 0;
    std::uint32_t key_crc32 = 0;
};

// What a decoded header is checked against. Every field is compared because
// the identity digest alone is only 64 bits: a collision must not be able to
// admit the wrong object.
struct HeaderExpectation {
    const ObjectKey* key = nullptr;   // required
    std::uint64_t payload_bytes = 0;  // required, must be nonzero
    std::uint64_t generation = 0;
    bool check_generation = true;     // false during header-scan recovery
};

// -------------------------------------------------------------------------
// encode_object_header
//
// Writes exactly ObjectHeaderLayout::kHeaderBytes into `out`, which must be
// at least that large and O_DIRECT aligned if it will be written directly.
// Bytes beyond the defined fields are zeroed, so the encoding is
// deterministic: the same inputs always produce identical media content,
// which makes a torn write detectable and makes tests reproducible.
//
// Returns false only if `out` is too small or the inputs are unusable
// (null/empty key, zero payload_bytes). Never partially fills the buffer.
// -------------------------------------------------------------------------
bool encode_object_header(std::uint8_t* out, std::size_t out_bytes,
                          const ObjectKey& key, std::uint64_t payload_bytes,
                          std::uint64_t generation,
                          std::uint64_t commit_seq) noexcept;

// -------------------------------------------------------------------------
// decode_object_header
//
// Validates and extracts, in the order below. The order matters: cheaper and
// more-specific checks come first so the reported rejection names the actual
// problem rather than a downstream symptom.
//
//   1. buffer large enough                  -> kTruncatedBuffer
//   2. magic all-zero                       -> kEmptySlot
//   3. magic not ours                       -> kBadMagic
//   4. version supported                    -> kUnsupportedVersion
//   5. header CRC over [0, kHeaderCrc32Offset) -> kBadHeaderCrc
//   6. identity == expected key's identity  -> kIdentityMismatch
//   7. key_len/key_crc32 == expected key's  -> kKeyCrcMismatch
//   8. payload_bytes == expected            -> kPayloadBytesMismatch
//   9. generation == expected (if checked)   -> kGenerationMismatch
//
// Checking the CRC before the identity is deliberate: a torn write can
// produce an arbitrary identity, and reporting kIdentityMismatch for it would
// mislead an operator into suspecting key derivation rather than a bad write.
//
// `fields` is populated only on success (returns kNone).
// -------------------------------------------------------------------------
HeaderRejection decode_object_header(const std::uint8_t* data,
                                     std::size_t data_bytes,
                                     const HeaderExpectation& expectation,
                                     ObjectHeaderFields* fields) noexcept;

// -------------------------------------------------------------------------
// peek_object_header
//
// Extracts fields with structural validation only (steps 1-5 above), without
// comparing against an expected key. Used by the recovery path that rebuilds
// an index by scanning media when the checkpoint is unusable: there, the
// header is the only source of identity, so there is nothing to compare to
// yet.
//
// Callers must still resolve identity to a real key before trusting the
// object. This function cannot detect stale reuse on its own.
// -------------------------------------------------------------------------
HeaderRejection peek_object_header(const std::uint8_t* data,
                                   std::size_t data_bytes,
                                   ObjectHeaderFields* fields) noexcept;

} // namespace tutti::storage_objects

// csrc/storage_objects/object_header_codec.cpp

#include "csrc/storage_objects/object_header_codec.h"

#include <cstring>

namespace tutti::storage_objects {
namespace {

// Little-endian field access. Explicit byte assembly rather than memcpy of a
// native integer: the on-media format must not depend on host endianness,
// since a pool could in principle be inspected from a different machine.
void store_u32(std::uint8_t* p, std::uint32_t v) noexcept {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
    p[2] = static_cast<std::uint8_t>(v >> 16);
    p[3] = static_cast<std::uint8_t>(v >> 24);
}

void store_u64(std::uint8_t* p, std::uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i) {
        p[i] = static_cast<std::uint8_t>(v >> (8 * i));
    }
}

std::uint32_t load_u32(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t load_u64(const std::uint8_t* p) noexcept {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

bool magic_is_zero(const std::uint8_t* p) noexcept {
    for (std::size_t i = 0; i < sizeof(ObjectHeaderLayout::kMagic); ++i) {
        if (p[i] != 0) return false;
    }
    return true;
}

bool magic_matches(const std::uint8_t* p) noexcept {
    return std::memcmp(p, ObjectHeaderLayout::kMagic,
                       sizeof(ObjectHeaderLayout::kMagic)) == 0;
}

// Structural validation shared by decode and peek: steps 1-5 of the documented
// order. Populates `fields` on success.
HeaderRejection validate_structure(const std::uint8_t* data,
                                   std::size_t data_bytes,
                                   ObjectHeaderFields* fields) noexcept {
    if (data == nullptr || data_bytes < ObjectHeaderLayout::kHeaderBytes) {
        return HeaderRejection::kTruncatedBuffer;
    }

    const std::uint8_t* magic = data + ObjectHeaderLayout::kMagicOffset;

    // Never-written space is all zeros. This is the expected state of free
    // slots, so it must be distinguishable from corruption -- reporting it as
    // kBadMagic would turn routine cold starts into apparent anomalies.
    if (magic_is_zero(magic)) return HeaderRejection::kEmptySlot;
    if (!magic_matches(magic)) return HeaderRejection::kBadMagic;

    const std::uint32_t version = load_u32(data + ObjectHeaderLayout::kVersionOffset);
    if (version != ObjectHeaderLayout::kVersion) {
        return HeaderRejection::kUnsupportedVersion;
    }

    // CRC before any content comparison: a torn write yields arbitrary field
    // values, and blaming identity for that would misdirect diagnosis.
    const std::uint32_t stored_crc =
        load_u32(data + ObjectHeaderLayout::kHeaderCrc32Offset);
    const std::uint32_t actual_crc =
        crc32c(data, ObjectHeaderLayout::kCrcCoveredBytes);
    if (stored_crc != actual_crc) return HeaderRejection::kBadHeaderCrc;

    if (fields != nullptr) {
        fields->identity = load_u64(data + ObjectHeaderLayout::kIdentityOffset);
        fields->payload_bytes =
            load_u64(data + ObjectHeaderLayout::kPayloadBytesOffset);
        fields->generation = load_u64(data + ObjectHeaderLayout::kGenerationOffset);
        fields->commit_seq = load_u64(data + ObjectHeaderLayout::kCommitSeqOffset);
        fields->key_len = load_u32(data + ObjectHeaderLayout::kKeyLenOffset);
        fields->key_crc32 = load_u32(data + ObjectHeaderLayout::kKeyCrc32Offset);
    }
    return HeaderRejection::kNone;
}

} // namespace

bool encode_object_header(std::uint8_t* out, std::size_t out_bytes,
                          const ObjectKey& key, std::uint64_t payload_bytes,
                          std::uint64_t generation,
                          std::uint64_t commit_seq) noexcept {
    if (out == nullptr || out_bytes < ObjectHeaderLayout::kHeaderBytes) {
        return false;
    }
    if (key.bytes.empty() || payload_bytes == 0) return false;

    // Zero the whole header first: padding must be deterministic so identical
    // inputs produce identical media, making torn writes detectable.
    std::memset(out, 0, ObjectHeaderLayout::kHeaderBytes);

    std::memcpy(out + ObjectHeaderLayout::kMagicOffset,
                ObjectHeaderLayout::kMagic,
                sizeof(ObjectHeaderLayout::kMagic));
    store_u32(out + ObjectHeaderLayout::kVersionOffset,
              ObjectHeaderLayout::kVersion);
    store_u64(out + ObjectHeaderLayout::kIdentityOffset, key.identity());
    store_u64(out + ObjectHeaderLayout::kPayloadBytesOffset, payload_bytes);
    store_u64(out + ObjectHeaderLayout::kGenerationOffset, generation);
    store_u32(out + ObjectHeaderLayout::kKeyLenOffset,
              static_cast<std::uint32_t>(key.bytes.size()));
    store_u32(out + ObjectHeaderLayout::kKeyCrc32Offset, key.key_crc32());
    store_u64(out + ObjectHeaderLayout::kCommitSeqOffset, commit_seq);

    // CRC last: it covers everything before its own field.
    store_u32(out + ObjectHeaderLayout::kHeaderCrc32Offset,
              crc32c(out, ObjectHeaderLayout::kCrcCoveredBytes));
    return true;
}

HeaderRejection decode_object_header(const std::uint8_t* data,
                                     std::size_t data_bytes,
                                     const HeaderExpectation& expectation,
                                     ObjectHeaderFields* fields) noexcept {
    if (expectation.key == nullptr || expectation.key->bytes.empty() ||
        expectation.payload_bytes == 0) {
        return HeaderRejection::kTruncatedBuffer;
    }

    ObjectHeaderFields decoded;
    const HeaderRejection structural =
        validate_structure(data, data_bytes, &decoded);
    if (structural != HeaderRejection::kNone) return structural;

    // The 64-bit identity is the fast path; the full key CRC and length back
    // it up so a digest collision cannot admit the wrong object.
    if (decoded.identity != expectation.key->identity()) {
        return HeaderRejection::kIdentityMismatch;
    }
    if (decoded.key_len != expectation.key->bytes.size() ||
        decoded.key_crc32 != expectation.key->key_crc32()) {
        return HeaderRejection::kKeyCrcMismatch;
    }
    if (decoded.payload_bytes != expectation.payload_bytes) {
        return HeaderRejection::kPayloadBytesMismatch;
    }
    // Generation catches ABA: the space was recycled and rewritten since the
    // record being validated was made.
    if (expectation.check_generation &&
        decoded.generation != expectation.generation) {
        return HeaderRejection::kGenerationMismatch;
    }

    if (fields != nullptr) *fields = decoded;
    return HeaderRejection::kNone;
}

HeaderRejection peek_object_header(const std::uint8_t* data,
                                   std::size_t data_bytes,
                                   ObjectHeaderFields* fields) noexcept {
    return validate_structure(data, data_bytes, fields);
}

} // namespace tutti::storage_objects

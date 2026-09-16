// csrc/storage_objects/checkpoint_region.cpp

#include "csrc/storage_objects/checkpoint_region.h"

#include <cstring>

namespace tutti::storage_objects {
namespace {

constexpr std::uint64_t kAlignment = 4096;

// Fixed per-entry cost in the body, excluding the key bytes themselves:
// u32 key_len + u64 slot + u64 payload_bytes + u64 generation + u64 commit_seq.
constexpr std::uint64_t kEntryFixedBytes = 4 + 8 + 8 + 8 + 8;
constexpr std::uint64_t kEntryCountBytes = 8;

void append_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>(v));
    out.push_back(static_cast<std::uint8_t>(v >> 8));
    out.push_back(static_cast<std::uint8_t>(v >> 16));
    out.push_back(static_cast<std::uint8_t>(v >> 24));
}

void append_u64(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    }
}

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

std::uint64_t round_up(std::uint64_t value, std::uint64_t multiple) noexcept {
    if (multiple == 0) return value;
    return ((value + multiple - 1) / multiple) * multiple;
}

} // namespace

const char* to_string(CheckpointRejection rejection) noexcept {
    switch (rejection) {
        case CheckpointRejection::kNone: return "none";
        case CheckpointRejection::kEmptyContainer: return "empty_container";
        case CheckpointRejection::kBadMagic: return "bad_magic";
        case CheckpointRejection::kUnsupportedVersion: return "unsupported_version";
        case CheckpointRejection::kTruncatedBuffer: return "truncated_buffer";
        case CheckpointRejection::kBadBodyCrc: return "bad_body_crc";
        case CheckpointRejection::kMalformedBody: return "malformed_body";
    }
    return "unknown";
}

std::vector<std::uint8_t> encode_checkpoint(
    const std::vector<CheckpointEntry>& entries, std::uint64_t sequence) {
    std::vector<std::uint8_t> body;
    std::uint64_t reserve = kEntryCountBytes;
    for (const auto& e : entries) {
        reserve += kEntryFixedBytes + e.key.bytes.size();
    }
    body.reserve(static_cast<std::size_t>(reserve));

    append_u64(body, static_cast<std::uint64_t>(entries.size()));
    for (const auto& e : entries) {
        append_u32(body, static_cast<std::uint32_t>(e.key.bytes.size()));
        append_u64(body, e.slot);
        append_u64(body, e.payload_bytes);
        append_u64(body, e.generation);
        append_u64(body, e.commit_seq);
        body.insert(body.end(), e.key.bytes.begin(), e.key.bytes.end());
    }

    const std::uint64_t total = round_up(
        CheckpointLayout::kContainerHeaderBytes + body.size(), kAlignment);

    std::vector<std::uint8_t> image(static_cast<std::size_t>(total), 0);
    std::memcpy(image.data() + CheckpointLayout::kMagicOffset,
                CheckpointLayout::kMagic, sizeof(CheckpointLayout::kMagic));
    store_u32(image.data() + CheckpointLayout::kVersionOffset,
              CheckpointLayout::kVersion);
    store_u64(image.data() + CheckpointLayout::kSequenceOffset, sequence);
    store_u64(image.data() + CheckpointLayout::kBodyBytesOffset,
              static_cast<std::uint64_t>(body.size()));
    store_u32(image.data() + CheckpointLayout::kBodyCrc32Offset,
              crc32c(body.data(), body.size()));

    if (!body.empty()) {
        std::memcpy(image.data() + CheckpointLayout::kContainerHeaderBytes,
                    body.data(), body.size());
    }
    return image;
}

CheckpointRejection decode_checkpoint(const std::uint8_t* data,
                                      std::size_t data_bytes,
                                      std::uint64_t* sequence,
                                      std::vector<CheckpointEntry>* entries) {
    if (data == nullptr || data_bytes < CheckpointLayout::kContainerHeaderBytes) {
        return CheckpointRejection::kTruncatedBuffer;
    }

    bool all_zero = true;
    for (std::size_t i = 0; i < sizeof(CheckpointLayout::kMagic); ++i) {
        if (data[CheckpointLayout::kMagicOffset + i] != 0) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) return CheckpointRejection::kEmptyContainer;

    if (std::memcmp(data + CheckpointLayout::kMagicOffset,
                    CheckpointLayout::kMagic,
                    sizeof(CheckpointLayout::kMagic)) != 0) {
        return CheckpointRejection::kBadMagic;
    }
    if (load_u32(data + CheckpointLayout::kVersionOffset) !=
        CheckpointLayout::kVersion) {
        return CheckpointRejection::kUnsupportedVersion;
    }

    const std::uint64_t seq = load_u64(data + CheckpointLayout::kSequenceOffset);
    const std::uint64_t body_bytes =
        load_u64(data + CheckpointLayout::kBodyBytesOffset);
    const std::uint32_t stored_crc =
        load_u32(data + CheckpointLayout::kBodyCrc32Offset);

    // A corrupted length field must not be able to read past the buffer.
    if (body_bytes > data_bytes - CheckpointLayout::kContainerHeaderBytes) {
        return CheckpointRejection::kTruncatedBuffer;
    }

    const std::uint8_t* body = data + CheckpointLayout::kContainerHeaderBytes;
    if (crc32c(body, static_cast<std::size_t>(body_bytes)) != stored_crc) {
        return CheckpointRejection::kBadBodyCrc;
    }

    // Body parse. Every read is bounds-checked against body_bytes even though
    // the CRC already passed: a valid CRC over a body whose internal lengths
    // are inconsistent is possible if the body was assembled incorrectly, and
    // that must be a clean rejection rather than a read overrun.
    if (body_bytes < kEntryCountBytes) return CheckpointRejection::kMalformedBody;

    std::uint64_t cursor = 0;
    const std::uint64_t count = load_u64(body + cursor);
    cursor += kEntryCountBytes;

    // Reject an entry count that cannot possibly fit, before allocating for it.
    if (count > (body_bytes - cursor) / kEntryFixedBytes) {
        return CheckpointRejection::kMalformedBody;
    }

    std::vector<CheckpointEntry> parsed;
    parsed.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        if (body_bytes - cursor < kEntryFixedBytes) {
            return CheckpointRejection::kMalformedBody;
        }
        CheckpointEntry entry;
        const std::uint32_t key_len = load_u32(body + cursor);
        cursor += 4;
        entry.slot = load_u64(body + cursor);
        cursor += 8;
        entry.payload_bytes = load_u64(body + cursor);
        cursor += 8;
        entry.generation = load_u64(body + cursor);
        cursor += 8;
        entry.commit_seq = load_u64(body + cursor);
        cursor += 8;

        if (key_len == 0 || key_len > body_bytes - cursor) {
            return CheckpointRejection::kMalformedBody;
        }
        entry.key.bytes.assign(body + cursor, body + cursor + key_len);
        cursor += key_len;
        parsed.push_back(std::move(entry));
    }

    if (sequence != nullptr) *sequence = seq;
    if (entries != nullptr) *entries = std::move(parsed);
    return CheckpointRejection::kNone;
}

std::uint64_t checkpoint_container_bytes(std::uint64_t max_entries,
                                        std::uint32_t max_key_bytes) noexcept {
    const std::uint64_t body =
        kEntryCountBytes +
        max_entries * (kEntryFixedBytes + static_cast<std::uint64_t>(max_key_bytes));
    return round_up(CheckpointLayout::kContainerHeaderBytes + body, kAlignment);
}

std::uint64_t checkpoint_container_offset(
    std::uint32_t index, std::uint64_t container_bytes) noexcept {
    return static_cast<std::uint64_t>(index) * container_bytes;
}

std::uint32_t next_checkpoint_container(const std::uint64_t* sequences,
                                        std::uint32_t count) noexcept {
    if (sequences == nullptr || count == 0) return 0;

    // Prefer an empty container: nothing valid is at risk there.
    for (std::uint32_t i = 0; i < count; ++i) {
        if (sequences[i] == 0) return i;
    }
    // Otherwise overwrite the OLDEST, so the newest valid state survives a
    // crash during this write. This is the entire basis of the scheme's
    // atomicity.
    std::uint32_t oldest = 0;
    for (std::uint32_t i = 1; i < count; ++i) {
        if (sequences[i] < sequences[oldest]) oldest = i;
    }
    return oldest;
}

bool select_checkpoint_container(const std::uint64_t* sequences,
                                std::uint32_t count,
                                std::uint32_t* chosen) noexcept {
    if (sequences == nullptr || count == 0) return false;
    bool found = false;
    std::uint32_t best = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
        if (sequences[i] == 0) continue;  // empty or unreadable
        if (!found || sequences[i] > sequences[best]) {
            best = i;
            found = true;
        }
    }
    if (found && chosen != nullptr) *chosen = best;
    return found;
}

} // namespace tutti::storage_objects

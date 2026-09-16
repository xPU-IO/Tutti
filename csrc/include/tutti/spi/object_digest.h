#pragma once

// tutti/spi/object_digest.h -- On-media digest primitives for the storage
// object layer.
//
// These two functions define persistent on-disk format. Once any deployment
// has written objects, changing either algorithm invalidates every stored
// object: recovery would compute a different identity/CRC than the header
// records and discard the data as corrupt. Treat them as frozen; a genuine
// algorithm change requires bumping ObjectHeaderLayout::kVersion and
// supporting both during a migration window.
//
// They live in the SPI (not in an implementation directory) because both the
// store implementations and their contract tests must agree on them bit for
// bit, and because an alternative backend implementing StorageObjectStore
// needs the same digests to write compatible headers.
//
// Header-inline on purpose: ObjectKey::identity() is a member of a struct
// every SPI consumer handles, so requiring a link dependency to hash a key
// would push an implementation-level obligation onto every consumer. The
// lookup table is constexpr-generated, so there is no runtime initialisation
// and no per-translation-unit cost beyond the linker folding one array.
//
// Neither function is a security primitive. They detect accidental
// corruption, stale reuse and misidentification -- not tampering.

#include <cstddef>
#include <cstdint>

namespace tutti {

// -------------------------------------------------------------------------
// CRC-32C (Castagnoli): polynomial 0x1EDC6F41, reflected form 0x82F63B78.
//
// Chosen over the zlib/Ethernet polynomial because this is the variant with
// hardware support (SSE4.2 _mm_crc32_*, ARMv8 CRC32C), leaving room to
// accelerate later without changing the on-media format.
//
// Convention: init 0xFFFFFFFF, final xor 0xFFFFFFFF, reflected in/out -- the
// standard CRC-32C of iSCSI/SCTP/ext4 metadata, so an independent
// implementation will agree bit for bit.
//
// Scope reminder: only ever applied to METADATA (object headers, key bytes,
// checkpoint bodies). Payload is never checksummed -- doing so would pull KV
// data through the host CPU and defeat the GPU-direct premise.
// -------------------------------------------------------------------------
namespace detail {

// Reflected CRC-32C polynomial.
constexpr std::uint32_t kCrc32cPolynomial = 0x82F63B78u;

struct Crc32cTable {
    std::uint32_t entry[256];
};

constexpr Crc32cTable make_crc32c_table() noexcept {
    Crc32cTable table{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t crc = i;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 1u) ? ((crc >> 1) ^ kCrc32cPolynomial) : (crc >> 1);
        }
        table.entry[i] = crc;
    }
    return table;
}

// constexpr, so the table is materialised at compile time: no runtime
// initialisation, no static-init order concerns.
inline constexpr Crc32cTable kCrc32cTable = make_crc32c_table();

} // namespace detail

constexpr std::uint32_t crc32c_init() noexcept { return 0xFFFFFFFFu; }

constexpr std::uint32_t crc32c_final(std::uint32_t state) noexcept {
    return state ^ 0xFFFFFFFFu;
}

// Incremental update. Feeding a buffer in any split yields the same state as
// feeding it whole:
//   crc32c_update(crc32c_update(s, a, na), b, nb)
//     == crc32c_update(s, ab, na + nb)
inline std::uint32_t crc32c_update(std::uint32_t state, const void* data,
                                   std::size_t bytes) noexcept {
    const auto* p = static_cast<const std::uint8_t*>(data);
    if (p == nullptr) return state;
    for (std::size_t i = 0; i < bytes; ++i) {
        state = detail::kCrc32cTable.entry[(state ^ p[i]) & 0xFFu] ^ (state >> 8);
    }
    return state;
}

// One-shot form.
//
// crc32c(nullptr, 0) == 0: a zero-length buffer yields 0 rather than the bare
// final-xor value, so "no data" stays distinguishable from "data whose CRC
// happens to be 0xFFFFFFFF".
inline std::uint32_t crc32c(const void* data, std::size_t bytes) noexcept {
    if (data == nullptr || bytes == 0) return 0;
    return crc32c_final(crc32c_update(crc32c_init(), data, bytes));
}

// -------------------------------------------------------------------------
// object_identity -- stable 64-bit digest of an ObjectKey's bytes, written
// into the object header so a slot can prove what it holds.
//
// FNV-1a 64 followed by a SplitMix64 finalisation mix. FNV-1a alone has poor
// high-bit diffusion for short keys, which matters if callers bucket on the
// digest; the mix gives full avalanche at negligible cost.
//
// Determinism, all satisfied by construction:
//   - byte-oriented, so host endianness does not enter;
//   - no seed and no randomisation. std::hash is explicitly permitted to vary
//     between processes and therefore MUST NOT be used for anything
//     persisted;
//   - identical across compilers and optimisation levels.
//
// Collisions are tolerable, not catastrophic. A 64-bit collision between two
// distinct keys cannot yield wrong data: recovery additionally compares the
// full key's CRC-32C and the recorded payload length, so a
// colliding-but-different key is rejected. The digest exists to make the
// common case a single 8-byte compare.
//
// object_identity(nullptr, 0) is defined (the mixed FNV offset basis), so an
// empty key is legal if useless.
// -------------------------------------------------------------------------
inline std::uint64_t object_identity(const void* data,
                                     std::size_t bytes) noexcept {
    constexpr std::uint64_t kFnvOffsetBasis = 0xCBF29CE484222325ull;
    constexpr std::uint64_t kFnvPrime = 0x00000100000001B3ull;

    std::uint64_t hash = kFnvOffsetBasis;
    const auto* p = static_cast<const std::uint8_t*>(data);
    if (p != nullptr) {
        for (std::size_t i = 0; i < bytes; ++i) {
            hash ^= static_cast<std::uint64_t>(p[i]);
            hash *= kFnvPrime;
        }
    }

    // SplitMix64 finaliser: avalanche the low-diffusion FNV output.
    hash ^= hash >> 30;
    hash *= 0xBF58476D1CE4E5B9ull;
    hash ^= hash >> 27;
    hash *= 0x94D049BB133111EBull;
    hash ^= hash >> 31;
    return hash;
}

} // namespace tutti

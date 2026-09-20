// tests/storage_objects_contract/storage_objects_contract_test.cpp
//
// Contract test for the four storage-object-layer units. Hardware-free except
// for the residency bitmap, which needs a real file to mmap and uses a
// temporary directory.
//
// These units carry the layer's crash-safety guarantees, so the tests are
// adversarial: they corrupt headers, tear checkpoint writes, desynchronise
// bitmaps and recycle space underneath live records. The invariant under test
// throughout is the direction of failure:
//
//   losing a valid object    -> recomputation, acceptable
//   accepting an invalid one -> silent corruption, forbidden
//
// Every check below exists to pin one of those two outcomes.

#include "csrc/storage_objects/checkpoint_region.h"
#include "csrc/storage_objects/object_header_codec.h"
#include "csrc/storage_objects/residency_bitmap.h"
#include "csrc/storage_objects/space_allocator.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using namespace tutti;
using namespace tutti::storage_objects;

int g_failures = 0;

void check(bool cond, const char* expr, int line) {
    if (!cond) {
        std::printf("FAIL [line %d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

// For preconditions that later code INDEXES ON. A plain CHECK keeps going and
// the following subscript reads out of range, turning a clear assertion failure
// into a segfault -- which is exactly the wrong trade, since the whole value of
// a contract test is naming the broken invariant. Blocks using REQUIRE are
// written as immediately-invoked lambdas so this can return from just that
// block, leaving the remaining sections to run.
#define REQUIRE(cond)                          \
    do {                                       \
        if (!(cond)) {                         \
            check(false, #cond, __LINE__);     \
            return;                            \
        }                                      \
    } while (0)

ObjectKey key_of(std::uint8_t tag, std::size_t len = 18) {
    ObjectKey k;
    k.bytes.assign(len, tag);
    return k;
}

constexpr std::uint64_t kPayload = 10ull * 1024 * 1024;  // 80 layers x 128 KiB

// Where scratch files go, in precedence order:
//   1. TMPDIR, so a caller can steer the test explicitly;
//   2. the build tree (compiled in), which is on whatever filesystem the user
//      chose to build on;
//   3. /tmp as a last resort.
//
// Not merely convention: a Tutti host typically has a small root filesystem and
// large data mounts, so a test that hardcodes /tmp can fill the root.
#ifndef TUTTI_TEST_TMPDIR_DEFAULT
#  define TUTTI_TEST_TMPDIR_DEFAULT "/tmp"
#endif

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = (base != nullptr && *base != '\0')
                           ? base
                           : TUTTI_TEST_TMPDIR_DEFAULT;
    tmpl += "/tutti_sobj_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char* dir = ::mkdtemp(buf.data());
    return dir == nullptr ? std::string() : std::string(dir);
}

// ======================================================================
// 1. Object header codec
// ======================================================================
void test_object_header() {
    std::vector<std::uint8_t> buf(ObjectHeaderLayout::kHeaderBytes);
    const ObjectKey key = key_of(0xA1);

    // --- round trip ---
    CHECK(encode_object_header(buf.data(), buf.size(), key, kPayload, 7, 42));
    HeaderExpectation expect;
    expect.key = &key;
    expect.payload_bytes = kPayload;
    expect.generation = 7;
    ObjectHeaderFields fields;
    CHECK(decode_object_header(buf.data(), buf.size(), expect, &fields) ==
          HeaderRejection::kNone);
    CHECK(fields.identity == key.identity());
    CHECK(fields.payload_bytes == kPayload);
    CHECK(fields.generation == 7);
    CHECK(fields.commit_seq == 42);
    CHECK(fields.key_len == key.bytes.size());
    CHECK(fields.key_crc32 == key.key_crc32());

    // --- encoding is deterministic ---
    // Identical inputs must produce identical media, otherwise a torn write
    // cannot be distinguished from a legitimately different encoding.
    std::vector<std::uint8_t> again(ObjectHeaderLayout::kHeaderBytes);
    CHECK(encode_object_header(again.data(), again.size(), key, kPayload, 7, 42));
    CHECK(std::memcmp(buf.data(), again.data(), buf.size()) == 0);

    // --- padding is zeroed ---
    // Leftover bytes must not leak whatever was in the buffer before.
    bool padding_clean = true;
    for (std::size_t i = ObjectHeaderLayout::kHeaderCrc32Offset + 4;
         i < ObjectHeaderLayout::kHeaderBytes; ++i) {
        if (buf[i] != 0) padding_clean = false;
    }
    CHECK(padding_clean);

    // --- never-written space is kEmptySlot, not corruption ---
    // Free slots are all zeros; misreporting that as kBadMagic would make
    // routine cold starts look like anomalies.
    std::vector<std::uint8_t> zeros(ObjectHeaderLayout::kHeaderBytes, 0);
    CHECK(decode_object_header(zeros.data(), zeros.size(), expect, nullptr) ==
          HeaderRejection::kEmptySlot);

    // --- foreign data is distinguishable from empty ---
    std::vector<std::uint8_t> foreign = zeros;
    std::memcpy(foreign.data(), "NOTTUTTI", 8);
    CHECK(decode_object_header(foreign.data(), foreign.size(), expect, nullptr) ==
          HeaderRejection::kBadMagic);

    // --- torn write is reported as a CRC failure, not as identity mismatch ---
    // A corrupted header yields arbitrary field values; blaming identity would
    // misdirect diagnosis toward key derivation.
    std::vector<std::uint8_t> torn = buf;
    torn[ObjectHeaderLayout::kIdentityOffset] ^= 0xFF;
    CHECK(decode_object_header(torn.data(), torn.size(), expect, nullptr) ==
          HeaderRejection::kBadHeaderCrc);

    // --- every byte the CRC covers is actually protected ---
    int unprotected = 0;
    for (std::size_t i = 0; i < ObjectHeaderLayout::kCrcCoveredBytes; ++i) {
        std::vector<std::uint8_t> mutated = buf;
        mutated[i] ^= 0x01;
        if (decode_object_header(mutated.data(), mutated.size(), expect,
                                 nullptr) == HeaderRejection::kNone) {
            ++unprotected;
        }
    }
    CHECK(unprotected == 0);

    // --- wrong key is rejected even with an intact header ---
    // This is stale space reuse: the slot holds a different object.
    const ObjectKey other = key_of(0xB2);
    HeaderExpectation wrong = expect;
    wrong.key = &other;
    CHECK(decode_object_header(buf.data(), buf.size(), wrong, nullptr) ==
          HeaderRejection::kIdentityMismatch);

    // --- a same-identity, different-key header is still rejected ---
    // Simulates a 64-bit digest collision: the full key CRC must catch it.
    // Constructed by re-encoding with a key of the same length but different
    // content, then forcing the identity field back to the expected value.
    {
        std::vector<std::uint8_t> collide(ObjectHeaderLayout::kHeaderBytes);
        const ObjectKey imposter = key_of(0xC3);
        CHECK(encode_object_header(collide.data(), collide.size(), imposter,
                                   kPayload, 7, 42));
        // Force identity to the expected key's, then repair the header CRC so
        // only the key CRC disagrees.
        const std::uint64_t want = key.identity();
        for (int i = 0; i < 8; ++i) {
            collide[ObjectHeaderLayout::kIdentityOffset + i] =
                static_cast<std::uint8_t>(want >> (8 * i));
        }
        const std::uint32_t fixed =
            crc32c(collide.data(), ObjectHeaderLayout::kCrcCoveredBytes);
        for (int i = 0; i < 4; ++i) {
            collide[ObjectHeaderLayout::kHeaderCrc32Offset + i] =
                static_cast<std::uint8_t>(fixed >> (8 * i));
        }
        CHECK(decode_object_header(collide.data(), collide.size(), expect,
                                   nullptr) == HeaderRejection::kKeyCrcMismatch);
    }

    // --- payload length disagreement is rejected ---
    HeaderExpectation short_payload = expect;
    short_payload.payload_bytes = kPayload / 2;
    CHECK(decode_object_header(buf.data(), buf.size(), short_payload, nullptr) ==
          HeaderRejection::kPayloadBytesMismatch);

    // --- ABA: recycled space is detected via generation ---
    HeaderExpectation stale = expect;
    stale.generation = 6;  // record predates the recycle
    CHECK(decode_object_header(buf.data(), buf.size(), stale, nullptr) ==
          HeaderRejection::kGenerationMismatch);
    // Header-scan recovery has no generation to compare against.
    stale.check_generation = false;
    CHECK(decode_object_header(buf.data(), buf.size(), stale, nullptr) ==
          HeaderRejection::kNone);

    // --- truncated buffers never read out of range ---
    CHECK(decode_object_header(buf.data(), 128, expect, nullptr) ==
          HeaderRejection::kTruncatedBuffer);
    CHECK(decode_object_header(nullptr, 0, expect, nullptr) ==
          HeaderRejection::kTruncatedBuffer);
    CHECK(!encode_object_header(buf.data(), 128, key, kPayload, 0, 0));

    // --- unusable inputs are refused rather than encoded ---
    const ObjectKey empty_key;
    CHECK(!encode_object_header(buf.data(), buf.size(), empty_key, kPayload, 0, 0));
    CHECK(!encode_object_header(buf.data(), buf.size(), key, 0, 0, 0));

    // --- peek validates structure without needing a key ---
    ObjectHeaderFields peeked;
    CHECK(peek_object_header(buf.data(), buf.size(), &peeked) ==
          HeaderRejection::kNone);
    CHECK(peeked.identity == key.identity());
    CHECK(peek_object_header(zeros.data(), zeros.size(), &peeked) ==
          HeaderRejection::kEmptySlot);
    CHECK(peek_object_header(torn.data(), torn.size(), &peeked) ==
          HeaderRejection::kBadHeaderCrc);
}

// ======================================================================
// 2. Checkpoint region
// ======================================================================
void test_checkpoint() {
    std::vector<CheckpointEntry> entries;
    for (std::uint8_t i = 1; i <= 5; ++i) {
        CheckpointEntry e;
        e.key = key_of(i);
        e.slot = i * 10;
        e.payload_bytes = kPayload;
        e.generation = i;
        e.commit_seq = 100 + i;
        entries.push_back(e);
    }

    const std::vector<std::uint8_t> image = encode_checkpoint(entries, 9);
    CHECK(image.size() % 4096 == 0);   // O_DIRECT writable
    CHECK(image.size() > CheckpointLayout::kContainerHeaderBytes);

    std::uint64_t seq = 0;
    std::vector<CheckpointEntry> decoded;
    CHECK(decode_checkpoint(image.data(), image.size(), &seq, &decoded) ==
          CheckpointRejection::kNone);
    CHECK(seq == 9);
    CHECK(decoded.size() == entries.size());
    bool all_match = true;
    for (std::size_t i = 0; i < decoded.size(); ++i) {
        if (decoded[i].key != entries[i].key ||
            decoded[i].slot != entries[i].slot ||
            decoded[i].payload_bytes != entries[i].payload_bytes ||
            decoded[i].generation != entries[i].generation ||
            decoded[i].commit_seq != entries[i].commit_seq) {
            all_match = false;
        }
    }
    CHECK(all_match);

    // --- empty checkpoint is legal (a store with nothing committed) ---
    const std::vector<std::uint8_t> empty_image = encode_checkpoint({}, 1);
    std::vector<CheckpointEntry> empty_decoded;
    CHECK(decode_checkpoint(empty_image.data(), empty_image.size(), &seq,
                            &empty_decoded) == CheckpointRejection::kNone);
    CHECK(seq == 1);
    CHECK(empty_decoded.empty());

    // --- never-written container is distinguishable from corruption ---
    std::vector<std::uint8_t> zeros(CheckpointLayout::kContainerHeaderBytes, 0);
    CHECK(decode_checkpoint(zeros.data(), zeros.size(), &seq, &decoded) ==
          CheckpointRejection::kEmptyContainer);

    // --- torn body write is caught by the CRC ---
    std::vector<std::uint8_t> torn = image;
    torn[CheckpointLayout::kContainerHeaderBytes + 16] ^= 0xFF;
    CHECK(decode_checkpoint(torn.data(), torn.size(), &seq, &decoded) ==
          CheckpointRejection::kBadBodyCrc);

    // --- a corrupted length field cannot walk the parser off the buffer ---
    std::vector<std::uint8_t> overlong = image;
    for (int i = 0; i < 8; ++i) {
        overlong[CheckpointLayout::kBodyBytesOffset + i] = 0xFF;
    }
    CHECK(decode_checkpoint(overlong.data(), overlong.size(), &seq, &decoded) ==
          CheckpointRejection::kTruncatedBuffer);

    // --- an absurd entry count is rejected before allocating for it ---
    {
        std::vector<std::uint8_t> huge = image;
        std::uint8_t* body = huge.data() + CheckpointLayout::kContainerHeaderBytes;
        for (int i = 0; i < 8; ++i) body[i] = 0xFF;   // entry_count = 2^64-1
        // Repair the body CRC so the count check is what rejects it, not the CRC.
        const std::uint64_t body_bytes = image.size() -
                                         CheckpointLayout::kContainerHeaderBytes;
        std::uint64_t declared = 0;
        for (int i = 0; i < 8; ++i) {
            declared |= static_cast<std::uint64_t>(
                            image[CheckpointLayout::kBodyBytesOffset + i]) << (8 * i);
        }
        (void)body_bytes;
        const std::uint32_t crc = crc32c(body, static_cast<std::size_t>(declared));
        for (int i = 0; i < 4; ++i) {
            huge[CheckpointLayout::kBodyCrc32Offset + i] =
                static_cast<std::uint8_t>(crc >> (8 * i));
        }
        CHECK(decode_checkpoint(huge.data(), huge.size(), &seq, &decoded) ==
              CheckpointRejection::kMalformedBody);
    }

    // --- bad magic / version ---
    std::vector<std::uint8_t> foreign = image;
    std::memcpy(foreign.data(), "OTHERIDX", 8);
    CHECK(decode_checkpoint(foreign.data(), foreign.size(), &seq, &decoded) ==
          CheckpointRejection::kBadMagic);

    std::vector<std::uint8_t> future = image;
    future[CheckpointLayout::kVersionOffset] = 99;
    CHECK(decode_checkpoint(future.data(), future.size(), &seq, &decoded) ==
          CheckpointRejection::kUnsupportedVersion);

    CHECK(decode_checkpoint(nullptr, 0, &seq, &decoded) ==
          CheckpointRejection::kTruncatedBuffer);

    // --- variable-length keys survive: assuming a fixed width would truncate ---
    {
        std::vector<CheckpointEntry> mixed;
        for (std::size_t len : {1u, 16u, 18u, 64u, 255u}) {
            CheckpointEntry e;
            e.key = key_of(static_cast<std::uint8_t>(len), len);
            e.slot = len;
            e.payload_bytes = kPayload;
            mixed.push_back(e);
        }
        const std::vector<std::uint8_t> img = encode_checkpoint(mixed, 3);
        std::vector<CheckpointEntry> out;
        CHECK(decode_checkpoint(img.data(), img.size(), &seq, &out) ==
              CheckpointRejection::kNone);
        CHECK(out.size() == mixed.size());
        bool lengths_ok = true;
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (out[i].key.bytes.size() != mixed[i].key.bytes.size() ||
                out[i].key != mixed[i].key) {
                lengths_ok = false;
            }
        }
        CHECK(lengths_ok);
    }

    // --- container sizing and placement ---
    const std::uint64_t container = checkpoint_container_bytes(1000, 18);
    CHECK(container % 4096 == 0);
    CHECK(container >= encode_checkpoint(entries, 1).size());
    CHECK(checkpoint_container_offset(0, container) == 0);
    CHECK(checkpoint_container_offset(1, container) == container);
    // Containers must not overlap, or a write would clobber its neighbour.
    CHECK(checkpoint_container_offset(1, container) >= container);

    // --- container rotation is what makes the scheme crash-atomic ---
    {
        // All empty: start at 0.
        std::uint64_t seqs[2] = {0, 0};
        CHECK(next_checkpoint_container(seqs, 2) == 0);
        std::uint32_t chosen = 99;
        CHECK(!select_checkpoint_container(seqs, 2, &chosen));  // nothing valid

        // One written: read it, write the other.
        seqs[0] = 5;
        CHECK(select_checkpoint_container(seqs, 2, &chosen));
        CHECK(chosen == 0);
        CHECK(next_checkpoint_container(seqs, 2) == 1);

        // Both written: read the newest, overwrite the oldest. This is the
        // property that keeps the newest valid state intact across a crash.
        seqs[1] = 6;
        CHECK(select_checkpoint_container(seqs, 2, &chosen));
        CHECK(chosen == 1);
        CHECK(next_checkpoint_container(seqs, 2) == 0);

        seqs[0] = 7;
        CHECK(select_checkpoint_container(seqs, 2, &chosen));
        CHECK(chosen == 0);
        CHECK(next_checkpoint_container(seqs, 2) == 1);
    }

    // --- a torn newest container falls back to the older one ---
    // The crash scenario: container 1 was being written when power was lost.
    {
        const std::vector<std::uint8_t> older = encode_checkpoint(entries, 5);
        std::vector<std::uint8_t> newer = encode_checkpoint(entries, 6);
        newer[CheckpointLayout::kContainerHeaderBytes + 4] ^= 0xFF;  // torn

        std::uint64_t seqs[2] = {0, 0};
        std::uint64_t s = 0;
        std::vector<CheckpointEntry> tmp;
        if (decode_checkpoint(older.data(), older.size(), &s, &tmp) ==
            CheckpointRejection::kNone) {
            seqs[0] = s;
        }
        if (decode_checkpoint(newer.data(), newer.size(), &s, &tmp) ==
            CheckpointRejection::kNone) {
            seqs[1] = s;
        }
        // The torn container reads as sequence 0, so the older one is selected:
        // recovery loses the newest commits but never reads garbage.
        CHECK(seqs[1] == 0);
        std::uint32_t chosen = 99;
        CHECK(select_checkpoint_container(seqs, 2, &chosen));
        CHECK(chosen == 0);
    }
}

// ======================================================================
// 3. Residency bitmap
// ======================================================================
void test_residency(const std::string& dir) {
    constexpr std::uint64_t kSlots = 1000;
    constexpr std::uint64_t kFingerprint = 0xABCDEF0123456789ull;

    CHECK(residency_file_bytes(kSlots) % 4096 == 0);
    CHECK(residency_file_bytes(kSlots) >=
          ResidencyBitmapLayout::kHeaderBytes + (kSlots + 7) / 8);
    // Bit addressing must land past the header.
    CHECK(residency_bit_offset(0) == ResidencyBitmapLayout::kHeaderBytes);
    CHECK(residency_bit_offset(8) == ResidencyBitmapLayout::kHeaderBytes + 1);
    CHECK(residency_bit_shift(0) == 0);
    CHECK(residency_bit_shift(9) == 1);

    const std::string path0 = dir + "/r0.bitmap";
    const std::string path1 = dir + "/r1.bitmap";

    // --- create, set, read back ---
    {
        ResidencyBitmap bm;
        CHECK(bm.open_writable(path0, 0, 2, kSlots, kFingerprint).ok());
        CHECK(bm.usable());
        CHECK(bm.slot_count() == kSlots);
        CHECK(!bm.test(5));
        bm.set(5);
        bm.set(999);
        CHECK(bm.test(5));
        CHECK(bm.test(999));
        bm.clear(5);
        CHECK(!bm.test(5));
        CHECK(bm.test(999));
        CHECK(bm.sync().ok());
    }

    // --- content survives reopen ---
    {
        ResidencyBitmap bm;
        CHECK(bm.open_writable(path0, 0, 2, kSlots, kFingerprint).ok());
        CHECK(bm.test(999));
        bm.clear_all();
        CHECK(!bm.test(999));
    }

    // --- geometry / namespace mismatches are refused, not "repaired" ---
    // Reinitialising could destroy a live peer's information, so the only safe
    // response is to decline and let the caller degrade to single-rank.
    {
        ResidencyBitmap bm;
        CHECK(!bm.open_writable(path0, 0, 2, kSlots * 2, kFingerprint).ok());
        CHECK(!bm.usable());
        ResidencyBitmap bm2;
        CHECK(!bm2.open_writable(path0, 0, 2, kSlots, kFingerprint + 1).ok());
        CHECK(!bm2.usable());
        ResidencyBitmap bm3;
        CHECK(!bm3.open_writable(path0, 0, 4, kSlots, kFingerprint).ok());
        CHECK(!bm3.usable());
    }

    // --- header validation classifies each disagreement ---
    {
        std::vector<std::uint8_t> hdr(ResidencyBitmapLayout::kHeaderBytes);
        encode_residency_header(hdr.data(), 3, 8, kSlots, kFingerprint);
        CHECK(validate_residency_header(hdr.data(), hdr.size(), 8, kSlots,
                                        kFingerprint) ==
              ResidencyRejection::kNone);
        CHECK(validate_residency_header(hdr.data(), hdr.size(), 8, kSlots + 1,
                                        kFingerprint) ==
              ResidencyRejection::kSlotCountMismatch);
        CHECK(validate_residency_header(hdr.data(), hdr.size(), 8, kSlots,
                                        kFingerprint ^ 1) ==
              ResidencyRejection::kFingerprintMismatch);
        CHECK(validate_residency_header(hdr.data(), hdr.size(), 4, kSlots,
                                        kFingerprint) ==
              ResidencyRejection::kRankCountMismatch);
        CHECK(validate_residency_header(hdr.data(), 64, 8, kSlots,
                                        kFingerprint) ==
              ResidencyRejection::kTruncatedBuffer);

        std::vector<std::uint8_t> zeros(ResidencyBitmapLayout::kHeaderBytes, 0);
        CHECK(validate_residency_header(zeros.data(), zeros.size(), 8, kSlots,
                                        kFingerprint) ==
              ResidencyRejection::kEmptyFile);

        std::vector<std::uint8_t> foreign = hdr;
        std::memcpy(foreign.data(), "OTHERRES", 8);
        CHECK(validate_residency_header(foreign.data(), foreign.size(), 8, kSlots,
                                        kFingerprint) ==
              ResidencyRejection::kBadMagic);

        std::vector<std::uint8_t> future = hdr;
        future[ResidencyBitmapLayout::kVersionOffset] = 99;
        CHECK(validate_residency_header(future.data(), future.size(), 8, kSlots,
                                        kFingerprint) ==
              ResidencyRejection::kUnsupportedVersion);
    }

    // --- out-of-range access is inert, never corrupting neighbours ---
    {
        ResidencyBitmap bm;
        CHECK(bm.open_writable(path0, 0, 2, kSlots, kFingerprint).ok());
        bm.set(kSlots);          // ignored
        bm.set(kSlots + 10000);  // ignored
        CHECK(!bm.test(kSlots));
        CHECK(!bm.test(kSlots + 10000));
        bm.set(0);
        CHECK(bm.test(0));       // in-range bit unaffected
    }

    // --- cross-rank AND, and the forbidden direction ---
    {
        ResidencyBitmap r0, r1;
        CHECK(r0.open_writable(path0, 0, 2, kSlots, kFingerprint).ok());
        CHECK(r1.open_writable(path1, 1, 2, kSlots, kFingerprint).ok());
        r0.clear_all();
        r1.clear_all();

        r0.set(42);
        r1.set(42);
        r0.set(43);   // rank 1 has not committed 43

        std::vector<const ResidencyBitmap*> ranks = {&r0, &r1};
        CHECK(all_ranks_committed(ranks, 42));
        CHECK(!all_ranks_committed(ranks, 43));   // under-report, correct
        CHECK(!all_ranks_committed(ranks, 44));

        // An unusable rank must make the answer false: claiming cross-rank
        // knowledge we do not have is the one thing that must never happen.
        ResidencyBitmap missing;
        std::vector<const ResidencyBitmap*> with_missing = {&r0, &missing};
        CHECK(!missing.usable());
        CHECK(!all_ranks_committed(with_missing, 42));

        std::vector<const ResidencyBitmap*> with_null = {&r0, nullptr};
        CHECK(!all_ranks_committed(with_null, 42));

        // No ranks at all is also false, not vacuously true.
        CHECK(!all_ranks_committed({}, 42));
    }

    // --- read-only mapping sees a peer's writes and cannot alter them ---
    {
        ResidencyBitmap writer;
        CHECK(writer.open_writable(path1, 1, 2, kSlots, kFingerprint).ok());
        writer.clear_all();
        writer.set(77);
        CHECK(writer.sync().ok());

        ResidencyBitmap reader;
        CHECK(reader.open_readonly(path1, 2, kSlots, kFingerprint).ok());
        CHECK(reader.usable());
        CHECK(reader.test(77));
        reader.set(78);            // must be ignored on a read-only map
        CHECK(!reader.test(78));
        reader.clear(77);          // must be ignored
        CHECK(reader.test(77));
    }

    // --- read-only never creates, and validates geometry ---
    {
        ResidencyBitmap reader;
        CHECK(!reader.open_readonly(dir + "/absent.bitmap", 2, kSlots,
                                    kFingerprint).ok());
        CHECK(!reader.usable());
        ResidencyBitmap wrong;
        CHECK(!wrong.open_readonly(path1, 2, kSlots * 2, kFingerprint).ok());
    }

    // --- an unusable bitmap reads as absent rather than asserting ---
    {
        ResidencyBitmap dead;
        CHECK(!dead.usable());
        CHECK(!dead.test(0));
        dead.set(0);            // no crash, no effect
        dead.clear(0);
        dead.clear_all();
        CHECK(dead.sync().ok()); // sync on an unmapped bitmap is a no-op
        CHECK(dead.slot_count() == 0);
    }

    // --- move semantics do not double-unmap ---
    {
        ResidencyBitmap a;
        CHECK(a.open_writable(path0, 0, 2, kSlots, kFingerprint).ok());
        a.set(11);
        ResidencyBitmap b = std::move(a);
        CHECK(!a.usable());
        CHECK(b.usable());
        CHECK(b.test(11));
        ResidencyBitmap c;
        c = std::move(b);
        CHECK(!b.usable());
        CHECK(c.test(11));
    }

    ::unlink(path0.c_str());
    ::unlink(path1.c_str());
}

// ======================================================================
// 4. Space allocator
// ======================================================================
void test_space_allocator() {
    // --- capacity in bytes, slot count derived ---
    {
        SpaceAllocator alloc;
        SpaceAllocatorConfig cfg;
        cfg.capacity_bytes = 100ull * 1024 * 1024;
        cfg.slot_bytes = kPayload;
        CHECK(alloc.configure(cfg));
        CHECK(alloc.total_slots() == 10);
        CHECK(alloc.slot_bytes() == kPayload);

        // A store that cannot hold one object is a misconfiguration.
        SpaceAllocator too_small;
        SpaceAllocatorConfig tiny = cfg;
        tiny.capacity_bytes = kPayload - 1;
        CHECK(!too_small.configure(tiny));

        SpaceAllocator no_geometry;
        SpaceAllocatorConfig bad = cfg;
        bad.slot_bytes = 0;
        CHECK(!no_geometry.configure(bad));

        // prewarm is clamped to what exists.
        SpaceAllocator warm;
        SpaceAllocatorConfig over = cfg;
        over.prewarm_slots = 1000;
        CHECK(warm.configure(over));
        CHECK(warm.prewarm_slots() == 10);
    }

    // --- reserve / commit / release lifecycle ---
    [&] {
        SpaceAllocator alloc;
        SpaceAllocatorConfig cfg;
        cfg.capacity_bytes = 10 * kPayload;
        cfg.slot_bytes = kPayload;
        CHECK(alloc.configure(cfg));

        SlotReservation r = alloc.reserve(3);
        REQUIRE(r.slots.size() == 3);
        CHECK(r.generations.size() == 3);
        CHECK(r.rejected == 0);
        CHECK(alloc.state_of(r.slots[0]) == SlotState::kReserved);

        alloc.commit(r.slots.data(), r.slots.size());
        CHECK(alloc.state_of(r.slots[0]) == SlotState::kCommitted);

        SpaceAllocatorStats s = alloc.stats();
        CHECK(s.total_slots == 10);
        CHECK(s.committed_slots == 3);
        CHECK(s.reserved_slots == 0);
        // Lazy materialisation: only what was handed out is accounted for.
        CHECK(s.unmaterialised_slots == 7);

        CHECK(alloc.release(r.slots.data(), 1) == 1);
        CHECK(alloc.state_of(r.slots[0]) == SlotState::kReclaiming);
        // Releasing again is a no-op, not a double free.
        CHECK(alloc.release(r.slots.data(), 1) == 0);
    }();

    // --- exhaustion is partial acceptance, never a block or a throw ---
    {
        SpaceAllocator alloc;
        SpaceAllocatorConfig cfg;
        cfg.capacity_bytes = 2 * kPayload;
        cfg.slot_bytes = kPayload;
        CHECK(alloc.configure(cfg));

        SlotReservation r = alloc.reserve(5);
        CHECK(r.slots.size() == 2);
        CHECK(r.rejected == 3);

        // Still exhausted, still not an error.
        SlotReservation again = alloc.reserve(1);
        CHECK(again.slots.empty());
        CHECK(again.rejected == 1);

        SlotReservation none = alloc.reserve(0);
        CHECK(none.slots.empty());
        CHECK(none.rejected == 0);
    }

    // --- abort cannot touch committed data ---
    [&] {
        SpaceAllocator alloc;
        SpaceAllocatorConfig cfg;
        cfg.capacity_bytes = 4 * kPayload;
        cfg.slot_bytes = kPayload;
        CHECK(alloc.configure(cfg));

        SlotReservation r = alloc.reserve(2);
        REQUIRE(r.slots.size() == 2);
        alloc.commit(r.slots.data(), 1);              // first committed
        alloc.abort(r.slots.data(), r.slots.size());  // abort both
        CHECK(alloc.state_of(r.slots[0]) == SlotState::kCommitted);
        CHECK(alloc.state_of(r.slots[1]) == SlotState::kReclaiming);
    }();

    // --- reclamation is asynchronous, and space is unavailable until zeroed ---
    // This models the real constraint: a slot must be physically re-zeroed
    // before reuse, so "eviction outran scrubbing" is observable rather than
    // showing up as a mysterious stall.
    [&] {
        SpaceAllocator alloc;
        SpaceAllocatorConfig cfg;
        cfg.capacity_bytes = 2 * kPayload;
        cfg.slot_bytes = kPayload;
        CHECK(alloc.configure(cfg));

        SlotReservation r = alloc.reserve(2);
        REQUIRE(r.slots.size() == 2);
        alloc.commit(r.slots.data(), r.slots.size());
        CHECK(alloc.release(r.slots.data(), 2) == 2);
        CHECK(alloc.stats().reclaiming_slots == 2);

        // Not yet scrubbed, so not yet allocatable.
        CHECK(alloc.reserve(1).rejected == 1);

        std::vector<std::uint64_t> scrubbing = alloc.take_reclaimable(10);
        REQUIRE(scrubbing.size() == 2);
        // Taken slots are not offered twice: two scrubbers cannot race.
        CHECK(alloc.take_reclaimable(10).empty());
        // Still not allocatable while in flight.
        CHECK(alloc.reserve(1).rejected == 1);

        const std::uint64_t gen_before = alloc.generation_of(scrubbing[0]);
        alloc.finish_reclaim(scrubbing.data(), scrubbing.size());
        CHECK(alloc.state_of(scrubbing[0]) == SlotState::kFree);
        // Generation bump makes any stale placement detectably obsolete.
        CHECK(alloc.generation_of(scrubbing[0]) == gen_before + 1);

        SlotReservation reused = alloc.reserve(2);
        CHECK(reused.slots.size() == 2);
        CHECK(reused.rejected == 0);
    }();

    // --- a failed scrub requeues instead of leaking capacity ---
    [&] {
        SpaceAllocator alloc;
        SpaceAllocatorConfig cfg;
        cfg.capacity_bytes = 1 * kPayload;
        cfg.slot_bytes = kPayload;
        CHECK(alloc.configure(cfg));

        SlotReservation r = alloc.reserve(1);
        REQUIRE(r.slots.size() == 1);
        alloc.commit(r.slots.data(), 1);
        alloc.release(r.slots.data(), 1);

        std::vector<std::uint64_t> taken = alloc.take_reclaimable(1);
        REQUIRE(taken.size() == 1);
        const std::uint64_t gen = alloc.generation_of(taken[0]);

        alloc.requeue_reclaim(taken.data(), taken.size());
        // Still reclaiming, generation unchanged: the slot was never zeroed.
        CHECK(alloc.state_of(taken[0]) == SlotState::kReclaiming);
        CHECK(alloc.generation_of(taken[0]) == gen);
        // And it is offered again, so capacity is not lost.
        CHECK(alloc.take_reclaimable(1).size() == 1);
    }();

    // --- finish_reclaim only accepts slots actually in flight ---
    [&] {
        SpaceAllocator alloc;
        SpaceAllocatorConfig cfg;
        cfg.capacity_bytes = 2 * kPayload;
        cfg.slot_bytes = kPayload;
        CHECK(alloc.configure(cfg));

        SlotReservation r = alloc.reserve(1);
        REQUIRE(r.slots.size() == 1);
        const std::uint64_t slot = r.slots[0];
        const std::uint64_t gen = alloc.generation_of(slot);
        // Reserved, not reclaiming: a spurious completion must not free it.
        alloc.finish_reclaim(&slot, 1);
        CHECK(alloc.state_of(slot) == SlotState::kReserved);
        CHECK(alloc.generation_of(slot) == gen);
    }();

    // --- out-of-range slots are ignored everywhere ---
    {
        SpaceAllocator alloc;
        SpaceAllocatorConfig cfg;
        cfg.capacity_bytes = 2 * kPayload;
        cfg.slot_bytes = kPayload;
        CHECK(alloc.configure(cfg));

        const std::uint64_t bogus = 9999;
        alloc.commit(&bogus, 1);
        alloc.abort(&bogus, 1);
        CHECK(alloc.release(&bogus, 1) == 0);
        alloc.finish_reclaim(&bogus, 1);
        alloc.requeue_reclaim(&bogus, 1);
        CHECK(alloc.state_of(bogus) == SlotState::kFree);
        CHECK(alloc.generation_of(bogus) == 0);
        // Null pointers are tolerated so callers need not special-case empty.
        alloc.commit(nullptr, 0);
        alloc.abort(nullptr, 0);
        CHECK(alloc.release(nullptr, 0) == 0);
        alloc.finish_reclaim(nullptr, 0);
        alloc.requeue_reclaim(nullptr, 0);
    }

    // --- slots are never handed out twice while live ---
    {
        SpaceAllocator alloc;
        SpaceAllocatorConfig cfg;
        cfg.capacity_bytes = 64 * kPayload;
        cfg.slot_bytes = kPayload;
        CHECK(alloc.configure(cfg));

        std::set<std::uint64_t> seen;
        bool duplicate = false;
        for (int round = 0; round < 8; ++round) {
            SlotReservation r = alloc.reserve(8);
            for (std::uint64_t slot : r.slots) {
                if (!seen.insert(slot).second) duplicate = true;
            }
            alloc.commit(r.slots.data(), r.slots.size());
        }
        CHECK(!duplicate);
        CHECK(seen.size() == 64);
        CHECK(alloc.stats().committed_slots == 64);
    }

    // --- state string mapping is total ---
    CHECK(std::string(to_string(SlotState::kFree)) == "free");
    CHECK(std::string(to_string(SlotState::kReserved)) == "reserved");
    CHECK(std::string(to_string(SlotState::kCommitted)) == "committed");
    CHECK(std::string(to_string(SlotState::kReclaiming)) == "reclaiming");
}

} // namespace

int main() {
    test_object_header();
    test_checkpoint();
    test_space_allocator();

    const std::string dir = temp_dir();
    if (dir.empty()) {
        std::printf("FAIL: could not create a temporary directory\n");
        return 1;
    }
    test_residency(dir);
    ::rmdir(dir.c_str());

    if (g_failures == 0) {
        std::printf("storage_objects contract: all checks passed\n");
        return 0;
    }
    std::printf("storage_objects contract: %d failure(s)\n", g_failures);
    return 1;
}

// tests/storage_objects_media_contract/storage_objects_media_contract_test.cpp
//
// Contract test for the placement policy and the physical media layer.
//
// Needs a real filesystem that supports O_DIRECT. That is not incidental: the
// whole layer exists to keep KV data out of the page cache, so a test on a
// filesystem without O_DIRECT would be testing a configuration this project
// does not support. When O_DIRECT is unavailable the media tests report SKIP
// loudly rather than passing silently -- a green run that never exercised the
// real path would be worse than no run.
//
// Scratch location: TMPDIR, else the build tree, else /tmp. A Tutti host
// typically has a small root filesystem and large data mounts, so hardcoding
// /tmp can fill the root.

#include "csrc/storage_objects/checkpoint_region.h"
#include "csrc/storage_objects/object_header_codec.h"
#include "csrc/storage_objects/slot_media.h"
#include "csrc/storage_objects/slot_placement_policy.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace tutti;
using namespace tutti::storage_objects;

int g_failures = 0;
int g_skipped = 0;

void check(bool cond, const char* expr, int line) {
    if (!cond) {
        std::printf("FAIL [line %d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

#define REQUIRE(cond)                          \
    do {                                       \
        if (!(cond)) {                         \
            check(false, #cond, __LINE__);     \
            return;                            \
        }                                      \
    } while (0)

#ifndef TUTTI_TEST_TMPDIR_DEFAULT
#  define TUTTI_TEST_TMPDIR_DEFAULT "/tmp"
#endif

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = (base != nullptr && *base != '\0')
                           ? base
                           : TUTTI_TEST_TMPDIR_DEFAULT;
    tmpl += "/tutti_media_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char* dir = ::mkdtemp(buf.data());
    return dir == nullptr ? std::string() : std::string(dir);
}

// Whether this filesystem accepts O_DIRECT at all. tmpfs does not, and neither
// do some overlay configurations.
bool o_direct_supported(const std::string& dir) {
    const std::string probe = dir + "/.odirect_probe";
    const int fd = ::open(probe.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
    const bool ok = (fd >= 0);
    if (fd >= 0) ::close(fd);
    ::unlink(probe.c_str());
    return ok;
}

constexpr std::uint64_t kSegmentBytes = 131072;   // 128 KiB, as deployed
constexpr std::uint32_t kSegmentCount = 80;       // 80 layers
constexpr std::uint64_t kPayload = kSegmentBytes * kSegmentCount;  // 10 MiB
// A slot's space is the placement's payload prefix plus the payload.
constexpr std::uint64_t kSlotBytes = ObjectHeaderLayout::kHeaderBytes + kPayload;

ObjectKey key_of(std::uint8_t tag, std::size_t len = 18) {
    ObjectKey k;
    k.bytes.assign(len, tag);
    return k;
}

// ======================================================================
// 1. SingleFilePlacement
// ======================================================================
void test_single_file_placement() {
    SingleFilePlacement policy("/mnt/nvme0/ns");

    CHECK(policy.shard_count() == 1);
    CHECK(policy.header_shard() == 0);
    CHECK(policy.header_offset_in_shard() == 0);
    // Segment 0 starts right after the header.
    CHECK(policy.payload_offset() == ObjectHeaderLayout::kHeaderBytes);

    std::vector<std::string> paths;
    CHECK(policy.paths_for_slot(7, &paths).ok());
    REQUIRE(paths.size() == 1);
    CHECK(paths[0] == "/mnt/nvme0/ns/slots/7.obj");

    // Slot files are named by NUMBER, not by key. This is what makes the layout
    // rename-free: binding a key to a slot is a metadata event, so a slot's path
    // never changes across reuse and path-keyed caches upstream stay warm.
    std::vector<std::string> again;
    CHECK(policy.paths_for_slot(7, &again).ok());
    CHECK(again == paths);

    CHECK(policy.shard_file_bytes(kSlotBytes) == kSlotBytes);
    // The URI format is dictated by the local-file resolver, which requires the
    // "file://" prefix followed by an absolute path and takes that path verbatim
    // as the backing file.
    CHECK(policy.resolver_scheme() == "file");
    CHECK(policy.uri_for_slot(7) == "file:///mnt/nvme0/ns/slots/7.obj");
    CHECK(policy.uri_for_slot(0).rfind("file:///", 0) == 0);
    // The URI's path must be exactly the path that gets materialised, or
    // materialisation and resolution would touch different files.
    CHECK(policy.uri_for_slot(7) == "file://" + paths[0]);

    // Distinct slots never collide.
    std::vector<std::string> a, b;
    CHECK(policy.paths_for_slot(1, &a).ok());
    CHECK(policy.paths_for_slot(2, &b).ok());
    CHECK(a != b);

    CHECK(!policy.paths_for_slot(0, nullptr).ok());

    // A root with a trailing slash must not produce a doubled separator.
    SingleFilePlacement trailing("/mnt/nvme0/ns/");
    std::vector<std::string> tp;
    CHECK(trailing.paths_for_slot(3, &tp).ok());
    REQUIRE(tp.size() == 1);
    CHECK(tp[0] == "/mnt/nvme0/ns/slots/3.obj");
}

// ======================================================================
// 2. StripedPlacement
// ======================================================================
void test_striped_placement() {
    const std::vector<std::string> mounts = {"/mnt/nvme0/st", "/mnt/nvme1/st"};
    constexpr std::uint64_t kUnit = 65536;  // 64 KiB, as deployed
    StripedPlacement policy(mounts, kUnit);

    CHECK(policy.shard_count() == 2);
    CHECK(policy.stripe_unit() == kUnit);
    // The header lives on shard 0 only. Replicating it would turn one atomic
    // commit into N fsyncs, where a crash could commit shard 0 and not shard 1.
    CHECK(policy.header_shard() == 0);
    CHECK(policy.header_offset_in_shard() == 0);

    std::vector<std::string> paths;
    CHECK(policy.paths_for_slot(5, &paths).ok());
    REQUIRE(paths.size() == 2);
    // These paths are NOT this policy's choice: the striped resolver derives
    // <mount_i>/striped/<name>.shard<i> from the URI itself, so the policy must
    // reproduce them exactly. A divergence would have materialisation write one
    // set of files while resolution maps another, leaving DMA pointed at
    // unallocated extents.
    CHECK(paths[0] == "/mnt/nvme0/st/striped/5.shard0");
    CHECK(paths[1] == "/mnt/nvme1/st/striped/5.shard1");
    // Shards must live on DIFFERENT mounts, or striping buys nothing.
    CHECK(paths[0].rfind("/mnt/nvme0/", 0) == 0);
    CHECK(paths[1].rfind("/mnt/nvme1/", 0) == 0);

    // Each shard is a whole number of stripe rounds. The formula must survive
    // the resolver's floor, which derives the object's logical size from the
    // shard FILES as  N * floor(shard_bytes / unit) * unit  (see
    // striped_local_nvme/payload.h): sizing from the payload alone leaves the
    // prefix outside that space and the last segment's write lands past the end
    // -- exactly how the first real 8-GPU run failed with
    // "target_offset + length exceeds target size".
    // Slot space = the placement's payload prefix + the payload.
    const std::uint64_t slot_bytes = policy.payload_offset() + kPayload;
    const std::uint64_t shard_bytes = policy.shard_file_bytes(slot_bytes);
    CHECK(shard_bytes % 4096 == 0);
    CHECK(shard_bytes % kUnit == 0);
    const std::uint64_t kPrefix = policy.payload_offset();
    CHECK(kPrefix % (kUnit * 2) == 0);         // payload starts on a whole round
    const std::uint64_t kPayloadAfterPrefix = slot_bytes - kPrefix;
    CHECK(kPayloadAfterPrefix == kPayload);    // payload size is unchanged
    const std::uint64_t kRoundsPerShard =
        (kPrefix + kPayloadAfterPrefix + kUnit * 2 - 1) / (kUnit * 2);
    CHECK(shard_bytes == kRoundsPerShard * kUnit);
    // The invariants that matter: the resolver's logical space covers the
    // payload, and the payload starts on a whole stripe round so a request is
    // confined to the shards it was sized for.
    const std::uint64_t logical_bytes = 2 * (shard_bytes / kUnit) * kUnit;
    CHECK(logical_bytes >= slot_bytes);
    CHECK(policy.payload_offset() % (kUnit * 2) == 0);

    // --- geometry validation ---
    // 10 MiB over 2 shards at 64 KiB units divides evenly: 80 whole rounds.
    CHECK(policy.geometry_valid(slot_bytes));

    // A payload that does not divide evenly into whole stripe rounds must be
    // rejected. Otherwise the final round is short and a segment's tail maps
    // past the end of some shard -- silent corruption, not a clean error.
    CHECK(!policy.geometry_valid(kPrefix + kUnit * 3));
    CHECK(!policy.geometry_valid(kPrefix + 100));
    // A slot with no room for a payload is not a valid geometry.
    CHECK(!policy.geometry_valid(kPrefix));
    CHECK(!policy.geometry_valid(0));

    // A zero stripe unit is nonsense and must not be silently defaulted.
    StripedPlacement no_unit(mounts, 0);
    CHECK(!no_unit.geometry_valid(slot_bytes));

    // The resolver requires a 4096-aligned stripe unit, so an unaligned one must
    // be rejected at open() rather than on the first IO.
    StripedPlacement unaligned(mounts, 1000);
    CHECK(!unaligned.geometry_valid(slot_bytes));

    // No mounts at all is a configuration error, reported rather than crashed.
    StripedPlacement empty({}, kUnit);
    CHECK(empty.shard_count() == 0);
    CHECK(!empty.geometry_valid(slot_bytes));
    std::vector<std::string> none;
    CHECK(!empty.paths_for_slot(0, &none).ok());

    // The URI must carry the name, the device list and the stripe unit in the
    // exact shape the striped resolver parses:
    //     striped://<name>?devs=<m1,m2,...>&unit=<bytes>
    CHECK(policy.resolver_scheme() == "striped");
    const std::string uri = policy.uri_for_slot(5);
    CHECK(uri == "striped://5?devs=/mnt/nvme0/st,/mnt/nvme1/st&unit=65536");
    CHECK(uri.rfind("striped://", 0) == 0);
    CHECK(uri.find("?devs=") != std::string::npos);
    CHECK(uri.find("&unit=65536") != std::string::npos);
    // A pure function of the slot number, so targets can be cached by slot.
    CHECK(policy.uri_for_slot(5) == uri);
    CHECK(policy.uri_for_slot(6) != uri);

    // --- four shards, the other deployed shape ---
    StripedPlacement four({"/a", "/b", "/c", "/d"}, kUnit);
    CHECK(four.shard_count() == 4);
    std::vector<std::string> fp;
    CHECK(four.paths_for_slot(9, &fp).ok());
    REQUIRE(fp.size() == 4);
    CHECK(fp[3] == "/d/striped/9.shard3");
    // Same rule as the two-shard case: payload starts on a whole stripe round
    // and every shard is a whole number of rounds.
    CHECK(four.payload_offset() % (kUnit * 4) == 0);
    const std::uint64_t four_slot_bytes = four.payload_offset() + kPayload;
    CHECK(four.geometry_valid(four_slot_bytes));
    const std::uint64_t four_bytes = four.shard_file_bytes(four_slot_bytes);
    CHECK(four_bytes % kUnit == 0);
    CHECK(4 * (four_bytes / kUnit) * kUnit >= four_slot_bytes);

    // --- the two policies agree on what they must agree on ---
    // Both put the header at shard 0 offset 0, and each keeps its payload
    // start on a whole stripe round (a single file has a one-round geometry,
    // so a stripe-aligned offset there is the same 4096B reservation).
    SingleFilePlacement single("/mnt/nvme0/ns");
    CHECK(single.header_shard() == policy.header_shard());
    CHECK(single.header_offset_in_shard() == policy.header_offset_in_shard());
    CHECK(single.payload_offset() == ObjectHeaderLayout::kHeaderBytes);
    // A single file has no stripe geometry to align to: the payload just has to
    // start on a device IO granularity boundary.
    CHECK(single.payload_offset() % 4096 == 0);
}

// ======================================================================
// 3. AlignedBuffer
// ======================================================================
void test_aligned_buffer() {
    AlignedBuffer buf(4096);
    REQUIRE(buf.valid());
    // O_DIRECT rejects an unaligned buffer address with EINVAL, which is why
    // std::vector cannot be used here.
    CHECK(reinterpret_cast<std::uintptr_t>(buf.data()) % 4096 == 0);
    CHECK(buf.size() == 4096);

    buf.data()[0] = 0xAB;
    buf.zero();
    CHECK(buf.data()[0] == 0);
    CHECK(buf.data()[4095] == 0);

    // A non-multiple request rounds up, so callers never have to.
    AlignedBuffer odd(100);
    REQUIRE(odd.valid());
    CHECK(odd.size() == 4096);
    CHECK(reinterpret_cast<std::uintptr_t>(odd.data()) % 4096 == 0);

    AlignedBuffer big(4096 * 3 + 1);
    REQUIRE(big.valid());
    CHECK(big.size() == 4096 * 4);

    AlignedBuffer empty(0);
    CHECK(!empty.valid());
    CHECK(empty.size() == 0);
    empty.zero();   // must not crash

    // Move must transfer ownership exactly once, or the free is doubled.
    AlignedBuffer src(8192);
    REQUIRE(src.valid());
    src.data()[0] = 0x5A;
    const std::uint8_t* original = src.data();
    AlignedBuffer moved = std::move(src);
    CHECK(!src.valid());
    CHECK(moved.data() == original);
    CHECK(moved.data()[0] == 0x5A);
    AlignedBuffer assigned;
    assigned = std::move(moved);
    CHECK(!moved.valid());
    CHECK(assigned.data() == original);
    CHECK(assigned.data()[0] == 0x5A);
}

// ======================================================================
// 4. Directory helpers
// ======================================================================
void test_directories(const std::string& dir) {
    const std::string nested = dir + "/a/b/c";
    CHECK(ensure_directory(nested).ok());
    struct stat st{};
    CHECK(::stat(nested.c_str(), &st) == 0);
    CHECK(S_ISDIR(st.st_mode));

    // Idempotent. Several ranks bring up the same namespace root concurrently,
    // so "already exists" is the normal case, not an error.
    CHECK(ensure_directory(nested).ok());
    CHECK(ensure_directory(dir).ok());

    CHECK(!ensure_directory("").ok());
    CHECK(sync_directory(dir).ok());
    CHECK(!sync_directory(dir + "/does_not_exist").ok());

    ::rmdir((dir + "/a/b/c").c_str());
    ::rmdir((dir + "/a/b").c_str());
    ::rmdir((dir + "/a").c_str());
}

// ======================================================================
// 5. Materialisation -- must allocate REAL blocks, not a sparse file
// ======================================================================
void test_materialisation(const std::string& dir) {
    const std::string root = dir + "/mat";
    SingleFilePlacement policy(root);

    std::vector<std::string> paths;
    REQUIRE(policy.paths_for_slot(0, &paths).ok());

    // Use a small slot so the test stays fast; the property under test is
    // block allocation, not size.
    constexpr std::uint64_t kSmallSlot = 4096 * 16;  // 64 KiB
    CHECK(materialise_slot(paths, kSmallSlot).ok());

    struct stat st{};
    REQUIRE(::stat(paths[0].c_str(), &st) == 0);
    CHECK(static_cast<std::uint64_t>(st.st_size) == kSmallSlot);

    // THE point of materialisation: blocks must actually be allocated. The
    // resolver maps files to physical extents via FIEMAP and fail-closed rejects
    // UNWRITTEN/DELALLOC extents, because DMA cannot target blocks the
    // filesystem has not committed. A sparse file would pass a size check and
    // then fail at resolve time -- or worse, DMA into nothing.
    const std::uint64_t allocated = static_cast<std::uint64_t>(st.st_blocks) * 512;
    CHECK(allocated >= kSmallSlot);

    // Idempotent: an existing file at the right size is left alone. Rewriting on
    // every restart would make bringing up an existing terabyte pool as
    // expensive as creating it -- and, worse, would destroy the objects in it.
    //
    // Tested by the CONSEQUENCE rather than by mtime: mtime has one-second
    // granularity, so a rewrite within the same second is invisible to it. What
    // actually matters is that re-materialising preserves committed data.
    {
        const ObjectKey survivor = key_of(0x7E);
        const std::uint64_t payload = kSmallSlot - ObjectHeaderLayout::kHeaderBytes;
        std::vector<std::uint8_t> header(ObjectHeaderLayout::kHeaderBytes);
        REQUIRE(encode_object_header(header.data(), header.size(), survivor,
                                     payload, 5, 55));
        REQUIRE(write_object_header(paths[0], 0, header.data(), header.size()).ok());

        // Re-materialise the same slot at the same size.
        CHECK(materialise_slot(paths, kSmallSlot).ok());

        std::vector<std::uint8_t> readback(ObjectHeaderLayout::kHeaderBytes);
        CHECK(read_object_header(paths[0], 0, readback.data(), readback.size()).ok());
        HeaderExpectation still;
        still.key = &survivor;
        still.payload_bytes = payload;
        still.generation = 5;
        // The object must still be there. A non-idempotent materialise would
        // have zeroed it, and the header would decode as kEmptySlot.
        CHECK(decode_object_header(readback.data(), readback.size(), still,
                                   nullptr) == HeaderRejection::kNone);
    }

    struct stat after{};
    REQUIRE(::stat(paths[0].c_str(), &after) == 0);
    CHECK(static_cast<std::uint64_t>(after.st_size) == kSmallSlot);

    // Unaligned or zero sizes are refused rather than silently rounded: the
    // caller's geometry is wrong and should be fixed, not papered over.
    CHECK(!materialise_slot(paths, 0).ok());
    CHECK(!materialise_slot(paths, 1000).ok());
    CHECK(!materialise_slot({}, kSmallSlot).ok());

    // --- striped materialisation creates every shard ---
    StripedPlacement striped({dir + "/st0", dir + "/st1"}, 65536);
    std::vector<std::string> shards;
    REQUIRE(striped.paths_for_slot(3, &shards).ok());
    REQUIRE(shards.size() == 2);
    CHECK(materialise_slot(shards, kSmallSlot).ok());
    for (const std::string& p : shards) {
        struct stat ss{};
        CHECK(::stat(p.c_str(), &ss) == 0);
        CHECK(static_cast<std::uint64_t>(ss.st_size) == kSmallSlot);
        CHECK(static_cast<std::uint64_t>(ss.st_blocks) * 512 >= kSmallSlot);
    }

    // --- zeroing invalidates an object by erasing its header ---
    // A zeroed header decodes as "never written" rather than as corruption,
    // which is what makes a reclaimed slot indistinguishable from a fresh one.
    {
        std::vector<std::uint8_t> header(ObjectHeaderLayout::kHeaderBytes);
        const ObjectKey key = key_of(0x42);
        REQUIRE(encode_object_header(header.data(), header.size(), key,
                                     kSmallSlot - ObjectHeaderLayout::kHeaderBytes,
                                     1, 1));
        CHECK(write_object_header(paths[0], 0, header.data(), header.size()).ok());

        std::vector<std::uint8_t> readback(ObjectHeaderLayout::kHeaderBytes);
        CHECK(read_object_header(paths[0], 0, readback.data(), readback.size()).ok());
        ObjectHeaderFields fields;
        CHECK(peek_object_header(readback.data(), readback.size(), &fields) ==
              HeaderRejection::kNone);

        CHECK(zero_slot(paths, kSmallSlot).ok());
        CHECK(read_object_header(paths[0], 0, readback.data(), readback.size()).ok());
        CHECK(peek_object_header(readback.data(), readback.size(), &fields) ==
              HeaderRejection::kEmptySlot);
    }

    CHECK(!zero_slot(paths, 999).ok());
    CHECK(!zero_slot({dir + "/absent.obj"}, kSmallSlot).ok());
}

// ======================================================================
// 6. Header IO round trip through real media
// ======================================================================
void test_header_io(const std::string& dir) {
    const std::string root = dir + "/hdr";
    SingleFilePlacement policy(root);
    std::vector<std::string> paths;
    REQUIRE(policy.paths_for_slot(11, &paths).ok());

    constexpr std::uint64_t kSmallSlot = 4096 * 8;
    REQUIRE(materialise_slot(paths, kSmallSlot).ok());

    const ObjectKey key = key_of(0xC7);
    const std::uint64_t payload = kSmallSlot - ObjectHeaderLayout::kHeaderBytes;

    std::vector<std::uint8_t> header(ObjectHeaderLayout::kHeaderBytes);
    REQUIRE(encode_object_header(header.data(), header.size(), key, payload, 3, 99));
    CHECK(write_object_header(paths[0], 0, header.data(), header.size()).ok());

    std::vector<std::uint8_t> readback(ObjectHeaderLayout::kHeaderBytes);
    CHECK(read_object_header(paths[0], 0, readback.data(), readback.size()).ok());
    // Byte-identical: the encoding is deterministic, which is what makes a torn
    // write detectable.
    CHECK(std::memcmp(header.data(), readback.data(), header.size()) == 0);

    HeaderExpectation expect;
    expect.key = &key;
    expect.payload_bytes = payload;
    expect.generation = 3;
    ObjectHeaderFields fields;
    CHECK(decode_object_header(readback.data(), readback.size(), expect, &fields) ==
          HeaderRejection::kNone);
    CHECK(fields.commit_seq == 99);

    // A never-written slot reads as empty, not as an error: this is the expected
    // state of free space and must not look like corruption.
    std::vector<std::string> fresh;
    REQUIRE(policy.paths_for_slot(12, &fresh).ok());
    REQUIRE(materialise_slot(fresh, kSmallSlot).ok());
    CHECK(read_object_header(fresh[0], 0, readback.data(), readback.size()).ok());
    CHECK(peek_object_header(readback.data(), readback.size(), &fields) ==
          HeaderRejection::kEmptySlot);

    // An absent slot is NOT_FOUND, distinct from an IO failure.
    const Status missing =
        read_object_header(dir + "/nope.obj", 0, readback.data(), readback.size());
    CHECK(!missing.ok());
    CHECK(missing.code() == StatusCode::NOT_FOUND);

    // O_DIRECT alignment is enforced, not silently worked around: an unaligned
    // request means the caller's geometry is wrong.
    CHECK(!write_object_header(paths[0], 1, header.data(), header.size()).ok());
    CHECK(!write_object_header(paths[0], 0, header.data(), 100).ok());
    CHECK(!write_object_header(paths[0], 0, nullptr, header.size()).ok());
    CHECK(!read_object_header(paths[0], 1, readback.data(), readback.size()).ok());
    CHECK(!read_object_header(paths[0], 0, readback.data(), 100).ok());

    // Overwriting a header in place works, so a slot can be recommitted without
    // being recreated.
    const ObjectKey second = key_of(0xD8);
    REQUIRE(encode_object_header(header.data(), header.size(), second, payload, 4, 100));
    CHECK(write_object_header(paths[0], 0, header.data(), header.size()).ok());
    CHECK(read_object_header(paths[0], 0, readback.data(), readback.size()).ok());
    HeaderExpectation second_expect;
    second_expect.key = &second;
    second_expect.payload_bytes = payload;
    second_expect.generation = 4;
    CHECK(decode_object_header(readback.data(), readback.size(), second_expect,
                              nullptr) == HeaderRejection::kNone);
    // And the previous key no longer resolves there.
    CHECK(decode_object_header(readback.data(), readback.size(), expect, nullptr) ==
          HeaderRejection::kIdentityMismatch);
}

// ======================================================================
// 7. Checkpoint IO round trip, including the crash-recovery sequence
// ======================================================================
void test_checkpoint_io(const std::string& dir) {
    const std::string path = dir + "/meta/checkpoint.bin";

    std::vector<CheckpointEntry> entries;
    for (std::uint8_t i = 1; i <= 4; ++i) {
        CheckpointEntry e;
        e.key = key_of(i);
        e.slot = i;
        e.payload_bytes = kPayload;
        e.generation = i;
        e.commit_seq = 200 + i;
        entries.push_back(e);
    }

    const std::uint64_t container = checkpoint_container_bytes(64, 18);
    const std::uint64_t region = container * CheckpointLayout::kContainerCount;
    CHECK(ensure_metadata_file(path, region).ok());

    struct stat st{};
    REQUIRE(::stat(path.c_str(), &st) == 0);
    CHECK(static_cast<std::uint64_t>(st.st_size) == region);
    // The checkpoint region is materialised too: it is read with O_DIRECT, so
    // its blocks must exist.
    CHECK(static_cast<std::uint64_t>(st.st_blocks) * 512 >= region);

    // Idempotent, and never shrinks an existing region.
    CHECK(ensure_metadata_file(path, region).ok());
    REQUIRE(::stat(path.c_str(), &st) == 0);
    CHECK(static_cast<std::uint64_t>(st.st_size) == region);

    // --- write into container 0, read it back ---
    const std::vector<std::uint8_t> image = encode_checkpoint(entries, 1);
    REQUIRE(image.size() <= container);
    CHECK(write_checkpoint_container(path, checkpoint_container_offset(0, container),
                                    image.data(), image.size()).ok());

    std::vector<std::uint8_t> readback(image.size());
    CHECK(read_checkpoint_container(path, checkpoint_container_offset(0, container),
                                   readback.data(), readback.size()).ok());
    std::uint64_t seq = 0;
    std::vector<CheckpointEntry> decoded;
    CHECK(decode_checkpoint(readback.data(), readback.size(), &seq, &decoded) ==
          CheckpointRejection::kNone);
    CHECK(seq == 1);
    CHECK(decoded.size() == entries.size());

    // --- container 1 is still empty, and reads as such ---
    std::vector<std::uint8_t> empty(CheckpointLayout::kContainerHeaderBytes);
    CHECK(read_checkpoint_container(path, checkpoint_container_offset(1, container),
                                   empty.data(), empty.size()).ok());
    CHECK(decode_checkpoint(empty.data(), empty.size(), &seq, &decoded) ==
          CheckpointRejection::kEmptyContainer);

    // --- the crash-recovery sequence end to end ---
    // Rotation must overwrite the OLDEST container, so the newest valid state
    // survives a crash during the write.
    {
        std::uint64_t seqs[2] = {1, 0};
        CHECK(next_checkpoint_container(seqs, 2) == 1);

        const std::vector<std::uint8_t> second = encode_checkpoint(entries, 2);
        CHECK(write_checkpoint_container(path,
                                        checkpoint_container_offset(1, container),
                                        second.data(), second.size()).ok());
        seqs[1] = 2;
        // Next write targets container 0 again -- never the one holding the
        // newest state.
        CHECK(next_checkpoint_container(seqs, 2) == 0);
        std::uint32_t chosen = 99;
        CHECK(select_checkpoint_container(seqs, 2, &chosen));
        CHECK(chosen == 1);

        // Simulate a crash midway through overwriting container 0: its body is
        // corrupt. Recovery must fall back to container 1, losing nothing that
        // was durable.
        std::vector<std::uint8_t> torn = encode_checkpoint(entries, 3);
        torn[CheckpointLayout::kContainerHeaderBytes + 8] ^= 0xFF;
        CHECK(write_checkpoint_container(path,
                                        checkpoint_container_offset(0, container),
                                        torn.data(), torn.size()).ok());

        std::uint64_t found[2] = {0, 0};
        for (std::uint32_t i = 0; i < 2; ++i) {
            std::vector<std::uint8_t> buf(container);
            if (!read_checkpoint_container(path,
                                           checkpoint_container_offset(i, container),
                                           buf.data(), buf.size()).ok()) {
                continue;
            }
            std::uint64_t s = 0;
            std::vector<CheckpointEntry> tmp;
            if (decode_checkpoint(buf.data(), buf.size(), &s, &tmp) ==
                CheckpointRejection::kNone) {
                found[i] = s;
            }
        }
        CHECK(found[0] == 0);   // torn container is unusable
        CHECK(found[1] == 2);   // previous state intact
        CHECK(select_checkpoint_container(found, 2, &chosen));
        CHECK(chosen == 1);
    }

    // --- alignment is enforced ---
    CHECK(!write_checkpoint_container(path, 1, image.data(), image.size()).ok());
    CHECK(!write_checkpoint_container(path, 0, image.data(), 100).ok());
    CHECK(!read_checkpoint_container(path, 1, readback.data(), readback.size()).ok());
    CHECK(!ensure_metadata_file(path, 100).ok());
    CHECK(!ensure_metadata_file(path, 0).ok());

    const Status absent = read_checkpoint_container(dir + "/meta/nope.bin", 0,
                                                    readback.data(),
                                                    readback.size());
    CHECK(!absent.ok());
    CHECK(absent.code() == StatusCode::NOT_FOUND);
}

} // namespace

int main() {
    // Placement policies are pure computation: always run.
    test_single_file_placement();
    test_striped_placement();
    test_aligned_buffer();

    const std::string dir = temp_dir();
    if (dir.empty()) {
        std::printf("FAIL: could not create a temporary directory\n");
        return 1;
    }

    test_directories(dir);

    if (o_direct_supported(dir)) {
        test_materialisation(dir);
        test_header_io(dir);
        test_checkpoint_io(dir);
    } else {
        // Reported loudly: a green run that never exercised the O_DIRECT path
        // would be more misleading than no run at all.
        std::printf("SKIP: %s does not support O_DIRECT; media tests not run.\n",
                    dir.c_str());
        std::printf("      Set TMPDIR to a directory on a real block-backed "
                    "filesystem to exercise them.\n");
        ++g_skipped;
    }

    // Best-effort cleanup; leftovers in a temp dir are harmless.
    std::string cmd = "rm -rf '" + dir + "'";
    if (std::system(cmd.c_str()) != 0) {
        std::printf("note: could not remove %s\n", dir.c_str());
    }

    if (g_failures == 0) {
        std::printf("storage_objects media contract: all checks passed%s\n",
                    g_skipped != 0 ? " (with skips)" : "");
        return 0;
    }
    std::printf("storage_objects media contract: %d failure(s)\n", g_failures);
    return 1;
}

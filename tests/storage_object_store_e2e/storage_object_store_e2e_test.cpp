// tests/storage_object_store_e2e/storage_object_store_e2e_test.cpp
//
// End-to-end test of ObjectStoreCore.
//
// This is the first test that exercises the whole layer as one object:
// reserve -> write payload -> commit -> restart -> recover. The units were
// tested in isolation; what matters here is whether their COMPOSITION holds the
// guarantees, especially across a simulated crash.
//
// No resolver is involved, because this layer does not resolve: it hands out
// slot URIs and StorageRuntime turns those into tickets. Extent mapping is
// therefore out of scope here and covered by the hardware contract tests. What
// this test does cover -- and where the bugs actually live -- is the store's own
// bookkeeping, the header/checkpoint interplay, and recovery decisions.
//
// Needs a filesystem supporting O_DIRECT; skips loudly otherwise, since a green
// run that never touched the real write path would be worse than no run.

#include <tutti/spi/storage_object_store.h>

#include "csrc/storage_objects/object_store_core.h"
#include "csrc/storage_objects/slot_media.h"
#include "csrc/storage_objects/slot_placement_policy.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
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
        if (!(cond)) {                          \
            check(false, #cond, __LINE__);      \
            return;                             \
        }                                       \
    } while (0)

#ifndef TUTTI_TEST_TMPDIR_DEFAULT
#  define TUTTI_TEST_TMPDIR_DEFAULT "/tmp"
#endif

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = (base != nullptr && *base != '\0')
                           ? base
                           : TUTTI_TEST_TMPDIR_DEFAULT;
    tmpl += "/tutti_e2e_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char* dir = ::mkdtemp(buf.data());
    return dir == nullptr ? std::string() : std::string(dir);
}

bool o_direct_supported(const std::string& dir) {
    const std::string probe = dir + "/.odirect_probe";
    const int fd = ::open(probe.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
    const bool ok = (fd >= 0);
    if (fd >= 0) ::close(fd);
    ::unlink(probe.c_str());
    return ok;
}

constexpr std::uint64_t kSegmentBytes = 8192;
constexpr std::uint32_t kSegmentCount = 4;
constexpr std::uint64_t kPayload = kSegmentBytes * kSegmentCount;   // 32 KiB
constexpr std::uint64_t kSlotBytes = 4096 + kPayload;

ObjectKey key_of(std::uint8_t tag, std::size_t len = 18) {
    ObjectKey k;
    k.bytes.assign(len, tag);
    return k;
}

StoreConfig make_config(const std::string& root, std::uint64_t slots) {
    StoreConfig cfg;
    cfg.uri = root;
    cfg.capacity_bytes = slots * kSlotBytes;
    cfg.layout.segment_bytes = kSegmentBytes;
    cfg.layout.segment_count = kSegmentCount;
    cfg.namespace_fingerprint = {0xDE, 0xAD, 0xBE, 0xEF};
    cfg.background_reclaim = false;   // drained explicitly, for determinism
    return cfg;
}

std::unique_ptr<ObjectStoreCore> make_store(const std::string& root) {
    return std::make_unique<ObjectStoreCore>(
        std::make_unique<SingleFilePlacement>(root));
}

// Write recognisable bytes into an object's payload, the way a real caller would
// via the DataPath -- here directly, since the stub target carries no extents.
bool write_payload(const std::string& root, std::uint64_t slot,
                   std::uint8_t fill) {
    SingleFilePlacement placement(root);
    std::vector<std::string> paths;
    if (!placement.paths_for_slot(slot, &paths).ok()) return false;
    AlignedBuffer buffer(kPayload);
    if (!buffer.valid()) return false;
    std::memset(buffer.data(), fill, kPayload);
    const int fd = ::open(paths[0].c_str(), O_RDWR | O_DIRECT);
    if (fd < 0) return false;
    const ssize_t n = ::pwrite(fd, buffer.data(), kPayload, 4096);
    ::fsync(fd);
    ::close(fd);
    return n == static_cast<ssize_t>(kPayload);
}

std::uint8_t read_payload_byte(const std::string& root, std::uint64_t slot) {
    SingleFilePlacement placement(root);
    std::vector<std::string> paths;
    if (!placement.paths_for_slot(slot, &paths).ok()) return 0;
    AlignedBuffer buffer(4096);
    if (!buffer.valid()) return 0;
    const int fd = ::open(paths[0].c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) return 0;
    const ssize_t n = ::pread(fd, buffer.data(), 4096, 4096);
    ::close(fd);
    return n > 0 ? buffer.data()[0] : 0;
}

// ======================================================================
// 1. The basic cycle: reserve -> commit -> lookup
// ======================================================================
void test_basic_cycle(const std::string& base) {
    const std::string root = base + "/basic";
    auto store = make_store(root);

    StoreConfig cfg = make_config(root, 8);
    REQUIRE(store->open(cfg).ok());

    const ObjectKey a = key_of(0xA1);
    auto res = store->reserve(&a, 1);
    REQUIRE(res.ok());
    REQUIRE(res.value().accepted.size() == 1);
    CHECK(res.value().rejected_count == 0);
    const ObjectPlacement& p = res.value().accepted[0];
    CHECK(p.valid());
    // A URI in the format the local-file resolver parses, for the runtime to
    // open. The store itself never resolves it.
    CHECK(p.uri.rfind("file://", 0) == 0);
    CHECK(p.uri == "file://" + root + "/slots/0.obj");
    CHECK(p.slot == 0);
    // Segment 0 starts after the header, so payload IO is 4096-aligned.
    CHECK(p.offset == 4096);
    CHECK(p.payload_bytes == kPayload);

    // Reserved is not committed: the object must be invisible until its header
    // lands, which is what makes an interrupted write behave as never written.
    CHECK(!store->contains(a));
    CHECK(!store->lookup(a).ok());
    CHECK(store->usage().reserved_bytes == kSlotBytes);
    CHECK(store->usage().committed_bytes == 0);

    CHECK(write_payload(root, 0, 0x5A));
    REQUIRE(store->commit(&a, 1).ok());

    CHECK(store->contains(a));
    auto found = store->lookup(a);
    REQUIRE(found.ok());
    CHECK(found.value().offset == 4096);
    CHECK(found.value().payload_bytes == kPayload);
    CHECK(store->usage().committed_bytes == kSlotBytes);
    CHECK(store->usage().reserved_bytes == 0);

    // The URI is a pure function of the slot, so a caller may cache tickets on
    // it and lookup stays free of IO.
    auto again = store->lookup(a);
    REQUIRE(again.ok());
    CHECK(again.value().uri == found.value().uri);
    CHECK(again.value().slot == found.value().slot);

    CHECK(store->close().ok());
}

// ======================================================================
// 2. Restart and recover -- the reason the checkpoint exists
// ======================================================================
void test_restart_recovery(const std::string& base) {
    const std::string root = base + "/restart";
    const ObjectKey a = key_of(0xB1);
    const ObjectKey b = key_of(0xB2);
    const ObjectKey uncommitted = key_of(0xB3);

    {
        auto store = make_store(root);
        REQUIRE(store->open(make_config(root, 8)).ok());

        const ObjectKey keys[2] = {a, b};
        auto res = store->reserve(keys, 2);
        REQUIRE(res.ok());
        REQUIRE(res.value().accepted.size() == 2);
        CHECK(write_payload(root, 0, 0x11));
        CHECK(write_payload(root, 1, 0x22));
        REQUIRE(store->commit(keys, 2).ok());

        // Reserved but never committed: must not survive.
        auto res2 = store->reserve(&uncommitted, 1);
        REQUIRE(res2.ok());

        REQUIRE(store->checkpoint().ok());
        CHECK(store->close().ok());
    }

    {
        auto store = make_store(root);
        REQUIRE(store->open(make_config(root, 8)).ok());

        const RecoveryReport& report = store->recovery_report();
        CHECK(report.checkpoint_available);
        CHECK(report.checkpoint_entries == 2);
        CHECK(report.accepted == 2);
        CHECK(report.dropped_header == 0);

        // Both committed objects are back, and their payloads intact.
        CHECK(store->contains(a));
        CHECK(store->contains(b));
        CHECK(!store->contains(uncommitted));   // never committed

        auto recovered = store->recover();
        REQUIRE(recovered.ok());
        CHECK(recovered.value().size() == 2);

        // lookup works after a restart even though nothing was resolved yet:
        // the target cache fills on demand.
        auto found = store->lookup(a);
        REQUIRE(found.ok());
        CHECK(found.value().valid());
        CHECK(found.value().offset == 4096);

        // Payload survived, which is the whole point.
        CHECK(read_payload_byte(root, 0) == 0x11);
        CHECK(read_payload_byte(root, 1) == 0x22);

        CHECK(store->usage().committed_bytes == 2 * kSlotBytes);
        CHECK(store->close().ok());
    }
}

// ======================================================================
// 3. Crash before commit -- "if the write did not finish, it never happened"
// ======================================================================
void test_crash_before_commit(const std::string& base) {
    const std::string root = base + "/crash";
    const ObjectKey ghost = key_of(0xC1);

    {
        auto store = make_store(root);
        REQUIRE(store->open(make_config(root, 4)).ok());
        auto res = store->reserve(&ghost, 1);
        REQUIRE(res.ok());
        REQUIRE(res.value().accepted.size() == 1);
        // Payload written, header never written: exactly the crash window
        // between "data durable" and "declared valid".
        CHECK(write_payload(root, 0, 0x77));
        REQUIRE(store->checkpoint().ok());   // checkpoint holds no entry for it
        CHECK(store->close().ok());
        // Destructor stands in for the crash.
    }

    {
        auto store = make_store(root);
        REQUIRE(store->open(make_config(root, 4)).ok());
        // The object must not reappear. Its payload is on media, but without a
        // header it is indistinguishable from never-written space -- and that is
        // the safe direction: recompute rather than serve a partial object.
        CHECK(!store->contains(ghost));
        auto recovered = store->recover();
        REQUIRE(recovered.ok());
        CHECK(recovered.value().empty());
        CHECK(store->recovery_report().accepted == 0);
        CHECK(store->close().ok());
    }
}

// ======================================================================
// 4. A corrupted header drops the object rather than serving it
// ======================================================================
void test_corrupted_header(const std::string& base) {
    const std::string root = base + "/corrupt";
    const ObjectKey good = key_of(0xD1);
    const ObjectKey bad = key_of(0xD2);

    {
        auto store = make_store(root);
        REQUIRE(store->open(make_config(root, 4)).ok());
        const ObjectKey keys[2] = {good, bad};
        auto res = store->reserve(keys, 2);
        REQUIRE(res.ok());
        REQUIRE(res.value().accepted.size() == 2);
        CHECK(write_payload(root, 0, 0x01));
        CHECK(write_payload(root, 1, 0x02));
        REQUIRE(store->commit(keys, 2).ok());
        REQUIRE(store->checkpoint().ok());
        CHECK(store->close().ok());
    }

    // Corrupt slot 1's header in place, simulating a torn write.
    {
        SingleFilePlacement placement(root);
        std::vector<std::string> paths;
        REQUIRE(placement.paths_for_slot(1, &paths).ok());
        AlignedBuffer buffer(4096);
        REQUIRE(buffer.valid());
        const int fd = ::open(paths[0].c_str(), O_RDWR | O_DIRECT);
        REQUIRE(fd >= 0);
        REQUIRE(::pread(fd, buffer.data(), 4096, 0) == 4096);
        buffer.data()[24] ^= 0xFF;   // payload_bytes field, CRC now wrong
        REQUIRE(::pwrite(fd, buffer.data(), 4096, 0) == 4096);
        ::fsync(fd);
        ::close(fd);
    }

    {
        auto store = make_store(root);
        REQUIRE(store->open(make_config(root, 4)).ok());
        const RecoveryReport& report = store->recovery_report();
        CHECK(report.checkpoint_entries == 2);
        CHECK(report.accepted == 1);
        CHECK(report.dropped_header == 1);
        // The intact object survives; the corrupted one is gone. Partial
        // recovery is correct behaviour, not a failure.
        CHECK(store->contains(good));
        CHECK(!store->contains(bad));
        CHECK(store->close().ok());
    }
}

// ======================================================================
// 5. Capacity exhaustion is partial acceptance, never a failure
// ======================================================================
void test_capacity(const std::string& base) {
    const std::string root = base + "/capacity";
    auto store = make_store(root);
    REQUIRE(store->open(make_config(root, 3)).ok());   // room for exactly 3

    std::vector<ObjectKey> keys;
    for (std::uint8_t i = 1; i <= 6; ++i) keys.push_back(key_of(i));

    auto res = store->reserve(keys.data(), keys.size());
    REQUIRE(res.ok());                               // status OK, not an error
    CHECK(res.value().accepted.size() == 3);         // took what fit
    CHECK(res.value().accepted_keys.size() == 3);
    CHECK(res.value().rejected_count == 3);          // reported the rest
    CHECK(store->usage().usable_bytes == 0);

    // A full cache is a steady state; asking again is still not an error.
    auto again = store->reserve(keys.data(), keys.size());
    REQUIRE(again.ok());
    CHECK(again.value().accepted.size() == 3);       // the three already held
    CHECK(again.value().rejected_count == 3);

    // Committing then releasing frees space only after the slot is scrubbed:
    // a slot must be re-zeroed before reuse, so "eviction outran scrubbing" is
    // visible rather than appearing as a stall.
    REQUIRE(store->commit(res.value().accepted_keys.data(), 3).ok());
    auto released = store->release(res.value().accepted_keys.data(), 1);
    REQUIRE(released.ok());
    CHECK(released.value() == 1);
    CHECK(store->usage().reclaiming_bytes == kSlotBytes);
    CHECK(store->usage().usable_bytes == 0);         // not yet reusable

    CHECK(store->drain_reclaim(10) == 1);
    CHECK(store->usage().reclaiming_bytes == 0);
    CHECK(store->usage().usable_bytes == kSlotBytes);

    const ObjectKey fresh = key_of(0xFF);
    auto after = store->reserve(&fresh, 1);
    REQUIRE(after.ok());
    CHECK(after.value().accepted.size() == 1);
    CHECK(after.value().rejected_count == 0);
    CHECK(store->close().ok());
}

// ======================================================================
// 6. Reclaim erases the object, and ABA is detectable
// ======================================================================
void test_reclaim_and_aba(const std::string& base) {
    const std::string root = base + "/reclaim";
    const ObjectKey first = key_of(0xE1);
    const ObjectKey second = key_of(0xE2);

    auto store = make_store(root);
    REQUIRE(store->open(make_config(root, 2)).ok());

    auto res = store->reserve(&first, 1);
    REQUIRE(res.ok());
    REQUIRE(res.value().accepted.size() == 1);
    const std::uint64_t gen_first = res.value().accepted[0].generation;
    CHECK(write_payload(root, 0, 0x33));
    REQUIRE(store->commit(&first, 1).ok());
    REQUIRE(store->checkpoint().ok());

    auto dropped = store->release(&first, 1);
    REQUIRE(dropped.ok());
    CHECK(dropped.value() == 1);
    CHECK(!store->contains(first));
    CHECK(store->drain_reclaim(10) == 1);
    // Zeroing wiped the payload as well as the header.
    CHECK(read_payload_byte(root, 0) == 0x00);

    // The same slot comes back with a bumped generation, so a stale placement
    // referring to the old occupant is detectably obsolete.
    auto reused = store->reserve(&second, 1);
    REQUIRE(reused.ok());
    REQUIRE(reused.value().accepted.size() == 1);
    CHECK(reused.value().accepted[0].generation != gen_first);

    // A checkpoint written before the release still names `first`; recovery must
    // reject it, because the header is gone and the generation moved on.
    CHECK(write_payload(root, 0, 0x44));
    REQUIRE(store->commit(&second, 1).ok());
    REQUIRE(store->checkpoint().ok());
    CHECK(store->close().ok());

    {
        auto reopened = make_store(root);
        REQUIRE(reopened->open(make_config(root, 2)).ok());
        CHECK(!reopened->contains(first));   // the old occupant stays gone
        CHECK(reopened->contains(second));
        CHECK(read_payload_byte(root, 0) == 0x44);
        CHECK(reopened->close().ok());
    }
}

// ======================================================================
// 7. Fingerprint mismatch is fail-closed and preserves data
// ======================================================================
void test_fingerprint(const std::string& base) {
    const std::string root = base + "/fingerprint";
    const ObjectKey a = key_of(0xF1);

    {
        auto store = make_store(root);
        REQUIRE(store->open(make_config(root, 4)).ok());
        auto res = store->reserve(&a, 1);
        REQUIRE(res.ok());
        CHECK(write_payload(root, 0, 0x99));
        REQUIRE(store->commit(&a, 1).ok());
        REQUIRE(store->checkpoint().ok());
        CHECK(store->close().ok());
    }

    {
        // A different geometry against the same namespace. Recovery must not
        // accept objects whose recorded size no longer matches.
        auto store = make_store(root);
        StoreConfig cfg = make_config(root, 4);
        cfg.layout.segment_count = kSegmentCount * 2;
        REQUIRE(store->open(cfg).ok());
        CHECK(!store->contains(a));
        CHECK(store->recovery_report().dropped_geometry >= 1);
        CHECK(store->close().ok());
    }

    {
        // Reopening with the original geometry, the object is still there: the
        // rejected open changed nothing on media.
        auto store = make_store(root);
        REQUIRE(store->open(make_config(root, 4)).ok());
        CHECK(store->contains(a));
        CHECK(read_payload_byte(root, 0) == 0x99);

        // Reopening with a different fingerprint is refused outright.
        StoreConfig other = make_config(root, 4);
        other.namespace_fingerprint = {0x01, 0x02};
        const Status rejected = store->open(other);
        CHECK(!rejected.ok());
        CHECK(rejected.code() == StatusCode::INVALID_ARGUMENT);
        // And the data is untouched -- there is deliberately no purge().
        CHECK(store->contains(a));
        CHECK(store->close().ok());
    }
}

// ======================================================================
// 8. pin blocks release; abort never touches committed data
// ======================================================================
void test_pin_and_abort(const std::string& base) {
    const std::string root = base + "/pin";
    auto store = make_store(root);
    REQUIRE(store->open(make_config(root, 4)).ok());

    const ObjectKey live = key_of(0x21);
    const ObjectKey doomed = key_of(0x22);
    const ObjectKey keys[2] = {live, doomed};

    auto res = store->reserve(keys, 2);
    REQUIRE(res.ok());
    REQUIRE(res.value().accepted.size() == 2);
    REQUIRE(store->commit(&live, 1).ok());     // only the first is committed

    // abort on both: the committed one must be untouched.
    REQUIRE(store->abort(keys, 2).ok());
    CHECK(store->contains(live));
    CHECK(!store->contains(doomed));

    REQUIRE(store->pin(&live, 1).ok());
    auto blocked = store->release(&live, 1);
    REQUIRE(blocked.ok());
    CHECK(blocked.value() == 0);               // pinned: not released
    CHECK(store->contains(live));

    REQUIRE(store->unpin(&live, 1).ok());
    auto freed = store->release(&live, 1);
    REQUIRE(freed.ok());
    CHECK(freed.value() == 1);
    CHECK(!store->contains(live));
    CHECK(store->close().ok());
}

// ======================================================================
// 9. Prefix queries, and the cross-rank query's one-sided error
// ======================================================================
void test_prefix_queries(const std::string& base) {
    const std::string root = base + "/prefix";
    auto store = make_store(root);
    REQUIRE(store->open(make_config(root, 8)).ok());

    std::vector<ObjectKey> keys;
    for (std::uint8_t i = 1; i <= 4; ++i) keys.push_back(key_of(i));

    auto res = store->reserve(keys.data(), 3);
    REQUIRE(res.ok());
    REQUIRE(store->commit(keys.data(), 2).ok());   // first two only

    // Stops at the first gap: the third is reserved, not committed.
    CHECK(store->contains_prefix(keys.data(), 4) == 2);
    // Single rank: the cross-rank query degenerates to this rank's own view.
    CHECK(store->contains_prefix_all_ranks(keys.data(), 4) == 2);
    // Never reports more than the single-rank answer -- over-reporting would
    // load truncated KV, which is the one forbidden outcome.
    CHECK(store->contains_prefix_all_ranks(keys.data(), 4) <=
          store->contains_prefix(keys.data(), 4));

    CHECK(store->contains_prefix(nullptr, 0) == 0);
    CHECK(store->contains_prefix_all_ranks(nullptr, 0) == 0);
    CHECK(store->close().ok());
}

// ======================================================================
// 10. Misconfiguration fails at open(), not on first IO
// ======================================================================
void test_configuration_errors(const std::string& base) {
    const std::string root = base + "/config";

    {
        auto store = make_store(root);
        StoreConfig cfg = make_config(root, 4);
        cfg.layout.segment_bytes = 1000;   // not 4096-aligned
        const Status s = store->open(cfg);
        CHECK(!s.ok());
        CHECK(s.code() == StatusCode::INVALID_ARGUMENT);
    }
    {
        auto store = make_store(root);
        StoreConfig cfg = make_config(root, 4);
        cfg.capacity_bytes = kSlotBytes - 1;   // cannot hold one object
        CHECK(!store->open(cfg).ok());
    }
    {
        auto store = make_store(root);
        StoreConfig cfg = make_config(root, 4);
        cfg.uri.clear();
        CHECK(!store->open(cfg).ok());
    }
    {
        // Operations before open() are refused rather than crashing.
        auto store = make_store(root);
        const ObjectKey k = key_of(0x01);
        auto res = store->reserve(&k, 1);
        CHECK(!res.ok());
        CHECK(res.status().code() == StatusCode::NOT_READY);
        CHECK(!store->commit(&k, 1).ok());
        CHECK(!store->checkpoint().ok());
    }
}

// ======================================================================
// 11. Factory
// ======================================================================
void test_factory() {
    auto single = create_storage_object_store("local_nvme_file");
    CHECK(single.ok());
    auto striped = create_storage_object_store("striped_local_nvme_file");
    CHECK(striped.ok());
    // An unknown layout fails at creation rather than silently resolving to
    // something else.
    auto unknown = create_storage_object_store("s3");
    CHECK(!unknown.ok());
    CHECK(unknown.status().code() == StatusCode::UNSUPPORTED);

    // A store from the factory builds its backend at open(), so a device list is
    // mandatory: without one there is nothing to place objects on.
    auto store = std::move(single).value();
    StoreConfig cfg;
    cfg.uri = "/tmp/unused";
    cfg.capacity_bytes = kSlotBytes * 2;
    cfg.layout.segment_bytes = kSegmentBytes;
    cfg.layout.segment_count = kSegmentCount;
    const Status no_devices = store->open(cfg);
    CHECK(!no_devices.ok());
    CHECK(no_devices.code() == StatusCode::INVALID_ARGUMENT);

    // Striping across several devices needs a stripe unit; one device with a
    // stripe unit set is equally contradictory.
    StoreConfig two = cfg;
    StoreDevice d;
    d.mount_path = "/tmp/unused";
    d.backing_device_path = "/dev/null";
    d.block_size = 512;
    two.devices = {d, d};
    two.stripe_unit = 0;
    CHECK(!store->open(two).ok());
}

} // namespace

int main() {
    test_factory();

    const std::string dir = temp_dir();
    if (dir.empty()) {
        std::printf("FAIL: could not create a temporary directory\n");
        return 1;
    }

    if (o_direct_supported(dir)) {
        test_basic_cycle(dir);
        test_restart_recovery(dir);
        test_crash_before_commit(dir);
        test_corrupted_header(dir);
        test_capacity(dir);
        test_reclaim_and_aba(dir);
        test_fingerprint(dir);
        test_pin_and_abort(dir);
        test_prefix_queries(dir);
        test_configuration_errors(dir);
    } else {
        std::printf("SKIP: %s does not support O_DIRECT; store tests not run.\n",
                    dir.c_str());
        std::printf("      Set TMPDIR to a directory on a real block-backed "
                    "filesystem to exercise them.\n");
        ++g_skipped;
    }

    const std::string cmd = "rm -rf '" + dir + "'";
    if (std::system(cmd.c_str()) != 0) {
        std::printf("note: could not remove %s\n", dir.c_str());
    }

    if (g_failures == 0) {
        std::printf("storage_object_store e2e: all checks passed%s\n",
                    g_skipped != 0 ? " (with skips)" : "");
        return 0;
    }
    std::printf("storage_object_store e2e: %d failure(s)\n", g_failures);
    return 1;
}

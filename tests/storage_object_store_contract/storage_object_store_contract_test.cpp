// tests/storage_object_store_contract/storage_object_store_contract_test.cpp
//
// Hardware-free contract test for <tutti/spi/storage_object_store.h>.
// Plain C++17 executable; no GTest, no CUDA SDK, no hardware, no IO.
//
// This test pins the SPI's *contract*, not any implementation:
//   1. On-media layout constants (alignment, field packing, CRC coverage).
//   2. The interface is implementable by a minimal subclass.
//   3. The semantic invariants that callers are allowed to rely on --
//      chiefly that capacity exhaustion is a partial-acceptance result with
//      status OK rather than an error, and that the cross-rank residency
//      query may only under-report.
//
// ResolvedTarget is only referenced by pointer in ObjectPlacement, so no
// resolver implementation is needed here.

#include <tutti/spi/storage_object_store.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// -------------------------------------------------------------------------
// Compile-time layout contract
// -------------------------------------------------------------------------
namespace {

using tutti::CheckpointLayout;
using tutti::ObjectHeaderLayout;
using tutti::ResidencyBitmapLayout;

// O_DIRECT requires 4096-byte alignment of buffer, offset and length. The
// object header is what shifts the payload start, so a misaligned header
// would misalign every payload IO behind it.
static_assert(ObjectHeaderLayout::kHeaderBytes % 4096 == 0,
              "object header must keep the payload O_DIRECT aligned");

// header_crc32 must cover exactly the fields that precede it: covering less
// would leave metadata unprotected, covering more would include itself.
static_assert(ObjectHeaderLayout::kCrcCoveredBytes ==
                  ObjectHeaderLayout::kHeaderCrc32Offset,
              "header crc must cover exactly the preceding fields");

// Every declared field must fit inside the header.
static_assert(ObjectHeaderLayout::kHeaderCrc32Offset + sizeof(std::uint32_t) <=
                  ObjectHeaderLayout::kHeaderBytes,
              "header fields must fit inside the header");

// Field offsets must not overlap. Checked pairwise in declaration order with
// each field's declared width.
static_assert(ObjectHeaderLayout::kMagicOffset + 8 <=
                  ObjectHeaderLayout::kVersionOffset, "magic/version overlap");
static_assert(ObjectHeaderLayout::kVersionOffset + 4 <=
                  ObjectHeaderLayout::kIdentityOffset, "version/identity overlap");
static_assert(ObjectHeaderLayout::kIdentityOffset + 8 <=
                  ObjectHeaderLayout::kPayloadBytesOffset,
              "identity/payload_bytes overlap");
static_assert(ObjectHeaderLayout::kPayloadBytesOffset + 8 <=
                  ObjectHeaderLayout::kGenerationOffset,
              "payload_bytes/generation overlap");
static_assert(ObjectHeaderLayout::kGenerationOffset + 8 <=
                  ObjectHeaderLayout::kKeyLenOffset, "generation/key_len overlap");
static_assert(ObjectHeaderLayout::kKeyLenOffset + 4 <=
                  ObjectHeaderLayout::kKeyCrc32Offset, "key_len/key_crc32 overlap");
static_assert(ObjectHeaderLayout::kKeyCrc32Offset + 4 <=
                  ObjectHeaderLayout::kCommitSeqOffset,
              "key_crc32/commit_seq overlap");
static_assert(ObjectHeaderLayout::kCommitSeqOffset + 8 <=
                  ObjectHeaderLayout::kHeaderCrc32Offset,
              "commit_seq/header_crc32 overlap");

// Atomicity of the checkpoint comes from never overwriting the container that
// currently holds the newest valid state, which needs at least two of them.
static_assert(CheckpointLayout::kContainerCount >= 2,
              "checkpoint atomicity requires mirrored containers");
static_assert(CheckpointLayout::kContainerHeaderBytes % 4096 == 0,
              "checkpoint container header must be O_DIRECT aligned");
static_assert(CheckpointLayout::kBodyCrc32Offset + sizeof(std::uint32_t) <=
                  CheckpointLayout::kContainerHeaderBytes,
              "checkpoint header fields must fit inside the container header");

// The residency bitmap is mmapped, so its body must start on a page boundary.
static_assert(ResidencyBitmapLayout::kHeaderBytes % 4096 == 0,
              "residency bitmap body must start page aligned");
static_assert(ResidencyBitmapLayout::kFingerprintOffset + 8 <=
                  ResidencyBitmapLayout::kHeaderBytes,
              "residency header fields must fit inside the header");

// The three on-media regions must be distinguishable from each other, so no
// two magics may be equal.
constexpr bool magics_differ(const char (&a)[8], const char (&b)[8]) {
    for (std::size_t i = 0; i < 8; ++i) {
        if (a[i] != b[i]) return true;
    }
    return false;
}
static_assert(magics_differ(ObjectHeaderLayout::kMagic, CheckpointLayout::kMagic),
              "object and checkpoint magics must differ");
static_assert(magics_differ(ObjectHeaderLayout::kMagic,
                            ResidencyBitmapLayout::kMagic),
              "object and residency magics must differ");
static_assert(magics_differ(CheckpointLayout::kMagic,
                            ResidencyBitmapLayout::kMagic),
              "checkpoint and residency magics must differ");

// The SPI must be abstract (no accidental concrete base) and non-copyable via
// its interface.
static_assert(std::is_abstract<tutti::StorageObjectStore>::value,
              "StorageObjectStore must be an abstract interface");
static_assert(std::has_virtual_destructor<tutti::StorageObjectStore>::value,
              "StorageObjectStore must have a virtual destructor");

} // namespace

// -------------------------------------------------------------------------
// FakeStore -- minimal in-memory implementation.
//
// It exists to prove the SPI is implementable and to exercise the semantic
// invariants. It models capacity in whole objects and keeps no media.
// -------------------------------------------------------------------------
namespace {

class FakeStore final : public tutti::StorageObjectStore {
public:
    tutti::Status open(const tutti::StoreConfig& config) override {
        if (config.layout.segment_bytes == 0 || config.layout.segment_count == 0) {
            return tutti::Status(tutti::StatusCode::INVALID_ARGUMENT,
                                 "layout must be fully specified");
        }
        if (opened_) {
            // Fail-closed on fingerprint/layout mismatch; never silently wipe.
            if (!(config.layout == config_.layout) ||
                config.namespace_fingerprint != config_.namespace_fingerprint) {
                return tutti::Status(tutti::StatusCode::INVALID_ARGUMENT,
                                     "namespace fingerprint or layout mismatch");
            }
        }
        config_ = config;
        opened_ = true;
        const std::uint64_t payload = config.layout.payload_bytes();
        capacity_objects_ = payload == 0 ? 0 : config.capacity_bytes / payload;
        return {};
    }

    tutti::Status close() override {
        opened_ = false;
        return {};
    }

    bool contains(const tutti::ObjectKey& key) const override {
        return committed_.count(hex(key)) != 0;
    }

    std::uint64_t contains_prefix(const tutti::ObjectKey* keys,
                                  std::size_t count) const override {
        std::uint64_t hit = 0;
        for (std::size_t i = 0; i < count; ++i) {
            if (!contains(keys[i])) break;
            ++hit;
        }
        return hit;
    }

    std::uint64_t contains_prefix_all_ranks(const tutti::ObjectKey* keys,
                                            std::size_t count) const override {
        // Single-rank config, or an unusable bitmap, degenerates to this
        // rank's own view. Never reports more than contains_prefix.
        if (config_.rank_count <= 1 || !bitmap_usable_) {
            return contains_prefix(keys, count);
        }
        std::uint64_t hit = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const std::string k = hex(keys[i]);
            if (!committed_.count(k)) break;
            if (!all_rank_bits_.count(k)) break;  // lagging bitmap: under-report
            ++hit;
        }
        return hit;
    }

    tutti::Result<tutti::ObjectPlacement> lookup(
        const tutti::ObjectKey& key) const override {
        auto it = committed_.find(hex(key));
        if (it == committed_.end()) {
            return tutti::Result<tutti::ObjectPlacement>::Failure(
                tutti::Status(tutti::StatusCode::NOT_FOUND, "not committed"));
        }
        return tutti::Result<tutti::ObjectPlacement>::Success(it->second);
    }

    tutti::StoreUsage usage() const override {
        const std::uint64_t payload = config_.layout.payload_bytes();
        tutti::StoreUsage u;
        u.capacity_bytes = capacity_objects_ * payload;
        u.committed_bytes = committed_.size() * payload;
        u.reserved_bytes = reserved_.size() * payload;
        u.usable_bytes =
            (capacity_objects_ - committed_.size() - reserved_.size()) * payload;
        return u;
    }

    tutti::Result<tutti::ReserveOutcome> reserve(const tutti::ObjectKey* keys,
                                                 std::size_t count) override {
        tutti::ReserveOutcome out;
        for (std::size_t i = 0; i < count; ++i) {
            const std::string k = hex(keys[i]);
            auto committed = committed_.find(k);
            if (committed != committed_.end()) {
                // Already durable: hand back the existing placement.
                out.accepted.push_back(committed->second);
                out.accepted_keys.push_back(keys[i]);
                continue;
            }
            if (reserved_.count(k)) {
                out.accepted.push_back(reserved_[k]);
                out.accepted_keys.push_back(keys[i]);
                continue;
            }
            if (committed_.size() + reserved_.size() >= capacity_objects_) {
                // Capacity exhausted is a steady state, not a fault: count it
                // and keep going, with status OK.
                ++out.rejected_count;
                continue;
            }
            tutti::ObjectPlacement p;
            p.target = reinterpret_cast<const tutti::ResolvedTarget*>(this);
            p.offset = next_offset_;
            p.payload_bytes = config_.layout.payload_bytes();
            p.generation = ++generation_;
            next_offset_ += tutti::ObjectHeaderLayout::kHeaderBytes + p.payload_bytes;
            reserved_[k] = p;
            out.accepted.push_back(p);
            out.accepted_keys.push_back(keys[i]);
        }
        return tutti::Result<tutti::ReserveOutcome>::Success(std::move(out));
    }

    tutti::Status commit(const tutti::ObjectKey* keys,
                         std::size_t count) override {
        for (std::size_t i = 0; i < count; ++i) {
            const std::string k = hex(keys[i]);
            auto it = reserved_.find(k);
            if (it == reserved_.end()) continue;
            committed_[k] = it->second;
            reserved_.erase(it);
            all_rank_bits_.insert(k);
        }
        return {};
    }

    tutti::Status abort(const tutti::ObjectKey* keys,
                        std::size_t count) override {
        for (std::size_t i = 0; i < count; ++i) {
            // Committed keys are ignored: abort can never delete live data.
            reserved_.erase(hex(keys[i]));
        }
        return {};
    }

    tutti::Result<std::uint64_t> release(const tutti::ObjectKey* keys,
                                         std::size_t count) override {
        std::uint64_t released = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const std::string k = hex(keys[i]);
            if (pins_.count(k) && pins_[k] > 0) continue;  // pinned: leave alone
            if (committed_.erase(k)) {
                all_rank_bits_.erase(k);
                ++released;
            }
        }
        return tutti::Result<std::uint64_t>::Success(released);
    }

    tutti::Status pin(const tutti::ObjectKey* keys, std::size_t count) override {
        for (std::size_t i = 0; i < count; ++i) ++pins_[hex(keys[i])];
        return {};
    }

    tutti::Status unpin(const tutti::ObjectKey* keys,
                        std::size_t count) override {
        for (std::size_t i = 0; i < count; ++i) {
            auto it = pins_.find(hex(keys[i]));
            if (it != pins_.end() && it->second > 0) --it->second;
        }
        return {};
    }

    tutti::Result<std::vector<tutti::ObjectKey>> recover() override {
        std::vector<tutti::ObjectKey> keys;
        for (const auto& entry : committed_) keys.push_back(unhex(entry.first));
        return tutti::Result<std::vector<tutti::ObjectKey>>::Success(
            std::move(keys));
    }

    tutti::Status checkpoint() override { return {}; }

    // ---- test-only seams ----
    void set_bitmap_usable(bool usable) { bitmap_usable_ = usable; }
    void drop_bitmap_bit(const tutti::ObjectKey& key) {
        all_rank_bits_.erase(hex(key));
    }

private:
    static std::string hex(const tutti::ObjectKey& key) {
        std::string out;
        out.reserve(key.bytes.size() * 2);
        static const char* d = "0123456789abcdef";
        for (std::uint8_t b : key.bytes) {
            out.push_back(d[b >> 4]);
            out.push_back(d[b & 0xF]);
        }
        return out;
    }
    static tutti::ObjectKey unhex(const std::string& s) {
        tutti::ObjectKey key;
        for (std::size_t i = 0; i + 1 < s.size(); i += 2) {
            auto nib = [](char c) -> std::uint8_t {
                return static_cast<std::uint8_t>(c <= '9' ? c - '0' : c - 'a' + 10);
            };
            key.bytes.push_back(
                static_cast<std::uint8_t>((nib(s[i]) << 4) | nib(s[i + 1])));
        }
        return key;
    }

    tutti::StoreConfig config_;
    bool opened_ = false;
    bool bitmap_usable_ = true;
    std::uint64_t capacity_objects_ = 0;
    std::uint64_t next_offset_ = 0;
    std::uint64_t generation_ = 0;
    std::unordered_map<std::string, tutti::ObjectPlacement> reserved_;
    std::unordered_map<std::string, tutti::ObjectPlacement> committed_;
    std::unordered_map<std::string, int> pins_;
    // Keys believed committed on every rank: stands in for the AND of all
    // rank bitmaps. Kept separate from committed_ so a test can make it lag.
    std::set<std::string> all_rank_bits_;
};

// -------------------------------------------------------------------------
// Driver
// -------------------------------------------------------------------------
int g_failures = 0;

void check(bool cond, const char* expr, int line) {
    if (!cond) {
        std::printf("FAIL [line %d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

tutti::ObjectKey key_of(std::uint8_t tag) {
    tutti::ObjectKey k;
    k.bytes.assign(16, tag);
    return k;
}

tutti::StoreConfig deployed_config() {
    // Mirrors the deployed geometry: 80 layers x 128 KiB = 10 MiB per object.
    tutti::StoreConfig cfg;
    cfg.uri = "local_nvme_file:///mnt/nvme0/contract";
    cfg.layout.segment_bytes = 131072;
    cfg.layout.segment_count = 80;
    cfg.namespace_fingerprint = {1, 2, 3, 4};
    return cfg;
}

} // namespace

int main() {
    // ------------------------------------------------------------------
    // 1. Geometry derivation matches the deployed pool.
    // ------------------------------------------------------------------
    {
        tutti::StoreConfig cfg = deployed_config();
        CHECK(cfg.layout.payload_bytes() == 10ull * 1024 * 1024);
    }

    // ------------------------------------------------------------------
    // 2. capacity_bytes is the input; object count is derived.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 100ull * 1024 * 1024;  // exactly 10 objects
        CHECK(store.open(cfg).ok());
        CHECK(store.usage().capacity_bytes == 100ull * 1024 * 1024);
        CHECK(store.usage().usable_bytes == 100ull * 1024 * 1024);
    }

    // ------------------------------------------------------------------
    // 3. Incomplete layout is rejected.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg;
        cfg.capacity_bytes = 1 << 20;
        const tutti::Status st = store.open(cfg);
        CHECK(!st.ok());
        CHECK(st.code() == tutti::StatusCode::INVALID_ARGUMENT);
    }

    // ------------------------------------------------------------------
    // 4. Fingerprint mismatch is fail-closed (data preserved, no purge).
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 100ull * 1024 * 1024;
        CHECK(store.open(cfg).ok());
        const tutti::ObjectKey k = key_of(0xA1);
        CHECK(store.reserve(&k, 1).ok());
        CHECK(store.commit(&k, 1).ok());
        CHECK(store.contains(k));

        tutti::StoreConfig other = cfg;
        other.namespace_fingerprint = {9, 9, 9};
        const tutti::Status st = store.open(other);
        CHECK(!st.ok());
        CHECK(st.code() == tutti::StatusCode::INVALID_ARGUMENT);
        // The existing object survives the rejected open.
        CHECK(store.contains(k));
    }

    // ------------------------------------------------------------------
    // 5. reserve/commit are distinct events: a reserved-but-uncommitted
    //    object is invisible to lookup and contains.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 100ull * 1024 * 1024;
        CHECK(store.open(cfg).ok());
        const tutti::ObjectKey k = key_of(0xB2);
        auto res = store.reserve(&k, 1);
        CHECK(res.ok());
        CHECK(res.value().accepted.size() == 1);
        CHECK(res.value().rejected_count == 0);
        CHECK(!store.contains(k));               // not yet valid
        CHECK(!store.lookup(k).ok());
        CHECK(store.usage().reserved_bytes == 10ull * 1024 * 1024);
        CHECK(store.commit(&k, 1).ok());
        CHECK(store.contains(k));                // now valid
        CHECK(store.lookup(k).ok());
        CHECK(store.usage().reserved_bytes == 0);
        CHECK(store.usage().committed_bytes == 10ull * 1024 * 1024);
    }

    // ------------------------------------------------------------------
    // 6. Placement skips the header, so segment 0 is O_DIRECT aligned and
    //    consecutive objects do not overlap.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 100ull * 1024 * 1024;
        CHECK(store.open(cfg).ok());
        const tutti::ObjectKey a = key_of(0xC1);
        const tutti::ObjectKey b = key_of(0xC2);
        const tutti::ObjectKey keys[2] = {a, b};
        auto res = store.reserve(keys, 2);
        CHECK(res.ok());
        CHECK(res.value().accepted.size() == 2);
        const auto& pa = res.value().accepted[0];
        const auto& pb = res.value().accepted[1];
        CHECK(pa.payload_bytes == 10ull * 1024 * 1024);
        CHECK(pb.offset >= pa.offset + pa.payload_bytes);
        CHECK((pb.offset - pa.offset) %
                  tutti::ObjectHeaderLayout::kHeaderBytes == 0);
        // Distinct space generations: enables ABA detection.
        CHECK(pa.generation != pb.generation);
    }

    // ------------------------------------------------------------------
    // 7. Capacity exhaustion is partial acceptance with status OK --
    //    never an error, never a block. This is the invariant that keeps a
    //    full cache from failing the request that produced the miss.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 20ull * 1024 * 1024;  // room for exactly 2
        CHECK(store.open(cfg).ok());
        const tutti::ObjectKey keys[5] = {key_of(1), key_of(2), key_of(3),
                                          key_of(4), key_of(5)};
        auto res = store.reserve(keys, 5);
        CHECK(res.ok());                                  // status OK
        CHECK(res.value().accepted.size() == 2);          // took what it could
        CHECK(res.value().accepted_keys.size() == 2);     // parallel arrays
        CHECK(res.value().rejected_count == 3);           // reported remainder
        CHECK(store.usage().usable_bytes == 0);
    }

    // ------------------------------------------------------------------
    // 8. Re-reserving a committed key returns its existing placement
    //    instead of consuming new space.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 20ull * 1024 * 1024;
        CHECK(store.open(cfg).ok());
        const tutti::ObjectKey k = key_of(0xD1);
        auto first = store.reserve(&k, 1);
        CHECK(first.ok());
        const std::uint64_t offset = first.value().accepted[0].offset;
        CHECK(store.commit(&k, 1).ok());
        auto again = store.reserve(&k, 1);
        CHECK(again.ok());
        CHECK(again.value().rejected_count == 0);
        CHECK(again.value().accepted.size() == 1);
        CHECK(again.value().accepted[0].offset == offset);
        CHECK(store.usage().committed_bytes == 10ull * 1024 * 1024);
    }

    // ------------------------------------------------------------------
    // 9. abort discards reservations but never touches committed data.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 100ull * 1024 * 1024;
        CHECK(store.open(cfg).ok());
        const tutti::ObjectKey live = key_of(0xE1);
        const tutti::ObjectKey dead = key_of(0xE2);
        const tutti::ObjectKey both[2] = {live, dead};
        CHECK(store.reserve(both, 2).ok());
        CHECK(store.commit(&live, 1).ok());
        CHECK(store.abort(both, 2).ok());
        CHECK(store.contains(live));       // committed survives abort
        CHECK(!store.contains(dead));
        CHECK(store.usage().reserved_bytes == 0);
    }

    // ------------------------------------------------------------------
    // 10. pin blocks release; unpin restores it.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 100ull * 1024 * 1024;
        CHECK(store.open(cfg).ok());
        const tutti::ObjectKey k = key_of(0xF1);
        CHECK(store.reserve(&k, 1).ok());
        CHECK(store.commit(&k, 1).ok());
        CHECK(store.pin(&k, 1).ok());
        auto blocked = store.release(&k, 1);
        CHECK(blocked.ok());
        CHECK(blocked.value() == 0);       // pinned: not released
        CHECK(store.contains(k));
        CHECK(store.unpin(&k, 1).ok());
        auto freed = store.release(&k, 1);
        CHECK(freed.ok());
        CHECK(freed.value() == 1);
        CHECK(!store.contains(k));
    }

    // ------------------------------------------------------------------
    // 11. contains_prefix stops at the first gap.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 100ull * 1024 * 1024;
        CHECK(store.open(cfg).ok());
        const tutti::ObjectKey keys[4] = {key_of(1), key_of(2), key_of(3),
                                          key_of(4)};
        const tutti::ObjectKey present[2] = {keys[0], keys[1]};
        CHECK(store.reserve(present, 2).ok());
        CHECK(store.commit(present, 2).ok());
        // keys[3] committed but keys[2] missing: prefix must stop at 2.
        CHECK(store.reserve(&keys[3], 1).ok());
        CHECK(store.commit(&keys[3], 1).ok());
        CHECK(store.contains_prefix(keys, 4) == 2);
    }

    // ------------------------------------------------------------------
    // 12. The cross-rank residency query may only UNDER-report.
    //
    //     Over-reporting would load truncated KV and corrupt results
    //     silently; under-reporting merely costs a recomputation. Both a
    //     lagging bit and an entirely unusable bitmap must stay on the safe
    //     side.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 100ull * 1024 * 1024;
        cfg.rank_count = 8;
        CHECK(store.open(cfg).ok());
        const tutti::ObjectKey keys[3] = {key_of(1), key_of(2), key_of(3)};
        CHECK(store.reserve(keys, 3).ok());
        CHECK(store.commit(keys, 3).ok());
        CHECK(store.contains_prefix_all_ranks(keys, 3) == 3);

        // A lagging bitmap bit under-reports; it must never exceed the
        // single-rank answer.
        store.drop_bitmap_bit(keys[1]);
        const std::uint64_t lagging = store.contains_prefix_all_ranks(keys, 3);
        CHECK(lagging == 1);
        CHECK(lagging <= store.contains_prefix(keys, 3));

        // An unusable bitmap degenerates to this rank's own view, which is
        // still safe (it can only be a superset of what all ranks hold when
        // this rank is the only writer -- the implementation must not claim
        // cross-rank knowledge it does not have).
        store.set_bitmap_usable(false);
        CHECK(store.contains_prefix_all_ranks(keys, 3) ==
              store.contains_prefix(keys, 3));
    }

    // ------------------------------------------------------------------
    // 13. Single-rank config never consults a bitmap.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 100ull * 1024 * 1024;
        cfg.rank_count = 1;
        CHECK(store.open(cfg).ok());
        const tutti::ObjectKey k = key_of(0x11);
        CHECK(store.reserve(&k, 1).ok());
        CHECK(store.commit(&k, 1).ok());
        store.drop_bitmap_bit(k);   // irrelevant when rank_count == 1
        CHECK(store.contains_prefix_all_ranks(&k, 1) == 1);
    }

    // ------------------------------------------------------------------
    // 14. recover enumerates exactly the committed set.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 100ull * 1024 * 1024;
        CHECK(store.open(cfg).ok());
        const tutti::ObjectKey keys[3] = {key_of(1), key_of(2), key_of(3)};
        CHECK(store.reserve(keys, 3).ok());
        CHECK(store.commit(keys, 2).ok());     // third stays reserved
        auto recovered = store.recover();
        CHECK(recovered.ok());
        CHECK(recovered.value().size() == 2);  // uncommitted is not recovered
    }

    // ------------------------------------------------------------------
    // 15. Empty batches are legal no-ops on every batched entry point.
    // ------------------------------------------------------------------
    {
        FakeStore store;
        tutti::StoreConfig cfg = deployed_config();
        cfg.capacity_bytes = 100ull * 1024 * 1024;
        CHECK(store.open(cfg).ok());
        auto res = store.reserve(nullptr, 0);
        CHECK(res.ok());
        CHECK(res.value().accepted.empty());
        CHECK(res.value().rejected_count == 0);
        CHECK(store.commit(nullptr, 0).ok());
        CHECK(store.abort(nullptr, 0).ok());
        CHECK(store.pin(nullptr, 0).ok());
        CHECK(store.unpin(nullptr, 0).ok());
        auto rel = store.release(nullptr, 0);
        CHECK(rel.ok());
        CHECK(rel.value() == 0);
        CHECK(store.contains_prefix(nullptr, 0) == 0);
        CHECK(store.contains_prefix_all_ranks(nullptr, 0) == 0);
    }

    if (g_failures == 0) {
        std::printf("storage_object_store contract: all checks passed\n");
        return 0;
    }
    std::printf("storage_object_store contract: %d failure(s)\n", g_failures);
    return 1;
}

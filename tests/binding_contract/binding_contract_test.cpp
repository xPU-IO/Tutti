// binding_contract_test.cpp
//
// Contract tests for tutti/bindings/ext4_local_nvme/binding.h.
// Plain C++17 executable, no GTest or third-party deps.

#include <csrc/bindings/ext4_local_nvme/binding.h>

#include <tutti/spi/storage_target_resolver.h>  // for mismatch test only

#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace binding = tutti::binding::ext4_local_nvme;

// =====================================================================
// Fake types
// =====================================================================

struct FakeLease {
    static std::atomic<int>& destroy_count() {
        static std::atomic<int> c{0};
        return c;
    }
    static void reset_count() { destroy_count() = 0; }

    std::string description;
    ~FakeLease() { ++destroy_count(); }
};

class FakeResolver : public tutti::StorageTargetResolver {
public:
    tutti::Result<tutti::ResolvedTarget> resolve(
        std::string_view uri,
        const tutti::ResolveOptions& /*opts*/) override {

        auto payload = binding::Ext4LocalNvmePayload::create(
            binding::NamespaceIdentity{"0000:08:00.0", 1, 4096},
            std::vector<binding::Extent>{
                {0,     0x10000, 0x1000},
                {0x1000, 0x20000, 0x1000},
                {0x2000, 0x30000, 0x1000},
            },
            0x3000);
        if (!payload.ok()) return tutti::Result<tutti::ResolvedTarget>::Failure(
            payload.status());

        auto lease = std::make_shared<FakeLease>();
        lease->description = std::string(uri);

        return binding::make_resolved_target(
            std::string(binding::kResolverTypeId),
            0x3000,
            std::move(payload).value(),
            std::move(lease));
    }
};

class FakeDataPathConsumer {
public:
    explicit FakeDataPathConsumer(const tutti::ResolvedTarget& target)
        : target_(target) {}

    tutti::Result<const binding::Ext4LocalNvmePayload*> get_payload() const {
        return binding::view_payload(target_);
    }

private:
    const tutti::ResolvedTarget& target_;
};

// =====================================================================
// Helpers
// =====================================================================

static int run_test(int idx, int (*fn)()) {
    int rc = fn();
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: test %d returned %d\n", idx, rc);
    }
    return rc;
}

// =====================================================================
// Tests
// =====================================================================

// 1. Normal pairing: resolver → consumer view_payload succeeds.
static int test_normal_pairing() {
    FakeResolver resolver;
    auto result = resolver.resolve("fake://test.txt", {"fake"});
    if (!result.ok()) return 1;
    if (!result.has_value()) return 1;
    if (!result.value().valid()) return 1;

    FakeDataPathConsumer consumer(result.value());
    auto pr = consumer.get_payload();
    if (!pr.ok()) return 1;
    if (!pr.has_value()) return 1;
    if (pr.value() == nullptr) return 1;

    const auto* p = pr.value();
    if (p->namespace_identity().controller_pci_addr != "0000:08:00.0") return 1;
    if (p->namespace_identity().namespace_id != 1) return 1;
    if (p->namespace_identity().block_size != 4096) return 1;
    if (p->file_size() != 0x3000) return 1;
    if (p->extents().size() != 3) return 1;
    return 0;
}

// 2. Payload type mismatch → UNSUPPORTED.
static int test_payload_type_mismatch() {
    auto payload = std::make_shared<int>(42);
    auto lease = std::make_shared<FakeLease>();

    auto wrong = tutti::ResolvedTarget::make<int, FakeLease>(
        "wrong-resolver",
        "wrong-payload-type-id",
        binding::kPayloadApiVersion,
        100,
        "some-key",
        std::move(payload),
        std::move(lease));
    if (!wrong.ok()) return 1;

    auto v = binding::view_payload(wrong.value());
    if (v.ok()) return 1;
    if (v.status().code() != tutti::StatusCode::UNSUPPORTED) return 1;
    return 0;
}

// 3. API version mismatch → UNSUPPORTED.
static int test_version_mismatch() {
    auto payload = binding::Ext4LocalNvmePayload::create(
        binding::NamespaceIdentity{"00:00.0", 1, 512},
        std::vector<binding::Extent>{{0, 0, 512}},
        512);
    if (!payload.ok()) return 1;
    auto lease = std::make_shared<FakeLease>();

    auto wrong = tutti::ResolvedTarget::make<
        binding::Ext4LocalNvmePayload, FakeLease>(
        "wrong-resolver",
        std::string(binding::kPayloadTypeId),
        999,
        512,
        "some-key",
        std::const_pointer_cast<binding::Ext4LocalNvmePayload>(
            std::move(payload).value()),
        std::move(lease));
    if (!wrong.ok()) return 1;

    auto v = binding::view_payload(wrong.value());
    if (v.ok()) return 1;
    if (v.status().code() != tutti::StatusCode::UNSUPPORTED) return 1;
    return 0;
}

// 4. Empty target → UNSUPPORTED, valid() == false.
static int test_empty_target() {
    tutti::ResolvedTarget empty;
    if (empty.valid()) return 1;

    auto v = binding::view_payload(empty);
    if (v.ok()) return 1;
    if (v.status().code() != tutti::StatusCode::UNSUPPORTED) return 1;
    return 0;
}

// 5. Payload owner lifetime: weak_ptr + destruction count.
static int test_owner_lifetime() {
    FakeLease::reset_count();

    std::weak_ptr<const binding::Ext4LocalNvmePayload> weak_payload;
    std::weak_ptr<FakeLease> weak_lease;

    {
        auto payload = binding::Ext4LocalNvmePayload::create(
            binding::NamespaceIdentity{"00:00.0", 1, 512},
            std::vector<binding::Extent>{{0, 0, 512}},
            512);
        if (!payload.ok()) return 1;

        auto lease = std::make_shared<FakeLease>();

        weak_payload = std::weak_ptr<const binding::Ext4LocalNvmePayload>(
            payload.value());
        weak_lease = lease;

        if (weak_payload.expired()) return 1;
        if (weak_lease.expired()) return 1;

        auto rt_result = binding::make_resolved_target(
            "test-resolver", 512,
            std::move(payload).value(), std::move(lease));
        if (!rt_result.ok()) return 1;

        auto rt = std::move(rt_result).value();

        // Local shared_ptrs released → weak pointers still valid.
        if (weak_payload.expired()) return 1;
        if (weak_lease.expired()) return 1;

        auto v = binding::view_payload(rt);
        if (!v.ok()) return 1;

        // rt destroyed here → both released.
    }

    if (!weak_payload.expired()) return 1;
    if (!weak_lease.expired()) return 1;

    // Lease destroyed exactly once.
    if (FakeLease::destroy_count() != 1) return 1;
    return 0;
}

// 6. Move semantics: moved-to retains owner, moved-from invalid.
static int test_move_retains_owner() {
    FakeResolver resolver;
    auto result = resolver.resolve("fake://move.txt", {"fake"});
    if (!result.ok()) return 1;

    tutti::ResolvedTarget original = std::move(result).value();
    tutti::ResolvedTarget moved_to = std::move(original);

    if (!moved_to.valid()) return 1;
    if (original.valid()) return 1;

    auto v = binding::view_payload(moved_to);
    if (!v.ok()) return 1;
    if (v.value() == nullptr) return 1;
    if (v.value()->file_size() != 0x3000) return 1;

    auto v2 = binding::view_payload(original);
    if (v2.ok()) return 1;
    return 0;
}

// 7. Immutable: view_payload returns const pointer.
static int test_immutable() {
    FakeResolver resolver;
    auto result = resolver.resolve("fake://imm.txt", {"fake"});
    if (!result.ok()) return 1;

    auto v = binding::view_payload(result.value());
    if (!v.ok()) return 1;

    // Static assertion: the returned pointer type is const.
    static_assert(
        std::is_same_v<
            std::remove_reference_t<decltype(v.value())>,
            const binding::Ext4LocalNvmePayload*>);

    (void)v;
    return 0;
}

// 8. Extent mapping: first, cross-boundary, last.
static int test_extent_mapping() {
    FakeResolver resolver;
    auto result = resolver.resolve("fake://map.txt", {"fake"});
    if (!result.ok()) return 1;

    FakeDataPathConsumer consumer(result.value());
    auto pr = consumer.get_payload();
    if (!pr.ok()) return 1;
    const auto* p = pr.value();

    // First segment interior.
    auto r1 = p->map_to_device_offset(0x100);
    if (!r1.ok()) return 1;
    if (r1.value() != 0x10100) return 1;

    // First segment last byte.
    auto r2 = p->map_to_device_offset(0x0FFF);
    if (!r2.ok()) return 1;
    if (r2.value() != 0x10FFF) return 1;

    // Cross-boundary: first byte of second segment.
    auto r3 = p->map_to_device_offset(0x1000);
    if (!r3.ok()) return 1;
    if (r3.value() != 0x20000) return 1;

    // Last segment interior.
    auto r4 = p->map_to_device_offset(0x2500);
    if (!r4.ok()) return 1;
    if (r4.value() != 0x30500) return 1;

    // Last byte of file.
    auto r5 = p->map_to_device_offset(0x2FFF);
    if (!r5.ok()) return 1;
    if (r5.value() != 0x30FFF) return 1;
    return 0;
}

// 9. Out-of-range mapping.
static int test_mapping_out_of_range() {
    FakeResolver resolver;
    auto result = resolver.resolve("fake://oor.txt", {"fake"});
    if (!result.ok()) return 1;

    auto pr = binding::view_payload(result.value());
    if (!pr.ok()) return 1;
    const auto* p = pr.value();

    auto r1 = p->map_to_device_offset(p->file_size());
    if (r1.ok()) return 1;
    if (r1.status().code() != tutti::StatusCode::OUT_OF_RANGE) return 1;

    auto r2 = p->map_to_device_offset(p->file_size() + 1);
    if (r2.ok()) return 1;
    if (r2.status().code() != tutti::StatusCode::OUT_OF_RANGE) return 1;
    return 0;
}

// 10. validate() accepts legal extent set.
static int test_validate_accepts_legal() {
    auto payload = binding::Ext4LocalNvmePayload::create(
        binding::NamespaceIdentity{"00:00.0", 1, 512},
        std::vector<binding::Extent>{
            {0,     0,     512},
            {512,   1024,  512},
            {1024,  2048,  512},
        },
        1536);
    if (!payload.ok()) return 1;
    return 0;
}

// 11. validate() rejects illegal sets.
static int test_validate_rejects_illegal() {
    using namespace binding;

    // (a) Hole.
    {
        NamespaceIdentity ns{"00:00.0", 1, 512};
        std::vector<Extent> exts{
            {0,   0,    256},
            {512, 1024, 256},
        };
        auto r = Ext4LocalNvmePayload::create(
            std::move(ns), std::move(exts), 768);
        if (r.ok()) return 1;
        if (r.status().code() != tutti::StatusCode::DATA_LOSS) return 1;
    }

    // (b) Overlap.
    {
        NamespaceIdentity ns{"00:00.0", 1, 512};
        std::vector<Extent> exts{
            {0,   0,    512},
            {256, 1024, 256},
        };
        auto r = Ext4LocalNvmePayload::create(
            std::move(ns), std::move(exts), 512);
        if (r.ok()) return 1;
        if (r.status().code() != tutti::StatusCode::DATA_LOSS) return 1;
    }

    // (c) Incomplete coverage.
    {
        NamespaceIdentity ns{"00:00.0", 1, 512};
        std::vector<Extent> exts{
            {0,   0,    256},
        };
        auto r = Ext4LocalNvmePayload::create(
            std::move(ns), std::move(exts), 512);
        if (r.ok()) return 1;
        if (r.status().code() != tutti::StatusCode::DATA_LOSS) return 1;
    }

    // (d) Unsorted.
    {
        NamespaceIdentity ns{"00:00.0", 1, 512};
        std::vector<Extent> exts{
            {512, 1024, 256},
            {0,   0,    512},
        };
        auto r = Ext4LocalNvmePayload::create(
            std::move(ns), std::move(exts), 768);
        if (r.ok()) return 1;
        if (r.status().code() != tutti::StatusCode::DATA_LOSS) return 1;
    }

    return 0;
}

// 12. Pairing convergence: no bare type id literals in test code.
static int test_pairing_convergence() {
    static_assert(binding::kPayloadTypeId.size() > 0);
    static_assert(binding::kPayloadApiVersion > 0);
    static_assert(binding::kRecommendedDataPathKey.size() > 0);

    FakeResolver resolver;
    auto result = resolver.resolve("fake://conv.txt", {"fake"});
    if (!result.ok()) return 1;

    auto v = binding::view_payload(result.value());
    if (!v.ok()) return 1;

    // The recommended_data_path_key on the target matches the binding
    // constant — resolver set it via make_resolved_target which uses
    // kRecommendedDataPathKey internally.
    if (result.value().recommended_data_path_key()
        != binding::kRecommendedDataPathKey) return 1;

    // The payload_type_id on the target matches the binding constant.
    if (result.value().payload_type_id()
        != binding::kPayloadTypeId) return 1;

    // The source_api_version on the target matches the binding constant.
    if (result.value().source_api_version()
        != binding::kPayloadApiVersion) return 1;

    return 0;
}

// =====================================================================
// Main
// =====================================================================

int main() {
    using TestFn = int (*)();
    const TestFn tests[] = {
        test_normal_pairing,           // 1
        test_payload_type_mismatch,    // 2
        test_version_mismatch,         // 3
        test_empty_target,             // 4
        test_owner_lifetime,           // 5
        test_move_retains_owner,       // 6
        test_immutable,                // 7
        test_extent_mapping,           // 8
        test_mapping_out_of_range,    // 9
        test_validate_accepts_legal,   // 10
        test_validate_rejects_illegal, // 11
        test_pairing_convergence,      // 12
    };

    const int n = static_cast<int>(sizeof(tests) / sizeof(tests[0]));
    for (int i = 0; i < n; ++i) {
        if (run_test(i, tests[i]) != 0) {
            return 1;
        }
    }

    std::printf("All %d binding contract tests passed.\n", n);
    return 0;
}

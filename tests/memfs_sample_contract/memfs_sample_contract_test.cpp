// memfs_sample_contract_test.cpp
//
// SAMPLE-ONLY contract test for the memfs extension.
// Plain C++17 executable, no GTest or third-party deps.
// Returns 0 on full pass, non-zero on any failure.
//
// Tests:
//   1. URI parsing (valid + invalid)
//   2. E2E via StorageRuntime: open → register → submit(WRITE) → wait →
//      submit(READ) → wait → verify data → release → close → shutdown
//   3. Boundary rejection (offset + length > logical_size)
//   4. Lease lifecycle (target invalid after close)

#include <tutti/storage_runtime.h>
#include "csrc/bindings/memfs/binding.h"
#include "csrc/bindings/memfs/memfs_data_path.h"
#include "csrc/resolvers/memfs/resolver.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// =====================================================================
// Helpers
// =====================================================================

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
        ++g_fail; \
        return 1; \
    } \
} while (0)

#define CHECK_STATUS(ok_expr) do { \
    if (!(ok_expr).ok()) { \
        std::fprintf(stderr, "FAIL [%s:%d]: status %d: %s\n", __FILE__, __LINE__, \
                     static_cast<int>((ok_expr).status().code()), \
                     (ok_expr).status().message().c_str()); \
        ++g_fail; \
        return 1; \
    } \
} while (0)

static int run_test(const char* name, int (*fn)()) {
    int rc = fn();
    if (rc == 0) {
        std::printf("PASS: %s\n", name);
        ++g_pass;
    } else {
        std::fprintf(stderr, "FAIL: %s\n", name);
    }
    return rc;
}

static tutti::MemoryView make_host_view(void* addr, std::uint64_t sz) {
    return tutti::MemoryView{
        addr, sz, tutti::MemoryKind::HOST,
        tutti::MemoryOwnership::CALLER_OWNED, -1, ""};
}

static tutti::HostSubmitContext host_ctx() {
    return tutti::HostSubmitContext{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};
}

// =====================================================================
// Test 1: URI parsing
// =====================================================================

static int test_uri_parsing() {
    using namespace tutti::resolver::memfs;

    MemfsResolver resolver;

    // Valid: memfs://4096
    {
        auto r = resolver.resolve("memfs://4096", {"memfs"});
        CHECK_STATUS(r);
        CHECK(r.value().valid());
        CHECK(r.value().logical_size() == 4096);
        CHECK(r.value().recommended_data_path_key() == "memfs");
    }

    // Valid: large size
    {
        auto r = resolver.resolve("memfs://1048576", {"memfs"});
        CHECK_STATUS(r);
        CHECK(r.value().logical_size() == 1048576);
    }

    // Invalid: missing size
    {
        auto r = resolver.resolve("memfs://", {"memfs"});
        CHECK(!r.ok());
        CHECK(r.status().code() == tutti::StatusCode::INVALID_ARGUMENT);
    }

    // Invalid: zero size
    {
        auto r = resolver.resolve("memfs://0", {"memfs"});
        CHECK(!r.ok());
        CHECK(r.status().code() == tutti::StatusCode::INVALID_ARGUMENT);
    }

    // Invalid: non-numeric
    {
        auto r = resolver.resolve("memfs://abc", {"memfs"});
        CHECK(!r.ok());
        CHECK(r.status().code() == tutti::StatusCode::INVALID_ARGUMENT);
    }

    // Invalid: wrong scheme
    {
        auto r = resolver.resolve("file:///tmp/x", {"memfs"});
        CHECK(!r.ok());
    }

    return 0;
}

// =====================================================================
// Test 2: E2E via StorageRuntime
// =====================================================================

static int test_e2e_roundtrip() {
    using namespace tutti::binding::memfs;
    using namespace tutti::resolver::memfs;

    MemfsResolver resolver;
    MemfsDataPath data_path;

    tutti::RuntimeComponents components;
    components.resolvers.push_back({"memfs", &resolver});
    components.data_paths.push_back({"memfs", &data_path, {}});

    auto rt_result = tutti::StorageRuntime::create(
        tutti::RuntimeConfig{}, std::move(components));
    CHECK_STATUS(rt_result);
    auto rt = std::move(rt_result).value();

    // Open target: memfs://4096
    auto open_result = rt->open("memfs://4096", {"memfs"});
    CHECK_STATUS(open_result);
    auto target = open_result.value();

    // Verify target info
    auto info = rt->query_target(target);
    CHECK_STATUS(info);
    CHECK(info.value().uri == "memfs://4096");
    CHECK(info.value().logical_size == 4096);

    // Register caller-owned host memory (8 KiB for read + write buffers)
    std::vector<std::uint8_t> write_buf(4096);
    std::vector<std::uint8_t> read_buf(4096);

    // Fill write buffer with a known pattern
    for (std::size_t i = 0; i < write_buf.size(); ++i) {
        write_buf[i] = static_cast<std::uint8_t>(0xA5 ^ (i & 0xFF));
    }

    auto mem_write = rt->register_memory(
        make_host_view(write_buf.data(), write_buf.size()));
    CHECK_STATUS(mem_write);

    auto mem_read = rt->register_memory(
        make_host_view(read_buf.data(), read_buf.size()));
    CHECK_STATUS(mem_read);

    // Submit WRITE: write_buf → target at offset 0
    {
        tutti::IoRequest req{
            tutti::IoDirection::WRITE,
            mem_write.value(),
            0,        // memory_offset
            target,
            0,        // target_offset
            4096,     // length
        };
        auto submit = rt->submit(&req, 1, host_ctx());
        CHECK(submit.status.ok());
        CHECK(submit.io.has_value());
        CHECK(submit.initial_states[0].state ==
              tutti::IoRequestState::ACCEPTED);

        auto wait = rt->wait(submit.io.value(), 5000);
        CHECK(wait.observation_status.ok());
        CHECK(wait.result.has_value());
        CHECK(wait.result->state == tutti::IoState::COMPLETED);

        CHECK(rt->release_io(submit.io.value()).ok());
    }

    // Submit READ: target at offset 0 → read_buf
    {
        // Pre-fill read_buf with zeros to ensure we actually read back data
        std::memset(read_buf.data(), 0, read_buf.size());

        tutti::IoRequest req{
            tutti::IoDirection::READ,
            mem_read.value(),
            0,        // memory_offset
            target,
            0,        // target_offset
            4096,     // length
        };
        auto submit = rt->submit(&req, 1, host_ctx());
        CHECK(submit.status.ok());
        CHECK(submit.io.has_value());

        auto wait = rt->wait(submit.io.value(), 5000);
        CHECK(wait.observation_status.ok());
        CHECK(wait.result->state == tutti::IoState::COMPLETED);

        CHECK(rt->release_io(submit.io.value()).ok());
    }

    // Verify data integrity
    CHECK(std::memcmp(write_buf.data(), read_buf.data(), 4096) == 0);

    // Cleanup
    CHECK(rt->unregister_memory(mem_read.value()).ok());
    CHECK(rt->unregister_memory(mem_write.value()).ok());
    CHECK(rt->close(target).ok());
    CHECK(rt->shutdown(1000).ok());

    return 0;
}

// =====================================================================
// Test 3: Boundary rejection
// =====================================================================

static int test_boundary_rejection() {
    using namespace tutti::binding::memfs;
    using namespace tutti::resolver::memfs;

    MemfsResolver resolver;
    MemfsDataPath data_path;

    tutti::RuntimeComponents components;
    components.resolvers.push_back({"memfs", &resolver});
    components.data_paths.push_back({"memfs", &data_path, {}});

    auto rt_result = tutti::StorageRuntime::create(
        tutti::RuntimeConfig{}, std::move(components));
    CHECK_STATUS(rt_result);
    auto rt = std::move(rt_result).value();

    auto open_result = rt->open("memfs://4096", {"memfs"});
    CHECK_STATUS(open_result);
    auto target = open_result.value();

    std::vector<std::uint8_t> buf(8192);
    auto mem = rt->register_memory(
        make_host_view(buf.data(), buf.size()));
    CHECK_STATUS(mem);

    // offset + length > logical_size (4096)
    {
        tutti::IoRequest req{
            tutti::IoDirection::WRITE,
            mem.value(),
            0,
            target,
            4096,     // target_offset: exactly at end
            1,        // length: 1 byte past end
        };
        auto submit = rt->submit(&req, 1, host_ctx());
        // Runtime validates bounds before DataPath submit
        CHECK(submit.initial_states[0].state ==
              tutti::IoRequestState::REJECTED);
        CHECK(!submit.io.has_value());
    }

    // length == 0
    {
        tutti::IoRequest req{
            tutti::IoDirection::READ,
            mem.value(),
            0,
            target,
            0,
            0,        // zero length
        };
        auto submit = rt->submit(&req, 1, host_ctx());
        CHECK(submit.initial_states[0].state ==
              tutti::IoRequestState::REJECTED);
    }

    // memory_offset + length > memory_size
    {
        tutti::IoRequest req{
            tutti::IoDirection::WRITE,
            mem.value(),
            8192,     // memory_offset: at end of 8192-byte buffer
            target,
            0,
            1,        // 1 byte past memory
        };
        auto submit = rt->submit(&req, 1, host_ctx());
        CHECK(submit.initial_states[0].state ==
              tutti::IoRequestState::REJECTED);
    }

    // Cleanup
    CHECK(rt->unregister_memory(mem.value()).ok());
    CHECK(rt->close(target).ok());
    CHECK(rt->shutdown(1000).ok());

    return 0;
}

// =====================================================================
// Test 4: Lease lifecycle (target invalid after close)
// =====================================================================

static int test_lease_lifecycle() {
    using namespace tutti::binding::memfs;
    using namespace tutti::resolver::memfs;

    MemfsResolver resolver;
    MemfsDataPath data_path;

    tutti::RuntimeComponents components;
    components.resolvers.push_back({"memfs", &resolver});
    components.data_paths.push_back({"memfs", &data_path, {}});

    auto rt_result = tutti::StorageRuntime::create(
        tutti::RuntimeConfig{}, std::move(components));
    CHECK_STATUS(rt_result);
    auto rt = std::move(rt_result).value();

    // Open and close
    auto open_result = rt->open("memfs://4096", {"memfs"});
    CHECK_STATUS(open_result);
    auto target = open_result.value();

    // Query before close — should succeed
    auto info = rt->query_target(target);
    CHECK_STATUS(info);

    // Close
    CHECK(rt->close(target).ok());

    // Query after close — should fail (NOT_FOUND)
    auto info2 = rt->query_target(target);
    CHECK(!info2.ok());

    // Re-open same URI — should succeed (new handle, new generation)
    auto open2 = rt->open("memfs://4096", {"memfs"});
    CHECK_STATUS(open2);

    CHECK(rt->close(open2.value()).ok());
    CHECK(rt->shutdown(1000).ok());

    return 0;
}

// =====================================================================
// Test 5: Partial submit (mix of valid + invalid in one batch)
// =====================================================================

static int test_partial_submit() {
    using namespace tutti::binding::memfs;
    using namespace tutti::resolver::memfs;

    MemfsResolver resolver;
    MemfsDataPath data_path;

    tutti::RuntimeComponents components;
    components.resolvers.push_back({"memfs", &resolver});
    components.data_paths.push_back({"memfs", &data_path, {}});

    auto rt_result = tutti::StorageRuntime::create(
        tutti::RuntimeConfig{}, std::move(components));
    CHECK_STATUS(rt_result);
    auto rt = std::move(rt_result).value();

    auto open_result = rt->open("memfs://4096", {"memfs"});
    CHECK_STATUS(open_result);
    auto target = open_result.value();

    std::vector<std::uint8_t> buf(8192);
    auto mem = rt->register_memory(
        make_host_view(buf.data(), buf.size()));
    CHECK_STATUS(mem);

    // Two requests: first valid, second out-of-bounds
    tutti::IoRequest reqs[2] = {
        {tutti::IoDirection::WRITE, mem.value(), 0, target, 0, 100},
        {tutti::IoDirection::WRITE, mem.value(), 0, target, 4096, 100},
    };
    auto submit = rt->submit(reqs, 2, host_ctx());

    // Should be partial: first ACCEPTED, second REJECTED
    CHECK(submit.io.has_value());  // at least one accepted
    CHECK(submit.initial_states[0].state == tutti::IoRequestState::ACCEPTED);
    CHECK(submit.initial_states[1].state == tutti::IoRequestState::REJECTED);

    // Wait for the accepted one
    auto wait = rt->wait(submit.io.value(), 5000);
    CHECK(wait.observation_status.ok());
    CHECK(wait.result->state == tutti::IoState::COMPLETED);
    CHECK(rt->release_io(submit.io.value()).ok());

    CHECK(rt->unregister_memory(mem.value()).ok());
    CHECK(rt->close(target).ok());
    CHECK(rt->shutdown(1000).ok());

    return 0;
}

// =====================================================================
// Main
// =====================================================================

int main() {
    std::printf("=== memfs sample contract tests ===\n");

    run_test("uri_parsing", test_uri_parsing);
    run_test("e2e_roundtrip", test_e2e_roundtrip);
    run_test("boundary_rejection", test_boundary_rejection);
    run_test("lease_lifecycle", test_lease_lifecycle);
    run_test("partial_submit", test_partial_submit);

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

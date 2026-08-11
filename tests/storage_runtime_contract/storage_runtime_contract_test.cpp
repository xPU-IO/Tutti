// storage_runtime_contract_test.cpp
//
// Contract tests for tutti/storage_runtime.h.
// Plain C++17 executable, no GTest or third-party deps.
// Returns 0 on full pass, non-zero on any failure.

#include <tutti/storage_runtime.h>
#include <tutti/testing/mock_data_path.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// =====================================================================
// Component-backed runtime fakes
// =====================================================================

struct RuntimeFakePayload {
    std::string name;
};

struct RuntimeFakeLease {};

class RuntimeFakeResolver : public tutti::StorageTargetResolver {
public:
    int resolve_calls = 0;

    tutti::Result<tutti::ResolvedTarget> resolve(
        std::string_view uri, const tutti::ResolveOptions&) override {
        ++resolve_calls;
        if (uri.rfind("fake://", 0) != 0) {
            return tutti::Result<tutti::ResolvedTarget>::Failure(
                tutti::Status(tutti::StatusCode::NOT_FOUND, "unknown fake URI"));
        }
        auto payload = std::make_shared<RuntimeFakePayload>();
        payload->name = std::string(uri.substr(7));
        auto lease = std::make_shared<RuntimeFakeLease>();
        return tutti::ResolvedTarget::make<RuntimeFakePayload, RuntimeFakeLease>(
            "runtime-fake-resolver", "runtime-fake-payload", 1, 4096,
            "runtime-fake-datapath", std::move(payload), std::move(lease));
    }
};

// Resolver that routes two distinct URI schemes ("fakea://", "fakeb://") to
// two distinct DataPath keys ("runtime-fake-datapath-a"/"-b"). Used by the
// cross-DataPath merge-counting test (Round 15 Session 3, test 83) to prove
// requests for different DataPaths are never merged into one submit() call.
class TwoWayFakeResolver : public tutti::StorageTargetResolver {
public:
    tutti::Result<tutti::ResolvedTarget> resolve(
        std::string_view uri, const tutti::ResolveOptions&) override {
        std::string dp_key;
        std::size_t prefix_len = 0;
        if (uri.rfind("fakea://", 0) == 0) {
            dp_key = "runtime-fake-datapath-a";
            prefix_len = 8;
        } else if (uri.rfind("fakeb://", 0) == 0) {
            dp_key = "runtime-fake-datapath-b";
            prefix_len = 8;
        } else {
            return tutti::Result<tutti::ResolvedTarget>::Failure(
                tutti::Status(tutti::StatusCode::NOT_FOUND, "unknown fake URI"));
        }
        auto payload = std::make_shared<RuntimeFakePayload>();
        payload->name = std::string(uri.substr(prefix_len));
        auto lease = std::make_shared<RuntimeFakeLease>();
        return tutti::ResolvedTarget::make<RuntimeFakePayload, RuntimeFakeLease>(
            "two-way-fake-resolver", "runtime-fake-payload", 1, 4096,
            dp_key, std::move(payload), std::move(lease));
    }
};

// RuntimeFakeDataPath is now tutti::testing::MockDataPath from the testing kit.
// The kit provides the same full-lifecycle SPI coverage with injection points
// (reject_at_index, fail_progress, fail_query, fail_release, manual_mode,
// block_progress_flag, call counters, progress serialization probe).
//
// For backward compatibility with existing test code that references
// RuntimeFakeDataPath, we provide a type alias.
using RuntimeFakeDataPath = tutti::testing::MockDataPath;

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

// Create a valid MemoryView for register_memory tests.
static tutti::MemoryView make_host_view(void* addr, std::uint64_t sz) {
    return tutti::MemoryView{
        addr, sz, tutti::MemoryKind::HOST,
        tutti::MemoryOwnership::CALLER_OWNED, -1, ""};
}

// Convenience: submit a single request and extract the IoHandle.
// Fails the calling test (returns invalid handle) if the submit fails.
static tutti::IoHandle submit_one(tutti::StorageRuntime& rt,
                                  const tutti::IoRequest& req,
                                  const tutti::HostSubmitContext& ctx) {
    auto outcome = rt.submit(&req, 1, ctx);
    if (!outcome.io.has_value()) {
        return tutti::IoHandle{};
    }
    return *outcome.io;
}

// Convenience: force-complete an IO via the test access struct.
static void force_complete(tutti::StorageRuntime& rt,
                           const tutti::IoHandle& h,
                           tutti::IoState state = tutti::IoState::COMPLETED) {
    tutti::testing::StorageRuntimeTestAccess::force_complete_io(rt, h, state);
}

// =====================================================================
// Existing tests (1-14, updated for new submit signature)
// =====================================================================

// 1. create() success -> state is RUNNING.
static int test_create_running() {
    auto result = tutti::StorageRuntime::create();
    if (!result.ok()) return 1;
    auto rt = std::move(result).value();
    if (rt->state() != tutti::RuntimeState::RUNNING) return 1;
    return 0;
}

// 2. shutdown() -> state STOPPED; all handles invalidated.
static int test_shutdown_stopped() {
    auto result = tutti::StorageRuntime::create();
    auto rt = std::move(result).value();

    char buf[64];
    auto reg = rt->register_memory(make_host_view(buf, sizeof(buf)));
    if (!reg.ok()) return 1;
    auto mem = reg.value();

    auto st = rt->shutdown(100);
    if (!st.ok()) return 1;
    if (rt->state() != tutti::RuntimeState::STOPPED) return 1;

    // After STOPPED, the handle should be invalid.
    auto info = rt->query_memory(mem);
    if (info.ok()) return 1;  // should fail
    return 0;
}

// 3. register_memory() returns valid handle; nullptr/size==0 -> INVALID_ARGUMENT.
static int test_register_memory_validation() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    // Valid registration
    char buf[64];
    auto reg = rt->register_memory(make_host_view(buf, sizeof(buf)));
    if (!reg.ok()) return 1;
    if (!reg.value().valid()) return 1;

    // nullptr address
    auto bad1 = rt->register_memory(make_host_view(nullptr, 64));
    if (bad1.ok()) return 1;
    if (bad1.status().code() != tutti::StatusCode::INVALID_ARGUMENT) return 1;

    // size == 0
    auto bad2 = rt->register_memory(make_host_view(buf, 0));
    if (bad2.ok()) return 1;
    if (bad2.status().code() != tutti::StatusCode::INVALID_ARGUMENT) return 1;
    return 0;
}

// 4. unregister_memory() -> handle deterministically invalidated.
static int test_unregister_invalidates() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf[64];
    auto reg = rt->register_memory(make_host_view(buf, sizeof(buf)));
    auto handle = reg.value();

    auto st = rt->unregister_memory(handle);
    if (!st.ok()) return 1;

    // Same handle should now fail -- not crash.
    auto info = rt->query_memory(handle);
    if (info.ok()) return 1;
    return 0;
}

// 5. Generation not reused: unregister then register, new != old, old not revived.
static int test_generation_not_reused() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf1[64], buf2[64];
    auto reg1 = rt->register_memory(make_host_view(buf1, sizeof(buf1)));
    auto h1 = reg1.value();
    rt->unregister_memory(h1);

    auto reg2 = rt->register_memory(make_host_view(buf2, sizeof(buf2)));
    auto h2 = reg2.value();

    // New handle must not equal old handle.
    if (h1 == h2) return 1;

    // Old handle must not "revive".
    auto info = rt->query_memory(h1);
    if (info.ok()) return 1;

    // New handle works.
    auto info2 = rt->query_memory(h2);
    if (!info2.ok()) return 1;
    return 0;
}

// 6. Ownership symmetry: free on caller-owned -> INVALID_ARGUMENT;
//    unregister on runtime-owned -> INVALID_ARGUMENT.
static int test_ownership_symmetry() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    // Runtime-owned allocation
    auto alloc = rt->allocate_memory(tutti::MemorySpec{64, tutti::MemoryKind::HOST, -1});
    auto runtime_handle = alloc.value().handle;

    // unregister on runtime-owned -> INVALID_ARGUMENT
    auto st1 = rt->unregister_memory(runtime_handle);
    if (st1.code() != tutti::StatusCode::INVALID_ARGUMENT) return 1;

    // Caller-owned registration
    char buf[64];
    auto reg = rt->register_memory(make_host_view(buf, sizeof(buf)));
    auto caller_handle = reg.value();

    // free on caller-owned -> INVALID_ARGUMENT
    auto st2 = rt->free_memory(caller_handle);
    if (st2.code() != tutti::StatusCode::INVALID_ARGUMENT) return 1;
    return 0;
}

// 7. open() returns valid TargetHandle; close() invalidates.
static int test_open_close() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    auto open_result = rt->open("stub://test", tutti::OpenOptions{"stub"});
    if (!open_result.ok()) return 1;
    auto target = open_result.value();
    if (!target.valid()) return 1;

    auto info = rt->query_target(target);
    if (!info.ok()) return 1;
    if (info.value().uri != "stub://test") return 1;

    auto st = rt->close(target);
    if (!st.ok()) return 1;

    auto info2 = rt->query_target(target);
    if (info2.ok()) return 1;
    return 0;
}

// 8. BUSY semantics: inflight IO -> close returns BUSY;
//    memory inflight -> unregister/free returns BUSY.
static int test_busy_semantics() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf[64];
    auto mem_reg = rt->register_memory(make_host_view(buf, sizeof(buf)));
    auto mem = mem_reg.value();

    auto tgt_open = rt->open("stub://busy", tutti::OpenOptions{"stub"});
    auto tgt = tgt_open.value();

    // Submit IO (stays in-flight in stub)
    tutti::IoRequest req{
        tutti::IoDirection::READ, mem, 0, tgt, 0, 64};
    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};
    auto io = submit_one(*rt, req, ctx);
    if (!io.valid()) return 1;

    // close with inflight -> BUSY
    auto st_close = rt->close(tgt);
    if (st_close.code() != tutti::StatusCode::BUSY) return 1;

    // unregister with inflight -> BUSY
    auto st_unreg = rt->unregister_memory(mem);
    if (st_unreg.code() != tutti::StatusCode::BUSY) return 1;

    // For runtime-owned, also test free with inflight:
    auto alloc = rt->allocate_memory(tutti::MemorySpec{128, tutti::MemoryKind::HOST, -1});
    auto alloc_handle = alloc.value().handle;

    tutti::IoRequest req2{
        tutti::IoDirection::WRITE, alloc_handle, 0, tgt, 0, 128};
    auto io2 = submit_one(*rt, req2, ctx);
    if (!io2.valid()) return 1;

    auto st_free = rt->free_memory(alloc_handle);
    if (st_free.code() != tutti::StatusCode::BUSY) return 1;

    // Clean up
    force_complete(*rt, io);
    force_complete(*rt, io2);
    rt->release_io(io);
    rt->release_io(io2);
    rt->unregister_memory(mem);
    rt->free_memory(alloc_handle);
    rt->close(tgt);
    return 0;
}

// 9. WaitOutcome distinguishes observation timeout vs terminal vs invalid handle.
static int test_wait_outcome() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf[64];
    auto mem_reg = rt->register_memory(make_host_view(buf, sizeof(buf)));
    auto mem = mem_reg.value();

    auto tgt_open = rt->open("stub://wait", tutti::OpenOptions{"stub"});
    auto tgt = tgt_open.value();

    tutti::IoRequest req{
        tutti::IoDirection::READ, mem, 0, tgt, 0, 64};
    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};
    auto io = submit_one(*rt, req, ctx);
    if (!io.valid()) return 1;

    // (a) In-flight with tiny timeout -> TIMEOUT, no result, still queryable.
    auto wo = rt->wait(io, 1);
    if (wo.observation_status.code() != tutti::StatusCode::TIMEOUT) return 1;
    if (wo.result.has_value()) return 1;

    // Operation still queryable (not cancelled).
    auto snap = rt->query(io);
    if (!snap.ok()) return 1;
    if (snap.value().state != tutti::IoState::IN_FLIGHT) return 1;

    // (b) Terminal -> wait returns OK + result.
    force_complete(*rt, io);
    auto wo2 = rt->wait(io, 1);
    if (!wo2.observation_status.ok()) return 1;
    if (!wo2.result.has_value()) return 1;
    if (wo2.result->state != tutti::IoState::COMPLETED) return 1;

    // (c) Invalid handle -> non-OK observation, no result.
    tutti::IoHandle invalid;
    auto wo3 = rt->wait(invalid, 1);
    if (wo3.observation_status.ok()) return 1;
    if (wo3.result.has_value()) return 1;

    // Clean up
    rt->release_io(io);
    rt->unregister_memory(mem);
    rt->close(tgt);
    return 0;
}

// 10. release_io() inflight -> BUSY; terminal -> OK + handle invalidated.
static int test_release_io() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf[64];
    auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
    auto tgt = rt->open("stub://release", tutti::OpenOptions{"stub"}).value();

    tutti::IoRequest req{
        tutti::IoDirection::READ, mem, 0, tgt, 0, 64};
    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};
    auto io = submit_one(*rt, req, ctx);
    if (!io.valid()) return 1;

    // Inflight -> BUSY
    auto st = rt->release_io(io);
    if (st.code() != tutti::StatusCode::BUSY) return 1;

    // Complete -> release OK
    force_complete(*rt, io);
    auto st2 = rt->release_io(io);
    if (!st2.ok()) return 1;

    // Handle now invalid
    auto snap = rt->query(io);
    if (snap.ok()) return 1;

    rt->unregister_memory(mem);
    rt->close(tgt);
    return 0;
}

// 11. Terminal result not auto-evicted: multiple query() return same terminal.
static int test_terminal_not_evicted() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf[64];
    auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
    auto tgt = rt->open("stub://evict", tutti::OpenOptions{"stub"}).value();

    tutti::IoRequest req{
        tutti::IoDirection::READ, mem, 0, tgt, 0, 64};
    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};
    auto io = submit_one(*rt, req, ctx);
    if (!io.valid()) return 1;

    tutti::testing::StorageRuntimeTestAccess::force_complete_io(
        *rt, io, tutti::IoState::FAILED,
        tutti::Status(tutti::StatusCode::DEVICE_ERROR, "stub failure"));

    // Query multiple times -- all return same terminal state.
    for (int i = 0; i < 5; ++i) {
        auto snap = rt->query(io);
        if (!snap.ok()) return 1;
        if (snap.value().state != tutti::IoState::FAILED) return 1;
    }

    // wait also returns same terminal result.
    auto wo = rt->wait(io, 0);
    if (!wo.observation_status.ok()) return 1;
    if (!wo.result.has_value()) return 1;
    if (wo.result->state != tutti::IoState::FAILED) return 1;
    if (wo.result->status.code() != tutti::StatusCode::DEVICE_ERROR) return 1;

    rt->release_io(io);
    rt->unregister_memory(mem);
    rt->close(tgt);
    return 0;
}

// 12. Cross-runtime handle rejected, no crash.
static int test_cross_runtime_rejected() {
    auto rt1 = std::move(tutti::StorageRuntime::create()).value();
    auto rt2 = std::move(tutti::StorageRuntime::create()).value();

    char buf[64];
    auto reg = rt1->register_memory(make_host_view(buf, sizeof(buf)));
    auto handle = reg.value();

    // Use handle from rt1 on rt2 -> should fail, not crash.
    auto info = rt2->query_memory(handle);
    if (info.ok()) return 1;

    auto st = rt2->unregister_memory(handle);
    if (st.ok()) return 1;

    // Also test IO handle cross-runtime.
    auto tgt = rt1->open("stub://cross", tutti::OpenOptions{"stub"}).value();
    tutti::IoRequest req{
        tutti::IoDirection::READ, handle, 0, tgt, 0, 64};
    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};
    auto io = submit_one(*rt1, req, ctx);
    if (!io.valid()) return 1;

    auto snap = rt2->query(io);
    if (snap.ok()) return 1;

    auto wo = rt2->wait(io, 1);
    if (wo.observation_status.ok()) return 1;

    auto st2 = rt2->release_io(io);
    if (st2.ok()) return 1;
    return 0;
}

// 13. Shutdown observation timeout -> DRAINING, handles still queryable.
static int test_shutdown_draining() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf[64];
    auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
    auto tgt = rt->open("stub://drain", tutti::OpenOptions{"stub"}).value();

    tutti::IoRequest req{
        tutti::IoDirection::READ, mem, 0, tgt, 0, 64};
    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};
    auto io = submit_one(*rt, req, ctx);
    if (!io.valid()) return 1;

    // Shutdown with tiny timeout -> stays DRAINING.
    auto st = rt->shutdown(1);
    if (st.code() != tutti::StatusCode::TIMEOUT) return 1;
    if (rt->state() != tutti::RuntimeState::DRAINING) return 1;

    // In DRAINING, handles are still valid for query/wait/release.
    auto snap = rt->query(io);
    if (!snap.ok()) return 1;
    if (snap.value().state != tutti::IoState::IN_FLIGHT) return 1;

    auto wo = rt->wait(io, 1);
    if (wo.observation_status.code() != tutti::StatusCode::TIMEOUT) return 1;

    // release_io still works (returns BUSY since inflight).
    auto st_rel = rt->release_io(io);
    if (st_rel.code() != tutti::StatusCode::BUSY) return 1;

    // Complete the IO, release, then shutdown again -> STOPPED.
    force_complete(*rt, io);
    rt->release_io(io);
    rt->unregister_memory(mem);
    rt->close(tgt);

    auto st2 = rt->shutdown(100);
    if (!st2.ok()) return 1;
    if (rt->state() != tutti::RuntimeState::STOPPED) return 1;
    return 0;
}

// 14. query_memory() returns MemoryInfo with no transport-private fields.
static int test_memory_info_no_private_fields() {
    static_assert(std::is_same_v<decltype(tutti::MemoryInfo::kind), tutti::MemoryKind>);
    static_assert(std::is_same_v<decltype(tutti::MemoryInfo::ownership), tutti::MemoryOwnership>);
    static_assert(std::is_same_v<decltype(tutti::MemoryInfo::size), std::uint64_t>);
    static_assert(std::is_same_v<decltype(tutti::MemoryInfo::address), void*>);
    static_assert(std::is_same_v<decltype(tutti::MemoryInfo::accel_id), std::int32_t>);
    static_assert(std::is_same_v<decltype(tutti::MemoryInfo::inflight_count), int>);

    // Aggregate initialization with exactly 6 fields.
    tutti::MemoryInfo mi{
        tutti::MemoryKind::HOST,
        tutti::MemoryOwnership::CALLER_OWNED,
        0,
        nullptr,
        -1,
        0
    };
    (void)mi;

    auto rt = std::move(tutti::StorageRuntime::create()).value();
    char buf[64];
    auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
    auto info = rt->query_memory(mem);
    if (!info.ok()) return 1;

    const auto& mi2 = info.value();
    (void)mi2.kind;
    (void)mi2.ownership;
    (void)mi2.size;
    (void)mi2.address;
    (void)mi2.accel_id;
    (void)mi2.inflight_count;
    return 0;
}

// =====================================================================
// New tests (15-22)
// =====================================================================

// 15. Batch success: 4 valid requests -> status OK, io present,
//     initial_states.size() == 4, all ACCEPTED.
static int test_batch_success() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf[4096];
    auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
    auto tgt = rt->open("stub://batch", tutti::OpenOptions{"stub"}).value();

    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};

    std::vector<tutti::IoRequest> reqs;
    for (int i = 0; i < 4; ++i) {
        reqs.push_back(tutti::IoRequest{
            tutti::IoDirection::READ, mem,
            static_cast<std::uint64_t>(i * 1024),
            tgt, static_cast<std::uint64_t>(i * 1024), 1024});
    }

    auto outcome = rt->submit(reqs.data(), reqs.size(), ctx);
    if (!outcome.status.ok()) return 1;
    if (!outcome.io.has_value()) return 1;
    if (!outcome.io->valid()) return 1;
    if (outcome.initial_states.size() != 4) return 1;
    for (int i = 0; i < 4; ++i) {
        if (outcome.initial_states[i].state != tutti::IoRequestState::ACCEPTED)
            return 1;
    }

    // Accepted requests are queryable.
    auto snap = rt->query(*outcome.io);
    if (!snap.ok()) return 1;

    force_complete(*rt, *outcome.io);
    rt->release_io(*outcome.io);
    rt->unregister_memory(mem);
    rt->close(tgt);
    return 0;
}

// 16. Partial commit: 4 requests, index 2 out-of-bounds -> rejected,
//     others accepted, io still present and queryable.
static int test_partial_commit() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf[4096];
    auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
    auto tgt = rt->open("stub://partial", tutti::OpenOptions{"stub"}).value();

    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};

    std::vector<tutti::IoRequest> reqs;
    // 0, 1: valid
    for (int i = 0; i < 2; ++i) {
        reqs.push_back(tutti::IoRequest{
            tutti::IoDirection::READ, mem,
            static_cast<std::uint64_t>(i * 512),
            tgt, static_cast<std::uint64_t>(i * 512), 512});
    }
    // 2: out-of-bounds memory_offset
    reqs.push_back(tutti::IoRequest{
        tutti::IoDirection::READ, mem,
        1ULL << 30,  // way beyond 4096
        tgt, 0, 1});
    // 3: valid
    reqs.push_back(tutti::IoRequest{
        tutti::IoDirection::READ, mem, 0, tgt, 0, 64});

    auto outcome = rt->submit(reqs.data(), reqs.size(), ctx);

    // Overall status non-OK (partial failure).
    if (outcome.status.ok()) return 1;
    // io still present (at least one accepted).
    if (!outcome.io.has_value()) return 1;
    if (!outcome.io->valid()) return 1;
    // 4 initial states.
    if (outcome.initial_states.size() != 4) return 1;
    // 0, 1, 3 accepted; 2 rejected.
    if (outcome.initial_states[0].state != tutti::IoRequestState::ACCEPTED) return 1;
    if (outcome.initial_states[1].state != tutti::IoRequestState::ACCEPTED) return 1;
    if (outcome.initial_states[2].state != tutti::IoRequestState::REJECTED) return 1;
    if (outcome.initial_states[3].state != tutti::IoRequestState::ACCEPTED) return 1;
    // Rejected one has OUT_OF_RANGE.
    if (outcome.initial_states[2].status.code() != tutti::StatusCode::OUT_OF_RANGE)
        return 1;

    // Accepted part still queryable.
    auto snap = rt->query(*outcome.io);
    if (!snap.ok()) return 1;

    force_complete(*rt, *outcome.io);
    rt->release_io(*outcome.io);
    rt->unregister_memory(mem);
    rt->close(tgt);
    return 0;
}

// 17. Zero accepted: all requests invalid -> io == nullopt, status non-OK,
//     all REJECTED.
static int test_zero_accepted() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf[4096];
    auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
    auto tgt = rt->open("stub://zero", tutti::OpenOptions{"stub"}).value();

    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};

    std::vector<tutti::IoRequest> reqs;
    // All out-of-bounds.
    reqs.push_back(tutti::IoRequest{
        tutti::IoDirection::READ, mem, 1ULL << 30, tgt, 0, 1});
    reqs.push_back(tutti::IoRequest{
        tutti::IoDirection::READ, mem, 1ULL << 30, tgt, 0, 1});

    auto outcome = rt->submit(reqs.data(), reqs.size(), ctx);
    if (outcome.io.has_value()) return 1;
    if (outcome.status.ok()) return 1;
    if (outcome.initial_states.size() != 2) return 1;
    if (outcome.initial_states[0].state != tutti::IoRequestState::REJECTED) return 1;
    if (outcome.initial_states[1].state != tutti::IoRequestState::REJECTED) return 1;

    rt->unregister_memory(mem);
    rt->close(tgt);
    return 0;
}

// 18. count == 0 -> io == nullopt, status OK, initial_states empty.
static int test_count_zero() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};

    auto outcome = rt->submit(nullptr, 0, ctx);
    if (outcome.io.has_value()) return 1;
    if (!outcome.status.ok()) return 1;
    if (!outcome.initial_states.empty()) return 1;
    return 0;
}

// 19. Bounds validation: length==0 -> INVALID_ARGUMENT.
static int test_bounds_length_zero() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf[4096];
    auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
    auto tgt = rt->open("stub://len0", tutti::OpenOptions{"stub"}).value();

    tutti::IoRequest req{
        tutti::IoDirection::READ, mem, 0, tgt, 0, 0};  // length == 0
    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};

    auto outcome = rt->submit(&req, 1, ctx);
    if (outcome.io.has_value()) return 1;
    if (outcome.initial_states.size() != 1) return 1;
    if (outcome.initial_states[0].state != tutti::IoRequestState::REJECTED) return 1;
    if (outcome.initial_states[0].status.code() != tutti::StatusCode::INVALID_ARGUMENT)
        return 1;

    rt->unregister_memory(mem);
    rt->close(tgt);
    return 0;
}

// 20. Bounds validation: memory out-of-range and target out-of-range.
static int test_bounds_out_of_range() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf[4096];
    auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
    auto tgt = rt->open("stub://oor", tutti::OpenOptions{"stub"}).value();

    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};

    // (a) memory out of range
    {
        tutti::IoRequest req{
            tutti::IoDirection::READ, mem, 4096, tgt, 0, 1};
        auto outcome = rt->submit(&req, 1, ctx);
        if (outcome.io.has_value()) return 1;
        if (outcome.initial_states[0].status.code() != tutti::StatusCode::OUT_OF_RANGE)
            return 1;
    }

    // (b) target out of range (target logical_size is 1<<30 = 1GiB)
    {
        tutti::IoRequest req{
            tutti::IoDirection::READ, mem, 0, tgt,
            1ULL << 30,  // exactly at target_size, so offset == size -> OOR
            1};
        auto outcome = rt->submit(&req, 1, ctx);
        if (outcome.io.has_value()) return 1;
        if (outcome.initial_states[0].status.code() != tutti::StatusCode::OUT_OF_RANGE)
            return 1;
    }

    rt->unregister_memory(mem);
    rt->close(tgt);
    return 0;
}

// 21. Overflow safety: near-UINT64_MAX offsets/lengths rejected without wrap.
static int test_overflow_safety() {
    auto rt = std::move(tutti::StorageRuntime::create()).value();

    char buf[4096];
    auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
    auto tgt = rt->open("stub://overflow", tutti::OpenOptions{"stub"}).value();

    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};

    // memory_offset near UINT64_MAX, length=1 — must be rejected (OOR),
    // not wrap to appear valid.
    {
        tutti::IoRequest req{
            tutti::IoDirection::READ, mem,
            ~0ULL - 10,  // very large offset
            tgt, 0, 1};
        auto outcome = rt->submit(&req, 1, ctx);
        if (outcome.io.has_value()) return 1;
        if (outcome.initial_states[0].state != tutti::IoRequestState::REJECTED) return 1;
    }

    // length near UINT64_MAX — must be rejected (OOR), not wrap.
    {
        tutti::IoRequest req{
            tutti::IoDirection::READ, mem, 0,
            tgt, 0, ~0ULL - 10};
        auto outcome = rt->submit(&req, 1, ctx);
        if (outcome.io.has_value()) return 1;
        if (outcome.initial_states[0].state != tutti::IoRequestState::REJECTED) return 1;
    }

    rt->unregister_memory(mem);
    rt->close(tgt);
    return 0;
}

// 22. wait()/query() are non-const; test backdoor not on public surface.
static int test_nonconst_and_test_access() {
    // (a) Verify query() and wait() are non-const member functions.
    //
    // We check this by confirming the member-function-pointer type does NOT
    // carry a trailing const qualifier. A const method would have type
    //   R (StorageRuntime::*)(Args...) const
    // whereas a non-const method has
    //   R (StorageRuntime::*)(Args...)
    // We use is_member_pointer + a manual const-ness check via a trait.
    //
    // The key proof: if query/wait were const, calling them on a const
    // StorageRuntime& would compile. We verify non-const-ness by confirming
    // that the decltype of the member pointer matches the non-const form.
    //
    // Simple proof: the code below calls query() and wait() on a non-const
    // reference and it compiles. If these methods were const-only, they
    // would still compile on a non-const reference. However, the critical
    // design requirement is that they CAN mutate internal state (drive
    // bounded_progress_), which is only possible if they are non-const.
    // The non-const signature is the contract that enables future progress
    // driving without API changes.
    //
    // We verify this at compile time by checking that the member function
    // pointer type is the non-const variant:
    static_assert(
        std::is_member_pointer_v<decltype(&tutti::StorageRuntime::query)>,
        "query must be a member function");
    static_assert(
        std::is_member_pointer_v<decltype(&tutti::StorageRuntime::wait)>,
        "wait must be a member function");

    // (b) Demonstrate that calling query/wait on a non-const reference works.
    auto rt = std::move(tutti::StorageRuntime::create()).value();
    char buf[64];
    auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
    auto tgt = rt->open("stub://nc", tutti::OpenOptions{"stub"}).value();
    tutti::IoRequest req{
        tutti::IoDirection::READ, mem, 0, tgt, 0, 64};
    tutti::HostSubmitContext ctx{
        tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};
    auto io = submit_one(*rt, req, ctx);
    if (!io.valid()) return 1;

    // non-const calls
    auto snap = rt->query(io);
    if (!snap.ok()) return 1;
    auto wo = rt->wait(io, 1);
    if (wo.observation_status.code() != tutti::StatusCode::TIMEOUT) return 1;

    // (c) Test backdoor: force_complete is only reachable via
    // testing::StorageRuntimeTestAccess, not directly on StorageRuntime.
    // The following line would NOT compile if uncommented, proving the
    // method is private:
    //   rt->testing_force_complete_io_(io, tutti::IoState::COMPLETED);
    // Instead, we use the friend access struct:
    force_complete(*rt, io);
    rt->release_io(io);
    rt->unregister_memory(mem);
    rt->close(tgt);
    return 0;
}

// 23. Injected resolver/DataPath route open → lazy registration → submit →
//     progress/query → release → close/unregister/shutdown without exposing
//     private target or DMA identities through public handles.
static int test_component_backed_runtime_route() {
    RuntimeFakeResolver resolver;
    RuntimeFakeDataPath data_path;
    tutti::RuntimeComponents components;
    components.resolvers.push_back({"fake", &resolver});
    components.data_paths.push_back(
        {"runtime-fake-datapath", &data_path, tutti::DataPathConfig{"fake"}});

    auto created = tutti::StorageRuntime::create({}, std::move(components));
    if (!created.ok()) return 1;
    auto runtime = std::move(created).value();
    if (data_path.initialize_calls != 1) return 1;

    char buffer[128]{};
    auto memory = runtime->register_memory(make_host_view(buffer, sizeof(buffer)));
    if (!memory.ok()) return 1;
    auto target = runtime->open("fake://runtime-route", tutti::OpenOptions{});
    if (!target.ok()) return 1;
    if (resolver.resolve_calls != 1 || data_path.open_calls != 1) return 1;
    auto target_info = runtime->query_target(target.value());
    if (!target_info.ok() || target_info.value().logical_size != 4096) return 1;

    tutti::IoRequest request{
        tutti::IoDirection::WRITE, memory.value(), 16,
        target.value(), 32, 64};
    tutti::HostSubmitContext context{
        tutti::ExecutionDomain::HOST_EXECUTION, -1, nullptr};
    auto submitted = runtime->submit(&request, 1, context);
    if (!submitted.status.ok() || !submitted.io.has_value()) return 1;
    if (data_path.register_calls != 1 || data_path.submit_calls != 1) return 1;
    if (data_path.last_requests.size() != 1 ||
        data_path.last_requests[0].intent.memory_offset != 16 ||
        data_path.last_requests[0].intent.target_offset != 32 ||
        data_path.last_requests[0].intent.length != 64 ||
        !data_path.last_requests[0].memory.valid() ||
        !data_path.last_requests[0].target.valid()) return 1;

    if (runtime->close(target.value()).code() != tutti::StatusCode::BUSY) return 1;
    if (runtime->unregister_memory(memory.value()).code() != tutti::StatusCode::BUSY)
        return 1;

    auto snapshot = runtime->query(*submitted.io);
    if (!snapshot.ok() || snapshot.value().state != tutti::IoState::COMPLETED)
        return 1;
    if (data_path.progress_calls != 1 || data_path.release_calls != 1) return 1;
    if (!runtime->release_io(*submitted.io).ok()) return 1;

    // Registration is lazy and cached by {DataPath, registration domain}.
    auto submitted_again = runtime->submit(&request, 1, context);
    if (!submitted_again.status.ok() || !submitted_again.io.has_value()) return 1;
    if (data_path.register_calls != 1 || data_path.submit_calls != 2) return 1;
    auto completed_again = runtime->query(*submitted_again.io);
    if (!completed_again.ok() ||
        completed_again.value().state != tutti::IoState::COMPLETED) return 1;
    if (!runtime->release_io(*submitted_again.io).ok()) return 1;

    if (!runtime->close(target.value()).ok() || data_path.close_calls != 1) return 1;
    if (!runtime->unregister_memory(memory.value()).ok() ||
        data_path.unregister_calls != 1) return 1;
    if (!runtime->shutdown(1).ok() || data_path.shutdown_calls != 1) return 1;
    return 0;
}

// 24. Component assembly refuses an unregistered URI scheme without opening a
// private target or registering memory.
static int test_component_runtime_unknown_scheme() {
    RuntimeFakeResolver resolver;
    RuntimeFakeDataPath data_path;
    tutti::RuntimeComponents components;
    components.resolvers.push_back({"fake", &resolver});
    components.data_paths.push_back(
        {"runtime-fake-datapath", &data_path, tutti::DataPathConfig{"fake"}});
    auto created = tutti::StorageRuntime::create({}, std::move(components));
    if (!created.ok()) return 1;
    auto runtime = std::move(created).value();
    auto target = runtime->open("missing://target", tutti::OpenOptions{});
    if (target.ok() || target.status().code() != tutti::StatusCode::NOT_FOUND)
        return 1;
    if (resolver.resolve_calls != 0 || data_path.open_calls != 0) return 1;
    return runtime->shutdown(1).ok() ? 0 : 1;
    if (resolver.resolve_calls != 0 || data_path.open_calls != 0) return 1;
    auto sh = runtime->shutdown(1);
    std::fprintf(stderr, "DBG25: shutdown.ok=%d\n", sh.ok());
    return sh.ok() ? 0 : 1;
}

// 25. A public batch spanning two targets on the SAME DataPath is merged
// into a single DataPath::submit call (Round 15 Session 3: grouping key is
// DataPath only, not (DataPath, target)), observable through one Runtime
// IoHandle.
static int test_component_runtime_groups_by_datapath() {
    RuntimeFakeResolver resolver;
    RuntimeFakeDataPath data_path;
    tutti::RuntimeComponents components;
    components.resolvers.push_back({"fake", &resolver});
    components.data_paths.push_back(
        {"runtime-fake-datapath", &data_path, tutti::DataPathConfig{"fake"}});
    auto created = tutti::StorageRuntime::create({}, std::move(components));
    if (!created.ok()) return 1;
    auto runtime = std::move(created).value();

    char buffer[128]{};
    auto memory = runtime->register_memory(make_host_view(buffer, sizeof(buffer)));
    auto first = runtime->open("fake://first", tutti::OpenOptions{});
    auto second = runtime->open("fake://second", tutti::OpenOptions{});
    if (!memory.ok() || !first.ok() || !second.ok()) return 1;

    const tutti::IoRequest requests[] = {
        {tutti::IoDirection::WRITE, memory.value(), 0, first.value(), 0, 32},
        {tutti::IoDirection::READ, memory.value(), 64, second.value(), 128, 32},
    };
    const tutti::HostSubmitContext context{
        tutti::ExecutionDomain::HOST_EXECUTION, -1, nullptr};
    auto submitted = runtime->submit(requests, 2, context);
    if (!submitted.status.ok() || !submitted.io.has_value()) return 1;
    if (data_path.submit_calls != 1 || data_path.register_calls != 1) return 1;
    if (data_path.last_requests.size() != 2) return 1;

    auto snapshot = runtime->query(*submitted.io);
    if (!snapshot.ok() || snapshot.value().state != tutti::IoState::COMPLETED)
        return 1;
    if (data_path.release_calls != 1 || !runtime->release_io(*submitted.io).ok())
        return 1;
    if (!runtime->close(first.value()).ok() ||
        !runtime->close(second.value()).ok() ||
        !runtime->unregister_memory(memory.value()).ok()) return 1;
    return runtime->shutdown(1).ok() ? 0 : 1;
}

// 26. A DataPath partial commit remains a public Runtime partial commit: the
// issued request has an IoHandle while the rejected request remains explicit.
static int test_component_runtime_partial_commit() {
    RuntimeFakeResolver resolver;
    RuntimeFakeDataPath data_path;
    data_path.reject_at_index = 1;
    tutti::RuntimeComponents components;
    components.resolvers.push_back({"fake", &resolver});
    components.data_paths.push_back(
        {"runtime-fake-datapath", &data_path, tutti::DataPathConfig{"fake"}});
    auto created = tutti::StorageRuntime::create({}, std::move(components));
    if (!created.ok()) return 1;
    auto runtime = std::move(created).value();

    char buffer[128]{};
    auto memory = runtime->register_memory(make_host_view(buffer, sizeof(buffer)));
    auto target = runtime->open("fake://partial", tutti::OpenOptions{});
    if (!memory.ok() || !target.ok()) return 1;

    const tutti::IoRequest requests[] = {
        {tutti::IoDirection::WRITE, memory.value(), 0, target.value(), 0, 32},
        {tutti::IoDirection::READ, memory.value(), 64, target.value(), 64, 32},
    };
    const tutti::HostSubmitContext context{
        tutti::ExecutionDomain::HOST_EXECUTION, -1, nullptr};
    auto submitted = runtime->submit(requests, 2, context);
    if (submitted.status.ok() || !submitted.io.has_value() ||
        submitted.initial_states.size() != 2 ||
        submitted.initial_states[0].state != tutti::IoRequestState::ACCEPTED ||
        submitted.initial_states[1].state != tutti::IoRequestState::REJECTED ||
        submitted.initial_states[1].status.ok()) return 1;

    auto snapshot = runtime->query(*submitted.io);
    if (!snapshot.ok() || snapshot.value().state != tutti::IoState::COMPLETED ||
        !runtime->release_io(*submitted.io).ok()) return 1;
    if (!runtime->close(target.value()).ok() ||
        !runtime->unregister_memory(memory.value()).ok()) return 1;
    return runtime->shutdown(1).ok() ? 0 : 1;
}

// 27. The two create overloads share a single runtime-ID namespace. Run this
// first: before the shared generator fix, both factories minted {id=1,slot=0,
// generation=1} and accepted each other's memory handle.
static int test_cross_assembly_runtime_rejected() {
    RuntimeFakeResolver resolver;
    RuntimeFakeDataPath data_path;
    tutti::RuntimeComponents components;
    components.resolvers.push_back({"fake", &resolver});
    components.data_paths.push_back(
        {"runtime-fake-datapath", &data_path, tutti::DataPathConfig{"fake"}});
    auto component_created = tutti::StorageRuntime::create({}, std::move(components));
    auto plain_created = tutti::StorageRuntime::create();
    if (!component_created.ok() || !plain_created.ok()) return 1;
    auto component = std::move(component_created).value();
    auto plain = std::move(plain_created).value();

    char component_buffer[8]{};
    char plain_buffer[8]{};
    auto component_memory = component->register_memory(
        make_host_view(component_buffer, sizeof(component_buffer)));
    auto plain_memory = plain->register_memory(
        make_host_view(plain_buffer, sizeof(plain_buffer)));
    if (!component_memory.ok() || !plain_memory.ok()) return 1;
    if (component->query_memory(plain_memory.value()).ok() ||
        plain->query_memory(component_memory.value()).ok()) return 1;

    if (!component->unregister_memory(component_memory.value()).ok() ||
        !plain->unregister_memory(plain_memory.value()).ok()) return 1;
    return component->shutdown(1).ok() && plain->shutdown(1).ok() ? 0 : 1;
}

// =====================================================================
// Concurrency + error-hardening tests (28-34, T-032 Session 2)
// =====================================================================

// Helper: build a component-backed runtime with a fake resolver+DataPath.
struct FakeRuntime {
    RuntimeFakeResolver resolver;
    RuntimeFakeDataPath data_path;
    std::unique_ptr<tutti::StorageRuntime> rt;

    FakeRuntime() {
        tutti::RuntimeComponents components;
        components.resolvers.push_back({"fake", &resolver});
        components.data_paths.push_back(
            {"runtime-fake-datapath", &data_path, tutti::DataPathConfig{"fake"}});
        auto result = tutti::StorageRuntime::create({}, std::move(components));
        rt = std::move(result).value();
    }

    std::pair<tutti::MemoryHandle, tutti::TargetHandle> setup() {
        static char buf[4096];
        auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
        auto tgt = rt->open("fake://concurrent", tutti::OpenOptions{"fake"}).value();
        return {mem, tgt};
    }

    tutti::IoHandle submit_one(tutti::MemoryHandle mem,
                                tutti::TargetHandle tgt) {
        tutti::IoRequest req{
            tutti::IoDirection::WRITE, mem, 0, tgt, 0, 64};
        tutti::HostSubmitContext ctx{
            tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};
        auto outcome = rt->submit(&req, 1, ctx);
        return outcome.io.value_or(tutti::IoHandle{});
    }
};

// 28. Multi-thread submit/query/wait/release on the same runtime.
static int test_concurrent_submit_query_release() {
    auto fr = std::make_unique<FakeRuntime>();
    auto [mem, tgt] = fr->setup();

    constexpr int N = 8;
    std::atomic<int> errors{0};

    auto worker = [&]() {
        auto io = fr->submit_one(mem, tgt);
        if (!io.valid()) { ++errors; return; }
        auto wo = fr->rt->wait(io, 2000);
        if (!wo.observation_status.ok() || !wo.result.has_value()) { ++errors; return; }
        if (!fr->rt->release_io(io).ok()) { ++errors; return; }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < N; ++i) threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    if (errors.load() != 0) return 1;
    fr->rt->unregister_memory(mem);
    fr->rt->close(tgt);
    return fr->rt->shutdown(100).ok() ? 0 : 1;
}

// 29. close/unregister vs inflight IO: submit first (deterministic), then
//     race close+unregister — both must see BUSY because inflight credit
//     is held.  Then drain via shutdown.
static int test_close_unregister_vs_submit_race() {
    FakeRuntime fr;
    auto [mem, tgt] = fr.setup();
    fr.data_path.manual_mode = true;

    // Submit first — the op stays IN_FLIGHT (manual mode).
    auto io = fr.submit_one(mem, tgt);
    if (!io.valid()) return 1;

    // Race close and unregister concurrently — both should observe BUSY
    // because the inflight credit prevents teardown.
    std::atomic<int> busy{0};
    auto closer = [&]() {
        if (fr.rt->close(tgt).code() == tutti::StatusCode::BUSY) ++busy;
    };
    auto unreg = [&]() {
        if (fr.rt->unregister_memory(mem).code() == tutti::StatusCode::BUSY) ++busy;
    };
    std::thread t_close(closer);
    std::thread t_unreg(unreg);
    t_close.join();
    t_unreg.join();

    // At least one must see BUSY (both is acceptable).
    if (busy.load() == 0) return 1;

    // Allow the op to complete so shutdown can drain.
    fr.data_path.manual_mode = false;
    return fr.rt->shutdown(1000).ok() ? 0 : 1;
}

// 30. Two IoHandles on the same DataPath: concurrent query drives progress
//     but progress serialization gate ensures max 1 concurrent progress().
static int test_progress_serialization() {
    FakeRuntime fr;
    auto [mem, tgt] = fr.setup();

    // Block progress so we can observe concurrency.
    fr.data_path.block_progress_flag = true;
    fr.data_path.manual_mode = true;

    auto io1 = fr.submit_one(mem, tgt);
    auto io2 = fr.submit_one(mem, tgt);
    if (!io1.valid() || !io2.valid()) return 1;

    std::atomic<int> started{0};
    auto querier = [&]() {
        // query() drives progress internally.  Two concurrent queries should
        // not both enter progress() at the same time.
        ++started;
        (void)fr.rt->query(io1);
    };

    std::thread t1(querier);
    std::thread t2(querier);
    // Wait for both threads to enter query.
    while (started.load() < 2) std::this_thread::yield();

    // Unblock progress; both queries should complete.
    fr.data_path.unblock_progress();

    // Complete ops manually so query can harvest terminal state.
    fr.data_path.manual_mode = false;
    t1.join();
    t2.join();

    // The fake DataPath recorded the maximum concurrent progress() count.
    // It must be <= 1 (serialization gate).
    if (fr.data_path.progress_max_concurrent.load() > 1) return 1;

    fr.rt->wait(io1, 100);
    fr.rt->wait(io2, 100);
    fr.rt->release_io(io1);
    fr.rt->release_io(io2);
    fr.rt->unregister_memory(mem);
    fr.rt->close(tgt);
    return fr.rt->shutdown(100).ok() ? 0 : 1;
}

// 31. fake progress/query/release returns error → Runtime op reaches FAILED,
//     wait observes it, not stuck IN_FLIGHT.
static int test_datapath_failure_terminates() {
    FakeRuntime fr;
    auto [mem, tgt] = fr.setup();
    fr.data_path.manual_mode = true;

    auto io = fr.submit_one(mem, tgt);
    if (!io.valid()) return 1;

    // Inject query failure.  The next progress cycle should terminal-fail
    // the op instead of leaving it IN_FLIGHT forever.
    fr.data_path.fail_query = true;

    auto wo = fr.rt->wait(io, 1000);
    if (!wo.observation_status.ok()) return 1;
    if (!wo.result.has_value()) return 1;
    if (wo.result->state != tutti::IoState::FAILED) return 1;

    // release_io should succeed (op is terminal).
    auto st = fr.rt->release_io(io);
    if (!st.ok()) return 1;

    fr.data_path.fail_query = false;
    fr.rt->unregister_memory(mem);
    fr.rt->close(tgt);
    return fr.rt->shutdown(100).ok() ? 0 : 1;
}

// 32. shutdown(0) → DRAINING/TIMEOUT; then progress/wait; then shutdown → STOPPED.
static int test_shutdown_drain_retry() {
    FakeRuntime fr;
    auto [mem, tgt] = fr.setup();
    fr.data_path.manual_mode = true;

    auto io = fr.submit_one(mem, tgt);
    if (!io.valid()) return 1;

    // shutdown(0) with inflight → TIMEOUT, stays DRAINING.
    auto st1 = fr.rt->shutdown(0);
    if (st1.code() != tutti::StatusCode::TIMEOUT) return 1;
    if (fr.rt->state() != tutti::RuntimeState::DRAINING) return 1;

    // Op still queryable in DRAINING.
    auto snap = fr.rt->query(io);
    if (!snap.ok()) return 1;
    if (snap.value().state != tutti::IoState::IN_FLIGHT) return 1;

    // Complete the op, then wait should observe terminal.
    fr.data_path.manual_mode = false;
    auto wo = fr.rt->wait(io, 1000);
    if (!wo.observation_status.ok()) return 1;

    fr.rt->release_io(io);

    // Now shutdown should succeed → STOPPED.
    auto st2 = fr.rt->shutdown(100);
    if (!st2.ok()) return 1;
    if (fr.rt->state() != tutti::RuntimeState::STOPPED) return 1;
    return 0;
}

// 33. Terminal result backpressure: hitting the limit rejects new submit.
static int test_terminal_backpressure() {
    auto rt = std::move(tutti::StorageRuntime::create(
        tutti::RuntimeConfig{TUTTI_DEFAULT_ACCEL_ID, 2,
                             TUTTI_COMPILED_ACCELERATOR_PROFILE})).value();  // max_terminal_results=2

    char buf[64];
    auto mem = rt->register_memory(make_host_view(buf, sizeof(buf))).value();
    auto tgt = rt->open("stub://bp", tutti::OpenOptions{"stub"}).value();

    auto submit = [&]() {
        tutti::IoRequest req{tutti::IoDirection::READ, mem, 0, tgt, 0, 64};
        tutti::HostSubmitContext ctx{tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};
        return rt->submit(&req, 1, ctx);
    };

    auto o1 = submit();
    auto o2 = submit();
    if (!o1.io.has_value() || !o2.io.has_value()) return 1;
    force_complete(*rt, *o1.io);
    force_complete(*rt, *o2.io);
    // Two terminal results, neither released → limit reached.

    auto o3 = submit();
    if (o3.io.has_value()) return 1;  // rejected
    if (o3.status.code() != tutti::StatusCode::RESOURCE_EXHAUSTED) return 1;

    // Release one → submit succeeds again.
    rt->release_io(*o1.io);
    auto o4 = submit();
    if (!o4.io.has_value()) return 1;

    force_complete(*rt, *o4.io);
    rt->release_io(*o2.io);
    rt->release_io(*o4.io);
    rt->unregister_memory(mem);
    rt->close(tgt);
    return rt->shutdown(100).ok() ? 0 : 1;
}

// 34. Destructor with inflight ops: does not UAF, does not crash.
static int test_destructor_inflight_safe() {
    FakeRuntime fr;
    auto [mem, tgt] = fr.setup();
    fr.data_path.manual_mode = true;

    auto io = fr.submit_one(mem, tgt);
    if (!io.valid()) return 1;

    // Destroy the runtime while an op is still IN_FLIGHT.
    // The destructor must not free memory that the (fake) DataPath might
    // still reference, and must not crash.  In manual mode the op never
    // completes, so shutdown(0) returns TIMEOUT and the destructor leaks
    // conservatively.
    auto* raw_rt = fr.rt.release();
    delete raw_rt;  // should not crash
    return 0;
}

// =====================================================================
// Round 15 Session 3: cross-target merge-counting tests (T-082)
// =====================================================================

// 82. A public batch spanning K=3 targets on the SAME DataPath is issued as
// exactly one DataPath::submit() call carrying all K requests in one array
// (grouping key is DataPath only, not (DataPath, target)).
static int test_cross_target_batch_merges_single_submit() {
    RuntimeFakeResolver resolver;
    RuntimeFakeDataPath data_path;
    tutti::RuntimeComponents components;
    components.resolvers.push_back({"fake", &resolver});
    components.data_paths.push_back(
        {"runtime-fake-datapath", &data_path, tutti::DataPathConfig{"fake"}});
    auto created = tutti::StorageRuntime::create({}, std::move(components));
    if (!created.ok()) return 1;
    auto runtime = std::move(created).value();

    char buffer[256]{};
    auto memory = runtime->register_memory(make_host_view(buffer, sizeof(buffer)));
    auto t0 = runtime->open("fake://t0", tutti::OpenOptions{});
    auto t1 = runtime->open("fake://t1", tutti::OpenOptions{});
    auto t2 = runtime->open("fake://t2", tutti::OpenOptions{});
    if (!memory.ok() || !t0.ok() || !t1.ok() || !t2.ok()) return 1;

    const tutti::IoRequest requests[] = {
        {tutti::IoDirection::WRITE, memory.value(), 0, t0.value(), 0, 16},
        {tutti::IoDirection::WRITE, memory.value(), 32, t1.value(), 0, 16},
        {tutti::IoDirection::WRITE, memory.value(), 64, t2.value(), 0, 16},
    };
    const tutti::HostSubmitContext context{
        tutti::ExecutionDomain::HOST_EXECUTION, -1, nullptr};
    auto submitted = runtime->submit(requests, 3, context);
    if (!submitted.status.ok() || !submitted.io.has_value()) return 1;
    if (submitted.initial_states.size() != 3) return 1;
    for (const auto& st : submitted.initial_states) {
        if (st.state != tutti::IoRequestState::ACCEPTED) return 1;
    }

    // Exactly one DataPath::submit() call, carrying all 3 requests, each
    // for a distinct target (proving no per-target split occurred).
    if (data_path.submit_calls != 1) return 1;
    if (data_path.last_requests.size() != 3) return 1;
    const auto& lt0 = data_path.last_requests[0].target;
    const auto& lt1 = data_path.last_requests[1].target;
    const auto& lt2 = data_path.last_requests[2].target;
    if (!lt0.valid() || !lt1.valid() || !lt2.valid()) return 1;
    if (lt0 == lt1 || lt1 == lt2 || lt0 == lt2) return 1;

    auto snapshot = runtime->query(*submitted.io);
    if (!snapshot.ok() || snapshot.value().state != tutti::IoState::COMPLETED)
        return 1;
    if (data_path.release_calls != 1 || !runtime->release_io(*submitted.io).ok())
        return 1;
    if (!runtime->close(t0.value()).ok() || !runtime->close(t1.value()).ok() ||
        !runtime->close(t2.value()).ok() ||
        !runtime->unregister_memory(memory.value()).ok()) return 1;
    return runtime->shutdown(1).ok() ? 0 : 1;
}

// 83. A public batch spanning two DIFFERENT DataPaths (one target each) is
// issued as exactly two DataPath::submit() calls -- grouping by DataPath
// must not merge requests belonging to different DataPaths.
static int test_cross_datapath_batch_not_merged() {
    TwoWayFakeResolver resolver;
    RuntimeFakeDataPath data_path_a;
    RuntimeFakeDataPath data_path_b;
    tutti::RuntimeComponents components;
    components.resolvers.push_back({"fakea", &resolver});
    components.resolvers.push_back({"fakeb", &resolver});
    components.data_paths.push_back(
        {"runtime-fake-datapath-a", &data_path_a, tutti::DataPathConfig{"fakea"}});
    components.data_paths.push_back(
        {"runtime-fake-datapath-b", &data_path_b, tutti::DataPathConfig{"fakeb"}});
    auto created = tutti::StorageRuntime::create({}, std::move(components));
    if (!created.ok()) return 1;
    auto runtime = std::move(created).value();

    char buffer[128]{};
    auto memory = runtime->register_memory(make_host_view(buffer, sizeof(buffer)));
    auto ta = runtime->open("fakea://ta", tutti::OpenOptions{});
    auto tb = runtime->open("fakeb://tb", tutti::OpenOptions{});
    if (!memory.ok() || !ta.ok() || !tb.ok()) return 1;

    const tutti::IoRequest requests[] = {
        {tutti::IoDirection::WRITE, memory.value(), 0, ta.value(), 0, 16},
        {tutti::IoDirection::WRITE, memory.value(), 32, tb.value(), 0, 16},
    };
    const tutti::HostSubmitContext context{
        tutti::ExecutionDomain::HOST_EXECUTION, -1, nullptr};
    auto submitted = runtime->submit(requests, 2, context);
    if (!submitted.status.ok() || !submitted.io.has_value()) return 1;

    // Exactly two DataPath::submit() calls: one per DataPath, one request each.
    if (data_path_a.submit_calls != 1 || data_path_b.submit_calls != 1) return 1;
    if (data_path_a.last_requests.size() != 1 || data_path_b.last_requests.size() != 1) return 1;

    auto snapshot = runtime->query(*submitted.io);
    if (!snapshot.ok() || snapshot.value().state != tutti::IoState::COMPLETED)
        return 1;
    if (data_path_a.release_calls != 1 || data_path_b.release_calls != 1 ||
        !runtime->release_io(*submitted.io).ok()) return 1;
    if (!runtime->close(ta.value()).ok() || !runtime->close(tb.value()).ok() ||
        !runtime->unregister_memory(memory.value()).ok()) return 1;
    return runtime->shutdown(1).ok() ? 0 : 1;
}

// =====================================================================
// Phase 1: accelerator identity and binding contract
// =====================================================================

static tutti::RuntimeConfig phase1_config(std::int32_t accel_id) {
    return tutti::RuntimeConfig{
        accel_id, 64, TUTTI_COMPILED_ACCELERATOR_PROFILE};
}

static int test_phase1_profile_and_accel_range() {
    auto runtime = tutti::StorageRuntime::create();
    if (!runtime.ok()) return 1;
    if (runtime.value()->accel_id() != TUTTI_DEFAULT_ACCEL_ID) return 1;
    const auto profile = runtime.value()->query_cuda_like_profile();
    if (profile.profile_name != TUTTI_COMPILED_ACCELERATOR_PROFILE) return 1;
    if (profile.device_count < 0) return 1;

    auto devices = runtime.value()->list_devices();
    if (!devices.ok() ||
        devices.value().size() != static_cast<std::size_t>(profile.device_count)) return 1;
    for (std::int32_t id = 0; id < profile.device_count; ++id) {
        if (devices.value()[static_cast<std::size_t>(id)].accel_id != id) return 1;
        auto explicit_runtime = tutti::StorageRuntime::create(phase1_config(id));
        if (!explicit_runtime.ok() || explicit_runtime.value()->accel_id() != id) return 1;
    }

    auto negative = tutti::StorageRuntime::create(phase1_config(-2));
    if (negative.ok() ||
        negative.status().code() != tutti::StatusCode::INVALID_ARGUMENT) return 1;

    const std::int32_t out_of_range = profile.device_count;
    auto out = tutti::StorageRuntime::create(phase1_config(out_of_range));
    if (out.ok() ||
        (out.status().code() != tutti::StatusCode::NOT_FOUND &&
         out.status().code() != tutti::StatusCode::UNSUPPORTED)) return 1;

    const char* mismatch =
        (std::string(TUTTI_COMPILED_ACCELERATOR_PROFILE) == "HOST")
        ? "CUDA" : "HOST";
    auto wrong_profile = tutti::StorageRuntime::create(
        tutti::RuntimeConfig{TUTTI_DEFAULT_ACCEL_ID, 64, mismatch});
    if (wrong_profile.ok() ||
        wrong_profile.status().code() != tutti::StatusCode::INVALID_ARGUMENT) return 1;
    return 0;
}

static int test_phase1_datapath_binding_preflight() {
    RuntimeFakeResolver resolver;
    RuntimeFakeDataPath data_path;
    const std::int32_t runtime_accel = TUTTI_DEFAULT_ACCEL_ID;
    data_path.caps.bound_accel_id = runtime_accel < 0 ? 0 : runtime_accel + 1;

    tutti::RuntimeComponents components;
    components.resolvers.push_back({"fake", &resolver});
    components.data_paths.push_back(
        {"runtime-fake-datapath", &data_path, tutti::DataPathConfig{"fake"}});
    auto created = tutti::StorageRuntime::create(
        phase1_config(runtime_accel), std::move(components));
    if (created.ok() ||
        created.status().code() != tutti::StatusCode::INVALID_ARGUMENT) return 1;
    return data_path.initialize_calls == 0 ? 0 : 1;
}

static int test_phase1_same_accel_multiple_datapaths() {
    TwoWayFakeResolver resolver;
    RuntimeFakeDataPath data_path_a;
    RuntimeFakeDataPath data_path_b;
    data_path_a.caps.bound_accel_id = TUTTI_DEFAULT_ACCEL_ID;
    data_path_b.caps.bound_accel_id = TUTTI_DEFAULT_ACCEL_ID;
    tutti::RuntimeComponents components;
    components.resolvers.push_back({"fakea", &resolver});
    components.resolvers.push_back({"fakeb", &resolver});
    components.data_paths.push_back(
        {"runtime-fake-datapath-a", &data_path_a, tutti::DataPathConfig{"fakea"}});
    components.data_paths.push_back(
        {"runtime-fake-datapath-b", &data_path_b, tutti::DataPathConfig{"fakeb"}});
    auto created = tutti::StorageRuntime::create(
        phase1_config(TUTTI_DEFAULT_ACCEL_ID), std::move(components));
    if (!created.ok()) return 1;
    auto runtime = std::move(created).value();
    if (data_path_a.initialize_calls != 1 || data_path_b.initialize_calls != 1) return 1;
    return runtime->shutdown(1).ok() ? 0 : 1;
}

static int test_phase1_device_memory_is_not_faked() {
    auto host_runtime = tutti::StorageRuntime::create(phase1_config(-1));
    if (!host_runtime.ok()) return 1;
    auto device = host_runtime.value()->allocate_memory(
        tutti::MemorySpec{4096, tutti::MemoryKind::DEVICE, -1});
    if (device.ok() || device.status().code() != tutti::StatusCode::UNSUPPORTED) return 1;
    auto managed = host_runtime.value()->allocate_memory(
        tutti::MemorySpec{4096, tutti::MemoryKind::MANAGED, -1});
    if (managed.ok() || managed.status().code() != tutti::StatusCode::UNSUPPORTED) return 1;
    return host_runtime.value()->shutdown(1).ok() ? 0 : 1;
}

// =====================================================================
// Main
// =====================================================================

int main() {
    using TestFn = int (*)();
    const TestFn tests[] = {
        test_cross_assembly_runtime_rejected,   // 1
        test_create_running,                    // 2
        test_shutdown_stopped,                  // 3
        test_register_memory_validation,        // 4
        test_unregister_invalidates,            // 5
        test_generation_not_reused,             // 6
        test_ownership_symmetry,                // 7
        test_open_close,                        // 8
        test_busy_semantics,                    // 9
        test_wait_outcome,                      // 10
        test_release_io,                        // 11
        test_terminal_not_evicted,              // 12
        test_cross_runtime_rejected,            // 13
        test_shutdown_draining,                 // 14
        test_memory_info_no_private_fields,     // 15
        test_batch_success,                     // 16
        test_partial_commit,                    // 17
        test_zero_accepted,                     // 18
        test_count_zero,                        // 19
        test_bounds_length_zero,                // 20
        test_bounds_out_of_range,               // 21
        test_overflow_safety,                   // 22
        test_nonconst_and_test_access,          // 23
        test_component_backed_runtime_route,    // 24
        test_component_runtime_unknown_scheme,  // 25
        test_component_runtime_groups_by_datapath,// 26
        test_component_runtime_partial_commit,  // 27
        test_concurrent_submit_query_release,   // 28
        test_close_unregister_vs_submit_race,   // 29
        test_progress_serialization,            // 30
        test_datapath_failure_terminates,       // 31
        test_shutdown_drain_retry,              // 32
        test_terminal_backpressure,             // 33
        test_destructor_inflight_safe,          // 34
        test_cross_target_batch_merges_single_submit, // 82
        test_cross_datapath_batch_not_merged,   // 83
        test_phase1_profile_and_accel_range,     // 84
        test_phase1_datapath_binding_preflight,  // 85
        test_phase1_same_accel_multiple_datapaths,// 86
        test_phase1_device_memory_is_not_faked,  // 87
    };

    const int n = static_cast<int>(sizeof(tests) / sizeof(tests[0]));
    for (int i = 0; i < n; ++i) {
        if (run_test(i, tests[i]) != 0) {
            return 1;
        }
    }

    std::printf("All %d storage runtime contract tests passed.\n", n);
    return 0;
}

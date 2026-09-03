// tests/data_path_contract/data_path_contract_test.cpp
//
// Hardware-free contract test for <tutti/spi/data_path.h>.
// Plain C++17 executable; no GTest, no CUDA SDK, no hardware, no IO.
// The Tutti SPI is obtained ONLY via <tutti/spi/data_path.h>.
//
// The target resolver / runtime core are other workers' headers and may not
// exist yet. Per the task, this test provides stub definitions for the two
// forward-declared types (ResolvedTarget, ResourceProvider) so a fake DataPath
// can be driven without including those headers.

#include <tutti/memory_types.h>
#include <tutti/spi/storage_target_resolver.h>
#include <tutti/spi/data_path.h>
#include <tutti/testing/mock_data_path.h>

#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// -------------------------------------------------------------------------
// FakeDataPath is now MockDataPath from the testing kit (tutti/testing/).
// The kit provides the same full-lifecycle SPI coverage with injection points.
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Test driver
// -------------------------------------------------------------------------
namespace {

int g_failures = 0;

void check(bool cond, const char* expr, int line) {
    if (!cond) {
        std::printf("FAIL [line %d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

tutti::DataPathRequest make_req() {
    return tutti::DataPathRequest{};
}

tutti::HostSubmitContext host_ctx() {
    return tutti::HostSubmitContext{tutti::ExecutionDomain::HOST_EXECUTION, 0, nullptr};
}

} // namespace

int main() {
    // ------------------------------------------------------------------
    // 1. fake DataPath implements the full SPI (instantiable + drivable).
    // ------------------------------------------------------------------
    {
        tutti::testing::MockDataPath dp;
        tutti::ResourceProvider rp;
        CHECK(dp.initialize(tutti::DataPathConfig{"mock"}, rp).ok());
        CHECK(dp.shutdown(0).ok());
        CHECK(dp.capabilities().name == "mock");
    }

    // ------------------------------------------------------------------
    // 2. capabilities express and read all minimum fields.
    // ------------------------------------------------------------------
    {
        tutti::testing::MockDataPath dp;
        const auto& c = dp.capabilities();
        CHECK(c.source_api_version == 1u);
        CHECK(c.supports_host_execution);
        CHECK(c.supports_device_execution);
        CHECK(c.supports_host_memory);
        CHECK(c.supports_device_memory);
        CHECK(c.supports_direct);
        CHECK(c.supports_staged);
        CHECK(c.supports_read);
        CHECK(c.supports_write);
        CHECK(c.target_alignment_bytes >= 1);
        CHECK(c.memory_alignment_bytes >= 1);
        CHECK(c.length_alignment_bytes >= 1);
        CHECK(c.max_single_io_bytes > 0);
        CHECK(c.max_batch_requests > 0);
        CHECK(c.max_batch_bytes > 0);
        CHECK(c.max_in_flight_operations > 0);
        CHECK(c.supports_scatter_gather);
        CHECK(c.max_scatter_gather_entries > 0);
        CHECK(c.registration_scope == tutti::RegistrationScope::PER_DEVICE);
        CHECK(c.progress_model == tutti::ProgressModel::HOST_POLL);
        CHECK(c.supports_multi_stream);
        CHECK(c.max_concurrent_streams > 0);
        CHECK(c.max_concurrent_operations > 0);
        CHECK(c.supports_multi_gpu);
        CHECK(c.optional_target_features.size() == 2);
    }

    // ------------------------------------------------------------------
    // 3. open/register/unregister/close use distinct opaque identities.
    // ------------------------------------------------------------------
    {
        tutti::testing::MockDataPath dp;
        tutti::ResolvedTarget rt{};
        auto r1 = dp.open(rt);
        auto r2 = dp.open(rt);
        CHECK(r1.ok() && r2.ok());
        CHECK(r1.value() != r2.value());
        CHECK(r1.value().valid() && r2.value().valid());

        tutti::RegistrationDomainKey dom{"domain-0"};
        auto m1 = dp.register_memory(tutti::DataPathMemoryView{}, dom);
        auto m2 = dp.register_memory(tutti::DataPathMemoryView{}, dom);
        CHECK(m1.ok() && m2.ok());
        CHECK(m1.value() != m2.value());

        CHECK(dp.unregister_memory(m1.value()).ok());
        CHECK(dp.close(r1.value()).ok());
    }

    // ------------------------------------------------------------------
    // 4. registration domain key does not leak a controller pointer.
    // ------------------------------------------------------------------
    {
        tutti::testing::MockDataPath dp;
        tutti::ResolvedTarget rt{};
        auto t = dp.open(rt);
        CHECK(t.ok());
        auto dom = dp.registration_domain(t.value());
        CHECK(dom.ok());
        // key is a small derived string, not a hex pointer / address.
        CHECK(dom.value().value == "mock-domain");
        CHECK(dom.value().value.rfind("mock-domain", 0) == 0);
        // stable across calls for the same target.
        auto dom2 = dp.registration_domain(t.value());
        CHECK(dom2.ok());
        CHECK(dom2.value().value == dom.value().value);
    }

    // ------------------------------------------------------------------
    // 5. submit 4 requests; 3rd irreversible issue then fail.
    // ------------------------------------------------------------------
    {
        tutti::testing::MockDataPath dp;
        dp.reject_at_index = 3;  // indices 0,1,2 issued; index 3 rejected
        std::vector<tutti::DataPathRequest> reqs(4, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());

        CHECK(!out.status.ok());                       // overall non-OK
        CHECK(out.op.has_value());                     // op still present
        CHECK(out.op->valid());
        CHECK(out.initial_states.size() == 4);         // one per request, in order
        CHECK(out.initial_states[0].state == tutti::RequestState::ACCEPTED);
        CHECK(out.initial_states[1].state == tutti::RequestState::ACCEPTED);
        CHECK(out.initial_states[2].state == tutti::RequestState::ACCEPTED);
        CHECK(out.initial_states[3].state == tutti::RequestState::REJECTED);
        CHECK(out.initial_states[0].status.ok());
        CHECK(!out.initial_states[3].status.ok());

        // the first 3 (issued) remain queryable via the op.
        auto q = dp.query(*out.op);
        CHECK(q.ok());
        CHECK(q.value().state == tutti::OpState::IN_FLIGHT);
    }

    // ------------------------------------------------------------------
    // 6. op == nullopt means zero transport requests issued.
    // ------------------------------------------------------------------
    {
        tutti::testing::MockDataPath dp;
        std::vector<tutti::DataPathRequest> empty;
        auto out = dp.submit(empty.data(), 0, host_ctx());
        CHECK(out.status.ok());
        CHECK(!out.op.has_value());            // zero issued
        CHECK(out.initial_states.empty());
    }
    {
        // All rejected (fail at index 0) -> zero issued -> op null, non-OK.
        tutti::testing::MockDataPath dp;
        dp.reject_at_index = 0;
        std::vector<tutti::DataPathRequest> reqs(2, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());
        CHECK(!out.status.ok());
        CHECK(!out.op.has_value());            // zero issued
        CHECK(out.initial_states.size() == 2);
        CHECK(out.initial_states[0].state == tutti::RequestState::REJECTED);
        CHECK(out.initial_states[1].state == tutti::RequestState::REJECTED);
    }

    // ------------------------------------------------------------------
    // 7. query() does not destroy the op.
    // ------------------------------------------------------------------
    {
        tutti::testing::MockDataPath dp;
        std::vector<tutti::DataPathRequest> reqs(2, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());
        CHECK(out.op.has_value());
        auto q1 = dp.query(*out.op);
        auto q2 = dp.query(*out.op);
        CHECK(q1.ok() && q2.ok());             // still queryable after first query
        CHECK(q1.value().state == tutti::OpState::IN_FLIGHT);
        CHECK(q2.value().state == tutti::OpState::IN_FLIGHT);
        CHECK(dp.total_op_count() == 1);       // not erased
        // release before terminal must fail (op still alive).
        CHECK(dp.release(*out.op).code() == tutti::StatusCode::BUSY);
    }

    // ------------------------------------------------------------------
    // 8. release() only accepts terminal ops; non-terminal returns BUSY.
    // ------------------------------------------------------------------
    {
        tutti::testing::MockDataPath dp;
        std::vector<tutti::DataPathRequest> reqs(1, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());
        CHECK(out.op.has_value());
        // non-terminal -> BUSY
        CHECK(dp.release(*out.op).code() == tutti::StatusCode::BUSY);
        CHECK(dp.total_op_count() == 1);       // not released
        // advance to terminal
        auto p = dp.progress(tutti::ProgressBudget{16, 0});
        CHECK(p.ok());
        CHECK(p.value().operations_terminal == 1);
        auto q = dp.query(*out.op);
        CHECK(q.ok());
        CHECK(q.value().state == tutti::OpState::COMPLETED);
        // terminal -> release OK
        CHECK(dp.release(*out.op).ok());
        CHECK(dp.total_op_count() == 0);
    }

    // ------------------------------------------------------------------
    // 9 & 10. progress budget bounds work units; result counts, backoff,
    //         optional deadline.
    // ------------------------------------------------------------------
    {
        tutti::testing::MockDataPath dp;
        // three in-flight ops, one request each.
        for (int i = 0; i < 3; ++i) {
            std::vector<tutti::DataPathRequest> r(1, make_req());
            auto out = dp.submit(r.data(), r.size(), host_ctx());
            CHECK(out.op.has_value());
        }
        CHECK(dp.in_flight_op_count() == 3);

        // budget caps at 2 work units.
        auto p1 = dp.progress(tutti::ProgressBudget{2, 1000});
        CHECK(p1.ok());
        const auto& r1 = p1.value();
        CHECK(r1.work_units_consumed == 2);     // never exceeds max_work_units
        CHECK(r1.work_units_consumed <= 2);
        CHECK(r1.operations_advanced == 2);
        CHECK(r1.operations_terminal == 2);
        CHECK(r1.more_work_likely == true);     // one still in-flight
        CHECK(r1.next_poll_deadline_ns.has_value());  // backoff suggested

        CHECK(dp.in_flight_op_count() == 1);

        // drain the rest.
        auto p2 = dp.progress(tutti::ProgressBudget{16, 1000});
        CHECK(p2.ok());
        const auto& r2 = p2.value();
        CHECK(r2.work_units_consumed == 1);
        CHECK(r2.operations_terminal == 1);
        CHECK(r2.more_work_likely == false);    // nothing left
        CHECK(!r2.next_poll_deadline_ns.has_value());  // no deadline when idle
        CHECK(dp.in_flight_op_count() == 0);
    }

    // ------------------------------------------------------------------
    // 11. DEVICE_EXECUTION capability distinguishes a real completion fence
    //     from device-autonomous progress (two independent fields).
    // ------------------------------------------------------------------
    {
        tutti::testing::MockDataPath dp;
        auto& c = dp.caps;
        // profile A: real fence, not autonomous.
        c.device_completion_fence_on_caller_stream = true;
        c.device_execution_autonomous = false;
        CHECK(c.device_completion_fence_on_caller_stream);
        CHECK(!c.device_execution_autonomous);
        // profile B: autonomous, no caller-stream fence.
        c.device_completion_fence_on_caller_stream = false;
        c.device_execution_autonomous = true;
        CHECK(!c.device_completion_fence_on_caller_stream);
        CHECK(c.device_execution_autonomous);
        // the two fields are independently observable.
        CHECK(c.device_completion_fence_on_caller_stream !=
              c.device_execution_autonomous);
    }

    // ------------------------------------------------------------------
    // 12. no infinite busy-poll / no transport-completion private types.
    //     progress is bounded by the budget; all types here come from the SPI.
    // ------------------------------------------------------------------
    {
        tutti::testing::MockDataPath dp;
        std::vector<tutti::DataPathRequest> reqs(5, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());
        CHECK(out.op.has_value());
        // a zero-budget progress call consumes nothing and returns.
        auto p = dp.progress(tutti::ProgressBudget{0, 0});
        CHECK(p.ok());
        CHECK(p.value().work_units_consumed == 0);
        CHECK(p.value().operations_terminal == 0);
        // bounded: a single call never advances more than the budget allows.
        auto p2 = dp.progress(tutti::ProgressBudget{1, 0});
        CHECK(p2.ok());
        CHECK(p2.value().work_units_consumed <= 1);
    }

    // ------------------------------------------------------------------
    // 13. fake DataPath does not reuse shared scratch across two in-flight
    //     ops; each op retains its own per-op scratch.
    // ------------------------------------------------------------------
    {
        tutti::testing::MockDataPath dp;
        std::vector<tutti::DataPathRequest> a(3, make_req());
        std::vector<tutti::DataPathRequest> b(2, make_req());
        auto oa = dp.submit(a.data(), a.size(), host_ctx());
        auto ob = dp.submit(b.data(), b.size(), host_ctx());
        CHECK(oa.op.has_value() && ob.op.has_value());
        CHECK(*oa.op != *ob.op);                       // distinct identities
        CHECK(dp.total_op_count() == 2);
        CHECK(dp.in_flight_op_count() == 2);
        // per-op scratch sized by each op's accepted count; not overwritten.
        CHECK(dp.op_scratch_size(*oa.op) == 3 * 16);
        CHECK(dp.op_scratch_size(*ob.op) == 2 * 16);
        // both remain independently queryable.
        CHECK(dp.query(*oa.op).ok());
        CHECK(dp.query(*ob.op).ok());
    }

    // ------------------------------------------------------------------
    // 14. Cross-header coexistence regression test.
    //
    // This test's real purpose is that the three headers
    //   <tutti/memory_types.h>
    //   <tutti/spi/storage_target_resolver.h>
    //   <tutti/spi/data_path.h>
    // compile together in one translation unit without the
    // "multiple definition of enum class tutti::MemoryKind" error that
    // existed before DataPathMemoryKind was introduced.
    //
    // The assertions below verify the rename is correct and the public
    // contract is intact.
    // ------------------------------------------------------------------
    {
        // (a) Public MemoryKind and SPI DataPathMemoryKind are distinct types.
        static_assert(!std::is_same_v<tutti::MemoryKind,
                                      tutti::DataPathMemoryKind>,
                      "public MemoryKind and SPI DataPathMemoryKind must be "
                      "distinct types");

        // (b) Public MemoryKind still has all four values (contract intact).
        tutti::MemoryKind mk_host    = tutti::MemoryKind::HOST;
        tutti::MemoryKind mk_pinned  = tutti::MemoryKind::PINNED_HOST;
        tutti::MemoryKind mk_device  = tutti::MemoryKind::DEVICE;
        tutti::MemoryKind mk_managed = tutti::MemoryKind::MANAGED;
        CHECK(mk_host    != mk_pinned);
        CHECK(mk_pinned  != mk_device);
        CHECK(mk_device  != mk_managed);

        // (c) DataPathMemoryView::kind has type DataPathMemoryKind.
        tutti::DataPathMemoryView dpmv;
        static_assert(std::is_same_v<decltype(dpmv.kind),
                                      tutti::DataPathMemoryKind>,
                      "DataPathMemoryView::kind must be DataPathMemoryKind");
        CHECK(dpmv.kind == tutti::DataPathMemoryKind::HOST);

        // (d) All three types can be default-constructed in the same scope.
        tutti::MemoryView mv{};
        tutti::ResolvedTarget rt{};
        tutti::DataPathMemoryView dpmv2{};
        (void)mv; (void)rt; (void)dpmv2;  // suppress unused warnings
    }

    if (g_failures == 0) {
        std::printf("tutti_data_path_contract_test: all checks passed\n");
        return 0;
    }
    std::printf("tutti_data_path_contract_test: %d failure(s)\n", g_failures);
    return 1;
}

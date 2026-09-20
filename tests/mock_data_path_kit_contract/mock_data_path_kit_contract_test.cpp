// tests/mock_data_path_kit_contract/mock_data_path_kit_contract_test.cpp
//
// Kit self-contract test: proves MockDataPath satisfies the DataPath SPI
// contract. This is the "reference implementation compliance" proof.
//
// If the kit itself fails these assertions, it is not a valid reference
// implementation and any test using it is testing nothing.

#include <tutti/spi/data_path.h>
#include <tutti/spi/storage_target_resolver.h>
#include "csrc/testing/mock_data_path.h"

#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <vector>

using namespace tutti;
using namespace tutti::testing;

static int g_failures = 0;

static void check(bool cond, const char* expr, int line) {
    if (!cond) {
        std::printf("FAIL [line %d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(cond) check((cond), #cond, __LINE__)

static DataPathRequest make_req() { return DataPathRequest{}; }
static HostSubmitContext host_ctx() {
    return HostSubmitContext{ExecutionDomain::HOST_EXECUTION, 0, nullptr};
}

int main() {
    // 1. MockDataPath is instantiable and implements the full SPI.
    {
        MockDataPath dp;
        ResourceProvider rp;
        CHECK(dp.initialize(DataPathConfig{"mock"}, rp).ok());
        CHECK(dp.shutdown(0).ok());
        CHECK(dp.capabilities().name == "mock");
    }

    // 2. Capabilities cover all minimum fields.
    {
        MockDataPath dp;
        const auto& c = dp.capabilities();
        CHECK(c.source_api_version >= 1);
        CHECK(c.supports_host_execution);
        CHECK(c.supports_read);
        CHECK(c.supports_write);
        CHECK(c.max_single_io_bytes > 0);
        CHECK(c.max_batch_requests > 0);
        CHECK(c.max_in_flight_operations > 0);
    }

    // 3. Custom capabilities injection.
    {
        MockDataPath dp;
        dp.caps.name = "custom-mock";
        dp.caps.max_single_io_bytes = 8192;
        dp.caps.max_in_flight_operations = 4;
        CHECK(dp.capabilities().name == "custom-mock");
        CHECK(dp.capabilities().max_single_io_bytes == 8192);
        CHECK(dp.capabilities().max_in_flight_operations == 4);
    }

    // 4. open/close use distinct opaque identities.
    {
        MockDataPath dp;
        ResolvedTarget rt{};
        auto r1 = dp.open(rt);
        auto r2 = dp.open(rt);
        CHECK(r1.ok() && r2.ok());
        CHECK(r1.value() != r2.value());
        CHECK(r1.value().valid() && r2.value().valid());
        CHECK(dp.close(r1.value()).ok());
    }

    // 5. registration_domain returns a string key (not a pointer).
    {
        MockDataPath dp;
        ResolvedTarget rt{};
        auto t = dp.open(rt);
        CHECK(t.ok());
        auto dom = dp.registration_domain(t.value());
        CHECK(dom.ok());
        CHECK(!dom.value().value.empty());
        CHECK(dp.close(t.value()).ok());
    }

    // 6. register/unregister use distinct opaque identities.
    {
        MockDataPath dp;
        RegistrationDomainKey dom{"mock-domain"};
        auto m1 = dp.register_memory(DataPathMemoryView{}, dom);
        auto m2 = dp.register_memory(DataPathMemoryView{}, dom);
        CHECK(m1.ok() && m2.ok());
        CHECK(m1.value() != m2.value());
        CHECK(dp.unregister_memory(m1.value()).ok());
    }

    // 7. submit with partial commit: 4 requests, reject at index 3.
    {
        MockDataPath dp;
        dp.set_reject_at_index(3);
        std::vector<DataPathRequest> reqs(4, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());
        CHECK(!out.status.ok());
        CHECK(out.op.has_value());
        CHECK(out.initial_states.size() == 4);
        CHECK(out.initial_states[0].state == IoRequestState::ACCEPTED);
        CHECK(out.initial_states[3].state == IoRequestState::REJECTED);
        CHECK(out.initial_states[0].status.ok());
        CHECK(!out.initial_states[3].status.ok());
    }

    // 8. op == nullopt means zero issued.
    {
        MockDataPath dp;
        std::vector<DataPathRequest> empty;
        auto out = dp.submit(empty.data(), 0, host_ctx());
        CHECK(out.status.ok());
        CHECK(!out.op.has_value());
    }

    // 9. All rejected -> op null, non-OK.
    {
        MockDataPath dp;
        dp.set_reject_at_index(0);
        std::vector<DataPathRequest> reqs(2, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());
        CHECK(!out.status.ok());
        CHECK(!out.op.has_value());
    }

    // 10. query does not destroy the op.
    {
        MockDataPath dp;
        std::vector<DataPathRequest> reqs(2, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());
        CHECK(out.op.has_value());
        auto q1 = dp.query(*out.op);
        auto q2 = dp.query(*out.op);
        CHECK(q1.ok() && q2.ok());
        CHECK(dp.total_op_count() == 1);
    }

    // 11. release only accepts terminal ops.
    {
        MockDataPath dp;
        std::vector<DataPathRequest> reqs(1, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());
        CHECK(out.op.has_value());
        CHECK(dp.release(*out.op).code() == StatusCode::BUSY);
        dp.progress(ProgressBudget{16, 0});
        CHECK(dp.release(*out.op).ok());
    }

    // 12. progress budget bounds work units.
    {
        MockDataPath dp;
        for (int i = 0; i < 3; ++i) {
            std::vector<DataPathRequest> r(1, make_req());
            dp.submit(r.data(), r.size(), host_ctx());
        }
        CHECK(dp.in_flight_op_count() == 3);
        auto p1 = dp.progress(ProgressBudget{2, 1000});
        CHECK(p1.ok());
        CHECK(p1.value().work_units_consumed == 2);
        CHECK(p1.value().more_work_likely == true);
        CHECK(dp.in_flight_op_count() == 1);
        auto p2 = dp.progress(ProgressBudget{16, 1000});
        CHECK(p2.ok());
        CHECK(p2.value().more_work_likely == false);
    }

    // 13. manual_mode: progress does not auto-complete.
    {
        MockDataPath dp;
        dp.set_manual_mode(true);
        std::vector<DataPathRequest> reqs(1, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());
        CHECK(out.op.has_value());
        auto p = dp.progress(ProgressBudget{16, 0});
        CHECK(p.ok());
        CHECK(p.value().operations_terminal == 0);
        CHECK(dp.in_flight_op_count() == 1);
        dp.manual_complete(*out.op);
        CHECK(dp.in_flight_op_count() == 0);
    }

    // 14. fail_progress injection.
    {
        MockDataPath dp;
        dp.set_fail_progress(true);
        std::vector<DataPathRequest> reqs(1, make_req());
        dp.submit(reqs.data(), reqs.size(), host_ctx());
        auto p = dp.progress(ProgressBudget{16, 0});
        CHECK(!p.ok());
        CHECK(p.status().code() == StatusCode::DEVICE_ERROR);
    }

    // 15. fail_query injection.
    {
        MockDataPath dp;
        std::vector<DataPathRequest> reqs(1, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());
        dp.set_fail_query(true);
        auto q = dp.query(*out.op);
        CHECK(!q.ok());
        CHECK(q.status().code() == StatusCode::DEVICE_ERROR);
    }

    // 16. fail_release injection.
    {
        MockDataPath dp;
        std::vector<DataPathRequest> reqs(1, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());
        dp.progress(ProgressBudget{16, 0});
        dp.set_fail_release(true);
        CHECK(dp.release(*out.op).code() == StatusCode::DEVICE_ERROR);
    }

    // 17. fail_submit injection.
    {
        MockDataPath dp;
        dp.set_fail_submit(true);
        std::vector<DataPathRequest> reqs(2, make_req());
        auto out = dp.submit(reqs.data(), reqs.size(), host_ctx());
        CHECK(!out.status.ok());
        CHECK(!out.op.has_value());
        CHECK(out.initial_states[0].state == IoRequestState::REJECTED);
    }

    // 18. Call counters track all SPI methods.
    {
        MockDataPath dp;
        ResourceProvider rp;
        dp.initialize(DataPathConfig{"mock"}, rp);
        CHECK(dp.initialize_calls == 1);
        ResolvedTarget rt{};
        dp.open(rt);
        CHECK(dp.open_calls == 1);
        RegistrationDomainKey dom{"mock-domain"};
        dp.register_memory(DataPathMemoryView{}, dom);
        CHECK(dp.register_calls == 1);
        std::vector<DataPathRequest> reqs(1, make_req());
        dp.submit(reqs.data(), 1, host_ctx());
        CHECK(dp.submit_calls == 1);
        dp.progress(ProgressBudget{16, 0});
        CHECK(dp.progress_calls == 1);
        CHECK(dp.shutdown(0).ok());
        CHECK(dp.shutdown_calls == 1);
    }

    // 19. Per-op scratch is private (not shared).
    {
        MockDataPath dp;
        std::vector<DataPathRequest> a(3, make_req());
        std::vector<DataPathRequest> b(2, make_req());
        auto oa = dp.submit(a.data(), a.size(), host_ctx());
        auto ob = dp.submit(b.data(), b.size(), host_ctx());
        CHECK(*oa.op != *ob.op);
        CHECK(dp.op_scratch_size(*oa.op) == 3 * 16);
        CHECK(dp.op_scratch_size(*ob.op) == 2 * 16);
    }

    if (g_failures == 0) {
        std::printf("tutti_mock_data_path_kit_contract_test: all checks passed\n");
        return 0;
    }
    std::printf("tutti_mock_data_path_kit_contract_test: %d failure(s)\n", g_failures);
    return 1;
}

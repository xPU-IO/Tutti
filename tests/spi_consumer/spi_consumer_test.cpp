// tests/spi_consumer/spi_consumer_test.cpp
//
// Build-boundary consumer test for the tutti_spi INTERFACE target.
//
// Proves that linking ONLY tutti_spi provides every usage requirement needed
// to consume the in-repo SPI headers: the repository include root (so
// <tutti/spi/...> resolves), the transitive public include root (so
// <tutti/status.h> / <tutti/io_types.h> resolve), and the TUTTI_USE_<PROFILE>
// macro (so <tutti/cuda_like.h> selects the HOST shim without a CUDA SDK).
//
// This test deliberately does NOT implement a fake DataPath or fake Resolver;
// those have dedicated contract tests. It only exercises stable SPI value
// types to prove the build boundary holds. It avoids the data-path memory-kind
// enum and the memory-view kind field (which are being renamed by a concurrent
// worker) and references only stable types.

#include <tutti/spi/data_path.h>
#include <tutti/spi/storage_target_resolver.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>

namespace {

int g_failures = 0;

void check(bool cond, const char* expr, int line) {
    if (!cond) {
        std::printf("FAIL [line %d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

} // namespace

int main() {
    // ---- data_path.h: opaque identities (stable, strong, distinct) ----

    tutti::DataPathTarget t1;
    tutti::DataPathMemory m1;
    tutti::DataPathOp     o1;
    CHECK(!t1.valid());
    CHECK(!m1.valid());
    CHECK(!o1.valid());
    CHECK(t1 == tutti::DataPathTarget{});
    CHECK(!(t1 != tutti::DataPathTarget{}));
    static_assert(!std::is_convertible_v<tutti::DataPathTarget, tutti::DataPathMemory>,
                  "handle types must not cross-convert");
    static_assert(!std::is_convertible_v<tutti::DataPathOp, tutti::DataPathTarget>,
                  "handle types must not cross-convert");

    // ---- RegistrationDomainKey: opaque string, never a pointer ----
    tutti::RegistrationDomainKey dom{"domain-0"};
    CHECK(dom.value == "domain-0");
    CHECK(dom == tutti::RegistrationDomainKey{"domain-0"});
    CHECK(dom != tutti::RegistrationDomainKey{"domain-1"});

    // ---- DataPathCapabilities: hard-constraint value type ----
    tutti::DataPathCapabilities caps;
    caps.name = "consumer";
    caps.source_api_version = 1u;
    caps.supports_host_execution = true;
    caps.supports_device_execution = false;
    caps.supports_read = true;
    caps.supports_write = true;
    caps.max_single_io_bytes = 1ull << 20;
    caps.registration_scope = tutti::RegistrationScope::PER_DEVICE;
    caps.progress_model = tutti::ProgressModel::HOST_POLL;
    caps.device_completion_fence_on_caller_stream = true;
    caps.device_execution_autonomous = false;
    caps.optional_target_features = {"fence"};
    CHECK(caps.name == "consumer");
    CHECK(caps.source_api_version == 1u);
    CHECK(caps.supports_host_execution);
    CHECK(!caps.supports_device_execution);
    CHECK(caps.max_single_io_bytes == (1ull << 20));
    CHECK(caps.registration_scope == tutti::RegistrationScope::PER_DEVICE);
    CHECK(caps.progress_model == tutti::ProgressModel::HOST_POLL);
    CHECK(caps.device_completion_fence_on_caller_stream !=
          caps.device_execution_autonomous);
    CHECK(caps.optional_target_features.size() == 1);

    // ---- DataPathConfig ----
    tutti::DataPathConfig cfg{"consumer"};
    CHECK(cfg.name == "consumer");

    // ---- IoRequestInitialState / SubmitOutcome ----
    tutti::IoRequestInitialState ris;
    ris.state = tutti::IoRequestState::ACCEPTED;
    CHECK(ris.state == tutti::IoRequestState::ACCEPTED);
    tutti::SubmitOutcome out;
    out.status = tutti::Status::Ok();
    CHECK(out.status.ok());
    CHECK(!out.op.has_value());  // zero issued
    out.initial_states.resize(2);
    CHECK(out.initial_states.size() == 2);

    // ---- IoState / DataPathSnapshot ----
    tutti::DataPathSnapshot snap;
    snap.state = tutti::IoState::IN_FLIGHT;
    CHECK(snap.state == tutti::IoState::IN_FLIGHT);
    CHECK(snap.bytes_transferred == 0);

    // ---- ProgressBudget / ProgressResult ----
    tutti::ProgressBudget budget{2, 1000};
    CHECK(budget.max_work_units == 2);
    CHECK(budget.timeout_ns == 1000);
    tutti::ProgressResult pr;
    pr.work_units_consumed = 1;
    pr.more_work_likely = true;
    pr.next_poll_deadline_ns = std::uint64_t{500};
    CHECK(pr.work_units_consumed == 1);
    CHECK(pr.more_work_likely);
    CHECK(pr.next_poll_deadline_ns.has_value());

    // ---- storage_target_resolver.h: ResolveOptions / ResolvedTarget ----
    tutti::ResolveOptions opts{"file"};
    CHECK(opts.scheme == "file");

    tutti::ResolvedTarget rt;  // default empty shell, move-only
    CHECK(!rt.valid());
    CHECK(rt.source_api_version() == 0);
    CHECK(rt.logical_size() == 0);
    CHECK(rt.payload_type_id().empty());
    CHECK(rt.recommended_data_path_key().empty());

    if (g_failures == 0) {
        std::printf("tutti_spi_consumer_test: all checks passed\n");
        return 0;
    }
    std::printf("tutti_spi_consumer_test: %d failure(s)\n", g_failures);
    return 1;
}

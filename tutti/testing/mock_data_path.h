#pragma once

// tutti/testing/mock_data_path.h
//
// Reusable MockDataPath contract kit — a test-only DataPath implementation
// that covers the full SPI lifecycle (open/close, registration_domain,
// register/unregister, submit with partial commit, progress, query, release,
// shutdown drain) with controllable injection points.
//
// This is a TESTING FACILITY, not a production DataPath. It must NOT be linked
// into any production target. It depends ONLY on public/SPI headers and the
// C++17 standard library — no CUDA, no libnvm, no hardware.
//
// Usage example:
//   #include <tutti/testing/mock_data_path.h>
//   tutti::testing::MockDataPath dp;
//   dp.set_reject_at_index(3);        // reject 4th request onward
//   dp.set_manual_mode(true);         // progress() does not auto-complete
//   dp.set_fail_progress(true);       // progress() returns DEVICE_ERROR
//   dp.set_capabilities({...});       // custom capabilities
//
// Migration target: this kit supersedes the inline FakeDataPath in
// tests/data_path_contract/ and RuntimeFakeDataPath in
// tests/storage_runtime_contract/.

#include <tutti/spi/data_path.h>
#include <tutti/status.h>
#include <tutti/io_types.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tutti::testing {

// -------------------------------------------------------------------------
// MockDataPath — full-lifecycle test DataPath with injection points.
//
// Default behavior:
//   - initialize/shutdown: succeed, increment call counters.
//   - open/close: mint opaque identities, track domain keys.
//   - register/unregister: mint opaque identities.
//   - submit: accept all requests, mint an op, auto-complete on next progress.
//   - progress: complete all IN_FLIGHT ops (unless manual_mode).
//   - query: return current state.
//   - release: succeed only for terminal ops (BUSY otherwise).
//
// Injection points (public, settable before or between calls):
//   - reject_at_index: reject this 0-based index and all after (partial commit).
//   - fail_submit: submit() returns error, no op minted.
//   - fail_progress: progress() returns DEVICE_ERROR.
//   - fail_query: query() returns DEVICE_ERROR.
//   - fail_release: release() returns DEVICE_ERROR.
//   - manual_mode: progress() does NOT auto-complete ops (simulates hang).
//   - block_progress: progress() blocks until flag cleared (serialization test).
//   - capabilities: fully customizable via caps field.
//   - bytes_per_request: override the per-request bytes_transferred (default 4096).
// -------------------------------------------------------------------------
class MockDataPath : public tutti::DataPath {
public:
    static constexpr std::size_t kNoRejection = ~static_cast<std::size_t>(0);

    // ---- Configurable capabilities (mutable, set before use) ----
    tutti::DataPathCapabilities caps;

    // ---- Injection points ----
    std::size_t reject_at_index = kNoRejection;  // reject this index and after
    std::atomic<bool> fail_submit{false};
    std::atomic<bool> fail_progress{false};
    std::atomic<bool> fail_query{false};
    std::atomic<bool> fail_release{false};
    std::atomic<bool> manual_mode{false};        // progress() does not auto-complete
    std::atomic<bool> block_progress_flag{false}; // progress() blocks until cleared
    std::uint64_t bytes_per_request = 4096;     // bytes counted per accepted request

    // Domain key: constant by default (all targets share one domain).
    // Set to "mock-domain-{token}" pattern by tests that need per-target domains.
    std::string domain_key_prefix = "mock-domain";

    // ---- Call counters (for observability) ----
    int initialize_calls = 0;
    int shutdown_calls = 0;
    int open_calls = 0;
    int close_calls = 0;
    int register_calls = 0;
    int unregister_calls = 0;
    int submit_calls = 0;
    int progress_calls = 0;
    int release_calls = 0;

    // ---- Progress serialization probe ----
    std::atomic<int> progress_concurrent{0};
    std::atomic<int> progress_max_concurrent{0};

    // ---- Last submit snapshot (for test inspection) ----
    std::vector<tutti::DataPathRequest> last_requests;

    // ---- Constructor: default capabilities ----
    MockDataPath() {
        caps.name = "mock";
        caps.source_api_version = 1;
        caps.supports_host_execution = true;
        caps.supports_device_execution = true;
        caps.supports_host_memory = true;
        caps.supports_device_memory = true;
        caps.supports_direct = true;
        caps.supports_staged = true;
        caps.supports_read = true;
        caps.supports_write = true;
        caps.target_alignment_bytes = 1;
        caps.memory_alignment_bytes = 1;
        caps.length_alignment_bytes = 1;
        caps.max_single_io_bytes = 1ull << 20;
        caps.max_batch_requests = 64;
        caps.max_batch_bytes = 1ull << 24;
        caps.max_in_flight_operations = 16;
        caps.supports_scatter_gather = true;
        caps.max_scatter_gather_entries = 32;
        caps.registration_scope = RegistrationScope::PER_DEVICE;
        caps.progress_model = ProgressModel::HOST_POLL;
        caps.device_completion_fence_on_caller_stream = true;
        caps.device_execution_autonomous = false;
        caps.supports_multi_stream = true;
        caps.max_concurrent_streams = 4;
        caps.max_concurrent_operations = 16;
        caps.bound_accel_id = TUTTI_DEFAULT_ACCEL_ID;
        caps.supports_multi_gpu = true;
        caps.supports_cross_device = false;
        caps.optional_target_features = {"feature_a", "feature_b"};
    }

    ~MockDataPath() override = default;

    // ---- Convenience setters (fluent style) ----
    MockDataPath& set_reject_at_index(std::size_t idx) {
        reject_at_index = idx;
        return *this;
    }
    MockDataPath& set_manual_mode(bool v) { manual_mode.store(v); return *this; }
    MockDataPath& set_fail_progress(bool v) { fail_progress.store(v); return *this; }
    MockDataPath& set_fail_query(bool v) { fail_query.store(v); return *this; }
    MockDataPath& set_fail_release(bool v) { fail_release.store(v); return *this; }
    MockDataPath& set_fail_submit(bool v) { fail_submit.store(v); return *this; }

    // ---- Manual op completion (for manual_mode tests) ----
    void manual_complete(tutti::DataPathOp op) {
        std::lock_guard<std::mutex> lock(ops_mtx_);
        auto it = ops_.find(op.token());
        if (it != ops_.end()) {
            it->second.state = OpState::COMPLETED;
            it->second.terminal_status = Status::Ok();
            it->second.bytes_transferred =
                it->second.accepted_count * bytes_per_request;
        }
    }

    void manual_fail(tutti::DataPathOp op, Status status) {
        std::lock_guard<std::mutex> lock(ops_mtx_);
        auto it = ops_.find(op.token());
        if (it != ops_.end()) {
            it->second.state = OpState::FAILED;
            it->second.terminal_status = status;
        }
    }

    // ---- Unblock progress (for block_progress_flag tests) ----
    void unblock_progress() {
        {
            std::lock_guard<std::mutex> lock(block_progress_mtx_);
            block_progress_flag.store(false);
        }
        block_progress_cv_.notify_all();
    }

    // ---- Test-only inspectors ----
    std::size_t total_op_count() const {
        std::lock_guard<std::mutex> lock(ops_mtx_);
        return ops_.size();
    }
    std::size_t in_flight_op_count() const {
        std::lock_guard<std::mutex> lock(ops_mtx_);
        std::size_t n = 0;
        for (const auto& kv : ops_) {
            if (kv.second.state == OpState::IN_FLIGHT) ++n;
        }
        return n;
    }

    // =====================================================================
    // DataPath SPI implementation
    // =====================================================================

    const DataPathCapabilities& capabilities() const override {
        return caps;
    }

    Status initialize(const DataPathConfig&, ResourceProvider&) override {
        ++initialize_calls;
        return Status::Ok();
    }

    Status shutdown(std::uint64_t) override {
        ++shutdown_calls;
        return Status::Ok();
    }

    Result<DataPathTarget> open(const ResolvedTarget&) override {
        ++open_calls;
        std::uint64_t tok = next_target_++;
        // Domain key: constant by default (all targets share one domain,
        // so memory is registered once per device, not per target).
        // Tests can override by setting domain_key_prefix.
        std::lock_guard<std::mutex> lock(ops_mtx_);
        target_domains_[tok] = domain_key_prefix;
        return detail::SpiIdentityMint::mint<detail::DataPathTargetTag>(tok, 1);
    }

    Status close(DataPathTarget target) override {
        ++close_calls;
        std::lock_guard<std::mutex> lock(ops_mtx_);
        target_domains_.erase(target.token());
        return Status::Ok();
    }

    Result<RegistrationDomainKey> registration_domain(
        DataPathTarget target) const override {
        std::lock_guard<std::mutex> lock(ops_mtx_);
        auto it = target_domains_.find(target.token());
        if (it == target_domains_.end()) {
            return Result<RegistrationDomainKey>::Failure(
                Status(StatusCode::NOT_FOUND, "unknown target"));
        }
        return RegistrationDomainKey{it->second};
    }

    Result<DataPathMemory> register_memory(
        const DataPathMemoryView&,
        const RegistrationDomainKey&) override {
        ++register_calls;
        return detail::SpiIdentityMint::mint<detail::DataPathMemoryTag>(next_memory_++, 1);
    }

    Status unregister_memory(DataPathMemory) override {
        ++unregister_calls;
        return Status::Ok();
    }

    SubmitOutcome submit(const DataPathRequest* requests,
                         std::size_t count,
                         const HostSubmitContext&) override {
        ++submit_calls;
        last_requests.assign(requests, requests + count);

        SubmitOutcome out;
        out.initial_states.resize(count);

        if (count == 0) {
            out.status = Status::Ok();
            return out;
        }

        if (fail_submit.load()) {
            out.status = Status(StatusCode::DEVICE_ERROR, "injected submit failure");
            for (std::size_t i = 0; i < count; ++i) {
                out.initial_states[i].state = RequestState::REJECTED;
                out.initial_states[i].status = out.status;
            }
            return out;
        }

        std::size_t accepted = 0;
        bool failed = false;
        for (std::size_t i = 0; i < count; ++i) {
            if (failed || i == reject_at_index) {
                failed = true;
                out.initial_states[i].state = RequestState::REJECTED;
                out.initial_states[i].status = Status(
                    StatusCode::RESOURCE_EXHAUSTED, "injected rejection");
                continue;
            }
            out.initial_states[i].state = RequestState::ACCEPTED;
            out.initial_states[i].status = Status::Ok();
            ++accepted;
        }

        if (accepted == 0) {
            out.status = (count == 0)
                ? Status::Ok()
                : Status(StatusCode::RESOURCE_EXHAUSTED, "all rejected");
            return out;
        }

        // Mint op with per-op private scratch (never shared between in-flight ops).
        std::uint64_t op_tok = next_op_++;
        {
            std::lock_guard<std::mutex> lock(ops_mtx_);
            OpRecord rec;
            rec.state = OpState::IN_FLIGHT;
            rec.accepted_count = accepted;
            rec.bytes_transferred = 0;
            rec.scratch.assign(accepted * 16, 0);
            ops_[op_tok] = std::move(rec);
        }
        out.op = detail::SpiIdentityMint::mint<detail::DataPathOpTag>(op_tok, 1);
        out.status = failed
            ? Status(StatusCode::RESOURCE_EXHAUSTED, "partial submit failure")
            : Status::Ok();
        return out;
    }

    Result<ProgressResult> progress(ProgressBudget budget) override {
        ++progress_calls;

        // Probe: count concurrent progress() invocations.
        int cur = ++progress_concurrent;
        int prev_max = progress_max_concurrent.load();
        while (cur > prev_max &&
               !progress_max_concurrent.compare_exchange_weak(prev_max, cur)) {}

        // Optional block for serialization tests.
        if (block_progress_flag.load()) {
            std::unique_lock<std::mutex> lk(block_progress_mtx_);
            block_progress_cv_.wait(lk, [this]{ return !block_progress_flag.load(); });
        }

        if (fail_progress.load()) {
            --progress_concurrent;
            return Result<ProgressResult>::Failure(
                Status(StatusCode::DEVICE_ERROR, "injected progress failure"));
        }

        ProgressResult result;
        std::uint64_t remaining = budget.max_work_units;

        {
            std::lock_guard<std::mutex> lock(ops_mtx_);
            for (auto& kv : ops_) {
                if (remaining == 0) break;
                OpRecord& rec = kv.second;
                if (rec.state == OpState::IN_FLIGHT && !manual_mode.load()) {
                    rec.state = OpState::COMPLETED;
                    rec.terminal_status = Status::Ok();
                    rec.bytes_transferred = rec.accepted_count * bytes_per_request;
                    ++result.work_units_consumed;
                    ++result.operations_advanced;
                    ++result.operations_terminal;
                    --remaining;
                }
            }
        }

        // Check if any ops remain in flight.
        bool more = false;
        {
            std::lock_guard<std::mutex> lock(ops_mtx_);
            for (const auto& kv : ops_) {
                if (kv.second.state == OpState::IN_FLIGHT) { more = true; break; }
            }
        }
        result.more_work_likely = more;
        result.next_poll_deadline_ns = more
            ? std::optional<std::uint64_t>(1000)
            : std::nullopt;

        --progress_concurrent;
        return result;
    }

    Result<DataPathSnapshot> query(DataPathOp op) const override {
        std::lock_guard<std::mutex> lock(ops_mtx_);
        auto it = ops_.find(op.token());
        if (it == ops_.end()) {
            return Result<DataPathSnapshot>::Failure(
                Status(StatusCode::NOT_FOUND, "unknown op"));
        }
        if (fail_query.load()) {
            return Result<DataPathSnapshot>::Failure(
                Status(StatusCode::DEVICE_ERROR, "injected query failure"));
        }
        const OpRecord& rec = it->second;
        DataPathSnapshot snap;
        snap.state = rec.state;
        snap.status = rec.terminal_status;
        snap.bytes_transferred = rec.bytes_transferred;
        return snap;
    }

    Status release(DataPathOp op) override {
        if (fail_release.load()) {
            return Status(StatusCode::DEVICE_ERROR, "injected release failure");
        }
        std::lock_guard<std::mutex> lock(ops_mtx_);
        auto it = ops_.find(op.token());
        if (it == ops_.end()) {
            return Status(StatusCode::NOT_FOUND, "unknown op");
        }
        if (it->second.state == OpState::IN_FLIGHT) {
            return Status(StatusCode::BUSY, "op not terminal");
        }
        ++release_calls;
        ops_.erase(it);
        return Status::Ok();
    }

    // ---- Test-only: per-op scratch observability ----
    std::size_t op_scratch_size(DataPathOp op) const {
        std::lock_guard<std::mutex> lock(ops_mtx_);
        auto it = ops_.find(op.token());
        return it == ops_.end() ? 0 : it->second.scratch.size();
    }

private:
    struct OpRecord {
        OpState state = OpState::IN_FLIGHT;
        Status terminal_status;  // OK while IN_FLIGHT
        std::uint64_t bytes_transferred = 0;
        std::size_t accepted_count = 0;
        std::vector<char> scratch;  // per-op private; never shared
    };

    std::uint64_t next_target_ = 1;
    std::uint64_t next_memory_ = 1;
    std::uint64_t next_op_ = 1;

    mutable std::mutex ops_mtx_;
    std::unordered_map<std::uint64_t, OpRecord> ops_;
    std::unordered_map<std::uint64_t, std::string> target_domains_;

    std::mutex block_progress_mtx_;
    std::condition_variable block_progress_cv_;
};

} // namespace tutti::testing

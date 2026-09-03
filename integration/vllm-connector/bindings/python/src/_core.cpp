// integration/vllm-connector/bindings/python/src/_core.cpp
//
// pybind11 extension `tutti_runtime._core`: wraps the tutti StorageRuntime
// public facade (tutti/storage_runtime.h) and the preset assembly factories
// (tutti/presets/local_nvme.h).
//
// Handles are passed to Python as opaque std::uint64_t tickets minted here;
// the bit layout of TargetHandle/MemoryHandle/IoHandle is never exposed.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tutti/presets/local_nvme.h>
#include <tutti/storage_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using tutti::IoHandle;
using tutti::MemoryHandle;
using tutti::TargetHandle;

namespace {

std::string status_code_str(tutti::StatusCode code) {
    switch (code) {
        case tutti::StatusCode::OK: return "OK";
        case tutti::StatusCode::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case tutti::StatusCode::OUT_OF_RANGE: return "OUT_OF_RANGE";
        case tutti::StatusCode::NOT_FOUND: return "NOT_FOUND";
        case tutti::StatusCode::UNSUPPORTED: return "UNSUPPORTED";
        case tutti::StatusCode::NOT_READY: return "NOT_READY";
        case tutti::StatusCode::BUSY: return "BUSY";
        case tutti::StatusCode::RESOURCE_EXHAUSTED: return "RESOURCE_EXHAUSTED";
        case tutti::StatusCode::TIMEOUT: return "TIMEOUT";
        case tutti::StatusCode::DEVICE_ERROR: return "DEVICE_ERROR";
        case tutti::StatusCode::DATA_LOSS: return "DATA_LOSS";
        case tutti::StatusCode::INTERNAL: return "INTERNAL";
    }
    return "UNKNOWN";
}

[[noreturn]] void throw_status(const std::string& prefix,
                               const tutti::Status& status) {
    throw std::runtime_error(prefix + ": " + status_code_str(status.code()) +
                            ": " + status.message());
}

struct SubmitResult {
    bool status_ok = false;
    std::string status_msg;
    std::optional<std::uint64_t> io_handle;  // None = all rejected
    std::vector<bool> initial_states;
    std::vector<std::int64_t> rejected;  // request indices, for resubmit
};

struct WaitResult {
    std::string observation;
    std::string state;
    std::uint64_t confirmed_bytes = 0;
    bool timeout_seen = false;
    std::vector<std::uint32_t> failed_request_indices;
    std::string failure_scope = "NONE";
    std::optional<std::string> failure_kind;
    std::optional<std::uint32_t> first_failed_entry;
    std::optional<std::uint32_t> raw_cq_status;
    std::string message;
};

const char* failure_kind_str(tutti::IoFailureKind kind) {
    switch (kind) {
        case tutti::IoFailureKind::NONE: return "NONE";
        case tutti::IoFailureKind::RESOLVE_LBA: return "RESOLVE_LBA";
        case tutti::IoFailureKind::CQ_TIMEOUT: return "CQ_TIMEOUT";
        case tutti::IoFailureKind::NVME_CQ_ERROR: return "NVME_CQ_ERROR";
        case tutti::IoFailureKind::CUDA_QUERY_ERROR: return "CUDA_QUERY_ERROR";
        case tutti::IoFailureKind::STATUS_D2H_ERROR: return "STATUS_D2H_ERROR";
        case tutti::IoFailureKind::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

const char* failure_scope_str(tutti::IoFailureScope scope) {
    switch (scope) {
        case tutti::IoFailureScope::NONE: return "NONE";
        case tutti::IoFailureScope::REQUEST_INDICES: return "REQUEST_INDICES";
        case tutti::IoFailureScope::WHOLE_OPERATION: return "WHOLE_OPERATION";
    }
    return "WHOLE_OPERATION";
}

class PyRuntime {
public:
    explicit PyRuntime(std::unique_ptr<tutti::StorageRuntime> rt, bool stub,
                       std::uint64_t max_batch, std::uint64_t max_inflight,
                       std::int32_t bound_accel_id,
                       std::uint64_t max_concurrent_streams)
        : rt_(std::move(rt)), stub_mode_(stub),
          caps_max_batch_(max_batch), caps_max_inflight_(max_inflight),
          caps_bound_accel_id_(bound_accel_id),
          caps_max_concurrent_streams_(max_concurrent_streams) {}

    // ---- caps ----
    std::unordered_map<std::string, std::uint64_t> caps() const {
        // stub mode: alignment = 1 (byte-aligned), limits 0 (= unlimited).
        // component mode: the public facade exposes no DataPath capability
        // query; report the preset-supplied limits and conservative
        // alignment = 1.
        std::unordered_map<std::string, std::uint64_t> caps{
            {"target_alignment_bytes", 1},
            {"memory_alignment_bytes", 1},
            {"length_alignment_bytes", 1},
            {"max_single_io_bytes", 0},
            {"max_batch_requests", caps_max_batch_},
            {"max_in_flight_operations", caps_max_inflight_},
            {"supports_multi_stream", caps_max_concurrent_streams_ >= 2},
            {"max_concurrent_streams", caps_max_concurrent_streams_},
        };
        if (caps_bound_accel_id_ >= 0) {
            caps["bound_accel_id"] = static_cast<std::uint64_t>(
                caps_bound_accel_id_);
        }
        return caps;
    }

    // ---- targets ----
    std::vector<std::uint64_t> open_batch(const std::vector<std::string>& uris) {
        std::vector<tutti::Result<TargetHandle>> results =
            rt_->open_batch(uris, tutti::OpenOptions{});
        for (std::size_t i = 0; i < results.size(); ++i) {
            if (results[i].ok()) continue;
            for (auto& result : results) {
                if (result.ok()) (void)rt_->close(result.value());
            }
            throw_status("open_batch failed at uri[" + std::to_string(i) +
                         "] (" + uris[i] + ")", results[i].status());
        }
        std::vector<std::uint64_t> tickets;
        tickets.reserve(results.size());
        for (std::size_t i = 0; i < results.size(); ++i) {
            tickets.push_back(mint_target_(results[i].value()));
        }
        return tickets;
    }

    void close_target(std::uint64_t ticket) {
        const TargetHandle handle = lookup_target_(ticket, "target");
        tutti::Status status = rt_->close(handle);
        if (!status.ok()) throw_status("close_target failed", status);
        targets_.erase(ticket);
    }

    void close_batch(const std::vector<std::uint64_t>& tickets) {
        std::vector<TargetHandle> handles;
        handles.reserve(tickets.size());
        for (std::uint64_t ticket : tickets) {
            handles.push_back(lookup_target_(ticket, "target"));
        }
        for (std::size_t i = 0; i < tickets.size(); ++i) {
            tutti::Status status = rt_->close(handles[i]);
            if (!status.ok()) {
                throw_status("close_batch failed at ticket[" +
                             std::to_string(i) + "]", status);
            }
            targets_.erase(tickets[i]);
        }
    }

    // ---- memory ----
    std::uint64_t register_memory(std::uintptr_t addr, std::uint64_t size,
                                  const std::string& kind,
                                  std::int32_t accel_id,
                                  std::uint64_t io_granularity) {
        tutti::MemoryView view;
        view.address = reinterpret_cast<void*>(addr);
        view.size = size;
        if (kind == "host") {
            view.expected_kind = tutti::MemoryKind::HOST;
        } else if (kind == "device") {
            view.expected_kind = tutti::MemoryKind::DEVICE;
        } else {
            throw py::value_error(
                "register_memory kind must be 'host' or 'device', got '" +
                kind + "'");
        }
        view.ownership = tutti::MemoryOwnership::CALLER_OWNED;
        view.expected_accel_id = accel_id;
        view.expected_profile = "";
        view.io_granularity = io_granularity;
        auto result = rt_->register_memory(view);
        if (!result.ok()) {
            throw_status("register_memory failed", result.status());
        }
        return mint_memory_(result.value());
    }

    void unregister_memory(std::uint64_t ticket) {
        const MemoryHandle handle = lookup_memory_(ticket, "memory");
        tutti::Status status;
        {
            py::gil_scoped_release release;
            status = rt_->unregister_memory(handle);
        }
        if (!status.ok()) {
            throw_status(
                "unregister_memory failed for memory ticket " +
                std::to_string(ticket), status);
        }
        // Preserve the ticket when Runtime rejects the unregister (for
        // example BUSY due to in-flight I/O), so callers can drain and retry.
        memories_.erase(ticket);
    }

    // ---- submit ----
    SubmitResult submit(py::sequence requests, std::int32_t accel_id,
                        py::object stream, const std::string& execution) {
        tutti::ExecutionDomain domain;
        if (execution == "device") {
            domain = tutti::ExecutionDomain::DEVICE_EXECUTION;
            if (stream.is_none()) {
                throw py::value_error(
                    "execution='device' requires a non-None stream "
                    "(cudaStream_t pointer)");
            }
        } else if (execution == "host") {
            domain = tutti::ExecutionDomain::HOST_EXECUTION;
        } else {
            throw py::value_error(
                "execution must be 'device' or 'host', got '" + execution +
                "'");
        }

        std::vector<tutti::IoRequest> reqs = parse_requests_(requests);

        tutti::HostSubmitContext ctx{};
        ctx.execution_domain = domain;
        ctx.accel_id = accel_id;
        ctx.stream = stream.is_none()
            ? nullptr
            : reinterpret_cast<cudaStream_t>(
                  checked_int<std::uintptr_t>(stream, "stream"));

        tutti::IoSubmitOutcome outcome =
            rt_->submit(reqs.data(), reqs.size(), ctx);

        SubmitResult out;
        out.status_ok = outcome.status.ok();
        out.status_msg = status_code_str(outcome.status.code()) + ": " +
                         outcome.status.message();
        if (outcome.io.has_value()) {
            out.io_handle = mint_io_(outcome.io.value());
        }
        for (std::size_t i = 0; i < outcome.initial_states.size(); ++i) {
            const bool accepted =
                outcome.initial_states[i].state ==
                tutti::IoRequestState::ACCEPTED;
            out.initial_states.push_back(accepted);
            if (!accepted) out.rejected.push_back(static_cast<std::int64_t>(i));
        }
        return out;
    }

    // ---- io lifecycle ----
    void release_io(std::uint64_t ticket) {
        const IoHandle handle = lookup_io_(ticket);
        // Capture terminal detail before Runtime invalidates the handle. The
        // ticket-level cache keeps diagnostics observable after release_io().
        WaitResult observed = wait_result(ticket, 0);
        tutti::Status status = rt_->release_io(handle);
        if (!status.ok()) {
            throw_status("release_io failed", status);
        }
        if (observed.observation == "OK" &&
            (observed.state == "COMPLETED" || observed.state == "FAILED")) {
            terminal_results_[ticket] = std::move(observed);
        }
        ios_.erase(ticket);
    }

    std::pair<std::string, std::string> wait(std::uint64_t ticket,
                                             std::uint64_t timeout_ms) {
        WaitResult result = wait_result(ticket, timeout_ms);
        return {result.observation, result.state};
    }

    WaitResult wait_result(std::uint64_t ticket,
                           std::uint64_t timeout_ms) {
        auto retained = terminal_results_.find(ticket);
        if (retained != terminal_results_.end()) {
            return retained->second;
        }
        const IoHandle handle = lookup_io_(ticket);
        tutti::WaitOutcome outcome;
        {
            py::gil_scoped_release release;
            outcome = rt_->wait_result(handle, timeout_ms);
        }
        WaitResult result;
        if (!outcome.observation_status.ok()) {
            result.observation = status_code_str(
                outcome.observation_status.code());
            result.message = outcome.observation_status.message();
            return result;
        }
        if (!outcome.result.has_value()) {
            result.observation = "INTERNAL";
            result.message = "terminal wait returned no result";
            return result;
        }
        result.observation = "OK";
        switch (outcome.result->state) {
            case tutti::IoState::COMPLETED: result.state = "COMPLETED"; break;
            case tutti::IoState::FAILED: result.state = "FAILED"; break;
            case tutti::IoState::IN_FLIGHT: result.state = "IN_FLIGHT"; break;
        }
        const auto& detail = outcome.result->detail;
        result.confirmed_bytes = detail.confirmed_bytes;
        result.timeout_seen = detail.timeout_seen;
        result.failed_request_indices = detail.failed_request_indices;
        result.failure_scope = failure_scope_str(detail.failure_scope);
        if (detail.failure_kind != tutti::IoFailureKind::NONE) {
            result.failure_kind = failure_kind_str(detail.failure_kind);
            result.raw_cq_status = detail.raw_cq_status;
        }
        if (detail.first_failed_entry != UINT32_MAX) {
            result.first_failed_entry = detail.first_failed_entry;
        }
        result.message = outcome.result->status.message();
        terminal_results_[ticket] = result;
        return result;
    }

    WaitResult wait_detail(std::uint64_t ticket,
                           std::uint64_t timeout_ms) {
        return wait_result(ticket, timeout_ms);
    }

    void shutdown(std::uint64_t timeout_ms) {
        tutti::Status status;
        {
            py::gil_scoped_release release;
            status = rt_->shutdown(timeout_ms);
        }
        if (!status.ok()) {
            throw_status("shutdown failed", status);
        }
        targets_.clear();
        memories_.clear();
        ios_.clear();
    }

    // ---- testing-only hook (stub mode) ----
    void testing_force_complete(std::uint64_t ticket,
                                const std::string& state) {
        if (!stub_mode_) {
            throw std::runtime_error(
                "testing_force_complete is only available in stub mode "
                "(component-backed IO is completed by its DataPath)");
        }
        tutti::IoState terminal;
        if (state == "COMPLETED") {
            terminal = tutti::IoState::COMPLETED;
        } else if (state == "FAILED") {
            terminal = tutti::IoState::FAILED;
        } else {
            throw py::value_error(
                "state must be 'COMPLETED' or 'FAILED', got '" + state + "'");
        }
        const IoHandle handle = lookup_io_(ticket);
        tutti::Status status =
            tutti::testing::StorageRuntimeTestAccess::force_complete_io(
                *rt_, handle, terminal);
        if (!status.ok()) {
            throw_status("testing_force_complete failed", status);
        }
    }

    void testing_inject_next_read_nvme_error(std::uint32_t) {
        throw std::runtime_error(
            "read NVMe error injection is disabled in this binding build");
    }

private:
    std::vector<tutti::IoRequest> parse_requests_(py::sequence requests) const {
        std::vector<tutti::IoRequest> reqs;
        reqs.reserve(py::len(requests));
        for (std::size_t i = 0; i < py::len(requests); ++i) {
            py::sequence item = py::reinterpret_borrow<py::sequence>(
                requests.attr("__getitem__")(i));
            if (py::len(item) != 6)
                throw py::value_error("step/request item must be a 6-tuple");
            const std::string index = "request[" + std::to_string(i) + "].";
            tutti::IoRequest req{};
            req.target = lookup_target_(
                checked_int<uint64_t>(item[0], index + "target"), "target");
            req.target_offset = checked_int<uint64_t>(item[1], index + "target_offset");
            req.memory = lookup_memory_(
                checked_int<uint64_t>(item[2], index + "memory"), "memory");
            req.memory_offset = checked_int<uint64_t>(item[3], index + "memory_offset");
            req.length = checked_int<uint64_t>(item[4], index + "length");
            const std::string direction = checked_str(item[5], index + "direction");
            if (direction == "read") req.direction = tutti::IoDirection::READ;
            else if (direction == "write") req.direction = tutti::IoDirection::WRITE;
            else throw py::value_error(index + "invalid direction");
            reqs.push_back(req);
        }
        return reqs;
    }

    SubmitResult submit_result_(tutti::IoSubmitOutcome outcome) {
        SubmitResult out;
        out.status_ok = outcome.status.ok();
        out.status_msg = status_code_str(outcome.status.code()) + ": " +
                         outcome.status.message();
        if (outcome.io.has_value()) out.io_handle = mint_io_(outcome.io.value());
        for (std::size_t i = 0; i < outcome.initial_states.size(); ++i) {
            const bool accepted = outcome.initial_states[i].state ==
                                  tutti::IoRequestState::ACCEPTED;
            out.initial_states.push_back(accepted);
            if (!accepted) out.rejected.push_back(static_cast<std::int64_t>(i));
        }
        return out;
    }
    template <typename T>
    static T checked_int(const py::handle& obj, const std::string& name) {
        if (py::isinstance<py::bool_>(obj) || !py::isinstance<py::int_>(obj)) {
            throw py::value_error(name + " must be an int");
        }
        return obj.cast<T>();
    }

    static std::string checked_str(const py::handle& obj,
                                   const std::string& name) {
        if (!py::isinstance<py::str>(obj)) {
            throw py::value_error(name + " must be a str");
        }
        return obj.cast<std::string>();
    }

    std::uint64_t mint_target_(const TargetHandle& h) {
        const std::uint64_t t = next_ticket_++;
        targets_.emplace(t, h);
        return t;
    }
    std::uint64_t mint_memory_(const MemoryHandle& h) {
        const std::uint64_t t = next_ticket_++;
        memories_.emplace(t, h);
        return t;
    }
    std::uint64_t mint_io_(const IoHandle& h) {
        const std::uint64_t t = next_ticket_++;
        ios_.emplace(t, h);
        return t;
    }

    const TargetHandle& lookup_target_(std::uint64_t t,
                                       const char* what) const {
        auto it = targets_.find(t);
        if (it == targets_.end()) {
            throw std::runtime_error(std::string("unknown ") + what +
                                    " handle: " + std::to_string(t));
        }
        return it->second;
    }
    const MemoryHandle& lookup_memory_(std::uint64_t t,
                                       const char* what) const {
        auto it = memories_.find(t);
        if (it == memories_.end()) {
            throw std::runtime_error(std::string("unknown ") + what +
                                    " handle: " + std::to_string(t));
        }
        return it->second;
    }
    IoHandle lookup_io_(std::uint64_t t) const {
        auto it = ios_.find(t);
        if (it == ios_.end()) {
            throw std::runtime_error("unknown io handle: " + std::to_string(t));
        }
        return it->second;
    }

    std::unique_ptr<tutti::StorageRuntime> rt_;
    bool stub_mode_;
    std::uint64_t caps_max_batch_;
    std::uint64_t caps_max_inflight_;
    std::int32_t caps_bound_accel_id_;
    std::uint64_t caps_max_concurrent_streams_;
    std::uint64_t next_ticket_ = 1;
    std::unordered_map<std::uint64_t, TargetHandle> targets_;
    std::unordered_map<std::uint64_t, MemoryHandle> memories_;
    std::unordered_map<std::uint64_t, IoHandle> ios_;
    std::unordered_map<std::uint64_t, WaitResult> terminal_results_;
};

// ---------------------------------------------------------------------------
// Preset dict parsing: unknown key / wrong type raise ValueError naming the
// offending key. Device-selection facts (pci_bdf / mount_path) are required;
// the character-device path is resolved from the BDF by the C++ preset
// assembler. Hardware geometry and queue-budget knobs are optional and
// fall back to the C++ struct defaults (single source of truth in
// tutti/presets/local_nvme.h).
// ---------------------------------------------------------------------------

[[noreturn]] void value_error(const std::string& msg) {
    throw py::value_error(msg);
}

py::dict checked_dict(const py::handle& obj, const std::string& name) {
    if (!py::isinstance<py::dict>(obj)) {
        value_error("preset key '" + name + "' must be a dict");
    }
    return py::reinterpret_borrow<py::dict>(obj);
}

std::int64_t get_int_field(const py::dict& d, const std::string& key) {
    if (!d.contains(key.c_str())) {
        value_error("missing preset key: '" + key + "'");
    }
    const py::object v = d[key.c_str()];
    if (py::isinstance<py::bool_>(v) || !py::isinstance<py::int_>(v)) {
        value_error("preset key '" + key + "' must be an int");
    }
    return v.cast<std::int64_t>();
}

// Optional int: keep the C++ struct default when the key is absent.
template <typename T>
void opt_int_field(const py::dict& d, const std::string& key, T& target) {
    if (!d.contains(key.c_str())) { return; }
    const py::object v = d[key.c_str()];
    if (py::isinstance<py::bool_>(v) || !py::isinstance<py::int_>(v)) {
        value_error("preset key '" + key + "' must be an int");
    }
    target = static_cast<T>(v.cast<std::int64_t>());
}

void opt_bool_field(const py::dict& d, const std::string& key, bool& target) {
    if (!d.contains(key.c_str())) return;
    const py::object v = d[key.c_str()];
    if (!py::isinstance<py::bool_>(v)) {
        value_error("preset key '" + key + "' must be a bool");
    }
    target = v.cast<bool>();
}

std::string get_str_field(const py::dict& d, const std::string& key) {
    if (!d.contains(key.c_str())) {
        value_error("missing preset key: '" + key + "'");
    }
    const py::object v = d[key.c_str()];
    if (!py::isinstance<py::str>(v)) {
        value_error("preset key '" + key + "' must be a str");
    }
    return v.cast<std::string>();
}

// Optional str: keep the C++ struct default when the key is absent.
void opt_str_field(const py::dict& d, const std::string& key,
                   std::string& target) {
    if (!d.contains(key.c_str())) { return; }
    const py::object v = d[key.c_str()];
    if (!py::isinstance<py::str>(v)) {
        value_error("preset key '" + key + "' must be a str");
    }
    target = v.cast<std::string>();
}

py::dict get_dict_field(const py::dict& d, const std::string& key) {
    if (!d.contains(key.c_str())) {
        value_error("missing preset key: '" + key + "'");
    }
    return checked_dict(d[key.c_str()], key);
}

void check_unknown_keys(const py::dict& d,
                        const std::vector<std::string>& expected,
                        const std::string& where) {
    for (const auto& item : d) {
        const std::string key = item.first.cast<std::string>();
        bool known = false;
        for (const auto& e : expected) {
            if (e == key) { known = true; break; }
        }
        if (!known) {
            value_error("unknown preset key in " + where + ": '" + key + "'");
        }
    }
}

tutti::presets::NvmeDeviceConfig parse_device(const py::dict& d) {
    check_unknown_keys(
        d,
        {"pci_bdf", "backing_device", "mount_path", "namespace_id",
         "block_size"},
        "device dict");
    tutti::presets::NvmeDeviceConfig dev;
    dev.pci_bdf = get_str_field(d, "pci_bdf");
    opt_str_field(d, "backing_device", dev.backing_device);
    dev.mount_path = get_str_field(d, "mount_path");
    opt_int_field(d, "namespace_id", dev.namespace_id);
    opt_int_field(d, "block_size", dev.block_size);
    return dev;
}

tutti::presets::LocalNvmePreset parse_local_preset(const py::dict& d) {
    check_unknown_keys(
        d,
        {"device", "gpu_id", "num_queues", "max_batch_entries",
         "max_in_flight_operations", "threads_per_block",
         "handle_cache_capacity", "prp_cache_capacity"},
        "local nvme preset");
    tutti::presets::LocalNvmePreset p;  // C++ 默认：预算字段的单一来源
    p.device = parse_device(get_dict_field(d, "device"));
    opt_int_field(d, "gpu_id", p.gpu_id);
    opt_int_field(d, "num_queues", p.num_queues);
    opt_int_field(d, "max_batch_entries", p.max_batch_entries);
    opt_int_field(d, "max_in_flight_operations", p.max_in_flight_operations);
    opt_int_field(d, "threads_per_block", p.threads_per_block);
    opt_int_field(d, "handle_cache_capacity", p.handle_cache_capacity);
    opt_int_field(d, "prp_cache_capacity", p.prp_cache_capacity);
    return p;
}

tutti::presets::StripedNvmePreset parse_striped_preset(const py::dict& d) {
    check_unknown_keys(
        d,
        {"devices", "gpu_id", "num_queues", "stripe_unit",
         "max_batch_entries", "max_in_flight_operations",
         "threads_per_block", "prp_cache_capacity"},
        "striped nvme preset");
    if (!d.contains("devices")) {
        value_error("missing preset key: 'devices'");
    }
    const py::object devices_obj = d["devices"];
    if (!py::isinstance<py::sequence>(devices_obj) ||
        py::isinstance<py::str>(devices_obj)) {
        value_error("preset key 'devices' must be a list of device dicts");
    }
    tutti::presets::StripedNvmePreset p;  // C++ 默认：预算字段的单一来源
    py::sequence devices = py::reinterpret_borrow<py::sequence>(devices_obj);
    for (std::size_t i = 0; i < py::len(devices); ++i) {
        p.devices.push_back(
            parse_device(checked_dict(
                devices.attr("__getitem__")(i),
                "devices[" + std::to_string(i) + "]")));
    }
    opt_int_field(d, "gpu_id", p.gpu_id);
    opt_int_field(d, "num_queues", p.num_queues);
    opt_int_field(d, "stripe_unit", p.stripe_unit);
    opt_int_field(d, "max_batch_entries", p.max_batch_entries);
    opt_int_field(d, "max_in_flight_operations", p.max_in_flight_operations);
    opt_int_field(d, "threads_per_block", p.threads_per_block);
    opt_int_field(d, "prp_cache_capacity", p.prp_cache_capacity);
    return p;
}

// ---------------------------------------------------------------------------
// Module-level factories
// ---------------------------------------------------------------------------

PyRuntime make_local_nvme_runtime(const py::dict& preset) {
    tutti::presets::LocalNvmePreset p = parse_local_preset(preset);
    auto assembled = tutti::presets::make_local_nvme_runtime(p);
    if (!assembled.runtime) {
        throw_status("make_local_nvme_runtime failed",
                     assembled.creation_status);
    }
    return PyRuntime(std::move(assembled.runtime), /*stub=*/false,
                     p.max_batch_entries, p.max_in_flight_operations,
                     p.gpu_id, /*max_concurrent_streams=*/2);
}

PyRuntime make_striped_nvme_runtime(const py::dict& preset) {
    tutti::presets::StripedNvmePreset p = parse_striped_preset(preset);
    auto assembled = tutti::presets::make_striped_nvme_runtime(p);
    if (!assembled.runtime) {
        throw_status("make_striped_nvme_runtime failed",
                     assembled.creation_status);
    }
    return PyRuntime(std::move(assembled.runtime), /*stub=*/false,
                     p.max_batch_entries, p.max_in_flight_operations,
                     p.gpu_id, /*max_concurrent_streams=*/2);
}

PyRuntime make_stub_runtime(std::int32_t accel_id) {
    tutti::RuntimeConfig config;
    config.accel_id = accel_id;
    auto created = tutti::StorageRuntime::create(config);
    if (!created.ok()) {
        throw_status("make_stub_runtime failed", created.status());
    }
    return PyRuntime(std::move(created).value(), /*stub=*/true,
                     /*max_batch=*/0, /*max_inflight=*/0,
                     accel_id, /*max_concurrent_streams=*/0);
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "tutti_runtime C++ core (pybind11)";

    py::class_<SubmitResult>(m, "SubmitResult")
        .def_readonly("status_ok", &SubmitResult::status_ok)
        .def_readonly("status_msg", &SubmitResult::status_msg)
        .def_readonly("io_handle", &SubmitResult::io_handle,
                      "int ticket, or None when every request was rejected")
        .def_readonly("initial_states", &SubmitResult::initial_states)
        .def_readonly("rejected", &SubmitResult::rejected);

    py::class_<WaitResult>(m, "WaitResult")
        .def_readonly("observation", &WaitResult::observation)
        .def_readonly("state", &WaitResult::state)
        .def_readonly("confirmed_bytes", &WaitResult::confirmed_bytes)
        .def_readonly("timeout_seen", &WaitResult::timeout_seen)
        .def_readonly("failed_request_indices",
                      &WaitResult::failed_request_indices)
        .def_readonly("failure_scope", &WaitResult::failure_scope)
        .def_readonly("failure_kind", &WaitResult::failure_kind)
        .def_readonly("first_failed_entry", &WaitResult::first_failed_entry)
        .def_readonly("raw_cq_status", &WaitResult::raw_cq_status)
        .def_readonly("message", &WaitResult::message);

    py::class_<PyRuntime>(m, "Runtime")
        .def("caps", &PyRuntime::caps)
        .def("open_batch", &PyRuntime::open_batch, py::arg("uris"))
        .def("close_target", &PyRuntime::close_target,
             py::arg("target_handle"))
        .def("close_batch", &PyRuntime::close_batch,
             py::arg("target_handles"))
        .def("register_memory", &PyRuntime::register_memory,
             py::arg("addr"), py::arg("size"), py::arg("kind"),
             py::arg("accel_id") = -1, py::arg("io_granularity") = 0)
        .def("unregister_memory", &PyRuntime::unregister_memory,
             py::arg("memory_ticket"))
        .def("submit", &PyRuntime::submit, py::arg("requests"),
             py::arg("accel_id"), py::arg("stream"),
             py::arg("execution") = "device")
        .def("release_io", &PyRuntime::release_io, py::arg("io_handle"))
        .def("wait", &PyRuntime::wait, py::arg("io_handle"),
             py::arg("timeout_ms"))
        .def("wait_result", &PyRuntime::wait_result, py::arg("io_handle"),
             py::arg("timeout_ms") = 0)
        .def("wait_detail", &PyRuntime::wait_detail, py::arg("io_handle"),
             py::arg("timeout_ms") = 0)
        .def("shutdown", &PyRuntime::shutdown, py::arg("timeout_ms"))
        .def("testing_force_complete", &PyRuntime::testing_force_complete,
             py::arg("io_handle"), py::arg("state") = "COMPLETED")
        .def("testing_inject_next_read_nvme_error",
             &PyRuntime::testing_inject_next_read_nvme_error,
             py::arg("raw_cq_status"));

    m.def("make_local_nvme_runtime", &make_local_nvme_runtime,
          py::arg("preset"));
    m.def("make_striped_nvme_runtime", &make_striped_nvme_runtime,
          py::arg("preset"));
    m.def("make_stub_runtime", &make_stub_runtime, py::arg("accel_id") = -1);
}

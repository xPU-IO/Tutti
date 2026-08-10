#pragma once

// tutti/storage_runtime.h -- Phase 1 StorageRuntime public facade.
//
// Header-only C++17 facade with two assembly modes:
//   - no injected components: hardware-free stub registry for contract tests;
//   - injected Resolver/DataPath components: real target routing, lazy
//     per-domain registration, DataPath submission, progress and teardown.
//
// Both modes share handle lifecycle, BUSY semantics, generation non-reuse,
// partial-commit, bounds validation, and WaitOutcome semantics. The test-only
// force-complete hook applies only to the hardware-free stub mode.
//
// Dependencies are public value headers plus the source-level Resolver/DataPath
// SPI; concrete transports and vendor SDK headers never enter this header.

#include <tutti/status.h>
#include <tutti/accelerator_device_guard.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>
#include <tutti/spi/data_path.h>
#include <tutti/spi/storage_target_resolver.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tutti {

// Public headers can also be consumed outside the repository CMake targets.
// Keep a deterministic HOST fallback for that source-only use; normal builds
// always provide these through the accelerator profile interface target.
#ifndef TUTTI_COMPILED_ACCELERATOR_PROFILE
#define TUTTI_COMPILED_ACCELERATOR_PROFILE "HOST"
#endif
#ifndef TUTTI_DEFAULT_ACCEL_ID
#define TUTTI_DEFAULT_ACCEL_ID -1
#endif

// Forward declaration for the test-access friend (gap 4).
namespace testing { struct StorageRuntimeTestAccess; }

// =========================================================================
// Value types
// =========================================================================

enum class RuntimeState {
    RUNNING,
    DRAINING,
    STOPPED,
};

struct RuntimeConfig {
    std::int32_t accel_id = TUTTI_DEFAULT_ACCEL_ID;
    std::uint64_t max_terminal_results = 64;
    std::string profile_name = TUTTI_COMPILED_ACCELERATOR_PROFILE;
};

// Static in-process assembly inputs. The runtime never owns these component
// objects; callers keep them alive through StorageRuntime::shutdown(). A
// DataPath is selected by ResolvedTarget::recommended_data_path_key(), while
// a resolver is selected by URI scheme. No transport-specific type appears in
// this public assembly contract.
struct ResolverBinding {
    std::string scheme;
    StorageTargetResolver* resolver = nullptr;
};

struct DataPathBinding {
    std::string key;
    DataPath* data_path = nullptr;
    DataPathConfig config;
};

struct RuntimeComponents {
    std::vector<ResolverBinding> resolvers;
    std::vector<DataPathBinding> data_paths;
    ResourceProvider* resources = nullptr;
};

struct CudaLikeProfileInfo {
    std::string profile_name;
    int device_count = 0;
};

struct DeviceInfo {
    std::int32_t accel_id = -1;
    std::string name;
    std::uint64_t total_memory = 0;
    std::string pci_bdf;
};

struct DeviceCapabilities {
    std::int32_t accel_id = -1;
    bool supports_host_execution = true;
    bool supports_device_execution = false;
    std::uint64_t max_io_size = 0;
    std::string pci_bdf;
};

struct MemorySpec {
    std::uint64_t size = 0;
    MemoryKind kind = MemoryKind::HOST;
    std::int32_t accel_id = -1;
};

struct MemoryAllocation {
    MemoryHandle handle;
    void* address = nullptr;
    std::uint64_t size = 0;
};

// MemoryInfo deliberately does NOT expose any data-path transport
// descriptor, physical bus address, or remote access key.
// The address field is the user-visible address only.
struct MemoryInfo {
    MemoryKind kind = MemoryKind::HOST;
    MemoryOwnership ownership = MemoryOwnership::CALLER_OWNED;
    std::uint64_t size = 0;
    void* address = nullptr;
    std::int32_t accel_id = -1;
    int inflight_count = 0;
};

struct OpenOptions {
    std::string scheme;
};

struct TargetInfo {
    std::string uri;
    std::uint64_t logical_size = 0;
    int inflight_count = 0;
};

enum class IoState {
    IN_FLIGHT,
    COMPLETED,
    FAILED,
};

struct IoSnapshot {
    IoState state = IoState::IN_FLIGHT;
};

struct IoResult {
    IoState state = IoState::COMPLETED;
    Status status;
};

// WaitOutcome distinguishes observation status from operation terminal state.
//   observation_status == OK       -> operation is terminal, result has value
//   observation_status == TIMEOUT  -> operation still in-flight, result empty
//   observation_status == error    -> handle error, result empty
struct WaitOutcome {
    Status observation_status;
    std::optional<IoResult> result;
};

// -------------------------------------------------------------------------
// Batch submit result types (gap 1).
//
// IoSubmitOutcome expresses partial commit: some requests may be ACCEPTED
// while others are REJECTED. The overall status may be non-OK even when
// io has a value (at least one request was irreversibly issued).
//
// Invariants:
//   1. io == nullopt  <=>  zero requests were irreversibly accepted
//      (status may still be non-OK, e.g. all rejected).
//   2. io != nullopt  <=>  at least one request was accepted and is still
//      observable, even if status reports a partial failure.
//   3. initial_states.size() == count, in input order.
//   4. Accepted requests are observable via the returned IoHandle.
//   5. count == 0 => io == nullopt, status OK, initial_states empty.
// -------------------------------------------------------------------------
enum class IoRequestState {
    ACCEPTED,  // issued or will be issued; observable via the IoHandle
    REJECTED,  // validation failed before any irreversible issue
};

struct IoRequestInitialState {
    IoRequestState state = IoRequestState::REJECTED;
    Status status;  // OK for ACCEPTED; error code+message for REJECTED
};

struct IoSubmitOutcome {
    Status status;
    std::optional<IoHandle> io;
    std::vector<IoRequestInitialState> initial_states;
};

// =========================================================================
// StorageRuntime
// =========================================================================

class StorageRuntime {
public:
    static Result<std::unique_ptr<StorageRuntime>> create(
        RuntimeConfig config = {}) {
        std::uint32_t rid = next_runtime_id_.fetch_add(
            1, std::memory_order_relaxed);
        auto runtime = std::unique_ptr<StorageRuntime>(
            new StorageRuntime(rid, std::move(config)));
        Status status = runtime->validate_runtime_config_();
        if (!status.ok()) {
            return Result<std::unique_ptr<StorageRuntime>>::Failure(
                std::move(status));
        }
        return runtime;
    }

    // Creates a runtime wired to statically injected in-process components.
    // Component initialization is transactional: a failed create() shuts down
    // already initialized DataPaths and returns no Runtime instance.
    static Result<std::unique_ptr<StorageRuntime>> create(
        RuntimeConfig config, RuntimeComponents components) {
        std::uint32_t rid = next_runtime_id_.fetch_add(
            1, std::memory_order_relaxed);
        auto runtime = std::unique_ptr<StorageRuntime>(
            new StorageRuntime(rid, std::move(config)));
        Status status = runtime->validate_runtime_config_();
        if (!status.ok()) {
            return Result<std::unique_ptr<StorageRuntime>>::Failure(
                std::move(status));
        }
        status = runtime->initialize_components_(std::move(components));
        if (!status.ok()) {
            return Result<std::unique_ptr<StorageRuntime>>::Failure(
                std::move(status));
        }
        return Result<std::unique_ptr<StorageRuntime>>(std::move(runtime));
    }

    // If the runtime was not explicitly shut down (state != STOPPED),
    // attempt a non-blocking drain. If inflight operations remain, we
    // conservatively LEAK runtime-owned allocations rather than free
    // memory that a still-active DataPath operation might be DMA-ing to.
    // This matches the shutdown(timeout=0) contract: never UAF to free.
    ~StorageRuntime() {
        if (state_.load() != RuntimeState::STOPPED) {
            (void)shutdown(0);
        }
        if (state_.load() != RuntimeState::STOPPED) {
            // Still has inflight operations after best-effort drain.
            // Do NOT free runtime-owned memory; leak to avoid UAF.
            return;
        }
        std::lock_guard<std::mutex> lock(registry_mutex_);
        for (auto& e : memory_entries_) {
            if (e.active &&
                e.ownership == MemoryOwnership::RUNTIME_OWNED) {
                std::free(e.address);
            }
        }
    }

    StorageRuntime(const StorageRuntime&) = delete;
    StorageRuntime& operator=(const StorageRuntime&) = delete;
    StorageRuntime(StorageRuntime&&) = delete;
    StorageRuntime& operator=(StorageRuntime&&) = delete;

    // ---- lifecycle ----

    Status shutdown(std::uint64_t timeout_ms) {
        std::unique_lock<std::mutex> lock(registry_mutex_);
        if (state_.load() == RuntimeState::STOPPED) {
            return Status::Ok();
        }
        state_.store(RuntimeState::DRAINING);

        if (count_inflight_io_() == 0) {
            return finalize_shutdown_locked_(lock);
        }
        if (timeout_ms == 0) {
            return Status(StatusCode::TIMEOUT,
                          "shutdown timed out with inflight operations");
        }
        auto deadline = std::chrono::steady_clock::now()
                       + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            // Drive progress without holding the registry lock so that
            // DataPath::progress (which may block on device completion)
            // does not starve other registry operations.
            drive_progress_unlocked_(lock);
            if (count_inflight_io_() == 0) {
                return finalize_shutdown_locked_(lock);
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return Status(StatusCode::TIMEOUT,
                              "shutdown timed out with inflight operations");
            }
            // Wait briefly, re-checking for terminal operations via the
            // condition variable that finish_io_ notifies.
            io_cv_.wait_for(lock, std::chrono::milliseconds(1),
                            [this]{ return count_inflight_io_() == 0; });
        }
    }

    RuntimeState state() const noexcept {
        return state_.load();
    }

    std::int32_t accel_id() const noexcept { return config_.accel_id; }

    // ---- discovery ----

    CudaLikeProfileInfo query_cuda_like_profile() const {
        auto count = backend_device_count_();
        return CudaLikeProfileInfo{config_.profile_name,
                                   count.ok() ? count.value() : 0};
    }

    Result<std::vector<DeviceInfo>> list_devices() const {
        if (config_.accel_id == -1 && compiled_profile_is_host_()) {
            return std::vector<DeviceInfo>{};
        }
        auto count = backend_device_count_();
        if (!count.ok()) {
            return Result<std::vector<DeviceInfo>>::Failure(count.status());
        }
        std::vector<DeviceInfo> devices;
        for (int id = 0; id < count.value(); ++id) {
            auto device = discover_device_(id);
            if (!device.ok()) {
                return Result<std::vector<DeviceInfo>>::Failure(device.status());
            }
            devices.push_back(std::move(device).value());
        }
        return devices;
    }

    Result<DeviceCapabilities> query_device_capabilities(
        std::int32_t accel_id) const {
        auto devices = list_devices();
        if (!devices.ok()) {
            return Result<DeviceCapabilities>::Failure(devices.status());
        }
        if (accel_id < 0 || accel_id >= static_cast<std::int32_t>(devices.value().size())) {
            return Result<DeviceCapabilities>::Failure(
                Status(StatusCode::NOT_FOUND, "device not found"));
        }
        const auto& info = devices.value()[static_cast<std::size_t>(accel_id)];
        return DeviceCapabilities{accel_id, true, true, 1ULL << 30,
                                  info.pci_bdf};
    }

    // ---- memory ----

    Result<MemoryAllocation> allocate_memory(const MemorySpec& spec) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        if (state_.load() != RuntimeState::RUNNING) {
            return Result<MemoryAllocation>::Failure(
                Status(StatusCode::NOT_READY,
                       "runtime is not in RUNNING state"));
        }
        if (spec.size == 0) {
            return Result<MemoryAllocation>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "size must be > 0"));
        }
        if (spec.kind == MemoryKind::DEVICE ||
            spec.kind == MemoryKind::MANAGED) {
            return Result<MemoryAllocation>::Failure(
                Status(StatusCode::UNSUPPORTED,
                       "backend allocation for DEVICE/MANAGED memory is not implemented"));
        }
        if (spec.accel_id >= 0 && config_.accel_id >= 0 &&
            spec.accel_id != config_.accel_id) {
            return Result<MemoryAllocation>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "memory accelerator does not match Runtime"));
        }
        if (spec.accel_id >= 0 && config_.accel_id < 0) {
            return Result<MemoryAllocation>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "host-only Runtime cannot allocate accelerator memory"));
        }
        void* ptr = std::malloc(static_cast<std::size_t>(spec.size));
        if (!ptr) {
            return Result<MemoryAllocation>::Failure(
                Status(StatusCode::RESOURCE_EXHAUSTED,
                       "allocation failed"));
        }
        std::uint32_t slot = find_free_memory_slot_();
        std::uint64_t gen = ++memory_gen_counter_;
        MemoryEntry& entry = memory_entries_[slot];
        entry.active = true;
        entry.generation = gen;
        entry.address = ptr;
        entry.size = spec.size;
        entry.ownership = MemoryOwnership::RUNTIME_OWNED;
        entry.kind = spec.kind;
        entry.accel_id = spec.accel_id;
        entry.inflight_count = 0;
        entry.data_path_registrations.clear();
        return MemoryAllocation{
            MemoryHandle(runtime_id_, slot, gen), ptr, spec.size};
    }

    Status free_memory(const MemoryHandle& handle) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        if (!validate_memory_(handle)) {
            return Status(StatusCode::NOT_FOUND, "invalid memory handle");
        }
        MemoryEntry& entry = memory_entries_[handle.slot_];
        if (entry.ownership != MemoryOwnership::RUNTIME_OWNED) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "free_memory requires RUNTIME_OWNED allocation");
        }
        if (entry.inflight_count > 0) {
            return Status(StatusCode::BUSY,
                          "memory has inflight operations");
        }
        Status cleanup = unregister_data_path_memory_(entry);
        if (!cleanup.ok()) {
            return cleanup;
        }
        std::free(entry.address);
        entry.active = false;
        return Status::Ok();
    }

    Result<MemoryHandle> register_memory(const MemoryView& view) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        if (state_.load() != RuntimeState::RUNNING) {
            return Result<MemoryHandle>::Failure(
                Status(StatusCode::NOT_READY,
                       "runtime is not in RUNNING state"));
        }
        if (view.address == nullptr || view.size == 0) {
            return Result<MemoryHandle>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "address and size must be non-null/non-zero"));
        }
        if (!view.expected_profile.empty() &&
            !profile_matches_(view.expected_profile)) {
            return Result<MemoryHandle>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "memory profile does not match compiled accelerator profile"));
        }
        if ((view.expected_kind == MemoryKind::DEVICE ||
             view.expected_kind == MemoryKind::MANAGED) &&
            config_.accel_id < 0) {
            return Result<MemoryHandle>::Failure(
                Status(StatusCode::UNSUPPORTED,
                       "host-only Runtime cannot register accelerator memory"));
        }
        Status pointer_status = validate_pointer_accel_(view.address,
                                                        view.expected_kind,
                                                        view.expected_accel_id);
        if (!pointer_status.ok()) {
            return Result<MemoryHandle>::Failure(std::move(pointer_status));
        }
        if (view.expected_accel_id >= 0 && config_.accel_id >= 0 &&
            view.expected_accel_id != config_.accel_id) {
            return Result<MemoryHandle>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "memory accelerator does not match Runtime"));
        }
        if (view.expected_accel_id >= 0 && config_.accel_id < 0 &&
            (view.expected_kind == MemoryKind::DEVICE ||
             view.expected_kind == MemoryKind::MANAGED)) {
            return Result<MemoryHandle>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "host-only Runtime cannot register accelerator memory"));
        }
        std::uint32_t slot = find_free_memory_slot_();
        std::uint64_t gen = ++memory_gen_counter_;
        MemoryEntry& entry = memory_entries_[slot];
        entry.active = true;
        entry.generation = gen;
        entry.address = view.address;
        entry.size = view.size;
        entry.ownership = MemoryOwnership::CALLER_OWNED;
        entry.kind = view.expected_kind;
        entry.accel_id = view.expected_accel_id >= 0
            ? view.expected_accel_id
            : ((view.expected_kind == MemoryKind::DEVICE ||
                view.expected_kind == MemoryKind::MANAGED)
               ? config_.accel_id : -1);
        entry.inflight_count = 0;
        entry.data_path_registrations.clear();
        return MemoryHandle(runtime_id_, slot, gen);
    }

    Status unregister_memory(const MemoryHandle& handle) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        if (!validate_memory_(handle)) {
            return Status(StatusCode::NOT_FOUND, "invalid memory handle");
        }
        MemoryEntry& entry = memory_entries_[handle.slot_];
        if (entry.ownership != MemoryOwnership::CALLER_OWNED) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "unregister_memory requires CALLER_OWNED registration");
        }
        if (entry.inflight_count > 0) {
            return Status(StatusCode::BUSY,
                          "memory has inflight operations");
        }
        Status cleanup = unregister_data_path_memory_(entry);
        if (!cleanup.ok()) {
            return cleanup;
        }
        entry.active = false;
        return Status::Ok();
    }

    Result<MemoryInfo> query_memory(const MemoryHandle& handle) const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        if (!validate_memory_(handle)) {
            return Result<MemoryInfo>::Failure(
                Status(StatusCode::NOT_FOUND, "invalid memory handle"));
        }
        const MemoryEntry& e = memory_entries_[handle.slot_];
        return MemoryInfo{
            e.kind, e.ownership, e.size, e.address,
            e.accel_id, e.inflight_count};
    }

    // ---- target ----

    Result<TargetHandle> open(std::string_view uri,
                              const OpenOptions& options) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        if (state_.load() != RuntimeState::RUNNING) {
            return Result<TargetHandle>::Failure(
                Status(StatusCode::NOT_READY,
                       "runtime is not in RUNNING state"));
        }

        // Preserve the existing hardware-free stub behavior when no assembly
        // components were injected. All component-backed opens take the real
        // resolver → DataPath route below.
        if (!components_enabled_) {
            std::uint32_t slot = find_free_target_slot_();
            std::uint64_t gen = ++target_gen_counter_;
            TargetEntry& entry = target_entries_[slot];
            entry.active = true;
            entry.generation = gen;
            entry.uri = std::string(uri);
            entry.logical_size = 1ULL << 30;
            entry.inflight_count = 0;
            return TargetHandle(runtime_id_, slot, gen);
        }

        std::string scheme = scheme_for_(uri, options);
        if (scheme.empty()) {
            return Result<TargetHandle>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "URI scheme is required for component-backed open"));
        }
        auto resolver_it = resolvers_.find(scheme);
        if (resolver_it == resolvers_.end()) {
            return Result<TargetHandle>::Failure(
                Status(StatusCode::NOT_FOUND,
                       "no resolver registered for URI scheme: " + scheme));
        }

        ResolveOptions resolve_options;
        resolve_options.scheme = scheme;
        auto resolved = resolver_it->second->resolve(uri, resolve_options);
        if (!resolved.ok()) {
            return Result<TargetHandle>::Failure(resolved.status());
        }
        if (!resolved.value().valid()) {
            return Result<TargetHandle>::Failure(
                Status(StatusCode::INTERNAL,
                       "resolver returned an invalid target"));
        }

        const std::string path_key(
            resolved.value().recommended_data_path_key());
        auto path_it = data_paths_.find(path_key);
        if (path_it == data_paths_.end()) {
            return Result<TargetHandle>::Failure(
                Status(StatusCode::NOT_FOUND,
                       "no DataPath registered for target key: " + path_key));
        }
        DataPath* data_path = path_it->second;
        auto private_target = [&]() -> Result<DataPathTarget> {
            std::lock_guard<std::mutex> dp_lock(datapath_open_mutex_);
            return call_data_path_result_<DataPathTarget>(
                *data_path, [&] { return data_path->open(resolved.value()); });
        }();
        if (!private_target.ok()) {
            return Result<TargetHandle>::Failure(private_target.status());
        }
        auto domain = [&]() -> Result<RegistrationDomainKey> {
            std::lock_guard<std::mutex> dp_lock(datapath_open_mutex_);
            return call_data_path_result_<RegistrationDomainKey>(
                *data_path,
                [&] { return data_path->registration_domain(private_target.value()); });
        }();
        if (!domain.ok()) {
            (void)call_data_path_status_(
                *data_path,
                [&] { return data_path->close(private_target.value()); });
            return Result<TargetHandle>::Failure(domain.status());
        }

        std::uint32_t slot = find_free_target_slot_();
        std::uint64_t gen = ++target_gen_counter_;
        TargetEntry& entry = target_entries_[slot];
        entry.active = true;
        entry.generation = gen;
        entry.uri = std::string(uri);
        entry.logical_size = resolved.value().logical_size();
        entry.inflight_count = 0;
        entry.data_path = data_path;
        entry.data_path_target = private_target.value();
        entry.registration_domain = std::move(domain).value();
        entry.resolved_target.emplace(std::move(resolved).value());
        return TargetHandle(runtime_id_, slot, gen);
    }

    // Round 19 S1: batch open — opens N targets concurrently.
    //
    // The hot path of a single open is resolver.resolve() (FIEMAP, host
    // IO-bound) followed by data_path->open() (GPU workspace build) and
    // a brief critical section that allocates a target slot.  In a
    // batch of N, the FIEMAP calls run concurrently (the resolver is
    // thread-safe — it only reads immutable config and issues syscalls),
    // overlapping host IO across files.  data_path->open() is serialized
    // by datapath_open_mutex_ because DataPath internals (targets_ map,
    // handle cache) are not thread-safe; the serial section is cheap
    // relative to FIEMAP.  The final slot allocation + entry write is
    // serialized under registry_mutex_ as in open().
    //
    // fail-closed per item: a single failing URI (NOT_FOUND, bad scheme,
    // DataPath error, ...) does NOT abort the batch — the corresponding
    // Result<TargetHandle> carries the failure status, other items
    // proceed normally.  This matches the per-request semantics of
    // submit().
    //
    // Mixed schemes work: each URI is routed by its scheme (file://,
    // striped://, ...), so a batch may span multiple resolvers and
    // DataPaths.  Handle-cache dedup (when cache is ON) happens inside
    // data_path->open() as usual; concurrent opens of the same extent
    // signature are serialized by datapath_open_mutex_ so the cache's
    // get_or_build stays race-free.
    std::vector<Result<TargetHandle>> open_batch(
        const std::vector<std::string>& uris,
        const OpenOptions& options) {

        std::vector<Result<TargetHandle>> results;
        results.reserve(uris.size());
        for (std::size_t i = 0; i < uris.size(); ++i) {
            results.emplace_back(Result<TargetHandle>::Failure(
                Status(StatusCode::INTERNAL, "uninitialized")));
        }
        if (uris.empty()) return results;

        // Phase 0: state check (serial, brief).
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            if (state_.load() != RuntimeState::RUNNING) {
                Status s(StatusCode::NOT_READY,
                         "runtime is not in RUNNING state");
                for (auto& r : results) r = Result<TargetHandle>::Failure(s);
                return results;
            }
        }

        // Stub mode: mirror open()'s stub behavior (no components).
        if (!components_enabled_) {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            for (std::size_t i = 0; i < uris.size(); ++i) {
                std::uint32_t slot = find_free_target_slot_();
                std::uint64_t gen = ++target_gen_counter_;
                TargetEntry& entry = target_entries_[slot];
                entry.active = true;
                entry.generation = gen;
                entry.uri = uris[i];
                entry.logical_size = 1ULL << 30;
                entry.inflight_count = 0;
                results[i] = TargetHandle(runtime_id_, slot, gen);
            }
            return results;
        }

        // Phase 1: parallel resolve (FIEMAP, host IO-bound).
        // Each worker resolves one URI to a ResolvedTarget.  Failures
        // are recorded per-item; failed items skip Phase 2.
        struct ResolvedItem {
            Result<ResolvedTarget> resolved = Result<ResolvedTarget>::Failure(
                Status(StatusCode::INTERNAL, "uninitialized"));
            StorageTargetResolver* resolver = nullptr;
            std::string recommended_data_path_key;
        };
        std::vector<ResolvedItem> items;
        items.reserve(uris.size());
        for (std::size_t i = 0; i < uris.size(); ++i) items.push_back(ResolvedItem{});

        auto resolve_one = [&](std::size_t i) {
            const std::string& uri = uris[i];
            std::string scheme = scheme_for_(uri, options);
            if (scheme.empty()) {
                items[i].resolved = Result<ResolvedTarget>::Failure(
                    Status(StatusCode::INVALID_ARGUMENT,
                           "URI scheme is required: " + uri));
                return;
            }
            // resolver lookup is read-only on resolvers_ (immutable
            // after create()), so no lock needed here.
            auto it = resolvers_.find(scheme);
            if (it == resolvers_.end()) {
                items[i].resolved = Result<ResolvedTarget>::Failure(
                    Status(StatusCode::NOT_FOUND,
                           "no resolver for scheme: " + scheme));
                return;
            }
            ResolveOptions opts;
            opts.scheme = scheme;
            items[i].resolver = it->second;
            items[i].resolved = it->second->resolve(uri, opts);
            if (items[i].resolved.ok()) {
                items[i].recommended_data_path_key =
                    items[i].resolved.value().recommended_data_path_key();
            }
        };

        // Use a simple worker pool: spawn min(N, hardware_concurrency)
        // threads, each pulling indices from an atomic counter.
        std::atomic<std::size_t> next_idx{0};
        std::size_t n_workers = std::min<std::size_t>(
            uris.size(),
            std::max<std::size_t>(1, std::thread::hardware_concurrency()));
        std::vector<std::thread> workers;
        workers.reserve(n_workers);
        for (std::size_t w = 0; w < n_workers; ++w) {
            workers.emplace_back([&]() {
                for (;;) {
                    std::size_t i = next_idx.fetch_add(1,
                        std::memory_order_relaxed);
                    if (i >= uris.size()) break;
                    resolve_one(i);
                }
            });
        }
        for (auto& t : workers) t.join();

        // Phase 2: serial data_path->open() + registration_domain.
        // DataPath internals are not thread-safe; datapath_open_mutex_
        // serializes across concurrent open_batch calls AND the single
        // open() path (open() also acquires it).
        for (std::size_t i = 0; i < uris.size(); ++i) {
            if (!items[i].resolved.ok()) {
                results[i] = Result<TargetHandle>::Failure(
                    items[i].resolved.status());
                continue;
            }
            const auto& rt = items[i].resolved.value();
            if (!rt.valid()) {
                results[i] = Result<TargetHandle>::Failure(
                    Status(StatusCode::INTERNAL,
                           "resolver returned invalid target: " + uris[i]));
                continue;
            }
            auto path_it = data_paths_.find(items[i].recommended_data_path_key);
            if (path_it == data_paths_.end()) {
                results[i] = Result<TargetHandle>::Failure(
                    Status(StatusCode::NOT_FOUND,
                           "no DataPath for key: " +
                           items[i].recommended_data_path_key));
                continue;
            }
            DataPath* data_path = path_it->second;

            DataPathTarget private_target;
            RegistrationDomainKey domain;
            {
                std::lock_guard<std::mutex> dp_lock(datapath_open_mutex_);
                auto pt = call_data_path_result_<DataPathTarget>(
                    *data_path, [&] { return data_path->open(rt); });
                if (!pt.ok()) {
                    results[i] = Result<TargetHandle>::Failure(pt.status());
                    continue;
                }
                private_target = pt.value();
                auto dom = call_data_path_result_<RegistrationDomainKey>(
                    *data_path,
                    [&] { return data_path->registration_domain(private_target); });
                if (!dom.ok()) {
                    (void)call_data_path_status_(
                        *data_path,
                        [&] { return data_path->close(private_target); });
                    results[i] = Result<TargetHandle>::Failure(dom.status());
                    continue;
                }
                domain = std::move(dom).value();
            }

            // Phase 3: serial slot allocation + entry write.
            std::lock_guard<std::mutex> lock(registry_mutex_);
            std::uint32_t slot = find_free_target_slot_();
            std::uint64_t gen = ++target_gen_counter_;
            TargetEntry& entry = target_entries_[slot];
            entry.active = true;
            entry.generation = gen;
            entry.uri = uris[i];
            entry.logical_size = rt.logical_size();
            entry.inflight_count = 0;
            entry.data_path = data_path;
            entry.data_path_target = private_target;
            entry.registration_domain = std::move(domain);
            entry.resolved_target.emplace(std::move(items[i].resolved).value());
            results[i] = TargetHandle(runtime_id_, slot, gen);
        }
        return results;
    }

    Status close(const TargetHandle& handle) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        if (!validate_target_(handle)) {
            return Status(StatusCode::NOT_FOUND, "invalid target handle");
        }
        TargetEntry& entry = target_entries_[handle.slot_];
        if (entry.inflight_count > 0) {
            return Status(StatusCode::BUSY,
                          "target has inflight operations");
        }
        if (entry.data_path != nullptr) {
            Status status = call_data_path_status_(
                *entry.data_path,
                [&] { return entry.data_path->close(entry.data_path_target); });
            if (!status.ok()) {
                return status;
            }
            entry.resolved_target.reset();
            entry.data_path = nullptr;
            entry.data_path_target = DataPathTarget{};
            entry.registration_domain = RegistrationDomainKey{};
        }
        entry.active = false;
        return Status::Ok();
    }

    Result<TargetInfo> query_target(const TargetHandle& handle) const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        if (!validate_target_(handle)) {
            return Result<TargetInfo>::Failure(
                Status(StatusCode::NOT_FOUND, "invalid target handle"));
        }
        const TargetEntry& e = target_entries_[handle.slot_];
        return TargetInfo{e.uri, e.logical_size, e.inflight_count};
    }

    // ---- IO ----

    // Batch submit with partial-commit semantics (gap 1 + gap 2).
    //
    // Validates each request independently. A request that fails bounds
    // validation is marked REJECTED in the output; it does not prevent
    // other valid requests from being accepted. If at least one request
    // is accepted, an IoHandle is returned even if the overall status is
    // non-OK (partial failure).
    //
    // PARTIAL-COMMIT CONTRACT (P0-3, Round 16):
    //   The returned IoHandle covers ONLY the requests that were ACCEPTED
    //   in this call.  The caller MUST inspect every entry of
    //   initial_states[] and re-submit any REJECTED requests (e.g.
    //   RESOURCE_EXHAUSTED back-pressure) in a subsequent windowed call.
    //   Failing to re-submit rejected requests causes SILENT DATA LOSS:
    //   wait(handle) returns success but the rejected IO never happened.
    //   See R14 S4 incident: 512-request batch, 16 accepted, 496 rejected
    //   by LocalNvmeDataPath in-flight cap — data was lost until the
    //   caller was fixed to window on initial_states.
    IoSubmitOutcome submit(const IoRequest* requests,
                           std::size_t count,
                           const HostSubmitContext& context) {
        std::unique_lock<std::mutex> lock(registry_mutex_);
        if (components_enabled_) {
            // submit_component_backed_ runs entirely under registry_mutex_;
            // the lock guard above is sufficient.  DataPath::submit is
            // called under the lock because DataPath implementations are
            // not guaranteed thread-safe for concurrent submit() calls.
            return submit_component_backed_(requests, count, context);
        }
        (void)context;

        IoSubmitOutcome outcome;
        outcome.initial_states.resize(count);

        if (state_.load() != RuntimeState::RUNNING) {
            for (std::size_t i = 0; i < count; ++i) {
                outcome.initial_states[i].state = IoRequestState::REJECTED;
                outcome.initial_states[i].status = Status(
                    StatusCode::NOT_READY, "runtime is not in RUNNING state");
            }
            outcome.status = Status(StatusCode::NOT_READY,
                                    "runtime is not in RUNNING state");
            // io stays nullopt — zero accepted.
            return outcome;
        }

        if (count == 0) {
            outcome.status = Status::Ok();
            // io stays nullopt, initial_states is empty.
            return outcome;
        }

        if (terminal_result_count_ >= config_.max_terminal_results) {
            for (std::size_t i = 0; i < count; ++i) {
                outcome.initial_states[i].state = IoRequestState::REJECTED;
                outcome.initial_states[i].status = Status(
                    StatusCode::RESOURCE_EXHAUSTED,
                    "terminal result limit reached");
            }
            outcome.status = Status(StatusCode::RESOURCE_EXHAUSTED,
                                    "terminal result limit reached");
            return outcome;
        }

        // Phase 1: validate every request independently (gap 2).
        std::size_t accepted_count = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const IoRequest& req = requests[i];
            Status v = validate_request_(req);
            if (v.ok()) {
                outcome.initial_states[i].state = IoRequestState::ACCEPTED;
                outcome.initial_states[i].status = Status::Ok();
                ++accepted_count;
            } else {
                outcome.initial_states[i].state = IoRequestState::REJECTED;
                outcome.initial_states[i].status = std::move(v);
            }
        }

        // If zero accepted, return without an IoHandle.
        if (accepted_count == 0) {
            outcome.status = Status(StatusCode::INVALID_ARGUMENT,
                                    "all requests rejected");
            return outcome;
        }

        // Phase 2: create one IoEntry for the batch. Accepted requests
        // share this IoHandle; rejected ones are recorded but do not
        // participate in the in-flight operation.
        std::uint32_t slot = find_free_io_slot_();
        std::uint64_t gen = ++io_gen_counter_;
        IoEntry& entry = io_entries_[slot];
        entry.active = true;
        entry.generation = gen;
        entry.state = IoState::IN_FLIGHT;
        entry.terminal = false;
        entry.released = false;
        entry.io_status = Status::Ok();
        entry.data_path_operations.clear();
        entry.memory_slots.clear();
        entry.target_slots.clear();
        entry.references_released = false;

        // Increment inflight for each accepted request's memory/target.
        for (std::size_t i = 0; i < count; ++i) {
            if (outcome.initial_states[i].state == IoRequestState::ACCEPTED) {
                memory_entries_[requests[i].memory.slot_].inflight_count++;
                target_entries_[requests[i].target.slot_].inflight_count++;
                entry.memory_slots.push_back(requests[i].memory.slot_);
                entry.target_slots.push_back(requests[i].target.slot_);
            }
        }

        outcome.io = IoHandle(runtime_id_, slot, gen);
        outcome.status = (accepted_count == count)
            ? Status::Ok()
            : Status(StatusCode::INVALID_ARGUMENT,
                     "partial submit: some requests rejected");
        return outcome;
    }

    // query() is non-const (gap 3): it may drive a bounded progress step.
    Result<IoSnapshot> query(const IoHandle& handle) {
        std::unique_lock<std::mutex> lock(registry_mutex_);
        if (!validate_io_(handle)) {
            return Result<IoSnapshot>::Failure(
                Status(StatusCode::NOT_FOUND, "invalid IO handle"));
        }
        std::uint32_t slot = handle.slot_;
        // Fast path: already terminal — no progress needed.
        if (io_entries_[slot].terminal) {
            return IoSnapshot{io_entries_[slot].state};
        }
        drive_progress_unlocked_(lock);
        return IoSnapshot{io_entries_[slot].state};
    }

    // wait() is non-const (gap 3): it drives bounded progress in its loop.
    WaitOutcome wait(const IoHandle& handle,
                     std::uint64_t timeout_ms) {
        std::unique_lock<std::mutex> lock(registry_mutex_);
        if (!validate_io_(handle)) {
            return WaitOutcome{
                Status(StatusCode::NOT_FOUND, "invalid IO handle"),
                std::nullopt};
        }
        std::uint32_t slot = handle.slot_;
        if (io_entries_[slot].terminal) {
            return WaitOutcome{
                Status::Ok(),
                IoResult{io_entries_[slot].state,
                         io_entries_[slot].io_status}};
        }
        if (timeout_ms == 0) {
            return WaitOutcome{
                Status(StatusCode::TIMEOUT, "observation timeout"),
                std::nullopt};
        }
        auto deadline = std::chrono::steady_clock::now()
                       + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            drive_progress_unlocked_(lock);
            if (io_entries_[slot].terminal) {
                return WaitOutcome{
                    Status::Ok(),
                    IoResult{io_entries_[slot].state,
                             io_entries_[slot].io_status}};
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return WaitOutcome{
                    Status(StatusCode::TIMEOUT, "observation timeout"),
                    std::nullopt};
                }
            // Wait for either a terminal notification or the deadline.
            io_cv_.wait_for(lock, std::chrono::milliseconds(1),
                            [this, slot]{ return io_entries_[slot].terminal; });
        }
    }

    Status release_io(const IoHandle& handle) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        if (!validate_io_(handle)) {
            return Status(StatusCode::NOT_FOUND, "invalid IO handle");
        }
        IoEntry& entry = io_entries_[handle.slot_];
        if (!entry.terminal) {
            return Status(StatusCode::BUSY, "IO is still in flight");
        }
        for (const auto& sub_op : entry.data_path_operations) {
            auto gate_it = progress_gates_.find(sub_op.data_path);
            std::unique_lock<std::mutex> gate_lock;
            if (gate_it != progress_gates_.end()) {
                gate_lock = std::unique_lock<std::mutex>(*gate_it->second);
            }
            Status status = call_data_path_status_(
                *sub_op.data_path,
                [&] { return sub_op.data_path->release(sub_op.op); });
            if (!status.ok()) return status;
        }
        entry.data_path_operations.clear();
        entry.released = true;
        entry.active = false;
        if (terminal_result_count_ > 0) {
            --terminal_result_count_;
        }
        return Status::Ok();
    }

private:
    struct DataPathMemoryRegistration {
        DataPath* data_path = nullptr;
        RegistrationDomainKey domain;
        DataPathMemory memory;
    };

    struct MemoryEntry {
        bool active = false;
        std::uint64_t generation = 0;
        void* address = nullptr;
        std::uint64_t size = 0;
        MemoryOwnership ownership = MemoryOwnership::CALLER_OWNED;
        MemoryKind kind = MemoryKind::HOST;
        std::int32_t accel_id = -1;
        int inflight_count = 0;
        std::vector<DataPathMemoryRegistration> data_path_registrations;
    };

    struct TargetEntry {
        bool active = false;
        std::uint64_t generation = 0;
        std::string uri;
        std::uint64_t logical_size = 0;
        int inflight_count = 0;
        DataPath* data_path = nullptr;
        DataPathTarget data_path_target;
        RegistrationDomainKey registration_domain;
        std::optional<ResolvedTarget> resolved_target;
    };

    struct DataPathOperation {
        DataPath* data_path = nullptr;
        DataPathOp op;
    };

    struct IoEntry {
        bool active = false;
        std::uint64_t generation = 0;
        IoState state = IoState::IN_FLIGHT;
        bool terminal = false;
        bool released = false;
        bool references_released = false;
        Status io_status;
        std::vector<DataPathOperation> data_path_operations;
        std::vector<std::uint32_t> memory_slots;
        std::vector<std::uint32_t> target_slots;
    };

    friend struct ::tutti::testing::StorageRuntimeTestAccess;

    StorageRuntime(std::uint32_t runtime_id, RuntimeConfig config)
        : runtime_id_(runtime_id), config_(std::move(config)) {}

    std::int32_t data_path_accel_id_(const DataPath& data_path) const noexcept {
        const auto bound = data_path.capabilities().bound_accel_id;
        return bound >= 0 ? bound : config_.accel_id;
    }

    template <typename T, typename Fn>
    Result<T> call_data_path_result_(DataPath& data_path, Fn&& fn) const {
        DeviceGuard guard(data_path_accel_id_(data_path));
        if (!guard.ok()) {
            return Result<T>::Failure(guard.status());
        }
        Result<T> result = fn();
        Status restored = guard.restore();
        if (!restored.ok()) {
            return Result<T>::Failure(std::move(restored));
        }
        return result;
    }

    template <typename Fn>
    Status call_data_path_status_(DataPath& data_path, Fn&& fn) const {
        DeviceGuard guard(data_path_accel_id_(data_path));
        if (!guard.ok()) {
            return guard.status();
        }
        Status result = fn();
        Status restored = guard.restore();
        if (!restored.ok()) {
            return restored;
        }
        return result;
    }

    Status finalize_shutdown_locked_(std::unique_lock<std::mutex>& lock) {
        // Caller holds registry_mutex_. No in-flight IO remains.
        // Release data-path targets before resolver leases, then per-memory
        // registrations, then DataPath lifecycle.
        for (auto& target : target_entries_) {
            if (!target.active) {
                continue;
            }
            if (target.data_path != nullptr) {
                // DataPath::close does not call back into the Runtime, so
                // holding the lock here is safe (no deadlock).
                Status status = call_data_path_status_(
                    *target.data_path,
                    [&] { return target.data_path->close(target.data_path_target); });
                if (!status.ok()) {
                    return status;
                }
                target.resolved_target.reset();
                target.data_path = nullptr;
                target.data_path_target = DataPathTarget{};
                target.registration_domain = RegistrationDomainKey{};
            }
            target.active = false;
        }
        for (auto& memory : memory_entries_) {
            if (!memory.active) {
                continue;
            }
            Status status = unregister_data_path_memory_(memory);
            if (!status.ok()) {
                return status;
            }
            if (memory.ownership == MemoryOwnership::RUNTIME_OWNED) {
                std::free(memory.address);
                memory.address = nullptr;
            }
            memory.active = false;
        }
        for (auto it = initialized_data_paths_.rbegin();
             it != initialized_data_paths_.rend(); ++it) {
            // DataPath::shutdown does not call back into the Runtime.
            Status status = call_data_path_status_(
                **it, [&] { return (*it)->shutdown(0); });
            if (!status.ok()) {
                return status;
            }
        }
        initialized_data_paths_.clear();
        progress_gates_.clear();
        state_.store(RuntimeState::STOPPED);
        (void)lock;
        return Status::Ok();
    }

    Status initialize_components_(RuntimeComponents components) {
        if (components.resolvers.empty() && components.data_paths.empty()) {
            return Status::Ok();
        }
        if (components.resolvers.empty() || components.data_paths.empty()) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "component runtime requires resolver and DataPath bindings");
        }

        // Complete all validation before touching a DataPath.  In particular,
        // a binding conflict must not leave an initialized component behind.
        Status preflight = preflight_components_(components);
        if (!preflight.ok()) {
            return preflight;
        }

        resources_ = components.resources ? components.resources : &default_resources_;
        for (const auto& binding : components.resolvers) {
            if (binding.scheme.empty() || binding.resolver == nullptr) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "resolver binding requires scheme and resolver");
            }
            if (!resolvers_.emplace(binding.scheme, binding.resolver).second) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "duplicate resolver scheme: " + binding.scheme);
            }
        }
        for (const auto& binding : components.data_paths) {
            if (binding.key.empty() || binding.data_path == nullptr) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "DataPath binding requires key and DataPath");
            }
            if (!data_paths_.emplace(binding.key, binding.data_path).second) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "duplicate DataPath key: " + binding.key);
            }
            if (std::find(initialized_data_paths_.begin(),
                          initialized_data_paths_.end(), binding.data_path)
                != initialized_data_paths_.end()) {
                continue;
            }
            DataPathConfig config = binding.config;
            if (config.name.empty()) {
                config.name = binding.key;
            }
            Status status = [&]() {
                DeviceGuard guard(data_path_accel_id_(*binding.data_path));
                if (!guard.ok()) return guard.status();
                Status initialize_status = binding.data_path->initialize(
                    config, *resources_);
                Status restore_status = guard.restore();
                if (!restore_status.ok()) return restore_status;
                return initialize_status;
            }();
            if (!status.ok()) {
                for (DataPath* initialized : initialized_data_paths_) {
                    (void)call_data_path_status_(*initialized,
                                                 [&] { return initialized->shutdown(0); });
                }
                initialized_data_paths_.clear();
                return status;
            }
            initialized_data_paths_.push_back(binding.data_path);
            progress_gates_.emplace(binding.data_path,
                                     std::make_unique<std::mutex>());
        }
        components_enabled_ = true;
        return Status::Ok();
    }

    static bool profile_matches_(std::string_view requested) {
        const std::string_view compiled = TUTTI_COMPILED_ACCELERATOR_PROFILE;
        if (requested.size() != compiled.size()) {
            return false;
        }
        for (std::size_t i = 0; i < requested.size(); ++i) {
            const char a = requested[i] >= 'a' && requested[i] <= 'z'
                ? static_cast<char>(requested[i] - 'a' + 'A') : requested[i];
            const char b = compiled[i] >= 'a' && compiled[i] <= 'z'
                ? static_cast<char>(compiled[i] - 'a' + 'A') : compiled[i];
            if (a != b) {
                return false;
            }
        }
        return true;
    }

    static bool compiled_profile_is_host_() {
        return profile_matches_("HOST");
    }

    static Result<int> backend_device_count_() {
#if defined(TUTTI_USE_HOST)
        return 0;
#elif defined(TUTTI_USE_CUDA) || defined(TUTTI_USE_MUSA) || defined(TUTTI_USE_MACA)
        int count = 0;
        const auto error = cudaGetDeviceCount(&count);
        if (error != cudaSuccess) {
            return Result<int>::Failure(
                Status(StatusCode::NOT_FOUND,
                       "compiled accelerator backend has no available devices"));
        }
        return count;
#else
        return Result<int>::Failure(
            Status(StatusCode::UNSUPPORTED,
                   "no compiled accelerator backend is available"));
#endif
    }

    Result<DeviceInfo> discover_device_(int accel_id) const {
#if defined(TUTTI_USE_CUDA) || defined(TUTTI_USE_MUSA) || defined(TUTTI_USE_MACA)
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, accel_id) != cudaSuccess) {
            return Result<DeviceInfo>::Failure(
                Status(StatusCode::DEVICE_ERROR, "failed to query accelerator properties"));
        }
        char pci[32]{};
        (void)cudaDeviceGetPCIBusId(
            pci, static_cast<int>(sizeof(pci)), accel_id);
        return DeviceInfo{accel_id, prop.name, prop.totalGlobalMem, pci};
#elif defined(TUTTI_USE_HOST)
        (void)accel_id;
        return Result<DeviceInfo>::Failure(
            Status(StatusCode::UNSUPPORTED,
                   "HOST profile has no accelerator device identity"));
#else
        (void)accel_id;
        return Result<DeviceInfo>::Failure(
            Status(StatusCode::UNSUPPORTED,
                   "accelerator backend device discovery is unavailable"));
#endif
    }

    Status validate_runtime_config_() const {
        if (!profile_matches_(config_.profile_name)) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "Runtime profile does not match compiled accelerator profile");
        }
        if (config_.accel_id < -1) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "accel_id must be -1 or non-negative");
        }
        if (config_.accel_id == -1) {
            return Status::Ok();
        }
        auto count = backend_device_count_();
        if (!count.ok()) {
            return count.status();
        }
        if (config_.accel_id >= count.value()) {
            return Status(StatusCode::NOT_FOUND,
                          "accel_id is outside the compiled backend device count");
        }
        return Status::Ok();
    }

    Status preflight_components_(const RuntimeComponents& components) const {
        std::unordered_map<std::string, bool> resolver_keys;
        for (const auto& binding : components.resolvers) {
            if (binding.scheme.empty() || binding.resolver == nullptr) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "resolver binding requires scheme and resolver");
            }
            if (!resolver_keys.emplace(binding.scheme, true).second) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "duplicate resolver scheme: " + binding.scheme);
            }
        }
        std::unordered_map<std::string, bool> path_keys;
        for (const auto& binding : components.data_paths) {
            if (binding.key.empty() || binding.data_path == nullptr) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "DataPath binding requires key and DataPath");
            }
            if (!path_keys.emplace(binding.key, true).second) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "duplicate DataPath key: " + binding.key);
            }
            const DataPathCapabilities& caps = binding.data_path->capabilities();
            if (caps.bound_accel_id < -1) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "DataPath bound_accel_id must be -1 or non-negative");
            }
            if (caps.bound_accel_id >= 0 &&
                caps.bound_accel_id != config_.accel_id) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "DataPath accelerator binding conflicts with Runtime");
            }
            if (config_.accel_id == -1 && caps.bound_accel_id >= 0) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "accelerator-bound DataPath cannot attach to host-only Runtime");
            }
        }
        return Status::Ok();
    }

    static std::string scheme_for_(std::string_view uri,
                                   const OpenOptions& options) {
        const std::size_t separator = uri.find("://");
        if (separator != std::string_view::npos) {
            return std::string(uri.substr(0, separator));
        }
        return options.scheme;
    }

    Status unregister_data_path_memory_(MemoryEntry& entry) {
        for (auto it = entry.data_path_registrations.rbegin();
             it != entry.data_path_registrations.rend(); ++it) {
            Status status = call_data_path_status_(
                *it->data_path,
                [&] { return it->data_path->unregister_memory(it->memory); });
            if (!status.ok()) {
                return status;
            }
        }
        entry.data_path_registrations.clear();
        return Status::Ok();
    }

    Result<DataPathMemory> registration_for_(MemoryEntry& memory,
                                              const TargetEntry& target) {
        for (const auto& registration : memory.data_path_registrations) {
            if (registration.data_path == target.data_path &&
                registration.domain == target.registration_domain) {
                return registration.memory;
            }
        }

        DataPathMemoryKind kind =
            (memory.kind == MemoryKind::DEVICE ||
             memory.kind == MemoryKind::MANAGED)
            ? DataPathMemoryKind::DEVICE
            : DataPathMemoryKind::HOST;
        DataPathMemoryView view{memory.address, memory.size, memory.accel_id, kind};
        auto registration = call_data_path_result_<DataPathMemory>(
            *target.data_path,
            [&] {
                return target.data_path->register_memory(
                    view, target.registration_domain);
            });
        if (!registration.ok()) {
            return Result<DataPathMemory>::Failure(registration.status());
        }
        memory.data_path_registrations.push_back(
            DataPathMemoryRegistration{target.data_path,
                                       target.registration_domain,
                                       registration.value()});
        return registration.value();
    }

    Status validate_pointer_accel_(void* address, MemoryKind kind,
                                   std::int32_t expected_accel_id) const {
        const bool device_memory = kind == MemoryKind::DEVICE ||
                                   kind == MemoryKind::MANAGED;
        if (!device_memory) return Status::Ok();
        if (config_.accel_id < 0) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "accelerator memory cannot belong to a host-only Runtime");
        }
        const std::int32_t effective_accel_id = expected_accel_id >= 0
            ? expected_accel_id : config_.accel_id;
        if (effective_accel_id != config_.accel_id) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "device pointer accelerator does not match Runtime");
        }
#if defined(TUTTI_USE_HOST)
        // HOST's shim has no device pointers.  DEVICE/MANAGED registration is
        // rejected by the host-only Runtime above.
        (void)address;
        return Status(StatusCode::UNSUPPORTED,
                      "HOST profile cannot query accelerator pointer ownership");
#elif defined(TUTTI_USE_CUDA) || defined(TUTTI_USE_MUSA) || defined(TUTTI_USE_MACA)
        if (address == nullptr) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "accelerator pointer must be non-null");
        }
        cudaPointerAttributes attributes{};
        const cudaError_t error = cudaPointerGetAttributes(&attributes, address);
        if (error != cudaSuccess) {
            return Status(StatusCode::DEVICE_ERROR,
                          "accelerator pointer ownership query failed: " +
                          std::string(cudaGetErrorString(error)));
        }
#if defined(TUTTI_USE_CUDA)
        if (attributes.type != cudaMemoryTypeDevice &&
            attributes.type != cudaMemoryTypeManaged) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "pointer is not device/managed accelerator memory");
        }
#elif defined(TUTTI_USE_MUSA) || defined(TUTTI_USE_MACA)
        // The vendor shims expose device ordinal but do not provide a
        // stable managed-memory type contract yet.  Do not accept a managed
        // pointer on an unverifiable backend.
        if (kind == MemoryKind::MANAGED ||
            attributes.type != cudaMemoryTypeDevice) {
            return Status(StatusCode::UNSUPPORTED,
                          "backend cannot verify managed/device pointer kind");
        }
#endif
        if (attributes.device != config_.accel_id) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "accelerator pointer belongs to a different Runtime");
        }
        return Status::Ok();
#else
        (void)address;
        return Status(StatusCode::UNSUPPORTED,
                      "accelerator pointer ownership query is unavailable");
#endif
    }

    Status validate_component_request_(const IoRequest& request,
                                       const TargetEntry& target,
                                       const MemoryEntry& memory,
                                       const HostSubmitContext& context) const {
        const DataPathCapabilities& caps = target.data_path->capabilities();
        // HOST_EXECUTION has no accelerator ownership.  Preserve the
        // historical host contract where callers may leave a legacy ordinal
        // in this field; it is ignored when this Runtime itself is host-only.
        // Accelerator Runtimes still reject an explicit mismatching ordinal.
        if (context.accel_id >= 0 && config_.accel_id >= 0 &&
            context.accel_id != config_.accel_id) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "submit accelerator does not match Runtime");
        }
        if (context.execution_domain == ExecutionDomain::DEVICE_EXECUTION &&
            context.accel_id >= 0 && config_.accel_id < 0) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "submit accelerator does not match Runtime");
        }
        if (context.execution_domain == ExecutionDomain::HOST_EXECUTION &&
            !caps.supports_host_execution) {
            return Status(StatusCode::UNSUPPORTED,
                          "DataPath does not support HOST_EXECUTION");
        }
        if (context.execution_domain == ExecutionDomain::DEVICE_EXECUTION) {
            const std::int32_t effective_accel_id = context.accel_id >= 0
                ? context.accel_id : config_.accel_id;
            if (config_.accel_id < 0 || effective_accel_id != config_.accel_id) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "submit accelerator does not match Runtime");
            }
            if (memory.kind == MemoryKind::DEVICE ||
                memory.kind == MemoryKind::MANAGED) {
                if (memory.accel_id >= 0 && memory.accel_id != config_.accel_id) {
                    return Status(StatusCode::INVALID_ARGUMENT,
                                  "device memory accelerator does not match Runtime");
                }
            }
            if (!caps.supports_device_execution) {
                return Status(StatusCode::UNSUPPORTED,
                              "DataPath does not support DEVICE_EXECUTION");
            }
            if (context.stream == nullptr) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "DEVICE_EXECUTION requires a non-null stream");
            }
#if defined(TUTTI_USE_CUDA)
            int stream_accel_id = -1;
            const cudaError_t stream_error =
                cudaStreamGetDevice(context.stream, &stream_accel_id);
            if (stream_error != cudaSuccess) {
                return Status(StatusCode::DEVICE_ERROR,
                              "stream accelerator ownership query failed: " +
                              std::string(cudaGetErrorString(stream_error)));
            }
            if (stream_accel_id != config_.accel_id) {
                return Status(StatusCode::INVALID_ARGUMENT,
                              "stream belongs to a different Runtime accelerator");
            }
#else
            // MUSA/MACA shims currently have no portable stream-owner query;
            // callers must create/pass the stream on the Runtime accelerator.
            // Pointer ownership remains fail-closed where the shim exposes it.
#endif
        }
        Status pointer_status = validate_pointer_accel_(
            memory.address, memory.kind, memory.accel_id);
        if (!pointer_status.ok()) return pointer_status;
        const bool device_memory = memory.kind == MemoryKind::DEVICE ||
                                   memory.kind == MemoryKind::MANAGED;
        if (device_memory ? !caps.supports_device_memory
                          : !caps.supports_host_memory) {
            return Status(StatusCode::UNSUPPORTED,
                          "DataPath does not support the request memory kind");
        }
        if ((request.direction == IoDirection::READ && !caps.supports_read) ||
            (request.direction == IoDirection::WRITE && !caps.supports_write)) {
            return Status(StatusCode::UNSUPPORTED,
                          "DataPath does not support the request direction");
        }
        if (caps.max_single_io_bytes != 0 &&
            request.length > caps.max_single_io_bytes) {
            return Status(StatusCode::OUT_OF_RANGE,
                          "request exceeds DataPath max_single_io_bytes");
        }
        const auto aligned = [](std::uint64_t value, std::uint64_t alignment) {
            return alignment != 0 && value % alignment == 0;
        };
        if (!aligned(request.target_offset, caps.target_alignment_bytes) ||
            !aligned(request.memory_offset, caps.memory_alignment_bytes) ||
            !aligned(request.length, caps.length_alignment_bytes)) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "request violates DataPath alignment");
        }
        return Status::Ok();
    }

    IoSubmitOutcome submit_component_backed_(
        const IoRequest* requests, std::size_t count,
        const HostSubmitContext& context) {
        IoSubmitOutcome outcome;
        outcome.initial_states.resize(count);

        const auto reject_all = [&](Status status) {
            outcome.status = status;
            for (auto& state : outcome.initial_states) {
                state.state = IoRequestState::REJECTED;
                state.status = status;
            }
        };
        if (state_.load() != RuntimeState::RUNNING) {
            reject_all(Status(StatusCode::NOT_READY,
                              "runtime is not in RUNNING state"));
            return outcome;
        }
        if (count == 0) {
            outcome.status = Status::Ok();
            return outcome;
        }
        if (requests == nullptr) {
            reject_all(Status(StatusCode::INVALID_ARGUMENT, "null requests"));
            return outcome;
        }
        if (terminal_result_count_ >= config_.max_terminal_results) {
            reject_all(Status(StatusCode::RESOURCE_EXHAUSTED,
                              "terminal result limit reached"));
            return outcome;
        }

        // PendingGroup groups requests by DataPath only: a single
        // DataPath::submit call may carry requests for multiple targets
        // within that DataPath (each DataPathRequest carries its own
        // target). See spi/data_path.h for the SPI contract.
        struct PendingGroup {
            DataPath* data_path = nullptr;
            std::vector<std::size_t> indices;
            std::vector<DataPathRequest> requests;
        };
        std::vector<PendingGroup> groups;
        Status first_rejection(StatusCode::INVALID_ARGUMENT,
                               "all requests rejected");
        bool have_rejection = false;

        const auto reject_one = [&](std::size_t index, Status status) {
            outcome.initial_states[index].state = IoRequestState::REJECTED;
            outcome.initial_states[index].status = status;
            if (!have_rejection) {
                first_rejection = std::move(status);
                have_rejection = true;
            }
        };

        for (std::size_t index = 0; index < count; ++index) {
            const IoRequest& request = requests[index];
            Status status = validate_request_(request);
            if (!status.ok()) {
                reject_one(index, std::move(status));
                continue;
            }
            TargetEntry& target = target_entries_[request.target.slot_];
            MemoryEntry& memory = memory_entries_[request.memory.slot_];
            if (target.data_path == nullptr) {
                reject_one(index, Status(StatusCode::NOT_READY,
                                         "target has no DataPath"));
                continue;
            }
            status = validate_component_request_(request, target, memory, context);
            if (!status.ok()) {
                reject_one(index, std::move(status));
                continue;
            }
            auto registration = registration_for_(memory, target);
            if (!registration.ok()) {
                reject_one(index, registration.status());
                continue;
            }

            PendingGroup* group = nullptr;
            for (auto& candidate : groups) {
                if (candidate.data_path == target.data_path) {
                    group = &candidate;
                    break;
                }
            }
            if (group == nullptr) {
                groups.push_back(PendingGroup{});
                group = &groups.back();
                group->data_path = target.data_path;
            }
            group->indices.push_back(index);
            group->requests.push_back(DataPathRequest{
                request, registration.value(), target.data_path_target});
        }

        if (groups.empty()) {
            outcome.status = first_rejection;
            return outcome;
        }

        // Reserve the public operation record and grant inflight credits
        // BEFORE the first DataPath call.  The credits prevent close/
        // unregister from tearing down a target/memory while DataPath::
        // submit is in flight (it may ring a doorbell irreversibly).
        const std::uint32_t slot = find_free_io_slot_();
        const std::uint64_t generation = ++io_gen_counter_;
        IoEntry& entry = io_entries_[slot];
        entry.active = true;
        entry.generation = generation;
        entry.state = IoState::IN_FLIGHT;
        entry.terminal = false;
        entry.released = false;
        entry.references_released = false;
        entry.io_status = Status::Ok();
        entry.data_path_operations.clear();
        entry.memory_slots.clear();
        entry.target_slots.clear();

        // Grant one inflight credit per accepted-request candidate so
        // that close/unregister observes BUSY while DataPath::submit
        // runs.  Credits are rolled back for requests that the DataPath
        // ultimately rejects.
        //
        // DataPath::submit is called UNDER the registry lock.  Rationale:
        // DataPath implementations (LocalNvmeDataPath, fakes) are not
        // guaranteed thread-safe for concurrent submit() calls, so the
        // Runtime must serialize them.  submit is control-plane (not a
        // hot data path), and DataPath::submit does not call back into
        // the Runtime (no deadlock).  progress(), the hot path, runs
        // outside the lock with a per-DataPath serialization gate.
        for (auto& group : groups) {
            for (std::size_t index : group.indices) {
                memory_entries_[requests[index].memory.slot_].inflight_count++;
                target_entries_[requests[index].target.slot_].inflight_count++;
                entry.memory_slots.push_back(requests[index].memory.slot_);
                entry.target_slots.push_back(requests[index].target.slot_);
            }
        }

        bool any_accepted = false;
        for (auto& group : groups) {
            // Acquire the per-DataPath progress gate during submit so that
            // progress() cannot run concurrently on the same DataPath.
            // This protects DataPath implementations whose submit() and
            // progress() share internal state (e.g. LocalNvmeDataPath's
            // ops_ map) without requiring them to be thread-safe.
            // No deadlock: drive_progress_unlocked_ releases the registry
            // lock before acquiring the progress gate, so a progress thread
            // never holds the registry lock while waiting for the gate.
            auto gate_it = progress_gates_.find(group.data_path);
            std::unique_ptr<std::mutex>* gate_ptr =
                (gate_it != progress_gates_.end()) ? &gate_it->second : nullptr;

            SubmitOutcome submitted;
            Status guard_status = Status::Ok();
            {
                DeviceGuard guard(data_path_accel_id_(*group.data_path));
                if (!guard.ok()) {
                    guard_status = guard.status();
                } else if (gate_ptr) {
                    std::lock_guard<std::mutex> pg(**gate_ptr);
                    submitted = group.data_path->submit(
                        group.requests.data(), group.requests.size(), context);
                    guard_status = guard.restore();
                } else {
                    submitted = group.data_path->submit(
                        group.requests.data(), group.requests.size(), context);
                    guard_status = guard.restore();
                }
            }
            if (!guard_status.ok()) {
                // A DataPath may have accepted the group before the outer
                // Runtime guard failed while restoring the caller device.
                // Reclaim that opaque operation while the DataPath target
                // remains selected; otherwise rolling back only Runtime
                // credits would leak device-side state.
                if (submitted.op.has_value()) {
                    (void)call_data_path_status_(
                        *group.data_path,
                        [&] { return group.data_path->release(submitted.op.value()); });
                }
                for (std::size_t index : group.indices) {
                    reject_one(index, guard_status);
                    auto& mem_e = memory_entries_[requests[index].memory.slot_];
                    auto& tgt_e = target_entries_[requests[index].target.slot_];
                    if (mem_e.inflight_count > 0) --mem_e.inflight_count;
                    if (tgt_e.inflight_count > 0) --tgt_e.inflight_count;
                }
                continue;
            }
            if (submitted.initial_states.size() != group.requests.size()) {
                if (submitted.op.has_value()) {
                    (void)call_data_path_status_(
                        *group.data_path,
                        [&] { return group.data_path->release(submitted.op.value()); });
                }
                for (std::size_t index : group.indices) {
                    reject_one(index, Status(StatusCode::INTERNAL,
                                             "DataPath returned invalid state count"));
                    auto& mem_e = memory_entries_[requests[index].memory.slot_];
                    auto& tgt_e = target_entries_[requests[index].target.slot_];
                    if (mem_e.inflight_count > 0) --mem_e.inflight_count;
                    if (tgt_e.inflight_count > 0) --tgt_e.inflight_count;
                }
                continue;
            }
            const bool has_op = submitted.op.has_value();
            bool group_has_accepted = false;
            for (std::size_t local = 0; local < group.indices.size(); ++local) {
                const std::size_t index = group.indices[local];
                const RequestInitialState& state = submitted.initial_states[local];
                if (state.state == RequestState::ACCEPTED && has_op) {
                    outcome.initial_states[index].state = IoRequestState::ACCEPTED;
                    outcome.initial_states[index].status = Status::Ok();
                    any_accepted = true;
                    group_has_accepted = true;
                    // Credit already granted; keep it.
                } else {
                    reject_one(index, state.state == RequestState::REJECTED
                                      ? state.status
                                      : Status(StatusCode::INTERNAL,
                                               "DataPath accepted request without operation"));
                    // Roll back the credit for this rejected request.
                    auto& mem_e = memory_entries_[requests[index].memory.slot_];
                    auto& tgt_e = target_entries_[requests[index].target.slot_];
                    if (mem_e.inflight_count > 0) --mem_e.inflight_count;
                    if (tgt_e.inflight_count > 0) --tgt_e.inflight_count;
                }
            }
            if (has_op && group_has_accepted) {
                entry.data_path_operations.push_back(
                    DataPathOperation{group.data_path, submitted.op.value()});
            } else if (has_op) {
                // DataPath returned an op but no request in this group
                // was accepted — release the orphan op.
                (void)call_data_path_status_(
                    *group.data_path,
                    [&] { return group.data_path->release(submitted.op.value()); });
            }
            if (!submitted.status.ok() && !group_has_accepted) {
                have_rejection = true;
                if (first_rejection.ok()) {
                    first_rejection = submitted.status;
                }
            }
        }

        if (!any_accepted) {
            // Roll back all credits and retire the entry.
            release_inflight_references_(entry);
            entry.data_path_operations.clear();
            entry.active = false;
            outcome.status = first_rejection;
            return outcome;
        }
        outcome.io = IoHandle(runtime_id_, slot, generation);
        outcome.status = have_rejection
            ? Status(first_rejection.code(),
                     "partial submit: " + first_rejection.message())
            : Status::Ok();
        return outcome;
    }

    // ---- per-request bounds validation (gap 2) ----
    //
    // Overflow-safe: checks offset > size first, then length > size - offset.
    // Never computes offset + length (which could wrap).

    Status validate_request_(const IoRequest& req) const {
        // length == 0
        if (req.length == 0) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "request length must be > 0");
        }

        // memory handle validity
        if (!validate_memory_(req.memory)) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "invalid memory handle in request");
        }

        // target handle validity
        if (!validate_target_(req.target)) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "invalid target handle in request");
        }

        std::uint64_t mem_size = memory_entries_[req.memory.slot_].size;
        // memory bounds (overflow-safe)
        if (req.memory_offset > mem_size) {
            return Status(StatusCode::OUT_OF_RANGE,
                          "memory_offset exceeds memory size");
        }
        if (req.length > mem_size - req.memory_offset) {
            return Status(StatusCode::OUT_OF_RANGE,
                          "memory_offset + length exceeds memory size");
        }

        std::uint64_t tgt_size =
            target_entries_[req.target.slot_].logical_size;
        // target bounds (overflow-safe)
        if (req.target_offset > tgt_size) {
            return Status(StatusCode::OUT_OF_RANGE,
                          "target_offset exceeds target size");
        }
        if (req.length > tgt_size - req.target_offset) {
            return Status(StatusCode::OUT_OF_RANGE,
                          "target_offset + length exceeds target size");
        }

        return Status::Ok();
    }

    void release_inflight_references_(IoEntry& entry) {
        if (entry.references_released) {
            return;
        }
        for (std::uint32_t slot : entry.memory_slots) {
            if (slot < memory_entries_.size() &&
                memory_entries_[slot].inflight_count > 0) {
                --memory_entries_[slot].inflight_count;
            }
        }
        for (std::uint32_t slot : entry.target_slots) {
            if (slot < target_entries_.size() &&
                target_entries_[slot].inflight_count > 0) {
                --target_entries_[slot].inflight_count;
            }
        }
        entry.references_released = true;
    }

    void finish_io_(IoEntry& entry, IoState state, Status status) {
        if (entry.terminal) {
            return;
        }
        entry.state = state;
        entry.terminal = true;
        entry.io_status = std::move(status);
        ++terminal_result_count_;
        release_inflight_references_(entry);
        // Notify any waiters blocked on shutdown()/wait().
        io_cv_.notify_all();
    }

    // Harvest terminal states from DataPath query for one io entry.
    // Must be called with registry_mutex_ held.  DataPath::query is const
    // and does not call back into the Runtime, so it is safe under the lock.
    //
    // If a DataPath::query returns an error (not OK), the sub-op is
    // treated as terminally FAILED — this prevents the Runtime op from
    // being stuck in IN_FLIGHT forever when the DataPath reports a
    // query/release failure.
    // Helper: look up the per-DataPath progress gate.
    std::mutex* progress_gate_for_(DataPath* dp) const {
        auto it = progress_gates_.find(dp);
        return (it != progress_gates_.end()) ? it->second.get() : nullptr;
    }

    void refresh_component_io_(IoEntry& entry) {
        if (entry.terminal || entry.data_path_operations.empty()) {
            return;
        }

        bool all_terminal = true;
        bool failed = false;
        Status first_failure = Status::Ok();
        for (const auto& sub_op : entry.data_path_operations) {
            // Acquire the progress gate so query() does not overlap with
            // a concurrent progress() on the same DataPath.
            auto* gate = progress_gate_for_(sub_op.data_path);
            std::unique_lock<std::mutex> gate_lock;
            if (gate) gate_lock = std::unique_lock<std::mutex>(*gate);
            auto snapshot = call_data_path_result_<DataPathSnapshot>(
                *sub_op.data_path,
                [&] { return sub_op.data_path->query(sub_op.op); });
            gate_lock.unlock();

            if (!snapshot.ok()) {
                // DataPath query error → terminal FAILED.  Do not wait
                // for other sub-ops; the op is unrecoverable.
                failed = true;
                if (first_failure.ok()) {
                    first_failure = snapshot.status();
                }
                continue;
            }
            if (snapshot.value().state == OpState::IN_FLIGHT) {
                all_terminal = false;
            } else if (snapshot.value().state == OpState::FAILED) {
                failed = true;
                if (first_failure.ok()) {
                    first_failure = snapshot.value().status;
                }
            }
        }
        if (!all_terminal && !failed) {
            // Still in flight and no query errors — keep waiting.
            return;
        }

        // Release all DataPath ops that are safe to release (terminal or
        // query-failed).  Ops still IN_FLIGHT (because a different sub-op
        // had a query error) are left for a subsequent progress cycle.
        for (const auto& sub_op : entry.data_path_operations) {
            auto* gate = progress_gate_for_(sub_op.data_path);
            std::unique_lock<std::mutex> gate_lock;
            if (gate) gate_lock = std::unique_lock<std::mutex>(*gate);
            (void)call_data_path_status_(
                *sub_op.data_path,
                [&] { return sub_op.data_path->release(sub_op.op); });
        }
        entry.data_path_operations.clear();
        finish_io_(entry, failed ? IoState::FAILED : IoState::COMPLETED,
                   failed ? std::move(first_failure) : Status::Ok());
    }

    // Drives each DataPath at most once and then harvests the states of all
    // runtime operations.  The caller must hold registry_mutex_ on entry;
    // the lock is released while DataPath::progress runs (so that
    // different DataPaths can progress concurrently and the registry is
    // not held during potentially-blocking device completion polls), then
    // re-acquired to refresh io states.
    void drive_progress_unlocked_(std::unique_lock<std::mutex>& lock) {
        // Collect the set of DataPaths that have non-terminal operations
        // and snapshot the per-DataPath progress mutexes.
        std::vector<std::pair<DataPath*, std::mutex*>> gates;
        for (const auto& entry : io_entries_) {
            if (!entry.active || entry.terminal) {
                continue;
            }
            for (const auto& sub_op : entry.data_path_operations) {
                auto it = progress_gates_.find(sub_op.data_path);
                if (it == progress_gates_.end()) continue;
                bool seen = false;
                for (auto& g : gates) {
                    if (g.first == sub_op.data_path) { seen = true; break; }
                }
                if (!seen) {
                    gates.emplace_back(sub_op.data_path, it->second.get());
                }
            }
        }
        if (gates.empty()) {
            // Still refresh in case query() alone can advance terminal ops.
            for (auto& entry : io_entries_) {
                refresh_component_io_(entry);
            }
            return;
        }

        lock.unlock();
        std::vector<std::pair<DataPath*, Status>> progress_errors;
        for (auto& [path, gate] : gates) {
            // Per-DataPath serialization: at most one progress() call per
            // DataPath at a time.  Different DataPaths progress concurrently.
            std::lock_guard<std::mutex> pg(*gate);
            DeviceGuard device_guard(data_path_accel_id_(*path));
            if (!device_guard.ok()) {
                progress_errors.emplace_back(path, device_guard.status());
                continue;
            }
            auto progress_result = path->progress(ProgressBudget{16, 1000000});
            if (!progress_result.ok()) {
                progress_errors.emplace_back(path, progress_result.status());
            }
            Status restore_status = device_guard.restore();
            if (!restore_status.ok()) {
                progress_errors.emplace_back(path, std::move(restore_status));
            }
        }
        lock.lock();

        for (const auto& error : progress_errors) {
            for (auto& entry : io_entries_) {
                if (entry.active && !entry.terminal) {
                    for (const auto& sub_op : entry.data_path_operations) {
                        if (sub_op.data_path == error.first) {
                            finish_io_(entry, IoState::FAILED, error.second);
                            break;
                        }
                    }
                }
            }
        }
        for (auto& entry : io_entries_) {
            refresh_component_io_(entry);
        }
    }

    // ---- testing-only: force-complete an inflight stub IO ----
    //
    // Component-backed IO must be completed by its DataPath. The hook remains
    // solely for the legacy hardware-free stub test path.
    Status testing_force_complete_io_(const IoHandle& handle,
                                      IoState terminal_state,
                                      Status io_status = Status::Ok()) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        if (!validate_io_(handle)) {
            return Status(StatusCode::NOT_FOUND, "invalid IO handle");
        }
        IoEntry& entry = io_entries_[handle.slot_];
        if (entry.terminal) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "IO is already terminal");
        }
        if (!entry.data_path_operations.empty()) {
            return Status(StatusCode::UNSUPPORTED,
                          "component-backed IO is completed by its DataPath");
        }
        finish_io_(entry, terminal_state, std::move(io_status));
        return Status::Ok();
    }

    // ---- handle validation ----

    bool validate_memory_(const MemoryHandle& h) const {
        if (state_.load() == RuntimeState::STOPPED) return false;
        if (!h.valid()) return false;
        if (h.runtime_id_ != runtime_id_) return false;
        if (h.slot_ >= memory_entries_.size()) return false;
        const auto& e = memory_entries_[h.slot_];
        if (!e.active) return false;
        if (e.generation != h.generation_) return false;
        return true;
    }

    bool validate_target_(const TargetHandle& h) const {
        if (state_.load() == RuntimeState::STOPPED) return false;
        if (!h.valid()) return false;
        if (h.runtime_id_ != runtime_id_) return false;
        if (h.slot_ >= target_entries_.size()) return false;
        const auto& e = target_entries_[h.slot_];
        if (!e.active) return false;
        if (e.generation != h.generation_) return false;
        return true;
    }

    bool validate_io_(const IoHandle& h) const {
        if (state_.load() == RuntimeState::STOPPED) return false;
        if (!h.valid()) return false;
        if (h.runtime_id_ != runtime_id_) return false;
        if (h.slot_ >= io_entries_.size()) return false;
        const auto& e = io_entries_[h.slot_];
        if (!e.active) return false;
        if (e.generation != h.generation_) return false;
        return true;
    }

    // ---- slot management ----

    std::uint32_t find_free_memory_slot_() {
        for (std::size_t i = 0; i < memory_entries_.size(); ++i) {
            if (!memory_entries_[i].active) {
                return static_cast<std::uint32_t>(i);
            }
        }
        memory_entries_.push_back({});
        return static_cast<std::uint32_t>(memory_entries_.size() - 1);
    }

    std::uint32_t find_free_target_slot_() {
        for (std::size_t i = 0; i < target_entries_.size(); ++i) {
            if (!target_entries_[i].active) {
                return static_cast<std::uint32_t>(i);
            }
        }
        target_entries_.push_back({});
        return static_cast<std::uint32_t>(target_entries_.size() - 1);
    }

    std::uint32_t find_free_io_slot_() {
        for (std::size_t i = 0; i < io_entries_.size(); ++i) {
            if (!io_entries_[i].active) {
                return static_cast<std::uint32_t>(i);
            }
        }
        io_entries_.push_back({});
        return static_cast<std::uint32_t>(io_entries_.size() - 1);
    }

    std::size_t count_inflight_io_() const {
        std::size_t count = 0;
        for (const auto& e : io_entries_) {
            if (e.active && !e.terminal) {
                ++count;
            }
        }
        return count;
    }

    // ---- members ----

    inline static std::atomic<std::uint32_t> next_runtime_id_{1};

    std::uint32_t runtime_id_;
    RuntimeConfig config_;
    std::atomic<RuntimeState> state_{RuntimeState::RUNNING};

    // Registry lock: protects all registry data (memory/target/io entries,
    // gen counters, terminal_result_count_, state transitions).  DataPath
    // virtual calls that may block (submit, progress) are made WITHOUT
    // holding this lock; close/unregister/query/release are made under the
    // lock because they are non-blocking and do not call back into the
    // Runtime (no deadlock risk).
    mutable std::mutex registry_mutex_;
    // Notified by finish_io_ when an io entry becomes terminal.  Used by
    // shutdown() and wait() to avoid busy-polling.
    std::condition_variable io_cv_;
    // Per-DataPath progress serialization gate.  Ensures at most one
    // progress() call per DataPath at a time; different DataPaths progress
    // concurrently.
    std::unordered_map<DataPath*, std::unique_ptr<std::mutex>> progress_gates_;

    bool components_enabled_ = false;
    // Round 19 S1: serializes DataPath::open/registration_domain across
    // concurrent open_batch and open() calls — DataPath internals
    // (targets_ map, handle cache) are not thread-safe.
    std::mutex datapath_open_mutex_;
    ResourceProvider default_resources_;
    ResourceProvider* resources_ = &default_resources_;
    std::unordered_map<std::string, StorageTargetResolver*> resolvers_;
    std::unordered_map<std::string, DataPath*> data_paths_;
    std::vector<DataPath*> initialized_data_paths_;

    std::vector<MemoryEntry> memory_entries_;
    std::vector<TargetEntry> target_entries_;
    std::vector<IoEntry> io_entries_;

    std::uint64_t memory_gen_counter_ = 0;
    std::uint64_t target_gen_counter_ = 0;
    std::uint64_t io_gen_counter_ = 0;
    std::uint64_t terminal_result_count_ = 0;
};

// =========================================================================
// Testing-only access (gap 4).
//
// StorageRuntimeTestAccess is the sole non-member route to the private
// force-complete hook. Contract tests include this header and call
// StorageRuntimeTestAccess::force_complete_io(). Ordinary consumer code
// that does not use this struct cannot reach the private method.
//
// This is NOT part of the application public API; it exists only so that
// stub IO entries can reach a terminal state in hardware-free tests.
// =========================================================================

namespace testing {

struct StorageRuntimeTestAccess {
    static Status force_complete_io(StorageRuntime& rt,
                                    const IoHandle& handle,
                                    IoState terminal_state,
                                    Status io_status = Status::Ok()) {
        return rt.testing_force_complete_io_(handle, terminal_state,
                                             std::move(io_status));
    }
};

} // namespace testing

} // namespace tutti

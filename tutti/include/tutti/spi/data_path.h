#pragma once

// tutti/spi/data_path.h -- In-repo data-plane source-level SPI.
//
// Host-side abstract DataPath plus the value types that flow across it:
// capabilities, opaque target/memory/op identities, request lowering, submit
// outcome, progress budget/result, and op snapshots. This is an in-repo SPI
// (lives under tutti/spi/), NOT an application public noun; the application
// public nouns live under tutti/include/tutti/.
//
// No transport-private, device-private, or kernel-private types appear here.
// The SPI speaks only in opaque integer identities and byte ranges.
//
// Allowed includes: <tutti/status.h>, <tutti/io_types.h>, and the C++17
// standard library headers required below.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tutti/status.h>
#include <tutti/io_types.h>

namespace tutti {

// Forward declarations of runtime-owned types this SPI references by
// reference only. They are defined elsewhere (the target resolver and the
// runtime core); this header does not include their definitions, so it does
// not depend on those headers existing at parse time.
class ResolvedTarget;  // product of the target resolver

// Shared runtime-owned resource accessor passed to DataPath lifecycle hooks.
// It intentionally has no transport-specific surface: a DataPath that needs
// controller, service, or accelerator resources obtains them through its own
// injected implementation rather than extending this common contract.
class ResourceProvider {
public:
    virtual ~ResourceProvider() = default;
};

namespace detail {

struct SpiIdentityMint;  // forward

// Phantom-tagged opaque identity for in-repo SPI objects. Distinct
// instantiations are distinct strong types with no cross-type conversion.
// generation == 0 denotes an invalid (never-minted) identity. Layout is not
// part of any stable ABI.
//
// The carried token is an opaque integer identity, NEVER a pointer or object
// address. Valid identities are minted only through SpiIdentityMint, which
// concrete DataPath implementations use; application code never sees it.
template <typename Tag>
class OpaqueSpiIdentity {
public:
    constexpr OpaqueSpiIdentity() noexcept = default;

    constexpr bool valid() const noexcept { return generation_ != 0; }

    // Accessors for SPI implementations to route lookups by identity.
    constexpr std::uint64_t token() const noexcept { return token_; }
    constexpr std::uint64_t generation() const noexcept { return generation_; }

    constexpr bool operator==(const OpaqueSpiIdentity& other) const noexcept {
        return token_ == other.token_ && generation_ == other.generation_;
    }
    constexpr bool operator!=(const OpaqueSpiIdentity& other) const noexcept {
        return !(*this == other);
    }

private:
    friend struct SpiIdentityMint;
    constexpr OpaqueSpiIdentity(std::uint64_t token,
                                std::uint64_t generation) noexcept
        : token_(token), generation_(generation) {}

    std::uint64_t token_      = 0;
    std::uint64_t generation_ = 0;
};

// Minting seam for in-repo SPI opaque identities.
struct SpiIdentityMint {
    template <typename Tag>
    static OpaqueSpiIdentity<Tag> mint(std::uint64_t token,
                                       std::uint64_t generation) noexcept {
        return OpaqueSpiIdentity<Tag>(token, generation);
    }
};

struct DataPathTargetTag {};
struct DataPathMemoryTag {};
struct DataPathPagedMemoryTag {};
struct DataPathOpTag {};

} // namespace detail

using DataPathTarget = detail::OpaqueSpiIdentity<detail::DataPathTargetTag>;
using DataPathMemory = detail::OpaqueSpiIdentity<detail::DataPathMemoryTag>;
// Kept distinct from DataPathMemory so flat byte-range requests cannot be
// accidentally submitted against a block-table-backed cache.
using DataPathPagedMemory =
    detail::OpaqueSpiIdentity<detail::DataPathPagedMemoryTag>;
using DataPathOp     = detail::OpaqueSpiIdentity<detail::DataPathOpTag>;

// -------------------------------------------------------------------------
// RegistrationDomainKey
//
// An opaque string key naming a memory-registration domain (e.g. a device or
// address-space scope). It is never a controller/object pointer. DataPath
// implementations derive it from target identity, not from a raw address.
// -------------------------------------------------------------------------
struct RegistrationDomainKey {
    std::string value;

    bool operator==(const RegistrationDomainKey& other) const noexcept {
        return value == other.value;
    }
    bool operator!=(const RegistrationDomainKey& other) const noexcept {
        return value != other.value;
    }
};

// Memory kind offered for data-path registration.
//
// Deliberately a distinct type from the public tutti::MemoryKind (which has
// four values: HOST, PINNED_HOST, DEVICE, MANAGED).  The SPI only needs to
// distinguish host memory from device memory; pinned/managed distinction is
// the public layer's concern, not the SPI's.
enum class DataPathMemoryKind {
    HOST,
    DEVICE,
};

// DataPathMemoryView -- caller memory offered for data-path registration.
//
// `base` is the caller's memory address (input to registration), NOT an
// internal data-path object address. The returned DataPathMemory is the
// opaque identity the caller keeps.
struct DataPathMemoryView {
    void* base = nullptr;
    std::uint64_t size_bytes = 0;
    std::int32_t expected_accel_id = -1;  // -1 for unspecified/host memory
    DataPathMemoryKind kind = DataPathMemoryKind::HOST;
    // Round 16 S5 (V3): io_granularity > 0 enables registration-time PRP
    // pre-build (legacy build_io_slice_table 9-stage path).  0 = dynamic.
    std::uint64_t io_granularity = 0;
};

// One layer of a paged cache.  A successful implementation copies this
// descriptor during registration; the caller need not retain the array.
struct DataPathPagedLayerView {
    void* base = nullptr;
    std::uint64_t size_bytes = 0;
};

// Physical layout of a paged cache.  Each block is contiguous within one
// layer; block IDs in DataPathPagedRequest select those physical blocks.  No
// transport descriptor or on-disk format is exposed by this contract.
struct DataPathPagedMemoryView {
    const DataPathPagedLayerView* layers = nullptr;
    std::size_t layer_count = 0;
    std::uint64_t block_size_bytes = 0;
    std::uint32_t tokens_per_block = 0;
    std::uint64_t bytes_per_token = 0;
    std::int32_t expected_accel_id = -1;
    DataPathMemoryKind kind = DataPathMemoryKind::DEVICE;
};

// Minimal configuration handed to initialize().
struct DataPathConfig {
    std::string name;
};

// Registration scope of a DataPath's memory-registration domain.
enum class RegistrationScope {
    PER_TARGET,
    PER_DEVICE,
    GLOBAL,
};

// How a DataPath makes progress on submitted work.
enum class ProgressModel {
    HOST_POLL,          // host polls for progress
    DEVICE_AUTONOMOUS,  // device advances without host polling
};

// -------------------------------------------------------------------------
// DataPathCapabilities -- hard constraints, NOT hints.
//
// The runtime treats every field below as a hard contract. A request that
// cannot be satisfied within these constraints must be rejected before any
// irreversible submit.
// -------------------------------------------------------------------------
struct DataPathCapabilities {
    // identity / version
    std::string name;
    std::uint32_t source_api_version = 0;

    // execution domains
    bool supports_host_execution = false;
    bool supports_device_execution = false;

    // memory kinds
    bool supports_host_memory = false;
    bool supports_device_memory = false;

    // Block-table-aware paged registration/submit.  This is intentionally
    // separate from supports_direct: direct contiguous device IO does not
    // imply that a backend can resolve paged cache addresses.
    bool supports_paged_memory = false;

    // data movement
    bool supports_direct = false;
    bool supports_staged = false;

    // directions
    bool supports_read = false;
    bool supports_write = false;

    // alignment (bytes; 1 means byte-aligned)
    std::uint64_t target_alignment_bytes = 1;
    std::uint64_t memory_alignment_bytes = 1;
    std::uint64_t length_alignment_bytes = 1;

    // limits
    std::uint64_t max_single_io_bytes = 0;
    std::uint64_t max_batch_requests = 0;
    std::uint64_t max_batch_bytes = 0;
    std::uint64_t max_in_flight_operations = 0;

    // scatter-gather
    bool supports_scatter_gather = false;
    std::uint64_t max_scatter_gather_entries = 0;

    // registration scope
    RegistrationScope registration_scope = RegistrationScope::PER_TARGET;

    // progress model
    ProgressModel progress_model = ProgressModel::HOST_POLL;

    // device-execution completion semantics. These two are independent:
    //   device_completion_fence_on_caller_stream == true means a real IO
    //     completion fence can be established on the caller's stream.
    //   device_execution_autonomous == true means device execution does not
    //     depend on host query/wait to make progress.
    bool device_completion_fence_on_caller_stream = false;
    bool device_execution_autonomous = false;

    // concurrency
    bool supports_multi_stream = false;
    std::uint64_t max_concurrent_streams = 0;
    std::uint64_t max_concurrent_operations = 0;

    // multi-device
    // -1: host-only/unbound; >= 0: fixed accelerator device.
    // A device-executing DataPath must match its owning Runtime exactly.
    std::int32_t bound_accel_id = -1;
    bool supports_multi_gpu = false;
    bool supports_cross_device = false;

    // optional target features (open set, named by string; not a closed enum)
    std::vector<std::string> optional_target_features;
};

// -------------------------------------------------------------------------
// DataPathRequest -- one lowered request handed to submit().
//
// Carries the public byte-range intent plus the data-path memory/target
// identities produced by open()/register_memory(). No transport-private
// fields.
// -------------------------------------------------------------------------
struct DataPathRequest {
    IoRequest intent;       // public byte-range intent
    DataPathMemory memory;  // registered memory identity
    DataPathTarget target;  // opened target identity
};

// One block-table-aware request against a registered paged cache.  The
// block_ids vector is caller-owned metadata and must be copied by an async
// implementation before submit_paged() returns.
struct DataPathPagedRequest {
    IoDirection direction = IoDirection::READ;
    DataPathPagedMemory memory;
    DataPathTarget target;
    std::uint32_t layer_index = 0;
    std::vector<std::uint32_t> block_ids;
    std::uint64_t token_offset = 0;
    std::uint64_t token_count = 0;
    std::uint64_t target_offset = 0;
    std::uint64_t length = 0;
};

// Per-request initial state, one per input request, in input order.
enum class RequestState {
    ACCEPTED,  // issued or will be issued; observable via the op
    REJECTED,  // validation failed before any irreversible issue
};

struct RequestInitialState {
    RequestState state = RequestState::REJECTED;
    Status status;  // OK for ACCEPTED; error code+message for REJECTED
};

// -------------------------------------------------------------------------
// SubmitOutcome -- result of submit().
//
// Invariants:
//   * op == nullopt  => no transport request was irreversibly issued (zero
//     issued). status may still be non-OK (e.g. validation rejected all).
//   * op != nullopt  => at least one request was irreversibly issued and is
//     still observable, even if overall status reports a partial failure.
//   * initial_states.size() == input request count, in input order.
// -------------------------------------------------------------------------
struct SubmitOutcome {
    Status status;
    std::optional<DataPathOp> op;
    std::vector<RequestInitialState> initial_states;
};

// Operational state of a DataPathOp.
enum class OpState {
    IN_FLIGHT,  // not terminal; query() keeps the op alive
    COMPLETED,  // terminal success
    FAILED,     // terminal failure
};

// Structured terminal failure classification shared by DataPath and Runtime.
// NONE is required for successful and in-flight snapshots.
enum class IoFailureKind : std::uint8_t {
    NONE,
    RESOLVE_LBA,
    CQ_TIMEOUT,
    NVME_CQ_ERROR,
    CUDA_QUERY_ERROR,
    STATUS_D2H_ERROR,
    UNKNOWN,
};

// REQUEST_INDICES refers to the request array passed to the corresponding
// submit() call. WHOLE_OPERATION is the conservative answer when a backend
// cannot attribute a terminal failure without risking an incomplete bitmap.
enum class IoFailureScope : std::uint8_t {
    NONE,
    REQUEST_INDICES,
    WHOLE_OPERATION,
};

struct IoCompletionDetail {
    std::uint64_t confirmed_bytes = 0;
    bool timeout_seen = false;
    // UINT32_MAX means no failed entry was observed or attribution is unknown.
    std::uint32_t first_failed_entry = UINT32_MAX;
    IoFailureKind failure_kind = IoFailureKind::NONE;
    std::uint32_t raw_cq_status = 0;
    IoFailureScope failure_scope = IoFailureScope::NONE;
    std::vector<std::uint32_t> failed_request_indices;
};

// Snapshot returned by query(). query() never destroys the op.
struct DataPathSnapshot {
    OpState state = OpState::IN_FLIGHT;
    Status status;  // OK while IN_FLIGHT; terminal status otherwise
    std::uint64_t bytes_transferred = 0;
    IoCompletionDetail detail;
};

enum class FeederLayerState : std::uint8_t {
    PENDING,
    READY,
    FAILED,
};

// Budget handed to progress(). Both fields are hard caps on one call.
struct ProgressBudget {
    std::uint64_t max_work_units = 0;  // hard cap on work units consumed
    std::uint64_t timeout_ns = 0;      // wall-clock cap in nanoseconds
};

// Result of one progress() call. progress must be bounded by the budget;
// infinite busy-polling must never be disguised as progress.
struct ProgressResult {
    std::uint64_t work_units_consumed = 0;
    std::uint64_t operations_advanced = 0;
    std::uint64_t operations_terminal = 0;
    bool more_work_likely = false;
    std::optional<std::uint64_t> next_poll_deadline_ns;  // backoff hint
};

// -------------------------------------------------------------------------
// DataPath -- host-side data-plane SPI.
//
// A concrete DataPath opens resolver-produced targets, registers data-path
// memory, lowers requests, and drives submit/progress/query/release. It owns
// all private state (queues, workspaces, etc.); none of that is visible here.
//
// Lifecycle: initialize() once -> open/register/submit/progress/query/release
// -> shutdown(). A DataPath must finish all foreseeable validation, capacity
// reservation, and leasing before the first irreversible submit of a batch.
//
// op lifecycle: query() never destroys an op. release() is only valid on a
// terminal op (returns BUSY otherwise). Once terminal, an op no longer
// touches caller memory.
// -------------------------------------------------------------------------
class DataPath {
public:
    virtual ~DataPath() = default;

    // identity / capabilities
    virtual const DataPathCapabilities& capabilities() const = 0;

    // lifecycle
    virtual Status initialize(const DataPathConfig& config,
                              ResourceProvider& resources) = 0;
    virtual Status shutdown(std::uint64_t timeout_ns) = 0;

    // target lifecycle
    virtual Result<DataPathTarget> open(const ResolvedTarget& target) = 0;
    virtual Status close(DataPathTarget target) = 0;
    virtual Result<RegistrationDomainKey> registration_domain(
        DataPathTarget target) const = 0;

    // memory registration
    virtual Result<DataPathMemory> register_memory(
        const DataPathMemoryView& view,
        const RegistrationDomainKey& domain) = 0;
    virtual Status unregister_memory(DataPathMemory memory) = 0;

    // Optional block-table-aware registration.  The default is fail-closed:
    // a backend must override both paged hooks and advertise
    // supports_paged_memory before callers can select this path.
    virtual Result<DataPathPagedMemory> register_paged_memory(
        const DataPathPagedMemoryView&, const RegistrationDomainKey&) {
        return Result<DataPathPagedMemory>::Failure(Status(
            StatusCode::UNSUPPORTED,
            "DataPath does not support paged memory registration"));
    }
    virtual Status unregister_paged_memory(DataPathPagedMemory) {
        return Status(StatusCode::UNSUPPORTED,
                      "DataPath does not support paged memory registration");
    }

    // submit / progress / query / release
    //
    // submit() contract: the `requests` array MAY span multiple distinct
    // `target` identities within this same DataPath (each DataPathRequest
    // carries its own `target`, resolved independently by the
    // implementation). Callers (the runtime) group requests by DataPath
    // only, NOT by (DataPath, target); a DataPath implementation must not
    // assume all requests in one submit() call share a single target.
    virtual SubmitOutcome submit(const DataPathRequest* requests,
                                 std::size_t count,
                                 const HostSubmitContext& ctx) = 0;

    // Optional block-table-aware submit.  Implementations must copy each
    // block_ids vector before returning when the operation is asynchronous.
    // The default rejects every request and never mints an operation.
    virtual SubmitOutcome submit_paged(const DataPathPagedRequest*,
                                        std::size_t count,
                                        const HostSubmitContext&) {
        SubmitOutcome out;
        out.status = Status(StatusCode::UNSUPPORTED,
                            "DataPath does not support paged IO");
        out.initial_states.resize(count);
        for (auto& state : out.initial_states) {
            state.state = RequestState::REJECTED;
            state.status = out.status;
        }
        return out;
    }
    virtual Result<ProgressResult> progress(ProgressBudget budget) = 0;
    virtual Result<DataPathSnapshot> query(DataPathOp op) const = 0;
    virtual Status release(DataPathOp op) = 0;

    virtual Result<FeederLayerState> wait_feeder_layer(
        DataPathOp, std::uint32_t, std::uint64_t) {
        return Result<FeederLayerState>::Failure(Status(
            StatusCode::UNSUPPORTED, "DataPath has no step feeder"));
    }
    virtual Status signal_feeder_layer(
        DataPathOp, std::uint32_t, cudaStream_t) {
        return Status(StatusCode::UNSUPPORTED,
                      "DataPath has no step feeder");
    }
};

} // namespace tutti

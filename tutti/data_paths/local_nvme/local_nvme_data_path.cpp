// tutti/data_paths/local_nvme/local_nvme_data_path.cpp
//
// LocalNvmeDataPath implementation.

#include "tutti/data_paths/local_nvme/local_nvme_data_path.h"
#include "tutti/data_paths/local_nvme/io/nvme_queue_group.h"
#include "tutti/data_paths/local_nvme/io/device_target.h"
#include "tutti/data_paths/local_nvme/io/submit_one.cuh"
#include "tutti/data_paths/local_nvme/io/prp_builder.h"

#include <tutti/cuda_like.h>
#include <tutti/accelerator_device_guard.h>
#include <nvm_types.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>

namespace tutti::data_paths::local_nvme {

namespace {

bool feeder_diag_enabled() {
    const char* value = std::getenv("TUTTI_FEEDER_DIAGNOSTICS");
    return value && std::strcmp(value, "1") == 0;
}

long long feeder_diag_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct NvtxIoStyle {
    const char* label;
    std::uint32_t argb;
};

NvtxIoStyle nvtx_io_style(const DataPathRequest* requests, std::size_t count) {
    bool has_read = false;
    bool has_write = false;
    for (std::size_t i = 0; i < count; ++i) {
        has_read |= requests[i].intent.direction == IoDirection::READ;
        has_write |= requests[i].intent.direction == IoDirection::WRITE;
    }
    if (has_read && !has_write) {
        return {"tutti.local_nvme.io_kernel|op=read", 0xFF00B8D9u};
    }
    if (has_write && !has_read) {
        return {"tutti.local_nvme.io_kernel|op=write", 0xFFFF8C00u};
    }
    return {"tutti.local_nvme.io_kernel|op=mixed", 0xFF8D99A6u};
}

void nvtx_push_io(const NvtxIoStyle& style) {
    nvtxEventAttributes_t attrs{};
    attrs.version = NVTX_VERSION;
    attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attrs.colorType = NVTX_COLOR_ARGB;
    attrs.color = style.argb;
    attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attrs.message.ascii = style.label;
    nvtxRangePushEx(&attrs);
}

IoFailureKind completion_failure_kind(std::uint32_t result) {
    switch (result) {
        case 1: return IoFailureKind::RESOLVE_LBA;
        case 2: return IoFailureKind::CQ_TIMEOUT;
        case 3: return IoFailureKind::NVME_CQ_ERROR;
        default: return IoFailureKind::UNKNOWN;
    }
}

void set_completion_failure(IoCompletionDetail& detail,
                            IoFailureKind kind,
                            std::uint64_t confirmed_bytes,
                            bool timeout_seen = false,
                            std::uint32_t first_failed_entry = UINT32_MAX,
                            std::uint32_t raw_cq_status = 0) {
    detail.confirmed_bytes = confirmed_bytes;
    detail.timeout_seen = timeout_seen;
    detail.first_failed_entry = first_failed_entry;
    detail.failure_kind = kind;
    detail.raw_cq_status = raw_cq_status;
    detail.failure_scope = IoFailureScope::WHOLE_OPERATION;
    detail.failed_request_indices.clear();
}

} // namespace

// -------------------------------------------------------------------------
// Construction / destruction
// -------------------------------------------------------------------------

LocalNvmeDataPath::LocalNvmeDataPath(
    std::string snvme_dev_path)
    : snvme_dev_path_(std::move(snvme_dev_path)) {
    caps_.name = "local_nvme";
    caps_.source_api_version = 1;
    caps_.supports_host_execution = false;
    caps_.supports_device_execution = true;
    // HOST memory registration works but HOST IO is not implemented.
    caps_.supports_host_memory = false;
    caps_.supports_device_memory = true;
    caps_.supports_direct = true;
    caps_.supports_staged = false;
    caps_.supports_read = true;
    caps_.supports_write = true;
    caps_.target_alignment_bytes = 4096;
    caps_.memory_alignment_bytes = 4096;
    caps_.length_alignment_bytes = 4096;
    caps_.max_single_io_bytes = 4096;
    caps_.max_batch_requests = 1;
    caps_.max_batch_bytes = 4096;
    caps_.max_in_flight_operations = 1;
    caps_.supports_scatter_gather = false;
    caps_.max_scatter_gather_entries = 0;
    caps_.registration_scope = RegistrationScope::PER_TARGET;
    caps_.progress_model = ProgressModel::HOST_POLL;
    caps_.device_completion_fence_on_caller_stream = true;
    caps_.device_execution_autonomous = true;
    caps_.supports_multi_stream = false;
    caps_.max_concurrent_streams = 0;
    caps_.max_concurrent_operations = 1;
    caps_.supports_multi_gpu = false;
    caps_.supports_cross_device = false;
    caps_.bound_accel_id = static_cast<std::int32_t>(cuda_device_);
    caps_.optional_target_features = {};
}

LocalNvmeDataPath::LocalNvmeDataPath(
    std::string snvme_dev_path, std::uint32_t cuda_device,
    std::uint32_t num_user_queues,
    std::uint32_t namespace_id,
    std::uint32_t block_size,
    std::uint64_t mdts_bytes,
    std::uint32_t max_batch_entries,
    std::uint32_t cq_poll_budget,
    std::uint32_t handle_cache_capacity,
    std::uint32_t prp_cache_capacity,
    std::uint64_t max_in_flight_operations,
    std::uint64_t max_batch_requests,
    std::uint64_t max_request_bytes_override,
    std::uint32_t handle_cache_l2_capacity,
    std::string controller_pci_addr,
    std::uint32_t threads_per_block)
    : snvme_dev_path_(std::move(snvme_dev_path)),
      cuda_device_(cuda_device), num_user_queues_(num_user_queues),
      namespace_id_(namespace_id),
      block_size_(block_size),
      controller_pci_addr_(std::move(controller_pci_addr)),
      mdts_bytes_(mdts_bytes),
      max_batch_entries_(max_batch_entries == 0 ? 256 : max_batch_entries),
      max_batch_requests_(max_batch_requests == 0
                           ? (max_batch_entries == 0 ? 256 : max_batch_entries)
                           : max_batch_requests),
      max_in_flight_operations_(max_in_flight_operations == 0
                                 ? 16 : max_in_flight_operations),
      max_request_bytes_override_(max_request_bytes_override),
      threads_per_block_(threads_per_block),
      cq_poll_budget_(cq_poll_budget == 0 ? 10000000 : cq_poll_budget),
      handle_cache_capacity_(handle_cache_capacity),
      handle_cache_l2_capacity_(handle_cache_l2_capacity),
      prp_cache_capacity_(prp_cache_capacity) {
    caps_.name = "local_nvme";
    caps_.source_api_version = 1;
    caps_.supports_host_execution = false;
    caps_.supports_device_execution = true;
    // HOST memory registration works but HOST IO is not implemented.
    caps_.supports_host_memory = false;
    caps_.supports_device_memory = true;
    caps_.supports_direct = true;
    caps_.supports_staged = false;
    caps_.supports_read = true;
    caps_.supports_write = true;
    caps_.target_alignment_bytes = 4096;
    caps_.memory_alignment_bytes = 4096;
    caps_.length_alignment_bytes = 4096;
    // Preliminary caps; initialize() updates effective_mdts_bytes_ and caps_.
    std::uint64_t prelim_mdts = (mdts_bytes == 0) ? 131072 : mdts_bytes;
    std::uint64_t prelim_request_bytes = (max_request_bytes_override != 0)
        ? max_request_bytes_override
        : static_cast<std::uint64_t>(max_batch_entries_) * prelim_mdts;
    caps_.max_single_io_bytes = prelim_request_bytes;
    caps_.max_batch_requests = max_batch_requests_;
    caps_.max_batch_bytes = prelim_request_bytes;
    caps_.max_in_flight_operations = max_in_flight_operations_;
    caps_.supports_scatter_gather = false;
    caps_.max_scatter_gather_entries = 0;
    caps_.registration_scope = RegistrationScope::PER_TARGET;
    caps_.progress_model = ProgressModel::HOST_POLL;
    caps_.device_completion_fence_on_caller_stream = true;
    caps_.device_execution_autonomous = true;
    // Multi-stream: enabled after S5 dual-stream data-isolation validation
    // (test 50) proved two concurrent ops on two streams keep per-op
    // entry/PRP workspaces distinct and read back correct distinct patterns.
    caps_.supports_multi_stream = true;
    caps_.max_concurrent_streams = 2;
    caps_.max_concurrent_operations = max_in_flight_operations_;
    caps_.supports_multi_gpu = false;
    caps_.supports_cross_device = false;
    caps_.bound_accel_id = static_cast<std::int32_t>(cuda_device_);
    caps_.optional_target_features = {};
}

LocalNvmeDataPath::~LocalNvmeDataPath() {
    if (!initialized_) return;
    DeviceGuard device_guard(static_cast<std::int32_t>(cuda_device_));
    if (!device_guard.ok()) return;

    // Check for in-flight ops.  If any exist, wait for their completion
    // fence before tearing down.  This may block, but it prevents UAF.
    bool has_inflight = false;
    for (const auto& [tok, op] : ops_) {
        if (op.state == OpState::IN_FLIGHT) {
            has_inflight = true;
            break;
        }
    }

    if (has_inflight) {
        for (auto& [tok, op] : ops_) {
            if (!op.feeder_event_recorded) continue;
            for (std::uint32_t layer = 0; layer < op.feeder_layer_count; ++layer) {
                op.h_feeder_ready[layer] = 1u;
                op.h_feeder_release[layer] = 1u;
            }
        }
        // TEST-ONLY SYNC POINT (drain_inflight_ops_):
        // This cudaStreamSynchronize is in the shutdown drain path, not
        // the production submit/progress path.  It ensures all in-flight
        // kernels have completed before freeing arena resources during
        // shutdown.  Production submit() and progress() never call
        // cudaStreamSynchronize (except the documented event-record
        // failure fallback).
        // Collect unique streams from in-flight ops and synchronize each.
        std::set<cudaStream_t> inflight_streams;
        for (const auto& [tok, op] : ops_) {
            if (op.state == OpState::IN_FLIGHT && op.stream) {
                inflight_streams.insert(
                    static_cast<cudaStream_t>(op.stream));
            }
        }
        bool sync_failed = false;
        for (cudaStream_t s : inflight_streams) {
            cudaError_t ce = cudaStreamSynchronize(s);
            if (ce != cudaSuccess) {
                sync_failed = true;
                // Clear the error so subsequent CUDA calls don't see it.
                cudaGetLastError();
            }
        }
        if (sync_failed) {
            // Conservative: do NOT free resources that may still be in use
            // by the device.  This leaks, but it is safer than UAF.
            return;
        }
        // Mark all as terminal (sync succeeded → IO completed).
        // D2H per-entry status to detect device-side failures.
        for (auto& [tok, op] : ops_) {
            if (op.state == OpState::IN_FLIGHT) {
                aggregate_completion_status_(op);
            }
        }
    }

    // GPU metadata is safe after stream synchronization. Timed-out commands
    // retain host PRP backing through controller teardown.
    bool any_timeout = timeout_prp_retained_;
    for (const auto& [tok, op] : ops_) {
        if (op.has_timeout) { any_timeout = true; break; }
    }
    arena_.shutdown();
    ops_.clear();

    // Cache shutdown: frees all cached handles + PRP pool.
    // Must happen BEFORE target cleanup loop (cache owns the handle memory).
    handle_cache_.shutdown();
    prp_cache_.shutdown(any_timeout);
    prp_buf_pool_.shutdown(any_timeout);

    for (auto& [tok, state] : targets_) {
        // Only free handles NOT owned by the cache.
        if (state.cache_entry == nullptr &&
            (state.dev_handle != nullptr || state.dev_overflow != nullptr)) {
            free_device_target(state.dev_handle, state.dev_overflow,
                               cuda_device_);
        }
        state.dev_handle = nullptr;
        state.dev_overflow = nullptr;
    }
    targets_.clear();

    queue_group_.reset();

    for (auto& [tok, reg] : mem_regs_) {
        if (!reg.unregistered && reg.dma) {
            nvm_dma_unmap(reg.dma);
            reg.dma = nullptr;
        }
    }
    mem_regs_.clear();
    if (ctrl_) {
        nvm_ctrl_free_client(ctrl_);
        ctrl_ = nullptr;
    }
    initialized_ = false;
    timeout_prp_retained_ = false;
}

// -------------------------------------------------------------------------
// capabilities
// -------------------------------------------------------------------------

const DataPathCapabilities& LocalNvmeDataPath::capabilities() const {
    return caps_;
}

// -------------------------------------------------------------------------
// lifecycle
// -------------------------------------------------------------------------

Status LocalNvmeDataPath::initialize(const DataPathConfig& config,
                                     ResourceProvider& resources) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    Status result = initialize_impl_(config, resources);
    Status restored = guard.restore();
    return restored.ok() ? result : restored;
}

Status LocalNvmeDataPath::shutdown(std::uint64_t timeout_ns) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    Status result = shutdown_impl_(timeout_ns);
    Status restored = guard.restore();
    return restored.ok() ? result : restored;
}

Result<DataPathTarget> LocalNvmeDataPath::open(const ResolvedTarget& target) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return Result<DataPathTarget>::Failure(guard.status());
    auto result = open_impl_(target);
    Status restored = guard.restore();
    return restored.ok() ? result
                         : Result<DataPathTarget>::Failure(std::move(restored));
}

Status LocalNvmeDataPath::close(DataPathTarget target) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    Status result = close_impl_(target);
    Status restored = guard.restore();
    return restored.ok() ? result : restored;
}

Result<RegistrationDomainKey> LocalNvmeDataPath::registration_domain(
    DataPathTarget target) const {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) {
        return Result<RegistrationDomainKey>::Failure(guard.status());
    }
    auto result = registration_domain_impl_(target);
    Status restored = guard.restore();
    return restored.ok() ? result
                         : Result<RegistrationDomainKey>::Failure(std::move(restored));
}

Result<DataPathMemory> LocalNvmeDataPath::register_memory(
    const DataPathMemoryView& view,
    const RegistrationDomainKey& domain) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return Result<DataPathMemory>::Failure(guard.status());
    auto result = register_memory_impl_(view, domain);
    Status restored = guard.restore();
    return restored.ok() ? result
                         : Result<DataPathMemory>::Failure(std::move(restored));
}

Status LocalNvmeDataPath::unregister_memory(DataPathMemory memory) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    Status result = unregister_memory_impl_(memory);
    Status restored = guard.restore();
    return restored.ok() ? result : restored;
}

SubmitOutcome LocalNvmeDataPath::submit(const DataPathRequest* requests,
                                        std::size_t count,
                                        const HostSubmitContext& ctx) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) {
        SubmitOutcome result;
        result.status = guard.status();
        result.initial_states.resize(count);
        for (auto& state : result.initial_states) {
            state.state = RequestState::REJECTED;
            state.status = result.status;
        }
        return result;
    }
    SubmitOutcome result = submit_impl_(requests, count, ctx);
    Status restored = guard.restore();
    if (!restored.ok()) {
        result.status = restored;
        return result;
    }
    return result;
}

Result<ProgressResult> LocalNvmeDataPath::progress(ProgressBudget budget) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return Result<ProgressResult>::Failure(guard.status());
    auto result = progress_impl_(budget);
    Status restored = guard.restore();
    return restored.ok() ? result
                         : Result<ProgressResult>::Failure(std::move(restored));
}

Result<DataPathSnapshot> LocalNvmeDataPath::query(DataPathOp op) const {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return Result<DataPathSnapshot>::Failure(guard.status());
    auto result = query_impl_(op);
    Status restored = guard.restore();
    return restored.ok() ? result
                         : Result<DataPathSnapshot>::Failure(std::move(restored));
}

Status LocalNvmeDataPath::release(DataPathOp op) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    Status result = release_impl_(op);
    Status restored = guard.restore();
    return restored.ok() ? result : restored;
}

Result<FeederLayerState> LocalNvmeDataPath::wait_feeder_layer(
    DataPathOp op, std::uint32_t layer, std::uint64_t timeout_ms) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return Result<FeederLayerState>::Failure(guard.status());
    OpEntry* entry = find_op_(op);
    if (!entry || entry->feeder_layer_count == 0 ||
        layer >= entry->feeder_layer_count) {
        return Result<FeederLayerState>::Failure(Status(
            StatusCode::NOT_FOUND, "step feeder layer not found"));
    }
    if (!entry->feeder_is_read && layer < entry->feeder_staging_depth)
        return FeederLayerState::READY;
    volatile std::uint32_t* host_flags = entry->feeder_is_read
        ? entry->h_feeder_ready : entry->h_feeder_release;
    const std::uint32_t index = entry->feeder_is_read
        ? layer : layer - entry->feeder_staging_depth;
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    std::uint32_t value = host_flags[index];
    if (feeder_diag_enabled())
        std::fprintf(stderr, "FEEDER_DIAG cpp wait_begin t_ns=%lld backend=local direction=%s layer=%u gate_index=%u value=%u\n",
                     feeder_diag_ns(), entry->feeder_is_read ? "read" : "write",
                     layer, index, value);
    while (value == 0) {
        if (timeout_ms == 0 || std::chrono::steady_clock::now() >= deadline)
            return FeederLayerState::PENDING;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        value = host_flags[index];
    }
    if (!entry->feeder_is_read) return FeederLayerState::READY;
    if (feeder_diag_enabled())
        std::fprintf(stderr, "FEEDER_DIAG cpp wait_consume t_ns=%lld backend=local direction=read layer=%u gate_index=%u value=%u\n",
                     feeder_diag_ns(), layer, index, value);
    return value == 1 ? FeederLayerState::READY : FeederLayerState::FAILED;
}

Status LocalNvmeDataPath::signal_feeder_layer(
    DataPathOp op, std::uint32_t layer, cudaStream_t stream) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    OpEntry* entry = find_op_(op);
    if (!entry || entry->feeder_layer_count == 0 ||
        layer >= entry->feeder_layer_count) {
        return Status(StatusCode::NOT_FOUND, "step feeder layer not found");
    }
    std::uint32_t* flag = entry->feeder_is_read
        ? entry->d_feeder_release + layer
        : entry->d_feeder_ready + layer;
    if (feeder_diag_enabled())
        std::fprintf(stderr, "FEEDER_DIAG cpp signal_enqueue t_ns=%lld backend=local direction=%s layer=%u\n",
                     feeder_diag_ns(), entry->feeder_is_read ? "read" : "write",
                     layer);
    cudaError_t ce = launch_feeder_signal(flag, stream);
    return ce == cudaSuccess
        ? Status::Ok()
        : Status(StatusCode::DEVICE_ERROR,
                 std::string("signal feeder layer failed: ") +
                 cudaGetErrorString(ce));
}

Status LocalNvmeDataPath::initialize_impl_(const DataPathConfig& config,
                                           ResourceProvider& /*resources*/) {
    if (initialized_) {
        return Status(StatusCode::BUSY, "already initialized");
    }
    if (threads_per_block_ == 0 || threads_per_block_ > 1024) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "threads_per_block must be in [1, 1024]");
    }
    cudaDeviceProp device_properties{};
    const cudaError_t properties_error = cudaGetDeviceProperties(
        &device_properties, static_cast<int>(cuda_device_));
    if (properties_error != cudaSuccess) {
        return Status(StatusCode::DEVICE_ERROR,
                      std::string("cudaGetDeviceProperties failed: ") +
                          cudaGetErrorString(properties_error));
    }
    if (threads_per_block_ >
        static_cast<std::uint32_t>(device_properties.maxThreadsPerBlock)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "threads_per_block exceeds device maximum");
    }

    if (!config.name.empty()) {
        caps_.name = config.name;
    }

    if (!snvme_dev_path_.empty()) {
        int rc = nvm_ctrl_attach_client(&ctrl_, snvme_dev_path_.c_str());
        if (rc != 0 || ctrl_ == nullptr) {
            return Status(StatusCode::NOT_READY,
                          "nvm_ctrl_attach_client(" + snvme_dev_path_ +
                          ") failed: rc " + std::to_string(rc));
        }

        struct disk dev_info;
        std::memset(&dev_info, 0, sizeof(dev_info));
        rc = ioctl_get_dev_info(ctrl_, &dev_info);
        if (rc != 0) {
            nvm_ctrl_free_client(ctrl_);
            ctrl_ = nullptr;
            return Status(StatusCode::NOT_READY,
                          "ioctl_get_dev_info failed: rc " +
                          std::to_string(rc));
        }

        // --- Hardware MDTS ---
        // Only compute MDTS / PRP capacity in production mode (block_size_ > 0).
        // Skeleton mode (block_size_ == 0) skips this — no IO will be submitted.
        hardware_mdts_bytes_ = dev_info.max_data_size;
        if (block_size_ > 0 && hardware_mdts_bytes_ > 0) {
            if (mdts_bytes_ == 0) {
                effective_mdts_bytes_ = hardware_mdts_bytes_;
            } else {
                effective_mdts_bytes_ = std::min(mdts_bytes_, hardware_mdts_bytes_);
            }

            if (effective_mdts_bytes_ == 0) {
                nvm_ctrl_free_client(ctrl_);
                ctrl_ = nullptr;
                return Status(StatusCode::NOT_READY,
                              "effective MDTS is 0");
            }
            if (effective_mdts_bytes_ % block_size_ != 0) {
                nvm_ctrl_free_client(ctrl_);
                ctrl_ = nullptr;
                return Status(StatusCode::INVALID_ARGUMENT,
                              "effective MDTS not a block-size multiple");
            }

            // PRP-list page capacity: max data pages expressable in one PRP-list page.
            const std::uint64_t page_size = static_cast<std::uint64_t>(ctrl_->page_size);
            prp_list_page_capacity_ = page_size / sizeof(std::uint64_t) + 1;

            // R19 S3b: initialize the host-pinned PRP buffer pool.
            prp_buf_pool_.init(ctrl_, page_size);

            std::uint64_t mdts_pages = effective_mdts_bytes_ / page_size;
            if (mdts_pages > prp_list_page_capacity_) {
                nvm_ctrl_free_client(ctrl_);
                ctrl_ = nullptr;
                return Status(StatusCode::INVALID_ARGUMENT,
                              "effective MDTS exceeds PRP-list page capacity");
            }

            // Update capabilities with real values.
            max_request_bytes_ = (max_request_bytes_override_ != 0)
                ? max_request_bytes_override_
                : static_cast<std::uint64_t>(max_batch_entries_) *
                  effective_mdts_bytes_;
            caps_.max_single_io_bytes = max_request_bytes_;
            caps_.max_batch_bytes = max_request_bytes_;
            caps_.max_batch_requests = max_batch_requests_;
            caps_.max_in_flight_operations = max_in_flight_operations_;
            caps_.max_concurrent_operations = max_in_flight_operations_;
            caps_.target_alignment_bytes = block_size_;
            caps_.memory_alignment_bytes = block_size_;
            caps_.length_alignment_bytes = block_size_;
        }

        if (num_user_queues_ > 0) {
            if (block_size_ == 0) {
                nvm_ctrl_free_client(ctrl_);
                ctrl_ = nullptr;
                return Status(StatusCode::INVALID_ARGUMENT,
                              "queue group requested but block_size is 0");
            }

            // SQ/CQ ring depth is fixed by the kernel module
            // (io_queue_depth) and obtained via NVM_GET_DEV_INFO — never
            // user-specified: the kernel builds user IOQ rings with
            // dev->q_depth unconditionally, so any other userspace ring
            // size silently desyncs SQ wrap / CQ phase (observed as CQ
            // poll timeouts once a queue wraps past the smaller depth).
            queue_depth_ = ctrl_->q_depth;

            struct disk disk_info = dev_info;
            disk_info.ns_id = namespace_id_;
            if (block_size_ > 0) {
                disk_info.block_size = block_size_;
            }
            std::string dname = snvme_dev_path_;
            if (dname.rfind("/dev/", 0) == 0) dname = dname.substr(5);
            dname += "n" + std::to_string(namespace_id_);
            std::strncpy(disk_info.disk_name, dname.c_str(),
                         sizeof(disk_info.disk_name) - 1);

            try {
                queue_group_ = std::make_unique<NvmeQueueGroup>(
                    ctrl_, disk_info, namespace_id_, cuda_device_,
                    num_user_queues_, queue_depth_);
            } catch (const std::runtime_error& e) {
                nvm_ctrl_free_client(ctrl_);
                ctrl_ = nullptr;
                return Status(StatusCode::NOT_READY,
                              std::string("queue group creation failed: ") +
                              e.what());
            }
            if (threads_per_block_ > queue_group_->n_qps()) {
                // Round-robin sharing (see striped path): the parallel
                // queue supports multiple concurrent submitters per
                // queue, so warn instead of failing.
                std::fprintf(
                    stderr,
                    "[local-nvme] warning: threads_per_block (%u) > "
                    "granted queues (%u); threads will share queues "
                    "round-robin\n",
                    threads_per_block_, queue_group_->n_qps());
            }

            // Initialize the MetadataArena: pre-allocate all per-op
            // workspace (events, entry/status arrays, PRP-list pool).
            // Capacity = max_in_flight_operations; each slot can hold
            // up to max_batch_entries entries and max_batch_entries PRP pages.
            MetadataArena::Config arena_cfg;
            // Arena capacity = 2 * max_in_flight_operations: in-flight slots
            // + terminal-but-unreleased slots.  Terminal ops retain their
            // workspace until release() (for test_copy_entry accessors);
            // the extra capacity lets a new submit acquire a slot even when
            // all in-flight ops have gone terminal but not yet released.
            arena_cfg.num_slots = static_cast<std::uint32_t>(max_in_flight_operations_) * 2;
            arena_cfg.max_entries_per_slot = max_batch_entries_;
            arena_cfg.page_size = static_cast<std::uint32_t>(ctrl_->page_size);
            arena_cfg.cuda_device = cuda_device_;
            if (!arena_.init(arena_cfg, ctrl_)) {
                nvm_ctrl_free_client(ctrl_);
                ctrl_ = nullptr;
                return Status(StatusCode::NOT_READY,
                              "MetadataArena init failed");
            }

            if (handle_cache_capacity_ > 0) {
                HandleWorkspaceCache::Config hcfg;
                hcfg.capacity = handle_cache_capacity_;
                // Round 16 S6b: L2 defaults to 4×L1 when not specified.
                hcfg.l2_capacity = (handle_cache_l2_capacity_ > 0)
                                   ? handle_cache_l2_capacity_
                                   : handle_cache_capacity_ * 4;
                hcfg.cuda_device = cuda_device_;
                handle_cache_.set_free_fn(&free_device_target);
                handle_cache_.set_snapshot_fn(&snapshot_device_target);
                handle_cache_.set_restore_fn(&restore_device_target);
                if (!handle_cache_.init(hcfg)) {
                    nvm_ctrl_free_client(ctrl_);
                    ctrl_ = nullptr;
                    return Status(StatusCode::NOT_READY,
                                  "HandleWorkspaceCache init failed");
                }
            }

            if (prp_cache_capacity_ > 0) {
                PrpPageCache::Config pcfg;
                pcfg.capacity = prp_cache_capacity_;
                pcfg.page_size = static_cast<std::uint32_t>(ctrl_->page_size);
                pcfg.cuda_device = cuda_device_;
                if (!prp_cache_.init(pcfg, ctrl_)) {
                    nvm_ctrl_free_client(ctrl_);
                    ctrl_ = nullptr;
                    return Status(StatusCode::NOT_READY,
                                  "PrpPageCache init failed");
                }
            }
        }
    }

    initialized_ = true;
    return Status::Ok();
}

Status LocalNvmeDataPath::shutdown_impl_(std::uint64_t timeout_ns) {
    if (!initialized_) return Status::Ok();  // idempotent

    // Check for in-flight ops.
    auto has_inflight = [&]() -> bool {
        for (const auto& [tok, op] : ops_) {
            if (op.state == OpState::IN_FLIGHT) return true;
        }
        return false;
    };

    if (has_inflight()) {
        if (timeout_ns == 0) {
            // Zero timeout with in-flight ops → TIMEOUT, retain all resources.
            return Status(StatusCode::TIMEOUT,
                          "shutdown: in-flight operations remain");
        }

        for (auto& [tok, op] : ops_) {
            if (!op.feeder_event_recorded) continue;
            for (std::uint32_t layer = 0; layer < op.feeder_layer_count; ++layer) {
                op.h_feeder_ready[layer] = 1u;
                op.h_feeder_release[layer] = 1u;
            }
        }

        // Drain within deadline.
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::nanoseconds(timeout_ns);
        while (std::chrono::steady_clock::now() < deadline) {
            auto remaining_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                deadline - std::chrono::steady_clock::now()).count();
            if (remaining_ns <= 0) break;

            ProgressBudget pb{
                max_in_flight_operations_,
                static_cast<std::uint64_t>(remaining_ns)};
            auto pr = progress(pb);
            if (!pr.ok()) break;
            if (!has_inflight()) break;
            // Avoid busy-poll: sleep a short time.
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }

        if (has_inflight()) {
            // Still in-flight after deadline → TIMEOUT.
            // DataPath stays initialized; all resources retained.
            return Status(StatusCode::TIMEOUT,
                          "shutdown: drain timeout, in-flight ops remain");
        }
    }

    // All ops terminal. GPU metadata is controller-independent once the
    // submit kernel returned. A timed-out command may still fetch host PRP
    // pages, so host cache/pool backing is conservatively retained.
    bool any_timeout = false;
    for (const auto& [tok, op] : ops_) {
        if (op.has_timeout) { any_timeout = true; break; }
    }
    arena_.shutdown();
    ops_.clear();

    // Cache shutdown: frees all cached handles + PRP pool.
    // Must happen BEFORE target cleanup (cache owns the handle memory).
    handle_cache_.shutdown();
    prp_cache_.shutdown(any_timeout);
    prp_buf_pool_.shutdown(any_timeout);

    for (auto& [tok, state] : targets_) {
        // Only free handles NOT owned by the cache.
        if (state.cache_entry == nullptr &&
            (state.dev_handle != nullptr || state.dev_overflow != nullptr)) {
            free_device_target(state.dev_handle, state.dev_overflow,
                               cuda_device_);
        }
        state.dev_handle = nullptr;
        state.dev_overflow = nullptr;
    }
    targets_.clear();

    queue_group_.reset();

    for (auto& [tok, reg] : mem_regs_) {
        if (!reg.unregistered && reg.dma) {
            nvm_dma_unmap(reg.dma);
            reg.dma = nullptr;
            reg.unregistered = true;
        }
    }
    mem_regs_.clear();

    if (ctrl_) {
        nvm_ctrl_free_client(ctrl_);
        ctrl_ = nullptr;
    }

    initialized_ = false;
    return Status::Ok();
}

// -------------------------------------------------------------------------
// target lifecycle
// -------------------------------------------------------------------------

Result<DataPathTarget> LocalNvmeDataPath::open_impl_(const ResolvedTarget& target) {
    if (!initialized_) {
        return Result<DataPathTarget>::Failure(
            Status(StatusCode::NOT_READY,
                   "DataPath not initialized"));
    }

    auto payload_result = binding::ext4_local_nvme::view_payload(target);
    if (!payload_result.ok()) {
        return Result<DataPathTarget>::Failure(
            Status(payload_result.status().code(),
                   "open: payload view failed: " +
                   payload_result.status().message()));
    }
    const auto* payload = payload_result.value();

    const auto& ns = payload->namespace_identity();
    if (ns.block_size == 0) {
        return Result<DataPathTarget>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "open: block_size is zero"));
    }
    if (ns.controller_pci_addr.empty()) {
        return Result<DataPathTarget>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "open: controller_pci_addr is empty"));
    }
    if (!controller_pci_addr_.empty() &&
        ns.controller_pci_addr != controller_pci_addr_) {
        return Result<DataPathTarget>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "open: controller_pci_addr does not match DataPath resource view"));
    }
    if (namespace_id_ != 0 && ns.namespace_id != namespace_id_) {
        return Result<DataPathTarget>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "open: namespace_id does not match DataPath resource view"));
    }
    if (block_size_ != 0 && ns.block_size != block_size_) {
        return Result<DataPathTarget>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "open: block_size does not match DataPath resource view"));
    }

    LocalNvmeTargetState state;
    state.ns = ns;
    state.file_size_bytes = payload->file_size();
    state.block_size_log = 0;
    {
        std::uint32_t bs = ns.block_size;
        while ((1u << state.block_size_log) < bs) {
            ++state.block_size_log;
        }
        if ((1u << state.block_size_log) != bs) {
            return Result<DataPathTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "open: block_size is not a power of two"));
        }
    }

    const auto& src_extents = payload->extents();
    state.lba_extents.reserve(src_extents.size());
    for (const auto& e : src_extents) {
        LbaExtent lba;
        lba.start_lba = e.device_offset / ns.block_size;
        lba.length_blocks = e.length / ns.block_size;
        lba.logical_offset_bytes = e.logical_offset;

        if (e.device_offset % ns.block_size != 0 ||
            e.length % ns.block_size != 0) {
            return Result<DataPathTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "open: extent not block-aligned"));
        }

        state.lba_extents.push_back(lba);
    }

    std::uint64_t token = next_token_++;
    std::uint64_t generation = 1;
    state.token = token;
    state.generation = generation;

    // Compute handle cache key (FNV-1a hash of file extent signature).
    if (handle_cache_.enabled()) {
        std::uint64_t h = 1469598103934665603ULL;
        auto fnv = [&](const void* data, std::size_t len) {
            const auto* p = static_cast<const std::uint8_t*>(data);
            for (std::size_t i = 0; i < len; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
        };
        fnv(ns.controller_pci_addr.data(), ns.controller_pci_addr.size());
        fnv(&ns.namespace_id, sizeof(ns.namespace_id));
        fnv(&state.file_size_bytes, sizeof(state.file_size_bytes));
        for (const auto& e : state.lba_extents) {
            fnv(&e.start_lba, sizeof(e.start_lba));
            fnv(&e.length_blocks, sizeof(e.length_blocks));
            fnv(&e.logical_offset_bytes, sizeof(e.logical_offset_bytes));
        }
        state.cache_key = h;
    }

    if (queue_group_ && queue_group_->d_qps() != nullptr) {
        DeviceTargetHandle tmpl;
        std::memset(&tmpl, 0, sizeof(tmpl));
        tmpl.file_id = token;
        tmpl.logical_size_bytes = state.file_size_bytes;
        tmpl.header_bytes = 0;
        tmpl.nvme_block_size = ns.block_size;
        tmpl.nvme_block_size_log = state.block_size_log;
        tmpl.namespace_id = ns.namespace_id;
        tmpl.num_extents = static_cast<uint32_t>(state.lba_extents.size());
        tmpl.d_qps = queue_group_->d_qps();
        tmpl.num_d_qps = queue_group_->n_qps();
        tmpl.extents_overflow = nullptr;

        uint32_t n_inline = std::min<uint32_t>(
            tmpl.num_extents, kDeviceTargetInlineExtents);
        for (uint32_t i = 0; i < n_inline; ++i) {
            tmpl.extents[i].start_lba = state.lba_extents[i].start_lba;
            tmpl.extents[i].length_blocks = state.lba_extents[i].length_blocks;
        }

        uint32_t n_overflow = (tmpl.num_extents > kDeviceTargetInlineExtents)
            ? tmpl.num_extents - kDeviceTargetInlineExtents
            : 0;
        const uint64_t overflow_bytes = (uint64_t)n_overflow * sizeof(DeviceLbaExtent);

        std::vector<DeviceLbaExtent> overflow_buf;
        const DeviceLbaExtent* overflow_ptr = nullptr;
        if (n_overflow > 0) {
            overflow_buf.resize(n_overflow);
            for (uint32_t i = 0; i < n_overflow; ++i) {
                overflow_buf[i].start_lba =
                    state.lba_extents[kDeviceTargetInlineExtents + i].start_lba;
                overflow_buf[i].length_blocks =
                    state.lba_extents[kDeviceTargetInlineExtents + i].length_blocks;
            }
            overflow_ptr = overflow_buf.data();
        }

        // Handle cache path: get_or_build by file extent signature.
        if (handle_cache_.enabled()) {
            auto* ce = handle_cache_.get_or_build(state.cache_key,
                [&](DeviceTargetHandle** out_h, void** out_ov,
                    std::uint64_t* out_ov_bytes) -> bool {
                    bool ok = build_device_target(tmpl, overflow_ptr, n_overflow,
                                                  cuda_device_, out_h, out_ov);
                    if (ok) *out_ov_bytes = overflow_bytes;
                    return ok;
                });
            if (ce == nullptr) {
                return Result<DataPathTarget>::Failure(
                    Status(StatusCode::DEVICE_ERROR,
                           "open: handle cache get_or_build failed "
                           "(pool exhausted or build failure)"));
            }
            state.dev_handle = ce->handle;
            state.dev_overflow = ce->overflow;
            state.cache_entry = ce;
        } else {
            // Direct build (no cache): current behavior.
            DeviceTargetHandle* dev_handle = nullptr;
            void* dev_overflow = nullptr;
            if (!build_device_target(tmpl, overflow_ptr, n_overflow,
                                     cuda_device_, &dev_handle, &dev_overflow)) {
                return Result<DataPathTarget>::Failure(
                    Status(StatusCode::DEVICE_ERROR,
                           "open: failed to build device target handle"));
            }
            state.dev_handle = dev_handle;
            state.dev_overflow = dev_overflow;
        }
    }

    targets_[token] = std::move(state);

    return Result<DataPathTarget>::Success(
        detail::SpiIdentityMint::mint<detail::DataPathTargetTag>(
            token, generation));
}

Status LocalNvmeDataPath::close_impl_(DataPathTarget target) {
    if (!target.valid()) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "close: target identity is invalid (never minted)");
    }

    auto it = targets_.find(target.token());
    if (it == targets_.end()) {
        return Status(StatusCode::NOT_FOUND,
                      "close: target not found (never opened or already closed)");
    }
    if (it->second.generation != target.generation()) {
        return Status(StatusCode::NOT_FOUND,
                      "close: generation mismatch (stale identity)");
    }

    if (target_has_inflight_ops_(target.token())) {
        return Status(StatusCode::BUSY,
                      "close: target has in-flight operations");
    }

    if (it->second.cache_entry != nullptr) {
        // Cache-owned: don't free, just mark as evictable (released).
        handle_cache_.release_entry(it->second.cache_entry);
        it->second.dev_handle = nullptr;
        it->second.dev_overflow = nullptr;
        it->second.cache_entry = nullptr;
    } else if (it->second.dev_handle != nullptr || it->second.dev_overflow != nullptr) {
        free_device_target(it->second.dev_handle, it->second.dev_overflow,
                           cuda_device_);
        it->second.dev_handle = nullptr;
        it->second.dev_overflow = nullptr;
    }

    targets_.erase(it);
    return Status::Ok();
}

Result<RegistrationDomainKey> LocalNvmeDataPath::registration_domain_impl_(
    DataPathTarget target) const {
    const auto* state = find_(target);
    if (!state) {
        return Result<RegistrationDomainKey>::Failure(
            Status(StatusCode::NOT_FOUND,
                   "registration_domain: target not found or already closed"));
    }

    std::string key = "local_nvme:";
    key += state->ns.controller_pci_addr;
    key += ":ns";
    key += std::to_string(state->ns.namespace_id);

    return Result<RegistrationDomainKey>::Success(
        RegistrationDomainKey{std::move(key)});
}

// -------------------------------------------------------------------------
// memory registration
// -------------------------------------------------------------------------

Result<DataPathMemory> LocalNvmeDataPath::register_memory_impl_(
    const DataPathMemoryView& view,
    const RegistrationDomainKey& /*domain*/) {
    if (!initialized_ || ctrl_ == nullptr) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::NOT_READY,
                   "register_memory: DataPath not initialized "
                   "(controller not attached)"));
    }

    if (view.base == nullptr) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "register_memory: view.base is null"));
    }
    if (view.size_bytes == 0) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "register_memory: view.size_bytes is 0"));
    }

    if (view.expected_accel_id >= 0 &&
        view.expected_accel_id != static_cast<std::int32_t>(cuda_device_)) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "register_memory: accelerator does not match DataPath"));
    }
    if (view.kind == DataPathMemoryKind::DEVICE) {
#if defined(TUTTI_USE_HOST)
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::UNSUPPORTED,
                   "register_memory: HOST profile has no device memory"));
#elif defined(TUTTI_USE_CUDA) || defined(TUTTI_USE_MUSA) || defined(TUTTI_USE_MACA)
        cudaPointerAttributes attributes{};
        const cudaError_t pointer_error =
            cudaPointerGetAttributes(&attributes, view.base);
        if (pointer_error != cudaSuccess) {
            return Result<DataPathMemory>::Failure(
                Status(StatusCode::DEVICE_ERROR,
                       "register_memory: pointer ownership query failed: " +
                       std::string(cudaGetErrorString(pointer_error))));
        }
#if defined(TUTTI_USE_CUDA)
        if (attributes.type != cudaMemoryTypeDevice &&
            attributes.type != cudaMemoryTypeManaged) {
            return Result<DataPathMemory>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "register_memory: pointer is not device memory"));
        }
#else
        if (attributes.type != cudaMemoryTypeDevice) {
            return Result<DataPathMemory>::Failure(
                Status(StatusCode::UNSUPPORTED,
                       "register_memory: backend cannot verify pointer kind"));
        }
#endif
        if (attributes.device != static_cast<int>(cuda_device_)) {
            return Result<DataPathMemory>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "register_memory: pointer belongs to another accelerator"));
        }
#endif
    }

    // DEVICE memory must be 64 KiB-aligned.
    if (view.kind == DataPathMemoryKind::DEVICE &&
        (reinterpret_cast<std::uintptr_t>(view.base) % 65536) != 0) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "register_memory: DEVICE view.base must be 64 KiB-aligned "
                   "(snvme pins GPU pages at 64 KiB granularity; an "
                   "unaligned base makes every PRP point before the buffer)"));
    }

    nvm_dma_t* dma = nullptr;
    int rc;
    if (view.kind == DataPathMemoryKind::DEVICE) {
        rc = nvm_dma_map_data_device(&dma, ctrl_,
                                     view.base,
                                     static_cast<size_t>(view.size_bytes));
    } else {  // HOST
        rc = nvm_dma_map_data_host(&dma, ctrl_,
                                   view.base,
                                   static_cast<size_t>(view.size_bytes));
    }

    if (rc != 0 || dma == nullptr) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::DEVICE_ERROR,
                   "nvm_dma_map_data_" +
                   std::string(view.kind == DataPathMemoryKind::DEVICE
                               ? "device" : "host") +
                   " failed: rc " + std::to_string(rc)));
    }

    std::uint64_t token = next_mem_token_++;
    std::uint64_t generation = 1;

    MemReg reg;
    reg.dma = dma;
    reg.base = view.base;
    reg.size_bytes = view.size_bytes;
    reg.kind = view.kind;
    reg.accel_id = view.expected_accel_id;
    reg.generation = generation;
    reg.unregistered = false;

    // Round 16 S5 (V3): if io_granularity > 0, pre-build AddressDescriptor[]
    // at registration time (legacy build_io_slice_table 9-stage path).
    // This makes submit a pure pointer-arithmetic operation (zero PRP H2D).
    if (view.io_granularity > 0) {
        std::string err_msg;
        if (!build_prebuilt_descriptors_(reg, view.io_granularity, err_msg)) {
            nvm_dma_unmap(dma);
            return Result<DataPathMemory>::Failure(
                Status(StatusCode::DEVICE_ERROR, err_msg));
        }
    }

    mem_regs_[token] = reg;

    return Result<DataPathMemory>::Success(
        detail::SpiIdentityMint::mint<detail::DataPathMemoryTag>(
            token, generation));
}

Status LocalNvmeDataPath::unregister_memory_impl_(DataPathMemory memory) {
    if (!memory.valid()) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "unregister_memory: memory identity is invalid");
    }

    auto it = mem_regs_.find(memory.token());
    if (it == mem_regs_.end()) {
        return Status(StatusCode::NOT_FOUND,
                      "unregister_memory: memory not found "
                      "(never registered or already unregistered)");
    }
    if (it->second.generation != memory.generation()) {
        return Status(StatusCode::NOT_FOUND,
                      "unregister_memory: generation mismatch (stale identity)");
    }
    if (it->second.unregistered) {
        return Status(StatusCode::NOT_FOUND,
                      "unregister_memory: already unregistered");
    }

    if (memory_has_inflight_ops_(memory.token())) {
        return Status(StatusCode::BUSY,
                      "unregister_memory: memory has in-flight operations");
    }

    if (it->second.dma) {
        nvm_dma_unmap(it->second.dma);
        it->second.dma = nullptr;
    }
    destroy_prebuilt_descriptors_(it->second);
    it->second.unregistered = true;

    // Invalidate PRP cache entries for this memory token.
    prp_cache_.invalidate_memory(memory.token());

    return Status::Ok();
}

// -------------------------------------------------------------------------
// Round 16 S5 (V3): registration-time pre-built descriptors
// (legacy build_io_slice_table 9-stage path).
// -------------------------------------------------------------------------
bool LocalNvmeDataPath::build_prebuilt_descriptors_(
    MemReg& reg, std::uint64_t io_granularity, std::string& status_msg) {

    // Stage 1-2: compute slice plan
    const std::uint64_t page_size = static_cast<std::uint64_t>(ctrl_->page_size);
    const std::uint64_t mdts = effective_mdts_bytes_;
    const std::uint64_t bytes_per_slice = io_granularity;
    if (bytes_per_slice == 0 || bytes_per_slice % page_size != 0) {
        status_msg = "io_granularity must be a positive controller-page multiple";
        return false;
    }
    const std::uint64_t num_slices = reg.size_bytes / bytes_per_slice;
    if (reg.size_bytes % bytes_per_slice != 0) {
        status_msg = "io_granularity does not evenly divide memory size";
        return false;
    }
    const std::uint64_t ios_per_slice =
        (bytes_per_slice + mdts - 1) / mdts;
    const std::uint64_t total_descs = num_slices * ios_per_slice;
    std::uint64_t num_prp_pages = 0;
    for (std::uint64_t sub = 0; sub < ios_per_slice; ++sub) {
        const std::uint64_t offset = sub * mdts;
        const std::uint64_t sub_io =
            std::min(mdts, bytes_per_slice - offset);
        if (sub_io / page_size > 2) {
            num_prp_pages += num_slices;
        }
    }

    // Stage 3: validate alignment (already checked 64KiB in register_memory)
    // Skip — alignment is validated upstream.

    // Stage 4: allocate PRP-list pages from the host-pinned pool (R19 S3b).
    //
    // PRP2 for a LIST command must identify a page-aligned PRP list. Give
    // every LIST sub-IO one full controller page: this supports the complete
    // single-page PRP capacity (up to page_size / 8 + 1 data pages) and keeps
    // the list valid for 256 KiB and larger IO granularities. The backing is
    // sub-allocated from one growing DataPath-level host DMA pool.
    PrpBufRef prp_buf_ref;
    if (num_prp_pages > 0) {
        prp_buf_ref = prp_buf_pool_.alloc_pages(num_prp_pages);
        if (!prp_buf_ref.valid) {
            status_msg = "PrpBufPool::alloc_pages failed (nvm_dma_map_data_host"
                         " segment growth failed?)";
            return false;
        }
    }

    // Stage 5-6: fill address descriptors + page-aligned PRP-list pages.
    std::vector<AddressDescriptor> h_descs(total_descs);
    std::uint64_t prp_page_idx = 0;
    for (std::uint64_t s = 0; s < num_slices; ++s) {
        const std::uint64_t slice_offset = s * bytes_per_slice;
        for (std::uint64_t sub = 0; sub < ios_per_slice; ++sub) {
            const std::uint64_t sub_offset = sub * mdts;
            const std::uint64_t sub_io =
                std::min(mdts, bytes_per_slice - sub_offset);
            const std::uint64_t start_page =
                (slice_offset + sub_offset) / page_size;
            const std::uint64_t pages_per_io = sub_io / page_size;
            AddressDescriptor& d = h_descs[s * ios_per_slice + sub];
            d.data_length = sub_io;
            d.prp1 = reg.dma->ioaddrs[start_page];

            if (pages_per_io == 1) {
                d.prp2 = 0;
            } else if (pages_per_io == 2) {
                d.prp2 = reg.dma->ioaddrs[start_page + 1];
            } else {
                const std::uint64_t prp_page =
                    prp_buf_ref.base_page + prp_page_idx++;
                auto* host_page = reinterpret_cast<std::uint64_t*>(
                    static_cast<char*>(prp_buf_ref.segment->vaddr) +
                    prp_page * page_size);
                fill_prp_list_page(host_page, reg.dma,
                                   static_cast<std::uint32_t>(start_page),
                                   static_cast<std::uint32_t>(pages_per_io),
                                   page_size);
                d.prp2 = prp_buf_ref.segment->ioaddrs[prp_page];
            }
        }
    }

    // Stage 7: allocate descriptors from the GPU pool (R19 S3 REQUIRED 3).
    // Previously: per-registration cudaMalloc — at 1.47M registrations this
    // caused minute-level latency. Now: sub-allocate from a DataPath-level
    // bump pool (segments grow as needed, freed on DataPath shutdown).
    void* d_descs = desc_pool_.alloc(total_descs * sizeof(AddressDescriptor));
    if (!d_descs) {
        status_msg = "DescPool::alloc failed (cudaMalloc segment growth failed?)";
        // prp_buf_ref is pool-managed — no per-registration free needed.
        return false;
    }
    cudaError_t ce = cudaMemcpy(d_descs, h_descs.data(),
                    total_descs * sizeof(AddressDescriptor),
                    cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) {
        status_msg = std::string("cudaMemcpy d_descs failed: ") + cudaGetErrorString(ce);
        // d_descs and prp_buf_ref are pool-managed — no per-registration free.
        return false;
    }

    // Stage 8 (R19 S3 REQUIRED 2): d_prp_gpu removed.
    // Previously: cudaMalloc'd a GPU copy of PRP pages, cudaMemcpy'd,
    // then cudaFree'd — all wasted work. The NVMe controller reads PRP
    // lists from the host-pinned nvm_dma buffer's IOVA (via PCIe DMA),
    // not from GPU global memory. prp2 points to the IOVA, the controller
    // fetches entries directly. No GPU-side PRP page copy needed.

    // Stage 9: store in MemReg::prebuilt
    reg.prebuilt.d_descs = d_descs;
    reg.prebuilt.num_descs = total_descs;
    reg.prebuilt.bytes_per_slice = bytes_per_slice;
    reg.prebuilt.ios_per_slice = ios_per_slice;
    reg.prebuilt.prp_buf_ref = prp_buf_ref;  // pool-managed; freed on shutdown
    reg.prebuilt.num_prp_pages = num_prp_pages;
    reg.prebuilt.valid = true;
    return true;
}

void LocalNvmeDataPath::destroy_prebuilt_descriptors_(MemReg& reg) {
    // R19 S3 REQUIRED 3: d_descs is pool-managed — do NOT cudaFree.
    // R19 S3b REQUIRED 1: prp_buf_ref is pool-managed — do NOT nvm_dma_unmap.
    // Both pools reclaim memory on DataPath shutdown.
    // Setting to nullptr/false prevents accidental use after unregister.
    reg.prebuilt.d_descs = nullptr;
    reg.prebuilt.prp_buf_ref = {};
    reg.prebuilt.valid = false;
}

// -------------------------------------------------------------------------
// submit / progress / query / release (real IO implementation)
// -------------------------------------------------------------------------

SubmitOutcome LocalNvmeDataPath::submit_impl_(
    const DataPathRequest* requests,
    std::size_t count,
    const HostSubmitContext& ctx) {

    ++test_submit_call_count_;  // Round 15 S4 test seam (see header).
    test_last_prebuilt_entry_count_ = 0;
    test_last_dynamic_entry_count_ = 0;

    SubmitOutcome outcome;
    outcome.op = std::nullopt;
    outcome.initial_states.resize(count);

    if (count == 0) {
        outcome.status = Status::Ok();
        return outcome;
    }

    auto reject_all = [&](StatusCode code, const std::string& msg) {
        outcome.status = Status(code, msg);
        for (std::size_t i = 0; i < count; ++i) {
            outcome.initial_states[i].state = RequestState::REJECTED;
            outcome.initial_states[i].status = Status(code, msg);
        }
    };

    auto reject_one = [&](std::size_t i, StatusCode code,
                          const std::string& msg) {
        outcome.initial_states[i].state = RequestState::REJECTED;
        outcome.initial_states[i].status = Status(code, msg);
    };

    // --- Global validation (all reversible, op stays nullopt) ---
    if (!initialized_) {
        reject_all(StatusCode::NOT_READY, "DataPath not initialized");
        return outcome;
    }
    if (requests == nullptr) {
        reject_all(StatusCode::INVALID_ARGUMENT, "null requests");
        return outcome;
    }
    if (ctx.execution_domain != ExecutionDomain::DEVICE_EXECUTION) {
        reject_all(StatusCode::UNSUPPORTED, "HOST_EXECUTION not supported");
        return outcome;
    }
    if (ctx.stream == nullptr) {
        reject_all(StatusCode::INVALID_ARGUMENT, "null stream");
        return outcome;
    }
    if (!queue_group_ || queue_group_->d_qps() == nullptr) {
        reject_all(StatusCode::NOT_READY, "no queue group");
        return outcome;
    }

    // Accelerator identity check (the daemon's NVMe device_id is unrelated).
    if (ctx.accel_id >= 0 &&
        ctx.accel_id != static_cast<std::int32_t>(queue_group_->cuda_device())) {
        reject_all(StatusCode::INVALID_ARGUMENT,
                   "ctx.accel_id does not match queue group's CUDA device");
        return outcome;
    }
#if defined(TUTTI_USE_CUDA)
    int stream_accel_id = -1;
    const cudaError_t stream_error =
        cudaStreamGetDevice(ctx.stream, &stream_accel_id);
    if (stream_error != cudaSuccess) {
        reject_all(StatusCode::DEVICE_ERROR,
                   std::string("stream ownership query failed: ") +
                   cudaGetErrorString(stream_error));
        return outcome;
    }
    if (stream_accel_id != static_cast<int>(queue_group_->cuda_device())) {
        reject_all(StatusCode::INVALID_ARGUMENT,
                   "stream belongs to another accelerator");
        return outcome;
    }
#endif

    // Request count check.
    if (count > max_batch_requests_) {
        reject_all(StatusCode::RESOURCE_EXHAUSTED, "too many requests");
        return outcome;
    }

    // In-flight capacity check (before any irreversible action). Terminal
    // operations remain observable until release(), but only IN_FLIGHT work
    // consumes the advertised concurrent-operation quota.
    std::uint64_t in_flight_count = 0;
    for (const auto& [token, op] : ops_) {
        if (op.state == OpState::IN_FLIGHT) {
            ++in_flight_count;
        }
    }
    if (in_flight_count >= max_in_flight_operations_) {
        reject_all(StatusCode::RESOURCE_EXHAUSTED,
                   "in-flight operation capacity exhausted");
        return outcome;
    }

    const std::uint32_t page_size = static_cast<std::uint32_t>(ctrl_->page_size);
    const std::uint64_t block_size = block_size_;
    const std::uint64_t effective_mdts = effective_mdts_bytes_;

    // --- Per-request validation + fan-out (still reversible) ---
    struct ListInfo {
        std::uint32_t entry_idx;
        std::uint32_t start_page;
        std::uint32_t pages_in_io;
        std::uint32_t desc_idx;  // Round 16 S6: index into pr.dynamic_descs
    };
    struct PendingReq {
        bool accepted = false;
        const MemReg* mreg = nullptr;
        const LocalNvmeTargetState* tstate = nullptr;
        std::vector<DeviceSubmitEntry> entries;
        std::uint64_t total_bytes = 0;
        std::uint64_t target_token = 0;
        std::uint64_t memory_token = 0;
        std::vector<ListInfo> list_infos;
        // Round 16 S6 (REQUIRED 0): dynamic-path descriptors.  For entries
        // on the dynamic path (not pre-built), the computed prp1/prp2/
        // data_length go here; entry.prp_entry is set to nullptr as a
        // sentinel, fixed up to lease.d_desc_pool + offset after H2D.
        std::vector<AddressDescriptor> dynamic_descs;
        // Round 16 S6: entry lengths (host-side, for aggregate).  Each entry
        // (pre-built OR dynamic) pushes its length here.
        std::vector<std::uint64_t> lengths;
    };

    std::vector<PendingReq> pending(count);
    std::uint32_t total_entries = 0;
    std::uint32_t total_list_ios = 0;
    bool has_rejection = false;
    StatusCode first_rejected_code = StatusCode::OK;
    std::string first_rejected_msg;

    auto process_request = [&](std::size_t i) -> bool {
        const auto& req = requests[i];
        const auto& intent = req.intent;
        auto& pr = pending[i];

        // Per-request byte limit.
        if (intent.length > max_request_bytes_) {
            reject_one(i, StatusCode::OUT_OF_RANGE, "request exceeds max_single_io_bytes");
            return false;
        }

        const auto* tstate = find_(req.target);
        if (!tstate || tstate->dev_handle == nullptr) {
            reject_one(i, StatusCode::NOT_FOUND, "target not found or no device handle");
            return false;
        }
        pr.tstate = tstate;

        const auto* mreg = find_mem_(req.memory);
        if (!mreg || mreg->unregistered) {
            reject_one(i, StatusCode::NOT_FOUND, "memory not found");
            return false;
        }
        if (mreg->kind != DataPathMemoryKind::DEVICE) {
            reject_one(i, StatusCode::UNSUPPORTED, "HOST memory submit not supported");
            return false;
        }
        pr.mreg = mreg;

        if (intent.target_offset % block_size != 0 ||
            intent.length % block_size != 0 ||
            intent.memory_offset % block_size != 0) {
            reject_one(i, StatusCode::INVALID_ARGUMENT, "not block-aligned");
            return false;
        }
        if (intent.length == 0) {
            reject_one(i, StatusCode::INVALID_ARGUMENT, "zero length");
            return false;
        }

        if (intent.target_offset > tstate->file_size_bytes ||
            intent.length > tstate->file_size_bytes - intent.target_offset) {
            reject_one(i, StatusCode::OUT_OF_RANGE, "target bounds exceeded");
            return false;
        }
        if (intent.memory_offset > mreg->size_bytes ||
            intent.length > mreg->size_bytes - intent.memory_offset) {
            reject_one(i, StatusCode::OUT_OF_RANGE, "memory bounds exceeded");
            return false;
        }

        // Fan-out by min(MDTS, extent_remaining).
        std::uint64_t remaining = intent.length;
        std::uint64_t cur_target = intent.target_offset;
        std::uint64_t cur_mem = intent.memory_offset;
        std::uint32_t direction = (intent.direction == IoDirection::READ) ? 0 : 1;

        const bool use_prebuilt = mreg->prebuilt.valid;

        while (remaining > 0) {
            std::uint64_t sub_io = std::min(remaining, effective_mdts);

            // Check extent boundary.
            std::uint64_t ext_end = 0;
            for (const auto& ext : tstate->lba_extents) {
                std::uint64_t ext_start = ext.logical_offset_bytes;
                std::uint64_t ext_len = ext.length_blocks * block_size_;
                std::uint64_t ext_e = ext_start + ext_len;
                if (cur_target >= ext_start && cur_target < ext_e) {
                    ext_end = ext_e;
                    break;
                }
            }
            if (ext_end > 0) {
                sub_io = std::min(sub_io, ext_end - cur_target);
            }

            const std::uint64_t slice_bytes =
                mreg->prebuilt.bytes_per_slice;
            const std::uint64_t offset_in_slice = use_prebuilt
                ? cur_mem % slice_bytes : 0;
            const bool at_prebuilt_boundary = use_prebuilt &&
                offset_in_slice % effective_mdts == 0;
            const std::uint64_t expected_prebuilt_bytes =
                at_prebuilt_boundary
                ? std::min(effective_mdts, slice_bytes - offset_in_slice)
                : 0;
            if (at_prebuilt_boundary && sub_io == expected_prebuilt_bytes) {
                const std::uint64_t slice_idx = cur_mem / slice_bytes;
                const std::uint64_t sub_idx =
                    offset_in_slice / effective_mdts;
                const std::uint64_t desc_idx =
                    slice_idx * mreg->prebuilt.ios_per_slice + sub_idx;
                if (sub_idx >= mreg->prebuilt.ios_per_slice ||
                    desc_idx >= mreg->prebuilt.num_descs) {
                    reject_one(i, StatusCode::OUT_OF_RANGE,
                               "pre-built descriptor index out of range");
                    return false;
                }
                DeviceSubmitEntry entry{};
                entry.target = tstate->dev_handle;
                entry.target_offset = cur_target;
                entry.direction = direction;
                entry.prp_entry = static_cast<const AddressDescriptor*>(
                    mreg->prebuilt.d_descs) + desc_idx;
                pr.entries.push_back(entry);
                pr.lengths.push_back(sub_io);
                total_entries++;
                ++test_last_prebuilt_entry_count_;
                cur_target += sub_io;
                cur_mem += sub_io;
                remaining -= sub_io;
                continue;
            }

            std::uint32_t start_page = static_cast<std::uint32_t>(cur_mem / page_size);
            std::uint32_t pages_in_io = static_cast<std::uint32_t>(
                (sub_io + page_size - 1) / page_size);
            PrpKind kind = classify_prp(pages_in_io);

            if (start_page + pages_in_io > mreg->dma->n_ioaddrs) {
                reject_one(i, StatusCode::OUT_OF_RANGE, "PRP page out of DMA range");
                return false;
            }

            DeviceSubmitEntry entry;
            entry.target = tstate->dev_handle;
            entry.target_offset = cur_target;
            entry.direction = direction;
            entry._pad = 0;
            // Round 16 S6 (REQUIRED 0): build a descriptor for this sub-IO
            // and store in the per-request dynamic_descs vector.  The entry
            // carries a nullptr sentinel, fixed up to the arena pool GPU ptr
            // after H2D.
            entry.prp_entry = nullptr;  // sentinel: "dynamic, needs fixup"

            AddressDescriptor desc{};
            desc.prp1 = mreg->dma->ioaddrs[start_page];
            desc.data_length = sub_io;

            if (kind == PrpKind::SINGLE) {
                desc.prp2 = 0;
            } else if (kind == PrpKind::DUAL) {
                desc.prp2 = mreg->dma->ioaddrs[start_page + 1];
            } else {  // LIST
                desc.prp2 = 0;  // placeholder, filled after PRP-list alloc
                pr.list_infos.push_back({
                    static_cast<std::uint32_t>(pr.entries.size()),
                    start_page, pages_in_io,
                    static_cast<std::uint32_t>(pr.dynamic_descs.size())});
                total_list_ios++;
            }

            pr.dynamic_descs.push_back(desc);
            ++test_last_dynamic_entry_count_;

            pr.entries.push_back(entry);
            pr.lengths.push_back(sub_io);
            total_entries++;
            cur_target += sub_io;
            cur_mem += sub_io;
            remaining -= sub_io;
        }

        pr.accepted = true;
        pr.total_bytes = intent.length;
        pr.target_token = req.target.token();
        pr.memory_token = req.memory.token();
        return true;
    };

    for (std::size_t i = 0; i < count; ++i) {
        if (!process_request(i)) {
            has_rejection = true;
            const auto& st = outcome.initial_states[i].status;
            if (first_rejected_code == StatusCode::OK) {
                first_rejected_code = st.code();
                first_rejected_msg = st.message();
            }
        }
    }

    // Check if at least one accepted.
    bool any_accepted = false;
    for (const auto& pr : pending) {
        if (pr.accepted) { any_accepted = true; break; }
    }
    if (!any_accepted) {
        outcome.status = Status(first_rejected_code != StatusCode::OK
                                ? first_rejected_code
                                : StatusCode::INVALID_ARGUMENT,
                                first_rejected_msg.empty()
                                ? "all requests rejected"
                                : first_rejected_msg);
        return outcome;
    }

    const bool step_feeder = ctx.step_layer_count > 0;
    if (step_feeder && has_rejection) {
        reject_all(StatusCode::INVALID_ARGUMENT,
                   "step feeder requires all requests to validate");
        return outcome;
    }

    // Entry count check (after fan-out).
    if (total_entries > max_batch_entries_) {
        reject_all(StatusCode::RESOURCE_EXHAUSTED, "too many sub-IOs (entries)");
        return outcome;
    }

    // Batch byte limit check.
    std::uint64_t total_batch_bytes = 0;
    for (const auto& pr : pending) {
        if (pr.accepted) total_batch_bytes += pr.total_bytes;
    }
    if (total_batch_bytes > caps_.max_batch_bytes) {
        reject_all(StatusCode::RESOURCE_EXHAUSTED, "batch bytes exceed limit");
        return outcome;
    }

    // ================================================================
    // Irreversible resource reservation (all before launch).
    // If any step fails here, nothing has been irreversibly issued,
    // so we can safely clean up and return op=nullopt.
    // ================================================================

    // ================================================================
    // Arena lease (zero cudaMalloc/cudaEventCreate in hot path).
    // ================================================================

    MetadataArena::Lease lease;
    if (!arena_.acquire(lease)) {
        reject_all(StatusCode::RESOURCE_EXHAUSTED,
                   "MetadataArena exhausted (all slots in use or leaked)");
        return outcome;
    }

    cudaEvent_t event = static_cast<cudaEvent_t>(lease.event);
    DeviceSubmitEntry* d_entries = lease.d_entries;
    EntryCompletionStatus* d_status = lease.d_status;
    cudaError_t ce;

    PrpBufRef prp_buf_ref;
    nvm_dma_t* prp_dma = nullptr;
    std::uint32_t prp_ioaddrs_base = 0;

    // Fill PRP-list pages if needed.
    // Two paths:
    //   - PrpPageCache enabled: content-addressed host-pinned cache.
    //   - Cache miss/exhaustion: growing host-pinned PrpBufPool.
    // No PRP-list page is ever cudaMalloc'd or copied H2D.
    std::vector<OpEntry::PrpCacheRef> prp_cache_refs;  // filled if cache used
    bool prp_all_from_cache = false;  // true if ALL PRP pages came from cache

    if (total_list_ios > 0) {
        if (prp_cache_.enabled()) {
            // Try PRP cache path: ONE locked batch resolves all LIST pages
            // (was: one cache mutex round-trip per entry).
            bool cache_ok = true;
            std::vector<PrpPageCache::BatchItem> items;
            items.reserve(total_list_ios);
            for (const auto& pr : pending) {
                if (!pr.accepted) continue;
                for (const auto& li : pr.list_infos) {
                    PrpPageCache::BatchItem item;
                    item.key.memory_token = pr.memory_token;
                    item.key.start_page = li.start_page;
                    item.key.pages_in_io = li.pages_in_io;
                    item.data_dma = pr.mreg->dma;
                    items.push_back(item);
                }
            }
            prp_cache_.get_or_build_batch(items.data(), items.size());
            std::size_t item_idx = 0;
            for (auto& pr : pending) {
                if (!pr.accepted) continue;
                for (const auto& li : pr.list_infos) {
                    auto* pe = items[item_idx++].result;
                    if (pe == nullptr) {
                        cache_ok = false;
                        break;
                    }
                    pr.dynamic_descs[li.desc_idx].prp2 = pe->ioaddr;
                    prp_cache_refs.push_back({pe, pe->ioaddr});
                }
                if (!cache_ok) break;
            }

            if (!cache_ok) {
                // Cache exhausted: release ALL checkouts (including items
                // after the failing one that got entries but no ref) and
                // fall back to the growing host-pinned pool.
                for (const auto& item : items) {
                    if (item.result != nullptr)
                        prp_cache_.release_checkout(item.result);
                }
                prp_cache_refs.clear();
                // Reset all prp2 fields to 0 (filled by host pool below).
                for (auto& pr : pending) {
                    if (!pr.accepted) continue;
                    for (const auto& li : pr.list_infos) {
                        pr.dynamic_descs[li.desc_idx].prp2 = 0;
                    }
                }
                // Fall through to host pool path below.
            } else {
                prp_all_from_cache = true;
            }
        }

        if (!prp_all_from_cache) {
            prp_buf_ref = prp_buf_pool_.alloc_pages(total_list_ios);
            if (!prp_buf_ref.valid) {
                arena_.release(lease.slot_index);
                reject_all(StatusCode::RESOURCE_EXHAUSTED,
                           "host PrpBufPool allocation failed");
                return outcome;
            }
            prp_dma = prp_buf_ref.segment;
            prp_ioaddrs_base = static_cast<std::uint32_t>(prp_buf_ref.base_page);
            std::uint32_t list_idx = 0;
            for (auto& pr : pending) {
                if (!pr.accepted) continue;
                for (const auto& li : pr.list_infos) {
                    auto* host_page = reinterpret_cast<std::uint64_t*>(
                        static_cast<char*>(prp_dma->vaddr) +
                        (prp_buf_ref.base_page + list_idx) * page_size);
                    fill_prp_list_page(host_page, pr.mreg->dma,
                                       li.start_page, li.pages_in_io, page_size);
                    pr.dynamic_descs[li.desc_idx].prp2 =
                        prp_dma->ioaddrs[prp_buf_ref.base_page + list_idx];
                    list_idx++;
                }
            }
        }
    }

    // Flatten entries + synchronous H2D into arena's entry pool.
    // Round 16 S6 (REQUIRED 0): also flatten dynamic descriptors, H2D them
    // to the arena's per-slot descriptor pool, and fix up entry.prp_entry
    // pointers from nullptr (sentinel) to the GPU pool address.
    std::vector<DeviceSubmitEntry> h_entries;
    h_entries.reserve(total_entries);
    std::vector<AddressDescriptor> h_dynamic_descs;
    std::vector<std::uint64_t> h_entry_lengths;  // Round 16 S6: for aggregate
    std::uint64_t total_bytes = 0;
    for (const auto& pr : pending) {
        if (!pr.accepted) continue;
        for (const auto& e : pr.entries)
            h_entries.push_back(e);
        for (const auto& d : pr.dynamic_descs)
            h_dynamic_descs.push_back(d);
        for (const auto& l : pr.lengths)
            h_entry_lengths.push_back(l);
        total_bytes += pr.total_bytes;
    }

    std::vector<StepFeederLayer> h_feeder_layers;
    if (step_feeder) {
        if (!ctx.step_layer_request_offsets || ctx.step_staging_depth == 0 ||
            ctx.step_layer_request_offsets[0] != 0 ||
            ctx.step_layer_request_offsets[ctx.step_layer_count] != count) {
            arena_.release(lease.slot_index);
            reject_all(StatusCode::INVALID_ARGUMENT,
                       "invalid step feeder layer offsets/depth");
            return outcome;
        }
        h_feeder_layers.resize(ctx.step_layer_count);
        std::uint32_t cursor = 0;
        for (std::uint32_t layer = 0; layer < ctx.step_layer_count; ++layer) {
            const std::uint32_t begin = ctx.step_layer_request_offsets[layer];
            const std::uint32_t end = ctx.step_layer_request_offsets[layer + 1];
            if (begin > end || end > count) {
                arena_.release(lease.slot_index);
                reject_all(StatusCode::INVALID_ARGUMENT,
                           "non-monotonic step feeder offsets");
                return outcome;
            }
            StepFeederLayer plan{cursor, 0};
            for (std::uint32_t req = begin; req < end; ++req)
                plan.entry_count += pending[req].entries.size();
            cursor += plan.entry_count;
            h_feeder_layers[layer] = plan;
        }
        if (cursor != total_entries) {
            arena_.release(lease.slot_index);
            reject_all(StatusCode::INTERNAL,
                       "step feeder entry coverage mismatch");
            return outcome;
        }
    }

    // H2D dynamic descriptors to the arena's per-slot descriptor pool,
    // then fix up entry.prp_entry from nullptr sentinel to GPU pointer.
    if (!h_dynamic_descs.empty()) {
        const AddressDescriptor* d_desc_base = lease.d_desc_pool;
        ce = cudaMemcpyAsync(const_cast<AddressDescriptor*>(d_desc_base),
                         h_dynamic_descs.data(),
                         h_dynamic_descs.size() * sizeof(AddressDescriptor),
                         cudaMemcpyHostToDevice, ctx.stream);
        if (ce != cudaSuccess) {
            arena_.release(lease.slot_index);
            for (const auto& ref : prp_cache_refs) {
                prp_cache_.release_checkout(ref.entry);
            }
            reject_all(StatusCode::DEVICE_ERROR, "H2D dynamic descriptors failed");
            return outcome;
        }
        // Fix up entries: walk h_entries, for each with prp_entry == nullptr
        // (dynamic sentinel), assign d_desc_base + running index.
        std::uint32_t desc_idx = 0;
        for (auto& e : h_entries) {
            if (e.prp_entry == nullptr) {
                e.prp_entry = d_desc_base + desc_idx;
                ++desc_idx;
            }
        }
    }

    // ASYNC H2D on caller stream: entries array is pageable host memory,
    // so cudaMemcpyAsync blocks the host until the copy completes (h_entries
    // is safe to destroy after this call).  The copy is on ctx.stream,
    // ordered with the subsequent kernel launch — no cross-stream barrier.
    ce = cudaMemcpyAsync(d_entries, h_entries.data(),
                     total_entries * sizeof(DeviceSubmitEntry),
                     cudaMemcpyHostToDevice, ctx.stream);
    if (ce != cudaSuccess) {
        arena_.release(lease.slot_index);
        reject_all(StatusCode::DEVICE_ERROR, "H2D entries failed");
        return outcome;
    }

    // Initialize every status as pending.  Success is written explicitly by
    // the device path, so a skipped or unexecuted entry cannot aggregate as
    // a false success.  0xFF maps to kEntryCompletionPending in result.
    ce = cudaMemsetAsync(d_status, 0xFF,
                     total_entries * sizeof(EntryCompletionStatus),
                     ctx.stream);
    if (ce != cudaSuccess) {
        arena_.release(lease.slot_index);
        reject_all(StatusCode::DEVICE_ERROR, "cudaMemset d_status failed");
        return outcome;
    }

    StepFeederLayer* d_feeder_layers = nullptr;
    std::uint32_t* h_feeder_ready = nullptr;
    std::uint32_t* d_feeder_ready = nullptr;
    std::uint32_t* h_feeder_release = nullptr;
    std::uint32_t* d_feeder_release = nullptr;
    cudaStream_t feeder_poll_stream = nullptr;
    auto free_feeder = [&]() {
        if (d_feeder_layers) cudaFree(d_feeder_layers);
        if (h_feeder_ready) cudaFreeHost(h_feeder_ready);
        if (h_feeder_release) cudaFreeHost(h_feeder_release);
        if (feeder_poll_stream) cudaStreamDestroy(feeder_poll_stream);
        d_feeder_layers = nullptr;
        h_feeder_ready = d_feeder_ready = nullptr;
        h_feeder_release = d_feeder_release = nullptr;
    };
    if (step_feeder) {
        const std::size_t plan_bytes =
            h_feeder_layers.size() * sizeof(StepFeederLayer);
        const std::size_t flags_bytes =
            ctx.step_layer_count * sizeof(std::uint32_t);
        ce = cudaMalloc(reinterpret_cast<void**>(&d_feeder_layers), plan_bytes);
        if (ce == cudaSuccess)
            ce = cudaMemcpyAsync(d_feeder_layers, h_feeder_layers.data(),
                                 plan_bytes, cudaMemcpyHostToDevice, ctx.stream);
        if (ce == cudaSuccess)
            ce = cudaHostAlloc(reinterpret_cast<void**>(&h_feeder_ready),
                               flags_bytes, cudaHostAllocMapped);
        if (ce == cudaSuccess)
            ce = cudaHostAlloc(reinterpret_cast<void**>(&h_feeder_release),
                               flags_bytes, cudaHostAllocMapped);
        if (ce == cudaSuccess) {
            std::memset(h_feeder_ready, 0, flags_bytes);
            std::memset(h_feeder_release, 0, flags_bytes);
            ce = cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&d_feeder_ready), h_feeder_ready, 0);
        }
        if (ce == cudaSuccess)
            ce = cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&d_feeder_release),
                h_feeder_release, 0);
        if (ce != cudaSuccess) {
            free_feeder();
            arena_.release(lease.slot_index);
            for (const auto& ref : prp_cache_refs)
                prp_cache_.release_checkout(ref.entry);
            reject_all(StatusCode::DEVICE_ERROR,
                       "step feeder metadata allocation failed");
            return outcome;
        }
    }

    // 6. Launch kernel.
    //    If test injection is active, simulate launch failure (kernel NOT issued).
    cudaError_t launch_err = cudaSuccess;
    std::uint32_t inject_flag = 0u;
    if (test_inject_resolve_lba_failure_) inject_flag |= 0x1u;
    if (test_inject_nvme_error_)          inject_flag |= 0x2u;
    if (test_inject_launch_failure_) {
        launch_err = cudaErrorUnknown;
    } else {
        // FIX 1: inject_flag is a scalar passed by value — no per-op device
        // allocation.  bit0 = resolve_lba failure, bit1 = synthesize NVMe CQ
        // error (both test-only; production path is inject_flag == 0).
        const NvtxIoStyle nvtx_style = nvtx_io_style(requests, count);
        nvtx_push_io(nvtx_style);
        // Keep the legacy exact marker as a nested range for existing report
        // queries; the outer range carries the direction and color.
        nvtxRangePushA("tutti.local_nvme.io_kernel");
        if (!step_feeder)
            launch_err = launch_submit_one(
                d_entries, d_status, total_entries, cq_poll_budget_,
                threads_per_block_, inject_flag, ctx.stream);
        nvtxRangePop();
        nvtxRangePop();
    }
    if (launch_err != cudaSuccess) {
        free_feeder();
        // Launch failed: kernel was NOT issued (cudaGetLastError returns
        // launch configuration errors, not runtime errors).
        // Safe to return the arena slot.
        arena_.release(lease.slot_index);
        // Release any PRP cache checkouts acquired (no IO issued).
        for (const auto& ref : prp_cache_refs) {
            prp_cache_.release_checkout(ref.entry);
        }
        reject_all(StatusCode::DEVICE_ERROR,
                   std::string("kernel launch failed: ") +
                   cudaGetErrorString(launch_err));
        return outcome;  // op = nullopt, zero issued
    }

    // Round 15 S4 test seam: kernel was successfully issued exactly once
    // for this submit() call (one launch_submit_one() call above, whether
    // or not the subsequent cudaEventRecord succeeds).
    test_kernel_launch_count_ += step_feeder
        ? (ctx.step_layer_count + ctx.step_staging_depth - 1) /
              ctx.step_staging_depth
        : 1;

    // 7. Record completion fence (after successful launch).
    //
    //    The cudaEventRecord on ctx.stream IS the completion fence:
    //    any work enqueued on ctx.stream AFTER this point is
    //    Happens-After the IO kernel's completion.  The event signals
    //    when the GPU kernel returns — which means all NVMe commands
    //    have been submitted and their CQ entries polled (the kernel
    //    does not return until all entries reach a terminal state).
    //
    //    Fence semantics:
    //      - Same-stream ordering:  caller enqueues compute kernel A,
    //        then calls submit() (which enqueues H2D + IO kernel +
    //        fence on ctx.stream), then enqueues compute kernel B.
    //        B reads data produced by the IO kernel — guaranteed by
    //        stream ordering: A → H2D → IO → fence → B.
    //
    //      - Cross-stream ordering: caller enqueues producer work on
    //        stream P, records event EP, then calls submit() on IO
    //        stream I with cudaStreamWaitEvent(I, EP) beforehand.
    //        After submit() returns, caller records event EI on I.
    //        Consumer stream C does cudaStreamWaitEvent(C, EI) before
    //        reading IO results.  No cudaStreamSynchronize needed.
    //
    //    If cudaEventRecord fails, IO has been irreversibly issued.
    //    We must NOT return op=nullopt (the kernel is running and
    //    will complete on its own — the arena slot is in use).
    ce = step_feeder ? cudaSuccess : (test_inject_event_record_failure_
        ? cudaErrorUnknown
        : cudaEventRecord(event, ctx.stream));
    if (ce != cudaSuccess) {
        // =============================================================
        // EXPLICIT EXCEPTION: cudaStreamSynchronize on ctx.stream.
        //
        // Justification: cudaEventRecord failed AFTER the IO kernel
        // was successfully launched on ctx.stream.  The kernel is
        // running and will modify device memory (d_entries, d_status)
        // and perform NVMe DMA.  We cannot return op=nullopt because
        // the arena slot is borrowed and the kernel is in flight.
        //
        // Without the event, progress() cannot use cudaEventQuery to
        // detect completion.  The only safe recovery is to synchronize
        // the stream: this blocks the host until the kernel finishes,
        // then we store the op in a terminal state (COMPLETED or
        // FAILED depending on sync result).
        //
        // This is the ONLY cudaStreamSynchronize on the caller's
        // stream in the production submit path.  It fires only on
        // cudaEventRecord failure — an extremely rare CUDA runtime
        // error.  In normal operation, this path is never taken.
        // =============================================================
        cudaError_t sync_err = cudaStreamSynchronize(ctx.stream);
        if (sync_err != cudaSuccess) {
            cudaGetLastError();  // clear error
        }

        std::uint64_t op_token = next_op_token_++;
        OpEntry op;
        op.state = (sync_err == cudaSuccess) ? OpState::COMPLETED : OpState::FAILED;
        op.status = (sync_err == cudaSuccess)
            ? Status::Ok()
            : Status(StatusCode::DEVICE_ERROR,
                     "stream sync failed after event record failure");
        op.bytes_transferred = (sync_err == cudaSuccess) ? total_bytes : 0;
        op.total_bytes = total_bytes;
        if (sync_err != cudaSuccess) {
            set_completion_failure(op.completion_detail,
                                   IoFailureKind::CUDA_QUERY_ERROR, 0);
        }
        op.d_entries = d_entries;
        op.d_status = d_status;
        op.entry_count = total_entries;
        op.entry_lengths = std::move(h_entry_lengths);  // Round 16 S6
        op.event = event;
        op.stream = ctx.stream;
        op.completion_mode = CompletionMode::EVENT;
        op.arena_slot = lease.slot_index;
        op.prp_list_dma = prp_dma;
        op.prp_ioaddrs_base = prp_ioaddrs_base;
        op.prp_buf_ref = prp_buf_ref;
        op.prp_list_page_count = total_list_ios;
        op.op_token = op_token;
        op.op_generation = 1;
        op.d_feeder_layers = d_feeder_layers;
        op.h_feeder_ready = h_feeder_ready;
        op.d_feeder_ready = d_feeder_ready;
        op.h_feeder_release = h_feeder_release;
        op.d_feeder_release = d_feeder_release;
        op.feeder_layer_count = ctx.step_layer_count;
        op.feeder_staging_depth = ctx.step_staging_depth;
        op.feeder_is_read = step_feeder && h_entries.front().direction == 0;
        op.feeder_poll_stream = feeder_poll_stream;
        // Pin cache entries (IO was issued; entries must survive until release).
        op.prp_cache_refs = std::move(prp_cache_refs);
        for (const auto& ref : op.prp_cache_refs) {
            prp_cache_.pin(ref.entry);
        }
        // Pin handle cache entries for all referenced targets.
        for (const auto& pr : pending) {
            if (!pr.accepted) continue;
            op.target_tokens.push_back(pr.target_token);
            op.memory_tokens.push_back(pr.memory_token);
            // Pin handle cache entry if cache-owned.
            auto tit = targets_.find(pr.target_token);
            if (tit != targets_.end() && tit->second.cache_entry != nullptr) {
                handle_cache_.pin(tit->second.cache_entry);
                op.handle_cache_refs.push_back(tit->second.cache_entry);
            }
        }
        // If sync succeeded, aggregate per-entry completion status.
        if (sync_err == cudaSuccess) {
            aggregate_completion_status_(op);
        }
        ops_[op_token] = std::move(op);

        // Outcome: partial commit status if any rejection.
        if (has_rejection) {
            outcome.status = Status(first_rejected_code,
                                    "partial commit: " + first_rejected_msg);
        } else {
            outcome.status = Status::Ok();
        }
        outcome.op = detail::SpiIdentityMint::mint<detail::DataPathOpTag>(
            op_token, 1);
        for (std::size_t i = 0; i < count; ++i) {
            if (pending[i].accepted) {
                outcome.initial_states[i].state = RequestState::ACCEPTED;
                outcome.initial_states[i].status = Status::Ok();
            }
        }
        return outcome;
    }

    // 8. Success: store op.
    std::uint64_t op_token = next_op_token_++;
    OpEntry op;
    op.state = OpState::IN_FLIGHT;
    op.status = Status::Ok();
    op.bytes_transferred = 0;
    op.total_bytes = total_bytes;
    op.d_entries = d_entries;
    op.d_status = d_status;
    op.entry_count = total_entries;
        op.entry_lengths = std::move(h_entry_lengths);  // Round 16 S6
    op.event = event;
    op.stream = ctx.stream;
    op.completion_mode = CompletionMode::EVENT;
    op.arena_slot = lease.slot_index;
    op.prp_list_dma = prp_dma;
    op.prp_ioaddrs_base = prp_ioaddrs_base;
    op.prp_buf_ref = prp_buf_ref;
    op.prp_list_page_count = total_list_ios;
    op.op_token = op_token;
    op.op_generation = 1;
    op.d_feeder_layers = d_feeder_layers;
    op.h_feeder_ready = h_feeder_ready;
    op.d_feeder_ready = d_feeder_ready;
    op.h_feeder_release = h_feeder_release;
    op.d_feeder_release = d_feeder_release;
    op.feeder_layer_count = ctx.step_layer_count;
    op.feeder_staging_depth = ctx.step_staging_depth;
    op.feeder_is_read = step_feeder && h_entries.front().direction == 0;
    op.feeder_poll_stream = feeder_poll_stream;
    op.feeder_event_recorded = step_feeder
        ? std::make_shared<std::atomic<bool>>(false) : nullptr;
    // Pin cache entries (IO was issued; entries must survive until release).
    op.prp_cache_refs = std::move(prp_cache_refs);
    for (const auto& ref : op.prp_cache_refs) {
        prp_cache_.pin(ref.entry);
    }
    // Pin handle cache entries for all referenced targets.
    for (const auto& pr : pending) {
        if (!pr.accepted) continue;
        op.target_tokens.push_back(pr.target_token);
        op.memory_tokens.push_back(pr.memory_token);
        // Pin handle cache entry if cache-owned.
        auto tit = targets_.find(pr.target_token);
        if (tit != targets_.end() && tit->second.cache_entry != nullptr) {
            handle_cache_.pin(tit->second.cache_entry);
            op.handle_cache_refs.push_back(tit->second.cache_entry);
        }
    }
    auto feeder_recorded = op.feeder_event_recorded;
    ops_[op_token] = std::move(op);
    if (step_feeder) {
        const auto layer_count = ctx.step_layer_count;
        const auto depth = ctx.step_staging_depth;
        const bool is_read = h_entries.front().direction == 0;
        const auto device = cuda_device_;
        const auto poll_budget = cq_poll_budget_;
        const auto threads = threads_per_block_;
        const bool diag = feeder_diag_enabled();
        std::thread([
            d_entries, d_status, d_feeder_layers, d_feeder_ready,
            d_feeder_release, h_feeder_ready, h_feeder_release,
            layer_count, depth, is_read, device, poll_budget, threads,
            inject_flag, event, stream = ctx.stream, feeder_recorded, diag
        ] {
            cudaSetDevice(device);
            bool failed = false;
            for (std::uint32_t base = 0; base < layer_count; base += depth) {
                const std::uint32_t window =
                    std::min(depth, layer_count - base);
                volatile std::uint32_t* gates = is_read
                    ? h_feeder_release : h_feeder_ready;
                const std::uint32_t gate_base = is_read && base > 0
                    ? base - depth : base;
                if (!is_read || base > 0) {
                    if (diag)
                        std::fprintf(stderr, "FEEDER_DIAG cpp window_wait_begin t_ns=%lld backend=local direction=%s base=%u gate_base=%u window=%u\n",
                                     feeder_diag_ns(), is_read ? "read" : "write",
                                     base, gate_base, window);
                    for (;;) {
                        bool ready = true;
                        for (std::uint32_t local = 0; local < window; ++local)
                            ready &= gates[gate_base + local] != 0;
                        if (ready) break;
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }
                    if (diag)
                        std::fprintf(stderr, "FEEDER_DIAG cpp window_wait_consume t_ns=%lld backend=local direction=%s base=%u gate_base=%u window=%u\n",
                                     feeder_diag_ns(), is_read ? "read" : "write",
                                     base, gate_base, window);
                }
                if (diag)
                    std::fprintf(stderr, "FEEDER_DIAG cpp window_launch t_ns=%lld backend=local direction=%s base=%u window=%u\n",
                                 feeder_diag_ns(), is_read ? "read" : "write",
                                 base, window);
                cudaError_t ce = launch_step_feeder(
                    d_entries, d_status, d_feeder_layers + base, window, base,
                    d_feeder_ready, d_feeder_release, poll_budget, threads,
                    inject_flag, stream);
                if (ce != cudaSuccess) {
                    failed = true;
                    break;
                }
            }
            if (failed) {
                for (std::uint32_t layer = 0; layer < layer_count; ++layer) {
                    if (is_read) h_feeder_ready[layer] = 2u;
                    else h_feeder_release[layer] = 1u;
                }
            }
            (void)cudaEventRecord(event, stream);
            feeder_recorded->store(true, std::memory_order_release);
            if (diag)
                std::fprintf(stderr, "FEEDER_DIAG cpp final_event_recorded t_ns=%lld backend=local direction=%s failed=%d\n",
                             feeder_diag_ns(), is_read ? "read" : "write",
                             failed ? 1 : 0);
        }).detach();
    }

    // Outcome: partial commit status if any rejection.
    if (has_rejection) {
        outcome.status = Status(first_rejected_code,
                                "partial commit: " + first_rejected_msg);
    } else {
        outcome.status = Status::Ok();
    }
    outcome.op = detail::SpiIdentityMint::mint<detail::DataPathOpTag>(
        op_token, 1);
    for (std::size_t i = 0; i < count; ++i) {
        if (pending[i].accepted) {
            outcome.initial_states[i].state = RequestState::ACCEPTED;
            outcome.initial_states[i].status = Status::Ok();
        }
    }

    return outcome;
}

Result<ProgressResult> LocalNvmeDataPath::progress_impl_(ProgressBudget budget) {
    ProgressResult result{};
    result.work_units_consumed = 0;
    result.operations_advanced = 0;
    result.operations_terminal = 0;
    result.more_work_likely = false;

    // Both caps are hard: if either is 0, return immediately with zero work.
    if (budget.max_work_units == 0 || budget.timeout_ns == 0) {
        // Still report whether there's in-flight work.
        for (const auto& [tok, op] : ops_) {
            if (op.state == OpState::IN_FLIGHT) {
                result.more_work_likely = true;
                break;
            }
        }
        return Result<ProgressResult>::Success(std::move(result));
    }

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::nanoseconds(budget.timeout_ns);

    std::uint64_t work_done = 0;

    for (auto& [tok, op] : ops_) {
        // Check deadline before each query.
        if (std::chrono::steady_clock::now() >= deadline) {
            result.more_work_likely = true;
            break;
        }

        if (op.state != OpState::IN_FLIGHT) continue;

        if (op.feeder_event_recorded &&
            !op.feeder_event_recorded->load(std::memory_order_acquire)) {
            ++result.operations_advanced;
            ++work_done;
            continue;
        }

        // Check work cap before each query.
        if (work_done >= budget.max_work_units) {
            result.more_work_likely = true;
            break;
        }

        // One query = one work unit. Runtime::wait() supplies the polling
        // cadence; spinning here defeats max_work_units and can generate
        // millions of CUDA API calls while an event is not ready.
        cudaError_t ce;
        if (op.completion_mode == CompletionMode::EVENT) {
            ce = cudaEventQuery(static_cast<cudaEvent_t>(op.event));
        } else {
            // STREAM_QUERY fallback.
            ce = cudaStreamQuery(static_cast<cudaStream_t>(op.stream));
        }

        // FIX 3 test seam: simulate a persistent query error (not NotReady)
        // to prove op → FAILED instead of stuck IN_FLIGHT.
        if (test_inject_query_error_) {
            ce = cudaErrorUnknown;
        }

        if (ce == cudaSuccess) {
            // Stream/event signaled: kernel finished. But kernel completion
            // only means the GPU kernel returned — we must D2H the per-entry
            // status array to check if any NVMe command actually failed.
            aggregate_completion_status_(op);
            cudaGetLastError();  // clear any error from D2H
            ++result.operations_terminal;
        } else if (ce == cudaErrorNotReady) {
            ++result.operations_advanced;
        } else {
            // FIX 3: real CUDA error (not success, not NotReady) → op FAILED.
            // Previously this branch was dead code because the error was
            // rewritten to NotReady, leaving the op stuck IN_FLIGHT forever.
            // We record the terminal state, then clear the sticky error so it
            // does not pollute subsequent queries in this loop.
            op.state = OpState::FAILED;
            op.status = Status(StatusCode::DEVICE_ERROR,
                               "cudaEventQuery error: " +
                               std::string(cudaGetErrorString(ce)));
            set_completion_failure(op.completion_detail,
                                   IoFailureKind::CUDA_QUERY_ERROR,
                                   op.bytes_transferred);
            cudaGetLastError();  // clear sticky error
            ++result.operations_terminal;
        }
        ++work_done;
    }

    result.work_units_consumed = work_done;

    // Check if any ops remain in flight.
    for (const auto& [tok, op] : ops_) {
        if (op.state == OpState::IN_FLIGHT) {
            result.more_work_likely = true;
            break;
        }
    }

    return Result<ProgressResult>::Success(std::move(result));
}

Result<DataPathSnapshot> LocalNvmeDataPath::query_impl_(DataPathOp op) const {
    const auto* entry = find_op_(op);
    if (!entry) {
        return Result<DataPathSnapshot>::Failure(
            Status(StatusCode::NOT_FOUND,
                   "query: op not found"));
    }

    DataPathSnapshot snap;
    snap.state = entry->state;
    snap.status = entry->status;
    snap.bytes_transferred = entry->bytes_transferred;
    snap.detail = entry->completion_detail;
    return Result<DataPathSnapshot>::Success(std::move(snap));
}

Status LocalNvmeDataPath::release_impl_(DataPathOp op) {
    auto* entry = find_op_(op);
    if (!entry) {
        return Status(StatusCode::NOT_FOUND, "release: op not found");
    }

    if (entry->state == OpState::IN_FLIGHT) {
        return Status(StatusCode::BUSY,
                      "release: op is still in flight");
    }
    if (entry->d_feeder_layers) cudaFree(entry->d_feeder_layers);
    if (entry->h_feeder_ready) cudaFreeHost(entry->h_feeder_ready);
    if (entry->h_feeder_release) cudaFreeHost(entry->h_feeder_release);
    if (entry->feeder_poll_stream)
        cudaStreamDestroy(static_cast<cudaStream_t>(entry->feeder_poll_stream));

    // Terminal: GPU metadata slot is reusable. Host PRP backing is pool-owned
    // and remains stable; timeout-safe retention is handled at pool shutdown.
    if (entry->arena_slot != UINT32_MAX) {
        if (entry->has_timeout) {
            arena_.release_with_timeout_leak(entry->arena_slot);
        } else {
            arena_.release(entry->arena_slot);
        }
        entry->arena_slot = UINT32_MAX;
    }
    if (entry->has_timeout) timeout_prp_retained_ = true;

    // Unpin cache entries (handle + PRP).
    // Note: for timeout ops, PRP cache entries are NOT unpinned (conservative
    // retention, consistent with arena slot leak).  Handle cache entries ARE
    // unpinned (the handle is target-owned, not op-owned; the target's close()
    // will handle handle cache release).
    if (!entry->has_timeout) {
        for (const auto& ref : entry->prp_cache_refs) {
            prp_cache_.unpin(ref.entry);
        }
    }
    for (auto* hc_entry : entry->handle_cache_refs) {
        handle_cache_.unpin(hc_entry);
    }
    entry->prp_cache_refs.clear();
    entry->handle_cache_refs.clear();

    ops_.erase(op.token());
    return Status::Ok();
}

// -------------------------------------------------------------------------
// test-only accessors
// -------------------------------------------------------------------------

const LocalNvmeTargetState* LocalNvmeDataPath::test_target_state(
    DataPathTarget target) const {
    return find_(target);
}

const nvm_dma_t* LocalNvmeDataPath::test_dma_handle(
    DataPathMemory memory) const {
    const auto* reg = find_mem_(memory);
    if (!reg || reg->unregistered) {
        return nullptr;
    }
    return reg->dma;
}

std::uint32_t LocalNvmeDataPath::test_queue_group_id() const {
    return queue_group_ ? queue_group_->group_id() : 0;
}

const void* LocalNvmeDataPath::test_d_qps() const {
    return queue_group_ ? queue_group_->d_qps() : nullptr;
}

std::uint32_t LocalNvmeDataPath::test_n_qps() const {
    return queue_group_ ? queue_group_->n_qps() : 0;
}

const void* LocalNvmeDataPath::test_dev_handle(DataPathTarget target) const {
    const auto* state = find_(target);
    if (!state) return nullptr;
    return state->dev_handle;
}

std::uint64_t LocalNvmeDataPath::test_hardware_mdts() const {
    return hardware_mdts_bytes_;
}

std::uint64_t LocalNvmeDataPath::test_effective_mdts() const {
    return effective_mdts_bytes_;
}

std::uint64_t LocalNvmeDataPath::test_prp_list_page_capacity() const {
    return prp_list_page_capacity_;
}

std::uint32_t LocalNvmeDataPath::test_in_flight_count() const {
    std::uint32_t n = 0;
    for (const auto& [tok, op] : ops_) {
        if (op.state == OpState::IN_FLIGHT) ++n;
    }
    return n;
}

bool LocalNvmeDataPath::test_op_has_resources(DataPathOp op) const {
    const auto* entry = find_op_(op);
    if (!entry || entry->arena_slot == UINT32_MAX) {
        return false;
    }
    return true;
}

void LocalNvmeDataPath::test_set_inject_launch_failure(bool v) {
    test_inject_launch_failure_ = v;
}

void LocalNvmeDataPath::test_set_inject_event_record_failure(bool v) {
    test_inject_event_record_failure_ = v;
}

// -------------------------------------------------------------------------
// test-only: per-op entry / PRP-list observability
// -------------------------------------------------------------------------

std::uint32_t LocalNvmeDataPath::test_entry_count(DataPathOp op) const {
    const auto* e = find_op_(op);
    return e ? e->entry_count : 0;
}

bool LocalNvmeDataPath::test_copy_entry(DataPathOp op, std::uint32_t index,
                                        DeviceSubmitEntry& out) const {
    const auto* e = find_op_(op);
    if (!e || !e->d_entries) return false;
    if (index >= e->entry_count) return false;
    // TEST-ONLY SYNC POINT: synchronous D2H of one entry for test
    // observability.  Entries are filled before launch and are not
    // mutated by the kernel, so this observes the submitted descriptor.
    // Not in the production submit/progress path.
    cudaError_t ce = cudaMemcpy(&out, e->d_entries + index,
                                sizeof(DeviceSubmitEntry),
                                cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
        cudaGetLastError();  // clear sticky error
        return false;
    }
    return true;
}

// Round 16 S6 (REQUIRED 0): copy a single entry's AddressDescriptor from
// GPU to host for test observability.  The entry's prp_entry pointer points
// to a GPU-resident AddressDescriptor (either pre-built or arena pool).
bool LocalNvmeDataPath::test_copy_entry_desc(DataPathOp op, std::uint32_t index,
                                             AddressDescriptor& out) const {
    DeviceSubmitEntry e{};
    if (!test_copy_entry(op, index, e)) return false;
    if (e.prp_entry == nullptr) return false;
    cudaError_t ce = cudaMemcpy(&out, e.prp_entry,
                                sizeof(AddressDescriptor),
                                cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

bool LocalNvmeDataPath::test_op_has_prp_list_dma(DataPathOp op) const {
    const auto* e = find_op_(op);
    return e && e->prp_list_page_count > 0;
}

std::uint64_t LocalNvmeDataPath::test_prp_list_ioaddr(
    DataPathOp op, std::uint32_t list_idx) const {
    const auto* e = find_op_(op);
    if (!e) return 0;
    // PRP cache path: return from cached entries.
    if (!e->prp_cache_refs.empty()) {
        if (list_idx >= e->prp_cache_refs.size()) return 0;
        return e->prp_cache_refs[list_idx].ioaddr;
    }
    // Arena path.
    if (!e->prp_list_dma || list_idx >= e->prp_list_page_count) return 0;
    return e->prp_list_dma->ioaddrs[e->prp_ioaddrs_base + list_idx];
}

std::uint32_t LocalNvmeDataPath::test_prp_list_page_count(DataPathOp op) const {
    const auto* e = find_op_(op);
    if (!e) return 0;
    // PRP cache path.
    if (!e->prp_cache_refs.empty()) {
        return static_cast<std::uint32_t>(e->prp_cache_refs.size());
    }
    // Arena path.
    return e->prp_list_page_count;
}

// -------------------------------------------------------------------------
// test-only: per-entry completion status observability
// -------------------------------------------------------------------------

bool LocalNvmeDataPath::test_copy_completion_status(
    DataPathOp op,
    std::vector<std::uint32_t>& out_results) const {
    const auto* e = find_op_(op);
    if (!e || !e->d_status || e->entry_count == 0) return false;

    // TEST-ONLY SYNC POINT: synchronous D2H of the status result field.
    // Not in the production submit/progress path.
    // We only copy the `result` field (first uint32_t of EntryCompletionStatus).
    std::vector<EntryCompletionStatus> h_status(e->entry_count);
    cudaError_t ce = cudaMemcpy(h_status.data(), e->d_status,
                                 e->entry_count * sizeof(EntryCompletionStatus),
                                 cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    out_results.resize(e->entry_count);
    for (std::uint32_t i = 0; i < e->entry_count; ++i) {
        out_results[i] = h_status[i].result;
    }
    return true;
}

std::uint32_t LocalNvmeDataPath::test_cq_poll_budget() const {
    return cq_poll_budget_;
}

std::uint64_t LocalNvmeDataPath::test_submit_call_count() const {
    return test_submit_call_count_;
}

std::uint64_t LocalNvmeDataPath::test_kernel_launch_count() const {
    return test_kernel_launch_count_;
}

std::uint64_t LocalNvmeDataPath::test_last_prebuilt_entry_count() const {
    return test_last_prebuilt_entry_count_;
}

std::uint64_t LocalNvmeDataPath::test_last_dynamic_entry_count() const {
    return test_last_dynamic_entry_count_;
}

void LocalNvmeDataPath::test_reset_submit_counters() {
    test_submit_call_count_ = 0;
    test_kernel_launch_count_ = 0;
}

void LocalNvmeDataPath::test_set_inject_resolve_lba_failure(bool v) {
    test_inject_resolve_lba_failure_ = v;
}

bool LocalNvmeDataPath::test_get_inject_resolve_lba_failure() const {
    return test_inject_resolve_lba_failure_;
}

// ---- test-only: NVMe CQ error injection seam (FIX 2) ----

void LocalNvmeDataPath::test_set_inject_nvme_error(bool v) {
    test_inject_nvme_error_ = v;
}

bool LocalNvmeDataPath::test_get_inject_nvme_error() const {
    return test_inject_nvme_error_;
}

// ---- test-only: progress() query-error injection seam (FIX 3) ----

void LocalNvmeDataPath::test_set_inject_query_error(bool v) {
    test_inject_query_error_ = v;
}

bool LocalNvmeDataPath::test_get_inject_query_error() const {
    return test_inject_query_error_;
}

// ---- test-only: has_timeout observability (FIX 4) ----

bool LocalNvmeDataPath::test_op_has_timeout(DataPathOp op) const {
    const auto* e = find_op_(op);
    return e && e->has_timeout;
}

// -------------------------------------------------------------------------
// private: aggregate per-entry completion status
// -------------------------------------------------------------------------

void LocalNvmeDataPath::aggregate_completion_status_(OpEntry& op) {
    op.completion_detail = IoCompletionDetail{};
    if (!op.d_status || op.entry_count == 0) {
        // No status array (shouldn't happen for real ops).
        op.state = OpState::COMPLETED;
        op.status = Status::Ok();
        op.bytes_transferred = op.total_bytes;
        op.completion_detail.confirmed_bytes = op.total_bytes;
        return;
    }

    // PROGRESS HARVEST D2H: the completion event has already signaled
    // (kernel finished), so synchronous cudaMemcpy is safe here.  The
    // default-stream sync it implies will not block because the caller's
    // stream is already done (the event was on the caller's stream).
    // This D2H is in the progress() harvest path, NOT in the submit()
    // production path — it does not affect stream-ordered async submit.
    std::vector<EntryCompletionStatus> h_status(op.entry_count);
    cudaError_t ce = cudaMemcpy(h_status.data(), op.d_status,
                                 op.entry_count * sizeof(EntryCompletionStatus),
                                 cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
        cudaGetLastError();
        op.state = OpState::FAILED;
        op.status = Status(StatusCode::DEVICE_ERROR,
                           "D2H completion status failed: " +
                           std::string(cudaGetErrorString(ce)));
        op.bytes_transferred = 0;
        set_completion_failure(op.completion_detail,
                               IoFailureKind::STATUS_D2H_ERROR, 0);
        return;
    }

    // Aggregate: all entries must succeed for COMPLETED.
    std::uint64_t confirmed_bytes = 0;
    bool any_failed = false;
    std::string first_error;
    std::uint32_t first_failed_entry = UINT32_MAX;
    std::uint32_t first_failure_result = 0;
    std::uint32_t first_raw_cq_status = 0;

    // D2H entries to get per-entry byte lengths.
    std::vector<DeviceSubmitEntry> h_entries(op.entry_count);
    ce = cudaMemcpy(h_entries.data(), op.d_entries,
                     op.entry_count * sizeof(DeviceSubmitEntry),
                     cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
        cudaGetLastError();
        // The device entry array is diagnostic-only here: host-side
        // entry_lengths already carries the exact byte count for each entry.
        // Keep confirmed_bytes accurate even when this optional D2H fails.
        std::uint64_t fallback_confirmed_bytes = 0;
        for (std::uint32_t i = 0; i < op.entry_count; ++i) {
            if (h_status[i].result == 0) {
                fallback_confirmed_bytes += op.entry_lengths[i];
            } else {
                any_failed = true;
                if (first_failed_entry == UINT32_MAX) {
                    first_failed_entry = i;
                    first_failure_result = h_status[i].result;
                    first_raw_cq_status = h_status[i].nvme_status_dword3;
                }
                if (h_status[i].result == 2) {
                    op.has_timeout = true;  // FIX 4
                }
                if (first_error.empty()) {
                    first_error = "entry " + std::to_string(i) +
                                 ": error " + std::to_string(h_status[i].result);
                }
            }
        }
        if (any_failed) {
            op.state = OpState::FAILED;
            op.status = Status(StatusCode::DEVICE_ERROR, first_error);
            op.bytes_transferred = fallback_confirmed_bytes;
            set_completion_failure(
                op.completion_detail,
                completion_failure_kind(first_failure_result),
                fallback_confirmed_bytes,
                op.has_timeout, first_failed_entry, first_raw_cq_status);
        } else {
            op.state = OpState::COMPLETED;
            op.status = Status::Ok();
            op.bytes_transferred = op.total_bytes;
            op.completion_detail.confirmed_bytes = op.total_bytes;
        }
        return;
    }

    for (std::uint32_t i = 0; i < op.entry_count; ++i) {
        const auto& s = h_status[i];
        if (s.result == 0) {
            // Success — count this entry's bytes.
            confirmed_bytes += op.entry_lengths[i];
        } else {
            any_failed = true;
            if (first_failed_entry == UINT32_MAX) {
                first_failed_entry = i;
                first_failure_result = s.result;
                first_raw_cq_status = s.nvme_status_dword3;
            }
            if (s.result == 2) {
                // FIX 4: a CQ timeout means the NVMe command may still be in
                // the controller queue; mark the op so release()/shutdown()
                // retain the PRP-list DMA mapping (conservative leak).
                op.has_timeout = true;
            }
            if (first_error.empty()) {
                switch (s.result) {
                    case 1:
                        first_error = "entry " + std::to_string(i) +
                                      ": resolve_lba failed";
                        break;
                    case 2:
                        first_error = "entry " + std::to_string(i) +
                                      ": CQ poll timeout";
                        break;
                    case 3:
                        first_error = "entry " + std::to_string(i) +
                                      ": NVMe CQ error (dword3=0x" +
                                      [&](){
                                          char buf[32];
                                          std::snprintf(buf, sizeof(buf),
                                                         "%x", s.nvme_status_dword3);
                                          return std::string(buf);
                                      }() + ")";
                        break;
                    case kEntryCompletionPending:
                        first_error = "entry " + std::to_string(i) +
                                      ": device submit was not executed";
                        break;
                    default:
                        first_error = "entry " + std::to_string(i) +
                                      ": unknown error " +
                                      std::to_string(s.result);
                        break;
                }
            }
        }
    }

    if (any_failed) {
        op.state = OpState::FAILED;
        op.status = Status(StatusCode::DEVICE_ERROR, first_error);
        op.bytes_transferred = confirmed_bytes;  // only confirmed bytes
        set_completion_failure(
            op.completion_detail,
            completion_failure_kind(first_failure_result), confirmed_bytes,
            op.has_timeout, first_failed_entry, first_raw_cq_status);
    } else {
        op.state = OpState::COMPLETED;
        op.status = Status::Ok();
        op.bytes_transferred = confirmed_bytes;
        op.completion_detail.confirmed_bytes = confirmed_bytes;
    }
}

// -------------------------------------------------------------------------
// private helpers
// -------------------------------------------------------------------------

const LocalNvmeTargetState* LocalNvmeDataPath::find_(
    DataPathTarget target) const {
    if (!target.valid()) {
        return nullptr;
    }
    auto it = targets_.find(target.token());
    if (it == targets_.end()) {
        return nullptr;
    }
    if (it->second.generation != target.generation()) {
        return nullptr;
    }
    return &it->second;
}

const LocalNvmeDataPath::MemReg* LocalNvmeDataPath::find_mem_(
    DataPathMemory memory) const {
    if (!memory.valid()) {
        return nullptr;
    }
    auto it = mem_regs_.find(memory.token());
    if (it == mem_regs_.end()) {
        return nullptr;
    }
    if (it->second.generation != memory.generation()) {
        return nullptr;
    }
    return &it->second;
}

LocalNvmeDataPath::MemReg* LocalNvmeDataPath::find_mem_(
    DataPathMemory memory) {
    if (!memory.valid()) return nullptr;
    auto it = mem_regs_.find(memory.token());
    if (it == mem_regs_.end()) return nullptr;
    if (it->second.generation != memory.generation()) return nullptr;
    return &it->second;
}

// -------------------------------------------------------------------------
// private: op helpers
// -------------------------------------------------------------------------

const LocalNvmeDataPath::OpEntry* LocalNvmeDataPath::find_op_(
    DataPathOp op) const {
    if (!op.valid()) return nullptr;
    auto it = ops_.find(op.token());
    if (it == ops_.end()) return nullptr;
    if (it->second.op_generation != op.generation()) return nullptr;
    return &it->second;
}

LocalNvmeDataPath::OpEntry* LocalNvmeDataPath::find_op_(DataPathOp op) {
    if (!op.valid()) return nullptr;
    auto it = ops_.find(op.token());
    if (it == ops_.end()) return nullptr;
    if (it->second.op_generation != op.generation()) return nullptr;
    return &it->second;
}

bool LocalNvmeDataPath::target_has_inflight_ops_(std::uint64_t token) const {
    for (const auto& [tok, op] : ops_) {
        if (op.state != OpState::IN_FLIGHT) continue;
        for (auto t : op.target_tokens)
            if (t == token) return true;
    }
    return false;
}

bool LocalNvmeDataPath::memory_has_inflight_ops_(std::uint64_t token) const {
    for (const auto& [tok, op] : ops_) {
        if (op.state != OpState::IN_FLIGHT) continue;
        for (auto t : op.memory_tokens)
            if (t == token) return true;
    }
    return false;
}

// -------------------------------------------------------------------------
// test-only: MetadataArena accessors
// -------------------------------------------------------------------------

std::uint32_t LocalNvmeDataPath::test_arena_capacity() const {
    return arena_.capacity();
}

std::uint32_t LocalNvmeDataPath::test_arena_available() const {
    return arena_.available();
}

MetadataArena::AllocCounts LocalNvmeDataPath::test_arena_alloc_counts() const {
    return arena_.alloc_counts();
}

void LocalNvmeDataPath::test_arena_reset_alloc_counts() {
    arena_.reset_alloc_counts();
}

// -------------------------------------------------------------------------
// test-only: cache accessors
// -------------------------------------------------------------------------

bool LocalNvmeDataPath::test_handle_cache_enabled() const {
    return handle_cache_.enabled();
}

HandleWorkspaceCache::Stats LocalNvmeDataPath::test_handle_cache_stats() const {
    return handle_cache_.stats();
}

bool LocalNvmeDataPath::test_prp_cache_enabled() const {
    return prp_cache_.enabled();
}

PrpPageCache::Stats LocalNvmeDataPath::test_prp_cache_stats() const {
    return prp_cache_.stats();
}

} // namespace tutti::data_paths::local_nvme

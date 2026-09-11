// tutti/data_paths/striped_local_nvme/striped_data_path.cpp
//
// StripedDataPath implementation — single-kernel fused submission across
// N NVMe devices.

#include "tutti/data_paths/striped_local_nvme/striped_data_path.h"

#include "tutti/data_paths/local_nvme/io/device_target.h"
#include "tutti/data_paths/local_nvme/io/nvme_submit_primitives.cuh"
#include "tutti/data_paths/local_nvme/io/submit_one.cuh"
#include "tutti/data_paths/local_nvme/io/prp_builder.h"
#include "tutti/data_paths/local_nvme/metadata/prp_page_cache.h"
#include "tutti/data_paths/local_nvme/io/nvme_queue_group.h"
#include "tutti/data_paths/striped_local_nvme/fused_submit_kernel.cuh"
#include "tutti/bindings/ext4_local_nvme/binding.h"

#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvtx3/nvToolsExt.h>

#include <tutti/cuda_like.h>
#include <tutti/accelerator_device_guard.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace tutti::data_paths::striped_local_nvme {

namespace {

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
        return {"tutti.striped_nvme.io_kernel|op=read", 0xFF00B8D9u};
    }
    if (has_write && !has_read) {
        return {"tutti.striped_nvme.io_kernel|op=write", 0xFFFF8C00u};
    }
    return {"tutti.striped_nvme.io_kernel|op=mixed", 0xFF8D99A6u};
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

using namespace tutti::binding::ext4_local_nvme;
using namespace tutti::binding::striped_local_nvme;
using tutti::data_paths::local_nvme::DeviceTargetHandle;
using tutti::data_paths::local_nvme::DeviceLbaExtent;
using tutti::data_paths::local_nvme::EntryCompletionStatus;
using tutti::data_paths::local_nvme::NvmeQueueGroup;
using tutti::data_paths::local_nvme::PrpKind;
using tutti::data_paths::local_nvme::classify_prp;
using tutti::data_paths::local_nvme::fill_prp_list_page;
using tutti::data_paths::local_nvme::kDeviceTargetInlineExtents;
using tutti::data_paths::local_nvme::build_device_target;
using tutti::data_paths::local_nvme::free_device_target;

// =========================================================================
// Constructor / Destructor
// =========================================================================

StripedDataPath::StripedDataPath(std::vector<DeviceDescriptor> devices,
                                 std::uint32_t cuda_device,
                                 std::uint64_t mdts_override,
                                 std::uint32_t cq_poll_budget,
                                 std::uint32_t max_batch_entries,
                                 std::uint32_t max_in_flight_operations,
                                 std::uint32_t handle_cache_capacity,
                                 std::uint32_t prp_cache_capacity,
                                 std::uint32_t threads_per_block)
    : device_descs_(std::move(devices)),
      cuda_device_(cuda_device),
      mdts_override_(mdts_override),
      cq_poll_budget_(cq_poll_budget == 0 ? 2000000 : cq_poll_budget),
      max_batch_entries_(max_batch_entries == 0 ? 256 : max_batch_entries),
      max_in_flight_operations_(max_in_flight_operations == 0
                                 ? 16 : max_in_flight_operations),
      threads_per_block_(threads_per_block),
      handle_cache_capacity_(handle_cache_capacity),
      prp_cache_capacity_(prp_cache_capacity) {

    caps_.name = "striped-local-nvme";
    caps_.source_api_version = 1;
    caps_.supports_host_execution = false;
    caps_.supports_device_execution = true;
    caps_.supports_host_memory = false;
    caps_.supports_device_memory = true;
    caps_.supports_direct = true;
    caps_.supports_staged = false;
    caps_.supports_read = true;
    caps_.supports_write = true;
    caps_.target_alignment_bytes = 4096;
    caps_.memory_alignment_bytes = 65536;  // 64 KiB (snvme pinning)
    caps_.length_alignment_bytes = 4096;
    // Preliminary caps; initialize() updates with real hardware values.
    caps_.max_single_io_bytes = static_cast<std::uint64_t>(max_batch_entries_) * 131072;
    caps_.max_batch_requests = max_batch_entries_;
    caps_.max_batch_bytes = caps_.max_single_io_bytes;
    caps_.max_in_flight_operations = max_in_flight_operations_;
    caps_.supports_scatter_gather = false;
    caps_.max_scatter_gather_entries = 0;
    caps_.registration_scope = RegistrationScope::PER_TARGET;
    caps_.progress_model = ProgressModel::HOST_POLL;
    caps_.device_completion_fence_on_caller_stream = true;
    caps_.device_execution_autonomous = true;
    caps_.supports_multi_stream = true;
    caps_.max_concurrent_streams = 2;
    caps_.max_concurrent_operations = max_in_flight_operations_;
    caps_.supports_multi_gpu = false;
    caps_.supports_cross_device = false;
    caps_.bound_accel_id = static_cast<std::int32_t>(cuda_device_);
    caps_.optional_target_features = {};
}

StripedDataPath::~StripedDataPath() {
    if (initialized_) {
        const Status stopped = shutdown(1000000000ULL);
        if (!stopped.ok()) {
            // Destruction cannot report TIMEOUT to a caller. Detach all host
            // PRP mappings/backing so member destructors cannot unmap memory
            // a controller may still fetch after an unresolved command.
            for (auto& cache : prp_caches_) {
                if (cache) cache->shutdown(/*retain=*/true);
            }
            for (auto& pool : prp_buf_pools_) {
                if (pool) pool->shutdown(/*retain=*/true);
            }
            prp_caches_.clear();
            prp_buf_pools_.clear();
        }
    }
}

// =========================================================================
// capabilities
// =========================================================================

const DataPathCapabilities& StripedDataPath::capabilities() const {
    return caps_;
}

// =========================================================================
// initialize — attach N controllers, create N queue groups, arena init
// =========================================================================

Status StripedDataPath::initialize(const DataPathConfig& config,
                                   ResourceProvider& resources) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    Status result = initialize_impl_(config, resources);
    Status restored = guard.restore();
    return restored.ok() ? result : restored;
}

Status StripedDataPath::shutdown(std::uint64_t timeout_ns) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    Status result = shutdown_impl_(timeout_ns);
    Status restored = guard.restore();
    return restored.ok() ? result : restored;
}

Result<DataPathTarget> StripedDataPath::open(const ResolvedTarget& target) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return Result<DataPathTarget>::Failure(guard.status());
    auto result = open_impl_(target);
    Status restored = guard.restore();
    return restored.ok() ? result
                         : Result<DataPathTarget>::Failure(std::move(restored));
}

Status StripedDataPath::close(DataPathTarget target) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    Status result = close_impl_(target);
    Status restored = guard.restore();
    return restored.ok() ? result : restored;
}

Result<RegistrationDomainKey> StripedDataPath::registration_domain(
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

Result<DataPathMemory> StripedDataPath::register_memory(
    const DataPathMemoryView& view,
    const RegistrationDomainKey& domain) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return Result<DataPathMemory>::Failure(guard.status());
    auto result = register_memory_impl_(view, domain);
    Status restored = guard.restore();
    return restored.ok() ? result
                         : Result<DataPathMemory>::Failure(std::move(restored));
}

Status StripedDataPath::unregister_memory(DataPathMemory memory) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    Status result = unregister_memory_impl_(memory);
    Status restored = guard.restore();
    return restored.ok() ? result : restored;
}

SubmitOutcome StripedDataPath::submit(const DataPathRequest* requests,
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

Result<ProgressResult> StripedDataPath::progress(ProgressBudget budget) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return Result<ProgressResult>::Failure(guard.status());
    auto result = progress_impl_(budget);
    Status restored = guard.restore();
    return restored.ok() ? result
                         : Result<ProgressResult>::Failure(std::move(restored));
}

Result<DataPathSnapshot> StripedDataPath::query(DataPathOp op) const {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return Result<DataPathSnapshot>::Failure(guard.status());
    auto result = query_impl_(op);
    Status restored = guard.restore();
    return restored.ok() ? result
                         : Result<DataPathSnapshot>::Failure(std::move(restored));
}

Status StripedDataPath::release(DataPathOp op) {
    DeviceGuard guard(static_cast<std::int32_t>(cuda_device_));
    if (!guard.ok()) return guard.status();
    Status result = release_impl_(op);
    Status restored = guard.restore();
    return restored.ok() ? result : restored;
}

Status StripedDataPath::initialize_impl_(const DataPathConfig& config,
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
    if (device_descs_.empty()) {
        return Status(StatusCode::INVALID_ARGUMENT, "no devices configured");
    }

    if (!config.name.empty()) {
        caps_.name = config.name;
    }

    // All shard block sizes must be uniform (documented assumption).
    block_size_ = device_descs_[0].block_size;
    for (const auto& d : device_descs_) {
        if (d.block_size != block_size_) {
            return Status(StatusCode::INVALID_ARGUMENT,
                         "all devices must share the same block_size");
        }
        if (d.block_size == 0) {
            return Status(StatusCode::INVALID_ARGUMENT,
                         "device descriptor missing block_size");
        }
    }

    devices_.reserve(device_descs_.size());

    auto rollback_devices = [&]() {
        for (auto it = devices_.rbegin(); it != devices_.rend(); ++it) {
            it->queue_group.reset();
            if (it->ctrl) { nvm_ctrl_free_client(it->ctrl); it->ctrl = nullptr; }
        }
        devices_.clear();
    };

    for (std::uint32_t i = 0; i < device_descs_.size(); ++i) {
        const auto& desc = device_descs_[i];
        DeviceSlot slot;
        slot.desc = desc;

        int rc = nvm_ctrl_attach_client(&slot.ctrl, desc.snvme_dev_path.c_str());
        if (rc != 0 || slot.ctrl == nullptr) {
            rollback_devices();
            return Status(StatusCode::NOT_READY,
                         "nvm_ctrl_attach_client(" + desc.snvme_dev_path +
                         ") failed: rc " + std::to_string(rc));
        }

        struct disk dev_info;
        std::memset(&dev_info, 0, sizeof(dev_info));
        rc = ioctl_get_dev_info(slot.ctrl, &dev_info);
        if (rc != 0) {
            nvm_ctrl_free_client(slot.ctrl);
            slot.ctrl = nullptr;
            rollback_devices();
            return Status(StatusCode::NOT_READY,
                         "ioctl_get_dev_info failed for device " +
                         std::to_string(i) + ": rc " + std::to_string(rc));
        }

        slot.hardware_mdts = dev_info.max_data_size;
        slot.page_size = static_cast<std::uint64_t>(slot.ctrl->page_size);

        // SQ/CQ ring depth comes from the kernel (NVM_GET_DEV_INFO,
        // mirrored into slot.ctrl->q_depth by ioctl_get_dev_info) — never
        // user-specified: the kernel builds user IOQ rings with
        // dev->q_depth unconditionally, so any other userspace ring size
        // silently desyncs SQ wrap / CQ phase (observed as CQ poll
        // timeouts once a queue wraps past the smaller depth).

        struct disk disk_info = dev_info;
        disk_info.ns_id = desc.namespace_id;
        disk_info.block_size = desc.block_size;
        std::string dname = desc.snvme_dev_path;
        if (dname.rfind("/dev/", 0) == 0) dname = dname.substr(5);
        dname += "n" + std::to_string(desc.namespace_id);
        std::strncpy(disk_info.disk_name, dname.c_str(),
                    sizeof(disk_info.disk_name) - 1);

        try {
            slot.queue_group = std::make_unique<NvmeQueueGroup>(
                slot.ctrl, disk_info, desc.namespace_id, desc.cuda_device,
                desc.num_user_queues, slot.ctrl->q_depth);
        } catch (const std::runtime_error& e) {
            nvm_ctrl_free_client(slot.ctrl);
            slot.ctrl = nullptr;
            rollback_devices();
            return Status(StatusCode::NOT_READY,
                         "queue group creation failed for device " +
                         std::to_string(i) + ": " + e.what());
        }
        if (threads_per_block_ > slot.queue_group->n_qps()) {
            // Round-robin sharing: QueueAcquireHelper::acquire_queue()
            // maps threads onto queues with % num_queues, and the
            // nvm_parallel_queue (atomic cid tickets, per-slot locks,
            // atomic head/tail advance) supports multiple concurrent
            // submitters per queue.  Warn instead of failing so a
            // daemon-side quota clamp (e.g. max_per_client) doesn't
            // kill the run.
            std::fprintf(
                stderr,
                "[striped-local-nvme] warning: threads_per_block (%u) > "
                "granted queues (%u) for device %u; threads will share "
                "queues round-robin\n",
                threads_per_block_, slot.queue_group->n_qps(), i);
        }

        devices_.push_back(std::move(slot));
    }

    // The public capability is the conservative minimum. Submission and
    // prebuild use each selected controller's own driver-reported MDTS.
    std::uint64_t min_mdts = UINT64_MAX;
    for (std::uint32_t i = 0; i < devices_.size(); ++i) {
        const std::uint64_t mdts = device_effective_mdts_(i);
        if (mdts > 0 && mdts < min_mdts) {
            min_mdts = mdts;
        }
    }
    if (min_mdts == UINT64_MAX || min_mdts == 0) {
        rollback_devices();
        return Status(StatusCode::NOT_READY, "no device reported a usable MDTS");
    }
    effective_mdts_bytes_ = min_mdts;
    if (effective_mdts_bytes_ % block_size_ != 0) {
        rollback_devices();
        return Status(StatusCode::INVALID_ARGUMENT,
                      "effective MDTS not a block-size multiple");
    }

    // PRP-list page capacity check (same formula as LocalNvmeDataPath).
    const std::uint64_t page_size = devices_[0].page_size;
    for (const auto& d : devices_) {
        if (d.page_size != page_size) {
            rollback_devices();
            return Status(StatusCode::INVALID_ARGUMENT,
                         "all devices must share the same controller page_size");
        }
    }
    std::uint64_t prp_list_page_capacity = page_size / sizeof(std::uint64_t) + 1;
    for (std::uint32_t i = 0; i < devices_.size(); ++i) {
        const std::uint64_t mdts = device_effective_mdts_(i);
        if (mdts == 0 || mdts % block_size_ != 0 || mdts % page_size != 0) {
            rollback_devices();
            return Status(StatusCode::INVALID_ARGUMENT,
                          "controller MDTS is not block/page aligned");
        }
        if (mdts / page_size > prp_list_page_capacity) {
            rollback_devices();
            return Status(StatusCode::INVALID_ARGUMENT,
                          "controller MDTS exceeds one PRP-list page capacity");
        }
    }

    max_request_bytes_ = static_cast<std::uint64_t>(max_batch_entries_) *
                         effective_mdts_bytes_;
    caps_.max_single_io_bytes = max_request_bytes_;
    caps_.max_batch_bytes = max_request_bytes_;
    caps_.max_batch_requests = max_batch_entries_;
    caps_.max_in_flight_operations = max_in_flight_operations_;
    caps_.max_concurrent_operations = max_in_flight_operations_;
    caps_.target_alignment_bytes = block_size_;
    caps_.memory_alignment_bytes = block_size_;
    caps_.length_alignment_bytes = block_size_;

    // Arena init: dev_table_capacity_per_slot = N (one submit's device
    // table spans exactly the striped target's N shards).
    std::vector<nvm_ctrl_t*> ctrls;
    ctrls.reserve(devices_.size());
    for (auto& d : devices_) ctrls.push_back(d.ctrl);

    StripedArena::Config arena_cfg;
    arena_cfg.num_slots = max_in_flight_operations_ * 2;
    arena_cfg.max_entries_per_slot = max_batch_entries_;
    arena_cfg.page_size = static_cast<std::uint32_t>(page_size);
    arena_cfg.cuda_device = cuda_device_;
    // Round 16 S4: device table supports M targets × N shards.
    // Capacity must cover the LARGEST legal batch: up to max_batch_entries
    // distinct targets, each contributing N shard pointers (e.g. a 921-
    // target read batch on 4 devices needs 3684 slots > the old 2048
    // hard-coded cap, which rejected whole 256K-context layers).
    arena_cfg.dev_table_capacity_per_slot =
        std::max(static_cast<std::uint32_t>(2048),
                 max_batch_entries_ *
                     static_cast<std::uint32_t>(devices_.size()));
    if (!arena_.init(arena_cfg, ctrls)) {
        rollback_devices();
        return Status(StatusCode::NOT_READY, "StripedArena init failed");
    }

    // Host-pinned PRP miss pools are per controller because each IOVA belongs
    // to that controller/IOMMU domain. They grow on demand and never allocate
    // GPU PRP backing.
    prp_buf_pools_.resize(devices_.size());
    for (std::size_t i = 0; i < devices_.size(); ++i) {
        prp_buf_pools_[i] =
            std::make_unique<tutti::data_paths::local_nvme::PrpBufPool>();
        prp_buf_pools_[i]->init(devices_[i].ctrl, page_size);
    }

    // Round 16 S5: initialize per-device PRP cache if enabled.
    if (prp_cache_capacity_ > 0) {
        prp_caches_.resize(devices_.size());
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            prp_caches_[i] = std::make_unique<tutti::data_paths::local_nvme::PrpPageCache>();
            tutti::data_paths::local_nvme::PrpPageCache::Config pcfg;
            pcfg.capacity = prp_cache_capacity_;
            pcfg.page_size = static_cast<std::uint32_t>(page_size);
            pcfg.cuda_device = cuda_device_;
            if (!prp_caches_[i]->init(pcfg, devices_[i].ctrl)) {
                return Status(StatusCode::NOT_READY, "PrpPageCache init failed");
            }
        }
    }

    initialized_ = true;
    return Status::Ok();
}

// =========================================================================
// shutdown — release arena, all N devices
// =========================================================================

Status StripedDataPath::shutdown_impl_(std::uint64_t timeout_ns) {
    if (!initialized_) return Status::Ok();

    auto has_inflight = [&]() -> bool {
        for (const auto& [tok, op] : ops_) {
            if (op.state == OpState::IN_FLIGHT) return true;
        }
        return false;
    };

    if (has_inflight()) {
        if (timeout_ns == 0) {
            return Status(StatusCode::TIMEOUT,
                          "shutdown: in-flight operations remain");
        }
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::nanoseconds(timeout_ns);
        while (std::chrono::steady_clock::now() < deadline) {
            auto remaining_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                deadline - std::chrono::steady_clock::now()).count();
            if (remaining_ns <= 0) break;
            ProgressBudget pb{max_in_flight_operations_,
                              static_cast<std::uint64_t>(remaining_ns)};
            auto pr = progress(pb);
            if (!pr.ok()) break;
            if (!has_inflight()) break;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        if (has_inflight()) {
            return Status(StatusCode::TIMEOUT,
                          "shutdown: drain timeout, in-flight ops remain");
        }
    }

    bool any_timeout = timeout_prp_retained_;
    for (const auto& [tok, op] : ops_) {
        if (op.has_timeout) { any_timeout = true; break; }
    }
    arena_.shutdown();
    for (auto& pc : prp_caches_) if (pc) pc->shutdown(any_timeout);
    for (auto& pool : prp_buf_pools_) if (pool) pool->shutdown(any_timeout);
    prp_caches_.clear();
    prp_buf_pools_.clear();
    ops_.clear();

    for (auto& [tok, tgt] : targets_) {
        for (std::uint32_t s = 0; s < tgt.num_shards; ++s) {
            if (tgt.dev_handles[s]) {
                free_device_target(tgt.dev_handles[s], tgt.overflow_allocs[s],
                                   cuda_device_);
            }
        }
    }
    targets_.clear();

    for (auto& [tok, mem] : memory_regs_) {
        for (auto* dma : mem.dmas) {
            if (dma) nvm_dma_unmap(dma);
        }
    }
    memory_regs_.clear();

    for (auto it = devices_.rbegin(); it != devices_.rend(); ++it) {
        it->queue_group.reset();
        if (it->ctrl) {
            nvm_ctrl_free_client(it->ctrl);
            it->ctrl = nullptr;
        }
    }
    devices_.clear();

    initialized_ = false;
    timeout_prp_retained_ = false;
    return Status::Ok();
}

// =========================================================================
// open — extract StripedLocalNvmePayload, build N DeviceTargetHandles
// =========================================================================

Result<DataPathTarget> StripedDataPath::open_impl_(const ResolvedTarget& target) {
    if (!initialized_) {
        return Result<DataPathTarget>::Failure(
            Status(StatusCode::NOT_READY, "not initialized"));
    }

    auto payload_result = tutti::binding::striped_local_nvme::view_payload(target);
    if (!payload_result.ok()) {
        return Result<DataPathTarget>::Failure(
            Status(payload_result.status().code(),
                   "open: payload view failed: " +
                   payload_result.status().message()));
    }
    const StripedLocalNvmePayload* p = payload_result.value();

    if (p->num_shards() != devices_.size()) {
        return Result<DataPathTarget>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "payload num_shards=" + std::to_string(p->num_shards()) +
                   " != devices=" + std::to_string(devices_.size())));
    }

    StripedTarget tgt;
    tgt.num_shards = p->num_shards();
    tgt.stripe_unit = p->stripe_unit();
    tgt.shard_rotation = p->shard_rotation();  // Round 16 S7
    tgt.logical_size = p->logical_size();
    tgt.dev_handles.assign(tgt.num_shards, nullptr);
    tgt.overflow_allocs.assign(tgt.num_shards, nullptr);
    tgt.shard_extents.resize(tgt.num_shards);

    for (std::uint32_t s = 0; s < tgt.num_shards; ++s) {
        if (!build_shard_handle_(s, p->shards()[s], tgt)) {
            for (std::uint32_t j = 0; j < s; ++j) {
                if (tgt.dev_handles[j]) {
                    free_device_target(tgt.dev_handles[j], tgt.overflow_allocs[j],
                                       cuda_device_);
                }
            }
            return Result<DataPathTarget>::Failure(
                Status(StatusCode::DEVICE_ERROR,
                       "open: failed to build handle for shard " +
                       std::to_string(s)));
        }
    }

    std::uint64_t tok = next_target_++;
    tgt.generation = 1;
    tgt.domain_key = "striped-local-nvme:" + std::to_string(tok);
    targets_[tok] = std::move(tgt);

    return Result<DataPathTarget>::Success(
        detail::SpiIdentityMint::mint<detail::DataPathTargetTag>(tok, 1));
}

bool StripedDataPath::build_shard_handle_(
    std::uint32_t dev_idx,
    const ResolvedTarget& shard_target,
    StripedTarget& out) {

    auto ep = tutti::binding::ext4_local_nvme::view_payload(shard_target);
    if (!ep.ok()) return false;
    const Ext4LocalNvmePayload* ext = ep.value();

    const auto& ns = ext->namespace_identity();
    const auto& src_extents = ext->extents();
    if (ns.block_size == 0) return false;

    DeviceSlot& slot = devices_[dev_idx];
    if ((!slot.desc.controller_pci_addr.empty() &&
         ns.controller_pci_addr != slot.desc.controller_pci_addr) ||
        ns.namespace_id != slot.desc.namespace_id ||
        ns.block_size != slot.desc.block_size) {
        return false;
    }
    if (!slot.queue_group || slot.queue_group->d_qps() == nullptr) return false;

    std::uint32_t bs = ns.block_size;
    std::uint32_t bs_log = 0;
    while ((1u << bs_log) < bs) ++bs_log;
    if ((1u << bs_log) != bs) return false;

    DeviceTargetHandle tmpl;
    std::memset(&tmpl, 0, sizeof(tmpl));
    tmpl.file_id = dev_idx;
    tmpl.logical_size_bytes = shard_target.logical_size();
    tmpl.header_bytes = 0;
    tmpl.nvme_block_size = bs;
    tmpl.nvme_block_size_log = bs_log;
    tmpl.namespace_id = ns.namespace_id;
    tmpl.num_extents = static_cast<std::uint32_t>(src_extents.size());
    tmpl.d_qps = slot.queue_group->d_qps();
    tmpl.num_d_qps = slot.queue_group->n_qps();
    tmpl.extents_overflow = nullptr;

    auto convert_extent = [&](const Extent& e) -> DeviceLbaExtent {
        DeviceLbaExtent d;
        d.start_lba = e.device_offset / bs;
        d.length_blocks = e.length / bs;
        return d;
    };

    std::uint32_t n_inline = std::min(tmpl.num_extents, kDeviceTargetInlineExtents);
    for (std::uint32_t i = 0; i < n_inline; ++i) {
        tmpl.extents[i] = convert_extent(src_extents[i]);
    }

    std::vector<DeviceLbaExtent> overflow_buf;
    const DeviceLbaExtent* overflow_ptr = nullptr;
    std::uint32_t n_overflow = 0;
    if (tmpl.num_extents > kDeviceTargetInlineExtents) {
        n_overflow = tmpl.num_extents - kDeviceTargetInlineExtents;
        overflow_buf.resize(n_overflow);
        for (std::uint32_t i = 0; i < n_overflow; ++i) {
            overflow_buf[i] = convert_extent(src_extents[kDeviceTargetInlineExtents + i]);
        }
        overflow_ptr = overflow_buf.data();
    }

    DeviceTargetHandle* dev_h = nullptr;
    void* dev_ov = nullptr;
    if (!build_device_target(tmpl, overflow_ptr, n_overflow,
                             cuda_device_, &dev_h, &dev_ov)) {
        return false;
    }

    out.dev_handles[dev_idx] = dev_h;
    out.overflow_allocs[dev_idx] = dev_ov;

    // Host-side byte extents for stripe-split boundary clamping.
    out.shard_extents[dev_idx].reserve(src_extents.size());
    for (const auto& e : src_extents) {
        out.shard_extents[dev_idx].push_back({e.logical_offset, e.length});
    }

    return true;
}

// =========================================================================
// close / registration_domain
// =========================================================================

Status StripedDataPath::close_impl_(DataPathTarget target) {
    if (!target.valid()) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "close: target identity is invalid");
    }
    auto it = targets_.find(target.token());
    if (it == targets_.end()) {
        return Status(StatusCode::NOT_FOUND, "close: target not found");
    }
    if (it->second.generation != target.generation()) {
        return Status(StatusCode::NOT_FOUND, "close: generation mismatch");
    }
    if (target_has_inflight_ops_(target.token())) {
        return Status(StatusCode::BUSY, "close: target has in-flight operations");
    }

    for (std::uint32_t s = 0; s < it->second.num_shards; ++s) {
        if (it->second.dev_handles[s]) {
            free_device_target(it->second.dev_handles[s],
                               it->second.overflow_allocs[s], cuda_device_);
        }
    }
    targets_.erase(it);
    return Status::Ok();
}

Result<RegistrationDomainKey> StripedDataPath::registration_domain_impl_(
    DataPathTarget target) const {
    const auto* tgt = find_target_(target);
    if (!tgt) {
        return Result<RegistrationDomainKey>::Failure(
            Status(StatusCode::NOT_FOUND, "registration_domain: target not found"));
    }
    return Result<RegistrationDomainKey>::Success(
        RegistrationDomainKey{tgt->domain_key});
}

// =========================================================================
// register_memory / unregister_memory — nvm_dma_map_data_device x N
// =========================================================================

Result<DataPathMemory> StripedDataPath::register_memory_impl_(
    const DataPathMemoryView& view,
    const RegistrationDomainKey& /*domain*/) {

    if (!initialized_) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::NOT_READY, "register_memory: not initialized"));
    }
    if (view.base == nullptr || view.size_bytes == 0) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::INVALID_ARGUMENT, "null/zero memory view"));
    }
    if (view.kind != DataPathMemoryKind::DEVICE) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::UNSUPPORTED, "only DEVICE memory supported"));
    }
    if (view.expected_accel_id >= 0 &&
        view.expected_accel_id != static_cast<std::int32_t>(cuda_device_)) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "memory accelerator does not match DataPath"));
    }
#if defined(TUTTI_USE_HOST)
    return Result<DataPathMemory>::Failure(
        Status(StatusCode::UNSUPPORTED,
               "HOST profile has no device memory"));
#elif defined(TUTTI_USE_CUDA) || defined(TUTTI_USE_MUSA) || defined(TUTTI_USE_MACA)
    cudaPointerAttributes attributes{};
    const cudaError_t pointer_error =
        cudaPointerGetAttributes(&attributes, view.base);
    if (pointer_error != cudaSuccess) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::DEVICE_ERROR,
                   "pointer ownership query failed: " +
                   std::string(cudaGetErrorString(pointer_error))));
    }
#if defined(TUTTI_USE_CUDA)
    if (attributes.type != cudaMemoryTypeDevice &&
        attributes.type != cudaMemoryTypeManaged) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "pointer is not device memory"));
    }
#else
    if (attributes.type != cudaMemoryTypeDevice) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::UNSUPPORTED,
                   "backend cannot verify pointer kind"));
    }
#endif
    if (attributes.device != static_cast<int>(cuda_device_)) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "pointer belongs to another accelerator"));
    }
#endif
    if ((reinterpret_cast<std::uintptr_t>(view.base) % 65536) != 0) {
        return Result<DataPathMemory>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "DEVICE view.base must be 64 KiB-aligned"));
    }

    StripedMemory mem;
    mem.base = view.base;
    mem.size = view.size_bytes;
    mem.dmas.assign(devices_.size(), nullptr);

    for (std::size_t i = 0; i < devices_.size(); ++i) {
        nvm_dma_t* dma = nullptr;
        int rc = nvm_dma_map_data_device(&dma, devices_[i].ctrl,
                                         view.base,
                                         static_cast<size_t>(view.size_bytes));
        if (rc != 0 || dma == nullptr) {
            for (std::size_t j = 0; j < i; ++j) {
                if (mem.dmas[j]) { nvm_dma_unmap(mem.dmas[j]); mem.dmas[j] = nullptr; }
            }
            return Result<DataPathMemory>::Failure(
                Status(StatusCode::DEVICE_ERROR,
                       "nvm_dma_map_data_device failed for device " +
                       std::to_string(i) + ": rc " + std::to_string(rc)));
        }
        mem.dmas[i] = dma;
    }

    std::uint64_t tok = next_memory_++;
    mem.generation = 1;

    // Round 16 S5 (V3): pre-build per-device AddressDescriptor[] if
    // io_granularity > 0 (legacy build_io_slice_table 9-stage path).
    if (view.io_granularity > 0) {
        std::string err_msg;
        if (!build_striped_prebuilt_(mem, view.io_granularity, err_msg)) {
            for (auto* dma : mem.dmas) if (dma) nvm_dma_unmap(dma);
            return Result<DataPathMemory>::Failure(
                Status(StatusCode::DEVICE_ERROR, err_msg));
        }
    }

    memory_regs_[tok] = std::move(mem);

    return Result<DataPathMemory>::Success(
        detail::SpiIdentityMint::mint<detail::DataPathMemoryTag>(tok, 1));
}

Status StripedDataPath::unregister_memory_impl_(DataPathMemory memory) {
    if (!memory.valid()) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "unregister_memory: memory identity is invalid");
    }
    auto it = memory_regs_.find(memory.token());
    if (it == memory_regs_.end()) {
        return Status(StatusCode::NOT_FOUND, "unregister_memory: not found");
    }
    if (it->second.generation != memory.generation()) {
        return Status(StatusCode::NOT_FOUND,
                      "unregister_memory: generation mismatch");
    }
    if (memory_has_inflight_ops_(memory.token())) {
        return Status(StatusCode::BUSY,
                      "unregister_memory: memory has in-flight operations");
    }
    destroy_striped_prebuilt_(it->second);
    for (auto* dma : it->second.dmas) {
        if (dma) nvm_dma_unmap(dma);
    }
    memory_regs_.erase(it);
    return Status::Ok();
}

// =========================================================================
// Round 16 S5 (V3): registration-time pre-built descriptors for striped.
// Builds per-device AddressDescriptor[] at register_memory time so submit
// can use pointer arithmetic (entry.prp_entry = d_descs[dev] + slice_idx)
// instead of per-submit PRP computation + H2D.
// =========================================================================
bool StripedDataPath::build_striped_prebuilt_(
    StripedMemory& mem, std::uint64_t io_granularity, std::string& status_msg) {

    const std::uint64_t page_size = devices_[0].page_size;
    const std::uint64_t bytes_per_slice = io_granularity;
    if (bytes_per_slice == 0 || bytes_per_slice % page_size != 0) {
        status_msg = "io_granularity must be a positive controller-page multiple";
        return false;
    }
    if (mem.size % bytes_per_slice != 0) {
        status_msg = "io_granularity does not evenly divide memory size";
        return false;
    }
    const std::uint64_t num_slices = mem.size / bytes_per_slice;

    mem.prebuilt.devices.assign(mem.dmas.size(), {});
    mem.prebuilt.bytes_per_slice = bytes_per_slice;
    mem.prebuilt.num_slices = num_slices;

    auto cleanup_descs = [&]() {
        for (auto& table : mem.prebuilt.devices) {
            if (table.d_descs) cudaFree(table.d_descs);
            table.d_descs = nullptr;
        }
    };

    for (std::size_t dev = 0; dev < mem.dmas.size(); ++dev) {
        nvm_dma_t* dma = mem.dmas[dev];
        if (!dma) {
            status_msg = "dma is null for device " + std::to_string(dev);
            cleanup_descs();
            return false;
        }
        auto& table = mem.prebuilt.devices[dev];
        table.mdts_bytes = device_effective_mdts_(static_cast<std::uint32_t>(dev));
        table.ios_per_slice =
            (bytes_per_slice + table.mdts_bytes - 1) / table.mdts_bytes;
        table.num_descs = num_slices * table.ios_per_slice;

        std::uint64_t num_prp_pages = 0;
        for (std::uint64_t sub = 0; sub < table.ios_per_slice; ++sub) {
            const std::uint64_t sub_offset = sub * table.mdts_bytes;
            const std::uint64_t sub_io = std::min(
                table.mdts_bytes, bytes_per_slice - sub_offset);
            if (sub_io / page_size > 2) num_prp_pages += num_slices;
        }
        if (num_prp_pages > 0) {
            table.prp_buf_ref = prp_buf_pools_[dev]->alloc_pages(num_prp_pages);
            if (!table.prp_buf_ref.valid) {
                status_msg = "striped prebuilt host PrpBufPool allocation failed";
                cleanup_descs();
                return false;
            }
        }
        table.num_prp_pages = num_prp_pages;

        std::vector<tutti::data_paths::local_nvme::AddressDescriptor> h_descs(
            table.num_descs);
        std::uint64_t prp_page_idx = 0;
        for (std::uint64_t s = 0; s < num_slices; ++s) {
            const std::uint64_t slice_offset = s * bytes_per_slice;
            for (std::uint64_t sub = 0; sub < table.ios_per_slice; ++sub) {
                const std::uint64_t sub_offset = sub * table.mdts_bytes;
                const std::uint64_t sub_io = std::min(
                    table.mdts_bytes, bytes_per_slice - sub_offset);
                const std::uint64_t start_page =
                    (slice_offset + sub_offset) / page_size;
                const std::uint64_t pages_per_io = sub_io / page_size;
                if (start_page + pages_per_io > dma->n_ioaddrs) {
                    status_msg = "striped prebuilt PRP page out of DMA range";
                    cleanup_descs();
                    return false;
                }
                auto& desc = h_descs[s * table.ios_per_slice + sub];
                desc.data_length = sub_io;
                desc.prp1 = dma->ioaddrs[start_page];
                if (pages_per_io == 1) {
                    desc.prp2 = 0;
                } else if (pages_per_io == 2) {
                    desc.prp2 = dma->ioaddrs[start_page + 1];
                } else {
                    const std::uint64_t prp_page =
                        table.prp_buf_ref.base_page + prp_page_idx++;
                    auto* host_page = reinterpret_cast<std::uint64_t*>(
                        static_cast<char*>(table.prp_buf_ref.segment->vaddr) +
                        prp_page * page_size);
                    fill_prp_list_page(
                        host_page, dma, static_cast<std::uint32_t>(start_page),
                        static_cast<std::uint32_t>(pages_per_io), page_size);
                    desc.prp2 = table.prp_buf_ref.segment->ioaddrs[prp_page];
                }
            }
        }

        void* d = nullptr;
        cudaError_t ce = cudaMalloc(
            &d, table.num_descs *
                    sizeof(tutti::data_paths::local_nvme::AddressDescriptor));
        if (ce != cudaSuccess) {
            status_msg = std::string("cudaMalloc d_descs failed: ") + cudaGetErrorString(ce);
            cleanup_descs();
            return false;
        }
        ce = cudaMemcpy(d, h_descs.data(),
                        table.num_descs *
                            sizeof(tutti::data_paths::local_nvme::AddressDescriptor),
                        cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            status_msg = std::string("cudaMemcpy d_descs failed: ") + cudaGetErrorString(ce);
            cudaFree(d);
            cleanup_descs();
            return false;
        }
        table.d_descs = d;
    }

    mem.prebuilt.valid = true;
    return true;
}

void StripedDataPath::destroy_striped_prebuilt_(StripedMemory& mem) {
    for (auto& table : mem.prebuilt.devices) {
        if (table.d_descs) cudaFree(table.d_descs);
        table.d_descs = nullptr;
    }
    mem.prebuilt.devices.clear();
    mem.prebuilt.valid = false;
}

std::uint64_t StripedDataPath::device_effective_mdts_(
    std::uint32_t device) const {
    if (device >= devices_.size()) return 0;
    const std::uint64_t hardware = devices_[device].hardware_mdts;
    return mdts_override_ > 0 ? std::min(mdts_override_, hardware) : hardware;
}

// =========================================================================
// submit — stripe split -> 1 H2D (entries + dev_table) -> 1 launch -> 1 event
// =========================================================================

SubmitOutcome StripedDataPath::submit_impl_(const DataPathRequest* requests,
                                            std::size_t count,
                                            const HostSubmitContext& ctx) {
    ++test_submit_call_count_;
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
    StatusCode first_rejected_code = StatusCode::OK;
    std::string first_rejected_msg;
    auto reject_one = [&](std::size_t i, StatusCode code, const std::string& msg) {
        outcome.initial_states[i].state = RequestState::REJECTED;
        outcome.initial_states[i].status = Status(code, msg);
        if (first_rejected_code == StatusCode::OK) {
            first_rejected_code = code;
            first_rejected_msg = msg;
        }
    };

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
    if (ctx.accel_id >= 0 &&
        ctx.accel_id != static_cast<std::int32_t>(cuda_device_)) {
        reject_all(StatusCode::INVALID_ARGUMENT,
                   "ctx.accel_id does not match this DataPath's CUDA device");
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
    if (stream_accel_id != static_cast<int>(cuda_device_)) {
        reject_all(StatusCode::INVALID_ARGUMENT,
                   "stream belongs to another accelerator");
        return outcome;
    }
#endif

    std::uint64_t in_flight_count = 0;
    for (const auto& [tok, op] : ops_) {
        if (op.state == OpState::IN_FLIGHT) ++in_flight_count;
    }
    if (in_flight_count >= max_in_flight_operations_) {
        reject_all(StatusCode::RESOURCE_EXHAUSTED,
                   "in-flight operation capacity exhausted");
        return outcome;
    }

    // Round 16 S4: multi-target batch support.  Device table now holds
    // M targets × N shards.  M is determined at submit time from the
    // distinct targets in the batch.  Each entry stores (target_idx, shard_idx)
    // so the kernel can index d_dev_table[target_idx * N + shard_idx].
    struct TargetInfo {
        const StripedTarget* tgt;
        std::uint64_t token;
        std::uint32_t index;  // 0-based index into device table
    };
    std::vector<TargetInfo> targets_in_batch;
    // token -> index map: the old double linear scan was O(count*targets)
    // (3.4M compares for a 1842-request / 921-target 256K-context batch).
    std::unordered_map<std::uint64_t, std::uint32_t> tgt_index_of;
    tgt_index_of.reserve(count * 2);
    std::vector<bool> rejected(count, false);
    for (std::size_t i = 0; i < count; ++i) {
        const auto* t = find_target_(requests[i].target);
        if (!t) {
            reject_one(i, StatusCode::NOT_FOUND, "target not found or closed");
            rejected[i] = true;
            continue;
        }
        auto [it, inserted] = tgt_index_of.emplace(
            requests[i].target.token(),
            static_cast<std::uint32_t>(targets_in_batch.size()));
        if (inserted) {
            targets_in_batch.push_back({t, requests[i].target.token(),
                                        it->second});
        }
    }
    if (targets_in_batch.empty()) {
        outcome.status = Status(StatusCode::NOT_FOUND, "no valid target in batch");
        return outcome;
    }
    const std::uint32_t n_targets = (std::uint32_t)targets_in_batch.size();
    const std::uint32_t total_dev_table = n_targets * static_cast<std::uint32_t>(devices_.size());

    const std::uint32_t page_size =
        static_cast<std::uint32_t>(devices_[0].page_size);

    struct ListInfo {
        std::uint32_t entry_idx;
        std::uint32_t start_page;
        std::uint32_t pages_in_io;
        std::uint32_t dev_idx;
        nvm_dma_t* dma;  // the exact DMA table used to compute this entry's prp1
        std::uint64_t memory_token = 0;  // Round 16 S5: for PRP cache key
        std::uint32_t desc_idx = 0;  // Round 16 S6: index into h_dynamic_descs
    };

    std::vector<StripedDeviceSubmitEntry> h_entries;
    h_entries.reserve(count * 2);
    std::vector<std::uint32_t> entry_request_indices;
    // Round 16 S6 (REQUIRED 0): dynamic-path descriptors for entries without
    // a pre-built descriptor.  H2D'd to the arena's per-slot descriptor pool;
    // entry.prp_entry fixed up from nullptr sentinel to GPU pointer.
    std::vector<tutti::data_paths::local_nvme::AddressDescriptor> h_dynamic_descs;
    std::vector<std::uint64_t> h_entry_lengths;  // Round 16 S6: for aggregate
    std::vector<ListInfo> list_infos;
    std::uint64_t total_bytes = 0;
    bool has_rejection = false;
    std::vector<const StripedMemory*> req_mregs(count, nullptr);

    for (std::size_t i = 0; i < count; ++i) {
        if (rejected[i]) {
            has_rejection = true;
            continue;
        }
        const auto& req = requests[i];
        const auto& intent = req.intent;

        if (intent.length > max_request_bytes_) {
            reject_one(i, StatusCode::OUT_OF_RANGE, "request exceeds max_single_io_bytes");
            rejected[i] = true;
            has_rejection = true;
            continue;
        }

        const auto* mreg = find_memory_(req.memory);
        if (!mreg) {
            reject_one(i, StatusCode::NOT_FOUND, "memory not found");
            rejected[i] = true;
            has_rejection = true;
            continue;
        }

        if (intent.target_offset % block_size_ != 0 ||
            intent.length % block_size_ != 0 ||
            intent.memory_offset % block_size_ != 0) {
            reject_one(i, StatusCode::INVALID_ARGUMENT, "not block-aligned");
            rejected[i] = true;
            has_rejection = true;
            continue;
        }
        if (intent.length == 0) {
            reject_one(i, StatusCode::INVALID_ARGUMENT, "zero length");
            rejected[i] = true;
            has_rejection = true;
            continue;
        }
        // Find this request's target index (hash lookup, same map as the
        // dedup pass above — replaces the second O(count*targets) scan).
        const auto tgt_it = tgt_index_of.find(req.target.token());
        const StripedTarget* tgt = (tgt_it != tgt_index_of.end())
            ? targets_in_batch[tgt_it->second].tgt : nullptr;
        const std::uint32_t tgt_idx = (tgt_it != tgt_index_of.end())
            ? tgt_it->second : 0;
        if (!tgt) {
            reject_one(i, StatusCode::NOT_FOUND, "target not in batch (shouldn't happen)");
            rejected[i] = true;
            has_rejection = true;
            continue;
        }
        if (intent.target_offset > tgt->logical_size ||
            intent.length > tgt->logical_size - intent.target_offset) {
            reject_one(i, StatusCode::OUT_OF_RANGE, "target bounds exceeded");
            rejected[i] = true;
            has_rejection = true;
            continue;
        }
        if (intent.memory_offset > mreg->size ||
            intent.length > mreg->size - intent.memory_offset) {
            reject_one(i, StatusCode::OUT_OF_RANGE, "memory bounds exceeded");
            rejected[i] = true;
            has_rejection = true;
            continue;
        }

        req_mregs[i] = mreg;
        std::uint32_t direction = (intent.direction == IoDirection::READ) ? 0 : 1;

        // Stripe-split + MDTS fan-out.
        std::uint64_t remaining = intent.length;
        std::uint64_t cur_off = intent.target_offset;
        std::uint64_t cur_mem = intent.memory_offset;
        bool req_ok = true;

        // Round 16 S5 (V3): pre-built descriptor fast path.
        // If memory was registered with io_granularity > 0 and the sub-IO
        // size matches bytes_per_slice, use pointer arithmetic instead of
        // per-submit PRP computation.  The pre-built descriptors are per-
        // device (each device has its own IOVA table), so we index by
        // (memory_offset / bytes_per_slice) into the correct device's array.
        const bool use_prebuilt = mreg->prebuilt.valid;

        while (remaining > 0) {
            // Round 16 S7: shard formula includes the per-target rotation
            // (legacy shard_placement equivalent).  rot == 0 reproduces the
            // pre-S7 formula exactly.  Must stay identical to
            // StripedLocalNvmePayload::map_to_shard().
            std::uint32_t shard = static_cast<std::uint32_t>(
                ((cur_off / tgt->stripe_unit) + tgt->shard_rotation) %
                tgt->num_shards);
            std::uint64_t shard_off =
                (cur_off / (tgt->stripe_unit * tgt->num_shards)) * tgt->stripe_unit +
                (cur_off % tgt->stripe_unit);

            std::uint64_t unit_remaining = tgt->stripe_unit - (cur_off % tgt->stripe_unit);
            std::uint64_t sub_io = std::min(remaining, unit_remaining);
            const std::uint64_t controller_mdts =
                device_effective_mdts_(shard);
            sub_io = std::min(sub_io, controller_mdts);

            // Clamp to the shard's own extent boundary.
            std::uint64_t ext_end = 0;
            for (const auto& ext : tgt->shard_extents[shard]) {
                std::uint64_t ext_start = ext.logical_offset_bytes;
                std::uint64_t ext_e = ext_start + ext.length_bytes;
                if (shard_off >= ext_start && shard_off < ext_e) {
                    ext_end = ext_e;
                    break;
                }
            }
            if (ext_end > 0) {
                sub_io = std::min(sub_io, ext_end - shard_off);
            }

            StripedDeviceSubmitEntry entry{};
            entry.dev_idx = tgt_idx * static_cast<std::uint32_t>(devices_.size()) + shard;
            entry.direction = direction;
            entry.shard_offset = shard_off;

            const std::uint64_t slice_bytes =
                mreg->prebuilt.bytes_per_slice;
            const auto* prebuilt_table =
                use_prebuilt && shard < mreg->prebuilt.devices.size()
                    ? &mreg->prebuilt.devices[shard]
                    : nullptr;
            const std::uint64_t offset_in_slice = use_prebuilt
                ? cur_mem % slice_bytes : 0;
            const bool at_prebuilt_boundary = prebuilt_table &&
                offset_in_slice % prebuilt_table->mdts_bytes == 0;
            const std::uint64_t expected_prebuilt_bytes =
                at_prebuilt_boundary
                    ? std::min(prebuilt_table->mdts_bytes,
                               slice_bytes - offset_in_slice)
                    : 0;
            if (at_prebuilt_boundary && sub_io == expected_prebuilt_bytes) {
                const std::uint64_t slice_idx = cur_mem / slice_bytes;
                const std::uint64_t sub_idx =
                    offset_in_slice / prebuilt_table->mdts_bytes;
                const std::uint64_t desc_idx =
                    slice_idx * prebuilt_table->ios_per_slice + sub_idx;
                if (slice_idx < mreg->prebuilt.num_slices &&
                    sub_idx < prebuilt_table->ios_per_slice &&
                    desc_idx < prebuilt_table->num_descs &&
                    prebuilt_table->d_descs) {
                    entry.prp_entry = static_cast<const tutti::data_paths::local_nvme::AddressDescriptor*>(
                        prebuilt_table->d_descs) + desc_idx;
                    // Skip PRP computation — kernel will read from prp_entry.
                    h_entries.push_back(entry);
                    entry_request_indices.push_back(static_cast<std::uint32_t>(i));
                    h_entry_lengths.push_back(sub_io);
                    total_bytes += sub_io;
                    ++test_last_prebuilt_entry_count_;
                    remaining -= sub_io;
                    cur_off += sub_io;
                    cur_mem += sub_io;
                    continue;
                }
            }

            // Dynamic path (original): compute PRP into a descriptor.
            // Round 16 S6 (REQUIRED 0): entry no longer carries inline
            // prp1/prp2/length — a descriptor is built and H2D'd to the
            // arena pool; entry.prp_entry is set to nullptr sentinel and
            // fixed up after H2D.
            entry.prp_entry = nullptr;  // sentinel: "dynamic, needs fixup"
            nvm_dma_t* dma = req_mregs[i]->dmas[shard];
            std::uint32_t start_page = static_cast<std::uint32_t>(cur_mem / page_size);
            std::uint32_t pages_in_io = static_cast<std::uint32_t>(
                (sub_io + page_size - 1) / page_size);
            PrpKind kind = classify_prp(pages_in_io);

            if (start_page + pages_in_io > dma->n_ioaddrs) {
                reject_one(i, StatusCode::OUT_OF_RANGE, "PRP page out of DMA range");
                rejected[i] = true;
                req_ok = false;
                break;
            }

            tutti::data_paths::local_nvme::AddressDescriptor desc{};
            desc.prp1 = dma->ioaddrs[start_page];
            desc.data_length = sub_io;

            if (kind == PrpKind::SINGLE) {
                desc.prp2 = 0;
            } else if (kind == PrpKind::DUAL) {
                desc.prp2 = dma->ioaddrs[start_page + 1];
            } else {  // LIST
                desc.prp2 = 0;  // filled after PRP-list alloc
                list_infos.push_back({
                    static_cast<std::uint32_t>(h_entries.size()),
                    start_page, pages_in_io, shard, dma,
                    requests[i].memory.token(),
                    static_cast<std::uint32_t>(h_dynamic_descs.size())});  // desc_idx
            }

            h_dynamic_descs.push_back(desc);
            ++test_last_dynamic_entry_count_;

            h_entries.push_back(entry);
            entry_request_indices.push_back(static_cast<std::uint32_t>(i));
            h_entry_lengths.push_back(sub_io);
            total_bytes += sub_io;
            cur_off += sub_io;
            cur_mem += sub_io;
            remaining -= sub_io;
        }
        if (!req_ok) { has_rejection = true; continue; }

        outcome.initial_states[i].state = RequestState::ACCEPTED;
        outcome.initial_states[i].status = Status::Ok();
    }

    if (h_entries.empty()) {
        outcome.status = Status(first_rejected_code != StatusCode::OK
                                ? first_rejected_code : StatusCode::INVALID_ARGUMENT,
                                "all requests rejected");
        return outcome;
    }

    const std::uint32_t total_entries = static_cast<std::uint32_t>(h_entries.size());
    if (total_entries > max_batch_entries_) {
        reject_all(StatusCode::RESOURCE_EXHAUSTED, "too many sub-IOs (entries)");
        return outcome;
    }
    if (total_bytes > caps_.max_batch_bytes) {
        reject_all(StatusCode::RESOURCE_EXHAUSTED, "batch bytes exceed limit");
        return outcome;
    }

    // ---- Irreversible resource reservation ----
    StripedArena::Lease lease;
    if (!arena_.acquire(lease)) {
        reject_all(StatusCode::RESOURCE_EXHAUSTED,
                   "StripedArena exhausted (all slots in use or leaked)");
        return outcome;
    }
    if (total_dev_table > lease.dev_table_capacity) {
        arena_.release(lease.slot_index);
        reject_all(StatusCode::RESOURCE_EXHAUSTED,
                   "M*N device-table exceeds arena capacity");
        return outcome;
    }

    cudaEvent_t event = static_cast<cudaEvent_t>(lease.event);
    StripedDeviceSubmitEntry* d_entries = lease.d_entries;
    EntryCompletionStatus* d_status = lease.d_status;
    cudaError_t ce;

    // Cache refs checked out during this submit: released on error paths
    // (release_checkout) or pinned for the op's lifetime (pin) on success.
    std::vector<OpEntry::PrpCacheRef> prp_cache_refs;
    auto release_cache_refs = [&]() {
        for (const auto& ref : prp_cache_refs)
            ref.cache->release_checkout(ref.entry);
    };

    // Fill PRP-list pages.
    // Round 16 S5: if PrpPageCache enabled, use cache (host-pinned pool —
    // the cache produces the page content itself from the data DMA table;
    // miss = plain host memcpy, no H2D, no staging).  Lookups run as ONE
    // locked batch per device (was: one mutex round-trip per entry).
    // Otherwise: growing host-pinned pool (no GPU backing or PRP H2D).
    std::vector<tutti::data_paths::local_nvme::PrpBufRef> prp_buf_refs;
    if (!list_infos.empty()) {
        std::vector<bool> li_cached(list_infos.size(), false);
        std::vector<std::uint64_t> li_ioaddr(list_infos.size(), 0);
        if (!prp_caches_.empty()) {
            std::vector<tutti::data_paths::local_nvme::PrpPageCache::BatchItem>
                by_dev;
            by_dev.reserve(list_infos.size());
            for (std::size_t d = 0; d < prp_caches_.size(); ++d) {
                auto* cache = prp_caches_[d].get();
                if (!cache || !cache->enabled()) continue;
                by_dev.clear();
                for (std::size_t i = 0; i < list_infos.size(); ++i) {
                    const auto& li = list_infos[i];
                    if (li.dev_idx != d) continue;
                    tutti::data_paths::local_nvme::PrpPageCache::BatchItem item;
                    item.key.memory_token = li.memory_token;
                    item.key.start_page = li.start_page;
                    item.key.pages_in_io = li.pages_in_io;
                    item.data_dma = li.dma;
                    item.user = static_cast<std::uint32_t>(i);
                    by_dev.push_back(item);
                }
                if (by_dev.empty()) continue;
                cache->get_or_build_batch(by_dev.data(), by_dev.size());
                for (const auto& item : by_dev) {
                    if (item.result == nullptr) continue;
                    li_cached[item.user] = true;
                    li_ioaddr[item.user] = item.result->ioaddr;
                    prp_cache_refs.push_back({cache, item.result});
                }
            }
        }
        std::vector<std::uint64_t> misses_per_dev(devices_.size(), 0);
        for (std::size_t i = 0; i < list_infos.size(); ++i) {
            if (!li_cached[i]) ++misses_per_dev[list_infos[i].dev_idx];
        }
        std::vector<tutti::data_paths::local_nvme::PrpBufRef> refs_by_dev(
            devices_.size());
        for (std::size_t d = 0; d < devices_.size(); ++d) {
            if (misses_per_dev[d] == 0) continue;
            refs_by_dev[d] = prp_buf_pools_[d]->alloc_pages(misses_per_dev[d]);
            if (!refs_by_dev[d].valid) {
                release_cache_refs();
                arena_.release(lease.slot_index);
                reject_all(StatusCode::RESOURCE_EXHAUSTED,
                           "striped host PrpBufPool allocation failed");
                return outcome;
            }
            prp_buf_refs.push_back(refs_by_dev[d]);
        }
        std::vector<std::uint64_t> next_page(devices_.size(), 0);
        for (std::size_t i = 0; i < list_infos.size(); ++i) {
            const auto& li = list_infos[i];
            if (li_cached[i]) {
                h_dynamic_descs[li.desc_idx].prp2 = li_ioaddr[i];
                continue;
            }
            auto& ref = refs_by_dev[li.dev_idx];
            const std::uint64_t page_index =
                ref.base_page + next_page[li.dev_idx]++;
            auto* host_page = reinterpret_cast<std::uint64_t*>(
                static_cast<char*>(ref.segment->vaddr) + page_index * page_size);
            fill_prp_list_page(host_page, li.dma,
                               li.start_page, li.pages_in_io, page_size);
            h_dynamic_descs[li.desc_idx].prp2 =
                ref.segment->ioaddrs[page_index];
        }
    }

    // Round 16 S6 (REQUIRED 0): H2D dynamic descriptors to arena pool,
    // then fix up entry.prp_entry from nullptr sentinel to GPU pointer.
    if (!h_dynamic_descs.empty()) {
        const tutti::data_paths::local_nvme::AddressDescriptor* d_desc_base =
            lease.d_desc_pool;
        ce = cudaMemcpyAsync(const_cast<tutti::data_paths::local_nvme::AddressDescriptor*>(d_desc_base),
                         h_dynamic_descs.data(),
                         h_dynamic_descs.size() * sizeof(tutti::data_paths::local_nvme::AddressDescriptor),
                         cudaMemcpyHostToDevice, ctx.stream);
        if (ce != cudaSuccess) {
            release_cache_refs();
            arena_.release(lease.slot_index);
            reject_all(StatusCode::DEVICE_ERROR, "H2D dynamic descriptors failed");
            return outcome;
        }
        std::uint32_t desc_idx = 0;
        for (auto& e : h_entries) {
            if (e.prp_entry == nullptr) {
                e.prp_entry = d_desc_base + desc_idx;
                ++desc_idx;
            }
        }
    }

    ce = cudaMemcpyAsync(d_entries, h_entries.data(),
                         total_entries * sizeof(StripedDeviceSubmitEntry),
                         cudaMemcpyHostToDevice, ctx.stream);
    if (ce != cudaSuccess) {
        release_cache_refs();
        arena_.release(lease.slot_index);
        reject_all(StatusCode::DEVICE_ERROR, "H2D entries failed");
        return outcome;
    }

    // Round 16 S4: H2D copy M targets' dev_handles into device table.
    // Layout: [target0_shard0, target0_shard1, ..., target0_shardN-1,
    //          target1_shard0, ...]
    {
        std::vector<DeviceTargetHandle*> h_dev_table;
        h_dev_table.reserve(total_dev_table);
        for (std::uint32_t ti = 0; ti < n_targets; ++ti) {
            const auto* t = targets_in_batch[ti].tgt;
            for (std::uint32_t s = 0; s < t->num_shards; ++s)
                h_dev_table.push_back(t->dev_handles[s]);
        }
        ce = cudaMemcpyAsync(const_cast<void*>(static_cast<const void*>(lease.d_dev_table)),
                            h_dev_table.data(),
                            total_dev_table * sizeof(DeviceTargetHandle*),
                            cudaMemcpyHostToDevice, ctx.stream);
    }
    if (ce != cudaSuccess) {
        release_cache_refs();
        arena_.release(lease.slot_index);
        reject_all(StatusCode::DEVICE_ERROR, "H2D dev_table failed");
        return outcome;
    }

    // Match local: pending must be fail-closed if an entry never executes.
    ce = cudaMemsetAsync(d_status, 0xFF,
                        total_entries * sizeof(EntryCompletionStatus),
                        ctx.stream);
    if (ce != cudaSuccess) {
        release_cache_refs();
        arena_.release(lease.slot_index);
        reject_all(StatusCode::DEVICE_ERROR, "cudaMemset d_status failed");
        return outcome;
    }

    const NvtxIoStyle nvtx_style = nvtx_io_style(requests, count);
    nvtx_push_io(nvtx_style);
    // Keep the legacy exact marker as a nested range for existing report
    // queries; the outer range carries the direction and color.
    nvtxRangePushA("tutti.striped_nvme.io_kernel");
    cudaError_t launch_err = launch_fused_submit(
        d_entries, d_status,
        reinterpret_cast<const DeviceTargetHandle* const*>(lease.d_dev_table),
        total_entries, total_dev_table, cq_poll_budget_, threads_per_block_,
        0, ctx.stream);
    nvtxRangePop();
    nvtxRangePop();
    if (launch_err != cudaSuccess) {
        release_cache_refs();
        arena_.release(lease.slot_index);
        reject_all(StatusCode::DEVICE_ERROR,
                   std::string("fused kernel launch failed: ") +
                   cudaGetErrorString(launch_err));
        return outcome;
    }
    test_kernel_launch_count_ += 1;

    ce = cudaEventRecord(event, ctx.stream);
    std::uint64_t op_token = next_op_token_++;
    OpEntry op;
    op.total_bytes = total_bytes;
    op.d_entries = d_entries;
    op.d_status = d_status;
    op.entry_count = total_entries;
    op.entry_lengths = std::move(h_entry_lengths);  // Round 16 S6
    op.event = event;
    op.stream = ctx.stream;
    op.arena_slot = lease.slot_index;
    op.prp_buf_refs = std::move(prp_buf_refs);
    op.op_token = op_token;
    op.op_generation = 1;
    op.target_token = targets_in_batch[0].token;  // Round 16 S4: first target for tracking
    // P0-2 fix: record all accepted requests' memory tokens so
    // memory_has_inflight_ops_() can prevent unregister during in-flight ops.
    for (std::size_t i = 0; i < count; ++i) {
        if (outcome.initial_states[i].state == RequestState::ACCEPTED) {
            op.memory_tokens.push_back(requests[i].memory.token());
        }
    }
    // Pin the PRP cache refs for the op's lifetime (checkout consumed by
    // pin); released in release_impl_.
    for (const auto& ref : prp_cache_refs) ref.cache->pin(ref.entry);
    op.prp_cache_refs = std::move(prp_cache_refs);

    if (ce != cudaSuccess) {
        // Same conservative fallback as LocalNvmeDataPath: sync the stream.
        cudaError_t sync_err = cudaStreamSynchronize(ctx.stream);
        if (sync_err != cudaSuccess) cudaGetLastError();
        op.state = (sync_err == cudaSuccess) ? OpState::COMPLETED : OpState::FAILED;
        op.status = (sync_err == cudaSuccess)
            ? Status::Ok()
            : Status(StatusCode::DEVICE_ERROR, "stream sync failed after event record failure");
        if (sync_err == cudaSuccess) {
            aggregate_completion_status_(op);
        } else {
            op.bytes_transferred = 0;
            set_completion_failure(op.completion_detail,
                                   IoFailureKind::CUDA_QUERY_ERROR, 0);
        }
    } else {
        op.state = OpState::IN_FLIGHT;
        op.status = Status::Ok();
    }
    ops_[op_token] = std::move(op);

    if (has_rejection) {
        outcome.status = Status(first_rejected_code != StatusCode::OK
                                ? first_rejected_code : StatusCode::INVALID_ARGUMENT,
                                "partial commit: " +
                                (first_rejected_msg.empty()
                                 ? std::string("some requests rejected")
                                 : first_rejected_msg));
    } else {
        outcome.status = Status::Ok();
    }
    outcome.op = detail::SpiIdentityMint::mint<detail::DataPathOpTag>(op_token, 1);
    return outcome;
}

// =========================================================================
// progress — poll events
// =========================================================================

Result<ProgressResult> StripedDataPath::progress_impl_(ProgressBudget budget) {
    ProgressResult result{};
    if (budget.max_work_units == 0 || budget.timeout_ns == 0) {
        for (const auto& [tok, op] : ops_) {
            if (op.state == OpState::IN_FLIGHT) { result.more_work_likely = true; break; }
        }
        return Result<ProgressResult>::Success(std::move(result));
    }

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::nanoseconds(budget.timeout_ns);
    std::uint64_t work_done = 0;

    for (auto& [tok, op] : ops_) {
        if (std::chrono::steady_clock::now() >= deadline) {
            result.more_work_likely = true;
            break;
        }
        if (op.state != OpState::IN_FLIGHT) continue;
        if (work_done >= budget.max_work_units) {
            result.more_work_likely = true;
            break;
        }

        // One query = one work unit. Runtime::wait() supplies the polling
        // cadence; spinning here defeats max_work_units and can generate
        // millions of CUDA API calls while an event is not ready.
        cudaError_t ce = cudaEventQuery(static_cast<cudaEvent_t>(op.event));
        if (ce == cudaSuccess) {
            aggregate_completion_status_(op);
            cudaGetLastError();
            ++result.operations_terminal;
        } else if (ce == cudaErrorNotReady) {
            ++result.operations_advanced;
        } else {
            op.state = OpState::FAILED;
            op.status = Status(StatusCode::DEVICE_ERROR,
                               "cudaEventQuery error: " +
                               std::string(cudaGetErrorString(ce)));
            set_completion_failure(op.completion_detail,
                                   IoFailureKind::CUDA_QUERY_ERROR,
                                   op.bytes_transferred);
            cudaGetLastError();
            ++result.operations_terminal;
        }
        ++work_done;
    }

    result.work_units_consumed = work_done;
    for (const auto& [tok, op] : ops_) {
        if (op.state == OpState::IN_FLIGHT) { result.more_work_likely = true; break; }
    }
    return Result<ProgressResult>::Success(std::move(result));
}

// =========================================================================
// aggregate_completion_status_
// =========================================================================

void StripedDataPath::aggregate_completion_status_(OpEntry& op) {
    op.completion_detail = IoCompletionDetail{};
    if (!op.d_status || op.entry_count == 0) {
        op.state = OpState::COMPLETED;
        op.status = Status::Ok();
        op.bytes_transferred = op.total_bytes;
        op.completion_detail.confirmed_bytes = op.total_bytes;
        return;
    }

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

    // The entries array is NOT D2H'd: it was only used to gate byte
    // accounting, and op.entry_lengths (host-side) already carries the
    // per-entry sizes.  Saves one (entry_count * 24B) D2H per batch.
    std::uint64_t confirmed_bytes = 0;
    bool any_failed = false;
    std::string first_error;
    std::uint32_t first_failed_entry = UINT32_MAX;
    std::uint32_t first_failure_result = 0;
    std::uint32_t first_raw_cq_status = 0;

    for (std::uint32_t i = 0; i < op.entry_count; ++i) {
        const auto& s = h_status[i];
        if (s.result == 0) {
            confirmed_bytes += op.entry_lengths[i];
        } else {
            any_failed = true;
            if (first_failed_entry == UINT32_MAX) {
                first_failed_entry = i;
                first_failure_result = s.result;
                first_raw_cq_status = s.nvme_status_dword3;
            }
            if (s.result == 2) op.has_timeout = true;
            if (first_error.empty()) {
                first_error = "entry " + std::to_string(i) + ": result " +
                             std::to_string(s.result);
            }
        }
    }

    if (any_failed) {
        op.state = OpState::FAILED;
        op.status = Status(StatusCode::DEVICE_ERROR, first_error);
        op.bytes_transferred = confirmed_bytes;
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

// =========================================================================
// query / release
// =========================================================================

Result<DataPathSnapshot> StripedDataPath::query_impl_(DataPathOp op) const {
    const auto* entry = find_op_(op);
    if (!entry) {
        return Result<DataPathSnapshot>::Failure(
            Status(StatusCode::NOT_FOUND, "query: op not found"));
    }
    DataPathSnapshot snap;
    snap.state = entry->state;
    snap.status = entry->status;
    snap.bytes_transferred = entry->bytes_transferred;
    snap.detail = entry->completion_detail;
    return Result<DataPathSnapshot>::Success(std::move(snap));
}

Status StripedDataPath::release_impl_(DataPathOp op) {
    auto* entry = find_op_(op);
    if (!entry) {
        return Status(StatusCode::NOT_FOUND, "release: op not found");
    }
    if (entry->state == OpState::IN_FLIGHT) {
        return Status(StatusCode::BUSY, "release: op is still in flight");
    }
    if (entry->arena_slot != UINT32_MAX) {
        if (entry->has_timeout) {
            arena_.release_with_timeout_leak(entry->arena_slot);
        } else {
            arena_.release(entry->arena_slot);
        }
        entry->arena_slot = UINT32_MAX;
    }
    if (entry->has_timeout) timeout_prp_retained_ = true;
    // A timed-out controller command may still fetch cached host PRP pages.
    if (!entry->has_timeout) {
        for (const auto& ref : entry->prp_cache_refs) {
            ref.cache->unpin(ref.entry);
        }
    }
    entry->prp_cache_refs.clear();
    ops_.erase(op.token());
    return Status::Ok();
}

// =========================================================================
// lookups
// =========================================================================

const StripedDataPath::StripedTarget* StripedDataPath::find_target_(
    DataPathTarget target) const {
    if (!target.valid()) return nullptr;
    auto it = targets_.find(target.token());
    if (it == targets_.end()) return nullptr;
    if (it->second.generation != target.generation()) return nullptr;
    return &it->second;
}
StripedDataPath::StripedTarget* StripedDataPath::find_target_(
    DataPathTarget target) {
    if (!target.valid()) return nullptr;
    auto it = targets_.find(target.token());
    if (it == targets_.end()) return nullptr;
    if (it->second.generation != target.generation()) return nullptr;
    return &it->second;
}
const StripedDataPath::StripedMemory* StripedDataPath::find_memory_(
    DataPathMemory memory) const {
    if (!memory.valid()) return nullptr;
    auto it = memory_regs_.find(memory.token());
    if (it == memory_regs_.end()) return nullptr;
    if (it->second.generation != memory.generation()) return nullptr;
    return &it->second;
}
StripedDataPath::StripedMemory* StripedDataPath::find_memory_(
    DataPathMemory memory) {
    if (!memory.valid()) return nullptr;
    auto it = memory_regs_.find(memory.token());
    if (it == memory_regs_.end()) return nullptr;
    if (it->second.generation != memory.generation()) return nullptr;
    return &it->second;
}
const StripedDataPath::OpEntry* StripedDataPath::find_op_(DataPathOp op) const {
    if (!op.valid()) return nullptr;
    auto it = ops_.find(op.token());
    if (it == ops_.end()) return nullptr;
    if (it->second.op_generation != op.generation()) return nullptr;
    return &it->second;
}
StripedDataPath::OpEntry* StripedDataPath::find_op_(DataPathOp op) {
    if (!op.valid()) return nullptr;
    auto it = ops_.find(op.token());
    if (it == ops_.end()) return nullptr;
    if (it->second.op_generation != op.generation()) return nullptr;
    return &it->second;
}
bool StripedDataPath::target_has_inflight_ops_(std::uint64_t token) const {
    for (const auto& [tok, op] : ops_) {
        if (op.state == OpState::IN_FLIGHT && op.target_token == token) return true;
    }
    return false;
}
bool StripedDataPath::memory_has_inflight_ops_(std::uint64_t token) const {
    for (const auto& [tok, op] : ops_) {
        if (op.state != OpState::IN_FLIGHT) continue;
        for (auto t : op.memory_tokens)
            if (t == token) return true;
    }
    return false;
}

// =========================================================================
// test-only accessors
// =========================================================================

bool StripedDataPath::test_op_has_timeout(DataPathOp op) const {
    const auto* entry = find_op_(op);
    return entry ? entry->has_timeout : false;
}
std::uint32_t StripedDataPath::test_entry_count(DataPathOp op) const {
    const auto* entry = find_op_(op);
    return entry ? entry->entry_count : 0;
}
bool StripedDataPath::test_copy_entry_dev_idx(
    DataPathOp op, std::vector<std::uint32_t>& out) const {
    const auto* entry = find_op_(op);
    if (!entry || entry->entry_count == 0 || !entry->d_entries) return false;
    std::vector<StripedDeviceSubmitEntry> h_entries(entry->entry_count);
    cudaError_t ce = cudaMemcpy(h_entries.data(), entry->d_entries,
                               entry->entry_count * sizeof(StripedDeviceSubmitEntry),
                               cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) { cudaGetLastError(); return false; }
    out.resize(entry->entry_count);
    for (std::uint32_t i = 0; i < entry->entry_count; ++i) {
        out[i] = h_entries[i].dev_idx;
    }
    return true;
}

} // namespace tutti::data_paths::striped_local_nvme

#include "nvmeservice_state.h"

#include <nvm_ctrl.h>
#include "tutti_verbose.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

namespace nvmeservice {
namespace {

constexpr char kSnvmeControlPath[] = "/dev/snvm_control";

/**
 * Parse /proc/<pid>/stat field 22 (starttime, clock ticks since boot).
 * The comm field may contain spaces and parentheses, so scan from the last
 * ')' before walking the space-separated fields.
 */
std::optional<uint64_t> read_starttime_impl(uint32_t pid) {
    std::ifstream input("/proc/" + std::to_string(pid) + "/stat");
    std::string line;
    if (!input.is_open() || !std::getline(input, line)) return std::nullopt;
    const auto right_paren = line.rfind(')');
    if (right_paren == std::string::npos) return std::nullopt;
    std::istringstream fields(line.substr(right_paren + 1));
    std::string field;
    for (int index = 0; index < 19; ++index) {
        if (!(fields >> field)) return std::nullopt;
    }
    uint64_t starttime = 0;
    if (!(fields >> starttime)) return std::nullopt;
    return starttime;
}

std::string format_pci_bdf(const pci_device_addr& address) {
    std::ostringstream out;
    out << std::hex << std::nouppercase << std::setfill('0')
        << std::setw(4) << address.domain << ':' << std::setw(2) << address.bus
        << ':' << std::setw(2) << address.slot << '.' << address.func;
    return out.str();
}

BringupValidationProbe production_validation_probe() {
    BringupValidationProbe probe;
    probe.is_character_device = [](const std::string& path) {
        struct stat info {};
        return ::stat(path.c_str(), &info) == 0 && S_ISCHR(info.st_mode);
    };
    probe.is_block_device = [](const std::string& path) {
        struct stat info {};
        return ::stat(path.c_str(), &info) == 0 && S_ISBLK(info.st_mode);
    };
    probe.character_minor = [](const std::string& path)
        -> std::optional<uint32_t> {
        struct stat info {};
        if (::stat(path.c_str(), &info) != 0 || !S_ISCHR(info.st_mode)) {
            return std::nullopt;
        }
        return static_cast<uint32_t>(minor(info.st_rdev));
    };
    probe.block_matches_pci_bdf = [](const std::string& block_path,
                                     const std::string& pci_bdf) {
        std::error_code error;
        const auto name = std::filesystem::path(block_path).filename();
        const auto resolved = std::filesystem::canonical(
            std::filesystem::path("/sys/class/block") / name / "device", error);
        return !error && resolved.string().find(pci_bdf) != std::string::npos;
    };
    return probe;
}

} // namespace

// ---------------------------------------------------------------------------
// Owner bring-up metadata validation
// ---------------------------------------------------------------------------

bool validate_bringup_metadata(const BringupMetadata& metadata,
                               const BringupValidationProbe& probe,
                               std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    if (metadata.configured_pci_bdf != metadata.observed_pci_bdf) {
        return fail("configured/observed PCI BDF mismatch");
    }
    if (metadata.chrdev_minor < 0 || metadata.chrdev_path.empty()) {
        return fail("owner returned an invalid chrdev minor/path");
    }
    if (metadata.disk_name.empty() || metadata.block_path.empty() ||
        metadata.block_path != "/dev/" + metadata.disk_name) {
        return fail("owner returned an invalid disk_name/block_path");
    }
    if (!probe.is_character_device ||
        !probe.is_character_device(metadata.chrdev_path)) {
        return fail("chrdev path does not exist or is not a character device");
    }
    if (!probe.character_minor) return fail("chrdev minor probe is missing");
    const auto observed_minor = probe.character_minor(metadata.chrdev_path);
    if (!observed_minor ||
        *observed_minor != static_cast<uint32_t>(metadata.chrdev_minor)) {
        return fail("stale chrdev minor/path mismatch");
    }
    if (!probe.is_block_device || !probe.is_block_device(metadata.block_path)) {
        return fail("block path does not exist or is not a block device");
    }
    if (!probe.block_matches_pci_bdf ||
        !probe.block_matches_pci_bdf(metadata.block_path,
                                     metadata.configured_pci_bdf)) {
        return fail("block device is not associated with configured PCI BDF");
    }
    return true;
}

std::string ServiceState::generate_allocation_id() {
    static thread_local std::mt19937_64 random{std::random_device{}()};
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << random()
        << std::setw(16) << random();
    return out.str();
}

std::optional<uint64_t> ServiceState::read_pid_starttime(uint32_t pid) {
    return read_starttime_impl(pid);
}

// ---------------------------------------------------------------------------
// Construction and hardware resource initialization
// ---------------------------------------------------------------------------

ServiceState::ServiceState(const ServiceConfig& config)
    : config_(config),
      process_dead_probe_([this](uint32_t pid, uint64_t starttime) {
          return default_process_dead(pid, starttime);
      }),
      allocation_id_generator_(&ServiceState::generate_allocation_id) {
    initialize_hardware_resources();
}

ServiceState::ServiceState(const ServiceConfig& config,
                           std::vector<NvmeResourceSnapshot> resources,
                           ProcessDeadProbe process_dead_probe,
                           AllocationIdGenerator allocation_id_generator)
    : config_(config),
      process_dead_probe_(std::move(process_dead_probe)),
      allocation_id_generator_(std::move(allocation_id_generator)) {
    if (!process_dead_probe_) {
        process_dead_probe_ = [this](uint32_t pid, uint64_t starttime) {
            return default_process_dead(pid, starttime);
        };
    }
    if (!allocation_id_generator_) {
        allocation_id_generator_ = &ServiceState::generate_allocation_id;
    }
    devices_.reserve(resources.size());
    for (auto& resource : resources) {
        DeviceState device;
        device.control_ready = resource.available;
        device.resource = std::move(resource);
        device.resource.reserved_queues = 0;
        device.resource.available_queues = device.resource.controller_queue_capacity;
        devices_.push_back(std::move(device));
    }
    rebuild_device_index();
}

ServiceState::~ServiceState() {
    stop_reaper();
    unpublish_accelerator_views();
    for (auto& device : devices_) {
        // Owner-side teardown cascades through device unbind and chrdev removal.
        if (device.ctrl) nvm_ctrl_free(device.ctrl);
        device.ctrl = nullptr;
    }
}

void ServiceState::initialize_hardware_resources() {
    devices_.reserve(config_.nvmes.size());
    for (const auto& entry : config_.nvmes) add_hardware_resource(entry);
    rebuild_device_index();

    std::vector<uint32_t> block_sizes;
    for (const auto& device : devices_) {
        if (device.control_ready) {
            block_sizes.push_back(device.resource.logical_block_size);
        }
    }
    if (block_sizes.size() > 1) {
        std::string error;
        if (!validate_uniform_block_size(block_sizes, &error)) {
            for (auto& device : devices_) {
                if (device.control_ready) {
                    device.control_ready = false;
                    device.resource.available = false;
                    device.resource.diagnostic = error;
                }
            }
        }
    }
}

// Owner-side bring-up returns the actual character-device minor/path and the
// ioctl disk name.  No path is derived from the YAML array position.
void ServiceState::add_hardware_resource(const NvmeEntry& entry) {
    DeviceState device;
    auto& resource = device.resource;
    resource.device_id = entry.device_id;
    resource.pci_bdf = entry.pci_addr;
    resource.backing_mount_path = entry.backing_mount_path;
    resource.namespace_id = entry.namespace_id;
    resource.allowed_accel_ids = entry.allowed_accel_ids;
    resource.heartbeat_interval_sec = config_.lease.heartbeat_interval_sec;
    resource.lease_timeout_sec = config_.lease.timeout_sec;

    struct disk disk {};
    struct nvm_owner_bringup_result bringup {};
    int status = nvm_controller_init_owner_with_result(
        &device.ctrl, kSnvmeControlPath, entry.pci_addr.c_str(),
        entry.kernel_ioq_cap, &disk, &bringup);
    if (status != 0) {
        resource.diagnostic = "owner bring-up failed errno=" +
                              std::to_string(status);
        devices_.push_back(std::move(device));
        return;
    }

    resource.chrdev_minor = bringup.chrdev_minor;
    resource.chrdev_path = bringup.chrdev_path;
    resource.disk_name = bringup.disk_name;
    resource.block_path = bringup.block_path;
    resource.bar0_size = device.ctrl->bar0_size;
    resource.dstrd = device.ctrl->dstrd;
    resource.max_user_qid = device.ctrl->max_user_qid;
    // QID 0 is the admin queue. Kernel I/O QPs occupy [1, start_cq_idx),
    // while the user reservation capacity is [start_cq_idx, max_user_qid].
    const uint32_t first_user_qid = device.ctrl->start_cq_idx;
    resource.kernel_io_qps = first_user_qid > 0 ? first_user_qid - 1 : 0;
    resource.controller_queue_capacity =
        first_user_qid > 0 && device.ctrl->max_user_qid >= first_user_qid
            ? device.ctrl->max_user_qid - first_user_qid + 1
            : 0;
    resource.available_queues = resource.controller_queue_capacity;
    resource.max_queues_per_group = device.ctrl->max_queues_per_group;
    resource.page_size = device.ctrl->page_size;
    resource.queue_depth = device.ctrl->q_depth;
    resource.max_data_size = disk.max_data_size;
    resource.logical_block_size = static_cast<uint32_t>(disk.block_size);
    if (resource.logical_block_size) {
        resource.logical_block_size_log =
            static_cast<uint32_t>(__builtin_ctz(resource.logical_block_size));
    }

    // Fail closed if the owner facts do not match the configured BDF or the
    // actual character/block device association.
    BringupMetadata metadata;
    metadata.configured_pci_bdf = entry.pci_addr;
    metadata.observed_pci_bdf = format_pci_bdf(device.ctrl->pdev_addr);
    metadata.chrdev_minor = bringup.chrdev_minor;
    metadata.chrdev_path = bringup.chrdev_path;
    metadata.disk_name = bringup.disk_name;
    metadata.block_path = bringup.block_path;
    std::string validation_error;
    device.control_ready = validate_bringup_metadata(
        metadata, production_validation_probe(), &validation_error);
    resource.available = false;
    resource.diagnostic = device.control_ready
        ? "backing mount and accelerator views are not published"
        : validation_error;

    TUTTI_INFO("nvmeservice: device_id=%d pci_bdf=%s chrdev=%s block=%s "
               "capacity=%u max_q_per_group=%u control_ready=%d\n",
               resource.device_id, resource.pci_bdf.c_str(),
               resource.chrdev_path.c_str(), resource.block_path.c_str(),
               resource.controller_queue_capacity,
               resource.max_queues_per_group, device.control_ready ? 1 : 0);
    devices_.push_back(std::move(device));
}

void ServiceState::rebuild_device_index() {
    device_index_.clear();
    for (size_t index = 0; index < devices_.size(); ++index) {
        if (!device_index_.emplace(devices_[index].resource.device_id, index).second) {
            throw std::runtime_error("duplicate device_id in resource snapshot");
        }
    }
}

ServiceState::DeviceState* ServiceState::find_device_locked(int32_t device_id) {
    const auto found = device_index_.find(device_id);
    return found == device_index_.end() ? nullptr : &devices_[found->second];
}

const ServiceState::DeviceState* ServiceState::find_device_locked(
    int32_t device_id) const {
    const auto found = device_index_.find(device_id);
    return found == device_index_.end() ? nullptr : &devices_[found->second];
}

const AcceleratorEntry* ServiceState::find_accelerator(int32_t accel_id) const {
    for (const auto& accelerator : config_.accelerators) {
        if (accelerator.accel_id == accel_id) return &accelerator;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Canonical snapshots and accelerator-view lifecycle
// ---------------------------------------------------------------------------

std::vector<AcceleratorSnapshot> ServiceState::list_accelerators() const {
    std::vector<AcceleratorSnapshot> result;
    result.reserve(config_.accelerators.size());
    for (const auto& accelerator : config_.accelerators) {
        result.push_back({accelerator.accel_id, accelerator.view_root});
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        return left.accel_id < right.accel_id;
    });
    return result;
}

std::vector<NvmeResourceSnapshot> ServiceState::list_nvme_resources() const {
    std::lock_guard<std::mutex> lock(state_mtx_);
    std::vector<NvmeResourceSnapshot> result;
    result.reserve(devices_.size());
    for (const auto& device : devices_) {
        auto resource = device.resource;
        resource.available_queues = resource.controller_queue_capacity >=
                resource.reserved_queues
            ? resource.controller_queue_capacity - resource.reserved_queues
            : 0;
        resource.available = device.control_ready && !resource.view_paths.empty() &&
                             resource.available_queues > 0;
        if (resource.available) resource.diagnostic.clear();
        result.push_back(std::move(resource));
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        return left.device_id < right.device_id;
    });
    return result;
}

bool ServiceState::publish_accelerator_views(int32_t device_id) {
    std::lock_guard<std::mutex> lock(state_mtx_);
    auto* device = find_device_locked(device_id);
    if (!device || !device->control_ready) return false;
    return install_accelerator_views_locked(*device);
}

bool ServiceState::install_accelerator_views_locked(DeviceState& device) {
    namespace fs = std::filesystem;
    bool complete = true;
    // The link basename comes from the owner-returned chrdev path.  It is not
    // synthesized from device_id or from the resource vector index.
    const std::string link_name =
        fs::path(device.resource.chrdev_path).filename().string();
    for (int32_t accel_id : device.resource.allowed_accel_ids) {
        const auto* accelerator = find_accelerator(accel_id);
        if (!accelerator || accelerator->view_root.empty() || link_name.empty()) {
            complete = false;
            continue;
        }
        const fs::path backing_subdir =
            fs::path(device.resource.backing_mount_path) /
            ("ACCEL" + std::to_string(accel_id));
        const fs::path view_path = fs::path(accelerator->view_root) / link_name;
        std::error_code error;
        fs::create_directories(backing_subdir, error);
        if (error) {
            complete = false;
            continue;
        }
        device.created_backing_subdirs.push_back(backing_subdir.string());
        fs::create_directories(accelerator->view_root, error);
        if (error) {
            complete = false;
            continue;
        }
        const auto status = fs::symlink_status(view_path, error);
        if (!error && status.type() == fs::file_type::symlink) {
            const auto target = fs::read_symlink(view_path, error);
            if (!error && target == backing_subdir) {
                device.created_symlinks.push_back(view_path.string());
                device.resource.view_paths[accel_id] = view_path.string();
                continue;
            }
            fs::remove(view_path, error);
        } else if (!error && fs::exists(status)) {
            complete = false;
            continue;
        }
        error.clear();
        fs::create_symlink(backing_subdir, view_path, error);
        if (error) {
            complete = false;
            continue;
        }
        device.created_symlinks.push_back(view_path.string());
        device.resource.view_paths[accel_id] = view_path.string();
    }
    device.resource.available = device.control_ready &&
                                !device.resource.view_paths.empty();
    device.resource.diagnostic = complete ? "" :
        "one or more accelerator views could not be published";
    return complete;
}

void ServiceState::unpublish_accelerator_views() {
    std::lock_guard<std::mutex> lock(state_mtx_);
    for (auto& device : devices_) remove_accelerator_views_locked(device);
}

void ServiceState::remove_accelerator_views_locked(DeviceState& device) {
    std::error_code error;
    for (const auto& path : device.created_symlinks) {
        std::filesystem::remove(path, error);
    }
    for (const auto& path : device.created_backing_subdirs) {
        ::rmdir(path.c_str());
    }
    device.created_symlinks.clear();
    device.created_backing_subdirs.clear();
    device.resource.view_paths.clear();
    device.resource.available = false;
}

uint32_t ServiceState::compute_grant_locked(const DeviceState& device,
                                            int32_t requested) const {
    int32_t grant = requested > 0 ? requested :
        config_.queue_pool.default_per_client;
    grant = std::min(grant, config_.queue_pool.max_per_client);
    if (device.resource.max_queues_per_group > 0) {
        grant = std::min(grant,
                         static_cast<int32_t>(device.resource.max_queues_per_group));
    }
    return grant > 0 ? static_cast<uint32_t>(grant) : 0;
}

// ---------------------------------------------------------------------------
// Atomic selection, queue reservation, and allocation creation
// ---------------------------------------------------------------------------

bool ServiceState::can_allocate_locked(const DeviceState& device,
                                       int32_t accel_id, uint32_t grant,
                                       std::string* error) const {
    auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    if (!device.control_ready) {
        return fail("resource is not available: " + device.resource.diagnostic);
    }
    if (std::find(device.resource.allowed_accel_ids.begin(),
                  device.resource.allowed_accel_ids.end(), accel_id) ==
        device.resource.allowed_accel_ids.end()) {
        return fail("accelerator is not allowed by resource ACL");
    }
    const auto view = device.resource.view_paths.find(accel_id);
    if (view == device.resource.view_paths.end() || view->second.empty()) {
        return fail("accelerator view is not published");
    }
    const auto* accelerator = find_accelerator(accel_id);
    if (!accelerator) return fail("unknown accelerator");
    const auto root = std::filesystem::path(accelerator->view_root).lexically_normal();
    const auto path = std::filesystem::path(view->second).lexically_normal();
    const auto relative = path.lexically_relative(root);
    if (relative.empty() || relative.is_absolute() ||
        *relative.begin() == "..") {
        return fail("view_path is outside the configured view_root");
    }
    if (grant == 0) return fail("queue policy yielded zero queues");
    if (device.resource.reserved_queues >
            device.resource.controller_queue_capacity ||
        grant > device.resource.controller_queue_capacity -
                    device.resource.reserved_queues) {
        return fail("controller queue budget is insufficient");
    }
    return true;
}

NvmeSliceGrant ServiceState::make_slice_locked(const DeviceState& device,
                                               int32_t accel_id,
                                               uint32_t grant) const {
    const auto& resource = device.resource;
    NvmeSliceGrant slice;
    slice.device_id = resource.device_id;
    slice.accel_id = accel_id;
    slice.pci_bdf = resource.pci_bdf;
    slice.chrdev_minor = resource.chrdev_minor;
    slice.chrdev_path = resource.chrdev_path;
    slice.block_path = resource.block_path;
    slice.backing_mount_path = resource.backing_mount_path;
    slice.view_path = resource.view_paths.at(accel_id);
    slice.namespace_id = resource.namespace_id;
    slice.page_size = resource.page_size;
    slice.logical_block_size = resource.logical_block_size;
    slice.logical_block_size_log = resource.logical_block_size_log;
    slice.queue_depth = resource.queue_depth;
    slice.dstrd = resource.dstrd;
    slice.bar0_size = resource.bar0_size;
    slice.max_data_size = resource.max_data_size;
    slice.controller_queue_capacity = resource.controller_queue_capacity;
    slice.granted_queues = grant;
    slice.max_queues_per_group = resource.max_queues_per_group;
    slice.allowed_accel_ids = resource.allowed_accel_ids;
    slice.heartbeat_interval_sec = config_.lease.heartbeat_interval_sec;
    slice.lease_timeout_sec = config_.lease.timeout_sec;
    return slice;
}

AcquireResult ServiceState::acquire(const AcquireRequest& request) {
    AcquireResult result;
    std::lock_guard<std::mutex> lock(state_mtx_);
    if (!find_accelerator(request.accel_id)) {
        result.error = "unknown accel_id";
        return result;
    }

    // Selection, ACL/view validation, queue-budget checks, and allocation
    // insertion all run under state_mtx_. A striped failure cannot partially
    // reserve an earlier controller.
    std::vector<DeviceState*> selected;
    if (request.selection == SelectionMode::Allowed) {
        if (!request.device_ids.empty()) {
            result.error = "allowed selection must not carry device_ids";
            return result;
        }
        std::vector<int32_t> ids;
        ids.reserve(device_index_.size());
        for (const auto& item : device_index_) ids.push_back(item.first);
        std::sort(ids.begin(), ids.end());
        for (int32_t device_id : ids) {
            auto* device = find_device_locked(device_id);
            const uint32_t grant = compute_grant_locked(
                *device, request.queues_per_controller);
            if (can_allocate_locked(*device, request.accel_id, grant, nullptr)) {
                selected.push_back(device);
                break;
            }
        }
        if (selected.empty()) {
            result.error = "no allowed resource is currently available";
            return result;
        }
    } else {
        const size_t required = request.selection == SelectionMode::Explicit ? 1 : 2;
        if ((request.selection == SelectionMode::Explicit &&
             request.device_ids.size() != required) ||
            (request.selection == SelectionMode::Striped &&
             request.device_ids.size() < required)) {
            result.error = request.selection == SelectionMode::Explicit
                ? "explicit selection requires exactly one device_id"
                : "striped selection requires at least two device_ids";
            return result;
        }
        std::unordered_set<int32_t> unique;
        for (int32_t device_id : request.device_ids) {
            if (!unique.insert(device_id).second) {
                result.error = "selection contains duplicate device_id";
                return result;
            }
            auto* device = find_device_locked(device_id);
            if (!device) {
                result.error = "unknown device_id=" + std::to_string(device_id);
                return result;
            }
            selected.push_back(device);
        }
    }

    std::vector<uint32_t> grants;
    grants.reserve(selected.size());
    for (const auto* device : selected) {
        const uint32_t grant = compute_grant_locked(
            *device, request.queues_per_controller);
        std::string error;
        if (!can_allocate_locked(*device, request.accel_id, grant, &error)) {
            result.error = "device_id=" +
                           std::to_string(device->resource.device_id) +
                           ": " + error;
            return result;
        }
        grants.push_back(grant);
    }

    Allocation allocation;
    allocation.allocation_id = allocation_id_generator_();
    if (allocation.allocation_id.empty() ||
        allocations_.count(allocation.allocation_id) ||
        released_allocation_ids_.count(allocation.allocation_id)) {
        result.error = "allocation_id generation collision";
        return result;
    }
    allocation.accel_id = request.accel_id;
    allocation.client_pid = request.client_pid;
    allocation.client_pid_starttime =
        read_pid_starttime(request.client_pid).value_or(0);
    allocation.last_heartbeat = std::chrono::steady_clock::now();
    for (size_t index = 0; index < selected.size(); ++index) {
        allocation.reservations.push_back(
            {selected[index]->resource.device_id, grants[index]});
    }

    const std::string allocation_id = allocation.allocation_id;
    try {
        // Apply every controller reservation only after all grants validate.
        for (size_t index = 0; index < selected.size(); ++index) {
            selected[index]->resource.reserved_queues += grants[index];
        }
        const auto inserted = allocations_.emplace(allocation_id,
                                                   std::move(allocation));
        if (!inserted.second) throw std::runtime_error("allocation insert failed");
    } catch (...) {
        for (size_t index = 0; index < selected.size(); ++index) {
            selected[index]->resource.reserved_queues -= grants[index];
        }
        result.error = "allocation insert failed";
        return result;
    }

    result.grant.allocation_id = allocation_id;
    for (size_t index = 0; index < selected.size(); ++index) {
        result.grant.slices.push_back(
            make_slice_locked(*selected[index], request.accel_id, grants[index]));
    }
    result.success = true;
    return result;
}

ReleaseResult ServiceState::release_locked(const std::string& allocation_id) {
    ReleaseResult result;
    const auto found = allocations_.find(allocation_id);
    if (found == allocations_.end()) {
        if (released_allocation_ids_.count(allocation_id)) {
            result.success = true;
            result.already_released = true;
        } else {
            result.error = "unknown allocation_id";
        }
        return result;
    }
    // Validate the complete reservation record before changing any ledger
    // entry, so corruption cannot cause a partial refund.
    for (const auto& reservation : found->second.reservations) {
        auto* device = find_device_locked(reservation.device_id);
        if (!device || device->resource.reserved_queues < reservation.queues) {
            result.error = "reservation ledger corruption";
            return result;
        }
    }
    for (const auto& reservation : found->second.reservations) {
        auto* device = find_device_locked(reservation.device_id);
        device->resource.reserved_queues -= reservation.queues;
    }
    allocations_.erase(found);
    released_allocation_ids_.insert(allocation_id);
    result.success = true;
    return result;
}

ReleaseResult ServiceState::release(const std::string& allocation_id) {
    std::lock_guard<std::mutex> lock(state_mtx_);
    return release_locked(allocation_id);
}

ServiceState::ConnectResult ServiceState::connect(int32_t device_id,
                                                  int32_t cuda_device,
                                                  int32_t num_queues,
                                                  uint32_t client_pid) {
    // Compatibility Connect is deliberately a single explicit acquisition;
    // it therefore cannot bypass the canonical ACL or queue ledger.
    AcquireRequest request;
    request.accel_id = cuda_device;
    request.selection = SelectionMode::Explicit;
    request.device_ids = {device_id};
    request.queues_per_controller = num_queues;
    request.client_pid = client_pid;
    const auto acquired = acquire(request);
    ConnectResult result;
    if (!acquired.success) {
        result.error = acquired.error;
        return result;
    }
    const auto& slice = acquired.grant.slices.front();
    result.grant.allocation_id = acquired.grant.allocation_id;
    result.grant.device_id = slice.device_id;
    result.grant.pci_addr = slice.pci_bdf;
    result.grant.snvme_dev_path = slice.chrdev_path;
    result.grant.bar0_size = slice.bar0_size;
    result.grant.dstrd = slice.dstrd;
    result.grant.granted_queues = static_cast<int32_t>(slice.granted_queues);
    result.grant.namespace_id = slice.namespace_id;
    result.grant.page_size = slice.page_size;
    result.grant.blk_size = slice.logical_block_size;
    result.grant.blk_size_log = slice.logical_block_size_log;
    result.grant.queue_depth = slice.queue_depth;
    result.grant.max_data_size = slice.max_data_size;
    result.grant.mount_path = slice.view_path;
    result.grant.heartbeat_interval_sec = slice.heartbeat_interval_sec;
    result.grant.lease_timeout_sec = slice.lease_timeout_sec;
    result.success = true;
    return result;
}

bool ServiceState::disconnect(const std::string& allocation_id,
                              uint32_t client_pid, std::string* error) {
    std::lock_guard<std::mutex> lock(state_mtx_);
    const auto found = allocations_.find(allocation_id);
    if (found != allocations_.end() && found->second.client_pid != client_pid) {
        if (error) *error = "pid mismatch";
        return false;
    }
    const auto result = release_locked(allocation_id);
    if (!result.success && error) *error = result.error;
    return result.success;
}

bool ServiceState::update_heartbeat(const std::string& allocation_id,
                                    std::string* error) {
    std::lock_guard<std::mutex> lock(state_mtx_);
    const auto found = allocations_.find(allocation_id);
    if (found == allocations_.end()) {
        if (error) *error = "unknown allocation_id";
        return false;
    }
    found->second.last_heartbeat = std::chrono::steady_clock::now();
    return true;
}

bool ServiceState::has_allocation(const std::string& allocation_id) const {
    std::lock_guard<std::mutex> lock(state_mtx_);
    return allocations_.count(allocation_id) != 0;
}

bool ServiceState::default_process_dead(uint32_t pid, uint64_t starttime) const {
    if (pid == 0 || ::kill(static_cast<pid_t>(pid), 0) != 0) return true;
    const auto current = read_pid_starttime(pid);
    return !current || starttime == 0 || *current != starttime;
}

// ---------------------------------------------------------------------------
// Lease reaper
// ---------------------------------------------------------------------------

size_t ServiceState::reap_once(std::chrono::steady_clock::time_point now) {
    std::lock_guard<std::mutex> lock(state_mtx_);
    const auto timeout = std::chrono::seconds(config_.lease.timeout_sec);
    std::vector<std::string> expired;
    for (const auto& item : allocations_) {
        const auto& allocation = item.second;
        if (now - allocation.last_heartbeat > timeout ||
            process_dead_probe_(allocation.client_pid,
                                allocation.client_pid_starttime)) {
            expired.push_back(item.first);
        }
    }
    // Heartbeat timeout and dead/reused PID cleanup use the same full Release
    // helper as explicit Release and legacy Disconnect.
    for (const auto& allocation_id : expired) {
        const auto result = release_locked(allocation_id);
        if (!result.success) {
            TUTTI_INFO("nvmeservice reaper failed allocation_id=%s: %s\n",
                       allocation_id.c_str(), result.error.c_str());
        }
    }
    return expired.size();
}

void ServiceState::start_reaper() {
    if (reaper_running_.exchange(true)) return;
    reaper_thread_ = std::thread(&ServiceState::reaper_loop, this);
}

void ServiceState::stop_reaper() {
    if (!reaper_running_.exchange(false)) return;
    if (reaper_thread_.joinable()) reaper_thread_.join();
}

void ServiceState::reaper_loop() {
    const auto tick = std::chrono::seconds(
        std::max<uint32_t>(1, config_.lease.heartbeat_interval_sec / 2));
    while (reaper_running_.load()) {
        std::this_thread::sleep_for(tick);
        if (reaper_running_.load()) reap_once(std::chrono::steady_clock::now());
    }
}

} // namespace nvmeservice

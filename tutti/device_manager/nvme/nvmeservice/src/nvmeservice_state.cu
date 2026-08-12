#include "nvmeservice_state.h"

#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_error.h>
#include "tutti_verbose.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace nvmeservice {

namespace {

constexpr char kSnvmeControlPath[] = "/dev/snvm_control";

/**
 * Parse /proc/<pid>/stat field 22 (starttime, clock ticks since boot).
 * The comm field may contain spaces and parens, so scan from the
 * last ')' and walk space-separated fields.
 */
std::optional<uint64_t> read_starttime_impl(uint32_t pid) {
    char path[64];
    std::snprintf(path, sizeof(path), "/proc/%u/stat", pid);
    std::ifstream in(path);
    if (!in.is_open()) return std::nullopt;

    std::string line;
    if (!std::getline(in, line)) return std::nullopt;

    auto rparen = line.rfind(')');
    if (rparen == std::string::npos) return std::nullopt;

    std::istringstream iss(line.substr(rparen + 1));
    std::string tok;
    for (int i = 0; i < 19; ++i) {
        if (!(iss >> tok)) return std::nullopt;
    }
    uint64_t starttime;
    if (!(iss >> starttime)) return std::nullopt;
    return starttime;
}

} // namespace

// ---------------------------------------------------------------------------
// Helpers (static)
// ---------------------------------------------------------------------------

std::string ServiceState::generate_allocation_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t hi = dist(rng);
    uint64_t lo = dist(rng);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << hi
        << std::setw(16) << lo;
    return oss.str();
}

std::optional<uint64_t> ServiceState::read_pid_starttime(uint32_t pid) {
    return read_starttime_impl(pid);
}

const std::string* ServiceState::gpu_mount_for(int gpu_id) const {
    for (const auto& g : cfg_.gpus) {
        if (g.id == gpu_id) return &g.mount_path;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

ServiceState::ServiceState(const ServiceConfig& cfg) : cfg_(cfg) {
    devices_.reserve(cfg_.nvmes.size());

    auto release_brought_up_devices = [&]() {
        for (auto& dev : devices_) {
            if (dev.ctrl != nullptr) {
                nvm_ctrl_free(dev.ctrl);
                dev.ctrl = nullptr;
            }
        }
    };

    try {
        for (size_t i = 0; i < cfg_.nvmes.size(); ++i) {
            init_device(cfg_.nvmes[i], static_cast<int32_t>(i));
        }

        // disk.block_size is the logical block size of each configured
        // namespace, in bytes (1 << namespace LBA shift), not a
        // controller-wide or filesystem block size.
        std::vector<uint32_t> block_sizes;
        block_sizes.reserve(devices_.size());
        for (const auto& dev : devices_) {
            block_sizes.push_back(dev.blk_size);
        }
        std::string block_size_error;
        if (!validate_uniform_block_size(block_sizes, &block_size_error)) {
            // Keep the 4 KiB warning visible even when the same startup is
            // rejected for a mixed-size topology.
            for (const auto& dev : devices_) {
                if (dev.blk_size != kExpectedNvmeBlockSize) {
                    std::fprintf(stderr,
                                 "WARNING: NVMe device_id=%d pci=%s reports "
                                 "block_size=%u bytes; expected %u bytes "
                                 "for striped operation\n",
                                 dev.device_id, dev.pci_addr.c_str(),
                                 dev.blk_size, kExpectedNvmeBlockSize);
                }
            }
            throw std::runtime_error(block_size_error);
        }
    } catch (...) {
        // A constructor that fails does not run ServiceState::~ServiceState.
        // Release controllers already brought up before propagating the
        // original error, including a later block-size validation failure.
        release_brought_up_devices();
        throw;
    }
}

ServiceState::~ServiceState() {
    stop_reaper();
    unpublish_gpu_views();
    for (auto& dev : devices_) {
        if (dev.ctrl != nullptr) {
            // Owner-side teardown: nvm_ctrl_free cascades through
            // SNVM_DEVICE_UNBIND + SNVM_CHRDEV_REMOVE.
            nvm_ctrl_free(dev.ctrl);
            dev.ctrl = nullptr;
        }
    }
}

// ---------------------------------------------------------------------------
// Device initialisation
// ---------------------------------------------------------------------------

void ServiceState::init_device(const NvmeEntry& nvme, int32_t device_id) {
    DeviceState dev;
    dev.device_id    = device_id;
    dev.pci_addr     = nvme.pci_addr;
    dev.mount_path   = nvme.mount_path;
    dev.namespace_id = nvme.namespace_id;

    // Owner-side bring-up: chrdev_create + cap + bind + probe.
    nvm_ctrl_t* ctrl = nullptr;
    struct disk disk;
    std::memset(&disk, 0, sizeof(disk));

    int rc = nvm_controller_init_owner(&ctrl,
                                       kSnvmeControlPath,
                                       nvme.pci_addr.c_str(),
                                       nvme.kernel_ioq_cap,
                                       &disk);
    if (rc != 0) {
        throw std::runtime_error(
            "nvm_controller_init_owner failed for pci=" + nvme.pci_addr +
            " errno=" + std::to_string(rc));
    }

    dev.ctrl                 = ctrl;
    dev.bar0_size            = ctrl->bar0_size;
    dev.dstrd                = ctrl->dstrd;
    dev.max_user_qid         = ctrl->max_user_qid;
    // QID 0 is the admin queue.  Kernel I/O QPs occupy
    // [1, start_cq_idx), while the user pool occupies the inclusive
    // range [start_cq_idx, max_user_qid].
    const uint32_t first_user_qid = ctrl->start_cq_idx;
    dev.kernel_io_qps        = first_user_qid > 0 ? first_user_qid - 1 : 0;
    dev.user_io_qps          = first_user_qid > 0 &&
                                ctrl->max_user_qid >= first_user_qid
                                    ? ctrl->max_user_qid - first_user_qid + 1
                                    : 0;
    dev.max_queues_per_group = ctrl->max_queues_per_group;
    dev.page_size            = ctrl->page_size;
    dev.queue_depth          = ctrl->q_depth;
    dev.max_data_size        = disk.max_data_size;   // CTRL.MDTS in bytes
    dev.blk_size             = static_cast<uint32_t>(disk.block_size);
    if (dev.blk_size > 0) {
        dev.blk_size_log = static_cast<uint32_t>(__builtin_ctz(dev.blk_size));
    }

    // Reconstruct /dev/ssnvme<minor> path from disk.disk_name.
    // PORTING.md: nvme namespace device "snvmeNn1" shares the minor N
    // with the chrdev "ssnvmeN".  TODO: extend ioctl to expose minor
    // directly; entry tracked in Todolist.md.
    {
        std::string nm = disk.disk_name;
        const std::string prefix = "snvme";
        if (nm.rfind(prefix, 0) == 0) nm.erase(0, prefix.size());
        std::string minor_digits;
        for (char c : nm) {
            if (c < '0' || c > '9') break;
            minor_digits.push_back(c);
        }
        if (minor_digits.empty()) minor_digits = "0";
        dev.snvme_dev_path = "/dev/ssnvme" + minor_digits;
    }

    // ACL: build the allowed_gpus set.  Empty list in YAML means
    // "every gpus[].id is allowed"; expand to all known ids so
    // downstream code (symlink install + ConnectRequest validation)
    // can use a single uniform check.
    if (nvme.allowed_gpus.empty()) {
        for (const auto& g : cfg_.gpus) dev.allowed_gpus.insert(g.id);
    } else {
        for (int gid : nvme.allowed_gpus) dev.allowed_gpus.insert(gid);
    }

    // Bring-up banner — info level, gated by TUTTI_VERBOSE.
    TUTTI_INFO(
        "nvmeservice: device=%d pci=%s snvme=%s ns=%u qdepth=%u "
        "io_qp_limit=%u kernel_io_qps=%u user_io_qps=%u "
        "max_user_qid=%u max_q_per_grp=%u allowed_gpus={",
        device_id, nvme.pci_addr.c_str(), dev.snvme_dev_path.c_str(),
        dev.namespace_id, dev.queue_depth,
        dev.max_user_qid, dev.kernel_io_qps, dev.user_io_qps,
        dev.max_user_qid, dev.max_queues_per_group);
    if (tutti_verbose()) {
        bool first = true;
        for (int gid : dev.allowed_gpus) {
            std::fprintf(stderr, "%s%d", first ? "" : ",", gid);
            first = false;
        }
        std::fprintf(stderr, "}\n");
    }

    devices_.push_back(std::move(dev));
}

// ---------------------------------------------------------------------------
// GPU-view symlink lifecycle
// ---------------------------------------------------------------------------

bool ServiceState::publish_gpu_views(int32_t device_id) {
    std::lock_guard<std::mutex> lock(state_mtx_);
    if (device_id < 0 || device_id >= static_cast<int32_t>(devices_.size())) {
        std::fprintf(stderr,
            "warning: cannot publish GPU views for invalid device_id=%d\n",
            device_id);
        return false;
    }
    return install_gpu_symlinks(devices_[device_id]);
}

void ServiceState::unpublish_gpu_views() {
    std::lock_guard<std::mutex> lock(state_mtx_);
    for (auto& dev : devices_) {
        remove_gpu_symlinks(dev);
    }
}

bool ServiceState::install_gpu_symlinks(DeviceState& dev) {
    namespace fs = std::filesystem;
    bool all_installed = true;

    // Symlink name under each consuming GPU's mount_path is the
    // basename of the snvme device node.
    std::string link_name;
    if (!dev.snvme_dev_path.empty()) {
        link_name = fs::path(dev.snvme_dev_path).filename().string();
    }
    if (link_name.empty()) {
        link_name = dev.pci_addr;
        for (auto& c : link_name) {
            if (c == ':' || c == '.' || c == '/') c = '_';
        }
    }

    for (int gpu_id : dev.allowed_gpus) {
        const std::string* gpu_mount = gpu_mount_for(gpu_id);
        if (gpu_mount == nullptr || gpu_mount->empty()) {
            std::fprintf(stderr,
                "warning: no GPU view root configured for gpu_id=%d\n",
                gpu_id);
            all_installed = false;
            continue;
        }

        const fs::path nvme_subdir = fs::path(dev.mount_path) /
            ("GPU" + std::to_string(gpu_id));
        const fs::path link_path   = fs::path(*gpu_mount) / link_name;

        std::error_code ec;
        fs::create_directories(nvme_subdir, ec);
        if (ec) {
            std::fprintf(stderr,
                "warning: mkdir %s failed: %s\n",
                nvme_subdir.c_str(), ec.message().c_str());
            all_installed = false;
            continue;
        }
        dev.created_nvme_subdirs.push_back(nvme_subdir.string());

        fs::create_directories(*gpu_mount, ec);
        if (ec) {
            std::fprintf(stderr,
                "warning: mkdir %s failed: %s\n",
                gpu_mount->c_str(), ec.message().c_str());
            all_installed = false;
            continue;
        }

        std::error_code link_ec;
        const fs::file_status st = fs::symlink_status(link_path, link_ec);
        if (st.type() == fs::file_type::symlink) {
            const fs::path existing = fs::read_symlink(link_path, link_ec);
            if (!link_ec && existing == nvme_subdir) {
                dev.created_symlinks.push_back(link_path.string());
                dev.gpu_view_paths[gpu_id] = link_path.string();
                continue;
            }
            fs::remove(link_path, link_ec);
        } else if (fs::exists(st)) {
            std::fprintf(stderr,
                "warning: %s exists and is not a symlink; refusing to overwrite\n",
                link_path.c_str());
            all_installed = false;
            continue;
        }

        fs::create_symlink(nvme_subdir, link_path, link_ec);
        if (link_ec) {
            std::fprintf(stderr,
                "warning: symlink %s -> %s failed: %s\n",
                link_path.c_str(), nvme_subdir.c_str(), link_ec.message().c_str());
            all_installed = false;
            continue;
        }
        dev.created_symlinks.push_back(link_path.string());
        dev.gpu_view_paths[gpu_id] = link_path.string();
    }
    return all_installed;
}

void ServiceState::remove_gpu_symlinks(DeviceState& dev) {
    namespace fs = std::filesystem;
    std::error_code ec;

    for (const auto& link : dev.created_symlinks) {
        fs::remove(link, ec);
    }
    dev.created_symlinks.clear();

    for (const auto& sub : dev.created_nvme_subdirs) {
        ::rmdir(sub.c_str());
    }
    dev.created_nvme_subdirs.clear();
    dev.gpu_view_paths.clear();
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

std::vector<DeviceSnapshot> ServiceState::list_devices() const {
    std::lock_guard<std::mutex> lock(state_mtx_);
    std::vector<DeviceSnapshot> out;
    out.reserve(devices_.size());
    for (const auto& d : devices_) {
        DeviceSnapshot s;
        s.device_id            = d.device_id;
        s.pci_addr             = d.pci_addr;
        s.snvme_dev_path       = d.snvme_dev_path;
        s.namespace_id         = d.namespace_id;
        s.page_size            = d.page_size;
        s.blk_size             = d.blk_size;
        s.blk_size_log         = d.blk_size_log;
        s.queue_depth          = d.queue_depth;
        s.dstrd                = d.dstrd;
        s.bar0_size            = d.bar0_size;
        s.max_user_qid         = d.max_user_qid;
        s.kernel_io_qps        = d.kernel_io_qps;
        s.user_io_qps          = d.user_io_qps;
        s.max_queues_per_group = d.max_queues_per_group;

        // Stable order for human readability: sort by gpu id.
        std::vector<int> sorted_ids(d.allowed_gpus.begin(), d.allowed_gpus.end());
        std::sort(sorted_ids.begin(), sorted_ids.end());
        s.allowed_gpus.reserve(sorted_ids.size());
        for (int gid : sorted_ids) {
            AllowedGpuView v;
            v.cuda_device = gid;
            auto it = d.gpu_view_paths.find(gid);
            if (it != d.gpu_view_paths.end()) v.mount_path = it->second;
            s.allowed_gpus.push_back(std::move(v));
        }
        out.push_back(std::move(s));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Connect / Disconnect / Heartbeat
// ---------------------------------------------------------------------------

ServiceState::ConnectResult ServiceState::connect(int32_t device_id,
                                                   int32_t cuda_device,
                                                   int32_t num_queues,
                                                   uint32_t client_pid) {
    ConnectResult result;

    std::lock_guard<std::mutex> lock(state_mtx_);

    if (device_id < 0 || device_id >= static_cast<int32_t>(devices_.size())) {
        result.error = "invalid device_id";
        return result;
    }
    DeviceState& dev = devices_[device_id];

    if (dev.allowed_gpus.find(cuda_device) == dev.allowed_gpus.end()) {
        result.error = "cuda_device=" + std::to_string(cuda_device) +
                       " not in allowed_gpus for device_id=" +
                       std::to_string(device_id);
        return result;
    }

    int32_t want = (num_queues > 0) ? num_queues : cfg_.queue_pool.default_per_client;
    want = std::min(want, cfg_.queue_pool.max_per_client);
    // Kernel hard cap: a single per-fd group cannot have more than
    // max_queues_per_group queues.  Clamp so we don't promise the
    // client more than libnvm will let them open.
    if (dev.max_queues_per_group > 0) {
        want = std::min(want, static_cast<int32_t>(dev.max_queues_per_group));
    }
    if (want <= 0) {
        result.error = "queue_pool policy yielded 0 queues";
        return result;
    }

    Allocation alloc;
    alloc.allocation_id        = generate_allocation_id();
    alloc.device_id            = device_id;
    alloc.cuda_device          = cuda_device;
    alloc.granted_queues       = want;
    alloc.client_pid           = client_pid;
    alloc.client_pid_starttime = read_pid_starttime(client_pid).value_or(0);
    alloc.last_heartbeat       = std::chrono::steady_clock::now();

    const std::string aid = alloc.allocation_id;
    allocations_.emplace(aid, std::move(alloc));

    ConnectGrant& g = result.grant;
    g.allocation_id          = aid;
    g.device_id              = device_id;
    g.pci_addr               = dev.pci_addr;
    g.snvme_dev_path         = dev.snvme_dev_path;
    g.bar0_size              = dev.bar0_size;
    g.dstrd                  = dev.dstrd;
    g.granted_queues         = want;
    g.namespace_id           = dev.namespace_id;
    g.page_size              = dev.page_size;
    g.blk_size               = dev.blk_size;
    g.blk_size_log           = dev.blk_size_log;
    g.queue_depth            = dev.queue_depth;
    g.max_data_size          = dev.max_data_size;
    auto it = dev.gpu_view_paths.find(cuda_device);
    if (it != dev.gpu_view_paths.end()) g.mount_path = it->second;
    g.heartbeat_interval_sec = cfg_.lease.heartbeat_interval_sec;
    g.lease_timeout_sec      = cfg_.lease.timeout_sec;

    result.success = true;
    return result;
}

bool ServiceState::disconnect(const std::string& allocation_id,
                               uint32_t client_pid,
                               std::string* error) {
    std::lock_guard<std::mutex> lock(state_mtx_);

    auto it = allocations_.find(allocation_id);
    if (it == allocations_.end()) {
        if (error) *error = "unknown allocation_id";
        return false;
    }

    const Allocation& alloc = it->second;
    if (alloc.client_pid != client_pid) {
        if (error) *error = "pid mismatch (recorded=" +
            std::to_string(alloc.client_pid) + " got=" +
            std::to_string(client_pid) + ")";
        return false;
    }

    // Nothing to refund: the kernel handles real queue / map
    // teardown via fd close.  Just drop the lease record.
    allocations_.erase(it);
    return true;
}

bool ServiceState::update_heartbeat(const std::string& allocation_id,
                                      std::string* error) {
    std::lock_guard<std::mutex> lock(state_mtx_);

    auto it = allocations_.find(allocation_id);
    if (it == allocations_.end()) {
        if (error) *error = "unknown allocation_id";
        return false;
    }
    it->second.last_heartbeat = std::chrono::steady_clock::now();
    return true;
}

bool ServiceState::has_allocation(const std::string& allocation_id) const {
    std::lock_guard<std::mutex> lock(state_mtx_);
    return allocations_.find(allocation_id) != allocations_.end();
}

// ---------------------------------------------------------------------------
// Reaper
// ---------------------------------------------------------------------------

void ServiceState::start_reaper() {
    if (reaper_running_.exchange(true)) return;
    reaper_thread_ = std::thread(&ServiceState::reaper_loop, this);
}

void ServiceState::stop_reaper() {
    if (!reaper_running_.exchange(false)) return;
    if (reaper_thread_.joinable()) {
        reaper_thread_.join();
    }
}

bool ServiceState::is_pid_dead(uint32_t pid, uint64_t recorded_starttime) const {
    if (::kill(static_cast<pid_t>(pid), 0) != 0) return true;
    auto st = read_pid_starttime(pid);
    if (!st.has_value()) return true;
    return st.value() != recorded_starttime;
}

void ServiceState::reaper_loop() {
    const auto tick = std::chrono::seconds(std::max<uint32_t>(
        1, cfg_.lease.heartbeat_interval_sec / 2));
    const auto timeout = std::chrono::seconds(cfg_.lease.timeout_sec);

    while (reaper_running_.load()) {
        std::this_thread::sleep_for(tick);
        if (!reaper_running_.load()) break;

        const auto now = std::chrono::steady_clock::now();
        std::vector<std::string> to_reap;

        std::lock_guard<std::mutex> lock(state_mtx_);
        for (const auto& kv : allocations_) {
            const Allocation& alloc = kv.second;
            if (now - alloc.last_heartbeat <= timeout) continue;

            if (is_pid_dead(alloc.client_pid, alloc.client_pid_starttime)) {
                to_reap.push_back(alloc.allocation_id);
            }
            // PID alive but silent: leave the allocation in place.
        }

        for (const auto& aid : to_reap) {
            auto it = allocations_.find(aid);
            if (it == allocations_.end()) continue;
            const Allocation& alloc = it->second;
            // Reaper diagnostic — info level, gated by TUTTI_VERBOSE.
            TUTTI_INFO(
                "nvmeservice reaper: dropped lease device=%d cuda_device=%d "
                "(pid=%u, allocation_id=%s); kernel fd-close already reclaimed "
                "the actual queues / DATA maps\n",
                alloc.device_id, alloc.cuda_device, alloc.client_pid,
                aid.c_str());
            allocations_.erase(it);
        }
    }
}

} // namespace nvmeservice

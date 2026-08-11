#include <tutti/config/tutti_config.h>
#include <tutti/bindings/ext4_local_nvme/binding.h>
#include <tutti/bindings/striped_local_nvme/binding.h>
#include <tutti/cuda_like.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>
#include <tutti/storage_runtime.h>

#include "nvmeservice_client.h"
#include "tutti/data_paths/striped_local_nvme/striped_data_path.h"
#include "tutti/resource/nvme/nvme_resource_internal.h"
#include "tutti/tutti_runtime/tutti_runtime_internal.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

extern "C" void phase5_fill_value(void* buffer, unsigned char value,
                                   std::uint64_t size, void* stream);
extern "C" void phase5_fill_position(void* buffer, std::uint64_t base,
                                      std::uint64_t size, void* stream);

namespace {

using tutti::ExecutionDomain;
using tutti::HostSubmitContext;
using tutti::IoDirection;
using tutti::IoHandle;
using tutti::IoRequest;
using tutti::IoRequestState;
using tutti::IoState;
using tutti::MemoryHandle;
using tutti::MemoryKind;
using tutti::MemoryOwnership;
using tutti::MemoryView;
using tutti::OpenOptions;
using tutti::Status;
using tutti::StorageRuntime;
using tutti::TargetHandle;
using tutti::config::TuttiRuntime;
using tutti::config::TuttiRuntimeTestingAccess;
using tutti::data_paths::striped_local_nvme::DeviceDescriptor;
using tutti::data_paths::striped_local_nvme::StripedDataPath;
using tutti::resources::nvme::NvmeResource;
using tutti::resources::nvme::NvmeResourceInspection;
using tutti::resources::nvme::NvmeResourceTestingAccess;
using tutti::resources::nvme::RuntimeNvmeSlice;
using nvmeservice::ClientAcceleratorInfo;
using nvmeservice::ClientNvmeResource;
using nvmeservice::NvmeServiceClient;

constexpr int kSkip = 77;
constexpr std::uint64_t kStripeUnit = 65536;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const std::string& message) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

struct Options {
    std::string endpoint = "127.0.0.1:50051";
    std::int32_t accel0 = 0;
    std::int32_t accel1 = 1;
    std::int32_t device0 = 0;
    std::int32_t device1 = 1;
    std::int32_t queues = 4;
    bool single_accelerator = false;
};

bool parse_i32(const char* text, std::int32_t& value) {
    if (text == nullptr || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed < INT32_MIN || parsed > INT32_MAX) {
        return false;
    }
    value = static_cast<std::int32_t>(parsed);
    return true;
}

void usage(const char* program) {
    std::fprintf(stderr,
        "Usage: %s [--endpoint HOST:PORT] [--accel0 ID] [--accel1 ID] "
        "[--device0 ID] [--device1 ID] [--queues N] "
        "[--single-accelerator]\n", program);
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--single-accelerator") {
            options.single_accelerator = true;
            continue;
        }
        if (i + 1 >= argc) return false;
        const char* value = argv[++i];
        if (arg == "--endpoint") {
            options.endpoint = value;
        } else if (arg == "--accel0") {
            if (!parse_i32(value, options.accel0)) return false;
        } else if (arg == "--accel1") {
            if (!parse_i32(value, options.accel1)) return false;
        } else if (arg == "--device0") {
            if (!parse_i32(value, options.device0)) return false;
        } else if (arg == "--device1") {
            if (!parse_i32(value, options.device1)) return false;
        } else if (arg == "--queues") {
            if (!parse_i32(value, options.queues)) return false;
        } else {
            return false;
        }
    }
    return options.queues > 0 && options.device0 != options.device1 &&
           (options.single_accelerator || options.accel0 != options.accel1);
}

template <typename Function>
auto call_preserving_device(const std::string& api, std::int32_t caller_device,
                            Function&& function) -> decltype(function()) {
    int before = -1;
    const cudaError_t before_status = cudaGetDevice(&before);
    check(before_status == cudaSuccess && before == caller_device,
          api + ": caller current device precondition");
    auto result = function();
    int after = -1;
    const cudaError_t after_status = cudaGetDevice(&after);
    check(after_status == cudaSuccess && after == caller_device,
          api + ": restored caller current device");
    return result;
}

class TempConfigDirectory {
public:
    bool create() {
        char path[] = "/tmp/tutti_phase7_XXXXXX";
        char* created = ::mkdtemp(path);
        if (created == nullptr) return false;
        path_ = created;
        return true;
    }

    std::string write(const std::string& name, const Options& options,
                      std::int32_t accel_id, const std::string& selection,
                      const std::vector<std::int32_t>& device_ids) {
        const std::string path = path_ + "/" + name + ".yaml";
        const bool striped = selection == "striped";
        std::ofstream output(path, std::ios::trunc);
        output << "accelerator:\n"
               << "  profile: \"CUDA\"\n\n"
               << "runtime:\n"
               << "  accel_id: " << accel_id << "\n\n"
               << "storage:\n"
               << "  resources:\n"
               << "    - id: nvme-resource\n"
               << "      type: nvme\n"
               << "      provider:\n"
               << "        type: nvme-service\n"
               << "        endpoint: \"" << options.endpoint << "\"\n"
               << "      allocation:\n"
               << "        selection: \"" << selection << "\"\n"
               << "        device_ids: [";
        for (std::size_t i = 0; i < device_ids.size(); ++i) {
            if (i != 0) output << ", ";
            output << device_ids[i];
        }
        output << "]\n"
               << "        queues_per_controller: " << options.queues << "\n"
               << "  resolvers:\n"
               << "    - id: storage-resolver\n"
               << "      type: "
               << (striped ? "striped-file" : "local-file") << "\n"
               << "      scheme: " << (striped ? "striped" : "file") << "\n"
               << "      config: {}\n"
               << "  datapaths:\n"
               << "    - id: storage-datapath\n"
               << "      type: "
               << (striped ? "striped-local-nvme" : "local-nvme") << "\n"
               << "      config: {}\n"
               << "  backends:\n"
               << "    - id: storage-backend\n"
               << "      contract: "
               << (striped ? "striped-local-nvme" : "ext4-local-nvme")
               << "\n"
               << "      resolver: storage-resolver\n"
               << "      datapath: storage-datapath\n"
               << "      resource: nvme-resource\n"
               << "      config: ";
        if (striped) {
            output << "{stripe_unit: " << kStripeUnit << "}\n";
        } else {
            output << "{}\n";
        }
        output.close();
        if (!output) return {};
        files_.push_back(path);
        std::printf("CONFIG path=%s accel=%d selection=%s\n",
                    path.c_str(), accel_id, selection.c_str());
        return path;
    }

    ~TempConfigDirectory() {
        for (const std::string& file : files_) (void)::unlink(file.c_str());
        (void)::rmdir(path_.c_str());
    }

private:
    std::string path_;
    std::vector<std::string> files_;
};

using ResourceMap = std::map<std::int32_t, ClientNvmeResource>;
using AcceleratorMap = std::map<std::int32_t, ClientAcceleratorInfo>;

AcceleratorMap snapshot_accelerators(NvmeServiceClient& client) {
    AcceleratorMap result;
    for (auto& accelerator : client.list_accelerators()) {
        result.emplace(accelerator.accel_id, std::move(accelerator));
    }
    return result;
}

ResourceMap snapshot_resources(NvmeServiceClient& client) {
    ResourceMap result;
    for (auto& resource : client.list_nvme_resources()) {
        result.emplace(resource.device_id, std::move(resource));
    }
    return result;
}

std::string join_ids(const std::vector<std::int32_t>& ids) {
    std::ostringstream output;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) output << ',';
        output << ids[index];
    }
    return output.str();
}

void print_snapshot(const char* label, const ResourceMap& resources) {
    std::printf("NVME_SNAPSHOT %s\n", label);
    for (const auto& row : resources) {
        const auto& resource = row.second;
        std::printf(
            "  device=%d pci=%s chrdev=%s block=%s backing=%s ns=%u lba=%u "
            "bar0=%llu mdts=%llu capacity_queues=%u reserved=%u available=%u "
            "allowed_accels=%s available_flag=%s\n",
            resource.device_id, resource.pci_bdf.c_str(),
            resource.chrdev_path.c_str(), resource.block_path.c_str(),
            resource.backing_mount_path.c_str(), resource.namespace_id,
            resource.logical_block_size,
            static_cast<unsigned long long>(resource.bar0_size),
            static_cast<unsigned long long>(resource.max_data_size),
            resource.controller_queue_capacity, resource.reserved_queues,
            resource.available_queues,
            join_ids(resource.allowed_accel_ids).c_str(),
            resource.available ? "true" : "false");
    }
}

void print_accelerator_snapshot(const AcceleratorMap& accelerators) {
    std::printf("ACCELERATOR_SNAPSHOT baseline\n");
    for (const auto& row : accelerators) {
        std::printf("  accel=%d view_root=%s\n", row.second.accel_id,
                    row.second.view_root.c_str());
    }
}

bool same_ledger(const ResourceMap& lhs, const ResourceMap& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (const auto& row : lhs) {
        const auto it = rhs.find(row.first);
        if (it == rhs.end() ||
            row.second.reserved_queues != it->second.reserved_queues ||
            row.second.available_queues != it->second.available_queues) {
            return false;
        }
    }
    return true;
}

bool wait_for_baseline(NvmeServiceClient& client, const ResourceMap& baseline,
                       ResourceMap& observed) {
    for (int attempt = 0; attempt < 30; ++attempt) {
        observed = snapshot_resources(client);
        if (same_ledger(baseline, observed)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

bool check_scenario_baseline(NvmeServiceClient& client,
                             const ResourceMap& baseline,
                             const std::string& label) {
    const ResourceMap observed = snapshot_resources(client);
    print_snapshot((label + " baseline").c_str(), observed);
    const bool clean = same_ledger(baseline, observed);
    check(clean, label + ": daemon ledger starts at exact baseline");
    return clean;
}

bool static_config_is_logical(
    const std::string& path, const Options& options,
    std::int32_t accel_id, tutti::config::NvmeSelection selection,
    const std::vector<std::int32_t>& device_ids,
    const ResourceMap& daemon_snapshot, const std::string& label) {
    const auto parsed = tutti::config::parse_tutti_config(path);
    bool ok = parsed.ok();
    if (parsed.ok()) {
        const auto& config = parsed.value();
        const auto& storage = config.canonical_storage;
        ok = ok && config.runtime_accel_id == accel_id && storage.present &&
             storage.resources.size() == 1 && storage.resolvers.size() == 1 &&
             storage.datapaths.size() == 1 && storage.backends.size() == 1;
        if (!storage.resources.empty()) {
            const auto& resource = storage.resources.front();
            ok = ok && resource.provider.endpoint == options.endpoint &&
                 resource.allocation.selection == selection &&
                 resource.allocation.device_ids == device_ids &&
                 resource.allocation.queues_per_controller == options.queues;
        }
    }

    std::ifstream input(path);
    const bool opened = input.is_open();
    const std::string yaml((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    const std::vector<std::string> forbidden_fields = {
        "pci_bdf:", "pci_addr:", "chrdev_path:", "block_path:",
        "backing_mount_path:", "view_path:", "mount_path:",
        "allocation_id:", "lease:", "granted_queues:",
        "allowed_accel_ids:"};
    ok = ok && opened && !input.bad();
    for (const std::string& field : forbidden_fields) {
        ok = ok && yaml.find(field) == std::string::npos;
    }
    for (const auto& row : daemon_snapshot) {
        const auto& resource = row.second;
        for (const std::string* fact : {
                 &resource.pci_bdf, &resource.chrdev_path,
                 &resource.block_path, &resource.backing_mount_path}) {
            ok = ok && (fact->empty() || yaml.find(*fact) == std::string::npos);
        }
    }
    check(ok, label + ": canonical static config contains only logical facts");
    return ok;
}

bool check_acquired_ledger(const ResourceMap& baseline,
                           const ResourceMap& acquired,
                           const std::vector<std::int32_t>& selected,
                           std::uint32_t queues) {
    bool ok = baseline.size() == acquired.size();
    for (const auto& row : baseline) {
        const auto it = acquired.find(row.first);
        if (it == acquired.end()) {
            ok = false;
            continue;
        }
        const bool is_selected =
            std::find(selected.begin(), selected.end(), row.first) != selected.end();
        const std::uint32_t delta = is_selected ? queues : 0;
        ok = ok && it->second.reserved_queues ==
                       row.second.reserved_queues + delta;
        ok = ok && it->second.available_queues + delta ==
                       row.second.available_queues;
    }
    check(ok, "Acquire reserves exactly the requested queues");
    return ok;
}

bool paths_share_filesystem(const std::string& lhs, const std::string& rhs) {
    struct stat lhs_stat {};
    struct stat rhs_stat {};
    return ::stat(lhs.c_str(), &lhs_stat) == 0 &&
           ::stat(rhs.c_str(), &rhs_stat) == 0 &&
           S_ISDIR(lhs_stat.st_mode) && S_ISDIR(rhs_stat.st_mode) &&
           lhs_stat.st_dev == rhs_stat.st_dev;
}

bool path_is_within(const std::string& path, const std::string& root) {
    if (path == root) return true;
    return !root.empty() && path.size() > root.size() &&
           path.compare(0, root.size(), root) == 0 && path[root.size()] == '/';
}

NvmeResourceInspection inspect_nvme_resource(const TuttiRuntime& bundle) {
    const auto* resource = dynamic_cast<const NvmeResource*>(
        TuttiRuntimeTestingAccess::resource(bundle));
    if (resource == nullptr) return {};
    return NvmeResourceTestingAccess::inspection(*resource);
}

bool validate_metadata(const TuttiRuntime& bundle,
                       const AcceleratorMap& accelerators,
                       const ResourceMap& baseline,
                       std::int32_t accel_id,
                       const std::vector<std::int32_t>& selected,
                       std::uint32_t queues) {
    const auto inspection = inspect_nvme_resource(bundle);
    bool ok = inspection.lease_state ==
                  tutti::resources::nvme::NvmeLeaseState::ACQUIRED &&
              !inspection.allocation.allocation_id.empty() &&
              inspection.allocation.slices.size() == selected.size();
    const auto resource_info = bundle.resource_info("nvme-resource");
    const auto manifests = bundle.backend_manifests();
    ok = ok && resource_info.ok() &&
         resource_info.value().id == "nvme-resource" &&
         resource_info.value().type == "nvme" &&
         resource_info.value().state == tutti::ResourceState::INITIALIZED &&
         manifests.size() == 1 &&
         manifests.front().id == "storage-backend" &&
         manifests.front().resolver_id == "storage-resolver" &&
         manifests.front().datapath_id == "storage-datapath" &&
         manifests.front().resource_id == "nvme-resource" &&
         bundle.storage_runtime() != nullptr;
    const auto accelerator_it = accelerators.find(accel_id);
    ok = ok && accelerator_it != accelerators.end();
    for (std::size_t i = 0; i < inspection.allocation.slices.size(); ++i) {
        const RuntimeNvmeSlice& slice = inspection.allocation.slices[i];
        const auto resource_it = baseline.find(slice.device_id);
        if (resource_it == baseline.end()) {
            ok = false;
            continue;
        }
        const ClientNvmeResource& resource = resource_it->second;
        ok = ok && i < selected.size() && slice.device_id == selected[i];
        ok = ok && slice.accel_id == accel_id;
        ok = ok && slice.pci_bdf == resource.pci_bdf;
        ok = ok && slice.chrdev_path == resource.chrdev_path;
        ok = ok && slice.block_path == resource.block_path;
        ok = ok && slice.backing_mount_path == resource.backing_mount_path;
        ok = ok && slice.namespace_id == resource.namespace_id;
        ok = ok && slice.logical_block_size == resource.logical_block_size;
        ok = ok && slice.bar0_size == resource.bar0_size;
        ok = ok && slice.max_data_size == resource.max_data_size;
        ok = ok && slice.granted_queues == queues;
        ok = ok && slice.allowed_accel_ids == resource.allowed_accel_ids;
        ok = ok && !slice.view_path.empty() &&
             paths_share_filesystem(slice.view_path,
                                    slice.backing_mount_path);
        if (accelerator_it != accelerators.end()) {
            ok = ok && path_is_within(slice.view_path,
                                      accelerator_it->second.view_root);
        }
        std::printf(
            "SLICE allocation=%s device=%d accel=%d pci=%s chrdev=%s block=%s "
            "backing=%s view=%s ns=%u lba=%u bar0=%llu mdts=%llu grant=%u\n",
            inspection.allocation.allocation_id.c_str(), slice.device_id,
            slice.accel_id,
            slice.pci_bdf.c_str(), slice.chrdev_path.c_str(),
            slice.block_path.c_str(), slice.backing_mount_path.c_str(),
            slice.view_path.c_str(), slice.namespace_id,
            slice.logical_block_size,
            static_cast<unsigned long long>(slice.bar0_size),
            static_cast<unsigned long long>(slice.max_data_size),
            slice.granted_queues);
    }
    check(ok, "allocation metadata matches daemon resource facts");
    return ok;
}

bool create_direct_file(const std::string& path, std::uint64_t size,
                        std::uint32_t alignment) {
    const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_DIRECT,
                          0600);
    if (fd < 0) return false;
    const std::size_t chunk_size = 1024 * 1024;
    void* buffer = nullptr;
    const std::size_t effective_alignment = std::max<std::size_t>(4096, alignment);
    if (::posix_memalign(&buffer, effective_alignment, chunk_size) != 0) {
        ::close(fd);
        ::unlink(path.c_str());
        return false;
    }
    std::memset(buffer, 0, chunk_size);
    bool ok = true;
    std::uint64_t written = 0;
    while (written < size) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(chunk_size, size - written));
        if (::write(fd, buffer, count) != static_cast<ssize_t>(count)) {
            ok = false;
            break;
        }
        written += count;
    }
    ok = ok && ::fsync(fd) == 0;
    std::free(buffer);
    ::close(fd);
    if (!ok) ::unlink(path.c_str());
    return ok;
}

struct GpuBuffer {
    void* raw = nullptr;
    void* aligned = nullptr;
    std::uint64_t size = 0;
    cudaStream_t stream = nullptr;
};

bool allocate_gpu_buffer(std::int32_t accel_id, std::int32_t caller_device,
                         std::uint64_t size, GpuBuffer& buffer) {
    if (cudaSetDevice(accel_id) != cudaSuccess) return false;
    buffer.size = size;
    if (cudaMalloc(&buffer.raw, size + kStripeUnit) != cudaSuccess) return false;
    buffer.aligned = reinterpret_cast<void*>(
        (reinterpret_cast<std::uintptr_t>(buffer.raw) + kStripeUnit - 1) &
        ~(static_cast<std::uintptr_t>(kStripeUnit) - 1));
    if (cudaStreamCreate(&buffer.stream) != cudaSuccess) {
        cudaFree(buffer.raw);
        buffer = {};
        return false;
    }
    return cudaSetDevice(caller_device) == cudaSuccess;
}

void free_gpu_buffer(std::int32_t accel_id, std::int32_t caller_device,
                     GpuBuffer& buffer) {
    (void)cudaSetDevice(accel_id);
    if (buffer.stream != nullptr) (void)cudaStreamDestroy(buffer.stream);
    if (buffer.raw != nullptr) (void)cudaFree(buffer.raw);
    buffer = {};
    (void)cudaSetDevice(caller_device);
}

bool fill_value(GpuBuffer& buffer, std::int32_t accel_id,
                std::int32_t caller_device, std::uint64_t offset,
                std::uint64_t size, unsigned char value) {
    if (cudaSetDevice(accel_id) != cudaSuccess) return false;
    phase5_fill_value(static_cast<unsigned char*>(buffer.aligned) + offset,
                      value, size, buffer.stream);
    const bool ok = cudaGetLastError() == cudaSuccess &&
                    cudaStreamSynchronize(buffer.stream) == cudaSuccess;
    return cudaSetDevice(caller_device) == cudaSuccess && ok;
}

bool fill_position(GpuBuffer& buffer, std::int32_t accel_id,
                   std::int32_t caller_device, std::uint64_t offset,
                   std::uint64_t size, std::uint64_t base) {
    if (cudaSetDevice(accel_id) != cudaSuccess) return false;
    phase5_fill_position(static_cast<unsigned char*>(buffer.aligned) + offset,
                         base, size, buffer.stream);
    const bool ok = cudaGetLastError() == cudaSuccess &&
                    cudaStreamSynchronize(buffer.stream) == cudaSuccess;
    return cudaSetDevice(caller_device) == cudaSuccess && ok;
}

bool verify_value(GpuBuffer& buffer, std::int32_t accel_id,
                  std::int32_t caller_device, std::uint64_t offset,
                  std::uint64_t size, unsigned char value) {
    std::vector<unsigned char> host(size);
    if (cudaSetDevice(accel_id) != cudaSuccess) return false;
    const bool copied = cudaMemcpy(
        host.data(), static_cast<unsigned char*>(buffer.aligned) + offset,
        size, cudaMemcpyDeviceToHost) == cudaSuccess;
    const bool restored = cudaSetDevice(caller_device) == cudaSuccess;
    return copied && restored &&
           std::all_of(host.begin(), host.end(),
                       [value](unsigned char byte) { return byte == value; });
}

bool verify_position(GpuBuffer& buffer, std::int32_t accel_id,
                     std::int32_t caller_device, std::uint64_t offset,
                     std::uint64_t size, std::uint64_t base) {
    std::vector<unsigned char> host(size);
    if (cudaSetDevice(accel_id) != cudaSuccess) return false;
    const bool copied = cudaMemcpy(
        host.data(), static_cast<unsigned char*>(buffer.aligned) + offset,
        size, cudaMemcpyDeviceToHost) == cudaSuccess;
    const bool restored = cudaSetDevice(caller_device) == cudaSuccess;
    if (!copied || !restored) return false;
    for (std::uint64_t i = 0; i < size; ++i) {
        if (host[i] != static_cast<unsigned char>((base + i) % 251u)) {
            return false;
        }
    }
    return true;
}

bool submit_and_wait(StorageRuntime& runtime, const IoRequest* requests,
                     std::size_t count, const HostSubmitContext& context,
                     std::int32_t caller_device, const std::string& label) {
    auto submitted = call_preserving_device(
        label + "/submit", caller_device,
        [&] { return runtime.submit(requests, count, context); });
    bool ok = submitted.status.ok() && submitted.io.has_value() &&
              submitted.initial_states.size() == count;
    for (const auto& state : submitted.initial_states) {
        ok = ok && state.state == IoRequestState::ACCEPTED && state.status.ok();
    }
    check(ok, label + ": all requests accepted");
    if (!submitted.io.has_value()) return false;
    const IoHandle io = *submitted.io;
    auto progress = call_preserving_device(
        label + "/progress", caller_device,
        [&] { return runtime.query(io); });
    check(progress.ok(), label + ": progress observation succeeds");
    auto waited = call_preserving_device(
        label + "/wait", caller_device,
        [&] { return runtime.wait(io, 30000); });
    ok = ok && waited.observation_status.ok() && waited.result.has_value() &&
         waited.result->state == IoState::COMPLETED && waited.result->status.ok();
    check(ok, label + ": wait reaches COMPLETED");
    Status released = call_preserving_device(
        label + "/release", caller_device,
        [&] { return runtime.release_io(io); });
    check(released.ok(), label + ": terminal I/O released");
    return ok && released.ok();
}

bool cleanup_runtime_objects(TuttiRuntime& bundle, TargetHandle target,
                             MemoryHandle memory, std::int32_t caller_device,
                             const std::string& label) {
    Status closed = call_preserving_device(
        label + "/close", caller_device,
        [&] { return bundle.storage_runtime()->close(target); });
    check(closed.ok(), label + ": target closed");
    Status unregistered = call_preserving_device(
        label + "/unregister", caller_device,
        [&] { return bundle.storage_runtime()->unregister_memory(memory); });
    check(unregistered.ok(), label + ": memory unregistered");
    return closed.ok() && unregistered.ok();
}

std::unique_ptr<TuttiRuntime> load_bundle(
    const std::string& config_path, std::int32_t caller_device,
    const std::string& label) {
    auto loaded = call_preserving_device(
        label + "/initialize", caller_device,
        [&] { return tutti::config::load_tutti_config(config_path); });
    check(loaded.ok(), label + ": load_tutti_config succeeds");
    if (!loaded.ok()) {
        std::fprintf(stderr, "%s: %s\n", label.c_str(),
                     loaded.status().message().c_str());
        return nullptr;
    }
    return std::move(loaded).value();
}

bool shutdown_and_check_ledger(std::unique_ptr<TuttiRuntime>& bundle,
                               std::int32_t caller_device,
                               NvmeServiceClient& observer,
                               const ResourceMap& baseline,
                               const std::string& label) {
    Status first = call_preserving_device(
        label + "/shutdown-1", caller_device,
        [&] { return bundle->shutdown(); });
    check(first.ok(), label + ": first bundle shutdown succeeds");
    Status second = call_preserving_device(
        label + "/shutdown-2", caller_device,
        [&] { return bundle->shutdown(); });
    check(second.ok(), label + ": second bundle shutdown is idempotent");
    bundle.reset();
    ResourceMap released;
    const bool restored = wait_for_baseline(observer, baseline, released);
    print_snapshot((label + " released").c_str(), released);
    check(restored, label + ": Release restores exact ledger baseline");
    return first.ok() && second.ok() && restored;
}

bool run_local_scenario(const Options& options, const std::string& config_path,
                        std::int32_t accel_id, std::int32_t caller_device,
                        std::int32_t device_id, NvmeServiceClient& observer,
                        const AcceleratorMap& accelerators,
                        const ResourceMap& baseline, const std::string& label) {
    std::printf("\n=== %s accel=%d device=%d caller=%d ===\n",
                label.c_str(), accel_id, device_id, caller_device);
    (void)cudaSetDevice(caller_device);
    bool ok = check_scenario_baseline(observer, baseline, label);
    ok = static_config_is_logical(
             config_path, options, accel_id,
             tutti::config::NvmeSelection::Explicit, {device_id}, baseline,
             label) && ok;
    auto bundle = load_bundle(config_path, caller_device, label);
    if (!bundle) return false;
    ok = validate_metadata(*bundle, accelerators, baseline, accel_id,
                           {device_id},
                           static_cast<std::uint32_t>(options.queues)) && ok;
    const auto manifest = bundle->backend_manifest("storage-backend");
    const bool manifest_ok = manifest.ok() &&
        manifest.value().contract == "ext4-local-nvme";
    check(manifest_ok,
          label + ": loader publishes one Local top-level binding");
    ok = manifest_ok && ok;
    const ResourceMap acquired = snapshot_resources(observer);
    print_snapshot((label + " acquired").c_str(), acquired);
    ok = check_acquired_ledger(baseline, acquired, {device_id}, options.queues) && ok;

    const auto slices = inspect_nvme_resource(*bundle).allocation.slices;
    if (slices.empty()) {
        (void)shutdown_and_check_ledger(bundle, caller_device, observer,
                                        baseline, label);
        return false;
    }
    const RuntimeNvmeSlice slice = slices.front();
    const std::string scratch = slice.view_path + "/tutti_phase7_" +
        std::to_string(::getpid()) + "_" + label + ".bin";
    std::printf("SCRATCH scenario=%s path=%s\n",
                label.c_str(), scratch.c_str());
    const std::uint64_t io_size = 4 * kStripeUnit;
    const bool scratch_created = create_direct_file(
        scratch, io_size, slice.logical_block_size);
    check(scratch_created,
          label + ": scratch file created from allocation view_path");
    ok = scratch_created && ok;

    auto opened = call_preserving_device(
        label + "/open", caller_device,
        [&] { return bundle->storage_runtime()->open("file://" + scratch,
                                                    OpenOptions{"file"}); });
    check(opened.ok(), label + ": file target opened");
    if (!opened.ok()) {
        (void)shutdown_and_check_ledger(bundle, caller_device, observer,
                                        baseline, label);
        ::unlink(scratch.c_str());
        return false;
    }

    bool io_ok = true;
    GpuBuffer buffer;
    io_ok = allocate_gpu_buffer(
        accel_id, caller_device, io_size, buffer) && io_ok;
    auto registered = call_preserving_device(
        label + "/register", caller_device,
        [&] {
            return bundle->storage_runtime()->register_memory(MemoryView{
                buffer.aligned, buffer.size, MemoryKind::DEVICE,
                MemoryOwnership::CALLER_OWNED, accel_id, "CUDA"});
        });
    check(registered.ok(), label + ": device memory registered");
    if (!registered.ok()) {
        (void)call_preserving_device(
            label + "/close", caller_device,
            [&] { return bundle->storage_runtime()->close(opened.value()); });
        free_gpu_buffer(accel_id, caller_device, buffer);
        (void)shutdown_and_check_ledger(bundle, caller_device, observer,
                                        baseline, label);
        ::unlink(scratch.c_str());
        return false;
    }

    const unsigned char pattern = static_cast<unsigned char>(
        0x30 + accel_id * 4 + (device_id & 3));
    io_ok = fill_value(
        buffer, accel_id, caller_device, 0, io_size, pattern) && io_ok;
    const HostSubmitContext context{
        ExecutionDomain::DEVICE_EXECUTION, accel_id, buffer.stream};
    IoRequest write{IoDirection::WRITE, registered.value(), 0,
                    opened.value(), 0, io_size};
    io_ok = submit_and_wait(*bundle->storage_runtime(), &write, 1, context,
                            caller_device, label + "/write") && io_ok;
    io_ok = fill_value(
        buffer, accel_id, caller_device, 0, io_size, 0xff) && io_ok;
    IoRequest read{IoDirection::READ, registered.value(), 0,
                   opened.value(), 0, io_size};
    io_ok = submit_and_wait(*bundle->storage_runtime(), &read, 1, context,
                            caller_device, label + "/read") && io_ok;
    io_ok = verify_value(
        buffer, accel_id, caller_device, 0, io_size, pattern) && io_ok;
    check(io_ok, label + ": Local write/read is byte-exact");
    ok = io_ok && ok;

    ok = cleanup_runtime_objects(*bundle, opened.value(), registered.value(),
                                 caller_device, label) && ok;
    free_gpu_buffer(accel_id, caller_device, buffer);
    ok = shutdown_and_check_ledger(bundle, caller_device, observer, baseline,
                                   label) && ok;
    check(::unlink(scratch.c_str()) == 0, label + ": scratch file removed");
    return ok;
}

std::string striped_uri(const std::string& name,
                        const std::vector<RuntimeNvmeSlice>& slices) {
    std::ostringstream uri;
    uri << "striped://" << name << "?devs=";
    for (std::size_t i = 0; i < slices.size(); ++i) {
        if (i != 0) uri << ',';
        uri << slices[i].view_path;
    }
    uri << "&unit=" << kStripeUnit;
    return uri.str();
}

std::vector<std::string> striped_paths(
    const std::string& name, const std::vector<RuntimeNvmeSlice>& slices) {
    std::vector<std::string> paths;
    for (std::size_t i = 0; i < slices.size(); ++i) {
        paths.push_back(slices[i].view_path + "/striped/" + name +
                        ".shard" + std::to_string(i));
    }
    return paths;
}

bool ensure_striped_files(const std::string& name,
                          const std::vector<RuntimeNvmeSlice>& slices,
                          std::vector<std::string>& paths) {
    paths = striped_paths(name, slices);
    bool ok = true;
    for (std::size_t i = 0; i < slices.size(); ++i) {
        const std::string directory = slices[i].view_path + "/striped";
        if (::mkdir(directory.c_str(), 0755) != 0 && errno != EEXIST) ok = false;
        ok = create_direct_file(paths[i], 8 * kStripeUnit,
                                slices[i].logical_block_size) && ok;
    }
    return ok;
}

bool run_striped_io(StorageRuntime& runtime, TargetHandle target,
                    MemoryHandle memory, GpuBuffer& buffer,
                    std::int32_t accel_id, std::int32_t caller_device,
                    bool write_phase, const std::string& label) {
    bool ok = true;
    const HostSubmitContext context{
        ExecutionDomain::DEVICE_EXECUTION, accel_id, buffer.stream};
    constexpr std::uint64_t cross_offset = kStripeUnit - 4096;
    constexpr std::uint64_t cross_size = 8192;
    constexpr std::uint64_t cross_base = 7000;
    constexpr std::uint64_t mixed_target_offsets[] = {
        2 * kStripeUnit, 3 * kStripeUnit, 4 * kStripeUnit};
    constexpr unsigned char mixed_values[] = {0x31, 0x52, 0x73};

    if (write_phase) {
        ok = fill_position(buffer, accel_id, caller_device, 0, cross_size,
                           cross_base) && ok;
        IoRequest cross_write{IoDirection::WRITE, memory, 0, target,
                              cross_offset, cross_size};
        ok = submit_and_wait(runtime, &cross_write, 1, context, caller_device,
                             label + "/cross-write") && ok;

        IoRequest mixed[3];
        for (std::size_t i = 0; i < 3; ++i) {
            ok = fill_value(buffer, accel_id, caller_device, i * kStripeUnit,
                            kStripeUnit, mixed_values[i]) && ok;
            mixed[i] = IoRequest{IoDirection::WRITE, memory, i * kStripeUnit,
                                 target, mixed_target_offsets[i], kStripeUnit};
        }
        ok = submit_and_wait(runtime, mixed, 3, context, caller_device,
                             label + "/mixed-write") && ok;
    }

    ok = fill_value(buffer, accel_id, caller_device, 0, buffer.size, 0xff) && ok;
    IoRequest cross_read{IoDirection::READ, memory, 0, target,
                         cross_offset, cross_size};
    ok = submit_and_wait(runtime, &cross_read, 1, context, caller_device,
                         label + "/cross-read") && ok;
    ok = verify_position(buffer, accel_id, caller_device, 0, cross_size,
                         cross_base) && ok;

    ok = fill_value(buffer, accel_id, caller_device, 0, buffer.size, 0xff) && ok;
    IoRequest mixed_reads[3];
    for (std::size_t i = 0; i < 3; ++i) {
        mixed_reads[i] = IoRequest{
            IoDirection::READ, memory, i * kStripeUnit, target,
            mixed_target_offsets[i], kStripeUnit};
    }
    ok = submit_and_wait(runtime, mixed_reads, 3, context, caller_device,
                         label + "/mixed-read") && ok;
    for (std::size_t i = 0; i < 3; ++i) {
        ok = verify_value(buffer, accel_id, caller_device, i * kStripeUnit,
                          kStripeUnit, mixed_values[i]) && ok;
    }
    check(ok, label + ": striped boundary and mixed batch are byte-exact");
    return ok;
}

bool validate_striped_component_order(
    TuttiRuntime& bundle, const std::string& uri,
    const std::vector<RuntimeNvmeSlice>& allocation_slices,
    const std::string& label) {
    auto* resolver = TuttiRuntimeTestingAccess::backend_resolver(
        bundle, "storage-backend");
    const auto* datapath = dynamic_cast<const StripedDataPath*>(
        TuttiRuntimeTestingAccess::backend_datapath(
            bundle, "storage-backend"));
    bool ok = resolver != nullptr && datapath != nullptr;
    if (!ok) {
        check(false, label + ": striped backend components are inspectable");
        return false;
    }

    tutti::ResolveOptions options;
    options.scheme = "striped";
    auto resolved = resolver->resolve(uri, options);
    ok = resolved.ok();
    if (!resolved.ok()) {
        check(false, label + ": resolver shard order is inspectable");
        return false;
    }
    const auto payload =
        tutti::binding::striped_local_nvme::view_payload(resolved.value());
    ok = ok && payload.ok();
    const auto& descriptors = datapath->test_device_descriptors();
    if (payload.ok()) {
        const auto* striped = payload.value();
        ok = ok && striped->shards().size() == descriptors.size() &&
             descriptors.size() == allocation_slices.size();
        const std::size_t count = std::min(
            striped->shards().size(),
            std::min(descriptors.size(), allocation_slices.size()));
        for (std::size_t index = 0; index < count; ++index) {
            const auto shard =
                tutti::binding::ext4_local_nvme::view_payload(
                    striped->shards()[index]);
            ok = ok && shard.ok();
            if (!shard.ok()) continue;
            const auto& identity = shard.value()->namespace_identity();
            const DeviceDescriptor& descriptor = descriptors[index];
            const RuntimeNvmeSlice& slice = allocation_slices[index];
            ok = ok && identity.controller_pci_addr ==
                           descriptor.controller_pci_addr &&
                 identity.namespace_id == descriptor.namespace_id &&
                 identity.block_size == descriptor.block_size &&
                 descriptor.controller_pci_addr == slice.pci_bdf &&
                 descriptor.snvme_dev_path == slice.chrdev_path &&
                 descriptor.namespace_id == slice.namespace_id &&
                 descriptor.block_size == slice.logical_block_size;
        }
    }
    check(ok, label +
                  ": resolver shard order equals DataPath descriptor order");
    return ok;
}

bool run_striped_bundle_phase(
    const Options& options, const std::string& config_path,
    std::int32_t accel_id, std::int32_t caller_device,
    const std::vector<std::int32_t>& devices, NvmeServiceClient& observer,
    const AcceleratorMap& accelerators,
    const ResourceMap& baseline, const std::string& name,
    const std::string& label, bool write_phase,
    std::vector<std::string>& scratch_paths) {
    (void)cudaSetDevice(caller_device);
    bool ok = check_scenario_baseline(observer, baseline, label);
    ok = static_config_is_logical(
             config_path, options, accel_id,
             tutti::config::NvmeSelection::Striped, devices, baseline,
             label) && ok;
    auto bundle = load_bundle(config_path, caller_device, label);
    if (!bundle) return false;
    ok = validate_metadata(*bundle, accelerators, baseline, accel_id, devices,
                           static_cast<std::uint32_t>(options.queues)) && ok;
    const auto manifest = bundle->backend_manifest("storage-backend");
    const bool manifest_ok = manifest.ok() &&
        manifest.value().contract == "striped-local-nvme";
    check(manifest_ok,
          label + ": loader publishes one Striped top-level binding");
    ok = manifest_ok && ok;
    const ResourceMap acquired = snapshot_resources(observer);
    print_snapshot((label + " acquired").c_str(), acquired);
    ok = check_acquired_ledger(baseline, acquired, devices, options.queues) && ok;

    if (write_phase) {
        const auto allocation_slices =
            inspect_nvme_resource(*bundle).allocation.slices;
        const bool created = ensure_striped_files(
            name, allocation_slices, scratch_paths);
        for (const std::string& path : scratch_paths) {
            std::printf("SCRATCH scenario=%s path=%s\n",
                        label.c_str(), path.c_str());
        }
        check(created, label + ": striped scratch files created from view_path");
        ok = created && ok;
    } else {
        const auto expected_paths = striped_paths(
            name, inspect_nvme_resource(*bundle).allocation.slices);
        check(expected_paths == scratch_paths,
              label + ": restart resolves the same allocation view paths");
        ok = expected_paths == scratch_paths && ok;
    }

    const std::string uri = striped_uri(
        name, inspect_nvme_resource(*bundle).allocation.slices);
    ok = validate_striped_component_order(
             *bundle, uri,
             inspect_nvme_resource(*bundle).allocation.slices, label) && ok;
    auto opened = call_preserving_device(
        label + "/open", caller_device,
        [&] { return bundle->storage_runtime()->open(
            uri, OpenOptions{"striped"}); });
    check(opened.ok(), label + ": striped target opened");
    if (!opened.ok()) {
        (void)shutdown_and_check_ledger(bundle, caller_device, observer,
                                        baseline, label);
        return false;
    }

    GpuBuffer buffer;
    ok = allocate_gpu_buffer(accel_id, caller_device, 3 * kStripeUnit,
                             buffer) && ok;
    auto registered = call_preserving_device(
        label + "/register", caller_device,
        [&] {
            return bundle->storage_runtime()->register_memory(MemoryView{
                buffer.aligned, buffer.size, MemoryKind::DEVICE,
                MemoryOwnership::CALLER_OWNED, accel_id, "CUDA"});
        });
    check(registered.ok(), label + ": device memory registered");
    if (!registered.ok()) {
        (void)call_preserving_device(
            label + "/close", caller_device,
            [&] { return bundle->storage_runtime()->close(opened.value()); });
        free_gpu_buffer(accel_id, caller_device, buffer);
        (void)shutdown_and_check_ledger(bundle, caller_device, observer,
                                        baseline, label);
        return false;
    }

    ok = run_striped_io(*bundle->storage_runtime(), opened.value(),
                        registered.value(),
                        buffer, accel_id, caller_device, write_phase, label) && ok;
    ok = cleanup_runtime_objects(*bundle, opened.value(), registered.value(),
                                 caller_device, label) && ok;
    free_gpu_buffer(accel_id, caller_device, buffer);
    ok = shutdown_and_check_ledger(bundle, caller_device, observer, baseline,
                                   label) && ok;
    return ok;
}

bool run_striped_scenario(
    const Options& options, const std::string& config_path,
    std::int32_t accel_id, std::int32_t caller_device,
    const std::vector<std::int32_t>& devices, NvmeServiceClient& observer,
    const AcceleratorMap& accelerators,
    const ResourceMap& baseline, const std::string& label) {
    std::printf("\n=== %s accel=%d devices=%d,%d caller=%d ===\n",
                label.c_str(), accel_id, devices[0], devices[1], caller_device);
    const std::string name = "tutti_phase7_" + std::to_string(::getpid()) +
                             "_" + label;
    std::vector<std::string> scratch_paths;
    bool ok = run_striped_bundle_phase(
        options, config_path, accel_id, caller_device, devices, observer,
        accelerators,
        baseline, name, label + "-write", true, scratch_paths);
    ok = run_striped_bundle_phase(
        options, config_path, accel_id, caller_device, devices, observer,
        accelerators,
        baseline, name, label + "-restart-read", false, scratch_paths) && ok;
    for (const std::string& path : scratch_paths) {
        check(::unlink(path.c_str()) == 0, label + ": striped scratch removed");
    }
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage(argv[0]);
        return 2;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        options.accel0 < 0 || options.accel0 >= device_count ||
        (!options.single_accelerator &&
         (options.accel1 < 0 || options.accel1 >= device_count))) {
        std::printf("SKIP: requested accelerator set is unavailable\n");
        return kSkip;
    }

    NvmeServiceClient observer(options.endpoint);
    const AcceleratorMap accelerators = snapshot_accelerators(observer);
    const ResourceMap baseline = snapshot_resources(observer);
    const std::size_t required_accelerators =
        options.single_accelerator ? 1 : 2;
    if (accelerators.size() < required_accelerators ||
        accelerators.count(options.accel0) == 0 ||
        (!options.single_accelerator &&
         accelerators.count(options.accel1) == 0) ||
        baseline.count(options.device0) == 0 ||
        baseline.count(options.device1) == 0 ||
        !baseline.at(options.device0).available ||
        !baseline.at(options.device1).available) {
        std::printf("SKIP: live daemon with two available NVMe resources required\n");
        return kSkip;
    }
    const std::vector<std::int32_t> requested_accelerators =
        options.single_accelerator
        ? std::vector<std::int32_t>{options.accel0}
        : std::vector<std::int32_t>{options.accel0, options.accel1};
    for (std::int32_t device_id : {options.device0, options.device1}) {
        for (std::int32_t accel_id : requested_accelerators) {
            const auto& allowed = baseline.at(device_id).allowed_accel_ids;
            if (std::find(allowed.begin(), allowed.end(), accel_id) ==
                allowed.end()) {
                std::printf(
                    "SKIP: device %d ACL does not allow accelerator %d\n",
                    device_id, accel_id);
                return kSkip;
            }
        }
    }
    print_accelerator_snapshot(accelerators);
    print_snapshot("baseline", baseline);
    if (baseline.at(options.device0).logical_block_size !=
        baseline.at(options.device1).logical_block_size) {
        std::fprintf(stderr, "ERROR: striped namespaces have different block sizes\n");
        return 1;
    }

    TempConfigDirectory configs;
    if (!configs.create()) {
        std::perror("mkdtemp");
        return 1;
    }

    bool ok = true;
    const std::vector<std::pair<std::int32_t, std::int32_t>> accelerators_to_run =
        options.single_accelerator
        ? std::vector<std::pair<std::int32_t, std::int32_t>>{
              {options.accel0, options.accel0}}
        : std::vector<std::pair<std::int32_t, std::int32_t>>{
              {options.accel0, options.accel1},
              {options.accel1, options.accel0}};
    for (std::size_t accel_index = 0;
         accel_index < accelerators_to_run.size(); ++accel_index) {
        const auto& accel = accelerators_to_run[accel_index];
        const char prefix = accel_index == 0 ? 'A' : 'B';
        int scenario = 0;
        for (std::int32_t device_id : {options.device0, options.device1}) {
            const std::string label = std::string(1, prefix) +
                                      std::to_string(scenario++);
            const std::string config = configs.write(
                label, options, accel.first, "explicit", {device_id});
            check(!config.empty(), label + ": temporary canonical config written");
            ok = !config.empty() && run_local_scenario(
                options, config, accel.first, accel.second, device_id,
                observer, accelerators, baseline, label) && ok;
        }
        const std::string label = std::string(1, prefix) +
                                  std::to_string(scenario);
        const std::vector<std::int32_t> devices = {
            options.device0, options.device1};
        const std::string config = configs.write(
            label, options, accel.first, "striped", devices);
        check(!config.empty(), label + ": temporary canonical config written");
        ok = !config.empty() && run_striped_scenario(
            options, config, accel.first, accel.second, devices,
            observer, accelerators, baseline, label) && ok;
    }

    ResourceMap final_snapshot;
    const bool final_clean = wait_for_baseline(observer, baseline, final_snapshot);
    print_snapshot("final", final_snapshot);
    check(final_clean, "final daemon ledger equals test baseline");

    std::printf("\nphase7 checks=%d failures=%d result=%s\n",
                g_checks, g_failures, ok && g_failures == 0 ? "PASS" : "FAIL");
    return ok && g_failures == 0 ? 0 : 1;
}

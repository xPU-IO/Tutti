#include <tutti/config/tutti_config.h>
#include <tutti/cuda_like.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>
#include <tutti/storage_runtime.h>

#include "nvmeservice_client.h"

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
using tutti::config::RuntimeNvmeSlice;
using tutti::config::TuttiRuntime;
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
        "[--device0 ID] [--device1 ID] [--queues N]\n", program);
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
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
    return options.queues > 0 && options.accel0 != options.accel1 &&
           options.device0 != options.device1;
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
        char path[] = "/tmp/tutti_phase5_XXXXXX";
        char* created = ::mkdtemp(path);
        if (created == nullptr) return false;
        path_ = created;
        return true;
    }

    std::string write(const std::string& name, const Options& options,
                      std::int32_t accel_id, const std::string& selection,
                      const std::vector<std::int32_t>& device_ids) {
        const std::string path = path_ + "/" + name + ".yaml";
        std::ofstream output(path, std::ios::trunc);
        output << "accelerator:\n"
               << "  profile: \"CUDA\"\n\n"
               << "runtime:\n"
               << "  accel_id: " << accel_id << "\n\n"
               << "nvme_service:\n"
               << "  endpoint: \"" << options.endpoint << "\"\n\n"
               << "nvme:\n"
               << "  selection: \"" << selection << "\"\n"
               << "  device_ids: [";
        for (std::size_t i = 0; i < device_ids.size(); ++i) {
            if (i != 0) output << ", ";
            output << device_ids[i];
        }
        output << "]\n"
               << "  queues_per_controller: " << options.queues << "\n"
               << "  stripe_unit: " << kStripeUnit << "\n";
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

ResourceMap snapshot_resources(NvmeServiceClient& client) {
    ResourceMap result;
    for (auto& resource : client.list_nvme_resources()) {
        result.emplace(resource.device_id, std::move(resource));
    }
    return result;
}

void print_snapshot(const char* label, const ResourceMap& resources) {
    std::printf("LEDGER %s\n", label);
    for (const auto& row : resources) {
        const auto& resource = row.second;
        std::printf("  device=%d reserved=%u available=%u available_flag=%s\n",
                    resource.device_id, resource.reserved_queues,
                    resource.available_queues,
                    resource.available ? "true" : "false");
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

bool validate_metadata(const TuttiRuntime& bundle,
                       const ResourceMap& baseline,
                       std::int32_t accel_id,
                       const std::vector<std::int32_t>& selected,
                       std::uint32_t queues) {
    bool ok = !bundle.allocation_id.empty() &&
              bundle.allocation_slices.size() == selected.size();
    ok = ok && bundle.resolver_schemes.size() == 1 &&
         bundle.data_path_keys.size() == 1;
    for (std::size_t i = 0; i < bundle.allocation_slices.size(); ++i) {
        const RuntimeNvmeSlice& slice = bundle.allocation_slices[i];
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
        ok = ok && !slice.view_path.empty() &&
             paths_share_filesystem(slice.view_path,
                                    slice.backing_mount_path);
        std::printf(
            "SLICE allocation=%s device=%d accel=%d pci=%s chrdev=%s block=%s "
            "backing=%s view=%s ns=%u lba=%u bar0=%llu mdts=%llu grant=%u\n",
            bundle.allocation_id.c_str(), slice.device_id, slice.accel_id,
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
        [&] { return bundle.runtime->close(target); });
    check(closed.ok(), label + ": target closed");
    Status unregistered = call_preserving_device(
        label + "/unregister", caller_device,
        [&] { return bundle.runtime->unregister_memory(memory); });
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
    Status status = call_preserving_device(
        label + "/shutdown", caller_device,
        [&] { return bundle->shutdown(); });
    check(status.ok(), label + ": bundle shutdown succeeds");
    bundle.reset();
    ResourceMap released;
    const bool restored = wait_for_baseline(observer, baseline, released);
    print_snapshot((label + " released").c_str(), released);
    check(restored, label + ": Release restores exact ledger baseline");
    return status.ok() && restored;
}

bool run_local_scenario(const Options& options, const std::string& config_path,
                        std::int32_t accel_id, std::int32_t caller_device,
                        std::int32_t device_id, NvmeServiceClient& observer,
                        const ResourceMap& baseline, const std::string& label) {
    std::printf("\n=== %s accel=%d device=%d caller=%d ===\n",
                label.c_str(), accel_id, device_id, caller_device);
    (void)cudaSetDevice(caller_device);
    auto bundle = load_bundle(config_path, caller_device, label);
    if (!bundle) return false;
    bool ok = validate_metadata(*bundle, baseline, accel_id, {device_id},
                                static_cast<std::uint32_t>(options.queues));
    ok = bundle->resolver_schemes == std::vector<std::string>{"file"} &&
         bundle->data_path_keys ==
             std::vector<std::string>{"local-nvme-ext4"} && ok;
    check(ok, label + ": loader publishes one Local top-level binding");
    const ResourceMap acquired = snapshot_resources(observer);
    print_snapshot((label + " acquired").c_str(), acquired);
    ok = check_acquired_ledger(baseline, acquired, {device_id}, options.queues) && ok;

    const RuntimeNvmeSlice slice = bundle->allocation_slices.front();
    const std::string scratch = slice.view_path + "/tutti_phase5_" +
        std::to_string(::getpid()) + "_" + label + ".bin";
    std::printf("SCRATCH scenario=%s path=%s\n",
                label.c_str(), scratch.c_str());
    const std::uint64_t io_size = 4 * kStripeUnit;
    ok = create_direct_file(scratch, io_size, slice.logical_block_size) && ok;
    check(ok, label + ": scratch file created from allocation view_path");

    auto opened = call_preserving_device(
        label + "/open", caller_device,
        [&] { return bundle->runtime->open("file://" + scratch,
                                          OpenOptions{"file"}); });
    check(opened.ok(), label + ": file target opened");
    if (!opened.ok()) {
        (void)shutdown_and_check_ledger(bundle, caller_device, observer,
                                        baseline, label);
        ::unlink(scratch.c_str());
        return false;
    }

    GpuBuffer buffer;
    ok = allocate_gpu_buffer(accel_id, caller_device, io_size, buffer) && ok;
    auto registered = call_preserving_device(
        label + "/register", caller_device,
        [&] {
            return bundle->runtime->register_memory(MemoryView{
                buffer.aligned, buffer.size, MemoryKind::DEVICE,
                MemoryOwnership::CALLER_OWNED, accel_id, "CUDA"});
        });
    check(registered.ok(), label + ": device memory registered");
    if (!registered.ok()) {
        (void)call_preserving_device(
            label + "/close", caller_device,
            [&] { return bundle->runtime->close(opened.value()); });
        free_gpu_buffer(accel_id, caller_device, buffer);
        (void)shutdown_and_check_ledger(bundle, caller_device, observer,
                                        baseline, label);
        ::unlink(scratch.c_str());
        return false;
    }

    const unsigned char pattern = static_cast<unsigned char>(
        0x30 + accel_id * 4 + (device_id & 3));
    ok = fill_value(buffer, accel_id, caller_device, 0, io_size, pattern) && ok;
    const HostSubmitContext context{
        ExecutionDomain::DEVICE_EXECUTION, accel_id, buffer.stream};
    IoRequest write{IoDirection::WRITE, registered.value(), 0,
                    opened.value(), 0, io_size};
    ok = submit_and_wait(*bundle->runtime, &write, 1, context,
                         caller_device, label + "/write") && ok;
    ok = fill_value(buffer, accel_id, caller_device, 0, io_size, 0xff) && ok;
    IoRequest read{IoDirection::READ, registered.value(), 0,
                   opened.value(), 0, io_size};
    ok = submit_and_wait(*bundle->runtime, &read, 1, context,
                         caller_device, label + "/read") && ok;
    ok = verify_value(buffer, accel_id, caller_device, 0, io_size, pattern) && ok;
    check(ok, label + ": Local write/read is byte-exact");

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

bool run_striped_bundle_phase(
    const Options& options, const std::string& config_path,
    std::int32_t accel_id, std::int32_t caller_device,
    const std::vector<std::int32_t>& devices, NvmeServiceClient& observer,
    const ResourceMap& baseline, const std::string& name,
    const std::string& label, bool write_phase,
    std::vector<std::string>& scratch_paths) {
    (void)cudaSetDevice(caller_device);
    auto bundle = load_bundle(config_path, caller_device, label);
    if (!bundle) return false;
    bool ok = validate_metadata(*bundle, baseline, accel_id, devices,
                                static_cast<std::uint32_t>(options.queues));
    ok = bundle->resolver_schemes == std::vector<std::string>{"striped"} &&
         bundle->data_path_keys ==
             std::vector<std::string>{"striped-local-nvme"} && ok;
    check(ok, label + ": loader publishes one Striped top-level binding");
    const ResourceMap acquired = snapshot_resources(observer);
    print_snapshot((label + " acquired").c_str(), acquired);
    ok = check_acquired_ledger(baseline, acquired, devices, options.queues) && ok;

    if (write_phase) {
        const bool created = ensure_striped_files(
            name, bundle->allocation_slices, scratch_paths);
        for (const std::string& path : scratch_paths) {
            std::printf("SCRATCH scenario=%s path=%s\n",
                        label.c_str(), path.c_str());
        }
        check(created, label + ": striped scratch files created from view_path");
        ok = created && ok;
    } else {
        const auto expected_paths = striped_paths(name, bundle->allocation_slices);
        check(expected_paths == scratch_paths,
              label + ": restart resolves the same allocation view paths");
        ok = expected_paths == scratch_paths && ok;
    }

    const std::string uri = striped_uri(name, bundle->allocation_slices);
    auto opened = call_preserving_device(
        label + "/open", caller_device,
        [&] { return bundle->runtime->open(uri, OpenOptions{"striped"}); });
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
            return bundle->runtime->register_memory(MemoryView{
                buffer.aligned, buffer.size, MemoryKind::DEVICE,
                MemoryOwnership::CALLER_OWNED, accel_id, "CUDA"});
        });
    check(registered.ok(), label + ": device memory registered");
    if (!registered.ok()) {
        (void)call_preserving_device(
            label + "/close", caller_device,
            [&] { return bundle->runtime->close(opened.value()); });
        free_gpu_buffer(accel_id, caller_device, buffer);
        (void)shutdown_and_check_ledger(bundle, caller_device, observer,
                                        baseline, label);
        return false;
    }

    ok = run_striped_io(*bundle->runtime, opened.value(), registered.value(),
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
    const ResourceMap& baseline, const std::string& label) {
    std::printf("\n=== %s accel=%d devices=%d,%d caller=%d ===\n",
                label.c_str(), accel_id, devices[0], devices[1], caller_device);
    const std::string name = "tutti_phase5_" + std::to_string(::getpid()) +
                             "_" + label;
    std::vector<std::string> scratch_paths;
    bool ok = run_striped_bundle_phase(
        options, config_path, accel_id, caller_device, devices, observer,
        baseline, name, label + "-write", true, scratch_paths);
    ok = run_striped_bundle_phase(
        options, config_path, accel_id, caller_device, devices, observer,
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
        options.accel0 < 0 || options.accel1 < 0 ||
        options.accel0 >= device_count || options.accel1 >= device_count) {
        std::printf("SKIP: phase 5 requires both requested accelerators\n");
        return kSkip;
    }

    NvmeServiceClient observer(options.endpoint);
    const auto accelerators = observer.list_accelerators();
    const ResourceMap baseline = snapshot_resources(observer);
    if (accelerators.size() < 2 || baseline.count(options.device0) == 0 ||
        baseline.count(options.device1) == 0 ||
        !baseline.at(options.device0).available ||
        !baseline.at(options.device1).available) {
        std::printf("SKIP: live daemon with two available NVMe resources required\n");
        return kSkip;
    }
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
    const std::vector<std::pair<std::int32_t, std::int32_t>> accelerators_to_run = {
        {options.accel0, options.accel1},
        {options.accel1, options.accel0},
    };
    int scenario = 1;
    for (const auto& accel : accelerators_to_run) {
        for (std::int32_t device_id : {options.device0, options.device1}) {
            const std::string label = "A" + std::to_string(scenario++);
            const std::string config = configs.write(
                label, options, accel.first, "explicit", {device_id});
            check(!config.empty(), label + ": temporary canonical config written");
            ok = !config.empty() && run_local_scenario(
                options, config, accel.first, accel.second, device_id,
                observer, baseline, label) && ok;
        }
        const std::string label = "A" + std::to_string(scenario++);
        const std::vector<std::int32_t> devices = {
            options.device0, options.device1};
        const std::string config = configs.write(
            label, options, accel.first, "striped", devices);
        check(!config.empty(), label + ": temporary canonical config written");
        ok = !config.empty() && run_striped_scenario(
            options, config, accel.first, accel.second, devices,
            observer, baseline, label) && ok;
    }

    ResourceMap final_snapshot;
    const bool final_clean = wait_for_baseline(observer, baseline, final_snapshot);
    print_snapshot("final", final_snapshot);
    check(final_clean, "final daemon ledger equals test baseline");

    std::printf("\nphase5 checks=%d failures=%d result=%s\n",
                g_checks, g_failures, ok && g_failures == 0 ? "PASS" : "FAIL");
    return ok && g_failures == 0 ? 0 : 1;
}

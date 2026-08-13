#include <tutti/tutti_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <tutti/config/tutti_runtime_config_parser.h>
#include <tutti/cuda_like.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>
#include <tutti/storage_runtime.h>

namespace {

constexpr std::size_t kRuntimeCount = 2;
constexpr std::uint64_t kDefaultIoSize = 4 * 1024 * 1024;
constexpr std::uint64_t kDeviceBufferAlignment = 64 * 1024;
constexpr std::uint64_t kBlockAlignment = 4096;
constexpr std::uint64_t kIoTimeoutMs = 30000;

std::mutex g_log_mutex;

struct Options {
    std::array<std::string, kRuntimeCount> config_paths{
        TUTTI_RUNTIME_MULTI_ACCEL_CONFIG_0,
        TUTTI_RUNTIME_MULTI_ACCEL_CONFIG_1,
    };
    std::array<std::vector<std::string>, kRuntimeCount> directories;
    std::array<std::vector<std::int32_t>, kRuntimeCount> device_ids;
    std::array<std::uint64_t, kRuntimeCount> stripe_units{{0, 0}};
    std::array<bool, kRuntimeCount> striped{{false, false}};
    std::uint64_t io_size = kDefaultIoSize;
    bool keep_files = false;
};

struct WorkerResult {
    bool ok = false;
    std::int32_t accel_id = -1;
    std::string target_uri;
    std::chrono::milliseconds elapsed{0};
};

void log_worker(std::size_t worker, FILE* stream, const char* format, ...) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::fprintf(stream, "[runtime %zu] ", worker);
    va_list arguments;
    va_start(arguments, format);
    std::vfprintf(stream, format, arguments);
    va_end(arguments);
    std::fputc('\n', stream);
    std::fflush(stream);
}

void print_usage(const char* program) {
    std::fprintf(
        stderr,
        "Usage: %s --directory-0 PATH [--directory-0 PATH] "
        "--directory-1 PATH [--directory-1 PATH] "
        "[--config-0 PATH] [--config-1 PATH] [--size BYTES] "
        "[--keep-files]\n\n"
        "Each config creates one TuttiRuntime. The configs must select "
        "different accelerators from the same daemon.\n",
        program);
}

bool parse_u64(const char* text, std::uint64_t& value) {
    if (text == nullptr || *text == '\0' || *text == '-') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return false;
    value = static_cast<std::uint64_t>(parsed);
    return true;
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (argument == "--keep-files") {
            options.keep_files = true;
            continue;
        }
        if (index + 1 >= argc) return false;
        const char* value = argv[++index];
        if (argument == "--config-0") {
            options.config_paths[0] = value;
        } else if (argument == "--config-1") {
            options.config_paths[1] = value;
        } else if (argument == "--directory-0") {
            options.directories[0].emplace_back(value);
        } else if (argument == "--directory-1") {
            options.directories[1].emplace_back(value);
        } else if (argument == "--size") {
            if (!parse_u64(value, options.io_size)) return false;
        } else {
            return false;
        }
    }
    return !options.config_paths[0].empty() &&
           !options.config_paths[1].empty() &&
           !options.directories[0].empty() &&
           !options.directories[1].empty() && options.io_size > 0 &&
           options.io_size % kBlockAlignment == 0 &&
           options.io_size <= std::numeric_limits<std::size_t>::max();
}

bool validate_config_pair(Options& options) {
    std::array<tutti::config::TuttiRuntimeSpec, kRuntimeCount> specs;
    for (std::size_t index = 0; index < kRuntimeCount; ++index) {
        auto parsed = tutti::config::parse_tutti_runtime_config(
            options.config_paths[index]);
        if (!parsed.ok()) {
            std::fprintf(stderr, "Config %zu is invalid: %s\n", index,
                         parsed.status().message().c_str());
            return false;
        }
        specs[index] = std::move(parsed).value();
    }
    if (specs[0].runtime.accel_id == specs[1].runtime.accel_id) {
        std::fprintf(stderr,
                     "Configs must select two different runtime.accel_id "
                     "values\n");
        return false;
    }

    std::array<std::string, kRuntimeCount> endpoints;
    for (std::size_t index = 0; index < kRuntimeCount; ++index) {
        if (specs[index].storage.resources.size() != 1) {
            std::fprintf(stderr,
                         "Config %zu must contain exactly one NVMe resource\n",
                         index);
            return false;
        }
        const auto* nvme = std::get_if<tutti::config::NvmeResourceConfig>(
            &specs[index].storage.resources.front().config);
        if (nvme == nullptr || nvme->allocation.device_ids.empty()) {
            std::fprintf(stderr,
                         "Config %zu must explicitly select NVMe devices\n",
                         index);
            return false;
        }
        const bool explicit_local =
            nvme->allocation.selection == tutti::config::NvmeSelection::Explicit &&
            nvme->allocation.device_ids.size() == 1;
        const bool striped =
            nvme->allocation.selection == tutti::config::NvmeSelection::Striped &&
            nvme->allocation.device_ids.size() >= 2;
        if (!explicit_local && !striped) {
            std::fprintf(stderr,
                         "Config %zu must use one explicit device or at least "
                         "two striped devices\n",
                         index);
            return false;
        }
        if (options.directories[index].size() !=
            nvme->allocation.device_ids.size()) {
            std::fprintf(stderr,
                         "Config %zu selects %zu devices but received %zu "
                         "directories\n",
                         index, nvme->allocation.device_ids.size(),
                         options.directories[index].size());
            return false;
        }
        options.device_ids[index] = nvme->allocation.device_ids;
        options.striped[index] = striped;
        if (striped) {
            if (specs[index].storage.backends.size() != 1) {
                std::fprintf(stderr,
                             "Striped config %zu must contain exactly one "
                             "backend\n",
                             index);
                return false;
            }
            const auto* backend =
                std::get_if<tutti::config::StripedLocalNvmeBackendConfig>(
                    &specs[index].storage.backends.front().config);
            if (backend == nullptr) {
                std::fprintf(stderr,
                             "Striped config %zu must use a "
                             "striped-local-nvme backend\n",
                             index);
                return false;
            }
            options.stripe_units[index] = backend->stripe_unit;
        }
        endpoints[index] = nvme->provider.endpoint;
    }
    if (endpoints[0] != endpoints[1]) {
        std::fprintf(stderr,
                     "Configs must use the same NVMe daemon endpoint\n");
        return false;
    }
    return true;
}

void print_status(std::size_t worker, const char* operation,
                  const tutti::Status& status) {
    log_worker(worker, stderr, "%s failed: %s", operation,
               status.message().c_str());
}

bool check_cuda(std::size_t worker, cudaError_t result,
                const char* operation) {
    if (result == cudaSuccess) return true;
    log_worker(worker, stderr, "%s failed: %s", operation,
               cudaGetErrorString(result));
    return false;
}

class Barrier {
public:
    explicit Barrier(std::size_t participants)
        : participants_(participants) {}

    bool arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cancelled_) return false;
        ++arrived_;
        if (arrived_ == participants_) {
            released_ = true;
            condition_.notify_all();
            return true;
        }
        condition_.wait(lock, [this] { return released_ || cancelled_; });
        return !cancelled_;
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        condition_.notify_all();
    }

private:
    const std::size_t participants_;
    std::size_t arrived_ = 0;
    bool released_ = false;
    bool cancelled_ = false;
    std::mutex mutex_;
    std::condition_variable condition_;
};

class IoCoordinator {
public:
    bool arrive_and_wait(std::size_t worker, bool ready,
                         std::int32_t accel_id) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_ = ready_ && ready;
        accel_ids_[worker] = accel_id;
        ++arrived_;
        if (arrived_ == kRuntimeCount) {
            if (accel_ids_[0] < 0 || accel_ids_[1] < 0 ||
                accel_ids_[0] == accel_ids_[1]) {
                ready_ = false;
            }
            released_ = true;
            condition_.notify_all();
        } else {
            condition_.wait(lock, [this] { return released_; });
        }
        return ready_;
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_ = false;
        released_ = true;
        condition_.notify_all();
    }

private:
    std::array<std::int32_t, kRuntimeCount> accel_ids_{{-1, -1}};
    std::size_t arrived_ = 0;
    bool ready_ = true;
    bool released_ = false;
    std::mutex mutex_;
    std::condition_variable condition_;
};

class PhaseCoordinator {
public:
    bool arrive_and_wait(bool succeeded) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cancelled_) return false;
        succeeded_ = succeeded_ && succeeded;
        ++arrived_;
        if (arrived_ == kRuntimeCount) {
            released_ = true;
            condition_.notify_all();
        } else {
            condition_.wait(lock,
                            [this] { return released_ || cancelled_; });
        }
        return succeeded_ && !cancelled_;
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        condition_.notify_all();
    }

private:
    std::size_t arrived_ = 0;
    bool succeeded_ = true;
    bool released_ = false;
    bool cancelled_ = false;
    std::mutex mutex_;
    std::condition_variable condition_;
};

std::string join_path(const std::string& directory,
                      const std::string& name) {
    std::string path = directory;
    if (!path.empty() && path.back() != '/') path.push_back('/');
    path += name;
    return path;
}

bool create_scratch_file(std::size_t worker, const std::string& path,
                         std::uint64_t size) {
    const int fd =
        ::open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_DIRECT, 0600);
    if (fd < 0) {
        log_worker(worker, stderr, "open(%s) failed: %s", path.c_str(),
                   std::strerror(errno));
        return false;
    }

    constexpr std::size_t kChunkSize = 1024 * 1024;
    void* zeros = nullptr;
    if (::posix_memalign(&zeros, kBlockAlignment, kChunkSize) != 0) {
        log_worker(worker, stderr, "failed to allocate aligned file buffer");
        ::close(fd);
        ::unlink(path.c_str());
        return false;
    }
    std::memset(zeros, 0, kChunkSize);

    bool ok = true;
    for (std::uint64_t written = 0; written < size;) {
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(kChunkSize, size - written));
        const ssize_t result = ::write(fd, zeros, chunk);
        if (result != static_cast<ssize_t>(chunk)) {
            log_worker(worker, stderr, "write(%s) failed: %s", path.c_str(),
                       result < 0 ? std::strerror(errno) : "short write");
            ok = false;
            break;
        }
        written += chunk;
    }
    if (ok && ::fsync(fd) != 0) {
        log_worker(worker, stderr, "fsync(%s) failed: %s", path.c_str(),
                   std::strerror(errno));
        ok = false;
    }
    std::free(zeros);
    if (::close(fd) != 0) ok = false;
    if (!ok) ::unlink(path.c_str());
    return ok;
}

struct ScratchTarget {
    std::string uri;
    std::vector<std::string> files;
    std::vector<std::string> created_directories;
};

bool cleanup_scratch_target(std::size_t worker,
                            const ScratchTarget& target) {
    bool ok = true;
    for (const std::string& path : target.files) {
        if (::unlink(path.c_str()) != 0 && errno != ENOENT) {
            log_worker(worker, stderr, "unlink(%s) failed: %s",
                       path.c_str(), std::strerror(errno));
            ok = false;
        }
    }
    for (auto directory = target.created_directories.rbegin();
         directory != target.created_directories.rend(); ++directory) {
        if (::rmdir(directory->c_str()) != 0 && errno != ENOENT &&
            errno != ENOTEMPTY) {
            log_worker(worker, stderr, "rmdir(%s) failed: %s",
                       directory->c_str(), std::strerror(errno));
            ok = false;
        }
    }
    return ok;
}

bool create_scratch_target(std::size_t worker, const Options& options,
                           ScratchTarget& target) {
    const std::string name =
        "tutti_runtime_multi_accelerator_example." +
        std::to_string(::getpid()) + ".runtime" +
        std::to_string(worker) + ".bin";
    const auto& directories = options.directories[worker];
    if (!options.striped[worker]) {
        const std::string path = join_path(directories.front(), name);
        if (!create_scratch_file(worker, path, options.io_size)) return false;
        target.files.push_back(path);
        target.uri = "file://" + path;
        return true;
    }

    const std::uint64_t device_count = directories.size();
    const std::uint64_t stripe_unit = options.stripe_units[worker];
    if (stripe_unit == 0 ||
        stripe_unit > std::numeric_limits<std::uint64_t>::max() /
                          device_count) {
        log_worker(worker, stderr, "invalid striped geometry");
        return false;
    }
    const std::uint64_t logical_cycle = stripe_unit * device_count;
    if (options.io_size >
        std::numeric_limits<std::uint64_t>::max() - (logical_cycle - 1)) {
        log_worker(worker, stderr, "striped I/O size rounding overflows");
        return false;
    }
    const std::uint64_t shard_size =
        ((options.io_size + logical_cycle - 1) / logical_cycle) *
        stripe_unit;

    std::string mounts;
    for (std::size_t index = 0; index < directories.size(); ++index) {
        if (index != 0) mounts.push_back(',');
        mounts += directories[index];

        const std::string striped_directory =
            join_path(directories[index], "striped");
        if (::mkdir(striped_directory.c_str(), 0755) == 0) {
            target.created_directories.push_back(striped_directory);
        } else if (errno != EEXIST) {
            log_worker(worker, stderr, "mkdir(%s) failed: %s",
                       striped_directory.c_str(), std::strerror(errno));
            (void)cleanup_scratch_target(worker, target);
            return false;
        } else {
            struct stat status {};
            if (::stat(striped_directory.c_str(), &status) != 0 ||
                !S_ISDIR(status.st_mode)) {
                log_worker(worker, stderr,
                           "%s exists but is not a directory",
                           striped_directory.c_str());
                (void)cleanup_scratch_target(worker, target);
                return false;
            }
        }

        const std::string shard_path = join_path(
            striped_directory,
            name + ".shard" + std::to_string(index));
        if (!create_scratch_file(worker, shard_path, shard_size)) {
            (void)cleanup_scratch_target(worker, target);
            return false;
        }
        target.files.push_back(shard_path);
    }
    target.uri = "striped://" + name + "?devs=" + mounts +
                 "&unit=" + std::to_string(stripe_unit);
    return true;
}

std::string format_device_ids(const std::vector<std::int32_t>& device_ids) {
    std::string result = "[";
    for (std::size_t index = 0; index < device_ids.size(); ++index) {
        if (index != 0) result += ",";
        result += std::to_string(device_ids[index]);
    }
    result += "]";
    return result;
}

struct DeviceBuffer {
    void* allocation = nullptr;
    void* aligned = nullptr;
    cudaStream_t stream = nullptr;

    ~DeviceBuffer() {
        if (stream != nullptr) (void)cudaStreamDestroy(stream);
        if (allocation != nullptr) (void)cudaFree(allocation);
    }
};

bool allocate_device_buffer(std::size_t worker, std::uint64_t size,
                            DeviceBuffer& buffer) {
    if (size > std::numeric_limits<std::size_t>::max() -
                   kDeviceBufferAlignment) {
        return false;
    }
    if (!check_cuda(
            worker,
            cudaMalloc(&buffer.allocation,
                       static_cast<std::size_t>(size +
                                                kDeviceBufferAlignment)),
            "cudaMalloc")) {
        return false;
    }
    const auto address = reinterpret_cast<std::uintptr_t>(buffer.allocation);
    buffer.aligned = reinterpret_cast<void*>(
        (address + kDeviceBufferAlignment - 1) &
        ~(static_cast<std::uintptr_t>(kDeviceBufferAlignment) - 1));
    return check_cuda(worker, cudaStreamCreate(&buffer.stream),
                      "cudaStreamCreate");
}

bool submit_and_wait(std::size_t worker, tutti::StorageRuntime& runtime,
                     const tutti::IoRequest& request,
                     const tutti::HostSubmitContext& context,
                     Barrier& submitted_barrier) {
    auto submitted = runtime.submit(&request, 1, context);
    const bool accepted =
        submitted.initial_states.size() == 1 &&
        submitted.initial_states.front().state ==
            tutti::IoRequestState::ACCEPTED;
    if (!submitted.status.ok()) {
        print_status(worker, "submit", submitted.status);
    }
    const bool peer_submitted = submitted_barrier.arrive_and_wait();
    if (!submitted.io.has_value()) {
        if (!submitted.initial_states.empty()) {
            print_status(worker, "request",
                         submitted.initial_states.front().status);
        }
        return false;
    }

    const tutti::IoHandle io = *submitted.io;
    const auto waited = runtime.wait(io, kIoTimeoutMs);
    const bool completed =
        waited.observation_status.ok() && waited.result.has_value() &&
        waited.result->state == tutti::IoState::COMPLETED &&
        waited.result->status.ok();
    if (!waited.observation_status.ok()) {
        print_status(worker, "wait", waited.observation_status);
    } else if (!waited.result.has_value()) {
        log_worker(worker, stderr, "wait failed: terminal result is missing");
    } else if (!waited.result->status.ok()) {
        print_status(worker, "I/O", waited.result->status);
    }

    const tutti::Status released = runtime.release_io(io);
    if (!released.ok()) print_status(worker, "release_io", released);
    return peer_submitted && submitted.status.ok() && accepted && completed &&
           released.ok();
}

bool verify_contents(std::size_t worker,
                     const std::vector<unsigned char>& expected,
                     const std::vector<unsigned char>& observed) {
    if (expected == observed) return true;
    std::size_t mismatch = 0;
    while (mismatch < expected.size() &&
           expected[mismatch] == observed[mismatch]) {
        ++mismatch;
    }
    log_worker(worker, stderr,
               "data verification failed at byte %zu: expected=%u "
               "observed=%u",
               mismatch, static_cast<unsigned>(expected[mismatch]),
               static_cast<unsigned>(observed[mismatch]));
    return false;
}

WorkerResult run_worker(std::size_t worker, const Options& options,
                        Barrier& start_barrier,
                        IoCoordinator& io_coordinator,
                        Barrier& write_submitted_barrier,
                        PhaseCoordinator& write_phase,
                        PhaseCoordinator& clear_phase,
                        Barrier& read_submitted_barrier) {
    WorkerResult result;
    if (!start_barrier.arrive_and_wait()) return result;
    const auto started = std::chrono::steady_clock::now();

    auto created = tutti::TuttiRuntime::create(options.config_paths[worker]);
    if (!created.ok()) {
        print_status(worker, "TuttiRuntime::create", created.status());
        (void)io_coordinator.arrive_and_wait(worker, false, -1);
        return result;
    }
    std::unique_ptr<tutti::TuttiRuntime> owner = std::move(created).value();
    tutti::StorageRuntime* runtime = owner->storage_runtime();
    result.accel_id = runtime == nullptr ? -1 : runtime->accel_id();

    bool setup_ok = runtime != nullptr && result.accel_id >= 0;
    if (runtime == nullptr) {
        log_worker(worker, stderr,
                   "TuttiRuntime did not create a StorageRuntime");
    }
    if (setup_ok) {
        setup_ok = check_cuda(worker, cudaSetDevice(result.accel_id),
                              "cudaSetDevice");
    }

    ScratchTarget scratch;
    bool scratch_created = false;
    if (setup_ok) {
        scratch_created = create_scratch_target(worker, options, scratch);
        setup_ok = scratch_created;
        result.target_uri = scratch.uri;
    }

    DeviceBuffer buffer;
    if (setup_ok) {
        setup_ok = allocate_device_buffer(worker, options.io_size, buffer);
    }

    std::vector<unsigned char> expected;
    if (setup_ok) {
        expected.resize(static_cast<std::size_t>(options.io_size));
        const std::size_t seed = worker == 0 ? 29 : 113;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            expected[index] =
                static_cast<unsigned char>((index * 17 + seed) % 251);
        }
        setup_ok = check_cuda(
            worker,
            cudaMemcpy(buffer.aligned, expected.data(), expected.size(),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy host-to-device");
    }

    tutti::TargetHandle target;
    tutti::MemoryHandle memory;
    if (setup_ok) {
        auto opened = runtime->open(scratch.uri, {});
        if (!opened.ok()) {
            print_status(worker, "open", opened.status());
            setup_ok = false;
        } else {
            target = opened.value();
        }
    }
    if (setup_ok) {
        auto registered = runtime->register_memory(tutti::MemoryView{
            buffer.aligned, options.io_size, tutti::MemoryKind::DEVICE,
            tutti::MemoryOwnership::CALLER_OWNED, result.accel_id,
            TUTTI_COMPILED_ACCELERATOR_PROFILE});
        if (!registered.ok()) {
            print_status(worker, "register_memory", registered.status());
            setup_ok = false;
        } else {
            memory = registered.value();
        }
    }

    bool ok = io_coordinator.arrive_and_wait(worker, setup_ok,
                                              result.accel_id);
    if (setup_ok && !ok) {
        log_worker(worker, stderr,
                   "parallel I/O cancelled: both Runtime setups must "
                   "succeed on different accelerators");
    }

    if (ok) {
        log_worker(worker, stdout,
                   "accelerator=%d devices=%s config=%s target=%s",
                   result.accel_id,
                   format_device_ids(options.device_ids[worker]).c_str(),
                   options.config_paths[worker].c_str(), scratch.uri.c_str());
        for (const std::string& path : scratch.files) {
            log_worker(worker, stdout, "backing file=%s", path.c_str());
        }
        const tutti::HostSubmitContext context{
            tutti::ExecutionDomain::DEVICE_EXECUTION, result.accel_id,
            buffer.stream};
        const tutti::IoRequest write{tutti::IoDirection::WRITE, memory, 0,
                                     target, 0, options.io_size};
        const bool write_ok = submit_and_wait(
            worker, *runtime, write, context, write_submitted_barrier);
        ok = write_phase.arrive_and_wait(write_ok);

        bool clear_ok = ok;
        if (clear_ok) {
            clear_ok = check_cuda(
                worker,
                cudaMemsetAsync(buffer.aligned, 0,
                                static_cast<std::size_t>(options.io_size),
                                buffer.stream),
                "cudaMemsetAsync");
            if (clear_ok) {
                clear_ok = check_cuda(
                    worker, cudaStreamSynchronize(buffer.stream),
                    "cudaStreamSynchronize");
            }
        }
        ok = clear_phase.arrive_and_wait(clear_ok);

        if (ok) {
            const tutti::IoRequest read{tutti::IoDirection::READ, memory, 0,
                                        target, 0, options.io_size};
            ok = submit_and_wait(worker, *runtime, read, context,
                                 read_submitted_barrier);
        }
        if (ok) {
            std::vector<unsigned char> observed(expected.size());
            ok = check_cuda(
                     worker,
                     cudaMemcpy(observed.data(), buffer.aligned,
                                observed.size(), cudaMemcpyDeviceToHost),
                     "cudaMemcpy device-to-host") &&
                 verify_contents(worker, expected, observed);
        }
    }

    if (target.valid()) {
        const tutti::Status closed = runtime->close(target);
        if (!closed.ok()) print_status(worker, "close", closed);
        ok = closed.ok() && ok;
    }
    if (memory.valid()) {
        const tutti::Status unregistered = runtime->unregister_memory(memory);
        if (!unregistered.ok()) {
            print_status(worker, "unregister_memory", unregistered);
        }
        ok = unregistered.ok() && ok;
    }

    const tutti::Status shutdown = owner->shutdown();
    if (!shutdown.ok()) {
        print_status(worker, "TuttiRuntime::shutdown", shutdown);
    }
    ok = shutdown.ok() && ok;

    if (scratch_created && !options.keep_files) {
        ok = cleanup_scratch_target(worker, scratch) && ok;
    }

    result.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    result.ok = ok;
    log_worker(worker, stdout,
               "hardware write/read verification: %s (%lld ms)",
               ok ? "PASS" : "FAIL",
               static_cast<long long>(result.elapsed.count()));
    return result;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 2;
    }
    if (!validate_config_pair(options)) return 2;

    Barrier start_barrier(kRuntimeCount);
    IoCoordinator io_coordinator;
    Barrier write_submitted_barrier(kRuntimeCount);
    PhaseCoordinator write_phase;
    PhaseCoordinator clear_phase;
    Barrier read_submitted_barrier(kRuntimeCount);
    std::array<WorkerResult, kRuntimeCount> results;
    std::array<std::thread, kRuntimeCount> workers;
    for (std::size_t worker = 0; worker < kRuntimeCount; ++worker) {
        workers[worker] = std::thread([&, worker] {
            try {
                results[worker] = run_worker(
                    worker, options, start_barrier, io_coordinator,
                    write_submitted_barrier, write_phase, clear_phase,
                    read_submitted_barrier);
            } catch (const std::exception& exception) {
                log_worker(worker, stderr, "unexpected exception: %s",
                           exception.what());
                start_barrier.cancel();
                io_coordinator.cancel();
                write_submitted_barrier.cancel();
                write_phase.cancel();
                clear_phase.cancel();
                read_submitted_barrier.cancel();
            } catch (...) {
                log_worker(worker, stderr,
                           "unexpected non-standard exception");
                start_barrier.cancel();
                io_coordinator.cancel();
                write_submitted_barrier.cancel();
                write_phase.cancel();
                clear_phase.cancel();
                read_submitted_barrier.cancel();
            }
        });
    }
    for (std::thread& worker : workers) worker.join();

    const bool distinct_accelerators =
        results[0].accel_id >= 0 && results[1].accel_id >= 0 &&
        results[0].accel_id != results[1].accel_id;
    const bool ok = results[0].ok && results[1].ok && distinct_accelerators;
    const std::string devices_0 = format_device_ids(options.device_ids[0]);
    const std::string devices_1 = format_device_ids(options.device_ids[1]);
    std::printf(
        "Two TuttiRuntime instances: accelerator %d -> NVMe devices %s, "
        "accelerator %d -> NVMe devices %s: %s\n",
        results[0].accel_id, devices_0.c_str(), results[1].accel_id,
        devices_1.c_str(), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#include <tutti/tutti_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <tutti/cuda_like.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>
#include <tutti/storage_runtime.h>

namespace {

constexpr std::uint64_t kDefaultIoSize = 4 * 1024 * 1024;
constexpr std::uint64_t kDeviceBufferAlignment = 64 * 1024;
constexpr std::uint64_t kBlockAlignment = 4096;
constexpr std::uint64_t kIoTimeoutMs = 30000;

struct Options {
    std::string config_path = TUTTI_RUNTIME_DEFAULT_CONFIG;
    std::vector<std::string> directories;
    std::uint64_t io_size = kDefaultIoSize;
    bool keep_file = false;
};

void print_usage(const char* program) {
    std::fprintf(
        stderr,
        "Usage: %s --directory PATH [--directory PATH ...] [--config PATH] "
        "[--size BYTES] [--keep-file]\n"
        "\n"
        "Pass one daemon-published accelerator view for local NVMe or two "
        "or more views for multi-device NVMe (files rotate across them).\n",
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
        if (argument == "--keep-file") {
            options.keep_file = true;
            continue;
        }
        if (index + 1 >= argc) return false;
        const char* value = argv[++index];
        if (argument == "--config") {
            options.config_path = value;
        } else if (argument == "--directory") {
            options.directories.emplace_back(value);
        } else if (argument == "--size") {
            if (!parse_u64(value, options.io_size)) return false;
        } else {
            return false;
        }
    }
    return !options.config_path.empty() &&
           options.directories.size() >= 1 &&
           options.directories.size() <= 4 &&
           options.io_size > 0 && options.io_size % kBlockAlignment == 0 &&
           options.io_size <= std::numeric_limits<std::size_t>::max();
}

void print_status(const char* operation, const tutti::Status& status) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 status.message().c_str());
}

bool check_cuda(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(result));
    return false;
}

std::string join_path(const std::string& directory, const std::string& name) {
    std::string path = directory;
    if (!path.empty() && path.back() != '/') path.push_back('/');
    path += name;
    return path;
}

bool create_scratch_file(const std::string& path, std::uint64_t size) {
    const int fd = ::open(path.c_str(),
                          O_CREAT | O_EXCL | O_RDWR | O_DIRECT,
                          0600);
    if (fd < 0) {
        std::fprintf(stderr, "open(%s) failed: %s\n", path.c_str(),
                     std::strerror(errno));
        return false;
    }

    constexpr std::size_t kChunkSize = 1024 * 1024;
    void* zeros = nullptr;
    if (::posix_memalign(&zeros, kBlockAlignment, kChunkSize) != 0) {
        std::fprintf(stderr, "failed to allocate aligned file buffer\n");
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
            std::fprintf(stderr, "write(%s) failed: %s\n", path.c_str(),
                         result < 0 ? std::strerror(errno) : "short write");
            ok = false;
            break;
        }
        written += chunk;
    }
    if (ok && ::fsync(fd) != 0) {
        std::fprintf(stderr, "fsync(%s) failed: %s\n", path.c_str(),
                     std::strerror(errno));
        ok = false;
    }
    std::free(zeros);
    if (::close(fd) != 0) ok = false;
    if (!ok) ::unlink(path.c_str());
    return ok;
}

struct ScratchTarget {
    std::vector<std::string> uris;
    std::vector<std::uint64_t> per_file_sizes;
    std::vector<std::string> files;
    std::vector<std::string> created_directories;
};

bool cleanup_scratch_target(const ScratchTarget& target) {
    bool ok = true;
    for (const std::string& path : target.files) {
        if (::unlink(path.c_str()) != 0 && errno != ENOENT) {
            std::fprintf(stderr, "unlink(%s) failed: %s\n", path.c_str(),
                         std::strerror(errno));
            ok = false;
        }
    }
    for (auto directory = target.created_directories.rbegin();
         directory != target.created_directories.rend(); ++directory) {
        if (::rmdir(directory->c_str()) != 0 && errno != ENOENT) {
            std::fprintf(stderr, "rmdir(%s) failed: %s\n", directory->c_str(),
                         std::strerror(errno));
            ok = false;
        }
    }
    return ok;
}

bool create_scratch_target(const Options& options, ScratchTarget& target) {
    const std::string name =
        "tutti_runtime_example." + std::to_string(::getpid()) + ".bin";
    const std::size_t device_count = options.directories.size();
    // The whole I/O is split evenly across the directories, one file each --
    // the shape a rotating multi-device layout produces (a single IO never
    // spans devices; the batch spans them instead).
    std::uint64_t per_file = options.io_size / device_count;
    per_file = (per_file + kBlockAlignment - 1) / kBlockAlignment * kBlockAlignment;
    if (per_file == 0) {
        std::fprintf(stderr,
                     "I/O size too small to split across %zu directories\n",
                     device_count);
        return false;
    }
    const std::uint64_t last_file =
        options.io_size - per_file * (device_count - 1);

    for (std::size_t index = 0; index < device_count; ++index) {
        const std::uint64_t file_size = index + 1 == device_count
                                             ? last_file
                                             : per_file;
        const std::string path = join_path(
            options.directories[index],
            name + "." + std::to_string(index));
        if (!create_scratch_file(path, file_size)) {
            (void)cleanup_scratch_target(target);
            return false;
        }
        target.files.push_back(path);
        target.per_file_sizes.push_back(file_size);
        target.uris.push_back("file://" + path);
    }
    return true;
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

bool allocate_device_buffer(std::uint64_t size, DeviceBuffer& buffer) {
    if (size > std::numeric_limits<std::size_t>::max() -
                   kDeviceBufferAlignment) {
        return false;
    }
    if (!check_cuda(
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
    return check_cuda(cudaStreamCreate(&buffer.stream), "cudaStreamCreate");
}

bool submit_and_wait(tutti::StorageRuntime& runtime,
                     const std::vector<tutti::IoRequest>& requests,
                     const tutti::HostSubmitContext& context) {
    auto submitted = runtime.submit(requests.data(), requests.size(), context);
    bool accepted = submitted.initial_states.size() == requests.size();
    if (accepted) {
        for (const auto& state : submitted.initial_states) {
            if (state.state != tutti::IoRequestState::ACCEPTED) accepted = false;
        }
    }
    if (!submitted.status.ok()) {
        print_status("submit", submitted.status);
    }
    if (!submitted.io.has_value()) {
        if (!submitted.initial_states.empty()) {
            print_status("request", submitted.initial_states.front().status);
        }
        return false;
    }

    const tutti::IoHandle io = *submitted.io;
    const auto waited = runtime.wait(io, kIoTimeoutMs);
    bool ok = waited.observation_status.ok() && waited.result.has_value() &&
              waited.result->state == tutti::IoState::COMPLETED &&
              waited.result->status.ok();
    if (!waited.observation_status.ok()) {
        print_status("wait", waited.observation_status);
    } else if (!waited.result.has_value()) {
        std::fprintf(stderr, "wait failed: terminal result is missing\n");
    } else if (!waited.result->status.ok()) {
        print_status("I/O", waited.result->status);
    }

    const tutti::Status released = runtime.release_io(io);
    if (!released.ok()) print_status("release_io", released);
    return submitted.status.ok() && accepted && ok && released.ok();
}

bool verify_contents(const std::vector<unsigned char>& expected,
                     const std::vector<unsigned char>& observed) {
    if (expected == observed) return true;
    std::size_t mismatch = 0;
    while (mismatch < expected.size() &&
           expected[mismatch] == observed[mismatch]) {
        ++mismatch;
    }
    std::fprintf(stderr,
                 "data verification failed at byte %zu: expected=%u "
                 "observed=%u\n",
                 mismatch, static_cast<unsigned>(expected[mismatch]),
                 static_cast<unsigned>(observed[mismatch]));
    return false;
}

bool run_io(tutti::StorageRuntime& runtime, const Options& options,
            const ScratchTarget& scratch) {
    const std::int32_t accel_id = runtime.accel_id();
    if (accel_id < 0 ||
        !check_cuda(cudaSetDevice(accel_id), "cudaSetDevice")) {
        return false;
    }

    DeviceBuffer buffer;
    if (!allocate_device_buffer(options.io_size, buffer)) return false;

    std::vector<unsigned char> expected(
        static_cast<std::size_t>(options.io_size));
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<unsigned char>((index * 17 + 29) % 251);
    }
    if (!check_cuda(cudaMemcpy(buffer.aligned, expected.data(), expected.size(),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy host-to-device")) {
        return false;
    }

    std::vector<tutti::TargetHandle> targets(scratch.uris.size());
    tutti::MemoryHandle memory;
    bool ok = false;
    do {
        for (std::size_t index = 0; index < scratch.uris.size(); ++index) {
            auto opened = runtime.open(scratch.uris[index], {});
            if (!opened.ok()) {
                print_status("open", opened.status());
                break;
            }
            targets[index] = opened.value();
        }
        if (std::any_of(targets.begin(), targets.end(),
                        [](const tutti::TargetHandle& t) { return !t.valid(); })) {
            break;
        }

        auto registered = runtime.register_memory(tutti::MemoryView{
            buffer.aligned, options.io_size, tutti::MemoryKind::DEVICE,
            tutti::MemoryOwnership::CALLER_OWNED, accel_id,
            TUTTI_COMPILED_ACCELERATOR_PROFILE});
        if (!registered.ok()) {
            print_status("register_memory", registered.status());
            break;
        }
        memory = registered.value();

        // One batch across all targets -- one request per file, exactly the
        // multi-device submission shape production uses.
        const tutti::HostSubmitContext context{
            tutti::ExecutionDomain::DEVICE_EXECUTION, accel_id, buffer.stream};
        std::vector<tutti::IoRequest> batch(targets.size());
        std::uint64_t memory_offset = 0;
        for (std::size_t index = 0; index < batch.size(); ++index) {
            batch[index] = {tutti::IoDirection::WRITE, memory, memory_offset,
                            targets[index], 0,
                            scratch.per_file_sizes[index]};
            memory_offset += scratch.per_file_sizes[index];
        }
        if (!submit_and_wait(runtime, batch, context)) break;

        if (!check_cuda(cudaMemset(buffer.aligned, 0,
                                   static_cast<std::size_t>(options.io_size)),
                        "cudaMemset")) {
            break;
        }
        for (auto& request : batch) request.direction = tutti::IoDirection::READ;
        if (!submit_and_wait(runtime, batch, context)) break;

        std::vector<unsigned char> observed(expected.size());
        if (!check_cuda(cudaMemcpy(observed.data(), buffer.aligned,
                                   observed.size(), cudaMemcpyDeviceToHost),
                        "cudaMemcpy device-to-host")) {
            break;
        }
        ok = verify_contents(expected, observed);
    } while (false);

    for (auto& target : targets) {
        if (target.valid()) {
            const tutti::Status closed = runtime.close(target);
            if (!closed.ok()) print_status("close", closed);
            ok = closed.ok() && ok;
        }
    }
    if (memory.valid()) {
        const tutti::Status unregistered = runtime.unregister_memory(memory);
        if (!unregistered.ok()) {
            print_status("unregister_memory", unregistered);
        }
        ok = unregistered.ok() && ok;
    }
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 2;
    }

    auto created = tutti::TuttiRuntime::create(options.config_path);
    if (!created.ok()) {
        print_status("TuttiRuntime::create", created.status());
        return 1;
    }
    std::unique_ptr<tutti::TuttiRuntime> owner = std::move(created).value();
    tutti::StorageRuntime* runtime = owner->storage_runtime();
    if (runtime == nullptr) {
        std::fprintf(stderr, "TuttiRuntime did not create a StorageRuntime\n");
        (void)owner->shutdown();
        return 1;
    }

    ScratchTarget scratch;
    std::printf("Config: %s\n", options.config_path.c_str());
    std::printf("I/O size: %llu bytes\n",
                static_cast<unsigned long long>(options.io_size));

    const bool scratch_created = create_scratch_target(options, scratch);
    if (scratch_created) {
        for (const std::string& uri : scratch.uris) {
            std::printf("Target URI: %s\n", uri.c_str());
        }
        for (const std::string& path : scratch.files) {
            std::printf("Backing file: %s\n", path.c_str());
        }
    }
    bool ok = scratch_created;
    if (scratch_created) ok = run_io(*runtime, options, scratch);

    const tutti::Status shutdown = owner->shutdown();
    if (!shutdown.ok()) print_status("TuttiRuntime::shutdown", shutdown);
    ok = shutdown.ok() && ok;

    if (scratch_created && !options.keep_file) {
        ok = cleanup_scratch_target(scratch) && ok;
    }

    std::printf("TuttiRuntime %slocal NVMe hardware I/O: %s\n",
                options.directories.size() == 2 ? "striped " : "",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

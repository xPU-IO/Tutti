#include <tutti/tutti_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <tutti/cuda_like.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>
#include <tutti/storage_runtime.h>

namespace {

constexpr int kSkip = 77;
constexpr std::uint64_t kDefaultIoSize = 4 * 1024 * 1024;
constexpr std::uint64_t kDeviceBufferAlignment = 64 * 1024;
constexpr std::uint64_t kIoTimeoutMs = 30000;

struct Options {
    std::string config_path;
    std::string uri;
    std::uint64_t io_size = kDefaultIoSize;
};

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
        if (index + 1 >= argc) return false;
        const char* value = argv[++index];
        if (argument == "--config") {
            options.config_path = value;
        } else if (argument == "--uri") {
            options.uri = value;
        } else if (argument == "--size") {
            if (!parse_u64(value, options.io_size)) return false;
        } else {
            return false;
        }
    }
    return !options.config_path.empty() && !options.uri.empty() &&
           options.io_size > 0 &&
           options.io_size <= std::numeric_limits<std::size_t>::max();
}

void print_status(const char* operation, const tutti::Status& status) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 status.message().c_str());
}

bool deployment_unavailable(const tutti::Status& status) {
    switch (status.code()) {
    case tutti::StatusCode::NOT_FOUND:
    case tutti::StatusCode::UNSUPPORTED:
    case tutti::StatusCode::NOT_READY:
    case tutti::StatusCode::RESOURCE_EXHAUSTED:
    case tutti::StatusCode::DEVICE_ERROR:
        return true;
    default:
        return false;
    }
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
    if (size > std::numeric_limits<std::uint64_t>::max() -
                   kDeviceBufferAlignment) {
        return false;
    }
    if (cudaMalloc(&buffer.allocation,
                   static_cast<std::size_t>(size + kDeviceBufferAlignment)) !=
        cudaSuccess) {
        return false;
    }
    const auto address = reinterpret_cast<std::uintptr_t>(buffer.allocation);
    buffer.aligned = reinterpret_cast<void*>(
        (address + kDeviceBufferAlignment - 1) &
        ~(static_cast<std::uintptr_t>(kDeviceBufferAlignment) - 1));
    return cudaStreamCreate(&buffer.stream) == cudaSuccess;
}

bool submit_and_wait(tutti::StorageRuntime& runtime,
                     const tutti::IoRequest& request,
                     const tutti::HostSubmitContext& context) {
    auto submitted = runtime.submit(&request, 1, context);
    if (!submitted.status.ok() || !submitted.io.has_value() ||
        submitted.initial_states.size() != 1 ||
        submitted.initial_states.front().state !=
            tutti::IoRequestState::ACCEPTED) {
        print_status("submit", submitted.status);
        return false;
    }
    const tutti::IoHandle io = *submitted.io;
    auto waited = runtime.wait(io, kIoTimeoutMs);
    const bool completed = waited.observation_status.ok() &&
        waited.result.has_value() &&
        waited.result->state == tutti::IoState::COMPLETED &&
        waited.result->status.ok();
    if (!completed) {
        if (!waited.observation_status.ok()) {
            print_status("wait", waited.observation_status);
        } else if (waited.result.has_value()) {
            print_status("I/O", waited.result->status);
        } else {
            std::fprintf(stderr, "wait failed: terminal result is missing\n");
        }
    }
    const tutti::Status released = runtime.release_io(io);
    if (!released.ok()) print_status("release_io", released);
    return completed && released.ok();
}

bool run_io(tutti::TuttiRuntime& owner, const Options& options) {
    tutti::StorageRuntime* runtime = owner.storage_runtime();
    if (runtime == nullptr) return false;
    const std::int32_t accel_id = runtime->accel_id();
    if (accel_id < 0 || cudaSetDevice(accel_id) != cudaSuccess) return false;

    DeviceBuffer buffer;
    if (!allocate_device_buffer(options.io_size, buffer)) return false;
    std::vector<unsigned char> expected(
        static_cast<std::size_t>(options.io_size));
    for (std::size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<unsigned char>((index * 17 + 29) % 251);
    }
    if (cudaMemcpy(buffer.aligned, expected.data(), expected.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    auto opened = runtime->open(options.uri, {});
    if (!opened.ok()) {
        print_status("open", opened.status());
        return false;
    }
    const tutti::TargetHandle target = opened.value();
    auto registered = runtime->register_memory(tutti::MemoryView{
        buffer.aligned, options.io_size, tutti::MemoryKind::DEVICE,
        tutti::MemoryOwnership::CALLER_OWNED, accel_id,
        TUTTI_COMPILED_ACCELERATOR_PROFILE});
    if (!registered.ok()) {
        print_status("register_memory", registered.status());
        (void)runtime->close(target);
        return false;
    }
    const tutti::MemoryHandle memory = registered.value();
    const tutti::HostSubmitContext context{
        tutti::ExecutionDomain::DEVICE_EXECUTION, accel_id, buffer.stream};

    const tutti::IoRequest write{
        tutti::IoDirection::WRITE, memory, 0, target, 0, options.io_size};
    bool ok = submit_and_wait(*runtime, write, context);
    ok = cudaMemset(buffer.aligned, 0, static_cast<std::size_t>(options.io_size)) ==
             cudaSuccess && ok;
    const tutti::IoRequest read{
        tutti::IoDirection::READ, memory, 0, target, 0, options.io_size};
    ok = submit_and_wait(*runtime, read, context) && ok;

    std::vector<unsigned char> observed(expected.size());
    ok = cudaMemcpy(observed.data(), buffer.aligned, observed.size(),
                    cudaMemcpyDeviceToHost) == cudaSuccess && ok;
    ok = observed == expected && ok;

    const tutti::Status unregistered = runtime->unregister_memory(memory);
    const tutti::Status closed = runtime->close(target);
    if (!unregistered.ok()) print_status("unregister_memory", unregistered);
    if (!closed.ok()) print_status("close", closed);
    return ok && closed.ok() && unregistered.ok();
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::printf(
            "SKIP: --config PATH and --uri URI are required for hardware E2E\n");
        return kSkip;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        std::printf("SKIP: no accelerator device is available\n");
        return kSkip;
    }

    auto created = tutti::TuttiRuntime::create(options.config_path);
    if (!created.ok()) {
        if (deployment_unavailable(created.status())) {
            std::printf("SKIP: Runtime deployment is unavailable: %s\n",
                        created.status().message().c_str());
            return kSkip;
        }
        print_status("TuttiRuntime::create", created.status());
        return 1;
    }
    std::unique_ptr<tutti::TuttiRuntime> runtime = std::move(created).value();
    bool ok = run_io(*runtime, options);
    const tutti::Status shutdown = runtime->shutdown();
    const tutti::Status repeated_shutdown = runtime->shutdown();
    if (!shutdown.ok()) print_status("shutdown", shutdown);
    runtime.reset();
    ok = shutdown.ok() && repeated_shutdown.ok() && ok;

    // Reacquiring the same logical resource through the public path proves
    // that shutdown released the first Runtime's allocation lease.
    auto reacquired = tutti::TuttiRuntime::create(options.config_path);
    if (!reacquired.ok()) {
        print_status("TuttiRuntime::create after shutdown", reacquired.status());
        ok = false;
    } else {
        auto second = std::move(reacquired).value();
        ok = second->shutdown().ok() && ok;
    }

    std::printf("TuttiRuntime hardware E2E: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

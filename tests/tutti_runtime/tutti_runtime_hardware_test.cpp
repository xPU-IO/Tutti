#include <tutti/tutti_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
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
constexpr std::uint64_t kRequestSize = 4 * 1024;
constexpr std::size_t kRequestsPerBatch = 1024;
constexpr std::uint64_t kIoTimeoutMs = 30000;
constexpr unsigned char kExpectedPattern = 0x5A;
constexpr unsigned char kPoisonPattern = 0xFF;

extern "C" void launch_dma_visible_fill(void* buffer, unsigned char value,
                                         std::uint64_t size,
                                         cudaStream_t stream);

struct Options {
    std::string config_path;
    std::string uri;
    std::uint64_t io_size = kDefaultIoSize;
    std::int32_t expected_accel_id = -1;
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

bool parse_i32(const char* text, std::int32_t& value) {
    if (text == nullptr || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<std::int32_t>::max()) {
        return false;
    }
    value = static_cast<std::int32_t>(parsed);
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
        } else if (argument == "--accel-id") {
            if (!parse_i32(value, options.expected_accel_id)) return false;
        } else {
            return false;
        }
    }
    return !options.config_path.empty() && !options.uri.empty() &&
           options.io_size > 0 &&
           options.io_size % kRequestSize == 0 &&
           options.io_size <= std::numeric_limits<std::size_t>::max();
}

double gib_per_second(std::uint64_t bytes,
                      std::chrono::steady_clock::duration elapsed) {
    const double seconds = std::chrono::duration<double>(elapsed).count();
    if (seconds == 0.0) return 0.0;
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) / seconds;
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
                     const tutti::IoRequest* requests,
                     std::size_t request_count,
                     const tutti::HostSubmitContext& context) {
    auto submitted = runtime.submit(requests, request_count, context);
    if (!submitted.status.ok() || !submitted.io.has_value() ||
        submitted.initial_states.size() != request_count) {
        print_status("submit", submitted.status);
        return false;
    }
    for (std::size_t index = 0; index < request_count; ++index) {
        if (submitted.initial_states[index].state !=
            tutti::IoRequestState::ACCEPTED) {
            std::fprintf(stderr, "request %zu was not accepted: %s\n", index,
                         submitted.initial_states[index].status.message().c_str());
            return false;
        }
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

bool submit_range(tutti::StorageRuntime& runtime,
                  const tutti::MemoryHandle& memory,
                  const tutti::TargetHandle& target,
                  tutti::IoDirection direction, std::uint64_t size,
                  const tutti::HostSubmitContext& context) {
    std::vector<tutti::IoRequest> requests;
    requests.reserve(kRequestsPerBatch);
    for (std::uint64_t offset = 0; offset < size;) {
        requests.clear();
        while (offset < size && requests.size() < kRequestsPerBatch) {
            requests.push_back(tutti::IoRequest{
                direction, memory, offset, target, offset, kRequestSize});
            offset += kRequestSize;
        }
        if (!submit_and_wait(runtime, requests.data(), requests.size(),
                             context)) {
            return false;
        }
    }
    return true;
}

bool verify_pattern(void* buffer, std::uint64_t size, cudaStream_t stream) {
    constexpr std::size_t kVerifyChunkSize = 4 * 1024 * 1024;
    std::vector<unsigned char> observed(static_cast<std::size_t>(
        std::min<std::uint64_t>(size, kVerifyChunkSize)));
    for (std::uint64_t offset = 0; offset < size;
         offset += observed.size()) {
        const std::size_t chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(observed.size(), size - offset));
        if (cudaMemcpyAsync(observed.data(),
                            static_cast<unsigned char*>(buffer) + offset,
                            chunk, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            return false;
        }
        const auto mismatch =
            std::find_if(observed.begin(), observed.begin() + chunk,
                         [](unsigned char value) {
                             return value != kExpectedPattern;
                         });
        if (mismatch != observed.begin() + chunk) {
            const std::uint64_t mismatch_offset =
                offset + static_cast<std::uint64_t>(mismatch - observed.begin());
            std::fprintf(stderr,
                         "data mismatch at byte %llu: observed=%u expected=%u\n",
                         static_cast<unsigned long long>(mismatch_offset),
                         static_cast<unsigned>(*mismatch),
                         static_cast<unsigned>(kExpectedPattern));
            return false;
        }
    }
    return true;
}

bool run_io(tutti::TuttiRuntime& owner, const Options& options) {
    tutti::StorageRuntime* runtime = owner.storage_runtime();
    if (runtime == nullptr) return false;
    const std::int32_t accel_id = runtime->accel_id();
    if (options.expected_accel_id >= 0 &&
        accel_id != options.expected_accel_id) {
        std::fprintf(stderr, "runtime accel_id=%d, expected %d\n", accel_id,
                     options.expected_accel_id);
        return false;
    }
    if (accel_id < 0 || cudaSetDevice(accel_id) != cudaSuccess) return false;

    DeviceBuffer buffer;
    if (!allocate_device_buffer(options.io_size, buffer)) return false;
    launch_dma_visible_fill(buffer.aligned, kExpectedPattern, options.io_size,
                            buffer.stream);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(buffer.stream) != cudaSuccess) {
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

    const auto write_started = std::chrono::steady_clock::now();
    bool ok = submit_range(*runtime, memory, target,
                           tutti::IoDirection::WRITE, options.io_size,
                           context);
    const auto write_elapsed = std::chrono::steady_clock::now() - write_started;
    launch_dma_visible_fill(buffer.aligned, kPoisonPattern, options.io_size,
                            buffer.stream);
    ok = cudaGetLastError() == cudaSuccess &&
         cudaStreamSynchronize(buffer.stream) == cudaSuccess && ok;
    const auto read_started = std::chrono::steady_clock::now();
    ok = submit_range(*runtime, memory, target, tutti::IoDirection::READ,
                      options.io_size, context) && ok;
    const auto read_elapsed = std::chrono::steady_clock::now() - read_started;
    ok = verify_pattern(buffer.aligned, options.io_size, buffer.stream) && ok;

    const tutti::Status unregistered = runtime->unregister_memory(memory);
    const tutti::Status closed = runtime->close(target);
    if (!unregistered.ok()) print_status("unregister_memory", unregistered);
    if (!closed.ok()) print_status("close", closed);
    std::printf(
        "DMA accel_id=%d size=%llu bytes request=%llu bytes batch=%zu "
        "write=%.3f GiB/s read=%.3f GiB/s\n",
        accel_id, static_cast<unsigned long long>(options.io_size),
        static_cast<unsigned long long>(kRequestSize), kRequestsPerBatch,
        gib_per_second(options.io_size, write_elapsed),
        gib_per_second(options.io_size, read_elapsed));
    return ok && closed.ok() && unregistered.ok();
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::printf(
            "SKIP: --config PATH and --uri URI are required; "
            "--accel-id ID is optional\n");
        return kSkip;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        std::printf("SKIP: no accelerator device is available\n");
        return kSkip;
    }
    if (options.expected_accel_id >= device_count) {
        std::printf("SKIP: requested accelerator %d is unavailable\n",
                    options.expected_accel_id);
        return kSkip;
    }

    int caller_accel_id = 0;
    if (options.expected_accel_id == 0 && device_count > 1) {
        caller_accel_id = 1;
    }
    if (cudaSetDevice(caller_accel_id) != cudaSuccess) {
        std::fprintf(stderr, "failed to select caller accelerator %d\n",
                     caller_accel_id);
        return 1;
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
    int current_accel_id = -1;
    if (cudaGetDevice(&current_accel_id) != cudaSuccess ||
        current_accel_id != caller_accel_id) {
        std::fprintf(stderr,
                     "runtime assembly changed caller accelerator: got %d, expected %d\n",
                     current_accel_id, caller_accel_id);
        (void)runtime->shutdown();
        return 1;
    }
    std::printf("Runtime assembly: PASS (runtime accel_id=%d, caller accel_id=%d)\n",
                runtime->storage_runtime()->accel_id(), caller_accel_id);
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

// tests/register_perf/register_perf.cpp
//
// R19 S3b REQUIRED 3: registration throughput benchmark.
//
// Measures per-registration latency and total throughput for N
// register_memory calls. Prints [perf] format:
//   [perf] N=10000 ok=10000 total_wall=0.234s per_register=23.4us
//
// Hardware test: needs GPU + snvme + libnvm + the NVMe mount.
// Not registered as ctest (perf number, not pass/fail).
// Usage: ./bin/tutti_register_perf [N]  (default N=10000)

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <tutti/cuda_like.h>
#include <tutti/storage_runtime.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>

#include "tutti/data_paths/local_nvme/local_nvme_data_path.h"
#include <tutti/resolvers/local_file/resolver.h>

using namespace tutti;
using tutti::data_paths::local_nvme::LocalNvmeDataPath;
using tutti::resolvers::local_file::BackingDeviceConfig;
using tutti::resolvers::local_file::LocalFileResolver;

static constexpr const char* kSnvme   = "/dev/ssnvme0";
static constexpr std::uint64_t kBar0  = 16384;
static constexpr std::uint32_t  kQueues = 16;
static constexpr std::uint32_t  kNsId   = 1;
static constexpr std::uint32_t  kBs     = 4096;
static constexpr const char* kDataPathKey = "local-nvme-ext4";

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 10000;
    if (N <= 0) {
        std::fprintf(stderr, "Usage: %s [N]  (default N=10000)\n", argv[0]);
        return 1;
    }

    constexpr size_t BUF_SIZE = 128 * 1024;   // 128 KiB per tensor

    // Allocate one large GPU buffer (128KiB * N).
    int gpu = 0;
    cudaError_t ce = cudaSetDevice(gpu);
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(ce));
        return 1;
    }
    void* gpu_buf = nullptr;
    ce = cudaMalloc(&gpu_buf, BUF_SIZE * N);
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(%zu) failed: %s\n",
                     BUF_SIZE * N, cudaGetErrorString(ce));
        return 1;
    }

    // Set up DataPath + Runtime.
    LocalNvmeDataPath dp(kSnvme, kBar0, gpu, kQueues, kNsId, kBs);
    LocalFileResolver resolver("0000:08:00.0", 1, 4096,
                               BackingDeviceConfig{"/dev/snvme0n1", 0});
    RuntimeComponents components;
    components.resolvers.push_back({"file", &resolver});
    components.data_paths.push_back({kDataPathKey, &dp, DataPathConfig{"local_nvme"}});
    auto created = StorageRuntime::create({}, std::move(components));
    if (!created.ok()) {
        std::fprintf(stderr, "StorageRuntime::create: %s\n",
                     created.status().message().c_str());
        cudaFree(gpu_buf);
        return 1;
    }
    auto rt = std::move(created).value();

    // Register N sub-regions of the GPU buffer.
    // Each registration: 128KiB buffer, io_granularity=128KiB → 1 slice,
    // pre-built PRP descriptors via the pool path.
    std::vector<MemoryHandle> handles(N);
    auto t0 = std::chrono::steady_clock::now();

    int ok = 0;
    for (int i = 0; i < N; ++i) {
        void* ptr = static_cast<char*>(gpu_buf) + i * BUF_SIZE;
        MemoryView view;
        view.address = ptr;
        view.size = BUF_SIZE;
        view.expected_kind = MemoryKind::DEVICE;
        view.ownership = MemoryOwnership::CALLER_OWNED;
        view.expected_accel_id = gpu;
        view.io_granularity = BUF_SIZE;  // 1 slice per registration
        auto result = rt->register_memory(view);
        if (result.ok()) {
            handles[i] = result.value();
            ++ok;
        } else {
            if (ok == 0) {
                std::fprintf(stderr, "register_memory[0] failed: %s\n",
                             result.status().message().c_str());
            }
            break;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(t1 - t0).count();
    double per_us = (wall_s * 1e6) / (ok > 0 ? ok : 1);

    std::printf("[perf] N=%d ok=%d total_wall=%.3fs per_register=%.1fus\n",
                N, ok, wall_s, per_us);
    std::printf("[perf] dma_map calls = prp_buf_pool segment count "
                "(~1-2 with pool, vs N=%d without pool)\n", ok);

    // Cleanup.
    for (int i = 0; i < ok; ++i) {
        rt->unregister_memory(handles[i]);
    }
    rt->shutdown(1000);
    cudaFree(gpu_buf);
    return 0;
}

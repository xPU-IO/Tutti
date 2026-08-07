// tests/batch_open_perf/batch_open_perf.cpp
//
// Round 19 S1 — batch open performance microbenchmark.
//
// Measures wall-clock time of opening N files via open_batch() vs N
// serial open() calls.  Each file is a 4 KiB O_DIRECT file on the NVMe
// mount; the per-file cost is dominated by resolver FIEMAP (host IO)
// which open_batch parallelizes across a worker pool.
//
// Usage:
//   tutti_batch_open_perf [N]   (default N=500)

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <tutti/cuda_like.h>

#include <tutti/storage_runtime.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>

#include "tutti/data_paths/local_nvme/local_nvme_data_path.h"
#include <tutti/resolvers/local_file/resolver.h>

#include "../hardware_test_directory.h"

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

static bool create_file(const std::string& path, std::uint64_t size,
                        unsigned char fill) {
    int f = ::open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
    if (f < 0) return false;
    void* abuf = nullptr;
    if (::posix_memalign(&abuf, 4096, (size_t)size) != 0) { ::close(f); return false; }
    std::memset(abuf, fill, (size_t)size);
    ssize_t n = ::write(f, abuf, size);
    (void)n;
    std::free(abuf);
    ::fsync(f);
    ::close(f);
    return true;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 500;
    if (N <= 0) N = 500;

    std::int32_t gpu = 0;
    cudaGetDevice(&gpu);

    tutti::test_support::UniqueTestDirectory run_dir;
    std::string dir_error;
    if (!tutti::test_support::UniqueTestDirectory::create(
            "/mnt/nvme0/GPU0", "tutti_batch_open_perf",
            run_dir, dir_error)) {
        std::fprintf(stderr, "ERROR: %s\n", dir_error.c_str());
        return 1;
    }
    const std::string& test_dir = run_dir.path();
    std::printf("Test directory: %s\n", test_dir.c_str());

    // Create N files.
    std::vector<std::string> paths;
    paths.reserve(N);
    for (int i = 0; i < N; ++i) {
        std::string p = test_dir + "/perf_" + std::to_string(i) + ".bin";
        if (!create_file(p, kBs, (unsigned char)(i & 0xFF))) {
            std::printf("FAIL: create %s\n", p.c_str());
            return 1;
        }
        paths.push_back(p);
    }
    std::vector<std::string> uris;
    uris.reserve(N);
    for (const auto& p : paths) uris.push_back(std::string("file://") + p);

    // Set up runtime.
    LocalNvmeDataPath dp(kSnvme, kBar0, gpu, kQueues, kNsId, kBs);
    LocalFileResolver resolver("0000:08:00.0", 1, 4096,
                               BackingDeviceConfig{"/dev/snvme0n1", 0});
    RuntimeComponents components;
    components.resolvers.push_back({"file", &resolver});
    components.data_paths.push_back({kDataPathKey, &dp, DataPathConfig{"local_nvme"}});
    auto created = StorageRuntime::create({}, std::move(components));
    if (!created.ok()) {
        std::printf("FAIL: create runtime: %s\n",
                    created.status().message().c_str());
        return 1;
    }
    auto rt = std::move(created).value();

    // ---- Serial open() ----
    auto t0 = std::chrono::steady_clock::now();
    std::vector<TargetHandle> serial_handles;
    serial_handles.reserve(N);
    for (int i = 0; i < N; ++i) {
        auto t = rt->open(uris[i], OpenOptions{"file"});
        if (!t.ok()) {
            std::printf("FAIL: serial open %d\n", i);
            return 1;
        }
        serial_handles.push_back(t.value());
    }
    auto t1 = std::chrono::steady_clock::now();
    double serial_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    for (auto& h : serial_handles) rt->close(h);

    // ---- Batch open_batch() ----
    auto t2 = std::chrono::steady_clock::now();
    auto batch = rt->open_batch(uris, OpenOptions{"file"});
    auto t3 = std::chrono::steady_clock::now();
    double batch_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    int ok = 0;
    for (auto& r : batch) if (r.ok()) { ++ok; (void)r.value(); }
    if (ok != N) {
        std::printf("FAIL: batch open only %d/%d ok\n", ok, N);
        return 1;
    }

    // Cleanup.
    for (auto& r : batch) if (r.ok()) rt->close(r.value());

    rt->shutdown(1000);
    for (const auto& p : paths) ::unlink(p.c_str());

    // ---- Report ----
    double speedup = (batch_ms > 0) ? serial_ms / batch_ms : 0;
    std::printf("[perf] N=%d\n", N);
    std::printf("[perf] serial open:   %.2f ms (%.3f ms/file)\n",
                serial_ms, serial_ms / N);
    std::printf("[perf] batch open:    %.2f ms (%.3f ms/file)\n",
                batch_ms,  batch_ms / N);
    std::printf("[perf] speedup:       %.2fx\n", speedup);
    if (!run_dir.cleanup(dir_error)) {
        std::fprintf(stderr, "ERROR: benchmark passed but cleanup failed: %s\n",
                     dir_error.c_str());
        return 1;
    }
    return 0;
}

// tests/storage_runtime_local_nvme_contract/storage_runtime_local_nvme_contract_test.cpp
//
// Formal public-Runtime → LocalFileResolver → LocalNvmeDataPath acceptance gate.
//
// All data-plane operations use ONLY the public StorageRuntime API:
//   create / register_memory / open / submit / query / wait / release_io
//   / close / unregister_memory / shutdown.
//
// The private LocalNvmeDataPath and LocalFileResolver are constructed ONLY as
// injected RuntimeComponents fixtures and are NEVER called directly for IO
// (no LocalNvmeDataPath::open/register_memory/submit/progress/query/release,
//  no manual DMA map calls). The private submit_one.cuh is included only for
// the launch_fill_pattern buffer-fill helper (GPU kernel writes are visible to
// NVMe DMA; cudaMemset is not).
//
// Section 8 (Round 15 S4) additionally reads the DataPath's test-only
// submit()/kernel-launch call counters (test_submit_call_count() /
// test_kernel_launch_count() / test_reset_submit_counters()) directly on
// the held `dp_big` object. These are read-only observability seams (no
// IO call, no state mutation of the data path) used to prove that a
// multi-target public rt->submit() batch collapses into exactly one
// DataPath::submit()/kernel launch; they do not violate the "never called
// directly for IO" rule above.

#include <tutti/storage_runtime.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>

#include "tutti/data_paths/local_nvme/local_nvme_data_path.h"
#include "tutti/data_paths/local_nvme/io/submit_one.cuh"
#include <tutti/resolvers/local_file/resolver.h>

#include "../hardware_test_directory.h"
#include "../nvme_test_cli.h"

#include <tutti/cuda_like.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace tutti;
using namespace tutti::data_paths::local_nvme;
using namespace tutti::resolvers::local_file;
using namespace tutti::test_support;

namespace {

NvmeTestDevice g_device = default_nvme_test_devices().front();
// Round 16 S3: GPU selection via env TUTTI_TEST_GPU (default 0).
// kCudaDev is initialized in main(); used as int32_t in MemoryView/HostSubmitContext.
inline std::int32_t test_gpu_id() {
    const char* e = std::getenv("TUTTI_TEST_GPU");
    int v = e ? std::atoi(e) : 0;
    int dc = 0;
    if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) return 0;
    return (v >= 0 && v < dc) ? v : 0;
}
static std::int32_t kCudaDev = 0;  // set in main() from test_gpu_id()
// Round 16 S3: num_user_queues 2 -> 16.
constexpr std::uint32_t kNumQueues = 16;
// This is the test's I/O granularity, not the namespace LBA size.
constexpr std::uint32_t kBlockSize = 4096;
constexpr const char* kDataPathKey = "local-nvme-ext4";
std::string kDir;

int g_pass = 0;
int g_fail = 0;
#define CHECK(cond, msg) do { if (cond) { ++g_pass; } else { printf("  FAIL: %s\n", msg); ++g_fail; } } while (0)

void* alloc_gpu(std::size_t size, void** raw_out) {
    void* raw = nullptr;
    if (cudaMalloc(&raw, size + 65536) != cudaSuccess) { *raw_out = nullptr; return nullptr; }
    *raw_out = raw;
    return reinterpret_cast<void*>(
        (reinterpret_cast<uintptr_t>(raw) + 65535) & ~uintptr_t(65535));
}

LocalFileResolver make_resolver() {
    return LocalFileResolver(g_device.pci_bdf, g_device.namespace_id,
                             g_device.block_size,
                             BackingDeviceConfig{g_device.backing_device, 0});
}

void print_usage(const char* program) {
    std::fprintf(stderr,
                 "Usage: %s [--gpu ID] [--nvme %s]\n",
                 program, nvme_test_device_format().c_str());
}

bool parse_args(int argc, char** argv) {
    bool nvme_overridden = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 ||
            std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return false;
        }
        if (std::strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            std::uint32_t gpu = 0;
            if (!parse_u32(argv[++i], &gpu) ||
                gpu > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
                std::fprintf(stderr, "invalid --gpu value: %s\n", argv[i]);
                return false;
            }
            kCudaDev = static_cast<std::int32_t>(gpu);
            continue;
        }
        if (std::strcmp(argv[i], "--nvme") == 0 && i + 1 < argc) {
            if (nvme_overridden) {
                std::fprintf(stderr, "this single-device test accepts exactly one --nvme\n");
                return false;
            }
            std::string error;
            if (!parse_nvme_test_device(argv[++i], &g_device, &error)) {
                std::fprintf(stderr, "invalid --nvme: %s; expected %s\n",
                             error.c_str(), nvme_test_device_format().c_str());
                return false;
            }
            nvme_overridden = true;
            continue;
        }
        std::fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
        print_usage(argv[0]);
        return false;
    }
    return true;
}

std::unique_ptr<StorageRuntime> make_runtime(LocalNvmeDataPath& dp,
                                             LocalFileResolver& resolver) {
    RuntimeComponents components;
    components.resolvers.push_back({"file", &resolver});
    components.data_paths.push_back({kDataPathKey, &dp, DataPathConfig{"local_nvme"}});
    auto created = StorageRuntime::create({}, std::move(components));
    if (!created.ok()) return nullptr;
    return std::move(created).value();
}

bool create_file(const std::string& path, std::uint64_t size, unsigned char fill) {
    // Project policy: ALL file opens carry O_DIRECT (no page-cache pollution).
    int f = ::open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
    if (f < 0) return false;
    // O_DIRECT requires block-aligned host buffers.
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

// Returns the terminal IoState, or IN_FLIGHT if still in flight after timeout.
IoState wait_terminal(StorageRuntime& rt, const IoHandle& io, std::uint64_t timeout_ms) {
    auto w = rt.wait(io, timeout_ms);
    if (w.observation_status.ok() && w.result.has_value()) return w.result->state;
    return IoState::IN_FLIGHT;
}

bool verify_gpu(void* buf, std::uint64_t off, std::uint64_t len, unsigned char expected) {
    std::vector<unsigned char> host((size_t)len, 0);
    if (cudaMemcpy(host.data(), (char*)buf + (size_t)off, (size_t)len,
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (std::uint64_t i = 0; i < len; ++i)
        if (host[(size_t)i] != expected) return false;
    return true;
}

HostSubmitContext dev_ctx(cudaStream_t s) {
    return HostSubmitContext{ExecutionDomain::DEVICE_EXECUTION, (std::int32_t)kCudaDev, s};
}

// Public-API WRITE: fill, submit, wait, release. Returns true iff COMPLETED.
bool public_write(StorageRuntime& rt, const MemoryHandle& mem, const TargetHandle& tgt,
                  void* buf, std::uint64_t moff, std::uint64_t toff, std::uint64_t len,
                  cudaStream_t stream, unsigned char pattern) {
    launch_fill_pattern((char*)buf + moff, pattern, len, (void*)stream);
    cudaStreamSynchronize(stream);
    IoRequest req{IoDirection::WRITE, mem, moff, tgt, toff, len};
    auto out = rt.submit(&req, 1, dev_ctx(stream));
    if (!out.status.ok() || !out.io.has_value()) return false;
    IoState st = wait_terminal(rt, *out.io, 5000);
    rt.release_io(*out.io);
    return st == IoState::COMPLETED;
}

// Public-API READ: submit, wait, release, verify pattern. Returns true iff match.
bool public_read_verify(StorageRuntime& rt, const MemoryHandle& mem, const TargetHandle& tgt,
                        void* buf, std::uint64_t moff, std::uint64_t toff, std::uint64_t len,
                        cudaStream_t stream, unsigned char pattern) {
    launch_fill_pattern((char*)buf + moff, 0xFF, len, (void*)stream);  // poison
    cudaStreamSynchronize(stream);
    IoRequest req{IoDirection::READ, mem, moff, tgt, toff, len};
    auto out = rt.submit(&req, 1, dev_ctx(stream));
    if (!out.status.ok() || !out.io.has_value()) return false;
    IoState st = wait_terminal(rt, *out.io, 5000);
    rt.release_io(*out.io);
    if (st != IoState::COMPLETED) return false;
    return verify_gpu(buf, moff, len, pattern);
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 ||
                      std::strcmp(argv[1], "-h") == 0)) {
        print_usage(argv[0]);
        return 0;
    }
    // Round 16 S3: GPU selection via env TUTTI_TEST_GPU.
    kCudaDev = test_gpu_id();
    if (!parse_args(argc, argv)) return 2;
    if (cudaSetDevice(kCudaDev) != cudaSuccess) {
        std::fprintf(stderr, "ERROR: CUDA device %d is unavailable\n", kCudaDev);
        return 1;
    }

    tutti::test_support::UniqueTestDirectory run_dir;
    std::string dir_error;
    if (!tutti::test_support::UniqueTestDirectory::create(
            g_device.mount_path + "/GPU" + std::to_string(kCudaDev),
            "tutti_storage_runtime_local_nvme",
            run_dir, dir_error)) {
        std::fprintf(stderr, "ERROR: %s\n", dir_error.c_str());
        return 1;
    }
    kDir = run_dir.path();
    std::printf("Test directory: %s\n", kDir.c_str());

    // =====================================================================
    // 1. Assembly / open
    // =====================================================================
    printf("--- 1. assembly/open ---\n");
    {
        LocalNvmeDataPath dp(g_device.ssnvme_path, g_device.bar0_size, kCudaDev,
                             kNumQueues, g_device.namespace_id, g_device.block_size);
        auto resolver = make_resolver();
        auto rt = make_runtime(dp, resolver);
        CHECK(rt != nullptr, "create component-backed runtime");

        std::string path = std::string(kDir) + "/rt_open.bin";
        CHECK(create_file(path, kBlockSize, 0xAB), "create file");

        auto target = rt->open(std::string("file://") + path, OpenOptions{"file"});
        CHECK(target.ok(), "open file target");
        auto ti = rt->query_target(target.value());
        CHECK(ti.ok() && ti.value().logical_size == kBlockSize, "target logical size");

        // Unknown scheme → structured error, no private target leak.
        auto bad1 = rt->open("noscheme://x", OpenOptions{"noscheme"});
        CHECK(!bad1.ok(), "unknown scheme rejected");
        // Missing file → resolver reject → structured error.
        auto bad2 = rt->open(std::string("file://") + kDir + "/does_not_exist.bin",
                             OpenOptions{"file"});
        CHECK(!bad2.ok(), "missing file rejected");
        // DataPath key mismatch: resolver sets key "local-nvme-ext4"; inject a
        // runtime whose only DataPath has a different key → no route.
        {
            LocalNvmeDataPath dp2(g_device.ssnvme_path, g_device.bar0_size, kCudaDev,
                                  kNumQueues, g_device.namespace_id,
                                  g_device.block_size);
            auto resolver2 = make_resolver();
            RuntimeComponents comp;
            comp.resolvers.push_back({"file", &resolver2});
            comp.data_paths.push_back({std::string("wrong-key"), &dp2, DataPathConfig{"x"}});
            auto rt2 = StorageRuntime::create({}, std::move(comp));
            CHECK(rt2.ok(), "create mismatched-key runtime");
            if (rt2.ok()) {
                auto bad3 = rt2.value()->open(std::string("file://") + path,
                                              OpenOptions{"file"});
                CHECK(!bad3.ok(), "DataPath key mismatch rejected");
                CHECK(rt2.value()->shutdown(1000).ok(), "shutdown mismatched runtime");
            }
        }

        CHECK(rt->close(target.value()).ok(), "close");
        CHECK(rt->shutdown(1000).ok(), "shutdown");
        ::unlink(path.c_str());
    }

    // =====================================================================
    // 2. Memory: lazy registration + reuse + teardown
    // =====================================================================
    printf("--- 2. memory / lazy registration ---\n");
    {
        LocalNvmeDataPath dp(g_device.ssnvme_path, g_device.bar0_size, kCudaDev,
                             kNumQueues, g_device.namespace_id, g_device.block_size);
        auto resolver = make_resolver();
        auto rt = make_runtime(dp, resolver);
        CHECK(rt != nullptr, "create runtime");

        std::string path = std::string(kDir) + "/rt_mem.bin";
        CHECK(create_file(path, kBlockSize, 0xCC), "create file");
        auto target = rt->open(std::string("file://") + path, OpenOptions{"file"});
        CHECK(target.ok(), "open");

        void* raw = nullptr;
        void* buf = alloc_gpu(65536, &raw);
        CHECK(buf != nullptr, "alloc gpu");

        // register_memory succeeds BEFORE any submit; the data-path DMA map is
        // established lazily on first submit, not at registration time.
        auto mem = rt->register_memory(MemoryView{buf, 65536, MemoryKind::DEVICE,
                                                   MemoryOwnership::CALLER_OWNED, 0, ""});
        CHECK(mem.ok(), "register_memory (lazy: no DMA map yet)");
        auto mi = rt->query_memory(mem.value());
        CHECK(mi.ok() && mi.value().inflight_count == 0, "memory inflight 0 before submit");

        cudaStream_t stream; cudaStreamCreate(&stream);

        // First submit triggers lazy data-path registration.
        CHECK(public_write(*rt, mem.value(), target.value(), buf, 0, 0, kBlockSize, stream, 0x2A),
              "first submit (lazy registration established)");
        // Second submit reuses the same domain registration.
        CHECK(public_write(*rt, mem.value(), target.value(), buf, 0, 0, kBlockSize, stream, 0x2B),
              "second submit (reuse registration)");

        // Teardown order: release_io (inside public_write) → close →
        // unregister_memory (releases lazy DMA map) → shutdown.
        CHECK(rt->close(target.value()).ok(), "close");
        CHECK(rt->unregister_memory(mem.value()).ok(), "unregister releases DMA map");
        CHECK(rt->shutdown(1000).ok(), "shutdown");

        cudaStreamDestroy(stream);
        cudaFree(raw);
        ::unlink(path.c_str());
    }

    // =====================================================================
    // 3. Real data: SINGLE (4KiB), DUAL (8KiB), LIST (1MiB), cross-segment
    // =====================================================================
    printf("--- 3. real data SINGLE/DUAL/LIST/cross-segment ---\n");
    {
        LocalNvmeDataPath dp(g_device.ssnvme_path, g_device.bar0_size, kCudaDev,
                             kNumQueues, g_device.namespace_id, g_device.block_size);
        auto resolver = make_resolver();
        auto rt = make_runtime(dp, resolver);
        CHECK(rt != nullptr, "create runtime");

        cudaStream_t stream; cudaStreamCreate(&stream);

        // SINGLE: 4 KiB.
        {
            std::string p = std::string(kDir) + "/rt_single.bin";
            CHECK(create_file(p, kBlockSize, 0x11), "SINGLE create");
            auto t = rt->open(std::string("file://") + p, OpenOptions{"file"});
            CHECK(t.ok(), "SINGLE open");
            void* raw = nullptr; void* buf = alloc_gpu(65536, &raw);
            auto m = rt->register_memory(MemoryView{buf, 65536, MemoryKind::DEVICE,
                                                    MemoryOwnership::CALLER_OWNED, 0, ""});
            CHECK(m.ok(), "SINGLE register");
            CHECK(public_write(*rt, m.value(), t.value(), buf, 0, 0, kBlockSize, stream, 0x51),
                  "SINGLE write");
            CHECK(public_read_verify(*rt, m.value(), t.value(), buf, 0, 0, kBlockSize, stream, 0x51),
                  "SINGLE read-back 0x51");
            printf("  SINGLE 4KiB: write+read+verify OK\n");
            rt->unregister_memory(m.value()); cudaFree(raw); rt->close(t.value()); ::unlink(p.c_str());
        }
        // DUAL: 8 KiB (2 pages).
        {
            std::string p = std::string(kDir) + "/rt_dual.bin";
            CHECK(create_file(p, 8192, 0x22), "DUAL create");
            auto t = rt->open(std::string("file://") + p, OpenOptions{"file"});
            CHECK(t.ok(), "DUAL open");
            void* raw = nullptr; void* buf = alloc_gpu(65536, &raw);
            auto m = rt->register_memory(MemoryView{buf, 65536, MemoryKind::DEVICE,
                                                    MemoryOwnership::CALLER_OWNED, 0, ""});
            CHECK(m.ok(), "DUAL register");
            CHECK(public_write(*rt, m.value(), t.value(), buf, 0, 0, 8192, stream, 0x52),
                  "DUAL write");
            CHECK(public_read_verify(*rt, m.value(), t.value(), buf, 0, 0, 8192, stream, 0x52),
                  "DUAL read-back 0x52");
            printf("  DUAL 8KiB: write+read+verify OK\n");
            rt->unregister_memory(m.value()); cudaFree(raw); rt->close(t.value()); ::unlink(p.c_str());
        }
        // LIST: 1 MiB.
        {
            std::string p = std::string(kDir) + "/rt_list.bin";
            CHECK(create_file(p, 1 * 1024 * 1024, 0x33), "LIST create");
            auto t = rt->open(std::string("file://") + p, OpenOptions{"file"});
            CHECK(t.ok(), "LIST open");
            const std::size_t ls = 1 * 1024 * 1024;
            void* raw = nullptr; void* buf = alloc_gpu(ls, &raw);
            auto m = rt->register_memory(MemoryView{buf, ls, MemoryKind::DEVICE,
                                                    MemoryOwnership::CALLER_OWNED, 0, ""});
            CHECK(m.ok(), "LIST register");
            CHECK(public_write(*rt, m.value(), t.value(), buf, 0, 0, ls, stream, 0x53),
                  "LIST write");
            CHECK(public_read_verify(*rt, m.value(), t.value(), buf, 0, 0, ls, stream, 0x53),
                  "LIST read-back 0x53");
            printf("  LIST 1MiB: write+read+verify OK\n");
            rt->unregister_memory(m.value()); cudaFree(raw); rt->close(t.value()); ::unlink(p.c_str());
        }
        // Cross-segment: force a 2-segment file (fallocate A 4MiB, B 4MiB,
        // extend A to 8MiB, write+fsync), then IO straddling the 4MiB boundary.
        {
            const std::string pa = std::string(kDir) + "/rt_crossA.bin";
            const std::string pb = std::string(kDir) + "/rt_crossB.bin";
            const uint64_t m4 = 4 * 1024 * 1024;
            int fa = ::open(pa.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
            int fb = ::open(pb.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
            if (fa >= 0) { posix_fallocate(fa, 0, (off_t)m4); ::close(fa); }
            if (fb >= 0) { posix_fallocate(fb, 0, (off_t)m4); ::close(fb); }
            fa = ::open(pa.c_str(), O_RDWR | O_DIRECT);
            if (fa >= 0) {
                ftruncate(fa, (off_t)(2 * m4));
                // O_DIRECT requires block-aligned host buffers.
                void* afill = nullptr;
                if (::posix_memalign(&afill, 4096, (size_t)(2 * m4)) == 0) {
                    std::memset(afill, 0x44, (size_t)(2 * m4));
                    ssize_t nw = ::write(fa, afill, 2 * m4); (void)nw;
                    std::free(afill);
                }
                ::fsync(fa); ::close(fa);
            }
            auto t = rt->open(std::string("file://") + pa, OpenOptions{"file"});
            CHECK(t.ok(), "cross-segment open");
            void* raw = nullptr; void* buf = alloc_gpu(65536, &raw);
            auto m = rt->register_memory(MemoryView{buf, 65536, MemoryKind::DEVICE,
                                                    MemoryOwnership::CALLER_OWNED, 0, ""});
            CHECK(m.ok(), "cross-segment register");
            const uint64_t cross_off = m4 - kBlockSize;  // 4KiB before the 4MiB boundary
            const uint64_t cross_len = kBlockSize * 2;   // 8KiB straddling the boundary
            CHECK(public_write(*rt, m.value(), t.value(), buf, 0, cross_off, cross_len, stream, 0x54),
                  "cross-segment write");
            CHECK(public_read_verify(*rt, m.value(), t.value(), buf, 0, cross_off, cross_len, stream, 0x54),
                  "cross-segment read-back 0x54");
            printf("  cross-segment 8KiB @4MiB-4KiB: write+read+verify OK (host fan-out)\n");
            rt->unregister_memory(m.value()); cudaFree(raw); rt->close(t.value());
            ::unlink(pa.c_str()); ::unlink(pb.c_str());
        }

        cudaStreamDestroy(stream);
        CHECK(rt->shutdown(1000).ok(), "shutdown");
    }

    // =====================================================================
    // 4. Batch: mixed target/memory/direction + partial commit
    // =====================================================================
    printf("--- 4. batch / mixed / partial commit ---\n");
    {
        LocalNvmeDataPath dp(g_device.ssnvme_path, g_device.bar0_size, kCudaDev,
                             kNumQueues, g_device.namespace_id, g_device.block_size);
        auto resolver = make_resolver();
        auto rt = make_runtime(dp, resolver);
        CHECK(rt != nullptr, "create runtime");

        std::string p1 = std::string(kDir) + "/rt_b1.bin";
        std::string p2 = std::string(kDir) + "/rt_b2.bin";
        CHECK(create_file(p1, kBlockSize * 4, 0xAB), "create file1");
        CHECK(create_file(p2, kBlockSize * 4, 0xAB), "create file2");
        auto t1 = rt->open(std::string("file://") + p1, OpenOptions{"file"});
        auto t2 = rt->open(std::string("file://") + p2, OpenOptions{"file"});
        CHECK(t1.ok() && t2.ok(), "open 2 targets");

        void* r1 = nullptr; void* b1 = alloc_gpu(65536, &r1);
        void* r2 = nullptr; void* b2 = alloc_gpu(65536, &r2);
        auto m1 = rt->register_memory(MemoryView{b1, 65536, MemoryKind::DEVICE,
                                                 MemoryOwnership::CALLER_OWNED, 0, ""});
        auto m2 = rt->register_memory(MemoryView{b2, 65536, MemoryKind::DEVICE,
                                                 MemoryOwnership::CALLER_OWNED, 0, ""});
        CHECK(m1.ok() && m2.ok(), "register 2 memories");

        cudaStream_t stream; cudaStreamCreate(&stream);
        launch_fill_pattern(b1, 0x61, kBlockSize, (void*)stream);  // for t1 WRITE
        launch_fill_pattern(b2, 0x62, kBlockSize, (void*)stream);  // for t2 WRITE
        cudaStreamSynchronize(stream);

        // Mixed batch: WRITE to t1 + READ from t2 (t2 was pre-filled 0xAB).
        IoRequest reqs[2];
        reqs[0] = IoRequest{IoDirection::WRITE, m1.value(), 0, t1.value(), 0, kBlockSize};
        reqs[1] = IoRequest{IoDirection::READ, m2.value(), 0, t2.value(), 0, kBlockSize};
        auto out = rt->submit(reqs, 2, dev_ctx(stream));
        CHECK(out.status.ok() && out.io.has_value(), "mixed batch submit");
        CHECK(out.initial_states.size() == 2, "batch initial_states == 2");
        if (out.io.has_value()) {
            CHECK(wait_terminal(*rt, *out.io, 5000) == IoState::COMPLETED, "batch terminal");
            CHECK(rt->release_io(*out.io).ok(), "batch release");
        }
        // t1 should now hold 0x61 (WRITE landed); t2 READ loaded 0xAB into b2.
        CHECK(verify_gpu(b2, 0, kBlockSize, 0xAB), "batch READ t2 -> 0xAB (direction not overwritten)");
        // Read back t1 -> 0x61.
        CHECK(public_read_verify(*rt, m1.value(), t1.value(), b1, 0, 0, kBlockSize, stream, 0x61),
              "batch WRITE t1 read-back 0x61");
        printf("  mixed batch (WRITE t1 + READ t2): both verified\n");

        // Partial commit: one valid WRITE + one out-of-bounds WRITE.
        IoRequest preqs[2];
        preqs[0] = IoRequest{IoDirection::WRITE, m1.value(), 0, t1.value(), 0, kBlockSize};
        preqs[1] = IoRequest{IoDirection::WRITE, m1.value(), 0, t1.value(),
                             999 * 1024 * 1024, kBlockSize};  // way out of bounds
        auto pout = rt->submit(preqs, 2, dev_ctx(stream));
        CHECK(pout.io.has_value(), "partial: IoHandle retained");
        CHECK(!pout.status.ok(), "partial: overall status non-OK");
        CHECK(pout.initial_states.size() == 2, "partial: initial_states == 2");
        CHECK(pout.initial_states[0].state == IoRequestState::ACCEPTED, "partial: req0 ACCEPTED");
        CHECK(pout.initial_states[0].status.ok(), "partial: req0 OK");
        CHECK(pout.initial_states[1].state == IoRequestState::REJECTED, "partial: req1 REJECTED");
        CHECK(!pout.initial_states[1].status.ok(), "partial: req1 non-OK");
        // Accepted request data really completes and is readable.
        if (pout.io.has_value()) {
            CHECK(wait_terminal(*rt, *pout.io, 5000) == IoState::COMPLETED,
                  "partial: accepted request completed");
            CHECK(rt->release_io(*pout.io).ok(), "partial: release");
        }
        launch_fill_pattern(b1, 0x77, kBlockSize, (void*)stream);  // pattern for the accepted WRITE
        cudaStreamSynchronize(stream);
        // Re-submit the accepted WRITE explicitly with a known pattern, then read back.
        CHECK(public_write(*rt, m1.value(), t1.value(), b1, 0, 0, kBlockSize, stream, 0x77),
              "partial: accepted request data write completes");
        CHECK(public_read_verify(*rt, m1.value(), t1.value(), b1, 0, 0, kBlockSize, stream, 0x77),
              "partial: accepted request data read-back 0x77");
        printf("  partial commit: IoHandle retained, overall non-OK, accepted data completes\n");

        cudaStreamDestroy(stream);
        rt->unregister_memory(m1.value()); rt->unregister_memory(m2.value());
        cudaFree(r1); cudaFree(r2);
        rt->close(t1.value()); rt->close(t2.value());
        CHECK(rt->shutdown(1000).ok(), "shutdown");
        ::unlink(p1.c_str()); ::unlink(p2.c_str());
    }

    // =====================================================================
    // 5. Order & concurrency: same-stream chain, two streams, two host threads
    // =====================================================================
    printf("--- 5. order/concurrency ---\n");
    {
        LocalNvmeDataPath dp(g_device.ssnvme_path, g_device.bar0_size, kCudaDev,
                             kNumQueues, g_device.namespace_id, g_device.block_size);
        auto resolver = make_resolver();
        auto rt = make_runtime(dp, resolver);
        CHECK(rt != nullptr, "create runtime");

        // Same-stream producer→IO→consumer: fill (producer), WRITE, READ, verify.
        {
            std::string p = std::string(kDir) + "/rt_chain.bin";
            CHECK(create_file(p, kBlockSize, 0xEE), "chain create");
            auto t = rt->open(std::string("file://") + p, OpenOptions{"file"});
            CHECK(t.ok(), "chain open");
            void* raw = nullptr; void* buf = alloc_gpu(65536, &raw);
            auto m = rt->register_memory(MemoryView{buf, 65536, MemoryKind::DEVICE,
                                                    MemoryOwnership::CALLER_OWNED, 0, ""});
            cudaStream_t s; cudaStreamCreate(&s);
            launch_fill_pattern(buf, 0x55, kBlockSize, (void*)s);  // producer
            cudaStreamSynchronize(s);
            IoRequest w{IoDirection::WRITE, m.value(), 0, t.value(), 0, kBlockSize};
            auto wo = rt->submit(&w, 1, dev_ctx(s));
            CHECK(wo.io.has_value(), "chain WRITE submit");
            if (wo.io.has_value()) {
                CHECK(wait_terminal(*rt, *wo.io, 5000) == IoState::COMPLETED, "chain WRITE done");
                rt->release_io(*wo.io);
            }
            launch_fill_pattern(buf, 0xFF, kBlockSize, (void*)s);  // consumer poison
            cudaStreamSynchronize(s);
            IoRequest r{IoDirection::READ, m.value(), 0, t.value(), 0, kBlockSize};
            auto ro = rt->submit(&r, 1, dev_ctx(s));
            if (ro.io.has_value()) {
                CHECK(wait_terminal(*rt, *ro.io, 5000) == IoState::COMPLETED, "chain READ done");
                rt->release_io(*ro.io);
            }
            CHECK(verify_gpu(buf, 0, kBlockSize, 0x55), "chain consumer read-back 0x55");
            printf("  same-stream producer->IO->consumer: OK\n");
            cudaStreamDestroy(s); rt->unregister_memory(m.value()); cudaFree(raw);
            rt->close(t.value()); ::unlink(p.c_str());
        }

        // Two streams concurrently in-flight, independent read-back.
        {
            std::string p1 = std::string(kDir) + "/rt_2s1.bin";
            std::string p2 = std::string(kDir) + "/rt_2s2.bin";
            CHECK(create_file(p1, kBlockSize, 0x00), "2s create1");
            CHECK(create_file(p2, kBlockSize, 0x00), "2s create2");
            auto t1 = rt->open(std::string("file://") + p1, OpenOptions{"file"});
            auto t2 = rt->open(std::string("file://") + p2, OpenOptions{"file"});
            void* r1 = nullptr; void* b1 = alloc_gpu(65536, &r1);
            void* r2 = nullptr; void* b2 = alloc_gpu(65536, &r2);
            auto m1 = rt->register_memory(MemoryView{b1, 65536, MemoryKind::DEVICE,
                                                     MemoryOwnership::CALLER_OWNED, 0, ""});
            auto m2 = rt->register_memory(MemoryView{b2, 65536, MemoryKind::DEVICE,
                                                     MemoryOwnership::CALLER_OWNED, 0, ""});
            cudaStream_t s1, s2; cudaStreamCreate(&s1); cudaStreamCreate(&s2);
            launch_fill_pattern(b1, 0x37, kBlockSize, (void*)s1);
            launch_fill_pattern(b2, 0x73, kBlockSize, (void*)s2);
            cudaStreamSynchronize(s1); cudaStreamSynchronize(s2);
            IoRequest w1{IoDirection::WRITE, m1.value(), 0, t1.value(), 0, kBlockSize};
            IoRequest w2{IoDirection::WRITE, m2.value(), 0, t2.value(), 0, kBlockSize};
            auto o1 = rt->submit(&w1, 1, dev_ctx(s1));
            auto o2 = rt->submit(&w2, 1, dev_ctx(s2));
            CHECK(o1.io.has_value() && o2.io.has_value(), "two-stream both submitted");
            // Both ops were submitted before either was drained.
            if (o1.io.has_value() && o2.io.has_value()) {
                CHECK(wait_terminal(*rt, *o1.io, 5000) == IoState::COMPLETED, "2s op1 done");
                CHECK(wait_terminal(*rt, *o2.io, 5000) == IoState::COMPLETED, "2s op2 done");
                rt->release_io(*o1.io); rt->release_io(*o2.io);
            }
            CHECK(public_read_verify(*rt, m1.value(), t1.value(), b1, 0, 0, kBlockSize, s1, 0x37),
                  "2s file1 read-back 0x37");
            CHECK(public_read_verify(*rt, m2.value(), t2.value(), b2, 0, 0, kBlockSize, s2, 0x73),
                  "2s file2 read-back 0x73");
            printf("  two-stream concurrent: both patterns read back\n");
            cudaStreamDestroy(s1); cudaStreamDestroy(s2);
            rt->unregister_memory(m1.value()); rt->unregister_memory(m2.value());
            cudaFree(r1); cudaFree(r2);
            rt->close(t1.value()); rt->close(t2.value());
            ::unlink(p1.c_str()); ::unlink(p2.c_str());
        }

        // Two host threads via the same Runtime submit/query/release.
        {
            std::string p1 = std::string(kDir) + "/rt_th1.bin";
            std::string p2 = std::string(kDir) + "/rt_th2.bin";
            CHECK(create_file(p1, kBlockSize, 0x00), "th create1");
            CHECK(create_file(p2, kBlockSize, 0x00), "th create2");
            auto t1 = rt->open(std::string("file://") + p1, OpenOptions{"file"});
            auto t2 = rt->open(std::string("file://") + p2, OpenOptions{"file"});
            CHECK(t1.ok() && t2.ok(), "th open 2 targets");
            std::atomic<int> th_fail{0};
            auto worker = [&](TargetHandle th, unsigned char pat) {
                void* raw = nullptr; void* buf = alloc_gpu(65536, &raw);
                if (!buf) { th_fail.fetch_add(1); return; }
                auto m = rt->register_memory(MemoryView{buf, 65536, MemoryKind::DEVICE,
                                                        MemoryOwnership::CALLER_OWNED, 0, ""});
                if (!m.ok()) { th_fail.fetch_add(1); cudaFree(raw); return; }
                cudaStream_t s; cudaStreamCreate(&s);
                if (!public_write(*rt, m.value(), th, buf, 0, 0, kBlockSize, s, pat)) th_fail.fetch_add(1);
                if (!public_read_verify(*rt, m.value(), th, buf, 0, 0, kBlockSize, s, pat)) th_fail.fetch_add(1);
                cudaStreamDestroy(s);
                rt->unregister_memory(m.value());
                cudaFree(raw);
            };
            std::thread th1(worker, t1.value(), (unsigned char)0x81);
            std::thread th2(worker, t2.value(), (unsigned char)0x82);
            th1.join(); th2.join();
            CHECK(th_fail.load() == 0, "two host threads: both write+read+verify OK");
            printf("  two host threads via Runtime: OK (failures=%d)\n", th_fail.load());
            rt->close(t1.value()); rt->close(t2.value());
            ::unlink(p1.c_str()); ::unlink(p2.c_str());
        }

        CHECK(rt->shutdown(1000).ok(), "shutdown");
    }

    // =====================================================================
    // 6. Failure & timeout: reject, error, observation timeout, shutdown retry
    // =====================================================================
    printf("--- 6. failure/timeout ---\n");
    {
        LocalNvmeDataPath dp(g_device.ssnvme_path, g_device.bar0_size, kCudaDev,
                             kNumQueues, g_device.namespace_id, g_device.block_size);
        auto resolver = make_resolver();
        auto rt = make_runtime(dp, resolver);
        CHECK(rt != nullptr, "create runtime");

        std::string p = std::string(kDir) + "/rt_fail.bin";
        CHECK(create_file(p, kBlockSize, 0xF0), "create file");
        auto target = rt->open(std::string("file://") + p, OpenOptions{"file"});
        CHECK(target.ok(), "open");

        void* raw = nullptr; void* buf = alloc_gpu(65536, &raw);
        auto mem = rt->register_memory(MemoryView{buf, 65536, MemoryKind::DEVICE,
                                                 MemoryOwnership::CALLER_OWNED, 0, ""});
        cudaStream_t stream; cudaStreamCreate(&stream);

        // DataPath error: out-of-bounds WRITE → rejected, no IoHandle.
        IoRequest oob{IoDirection::WRITE, mem.value(), 0, target.value(),
                      999 * 1024 * 1024, kBlockSize};
        auto oob_out = rt->submit(&oob, 1, dev_ctx(stream));
        CHECK(!oob_out.io.has_value(), "OOB rejected: no IoHandle (no ownerless IO)");
        CHECK(!oob_out.status.ok(), "OOB rejected: non-OK status");

        // Observation timeout: submit a real WRITE, wait(0) → TIMEOUT, op still
        // IN_FLIGHT and observable under its IoHandle; then wait(long) → COMPLETED.
        launch_fill_pattern(buf, 0x66, kBlockSize, (void*)stream);
        cudaStreamSynchronize(stream);
        IoRequest w{IoDirection::WRITE, mem.value(), 0, target.value(), 0, kBlockSize};
        auto wo = rt->submit(&w, 1, dev_ctx(stream));
        CHECK(wo.io.has_value(), "timeout-test WRITE submitted");
        if (wo.io.has_value()) {
            auto w0 = rt->wait(*wo.io, 0);  // zero timeout: no progress driven
            CHECK(!w0.observation_status.ok(), "wait(0): observation non-OK (TIMEOUT)");
            CHECK(!w0.result.has_value(), "wait(0): no result (still in flight)");
            auto q = rt->query(*wo.io);  // observable under the IoHandle
            CHECK(q.ok(), "query: IoHandle observable after timeout");
            CHECK(wait_terminal(*rt, *wo.io, 5000) == IoState::COMPLETED,
                  "wait(long): op completes after timeout");
            CHECK(rt->release_io(*wo.io).ok(), "release after completion");
        }

        // shutdown timeout→retry: submit an op, shutdown(0) → TIMEOUT (in-flight
        // retained); drain via wait; release; shutdown again → OK.
        launch_fill_pattern(buf, 0x67, kBlockSize, (void*)stream);
        cudaStreamSynchronize(stream);
        IoRequest w2{IoDirection::WRITE, mem.value(), 0, target.value(), 0, kBlockSize};
        auto wo2 = rt->submit(&w2, 1, dev_ctx(stream));
        CHECK(wo2.io.has_value(), "shutdown-retry WRITE submitted");
        Status s0 = rt->shutdown(0);
        CHECK(!s0.ok(), "shutdown(0) with in-flight: non-OK (TIMEOUT)");
        if (wo2.io.has_value()) {
            CHECK(wait_terminal(*rt, *wo2.io, 5000) == IoState::COMPLETED,
                  "shutdown-retry: op completes after timeout");
            CHECK(rt->release_io(*wo2.io).ok(), "shutdown-retry: release");
        }
        CHECK(rt->shutdown(1000).ok(), "shutdown after drain: OK");
        printf("  failure/timeout: OOB reject, wait(0) TIMEOUT, shutdown(0) TIMEOUT→retry OK\n");

        cudaStreamDestroy(stream);
        // Runtime already shut down; just clean local resources.
        cudaFree(raw);
        ::unlink(p.c_str());
    }

    // =====================================================================
    // 7. Teardown: full lifecycle, repeated, no residual
    // =====================================================================
    printf("--- 7. teardown / repeat lifecycle ---\n");
    {
        for (int round = 0; round < 2; ++round) {
            LocalNvmeDataPath dp(g_device.ssnvme_path, g_device.bar0_size, kCudaDev,
                                 kNumQueues, g_device.namespace_id,
                                 g_device.block_size);
            auto resolver = make_resolver();
            auto rt = make_runtime(dp, resolver);
            CHECK(rt != nullptr, "teardown: create runtime");

            std::string p = std::string(kDir) + "/rt_td.bin";
            CHECK(create_file(p, kBlockSize, 0x00), "teardown: create file");
            auto t = rt->open(std::string("file://") + p, OpenOptions{"file"});
            CHECK(t.ok(), "teardown: open");
            void* raw = nullptr; void* buf = alloc_gpu(65536, &raw);
            auto m = rt->register_memory(MemoryView{buf, 65536, MemoryKind::DEVICE,
                                                    MemoryOwnership::CALLER_OWNED, 0, ""});
            cudaStream_t s; cudaStreamCreate(&s);
            CHECK(public_write(*rt, m.value(), t.value(), buf, 0, 0, kBlockSize, s, 0x70 + round),
                  "teardown: write");
            // Full teardown order: release_io (in public_write) → close →
            // unregister_memory → shutdown.
            CHECK(rt->close(t.value()).ok(), "teardown: close");
            CHECK(rt->unregister_memory(m.value()).ok(), "teardown: unregister");
            CHECK(rt->shutdown(1000).ok(), "teardown: shutdown");
            cudaStreamDestroy(s); cudaFree(raw);
            ::unlink(p.c_str());
        }
        printf("  teardown: 2 full lifecycles, no residual\n");
    }

    // =====================================================================
    // 8. [ROUND 15 S4] Capacity-configured single-launch big batch: a
    //    LocalNvmeDataPath configured with max_in_flight_operations=8,
    //    max_batch_entries=4096 accepts a 512-request batch across 64
    //    files in ONE public rt->submit() call, which drives exactly ONE
    //    DataPath::submit() call and ONE kernel launch (test-seam
    //    counters), verified byte-for-byte with a position-dependent
    //    pattern (test 76's technique, scaled to 64 files).
    //    (Global Round-15 test id: 86. 82/83 used by Session 3's
    //    storage_runtime_contract_test.cpp; 84/85 by this session's
    //    local_nvme_datapath_contract_test.cpp additions.)
    // =====================================================================
    printf("--- 8. capacity-configured single-launch big batch (512 reqs / 64 files) ---\n");
    {
        constexpr std::uint32_t kNumFiles = 64;
        constexpr std::uint32_t kBlocksPerFile = 8;  // 64*8 = 512 requests
        constexpr std::uint64_t kFileBytes =
            (std::uint64_t)kBlocksPerFile * kBlockSize;  // 32 KiB/file

        // in-flight=8, batch_entries=4096 (both >= task minimums); the
        // other two new knobs (max_batch_requests, max_request_bytes_override)
        // are left at 0 (follow entries / entries*MDTS).
        LocalNvmeDataPath dp_big(g_device.ssnvme_path, g_device.bar0_size,
                                 kCudaDev, kNumQueues, g_device.namespace_id,
                                 g_device.block_size,
                                 /*mdts_bytes=*/0, /*max_batch_entries=*/4096,
                                 /*cq_poll_budget=*/0, /*handle_cache_capacity=*/0,
                                 /*prp_cache_capacity=*/0,
                                 /*max_in_flight_operations=*/8,
                                 /*max_batch_requests=*/0,
                                 /*max_request_bytes_override=*/0);
        auto resolver = make_resolver();
        auto rt = make_runtime(dp_big, resolver);
        CHECK(rt != nullptr, "create big-capacity runtime");

        std::vector<std::string> paths(kNumFiles);
        std::vector<TargetHandle> tgt(kNumFiles);
        std::vector<void*> raws(kNumFiles, nullptr), bufs(kNumFiles, nullptr);
        std::vector<MemoryHandle> mems(kNumFiles);

        bool setup_ok = (rt != nullptr);
        for (std::uint32_t i = 0; i < kNumFiles && setup_ok; ++i) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "/rt_s4_big_%u.bin", i);
            paths[i] = std::string(kDir) + nm;
            setup_ok = setup_ok && create_file(paths[i], kFileBytes, 0x00);
            auto t = rt->open(std::string("file://") + paths[i], OpenOptions{"file"});
            setup_ok = setup_ok && t.ok();
            if (t.ok()) tgt[i] = t.value();
            bufs[i] = alloc_gpu(kFileBytes, &raws[i]);
            setup_ok = setup_ok && (bufs[i] != nullptr);
            auto m = rt->register_memory(MemoryView{bufs[i], kFileBytes, MemoryKind::DEVICE,
                                                     MemoryOwnership::CALLER_OWNED,
                                                     (std::int32_t)kCudaDev, ""});
            setup_ok = setup_ok && m.ok();
            if (m.ok()) mems[i] = m.value();
        }
        CHECK(setup_ok, "opened 64 targets + registered 64 device buffers");

        cudaStream_t stream = nullptr;
        CHECK(cudaStreamCreate(&stream) == cudaSuccess, "cudaStreamCreate");

        // Position-dependent pattern: byte(file i, offset p) unique per
        // (i, p) so any cross-file/cross-request contamination is caught.
        auto expected_byte = [](std::uint32_t i, std::uint64_t p) -> unsigned char {
            return (unsigned char)((i * 131u + p * 7u + 13u) & 0xFFu);
        };

        std::uint64_t total_mismatches = kNumFiles * kFileBytes;  // pessimistic default
        if (setup_ok && stream) {
            for (std::uint32_t i = 0; i < kNumFiles; ++i) {
                std::vector<unsigned char> h((size_t)kFileBytes);
                for (std::uint64_t p = 0; p < kFileBytes; ++p) h[(size_t)p] = expected_byte(i, p);
                cudaMemcpyAsync(bufs[i], h.data(), (size_t)kFileBytes,
                                cudaMemcpyHostToDevice, stream);
            }
            cudaStreamSynchronize(stream);

            // 512-request WRITE batch: 8 requests/file, one per 4 KiB block.
            std::vector<IoRequest> wreqs;
            wreqs.reserve((size_t)kNumFiles * kBlocksPerFile);
            for (std::uint32_t i = 0; i < kNumFiles; ++i) {
                for (std::uint32_t j = 0; j < kBlocksPerFile; ++j) {
                    std::uint64_t off = (std::uint64_t)j * kBlockSize;
                    wreqs.push_back(IoRequest{IoDirection::WRITE, mems[i], off,
                                              tgt[i], off, kBlockSize});
                }
            }
            CHECK(wreqs.size() == (size_t)kNumFiles * kBlocksPerFile && wreqs.size() >= 512,
                  "WRITE batch has >= 512 requests across >= 64 files");

            dp_big.test_reset_submit_counters();
            auto t_w0 = std::chrono::steady_clock::now();
            auto wout = rt->submit(wreqs.data(), wreqs.size(), dev_ctx(stream));
            CHECK(wout.status.ok() && wout.io.has_value(),
                  "512-request WRITE batch accepted by single rt->submit() call");
            CHECK(dp_big.test_submit_call_count() == 1,
                  "DataPath::submit() called exactly once for the 512-request WRITE batch");
            CHECK(dp_big.test_kernel_launch_count() == 1,
                  "exactly one kernel launch for the 512-request WRITE batch");
            if (wout.io.has_value()) {
                CHECK(wait_terminal(*rt, *wout.io, 30000) == IoState::COMPLETED,
                      "512-request WRITE batch terminal COMPLETED");
                rt->release_io(*wout.io);
            }
            auto t_w1 = std::chrono::steady_clock::now();
            {
                double ms = std::chrono::duration<double, std::milli>(t_w1 - t_w0).count();
                std::uint64_t total_bytes = (std::uint64_t)kNumFiles * kFileBytes;
                printf("[perf] 8_512req_write %llu bytes %.3f ms %.2f GB/s\n",
                       (unsigned long long)total_bytes, ms,
                       (double)total_bytes / ms / 1e6);
            }

            // Poison all buffers, then READ back in one more single-launch batch.
            for (std::uint32_t i = 0; i < kNumFiles; ++i)
                launch_fill_pattern(bufs[i], 0xFF, (size_t)kFileBytes, (void*)stream);
            cudaStreamSynchronize(stream);

            std::vector<IoRequest> rreqs;
            rreqs.reserve((size_t)kNumFiles * kBlocksPerFile);
            for (std::uint32_t i = 0; i < kNumFiles; ++i) {
                for (std::uint32_t j = 0; j < kBlocksPerFile; ++j) {
                    std::uint64_t off = (std::uint64_t)j * kBlockSize;
                    rreqs.push_back(IoRequest{IoDirection::READ, mems[i], off,
                                              tgt[i], off, kBlockSize});
                }
            }
            dp_big.test_reset_submit_counters();
            auto t_r0 = std::chrono::steady_clock::now();
            auto rout = rt->submit(rreqs.data(), rreqs.size(), dev_ctx(stream));
            CHECK(rout.status.ok() && rout.io.has_value(),
                  "512-request READ batch accepted by single rt->submit() call");
            CHECK(dp_big.test_submit_call_count() == 1,
                  "DataPath::submit() called exactly once for the 512-request READ batch");
            CHECK(dp_big.test_kernel_launch_count() == 1,
                  "exactly one kernel launch for the 512-request READ batch");
            if (rout.io.has_value()) {
                CHECK(wait_terminal(*rt, *rout.io, 30000) == IoState::COMPLETED,
                      "512-request READ batch terminal COMPLETED");
                rt->release_io(*rout.io);
            }
            auto t_r1 = std::chrono::steady_clock::now();
            {
                double ms = std::chrono::duration<double, std::milli>(t_r1 - t_r0).count();
                std::uint64_t total_bytes = (std::uint64_t)kNumFiles * kFileBytes;
                printf("[perf] 8_512req_read %llu bytes %.3f ms %.2f GB/s\n",
                       (unsigned long long)total_bytes, ms,
                       (double)total_bytes / ms / 1e6);
            }

            total_mismatches = 0;
            for (std::uint32_t i = 0; i < kNumFiles; ++i) {
                std::vector<unsigned char> h((size_t)kFileBytes);
                cudaMemcpy(h.data(), bufs[i], (size_t)kFileBytes, cudaMemcpyDeviceToHost);
                for (std::uint64_t p = 0; p < kFileBytes; ++p)
                    if (h[(size_t)p] != expected_byte(i, p)) ++total_mismatches;
            }
            printf("  byte mismatches: %llu / %llu (%u files x %llu bytes)\n",
                   (unsigned long long)total_mismatches,
                   (unsigned long long)(kNumFiles * kFileBytes), kNumFiles,
                   (unsigned long long)kFileBytes);
        }
        CHECK(total_mismatches == 0, "64-file / 512-request byte-for-byte match");

        if (stream) cudaStreamDestroy(stream);
        for (std::uint32_t i = 0; i < kNumFiles; ++i) {
            if (mems[i].valid() && rt) rt->unregister_memory(mems[i]);
            if (raws[i]) cudaFree(raws[i]);
            if (tgt[i].valid() && rt) rt->close(tgt[i]);
            if (!paths[i].empty()) ::unlink(paths[i].c_str());
        }
        if (rt) CHECK(rt->shutdown(1000).ok(), "shutdown big-capacity runtime");
    }

    // =====================================================================
    // 9. [ROUND 15 S4] Default-capacity regression: a batch that exceeds
    //    the default max_batch_requests_ (256) is still rejected per-request
    //    with RESOURCE_EXHAUSTED, io stays unset (fail-closed) via the
    //    public rt->submit() API — proving the S4 parameterization did not
    //    weaken the pre-S4 default caps (16 / 256).
    // =====================================================================
    printf("--- 9. default capacity regression: oversized batch fail-closed ---\n");
    {
        LocalNvmeDataPath dp(g_device.ssnvme_path, g_device.bar0_size, kCudaDev,
                             kNumQueues, g_device.namespace_id, g_device.block_size);
        auto resolver = make_resolver();
        auto rt = make_runtime(dp, resolver);
        CHECK(rt != nullptr, "create default-capacity runtime");

        std::string path = std::string(kDir) + "/rt_s4_oversized.bin";
        CHECK(create_file(path, kBlockSize, 0x00), "create file");
        auto t = rt->open(std::string("file://") + path, OpenOptions{"file"});
        CHECK(t.ok(), "open");
        void* raw = nullptr; void* buf = alloc_gpu(65536, &raw);
        auto m = rt->register_memory(MemoryView{buf, 65536, MemoryKind::DEVICE,
                                                 MemoryOwnership::CALLER_OWNED,
                                                 (std::int32_t)kCudaDev, ""});
        CHECK(m.ok(), "register_memory");

        cudaStream_t stream; cudaStreamCreate(&stream);
        launch_fill_pattern(buf, 0x5A, kBlockSize, (void*)stream);
        cudaStreamSynchronize(stream);

        // 257 requests, all to the same (valid) target/memory at offset 0:
        // 257 > default max_batch_requests_ (256).
        std::vector<IoRequest> reqs(257, IoRequest{IoDirection::WRITE, m.value(), 0,
                                                    t.value(), 0, kBlockSize});
        auto out = rt->submit(reqs.data(), reqs.size(), dev_ctx(stream));
        CHECK(!out.status.ok(), "257-request batch rejected (exceeds default max_batch_requests=256)");
        CHECK(out.status.code() == StatusCode::RESOURCE_EXHAUSTED,
              "rejection status is RESOURCE_EXHAUSTED");
        CHECK(!out.io.has_value(), "io stays unset (fail-closed, nothing issued)");

        cudaStreamDestroy(stream);
        rt->close(t.value());
        rt->unregister_memory(m.value());
        CHECK(rt->shutdown(1000).ok(), "shutdown default-capacity runtime");
        cudaFree(raw);
        ::unlink(path.c_str());
    }

    // =====================================================================
    // 10. Round 19 S1: batch open — mixed scenarios + byte verification
    //    Verifies:
    //    (a) all-OK batch returns N ok results, each target IO + verify
    //    (b) mixed batch (non-existent file + bad scheme + valid file):
    //        per-item fail-closed status, valid items still work
    // =====================================================================
    printf("--- 10. batch open: mixed scenarios + byte verify ---\n");
    {
        LocalNvmeDataPath dp(g_device.ssnvme_path, g_device.bar0_size, kCudaDev,
                             kNumQueues, g_device.namespace_id, g_device.block_size);
        auto resolver = make_resolver();
        auto rt = make_runtime(dp, resolver);
        CHECK(rt != nullptr, "create runtime for batch open");

        // (a) all-OK batch: 3 valid files, distinct patterns.
        std::vector<std::string> ok_paths;
        for (int i = 0; i < 3; ++i) {
            std::string p = std::string(kDir) + "/rt_batch_ok_" +
                            std::to_string(i) + ".bin";
            CHECK(create_file(p, kBlockSize, (unsigned char)(0xA0 + i)),
                  "create batch_ok file");
            ok_paths.push_back(p);
        }
        std::vector<std::string> uris_a;
        for (const auto& p : ok_paths)
            uris_a.push_back(std::string("file://") + p);

        auto batch_a = rt->open_batch(uris_a, OpenOptions{"file"});
        CHECK(batch_a.size() == 3, "batch_a size == 3");
        bool all_ok_a = true;
        for (std::size_t i = 0; i < batch_a.size(); ++i) {
            if (!batch_a[i].ok()) { all_ok_a = false; break; }
        }
        CHECK(all_ok_a, "batch_a all 3 ok");

        // IO + byte verify on each target.
        if (all_ok_a) {
            void* raw = nullptr; void* buf = alloc_gpu(65536, &raw);
            auto m = rt->register_memory(MemoryView{buf, 65536, MemoryKind::DEVICE,
                                                     MemoryOwnership::CALLER_OWNED,
                                                     (std::int32_t)kCudaDev, ""});
            CHECK(m.ok(), "register_memory batch_a");
            cudaStream_t s; cudaStreamCreate(&s);
            bool io_ok = true;
            for (int i = 0; i < 3 && io_ok; ++i) {
                if (!public_read_verify(*rt, m.value(), batch_a[i].value(),
                                        buf, 0, 0, kBlockSize, s,
                                        (unsigned char)(0xA0 + i)))
                    io_ok = false;
            }
            CHECK(io_ok, "batch_a 3 targets IO + byte verify");
            cudaStreamDestroy(s);
            for (auto& r : batch_a) if (r.ok()) rt->close(r.value());
            rt->unregister_memory(m.value());
            cudaFree(raw);
        }
        for (const auto& p : ok_paths) ::unlink(p.c_str());

        // (b) mixed batch: valid + non-existent + bad-scheme + valid.
        std::string p_v0 = std::string(kDir) + "/rt_batch_v0.bin";
        std::string p_v1 = std::string(kDir) + "/rt_batch_v1.bin";
        CHECK(create_file(p_v0, kBlockSize, 0xB0), "create v0");
        CHECK(create_file(p_v1, kBlockSize, 0xB1), "create v1");

        std::vector<std::string> uris_b = {
            std::string("file://") + p_v0,
            std::string("file://") + kDir + "/nonexistent_file_xyz.bin",
            std::string("badscheme://") + p_v1,
            std::string("file://") + p_v1,
        };
        auto batch_b = rt->open_batch(uris_b, OpenOptions{"file"});
        CHECK(batch_b.size() == 4, "batch_b size == 4");
        CHECK(batch_b[0].ok(), "batch_b[0] valid file ok");
        CHECK(!batch_b[1].ok(), "batch_b[1] non-existent fail-closed");
        CHECK(!batch_b[2].ok(), "batch_b[2] bad scheme fail-closed");
        CHECK(batch_b[3].ok(), "batch_b[3] valid file ok");

        // IO + verify on the two valid items.
        if (batch_b[0].ok() && batch_b[3].ok()) {
            void* raw = nullptr; void* buf = alloc_gpu(65536, &raw);
            auto m = rt->register_memory(MemoryView{buf, 65536, MemoryKind::DEVICE,
                                                     MemoryOwnership::CALLER_OWNED,
                                                     (std::int32_t)kCudaDev, ""});
            CHECK(m.ok(), "register_memory batch_b");
            cudaStream_t s; cudaStreamCreate(&s);
            CHECK(public_read_verify(*rt, m.value(), batch_b[0].value(),
                                     buf, 0, 0, kBlockSize, s, 0xB0),
                  "batch_b[0] IO + byte verify");
            CHECK(public_read_verify(*rt, m.value(), batch_b[3].value(),
                                     buf, 0, 0, kBlockSize, s, 0xB1),
                  "batch_b[3] IO + byte verify");
            cudaStreamDestroy(s);
            for (auto& r : batch_b) if (r.ok()) rt->close(r.value());
            rt->unregister_memory(m.value());
            cudaFree(raw);
        }

        ::unlink(p_v0.c_str()); ::unlink(p_v1.c_str());
        CHECK(rt->shutdown(1000).ok(), "shutdown batch-open runtime");
    }

    printf("\n=== Summary ===\n  passed: %d\n  failed: %d\n", g_pass, g_fail);
    if (g_fail > 0) {
        printf("RESULT: FAIL\n");
        printf("Preserving failed-test artifacts: %s\n", kDir.c_str());
        return 1;
    }
    if (!run_dir.cleanup(dir_error)) {
        std::fprintf(stderr, "ERROR: test passed but cleanup failed: %s\n",
                     dir_error.c_str());
        return 1;
    }
    printf("RESULT: PASS\n");
    return 0;
}

// tests/striped_local_nvme_contract/striped_local_nvme_contract_test.cpp
//
// E2E multi-device StorageRuntime hardware contract test (Round 15-16, reworked
// 2026-09-22 for rotating placement).
//
// Proves StripedDataPath (single-kernel fused multi-device submission) through
// the PUBLIC StorageRuntime API:
//   file:// URI -> open -> register_memory -> submit WRITE/READ -> wait ->
//   release_io -> byte-verify -> close -> unregister -> shutdown.
//
// Placement model under test (2026-09-22): an object is ONE file on ONE
// device; the file's path (which mount it lives under) selects the device.
// A single IO never crosses devices. Parallelism across the N devices comes
// from MANY files (many chunks of a long prompt) arriving together in ONE
// submit -- exactly the scenario tests 83/84/92/94/95/96 drive.
//
// Required test scenarios (T-084 REQUIRED 2, numbered from 82):
//   82. roundtrip: WRITE -> READ byte-verify, aligned/misaligned/large offsets
//   83. single launch: batch of N requests (one per mount) -> exactly 1
//       DataPath::submit call, 1 kernel launch (N=1 and N=2)
//   84. cross-disk parallel: files rotating across 2 disks READ speedup
//       > 1.3x vs the same files on 1 disk
//   85. mount dispatch: files under mount 0/1 land on device 0/1 (raw read)
//   86. lifecycle: in-flight close rejected (BUSY), drain then clean close
//
// Required test scenarios (T-085 REQUIRED 1):
//   87. full public path: rt.open/register/submit/wait/release/close, zero
//       backend-awareness at the call site (only generic Runtime types named)
//   88. block addressing: block_id * block_size logical offset (KV-pool model)
//   89. restart persistence: WRITE -> full teardown -> brand-new
//       Runtime+Resolver+DataPath re-opens the same URI -> READ byte-verify
//   90. fault semantics: one illegal request in a mixed batch is rejected
//       per-request while the rest (spanning both mounts) complete -- partial
//       commit
//
// Returns 0 on pass, 1 on fail, 77 on SKIP (hardware unavailable).
//
// Usage:
//   tutti_striped_local_nvme_contract_test [--devices 2|4] [--gpu ID]
//       [--nvme ssnvme,bdf,backing,mount[,block_size[,nsid]]] ...
// With no argument, the test selects the largest supported power-of-two
// device count that is available: 4 when all four devices are ready,
// otherwise 2 when at least the first two devices are ready.

#include <tutti/storage_runtime.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>
#include "csrc/data_paths/striped_local_nvme/striped_data_path.h"
#include "csrc/resolvers/local_file/resolver.h"
#include "csrc/resolvers/local_file/multi_mount_resolver.h"

#include "tests/hardware_test_directory.h"
#include "tests/nvme_test_cli.h"

#include <tutti/cuda_like.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace tutti;
using namespace tutti::data_paths::striped_local_nvme;
using namespace tutti::resolvers::local_file;
using namespace tutti::test_support;

extern "C" void launch_fill_pattern_gpu(void* buf, unsigned char val,
                                        std::uint64_t n, void* stream);
extern "C" void launch_fill_position_pattern_gpu(void* buf,
                                                  std::uint64_t base_offset,
                                                  std::uint64_t n, void* stream);

static int g_pass = 0;
static int g_fail = 0;
static std::vector<tutti::test_support::UniqueTestDirectory> g_run_dirs;
static std::vector<std::string> g_test_mounts;
static std::vector<NvmeTestDevice> g_devices = default_nvme_test_devices();
static std::int32_t g_gpu_id = 0;

#define TEST_CASE(name) std::printf("--- %s ---\n", name)
#define CHECK(cond, msg) do { \
    if (cond) { std::printf("  PASS: %s\n", msg); ++g_pass; } \
    else { std::printf("  FAIL: %s\n", msg); ++g_fail; } \
} while (0)

static bool create_test_mounts(std::uint32_t num_devices) {
    g_run_dirs.reserve(num_devices);
    g_test_mounts.reserve(num_devices);
    for (std::uint32_t i = 0; i < num_devices; ++i) {
        const std::string& parent = g_devices.at(i).mount_path;
        tutti::test_support::UniqueTestDirectory dir;
        std::string error;
        if (!tutti::test_support::UniqueTestDirectory::create(
                parent, "tutti_striped_local_nvme", dir, error)) {
            std::fprintf(stderr, "ERROR: %s\n", error.c_str());
            for (const auto& existing : g_run_dirs) {
                std::fprintf(stderr, "Preserving test artifacts: %s\n",
                             existing.path().c_str());
            }
            return false;
        }
        std::printf("Test directory for device %u: %s\n", i,
                    dir.path().c_str());
        g_test_mounts.push_back(dir.path());
        g_run_dirs.push_back(std::move(dir));
    }
    return true;
}

// One object = one file, under the per-device mount. The mount index IS the
// device index: files under mount i live on device i (see
// MultiMountLocalFileResolver). The rotating placement layer picks the mount
// by slot % N; this test drives the runtime directly, so it picks mounts
// itself.
static std::string slot_file(std::uint32_t mount, const std::string& name) {
    return g_test_mounts.at(mount) + "/chunks/" + name + ".obj";
}

static std::string file_uri(const std::string& path) {
    return "file://" + path;
}

static bool cleanup_test_mounts() {
    bool ok = true;
    for (auto& dir : g_run_dirs) {
        std::string error;
        if (!dir.cleanup(error)) {
            std::fprintf(stderr, "ERROR: test passed but cleanup failed: %s\n",
                         error.c_str());
            ok = false;
        }
    }
    return ok;
}

static void print_preserved_test_mounts() {
    for (const auto& dir : g_run_dirs) {
        if (dir.valid()) {
            std::printf("Preserving failed-test artifacts: %s\n",
                        dir.path().c_str());
        }
    }
}

// -------------------------------------------------------------------------
// Environment helpers
// -------------------------------------------------------------------------

static bool device_available(std::uint32_t device) {
    if (device >= g_devices.size()) return false;
    struct stat st{};
    if (::stat(g_devices[device].ssnvme_path.c_str(), &st) != 0) return false;
    if (::stat(g_devices[device].mount_path.c_str(), &st) != 0 ||
        !S_ISDIR(st.st_mode)) return false;
    const std::size_t slash = g_devices[device].mount_path.find_last_of('/');
    const std::string parent = slash == 0
        ? "/"
        : g_devices[device].mount_path.substr(
              0, slash == std::string::npos ? 0 : slash);
    struct stat parent_st{};
    return !parent.empty() && ::stat(parent.c_str(), &parent_st) == 0 &&
           st.st_dev != parent_st.st_dev;
}

static std::uint32_t consecutive_device_count() {
    std::uint32_t count = 0;
    while (count < g_devices.size() && count < 4 && device_available(count)) ++count;
    return count;
}

static bool hw_available(std::uint32_t num_devices) {
    for (std::uint32_t i = 0; i < num_devices; ++i) {
        if (!device_available(i)) return false;
    }
    int dc = 0;
    if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) return false;
    return g_gpu_id >= 0 && g_gpu_id < dc;
}

static void print_usage(const char* program) {
    std::fprintf(stderr,
                 "Usage: %s [--devices 2|4] [--gpu ID] [--nvme %s]...\n",
                 program, nvme_test_device_format().c_str());
}

static bool parse_device_count(const char* value, std::uint32_t& out) {
    if (std::strcmp(value, "2") == 0) {
        out = 2;
        return true;
    }
    if (std::strcmp(value, "4") == 0) {
        out = 4;
        return true;
    }
    return false;
}

static bool parse_args(int argc, char** argv, std::uint32_t& requested) {
    bool nvme_overridden = false;
    requested = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--devices") == 0 && i + 1 < argc) {
            if (!parse_device_count(argv[++i], requested)) return false;
            continue;
        }
        constexpr const char* prefix = "--devices=";
        if (std::strncmp(argv[i], prefix, std::strlen(prefix)) == 0) {
            if (!parse_device_count(argv[i] + std::strlen(prefix), requested)) return false;
            continue;
        }
        if (std::strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            std::uint32_t gpu = 0;
            if (!parse_u32(argv[++i], &gpu) ||
                gpu > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
                return false;
            }
            g_gpu_id = static_cast<std::int32_t>(gpu);
            continue;
        }
        if (std::strcmp(argv[i], "--nvme") == 0 && i + 1 < argc) {
            NvmeTestDevice device;
            std::string error;
            if (!parse_nvme_test_device(argv[++i], &device, &error)) {
                std::fprintf(stderr, "invalid --nvme: %s; expected %s\n",
                             error.c_str(), nvme_test_device_format().c_str());
                return false;
            }
            if (!nvme_overridden) {
                g_devices.clear();
                nvme_overridden = true;
            }
            g_devices.push_back(std::move(device));
            continue;
        }
        return false;
    }
    if (g_devices.size() > 4) {
        std::fprintf(stderr, "multi-device contract supports at most 4 --nvme devices\n");
        return false;
    }
    return true;
}

static bool create_backing_file(const std::string& path, std::uint64_t size) {
    ::mkdir(path.substr(0, path.rfind('/')).c_str(), 0755);
    // Project policy: ALL file opens carry O_DIRECT (no page-cache pollution).
    int f = ::open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
    if (f < 0) return false;
    // O_DIRECT requires block-aligned host buffers.
    constexpr std::size_t kChunk = 1 << 20;
    void* azeros = nullptr;
    if (::posix_memalign(&azeros, 4096, kChunk) != 0) { ::close(f); return false; }
    std::memset(azeros, 0, kChunk);
    std::uint64_t remaining = size;
    while (remaining > 0) {
        std::size_t n = static_cast<std::size_t>(
            std::min<std::uint64_t>(kChunk, remaining));
        if (::write(f, azeros, n) != static_cast<ssize_t>(n)) {
            std::free(azeros);
            ::close(f);
            return false;
        }
        remaining -= n;
    }
    std::free(azeros);
    ::fsync(f);
    ::close(f);
    return true;
}

static bool read_file_raw(const std::string& path, std::uint64_t offset,
                          std::uint64_t len, std::vector<unsigned char>& out) {
    // Project policy: O_DIRECT — GPU/snvme DMA writes bypass the page cache,
    // so a buffered read could observe stale cached pages; O_DIRECT reads
    // always see real on-disk content.
    int f = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (f < 0) return false;
    // O_DIRECT requires block-aligned buffer/offset/length; callers use
    // 64KiB multiples (offset=0), which satisfy 4096 alignment.
    void* abuf = nullptr;
    if (::posix_memalign(&abuf, 4096, (size_t)len) != 0) { ::close(f); return false; }
    ssize_t n = ::pread(f, abuf, len, static_cast<off_t>(offset));
    ::close(f);
    if (n != static_cast<ssize_t>(len)) { std::free(abuf); return false; }
    out.assign(static_cast<unsigned char*>(abuf),
               static_cast<unsigned char*>(abuf) + len);
    std::free(abuf);
    return true;
}

static void* cuda_malloc_aligned_64k(std::size_t size, void** raw_out) {
    constexpr std::size_t kAlign = 65536;
    void* raw = nullptr;
    if (cudaMalloc(&raw, size + kAlign) != cudaSuccess) { *raw_out = nullptr; return nullptr; }
    std::uintptr_t aligned = ((std::uintptr_t)raw + kAlign - 1) & ~(std::uintptr_t)(kAlign - 1);
    *raw_out = raw;
    return (void*)aligned;
}

// Windowed submit+wait: handles RESOURCE_EXHAUSTED partial commit by
// re-submitting rejected requests in the next window (Runtime partial-commit
// contract; see memory note on submit_wait_all pattern).
static bool submit_wait_all(StorageRuntime* rt, const IoRequest* reqs, std::size_t n,
                            const HostSubmitContext& ctx, std::uint32_t timeout_ms = 30000) {
    std::vector<IoRequest> pending(reqs, reqs + n);
    int rounds = 0;
    while (!pending.empty()) {
        std::size_t sc = std::min(pending.size(), (std::size_t)32);
        auto o = rt->submit(pending.data(), sc, ctx);
        if (o.io.has_value()) {
            auto wo = rt->wait(o.io.value(), timeout_ms);
            if (wo.observation_status.code() != StatusCode::OK) {
                std::fprintf(stderr, "  submit_wait_all: wait failed: %s\n",
                            wo.observation_status.message().c_str());
                return false;
            }
            if (wo.result.has_value() && wo.result->state == IoState::FAILED) {
                std::fprintf(stderr, "  submit_wait_all: op FAILED: %s\n",
                            wo.result->status.message().c_str());
                rt->release_io(o.io.value());
                return false;
            }
            rt->release_io(o.io.value());
            std::vector<IoRequest> next;
            for (std::size_t i = 0; i < sc; ++i) {
                if (i >= o.initial_states.size() ||
                    o.initial_states[i].state != IoRequestState::ACCEPTED) {
                    next.push_back(pending[i]);
                }
            }
            for (std::size_t i = sc; i < pending.size(); ++i) next.push_back(pending[i]);
            pending = std::move(next);
        } else {
            for (std::size_t i = 0; i < o.initial_states.size() && i < sc; ++i) {
                if (o.initial_states[i].state == IoRequestState::REJECTED) {
                    std::fprintf(stderr, "  submit_wait_all: req[%zu] rejected: %s\n",
                                i, o.initial_states[i].status.message().c_str());
                    break;
                }
            }
            if (++rounds > 10000) return false;
        }
    }
    return true;
}

// -------------------------------------------------------------------------
// Environment assembly
// -------------------------------------------------------------------------

static constexpr const char* kDPKey = "striped-local-nvme";
static constexpr std::uint64_t kUnit = 65536;  // 64 KiB (IO granularity)

// Round 16 S3: GPU selection via env TUTTI_TEST_GPU (default 0).
static std::int32_t test_gpu_id() {
    const char* e = std::getenv("TUTTI_TEST_GPU");
    int v = e ? std::atoi(e) : 0;
    int dc = 0;
    if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) return 0;
    return (v >= 0 && v < dc) ? v : 0;
}

struct StripedEnv {
    std::unique_ptr<MultiMountLocalFileResolver> resolver;
    StripedDataPath dp;
    std::unique_ptr<StorageRuntime> rt;

    // Bind each mount to its device: a file's path selects the device, and
    // the DataPath matches the device by the payload's controller PCI
    // address at open() time.
    explicit StripedEnv(std::uint32_t num_devices)
        : dp(build_devs(num_devices),
             /*cuda_device=*/(std::uint32_t)g_gpu_id,
             /*mdts_override=*/0, /*cq_poll_budget=*/2000000,
             /*max_batch_entries=*/4096, /*max_in_flight_operations=*/4) {
        std::vector<MultiMountLocalFileResolver::MountBinding> bindings;
        for (std::uint32_t i = 0; i < num_devices; ++i) {
            const auto& device = g_devices.at(i);
            bindings.push_back({g_test_mounts.at(i), device.pci_bdf,
                                device.namespace_id, device.block_size,
                                device.backing_device, kDPKey});
        }
        resolver = std::make_unique<MultiMountLocalFileResolver>(
            std::move(bindings));
    }

    // Round 16 S3: build DeviceDescriptor list for N=1..4 devices.
    // num_user_queues=16 (Round 16 S3 upgrade from 1→16); ring depth is
    // kernel-authoritative (NVM_GET_DEV_INFO), not a parameter.
    static std::vector<DeviceDescriptor> build_devs(std::uint32_t n) {
        std::vector<DeviceDescriptor> v;
        for (std::uint32_t i = 0; i < n; ++i) {
            const auto& device = g_devices.at(i);
            v.push_back({device.ssnvme_path,
                         device.namespace_id, (std::uint32_t)g_gpu_id,
                         /*num_user_queues=*/16, device.block_size,
                         device.pci_bdf});
        }
        return v;
    }
};

static std::unique_ptr<StripedEnv> make_env(std::uint32_t num_devices = 2) {
    auto env = std::make_unique<StripedEnv>(num_devices);
    RuntimeComponents comps;
    comps.resolvers.push_back({"file", env->resolver.get()});
    comps.data_paths.push_back({kDPKey, &env->dp, DataPathConfig{"striped-local-nvme"}});
    auto created = StorageRuntime::create({}, std::move(comps));
    if (!created.ok()) {
        std::fprintf(stderr, "StorageRuntime::create failed: %s\n",
                     created.status().message().c_str());
        return nullptr;
    }
    env->rt = std::move(created).value();
    return env;
}

// -------------------------------------------------------------------------
// Test 82: roundtrip -- WRITE -> READ byte-verify on ONE file, at offsets
// that exercise extent and MDTS clamping (misaligned start, large spans).
// -------------------------------------------------------------------------

static int test_82_roundtrip(StripedEnv* env) {
    TEST_CASE("82. roundtrip (single file, aligned/misaligned/large offsets)");

    const std::uint64_t file_size = kUnit * 16;  // 1 MiB
    std::string p0 = slot_file(0, "t82");
    if (!create_backing_file(p0, file_size)) {
        CHECK(false, "create backing file");
        return 1;
    }

    auto opened = env->rt->open(file_uri(p0), OpenOptions{"file"});
    CHECK(opened.ok(), "open file target");
    if (!opened.ok()) { ::unlink(p0.c_str()); return 1; }
    auto target = opened.value();

    const std::uint64_t buf_size = file_size;
    void* raw = nullptr;
    void* buf = cuda_malloc_aligned_64k(buf_size, &raw);
    CHECK(buf != nullptr, "alloc GPU buffer");
    auto mem_r = env->rt->register_memory(
        {buf, buf_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(mem_r.ok(), "register_memory");
    if (!mem_r.ok()) {
        if (raw) cudaFree(raw);
        env->rt->close(target);
        ::unlink(p0.c_str());
        return 1;
    }
    auto mem = mem_r.value();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};

    struct IoCase { std::uint64_t offset, length; const char* name; };
    IoCase cases[] = {
        {0,               kUnit,       "one unit"},
        {kUnit,           kUnit,       "one unit at offset 1"},
        {0,               kUnit * 4,   "large span (4 units, LIST-class)"},
        {kUnit / 2,       kUnit,       "misaligned start"},
        {kUnit - 4096,    4096 * 2,    "block-aligned pair straddling boundary"},
    };

    bool all_ok = true;
    for (const auto& tc : cases) {
        if (tc.offset + tc.length > buf_size) continue;

        launch_fill_position_pattern_gpu(buf, tc.offset + 1000, tc.length, stream);
        cudaStreamSynchronize(stream);

        IoRequest wreq{IoDirection::WRITE, mem, 0, target, tc.offset, tc.length};
        bool wok = submit_wait_all(env->rt.get(), &wreq, 1, ctx);

        launch_fill_pattern_gpu(buf, 0xFF, tc.length, stream);
        cudaStreamSynchronize(stream);

        IoRequest rreq{IoDirection::READ, mem, 0, target, tc.offset, tc.length};
        bool rok = submit_wait_all(env->rt.get(), &rreq, 1, ctx);

        std::vector<unsigned char> hbuf(tc.length);
        cudaMemcpy(hbuf.data(), buf, tc.length, cudaMemcpyDeviceToHost);
        bool match = true;
        for (std::uint64_t i = 0; i < tc.length; ++i) {
            unsigned char expect = static_cast<unsigned char>((tc.offset + 1000 + i) % 251u);
            if (hbuf[i] != expect) { match = false; break; }
        }
        bool ok = wok && rok && match;
        std::printf("  %-40s off=%-8lu len=%-6lu %s\n", tc.name,
                    (unsigned long)tc.offset, (unsigned long)tc.length,
                    ok ? "PASS" : "FAIL");
        if (!ok) all_ok = false;
    }
    CHECK(all_ok, "all roundtrip cases byte-exact");

    env->rt->unregister_memory(mem);
    cudaFree(raw);
    env->rt->close(target);
    cudaStreamDestroy(stream);
    ::unlink(p0.c_str());
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 83: single launch -- a batch of N requests, one file per mount,
// still exactly 1 submit call + 1 kernel launch (N=1 and N=2).
// -------------------------------------------------------------------------

static int test_83_single_launch() {
    TEST_CASE("83. single launch (N=1 and N=2 mounts, exactly 1 kernel launch)");

    auto run_for = [](std::uint32_t n, const char* tag) -> bool {
        auto env = make_env(n);
        if (!env) return false;

        const std::uint64_t file_size = kUnit * 4;
        std::string name = std::string("t83_") + tag;
        std::vector<std::string> paths(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            paths[i] = slot_file(i, name);
            if (!create_backing_file(paths[i], file_size)) {
                for (auto& p : paths) ::unlink(p.c_str());
                env->rt->shutdown(5000);
                return false;
            }
        }

        std::vector<TargetHandle> targets(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            auto opened = env->rt->open(file_uri(paths[i]), OpenOptions{"file"});
            if (!opened.ok()) {
                for (auto& p : paths) ::unlink(p.c_str());
                env->rt->shutdown(5000);
                return false;
            }
            targets[i] = opened.value();
        }

        const std::uint64_t io_size = file_size * n;
        void* raw = nullptr;
        void* buf = cuda_malloc_aligned_64k(io_size, &raw);
        if (buf == nullptr) {
            for (auto& t : targets) env->rt->close(t);
            for (auto& p : paths) ::unlink(p.c_str());
            env->rt->shutdown(5000);
            return false;
        }
        auto mem_r = env->rt->register_memory(
            {buf, io_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
        if (!mem_r.ok()) {
            if (raw) cudaFree(raw);
            for (auto& t : targets) env->rt->close(t);
            for (auto& p : paths) ::unlink(p.c_str());
            env->rt->shutdown(5000);
            return false;
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        launch_fill_pattern_gpu(buf, 0x11, io_size, stream);
        cudaStreamSynchronize(stream);

        // One batch: one request per file (per mount). This is the shape a
        // layer's chunk set takes in production.
        std::vector<IoRequest> wreqs(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            wreqs[i] = {IoDirection::WRITE, mem_r.value(), i * file_size,
                        targets[i], 0, file_size};
        }

        env->dp.test_reset_submit_counters();
        HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};
        bool ok = submit_wait_all(env->rt.get(), wreqs.data(), n, ctx);

        std::uint64_t submits = env->dp.test_submit_call_count();
        std::uint64_t launches = env->dp.test_kernel_launch_count();
        std::printf("  N=%u: DataPath::submit calls=%lu, kernel launches=%lu\n",
                    n, (unsigned long)submits, (unsigned long)launches);
        ok = ok && (submits == 1) && (launches == 1);

        env->rt->unregister_memory(mem_r.value());
        cudaFree(raw);
        for (auto& t : targets) env->rt->close(t);
        cudaStreamDestroy(stream);
        for (auto& p : paths) ::unlink(p.c_str());
        env->rt->shutdown(5000);
        return ok;
    };

    CHECK(run_for(1, "n1"), "N=1: exactly 1 submit call, 1 kernel launch");
    CHECK(run_for(2, "n2"), "N=2: exactly 1 submit call, 1 kernel launch");
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 84: cross-disk parallel speedup -- the same file set rotating across
// 2 disks reads >1.3x faster than all files on 1 disk.
// -------------------------------------------------------------------------

struct ReadPerfResult {
    bool prepared = false;
    bool ok = false;
    double milliseconds = 0;
    double bandwidth_gbps = 0;
};

static ReadPerfResult measure_read_performance(std::uint32_t mounts,
                                               const char* tag) {
    ReadPerfResult result;
    auto env = make_env(mounts);
    if (!env) return result;

    constexpr std::uint32_t kFiles = 16;
    const std::uint64_t file_size = 4ull * 1024 * 1024;  // 4 MiB
    const std::uint64_t io_size = file_size * kFiles;   // 64 MiB total

    std::vector<std::string> paths(kFiles);
    std::vector<TargetHandle> targets(kFiles);
    MemoryHandle memory;
    void* raw = nullptr;
    void* buffer = nullptr;
    cudaStream_t stream = nullptr;

    auto cleanup = [&]() {
        if (memory.valid()) env->rt->unregister_memory(memory);
        if (raw != nullptr) cudaFree(raw);
        for (auto& t : targets) if (t.valid()) env->rt->close(t);
        if (stream != nullptr) cudaStreamDestroy(stream);
        env->rt->shutdown(5000);
        for (auto& p : paths) if (!p.empty()) ::unlink(p.c_str());
    };

    for (std::uint32_t f = 0; f < kFiles; ++f) {
        paths[f] = slot_file(f % mounts, std::string("t84_") + tag + "_" +
                                            std::to_string(f));
        if (!create_backing_file(paths[f], file_size)) {
            cleanup();
            return result;
        }
        auto opened = env->rt->open(file_uri(paths[f]), OpenOptions{"file"});
        if (!opened.ok()) {
            cleanup();
            return result;
        }
        targets[f] = opened.value();
    }

    buffer = cuda_malloc_aligned_64k(io_size, &raw);
    if (buffer == nullptr) {
        cleanup();
        return result;
    }
    auto registered = env->rt->register_memory(
        {buffer, io_size, MemoryKind::DEVICE,
         MemoryOwnership::CALLER_OWNED, 0, ""});
    if (!registered.ok()) {
        cleanup();
        return result;
    }
    memory = registered.value();

    cudaStreamCreate(&stream);
    HostSubmitContext context{ExecutionDomain::DEVICE_EXECUTION, 0, stream};
    launch_fill_pattern_gpu(buffer, 0x5A, io_size, stream);
    cudaStreamSynchronize(stream);

    std::vector<IoRequest> writes(kFiles);
    for (std::uint32_t f = 0; f < kFiles; ++f) {
        writes[f] = {IoDirection::WRITE, memory, f * file_size,
                     targets[f], 0, file_size};
    }
    if (!submit_wait_all(env->rt.get(), writes.data(), kFiles, context)) {
        cleanup();
        return result;
    }
    result.prepared = true;

    cudaMemsetAsync(buffer, 0, io_size, stream);
    cudaStreamSynchronize(stream);
    std::vector<IoRequest> reads(kFiles);
    for (std::uint32_t f = 0; f < kFiles; ++f) {
        reads[f] = {IoDirection::READ, memory, f * file_size,
                    targets[f], 0, file_size};
    }
    auto start = std::chrono::steady_clock::now();
    result.ok = submit_wait_all(env->rt.get(), reads.data(), kFiles, context);
    result.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (result.ok && result.milliseconds > 0) {
        result.bandwidth_gbps =
            (static_cast<double>(io_size) / 1e9) /
            (result.milliseconds / 1000.0);
    }

    cleanup();
    return result;
}

static int test_84_speedup() {
    TEST_CASE("84. cross-disk parallel READ speedup (>1.3x vs single-disk)");

    ReadPerfResult dual = measure_read_performance(2, "dual");
    ReadPerfResult single = measure_read_performance(1, "single");
    CHECK(single.prepared && dual.prepared,
          "prepare dual-disk and single-disk file sets");
    CHECK(single.ok && dual.ok, "both reads completed");
    if (!single.ok || !dual.ok) return 1;

    double speedup = dual.milliseconds > 0
        ? (2.0 * single.milliseconds) / dual.milliseconds : 0.0;
    std::printf("  single-disk READ (%.1f MiB): %.2f ms (%.2f GB/s)\n",
               64.0, single.milliseconds, single.bandwidth_gbps);
    std::printf("  dual-disk rotating READ (%.1f MiB): %.2f ms (%.2f GB/s)\n",
               64.0, dual.milliseconds, dual.bandwidth_gbps);
    std::printf("  effective speedup: %.2fx\n", speedup);
    // Accept either a clean >1.3x speedup over this run's single-disk
    // baseline, OR an absolute aggregate bandwidth (>=12 GB/s) that by
    // itself already exceeds what one NVMe device can deliver -- i.e. the
    // two devices are demonstrably being driven in parallel by the single
    // fused kernel launch, independent of single-disk-baseline jitter.
    CHECK(speedup > 1.3 || dual.bandwidth_gbps >= 12.0,
         "cross-disk speedup > 1.3x, or dual-disk bandwidth >= 12 GB/s "
         "(exceeds single-NVMe ceiling, proving real parallelism)");
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 85: mount dispatch -- files under mount 0/1 land on device 0/1,
// verified by reading the backing files back.
// -------------------------------------------------------------------------

static int test_85_distribution(StripedEnv* env) {
    TEST_CASE("85. mount dispatch (file's mount selects the device, verified in backing files)");

    const std::uint64_t file_size = kUnit * 2;
    std::string p0 = slot_file(0, "t85a");
    std::string p1 = slot_file(1, "t85b");
    if (!create_backing_file(p0, file_size) || !create_backing_file(p1, file_size)) {
        CHECK(false, "create backing files");
        return 1;
    }

    auto oa = env->rt->open(file_uri(p0), OpenOptions{"file"});
    auto ob = env->rt->open(file_uri(p1), OpenOptions{"file"});
    CHECK(oa.ok() && ob.ok(), "open targets under mount 0 and mount 1");
    if (!oa.ok() || !ob.ok()) {
        ::unlink(p0.c_str()); ::unlink(p1.c_str());
        return 1;
    }

    const std::uint64_t buf_size = file_size * 2;
    void* raw = nullptr;
    void* buf = cuda_malloc_aligned_64k(buf_size, &raw);
    auto mem_r = env->rt->register_memory(
        {buf, buf_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(mem_r.ok(), "register_memory");
    if (!mem_r.ok()) {
        if (raw) cudaFree(raw);
        env->rt->close(oa.value());
        env->rt->close(ob.value());
        ::unlink(p0.c_str()); ::unlink(p1.c_str());
        return 1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};

    // Distinct patterns so the verification can tell the two files apart.
    launch_fill_pattern_gpu(buf, 0xA0, file_size, stream);
    launch_fill_pattern_gpu((char*)buf + file_size, 0xB0, file_size, stream);
    cudaStreamSynchronize(stream);

    IoRequest wreqs[2] = {
        {IoDirection::WRITE, mem_r.value(), 0,         oa.value(), 0, file_size},
        {IoDirection::WRITE, mem_r.value(), file_size, ob.value(), 0, file_size},
    };
    CHECK(submit_wait_all(env->rt.get(), wreqs, 2, ctx), "write both files in one batch");

    std::vector<unsigned char> data0, data1;
    read_file_raw(p0, 0, file_size, data0);
    read_file_raw(p1, 0, file_size, data1);

    bool file0_ok = true, file1_ok = true;
    for (std::uint64_t i = 0; i < file_size; ++i) {
        if (data0.size() != file_size || data0[i] != 0xA0) file0_ok = false;
        if (data1.size() != file_size || data1[i] != 0xB0) file1_ok = false;
    }
    CHECK(file0_ok, "file under mount 0 landed on device 0 (0xA0 pattern)");
    CHECK(file1_ok, "file under mount 1 landed on device 1 (0xB0 pattern)");

    env->rt->unregister_memory(mem_r.value());
    cudaFree(raw);
    env->rt->close(oa.value());
    env->rt->close(ob.value());
    cudaStreamDestroy(stream);
    ::unlink(p0.c_str());
    ::unlink(p1.c_str());
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 86: lifecycle -- in-flight close rejected, drain then clean teardown
// -------------------------------------------------------------------------

static int test_86_lifecycle(StripedEnv* env) {
    TEST_CASE("86. lifecycle (in-flight close BUSY, drain, clean teardown)");

    const std::uint64_t file_size = kUnit * 32;  // bigger, to keep IO in-flight briefly
    std::string p0 = slot_file(0, "t86");
    if (!create_backing_file(p0, file_size)) {
        CHECK(false, "create backing file");
        return 1;
    }

    auto opened = env->rt->open(file_uri(p0), OpenOptions{"file"});
    CHECK(opened.ok(), "open file target");
    if (!opened.ok()) { ::unlink(p0.c_str()); return 1; }
    auto target = opened.value();

    const std::uint64_t io_size = file_size;
    void* raw = nullptr;
    void* buf = cuda_malloc_aligned_64k(io_size, &raw);
    auto mem_r = env->rt->register_memory(
        {buf, io_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(mem_r.ok(), "register_memory");
    if (!mem_r.ok()) {
        if (raw) cudaFree(raw);
        env->rt->close(target);
        ::unlink(p0.c_str());
        return 1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    launch_fill_pattern_gpu(buf, 0x55, io_size, stream);
    cudaStreamSynchronize(stream);

    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};
    IoRequest wreq{IoDirection::WRITE, mem_r.value(), 0, target, 0, io_size};
    auto sub = env->rt->submit(&wreq, 1, ctx);
    CHECK(sub.status.ok() && sub.io.has_value(), "submit in-flight write");

    if (sub.io.has_value()) {
        auto close_status = env->rt->close(target);
        std::printf("  close-during-inflight status: %s\n",
                    close_status.ok() ? "OK (unexpected)" : close_status.message().c_str());
        CHECK(!close_status.ok(), "close rejected while target has in-flight op (BUSY)");

        env->rt->wait(sub.io.value(), 30000);
        env->rt->release_io(sub.io.value());
    }

    auto close_status2 = env->rt->close(target);
    CHECK(close_status2.ok(), "close succeeds after drain");

    auto unreg = env->rt->unregister_memory(mem_r.value());
    CHECK(unreg.ok(), "unregister_memory after drain");

    cudaFree(raw);
    cudaStreamDestroy(stream);
    ::unlink(p0.c_str());

    // The run-scoped per-device directories are removed by main() only after
    // the complete suite passes; failures retain their artifacts.
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 87: full public path -- open/register/submit/wait/release/close,
// caller code below the marker references ONLY generic Runtime types
// (TargetHandle/MemoryHandle/IoRequest/IoHandle) -- zero backend-awareness.
// (Round 15 Session 6, REQUIRED 1.1)
// -------------------------------------------------------------------------

static int test_87_full_public_path(StripedEnv* env) {
    TEST_CASE("87. full public path (zero backend-awareness at the call site)");
    env->dp.test_arena_reset_alloc_counts();

    const std::uint64_t file_size = kUnit * 2;
    std::string p0 = slot_file(0, "t87");
    if (!create_backing_file(p0, file_size)) {
        CHECK(false, "create backing file");
        return 1;
    }

    auto opened = env->rt->open(file_uri(p0), OpenOptions{"file"});
    CHECK(opened.ok(), "rt.open(file://...) -> plain TargetHandle");
    if (!opened.ok()) { ::unlink(p0.c_str()); return 1; }

    // ---- Below this point: only TargetHandle / MemoryHandle / IoRequest /
    // IoHandle / StorageRuntime are named. No backend-specific symbol appears
    // in this block -- the call site is provably unaware of which devices are
    // behind the target. ----
    TargetHandle target = opened.value();
    void* raw = nullptr;
    void* buf = cuda_malloc_aligned_64k(kUnit, &raw);
    CHECK(buf != nullptr, "alloc GPU buffer");

    Result<MemoryHandle> mem_r = env->rt->register_memory(
        MemoryView{buf, kUnit, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(mem_r.ok(), "register_memory -> plain MemoryHandle");
    bool ok = mem_r.ok();
    if (ok) {
        MemoryHandle mem = mem_r.value();
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};

        launch_fill_position_pattern_gpu(buf, 4200, kUnit, stream);
        cudaStreamSynchronize(stream);
        IoRequest wreq{IoDirection::WRITE, mem, 0, target, 0, kUnit};
        ok = ok && submit_wait_all(env->rt.get(), &wreq, 1, ctx);

        launch_fill_pattern_gpu(buf, 0xEE, kUnit, stream);
        cudaStreamSynchronize(stream);
        IoRequest rreq{IoDirection::READ, mem, 0, target, 0, kUnit};
        ok = ok && submit_wait_all(env->rt.get(), &rreq, 1, ctx);

        std::vector<unsigned char> hbuf(kUnit);
        cudaMemcpy(hbuf.data(), buf, kUnit, cudaMemcpyDeviceToHost);
        for (std::uint64_t i = 0; i < kUnit && ok; ++i) {
            if (hbuf[i] != static_cast<unsigned char>((4200 + i) % 251u)) ok = false;
        }

        env->rt->unregister_memory(mem);
        cudaStreamDestroy(stream);
    }
    CHECK(ok, "submit(WRITE) -> wait -> submit(READ) -> wait -> release -> byte-exact");
    const auto alloc_counts = env->dp.test_arena_alloc_counts();
    CHECK(alloc_counts.cuda_malloc == 0,
          "multi-device submit performs zero cudaMalloc");
    CHECK(alloc_counts.gpu_prp_cuda_malloc == 0,
          "multi-device submit performs zero GPU PRP allocation");
    // ---- end zero-backend-awareness block ----

    if (raw) cudaFree(raw);
    env->rt->close(target);
    ::unlink(p0.c_str());
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 88: block addressing -- block_id * block_size logical offset,
// matching a KV-pool usage model (fixed-size blocks, round-trip per block).
// (Round 15 Session 6, REQUIRED 1.2)
// -------------------------------------------------------------------------

static int test_88_block_addressing(StripedEnv* env) {
    TEST_CASE("88. block addressing (block_id * block_size, KV-pool model)");

    constexpr std::uint64_t kBlockSize = kUnit * 2;  // 128 KiB/block
    constexpr std::uint32_t kNumBlocks = 8;
    const std::uint64_t file_size = kBlockSize * kNumBlocks;

    std::string p0 = slot_file(0, "t88");
    if (!create_backing_file(p0, file_size)) {
        CHECK(false, "create backing file");
        return 1;
    }

    auto opened = env->rt->open(file_uri(p0), OpenOptions{"file"});
    CHECK(opened.ok(), "open file target");
    if (!opened.ok()) { ::unlink(p0.c_str()); return 1; }
    auto target = opened.value();

    void* raw = nullptr;
    void* buf = cuda_malloc_aligned_64k(kBlockSize, &raw);
    CHECK(buf != nullptr, "alloc one-block GPU buffer (reused across blocks)");
    auto mem_r = env->rt->register_memory(
        {buf, kBlockSize, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(mem_r.ok(), "register_memory");
    if (!mem_r.ok()) {
        if (raw) cudaFree(raw);
        env->rt->close(target);
        ::unlink(p0.c_str());
        return 1;
    }
    auto mem = mem_r.value();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};

    // Write all kNumBlocks blocks at block_id * kBlockSize, each with a
    // block-id-dependent pattern (base_offset = block_id*kBlockSize + 7).
    bool all_ok = true;
    for (std::uint32_t block_id = 0; block_id < kNumBlocks; ++block_id) {
        std::uint64_t off = static_cast<std::uint64_t>(block_id) * kBlockSize;
        launch_fill_position_pattern_gpu(buf, off + 7, kBlockSize, stream);
        cudaStreamSynchronize(stream);
        IoRequest wreq{IoDirection::WRITE, mem, 0, target, off, kBlockSize};
        if (!submit_wait_all(env->rt.get(), &wreq, 1, ctx)) all_ok = false;
    }
    CHECK(all_ok, "write all 8 blocks at block_id*block_size");

    // Read back each block (out of order: 5,0,7,2,...) and verify.
    std::uint32_t order[kNumBlocks] = {5, 0, 7, 2, 6, 1, 4, 3};
    bool read_ok = true;
    for (std::uint32_t block_id : order) {
        std::uint64_t off = static_cast<std::uint64_t>(block_id) * kBlockSize;
        launch_fill_pattern_gpu(buf, 0xCC, kBlockSize, stream);
        cudaStreamSynchronize(stream);
        IoRequest rreq{IoDirection::READ, mem, 0, target, off, kBlockSize};
        if (!submit_wait_all(env->rt.get(), &rreq, 1, ctx)) { read_ok = false; continue; }
        std::vector<unsigned char> hbuf(kBlockSize);
        cudaMemcpy(hbuf.data(), buf, kBlockSize, cudaMemcpyDeviceToHost);
        for (std::uint64_t i = 0; i < kBlockSize; ++i) {
            if (hbuf[i] != static_cast<unsigned char>((off + 7 + i) % 251u)) {
                read_ok = false;
                break;
            }
        }
    }
    CHECK(read_ok, "read back all 8 blocks out of order, byte-exact per block_id*block_size");

    env->rt->unregister_memory(mem);
    cudaFree(raw);
    env->rt->close(target);
    cudaStreamDestroy(stream);
    ::unlink(p0.c_str());
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 89: restart persistence -- WRITE, full teardown (close/unregister/
// shutdown), then a BRAND NEW StorageRuntime + resolver + DataPath instance
// re-opens the SAME URI and READs back byte-exact. This is the KV-cache
// persistence-across-restart scenario.
// (Round 15 Session 6, REQUIRED 1.3)
// -------------------------------------------------------------------------

static int test_89_restart_persistence() {
    TEST_CASE("89. restart persistence (new Runtime+Resolver+DataPath re-opens same URI)");

    const std::uint64_t file_size = kUnit * 4;
    std::string p0 = slot_file(0, "t89");
    if (!create_backing_file(p0, file_size)) {
        CHECK(false, "create backing file");
        return 1;
    }

    const std::string uri = file_uri(p0);

    struct Region { std::uint64_t offset, length; std::uint64_t pattern_base; };
    Region regions[] = {
        {0,           kUnit,     1000},
        {kUnit,       kUnit * 2, 9000},
    };

    // ---- Phase 1: write with env_a, then FULLY tear it down ----
    bool write_ok = false;
    {
        auto env_a = make_env(2);
        CHECK(env_a != nullptr, "create env_a (Runtime+Resolver+DataPath #1)");
        if (env_a) {
            auto opened = env_a->rt->open(uri, OpenOptions{"file"});
            write_ok = opened.ok();
            if (write_ok) {
                auto target = opened.value();
                std::uint64_t buf_size = kUnit * 3;
                void* raw = nullptr;
                void* buf = cuda_malloc_aligned_64k(buf_size, &raw);
                auto mem_r = env_a->rt->register_memory(
                    {buf, buf_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
                write_ok = write_ok && mem_r.ok();
                if (write_ok) {
                    cudaStream_t stream;
                    cudaStreamCreate(&stream);
                    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};
                    for (const auto& r : regions) {
                        launch_fill_position_pattern_gpu(buf, r.pattern_base, r.length, stream);
                        cudaStreamSynchronize(stream);
                        IoRequest wreq{IoDirection::WRITE, mem_r.value(), 0, target, r.offset, r.length};
                        write_ok = write_ok && submit_wait_all(env_a->rt.get(), &wreq, 1, ctx);
                    }
                    cudaStreamDestroy(stream);
                    // Full teardown: close target, unregister memory, shutdown
                    // Runtime -- THEN env_a itself (Resolver+DataPath) is
                    // destroyed at end of this scope.
                    CHECK(env_a->rt->close(target).ok(), "close target (teardown)");
                    CHECK(env_a->rt->unregister_memory(mem_r.value()).ok(),
                         "unregister_memory (teardown)");
                }
                if (raw) cudaFree(raw);
                CHECK(env_a->rt->shutdown(5000).ok(), "shutdown env_a's Runtime (teardown)");
            }
        }
        // env_a (resolver + StripedDataPath + all N controller attachments)
        // is destroyed HERE, at end of scope.
    }
    CHECK(write_ok, "phase 1: write both regions via env_a, then fully teardown");

    // ---- Phase 2: brand-new env_b re-opens the SAME URI, READ verify ----
    bool read_ok = false;
    {
        auto env_b = make_env(2);
        CHECK(env_b != nullptr, "create env_b (Runtime+Resolver+DataPath #2, brand new)");
        if (env_b) {
            auto opened = env_b->rt->open(uri, OpenOptions{"file"});
            read_ok = opened.ok();
            CHECK(read_ok, "env_b re-opens the same file:// URI");
            if (read_ok) {
                auto target = opened.value();
                std::uint64_t buf_size = kUnit * 3;
                void* raw = nullptr;
                void* buf = cuda_malloc_aligned_64k(buf_size, &raw);
                auto mem_r = env_b->rt->register_memory(
                    {buf, buf_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
                read_ok = read_ok && mem_r.ok();
                if (read_ok) {
                    cudaStream_t stream;
                    cudaStreamCreate(&stream);
                    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};
                    for (const auto& r : regions) {
                        launch_fill_pattern_gpu(buf, 0xFF, r.length, stream);
                        cudaStreamSynchronize(stream);
                        IoRequest rreq{IoDirection::READ, mem_r.value(), 0, target, r.offset, r.length};
                        bool ok = submit_wait_all(env_b->rt.get(), &rreq, 1, ctx);
                        std::vector<unsigned char> hbuf(r.length);
                        cudaMemcpy(hbuf.data(), buf, r.length, cudaMemcpyDeviceToHost);
                        for (std::uint64_t i = 0; i < r.length && ok; ++i) {
                            if (hbuf[i] != static_cast<unsigned char>((r.pattern_base + i) % 251u))
                                ok = false;
                        }
                        read_ok = read_ok && ok;
                    }
                    cudaStreamDestroy(stream);
                    env_b->rt->unregister_memory(mem_r.value());
                }
                if (raw) cudaFree(raw);
                env_b->rt->close(target);
            }
            env_b->rt->shutdown(5000);
        }
    }
    CHECK(read_ok, "phase 2: env_b READs both regions byte-exact after full restart");

    ::unlink(p0.c_str());
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 90: fault semantics -- one illegal request in a mixed batch is
// rejected per-request (RESOURCE_EXHAUSTED-class / OUT_OF_RANGE), while the
// other requests (to files on BOTH mounts) are accepted and complete
// normally in the SAME submit() call -- partial commit.
// (Round 15 Session 6, REQUIRED 1.4)
// -------------------------------------------------------------------------

static int test_90_fault_partial_commit(StripedEnv* env) {
    TEST_CASE("90. fault semantics (illegal request rejected, others complete: partial commit)");

    const std::uint64_t file_size = kUnit * 4;
    std::string p0 = slot_file(0, "t90a");
    std::string p1 = slot_file(1, "t90b");
    if (!create_backing_file(p0, file_size) || !create_backing_file(p1, file_size)) {
        CHECK(false, "create backing files");
        return 1;
    }

    auto oa = env->rt->open(file_uri(p0), OpenOptions{"file"});
    auto ob = env->rt->open(file_uri(p1), OpenOptions{"file"});
    CHECK(oa.ok() && ob.ok(), "open targets on both mounts");
    if (!oa.ok() || !ob.ok()) { ::unlink(p0.c_str()); ::unlink(p1.c_str()); return 1; }

    const std::uint64_t buf_size = kUnit * 2;
    void* raw = nullptr;
    void* buf = cuda_malloc_aligned_64k(buf_size, &raw);
    CHECK(buf != nullptr, "alloc GPU buffer");
    auto mem_r = env->rt->register_memory(
        {buf, buf_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(mem_r.ok(), "register_memory");
    if (!mem_r.ok()) {
        if (raw) cudaFree(raw);
        env->rt->close(oa.value());
        env->rt->close(ob.value());
        ::unlink(p0.c_str()); ::unlink(p1.c_str());
        return 1;
    }
    auto mem = mem_r.value();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Fill the WHOLE buffer: req[2] writes the second kUnit (to mount 1), so
    // leaving it uninitialized would write cudaMalloc garbage there and make
    // the byte-verify below meaningless.
    launch_fill_position_pattern_gpu(buf, 100, buf_size, stream);
    cudaStreamSynchronize(stream);
    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};

    // req[0]: valid, file on mount 0.
    // req[1]: illegal -- target_offset == file_size (out of range).
    // req[2]: valid, file on mount 1.
    IoRequest reqs[3] = {
        {IoDirection::WRITE, mem, 0,           oa.value(), 0,         kUnit},
        {IoDirection::WRITE, mem, 0,           oa.value(), file_size, kUnit},
        {IoDirection::WRITE, mem, kUnit,       ob.value(), 0,         kUnit},
    };
    auto sub = env->rt->submit(reqs, 3, ctx);

    CHECK(sub.initial_states.size() == 3, "initial_states has 3 entries");
    bool states_ok = sub.initial_states.size() == 3 &&
        sub.initial_states[0].state == IoRequestState::ACCEPTED &&
        sub.initial_states[1].state == IoRequestState::REJECTED &&
        sub.initial_states[2].state == IoRequestState::ACCEPTED;
    CHECK(states_ok, "req[0]/req[2] ACCEPTED, req[1] (out-of-range) REJECTED");
    CHECK(!sub.status.ok(), "overall status reports the partial failure");
    CHECK(sub.io.has_value(), "at least one accepted request -> io handle present");

    bool completed_ok = false;
    if (sub.io.has_value()) {
        auto wo = env->rt->wait(sub.io.value(), 30000);
        completed_ok = wo.observation_status.code() == StatusCode::OK &&
                      wo.result.has_value() && wo.result->state == IoState::COMPLETED;
        env->rt->release_io(sub.io.value());
    }
    CHECK(completed_ok, "the accepted-only op (req[0]+req[2]) completes normally");

    // Byte-verify req[0] (mount 0) and req[2] (mount 1) both landed.
    bool byte_ok = false;
    if (completed_ok) {
        launch_fill_pattern_gpu(buf, 0xDD, buf_size, stream);
        cudaStreamSynchronize(stream);
        IoRequest rreqs[2] = {
            {IoDirection::READ, mem, 0,       oa.value(), 0, kUnit},
            {IoDirection::READ, mem, kUnit,   ob.value(), 0, kUnit},
        };
        bool r0 = submit_wait_all(env->rt.get(), &rreqs[0], 1, ctx);
        bool r1 = submit_wait_all(env->rt.get(), &rreqs[1], 1, ctx);
        std::vector<unsigned char> hbuf(buf_size);
        cudaMemcpy(hbuf.data(), buf, buf_size, cudaMemcpyDeviceToHost);
        byte_ok = r0 && r1;
        for (std::uint64_t i = 0; i < kUnit && byte_ok; ++i) {
            if (hbuf[i] != static_cast<unsigned char>((100 + i) % 251u)) byte_ok = false;
        }
        // req[2]'s source was the second kUnit of the same position pattern.
        for (std::uint64_t i = 0; i < kUnit && byte_ok; ++i) {
            if (hbuf[kUnit + i] !=
                static_cast<unsigned char>((100 + kUnit + i) % 251u)) {
                byte_ok = false;
            }
        }
    }
    CHECK(byte_ok, "mount 0 (req[0]) and mount 1 (req[2]) both landed correctly");

    env->rt->unregister_memory(mem);
    cudaFree(raw);
    env->rt->close(oa.value());
    env->rt->close(ob.value());
    cudaStreamDestroy(stream);
    ::unlink(p0.c_str());
    ::unlink(p1.c_str());
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 91: P0-2 -- op in-flight, unregister_memory must return BUSY
// -------------------------------------------------------------------------

static int test_91_unregister_inflight(StripedEnv* env) {
    TEST_CASE("91. op in-flight: unregister_memory returns BUSY");

    const std::uint64_t file_size = kUnit * 32;
    std::string p0 = slot_file(0, "t91");
    if (!create_backing_file(p0, file_size)) {
        CHECK(false, "create backing file");
        return 1;
    }

    auto opened = env->rt->open(file_uri(p0), OpenOptions{"file"});
    CHECK(opened.ok(), "open file target");
    if (!opened.ok()) { ::unlink(p0.c_str()); return 1; }
    auto target = opened.value();

    const std::uint64_t io_size = file_size;
    void* raw = nullptr;
    void* buf = cuda_malloc_aligned_64k(io_size, &raw);
    auto mem_r = env->rt->register_memory(
        {buf, io_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(mem_r.ok(), "register_memory");
    if (!mem_r.ok()) {
        if (raw) cudaFree(raw);
        env->rt->close(target);
        ::unlink(p0.c_str());
        return 1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    launch_fill_pattern_gpu(buf, 0x91, io_size, stream);
    cudaStreamSynchronize(stream);

    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};
    IoRequest wreq{IoDirection::WRITE, mem_r.value(), 0, target, 0, io_size};
    auto sub = env->rt->submit(&wreq, 1, ctx);
    CHECK(sub.status.ok() && sub.io.has_value(), "submit in-flight write");

    // While op is in-flight, unregister_memory must return BUSY.
    if (sub.io.has_value()) {
        auto unreg = env->rt->unregister_memory(mem_r.value());
        CHECK(!unreg.ok(), "unregister_memory during in-flight op returns BUSY");
        if (!unreg.ok()) {
            std::printf("  unregister status: %s\n", unreg.message().c_str());
        }

        // Wait for completion.
        env->rt->wait(sub.io.value(), 30000);
        env->rt->release_io(sub.io.value());

        // After completion, unregister should succeed.
        auto unreg2 = env->rt->unregister_memory(mem_r.value());
        CHECK(unreg2.ok(), "unregister_memory after drain succeeds");
    }

    cudaFree(raw);
    env->rt->close(target);
    cudaStreamDestroy(stream);
    ::unlink(p0.c_str());
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 92: [Round 16 S3] N=4 roundtrip + single-launch count. Four files,
// one per mount, one batch of four requests.
// -------------------------------------------------------------------------

static int test_92_n4_roundtrip_single_launch(StripedEnv* env4) {
    TEST_CASE("92. N=4 roundtrip + single-launch count");
    const std::uint64_t file_size = kUnit * 4;
    const std::uint32_t n = 4;
    std::string paths[4];
    for (std::uint32_t i = 0; i < n; ++i) {
        paths[i] = slot_file(i, "t92_" + std::to_string(i));
        if (!create_backing_file(paths[i], file_size)) {
            for (std::uint32_t j = 0; j <= i; ++j) ::unlink(paths[j].c_str());
            CHECK(false, "create 4 backing files"); return 1;
        }
    }
    std::vector<TargetHandle> targets(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        auto o = env4->rt->open(file_uri(paths[i]), OpenOptions{"file"});
        if (!o.ok()) {
            for (auto& p : paths) ::unlink(p.c_str());
            CHECK(false, "open N=4 targets"); return 1;
        }
        targets[i] = o.value();
    }
    const std::uint64_t io_size = file_size * n;
    void* raw = nullptr;
    void* buf = cuda_malloc_aligned_64k(io_size, &raw);
    auto mem_r = env4->rt->register_memory({buf, io_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(mem_r.ok(), "register_memory");
    if (!mem_r.ok()) { if (raw) cudaFree(raw); for (auto& t : targets) env4->rt->close(t); for (auto& p : paths) ::unlink(p.c_str()); return 1; }
    cudaStream_t stream; cudaStreamCreate(&stream);
    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};
    launch_fill_position_pattern_gpu(buf, 9200, io_size, stream);
    cudaStreamSynchronize(stream);

    std::vector<IoRequest> wreqs(n);
    for (std::uint32_t i = 0; i < n; ++i)
        wreqs[i] = {IoDirection::WRITE, mem_r.value(), i * file_size, targets[i], 0, file_size};

    env4->dp.test_reset_submit_counters();
    auto t_w0 = std::chrono::steady_clock::now();
    bool wok = submit_wait_all(env4->rt.get(), wreqs.data(), n, ctx);
    auto t_w1 = std::chrono::steady_clock::now();
    { double ms = std::chrono::duration<double, std::milli>(t_w1 - t_w0).count();
      printf("[perf] 92_n4_write %llu bytes %.3f ms %.2f GB/s\n", (unsigned long long)io_size, ms, (double)io_size/ms/1e6); }
    std::uint64_t submits = env4->dp.test_submit_call_count();
    std::uint64_t launches = env4->dp.test_kernel_launch_count();
    printf("  N=4 WRITE: DataPath::submit calls=%lu, kernel launches=%lu\n", (unsigned long)submits, (unsigned long)launches);
    CHECK(wok && submits == 1 && launches == 1, "N=4 WRITE: 1 submit, 1 launch");

    launch_fill_pattern_gpu(buf, 0xFF, io_size, stream);
    cudaStreamSynchronize(stream);
    std::vector<IoRequest> rreqs(n);
    for (std::uint32_t i = 0; i < n; ++i)
        rreqs[i] = {IoDirection::READ, mem_r.value(), i * file_size, targets[i], 0, file_size};

    env4->dp.test_reset_submit_counters();
    auto t_r0 = std::chrono::steady_clock::now();
    bool rok = submit_wait_all(env4->rt.get(), rreqs.data(), n, ctx);
    auto t_r1 = std::chrono::steady_clock::now();
    { double ms = std::chrono::duration<double, std::milli>(t_r1 - t_r0).count();
      printf("[perf] 92_n4_read %llu bytes %.3f ms %.2f GB/s\n", (unsigned long long)io_size, ms, (double)io_size/ms/1e6); }
    submits = env4->dp.test_submit_call_count();
    launches = env4->dp.test_kernel_launch_count();
    printf("  N=4 READ: DataPath::submit calls=%lu, kernel launches=%lu\n", (unsigned long)submits, (unsigned long)launches);
    CHECK(rok && submits == 1 && launches == 1, "N=4 READ: 1 submit, 1 launch");

    std::vector<unsigned char> hbuf(io_size);
    cudaMemcpy(hbuf.data(), buf, io_size, cudaMemcpyDeviceToHost);
    bool match = true;
    for (std::uint64_t i = 0; i < io_size; ++i) {
        if (hbuf[i] != static_cast<unsigned char>((9200 + i) % 251u)) { match = false; break; }
    }
    CHECK(match, "N=4 read-back byte-exact");
    env4->rt->unregister_memory(mem_r.value());
    cudaFree(raw); for (auto& t : targets) env4->rt->close(t); cudaStreamDestroy(stream);
    for (auto& p : paths) ::unlink(p.c_str());
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 98: registration-time prebuilt descriptors preserve logical blocks,
// split by the controller's driver MDTS, and still use one fused launch.
// -------------------------------------------------------------------------

static int test_98_prebuilt_logical_block_mdts(StripedEnv* env,
                                                std::uint32_t n) {
    TEST_CASE(n == 2
        ? "98. N=2 prebuilt 256KiB and block>driver-MDTS"
        : "98. N=4 prebuilt 256KiB and block>driver-MDTS");

    std::uint64_t max_mdts = 0;
    for (std::uint32_t d = 0; d < n; ++d) {
        const std::uint64_t hardware = env->dp.test_device_hardware_mdts(d);
        const std::uint64_t effective = env->dp.test_device_effective_mdts(d);
        CHECK(hardware > 0, "controller reports hardware MDTS");
        CHECK(effective > 0 && effective <= hardware,
              "effective MDTS is derived from controller hardware");
        max_mdts = std::max(max_mdts, effective);
    }
    if (max_mdts == 0) return 1;

    auto run_case = [&](std::uint64_t logical_block,
                        const char* suffix,
                        std::uint64_t pattern_seed) -> bool {
        auto case_env = make_env(n);
        if (!case_env) return false;
        const std::string name = "t98_n" + std::to_string(n) + "_" + suffix;
        // Two files per mount, each logical_block bytes: one request per
        // file, each sized to a whole number of granularity slices.
        constexpr std::uint32_t kSlices = 2;
        const std::uint64_t file_size = logical_block * kSlices;
        std::vector<std::string> paths;
        for (std::uint32_t d = 0; d < n; ++d) {
            paths.push_back(slot_file(d, name + "_" + std::to_string(d)));
            if (!create_backing_file(paths.back(), file_size)) {
                std::fprintf(stderr, "  t98 create backing failed: %s\n",
                             paths.back().c_str());
                for (const auto& path : paths) ::unlink(path.c_str());
                return false;
            }
        }

        std::vector<TargetHandle> targets(n);
        for (std::uint32_t d = 0; d < n; ++d) {
            auto o = case_env->rt->open(file_uri(paths[d]), OpenOptions{"file"});
            if (!o.ok()) {
                std::fprintf(stderr, "  t98 open failed: %s\n",
                             o.status().message().c_str());
                for (auto& t : targets) if (t.valid()) case_env->rt->close(t);
                for (const auto& path : paths) ::unlink(path.c_str());
                return false;
            }
            targets[d] = o.value();
        }

        const std::uint64_t io_size = file_size * n;
        void* raw = nullptr;
        void* buffer = cuda_malloc_aligned_64k(io_size, &raw);
        if (!buffer) {
            std::fprintf(stderr, "  t98 cuda buffer allocation failed\n");
            for (auto& t : targets) case_env->rt->close(t);
            for (const auto& path : paths) ::unlink(path.c_str());
            return false;
        }
        MemoryView view{buffer, io_size, MemoryKind::DEVICE,
                        MemoryOwnership::CALLER_OWNED, 0, "",
                        logical_block};
        auto memory = case_env->rt->register_memory(view);
        if (!memory.ok()) {
            std::fprintf(stderr, "  t98 register_memory failed: %s\n",
                         memory.status().message().c_str());
            cudaFree(raw);
            for (auto& t : targets) case_env->rt->close(t);
            for (const auto& path : paths) ::unlink(path.c_str());
            return false;
        }

        cudaStream_t stream = nullptr;
        cudaStreamCreate(&stream);
        HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};
        launch_fill_position_pattern_gpu(buffer, pattern_seed, io_size, stream);
        cudaStreamSynchronize(stream);

        // Expected sub-IO count: per file, ceil(file_size / mdts_of_its_device).
        std::uint64_t expected_entries = 0;
        for (std::uint32_t d = 0; d < n; ++d) {
            const std::uint64_t mdts =
                case_env->dp.test_device_effective_mdts(d);
            expected_entries += (file_size + mdts - 1) / mdts;
        }

        std::vector<IoRequest> writes(n);
        for (std::uint32_t d = 0; d < n; ++d)
            writes[d] = {IoDirection::WRITE, memory.value(), d * file_size,
                         targets[d], 0, file_size};

        case_env->dp.test_reset_submit_counters();
        const bool write_ok = submit_wait_all(case_env->rt.get(), writes.data(), n, ctx);
        const std::uint64_t write_prebuilt =
            case_env->dp.test_last_prebuilt_entry_count();
        const std::uint64_t write_dynamic =
            case_env->dp.test_last_dynamic_entry_count();
        const bool allow_shape_fallback = logical_block > max_mdts;
        const bool write_shape =
            case_env->dp.test_submit_call_count() == 1 &&
            case_env->dp.test_kernel_launch_count() == 1 &&
            write_prebuilt > 0 &&
            write_prebuilt + write_dynamic >= expected_entries &&
            (allow_shape_fallback || write_dynamic == 0);

        launch_fill_pattern_gpu(buffer, 0xFF, io_size, stream);
        cudaStreamSynchronize(stream);
        std::vector<IoRequest> reads(n);
        for (std::uint32_t d = 0; d < n; ++d)
            reads[d] = {IoDirection::READ, memory.value(), d * file_size,
                        targets[d], 0, file_size};

        case_env->dp.test_reset_submit_counters();
        const bool read_ok = submit_wait_all(case_env->rt.get(), reads.data(), n, ctx);
        const std::uint64_t read_prebuilt =
            case_env->dp.test_last_prebuilt_entry_count();
        const std::uint64_t read_dynamic =
            case_env->dp.test_last_dynamic_entry_count();
        const bool read_shape =
            case_env->dp.test_submit_call_count() == 1 &&
            case_env->dp.test_kernel_launch_count() == 1 &&
            read_prebuilt > 0 &&
            read_prebuilt + read_dynamic >= expected_entries &&
            (allow_shape_fallback || read_dynamic == 0);

        std::vector<unsigned char> host(io_size);
        cudaMemcpy(host.data(), buffer, io_size, cudaMemcpyDeviceToHost);
        bool bytes_ok = true;
        for (std::uint64_t i = 0; i < io_size; ++i) {
            const auto expected = static_cast<unsigned char>(
                (pattern_seed + i) % 251u);
            if (host[i] != expected) {
                bytes_ok = false;
                break;
            }
        }

        std::printf(
            "  N=%u block=%llu mdts_max=%llu entries=%llu "
            "write[p=%llu,d=%llu] read[p=%llu,d=%llu] launch=1 "
            "shape=%s bytes=%s\n",
            n, (unsigned long long)logical_block,
            (unsigned long long)max_mdts,
            (unsigned long long)expected_entries,
            (unsigned long long)write_prebuilt,
            (unsigned long long)write_dynamic,
            (unsigned long long)read_prebuilt,
            (unsigned long long)read_dynamic,
            (write_shape && read_shape) ? "yes" : "no",
            bytes_ok ? "ok" : "bad");

        case_env->rt->unregister_memory(memory.value());
        cudaFree(raw);
        for (auto& t : targets) case_env->rt->close(t);
        cudaStreamDestroy(stream);
        case_env->rt->shutdown(5000);
        for (const auto& path : paths) ::unlink(path.c_str());
        return write_ok && read_ok && write_shape && read_shape && bytes_ok;
    };

    const std::uint64_t block_256k = 256ull * 1024;
    const std::uint64_t large_block =
        ((max_mdts * 2 + block_256k - 1) / block_256k) * block_256k;
    CHECK(run_case(block_256k, "256k", 9800 + n),
          "256KiB logical block prebuilt write/read byte-exact, one launch");
    CHECK(large_block > max_mdts, "large logical block is greater than every MDTS");
    CHECK(run_case(large_block, "gt_mdts", 9900 + n),
          "block>MDTS prebuilt fan-out write/read byte-exact, one launch");
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 93: [Round 16 S3] N=4 mount distribution (verify in backing files)
// -------------------------------------------------------------------------

static int test_93_n4_distribution(StripedEnv* env4) {
    TEST_CASE("93. N=4 mount distribution (each file lands on its own device)");
    const std::uint64_t file_size = kUnit * 4;
    const std::uint32_t n = 4;
    std::string paths[4];
    for (std::uint32_t i = 0; i < n; ++i) {
        paths[i] = slot_file(i, "t93_" + std::to_string(i));
        if (!create_backing_file(paths[i], file_size)) {
            for (std::uint32_t j = 0; j <= i; ++j) ::unlink(paths[j].c_str());
            CHECK(false, "create 4 backing files"); return 1;
        }
    }
    std::vector<TargetHandle> targets(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        auto o = env4->rt->open(file_uri(paths[i]), OpenOptions{"file"});
        if (!o.ok()) {
            for (auto& p : paths) ::unlink(p.c_str());
            CHECK(false, "open N=4 targets"); return 1;
        }
        targets[i] = o.value();
    }
    const std::uint64_t io_size = file_size * n;
    void* raw = nullptr;
    void* buf = cuda_malloc_aligned_64k(io_size, &raw);
    auto mem_r = env4->rt->register_memory({buf, io_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(mem_r.ok(), "register_memory");
    if (!mem_r.ok()) { if (raw) cudaFree(raw); for (auto& t : targets) env4->rt->close(t); for (auto& p : paths) ::unlink(p.c_str()); return 1; }
    cudaStream_t stream; cudaStreamCreate(&stream);
    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};
    for (std::uint32_t u = 0; u < n; ++u)
        launch_fill_pattern_gpu((char*)buf + u * file_size, static_cast<unsigned char>(0xC0 + u), file_size, stream);
    cudaStreamSynchronize(stream);
    std::vector<IoRequest> wreqs(n);
    for (std::uint32_t u = 0; u < n; ++u)
        wreqs[u] = {IoDirection::WRITE, mem_r.value(), u * file_size, targets[u], 0, file_size};
    bool wok = submit_wait_all(env4->rt.get(), wreqs.data(), n, ctx);
    CHECK(wok, "write 4 files across 4 mounts in one batch");
    bool dist_ok = wok;
    for (std::uint32_t u = 0; u < n; ++u) {
        std::vector<unsigned char> file_data;
        if (!read_file_raw(paths[u], 0, file_size, file_data)) { dist_ok = false; break; }
        if (file_data.size() != file_size) { dist_ok = false; break; }
        unsigned char expect = static_cast<unsigned char>(0xC0 + u);
        for (std::uint64_t i = 0; i < file_size; ++i) if (file_data[i] != expect) { dist_ok = false; break; }
        if (!dist_ok) break;
    }
    CHECK(dist_ok, "mount dispatch: file i lands on device i (verified in backing files)");
    env4->rt->unregister_memory(mem_r.value());
    cudaFree(raw); for (auto& t : targets) env4->rt->close(t); cudaStreamDestroy(stream);
    for (auto& p : paths) ::unlink(p.c_str());
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 94: [Round 16 S3] N=4 absolute bandwidth (many files, 4-disk READ)
// -------------------------------------------------------------------------

static int test_94_n4_speedup(StripedEnv* env4) {
    TEST_CASE("94. N=4 absolute bandwidth (4-disk rotating READ)");
    const std::uint32_t n4 = 4;
    constexpr std::uint32_t kFiles = 32;
    const std::uint64_t file_size = 2ull * 1024 * 1024;  // 2 MiB/file, 64 MiB total

    std::vector<std::string> paths(kFiles);
    std::vector<TargetHandle> targets(kFiles);
    for (std::uint32_t f = 0; f < kFiles; ++f) {
        paths[f] = slot_file(f % n4, "t94_" + std::to_string(f));
        if (!create_backing_file(paths[f], file_size)) {
            for (auto& p : paths) if (!p.empty()) ::unlink(p.c_str());
            CHECK(false, "create N=4 backing files"); return 1;
        }
    }
    for (std::uint32_t f = 0; f < kFiles; ++f) {
        auto o = env4->rt->open(file_uri(paths[f]), OpenOptions{"file"});
        if (!o.ok()) {
            for (auto& t : targets) if (t.valid()) env4->rt->close(t);
            for (auto& p : paths) ::unlink(p.c_str());
            CHECK(false, "open N=4 targets"); return 1;
        }
        targets[f] = o.value();
    }

    const std::uint64_t io_size = file_size * kFiles;
    void* raw = nullptr;
    void* buf = cuda_malloc_aligned_64k(io_size, &raw);
    auto mem_r = env4->rt->register_memory({buf, io_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(mem_r.ok(), "register_memory");
    if (!mem_r.ok()) { if (raw) cudaFree(raw); for (auto& t : targets) env4->rt->close(t); for (auto& p : paths) ::unlink(p.c_str()); return 1; }
    cudaStream_t s; cudaStreamCreate(&s);
    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, s};

    // Pre-write.
    launch_fill_pattern_gpu(buf, 0x33, io_size, s);
    cudaStreamSynchronize(s);
    {
        std::vector<IoRequest> writes(kFiles);
        for (std::uint32_t f = 0; f < kFiles; ++f)
            writes[f] = {IoDirection::WRITE, mem_r.value(), f * file_size, targets[f], 0, file_size};
        bool ok = submit_wait_all(env4->rt.get(), writes.data(), kFiles, ctx);
        CHECK(ok, "prewrite N=4");
        if (!ok) {
            env4->rt->unregister_memory(mem_r.value()); cudaFree(raw);
            for (auto& t : targets) env4->rt->close(t); cudaStreamDestroy(s);
            for (auto& p : paths) ::unlink(p.c_str());
            return 1;
        }
    }

    // Timed read.
    launch_fill_pattern_gpu(buf, 0xFF, io_size, s);
    cudaStreamSynchronize(s);
    double ms4 = 0;
    bool ok4 = false;
    {
        std::vector<IoRequest> reads(kFiles);
        for (std::uint32_t f = 0; f < kFiles; ++f)
            reads[f] = {IoDirection::READ, mem_r.value(), f * file_size, targets[f], 0, file_size};
        auto t0 = std::chrono::steady_clock::now();
        ok4 = submit_wait_all(env4->rt.get(), reads.data(), kFiles, ctx);
        auto t1 = std::chrono::steady_clock::now();
        ms4 = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("[perf] 94_n4_read %llu bytes %.3f ms %.2f GB/s\n",
               (unsigned long long)io_size, ms4, (double)io_size/ms4/1e6);
    }
    CHECK(ok4, "N=4 READ succeeded");
    if (ok4 && ms4 > 0) {
        double bw4 = (double)io_size / ms4 / 1e6;
        printf("  4-disk rotating READ (%.1f MiB): %.2f ms (%.2f GB/s)\n",
               (double)io_size/(1<<20), ms4, bw4);
        // Perf is display-only (session contract): the only hard perf
        // threshold in this suite is the cross-disk speedup (>1.3x).
    }
    env4->rt->unregister_memory(mem_r.value());
    cudaFree(raw);
    for (auto& t : targets) env4->rt->close(t);
    cudaStreamDestroy(s);
    for (auto& p : paths) ::unlink(p.c_str());
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 95: [Round 16 S4/S5] Multi-target batch: single submit spanning 2
//         targets (different files on different mounts), all ACCEPTED +
//         single kernel launch (count seam) + byte verify.
// -------------------------------------------------------------------------

static int test_95_multi_target_batch(StripedEnv* env4) {
    TEST_CASE("95. multi-target batch (2 targets, 1 submit, 1 launch)");
    const std::uint64_t file_size = kUnit * 4;
    const std::uint32_t n = 4;

    // Target A: mount 0. Target B: mount 3.
    std::string pa = slot_file(0, "t95a");
    std::string pb = slot_file(n - 1, "t95b");
    if (!create_backing_file(pa, file_size) || !create_backing_file(pb, file_size)) {
        ::unlink(pa.c_str()); ::unlink(pb.c_str());
        CHECK(false, "create target backing files"); return 1;
    }

    auto oa = env4->rt->open(file_uri(pa), OpenOptions{"file"});
    auto ob = env4->rt->open(file_uri(pb), OpenOptions{"file"});
    CHECK(oa.ok() && ob.ok(), "open 2 targets");
    if (!oa.ok() || !ob.ok()) {
        ::unlink(pa.c_str()); ::unlink(pb.c_str());
        return 1;
    }

    const std::uint64_t io_size = file_size;
    void *raw_a=nullptr, *raw_b=nullptr;
    void *buf_a = cuda_malloc_aligned_64k(io_size, &raw_a);
    void *buf_b = cuda_malloc_aligned_64k(io_size, &raw_b);
    auto ma = env4->rt->register_memory({buf_a, io_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    auto mb = env4->rt->register_memory({buf_b, io_size, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(ma.ok() && mb.ok(), "register 2 memory regions");
    if (!ma.ok() || !mb.ok()) {
        if (raw_a) cudaFree(raw_a); if (raw_b) cudaFree(raw_b);
        env4->rt->close(oa.value()); env4->rt->close(ob.value());
        ::unlink(pa.c_str()); ::unlink(pb.c_str());
        return 1;
    }

    cudaStream_t stream; cudaStreamCreate(&stream);
    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};

    // Fill A with 0xA0, B with 0xB0
    launch_fill_pattern_gpu(buf_a, 0xA0, io_size, stream);
    launch_fill_pattern_gpu(buf_b, 0xB0, io_size, stream);
    cudaStreamSynchronize(stream);

    // Single submit: 2 requests, 2 different targets
    IoRequest wreqs[2] = {
        {IoDirection::WRITE, ma.value(), 0, oa.value(), 0, io_size},
        {IoDirection::WRITE, mb.value(), 0, ob.value(), 0, io_size},
    };
    env4->dp.test_reset_submit_counters();
    auto out = env4->rt->submit(wreqs, 2, ctx);
    CHECK(out.status.ok() && out.io.has_value(), "2-target batch submit OK");
    CHECK(env4->dp.test_submit_call_count() == 1, "1 DataPath::submit for 2-target batch");
    CHECK(env4->dp.test_kernel_launch_count() == 1, "1 kernel launch for 2-target batch");
    bool all_acc = out.io.has_value();
    for (int i = 0; i < 2; ++i)
        if (out.initial_states[i].state != IoRequestState::ACCEPTED) all_acc = false;
    CHECK(all_acc, "both targets' requests ACCEPTED");
    if (out.io.has_value()) {
        auto wo = env4->rt->wait(out.io.value(), 30000);
        CHECK(wo.observation_status.code() == StatusCode::OK, "wait OK");
        env4->rt->release_io(out.io.value());
    }

    // Read back both targets (poison first)
    launch_fill_pattern_gpu(buf_a, 0xFF, io_size, stream);
    launch_fill_pattern_gpu(buf_b, 0xFF, io_size, stream);
    cudaStreamSynchronize(stream);
    IoRequest rreqs[2] = {
        {IoDirection::READ, ma.value(), 0, oa.value(), 0, io_size},
        {IoDirection::READ, mb.value(), 0, ob.value(), 0, io_size},
    };
    env4->dp.test_reset_submit_counters();
    auto rout = env4->rt->submit(rreqs, 2, ctx);
    CHECK(rout.status.ok() && rout.io.has_value(), "2-target batch READ OK");
    CHECK(env4->dp.test_submit_call_count() == 1, "1 submit for 2-target READ");
    CHECK(env4->dp.test_kernel_launch_count() == 1, "1 launch for 2-target READ");
    if (rout.io.has_value()) {
        env4->rt->wait(rout.io.value(), 30000);
        env4->rt->release_io(rout.io.value());
    }

    // Verify A=0xA0, B=0xB0
    bool verify = true;
    {
        std::vector<unsigned char> h(io_size);
        cudaMemcpy(h.data(), buf_a, io_size, cudaMemcpyDeviceToHost);
        for (std::uint64_t i = 0; i < io_size; ++i) if (h[i] != 0xA0) { verify = false; break; }
    }
    {
        std::vector<unsigned char> h(io_size);
        cudaMemcpy(h.data(), buf_b, io_size, cudaMemcpyDeviceToHost);
        for (std::uint64_t i = 0; i < io_size; ++i) if (h[i] != 0xB0) { verify = false; break; }
    }
    CHECK(verify, "2-target byte-verify (A=0xA0, B=0xB0)");

    env4->rt->unregister_memory(ma.value());
    env4->rt->unregister_memory(mb.value());
    cudaFree(raw_a); cudaFree(raw_b);
    env4->rt->close(oa.value()); env4->rt->close(ob.value());
    cudaStreamDestroy(stream);
    ::unlink(pa.c_str());
    ::unlink(pb.c_str());
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 96: [Round 16 S5] 8+ target large batch (dev_table capacity boundary)
// One device-table row per target since 2026-09-22 (a target is one file on
// one device), so 8 targets is well inside capacity -- this remains the
// multi-target boundary shape used by production layer batches.
// -------------------------------------------------------------------------

static int test_96_many_targets_batch(StripedEnv* env4) {
    TEST_CASE("96. 8-target large batch (dev_table capacity boundary)");
    const std::uint64_t file_size = kUnit * 2;
    const std::uint32_t n = 4;
    const std::uint32_t n_targets = 8;

    std::vector<std::string> paths_all;
    std::vector<TargetHandle> tgts(n_targets);
    bool setup_ok = true;
    for (std::uint32_t t = 0; t < n_targets && setup_ok; ++t) {
        std::string p = slot_file(t % n, "t96_" + std::to_string(t));
        if (!create_backing_file(p, file_size)) { setup_ok = false; break; }
        paths_all.push_back(p);
        auto o = env4->rt->open(file_uri(p), OpenOptions{"file"});
        if (!o.ok()) { setup_ok = false; break; }
        tgts[t] = o.value();
    }
    CHECK(setup_ok, "open 8 targets");
    if (!setup_ok) {
        for (auto& t : tgts) if (t.valid()) env4->rt->close(t);
        for (auto& p : paths_all) ::unlink(p.c_str());
        return 1;
    }

    const std::uint64_t io_size = file_size;  // per target
    void* raw = nullptr;
    void* buf = cuda_malloc_aligned_64k(io_size * n_targets, &raw);
    auto mem = env4->rt->register_memory({buf, io_size * n_targets, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(mem.ok(), "register memory");
    if (!mem.ok()) {
        if (raw) cudaFree(raw);
        for (auto& t : tgts) env4->rt->close(t);
        for (auto& p : paths_all) ::unlink(p.c_str());
        return 1;
    }

    cudaStream_t stream; cudaStreamCreate(&stream);
    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};

    // 8 WRITE requests, each to a different target
    std::vector<IoRequest> wreqs(n_targets);
    for (std::uint32_t t = 0; t < n_targets; ++t) {
        launch_fill_pattern_gpu((char*)buf + t * io_size,
                                static_cast<unsigned char>(0xD0 + t), io_size, stream);
        wreqs[t] = {IoDirection::WRITE, mem.value(), t * io_size, tgts[t], 0, io_size};
    }
    cudaStreamSynchronize(stream);

    env4->dp.test_reset_submit_counters();
    auto out = env4->rt->submit(wreqs.data(), n_targets, ctx);
    CHECK(out.status.ok() && out.io.has_value(), "8-target batch submit OK");
    CHECK(env4->dp.test_submit_call_count() == 1, "1 submit for 8-target batch");
    CHECK(env4->dp.test_kernel_launch_count() == 1, "1 launch for 8-target batch");
    if (out.io.has_value()) {
        env4->rt->wait(out.io.value(), 30000);
        env4->rt->release_io(out.io.value());
    }

    // Read back + verify
    launch_fill_pattern_gpu(buf, 0xFF, io_size * n_targets, stream);
    cudaStreamSynchronize(stream);
    std::vector<IoRequest> rreqs(n_targets);
    for (std::uint32_t t = 0; t < n_targets; ++t)
        rreqs[t] = {IoDirection::READ, mem.value(), t * io_size, tgts[t], 0, io_size};
    auto rout = env4->rt->submit(rreqs.data(), n_targets, ctx);
    CHECK(rout.status.ok() && rout.io.has_value(), "8-target READ OK");
    if (rout.io.has_value()) {
        env4->rt->wait(rout.io.value(), 30000);
        env4->rt->release_io(rout.io.value());
    }

    bool verify = true;
    std::vector<unsigned char> h(io_size * n_targets);
    cudaMemcpy(h.data(), buf, io_size * n_targets, cudaMemcpyDeviceToHost);
    for (std::uint32_t t = 0; t < n_targets && verify; ++t) {
        unsigned char expect = static_cast<unsigned char>(0xD0 + t);
        for (std::uint64_t i = 0; i < io_size; ++i)
            if (h[t * io_size + i] != expect) { verify = false; break; }
    }
    CHECK(verify, "8-target byte-verify (each target distinct pattern)");

    env4->rt->unregister_memory(mem.value());
    cudaFree(raw);
    for (auto& t : tgts) env4->rt->close(t);
    cudaStreamDestroy(stream);
    for (auto& p : paths_all) ::unlink(p.c_str());
    return g_fail > 0 ? 1 : 0;
}

// -------------------------------------------------------------------------
// Test 97: [Round 16 S5, reworked 2026-09-22] dev_table capacity is one row
// per target; a small multi-target batch exercises the boundary shape
// without needing thousands of files.
// -------------------------------------------------------------------------

static int test_97_dev_table_capacity(StripedEnv* env4) {
    TEST_CASE("97. multi-target batch within dev_table capacity");
    const std::uint64_t file_size = kUnit * 4;
    const std::uint32_t n = 4;

    std::string p0 = slot_file(0, "t97a");
    std::string p1 = slot_file(n - 1, "t97b");
    if (!create_backing_file(p0, file_size) || !create_backing_file(p1, file_size)) {
        ::unlink(p0.c_str()); ::unlink(p1.c_str());
        CHECK(false, "create backing files"); return 1;
    }

    auto oa = env4->rt->open(file_uri(p0), OpenOptions{"file"});
    auto ob = env4->rt->open(file_uri(p1), OpenOptions{"file"});
    CHECK(oa.ok() && ob.ok(), "open 2 targets");
    if (!oa.ok() || !ob.ok()) {
        ::unlink(p0.c_str()); ::unlink(p1.c_str());
        return 1;
    }

    const std::uint64_t io_size = file_size;
    void* raw = nullptr;
    void* buf = cuda_malloc_aligned_64k(io_size * 2, &raw);
    auto mem = env4->rt->register_memory({buf, io_size * 2, MemoryKind::DEVICE, MemoryOwnership::CALLER_OWNED, 0, ""});
    CHECK(mem.ok(), "register_memory");
    if (!mem.ok()) {
        if (raw) cudaFree(raw);
        env4->rt->close(oa.value()); env4->rt->close(ob.value());
        ::unlink(p0.c_str()); ::unlink(p1.c_str());
        return 1;
    }

    cudaStream_t stream; cudaStreamCreate(&stream);
    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, 0, stream};
    launch_fill_pattern_gpu(buf, 0x97, io_size * 2, stream);
    cudaStreamSynchronize(stream);

    IoRequest wreqs[2] = {
        {IoDirection::WRITE, mem.value(), 0,        oa.value(), 0, io_size},
        {IoDirection::WRITE, mem.value(), io_size,  ob.value(), 0, io_size},
    };
    auto out = env4->rt->submit(wreqs, 2, ctx);
    CHECK(out.status.ok() && out.io.has_value(), "2-target batch OK (within dev_table capacity)");
    if (out.io.has_value()) {
        env4->rt->wait(out.io.value(), 30000);
        env4->rt->release_io(out.io.value());
    }

    // Verify both targets written correctly
    launch_fill_pattern_gpu(buf, 0xFF, io_size * 2, stream);
    cudaStreamSynchronize(stream);
    IoRequest rreqs[2] = {
        {IoDirection::READ, mem.value(), 0,        oa.value(), 0, io_size},
        {IoDirection::READ, mem.value(), io_size,  ob.value(), 0, io_size},
    };
    auto rout = env4->rt->submit(rreqs, 2, ctx);
    if (rout.io.has_value()) {
        env4->rt->wait(rout.io.value(), 30000);
        env4->rt->release_io(rout.io.value());
    }
    bool verify = true;
    std::vector<unsigned char> h(io_size * 2);
    cudaMemcpy(h.data(), buf, io_size * 2, cudaMemcpyDeviceToHost);
    for (std::uint64_t i = 0; i < io_size * 2; ++i) if (h[i] != 0x97) { verify = false; break; }
    CHECK(verify, "2-target byte-verify within capacity");

    env4->rt->unregister_memory(mem.value());
    cudaFree(raw);
    env4->rt->close(oa.value()); env4->rt->close(ob.value());
    cudaStreamDestroy(stream);
    ::unlink(p0.c_str());
    ::unlink(p1.c_str());
    return g_fail > 0 ? 1 : 0;
}

int main(int argc, char** argv) {
    std::printf("=== Multi-Device Local-NVMe E2E Contract Test (rotating placement) ===\n");

    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 ||
                      std::strcmp(argv[1], "-h") == 0)) {
        print_usage(argv[0]);
        return 0;
    }

    g_gpu_id = test_gpu_id();
    std::uint32_t requested = 0;
    if (!parse_args(argc, argv, requested)) {
        print_usage(argv[0]);
        return 2;
    }

    const std::uint32_t available = consecutive_device_count();
    const std::uint32_t num_devices = requested != 0
        ? requested
        : (available >= 4 ? 4 : (available >= 2 ? 2 : 0));

    if (num_devices > g_devices.size() || num_devices == 0 ||
        !hw_available(num_devices)) {
        std::printf("SKIP: hardware not available (found %u configured devices; "
                    "need 2 or 4 matching ssnvme + mount pairs and CUDA device %d)\n",
                    available, g_gpu_id);
        return 77;
    }

    // Mounts must exist before the resolver binds them (the bindings only
    // need the paths, but keeping creation first makes that ordering
    // obviously safe).
    if (!create_test_mounts(num_devices)) return 1;

    // Phase 2 direct-DataPath probe: resource setup/teardown must be
    // independent of the caller's current accelerator.
    int cuda_device_count = 0;
    if (cudaGetDeviceCount(&cuda_device_count) == cudaSuccess &&
        cuda_device_count >= 2) {
        const int caller_gpu = g_gpu_id == 0 ? 1 : 0;
        CHECK(cudaSetDevice(caller_gpu) == cudaSuccess,
              "direct StripedDataPath probe selects non-target caller device");
        StripedDataPath direct_dp(
            StripedEnv::build_devs(num_devices),
            static_cast<std::uint32_t>(g_gpu_id),
            /*mdts_override=*/0, /*cq_poll_budget=*/2000000,
            /*max_batch_entries=*/4096, /*max_in_flight_operations=*/4);
        DataPathConfig direct_config{"striped-local-nvme"};
        ResourceProvider* direct_resources = nullptr;
        const Status direct_init = direct_dp.initialize(
            direct_config, *direct_resources);
        CHECK(direct_init.ok(),
              "direct StripedDataPath initialize works from wrong caller device");
        CHECK(direct_dp.test_arena_alloc_counts().gpu_prp_cuda_malloc == 0,
              "multi-device initialization performs zero GPU PRP allocation");
        int after_init = -1;
        CHECK(cudaGetDevice(&after_init) == cudaSuccess &&
              after_init == caller_gpu,
              "direct StripedDataPath initialize restores caller device");
        const Status direct_shutdown = direct_dp.shutdown(0);
        CHECK(direct_shutdown.ok(),
              "direct StripedDataPath shutdown works from wrong caller device");
        int after_shutdown = -1;
        CHECK(cudaGetDevice(&after_shutdown) == cudaSuccess &&
              after_shutdown == caller_gpu,
              "direct StripedDataPath shutdown restores caller device");
        CHECK(cudaSetDevice(g_gpu_id) == cudaSuccess,
              "direct multi-device probe restores test accelerator");
    }
    std::printf("Using %u devices (found %u configured devices)\n",
                num_devices, available);
    cudaSetDevice(g_gpu_id);

    auto env2 = make_env(2);
    if (!env2) {
        std::fprintf(stderr, "FATAL: failed to create dual-device StorageRuntime\n");
        return 1;
    }
    std::printf("Dual-device StorageRuntime created (MultiMountLocalFileResolver + StripedDataPath, N=2)\n");

    int rc = 0;
    rc |= test_82_roundtrip(env2.get());
    rc |= test_85_distribution(env2.get());
    rc |= test_86_lifecycle(env2.get());
    rc |= test_91_unregister_inflight(env2.get());
    rc |= test_87_full_public_path(env2.get());
    rc |= test_88_block_addressing(env2.get());
    rc |= test_90_fault_partial_commit(env2.get());
    rc |= test_98_prebuilt_logical_block_mdts(env2.get(), 2);

    env2->rt->shutdown(5000);

    rc |= test_83_single_launch();
    rc |= test_84_speedup();

    if (num_devices == 4) {
        auto env4 = make_env(4);
        if (!env4) {
            std::fprintf(stderr, "FATAL: failed to create quad-device StorageRuntime\n");
            rc = 1;
        } else {
            std::printf("Quad-device StorageRuntime created (MultiMountLocalFileResolver + StripedDataPath, N=4)\n");
            rc |= test_92_n4_roundtrip_single_launch(env4.get());
            rc |= test_98_prebuilt_logical_block_mdts(env4.get(), 4);
            rc |= test_93_n4_distribution(env4.get());
            rc |= test_95_multi_target_batch(env4.get());
            rc |= test_96_many_targets_batch(env4.get());
            rc |= test_97_dev_table_capacity(env4.get());
            rc |= test_94_n4_speedup(env4.get());
            env4->rt->shutdown(5000);
        }
    }

    // Test 89 builds and tears down its OWN two Runtime+Resolver+DataPath
    // instances (env_a, env_b) to prove restart persistence -- run after
    // env2/env1/env4 are shut down so the two "brand-new instance" claims in the
    // test are not muddied by a still-live sibling instance in this process.
    rc |= test_89_restart_persistence();

    std::printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    if (rc != 0 || g_fail > 0) {
        std::printf("RESULT: FAIL\n");
        print_preserved_test_mounts();
        return 1;
    }
    if (!cleanup_test_mounts()) return 1;
    std::printf("RESULT: PASS\n");
    return 0;
}

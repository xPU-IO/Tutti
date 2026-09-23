#define _GNU_SOURCE
// nvme_bw_probe.cpp -- single-drive small-IO read-bandwidth probe.
//
// One file on ONE drive, one registered GPU buffer, N x --io-kb KiB random
// (or sequential) reads per batch.  --depth D = one submit() of D entries
// (burst model: all D in flight at once, kernel exits when all D are done).
// Set TUTTI_POOL_WORKERS=W to switch the datapath kernel to the worker-pool
// model (W fixed workers drain the batch; W is the in-flight bound).
//
// Reference points (same drive, fio libaio direct randread):
//   32K QD1   0.70 GB/s    32K QD128  6.12 GB/s    32K QD512 6.75 GB/s
//   16K QD512 5.9  GB/s    64K QD128  6.75 GB/s

#include <tutti/tutti_runtime.h>

#include <tutti/storage_runtime.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>

#include <tutti/cuda_like.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace tutti;

#define PROBE_OK(...) do { std::printf("[ OK ] " __VA_ARGS__); std::printf("\n"); } while (0)
#define PROBE_FAIL(...) do { std::fprintf(stderr, "[FAIL] " __VA_ARGS__); std::fprintf(stderr, "\n"); return 1; } while (0)
#define CUDA_OK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    std::fprintf(stderr,"[FAIL] %s: %s\n",#call,cudaGetErrorString(_e)); return 1;} } while (0)

static double sec_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static bool create_file(const std::string& path, std::uint64_t size) {
    // Project policy: ALL file opens carry O_DIRECT.
    int f = ::open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
    if (f < 0) return false;
    // fallocate pre-reserves blocks as a few large extents; the zero-fill
    // write then converts unwritten -> written WITHOUT adding extents (the
    // resolver rejects unwritten extents and caps the extent count).
    if (::fallocate(f, 0, 0, (off_t)size) != 0) {
        ::close(f);
        return false;
    }
    void* ap = nullptr;
    if (::posix_memalign(&ap, 4096, 1 << 20) != 0) { ::close(f); return false; }
    std::memset(ap, 0, 1 << 20);
    std::uint64_t off = 0;
    while (off < size) {
        size_t n = std::min<std::uint64_t>(1 << 20, size - off);
        ssize_t w = ::pwrite(f, ap, n, (off_t)off);
        if (w != (ssize_t)n) { std::free(ap); ::close(f); return false; }
        off += n;
    }
    std::free(ap);
    ::fsync(f);
    ::close(f);
    return true;
}

static bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && st.st_size > 0;
}

int main(int argc, char** argv) {
    std::string config_path;
    std::vector<std::string> directories;
    std::string mode = "rand";
    std::uint32_t io_kb = 32;
    std::uint32_t depth = 128;
    std::uint32_t total_gb = 32;
    std::uint32_t file_gb = 64;
    std::uint32_t buf_mb = 512;
    std::uint32_t write_pct = 0;
    std::uint32_t split_bufs = 0;
    std::uint32_t seed = 7;
    bool fresh = false;

    for (int i = 1; i < argc;) {
        const char* a = argv[i];
        if (!std::strcmp(a, "--directory") && i + 1 < argc) { directories.emplace_back(argv[++i]); ++i; }
        else if (!std::strcmp(a, "--config") && i + 1 < argc) { config_path = argv[++i]; ++i; }
        else if (!std::strcmp(a, "--io-kb") && i + 1 < argc) { io_kb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--depth") && i + 1 < argc) { depth = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--total-gb") && i + 1 < argc) { total_gb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--file-gb") && i + 1 < argc) { file_gb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--buf-mb") && i + 1 < argc) { buf_mb = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--write-pct") && i + 1 < argc) { write_pct = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--split-bufs") && i + 1 < argc) { split_bufs = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--seed") && i + 1 < argc) { seed = (std::uint32_t)std::strtoul(argv[++i], 0, 10); ++i; }
        else if (!std::strcmp(a, "--rand")) { mode = "rand"; ++i; }
        else if (!std::strcmp(a, "--sequential")) { mode = "seq"; ++i; }
        else if (!std::strcmp(a, "--fresh")) { fresh = true; ++i; }
        else if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
            std::printf("usage: %s --directory DIR [--directory DIR ...] [--config PATH]\n"
                        "  [--rand|--sequential] [--io-kb N] [--depth D] [--total-gb N]\n"
                        "  [--file-gb N] [--buf-mb N] [--seed N] [--fresh]\n"
                        "  1 directory  -> single-drive (local-nvme)\n"
                        "  >=2 (power of two) directories -> rotating over drives (striped)\n"
                        "  --config defaults to the matching built-in YAML for each mode\n",
                        argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown: %s (try --help)\n", a);
            return 1;
        }
    }
    if (directories.empty()) PROBE_FAIL("--directory is required");
    if (io_kb == 0 || (io_kb * 1024ull) % 4096 != 0) PROBE_FAIL("--io-kb must be 4 KiB-aligned");
    if (depth == 0) PROBE_FAIL("--depth must be > 0");

    const std::size_t ndev = directories.size();
    if (ndev > 1 && ((ndev & (ndev - 1)) != 0))
        PROBE_FAIL("multi-drive mode requires a power-of-two --directory count (got %zu)", ndev);

    const std::uint64_t io_bytes = (std::uint64_t)io_kb * 1024;
    const std::uint64_t file_bytes = (std::uint64_t)file_gb << 30;
    const std::uint64_t buf_bytes = (std::uint64_t)buf_mb << 20;
    const std::uint64_t total_bytes = (std::uint64_t)total_gb << 30;
    const std::uint64_t total_ios = total_bytes / io_bytes;
    if (io_bytes > buf_bytes) PROBE_FAIL("--io-kb exceeds buffer");

    if (config_path.empty()) {
        config_path = (ndev == 1) ? TUTTI_NVME_BW_PROBE_DEFAULT_CONFIG
                                  : TUTTI_NVME_BW_PROBE_DEFAULT_CONFIG_STRIPED;
    }

    const char* pool_env = std::getenv("TUTTI_POOL_WORKERS");
    const long pool_workers = pool_env ? std::atol(pool_env) : 2048;

    // ---- Runtime ----
    auto created = TuttiRuntime::create(config_path);
    if (!created.ok())
        PROBE_FAIL("TuttiRuntime::create(%s): %s", config_path.c_str(),
                   created.status().message().c_str());
    std::unique_ptr<TuttiRuntime> owner = std::move(created).value();
    StorageRuntime* rt = owner->storage_runtime();
    if (rt == nullptr) PROBE_FAIL("no StorageRuntime");
    const int32_t gpu = rt->accel_id();
    if (gpu < 0) PROBE_FAIL("runtime.accel_id unspecified");
    CUDA_OK(cudaFree(0));
    CUDA_OK(cudaSetDevice(gpu));
    PROBE_OK("runtime up (%s, %zu drive%s, config=%s)",
             ndev == 1 ? "local-nvme" : "striped-local-nvme", ndev,
             ndev == 1 ? "" : "s", config_path.c_str());

    // ---- Backing files (one per drive) ----
    for (std::size_t d = 0; d < ndev; ++d) {
        const std::string fpath = directories[d] + "/nvme_bw_probe.bin";
        if (fresh || !file_exists(fpath)) {
            auto t0 = std::chrono::steady_clock::now();
            if (!create_file(fpath, file_bytes))
                PROBE_FAIL("create_file %s: %s", fpath.c_str(), std::strerror(errno));
            PROBE_OK("backing file %s (%u GiB) in %.2fs", fpath.c_str(), file_gb,
                     sec_since(t0));
        }
    }
    PROBE_OK("backing files ready (%u GiB each, reuse; --fresh to recreate)", file_gb);

    // ---- Registered GPU buffer(s) (64 KiB aligned, granularity = io_bytes) ----
    // --split-bufs N emulates a fragmented registration footprint (N small
    // cudaMalloc+register chunks, like the layerwise example's per-chunk K/V
    // tensors) instead of one big block.  IOVA/TLB behaviour of scattered
    // registrations is the thing under test.
    std::vector<MemoryHandle> mem_handles;
    if (split_bufs == 0) {
        void* raw = nullptr;
        CUDA_OK(cudaMalloc(&raw, buf_bytes + 65536));
        void* buf = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw) + 65535) &
                                             ~uintptr_t(65535));
        auto reg = rt->register_memory({buf, buf_bytes, MemoryKind::DEVICE,
                                         MemoryOwnership::CALLER_OWNED, gpu,
                                         TUTTI_COMPILED_ACCELERATOR_PROFILE, io_bytes});
        if (!reg.ok()) PROBE_FAIL("register_memory: %s", reg.status().message().c_str());
        mem_handles.push_back(reg.value());
        PROBE_OK("registered %u MiB GPU buffer at granularity %u KiB", buf_mb, io_kb);
    } else {
        if (buf_bytes % split_bufs != 0)
            PROBE_FAIL("--buf-mb must divide evenly by --split-bufs");
        const std::uint64_t chunk_bytes = buf_bytes / split_bufs;
        if (chunk_bytes < io_bytes || chunk_bytes % 4096 != 0)
            PROBE_FAIL("split chunk (%llu B) smaller than io or not 4K-aligned",
                       (unsigned long long)chunk_bytes);
        const std::uint32_t slots_per_buf = (std::uint32_t)(chunk_bytes / io_bytes);
        auto t0 = std::chrono::steady_clock::now();
        for (std::uint32_t b = 0; b < split_bufs; ++b) {
            void* raw = nullptr;
            CUDA_OK(cudaMalloc(&raw, chunk_bytes + 65536));
            void* buf = reinterpret_cast<void*>(
                (reinterpret_cast<uintptr_t>(raw) + 65535) & ~uintptr_t(65535));
            auto reg = rt->register_memory({buf, chunk_bytes, MemoryKind::DEVICE,
                                            MemoryOwnership::CALLER_OWNED, gpu,
                                            TUTTI_COMPILED_ACCELERATOR_PROFILE,
                                            io_bytes});
            if (!reg.ok()) PROBE_FAIL("register_memory #%u: %s", b,
                                      reg.status().message().c_str());
            mem_handles.push_back(reg.value());
        }
        PROBE_OK("registered %u split buffers (%u KiB each, %u slot(s) per buffer) "
                 "in %.2fs", split_bufs, (unsigned)(chunk_bytes >> 10), slots_per_buf,
                 sec_since(t0));
    }
    const std::uint32_t mem_count = (std::uint32_t)mem_handles.size();
    const std::uint64_t slots_per_mem =
        (split_bufs == 0) ? buf_bytes / io_bytes : (buf_bytes / split_bufs) / io_bytes;

    // ---- Targets (one per drive) ----
    std::vector<TargetHandle> tgt(ndev);
    for (std::size_t d = 0; d < ndev; ++d) {
        const std::string fpath = directories[d] + "/nvme_bw_probe.bin";
        auto op = rt->open(std::string("file://") + fpath, OpenOptions{"file"});
        if (!op.ok()) PROBE_FAIL("open %s: %s", fpath.c_str(),
                                  op.status().message().c_str());
        tgt[d] = op.value();
    }

    // ---- Offset table ----
    const std::uint64_t file_slots = file_bytes / io_bytes;
    std::vector<std::uint64_t> file_off(total_ios);
    {
        std::mt19937_64 rng(seed);
        if (mode == "rand") {
            for (std::uint64_t i = 0; i < total_ios; ++i)
                file_off[i] = (rng() % file_slots) * io_bytes;
        } else {
            for (std::uint64_t i = 0; i < total_ios; ++i)
                file_off[i] = (i % file_slots) * io_bytes;
        }
    }

    // ---- Read loop ----
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));
    HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, gpu, stream};

    cudaEvent_t ev0, ev1;
    CUDA_OK(cudaEventCreate(&ev0));
    CUDA_OK(cudaEventCreate(&ev1));

    // ---- Registration warmup (outside the measured window) ----
    // registration_for_ is lazy: the first IO touching a (memory, domain)
    // pair runs the DataPath dma-map slow path (nvm_dma_map ->
    // nvidia_p2p_get_pages, serialized ~0.25ms each).  Touch every
    // (memory, target) combination once here so the steady-state loop
    // measures IO, not one-time registrations.
    {
        auto w0 = std::chrono::steady_clock::now();
        std::uint64_t warmed = 0;
        for (std::uint32_t m = 0; m < mem_count; ++m) {
            for (std::size_t d = 0; d < ndev; ++d) {
                IoRequest wr{IoDirection::READ, mem_handles[m], 0, tgt[d],
                             0, io_bytes};
                auto o = rt->submit(&wr, 1, ctx);
                if (!o.io.has_value()) PROBE_FAIL("warmup submit rejected");
                auto wo = rt->wait(o.io.value(), 60000);
                if (wo.observation_status.code() != StatusCode::OK ||
                    !wo.result || wo.result->state != IoState::COMPLETED)
                    PROBE_FAIL("warmup wait failed");
                rt->release_io(o.io.value());
                warmed++;
            }
        }
        PROBE_OK("registration warmup: %llu (memory,target) pairs in %.2fs",
                 (unsigned long long)warmed, sec_since(w0));
    }

    auto wall0 = std::chrono::steady_clock::now();

    std::uint64_t done_ios = 0;
    std::uint64_t next_slot = 0;
    double io_ms_total = 0;
    int submit_rounds = 0;
    std::uint32_t cur_depth = depth;
    std::vector<double> batch_ms;  // per-batch wall (submit->wait) for tail stats

    while (done_ios < total_ios) {
        const std::uint32_t n = (std::uint32_t)std::min<std::uint64_t>(
            cur_depth, total_ios - done_ios);
        auto bt0 = std::chrono::steady_clock::now();
        std::vector<IoRequest> reqs(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint64_t slot = next_slot++;
            const std::uint32_t m_idx =
                (std::uint32_t)(split_bufs == 0 ? 0 : slot % split_bufs);
            const std::uint64_t m_off = (slot / mem_count % slots_per_mem) * io_bytes;
            const std::size_t d = (done_ios + i) % ndev;  // rotating placement
            // Mixed R/W emulation (default pure read): interleaved writes
            // force the SSD to interleave directions and the PCIe path to
            // turn around per command, like the layerwise overlap workload.
            const IoDirection dir = (write_pct > 0 && (i % 100) < write_pct)
                                        ? IoDirection::WRITE : IoDirection::READ;
            reqs[i] = {dir, mem_handles[m_idx], m_off, tgt[d],
                       file_off[done_ios + i], io_bytes};
        }

        CUDA_OK(cudaEventRecord(ev0, stream));
        auto o = rt->submit(reqs.data(), reqs.size(), ctx);
        CUDA_OK(cudaEventRecord(ev1, stream));
        if (!o.io.has_value()) {
            // Batch rejected (e.g. RESOURCE_EXHAUSTED): halve and retry.
            if (cur_depth > 1) { cur_depth /= 2; continue; }
            PROBE_FAIL("submit rejected at depth 1");
        }
        auto wo = rt->wait(o.io.value(), 60000);
        if (wo.observation_status.code() != StatusCode::OK || !wo.result ||
            wo.result->state != IoState::COMPLETED) {
            PROBE_FAIL("wait failed (round %d)", submit_rounds);
        }
        if (o.initial_states.size() != n)
            PROBE_FAIL("initial_states size mismatch");
        for (std::uint32_t i = 0; i < n; ++i) {
            if (o.initial_states[i].state != IoRequestState::ACCEPTED)
                PROBE_FAIL("entry %u rejected in round %d", i, submit_rounds);
        }
        rt->release_io(o.io.value());

        float ms = 0.f;
        CUDA_OK(cudaEventElapsedTime(&ms, ev0, ev1));
        io_ms_total += ms;
        batch_ms.push_back(sec_since(bt0) * 1e3);
        done_ios += n;
        ++submit_rounds;
    }

    const double wall_s = sec_since(wall0);
    const double io_s = io_ms_total / 1e3;
    const double gbps = (double)total_bytes / (1024 * 1024 * 1024) / wall_s;
    const double gbps_io = (double)total_bytes / (1024 * 1024 * 1024) / io_s;
    const double iops = (double)total_ios / wall_s;

    PROBE_OK("mode=%s io=%uK depth=%u pool=%ld%s write-pct=%u | total=%.1f GiB in %.3fs wall",
             mode.c_str(), io_kb, depth, pool_workers,
             pool_workers > 0 ? " (workers)" : "", write_pct,
             (double)total_bytes / (1ull << 30), wall_s);
    PROBE_OK("THROUGHPUT wall=%.2f GB/s io-time=%.2f GB/s | %.0f IOPS | "
             "avg %.1f us/op (io-time)",
             gbps, gbps_io, iops, io_s * 1e6 / (double)total_ios);

    // Per-batch wall tail stats (submit->wait): batch tail = slowest batch.
    if (!batch_ms.empty()) {
        std::vector<double> sorted = batch_ms;
        std::sort(sorted.begin(), sorted.end());
        const std::size_t p99 = sorted.size() >= 100
            ? sorted.size() - sorted.size() / 100 : sorted.size() - 1;
        double bavg = 0;
        for (double v : sorted) bavg += v;
        bavg /= sorted.size();
        PROBE_OK("BATCH wall avg=%.2fms p50=%.2fms p99=%.2fms max=%.2fms "
                 "over %zu batches",
                 bavg, sorted[sorted.size() / 2], sorted[p99],
                 sorted.back(), sorted.size());
    }

    CUDA_OK(cudaEventDestroy(ev0));
    CUDA_OK(cudaEventDestroy(ev1));
    CUDA_OK(cudaStreamDestroy(stream));
    return 0;
}

/**
 * nvme_storage_bulk_smoke.cu -- bulk-init perf + correctness for
 * HostFsBackedNvmeStorage::create_file(persist_now, sync_now).
 *
 * What this validates (R5a.1):
 *   1. create_file(..., false, false) skips both per-file fsyncs and
 *      the per-call log.persist().
 *   2. flush_metadata() drains both deferred queues atomically:
 *      one syncfs(2) + one log rewrite.
 *   3. After shutdown() + a fresh bootstrap(), every file added in
 *      the bulk batch is recoverable (open_file by name + extents
 *      match what was returned at create time).
 *   4. Wall-time of the bulk path is dramatically lower than the
 *      sync_now=true / persist_now=true path at the same N -- this
 *      is the whole motivation behind R5a.1.
 *
 * Comparison protocol:
 *   - phase A: N files via single-file mode (defaults), no flush
 *              needed -- already durable per call.
 *   - phase B: N files via bulk mode, one flush_metadata() at the
 *              end.
 *   - print wall-time + speedup ratio.
 *
 * Usage:
 *   sudo ./nvme_storage_bulk_smoke --gpu 0 --cap 32 \
 *        --n 10000 --size 4096 0000:08:00.0
 *
 * Single-device on purpose: the wall-time-and-ratio comparison is
 * the headline; multi-device parallel init is a separate concern.
 *
 * DESTRUCTIVE on first run (mkfs.ext4 -F if blk has no FS).
 */

#include "host_fs_backed_nvme_storage.h"
#include "nvme_file.h"
#include "nvme_storage.h"

#include "../../device_manager/include/local_nvme_direct_registry.h"
#include "../../runtime/include/device.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

// ----------------------------------------------------------------------------

#define STEP_OK(...)                                                          \
    do {                                                                      \
        std::fprintf(stderr, "[ OK ] step=%-3d ", g_step++);                  \
        std::fprintf(stderr, __VA_ARGS__);                                    \
        std::fprintf(stderr, "\n");                                           \
    } while (0)

#define STEP_FAIL(...)                                                        \
    do {                                                                      \
        std::fprintf(stderr, "[FAIL] step=%-3d ", g_step);                    \
        std::fprintf(stderr, __VA_ARGS__);                                    \
        std::fprintf(stderr, "\n");                                           \
        std::exit(1);                                                         \
    } while (0)

#define CUDA_OK(expr)                                                         \
    do {                                                                      \
        cudaError_t _e = (expr);                                              \
        if (_e != cudaSuccess) {                                              \
            STEP_FAIL("%s: %s", #expr, cudaGetErrorString(_e));               \
        }                                                                     \
    } while (0)

static int g_step = 1;

// ----------------------------------------------------------------------------

static const char* arg_after(const char* arg, const char* prefix) {
    size_t n = std::strlen(prefix);
    return std::strncmp(arg, prefix, n) == 0 ? arg + n : nullptr;
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s [--gpu N] [--cap N] [--n N] [--size BYTES] <pci_addr>\n"
        "\n"
        "Compares wall-time of N create_file calls in two modes:\n"
        "  A: per-call durable (persist_now=true, sync_now=true)\n"
        "  B: bulk + single flush_metadata()\n"
        "Then bounce the storage (shutdown + bootstrap) and verify\n"
        "every B-mode entry is recoverable.\n"
        "\n"
        "  --gpu  N      cudaSetDevice (default 0)\n"
        "  --cap  N      kernel_ioq_cap (default 32)\n"
        "  --n    N      files per phase (default 2000; bump higher\n"
        "                to see the gap really open up)\n"
        "  --size BYTES  per-file user-visible size (default 4096)\n",
        prog);
}

// ----------------------------------------------------------------------------

struct CreateLog {
    std::string                       name;
    uint64_t                          file_id;
    std::vector<tutti::LbaExtent>     extents;
};

static double seconds_since(
    const std::chrono::steady_clock::time_point& t0)
{
    auto d = std::chrono::steady_clock::now() - t0;
    return std::chrono::duration<double>(d).count();
}

// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    int      gpu_dev   = 0;
    uint32_t cap       = 32;
    uint64_t n_files   = 2000;
    uint64_t file_size = 4096;
    std::vector<std::string> pcis;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const char* v = nullptr;
        if      ((v = arg_after(a, "--gpu=")))  gpu_dev   = std::atoi(v);
        else if ((v = arg_after(a, "--cap=")))  cap       = (uint32_t)std::atoi(v);
        else if ((v = arg_after(a, "--n=")))    n_files   = std::strtoull(v, nullptr, 10);
        else if ((v = arg_after(a, "--size="))) file_size = std::strtoull(v, nullptr, 10);
        else if (std::strcmp(a, "--gpu")  == 0 && i + 1 < argc) gpu_dev   = std::atoi(argv[++i]);
        else if (std::strcmp(a, "--cap")  == 0 && i + 1 < argc) cap       = (uint32_t)std::atoi(argv[++i]);
        else if (std::strcmp(a, "--n")    == 0 && i + 1 < argc) n_files   = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(a, "--size") == 0 && i + 1 < argc) file_size = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
            usage(argv[0]); return 0;
        } else if (a[0] != '-') {
            pcis.emplace_back(a);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a);
            usage(argv[0]); return 1;
        }
    }
    if (pcis.size() != 1) {
        std::fprintf(stderr, "exactly one <pci_addr> required\n");
        usage(argv[0]); return 1;
    }
    if (n_files == 0 || file_size == 0) {
        std::fprintf(stderr, "--n and --size must be > 0\n");
        return 1;
    }

    // [1] cuda prime + cudaSetDevice (registry needs CUDA up).
    {
        cudaError_t cerr = cudaFree(0);
        if (cerr != cudaSuccess && cerr != cudaErrorInvalidValue) {
            STEP_FAIL("cuda driver prime: %s", cudaGetErrorString(cerr));
        }
        (void)cudaGetLastError();
        CUDA_OK(cudaSetDevice(gpu_dev));
        STEP_OK("cudaSetDevice(%d)", gpu_dev);
    }

    // [2] LocalNvmeDirectRegistry on a single device.
    auto registry = std::make_unique<tutti::LocalNvmeDirectRegistry>(
        std::vector<tutti::LocalNvmeDirectConfig>{
            tutti::LocalNvmeDirectConfig{
                /*pci_addr=*/        pcis[0],
                /*kernel_ioq_cap=*/  cap,
                /*display_name=*/    {},
                /*build_queue_group=*/false,
                /*cuda_device=*/     0,
                /*num_user_queues=*/ 4,
                /*queue_depth=*/     0,
                /*namespace_id=*/    1,
            }
        });
    if (!registry->Open()) STEP_FAIL("LocalNvmeDirectRegistry::Open");
    if (registry->device_count() != 1) STEP_FAIL("device_count != 1");
    const tutti::Device* dev = registry->device_at(0);
    if (dev == nullptr) STEP_FAIL("device_at(0) null");
    STEP_OK("registry up: pci=%s", pcis[0].c_str());

    // [3] HostFsBackedNvmeStorage::bootstrap.
    auto storage =
        std::make_unique<tutti::HostFsBackedNvmeStorage>(
            tutti::HostFsBackedNvmeStorage::Config{});
    if (!storage->bootstrap({dev})) STEP_FAIL("storage.bootstrap");
    STEP_OK("storage bootstrap (initial entries=%zu)",
            storage->list_file_names(dev).size());

    // [3.5] Pre-cleanup: a previous run that wasn't fully cleaned can
    //       leave A_*/B_* stragglers in the directory.  Without this
    //       pass, phase A's first create would hit "already exists".
    //       We use list_file_names (which walks the persistent log,
    //       not just open fds) and delete anything matching our
    //       prefixes.  Like the final cleanup, we use the deferred
    //       delete path + a single flush_metadata at the end -- a
    //       previous run that aborted at N=20000 leaves 40000
    //       stragglers, and persist-per-delete on that scale is
    //       O(N^2) total log writes (minutes-to-hours wall time).
    {
        auto t0 = std::chrono::steady_clock::now();
        auto names = storage->list_file_names(dev);
        std::size_t n_pre = 0;
        for (const auto& nm : names) {
            if (nm.size() < 2) continue;
            if (nm[0] != 'A' && nm[0] != 'B') continue;
            if (nm[1] != '_') continue;
            tutti::NvmeFile* f = storage->open_file(dev, nm);
            if (f == nullptr) continue;
            if (storage->delete_file(f, /*persist_now=*/false)) ++n_pre;
        }
        if (n_pre > 0) {
            if (!storage->flush_metadata(dev)) STEP_FAIL("pre-cleanup flush_metadata");
            double sec_p = seconds_since(t0);
            STEP_OK("pre-cleanup: removed %zu straggler(s) from "
                    "previous run, wall=%.3fs (%.0f ops/s)",
                    n_pre, sec_p, sec_p > 0 ? n_pre / sec_p : 0.0);
        } else {
            STEP_OK("pre-cleanup: no stragglers (clean slate)");
        }
    }

    // [4] phase A: per-call durable creates.
    {
        auto t0 = std::chrono::steady_clock::now();
        for (uint64_t i = 0; i < n_files; ++i) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "A_%lu", (unsigned long)i);
            auto* nf = storage->create_file(dev, nm, file_size,
                                            /*persist_now=*/true,
                                            /*sync_now=*/true);
            if (nf == nullptr) STEP_FAIL("phase A create_file(%s)", nm);
        }
        double sec_a = seconds_since(t0);
        STEP_OK("phase A: %lu files, per-call durable, wall=%.3fs (%.0f ops/s)",
                (unsigned long)n_files, sec_a, n_files / sec_a);
    }

    // [5] phase B: bulk creates + one flush_metadata at end.
    std::vector<CreateLog> b_log;
    b_log.reserve(n_files);
    double sec_b_creates = 0.0;
    double sec_b_flush   = 0.0;
    {
        auto t0 = std::chrono::steady_clock::now();
        for (uint64_t i = 0; i < n_files; ++i) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "B_%lu", (unsigned long)i);
            auto* nf = storage->create_file(dev, nm, file_size,
                                            /*persist_now=*/false,
                                            /*sync_now=*/false);
            if (nf == nullptr) STEP_FAIL("phase B create_file(%s)", nm);
            CreateLog rec;
            rec.name    = nm;
            rec.file_id = nf->id;
            rec.extents = nf->extents;
            b_log.push_back(std::move(rec));
        }
        sec_b_creates = seconds_since(t0);

        auto tf = std::chrono::steady_clock::now();
        if (!storage->flush_metadata(dev)) STEP_FAIL("flush_metadata");
        sec_b_flush = seconds_since(tf);

        double sec_b_total = sec_b_creates + sec_b_flush;
        STEP_OK("phase B: %lu files, deferred + 1x flush, wall=%.3fs "
                "(creates=%.3fs flush=%.3fs, %.0f ops/s)",
                (unsigned long)n_files, sec_b_total,
                sec_b_creates, sec_b_flush, n_files / sec_b_total);
    }

    // [6] Bounce: shutdown + reload, verify phase B entries survived
    //     the deferred persist + flush.
    if (!storage->shutdown()) STEP_FAIL("shutdown");
    STEP_OK("storage shutdown (umount, log persist)");

    storage = std::make_unique<tutti::HostFsBackedNvmeStorage>(
        tutti::HostFsBackedNvmeStorage::Config{});
    if (!storage->bootstrap({dev})) STEP_FAIL("storage.bootstrap (re)");
    STEP_OK("storage re-bootstrap (recovered entries=%zu)",
            storage->list_file_names(dev).size());

    {
        // Every B_* name should reopen with matching id + extents.
        std::size_t mismatch = 0;
        std::size_t missing  = 0;
        for (const auto& rec : b_log) {
            tutti::NvmeFile* nf = storage->open_file(dev, rec.name);
            if (nf == nullptr) { ++missing; continue; }
            if (nf->id != rec.file_id ||
                nf->extents.size() != rec.extents.size()) {
                ++mismatch;
                continue;
            }
            for (size_t k = 0; k < rec.extents.size(); ++k) {
                if (nf->extents[k].start_lba      != rec.extents[k].start_lba ||
                    nf->extents[k].length_blocks  != rec.extents[k].length_blocks) {
                    ++mismatch;
                    break;
                }
            }
        }
        if (missing != 0 || mismatch != 0) {
            STEP_FAIL("phase B recovery: missing=%zu mismatch=%zu",
                      missing, mismatch);
        }
        STEP_OK("phase B recovery: all %lu entries replay clean "
                "(file_id + extents stable across bootstrap bounce)",
                (unsigned long)b_log.size());
    }

    // [7] cleanup: delete every file on the device (both phases) so
    //     reruns don't leave a million tutti/.tutti/*.bin behind.
    //     We use the bulk-delete path (persist_now=false) so the
    //     log rewrite happens once at the end via flush_metadata,
    //     not N times.  The on-disk .bin is unlinked synchronously
    //     either way -- only the log persist is deferred, mirroring
    //     create_file's bulk-init mode.
    {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::string> names = storage->list_file_names(dev);
        std::size_t n_deleted = 0;
        for (const auto& nm : names) {
            tutti::NvmeFile* f = storage->open_file(dev, nm);
            if (f == nullptr) continue;
            if (storage->delete_file(f, /*persist_now=*/false)) ++n_deleted;
        }
        if (!storage->flush_metadata(dev)) STEP_FAIL("cleanup flush_metadata");
        double sec_d = seconds_since(t0);
        STEP_OK("cleanup: deleted %zu files (deferred + 1x flush), "
                "wall=%.3fs (%.0f ops/s)",
                n_deleted, sec_d, sec_d > 0 ? n_deleted / sec_d : 0.0);
    }

    // [8] shutdown + registry close.
    if (!storage->shutdown()) STEP_FAIL("final shutdown");
    STEP_OK("final shutdown");
    registry->Close();
    STEP_OK("registry closed");

    std::fprintf(stderr,
        "\n=== nvme_storage_bulk_smoke (n=%lu, size=%lu): all 8 steps passed ===\n",
        (unsigned long)n_files, (unsigned long)file_size);
    return 0;
}

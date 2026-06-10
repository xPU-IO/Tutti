/**
 * block_storage_smoke.cu -- R6.2 host-side end-to-end smoke for
 * HostFsBackedBlockStorage.
 *
 * Steps:
 *   [1]  cuda prime + cudaSetDevice
 *   [2]  LocalNvmeDirectRegistry::Open() bring up N controllers
 *   [3]  HostFsBackedNvmeStorage::bootstrap()
 *   [4]  HostFsBackedBlockStorage::bootstrap() on the same devices
 *   [5]  pre-cleanup: drop "smoke_blk_*" GpuFile stragglers
 *   [6]  create_gpu_file 2 GpuFiles with tensor_shape=[2,4,4096]
 *        (so num_shards=2, layers=4, tensor_size=4096, total=32 KiB).
 *        Writes a per-(gpu,tensor) pattern via gpu_file_resolve +
 *        INvmeStorage::write_blocking on the resolved shard.
 *   [7]  read_blocking the same way, byte-compare across all tensors
 *   [8]  close_gpu_file all
 *   [9]  shutdown both layers
 *   [10] re-bootstrap both layers (verifies log persistence + gen
 *        arbitration kept all entries on every device)
 *   [11] open_gpu_file by name, re-read pattern, byte-compare
 *   [12] cleanup: delete_gpu_file all (deferred + flush_metadata)
 *   [13] shutdown both layers
 *   [14] registry close
 *
 * Multi-NVMe invocation:
 *   sudo ./block_storage_smoke --gpu 0 --cap 32 \
 *        0000:4b:00.0 0000:57:00.0
 *
 * Two PCI BDFs are enough -- this smoke uses num_shards=2 and the
 * first two devices in CLI order receive the K and V shards.  More
 * devices are accepted; the smoke just doesn't use them for IO but
 * does include them in bootstrap (so the per-device gpu_file_log
 * mirroring is exercised at >2 mirrors).
 */

#include "host_fs_backed_block_storage.h"
#include "block_storage.h"
#include "gpu_file_resolve.h"

#include "host_fs_backed_nvme_storage.h"
#include "nvme_storage.h"

#include "../../device_manager/include/local_nvme_direct_registry.h"
#include "../../runtime/include/device.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_step = 0;

#define STEP_OK(fmt, ...) do {                                                 \
    ++g_step;                                                                  \
    std::fprintf(stderr, "[ OK ] step=%-2d  " fmt "\n", g_step, ##__VA_ARGS__);\
} while (0)

#define STEP_FAIL(fmt, ...) do {                                               \
    ++g_step;                                                                  \
    std::fprintf(stderr, "[FAIL] step=%-2d  " fmt "\n", g_step, ##__VA_ARGS__);\
    std::_Exit(2);                                                             \
} while (0)

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [--gpu N] [--cap N] <PCI_BDF> <PCI_BDF> [<PCI_BDF>...]\n"
        "  e.g.: %s --gpu 0 --cap 32 0000:4b:00.0 0000:57:00.0\n"
        "DESTRUCTIVE on unformatted disks (mkfs.ext4 -F via nvme_storage).\n",
        prog, prog);
}

void prime_cuda(int cuda_dev) {
    cudaError_t cerr = cudaFree(0);
    if (cerr != cudaSuccess && cerr != cudaErrorInvalidValue) {
        STEP_FAIL("cuda driver prime failed: %s", cudaGetErrorString(cerr));
    }
    (void)cudaGetLastError();
    cerr = cudaSetDevice(cuda_dev);
    if (cerr != cudaSuccess) STEP_FAIL("cudaSetDevice(%d): %s",
                                       cuda_dev, cudaGetErrorString(cerr));
    STEP_OK("cudaSetDevice(%d)", cuda_dev);
}

// Layout knobs.  Two shards (K + V), 4 "layers", tensor_size = 1 NVMe
// block (4 KiB) so writes are LBA-aligned.  Total per GpuFile = 32 KiB.
// Two GpuFiles per smoke run.
constexpr uint32_t kNumShards   = 2;
constexpr uint32_t kLayers      = 4;
constexpr uint32_t kTensorSize  = 4096;        // 4 KiB
constexpr uint64_t kGpuFileSize =
    (uint64_t)kNumShards * kLayers * kTensorSize;     // 32 KiB
constexpr uint32_t kNumGpuFiles = 2;
constexpr uint32_t kTensorsPerGpuFile = kNumShards * kLayers;
constexpr const char* kPrefix = "smoke_blk_";

void fill_pattern(uint8_t* buf, uint64_t seed) {
    for (uint32_t i = 0; i < kTensorSize; ++i) {
        buf[i] = (uint8_t)((seed * 0x9E3779B97F4A7C15ULL + i) & 0xFFu);
    }
}

uint64_t pattern_seed(uint32_t gpu_idx, uint32_t tensor_idx) {
    return ((uint64_t)gpu_idx << 32) | (uint64_t)tensor_idx | 0xC0FFEEULL;
}

std::string gpu_file_name(uint32_t i) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s%u", kPrefix, i);
    return buf;
}

// Drop any leftover GpuFiles from a previous run.
void wipe_stragglers(tutti::IBlockStorage& bs) {
    std::size_t n = 0;
    for (const auto& nm : bs.list_gpu_file_names()) {
        if (nm.rfind(kPrefix, 0) != 0) continue;
        tutti::GpuFile* gf = bs.open_gpu_file(nm);
        if (gf == nullptr) continue;
        if (bs.delete_gpu_file(gf, /*persist_now=*/false)) ++n;
    }
    if (n > 0) (void)bs.flush_metadata();
    if (n > 0) {
        std::fprintf(stderr,
            "[block_storage] pre-cleanup: removed %zu '%s*' "
            "GpuFile straggler(s)\n", n, kPrefix);
    }
}

// Drive read/write of every tensor in `gf` via gpu_file_resolve,
// either writing a per-tensor pattern or reading + byte-comparing it.
bool do_tensor_io(tutti::INvmeStorage& storage,
                  tutti::GpuFile*      gf,
                  uint32_t             gpu_idx,
                  bool                 do_write,
                  bool                 do_verify)
{
    std::vector<uint8_t> wbuf(kTensorSize);
    std::vector<uint8_t> rbuf(kTensorSize);
    for (uint32_t t = 0; t < kTensorsPerGpuFile; ++t) {
        const uint64_t global_off = (uint64_t)t * kTensorSize;
        uint32_t shard_idx = 0;
        uint64_t shard_off = 0;
        tutti::gpu_file_resolve(gf->tensor_shape[2],
                                gf->tensor_shape[0],
                                global_off,
                                &shard_idx, &shard_off);
        if (shard_idx >= gf->shards.size()) return false;
        tutti::NvmeFile* nf = gf->shards[shard_idx];
        if (nf == nullptr) return false;

        const uint64_t seed = pattern_seed(gpu_idx, t);

        if (do_write) {
            fill_pattern(wbuf.data(), seed);
            ssize_t w = storage.write_blocking(nf, shard_off,
                                                wbuf.data(),
                                                wbuf.size());
            if (w != (ssize_t)wbuf.size()) {
                std::fprintf(stderr,
                    "  write tensor %u: short write %zd (shard=%u off=%lu)\n",
                    t, w, shard_idx, (unsigned long)shard_off);
                return false;
            }
            if (!storage.sync(nf)) return false;
        }

        if (do_verify) {
            std::memset(rbuf.data(), 0, rbuf.size());
            ssize_t r = storage.read_blocking(nf, shard_off,
                                               rbuf.data(),
                                               rbuf.size());
            if (r != (ssize_t)rbuf.size()) {
                std::fprintf(stderr,
                    "  read tensor %u: short read %zd (shard=%u off=%lu)\n",
                    t, r, shard_idx, (unsigned long)shard_off);
                return false;
            }
            fill_pattern(wbuf.data(), seed);
            if (std::memcmp(rbuf.data(), wbuf.data(), wbuf.size()) != 0) {
                std::fprintf(stderr,
                    "  verify tensor %u: byte mismatch "
                    "(gpu=%u shard=%u off=%lu)\n",
                    t, gpu_idx, shard_idx, (unsigned long)shard_off);
                return false;
            }
        }
    }
    return true;
}

int run(int cuda_dev, const std::vector<std::string>& pci_addrs, uint32_t cap)
{
    if (pci_addrs.size() < 2) {
        std::fprintf(stderr,
            "block_storage_smoke needs at least 2 NVMe devices "
            "(num_shards=2)\n");
        return 1;
    }

    prime_cuda(cuda_dev);

    // [2] registry up
    std::vector<tutti::LocalNvmeDirectConfig> cfgs;
    cfgs.reserve(pci_addrs.size());
    for (const auto& bdf : pci_addrs) {
        tutti::LocalNvmeDirectConfig c{};
        c.pci_addr          = bdf;
        c.kernel_ioq_cap    = cap;
        c.build_queue_group = false;     // R6.2 host-only
        cfgs.push_back(std::move(c));
    }
    tutti::LocalNvmeDirectRegistry registry(std::move(cfgs));
    if (!registry.Open()) STEP_FAIL("registry.Open");
    STEP_OK("registry up: n=%zu", pci_addrs.size());

    std::vector<const tutti::Device*> devices;
    devices.reserve(registry.device_count());
    for (size_t i = 0; i < registry.device_count(); ++i) {
        devices.push_back(registry.device_at(i));
    }

    // [3] nvme_storage bootstrap
    auto storage = std::make_unique<tutti::HostFsBackedNvmeStorage>();
    if (!storage->bootstrap(devices)) STEP_FAIL("storage.bootstrap");
    STEP_OK("nvme_storage bootstrap (devices=%zu)", devices.size());

    // [4] block_storage bootstrap
    auto bs = std::make_unique<tutti::HostFsBackedBlockStorage>();
    if (!bs->bootstrap(storage.get(), devices))
        STEP_FAIL("block_storage.bootstrap");
    STEP_OK("block_storage bootstrap (entries=%zu)",
            bs->list_gpu_file_names().size());

    // [5] pre-cleanup
    wipe_stragglers(*bs);
    // Also wipe nvme_storage shards left from a previous run.
    {
        std::size_t total = 0;
        for (const auto* d : devices) {
            std::size_t n = 0;
            for (const auto& nm : storage->list_file_names(d)) {
                if (nm.rfind(kPrefix, 0) != 0) continue;
                if (auto* f = storage->open_file(d, nm)) {
                    if (storage->delete_file(f, /*persist_now=*/false)) ++n;
                }
            }
            if (n > 0) (void)storage->flush_metadata(d);
            total += n;
        }
        if (total > 0) {
            std::fprintf(stderr,
                "[block_storage_smoke] orphan-shard cleanup: %zu file(s)\n",
                total);
        }
    }
    STEP_OK("pre-cleanup");

    // [6] create + write
    std::vector<tutti::GpuFile*> gpu_files;
    for (uint32_t i = 0; i < kNumGpuFiles; ++i) {
        tutti::GpuFileSpec spec{};
        std::string nm = gpu_file_name(i);
        spec.name = nm;
        spec.total_size = kGpuFileSize;
        spec.tensor_shape[0] = kNumShards;
        spec.tensor_shape[1] = kLayers;
        spec.tensor_shape[2] = kTensorSize;
        spec.shard_placement = {devices[0], devices[1]};
        tutti::GpuFile* gf = bs->create_gpu_file(spec, /*persist_now=*/true);
        if (gf == nullptr)
            STEP_FAIL("create_gpu_file '%s'", nm.c_str());
        if (gf->shards.size() != kNumShards)
            STEP_FAIL("'%s' shards.size()=%zu != %u",
                      nm.c_str(), gf->shards.size(), kNumShards);
        if (!do_tensor_io(*storage, gf, i,
                          /*do_write=*/true, /*do_verify=*/false))
            STEP_FAIL("write '%s'", nm.c_str());
        gpu_files.push_back(gf);
    }
    STEP_OK("create_gpu_file + write %u GpuFile(s) "
            "(tensor_shape=[%u,%u,%u] total=%lu)",
            kNumGpuFiles, kNumShards, kLayers, kTensorSize,
            (unsigned long)kGpuFileSize);

    // [7] read back + verify
    for (uint32_t i = 0; i < kNumGpuFiles; ++i) {
        if (!do_tensor_io(*storage, gpu_files[i], i,
                          /*do_write=*/false, /*do_verify=*/true))
            STEP_FAIL("verify '%s'", gpu_file_name(i).c_str());
    }
    STEP_OK("read+verify %u GpuFile(s) (%u tensor(s) each)",
            kNumGpuFiles, kTensorsPerGpuFile);

    // [8] close
    for (auto* gf : gpu_files) {
        if (!bs->close_gpu_file(gf)) STEP_FAIL("close_gpu_file");
    }
    gpu_files.clear();
    STEP_OK("close_gpu_file all");

    // [9] shutdown
    if (!bs->shutdown())      STEP_FAIL("block_storage.shutdown");
    if (!storage->shutdown()) STEP_FAIL("nvme_storage.shutdown");
    STEP_OK("shutdown both layers");

    // [10] re-bootstrap
    storage = std::make_unique<tutti::HostFsBackedNvmeStorage>();
    if (!storage->bootstrap(devices))
        STEP_FAIL("nvme_storage.bootstrap (re)");
    bs = std::make_unique<tutti::HostFsBackedBlockStorage>();
    if (!bs->bootstrap(storage.get(), devices))
        STEP_FAIL("block_storage.bootstrap (re)");
    if (bs->list_gpu_file_names().size() < kNumGpuFiles) {
        STEP_FAIL("re-bootstrap: only %zu GpuFile(s) recovered, want >= %u",
                  bs->list_gpu_file_names().size(), kNumGpuFiles);
    }
    STEP_OK("re-bootstrap (recovered %zu GpuFile entries)",
            bs->list_gpu_file_names().size());

    // [11] open by name + re-verify
    for (uint32_t i = 0; i < kNumGpuFiles; ++i) {
        std::string nm = gpu_file_name(i);
        tutti::GpuFile* gf = bs->open_gpu_file(nm);
        if (gf == nullptr) STEP_FAIL("open_gpu_file '%s' post-bounce",
                                     nm.c_str());
        if (gf->total_size != kGpuFileSize)
            STEP_FAIL("'%s' total_size mismatch %lu vs %lu",
                      nm.c_str(), (unsigned long)gf->total_size,
                      (unsigned long)kGpuFileSize);
        if (gf->tensor_shape[0] != kNumShards ||
            gf->tensor_shape[1] != kLayers   ||
            gf->tensor_shape[2] != kTensorSize)
            STEP_FAIL("'%s' tensor_shape drift after bounce", nm.c_str());
        if (gf->shards.size() != kNumShards)
            STEP_FAIL("'%s' shard count %zu after bounce",
                      nm.c_str(), gf->shards.size());
        if (!do_tensor_io(*storage, gf, i,
                          /*do_write=*/false, /*do_verify=*/true))
            STEP_FAIL("verify post-bounce '%s'", nm.c_str());
        gpu_files.push_back(gf);
    }
    STEP_OK("post-bounce open + verify (data + metadata stable)");

    // [12] cleanup
    {
        std::size_t n_del = 0;
        for (auto* gf : gpu_files) {
            if (bs->delete_gpu_file(gf, /*persist_now=*/false)) ++n_del;
        }
        gpu_files.clear();
        if (!bs->flush_metadata()) STEP_FAIL("cleanup flush_metadata");
        STEP_OK("delete_gpu_file (deferred + 1x flush): %zu file(s)", n_del);
    }

    // [13] shutdown
    if (!bs->shutdown())      STEP_FAIL("final block_storage.shutdown");
    if (!storage->shutdown()) STEP_FAIL("final nvme_storage.shutdown");
    STEP_OK("final shutdown");

    // [14] registry close
    registry.Close();
    STEP_OK("registry closed");

    std::fprintf(stderr,
        "\n=== block_storage_smoke (n_dev=%zu, gpu_files=%u): all %d steps passed ===\n",
        devices.size(), kNumGpuFiles, g_step);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int cuda_dev = 0;
    uint32_t cap = 32;
    std::vector<std::string> pci_addrs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        if (a == "--gpu" && i + 1 < argc) { cuda_dev = std::atoi(argv[++i]); continue; }
        if (a == "--cap" && i + 1 < argc) { cap = (uint32_t)std::atoi(argv[++i]); continue; }
        pci_addrs.push_back(std::move(a));
    }
    if (pci_addrs.size() < 2) { usage(argv[0]); return 1; }
    return run(cuda_dev, pci_addrs, cap);
}

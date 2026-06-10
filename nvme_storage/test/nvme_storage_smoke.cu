/**
 * nvme_storage_smoke.cu -- exercise HostFsBackedNvmeStorage end-to-end
 * over one OR multiple NVMe controllers.
 *
 * Steps:
 *   [1]  cuda prime + cudaSetDevice
 *   [2]  LocalNvmeDirectRegistry::Open() bring up N controllers
 *   [3]  HostFsBackedNvmeStorage::bootstrap()  (mkfs.ext4 if empty,
 *                                                mount, init log)
 *   [4]  for each device: total/available capacity print
 *   [5]  for each device: create_file("smoke_<i>.bin", 1 MiB)
 *                          -> verify extents non-empty
 *   [6]  for each device: write_blocking pattern (0..1MiB-1 byte-counter)
 *   [7]  for each device: read_blocking back, verify byte-by-byte
 *   [8]  for each device: close_file
 *   [9]  for each device: open_file by name (round-trip via persistent log)
 *                          -> verify extents survive
 *   [10] for each device: delete_file
 *   [11] HostFsBackedNvmeStorage::shutdown()  (umount)
 *   [12] LocalNvmeDirectRegistry::Close()
 *
 * Single-NVMe invocation:
 *   sudo ./nvme_storage_smoke --gpu 0 --cap 32 0000:08:00.0
 *
 * Multi-NVMe invocation:
 *   sudo ./nvme_storage_smoke --gpu 0 --cap 32 \
 *        0000:4b:00.0 0000:57:00.0 0000:63:00.0
 *
 * DESTRUCTIVE on first run: if a target NVMe has no recognised
 * filesystem, mkfs.ext4 -F runs over it.  Don't run on disks you
 * care about.
 */

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
#include <string>
#include <vector>

namespace {

int g_step = 0;

#define STEP_OK(fmt, ...) do { \
    ++g_step; \
    std::fprintf(stderr, "[ OK ] step=%-2d  " fmt "\n", g_step, ##__VA_ARGS__); \
} while (0)

#define STEP_FAIL(fmt, ...) do { \
    ++g_step; \
    std::fprintf(stderr, "[FAIL] step=%-2d  " fmt "\n", g_step, ##__VA_ARGS__); \
    std::_Exit(2); \
} while (0)

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [--gpu N] [--cap N] [--mount-root PATH] <PCI_BDF> [<PCI_BDF>...]\n"
        "  e.g.: %s --gpu 0 --cap 32 0000:4b:00.0 0000:57:00.0 0000:63:00.0\n"
        "DESTRUCTIVE on unformatted disks (mkfs.ext4 -F).\n",
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

constexpr uint64_t kSmokeFileBytes = 1024 * 1024;   // 1 MiB

// Drop any "<prefix>*" stragglers from a previous aborted run on
// every device so this run is idempotent.  Uses the bulk-delete
// path (persist_now=false + flush_metadata) so a previous run that
// died with N=20000 unsynced files in flight doesn't take minutes
// to walk.
void wipe_stragglers(tutti::HostFsBackedNvmeStorage& storage,
                     const std::vector<const tutti::Device*>& devices,
                     const std::string& prefix)
{
    std::size_t total = 0;
    for (const auto* d : devices) {
        std::size_t n_dev = 0;
        for (const auto& nm : storage.list_file_names(d)) {
            if (nm.rfind(prefix, 0) != 0) continue;
            tutti::NvmeFile* f = storage.open_file(d, nm);
            if (f == nullptr) continue;
            if (storage.delete_file(f, /*persist_now=*/false)) ++n_dev;
        }
        if (n_dev > 0) (void)storage.flush_metadata(d);
        total += n_dev;
    }
    if (total > 0) {
        std::fprintf(stderr,
            "[nvme_storage] pre-cleanup: removed %zu '%s*' straggler(s) "
            "from previous run\n", total, prefix.c_str());
    }
}

void fill_pattern(std::vector<uint8_t>& buf, uint64_t seed) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = (uint8_t)((seed + i) & 0xFFu);
    }
}

bool buffers_equal(const std::vector<uint8_t>& a,
                   const std::vector<uint8_t>& b)
{
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size()) == 0;
}

int run(int cuda_dev, const std::vector<std::string>& pci_addrs,
        uint32_t cap, const std::string& mount_root)
{
    prime_cuda(cuda_dev);

    // [2] Bring up controllers.
    std::vector<tutti::LocalNvmeDirectConfig> cfgs;
    cfgs.reserve(pci_addrs.size());
    for (const auto& bdf : pci_addrs) {
        cfgs.push_back({bdf, cap, /*display_name=*/{}});
    }

    tutti::LocalNvmeDirectRegistry reg(std::move(cfgs));
    if (!reg.Open()) STEP_FAIL("LocalNvmeDirectRegistry::Open() (n=%zu)",
                                pci_addrs.size());

    std::vector<const tutti::Device*> devices;
    devices.reserve(reg.device_count());
    for (size_t i = 0; i < reg.device_count(); ++i) {
        devices.push_back(reg.device_at(i));
    }
    STEP_OK("registry up: n=%zu", devices.size());

    // [3] Storage bootstrap.
    tutti::HostFsBackedNvmeStorage::Config sc;
    sc.mount_root = mount_root;
    sc.auto_mkfs  = true;

    tutti::HostFsBackedNvmeStorage storage(sc);
    if (!storage.bootstrap(devices))
        STEP_FAIL("HostFsBackedNvmeStorage::bootstrap()");
    STEP_OK("HostFsBackedNvmeStorage bootstrap: mount_root=%s", mount_root.c_str());

    // Idempotency: drop any "smoke_*" leftovers from a previous run
    // that aborted before reaching delete_file.
    wipe_stragglers(storage, devices, "smoke_");

    // [4] Capacity print.
    for (size_t i = 0; i < devices.size(); ++i) {
        uint64_t total = storage.total_capacity   (devices[i]);
        uint64_t avail = storage.available_capacity(devices[i]);
        if (total == 0) STEP_FAIL("device %zu total_capacity == 0", i);
        STEP_OK("device %zu (id=%d) capacity total=%.2f GiB avail=%.2f GiB",
                i, devices[i]->device_id,
                total / (1024.0 * 1024.0 * 1024.0),
                avail / (1024.0 * 1024.0 * 1024.0));
    }

    // [5] create_file per device.
    std::vector<tutti::NvmeFile*> created(devices.size(), nullptr);
    for (size_t i = 0; i < devices.size(); ++i) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "smoke_%zu", i);
        auto* nf = storage.create_file(devices[i], nm, kSmokeFileBytes);
        if (nf == nullptr) STEP_FAIL("create_file(%s) on dev[%zu]", nm, i);
        if (nf->extents.empty())
            STEP_FAIL("create_file(%s) on dev[%zu]: empty extents", nm, i);
        created[i] = nf;
        STEP_OK("create_file '%s' on dev[%zu] id=%lu extents=%zu "
                "first_lba=%lu len=%lu",
                nm, i, (unsigned long)nf->id,
                nf->extents.size(),
                (unsigned long)nf->extents[0].start_lba,
                (unsigned long)nf->extents[0].length_blocks);
    }

    // [6] write_blocking pattern.
    std::vector<uint8_t> wbuf(kSmokeFileBytes, 0);
    for (size_t i = 0; i < devices.size(); ++i) {
        fill_pattern(wbuf, /*seed=*/0xC0DEu + i * 17);
        ssize_t r = storage.write_blocking(created[i], 0,
                                            wbuf.data(), wbuf.size());
        if (r != (ssize_t)wbuf.size())
            STEP_FAIL("write_blocking dev[%zu] r=%zd want=%zu (errno=%d)",
                      i, r, wbuf.size(), errno);
    }
    if (!storage.sync(created[0]))
        STEP_FAIL("sync(created[0])");
    STEP_OK("write_blocking + sync OK across %zu devices (1 MiB each)",
            devices.size());

    // [7] read_blocking back, byte-compare.
    std::vector<uint8_t> rbuf(kSmokeFileBytes, 0);
    for (size_t i = 0; i < devices.size(); ++i) {
        std::memset(rbuf.data(), 0, rbuf.size());
        ssize_t r = storage.read_blocking(created[i], 0,
                                            rbuf.data(), rbuf.size());
        if (r != (ssize_t)rbuf.size())
            STEP_FAIL("read_blocking dev[%zu] r=%zd want=%zu",
                      i, r, rbuf.size());

        fill_pattern(wbuf, /*seed=*/0xC0DEu + i * 17);
        if (!buffers_equal(rbuf, wbuf))
            STEP_FAIL("byte-compare mismatch on dev[%zu]", i);
    }
    STEP_OK("read_blocking byte-compare matches across %zu devices",
            devices.size());

    // [8] close_file each.
    for (size_t i = 0; i < devices.size(); ++i) {
        if (!storage.close_file(created[i]))
            STEP_FAIL("close_file dev[%zu]", i);
    }
    STEP_OK("close_file across %zu devices (fsync + log persist)",
            devices.size());

    // [9] open_file by name -- verify persistent log round-trip.
    for (size_t i = 0; i < devices.size(); ++i) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "smoke_%zu", i);
        auto* nf = storage.open_file(devices[i], nm);
        if (nf == nullptr) STEP_FAIL("open_file(%s) on dev[%zu]", nm, i);
        if (nf->extents.empty())
            STEP_FAIL("open_file(%s) on dev[%zu]: empty extents (log lost?)",
                      nm, i);
        if (nf->size_bytes != kSmokeFileBytes)
            STEP_FAIL("open_file(%s) wrong size %lu", nm,
                      (unsigned long)nf->size_bytes);
        created[i] = nf;
    }
    STEP_OK("open_file round-trip via PersistentFileLog OK");

    // [10] delete_file each.
    for (size_t i = 0; i < devices.size(); ++i) {
        if (!storage.delete_file(created[i]))
            STEP_FAIL("delete_file dev[%zu]", i);
    }
    STEP_OK("delete_file across %zu devices", devices.size());

    // [11] shutdown (umount).
    if (!storage.shutdown()) STEP_FAIL("HostFsBackedNvmeStorage::shutdown()");
    STEP_OK("storage shutdown (umount, log persist)");

    // [12] registry close (chrdev_remove + unbind).
    reg.Close();
    STEP_OK("registry closed (chrdev_remove + unbind, n=%zu)", devices.size());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int      cuda_dev    = 0;
    uint32_t cap         = 32;
    std::string mount_root = "/mnt/tutti";
    int      argi        = 1;

    while (argi < argc && argv[argi][0] == '-' && argv[argi][1] == '-') {
        const char* a = argv[argi];
        if (std::strcmp(a, "--gpu") == 0 && argi + 1 < argc) {
            cuda_dev = std::atoi(argv[++argi]); ++argi;
        } else if (std::strcmp(a, "--cap") == 0 && argi + 1 < argc) {
            cap = (uint32_t)std::atoi(argv[++argi]); ++argi;
        } else if (std::strcmp(a, "--mount-root") == 0 && argi + 1 < argc) {
            mount_root = argv[++argi]; ++argi;
        } else { usage(argv[0]); return 1; }
    }
    if (argi >= argc) { usage(argv[0]); return 1; }

    std::vector<std::string> pci_addrs;
    while (argi < argc) pci_addrs.emplace_back(argv[argi++]);

    int rc = run(cuda_dev, pci_addrs, cap, mount_root);
    if (rc == 0) {
        std::fprintf(stderr,
            "\n=== nvme_storage_smoke (n=%zu): all %d steps passed ===\n",
            pci_addrs.size(), g_step);
    }
    return rc;
}

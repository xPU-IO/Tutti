/**
 * nvme_storage_gpu_smoke.cu -- end-to-end GPU device-side submit (R5b).
 *
 * Layer: nvme_storage tests.
 *
 * Verifies the full GPU read path for one or many NVMe controllers:
 *   [ 1] cudaSetDevice(0) + cudaFree(0) prime
 *   [ 2] LocalNvmeDirectRegistry::Open() with build_queue_group=true
 *        and num_user_queues=4 per device, so each LocalNvmeDevice has
 *        a libnvm Controller with d_qps[] populated on the GPU.
 *   [ 3] HostFsBackedNvmeStorage::bootstrap()
 *   [ 4] for each device:  create_file("gpu_smoke_<i>", 1 MiB)
 *   [ 5] for each device:  write known pattern via write_blocking
 *                          (host pwrite, so we have known bytes to
 *                           read back on the GPU side later)
 *   [ 6] for each device:  cudaMalloc 1 MiB device buffer
 *                          + nvm_dma_map_data_device  -> ioaddrs[256]
 *   [ 7] for each device:  acquire_device_handle(file)  -> NvmeFileDeviceHandle*
 *                          (lives on GPU)
 *   [ 8] for each device:  launch <<<1,1>>> kernel that calls
 *                          submit_read_one(handle, ioaddrs[0], 0,
 *                                          off=0, nbytes=4096).
 *   [ 9] cudaDeviceSynchronize
 *   [10] for each device:  cudaMemcpy GPU buffer first 4 KiB to host
 *                          + byte-compare with the pattern from [5].
 *   [11] for each device:  release_device_handle(dh) + nvm_dma_unmap
 *                          + cudaFree(dev_buf) + close_file(file)
 *   [12] storage.shutdown
 *   [13] registry.Close
 *
 * If any byte mismatches at [10], the smoke fails loudly and exits.
 *
 * Usage:
 *   nvme_storage_gpu_smoke [--gpu N] [--cap N] [--cuda N]
 *       <pci_addr_0> [pci_addr_1 ...]
 *
 * Example (3 NVMe machine):
 *   sudo ./nvme_storage_gpu_smoke --gpu 0 --cap 32 --cuda 0 \
 *        0000:4b:00.0 0000:57:00.0 0000:63:00.0
 *
 * The NVMeService daemon MUST be down (this is direct mode); each
 * NVMe must already be unbound from stock nvme.
 */

#include "host_fs_backed_nvme_storage.h"
#include "nvme_file.h"
#include "nvme_file_device_handle.h"
#include "nvme_storage_device.cuh"

#include "../../device_manager/include/device_registry.h"
#include "../../device_manager/include/local_nvme_direct_registry.h"
#include "../../device_manager/include/nvmeservice_backed_registry.h"
#include "../../device_manager/include/local_nvme_device.h"
#include "../../runtime/include/device.h"

#include <nvm_ctrl.h>
#include <nvm_dma.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Step harness
// ----------------------------------------------------------------------------
static int g_step = 0;
#define STEP_OK(...)  do { \
    ++g_step; \
    std::fprintf(stderr, "[ OK ] step=%-3d ", g_step); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fputc('\n', stderr); \
} while (0)
#define STEP_FAIL(...) do { \
    ++g_step; \
    std::fprintf(stderr, "[FAIL] step=%-3d ", g_step); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fputc('\n', stderr); \
    std::exit(1); \
} while (0)
#define CUDA_OK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) STEP_FAIL("CUDA error: %s (%s)", \
                                     #call, cudaGetErrorString(_e)); \
} while (0)

// ----------------------------------------------------------------------------
// Kernel: one thread, one NVMe read.
// ----------------------------------------------------------------------------
__global__ void submit_read_one_kernel(const tutti::NvmeFileDeviceHandle* dh,
                                       uint64_t prp1, uint64_t prp2,
                                       uint64_t logical_off, uint64_t nbytes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        tutti::submit_read_one(dh, prp1, prp2, logical_off, nbytes);
    }
}

// ----------------------------------------------------------------------------
// Idempotency helper
// ----------------------------------------------------------------------------
//
// Drop any "<prefix>*" stragglers from a previous aborted run so this
// run can re-create the canonical names.  Uses the bulk-delete path
// (persist_now=false + flush_metadata) so a previous run that died
// with N >> 1 unsynced files in flight doesn't take minutes.
static void wipe_stragglers(
    tutti::HostFsBackedNvmeStorage& storage,
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

// ----------------------------------------------------------------------------
// CLI parsing
// ----------------------------------------------------------------------------
static const char* arg_after(const char* a, const char* prefix) {
    size_t n = std::strlen(prefix);
    if (std::strncmp(a, prefix, n) == 0) return a + n;
    return nullptr;
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --mode=direct  [--gpu N] [--cap N] [--cuda N] <pci_addr> [pci_addr ...]\n"
        "  %s --mode=service [--gpu N] [--cuda N] [--endpoint host:port]\n"
        "                    --device=<id0,id1,...> [--queues N]\n"
        "\n"
        "DIRECT  mode: NVMeService daemon must be DOWN.\n"
        "              Brings up every PCI BDF via LocalNvmeDirectRegistry\n"
        "              with build_queue_group=true.\n"
        "SERVICE mode: NVMeService daemon must be UP.\n"
        "              Connect()s a session per --device, attaches via\n"
        "              nvm_ctrl_attach_client, then the client itself runs\n"
        "              nvm_create_group + nvm_add_user_queue on its own fd\n"
        "              (Controller wrap mode).\n"
        "\n"
        "Common flow (both modes):\n"
        "  - storage.bootstrap (mount ext4 on each /dev/snvmeNn1)\n"
        "  - create_file(1 MiB) per device\n"
        "  - host-side write_blocking 4 KiB pattern + sync\n"
        "  - cudaMalloc + nvm_dma_map_data_device 1 MiB GPU buf per device\n"
        "  - acquire_device_handle + submit_read_one kernel reads first 4 KiB\n"
        "  - byte-compare GPU buffer against the host-written pattern\n"
        "\n"
        "Common args:\n"
        "  --gpu  N   cudaSetDevice + driver prime device (default 0)\n"
        "  --cuda N   cuda_device for Controller's d_qps allocations (default 0)\n"
        "Direct-only:\n"
        "  --cap  N   kernel_ioq_cap for NVM_SET_KERNEL_IOQ_CAP (default 32)\n"
        "Service-only:\n"
        "  --endpoint host:port  daemon gRPC endpoint (default 127.0.0.1:50051)\n"
        "  --device=<list>       comma-separated daemon device ids\n"
        "  --queues N            num_queues to request from daemon (default 4)\n"
        ,
        prog, prog);
}

static std::vector<int32_t> parse_int_csv(const char* s) {
    std::vector<int32_t> out;
    if (s == nullptr || *s == '\0') return out;
    const char* p = s;
    while (*p) {
        char* endp = nullptr;
        long v = std::strtol(p, &endp, 10);
        if (endp == p) break;
        out.push_back((int32_t)v);
        p = endp;
        while (*p == ',' || *p == ' ') ++p;
    }
    return out;
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string mode;
    int      gpu_dev   = 0;
    int      cuda_dev  = 0;
    uint32_t cap       = 32;
    std::vector<std::string> pcis;
    std::string endpoint = "127.0.0.1:50051";
    std::vector<int32_t> daemon_dev_ids;
    int32_t num_queues = 4;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const char* v = nullptr;
        if      ((v = arg_after(a, "--mode=")))     mode      = v;
        else if ((v = arg_after(a, "--gpu=")))      gpu_dev   = std::atoi(v);
        else if ((v = arg_after(a, "--cuda=")))     cuda_dev  = std::atoi(v);
        else if ((v = arg_after(a, "--cap=")))      cap       = (uint32_t)std::atoi(v);
        else if ((v = arg_after(a, "--endpoint="))) endpoint  = v;
        else if ((v = arg_after(a, "--device=")))   daemon_dev_ids = parse_int_csv(v);
        else if ((v = arg_after(a, "--queues=")))   num_queues = std::atoi(v);
        else if (std::strcmp(a, "--gpu")    == 0 && i + 1 < argc) { gpu_dev   = std::atoi(argv[++i]); }
        else if (std::strcmp(a, "--cuda")   == 0 && i + 1 < argc) { cuda_dev  = std::atoi(argv[++i]); }
        else if (std::strcmp(a, "--cap")    == 0 && i + 1 < argc) { cap       = (uint32_t)std::atoi(argv[++i]); }
        else if (std::strcmp(a, "--queues") == 0 && i + 1 < argc) { num_queues = std::atoi(argv[++i]); }
        else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
            usage(argv[0]); return 0;
        }
        else if (a[0] != '-') {
            pcis.emplace_back(a);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a);
            usage(argv[0]); return 1;
        }
    }
    if (mode != "direct" && mode != "service") {
        std::fprintf(stderr, "missing or invalid --mode=direct|service\n");
        usage(argv[0]); return 1;
    }
    if (mode == "direct" && pcis.empty()) {
        std::fprintf(stderr, "direct mode needs at least one <pci_addr>\n");
        usage(argv[0]); return 1;
    }
    if (mode == "service" && daemon_dev_ids.empty()) {
        std::fprintf(stderr, "service mode needs --device=<id,id,...>\n");
        usage(argv[0]); return 1;
    }

    // [1] cuda prime + cudaSetDevice
    {
        cudaError_t cerr = cudaFree(0);
        if (cerr != cudaSuccess && cerr != cudaErrorInvalidValue) {
            STEP_FAIL("cuda driver prime (cudaFree(0)) failed: %s",
                      cudaGetErrorString(cerr));
        }
        (void)cudaGetLastError();
        CUDA_OK(cudaSetDevice(gpu_dev));
        STEP_OK("cudaSetDevice(%d)", gpu_dev);
    }

    // [2] registry up.  We hold the parent through an IDeviceRegistry*
    //     so the rest of the smoke is mode-agnostic; the concrete impl
    //     is owned by `direct_reg` or `svc_reg` below.
    std::unique_ptr<tutti::LocalNvmeDirectRegistry>    direct_reg;
    std::unique_ptr<tutti::NvmeServiceBackedRegistry>  svc_reg;
    tutti::IDeviceRegistry* registry = nullptr;

    if (mode == "direct") {
        std::vector<tutti::LocalNvmeDirectConfig> cfgs;
        cfgs.reserve(pcis.size());
        for (const auto& p : pcis) {
            tutti::LocalNvmeDirectConfig c;
            c.pci_addr             = p;
            c.kernel_ioq_cap       = cap;
            c.build_queue_group = true;
            c.cuda_device          = cuda_dev;
            c.num_user_queues      = 4;
            cfgs.push_back(std::move(c));
        }
        direct_reg = std::make_unique<tutti::LocalNvmeDirectRegistry>(
            std::move(cfgs));
        if (!direct_reg->Open()) STEP_FAIL("LocalNvmeDirectRegistry::Open");
        registry = direct_reg.get();
        STEP_OK("registry up: mode=direct n=%zu (build_queue_group=true, q/dev=4)",
                registry->device_count());
    } else {
        std::vector<tutti::NvmeServiceBackedRequest> reqs;
        reqs.reserve(daemon_dev_ids.size());
        for (int32_t did : daemon_dev_ids) {
            tutti::NvmeServiceBackedRequest r;
            r.daemon_device_id     = did;
            r.cuda_device          = cuda_dev;
            r.num_queues           = num_queues;
            r.build_queue_group = true;
            r.num_user_queues      = 0;     // 0 == use full granted quota
            reqs.push_back(std::move(r));
        }
        svc_reg = std::make_unique<tutti::NvmeServiceBackedRegistry>(
            endpoint, std::move(reqs));
        if (!svc_reg->Open()) STEP_FAIL("NvmeServiceBackedRegistry::Open");
        registry = svc_reg.get();
        STEP_OK("registry up: mode=service endpoint=%s n=%zu "
                "(build_queue_group=true, q/dev=%d)",
                endpoint.c_str(), registry->device_count(), num_queues);
    }

    const std::size_t N = registry->device_count();
    if (N == 0) STEP_FAIL("device_count == 0");

    std::vector<const tutti::Device*> devices;
    devices.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        const tutti::Device* d = registry->device_at(i);
        if (d == nullptr) STEP_FAIL("device_at(%zu) returned null", i);
        auto* bp = static_cast<tutti::LocalNvmeDevice*>(d->backend_private);
        if (bp == nullptr || !bp->queue_group) {
            STEP_FAIL("dev[%zu] queue_group is null", i);
        }
        devices.push_back(d);
    }

    // [3] storage bootstrap
    auto storage = std::make_unique<tutti::HostFsBackedNvmeStorage>();
    if (!storage->bootstrap(devices)) STEP_FAIL("storage.bootstrap");
    STEP_OK("HostFsBackedNvmeStorage bootstrap");

    // Idempotency: drop any "gpu_smoke_*" leftovers from a previous
    // run that aborted before delete_file.
    wipe_stragglers(*storage, devices, "gpu_smoke_");

    constexpr uint64_t kFileBytes = 1ull * 1024 * 1024;   // 1 MiB
    constexpr uint64_t kIoBytes   = 4096;                  // one NVMe block

    // [4] create_file per device
    std::vector<tutti::NvmeFile*> files;
    files.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "gpu_smoke_%zu", i);
        auto* nf = storage->create_file(devices[i], nm, kFileBytes);
        if (nf == nullptr) STEP_FAIL("create_file(%s) on dev[%zu]", nm, i);
        if (nf->extents.empty()) STEP_FAIL("create_file(%s) empty extents", nm);
        files.push_back(nf);
    }
    STEP_OK("create_file 1 MiB on %zu device(s)", N);

    // [5] host-side write known pattern.  Use a pattern that's
    //     unique per device + per byte so a buffer mix-up would be
    //     obvious.
    std::vector<std::vector<uint8_t>> patterns(N);
    for (std::size_t i = 0; i < N; ++i) {
        patterns[i].resize(kIoBytes);
        const uint8_t base = (uint8_t)(0x40 + i);
        for (uint64_t k = 0; k < kIoBytes; ++k) {
            patterns[i][k] = base ^ (uint8_t)(k & 0xff);
        }
        ssize_t wr = storage->write_blocking(files[i], 0,
                                              patterns[i].data(), kIoBytes);
        if (wr != (ssize_t)kIoBytes) {
            STEP_FAIL("write_blocking(%zu) wr=%zd", i, wr);
        }
        if (!storage->sync(files[i])) STEP_FAIL("sync(%zu)", i);
    }
    STEP_OK("host-side write_blocking 4 KiB pattern + sync x %zu", N);

    // [6] per-device cudaMalloc + nvm_dma_map_data_device.
    struct DevBuf {
        void*       devptr     = nullptr;
        nvm_dma_t*  dma        = nullptr;
        uint64_t    prp1       = 0;
    };
    std::vector<DevBuf> bufs(N);

    for (std::size_t i = 0; i < N; ++i) {
        auto* bp = static_cast<tutti::LocalNvmeDevice*>(
            devices[i]->backend_private);

        CUDA_OK(cudaMalloc(&bufs[i].devptr, kFileBytes));
        // Zero the buffer so any read failure leaves obvious 0s.
        CUDA_OK(cudaMemset(bufs[i].devptr, 0, kFileBytes));

        int rc = nvm_dma_map_data_device(&bufs[i].dma,
                                         bp->ctrl,
                                         bufs[i].devptr,
                                         kFileBytes);
        if (rc != 0 || bufs[i].dma == nullptr) {
            STEP_FAIL("nvm_dma_map_data_device(dev[%zu]) rc=%d", i, rc);
        }
        if (bufs[i].dma->n_ioaddrs == 0) {
            STEP_FAIL("dev[%zu] DMA mapping has 0 ioaddrs", i);
        }
        bufs[i].prp1 = bufs[i].dma->ioaddrs[0];
    }
    STEP_OK("cudaMalloc(1 MiB) + nvm_dma_map_data_device per device "
            "(prp1[0]=0x%llx)", (unsigned long long)bufs[0].prp1);

    // [7] acquire_device_handle per file
    std::vector<tutti::NvmeFileDeviceHandle*> dhs(N, nullptr);
    for (std::size_t i = 0; i < N; ++i) {
        dhs[i] = storage->acquire_device_handle(files[i]);
        if (dhs[i] == nullptr) STEP_FAIL("acquire_device_handle(file[%zu])", i);
    }
    STEP_OK("acquire_device_handle per file (n=%zu)", N);

    // [8] launch one kernel per device, each issuing one
    //     submit_read_one for the first 4 KiB.
    for (std::size_t i = 0; i < N; ++i) {
        submit_read_one_kernel<<<1, 1>>>(dhs[i],
                                          bufs[i].prp1,
                                          /*prp2=*/0,
                                          /*logical_off=*/0,
                                          /*nbytes=*/kIoBytes);
    }
    STEP_OK("launched %zu submit_read_one kernel(s)", N);

    // [9] sync
    CUDA_OK(cudaDeviceSynchronize());
    {
        cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            STEP_FAIL("kernel reported error: %s", cudaGetErrorString(cerr));
        }
    }
    STEP_OK("cudaDeviceSynchronize after submit_read_one kernels");

    // [10] copy first 4 KiB back to host + byte-compare against pattern
    int total_mismatch = 0;
    for (std::size_t i = 0; i < N; ++i) {
        std::vector<uint8_t> got(kIoBytes, 0);
        CUDA_OK(cudaMemcpy(got.data(), bufs[i].devptr, kIoBytes,
                           cudaMemcpyDeviceToHost));
        size_t mismatch = 0;
        size_t first_bad = 0;
        for (uint64_t k = 0; k < kIoBytes; ++k) {
            if (got[k] != patterns[i][k]) {
                if (mismatch == 0) first_bad = k;
                ++mismatch;
            }
        }
        if (mismatch != 0) {
            std::fprintf(stderr,
                "  dev[%zu]: %zu mismatched bytes; first @ off=%zu "
                "(got=0x%02x want=0x%02x)\n",
                i, mismatch, first_bad,
                (unsigned)got[first_bad],
                (unsigned)patterns[i][first_bad]);
            total_mismatch += (int)mismatch;
        }
    }
    if (total_mismatch != 0) {
        STEP_FAIL("byte-compare FAILED: %d total mismatches across %zu device(s)",
                  total_mismatch, N);
    }
    STEP_OK("GPU read byte-compare matches host write across %zu device(s)", N);

    // [11] tear down per-device GPU buffers + handles
    for (std::size_t i = 0; i < N; ++i) {
        storage->release_device_handle(dhs[i]);
        if (bufs[i].dma != nullptr) {
            nvm_dma_unmap(bufs[i].dma);
            bufs[i].dma = nullptr;
        }
        if (bufs[i].devptr != nullptr) {
            cudaFree(bufs[i].devptr);
            bufs[i].devptr = nullptr;
        }
        if (!storage->close_file(files[i])) STEP_FAIL("close_file(%zu)", i);
    }
    STEP_OK("release_device_handle + nvm_dma_unmap + cudaFree + close_file (n=%zu)", N);

    // [12] storage shutdown
    if (!storage->shutdown()) STEP_FAIL("storage.shutdown");
    STEP_OK("storage.shutdown (umount, log persist)");

    // [13] registry close
    // [13] registry close
    if (direct_reg) {
        direct_reg->Close();
        STEP_OK("registry.Close direct (chrdev_remove + unbind, n=%zu)", N);
    } else {
        svc_reg->Close();
        STEP_OK("registry.Close service (free_client + Disconnect, n=%zu)", N);
    }

    std::fprintf(stderr,
        "\n=== nvme_storage_gpu_smoke (n=%zu): all %d steps passed ===\n",
        N, g_step);
    return 0;
}

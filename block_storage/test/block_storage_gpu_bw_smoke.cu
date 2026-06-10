/**
 * block_storage_gpu_bw_smoke.cu -- GpuFile end-to-end bandwidth +
 * compute/IO overlap demo (R6.4 prototype, includes R7-track logic).
 *
 * Layer: block_storage tests (R6 hot path under realistic load).
 *
 * Goals (set by user, demo deck):
 *   - 4 NVMe drives -> one big GpuFile (num_shards = 4).
 *   - Drive a multi-GiB read through the GpuFile abstraction at
 *     full per-drive PCIe x4 utilisation, so we can put a
 *     bandwidth number on the slide.
 *   - Increase per-thread IO size (= one NVMe IO at MDTS = 128 KiB)
 *     so we are not IOPS-bound but bandwidth-bound.
 *   - Concurrent GPU compute + IO on two streams: confirm they
 *     overlap (read kernel || compute kernel <= max(serial)).
 *
 * R6 layering caveat (READ THIS):
 *   - tensor_size = 128 KiB > 2 NVMe pages, so each NVMe IO needs
 *     a full PRP list (NVMe spec: prp1 = data_page[0] IOVA;
 *     prp2 = IOVA of a 4 KiB page packing the remaining
 *     data_page[1..N-1] IOVAs as little-endian uint64_t entries).
 *   - PRP-list construction is the R7 `memory/` layer's job once
 *     descriptor_slice / register_tensor land.  This smoke
 *     prototypes that logic IN-FILE (build_prp_lists() below) so
 *     we can ship the bandwidth number now.  When R7 lands, the
 *     prototype goes away and io_engine pulls PRPMappingEntry
 *     tables straight from memory/.
 *   - block_storage's public surface is unchanged.  We only
 *     consume R6.3's GpuFileHandle::d_shards_host[] and the
 *     R5b `submit_read_one(prp1, prp2)` device entry-point.
 *
 * Layout (defaults; CLI tunable):
 *
 *   num_shards      = 4
 *   tensor_size     = 128 KiB                  (32 x 4 KiB pages)
 *   total_bytes     = 8 GiB                    (--total-gib)
 *   tensors_per_run = 65536                    (= total / tensor_size)
 *   tensors/shard   = 16384
 *   per-shard data buffer = 2 GiB              (cudaMalloc + nvm_dma_map)
 *   per-shard PRP-list buffer = tensors/shard * 4 KiB = 64 MiB
 *   total GPU memory ~ 4 * (2 GiB + 64 MiB)   ~ 8.25 GiB
 *
 * Step list:
 *   [ 1] cuda prime + cudaSetDevice
 *   [ 2] LocalNvmeDirectRegistry::Open (4 BDFs, build_queue_group=true,
 *        num_user_queues=N from --queues)
 *   [ 3] HostFsBackedNvmeStorage::bootstrap
 *   [ 4] HostFsBackedBlockStorage::bootstrap
 *   [ 5] pre-cleanup gpu_bw_*
 *   [ 6] create_gpu_file (4 shards over devices[0..3])
 *   [ 7] per-shard cudaMalloc(2 GiB) + nvm_dma_map_data_device for data
 *   [ 8] per-shard cudaMalloc(prp_list_bytes) + nvm_dma_map_data_device
 *        for PRP lists; pack PRP-list pages on host then cudaMemcpy
 *   [ 9] block_storage->acquire_device_handle
 *   [10] host-stage IO context array (dh, prp1, prp2, shard_off, len);
 *        cudaMemcpy to GPU
 *   [11] warmup IO kernel (results discarded, not timed)
 *   [12] timed IO kernel via cudaEvent -> bandwidth GB/s
 *   [13] timed compute kernel only -> baseline
 *   [14] timed serial(IO -> compute) on default stream
 *   [15] timed overlap(IO || compute) on stream_io / stream_compute
 *        report speedup serial / overlap
 *   [16] release_device_handle + nvm_dma_unmap (data + prp_list) +
 *        cudaFree per shard
 *   [17] delete_gpu_file
 *   [18] shutdown both layers
 *   [19] registry close
 *
 * This smoke is READ-ONLY; it does not write data to NVMe before
 * reading.  The bytes that come back are whatever happened to be
 * on those LBAs; that's fine for a bandwidth test.  Correctness of
 * the GpuFile read path is already covered by block_storage_gpu_smoke.
 *
 * Usage:
 *   sudo ./block_storage_gpu_bw_smoke --gpu 0 --cuda 0 --cap 32 \
 *        --queues 16 --total-gib 8 --compute-iters 5000000 \
 *        0000:08:00.0 0000:4b:00.0 0000:57:00.0 0000:63:00.0
 */

#include "host_fs_backed_block_storage.h"
#include "block_storage.h"
#include "gpu_file_resolve.h"

#include "host_fs_backed_nvme_storage.h"
#include "nvme_storage.h"
#include "nvme_file.h"
#include "nvme_file_device_handle.h"
#include "nvme_storage_device.cuh"

#include "../../device_manager/include/local_nvme_direct_registry.h"
#include "../../device_manager/include/local_nvme_device.h"
#include "../../runtime/include/device.h"

#include <nvm_ctrl.h>
#include <nvm_dma.h>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// NVTX range helper.  Push at construction, pop at destruction; in
// nsys timelines this shows up as a coloured strip with the supplied
// name, parented to the calling thread's NVTX swimlane.
//
// Two flavours used here:
//   NvtxRange("name")                  -- default colour, simple push
//   NvtxRange("name", 0xFF00FFAA)      -- supply ARGB so the demo
//                                         video uses consistent
//                                         colours across phases
// ---------------------------------------------------------------------------
struct NvtxRange {
    NvtxRange(const char* name, uint32_t argb) {
        nvtxEventAttributes_t a = {};
        a.version       = NVTX_VERSION;
        a.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        a.colorType     = NVTX_COLOR_ARGB;
        a.color         = argb;
        a.messageType   = NVTX_MESSAGE_TYPE_ASCII;
        a.message.ascii = name;
        nvtxRangePushEx(&a);
    }
    explicit NvtxRange(const char* name) {
        nvtxRangePushA(name);
    }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&)            = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

// Common phase colours so the timeline is easy to read at a glance.
constexpr uint32_t kClrSetup    = 0xFF607D8B;   // grey-blue
constexpr uint32_t kClrAlloc    = 0xFF8B5CF6;   // purple
constexpr uint32_t kClrStage    = 0xFFEAB308;   // amber
constexpr uint32_t kClrIO       = 0xFF22C55E;   // green
constexpr uint32_t kClrCompute  = 0xFFEF4444;   // red
constexpr uint32_t kClrOverlap  = 0xFF3B82F6;   // blue
constexpr uint32_t kClrTeardown = 0xFF6B7280;   // dim grey

// ---------------------------------------------------------------------------
// Step harness
// ---------------------------------------------------------------------------
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

#define CUDA_OK(call) do {                                                     \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) STEP_FAIL("CUDA error: %s (%s)",                    \
                                     #call, cudaGetErrorString(_e));           \
} while (0)

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [--gpu N] [--cuda N] [--cap N] [--queues N]\n"
        "          [--total-gib F] [--compute-iters N]\n"
        "          <PCI0> <PCI1> <PCI2> <PCI3>\n"
        "Defaults: queues=16 total-gib=8 compute-iters=5000000\n"
        "Needs exactly 4 PCI BDFs (num_shards = 4).\n"
        "DESTRUCTIVE on unformatted disks (mkfs.ext4 -F via nvme_storage).\n",
        prog);
}

// ---------------------------------------------------------------------------
// Layout knobs (hard-coded / runtime params)
// ---------------------------------------------------------------------------
constexpr uint32_t kNumShards       = 4;
constexpr uint32_t kTensorSize      = 128 * 1024;        // 128 KiB
constexpr uint64_t kPageSize        = 4096;
constexpr uint32_t kPagesPerTensor  = kTensorSize / kPageSize;   // 32
constexpr uint32_t kPrpListBytes    = (uint32_t)kPageSize;       // 4 KiB
constexpr uint32_t kPrpListEntries  = kPrpListBytes / sizeof(uint64_t);
constexpr const char* kGpuFileName  = "gpu_bw_blk_0";
constexpr const char* kPrefix       = "gpu_bw_";

// ---------------------------------------------------------------------------
// Per-IO context.  `dh` is GPU-resident (block_storage acquired),
// `prp1` is the IOVA of data_page[0] within the shard's data buffer,
// `prp2` is the IOVA of the PRP-list page that holds
// data_page[1..N-1] IOVAs.
// ---------------------------------------------------------------------------
struct R6BwIoCtx {
    const tutti::NvmeFileDeviceHandle* dh;
    uint64_t                           prp1;
    uint64_t                           prp2;
    uint64_t                           shard_off;
    uint64_t                           nbytes;
};

__global__ void bw_read_kernel(const R6BwIoCtx* ctx, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const auto& c = ctx[tid];
    tutti::submit_read_one(c.dh, c.prp1, c.prp2, c.shard_off, c.nbytes);
}

// Dummy compute kernel: each thread runs `iters` FMAs on a single
// register.  Output is written to `buf[tid]` so the optimizer can't
// elide it.  Time scales linearly with `iters`; tune --compute-iters
// to roughly match expected IO time so overlap is visible.
__global__ void compute_kernel(float* __restrict__ buf,
                               uint32_t           n,
                               uint32_t           iters)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float x = buf[tid] + 1.0f;
    for (uint32_t i = 0; i < iters; ++i) {
        x = x * 1.000001f + 0.000001f;
    }
    buf[tid] = x;
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

}  // namespace

int main(int argc, char** argv) {
    int gpu_dev          = 0;
    int cuda_dev         = 0;
    uint32_t cap         = 32;
    uint32_t num_queues  = 16;
    double   total_gib   = 8.0;
    uint32_t compute_iters = 5000000;
    std::vector<std::string> pci_addrs;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if      (std::strcmp(a, "--gpu")    == 0 && i + 1 < argc) gpu_dev    = std::atoi(argv[++i]);
        else if (std::strcmp(a, "--cuda")   == 0 && i + 1 < argc) cuda_dev   = std::atoi(argv[++i]);
        else if (std::strcmp(a, "--cap")    == 0 && i + 1 < argc) cap        = (uint32_t)std::atoi(argv[++i]);
        else if (std::strcmp(a, "--queues") == 0 && i + 1 < argc) num_queues = (uint32_t)std::atoi(argv[++i]);
        else if (std::strcmp(a, "--total-gib") == 0 && i + 1 < argc) total_gib = std::atof(argv[++i]);
        else if (std::strcmp(a, "--compute-iters") == 0 && i + 1 < argc) compute_iters = (uint32_t)std::atoi(argv[++i]);
        else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) { usage(argv[0]); return 0; }
        else if (a[0] != '-') pci_addrs.emplace_back(a);
        else { std::fprintf(stderr, "unknown arg: %s\n", a); usage(argv[0]); return 1; }
    }
    if (pci_addrs.size() != kNumShards) {
        std::fprintf(stderr,
            "need exactly %u PCI BDFs (got %zu)\n",
            kNumShards, pci_addrs.size());
        usage(argv[0]); return 1;
    }
    if (total_gib <= 0.0) {
        std::fprintf(stderr, "--total-gib must be > 0\n");
        return 1;
    }

    // Derived sizing.  Round total down so it divides into integer
    // tensors evenly across all shards.
    const uint64_t total_bytes_req = (uint64_t)(total_gib * (1ull << 30));
    const uint32_t num_tensors_total =
        (uint32_t)((total_bytes_req / kTensorSize / kNumShards) * kNumShards);
    if (num_tensors_total == 0) {
        std::fprintf(stderr, "computed num_tensors_total == 0; bump --total-gib\n");
        return 1;
    }
    const uint64_t total_bytes  = (uint64_t)num_tensors_total * kTensorSize;
    const uint32_t tensors_per_shard = num_tensors_total / kNumShards;
    const uint64_t per_shard_data    = (uint64_t)tensors_per_shard * kTensorSize;
    const uint64_t per_shard_prps    = (uint64_t)tensors_per_shard * kPrpListBytes;

    // [1]
    prime_cuda(cuda_dev);
    (void)gpu_dev;

    // Outermost NVTX range so the whole demo is one strip in the
    // timeline; sub-ranges below mark each phase.
    nvtxRangePushA("block_storage_gpu_bw_smoke");

    // [2]
    nvtxRangePushA("setup:registry+bootstrap");
    std::vector<tutti::LocalNvmeDirectConfig> cfgs;
    cfgs.reserve(pci_addrs.size());
    for (const auto& bdf : pci_addrs) {
        tutti::LocalNvmeDirectConfig c{};
        c.pci_addr           = bdf;
        c.kernel_ioq_cap     = cap;
        c.build_queue_group  = true;
        c.cuda_device        = cuda_dev;
        c.num_user_queues    = num_queues;
        cfgs.push_back(std::move(c));
    }
    tutti::LocalNvmeDirectRegistry registry(std::move(cfgs));
    if (!registry.Open()) STEP_FAIL("registry.Open");
    STEP_OK("registry up: n=%zu (build_queue_group=true, q/dev=%u, cap=%u)",
            pci_addrs.size(), num_queues, cap);

    const std::size_t N_dev = registry.device_count();
    std::vector<const tutti::Device*> devices;
    devices.reserve(N_dev);
    for (std::size_t i = 0; i < N_dev; ++i) {
        const tutti::Device* d = registry.device_at(i);
        if (d == nullptr) STEP_FAIL("device_at(%zu) is null", i);
        devices.push_back(d);
    }

    // [3] [4]
    auto storage = std::make_unique<tutti::HostFsBackedNvmeStorage>();
    if (!storage->bootstrap(devices)) STEP_FAIL("storage.bootstrap");
    STEP_OK("nvme_storage bootstrap");

    auto bs = std::make_unique<tutti::HostFsBackedBlockStorage>();
    if (!bs->bootstrap(storage.get(), devices)) STEP_FAIL("block_storage.bootstrap");
    STEP_OK("block_storage bootstrap");

    // [5]
    wipe_stragglers(*bs);
    STEP_OK("pre-cleanup");
    nvtxRangePop();   // setup:registry+bootstrap

    // [6]
    nvtxRangePushA("alloc:create_gpu_file+cudaMalloc+nvm_dma_map");
    tutti::GpuFileSpec spec{};
    spec.name              = kGpuFileName;
    spec.total_size        = total_bytes;
    spec.tensor_shape[0]   = kNumShards;
    spec.tensor_shape[1]   = tensors_per_shard;          // metadata only
    spec.tensor_shape[2]   = kTensorSize;
    spec.shard_placement   = { devices[0], devices[1], devices[2], devices[3] };
    tutti::GpuFile* gf = bs->create_gpu_file(spec, /*persist_now=*/true);
    if (gf == nullptr) STEP_FAIL("create_gpu_file '%s'", kGpuFileName);
    STEP_OK("create_gpu_file '%s' total=%.2f GiB shards=%u tensor=%u "
            "(tensors_total=%u tensors/shard=%u)",
            kGpuFileName, total_bytes / (double)(1ull << 30),
            kNumShards, kTensorSize, num_tensors_total, tensors_per_shard);

    // [7] [8] per-shard data + prp_list buffers
    struct ShardBuf {
        void*       data_devptr      = nullptr;
        nvm_dma_t*  data_dma         = nullptr;          // owned
        void*       prp_list_devptr  = nullptr;
        nvm_dma_t*  prp_list_dma     = nullptr;          // owned
    };
    std::vector<ShardBuf> bufs(kNumShards);
    for (uint32_t s = 0; s < kNumShards; ++s) {
        const tutti::Device* dev = devices[s];
        auto* lnd = static_cast<tutti::LocalNvmeDevice*>(dev->backend_private);
        if (lnd == nullptr || lnd->ctrl == nullptr)
            STEP_FAIL("shard %u: no LocalNvmeDevice/ctrl", s);

        // -- data buffer
        CUDA_OK(cudaMalloc(&bufs[s].data_devptr, per_shard_data));
        int rc = nvm_dma_map_data_device(&bufs[s].data_dma, lnd->ctrl,
                                         bufs[s].data_devptr, per_shard_data);
        if (rc != 0 || bufs[s].data_dma == nullptr)
            STEP_FAIL("nvm_dma_map_data_device(shard=%u, data=%llu) rc=%d",
                      s, (unsigned long long)per_shard_data, rc);
        if (bufs[s].data_dma->page_size != kPageSize)
            STEP_FAIL("shard %u data dma page_size=%zu != %llu",
                      s, (size_t)bufs[s].data_dma->page_size,
                      (unsigned long long)kPageSize);
        if (bufs[s].data_dma->n_ioaddrs * (size_t)bufs[s].data_dma->page_size <
            (size_t)per_shard_data)
            STEP_FAIL("shard %u data dma covers %zu B < want %llu B",
                      s,
                      bufs[s].data_dma->n_ioaddrs * (size_t)bufs[s].data_dma->page_size,
                      (unsigned long long)per_shard_data);

        // -- prp_list buffer
        CUDA_OK(cudaMalloc(&bufs[s].prp_list_devptr, per_shard_prps));
        rc = nvm_dma_map_data_device(&bufs[s].prp_list_dma, lnd->ctrl,
                                     bufs[s].prp_list_devptr, per_shard_prps);
        if (rc != 0 || bufs[s].prp_list_dma == nullptr)
            STEP_FAIL("nvm_dma_map_data_device(shard=%u, prp_list=%llu) rc=%d",
                      s, (unsigned long long)per_shard_prps, rc);
        if (bufs[s].prp_list_dma->page_size != kPageSize)
            STEP_FAIL("shard %u prp_list dma page_size=%zu != %llu",
                      s, (size_t)bufs[s].prp_list_dma->page_size,
                      (unsigned long long)kPageSize);
    }
    STEP_OK("per-shard cudaMalloc + nvm_dma_map: data=%llu B prp_list=%llu B "
            "(per shard); ioaddr[s0][0]=0x%llx",
            (unsigned long long)per_shard_data,
            (unsigned long long)per_shard_prps,
            (unsigned long long)bufs[0].data_dma->ioaddrs[0]);

    // Build PRP-list pages on host (R7 prototype).  For each IO i in
    // a shard, page i in prp_list_buf packs IOVAs of data_page[1..31]
    // for that IO's tensor.  prp_list_dma->ioaddrs[i] is what the
    // controller dereferences for prp2.
    {
        std::vector<uint64_t> page_buf(kPrpListEntries, 0);
        for (uint32_t s = 0; s < kNumShards; ++s) {
            // Allocate the whole shard's prp_list area as one host blob
            // and cudaMemcpy once.
            std::vector<uint8_t> all((size_t)tensors_per_shard * kPrpListBytes, 0);
            for (uint32_t io = 0; io < tensors_per_shard; ++io) {
                uint64_t* page = (uint64_t*)(all.data() + (size_t)io * kPrpListBytes);
                const uint32_t base_data_page = io * kPagesPerTensor;
                // PRP list entries are little-endian uint64_t (NVMe spec).
                // x86 is LE, so direct assignment is correct.
                for (uint32_t p = 1; p < kPagesPerTensor; ++p) {
                    page[p - 1] = bufs[s].data_dma->ioaddrs[base_data_page + p];
                }
                // Remaining entries left at 0 (unused; controller stops at N-1).
            }
            CUDA_OK(cudaMemcpy(bufs[s].prp_list_devptr, all.data(),
                               (size_t)tensors_per_shard * kPrpListBytes,
                               cudaMemcpyHostToDevice));
        }
    }
    STEP_OK("packed + uploaded PRP-list pages (%u/shard, %u entries each)",
            tensors_per_shard, kPagesPerTensor - 1);

    // [9]
    tutti::GpuFileHandle* h = bs->acquire_device_handle(gf);
    if (h == nullptr) STEP_FAIL("acquire_device_handle");
    if (h->num_shards != kNumShards) STEP_FAIL("handle->num_shards");
    STEP_OK("acquire_device_handle: %u shard handle(s)", h->num_shards);
    nvtxRangePop();   // alloc:create_gpu_file+cudaMalloc+nvm_dma_map

    // [10] host-stage IO contexts.  ext4 caps a single extent at
    nvtxRangePushA("stage:host_build_ctx+upload");
    // 32768 4 KiB blocks (= 128 MiB), so any file > 128 MiB has at
    // least one logical-extent boundary; an NVMe IO that crosses
    // such a boundary makes submit_read_one's resolve_lba return
    // false (R5b deliberately handles only single-extent IOs --
    // cross-extent splitting belongs in a later layer).  This
    // smoke is bandwidth-focused, so we filter out the few IOs
    // that would straddle a boundary; the bandwidth numbers below
    // are computed against the actually-submitted byte count.

    // (a) Compute per-shard logical-extent boundary array.  Element
    //     i is the cumulative end (in payload bytes, header-stripped)
    //     of logical extent i.  Both endpoints of an IO must fall
    //     in the same half-open interval [bnd[i-1], bnd[i]) to be
    //     submittable.
    constexpr uint64_t kNvmeBlockSize = 4096;          // matches our drives
    constexpr uint64_t kFileHeaderBytes = 4096;        // NvmeFileHeader
    std::vector<std::vector<uint64_t>> shard_ext_boundaries(kNumShards);
    for (uint32_t s = 0; s < kNumShards; ++s) {
        const auto& exts = gf->shards[s]->extents;
        if (exts.empty()) STEP_FAIL("shard %u has no extents", s);
        uint64_t cum = 0;
        for (std::size_t i = 0; i < exts.size(); ++i) {
            uint64_t bytes = exts[i].length_blocks * kNvmeBlockSize;
            if (i == 0) {
                if (bytes <= kFileHeaderBytes)
                    STEP_FAIL("shard %u extent[0] is shorter than file header",
                              s);
                bytes -= kFileHeaderBytes;
            }
            cum += bytes;
            shard_ext_boundaries[s].push_back(cum);
        }
    }

    // (b) Stage ctx, skipping any IO that crosses an extent boundary.
    std::vector<R6BwIoCtx> h_ctx;
    h_ctx.reserve(num_tensors_total);
    uint32_t skipped = 0;
    for (uint32_t t = 0; t < num_tensors_total; ++t) {
        const uint64_t global_off = (uint64_t)t * kTensorSize;
        uint32_t shard_idx     = 0;
        uint64_t shard_byte_off = 0;
        tutti::gpu_file_resolve(kTensorSize, kNumShards, global_off,
                                &shard_idx, &shard_byte_off);
        const uint64_t end_off = shard_byte_off + kTensorSize - 1;
        const auto& bnd = shard_ext_boundaries[shard_idx];
        // First boundary strictly greater than each endpoint == the
        // logical extent that endpoint belongs to.  If they differ,
        // the IO would straddle a boundary -> drop.
        auto it_start = std::upper_bound(bnd.begin(), bnd.end(), shard_byte_off);
        auto it_end   = std::upper_bound(bnd.begin(), bnd.end(), end_off);
        if (it_start != it_end) {
            ++skipped;
            continue;
        }
        const uint32_t io_in_shard    = (uint32_t)(shard_byte_off / kTensorSize);
        const uint32_t base_data_page = io_in_shard * kPagesPerTensor;
        R6BwIoCtx c{};
        c.dh        = h->d_shards_host[shard_idx];
        c.prp1      = bufs[shard_idx].data_dma->ioaddrs[base_data_page];
        c.prp2      = bufs[shard_idx].prp_list_dma->ioaddrs[io_in_shard];
        c.shard_off = shard_byte_off;
        c.nbytes    = kTensorSize;
        h_ctx.push_back(c);
    }
    const uint32_t num_ios_actual = (uint32_t)h_ctx.size();
    const uint64_t total_bytes_actual =
        (uint64_t)num_ios_actual * kTensorSize;
    R6BwIoCtx* d_ctx = nullptr;
    CUDA_OK(cudaMalloc(&d_ctx, sizeof(R6BwIoCtx) * num_ios_actual));
    CUDA_OK(cudaMemcpy(d_ctx, h_ctx.data(),
                       sizeof(R6BwIoCtx) * num_ios_actual,
                       cudaMemcpyHostToDevice));
    STEP_OK("staged %u IO context(s) (host -> GPU); skipped %u "
            "cross-extent IO(s) [%.3f%% of %u]",
            num_ios_actual, skipped,
            100.0 * skipped / (double)num_tensors_total,
            num_tensors_total);
    nvtxRangePop();   // stage:host_build_ctx+upload

    // [11] warmup
    constexpr uint32_t kBlockDim = 64;
    const     uint32_t kGridDim  = (num_ios_actual + kBlockDim - 1) / kBlockDim;
    {
        NvtxRange _r("warmup_io_kernel", kClrSetup);
        bw_read_kernel<<<kGridDim, kBlockDim>>>(d_ctx, num_ios_actual);
        CUDA_OK(cudaDeviceSynchronize());
    }
    STEP_OK("warmup IO kernel grid=%u block=%u", kGridDim, kBlockDim);

    // [12] timed IO bandwidth
    cudaEvent_t e_start, e_end;
    CUDA_OK(cudaEventCreate(&e_start));
    CUDA_OK(cudaEventCreate(&e_end));

    float io_ms = 0.0f;
    {
        NvtxRange _r("phase:IO_only (8 GiB read)", kClrIO);
        CUDA_OK(cudaEventRecord(e_start));
        bw_read_kernel<<<kGridDim, kBlockDim>>>(d_ctx, num_ios_actual);
        CUDA_OK(cudaEventRecord(e_end));
        CUDA_OK(cudaEventSynchronize(e_end));
        CUDA_OK(cudaEventElapsedTime(&io_ms, e_start, e_end));
    }
    const double io_gibps = (double)total_bytes_actual / (1ull << 30) / (io_ms / 1000.0);
    const double io_gbps  = (double)total_bytes_actual / 1e9          / (io_ms / 1000.0);
    STEP_OK("IO bandwidth: %.2f GB/s (%.2f GiB/s) -- %.2f ms for %.2f GiB across %u shards",
            io_gbps, io_gibps, io_ms,
            total_bytes_actual / (double)(1ull << 30), kNumShards);

    // [13] timed compute baseline (independent buffer)
    constexpr uint32_t kComputeThreads = 65536;
    constexpr uint32_t kComputeBlockDim = 256;
    constexpr uint32_t kComputeGridDim  = kComputeThreads / kComputeBlockDim;
    float* d_compute = nullptr;
    CUDA_OK(cudaMalloc(&d_compute, kComputeThreads * sizeof(float)));
    CUDA_OK(cudaMemset(d_compute, 0, kComputeThreads * sizeof(float)));

    // warmup compute
    {
        NvtxRange _r("warmup_compute_kernel", kClrSetup);
        compute_kernel<<<kComputeGridDim, kComputeBlockDim>>>(d_compute, kComputeThreads, compute_iters);
        CUDA_OK(cudaDeviceSynchronize());
    }

    float compute_ms = 0.0f;
    {
        NvtxRange _r("phase:compute_only", kClrCompute);
        CUDA_OK(cudaEventRecord(e_start));
        compute_kernel<<<kComputeGridDim, kComputeBlockDim>>>(d_compute, kComputeThreads, compute_iters);
        CUDA_OK(cudaEventRecord(e_end));
        CUDA_OK(cudaEventSynchronize(e_end));
        CUDA_OK(cudaEventElapsedTime(&compute_ms, e_start, e_end));
    }
    STEP_OK("compute baseline: %.2f ms (iters=%u, threads=%u)",
            compute_ms, compute_iters, kComputeThreads);

    // [14] serial: IO -> compute, both on default stream
    float serial_ms = 0.0f;
    {
        NvtxRange _r("phase:serial(IO+compute)", kClrCompute);
        CUDA_OK(cudaEventRecord(e_start));
        bw_read_kernel<<<kGridDim, kBlockDim>>>(d_ctx, num_ios_actual);
        compute_kernel<<<kComputeGridDim, kComputeBlockDim>>>(d_compute, kComputeThreads, compute_iters);
        CUDA_OK(cudaEventRecord(e_end));
        CUDA_OK(cudaEventSynchronize(e_end));
        CUDA_OK(cudaEventElapsedTime(&serial_ms, e_start, e_end));
    }
    STEP_OK("serial(IO + compute) on default stream: %.2f ms", serial_ms);

    // [15] overlap: IO on stream_io, compute on stream_compute
    cudaStream_t s_io = nullptr, s_cmp = nullptr;
    CUDA_OK(cudaStreamCreate(&s_io));
    CUDA_OK(cudaStreamCreate(&s_cmp));

    float overlap_ms = 0.0f;
    {
        NvtxRange _r("phase:overlap(IO||compute) on 2 streams", kClrOverlap);
        CUDA_OK(cudaEventRecord(e_start));
        {
            NvtxRange _r1("stream:IO_kernel", kClrIO);
            bw_read_kernel<<<kGridDim, kBlockDim, 0, s_io>>>(d_ctx, num_ios_actual);
        }
        {
            NvtxRange _r2("stream:compute_kernel", kClrCompute);
            compute_kernel<<<kComputeGridDim, kComputeBlockDim, 0, s_cmp>>>(
                d_compute, kComputeThreads, compute_iters);
        }
        CUDA_OK(cudaStreamSynchronize(s_io));
        CUDA_OK(cudaStreamSynchronize(s_cmp));
        CUDA_OK(cudaEventRecord(e_end));
        CUDA_OK(cudaEventSynchronize(e_end));
        CUDA_OK(cudaEventElapsedTime(&overlap_ms, e_start, e_end));
    }
    const double speedup = (double)serial_ms / (double)overlap_ms;
    const double max_kernel_ms = (io_ms > compute_ms) ? io_ms : compute_ms;
    const double overhead_pct =
        ((double)overlap_ms - max_kernel_ms) / max_kernel_ms * 100.0;
    STEP_OK("overlap(IO || compute) on 2 streams: %.2f ms "
            "(speedup vs serial = %.2fx, overhead vs max(IO,compute) = %+.1f%%)",
            overlap_ms, speedup, overhead_pct);

    // [16] release
    nvtxRangePushA("teardown");
    if (d_compute) cudaFree(d_compute);
    if (d_ctx)     cudaFree(d_ctx);
    cudaStreamDestroy(s_io);
    cudaStreamDestroy(s_cmp);
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_end);
    bs->release_device_handle(h); h = nullptr;
    for (uint32_t s = 0; s < kNumShards; ++s) {
        if (bufs[s].prp_list_dma) { nvm_dma_unmap(bufs[s].prp_list_dma); bufs[s].prp_list_dma = nullptr; }
        if (bufs[s].prp_list_devptr) { cudaFree(bufs[s].prp_list_devptr); bufs[s].prp_list_devptr = nullptr; }
        if (bufs[s].data_dma) { nvm_dma_unmap(bufs[s].data_dma); bufs[s].data_dma = nullptr; }
        if (bufs[s].data_devptr) { cudaFree(bufs[s].data_devptr); bufs[s].data_devptr = nullptr; }
    }
    STEP_OK("release_device_handle + nvm_dma_unmap + cudaFree");

    // [17]
    if (!bs->delete_gpu_file(gf, /*persist_now=*/true))
        STEP_FAIL("delete_gpu_file");
    STEP_OK("delete_gpu_file '%s'", kGpuFileName);

    // [18]
    if (!bs->shutdown())      STEP_FAIL("block_storage.shutdown");
    if (!storage->shutdown()) STEP_FAIL("nvme_storage.shutdown");
    STEP_OK("shutdown both layers");

    // [19]
    registry.Close();
    STEP_OK("registry closed");
    nvtxRangePop();   // teardown
    nvtxRangePop();   // block_storage_gpu_bw_smoke

    std::fprintf(stderr,
        "\n=== block_storage_gpu_bw_smoke (n_dev=%zu, ts=%u, ios=%u, "
        "skipped=%u, total=%.2f GiB): all %d steps passed ===\n"
        "    IO bandwidth   : %.2f GB/s (%.2f ms)\n"
        "    compute        : %.2f ms (iters=%u)\n"
        "    serial         : %.2f ms\n"
        "    overlap        : %.2f ms (speedup=%.2fx)\n",
        N_dev, kTensorSize, num_ios_actual, skipped,
        total_bytes_actual / (double)(1ull << 30), g_step,
        io_gbps, io_ms, compute_ms, compute_iters,
        serial_ms, overlap_ms, speedup);
    return 0;
}

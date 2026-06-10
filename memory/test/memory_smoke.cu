/**
 * memory_smoke.cu -- exercise HostDeviceMemorySubsystem end-to-end
 * over one OR multiple NVMe controllers.
 *
 * Single-process: brings up libnvm via LocalNvmeDirectRegistry (which
 * is what real R5+ callers will do).
 *
 * Single-NVMe invocation:
 *   sudo ./memory_smoke --gpu 0 --cap 32 0000:08:00.0
 *
 * Multi-NVMe invocation (every requested ctrl is bound via
 * mem.bind_devices(), exercising the cluster-wide DMA mapping
 * path):
 *   sudo ./memory_smoke --gpu 0 --cap 32 \
 *        0000:4b:00.0 0000:57:00.0 0000:63:00.0
 *
 * The smoke validates:
 *   - cuda driver prime + cudaSetDevice
 *   - registry brings up N controllers
 *   - HostDeviceMemorySubsystem ctor is parameter-less
 *   - bind_devices(N) caches cluster-wide caps (min MDTS, page_size)
 *   - register_tensor() implicitly DMA-maps to every bound ctrl
 *   - per-(region, Device*) DMA tracking + idempotent re-call
 *   - set_descriptor_format() acceptance
 *   - descriptor_slice() returns false (R7 stub)
 *   - R7: granularity > 0 builds ONE cluster-wide IO-slice table
 *         + ONE INTERNAL PRP-list buffer (auto cudaMalloc + per-ctrl
 *         nvm_dma_map_data_device); lookup_io_slice O(log N) hits
 *   - free / unregister cleanly tears down all DMA maps + the
 *     IO-slice table + the internal PRP-list buffer
 *
 * Test contract:
 *   - sole owner of every NVMe (NVMeService daemon MUST NOT be running)
 *   - CUDA-visible GPU at --gpu N
 *
 * NOT destructive: no LBA writes; only DMA mapping.
 */

#include "host_device_memory_subsystem.h"
#include "cuda_helpers.cuh"

#include "../../device_manager/include/local_nvme_direct_registry.h"
#include "../../runtime/include/device.h"

#include <cuda_runtime.h>

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
        "Usage: %s [--gpu N] [--cap N] <PCI_BDF> [<PCI_BDF>...]\n"
        "  e.g. (single):  %s --gpu 0 --cap 32 0000:08:00.0\n"
        "  e.g. (multi):   %s --gpu 0 --cap 32 0000:4b:00.0 0000:57:00.0\n"
        "Exercises tutti::HostDeviceMemorySubsystem against one or\n"
        "multiple libnvm controllers via LocalNvmeDirectRegistry.\n"
        "Daemon MUST NOT be running.  Non-destructive (no LBA writes).\n",
        prog, prog, prog);
}

void prime_cuda(int cuda_dev) {
    cudaError_t cerr = cudaFree(0);
    if (cerr != cudaSuccess && cerr != cudaErrorInvalidValue) {
        STEP_FAIL("cuda driver prime (cudaFree(0)) failed: %s",
                  cudaGetErrorString(cerr));
    }
    (void)cudaGetLastError();

    CUDA_OK(cudaSetDevice(cuda_dev));
    STEP_OK("cudaSetDevice(%d)", cuda_dev);
}

int run(int cuda_dev, const std::vector<std::string>& pci_addrs, uint32_t cap) {
    prime_cuda(cuda_dev);

    // [1] Bring up every requested controller via the direct registry.
    std::vector<tutti::LocalNvmeDirectConfig> cfgs;
    cfgs.reserve(pci_addrs.size());
    for (const auto& bdf : pci_addrs) {
        cfgs.push_back({bdf, cap, /*display_name=*/{}});
    }

    tutti::LocalNvmeDirectRegistry reg(std::move(cfgs));
    if (!reg.Open()) STEP_FAIL("LocalNvmeDirectRegistry::Open() (n=%zu)",
                                pci_addrs.size());
    if (reg.device_count() != pci_addrs.size())
        STEP_FAIL("device_count=%zu != requested=%zu",
                  reg.device_count(), pci_addrs.size());

    std::vector<const tutti::Device*> devices;
    devices.reserve(pci_addrs.size());
    for (size_t i = 0; i < pci_addrs.size(); ++i) {
        devices.push_back(reg.device_at(i));
    }
    STEP_OK("registry up: n=%zu", devices.size());

    // [2] Build the memory subsystem (no ctrl in ctor).
    tutti::HostDeviceMemorySubsystem mem;
    mem.set_descriptor_format(tutti::DescriptorFormat::PRP);
    if (mem.descriptor_format() != tutti::DescriptorFormat::PRP)
        STEP_FAIL("descriptor_format() != PRP after set");
    STEP_OK("HostDeviceMemorySubsystem instantiated, format=PRP");

    // [3] bind cluster devices.  Caches min MDTS + page_size; every
    //     subsequent register_tensor implicitly DMA-maps to all
    //     bound devices (no per-region target_devices subset).
    mem.bind_devices(devices);
    STEP_OK("bind_devices(n=%zu) -- cluster-wide caps cached", devices.size());

    // [4] allocate HOST
    auto* r_host = mem.allocate_host(64 * 1024, tutti::MemoryKind::HOST);
    if (r_host == nullptr) STEP_FAIL("allocate_host(HOST)");
    STEP_OK("allocate_host(HOST) id=%lu host=%p size=%lu",
            (unsigned long)r_host->region_id, r_host->host_ptr,
            (unsigned long)r_host->size);

    // [5] allocate PINNED_HOST
    auto* r_pin = mem.allocate_host(4096, tutti::MemoryKind::PINNED_HOST);
    if (r_pin == nullptr) STEP_FAIL("allocate_host(PINNED_HOST)");
    STEP_OK("allocate_host(PINNED_HOST) id=%lu host=%p",
            (unsigned long)r_pin->region_id, r_pin->host_ptr);

    // [6] allocate DEVICE
    constexpr std::size_t kDeviceSize = 1ull << 20;     // 1 MiB
    auto* r_dev = mem.allocate_device(kDeviceSize, tutti::MemoryKind::DEVICE,
                                       cuda_dev);
    if (r_dev == nullptr) STEP_FAIL("allocate_device(DEVICE)");
    STEP_OK("allocate_device(DEVICE) id=%lu dev=%p size=1MiB on cuda=%d",
            (unsigned long)r_dev->region_id, r_dev->device_ptr,
            r_dev->cuda_device);

    // [7] register_tensor on the device region.  bind_devices() at
    //     step [3] told the subsystem about the whole cluster, so
    //     this register_tensor implicitly DMA-maps to all N
    //     controllers.  No per-spec target_devices subset.
    //     cluster-wide invariant says ioaddrs[] are numerically
    //     identical across ctrls, so iterating + reporting the
    //     per-ctrl handle adds no information.  query_nvme_mapping
    //     is exercised below in the R7 GPU AddressDescriptor sanity
    //     check (step (e)) on the ref device.
    {
        tutti::TensorRegistrationSpec spec{};
        spec.ptr             = r_dev->device_ptr;
        spec.size            = r_dev->size;
        auto* same = mem.register_tensor(spec);
        if (same != r_dev) STEP_FAIL("register_tensor: returned region != r_dev "
                                      "(got=%p, want=%p)", (void*)same, (void*)r_dev);
    }
    STEP_OK("register_tensor(device) DMA-mapped to %zu device(s)",
            devices.size());

    // [8] register_tensor idempotency: calling again with the same set
    //     should be a no-op for already-mapped (region, device) pairs.
    {
        tutti::TensorRegistrationSpec spec{};
        spec.ptr             = r_dev->device_ptr;
        spec.size            = r_dev->size;
        if (mem.register_tensor(spec) != r_dev)
            STEP_FAIL("register_tensor idempotent re-call");
    }
    STEP_OK("register_tensor(device) idempotent re-call ok across %zu devices",
            devices.size());

    // [9] register_external(APP_MANAGED) on a caller-cudaMalloc'd buffer,
    //     then register_tensor against ALL devices to drive DMA mapping.
    void* app_buf = nullptr;
    CUDA_OK(cudaMalloc(&app_buf, 256 * 1024));
    tutti::ExternalMemorySpec espec{};
    espec.source = tutti::ExternalMemorySource::APP_MANAGED;
    auto* r_ext = mem.register_external(/*host=*/nullptr, app_buf,
                                          256 * 1024, espec);
    if (r_ext == nullptr) STEP_FAIL("register_external(APP_MANAGED)");
    {
        tutti::TensorRegistrationSpec spec{};
        spec.ptr             = app_buf;
        spec.size            = 256 * 1024;
        if (mem.register_tensor(spec) != r_ext)
            STEP_FAIL("register_tensor(external) returned wrong region");
    }
    STEP_OK("register_external(APP_MANAGED) id=%lu dev=%p mapped to %zu devices",
            (unsigned long)r_ext->region_id, app_buf, devices.size());

    // [10] descriptor_slice is a v0.1 stub.
    {
        tutti::AddressDescriptor d{};
        std::size_t cnt = 1;
        bool ok = mem.descriptor_slice(r_dev, devices[0], 0, 4096, &d, &cnt);
        if (ok) STEP_FAIL("descriptor_slice: expected stub-false, got true");
    }
    STEP_OK("descriptor_slice() returns Unimplemented (R7 stub)");

    // -- R7 ---------------------------------------------------------
    //
    // register_tensor with granularity > 0 builds a per-device
    // (slice_addr -> AddressDescriptor[]) table on the GPU.  We
    // exercise both fast paths (no PRP list when effective_io <= 8
    // KiB) and slow path (PRP list required when one IO spans more
    // than 2 NVMe pages).
    //
    // Layout: r_dev is 1 MiB on GPU, granularity = 128 KiB ==
    // typical MDTS, so each granule is exactly one IO of 128 KiB
    // (32 pages -> needs PRP list).  num_slices = 8.
    constexpr std::size_t kGranularity   = 128 * 1024;       // 128 KiB
    constexpr std::size_t kPageSize      = 4096;
    constexpr std::size_t kPagesPerIO    = kGranularity / kPageSize;  // 32
    const     std::size_t kNumGranules   = kDeviceSize / kGranularity; // 8

    // (a) register_tensor on r_dev with granularity = 128 KiB.
    //     Cluster-wide build: bind_devices() at step [3] told the
    //     subsystem about all N ctrls, so this call automatically
    //     DMA-maps to all of them and builds ONE IO-slice table +
    //     ONE INTERNAL PRP-list buffer.  Caller does NOT supply
    //     the PRP-list buffer (matches legacy
    //     GPUController::initializePRPList).
    {
        tutti::TensorRegistrationSpec ds{};
        ds.ptr              = r_dev->device_ptr;
        ds.size             = kDeviceSize;
        ds.granularity      = kGranularity;
        if (mem.register_tensor(ds) != r_dev)
            STEP_FAIL("register_tensor(r_dev, granularity=%zu)",
                      kGranularity);
    }
    STEP_OK("R7: register_tensor(r_dev, granularity=%zu KiB) "
            "built ONE cluster-wide IO-slice table + INTERNAL "
            "PRP-list buffer (auto-mapped to %zu device(s))",
            kGranularity / 1024, devices.size());

    // (a2) Idempotent re-register with granularity must be a no-op
    //      (legacy registerTesnsorList semantics: skip the rebuild
    //      path; caller must unregister to change granularity).
    //      We verify by capturing the d_ios pointer of slice 0
    //      across calls -- it must stay identical.
    {
        const tutti::IoSliceView* v_before =
            mem.lookup_io_slice(r_dev,
                reinterpret_cast<std::uint64_t>(r_dev->device_ptr));
        if (v_before == nullptr) STEP_FAIL("lookup_io_slice[0] (before)");
        const void* d_ios_before = v_before->d_ios;

        tutti::TensorRegistrationSpec ds{};
        ds.ptr         = r_dev->device_ptr;
        ds.size        = kDeviceSize;
        ds.granularity = kGranularity;
        if (mem.register_tensor(ds) != r_dev)
            STEP_FAIL("register_tensor idempotent re-call w/ granularity");

        const tutti::IoSliceView* v_after =
            mem.lookup_io_slice(r_dev,
                reinterpret_cast<std::uint64_t>(r_dev->device_ptr));
        if (v_after == nullptr) STEP_FAIL("lookup_io_slice[0] (after)");
        if (v_after->d_ios != d_ios_before)
            STEP_FAIL("idempotent re-call rebuilt the table: d_ios changed "
                      "(before=%p after=%p)",
                      d_ios_before, (const void*)v_after->d_ios);
    }
    STEP_OK("R7: register_tensor idempotent re-call w/ granularity is no-op "
            "(d_ios pointer stable -- no double-allocation)");

    // (b) list_io_slices + invariants.
    {
        auto views = mem.list_io_slices(r_dev);
        if (views.size() != kNumGranules)
            STEP_FAIL("list_io_slices size=%zu (want %zu)",
                      views.size(), kNumGranules);
        const std::uint64_t base =
            reinterpret_cast<std::uint64_t>(r_dev->device_ptr);
        for (std::size_t g = 0; g < views.size(); ++g) {
            const auto& v = views[g];
            if (v.slice_addr != base + g * kGranularity)
                STEP_FAIL("views[%zu].slice_addr=0x%llx (want 0x%llx)",
                          g, (unsigned long long)v.slice_addr,
                          (unsigned long long)(base + g * kGranularity));
            if (v.num_ios != 1)
                STEP_FAIL("views[%zu].num_ios=%u (want 1; one IO per "
                          "granule when granularity == MDTS)",
                          g, v.num_ios);
            if (v.total_bytes != kGranularity)
                STEP_FAIL("views[%zu].total_bytes=%llu (want %zu)",
                          g, (unsigned long long)v.total_bytes, kGranularity);
            if (v.d_ios == nullptr)
                STEP_FAIL("views[%zu].d_ios == nullptr", g);
            cudaPointerAttributes attr{};
            cudaError_t cerr = cudaPointerGetAttributes(&attr, v.d_ios);
            if (cerr != cudaSuccess || attr.type != cudaMemoryTypeDevice)
                STEP_FAIL("views[%zu].d_ios not GPU memory (err=%s, "
                          "type=%d)",
                          g, cudaGetErrorString(cerr), (int)attr.type);
        }
    }
    STEP_OK("R7: list_io_slices x %zu, every view GPU-resident + ordered",
            kNumGranules);

    // (c) lookup_io_slice returns the same views (in-place).
    for (std::size_t g = 0; g < kNumGranules; ++g) {
        const std::uint64_t addr =
            reinterpret_cast<std::uint64_t>(r_dev->device_ptr)
            + g * kGranularity;
        const tutti::IoSliceView* v =
            mem.lookup_io_slice(r_dev, addr);
        if (v == nullptr) STEP_FAIL("lookup_io_slice[%zu] = nullptr", g);
        if (v->slice_addr != addr) STEP_FAIL("lookup_io_slice[%zu] slice_addr "
                                              "mismatch", g);
        if (v->num_ios != 1)    STEP_FAIL("lookup_io_slice[%zu] num_ios "
                                              "= %u", g, v->num_ios);
    }
    {
        // Out-of-range / non-aligned addresses must miss.
        const std::uint64_t base =
            reinterpret_cast<std::uint64_t>(r_dev->device_ptr);
        if (mem.lookup_io_slice(r_dev, base + 1) != nullptr)
            STEP_FAIL("lookup_io_slice unaligned must miss");
        if (mem.lookup_io_slice(r_dev,
                               base + kNumGranules * kGranularity) != nullptr)
            STEP_FAIL("lookup_io_slice past-end must miss");
    }
    STEP_OK("R7: lookup_io_slice O(log N) hits + misses out-of-range/non-aligned");

    // (d) GPU-side AddressDescriptor sanity.  Copy back the first
    //     entry of granule 0 and verify its prp1 matches the
    //     query_nvme_mapping first ioaddr (= data_dma->ioaddrs[0]).
    {
        const tutti::IoSliceView* v0 =
            mem.lookup_io_slice(r_dev,
                               reinterpret_cast<std::uint64_t>(r_dev->device_ptr));
        if (v0 == nullptr) STEP_FAIL("lookup_io_slice[0] = nullptr");

        tutti::AddressDescriptor host_desc{};
        cudaError_t cerr = cudaMemcpy(&host_desc, v0->d_ios,
                                      sizeof(host_desc),
                                      cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess)
            STEP_FAIL("cudaMemcpy(d_ios[0]) failed: %s",
                      cudaGetErrorString(cerr));

        std::size_t n = 0, ps_b = 0;
        std::uint64_t first_ioaddr = 0;
        if (!mem.query_nvme_mapping(r_dev, devices[0], &n, &ps_b, &first_ioaddr))
            STEP_FAIL("query_nvme_mapping(r_dev) failed");
        if (host_desc.prp1 != first_ioaddr)
            STEP_FAIL("granule[0].prp1=0x%llx != first_ioaddr=0x%llx",
                      (unsigned long long)host_desc.prp1,
                      (unsigned long long)first_ioaddr);
        if (host_desc.data_length != kGranularity)
            STEP_FAIL("granule[0].data_length=%llu (want %zu)",
                      (unsigned long long)host_desc.data_length, kGranularity);
        if (host_desc.prp2 == 0)
            STEP_FAIL("granule[0].prp2 == 0; expected PRP-list IOVA "
                      "(pages_per_io=%zu > 2)", kPagesPerIO);
    }
    STEP_OK("R7: GPU AddressDescriptor[0] prp1 == data_dma.ioaddrs[0], "
            "prp2 != 0 (PRP-list path active)");

    // (e) Reject paths.  Non page_size-aligned granularity must fail.
    {
        // Allocate a fresh buffer so we don't pollute r_dev's table
        // (which already has a slice table built).
        auto* r_dev2 = mem.allocate_device(64 * 1024,
                                           tutti::MemoryKind::DEVICE,
                                           cuda_dev);
        if (r_dev2 == nullptr) STEP_FAIL("allocate_device(reject test)");

        tutti::TensorRegistrationSpec rs{};
        rs.ptr              = r_dev2->device_ptr;
        rs.size             = 64 * 1024;
        rs.granularity      = 4097;     // not page_size-aligned
        if (mem.register_tensor(rs) != nullptr)
            STEP_FAIL("register_tensor with granularity=4097 must reject");

        // size = 60 KiB, granularity = 16 KiB -> not a multiple
        // (60 / 16 = 3.75).  Must reject.
        rs.size             = 60 * 1024;
        rs.granularity      = 16 * 1024;
        if (mem.register_tensor(rs) != nullptr)
            STEP_FAIL("register_tensor with size=%zu not a multiple of "
                      "granularity=%zu must reject",
                      (size_t)60 * 1024, (size_t)16 * 1024);
        mem.free(r_dev2);
    }
    STEP_OK("R7: register_tensor rejects bad alignment + non-divisible size");

    // -- /R7 --------------------------------------------------------

    // [11] lookup
    {
        tutti::MemoryLookupKey k{};
        k.by  = tutti::MemoryLookupKey::By::HOST_PTR;
        k.ptr = r_host->host_ptr;
        if (mem.lookup(k) != r_host) STEP_FAIL("lookup HOST_PTR");
        k.by  = tutti::MemoryLookupKey::By::DEVICE_PTR;
        k.ptr = r_dev->device_ptr;
        if (mem.lookup(k) != r_dev) STEP_FAIL("lookup DEVICE_PTR");
        k.by        = tutti::MemoryLookupKey::By::REGION_ID;
        k.region_id = r_ext->region_id;
        if (mem.lookup(k) != r_ext) STEP_FAIL("lookup REGION_ID");
    }
    STEP_OK("lookup HOST_PTR / DEVICE_PTR / REGION_ID ok (regions=%zu)",
            mem.region_count());

    // [12] release.  mem.free(r_dev) also drops the cluster-wide
    //      IO-slice table together with its INTERNAL PRP-list
    //      buffer (cudaMalloc'd by build_io_slice_table_locked at
    //      step (a) above) and per-ctrl DMA handles for that
    //      buffer.
    mem.unregister(r_ext);
    cudaFree(app_buf);
    mem.free(r_dev);
    mem.free(r_pin);
    mem.free(r_host);
    if (mem.region_count() != 0)
        STEP_FAIL("region_count != 0 after release: %zu", mem.region_count());
    STEP_OK("free / unregister cleared the table (drops 2 data DMA "
            "+ 1 cluster-wide IO-slice table + 1 PRP-list DMA + buffer)");

    // [13] tear down the registry (closes ctrls via nvm_ctrl_free).
    reg.Close();
    STEP_OK("registry closed (chrdev_remove + unbind, n=%zu)", devices.size());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int      cuda_dev = 0;
    uint32_t cap      = 32;
    int      argi     = 1;

    while (argi < argc && argv[argi][0] == '-' && argv[argi][1] == '-') {
        const char* a = argv[argi];
        if (std::strcmp(a, "--gpu") == 0 && argi + 1 < argc) {
            cuda_dev = std::atoi(argv[++argi]);
            ++argi;
        } else if (std::strcmp(a, "--cap") == 0 && argi + 1 < argc) {
            cap = (uint32_t)std::atoi(argv[++argi]);
            ++argi;
        } else { usage(argv[0]); return 1; }
    }
    if (argi >= argc) { usage(argv[0]); return 1; }

    std::vector<std::string> pci_addrs;
    while (argi < argc) {
        pci_addrs.emplace_back(argv[argi++]);
    }

    int rc = run(cuda_dev, pci_addrs, cap);
    if (rc == 0) {
        std::fprintf(stderr,
            "\n=== memory_smoke (n=%zu): all %d steps passed ===\n",
            pci_addrs.size(), g_step);
    }
    return rc;
}

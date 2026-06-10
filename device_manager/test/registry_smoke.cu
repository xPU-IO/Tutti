/**
 * registry_smoke.cu -- exercise both IDeviceRegistry implementations
 * over one OR multiple NVMe controllers.
 *
 *   --mode=direct   --pci=<BDF>[,<BDF>...]     LocalNvmeDirectRegistry
 *   --mode=service  --device=<id>[,<id>...]    NvmeServiceBackedRegistry
 *
 * For each mode:
 *
 *   [1] cuda driver prime + cudaSetDevice
 *   [2] open registry (multi-device input -> 1 Open() call)
 *   [3] verify device_count() matches the input count
 *   [4] enumerate via device_at() / find_by_id() / list()
 *       and sanity-check every LocalNvmeDevice payload
 *   [5] close registry; verify everything dropped via the right
 *       libnvm path (direct: nvm_ctrl_free; service: nvm_ctrl_free_client)
 *
 * "Direct" mode requires this process to be the sole owner of every
 * PCI device passed in -- run with the daemon stopped.
 * "Service" mode requires nvmeservice_daemon already running with
 * those devices in its sys_config.yaml.
 *
 * NOT destructive: no LBA writes; only chrdev / bind / probe.
 */

#include "local_nvme_direct_registry.h"
#include "nvmeservice_backed_registry.h"
#include "../../runtime/include/device.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        "Usage:\n"
        "  %s --mode=direct  --pci=<BDF>[,<BDF>...] [--gpu N] [--cap N]\n"
        "  %s --mode=service --endpoint=host:port --device=<id>[,<id>...]\n"
        "                                              [--cuda=N] [--count=N]\n"
        "\n"
        "  Multi-NVMe input is comma-separated; the registry brings up\n"
        "  every entry in one Open() call.\n"
        "\n"
        "direct mode:  brings up each NVMe via nvm_controller_init_b3.\n"
        "              Daemon MUST NOT be running; this process owns the chrdev.\n"
        "              CUDA must be initialised first (registry_smoke does\n"
        "              cudaSetDevice(--gpu) before calling Open()) because\n"
        "              nvm_controller_init_b3 internally cudaHostRegister()s\n"
        "              the BAR0 mapping.\n"
        "service mode: connects to running nvmeservice_daemon and Connects()\n"
        "              one session per --device given.  nvmeservice_daemon\n"
        "              must already be up and have all those devices in\n"
        "              its sys_config.yaml.\n"
        "Non-destructive (no LBA writes).\n",
        prog, prog);
}

// Parse "a,b,c" -> ["a","b","c"].  Empty entries are dropped; trailing
// or leading commas are tolerated.
std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    size_t b = 0;
    while (b <= s.size()) {
        size_t e = s.find(',', b);
        if (e == std::string::npos) e = s.size();
        if (e > b) out.emplace_back(s.substr(b, e - b));
        b = e + 1;
    }
    return out;
}

void check_device_shape(const tutti::Device* d,
                         const char* expect_pci_prefix,
                         tutti::LocalNvmeAttachMode expected_mode)
{
    if (d == nullptr) STEP_FAIL("Device* is null");
    if (d->backend_type != tutti::BackendType::LOCAL_NVME)
        STEP_FAIL("backend_type != LOCAL_NVME");
    if (d->backend_private == nullptr)
        STEP_FAIL("backend_private is null");
    if (expect_pci_prefix && d->pci_addr.find(expect_pci_prefix) == std::string::npos)
        STEP_FAIL("pci_addr '%s' missing prefix '%s'",
                  d->pci_addr.c_str(), expect_pci_prefix);

    auto* bp = static_cast<tutti::LocalNvmeDevice*>(d->backend_private);
    if (bp->ctrl == nullptr)        STEP_FAIL("LocalNvmeDevice.ctrl is null");
    if (bp->attach_mode != expected_mode)
        STEP_FAIL("attach_mode mismatch (got=%u, want=%u)",
                  (unsigned)bp->attach_mode, (unsigned)expected_mode);
    if (bp->blk_size == 0)         STEP_FAIL("blk_size == 0");
    if (bp->queue_depth == 0)      STEP_FAIL("queue_depth == 0");
    if (bp->page_size == 0)        STEP_FAIL("page_size == 0");
    // CTRL.MDTS sanity: must be reported (not 0) and at least one
    // NVMe block.  A typical H20 reports 128 KiB; older drives can
    // be as small as 4 KiB.  We refuse 0 because that would make
    // every downstream consumer (memory/, block_storage) fall back
    // to a conservative default and silently lose perf.
    if (bp->max_data_size == 0)
        STEP_FAIL("max_data_size (CTRL.MDTS) == 0; registry didn't "
                  "surface the controller's max single-IO size. Check "
                  "%s for a real-MDTS plumbing regression.",
                  expected_mode == tutti::LocalNvmeAttachMode::DIRECT
                      ? "local_nvme_direct_registry.cpp"
                      : "nvmeservice_backed_registry.cpp");
    if (bp->max_data_size < bp->blk_size)
        STEP_FAIL("max_data_size %zu < blk_size %u (impossible -- one "
                  "NVMe IO must be at least one block)",
                  bp->max_data_size, bp->blk_size);

    STEP_OK("device check: id=%d pci=%s snvme=%s mode=%s blk=%u qdepth=%u "
            "max_q/grp=%u mdts=%zu",
            d->device_id, d->pci_addr.c_str(),
            bp->snvme_dev_path.c_str(),
            expected_mode == tutti::LocalNvmeAttachMode::DIRECT ? "direct" : "service",
            bp->blk_size, bp->queue_depth, bp->max_queues_per_group,
            bp->max_data_size);
}

void prime_cuda(int cuda_dev) {
    cudaError_t cerr = cudaFree(0);
    if (cerr != cudaSuccess && cerr != cudaErrorInvalidValue) {
        STEP_FAIL("cuda driver prime (cudaFree(0)) failed: %s. "
                  "Check nvidia-smi / driver / cgroup.",
                  cudaGetErrorString(cerr));
    }
    (void)cudaGetLastError();

    cerr = cudaSetDevice(cuda_dev);
    if (cerr != cudaSuccess) STEP_FAIL("cudaSetDevice(%d): %s",
                                        cuda_dev, cudaGetErrorString(cerr));
    STEP_OK("cudaSetDevice(%d)", cuda_dev);
}

int run_direct(const std::vector<std::string>& pci_addrs,
                int cuda_dev, uint32_t cap)
{
    prime_cuda(cuda_dev);

    std::vector<tutti::LocalNvmeDirectConfig> cfgs;
    cfgs.reserve(pci_addrs.size());
    for (const auto& bdf : pci_addrs) {
        cfgs.push_back({bdf, cap, /*display_name=*/{}});
    }

    tutti::LocalNvmeDirectRegistry reg(std::move(cfgs));

    if (!reg.Open()) STEP_FAIL("LocalNvmeDirectRegistry::Open() (n=%zu)",
                                pci_addrs.size());
    STEP_OK("LocalNvmeDirectRegistry::Open() n=%zu cap=%u",
            pci_addrs.size(), cap);

    if (reg.device_count() != pci_addrs.size())
        STEP_FAIL("device_count=%zu != requested=%zu",
                  reg.device_count(), pci_addrs.size());
    STEP_OK("device_count = %zu", reg.device_count());

    // Per-device sanity check.
    for (size_t i = 0; i < pci_addrs.size(); ++i) {
        const auto* d = reg.device_at(i);
        check_device_shape(d, pci_addrs[i].c_str(),
                            tutti::LocalNvmeAttachMode::DIRECT);

        const auto* d_lookup = reg.find_by_id(d->device_id);
        if (d_lookup != d) STEP_FAIL("find_by_id(%d) mismatch", d->device_id);
    }
    STEP_OK("find_by_id matches device_at for all %zu devices",
            pci_addrs.size());

    auto snapshot = reg.list();
    if (snapshot.size() != pci_addrs.size())
        STEP_FAIL("list().size=%zu != %zu", snapshot.size(), pci_addrs.size());
    STEP_OK("list() returns %zu devices", snapshot.size());

    reg.Close();
    if (reg.device_count() != 0) STEP_FAIL("device_count != 0 after close");
    STEP_OK("LocalNvmeDirectRegistry::Close() (chrdev_remove + unbind, n=%zu)",
            pci_addrs.size());
    return 0;
}

int run_service(const std::string& endpoint,
                 const std::vector<int32_t>& device_ids,
                 int32_t cuda_dev, int32_t count)
{
    // Service mode also needs CUDA primed for the libnvm-side
    // attach_client path on Hopper.
    prime_cuda(cuda_dev);

    std::vector<tutti::NvmeServiceBackedRequest> reqs;
    reqs.reserve(device_ids.size());
    for (int32_t did : device_ids) {
        tutti::NvmeServiceBackedRequest r{};
        r.daemon_device_id = did;
        r.cuda_device      = cuda_dev;
        r.num_queues       = count;
        reqs.push_back(std::move(r));
    }

    tutti::NvmeServiceBackedRegistry reg(endpoint, std::move(reqs));

    if (!reg.Open()) STEP_FAIL("NvmeServiceBackedRegistry::Open() (n=%zu)",
                                device_ids.size());
    STEP_OK("NvmeServiceBackedRegistry::Open() endpoint=%s n=%zu cuda=%d count=%d",
            endpoint.c_str(), device_ids.size(), cuda_dev, count);

    if (reg.device_count() != device_ids.size())
        STEP_FAIL("device_count=%zu != requested=%zu",
                  reg.device_count(), device_ids.size());
    STEP_OK("device_count = %zu", reg.device_count());

    for (size_t i = 0; i < device_ids.size(); ++i) {
        const auto* d = reg.device_at(i);
        check_device_shape(d, /*pci_prefix=*/nullptr,
                            tutti::LocalNvmeAttachMode::SERVICE_CLIENT);

        const auto* d_lookup = reg.find_by_id(d->device_id);
        if (d_lookup != d) STEP_FAIL("find_by_id(%d) mismatch", d->device_id);
    }
    STEP_OK("find_by_id matches device_at for all %zu devices",
            device_ids.size());

    auto snapshot = reg.list();
    if (snapshot.size() != device_ids.size())
        STEP_FAIL("list().size=%zu != %zu", snapshot.size(), device_ids.size());
    STEP_OK("list() returns %zu devices", snapshot.size());

    reg.Close();
    if (reg.device_count() != 0) STEP_FAIL("device_count != 0 after close");
    STEP_OK("NvmeServiceBackedRegistry::Close() (free_client + Disconnect, n=%zu)",
            device_ids.size());
    return 0;
}

const char* arg_after(const char* a, const char* prefix) {
    size_t n = std::strlen(prefix);
    if (std::strncmp(a, prefix, n) == 0) return a + n;
    return nullptr;
}

} // namespace

int main(int argc, char** argv) {
    std::string mode;
    std::string pci_csv;
    std::string device_csv;
    std::string endpoint = "127.0.0.1:50051";
    int32_t  cuda_dev  = 0;
    int32_t  count     = 4;
    uint32_t cap       = 32;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const char* v = nullptr;
        if      ((v = arg_after(a, "--mode=")))     mode       = v;
        else if ((v = arg_after(a, "--pci=")))      pci_csv    = v;
        else if ((v = arg_after(a, "--endpoint="))) endpoint   = v;
        else if ((v = arg_after(a, "--device=")))   device_csv = v;
        else if ((v = arg_after(a, "--cuda=")))     cuda_dev   = std::atoi(v);
        else if ((v = arg_after(a, "--gpu=")))      cuda_dev   = std::atoi(v);
        else if ((v = arg_after(a, "--count=")))    count      = std::atoi(v);
        else if ((v = arg_after(a, "--cap=")))      cap        = (uint32_t)std::atoi(v);
        else { usage(argv[0]); return 1; }
    }

    if (mode == "direct") {
        auto pci_addrs = split_csv(pci_csv);
        if (pci_addrs.empty()) { usage(argv[0]); return 1; }
        int rc = run_direct(pci_addrs, cuda_dev, cap);
        if (rc == 0) std::fprintf(stderr,
            "\n=== registry_smoke (direct, n=%zu): all %d steps passed ===\n",
            pci_addrs.size(), g_step);
        return rc;
    }

    if (mode == "service") {
        auto device_strs = split_csv(device_csv);
        if (device_strs.empty()) { usage(argv[0]); return 1; }
        std::vector<int32_t> device_ids;
        device_ids.reserve(device_strs.size());
        for (const auto& s : device_strs) device_ids.push_back(std::atoi(s.c_str()));
        int rc = run_service(endpoint, device_ids, cuda_dev, count);
        if (rc == 0) std::fprintf(stderr,
            "\n=== registry_smoke (service, n=%zu): all %d steps passed ===\n",
            device_ids.size(), g_step);
        return rc;
    }

    usage(argv[0]);
    return 1;
}

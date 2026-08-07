// layerwise_kv_overlap.cu -- StorageRuntime port of the layerwise KV-cache
// overlap simulator.
//
// HY3-shaped 128K-context request (80 layers, 512 x 256-token chunks,
// 90% prefix hit) with 3-stream layerwise pipeline:
//   read(L+1) || SGEMM compute(L) || write(L-1)
//
// All data-plane operations use ONLY the public StorageRuntime API.
//
// MEMORY ARCHITECTURE (matches legacy Coordinator):
//   Per-chunk K/V tensors: each tensor_size (512 KiB), registered
//   individually with the DataPath.  NVMe DMA goes directly to/from
//   tensors — no scratch buffer, no D2D bounce.
//
// PARTIAL-COMMIT CONTRACT / CAPACITY (Round 15 S4):
//   The DataPath below is constructed with an explicit large capacity
//   (max_in_flight_operations=4, max_batch_entries=4096; max_batch_requests
//   and max_request_bytes_override left at their "follow entries" / "entries
//   * MDTS" defaults) so that a full per-layer per-direction batch (up to
//   2*n_hit requests) is accepted by a single DataPath::submit() call — one
//   kernel launch — on the normal path. windowed_submit_wait() still
//   window-loops on partial commit (per-request RESOURCE_EXHAUSTED etc.) as
//   a defensive safety net, but under this capacity it always completes in
//   exactly one round for the shapes exercised here (checked below via the
//   DataPath's test_submit_call_count()/test_kernel_launch_count() seams).
//
// PIPELINE (Option B — interleaved single-thread):
//   Per layer L:
//     1. submit read(L+1) windows on s_r (non-blocking after wait)
//     2. submit write(L-1) windows on s_w (non-blocking after wait)
//     3. compute(L) on s_c (async, depends on read(L) event)
//   read and write IO are on separate streams → GPU-side concurrent.
//   Host blocks during each window's wait, but compute runs on s_c
//   concurrently with the other stream's IO.
//
// HONEST TIMING:
//   IO time = host wall clock of the submit+wait loop (all windows).
//   IO bytes = only actually-accepted bytes (sum of accepted * ts).
//   Bandwidth = accepted_bytes / io_time.

#include <tutti/storage_runtime.h>
#include <tutti/io_types.h>
#include <tutti/memory_types.h>
#include <tutti/presets/local_nvme.h>

#include <tutti/cuda_like.h>
#if defined(TUTTI_USE_CUDA)
#include <cuda_profiler_api.h>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace tutti;

#define STEP_OK(...) do { char _b[1024]; std::snprintf(_b,sizeof(_b),__VA_ARGS__); std::fprintf(stderr,"[ OK ] %s\n",_b); } while(0)
#define STEP_FAIL(...) do { char _b[1024]; std::snprintf(_b,sizeof(_b),__VA_ARGS__); std::fprintf(stderr,"[FAIL] %s\n",_b); std::_Exit(2); } while(0)
#define LOG_INFO(...) do { char _b[1024]; std::snprintf(_b,sizeof(_b),__VA_ARGS__); std::fprintf(stderr,"[INFO] %s\n",_b); } while(0)
#define CUDA_OK(c) do{cudaError_t _e=(c);if(_e!=cudaSuccess)STEP_FAIL("CUDA: %s (%s)",#c,cudaGetErrorString(_e));}while(0)
#define RT_STATUS(c) do{Status _s=(c);if(!_s.ok())STEP_FAIL("RT: %s",_s.message().c_str());}while(0)

static double sec_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}

static bool ensure_nofile_limit(uint64_t required, uint64_t* previous) {
    struct rlimit limit{};
    if (::getrlimit(RLIMIT_NOFILE,&limit)!=0)return false;
    *previous=limit.rlim_cur==RLIM_INFINITY
        ? std::numeric_limits<uint64_t>::max()
        : static_cast<uint64_t>(limit.rlim_cur);
    if(*previous>=required)return true;
    if(limit.rlim_max!=RLIM_INFINITY&&
       static_cast<uint64_t>(limit.rlim_max)<required){
        errno=EPERM;
        return false;
    }
    limit.rlim_cur=static_cast<rlim_t>(required);
    return ::setrlimit(RLIMIT_NOFILE,&limit)==0;
}

static bool parse_nvme_device(const char* value,
                              presets::NvmeDeviceConfig* device) {
    std::vector<std::string> fields;
    std::string input(value);
    size_t begin = 0;
    while (begin <= input.size()) {
        size_t comma = input.find(',', begin);
        fields.push_back(input.substr(begin, comma - begin));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    if (fields.size() != 4 ||
        std::any_of(fields.begin(), fields.end(),
                    [](const std::string& field) { return field.empty(); })) {
        return false;
    }
    device->ssnvme_path = std::move(fields[0]);
    device->pci_bdf = std::move(fields[1]);
    device->backing_device = std::move(fields[2]);
    device->mount_path = std::move(fields[3]);
    return true;
}

bool create_file(const std::string& path, uint64_t size) {
    // Project policy: ALL file opens carry O_DIRECT (no page-cache pollution).
    int f=::open(path.c_str(),O_CREAT|O_RDWR|O_TRUNC|O_DIRECT,0644);
    if(f<0)return false;
    // O_DIRECT requires block-aligned buffers.
    void* ap=nullptr; if(::posix_memalign(&ap,4096,1<<20)!=0){::close(f);return false;}
    std::memset(ap,0,1<<20);
    while(size>0){size_t n=std::min((uint64_t)(1<<20),size);if(::write(f,ap,n)!=(ssize_t)n){std::free(ap);::close(f);return false;}size-=n;}
    std::free(ap);
    ::fsync(f);::close(f);return true;
}

// Round 16 S3: GPU selection via env TUTTI_TEST_GPU (default 0).
inline int32_t test_gpu_id(){const char*e=std::getenv("TUTTI_TEST_GPU");int v=e?std::atoi(e):0;int dc=0;if(cudaGetDeviceCount(&dc)!=cudaSuccess||dc==0)return 0;return(v>=0&&v<dc)?v:0;}

// ---------------------------------------------------------------------------
// DpSeam — abstracts test-seam counters so windowed_submit_wait
// works with both LocalNvmeDataPath and StripedDataPath.
// R20 S2: now uses public RuntimeTelemetry from presets/local_nvme.h.
// ---------------------------------------------------------------------------
struct DpSeam {
    std::function<uint64_t()> submit_count;
    std::function<uint64_t()> launch_count;
    std::function<void()>     reset;
};
static DpSeam make_dp_seam(const presets::RuntimeTelemetry& t) {
    return DpSeam{t.submit_call_count, t.kernel_launch_count, t.reset_counters};
}

// ---------------------------------------------------------------------------
// Windowed submit+wait: handles partial-commit by retrying rejected requests.
// Returns {total_ms_host, total_accepted_bytes}.
//
// Round 15 S4 instrumentation: `dp` is the single LocalNvmeDataPath backing
// `rt`. On the normal (single-round) path, one call here must drive exactly
// one DataPath::submit() and one kernel launch (test_submit_call_count() /
// test_kernel_launch_count() seams) — this is asserted below and tallied
// into the g_windowed_* globals for the end-of-run instrumentation summary.
// ---------------------------------------------------------------------------
struct WindowedIoResult { double ms; uint64_t bytes; };

static uint64_t g_windowed_calls = 0;
static uint64_t g_windowed_multi_round_calls = 0;
static uint64_t g_windowed_rounds_total = 0;

// Round 16 S4: uses DpSeam instead of template (works with both DataPath types).
static WindowedIoResult windowed_submit_wait(
        StorageRuntime* rt,
        DpSeam& dp,
        const IoRequest* all_reqs, size_t n_reqs,
        const HostSubmitContext& ctx,
        cudaStream_t /*stream*/,  // Round 16 S7: no longer synced here; the
                                  // caller's event DAG expresses ordering.
        uint64_t bytes_per_req,
        const char* tag) {
    // Build a mutable copy of request indices that still need submission
    std::vector<IoRequest> pending(all_reqs, all_reqs + n_reqs);
    double total_ms = 0;
    uint64_t total_bytes = 0;
    int rounds = 0;
    ++g_windowed_calls;

    while (!pending.empty()) {
        // Submit everything still pending in one call; the DataPath is
        // configured with enough capacity (see Phase 0 construction) that
        // this is normally the full remaining batch and the only round.
        // The loop itself remains a generic safety net for any
        // over-capacity / partial-commit scenario.
        size_t submit_count = pending.size();

        uint64_t dp_submit0 = dp.submit_count();
        uint64_t dp_launch0 = dp.launch_count();

        auto t0 = std::chrono::steady_clock::now();
        auto o = rt->submit(pending.data(), submit_count, ctx);

        if (o.io.has_value()) {
            // Some accepted — wait for them
            auto wo = rt->wait(o.io.value(), 60000);
            if (wo.observation_status.code() != StatusCode::OK ||
                !wo.result || wo.result->state != IoState::COMPLETED) {
                STEP_FAIL("%s: wait failed (round %d)", tag, rounds);
            }
            RT_STATUS(rt->release_io(o.io.value()));
            // Round 16 S7: removed cudaStreamSynchronize(stream) here.
            // rt->wait() already polls until the kernel (launched on
            // `stream`) has completed -- the extra host sync was redundant
            // and prevented GPU-side read/write overlap across layers.
            // Cross-layer ordering is expressed via the event DAG
            // (cudaStreamWaitEvent in Phase G), not host sync.
            // rt->wait() already polls until the kernel (launched on
            // `stream`) has completed — the extra host sync was redundant
            // and prevented GPU-side read/write overlap across layers.
            // Cross-layer ordering is expressed via the event DAG
            // (cudaStreamWaitEvent in Phase G), not host sync.

            // Count accepted
            size_t accepted = 0;
            for (size_t i = 0; i < o.initial_states.size() && i < submit_count; ++i) {
                if (o.initial_states[i].state == IoRequestState::ACCEPTED) {
                    ++accepted;
                    total_bytes += bytes_per_req;
                }
            }
            // Collect rejected for next round
            std::vector<IoRequest> next_round;
            for (size_t i = 0; i < submit_count; ++i) {
                if (i >= o.initial_states.size() ||
                    o.initial_states[i].state != IoRequestState::ACCEPTED) {
                    next_round.push_back(pending[i]);
                }
            }
            // Append un-submitted (beyond submit_count)
            for (size_t i = submit_count; i < pending.size(); ++i)
                next_round.push_back(pending[i]);

            pending = std::move(next_round);
        } else {
            // All rejected — no handle, all in initial_states
            std::vector<IoRequest> next_round;
            for (size_t i = 0; i < submit_count; ++i)
                next_round.push_back(pending[i]);
            for (size_t i = submit_count; i < pending.size(); ++i)
                next_round.push_back(pending[i]);
            pending = std::move(next_round);
            // Safety: avoid infinite loop
            if (rounds > 10000) STEP_FAIL("%s: too many rounds (stuck)", tag);
        }

        total_ms += sec_since(t0) * 1e3;
        ++rounds;

        // Round 15 S4: on this (necessarily single-DataPath) runtime, a
        // round that reaches the DataPath at all must drive exactly one
        // DataPath::submit() and, on the normal path, exactly one kernel
        // launch (see LocalNvmeDataPath::submit()'s early reject_all path
        // for the only case where a call is counted without a launch).
        uint64_t dp_submit_delta = dp.submit_count() - dp_submit0;
        uint64_t dp_launch_delta = dp.launch_count() - dp_launch0;
        if (dp_submit_delta != 1)
            STEP_FAIL("%s: round %d drove %lu DataPath::submit() calls (expected 1)",
                      tag, rounds, (unsigned long)dp_submit_delta);
        if (o.io.has_value() && dp_launch_delta != 1)
            STEP_FAIL("%s: round %d drove %lu kernel launches for an accepted op (expected 1)",
                      tag, rounds, (unsigned long)dp_launch_delta);
    }

    g_windowed_rounds_total += (uint64_t)rounds;
    if (rounds > 1) ++g_windowed_multi_round_calls;

    return {total_ms, total_bytes};
}

int main(int argc,char**argv){
    uint32_t n_layers=80, chunk_tokens=256, hit_pct=90, tensor_kb=512, n_requests=2;
    uint32_t gemm_n=1024, compute_sms=64;
    uint64_t ctx_tokens=131072, compute_us=0;
    std::string data_dir="/mnt/nvme0/GPU0";
    std::vector<presets::NvmeDeviceConfig> nvme_devices = {
        {"/dev/ssnvme0","0000:08:00.0","/dev/snvme0n1","/mnt/nvme0"},
        {"/dev/ssnvme1","0000:4b:00.0","/dev/snvme1n1","/mnt/nvme1"},
        {"/dev/ssnvme2","0000:57:00.0","/dev/snvme2n1","/mnt/nvme2"},
        {"/dev/ssnvme3","0000:63:00.0","/dev/snvme3n1","/mnt/nvme3"},
    };
    bool nvme_overridden=false;
    bool verify=true;
    bool striped=true;
    for(int i=1;i<argc;){
        const char*a=argv[i];
        if(!std::strcmp(a,"--layers")&&i+1<argc){n_layers=(uint32_t)std::strtoul(argv[++i],0,10);++i;}
        else if(!std::strcmp(a,"--ctx-tokens")&&i+1<argc){ctx_tokens=std::strtoull(argv[++i],0,10);++i;}
        else if(!std::strcmp(a,"--chunk-tokens")&&i+1<argc){chunk_tokens=(uint32_t)std::strtoul(argv[++i],0,10);++i;}
        else if(!std::strcmp(a,"--hit-pct")&&i+1<argc){hit_pct=(uint32_t)std::strtoul(argv[++i],0,10);++i;}
        else if(!std::strcmp(a,"--tensor-kb")&&i+1<argc){tensor_kb=(uint32_t)std::strtoul(argv[++i],0,10);++i;}
        else if(!std::strcmp(a,"--requests")&&i+1<argc){n_requests=(uint32_t)std::strtoul(argv[++i],0,10);++i;}
        else if(!std::strcmp(a,"--compute-us")&&i+1<argc){compute_us=std::strtoull(argv[++i],0,10);++i;}
        else if(!std::strcmp(a,"--gemm-n")&&i+1<argc){gemm_n=(uint32_t)std::strtoul(argv[++i],0,10);++i;}
        else if(!std::strcmp(a,"--compute-sms")&&i+1<argc){compute_sms=(uint32_t)std::strtoul(argv[++i],0,10);++i;}
        else if(!std::strcmp(a,"--data-dir")&&i+1<argc){data_dir=argv[++i];++i;}
        else if(!std::strcmp(a,"--nvme")&&i+1<argc){
            presets::NvmeDeviceConfig device;
            if(!parse_nvme_device(argv[++i],&device)){
                std::fprintf(stderr,
                    "invalid --nvme: expected ssnvme_path,pci_bdf,backing_device,mount_path\n");
                return 1;
            }
            if(!nvme_overridden){nvme_devices.clear();nvme_overridden=true;}
            nvme_devices.push_back(std::move(device));
            ++i;
        }
        else if(!std::strcmp(a,"--verify")){verify=true;++i;}
        else if(!std::strcmp(a,"--no-verify")){verify=false;++i;}
        else if(!std::strcmp(a,"--striped")){striped=true;++i;}
        else if(!std::strcmp(a,"--single")){striped=false;++i;}
        else{std::fprintf(stderr,"unknown: %s\n",a);return 1;}
    }
    const uint64_t ts=(uint64_t)tensor_kb*1024;
    const uint64_t n_chunks=ctx_tokens/chunk_tokens;
    const uint64_t n_hit=n_chunks*hit_pct/100, n_miss=n_chunks-n_hit;
    const uint64_t file_total=2ull*n_layers*ts;
    if(!n_chunks||!n_hit||!n_miss)STEP_FAIL("bad geometry");
    if(!ts||ts%4096)STEP_FAIL("--tensor-kb must produce a positive 4 KiB-aligned tensor size");
    if(nvme_devices.empty())STEP_FAIL("no NVMe devices configured");
    const size_t ndev=nvme_devices.size();
    if(striped&&(ndev<2||(ndev&(ndev-1))!=0))
        STEP_FAIL("striped mode requires a power-of-two --nvme device count >= 2 (got %lu)",
                  (unsigned long)ndev);
    constexpr uint64_t kFdHeadroom=256;
    const uint64_t fds_per_target=striped?ndev:1;
    if(n_chunks>(std::numeric_limits<uint64_t>::max()-kFdHeadroom)/fds_per_target)
        STEP_FAIL("open-file requirement overflow");
    const uint64_t required_fds=n_chunks*fds_per_target+kFdHeadroom;
    uint64_t previous_nofile=0;
    if(!ensure_nofile_limit(required_fds,&previous_nofile))
        STEP_FAIL("need RLIMIT_NOFILE >= %lu for %lu targets x %lu file(s) plus headroom: %s",
                  (unsigned long)required_fds,(unsigned long)n_chunks,
                  (unsigned long)fds_per_target,std::strerror(errno));
    if(previous_nofile<required_fds)
        STEP_OK("RLIMIT_NOFILE raised from %lu to %lu",
                (unsigned long)previous_nofile,(unsigned long)required_fds);

    std::string striped_devs;
    for(size_t i=0;i<nvme_devices.size();++i){
        if(i)striped_devs+=',';
        striped_devs+=nvme_devices[i].mount_path;
    }

    CUDA_OK(cudaFree(0));
    int32_t gpu=test_gpu_id();
    CUDA_OK(cudaSetDevice(gpu));
    STEP_OK("cudaSetDevice(%d)",gpu);

    // ---- Runtime ----
    // R20 S2: uses public preset factory (no private headers).
    // Capacity knobs: in-flight=4, batch_entries=4096 (single) or 8192 (striped)
    // so that a full per-layer per-direction batch fits in a single submit.
    DpSeam seam;
    std::unique_ptr<StorageRuntime> rt;

    if (striped) {
        presets::StripedNvmePreset preset;
        preset.devices = nvme_devices;
        preset.gpu_id = gpu;
        preset.num_queues = 32;
        preset.stripe_unit = ts;  // Round 16 S7: unit=ts (512KiB, tensor-aligned)
        preset.max_batch_entries = 8192;
        preset.max_in_flight_operations = 4;
        preset.prp_cache_capacity = 4096;

        auto rwt = presets::make_striped_nvme_runtime(preset);
        if (!rwt.runtime) STEP_FAIL("create striped runtime");
        rt = std::move(rwt.runtime);
        seam = make_dp_seam(rwt.telemetry);
        STEP_OK("StorageRuntime created (StripedDataPath, N=%lu)",
                (unsigned long)ndev);
    } else {
        presets::LocalNvmePreset preset;
        preset.device = nvme_devices.front();
        preset.gpu_id = gpu;
        preset.num_queues = 16;
        preset.max_batch_entries = 4096;
        preset.max_in_flight_operations = 4;
        preset.handle_cache_capacity = 4096;
        preset.prp_cache_capacity = 4096;

        auto rwt = presets::make_local_nvme_runtime(preset);
        if (!rwt.runtime) STEP_FAIL("create runtime");
        rt = std::move(rwt.runtime);
        seam = make_dp_seam(rwt.telemetry);
        STEP_OK("StorageRuntime created (LocalNvmeDataPath)");
    }

    // ---- Phase A: create files ----
    auto t0=std::chrono::steady_clock::now();
    std::vector<std::string> paths(n_chunks);
    if (striped) {
        // One backing file per chunk and device/shard.
        // StripedResolver path: <mount>/striped/<name>.shard<i>
        const uint64_t total_stripes = 2ull * n_layers;
        const uint64_t shard_size = ((total_stripes + ndev - 1) / ndev) * ts;
        for (size_t d = 0; d < ndev; ++d) {
            std::string sdir = nvme_devices[d].mount_path + "/striped";
            ::mkdir(sdir.c_str(), 0755);
        }
        for(uint64_t i=0;i<n_chunks;++i){
            char nm[128];
            std::snprintf(nm,sizeof(nm),"kvlw_%lu",(unsigned long)i);
            paths[i]=nm;  // store the name (not path) for striped:// URI
            for(size_t d=0;d<ndev;++d){
                std::string sp=nvme_devices[d].mount_path+"/striped/"+nm+
                               ".shard"+std::to_string(d);
                if(!create_file(sp,shard_size))STEP_FAIL("create_file %s",sp.c_str());
            }
        }
        STEP_OK("Phase A (striped): %lu targets x %lu shards (%.1f GB) in %.2fs",
                (unsigned long)n_chunks,(unsigned long)ndev,
                (double)(n_chunks*file_total)/(1024*1024*1024),sec_since(t0));
    } else {
        for(uint64_t i=0;i<n_chunks;++i){
            char nm[128];std::snprintf(nm,sizeof(nm),"%s/kvlw_%lu",data_dir.c_str(),(unsigned long)i);
            paths[i]=nm;
            if(!create_file(paths[i],file_total))STEP_FAIL("create_file %s",nm);
        }
        STEP_OK("Phase A: %lu files (%.1f GB) in %.2fs",(unsigned long)n_chunks,
                (double)(n_chunks*file_total)/(1024*1024*1024),sec_since(t0));
    }

    // ---- Phase B: per-chunk K/V tensors + register each ----
    std::vector<void*> _raw_ptrs;
    auto gpu_alloc=[&](uint64_t sz)->void*{
        void*raw=nullptr;
        CUDA_OK(cudaMalloc(&raw,sz+65536));
        _raw_ptrs.push_back(raw);
        return reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw)+65535)&~uintptr_t(65535));
    };
    std::vector<void*> hk(n_hit),hv(n_hit),mk(n_miss),mv(n_miss);
    std::vector<MemoryHandle> hk_m(n_hit),hv_m(n_hit),mk_m(n_miss),mv_m(n_miss);
    for(uint64_t b=0;b<n_hit;++b){
        hk[b]=gpu_alloc(ts);
        hv[b]=gpu_alloc(ts);
        auto r=rt->register_memory({hk[b],ts,MemoryKind::DEVICE,MemoryOwnership::CALLER_OWNED,(int32_t)gpu,"",ts});
        if(!r.ok())STEP_FAIL("reg K hit %lu: %s",(unsigned long)b,r.status().message().c_str());
        hk_m[b]=r.value();
        r=rt->register_memory({hv[b],ts,MemoryKind::DEVICE,MemoryOwnership::CALLER_OWNED,(int32_t)gpu,"",ts});
        if(!r.ok())STEP_FAIL("reg V hit %lu: %s",(unsigned long)b,r.status().message().c_str());
        hv_m[b]=r.value();
    }
    for(uint64_t b=0;b<n_miss;++b){
        mk[b]=gpu_alloc(ts);
        mv[b]=gpu_alloc(ts);
        auto r=rt->register_memory({mk[b],ts,MemoryKind::DEVICE,MemoryOwnership::CALLER_OWNED,(int32_t)gpu,"",ts});
        if(!r.ok())STEP_FAIL("reg K miss %lu: %s",(unsigned long)b,r.status().message().c_str());
        mk_m[b]=r.value();
        r=rt->register_memory({mv[b],ts,MemoryKind::DEVICE,MemoryOwnership::CALLER_OWNED,(int32_t)gpu,"",ts});
        if(!r.ok())STEP_FAIL("reg V miss %lu: %s",(unsigned long)b,r.status().message().c_str());
        mv_m[b]=r.value();
    }
    STEP_OK("Phase B: %lu hit + %lu miss chunks, %lu tensors registered",
            (unsigned long)n_hit,(unsigned long)n_miss,
            (unsigned long)(2*n_chunks));

    // ---- Phase C: open targets ----
    std::vector<TargetHandle> tgt(n_chunks);
    for(uint64_t i=0;i<n_chunks;++i){
        std::string uri;
        OpenOptions opts;
        if (striped) {
            // Round 16 S7: per-chunk shard rotation (rot=i%N), mirroring
            // legacy's shard_placement (kv_cache_layerwise_overlap.cu:293):
            //     shard_placement = {coord_devs[(2i)%ndev],
            //                        coord_devs[(2i+1)% ndev]}
            // Rationale (legacy comment :287-291): "every batch then mixes
            // chunks living on every device, so a single
            // nvme_batch_xfer_kernel drives all N NVMes concurrently".
            //
            // Without rotation, every request in a layer shares
            // target_offset = L*ts, so shard = (L*ts/ts) % N = L % N is
            // CONSTANT across the whole batch -> the entire layer lands on
            // ONE disk and aggregation degenerates to single-disk bandwidth.
            // With rot=i%N, chunk i's shard (L%N) maps to device
            // ((L%N)+i)%N, so a layer's chunks spread evenly over all devices
            // within one fused kernel launch.
            uri = std::string("striped://") + paths[i] +
                  "?devs=" + striped_devs + "&unit=" +
                  std::to_string(ts) + "&rot=" + std::to_string(i % ndev);
            opts = OpenOptions{"striped"};
        } else {
            uri = std::string("file://") + paths[i];
            opts = OpenOptions{"file"};
        }
        auto o=rt->open(uri,opts);
        if(!o.ok())STEP_FAIL("open %s: %s",uri.c_str(),o.status().message().c_str());
        tgt[i]=o.value();
    }
    STEP_OK("Phase C: opened %lu targets (%s)",(unsigned long)n_chunks,
            striped?"striped":"file");

    // ---- Phase D: 3 streams ----
    int pl=0,ph=0;CUDA_OK(cudaDeviceGetStreamPriorityRange(&pl,&ph));
    cudaStream_t s_r,s_c,s_w;
    CUDA_OK(cudaStreamCreateWithPriority(&s_r,cudaStreamDefault,ph));
    CUDA_OK(cudaStreamCreateWithPriority(&s_c,cudaStreamDefault,pl));
    CUDA_OK(cudaStreamCreateWithPriority(&s_w,cudaStreamDefault,ph));

    // Helper: build read requests for a set of chunks at layer L
    auto build_reads = [&](uint32_t L, const std::vector<uint64_t>&idx,
                           const std::vector<MemoryHandle>&km,
                           const std::vector<MemoryHandle>&vm) -> std::vector<IoRequest> {
        std::vector<IoRequest> r(2*idx.size());
        for(size_t i=0;i<idx.size();++i){
            uint64_t b=idx[i];
            r[2*i]={IoDirection::READ,km[i],0,tgt[b],L*ts,ts};
            r[2*i+1]={IoDirection::READ,vm[i],0,tgt[b],n_layers*ts+L*ts,ts};
        }
        return r;
    };
    // Helper: build write requests
    auto build_writes = [&](uint32_t L, const std::vector<uint64_t>&idx,
                            const std::vector<MemoryHandle>&km,
                            const std::vector<MemoryHandle>&vm) -> std::vector<IoRequest> {
        std::vector<IoRequest> r(2*idx.size());
        for(size_t i=0;i<idx.size();++i){
            uint64_t b=idx[i];
            r[2*i]={IoDirection::WRITE,km[i],0,tgt[b],L*ts,ts};
            r[2*i+1]={IoDirection::WRITE,vm[i],0,tgt[b],n_layers*ts+L*ts,ts};
        }
        return r;
    };

    // ---- Phase E: pre-write HIT chunks ----
    {
        auto tw=std::chrono::steady_clock::now();
        HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION,gpu,s_w};
        for(uint32_t L=0;L<n_layers;++L){
            // Fill all hit K/V tensors with launch_fill_pattern (DMA-visible)
            for(uint64_t b=0;b<n_hit;++b){
                uint8_t sk=(uint8_t)((b*n_layers+L)&0xFF);
                uint8_t sv=(uint8_t)(sk^0xA5);
                presets::launch_fill_pattern(hk[b],sk,ts,s_w);
                presets::launch_fill_pattern(hv[b],sv,ts,s_w);
            }
            CUDA_OK(cudaStreamSynchronize(s_w));
            // Windowed write
            auto reqs = build_writes(L, [&]{std::vector<uint64_t>v(n_hit);for(uint64_t i=0;i<n_hit;++i)v[i]=i;return v;}(),
                                          hk_m, hv_m);
            auto res = windowed_submit_wait(rt.get(), seam, reqs.data(), reqs.size(),
                                            ctx, s_w, ts, "prewrite");
            if(res.bytes != reqs.size() * ts)
                STEP_FAIL("Phase E L=%u: only %lu/%lu bytes written",
                          L,(unsigned long)res.bytes,(unsigned long)(reqs.size()*ts));
        }
        STEP_OK("Phase E: pre-wrote %lu chunks x %u layers (%.2f GB) in %.2fs",
                (unsigned long)n_hit,(unsigned)n_layers,
                (double)(n_hit*2*n_layers*ts)/(1024*1024*1024),sec_since(tw));
    }

    // ---- Phase F: SGEMM + calibration ----
    float *dA,*dB,*dC;
    CUDA_OK(cudaMalloc(&dA,(size_t)gemm_n*gemm_n*4));
    CUDA_OK(cudaMalloc(&dB,(size_t)gemm_n*gemm_n*4));
    CUDA_OK(cudaMalloc(&dC,(size_t)gemm_n*gemm_n*4));
    CUDA_OK(cudaMemset(dA,0x3f,(size_t)gemm_n*gemm_n*4));
    CUDA_OK(cudaMemset(dB,0x2b,(size_t)gemm_n*gemm_n*4));
    const dim3 gb(256),gg(compute_sms);
    cudaEvent_t ew0,ew1;CUDA_OK(cudaEventCreate(&ew0));CUDA_OK(cudaEventCreate(&ew1));
    presets::launch_sgemm(dA,dB,dC,(int)gemm_n,1,s_c);
    CUDA_OK(cudaStreamSynchronize(s_c));
    CUDA_OK(cudaEventRecord(ew0,s_c));
    presets::launch_sgemm(dA,dB,dC,(int)gemm_n,1,s_c);
    CUDA_OK(cudaEventRecord(ew1,s_c));CUDA_OK(cudaEventSynchronize(ew1));
    float gemm_ms=0;CUDA_OK(cudaEventElapsedTime(&gemm_ms,ew0,ew1));
    CUDA_OK(cudaEventDestroy(ew0));CUDA_OK(cudaEventDestroy(ew1));

    std::vector<uint64_t> hi(n_hit),mi(n_miss);
    for(uint64_t i=0;i<n_hit;++i)hi[i]=i;
    for(uint64_t i=0;i<n_miss;++i)mi[i]=n_hit+i;

    // do_read / do_write: windowed submit+wait, returns {ms, bytes}
    HostSubmitContext ctx_r{ExecutionDomain::DEVICE_EXECUTION,gpu,s_r};
    HostSubmitContext ctx_w{ExecutionDomain::DEVICE_EXECUTION,gpu,s_w};

    auto do_read=[&](uint32_t L,const std::vector<uint64_t>&idx,
                     const std::vector<MemoryHandle>&km,
                     const std::vector<MemoryHandle>&vm)->WindowedIoResult{
        auto reqs=build_reads(L,idx,km,vm);
        return windowed_submit_wait(rt.get(),seam,reqs.data(),reqs.size(),
                                    ctx_r,s_r,ts,"read");
    };
    auto do_write=[&](uint32_t L,const std::vector<uint64_t>&idx,
                      const std::vector<MemoryHandle>&km,
                      const std::vector<MemoryHandle>&vm)->WindowedIoResult{
        auto reqs=build_writes(L,idx,km,vm);
        return windowed_submit_wait(rt.get(),seam,reqs.data(),reqs.size(),
                                    ctx_w,s_w,ts,"write");
    };

    // Calibrate
    if(compute_us==0){
        auto tr=std::chrono::steady_clock::now();
        auto rr=do_read(0,hi,hk_m,hv_m);
        double tr_ms=sec_since(tr)*1e3;
        auto tw=std::chrono::steady_clock::now();
        auto rw=do_write(0,mi,mk_m,mv_m);
        double tw_ms=sec_since(tw)*1e3;
        compute_us=(uint64_t)((tr_ms+tw_ms)*1e3);
        STEP_OK("Phase F: auto compute_us=%lu us (read %.3f ms / %.2f GB = %.1f GB/s, "
                "write %.3f ms / %.2f GB = %.1f GB/s)",
                (unsigned long)compute_us,tr_ms,
                (double)rr.bytes/1e9,rr.bytes/1e9/(tr_ms/1e3),
                tw_ms,(double)rw.bytes/1e9,rw.bytes/1e9/(tw_ms/1e3));
    }
    const uint32_t ci=(uint32_t)std::max(1.0,(double)compute_us/((double)gemm_ms*1000)+0.5);
    LOG_INFO("pipeline: layers=%u chunks=%lu (hit=%lu miss=%lu) compute=%lu us = %u iters "
             "(DataPath in-flight=4/batch_entries=4096, 1 submit/layer/direction expected)",
             (unsigned)n_layers,(unsigned long)n_chunks,(unsigned long)n_hit,(unsigned long)n_miss,
             (unsigned long)compute_us,(unsigned)ci);

    // ---- Phase G: 3-stream pipeline (Option B: interleaved) ----
    // Per layer L:
    //   1. submit read(L+1) windows on s_r, wait all
    //   2. submit write(L-1) windows on s_w, wait all
    //   3. compute(L) on s_c (async, depends on read(L) via event)
    // read and write are on separate streams; while host waits on write
    // windows, s_r is idle but s_c compute may be running from a prior
    // async launch.  The overlap comes from compute running on s_c
    // while host is blocked in IO waits.
    std::vector<cudaEvent_t> er(n_layers),ec(n_layers);
    for(auto&e:er)CUDA_OK(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));
    for(auto&e:ec)CUDA_OK(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));

    double sim_wall=0, sim_rms=0, sim_wms=0;
    uint64_t sim_rbytes=0, sim_wbytes=0;

    // Profiling support (CUDA only; no-op on other profiles).
#if defined(TUTTI_USE_CUDA)
    CUDA_OK(cudaProfilerStart());
#endif
    for(uint32_t rq=0;rq<n_requests;++rq){
        auto t0r=std::chrono::steady_clock::now();
        double rrms=0,rwms=0;float lr_ms=0,lw_ms=0;
        uint64_t lr_bytes=0,lw_bytes=0;

        // Prefetch L0
        {
            auto r=do_read(0,hi,hk_m,hv_m);
            lr_ms=r.ms;lr_bytes=r.bytes;rrms+=r.ms;sim_rbytes+=r.bytes;
        }
        CUDA_OK(cudaEventRecord(er[0],s_r));

        for(uint32_t L=0;L<n_layers;++L){
            // (1) compute(L) — async on s_c, depends on read(L)
            CUDA_OK(cudaStreamWaitEvent(s_c,er[L],0));
            presets::launch_sgemm(dA,dB,dC,(int)gemm_n,(int)ci,s_c);
            CUDA_OK(cudaGetLastError());
            CUDA_OK(cudaEventRecord(ec[L],s_c));

            // (2) write(L-1) — windowed on s_w, depends on compute(L-1)
            if(L>=1){
                CUDA_OK(cudaStreamWaitEvent(s_w,ec[L-1],0));
                auto r=do_write(L-1,mi,mk_m,mv_m);
                lw_ms=r.ms;lw_bytes=r.bytes;rwms+=r.ms;sim_wbytes+=r.bytes;
            }

            // (3) read(L+1) — windowed on s_r
            if(L+1<n_layers){
                auto r=do_read(L+1,hi,hk_m,hv_m);
                lr_ms=r.ms;lr_bytes=r.bytes;rrms+=r.ms;sim_rbytes+=r.bytes;
                CUDA_OK(cudaEventRecord(er[L+1],s_r));
            }

            if(L%10==9||L+1==n_layers){
                LOG_INFO("rq%u L%-3u read %.1fMB/%.2fms=%.1fGB/s write %.1fMB/%.2fms=%.1fGB/s",
                         rq+1,L,(double)lr_bytes/1e6,(double)lr_ms,
                         lr_ms>0?(double)lr_bytes/1e9/((double)lr_ms/1e3):0,
                         (double)lw_bytes/1e6,(double)lw_ms,
                         lw_ms>0?(double)lw_bytes/1e9/((double)lw_ms/1e3):0);
            }
        }
        // Drain last layer write
        CUDA_OK(cudaStreamWaitEvent(s_w,ec[n_layers-1],0));
        {
            auto r=do_write(n_layers-1,mi,mk_m,mv_m);
            lw_ms=r.ms;lw_bytes=r.bytes;rwms+=r.ms;sim_wbytes+=r.bytes;
        }
        CUDA_OK(cudaStreamSynchronize(s_c));
        CUDA_OK(cudaStreamSynchronize(s_r));
        CUDA_OK(cudaStreamSynchronize(s_w));
        double rw=sec_since(t0r);
        sim_wall+=rw;sim_rms+=rrms;sim_wms+=rwms;
        double serial=(rrms+rwms)/1e3+(double)n_layers*(double)compute_us/1e6;
        STEP_OK("Phase G: req %u %.3fs (serial %.3fs, saving %.0f%%) "
                "READ %.2fGB=%.1fGB/s WRITE %.2fGB=%.1fGB/s",
                rq+1,rw,serial,100.0*(1.0-rw/serial),
                (double)sim_rbytes/1e9/n_layers,(double)sim_rbytes/1e9/(sim_rms/1e3),
                (double)sim_wbytes/1e9/n_layers,(double)sim_wbytes/1e9/(sim_wms/1e3));
    }
#if defined(TUTTI_USE_CUDA)
    CUDA_OK(cudaProfilerStop());
#endif

    double t_serial=(sim_rms+sim_wms)/1e3+(double)n_layers*n_requests*(double)compute_us/1e6;
    double ov=t_serial>0?100.0*(1.0-sim_wall/t_serial):0;
    STEP_OK("SIM TOTAL: %u req wall=%.3fs | READ %.2fGB=%.1fGB/s | WRITE %.2fGB=%.1fGB/s | "
            "serial=%.3fs overlap %.0f%% (measured: host wall / accepted-bytes IO-time + compute)",
            (unsigned)n_requests,sim_wall,
            (double)sim_rbytes/1e9,sim_rbytes>0?(double)sim_rbytes/1e9/(sim_rms/1e3):0,
            (double)sim_wbytes/1e9,sim_wbytes>0?(double)sim_wbytes/1e9/(sim_wms/1e3):0,
            t_serial,ov);

    // ---- Phase H: verify ----
    if(verify){
        void*hb;CUDA_OK(cudaMallocHost(&hb,ts));
        uint64_t mm=0,ck=0;
        // Hit: read back, check seed
        for(uint64_t b=0;b<n_hit;b+=std::max<uint64_t>(1,n_hit/16)){
            uint32_t L=(uint32_t)(b%n_layers);
            uint8_t exp=(uint8_t)((b*n_layers+L)&0xFF);
            presets::launch_fill_pattern(hk[b],0xFF,ts,s_r);
            CUDA_OK(cudaStreamSynchronize(s_r));
            std::vector<uint64_t>one={b};
            std::vector<MemoryHandle>okm={hk_m[b]},ovm={hv_m[b]};
            do_read(L,one,okm,ovm);
            CUDA_OK(cudaMemcpyAsync(hb,hk[b],ts,cudaMemcpyDeviceToHost,s_r));
            CUDA_OK(cudaStreamSynchronize(s_r));
            if(((uint8_t*)hb)[0]!=exp){++mm;LOG_INFO("  mismatch b=%lu L=%u exp=%02X got=%02X",(unsigned long)b,L,exp,((uint8_t*)hb)[0]);}
            ++ck;
        }
        // Miss: write+read consistency
        for(uint64_t i=0;i<n_miss;i+=std::max<uint64_t>(1,n_miss/8)){
            uint64_t b=n_hit+i;
            uint32_t L=(uint32_t)(i%n_layers);
            uint8_t seed=(uint8_t)((b*31+L*7)&0xFF);
            presets::launch_fill_pattern(mk[i],seed,ts,s_w);
            presets::launch_fill_pattern(mv[i],seed^0xA5,ts,s_w);
            CUDA_OK(cudaStreamSynchronize(s_w));
            std::vector<uint64_t>one={b};
            std::vector<MemoryHandle>okm={mk_m[i]},ovm={mv_m[i]};
            do_write(L,one,okm,ovm);
            presets::launch_fill_pattern(mk[i],0,ts,s_r);
            CUDA_OK(cudaStreamSynchronize(s_r));
            do_read(L,one,okm,ovm);
            CUDA_OK(cudaMemcpyAsync(hb,mk[i],ts,cudaMemcpyDeviceToHost,s_r));
            CUDA_OK(cudaStreamSynchronize(s_r));
            if(((uint8_t*)hb)[0]!=seed){++mm;LOG_INFO("  miss mismatch b=%lu L=%u exp=%02X got=%02X",(unsigned long)b,L,seed,((uint8_t*)hb)[0]);}
            ++ck;
        }
        CUDA_OK(cudaFreeHost(hb));
        if(mm)STEP_FAIL("Phase H: %lu/%lu mismatch",(unsigned long)mm,(unsigned long)ck);
        STEP_OK("Phase H: verified %lu samples, all correct",(unsigned long)ck);
    }

    // ---- Round 15 S4 instrumentation summary ----
    // Every windowed_submit_wait() call above (Phase E prewrite, Phase F
    // calibration, Phase G per-layer read/write, Phase H verify) already
    // hard-asserted per-round DataPath::submit()==1 and (when accepted)
    // kernel-launch==1. Aggregate calls==rounds (and multi_round==0) proves
    // every one of those calls completed in exactly one round, i.e. one
    // rt.submit -> one DataPath::submit -> one kernel launch per
    // layer/direction, end to end.
    LOG_INFO("Round15 S4 instrumentation: windowed_submit_wait calls=%lu total_rounds=%lu "
             "multi_round_calls=%lu (expect rounds==calls, multi_round==0 at this capacity)",
             (unsigned long)g_windowed_calls,(unsigned long)g_windowed_rounds_total,
             (unsigned long)g_windowed_multi_round_calls);
    if (g_windowed_multi_round_calls != 0 || g_windowed_rounds_total != g_windowed_calls)
        STEP_FAIL("Round15 S4: expected every windowed_submit_wait call to complete in exactly "
                  "1 round (calls=%lu rounds=%lu multi_round=%lu)",
                  (unsigned long)g_windowed_calls,(unsigned long)g_windowed_rounds_total,
                  (unsigned long)g_windowed_multi_round_calls);

    // ---- Cleanup ----
    CUDA_OK(cudaFree(dA));CUDA_OK(cudaFree(dB));CUDA_OK(cudaFree(dC));
    for(auto&e:er)CUDA_OK(cudaEventDestroy(e));
    for(auto&e:ec)CUDA_OK(cudaEventDestroy(e));
    for(uint64_t b=0;b<n_hit;++b){RT_STATUS(rt->unregister_memory(hk_m[b]));RT_STATUS(rt->unregister_memory(hv_m[b]));}
    for(uint64_t b=0;b<n_miss;++b){RT_STATUS(rt->unregister_memory(mk_m[b]));RT_STATUS(rt->unregister_memory(mv_m[b]));}
    for(void* p:_raw_ptrs)CUDA_OK(cudaFree(p));
    for(uint64_t i=0;i<n_chunks;++i){
        RT_STATUS(rt->close(tgt[i]));
        if (striped) {
            for(size_t d=0;d<ndev;++d){
                std::string sp=nvme_devices[d].mount_path+"/striped/"+paths[i]+
                               ".shard"+std::to_string(d);
                if(::unlink(sp.c_str())!=0)
                    STEP_FAIL("unlink %s failed",sp.c_str());
            }
        } else {
            ::unlink(paths[i].c_str());
        }
    }
    CUDA_OK(cudaStreamDestroy(s_r));CUDA_OK(cudaStreamDestroy(s_c));CUDA_OK(cudaStreamDestroy(s_w));
    RT_STATUS(rt->shutdown(5000));
    std::fprintf(stderr,"\n=== layerwise_kv_overlap: PASSED ===\n");
    return 0;
}

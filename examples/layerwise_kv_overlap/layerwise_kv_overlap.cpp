// layerwise_kv_overlap.cpp -- Tutti's layerwise KV-cache reference workload.
//
// It simulates a HY3-shaped 128K-context request (80 layers, 512 x 256-token
// chunks, 90% prefix hit) with a three-stream pipeline:
//   read(L+1) || SGEMM compute(L) || write(L-1)
//
// TuttiRuntime assembles devices, queues, and DataPaths from --config. The
// daemon owns deployment details; callers supply only daemon-published
// accelerator view directories through --directory. Each 512 KiB K/V tensor
// is registered independently, so NVMe DMA directly reads and writes GPU
// tensors without a scratch buffer or D2D bounce.
//
// Run from the repository root after `build/bin/nvmeservice_client --endpoint
// 127.0.0.1:50051 --list-only` returns the accelerator view paths.
//
// Four-drive striped mode requires --directory values in the same order as
// tutti_layerwise_striped.yaml's device_ids:
//   sudo build/bin/tutti_layerwise_kv_overlap --striped \
//     --directory <view-0> --directory <view-1> \
//     --directory <view-2> --directory <view-3>
//
// Single-drive mode:
//   sudo build/bin/tutti_layerwise_kv_overlap --single \
//     --config examples/layerwise_kv_overlap/tutti_layerwise_local.yaml \
//     --directory <view-0>
//
// Use `build/bin/tutti_layerwise_kv_overlap --help` for all runtime options.

#include <tutti/tutti_runtime.h>

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
#include <limits>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace tutti;

#ifndef TUTTI_LAYERWISE_DEFAULT_CONFIG
#define TUTTI_LAYERWISE_DEFAULT_CONFIG "tutti_layerwise_striped.yaml"
#endif

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

static std::string join_path(const std::string& directory, const std::string& name) {
    std::string path = directory;
    if (!path.empty() && path.back() != '/') path.push_back('/');
    path += name;
    return path;
}

bool create_file(const std::string& path, uint64_t size) {
    // Project policy: ALL file opens carry O_DIRECT (no page-cache pollution).
    int f=::open(path.c_str(),O_CREAT|O_RDWR|O_TRUNC|O_DIRECT,0644);
    if(f<0)return false;
    void* ap=nullptr; if(::posix_memalign(&ap,4096,1<<20)!=0){::close(f);return false;}
    std::memset(ap,0,1<<20);
    while(size>0){size_t n=std::min((uint64_t)(1<<20),size);if(::write(f,ap,n)!=(ssize_t)n){std::free(ap);::close(f);return false;}size-=n;}
    std::free(ap);
    ::fsync(f);::close(f);return true;
}

// ---------------------------------------------------------------------------
// Windowed submit+wait: handles partial-commit by retrying rejected requests.
// Returns {total_ms_host, total_io_ms, total_accepted_bytes}.
//
// ms    = host wall time of submit+wait (INCLUDES waits on stream
//         dependencies the caller armed beforehand, e.g. Phase G's
//         cudaStreamWaitEvent(s_w, ec[L-1]) before write(L-1)).
// io_ms = stream time of the actual submission machinery + fused kernel,
//         measured with events recorded around rt->submit().  ev_io0 fires
//         only after prior stream dependencies are satisfied, so this
//         number excludes dependency waits and reflects real IO speed.
// ---------------------------------------------------------------------------
struct WindowedIoResult { double ms; double io_ms; uint64_t bytes; };

static WindowedIoResult windowed_submit_wait(
        StorageRuntime* rt,
        const IoRequest* all_reqs, size_t n_reqs,
        const HostSubmitContext& ctx,
        uint64_t bytes_per_req,
        const char* tag) {
    std::vector<IoRequest> pending(all_reqs, all_reqs + n_reqs);
    double total_ms = 0;
    double total_io_ms = 0;
    uint64_t total_bytes = 0;
    int rounds = 0;

    cudaEvent_t ev_io0 = nullptr, ev_io1 = nullptr;
    CUDA_OK(cudaEventCreate(&ev_io0));
    CUDA_OK(cudaEventCreate(&ev_io1));

    while (!pending.empty()) {
        size_t submit_count = pending.size();
        auto t0 = std::chrono::steady_clock::now();
        CUDA_OK(cudaEventRecord(ev_io0, ctx.stream));
        auto o = rt->submit(pending.data(), submit_count, ctx);
        CUDA_OK(cudaEventRecord(ev_io1, ctx.stream));

        if (o.io.has_value()) {
            auto wo = rt->wait(o.io.value(), 60000);
            if (wo.observation_status.code() != StatusCode::OK ||
                !wo.result || wo.result->state != IoState::COMPLETED) {
                STEP_FAIL("%s: wait failed (round %d)", tag, rounds);
            }
            RT_STATUS(rt->release_io(o.io.value()));

            std::vector<IoRequest> next_round;
            for (size_t i = 0; i < submit_count; ++i) {
                if (i < o.initial_states.size() &&
                    o.initial_states[i].state == IoRequestState::ACCEPTED) {
                    total_bytes += bytes_per_req;
                } else {
                    next_round.push_back(pending[i]);
                }
            }
            pending = std::move(next_round);
        } else {
            if (rounds > 10000) STEP_FAIL("%s: too many rounds (stuck)", tag);
        }

        if (o.io.has_value()) {
            // wait() above synced the op event, which is ordered after
            // ev_io1 on this stream — ev_io1 is complete here.
            float io_ms = 0.f;
            CUDA_OK(cudaEventElapsedTime(&io_ms, ev_io0, ev_io1));
            total_io_ms += io_ms;
        }
        total_ms += sec_since(t0) * 1e3;
        ++rounds;
    }

    CUDA_OK(cudaEventDestroy(ev_io0));
    CUDA_OK(cudaEventDestroy(ev_io1));
    return {total_ms, total_io_ms, total_bytes};
}

int main(int argc,char**argv){
    uint32_t n_layers=80, chunk_tokens=256, hit_pct=90, tensor_kb=512, n_requests=2;
    uint32_t gemm_n=1024, compute_sms=64;
    uint64_t ctx_tokens=131072, compute_us=0;
    std::string config_path=TUTTI_LAYERWISE_DEFAULT_CONFIG;
    std::vector<std::string> directories;
    bool verify=true;
    bool striped=true;   // default matches the default (striped) YAML
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
        else if(!std::strcmp(a,"--config")&&i+1<argc){config_path=argv[++i];++i;}
        else if(!std::strcmp(a,"--directory")&&i+1<argc){directories.emplace_back(argv[++i]);++i;}
        else if(!std::strcmp(a,"--verify")){verify=true;++i;}
        else if(!std::strcmp(a,"--no-verify")){verify=false;++i;}
        else if(!std::strcmp(a,"--striped")){striped=true;++i;}
        else if(!std::strcmp(a,"--single")){striped=false;++i;}
        else if(!std::strcmp(a,"--help")||!std::strcmp(a,"-h")){
            std::fprintf(stderr,
                "Usage: %s --directory PATH [--directory PATH ...] [--config PATH] [workload flags]\n"
                "  --directory  daemon-published accelerator view (1 = single, >=2 power-of-two = striped)\n"
                "  --config     Tutti YAML (default: " TUTTI_LAYERWISE_DEFAULT_CONFIG ")\n",
                argv[0]);
            return 0;
        }
        else{std::fprintf(stderr,"unknown: %s (try --help)\n",a);return 1;}
    }

    if(directories.empty())
        STEP_FAIL("no view directories given: pass --directory <daemon-published view> "
                  "(e.g. /mnt/gpu0/ssnvme0; see `nvmeservice_client --list-only`)");
    const uint64_t ts=(uint64_t)tensor_kb*1024;
    const uint64_t n_chunks=ctx_tokens/chunk_tokens;
    const uint64_t n_hit=n_chunks*hit_pct/100, n_miss=n_chunks-n_hit;
    const uint64_t file_total=2ull*n_layers*ts;
    if(!n_chunks||!n_hit||!n_miss)STEP_FAIL("bad geometry");
    if(!ts||ts%4096)STEP_FAIL("--tensor-kb must produce a positive 4 KiB-aligned tensor size");
    const size_t ndev=directories.size();
    if(!striped&&ndev!=1)
        STEP_FAIL("--single requires exactly one --directory (got %lu)",(unsigned long)ndev);
    if(striped&&(ndev<2||(ndev&(ndev-1))!=0))
        STEP_FAIL("striped mode requires a power-of-two --directory count >= 2 (got %lu)",
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

    // ---- Runtime: assembled from the Tutti YAML; devices come from the daemon ----
    auto created = TuttiRuntime::create(config_path);
    if(!created.ok())
        STEP_FAIL("TuttiRuntime::create(%s): %s",config_path.c_str(),
                  created.status().message().c_str());
    std::unique_ptr<TuttiRuntime> owner = std::move(created).value();
    StorageRuntime* rt = owner->storage_runtime();
    if(rt==nullptr)
        STEP_FAIL("TuttiRuntime did not create a StorageRuntime");
    const int32_t gpu = rt->accel_id();
    if(gpu<0)
        STEP_FAIL("runtime accel_id is unspecified (< 0); set runtime.accel_id in the YAML");
    CUDA_OK(cudaFree(0));
    CUDA_OK(cudaSetDevice(gpu));
    STEP_OK("cudaSetDevice(%d) via TuttiRuntime (%s)",gpu,config_path.c_str());
    STEP_OK("StorageRuntime created (%s, N=%lu)",striped?"StripedDataPath":"LocalNvmeDataPath",
            (unsigned long)ndev);

    // ---- Phase A: create backing files under the daemon-published views ----
    auto t0=std::chrono::steady_clock::now();
    std::vector<std::string> paths(n_chunks);
    if (striped) {
        // One backing file per chunk and device/shard, under <view>/striped/.
        const uint64_t total_stripes = 2ull * n_layers;
        const uint64_t shard_size = ((total_stripes + ndev - 1) / ndev) * ts;
        for (size_t d = 0; d < ndev; ++d) {
            std::string sdir = join_path(directories[d], "striped");
            if(::mkdir(sdir.c_str(),0755)!=0&&errno!=EEXIST)
                STEP_FAIL("mkdir %s: %s",sdir.c_str(),std::strerror(errno));
        }
        for(uint64_t i=0;i<n_chunks;++i){
            char nm[128];
            std::snprintf(nm,sizeof(nm),"kvlw_%lu",(unsigned long)i);
            paths[i]=nm;  // store the name (not path) for striped:// URI
            for(size_t d=0;d<ndev;++d){
                std::string sp=join_path(join_path(directories[d],"striped"),
                                         std::string(nm)+".shard"+std::to_string(d));
                if(!create_file(sp,shard_size))STEP_FAIL("create_file %s",sp.c_str());
            }
        }
        STEP_OK("Phase A (striped): %lu targets x %lu shards (%.1f GB) in %.2fs",
                (unsigned long)n_chunks,(unsigned long)ndev,
                (double)(n_chunks*file_total)/(1024*1024*1024),sec_since(t0));
    } else {
        for(uint64_t i=0;i<n_chunks;++i){
            char nm[128];std::snprintf(nm,sizeof(nm),"kvlw_%lu",(unsigned long)i);
            paths[i]=join_path(directories.front(),nm);
            if(!create_file(paths[i],file_total))STEP_FAIL("create_file %s",paths[i].c_str());
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
        auto r=rt->register_memory({hk[b],ts,MemoryKind::DEVICE,MemoryOwnership::CALLER_OWNED,gpu,TUTTI_COMPILED_ACCELERATOR_PROFILE,ts});
        if(!r.ok())STEP_FAIL("reg K hit %lu: %s",(unsigned long)b,r.status().message().c_str());
        hk_m[b]=r.value();
        r=rt->register_memory({hv[b],ts,MemoryKind::DEVICE,MemoryOwnership::CALLER_OWNED,gpu,TUTTI_COMPILED_ACCELERATOR_PROFILE,ts});
        if(!r.ok())STEP_FAIL("reg V hit %lu: %s",(unsigned long)b,r.status().message().c_str());
        hv_m[b]=r.value();
    }
    for(uint64_t b=0;b<n_miss;++b){
        mk[b]=gpu_alloc(ts);
        mv[b]=gpu_alloc(ts);
        auto r=rt->register_memory({mk[b],ts,MemoryKind::DEVICE,MemoryOwnership::CALLER_OWNED,gpu,TUTTI_COMPILED_ACCELERATOR_PROFILE,ts});
        if(!r.ok())STEP_FAIL("reg K miss %lu: %s",(unsigned long)b,r.status().message().c_str());
        mk_m[b]=r.value();
        r=rt->register_memory({mv[b],ts,MemoryKind::DEVICE,MemoryOwnership::CALLER_OWNED,gpu,TUTTI_COMPILED_ACCELERATOR_PROFILE,ts});
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
            // Per-chunk shard rotation (rot=i%N): without it every request in a
            // layer shares target_offset = L*ts, so shard = L%N is constant and
            // the whole layer lands on ONE disk. With rot=i%N a layer's chunks
            // spread evenly over all devices within one fused kernel launch.
            std::string devs;
            for(size_t d=0;d<ndev;++d){if(d)devs+=',';devs+=directories[d];}
            uri = std::string("striped://") + paths[i] +
                  "?devs=" + devs + "&unit=" +
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
            for(uint64_t b=0;b<n_hit;++b){
                uint8_t sk=(uint8_t)((b*n_layers+L)&0xFF);
                uint8_t sv=(uint8_t)(sk^0xA5);
                presets::launch_fill_pattern(hk[b],sk,ts,s_w);
                presets::launch_fill_pattern(hv[b],sv,ts,s_w);
            }
            CUDA_OK(cudaStreamSynchronize(s_w));
            auto reqs = build_writes(L, [&]{std::vector<uint64_t>v(n_hit);for(uint64_t i=0;i<n_hit;++i)v[i]=i;return v;}(),
                                          hk_m, hv_m);
            auto res = windowed_submit_wait(rt, reqs.data(), reqs.size(),
                                            ctx, ts, "prewrite");
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

    HostSubmitContext ctx_r{ExecutionDomain::DEVICE_EXECUTION,gpu,s_r};
    HostSubmitContext ctx_w{ExecutionDomain::DEVICE_EXECUTION,gpu,s_w};

    auto do_read=[&](uint32_t L,const std::vector<uint64_t>&idx,
                     const std::vector<MemoryHandle>&km,
                     const std::vector<MemoryHandle>&vm)->WindowedIoResult{
        auto reqs=build_reads(L,idx,km,vm);
        return windowed_submit_wait(rt,reqs.data(),reqs.size(),ctx_r,ts,"read");
    };
    auto do_write=[&](uint32_t L,const std::vector<uint64_t>&idx,
                      const std::vector<MemoryHandle>&km,
                      const std::vector<MemoryHandle>&vm)->WindowedIoResult{
        auto reqs=build_writes(L,idx,km,vm);
        return windowed_submit_wait(rt,reqs.data(),reqs.size(),ctx_w,ts,"write");
    };

    // Warm the miss-tensor (mk/mv) PRP-cache pages BEFORE calibrating: the
    // first-ever write batch pays a one-time PRP cold-build (H2D per page),
    // which would otherwise be measured as "write time" below and inflate
    // compute_us (and with it the simulated compute iterations).
    {
        auto warm = do_write(0, mi, mk_m, mv_m);
        (void)warm;
    }

    // Calibrate
    if(compute_us==0){
        auto tr=std::chrono::steady_clock::now();
        auto rr=do_read(0,hi,hk_m,hv_m);
        double tr_ms=sec_since(tr)*1e3;
        auto tw=std::chrono::steady_clock::now();
        auto rw=do_write(0,mi,mk_m,mv_m);
        double tw_ms=sec_since(tw)*1e3;
        compute_us=(uint64_t)((tr_ms+tw_ms)*1e3);
        STEP_OK("Phase F: auto compute_us=%lu us (read %.3f ms (io %.3f) / %.2f GB = %.1f GB/s, "
                "write %.3f ms (io %.3f) / %.2f GB = %.1f GB/s)",
                (unsigned long)compute_us,tr_ms,rr.io_ms,
                (double)rr.bytes/1e9,rr.bytes/1e9/(rr.io_ms/1e3),
                tw_ms,rw.io_ms,
                (double)rw.bytes/1e9,rw.bytes/1e9/(rw.io_ms/1e3));
    }
    const uint32_t ci=(uint32_t)std::max(1.0,(double)compute_us/((double)gemm_ms*1000)+0.5);
    LOG_INFO("pipeline: layers=%u chunks=%lu (hit=%lu miss=%lu) compute=%lu us = %u iters",
             (unsigned)n_layers,(unsigned long)n_chunks,(unsigned long)n_hit,(unsigned long)n_miss,
             (unsigned long)compute_us,(unsigned)ci);

    // ---- Phase G: 3-stream pipeline ----
    std::vector<cudaEvent_t> er(n_layers),ec(n_layers);
    for(auto&e:er)CUDA_OK(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));
    for(auto&e:ec)CUDA_OK(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));

    double sim_wall=0, sim_rms=0, sim_wms=0, sim_rio=0, sim_wio=0;
    uint64_t sim_rbytes=0, sim_wbytes=0;

#if defined(TUTTI_USE_CUDA)
    CUDA_OK(cudaProfilerStart());
#endif
    for(uint32_t rq=0;rq<n_requests;++rq){
        auto t0r=std::chrono::steady_clock::now();
        double rrms=0,rwms=0,rrio=0,rwio=0;float lr_ms=0,lw_ms=0,lr_io=0,lw_io=0;
        uint64_t lr_bytes=0,lw_bytes=0;

        // Prefetch L0
        {
            auto r=do_read(0,hi,hk_m,hv_m);
            lr_ms=r.ms;lr_io=r.io_ms;lr_bytes=r.bytes;
            rrms+=r.ms;rrio+=r.io_ms;sim_rbytes+=r.bytes;
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
                lw_ms=r.ms;lw_io=r.io_ms;lw_bytes=r.bytes;
                rwms+=r.ms;rwio+=r.io_ms;sim_wbytes+=r.bytes;
            }

            // (3) read(L+1) — windowed on s_r
            if(L+1<n_layers){
                auto r=do_read(L+1,hi,hk_m,hv_m);
                lr_ms=r.ms;lr_io=r.io_ms;lr_bytes=r.bytes;
                rrms+=r.ms;rrio+=r.io_ms;sim_rbytes+=r.bytes;
                CUDA_OK(cudaEventRecord(er[L+1],s_r));
            }

            if(L%10==9||L+1==n_layers){
                LOG_INFO("rq%u L%-3u read %.1fMB io=%.2fms=%.1fGB/s | write %.1fMB io=%.2fms=%.1fGB/s "
                         "(wall r=%.2f w=%.2f ms, w-wall includes wait on compute)",
                         rq+1,L,(double)lr_bytes/1e6,(double)lr_io,
                         lr_io>0?(double)lr_bytes/1e9/((double)lr_io/1e3):0,
                         (double)lw_bytes/1e6,(double)lw_io,
                         lw_io>0?(double)lw_bytes/1e9/((double)lw_io/1e3):0,
                         (double)lr_ms,(double)lw_ms);
            }
        }
        // Drain last layer write
        CUDA_OK(cudaStreamWaitEvent(s_w,ec[n_layers-1],0));
        {
            auto r=do_write(n_layers-1,mi,mk_m,mv_m);
            lw_ms=r.ms;lw_io=r.io_ms;lw_bytes=r.bytes;
            rwms+=r.ms;rwio+=r.io_ms;sim_wbytes+=r.bytes;
        }
        CUDA_OK(cudaStreamSynchronize(s_c));
        CUDA_OK(cudaStreamSynchronize(s_r));
        CUDA_OK(cudaStreamSynchronize(s_w));
        double rw=sec_since(t0r);
        sim_wall+=rw;sim_rms+=rrms;sim_wms+=rwms;sim_rio+=rrio;sim_wio+=rwio;
        // Serial baseline uses IO stream time (dependency waits are a
        // property of the overlapped pipeline, not of the IO itself).
        double serial=(rrio+rwio)/1e3+(double)n_layers*(double)compute_us/1e6;
        STEP_OK("Phase G: req %u %.3fs (serial %.3fs, saving %.0f%%) "
                "READ %.2fGB=%.1fGB/s WRITE %.2fGB=%.1fGB/s (io-time based)",
                rq+1,rw,serial,100.0*(1.0-rw/serial),
                (double)sim_rbytes/1e9/n_layers,(double)sim_rbytes/1e9/(sim_rio/1e3),
                (double)sim_wbytes/1e9/n_layers,(double)sim_wbytes/1e9/(sim_wio/1e3));
    }
#if defined(TUTTI_USE_CUDA)
    CUDA_OK(cudaProfilerStop());
#endif

    double t_serial=(sim_rio+sim_wio)/1e3+(double)n_layers*n_requests*(double)compute_us/1e6;
    double ov=t_serial>0?100.0*(1.0-sim_wall/t_serial):0;
    STEP_OK("SIM TOTAL: %u req wall=%.3fs | READ %.2fGB=%.1fGB/s | WRITE %.2fGB=%.1fGB/s | "
            "serial=%.3fs overlap %.0f%% (bandwidths are io-time based; wall write time "
            "also waits on compute, by design)",
            (unsigned)n_requests,sim_wall,
            (double)sim_rbytes/1e9,sim_rbytes>0?(double)sim_rbytes/1e9/(sim_rio/1e3):0,
            (double)sim_wbytes/1e9,sim_wbytes>0?(double)sim_wbytes/1e9/(sim_wio/1e3):0,
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
                std::string sp=join_path(join_path(directories[d],"striped"),
                                         paths[i]+".shard"+std::to_string(d));
                if(::unlink(sp.c_str())!=0)
                    STEP_FAIL("unlink %s failed",sp.c_str());
            }
        } else {
            ::unlink(paths[i].c_str());
        }
    }
    CUDA_OK(cudaStreamDestroy(s_r));CUDA_OK(cudaStreamDestroy(s_c));CUDA_OK(cudaStreamDestroy(s_w));
    {
        const Status shutdown = owner->shutdown();
        if(!shutdown.ok())STEP_FAIL("TuttiRuntime::shutdown: %s",shutdown.message().c_str());
    }
    std::fprintf(stderr,"\n=== layerwise_kv_overlap: PASSED ===\n");
    return 0;
}

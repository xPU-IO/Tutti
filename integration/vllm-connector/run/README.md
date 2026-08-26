# vLLM connector runtime rules

This directory contains machine-local run artifacts and the reproducible
profiling contract for the Hy3-FP8 TP4 deployment.

## Storage rules

- Logs, profiles, temporary files, compiler caches, and Python caches must be
  under `/data/home/ryeqiu` or `/data2`. Never use `/tmp`, `/root/.cache`, or
  another directory on `/`.
- Profile logs, reports, temporary files, FlashInfer caches, vLLM caches, and
  Python caches are under `/data2/ryeqiu/tutti-profile`.
- KV payload files are the exception: the local-NVMe preset for `device_id=0`
  resolves FIEMAP extents on `/mnt/nvme0`, so connector KV roots must remain on
  that backing mount. `/mnt/nvme0` is a separate 5.8 TB device, not `/`.

Source `profile-env.sh` before starting vLLM or Nsight Systems:

```bash
source integration/vllm-connector/run/profile-env.sh
```

## Profile workload

The acceptance workload is Hy3-FP8, TP4, eager execution, 65,528 prompt
tokens, and two requests. Request B shares 80 percent of request A's prefix.
The service uses the real `TuttiConnectorV1` and local NVMe data path.

Before profiling a new software/cache state, start vLLM once without `nsys`
and wait for `/health` so FlashInfer JIT compilation and autotuning complete.
Do not profile first-time compilation.

The request driver is:

```bash
python integration/vllm-connector/scripts/vllm_profile_dummy.py \
  --port 8193 --model /data2/tencent/Hy3-FP8 \
  --tokens 65528 --reuse-pct 80 --requests 2
```

The equivalent offline `LLM.generate()` driver is:

```bash
python integration/vllm-connector/scripts/vllm_profile_offline.py \
  --model /data2/tencent/Hy3-FP8 --load-format phxsafetensors \
  --tokens 65528 --reuse-pct 80 \
  --reset-local-prefix-between-requests
```

The reset flag clears only vLLM's local GPU prefix-cache hashes between A and
B. It leaves the connector cache intact, forcing B's shared prefix through the
external Tutti load path. Without this reset, the sequential offline workload
can be satisfied by vLLM's local APC and produce no NVMe reads.

Qwen3.8-27B is used for fast model/profiler lifecycle iteration. It is a
hybrid GDN/full-attention model, while `TuttiConnectorV1` currently supports
one KV cache group and does not implement vLLM HMA. Use `--without-tutti` for
these profiler-only checks; this mode does not validate the Tutti data path:

```bash
python integration/vllm-connector/scripts/vllm_profile_offline.py \
  --model /data2/qwen --load-format auto --without-tutti \
  --tokens 2040 --reuse-pct 80
```

For an interactive Nsight capture, use both file gates. Launch the application
into a new session first; after the driver reports `profile ready`, start
collection and release the workload. When it reports `profile workload
complete`, stop collection, wait for the report, and release process shutdown:

```bash
source integration/vllm-connector/run/profile-env.sh
run_dir=/data2/ryeqiu/tutti-profile/run/qwen-r1
NSYS=/data2/ryeqiu/tools/nsight-systems-2026.4.1/opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/nsys
mkdir -p "$run_dir"

"$NSYS" launch --session-new=tutti_qwen_r1 \
  --trace=cuda,nvtx --cudabacktrace=none \
  --cuda-event-trace=false --cuda-memory-usage=false \
  --cuda-um-cpu-page-faults=false --cuda-um-gpu-page-faults=false \
  python integration/vllm-connector/scripts/vllm_profile_offline.py \
    --model /data2/qwen --load-format auto --without-tutti \
    --tokens 2040 --reuse-pct 80 \
    --wait-for-start-file "$run_dir/profile.start" \
    --wait-for-exit-file "$run_dir/profile.exit"

# In another shell, after "profile ready":
source integration/vllm-connector/run/profile-env.sh
run_dir=/data2/ryeqiu/tutti-profile/run/qwen-r1
NSYS=/data2/ryeqiu/tools/nsight-systems-2026.4.1/opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/nsys
"$NSYS" start --session=tutti_qwen_r1 \
  --sample=none --cpuctxsw=none --gpu-metrics-devices=none \
  --gpuctxsw=false --output="$TUTTI_PROFILE_ROOT/reports/qwen-r1"
touch "$run_dir/profile.start"

# After "profile workload complete":
"$NSYS" stop --session=tutti_qwen_r1
"$NSYS" shutdown --kill=none --session=tutti_qwen_r1
touch "$run_dir/profile.exit"
```

The gate files must not exist when the driver starts. Do not call
`llm.stop_profile()`; stopping all TP workers through `cudaProfilerStop()` can
disconnect Nsight agents before they flush.

Keep verbose per-Module NVTX off for routine captures. Enable it only with
`--layerwise-nvtx` when a Module-range proof is explicitly required; on the
64K Hy3 eager workload it expands the event volume substantially. CUDA kernel
activity itself remains large even without these hooks.

## NVTX colors

With `TUTTI_NVTX=1`, connector and DataPath ranges use a fixed palette:

- read/load/scatter: cyan (`0xFF00B8D9`);
- write/save/store/gather: orange (`0xFFFF8C00`);
- waits or mixed-direction batches: gray (`0xFF8D99A6`).

The C++ launch ranges use a colored outer range named
`tutti.local_nvme.io_kernel|op=read` or `|op=write` (and equivalent striped
names), with the legacy exact `tutti.local_nvme.io_kernel` marker nested inside
it. In the Nsight Systems GUI, select NVTX-based coloring/projection when
kernel rectangles do not inherit the range color automatically. vLLM compute
kernels keep Nsight's compute/kernel color, so storage work remains visually
distinct while existing exact-marker queries continue to work.

## System metrics capture

The low-overhead timeline command above deliberately uses
`--gpu-metrics-devices=none`; such a report cannot show GPU utilization or
PCIe counters. Run a separate metrics capture by replacing the `nsys start`
command with:

```bash
"$NSYS" start --session=tutti_qwen_r1 \
  --sample=none --cpuctxsw=none --gpuctxsw=false \
  --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=1000 \
  '--storage-metrics=--storage-devices=snvme0n1,--interval=10,--cache-samples=10' \
  --output="$TUTTI_PROFILE_ROOT/reports/qwen-r1-metrics"
```

GPU and storage metrics are system-scope features and can require elevated
perf/driver permissions. The available GPU metric set is device-specific; if
it does not expose PCIe RX/TX on this platform, record that limitation and use
DCGM/NVML or system PCIe counters alongside the Nsight report. Keep the 1 kHz
sampling rate initially to limit overhead and report growth.

DataPath progress must perform one CUDA event query per work unit. An inner
spin-until-deadline loop generated more than 147 million `cudaEventQuery`
calls in one 64K capture. The one-query implementation reduced the same report
from 4.4 GiB to 145 MiB while preserving request latency and overlap evidence.

Before capturing C++ IO NVTX, re-link the Python binding against the current
static libraries and verify both marker strings:

```bash
cd integration/vllm-connector/bindings/python
TUTTI_ROOT=/data/home/ryeqiu/Tutti \
TUTTI_BUILD_DIR=/data/home/ryeqiu/Tutti/build/cuda-module \
  python setup.py build_ext --inplace --force
strings src/tutti_runtime/_core*.so | \
  grep -E 'tutti\.(local|striped)_nvme\.io_kernel'
```

## Acceptance

An `.nsys-rep` is valid only when all of the following hold:

- both offline requests finish without exception and return completed outputs
  (the server driver, when used, must return HTTP 200);
- CUDA GPU kernel activity is present for the vLLM TP workers;
- `tutti.*` NVTX ranges are present;
- `tutti.local_nvme.io_kernel|op=read` or `|op=write` is present;
- `submit_one_kernel` is present as CUDA GPU activity;
- an external-reuse run contains `tutti.runtime.submit|op=read` and
  `tutti.request.load` (the local APC reset is required for the offline
  two-request driver);
- request-time collection includes real model execution, not only startup,
  replay code, or a minimal CUDA diagnostic.

Validate the report from the CLI before handing it off:

```bash
nsys stats --report cuda_gpu_kern_sum,nvtx_sum \
  /data2/ryeqiu/tutti-profile/reports/<report>.nsys-rep
```

For a TP worker profile, use the interactive launch/start/stop ordering above.
Starting collection only after model initialization keeps weight loading and
JIT setup out of the report. End collection with `nsys stop --session=<name>`
while the driver and workers are held alive, then release the driver. Do not
use the vLLM stop-profile API: four workers call `cudaProfilerStop()`
concurrently, which can disconnect the Nsight agent before it flushes a valid
report.

Nsight Systems 2025.3.2 with NVIDIA open driver 580.105.08 currently emits
`pSmIssueThrottleCtrl` and `refcntRequestReference_IMPL` assertions during
profiler injection. Minimal PyTorch reproduction shows these happen without
vLLM, Tutti, SNVMe, or even CUDA API tracing (`--trace=nvtx,osrt` is enough).
Treat this as an unresolved profiler/driver compatibility problem, not a clean
health check. Record assertion counts as well as Xid, ECC, and PCIe AER after
each run.

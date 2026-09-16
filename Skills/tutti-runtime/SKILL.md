---
name: tutti-runtime
description: Tutti GPU-direct NVMe KV cache offloading runtime. Use this skill when working in the Tutti repository — building it from scratch, bringing up the host environment (kernel modules, tutti_daemon, mounts), running the Python or C++ test suites, running vLLM KV-offload benchmarks, profiling with nsys or the torch profiler, or debugging symptoms such as zero cache hits, read-plan failures, ImportError on libnvm.so, or a first request far slower than steady state. Also use it when navigating the repository layout (csrc/ vs tutti/ vs scripts/) or adding support for an inference framework other than vLLM.
---

# Tutti Runtime

## Overview

Tutti offloads LLM KV cache to local NVMe over a GPU-direct path: NVMe queues are
driven from CUDA kernels and DMA lands straight in GPU memory — no host bounce
buffer, no `cudaMemcpy` on the data path. It ships a vLLM v1 `KVConnector` so a
prefill-heavy workload reuses cached prefixes instead of recomputing them.

This skill covers going from a fresh clone to a running benchmark, the invariants
that must not be broken, and the diagnostic playbook for failures whose symptoms
are far from their cause.

## One Entry Point for Everything

`scripts/tutti-env.sh` is the single interface for build, kernel modules, daemon
and tests. Prefer it over hand-typed commands — it encodes the ordering
constraints and verifies its own results.

```bash
scripts/tutti-env.sh status        # 只读检查 14 项；退出码非 0 = 环境不完整
scripts/tutti-env.sh bootstrap     # 从零：build -> build-ext -> modules -> daemon
scripts/tutti-env.sh build         # cmake configure + 全部 C++ 目标（含内核模块）
scripts/tutti-env.sh build-ext     # 两个 Python 扩展
scripts/tutti-env.sh modules load|unload|rebuild
scripts/tutti-env.sh daemon start|stop|restart
scripts/tutti-env.sh up|down       # 加载模块+起 daemon（不编译）/ 停 daemon
scripts/tutti-env.sh test py|cpp|all
scripts/tutti-env.sh env           # eval 用的 PYTHONPATH 等
```

Options: `--dry-run`, `--no-phoenix`, `--clean`, `-j N`.

**Always run `status` first when something misbehaves.** It distinguishes missing
build artifacts, an interpreter/ABI mismatch, unloaded modules, a dead daemon and
un-mounted devices — most "mysterious" failures are one of these.

### Interpreter selection matters

Extensions are compiled for one CPython ABI (`_core.cpython-311-*.so`). The
script picks `PYTHON_BIN` → active virtualenv → `python3`. A distro `python3` is
often too old (3.6 on some hosts) and yields a confusing `ImportError`; `status`
reports this explicitly as an ABI mismatch. Pass `PYTHON_BIN=/path/to/python` or
activate the project venv.

## From a Fresh Clone

Prerequisites: CUDA toolkit + nvcc, a CUDA-capable `torch` in the target
interpreter, kernel headers for the running kernel, `cmake` ≥ 3.20, and root/sudo
for module load and daemon start. NVMe devices must be available for Tutti to
claim (see `config/local/tutti_daemon.yaml`).

```bash
git clone <repo> && cd Tutti
PYTHON_BIN=/path/to/python scripts/tutti-env.sh bootstrap
PYTHON_BIN=/path/to/python scripts/tutti-env.sh test all
```

`bootstrap` refuses to continue on failure and prints the exact next command.
Ordering is not interchangeable and the script enforces it: kernel modules →
`tutti_daemon` (creates `/dev/snvme*n1`, performs `auto_mount`, serves gRPC) →
workload. Mounting before the daemon runs fails because the block devices do not
exist yet.

The script never signs kernel modules. If load fails and `dmesg` shows
"Required key not available", sign `build/module/*.ko` per your site's procedure
and retry.

## Running a vLLM Benchmark

`scripts/vllm/vllm_profile_offline.py` drives an offline A/B workload: request A
populates the cache, request B reuses a configurable prefix fraction.

```bash
eval "$(scripts/tutti-env.sh env)"
python scripts/vllm/vllm_profile_offline.py \
    --model /path/to/model --tensor-parallel-size 8 --block-size 64 \
    --tokens 10000 --reuse-pct 90 --rounds 2 \
    --kv-layout striped --stripe-unit 65536 \
    --device-groups "0,1;2,3" --num-queues 8 \
    --kv-root '/mnt/nvme0/tutti-kv-{LOCAL_RANK}' \
    --kv-load-failure-policy fail --direct-transfer-strict
```

Read `references/benchmarking.md` before multi-GPU or long-context runs: it
documents the capacity/queue sizing rules that cause hangs when wrong, why an
aborted run poisons its pool, and how to measure IO/compute overlap. Judge
performance on the reported **steady** mean, never the first round.

## Invariants — Breaking These Causes Silent Data Loss or Deep Failures

1. **The scheduler process must never import the data plane.** vLLM runs
   scheduler and workers in separate processes. `tutti.engine.metadata`,
   `tutti.storage.metadata` and `tutti.index.chunk_index` must stay free of
   `torch`/`tutti_runtime`/CUDA imports. Every `__init__.py` is deliberately
   side-effect free — do not add convenience re-exports.

2. **`csrc/include/tutti/...` is a public namespace, not a directory path.** When
   moving C++ sources, rewrite only prefixes that resolve from the repository
   root; decide per prefix by whether it exists under `csrc/include/tutti/`.
   `<tutti/config/...>` additionally resolves through a generated build-tree
   mirror. Never blanket-replace `tutti/`.

3. **Every data file must be opened `O_DIRECT`.** Buffered IO poisons the page
   cache, competes with GPU DMA for SSD bandwidth, and lets a buffered read
   return stale data after a GPU DMA write. Requires 4096-byte alignment of
   buffer, offset and length. Device nodes are exempt.

4. **`SubmitOutcome.io.has_value() == true` does not mean all requests were
   accepted.** On partial acceptance the handle tracks only accepted ops;
   rejected ones appear in `initial_states`. Callers must inspect it or window
   their submissions, otherwise `wait()` reports success for IO that never ran.

5. **`queue_depth` is owned by the kernel module,** fixed at install time. User
   space must read `ctrl->q_depth`; a smaller shadow ring desyncs the CQ phase
   and hangs only after enough completions accumulate.

## Repository Layout

```
csrc/                 C++/CUDA sources (data paths, device manager, resolvers, daemon)
  csrc/python/          pybind extension `tutti_runtime` (own setup.py)
  csrc/kv_transfer/     CUDA extension `tutti-kv-transfer` (own setup.py)
  csrc/include/tutti/   PUBLIC headers — this prefix is the installed namespace
tutti/                Python package (the importable one)
  tutti/{common,index,engine,storage}/   framework-agnostic core
  tutti/integration/vllm/                vLLM-specific adapter
tests/                C++ hardware contract tests (ctest)
tests/python/         Python test suite (pytest)
scripts/              tutti-env.sh + scripts/vllm/ benchmarks
config/local/         tutti_daemon.yaml — single source of truth for hardware facts
Skills/               this skill
```

`references/architecture.md` covers layer responsibilities, the control flow of
one request, why the index is memory-authoritative, KV layouts, the object pool
and lazy registration.

## Diagnostics

`references/diagnostics.md` documents, with the evidence chain for each: zero
cache hits with no warning; read-plan `has no recorded ready event`; a first
request far slower than steady state; `ImportError: libnvm.so`; queue creation
`EAGAIN`; mid-request stalls.

General method, in order: GPU utilisation timeline to decide host-blocked vs
device-bound → `py-spy dump` for the Python stack → `perf record` for kernel
hotspots → `eu-stack` for the native call chain. Use `nsys` for kernel/NVTX
timelines, the torch profiler when Python-level stacks are needed.

## Extending to Another Framework

Add a sibling package under `tutti/integration/<framework>/`. Keep
`tutti/{common,index,engine,storage}` framework-agnostic: the adapter translates
framework callbacks into engine calls and must not push framework types downward.
Register stores through `tutti/storage/registry.py`, which keeps two registration
faces — concrete classes for the data plane, and lazy `"module:Class"` strings for
the scheduler side so naming a store never pulls device bindings into the
scheduler process.

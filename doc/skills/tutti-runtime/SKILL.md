---
name: tutti-runtime
description: Tutti GPU-direct NVMe KV cache offloading runtime. This skill should be used when working in the Tutti repository or with its vLLM connector — including bringing up the host environment (kernel modules, tutti_daemon, mounts), building C++/CUDA targets or the pybind extension, running Python/C++ test suites, running 8-GPU striped benchmarks, profiling with nsys or torch profiler, or debugging KV offload issues such as zero cache hits, read-plan failures, or slow first requests. Also use it when navigating the repository layout (csrc/ vs tutti/ vs scripts/) or adding support for a new inference framework alongside vLLM.
---

# Tutti Runtime

## Overview

Tutti offloads LLM KV cache to local NVMe over a GPU-direct path (no host bounce
buffer): NVMe queues are driven from CUDA kernels, and DMA lands straight in GPU
memory. It ships a vLLM v1 `KVConnector` so a prefill-heavy workload can reuse
cached prefixes across requests instead of recomputing them.

This skill provides the repository layout, the invariants that must not be
broken, the bring-up and verification workflow, and the diagnostic playbook for
failures whose symptoms are far from their cause.

## Repository Layout

```
csrc/                 C++/CUDA sources (data paths, device manager, resolvers, daemon)
  csrc/python/          pybind extension `tutti_runtime` (own setup.py, NOT in root wheel)
  csrc/kv_transfer/     CUDA extension `tutti-kv-transfer` (own setup.py)
  csrc/include/tutti/   PUBLIC headers — this prefix is the installed namespace
tutti/                Python package (the importable one)
  tutti/{common,index,engine,storage}/   framework-agnostic core
  tutti/integration/vllm/                vLLM-specific adapter
tests/                C++ hardware contract tests (ctest)
tests/python/         Python test suite (pytest)
scripts/              tutti-env.sh (host bring-up) + scripts/vllm/ (benchmarks)
config/local/         tutti_daemon.yaml — single source of truth for hardware facts
doc/                  manuals and design/review records
```

Read `references/architecture.md` for the layer responsibilities, the data/control
flow of one request, and where to extend for a new framework.

## Invariants — Breaking These Causes Silent Data Loss or Deep Failures

1. **The scheduler process must never import the data plane.** vLLM runs the
   scheduler and workers in separate processes. `tutti.engine.metadata`,
   `tutti.storage.metadata` and `tutti.index.chunk_index` must stay free of
   `torch`/`tutti_runtime`/CUDA imports. `tutti/__init__.py` and every subpackage
   `__init__.py` are deliberately side-effect free — do not add convenience
   re-exports there.

2. **`csrc/include/tutti/...` is a public namespace, not a directory path.**
   When moving C++ sources, rewrite only include prefixes that resolve from the
   repository root. Decide per prefix by checking whether it exists under
   `csrc/include/tutti/`; `<tutti/config/...>` additionally resolves through a
   generated build-tree mirror. Never blanket-replace `tutti/`.

3. **Every data file must be opened `O_DIRECT`.** Buffered IO poisons the page
   cache, competes with GPU DMA for SSD bandwidth, and lets a buffered read
   return stale data after a GPU DMA write. Requires 4096-byte alignment for
   buffer, offset and length. Device nodes (`/dev/ssnvme*`) are exempt.

4. **`SubmitOutcome.io.has_value() == true` does not mean all requests were
   accepted.** On partial acceptance the handle only tracks accepted ops;
   rejected ones (e.g. `RESOURCE_EXHAUSTED` back-pressure) appear in
   `initial_states`. Callers must inspect `initial_states` or window their
   submissions, otherwise `wait()` returns success for IO that never happened.

5. **`queue_depth` is owned by the kernel module,** established at install time.
   User space must not specify it; read `ctrl->q_depth`. A mismatched shadow ring
   causes CQ phase desync that only manifests after ~64 completions per queue.

## Bring-Up and Verification Workflow

### Host environment (root, once per boot)

Order is not interchangeable — `/dev/snvme*n1` only appears after the daemon
completes bring-up, so mounting earlier fails:

```bash
scripts/tutti-env.sh status     # read-only; non-zero exit means incomplete
scripts/tutti-env.sh up         # idempotent: modules -> daemon -> verify
scripts/tutti-env.sh down       # SIGTERM, triggers the daemon's umount path
```

Run `status` **before** any benchmark or hardware test and treat a non-zero exit
as a hard stop. The script never compiles or signs modules; on missing artifacts
it prints the minimal build command.

### Build

```bash
cmake -B build -S . -DTUTTI_ACCELERATOR=CUDA -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j16
```

The pybind extension is a separate distribution and must be rebuilt whenever the
build tree layout changes, because its RUNPATH points into `build/csrc/...`:

```bash
cd csrc/python && rm -rf build src/tutti_runtime/*.so   # stale objects keep the old RUNPATH
TUTTI_BUILD_DIR=<repo>/build python setup.py build_ext --inplace
```

**Never run `rm -rf build` from an unspecified working directory** — deleting the
repository `build/` breaks every deployed `.so` whose RUNPATH points there.

### Tests

```bash
# Python (needs tutti_runtime + tutti_kv_transfer on PYTHONPATH)
unset PYTHONPATH
PYTHONPATH=<repo>:<deploy>/transfer-python:<deploy>/python python -m pytest tests/python -q

# C++ hardware contract tests (require scripts/tutti-env.sh status to pass)
cd build && ctest --output-on-failure
```

Never run pytest with the repository root as the current working directory of a
plain `python` invocation when `vllm` must be imported — the repo root wins
`sys.path[0]` and turns `import vllm` into a namespace package.

## Diagnostics

For failures where the symptom is far from the cause, follow
`references/diagnostics.md`. It covers, with the evidence chain that identified
each one:

- zero cache hits with no warning (scheduler-side cold reconciliation)
- read-plan `has no recorded ready event` (host feeder vs compute race)
- a first request an order of magnitude slower than steady state (lazy
  peer-memory registration)
- `ImportError: libnvm.so` (RUNPATH pointing at a deleted build tree)
- queue creation `EAGAIN` (per-device queue-ID pool exhausted by TP ranks)

The general method, in order: GPU utilisation timeline to decide host-blocked vs
device-bound → `py-spy dump` for the Python stack → `perf record` for kernel
hotspots → `eu-stack` for the native call chain. Use `nsys` for kernel/NVTX
timelines and the torch profiler when Python-level stacks are needed.

## Benchmarks and Profiling

`scripts/vllm/vllm_profile_offline.py` drives an offline A/B workload: request A
populates the cache, request B reuses a configurable fraction of it. Key flags:
`--tokens`, `--reuse-pct`, `--rounds`, `--kv-layout {file_per_chunk,striped}`,
`--stripe-unit`, `--device-groups`, `--num-queues`, `--torch-profiler-dir`,
`--profile-window {all,b-only}`.

Read `references/benchmarking.md` before running multi-GPU or long-context
benchmarks — it documents the deployment shape, the capacity/queue sizing rules
that cause hangs when wrong, how to isolate runs so results are not polluted,
and how to measure IO/compute overlap from an nsys export.

## Extending to Another Framework

Add a sibling package under `tutti/integration/<framework>/`. Keep
`tutti/{common,index,engine,storage}` framework-agnostic: the adapter translates
framework callbacks into engine calls and must not push framework types
downward. Register new stores through `tutti/storage/registry.py`, which keeps
two registration faces — concrete classes for the data plane, and lazy
`"module:Class"` strings for the scheduler side so that importing a store name
never pulls in device bindings.

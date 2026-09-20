# Benchmarking and Profiling

## Workload Driver

`scripts/vllm/vllm_profile_offline.py` runs an offline A/B pair per round:

- **A-cold** — populates the cache (writes KV, no reads).
- **B** — shares a `--reuse-pct` prefix with A, so it loads that prefix from NVMe
  and only prefills the remainder.

Per-round token ids are perturbed at the first token so rounds do not collide in
the store while keeping shapes identical. `--rounds N` reports `first` and
`steady_mean` separately — always judge performance on **steady**, because the
first round pays one-time per-process costs (see diagnostics).

Flags that change behaviour materially:

| Flag | Effect |
|---|---|
| `--kv-layout {file_per_chunk,striped}` | single-device vs multi-device striping |
| `--stripe-unit` | striping granularity; must not fragment a tensor |
| `--device-groups "0,1;2,3"` | which ranks share which devices |
| `--num-queues` | per-rank queues; oversubscription causes `EAGAIN` |
| `--kv-load-failure-policy {fail,recompute}` | `fail` surfaces bugs; `recompute` masks them as silent full recompute |
| `--rounds`, `--reset-local-prefix-between-{requests,rounds}` | steady-state measurement |
| `--torch-profiler-dir`, `--profile-window {all,b-only}` | torch profiler with Python stacks |
| `--torch-profiler-skip-table` | skip the post-run `key_averages()` table |

`--kv-root` accepts a `{LOCAL_RANK}` placeholder; every rank **must** get a
distinct root.

## Sizing Rules That Cause Hangs When Wrong

- **Pool capacity / watermark** must cover the entire workload's unique chunk set
  (`per_request_chunks × (rounds + 2) + margin`). Too low and the pool grows
  mid-request — writing real zeros and fsyncing while the forward thread waits.
- **Queues**: `ranks_per_device × num_queues` must fit the device's user
  queue-ID pool (≈119 here). The striped preset default is too high for 4 ranks
  per device.
- **Stripe unit** should keep a layer segment whole or split it across few
  devices; per-tensor placement is the design intent, not sub-tensor interleaving.
- **Striped data files must be per-rank** (`<mount>/striped/r<rank>/`), otherwise
  ranks append to identically named files.

## Run Isolation — Non-Obvious and Bites Hard

- **A killed or interrupted run leaves the pool in a state that makes the next
  run on the same `kv_root` produce zero IO.** Always use a fresh pool tag when a
  previous run was aborted, and never compare numbers across a polluted pool.
- Give every run its own log directory (a run id) and write a summary file once.
  Re-running with the same output paths silently overwrites the authoritative
  numbers — verify against the summary/meta files, not a possibly-overwritten log.
- Check for leftover worker processes before starting: orphaned vLLM workers hold
  GPU memory and the next run fails during engine init with a confusing error.

## Measuring IO/Compute Overlap From an nsys Export

Export and query the SQLite directly:

```bash
nsys export --type sqlite --force-overwrite true --output run.sqlite run.nsys-rep
```

Classification that actually works:

- **IO kernels** — demangled name contains `tutti::`. There is effectively one
  such kernel (`fused_submit_kernel`). **`gridX` is the batch size, not the
  direction** — do not use it to separate reads from writes.
- **Compute kernels** — everything else (attention, GEMM, all-reduce, MoE).

Then, per rank (`globalPid`): merge each class's busy intervals, intersect them,
and report `intersection / io_busy`. High values mean reads and compute are truly
concurrent. A high value with `io_busy ≈ span` means IO is saturated and is the
limiter; a low value with `io_busy ≪ span` means the device is not being fed.

Kernel timings live in `CUPTI_ACTIVITY_KIND_KERNEL` joined to `StringIds` on
`demangledName`. NVTX ranges are often absent from the export even when NVTX is
enabled — prefer kernel-derived windows over NVTX for automated analysis.

## PCIe Counters

nsys GPU metrics expose `PCIe RX/TX Throughput [Throughput %]`. Calibrate the
percentage against a known pinned H2D/D2H transfer before trusting absolute
numbers; the counters can overstate peer-to-peer traffic. Use them for relative
trends, not for byte accounting.

## Torch Profiler

Prefer `--profile-window b-only`: a long-context cold prefill produces an
enormous trace with little added information. Each rank writes its own
`*.pt.trace.json.gz` (tens of MB); load one or two in Perfetto, not all.

For "where does Python time go", compute per-function **self time** from
`python_function` events: `dur` minus the summed `dur` of children (children are
linked via the `Python parent id` argument). `dur` is in **microseconds**.

Note that `stop_profile` runs `key_averages()`, which can take minutes on a large
trace and may look like a hang — the trace files are already flushed by then.

## Reporting Results Honestly

- Quote steady-state numbers; state the first-round cost separately.
- Verify the data plane actually ran (write-plan and read-plan counts, commit
  markers) before attributing a speedup to caching. A run that silently fell back
  to full recompute looks like "no regression" rather than a failure.
- When a metric changes by a few percent on a single sample, do not claim an
  effect; repeat with more rounds.

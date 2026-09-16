# Diagnostics Playbook

Every entry below was diagnosed on real hardware. The evidence chain is included
because the symptom is usually far from the cause — reproducing the reasoning is
more useful than the fix itself.

## Method (apply in this order)

1. **GPU utilisation timeline** — decide host-blocked vs device-bound before
   anything else. A stalled GPU with idle SMs means the host is stuck.
2. **`py-spy dump --pid <pid>`** — Python stack of a live process. Match
   `(active)` *and* `(active+gil)`; matching only `(active)` misclassifies busy
   threads as idle.
3. **`perf record` / `perf report`** — kernel-side hotspots (lock contention,
   driver paths).
4. **`eu-stack -p <pid>`** — native call chain when the Python stack bottoms out
   in a C extension.
5. **`nsys`** — kernel and NVTX timelines; **torch profiler** when Python-level
   stacks are needed inside the trace.

Two traps observed repeatedly:

- Python thread names do not change the OS `comm`, so `perf` attributes samples
  to a misleading thread name.
- A GPU kernel appearing right after an idle gap is **not** what the gap was
  spent on. Verify with a stack sample, not by reading the next kernel's name.

## Zero cache hits, no warning at all

**Symptom** — reuse request takes the same wall time as the cold request; write
plans 0, read plans 0, hit tokens 0; nothing in the log.

**Do not assume the pool is corrupt.** Check the store first:

```bash
ls <kv_root>-rank0/meta | wc -l      # per-layer .ok markers
# group by chunk id; a chunk is complete only with all num_layers markers,
# and it must be complete on every rank
```

If markers are complete but the scheduler reports `resident=0` at the first step,
the failure is in **cold reconciliation**, not in the data.

**Root cause found this way** — `_valid_all_rank_chunks` fails closed when
`layout.layer_span` is unset, and `layer_span` was only injected on the worker
side after bind. The scheduler-side metadata store never set it, so `scan()`
always returned empty. Because the index is memory-authoritative afterwards, hit
rate stayed zero for the whole process lifetime.

**Generalisation** — this failure signature (index never advances, hit rate zero,
no log) is shared by any break in the worker→scheduler feedback loop. A generic
guard ("hit rate zero for N consecutive steps while the store is non-empty →
warn") is worth more than fixing instances one at a time.

## `read plan failed closed ... has no recorded ready event`

**Symptom** — a single TP rank fails, request is killed under
`--kv-load-failure-policy fail`; intermittent, more likely under nsys or CPU
contention, and more likely at long context.

**Cause** — a race between the host feeder thread (submits layers 1..N-1) and
compute reaching layer N. Per-layer submit cost and per-layer compute cost are the
same order of magnitude, so the margin is near zero.

**Key distinction** — the CUDA synchronisation itself was never wrong. The fence
is `event.record(read_stream)` + `current_stream().wait_event(event)`, a pure
device-side dependency. The bug was that the host chose to *fail* when the event
object did not exist yet, instead of waiting for it to be published.

**Fix shape** — wait on a condition variable until the feeder registers the
layer, with a generous timeout, checking the failure channel and feeder-stopped
flag while waiting. The condition variable must be **separate from the state
lock**: the feeder holds the state lock while submitting a layer (hundreds of ms),
so sharing it degrades the wait into lock queueing and makes the timeout useless.

**Also note** — the default policy `recompute` masks this as a silent full
recompute: correctness is preserved, the entire performance benefit is not.

## First request far slower than steady state

**Symptom** — first A request seconds slower than subsequent identical requests;
GPU idle during the gap.

**Rule out shape/JIT first**: if the trace has zero cuBLAS calls and identical
kernel counts between rounds, it is not autotuning.

**Cause** — first `write` submit performs `nvm_dma_map_data_device` for the KV
pool (hundreds of ms per device), serialised by a global NVIDIA RM lock, executed
inside the forward thread while holding a registry lock.

**Evidence chain** — `py-spy` shows all workers blocked in `runtime.submit()` →
`perf` shows `osq_lock → rmapiLockAcquire → nvidia_p2p_dma_map_pages` →
`eu-stack` confirms `register_memory ← submit_component_backed_`.

**Fixes** — key the registration domain per device set (not per target), and warm
up registration at bind time with a one-page read.

**Separate, legitimate first-shape cost** — the first GEMM of a new shape pays
cuBLAS algorithm selection plus kernel image mmap on the host. `py-spy` shows
consecutive samples on the same `F.linear` line with the main thread active but
not holding the GIL. This is per-process per-shape, affects plain vLLM equally,
and is a benchmark artefact rather than a Tutti issue.

## `ImportError: libnvm.so: cannot open shared object file`

The pybind extension's RUNPATH points into the build tree
(`build/csrc/device_manager/nvme/libnvm`). Deleting or relocating `build/` breaks
every deployed copy.

```bash
readelf -d <path>/_core*.so | grep -i runpath   # what it expects
ls <that path>/libnvm.so                        # whether it is there
```

Rebuild the missing target, then rebuild the extension **after clearing its build
directory and stale `.so`** — otherwise `build_ext` relinks nothing and the old
RUNPATH survives.

## Queue creation fails with `EAGAIN`

Each device exposes a bounded user queue-ID pool (≈119 in this deployment).
Multiple TP ranks sharing a device multiply their per-rank `num_queues` against
that pool. Reduce `--num-queues` so that `ranks_per_device × num_queues` fits.

Related: `queue_depth` is fixed by the kernel module at install time. User space
must read `ctrl->q_depth`; specifying a smaller shadow ring desyncs the CQ phase
and hangs only after enough completions accumulate.

## Hangs / stalls mid-request

- **Capacity watermark too low** — the pool grows mid-request, writing real zeros
  and fsyncing while the forward thread waits for a ready slot. Size the
  watermark to cover the whole workload's unique chunk set.
- **Partial submit acceptance** — `wait()` can return success for IO that was
  never submitted. Inspect `initial_states` for non-`ACCEPTED` entries.

## Environment gotchas

- Run `scripts/tutti-env.sh status` first; a non-zero exit explains most
  "mysterious" failures immediately.
- Hardware contract tests need their mount points to exist and be writable
  (e.g. `/mnt/nvme0/GPU0`); a missing directory looks like a functional failure.
- Before judging a test failure as a regression, **reproduce it on the previous
  commit** (`git worktree add` a detached checkout and build there). A test can
  encode a hardware assumption that was never true on this machine.

Host-level and destructive behaviour — module removal making `/dev/snvme*`
disappear, bring-up ordering, the module↔user-space ABI handshake, the
`phoenixfs`/`nvidia_peermem` exclusion, safe-delete interception of bulk pool
deletes — is in `hardware.md`, not duplicated here.

## Adding an entry

Include the **discriminating command** for the symptom, not only the fix: the
value of this file is deciding *whether you are looking at this failure*. Prefer
turning the observation into a `status` check, a test, or a runtime guard; write
it here only when it genuinely needs human judgement. When you add one, re-run
the discriminating commands of the existing entries and delete whatever no longer
applies.

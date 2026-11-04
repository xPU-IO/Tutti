# Replica ACK experiment: 2026-11-04

This experiment validates the opt-in `TUTTI_REPLICA_NO_ACK=1` path. It does
not change the default. Each entity retains owner CQ-success gating, owner
IPC readiness and a local scatter-completion fence. Consumer ACKs are omitted;
source pages, mappings, plans and events remain retained until request-level
quiescence drains local IO/copy work and all ranks reach the barrier.

## Controlled configuration

- Eight H20 GPUs, MiniMax-M3-MXFP8, TP8/EP8/DCP1, main KV FP8, index BF16.
- Block/chunk 256 tokens, 64 KiB physical entity pages, dynamic PRP, async
  completion, direct window 4, replica window 8, LOCAL_FENCE=1. Only NO_ACK
  changes between cases; worker/queue settings and vLLM sources are unchanged.
- Prompts: 65,528 and 262,136 tokens, 95% shared prefix. Actual reused tokens:
  62,208 and 248,832. Each case uses isolated metadata and payload tags.
- Three independent cold-A/warm-B rounds, HBM prefix reset before B, fixed
  PYTHONHASHSEED=0 and deterministic sampling. Output is capped at one token.
- Wall time below is offline B-request latency, not network TTFT or isolated
  SSD transfer time. Steady means r1/r2 only; samples do not support claims
  about sub-percent performance changes. Clean runs and nsys runs are separate.

## Clean measurements

| Metric | Per-entity ACK | Request-scoped retention / NO_ACK |
|---|---:|---:|
| 64K B wall r0/r1/r2, seconds | 0.762 / 0.758 / 0.758 | 0.769 / 0.761 / 0.758 |
| 64K steady B mean, seconds | 0.758 | 0.760 |
| 64K plan ready r1/r2, per-rank median ms | 306.618 / 319.649 | 262.279 / 271.574 |
| 256K B wall r0/r1/r2, seconds | 3.330 / 3.272 / 3.278 | 3.313 / 3.328 / 3.274 |
| 256K steady B mean, seconds | 3.275 | 3.301 |
| 256K plan ready r1/r2, per-rank median ms | 1385.018 / 1397.803 | 1215.727 / 1196.930 |

Plan readiness is host fence registration, not GPU DMA completion. NO_ACK
advances this registration earlier without a measured end-to-end gain. It
remains opt-in rather than becoming the default on the strength of this test.

The protocols have identical successful NVMe read bytes and logical peer-copy
bytes. Across eight GPUs, first B reads 4.40496826171875 GiB from NVMe and
copies 9.47735595703125 GiB from peers at 64K. At 256K the respective volumes
are 17.619873046875 and 37.909423828125 GiB. Peer-copy bytes are derived from
the page mapping, not NVLink hardware counters. A bug in the previous NO_ACK
branch omitted these statistics; accounting now occurs once after scatter
enqueue. This does not represent confirmed copy completion.

## nsys: request-window and per-GPU analysis

The analysis script in the evidence directory selects each B NVTX window,
correlates CUDA launches by globalPid/correlationId, and uses the launching
thread's read_owned range to classify reads. It merges intervals separately
on each GPU before measuring intersections. All selected IO kernels had a
matching launch. It never sums overlapping GPUs as wall time.

- Each GPU and each B has 117 read-submit kernels, 117 read-wait kernels and
  117 peer-scatter kernels in both protocols. Per-layer completion fences
  remain; no whole-request read barrier was inserted ahead of layer compute.
- NO_ACK removes exactly 3,672 cudaStreamWaitEvent calls per B across eight
  GPUs, matching the main-pair and index-group peer ACK dependencies.
- In r1/r2, IO-kernel/model-kernel interval overlap is 90.60%-95.18% with ACK
  and 91.97%-96.96% with NO_ACK, across individual GPUs and rounds. These
  include CQ-wait kernels and are not SSD DMA utilization or bandwidth.
- Peer-scatter/model overlap is 92.70%-96.76% and 94.22%-97.99%, respectively.
  Per-GPU scatter union is approximately 6-7 ms. Layerwise overlap survives.
- Per-GPU read-plus-scatter spans vary substantially: 244-304 ms with ACK,
  212-294 ms with NO_ACK in the warm rounds. These span measurements include
  host/control gaps and copy time, so they cannot certify peak SSD bandwidth.

The evidence supports reducing control dependencies, but does not show a
request-latency benefit. It does not establish the next bottleneck or justify
coalescing small IO, changing workers/queues, or claiming a bandwidth ceiling.

## Correctness and limitations

- 385 related software tests passed: replica, heterogeneous page geometry,
  object layout, read service, connector integration and launch scripts.
- Added NO_ACK regressions cover local-fence completion, owned-CQ-success
  gating, copy-byte accounting and retention after peer-quiescence timeout.
- Eight-GPU real-NVMe byte-exact recovery passed with NO_ACK=1 and async
  completion=1: 140 chunks, pair/all-rank groups, unequal page widths, repeated
  epochs, delayed consumption, untouched-page protection, injected pre-submit
  read failure and shutdown. The previously stale test now creates the same
  ReadService used by the engine and intercepts the actual submit path.
- Both protocols produced identical deterministic tokens in all model rounds:
  64K A=46, 256K A=10904, all B=7750.
- The per-entity ACK control path also passed the same eight-GPU NVMe test
  after the test harness repair, with async completion enabled.
- This does not certify bad-device CQ errors, peer-process crashes or every
  cancellation schedule. Compact fragment allocation is not implemented by
  this change; payload-byte deduplication is not physical-space deduplication.

## Reproducibility artifacts

All paths below are relative to
`/mnt/nvme4/tutti-profile/optimization-20261102/`:

- Clean logs: `logs/{ack,noack}-prefill-driver.log` and
  `logs/{ack,noack}-256k-prefill-driver.log`.
- Byte-exact test: `logs/noack-byte-exact3-20261104.log`.
  Control-path regression: `logs/ack-byte-exact-regression-20261104.log`.
- nsys reports and SQLite exports:
  `reports/8gpu-tutti-65528-reuse95-ack-prefill-nsys-20261104-154628.*` and
  `reports/8gpu-tutti-65528-reuse95-noack-prefill-nsys-20261104-160036.*`.
- Analysis: `reports/analyze-replica-window.py`,
  `reports/{ack,noack}-prefill-window.json`.

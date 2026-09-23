#pragma once
#include "csrc/data_paths/local_nvme/io/prp_builder.h"  // AddressDescriptor

// csrc/data_paths/striped_local_nvme/fused_submit_kernel.cuh
//
// Fused submit kernel: a SINGLE cudaLaunchKernel that dispatches IO entries
// to N NVMe devices.  Each entry carries a dev_idx that indexes into a
// device table (array of N DeviceTargetHandle pointers).  The kernel reuses
// the shared device primitives (resolve_lba + queue acquire + SQE issue +
// CQ bounded poll) from nvme_submit_primitives.cuh -- no duplication of
// LocalNvmeDataPath's device code.
//
// Design (maintainer resolution, 2026-08-02):
//   - One kernel = one launch = one event = one stream fence.
//   - The host does stripe-split fan-out (entries with dev_idx), but the
//     GPU does the actual NVMe submission + CQ poll in a single kernel.
//   - Legacy nvme_batch_xfer_kernel (third_pkgs/Tutti) proved this model
//     works: same GPU P2P-maps multiple NVMe BARs/queues, one kernel
//     writes multiple doorbells.
//
// Top section (StripedDeviceSubmitEntry + launch declaration) is
// host-visible; the __CUDACC__-guarded section defines the kernel.

#include <cstdint>

#include "csrc/data_paths/local_nvme/io/nvme_submit_primitives.cuh"

namespace tutti::data_paths::local_nvme {
    struct DeviceTargetHandle;  // forward-decl (full def in device_target.h)
}

namespace tutti::data_paths::striped_local_nvme {

using tutti::data_paths::local_nvme::DeviceTargetHandle;
using tutti::data_paths::local_nvme::EntryCompletionStatus;

// -------------------------------------------------------------------------
// StripedDeviceSubmitEntry — one IO request entry for the fused kernel.
//
// POD; filled host-side, cudaMemcpy'd to device, consumed by the kernel.
// One entry = one block-aligned NVMe read or write of up to MDTS on one
// shard device.
//
// Key difference from local_nvme::DeviceSubmitEntry:
//   - dev_idx replaces the target pointer: the kernel looks up
//     dev_table[dev_idx] to get the DeviceTargetHandle*.
//   - shard_offset is the per-shard virtual offset (already mapped from
//     the logical offset via the stripe formula on the host).
// -------------------------------------------------------------------------
struct StripedDeviceSubmitEntry {
    std::uint32_t dev_idx;          // index into the device table [0, M*N)
    std::uint32_t direction;        // 0 = read, 1 = write
    std::uint64_t shard_offset;     // byte offset within this shard's target
    // Round 16 S6 (REQUIRED 0): ALWAYS non-null — points to a GPU-resident
    // AddressDescriptor (pre-built or arena descriptor pool).
    const tutti::data_paths::local_nvme::AddressDescriptor* prp_entry = nullptr;
};

// Host launcher: launches the fused kernel on the given CUDA stream.
// Defined in fused_submit_kernel.cu.
//   d_entries   — device array of StripedDeviceSubmitEntry[count]
//   d_status    — device array of EntryCompletionStatus[count]
//   d_dev_table — device array of const DeviceTargetHandle* [num_devs]
//   count       — number of entries
//   num_devs    — number of devices in the table (N)
//   cq_poll_budget — max CQ poll iterations before timeout
//   threads_per_block — configured CUDA block size
//   inject_flag — test seam bitmask (0 = normal production)
//   d_timing    — 4 x count globaltimer stamps, or nullptr
//   stream      — CUDA stream
//   pool_workers — 0 = one thread per entry (legacy); >0 = worker-pool
//                  model with exactly this many worker threads
//   d_task_counter — device atomic task cursor for the pool model; must
//                  be a valid allocation whenever pool_workers > 0 (the
//                  launcher resets it to 0 on the same stream before the
//                  launch). Ignored when pool_workers == 0.
// Returns cudaError_t from cudaGetLastError() after the launch.
cudaError_t launch_fused_submit(
    const StripedDeviceSubmitEntry* d_entries,
    EntryCompletionStatus*          d_status,
    const DeviceTargetHandle* const* d_dev_table,
    std::uint32_t                   count,
    std::uint32_t                   num_devs,
    std::uint32_t                   cq_poll_budget,
    std::uint32_t                   threads_per_block,
    std::uint32_t                   inject_flag,
    unsigned long long*             d_timing,   // 4 x count, or nullptr
    void*                           stream,
    std::uint32_t                   pool_workers,
    unsigned int*                   d_task_counter);

} // namespace tutti::data_paths::striped_local_nvme

// =========================================================================
// Device-only code below — compiled only by nvcc.
// =========================================================================
#if defined(__CUDACC__)

namespace tutti::data_paths::striped_local_nvme {

using tutti::data_paths::local_nvme::submit_read_one;
using tutti::data_paths::local_nvme::submit_write_one;
using tutti::data_paths::local_nvme::AddressDescriptor;

// -------------------------------------------------------------------------
// fused_submit_kernel — one thread per entry, single launch across N devices.
//
// Each thread:
//   1. Reads its StripedDeviceSubmitEntry from device memory.
//   2. Looks up dev_table[entry.dev_idx] to get the DeviceTargetHandle*.
//   3. Calls submit_read_one or submit_write_one (shared primitives),
//      which does resolve_lba + queue acquire + SQE issue + CQ bounded poll
//      on THAT device's queue.
//
// Because all entries are in a single kernel launch, the GPU can submit
// to multiple devices' doorbells concurrently (warp-level parallelism).
// Completion is a single kernel exit = single event = single stream fence.
// -------------------------------------------------------------------------
TUTTI_GLOBAL
void fused_submit_kernel(const StripedDeviceSubmitEntry* entries,
                         EntryCompletionStatus*          status,
                         const DeviceTargetHandle* const* dev_table,
                         std::uint32_t                   count,
                         std::uint32_t                   num_devs,
                         std::uint32_t                   cq_poll_budget,
                         std::uint32_t                   inject_flag,
                         unsigned long long*             timing)
{
    const std::uint32_t tid = TUTTI_THREAD_IDX_X + TUTTI_BLOCK_IDX_X * TUTTI_BLOCK_DIM_X;
    if (tid >= count) return;

    const StripedDeviceSubmitEntry e = entries[tid];
    EntryCompletionStatus* s = status ? &status[tid] : nullptr;
    unsigned long long* t = timing ? timing + 4u * tid : nullptr;

    // Bounds-check dev_idx (defensive; host should never produce OOB).
    if (e.dev_idx >= num_devs) {
        if (s) { s->result = 1; }  // treat as resolve failure
        return;
    }

    const DeviceTargetHandle* h = dev_table[e.dev_idx];

    // Round 16 S6 (REQUIRED 0): kernel single-path — ALWAYS read
    // prp1/prp2/data_length from the GPU-resident AddressDescriptor.
    const AddressDescriptor* desc = e.prp_entry;
    if (e.direction == 0) {
        submit_read_one(h, desc->prp1, desc->prp2, e.shard_offset, desc->data_length,
                        s, cq_poll_budget, inject_flag, t);
    } else {
        submit_write_one(h, desc->prp1, desc->prp2, e.shard_offset, desc->data_length,
                         s, cq_poll_budget, inject_flag, t);
    }
}

// -------------------------------------------------------------------------
// fused_submit_kernel_pool — worker-pool model (env TUTTI_POOL_WORKERS=N).
//
// A fixed set of `total_workers` threads (instead of one thread per entry)
// pulls entries from a device-side atomic task cursor and runs each entry
// to completion (LBA resolve + SQE submit + CQ poll) before pulling the
// next one.  Every atomicAdd hands out each entry index exactly once, so
// completion is still a single kernel exit = single event = single stream
// fence; per-entry status/timing slots are exclusive to their worker.
//
// Why: with one thread per entry, a large batch spawns thousands of
// concurrently spinning threads, and measured per-command turnaround
// degrades as in-flight depth grows (CQ poll traffic + queue-lock
// contention).  Here the worker count IS the in-flight bound — a small
// pool keeps each command's completion latency near SSD service time.
// -------------------------------------------------------------------------
TUTTI_GLOBAL
void fused_submit_kernel_pool(const StripedDeviceSubmitEntry* entries,
                              EntryCompletionStatus*          status,
                              const DeviceTargetHandle* const* dev_table,
                              std::uint32_t                   count,
                              std::uint32_t                   num_devs,
                              std::uint32_t                   cq_poll_budget,
                              std::uint32_t                   inject_flag,
                              unsigned long long*             timing,
                              unsigned int*                   next_task,
                              std::uint32_t                   total_workers)
{
    const std::uint32_t tid = TUTTI_THREAD_IDX_X + TUTTI_BLOCK_IDX_X * TUTTI_BLOCK_DIM_X;
    if (tid >= total_workers) return;

    for (;;) {
        const std::uint32_t idx = atomicAdd(next_task, 1u);
        if (idx >= count) return;

        const StripedDeviceSubmitEntry e = entries[idx];
        EntryCompletionStatus* s = status ? &status[idx] : nullptr;
        unsigned long long* t = timing ? timing + 4u * idx : nullptr;

        if (e.dev_idx >= num_devs) {
            if (s) { s->result = 1; }  // treat as resolve failure
            continue;
        }

        const DeviceTargetHandle* h = dev_table[e.dev_idx];
        const AddressDescriptor* desc = e.prp_entry;
        if (e.direction == 0) {
            submit_read_one(h, desc->prp1, desc->prp2, e.shard_offset, desc->data_length,
                            s, cq_poll_budget, inject_flag, t);
        } else {
            submit_write_one(h, desc->prp1, desc->prp2, e.shard_offset, desc->data_length,
                             s, cq_poll_budget, inject_flag, t);
        }
    }
}

} // namespace tutti::data_paths::striped_local_nvme

#endif // __CUDACC__

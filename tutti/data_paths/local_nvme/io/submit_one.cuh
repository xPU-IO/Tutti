#pragma once

#include "tutti/data_paths/local_nvme/io/prp_builder.h"  // AddressDescriptor

// tutti/data_paths/local_nvme/io/submit_one.cuh
//
// Private device helpers for batched, block-aligned NVMe IO.
// Ported from main: nvme_storage_device.cuh + queue_acquire_helper.cuh.
//
// Top section (DeviceSubmitEntry + launch declaration) is host-visible
// and can be included from .cpp files.  The device-only section is
// guarded by __CUDACC__ and includes libnvm device headers.
//
// Round 15 Session 5: the shared primitives (EntryCompletionStatus,
// QueueAcquireHelper, resolve_lba, submit_read_one, submit_write_one) were
// extracted verbatim into
// tutti/data_paths/local_nvme/io/nvme_submit_primitives.cuh so the striped
// multi-device fused kernel can reuse them without duplication.  This file
// now just includes that shared header and defines the single-device
// submit_one_kernel on top of it.  No behavioral change.

#include "tutti/data_paths/local_nvme/io/nvme_submit_primitives.cuh"

#include <cstdint>

namespace tutti::data_paths::local_nvme {

// Forward declaration of the GPU-resident target handle.
struct DeviceTargetHandle;

// -------------------------------------------------------------------------
// DeviceSubmitEntry -- one IO request entry for the device kernel.
//
// POD; filled host-side, cudaMemcpy'd to device, consumed by the kernel.
// One entry = one block-aligned NVMe read or write of up to the DataPath MDTS.
//
// NOTE on prp1: it must be the controller IOVA of the *exact* byte the
// transfer starts at.  snvme.ko pins GPU memory at 64 KiB granularity
// (nvm_dma_map_data_device passes page_size = 1ULL << 16), but
// nvm_dma_t::ioaddrs[] is expanded at the controller's MPS (typically
// 4 KiB).  So ioaddrs[i] is the IOVA of the i-th 4 KiB page within the
// registration, and the caller must index by memory_offset / ctrl->page_size.
// Registered buffers must be 64 KiB-aligned — see register_memory.
// -------------------------------------------------------------------------

// Round 16 S6 (REQUIRED 0): kernel single-path.  The inline prp1/prp2/length
// fields have been REMOVED — the kernel ALWAYS reads them from the GPU-
// resident AddressDescriptor pointed to by prp_entry (which is now always
// non-null).  Dynamic-path entries get their descriptor written into the
// arena's per-slot descriptor pool + H2D before launch; pre-built entries
// point to the registration-time GPU-resident descriptor array.

struct DeviceSubmitEntry {
    DeviceTargetHandle* target;     // GPU pointer to device target handle
    std::uint64_t       target_offset;  // byte offset within target
    std::uint32_t       direction;  // 0 = read, 1 = write
    std::uint32_t       _pad = 0;
    // ALWAYS non-null: points to a GPU-resident AddressDescriptor
    // (either pre-built at registration time, or written into the arena's
    // per-slot descriptor pool for the dynamic path).
    const AddressDescriptor* prp_entry = nullptr;
};

// EntryCompletionStatus is defined in nvme_submit_primitives.cuh (shared
// with the striped fused kernel).

// Host launcher: launches the one-thread-per-entry kernel on the given
// CUDA stream.  Defined in submit_one.cu.
// stream is passed as void* to avoid pulling cuda_runtime.h here;
// the caller (local_nvme_data_path.cpp) passes a cudaStream_t.
// Returns cudaError_t from cudaGetLastError() after the launch,
// mirroring main's launch_nvme_batch_xfer pattern.
// d_status is the device array of EntryCompletionStatus (one per entry).
// cq_poll_budget is the max CQ poll iterations before timeout.
// threads_per_block is the configured CUDA block size.
// inject_flag is a scalar bitmask passed by value (FIX 1: no per-op
// device allocation): bit0 = force resolve_lba failure, bit1 = synthesize
// NVMe CQ error on normal completion (test seams, 0 = normal production).
cudaError_t launch_submit_one(
    const DeviceSubmitEntry* d_entries,
    EntryCompletionStatus*   d_status,
    std::uint32_t            count,
    std::uint32_t            cq_poll_budget,
    std::uint32_t            threads_per_block,
    std::uint32_t            inject_flag,
    void*                    stream);

// Fill a GPU buffer with a byte value via a GPU kernel (not cudaMemset).
// GPU kernel writes are visible to NVMe DMA; cudaMemsetAsync may not be
// due to L2 cache coherency.  Defined in submit_one.cu.
void launch_fill_pattern(void* buf, unsigned char val, std::uint64_t n,
                          void* stream);

} // namespace tutti::data_paths::local_nvme

// =========================================================================
// Device-only code below — compiled only by nvcc.
// =========================================================================
#if defined(__CUDACC__)

namespace tutti::data_paths::local_nvme {

// -------------------------------------------------------------------------
// submit_one_kernel — one thread per entry.
//
// Each thread reads its DeviceSubmitEntry from device memory, then
// calls submit_read_one or submit_write_one.  The helper internally
// polls CQ to completion, so when the kernel finishes, real NVMe IO
// has completed for every entry.
// -------------------------------------------------------------------------

TUTTI_GLOBAL
void submit_one_kernel(const DeviceSubmitEntry* entries,
                       EntryCompletionStatus*   status,
                       std::uint32_t            count,
                       std::uint32_t            cq_poll_budget,
                       std::uint32_t            inject_flag)
{
    const std::uint32_t tid = TUTTI_THREAD_IDX_X + TUTTI_BLOCK_IDX_X * TUTTI_BLOCK_DIM_X;
    if (tid >= count) return;

    const DeviceSubmitEntry e = entries[tid];
    EntryCompletionStatus* s = status ? &status[tid] : nullptr;

    // Round 16 S6 (REQUIRED 0): kernel single-path — ALWAYS read
    // prp1/prp2/data_length from the GPU-resident AddressDescriptor.
    // The dual-path `if (e.prp_entry != nullptr)` branch is gone.
    const AddressDescriptor* desc = e.prp_entry;
    if (e.direction == 0) {
        submit_read_one(e.target, desc->prp1, desc->prp2,
                        e.target_offset, desc->data_length,
                        s, cq_poll_budget, inject_flag);
    } else {
        submit_write_one(e.target, desc->prp1, desc->prp2,
                         e.target_offset, desc->data_length,
                         s, cq_poll_budget, inject_flag);
    }
}

} // namespace tutti::data_paths::local_nvme

#endif // __CUDACC__

// tutti/data_paths/local_nvme/io/submit_one.cu
//
// Host launcher for the one-thread-per-entry NVMe IO kernel.
// Ported from main's nvme_batch_xfer_kernel.cu launch shape. The launcher
// accepts the full per-operation entry batch and spans as many CUDA blocks
// as required.

#include "tutti/data_paths/local_nvme/io/submit_one.cuh"

#include <tutti/cuda_like.h>

namespace tutti::data_paths::local_nvme {

cudaError_t launch_submit_one(
    const DeviceSubmitEntry* d_entries,
    EntryCompletionStatus*   d_status,
    std::uint32_t            count,
    std::uint32_t            cq_poll_budget,
    std::uint32_t            threads_per_block,
    std::uint32_t            inject_flag,
    void*                    stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const std::uint32_t blocks = count == 0
        ? 1 : 1 + (count - 1) / threads_per_block;
    submit_one_kernel<<<blocks, threads_per_block, 0, s>>>(
        d_entries, d_status, count, cq_poll_budget, inject_flag);
    cudaError_t err = cudaGetLastError();
    return err;
}

// Fill kernel: writes val to the first n bytes of buf.
TUTTI_GLOBAL
void fill_pattern_kernel(unsigned char* buf, unsigned char val, std::uint64_t n)
{
    std::uint64_t tid = TUTTI_THREAD_IDX_X + (std::uint64_t)TUTTI_BLOCK_IDX_X * TUTTI_BLOCK_DIM_X;
    if (tid < n) buf[tid] = val;
}

void launch_fill_pattern(void* buf, unsigned char val, std::uint64_t n,
                          void* stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    std::uint32_t threads = 256;  // fill kernel, not IO
    std::uint32_t blocks = (std::uint32_t)((n + threads - 1) / threads);
    fill_pattern_kernel<<<blocks, threads, 0, s>>>(
        (unsigned char*)buf, val, n);
}

} // namespace tutti::data_paths::local_nvme

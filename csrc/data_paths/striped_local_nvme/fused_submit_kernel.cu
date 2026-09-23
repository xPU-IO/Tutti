// csrc/data_paths/striped_local_nvme/fused_submit_kernel.cu
//
// Host launcher for the fused multi-device submit kernel.
// Compiled by nvcc; links against libnvm + CUDA runtime.

#include "csrc/data_paths/striped_local_nvme/fused_submit_kernel.cuh"

#include <tutti/cuda_like.h>

namespace tutti::data_paths::striped_local_nvme {

cudaError_t launch_fused_submit(
    const StripedDeviceSubmitEntry* d_entries,
    EntryCompletionStatus*          d_status,
    const DeviceTargetHandle* const* d_dev_table,
    std::uint32_t                   count,
    std::uint32_t                   num_devs,
    std::uint32_t                   cq_poll_budget,
    std::uint32_t                   threads_per_block,
    std::uint32_t                   inject_flag,
    unsigned long long*             d_timing,
    void*                           stream,
    std::uint32_t                   pool_workers,
    unsigned int*                   d_task_counter)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    if (pool_workers == 0 || d_task_counter == nullptr) {
        // Legacy model: one thread per entry.
        const std::uint32_t blocks = count == 0
            ? 1 : 1 + (count - 1) / threads_per_block;
        fused_submit_kernel<<<blocks, threads_per_block, 0, s>>>(
            d_entries, d_status, d_dev_table, count, num_devs,
            cq_poll_budget, inject_flag, d_timing);
        return cudaGetLastError();
    }

    // Worker-pool model: exactly pool_workers worker threads, task cursor
    // reset to 0 on the same stream so the previous batch's cursor cannot
    // leak into this one.
    cudaError_t me = cudaMemsetAsync(d_task_counter, 0, sizeof(unsigned int), s);
    if (me != cudaSuccess) {
        return me;
    }
    // Clamp the block size so the grid holds no more threads than the
    // requested worker count (the kernel also bounds-checks tid).
    const std::uint32_t tpb = pool_workers < threads_per_block
        ? (pool_workers ? pool_workers : 1u) : threads_per_block;
    const std::uint32_t blocks = 1 + (pool_workers - 1) / tpb;
    fused_submit_kernel_pool<<<blocks, tpb, 0, s>>>(
        d_entries, d_status, d_dev_table, count, num_devs,
        cq_poll_budget, inject_flag, d_timing, d_task_counter, pool_workers);
    return cudaGetLastError();
}

} // namespace tutti::data_paths::striped_local_nvme

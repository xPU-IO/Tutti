// tutti/data_paths/striped_local_nvme/fused_submit_kernel.cu
//
// Host launcher for the fused multi-device submit kernel.
// Compiled by nvcc; links against libnvm + CUDA runtime.

#include "tutti/data_paths/striped_local_nvme/fused_submit_kernel.cuh"

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
    void*                           stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const std::uint32_t blocks = count == 0
        ? 1 : 1 + (count - 1) / threads_per_block;
    fused_submit_kernel<<<blocks, threads_per_block, 0, s>>>(
        d_entries, d_status, d_dev_table, count, num_devs,
        cq_poll_budget, inject_flag);
    return cudaGetLastError();
}

} // namespace tutti::data_paths::striped_local_nvme

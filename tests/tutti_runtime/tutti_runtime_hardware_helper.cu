#include <cstdint>

#include <tutti/cuda_like.h>

namespace {

__global__ void fill_pattern(unsigned char* buffer, unsigned char value,
                             std::uint64_t size) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < size; index += stride) {
        buffer[index] = value;
    }
    __threadfence_system();
}

} // namespace

extern "C" void launch_dma_visible_fill(void* buffer, unsigned char value,
                                         std::uint64_t size,
                                         cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((size + block_size - 1) / block_size);
    if (grid_size > 1024) grid_size = 1024;
    if (grid_size < 1) grid_size = 1;
    fill_pattern<<<grid_size, block_size, 0, stream>>>(
        static_cast<unsigned char*>(buffer), value, size);
}

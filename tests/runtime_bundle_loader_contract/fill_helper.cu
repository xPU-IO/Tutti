#include <cstdint>

#include <tutti/cuda_like.h>

namespace {

__global__ void fill_value_kernel(unsigned char* buffer, unsigned char value,
                                  std::uint64_t size) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        buffer[i] = value;
    }
    __threadfence_system();
}

__global__ void fill_position_kernel(unsigned char* buffer,
                                     std::uint64_t base,
                                     std::uint64_t size) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        buffer[i] = static_cast<unsigned char>((base + i) % 251u);
    }
    __threadfence_system();
}

int grid_size(std::uint64_t size) {
    constexpr int block_size = 256;
    int grid = static_cast<int>((size + block_size - 1) / block_size);
    if (grid < 1) grid = 1;
    if (grid > 1024) grid = 1024;
    return grid;
}

} // namespace

extern "C" void phase5_fill_value(void* buffer, unsigned char value,
                                   std::uint64_t size, void* stream) {
    fill_value_kernel<<<grid_size(size), 256, 0,
                        static_cast<cudaStream_t>(stream)>>>(
        static_cast<unsigned char*>(buffer), value, size);
}

extern "C" void phase5_fill_position(void* buffer, std::uint64_t base,
                                      std::uint64_t size, void* stream) {
    fill_position_kernel<<<grid_size(size), 256, 0,
                           static_cast<cudaStream_t>(stream)>>>(
        static_cast<unsigned char*>(buffer), base, size);
}

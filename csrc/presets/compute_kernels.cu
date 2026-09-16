// tutti/presets/compute_kernels.cu
//
// CUDA kernels used by the preset layer's helper functions.
// Compiled by nvcc; linked into tutti_presets.

#include <cstdint>
#include <tutti/cuda_like.h>

namespace tutti::presets {

// Fill GPU buffer with a byte pattern (DMA-visible).
__global__ void fill_pattern_kernel(unsigned char* buf, unsigned char val, std::uint64_t n) {
    std::uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) buf[tid] = val;
}

void launch_fill_pattern(void* buf, unsigned char val, std::uint64_t n, void* stream) {
    std::uint32_t threads = 256;
    std::uint32_t blocks = (std::uint32_t)((n + threads - 1) / threads);
    fill_pattern_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (unsigned char*)buf, val, n);
}

// Simple SGEMM: C = A * B (n×n float matrices, row-major).
__global__ void sgemm_kernel(const float* A, const float* B, float* C, int n, int iters) {
    int s = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n * n; idx += s) {
        int r = idx / n, c = idx % n;
        for (int it = 0; it < iters; ++it) {
            float a = 0;
            for (int k = 0; k < n; ++k) a += A[r * n + k] * B[k * n + c];
            C[r * n + c] = a;
        }
    }
}

void launch_sgemm(const float* A, const float* B, float* C, int n, int iters, void* stream) {
    std::uint32_t threads = 256;
    int total = n * n;
    std::uint32_t blocks = (std::uint32_t)((total + threads - 1) / threads);
    sgemm_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(A, B, C, n, iters);
}

} // namespace tutti::presets

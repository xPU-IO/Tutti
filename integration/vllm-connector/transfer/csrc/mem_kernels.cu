// SPDX-License-Identifier: Apache-2.0
//
// Minimal port of the single_layer_kv_transfer path from
// third_pkgs/LMCache/csrc/cuda/mem_kernels.cu.
// Kept: single_layer_kv_transfer_kernel + its host-side dispatch wrapper,
// the EngineKVFormat guards (is_hnd / check_block_size / check_head_size)
// and get_kernel_ptr. Dropped: multi_layer* / _unilateral / _fused_ptr /
// cachegen / rocm(sgl) paths. Original license: Apache-2.0; see NOTICE.

#include <torch/extension.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#ifdef USE_ROCM
  #include <hip/hip_fp8.h>
#else
  #include <cuda_fp8.h>
#endif

#include "engine_kv_format.h"
#include "kv_transfer_types.h"

namespace py = pybind11;

namespace lmc {

// inline helper to check HND layout (callable from device and host)
__host__ __device__ __forceinline__ bool is_hnd(
    const EngineKVFormat engine_kv_format) {
  return engine_kv_format ==
             EngineKVFormat::NL_X_TWO_NB_NH_BS_HS ||  // flash attn HND
         engine_kv_format ==
             EngineKVFormat::NL_X_NB_TWO_NH_BS_HS;  // flash infer HND
}

// All paged (non-MLA) formats rely on block_size for offset computation.
// The blocked-scale indexer format is MLA-like for kv_size but its paged
// addressing is per-block, so it needs block_size too.
inline void check_block_size(const EngineKVFormat engine_kv_format,
                             const int block_size) {
  TORCH_CHECK((is_mla(engine_kv_format) &&
               engine_kv_format != EngineKVFormat::NL_X_NB_BSV_BSS) ||
                  block_size > 0,
              "block_size is required (must be > 0) for EngineKVFormat ",
              static_cast<int>(engine_kv_format));
}

// HND formats additionally need head_size to decompose scalar offsets.
inline void check_head_size(const EngineKVFormat engine_kv_format,
                            const int head_size) {
  TORCH_CHECK(!is_hnd(engine_kv_format) || head_size > 0,
              "head_size is required (must be > 0) for EngineKVFormat ",
              static_cast<int>(engine_kv_format));
}

template <typename scalar_t, EngineKVFormat format>
__global__ void single_layer_kv_transfer_kernel(
    scalar_t* __restrict__ lmc_key_value_cache,
    scalar_t* __restrict__ vllm_key_value_cache,
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int vllm_block_key_stride_in_64bit, const int vllm_value_offset,
    const int lmc_stride, const int lmc_value_offset, const int num_heads,
    const int head_size_in_64bit, const int block_size,
    const TransferDirection direction) {
  constexpr bool USE_MLA = (format == EngineKVFormat::NL_X_NB_BS_HS);
  constexpr bool HND_LAYOUT = (format == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS ||
                               format == EngineKVFormat::NL_X_NB_TWO_NH_BS_HS);

  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t lmc_key_idx = token_idx * lmc_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;

    int64_t vllm_key_idx;
    if constexpr (HND_LAYOUT) {
      // HND layout: [..., num_heads, block_size, head_size]
      vllm_key_idx = block_idx * vllm_block_key_stride_in_64bit +
                     head_idx * block_size * head_size_in_64bit +
                     block_offset * head_size_in_64bit + head_offset;
    } else {
      // NHD layout: [..., block_size, num_heads, head_size]
      // (also correct for MLA where num_heads==1)
      vllm_key_idx = block_idx * vllm_block_key_stride_in_64bit +
                     block_offset * num_heads * head_size_in_64bit +
                     head_idx * head_size_in_64bit + head_offset;
    }

    if (direction == TransferDirection::D2H) {
      // GPU to LMCache (gather: paged -> staging)
      lmc_key_value_cache[lmc_key_idx] = vllm_key_value_cache[vllm_key_idx];
      if constexpr (!USE_MLA) {
        const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
        const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
        lmc_key_value_cache[lmc_value_idx] =
            vllm_key_value_cache[vllm_value_idx];
      }
    } else {
      // LMCache to GPU (scatter: staging -> paged)
      vllm_key_value_cache[vllm_key_idx] = lmc_key_value_cache[lmc_key_idx];
      if constexpr (!USE_MLA) {
        const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
        const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
        vllm_key_value_cache[vllm_value_idx] =
            lmc_key_value_cache[lmc_value_idx];
      }
    }
  }
}

}  // namespace lmc

template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
  // Get the kernel-accessible pointer of the given type T
  // Returns NULL if the tensor is on CPU and non-pinned
  torch::Device device = tensor.device();
  if (device.is_cuda()) {
    return static_cast<T*>(tensor.data_ptr());
  } else if (device.is_cpu()) {
    T* ptr;
    auto st = cudaHostGetDevicePointer(
        (void**)&ptr, static_cast<void*>(tensor.data_ptr()), 0);
    TORCH_CHECK(st == cudaSuccess,
                "Host tensor not registered/pinned (or bad ptr)");
    return ptr;
  } else {
    TORCH_CHECK(false, "Invalid device. Device must be cuda or pinned cpu.");
  }
}

void single_layer_kv_transfer(
    torch::Tensor& lmc_key_value_cache,  // [num_tokens, 2, num_heads*head_size]
                                         // or
                                         // [2, num_tokens, num_heads*head_size]
                                         // or for MLA:
                                         // [num_tokens, aligned_head_size]

    torch::Tensor&
        vllm_key_value_cache,  // NHD: [2, num_blocks, block_size, num_heads,
                               //       head_size] for flash attention
                               //      [num_blocks, 2, block_size, num_heads,
                               //       head_size] for flash infer
                               // HND: [2, num_blocks, num_heads, block_size,
                               //       head_size] for flash attention
                               //      [num_blocks, 2, num_heads, block_size,
                               //       head_size] for flash infer
                               // MLA: [num_blocks, block_size, head_size]

    torch::Tensor& slot_mapping,  // [num_tokens]
    const TransferDirection direction, const EngineKVFormat engine_kv_format,
    const bool token_major  // true: lmc_key_value_cache is
                            // [num_tokens, 2, num_heads*head_size]
                            // false: lmc_key_value_cache is
                            // [2, num_tokens, num_heads*head_size]
) {
  int64_t* lmc_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(lmc_key_value_cache);

  int64_t* vllm_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(vllm_key_value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / vllm_key_value_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads;
  int head_size_in_64bit;
  int block_size;

  const bool use_mla = ::is_mla(engine_kv_format);
  const bool hnd_layout = lmc::is_hnd(engine_kv_format);

  if (use_mla) {
    // MLA format: [num_blocks, block_size, head_size]
    num_heads = 1;
    block_size = vllm_key_value_cache.size(1);
    head_size_in_64bit = vllm_key_value_cache.size(2) / elements_per_entry;
  } else if (hnd_layout) {
    // HND format: [..., num_heads, block_size, head_size]
    num_heads = vllm_key_value_cache.size(2);
    block_size = vllm_key_value_cache.size(3);
    head_size_in_64bit = vllm_key_value_cache.size(4) / elements_per_entry;
  } else {
    // NHD format: [..., block_size, num_heads, head_size]
    block_size = vllm_key_value_cache.size(2);
    num_heads = vllm_key_value_cache.size(3);
    head_size_in_64bit = vllm_key_value_cache.size(4) / elements_per_entry;
  }

  lmc::check_block_size(engine_kv_format, block_size);
  lmc::check_head_size(engine_kv_format, head_size_in_64bit);

  int lmc_stride;
  int lmc_value_offset;
  if (use_mla) {
    // MLA format: [num_tokens, aligned_head_size]
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = 0;  // No separate K/V for MLA
  } else if (token_major) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(1) / elements_per_entry;
  } else {
    lmc_stride = lmc_key_value_cache.stride(1) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(0) / elements_per_entry;
  }

  int vllm_block_key_stride_in_64bit;
  int vllm_value_offset;
  if (use_mla) {
    // MLA format: [num_blocks, block_size, head_size]
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = 0;  // No separate K/V for MLA
  } else if (engine_kv_format == EngineKVFormat::NL_X_TWO_NB_BS_NH_HS ||
             engine_kv_format == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS) {
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(1) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(0) / elements_per_entry;
  } else {  // engine_kv_format == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(1) / elements_per_entry;
  }

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(
      device_of(vllm_key_value_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Dispatch to the appropriate template specialization based on EngineKVFormat
#define LAUNCH_SINGLE_LAYER_KERNEL(FORMAT)                                     \
  lmc::single_layer_kv_transfer_kernel<int64_t, FORMAT>                        \
      <<<grid, block, 0, stream>>>(                                            \
          lmc_key_value_cache_ptr, vllm_key_value_cache_ptr, slot_mapping_ptr, \
          vllm_block_key_stride_in_64bit, vllm_value_offset, lmc_stride,       \
          lmc_value_offset, num_heads, head_size_in_64bit, block_size,         \
          direction);                                                          \
  break;

  switch (engine_kv_format) {
    case EngineKVFormat::NL_X_NB_BS_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_NB_BS_HS)
    case EngineKVFormat::NL_X_TWO_NB_BS_NH_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_TWO_NB_BS_NH_HS)
    case EngineKVFormat::NL_X_NB_TWO_BS_NH_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_NB_TWO_BS_NH_HS)
    case EngineKVFormat::NL_X_TWO_NB_NH_BS_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_TWO_NB_NH_BS_HS)
    case EngineKVFormat::NL_X_NB_TWO_NH_BS_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_NB_TWO_NH_BS_HS)
    default:
      TORCH_CHECK(false,
                  "Unsupported EngineKVFormat for single_layer_kv_transfer: ",
                  static_cast<int>(engine_kv_format));
  }
#undef LAUNCH_SINGLE_LAYER_KERNEL
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "single_layer_kv_transfer",
      [](torch::Tensor lmc_key_value_cache,
         torch::Tensor vllm_key_value_cache, torch::Tensor slot_mapping,
         int direction, int engine_kv_format, bool token_major) {
        return single_layer_kv_transfer(
            lmc_key_value_cache, vllm_key_value_cache, slot_mapping,
            static_cast<TransferDirection>(direction),
            static_cast<EngineKVFormat>(engine_kv_format), token_major);
      },
      py::arg("lmc_key_value_cache"), py::arg("vllm_key_value_cache"),
      py::arg("slot_mapping"), py::arg("direction"),
      py::arg("engine_kv_format"), py::arg("token_major"));
}

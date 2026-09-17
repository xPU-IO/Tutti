#pragma once

// tutti/cuda_like.h -- Unified accelerator header selector
//
// Exactly one of TUTTI_USE_CUDA / TUTTI_USE_HOST must be defined by the
// build profile (via tutti_cuda_like interface library).
// CUDA:  includes NVIDIA CUDA driver + runtime headers + libcu++ <cuda/atomic>.
// HOST:  includes the header-only HOST shim for contract tests.

#if defined(TUTTI_USE_CUDA) && defined(TUTTI_USE_HOST)
#error "Both TUTTI_USE_CUDA and TUTTI_USE_HOST are defined; exactly one must be set"
#endif

#if !defined(TUTTI_USE_CUDA) && !defined(TUTTI_USE_HOST) && !defined(TUTTI_USE_MACA) && !defined(TUTTI_USE_MUSA)
#error "Neither TUTTI_USE_CUDA , TUTTI_USE_MACA, TUTTI_USE_MUSA, nor TUTTI_USE_HOST is defined; exactly one must be set"
#endif

#if defined(TUTTI_USE_MACA) && defined(TUTTI_USE_MUSA)
#error "TUTTI_USE_MACA and TUTTI_USE_MUSA are mutually exclusive"
#endif

#if defined(TUTTI_USE_CUDA)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/atomic>

#elif defined(TUTTI_USE_HOST)

#include <tutti/gpu_vendor/host.h>

#elif defined(TUTTI_USE_MACA)
#include <tutti/gpu_vendor/maca.h>

#elif defined(TUTTI_USE_MUSA)
#include <tutti/gpu_vendor/musa.h>

#endif

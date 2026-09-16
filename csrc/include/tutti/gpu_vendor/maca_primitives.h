#pragma once

// tutti/gpu_vendor/maca_primitives.h -- MACA kernel primitives
//
// Definitive implementation.  MACA's cu-bridge compiler accepts CUDA-like
// intrinsics (__device__, __global__, __threadfence_system, atomicAdd,
// __syncthreads, threadIdx, blockIdx, blockDim, __ldg) — so the TUTTI_*
// macros expand to the same tokens as the CUDA branch.
//
// Ported from Mooncake's
// mooncake-transfer-engine/include/transport/device/maca/maca_ops.cuh
// (Apache 2.0), which confirms:
//   - __threadfence_system() works (Mooncake uses it for release stores).
//   - __ldg() works for non-coherent loads.
//   - __syncthreads() works for block barriers.
//   - cu-bridge does NOT reliably compile PTX acquire/release/barrier
//     instructions, but Tutti's macros use the high-level intrinsics
//     (not PTX), so this is not an issue.
//
// Reference: third_pkgs/Mooncake/mooncake-transfer-engine/include/transport/device/maca/maca_ops.cuh
// License: Apache 2.0

// ---------------------------------------------------------------------------
// Function attributes — accepted by cu-bridge as-is
// ---------------------------------------------------------------------------
#define TUTTI_DEVICE              __device__
#define TUTTI_HOST                __host__
#define TUTTI_GLOBAL              __global__
#define TUTTI_FORCEINLINE         __forceinline__
#define TUTTI_SHARED              __shared__

// ---------------------------------------------------------------------------
// Built-in variables — accepted by cu-bridge as-is
// ---------------------------------------------------------------------------
#define TUTTI_THREAD_IDX_X        threadIdx.x
#define TUTTI_BLOCK_IDX_X         blockIdx.x
#define TUTTI_BLOCK_DIM_X         blockDim.x

// ---------------------------------------------------------------------------
// Memory fences — __threadfence_system() confirmed working by Mooncake
// ---------------------------------------------------------------------------
#define TUTTI_THREADFENCE_SYSTEM  __threadfence_system()
#define TUTTI_THREADFENCE         __threadfence()
#define TUTTI_SYNC_THREADS        __syncthreads()

// ---------------------------------------------------------------------------
// Atomics — block-scope atomicAdd confirmed working by Mooncake
// ---------------------------------------------------------------------------
#define TUTTI_ATOMIC_ADD(ptr,v)   atomicAdd((ptr),(v))
#define TUTTI_ATOMIC_CAS(ptr,exp,des) atomicCAS((ptr),(exp),(des))

// ---------------------------------------------------------------------------
// Warp intrinsics — accepted by cu-bridge as-is
// ---------------------------------------------------------------------------
#define TUTTI_SHFL_SYNC(mask,val,src,lane) __shfl_sync((mask),(val),(src),(lane))
#define TUTTI_BALLOT_SYNC(mask,pred)       __ballot_sync((mask),(pred))

// ---------------------------------------------------------------------------
// Misc — accepted by cu-bridge as-is
// ---------------------------------------------------------------------------
#define TUTTI_NANOSLEEP(ns)       __nanosleep((ns))
#define TUTTI_CLOCK              clock()
#define TUTTI_CLOCK64            clock64()

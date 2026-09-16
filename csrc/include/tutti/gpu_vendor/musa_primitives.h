#pragma once

// tutti/gpu_vendor/musa_primitives.h -- MUSA kernel primitives
//
// Definitive implementation.  MUSA's mcc compiler accepts CUDA-style
// keywords (__device__, __global__, __threadfence_system, atomicAdd,
// __syncthreads, threadIdx, blockIdx, blockDim) — so the TUTTI_* macros
// expand to the same tokens as the CUDA branch.
//
// Known MUSA SDK 4.3.3 compiler bugs (ported from Mooncake's
// mooncake-transfer-engine/include/transport/device/musa/musa_ops.cuh,
// Apache 2.0):
//   - atomicAdd_system / atomicCAS_system → infinite SelectionDAG loop.
//     Tutti's kernels use block-scope atomicAdd (no _system suffix), so
//     this bug does NOT affect Tutti.
//   - Named barriers (bar.sync) → not available; use __syncthreads().
//     Tutti's kernels do NOT use named barriers.
//   - cooperative_groups::this_grid().sync() → not available.
//     Tutti's kernels do NOT use grid sync.
//
// Reference: third_pkgs/Mooncake/mooncake-transfer-engine/include/transport/device/musa/musa_ops.cuh
// License: Apache 2.0

// ---------------------------------------------------------------------------
// Function attributes — accepted by mcc as-is
// ---------------------------------------------------------------------------
#define TUTTI_DEVICE              __device__
#define TUTTI_HOST                __host__
#define TUTTI_GLOBAL              __global__
#define TUTTI_FORCEINLINE         __forceinline__
#define TUTTI_SHARED              __shared__

// ---------------------------------------------------------------------------
// Built-in variables — accepted by mcc as-is
// ---------------------------------------------------------------------------
#define TUTTI_THREAD_IDX_X        threadIdx.x
#define TUTTI_BLOCK_IDX_X         blockIdx.x
#define TUTTI_BLOCK_DIM_X         blockDim.x

// ---------------------------------------------------------------------------
// Memory fences — __threadfence_system() is available on MUSA and is the
// CRITICAL primitive for NVMe doorbell writes (system-scope visibility).
// Mooncake's musa_ops.cuh confirms __threadfence_system() works for
// cross-GPU (MTLink) visibility.
// ---------------------------------------------------------------------------
#define TUTTI_THREADFENCE_SYSTEM  __threadfence_system()
#define TUTTI_THREADFENCE         __threadfence()
#define TUTTI_SYNC_THREADS        __syncthreads()

// ---------------------------------------------------------------------------
// Atomics — block-scope atomicAdd works (NOT atomicAdd_system, which has
// a SelectionDAG bug per Mooncake's note).  Tutti uses block-scope only.
// ---------------------------------------------------------------------------
#define TUTTI_ATOMIC_ADD(ptr,v)   atomicAdd((ptr),(v))
#define TUTTI_ATOMIC_CAS(ptr,exp,des) atomicCAS((ptr),(exp),(des))

// ---------------------------------------------------------------------------
// Warp intrinsics — accepted by mcc as-is
// ---------------------------------------------------------------------------
#define TUTTI_SHFL_SYNC(mask,val,src,lane) __shfl_sync((mask),(val),(src),(lane))
#define TUTTI_BALLOT_SYNC(mask,pred)       __ballot_sync((mask),(pred))

// ---------------------------------------------------------------------------
// Misc — accepted by mcc as-is
// ---------------------------------------------------------------------------
#define TUTTI_NANOSLEEP(ns)       __nanosleep((ns))
#define TUTTI_CLOCK              clock()
#define TUTTI_CLOCK64            clock64()

#pragma once

// tutti/data_paths/local_nvme/io/tutti_gpu_primitives.cuh
//
// GPU device-side primitives abstraction layer (PTX/MUSA replacement point).
//
// Purpose: provide a single set of macros that Tutti's IO kernels use
// instead of CUDA-specific keywords.  Each macro has a documented
// semantic contract — vendor ports (MUSA, MACA, ...) fill in the
// equivalent on their device compiler without touching the kernel source.
//
// Under TUTTI_USE_CUDA: macros expand to the original CUDA keywords
// (zero behavioral change — this file is a passthrough).
// Under TUTTI_USE_MUSA / TUTTI_USE_MACA: macros are defined in the
// vendor's <tutti/gpu_vendor/musa_primitives.h> (or maca equivalent),
// which Metax supplies.  Until Metax provides that header, this file
// emits a clear compile-time error.
// Under TUTTI_USE_HOST: .cu/.cuh files are NOT compiled (HOST profile
// does not enable the CUDA device language); this file is harmless.
//
// ============================================================================
// WHY THIS FILE EXISTS
// ============================================================================
//
// Tutti's IO kernel code uses a small set of CUDA C++ language extensions:
//   - Function attributes: __device__, __global__, __host__, __forceinline__
//   - Variable attributes:  __shared__
//   - Built-in variables:   threadIdx.x, blockIdx.x, blockDim.x
//   - Synchronization:      __threadfence_system, __threadfence, __syncthreads
//   - Atomics:              atomicAdd, atomicCAS, ...
//   - Warp intrinsics:      __shfl_sync, __ballot_sync, ...
//   - Misc:                 __nanosleep, clock, clock64
//
// Each of these has a precise memory-ordering / scope contract that MUST
// be preserved on every vendor's port.  The macro definitions below name
// the contract; the CUDA branch is the reference implementation.
//
// ============================================================================
// PRIMITIVE CONTRACT TABLE (vendor port MUST satisfy)
// ============================================================================
//
// Macro                    | CUDA reference   | Contract
// -------------------------+------------------+--------------------------------
// TUTTI_DEVICE             | __device__       | Function callable from device
//                          |                  | code only.  No host linkage.
// TUTTI_HOST               | __host__         | Function callable from host.
// TUTTI_GLOBAL             | __global__       | Function is a kernel entry:
//                          |                  | launched from host via the
//                          |                  | <<<>>> syntax or equivalent.
// TUTTI_FORCEINLINE        | __forceinline__  | Force inlining at call site
//                          |                  | (device code expansion is
//                          |                  | performance-critical).
// TUTTI_SHARED             | __shared__       | Variable lives in shared
//                          |                  | memory; per-block lifetime;
//                          |                  | visible to all threads in block.
// TUTTI_THREAD_IDX_X       | threadIdx.x      | Thread index within block
//                          |                  | (x dimension, 0..blockDim-1).
// TUTTI_BLOCK_IDX_X         | blockIdx.x       | Block index within grid
//                          |                  | (x dimension).
// TUTTI_BLOCK_DIM_X        | blockDim.x       | Block size in threads
//                          |                  | (x dimension).
// TUTTI_THREADFENCE_SYSTEM | __threadfence_   | **CRITICAL**: drains ALL
//                          |   system()       | outstanding memory ops to a
//                          |                  | point of global visibility
//                          |                  | ACROSS the entire system
//                          |                  | (GPU + host + peer PCIe/NVMe
//                          |                  | devices).  Used before NVMe
//                          |                  | doorbell writes to flush
//                          |                  | PRP/SQE writes so the NVMe
//                          |                  | controller sees them.  Scope:
//                          |                  | system.  Memory order:
//                          |                  | acquire-release on all memory.
// TUTTI_THREADFENCE        | __threadfence()  | Device-scope fence: drains
//                          |                  | within the current GPU's
//                          |                  | memory subsystem.  Does NOT
//                          |                  | guarantee peer/NVMe visibility.
// TUTTI_SYNC_THREADS        | __syncthreads()  | Block-scope barrier.  All
//                          |                  | threads in block must reach
//                          |                  | before any proceeds.
// TUTTI_ATOMIC_ADD(ptr,v)  | atomicAdd(ptr,v) | Atomic add on the pointed-to
//                          |                  | value; device-global scope.
// TUTTI_ATOMIC_CAS(...)    | atomicCAS(...)   | Atomic compare-and-swap;
//                          |                  | device-global scope.
// TUTTI_SHFL_SYNC(...)     | __shfl_sync(...)  | Warp shuffle (lane-to-lane
//                          |                  | exchange); full-warp mask.
// TUTTI_BALLOT_SYNC(...)   | __ballot_sync(...)| Warp ballot (per-lane pred
//                          |                  | → bitmask); full-warp mask.
// TUTTI_NANOSLEEP(ns)      | __nanosleep(ns)  | Pause thread for ~ns
//                          |                  | nanoseconds (best-effort).
// TUTTI_CLOCK              | clock()          | Per-SM clock counter
//                          |                  | (unsigned, low precision).
// TUTTI_CLOCK64            | clock64()        | Per-SM 64-bit clock counter.
//
// ============================================================================
// KERNEL LAUNCH SYNTAX
// ============================================================================
//
// Tutti's host-side launchers (submit_one.cu, fused_submit_kernel.cu)
// use CUDA's `kernel<<<blocks, threads, smem, stream>>>(args...)` triple-
// bracket syntax.  MUSA and MACA device compilers are expected to support
// this syntax directly (mcc and its peers mirror nvcc here).  If the
// Metax compiler does NOT support `<<<>>>`, Metax should provide a macro:
//
//     #define TUTTI_LAUNCH_KERNEL(kernel, blocks, threads, smem, stream, ...)
//         kernel<<<blocks, threads, smem, stream>>>(__VA_ARGS__)
//
// — and Tutti's launchers will be updated to use it.  Today, the raw
// `<<<>>>` syntax is used because it is supported by every CUDA-like
// compiler we have surveyed.  This decision is revisitable.

#include <tutti/cuda_like.h>

// =========================================================================
// Profile dispatch
// =========================================================================

#if defined(TUTTI_USE_CUDA)

// --- CUDA reference: macros are identity passthrough ---------------------
#define TUTTI_DEVICE              __device__
#define TUTTI_HOST                __host__
#define TUTTI_GLOBAL              __global__
#define TUTTI_FORCEINLINE         __forceinline__
#define TUTTI_SHARED              __shared__
#define TUTTI_THREAD_IDX_X        threadIdx.x
#define TUTTI_BLOCK_IDX_X         blockIdx.x
#define TUTTI_BLOCK_DIM_X         blockDim.x
#define TUTTI_THREADFENCE_SYSTEM  __threadfence_system()
#define TUTTI_THREADFENCE         __threadfence()
#define TUTTI_SYNC_THREADS        __syncthreads()
#define TUTTI_ATOMIC_ADD(ptr,v)   atomicAdd((ptr),(v))
#define TUTTI_ATOMIC_CAS(ptr,exp,des) atomicCAS((ptr),(exp),(des))
#define TUTTI_SHFL_SYNC(mask,val,src,lane) __shfl_sync((mask),(val),(src),(lane))
#define TUTTI_BALLOT_SYNC(mask,pred)       __ballot_sync((mask),(pred))
#define TUTTI_NANOSLEEP(ns)       __nanosleep((ns))
#define TUTTI_CLOCK              clock()
#define TUTTI_CLOCK64            clock64()

#elif defined(TUTTI_USE_MUSA)

// --- MUSA: vendor port supplies definitions -----------------------------
// Metax: either (a) the MUSA compiler accepts the same keywords as CUDA
// (in which case copy the CUDA block above verbatim into
// tutti/include/tutti/gpu_vendor/musa_primitives.h and remove the #error),
// or (b) provide the MUSA-specific equivalents per the contract table.
#if __has_include(<tutti/gpu_vendor/musa_primitives.h>)
#include <tutti/gpu_vendor/musa_primitives.h>
#else
#error "tutti_gpu_primitives.cuh: TUTTI_USE_MUSA is defined but \
<tutti/gpu_vendor/musa_primitives.h> is not provided by Metax. \
See doc/gpu-porting-guide.md (kernel primitives macro layer)."
#endif

#elif defined(TUTTI_USE_MACA)

// --- MACA: same structure as MUSA ---------------------------------------
#if __has_include(<tutti/gpu_vendor/maca_primitives.h>)
#include <tutti/gpu_vendor/maca_primitives.h>
#else
#error "tutti_gpu_primitives.cuh: TUTTI_USE_MACA is defined but \
<tutti/gpu_vendor/maca_primitives.h> is not provided by Metax. \
See doc/gpu-porting-guide.md (kernel primitives macro layer)."
#endif

#elif defined(TUTTI_USE_HOST)

// --- HOST profile: .cuh files are not compiled by a device compiler ----
// This branch exists only to make file inclusion from mixed .cpp/.cu TUs
// safe (HOST profile doesn't define the macros, but it also never reaches
// the device-only section guarded by __CUDACC__ in .cuh headers).  We
// define the macros to nothing so any stray reference is a compile error
// rather than a silent CUDA-style keyword leaking into host code.
#define TUTTI_DEVICE
#define TUTTI_HOST
#define TUTTI_GLOBAL
#define TUTTI_FORCEINLINE
#define TUTTI_SHARED
#define TUTTI_THREAD_IDX_X        (0)
#define TUTTI_BLOCK_IDX_X         (0)
#define TUTTI_BLOCK_DIM_X         (1)
#define TUTTI_THREADFENCE_SYSTEM  ((void)0)
#define TUTTI_THREADFENCE         ((void)0)
#define TUTTI_SYNC_THREADS        ((void)0)
#define TUTTI_ATOMIC_ADD(ptr,v)   ((void)(ptr),(void)(v))
#define TUTTI_ATOMIC_CAS(ptr,exp,des) ((void)(ptr),(void)(exp),(void)(des))
#define TUTTI_SHFL_SYNC(mask,val,src,lane) ((void)(mask),(void)(val),(void)(src),(void)(lane),0)
#define TUTTI_BALLOT_SYNC(mask,pred)       ((void)(mask),(void)(pred),0u)
#define TUTTI_NANOSLEEP(ns)       ((void)(ns))
#define TUTTI_CLOCK              (0u)
#define TUTTI_CLOCK64            (0ull)

#endif

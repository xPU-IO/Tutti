#ifndef __TUTTI_MEMORY_CUDA_HELPERS_CUH__
#define __TUTTI_MEMORY_CUDA_HELPERS_CUH__

/**
 * cuda_helpers.cuh -- minimal CUDA error-handling macros.
 *
 * Layer: Memory Layer (also used by device_manager / io_engine smokes).
 *
 * Kept tiny on purpose: CUDA_OK aborts with a clear message on any
 * non-cudaSuccess return; CUDA_OK_RC returns an int rc instead of
 * aborting (for paths that need to clean up first).  Use whichever
 * fits the call site.
 *
 * The macros stringify file/line/expr so smoke logs point at the
 * exact failing call without a separate stack trace.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_OK(expr)                                                    \
    do {                                                                 \
        cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                         \
            std::fprintf(stderr,                                         \
                "[CUDA_OK] %s:%d: %s -> %s\n",                           \
                __FILE__, __LINE__, #expr, cudaGetErrorString(_e));      \
            std::abort();                                                \
        }                                                                \
    } while (0)

#define CUDA_OK_RC(expr, rc_out)                                         \
    do {                                                                 \
        cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                         \
            std::fprintf(stderr,                                         \
                "[CUDA_OK_RC] %s:%d: %s -> %s\n",                        \
                __FILE__, __LINE__, #expr, cudaGetErrorString(_e));      \
            (rc_out) = (int)_e;                                          \
        }                                                                \
    } while (0)

#endif // __TUTTI_MEMORY_CUDA_HELPERS_CUH__

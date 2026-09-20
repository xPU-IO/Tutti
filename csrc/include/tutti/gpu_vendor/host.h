#pragma once

// tutti/gpu_vendor/host.h -- HOST profile CUDA-like shim
//
// Header-only inline implementation of a minimal CUDA-compatible API backed
// by host memory (std::malloc / std::memcpy / std::memset).  Intended for
// hardware-free contract tests, NOT production use.

#include <cstdlib>
#include <cstring>
#include <cstddef>

// =========================================================================
// Error type and codes
// =========================================================================

enum cudaError_t {
    cudaSuccess           = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorNotSupported = 2,
    cudaErrorNotReady     = 3
};

// =========================================================================
// Opaque stream / event handles (heap-allocated tokens)
// =========================================================================

struct _tutti_host_stream { int _placeholder; };
struct _tutti_host_event  { int _placeholder; };

typedef struct _tutti_host_stream *cudaStream_t;
typedef struct _tutti_host_event  *cudaEvent_t;

// =========================================================================
// Enumeration types
// =========================================================================

enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4
};

enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost         = 1,
    cudaMemoryTypeDevice       = 2,
    cudaMemoryTypeManaged      = 3
};

// =========================================================================
// Pointer attributes
// =========================================================================

struct cudaPointerAttributes {
    cudaMemoryType type;
    int            device;
    void          *devicePointer;
    void          *hostPointer;
};

// =========================================================================
// Stream / event flags
// =========================================================================

#define cudaStreamDefault      0
#define cudaStreamNonBlocking  1
#define cudaEventDefault       0
#define cudaEventDisableTiming 2

// =========================================================================
// Device management
// =========================================================================

inline cudaError_t cudaGetDeviceCount(int *count) {
    if (!count) return cudaErrorInvalidValue;
    *count = 1;
    return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int device) {
    if (device < 0) return cudaErrorInvalidValue;
    return cudaSuccess;
}

inline cudaError_t cudaGetDevice(int *device) {
    if (!device) return cudaErrorInvalidValue;
    *device = 0;
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize(void) {
    // HOST profile: all operations are synchronous, nothing to wait for.
    return cudaSuccess;
}

// Minimal device properties struct (matches CUDA's field subset used by
// contract tests; unused fields are omitted).
struct cudaDeviceProp {
    char          name[256];
    int           major;
    int           minor;
    size_t        totalGlobalMem;
    int           multiProcessorCount;
    int           maxThreadsPerBlock;
};

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    if (!prop) return cudaErrorInvalidValue;
    if (device < 0) return cudaErrorInvalidValue;
    // HOST shim: report a synthetic "device" with conservative defaults.
    std::memset(prop, 0, sizeof(*prop));
    static const char kHostName[] = "Tutti HOST shim";
    std::memcpy(prop->name, kHostName, sizeof(kHostName));
    prop->major              = 0;
    prop->minor              = 0;
    prop->totalGlobalMem     = 0;
    prop->multiProcessorCount = 0;
    prop->maxThreadsPerBlock = 1;
    return cudaSuccess;
}

// =========================================================================
// Memory allocation
// =========================================================================

inline cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (!ptr) return cudaErrorInvalidValue;
    *ptr = std::malloc(size);
    if (!*ptr && size > 0) return cudaErrorInvalidValue;
    return cudaSuccess;
}

inline cudaError_t cudaFree(void *ptr) {
    if (!ptr) return cudaErrorInvalidValue;
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMallocHost(void **ptr, size_t size) {
    if (!ptr) return cudaErrorInvalidValue;
    *ptr = std::malloc(size);
    if (!*ptr && size > 0) return cudaErrorInvalidValue;
    return cudaSuccess;
}

inline cudaError_t cudaFreeHost(void *ptr) {
    if (!ptr) return cudaErrorInvalidValue;
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
    (void)size;
    (void)flags;
    if (!ptr) return cudaErrorInvalidValue;
    return cudaErrorNotSupported;
}

inline cudaError_t cudaHostUnregister(void *ptr) {
    if (!ptr) return cudaErrorInvalidValue;
    return cudaErrorNotSupported;
}

inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes,
                                            const void *ptr) {
    if (!attributes) return cudaErrorInvalidValue;
    if (!ptr) return cudaErrorInvalidValue;
    attributes->type          = cudaMemoryTypeHost;
    attributes->device        = 0;
    attributes->devicePointer = nullptr;
    attributes->hostPointer   = const_cast<void *>(ptr);
    return cudaSuccess;
}

// =========================================================================
// Stream management
// =========================================================================

inline cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    if (!stream) return cudaErrorInvalidValue;
    *stream = static_cast<cudaStream_t>(
        std::malloc(sizeof(struct _tutti_host_stream)));
    if (!*stream) return cudaErrorInvalidValue;
    return cudaSuccess;
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream,
                                             unsigned int flags) {
    (void)flags;
    return cudaStreamCreate(stream);
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (!stream) return cudaErrorInvalidValue;
    std::free(stream);
    return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    if (!stream) return cudaErrorInvalidValue;
    return cudaSuccess;
}

inline cudaError_t cudaStreamQuery(cudaStream_t stream) {
    if (!stream) return cudaErrorInvalidValue;
    // HOST streams are synchronous — all work is already complete.
    return cudaSuccess;
}

// =========================================================================
// Event management
// =========================================================================

inline cudaError_t cudaEventCreate(cudaEvent_t *event) {
    if (!event) return cudaErrorInvalidValue;
    *event = static_cast<cudaEvent_t>(
        std::malloc(sizeof(struct _tutti_host_event)));
    if (!*event) return cudaErrorInvalidValue;
    return cudaSuccess;
}

inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
                                            unsigned int flags) {
    (void)flags;
    return cudaEventCreate(event);
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (!event) return cudaErrorInvalidValue;
    std::free(event);
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (!event) return cudaErrorInvalidValue;
    (void)stream;
    return cudaSuccess;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!event) return cudaErrorInvalidValue;
    return cudaSuccess;
}

inline cudaError_t cudaEventQuery(cudaEvent_t event) {
    if (!event) return cudaErrorInvalidValue;
    return cudaSuccess;
}

inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                       unsigned int flags) {
    if (!stream) return cudaErrorInvalidValue;
    if (!event)  return cudaErrorInvalidValue;
    (void)flags;
    return cudaSuccess;
}

// =========================================================================
// Memory copy
// =========================================================================

inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                              cudaMemcpyKind kind) {
    (void)kind;
    if (count == 0) return cudaSuccess;
    if (!dst || !src) return cudaErrorInvalidValue;
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                   cudaMemcpyKind kind, cudaStream_t stream) {
    (void)stream;
    return cudaMemcpy(dst, src, count, kind);
}

// =========================================================================
// Memory set
// =========================================================================

inline cudaError_t cudaMemset(void *ptr, int value, size_t count) {
    if (count == 0) return cudaSuccess;
    if (!ptr) return cudaErrorInvalidValue;
    std::memset(ptr, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemsetAsync(void *ptr, int value, size_t count,
                                   cudaStream_t stream) {
    (void)stream;
    return cudaMemset(ptr, value, count);
}

// =========================================================================
// Error handling
// =========================================================================

inline const char *cudaGetErrorString(cudaError_t err) {
    switch (err) {
        case cudaSuccess:           return "success";
        case cudaErrorInvalidValue: return "invalid value";
        case cudaErrorNotSupported: return "not supported";
        case cudaErrorNotReady:     return "not ready";
        default:                    return "unknown error";
    }
}

inline cudaError_t cudaGetLastError()     { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError()  { return cudaSuccess; }

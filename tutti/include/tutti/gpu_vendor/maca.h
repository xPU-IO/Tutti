#pragma once

// tutti/gpu_vendor/maca.h -- MACA profile (Metax MACA SDK)
//
// Definitive implementation ported from Mooncake (Apache 2.0).
// Mooncake's mooncake-transfer-engine/include/gpu_vendor/maca.h has been
// validated against the MACA SDK (cu-bridge compiler) in production.
// Tutti adopts it verbatim (minus Mooncake-specific GPU_PREFIX and
// IBGDA driver API mappings that Tutti does not use).
//
// Reference: third_pkgs/Mooncake/mooncake-transfer-engine/include/gpu_vendor/maca.h
// License: Apache 2.0 (see third_pkgs/Mooncake/LICENSE-APACHE)

#include <cstddef>
#include <mcr/maca.h>
#include <mcr/mc_runtime.h>
#include <mcr/mc_runtime_api.h>
#include <cuda/atomic>

// ---------------------------------------------------------------------------
// Runtime API macro mapping (cuda* -> mc*)
// ---------------------------------------------------------------------------
// maca's cu-bridge already defines CUDA/CU compatibility macros after
// ATen/cuda headers are included. Keep this fallback mapping for translation
// units that include Mooncake's cuda_alike.h without cu-bridge first.
#ifndef __CUDA_TO_MACA_ADAPTOR_H__
#define cudaError_t mcError_t
#define cudaSuccess mcSuccess
#define cudaErrorNotReady mcErrorNotReady
#define cudaErrorPeerAccessAlreadyEnabled mcErrorPeerAccessAlreadyEnabled

#define cudaFree mcFree
#define cudaFreeHost mcFreeHost
#define cudaGetDevice mcGetDevice
#define cudaGetDeviceCount mcGetDeviceCount
#define cudaGetErrorString mcGetErrorString
#define cudaGetLastError mcGetLastError
#define cudaHostRegister mcHostRegister
#define cudaHostRegisterPortable mcHostRegisterPortable
#define cudaHostRegisterMapped mcHostRegisterMapped
#define cudaHostUnregister mcHostUnregister
#define cudaIpcCloseMemHandle mcIpcCloseMemHandle
#define cudaIpcGetMemHandle mcIpcGetMemHandle
#define cudaIpcMemHandle_t mcIpcMemHandle_t
#define cudaIpcMemLazyEnablePeerAccess mcIpcMemLazyEnablePeerAccess
#define cudaIpcOpenMemHandle mcIpcOpenMemHandle
#define cudaMalloc mcMalloc
#define cudaMallocHost mcMallocHost
#define cudaMemcpy mcMemcpy
#define cudaMemcpyAsync mcMemcpyAsync
#define cudaMemcpyDefault mcMemcpyDefault
#define cudaMemcpyDeviceToHost mcMemcpyDeviceToHost
#define cudaMemcpyHostToDevice mcMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice mcMemcpyDeviceToDevice
#define cudaMemcpyKind mcMemcpyKind
#define cudaMemcpyPeerAsync mcMemcpyPeerAsync
#define cudaMemset mcMemset
#define cudaMemsetAsync mcMemsetAsync
#define cudaMemoryTypeDevice mcMemoryTypeDevice
#define cudaMemoryTypeHost mcMemoryTypeHost
#define cudaMemoryTypeUnregistered mcMemoryTypeUnregistered
#define cudaPointerAttributes mcPointerAttribute_t
#define cudaPointerGetAttributes mcPointerGetAttributes
#define cudaSetDevice mcSetDevice
#define cudaStreamCreate mcStreamCreate
#define cudaStreamCreateWithFlags mcStreamCreateWithFlags
#define cudaStreamNonBlocking mcStreamNonBlocking
#define cudaStreamDestroy mcStreamDestroy
#define cudaStreamPerThread mcStreamPerThread
#define cudaStreamQuery mcStreamQuery
#define cudaStreamSynchronize mcStreamSynchronize
#define cudaStream_t mcStream_t
#define cudaStreamWaitEvent mcStreamWaitEvent
#define cudaDeviceCanAccessPeer mcDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess mcDeviceEnablePeerAccess
#define cudaDeviceGetPCIBusId mcDeviceGetPCIBusId
#define cudaDeviceGetAttribute mcDeviceGetAttribute
#define cudaDeviceSynchronize mcDeviceSynchronize
#define cudaDevAttrMultiProcessorCount mcDeviceAttributeMultiProcessorCount
#define cudaEvent_t mcEvent_t
#define cudaEventCreate mcEventCreate
#define cudaEventCreateWithFlags mcEventCreateWithFlags
#define cudaEventDisableTiming mcEventDisableTiming
#define cudaEventDestroy mcEventDestroy
#define cudaEventElapsedTime mcEventElapsedTime
#define cudaEventQuery mcEventQuery
#define cudaEventRecord mcEventRecord
#define cudaEventSynchronize mcEventSynchronize
#define cudaFuncAttributes mcFuncAttributes
#define cudaHostAlloc mcMallocHost
#define cudaHostAllocDefault mcMallocHostDefault
#define cudaHostAllocMapped mcMallocHostMapped
#define cudaHostAllocPortable mcMallocHostPortable
#define cudaHostAllocWriteCombined mcMallocHostWriteCombined
#define cudaHostRegisterIoMemory mcHostRegisterIoMemory
#define cudaErrorUnknown mcErrorUnknown
#define cudaDeviceGetStreamPriorityRange mcDeviceGetStreamPriorityRange
#define cudaStreamCreateWithPriority mcStreamCreateWithPriority
#define cudaStreamDefault mcStreamDefault
// ---------------------------------------------------------------------------
// Templated wrappers that MACA's cu-bridge provides as functions (not macros)
// ---------------------------------------------------------------------------
#ifdef __cplusplus
template <class T>
static inline mcError_t cudaFuncGetAttributes(mcFuncAttributes *attr, T func) {
    return mcFuncGetAttributes(attr, reinterpret_cast<const void *>(func));
}

template <class T>
static inline mcError_t cudaHostGetDevicePointer(T **dev_ptr, void *host_ptr,
                                                 unsigned int flags) {
    return mcHostGetDevicePointer(reinterpret_cast<void **>(dev_ptr), host_ptr,
                                  flags);
}
#else
static inline mcError_t cudaHostGetDevicePointer(void **dev_ptr, void *host_ptr,
                                                 unsigned int flags) {
    return mcHostGetDevicePointer(dev_ptr, host_ptr, flags);
}
#endif
#endif

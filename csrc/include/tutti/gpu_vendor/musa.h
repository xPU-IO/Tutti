#pragma once

// tutti/gpu_vendor/musa.h -- MUSA profile (Metax)
//
// Definitive implementation ported from Mooncake (Apache 2.0).
// Mooncake's mooncake-transfer-engine/include/gpu_vendor/musa.h has been
// validated against the MUSA SDK in production.  Tutti adopts it verbatim
// (minus Mooncake-specific GPU_PREFIX / IBGDA driver API mappings that
// Tutti does not use).
//
// Reference: third_pkgs/Mooncake/mooncake-transfer-engine/include/gpu_vendor/musa.h
// License: Apache 2.0 (see third_pkgs/Mooncake/LICENSE-APACHE)

#include <musa.h>
#include <musa_runtime.h>

// ---------------------------------------------------------------------------
// Runtime API macro mapping (cuda* -> musa*)
// ---------------------------------------------------------------------------
#define cudaError_t musaError_t
#define cudaSuccess musaSuccess
#define cudaErrorNotReady musaErrorNotReady
#define cudaErrorPeerAccessAlreadyEnabled musaErrorPeerAccessAlreadyEnabled

#define cudaFree musaFree
#define cudaFreeHost musaFreeHost
#define cudaGetDevice musaGetDevice
#define cudaGetDeviceCount musaGetDeviceCount
#define cudaGetErrorString musaGetErrorString
#define cudaGetLastError musaGetLastError
#define cudaHostRegister musaHostRegister
#define cudaHostRegisterPortable musaHostRegisterPortable
#define cudaHostUnregister musaHostUnregister
#define cudaIpcCloseMemHandle musaIpcCloseMemHandle
#define cudaIpcGetMemHandle musaIpcGetMemHandle
#define cudaIpcMemHandle_t musaIpcMemHandle_t
#define cudaIpcMemLazyEnablePeerAccess musaIpcMemLazyEnablePeerAccess
#define cudaIpcOpenMemHandle musaIpcOpenMemHandle
#define cudaMalloc musaMalloc
#define cudaMallocHost musaMallocHost
#define cudaMemcpy musaMemcpy
#define cudaMemcpyAsync musaMemcpyAsync
#define cudaMemcpyDefault musaMemcpyDefault
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#define cudaMemcpyKind musaMemcpyKind
#define cudaMemset musaMemset
#define cudaMemsetAsync musaMemsetAsync
#define cudaMemoryTypeDevice musaMemoryTypeDevice
#define cudaMemoryTypeHost musaMemoryTypeHost
#define cudaMemoryTypeUnregistered musaMemoryTypeUnregistered
#define cudaPointerAttributes musaPointerAttributes
#define cudaPointerGetAttributes musaPointerGetAttributes
#define cudaSetDevice musaSetDevice
#define cudaStreamCreate musaStreamCreate
#define cudaStreamCreateWithFlags musaStreamCreateWithFlags
#define cudaStreamNonBlocking musaStreamNonBlocking
#define cudaStreamDestroy musaStreamDestroy
#define cudaStreamPerThread musaStreamPerThread
#define cudaStreamQuery musaStreamQuery
#define cudaStreamWaitEvent musaStreamWaitEvent
#define cudaDeviceSynchronize musaDeviceSynchronize
#define cudaStreamSynchronize musaStreamSynchronize
#define cudaStream_t musaStream_t
#define cudaDeviceCanAccessPeer musaDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess musaDeviceEnablePeerAccess
#define cudaDeviceGetPCIBusId musaDeviceGetPCIBusId
#define cudaDeviceGetAttribute musaDeviceGetAttribute
#define cudaEvent_t musaEvent_t
#define cudaEventCreateWithFlags musaEventCreateWithFlags
#define cudaEventDisableTiming musaEventDisableTiming
#define cudaEventDestroy musaEventDestroy
#define cudaEventRecord musaEventRecord
#define cudaEventSynchronize musaEventSynchronize
#define cudaDeviceProp musaDeviceProp
#define cudaGetDeviceProperties musaGetDeviceProperties
#define cudaDevAttrClockRate musaDevAttrClockRate
#define cudaHostRegisterMapped musaHostRegisterMapped
#define cudaHostRegisterIoMemory musaHostRegisterIoMemory
#define cudaHostGetDevicePointer musaHostGetDevicePointer

// Launch API (Tutti uses <<<>>> syntax, but expose for completeness)
#define cudaLaunchConfig_t musaLaunchConfig_t
#define cudaLaunchAttribute musaLaunchAttribute
#define cudaLaunchAttributeCooperative musaLaunchAttributeCooperative
#define cudaLaunchKernelEx musaLaunchKernelEx

#ifndef __BENCHMARK_BUFFER_H__
#define __BENCHMARK_BUFFER_H__


// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <memory>
#include <cstddef>
#include <cstdint>
#include <tutti/cuda_like.h>
#include "nvm_types.h"
#include "nvm_dma.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <new>
#include <cstdlib>
#include <iostream>
#include "util.h"

using error = std::runtime_error;
using std::string;



typedef std::shared_ptr<nvm_dma_t> DmaPtr;

typedef std::shared_ptr<void> BufferPtr;



DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size);


DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice);


DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t adapter, uint32_t id);


DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice, uint32_t adapter, uint32_t id);

// HYF
DmaPtr createDma(const nvm_ctrl_t* ctrl, void* buffer, size_t size,  int cudaDevice);

DmaPtr getDeviceDma(const nvm_ctrl_t* ctrl, void* buffer, size_t size,  int cudaDevice);



BufferPtr createBuffer(size_t size);


BufferPtr createBuffer(size_t size, int cudaDevice);

static void getDeviceMemory(int device, void*& bufferPtr, void*& devicePtr, size_t size, void*& origPtr)
{
    bufferPtr = nullptr;
    devicePtr = nullptr;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }
    size += 64*1024; // 猜测Bam这么写是为了避免分配小内存页，实现64KB对齐
    //std::cout << "DMA Size: "<< size << std::endl;
    err = cudaMalloc(&bufferPtr, size);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate device memory: ") + cudaGetErrorString(err));
    }
    /*
    err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }
    */

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, bufferPtr);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to get pointer attributes: ") + cudaGetErrorString(err));
    }

    origPtr = bufferPtr;
    //devicePtr = (void*) (((uint64_t)attrs.devicePointer));
    devicePtr = (void*) ((((uint64_t)attrs.devicePointer) + (64*1024)) & 0xffffffffff0000);
    bufferPtr = (void*) ((((uint64_t)bufferPtr) + (64*1024))  & 0xffffffffff0000);
}

[[maybe_unused]]
static void getDeviceMemory3(int device, void*& bufferPtr, void*& devicePtr, [[maybe_unused]]size_t size)
{

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, bufferPtr);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to get pointer attributes: ") + cudaGetErrorString(err));
    }
    if (attrs.type != cudaMemoryTypeDevice)
    {
        throw error("Buffer is not a valid CUDA device pointer.");
    }
    //devicePtr = (void*) (((uint64_t)attrs.devicePointer));
    devicePtr = (void*) ((((uint64_t)attrs.devicePointer)) & 0xffffffffff0000);
    bufferPtr = (void*) ((((uint64_t)bufferPtr))  & 0xffffffffff0000);
}

static void getDeviceMemory2(int device, void*& bufferPtr, size_t size, void*& origPtr)
{
    bufferPtr = nullptr;
    //devicePtr = nullptr;
    size += 128;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }
    err = cudaMalloc(&bufferPtr, size);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate device memory: ") + cudaGetErrorString(err));
    }

    err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }
/*
    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, bufferPtr);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to get pointer attributes: ") + cudaGetErrorString(err));
    }

    devicePtr = (void*) (((uint64_t)attrs.devicePointer));
*/
    origPtr = bufferPtr;
    bufferPtr = (void*) ((((uint64_t)bufferPtr) + (128))  & 0xffffffffffffe0);
    //std::cout << "getdeviceMemory: " << std::hex << bufferPtr <<  std::endl;
}

// static void getDeviceMemory(int device, void*& bufferPtr, size_t size)
// {
//     void* notUsed = nullptr;
// }

/*
static void getDeviceMemory2(int device, void*& bufferPtr, size_t size)
{
    void* notUsed = nullptr;
    getDeviceMemory2(device, bufferPtr, size);
}
*/




inline DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size)
{
    nvm_dma_t* dma = nullptr;
    void* buffer = nullptr;

    /*
    cudaError_t err = cudaHostAlloc(&buffer, size, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate host memory: ") + cudaGetErrorString(err));
    }
    */

    int err  = posix_memalign(&buffer, 4096, size);

    if (err) {
        throw error(string("Failed to allocate host memory: ") + std::to_string(err));
    }
    int status = nvm_dma_map_host(&dma, ctrl, buffer, size,-1,-1);
    if (!nvm_ok(status))
    {
        //cudaFreeHost(buffer);
        free(buffer);
        throw error(string("Failed to map host memory: ") + nvm_strerror(status));
    }

    return DmaPtr(dma, [buffer](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        //cudaFreeHost(buffer);
        free(buffer);
    });
}



inline DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice)
{
    if (cudaDevice < 0)
    {
        return createDma(ctrl, size);
    }

    nvm_dma_t* dma = nullptr;
    void* bufferPtr = nullptr;
    void* devicePtr = nullptr;
    void* origPtr = nullptr;

    getDeviceMemory(cudaDevice, bufferPtr, devicePtr, size, origPtr);

    //std::cout << "Got Device mem\n";
    int status = nvm_dma_map_device(&dma, ctrl, bufferPtr, size);
    //std::cout << "Got dma_map_devce\n";
    if (!nvm_ok(status))
    {
        //std::cout << "Got dma_map_devce failed\n";
        //cudaFree(bufferPtr);
        throw error(string("Failed to map device memory: ") + nvm_strerror(status));
    }
    cudaError_t err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }
    dma->vaddr = bufferPtr;

    return DmaPtr(dma, [bufferPtr, origPtr](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        cudaFree(origPtr);
        //std::cout << "Deleting DMA\n";
    });
}
// HYF
inline DmaPtr createDma(const nvm_ctrl_t* ctrl, void* bufferPtr, size_t size, int cudaDevice) {
    if (cudaDevice < 0) {
        return createDma(ctrl, size);
    }

    nvm_dma_t* dma = nullptr;
    void* devicePtr = nullptr;

    // 验证并获取设备指针属性
    cudaError_t err = cudaSetDevice(cudaDevice);
    if (err != cudaSuccess) {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, bufferPtr);
    if (err != cudaSuccess) {
        throw error(string("Invalid device memory pointer: ") + cudaGetErrorString(err));
    }

    if (attrs.type != cudaMemoryTypeDevice) {
        throw error("Buffer is not a valid CUDA device pointer.");
    }

    // 确保指针对齐到 64KB 边界
    // devicePtr = (void*) ((((uint64_t)attrs.devicePointer) + (64 * 1024)) & 0xffffffffff0000);
    // 映射 DMA
    std::cout << "devicePtr: " << std::hex << devicePtr << std::endl;
    int status = nvm_dma_map_device(&dma, ctrl, devicePtr, size);
    if (!nvm_ok(status)) {
        throw error(string("Failed to map device memory: ") + nvm_strerror(status));
    }

    // 对内存进行清零操作
    err = cudaMemset(devicePtr, 0, size);
    if (err != cudaSuccess) {
        nvm_dma_unmap(dma);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }

    // 确保vaddr指向对齐后的设备地址
    dma->vaddr = devicePtr;

    return DmaPtr(dma, [](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
    });
}

inline DmaPtr create_queue_Dma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice, unsigned int is_cq, uint16_t qno)
{
    if (cudaDevice < 0)
    {
        return createDma(ctrl, size);
    }

    nvm_dma_t* dma = nullptr;
    void* bufferPtr = nullptr;
    void* devicePtr = nullptr;
    void* origPtr = nullptr;

    getDeviceMemory(cudaDevice, bufferPtr, devicePtr, size, origPtr);

    // std::cout << "Got Device mem\n";
    int status = nvm_dma_map_queue_device(&dma, ctrl, bufferPtr, size,is_cq,qno);
    //std::cout << "Got dma_map_devce\n";
    if (!nvm_ok(status))
    {
        //std::cout << "Got dma_map_devce failed\n";
        //cudaFree(bufferPtr);
        throw error(string("Failed to map device memory: ") + nvm_strerror(status));
    }
    cudaError_t err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }
    dma->vaddr = bufferPtr;

    return DmaPtr(dma, [bufferPtr, origPtr](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        cudaFree(origPtr);
        //std::cout << "Deleting DMA\n";
    });
}

/*
 * Queue-group ring DMA helper.
 *
 * Successor to create_queue_Dma above.  Allocates a GPU buffer (pinned
 * for DMA + reachable from host VA via cudaHostRegister inside
 * getDeviceMemory) and registers it with the kernel as a ring of the
 * given kind (RING_SQ if is_cq=0, RING_CQ if is_cq=1) attached to the
 * supplied queue group.
 *
 * Caller MUST have allocated the group via nvm_create_group() on the
 * same fd; the kernel will -EINVAL otherwise (RING_* maps require a
 * non-zero group_id).
 *
 * Lifetime: the returned DmaPtr's deleter calls nvm_dma_unmap (which
 * is a no-op kernel-side under B6 -- ring maps are released by
 * nvm_destroy_group's cascade -- but still drops the libnvm-internal
 * refcount) and cudaFree's the GPU buffer.  The natural cleanup
 * order in Controller::~Controller is therefore:
 *   nvm_destroy_group(ctrl, group_id)   // kernel: cascade ring maps
 *   then drop the DmaPtr members        // libnvm: cudaFree + bookkeeping
 */
inline DmaPtr create_ring_Dma(const nvm_ctrl_t* ctrl,
                              size_t size,
                              int cudaDevice,
                              uint32_t group_id,
                              unsigned int is_cq)
{
    if (cudaDevice < 0) {
        // Host-resident rings are not implemented in this code path
        // yet; the legacy createDma() path was reused as a placeholder
        // for that case.  Refuse here so the caller fails loud
        // instead of silently bypassing the group/kind enforcement.
        throw error(string("create_ring_Dma: host-resident rings not "
                           "supported (cudaDevice < 0)"));
    }

    nvm_dma_t* dma = nullptr;
    void* bufferPtr = nullptr;
    void* devicePtr = nullptr;
    void* origPtr = nullptr;

    getDeviceMemory(cudaDevice, bufferPtr, devicePtr, size, origPtr);

    int status = nvm_dma_map_ring_device(&dma, ctrl, group_id,
                                         bufferPtr, size,
                                         (int)is_cq);
    if (!nvm_ok(status)) {
        cudaFree(origPtr);
        throw error(string("nvm_dma_map_ring_device failed: ")
                    + nvm_strerror(status));
    }
    cudaError_t err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess) {
        nvm_dma_unmap(dma);
        cudaFree(origPtr);
        throw error(string("Failed to clear ring memory: ")
                    + cudaGetErrorString(err));
    }
    dma->vaddr = bufferPtr;

    return DmaPtr(dma, [bufferPtr, origPtr](nvm_dma_t* d) {
        nvm_dma_unmap(d);
        cudaFree(origPtr);
    });
}


inline BufferPtr createBuffer(size_t size)
{
    void* buffer = nullptr;

    cudaError_t err = cudaHostAlloc(&buffer, size, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate host memory: ") + cudaGetErrorString(err));
    }

    return BufferPtr(buffer, [](void* ptr) {

        cudaFreeHost(ptr);
    });
}



inline BufferPtr createBuffer(size_t size, int cudaDevice)
{
    if (cudaDevice < 0)
    {
        return createBuffer(size);
    }

    void* bufferPtr = nullptr;
    void* origPtr = nullptr;

    getDeviceMemory2(cudaDevice, bufferPtr, size, origPtr);
    // std::cout << "createbuffer: " << std::hex << bufferPtr <<  std::endl;

    return BufferPtr(bufferPtr, [origPtr](void* ptr) {
        __ignore(ptr);
        cudaFree(origPtr);
        // std::cout << "Deleting Buffer\n";
    });
}

inline DmaPtr getDeviceDma(const nvm_ctrl_t* ctrl, void* buffer, size_t size, int cudaDevice) {
    nvm_dma_t* dma = nullptr;
    cudaPointerAttributes attrs;

    if (cudaDevice < 0)
    {
        throw error(string("Must be a valid CUDA device: ") + std::to_string(cudaDevice));
    }
    cudaError_t err = cudaSetDevice(cudaDevice);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }
    err = cudaPointerGetAttributes(&attrs, buffer);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to get pointer attributes: ") + cudaGetErrorString(err));
    }
    
    if (attrs.type != cudaMemoryTypeDevice)
    {
        throw error("Buffer is not a valid CUDA device pointer.");
    }
    
    int status = nvm_dma_map_device(&dma, ctrl, buffer, size);
    // std::cout << "Got dma_map_devce\n";
    if (!nvm_ok(status))
    {
        std::cout << "Got dma_map_devce failed\n";
        throw error(string("Failed to map device memory: ") + nvm_strerror(status));
    }
    dma->vaddr = buffer;

    return DmaPtr(dma, [buffer](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        // std::cout << "Deleting DMA\n";
    });
}

/*

DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size)
{
    return createDma(ctrl, size);
}



DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice)
{
    return createDma(ctrl, size, cudaDevice);
}

*/
#endif

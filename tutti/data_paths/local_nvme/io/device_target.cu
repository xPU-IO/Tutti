// tutti/data_paths/local_nvme/io/device_target.cu
//
// Device target handle build/free — ported from main's
// HostFsBackedNvmeStorage::build_handle_template_ / release path.

#include "tutti/data_paths/local_nvme/io/device_target.h"

#include <tutti/cuda_like.h>
#include <tutti/accelerator_device_guard.h>

#include <cstdio>
#include <cstring>

namespace tutti::data_paths::local_nvme {

bool build_device_target(
    const DeviceTargetHandle& host_template,
    const DeviceLbaExtent* overflow_extents,
    uint32_t n_overflow,
    uint32_t cuda_device,
    DeviceTargetHandle** out_handle,
    void** out_overflow_dev)
{
    *out_handle = nullptr;
    *out_overflow_dev = nullptr;

    DeviceGuard device_guard(static_cast<std::int32_t>(cuda_device));
    if (!device_guard.ok()) return false;

    cudaError_t cerr;

    // 1. Allocate overflow buffer (if needed) — done first so we can
    //    clean up on failure before allocating the main handle.
    void* overflow_dev = nullptr;
    if (n_overflow > 0 && overflow_extents != nullptr) {
        cerr = cudaMalloc(&overflow_dev,
            (std::size_t)n_overflow * sizeof(DeviceLbaExtent));
        if (cerr != cudaSuccess) {
            std::fprintf(stderr,
                "[device_target] cudaMalloc(overflow, %u extents): %s\n",
                n_overflow, cudaGetErrorString(cerr));
            return false;
        }
        cerr = cudaMemcpy(overflow_dev, overflow_extents,
            (std::size_t)n_overflow * sizeof(DeviceLbaExtent),
            cudaMemcpyHostToDevice);
        if (cerr != cudaSuccess) {
            std::fprintf(stderr,
                "[device_target] cudaMemcpy(overflow): %s\n",
                cudaGetErrorString(cerr));
            cudaFree(overflow_dev);
            return false;
        }
    }

    // 2. Fill the overflow pointer in the template.
    DeviceTargetHandle tmpl = host_template;
    tmpl.extents_overflow = (DeviceLbaExtent*)overflow_dev;

    // 3. Allocate device handle and H2D copy.
    DeviceTargetHandle* dev_handle = nullptr;
    cerr = cudaMalloc(&dev_handle, sizeof(DeviceTargetHandle));
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[device_target] cudaMalloc(handle): %s\n",
            cudaGetErrorString(cerr));
        if (overflow_dev) cudaFree(overflow_dev);
        return false;
    }

    cerr = cudaMemcpy(dev_handle, &tmpl, sizeof(DeviceTargetHandle),
                      cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[device_target] cudaMemcpy(handle): %s\n",
            cudaGetErrorString(cerr));
        cudaFree(dev_handle);
        if (overflow_dev) cudaFree(overflow_dev);
        return false;
    }

    *out_handle = dev_handle;
    *out_overflow_dev = overflow_dev;
    return true;
}

void free_device_target(
    DeviceTargetHandle* handle,
    void* overflow_dev,
    uint32_t cuda_device)
{
    DeviceGuard device_guard(static_cast<std::int32_t>(cuda_device));
    if (!device_guard.ok()) return;
    if (overflow_dev != nullptr) {
        cudaFree(overflow_dev);
    }
    if (handle != nullptr) {
        cudaFree(handle);
    }
}

bool snapshot_device_target(
    const DeviceTargetHandle* gpu_handle,
    const void* gpu_overflow,
    uint64_t overflow_bytes,
    uint32_t cuda_device,
    DeviceTargetHandle* out_handle_image,
    void* out_overflow_image)
{
    if (gpu_handle == nullptr || out_handle_image == nullptr) return false;
    if (overflow_bytes > 0 &&
        (gpu_overflow == nullptr || out_overflow_image == nullptr)) {
        return false;
    }
    DeviceGuard device_guard(static_cast<std::int32_t>(cuda_device));
    if (!device_guard.ok()) return false;
    cudaError_t cerr;

    cerr = cudaMemcpy(out_handle_image, gpu_handle,
                      sizeof(DeviceTargetHandle), cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) return false;

    if (overflow_bytes > 0) {
        cerr = cudaMemcpy(out_overflow_image, gpu_overflow, overflow_bytes,
                          cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) return false;
    }
    return true;
}

bool restore_device_target(
    const DeviceTargetHandle* handle_image,
    const void* overflow_image,
    uint64_t overflow_bytes,
    uint32_t cuda_device,
    DeviceTargetHandle** out_handle,
    void** out_overflow_dev)
{
    *out_handle = nullptr;
    *out_overflow_dev = nullptr;
    if (handle_image == nullptr) return false;
    if (overflow_bytes > 0 && overflow_image == nullptr) return false;
    if (overflow_bytes % sizeof(DeviceLbaExtent) != 0) return false;

    DeviceGuard device_guard(static_cast<std::int32_t>(cuda_device));
    if (!device_guard.ok()) return false;
    cudaError_t cerr;

    // 1. Overflow buffer + content restore.
    void* overflow_dev = nullptr;
    if (overflow_bytes > 0) {
        cerr = cudaMalloc(&overflow_dev, (std::size_t)overflow_bytes);
        if (cerr != cudaSuccess) return false;
        cerr = cudaMemcpy(overflow_dev, overflow_image, overflow_bytes,
                          cudaMemcpyHostToDevice);
        if (cerr != cudaSuccess) {
            cudaFree(overflow_dev);
            return false;
        }
    }

    // 2. Handle struct restore, with extents_overflow patched to the
    //    FRESH overflow allocation (the image carries a stale pointer).
    DeviceTargetHandle h = *handle_image;
    h.extents_overflow = (DeviceLbaExtent*)overflow_dev;

    DeviceTargetHandle* dev_handle = nullptr;
    cerr = cudaMalloc(&dev_handle, sizeof(DeviceTargetHandle));
    if (cerr != cudaSuccess) {
        if (overflow_dev) cudaFree(overflow_dev);
        return false;
    }
    cerr = cudaMemcpy(dev_handle, &h, sizeof(DeviceTargetHandle),
                      cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        cudaFree(dev_handle);
        if (overflow_dev) cudaFree(overflow_dev);
        return false;
    }

    *out_handle = dev_handle;
    *out_overflow_dev = overflow_dev;
    return true;
}

} // namespace tutti::data_paths::local_nvme

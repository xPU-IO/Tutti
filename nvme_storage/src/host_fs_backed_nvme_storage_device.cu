/**
 * host_fs_backed_nvme_storage_device.cu -- GPU-side bring-up + tear-
 *                                          down of NvmeFileDeviceHandle.
 *
 * Layer: nvme_storage (R5b).
 *
 * Split out from host_fs_backed_nvme_storage.cpp so the host-side
 * file (which is .cpp) doesn't have to start dragging in CUDA
 * runtime.  Everything else still lives in the .cpp file; this
 * unit only knows about cudaMalloc / cudaMemcpy / cudaFree.
 */

#include "host_fs_backed_nvme_storage.h"

#include "nvme_file.h"
#include "nvme_file_device_handle.h"
#include "nvme_file_header.h"           // sizeof(NvmeFileHeader)
#include "../../runtime/include/device.h"
#include "../../device_manager/include/local_nvme_device.h"
#include "../../device_manager/include/nvme_queue_group.h"   // R5b

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace tutti {

NvmeFileDeviceHandle* HostFsBackedNvmeStorage::acquire_device_handle(NvmeFile* file)
{
    if (file == nullptr || file->device == nullptr) return nullptr;

    auto* bp = static_cast<LocalNvmeDevice*>(file->device->backend_private);
    if (bp == nullptr) {
        std::fprintf(stderr,
            "[nvme_storage] acquire_device_handle: device has no LocalNvmeDevice "
            "payload\n");
        return nullptr;
    }
    if (!bp->queue_group) {
        std::fprintf(stderr,
            "[nvme_storage] acquire_device_handle: device(id=%d) has no queue_group. "
            "The registry that produced this device must have been opened "
            "with build_queue_group=true (works for both DIRECT and "
            "SERVICE_CLIENT registries; in service mode the client itself "
            "creates the queue group on its attach_client fd).\n",
            file->device->device_id);
        return nullptr;
    }
    NvmeQueueGroup* qg = bp->queue_group.get();
    if (qg->d_qps() == nullptr || qg->n_qps() == 0) {
        std::fprintf(stderr,
            "[nvme_storage] acquire_device_handle: queue_group has no d_qps "
            "(n_qps=%u, d_qps=%p)\n",
            (unsigned)qg->n_qps(), (void*)qg->d_qps());
        return nullptr;
    }
    if (file->extents.empty()) {
        std::fprintf(stderr,
            "[nvme_storage] acquire_device_handle: file has no extents\n");
        return nullptr;
    }

    // Build the host-side template.
    NvmeFileDeviceHandle h_host;
    std::memset(&h_host, 0, sizeof(h_host));
    h_host.file_id              = file->id;
    h_host.logical_size_bytes   = file->size_bytes;
    h_host.header_bytes         = (uint32_t)sizeof(NvmeFileHeader);
    h_host.nvme_block_size      = bp->blk_size;
    h_host.nvme_block_size_log  = bp->blk_size_log;
    h_host.namespace_id         = bp->namespace_id;
    h_host.num_extents          = (uint32_t)file->extents.size();
    if (h_host.num_extents > kNvmeFileHeaderMaxExtents) {
        std::fprintf(stderr,
            "[nvme_storage] acquire_device_handle: too many extents (%u > %u)\n",
            (unsigned)h_host.num_extents,
            (unsigned)kNvmeFileHeaderMaxExtents);
        return nullptr;
    }
    for (uint32_t i = 0; i < h_host.num_extents; ++i) {
        h_host.extents[i] = file->extents[i];
    }
    h_host.d_qps     = qg->d_qps();
    h_host.num_d_qps = qg->n_qps();

    // cudaMalloc + memcpy onto the same cuda device the queue group
    // allocated d_qps on.  cudaSetDevice is sticky to the calling
    // thread; we restore it on exit so this method has no side
    // effects on the caller's selected device.
    int prev_dev = -1;
    cudaError_t cerr = cudaGetDevice(&prev_dev);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[nvme_storage] acquire_device_handle: cudaGetDevice: %s\n",
            cudaGetErrorString(cerr));
        return nullptr;
    }
    cerr = cudaSetDevice((int)qg->cuda_device());
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[nvme_storage] acquire_device_handle: cudaSetDevice(%u): %s\n",
            (unsigned)qg->cuda_device(), cudaGetErrorString(cerr));
        return nullptr;
    }

    NvmeFileDeviceHandle* dh = nullptr;
    cerr = cudaMalloc((void**)&dh, sizeof(NvmeFileDeviceHandle));
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[nvme_storage] acquire_device_handle: cudaMalloc(handle): %s\n",
            cudaGetErrorString(cerr));
        cudaSetDevice(prev_dev);
        return nullptr;
    }

    cerr = cudaMemcpy(dh, &h_host, sizeof(NvmeFileDeviceHandle),
                      cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[nvme_storage] acquire_device_handle: cudaMemcpy: %s\n",
            cudaGetErrorString(cerr));
        cudaFree(dh);
        cudaSetDevice(prev_dev);
        return nullptr;
    }

    cudaSetDevice(prev_dev);
    return dh;
}

void HostFsBackedNvmeStorage::release_device_handle(NvmeFileDeviceHandle* dh)
{
    if (dh == nullptr) return;
    cudaError_t cerr = cudaFree(dh);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr,
            "[nvme_storage] release_device_handle: cudaFree: %s\n",
            cudaGetErrorString(cerr));
    }
}

} // namespace tutti

#ifndef __TUTTI_DEVICE_MANAGER_LOCAL_NVME_DEVICE_H__
#define __TUTTI_DEVICE_MANAGER_LOCAL_NVME_DEVICE_H__

/**
 * local_nvme_device.h -- the concrete payload behind a tutti::Device
 * when the backend is local NVMe (libnvm + snvme).
 *
 * Layer: Device Manager.
 *
 * Naming note (R4.5):
 *   This used to be `LocalNvmeBlockDevice`.  In the eight-layer
 *   architecture the word "Block" is reserved for the
 *   block_storage/ layer (virtual block-handle composed of N
 *   NvmeFiles).  This struct is just the per-controller payload
 *   we stash on `tutti::Device::backend_private`, so it's a plain
 *   "LocalNvmeDevice".  See doc/refactor/LegacyDecomposition.md
 *   §2 for the layer cake.
 *
 * Both registries (LocalNvmeDirectRegistry and
 * NvmeServiceBackedRegistry) produce LocalNvmeDevice instances and
 * stick them on the runtime-visible `Device` via
 * `Device::backend_private`.  The shape is identical regardless of
 * how the device was brought up; only how `ctrl` was obtained
 * differs.
 *
 * Lifetime
 *   - Owned by the IDeviceRegistry that produced the Device.  When
 *     the registry shuts down it calls release() on each
 *     LocalNvmeDevice, which closes the libnvm controller via
 *     the appropriate path (nvm_ctrl_free for direct, nvm_ctrl_free_client
 *     for service-attached) and disconnects the NVMeService session
 *     if there is one.
 *   - Holders of `Device*` MUST NOT delete the LocalNvmeDevice
 *     directly.
 */

#include <cstdint>
#include <memory>
#include <string>

#include <nvm_types.h>   // nvm_ctrl_t

namespace nvmeservice {
class NvmeServiceClient;          // src/nvmeservice_client.h
}

namespace tutti {

class NvmeQueueGroup;             // device_manager/include/nvme_queue_group.h

/// Bring-up mode for a LocalNvmeDevice.  Mirrors RuntimeConfig's
/// DeviceManagerMode but at the per-Device level so a future
/// IDeviceRegistry could mix modes if it wanted to.
enum class LocalNvmeAttachMode : uint32_t {
    DIRECT          = 0,   // owner: nvm_controller_init_b3 + nvm_ctrl_free
    SERVICE_CLIENT  = 1,   // attached: nvm_ctrl_attach_client + nvm_ctrl_free_client
};

struct LocalNvmeDevice {
    // ---- Identity ----
    int32_t      device_id;      // dense index inside the registry
    std::string  pci_addr;       // e.g. "0000:08:00.0"
    std::string  snvme_dev_path; // e.g. "/dev/ssnvme0"; populated by both modes
    std::string  display_name;   // human-readable

    // ---- libnvm handle ----
    nvm_ctrl_t*  ctrl = nullptr;
    LocalNvmeAttachMode attach_mode = LocalNvmeAttachMode::DIRECT;

    // ---- libnvm C++ wrapper for GPU-side d_qps[] (R5b) ----
    //
    // When set, owns a NvmeQueueGroup that wraps `ctrl` above and
    // exposes `d_qps()` -- the GPU-resident QueuePair[] kernels use
    // for direct NVMe submit.
    //
    // Non-null only when:
    //   - the registry that produced this device was opened with
    //     `build_queue_group=true` in its config, AND
    //   - `nvme_create_group` + `nvm_add_user_queue` succeeded on
    //     this controller's fd.
    //
    // Works the same way under DIRECT mode (ctrl came from
    // nvm_controller_init_b3) or SERVICE_CLIENT mode (ctrl came
    // from nvm_ctrl_attach_client) -- the queue group is per-fd,
    // not per-process.
    //
    // Lifetime: destroyed by the registry BEFORE freeing `ctrl`,
    // so nvm_destroy_group + cudaFree d_qps run while ctrl is
    // still alive.
    std::shared_ptr<NvmeQueueGroup> queue_group;

    // ---- Disk metadata (NVM_GET_DEV_INFO) ----
    uint32_t     namespace_id = 1;
    uint32_t     bar0_size    = 0;
    uint32_t     dstrd        = 0;
    uint32_t     blk_size     = 0;
    uint32_t     blk_size_log = 0;
    uint32_t     queue_depth  = 0;
    uint32_t     max_user_qid = 0;
    uint32_t     max_queues_per_group = 0;
    size_t       page_size    = 0;
    /// CTRL.MDTS in bytes -- the largest single-NVMe-IO transfer the
    /// controller will accept.  This is a property of the NVMe device
    /// itself (CAP/MDTS register reported by Identify Controller),
    /// NOT of the host or GPU; the value is whatever the drive
    /// firmware advertises (typ. 128 KiB on common enterprise NVMe).
    /// Populated from `nvm_disk_info::max_data_size` (DIRECT mode)
    /// or `nvm_ioctl_dev::max_data_size` returned by
    /// nvm_wait_dev_info(ctrl, ..) on the client fd (SERVICE_CLIENT
    /// mode).  Both are already byte-valued (PORTING.md §"max_data_size
    /// in bytes").  Consumers (memory/, block_storage) compute the
    /// per-tensor PRP fan-out as ceil(tensor_size / max_data_size).
    size_t       max_data_size = 0;

    // ---- Service-mode bookkeeping (nullptr for DIRECT) ----
    // Holds the gRPC session + heartbeat thread.  Owned by the
    // registry, released on shutdown via the registry's session
    // smart-pointer table.
    void*        service_session_opaque = nullptr;

    // ---- Last-seen mount path the daemon installed for this NVMe.
    //      Empty for DIRECT mode (no daemon).  Used by upper layers
    //      to resolve filesystem paths if needed.
    std::string  mount_path;
};

} // namespace tutti

#endif // __TUTTI_DEVICE_MANAGER_LOCAL_NVME_DEVICE_H__

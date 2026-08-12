#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tutti/resource/nvme/nvme_resource.h"

namespace tutti::resources::nvme {

struct RuntimeNvmeSlice {
    std::int32_t device_id = -1;
    std::int32_t accel_id = -1;
    std::string pci_bdf;
    std::string chrdev_path;
    std::string block_path;
    std::string backing_mount_path;
    std::string view_path;
    std::uint32_t namespace_id = 0;
    std::uint32_t logical_block_size = 0;
    std::uint64_t bar0_size = 0;
    std::uint64_t max_data_size = 0;
    std::uint32_t granted_queues = 0;
    std::vector<std::int32_t> allowed_accel_ids;
};

struct RuntimeNvmeAllocation {
    std::string allocation_id;
    std::vector<RuntimeNvmeSlice> slices;
};

enum class NvmeLeaseState {
    NONE,
    ACQUIRED,
    RELEASING,
    RELEASED,
};

struct NvmeResourceInspection {
    NvmeLeaseState lease_state = NvmeLeaseState::NONE;
    RuntimeNvmeAllocation allocation;
};

struct NvmeAcceleratorInfo {
    std::int32_t accel_id = -1;
    std::string view_root;
};

struct NvmeProviderResource {
    std::int32_t device_id = -1;
    std::vector<std::int32_t> allowed_accel_ids;
    bool available = false;
};

class NvmeResourceClient {
public:
    virtual ~NvmeResourceClient() = default;

    virtual Result<std::vector<NvmeAcceleratorInfo>>
    list_accelerators() = 0;

    virtual Result<std::vector<NvmeProviderResource>>
    list_nvme_resources() = 0;

    virtual Result<RuntimeNvmeAllocation> acquire_nvme_slices(
        std::int32_t accel_id,
        config::NvmeSelection selection,
        const std::vector<std::int32_t>& device_ids,
        std::int32_t queues_per_controller) = 0;

    virtual Status release(const std::string& allocation_id) = 0;
};

class NvmeResourceTestingAccess {
public:
    static std::unique_ptr<NvmeResource> make(
        std::string resource_id,
        const config::NvmeResourceConfig& config,
        std::int32_t runtime_accel_id,
        std::unique_ptr<NvmeResourceClient> client);

    static NvmeResourceInspection inspection(const NvmeResource& resource);
};

std::unique_ptr<NvmeResourceClient> make_nvme_resource_client(
    const std::string& endpoint);

} // namespace tutti::resources::nvme

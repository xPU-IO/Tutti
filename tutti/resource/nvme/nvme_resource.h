#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tutti/resource.h>
#include "tutti/config/storage_config.h"

namespace tutti::resources::nvme {

struct NvmeResourceSpec {
    std::string id;
    std::int32_t accel_id = -1;
    config::ProviderSpec provider;
    config::AllocationSpec allocation;
};

struct NvmeResolverSliceView {
    std::int32_t device_id = -1;
    std::string pci_bdf;
    std::string block_path;
    std::string backing_mount_path;
    std::string view_path;
    std::uint32_t namespace_id = 0;
    std::uint32_t logical_block_size = 0;
};

struct NvmeResolverResourceView {
    std::vector<NvmeResolverSliceView> slices;
};

struct NvmeDataPathSliceView {
    std::int32_t device_id = -1;
    std::string pci_bdf;
    std::int32_t accel_id = -1;
    std::string chrdev_path;
    std::uint32_t namespace_id = 0;
    std::uint32_t logical_block_size = 0;
    std::uint64_t bar0_size = 0;
    std::uint64_t max_data_size = 0;
    std::uint32_t granted_queues = 0;
};

struct NvmeDataPathResourceView {
    std::vector<NvmeDataPathSliceView> slices;
};

class NvmeResource final : public Resource {
public:
    ~NvmeResource() override;

    const ResourceCapabilities& capabilities() const override;
    Status initialize() override;
    Status shutdown() override;
    ResourceInfo info() const override;

    Result<NvmeResolverResourceView> resolver_view() const;
    Result<NvmeDataPathResourceView> datapath_view() const;

private:
    struct Impl;

    NvmeResource(NvmeResourceSpec spec, std::unique_ptr<Impl> impl);
    Status validate_allocation_metadata_() const;
    Status release_owned_allocation_();

    NvmeResourceSpec spec_;
    std::unique_ptr<Impl> impl_;

    friend class NvmeResourceTestingAccess;
    friend Result<std::unique_ptr<NvmeResource>> make_nvme_resource(
        NvmeResourceSpec spec);
};

Result<std::unique_ptr<NvmeResource>> make_nvme_resource(
    NvmeResourceSpec spec);

} // namespace tutti::resources::nvme

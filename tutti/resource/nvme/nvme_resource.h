#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tutti/config/spec/resource/nvme_spec.h>
#include <tutti/resource.h>

namespace tutti::resources::nvme {

struct NvmeResolverSliceView {
    std::int32_t device_id = -1;
    std::string pci_bdf;
    std::string block_path;
    std::string backing_mount_path;
    std::string view_path;
    std::uint32_t namespace_id = 0;
    std::uint32_t logical_block_size = 0;
};

struct NvmeResolverResourceView final : ResourceView {
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

struct NvmeDataPathResourceView final : ResourceView {
    std::vector<NvmeDataPathSliceView> slices;
};

class NvmeResource final : public Resource {
public:
    ~NvmeResource() override;

    const ResourceCapabilities& capabilities() const override;
    Status initialize() override;
    Status shutdown() override;
    ResourceInfo info() const override;
    Result<std::unique_ptr<const ResourceView>>
    get_resolver_view() const override;
    Result<std::unique_ptr<const ResourceView>>
    get_datapath_view() const override;

    Result<NvmeResolverResourceView> resolver_view() const;
    Result<NvmeDataPathResourceView> datapath_view() const;

private:
    struct Impl;

    NvmeResource(std::string id,
                 std::int32_t runtime_accel_id,
                 config::NvmeResourceConfig config,
                 std::unique_ptr<Impl> impl);
    Status validate_allocation_metadata_() const;
    Status release_owned_allocation_();

    std::string id_;
    std::int32_t runtime_accel_id_ = -1;
    config::NvmeAllocationSpec allocation_;
    std::unique_ptr<Impl> impl_;

    friend class NvmeResourceTestingAccess;
    friend Result<std::unique_ptr<NvmeResource>> create_nvme_resource(
        std::string resource_id,
        const config::NvmeResourceConfig& config,
        std::int32_t runtime_accel_id);
};

Result<std::unique_ptr<NvmeResource>> create_nvme_resource(
    std::string resource_id,
    const config::NvmeResourceConfig& config,
    std::int32_t runtime_accel_id);

} // namespace tutti::resources::nvme

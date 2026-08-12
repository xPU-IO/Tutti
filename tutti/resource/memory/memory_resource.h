#pragma once

#include <memory>
#include <string>

#include <tutti/config/spec/resource/memory_spec.h>
#include <tutti/resource.h>

namespace tutti::resources::memory {

struct MemoryResourceView final : ResourceView {
    std::uint64_t capacity_bytes = 0;
};

class MemoryResource final : public Resource {
public:
    const ResourceCapabilities& capabilities() const override;
    Status initialize() override;
    Status shutdown() override;
    ResourceInfo info() const override;
    Result<std::unique_ptr<const ResourceView>>
    get_resolver_view() const override;
    Result<std::unique_ptr<const ResourceView>>
    get_datapath_view() const override;

private:
    MemoryResource(std::string id, std::uint64_t capacity_bytes);
    Result<MemoryResourceView> memory_view() const;

    std::string id_;
    std::uint64_t capacity_bytes_ = 0;
    ResourceState state_ = ResourceState::CREATED;

    friend Result<std::unique_ptr<MemoryResource>> create_memory_resource(
        std::string resource_id,
        const config::MemoryResourceConfig& config);
};

Result<std::unique_ptr<MemoryResource>> create_memory_resource(
    std::string resource_id,
    const config::MemoryResourceConfig& config);

} // namespace tutti::resources::memory

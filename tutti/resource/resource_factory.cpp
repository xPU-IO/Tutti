#include <tutti/resource.h>

#include <type_traits>
#include <utility>
#include <variant>

#include <tutti/config/spec/resource/resource_spec.h>

#include "tutti/resource/memory/memory_resource.h"
#include "tutti/resource/nvme/nvme_resource.h"

namespace tutti::resources {
namespace {

Result<std::unique_ptr<Resource>> failure(Status status) {
    return Result<std::unique_ptr<Resource>>::Failure(std::move(status));
}

Status invalid(std::string message) {
    return Status(StatusCode::INVALID_ARGUMENT, std::move(message));
}

} // namespace

Result<std::unique_ptr<Resource>> create_resource(
    const config::ResourceSpec& spec,
    const ResourceCreateContext& context) {
    if (spec.id.empty()) {
        return failure(invalid("ResourceSpec.id must not be empty"));
    }

    return std::visit(
        [&](const auto& resource_config)
            -> Result<std::unique_ptr<Resource>> {
            using Config = std::decay_t<decltype(resource_config)>;

            if constexpr (std::is_same_v<Config,
                                         config::NvmeResourceConfig>) {
                if (spec.type != "nvme") {
                    return failure(invalid(
                        "ResourceSpec.type does not match NVMe config"));
                }
                auto created = nvme::create_nvme_resource(
                    spec.id, resource_config, context.runtime_accel_id);
                if (!created.ok()) {
                    return failure(created.status());
                }
                std::unique_ptr<Resource> resource =
                    std::move(created).value();
                return Result<std::unique_ptr<Resource>>::Success(
                    std::move(resource));
            } else if constexpr (std::is_same_v<
                                     Config,
                                     config::MemoryResourceConfig>) {
                if (spec.type != "memory") {
                    return failure(invalid(
                        "ResourceSpec.type does not match memory config"));
                }
                auto created = memory::create_memory_resource(
                    spec.id, resource_config);
                if (!created.ok()) {
                    return failure(created.status());
                }
                std::unique_ptr<Resource> resource =
                    std::move(created).value();
                return Result<std::unique_ptr<Resource>>::Success(
                    std::move(resource));
            }
        },
        spec.config);
}

} // namespace tutti::resources

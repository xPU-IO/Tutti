#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tutti/config/tutti_runtime_spec.h>
#include <tutti/data_paths/data_path_factory.h>
#include <tutti/resolvers/resolver_factory.h>
#include <tutti/spi/data_path.h>
#include <tutti/spi/storage_target_resolver.h>
#include <tutti/storage_runtime.h>
#include <tutti/tutti_runtime.h>

namespace tutti::tutti_runtime {

struct TuttiRuntimeCreateInternalOptions {
    TuttiRuntimeCreateOptions public_options;
    std::function<Result<int>()> accelerator_device_count;
    std::function<Result<std::unique_ptr<StorageRuntime>>(
        RuntimeConfig, RuntimeComponents)> runtime_factory;
    std::function<Result<std::unique_ptr<Resource>>(
        const config::ResourceSpec&,
        const resources::ResourceCreateContext&)> resource_factory;
    std::function<Result<std::unique_ptr<StorageTargetResolver>>(
        const config::ResolverSpec&,
        const resolvers::ResolverCreateContext&)> resolver_factory;
    std::function<Result<data_paths::CreatedDataPath>(
        const config::DataPathSpec&,
        const data_paths::DataPathCreateContext&)> data_path_factory;
    std::function<Status(StorageRuntime&)> runtime_shutdown_hook;
    std::function<void(TuttiRuntimeShutdownStage)> shutdown_observer;
};

} // namespace tutti::tutti_runtime

namespace tutti::testing {

class TuttiRuntimeTestAccess {
public:
    static Result<std::unique_ptr<TuttiRuntime>> create(
        config::TuttiRuntimeSpec spec,
        tutti_runtime::TuttiRuntimeCreateInternalOptions options) {
        return TuttiRuntime::create_with_options_(
            std::move(spec), std::move(options));
    }

    static const Resource* resource(const TuttiRuntime& runtime) noexcept {
        if (runtime.resource_initialization_order_.empty()) return nullptr;
        return runtime.find_resource_(
            runtime.resource_initialization_order_.front());
    }
    static StorageTargetResolver* resolver(
        TuttiRuntime& runtime, std::string_view id) noexcept {
        const auto found = runtime.resolvers_.find(std::string(id));
        return found == runtime.resolvers_.end() ? nullptr
                                                 : found->second.get();
    }
    static const DataPath* data_path(
        const TuttiRuntime& runtime, std::string_view id) noexcept {
        const auto found = runtime.datapaths_.find(std::string(id));
        return found == runtime.datapaths_.end() ? nullptr
                                                 : found->second.get();
    }
    static const std::vector<std::string>& resource_initialization_order(
        const TuttiRuntime& runtime) noexcept {
        return runtime.resource_initialization_order_;
    }
    static std::size_t resolver_count(const TuttiRuntime& runtime) noexcept {
        return runtime.resolvers_.size();
    }
    static std::size_t data_path_count(const TuttiRuntime& runtime) noexcept {
        return runtime.datapaths_.size();
    }
};

} // namespace tutti::testing

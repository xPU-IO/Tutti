#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tutti/config/tutti_runtime_spec.h>
#include <tutti/spi/data_path.h>
#include <tutti/spi/storage_target_resolver.h>
#include <tutti/storage_runtime.h>
#include <tutti/tutti_runtime.h>

#include "tutti/tutti_runtime/backend_factory.h"

namespace tutti::tutti_runtime {

struct EffectiveCacheConfig {
    std::uint32_t handle_cache_capacity = 0;
    std::uint32_t prp_cache_capacity = 0;
    std::uint32_t handle_cache_l2_capacity = 0;
};

struct TuttiRuntimeCreateInternalOptions {
    TuttiRuntimeCreateOptions public_options;
    std::function<Result<int>()> backend_device_count;
    std::function<Result<std::unique_ptr<StorageRuntime>>(
        RuntimeConfig, RuntimeComponents)> runtime_factory;
    std::function<Result<std::unique_ptr<Resource>>(
        const config::ResourceSpec&, std::int32_t)> resource_factory;
    BackendFactory backend_factory;
    std::function<Status(StorageRuntime&)> runtime_shutdown_hook;
    std::function<void(TuttiRuntimeShutdownStage)> shutdown_observer;
};

EffectiveCacheConfig resolve_cache_config(
    const config::TuttiRuntimeSpec& spec,
    const TuttiRuntimeCreateOptions& options);

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
    static StorageTargetResolver* backend_resolver(
        TuttiRuntime& runtime, std::string_view id) noexcept {
        const auto found = runtime.backends_.find(std::string(id));
        return found == runtime.backends_.end() ? nullptr
                                                : found->second.resolver;
    }
    static const DataPath* backend_datapath(
        const TuttiRuntime& runtime, std::string_view id) noexcept {
        const auto found = runtime.backends_.find(std::string(id));
        return found == runtime.backends_.end() ? nullptr
                                                : found->second.datapath;
    }
    static const std::vector<std::string>& resource_initialization_order(
        const TuttiRuntime& runtime) noexcept {
        return runtime.resource_initialization_order_;
    }
    static const std::string& validated_spec_debug(
        const TuttiRuntime& runtime) noexcept {
        return runtime.validated_spec_debug_;
    }
    static Status shutdown_resource(TuttiRuntime& runtime,
                                    std::string_view id) {
        return runtime.shutdown_resource_(id);
    }
};

} // namespace tutti::testing

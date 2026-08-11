#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tutti/spi/data_path.h>
#include <tutti/spi/storage_target_resolver.h>
#include <tutti/storage_runtime.h>
#include <tutti/tutti_runtime.h>

namespace tutti::config {

class TuttiRuntimeAssemblyAccess {
public:
    static Status adopt_resource(TuttiRuntime& runtime, std::string id,
                                 std::unique_ptr<Resource> resource) {
        return runtime.adopt_resource_(std::move(id), std::move(resource));
    }

    static const Resource* resource(const TuttiRuntime& runtime,
                                    std::string_view id) noexcept {
        return runtime.find_resource_(id);
    }

    static Status register_datapath(TuttiRuntime& runtime, std::string id,
                                    std::unique_ptr<DataPath> data_path,
                                    std::string key, DataPath*& borrowed) {
        return runtime.register_datapath_(
            std::move(id), std::move(data_path), std::move(key), borrowed);
    }

    static Status register_resolver(
        TuttiRuntime& runtime, std::string id,
        std::unique_ptr<StorageTargetResolver> resolver,
        std::string scheme, StorageTargetResolver*& borrowed) {
        return runtime.register_resolver_(
            std::move(id), std::move(resolver), std::move(scheme), borrowed);
    }

    static Status register_backend(TuttiRuntime& runtime,
                                   BackendManifest manifest,
                                   const Resource* resource,
                                   StorageTargetResolver* resolver,
                                   DataPath* datapath) {
        return runtime.register_backend_(std::move(manifest), resource,
                                         resolver, datapath);
    }

    static Status set_storage_runtime(
        TuttiRuntime& runtime, std::unique_ptr<StorageRuntime> storage_runtime) {
        return runtime.set_storage_runtime_(std::move(storage_runtime));
    }
};

class TuttiRuntimeTestingAccess {
public:
    static const Resource* resource(const TuttiRuntime& runtime) noexcept {
        if (runtime.resource_initialization_order_.empty()) return nullptr;
        return runtime.find_resource_(
            runtime.resource_initialization_order_.front());
    }

    static const Resource* resource(const TuttiRuntime& runtime,
                                    std::string_view id) noexcept {
        return runtime.find_resource_(id);
    }

    static const std::vector<std::string>& resource_initialization_order(
        const TuttiRuntime& runtime) noexcept {
        return runtime.resource_initialization_order_;
    }

    static Status adopt_resource(TuttiRuntime& runtime, std::string id,
                                 std::unique_ptr<Resource> resource) {
        return runtime.adopt_resource_(std::move(id), std::move(resource));
    }

    static Status shutdown_resource(TuttiRuntime& runtime,
                                    std::string_view id) {
        return runtime.shutdown_resource_(id);
    }

    static void set_runtime_shutdown_hook(
        TuttiRuntime& runtime,
        std::function<Status(StorageRuntime&)> hook) {
        runtime.runtime_shutdown_hook_ = std::move(hook);
    }

    static void set_shutdown_observer(
        TuttiRuntime& runtime,
        std::function<void(TuttiRuntimeShutdownStage)> observer) {
        runtime.shutdown_observer_ = std::move(observer);
    }
};

} // namespace tutti::config

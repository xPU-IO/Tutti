#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

    static Status register_backend(TuttiRuntime& runtime,
                                   BackendManifest manifest,
                                   const Resource* resource,
                                   StorageTargetResolver* resolver,
                                   DataPath* datapath) {
        return runtime.register_backend_(std::move(manifest), resource,
                                         resolver, datapath);
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
};

} // namespace tutti::config

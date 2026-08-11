// tutti/include/tutti/tutti_runtime.h
//
// Application runtime ownership and lifecycle. Config parsing and loader
// assembly may populate this aggregate, but cleanup is implemented in the
// companion source file so parsing cannot accidentally own runtime policy.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <tutti/resource.h>
#include <tutti/status.h>

namespace tutti {

class DataPath;
class StorageRuntime;
class StorageTargetResolver;

namespace config {

enum class TuttiRuntimeState {
    RUNNING,
    SHUTTING_DOWN,
    STOPPED,
};

enum class TuttiRuntimeShutdownStage {
    STORAGE_RUNTIME_SHUTDOWN,
    STORAGE_RUNTIME_DESTROYED,
    RESOLVERS_DESTROYED,
    DATAPATHS_DESTROYED,
    RESOURCE_SHUTDOWN,
    COMPLETE,
};

class TuttiRuntimeTestingAccess;
class TuttiRuntimeAssemblyAccess;

struct BackendManifest {
    std::string id;
    std::string contract;
    std::string resolver_id;
    std::string datapath_id;
    std::string resource_id;
};

// Owned runtime bundle. Component vectors remain as a compatibility surface;
// Resource ownership and backend relationships are private ID registries.
struct TuttiRuntime {
    std::unique_ptr<StorageRuntime> runtime;
    std::vector<std::unique_ptr<DataPath>> datapaths;
    std::vector<std::unique_ptr<StorageTargetResolver>> resolvers;
    std::vector<std::string> resolver_schemes;
    std::vector<std::string> data_path_keys;

    TuttiRuntime();
    ~TuttiRuntime();
    Status shutdown();

    TuttiRuntimeState state() const noexcept { return state_; }
    Result<ResourceInfo> resource_info(std::string_view id) const;
    std::vector<ResourceInfo> resource_infos() const;
    Result<BackendManifest> backend_manifest(std::string_view id) const;
    std::vector<BackendManifest> backend_manifests() const;

    // Single-resource compatibility seam retained for current callers.
    Result<ResourceInfo> resource_info() const;

    // Transfer component ownership into the runtime and register its route
    // key. The returned pointer is borrowed and remains valid until shutdown.
    // Registration is rejected after shutdown has started.
    DataPath* register_datapath(std::unique_ptr<DataPath> data_path,
                                std::string key);
    StorageTargetResolver* register_resolver(
        std::unique_ptr<StorageTargetResolver> resolver,
        std::string scheme);

    // Test-only lifecycle injection. Ownership remains with TuttiRuntime.
    void set_runtime_shutdown_hook(
        std::function<Status(StorageRuntime&)> hook) {
        runtime_shutdown_hook_ = std::move(hook);
    }
    void set_shutdown_observer(
        std::function<void(TuttiRuntimeShutdownStage)> observer) {
        shutdown_observer_ = std::move(observer);
    }

private:
    friend class TuttiRuntimeTestingAccess;
    friend class TuttiRuntimeAssemblyAccess;

    struct BackendInstance {
        BackendManifest manifest;
        const Resource* resource = nullptr;
        StorageTargetResolver* resolver = nullptr;
        DataPath* datapath = nullptr;
    };

    void observe_(TuttiRuntimeShutdownStage stage) noexcept;
    Status adopt_resource_(std::string id,
                           std::unique_ptr<Resource> resource);
    const Resource* find_resource_(std::string_view id) const noexcept;
    Status register_backend_(BackendManifest manifest,
                             const Resource* resource,
                             StorageTargetResolver* resolver,
                             DataPath* datapath);
    Status shutdown_resource_(std::string_view id);

    TuttiRuntimeState state_ = TuttiRuntimeState::RUNNING;
    std::unordered_map<std::string, std::unique_ptr<Resource>> resources_;
    std::vector<std::string> resource_initialization_order_;
    std::unordered_map<std::string, BackendInstance> backends_;
    std::vector<std::string> backend_registration_order_;
    std::function<Status(StorageRuntime&)> runtime_shutdown_hook_;
    std::function<void(TuttiRuntimeShutdownStage)> shutdown_observer_;
};

} // namespace config
} // namespace tutti

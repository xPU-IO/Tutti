// tutti/include/tutti/tutti_runtime.h
//
// Application runtime ownership and lifecycle. Config parsing and loader
// assembly may populate this aggregate, but cleanup is implemented in the
// companion source file so parsing cannot accidentally own runtime policy.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <tutti/resource.h>
#include <tutti/config/storage_config.h>
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

// Owned runtime bundle. Component vectors remain as a P2 compatibility
// surface; Resource ownership is private and moves to an ID registry in P4.
struct TuttiRuntime {
    std::unique_ptr<StorageRuntime> runtime;
    std::vector<std::unique_ptr<DataPath>> datapaths;
    std::vector<std::unique_ptr<StorageTargetResolver>> resolvers;
    std::vector<std::string> resolver_schemes;
    std::vector<std::string> data_path_keys;

    ~TuttiRuntime();
    Status shutdown();

    TuttiRuntimeState state() const noexcept { return state_; }
    Result<ResourceInfo> resource_info() const;

    Status adopt_resource(std::unique_ptr<Resource> resource);

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

    void observe_(TuttiRuntimeShutdownStage stage) noexcept;

    TuttiRuntimeState state_ = TuttiRuntimeState::RUNNING;
    std::unique_ptr<Resource> resource_;
    std::function<Status(StorageRuntime&)> runtime_shutdown_hook_;
    std::function<void(TuttiRuntimeShutdownStage)> shutdown_observer_;
};

} // namespace config
} // namespace tutti

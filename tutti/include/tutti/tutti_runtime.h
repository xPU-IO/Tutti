#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <tutti/config/tutti_runtime_spec.h>
#include <tutti/resource.h>
#include <tutti/status.h>

namespace tutti {

class DataPath;
class StorageRuntime;
class StorageTargetResolver;

namespace tutti_runtime {
struct TuttiRuntimeCreateInternalOptions;
}
namespace testing {
class TuttiRuntimeTestAccess;
}

enum class TuttiRuntimeState {
    INITIALIZING,
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

struct TuttiRuntimeCreateOptions {
    std::uint32_t handle_cache_capacity = 0;
    std::uint32_t prp_cache_capacity = 0;
    std::uint32_t handle_cache_l2_capacity = 0;
    std::function<void(std::string_view)> spec_debug_logger;
};

struct BackendManifest {
    std::string id;
    std::string contract;
    std::string resolver_id;
    std::string datapath_id;
    std::string resource_id;
};

class TuttiRuntime {
public:
    static Result<std::unique_ptr<TuttiRuntime>> create(
        const std::string& config_path,
        TuttiRuntimeCreateOptions options = {});
    static Result<std::unique_ptr<TuttiRuntime>> create(
        config::TuttiRuntimeSpec spec,
        TuttiRuntimeCreateOptions options = {});

    ~TuttiRuntime();
    Status shutdown();

    TuttiRuntimeState state() const noexcept { return state_; }
    StorageRuntime* storage_runtime() noexcept { return runtime_.get(); }
    const StorageRuntime* storage_runtime() const noexcept {
        return runtime_.get();
    }
    Result<ResourceInfo> resource_info(std::string_view id) const;
    std::vector<ResourceInfo> resource_infos() const;
    Result<BackendManifest> backend_manifest(std::string_view id) const;
    std::vector<BackendManifest> backend_manifests() const;

private:
    friend class testing::TuttiRuntimeTestAccess;

    struct BackendInstance {
        BackendManifest manifest;
        const Resource* resource = nullptr;
        StorageTargetResolver* resolver = nullptr;
        DataPath* datapath = nullptr;
    };

    TuttiRuntime();
    static Result<std::unique_ptr<TuttiRuntime>> create_with_options_(
        config::TuttiRuntimeSpec spec,
        tutti_runtime::TuttiRuntimeCreateInternalOptions options);

    void observe_(TuttiRuntimeShutdownStage stage) noexcept;
    Status adopt_resource_(std::string id,
                           std::unique_ptr<Resource> resource);
    const Resource* find_resource_(std::string_view id) const noexcept;
    Status register_datapath_(std::string id,
                              std::unique_ptr<DataPath> data_path,
                              std::string key,
                              DataPath*& borrowed);
    Status register_resolver_(std::string id,
                              std::unique_ptr<StorageTargetResolver> resolver,
                              std::string scheme,
                              StorageTargetResolver*& borrowed);
    Status register_backend_(BackendManifest manifest,
                             const Resource* resource,
                             StorageTargetResolver* resolver,
                             DataPath* datapath);
    Status set_storage_runtime_(std::unique_ptr<StorageRuntime> runtime);
    Status shutdown_resource_(std::string_view id);

    TuttiRuntimeState state_ = TuttiRuntimeState::INITIALIZING;
    std::string validated_spec_debug_;
    std::unique_ptr<StorageRuntime> runtime_;
    std::unordered_map<std::string, std::unique_ptr<Resource>> resources_;
    std::vector<std::string> resource_initialization_order_;
    std::unordered_map<std::string, std::unique_ptr<StorageTargetResolver>>
        resolvers_;
    std::vector<std::string> resolver_registration_order_;
    std::unordered_map<std::string, std::string> resolver_schemes_;
    std::unordered_map<std::string, std::unique_ptr<DataPath>> datapaths_;
    std::vector<std::string> datapath_registration_order_;
    std::unordered_map<std::string, std::string> data_path_keys_;
    std::unordered_map<std::string, BackendInstance> backends_;
    std::vector<std::string> backend_registration_order_;
    std::function<Status(StorageRuntime&)> runtime_shutdown_hook_;
    std::function<void(TuttiRuntimeShutdownStage)> shutdown_observer_;
};

} // namespace tutti

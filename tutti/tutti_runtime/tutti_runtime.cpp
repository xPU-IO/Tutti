// tutti/tutti_runtime/tutti_runtime.cpp

#include <tutti/tutti_runtime.h>

#include <algorithm>
#include <utility>

#include <tutti/spi/data_path.h>
#include <tutti/spi/storage_target_resolver.h>
#include <tutti/storage_runtime.h>

namespace tutti::config {
namespace {

Status lifecycle_error(StatusCode code, const char* message) {
    return Status(code, message);
}

void keep_first_error(Status& first_error, Status status) {
    if (!status.ok() && first_error.ok()) {
        first_error = std::move(status);
    }
}

} // namespace

TuttiRuntime::TuttiRuntime() = default;

TuttiRuntime::~TuttiRuntime() {
    (void)shutdown();
}

Status TuttiRuntime::adopt_resource_(std::string id,
                                    std::unique_ptr<Resource> resource) {
    if (!resource) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "cannot adopt a null Resource");
    }
    const ResourceInfo info = resource->info();
    if (id.empty() || info.id != id) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "Resource registry ID does not match ResourceInfo");
    }
    if (info.state != ResourceState::INITIALIZED) {
        return lifecycle_error(StatusCode::NOT_READY,
                               "Resource must be INITIALIZED before registration");
    }
    if (state_ != TuttiRuntimeState::RUNNING) {
        return lifecycle_error(StatusCode::BUSY,
                               "TuttiRuntime cannot adopt Resource");
    }
    if (resources_.count(id) != 0) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "TuttiRuntime Resource ID is already registered");
    }

    Resource* borrowed = resource.get();
    auto inserted = resources_.emplace(id, std::move(resource));
    try {
        resource_initialization_order_.push_back(std::move(id));
    } catch (...) {
        if (inserted.second && inserted.first->second.get() == borrowed) {
            resources_.erase(inserted.first);
        }
        throw;
    }
    return Status::Ok();
}

const Resource* TuttiRuntime::find_resource_(std::string_view id) const noexcept {
    for (const auto& resource : resources_) {
        if (resource.first.size() == id.size() &&
            std::equal(resource.first.begin(), resource.first.end(),
                       id.begin())) {
            return resource.second.get();
        }
    }
    return nullptr;
}

Status TuttiRuntime::register_backend_(BackendManifest manifest,
                                      const Resource* resource,
                                      StorageTargetResolver* resolver,
                                      DataPath* datapath) {
    if (state_ != TuttiRuntimeState::RUNNING) {
        return lifecycle_error(StatusCode::BUSY,
                               "TuttiRuntime cannot register backend");
    }
    if (manifest.id.empty() || manifest.contract.empty() ||
        manifest.resolver_id.empty() || manifest.datapath_id.empty() ||
        manifest.resource_id.empty() || resource == nullptr ||
        resolver == nullptr || datapath == nullptr) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "backend registration is incomplete");
    }
    if (find_resource_(manifest.resource_id) != resource) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "backend Resource does not match registry ID");
    }
    if (std::none_of(resolvers.begin(), resolvers.end(),
                     [&](const auto& owned) { return owned.get() == resolver; }) ||
        std::none_of(datapaths.begin(), datapaths.end(),
                     [&](const auto& owned) { return owned.get() == datapath; })) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "backend components are not owned by TuttiRuntime");
    }
    if (backends_.count(manifest.id) != 0) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "TuttiRuntime backend ID is already registered");
    }
    for (const auto& entry : backends_) {
        if (entry.second.resource == resource &&
            entry.second.datapath != datapath) {
            return lifecycle_error(
                StatusCode::INVALID_ARGUMENT,
                "Resource cannot be consumed by independent DataPaths");
        }
    }

    BackendInstance instance{std::move(manifest), resource, resolver, datapath};
    const std::string backend_id = instance.manifest.id;
    auto inserted = backends_.emplace(backend_id, std::move(instance));
    try {
        backend_registration_order_.push_back(backend_id);
    } catch (...) {
        if (inserted.second) backends_.erase(inserted.first);
        throw;
    }
    return Status::Ok();
}

DataPath* TuttiRuntime::register_datapath(
    std::unique_ptr<DataPath> data_path, std::string key) {
    if (!data_path || state_ != TuttiRuntimeState::RUNNING) return nullptr;
    DataPath* borrowed = data_path.get();
    try {
        datapaths.push_back(std::move(data_path));
        data_path_keys.push_back(std::move(key));
    } catch (...) {
        // Keep the aggregate internally consistent if route registration
        // cannot allocate its diagnostic key.
        if (!datapaths.empty() && datapaths.back().get() == borrowed) {
            datapaths.pop_back();
        }
        throw;
    }
    return borrowed;
}

StorageTargetResolver* TuttiRuntime::register_resolver(
    std::unique_ptr<StorageTargetResolver> resolver, std::string scheme) {
    if (!resolver || state_ != TuttiRuntimeState::RUNNING) return nullptr;
    StorageTargetResolver* borrowed = resolver.get();
    try {
        resolvers.push_back(std::move(resolver));
        resolver_schemes.push_back(std::move(scheme));
    } catch (...) {
        if (!resolvers.empty() && resolvers.back().get() == borrowed) {
            resolvers.pop_back();
        }
        throw;
    }
    return borrowed;
}

void TuttiRuntime::observe_(TuttiRuntimeShutdownStage stage) noexcept {
    if (!shutdown_observer_) return;
    try {
        shutdown_observer_(stage);
    } catch (...) {
        // Observers are diagnostics only and must never interrupt cleanup.
    }
}

Status TuttiRuntime::shutdown() {
    if (state_ == TuttiRuntimeState::STOPPED) {
        return Status::Ok();
    }
    state_ = TuttiRuntimeState::SHUTTING_DOWN;

    Status first_error;

    observe_(TuttiRuntimeShutdownStage::STORAGE_RUNTIME_SHUTDOWN);
    if (runtime) {
        try {
            Status status = runtime_shutdown_hook_
                ? runtime_shutdown_hook_(*runtime)
                : runtime->shutdown(0);
            keep_first_error(first_error, std::move(status));
        } catch (...) {
            keep_first_error(first_error,
                             lifecycle_error(StatusCode::INTERNAL,
                                             "StorageRuntime shutdown threw"));
        }
        runtime.reset();
    }
    observe_(TuttiRuntimeShutdownStage::STORAGE_RUNTIME_DESTROYED);

    for (auto& backend : backends_) {
        backend.second.resource = nullptr;
        backend.second.resolver = nullptr;
        backend.second.datapath = nullptr;
    }

    // Destroy owned components in reverse construction order while their
    // route bindings are no longer used by StorageRuntime.
    while (!resolvers.empty()) resolvers.pop_back();
    observe_(TuttiRuntimeShutdownStage::RESOLVERS_DESTROYED);
    while (!datapaths.empty()) datapaths.pop_back();
    observe_(TuttiRuntimeShutdownStage::DATAPATHS_DESTROYED);
    // Keep route strings as immutable diagnostics for the temporary P2
    // inspection seam; the owning component vectors above are empty.

    for (auto id = resource_initialization_order_.rbegin();
         id != resource_initialization_order_.rend(); ++id) {
        auto resource = resources_.find(*id);
        if (resource == resources_.end() || !resource->second) continue;
        try {
            keep_first_error(first_error, resource->second->shutdown());
        } catch (...) {
            keep_first_error(first_error,
                             lifecycle_error(StatusCode::INTERNAL,
                                             "Resource shutdown threw"));
        }
    }
    observe_(TuttiRuntimeShutdownStage::RESOURCE_SHUTDOWN);

    state_ = TuttiRuntimeState::STOPPED;
    observe_(TuttiRuntimeShutdownStage::COMPLETE);
    return first_error;
}

Result<ResourceInfo> TuttiRuntime::resource_info() const {
    if (resource_initialization_order_.size() != 1) {
        return Result<ResourceInfo>::Failure(
            lifecycle_error(StatusCode::NOT_FOUND,
                            "TuttiRuntime does not have exactly one Resource"));
    }
    return resource_info(resource_initialization_order_.front());
}

Result<ResourceInfo> TuttiRuntime::resource_info(std::string_view id) const {
    const Resource* resource = find_resource_(id);
    if (resource == nullptr) {
        return Result<ResourceInfo>::Failure(
            lifecycle_error(StatusCode::NOT_FOUND,
                            "TuttiRuntime Resource ID was not found"));
    }
    return Result<ResourceInfo>::Success(resource->info());
}

std::vector<ResourceInfo> TuttiRuntime::resource_infos() const {
    std::vector<ResourceInfo> infos;
    infos.reserve(resource_initialization_order_.size());
    for (const std::string& id : resource_initialization_order_) {
        const Resource* resource = find_resource_(id);
        if (resource != nullptr) infos.push_back(resource->info());
    }
    return infos;
}

Result<BackendManifest> TuttiRuntime::backend_manifest(
    std::string_view id) const {
    const auto found = backends_.find(std::string(id));
    if (found == backends_.end()) {
        return Result<BackendManifest>::Failure(
            lifecycle_error(StatusCode::NOT_FOUND,
                            "TuttiRuntime backend ID was not found"));
    }
    return Result<BackendManifest>::Success(found->second.manifest);
}

std::vector<BackendManifest> TuttiRuntime::backend_manifests() const {
    std::vector<BackendManifest> manifests;
    manifests.reserve(backend_registration_order_.size());
    for (const std::string& id : backend_registration_order_) {
        const auto found = backends_.find(id);
        if (found != backends_.end()) {
            manifests.push_back(found->second.manifest);
        }
    }
    return manifests;
}

Status TuttiRuntime::shutdown_resource_(std::string_view id) {
    auto found = resources_.find(std::string(id));
    if (found == resources_.end()) {
        return lifecycle_error(StatusCode::NOT_FOUND,
                               "TuttiRuntime Resource ID was not found");
    }
    for (const auto& backend : backends_) {
        if (backend.second.resource == found->second.get()) {
            return lifecycle_error(StatusCode::BUSY,
                                   "Resource is still referenced by a backend");
        }
    }
    return found->second->shutdown();
}

} // namespace tutti::config

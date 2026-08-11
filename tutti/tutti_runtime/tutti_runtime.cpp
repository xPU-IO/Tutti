// tutti/tutti_runtime/tutti_runtime.cpp

#include <tutti/tutti_runtime.h>

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

TuttiRuntime::~TuttiRuntime() {
    (void)shutdown();
}

Status TuttiRuntime::adopt_resource(std::unique_ptr<Resource> resource) {
    if (!resource) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "cannot adopt a null Resource");
    }
    if (state_ != TuttiRuntimeState::RUNNING || resource_) {
        return lifecycle_error(StatusCode::BUSY,
                               "TuttiRuntime cannot adopt Resource");
    }
    resource_ = std::move(resource);
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

    // Destroy owned components in reverse construction order while their
    // route bindings are no longer used by StorageRuntime.
    while (!resolvers.empty()) resolvers.pop_back();
    observe_(TuttiRuntimeShutdownStage::RESOLVERS_DESTROYED);
    while (!datapaths.empty()) datapaths.pop_back();
    observe_(TuttiRuntimeShutdownStage::DATAPATHS_DESTROYED);
    // Keep route strings as immutable diagnostics for the temporary P2
    // inspection seam; the owning component vectors above are empty.

    if (resource_) {
        try {
            keep_first_error(first_error, resource_->shutdown());
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
    if (!resource_) {
        return Result<ResourceInfo>::Failure(
            lifecycle_error(StatusCode::NOT_FOUND,
                            "TuttiRuntime has no Resource"));
    }
    return Result<ResourceInfo>::Success(resource_->info());
}

} // namespace tutti::config

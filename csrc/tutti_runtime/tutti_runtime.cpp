// tutti/tutti_runtime/tutti_runtime.cpp

#include <tutti/tutti_runtime.h>

#include <algorithm>
#include <utility>

#include <tutti/spi/data_path.h>
#include <tutti/spi/storage_target_resolver.h>
#include <tutti/storage_runtime.h>

namespace tutti {
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
    /*
     * 析构无法返回状态，所以策略必须明确且自洽：先做**有界**阻塞 drain；
     * 若仍有非终态 I/O，就**泄漏完整 ownership graph**。
     *
     * 为什么不是"只释放一部分"：这正是修复前的缺陷。StorageRuntime 内部
     * 已经做了"宁可泄漏 runtime-owned memory 也不 UAF"的保护，但析构随后
     * 仍销毁 DataPath / queue group / DMA registration / NVMe lease，把那份
     * 保护抵消掉——GPU/NVMe 可能仍在向那些内存 DMA。
     *
     * 一次性的少量泄漏由进程退出时归还操作系统。相比在途 DMA 打向已释放的
     * 内存，这是严格更好的结局。
     */
    (void)shutdown(kDefaultDrainTimeoutMs);
    if (state_ == TuttiRuntimeState::STOPPED) return;

    // 只可能由 TIMEOUT 到达这里：shutdown() 的其它出口都会推进到 STOPPED。
    // 刻意 release 而不 reset：对象本身留活。
    (void)runtime_.release();
    for (auto& entry : resolvers_) (void)entry.second.release();
    for (auto& entry : datapaths_) (void)entry.second.release();
    for (auto& entry : resources_) (void)entry.second.release();
    std::fprintf(stderr,
                 "TuttiRuntime: drain timed out with non-terminal I/O; "
                 "leaking the ownership graph (StorageRuntime, resolvers, "
                 "DataPaths, Resources) rather than tearing down state that "
                 "in-flight DMA may still reference\n");
}

Status TuttiRuntime::adopt_resource_(std::string id,
                                    std::unique_ptr<Resource> resource) {
    const auto reject = [&](Status status) {
        try { (void)resource->shutdown(); } catch (...) {}
        return status;
    };
    if (!resource) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "cannot adopt a null Resource");
    }
    const ResourceInfo info = resource->info();
    if (id.empty() || info.id != id) {
        return reject(lifecycle_error(
            StatusCode::INVALID_ARGUMENT,
            "Resource registry ID does not match ResourceInfo"));
    }
    if (info.state != ResourceState::INITIALIZED) {
        return reject(lifecycle_error(
            StatusCode::NOT_READY,
            "Resource must be INITIALIZED before registration"));
    }
    if (state_ != TuttiRuntimeState::INITIALIZING) {
        return reject(lifecycle_error(
            StatusCode::BUSY,
            "TuttiRuntime cannot adopt Resource"));
    }
    if (resources_.count(id) != 0) {
        return reject(lifecycle_error(
            StatusCode::INVALID_ARGUMENT,
            "TuttiRuntime Resource ID is already registered"));
    }

    Resource* borrowed = resource.get();
    try {
        resource_initialization_order_.push_back(id);
    } catch (...) {
        try { (void)resource->shutdown(); } catch (...) {}
        throw;
    }
    try {
        const auto inserted = resources_.emplace(id, std::move(resource));
        if (!inserted.second) {
            resource_initialization_order_.pop_back();
            return lifecycle_error(
                StatusCode::INVALID_ARGUMENT,
                "TuttiRuntime Resource ID is already registered");
        }
    } catch (...) {
        auto inserted = resources_.find(id);
        if (inserted != resources_.end() &&
            inserted->second.get() == borrowed) {
            try { (void)inserted->second->shutdown(); } catch (...) {}
            resources_.erase(inserted);
        } else if (resource) {
            try { (void)resource->shutdown(); } catch (...) {}
        }
        resource_initialization_order_.pop_back();
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

Status TuttiRuntime::register_datapath_(
    std::string id, std::unique_ptr<DataPath> data_path, DataPath*& borrowed) {
    borrowed = nullptr;
    if (id.empty() || !data_path) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "DataPath registration is incomplete");
    }
    if (state_ != TuttiRuntimeState::INITIALIZING) {
        return lifecycle_error(StatusCode::BUSY,
                               "TuttiRuntime cannot register DataPath");
    }
    if (datapaths_.count(id) != 0) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "TuttiRuntime DataPath ID is already registered");
    }
    DataPath* candidate = data_path.get();
    auto inserted = datapaths_.emplace(id, std::move(data_path));
    try {
        datapath_registration_order_.push_back(id);
    } catch (...) {
        if (!datapath_registration_order_.empty() &&
            datapath_registration_order_.back() == id) {
            datapath_registration_order_.pop_back();
        }
        if (inserted.second && inserted.first->second.get() == candidate)
            datapaths_.erase(inserted.first);
        throw;
    }
    borrowed = candidate;
    return Status::Ok();
}

Status TuttiRuntime::register_resolver_(
    std::string id, std::unique_ptr<StorageTargetResolver> resolver,
    StorageTargetResolver*& borrowed) {
    borrowed = nullptr;
    if (id.empty() || !resolver) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "resolver registration is incomplete");
    }
    if (state_ != TuttiRuntimeState::INITIALIZING) {
        return lifecycle_error(StatusCode::BUSY,
                               "TuttiRuntime cannot register resolver");
    }
    if (resolvers_.count(id) != 0) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "TuttiRuntime resolver ID is already registered");
    }
    StorageTargetResolver* candidate = resolver.get();
    auto inserted = resolvers_.emplace(id, std::move(resolver));
    try {
        resolver_registration_order_.push_back(id);
    } catch (...) {
        if (!resolver_registration_order_.empty() &&
            resolver_registration_order_.back() == id) {
            resolver_registration_order_.pop_back();
        }
        if (inserted.second && inserted.first->second.get() == candidate)
            resolvers_.erase(inserted.first);
        throw;
    }
    borrowed = candidate;
    return Status::Ok();
}

Status TuttiRuntime::set_storage_runtime_(
    std::unique_ptr<StorageRuntime> runtime) {
    if (!runtime) {
        return lifecycle_error(StatusCode::INVALID_ARGUMENT,
                               "cannot register a null StorageRuntime");
    }
    if (state_ != TuttiRuntimeState::INITIALIZING || runtime_) {
        return lifecycle_error(StatusCode::BUSY,
                               "TuttiRuntime already owns a StorageRuntime");
    }
    runtime_ = std::move(runtime);
    return Status::Ok();
}

void TuttiRuntime::observe_(TuttiRuntimeShutdownStage stage) noexcept {
    if (!shutdown_observer_) return;
    try {
        shutdown_observer_(stage);
    } catch (...) {
        // Observers are diagnostics only and must never interrupt cleanup.
    }
}

Status TuttiRuntime::shutdown(std::uint64_t drain_timeout_ms) {
    if (state_ == TuttiRuntimeState::STOPPED) {
        return Status::Ok();
    }
    state_ = TuttiRuntimeState::SHUTTING_DOWN;

    Status first_error;

    observe_(TuttiRuntimeShutdownStage::STORAGE_RUNTIME_SHUTDOWN);
    if (runtime_) {
        Status status;
        try {
            status = runtime_shutdown_hook_
                ? runtime_shutdown_hook_(*runtime_)
                : runtime_->shutdown(drain_timeout_ms);
        } catch (...) {
            status = lifecycle_error(StatusCode::INTERNAL,
                                     "StorageRuntime shutdown threw");
        }
        if (status.code() == StatusCode::TIMEOUT) {
            /*
             * 提前返回，**不销毁任何东西**。
             *
             * 在途 I/O 仍可能正在 DMA 到 DataPath 与 Resource 所拥有的内存，
             * 销毁它们会把 StorageRuntime 内部那份"宁可泄漏也不 UAF"的保护
             * 抵消掉。状态留在 SHUTTING_DOWN，使调用方能在 I/O 完成后重试，
             * 无需重建 runtime。
             *
             * 注意这里连 runtime_ 都不能 reset：它持有 datapath/resolver 的
             * 借用指针，先于它们消亡即是悬垂。
             */
            return status;
        }
        keep_first_error(first_error, std::move(status));
        runtime_.reset();
    }
    observe_(TuttiRuntimeShutdownStage::STORAGE_RUNTIME_DESTROYED);

    // Destroy owned components in reverse construction order while their
    // route bindings are no longer used by StorageRuntime.
    for (auto id = resolver_registration_order_.rbegin();
         id != resolver_registration_order_.rend(); ++id) {
        resolvers_.erase(*id);
    }
    resolver_registration_order_.clear();
    observe_(TuttiRuntimeShutdownStage::RESOLVERS_DESTROYED);
    for (auto id = datapath_registration_order_.rbegin();
         id != datapath_registration_order_.rend(); ++id) {
        datapaths_.erase(*id);
    }
    datapath_registration_order_.clear();
    observe_(TuttiRuntimeShutdownStage::DATAPATHS_DESTROYED);

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
        resources_.erase(resource);
    }
    observe_(TuttiRuntimeShutdownStage::RESOURCE_SHUTDOWN);
    resource_initialization_order_.clear();

    state_ = TuttiRuntimeState::STOPPED;
    observe_(TuttiRuntimeShutdownStage::COMPLETE);
    return first_error;
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

} // namespace tutti

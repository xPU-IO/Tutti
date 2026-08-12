#include "tutti/resource/memory/memory_resource.h"

#include <utility>

namespace tutti::resources::memory {
namespace {

Status error(StatusCode code, std::string message) {
    return Status(code, std::move(message));
}

} // namespace

MemoryResource::MemoryResource(std::string id, std::uint64_t capacity_bytes)
    : id_(std::move(id)), capacity_bytes_(capacity_bytes) {}

Result<std::unique_ptr<MemoryResource>> create_memory_resource(
    std::string resource_id,
    const config::MemoryResourceConfig& config) {
    if (resource_id.empty()) {
        return Result<std::unique_ptr<MemoryResource>>::Failure(
            error(StatusCode::INVALID_ARGUMENT,
                  "memory Resource ID must not be empty"));
    }
    if (config.capacity_bytes == 0) {
        return Result<std::unique_ptr<MemoryResource>>::Failure(
            error(StatusCode::INVALID_ARGUMENT,
                  "memory Resource capacity must be greater than zero"));
    }
    return Result<std::unique_ptr<MemoryResource>>::Success(
        std::unique_ptr<MemoryResource>(
            new MemoryResource(std::move(resource_id),
                               config.capacity_bytes)));
}

const ResourceCapabilities& MemoryResource::capabilities() const {
    static const ResourceCapabilities capabilities{
        "memory",
        true,
        true,
    };
    return capabilities;
}

Status MemoryResource::initialize() {
    if (state_ != ResourceState::CREATED) {
        return error(StatusCode::BUSY,
                     "memory Resource initialize called outside CREATED state");
    }
    state_ = ResourceState::INITIALIZED;
    return Status::Ok();
}

Status MemoryResource::shutdown() {
    if (state_ == ResourceState::STOPPED) return Status::Ok();
    if (state_ == ResourceState::SHUTTING_DOWN) {
        return error(StatusCode::BUSY,
                     "memory Resource shutdown is already in progress");
    }
    state_ = ResourceState::SHUTTING_DOWN;
    state_ = ResourceState::STOPPED;
    return Status::Ok();
}

ResourceInfo MemoryResource::info() const {
    return ResourceInfo{id_, "memory", state_};
}

Result<std::unique_ptr<const ResourceView>>
MemoryResource::get_resolver_view() const {
    auto view = memory_view();
    if (!view.ok()) {
        return Result<std::unique_ptr<const ResourceView>>::Failure(
            view.status());
    }
    auto typed_result = std::make_unique<MemoryResourceView>();
    typed_result->capacity_bytes = view.value().capacity_bytes;
    std::unique_ptr<const ResourceView> result = std::move(typed_result);
    return Result<std::unique_ptr<const ResourceView>>::Success(
        std::move(result));
}

Result<std::unique_ptr<const ResourceView>>
MemoryResource::get_datapath_view() const {
    auto view = memory_view();
    if (!view.ok()) {
        return Result<std::unique_ptr<const ResourceView>>::Failure(
            view.status());
    }
    auto typed_result = std::make_unique<MemoryResourceView>();
    typed_result->capacity_bytes = view.value().capacity_bytes;
    std::unique_ptr<const ResourceView> result = std::move(typed_result);
    return Result<std::unique_ptr<const ResourceView>>::Success(
        std::move(result));
}

Result<MemoryResourceView> MemoryResource::memory_view() const {
    if (state_ != ResourceState::INITIALIZED) {
        return Result<MemoryResourceView>::Failure(
            error(StatusCode::NOT_READY,
                  "memory Resource view requires INITIALIZED resource"));
    }
    MemoryResourceView view;
    view.capacity_bytes = capacity_bytes_;
    return Result<MemoryResourceView>::Success(std::move(view));
}

} // namespace tutti::resources::memory

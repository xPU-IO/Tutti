#include "tutti/resource/nvme/nvme_resource_internal.h"

#include <algorithm>
#include <exception>
#include <utility>

namespace tutti::resources::nvme {
namespace {

Status error(StatusCode code, std::string message) {
    return Status(code, std::move(message));
}

bool contains_accelerator(const std::vector<std::int32_t>& ids,
                          std::int32_t accel_id) {
    return std::find(ids.begin(), ids.end(), accel_id) != ids.end();
}

Status validate_provider_snapshot(
    std::int32_t runtime_accel_id,
    const config::NvmeAllocationSpec& allocation,
    const std::vector<NvmeAcceleratorInfo>& accelerators,
    const std::vector<NvmeProviderResource>& resources) {
    const auto accelerator = std::find_if(
        accelerators.begin(), accelerators.end(),
        [&](const NvmeAcceleratorInfo& row) {
            return row.accel_id == runtime_accel_id;
        });
    if (accelerator == accelerators.end()) {
        return error(StatusCode::NOT_FOUND,
                     "daemon does not advertise requested accel_id");
    }
    if (accelerator->view_root.empty()) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "daemon accelerator snapshot has empty view_root");
    }

    if (allocation.selection == config::NvmeSelection::Allowed) {
        const auto usable = std::find_if(
            resources.begin(), resources.end(),
            [&](const NvmeProviderResource& row) {
                return row.available && contains_accelerator(
                    row.allowed_accel_ids, runtime_accel_id);
            });
        if (usable == resources.end()) {
            return error(StatusCode::NOT_READY,
                         "daemon has no available NVMe resource for accel_id");
        }
        return Status::Ok();
    }

    for (std::int32_t device_id : allocation.device_ids) {
        const auto resource = std::find_if(
            resources.begin(), resources.end(),
            [&](const NvmeProviderResource& row) {
                return row.device_id == device_id;
            });
        if (resource == resources.end()) {
            return error(StatusCode::NOT_FOUND,
                         "requested NVMe device is not advertised by daemon");
        }
        if (!contains_accelerator(resource->allowed_accel_ids,
                                  runtime_accel_id)) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "requested NVMe resource ACL excludes accel_id");
        }
        if (!resource->available) {
            return error(StatusCode::NOT_READY,
                         "requested NVMe resource is not available");
        }
    }
    return Status::Ok();
}

} // namespace

struct NvmeResource::Impl {
    explicit Impl(std::unique_ptr<NvmeResourceClient> source_client)
        : client(std::move(source_client)) {}

    std::unique_ptr<NvmeResourceClient> client;
    RuntimeNvmeAllocation allocation;
    ResourceState state = ResourceState::CREATED;
    NvmeLeaseState lease_state = NvmeLeaseState::NONE;
};

NvmeResource::NvmeResource(std::string id,
                           std::int32_t runtime_accel_id,
                           config::NvmeResourceConfig config,
                           std::unique_ptr<Impl> impl)
    : id_(std::move(id)),
      runtime_accel_id_(runtime_accel_id),
      allocation_(std::move(config.allocation)),
      impl_(std::move(impl)) {}

NvmeResource::~NvmeResource() {
    (void)shutdown();
}

Result<std::unique_ptr<NvmeResource>> create_nvme_resource(
    std::string resource_id,
    const config::NvmeResourceConfig& config,
    std::int32_t runtime_accel_id) {
    if (resource_id.empty()) {
        return Result<std::unique_ptr<NvmeResource>>::Failure(
            error(StatusCode::INVALID_ARGUMENT,
                  "NVMe Resource ID must not be empty"));
    }
    auto client = make_nvme_resource_client(config.provider.endpoint);
    if (!client) {
        return Result<std::unique_ptr<NvmeResource>>::Failure(
            error(StatusCode::UNSUPPORTED,
                  "nvmeservice client is not available in this build"));
    }
    auto impl = std::make_unique<NvmeResource::Impl>(std::move(client));
    return Result<std::unique_ptr<NvmeResource>>::Success(
        std::unique_ptr<NvmeResource>(
            new NvmeResource(std::move(resource_id), runtime_accel_id,
                             config, std::move(impl))));
}

std::unique_ptr<NvmeResource> NvmeResourceTestingAccess::make(
    std::string resource_id,
    const config::NvmeResourceConfig& config,
    std::int32_t runtime_accel_id,
    std::unique_ptr<NvmeResourceClient> client) {
    auto impl = std::make_unique<NvmeResource::Impl>(std::move(client));
    return std::unique_ptr<NvmeResource>(
        new NvmeResource(std::move(resource_id), runtime_accel_id,
                         config, std::move(impl)));
}

NvmeResourceInspection NvmeResourceTestingAccess::inspection(
    const NvmeResource& resource) {
    return NvmeResourceInspection{
        resource.impl_->lease_state,
        resource.impl_->allocation,
    };
}

const ResourceCapabilities& NvmeResource::capabilities() const {
    static const ResourceCapabilities capabilities{
        "nvme",
        true,
        true,
    };
    return capabilities;
}

Status NvmeResource::validate_allocation_metadata_() const {
    if (impl_->allocation.allocation_id.empty()) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "AcquireNvmeSlices returned empty allocation_id");
    }
    if (impl_->allocation.slices.empty()) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "AcquireNvmeSlices returned no slices");
    }

    if (allocation_.selection == config::NvmeSelection::Striped) {
        if (impl_->allocation.slices.size() !=
            allocation_.device_ids.size()) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "striped allocation slice count does not match request");
        }
        for (std::size_t index = 0;
             index < impl_->allocation.slices.size(); ++index) {
            if (impl_->allocation.slices[index].device_id !=
                allocation_.device_ids[index]) {
                return error(StatusCode::INVALID_ARGUMENT,
                             "striped allocation slice order does not match request");
            }
        }
    } else if (impl_->allocation.slices.size() != 1) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "single-device selection must return exactly one slice");
    } else if (allocation_.selection == config::NvmeSelection::Explicit) {
        if (allocation_.device_ids.size() != 1 ||
            impl_->allocation.slices.front().device_id !=
                allocation_.device_ids.front()) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "explicit allocation device does not match request");
        }
    }

    const std::uint32_t logical_block_size =
        impl_->allocation.slices.front().logical_block_size;
    for (const RuntimeNvmeSlice& slice : impl_->allocation.slices) {
        if (slice.device_id < 0) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "NVMe allocation slice has invalid device_id");
        }
        if (slice.accel_id != runtime_accel_id_) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "NVMe allocation slice accel_id does not match Runtime");
        }
        if (!contains_accelerator(slice.allowed_accel_ids,
                                  runtime_accel_id_)) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "NVMe allocation slice ACL excludes Runtime accelerator");
        }
        if (slice.pci_bdf.empty() || slice.chrdev_path.empty() ||
            slice.block_path.empty() || slice.backing_mount_path.empty() ||
            slice.view_path.empty()) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "NVMe allocation slice is missing required path metadata");
        }
        if (slice.namespace_id == 0) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "NVMe allocation slice has invalid namespace_id");
        }
        if (slice.logical_block_size == 0) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "NVMe allocation slice has invalid logical block size");
        }
        if (slice.bar0_size == 0) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "NVMe allocation slice has invalid BAR0 size");
        }
        if (slice.max_data_size == 0) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "NVMe allocation slice has invalid max data size");
        }
        if (slice.granted_queues == 0) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "NVMe allocation slice has no granted queues");
        }
        if (slice.logical_block_size != logical_block_size) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "allocation slices have inconsistent logical block sizes");
        }
        if (allocation_.queues_per_controller > 0 &&
            slice.granted_queues > static_cast<std::uint32_t>(
                allocation_.queues_per_controller)) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "allocation granted queues exceed request");
        }
    }
    return Status::Ok();
}

Status NvmeResource::initialize() {
    if (impl_->state != ResourceState::CREATED) {
        return error(StatusCode::BUSY,
                     "NVMe resource initialize called outside CREATED state");
    }
    if (!impl_->client) {
        impl_->state = ResourceState::FAILED;
        return error(StatusCode::INVALID_ARGUMENT,
                     "NVMe resource requires a client");
    }

    try {
        auto accelerators = impl_->client->list_accelerators();
        if (!accelerators.ok()) {
            impl_->state = ResourceState::FAILED;
            return accelerators.status();
        }
        auto resources = impl_->client->list_nvme_resources();
        if (!resources.ok()) {
            impl_->state = ResourceState::FAILED;
            return resources.status();
        }
        Status status = validate_provider_snapshot(
            runtime_accel_id_, allocation_, accelerators.value(),
            resources.value());
        if (!status.ok()) {
            impl_->state = ResourceState::FAILED;
            return status;
        }

        auto acquired = impl_->client->acquire_nvme_slices(
            runtime_accel_id_,
            allocation_.selection,
            allocation_.device_ids,
            allocation_.queues_per_controller);
        if (!acquired.ok()) {
            impl_->state = ResourceState::FAILED;
            return acquired.status();
        }

        impl_->allocation = std::move(acquired).value();
        impl_->lease_state = NvmeLeaseState::ACQUIRED;
        status = validate_allocation_metadata_();
        if (!status.ok()) {
            (void)release_owned_allocation_();
            impl_->state = ResourceState::FAILED;
            return status;
        }
        impl_->state = ResourceState::INITIALIZED;
        return Status::Ok();
    } catch (const std::exception& exception) {
        if (impl_->lease_state == NvmeLeaseState::ACQUIRED) {
            (void)release_owned_allocation_();
        }
        impl_->state = ResourceState::FAILED;
        return error(StatusCode::INTERNAL,
                     std::string("NVMe resource initialize threw: ") +
                         exception.what());
    } catch (...) {
        if (impl_->lease_state == NvmeLeaseState::ACQUIRED) {
            (void)release_owned_allocation_();
        }
        impl_->state = ResourceState::FAILED;
        return error(StatusCode::INTERNAL,
                     "NVMe resource initialize threw");
    }
}

Status NvmeResource::release_owned_allocation_() {
    if (impl_->lease_state != NvmeLeaseState::ACQUIRED) return Status::Ok();
    impl_->lease_state = NvmeLeaseState::RELEASING;

    Status status;
    if (!impl_->client) {
        status = error(StatusCode::INTERNAL,
                       "NVMe allocation exists without its client");
    } else {
        try {
            status = impl_->client->release(impl_->allocation.allocation_id);
        } catch (const std::exception& exception) {
            status = error(StatusCode::INTERNAL,
                           std::string("NVMe resource release threw: ") +
                               exception.what());
        } catch (...) {
            status = error(StatusCode::INTERNAL,
                           "NVMe resource release threw");
        }
    }

    impl_->lease_state = NvmeLeaseState::RELEASED;
    impl_->client.reset();
    return status;
}

Status NvmeResource::shutdown() {
    if (impl_->state == ResourceState::STOPPED) return Status::Ok();
    if (impl_->state == ResourceState::SHUTTING_DOWN) {
        return error(StatusCode::BUSY,
                     "NVMe resource shutdown is already in progress");
    }

    if (impl_->lease_state != NvmeLeaseState::ACQUIRED) {
        impl_->client.reset();
        if (impl_->state == ResourceState::CREATED) {
            impl_->state = ResourceState::STOPPED;
        }
        return Status::Ok();
    }

    impl_->state = ResourceState::SHUTTING_DOWN;
    Status status = release_owned_allocation_();
    impl_->state = status.ok() ? ResourceState::STOPPED
                               : ResourceState::FAILED;
    return status;
}

ResourceInfo NvmeResource::info() const {
    return ResourceInfo{id_, "nvme", impl_->state};
}

Result<std::unique_ptr<const ResourceView>>
NvmeResource::get_resolver_view() const {
    auto view = resolver_view();
    if (!view.ok()) {
        return Result<std::unique_ptr<const ResourceView>>::Failure(
            view.status());
    }
    auto typed_result = std::make_unique<NvmeResolverResourceView>();
    typed_result->slices = std::move(view).value().slices;
    std::unique_ptr<const ResourceView> result = std::move(typed_result);
    return Result<std::unique_ptr<const ResourceView>>::Success(
        std::move(result));
}

Result<std::unique_ptr<const ResourceView>>
NvmeResource::get_datapath_view() const {
    auto view = datapath_view();
    if (!view.ok()) {
        return Result<std::unique_ptr<const ResourceView>>::Failure(
            view.status());
    }
    auto typed_result = std::make_unique<NvmeDataPathResourceView>();
    typed_result->slices = std::move(view).value().slices;
    std::unique_ptr<const ResourceView> result = std::move(typed_result);
    return Result<std::unique_ptr<const ResourceView>>::Success(
        std::move(result));
}

Result<NvmeResolverResourceView> NvmeResource::resolver_view() const {
    if (impl_->state != ResourceState::INITIALIZED) {
        return Result<NvmeResolverResourceView>::Failure(
            error(StatusCode::NOT_READY,
                  "NVMe resolver view requires INITIALIZED resource"));
    }
    NvmeResolverResourceView view;
    view.slices.reserve(impl_->allocation.slices.size());
    for (const RuntimeNvmeSlice& slice : impl_->allocation.slices) {
        view.slices.push_back({
            slice.device_id,
            slice.pci_bdf,
            slice.block_path,
            slice.backing_mount_path,
            slice.view_path,
            slice.namespace_id,
            slice.logical_block_size,
        });
    }
    return Result<NvmeResolverResourceView>::Success(std::move(view));
}

Result<NvmeDataPathResourceView> NvmeResource::datapath_view() const {
    if (impl_->state != ResourceState::INITIALIZED) {
        return Result<NvmeDataPathResourceView>::Failure(
            error(StatusCode::NOT_READY,
                  "NVMe DataPath view requires INITIALIZED resource"));
    }
    NvmeDataPathResourceView view;
    view.slices.reserve(impl_->allocation.slices.size());
    for (const RuntimeNvmeSlice& slice : impl_->allocation.slices) {
        view.slices.push_back({
            slice.device_id,
            slice.pci_bdf,
            slice.accel_id,
            slice.chrdev_path,
            slice.namespace_id,
            slice.logical_block_size,
            slice.bar0_size,
            slice.max_data_size,
            slice.granted_queues,
        });
    }
    return Result<NvmeDataPathResourceView>::Success(std::move(view));
}

} // namespace tutti::resources::nvme

#include "csrc/resource/nvme/nvme_resource_internal.h"

#include <unordered_map>
#include <utility>

#if defined(TUTTI_NVME_RESOURCE_HAS_NVMESERVICE)
#include "nvmeservice_client.h"
#endif

namespace tutti::resources::nvme {
namespace {

#if defined(TUTTI_NVME_RESOURCE_HAS_NVMESERVICE)
Status error(StatusCode code, std::string message) {
    return Status(code, std::move(message));
}

nvmeservice::ClientSelectionMode to_client_selection(
    config::NvmeSelection selection) {
    switch (selection) {
        case config::NvmeSelection::Allowed:
            return nvmeservice::ClientSelectionMode::Allowed;
        case config::NvmeSelection::Explicit:
            return nvmeservice::ClientSelectionMode::Explicit;
        case config::NvmeSelection::Striped:
            return nvmeservice::ClientSelectionMode::Striped;
    }
    return nvmeservice::ClientSelectionMode::Allowed;
}

class GrpcNvmeResourceClient final : public NvmeResourceClient {
public:
    explicit GrpcNvmeResourceClient(std::string endpoint)
        : client_(std::move(endpoint)) {}

    Result<std::vector<NvmeAcceleratorInfo>>
    list_accelerators() override {
        auto source = client_.list_accelerators();
        if (source.empty()) {
            return Result<std::vector<NvmeAcceleratorInfo>>::Failure(
                error(StatusCode::NOT_READY,
                      "ListAccelerators returned no accelerators"));
        }
        std::vector<NvmeAcceleratorInfo> result;
        result.reserve(source.size());
        for (const auto& row : source) {
            result.push_back({row.accel_id, row.view_root});
        }
        return Result<std::vector<NvmeAcceleratorInfo>>::Success(
            std::move(result));
    }

    Result<std::vector<NvmeProviderResource>>
    list_nvme_resources() override {
        auto source = client_.list_nvme_resources();
        if (source.empty()) {
            return Result<std::vector<NvmeProviderResource>>::Failure(
                error(StatusCode::NOT_READY,
                      "ListNvmeResources returned no NVMe resources"));
        }
        std::vector<NvmeProviderResource> result;
        result.reserve(source.size());
        for (const auto& row : source) {
            result.push_back({
                row.device_id,
                row.allowed_accel_ids,
                row.available,
            });
        }
        return Result<std::vector<NvmeProviderResource>>::Success(
            std::move(result));
    }

    Result<RuntimeNvmeAllocation> acquire_nvme_slices(
        std::int32_t accel_id,
        config::NvmeSelection selection,
        const std::vector<std::int32_t>& device_ids,
        std::int32_t queues_per_controller) override {
        auto allocation = client_.acquire_nvme_slices(
            accel_id, to_client_selection(selection), device_ids,
            queues_per_controller);
        if (!allocation) {
            return Result<RuntimeNvmeAllocation>::Failure(
                error(StatusCode::NOT_READY,
                      "AcquireNvmeSlices failed or was rejected"));
        }

        RuntimeNvmeAllocation result;
        result.allocation_id = allocation->allocation_id;
        result.slices.reserve(allocation->slices.size());
        for (const auto& source : allocation->slices) {
            RuntimeNvmeSlice slice;
            slice.device_id = source.device_id;
            slice.accel_id = source.accel_id;
            slice.pci_bdf = source.pci_bdf;
            slice.chrdev_path = source.chrdev_path;
            slice.block_path = source.block_path;
            slice.backing_mount_path = source.backing_mount_path;
            slice.view_path = source.view_path;
            slice.namespace_id = source.namespace_id;
            slice.logical_block_size = source.logical_block_size;
            slice.max_data_size = source.max_data_size;
            slice.granted_queues = source.granted_queues;
            slice.allowed_accel_ids = source.allowed_accel_ids;
            result.slices.push_back(std::move(slice));
        }
        allocations_.emplace(result.allocation_id, std::move(allocation));
        return Result<RuntimeNvmeAllocation>::Success(
            std::move(result));
    }

    Status release(const std::string& allocation_id) override {
        const auto allocation = allocations_.find(allocation_id);
        if (allocation == allocations_.end()) return Status::Ok();

        // Ask the daemon first, then drop the handle only if it confirmed.
        //
        // The previous order was erase-then-OK: erasing destroys the Allocation,
        // whose destructor sends Release and can only print on failure, so this
        // returned Ok() unconditionally. A failed release therefore looked like
        // a clean shutdown while the daemon kept the slices reserved -- the
        // caller marked the lease RELEASED and dropped the client, and the
        // capacity stayed unavailable until the heartbeat reaper collected it.
        // Not named `error`: a local of that name would shadow the error()
        // helper this function needs to build the failure status.
        std::string release_error;
        if (!allocation->second->release(&release_error)) {
            // Keep the handle so the caller can retry; NvmeResource propagates
            // this status out of release_owned_allocation_().
            return error(StatusCode::INTERNAL,
                         "NVMe allocation release failed: " + release_error);
        }
        allocations_.erase(allocation);
        return Status::Ok();
    }

private:
    nvmeservice::NvmeServiceClient client_;
    std::unordered_map<
        std::string,
        std::unique_ptr<nvmeservice::NvmeServiceClient::Allocation>>
        allocations_;
};
#endif

} // namespace

std::unique_ptr<NvmeResourceClient> make_nvme_resource_client(
    const std::string& endpoint) {
#if defined(TUTTI_NVME_RESOURCE_HAS_NVMESERVICE)
    return std::make_unique<GrpcNvmeResourceClient>(endpoint);
#else
    (void)endpoint;
    return nullptr;
#endif
}

} // namespace tutti::resources::nvme

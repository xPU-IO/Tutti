// tutti/config/tutti_config.cpp
//
// Allocation-driven config loader. Parse/resolve logic lives in
// tutti_config_parse.cpp (pure host).

#include "tutti/config/tutti_config.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tutti/bindings/ext4_local_nvme/binding.h>
#include <tutti/bindings/striped_local_nvme/binding.h>
#include <tutti/cuda_like.h>
#include <tutti/io_types.h>
#include <tutti/resolvers/local_file/resolver.h>
#include <tutti/resolvers/striped_file/resolver.h>
#include <tutti/storage_runtime.h>
#include "tutti/data_paths/local_nvme/local_nvme_data_path.h"
#include "tutti/data_paths/striped_local_nvme/striped_data_path.h"

#if defined(TUTTI_CONFIG_HAS_NVMESERVICE)
#include "nvmeservice_client.h"
#endif

namespace tutti::config {
namespace {

Status error(StatusCode code, std::string message) {
    return Status(code, std::move(message));
}

std::string upper(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return value;
}

bool contains_accel(const std::vector<std::int32_t>& ids,
                    std::int32_t accel_id) {
    return std::find(ids.begin(), ids.end(), accel_id) != ids.end();
}

Result<int> default_backend_device_count() {
#if defined(TUTTI_USE_HOST)
    return Result<int>::Success(0);
#elif defined(TUTTI_USE_CUDA) || defined(TUTTI_USE_MUSA) || defined(TUTTI_USE_MACA)
    int count = 0;
    const auto rc = cudaGetDeviceCount(&count);
    if (rc != cudaSuccess) {
        return Result<int>::Failure(
            error(StatusCode::NOT_FOUND,
                  "compiled accelerator backend has no available devices"));
    }
    return Result<int>::Success(count);
#else
    return Result<int>::Failure(
        error(StatusCode::UNSUPPORTED,
              "no compiled accelerator backend is available"));
#endif
}

#if defined(TUTTI_CONFIG_HAS_NVMESERVICE)
nvmeservice::ClientSelectionMode to_client_selection(NvmeSelection selection) {
    switch (selection) {
        case NvmeSelection::Allowed:
            return nvmeservice::ClientSelectionMode::Allowed;
        case NvmeSelection::Explicit:
            return nvmeservice::ClientSelectionMode::Explicit;
        case NvmeSelection::Striped:
            return nvmeservice::ClientSelectionMode::Striped;
    }
    return nvmeservice::ClientSelectionMode::Allowed;
}

class GrpcRuntimeResourceClient final : public RuntimeResourceClient {
public:
    explicit GrpcRuntimeResourceClient(std::string endpoint)
        : client_(std::move(endpoint)) {}

    Result<std::vector<RuntimeAcceleratorInfo>> list_accelerators() override {
        auto source = client_.list_accelerators();
        if (source.empty()) {
            return Result<std::vector<RuntimeAcceleratorInfo>>::Failure(
                error(StatusCode::NOT_READY,
                      "ListAccelerators returned no accelerators"));
        }
        std::vector<RuntimeAcceleratorInfo> out;
        out.reserve(source.size());
        for (const auto& row : source) {
            out.push_back({row.accel_id, row.view_root});
        }
        return Result<std::vector<RuntimeAcceleratorInfo>>::Success(
            std::move(out));
    }

    Result<std::vector<RuntimeNvmeResource>> list_nvme_resources() override {
        auto source = client_.list_nvme_resources();
        if (source.empty()) {
            return Result<std::vector<RuntimeNvmeResource>>::Failure(
                error(StatusCode::NOT_READY,
                      "ListNvmeResources returned no NVMe resources"));
        }
        std::vector<RuntimeNvmeResource> out;
        out.reserve(source.size());
        for (const auto& row : source) {
            RuntimeNvmeResource resource;
            resource.device_id = row.device_id;
            resource.allowed_accel_ids = row.allowed_accel_ids;
            resource.available = row.available;
            out.push_back(std::move(resource));
        }
        return Result<std::vector<RuntimeNvmeResource>>::Success(
            std::move(out));
    }

    Result<RuntimeNvmeAllocation> acquire_nvme_slices(
        std::int32_t accel_id,
        NvmeSelection selection,
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

        RuntimeNvmeAllocation out;
        out.allocation_id = allocation->allocation_id;
        out.slices.reserve(allocation->slices.size());
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
            slice.bar0_size = source.bar0_size;
            slice.max_data_size = source.max_data_size;
            slice.granted_queues = source.granted_queues;
            slice.allowed_accel_ids = source.allowed_accel_ids;
            out.slices.push_back(std::move(slice));
        }
        allocations_.emplace(out.allocation_id, std::move(allocation));
        return Result<RuntimeNvmeAllocation>::Success(std::move(out));
    }

    Status release(const std::string& allocation_id) override {
        const auto it = allocations_.find(allocation_id);
        if (it == allocations_.end()) {
            return Status::Ok();
        }
        allocations_.erase(it);
        return Status::Ok();
    }

private:
    nvmeservice::NvmeServiceClient client_;
    std::unordered_map<std::string,
                       std::unique_ptr<nvmeservice::NvmeServiceClient::Allocation>>
        allocations_;
};
#endif

std::unique_ptr<RuntimeResourceClient> default_resource_client_factory(
    const std::string& endpoint) {
#if defined(TUTTI_CONFIG_HAS_NVMESERVICE)
    return std::make_unique<GrpcRuntimeResourceClient>(endpoint);
#else
    (void)endpoint;
    return nullptr;
#endif
}

Result<std::unique_ptr<StorageRuntime>> default_runtime_factory(
    RuntimeConfig config, RuntimeComponents components) {
    return StorageRuntime::create(std::move(config), std::move(components));
}

Status validate_profile_and_accel(const ParsedConfig& parsed,
                                  const LoadTuttiConfigOptions& options) {
    if (upper(parsed.accelerator_profile) !=
        upper(TUTTI_COMPILED_ACCELERATOR_PROFILE)) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "accelerator.profile does not match compiled profile");
    }

    if (parsed.runtime_accel_id == -1) {
        if (upper(parsed.accelerator_profile) == "HOST") {
            return Status::Ok();
        }
        return error(StatusCode::INVALID_ARGUMENT,
                     "accelerator runtime requires non-negative accel_id");
    }

    auto count = options.backend_device_count
        ? options.backend_device_count()
        : default_backend_device_count();
    if (!count.ok()) return count.status();
    if (parsed.runtime_accel_id >= count.value()) {
        return error(StatusCode::NOT_FOUND,
                     "runtime.accel_id is outside compiled backend device count");
    }
    return Status::Ok();
}

Status validate_resource_snapshot(
    const ParsedConfig& parsed,
    const std::vector<RuntimeAcceleratorInfo>& accelerators,
    const std::vector<RuntimeNvmeResource>& resources) {
    const auto accel_it = std::find_if(
        accelerators.begin(), accelerators.end(),
        [&](const RuntimeAcceleratorInfo& row) {
            return row.accel_id == parsed.runtime_accel_id;
        });
    if (accel_it == accelerators.end()) {
        return error(StatusCode::NOT_FOUND,
                     "daemon does not advertise requested accel_id");
    }

    for (std::int32_t device_id : parsed.nvme_device_ids) {
        const auto resource_it = std::find_if(
            resources.begin(), resources.end(),
            [&](const RuntimeNvmeResource& row) {
                return row.device_id == device_id;
            });
        if (resource_it == resources.end()) {
            return error(StatusCode::NOT_FOUND,
                         "requested nvme.device_id is not advertised by daemon");
        }
        if (!contains_accel(resource_it->allowed_accel_ids,
                            parsed.runtime_accel_id)) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "requested NVMe resource ACL does not include accel_id");
        }
        if (!resource_it->available) {
            return error(StatusCode::NOT_READY,
                         "requested NVMe resource is not available");
        }
    }
    return Status::Ok();
}

Status validate_slice_common(const RuntimeNvmeSlice& slice,
                             std::int32_t expected_accel_id) {
    if (slice.accel_id != expected_accel_id) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "allocation slice accel_id does not match Runtime");
    }
    if (!contains_accel(slice.allowed_accel_ids, expected_accel_id)) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "allocation slice ACL does not include Runtime accel_id");
    }
    if (slice.chrdev_path.empty() || slice.block_path.empty() ||
        slice.pci_bdf.empty() || slice.namespace_id == 0 ||
        slice.logical_block_size == 0 || slice.bar0_size == 0) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "allocation slice is missing required metadata");
    }
    return Status::Ok();
}

Status validate_allocation(const ParsedConfig& parsed,
                           const RuntimeNvmeAllocation& allocation) {
    if (allocation.allocation_id.empty()) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "AcquireNvmeSlices returned empty allocation_id");
    }
    if (allocation.slices.empty()) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "AcquireNvmeSlices returned no slices");
    }

    if (parsed.nvme_selection == NvmeSelection::Striped) {
        if (allocation.slices.size() != parsed.nvme_device_ids.size()) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "striped allocation slice count does not match request");
        }
        for (std::size_t i = 0; i < allocation.slices.size(); ++i) {
            if (allocation.slices[i].device_id != parsed.nvme_device_ids[i]) {
                return error(StatusCode::INVALID_ARGUMENT,
                             "striped allocation slice order does not match request");
            }
        }
    } else if (allocation.slices.size() != 1) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "single-device selection must return exactly one slice");
    }

    const std::uint32_t block_size = allocation.slices.front().logical_block_size;
    for (const auto& slice : allocation.slices) {
        Status status = validate_slice_common(slice, parsed.runtime_accel_id);
        if (!status.ok()) return status;
        if (slice.logical_block_size != block_size) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "allocation slices have inconsistent logical block sizes");
        }
    }
    return Status::Ok();
}

std::uint32_t checked_u32(std::uint64_t value, const char* name) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds uint32_t");
    }
    return static_cast<std::uint32_t>(value);
}

void add_local_components(TuttiRuntime& tr,
                          RuntimeComponents& components,
                          const ParsedConfig& parsed,
                          const EffectiveCacheConfig& eff,
                          const RuntimeNvmeSlice& slice) {
    namespace local_dp = data_paths::local_nvme;
    namespace local_resolver = resolvers::local_file;
    namespace local_binding = tutti::binding::ext4_local_nvme;

    auto dp = std::make_unique<local_dp::LocalNvmeDataPath>(
        slice.chrdev_path,
        checked_u32(slice.bar0_size, "bar0_size"),
        static_cast<std::uint32_t>(slice.accel_id),
        slice.granted_queues,
        slice.namespace_id,
        slice.logical_block_size,
        slice.max_data_size,
        static_cast<std::uint32_t>(parsed.max_batch_entries),
        0,
        eff.handle_cache_capacity,
        eff.prp_cache_capacity,
        parsed.max_in_flight_operations,
        parsed.max_batch_entries,
        0,
        eff.handle_cache_l2_capacity);
    components.data_paths.push_back(
        {std::string(local_binding::kRecommendedDataPathKey), dp.get(),
         DataPathConfig{"local_nvme"}});
    tr.data_path_keys.push_back(
        std::string(local_binding::kRecommendedDataPathKey));
    tr.datapaths.push_back(std::move(dp));

    auto resolver = std::make_unique<local_resolver::LocalFileResolver>(
        slice.pci_bdf,
        slice.namespace_id,
        slice.logical_block_size,
        local_resolver::BackingDeviceConfig{slice.block_path, 0});
    components.resolvers.push_back({"file", resolver.get()});
    tr.resolver_schemes.push_back("file");
    tr.resolvers.push_back(std::move(resolver));
}

void add_striped_components(TuttiRuntime& tr,
                            RuntimeComponents& components,
                            const ParsedConfig& parsed,
                            const EffectiveCacheConfig& eff,
                            const std::vector<RuntimeNvmeSlice>& slices) {
    namespace striped_dp = data_paths::striped_local_nvme;
    namespace local_resolver = resolvers::local_file;
    namespace striped_resolver = resolvers::striped_file;
    namespace striped_binding = tutti::binding::striped_local_nvme;

    std::vector<striped_dp::DeviceDescriptor> descriptors;
    descriptors.reserve(slices.size());
    std::vector<std::unique_ptr<StorageTargetResolver>> shard_resolvers;
    shard_resolvers.reserve(slices.size());

    std::uint64_t mdts = 0;
    for (const auto& slice : slices) {
        descriptors.push_back({
            slice.chrdev_path,
            slice.bar0_size,
            slice.namespace_id,
            static_cast<std::uint32_t>(slice.accel_id),
            slice.granted_queues,
            slice.logical_block_size});
        if (slice.max_data_size != 0) {
            mdts = mdts == 0 ? slice.max_data_size
                             : std::min(mdts, slice.max_data_size);
        }
        shard_resolvers.push_back(
            std::make_unique<local_resolver::LocalFileResolver>(
                slice.pci_bdf,
                slice.namespace_id,
                slice.logical_block_size,
                local_resolver::BackingDeviceConfig{slice.block_path, 0}));
    }

    auto dp = std::make_unique<striped_dp::StripedDataPath>(
        std::move(descriptors),
        static_cast<std::uint32_t>(parsed.runtime_accel_id),
        mdts,
        0,
        static_cast<std::uint32_t>(parsed.max_batch_entries),
        static_cast<std::uint32_t>(parsed.max_in_flight_operations),
        eff.handle_cache_capacity,
        eff.prp_cache_capacity);
    components.data_paths.push_back(
        {std::string(striped_binding::kRecommendedDataPathKey), dp.get(),
         DataPathConfig{"striped-local-nvme"}});
    tr.data_path_keys.push_back(
        std::string(striped_binding::kRecommendedDataPathKey));
    tr.datapaths.push_back(std::move(dp));

    auto resolver = std::make_unique<striped_resolver::StripedResolver>(
        std::move(shard_resolvers), parsed.stripe_unit);
    components.resolvers.push_back({"striped", resolver.get()});
    tr.resolver_schemes.push_back("striped");
    tr.resolvers.push_back(std::move(resolver));
}

} // namespace

TuttiRuntime::~TuttiRuntime() {
    (void)shutdown();
}

Status TuttiRuntime::shutdown() {
    if (shutdown_complete_) return Status::Ok();

    Status first_error;
    if (runtime) {
        Status status = runtime->shutdown(0);
        if (!status.ok() && first_error.ok()) first_error = status;
        runtime.reset();
    }

    resolvers.clear();
    datapaths.clear();

    if (!allocation_id.empty() && !allocation_released_) {
        if (resource_client) {
            Status status = resource_client->release(allocation_id);
            if (!status.ok() && first_error.ok()) first_error = status;
        } else if (first_error.ok()) {
            first_error = error(StatusCode::INTERNAL,
                                "allocation exists without resource client");
        }
        allocation_released_ = true;
    }
    resource_client.reset();
    shutdown_complete_ = true;
    return first_error;
}

Result<std::unique_ptr<TuttiRuntime>> load_tutti_config(
    const std::string& path,
    const ProgrammaticOverrides& overrides) {
    LoadTuttiConfigOptions options;
    options.overrides = overrides;
    return load_tutti_config(path, std::move(options));
}

Result<std::unique_ptr<TuttiRuntime>> load_tutti_config(
    const std::string& path,
    LoadTuttiConfigOptions options) {
    auto parsed_result = parse_tutti_config(path);
    if (!parsed_result.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            parsed_result.status());
    }
    const auto& parsed = parsed_result.value();

    Status status = validate_profile_and_accel(parsed, options);
    if (!status.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(status);
    }

    RuntimeConfig runtime_config;
    runtime_config.accel_id = parsed.runtime_accel_id;
    runtime_config.profile_name = parsed.accelerator_profile;

    if (parsed.runtime_accel_id == -1) {
        auto created = options.runtime_factory
            ? options.runtime_factory(runtime_config, RuntimeComponents{})
            : StorageRuntime::create(runtime_config);
        if (!created.ok()) {
            return Result<std::unique_ptr<TuttiRuntime>>::Failure(
                created.status());
        }
        auto tr = std::make_unique<TuttiRuntime>();
        tr->runtime = std::move(created).value();
        return Result<std::unique_ptr<TuttiRuntime>>::Success(std::move(tr));
    }

    auto tr = std::make_unique<TuttiRuntime>();
    tr->resource_client = options.resource_client_factory
        ? options.resource_client_factory(parsed.nvme_service_endpoint)
        : default_resource_client_factory(parsed.nvme_service_endpoint);
    if (!tr->resource_client) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::UNSUPPORTED,
                  "nvmeservice client is not available in this build"));
    }

    auto accelerators = tr->resource_client->list_accelerators();
    if (!accelerators.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            accelerators.status());
    }
    auto resources = tr->resource_client->list_nvme_resources();
    if (!resources.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            resources.status());
    }
    status = validate_resource_snapshot(parsed, accelerators.value(),
                                        resources.value());
    if (!status.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(status);
    }

    auto allocation = tr->resource_client->acquire_nvme_slices(
        parsed.runtime_accel_id,
        parsed.nvme_selection,
        parsed.nvme_device_ids,
        parsed.queues_per_controller);
    if (!allocation.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            allocation.status());
    }
    status = validate_allocation(parsed, allocation.value());
    if (!status.ok()) {
        tr->allocation_id = allocation.value().allocation_id;
        tr->allocation_slices = allocation.value().slices;
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(status);
    }

    tr->allocation_id = allocation.value().allocation_id;
    tr->allocation_slices = allocation.value().slices;

    RuntimeComponents components;
    const auto eff = resolve_cache_config(parsed, options.overrides);
    try {
        if (tr->allocation_slices.size() == 1) {
            add_local_components(*tr, components, parsed, eff,
                                 tr->allocation_slices.front());
        } else {
            add_striped_components(*tr, components, parsed, eff,
                                   tr->allocation_slices);
        }
    } catch (const std::exception& e) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::INVALID_ARGUMENT, e.what()));
    }

    auto created = options.runtime_factory
        ? options.runtime_factory(runtime_config, std::move(components))
        : default_runtime_factory(runtime_config, std::move(components));
    if (!created.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            created.status());
    }
    tr->runtime = std::move(created).value();
    return Result<std::unique_ptr<TuttiRuntime>>::Success(std::move(tr));
}

} // namespace tutti::config

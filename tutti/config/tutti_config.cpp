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
#include "tutti/resource/nvme/nvme_resource.h"

namespace tutti::config {
namespace {

namespace nvme_resource = tutti::resources::nvme;
using nvme_resource::NvmeDataPathResourceView;
using nvme_resource::NvmeDataPathSliceView;
using nvme_resource::NvmeResolverResourceView;
using nvme_resource::NvmeResolverSliceView;

Status error(StatusCode code, std::string message) {
    return Status(code, std::move(message));
}

std::string upper(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return value;
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

Result<std::unique_ptr<StorageRuntime>> default_runtime_factory(
    RuntimeConfig config, RuntimeComponents components) {
    return StorageRuntime::create(std::move(config), std::move(components));
}

Result<std::unique_ptr<Resource>> default_resource_factory(
    const ResourceSpec& resource_spec,
    std::int32_t accel_id) {
    if (resource_spec.type != "nvme") {
        return Result<std::unique_ptr<Resource>>::Failure(
            error(StatusCode::UNSUPPORTED,
                  "configured Resource type is not implemented"));
    }
    auto created = nvme_resource::make_nvme_resource(
        nvme_resource::NvmeResourceSpec{
            resource_spec.id,
            accel_id,
            resource_spec.provider,
            resource_spec.allocation,
        });
    if (!created.ok()) {
        return Result<std::unique_ptr<Resource>>::Failure(created.status());
    }
    std::unique_ptr<Resource> resource = std::move(created).value();
    return Result<std::unique_ptr<Resource>>::Success(std::move(resource));
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
                          const NvmeResolverSliceView& resolver_slice,
                          const NvmeDataPathSliceView& datapath_slice) {
    namespace local_dp = data_paths::local_nvme;
    namespace local_resolver = resolvers::local_file;
    namespace local_binding = tutti::binding::ext4_local_nvme;

    auto dp = std::make_unique<local_dp::LocalNvmeDataPath>(
        datapath_slice.chrdev_path,
        checked_u32(datapath_slice.bar0_size, "bar0_size"),
        static_cast<std::uint32_t>(datapath_slice.accel_id),
        datapath_slice.granted_queues,
        datapath_slice.namespace_id,
        datapath_slice.logical_block_size,
        datapath_slice.max_data_size,
        static_cast<std::uint32_t>(parsed.max_batch_entries),
        0,
        eff.handle_cache_capacity,
        eff.prp_cache_capacity,
        parsed.max_in_flight_operations,
        parsed.max_batch_entries,
        0,
        eff.handle_cache_l2_capacity);
    auto* dp_ptr = tr.register_datapath(
        std::move(dp), std::string(local_binding::kRecommendedDataPathKey));
    if (dp_ptr == nullptr) {
        throw std::runtime_error("TuttiRuntime rejected DataPath registration");
    }
    components.data_paths.push_back(
        {std::string(local_binding::kRecommendedDataPathKey), dp_ptr,
         DataPathConfig{"local_nvme"}});

    auto resolver = std::make_unique<local_resolver::LocalFileResolver>(
        resolver_slice.pci_bdf,
        resolver_slice.namespace_id,
        resolver_slice.logical_block_size,
        local_resolver::BackingDeviceConfig{resolver_slice.block_path, 0});
    auto* resolver_ptr = tr.register_resolver(std::move(resolver), "file");
    if (resolver_ptr == nullptr) {
        throw std::runtime_error("TuttiRuntime rejected resolver registration");
    }
    components.resolvers.push_back({"file", resolver_ptr});
}

void add_striped_components(TuttiRuntime& tr,
                            RuntimeComponents& components,
                            const ParsedConfig& parsed,
                            const EffectiveCacheConfig& eff,
                            const NvmeResolverResourceView& resolver_view,
                            const NvmeDataPathResourceView& datapath_view) {
    namespace striped_dp = data_paths::striped_local_nvme;
    namespace local_resolver = resolvers::local_file;
    namespace striped_resolver = resolvers::striped_file;
    namespace striped_binding = tutti::binding::striped_local_nvme;

    std::vector<striped_dp::DeviceDescriptor> descriptors;
    descriptors.reserve(datapath_view.slices.size());
    std::vector<std::unique_ptr<StorageTargetResolver>> shard_resolvers;
    shard_resolvers.reserve(resolver_view.slices.size());

    std::uint64_t mdts = 0;
    for (std::size_t index = 0; index < datapath_view.slices.size(); ++index) {
        const NvmeDataPathSliceView& datapath_slice =
            datapath_view.slices[index];
        const NvmeResolverSliceView& resolver_slice =
            resolver_view.slices[index];
        descriptors.push_back({
            datapath_slice.chrdev_path,
            datapath_slice.bar0_size,
            datapath_slice.namespace_id,
            static_cast<std::uint32_t>(datapath_slice.accel_id),
            datapath_slice.granted_queues,
            datapath_slice.logical_block_size});
        if (datapath_slice.max_data_size != 0) {
            mdts = mdts == 0 ? datapath_slice.max_data_size
                             : std::min(mdts, datapath_slice.max_data_size);
        }
        shard_resolvers.push_back(
            std::make_unique<local_resolver::LocalFileResolver>(
                resolver_slice.pci_bdf,
                resolver_slice.namespace_id,
                resolver_slice.logical_block_size,
                local_resolver::BackingDeviceConfig{
                    resolver_slice.block_path, 0}));
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
    auto* dp_ptr = tr.register_datapath(
        std::move(dp), std::string(striped_binding::kRecommendedDataPathKey));
    if (dp_ptr == nullptr) {
        throw std::runtime_error("TuttiRuntime rejected DataPath registration");
    }
    components.data_paths.push_back(
        {std::string(striped_binding::kRecommendedDataPathKey), dp_ptr,
         DataPathConfig{"striped-local-nvme"}});

    auto resolver = std::make_unique<striped_resolver::StripedResolver>(
        std::move(shard_resolvers), parsed.stripe_unit);
    auto* resolver_ptr = tr.register_resolver(std::move(resolver), "striped");
    if (resolver_ptr == nullptr) {
        throw std::runtime_error("TuttiRuntime rejected resolver registration");
    }
    components.resolvers.push_back({"striped", resolver_ptr});
}

} // namespace

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
        tr->set_runtime_shutdown_hook(std::move(options.runtime_shutdown_hook));
        tr->set_shutdown_observer(std::move(options.shutdown_observer));
        return Result<std::unique_ptr<TuttiRuntime>>::Success(std::move(tr));
    }

    auto tr = std::make_unique<TuttiRuntime>();
    tr->set_runtime_shutdown_hook(std::move(options.runtime_shutdown_hook));
    tr->set_shutdown_observer(std::move(options.shutdown_observer));
    const ResourceSpec& resource_spec =
        parsed.canonical_storage.resources.front();
    auto created_resource = options.resource_factory
        ? options.resource_factory(resource_spec, parsed.runtime_accel_id)
        : default_resource_factory(resource_spec, parsed.runtime_accel_id);
    if (!created_resource.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            created_resource.status());
    }

    std::unique_ptr<Resource> resource =
        std::move(created_resource).value();
    auto* nvme = dynamic_cast<nvme_resource::NvmeResource*>(resource.get());
    if (nvme == nullptr) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::INVALID_ARGUMENT,
                  "Resource factory returned incompatible implementation"));
    }
    status = resource->initialize();
    if (!status.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(status);
    }

    auto resolver_view = nvme->resolver_view();
    if (!resolver_view.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            resolver_view.status());
    }
    auto datapath_view = nvme->datapath_view();
    if (!datapath_view.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            datapath_view.status());
    }
    if (resolver_view.value().slices.size() !=
        datapath_view.value().slices.size()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::INTERNAL,
                  "NVMe resource views have inconsistent cardinality"));
    }
    status = tr->adopt_resource(std::move(resource));
    if (!status.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(status);
    }

    RuntimeComponents components;
    const auto eff = resolve_cache_config(parsed, options.overrides);
    try {
        if (datapath_view.value().slices.size() == 1) {
            add_local_components(*tr, components, parsed, eff,
                                 resolver_view.value().slices.front(),
                                 datapath_view.value().slices.front());
        } else {
            add_striped_components(*tr, components, parsed, eff,
                                   resolver_view.value(),
                                   datapath_view.value());
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

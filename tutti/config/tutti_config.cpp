// tutti/config/tutti_config.cpp
//
// Allocation-driven config loader. Parse/resolve logic lives in
// tutti_config_parse.cpp (pure host).

#include "tutti/config/tutti_config.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
#include "tutti/tutti_runtime/tutti_runtime_internal.h"

namespace tutti::config {
namespace {

namespace nvme_resource = tutti::resources::nvme;
using nvme_resource::NvmeDataPathResourceView;
using nvme_resource::NvmeDataPathSliceView;
using nvme_resource::NvmeResolverResourceView;
using nvme_resource::NvmeResolverSliceView;

struct StorageIdTables {
    std::unordered_map<std::string, const ResourceSpec*> resources;
    std::unordered_map<std::string, const ResolverSpec*> resolvers;
    std::unordered_map<std::string, const DataPathSpec*> datapaths;
    std::unordered_map<std::string, const BackendSpec*> backends;
};

struct BackendComponents {
    StorageTargetResolver* resolver = nullptr;
    DataPath* datapath = nullptr;
};

Status error(StatusCode code, std::string message) {
    return Status(code, std::move(message));
}

template <typename Spec>
Status index_specs(const std::vector<Spec>& specs, const char* group,
                   std::unordered_map<std::string, const Spec*>& table) {
    for (const Spec& spec : specs) {
        if (spec.id.empty()) {
            return error(StatusCode::INVALID_ARGUMENT,
                         std::string("storage.") + group +
                             " contains an empty ID");
        }
        if (!table.emplace(spec.id, &spec).second) {
            return error(StatusCode::INVALID_ARGUMENT,
                         std::string("storage.") + group +
                             " contains duplicate ID " + spec.id);
        }
    }
    return Status::Ok();
}

Status validate_requested_cardinality(const ResourceSpec& resource,
                                      const BackendSpec& backend,
                                      const StorageContract& contract) {
    std::size_t requested_cardinality = 1;
    if (resource.type == "nvme" &&
        resource.allocation.selection == NvmeSelection::Striped) {
        requested_cardinality = resource.allocation.device_ids.size();
    }
    if (requested_cardinality < contract.minimum_cardinality ||
        requested_cardinality > contract.maximum_cardinality) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "backend " + backend.id +
                         " Resource request cardinality does not match contract " +
                         backend.contract);
    }
    if (contract.maximum_cardinality == 1 &&
        resource.allocation.selection == NvmeSelection::Striped) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "local backend cannot use striped Resource selection");
    }
    if (contract.minimum_cardinality >= 2 &&
        resource.allocation.selection != NvmeSelection::Striped) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "striped backend requires striped Resource selection");
    }
    return Status::Ok();
}

Status build_storage_id_tables(const CanonicalStorageConfig& storage,
                               StorageIdTables& tables) {
    if (!storage.present && storage.resources.empty() &&
        storage.resolvers.empty() && storage.datapaths.empty() &&
        storage.backends.empty()) {
        return Status::Ok();
    }

    Status status = index_specs(storage.resources, "resources",
                                tables.resources);
    if (!status.ok()) return status;
    status = index_specs(storage.resolvers, "resolvers", tables.resolvers);
    if (!status.ok()) return status;
    status = index_specs(storage.datapaths, "datapaths", tables.datapaths);
    if (!status.ok()) return status;
    status = index_specs(storage.backends, "backends", tables.backends);
    if (!status.ok()) return status;

    std::unordered_set<std::string> reachable_resources;
    std::unordered_set<std::string> reachable_resolvers;
    std::unordered_set<std::string> reachable_datapaths;
    std::unordered_map<std::string, std::string> resource_datapaths;
    for (const BackendSpec& backend : storage.backends) {
        const auto resource = tables.resources.find(backend.resource);
        const auto resolver = tables.resolvers.find(backend.resolver);
        const auto datapath = tables.datapaths.find(backend.datapath);
        const StorageContract* contract =
            find_storage_contract(backend.contract);
        if (resource == tables.resources.end() ||
            resolver == tables.resolvers.end() ||
            datapath == tables.datapaths.end()) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "backend contains a dangling storage ID reference");
        }
        if (contract == nullptr) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "backend references an unknown storage contract");
        }
        if (resource->second->type != contract->resource_type ||
            resolver->second->type != contract->resolver_type ||
            datapath->second->type != contract->datapath_type) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "backend storage types do not match its contract");
        }
        status = validate_requested_cardinality(
            *resource->second, backend, *contract);
        if (!status.ok()) return status;

        const auto consumer = resource_datapaths.emplace(
            backend.resource, backend.datapath);
        if (!consumer.second && consumer.first->second != backend.datapath) {
            return error(
                StatusCode::INVALID_ARGUMENT,
                "Resource cannot be consumed by independent DataPaths");
        }
        reachable_resources.emplace(backend.resource);
        reachable_resolvers.emplace(backend.resolver);
        reachable_datapaths.emplace(backend.datapath);
    }

    if (reachable_resources.size() != storage.resources.size() ||
        reachable_resolvers.size() != storage.resolvers.size() ||
        reachable_datapaths.size() != storage.datapaths.size()) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "all storage declarations must be backend-reachable");
    }
    if (storage.backends.size() != 1) {
        return error(StatusCode::INVALID_ARGUMENT,
                     "storage must contain exactly one backend");
    }
    return Status::Ok();
}

void discard_resource(std::unique_ptr<Resource>& resource) noexcept {
    if (!resource) return;
    try {
        (void)resource->shutdown();
    } catch (...) {
    }
    resource.reset();
}

Result<std::unique_ptr<Resource>> resolve_resource_spec(
    const ResourceSpec& spec, std::int32_t accel_id,
    const std::function<Result<std::unique_ptr<Resource>>(
        const ResourceSpec&, std::int32_t)>& factory) {
    Result<std::unique_ptr<Resource>> created =
        Result<std::unique_ptr<Resource>>::Failure(
            error(StatusCode::INTERNAL,
                  "ResourceFactory did not return a result"));
    try {
        created = factory(spec, accel_id);
    } catch (const std::exception& exception) {
        return Result<std::unique_ptr<Resource>>::Failure(
            error(StatusCode::INTERNAL,
                  std::string("ResourceFactory threw: ") + exception.what()));
    } catch (...) {
        return Result<std::unique_ptr<Resource>>::Failure(
            error(StatusCode::INTERNAL, "ResourceFactory threw"));
    }
    if (!created.ok()) return created;

    std::unique_ptr<Resource> resource = std::move(created).value();
    if (!resource) {
        return Result<std::unique_ptr<Resource>>::Failure(
            error(StatusCode::INVALID_ARGUMENT,
                  "ResourceFactory returned null"));
    }
    ResourceInfo info = resource->info();
    if (info.id != spec.id || info.type != spec.type ||
        resource->capabilities().resource_type != spec.type ||
        info.state != ResourceState::CREATED) {
        discard_resource(resource);
        return Result<std::unique_ptr<Resource>>::Failure(
            error(StatusCode::INVALID_ARGUMENT,
                  "ResourceFactory result does not match ResourceSpec"));
    }

    Status status;
    try {
        status = resource->initialize();
    } catch (const std::exception& exception) {
        discard_resource(resource);
        return Result<std::unique_ptr<Resource>>::Failure(
            error(StatusCode::INTERNAL,
                  std::string("Resource initialize threw: ") +
                      exception.what()));
    } catch (...) {
        discard_resource(resource);
        return Result<std::unique_ptr<Resource>>::Failure(
            error(StatusCode::INTERNAL, "Resource initialize threw"));
    }
    if (!status.ok()) {
        discard_resource(resource);
        return Result<std::unique_ptr<Resource>>::Failure(std::move(status));
    }
    info = resource->info();
    if (info.id != spec.id || info.type != spec.type ||
        info.state != ResourceState::INITIALIZED) {
        discard_resource(resource);
        return Result<std::unique_ptr<Resource>>::Failure(
            error(StatusCode::INVALID_ARGUMENT,
                  "Resource initialize did not produce the expected instance"));
    }
    return Result<std::unique_ptr<Resource>>::Success(std::move(resource));
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

BackendComponents add_local_components(
    TuttiRuntime& tr, RuntimeComponents& components,
    const ParsedConfig& parsed, const EffectiveCacheConfig& eff,
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
    return BackendComponents{resolver_ptr, dp_ptr};
}

BackendComponents add_striped_components(
    TuttiRuntime& tr, RuntimeComponents& components,
    const ParsedConfig& parsed, const EffectiveCacheConfig& eff,
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
    return BackendComponents{resolver_ptr, dp_ptr};
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

    StorageIdTables id_tables;
    Status status = build_storage_id_tables(parsed.canonical_storage,
                                             id_tables);
    if (!status.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(status);
    }

    status = validate_profile_and_accel(parsed, options);
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

    std::function<Result<std::unique_ptr<Resource>>(
        const ResourceSpec&, std::int32_t)> resource_factory =
        options.resource_factory
            ? std::move(options.resource_factory)
            : std::function<Result<std::unique_ptr<Resource>>(
                  const ResourceSpec&, std::int32_t)>(default_resource_factory);
    for (const BackendSpec& backend : parsed.canonical_storage.backends) {
        if (TuttiRuntimeAssemblyAccess::resource(*tr, backend.resource) !=
            nullptr) {
            continue;
        }
        const auto resource_spec = id_tables.resources.find(backend.resource);
        if (resource_spec == id_tables.resources.end()) {
            return Result<std::unique_ptr<TuttiRuntime>>::Failure(
                error(StatusCode::INTERNAL,
                      "backend Resource is missing from the ID table"));
        }
        auto resolved = resolve_resource_spec(
            *resource_spec->second, parsed.runtime_accel_id, resource_factory);
        if (!resolved.ok()) {
            return Result<std::unique_ptr<TuttiRuntime>>::Failure(
                resolved.status());
        }
        try {
            status = TuttiRuntimeAssemblyAccess::adopt_resource(
                *tr, backend.resource, std::move(resolved).value());
        } catch (const std::exception& exception) {
            return Result<std::unique_ptr<TuttiRuntime>>::Failure(
                error(StatusCode::INTERNAL,
                      std::string("Resource registry insertion threw: ") +
                          exception.what()));
        } catch (...) {
            return Result<std::unique_ptr<TuttiRuntime>>::Failure(
                error(StatusCode::INTERNAL,
                      "Resource registry insertion threw"));
        }
        if (!status.ok()) {
            return Result<std::unique_ptr<TuttiRuntime>>::Failure(status);
        }
    }

    const BackendSpec& backend = parsed.canonical_storage.backends.front();
    const Resource* resource =
        TuttiRuntimeAssemblyAccess::resource(*tr, backend.resource);
    const auto* nvme =
        dynamic_cast<const nvme_resource::NvmeResource*>(resource);
    if (nvme == nullptr) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::INVALID_ARGUMENT,
                  "ResourceFactory returned incompatible implementation"));
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
    RuntimeComponents components;
    const auto eff = resolve_cache_config(parsed, options.overrides);
    BackendComponents backend_components;
    try {
        if (backend.contract == "ext4-local-nvme") {
            if (datapath_view.value().slices.size() != 1) {
                return Result<std::unique_ptr<TuttiRuntime>>::Failure(
                    error(StatusCode::INVALID_ARGUMENT,
                          "local backend requires exactly one Resource slice"));
            }
            backend_components = add_local_components(
                *tr, components, parsed, eff,
                resolver_view.value().slices.front(),
                datapath_view.value().slices.front());
        } else if (backend.contract == "striped-local-nvme") {
            const auto resource_spec = id_tables.resources.find(
                backend.resource);
            const std::size_t requested =
                resource_spec->second->allocation.device_ids.size();
            if (requested < 2 ||
                datapath_view.value().slices.size() != requested) {
                return Result<std::unique_ptr<TuttiRuntime>>::Failure(
                    error(StatusCode::INVALID_ARGUMENT,
                          "striped backend Resource slice count does not match request"));
            }
            backend_components = add_striped_components(
                *tr, components, parsed, eff, resolver_view.value(),
                datapath_view.value());
        } else {
            return Result<std::unique_ptr<TuttiRuntime>>::Failure(
                error(StatusCode::UNSUPPORTED,
                      "backend contract factory is not implemented"));
        }
    } catch (const std::exception& e) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::INVALID_ARGUMENT, e.what()));
    }

    status = TuttiRuntimeAssemblyAccess::register_backend(
        *tr,
        BackendManifest{backend.id, backend.contract, backend.resolver,
                        backend.datapath, backend.resource},
        resource, backend_components.resolver, backend_components.datapath);
    if (!status.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(status);
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

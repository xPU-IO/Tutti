// tutti/config/tutti_config.cpp
//
// Allocation-driven config loader. Parse/resolve logic lives in
// tutti_config_parse.cpp (pure host).

#include "tutti/config/tutti_config.h"

#include <cctype>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tutti/cuda_like.h>
#include <tutti/io_types.h>
#include <tutti/storage_runtime.h>
#include "tutti/resource/nvme/nvme_resource.h"
#include "tutti/tutti_runtime/tutti_runtime_internal.h"

namespace tutti::config {
namespace {

namespace nvme_resource = tutti::resources::nvme;
struct StorageIdTables {
    std::unordered_map<std::string, const ResourceSpec*> resources;
    std::unordered_map<std::string, const ResolverSpec*> resolvers;
    std::unordered_map<std::string, const DataPathSpec*> datapaths;
    std::unordered_map<std::string, const BackendSpec*> backends;
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
        const auto consumer = resource_datapaths.emplace(
            backend.resource, backend.datapath);
        if (!consumer.second && consumer.first->second != backend.datapath) {
            return error(
                StatusCode::INVALID_ARGUMENT,
                "Resource cannot be consumed by independent DataPaths");
        }
        if (resource->second->type != contract->resource_type ||
            resolver->second->type != contract->resolver_type ||
            resolver->second->scheme != contract->resolver_scheme ||
            datapath->second->type != contract->datapath_type) {
            return error(StatusCode::INVALID_ARGUMENT,
                         "backend storage types do not match its contract");
        }
        status = validate_requested_cardinality(
            *resource->second, backend, *contract);
        if (!status.ok()) return status;
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

Result<std::unique_ptr<StorageRuntime>> invoke_runtime_factory(
    const LoadTuttiConfigOptions& options,
    RuntimeConfig config,
    RuntimeComponents components) {
    try {
        auto created = options.runtime_factory
            ? options.runtime_factory(std::move(config),
                                      std::move(components))
            : default_runtime_factory(std::move(config),
                                      std::move(components));
        if (created.ok() && !created.value()) {
            return Result<std::unique_ptr<StorageRuntime>>::Failure(
                error(StatusCode::INVALID_ARGUMENT,
                      "Runtime factory returned null"));
        }
        return created;
    } catch (const std::exception& exception) {
        return Result<std::unique_ptr<StorageRuntime>>::Failure(
            error(StatusCode::INTERNAL,
                  std::string("Runtime factory threw: ") +
                      exception.what()));
    } catch (...) {
        return Result<std::unique_ptr<StorageRuntime>>::Failure(
            error(StatusCode::INTERNAL, "Runtime factory threw"));
    }
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
        auto created = invoke_runtime_factory(
            options, runtime_config, RuntimeComponents{});
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

    const BackendSpec& backend = parsed.canonical_storage.backends.front();
    const StorageContract* contract = find_storage_contract(backend.contract);
    const BackendFactoryRegistration* registered_factory =
        find_backend_factory(backend.contract);
    if (contract == nullptr || registered_factory == nullptr ||
        registered_factory->contract != contract) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::UNSUPPORTED,
                  "backend contract factory is not implemented"));
    }

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

    const Resource* resource =
        TuttiRuntimeAssemblyAccess::resource(*tr, backend.resource);
    if (resource == nullptr) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::INTERNAL,
                  "backend Resource is missing from Runtime registry"));
    }

    const auto resolver_spec = id_tables.resolvers.find(backend.resolver);
    const auto datapath_spec = id_tables.datapaths.find(backend.datapath);
    const auto resource_spec = id_tables.resources.find(backend.resource);
    if (resolver_spec == id_tables.resolvers.end() ||
        datapath_spec == id_tables.datapaths.end() ||
        resource_spec == id_tables.resources.end()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::INTERNAL,
                  "backend assembly IDs are missing from lookup tables"));
    }
    const auto eff = resolve_cache_config(parsed, options.overrides);
    const BackendFactoryContext factory_context{
        backend,
        *resolver_spec->second,
        *datapath_spec->second,
        *resource_spec->second,
        *resource,
        parsed.runtime_accel_id,
        BackendFactoryCacheConfig{
            eff.handle_cache_capacity,
            eff.prp_cache_capacity,
            eff.handle_cache_l2_capacity,
        },
    };

    Result<BackendFactoryProduct> factory_result =
        Result<BackendFactoryProduct>::Failure(
            error(StatusCode::INTERNAL,
                  "backend factory did not return a result"));
    try {
        factory_result = options.backend_factory
            ? options.backend_factory(factory_context)
            : create_backend_from_registry(factory_context);
    } catch (const std::exception& e) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::INVALID_ARGUMENT,
                  std::string("backend factory threw: ") + e.what()));
    } catch (...) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::INTERNAL, "backend factory threw"));
    }
    if (!factory_result.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            factory_result.status());
    }
    BackendFactoryProduct product = std::move(factory_result).value();
    status = validate_backend_factory_product(
        factory_context, *contract, product);
    if (!status.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(status);
    }

    RuntimeComponents components;
    StorageTargetResolver* resolver = nullptr;
    DataPath* datapath = nullptr;
    try {
        resolver = tr->register_resolver(
            std::move(product.resolver), product.scheme);
        if (resolver == nullptr) {
            return Result<std::unique_ptr<TuttiRuntime>>::Failure(
                error(StatusCode::INVALID_ARGUMENT,
                      "TuttiRuntime rejected resolver registration"));
        }
        datapath = tr->register_datapath(
            std::move(product.datapath), product.data_path_key);
        if (datapath == nullptr) {
            return Result<std::unique_ptr<TuttiRuntime>>::Failure(
                error(StatusCode::INVALID_ARGUMENT,
                      "TuttiRuntime rejected DataPath registration"));
        }
        components.resolvers.push_back({product.scheme, resolver});
        components.data_paths.push_back({
            product.data_path_key, datapath, product.data_path_config});
    } catch (const std::exception& exception) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::INTERNAL,
                  std::string("backend component registration threw: ") +
                      exception.what()));
    } catch (...) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            error(StatusCode::INTERNAL,
                  "backend component registration threw"));
    }

    status = TuttiRuntimeAssemblyAccess::register_backend(
        *tr,
        BackendManifest{backend.id, backend.contract, backend.resolver,
                        backend.datapath, backend.resource},
        resource, resolver, datapath);
    if (!status.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(status);
    }

    auto created = invoke_runtime_factory(
        options, runtime_config, std::move(components));
    if (!created.ok()) {
        return Result<std::unique_ptr<TuttiRuntime>>::Failure(
            created.status());
    }
    tr->runtime = std::move(created).value();
    return Result<std::unique_ptr<TuttiRuntime>>::Success(std::move(tr));
}

} // namespace tutti::config

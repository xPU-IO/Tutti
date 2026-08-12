#include <tutti/tutti_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include <tutti/config/tutti_runtime_config_parser.h>
#include <tutti/cuda_like.h>
#include <tutti/storage_runtime.h>

#include "tutti/resource/nvme/nvme_resource.h"
#include "tutti/tutti_runtime/backend_factory.h"
#include "tutti/tutti_runtime/tutti_runtime_internal.h"

namespace tutti {
namespace {

namespace nvme_resource = tutti::resources::nvme;

Status create_error(StatusCode code, std::string message) {
    return Status(code, std::move(message));
}

template <typename T>
Result<T> failure(Status status) {
    return Result<T>::Failure(std::move(status));
}

std::string upper(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(
            std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

Result<int> default_backend_device_count() {
#if defined(TUTTI_USE_HOST)
    return Result<int>::Success(0);
#elif defined(TUTTI_USE_CUDA) || defined(TUTTI_USE_MUSA) || defined(TUTTI_USE_MACA)
    int count = 0;
    const auto result = cudaGetDeviceCount(&count);
    if (result != cudaSuccess) {
        return failure<int>(create_error(
            StatusCode::NOT_FOUND,
            "compiled accelerator backend has no available devices"));
    }
    return Result<int>::Success(count);
#else
    return failure<int>(create_error(
        StatusCode::UNSUPPORTED,
        "no compiled accelerator backend is available"));
#endif
}

Status validate_runtime_environment(
    const config::TuttiRuntimeSpec& spec,
    const tutti_runtime::TuttiRuntimeCreateInternalOptions& options) {
    if (upper(spec.accelerator.profile) !=
        upper(TUTTI_COMPILED_ACCELERATOR_PROFILE)) {
        return create_error(
            StatusCode::INVALID_ARGUMENT,
            "accelerator.profile does not match compiled profile");
    }
    if (spec.runtime.accel_id == -1) return Status::Ok();
    auto count = options.backend_device_count
        ? options.backend_device_count() : default_backend_device_count();
    if (!count.ok()) return count.status();
    if (spec.runtime.accel_id >= count.value()) {
        return create_error(
            StatusCode::NOT_FOUND,
            "runtime.accel_id is outside compiled backend device count");
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

Result<std::unique_ptr<Resource>> default_resource_factory(
    const config::ResourceSpec& resource_spec, std::int32_t accel_id) {
    if (resource_spec.type != "nvme") {
        return failure<std::unique_ptr<Resource>>(create_error(
            StatusCode::UNSUPPORTED,
            "configured Resource type is not implemented"));
    }
    const auto* nvme =
        std::get_if<config::NvmeResourceConfig>(&resource_spec.config);
    if (nvme == nullptr) {
        return failure<std::unique_ptr<Resource>>(create_error(
            StatusCode::INVALID_ARGUMENT,
            "NVMe Resource config has the wrong typed value"));
    }
    auto created = nvme_resource::make_nvme_resource(
        nvme_resource::NvmeResourceSpec{
            resource_spec.id, accel_id, nvme->provider, nvme->allocation});
    if (!created.ok()) {
        return failure<std::unique_ptr<Resource>>(created.status());
    }
    std::unique_ptr<Resource> resource = std::move(created).value();
    return Result<std::unique_ptr<Resource>>::Success(std::move(resource));
}

Result<std::unique_ptr<Resource>> create_and_initialize_resource(
    const config::ResourceSpec& spec, std::int32_t accel_id,
    const tutti_runtime::TuttiRuntimeCreateInternalOptions& options) {
    Result<std::unique_ptr<Resource>> created =
        failure<std::unique_ptr<Resource>>(create_error(
            StatusCode::INTERNAL, "Resource factory did not return a result"));
    try {
        created = options.resource_factory
            ? options.resource_factory(spec, accel_id)
            : default_resource_factory(spec, accel_id);
    } catch (const std::exception& exception) {
        return failure<std::unique_ptr<Resource>>(create_error(
            StatusCode::INTERNAL,
            std::string("Resource factory threw: ") + exception.what()));
    } catch (...) {
        return failure<std::unique_ptr<Resource>>(create_error(
            StatusCode::INTERNAL, "Resource factory threw"));
    }
    if (!created.ok()) return created;

    std::unique_ptr<Resource> resource = std::move(created).value();
    if (!resource) {
        return failure<std::unique_ptr<Resource>>(create_error(
            StatusCode::INVALID_ARGUMENT,
            "Resource factory returned null"));
    }
    ResourceInfo info = resource->info();
    if (info.id != spec.id || info.type != spec.type ||
        resource->capabilities().resource_type != spec.type ||
        info.state != ResourceState::CREATED) {
        discard_resource(resource);
        return failure<std::unique_ptr<Resource>>(create_error(
            StatusCode::INVALID_ARGUMENT,
            "Resource factory result does not match ResourceSpec"));
    }
    Status status;
    try {
        status = resource->initialize();
    } catch (const std::exception& exception) {
        discard_resource(resource);
        return failure<std::unique_ptr<Resource>>(create_error(
            StatusCode::INTERNAL,
            std::string("Resource initialize threw: ") + exception.what()));
    } catch (...) {
        discard_resource(resource);
        return failure<std::unique_ptr<Resource>>(create_error(
            StatusCode::INTERNAL, "Resource initialize threw"));
    }
    if (!status.ok()) {
        discard_resource(resource);
        return failure<std::unique_ptr<Resource>>(std::move(status));
    }
    info = resource->info();
    if (info.id != spec.id || info.type != spec.type ||
        info.state != ResourceState::INITIALIZED) {
        discard_resource(resource);
        return failure<std::unique_ptr<Resource>>(create_error(
            StatusCode::INVALID_ARGUMENT,
            "Resource initialize did not produce the expected instance"));
    }
    return Result<std::unique_ptr<Resource>>::Success(std::move(resource));
}

Result<std::unique_ptr<StorageRuntime>> create_storage_runtime(
    const tutti_runtime::TuttiRuntimeCreateInternalOptions& options,
    RuntimeConfig config, RuntimeComponents components) {
    try {
        auto created = options.runtime_factory
            ? options.runtime_factory(std::move(config), std::move(components))
            : StorageRuntime::create(std::move(config), std::move(components));
        if (created.ok() && !created.value()) {
            return failure<std::unique_ptr<StorageRuntime>>(create_error(
                StatusCode::INVALID_ARGUMENT,
                "StorageRuntime factory returned null"));
        }
        return created;
    } catch (const std::exception& exception) {
        return failure<std::unique_ptr<StorageRuntime>>(create_error(
            StatusCode::INTERNAL,
            std::string("StorageRuntime factory threw: ") + exception.what()));
    } catch (...) {
        return failure<std::unique_ptr<StorageRuntime>>(create_error(
            StatusCode::INTERNAL, "StorageRuntime factory threw"));
    }
}

template <typename Spec>
const Spec* find_by_id(const std::vector<Spec>& specs,
                       const std::string& id) {
    const auto found = std::find_if(
        specs.begin(), specs.end(),
        [&](const Spec& candidate) { return candidate.id == id; });
    return found == specs.end() ? nullptr : &*found;
}

} // namespace

namespace tutti_runtime {

EffectiveCacheConfig resolve_cache_config(
    const config::TuttiRuntimeSpec& spec,
    const TuttiRuntimeCreateOptions& options) {
    const config::NvmeDataPathTuning* tuning = nullptr;
    if (!spec.storage.backends.empty()) {
        const config::DataPathSpec* datapath = find_by_id(
            spec.storage.datapaths, spec.storage.backends.front().datapath);
        if (datapath != nullptr) {
            if (const auto* local =
                    std::get_if<config::LocalNvmeDataPathConfig>(
                        &datapath->config)) {
                tuning = local;
            } else if (const auto* striped =
                           std::get_if<config::StripedLocalNvmeDataPathConfig>(
                               &datapath->config)) {
                tuning = striped;
            }
        }
    }
    const auto env_or_zero = [](const char* name) {
        const char* value = std::getenv(name);
        return value == nullptr
            ? 0U : static_cast<std::uint32_t>(std::atoi(value));
    };
    const std::uint32_t config_handle =
        tuning == nullptr ? 0 : tuning->handle_cache_capacity;
    const std::uint32_t config_prp =
        tuning == nullptr ? 0 : tuning->prp_cache_capacity;
    const std::uint32_t config_l2 =
        tuning == nullptr ? 0 : tuning->handle_cache_l2_capacity;

    EffectiveCacheConfig effective;
    effective.handle_cache_capacity = options.handle_cache_capacity > 0
        ? options.handle_cache_capacity
        : (config_handle > 0 ? config_handle
                             : env_or_zero("TUTTI_HANDLE_CACHE_CAP"));
    effective.prp_cache_capacity = options.prp_cache_capacity > 0
        ? options.prp_cache_capacity
        : (config_prp > 0 ? config_prp
                          : env_or_zero("TUTTI_PRP_CACHE_CAP"));
    effective.handle_cache_l2_capacity =
        options.handle_cache_l2_capacity > 0
            ? options.handle_cache_l2_capacity : config_l2;
    return effective;
}

} // namespace tutti_runtime

Result<std::unique_ptr<TuttiRuntime>> TuttiRuntime::create(
    const std::string& config_path, TuttiRuntimeCreateOptions options) {
    auto parsed = config::parse_tutti_runtime_config(config_path);
    if (!parsed.ok()) {
        return failure<std::unique_ptr<TuttiRuntime>>(parsed.status());
    }
    return create(std::move(parsed).value(), std::move(options));
}

Result<std::unique_ptr<TuttiRuntime>> TuttiRuntime::create(
    config::TuttiRuntimeSpec spec, TuttiRuntimeCreateOptions options) {
    tutti_runtime::TuttiRuntimeCreateInternalOptions internal;
    internal.public_options = std::move(options);
    return create_with_options_(std::move(spec), std::move(internal));
}

Result<std::unique_ptr<TuttiRuntime>> TuttiRuntime::create_with_options_(
    config::TuttiRuntimeSpec spec,
    tutti_runtime::TuttiRuntimeCreateInternalOptions options) {
    Status status = spec.validate();
    if (!status.ok()) {
        return failure<std::unique_ptr<TuttiRuntime>>(std::move(status));
    }
    auto debug = spec.to_debug_string();
    if (!debug.ok()) {
        return failure<std::unique_ptr<TuttiRuntime>>(debug.status());
    }

    auto runtime = std::unique_ptr<TuttiRuntime>(new TuttiRuntime());
    runtime->validated_spec_debug_ = std::move(debug).value();
    if (options.public_options.spec_debug_logger) {
        try {
            options.public_options.spec_debug_logger(
                runtime->validated_spec_debug_);
        } catch (const std::exception& exception) {
            return failure<std::unique_ptr<TuttiRuntime>>(create_error(
                StatusCode::INTERNAL,
                std::string("spec debug logger threw: ") + exception.what()));
        } catch (...) {
            return failure<std::unique_ptr<TuttiRuntime>>(create_error(
                StatusCode::INTERNAL, "spec debug logger threw"));
        }
    }

    status = validate_runtime_environment(spec, options);
    if (!status.ok()) {
        return failure<std::unique_ptr<TuttiRuntime>>(std::move(status));
    }
    const auto cache = tutti_runtime::resolve_cache_config(
        spec, options.public_options);

    runtime->runtime_shutdown_hook_ = std::move(options.runtime_shutdown_hook);
    runtime->shutdown_observer_ = std::move(options.shutdown_observer);

    for (const config::ResourceSpec& resource_spec : spec.storage.resources) {
        auto resource = create_and_initialize_resource(
            resource_spec, spec.runtime.accel_id, options);
        if (!resource.ok()) {
            return failure<std::unique_ptr<TuttiRuntime>>(resource.status());
        }
        try {
            status = runtime->adopt_resource_(
                resource_spec.id, std::move(resource).value());
        } catch (const std::exception& exception) {
            return failure<std::unique_ptr<TuttiRuntime>>(create_error(
                StatusCode::INTERNAL,
                std::string("Resource registry insertion threw: ") +
                    exception.what()));
        } catch (...) {
            return failure<std::unique_ptr<TuttiRuntime>>(create_error(
                StatusCode::INTERNAL,
                "Resource registry insertion threw"));
        }
        if (!status.ok()) {
            return failure<std::unique_ptr<TuttiRuntime>>(std::move(status));
        }
    }

    const config::BackendSpec& backend = spec.storage.backends.front();
    const config::ResourceSpec* resource_spec = find_by_id(
        spec.storage.resources, backend.resource);
    const config::ResolverSpec* resolver_spec = find_by_id(
        spec.storage.resolvers, backend.resolver);
    const config::DataPathSpec* datapath_spec = find_by_id(
        spec.storage.datapaths, backend.datapath);
    const Resource* resource = runtime->find_resource_(backend.resource);
    if (resource_spec == nullptr || resolver_spec == nullptr ||
        datapath_spec == nullptr || resource == nullptr) {
        return failure<std::unique_ptr<TuttiRuntime>>(create_error(
            StatusCode::INTERNAL,
            "validated backend IDs are missing during Runtime assembly"));
    }

    const tutti_runtime::RuntimeBackendRegistration* registration =
        tutti_runtime::find_backend_factory(backend.contract);
    if (registration == nullptr) {
        return failure<std::unique_ptr<TuttiRuntime>>(create_error(
            StatusCode::UNSUPPORTED,
            "backend contract has no Runtime registration"));
    }

    const tutti_runtime::BackendFactoryContext context{
        backend, *resolver_spec, *datapath_spec, *resource_spec, *resource,
        spec.runtime.accel_id,
        {cache.handle_cache_capacity, cache.prp_cache_capacity,
         cache.handle_cache_l2_capacity},
    };
    Result<tutti_runtime::BackendFactoryProduct> factory_result =
        failure<tutti_runtime::BackendFactoryProduct>(create_error(
            StatusCode::INTERNAL,
            "backend factory did not return a result"));
    try {
        factory_result = options.backend_factory
            ? options.backend_factory(context)
            : tutti_runtime::create_backend_from_registry(context);
    } catch (const std::exception& exception) {
        return failure<std::unique_ptr<TuttiRuntime>>(create_error(
            StatusCode::INVALID_ARGUMENT,
            std::string("backend factory threw: ") + exception.what()));
    } catch (...) {
        return failure<std::unique_ptr<TuttiRuntime>>(create_error(
            StatusCode::INTERNAL, "backend factory threw"));
    }
    if (!factory_result.ok()) {
        return failure<std::unique_ptr<TuttiRuntime>>(factory_result.status());
    }
    tutti_runtime::BackendFactoryProduct product =
        std::move(factory_result).value();
    status = tutti_runtime::validate_backend_factory_product(
        context, *registration, product);
    if (!status.ok()) {
        return failure<std::unique_ptr<TuttiRuntime>>(std::move(status));
    }

    RuntimeComponents components;
    StorageTargetResolver* resolver = nullptr;
    DataPath* datapath = nullptr;
    try {
        status = runtime->register_resolver_(
            backend.resolver, std::move(product.resolver), product.scheme,
            resolver);
        if (!status.ok()) {
            return failure<std::unique_ptr<TuttiRuntime>>(std::move(status));
        }
        status = runtime->register_datapath_(
            backend.datapath, std::move(product.datapath),
            product.data_path_key, datapath);
        if (!status.ok()) {
            return failure<std::unique_ptr<TuttiRuntime>>(std::move(status));
        }
        components.resolvers.push_back({product.scheme, resolver});
        components.data_paths.push_back(
            {product.data_path_key, datapath, product.data_path_config});
    } catch (const std::exception& exception) {
        return failure<std::unique_ptr<TuttiRuntime>>(create_error(
            StatusCode::INTERNAL,
            std::string("backend component registration threw: ") +
                exception.what()));
    } catch (...) {
        return failure<std::unique_ptr<TuttiRuntime>>(create_error(
            StatusCode::INTERNAL,
            "backend component registration threw"));
    }

    try {
        status = runtime->register_backend_(
            {backend.id, backend.contract, backend.resolver, backend.datapath,
             backend.resource},
            resource, resolver, datapath);
    } catch (const std::exception& exception) {
        return failure<std::unique_ptr<TuttiRuntime>>(create_error(
            StatusCode::INTERNAL,
            std::string("backend manifest registration threw: ") +
                exception.what()));
    } catch (...) {
        return failure<std::unique_ptr<TuttiRuntime>>(create_error(
            StatusCode::INTERNAL,
            "backend manifest registration threw"));
    }
    if (!status.ok()) {
        return failure<std::unique_ptr<TuttiRuntime>>(std::move(status));
    }

    RuntimeConfig runtime_config;
    runtime_config.accel_id = spec.runtime.accel_id;
    runtime_config.profile_name = spec.accelerator.profile;
    auto storage_runtime = create_storage_runtime(
        options, std::move(runtime_config), std::move(components));
    if (!storage_runtime.ok()) {
        return failure<std::unique_ptr<TuttiRuntime>>(
            storage_runtime.status());
    }
    status = runtime->set_storage_runtime_(
        std::move(storage_runtime).value());
    if (!status.ok()) {
        return failure<std::unique_ptr<TuttiRuntime>>(std::move(status));
    }
    runtime->state_ = TuttiRuntimeState::RUNNING;
    return Result<std::unique_ptr<TuttiRuntime>>::Success(std::move(runtime));
}

} // namespace tutti

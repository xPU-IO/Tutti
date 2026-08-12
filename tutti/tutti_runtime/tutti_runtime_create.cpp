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
#include <tutti/data_paths/data_path_factory.h>
#include <tutti/resolvers/resolver_factory.h>
#include <tutti/storage_runtime.h>

#include <tutti/resource.h>
#include "tutti/tutti_runtime/tutti_runtime_internal.h"

namespace tutti {
namespace {

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

Result<int> default_accelerator_device_count() {
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
    auto count = options.accelerator_device_count
        ? options.accelerator_device_count()
        : default_accelerator_device_count();
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

Result<std::unique_ptr<Resource>> create_and_initialize_resource(
    const config::ResourceSpec& spec, std::int32_t accel_id,
    const tutti_runtime::TuttiRuntimeCreateInternalOptions& options) {
    Result<std::unique_ptr<Resource>> created =
        failure<std::unique_ptr<Resource>>(create_error(
            StatusCode::INTERNAL, "Resource factory did not return a result"));
    try {
        const resources::ResourceCreateContext context{accel_id};
        created = options.resource_factory
            ? options.resource_factory(spec, context)
            : resources::create_resource(spec, context);
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

Result<std::unique_ptr<StorageTargetResolver>> create_resolver_component(
    const config::ResolverSpec& spec,
    const resolvers::ResolverCreateContext& context,
    const tutti_runtime::TuttiRuntimeCreateInternalOptions& options) {
    Result<std::unique_ptr<StorageTargetResolver>> created =
        failure<std::unique_ptr<StorageTargetResolver>>(create_error(
            StatusCode::INTERNAL, "resolver factory did not return a result"));
    try {
        created = options.resolver_factory
            ? options.resolver_factory(spec, context)
            : resolvers::create_resolver(spec, context);
    } catch (const std::exception& exception) {
        return failure<std::unique_ptr<StorageTargetResolver>>(create_error(
            StatusCode::INTERNAL,
            std::string("resolver factory threw: ") + exception.what()));
    } catch (...) {
        return failure<std::unique_ptr<StorageTargetResolver>>(create_error(
            StatusCode::INTERNAL, "resolver factory threw"));
    }
    if (!created.ok()) return created;
    if (!created.value()) {
        return failure<std::unique_ptr<StorageTargetResolver>>(create_error(
            StatusCode::INVALID_ARGUMENT, "resolver factory returned null"));
    }
    return created;
}

Result<data_paths::CreatedDataPath> create_data_path_component(
    const config::DataPathSpec& spec,
    const data_paths::DataPathCreateContext& context,
    const tutti_runtime::TuttiRuntimeCreateInternalOptions& options) {
    Result<data_paths::CreatedDataPath> created =
        failure<data_paths::CreatedDataPath>(create_error(
            StatusCode::INTERNAL, "DataPath factory did not return a result"));
    try {
        created = options.data_path_factory
            ? options.data_path_factory(spec, context)
            : data_paths::create_data_path(spec, context);
    } catch (const std::exception& exception) {
        return failure<data_paths::CreatedDataPath>(create_error(
            StatusCode::INTERNAL,
            std::string("DataPath factory threw: ") + exception.what()));
    } catch (...) {
        return failure<data_paths::CreatedDataPath>(create_error(
            StatusCode::INTERNAL, "DataPath factory threw"));
    }
    if (!created.ok()) return created;
    if (!created.value().instance) {
        return failure<data_paths::CreatedDataPath>(create_error(
            StatusCode::INVALID_ARGUMENT, "DataPath factory returned null"));
    }
    return created;
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
    const config::DataPathSpec& spec,
    const TuttiRuntimeCreateOptions& options) {
    const config::NvmeDataPathTuning* tuning = nullptr;
    if (const auto* local = std::get_if<config::LocalNvmeDataPathConfig>(
            &spec.config)) {
        tuning = local;
    } else if (const auto* striped =
                   std::get_if<config::StripedLocalNvmeDataPathConfig>(
                       &spec.config)) {
        tuning = striped;
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

    if (options.public_options.spec_debug_logger) {
        try {
            options.public_options.spec_debug_logger(
                debug.value());
        } catch (const std::exception& exception) {
            return failure<std::unique_ptr<TuttiRuntime>>(create_error(
                StatusCode::INTERNAL,
                std::string("spec debug logger threw: ") + exception.what()));
        } catch (...) {
            return failure<std::unique_ptr<TuttiRuntime>>(create_error(
                StatusCode::INTERNAL, "spec debug logger threw"));
        }
    }

    auto runtime = std::unique_ptr<TuttiRuntime>(new TuttiRuntime());
    status = validate_runtime_environment(spec, options);
    if (!status.ok()) {
        return failure<std::unique_ptr<TuttiRuntime>>(std::move(status));
    }
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

    RuntimeComponents components;
    components.resolvers.reserve(spec.storage.resolvers.size());
    components.data_paths.reserve(spec.storage.datapaths.size());
    for (const config::BackendSpec& relation : spec.storage.backends) {
        const config::ResolverSpec* resolver_spec = find_by_id(
            spec.storage.resolvers, relation.resolver);
        const config::DataPathSpec* data_path_spec = find_by_id(
            spec.storage.datapaths, relation.datapath);
        const Resource* resource = runtime->find_resource_(relation.resource);
        if (resolver_spec == nullptr || data_path_spec == nullptr ||
            resource == nullptr) {
            return failure<std::unique_ptr<TuttiRuntime>>(create_error(
                StatusCode::INTERNAL,
                "validated component IDs are missing during Runtime assembly"));
        }

        auto resolver_result = create_resolver_component(
            *resolver_spec,
            resolvers::ResolverCreateContext{
                *resource, relation, data_path_spec->id},
            options);
        if (!resolver_result.ok()) {
            return failure<std::unique_ptr<TuttiRuntime>>(
                resolver_result.status());
        }

        const auto cache = tutti_runtime::resolve_cache_config(
            *data_path_spec, options.public_options);
        auto data_path_result = create_data_path_component(
            *data_path_spec,
            data_paths::DataPathCreateContext{
                *resource, relation, spec.runtime.accel_id,
                {cache.handle_cache_capacity, cache.prp_cache_capacity,
                 cache.handle_cache_l2_capacity}},
            options);
        if (!data_path_result.ok()) {
            return failure<std::unique_ptr<TuttiRuntime>>(
                data_path_result.status());
        }
        data_paths::CreatedDataPath created_data_path =
            std::move(data_path_result).value();

        StorageTargetResolver* resolver = nullptr;
        DataPath* data_path = nullptr;
        try {
            status = runtime->register_resolver_(
                resolver_spec->id, std::move(resolver_result).value(), resolver);
            if (!status.ok()) {
                return failure<std::unique_ptr<TuttiRuntime>>(
                    std::move(status));
            }
            status = runtime->register_datapath_(
                data_path_spec->id, std::move(created_data_path.instance),
                data_path);
            if (!status.ok()) {
                return failure<std::unique_ptr<TuttiRuntime>>(
                    std::move(status));
            }
            components.resolvers.push_back(
                {resolver_spec->scheme, resolver});
            components.data_paths.push_back(
                {data_path_spec->id, data_path,
                 std::move(created_data_path.initialize_config)});
        } catch (const std::exception& exception) {
            return failure<std::unique_ptr<TuttiRuntime>>(create_error(
                StatusCode::INTERNAL,
                std::string("component registration threw: ") +
                    exception.what()));
        } catch (...) {
            return failure<std::unique_ptr<TuttiRuntime>>(create_error(
                StatusCode::INTERNAL, "component registration threw"));
        }
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

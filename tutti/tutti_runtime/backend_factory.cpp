#include "tutti/tutti_runtime/backend_factory.h"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tutti/bindings/ext4_local_nvme/binding.h>
#include <tutti/bindings/striped_local_nvme/binding.h>
#include "tutti/bindings/memfs/binding.h"
#if defined(TUTTI_RUNTIME_HAS_LOCAL_NVME)
#include <tutti/resolvers/local_file/resolver.h>
#include <tutti/resolvers/striped_file/resolver.h>
#include "tutti/data_paths/local_nvme/local_nvme_data_path.h"
#include "tutti/data_paths/striped_local_nvme/striped_data_path.h"
#endif
#include "tutti/resource/nvme/nvme_resource.h"

namespace tutti::tutti_runtime {
namespace {

namespace nvme_resource = tutti::resources::nvme;
#if defined(TUTTI_RUNTIME_HAS_LOCAL_NVME)
namespace local_datapath = tutti::data_paths::local_nvme;
namespace local_resolver = tutti::resolvers::local_file;
namespace striped_datapath = tutti::data_paths::striped_local_nvme;
namespace striped_resolver = tutti::resolvers::striped_file;
#endif
namespace ext4_binding = tutti::binding::ext4_local_nvme;
namespace memfs_binding = tutti::binding::memfs;
namespace striped_binding = tutti::binding::striped_local_nvme;

Status invalid(std::string message) {
    return Status(StatusCode::INVALID_ARGUMENT, std::move(message));
}

template <typename T>
Result<T> failure(Status status) {
    return Result<T>::Failure(std::move(status));
}

const config::NvmeResourceConfig& nvme_config(
    const config::ResourceSpec& resource) {
    return std::get<config::NvmeResourceConfig>(resource.config);
}

#if defined(TUTTI_RUNTIME_HAS_LOCAL_NVME)
Result<std::uint32_t> checked_u32(std::uint64_t value,
                                  const char* field) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        return failure<std::uint32_t>(
            invalid(std::string(field) + " exceeds uint32_t"));
    }
    return Result<std::uint32_t>::Success(
        static_cast<std::uint32_t>(value));
}

const config::NvmeDataPathTuning& datapath_tuning(
    const config::DataPathSpec& datapath) {
    if (const auto* local =
            std::get_if<config::LocalNvmeDataPathConfig>(&datapath.config)) {
        return *local;
    }
    return std::get<config::StripedLocalNvmeDataPathConfig>(datapath.config);
}
#endif

std::uint64_t stripe_unit(const config::BackendSpec& backend) {
    const auto* striped =
        std::get_if<config::StripedLocalNvmeBackendConfig>(&backend.config);
    return striped == nullptr ? 0 : striped->stripe_unit;
}

std::size_t requested_cardinality(const config::ResourceSpec& resource) {
    if (resource.type != "nvme") return 1;
    const auto& allocation = nvme_config(resource).allocation;
    if (resource.type == "nvme" &&
        allocation.selection == config::NvmeSelection::Striped) {
        return allocation.device_ids.size();
    }
    return 1;
}

Status validate_factory_context(const BackendFactoryContext& context,
                                const RuntimeBackendRegistration& registration) {
    if (context.backend.contract != registration.contract) {
        return invalid("backend factory context does not match contract registry");
    }
    const ResourceInfo info = context.resource.info();
    const ResourceCapabilities& capabilities =
        context.resource.capabilities();
    if (info.id != context.resource_spec.id ||
        info.type != context.resource_spec.type ||
        info.state != ResourceState::INITIALIZED ||
        capabilities.resource_type != context.resource_spec.type ||
        !capabilities.provides_resolver_view ||
        !capabilities.provides_datapath_view) {
        return invalid("backend Resource instance is incompatible with contract");
    }
    const std::size_t cardinality = requested_cardinality(
        context.resource_spec);
    if (cardinality < registration.minimum_cardinality ||
        cardinality > registration.maximum_cardinality) {
        return invalid("backend Resource request cardinality does not match contract");
    }
    return Status::Ok();
}

BackendShardProjection resolver_projection(
    const nvme_resource::NvmeResolverSliceView& slice) {
    return BackendShardProjection{
        slice.device_id,
        slice.pci_bdf,
        slice.namespace_id,
        slice.logical_block_size,
        0,
    };
}

BackendShardProjection datapath_projection(
    const nvme_resource::NvmeDataPathSliceView& slice) {
    return BackendShardProjection{
        slice.device_id,
        slice.pci_bdf,
        slice.namespace_id,
        slice.logical_block_size,
        slice.max_data_size,
    };
}

Status validate_nvme_views(
    const BackendFactoryContext& context,
    const RuntimeBackendRegistration& registration,
    const nvme_resource::NvmeResolverResourceView& resolver_view,
    const nvme_resource::NvmeDataPathResourceView& datapath_view) {
    if (resolver_view.slices.size() != datapath_view.slices.size()) {
        return invalid("NVMe resource views have inconsistent cardinality");
    }
    const std::size_t cardinality = resolver_view.slices.size();
    if (cardinality < registration.minimum_cardinality ||
        cardinality > registration.maximum_cardinality) {
        return invalid("backend Resource view cardinality does not match contract");
    }
    const auto& allocation = nvme_config(context.resource_spec).allocation;
    if (allocation.selection != config::NvmeSelection::Allowed &&
        cardinality != allocation.device_ids.size()) {
        return invalid("backend Resource view cardinality does not match request");
    }
    for (std::size_t index = 0; index < cardinality; ++index) {
        const auto& resolver_slice = resolver_view.slices[index];
        const auto& datapath_slice = datapath_view.slices[index];
        if (resolver_slice.device_id != datapath_slice.device_id ||
            resolver_slice.pci_bdf != datapath_slice.pci_bdf ||
            resolver_slice.namespace_id != datapath_slice.namespace_id ||
            resolver_slice.logical_block_size !=
                datapath_slice.logical_block_size) {
            return invalid("resolver and DataPath resource views disagree at shard " +
                           std::to_string(index));
        }
        if (datapath_slice.accel_id != context.runtime_accel_id) {
            return invalid("DataPath resource view accelerator does not match Runtime");
        }
        if (allocation.selection != config::NvmeSelection::Allowed &&
            resolver_slice.device_id !=
                allocation.device_ids[index]) {
            return invalid("backend Resource view slice order does not match request");
        }
    }
    return Status::Ok();
}

Result<std::pair<nvme_resource::NvmeResolverResourceView,
                 nvme_resource::NvmeDataPathResourceView>>
nvme_views(const BackendFactoryContext& context,
           const RuntimeBackendRegistration& registration) {
    const auto* resource = dynamic_cast<const nvme_resource::NvmeResource*>(
        &context.resource);
    if (resource == nullptr) {
        return failure<std::pair<nvme_resource::NvmeResolverResourceView,
                                 nvme_resource::NvmeDataPathResourceView>>(
            invalid("backend factory requires NvmeResource typed views"));
    }
    auto resolver_view = resource->resolver_view();
    if (!resolver_view.ok()) {
        return failure<std::pair<nvme_resource::NvmeResolverResourceView,
                                 nvme_resource::NvmeDataPathResourceView>>(
            resolver_view.status());
    }
    auto datapath_view = resource->datapath_view();
    if (!datapath_view.ok()) {
        return failure<std::pair<nvme_resource::NvmeResolverResourceView,
                                 nvme_resource::NvmeDataPathResourceView>>(
            datapath_view.status());
    }
    Status status = validate_nvme_views(
        context, registration, resolver_view.value(), datapath_view.value());
    if (!status.ok()) {
        return failure<std::pair<nvme_resource::NvmeResolverResourceView,
                                 nvme_resource::NvmeDataPathResourceView>>(
            std::move(status));
    }
    return Result<std::pair<nvme_resource::NvmeResolverResourceView,
                            nvme_resource::NvmeDataPathResourceView>>::Success(
        {std::move(resolver_view).value(),
         std::move(datapath_view).value()});
}

#if defined(TUTTI_RUNTIME_HAS_LOCAL_NVME)
void set_contract_product_fields(BackendFactoryProduct& product,
                                 const RuntimeBackendRegistration& registration,
                                 const BackendFactoryContext& context) {
    product.scheme = context.resolver.scheme;
    product.data_path_key = std::string(registration.data_path_key);
    product.resolver_type_id = std::string(registration.resolver_type_id);
    product.payload_type_id = std::string(registration.payload_type_id);
    product.payload_api_version = registration.payload_api_version;
}

#endif

#if defined(TUTTI_RUNTIME_HAS_LOCAL_NVME)
Result<BackendFactoryProduct> create_local_backend(
    const BackendFactoryContext& context) {
    const RuntimeBackendRegistration* registration = find_backend_factory(
        "ext4-local-nvme");
    if (registration == nullptr) {
        return failure<BackendFactoryProduct>(
            Status(StatusCode::INTERNAL,
                   "ext4-local-nvme contract is not registered"));
    }
    auto views = nvme_views(context, *registration);
    if (!views.ok()) return failure<BackendFactoryProduct>(views.status());
    auto typed_views = std::move(views).value();
    if (typed_views.first.slices.size() != 1) {
        return failure<BackendFactoryProduct>(
            invalid("local backend requires exactly one Resource slice"));
    }

    const auto& resolver_slice = typed_views.first.slices.front();
    const auto& datapath_slice = typed_views.second.slices.front();
    auto bar0_size = checked_u32(datapath_slice.bar0_size, "bar0_size");
    if (!bar0_size.ok()) {
        return failure<BackendFactoryProduct>(bar0_size.status());
    }
    const auto& tuning = datapath_tuning(context.datapath);
    auto max_batch_entries = checked_u32(
        tuning.max_batch_entries, "max_batch_entries");
    if (!max_batch_entries.ok()) {
        return failure<BackendFactoryProduct>(max_batch_entries.status());
    }

    BackendFactoryProduct product;
    set_contract_product_fields(product, *registration, context);
    product.data_path_config = DataPathConfig{"local_nvme"};
    product.resolver_shards.push_back(resolver_projection(resolver_slice));
    product.datapath_shards.push_back(datapath_projection(datapath_slice));
    product.effective_max_data_size = datapath_slice.max_data_size;
    product.resolver = std::make_unique<local_resolver::LocalFileResolver>(
        resolver_slice.pci_bdf,
        resolver_slice.namespace_id,
        resolver_slice.logical_block_size,
        local_resolver::BackingDeviceConfig{resolver_slice.block_path, 0});
    product.datapath = std::make_unique<local_datapath::LocalNvmeDataPath>(
        datapath_slice.chrdev_path,
        bar0_size.value(),
        static_cast<std::uint32_t>(datapath_slice.accel_id),
        datapath_slice.granted_queues,
        datapath_slice.namespace_id,
        datapath_slice.logical_block_size,
        datapath_slice.max_data_size,
        max_batch_entries.value(),
        0,
        context.cache.handle_cache_capacity,
        context.cache.prp_cache_capacity,
        tuning.max_in_flight_operations,
        tuning.max_batch_entries,
        0,
        context.cache.handle_cache_l2_capacity,
        datapath_slice.pci_bdf);
    return Result<BackendFactoryProduct>::Success(std::move(product));
}

Result<BackendFactoryProduct> create_striped_backend(
    const BackendFactoryContext& context) {
    const RuntimeBackendRegistration* registration = find_backend_factory(
        "striped-local-nvme");
    if (registration == nullptr) {
        return failure<BackendFactoryProduct>(
            Status(StatusCode::INTERNAL,
                   "striped-local-nvme contract is not registered"));
    }
    auto views = nvme_views(context, *registration);
    if (!views.ok()) return failure<BackendFactoryProduct>(views.status());
    auto typed_views = std::move(views).value();

    const auto& tuning = datapath_tuning(context.datapath);
    auto max_batch_entries = checked_u32(
        tuning.max_batch_entries, "max_batch_entries");
    if (!max_batch_entries.ok()) {
        return failure<BackendFactoryProduct>(max_batch_entries.status());
    }
    auto max_in_flight = checked_u32(
        tuning.max_in_flight_operations,
        "max_in_flight_operations");
    if (!max_in_flight.ok()) {
        return failure<BackendFactoryProduct>(max_in_flight.status());
    }

    std::vector<striped_datapath::DeviceDescriptor> descriptors;
    descriptors.reserve(typed_views.second.slices.size());
    std::vector<std::unique_ptr<StorageTargetResolver>> shard_resolvers;
    shard_resolvers.reserve(typed_views.first.slices.size());

    BackendFactoryProduct product;
    set_contract_product_fields(product, *registration, context);
    product.data_path_config = DataPathConfig{"striped-local-nvme"};
    product.stripe_unit = stripe_unit(context.backend);
    for (std::size_t index = 0;
         index < typed_views.second.slices.size(); ++index) {
        const auto& resolver_slice = typed_views.first.slices[index];
        const auto& datapath_slice = typed_views.second.slices[index];

        striped_datapath::DeviceDescriptor descriptor;
        descriptor.snvme_dev_path = datapath_slice.chrdev_path;
        descriptor.bar0_size = datapath_slice.bar0_size;
        descriptor.namespace_id = datapath_slice.namespace_id;
        descriptor.cuda_device = static_cast<std::uint32_t>(
            datapath_slice.accel_id);
        descriptor.num_user_queues = datapath_slice.granted_queues;
        descriptor.block_size = datapath_slice.logical_block_size;
        descriptor.controller_pci_addr = datapath_slice.pci_bdf;
        descriptors.push_back(std::move(descriptor));

        shard_resolvers.push_back(
            std::make_unique<local_resolver::LocalFileResolver>(
                resolver_slice.pci_bdf,
                resolver_slice.namespace_id,
                resolver_slice.logical_block_size,
                local_resolver::BackingDeviceConfig{
                    resolver_slice.block_path, 0}));
        product.resolver_shards.push_back(
            resolver_projection(resolver_slice));
        product.datapath_shards.push_back(
            datapath_projection(datapath_slice));
        product.effective_max_data_size =
            product.effective_max_data_size == 0
                ? datapath_slice.max_data_size
                : std::min(product.effective_max_data_size,
                           datapath_slice.max_data_size);
    }

    product.resolver = std::make_unique<striped_resolver::StripedResolver>(
        std::move(shard_resolvers), stripe_unit(context.backend));
    product.datapath = std::make_unique<striped_datapath::StripedDataPath>(
        std::move(descriptors),
        static_cast<std::uint32_t>(context.runtime_accel_id),
        product.effective_max_data_size,
        0,
        max_batch_entries.value(),
        max_in_flight.value(),
        context.cache.handle_cache_capacity,
        context.cache.prp_cache_capacity);
    return Result<BackendFactoryProduct>::Success(std::move(product));
}
#endif

const std::array<RuntimeBackendRegistration, 3>& backend_factories() {
#if defined(TUTTI_RUNTIME_HAS_LOCAL_NVME)
    constexpr RegisteredBackendFactory local_factory = create_local_backend;
    constexpr RegisteredBackendFactory striped_factory = create_striped_backend;
#else
    constexpr RegisteredBackendFactory local_factory = nullptr;
    constexpr RegisteredBackendFactory striped_factory = nullptr;
#endif
    static const std::array<RuntimeBackendRegistration, 3> factories{{
        {"ext4-local-nvme", ext4_binding::kResolverTypeId,
         ext4_binding::kPayloadTypeId, ext4_binding::kPayloadApiVersion,
         ext4_binding::kRecommendedDataPathKey, 1, 1, local_factory},
        {"striped-local-nvme", striped_binding::kResolverTypeId,
         striped_binding::kPayloadTypeId, striped_binding::kPayloadApiVersion,
         striped_binding::kRecommendedDataPathKey, 2,
         std::numeric_limits<std::size_t>::max(), striped_factory},
        {"memfs", memfs_binding::kResolverTypeId,
         memfs_binding::kPayloadTypeId, memfs_binding::kPayloadApiVersion,
         memfs_binding::kRecommendedDataPathKey, 1, 1, nullptr},
    }};
    return factories;
}

bool same_projection(const BackendShardProjection& resolver,
                     const BackendShardProjection& datapath) {
    return resolver.device_id == datapath.device_id &&
           resolver.controller_pci_addr == datapath.controller_pci_addr &&
           resolver.namespace_id == datapath.namespace_id &&
           resolver.logical_block_size == datapath.logical_block_size;
}

bool exact_projection(const BackendShardProjection& left,
                      const BackendShardProjection& right) {
    return same_projection(left, right) &&
           left.max_data_size == right.max_data_size;
}

} // namespace

const RuntimeBackendRegistration* find_backend_factory(
    std::string_view contract) {
    const auto& factories = backend_factories();
    const auto found = std::find_if(
        factories.begin(), factories.end(),
        [&](const RuntimeBackendRegistration& registration) {
            return registration.contract == contract;
        });
    return found == factories.end() ? nullptr : &*found;
}

Result<BackendFactoryProduct> create_backend_from_registry(
    const BackendFactoryContext& context) {
    const RuntimeBackendRegistration* registration =
        find_backend_factory(context.backend.contract);
    if (registration == nullptr || registration->create == nullptr) {
        return failure<BackendFactoryProduct>(
            Status(StatusCode::UNSUPPORTED,
                   "backend contract factory is not implemented"));
    }
    Status status = validate_factory_context(context, *registration);
    if (!status.ok()) return failure<BackendFactoryProduct>(std::move(status));

    try {
        return registration->create(context);
    } catch (const std::exception& exception) {
        return failure<BackendFactoryProduct>(
            Status(StatusCode::INVALID_ARGUMENT,
                   std::string("backend factory threw: ") +
                       exception.what()));
    } catch (...) {
        return failure<BackendFactoryProduct>(
            Status(StatusCode::INTERNAL, "backend factory threw"));
    }
}

Status validate_backend_factory_product(
    const BackendFactoryContext& context,
    const RuntimeBackendRegistration& registration,
    const BackendFactoryProduct& product) {
    Status status = validate_factory_context(context, registration);
    if (!status.ok()) return status;
    if (!product.resolver || !product.datapath) {
        return invalid("backend factory returned incomplete components");
    }
    if (product.scheme != context.resolver.scheme ||
        product.data_path_key != registration.data_path_key) {
        return invalid("backend factory returned incompatible Runtime bindings");
    }
    if (product.resolver_type_id != registration.resolver_type_id ||
        product.payload_type_id != registration.payload_type_id ||
        product.payload_api_version != registration.payload_api_version) {
        return Status(StatusCode::UNSUPPORTED,
                      "backend factory payload contract mismatch");
    }
    if (context.resource_spec.type == "memory") {
        if (!product.resolver_shards.empty() ||
            !product.datapath_shards.empty() ||
            product.effective_max_data_size != 0 || product.stripe_unit != 0) {
            return invalid("memory backend factory returned NVMe projections");
        }
        return Status::Ok();
    }
    if (product.resolver_shards.size() != product.datapath_shards.size()) {
        return invalid("backend factory shard view cardinality mismatch");
    }
    const std::size_t cardinality = product.resolver_shards.size();
    if (cardinality < registration.minimum_cardinality ||
        cardinality > registration.maximum_cardinality) {
        return invalid("backend factory product cardinality does not match contract");
    }
    const auto& allocation = nvme_config(context.resource_spec).allocation;
    if (allocation.selection != config::NvmeSelection::Allowed &&
        cardinality != allocation.device_ids.size()) {
        return invalid("backend factory product cardinality does not match request");
    }

    auto expected_views = nvme_views(context, registration);
    if (!expected_views.ok()) return expected_views.status();
    const auto& resolver_view = expected_views.value().first;
    const auto& datapath_view = expected_views.value().second;
    if (resolver_view.slices.size() != cardinality ||
        datapath_view.slices.size() != cardinality) {
        return invalid("backend factory product does not match Resource views");
    }

    std::uint64_t minimum_mdts = 0;
    for (std::size_t index = 0; index < cardinality; ++index) {
        if (!exact_projection(
                product.resolver_shards[index],
                resolver_projection(resolver_view.slices[index])) ||
            !exact_projection(
                product.datapath_shards[index],
                datapath_projection(datapath_view.slices[index]))) {
            return invalid("backend factory projection does not match Resource view");
        }
        if (!same_projection(product.resolver_shards[index],
                             product.datapath_shards[index])) {
            return invalid("backend factory namespace or shard order mismatch");
        }
        if (allocation.selection != config::NvmeSelection::Allowed &&
            product.resolver_shards[index].device_id !=
                allocation.device_ids[index]) {
            return invalid("backend factory shard order does not match request");
        }
        const std::uint64_t shard_mdts =
            product.datapath_shards[index].max_data_size;
        if (shard_mdts == 0) {
            return invalid("backend factory DataPath shard has zero MDTS");
        }
        minimum_mdts = minimum_mdts == 0
            ? shard_mdts : std::min(minimum_mdts, shard_mdts);
    }
    if (product.effective_max_data_size != minimum_mdts) {
        return invalid("backend factory effective MDTS is not the shard minimum");
    }
    if (registration.contract == "striped-local-nvme") {
        if (product.stripe_unit != stripe_unit(context.backend) ||
            product.stripe_unit == 0) {
            return invalid("striped backend factory lost backend stripe_unit");
        }
    } else if (product.stripe_unit != 0) {
        return invalid("non-striped backend factory returned stripe_unit");
    }
    const DataPathCapabilities& capabilities =
        product.datapath->capabilities();
    if (capabilities.bound_accel_id != context.runtime_accel_id) {
        return invalid("backend DataPath accelerator does not match Runtime");
    }
    return Status::Ok();
}

} // namespace tutti::tutti_runtime

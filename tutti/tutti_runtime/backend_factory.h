#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <tutti/config/tutti_runtime_spec.h>
#include <tutti/resource.h>
#include <tutti/spi/data_path.h>
#include <tutti/spi/storage_target_resolver.h>
#include <tutti/status.h>

namespace tutti::tutti_runtime {

struct BackendFactoryCacheConfig {
    std::uint32_t handle_cache_capacity = 0;
    std::uint32_t prp_cache_capacity = 0;
    std::uint32_t handle_cache_l2_capacity = 0;
};

struct BackendFactoryContext {
    const config::BackendSpec& backend;
    const config::ResolverSpec& resolver;
    const config::DataPathSpec& datapath;
    const config::ResourceSpec& resource_spec;
    const Resource& resource;
    std::int32_t runtime_accel_id = -1;
    BackendFactoryCacheConfig cache;
};

struct BackendShardProjection {
    std::int32_t device_id = -1;
    std::string controller_pci_addr;
    std::uint32_t namespace_id = 0;
    std::uint32_t logical_block_size = 0;
    std::uint64_t max_data_size = 0;
};

struct BackendFactoryProduct {
    std::unique_ptr<StorageTargetResolver> resolver;
    std::unique_ptr<DataPath> datapath;
    std::string scheme;
    std::string data_path_key;
    DataPathConfig data_path_config;

    std::string resolver_type_id;
    std::string payload_type_id;
    std::uint32_t payload_api_version = 0;
    std::vector<BackendShardProjection> resolver_shards;
    std::vector<BackendShardProjection> datapath_shards;
    std::uint64_t effective_max_data_size = 0;
    std::uint64_t stripe_unit = 0;
};

using BackendFactory = std::function<Result<BackendFactoryProduct>(
    const BackendFactoryContext&)>;
using RegisteredBackendFactory = Result<BackendFactoryProduct> (*)(
    const BackendFactoryContext&);

struct RuntimeBackendRegistration {
    std::string_view contract;
    std::string_view resolver_type_id;
    std::string_view payload_type_id;
    std::uint32_t payload_api_version = 0;
    std::string_view data_path_key;
    std::size_t minimum_cardinality = 0;
    std::size_t maximum_cardinality = 0;
    RegisteredBackendFactory create = nullptr;
};

const RuntimeBackendRegistration* find_backend_factory(
    std::string_view contract);

Result<BackendFactoryProduct> create_backend_from_registry(
    const BackendFactoryContext& context);

Status validate_backend_factory_product(
    const BackendFactoryContext& context,
    const RuntimeBackendRegistration& registration,
    const BackendFactoryProduct& product);

} // namespace tutti::tutti_runtime

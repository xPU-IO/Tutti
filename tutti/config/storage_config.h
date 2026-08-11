#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tutti::config {

enum class NvmeSelection {
    Allowed,
    Explicit,
    Striped,
};

enum class ConfigSyntax {
    Legacy,
    Canonical,
};

struct ProviderSpec {
    std::string type;
    std::string endpoint;
};

struct AllocationSpec {
    NvmeSelection selection = NvmeSelection::Allowed;
    std::vector<std::int32_t> device_ids;
    std::int32_t queues_per_controller = 0;
};

struct ResourceSpec {
    std::string id;
    std::string type;
    ProviderSpec provider;
    AllocationSpec allocation;
    std::uint64_t capacity_bytes = 0;
};

struct ResolverSpec {
    std::string id;
    std::string type;
    std::string scheme;
};

struct DataPathSpec {
    std::string id;
    std::string type;
    std::uint32_t handle_cache_capacity = 0;
    std::uint32_t prp_cache_capacity = 0;
    std::uint32_t handle_cache_l2_capacity = 0;
    std::uint64_t max_in_flight_operations = 0;
    std::uint64_t max_batch_entries = 0;
    std::uint64_t io_granularity = 0;
};

struct BackendSpec {
    std::string id;
    std::string contract;
    std::string resolver;
    std::string datapath;
    std::string resource;
    std::uint64_t stripe_unit = 0;
};

struct CanonicalStorageConfig {
    bool present = false;
    std::vector<ResourceSpec> resources;
    std::vector<ResolverSpec> resolvers;
    std::vector<DataPathSpec> datapaths;
    std::vector<BackendSpec> backends;
};

struct StorageContract {
    std::string_view name;
    std::string_view resolver_type;
    std::string_view resolver_scheme;
    std::string_view datapath_type;
    std::string_view resource_type;
    std::size_t minimum_cardinality = 0;
    std::size_t maximum_cardinality = 0;
    std::string_view resolver_type_id;
    std::string_view payload_type_id;
    std::uint32_t payload_api_version = 0;
    std::string_view data_path_key;
    bool implemented = false;
};

const StorageContract* find_storage_contract(std::string_view name);

} // namespace tutti::config

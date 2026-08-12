#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <tutti/status.h>

#ifndef TUTTI_COMPILED_ACCELERATOR_PROFILE
#define TUTTI_COMPILED_ACCELERATOR_PROFILE "HOST"
#endif
#ifndef TUTTI_DEFAULT_ACCEL_ID
#define TUTTI_DEFAULT_ACCEL_ID -1
#endif

namespace tutti::config {

inline constexpr std::uint64_t kDefaultStripedStripeUnit = 512 * 1024;

enum class NvmeSelection {
    Allowed,
    Explicit,
    Striped,
};

struct AcceleratorSpec {
    std::string profile = TUTTI_COMPILED_ACCELERATOR_PROFILE;
};

struct RuntimeSpec {
    std::int32_t accel_id = TUTTI_DEFAULT_ACCEL_ID;
};

struct NvmeProviderSpec {
    std::string type;
    std::string endpoint;
};

struct NvmeAllocationSpec {
    NvmeSelection selection = NvmeSelection::Allowed;
    std::vector<std::int32_t> device_ids;
    std::int32_t queues_per_controller = 0;
};

struct NvmeResourceConfig {
    NvmeProviderSpec provider;
    NvmeAllocationSpec allocation;
};

struct MemoryResourceConfig {
    std::uint64_t capacity_bytes = 0;
};

using ResourceConfig = std::variant<NvmeResourceConfig, MemoryResourceConfig>;

struct ResourceSpec {
    std::string id;
    std::string type;
    ResourceConfig config;
};

struct LocalFileResolverConfig {};
struct StripedFileResolverConfig {};
struct MemfsResolverConfig {};

using ResolverConfig = std::variant<LocalFileResolverConfig,
                                    StripedFileResolverConfig,
                                    MemfsResolverConfig>;

struct ResolverSpec {
    std::string id;
    std::string type;
    std::string scheme;
    ResolverConfig config;
};

struct NvmeDataPathTuning {
    std::uint32_t handle_cache_capacity = 0;
    std::uint32_t prp_cache_capacity = 0;
    std::uint32_t handle_cache_l2_capacity = 0;
    std::uint64_t max_in_flight_operations = 0;
    std::uint64_t max_batch_entries = 0;
    std::uint64_t io_granularity = 0;
};

struct LocalNvmeDataPathConfig : NvmeDataPathTuning {};
struct StripedLocalNvmeDataPathConfig : NvmeDataPathTuning {};
struct MemfsDataPathConfig {};

using DataPathConfig = std::variant<LocalNvmeDataPathConfig,
                                    StripedLocalNvmeDataPathConfig,
                                    MemfsDataPathConfig>;

struct DataPathSpec {
    std::string id;
    std::string type;
    DataPathConfig config;
};

struct Ext4LocalNvmeBackendConfig {};

struct StripedLocalNvmeBackendConfig {
    std::uint64_t stripe_unit = kDefaultStripedStripeUnit;
};

struct MemfsBackendConfig {};

using BackendConfig = std::variant<Ext4LocalNvmeBackendConfig,
                                   StripedLocalNvmeBackendConfig,
                                   MemfsBackendConfig>;

struct BackendSpec {
    std::string id;
    std::string contract;
    std::string resolver;
    std::string datapath;
    std::string resource;
    BackendConfig config;
};

struct StorageSpec {
    std::vector<ResourceSpec> resources;
    std::vector<ResolverSpec> resolvers;
    std::vector<DataPathSpec> datapaths;
    std::vector<BackendSpec> backends;
};

struct TuttiRuntimeSpec {
    AcceleratorSpec accelerator;
    RuntimeSpec runtime;
    StorageSpec storage;

    Status validate() const;
    Result<std::string> to_debug_string() const;
};

} // namespace tutti::config

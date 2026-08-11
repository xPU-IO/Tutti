// tutti/config/tutti_config.h
//
// Round 20 S1 — application config loader.
//
// Translates config/tutti_config.yaml into a RuntimeComponents struct
// passed to StorageRuntime::create().  The loader does NOT bypass the
// public API — it constructs DataPaths and Resolvers via the same
// constructors that programmatic callers use, then injects them.
//
// Priority (highest wins):
//   1. Programmatic injection (caller builds RuntimeComponents directly)
//   2. Config file (this loader)
//   3. Built-in DataPath defaults (0 = OFF / auto)
//
// TUTTI_HANDLE_CACHE_CAP / TUTTI_PRP_CACHE_CAP env vars are test-only
// backdoors: they apply ONLY when the config file omits the key.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <tutti/config/storage_config.h>
#include <tutti/status.h>
#include <tutti/tutti_runtime.h>

namespace tutti {

class DataPath;
class StorageRuntime;
class StorageTargetResolver;
struct RuntimeComponents;
struct RuntimeConfig;

#ifndef TUTTI_COMPILED_ACCELERATOR_PROFILE
#define TUTTI_COMPILED_ACCELERATOR_PROFILE "HOST"
#endif
#ifndef TUTTI_DEFAULT_ACCEL_ID
#define TUTTI_DEFAULT_ACCEL_ID -1
#endif

namespace config {

//   programmatic overrides for cache capacities that take precedence over the
//   config file. Pass 0 to defer to the config file (then env, then default).
struct ProgrammaticOverrides {
    std::uint32_t handle_cache_capacity = 0;  // 0 = defer to config/env
    std::uint32_t prp_cache_capacity = 0;
    std::uint32_t handle_cache_l2_capacity = 0;
};

struct LoadTuttiConfigOptions {
    ProgrammaticOverrides overrides;

    // Test seam: production builds leave these empty and use the compiled
    // backend + Resource factory + StorageRuntime::create.
    std::function<Result<int>()> backend_device_count;
    std::function<Result<std::unique_ptr<StorageRuntime>>(
        RuntimeConfig, RuntimeComponents)> runtime_factory;
    std::function<Result<std::unique_ptr<Resource>>(
        const ResourceSpec&, std::int32_t accel_id)> resource_factory;

    // Optional lifecycle seams used by contract tests. Production callers
    // leave these empty and TuttiRuntime invokes StorageRuntime::shutdown().
    std::function<Status(StorageRuntime&)> runtime_shutdown_hook;
    std::function<void(TuttiRuntimeShutdownStage)> shutdown_observer;
};

// Legacy parse-only device map entry. The phase-4 product loader no longer
// derives /dev paths from local_nvme_config, accelerator ordinal, or array
// order; it consumes daemon allocation metadata through NvmeResource.
struct DeviceSpec {
    std::uint32_t cuda_device = 0;
    std::string snvme_dev;
    std::uint32_t bar0_size = 16384;
    std::uint32_t namespace_id = 1;
    std::uint32_t block_size = 4096;
};

Result<std::unique_ptr<TuttiRuntime>> load_tutti_config(
    const std::string& path,
    const ProgrammaticOverrides& overrides = {});

Result<std::unique_ptr<TuttiRuntime>> load_tutti_config(
    const std::string& path,
    LoadTuttiConfigOptions options);

// Parse-only: returns the parsed config values without constructing
// any objects.  Useful for testing and validation.
struct ParsedConfig {
    ConfigSyntax syntax = ConfigSyntax::Legacy;
    CanonicalStorageConfig canonical_storage;

    std::string gpu_vendor = "nvidia";
    std::string accelerator_profile = TUTTI_COMPILED_ACCELERATOR_PROFILE;
    std::string storage_backend = "local-nvme";
    std::uint64_t default_stripe_unit = 0;

    std::int32_t runtime_accel_id = TUTTI_DEFAULT_ACCEL_ID;

    std::string nvme_service_endpoint = "127.0.0.1:50051";
    NvmeSelection nvme_selection = NvmeSelection::Allowed;
    std::string nvme_selection_text = "allowed";
    std::vector<std::int32_t> nvme_device_ids;
    std::int32_t queues_per_controller = 0;
    std::uint64_t stripe_unit = 65536;

    // local_nvme tuning
    std::uint32_t handle_cache_capacity = 0;
    std::uint32_t prp_cache_capacity = 0;
    std::uint32_t handle_cache_l2_capacity = 0;
    std::uint64_t max_in_flight_operations = 0;
    std::uint64_t max_batch_entries = 0;
    std::uint32_t num_user_queues = 0;
    std::uint64_t io_granularity = 0;

    // Link to the local-NVMe deployment fact file (may be empty;
    // resolved relative to the tutti_config.yaml directory).
    std::string local_nvme_config;
};

Result<ParsedConfig> parse_tutti_config(const std::string& path);

// Legacy helper for compatibility tests only. New loader code must not call
// this path because it derives /dev names from legacy YAML array order.
Result<std::vector<DeviceSpec>> derive_local_nvme_devices(
    const std::string& path);

// Resolve the effective cache capacities for a given config + overrides,
// applying the priority chain (programmatic > config file > env > default).
// Exposed for testing.
struct EffectiveCacheConfig {
    std::uint32_t handle_cache_capacity = 0;
    std::uint32_t prp_cache_capacity = 0;
    std::uint32_t handle_cache_l2_capacity = 0;
};
EffectiveCacheConfig resolve_cache_config(
    const ParsedConfig& parsed,
    const ProgrammaticOverrides& overrides);

} // namespace config
} // namespace tutti

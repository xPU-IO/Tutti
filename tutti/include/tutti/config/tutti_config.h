#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <tutti/config/storage_config.h>
#include <tutti/status.h>
#include <tutti/tutti_runtime.h>

#ifndef TUTTI_COMPILED_ACCELERATOR_PROFILE
#define TUTTI_COMPILED_ACCELERATOR_PROFILE "HOST"
#endif
#ifndef TUTTI_DEFAULT_ACCEL_ID
#define TUTTI_DEFAULT_ACCEL_ID -1
#endif

namespace tutti::config {

struct ProgrammaticOverrides {
    std::uint32_t handle_cache_capacity = 0;
    std::uint32_t prp_cache_capacity = 0;
    std::uint32_t handle_cache_l2_capacity = 0;
};

Result<std::unique_ptr<TuttiRuntime>> load_tutti_config(
    const std::string& path,
    const ProgrammaticOverrides& overrides = {});

struct ParsedConfig {
    CanonicalStorageConfig canonical_storage;
    std::string accelerator_profile = TUTTI_COMPILED_ACCELERATOR_PROFILE;
    std::int32_t runtime_accel_id = TUTTI_DEFAULT_ACCEL_ID;
};

Result<ParsedConfig> parse_tutti_config(const std::string& path);

struct EffectiveCacheConfig {
    std::uint32_t handle_cache_capacity = 0;
    std::uint32_t prp_cache_capacity = 0;
    std::uint32_t handle_cache_l2_capacity = 0;
};

EffectiveCacheConfig resolve_cache_config(
    const ParsedConfig& parsed,
    const ProgrammaticOverrides& overrides);

} // namespace tutti::config

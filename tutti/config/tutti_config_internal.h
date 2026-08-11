#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include <tutti/config/tutti_config.h>
#include "tutti/config/backend_factory.h"

namespace tutti {

class Resource;
class StorageRuntime;
struct RuntimeComponents;
struct RuntimeConfig;

namespace config {

struct LoadTuttiConfigOptions {
    ProgrammaticOverrides overrides;
    std::function<Result<int>()> backend_device_count;
    std::function<Result<std::unique_ptr<StorageRuntime>>(
        RuntimeConfig, RuntimeComponents)> runtime_factory;
    std::function<Result<std::unique_ptr<Resource>>(
        const ResourceSpec&, std::int32_t accel_id)> resource_factory;
    BackendFactory backend_factory;
    std::function<Status(StorageRuntime&)> runtime_shutdown_hook;
    std::function<void(TuttiRuntimeShutdownStage)> shutdown_observer;
};

Result<std::unique_ptr<TuttiRuntime>> load_tutti_config(
    const std::string& path,
    LoadTuttiConfigOptions options);

} // namespace config
} // namespace tutti

#pragma once

#include <cstdint>
#include <memory>

#include <tutti/config/spec/backend/backend_spec.h>
#include <tutti/config/spec/datapath/datapath_spec.h>
#include <tutti/resource.h>
#include <tutti/spi/data_path.h>

namespace tutti::data_paths {

struct DataPathCreateContext {
    const Resource& resource;
    const config::BackendSpec& relation;
    std::int32_t runtime_accel_id = -1;
};

struct CreatedDataPath {
    std::unique_ptr<DataPath> instance;
    DataPathConfig initialize_config;
};

Result<CreatedDataPath> create_data_path(
    const config::DataPathSpec& spec,
    const DataPathCreateContext& context);

} // namespace tutti::data_paths

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tutti/status.h>

#include <tutti/config/spec/backend/backend_spec.h>
#include <tutti/config/spec/datapath/datapath_spec.h>
#include <tutti/config/spec/resolver/resolver_spec.h>
#include <tutti/config/spec/resource/resource_spec.h>

#ifndef TUTTI_COMPILED_ACCELERATOR_PROFILE
#define TUTTI_COMPILED_ACCELERATOR_PROFILE "HOST"
#endif
#ifndef TUTTI_DEFAULT_ACCEL_ID
#define TUTTI_DEFAULT_ACCEL_ID -1
#endif

namespace tutti::config {

struct AcceleratorSpec {
    std::string profile = TUTTI_COMPILED_ACCELERATOR_PROFILE;
};

struct RuntimeSpec {
    std::int32_t accel_id = TUTTI_DEFAULT_ACCEL_ID;
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

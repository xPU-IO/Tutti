#pragma once

#include <string>
#include <variant>

#include <tutti/config/spec/resource/memory_spec.h>
#include <tutti/config/spec/resource/nvme_spec.h>

namespace tutti::config {

using ResourceConfig = std::variant<NvmeResourceConfig, MemoryResourceConfig>;

struct ResourceSpec {
    std::string id;
    std::string type;
    ResourceConfig config;
};

} // namespace tutti::config

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tutti::config {

enum class NvmeSelection {
    Allowed,
    Explicit,
    Striped,
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

} // namespace tutti::config

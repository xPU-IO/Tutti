#include "tutti/config/spec/spec_internal.h"

#include <limits>
#include <unordered_set>

namespace tutti::config::detail {

Status validate_nvme_resource(const ResourceSpec& spec,
                              const std::string& path) {
    const auto* config = std::get_if<NvmeResourceConfig>(&spec.config);
    if (config == nullptr) {
        return invalid_spec(path + ".config does not match type nvme");
    }
    if (config->provider.type != "nvme-service") {
        return invalid_spec(path + ".provider.type must be nvme-service");
    }
    if (config->provider.endpoint.empty()) {
        return invalid_spec(path + ".provider.endpoint must not be empty");
    }
    if (config->allocation.queues_per_controller <= 0) {
        return invalid_spec(path +
                            ".allocation.queues_per_controller must be greater than zero");
    }

    std::unordered_set<std::int32_t> device_ids;
    for (const std::int32_t id : config->allocation.device_ids) {
        if (id < 0) {
            return invalid_spec(path +
                                ".allocation.device_ids entries must be non-negative");
        }
        if (!device_ids.emplace(id).second) {
            return invalid_spec(path +
                                ".allocation.device_ids must not contain duplicates");
        }
    }

    const std::size_t count = config->allocation.device_ids.size();
    switch (config->allocation.selection) {
    case NvmeSelection::Allowed:
        if (count != 0) {
            return invalid_spec(path +
                                ".allocation selection=allowed requires empty device_ids");
        }
        break;
    case NvmeSelection::Explicit:
        if (count != 1) {
            return invalid_spec(path +
                                ".allocation selection=explicit requires exactly one device_id");
        }
        break;
    case NvmeSelection::Striped:
        if (count < 2) {
            return invalid_spec(path +
                                ".allocation selection=striped requires at least two device_ids");
        }
        break;
    }
    return Status::Ok();
}

} // namespace tutti::config::detail

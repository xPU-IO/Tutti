#include "tutti/config/spec/spec_internal.h"

namespace tutti::config::detail {

Status validate_striped_local_nvme_backend(const ResourceSpec& resource,
                                           const BackendSpec& backend,
                                           const SpecContract& contract,
                                           const std::string& path) {
    const auto* backend_config =
        std::get_if<StripedLocalNvmeBackendConfig>(&backend.config);
    if (backend_config == nullptr) {
        return invalid_spec(path +
                            ".config does not match contract striped-local-nvme");
    }
    if (backend_config->stripe_unit == 0 ||
        backend_config->stripe_unit % 4096 != 0) {
        return invalid_spec(path +
                            ".config.stripe_unit must be non-zero and 4096-byte aligned");
    }
    const auto* config = std::get_if<NvmeResourceConfig>(&resource.config);
    if (config == nullptr) {
        return invalid_spec(path + ".resource does not reference nvme config");
    }
    const std::size_t cardinality = config->allocation.device_ids.size();
    if (config->allocation.selection != NvmeSelection::Striped ||
        cardinality < contract.minimum_cardinality ||
        cardinality > contract.maximum_cardinality) {
        return invalid_spec(path +
                            " resource cardinality does not match striped-local-nvme");
    }
    return Status::Ok();
}

} // namespace tutti::config::detail

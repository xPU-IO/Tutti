#include "tutti/config/spec/spec_internal.h"

namespace tutti::config::detail {

Status validate_ext4_local_nvme_backend(const ResourceSpec& resource,
                                        const BackendSpec& backend,
                                        const SpecContract& contract,
                                        const std::string& path) {
    if (!std::holds_alternative<Ext4LocalNvmeBackendConfig>(backend.config)) {
        return invalid_spec(path +
                            ".config does not match contract ext4-local-nvme");
    }
    const auto* config = std::get_if<NvmeResourceConfig>(&resource.config);
    if (config == nullptr) {
        return invalid_spec(path + ".resource does not reference nvme config");
    }
    const std::size_t cardinality =
        config->allocation.selection == NvmeSelection::Allowed
            ? 1 : config->allocation.device_ids.size();
    if (cardinality < contract.minimum_cardinality ||
        cardinality > contract.maximum_cardinality ||
        config->allocation.selection == NvmeSelection::Striped) {
        return invalid_spec(path +
                            " resource cardinality does not match ext4-local-nvme");
    }
    return Status::Ok();
}

} // namespace tutti::config::detail

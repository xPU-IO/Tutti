#include "tutti/config/storage/parse_internal.h"

namespace tutti::config::detail {

Status parse_ext4_local_nvme_backend_config(const YAML::Node& config,
                                            const std::string& context,
                                            BackendSpec& backend) {
    if (!config) return Status::Ok();
    Status status = validate_keys(config, context, {"stripe_unit"});
    if (!status.ok()) return status;
    return read_nonnegative_u64(config, "stripe_unit", context,
                                backend.stripe_unit);
}

Status validate_ext4_local_nvme_backend(const ResourceSpec& resource,
                                        const BackendSpec& backend,
                                        const StorageContract& contract) {
    const std::size_t cardinality =
        resource.allocation.selection == NvmeSelection::Allowed
            ? 1
            : resource.allocation.device_ids.size();
    if (cardinality < contract.minimum_cardinality ||
        cardinality > contract.maximum_cardinality) {
        return invalid("backend " + backend.id +
                       " resource cardinality does not match contract " +
                       backend.contract);
    }
    if (resource.allocation.selection == NvmeSelection::Striped) {
        return invalid("ext4-local-nvme cannot use striped allocation");
    }
    if (backend.stripe_unit != 0) {
        return invalid("backend " + backend.id +
                       " does not allow stripe_unit");
    }
    return Status::Ok();
}

} // namespace tutti::config::detail

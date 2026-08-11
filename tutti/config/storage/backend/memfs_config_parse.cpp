#include "tutti/config/storage/parse_internal.h"

namespace tutti::config::detail {

Status parse_memfs_backend_config(const YAML::Node& config,
                                  const std::string& context,
                                  BackendSpec&) {
    if (!config) return Status::Ok();
    return validate_keys(config, context, {});
}

Status validate_memfs_backend(const ResourceSpec&,
                              const BackendSpec& backend,
                              const StorageContract& contract) {
    constexpr std::size_t cardinality = 1;
    if (cardinality < contract.minimum_cardinality ||
        cardinality > contract.maximum_cardinality) {
        return invalid("backend " + backend.id +
                       " resource cardinality does not match contract " +
                       backend.contract);
    }
    if (backend.stripe_unit != 0) {
        return invalid("backend " + backend.id +
                       " does not allow stripe_unit");
    }
    return Status::Ok();
}

} // namespace tutti::config::detail

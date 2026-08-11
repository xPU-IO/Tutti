#include "tutti/config/storage/parse_internal.h"

namespace tutti::config::detail {

Status parse_memory_resource(const YAML::Node& node,
                             const std::string& context,
                             ResourceSpec& resource) {
    if (node["provider"] || node["allocation"]) {
        return invalid(
            context + " memory resources cannot declare provider or allocation");
    }

    const YAML::Node config = node["config"];
    Status status = validate_keys(config, context + ".config",
                                  {"capacity_bytes"}, {"capacity_bytes"});
    if (!status.ok()) return status;
    status = read_nonnegative_u64(config, "capacity_bytes", context + ".config",
                                  resource.capacity_bytes, true);
    if (!status.ok()) return status;
    if (resource.capacity_bytes == 0) {
        return invalid(context +
                       ".config.capacity_bytes must be greater than zero");
    }
    return Status::Ok();
}

} // namespace tutti::config::detail

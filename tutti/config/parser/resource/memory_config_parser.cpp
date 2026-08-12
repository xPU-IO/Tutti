#include "tutti/config/parser/parser_internal.h"

namespace tutti::config::detail {

Status parse_memory_resource(const YAML::Node& node, const std::string& path,
                             ResourceSpec& resource) {
    if (node["provider"] || node["allocation"]) {
        return parse_error(
            path + " memory resource cannot declare provider or allocation");
    }
    const YAML::Node specific = node["config"];
    Status status = validate_keys(specific, path + ".config",
                                  {"capacity_bytes"}, {"capacity_bytes"});
    if (!status.ok()) return status;
    MemoryResourceConfig config;
    status = read_required_u64(specific, "capacity_bytes", path + ".config",
                               config.capacity_bytes);
    if (!status.ok()) return status;
    resource.config = config;
    return Status::Ok();
}

} // namespace tutti::config::detail

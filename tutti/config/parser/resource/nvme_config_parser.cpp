#include "tutti/config/parser/resource/nvme_parser_internal.h"

namespace tutti::config::detail {

Status parse_nvme_resource(const YAML::Node& node, const std::string& path,
                           ResourceSpec& resource) {
    NvmeResourceConfig config;
    const YAML::Node provider = node["provider"];
    Status status = validate_keys(provider, path + ".provider",
                                  {"type", "endpoint"},
                                  {"type", "endpoint"});
    if (!status.ok()) return status;
    status = read_required_string(provider, "type", path + ".provider",
                                  config.provider.type);
    if (!status.ok()) return status;
    status = read_required_string(provider, "endpoint", path + ".provider",
                                  config.provider.endpoint);
    if (!status.ok()) return status;

    const YAML::Node allocation = node["allocation"];
    status = validate_keys(
        allocation, path + ".allocation",
        {"selection", "device_ids", "queues_per_controller"},
        {"selection", "queues_per_controller"});
    if (!status.ok()) return status;
    std::string selection;
    status = read_required_string(allocation, "selection",
                                  path + ".allocation", selection);
    if (!status.ok()) return status;
    auto parsed_selection = parse_nvme_selection(
        selection, path + ".allocation.selection");
    if (!parsed_selection.ok()) return parsed_selection.status();
    config.allocation.selection = parsed_selection.value();
    status = parse_nvme_device_ids(
        allocation["device_ids"], path + ".allocation.device_ids",
        config.allocation.device_ids);
    if (!status.ok()) return status;

    std::uint64_t queues = 0;
    status = read_required_u64(allocation, "queues_per_controller",
                               path + ".allocation", queues);
    if (!status.ok()) return status;
    if (queues > static_cast<std::uint64_t>(
                     std::numeric_limits<std::int32_t>::max())) {
        return parse_error(path +
                           ".allocation.queues_per_controller exceeds int32");
    }
    config.allocation.queues_per_controller =
        static_cast<std::int32_t>(queues);

    if (const YAML::Node specific = node["config"]) {
        status = validate_keys(specific, path + ".config", {});
        if (!status.ok()) return status;
    }
    resource.config = std::move(config);
    return Status::Ok();
}

} // namespace tutti::config::detail

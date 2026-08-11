#include "tutti/config/storage/resource/nvme_parse_internal.h"

namespace tutti::config::detail {

Status parse_nvme_resource(const YAML::Node& node,
                           const std::string& context,
                           ResourceSpec& resource) {
    const YAML::Node provider = node["provider"];
    Status status = validate_keys(provider, context + ".provider",
                                  {"type", "endpoint"},
                                  {"type", "endpoint"});
    if (!status.ok()) return status;
    status = read_required_string(provider, "type", context + ".provider",
                                  resource.provider.type);
    if (!status.ok()) return status;
    status = read_required_string(provider, "endpoint", context + ".provider",
                                  resource.provider.endpoint);
    if (!status.ok()) return status;
    if (resource.provider.type != "nvme-service") {
        return invalid(context + ".provider.type must be nvme-service");
    }

    const YAML::Node allocation = node["allocation"];
    status = validate_keys(
        allocation, context + ".allocation",
        {"selection", "device_ids", "queues_per_controller"},
        {"selection", "queues_per_controller"});
    if (!status.ok()) return status;
    std::string selection;
    status = read_required_string(allocation, "selection",
                                  context + ".allocation", selection);
    if (!status.ok()) return status;
    auto parsed_selection = parse_nvme_selection(selection);
    if (!parsed_selection.ok()) return parsed_selection.status();
    resource.allocation.selection = parsed_selection.value();
    status = parse_nvme_device_ids(allocation["device_ids"],
                                   context + ".allocation.device_ids",
                                   resource.allocation.device_ids);
    if (!status.ok()) return status;
    status = read_nonnegative_i32(
        allocation, "queues_per_controller", context + ".allocation",
        resource.allocation.queues_per_controller, true);
    if (!status.ok()) return status;
    status = validate_nvme_selection(resource.allocation,
                                     context + ".allocation");
    if (!status.ok()) return status;
    if (const YAML::Node config = node["config"])
        return validate_keys(config, context + ".config", {});
    return Status::Ok();
}

} // namespace tutti::config::detail

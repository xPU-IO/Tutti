#include <tutti/config/tutti_runtime_config_parser.h>

#include <limits>
#include <string>
#include <unordered_set>
#include <utility>

#include "tutti/config/parser/parser_internal.h"

namespace tutti::config {
namespace detail {

Status parse_error(std::string message) {
    return Status(StatusCode::INVALID_ARGUMENT, std::move(message));
}

Status validate_keys(const YAML::Node& node, const std::string& path,
                     std::initializer_list<const char*> allowed,
                     std::initializer_list<const char*> required) {
    if (!node || !node.IsMap()) {
        return parse_error(path + " must be a mapping");
    }
    std::unordered_set<std::string> allowed_keys;
    for (const char* key : allowed) allowed_keys.emplace(key);
    for (const auto& entry : node) {
        if (!entry.first.IsScalar()) {
            return parse_error(path + " keys must be strings");
        }
        const std::string key = entry.first.as<std::string>();
        if (allowed_keys.count(key) == 0) {
            return parse_error(path + " contains unknown field " + key);
        }
    }
    for (const char* key : required) {
        if (!node[key]) {
            return parse_error(path + " requires field " + key);
        }
    }
    return Status::Ok();
}

Status read_required_string(const YAML::Node& node, const char* key,
                            const std::string& path, std::string& value) {
    const YAML::Node field = node[key];
    if (!field || !field.IsScalar()) {
        return parse_error(path + "." + key + " must be a string scalar");
    }
    value = field.as<std::string>();
    if (value.empty()) {
        return parse_error(path + "." + key + " must not be empty");
    }
    return Status::Ok();
}

Status read_optional_u32(const YAML::Node& node, const char* key,
                         const std::string& path, std::uint32_t& value) {
    const YAML::Node field = node[key];
    if (!field) return Status::Ok();
    if (!field.IsScalar()) {
        return parse_error(path + "." + key + " must be a uint32 scalar");
    }
    std::int64_t parsed = 0;
    try {
        parsed = field.as<std::int64_t>();
    } catch (const YAML::Exception&) {
        return parse_error(path + "." + key + " must be a uint32");
    }
    if (parsed < 0 || static_cast<std::uint64_t>(parsed) >
                          std::numeric_limits<std::uint32_t>::max()) {
        return parse_error(path + "." + key + " must be a uint32");
    }
    value = static_cast<std::uint32_t>(parsed);
    return Status::Ok();
}

Status read_optional_u64(const YAML::Node& node, const char* key,
                         const std::string& path, std::uint64_t& value) {
    const YAML::Node field = node[key];
    if (!field) return Status::Ok();
    if (!field.IsScalar()) {
        return parse_error(path + "." + key + " must be a uint64 scalar");
    }
    try {
        value = field.as<std::uint64_t>();
    } catch (const YAML::Exception&) {
        return parse_error(path + "." + key + " must be a uint64");
    }
    return Status::Ok();
}

Status read_required_u64(const YAML::Node& node, const char* key,
                         const std::string& path, std::uint64_t& value) {
    if (!node[key]) return parse_error(path + " requires field " + key);
    return read_optional_u64(node, key, path, value);
}

} // namespace detail
namespace {

template <typename T>
Result<T> failure(Status status) {
    return Result<T>::Failure(std::move(status));
}

Status parse_resource(const YAML::Node& node, std::size_t index,
                      ResourceSpec& resource) {
    const std::string path =
        "storage.resources[" + std::to_string(index) + "]";
    Status status = detail::validate_keys(
        node, path, {"id", "type", "provider", "allocation", "config"},
        {"id", "type"});
    if (!status.ok()) return status;
    status = detail::read_required_string(node, "id", path, resource.id);
    if (!status.ok()) return status;
    status = detail::read_required_string(node, "type", path, resource.type);
    if (!status.ok()) return status;
    if (resource.type == "nvme") {
        return detail::parse_nvme_resource(node, path, resource);
    }
    if (resource.type == "memory") {
        return detail::parse_memory_resource(node, path, resource);
    }
    return detail::parse_error(path + ".type is unknown: " + resource.type);
}

Status parse_resolver(const YAML::Node& node, std::size_t index,
                      ResolverSpec& resolver) {
    const std::string path =
        "storage.resolvers[" + std::to_string(index) + "]";
    Status status = detail::validate_keys(
        node, path, {"id", "type", "scheme", "config"},
        {"id", "type", "scheme"});
    if (!status.ok()) return status;
    status = detail::read_required_string(node, "id", path, resolver.id);
    if (!status.ok()) return status;
    status = detail::read_required_string(node, "type", path, resolver.type);
    if (!status.ok()) return status;
    status = detail::read_required_string(node, "scheme", path,
                                          resolver.scheme);
    if (!status.ok()) return status;
    const YAML::Node config = node["config"];
    if (resolver.type == "local-file") {
        return detail::parse_local_file_resolver(config, path + ".config",
                                                 resolver);
    }
    if (resolver.type == "striped-file") {
        return detail::parse_striped_file_resolver(config, path + ".config",
                                                   resolver);
    }
    if (resolver.type == "memfs") {
        return detail::parse_memfs_resolver(config, path + ".config", resolver);
    }
    return detail::parse_error(path + ".type is unknown: " + resolver.type);
}

Status parse_datapath(const YAML::Node& node, std::size_t index,
                      DataPathSpec& datapath) {
    const std::string path =
        "storage.datapaths[" + std::to_string(index) + "]";
    Status status = detail::validate_keys(node, path,
                                          {"id", "type", "config"},
                                          {"id", "type"});
    if (!status.ok()) return status;
    status = detail::read_required_string(node, "id", path, datapath.id);
    if (!status.ok()) return status;
    status = detail::read_required_string(node, "type", path, datapath.type);
    if (!status.ok()) return status;
    const YAML::Node config = node["config"];
    if (datapath.type == "local-nvme") {
        return detail::parse_local_nvme_datapath(config, path + ".config",
                                                 datapath);
    }
    if (datapath.type == "striped-local-nvme") {
        return detail::parse_striped_local_nvme_datapath(
            config, path + ".config", datapath);
    }
    if (datapath.type == "memfs") {
        return detail::parse_memfs_datapath(config, path + ".config", datapath);
    }
    return detail::parse_error(path + ".type is unknown: " + datapath.type);
}

Status parse_backend(const YAML::Node& node, std::size_t index,
                     BackendSpec& backend) {
    const std::string path =
        "storage.backends[" + std::to_string(index) + "]";
    Status status = detail::validate_keys(
        node, path,
        {"id", "contract", "resolver", "datapath", "resource", "config"},
        {"id", "contract", "resolver", "datapath", "resource"});
    if (!status.ok()) return status;
    status = detail::read_required_string(node, "id", path, backend.id);
    if (!status.ok()) return status;
    status = detail::read_required_string(node, "contract", path,
                                          backend.contract);
    if (!status.ok()) return status;
    status = detail::read_required_string(node, "resolver", path,
                                          backend.resolver);
    if (!status.ok()) return status;
    status = detail::read_required_string(node, "datapath", path,
                                          backend.datapath);
    if (!status.ok()) return status;
    status = detail::read_required_string(node, "resource", path,
                                          backend.resource);
    if (!status.ok()) return status;
    const YAML::Node config = node["config"];
    if (backend.contract == "ext4-local-nvme") {
        return detail::parse_ext4_local_nvme_backend(
            config, path + ".config", backend);
    }
    if (backend.contract == "striped-local-nvme") {
        return detail::parse_striped_local_nvme_backend(
            config, path + ".config", backend);
    }
    if (backend.contract == "memfs") {
        return detail::parse_memfs_backend(config, path + ".config", backend);
    }
    if (config) {
        status = detail::validate_keys(config, path + ".config", {});
        if (!status.ok()) return status;
    }
    backend.config = Ext4LocalNvmeBackendConfig{};
    return Status::Ok();
}

Status parse_storage(const YAML::Node& node, StorageSpec& storage) {
    Status status = detail::validate_keys(
        node, "storage", {"resources", "resolvers", "datapaths", "backends"},
        {"resources", "resolvers", "datapaths", "backends"});
    if (!status.ok()) return status;
    const YAML::Node resources = node["resources"];
    const YAML::Node resolvers = node["resolvers"];
    const YAML::Node datapaths = node["datapaths"];
    const YAML::Node backends = node["backends"];
    if (!resources.IsSequence()) {
        return detail::parse_error("storage.resources must be a sequence");
    }
    if (!resolvers.IsSequence()) {
        return detail::parse_error("storage.resolvers must be a sequence");
    }
    if (!datapaths.IsSequence()) {
        return detail::parse_error("storage.datapaths must be a sequence");
    }
    if (!backends.IsSequence()) {
        return detail::parse_error("storage.backends must be a sequence");
    }
    for (std::size_t index = 0; index < resources.size(); ++index) {
        ResourceSpec resource;
        status = parse_resource(resources[index], index, resource);
        if (!status.ok()) return status;
        storage.resources.push_back(std::move(resource));
    }
    for (std::size_t index = 0; index < resolvers.size(); ++index) {
        ResolverSpec resolver;
        status = parse_resolver(resolvers[index], index, resolver);
        if (!status.ok()) return status;
        storage.resolvers.push_back(std::move(resolver));
    }
    for (std::size_t index = 0; index < datapaths.size(); ++index) {
        DataPathSpec datapath;
        status = parse_datapath(datapaths[index], index, datapath);
        if (!status.ok()) return status;
        storage.datapaths.push_back(std::move(datapath));
    }
    for (std::size_t index = 0; index < backends.size(); ++index) {
        BackendSpec backend;
        status = parse_backend(backends[index], index, backend);
        if (!status.ok()) return status;
        storage.backends.push_back(std::move(backend));
    }
    return Status::Ok();
}

} // namespace

Result<TuttiRuntimeSpec> parse_tutti_runtime_config(const std::string& path) {
    try {
        const YAML::Node root = YAML::LoadFile(path);
        if (!root || root.IsNull() || !root.IsMap()) {
            return failure<TuttiRuntimeSpec>(
                detail::parse_error("root must be a mapping"));
        }
        Status status = detail::validate_keys(
            root, "root", {"accelerator", "runtime", "storage"},
            {"storage"});
        if (!status.ok()) return failure<TuttiRuntimeSpec>(std::move(status));

        TuttiRuntimeSpec spec;
        if (const YAML::Node accelerator = root["accelerator"]) {
            status = detail::validate_keys(accelerator, "accelerator",
                                           {"profile"});
            if (!status.ok()) {
                return failure<TuttiRuntimeSpec>(std::move(status));
            }
            if (accelerator["profile"]) {
                status = detail::read_required_string(
                    accelerator, "profile", "accelerator",
                    spec.accelerator.profile);
                if (!status.ok()) {
                    return failure<TuttiRuntimeSpec>(std::move(status));
                }
            }
        }
        if (const YAML::Node runtime = root["runtime"]) {
            status = detail::validate_keys(runtime, "runtime", {"accel_id"});
            if (!status.ok()) {
                return failure<TuttiRuntimeSpec>(std::move(status));
            }
            if (const YAML::Node accel_id = runtime["accel_id"]) {
                if (!accel_id.IsScalar()) {
                    return failure<TuttiRuntimeSpec>(detail::parse_error(
                        "runtime.accel_id must be an int32 scalar"));
                }
                std::int64_t parsed = 0;
                try {
                    parsed = accel_id.as<std::int64_t>();
                } catch (const YAML::Exception&) {
                    return failure<TuttiRuntimeSpec>(detail::parse_error(
                        "runtime.accel_id must be an int32 scalar"));
                }
                if (parsed < std::numeric_limits<std::int32_t>::min() ||
                    parsed > std::numeric_limits<std::int32_t>::max()) {
                    return failure<TuttiRuntimeSpec>(detail::parse_error(
                        "runtime.accel_id must fit int32"));
                }
                spec.runtime.accel_id = static_cast<std::int32_t>(parsed);
            }
        }
        status = parse_storage(root["storage"], spec.storage);
        if (!status.ok()) return failure<TuttiRuntimeSpec>(std::move(status));
        return Result<TuttiRuntimeSpec>::Success(std::move(spec));
    } catch (const YAML::Exception& exception) {
        return failure<TuttiRuntimeSpec>(detail::parse_error(
            "yaml parse error: " + std::string(exception.what())));
    } catch (const std::exception& exception) {
        return failure<TuttiRuntimeSpec>(detail::parse_error(
            "config parse error: " + std::string(exception.what())));
    }
}

} // namespace tutti::config

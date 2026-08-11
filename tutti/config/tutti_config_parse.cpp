#include "tutti/config/storage/parse_internal.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tutti::config {

namespace detail {

Status invalid(std::string message) {
    return Status(StatusCode::INVALID_ARGUMENT, std::move(message));
}

Status validate_keys(const YAML::Node& node, const std::string& context,
                     std::initializer_list<const char*> allowed,
                     std::initializer_list<const char*> required) {
    if (!node || !node.IsMap()) {
        return invalid(context + " must be a mapping");
    }
    std::unordered_set<std::string> allowed_keys;
    for (const char* key : allowed) allowed_keys.emplace(key);
    for (const auto& entry : node) {
        if (!entry.first.IsScalar()) {
            return invalid(context + " keys must be strings");
        }
        const std::string key = entry.first.as<std::string>();
        if (allowed_keys.count(key) == 0) {
            return invalid(context + " contains unknown field " + key);
        }
    }
    for (const char* key : required) {
        if (!node[key]) {
            return invalid(context + " requires field " + key);
        }
    }
    return Status::Ok();
}

Status read_required_string(const YAML::Node& node, const char* key,
                            const std::string& context, std::string& value) {
    const YAML::Node field = node[key];
    if (!field || !field.IsScalar()) {
        return invalid(context + "." + key + " must be a string");
    }
    value = field.as<std::string>();
    if (value.empty()) {
        return invalid(context + "." + key + " must not be empty");
    }
    return Status::Ok();
}

Status read_nonnegative_i32(const YAML::Node& node, const char* key,
                            const std::string& context, std::int32_t& value,
                            bool required) {
    const YAML::Node field = node[key];
    if (!field) {
        return required ? invalid(context + " requires field " + key)
                        : Status::Ok();
    }
    const std::int64_t parsed = field.as<std::int64_t>();
    if (parsed < 0 || parsed > std::numeric_limits<std::int32_t>::max()) {
        return invalid(context + "." + key + " must be a non-negative int32");
    }
    value = static_cast<std::int32_t>(parsed);
    return Status::Ok();
}

Status read_nonnegative_u32(const YAML::Node& node, const char* key,
                            const std::string& context, std::uint32_t& value) {
    const YAML::Node field = node[key];
    if (!field) return Status::Ok();
    const std::int64_t parsed = field.as<std::int64_t>();
    if (parsed < 0 ||
        static_cast<std::uint64_t>(parsed) >
            std::numeric_limits<std::uint32_t>::max()) {
        return invalid(context + "." + key + " must be a non-negative uint32");
    }
    value = static_cast<std::uint32_t>(parsed);
    return Status::Ok();
}

Status read_nonnegative_u64(const YAML::Node& node, const char* key,
                            const std::string& context, std::uint64_t& value,
                            bool required) {
    const YAML::Node field = node[key];
    if (!field) {
        return required ? invalid(context + " requires field " + key)
                        : Status::Ok();
    }
    const std::int64_t parsed = field.as<std::int64_t>();
    if (parsed < 0) {
        return invalid(context + "." + key + " must be non-negative");
    }
    value = static_cast<std::uint64_t>(parsed);
    return Status::Ok();
}

} // namespace detail

namespace {

using detail::invalid;
using detail::read_required_string;
using detail::validate_keys;

constexpr std::array<StorageContract, 3> kStorageContracts{{
    {"ext4-local-nvme", "local-file", "local-nvme", "nvme",
     "local-nvme-ext4", 1, 1, true},
    {"striped-local-nvme", "striped-file", "striped-local-nvme", "nvme",
     "striped-local-nvme", 2, std::numeric_limits<std::size_t>::max(), true},
    {"memfs", "memfs", "memfs", "memory", "memfs", 1, 1, false},
}};

template <typename T>
Result<T> failure(Status status) {
    return Result<T>::Failure(std::move(status));
}

Status parse_common(const YAML::Node& root, ParsedConfig& config) {
    if (const YAML::Node gpu = root["gpu"]) {
        if (const YAML::Node vendor = gpu["vendor"])
            config.gpu_vendor = vendor.as<std::string>();
    }
    if (const YAML::Node accelerator = root["accelerator"]) {
        if (const YAML::Node profile = accelerator["profile"])
            config.accelerator_profile = profile.as<std::string>();
    }
    if (const YAML::Node runtime = root["runtime"]) {
        if (const YAML::Node accel_id = runtime["accel_id"])
            config.runtime_accel_id = accel_id.as<std::int32_t>();
    }
    if (config.runtime_accel_id < -1) {
        return invalid("runtime.accel_id must be -1 or non-negative");
    }
    return Status::Ok();
}

Status parse_resource(const YAML::Node& node, std::size_t index,
                      ResourceSpec& resource) {
    const std::string context =
        "storage.resources[" + std::to_string(index) + "]";
    Status status = validate_keys(
        node, context, {"id", "type", "provider", "allocation", "config"},
        {"id", "type"});
    if (!status.ok()) return status;
    status = read_required_string(node, "id", context, resource.id);
    if (!status.ok()) return status;
    status = read_required_string(node, "type", context, resource.type);
    if (!status.ok()) return status;

    if (resource.type == "nvme")
        return detail::parse_nvme_resource(node, context, resource);
    if (resource.type == "memory")
        return detail::parse_memory_resource(node, context, resource);

    return invalid(context + ".type is unknown: " + resource.type);
}

Status parse_resolver(const YAML::Node& node, std::size_t index,
                      ResolverSpec& resolver) {
    const std::string context =
        "storage.resolvers[" + std::to_string(index) + "]";
    Status status = validate_keys(node, context, {"id", "type", "scheme", "config"},
                                  {"id", "type", "scheme"});
    if (!status.ok()) return status;
    status = read_required_string(node, "id", context, resolver.id);
    if (!status.ok()) return status;
    status = read_required_string(node, "type", context, resolver.type);
    if (!status.ok()) return status;
    status = read_required_string(node, "scheme", context, resolver.scheme);
    if (!status.ok()) return status;
    const YAML::Node config = node["config"];
    if (resolver.type == "local-file") {
        return detail::parse_local_file_resolver_config(
            config, context + ".config", resolver);
    }
    if (resolver.type == "striped-file") {
        return detail::parse_striped_file_resolver_config(
            config, context + ".config", resolver);
    }
    if (resolver.type == "memfs") {
        return detail::parse_memfs_resolver_config(
            config, context + ".config", resolver);
    }
    return invalid(context + ".type is unknown: " + resolver.type);
}

Status parse_datapath(const YAML::Node& node, std::size_t index,
                      DataPathSpec& datapath) {
    const std::string context =
        "storage.datapaths[" + std::to_string(index) + "]";
    Status status = validate_keys(node, context, {"id", "type", "config"},
                                  {"id", "type"});
    if (!status.ok()) return status;
    status = read_required_string(node, "id", context, datapath.id);
    if (!status.ok()) return status;
    status = read_required_string(node, "type", context, datapath.type);
    if (!status.ok()) return status;
    const YAML::Node config = node["config"];
    if (datapath.type == "local-nvme") {
        return detail::parse_local_nvme_datapath_config(
            config, context + ".config", datapath);
    }
    if (datapath.type == "striped-local-nvme") {
        return detail::parse_striped_local_nvme_datapath_config(
            config, context + ".config", datapath);
    }
    if (datapath.type == "memfs") {
        return detail::parse_memfs_datapath_config(
            config, context + ".config", datapath);
    }
    return invalid(context + ".type is unknown: " + datapath.type);
}

Status parse_backend(const YAML::Node& node, std::size_t index,
                     BackendSpec& backend) {
    const std::string context =
        "storage.backends[" + std::to_string(index) + "]";
    Status status = validate_keys(
        node, context,
        {"id", "contract", "resolver", "datapath", "resource", "config"},
        {"id", "contract", "resolver", "datapath", "resource"});
    if (!status.ok()) return status;
    status = read_required_string(node, "id", context, backend.id);
    if (!status.ok()) return status;
    status = read_required_string(node, "contract", context, backend.contract);
    if (!status.ok()) return status;
    status = read_required_string(node, "resolver", context, backend.resolver);
    if (!status.ok()) return status;
    status = read_required_string(node, "datapath", context, backend.datapath);
    if (!status.ok()) return status;
    status = read_required_string(node, "resource", context, backend.resource);
    if (!status.ok()) return status;
    const YAML::Node config = node["config"];
    if (backend.contract == "ext4-local-nvme") {
        return detail::parse_ext4_local_nvme_backend_config(
            config, context + ".config", backend);
    }
    if (backend.contract == "striped-local-nvme") {
        return detail::parse_striped_local_nvme_backend_config(
            config, context + ".config", backend);
    }
    if (backend.contract == "memfs") {
        return detail::parse_memfs_backend_config(
            config, context + ".config", backend);
    }
    if (config) return validate_keys(config, context + ".config", {});
    return Status::Ok();
}

template <typename Spec>
Status index_specs(const std::vector<Spec>& specs, const std::string& group,
                   std::unordered_map<std::string, const Spec*>& index) {
    for (const auto& spec : specs) {
        if (!index.emplace(spec.id, &spec).second) {
            return invalid("storage." + group + " contains duplicate id " + spec.id);
        }
    }
    return Status::Ok();
}

bool valid_scheme(const std::string& scheme) {
    if (scheme.empty() || scheme[0] < 'a' || scheme[0] > 'z') return false;
    return std::all_of(scheme.begin() + 1, scheme.end(), [](char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') ||
               ch == '+' || ch == '-' || ch == '.';
    });
}

Status validate_canonical(const CanonicalStorageConfig& storage) {
    std::unordered_map<std::string, const ResourceSpec*> resources;
    std::unordered_map<std::string, const ResolverSpec*> resolvers;
    std::unordered_map<std::string, const DataPathSpec*> datapaths;
    std::unordered_map<std::string, const BackendSpec*> backends;
    Status status = index_specs(storage.resources, "resources", resources);
    if (!status.ok()) return status;
    status = index_specs(storage.resolvers, "resolvers", resolvers);
    if (!status.ok()) return status;
    status = index_specs(storage.datapaths, "datapaths", datapaths);
    if (!status.ok()) return status;
    status = index_specs(storage.backends, "backends", backends);
    if (!status.ok()) return status;

    std::unordered_set<std::string> schemes;
    for (const auto& resolver : storage.resolvers) {
        if (!valid_scheme(resolver.scheme)) {
            return invalid("resolver " + resolver.id + " has invalid scheme " +
                           resolver.scheme);
        }
        if (!schemes.emplace(resolver.scheme).second) {
            return invalid("duplicate resolver scheme " + resolver.scheme);
        }
    }

    if (storage.backends.empty()) {
        return invalid("storage.backends must contain exactly one backend");
    }

    std::unordered_set<std::string> reachable_resources;
    std::unordered_set<std::string> reachable_resolvers;
    std::unordered_set<std::string> reachable_datapaths;
    std::unordered_set<std::string> data_path_keys;
    bool unsupported_contract = false;

    for (const auto& backend : storage.backends) {
        const StorageContract* contract = find_storage_contract(backend.contract);
        if (contract == nullptr) {
            return invalid("backend " + backend.id + " has unknown contract " +
                           backend.contract);
        }
        const auto resolver_it = resolvers.find(backend.resolver);
        const auto datapath_it = datapaths.find(backend.datapath);
        const auto resource_it = resources.find(backend.resource);
        if (resolver_it == resolvers.end()) {
            return invalid("backend " + backend.id + " references unknown resolver " +
                           backend.resolver);
        }
        if (datapath_it == datapaths.end()) {
            return invalid("backend " + backend.id + " references unknown datapath " +
                           backend.datapath);
        }
        if (resource_it == resources.end()) {
            return invalid("backend " + backend.id + " references unknown resource " +
                           backend.resource);
        }
        if (resolver_it->second->type != contract->resolver_type ||
            datapath_it->second->type != contract->datapath_type ||
            resource_it->second->type != contract->resource_type) {
            return invalid("backend " + backend.id +
                           " types do not match contract " + backend.contract);
        }
        if (!data_path_keys.emplace(contract->data_path_key).second) {
            return invalid("duplicate contract DataPath key " +
                           std::string(contract->data_path_key));
        }

        const ResourceSpec& resource = *resource_it->second;
        if (backend.contract == "ext4-local-nvme") {
            status = detail::validate_ext4_local_nvme_backend(
                resource, backend, *contract);
        } else if (backend.contract == "striped-local-nvme") {
            status = detail::validate_striped_local_nvme_backend(
                resource, backend, *contract);
        } else if (backend.contract == "memfs") {
            status = detail::validate_memfs_backend(resource, backend, *contract);
        } else {
            return invalid("backend " + backend.id +
                           " has unsupported contract " + backend.contract);
        }
        if (!status.ok()) return status;

        reachable_resources.emplace(backend.resource);
        reachable_resolvers.emplace(backend.resolver);
        reachable_datapaths.emplace(backend.datapath);
        unsupported_contract = unsupported_contract || !contract->implemented;
    }

    if (storage.backends.size() != 1) {
        return invalid("storage.backends must contain exactly one backend");
    }
    if (reachable_resources.size() != storage.resources.size() ||
        reachable_resolvers.size() != storage.resolvers.size() ||
        reachable_datapaths.size() != storage.datapaths.size()) {
        return invalid("all storage declarations must be reachable from the backend");
    }
    if (unsupported_contract) {
        return Status(StatusCode::UNSUPPORTED,
                      "memfs canonical config factory is not implemented");
    }
    return Status::Ok();
}

Status parse_storage_arrays(const YAML::Node& storage_node,
                            CanonicalStorageConfig& storage) {
    Status status = validate_keys(
        storage_node, "storage", {"resources", "resolvers", "datapaths", "backends"},
        {"resources", "resolvers", "datapaths", "backends"});
    if (!status.ok()) return status;
    storage.present = true;

    const YAML::Node resources = storage_node["resources"];
    const YAML::Node resolvers = storage_node["resolvers"];
    const YAML::Node datapaths = storage_node["datapaths"];
    const YAML::Node backends = storage_node["backends"];
    if (!resources.IsSequence() || !resolvers.IsSequence() ||
        !datapaths.IsSequence() || !backends.IsSequence()) {
        return invalid("storage resources/resolvers/datapaths/backends must be sequences");
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
    return validate_canonical(storage);
}

void apply_canonical_compatibility(ParsedConfig& config) {
    if (!config.canonical_storage.present ||
        config.canonical_storage.backends.empty()) {
        return;
    }
    const BackendSpec& backend = config.canonical_storage.backends.front();
    const auto resource_it = std::find_if(
        config.canonical_storage.resources.begin(),
        config.canonical_storage.resources.end(),
        [&](const ResourceSpec& resource) { return resource.id == backend.resource; });
    const auto datapath_it = std::find_if(
        config.canonical_storage.datapaths.begin(),
        config.canonical_storage.datapaths.end(),
        [&](const DataPathSpec& datapath) { return datapath.id == backend.datapath; });
    if (resource_it != config.canonical_storage.resources.end() &&
        resource_it->type == "nvme" &&
        datapath_it != config.canonical_storage.datapaths.end()) {
        detail::apply_nvme_compatibility(
            *resource_it, *datapath_it, backend, config);
    }
}

Result<ParsedConfig> parse_canonical(const YAML::Node& root) {
    Status status = validate_keys(root, "root", {"accelerator", "runtime", "storage"});
    if (!status.ok()) return failure<ParsedConfig>(status);
    if (const YAML::Node accelerator = root["accelerator"]) {
        status = validate_keys(accelerator, "accelerator", {"profile"});
        if (!status.ok()) return failure<ParsedConfig>(status);
    }
    if (const YAML::Node runtime = root["runtime"]) {
        status = validate_keys(runtime, "runtime", {"accel_id"});
        if (!status.ok()) return failure<ParsedConfig>(status);
    }

    ParsedConfig config;
    config.syntax = ConfigSyntax::Canonical;
    status = parse_common(root, config);
    if (!status.ok()) return failure<ParsedConfig>(status);
    if (!root["storage"]) {
        if (config.accelerator_profile != "HOST" || config.runtime_accel_id != -1) {
            return failure<ParsedConfig>(
                invalid("storage is required unless runtime uses HOST accel_id=-1"));
        }
        return Result<ParsedConfig>::Success(std::move(config));
    }
    status = parse_storage_arrays(root["storage"], config.canonical_storage);
    if (!status.ok()) return failure<ParsedConfig>(status);
    apply_canonical_compatibility(config);
    return Result<ParsedConfig>::Success(std::move(config));
}

Result<ParsedConfig> parse_legacy(const YAML::Node& root) {
    ParsedConfig config;
    config.syntax = ConfigSyntax::Legacy;
    Status status = parse_common(root, config);
    if (!status.ok()) return failure<ParsedConfig>(status);
    status = detail::parse_legacy_nvme(root, config);
    if (!status.ok()) return failure<ParsedConfig>(status);
    config.canonical_storage = detail::adapt_legacy_nvme(config);
    status = validate_canonical(config.canonical_storage);
    if (!status.ok()) return failure<ParsedConfig>(status);
    return Result<ParsedConfig>::Success(std::move(config));
}

bool has_key(const YAML::Node& node, const char* key) {
    return node && node.IsMap() && static_cast<bool>(node[key]);
}

} // namespace

const StorageContract* find_storage_contract(std::string_view name) {
    const auto found = std::find_if(
        kStorageContracts.begin(), kStorageContracts.end(),
        [&](const StorageContract& contract) { return contract.name == name; });
    return found == kStorageContracts.end() ? nullptr : &*found;
}

Result<ParsedConfig> parse_tutti_config(const std::string& path) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        if (!root || root.IsNull()) root = YAML::Node(YAML::NodeType::Map);
        if (!root.IsMap()) {
            return failure<ParsedConfig>(invalid("root must be a mapping"));
        }

        const YAML::Node storage = root["storage"];
        const bool canonical_storage =
            has_key(storage, "resources") || has_key(storage, "resolvers") ||
            has_key(storage, "datapaths") || has_key(storage, "backends");
        const bool legacy_storage =
            has_key(storage, "backend") || has_key(storage, "default_stripe_unit");
        const bool legacy_root =
            has_key(root, "gpu") || has_key(root, "nvme_service") ||
            has_key(root, "nvme") || has_key(root, "local_nvme") ||
            has_key(root, "local_nvme_config");

        if (canonical_storage && (legacy_storage || legacy_root)) {
            return failure<ParsedConfig>(
                invalid("canonical and legacy storage fields cannot be mixed"));
        }
        if (canonical_storage) return parse_canonical(root);
        if (legacy_storage || legacy_root) return parse_legacy(root);
        if (storage || root["accelerator"] || root["runtime"] || root.size() != 0)
            return parse_canonical(root);
        return parse_legacy(root);
    } catch (const YAML::Exception& exception) {
        return failure<ParsedConfig>(
            invalid("yaml parse error: " + std::string(exception.what())));
    } catch (const std::exception& exception) {
        return failure<ParsedConfig>(
            invalid("config parse error: " + std::string(exception.what())));
    }
}

} // namespace tutti::config

#include "tutti/config/tutti_config.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <yaml-cpp/yaml.h>

namespace tutti::config {

namespace {

constexpr std::array<StorageContract, 3> kStorageContracts{{
    {"ext4-local-nvme", "local-file", "local-nvme", "nvme",
     "local-nvme-ext4", 1, 1, true},
    {"striped-local-nvme", "striped-file", "striped-local-nvme", "nvme",
     "striped-local-nvme", 2, std::numeric_limits<std::size_t>::max(), true},
    {"memfs", "memfs", "memfs", "memory", "memfs", 1, 1, false},
}};

Status invalid(std::string message) {
    return Status(StatusCode::INVALID_ARGUMENT, std::move(message));
}

template <typename T>
Result<T> failure(Status status) {
    return Result<T>::Failure(std::move(status));
}

Result<NvmeSelection> parse_selection(const std::string& value) {
    if (value == "allowed") {
        return Result<NvmeSelection>::Success(NvmeSelection::Allowed);
    }
    if (value == "explicit") {
        return Result<NvmeSelection>::Success(NvmeSelection::Explicit);
    }
    if (value == "striped") {
        return Result<NvmeSelection>::Success(NvmeSelection::Striped);
    }
    return Result<NvmeSelection>::Failure(
        invalid("selection must be allowed, explicit, or striped"));
}

const char* selection_name(NvmeSelection selection) {
    switch (selection) {
    case NvmeSelection::Allowed:
        return "allowed";
    case NvmeSelection::Explicit:
        return "explicit";
    case NvmeSelection::Striped:
        return "striped";
    }
    return "allowed";
}

Status validate_keys(const YAML::Node& node, const std::string& context,
                     std::initializer_list<const char*> allowed,
                     std::initializer_list<const char*> required = {}) {
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
                            bool required = false) {
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

Status parse_device_ids(const YAML::Node& node, const std::string& context,
                        std::vector<std::int32_t>& device_ids) {
    if (!node) return Status::Ok();
    if (!node.IsSequence()) {
        return invalid(context + " must be a sequence");
    }
    std::unordered_set<std::int32_t> unique;
    for (const auto& entry : node) {
        const std::int64_t parsed = entry.as<std::int64_t>();
        if (parsed < 0 || parsed > std::numeric_limits<std::int32_t>::max()) {
            return invalid(context + " entries must be non-negative int32 values");
        }
        const auto device_id = static_cast<std::int32_t>(parsed);
        if (!unique.emplace(device_id).second) {
            return invalid(context + " must not contain duplicates");
        }
        device_ids.push_back(device_id);
    }
    return Status::Ok();
}

Status validate_selection(const AllocationSpec& allocation,
                          const std::string& context) {
    if (allocation.selection == NvmeSelection::Allowed &&
        !allocation.device_ids.empty()) {
        return invalid(context + " selection=allowed requires empty device_ids");
    }
    if (allocation.selection == NvmeSelection::Explicit &&
        allocation.device_ids.size() != 1) {
        return invalid(context + " selection=explicit requires exactly one device_id");
    }
    if (allocation.selection == NvmeSelection::Striped &&
        allocation.device_ids.size() < 2) {
        return invalid(context + " selection=striped requires at least two device_ids");
    }
    return Status::Ok();
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

    if (resource.type == "nvme") {
        const YAML::Node provider = node["provider"];
        status = validate_keys(provider, context + ".provider",
                               {"type", "endpoint"}, {"type", "endpoint"});
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
        auto parsed_selection = parse_selection(selection);
        if (!parsed_selection.ok()) return parsed_selection.status();
        resource.allocation.selection = parsed_selection.value();
        status = parse_device_ids(allocation["device_ids"],
                                  context + ".allocation.device_ids",
                                  resource.allocation.device_ids);
        if (!status.ok()) return status;
        status = read_nonnegative_i32(
            allocation, "queues_per_controller", context + ".allocation",
            resource.allocation.queues_per_controller, true);
        if (!status.ok()) return status;
        status = validate_selection(resource.allocation, context + ".allocation");
        if (!status.ok()) return status;
        if (const YAML::Node config = node["config"]) {
            status = validate_keys(config, context + ".config", {});
            if (!status.ok()) return status;
        }
        return Status::Ok();
    }

    if (resource.type == "memory") {
        if (node["provider"] || node["allocation"]) {
            return invalid(context + " memory resources cannot declare provider or allocation");
        }
        const YAML::Node config = node["config"];
        status = validate_keys(config, context + ".config", {"capacity_bytes"},
                               {"capacity_bytes"});
        if (!status.ok()) return status;
        status = read_nonnegative_u64(config, "capacity_bytes", context + ".config",
                                      resource.capacity_bytes, true);
        if (!status.ok()) return status;
        if (resource.capacity_bytes == 0) {
            return invalid(context + ".config.capacity_bytes must be greater than zero");
        }
        return Status::Ok();
    }

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
    if (resolver.type != "local-file" && resolver.type != "striped-file" &&
        resolver.type != "memfs") {
        return invalid(context + ".type is unknown: " + resolver.type);
    }
    if (const YAML::Node config = node["config"]) {
        status = validate_keys(config, context + ".config", {});
        if (!status.ok()) return status;
    }
    return Status::Ok();
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
    if (datapath.type != "local-nvme" &&
        datapath.type != "striped-local-nvme" && datapath.type != "memfs") {
        return invalid(context + ".type is unknown: " + datapath.type);
    }
    if (const YAML::Node config = node["config"]) {
        status = validate_keys(
            config, context + ".config",
            {"handle_cache_capacity", "prp_cache_capacity",
             "handle_cache_l2_capacity", "max_in_flight_operations",
             "max_batch_entries", "io_granularity"});
        if (!status.ok()) return status;
        status = read_nonnegative_u32(config, "handle_cache_capacity",
                                      context + ".config",
                                      datapath.handle_cache_capacity);
        if (!status.ok()) return status;
        status = read_nonnegative_u32(config, "prp_cache_capacity",
                                      context + ".config",
                                      datapath.prp_cache_capacity);
        if (!status.ok()) return status;
        status = read_nonnegative_u32(config, "handle_cache_l2_capacity",
                                      context + ".config",
                                      datapath.handle_cache_l2_capacity);
        if (!status.ok()) return status;
        status = read_nonnegative_u64(config, "max_in_flight_operations",
                                      context + ".config",
                                      datapath.max_in_flight_operations);
        if (!status.ok()) return status;
        status = read_nonnegative_u64(config, "max_batch_entries",
                                      context + ".config",
                                      datapath.max_batch_entries);
        if (!status.ok()) return status;
        status = read_nonnegative_u64(config, "io_granularity",
                                      context + ".config",
                                      datapath.io_granularity);
        if (!status.ok()) return status;
        if (datapath.type == "memfs" && config.size() != 0) {
            return invalid(context + ".config must be empty for memfs");
        }
    }
    return Status::Ok();
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
    if (const YAML::Node config = node["config"]) {
        status = validate_keys(config, context + ".config", {"stripe_unit"});
        if (!status.ok()) return status;
        status = read_nonnegative_u64(config, "stripe_unit", context + ".config",
                                      backend.stripe_unit);
        if (!status.ok()) return status;
    }
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
        std::size_t cardinality = 1;
        if (resource.type == "nvme") {
            cardinality = resource.allocation.selection == NvmeSelection::Allowed
                ? 1
                : resource.allocation.device_ids.size();
        }
        if (cardinality < contract->minimum_cardinality ||
            cardinality > contract->maximum_cardinality) {
            return invalid("backend " + backend.id +
                           " resource cardinality does not match contract " +
                           backend.contract);
        }
        if (backend.contract == "ext4-local-nvme" &&
            resource.allocation.selection == NvmeSelection::Striped) {
            return invalid("ext4-local-nvme cannot use striped allocation");
        }
        if (backend.contract == "striped-local-nvme" &&
            resource.allocation.selection != NvmeSelection::Striped) {
            return invalid("striped-local-nvme requires striped allocation");
        }
        if (backend.contract == "striped-local-nvme") {
            if (backend.stripe_unit == 0 || backend.stripe_unit % 4096 != 0) {
                return invalid("striped backend stripe_unit must be non-zero and 4096-byte aligned");
            }
        } else if (backend.stripe_unit != 0) {
            return invalid("backend " + backend.id +
                           " does not allow stripe_unit");
        }

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
        resource_it->type == "nvme") {
        config.nvme_service_endpoint = resource_it->provider.endpoint;
        config.nvme_selection = resource_it->allocation.selection;
        config.nvme_selection_text = selection_name(resource_it->allocation.selection);
        config.nvme_device_ids = resource_it->allocation.device_ids;
        config.queues_per_controller =
            resource_it->allocation.queues_per_controller;
    }
    if (datapath_it != config.canonical_storage.datapaths.end()) {
        config.storage_backend = datapath_it->type;
        config.handle_cache_capacity = datapath_it->handle_cache_capacity;
        config.prp_cache_capacity = datapath_it->prp_cache_capacity;
        config.handle_cache_l2_capacity = datapath_it->handle_cache_l2_capacity;
        config.max_in_flight_operations = datapath_it->max_in_flight_operations;
        config.max_batch_entries = datapath_it->max_batch_entries;
        config.io_granularity = datapath_it->io_granularity;
    }
    if (backend.stripe_unit != 0) config.stripe_unit = backend.stripe_unit;
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

CanonicalStorageConfig adapt_legacy(const ParsedConfig& config) {
    CanonicalStorageConfig storage;
    storage.present = true;

    ResourceSpec resource;
    resource.id = "legacy-nvme-resource";
    resource.type = "nvme";
    resource.provider.type = "nvme-service";
    resource.provider.endpoint = config.nvme_service_endpoint;
    resource.allocation.selection = config.nvme_selection;
    resource.allocation.device_ids = config.nvme_device_ids;
    resource.allocation.queues_per_controller = config.queues_per_controller;
    storage.resources.push_back(std::move(resource));

    const bool striped = config.nvme_selection == NvmeSelection::Striped;
    ResolverSpec resolver;
    resolver.id = "legacy-resolver";
    resolver.type = striped ? "striped-file" : "local-file";
    resolver.scheme = striped ? "striped" : "file";
    storage.resolvers.push_back(std::move(resolver));

    DataPathSpec datapath;
    datapath.id = "legacy-datapath";
    datapath.type = striped ? "striped-local-nvme" : "local-nvme";
    datapath.handle_cache_capacity = config.handle_cache_capacity;
    datapath.prp_cache_capacity = config.prp_cache_capacity;
    datapath.handle_cache_l2_capacity = config.handle_cache_l2_capacity;
    datapath.max_in_flight_operations = config.max_in_flight_operations;
    datapath.max_batch_entries = config.max_batch_entries;
    datapath.io_granularity = config.io_granularity;
    storage.datapaths.push_back(std::move(datapath));

    BackendSpec backend;
    backend.id = "legacy-backend";
    backend.contract = striped ? "striped-local-nvme" : "ext4-local-nvme";
    backend.resolver = "legacy-resolver";
    backend.datapath = "legacy-datapath";
    backend.resource = "legacy-nvme-resource";
    backend.stripe_unit = striped ? config.stripe_unit : 0;
    storage.backends.push_back(std::move(backend));
    return storage;
}

Result<ParsedConfig> parse_legacy(const YAML::Node& root) {
    ParsedConfig config;
    config.syntax = ConfigSyntax::Legacy;
    Status status = parse_common(root, config);
    if (!status.ok()) return failure<ParsedConfig>(status);

    if (const YAML::Node storage = root["storage"]) {
        if (const YAML::Node backend = storage["backend"])
            config.storage_backend = backend.as<std::string>();
        if (const YAML::Node stripe = storage["default_stripe_unit"])
            config.default_stripe_unit = stripe.as<std::uint64_t>();
    }
    if (config.storage_backend == "rdma") {
        return failure<ParsedConfig>(Status(
            StatusCode::UNSUPPORTED, "storage.backend=rdma is not yet implemented"));
    }
    if (config.storage_backend != "local-nvme") {
        return failure<ParsedConfig>(
            invalid("legacy storage.backend must be local-nvme"));
    }

    if (const YAML::Node service = root["nvme_service"]) {
        if (const YAML::Node endpoint = service["endpoint"])
            config.nvme_service_endpoint = endpoint.as<std::string>();
    }

    bool stripe_unit_explicit = false;
    if (const YAML::Node nvme = root["nvme"]) {
        if (const YAML::Node selection = nvme["selection"]) {
            config.nvme_selection_text = selection.as<std::string>();
            auto parsed_selection = parse_selection(config.nvme_selection_text);
            if (!parsed_selection.ok()) return failure<ParsedConfig>(parsed_selection.status());
            config.nvme_selection = parsed_selection.value();
        }
        status = parse_device_ids(nvme["device_ids"], "nvme.device_ids",
                                  config.nvme_device_ids);
        if (!status.ok()) return failure<ParsedConfig>(status);
        status = read_nonnegative_i32(nvme, "queues_per_controller", "nvme",
                                      config.queues_per_controller, false);
        if (!status.ok()) return failure<ParsedConfig>(status);
        if (const YAML::Node stripe = nvme["stripe_unit"]) {
            config.stripe_unit = stripe.as<std::uint64_t>();
            stripe_unit_explicit = true;
        }
    }
    if (!stripe_unit_explicit && config.default_stripe_unit != 0)
        config.stripe_unit = config.default_stripe_unit;

    AllocationSpec allocation;
    allocation.selection = config.nvme_selection;
    allocation.device_ids = config.nvme_device_ids;
    allocation.queues_per_controller = config.queues_per_controller;
    status = validate_selection(allocation, "nvme");
    if (!status.ok()) return failure<ParsedConfig>(status);

    if (const YAML::Node local_nvme = root["local_nvme"]) {
        if (const YAML::Node value = local_nvme["handle_cache_capacity"])
            config.handle_cache_capacity = value.as<std::uint32_t>();
        if (const YAML::Node value = local_nvme["prp_cache_capacity"])
            config.prp_cache_capacity = value.as<std::uint32_t>();
        if (const YAML::Node value = local_nvme["handle_cache_l2_capacity"])
            config.handle_cache_l2_capacity = value.as<std::uint32_t>();
        if (const YAML::Node value = local_nvme["max_in_flight_operations"])
            config.max_in_flight_operations = value.as<std::uint64_t>();
        if (const YAML::Node value = local_nvme["max_batch_entries"])
            config.max_batch_entries = value.as<std::uint64_t>();
        if (const YAML::Node value = local_nvme["num_user_queues"])
            config.num_user_queues = value.as<std::uint32_t>();
        if (const YAML::Node value = local_nvme["io_granularity"])
            config.io_granularity = value.as<std::uint64_t>();
    }
    if (const YAML::Node value = root["local_nvme_config"])
        config.local_nvme_config = value.as<std::string>();

    config.canonical_storage = adapt_legacy(config);
    status = validate_canonical(config.canonical_storage);
    if (!status.ok()) return failure<ParsedConfig>(status);
    return Result<ParsedConfig>::Success(std::move(config));
}

bool has_key(const YAML::Node& node, const char* key) {
    return node && node.IsMap() && static_cast<bool>(node[key]);
}

std::uint32_t env_or_zero(const char* name) {
    const char* value = std::getenv(name);
    return value ? static_cast<std::uint32_t>(std::atoi(value)) : 0;
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

Result<std::vector<DeviceSpec>> derive_local_nvme_devices(
    const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& exception) {
        return Result<std::vector<DeviceSpec>>::Failure(
            invalid("local_nvme_config yaml parse error: " +
                    std::string(exception.what())));
    }

    std::vector<DeviceSpec> devices;
    std::uint32_t minor = 0;
    if (const YAML::Node nvmes = root["nvmes"]) {
        for (const auto& nvme : nvmes) {
            std::uint32_t namespace_id = 1;
            if (const YAML::Node value = nvme["namespace_id"])
                namespace_id = value.as<std::uint32_t>();
            if (const YAML::Node allowed = nvme["allowed_gpus"]) {
                for (const auto& gpu : allowed) {
                    DeviceSpec device;
                    device.cuda_device = gpu.as<std::uint32_t>();
                    device.snvme_dev = "/dev/ssnvme" + std::to_string(minor);
                    device.namespace_id = namespace_id;
                    devices.push_back(std::move(device));
                }
            }
            ++minor;
        }
    }
    return Result<std::vector<DeviceSpec>>::Success(std::move(devices));
}

EffectiveCacheConfig resolve_cache_config(
    const ParsedConfig& parsed, const ProgrammaticOverrides& overrides) {
    EffectiveCacheConfig effective;
    if (overrides.handle_cache_capacity > 0) {
        effective.handle_cache_capacity = overrides.handle_cache_capacity;
    } else if (parsed.handle_cache_capacity > 0) {
        effective.handle_cache_capacity = parsed.handle_cache_capacity;
    } else {
        effective.handle_cache_capacity = env_or_zero("TUTTI_HANDLE_CACHE_CAP");
    }
    if (overrides.prp_cache_capacity > 0) {
        effective.prp_cache_capacity = overrides.prp_cache_capacity;
    } else if (parsed.prp_cache_capacity > 0) {
        effective.prp_cache_capacity = parsed.prp_cache_capacity;
    } else {
        effective.prp_cache_capacity = env_or_zero("TUTTI_PRP_CACHE_CAP");
    }
    if (overrides.handle_cache_l2_capacity > 0) {
        effective.handle_cache_l2_capacity = overrides.handle_cache_l2_capacity;
    } else if (parsed.handle_cache_l2_capacity > 0) {
        effective.handle_cache_l2_capacity = parsed.handle_cache_l2_capacity;
    }
    return effective;
}

} // namespace tutti::config

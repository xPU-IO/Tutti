#include "tutti/config/storage/resource/nvme_parse_internal.h"

#include <cstdlib>
#include <utility>

namespace tutti::config {

namespace {

std::uint32_t env_or_zero(const char* name) {
    const char* value = std::getenv(name);
    return value ? static_cast<std::uint32_t>(std::atoi(value)) : 0;
}

} // namespace

namespace detail {

Status parse_legacy_nvme(const YAML::Node& root, ParsedConfig& config) {
    if (const YAML::Node storage = root["storage"]) {
        if (const YAML::Node backend = storage["backend"])
            config.storage_backend = backend.as<std::string>();
        if (const YAML::Node stripe = storage["default_stripe_unit"])
            config.default_stripe_unit = stripe.as<std::uint64_t>();
    }
    if (config.storage_backend == "rdma") {
        return Status(StatusCode::UNSUPPORTED,
                      "storage.backend=rdma is not yet implemented");
    }
    if (config.storage_backend != "local-nvme") {
        return invalid("legacy storage.backend must be local-nvme");
    }

    if (const YAML::Node service = root["nvme_service"]) {
        if (const YAML::Node endpoint = service["endpoint"])
            config.nvme_service_endpoint = endpoint.as<std::string>();
    }

    bool stripe_unit_explicit = false;
    if (const YAML::Node nvme = root["nvme"]) {
        if (const YAML::Node selection = nvme["selection"]) {
            config.nvme_selection_text = selection.as<std::string>();
            auto parsed_selection =
                parse_nvme_selection(config.nvme_selection_text);
            if (!parsed_selection.ok()) return parsed_selection.status();
            config.nvme_selection = parsed_selection.value();
        }
        Status status = parse_nvme_device_ids(
            nvme["device_ids"], "nvme.device_ids", config.nvme_device_ids);
        if (!status.ok()) return status;
        status = read_nonnegative_i32(
            nvme, "queues_per_controller", "nvme",
            config.queues_per_controller, false);
        if (!status.ok()) return status;
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
    Status status = validate_nvme_selection(allocation, "nvme");
    if (!status.ok()) return status;

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
    return Status::Ok();
}

CanonicalStorageConfig adapt_legacy_nvme(const ParsedConfig& config) {
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

void apply_nvme_compatibility(const ResourceSpec& resource,
                              const DataPathSpec& datapath,
                              const BackendSpec& backend,
                              ParsedConfig& config) {
    config.nvme_service_endpoint = resource.provider.endpoint;
    config.nvme_selection = resource.allocation.selection;
    config.nvme_selection_text =
        nvme_selection_name(resource.allocation.selection);
    config.nvme_device_ids = resource.allocation.device_ids;
    config.queues_per_controller = resource.allocation.queues_per_controller;
    config.storage_backend = datapath.type;
    config.handle_cache_capacity = datapath.handle_cache_capacity;
    config.prp_cache_capacity = datapath.prp_cache_capacity;
    config.handle_cache_l2_capacity = datapath.handle_cache_l2_capacity;
    config.max_in_flight_operations = datapath.max_in_flight_operations;
    config.max_batch_entries = datapath.max_batch_entries;
    config.io_granularity = datapath.io_granularity;
    if (backend.stripe_unit != 0) config.stripe_unit = backend.stripe_unit;
}

} // namespace detail

Result<std::vector<DeviceSpec>> derive_local_nvme_devices(
    const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& exception) {
        return Result<std::vector<DeviceSpec>>::Failure(
            detail::invalid("local_nvme_config yaml parse error: " +
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
                    device.snvme_dev =
                        "/dev/ssnvme" + std::to_string(minor);
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
        effective.handle_cache_capacity =
            env_or_zero("TUTTI_HANDLE_CACHE_CAP");
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

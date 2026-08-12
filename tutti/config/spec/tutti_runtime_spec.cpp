#include <tutti/config/tutti_runtime_spec.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "tutti/config/spec/spec_internal.h"

namespace tutti::config {
namespace {

constexpr std::size_t kMaximumIdLength = 255;

std::string upper_ascii(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(
            std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

template <typename Spec>
Status validate_identity(const Spec& spec, const std::string& path,
                         const char* kind) {
    if (spec.id.empty()) {
        return detail::invalid_spec(path + ".id must not be empty");
    }
    if (spec.id.size() > kMaximumIdLength) {
        return detail::invalid_spec(path + ".id exceeds 255 bytes");
    }
    if (spec.type.empty()) {
        return detail::invalid_spec(path + ".type must not be empty");
    }
    (void)kind;
    return Status::Ok();
}

Status validate_backend_identity(const BackendSpec& spec,
                                 const std::string& path) {
    if (spec.id.empty()) {
        return detail::invalid_spec(path + ".id must not be empty");
    }
    if (spec.id.size() > kMaximumIdLength) {
        return detail::invalid_spec(path + ".id exceeds 255 bytes");
    }
    if (spec.contract.empty()) {
        return detail::invalid_spec(path + ".contract must not be empty");
    }
    if (spec.resolver.empty()) {
        return detail::invalid_spec(path + ".resolver must not be empty");
    }
    if (spec.datapath.empty()) {
        return detail::invalid_spec(path + ".datapath must not be empty");
    }
    if (spec.resource.empty()) {
        return detail::invalid_spec(path + ".resource must not be empty");
    }
    return Status::Ok();
}

Status validate_resource(const ResourceSpec& spec, const std::string& path) {
    Status status = validate_identity(spec, path, "resource");
    if (!status.ok()) return status;
    if (spec.type == "nvme") {
        return detail::validate_nvme_resource(spec, path);
    }
    if (spec.type == "memory") {
        return detail::validate_memory_resource(spec, path);
    }
    return detail::invalid_spec(path + ".type is unknown: " + spec.type);
}

Status validate_resolver(const ResolverSpec& spec, const std::string& path) {
    Status status = validate_identity(spec, path, "resolver");
    if (!status.ok()) return status;
    if (spec.scheme.empty()) {
        return detail::invalid_spec(path + ".scheme must not be empty");
    }
    if (spec.type == "local-file") {
        return detail::validate_local_file_resolver(spec, path);
    }
    if (spec.type == "striped-file") {
        return detail::validate_striped_file_resolver(spec, path);
    }
    if (spec.type == "memfs") {
        return detail::validate_memfs_resolver(spec, path);
    }
    return detail::invalid_spec(path + ".type is unknown: " + spec.type);
}

Status validate_datapath(const DataPathSpec& spec, const std::string& path) {
    Status status = validate_identity(spec, path, "datapath");
    if (!status.ok()) return status;
    if (spec.type == "local-nvme") {
        return detail::validate_local_nvme_datapath(spec, path);
    }
    if (spec.type == "striped-local-nvme") {
        return detail::validate_striped_local_nvme_datapath(spec, path);
    }
    if (spec.type == "memfs") {
        return detail::validate_memfs_datapath(spec, path);
    }
    return detail::invalid_spec(path + ".type is unknown: " + spec.type);
}

template <typename Spec>
Status index_specs(const std::vector<Spec>& specs, const char* group,
                   std::unordered_map<std::string, const Spec*>& index) {
    for (std::size_t position = 0; position < specs.size(); ++position) {
        const Spec& spec = specs[position];
        if (!index.emplace(spec.id, &spec).second) {
            return detail::invalid_spec(
                "storage." + std::string(group) + "[" +
                std::to_string(position) + "].id duplicates " + spec.id);
        }
    }
    return Status::Ok();
}

bool valid_scheme(const std::string& scheme) {
    if (scheme.empty() || scheme.front() < 'a' || scheme.front() > 'z') {
        return false;
    }
    return std::all_of(scheme.begin() + 1, scheme.end(), [](char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') ||
               ch == '+' || ch == '-' || ch == '.';
    });
}

std::string quote(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
        case '\\': output << "\\\\"; break;
        case '"': output << "\\\""; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (ch < 0x20 || ch == 0x7f) {
                constexpr char digits[] = "0123456789abcdef";
                output << "\\x" << digits[ch >> 4] << digits[ch & 0xf];
            } else {
                output << static_cast<char>(ch);
            }
        }
    }
    output << '"';
    return output.str();
}

const char* selection_name(NvmeSelection selection) {
    switch (selection) {
    case NvmeSelection::Allowed: return "allowed";
    case NvmeSelection::Explicit: return "explicit";
    case NvmeSelection::Striped: return "striped";
    }
    return "unknown";
}

template <typename T>
void line(std::ostringstream& output, const std::string& path,
          const T& value) {
    output << path << " = " << value << '\n';
}

void string_line(std::ostringstream& output, const std::string& path,
                 std::string_view value) {
    output << path << " = " << quote(value) << '\n';
}

void device_ids_line(std::ostringstream& output, const std::string& path,
                     const std::vector<std::int32_t>& ids) {
    output << path << " = [";
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) output << ", ";
        output << ids[index];
    }
    output << "]\n";
}

void debug_datapath_tuning(std::ostringstream& output,
                           const std::string& path,
                           const NvmeDataPathTuning& config) {
    line(output, path + ".handle_cache_capacity",
         config.handle_cache_capacity);
    line(output, path + ".prp_cache_capacity", config.prp_cache_capacity);
    line(output, path + ".handle_cache_l2_capacity",
         config.handle_cache_l2_capacity);
    line(output, path + ".threads_per_block", config.threads_per_block);
    line(output, path + ".max_in_flight_operations",
         config.max_in_flight_operations);
    line(output, path + ".max_batch_entries", config.max_batch_entries);
    line(output, path + ".io_granularity", config.io_granularity);
}

} // namespace

namespace detail {

Status invalid_spec(std::string message) {
    return Status(StatusCode::INVALID_ARGUMENT, std::move(message));
}

const SpecContract* find_spec_contract(std::string_view name) {
    static constexpr std::array<SpecContract, 3> contracts{{
        {"ext4-local-nvme", "local-file", "file", "local-nvme", "nvme",
         1, 1},
        {"striped-local-nvme", "striped-file", "striped",
         "striped-local-nvme", "nvme", 2,
         std::numeric_limits<std::size_t>::max()},
        {"memfs", "memfs", "memfs", "memfs", "memory", 1, 1},
    }};
    const auto found = std::find_if(
        contracts.begin(), contracts.end(),
        [&](const SpecContract& contract) { return contract.name == name; });
    return found == contracts.end() ? nullptr : &*found;
}

} // namespace detail

Status TuttiRuntimeSpec::validate() const {
    // 1. Top-level fields.
    const std::string profile = upper_ascii(accelerator.profile);
    if (profile != "HOST" && profile != "CUDA" &&
        profile != "MUSA" && profile != "MACA") {
        return detail::invalid_spec("accelerator.profile is unknown: " +
                                    accelerator.profile);
    }
    if (runtime.accel_id < -1) {
        return detail::invalid_spec(
            "runtime.accel_id must be -1 or non-negative");
    }
    if ((profile == "HOST") != (runtime.accel_id == -1)) {
        return detail::invalid_spec(
            "accelerator.profile and runtime.accel_id are inconsistent");
    }

    // 2. Individual declarations and type-specific fields.
    for (std::size_t index = 0; index < storage.resources.size(); ++index) {
        Status status = validate_resource(
            storage.resources[index],
            "storage.resources[" + std::to_string(index) + "]");
        if (!status.ok()) return status;
    }
    for (std::size_t index = 0; index < storage.resolvers.size(); ++index) {
        Status status = validate_resolver(
            storage.resolvers[index],
            "storage.resolvers[" + std::to_string(index) + "]");
        if (!status.ok()) return status;
    }
    for (std::size_t index = 0; index < storage.datapaths.size(); ++index) {
        Status status = validate_datapath(
            storage.datapaths[index],
            "storage.datapaths[" + std::to_string(index) + "]");
        if (!status.ok()) return status;
    }
    for (std::size_t index = 0; index < storage.backends.size(); ++index) {
        Status status = validate_backend_identity(
            storage.backends[index],
            "storage.backends[" + std::to_string(index) + "]");
        if (!status.ok()) return status;
    }

    // 3. Per-group ID indexes.
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

    // 4. References.
    for (std::size_t index = 0; index < storage.backends.size(); ++index) {
        const BackendSpec& backend = storage.backends[index];
        const std::string path =
            "storage.backends[" + std::to_string(index) + "]";
        if (resources.count(backend.resource) == 0) {
            return detail::invalid_spec(path + ".resource references unknown id " +
                                        backend.resource);
        }
        if (resolvers.count(backend.resolver) == 0) {
            return detail::invalid_spec(path + ".resolver references unknown id " +
                                        backend.resolver);
        }
        if (datapaths.count(backend.datapath) == 0) {
            return detail::invalid_spec(path + ".datapath references unknown id " +
                                        backend.datapath);
        }
    }

    // 5. Static backend contracts.
    for (std::size_t index = 0; index < storage.backends.size(); ++index) {
        const BackendSpec& backend = storage.backends[index];
        const std::string path =
            "storage.backends[" + std::to_string(index) + "]";
        const detail::SpecContract* contract =
            detail::find_spec_contract(backend.contract);
        if (contract == nullptr) {
            return detail::invalid_spec(path + ".contract is unknown: " +
                                        backend.contract);
        }
        const ResourceSpec& resource = *resources.at(backend.resource);
        const ResolverSpec& resolver = *resolvers.at(backend.resolver);
        const DataPathSpec& datapath = *datapaths.at(backend.datapath);
        if (resource.type != contract->resource_type ||
            resolver.type != contract->resolver_type ||
            resolver.scheme != contract->resolver_scheme ||
            datapath.type != contract->datapath_type) {
            return detail::invalid_spec(path +
                                        " types do not match contract " +
                                        backend.contract);
        }
        if (backend.contract == "ext4-local-nvme") {
            status = detail::validate_ext4_local_nvme_backend(
                resource, backend, *contract, path);
        } else if (backend.contract == "striped-local-nvme") {
            status = detail::validate_striped_local_nvme_backend(
                resource, backend, *contract, path);
        } else {
            status = detail::validate_memfs_backend(
                resource, backend, *contract, path);
        }
        if (!status.ok()) return status;
    }

    // 6. Resolver routing topology.
    std::unordered_set<std::string> schemes;
    for (std::size_t index = 0; index < storage.resolvers.size(); ++index) {
        const ResolverSpec& resolver = storage.resolvers[index];
        const std::string path =
            "storage.resolvers[" + std::to_string(index) + "].scheme";
        if (!valid_scheme(resolver.scheme)) {
            return detail::invalid_spec(path + " is invalid: " +
                                        resolver.scheme);
        }
        if (!schemes.emplace(resolver.scheme).second) {
            return detail::invalid_spec(path + " duplicates route " +
                                        resolver.scheme);
        }
    }

    // 7. A Resource cannot be consumed by independent DataPaths.
    std::unordered_map<std::string, std::string> resource_datapaths;
    for (std::size_t index = 0; index < storage.backends.size(); ++index) {
        const BackendSpec& backend = storage.backends[index];
        const auto inserted = resource_datapaths.emplace(
            backend.resource, backend.datapath);
        if (!inserted.second && inserted.first->second != backend.datapath) {
            return detail::invalid_spec(
                "storage.backends[" + std::to_string(index) +
                "].resource is consumed by a different datapath");
        }
    }

    // 8. Every declaration must be reachable from a backend.
    std::unordered_set<std::string> reachable_resources;
    std::unordered_set<std::string> reachable_resolvers;
    std::unordered_set<std::string> reachable_datapaths;
    for (const BackendSpec& backend : storage.backends) {
        reachable_resources.emplace(backend.resource);
        reachable_resolvers.emplace(backend.resolver);
        reachable_datapaths.emplace(backend.datapath);
    }
    if (reachable_resources.size() != storage.resources.size() ||
        reachable_resolvers.size() != storage.resolvers.size() ||
        reachable_datapaths.size() != storage.datapaths.size()) {
        return detail::invalid_spec(
            "all storage declarations must be reachable from a backend");
    }

    // 9. Current TuttiRuntime product constraint.
    if (storage.backends.size() != 1) {
        return detail::invalid_spec(
            "storage.backends must contain exactly one backend");
    }
    return Status::Ok();
}

Result<std::string> TuttiRuntimeSpec::to_debug_string() const {
    Status status = validate();
    if (!status.ok()) return Result<std::string>::Failure(std::move(status));

    std::ostringstream output;
    string_line(output, "accelerator.profile", accelerator.profile);
    line(output, "runtime.accel_id", runtime.accel_id);

    for (std::size_t index = 0; index < storage.resources.size(); ++index) {
        const ResourceSpec& resource = storage.resources[index];
        const std::string path =
            "storage.resources[" + std::to_string(index) + "]";
        string_line(output, path + ".id", resource.id);
        string_line(output, path + ".type", resource.type);
        if (const auto* config =
                std::get_if<NvmeResourceConfig>(&resource.config)) {
            string_line(output, path + ".provider.type", config->provider.type);
            string_line(output, path + ".provider.endpoint",
                        config->provider.endpoint);
            string_line(output, path + ".allocation.selection",
                        selection_name(config->allocation.selection));
            device_ids_line(output, path + ".allocation.device_ids",
                            config->allocation.device_ids);
            line(output, path + ".allocation.queues_per_controller",
                 config->allocation.queues_per_controller);
        } else {
            line(output, path + ".config.capacity_bytes",
                 std::get<MemoryResourceConfig>(resource.config).capacity_bytes);
        }
    }
    for (std::size_t index = 0; index < storage.resolvers.size(); ++index) {
        const ResolverSpec& resolver = storage.resolvers[index];
        const std::string path =
            "storage.resolvers[" + std::to_string(index) + "]";
        string_line(output, path + ".id", resolver.id);
        string_line(output, path + ".type", resolver.type);
        string_line(output, path + ".scheme", resolver.scheme);
    }
    for (std::size_t index = 0; index < storage.datapaths.size(); ++index) {
        const DataPathSpec& datapath = storage.datapaths[index];
        const std::string path =
            "storage.datapaths[" + std::to_string(index) + "]";
        string_line(output, path + ".id", datapath.id);
        string_line(output, path + ".type", datapath.type);
        if (const auto* config =
                std::get_if<LocalNvmeDataPathConfig>(&datapath.config)) {
            debug_datapath_tuning(output, path + ".config", *config);
        } else if (const auto* config =
                       std::get_if<StripedLocalNvmeDataPathConfig>(
                           &datapath.config)) {
            debug_datapath_tuning(output, path + ".config", *config);
        }
    }
    for (std::size_t index = 0; index < storage.backends.size(); ++index) {
        const BackendSpec& backend = storage.backends[index];
        const std::string path =
            "storage.backends[" + std::to_string(index) + "]";
        string_line(output, path + ".id", backend.id);
        string_line(output, path + ".contract", backend.contract);
        string_line(output, path + ".resolver", backend.resolver);
        string_line(output, path + ".datapath", backend.datapath);
        string_line(output, path + ".resource", backend.resource);
        if (const auto* config =
                std::get_if<StripedLocalNvmeBackendConfig>(&backend.config)) {
            line(output, path + ".config.stripe_unit", config->stripe_unit);
        }
    }
    return Result<std::string>::Success(output.str());
}

} // namespace tutti::config

#include <tutti/config/tutti_config.h>
#include <tutti/bindings/ext4_local_nvme/binding.h>
#include <tutti/bindings/striped_local_nvme/binding.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

int checks = 0;
int failures = 0;

void check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

class TempConfig {
public:
    explicit TempConfig(const std::string& contents) {
        char path[] = "/tmp/tutti_storage_config_XXXXXX";
        const int descriptor = ::mkstemp(path);
        if (descriptor < 0) return;
        path_ = path;
        const ssize_t written = ::write(descriptor, contents.data(), contents.size());
        valid_ = written == static_cast<ssize_t>(contents.size());
        ::close(descriptor);
    }

    ~TempConfig() {
        if (!path_.empty()) ::unlink(path_.c_str());
    }

    const std::string& path() const { return path_; }
    bool valid() const { return valid_; }

private:
    std::string path_;
    bool valid_ = false;
};

std::string local_yaml(
    const std::string& selection = "explicit",
    const std::string& device_ids = "[0]",
    const std::string& queues = "4",
    const std::string& resource_extra = {},
    const std::string& resolver_extra = {},
    const std::string& datapath_extra = {},
    const std::string& backend_entries = {}) {
    const std::string default_backend = R"(    - id: model-storage
      contract: ext4-local-nvme
      resolver: file-resolver-0
      datapath: local-nvme-datapath-0
      resource: nvme-local-0
      config: {}
)";
    return "accelerator: {profile: CUDA}\n"
           "runtime: {accel_id: 0}\n"
           "storage:\n"
           "  resources:\n"
           "    - id: nvme-local-0\n"
           "      type: nvme\n"
           "      provider: {type: nvme-service, endpoint: '127.0.0.1:50051'}\n"
           "      allocation:\n"
           "        selection: " + selection + "\n"
           "        device_ids: " + device_ids + "\n"
           "        queues_per_controller: " + queues + "\n" +
           resource_extra +
           "  resolvers:\n"
           "    - id: file-resolver-0\n"
           "      type: local-file\n"
           "      scheme: file\n"
           "      config: {}\n" +
           resolver_extra +
           "  datapaths:\n"
           "    - id: local-nvme-datapath-0\n"
           "      type: local-nvme\n"
           "      config:\n"
           "        handle_cache_capacity: 11\n"
           "        prp_cache_capacity: 12\n"
           "        handle_cache_l2_capacity: 13\n"
           "        max_in_flight_operations: 14\n"
           "        max_batch_entries: 15\n"
           "        io_granularity: 4096\n" +
           datapath_extra +
           "  backends:\n" +
           (backend_entries.empty() ? default_backend : backend_entries);
}

std::string striped_yaml(const std::string& stripe_unit = "65536") {
    return "accelerator: {profile: CUDA}\n"
           "runtime: {accel_id: 0}\n"
           "storage:\n"
           "  resources:\n"
           "    - id: nvme-striped-0\n"
           "      type: nvme\n"
           "      provider: {type: nvme-service, endpoint: '127.0.0.1:50051'}\n"
           "      allocation:\n"
           "        selection: striped\n"
           "        device_ids: [1, 0]\n"
           "        queues_per_controller: 4\n"
           "  resolvers:\n"
           "    - id: striped-resolver-0\n"
           "      type: striped-file\n"
           "      scheme: striped\n"
           "      config: {}\n"
           "  datapaths:\n"
           "    - id: striped-datapath-0\n"
           "      type: striped-local-nvme\n"
           "      config: {max_in_flight_operations: 32, max_batch_entries: 64}\n"
           "  backends:\n"
           "    - id: striped-storage\n"
           "      contract: striped-local-nvme\n"
           "      resolver: striped-resolver-0\n"
           "      datapath: striped-datapath-0\n"
           "      resource: nvme-striped-0\n"
           "      config: {stripe_unit: " + stripe_unit + "}\n";
}

std::string replace_once(std::string value, const std::string& from,
                         const std::string& to) {
    const std::size_t position = value.find(from);
    if (position != std::string::npos) value.replace(position, from.size(), to);
    return value;
}

void expect_failure(const std::string& yaml, const std::string& label,
                    tutti::StatusCode expected = tutti::StatusCode::INVALID_ARGUMENT) {
    TempConfig config(yaml);
    check(config.valid(), label + ": temp config written");
    const auto parsed = tutti::config::parse_tutti_config(config.path());
    check(!parsed.ok(), label + ": rejected");
    if (!parsed.ok()) {
        check(parsed.status().code() == expected, label + ": status code");
    }
}

} // namespace

int main() {
    using namespace tutti::config;

    {
        TempConfig config(local_yaml());
        auto parsed = parse_tutti_config(config.path());
        check(parsed.ok(), "canonical local accepted");
        if (parsed.ok()) {
            const ParsedConfig& value = parsed.value();
            check(value.syntax == ConfigSyntax::Canonical, "canonical syntax tagged");
            check(value.canonical_storage.resources.size() == 1,
                  "canonical local resource count");
            check(value.canonical_storage.backends.front().contract ==
                      "ext4-local-nvme",
                  "canonical local contract");
            check(value.nvme_service_endpoint == "127.0.0.1:50051",
                  "canonical endpoint compatibility");
            check(value.nvme_selection == NvmeSelection::Explicit &&
                      value.nvme_device_ids == std::vector<std::int32_t>{0},
                  "canonical allocation compatibility");
            check(value.handle_cache_capacity == 11 &&
                      value.max_batch_entries == 15,
                  "canonical datapath compatibility");
        }
    }

    {
        TempConfig config(striped_yaml());
        auto parsed = parse_tutti_config(config.path());
        check(parsed.ok(), "canonical striped accepted");
        if (parsed.ok()) {
            const auto& value = parsed.value();
            check(value.nvme_device_ids == std::vector<std::int32_t>({1, 0}),
                  "striped device order preserved");
            check(value.stripe_unit == 65536, "striped unit adapted");
            check(value.canonical_storage.backends.front().contract ==
                      "striped-local-nvme",
                  "striped contract preserved");
        }
    }

    {
        TempConfig config("accelerator: {profile: HOST}\nruntime: {accel_id: -1}\n");
        auto parsed = parse_tutti_config(config.path());
        check(parsed.ok(), "HOST without storage accepted");
        if (parsed.ok()) {
            check(parsed.value().syntax == ConfigSyntax::Canonical,
                  "HOST no-storage canonical syntax");
            check(!parsed.value().canonical_storage.present,
                  "HOST no-storage model empty");
        }
    }

    {
        TempConfig config(
            "accelerator: {profile: CUDA}\n"
            "runtime: {accel_id: 0}\n"
            "nvme_service: {endpoint: legacy}\n"
            "nvme: {selection: striped, device_ids: [0, 1], queues_per_controller: 4, stripe_unit: 65536}\n"
            "local_nvme: {handle_cache_capacity: 7}\n");
        auto parsed = parse_tutti_config(config.path());
        check(parsed.ok(), "legacy config accepted");
        if (parsed.ok()) {
            check(parsed.value().syntax == ConfigSyntax::Legacy,
                  "legacy syntax tagged");
            check(parsed.value().canonical_storage.backends.front().contract ==
                      "striped-local-nvme",
                  "legacy converted once to canonical contract");
            check(parsed.value().canonical_storage.resources.front().provider.endpoint ==
                      "legacy",
                  "legacy provider adapted");
        }
    }

    const std::string duplicate_resource =
        "    - id: nvme-local-0\n"
        "      type: nvme\n"
        "      provider: {type: nvme-service, endpoint: duplicate}\n"
        "      allocation: {selection: explicit, device_ids: [1], queues_per_controller: 4}\n";
    const std::string duplicate_resolver =
        "    - id: file-resolver-1\n"
        "      type: local-file\n"
        "      scheme: file\n"
        "      config: {}\n";
    const std::string unused_resolver =
        "    - id: unused-resolver\n"
        "      type: local-file\n"
        "      scheme: unused\n"
        "      config: {}\n";
    const std::string default_backend =
        "    - id: model-storage\n"
        "      contract: ext4-local-nvme\n"
        "      resolver: file-resolver-0\n"
        "      datapath: local-nvme-datapath-0\n"
        "      resource: nvme-local-0\n"
        "      config: {}\n";

    expect_failure(
        replace_once(local_yaml(), "    - id: nvme-local-0\n", "    - id: ''\n"),
        "missing id");
    expect_failure(local_yaml("explicit", "[0]", "4", duplicate_resource),
                   "duplicate id");
    expect_failure(
        replace_once(local_yaml(), "      resource: nvme-local-0\n",
                     "      resource: missing-resource\n"),
        "dangling reference");
    expect_failure(local_yaml("explicit", "[0]", "4", {}, duplicate_resolver),
                   "duplicate scheme");
    expect_failure(local_yaml("explicit", "[0]", "4", {}, {}, {},
                              default_backend + replace_once(
                                  default_backend, "id: model-storage",
                                  "id: model-storage-2")),
                   "duplicate contract DataPath key");
    expect_failure(local_yaml("explicit", "[0]", "4", {}, {}, {}, "[]\n"),
                   "zero backend");
    expect_failure(local_yaml("explicit", "[0]", "4", {}, unused_resolver),
                   "unreferenced declaration");
    expect_failure(replace_once(local_yaml(), "contract: ext4-local-nvme",
                                "contract: unknown-contract"),
                   "unknown contract");
    expect_failure(replace_once(local_yaml(), "type: local-file",
                                "type: unknown-resolver"),
                   "unknown type");
    expect_failure(replace_once(local_yaml(), "scheme: file", "scheme: 'Bad Scheme'"),
                   "invalid scheme");
    expect_failure(
        local_yaml("explicit", "[0]", "4",
                   "      config: {capacity_bytes: 4096}\n"),
        "nvme resource config must be empty");
    expect_failure(
        replace_once(local_yaml(),
                     "      type: local-file\n"
                     "      scheme: file\n"
                     "      config: {}\n",
                     "      type: local-file\n"
                     "      scheme: file\n"
                     "      config: {unknown: true}\n"),
        "local file resolver config must be empty");
    expect_failure(
        replace_once(striped_yaml(),
                     "      type: striped-file\n"
                     "      scheme: striped\n"
                     "      config: {}\n",
                     "      type: striped-file\n"
                     "      scheme: striped\n"
                     "      config: {unknown: true}\n"),
        "striped file resolver config must be empty");
    expect_failure(
        replace_once(local_yaml(), "        handle_cache_capacity: 11\n",
                     "        unknown_tuning: 11\n"),
        "local NVMe datapath unknown config");
    expect_failure(
        replace_once(
            striped_yaml(),
            "      config: {max_in_flight_operations: 32, max_batch_entries: 64}\n",
            "      config: {max_in_flight_operations: 32, stripe_unit: 65536}\n"),
        "striped NVMe datapath unknown config");
    expect_failure(
        replace_once(local_yaml(),
                     "      resource: nvme-local-0\n"
                     "      config: {}\n",
                     "      resource: nvme-local-0\n"
                     "      config: {stripe_unit: 4096}\n"),
        "ext4 backend disallows stripe unit");
    expect_failure(local_yaml("allowed", "[0]"), "allowed with device id");
    expect_failure(local_yaml("explicit", "[]"), "explicit without device id");
    expect_failure(local_yaml("explicit", "[0, 1]"), "explicit with many ids");
    expect_failure(replace_once(striped_yaml(), "device_ids: [1, 0]",
                                "device_ids: [0]"),
                   "striped with one id");
    expect_failure(replace_once(striped_yaml(), "device_ids: [1, 0]",
                                "device_ids: [0, 0]"),
                   "striped duplicate ids");
    expect_failure(local_yaml("explicit", "[0]", "-1"), "negative queues");
    expect_failure(striped_yaml("0"), "zero stripe unit");
    expect_failure(striped_yaml("65537"), "unaligned stripe unit");
    expect_failure(local_yaml() + "nvme_service: {endpoint: legacy}\n",
                   "canonical legacy mixing");
    expect_failure(local_yaml() + "unexpected: true\n", "unknown canonical field");
    expect_failure(
        replace_once(local_yaml(), "type: local-nvme", "type: memfs"),
        "contract type mismatch");

    const std::string memfs =
        "accelerator: {profile: HOST}\n"
        "runtime: {accel_id: -1}\n"
        "storage:\n"
        "  resources:\n"
        "    - id: memory-0\n"
        "      type: memory\n"
        "      config: {capacity_bytes: 4096}\n"
        "  resolvers:\n"
        "    - {id: memfs-resolver, type: memfs, scheme: memfs, config: {}}\n"
        "  datapaths:\n"
        "    - {id: memfs-datapath, type: memfs, config: {}}\n"
        "  backends:\n"
        "    - {id: memfs-backend, contract: memfs, resolver: memfs-resolver, datapath: memfs-datapath, resource: memory-0, config: {}}\n";
    expect_failure(memfs, "memfs factory unsupported", tutti::StatusCode::UNSUPPORTED);
    expect_failure(
        replace_once(memfs, "capacity_bytes: 4096", "capacity_bytes: 0"),
        "memfs zero capacity");
    expect_failure(
        replace_once(
            memfs, "      type: memory\n",
            "      type: memory\n"
            "      provider: {type: nvme-service, endpoint: local}\n"),
        "memfs resource with provider");
    expect_failure(
        replace_once(
            memfs, "      type: memory\n",
            "      type: memory\n"
            "      allocation: {selection: allowed, queues_per_controller: 1}\n"),
        "memfs resource with allocation");
    expect_failure(
        replace_once(memfs,
                     "    - {id: memfs-resolver, type: memfs, scheme: memfs, config: {}}\n",
                     "    - {id: memfs-resolver, type: memfs, scheme: memfs, config: {unknown: true}}\n"),
        "memfs resolver config must be empty");
    expect_failure(
        replace_once(memfs,
                     "    - {id: memfs-datapath, type: memfs, config: {}}\n",
                     "    - {id: memfs-datapath, type: memfs, config: {io_granularity: 4096}}\n"),
        "memfs datapath config must be empty");
    expect_failure(
        replace_once(memfs, "resource: memory-0, config: {}}",
                     "resource: memory-0, config: {stripe_unit: 4096}}"),
        "memfs backend config must be empty");

    const StorageContract* local_contract = find_storage_contract("ext4-local-nvme");
    const StorageContract* striped_contract =
        find_storage_contract("striped-local-nvme");
    const StorageContract* memfs_contract = find_storage_contract("memfs");
    check(local_contract != nullptr &&
              local_contract->resolver_type == "local-file" &&
              local_contract->resolver_scheme == "file" &&
              local_contract->datapath_type == "local-nvme" &&
              local_contract->resource_type == "nvme" &&
              local_contract->minimum_cardinality == 1 &&
              local_contract->maximum_cardinality == 1 &&
              local_contract->resolver_type_id ==
                  tutti::binding::ext4_local_nvme::kResolverTypeId &&
              local_contract->payload_type_id ==
                  tutti::binding::ext4_local_nvme::kPayloadTypeId &&
              local_contract->payload_api_version ==
                  tutti::binding::ext4_local_nvme::kPayloadApiVersion &&
              local_contract->data_path_key ==
                  tutti::binding::ext4_local_nvme::kRecommendedDataPathKey &&
              local_contract->implemented,
          "local contract registry entry");
    check(striped_contract != nullptr &&
              striped_contract->resolver_type == "striped-file" &&
              striped_contract->resolver_scheme == "striped" &&
              striped_contract->datapath_type == "striped-local-nvme" &&
              striped_contract->resource_type == "nvme" &&
              striped_contract->minimum_cardinality == 2 &&
              striped_contract->resolver_type_id ==
                  tutti::binding::striped_local_nvme::kResolverTypeId &&
              striped_contract->payload_type_id ==
                  tutti::binding::striped_local_nvme::kPayloadTypeId &&
              striped_contract->payload_api_version ==
                  tutti::binding::striped_local_nvme::kPayloadApiVersion &&
              striped_contract->data_path_key ==
                  tutti::binding::striped_local_nvme::kRecommendedDataPathKey,
          "striped contract registry entry");
    check(memfs_contract != nullptr && !memfs_contract->implemented,
          "memfs schema-only contract registry entry");
    check(find_storage_contract("missing") == nullptr,
          "unknown contract registry lookup");

    std::printf("storage config checks=%d failures=%d\n", checks, failures);
    return failures == 0 ? 0 : 1;
}

#include <tutti/config/tutti_runtime_config_parser.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

namespace {

int failures = 0;

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,      \
                         #condition);                                          \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

class TempConfig {
public:
    explicit TempConfig(const std::string& contents) {
        char path[] = "/tmp/tutti-config-parser-XXXXXX";
        const int descriptor = ::mkstemp(path);
        CHECK(descriptor >= 0);
        if (descriptor >= 0) ::close(descriptor);
        path_ = path;
        std::ofstream stream(path_);
        stream << contents;
    }
    ~TempConfig() { if (!path_.empty()) ::unlink(path_.c_str()); }
    const std::string& path() const { return path_; }
private:
    std::string path_;
};

std::string local_yaml(const std::string& backend_config = "{}") {
    return
        "accelerator: {profile: CUDA}\n"
        "runtime: {accel_id: 0}\n"
        "storage:\n"
        "  resources:\n"
        "    - id: nvme-0\n"
        "      type: nvme\n"
        "      provider: {type: nvme-service, endpoint: endpoint}\n"
        "      allocation: {selection: explicit, device_ids: [2], queues_per_controller: 4}\n"
        "  resolvers:\n"
        "    - {id: resolver-0, type: local-file, scheme: file, config: {}}\n"
        "  datapaths:\n"
        "    - id: datapath-0\n"
        "      type: local-nvme\n"
        "      config: {handle_cache_capacity: 7, prp_cache_capacity: 8, handle_cache_l2_capacity: 9, threads_per_block: 64, max_in_flight_operations: 10, max_batch_entries: 11, io_granularity: 12}\n"
        "  backends:\n"
        "    - id: backend-0\n"
        "      contract: ext4-local-nvme\n"
        "      resolver: resolver-0\n"
        "      datapath: datapath-0\n"
        "      resource: nvme-0\n"
        "      config: " + backend_config + "\n";
}

} // namespace

int main() {
    {
        TempConfig file(local_yaml());
        auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
        CHECK(parsed.ok());
        if (parsed.ok()) {
            const auto& spec = parsed.value();
            CHECK(spec.accelerator.profile == "CUDA");
            CHECK(spec.runtime.accel_id == 0);
            const auto* resource = std::get_if<tutti::config::NvmeResourceConfig>(
                &spec.storage.resources.front().config);
            CHECK(resource != nullptr);
            if (resource != nullptr) {
                CHECK(resource->allocation.device_ids ==
                      std::vector<std::int32_t>{2});
            }
            const auto* datapath =
                std::get_if<tutti::config::LocalNvmeDataPathConfig>(
                    &spec.storage.datapaths.front().config);
            CHECK(datapath != nullptr);
            if (datapath != nullptr) {
                CHECK(datapath->handle_cache_capacity == 7);
                CHECK(datapath->prp_cache_capacity == 8);
                CHECK(datapath->handle_cache_l2_capacity == 9);
                CHECK(datapath->threads_per_block == 64);
                CHECK(datapath->max_in_flight_operations == 10);
                CHECK(datapath->max_batch_entries == 11);
                CHECK(datapath->io_granularity == 12);
            }
        }
    }
    {
        TempConfig file("storage:\n  resources: {}\n  resolvers: []\n  datapaths: []\n  backends: []\n");
        auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
        CHECK(!parsed.ok());
        CHECK(parsed.status().message().find("storage.resources") !=
              std::string::npos);
    }
    {
        TempConfig file(local_yaml("{stripe_unit: 4096}"));
        auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
        CHECK(!parsed.ok());
        CHECK(parsed.status().message().find("unknown field stripe_unit") !=
              std::string::npos);
    }
    {
        const std::string striped_bad_type =
            "accelerator: {profile: CUDA}\n"
            "runtime: {accel_id: 0}\n"
            "storage:\n"
            "  resources:\n"
            "    - {id: nvme, type: nvme, provider: {type: nvme-service, endpoint: e}, allocation: {selection: striped, device_ids: [0, 1], queues_per_controller: 2}}\n"
            "  resolvers: [{id: r, type: striped-file, scheme: striped}]\n"
            "  datapaths: [{id: d, type: striped-local-nvme}]\n"
            "  backends: [{id: b, contract: striped-local-nvme, resolver: r, datapath: d, resource: nvme, config: {stripe_unit: wrong}}]\n";
        TempConfig file(striped_bad_type);
        auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
        CHECK(!parsed.ok());
        CHECK(parsed.status().message().find(
                  "storage.backends[0].config.stripe_unit") !=
              std::string::npos);
    }
    {
        std::string yaml = local_yaml();
        const auto position = yaml.find("  resolvers:\n");
        yaml.insert(position,
            "    - id: nvme-0\n"
            "      type: nvme\n"
            "      provider: {type: nvme-service, endpoint: second}\n"
            "      allocation: {selection: explicit, device_ids: [3], queues_per_controller: 1}\n");
        TempConfig file(yaml);
        auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
        CHECK(parsed.ok());
        if (parsed.ok()) CHECK(!parsed.value().validate().ok());
    }
    {
        const std::string striped =
            "accelerator: {profile: CUDA}\n"
            "runtime: {accel_id: 0}\n"
            "storage:\n"
            "  resources:\n"
            "    - {id: nvme, type: nvme, provider: {type: nvme-service, endpoint: e}, allocation: {selection: striped, device_ids: [0, 1], queues_per_controller: 2}}\n"
            "  resolvers: [{id: r, type: striped-file, scheme: striped}]\n"
            "  datapaths: [{id: d, type: striped-local-nvme}]\n"
            "  backends: [{id: b, contract: striped-local-nvme, resolver: r, datapath: d, resource: nvme}]\n";
        TempConfig file(striped);
        auto parsed = tutti::config::parse_tutti_runtime_config(file.path());
        CHECK(parsed.ok());
        if (parsed.ok()) {
            const auto* config =
                std::get_if<tutti::config::StripedLocalNvmeBackendConfig>(
                    &parsed.value().storage.backends.front().config);
            CHECK(config != nullptr);
            CHECK(config != nullptr &&
                  config->stripe_unit == tutti::config::kDefaultStripedStripeUnit);
        }
    }
    std::printf("parser tests: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

// tests/config_loader/config_loader_test.cpp
//
// Round 20 S1 — tutti_config loader unit tests.
// Pure host-side (no GPU/snsvm needed); tests:
//   1. parse priority chain: programmatic > config > env > default
//   2. RDMA placeholder → UNSUPPORTED
//   3. bad yaml → fail-closed
//   4. valid parse of all keys

#include <cassert>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include <tutti/config/tutti_config.h>
#include <tutti/storage_runtime.h>
#include "tutti/resource/nvme/nvme_resource_internal.h"

using namespace tutti::config;
using tutti::resources::nvme::NvmeAcceleratorInfo;
using tutti::resources::nvme::NvmeProviderResource;
using tutti::resources::nvme::NvmeResourceClient;
using tutti::resources::nvme::NvmeResourceSpec;
using tutti::resources::nvme::NvmeResourceTestingAccess;
using tutti::resources::nvme::RuntimeNvmeAllocation;
using tutti::resources::nvme::RuntimeNvmeSlice;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; } \
    else { std::printf("  FAIL: %s\n", msg); ++g_fail; } \
} while(0)

static std::string write_tmp(const std::string& content) {
    char tmpl[] = "/tmp/tutti_cfg_XXXXXX";
    int fd = mkstemp(tmpl);
    assert(fd >= 0);
    write(fd, content.data(), content.size());
    close(fd);
    return std::string(tmpl);
}

struct FakeClientState {
    std::vector<NvmeAcceleratorInfo> accelerators;
    std::vector<NvmeProviderResource> resources;
    RuntimeNvmeAllocation allocation;
    tutti::Status acquire_status;
    int list_accelerators_calls = 0;
    int list_resources_calls = 0;
    int acquire_calls = 0;
    int release_calls = 0;
    tutti::Status release_status;
    int runtime_shutdown_calls = 0;
    std::vector<TuttiRuntimeShutdownStage> shutdown_stages;
    std::string released_allocation;
    std::string endpoint;
};

class FakeClient final : public NvmeResourceClient {
public:
    explicit FakeClient(std::shared_ptr<FakeClientState> state)
        : state_(std::move(state)) {}

    tutti::Result<std::vector<NvmeAcceleratorInfo>>
    list_accelerators() override {
        ++state_->list_accelerators_calls;
        return tutti::Result<std::vector<NvmeAcceleratorInfo>>::Success(
            state_->accelerators);
    }

    tutti::Result<std::vector<NvmeProviderResource>>
    list_nvme_resources() override {
        ++state_->list_resources_calls;
        return tutti::Result<std::vector<NvmeProviderResource>>::Success(
            state_->resources);
    }

    tutti::Result<RuntimeNvmeAllocation> acquire_nvme_slices(
        std::int32_t,
        NvmeSelection,
        const std::vector<std::int32_t>&,
        std::int32_t) override {
        ++state_->acquire_calls;
        if (!state_->acquire_status.ok()) {
            return tutti::Result<RuntimeNvmeAllocation>::Failure(
                state_->acquire_status);
        }
        return tutti::Result<RuntimeNvmeAllocation>::Success(
            state_->allocation);
    }

    tutti::Status release(const std::string& allocation_id) override {
        ++state_->release_calls;
        state_->released_allocation = allocation_id;
        return state_->release_status;
    }

private:
    std::shared_ptr<FakeClientState> state_;
};

static RuntimeNvmeSlice make_slice(std::int32_t device_id,
                                   std::int32_t accel_id = 0) {
    RuntimeNvmeSlice slice;
    slice.device_id = device_id;
    slice.accel_id = accel_id;
    slice.pci_bdf = "0000:4" + std::to_string(device_id) + ":00.0";
    slice.chrdev_path = "/dev/from-daemon/ssnvme" + std::to_string(device_id);
    slice.block_path = "/dev/from-daemon/snvme" + std::to_string(device_id) + "n1";
    slice.backing_mount_path = "/mnt/backing/nvme" + std::to_string(device_id);
    slice.view_path = "/mnt/view/gpu0/nvme" + std::to_string(device_id);
    slice.namespace_id = 1;
    slice.logical_block_size = 4096;
    slice.bar0_size = 16384;
    slice.max_data_size = 131072;
    slice.granted_queues = 4;
    slice.allowed_accel_ids = {0, 1};
    return slice;
}

static std::shared_ptr<FakeClientState> make_state(
    std::vector<RuntimeNvmeSlice> slices) {
    auto state = std::make_shared<FakeClientState>();
    state->accelerators = {{0, "/views/zero"}, {1, "/views/one"}};
    state->resources = {
        {0, {0, 1}, true},
        {1, {0, 1}, true},
    };
    state->allocation.allocation_id = "alloc-test";
    state->allocation.slices = std::move(slices);
    return state;
}

static std::string canonical_local_yaml(
    const std::string& selection = "explicit",
    const std::string& device_ids = "[0]",
    const std::string& queues = "4",
    const std::string& resource_id = "nvme-local-0",
    const std::string& resolver_type = "local-file",
    const std::string& scheme = "file",
    const std::string& contract = "ext4-local-nvme",
    const std::string& resource_reference = "nvme-local-0",
    const std::string& resource_extra = {},
    const std::string& resolver_extra = {},
    const std::string& backend_extra = {},
    const std::string& root_extra = {}) {
    return "accelerator: {profile: CUDA}\n"
           "runtime: {accel_id: 0}\n"
           "storage:\n"
           "  resources:\n"
           "    - id: '" + resource_id + "'\n"
           "      type: nvme\n"
           "      provider: {type: nvme-service, endpoint: fake-endpoint}\n"
           "      allocation:\n"
           "        selection: " + selection + "\n"
           "        device_ids: " + device_ids + "\n"
           "        queues_per_controller: " + queues + "\n" +
           resource_extra +
           "  resolvers:\n"
           "    - id: file-resolver-0\n"
           "      type: " + resolver_type + "\n"
           "      scheme: '" + scheme + "'\n"
           "      config: {}\n" +
           resolver_extra +
           "  datapaths:\n"
           "    - id: local-nvme-datapath-0\n"
           "      type: local-nvme\n"
           "      config: {}\n"
           "  backends:\n"
           "    - id: model-storage\n"
           "      contract: " + contract + "\n"
           "      resolver: file-resolver-0\n"
           "      datapath: local-nvme-datapath-0\n"
           "      resource: " + resource_reference + "\n"
           "      config: {}\n" +
           backend_extra + root_extra;
}

static std::string canonical_striped_yaml(const std::string& device_ids,
                                          const std::string& stripe_unit) {
    return "accelerator: {profile: CUDA}\n"
           "runtime: {accel_id: 0}\n"
           "storage:\n"
           "  resources:\n"
           "    - id: nvme-striped-0\n"
           "      type: nvme\n"
           "      provider: {type: nvme-service, endpoint: fake-endpoint}\n"
           "      allocation: {selection: striped, device_ids: " + device_ids +
           ", queues_per_controller: 4}\n"
           "  resolvers:\n"
           "    - {id: striped-resolver-0, type: striped-file, scheme: striped, config: {}}\n"
           "  datapaths:\n"
           "    - {id: striped-datapath-0, type: striped-local-nvme, config: {}}\n"
           "  backends:\n"
           "    - id: striped-storage\n"
           "      contract: striped-local-nvme\n"
           "      resolver: striped-resolver-0\n"
           "      datapath: striped-datapath-0\n"
           "      resource: nvme-striped-0\n"
           "      config: {stripe_unit: " + stripe_unit + "}\n";
}

static LoadTuttiConfigOptions fake_options(
    const std::shared_ptr<FakeClientState>& state,
    bool fail_runtime_create = false) {
    LoadTuttiConfigOptions options;
    options.backend_device_count = [] {
        return tutti::Result<int>::Success(2);
    };
    options.resource_factory = [state](const ResourceSpec& resource_spec,
                                       std::int32_t accel_id) {
        state->endpoint = resource_spec.provider.endpoint;
        auto nvme_resource = NvmeResourceTestingAccess::make(
            NvmeResourceSpec{
                resource_spec.id,
                accel_id,
                resource_spec.provider,
                resource_spec.allocation,
            },
            std::make_unique<FakeClient>(state));
        std::unique_ptr<tutti::Resource> resource =
            std::move(nvme_resource);
        return tutti::Result<std::unique_ptr<tutti::Resource>>::Success(
            std::move(resource));
    };
    options.runtime_factory =
        [fail_runtime_create](tutti::RuntimeConfig,
                              tutti::RuntimeComponents components) {
            if (fail_runtime_create) {
                return tutti::Result<std::unique_ptr<tutti::StorageRuntime>>::Failure(
                    tutti::Status(tutti::StatusCode::INTERNAL,
                                  "injected runtime create failure"));
            }
            CHECK(components.resolvers.size() == 1,
                  "loader: one top-level resolver binding");
            CHECK(components.data_paths.size() == 1,
                  "loader: one top-level DataPath binding");
            tutti::RuntimeConfig host_cfg;
            host_cfg.accel_id = -1;
            host_cfg.profile_name = TUTTI_COMPILED_ACCELERATOR_PROFILE;
            return tutti::StorageRuntime::create(host_cfg);
        };
    return options;
}

int main() {
    // 1. Valid parse — all keys.
    {
        std::string yaml = R"(
gpu:
  vendor: nvidia
storage:
  backend: local-nvme
  default_stripe_unit: 524288
local_nvme:
  handle_cache_capacity: 64
  prp_cache_capacity: 128
  handle_cache_l2_capacity: 256
  max_in_flight_operations: 32
  max_batch_entries: 512
  num_user_queues: 8
  io_granularity: 524288
local_nvme_config: local_nvme_config.yaml
accelerator:
  profile: CUDA
runtime:
  accel_id: 0
nvme_service:
  endpoint: "127.0.0.1:50051"
nvme:
  selection: "striped"
  device_ids: [0, 1]
  queues_per_controller: 4
  stripe_unit: 65536
)";
        auto path = write_tmp(yaml);
        auto r = parse_tutti_config(path);
        CHECK(r.ok(), "valid parse: ok");
        if (r.ok()) {
            const auto& c = r.value();
            CHECK(c.gpu_vendor == "nvidia", "gpu_vendor");
            CHECK(c.storage_backend == "local-nvme", "storage_backend");
            CHECK(c.default_stripe_unit == 524288, "stripe_unit");
            CHECK(c.handle_cache_capacity == 64, "handle_cache_capacity");
            CHECK(c.prp_cache_capacity == 128, "prp_cache_capacity");
            CHECK(c.handle_cache_l2_capacity == 256, "l2_capacity");
            CHECK(c.max_in_flight_operations == 32, "max_in_flight");
            CHECK(c.max_batch_entries == 512, "max_batch_entries");
            CHECK(c.num_user_queues == 8, "num_user_queues");
            CHECK(c.io_granularity == 524288, "io_granularity");
            CHECK(c.local_nvme_config == "local_nvme_config.yaml",
                  "local_nvme_config link");
            CHECK(c.accelerator_profile == "CUDA", "accelerator profile");
            CHECK(c.runtime_accel_id == 0, "runtime accel_id");
            CHECK(c.nvme_selection == NvmeSelection::Striped,
                  "nvme selection");
            CHECK(c.nvme_device_ids.size() == 2 &&
                  c.nvme_device_ids[1] == 1, "nvme device ids");
            CHECK(c.queues_per_controller == 4, "queues per controller");
            CHECK(c.stripe_unit == 65536, "stripe unit");
        }
        ::unlink(path.c_str());
    }

    // 2. RDMA placeholder → UNSUPPORTED.
    {
        std::string yaml = "storage:\n  backend: rdma\n";
        auto path = write_tmp(yaml);
        auto r = parse_tutti_config(path);
        CHECK(!r.ok(), "rdma: fail-closed");
        if (!r.ok()) {
            CHECK((int)r.status().code() == (int)tutti::StatusCode::UNSUPPORTED,
                  "rdma: UNSUPPORTED status");
        }
        ::unlink(path.c_str());
    }

    // 3. Bad yaml → fail-closed.
    {
        std::string yaml = "this is not: [valid: yaml\n";
        auto path = write_tmp(yaml);
        auto r = parse_tutti_config(path);
        CHECK(!r.ok(), "bad yaml: fail-closed");
        ::unlink(path.c_str());
    }

    // 4. Empty/minimal config → defaults.
    {
        std::string yaml = "";
        auto path = write_tmp(yaml);
        auto r = parse_tutti_config(path);
        CHECK(r.ok(), "empty yaml: ok (defaults)");
        if (r.ok()) {
            const auto& c = r.value();
            CHECK(c.gpu_vendor == "nvidia", "default vendor");
            CHECK(c.storage_backend == "local-nvme", "default backend");
            CHECK(c.handle_cache_capacity == 0, "default cache cap=0");
            CHECK(c.local_nvme_config.empty(), "default no link");
        }
        ::unlink(path.c_str());
    }

    // 5. Priority chain: programmatic > config > env > default.
    {
        // Config sets handle_cache_capacity=64.
        std::string yaml = "local_nvme:\n  handle_cache_capacity: 64\n";
        auto path = write_tmp(yaml);
        auto r = parse_tutti_config(path);
        CHECK(r.ok(), "priority: parse ok");
        if (r.ok()) {
            // 5a. programmatic override > config
            ProgrammaticOverrides ov;
            ov.handle_cache_capacity = 100;
            auto eff = resolve_cache_config(r.value(), ov);
            CHECK(eff.handle_cache_capacity == 100,
                  "priority: programmatic > config");

            // 5b. config > env (set env, no programmatic)
            setenv("TUTTI_HANDLE_CACHE_CAP", "200", 1);
            ProgrammaticOverrides ov2;  // all 0 = defer
            auto eff2 = resolve_cache_config(r.value(), ov2);
            CHECK(eff2.handle_cache_capacity == 64,
                  "priority: config > env");
            unsetenv("TUTTI_HANDLE_CACHE_CAP");

            // 5c. env > default (no config key, no programmatic)
            ParsedConfig empty;
            setenv("TUTTI_HANDLE_CACHE_CAP", "300", 1);
            auto eff3 = resolve_cache_config(empty, {});
            CHECK(eff3.handle_cache_capacity == 300,
                  "priority: env > default");
            unsetenv("TUTTI_HANDLE_CACHE_CAP");

            // 5d. default when nothing set
            auto eff4 = resolve_cache_config(empty, {});
            CHECK(eff4.handle_cache_capacity == 0,
                  "priority: default = 0");
        }
        ::unlink(path.c_str());
    }

    // 6. Non-existent file → fail-closed.
    {
        auto r = parse_tutti_config("/tmp/tutti_nonexistent_config.yaml");
        CHECK(!r.ok(), "non-existent file: fail-closed");
    }

    // 7. New nvme selection validation.
    {
        auto bad_selection = write_tmp(
            "nvme:\n  selection: mystery\n");
        CHECK(!parse_tutti_config(bad_selection).ok(),
              "unknown selection fail-closed");
        ::unlink(bad_selection.c_str());

        auto explicit_zero = write_tmp(
            "nvme:\n  selection: explicit\n  device_ids: []\n");
        CHECK(!parse_tutti_config(explicit_zero).ok(),
              "explicit zero ids fail-closed");
        ::unlink(explicit_zero.c_str());

        auto explicit_many = write_tmp(
            "nvme:\n  selection: explicit\n  device_ids: [0, 1]\n");
        CHECK(!parse_tutti_config(explicit_many).ok(),
              "explicit many ids fail-closed");
        ::unlink(explicit_many.c_str());

        auto striped_one = write_tmp(
            "nvme:\n  selection: striped\n  device_ids: [0]\n");
        CHECK(!parse_tutti_config(striped_one).ok(),
              "striped one id fail-closed");
        ::unlink(striped_one.c_str());

        auto duplicate = write_tmp(
            "nvme:\n  selection: striped\n  device_ids: [0, 0]\n");
        CHECK(!parse_tutti_config(duplicate).ok(),
              "duplicate ids fail-closed");
        ::unlink(duplicate.c_str());
    }

    // 8. derive_local_nvme_devices — legacy parse-only topology helper.
    {
        std::string yaml = R"(
nvmes:
  - pci_addr: "0000:08:00.0"
    namespace_id: 1
    allowed_gpus: [0]
  - pci_addr: "0000:4b:00.0"
    namespace_id: 2
    allowed_gpus: [1, 2]
  - pci_addr: "0000:57:00.0"
)";
        auto path = write_tmp(yaml);
        auto r = derive_local_nvme_devices(path);
        CHECK(r.ok(), "derive: ok");
        if (r.ok()) {
            const auto& d = r.value();
            CHECK(d.size() == 3, "derive: 3 specs (1+2, third skipped)");
            if (d.size() == 3) {
                CHECK(d[0].snvme_dev == "/dev/ssnvme0" &&
                      d[0].cuda_device == 0, "derive: nvme0 -> gpu0");
                CHECK(d[1].snvme_dev == "/dev/ssnvme1" &&
                      d[1].cuda_device == 1 &&
                      d[1].namespace_id == 2, "derive: nvme1 -> gpu1 ns2");
                CHECK(d[2].cuda_device == 2, "derive: nvme1 -> gpu2");
            }
        }
        ::unlink(path.c_str());

        auto r2 = derive_local_nvme_devices("/tmp/tutti_nonexistent_lnvc.yaml");
        CHECK(!r2.ok(), "derive: missing file fail-closed");
    }

    // 9. Fake-client loader: single-slice allocation produces one top-level
    // file resolver and one local DataPath binding, then releases once.
    {
        std::string yaml = R"(
accelerator: {profile: CUDA}
runtime: {accel_id: 0}
nvme_service: {endpoint: "fake-endpoint"}
nvme:
  selection: explicit
  device_ids: [0]
  queues_per_controller: 4
)";
        auto path = write_tmp(yaml);
        auto state = make_state({make_slice(0)});
        auto r = load_tutti_config(path, fake_options(state));
        CHECK(r.ok(), "loader single-slice: ok");
        if (r.ok()) {
            auto bundle = std::move(r).value();
            CHECK(bundle->resolver_schemes.size() == 1 &&
                  bundle->resolver_schemes[0] == "file",
                  "loader single-slice: file resolver");
            CHECK(bundle->data_path_keys.size() == 1 &&
                  bundle->data_path_keys[0] == "local-nvme-ext4",
                  "loader single-slice: local DataPath key");
            CHECK(state->acquire_calls == 1, "loader single-slice: acquire once");
            CHECK(bundle->state() == TuttiRuntimeState::RUNNING,
                  "loader single-slice: runtime starts RUNNING");
            auto resource_info = bundle->resource_info();
            CHECK(resource_info.ok() &&
                      resource_info.value().id == "legacy-nvme-resource" &&
                      resource_info.value().type == "nvme" &&
                      resource_info.value().state ==
                          tutti::ResourceState::INITIALIZED,
                  "loader single-slice: Resource identity and state");
            if (resource_info.ok()) {
                resource_info.value().id = "mutated";
                CHECK(bundle->resource_info().value().id ==
                          "legacy-nvme-resource",
                      "loader single-slice: ResourceInfo is a copy");
            }
            CHECK(bundle->shutdown().ok(), "loader single-slice: shutdown ok");
            CHECK(bundle->state() == TuttiRuntimeState::STOPPED,
                  "loader single-slice: runtime stops");
            CHECK(bundle->resource_info().ok() &&
                      bundle->resource_info().value().state ==
                          tutti::ResourceState::STOPPED,
                  "loader single-slice: Resource stops after components");
            CHECK(bundle->shutdown().ok(),
                  "loader single-slice: second shutdown is idempotent");
            CHECK(state->release_calls == 1 &&
                  state->released_allocation == "alloc-test",
                  "loader single-slice: release once");
            bundle.reset();
            CHECK(state->release_calls == 1,
                  "loader single-slice: destructor no double release");
        }
        ::unlink(path.c_str());
    }

    // 10a. Destructor fallback and reverse lifecycle observation release once.
    {
        std::string yaml = R"(
accelerator: {profile: CUDA}
runtime: {accel_id: 0}
nvme:
  selection: explicit
  device_ids: [0]
)";
        auto path = write_tmp(yaml);
        auto state = make_state({make_slice(0)});
        auto options = fake_options(state);
        options.shutdown_observer = [state](TuttiRuntimeShutdownStage stage) {
            state->shutdown_stages.push_back(stage);
        };
        auto r = load_tutti_config(path, std::move(options));
        CHECK(r.ok(), "loader destructor fallback: load ok");
        if (r.ok()) {
            { auto bundle = std::move(r).value(); }
            CHECK(state->release_calls == 1,
                  "loader destructor fallback: release once");
            const std::vector<TuttiRuntimeShutdownStage> expected = {
                TuttiRuntimeShutdownStage::STORAGE_RUNTIME_SHUTDOWN,
                TuttiRuntimeShutdownStage::STORAGE_RUNTIME_DESTROYED,
                TuttiRuntimeShutdownStage::RESOLVERS_DESTROYED,
                TuttiRuntimeShutdownStage::DATAPATHS_DESTROYED,
                TuttiRuntimeShutdownStage::RESOURCE_SHUTDOWN,
                TuttiRuntimeShutdownStage::COMPLETE,
            };
            CHECK(state->shutdown_stages == expected,
                  "loader destructor fallback: reverse lifecycle order");
        }
        ::unlink(path.c_str());
    }

    // 10b. A StorageRuntime shutdown error is returned first, while resource
    // release still runs and remains idempotent.
    {
        std::string yaml = R"(
accelerator: {profile: CUDA}
runtime: {accel_id: 0}
nvme:
  selection: explicit
  device_ids: [0]
)";
        auto path = write_tmp(yaml);
        auto state = make_state({make_slice(0)});
        auto options = fake_options(state);
        options.runtime_shutdown_hook = [state](tutti::StorageRuntime&) {
            ++state->runtime_shutdown_calls;
            return tutti::Status(tutti::StatusCode::TIMEOUT,
                                  "injected runtime shutdown failure");
        };
        auto r = load_tutti_config(path, std::move(options));
        CHECK(r.ok(), "loader shutdown failure: load ok");
        if (r.ok()) {
            auto bundle = std::move(r).value();
            const auto status = bundle->shutdown();
            CHECK(!status.ok() && status.code() == tutti::StatusCode::TIMEOUT,
                  "loader shutdown failure: first error propagated");
            CHECK(state->runtime_shutdown_calls == 1 &&
                  state->release_calls == 1,
                  "loader shutdown failure: remaining cleanup attempted");
            CHECK(bundle->shutdown().ok() && state->release_calls == 1,
                  "loader shutdown failure: retry is idempotent");
        }
        ::unlink(path.c_str());
    }

    // 10c. A release error is returned, but the client is not called twice.
    {
        std::string yaml = R"(
accelerator: {profile: CUDA}
runtime: {accel_id: 0}
nvme:
  selection: explicit
  device_ids: [0]
)";
        auto path = write_tmp(yaml);
        auto state = make_state({make_slice(0)});
        state->release_status = tutti::Status(
            tutti::StatusCode::DEVICE_ERROR, "injected release failure");
        auto r = load_tutti_config(path, fake_options(state));
        CHECK(r.ok(), "loader release failure: load ok");
        if (r.ok()) {
            auto bundle = std::move(r).value();
            const auto status = bundle->shutdown();
            CHECK(!status.ok() && status.code() == tutti::StatusCode::DEVICE_ERROR,
                  "loader release failure: status propagated");
            CHECK(state->release_calls == 1,
                  "loader release failure: attempted once");
            CHECK(bundle->shutdown().ok() && state->release_calls == 1,
                  "loader release failure: retry is idempotent");
        }
        ::unlink(path.c_str());
    }

    // 10. Fake-client loader: two slices produce one striped top-level pair.
    {
        std::string yaml = R"(
accelerator: {profile: CUDA}
runtime: {accel_id: 0}
nvme_service: {endpoint: "fake-endpoint"}
nvme:
  selection: striped
  device_ids: [0, 1]
  queues_per_controller: 4
  stripe_unit: 65536
)";
        auto path = write_tmp(yaml);
        auto state = make_state({make_slice(0), make_slice(1)});
        auto r = load_tutti_config(path, fake_options(state));
        CHECK(r.ok(), "loader striped: ok");
        if (r.ok()) {
            auto bundle = std::move(r).value();
            CHECK(bundle->resolver_schemes.size() == 1 &&
                  bundle->resolver_schemes[0] == "striped",
                  "loader striped: striped resolver");
            CHECK(bundle->data_path_keys.size() == 1 &&
                  bundle->data_path_keys[0] == "striped-local-nvme",
                  "loader striped: striped DataPath key");
            (void)bundle->shutdown();
            CHECK(state->release_calls == 1, "loader striped: release once");
        }
        ::unlink(path.c_str());
    }

    // 11. Failures after Acquire release the allocation; preflight failures
    // do not acquire.
    {
        std::string yaml = R"(
accelerator: {profile: CUDA}
runtime: {accel_id: 0}
nvme:
  selection: striped
  device_ids: [0, 1]
)";
        auto path = write_tmp(yaml);
        auto state = make_state({make_slice(1), make_slice(0)});
        auto r = load_tutti_config(path, fake_options(state));
        CHECK(!r.ok(), "loader striped wrong order: fail");
        CHECK(state->acquire_calls == 1, "loader wrong order: acquired");
        CHECK(state->release_calls == 1, "loader wrong order: released");
        ::unlink(path.c_str());
    }

    {
        std::string yaml = R"(
accelerator: {profile: CUDA}
runtime: {accel_id: 0}
nvme:
  selection: explicit
  device_ids: [0]
)";
        auto path = write_tmp(yaml);
        auto state = make_state({make_slice(0, 1)});
        auto r = load_tutti_config(path, fake_options(state));
        CHECK(!r.ok(), "loader accel mismatch slice: fail");
        CHECK(state->acquire_calls == 1,
              "loader accel mismatch slice: acquired");
        CHECK(state->release_calls == 1,
              "loader accel mismatch slice: released");
        ::unlink(path.c_str());
    }

    {
        std::string yaml = R"(
accelerator: {profile: CUDA}
runtime: {accel_id: 1}
nvme:
  selection: explicit
  device_ids: [0]
)";
        auto path = write_tmp(yaml);
        auto state = make_state({make_slice(0, 1)});
        state->accelerators = {{0, "/views/zero"}};
        auto r = load_tutti_config(path, fake_options(state));
        CHECK(!r.ok(), "loader missing accelerator: fail");
        CHECK(state->acquire_calls == 0, "loader missing accelerator: no acquire");
        CHECK(state->release_calls == 0, "loader missing accelerator: no release");
        ::unlink(path.c_str());
    }

    {
        std::string yaml = R"(
accelerator: {profile: CUDA}
runtime: {accel_id: 0}
nvme:
  selection: explicit
  device_ids: [0]
)";
        auto path = write_tmp(yaml);
        auto state = make_state({make_slice(0)});
        state->acquire_status = tutti::Status(
            tutti::StatusCode::NOT_READY, "injected acquire failure");
        auto r = load_tutti_config(path, fake_options(state));
        CHECK(!r.ok(), "loader acquire failure: fail");
        CHECK(state->acquire_calls == 1, "loader acquire failure: acquire once");
        CHECK(state->release_calls == 0, "loader acquire failure: no release");
        ::unlink(path.c_str());
    }

    {
        std::string yaml = R"(
accelerator: {profile: CUDA}
runtime: {accel_id: 0}
nvme:
  selection: explicit
  device_ids: [0]
)";
        auto path = write_tmp(yaml);
        auto state = make_state({make_slice(0)});
        auto r = load_tutti_config(path, fake_options(state, true));
        CHECK(!r.ok(), "loader runtime create failure: fail");
        CHECK(state->release_calls == 1,
              "loader runtime create failure: release once");
        ::unlink(path.c_str());
    }

    // 12. Canonical configs feed the unchanged assembly compatibility layer.
    {
        auto path = write_tmp(canonical_local_yaml());
        auto state = make_state({make_slice(0)});
        auto r = load_tutti_config(path, fake_options(state));
        CHECK(r.ok(), "canonical loader local: ok");
        CHECK(state->endpoint == "fake-endpoint",
              "canonical loader local: provider endpoint adapted");
        CHECK(state->list_accelerators_calls == 1 &&
              state->list_resources_calls == 1 && state->acquire_calls == 1,
              "canonical loader local: one preflight and acquire");
        if (r.ok()) {
            auto bundle = std::move(r).value();
            CHECK(bundle->resolver_schemes == std::vector<std::string>{"file"},
                  "canonical loader local: resolver route");
            CHECK(bundle->data_path_keys ==
                      std::vector<std::string>{"local-nvme-ext4"},
                  "canonical loader local: DataPath route");
            (void)bundle->shutdown();
            CHECK(state->release_calls == 1,
                  "canonical loader local: release once");
        }
        ::unlink(path.c_str());
    }

    {
        auto path = write_tmp(canonical_striped_yaml("[0, 1]", "65536"));
        auto state = make_state({make_slice(0), make_slice(1)});
        auto r = load_tutti_config(path, fake_options(state));
        CHECK(r.ok(), "canonical loader striped: ok");
        if (r.ok()) {
            auto bundle = std::move(r).value();
            CHECK(bundle->resolver_schemes == std::vector<std::string>{"striped"},
                  "canonical loader striped: resolver route");
            CHECK(bundle->data_path_keys ==
                      std::vector<std::string>{"striped-local-nvme"},
                  "canonical loader striped: DataPath route");
            (void)bundle->shutdown();
            CHECK(state->release_calls == 1,
                  "canonical loader striped: release once");
        }
        ::unlink(path.c_str());
    }

    // 13. Every canonical static failure returns before client creation/list/acquire.
    {
        const std::string duplicate_resource =
            "    - id: nvme-local-0\n"
            "      type: nvme\n"
            "      provider: {type: nvme-service, endpoint: duplicate}\n"
            "      allocation: {selection: explicit, device_ids: [1], queues_per_controller: 4}\n";
        const std::string duplicate_resolver =
            "    - {id: file-resolver-1, type: local-file, scheme: file, config: {}}\n";
        const std::string unused_resolver =
            "    - {id: unused-resolver, type: local-file, scheme: unused, config: {}}\n";
        const std::string duplicate_backend =
            "    - id: model-storage-2\n"
            "      contract: ext4-local-nvme\n"
            "      resolver: file-resolver-0\n"
            "      datapath: local-nvme-datapath-0\n"
            "      resource: nvme-local-0\n"
            "      config: {}\n";
        std::string zero_backend = canonical_local_yaml();
        const std::size_t backends_position = zero_backend.find("  backends:\n");
        zero_backend.erase(backends_position);
        zero_backend += "  backends: []\n";
        const std::vector<std::pair<std::string, std::string>> invalid_configs = {
            {"missing id", canonical_local_yaml("explicit", "[0]", "4", "")},
            {"duplicate id", canonical_local_yaml(
                "explicit", "[0]", "4", "nvme-local-0", "local-file", "file",
                "ext4-local-nvme", "nvme-local-0", duplicate_resource)},
            {"dangling reference", canonical_local_yaml(
                "explicit", "[0]", "4", "nvme-local-0", "local-file", "file",
                "ext4-local-nvme", "missing-resource")},
            {"duplicate scheme", canonical_local_yaml(
                "explicit", "[0]", "4", "nvme-local-0", "local-file", "file",
                "ext4-local-nvme", "nvme-local-0", {}, duplicate_resolver)},
            {"duplicate key", canonical_local_yaml(
                "explicit", "[0]", "4", "nvme-local-0", "local-file", "file",
                "ext4-local-nvme", "nvme-local-0", {}, {}, duplicate_backend)},
            {"zero backend", zero_backend},
            {"unreferenced", canonical_local_yaml(
                "explicit", "[0]", "4", "nvme-local-0", "local-file", "file",
                "ext4-local-nvme", "nvme-local-0", {}, unused_resolver)},
            {"unknown contract", canonical_local_yaml(
                "explicit", "[0]", "4", "nvme-local-0", "local-file", "file",
                "unknown-contract")},
            {"unknown type", canonical_local_yaml(
                "explicit", "[0]", "4", "nvme-local-0", "unknown-resolver")},
            {"invalid scheme", canonical_local_yaml(
                "explicit", "[0]", "4", "nvme-local-0", "local-file", "Bad Scheme")},
            {"allowed ids", canonical_local_yaml("allowed", "[0]")},
            {"explicit zero", canonical_local_yaml("explicit", "[]")},
            {"explicit many", canonical_local_yaml("explicit", "[0, 1]")},
            {"striped one", canonical_striped_yaml("[0]", "65536")},
            {"striped duplicate", canonical_striped_yaml("[0, 0]", "65536")},
            {"negative queues", canonical_local_yaml("explicit", "[0]", "-1")},
            {"zero stripe", canonical_striped_yaml("[0, 1]", "0")},
            {"unaligned stripe", canonical_striped_yaml("[0, 1]", "65537")},
            {"mixed syntax", canonical_local_yaml(
                "explicit", "[0]", "4", "nvme-local-0", "local-file", "file",
                "ext4-local-nvme", "nvme-local-0", {}, {}, {},
                "nvme_service: {endpoint: legacy}\n")},
        };
        for (const auto& entry : invalid_configs) {
            auto path = write_tmp(entry.second);
            auto state = make_state({make_slice(0), make_slice(1)});
            auto r = load_tutti_config(path, fake_options(state));
            CHECK(!r.ok(), ("canonical static failure rejected: " + entry.first).c_str());
            CHECK(state->endpoint.empty() &&
                  state->list_accelerators_calls == 0 &&
                  state->list_resources_calls == 0 &&
                  state->acquire_calls == 0 && state->release_calls == 0,
                  ("canonical static failure has zero RPC: " + entry.first).c_str());
            ::unlink(path.c_str());
        }
    }

    std::printf("\n=== Summary ===\n  passed: %d\n  failed: %d\n", g_pass, g_fail);
    if (g_fail > 0) { std::printf("RESULT: FAIL\n"); return 1; }
    std::printf("RESULT: PASS\n");
    return 0;
}

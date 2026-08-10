#include "nvmeservice_client.h"
#include "nvmeservice_config.h"
#include "nvmeservice_server.h"
#include "nvmeservice_state.h"

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

int failures = 0;
int checks = 0;

#define CHECK(condition, message)                                             \
    do {                                                                      \
        ++checks;                                                             \
        if (!(condition)) {                                                   \
            ++failures;                                                       \
            std::cerr << "FAIL line=" << __LINE__ << ": " << message << '\n'; \
        }                                                                     \
    } while (false)

class TempConfigs {
public:
    TempConfigs() {
        directory_ = std::filesystem::temp_directory_path() /
            ("tutti-phase3-" + std::to_string(::getpid()));
        std::filesystem::create_directories(directory_);
    }
    ~TempConfigs() {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }
    std::string write(const std::string& name, const std::string& contents) {
        const auto path = directory_ / name;
        std::ofstream output(path);
        output << contents;
        return path.string();
    }
private:
    std::filesystem::path directory_;
};

std::string suffix() {
    return R"(
queue_pool: {default_per_client: 2, max_per_client: 4}
lease: {heartbeat_interval_sec: 1, timeout_sec: 3}
unmount_retry: {interval_ms: 1, max: 1}
)";
}

std::string canonical(const std::string& nvmes) {
    return R"(
grpc: {endpoint: "127.0.0.1:0"}
accelerators:
  - {accel_id: 1, view_root: "/views/one"}
  - {accel_id: 0, view_root: "/views/zero"}
nvmes:
)" + nvmes + suffix();
}

void expect_parse_failure(TempConfigs& files, const std::string& name,
                          const std::string& yaml) {
    std::string error;
    CHECK(!nvmeservice::parse_config_file(files.write(name, yaml), &error),
          name + " must fail");
    CHECK(!error.empty(), name + " must report an error");
}

nvmeservice::ServiceConfig make_config() {
    nvmeservice::ServiceConfig config;
    config.grpc.endpoint = "127.0.0.1:0";
    config.accelerators = {{0, "/views/zero"}, {1, "/views/one"}};
    config.nvmes = {
        {20, "0000:44:00.0", "/backing/twenty", 1, 0, {0, 1}, false},
        {10, "0000:41:00.0", "/backing/ten", 1, 0, {0, 1}, false},
    };
    config.queue_pool.default_per_client = 2;
    config.queue_pool.max_per_client = 4;
    config.lease.heartbeat_interval_sec = 1;
    config.lease.timeout_sec = 3;
    config.unmount_retry.interval_ms = 1;
    config.unmount_retry.max = 1;
    return config;
}

nvmeservice::NvmeResourceSnapshot resource(int32_t device_id,
                                            uint32_t capacity = 8) {
    nvmeservice::NvmeResourceSnapshot result;
    result.device_id = device_id;
    result.pci_bdf = device_id == 10 ? "0000:41:00.0" : "0000:44:00.0";
    result.chrdev_minor = device_id == 10 ? 7 : 9;
    result.chrdev_path = device_id == 10 ? "/dev/ssnvme7" : "/dev/ssnvme9";
    result.disk_name = device_id == 10 ? "snvme7n1" : "snvme9n1";
    result.block_path = "/dev/" + result.disk_name;
    result.backing_mount_path = device_id == 10
        ? "/backing/ten" : "/backing/twenty";
    result.namespace_id = 1;
    result.page_size = 4096;
    result.logical_block_size = 4096;
    result.logical_block_size_log = 12;
    result.queue_depth = 1024;
    result.dstrd = 0;
    result.bar0_size = 32768;
    result.max_data_size = 524288;
    result.max_user_qid = 96;
    result.kernel_io_qps = 32;
    result.controller_queue_capacity = capacity;
    result.max_queues_per_group = 4;
    result.allowed_accel_ids = {0, 1};
    result.view_paths = {
        {0, "/views/zero/" + std::filesystem::path(result.chrdev_path).filename().string()},
        {1, "/views/one/" + std::filesystem::path(result.chrdev_path).filename().string()},
    };
    result.available = true;
    result.heartbeat_interval_sec = 1;
    result.lease_timeout_sec = 3;
    return result;
}

auto never_dead() {
    return [](uint32_t, uint64_t) { return false; };
}

nvmeservice::AcquireRequest request(nvmeservice::SelectionMode selection,
                                    std::vector<int32_t> device_ids = {},
                                    int32_t queues = 1,
                                    int32_t accel_id = 0) {
    nvmeservice::AcquireRequest result;
    result.accel_id = accel_id;
    result.selection = selection;
    result.device_ids = std::move(device_ids);
    result.queues_per_controller = queues;
    result.client_pid = static_cast<uint32_t>(::getpid());
    return result;
}

uint32_t reserved(const nvmeservice::ServiceState& state, int32_t device_id) {
    for (const auto& item : state.list_nvme_resources()) {
        if (item.device_id == device_id) return item.reserved_queues;
    }
    return UINT32_MAX;
}

void test_config_contract() {
    TempConfigs files;
    std::string error;
    nvmeservice::ConfigDiagnostics diagnostics;
    const auto parsed = nvmeservice::parse_config_file(files.write(
        "canonical.yaml", canonical(R"(
  - {device_id: 9, pci_addr: "0000:0A:00.0", backing_mount_path: "/backing/nine", allowed_accel_ids: []}
  - {device_id: 2, pci_addr: "0000:0b:00.0", backing_mount_path: "/backing/two", allowed_accel_ids: [1]}
)")), &error, &diagnostics);
    CHECK(parsed.has_value(), "canonical schema parses");
    CHECK(diagnostics.warnings.empty(), "canonical schema has no warning");
    if (parsed) {
        CHECK(parsed->nvmes[0].pci_addr == "0000:0a:00.0", "BDF normalized");
        CHECK(parsed->nvmes[0].allowed_accel_ids == std::vector<int32_t>({0, 1}),
              "empty ACL expands and sorts");
        CHECK(parsed->nvmes[0].device_id == 9, "explicit device identity retained");
    }

    diagnostics.warnings.clear();
    const auto legacy = nvmeservice::parse_config_file(files.write(
        "legacy.yaml", R"(
gpus:
  - {id: 0, mount_path: "/legacy/view"}
nvmes:
  - {pci_addr: "0000:41:00.0", mount_path: "/legacy/backing", allowed_gpus: [0]}
)" + suffix()), &error, &diagnostics);
    CHECK(legacy.has_value(), "legacy-only schema parses");
    CHECK(diagnostics.warnings.size() >= 2, "legacy warnings are injectable");
    if (legacy) {
        CHECK(legacy->nvmes[0].device_id == 0, "legacy array index normalized once");
        CHECK(legacy->nvmes[0].allowed_accel_ids == std::vector<int32_t>({0}),
              "legacy ACL normalized");
    }

    expect_parse_failure(files, "mixed-top.yaml", R"(
accelerators: [{accel_id: 0, view_root: "/v0"}]
gpus: [{id: 0, mount_path: "/old"}]
nvmes: []
)" + suffix());
    expect_parse_failure(files, "mixed-entry.yaml", canonical(R"(
  - {device_id: 0, pci_addr: "0000:41:00.0", mount_path: "/old", backing_mount_path: "/new"}
)"));
    expect_parse_failure(files, "mixed-accelerator-entry.yaml", R"(
accelerators: [{accel_id: 0, id: 0, view_root: "/v"}]
nvmes: [{device_id: 0, pci_addr: "0000:41:00.0", backing_mount_path: "/b"}]
)" + suffix());
    expect_parse_failure(files, "missing-id.yaml", canonical(R"(
  - {pci_addr: "0000:41:00.0", backing_mount_path: "/b"}
)"));
    expect_parse_failure(files, "negative-id.yaml", canonical(R"(
  - {device_id: -1, pci_addr: "0000:41:00.0", backing_mount_path: "/b"}
)"));
    expect_parse_failure(files, "duplicate-id.yaml", canonical(R"(
  - {device_id: 1, pci_addr: "0000:41:00.0", backing_mount_path: "/b1"}
  - {device_id: 1, pci_addr: "0000:42:00.0", backing_mount_path: "/b2"}
)"));
    expect_parse_failure(files, "duplicate-bdf.yaml", canonical(R"(
  - {device_id: 1, pci_addr: "0000:0A:00.0", backing_mount_path: "/b1"}
  - {device_id: 2, pci_addr: "0000:0a:00.0", backing_mount_path: "/b2"}
)"));
    expect_parse_failure(files, "duplicate-path.yaml", canonical(R"(
  - {device_id: 1, pci_addr: "0000:41:00.0", backing_mount_path: "/views/zero"}
)"));
    expect_parse_failure(files, "unknown-acl.yaml", canonical(R"(
  - {device_id: 1, pci_addr: "0000:41:00.0", backing_mount_path: "/b", allowed_accel_ids: [8]}
)"));
    expect_parse_failure(files, "duplicate-acl.yaml", canonical(R"(
  - {device_id: 1, pci_addr: "0000:41:00.0", backing_mount_path: "/b", allowed_accel_ids: [0, 0]}
)"));
    expect_parse_failure(files, "overflow-accel.yaml", R"(
accelerators: [{accel_id: 2147483648, view_root: "/v"}]
nvmes: [{device_id: 0, pci_addr: "0000:41:00.0", backing_mount_path: "/b"}]
)" + suffix());
    expect_parse_failure(files, "negative-accel.yaml", R"(
accelerators: [{accel_id: -1, view_root: "/v"}]
nvmes: [{device_id: 0, pci_addr: "0000:41:00.0", backing_mount_path: "/b"}]
)" + suffix());
    expect_parse_failure(files, "duplicate-accel.yaml", R"(
accelerators:
  - {accel_id: 0, view_root: "/v0"}
  - {accel_id: 0, view_root: "/v1"}
nvmes: [{device_id: 0, pci_addr: "0000:41:00.0", backing_mount_path: "/b"}]
)" + suffix());
}

void test_bringup_validation() {
    nvmeservice::BringupMetadata metadata;
    metadata.configured_pci_bdf = "0000:41:00.0";
    metadata.observed_pci_bdf = "0000:41:00.0";
    metadata.chrdev_minor = 7;
    metadata.chrdev_path = "/dev/ssnvme7";
    metadata.disk_name = "snvme7n1";
    metadata.block_path = "/dev/snvme7n1";
    nvmeservice::BringupValidationProbe probe;
    probe.is_character_device = [](const std::string&) { return true; };
    probe.is_block_device = [](const std::string&) { return true; };
    probe.character_minor = [](const std::string&) { return std::optional<uint32_t>(7); };
    probe.block_matches_pci_bdf = [](const std::string&, const std::string&) {
        return true;
    };
    std::string error;
    CHECK(nvmeservice::validate_bringup_metadata(metadata, probe, &error),
          "valid bring-up metadata accepted");
    auto bad = metadata;
    bad.observed_pci_bdf = "0000:44:00.0";
    CHECK(!nvmeservice::validate_bringup_metadata(bad, probe, &error),
          "BDF mismatch fails closed");
    bad = metadata;
    bad.disk_name.clear();
    CHECK(!nvmeservice::validate_bringup_metadata(bad, probe, &error),
          "empty disk name fails closed");
    auto stale_probe = probe;
    stale_probe.character_minor = [](const std::string&) {
        return std::optional<uint32_t>(8);
    };
    CHECK(!nvmeservice::validate_bringup_metadata(metadata, stale_probe, &error),
          "stale minor fails closed");
    auto mismatch_probe = probe;
    mismatch_probe.block_matches_pci_bdf = [](const std::string&, const std::string&) {
        return false;
    };
    CHECK(!nvmeservice::validate_bringup_metadata(metadata, mismatch_probe, &error),
          "block/BDF mismatch fails closed");
}

void test_allocator_contract() {
    auto config = make_config();
    nvmeservice::ServiceState state(config, {resource(20), resource(10)}, never_dead());
    const auto listed = state.list_nvme_resources();
    CHECK(listed[0].device_id == 10 && listed[1].device_id == 20,
          "list order uses explicit device_id");

    auto allowed = state.acquire(request(nvmeservice::SelectionMode::Allowed));
    CHECK(allowed.success && allowed.grant.slices.size() == 1,
          "allowed returns one slice");
    CHECK(allowed.success && allowed.grant.slices[0].device_id == 10,
          "allowed selects lowest device_id, not array order");
    CHECK(state.release(allowed.grant.allocation_id).success,
          "allowed allocation releases");

    auto explicit_result = state.acquire(request(
        nvmeservice::SelectionMode::Explicit, {20}));
    CHECK(explicit_result.success &&
          explicit_result.grant.slices[0].device_id == 20,
          "explicit returns requested resource");
    CHECK(state.release(explicit_result.grant.allocation_id).success,
          "explicit releases");

    auto striped = state.acquire(request(
        nvmeservice::SelectionMode::Striped, {20, 10}, 2));
    CHECK(striped.success && striped.grant.slices.size() == 2,
          "striped has one allocation and two slices");
    CHECK(striped.success && striped.grant.slices[0].device_id == 20 &&
          striped.grant.slices[1].device_id == 10,
          "striped preserves request order");
    CHECK(reserved(state, 10) == 2 && reserved(state, 20) == 2,
          "striped reserves every controller");
    const auto release = state.release(striped.grant.allocation_id);
    CHECK(release.success && reserved(state, 10) == 0 && reserved(state, 20) == 0,
          "one Release refunds every striped reservation");
    const auto duplicate_release = state.release(striped.grant.allocation_id);
    CHECK(duplicate_release.success && duplicate_release.already_released &&
          reserved(state, 10) == 0 && reserved(state, 20) == 0,
          "repeat Release is idempotent without double refund");

    CHECK(!state.acquire(request(nvmeservice::SelectionMode::Explicit, {})).success,
          "explicit requires one ID");
    CHECK(!state.acquire(request(nvmeservice::SelectionMode::Striped, {10})).success,
          "striped requires two IDs");
    CHECK(!state.acquire(request(nvmeservice::SelectionMode::Striped, {10, 10})).success,
          "striped rejects duplicates");
    CHECK(!state.acquire(request(nvmeservice::SelectionMode::Explicit, {99})).success,
          "unknown device fails");
    CHECK(!state.acquire(request(nvmeservice::SelectionMode::Allowed, {10})).success,
          "allowed rejects explicit IDs");
    CHECK(!state.acquire(request(nvmeservice::SelectionMode::Allowed, {}, 1, 7)).success,
          "unknown accelerator fails");

    auto acl_resource = resource(10);
    acl_resource.allowed_accel_ids = {1};
    acl_resource.view_paths.erase(0);
    nvmeservice::ServiceState acl_state(config, {acl_resource}, never_dead());
    CHECK(!acl_state.acquire(request(nvmeservice::SelectionMode::Explicit, {10})).success,
          "ACL rejection is explicit");

    auto unavailable_resource = resource(10);
    unavailable_resource.available = false;
    unavailable_resource.diagnostic = "stale minor";
    nvmeservice::ServiceState unavailable(config, {unavailable_resource}, never_dead());
    CHECK(!unavailable.acquire(request(nvmeservice::SelectionMode::Explicit, {10})).success,
          "unavailable resource cannot acquire");

    auto bad_view = resource(10);
    bad_view.view_paths[0] = "/other/root/ssnvme7";
    nvmeservice::ServiceState bad_view_state(config, {bad_view}, never_dead());
    CHECK(!bad_view_state.acquire(request(nvmeservice::SelectionMode::Explicit, {10})).success,
          "view outside configured root cannot acquire");

    nvmeservice::ServiceState budget(config, {resource(10, 3)}, never_dead());
    auto full = budget.acquire(request(nvmeservice::SelectionMode::Explicit, {10}, 3));
    CHECK(full.success && reserved(budget, 10) == 3, "capacity can be fully reserved");
    CHECK(!budget.acquire(request(nvmeservice::SelectionMode::Explicit, {10}, 1)).success,
          "budget exhaustion rejects without overcommit");
    budget.release(full.grant.allocation_id);

    auto first_id = std::make_shared<std::atomic<int>>(0);
    auto collision_generator = [first_id]() {
        ++(*first_id);
        return std::string("collision");
    };
    nvmeservice::ServiceState collision(config, {resource(10), resource(20)},
                                         never_dead(), collision_generator);
    auto first = collision.acquire(request(nvmeservice::SelectionMode::Explicit, {10}));
    CHECK(first.success, "first deterministic allocation inserts");
    const uint32_t before = reserved(collision, 20);
    auto collided = collision.acquire(request(nvmeservice::SelectionMode::Explicit, {20}));
    CHECK(!collided.success && reserved(collision, 20) == before,
          "allocation insert collision leaves no partial reservation");
    collision.release(first.grant.allocation_id);

    auto legacy = state.connect(10, 0, 2, static_cast<uint32_t>(::getpid()));
    CHECK(legacy.success && reserved(state, 10) == 2,
          "legacy Connect uses canonical ledger");
    std::string disconnect_error;
    CHECK(state.disconnect(legacy.grant.allocation_id,
                           static_cast<uint32_t>(::getpid()),
                           &disconnect_error) && reserved(state, 10) == 0,
          "legacy Disconnect uses unified refund");
}

void test_reaper_and_concurrency() {
    auto config = make_config();
    nvmeservice::ServiceState timeout_state(config, {resource(10)}, never_dead());
    auto allocation = timeout_state.acquire(request(
        nvmeservice::SelectionMode::Explicit, {10}, 2));
    CHECK(allocation.success, "timeout allocation created");
    CHECK(timeout_state.reap_once(std::chrono::steady_clock::now() +
                                  std::chrono::seconds(10)) == 1 &&
          reserved(timeout_state, 10) == 0,
          "heartbeat timeout refunds allocation");

    nvmeservice::ServiceState dead_state(config, {resource(10)},
        [](uint32_t, uint64_t) { return true; });
    auto dead = dead_state.acquire(request(
        nvmeservice::SelectionMode::Explicit, {10}, 2));
    CHECK(dead.success && dead_state.reap_once(std::chrono::steady_clock::now()) == 1 &&
          reserved(dead_state, 10) == 0,
          "dead/reused PID probe refunds allocation");

    nvmeservice::ServiceState concurrent(config, {resource(10, 32)}, never_dead());
    std::atomic<bool> bad{false};
    std::vector<std::thread> threads;
    for (int thread_index = 0; thread_index < 8; ++thread_index) {
        threads.emplace_back([&]() {
            for (int iteration = 0; iteration < 100; ++iteration) {
                auto acquired = concurrent.acquire(request(
                    nvmeservice::SelectionMode::Explicit, {10}, 1));
                if (!acquired.success) continue;
                if (reserved(concurrent, 10) > 32) bad = true;
                if (!concurrent.release(acquired.grant.allocation_id).success) bad = true;
            }
        });
    }
    for (auto& thread : threads) thread.join();
    CHECK(!bad.load(), "concurrent reservations never exceed capacity");
    CHECK(reserved(concurrent, 10) == 0, "concurrent ledger returns to zero");
}

void test_rpc_contract() {
    auto config = make_config();
    auto state = std::make_shared<nvmeservice::ServiceState>(
        config, std::vector<nvmeservice::NvmeResourceSnapshot>{resource(20), resource(10)},
        never_dead());
    nvmeservice::NvmeServiceImpl service(state);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    CHECK(server && port > 0, "in-process gRPC server starts");
    if (!server) return;

    const std::string endpoint = "127.0.0.1:" + std::to_string(port);
    auto channel = grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials());
    auto stub = nvmeservice::NvmeService::NewStub(channel);

    nvmeservice::Empty empty;
    nvmeservice::AcceleratorListResponse accelerators;
    grpc::ClientContext accelerator_context;
    CHECK(stub->ListAccelerators(&accelerator_context, empty, &accelerators).ok() &&
          accelerators.accelerators_size() == 2,
          "ListAccelerators RPC works");

    nvmeservice::NvmeResourceListResponse resources;
    grpc::ClientContext resource_context;
    CHECK(stub->ListNvmeResources(&resource_context, empty, &resources).ok() &&
          resources.resources_size() == 2,
          "ListNvmeResources RPC works");
    if (resources.resources_size() == 2) {
        const auto& item = resources.resources(0);
        CHECK(item.device_id() == 10 && !item.pci_bdf().empty() &&
              !item.chrdev_path().empty() && !item.block_path().empty() &&
              !item.backing_mount_path().empty() && item.namespace_id() == 1 &&
              item.page_size() == 4096 && item.logical_block_size() == 4096 &&
              item.queue_depth() == 1024 && item.bar0_size() == 32768 &&
              item.controller_queue_capacity() == 8 &&
              item.allowed_accel_ids_size() == 2,
              "new resource metadata is complete");
    }

    nvmeservice::AcquireNvmeSlicesRequest acquire_request;
    acquire_request.set_accel_id(0);
    acquire_request.set_selection(nvmeservice::NVME_SELECTION_STRIPED);
    acquire_request.add_device_ids(20);
    acquire_request.add_device_ids(10);
    acquire_request.set_queues_per_controller(2);
    acquire_request.set_client_pid(static_cast<uint32_t>(::getpid()));
    nvmeservice::AcquireNvmeSlicesResponse acquire_response;
    grpc::ClientContext acquire_context;
    CHECK(stub->AcquireNvmeSlices(&acquire_context, acquire_request,
                                  &acquire_response).ok() &&
          acquire_response.error_message().empty() &&
          acquire_response.slices_size() == 2 &&
          acquire_response.slices(0).device_id() == 20 &&
          acquire_response.slices(1).device_id() == 10,
          "AcquireNvmeSlices returns one ordered striped allocation");

    nvmeservice::ReleaseRequest release_request;
    release_request.set_allocation_id(acquire_response.allocation_id());
    nvmeservice::ReleaseResponse release_response;
    grpc::ClientContext release_context;
    CHECK(stub->Release(&release_context, release_request, &release_response).ok() &&
          release_response.success() && reserved(*state, 10) == 0 &&
          reserved(*state, 20) == 0,
          "Release RPC refunds all slices");

    nvmeservice::DeviceListResponse legacy_list;
    grpc::ClientContext legacy_list_context;
    CHECK(stub->ListDevices(&legacy_list_context, empty, &legacy_list).ok() &&
          legacy_list.devices_size() == 2 &&
          legacy_list.devices(0).device_id() == 10,
          "legacy ListDevices adapts canonical snapshot");

    nvmeservice::ConnectRequest connect_request;
    connect_request.set_device_id(10);
    connect_request.set_cuda_device(0);
    connect_request.set_num_queues(2);
    connect_request.set_client_pid(static_cast<uint32_t>(::getpid()));
    nvmeservice::ConnectResponse connect_response;
    grpc::ClientContext connect_context;
    CHECK(stub->Connect(&connect_context, connect_request, &connect_response).ok() &&
          connect_response.error_message().empty() && reserved(*state, 10) == 2,
          "legacy Connect shares canonical ledger");

    grpc::ClientContext heartbeat_context;
    auto heartbeat = stub->Heartbeat(&heartbeat_context);
    nvmeservice::HeartbeatMsg heartbeat_message;
    heartbeat_message.set_allocation_id(connect_response.allocation_id());
    heartbeat_message.set_timestamp_ns(1);
    CHECK(heartbeat->Write(heartbeat_message), "legacy Heartbeat accepts allocation");
    heartbeat->WritesDone();
    nvmeservice::HeartbeatMsg heartbeat_reply;
    CHECK(heartbeat->Read(&heartbeat_reply) && !heartbeat_reply.has_notice(),
          "legacy Heartbeat remains compatible");
    heartbeat->Finish();

    nvmeservice::DisconnectRequest disconnect_request;
    disconnect_request.set_allocation_id(connect_response.allocation_id());
    disconnect_request.set_client_pid(static_cast<uint32_t>(::getpid()));
    nvmeservice::DisconnectResponse disconnect_response;
    grpc::ClientContext disconnect_context;
    CHECK(stub->Disconnect(&disconnect_context, disconnect_request,
                           &disconnect_response).ok() &&
          disconnect_response.success() && reserved(*state, 10) == 0,
          "legacy Disconnect refunds canonical reservation");

    const auto* connect_descriptor = nvmeservice::ConnectRequest::descriptor();
    CHECK(connect_descriptor->FindFieldByName("device_id")->number() == 1 &&
          connect_descriptor->FindFieldByName("cuda_device")->number() == 2 &&
          connect_descriptor->FindFieldByName("num_queues")->number() == 3 &&
          connect_descriptor->FindFieldByName("client_pid")->number() == 4,
          "legacy protobuf field numbers are preserved");

    server->Shutdown();
    server->Wait();
}

} // namespace

int main() {
    test_config_contract();
    test_bringup_validation();
    test_allocator_contract();
    test_reaper_and_concurrency();
    test_rpc_contract();
    if (failures == 0) {
        std::cout << "nvmeservice phase 3 contract: PASS (" << checks
                  << " checks)\n";
        return 0;
    }
    std::cerr << "nvmeservice phase 3 contract: FAIL (" << failures
              << "/" << checks << ")\n";
    return 1;
}

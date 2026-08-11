#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tutti/resource.h>
#include "tutti/resource/nvme/nvme_resource_internal.h"

namespace {

using tutti::Resource;
using tutti::ResourceState;
using tutti::Result;
using tutti::Status;
using tutti::StatusCode;
using tutti::config::NvmeSelection;
using tutti::resources::nvme::NvmeAcceleratorInfo;
using tutti::resources::nvme::NvmeDataPathResourceView;
using tutti::resources::nvme::NvmeLeaseState;
using tutti::resources::nvme::NvmeProviderResource;
using tutti::resources::nvme::NvmeResolverResourceView;
using tutti::resources::nvme::NvmeResource;
using tutti::resources::nvme::NvmeResourceClient;
using tutti::resources::nvme::NvmeResourceSpec;
using tutti::resources::nvme::NvmeResourceTestingAccess;
using tutti::resources::nvme::RuntimeNvmeAllocation;
using tutti::resources::nvme::RuntimeNvmeSlice;

int passed = 0;
int failed = 0;

#define CHECK(condition, message) do { \
    if (condition) { \
        ++passed; \
    } else { \
        std::printf("FAIL: %s\n", message); \
        ++failed; \
    } \
} while (0)

struct FakeState {
    std::vector<NvmeAcceleratorInfo> accelerators{{0, "/mnt/gpu0"}};
    std::vector<NvmeProviderResource> resources{{0, {0}, true}};
    RuntimeNvmeAllocation allocation;
    Status list_accelerators_status;
    Status list_resources_status;
    Status acquire_status;
    Status release_status;
    bool throw_on_release = false;
    int list_accelerators_calls = 0;
    int list_resources_calls = 0;
    int acquire_calls = 0;
    int release_calls = 0;
    std::string released_allocation_id;
};

class FakeNvmeResourceClient final : public NvmeResourceClient {
public:
    explicit FakeNvmeResourceClient(std::shared_ptr<FakeState> state)
        : state_(std::move(state)) {}

    Result<std::vector<NvmeAcceleratorInfo>>
    list_accelerators() override {
        ++state_->list_accelerators_calls;
        if (!state_->list_accelerators_status.ok()) {
            return Result<std::vector<NvmeAcceleratorInfo>>::Failure(
                state_->list_accelerators_status);
        }
        return Result<std::vector<NvmeAcceleratorInfo>>::Success(
            state_->accelerators);
    }

    Result<std::vector<NvmeProviderResource>>
    list_nvme_resources() override {
        ++state_->list_resources_calls;
        if (!state_->list_resources_status.ok()) {
            return Result<std::vector<NvmeProviderResource>>::Failure(
                state_->list_resources_status);
        }
        return Result<std::vector<NvmeProviderResource>>::Success(
            state_->resources);
    }

    Result<RuntimeNvmeAllocation> acquire_nvme_slices(
        std::int32_t,
        NvmeSelection,
        const std::vector<std::int32_t>&,
        std::int32_t) override {
        ++state_->acquire_calls;
        if (!state_->acquire_status.ok()) {
            return Result<RuntimeNvmeAllocation>::Failure(
                state_->acquire_status);
        }
        return Result<RuntimeNvmeAllocation>::Success(state_->allocation);
    }

    Status release(const std::string& allocation_id) override {
        ++state_->release_calls;
        state_->released_allocation_id = allocation_id;
        if (state_->throw_on_release) {
            throw std::runtime_error("injected release exception");
        }
        return state_->release_status;
    }

private:
    std::shared_ptr<FakeState> state_;
};

RuntimeNvmeSlice valid_slice() {
    RuntimeNvmeSlice slice;
    slice.device_id = 0;
    slice.accel_id = 0;
    slice.pci_bdf = "0000:b1:00.0";
    slice.chrdev_path = "/dev/ssnvme0";
    slice.block_path = "/dev/snvme0n1";
    slice.backing_mount_path = "/mnt/nvme0";
    slice.view_path = "/mnt/gpu0/ssnvme0";
    slice.namespace_id = 1;
    slice.logical_block_size = 4096;
    slice.bar0_size = 16384;
    slice.max_data_size = 131072;
    slice.granted_queues = 4;
    slice.allowed_accel_ids = {0};
    return slice;
}

std::shared_ptr<FakeState> valid_state() {
    auto state = std::make_shared<FakeState>();
    state->allocation.allocation_id = "allocation-0";
    state->allocation.slices = {valid_slice()};
    return state;
}

NvmeResourceSpec valid_spec() {
    NvmeResourceSpec spec;
    spec.id = "nvme-local-0";
    spec.accel_id = 0;
    spec.provider.type = "nvme-service";
    spec.provider.endpoint = "fake-endpoint";
    spec.allocation.selection = NvmeSelection::Explicit;
    spec.allocation.device_ids = {0};
    spec.allocation.queues_per_controller = 4;
    return spec;
}

std::unique_ptr<NvmeResource> make_resource(
    const std::shared_ptr<FakeState>& state) {
    return NvmeResourceTestingAccess::make(
        valid_spec(), std::make_unique<FakeNvmeResourceClient>(state));
}

void test_success_and_read_only_views() {
    auto state = valid_state();
    auto resource = make_resource(state);
    Resource* common = resource.get();

    CHECK(common->info().id == "nvme-local-0" &&
              common->info().type == "nvme" &&
              common->info().state == ResourceState::CREATED,
          "ResourceInfo matches CREATED spec identity");
    CHECK(common->capabilities().resource_type == "nvme" &&
              common->capabilities().provides_resolver_view &&
              common->capabilities().provides_datapath_view,
          "ResourceCapabilities expose stable NVMe capabilities");
    CHECK(common->initialize().ok(), "NVMe Resource initialize succeeds");
    CHECK(state->list_accelerators_calls == 1 &&
              state->list_resources_calls == 1 &&
              state->acquire_calls == 1,
          "initialize performs one snapshot and one Acquire");
    CHECK(common->info().state == ResourceState::INITIALIZED,
          "ResourceInfo enters INITIALIZED");
    CHECK(NvmeResourceTestingAccess::inspection(*resource).lease_state ==
              NvmeLeaseState::ACQUIRED,
          "successful initialize owns the lease");

    auto resolver_view = resource->resolver_view();
    auto datapath_view = resource->datapath_view();
    CHECK(resolver_view.ok() && resolver_view.value().slices.size() == 1 &&
              resolver_view.value().slices.front().block_path ==
                  "/dev/snvme0n1",
          "resolver view contains resolver-only metadata");
    CHECK(datapath_view.ok() && datapath_view.value().slices.size() == 1 &&
              datapath_view.value().slices.front().chrdev_path ==
                  "/dev/ssnvme0",
          "DataPath view contains datapath-only metadata");
    if (resolver_view.ok() && datapath_view.ok()) {
        resolver_view.value().slices.front().block_path = "mutated";
        datapath_view.value().slices.front().chrdev_path = "mutated";
        CHECK(resource->resolver_view().value().slices.front().block_path ==
                  "/dev/snvme0n1" &&
                  resource->datapath_view().value().slices.front().chrdev_path ==
                      "/dev/ssnvme0",
              "NVMe construction views are detached read-only copies");
    }
    auto info = common->info();
    info.id = "mutated";
    CHECK(common->info().id == "nvme-local-0",
          "ResourceInfo is a detached snapshot");

    const Status duplicate_initialize = common->initialize();
    CHECK(!duplicate_initialize.ok() &&
              duplicate_initialize.code() == StatusCode::BUSY &&
              state->acquire_calls == 1,
          "duplicate initialize does not Acquire twice");
    CHECK(common->shutdown().ok(), "NVMe Resource shutdown succeeds");
    CHECK(common->info().state == ResourceState::STOPPED &&
              NvmeResourceTestingAccess::inspection(*resource).lease_state ==
                  NvmeLeaseState::RELEASED,
          "shutdown reaches STOPPED and consumes release slot");
    CHECK(state->release_calls == 1 &&
              state->released_allocation_id == "allocation-0",
          "shutdown releases the owned allocation once");
    CHECK(common->shutdown().ok() && state->release_calls == 1,
          "duplicate shutdown is idempotent");
    resource.reset();
    CHECK(state->release_calls == 1,
          "destructor does not release after explicit shutdown");
}

void test_acquire_failure() {
    auto state = valid_state();
    state->acquire_status = Status(StatusCode::NOT_READY,
                                   "injected acquire failure");
    {
        auto resource = make_resource(state);
        const Status status = resource->initialize();
        CHECK(!status.ok() && status.code() == StatusCode::NOT_READY,
              "Acquire failure is propagated");
        CHECK(resource->info().state == ResourceState::FAILED,
              "Acquire failure enters FAILED");
    }
    CHECK(state->acquire_calls == 1 && state->release_calls == 0,
          "Acquire failure has no allocation to release");
}

void test_invalid_allocation_releases_once() {
    {
        auto state = valid_state();
        state->allocation.allocation_id.clear();
        auto resource = make_resource(state);
        const Status status = resource->initialize();
        CHECK(!status.ok() && status.code() == StatusCode::INVALID_ARGUMENT,
              "empty allocation ID is rejected");
        CHECK(resource->info().state == ResourceState::FAILED &&
                  NvmeResourceTestingAccess::inspection(*resource).lease_state ==
                      NvmeLeaseState::RELEASED,
              "invalid allocation enters FAILED after release attempt");
        CHECK(state->release_calls == 1 &&
                  state->released_allocation_id.empty(),
              "empty-ID allocation still consumes one release attempt");
        resource.reset();
        CHECK(state->release_calls == 1,
              "empty-ID allocation is not released twice");
    }

    {
        auto state = valid_state();
        state->allocation.slices.front().view_path.clear();
        auto resource = make_resource(state);
        const Status status = resource->initialize();
        CHECK(!status.ok() && status.code() == StatusCode::INVALID_ARGUMENT,
              "missing allocation metadata is rejected");
        CHECK(state->release_calls == 1,
              "missing metadata rolls back the acquired allocation");
    }
}

void test_release_failures_are_not_retried() {
    {
        auto state = valid_state();
        state->release_status = Status(StatusCode::DEVICE_ERROR,
                                       "injected release failure");
        auto resource = make_resource(state);
        CHECK(resource->initialize().ok(),
              "release-status case initializes");
        const Status status = resource->shutdown();
        CHECK(!status.ok() && status.code() == StatusCode::DEVICE_ERROR,
              "release status is propagated");
        CHECK(resource->info().state == ResourceState::FAILED &&
                  state->release_calls == 1,
              "release failure enters FAILED after one attempt");
        CHECK(resource->shutdown().ok() && state->release_calls == 1,
              "release status failure is not retried");
    }

    {
        auto state = valid_state();
        state->throw_on_release = true;
        auto resource = make_resource(state);
        CHECK(resource->initialize().ok(),
              "release-exception case initializes");
        const Status status = resource->shutdown();
        CHECK(!status.ok() && status.code() == StatusCode::INTERNAL,
              "release exception becomes INTERNAL");
        resource.reset();
        CHECK(state->release_calls == 1,
              "release exception consumes the only release attempt");
    }
}

void test_destructor_fallback_and_preinit_view() {
    auto state = valid_state();
    {
        auto resource = make_resource(state);
        CHECK(!resource->resolver_view().ok() &&
                  !resource->datapath_view().ok(),
              "construction views reject CREATED resource");
        CHECK(resource->initialize().ok(),
              "destructor fallback case initializes");
    }
    CHECK(state->release_calls == 1,
          "destructor fallback releases exactly once");

    auto null_client = NvmeResourceTestingAccess::make(
        valid_spec(), nullptr);
    const Status status = null_client->initialize();
    CHECK(!status.ok() && status.code() == StatusCode::INVALID_ARGUMENT &&
              null_client->info().state == ResourceState::FAILED,
          "null client fails closed without transport access");
}

} // namespace

int main() {
    test_success_and_read_only_views();
    test_acquire_failure();
    test_invalid_allocation_releases_once();
    test_release_failures_are_not_retried();
    test_destructor_fallback_and_preinit_view();

    std::printf("passed: %d\nfailed: %d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}

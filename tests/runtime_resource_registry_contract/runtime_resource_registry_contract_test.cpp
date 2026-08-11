#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tutti/resource.h>
#include "tutti/tutti_runtime/tutti_runtime_internal.h"

namespace {

using tutti::Resource;
using tutti::ResourceCapabilities;
using tutti::ResourceInfo;
using tutti::ResourceState;
using tutti::Status;
using tutti::StatusCode;
using tutti::config::TuttiRuntime;
using tutti::config::TuttiRuntimeTestingAccess;

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

struct Ledger {
    int live_allocations = 0;
    std::vector<std::string> events;
};

class FakeResource final : public Resource {
public:
    FakeResource(std::string id, std::shared_ptr<Ledger> ledger,
                 Status initialize_status = Status::Ok(),
                 Status shutdown_status = Status::Ok())
        : id_(std::move(id)), ledger_(std::move(ledger)),
          initialize_status_(std::move(initialize_status)),
          shutdown_status_(std::move(shutdown_status)) {}

    ~FakeResource() override { (void)shutdown(); }

    const ResourceCapabilities& capabilities() const override {
        static const ResourceCapabilities capabilities{"fake", true, true};
        return capabilities;
    }

    Status initialize() override {
        ledger_->events.push_back("initialize:" + id_);
        if (!initialize_status_.ok()) {
            state_ = ResourceState::FAILED;
            return initialize_status_;
        }
        state_ = ResourceState::INITIALIZED;
        owns_allocation_ = true;
        ++ledger_->live_allocations;
        return Status::Ok();
    }

    Status shutdown() override {
        if (!owns_allocation_) return Status::Ok();
        owns_allocation_ = false;
        --ledger_->live_allocations;
        ledger_->events.push_back("shutdown:" + id_);
        state_ = shutdown_status_.ok() ? ResourceState::STOPPED
                                       : ResourceState::FAILED;
        return shutdown_status_;
    }

    ResourceInfo info() const override {
        return ResourceInfo{id_, "fake", state_};
    }

private:
    std::string id_;
    std::shared_ptr<Ledger> ledger_;
    Status initialize_status_;
    Status shutdown_status_;
    ResourceState state_ = ResourceState::CREATED;
    bool owns_allocation_ = false;
};

std::unique_ptr<FakeResource> initialized_resource(
    const std::string& id, const std::shared_ptr<Ledger>& ledger,
    Status shutdown_status = Status::Ok()) {
    auto resource = std::make_unique<FakeResource>(
        id, ledger, Status::Ok(), std::move(shutdown_status));
    CHECK(resource->initialize().ok(), "fake Resource initialize succeeds");
    return resource;
}

void test_registry_and_reverse_shutdown() {
    auto ledger = std::make_shared<Ledger>();
    TuttiRuntime runtime;
    CHECK(TuttiRuntimeTestingAccess::adopt_resource(
              runtime, "resource-0",
              initialized_resource("resource-0", ledger)).ok(),
          "first initialized Resource enters registry");
    CHECK(TuttiRuntimeTestingAccess::adopt_resource(
              runtime, "resource-1",
              initialized_resource("resource-1", ledger)).ok(),
          "second initialized Resource enters registry");

    const auto infos = runtime.resource_infos();
    CHECK(infos.size() == 2 && infos[0].id == "resource-0" &&
              infos[1].id == "resource-1",
          "ResourceInfo snapshots follow initialization order");
    CHECK(runtime.resource_info("resource-0").ok() &&
              !runtime.resource_info().ok(),
          "multi-Resource lookup is ID-addressed");
    CHECK(TuttiRuntimeTestingAccess::resource_initialization_order(runtime) ==
              std::vector<std::string>({"resource-0", "resource-1"}),
          "Runtime records successful initialization order");

    CHECK(runtime.shutdown().ok(), "multi-Resource shutdown succeeds");
    CHECK(ledger->events == std::vector<std::string>({
              "initialize:resource-0", "initialize:resource-1",
              "shutdown:resource-1", "shutdown:resource-0"}),
          "Runtime shuts Resources down in reverse initialization order");
    CHECK(ledger->live_allocations == 0 && runtime.shutdown().ok(),
          "Runtime shutdown returns the ledger to zero and is idempotent");
}

void test_second_snapshot_failure_rolls_back_first() {
    auto ledger = std::make_shared<Ledger>();
    {
        TuttiRuntime runtime;
        CHECK(TuttiRuntimeTestingAccess::adopt_resource(
                  runtime, "resource-0",
                  initialized_resource("resource-0", ledger)).ok(),
              "snapshot failure setup registers first Resource");
        auto second = std::make_unique<FakeResource>(
            "resource-1", ledger,
            Status(StatusCode::NOT_FOUND,
                   "second Resource provider snapshot is missing"));
        const Status status = second->initialize();
        CHECK(!status.ok() && status.code() == StatusCode::NOT_FOUND,
              "second Resource snapshot failure is explicit");
        CHECK(TuttiRuntimeTestingAccess::resource_initialization_order(
                  runtime) == std::vector<std::string>{"resource-0"},
              "failed second Resource is absent from initialization order");
    }
    CHECK(ledger->live_allocations == 0 &&
              ledger->events.back() == "shutdown:resource-0",
          "failed second Resource unwinds the first allocation once");
}

void test_second_factory_failure_rolls_back_first() {
    auto ledger = std::make_shared<Ledger>();
    {
        TuttiRuntime runtime;
        ledger->events.push_back("factory:resource-0");
        CHECK(TuttiRuntimeTestingAccess::adopt_resource(
                  runtime, "resource-0",
                  initialized_resource("resource-0", ledger)).ok(),
              "factory failure setup registers first Resource");
        ledger->events.push_back("factory:resource-1:failed");
        CHECK(TuttiRuntimeTestingAccess::resource_initialization_order(
                  runtime) == std::vector<std::string>{"resource-0"},
              "factory failure leaves only successful Resource in order");
    }
    CHECK(ledger->live_allocations == 0 &&
              ledger->events.back() == "shutdown:resource-0",
          "second ResourceFactory failure unwinds initialized Resources");
}

void test_duplicate_id_and_shutdown_error() {
    auto ledger = std::make_shared<Ledger>();
    TuttiRuntime runtime;
    CHECK(TuttiRuntimeTestingAccess::adopt_resource(
              runtime, "resource-0",
              initialized_resource("resource-0", ledger)).ok(),
          "duplicate test registers first Resource");
    const Status duplicate = TuttiRuntimeTestingAccess::adopt_resource(
        runtime, "resource-0", initialized_resource("resource-0", ledger));
    CHECK(!duplicate.ok() && duplicate.code() == StatusCode::INVALID_ARGUMENT,
          "duplicate Resource ID is rejected");
    CHECK(ledger->live_allocations == 1,
          "rejected duplicate Resource releases its own allocation");

    CHECK(TuttiRuntimeTestingAccess::adopt_resource(
              runtime, "resource-1",
              initialized_resource(
                  "resource-1", ledger,
                  Status(StatusCode::DEVICE_ERROR,
                         "injected Resource shutdown failure"))).ok(),
          "shutdown error test registers second Resource");
    const Status status = runtime.shutdown();
    CHECK(!status.ok() && status.code() == StatusCode::DEVICE_ERROR,
          "reverse cleanup returns the first shutdown error");
    CHECK(ledger->live_allocations == 0 &&
              ledger->events[ledger->events.size() - 2] ==
                  "shutdown:resource-1" &&
              ledger->events.back() == "shutdown:resource-0",
          "shutdown error does not skip earlier Resources");
}

} // namespace

int main() {
    test_registry_and_reverse_shutdown();
    test_second_snapshot_failure_rolls_back_first();
    test_second_factory_failure_rolls_back_first();
    test_duplicate_id_and_shutdown_error();

    std::printf("passed: %d\nfailed: %d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}

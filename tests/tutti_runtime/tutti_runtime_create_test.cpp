#include <tutti/tutti_runtime.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include <tutti/storage_runtime.h>

#include "tutti/bindings/memfs/binding.h"
#include "tutti/testing/mock_data_path.h"
#include "tutti/tutti_runtime/tutti_runtime_internal.h"

namespace {

int failures = 0;
#define CHECK(condition)                                                       \
    do { if (!(condition)) {                                                   \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,          \
                     #condition); ++failures;                                  \
    } } while (false)

struct State {
    int resource_initialize = 0;
    int resource_shutdown = 0;
    int resolver_destroy = 0;
    int datapath_initialize = 0;
    int datapath_destroy = 0;
    int storage_factory = 0;
    std::vector<tutti::TuttiRuntimeShutdownStage> stages;
    std::vector<std::string> events;
};

class FakeResource final : public tutti::Resource {
public:
    explicit FakeResource(std::shared_ptr<State> state,
                          std::string id = "memory")
        : state_(std::move(state)), id_(std::move(id)) {
        capabilities_.resource_type = "memory";
        capabilities_.provides_resolver_view = true;
        capabilities_.provides_datapath_view = true;
    }
    const tutti::ResourceCapabilities& capabilities() const override {
        return capabilities_;
    }
    tutti::Status initialize() override {
        state_->events.push_back("resource_initialize");
        ++state_->resource_initialize;
        state_value_ = tutti::ResourceState::INITIALIZED;
        return tutti::Status::Ok();
    }
    tutti::Status shutdown() override {
        if (state_value_ != tutti::ResourceState::STOPPED) {
            ++state_->resource_shutdown;
            state_value_ = tutti::ResourceState::STOPPED;
        }
        return tutti::Status::Ok();
    }
    tutti::ResourceInfo info() const override {
        tutti::ResourceInfo result;
        result.id = id_;
        result.type = "memory";
        result.state = state_value_;
        return result;
    }
private:
    std::shared_ptr<State> state_;
    std::string id_;
    tutti::ResourceCapabilities capabilities_;
    tutti::ResourceState state_value_ = tutti::ResourceState::CREATED;
};

class FakeResolver final : public tutti::StorageTargetResolver {
public:
    explicit FakeResolver(std::shared_ptr<State> state)
        : state_(std::move(state)) {}
    ~FakeResolver() override { ++state_->resolver_destroy; }
    tutti::Result<tutti::ResolvedTarget> resolve(
        std::string_view, const tutti::ResolveOptions&) override {
        return tutti::Result<tutti::ResolvedTarget>::Failure(
            tutti::Status(tutti::StatusCode::NOT_FOUND, "unused"));
    }
private:
    std::shared_ptr<State> state_;
};

class FakeDataPath final : public tutti::testing::MockDataPath {
public:
    FakeDataPath(std::shared_ptr<State> state, tutti::Status status,
                 std::int32_t accel_id)
        : state_(std::move(state)), status_(std::move(status)) {
        caps.bound_accel_id = accel_id;
    }
    ~FakeDataPath() override { ++state_->datapath_destroy; }
    void bind_accel_id(std::int32_t accel_id) { caps.bound_accel_id = accel_id; }
    tutti::Status initialize(const tutti::DataPathConfig&,
                             tutti::ResourceProvider&) override {
        ++state_->datapath_initialize;
        return status_;
    }
private:
    std::shared_ptr<State> state_;
    tutti::Status status_;
};

tutti::config::TuttiRuntimeSpec memfs_spec() {
    using namespace tutti::config;
    TuttiRuntimeSpec spec;
    spec.accelerator.profile = TUTTI_COMPILED_ACCELERATOR_PROFILE;
    spec.runtime.accel_id = TUTTI_DEFAULT_ACCEL_ID;
    spec.storage.resources.push_back(
        {"memory", "memory", MemoryResourceConfig{4096}});
    spec.storage.resolvers.push_back(
        {"resolver", "memfs", "memfs", MemfsResolverConfig{}});
    spec.storage.datapaths.push_back(
        {"datapath", "memfs", MemfsDataPathConfig{}});
    spec.storage.backends.push_back(
        {"backend", "memfs", "resolver", "datapath", "memory",
         MemfsBackendConfig{}});
    return spec;
}

tutti::tutti_runtime::TuttiRuntimeCreateInternalOptions options(
    const std::shared_ptr<State>& state) {
    tutti::tutti_runtime::TuttiRuntimeCreateInternalOptions result;
    result.backend_device_count = [] {
        return tutti::Result<int>::Success(1);
    };
    result.resource_factory = [state](const tutti::config::ResourceSpec&,
                                      std::int32_t) {
        std::unique_ptr<tutti::Resource> resource =
            std::make_unique<FakeResource>(state);
        return tutti::Result<std::unique_ptr<tutti::Resource>>::Success(
            std::move(resource));
    };
    result.shutdown_observer = [state](tutti::TuttiRuntimeShutdownStage stage) {
        state->stages.push_back(stage);
    };
    return result;
}

void install_backend_factory(
    tutti::tutti_runtime::TuttiRuntimeCreateInternalOptions& options,
    const std::shared_ptr<State>& state, tutti::Status initialize_status) {
    options.backend_factory =
        [state, initialize_status = std::move(initialize_status)](
            const tutti::tutti_runtime::BackendFactoryContext& context) {
            tutti::tutti_runtime::BackendFactoryProduct product;
            product.resolver = std::make_unique<FakeResolver>(state);
            product.datapath = std::make_unique<FakeDataPath>(
                state, initialize_status, context.runtime_accel_id);
            product.scheme = context.resolver.scheme;
            product.data_path_key = "memfs";
            product.data_path_config = tutti::DataPathConfig{"memfs"};
            product.resolver_type_id =
                std::string(tutti::binding::memfs::kResolverTypeId);
            product.payload_type_id =
                std::string(tutti::binding::memfs::kPayloadTypeId);
            product.payload_api_version =
                tutti::binding::memfs::kPayloadApiVersion;
            return tutti::Result<tutti::tutti_runtime::BackendFactoryProduct>::
                Success(std::move(product));
        };
}

tutti::Result<std::unique_ptr<tutti::StorageRuntime>>
create_fake_storage_runtime(
    const std::shared_ptr<State>& state, tutti::RuntimeConfig config,
    tutti::RuntimeComponents components) {
    ++state->storage_factory;
#if !defined(TUTTI_USE_HOST)
    // StorageRuntime performs its own device discovery. Keep this fake-only
    // lifecycle test independent of installed accelerator hardware after
    // TuttiRuntime's injected preflight has already been exercised.
    config.accel_id = -1;
    for (auto& binding : components.data_paths) {
        if (auto* datapath = dynamic_cast<FakeDataPath*>(binding.data_path)) {
            datapath->bind_accel_id(-1);
        }
    }
#endif
    return tutti::StorageRuntime::create(
        std::move(config), std::move(components));
}

} // namespace

int main() {
    {
        char path[] = "/tmp/tutti-runtime-path-XXXXXX";
        const int descriptor = ::mkstemp(path);
        CHECK(descriptor >= 0);
        if (descriptor >= 0) ::close(descriptor);
        std::ofstream stream(path);
        stream <<
            "accelerator: {profile: "
               << TUTTI_COMPILED_ACCELERATOR_PROFILE << "}\n"
            "runtime: {accel_id: " << TUTTI_DEFAULT_ACCEL_ID << "}\n"
            "storage:\n"
            "  resources: []\n"
            "  resolvers: []\n"
            "  datapaths: []\n"
            "  backends: []\n";
        stream.close();
        auto result = tutti::TuttiRuntime::create(path);
        CHECK(!result.ok());
        CHECK(result.status().message().find("exactly one backend") !=
              std::string::npos);
        ::unlink(path);
    }
#if !defined(TUTTI_USE_HOST)
    {
        auto spec = memfs_spec();
        spec.runtime.accel_id = 1;
        auto state = std::make_shared<State>();
        auto injected = options(state);
        injected.backend_device_count = [] {
            return tutti::Result<int>::Success(1);
        };
        auto result = tutti::testing::TuttiRuntimeTestAccess::create(
            std::move(spec), std::move(injected));
        CHECK(!result.ok());
        CHECK(result.status().code() == tutti::StatusCode::NOT_FOUND);
        CHECK(result.status().message().find("device count") !=
              std::string::npos);
        CHECK(state->resource_initialize == 0);
    }
#endif
    {
        auto state = std::make_shared<State>();
        auto injected = options(state);
        injected.resource_factory =
            [state](const tutti::config::ResourceSpec&, std::int32_t) {
                std::unique_ptr<tutti::Resource> resource =
                    std::make_unique<FakeResource>(state, "wrong-id");
                return tutti::Result<std::unique_ptr<tutti::Resource>>::Success(
                    std::move(resource));
            };
        auto result = tutti::testing::TuttiRuntimeTestAccess::create(
            memfs_spec(), std::move(injected));
        CHECK(!result.ok());
        CHECK(result.status().code() == tutti::StatusCode::INVALID_ARGUMENT);
        CHECK(state->resource_initialize == 0);
        CHECK(state->resource_shutdown == 1);
    }
    {
        auto state = std::make_shared<State>();
        auto injected = options(state);
        injected.public_options.spec_debug_logger =
            [state](std::string_view debug) {
                CHECK(debug.find("storage.backends[0].contract") !=
                      std::string_view::npos);
                state->events.push_back("spec_debug");
            };
        auto result = tutti::testing::TuttiRuntimeTestAccess::create(
            memfs_spec(), std::move(injected));
        CHECK(!result.ok());
        CHECK(result.status().code() == tutti::StatusCode::UNSUPPORTED);
        CHECK(state->resource_initialize == 1);
        CHECK(state->resource_shutdown == 1);
        const std::vector<std::string> expected_events{
            "spec_debug", "resource_initialize"};
        CHECK(state->events == expected_events);
    }
    {
        auto state = std::make_shared<State>();
        auto injected = options(state);
        install_backend_factory(
            injected, state,
            tutti::Status(tutti::StatusCode::NOT_READY, "injected failure"));
        injected.runtime_factory =
            [state](tutti::RuntimeConfig config,
                    tutti::RuntimeComponents components) {
                return create_fake_storage_runtime(
                    state,
                    std::move(config), std::move(components));
            };
        auto result = tutti::testing::TuttiRuntimeTestAccess::create(
            memfs_spec(), std::move(injected));
        if (!result.ok()) {
            std::fprintf(stderr, "initialize failure status: %s\n",
                         result.status().message().c_str());
        }
        CHECK(!result.ok());
        CHECK(result.status().code() == tutti::StatusCode::NOT_READY);
        CHECK(state->datapath_initialize == 1);
        CHECK(state->resolver_destroy == 1);
        CHECK(state->datapath_destroy == 1);
        CHECK(state->resource_shutdown == 1);
    }
    {
        auto state = std::make_shared<State>();
        auto injected = options(state);
        install_backend_factory(injected, state, tutti::Status::Ok());
        injected.runtime_factory =
            [state](tutti::RuntimeConfig config,
                    tutti::RuntimeComponents components) {
                return create_fake_storage_runtime(
                    state,
                    std::move(config), std::move(components));
            };
        auto result = tutti::testing::TuttiRuntimeTestAccess::create(
            memfs_spec(), std::move(injected));
        if (!result.ok()) {
            std::fprintf(stderr, "success path status: %s\n",
                         result.status().message().c_str());
        }
        CHECK(result.ok());
        if (result.ok()) {
            auto runtime = std::move(result).value();
            CHECK(runtime->state() == tutti::TuttiRuntimeState::RUNNING);
            CHECK(runtime->storage_runtime() != nullptr);
            CHECK(!tutti::testing::TuttiRuntimeTestAccess::validated_spec_debug(
                       *runtime).empty());
            CHECK(runtime->shutdown().ok());
            CHECK(runtime->state() == tutti::TuttiRuntimeState::STOPPED);
            CHECK(runtime->shutdown().ok());
            CHECK(state->resource_shutdown == 1);
            CHECK(state->resolver_destroy == 1);
            CHECK(state->datapath_destroy == 1);
            const std::vector<tutti::TuttiRuntimeShutdownStage> expected{
                tutti::TuttiRuntimeShutdownStage::STORAGE_RUNTIME_SHUTDOWN,
                tutti::TuttiRuntimeShutdownStage::STORAGE_RUNTIME_DESTROYED,
                tutti::TuttiRuntimeShutdownStage::RESOLVERS_DESTROYED,
                tutti::TuttiRuntimeShutdownStage::DATAPATHS_DESTROYED,
                tutti::TuttiRuntimeShutdownStage::RESOURCE_SHUTDOWN,
                tutti::TuttiRuntimeShutdownStage::COMPLETE,
            };
            CHECK(state->stages == expected);
        }
    }
    std::printf("TuttiRuntime tests: %s\n",
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

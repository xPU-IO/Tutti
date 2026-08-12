#include <tutti/tutti_runtime.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tutti/data_paths/data_path_factory.h>
#include <tutti/resolvers/resolver_factory.h>
#include <tutti/storage_runtime.h>

#include "tutti/testing/mock_data_path.h"
#include "tutti/tutti_runtime/tutti_runtime_internal.h"

namespace {

int failures = 0;
#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,    \
                         #condition);                                          \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

struct State {
    int resource_factory_calls = 0;
    int resource_initialize_calls = 0;
    int resource_shutdown_calls = 0;
    int resolver_factory_calls = 0;
    int resolver_destroy_calls = 0;
    int data_path_factory_calls = 0;
    int data_path_initialize_calls = 0;
    int data_path_shutdown_calls = 0;
    int data_path_destroy_calls = 0;
    int runtime_factory_calls = 0;
    tutti::StorageTargetResolver* resolver = nullptr;
    tutti::DataPath* data_path = nullptr;
    std::vector<std::string> events;
    std::vector<tutti::TuttiRuntimeShutdownStage> stages;
};

class FakeResource final : public tutti::Resource {
public:
    explicit FakeResource(std::shared_ptr<State> state)
        : state_(std::move(state)) {
        capabilities_.resource_type = "memory";
        capabilities_.provides_resolver_view = true;
        capabilities_.provides_datapath_view = true;
    }

    const tutti::ResourceCapabilities& capabilities() const override {
        return capabilities_;
    }

    tutti::Status initialize() override {
        state_->events.push_back("resource.initialize");
        ++state_->resource_initialize_calls;
        state_value_ = tutti::ResourceState::INITIALIZED;
        return tutti::Status::Ok();
    }

    tutti::Status shutdown() override {
        if (state_value_ != tutti::ResourceState::STOPPED) {
            state_->events.push_back("resource.shutdown");
            ++state_->resource_shutdown_calls;
            state_value_ = tutti::ResourceState::STOPPED;
        }
        return tutti::Status::Ok();
    }

    tutti::ResourceInfo info() const override {
        return {"memory-resource", "memory", state_value_};
    }

    tutti::Result<std::unique_ptr<const tutti::ResourceView>>
    get_resolver_view() const override {
        return tutti::Result<std::unique_ptr<const tutti::ResourceView>>::Failure(
            tutti::Status(tutti::StatusCode::UNSUPPORTED, "unused in test"));
    }

    tutti::Result<std::unique_ptr<const tutti::ResourceView>>
    get_datapath_view() const override {
        return tutti::Result<std::unique_ptr<const tutti::ResourceView>>::Failure(
            tutti::Status(tutti::StatusCode::UNSUPPORTED, "unused in test"));
    }

private:
    std::shared_ptr<State> state_;
    tutti::ResourceCapabilities capabilities_;
    tutti::ResourceState state_value_ = tutti::ResourceState::CREATED;
};

class FakeResolver final : public tutti::StorageTargetResolver {
public:
    explicit FakeResolver(std::shared_ptr<State> state)
        : state_(std::move(state)) {}

    ~FakeResolver() override {
        state_->events.push_back("resolver.destroy");
        ++state_->resolver_destroy_calls;
    }

    tutti::Result<tutti::ResolvedTarget> resolve(
        std::string_view, const tutti::ResolveOptions&) override {
        return tutti::Result<tutti::ResolvedTarget>::Failure(
            tutti::Status(tutti::StatusCode::NOT_FOUND, "unused in test"));
    }

private:
    std::shared_ptr<State> state_;
};

class FakeDataPath final : public tutti::testing::MockDataPath {
public:
    FakeDataPath(std::shared_ptr<State> state, std::int32_t accel_id)
        : state_(std::move(state)) {
        caps.bound_accel_id = accel_id;
    }

    ~FakeDataPath() override {
        state_->events.push_back("datapath.destroy");
        ++state_->data_path_destroy_calls;
    }

    tutti::Status initialize(const tutti::DataPathConfig& config,
                             tutti::ResourceProvider&) override {
        CHECK(config.name == "factory-config");
        state_->events.push_back("datapath.initialize");
        ++state_->data_path_initialize_calls;
        return tutti::Status::Ok();
    }

    tutti::Status shutdown(std::uint64_t) override {
        state_->events.push_back("datapath.shutdown");
        ++state_->data_path_shutdown_calls;
        return tutti::Status::Ok();
    }

    void bind_accel_id(std::int32_t accel_id) {
        caps.bound_accel_id = accel_id;
    }

private:
    std::shared_ptr<State> state_;
};

tutti::config::TuttiRuntimeSpec memfs_spec() {
    using namespace tutti::config;
    TuttiRuntimeSpec spec;
    spec.accelerator.profile = TUTTI_COMPILED_ACCELERATOR_PROFILE;
    spec.runtime.accel_id = TUTTI_DEFAULT_ACCEL_ID;
    spec.storage.resources.push_back(
        {"memory-resource", "memory", MemoryResourceConfig{4096}});
    spec.storage.resolvers.push_back(
        {"memfs-resolver", "memfs", "memfs", MemfsResolverConfig{}});
    spec.storage.datapaths.push_back(
        {"memfs-datapath", "memfs", MemfsDataPathConfig{}});
    spec.storage.backends.push_back(
        {"assembly-relation", "memfs", "memfs-resolver", "memfs-datapath",
         "memory-resource", MemfsBackendConfig{}});
    return spec;
}

tutti::Result<std::unique_ptr<tutti::StorageRuntime>> create_test_runtime(
    std::shared_ptr<State> state, tutti::RuntimeConfig config,
    tutti::RuntimeComponents components) {
    state->events.push_back("runtime.factory");
    ++state->runtime_factory_calls;
    CHECK(config.profile_name == TUTTI_COMPILED_ACCELERATOR_PROFILE);
    CHECK(config.accel_id == TUTTI_DEFAULT_ACCEL_ID);
    CHECK(components.resolvers.size() == 1);
    CHECK(components.data_paths.size() == 1);
    if (components.resolvers.size() == 1) {
        CHECK(components.resolvers.front().scheme == "memfs");
        CHECK(components.resolvers.front().resolver == state->resolver);
    }
    if (components.data_paths.size() == 1) {
        CHECK(components.data_paths.front().key == "memfs-datapath");
        CHECK(components.data_paths.front().data_path == state->data_path);
        CHECK(components.data_paths.front().config.name == "factory-config");
    }

#if !defined(TUTTI_USE_HOST)
    config.accel_id = -1;
    if (auto* data_path = dynamic_cast<FakeDataPath*>(state->data_path)) {
        data_path->bind_accel_id(-1);
    }
#endif
    return tutti::StorageRuntime::create(
        std::move(config), std::move(components));
}

tutti::tutti_runtime::TuttiRuntimeCreateInternalOptions injected_options(
    const std::shared_ptr<State>& state) {
    tutti::tutti_runtime::TuttiRuntimeCreateInternalOptions options;
    options.accelerator_device_count = [] {
        return tutti::Result<int>::Success(1);
    };
    options.resource_factory =
        [state](const tutti::config::ResourceSpec& spec,
                const tutti::resources::ResourceCreateContext& context) {
            state->events.push_back("resource.factory");
            ++state->resource_factory_calls;
            CHECK(spec.id == "memory-resource");
            CHECK(spec.type == "memory");
            CHECK(context.runtime_accel_id == TUTTI_DEFAULT_ACCEL_ID);
            std::unique_ptr<tutti::Resource> resource =
                std::make_unique<FakeResource>(state);
            return tutti::Result<std::unique_ptr<tutti::Resource>>::Success(
                std::move(resource));
        };
    options.resolver_factory =
        [state](const tutti::config::ResolverSpec& spec,
                const tutti::resolvers::ResolverCreateContext& context) {
            state->events.push_back("resolver.factory");
            ++state->resolver_factory_calls;
            CHECK(spec.id == "memfs-resolver");
            CHECK(spec.scheme == "memfs");
            CHECK(context.resource.info().id == "memory-resource");
            CHECK(context.resource.info().state ==
                  tutti::ResourceState::INITIALIZED);
            CHECK(context.relation.id == "assembly-relation");
            CHECK(context.relation.resolver == spec.id);
            CHECK(context.data_path_key == "memfs-datapath");
            auto resolver = std::make_unique<FakeResolver>(state);
            state->resolver = resolver.get();
            std::unique_ptr<tutti::StorageTargetResolver> result =
                std::move(resolver);
            return tutti::Result<
                std::unique_ptr<tutti::StorageTargetResolver>>::Success(
                    std::move(result));
        };
    options.data_path_factory =
        [state](const tutti::config::DataPathSpec& spec,
                const tutti::data_paths::DataPathCreateContext& context) {
            state->events.push_back("datapath.factory");
            ++state->data_path_factory_calls;
            CHECK(spec.id == "memfs-datapath");
            CHECK(context.resource.info().id == "memory-resource");
            CHECK(context.resource.info().state ==
                  tutti::ResourceState::INITIALIZED);
            CHECK(context.relation.id == "assembly-relation");
            CHECK(context.relation.datapath == spec.id);
            CHECK(context.runtime_accel_id == TUTTI_DEFAULT_ACCEL_ID);
            tutti::data_paths::CreatedDataPath result;
            result.instance =
                std::make_unique<FakeDataPath>(state, context.runtime_accel_id);
            state->data_path = result.instance.get();
            result.initialize_config = tutti::DataPathConfig{"factory-config"};
            return tutti::Result<tutti::data_paths::CreatedDataPath>::Success(
                std::move(result));
        };
    options.runtime_factory =
        [state](tutti::RuntimeConfig config,
                tutti::RuntimeComponents components) {
            return create_test_runtime(
                state, std::move(config), std::move(components));
        };
    options.shutdown_observer =
        [state](tutti::TuttiRuntimeShutdownStage stage) {
            state->stages.push_back(stage);
        };
    return options;
}

void check_successful_assembly() {
    auto state = std::make_shared<State>();
    auto result = tutti::testing::TuttiRuntimeTestAccess::create(
        memfs_spec(), injected_options(state));
    CHECK(result.ok());
    if (!result.ok()) return;

    auto runtime = std::move(result).value();
    CHECK(runtime->state() == tutti::TuttiRuntimeState::RUNNING);
    CHECK(runtime->storage_runtime() != nullptr);
    CHECK(runtime->resource_infos().size() == 1);
    CHECK(runtime->resource_info("memory-resource").ok());
    CHECK(tutti::testing::TuttiRuntimeTestAccess::resolver_count(*runtime) == 1);
    CHECK(tutti::testing::TuttiRuntimeTestAccess::data_path_count(*runtime) == 1);
    CHECK(tutti::testing::TuttiRuntimeTestAccess::resolver(
              *runtime, "memfs-resolver") == state->resolver);
    CHECK(tutti::testing::TuttiRuntimeTestAccess::data_path(
              *runtime, "memfs-datapath") == state->data_path);
    CHECK(tutti::testing::TuttiRuntimeTestAccess::resolver(
              *runtime, "assembly-relation") == nullptr);
    CHECK(tutti::testing::TuttiRuntimeTestAccess::data_path(
              *runtime, "assembly-relation") == nullptr);

    const std::vector<std::string> creation_events{
        "resource.factory", "resource.initialize", "resolver.factory",
        "datapath.factory", "runtime.factory", "datapath.initialize"};
    CHECK(state->events == creation_events);

    CHECK(runtime->shutdown().ok());
    CHECK(runtime->state() == tutti::TuttiRuntimeState::STOPPED);
    CHECK(runtime->shutdown().ok());
    CHECK(state->resource_shutdown_calls == 1);
    CHECK(state->resolver_destroy_calls == 1);
    CHECK(state->data_path_shutdown_calls == 1);
    CHECK(state->data_path_destroy_calls == 1);
    const std::vector<std::string> lifecycle_events{
        "resource.factory", "resource.initialize", "resolver.factory",
        "datapath.factory", "runtime.factory", "datapath.initialize",
        "datapath.shutdown", "resolver.destroy", "datapath.destroy",
        "resource.shutdown"};
    CHECK(state->events == lifecycle_events);
    const std::vector<tutti::TuttiRuntimeShutdownStage> expected_stages{
        tutti::TuttiRuntimeShutdownStage::STORAGE_RUNTIME_SHUTDOWN,
        tutti::TuttiRuntimeShutdownStage::STORAGE_RUNTIME_DESTROYED,
        tutti::TuttiRuntimeShutdownStage::RESOLVERS_DESTROYED,
        tutti::TuttiRuntimeShutdownStage::DATAPATHS_DESTROYED,
        tutti::TuttiRuntimeShutdownStage::RESOURCE_SHUTDOWN,
        tutti::TuttiRuntimeShutdownStage::COMPLETE,
    };
    CHECK(state->stages == expected_stages);
}

void check_resolver_failure_rolls_back_resource() {
    auto state = std::make_shared<State>();
    auto options = injected_options(state);
    options.resolver_factory =
        [state](const tutti::config::ResolverSpec&,
                const tutti::resolvers::ResolverCreateContext&) {
            state->events.push_back("resolver.factory.failure");
            ++state->resolver_factory_calls;
            return tutti::Result<
                std::unique_ptr<tutti::StorageTargetResolver>>::Failure(
                    tutti::Status(tutti::StatusCode::NOT_READY,
                                  "resolver unavailable"));
        };
    auto result = tutti::testing::TuttiRuntimeTestAccess::create(
        memfs_spec(), std::move(options));
    CHECK(!result.ok());
    CHECK(result.status().code() == tutti::StatusCode::NOT_READY);
    CHECK(state->data_path_factory_calls == 0);
    CHECK(state->runtime_factory_calls == 0);
    CHECK(state->resource_shutdown_calls == 1);
}

void check_null_datapath_rolls_back_resolver_and_resource() {
    auto state = std::make_shared<State>();
    auto options = injected_options(state);
    options.data_path_factory =
        [state](const tutti::config::DataPathSpec&,
                const tutti::data_paths::DataPathCreateContext&) {
            state->events.push_back("datapath.factory.null");
            ++state->data_path_factory_calls;
            return tutti::Result<tutti::data_paths::CreatedDataPath>::Success(
                tutti::data_paths::CreatedDataPath{});
        };
    auto result = tutti::testing::TuttiRuntimeTestAccess::create(
        memfs_spec(), std::move(options));
    CHECK(!result.ok());
    CHECK(result.status().code() == tutti::StatusCode::INVALID_ARGUMENT);
    CHECK(result.status().message().find("DataPath factory returned null") !=
          std::string::npos);
    CHECK(state->resolver_destroy_calls == 1);
    CHECK(state->resource_shutdown_calls == 1);
    CHECK(state->runtime_factory_calls == 0);
}

void check_runtime_factory_throw_rolls_back_registries() {
    auto state = std::make_shared<State>();
    auto options = injected_options(state);
    options.runtime_factory =
        [state](tutti::RuntimeConfig, tutti::RuntimeComponents) ->
            tutti::Result<std::unique_ptr<tutti::StorageRuntime>> {
            state->events.push_back("runtime.factory.throw");
            ++state->runtime_factory_calls;
            throw std::runtime_error("injected runtime failure");
        };
    auto result = tutti::testing::TuttiRuntimeTestAccess::create(
        memfs_spec(), std::move(options));
    CHECK(!result.ok());
    CHECK(result.status().code() == tutti::StatusCode::INTERNAL);
    CHECK(result.status().message().find("StorageRuntime factory threw") !=
          std::string::npos);
    CHECK(state->data_path_initialize_calls == 0);
    CHECK(state->resolver_destroy_calls == 1);
    CHECK(state->data_path_destroy_calls == 1);
    CHECK(state->resource_shutdown_calls == 1);
}

void check_invalid_spec_stops_before_factories() {
    auto state = std::make_shared<State>();
    auto spec = memfs_spec();
    spec.storage.backends.front().resource = "missing";
    auto result = tutti::testing::TuttiRuntimeTestAccess::create(
        std::move(spec), injected_options(state));
    CHECK(!result.ok());
    CHECK(result.status().code() == tutti::StatusCode::INVALID_ARGUMENT);
    CHECK(state->resource_factory_calls == 0);
    CHECK(state->resolver_factory_calls == 0);
    CHECK(state->data_path_factory_calls == 0);
}

#if defined(TUTTI_USE_HOST)
void check_real_factories_assemble_memfs() {
    auto result = tutti::TuttiRuntime::create(memfs_spec());
    CHECK(result.ok());
    if (!result.ok()) return;
    auto runtime = std::move(result).value();
    CHECK(runtime->state() == tutti::TuttiRuntimeState::RUNNING);
    CHECK(runtime->resource_info("memory-resource").ok());
    CHECK(runtime->shutdown().ok());
}
#endif

} // namespace

int main() {
    check_invalid_spec_stops_before_factories();
    check_resolver_failure_rolls_back_resource();
    check_null_datapath_rolls_back_resolver_and_resource();
    check_runtime_factory_throw_rolls_back_registries();
    check_successful_assembly();
#if defined(TUTTI_USE_HOST)
    check_real_factories_assemble_memfs();
#endif
    std::printf("TuttiRuntime assembly tests: %s\n",
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

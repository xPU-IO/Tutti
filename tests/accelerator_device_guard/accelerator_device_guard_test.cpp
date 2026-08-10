#include <tutti/accelerator_device_guard.h>
#include <tutti/storage_runtime.h>
#include <tutti/testing/mock_data_path.h>

#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace {

int failures = 0;

#define CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            std::fprintf(stderr, "FAIL: %s\n", message); \
            ++failures; \
        } \
    } while (0)

void test_invalid_id() {
    tutti::DeviceGuard guard(-2);
    CHECK(!guard.ok(), "negative accelerator other than -1 is rejected");
    CHECK(guard.state() == tutti::DeviceGuardState::ENTER_FAILED,
          "invalid accelerator has deterministic enter state");
    CHECK(guard.status().code() == tutti::StatusCode::INVALID_ARGUMENT,
          "invalid accelerator reports INVALID_ARGUMENT");
}

void test_host_noop() {
#if defined(TUTTI_USE_HOST)
    tutti::DeviceGuard guard(0);
    CHECK(guard.ok(), "HOST guard is a no-op");
    CHECK(guard.host_noop(), "HOST guard identifies its no-op backend");
    CHECK(guard.previous_accel_id() == -1,
          "HOST guard does not query a synthetic current device");
    CHECK(guard.restore().ok(), "HOST guard restores successfully");
    CHECK(guard.state() == tutti::DeviceGuardState::RESTORED,
          "HOST guard reaches RESTORED state");
#endif
}

void test_optional_noop() {
    tutti::DeviceGuard guard(-1);
    CHECK(guard.ok(), "optional -1 guard is active no-op");
    CHECK(guard.previous_accel_id() == -1,
          "optional guard does not query current device");
    CHECK(guard.restore().ok(), "optional guard restores successfully");
}

#if defined(TUTTI_USE_CUDA) || defined(TUTTI_USE_MUSA) || defined(TUTTI_USE_MACA)
void destructor_restore(int target) {
    tutti::DeviceGuard guard(target);
    CHECK(guard.ok(), "destructor fallback guard enters target");
}

void exception_restore(int target) {
    try {
        tutti::DeviceGuard guard(target);
        CHECK(guard.ok(), "exception cleanup guard enters target");
        throw 7;
    } catch (int) {
    }
}
#endif

#if defined(TUTTI_USE_CUDA)
struct GuardTestPayload {};
struct GuardTestLease {};

class GuardTestResolver final : public tutti::StorageTargetResolver {
public:
    tutti::Result<tutti::ResolvedTarget> resolve(
        std::string_view,
        const tutti::ResolveOptions&) override {
        return tutti::ResolvedTarget::make<GuardTestPayload, GuardTestLease>(
            "guard-test-resolver", "guard-test-payload", 1, 4096,
            "guard-test-path", std::make_shared<GuardTestPayload>(),
            std::make_shared<GuardTestLease>());
    }
};

class ObservingDataPath final : public tutti::testing::MockDataPath {
public:
    explicit ObservingDataPath(int expected) : expected_(expected) {
        caps.bound_accel_id = expected;
    }

    bool observed_only_expected() const { return observed_ok_.load(); }
    std::atomic<bool> fail_initialize{false};
    std::atomic<bool> fail_shutdown{false};

    tutti::Status initialize(const tutti::DataPathConfig& config,
                             tutti::ResourceProvider& resources) override {
        observe_();
        if (fail_initialize.load()) {
            return tutti::Status(tutti::StatusCode::DEVICE_ERROR,
                                 "injected initialize failure");
        }
        return MockDataPath::initialize(config, resources);
    }
    tutti::Status shutdown(std::uint64_t timeout_ns) override {
        observe_();
        if (fail_shutdown.load()) {
            return tutti::Status(tutti::StatusCode::DEVICE_ERROR,
                                 "injected shutdown failure");
        }
        return MockDataPath::shutdown(timeout_ns);
    }
    tutti::Result<tutti::DataPathTarget> open(
        const tutti::ResolvedTarget& target) override {
        observe_();
        return MockDataPath::open(target);
    }
    tutti::Status close(tutti::DataPathTarget target) override {
        observe_();
        return MockDataPath::close(target);
    }
    tutti::Result<tutti::RegistrationDomainKey> registration_domain(
        tutti::DataPathTarget target) const override {
        observe_();
        return MockDataPath::registration_domain(target);
    }
    tutti::Result<tutti::DataPathMemory> register_memory(
        const tutti::DataPathMemoryView& view,
        const tutti::RegistrationDomainKey& domain) override {
        observe_();
        return MockDataPath::register_memory(view, domain);
    }
    tutti::Status unregister_memory(tutti::DataPathMemory memory) override {
        observe_();
        return MockDataPath::unregister_memory(memory);
    }
    tutti::SubmitOutcome submit(const tutti::DataPathRequest* requests,
                                std::size_t count,
                                const tutti::HostSubmitContext& context) override {
        observe_();
        return MockDataPath::submit(requests, count, context);
    }
    tutti::Result<tutti::ProgressResult> progress(
        tutti::ProgressBudget budget) override {
        observe_();
        return MockDataPath::progress(budget);
    }
    tutti::Result<tutti::DataPathSnapshot> query(
        tutti::DataPathOp op) const override {
        observe_();
        return MockDataPath::query(op);
    }
    tutti::Status release(tutti::DataPathOp op) override {
        observe_();
        return MockDataPath::release(op);
    }

private:
    void observe_() const {
        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess || current != expected_) {
            observed_ok_.store(false);
        }
    }

    int expected_ = -1;
    mutable std::atomic<bool> observed_ok_{true};
};

bool current_is(int expected) {
    int current = -1;
    return cudaGetDevice(&current) == cudaSuccess && current == expected;
}

struct InjectedGuardBackendState {
    std::int32_t current = 1;
    int get_failures = 0;
    int set_failures = 0;
};

tutti::Status injected_get_device(void* opaque, std::int32_t& device) {
    auto& state = *static_cast<InjectedGuardBackendState*>(opaque);
    if (state.get_failures > 0) {
        --state.get_failures;
        return tutti::Status(tutti::StatusCode::DEVICE_ERROR,
                             "injected get-device failure");
    }
    device = state.current;
    return tutti::Status::Ok();
}

tutti::Status injected_set_device(void* opaque, std::int32_t device) {
    auto& state = *static_cast<InjectedGuardBackendState*>(opaque);
    if (state.set_failures > 0) {
        --state.set_failures;
        return tutti::Status(tutti::StatusCode::DEVICE_ERROR,
                             "injected set-device failure");
    }
    state.current = device;
    return tutti::Status::Ok();
}

void test_injected_guard_failures() {
    InjectedGuardBackendState get_failure;
    get_failure.get_failures = 1;
    tutti::testing::DeviceGuardBackend get_backend{
        &get_failure, &injected_get_device, &injected_set_device};
    auto failed_enter = tutti::testing::DeviceGuardTestAccess::create(
        0, get_backend);
    CHECK(!failed_enter.ok(), "injected current-device query fails enter");
    CHECK(failed_enter.state() == tutti::DeviceGuardState::ENTER_FAILED,
          "injected query failure has deterministic state");
    CHECK(failed_enter.status().code() == tutti::StatusCode::DEVICE_ERROR,
          "injected query failure has deterministic status");
    CHECK(failed_enter.diagnostic().find("injected get-device failure") !=
              std::string::npos,
          "injected query failure retains backend diagnostic");

    InjectedGuardBackendState restore_failure;
    tutti::testing::DeviceGuardBackend restore_backend{
        &restore_failure, &injected_get_device, &injected_set_device};
    {
        auto guard = tutti::testing::DeviceGuardTestAccess::create(
            0, restore_backend);
        CHECK(guard.ok() && restore_failure.current == 0,
              "injected backend enters target device");
        restore_failure.set_failures = 1;
        tutti::Status restored = guard.restore();
        CHECK(!restored.ok(), "injected restore failure is propagated");
        CHECK(guard.state() == tutti::DeviceGuardState::RESTORE_FAILED,
              "restore failure has deterministic state");
        CHECK(guard.diagnostic().find("restore current accelerator 1") !=
                  std::string::npos,
              "restore failure identifies the caller device");
    }
    CHECK(restore_failure.current == 1,
          "destructor retries restoration after explicit restore failure");
}

void test_runtime_boundaries_and_ownership() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::puts("SKIP: Runtime ownership test needs two accelerator devices");
        return;
    }

    GuardTestResolver resolver;
    ObservingDataPath data_path(0);
    tutti::RuntimeComponents components;
    components.resolvers.push_back({"guard", &resolver});
    components.data_paths.push_back({"guard-test-path", &data_path, {}});

    CHECK(cudaSetDevice(1) == cudaSuccess, "caller selects accelerator 1");
    auto runtime_result = tutti::StorageRuntime::create(
        tutti::RuntimeConfig{0, 64, TUTTI_COMPILED_ACCELERATOR_PROFILE},
        std::move(components));
    CHECK(runtime_result.ok(), "Runtime 0 initializes under caller device 1");
    if (!runtime_result.ok()) return;
    auto runtime = std::move(runtime_result).value();
    CHECK(current_is(1), "Runtime initialize restores caller device");

    auto target = runtime->open("guard://target", {});
    CHECK(target.ok(), "Runtime open succeeds");
    CHECK(current_is(1), "Runtime open restores caller device");

    char host_memory[64]{};
    auto host = runtime->register_memory(tutti::MemoryView{
        host_memory, sizeof(host_memory), tutti::MemoryKind::HOST,
        tutti::MemoryOwnership::CALLER_OWNED, -1, ""});
    CHECK(host.ok(), "Runtime registers host memory");
    if (!target.ok() || !host.ok()) return;

    tutti::IoRequest host_request{
        tutti::IoDirection::WRITE, host.value(), 0,
        target.value(), 0, sizeof(host_memory)};
    auto host_submit = runtime->submit(
        &host_request, 1,
        tutti::HostSubmitContext{tutti::ExecutionDomain::HOST_EXECUTION,
                                 -1, nullptr});
    CHECK(host_submit.io.has_value(), "Runtime submit succeeds");
    CHECK(current_is(1), "Runtime submit restores caller device");
    if (host_submit.io.has_value()) {
        auto snapshot = runtime->query(*host_submit.io);
        CHECK(snapshot.ok(), "Runtime progress/query succeeds");
        CHECK(current_is(1), "Runtime progress/query restores caller device");
        CHECK(runtime->release_io(*host_submit.io).ok(),
              "Runtime release succeeds");
        CHECK(current_is(1), "Runtime release restores caller device");
    }

    void* pointer0 = nullptr;
    void* pointer1 = nullptr;
    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;
    CHECK(cudaSetDevice(0) == cudaSuccess &&
          cudaMalloc(&pointer0, 4096) == cudaSuccess &&
          cudaStreamCreate(&stream0) == cudaSuccess,
          "allocate pointer/stream on accelerator 0");
    CHECK(cudaSetDevice(1) == cudaSuccess &&
          cudaMalloc(&pointer1, 4096) == cudaSuccess &&
          cudaStreamCreate(&stream1) == cudaSuccess,
          "allocate pointer/stream on accelerator 1");

    auto wrong_pointer = runtime->register_memory(tutti::MemoryView{
        pointer1, 4096, tutti::MemoryKind::DEVICE,
        tutti::MemoryOwnership::CALLER_OWNED, -1, ""});
    CHECK(!wrong_pointer.ok() &&
          wrong_pointer.status().code() == tutti::StatusCode::INVALID_ARGUMENT,
          "Runtime 0 rejects accelerator 1 pointer");
    CHECK(current_is(1), "pointer rejection preserves caller device");

    auto device_memory = runtime->register_memory(tutti::MemoryView{
        pointer0, 4096, tutti::MemoryKind::DEVICE,
        tutti::MemoryOwnership::CALLER_OWNED, -1, ""});
    CHECK(device_memory.ok(), "Runtime 0 accepts accelerator 0 pointer");
    if (device_memory.ok()) {
        tutti::IoRequest request{
            tutti::IoDirection::WRITE, device_memory.value(), 0,
            target.value(), 0, 4096};
        auto wrong_context = runtime->submit(
            &request, 1,
            tutti::HostSubmitContext{tutti::ExecutionDomain::DEVICE_EXECUTION,
                                     1, stream0});
        CHECK(!wrong_context.io.has_value(),
              "Runtime 0 rejects explicit accelerator 1 context");

        auto wrong_stream = runtime->submit(
            &request, 1,
            tutti::HostSubmitContext{tutti::ExecutionDomain::DEVICE_EXECUTION,
                                     0, stream1});
        CHECK(!wrong_stream.io.has_value(),
              "Runtime 0 rejects accelerator 1 stream");
        CHECK(current_is(1), "ownership rejection preserves caller device");
        CHECK(runtime->unregister_memory(device_memory.value()).ok(),
              "unregister accepted device pointer");
    }

    CHECK(runtime->unregister_memory(host.value()).ok(),
          "unregister host memory");
    CHECK(runtime->close(target.value()).ok(), "close target");
    CHECK(runtime->shutdown(100).ok(), "shutdown Runtime");
    CHECK(current_is(1), "Runtime shutdown restores caller device");
    CHECK(data_path.observed_only_expected(),
          "every DataPath boundary observed Runtime accelerator 0");

    cudaStreamDestroy(stream1);
    cudaFree(pointer1);
    CHECK(cudaSetDevice(0) == cudaSuccess, "select accelerator 0 for cleanup");
    cudaStreamDestroy(stream0);
    cudaFree(pointer0);
    CHECK(cudaSetDevice(1) == cudaSuccess, "restore caller after cleanup");
}

void test_runtime_failure_restoration() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::puts("SKIP: Runtime failure restoration needs two accelerators");
        return;
    }

    GuardTestResolver resolver;
    ObservingDataPath data_path(0);
    auto components = [&] {
        tutti::RuntimeComponents value;
        value.resolvers.push_back({"guard", &resolver});
        value.data_paths.push_back({"guard-test-path", &data_path, {}});
        return value;
    };

    CHECK(cudaSetDevice(1) == cudaSuccess, "failure test selects caller device 1");
    data_path.fail_initialize = true;
    auto failed_create = tutti::StorageRuntime::create(
        tutti::RuntimeConfig{0, 64, TUTTI_COMPILED_ACCELERATOR_PROFILE},
        components());
    CHECK(!failed_create.ok() &&
          failed_create.status().code() == tutti::StatusCode::DEVICE_ERROR,
          "Runtime propagates injected initialize failure");
    CHECK(current_is(1), "initialize failure restores caller device");

    data_path.fail_initialize = false;
    auto runtime_result = tutti::StorageRuntime::create(
        tutti::RuntimeConfig{0, 64, TUTTI_COMPILED_ACCELERATOR_PROFILE},
        components());
    CHECK(runtime_result.ok(), "failure test Runtime initializes");
    if (!runtime_result.ok()) return;
    auto runtime = std::move(runtime_result).value();
    auto target = runtime->open("guard://failure", {});
    char memory_bytes[64]{};
    auto memory = runtime->register_memory(tutti::MemoryView{
        memory_bytes, sizeof(memory_bytes), tutti::MemoryKind::HOST,
        tutti::MemoryOwnership::CALLER_OWNED, -1, ""});
    if (!target.ok() || !memory.ok()) return;

    tutti::IoRequest request{
        tutti::IoDirection::WRITE, memory.value(), 0,
        target.value(), 0, sizeof(memory_bytes)};
    const tutti::HostSubmitContext context{
        tutti::ExecutionDomain::HOST_EXECUTION, -1, nullptr};

    data_path.fail_submit = true;
    auto failed_submit = runtime->submit(&request, 1, context);
    CHECK(!failed_submit.io.has_value() &&
          failed_submit.status.code() == tutti::StatusCode::DEVICE_ERROR,
          "Runtime propagates injected submit failure");
    CHECK(current_is(1), "submit failure restores caller device");

    data_path.fail_submit = false;
    auto submitted = runtime->submit(&request, 1, context);
    CHECK(submitted.io.has_value(), "progress failure test submits an operation");
    if (submitted.io.has_value()) {
        data_path.fail_progress = true;
        auto snapshot = runtime->query(*submitted.io);
        CHECK(snapshot.ok() && snapshot.value().state == tutti::IoState::FAILED,
              "Runtime converts injected progress failure to terminal failure");
        CHECK(current_is(1), "progress failure restores caller device");

        // Complete the mock's underlying op so Runtime can release it after
        // the injected progress failure has made the Runtime IO terminal.
        data_path.fail_progress = false;
        tutti::DeviceGuard completion_guard(0);
        CHECK(completion_guard.ok(), "cleanup enters Runtime accelerator");
        auto progress = data_path.progress(tutti::ProgressBudget{16, 1000000});
        CHECK(progress.ok(), "cleanup completes injected-progress operation");
        CHECK(completion_guard.restore().ok(), "cleanup restores caller device");
        CHECK(runtime->release_io(*submitted.io).ok(),
              "release succeeds after progress failure cleanup");
        CHECK(current_is(1), "release after progress failure restores caller");
    }

    CHECK(runtime->unregister_memory(memory.value()).ok(),
          "failure test unregisters memory");
    CHECK(runtime->close(target.value()).ok(), "failure test closes target");
    data_path.fail_shutdown = true;
    auto failed_shutdown = runtime->shutdown(100);
    CHECK(!failed_shutdown.ok() &&
          failed_shutdown.code() == tutti::StatusCode::DEVICE_ERROR,
          "Runtime propagates injected shutdown failure");
    CHECK(current_is(1), "shutdown failure restores caller device");
    data_path.fail_shutdown = false;
    CHECK(runtime->shutdown(100).ok(), "Runtime retries shutdown successfully");
    CHECK(current_is(1), "successful shutdown retry restores caller device");
    CHECK(data_path.observed_only_expected(),
          "failure paths execute every DataPath call on Runtime device 0");
}

void test_two_runtime_threads() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::puts("SKIP: threaded Runtime test needs two accelerators");
        return;
    }

    std::atomic<bool> ok{true};
    auto worker = [&](int runtime_accel, int caller_accel) {
        GuardTestResolver resolver;
        ObservingDataPath data_path(runtime_accel);
        tutti::RuntimeComponents components;
        components.resolvers.push_back({"guard", &resolver});
        components.data_paths.push_back({"guard-test-path", &data_path, {}});

        if (cudaSetDevice(caller_accel) != cudaSuccess) {
            ok.store(false);
            return;
        }
        auto runtime_result = tutti::StorageRuntime::create(
            tutti::RuntimeConfig{runtime_accel, 64,
                                 TUTTI_COMPILED_ACCELERATOR_PROFILE},
            std::move(components));
        if (!runtime_result.ok() || !current_is(caller_accel)) {
            ok.store(false);
            return;
        }
        auto runtime = std::move(runtime_result).value();
        auto target = runtime->open("guard://thread", {});
        char bytes[64]{};
        auto memory = runtime->register_memory(tutti::MemoryView{
            bytes, sizeof(bytes), tutti::MemoryKind::HOST,
            tutti::MemoryOwnership::CALLER_OWNED, -1, ""});
        if (!target.ok() || !memory.ok() || !current_is(caller_accel)) {
            ok.store(false);
            return;
        }
        tutti::IoRequest request{
            tutti::IoDirection::WRITE, memory.value(), 0,
            target.value(), 0, sizeof(bytes)};
        auto submitted = runtime->submit(
            &request, 1,
            tutti::HostSubmitContext{
                tutti::ExecutionDomain::HOST_EXECUTION, -1, nullptr});
        if (!submitted.io.has_value() || !current_is(caller_accel)) {
            ok.store(false);
            return;
        }
        auto snapshot = runtime->query(*submitted.io);
        if (!snapshot.ok() || snapshot.value().state != tutti::IoState::COMPLETED ||
            !runtime->release_io(*submitted.io).ok() ||
            !runtime->unregister_memory(memory.value()).ok() ||
            !runtime->close(target.value()).ok() ||
            !runtime->shutdown(100).ok() || !current_is(caller_accel) ||
            !data_path.observed_only_expected()) {
            ok.store(false);
        }
    };

    std::thread runtime0(worker, 0, 1);
    std::thread runtime1(worker, 1, 0);
    runtime0.join();
    runtime1.join();
    CHECK(ok.load(),
          "Runtime 0/1 threads preserve independent caller current devices");
}
#endif

void test_cuda_threads() {
#if defined(TUTTI_USE_CUDA) || defined(TUTTI_USE_MUSA) || defined(TUTTI_USE_MACA)
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::puts("SKIP: fewer than two accelerator devices");
        return;
    }

    cudaSetDevice(1);
    int before = -1;
    CHECK(cudaGetDevice(&before) == cudaSuccess && before == 1,
          "caller starts on accelerator 1");
    {
        tutti::DeviceGuard guard(0);
        CHECK(guard.ok(), "guard enters accelerator 0");
        int inside = -1;
        CHECK(cudaGetDevice(&inside) == cudaSuccess && inside == 0,
              "guard selects target accelerator");
        CHECK(guard.restore().ok(), "explicit restore succeeds");
    }
    int after = -1;
    CHECK(cudaGetDevice(&after) == cudaSuccess && after == 1,
          "explicit restore returns caller to accelerator 1");

    destructor_restore(0);
    CHECK(cudaGetDevice(&after) == cudaSuccess && after == 1,
          "destructor fallback restores after early return");

    exception_restore(0);
    CHECK(cudaGetDevice(&after) == cudaSuccess && after == 1,
          "destructor fallback restores during exception cleanup");

    {
        tutti::DeviceGuard outer(0);
        CHECK(outer.ok(), "outer nested guard enters accelerator 0");
        tutti::DeviceGuard inner(1);
        CHECK(inner.ok(), "inner nested guard enters accelerator 1");
        CHECK(inner.restore().ok(), "inner nested guard restores");
        int nested = -1;
        CHECK(cudaGetDevice(&nested) == cudaSuccess && nested == 0,
              "inner guard restores outer target");
        CHECK(outer.restore().ok(), "outer nested guard restores");
    }
    CHECK(cudaGetDevice(&after) == cudaSuccess && after == 1,
          "nested guards restore original caller device");

    {
        tutti::DeviceGuard invalid(count);
        CHECK(!invalid.ok(), "out-of-range accelerator switch fails");
        CHECK(invalid.state() == tutti::DeviceGuardState::ENTER_FAILED,
              "failed switch has deterministic state");
        CHECK(invalid.status().code() == tutti::StatusCode::DEVICE_ERROR,
              "failed switch reports DEVICE_ERROR");
        CHECK(!invalid.diagnostic().empty(),
              "failed switch includes diagnostic");
    }
    CHECK(cudaGetDevice(&after) == cudaSuccess && after == 1,
          "failed switch preserves caller current device");

    std::atomic<bool> ok{true};
    auto worker = [&](int target, int caller) {
        cudaSetDevice(caller);
        tutti::DeviceGuard guard(target);
        int inside = -1;
        if (!guard.ok() || cudaGetDevice(&inside) != cudaSuccess ||
            inside != target) ok.store(false);
        if (!guard.restore().ok()) ok.store(false);
        int restored = -1;
        if (cudaGetDevice(&restored) != cudaSuccess || restored != caller) {
            ok.store(false);
        }
    };
    std::thread a(worker, 0, 1);
    std::thread b(worker, 1, 0);
    a.join();
    b.join();
    CHECK(ok.load(), "two threads keep independent current devices");
#endif
}

} // namespace

int main() {
    test_invalid_id();
    test_host_noop();
    test_optional_noop();
    test_cuda_threads();
#if defined(TUTTI_USE_CUDA)
    test_injected_guard_failures();
    test_runtime_boundaries_and_ownership();
    test_runtime_failure_restoration();
    test_two_runtime_threads();
#endif
    return failures == 0 ? 0 : 1;
}

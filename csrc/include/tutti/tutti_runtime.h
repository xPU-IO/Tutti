#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <tutti/config/tutti_runtime_spec.h>
#include <tutti/resource.h>
#include <tutti/status.h>

namespace tutti {

class DataPath;
class StorageRuntime;
class StorageTargetResolver;

namespace tutti_runtime {
struct TuttiRuntimeCreateInternalOptions;
}
namespace testing {
class TuttiRuntimeTestAccess;
}

enum class TuttiRuntimeState {
    INITIALIZING,
    RUNNING,
    SHUTTING_DOWN,
    STOPPED,
};

enum class TuttiRuntimeShutdownStage {
    STORAGE_RUNTIME_SHUTDOWN,
    STORAGE_RUNTIME_DESTROYED,
    RESOLVERS_DESTROYED,
    DATAPATHS_DESTROYED,
    RESOURCE_SHUTDOWN,
    COMPLETE,
};

struct TuttiRuntimeCreateOptions {
    std::function<void(std::string_view)> spec_debug_logger;
};

class TuttiRuntime {
public:
    static Result<std::unique_ptr<TuttiRuntime>> create(
        const std::string& config_path,
        TuttiRuntimeCreateOptions options = {});
    static Result<std::unique_ptr<TuttiRuntime>> create(
        config::TuttiRuntimeSpec spec,
        TuttiRuntimeCreateOptions options = {});

    ~TuttiRuntime();

    // shutdown() 与析构路径共用的 drain 超时。
    //
    // 有界阻塞：仅在确有在途 I/O 时才会真的等待（无在途时立即完成），而那种
    // 情形下等待几乎总是划算的——I/O 通常即将完成，等一会儿比泄漏整个对象图好。
    // 上限保证析构不会无限期挂住。
    static constexpr std::uint64_t kDefaultDrainTimeoutMs = 5000;

    // 排空在途 I/O 后关闭。
    //
    // **超时返回 TIMEOUT，并保留完整对象图**——StorageRuntime、resolver、
    // DataPath、Resource 与 NVMe lease 全部存活，状态留在 SHUTTING_DOWN。
    // 调用方可在在途 I/O 完成后直接重试，无需重建整个 runtime。
    //
    // 为什么必须保留：StorageRuntime 内部已有"宁可泄漏 memory 也不 UAF"的保护，
    // 但上层若在 TIMEOUT 后继续销毁 DataPath / queue group / DMA registration /
    // NVMe lease，就等于把那份保护抵消掉，GPU/NVMe 仍在执行时可能 UAF 或错误
    // DMA。销毁这些对象的唯一安全前提是确认没有非终态 I/O。
    Status shutdown(std::uint64_t drain_timeout_ms = kDefaultDrainTimeoutMs);

    TuttiRuntimeState state() const noexcept { return state_; }
    StorageRuntime* storage_runtime() noexcept { return runtime_.get(); }
    const StorageRuntime* storage_runtime() const noexcept {
        return runtime_.get();
    }
    Result<ResourceInfo> resource_info(std::string_view id) const;
    std::vector<ResourceInfo> resource_infos() const;

private:
    friend class testing::TuttiRuntimeTestAccess;

    TuttiRuntime();
    static Result<std::unique_ptr<TuttiRuntime>> create_with_options_(
        config::TuttiRuntimeSpec spec,
        tutti_runtime::TuttiRuntimeCreateInternalOptions options);

    void observe_(TuttiRuntimeShutdownStage stage) noexcept;
    Status adopt_resource_(std::string id,
                           std::unique_ptr<Resource> resource);
    const Resource* find_resource_(std::string_view id) const noexcept;
    Status register_datapath_(std::string id,
                              std::unique_ptr<DataPath> data_path,
                              DataPath*& borrowed);
    Status register_resolver_(std::string id,
                              std::unique_ptr<StorageTargetResolver> resolver,
                              StorageTargetResolver*& borrowed);
    Status set_storage_runtime_(std::unique_ptr<StorageRuntime> runtime);

    TuttiRuntimeState state_ = TuttiRuntimeState::INITIALIZING;
    std::unique_ptr<StorageRuntime> runtime_;
    std::unordered_map<std::string, std::unique_ptr<Resource>> resources_;
    std::vector<std::string> resource_initialization_order_;
    std::unordered_map<std::string, std::unique_ptr<StorageTargetResolver>>
        resolvers_;
    std::vector<std::string> resolver_registration_order_;
    std::unordered_map<std::string, std::unique_ptr<DataPath>> datapaths_;
    std::vector<std::string> datapath_registration_order_;
    std::function<Status(StorageRuntime&)> runtime_shutdown_hook_;
    std::function<void(TuttiRuntimeShutdownStage)> shutdown_observer_;
};

} // namespace tutti

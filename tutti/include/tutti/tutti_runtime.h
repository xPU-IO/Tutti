// tutti/include/tutti/tutti_runtime.h
//
// Application runtime ownership and lifecycle. Config parsing and loader
// assembly may populate this aggregate, but cleanup is implemented in the
// companion source file so parsing cannot accidentally own runtime policy.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <tutti/config/storage_config.h>
#include <tutti/status.h>

namespace tutti {

class DataPath;
class StorageRuntime;
class StorageTargetResolver;

namespace config {

struct RuntimeAcceleratorInfo {
    std::int32_t accel_id = -1;
    std::string view_root;
};

struct RuntimeNvmeResource {
    std::int32_t device_id = -1;
    std::vector<std::int32_t> allowed_accel_ids;
    bool available = false;
};

struct RuntimeNvmeSlice {
    std::int32_t device_id = -1;
    std::int32_t accel_id = -1;
    std::string pci_bdf;
    std::string chrdev_path;
    std::string block_path;
    std::string backing_mount_path;
    std::string view_path;
    std::uint32_t namespace_id = 0;
    std::uint32_t logical_block_size = 0;
    std::uint64_t bar0_size = 0;
    std::uint64_t max_data_size = 0;
    std::uint32_t granted_queues = 0;
    std::vector<std::int32_t> allowed_accel_ids;
};

struct RuntimeNvmeAllocation {
    std::string allocation_id;
    std::vector<RuntimeNvmeSlice> slices;
};

// NVMe loader seam. This is intentionally not a public ResourceProvider
// abstraction and does not add transport-specific fields to StorageRuntime.
class RuntimeResourceClient {
public:
    virtual ~RuntimeResourceClient() = default;

    virtual Result<std::vector<RuntimeAcceleratorInfo>>
    list_accelerators() = 0;

    virtual Result<std::vector<RuntimeNvmeResource>>
    list_nvme_resources() = 0;

    virtual Result<RuntimeNvmeAllocation> acquire_nvme_slices(
        std::int32_t accel_id,
        NvmeSelection selection,
        const std::vector<std::int32_t>& device_ids,
        std::int32_t queues_per_controller) = 0;

    virtual Status release(const std::string& allocation_id) = 0;
};

enum class TuttiRuntimeState {
    RUNNING,
    SHUTTING_DOWN,
    STOPPED,
};

enum class TuttiRuntimeShutdownStage {
    STORAGE_RUNTIME_SHUTDOWN,
    STORAGE_RUNTIME_DESTROYED,
    RESOLVERS_DESTROYED,
    DATAPATHS_DESTROYED,
    ALLOCATION_RELEASED,
    COMPLETE,
};

struct TuttiRuntimeInspection {
    TuttiRuntimeState state = TuttiRuntimeState::RUNNING;
    bool allocation_released = false;
    std::string allocation_id;
    std::vector<RuntimeNvmeSlice> allocation_slices;
    std::vector<std::string> resolver_schemes;
    std::vector<std::string> data_path_keys;
};

// Owned runtime bundle. The public vectors and allocation fields remain as a
// P2 compatibility surface; later phases can move them behind registries.
struct TuttiRuntime {
    std::unique_ptr<StorageRuntime> runtime;
    std::vector<std::unique_ptr<DataPath>> datapaths;
    std::vector<std::unique_ptr<StorageTargetResolver>> resolvers;
    std::unique_ptr<RuntimeResourceClient> resource_client;
    std::string allocation_id;
    std::vector<RuntimeNvmeSlice> allocation_slices;
    std::vector<std::string> resolver_schemes;
    std::vector<std::string> data_path_keys;

    ~TuttiRuntime();
    Status shutdown();

    TuttiRuntimeState state() const noexcept { return state_; }
    TuttiRuntimeInspection inspection() const;
    TuttiRuntimeInspection inspect() const { return inspection(); }

    // Transfer component ownership into the runtime and register its route
    // key. The returned pointer is borrowed and remains valid until shutdown.
    // Registration is rejected after shutdown has started.
    DataPath* register_datapath(std::unique_ptr<DataPath> data_path,
                                std::string key);
    StorageTargetResolver* register_resolver(
        std::unique_ptr<StorageTargetResolver> resolver,
        std::string scheme);

    // Test-only lifecycle injection. Ownership remains with TuttiRuntime.
    void set_runtime_shutdown_hook(
        std::function<Status(StorageRuntime&)> hook) {
        runtime_shutdown_hook_ = std::move(hook);
    }
    void set_shutdown_observer(
        std::function<void(TuttiRuntimeShutdownStage)> observer) {
        shutdown_observer_ = std::move(observer);
    }

private:
    void observe_(TuttiRuntimeShutdownStage stage) noexcept;

    TuttiRuntimeState state_ = TuttiRuntimeState::RUNNING;
    bool allocation_released_ = false;
    std::function<Status(StorageRuntime&)> runtime_shutdown_hook_;
    std::function<void(TuttiRuntimeShutdownStage)> shutdown_observer_;
};

} // namespace config
} // namespace tutti

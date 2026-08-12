// tutti/presets/local_nvme_preset.cpp
//
// Preset assembly layer implementation.
// Includes private headers to construct DataPaths + resolvers, returns
// public types (StorageRuntime + RuntimeTelemetry).

#include "tutti/presets/local_nvme.h"

#include <tutti/storage_runtime.h>
#include <tutti/cuda_like.h>

// Private headers — included here ONLY, never by consumer code.
#include "tutti/data_paths/local_nvme/local_nvme_data_path.h"
#include "tutti/data_paths/striped_local_nvme/striped_data_path.h"
#include "tutti/bindings/striped_local_nvme/binding.h"
#include <tutti/resolvers/local_file/resolver.h>
#include <tutti/resolvers/striped_file/resolver.h>

namespace tutti::presets {

RuntimeWithTelemetry make_local_nvme_runtime(const LocalNvmePreset& p) {
    namespace lnvme = tutti::data_paths::local_nvme;
    using tutti::resolvers::local_file::LocalFileResolver;
    using tutti::resolvers::local_file::BackingDeviceConfig;

    // Heap-allocate the DataPath + resolver — injected as raw pointers
    // into RuntimeComponents, lifetime tied to the runtime.
    auto* dp = new lnvme::LocalNvmeDataPath(
        p.device.ssnvme_path,
        p.device.bar0_size,
        p.gpu_id,
        p.num_queues,
        p.device.namespace_id,
        p.device.block_size,
        /*mdts_bytes=*/0,
        p.max_batch_entries,
        /*cq_poll_budget=*/0,
        p.handle_cache_capacity,
        p.prp_cache_capacity,
        p.max_in_flight_operations,
        /*max_batch_requests=*/0,
        /*max_request_bytes_override=*/0,
        /*handle_cache_l2_capacity=*/0,
        p.device.pci_bdf,
        p.threads_per_block);

    auto* resolver = new LocalFileResolver(
        p.device.pci_bdf,
        p.device.namespace_id,
        p.device.block_size,
        BackingDeviceConfig{p.device.backing_device, 0});

    RuntimeComponents comps;
    comps.resolvers.push_back({"file", resolver});
    comps.data_paths.push_back({"local-nvme-ext4", dp, DataPathConfig{"local_nvme"}});

    auto created = StorageRuntime::create({}, std::move(comps));
    if (!created.ok()) return {};

    RuntimeWithTelemetry result;
    result.runtime = std::move(created).value();
    result.telemetry = RuntimeTelemetry{
        [dp]() -> std::uint64_t { return dp->test_submit_call_count(); },
        [dp]() -> std::uint64_t { return dp->test_kernel_launch_count(); },
        [dp]() { dp->test_reset_submit_counters(); }
    };
    return result;
}

RuntimeWithTelemetry make_striped_nvme_runtime(const StripedNvmePreset& p) {
    namespace snvme = tutti::data_paths::striped_local_nvme;
    using tutti::resolvers::local_file::LocalFileResolver;
    using tutti::resolvers::local_file::BackingDeviceConfig;
    using tutti::resolvers::striped_file::StripedResolver;

    std::vector<snvme::DeviceDescriptor> sdevs;
    for (const auto& d : p.devices) {
        sdevs.push_back({d.ssnvme_path, d.bar0_size, d.namespace_id,
                         (std::uint32_t)p.gpu_id, p.num_queues, d.block_size,
                         d.pci_bdf});
    }

    auto* dp = new snvme::StripedDataPath(
        std::move(sdevs), (std::uint32_t)p.gpu_id,
        /*mdts_override=*/0, /*cq_poll_budget=*/0,
        p.max_batch_entries, p.max_in_flight_operations,
        /*handle_cache_capacity=*/0, p.prp_cache_capacity,
        p.threads_per_block);

    std::vector<std::unique_ptr<StorageTargetResolver>> sub_resolvers;
    for (const auto& d : p.devices) {
        sub_resolvers.push_back(std::make_unique<LocalFileResolver>(
            d.pci_bdf, d.namespace_id, d.block_size,
            BackingDeviceConfig{d.backing_device, 0}));
    }
    auto* resolver = new StripedResolver(std::move(sub_resolvers), p.stripe_unit);

    RuntimeComponents comps;
    comps.resolvers.push_back({"striped", resolver});
    comps.data_paths.push_back({std::string(tutti::binding::striped_local_nvme::kRecommendedDataPathKey),
                                 dp, DataPathConfig{"striped-nvme"}});

    auto created = StorageRuntime::create({}, std::move(comps));
    if (!created.ok()) return {};

    RuntimeWithTelemetry result;
    result.runtime = std::move(created).value();
    result.telemetry = RuntimeTelemetry{
        [dp]() -> std::uint64_t { return dp->test_submit_call_count(); },
        [dp]() -> std::uint64_t { return dp->test_kernel_launch_count(); },
        [dp]() { dp->test_reset_submit_counters(); }
    };
    return result;
}

} // namespace tutti::presets

#include "tutti/data_paths/data_path_factory.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tutti/bindings/memfs/memfs_data_path.h"
#if defined(TUTTI_DATA_PATH_FACTORY_HAS_LOCAL_NVME)
#include "tutti/data_paths/local_nvme/local_nvme_data_path.h"
#include "tutti/data_paths/striped_local_nvme/striped_data_path.h"
#endif
#include "tutti/resource/memory/memory_resource.h"
#include "tutti/resource/nvme/nvme_resource.h"

namespace tutti::data_paths {
namespace {

namespace memory_resource = tutti::resources::memory;
namespace nvme_resource = tutti::resources::nvme;

Status invalid(std::string message) {
    return Status(StatusCode::INVALID_ARGUMENT, std::move(message));
}

template <typename T>
Result<T> failure(Status status) {
    return Result<T>::Failure(std::move(status));
}

Status validate_context(const config::DataPathSpec& spec,
                        const DataPathCreateContext& context) {
    if (spec.id.empty() || spec.type.empty()) {
        return invalid("DataPathSpec ID and type must not be empty");
    }
    if (context.relation.datapath != spec.id) {
        return invalid("backend relation does not reference DataPathSpec");
    }
    const ResourceInfo info = context.resource.info();
    if (context.relation.resource != info.id) {
        return invalid("backend relation does not reference Resource instance");
    }
    return Status::Ok();
}

Result<std::unique_ptr<const ResourceView>> datapath_view(
    const DataPathCreateContext& context) {
    auto view = context.resource.get_datapath_view();
    if (!view.ok()) return view;
    if (!view.value()) {
        return failure<std::unique_ptr<const ResourceView>>(
            invalid("Resource returned null DataPath view"));
    }
    return view;
}

Result<CreatedDataPath> create_memfs(
    const config::DataPathSpec& spec,
    const DataPathCreateContext& context) {
    if (spec.type != "memfs" ||
        !std::holds_alternative<config::MemfsDataPathConfig>(spec.config) ||
        context.relation.contract != "memfs" ||
        !std::holds_alternative<config::MemfsBackendConfig>(
            context.relation.config)) {
        return failure<CreatedDataPath>(
            invalid("memfs DataPathSpec does not match backend relation"));
    }
    auto base_view = datapath_view(context);
    if (!base_view.ok()) return failure<CreatedDataPath>(base_view.status());
    const auto* view = dynamic_cast<const memory_resource::MemoryResourceView*>(
        base_view.value().get());
    if (view == nullptr) {
        return failure<CreatedDataPath>(
            invalid("memfs DataPath requires memory Resource view"));
    }

    CreatedDataPath result;
    result.instance = std::make_unique<binding::memfs::MemfsDataPath>();
    result.initialize_config = DataPathConfig{"memfs"};
    return Result<CreatedDataPath>::Success(std::move(result));
}

#if defined(TUTTI_DATA_PATH_FACTORY_HAS_LOCAL_NVME)
Result<std::uint32_t> checked_u32(std::uint64_t value,
                                  const char* field) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        return failure<std::uint32_t>(
            invalid(std::string(field) + " exceeds uint32_t"));
    }
    return Result<std::uint32_t>::Success(
        static_cast<std::uint32_t>(value));
}

const config::NvmeDataPathTuning& tuning(
    const config::DataPathSpec& spec) {
    if (const auto* local =
            std::get_if<config::LocalNvmeDataPathConfig>(&spec.config)) {
        return *local;
    }
    return std::get<config::StripedLocalNvmeDataPathConfig>(spec.config);
}

Result<CreatedDataPath> create_local_nvme(
    const config::DataPathSpec& spec,
    const DataPathCreateContext& context) {
    if (spec.type != "local-nvme" ||
        !std::holds_alternative<config::LocalNvmeDataPathConfig>(spec.config) ||
        context.relation.contract != "ext4-local-nvme" ||
        !std::holds_alternative<config::Ext4LocalNvmeBackendConfig>(
            context.relation.config)) {
        return failure<CreatedDataPath>(
            invalid("local-nvme DataPathSpec does not match backend relation"));
    }
    auto base_view = datapath_view(context);
    if (!base_view.ok()) return failure<CreatedDataPath>(base_view.status());
    const auto* view = dynamic_cast<const nvme_resource::NvmeDataPathResourceView*>(
        base_view.value().get());
    if (view == nullptr) {
        return failure<CreatedDataPath>(
            invalid("local-nvme DataPath requires NVMe DataPath view"));
    }
    if (view->slices.size() != 1) {
        return failure<CreatedDataPath>(
            invalid("local-nvme DataPath requires exactly one NVMe slice"));
    }

    const auto& slice = view->slices.front();
    const auto& config = tuning(spec);
    if (config.threads_per_block > slice.granted_queues) {
        return failure<CreatedDataPath>(invalid(
            "local-nvme threads_per_block exceeds granted queues"));
    }
    auto bar0_size = checked_u32(slice.bar0_size, "bar0_size");
    if (!bar0_size.ok()) return failure<CreatedDataPath>(bar0_size.status());
    auto max_batch_entries = checked_u32(
        tuning(spec).max_batch_entries, "max_batch_entries");
    if (!max_batch_entries.ok()) {
        return failure<CreatedDataPath>(max_batch_entries.status());
    }

    CreatedDataPath result;
    result.instance = std::make_unique<local_nvme::LocalNvmeDataPath>(
        slice.chrdev_path,
        bar0_size.value(),
        static_cast<std::uint32_t>(slice.accel_id),
        slice.granted_queues,
        slice.namespace_id,
        slice.logical_block_size,
        slice.max_data_size,
        max_batch_entries.value(),
        0,
        config.handle_cache_capacity,
        config.prp_cache_capacity,
        config.max_in_flight_operations,
        config.max_batch_entries,
        0,
        config.handle_cache_l2_capacity,
        slice.pci_bdf,
        config.threads_per_block);
    result.initialize_config = DataPathConfig{"local_nvme"};
    return Result<CreatedDataPath>::Success(std::move(result));
}

Result<CreatedDataPath> create_striped_local_nvme(
    const config::DataPathSpec& spec,
    const DataPathCreateContext& context) {
    if (spec.type != "striped-local-nvme" ||
        !std::holds_alternative<config::StripedLocalNvmeDataPathConfig>(
            spec.config) ||
        context.relation.contract != "striped-local-nvme") {
        return failure<CreatedDataPath>(
            invalid("striped-local-nvme DataPathSpec does not match backend relation"));
    }
    auto base_view = datapath_view(context);
    if (!base_view.ok()) return failure<CreatedDataPath>(base_view.status());
    const auto* view = dynamic_cast<const nvme_resource::NvmeDataPathResourceView*>(
        base_view.value().get());
    if (view == nullptr) {
        return failure<CreatedDataPath>(
            invalid("striped-local-nvme DataPath requires NVMe DataPath view"));
    }
    if (view->slices.size() < 2) {
        return failure<CreatedDataPath>(
            invalid("striped-local-nvme DataPath requires at least two NVMe slices"));
    }

    const auto& config = tuning(spec);
    auto max_batch_entries = checked_u32(
        config.max_batch_entries, "max_batch_entries");
    if (!max_batch_entries.ok()) {
        return failure<CreatedDataPath>(max_batch_entries.status());
    }
    auto max_in_flight = checked_u32(
        config.max_in_flight_operations, "max_in_flight_operations");
    if (!max_in_flight.ok()) {
        return failure<CreatedDataPath>(max_in_flight.status());
    }

    std::vector<striped_local_nvme::DeviceDescriptor> descriptors;
    descriptors.reserve(view->slices.size());
    std::uint64_t effective_mdts = 0;
    for (std::size_t index = 0; index < view->slices.size(); ++index) {
        const auto& slice = view->slices[index];
        if (config.threads_per_block > slice.granted_queues) {
            return failure<CreatedDataPath>(invalid(
                "striped-local-nvme threads_per_block exceeds granted queues "
                "for slice " + std::to_string(index)));
        }
        striped_local_nvme::DeviceDescriptor descriptor;
        descriptor.snvme_dev_path = slice.chrdev_path;
        descriptor.bar0_size = slice.bar0_size;
        descriptor.namespace_id = slice.namespace_id;
        descriptor.cuda_device = static_cast<std::uint32_t>(slice.accel_id);
        descriptor.num_user_queues = slice.granted_queues;
        descriptor.block_size = slice.logical_block_size;
        descriptor.controller_pci_addr = slice.pci_bdf;
        descriptors.push_back(std::move(descriptor));
        effective_mdts = effective_mdts == 0
            ? slice.max_data_size
            : std::min(effective_mdts, slice.max_data_size);
    }

    CreatedDataPath result;
    result.instance = std::make_unique<striped_local_nvme::StripedDataPath>(
        std::move(descriptors),
        static_cast<std::uint32_t>(view->slices.front().accel_id),
        effective_mdts,
        0,
        max_batch_entries.value(),
        max_in_flight.value(),
        config.handle_cache_capacity,
        config.prp_cache_capacity,
        config.threads_per_block);
    result.initialize_config = DataPathConfig{"striped-local-nvme"};
    return Result<CreatedDataPath>::Success(std::move(result));
}
#endif

} // namespace

Result<CreatedDataPath> create_data_path(
    const config::DataPathSpec& spec,
    const DataPathCreateContext& context) {
    Status status = validate_context(spec, context);
    if (!status.ok()) return failure<CreatedDataPath>(std::move(status));

    if (std::holds_alternative<config::MemfsDataPathConfig>(spec.config)) {
        return create_memfs(spec, context);
    }
#if defined(TUTTI_DATA_PATH_FACTORY_HAS_LOCAL_NVME)
    if (std::holds_alternative<config::LocalNvmeDataPathConfig>(spec.config)) {
        return create_local_nvme(spec, context);
    }
    if (std::holds_alternative<config::StripedLocalNvmeDataPathConfig>(
            spec.config)) {
        return create_striped_local_nvme(spec, context);
    }
#else
    if (std::holds_alternative<config::LocalNvmeDataPathConfig>(spec.config) ||
        std::holds_alternative<config::StripedLocalNvmeDataPathConfig>(
            spec.config)) {
        return failure<CreatedDataPath>(
            Status(StatusCode::UNSUPPORTED,
                   "local-NVMe DataPath is not available in this build"));
    }
#endif
    return failure<CreatedDataPath>(
        Status(StatusCode::UNSUPPORTED,
               "DataPathSpec configuration is not supported"));
}

} // namespace tutti::data_paths

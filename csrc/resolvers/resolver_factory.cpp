#include "csrc/resolvers/resolver_factory.h"

#include "csrc/common/backend_ids.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "csrc/resolvers/local_file/resolver.h"
#include "csrc/resolvers/local_file/multi_mount_resolver.h"

#include "csrc/resolvers/memfs/resolver.h"
#include "csrc/resource/memory/memory_resource.h"
#include "csrc/resource/nvme/nvme_resource.h"

namespace tutti::resolvers {
namespace {

namespace backend_ids = tutti::detail::backend_ids;
namespace local_file = tutti::resolvers::local_file;
namespace memory_resource = tutti::resources::memory;
namespace nvme_resource = tutti::resources::nvme;

Status invalid(std::string message) {
    return Status(StatusCode::INVALID_ARGUMENT, std::move(message));
}

template <typename T>
Result<T> failure(Status status) {
    return Result<T>::Failure(std::move(status));
}

Status validate_context(const config::ResolverSpec& spec,
                        const ResolverCreateContext& context) {
    if (spec.id.empty() || spec.type.empty()) {
        return invalid("ResolverSpec ID and type must not be empty");
    }
    if (context.data_path_key.empty()) {
        return invalid("resolver DataPath key must not be empty");
    }
    if (context.relation.resolver != spec.id) {
        return invalid("backend relation does not reference ResolverSpec");
    }
    if (context.relation.datapath != context.data_path_key) {
        return invalid("resolver DataPath key does not match backend relation");
    }
    const ResourceInfo info = context.resource.info();
    if (context.relation.resource != info.id) {
        return invalid("backend relation does not reference Resource instance");
    }
    return Status::Ok();
}

Result<std::unique_ptr<const ResourceView>> resolver_view(
    const ResolverCreateContext& context) {
    auto view = context.resource.get_resolver_view();
    if (!view.ok()) return view;
    if (!view.value()) {
        return failure<std::unique_ptr<const ResourceView>>(
            invalid("Resource returned null resolver view"));
    }
    return view;
}

Result<std::unique_ptr<StorageTargetResolver>> create_local_file(
    const config::ResolverSpec& spec,
    const ResolverCreateContext& context) {
    if (spec.type != tutti::detail::backend_ids::kExt4ResolverType ||
        !std::holds_alternative<config::LocalFileResolverConfig>(spec.config)) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            invalid("local-file ResolverSpec does not match backend relation"));
    }
    // Which relation this resolver serves decides the slice cardinality:
    // ext4-local-nvme is exactly one device; striped-local-nvme (the
    // multi-device contract, whose files rotate across mounts) is two or
    // more. Everything else is a configuration error.
    const bool single = context.relation.contract == backend_ids::kExt4Contract;
    const bool multi = context.relation.contract == backend_ids::kStripedContract;
    if (!single && !multi) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            invalid("local-file resolver does not serve contract " +
                    context.relation.contract));
    }
    if (single && !std::holds_alternative<config::Ext4LocalNvmeBackendConfig>(
                     context.relation.config)) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            invalid("local-file ResolverSpec does not match backend relation"));
    }
    if (multi && !std::holds_alternative<config::StripedLocalNvmeBackendConfig>(
                     context.relation.config)) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            invalid("local-file ResolverSpec does not match backend relation"));
    }
    auto base_view = resolver_view(context);
    if (!base_view.ok()) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            base_view.status());
    }
    const auto* view = dynamic_cast<const nvme_resource::NvmeResolverResourceView*>(
        base_view.value().get());
    if (view == nullptr) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            invalid("local-file resolver requires NVMe resolver view"));
    }
    if (view->slices.empty()) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            invalid("local-file resolver requires at least one NVMe slice"));
    }

    // One device: the original single-mount resolver.
    if (view->slices.size() == 1) {
        if (!single) {
            return failure<std::unique_ptr<StorageTargetResolver>>(
                invalid("single NVMe slice requires the ext4-local-nvme "
                        "contract"));
        }
        const auto& slice = view->slices.front();
        auto result = std::make_unique<local_file::LocalFileResolver>(
            slice.pci_bdf,
            slice.namespace_id,
            slice.logical_block_size,
            local_file::BackingDeviceConfig{slice.block_path, 0},
            local_file::kFiemapMaxExtentsPerCall,
            context.data_path_key);
        std::unique_ptr<StorageTargetResolver> resolver = std::move(result);
        return Result<std::unique_ptr<StorageTargetResolver>>::Success(
            std::move(resolver));
    }

    // Several devices: one "file" resolver dispatching by mount prefix. A
    // slot's file path already names the device it lives on (placement
    // rotated slots across mounts), so resolution is prefix matching plus
    // the full single-device pipeline per delegate.
    if (!multi) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            invalid("multiple NVMe slices require the striped-local-nvme "
                    "contract"));
    }
    std::vector<local_file::MultiMountLocalFileResolver::MountBinding> bindings;
    bindings.reserve(view->slices.size());
    for (const auto& slice : view->slices) {
        bindings.push_back({slice.backing_mount_path, slice.pci_bdf,
                            slice.namespace_id, slice.logical_block_size,
                            slice.block_path, context.data_path_key});
    }
    auto result = std::make_unique<local_file::MultiMountLocalFileResolver>(
        std::move(bindings));
    std::unique_ptr<StorageTargetResolver> resolver = std::move(result);
    return Result<std::unique_ptr<StorageTargetResolver>>::Success(
        std::move(resolver));
}

Result<std::unique_ptr<StorageTargetResolver>> create_memfs(
    const config::ResolverSpec& spec,
    const ResolverCreateContext& context) {
    if (spec.type != "memfs" ||
        !std::holds_alternative<config::MemfsResolverConfig>(spec.config) ||
        context.relation.contract != "memfs" ||
        !std::holds_alternative<config::MemfsBackendConfig>(
            context.relation.config)) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            invalid("memfs ResolverSpec does not match backend relation"));
    }
    auto base_view = resolver_view(context);
    if (!base_view.ok()) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            base_view.status());
    }
    const auto* view = dynamic_cast<const memory_resource::MemoryResourceView*>(
        base_view.value().get());
    if (view == nullptr) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            invalid("memfs resolver requires memory Resource view"));
    }
    auto result = std::make_unique<tutti::resolver::memfs::MemfsResolver>(
        view->capacity_bytes, context.data_path_key);
    std::unique_ptr<StorageTargetResolver> resolver = std::move(result);
    return Result<std::unique_ptr<StorageTargetResolver>>::Success(
        std::move(resolver));
}

} // namespace

Result<std::unique_ptr<StorageTargetResolver>> create_resolver(
    const config::ResolverSpec& spec,
    const ResolverCreateContext& context) {
    Status status = validate_context(spec, context);
    if (!status.ok()) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            std::move(status));
    }
    if (std::holds_alternative<config::LocalFileResolverConfig>(spec.config)) {
        return create_local_file(spec, context);
    }
    if (std::holds_alternative<config::MemfsResolverConfig>(spec.config)) {
        return create_memfs(spec, context);
    }
    return failure<std::unique_ptr<StorageTargetResolver>>(
        Status(StatusCode::UNSUPPORTED,
               "ResolverSpec configuration is not supported"));
}

} // namespace tutti::resolvers

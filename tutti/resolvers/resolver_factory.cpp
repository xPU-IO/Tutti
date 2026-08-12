#include "tutti/resolvers/resolver_factory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tutti/resolvers/local_file/resolver.h>
#include <tutti/resolvers/striped_file/resolver.h>

#include "tutti/resolvers/memfs/resolver.h"
#include "tutti/resource/memory/memory_resource.h"
#include "tutti/resource/nvme/nvme_resource.h"

namespace tutti::resolvers {
namespace {

namespace local_file = tutti::resolvers::local_file;
namespace memory_resource = tutti::resources::memory;
namespace nvme_resource = tutti::resources::nvme;
namespace striped_file = tutti::resolvers::striped_file;

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
    if (spec.type != "local-file" ||
        !std::holds_alternative<config::LocalFileResolverConfig>(spec.config) ||
        context.relation.contract != "ext4-local-nvme" ||
        !std::holds_alternative<config::Ext4LocalNvmeBackendConfig>(
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
    if (view->slices.size() != 1) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            invalid("local-file resolver requires exactly one NVMe slice"));
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

Result<std::unique_ptr<StorageTargetResolver>> create_striped_file(
    const config::ResolverSpec& spec,
    const ResolverCreateContext& context) {
    const auto* relation =
        std::get_if<config::StripedLocalNvmeBackendConfig>(
            &context.relation.config);
    if (spec.type != "striped-file" ||
        !std::holds_alternative<config::StripedFileResolverConfig>(spec.config) ||
        context.relation.contract != "striped-local-nvme" ||
        relation == nullptr) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            invalid("striped-file ResolverSpec does not match backend relation"));
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
            invalid("striped-file resolver requires NVMe resolver view"));
    }
    if (view->slices.size() < 2) {
        return failure<std::unique_ptr<StorageTargetResolver>>(
            invalid("striped-file resolver requires at least two NVMe slices"));
    }

    std::vector<std::unique_ptr<StorageTargetResolver>> shards;
    shards.reserve(view->slices.size());
    for (const auto& slice : view->slices) {
        shards.push_back(std::make_unique<local_file::LocalFileResolver>(
            slice.pci_bdf,
            slice.namespace_id,
            slice.logical_block_size,
            local_file::BackingDeviceConfig{slice.block_path, 0}));
    }
    auto result = std::make_unique<striped_file::StripedResolver>(
        std::move(shards), relation->stripe_unit, context.data_path_key);
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
    if (std::holds_alternative<config::StripedFileResolverConfig>(spec.config)) {
        return create_striped_file(spec, context);
    }
    if (std::holds_alternative<config::MemfsResolverConfig>(spec.config)) {
        return create_memfs(spec, context);
    }
    return failure<std::unique_ptr<StorageTargetResolver>>(
        Status(StatusCode::UNSUPPORTED,
               "ResolverSpec configuration is not supported"));
}

} // namespace tutti::resolvers

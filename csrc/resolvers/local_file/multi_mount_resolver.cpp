// csrc/resolvers/local_file/multi_mount_resolver.cpp

#include "csrc/resolvers/local_file/multi_mount_resolver.h"

#include <tutti/status.h>

namespace tutti::resolvers::local_file {

using tutti::Result;
using tutti::ResolvedTarget;
using tutti::StatusCode;
using tutti::Status;
using tutti::ResolveOptions;

MultiMountLocalFileResolver::MultiMountLocalFileResolver(
    std::vector<MountBinding> bindings)
    : bindings_(std::move(bindings)) {
    delegates_.reserve(bindings_.size());
    for (const auto& b : bindings_) {
        delegates_.push_back(std::make_unique<LocalFileResolver>(
            b.controller_pci_addr, b.namespace_id, b.block_size,
            BackingDeviceConfig{b.backing_device_path, 0},
            kFiemapMaxExtentsPerCall, b.data_path_key));
    }
}

int MultiMountLocalFileResolver::match_mount_(
    std::string_view path) const {
    int best = -1;
    std::size_t best_len = 0;
    for (std::size_t i = 0; i < bindings_.size(); ++i) {
        const std::string& mount = bindings_[i].mount_path;
        if (mount.empty()) continue;
        // A prefix match must end at a path separator: "/mnt/a" must not
        // match "/mnt/abc". (The mount itself never ends in '/'.)
        if (path.size() < mount.size() + 1) continue;
        if (path.compare(0, mount.size(), mount) != 0) continue;
        if (path[mount.size()] != '/') continue;
        if (mount.size() > best_len) {
            best = static_cast<int>(i);
            best_len = mount.size();
        }
    }
    return best;
}

Result<ResolvedTarget> MultiMountLocalFileResolver::resolve(
    std::string_view uri,
    const ResolveOptions& options) {

    // Scheme gate: same as the single-device resolver.
    if (options.scheme != kScheme) {
        return Result<ResolvedTarget>::Failure(
            Status(StatusCode::UNSUPPORTED,
                   "scheme '" + options.scheme +
                   "' != '" + std::string(kScheme) + "'"));
    }

    // Extract the path (same parse as LocalFileResolver; kept in sync by
    // construction: both take "file://<absolute-path>").
    constexpr std::string_view prefix = "file://";
    if (uri.size() < prefix.size() ||
        uri.substr(0, prefix.size()) != prefix) {
        return Result<ResolvedTarget>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "uri must start with 'file://': " + std::string(uri)));
    }
    const std::string_view path = uri.substr(prefix.size());
    if (path.empty() || path[0] != '/') {
        return Result<ResolvedTarget>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "uri path must be absolute: " + std::string(uri)));
    }

    const int idx = match_mount_(path);
    if (idx < 0) {
        return Result<ResolvedTarget>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "path is under no configured mount: " + std::string(path)));
    }

    // The delegate re-parses the URI and runs the full single-device
    // pipeline (regular-file check, st_dev identity, FIEMAP, payload).
    // Its data_path_key routes the target to the fused DataPath shared
    // by every device, so nothing here needs to know about devices.
    return delegates_[static_cast<std::size_t>(idx)]->resolve(uri, options);
}

} // namespace tutti::resolvers::local_file

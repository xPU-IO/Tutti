#pragma once

// csrc/resolvers/local_file/multi_mount_resolver.h
//
// MultiMountLocalFileResolver — one "file://" resolver serving several
// mounted NVMe devices.
//
// Why this exists: with rotating placement, an object is ONE file on ONE of
// N devices, and all N devices sit behind a single fused DataPath. The
// runtime maps one resolver per scheme, so instead of registering N
// resolvers under N schemes we register ONE resolver here that dispatches
// by path prefix: the file's path already says which mount -- and therefore
// which device -- it lives on.
//
// Each delegate is a full LocalFileResolver (with its own namespace
// identity and backing-device verification), so every guarantee the
// single-device resolver gives per file is preserved unchanged: regular
// file, st_dev matches the configured block device, FIEMAP collected
// fail-closed, device offsets carry the namespace base.
//
// Dispatch rule: longest mount prefix wins. A path that is not under any
// configured mount is INVALID_ARGUMENT -- it could not be DMA-read by any
// of this runtime's devices, so resolving it would be a lie.

#include "csrc/resolvers/local_file/resolver.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tutti::resolvers::local_file {

class MultiMountLocalFileResolver : public StorageTargetResolver {
public:
    // One device per entry: the mount directory and the LocalFileResolver
    // parameters for it. `data_path_key` routes resolution to the fused
    // multi-device DataPath (all delegates share it).
    struct MountBinding {
        std::string mount_path;  // absolute, no trailing slash
        std::string controller_pci_addr;
        std::uint32_t namespace_id = 1;
        std::uint32_t block_size = 4096;
        std::string backing_device_path;
        std::string data_path_key;
    };

    explicit MultiMountLocalFileResolver(
        std::vector<MountBinding> bindings);

    Result<ResolvedTarget> resolve(
        std::string_view uri,
        const ResolveOptions& options) override;

    const std::vector<MountBinding>& bindings() const { return bindings_; }

private:
    // Returns the index of the binding whose mount is the longest prefix of
    // `path`, or -1 when the path is under no mount.
    int match_mount_(std::string_view path) const;

    std::vector<MountBinding> bindings_;
    std::vector<std::unique_ptr<LocalFileResolver>> delegates_;
};

} // namespace tutti::resolvers::local_file

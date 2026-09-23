// csrc/storage_objects/slot_placement_policy.cpp

#include "csrc/storage_objects/slot_placement_policy.h"

namespace tutti::storage_objects {
namespace {

// Slot files are named by number, not by key. This is what makes the layout
// rename-free: binding a key to a slot is a metadata event (header plus
// checkpoint) rather than a directory operation, so a slot's path is stable
// across every reuse. The previous path-per-key design had to rename on each
// allocation, which invalidated path-keyed caches upstream and forced a fresh
// FIEMAP resolution every time.
std::string join(const std::string& dir, const std::string& leaf) {
    if (dir.empty()) return leaf;
    if (dir.back() == '/') return dir + leaf;
    return dir + "/" + leaf;
}

} // namespace

// -------------------------------------------------------------------------
// SingleFilePlacement
// -------------------------------------------------------------------------

SingleFilePlacement::SingleFilePlacement(std::string root)
    : root_(std::move(root)) {}

std::string SingleFilePlacement::slot_path(std::uint64_t slot) const {
    return join(join(root_, "slots"), std::to_string(slot) + ".obj");
}

Status SingleFilePlacement::paths_for_slot(std::uint64_t slot,
                                           std::vector<std::string>* out) const {
    if (out == nullptr) {
        return Status(StatusCode::INVALID_ARGUMENT, "out must not be null");
    }
    out->clear();
    out->push_back(slot_path(slot));
    return {};
}

std::uint64_t SingleFilePlacement::shard_file_bytes(
    std::uint64_t slot_bytes) const {
    // slot_bytes already includes the header prefix by the store's definition.
    return slot_bytes;
}

std::string SingleFilePlacement::uri_for_slot(std::uint64_t slot) const {
    // The local-file resolver expects "file://" followed by an absolute path,
    // and takes that path verbatim as the backing file.
    return "file://" + slot_path(slot);
}

// -------------------------------------------------------------------------
// RotatingFilePlacement
// -------------------------------------------------------------------------

RotatingFilePlacement::RotatingFilePlacement(std::vector<std::string> mounts,
                                             std::string subdir)
    : mounts_(std::move(mounts)), subdir_(std::move(subdir)) {}

std::string RotatingFilePlacement::slot_path(std::uint64_t slot) const {
    // The local-file resolver takes the URI's path verbatim as the backing
    // file, so this path IS the node: it is what materialisation creates and
    // what uri_for_slot() names. Divergence between the two would have
    // materialisation write one file while resolution maps another, and DMA
    // would end up pointed at unallocated extents.
    return join(join(mounts_[device_for_slot(slot)], subdir_),
                std::to_string(slot) + ".obj");
}

Status RotatingFilePlacement::paths_for_slot(
    std::uint64_t slot, std::vector<std::string>* out) const {
    if (out == nullptr) {
        return Status(StatusCode::INVALID_ARGUMENT, "out must not be null");
    }
    if (mounts_.empty()) {
        return Status(StatusCode::INVALID_ARGUMENT, "no mounts configured");
    }
    out->clear();
    out->push_back(slot_path(slot));
    return {};
}

std::string RotatingFilePlacement::uri_for_slot(std::uint64_t slot) const {
    // The local-file resolver expects "file://" followed by an absolute path
    // and takes that path verbatim as the backing file.
    return "file://" + slot_path(slot);
}

} // namespace tutti::storage_objects

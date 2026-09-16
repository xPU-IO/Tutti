// csrc/storage_objects/slot_placement_policy.cpp

#include "csrc/storage_objects/slot_placement_policy.h"

namespace tutti::storage_objects {
namespace {

// Slot files are named by number, not by key. This is what makes the layout
// rename-free: binding a key to a slot is a metadata event (header + checkpoint)
// rather than a directory operation, so a slot's path is stable across every
// reuse. The previous path-per-key design had to rename on every allocation,
// which invalidated path-keyed caches upstream.
std::string join(const std::string& dir, const std::string& leaf) {
    if (dir.empty()) return leaf;
    if (dir.back() == '/') return dir + leaf;
    return dir + "/" + leaf;
}

std::uint64_t round_up(std::uint64_t value, std::uint64_t multiple) noexcept {
    if (multiple == 0) return value;
    return ((value + multiple - 1) / multiple) * multiple;
}

} // namespace

// -------------------------------------------------------------------------
// SingleFilePlacement
// -------------------------------------------------------------------------

SingleFilePlacement::SingleFilePlacement(std::string root,
                                         std::string uri_scheme)
    : root_(std::move(root)), uri_scheme_(std::move(uri_scheme)) {}

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
    return uri_scheme_ + "://" + slot_path(slot);
}

// -------------------------------------------------------------------------
// StripedPlacement
// -------------------------------------------------------------------------

StripedPlacement::StripedPlacement(std::vector<std::string> mounts,
                                   std::uint64_t stripe_unit,
                                   std::string uri_scheme)
    : mounts_(std::move(mounts)),
      stripe_unit_(stripe_unit),
      uri_scheme_(std::move(uri_scheme)) {}

std::string StripedPlacement::shard_path(std::uint64_t slot,
                                         std::uint32_t shard) const {
    return join(join(mounts_[shard], "slots"),
                std::to_string(slot) + ".s" + std::to_string(shard));
}

Status StripedPlacement::paths_for_slot(std::uint64_t slot,
                                        std::vector<std::string>* out) const {
    if (out == nullptr) {
        return Status(StatusCode::INVALID_ARGUMENT, "out must not be null");
    }
    if (mounts_.empty()) {
        return Status(StatusCode::INVALID_ARGUMENT, "no mounts configured");
    }
    out->clear();
    out->reserve(mounts_.size());
    for (std::uint32_t shard = 0; shard < mounts_.size(); ++shard) {
        out->push_back(shard_path(slot, shard));
    }
    return {};
}

std::uint64_t StripedPlacement::shard_file_bytes(
    std::uint64_t slot_bytes) const {
    if (mounts_.empty()) return 0;
    const std::uint64_t header = ObjectHeaderLayout::kHeaderBytes;
    if (slot_bytes <= header) return header;
    const std::uint64_t payload = slot_bytes - header;
    const std::uint64_t shards = mounts_.size();
    // Round the per-shard share up to a whole stripe unit so the last stripe
    // round is not partial: a partial round would make a segment's tail land on
    // a shard that has no space reserved for it.
    const std::uint64_t share = round_up((payload + shards - 1) / shards,
                                         stripe_unit_ == 0 ? 1 : stripe_unit_);
    // Every shard reserves the header prefix, used only on shard 0, so all
    // shards are the same size and the payload offset is uniform.
    return header + share;
}

std::string StripedPlacement::uri_for_slot(std::uint64_t slot) const {
    // The striped resolver takes the shard paths and the stripe unit; encoding
    // them in the URI keeps the slot->target mapping a pure function of the slot
    // number, which is what lets targets be cached by slot.
    std::string uri = uri_scheme_ + "://";
    uri += "unit=" + std::to_string(stripe_unit_);
    for (std::uint32_t shard = 0; shard < mounts_.size(); ++shard) {
        uri += (shard == 0 ? "&paths=" : ",");
        uri += shard_path(slot, shard);
    }
    return uri;
}

bool StripedPlacement::geometry_valid(std::uint64_t slot_bytes) const {
    if (mounts_.empty()) return false;
    if (stripe_unit_ == 0) return false;
    const std::uint64_t header = ObjectHeaderLayout::kHeaderBytes;
    if (slot_bytes <= header) return false;

    const std::uint64_t payload = slot_bytes - header;
    const std::uint64_t shards = mounts_.size();

    // The payload must divide evenly across shards in whole stripe units.
    // Otherwise the final stripe round is short and a segment's bytes would map
    // past the end of some shard -- silent corruption rather than a clean error.
    if (payload % (stripe_unit_ * shards) != 0) return false;
    return true;
}

} // namespace tutti::storage_objects

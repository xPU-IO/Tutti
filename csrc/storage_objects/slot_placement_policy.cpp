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

std::uint64_t round_up(std::uint64_t value, std::uint64_t multiple) noexcept {
    if (multiple == 0) return value;
    return ((value + multiple - 1) / multiple) * multiple;
}

// The striped resolver requires the stripe unit to be a multiple of this.
constexpr std::uint64_t kMinStripeUnit = 4096;

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
// StripedPlacement
// -------------------------------------------------------------------------

StripedPlacement::StripedPlacement(std::vector<std::string> mounts,
                                   std::uint64_t stripe_unit)
    : mounts_(std::move(mounts)), stripe_unit_(stripe_unit) {}

std::string StripedPlacement::shard_path(std::uint64_t slot,
                                         std::uint32_t shard) const {
    // Must match exactly what the striped resolver derives from the URI:
    //     <mount_i>/striped/<name>.shard<i>
    // with <name> being the slot number. If these diverge, materialisation
    // writes one set of files while resolution maps another, and DMA ends up
    // pointed at unallocated extents.
    return join(join(mounts_[shard], "striped"),
                std::to_string(slot) + ".shard" + std::to_string(shard));
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

std::uint64_t StripedPlacement::payload_offset() const {
    const std::uint64_t round = stripe_unit_ * mounts_.size();
    if (round == 0) return ObjectHeaderLayout::kHeaderBytes;
    return round_up(ObjectHeaderLayout::kHeaderBytes, round);
}

std::uint64_t StripedPlacement::shard_file_bytes(
    std::uint64_t slot_bytes) const {
    if (mounts_.empty()) return 0;
    const std::uint64_t prefix = payload_offset();
    const std::uint64_t payload = slot_bytes > prefix ? slot_bytes - prefix : 0;
    const std::uint64_t shards = mounts_.size();
    const std::uint64_t unit = stripe_unit_ == 0 ? 1 : stripe_unit_;
    // The resolver derives the object's logical size from the shard FILES as
    //     N * floor(min_shard_bytes / unit) * unit
    // so the file size must survive that floor: sizing from the payload alone
    // leaves the prefix outside the logical space and the payload's last
    // segment lands past the end (OUT_OF_RANGE on real hardware -- how the
    // striped object layer failed its first 8-GPU run). Rounding the total up
    // to whole stripe rounds keeps file offsets, logical offsets and shard
    // sizes in one coordinate system.
    const std::uint64_t round = unit * shards;
    const std::uint64_t rounds = (prefix + payload + round - 1) / round;
    return rounds * unit;
}

std::string StripedPlacement::uri_for_slot(std::uint64_t slot) const {
    // Format fixed by the striped resolver:
    //     striped://<name>?devs=<m1,m2,...>&unit=<bytes>
    std::string uri = "striped://" + std::to_string(slot) + "?devs=";
    for (std::uint32_t shard = 0; shard < mounts_.size(); ++shard) {
        if (shard != 0) uri += ",";
        uri += mounts_[shard];
    }
    uri += "&unit=" + std::to_string(stripe_unit_);
    return uri;
}

bool StripedPlacement::geometry_valid(std::uint64_t slot_bytes) const {
    if (mounts_.empty()) return false;
    // The resolver rejects a stripe unit that is zero or not 4096-aligned;
    // checking here means a bad geometry fails at open() rather than on the
    // first IO.
    if (stripe_unit_ == 0 || stripe_unit_ % kMinStripeUnit != 0) return false;

    const std::uint64_t prefix = payload_offset();
    if (slot_bytes <= prefix) return false;

    const std::uint64_t payload = slot_bytes - prefix;
    const std::uint64_t shards = mounts_.size();

    // The payload must divide evenly across shards in whole stripe units.
    // Otherwise the final stripe round is short and a segment's bytes map past
    // the end of some shard -- silent corruption rather than a clean error.
    if (payload % (stripe_unit_ * shards) != 0) return false;
    return true;
}

} // namespace tutti::storage_objects

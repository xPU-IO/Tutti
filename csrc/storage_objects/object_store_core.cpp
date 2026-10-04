// csrc/storage_objects/object_store_core.cpp

#include "csrc/storage_objects/object_store_core.h"

#include <algorithm>
#include <chrono>
#include <utility>

#include "csrc/storage_objects/slot_media.h"

namespace tutti::storage_objects {
namespace {

std::string join(const std::string& dir, const std::string& leaf) {
    if (dir.empty()) return leaf;
    if (dir.back() == '/') return dir + leaf;
    return dir + "/" + leaf;
}

// Upper bound on entries a checkpoint container must hold. Sized to the slot
// count so a full store can always be checkpointed: a checkpoint that cannot
// represent the whole index would silently lose objects on restart.
std::uint64_t checkpoint_entry_capacity(std::uint64_t total_slots) {
    return total_slots;
}

// Generous bound on key length. Keys are caller-defined; the deployed shape is
// 18 bytes (16-byte chunk id plus a 2-byte layer index), so 256 leaves room
// without materially enlarging the region.
constexpr std::uint32_t kMaxKeyBytes = 256;

// Rejection returned by every mutating entry point of a read-only store. Not a
// silent no-op: a caller that believes it is writing must see the failure.
Status read_only_error(const char* operation) {
    return Status(StatusCode::UNSUPPORTED,
                  std::string(operation) + ": store is read-only");
}

} // namespace

ObjectStoreCore::ObjectStoreCore() = default;

ObjectStoreCore::~ObjectStoreCore() {
    // No implicit checkpoint: persisting during destruction would make an error
    // unreportable and could block an exit path. Callers checkpoint explicitly.
    (void)close();
}

std::string ObjectStoreCore::index_key(const ObjectKey& key) {
    return std::string(reinterpret_cast<const char*>(key.bytes.data()),
                       key.bytes.size());
}

std::uint64_t ObjectStoreCore::fingerprint_digest_locked() const {
    return object_identity(config_.namespace_fingerprint.data(),
                           config_.namespace_fingerprint.size());
}

std::string ObjectStoreCore::checkpoint_path_locked() const {
    return join(join(config_.uri, "meta"), "checkpoint.bin");
}

std::string ObjectStoreCore::residency_path_locked(std::uint32_t rank) const {
    return join(join(config_.uri, "residency"),
                "r" + std::to_string(rank) + ".bitmap");
}

// -------------------------------------------------------------------------
// open / close
// -------------------------------------------------------------------------

Status ObjectStoreCore::open(const StoreConfig& config) {
    std::lock_guard<std::mutex> guard(mutex_);

    if (config.uri.empty()) {
        return Status(StatusCode::INVALID_ARGUMENT, "uri must not be empty");
    }
    if (config.layout.segment_bytes == 0 || config.layout.segment_count == 0) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "layout must specify segment_bytes and segment_count");
    }
    // O_DIRECT operates in whole blocks, so a segment that is not a multiple of
    // 4096 would make some segment boundary unaligned and every IO past it
    // illegal. Rejecting here beats failing on the first write.
    if (config.layout.segment_bytes % 4096 != 0) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "segment_bytes must be 4096-aligned for O_DIRECT");
    }
    if (config.segment_file_slots == 0) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "segment_file_slots must be positive");
    }
    const std::uint64_t header_bytes = config.segment_header_bytes;
    if (header_bytes < ObjectHeaderLayout::kHeaderBytes ||
        header_bytes % 4096 != 0) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "segment_header_bytes must be at least 4096 and 4096-aligned");
    }
    const std::uint64_t payload_bytes = config.layout.payload_bytes();
    if (payload_bytes > UINT64_MAX - header_bytes ||
        config.segment_file_slots >
            UINT64_MAX / (header_bytes + payload_bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "segment_file_slots overflows file size");
    }

    if (opened_) {
        // Reopening the same namespace must agree on geometry and identity.
        // Fail-closed and preserve the data: there is deliberately no purge()
        // entry point, because destroying a populated cache is an operational
        // action, not a runtime capability.
        if (!(config.layout == config_.layout) ||
            config.segment_file_slots != config_.segment_file_slots ||
            config.segment_header_bytes != config_.segment_header_bytes ||
            config.namespace_fingerprint != config_.namespace_fingerprint ||
            config.uri != config_.uri) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "namespace fingerprint, layout or uri mismatch");
        }
        return {};
    }

    config_ = config;
    reclaimed_total_ = 0;
    reclaim_failures_ = 0;
    reclaim_stop_ = false;
    reclaim_pending_ = false;

    // A slot occupies its reserved prefix (object header) plus the payload.
    slot_bytes_ = header_bytes + payload_bytes;

    // Build placement before anything consults it.
    const Status backend = build_backend_locked();
    if (!backend.ok()) return backend;

    SpaceAllocatorConfig alloc_config;
    // capacity is a ceiling; the slot count follows from slot_bytes_. Callers
    // may express either in bytes or directly in slots -- the slot form is
    // preferred because only this layer knows the real per-slot space.
    alloc_config.capacity_bytes =
        config.capacity_slots > 0 ? config.capacity_slots * slot_bytes_
                                  : config.capacity_bytes;
    alloc_config.slot_bytes = slot_bytes_;
    if (!allocator_.configure(alloc_config)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "capacity_bytes too small to hold a single object");
    }
    if (config.segment_file_slots > UINT64_MAX / config.devices.size()) {
        return Status(StatusCode::INVALID_ARGUMENT, "segment group size overflows");
    }
    group_slots_ = config.segment_file_slots * config.devices.size();

    if (!config_.read_only) {
        const Status layout_status = ensure_layout_locked();
        if (!layout_status.ok()) return layout_status;
    }

    container_bytes_ = checkpoint_container_bytes(
        checkpoint_entry_capacity(allocator_.total_slots()), kMaxKeyBytes);
    const std::uint64_t region =
        container_bytes_ * CheckpointLayout::kContainerCount;
    if (!config_.read_only) {
        const Status meta_status =
            ensure_metadata_file(checkpoint_path_locked(), region);
        if (!meta_status.ok()) return meta_status;
    }

    // Recover before touching media: a warm pool is adopted as-is.
    const Status recovered = load_checkpoint_locked();
    if (!recovered.ok()) return recovered;

    // Writer: adopt every file group already on media; on a cold pool build the
    // first group here, so a store is writable the moment open() returns. This
    // is where service startup pays for its first files (~50 s for 8 ranks x 4
    // disks x 20 GiB). Later groups are built by precreate_step().
    if (!config_.read_only) {
        const Status reused = adopt_existing_groups_locked();
        if (!reused.ok()) return reused;
        if (precreated_ == 0) {
            std::vector<std::string> paths;
            const Status resolved = group_paths_locked(0, &paths);
            if (!resolved.ok()) return resolved;
            const Status made = materialise_group(paths);
            if (!made.ok()) return made;
            precreated_ = std::min(group_slots_, allocator_.total_slots());
        }
        allocator_.set_ready_limit(precreated_);
    }

    // Residency bitmaps last: their slot_count depends on the final geometry.
    if (config_.rank_count > 1) {
        const std::uint64_t slots = allocator_.total_slots();
        const std::uint64_t digest = fingerprint_digest_locked();
        if (!config_.read_only) {
            const Status dir = ensure_directory(join(config_.uri, "residency"));
            if (!dir.ok()) return dir;
        }

        // A bitmap that cannot be opened is not fatal: the cross-rank query
        // degrades to this rank's own view, which under-reports and is
        // therefore safe. Refusing to start would trade a performance loss for
        // an outage.
        if (config_.read_only) {
            (void)own_residency_.open_readonly(
                residency_path_locked(config_.rank_id), config_.rank_count,
                slots, digest);
        } else {
            (void)own_residency_.open_writable(
                residency_path_locked(config_.rank_id), config_.rank_id,
                config_.rank_count, slots, digest);
        }
        peer_residency_.clear();
        peer_residency_.resize(config_.rank_count);
        for (std::uint32_t rank = 0; rank < config_.rank_count; ++rank) {
            if (rank == config_.rank_id) continue;
            (void)peer_residency_[rank].open_readonly(residency_path_locked(rank),
                                                      config_.rank_count, slots,
                                                      digest);
        }
    }

    opened_ = true;
    if (config_.background_reclaim && !config_.read_only) {
        reclaim_pending_ = allocator_.stats().reclaiming_slots != 0;
        reclaim_worker_ = std::thread(&ObjectStoreCore::reclaim_worker_loop,
                                      this);
    }
    return {};
}

Status ObjectStoreCore::close() {
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!opened_) return {};
        reclaim_stop_ = true;
        reclaim_cv_.notify_all();
    }
    if (reclaim_worker_.joinable()) reclaim_worker_.join();

    std::lock_guard<std::mutex> guard(mutex_);
    if (!opened_) return {};
    // Flush a dirty checkpoint first. commit() writes the object header (the
    // durable statement) and marks the checkpoint dirty, so without this a
    // clean restart would report every object committed since the last
    // checkpoint as absent -- its header is on media, but the key set lives in
    // the checkpoint. close() is explicit and returns a Status, so unlike the
    // destructor it can report the failure.
    Status status;
    if (checkpoint_dirty_) {
        status = persist_checkpoint_locked();
    }
    // Flush the bitmap so a clean shutdown does not lose residency information
    // that the next start would otherwise have to rebuild.
    (void)own_residency_.sync();
    own_residency_.close();
    peer_residency_.clear();
    opened_ = false;
    reclaim_stop_ = false;
    reclaim_pending_ = false;
    return status;
}

Status ObjectStoreCore::build_backend_locked() {
    if (config_.devices.empty()) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "at least one device must be configured");
    }
    for (const StoreDevice& device : config_.devices) {
        // mount_path is what this layer needs. The remaining device identity
        // (controller, namespace, block size) is carried in StoreConfig for the
        // runtime's resolver, which is the component that must prove a file's
        // FIEMAP extents belong to the namespace it was configured for.
        if (device.mount_path.empty()) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "device needs a mount_path");
        }
    }

    // Slot numbers rotate over the mounts; every segment file is scoped to a
    // rank so ranks with the same slot numbers cannot overwrite one another.
    std::vector<std::string> mounts;
    mounts.reserve(config_.devices.size());
    for (const StoreDevice& device : config_.devices) {
        mounts.push_back(device.mount_path);
    }
    auto segmented = std::make_unique<FixedSegmentFilePlacement>(
        std::move(mounts), "r" + std::to_string(config_.rank_id) + "/segments",
        slot_bytes_, slot_bytes_ * config_.segment_file_slots,
        config_.segment_header_bytes);
    if (!segmented->geometry_valid()) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "invalid fixed segment file geometry");
    }
    placement_ = std::move(segmented);
    return {};
}

Status ObjectStoreCore::ensure_layout_locked() {
    const Status root = ensure_directory(config_.uri);
    if (!root.ok()) return root;
    const Status meta = ensure_directory(join(config_.uri, "meta"));
    if (!meta.ok()) return meta;

    // One segment directory per device: slots 0..N-1 land on devices 0..N-1.
    const std::uint64_t devices =
        std::min<std::uint64_t>(config_.devices.size(), allocator_.total_slots());
    for (std::uint64_t slot = 0; slot < devices; ++slot) {
        const std::string path = placement_->path_for_slot(slot);
        const std::size_t slash = path.find_last_of('/');
        if (slash == std::string::npos) continue;
        const Status dir = ensure_directory(path.substr(0, slash));
        if (!dir.ok()) return dir;
    }
    return {};
}

// -------------------------------------------------------------------------
// precreate
// -------------------------------------------------------------------------

// A file group is `group_slots_` consecutive slots starting at a multiple of
// group_slots_; slot group_start + d lives in the group's file on device d.
Status ObjectStoreCore::group_paths_locked(std::uint64_t slot,
                                          std::vector<std::string>* out) const {
    out->clear();
    const std::uint64_t group_start = (slot / group_slots_) * group_slots_;
    for (std::size_t device = 0; device < config_.devices.size(); ++device) {
        const std::uint64_t first = group_start + device;
        if (first >= allocator_.total_slots()) break;
        out->push_back(placement_->path_for_slot(first));
    }
    return {};
}

Status ObjectStoreCore::materialise_group(
    const std::vector<std::string>& paths) const {
    for (const std::string& path : paths) {
        const Status made = materialise_segment_file(path, placement_->file_bytes());
        if (!made.ok()) return made;
    }
    return {};
}

Status ObjectStoreCore::adopt_existing_groups_locked() {
    const std::uint64_t total = allocator_.total_slots();
    while (precreated_ < total) {
        std::vector<std::string> paths;
        const Status resolved = group_paths_locked(precreated_, &paths);
        if (!resolved.ok()) return resolved;
        for (const std::string& path : paths) {
            bool present = false;
            const Status probed = segment_file_is_ready(path, placement_->file_bytes(), &present);
            if (!probed.ok()) return probed;
            if (!present) return {};  // never publish a partially ready group
        }
        precreated_ = std::min(total, precreated_ + group_slots_);
    }
    return {};
}

ObjectPlacement ObjectStoreCore::placement_locked(
    std::uint64_t slot, std::uint64_t generation) const {
    ObjectPlacement p;
    // A URI, not a resolved target: the runtime owns resolution, and doing it
    // here too would pay the FIEMAP plus peer-memory mapping cost twice.
    p.uri = placement_->uri_for_slot(slot);
    // Skip the header so segment 0 begins exactly at p.offset.
    p.offset = placement_->payload_offset_for_slot(slot);
    p.payload_bytes = config_.layout.payload_bytes();
    p.slot = slot;
    p.generation = generation;
    return p;
}

// -------------------------------------------------------------------------
// queries
// -------------------------------------------------------------------------

bool ObjectStoreCore::contains(const ObjectKey& key) const {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = index_.find(index_key(key));
    return it != index_.end() && it->second.committed;
}

std::uint64_t ObjectStoreCore::contains_prefix(const ObjectKey* keys,
                                               std::size_t count) const {
    if (keys == nullptr) return 0;
    std::lock_guard<std::mutex> guard(mutex_);
    std::uint64_t hit = 0;
    for (std::size_t i = 0; i < count; ++i) {
        auto it = index_.find(index_key(keys[i]));
        if (it == index_.end() || !it->second.committed) break;
        ++hit;
    }
    return hit;
}

std::uint64_t ObjectStoreCore::contains_prefix_all_ranks(
    const ObjectKey* keys, std::size_t count) const {
    if (keys == nullptr) return 0;
    std::lock_guard<std::mutex> guard(mutex_);

    // Single rank, or an unusable own bitmap: fall back to this rank's view.
    // That can only under-report relative to true cross-rank residency, which
    // is the safe direction -- the cost is a recomputation.
    if (config_.rank_count <= 1 || !own_residency_.usable()) {
        std::uint64_t hit = 0;
        for (std::size_t i = 0; i < count; ++i) {
            auto it = index_.find(index_key(keys[i]));
            if (it == index_.end() || !it->second.committed) break;
            ++hit;
        }
        return hit;
    }

    std::vector<const ResidencyBitmap*> ranks;
    ranks.reserve(config_.rank_count);
    for (std::uint32_t rank = 0; rank < config_.rank_count; ++rank) {
        ranks.push_back(rank == config_.rank_id ? &own_residency_
                                                : &peer_residency_[rank]);
    }

    std::uint64_t hit = 0;
    for (std::size_t i = 0; i < count; ++i) {
        auto it = index_.find(index_key(keys[i]));
        if (it == index_.end() || !it->second.committed) break;
        if (!all_ranks_committed(ranks, it->second.slot)) break;
        ++hit;
    }
    return hit;
}

Result<ObjectPlacement> ObjectStoreCore::lookup(const ObjectKey& key) const {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = index_.find(index_key(key));
    if (it == index_.end() || !it->second.committed) {
        return Result<ObjectPlacement>::Failure(
            Status(StatusCode::NOT_FOUND, "object not committed"));
    }
    // Returning a URI rather than a resolved target means lookup works
    // immediately after a restart, with no resolution on this path at all.
    return Result<ObjectPlacement>::Success(
        placement_locked(it->second.slot, it->second.generation));
}

StoreUsage ObjectStoreCore::usage() const {
    std::lock_guard<std::mutex> guard(mutex_);
    const SpaceAllocatorStats s = allocator_.stats();
    StoreUsage u;
    u.capacity_bytes = s.total_slots * slot_bytes_;
    u.committed_bytes = s.committed_slots * slot_bytes_;
    u.reserved_bytes = s.reserved_slots * slot_bytes_;
    u.reclaiming_bytes = s.reclaiming_slots * slot_bytes_;
    u.usable_bytes = (s.free_slots + s.unmaterialised_slots) * slot_bytes_;
    u.free_slots = s.free_slots;
    u.reclaiming_slots = s.reclaiming_slots;
    u.reclaimed_total = reclaimed_total_;
    u.reclaim_failures = reclaim_failures_;

    // Per device: slots rotate over the devices, so slot s is on s % N.
    const std::uint64_t devices = config_.devices.size();
    if (devices > 0) {
        const std::uint64_t used = s.committed_slots + s.reserved_slots +
                                   s.reclaiming_slots;
        u.per_device_bytes.assign(devices, used / devices * slot_bytes_);
        for (std::uint64_t d = 0; d < used % devices; ++d) {
            u.per_device_bytes[d] += slot_bytes_;
        }
    }
    return u;
}

std::uint64_t ObjectStoreCore::ready_slots() const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!opened_) return 0;
    // precreated_ counts slots whose file group is ready on media. Do NOT
    // derive this from SpaceAllocatorStats::unmaterialised_slots: that is the
    // allocator's high-water mark of *handed out* slots, which is zero at bind
    // time, so the warm-up would see an empty pool and the first IO would pay
    // the whole peer-memory registration on the request path.
    return precreated_;
}

std::string ObjectStoreCore::slot_uri(std::uint64_t slot) const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!opened_ || placement_ == nullptr) return {};
    if (slot >= allocator_.total_slots()) return {};
    return placement_->uri_for_slot(slot);
}

std::uint64_t ObjectStoreCore::slot_generation(std::uint64_t slot) const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!opened_ || slot >= allocator_.total_slots()) return 0;
    return allocator_.generation_of(slot);
}

// -------------------------------------------------------------------------
// write path
// -------------------------------------------------------------------------

Result<ReserveOutcome> ObjectStoreCore::reserve(const ObjectKey* keys,
                                                std::size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (config_.read_only) {
        return Result<ReserveOutcome>::Failure(read_only_error("reserve"));
    }
    ReserveOutcome out;
    if (!opened_) {
        return Result<ReserveOutcome>::Failure(
            Status(StatusCode::NOT_READY, "store is not open"));
    }
    if (keys == nullptr || count == 0) {
        return Result<ReserveOutcome>::Success(std::move(out));
    }

    out.accepted.reserve(count);
    out.accepted_keys.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        const std::string ik = index_key(keys[i]);
        if (ik.empty() || ik.size() > kMaxKeyBytes) {
            // A key that cannot be checkpointed must not be admitted: it would
            // be lost on restart while appearing present until then.
            ++out.rejected_count;
            continue;
        }

        auto existing = index_.find(ik);
        if (existing != index_.end()) {
            // Already reserved or committed: hand back the existing placement
            // rather than consuming more space.
            out.accepted.push_back(placement_locked(existing->second.slot,
                                                    existing->second.generation));
            out.accepted_keys.push_back(keys[i]);
            continue;
        }

        SlotReservation got = allocator_.reserve(1);
        if (got.slots.empty()) {
            // Capacity exhausted, or the next file group is not published yet
            // (the allocator only hands out slots below precreated_; growth
            // belongs to precreate_step(), never to the forward thread). Either
            // way this is a steady state, not a fault: report the remainder and
            // let the caller trim its write batch. Never block, never raise.
            out.rejected_count += count - i;
            break;
        }
        const std::uint64_t slot = got.slots[0];
        const std::uint64_t generation = got.generations[0];

        Entry entry;
        entry.slot = slot;
        entry.generation = generation;
        entry.committed = false;
        index_.emplace(ik, entry);
        slot_owner_[slot] = ik;

        out.accepted.push_back(placement_locked(slot, generation));
        out.accepted_keys.push_back(keys[i]);
    }
    return Result<ReserveOutcome>::Success(std::move(out));
}

Status ObjectStoreCore::write_header_locked(std::uint64_t slot,
                                           const ObjectKey& key,
                                           std::uint64_t generation,
                                           std::uint64_t commit_seq) {
    std::vector<std::uint8_t> header(ObjectHeaderLayout::kHeaderBytes);
    if (!encode_object_header(header.data(), header.size(), key,
                              config_.layout.payload_bytes(), generation,
                              commit_seq)) {
        return Status(StatusCode::INVALID_ARGUMENT, "could not encode header");
    }
    return write_object_header(placement_->path_for_slot(slot),
                               placement_->header_offset_for_slot(slot),
                               header.data(), header.size());
}

Status ObjectStoreCore::commit(const ObjectKey* keys, std::size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (config_.read_only) return read_only_error("commit");
    if (!opened_) {
        return Status(StatusCode::NOT_READY, "store is not open");
    }
    if (keys == nullptr || count == 0) return {};

    std::vector<std::uint64_t> committed_slots;
    committed_slots.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        auto it = index_.find(index_key(keys[i]));
        if (it == index_.end()) continue;      // never reserved
        if (it->second.committed) continue;    // duplicate commit is harmless

        const std::uint64_t seq = ++commit_seq_;
        // Writing the header IS the commit. The caller guarantees the payload
        // is already durable; the ordering "payload fsync then header fsync" is
        // what makes a valid header imply durable data. One header per object,
        // not per segment, so a partially written object is uniformly invalid.
        const Status written =
            write_header_locked(it->second.slot, keys[i], it->second.generation, seq);
        if (!written.ok()) {
            // Leave the reservation in place: the caller may retry, and the
            // object is invalid on media either way.
            return written;
        }
        it->second.committed = true;
        it->second.commit_seq = seq;
        committed_slots.push_back(it->second.slot);

        // Residency is memory-only here; durability comes from kernel writeback
        // plus the periodic msync. A lost bit costs a recomputation.
        own_residency_.set(it->second.slot);
    }

    if (!committed_slots.empty()) {
        allocator_.commit(committed_slots.data(), committed_slots.size());
        checkpoint_dirty_ = true;
    }
    return {};
}

Status ObjectStoreCore::abort(const ObjectKey* keys, std::size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (config_.read_only) return read_only_error("abort");
    if (keys == nullptr || count == 0) return {};

    std::vector<std::uint64_t> slots;
    for (std::size_t i = 0; i < count; ++i) {
        const std::string ik = index_key(keys[i]);
        auto it = index_.find(ik);
        if (it == index_.end()) continue;
        // Committed keys are ignored so a stale abort can never destroy live
        // data.
        if (it->second.committed) continue;
        slots.push_back(it->second.slot);
        slot_owner_.erase(it->second.slot);
        index_.erase(it);
    }
    if (!slots.empty()) {
        allocator_.abort(slots.data(), slots.size());
        reclaim_pending_ = true;
        reclaim_cv_.notify_one();
    }
    return {};
}

// -------------------------------------------------------------------------
// release / pin
// -------------------------------------------------------------------------

Result<std::uint64_t> ObjectStoreCore::release(const ObjectKey* keys,
                                               std::size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (config_.read_only) {
        return Result<std::uint64_t>::Failure(read_only_error("release"));
    }
    if (keys == nullptr || count == 0) {
        return Result<std::uint64_t>::Success(0);
    }

    std::vector<std::uint64_t> slots;
    std::uint64_t released = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const std::string ik = index_key(keys[i]);
        auto it = index_.find(ik);
        if (it == index_.end() || !it->second.committed) continue;

        auto pin = pins_.find(ik);
        if (pin != pins_.end() && pin->second > 0) continue;  // pinned: leave it

        own_residency_.clear(it->second.slot);
        slots.push_back(it->second.slot);
        slot_owner_.erase(it->second.slot);
        index_.erase(it);
        ++released;
    }
    if (!slots.empty()) {
        allocator_.release(slots.data(), slots.size());
        checkpoint_dirty_ = true;
        reclaim_pending_ = true;
        reclaim_cv_.notify_one();
    }
    return Result<std::uint64_t>::Success(released);
}

Status ObjectStoreCore::pin(const ObjectKey* keys, std::size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (config_.read_only) return read_only_error("pin");
    if (keys == nullptr) return {};
    for (std::size_t i = 0; i < count; ++i) ++pins_[index_key(keys[i])];
    return {};
}

Status ObjectStoreCore::unpin(const ObjectKey* keys, std::size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (config_.read_only) return read_only_error("unpin");
    if (keys == nullptr) return {};
    for (std::size_t i = 0; i < count; ++i) {
        auto it = pins_.find(index_key(keys[i]));
        if (it == pins_.end()) continue;
        if (it->second > 0) --it->second;
        if (it->second == 0) pins_.erase(it);
    }
    return {};
}

// -------------------------------------------------------------------------
// reclamation
// -------------------------------------------------------------------------

std::uint64_t ObjectStoreCore::drain_reclaim(std::uint64_t max) {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<std::uint64_t> taken = allocator_.take_reclaimable(max);
    if (taken.empty()) return 0;

    std::vector<std::uint64_t> done;
    std::vector<std::uint64_t> failed;
    done.reserve(taken.size());

    for (std::uint64_t slot : taken) {
        // Zeroing the header is what actually invalidates the object: a zero
        // magic decodes as "never written" rather than as corruption, making a
        // reclaimed slot indistinguishable from a fresh one.
        const Status zeroed = zero_file_range(placement_->path_for_slot(slot),
                                              placement_->header_offset_for_slot(slot),
                                              slot_bytes_);
        if (!zeroed.ok()) {
            failed.push_back(slot);
            continue;
        }
        done.push_back(slot);
    }

    if (!failed.empty()) {
        // Requeue rather than drop: a transient IO error must not permanently
        // leak capacity.
        allocator_.requeue_reclaim(failed.data(), failed.size());
        reclaim_failures_ += failed.size();
    }
    if (!done.empty()) {
        allocator_.finish_reclaim(done.data(), done.size());
        reclaimed_total_ += done.size();
    }
    return done.size();
}

void ObjectStoreCore::reclaim_worker_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        reclaim_cv_.wait(lock, [this] {
            return reclaim_stop_ || reclaim_pending_;
        });
        if (reclaim_stop_) return;
        reclaim_pending_ = false;
        lock.unlock();
        (void)drain_reclaim(64);
        lock.lock();

        // Keep retrying a failed scrub, but avoid a tight loop when the media
        // is temporarily unavailable. A release racing with this check will
        // set reclaim_pending_ and notify the same worker.
        if (!reclaim_stop_ && allocator_.stats().reclaiming_slots != 0) {
            reclaim_pending_ = true;
            reclaim_cv_.wait_for(lock, std::chrono::milliseconds(10),
                                 [this] { return reclaim_stop_; });
        }
    }
}

std::uint64_t ObjectStoreCore::precreated_slots() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return precreated_;
}

std::uint64_t ObjectStoreCore::precreate_target() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return precreate_target_;
}

Result<std::uint64_t> ObjectStoreCore::precreate_step(std::uint64_t headroom) {
    std::uint64_t start = 0;
    std::uint64_t end = 0;
    std::vector<std::string> paths;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!opened_) {
            return Result<std::uint64_t>::Failure(
                Status(StatusCode::NOT_READY, "store is not open"));
        }
        if (config_.read_only) return Result<std::uint64_t>::Success(0);
        const std::uint64_t total = allocator_.total_slots();
        // Allocation frontier: slots the allocator has ever handed out.
        const std::uint64_t frontier = total - allocator_.stats().unmaterialised_slots;
        const std::uint64_t wanted =
            headroom >= total - frontier ? total : frontier + headroom;
        // Whole groups only: a partially published group would make slots look
        // ready on disks whose file does not exist yet.
        const std::uint64_t rounded =
            (wanted + group_slots_ - 1) / group_slots_ * group_slots_;
        precreate_target_ = std::min(total, rounded);
        if (precreate_busy_ || precreated_ >= precreate_target_) {
            return Result<std::uint64_t>::Success(0);
        }
        start = precreated_;
        end = std::min(total, start + group_slots_);
        const Status resolved = group_paths_locked(start, &paths);
        if (!resolved.ok()) return Result<std::uint64_t>::Failure(resolved);
        precreate_busy_ = true;
    }
    // Materialise outside the lock: a group is tens of GiB of zero-fill, and
    // the forward path must keep reserving slots that are already published.
    const Status made = materialise_group(paths);
    {
        std::lock_guard<std::mutex> guard(mutex_);
        precreate_busy_ = false;
        if (made.ok()) {
            precreated_ = end;
            allocator_.set_ready_limit(precreated_);
        }
    }
    if (!made.ok()) return Result<std::uint64_t>::Failure(made);
    return Result<std::uint64_t>::Success(end - start);
}

// -------------------------------------------------------------------------
// checkpoint and recovery
// -------------------------------------------------------------------------

// Read every container and report its sequence number, 0 meaning empty or
// unreadable. Reads whole containers rather than just headers: a header-only
// read cannot confirm the body's CRC, and a container whose body is torn must
// count as unusable, not as its recorded sequence.
void ObjectStoreCore::read_container_sequences_locked(
    std::uint64_t* sequences, std::vector<CheckpointEntry>* bodies) const {
    for (std::uint32_t i = 0; i < CheckpointLayout::kContainerCount; ++i) {
        sequences[i] = 0;
        std::vector<std::uint8_t> buf(static_cast<std::size_t>(container_bytes_));
        if (!read_checkpoint_container(
                checkpoint_path_locked(),
                checkpoint_container_offset(i, container_bytes_), buf.data(),
                buf.size()).ok()) {
            continue;
        }
        std::uint64_t seq = 0;
        std::vector<CheckpointEntry> entries;
        if (decode_checkpoint(buf.data(), buf.size(), &seq, &entries) ==
            CheckpointRejection::kNone) {
            sequences[i] = seq;
            if (bodies != nullptr) bodies[i] = std::move(entries);
        }
    }
}

Status ObjectStoreCore::persist_checkpoint_locked() {
    std::vector<CheckpointEntry> entries;
    entries.reserve(index_.size());
    for (const auto& pair : index_) {
        if (!pair.second.committed) continue;   // reservations are not durable
        CheckpointEntry e;
        e.key.bytes.assign(pair.first.begin(), pair.first.end());
        e.slot = pair.second.slot;
        e.payload_bytes = config_.layout.payload_bytes();
        e.generation = pair.second.generation;
        e.commit_seq = pair.second.commit_seq;
        entries.push_back(std::move(e));
    }

    // Which container to overwrite: always the OLDEST, so the newest valid state
    // is never the one at risk during the write. That is the entire basis of
    // crash atomicity here -- no rename, no temp file.
    std::uint64_t sequences[CheckpointLayout::kContainerCount] = {};
    read_container_sequences_locked(sequences, nullptr);

    const std::uint32_t container =
        next_checkpoint_container(sequences, CheckpointLayout::kContainerCount);
    const std::uint64_t highest = *std::max_element(
        sequences, sequences + CheckpointLayout::kContainerCount);
    const std::uint64_t next_seq = std::max(checkpoint_seq_, highest) + 1;

    const std::vector<std::uint8_t> image = encode_checkpoint(entries, next_seq);
    if (image.size() > container_bytes_) {
        return Status(StatusCode::INTERNAL,
                      "checkpoint image exceeds its container");
    }
    // Pad to the full container. Two reasons: the write stays 4096-aligned, and
    // a stale tail from a previously larger checkpoint is overwritten -- leaving
    // it would let an old, longer body be parsed behind the new one.
    std::vector<std::uint8_t> padded(static_cast<std::size_t>(container_bytes_), 0);
    std::copy(image.begin(), image.end(), padded.begin());

    const Status written = write_checkpoint_container(
        checkpoint_path_locked(),
        checkpoint_container_offset(container, container_bytes_), padded.data(),
        padded.size());
    if (!written.ok()) return written;

    checkpoint_seq_ = next_seq;
    checkpoint_dirty_ = false;
    return {};
}

Status ObjectStoreCore::checkpoint() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (config_.read_only) return read_only_error("checkpoint");
    if (!opened_) {
        return Status(StatusCode::NOT_READY, "store is not open");
    }
    const Status persisted = persist_checkpoint_locked();
    if (!persisted.ok()) return persisted;
    // Flush residency alongside, so a clean checkpoint and the bitmap agree.
    (void)own_residency_.sync();
    return {};
}

Status ObjectStoreCore::load_checkpoint_locked() {
    recovery_ = RecoveryReport{};

    std::uint64_t sequences[CheckpointLayout::kContainerCount] = {};
    std::vector<CheckpointEntry> bodies[CheckpointLayout::kContainerCount];
    read_container_sequences_locked(sequences, bodies);

    std::uint32_t chosen = 0;
    if (!select_checkpoint_container(sequences, CheckpointLayout::kContainerCount,
                                    &chosen)) {
        // No usable checkpoint. This is a cold start, or every container was
        // lost. Either way the store opens empty rather than refusing: the
        // checkpoint is an accelerator, not the truth, and objects can be
        // recovered later by scanning headers if desired.
        recovery_.checkpoint_available = false;
        return {};
    }

    recovery_.checkpoint_available = true;
    checkpoint_seq_ = sequences[chosen];
    const std::vector<CheckpointEntry>& entries = bodies[chosen];
    recovery_.checkpoint_entries = entries.size();

    // Everything the checkpoint offers is a CANDIDATE. Authority is the object
    // header: the checkpoint says "key K is at slot S", and recovery goes and
    // asks slot S whether it really holds K. A crash can leave the two
    // disagreeing, and only the header was written in the same causal chain as
    // the payload.
    std::vector<std::uint8_t> header(ObjectHeaderLayout::kHeaderBytes);
    std::vector<std::uint64_t> to_commit;
    to_commit.reserve(entries.size());
    std::uint64_t highest_seq = 0;

    for (const CheckpointEntry& entry : entries) {
        if (entry.slot >= allocator_.total_slots() ||
            entry.payload_bytes != config_.layout.payload_bytes()) {
            // Geometry changed under us: the recorded slot does not exist in
            // this configuration, or the object is a different size.
            ++recovery_.dropped_geometry;
            continue;
        }
        const std::string ik = index_key(entry.key);
        if (index_.count(ik) != 0 || slot_owner_.count(entry.slot) != 0) {
            // The same key twice, or two keys claiming one slot. Neither can be
            // trusted, so drop rather than pick arbitrarily.
            ++recovery_.dropped_duplicate;
            continue;
        }

        if (!read_object_header(placement_->path_for_slot(entry.slot),
                                placement_->header_offset_for_slot(entry.slot),
                                header.data(), header.size()).ok()) {
            ++recovery_.dropped_header;
            continue;
        }

        HeaderExpectation expect;
        expect.key = &entry.key;
        expect.payload_bytes = entry.payload_bytes;
        expect.generation = entry.generation;
        expect.check_generation = true;
        ObjectHeaderFields fields;
        if (decode_object_header(header.data(), header.size(), expect, &fields) !=
            HeaderRejection::kNone) {
            // Any disagreement drops the object: losing a valid object costs a
            // recomputation, accepting an invalid one corrupts results.
            ++recovery_.dropped_header;
            continue;
        }

        Entry live;
        live.slot = entry.slot;
        live.generation = entry.generation;
        live.commit_seq = fields.commit_seq;
        live.committed = true;
        index_.emplace(ik, live);
        slot_owner_[entry.slot] = ik;
        to_commit.push_back(entry.slot);
        highest_seq = std::max(highest_seq, fields.commit_seq);
        ++recovery_.accepted;
    }

    // precreated_ is not derived from recovered slots: open() adopts whole file
    // groups from media right after this, which keeps it group-aligned.
    commit_seq_ = std::max(commit_seq_, highest_seq);

    // Move the recovered slots through reserve->commit in the allocator so its
    // accounting matches the index. Slots are claimed in ascending order to
    // keep the allocator's own high-water mark consistent with the index.
    if (!to_commit.empty()) {
        std::sort(to_commit.begin(), to_commit.end());
        const std::uint64_t claim_through = to_commit.back() + 1;
        SlotReservation claimed = allocator_.reserve(claim_through);
        // Commit the recovered ones; abort the gaps so they return to the pool.
        std::vector<std::uint64_t> gaps;
        for (std::uint64_t slot : claimed.slots) {
            if (!std::binary_search(to_commit.begin(), to_commit.end(), slot)) {
                gaps.push_back(slot);
            }
        }
        allocator_.commit(to_commit.data(), to_commit.size());
        if (!gaps.empty()) {
            // Gap slots were never written by this recovery, but they have been
            // materialised, so aborting sends them through reclaim and they
            // come back zeroed and reusable.
            allocator_.abort(gaps.data(), gaps.size());
        }
    }

    // Rebuild residency from the VERIFIED set rather than trusting what is on
    // media. This is what keeps the bitmap's error one-sided: it can lag behind
    // reality, but it never claims an object that recovery rejected.
    if (own_residency_.usable()) {
        own_residency_.clear_all();
        for (const auto& pair : index_) {
            if (pair.second.committed) own_residency_.set(pair.second.slot);
        }
    }
    return {};
}

Result<std::vector<ObjectKey>> ObjectStoreCore::recover() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!opened_) {
        return Result<std::vector<ObjectKey>>::Failure(
            Status(StatusCode::NOT_READY, "store is not open"));
    }
    std::vector<ObjectKey> keys;
    keys.reserve(index_.size());
    for (const auto& pair : index_) {
        if (!pair.second.committed) continue;
        ObjectKey key;
        key.bytes.assign(pair.first.begin(), pair.first.end());
        keys.push_back(std::move(key));
    }
    return Result<std::vector<ObjectKey>>::Success(std::move(keys));
}

} // namespace tutti::storage_objects

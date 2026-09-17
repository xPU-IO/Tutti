// csrc/storage_objects/object_store_core.cpp

#include "csrc/storage_objects/object_store_core.h"

#include <algorithm>
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

} // namespace

ObjectStoreCore::ObjectStoreCore() = default;

ObjectStoreCore::ObjectStoreCore(
    std::unique_ptr<SlotPlacementPolicy> placement)
    : placement_(std::move(placement)) {}

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

    if (opened_) {
        // Reopening the same namespace must agree on geometry and identity.
        // Fail-closed and preserve the data: there is deliberately no purge()
        // entry point, because destroying a populated cache is an operational
        // action, not a runtime capability.
        if (!(config.layout == config_.layout) ||
            config.namespace_fingerprint != config_.namespace_fingerprint ||
            config.uri != config_.uri) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "namespace fingerprint, layout or uri mismatch");
        }
        return {};
    }

    config_ = config;

    // Build placement and resolver before anything consults them.
    const Status backend = build_backend_locked();
    if (!backend.ok()) return backend;

    slot_bytes_ = ObjectHeaderLayout::kHeaderBytes + config.layout.payload_bytes();
    shard_bytes_ = placement_->shard_file_bytes(slot_bytes_);

    // Striped geometry must divide into whole stripe rounds. Otherwise the last
    // round is short and a segment's tail maps past the end of a shard --
    // silent corruption rather than a clean failure.
    if (auto* striped = dynamic_cast<StripedPlacement*>(placement_.get())) {
        if (!striped->geometry_valid(slot_bytes_)) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "striped geometry does not divide into whole stripe "
                          "rounds for this payload size");
        }
    }

    SpaceAllocatorConfig alloc_config;
    // capacity_bytes is a ceiling on PAYLOAD-plus-header space; the slot count
    // is derived from it, so callers size the cache in the unit they think in.
    alloc_config.capacity_bytes = config.capacity_bytes;
    alloc_config.slot_bytes = slot_bytes_;
    alloc_config.prewarm_slots =
        config.prewarm_bytes == 0 ? 0 : config.prewarm_bytes / slot_bytes_;
    if (!allocator_.configure(alloc_config)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "capacity_bytes too small to hold a single object");
    }

    const Status layout_status = ensure_layout_locked();
    if (!layout_status.ok()) return layout_status;

    container_bytes_ = checkpoint_container_bytes(
        checkpoint_entry_capacity(allocator_.total_slots()), kMaxKeyBytes);
    const std::uint64_t region =
        container_bytes_ * CheckpointLayout::kContainerCount;
    const Status meta_status =
        ensure_metadata_file(checkpoint_path_locked(), region);
    if (!meta_status.ok()) return meta_status;

    // Recover before prewarming. Recovery may find slots already materialised
    // and in use, and prewarming first would be wasted work on a warm pool.
    const Status recovered = load_checkpoint_locked();
    if (!recovered.ok()) return recovered;

    // Prewarm the rest of the requested prefix. This is where a large
    // prewarm_bytes costs real time: extents must be genuinely written, at
    // roughly 225 MB/s per rank, so a terabyte is on the order of an hour.
    // capacity_bytes is only a ceiling; prewarm_bytes is what open() pays for.
    const Status warmed =
        materialise_through_locked(allocator_.prewarm_slots());
    if (!warmed.ok()) return warmed;

    // Residency bitmaps last: their slot_count depends on the final geometry.
    if (config_.rank_count > 1) {
        const std::uint64_t slots = allocator_.total_slots();
        const std::uint64_t digest = fingerprint_digest_locked();
        const Status dir = ensure_directory(join(config_.uri, "residency"));
        if (!dir.ok()) return dir;

        // A bitmap that cannot be opened is not fatal: the cross-rank query
        // degrades to this rank's own view, which under-reports and is
        // therefore safe. Refusing to start would trade a performance loss for
        // an outage.
        (void)own_residency_.open_writable(residency_path_locked(config_.rank_id),
                                          config_.rank_id, config_.rank_count,
                                          slots, digest);
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
    return {};
}

Status ObjectStoreCore::close() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!opened_) return {};
    // Flush the bitmap so a clean shutdown does not lose residency information
    // that the next start would otherwise have to rebuild.
    (void)own_residency_.sync();
    own_residency_.close();
    peer_residency_.clear();
    opened_ = false;
    return {};
}

Status ObjectStoreCore::build_backend_locked() {
    // Already injected: nothing to build. This is the path tests take.
    if (placement_ != nullptr) return {};

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

    if (config_.stripe_unit == 0) {
        if (config_.devices.size() != 1) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "single-file layout requires exactly one device; set "
                          "stripe_unit to stripe across several");
        }
        placement_ = std::make_unique<SingleFilePlacement>(
            config_.devices[0].mount_path);
        return {};
    }

    std::vector<std::string> mounts;
    mounts.reserve(config_.devices.size());
    for (const StoreDevice& device : config_.devices) {
        mounts.push_back(device.mount_path);
    }
    placement_ =
        std::make_unique<StripedPlacement>(mounts, config_.stripe_unit);
    return {};
}

Status ObjectStoreCore::ensure_layout_locked() {
    const Status root = ensure_directory(config_.uri);
    if (!root.ok()) return root;
    const Status meta = ensure_directory(join(config_.uri, "meta"));
    if (!meta.ok()) return meta;

    // Slot directories live under whichever mounts the placement policy uses,
    // so ask it rather than assuming a single root: for a striped layout the
    // slots are spread across devices.
    std::vector<std::string> paths;
    const Status probe = placement_->paths_for_slot(0, &paths);
    if (!probe.ok()) return probe;
    for (const std::string& path : paths) {
        const std::size_t slash = path.find_last_of('/');
        if (slash == std::string::npos) continue;
        const Status dir = ensure_directory(path.substr(0, slash));
        if (!dir.ok()) return dir;
    }
    return {};
}

// -------------------------------------------------------------------------
// materialisation
// -------------------------------------------------------------------------

Status ObjectStoreCore::materialise_through_locked(
    std::uint64_t slot_exclusive_end) {
    const std::uint64_t limit =
        std::min(slot_exclusive_end, allocator_.total_slots());
    while (materialised_ < limit) {
        std::vector<std::string> paths;
        const Status resolved = placement_->paths_for_slot(materialised_, &paths);
        if (!resolved.ok()) return resolved;
        const Status made = materialise_slot(paths, shard_bytes_);
        if (!made.ok()) return made;
        ++materialised_;
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
    p.offset = placement_->payload_offset();
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

    // Per device. The apportioning rule is implementation-private: striped
    // layouts are not required to be balanced, only to stay within each
    // device's available space.
    const std::uint32_t shards = placement_->shard_count();
    if (shards > 0) {
        const std::uint64_t used = s.committed_slots + s.reserved_slots +
                                   s.reclaiming_slots;
        u.per_device_bytes.assign(shards, used * shard_bytes_ / shards);
    }
    return u;
}

// -------------------------------------------------------------------------
// write path
// -------------------------------------------------------------------------

Result<ReserveOutcome> ObjectStoreCore::reserve(const ObjectKey* keys,
                                                std::size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
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
            // Capacity exhausted. A cache that never fills is over-provisioned,
            // so this is a steady state, not a fault: report the remainder and
            // let the caller trim its write batch. Never block, never raise.
            out.rejected_count += count - i;
            break;
        }
        const std::uint64_t slot = got.slots[0];
        const std::uint64_t generation = got.generations[0];

        // Materialise on demand when prewarm did not cover this slot. This is
        // the one place reserve can be slow (~44 ms per slot), which is why
        // prewarm_bytes should cover the working set.
        if (slot >= materialised_) {
            const Status made = materialise_through_locked(slot + 1);
            if (!made.ok()) {
                allocator_.abort(&slot, 1);
                ++out.rejected_count;
                continue;
            }
        }

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
    std::vector<std::string> paths;
    const Status resolved = placement_->paths_for_slot(slot, &paths);
    if (!resolved.ok()) return resolved;

    const std::uint32_t shard = placement_->header_shard();
    if (shard >= paths.size()) {
        return Status(StatusCode::INTERNAL, "header shard out of range");
    }

    std::vector<std::uint8_t> header(ObjectHeaderLayout::kHeaderBytes);
    if (!encode_object_header(header.data(), header.size(), key,
                              config_.layout.payload_bytes(), generation,
                              commit_seq)) {
        return Status(StatusCode::INVALID_ARGUMENT, "could not encode header");
    }
    return write_object_header(paths[shard],
                               placement_->header_offset_in_shard(),
                               header.data(), header.size());
}

Status ObjectStoreCore::commit(const ObjectKey* keys, std::size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
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
    if (!slots.empty()) allocator_.abort(slots.data(), slots.size());
    return {};
}

// -------------------------------------------------------------------------
// release / pin
// -------------------------------------------------------------------------

Result<std::uint64_t> ObjectStoreCore::release(const ObjectKey* keys,
                                               std::size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
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
    }
    return Result<std::uint64_t>::Success(released);
}

Status ObjectStoreCore::pin(const ObjectKey* keys, std::size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (keys == nullptr) return {};
    for (std::size_t i = 0; i < count; ++i) ++pins_[index_key(keys[i])];
    return {};
}

Status ObjectStoreCore::unpin(const ObjectKey* keys, std::size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
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
        std::vector<std::string> paths;
        if (!placement_->paths_for_slot(slot, &paths).ok()) {
            failed.push_back(slot);
            continue;
        }
        // Zeroing the header is what actually invalidates the object: a zero
        // magic decodes as "never written" rather than as corruption, making a
        // reclaimed slot indistinguishable from a fresh one.
        if (!zero_slot(paths, shard_bytes_).ok()) {
            failed.push_back(slot);
            continue;
        }
        done.push_back(slot);
    }

    if (!failed.empty()) {
        // Requeue rather than drop: a transient IO error must not permanently
        // leak capacity.
        allocator_.requeue_reclaim(failed.data(), failed.size());
    }
    if (!done.empty()) {
        allocator_.finish_reclaim(done.data(), done.size());
    }
    return done.size();
}

std::uint64_t ObjectStoreCore::materialised_slots() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return materialised_;
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
    std::uint64_t highest_slot = 0;

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

        std::vector<std::string> paths;
        if (!placement_->paths_for_slot(entry.slot, &paths).ok()) {
            ++recovery_.dropped_geometry;
            continue;
        }
        const std::uint32_t shard = placement_->header_shard();
        if (shard >= paths.size()) {
            ++recovery_.dropped_geometry;
            continue;
        }
        if (!read_object_header(paths[shard], placement_->header_offset_in_shard(),
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
        highest_slot = std::max(highest_slot, entry.slot + 1);
        ++recovery_.accepted;
    }

    // Recovered slots are already materialised on media; record that so
    // reserve() does not try to materialise them again.
    materialised_ = std::max(materialised_, highest_slot);
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

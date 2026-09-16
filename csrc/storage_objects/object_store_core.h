#pragma once

// csrc/storage_objects/object_store_core.h -- StorageObjectStore implementation.
//
// IMPLEMENTATION DETAIL. Consumers reach this only through
// <tutti/spi/storage_object_store.h> and create_storage_object_store().
//
// Assembles the six units of this layer into one store:
//
//   space_allocator        slot numbers, states, generations
//   slot_placement_policy  slot -> paths/URI (the only striping-aware seam)
//   slot_media             O_DIRECT materialisation and metadata IO
//   object_header_codec    the validity statement, one header per object
//   checkpoint_region      mirrored index snapshots
//   residency_bitmap       cross-rank residency, mmap, eventually consistent
//
// The core owns the key->slot index and is authoritative for it in steady
// state. Media is consulted only at open(): recovery reads the checkpoint and
// cross-checks every entry against its object header. Steady-state reads never
// touch metadata media, which is what keeps the hot path free of scans.
//
// TARGET OWNERSHIP. ObjectPlacement carries a borrowed ResolvedTarget pointer,
// so the store owns a per-slot cache of resolved targets and hands out pointers
// into it. This is safe across rehashing because std::unordered_map only
// invalidates references to erased elements.
//
// The cache never needs invalidating on reuse: slot paths are stable (named by
// slot number, not by key) and reclaim rewrites zeros in place without
// reallocating extents, so a slot's ResolvedTarget stays correct for the
// lifetime of the store. That is a direct payoff of the rename-free layout --
// the previous path-per-key design invalidated its target cache on every
// allocation.

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <tutti/spi/storage_object_store.h>
#include <tutti/spi/storage_target_resolver.h>

#include "csrc/storage_objects/checkpoint_region.h"
#include "csrc/storage_objects/object_header_codec.h"
#include "csrc/storage_objects/residency_bitmap.h"
#include "csrc/storage_objects/slot_placement_policy.h"
#include "csrc/storage_objects/space_allocator.h"

namespace tutti::storage_objects {

// Why a recovered object was dropped, for reporting. Recovery is expected to
// drop things after a crash, so these are counted rather than logged one by one.
struct RecoveryReport {
    std::uint64_t checkpoint_entries = 0;   // entries the checkpoint offered
    std::uint64_t accepted = 0;             // survived header cross-check
    std::uint64_t dropped_header = 0;       // header missing/torn/mismatched
    std::uint64_t dropped_geometry = 0;     // slot out of range for this config
    std::uint64_t dropped_duplicate = 0;    // same key twice, or slot reused
    bool checkpoint_available = false;      // false => cold or unreadable
};

class ObjectStoreCore final : public StorageObjectStore {
public:
    // Deferred form, used by create_storage_object_store(): placement and
    // resolver are built during open() from StoreConfig::devices, because until
    // then neither the mount paths nor the device identities are known.
    ObjectStoreCore();

    // Injected form: the core takes both dependencies directly. Keeps it
    // testable with a stub resolver and lets a deployment supply its own. The
    // resolver scheme comes from the placement policy, which is what knows the
    // URI format its resolver expects.
    ObjectStoreCore(std::unique_ptr<SlotPlacementPolicy> placement,
                    std::shared_ptr<StorageTargetResolver> resolver);
    ~ObjectStoreCore() override;

    // ---- StorageObjectStore ----

    Status open(const StoreConfig& config) override;
    Status close() override;

    bool contains(const ObjectKey& key) const override;
    std::uint64_t contains_prefix(const ObjectKey* keys,
                                  std::size_t count) const override;
    std::uint64_t contains_prefix_all_ranks(const ObjectKey* keys,
                                            std::size_t count) const override;
    Result<ObjectPlacement> lookup(const ObjectKey& key) const override;
    StoreUsage usage() const override;

    Result<ReserveOutcome> reserve(const ObjectKey* keys,
                                   std::size_t count) override;
    Status commit(const ObjectKey* keys, std::size_t count) override;
    Status abort(const ObjectKey* keys, std::size_t count) override;

    Result<std::uint64_t> release(const ObjectKey* keys,
                                 std::size_t count) override;
    Status pin(const ObjectKey* keys, std::size_t count) override;
    Status unpin(const ObjectKey* keys, std::size_t count) override;

    Result<std::vector<ObjectKey>> recover() override;
    Status checkpoint() override;

    // ---- test seams ----

    const RecoveryReport& recovery_report() const { return recovery_; }
    // Drain the reclaim queue: zero the slots and return them to free. Normally
    // driven by a background thread; exposed so a single-threaded test can step
    // it deterministically. Returns the number reclaimed.
    std::uint64_t drain_reclaim(std::uint64_t max);
    std::uint64_t materialised_slots() const;

private:
    struct Entry {
        std::uint64_t slot = 0;
        std::uint64_t generation = 0;
        std::uint64_t commit_seq = 0;
        bool committed = false;
    };

    // A stable string form of a key, for use as a map key. Hashing the bytes
    // directly would need a custom hasher on a vector; a byte-identical string
    // is simpler and avoids any chance of an identity-digest collision merging
    // two distinct keys in the index.
    static std::string index_key(const ObjectKey& key);

    Status ensure_layout_locked();
    // Build placement and resolver from config_.devices when they were not
    // injected. No-op for the injected form.
    Status build_backend_locked();
    Status materialise_through_locked(std::uint64_t slot_exclusive_end);
    // Const because the target map is a cache: resolving is logically a read.
    // Needed on the lookup path, which must work after a restart when recovery
    // has rebuilt the index but no targets are resolved yet.
    Result<const ResolvedTarget*> target_for_slot_locked(std::uint64_t slot) const;
    ObjectPlacement placement_locked(std::uint64_t slot, std::uint64_t generation,
                                     const ResolvedTarget* target) const;
    Status write_header_locked(std::uint64_t slot, const ObjectKey& key,
                               std::uint64_t generation, std::uint64_t commit_seq);
    Status load_checkpoint_locked();
    Status persist_checkpoint_locked();
    // Sequence number in each container, 0 meaning empty or unreadable.
    void read_container_sequences_locked(
        std::uint64_t* sequences,
        std::vector<CheckpointEntry>* bodies) const;
    std::string checkpoint_path_locked() const;
    std::string residency_path_locked(std::uint32_t rank) const;
    std::uint64_t fingerprint_digest_locked() const;

    mutable std::mutex mutex_;

    std::unique_ptr<SlotPlacementPolicy> placement_;
    std::shared_ptr<StorageTargetResolver> resolver_;

    StoreConfig config_;
    bool opened_ = false;

    std::uint64_t slot_bytes_ = 0;        // header + payload, per slot
    std::uint64_t shard_bytes_ = 0;       // per shard file
    std::uint64_t materialised_ = 0;      // slots [0, materialised_) are real
    std::uint64_t commit_seq_ = 0;        // monotonic, shared with checkpoints
    std::uint64_t checkpoint_seq_ = 0;
    std::uint64_t container_bytes_ = 0;
    bool checkpoint_dirty_ = false;

    SpaceAllocator allocator_;

    std::unordered_map<std::string, Entry> index_;
    std::unordered_map<std::uint64_t, std::string> slot_owner_;
    std::unordered_map<std::string, std::uint32_t> pins_;

    // Slot -> resolved target. Node-based, so handing out pointers into it is
    // safe across insertion. Mutable because it is a cache filled on demand,
    // including from the const lookup path.
    mutable std::unordered_map<std::uint64_t, ResolvedTarget> targets_;

    ResidencyBitmap own_residency_;
    std::vector<ResidencyBitmap> peer_residency_;

    RecoveryReport recovery_;
};

} // namespace tutti::storage_objects

#pragma once

// csrc/storage_objects/object_store_core.h -- StorageObjectStore implementation.
//
// IMPLEMENTATION DETAIL. Consumers reach this only through
// <tutti/spi/storage_object_store.h> and create_storage_object_store().
//
// Assembles the six units of this layer into one store:
//
//   space_allocator        slot numbers, states, generations
//   slot_placement_policy  slot -> segment file path/URI/offsets
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
// NO RESOLUTION HAPPENS HERE. StorageRuntime owns that: it takes a URI, selects
// a resolver by scheme, and mints the ticket the data path uses. This core hands
// out slot URIs and lets the runtime resolve them once. Resolving here as well
// would double the cost of a path that is open + fstat + fsync + FIEMAP plus a
// globally-serialised peer-memory DMA mapping -- historically the source of a
// multi-second stall on the first write.
//
// Everything this core still does is host-side metadata: object headers,
// checkpoint containers, the residency bitmap, and space accounting. None of it
// needs extents.

#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tutti/spi/storage_object_store.h>

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
    // The placement is built during open() from StoreConfig::devices, because
    // until then the mount paths are not known.
    ObjectStoreCore();
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

    std::uint64_t ready_slots() const override;
    std::string slot_uri(std::uint64_t slot) const override;
    std::uint64_t slot_generation(std::uint64_t slot) const override;

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

    // ---- growth: file-group precreate (see the SPI contract) ----

    Result<std::uint64_t> precreate_step(std::uint64_t headroom) override;
    std::uint64_t precreated_slots() const override;
    std::uint64_t precreate_target() const override;

    // ---- test seams ----

    const RecoveryReport& recovery_report() const { return recovery_; }
    // Drain the reclaim queue: zero the slots and return them to free. Normally
    // driven by a background thread; exposed so a single-threaded test can step
    // it deterministically. Returns the number reclaimed.
    std::uint64_t drain_reclaim(std::uint64_t max);

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
    // Build the placement from config_.devices and the segment geometry.
    Status build_backend_locked();
    // Adopt the leading file groups already on media (probe only, writes
    // nothing), stopping at the first group with a missing or unready file.
    Status adopt_existing_groups_locked();
    // Paths of the files forming the group that contains `slot`.
    Status group_paths_locked(std::uint64_t slot,
                              std::vector<std::string>* out) const;
    // Zero-fill, fsync and verify every file of `paths` (no lock held).
    Status materialise_group(const std::vector<std::string>& paths) const;
    ObjectPlacement placement_locked(std::uint64_t slot,
                                     std::uint64_t generation) const;
    Status write_header_locked(std::uint64_t slot, const ObjectKey& key,
                               std::uint64_t generation, std::uint64_t commit_seq);
    Status load_checkpoint_locked();
    Status persist_checkpoint_locked();
    void reclaim_worker_loop();
    // Sequence number in each container, 0 meaning empty or unreadable.
    void read_container_sequences_locked(
        std::uint64_t* sequences,
        std::vector<CheckpointEntry>* bodies) const;
    std::string checkpoint_path_locked() const;
    std::string residency_path_locked(std::uint32_t rank) const;
    std::uint64_t fingerprint_digest_locked() const;

    mutable std::mutex mutex_;

    std::unique_ptr<FixedSegmentFilePlacement> placement_;

    StoreConfig config_;
    bool opened_ = false;

    std::uint64_t slot_bytes_ = 0;        // header + payload, per slot
    std::uint64_t precreated_ = 0;        // slots [0, precreated_) are ready
    // Slots per file group (segment_file_slots x devices). A group is the
    // precreate unit and is published only when all its files are ready.
    std::uint64_t group_slots_ = 0;
    // True while a precreate_step() is materialising a group outside the lock.
    bool precreate_busy_ = false;
    std::uint64_t precreate_target_ = 0;
    std::uint64_t commit_seq_ = 0;        // monotonic, shared with checkpoints
    std::uint64_t checkpoint_seq_ = 0;
    std::uint64_t container_bytes_ = 0;
    bool checkpoint_dirty_ = false;

    std::thread reclaim_worker_;
    std::condition_variable reclaim_cv_;
    bool reclaim_stop_ = false;
    bool reclaim_pending_ = false;
    std::uint64_t reclaimed_total_ = 0;
    std::uint64_t reclaim_failures_ = 0;

    SpaceAllocator allocator_;

    std::unordered_map<std::string, Entry> index_;
    std::unordered_map<std::uint64_t, std::string> slot_owner_;
    std::unordered_map<std::string, std::uint32_t> pins_;

    ResidencyBitmap own_residency_;
    std::vector<ResidencyBitmap> peer_residency_;

    RecoveryReport recovery_;
};

} // namespace tutti::storage_objects

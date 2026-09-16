#pragma once

// csrc/storage_objects/space_allocator.h -- Slot-number space allocator.
//
// IMPLEMENTATION DETAIL. Reached only through the SPI.
//
// Allocates SLOT NUMBERS, never paths. This is the single decision that keeps
// placement pluggable: whether slot 7 becomes one file, four striped shards, or
// an offset into a raw device is decided elsewhere. Because the allocator never
// learns, everything below stays shared between the single-file and striped
// backends -- in the previous path-based design that same distinction had
// contaminated two dozen methods.
//
// Capacity is expressed in bytes at the SPI and converted to a slot count here,
// so callers size a cache the way they think about it while the allocator works
// in the unit it can actually manage.
//
// A full cache is a steady state, not a fault: a cache that never reaches its
// ceiling is over-provisioned. Allocation therefore reports "no space" as an
// ordinary result and never blocks or throws.
//
// Reclamation is asynchronous because a slot must be physically re-zeroed
// before it can be reused (FIEMAP-backed DMA cannot use sparse extents). The
// allocator models this with a distinct reclaiming state rather than pretending
// a released slot is instantly available, which is what makes the "eviction
// outruns scrubbing" condition observable instead of appearing as a stall.

#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace tutti::storage_objects {

// A slot's lifecycle. Reserved and committed are distinguished because space
// occupancy and data validity are separate events with an IO window between
// them; reclaiming is distinguished because re-zeroing takes real time.
enum class SlotState {
    kFree = 0,
    kReserved,
    kCommitted,
    kReclaiming,
};

const char* to_string(SlotState state) noexcept;

struct SpaceAllocatorConfig {
    // Ceiling in bytes; the slot count is derived from it.
    std::uint64_t capacity_bytes = 0;
    // On-media bytes per slot, header included.
    std::uint64_t slot_bytes = 0;
    // Slots to materialise up front. Clamped to the derived slot count.
    std::uint64_t prewarm_slots = 0;
};

struct SpaceAllocatorStats {
    std::uint64_t total_slots = 0;
    std::uint64_t free_slots = 0;
    std::uint64_t reserved_slots = 0;
    std::uint64_t committed_slots = 0;
    std::uint64_t reclaiming_slots = 0;
    // Slots never handed out yet: materialisation is lazy beyond prewarm.
    std::uint64_t unmaterialised_slots = 0;
};

// Result of a reservation attempt. `slots` may be shorter than requested --
// partial acceptance is normal, not an error.
struct SlotReservation {
    std::vector<std::uint64_t> slots;
    std::vector<std::uint64_t> generations;  // parallel to `slots`
    std::uint64_t rejected = 0;
};

// -------------------------------------------------------------------------
// SpaceAllocator
//
// Thread-safe. Every method takes an internal lock; none performs IO, so no
// call blocks on the device. Physical zeroing is the caller's job, driven by
// take_reclaimable() and finish_reclaim().
// -------------------------------------------------------------------------
class SpaceAllocator {
public:
    SpaceAllocator() = default;

    // Derives the slot count as capacity_bytes / slot_bytes. Returns false for
    // a zero slot_bytes or a capacity too small for even one slot -- a store
    // that cannot hold anything is a configuration error, not an empty cache.
    bool configure(const SpaceAllocatorConfig& config);

    std::uint64_t total_slots() const;
    std::uint64_t slot_bytes() const;

    // Slots that should be materialised before serving traffic.
    std::uint64_t prewarm_slots() const;

    // Reserve up to `count` slots. Takes what is available and reports the
    // remainder in `rejected`. Never blocks, never fails as a whole.
    SlotReservation reserve(std::uint64_t count);

    // Reserved -> Committed. Ignores slots not currently reserved, so a
    // duplicate commit is harmless.
    void commit(const std::uint64_t* slots, std::size_t count);

    // Reserved -> Reclaiming. Committed slots are ignored: abort must never be
    // able to drop live data.
    void abort(const std::uint64_t* slots, std::size_t count);

    // Committed -> Reclaiming. Returns how many actually moved.
    std::uint64_t release(const std::uint64_t* slots, std::size_t count);

    // Hand out up to `max` slots needing physical zeroing. The returned slots
    // stay in kReclaiming until finish_reclaim(), so two callers cannot scrub
    // the same slot.
    std::vector<std::uint64_t> take_reclaimable(std::uint64_t max);

    // Reclaiming -> Free, bumping the generation so any stale placement
    // referring to this space is detectably obsolete (ABA).
    void finish_reclaim(const std::uint64_t* slots, std::size_t count);

    // Return slots to the reclaiming queue when zeroing failed, so a transient
    // IO error does not permanently leak capacity.
    void requeue_reclaim(const std::uint64_t* slots, std::size_t count);

    SlotState state_of(std::uint64_t slot) const;
    std::uint64_t generation_of(std::uint64_t slot) const;
    SpaceAllocatorStats stats() const;

private:
    std::uint64_t next_free_locked();

    mutable std::mutex mutex_;
    std::uint64_t total_slots_ = 0;
    std::uint64_t slot_bytes_ = 0;
    std::uint64_t prewarm_slots_ = 0;

    // Slots beyond this have never been handed out. Materialisation is lazy so
    // a large ceiling does not imply a long open().
    std::uint64_t high_water_ = 0;

    std::vector<SlotState> states_;
    std::vector<std::uint64_t> generations_;
    std::deque<std::uint64_t> free_list_;
    std::deque<std::uint64_t> reclaim_queue_;
    std::unordered_set<std::uint64_t> reclaim_in_flight_;
};

} // namespace tutti::storage_objects

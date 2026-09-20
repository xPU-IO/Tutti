// csrc/storage_objects/space_allocator.cpp

#include "csrc/storage_objects/space_allocator.h"

#include <algorithm>

namespace tutti::storage_objects {

const char* to_string(SlotState state) noexcept {
    switch (state) {
        case SlotState::kFree: return "free";
        case SlotState::kReserved: return "reserved";
        case SlotState::kCommitted: return "committed";
        case SlotState::kReclaiming: return "reclaiming";
    }
    return "unknown";
}

bool SpaceAllocator::configure(const SpaceAllocatorConfig& config) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (config.slot_bytes == 0) return false;

    const std::uint64_t derived = config.capacity_bytes / config.slot_bytes;
    // A store that cannot hold a single object is a misconfiguration, not an
    // empty cache: failing here surfaces the mistake instead of silently
    // rejecting every future write.
    if (derived == 0) return false;

    total_slots_ = derived;
    slot_bytes_ = config.slot_bytes;
    prewarm_slots_ = std::min(config.prewarm_slots, derived);
    high_water_ = 0;

    states_.assign(static_cast<std::size_t>(derived), SlotState::kFree);
    generations_.assign(static_cast<std::size_t>(derived), 0);
    free_list_.clear();
    reclaim_queue_.clear();
    reclaim_in_flight_.clear();
    return true;
}

std::uint64_t SpaceAllocator::total_slots() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return total_slots_;
}

std::uint64_t SpaceAllocator::slot_bytes() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return slot_bytes_;
}

std::uint64_t SpaceAllocator::prewarm_slots() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return prewarm_slots_;
}

// Prefers recycled slots over never-used ones so the working set stays compact:
// that keeps the materialised prefix small, which matters because
// materialisation is the expensive part.
std::uint64_t SpaceAllocator::next_free_locked() {
    if (!free_list_.empty()) {
        const std::uint64_t slot = free_list_.front();
        free_list_.pop_front();
        return slot;
    }
    if (high_water_ < total_slots_) return high_water_++;
    return total_slots_;  // sentinel: exhausted
}

SlotReservation SpaceAllocator::reserve(std::uint64_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    SlotReservation out;
    out.slots.reserve(static_cast<std::size_t>(count));
    out.generations.reserve(static_cast<std::size_t>(count));

    for (std::uint64_t i = 0; i < count; ++i) {
        const std::uint64_t slot = next_free_locked();
        if (slot >= total_slots_) {
            // Out of space. Report the remainder and stop: exhaustion is an
            // ordinary outcome, so there is nothing to wait for or raise.
            out.rejected = count - i;
            break;
        }
        states_[static_cast<std::size_t>(slot)] = SlotState::kReserved;
        out.slots.push_back(slot);
        out.generations.push_back(generations_[static_cast<std::size_t>(slot)]);
    }
    return out;
}

void SpaceAllocator::commit(const std::uint64_t* slots, std::size_t count) {
    if (slots == nullptr) return;
    std::lock_guard<std::mutex> guard(mutex_);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint64_t slot = slots[i];
        if (slot >= total_slots_) continue;
        auto& state = states_[static_cast<std::size_t>(slot)];
        if (state == SlotState::kReserved) state = SlotState::kCommitted;
    }
}

void SpaceAllocator::abort(const std::uint64_t* slots, std::size_t count) {
    if (slots == nullptr) return;
    std::lock_guard<std::mutex> guard(mutex_);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint64_t slot = slots[i];
        if (slot >= total_slots_) continue;
        auto& state = states_[static_cast<std::size_t>(slot)];
        // Only reserved slots are abortable. Committed data is untouchable here
        // so a stale abort cannot destroy a live object.
        if (state != SlotState::kReserved) continue;
        state = SlotState::kReclaiming;
        reclaim_queue_.push_back(slot);
    }
}

std::uint64_t SpaceAllocator::release(const std::uint64_t* slots,
                                      std::size_t count) {
    if (slots == nullptr) return 0;
    std::lock_guard<std::mutex> guard(mutex_);
    std::uint64_t released = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint64_t slot = slots[i];
        if (slot >= total_slots_) continue;
        auto& state = states_[static_cast<std::size_t>(slot)];
        if (state != SlotState::kCommitted) continue;
        state = SlotState::kReclaiming;
        reclaim_queue_.push_back(slot);
        ++released;
    }
    return released;
}

std::vector<std::uint64_t> SpaceAllocator::take_reclaimable(std::uint64_t max) {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<std::uint64_t> out;
    while (out.size() < max && !reclaim_queue_.empty()) {
        const std::uint64_t slot = reclaim_queue_.front();
        reclaim_queue_.pop_front();
        // Track in-flight scrubs so a second caller cannot pick up the same
        // slot and race on zeroing it.
        reclaim_in_flight_.insert(slot);
        out.push_back(slot);
    }
    return out;
}

void SpaceAllocator::finish_reclaim(const std::uint64_t* slots,
                                    std::size_t count) {
    if (slots == nullptr) return;
    std::lock_guard<std::mutex> guard(mutex_);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint64_t slot = slots[i];
        if (slot >= total_slots_) continue;
        if (reclaim_in_flight_.erase(slot) == 0) continue;
        // Bump the generation so any placement still referring to this space is
        // detectably stale (ABA detection at recovery and lookup time).
        ++generations_[static_cast<std::size_t>(slot)];
        states_[static_cast<std::size_t>(slot)] = SlotState::kFree;
        free_list_.push_back(slot);
    }
}

void SpaceAllocator::requeue_reclaim(const std::uint64_t* slots,
                                     std::size_t count) {
    if (slots == nullptr) return;
    std::lock_guard<std::mutex> guard(mutex_);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint64_t slot = slots[i];
        if (slot >= total_slots_) continue;
        if (reclaim_in_flight_.erase(slot) == 0) continue;
        // Stays kReclaiming and goes back on the queue: a transient zeroing
        // failure must not leak the slot permanently.
        reclaim_queue_.push_back(slot);
    }
}

SlotState SpaceAllocator::state_of(std::uint64_t slot) const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (slot >= total_slots_) return SlotState::kFree;
    return states_[static_cast<std::size_t>(slot)];
}

std::uint64_t SpaceAllocator::generation_of(std::uint64_t slot) const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (slot >= total_slots_) return 0;
    return generations_[static_cast<std::size_t>(slot)];
}

SpaceAllocatorStats SpaceAllocator::stats() const {
    std::lock_guard<std::mutex> guard(mutex_);
    SpaceAllocatorStats s;
    s.total_slots = total_slots_;
    for (std::uint64_t slot = 0; slot < high_water_; ++slot) {
        switch (states_[static_cast<std::size_t>(slot)]) {
            case SlotState::kFree: ++s.free_slots; break;
            case SlotState::kReserved: ++s.reserved_slots; break;
            case SlotState::kCommitted: ++s.committed_slots; break;
            case SlotState::kReclaiming: ++s.reclaiming_slots; break;
        }
    }
    s.unmaterialised_slots = total_slots_ - high_water_;
    return s;
}

} // namespace tutti::storage_objects

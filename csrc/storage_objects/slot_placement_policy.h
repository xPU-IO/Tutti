#pragma once

// csrc/storage_objects/slot_placement_policy.h -- slot number -> media location.
//
// IMPLEMENTATION DETAIL. Reached only through the SPI.
//
// This is the ONLY seam between the shared object-store core and the question
// of how a slot maps onto media. The core allocates slot numbers, encodes
// headers, maintains checkpoints and bitmaps, and recovers -- all without
// knowing whether a slot is one file or a striped set.
//
// Why the interface is this small: striping is already handled a layer below.
// The striped resolver bundles N shard ResolvedTargets into ONE ResolvedTarget
// whose payload carries the stripe geometry, and the DataPath fans out using
//     shard     = (off / unit) % N
//     shard_off = (off / (unit * N)) * unit + (off % unit)
// So a striped object still has a single ResolvedTarget and a single logical
// offset space. Placement therefore does not need to enumerate IO pieces, and
// the previous path-based design's two dozen striped/non-striped differences
// collapse to the four methods below.
//
// Header placement: the object header lives at offset 0 of shard 0 only, and
// every shard reserves the same header-sized prefix so all shards stay the same
// size and the payload offset is uniform. Replicating the header across shards
// was rejected: it would turn one atomic commit into N fsyncs, where a crash
// could leave shard 0 committed and shard 1 not. Single-commit atomicity is
// worth more than header redundancy.

#include <cstdint>
#include <string>
#include <vector>

#include <tutti/status.h>
#include <tutti/spi/storage_object_store.h>

namespace tutti::storage_objects {

class SlotPlacementPolicy {
public:
    virtual ~SlotPlacementPolicy() = default;

    // Physical files backing one slot. 1 for a single-file layout, N for a
    // striped one.
    virtual std::uint32_t shard_count() const = 0;

    // Filesystem paths of the files backing `slot`, in stripe order. Used for
    // materialisation (writing real zeros) and for header IO -- never handed to
    // the caller, who only ever sees an ObjectPlacement.
    virtual Status paths_for_slot(std::uint64_t slot,
                                  std::vector<std::string>* out) const = 0;

    // Bytes each individual shard file must be for a slot of `slot_bytes`
    // total. For a single file this is slot_bytes; for N shards it is the
    // per-shard share plus the reserved header prefix.
    virtual std::uint64_t shard_file_bytes(std::uint64_t slot_bytes) const = 0;

    // URI to hand the resolver to obtain this slot's ResolvedTarget. The scheme
    // selects the resolver, so this string is what ties a slot to its backend.
    virtual std::string uri_for_slot(std::uint64_t slot) const = 0;

    // Where the object header lives: which shard, and at what offset in that
    // shard's file. Always shard 0 at offset 0 in the current layouts, but
    // expressed explicitly so a backend with a different metadata region does
    // not have to fight the core.
    virtual std::uint32_t header_shard() const { return 0; }
    virtual std::uint64_t header_offset_in_shard() const { return 0; }

    // Offset of the payload within the object's logical address space. Equals
    // the reserved header prefix, so segment 0 starts here.
    virtual std::uint64_t payload_offset() const {
        return ObjectHeaderLayout::kHeaderBytes;
    }
};

// -------------------------------------------------------------------------
// SingleFilePlacement -- one file per slot.
//
// Layout of "<root>/slots/<slot>.obj":
//   [0, 4096)                 object header
//   [4096, 4096 + payload)    payload, segments back to back
// -------------------------------------------------------------------------
class SingleFilePlacement final : public SlotPlacementPolicy {
public:
    // `uri_scheme` selects the resolver (e.g. "local_nvme_file"). `root` is the
    // namespace directory on the mounted filesystem.
    SingleFilePlacement(std::string root, std::string uri_scheme);

    std::uint32_t shard_count() const override { return 1; }
    Status paths_for_slot(std::uint64_t slot,
                          std::vector<std::string>* out) const override;
    std::uint64_t shard_file_bytes(std::uint64_t slot_bytes) const override;
    std::string uri_for_slot(std::uint64_t slot) const override;

    const std::string& root() const { return root_; }

private:
    std::string slot_path(std::uint64_t slot) const;

    std::string root_;
    std::string uri_scheme_;
};

// -------------------------------------------------------------------------
// StripedPlacement -- N shard files per slot, one per device.
//
// Layout of "<mount_i>/slots/<slot>.s<i>":
//   [0, 4096)                 header region; only shard 0's is written, the
//                             rest is reserved so all shards are equal size
//   [4096, 4096 + share)      this shard's slice of the payload
//
// The payload's logical space is mapped onto the shards by the resolver using
// the stripe formula; this policy only has to produce the right paths and
// sizes.
// -------------------------------------------------------------------------
class StripedPlacement final : public SlotPlacementPolicy {
public:
    // `mounts` is one directory per device, in stripe order. `stripe_unit` is
    // the round-robin granularity and must divide the per-shard payload share
    // so no segment straddles a shard boundary unevenly.
    StripedPlacement(std::vector<std::string> mounts, std::uint64_t stripe_unit,
                     std::string uri_scheme);

    std::uint32_t shard_count() const override {
        return static_cast<std::uint32_t>(mounts_.size());
    }
    Status paths_for_slot(std::uint64_t slot,
                          std::vector<std::string>* out) const override;
    std::uint64_t shard_file_bytes(std::uint64_t slot_bytes) const override;
    std::string uri_for_slot(std::uint64_t slot) const override;

    std::uint64_t stripe_unit() const { return stripe_unit_; }
    const std::vector<std::string>& mounts() const { return mounts_; }

    // True when the geometry is self-consistent: at least one mount, a nonzero
    // stripe unit, and a payload that divides evenly across shards in whole
    // stripe units. Checked by the store at open() so a bad configuration fails
    // loudly instead of producing objects whose segments straddle shards.
    bool geometry_valid(std::uint64_t slot_bytes) const;

private:
    std::string shard_path(std::uint64_t slot, std::uint32_t shard) const;

    std::vector<std::string> mounts_;
    std::uint64_t stripe_unit_;
    std::string uri_scheme_;
};

} // namespace tutti::storage_objects

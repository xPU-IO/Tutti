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

    // Scheme to put in ResolveOptions. Distinct from the URI prefix only in
    // that it carries no "://" -- the resolvers check this field before parsing.
    virtual std::string resolver_scheme() const = 0;

    // Filesystem paths of the files backing `slot`, in stripe order. Used for
    // materialisation (writing real zeros) and for header IO -- never handed to
    // the caller, who only ever sees an ObjectPlacement.
    //
    // MUST agree byte for byte with the paths the resolver derives from
    // uri_for_slot(). The resolvers build their own backing paths from the URI,
    // so a mismatch would have materialisation write one set of files while
    // resolution reads another -- and DMA would target unallocated extents.
    virtual Status paths_for_slot(std::uint64_t slot,
                                  std::vector<std::string>* out) const = 0;

    // Bytes each individual shard file must be for a slot of `slot_bytes`
    // total. For a single file this is slot_bytes; for N shards it is the
    // per-shard share plus the reserved header prefix.
    virtual std::uint64_t shard_file_bytes(std::uint64_t slot_bytes) const = 0;

    // URI identifying this slot, for the runtime to open. Its format is fixed by
    // the resolver that will parse it, not chosen here.
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
//
// The local-file resolver takes the path verbatim from a "file://<abs path>"
// URI, so the path shape is this policy's choice.
// -------------------------------------------------------------------------
class SingleFilePlacement final : public SlotPlacementPolicy {
public:
    explicit SingleFilePlacement(std::string root);

    std::uint32_t shard_count() const override { return 1; }
    std::string resolver_scheme() const override { return "file"; }
    Status paths_for_slot(std::uint64_t slot,
                          std::vector<std::string>* out) const override;
    std::uint64_t shard_file_bytes(std::uint64_t slot_bytes) const override;
    std::string uri_for_slot(std::uint64_t slot) const override;

    const std::string& root() const { return root_; }

private:
    std::string slot_path(std::uint64_t slot) const;

    std::string root_;
};

// -------------------------------------------------------------------------
// StripedPlacement -- N shard files per slot, one per device.
//
// The striped resolver parses
//     striped://<name>?devs=<m1,m2,...>&unit=<bytes>
// and derives each shard path itself as
//     <mount_i>/striped/<name>.shard<i>
// so BOTH the URI format and the resulting paths are dictated by the resolver.
// This policy uses the slot number as <name> and must reproduce those paths
// exactly, since it is what materialises the files the resolver will map.
//
// Layout of each shard:
//   [0, 4096)                 header region; only shard 0's is written, the
//                             rest is reserved so all shards are equal size
//   [4096, 4096 + share)      this shard's slice of the payload
// -------------------------------------------------------------------------
class StripedPlacement final : public SlotPlacementPolicy {
public:
    // `mounts` is one directory per device, in stripe order. `stripe_unit` is
    // the round-robin granularity and must divide the per-shard payload share
    // so no segment straddles a shard boundary unevenly.
    StripedPlacement(std::vector<std::string> mounts, std::uint64_t stripe_unit);

    std::uint32_t shard_count() const override {
        return static_cast<std::uint32_t>(mounts_.size());
    }
    std::string resolver_scheme() const override { return "striped"; }

    // Payload start in the object's logical address space, rounded up to a whole
    // stripe round.
    //
    // Reserving only the header (4096B) leaves every segment -- and therefore
    // every IO request -- misaligned to the stripe unit: a 128 KiB request then
    // touches three shards instead of two, and the extra command per request
    // showed up as +64% write IO kernel time on real hardware (2.64ms -> 4.34ms
    // per submit, 4.6s -> 7.4s of device time for a 10k-token cold request).
    // Aligning to a whole round costs at most one round of space per slot and
    // keeps each request inside the shards it was sized for.
    std::uint64_t payload_offset() const override;
    Status paths_for_slot(std::uint64_t slot,
                          std::vector<std::string>* out) const override;
    std::uint64_t shard_file_bytes(std::uint64_t slot_bytes) const override;
    std::string uri_for_slot(std::uint64_t slot) const override;

    std::uint64_t stripe_unit() const { return stripe_unit_; }
    const std::vector<std::string>& mounts() const { return mounts_; }

    // True when the geometry is self-consistent: at least one mount, a stripe
    // unit that is nonzero and 4096-aligned (the resolver's own requirement),
    // and a payload that divides evenly across shards in whole stripe units.
    // Checked by the store at open() so a bad configuration fails loudly instead
    // of producing objects whose segments straddle shards.
    bool geometry_valid(std::uint64_t slot_bytes) const;

private:
    std::string shard_path(std::uint64_t slot, std::uint32_t shard) const;

    std::vector<std::string> mounts_;
    std::uint64_t stripe_unit_;
};

} // namespace tutti::storage_objects

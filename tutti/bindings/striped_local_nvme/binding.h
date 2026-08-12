#pragma once

// tutti/bindings/striped_local_nvme/binding.h
//
// Striped binding: pair-private payload contract for striped://
// targets backed by N local NVMe devices.
//
// This header is the SINGLE declaration point for:
//   - the payload type identity (string ID + API version)
//   - the C++ payload type (StripedLocalNvmePayload)
//   - the stripe offset mapping formula
//   - the pairing helpers (make_resolved_target / view_payload)
//
// Design (per abstract resolution):
//   - Zero new public nouns: striped files open via rt.open(uri) and
//     return a plain ResolvedTarget.  Striping metadata is entirely
//     inside the pair-private payload.
//   - The payload holds N sub-ResolvedTargets (one per shard), each
//     carrying its own Ext4LocalNvmePayload + FileDescriptorLease.
//   - The outer lease (StripeBundleLease) is a marker; the real fd
//     cleanup happens when the payload's shard vector is destroyed
//     (each shard's shared_ptr<void> lease refcount → 0 → fd close).
//
// Stripe semantics:
//   unit-granularity round-robin across N shards.
//
//   For a logical byte offset `off`:
//     shard      = (off / unit) % N
//     shard_off  = (off / (unit * N)) * unit + (off % unit)
//
//   logical_size = N * min_shard_size  (aligned to stripe_unit boundary;
//   the last partial stripe round is truncated to the smallest shard).
//
// Header-only C++17.  Depends on <tutti/status.h>,
// <tutti/spi/storage_target_resolver.h>, and the ext4_local_nvme binding.

#include <tutti/status.h>
#include <tutti/spi/storage_target_resolver.h>
#include <tutti/bindings/ext4_local_nvme/binding.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tutti::binding::striped_local_nvme {

// -------------------------------------------------------------------------
// Identity constants
// -------------------------------------------------------------------------
inline constexpr std::string_view kPayloadTypeId =
    "striped-local-nvme-payload-v1";

inline constexpr std::uint32_t kPayloadApiVersion = 1;

inline constexpr std::string_view kRecommendedDataPathKey =
    "striped-local-nvme";

inline constexpr std::string_view kResolverTypeId =
    "striped-resolver-v1";

// -------------------------------------------------------------------------
// StripeBundleLease
//
// Owner lease for the striped bundle.  Holds the N sub-leases extracted
// from the per-shard ResolvedTargets.  When destroyed, all N sub-leases
// are released (closing N fds).
//
// Note: the sub-leases are ALSO held inside the StripedLocalNvmePayload's
// shard ResolvedTargets (via shared_ptr<void> lease).  The StripeBundleLease
// provides an explicit, independent lease reference so that the outer
// ResolvedTarget's lease is non-null (required by ResolvedTarget::make).
// Both references must be released before the fds close.
// -------------------------------------------------------------------------
struct StripeBundleLease {
    std::vector<std::shared_ptr<void>> sub_leases;

    explicit StripeBundleLease(std::vector<std::shared_ptr<void>> leases)
        : sub_leases(std::move(leases)) {}
};

// -------------------------------------------------------------------------
// StripedLocalNvmePayload
//
// Immutable payload produced by StripedResolver and consumed by the
// future StripedDataPath (Session 3).  Contains:
//   - num_shards:   N (number of devices)
//   - stripe_unit:  round-robin granularity in bytes
//   - shards:       N ResolvedTargets, each carrying an Ext4LocalNvmePayload
//   - logical_size: N * min_shard_size (truncated to stripe boundary)
// -------------------------------------------------------------------------
class StripedLocalNvmePayload {
public:
    // -------------------------------------------------------------------
    // Factory: validates the shard set before construction.
    // -------------------------------------------------------------------
    static Result<std::shared_ptr<StripedLocalNvmePayload>>
    create(std::uint32_t num_shards,
           std::uint64_t stripe_unit,
           std::vector<ResolvedTarget> shards,
           std::uint32_t shard_rotation = 0) {

        if (num_shards == 0) {
            return Result<std::shared_ptr<StripedLocalNvmePayload>>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "num_shards == 0"));
        }
        if (stripe_unit == 0) {
            return Result<std::shared_ptr<StripedLocalNvmePayload>>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "stripe_unit == 0"));
        }
        if (shards.size() != num_shards) {
            return Result<std::shared_ptr<StripedLocalNvmePayload>>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "shards.size()=" + std::to_string(shards.size()) +
                       " != num_shards=" + std::to_string(num_shards)));
        }
        // All shards must be valid.
        for (std::size_t i = 0; i < shards.size(); ++i) {
            if (!shards[i].valid()) {
                return Result<std::shared_ptr<StripedLocalNvmePayload>>::
                    Failure(Status(StatusCode::INVALID_ARGUMENT,
                                   "shard " + std::to_string(i) +
                                   " is not valid"));
            }
        }

        // logical_size = N * min_shard_logical_size, truncated to
        // the largest complete stripe round.
        std::uint64_t min_shard = UINT64_MAX;
        for (const auto& s : shards) {
            min_shard = std::min(min_shard, s.logical_size());
        }
        // Truncate to the last complete stripe round: each shard
        // contributes floor(min_shard / unit) full units.  The logical
        // size is N * (floor(min_shard / unit) * unit).
        std::uint64_t full_units_per_shard = min_shard / stripe_unit;
        std::uint64_t logical = full_units_per_shard * stripe_unit * num_shards;

        auto ptr = std::shared_ptr<StripedLocalNvmePayload>(
            new StripedLocalNvmePayload(
                num_shards, stripe_unit, std::move(shards), logical,
                shard_rotation % num_shards));
        return Result<std::shared_ptr<StripedLocalNvmePayload>>::Success(
            std::move(ptr));
    }

    // ---- Read-only accessors ----
    std::uint32_t num_shards() const noexcept { return num_shards_; }
    std::uint64_t stripe_unit() const noexcept { return stripe_unit_; }
    std::uint64_t logical_size() const noexcept { return logical_size_; }
    // Round 16 S7: per-target shard rotation (legacy shard_placement
    // equivalent, kv_cache_layerwise_overlap.cu:293).  0 = no rotation
    // (default; identical to pre-S7 behavior).
    std::uint32_t shard_rotation() const noexcept { return shard_rotation_; }
    const std::vector<ResolvedTarget>& shards() const noexcept {
        return shards_;
    }

    // -------------------------------------------------------------------
    // map_to_shard — stripe offset mapping.
    //
    // Given a logical byte offset, returns {shard_index, shard_offset}.
    // Returns OUT_OF_RANGE if offset >= logical_size.
    //
    // Formula:
    //   shard      = ((offset / unit) + rot) % N(rot = shard_rotation)
    //   shard_off= (offset / (unit * N)) * unit + (offset % unit)
    //
    // The rotation is injective per target (it only permutes which shard
    // holds a given logical unit), so read/write of the same offset always
    // resolve to the same (shard, shard_offset) pair.  rot == 0 reproduces
    // the pre-S7 formula exactly.
    // -------------------------------------------------------------------
    struct ShardLocation {
        std::uint32_t shard_index;
        std::uint64_t shard_offset;
    };

    Result<ShardLocation> map_to_shard(std::uint64_t offset) const {
        if (offset >= logical_size_) {
            return Result<ShardLocation>::Failure(
                Status(StatusCode::OUT_OF_RANGE,
                       "offset " + std::to_string(offset) +
                       " >= logical_size " + std::to_string(logical_size_)));
        }
        ShardLocation loc;
        loc.shard_index = static_cast<std::uint32_t>(
            ((offset / stripe_unit_) + shard_rotation_) % num_shards_);
        loc.shard_offset =
            (offset / (stripe_unit_ * num_shards_)) * stripe_unit_ +
            (offset % stripe_unit_);
        return loc;
    }

private:
    StripedLocalNvmePayload(std::uint32_t num_shards,
                            std::uint64_t stripe_unit,
                            std::vector<ResolvedTarget> shards,
                            std::uint64_t logical_size,
                            std::uint32_t shard_rotation)
        : num_shards_(num_shards),
          stripe_unit_(stripe_unit),
          shards_(std::move(shards)),
          logical_size_(logical_size),
          shard_rotation_(shard_rotation) {}

    std::uint32_t num_shards_ = 0;
    std::uint64_t stripe_unit_ = 0;
    std::vector<ResolvedTarget> shards_;
    std::uint64_t logical_size_ = 0;
    std::uint32_t shard_rotation_ = 0;
};

// -------------------------------------------------------------------------
// Pairing helpers
// -------------------------------------------------------------------------
template <typename OwnerLease>
inline Result<ResolvedTarget> make_resolved_target(
    std::shared_ptr<StripedLocalNvmePayload> payload,
    std::shared_ptr<OwnerLease> owner_lease,
    std::string recommended_data_path_key =
        std::string(kRecommendedDataPathKey)) {

    return ResolvedTarget::make<StripedLocalNvmePayload, OwnerLease>(
        std::string(kResolverTypeId),
        std::string(kPayloadTypeId),
        kPayloadApiVersion,
        payload->logical_size(),
        std::move(recommended_data_path_key),
        std::move(payload),
        std::move(owner_lease));
}

inline Result<const StripedLocalNvmePayload*>
view_payload(const ResolvedTarget& target) {
    return target.view<StripedLocalNvmePayload>(
        kPayloadTypeId, kPayloadApiVersion);
}

} // namespace tutti::binding::striped_local_nvme

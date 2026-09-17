#pragma once

// tutti/bindings/ext4_local_nvme/binding.h
//
// First binding: the private payload contract shared between the
// ext4 resolver and the local NVMe DataPath.
//
// This header is the SINGLE declaration point for:
//   - the payload type identity (string ID + API version)
//   - the C++ payload type (Ext4LocalNvmePayload)
//   - the pairing helpers (make_resolved_target / view_payload)
//
// Neither resolver nor DataPath implementation should write the payload
// type id string or the API version as bare literals. Both call the
// helpers in this file, so the pairing physically cannot diverge.
//
// Header-only C++17. Depends only on <tutti/status.h> and
// <tutti/spi/storage_target_resolver.h> plus the standard library.

#include <tutti/status.h>
#include <tutti/spi/storage_target_resolver.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tutti::binding::ext4_local_nvme {

// -------------------------------------------------------------------------
// Identity constants
//
// These are the ONLY place where the payload type id string and API
// version appear.  Both make_resolved_target() and view_payload() use
// them, so resolver and DataPath always agree.
// -------------------------------------------------------------------------

inline constexpr std::string_view kPayloadTypeId =
    "ext4-local-nvme-payload-v1";

inline constexpr std::uint32_t kPayloadApiVersion = 1;

inline constexpr std::string_view kRecommendedDataPathKey =
    "local-nvme-ext4";

inline constexpr std::string_view kResolverTypeId =
    "ext4-extent-resolver-v1";

// -------------------------------------------------------------------------
// Extent
//
// A single contiguous physical mapping: a byte range in the file's
// logical address space maps to a byte range on the NVMe device.
//
//   logical_offset  : starting byte offset within the file
//   device_offset   : starting byte offset on the NVMe namespace
//                     (includes the configured namespace_base_bytes;
//                      see LocalFileResolver::BackingDeviceConfig)
//   length          : number of bytes in this extent
//
// device_offset is the NAMESPACE byte offset, NOT the raw FIEMAP
// fe_physical.  The resolver adds namespace_base_bytes to fe_physical
// to produce device_offset.  For a whole-disk ext4 filesystem on
// /dev/snvme0n1 with no partition, namespace_base_bytes == 0 and
// device_offset == fe_physical.
//
// All offsets and lengths are in bytes.  LBA granularity is a
// resolver/DataPath implementation detail; this contract operates in
// raw byte offsets.
// -------------------------------------------------------------------------
struct Extent {
    std::uint64_t logical_offset = 0;
    std::uint64_t device_offset  = 0;
    std::uint64_t length         = 0;
};

// -------------------------------------------------------------------------
// NamespaceIdentity
//
// Identifies the NVMe namespace that backs this target.  Carried in
// the payload so the DataPath knows which controller/namespace to
// submit IO against, but does NOT contain any transport-private
// descriptor (no transport queue, no completion register).
// -------------------------------------------------------------------------
struct NamespaceIdentity {
    std::string controller_pci_addr;   // e.g. "0000:08:00.0"
    std::uint32_t namespace_id = 1;    // NVMe namespace id
    std::uint32_t block_size    = 0;   // logical block size in bytes
};

// -------------------------------------------------------------------------
// Ext4LocalNvmePayload
//
// Immutable payload produced by the resolver and consumed by the
// DataPath.  Contains:
//   - the namespace identity
//   - an ordered, validated set of extents covering [0, file_size)
//   - the logical file size
//
// The constructor is private; use the factory which runs validate().
// All accessors return const references / values only.
// -------------------------------------------------------------------------
class Ext4LocalNvmePayload {
public:
    // Factory: validates the extent set before construction.
    // Returns DATA_LOSS if extents have holes, overlaps, are out of
    // order, or do not fully cover [0, file_size).
    static Result<std::shared_ptr<const Ext4LocalNvmePayload>>
    create(NamespaceIdentity ns,
           std::vector<Extent> extents,
           std::uint64_t file_size) {

        // Build a temporary to validate.
        auto tmp = std::unique_ptr<Ext4LocalNvmePayload>(
            new Ext4LocalNvmePayload(
                std::move(ns), std::move(extents), file_size));

        Status vs = tmp->validate();
        if (!vs.ok()) {
            return Result<std::shared_ptr<const Ext4LocalNvmePayload>>::
                Failure(std::move(vs));
        }
        // Wrap in shared_ptr<const ...>.
        return std::shared_ptr<const Ext4LocalNvmePayload>(tmp.release());
    }

    // ---- Read-only accessors ----

    const NamespaceIdentity& namespace_identity() const noexcept {
        return ns_;
    }

    const std::vector<Extent>& extents() const noexcept {
        return extents_;
    }

    std::uint64_t file_size() const noexcept {
        return file_size_;
    }

    // Map a logical byte offset within the file to a device byte offset.
    // Returns OUT_OF_RANGE if the offset falls in a hole or is >= file_size.
    Result<std::uint64_t> map_to_device_offset(
        std::uint64_t logical_offset) const {

        if (logical_offset >= file_size_) {
            return Result<std::uint64_t>::Failure(
                Status(StatusCode::OUT_OF_RANGE,
                       "logical offset >= file size"));
        }
        // Linear scan (extents are sorted; binary search is a future
        // optimization, not needed for contract tests).
        for (const auto& e : extents_) {
            if (logical_offset >= e.logical_offset &&
                logical_offset <  e.logical_offset + e.length) {
                return e.device_offset + (logical_offset - e.logical_offset);
            }
        }
        // Should not happen if validate() passed, but guard anyway.
        return Result<std::uint64_t>::Failure(
            Status(StatusCode::OUT_OF_RANGE,
                   "logical offset falls in a hole"));
    }

    // Self-check: validates that extents are:
    //   1. sorted by logical_offset ascending
    //   2. non-overlapping
    //   3. no holes (each extent starts where the previous ended)
    //   4. fully cover [0, file_size)
    //
    // Returns DATA_LOSS on any violation (corrupt extent map).
    Status validate() const {
        if (extents_.empty()) {
            if (file_size_ == 0) return Status::Ok();
            return Status(StatusCode::DATA_LOSS,
                          "non-zero file_size with no extents");
        }

        // Check first extent starts at 0.
        if (extents_[0].logical_offset != 0) {
            return Status(StatusCode::DATA_LOSS,
                          "first extent does not start at offset 0");
        }

        std::uint64_t expected_next = 0;
        for (std::size_t i = 0; i < extents_.size(); ++i) {
            const auto& e = extents_[i];

            // Check sorted ascending by logical_offset.
            if (e.logical_offset < expected_next) {
                return Status(StatusCode::DATA_LOSS,
                              "extents not sorted or overlap detected");
            }

            // Check for hole: current extent must start at expected_next.
            if (e.logical_offset != expected_next) {
                return Status(StatusCode::DATA_LOSS,
                              "hole detected before this extent");
            }

            // Check for overlap: already handled by the sorted check
            // (if e.logical_offset < expected_next, it overlaps or
            // is out of order).  The check above covers both.

            if (e.length == 0) {
                return Status(StatusCode::DATA_LOSS,
                              "zero-length extent");
            }

            expected_next = e.logical_offset + e.length;
        }

        // Check full coverage up to file_size.
        if (expected_next != file_size_) {
            return Status(StatusCode::DATA_LOSS,
                           "extents do not fully cover [0, file_size)");
        }

        return Status::Ok();
    }

private:
    Ext4LocalNvmePayload(NamespaceIdentity ns,
                         std::vector<Extent> extents,
                         std::uint64_t file_size)
        : ns_(std::move(ns)),
          extents_(std::move(extents)),
          file_size_(file_size) {}

    NamespaceIdentity ns_;
    std::vector<Extent> extents_;
    std::uint64_t file_size_ = 0;
};

// -------------------------------------------------------------------------
// Pairing helpers
//
// These are the SINGLE entry points that resolver and DataPath use.
// Neither caller needs to write the payload type id string or the
// API version as a bare literal.
//
// const-qualification alignment:
//   make_resolved_target receives shared_ptr<const Ext4LocalNvmePayload>
//   (immutable from the caller's perspective).  It uses
//   const_pointer_cast to obtain shared_ptr<Ext4LocalNvmePayload> for
//   storage in ResolvedTarget::make<Ext4LocalNvmePayload, OwnerLease>,
//   because the SPI stores payload as shared_ptr<void> and
//   shared_ptr<const T> does not implicitly convert to shared_ptr<void>.
//
//   This const_cast is safe: the object was created via new (non-const
//   storage) by Ext4LocalNvmePayload::create().  The const in the
//   shared_ptr<const ...> return type is a logical immutability
//   constraint, not a physical const qualification.
//
//   view_payload calls target.view<Ext4LocalNvmePayload>(...), which
//   does static_cast<const Ext4LocalNvmePayload*>(payload.get()).
//   The make template parameter and view template parameter are the
//   SAME non-const type (Ext4LocalNvmePayload), so the static_cast is
//   well-defined.  view returns const Ext4LocalNvmePayload*, preserving
//   read-only access at the API boundary.
// -------------------------------------------------------------------------

// Resolver side: pack payload + lease into a ResolvedTarget.
// Internally fixes the payload type id and API version.
template <typename OwnerLease>
inline Result<ResolvedTarget> make_resolved_target(
    std::string resolver_type_id,
    std::uint64_t logical_size,
    std::shared_ptr<const Ext4LocalNvmePayload> payload,
    std::shared_ptr<OwnerLease> owner_lease,
    std::string recommended_data_path_key =
        std::string(kRecommendedDataPathKey)) {

    // Safe const_cast: object was heap-allocated non-const by create().
    auto mutable_payload = std::const_pointer_cast<Ext4LocalNvmePayload>(
        std::move(payload));

    return ResolvedTarget::make<Ext4LocalNvmePayload, OwnerLease>(
        std::move(resolver_type_id),
        std::string(kPayloadTypeId),
        kPayloadApiVersion,
        logical_size,
        std::move(recommended_data_path_key),
        std::move(mutable_payload),
        std::move(owner_lease));
}

// DataPath side: checked extraction of the read-only payload.
// Internally fixes the expected type id and supported version.
inline Result<const Ext4LocalNvmePayload*>
view_payload(const ResolvedTarget& target) {
    return target.view<Ext4LocalNvmePayload>(
        kPayloadTypeId, kPayloadApiVersion);
}

} // namespace tutti::binding::ext4_local_nvme

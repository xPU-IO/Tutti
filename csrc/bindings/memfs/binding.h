#pragma once

// tutti/bindings/memfs/binding.h
//
// SAMPLE-ONLY binding: the private payload contract shared between the
// memfs resolver and the memfs DataPath.
//
// This is a community-extension sample (doc/extending_tutti.md).  It is
// NOT a production backend.  The payload type, identity constants, and
// pairing helpers live entirely in this sample package — no core header
// or Runtime source file references any type declared here.
//
// Header-only C++17.  Depends only on <tutti/status.h> and
// <tutti/spi/storage_target_resolver.h> plus the standard library.

#include <tutti/status.h>
#include <tutti/spi/storage_target_resolver.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tutti::binding::memfs {

// -------------------------------------------------------------------------
// Identity constants — the single declaration point for payload type id,
// API version, recommended DataPath key, and resolver type id.
// -------------------------------------------------------------------------

inline constexpr std::string_view kPayloadTypeId =
    "memfs-payload-v1";

inline constexpr std::uint32_t kPayloadApiVersion = 1;

inline constexpr std::string_view kRecommendedDataPathKey =
    "memfs";

inline constexpr std::string_view kResolverTypeId =
    "memfs-resolver-v1";

// -------------------------------------------------------------------------
// MemfsPayload
//
// Immutable payload produced by the memfs resolver and consumed by the
// memfs DataPath.  Contains a shared backing buffer (host memory that
// simulates a storage device) and the logical size.
//
// The backing buffer is zero-initialized on creation.  The DataPath reads
// from and writes to this buffer via memcpy.
// -------------------------------------------------------------------------

class MemfsPayload {
public:
    static Result<std::shared_ptr<const MemfsPayload>>
    create(std::uint64_t logical_size) {
        if (logical_size == 0) {
            return Result<std::shared_ptr<const MemfsPayload>>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "memfs logical size must be > 0"));
        }
        auto backing = std::make_shared<std::vector<std::uint8_t>>(
            static_cast<std::size_t>(logical_size), 0);
        auto tmp = std::unique_ptr<MemfsPayload>(
            new MemfsPayload(std::move(backing), logical_size));
        return std::shared_ptr<const MemfsPayload>(tmp.release());
    }

    const std::vector<std::uint8_t>& backing() const noexcept {
        return *backing_;
    }

    std::uint64_t logical_size() const noexcept {
        return logical_size_;
    }

    // Non-const backing access for the DataPath (write path).
    // The DataPath holds a shared_ptr<MemfsPayload> (non-const) internally
    // so it can mutate the backing buffer for writes.
    std::vector<std::uint8_t>& mutable_backing() const noexcept {
        return *backing_;
    }

private:
    MemfsPayload(std::shared_ptr<std::vector<std::uint8_t>> backing,
                 std::uint64_t logical_size)
        : backing_(std::move(backing)), logical_size_(logical_size) {}

    // shared_ptr so both the ResolvedTarget (via payload) and the DataPath
    // (via open()) keep the backing buffer alive.
    std::shared_ptr<std::vector<std::uint8_t>> backing_;
    std::uint64_t logical_size_ = 0;
};

// -------------------------------------------------------------------------
// MemfsOwnerLease
//
// Trivial lease — memfs owns no external resources (no fd, no device
// handle).  The backing buffer lifetime is managed by the payload's
// shared_ptr.  The lease exists only to satisfy the ResolvedTarget
// factory's non-null requirement.
// -------------------------------------------------------------------------

struct MemfsOwnerLease {};

// -------------------------------------------------------------------------
// Pairing helpers — the single entry points used by resolver and DataPath.
// -------------------------------------------------------------------------

// Resolver side: pack payload + lease into a ResolvedTarget.
template <typename OwnerLease = MemfsOwnerLease>
inline Result<ResolvedTarget> make_resolved_target(
    std::uint64_t logical_size,
    std::shared_ptr<const MemfsPayload> payload,
    std::shared_ptr<OwnerLease> owner_lease = std::make_shared<OwnerLease>(),
    std::string recommended_data_path_key =
        std::string(kRecommendedDataPathKey)) {

    // Safe const_cast: object was heap-allocated non-const by create().
    auto mutable_payload = std::const_pointer_cast<MemfsPayload>(
        std::move(payload));

    return ResolvedTarget::make<MemfsPayload, OwnerLease>(
        std::string(kResolverTypeId),
        std::string(kPayloadTypeId),
        kPayloadApiVersion,
        logical_size,
        std::move(recommended_data_path_key),
        std::move(mutable_payload),
        std::move(owner_lease));
}

// DataPath side: checked extraction of the read-only payload.
inline Result<const MemfsPayload*>
view_payload(const ResolvedTarget& target) {
    return target.view<MemfsPayload>(
        kPayloadTypeId, kPayloadApiVersion);
}

} // namespace tutti::binding::memfs

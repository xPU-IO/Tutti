#pragma once

// tutti/spi/storage_target_resolver.h
//
// Phase 1 source-level SPI for namespace/name → target resolution.
//
// ResolvedTarget is a type-erased, move-only, owner-bearing value:
//   - holds shared_ptr to immutable payload and shared_ptr to owner lease
//   - DataPath obtains a const Payload* via checked view<Payload>()
//   - payload type identity uses string + uint32_t API version, not RTTI
//   - no common variant/union; no vendor or transport types in the shell
//
// Header-only C++17. Depends only on <tutti/status.h> and the standard library.

#include <tutti/status.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace tutti {

// -------------------------------------------------------------------------
// ResolveOptions
//
// Minimal options carried to the resolver.  The URI scheme selects which
// resolver is invoked; options carry supplementary hints.
// This round adds no fields beyond scheme to avoid over-design.
// -------------------------------------------------------------------------
struct ResolveOptions {
    std::string scheme;
};

// -------------------------------------------------------------------------
// ResolvedTarget
//
// Move-only, type-erased target descriptor produced by a resolver.
// Owns (via shared_ptr) an immutable payload and an owner lease.
// The payload is opaque to Runtime; DataPath accesses it through a
// checked, source-level view that verifies payload_type_id and
// source_api_version before handing back a const pointer.
// -------------------------------------------------------------------------
class ResolvedTarget {
public:
    // Default: empty shell. valid() == false, view() returns UNSUPPORTED.
    ResolvedTarget() = default;
    ~ResolvedTarget() = default;

    // Move-only.
    ResolvedTarget(ResolvedTarget&&) noexcept = default;
    ResolvedTarget& operator=(ResolvedTarget&&) noexcept = default;
    ResolvedTarget(const ResolvedTarget&) = delete;
    ResolvedTarget& operator=(const ResolvedTarget&) = delete;

    // ---- Metadata accessors (empty target returns zero/empty) ----

    std::string_view resolver_type_id() const noexcept;
    std::string_view payload_type_id() const noexcept;
    std::uint32_t source_api_version() const noexcept;
    std::uint64_t logical_size() const noexcept;
    std::string_view recommended_data_path_key() const noexcept;
    bool valid() const noexcept;

    // ---- Checked payload view ----
    //
    // Returns a const Payload* when:
    //   expected_payload_type_id == recorded payload_type_id
    //   supported_api_version    == recorded source_api_version
    // Otherwise returns UNSUPPORTED.
    //
    // The returned pointer borrows from this ResolvedTarget's payload
    // owner; it does NOT extend the lease.  Caller must ensure the
    // ResolvedTarget outlives all uses of the pointer.
    template <typename Payload>
    Result<const Payload*> view(
        std::string_view expected_payload_type_id,
        std::uint32_t supported_api_version) const;

    // ---- Factory ----
    //
    // Creates a ResolvedTarget owning the given immutable payload and
    // owner lease.  Both shared_ptrs must be non-null.
    template <typename Payload, typename OwnerLease>
    static Result<ResolvedTarget> make(
        std::string resolver_type_id,
        std::string payload_type_id,
        std::uint32_t source_api_version,
        std::uint64_t logical_size,
        std::string recommended_data_path_key,
        std::shared_ptr<Payload> immutable_payload,
        std::shared_ptr<OwnerLease> owner_lease);

private:
    // Type-erased storage.  payload and lease are shared_ptr<void> so
    // the shell does not expose any specific payload or lease type.
    // The original types are recovered in view<P>() via static_cast,
    // which is safe because the same P was used in make<P, L>().
    struct Storage {
        std::string resolver_type_id;
        std::string payload_type_id;
        std::uint32_t source_api_version = 0;
        std::uint64_t logical_size = 0;
        std::string recommended_data_path_key;
        std::shared_ptr<void> payload;
        std::shared_ptr<void> lease;
    };

    explicit ResolvedTarget(std::unique_ptr<Storage> s)
        : storage_(std::move(s)) {}

    std::unique_ptr<Storage> storage_;
};

// -------------------------------------------------------------------------
// StorageTargetResolver
//
// Abstract SPI: resolve a URI + options to a ResolvedTarget.
// Implementations open/read whatever namespace they understand
// (extent maps, KV-store, etc.) but must NOT submit IO or understand
// PRP/CQ/kernel internals.
// -------------------------------------------------------------------------
class StorageTargetResolver {
public:
    virtual ~StorageTargetResolver() = default;
    virtual Result<ResolvedTarget> resolve(
        std::string_view uri,
        const ResolveOptions& options) = 0;
};

// =========================================================================
// Inline / template implementations
// =========================================================================

inline std::string_view
ResolvedTarget::resolver_type_id() const noexcept {
    return storage_
        ? std::string_view(storage_->resolver_type_id)
        : std::string_view{};
}

inline std::string_view
ResolvedTarget::payload_type_id() const noexcept {
    return storage_
        ? std::string_view(storage_->payload_type_id)
        : std::string_view{};
}

inline std::uint32_t
ResolvedTarget::source_api_version() const noexcept {
    return storage_ ? storage_->source_api_version : 0;
}

inline std::uint64_t
ResolvedTarget::logical_size() const noexcept {
    return storage_ ? storage_->logical_size : 0;
}

inline std::string_view
ResolvedTarget::recommended_data_path_key() const noexcept {
    return storage_
        ? std::string_view(storage_->recommended_data_path_key)
        : std::string_view{};
}

inline bool
ResolvedTarget::valid() const noexcept {
    return storage_ != nullptr;
}

template <typename Payload>
inline Result<const Payload*>
ResolvedTarget::view(
    std::string_view expected_payload_type_id,
    std::uint32_t supported_api_version) const {

    if (!storage_) {
        return Result<const Payload*>::Failure(
            Status(StatusCode::UNSUPPORTED, "target is empty"));
    }
    if (std::string_view(storage_->payload_type_id)
        != expected_payload_type_id) {
        return Result<const Payload*>::Failure(
            Status(StatusCode::UNSUPPORTED, "payload type mismatch"));
    }
    if (storage_->source_api_version != supported_api_version) {
        return Result<const Payload*>::Failure(
            Status(StatusCode::UNSUPPORTED, "API version mismatch"));
    }
    // Safe: the pointer was originally a Payload* stored via
    // make<Payload, ...>().  static_cast back to const Payload* is
    // well-defined because the object IS a Payload.
    return static_cast<const Payload*>(storage_->payload.get());
}

template <typename Payload, typename OwnerLease>
inline Result<ResolvedTarget>
ResolvedTarget::make(
    std::string resolver_type_id,
    std::string payload_type_id,
    std::uint32_t source_api_version,
    std::uint64_t logical_size,
    std::string recommended_data_path_key,
    std::shared_ptr<Payload> immutable_payload,
    std::shared_ptr<OwnerLease> owner_lease) {

    if (!immutable_payload) {
        return Result<ResolvedTarget>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "immutable_payload is null"));
    }
    if (!owner_lease) {
        return Result<ResolvedTarget>::Failure(
            Status(StatusCode::INVALID_ARGUMENT,
                   "owner_lease is null"));
    }

    auto s = std::make_unique<Storage>();
    s->resolver_type_id         = std::move(resolver_type_id);
    s->payload_type_id          = std::move(payload_type_id);
    s->source_api_version       = source_api_version;
    s->logical_size             = logical_size;
    s->recommended_data_path_key = std::move(recommended_data_path_key);
    // shared_ptr<Payload> → shared_ptr<void>: control block preserved,
    // deleter will correctly destroy the Payload.
    s->payload                  = std::move(immutable_payload);
    s->lease                    = std::move(owner_lease);

    return ResolvedTarget(std::move(s));
}

} // namespace tutti

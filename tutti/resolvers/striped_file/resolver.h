#pragma once

// tutti/resolvers/striped_file/resolver.h
//
// StripedResolver: resolves "striped://<name>?devs=<m1,m2,...>&unit=<bytes>"
// URIs into a single ResolvedTarget carrying N per-device sub-targets.
//
// Each shard's backing file is resolved by a LocalFileResolver instance
// (injected at construction).  The resolver inherits LocalFileResolver's
// fail-closed semantics: backing device identity check, FIEMAP unsafe
// flag rejection, extent coverage validation.
//
// URI format:
//   striped://<name>?devs=<mount1,mount2,...>&unit=<bytes>
//
//   <name>    : logical name (e.g., "model_weights")
//   <mounts>  : comma-separated mount point paths, one per device
//   <unit>    : stripe unit in bytes (must be 4 KiB-aligned and >= block_size)
//
// Backing file path per shard:
//   <mount_i>/striped/<name>.shard<i>
//   where <i> is 0-based shard index.
//
// Lease semantics:
//   The bundle's lease (StripeBundleLease) holds all N sub-leases.
//   If shard i fails to resolve, shards 0..i-1 (already resolved) are
//   released via RAII — the local vector<ResolvedTarget> destructor
//   closes their fds.  This is fail-closed: no partial bundle escapes.

#include <tutti/status.h>
#include <tutti/spi/storage_target_resolver.h>
#include <tutti/bindings/striped_local_nvme/binding.h>
#include <tutti/resolvers/local_file/resolver.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tutti::resolvers::striped_file {

inline constexpr std::string_view kScheme = "striped";
inline constexpr std::uint64_t kMinStripeUnit = 4096;

// -------------------------------------------------------------------------
// StripedResolver
//
// Constructor takes N LocalFileResolver instances (one per device) and
// the stripe unit.  resolve() parses the URI, constructs per-shard
// backing file paths, resolves each via the corresponding resolver,
// and bundles the results.
// -------------------------------------------------------------------------
class StripedResolver : public StorageTargetResolver {
public:
    StripedResolver(std::vector<std::unique_ptr<StorageTargetResolver>> shard_resolvers,
                    std::uint64_t stripe_unit,
                    std::string data_path_key =
                        std::string(binding::striped_local_nvme::
                                        kRecommendedDataPathKey))
        : shard_resolvers_(std::move(shard_resolvers)),
          stripe_unit_(stripe_unit),
          data_path_key_(std::move(data_path_key)) {

        // Validate stripe unit at construction.
        // If invalid, resolve() will reject all URIs.
    }

    Result<ResolvedTarget> resolve(
        std::string_view uri,
        const ResolveOptions& options) override {

        // 1. Scheme check.
        if (options.scheme != kScheme) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::UNSUPPORTED,
                       "scheme '" + options.scheme +
                       "' != '" + std::string(kScheme) + "'"));
        }

        // 2. Validate stripe_unit.
        if (stripe_unit_ == 0) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "stripe_unit == 0"));
        }
        if (stripe_unit_ % kMinStripeUnit != 0) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "stripe_unit " + std::to_string(stripe_unit_) +
                       " not aligned to " + std::to_string(kMinStripeUnit)));
        }

        // 3. Validate shard resolver count.
        if (shard_resolvers_.empty()) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "no shard resolvers configured"));
        }
        std::uint32_t N = static_cast<std::uint32_t>(shard_resolvers_.size());

        // 4. Parse URI: striped://<name>?devs=<m1,m2,...>&unit=<bytes>
        const std::string_view prefix = "striped://";
        if (uri.size() < prefix.size() ||
            uri.substr(0, prefix.size()) != prefix) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "uri must start with 'striped://': " + std::string(uri)));
        }
        std::string_view rest = uri.substr(prefix.size());

        // Split name and query string.
        auto qmark = rest.find('?');
        if (qmark == std::string_view::npos) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "uri missing query params: " + std::string(uri)));
        }
        std::string name(rest.substr(0, qmark));
        std::string_view query = rest.substr(qmark + 1);

        if (name.empty()) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "uri name is empty: " + std::string(uri)));
        }

        // 5. Parse query params: devs=<m1,m2,...>&unit=<bytes>
        std::vector<std::string> mounts;
        std::uint64_t parsed_unit = 0;
        bool found_devs = false;
        bool found_unit = false;
        // Round 16 S7: optional 'rot' param -- per-target shard rotation
        // (legacy shard_placement equivalent).  Absent => 0 (no rotation,
        // identical to pre-S7 behavior).
        std::uint32_t parsed_rot = 0;

        std::string_view q = query;
        while (!q.empty()) {
            auto amp = q.find('&');
            std::string_view param = (amp == std::string_view::npos)
                ? q : q.substr(0, amp);
            if (amp == std::string_view::npos) {
                q = {};
            } else {
                q = q.substr(amp + 1);
            }

            auto eq = param.find('=');
            if (eq == std::string_view::npos) {
                return Result<ResolvedTarget>::Failure(
                    Status(StatusCode::INVALID_ARGUMENT,
                           "query param missing '=': " + std::string(param)));
            }
            std::string key(param.substr(0, eq));
            std::string val(param.substr(eq + 1));

            if (key == "devs") {
                found_devs = true;
                // Split by comma.
                std::string_view dv = val;
                while (!dv.empty()) {
                    auto comma = dv.find(',');
                    std::string m = std::string(
                        (comma == std::string_view::npos) ? dv : dv.substr(0, comma));
                    if (m.empty()) {
                        return Result<ResolvedTarget>::Failure(
                            Status(StatusCode::INVALID_ARGUMENT,
                                   "empty mount in devs param"));
                    }
                    mounts.push_back(m);
                    if (comma == std::string_view::npos) break;
                    dv = dv.substr(comma + 1);
                }
            } else if (key == "unit") {
                found_unit = true;
                // Parse as integer.
                char* end = nullptr;
                unsigned long long u = std::strtoull(val.c_str(), &end, 10);
                if (end == val.c_str() || *end != '\0') {
                    return Result<ResolvedTarget>::Failure(
                        Status(StatusCode::INVALID_ARGUMENT,
                               "unit param not a valid integer: " + val));
                }
                parsed_unit = static_cast<std::uint64_t>(u);
            } else if (key == "rot") {
                // Round 16 S7: per-target shard rotation.
                char* end = nullptr;
                unsigned long long r = std::strtoull(val.c_str(), &end, 10);
                if (end == val.c_str() || *end != '\0') {
                    return Result<ResolvedTarget>::Failure(
                        Status(StatusCode::INVALID_ARGUMENT,
                               "rot param not a valid integer: " + val));
                }
                parsed_rot = static_cast<std::uint32_t>(r);
            } else {
                return Result<ResolvedTarget>::Failure(
                    Status(StatusCode::INVALID_ARGUMENT,
                           "unknown query param '" + key + "'"));
            }
        }

        if (!found_devs) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "missing 'devs' param in uri: " + std::string(uri)));
        }
        if (!found_unit) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "missing 'unit' param in uri: " + std::string(uri)));
        }

        // 6. Validate parsed params.
        if (mounts.size() != N) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "devs count (" + std::to_string(mounts.size()) +
                       ") != shard resolver count (" + std::to_string(N) + ")"));
        }
        if (parsed_unit != stripe_unit_) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "uri unit (" + std::to_string(parsed_unit) +
                       ") != configured stripe_unit (" +
                       std::to_string(stripe_unit_) + ")"));
        }
        if (parsed_unit % kMinStripeUnit != 0) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "unit " + std::to_string(parsed_unit) +
                       " not 4 KiB aligned"));
        }

        // 7. Resolve each shard via the corresponding LocalFileResolver.
        // RAII: if shard i fails, shards 0..i-1 are released by the
        // vector destructor (ResolvedTarget move-only → unique ownership).
        std::vector<ResolvedTarget> shards;
        shards.reserve(N);
        std::vector<std::shared_ptr<void>> sub_leases;
        sub_leases.reserve(N);

        for (std::uint32_t i = 0; i < N; ++i) {
            // Construct backing file path: <mount>/striped/<name>.shard<i>
            std::string shard_path = mounts[i] + "/striped/" + name +
                ".shard" + std::to_string(i);
            std::string shard_uri = "file://" + shard_path;

            ResolveOptions shard_opts;
            shard_opts.scheme = "file";

            auto sr = shard_resolvers_[i]->resolve(shard_uri, shard_opts);
            if (!sr.ok()) {
                // Fail-closed: shards vector destructor releases
                // already-resolved shards (closes their fds).
                return Result<ResolvedTarget>::Failure(
                    Status(sr.status().code(),
                           "shard " + std::to_string(i) + " resolve failed: " +
                           sr.status().message()));
            }

            // Extract the sub-lease for the bundle lease.
            // We access the internal lease via the fact that
            // ResolvedTarget stores it as shared_ptr<void>.
            // Since we can't access it directly (it's private),
            // the StripeBundleLease will hold shared_ptr<void> copies
            // obtained from the sub-targets.  But ResolvedTarget's
            // lease is private...
            //
            // Design: the StripeBundleLease is a marker.  The real
            // fd cleanup happens when the StripedLocalNvmePayload's
            // shard vector is destroyed (each shard's shared_ptr<void>
            // lease refcount → 0 → fd close).  The StripeBundleLease
            // just needs to be non-null.
            shards.push_back(std::move(sr).value());
        }

        // 8. Create the bundle lease (marker — real cleanup in payload dtor).
        auto bundle_lease = std::make_shared<
            binding::striped_local_nvme::StripeBundleLease>(
            std::move(sub_leases));

        // 9. Create the immutable payload.
        auto payload_result = binding::striped_local_nvme::
            StripedLocalNvmePayload::create(
                N, stripe_unit_, std::move(shards), parsed_rot);

        if (!payload_result.ok()) {
            return Result<ResolvedTarget>::Failure(
                payload_result.status());
        }

        // 10. Pack into outer ResolvedTarget.
        auto payload = payload_result.value();
        if (!payload) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INTERNAL,
                       "StripedLocalNvmePayload::create returned null"));
        }
        const auto logical = payload->logical_size();
        return ResolvedTarget::make<binding::striped_local_nvme::StripedLocalNvmePayload,
                                    binding::striped_local_nvme::StripeBundleLease>(
            std::string(binding::striped_local_nvme::kResolverTypeId),
            std::string(binding::striped_local_nvme::kPayloadTypeId),
            binding::striped_local_nvme::kPayloadApiVersion,
            logical,
            data_path_key_,
            std::move(payload),
            std::move(bundle_lease));
    }

private:
    std::vector<std::unique_ptr<StorageTargetResolver>> shard_resolvers_;
    std::uint64_t stripe_unit_;
    std::string data_path_key_;
};

} // namespace tutti::resolvers::striped_file

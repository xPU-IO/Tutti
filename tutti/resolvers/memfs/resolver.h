#pragma once

// tutti/resolvers/memfs/resolver.h
//
// SAMPLE-ONLY resolver for the memfs binding.  Parses `memfs://<size>`
// URIs and produces a ResolvedTarget backed by an in-memory buffer.
//
// This is a community-extension sample, NOT a production resolver.  It
// proves that a new resolver can be added without modifying core Runtime
// or SPI headers.
//
// URI format:
//   memfs://<size>
//
//   <size> is a decimal byte count (e.g. 4096, 1048576).  The resolver
//   allocates a zero-initialized backing buffer of that size and packs it
//   into a MemfsPayload.
//
// Header-only C++17.  Depends only on public/SPI headers, the memfs
// binding header, and the standard library.

#include <tutti/status.h>
#include <tutti/spi/storage_target_resolver.h>
#include "tutti/bindings/memfs/binding.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace tutti::resolver::memfs {

// -------------------------------------------------------------------------
// MemfsResolver — resolves memfs:// URIs to in-memory targets.
// -------------------------------------------------------------------------

class MemfsResolver : public StorageTargetResolver {
public:
    explicit MemfsResolver(
        std::uint64_t capacity_bytes = 0,
        std::string data_path_key =
            std::string(tutti::binding::memfs::kRecommendedDataPathKey))
        : capacity_bytes_(capacity_bytes),
          data_path_key_(std::move(data_path_key)) {}

    Result<ResolvedTarget> resolve(
        std::string_view uri,
        const ResolveOptions& options) override {

        // Verify scheme (optional: the Runtime already routes by scheme,
        // but a resolver should be defensive).
        if (options.scheme != "memfs") {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "memfs resolver requires scheme 'memfs'"));
        }

        // Parse memfs://<size>
        constexpr std::string_view kPrefix = "memfs://";
        if (uri.size() <= kPrefix.size() ||
            uri.substr(0, kPrefix.size()) != kPrefix) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "URI must start with 'memfs://'"));
        }

        std::string_view size_str = uri.substr(kPrefix.size());
        if (size_str.empty()) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "memfs URI requires a size: memfs://<bytes>"));
        }

        // Parse decimal size.
        // Use strtoull for overflow-safe parsing.
        char* end = nullptr;
        errno = 0;
        std::string tmp(size_str);
        unsigned long long size = std::strtoull(tmp.c_str(), &end, 10);
        if (errno != 0 || end == tmp.c_str() || *end != '\0') {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "memfs URI size is not a valid decimal number"));
        }
        if (size == 0) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::INVALID_ARGUMENT,
                       "memfs URI size must be > 0"));
        }
        if (capacity_bytes_ > 0 && size > capacity_bytes_) {
            return Result<ResolvedTarget>::Failure(
                Status(StatusCode::RESOURCE_EXHAUSTED,
                       "memfs URI size exceeds Resource capacity"));
        }

        // Create the payload (backing buffer).
        auto payload = tutti::binding::memfs::MemfsPayload::create(
            static_cast<std::uint64_t>(size));
        if (!payload.ok()) {
            return Result<ResolvedTarget>::Failure(payload.status());
        }

        // Pack into a ResolvedTarget.
        return tutti::binding::memfs::make_resolved_target(
            static_cast<std::uint64_t>(size),
            std::move(payload).value(),
            std::make_shared<tutti::binding::memfs::MemfsOwnerLease>(),
            data_path_key_);
    }

private:
    std::uint64_t capacity_bytes_ = 0;
    std::string data_path_key_;
};

} // namespace tutti::resolver::memfs

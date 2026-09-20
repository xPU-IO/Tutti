#pragma once

#include <cstdint>

namespace tutti::config {

inline constexpr std::uint64_t kDefaultStripedStripeUnit = 512 * 1024;

struct StripedLocalNvmeBackendConfig {
    std::uint64_t stripe_unit = kDefaultStripedStripeUnit;
};

} // namespace tutti::config

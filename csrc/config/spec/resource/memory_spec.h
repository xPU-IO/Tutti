#pragma once

#include <cstdint>

namespace tutti::config {

struct MemoryResourceConfig {
    std::uint64_t capacity_bytes = 0;
};

} // namespace tutti::config

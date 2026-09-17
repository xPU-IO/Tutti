#pragma once

#include <cstdint>

namespace tutti::config {

struct NvmeDataPathTuning {
    static constexpr std::uint32_t kDefaultThreadsPerBlock = 16;

    std::uint32_t handle_cache_capacity = 0;
    std::uint32_t prp_cache_capacity = 0;
    std::uint32_t handle_cache_l2_capacity = 0;
    std::uint32_t threads_per_block = kDefaultThreadsPerBlock;
    std::uint64_t max_in_flight_operations = 0;
    std::uint64_t max_batch_entries = 0;
    std::uint64_t io_granularity = 0;
};

} // namespace tutti::config

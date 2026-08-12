#pragma once

#include "tutti/config/parser/parser_internal.h"

namespace tutti::config::detail {

template <typename Config>
Status parse_nvme_datapath_tuning(const YAML::Node& config,
                                  const std::string& path,
                                  Config& parsed) {
    if (!config) return Status::Ok();
    Status status = validate_keys(
        config, path,
        {"handle_cache_capacity", "prp_cache_capacity",
         "handle_cache_l2_capacity", "threads_per_block",
         "max_in_flight_operations", "max_batch_entries", "io_granularity"});
    if (!status.ok()) return status;
    status = read_optional_u32(config, "handle_cache_capacity", path,
                               parsed.handle_cache_capacity);
    if (!status.ok()) return status;
    status = read_optional_u32(config, "prp_cache_capacity", path,
                               parsed.prp_cache_capacity);
    if (!status.ok()) return status;
    status = read_optional_u32(config, "handle_cache_l2_capacity", path,
                               parsed.handle_cache_l2_capacity);
    if (!status.ok()) return status;
    status = read_optional_u32(config, "threads_per_block", path,
                               parsed.threads_per_block);
    if (!status.ok()) return status;
    status = read_optional_u64(config, "max_in_flight_operations", path,
                               parsed.max_in_flight_operations);
    if (!status.ok()) return status;
    status = read_optional_u64(config, "max_batch_entries", path,
                               parsed.max_batch_entries);
    if (!status.ok()) return status;
    return read_optional_u64(config, "io_granularity", path,
                             parsed.io_granularity);
}

} // namespace tutti::config::detail

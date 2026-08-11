#pragma once

#include "tutti/config/storage/parse_internal.h"

namespace tutti::config::detail {

inline Status parse_nvme_datapath_tuning(const YAML::Node& config,
                                         const std::string& context,
                                         DataPathSpec& datapath) {
    if (!config) return Status::Ok();
    Status status = validate_keys(
        config, context,
        {"handle_cache_capacity", "prp_cache_capacity",
         "handle_cache_l2_capacity", "max_in_flight_operations",
         "max_batch_entries", "io_granularity"});
    if (!status.ok()) return status;
    status = read_nonnegative_u32(config, "handle_cache_capacity", context,
                                  datapath.handle_cache_capacity);
    if (!status.ok()) return status;
    status = read_nonnegative_u32(config, "prp_cache_capacity", context,
                                  datapath.prp_cache_capacity);
    if (!status.ok()) return status;
    status = read_nonnegative_u32(config, "handle_cache_l2_capacity", context,
                                  datapath.handle_cache_l2_capacity);
    if (!status.ok()) return status;
    status = read_nonnegative_u64(config, "max_in_flight_operations", context,
                                  datapath.max_in_flight_operations);
    if (!status.ok()) return status;
    status = read_nonnegative_u64(config, "max_batch_entries", context,
                                  datapath.max_batch_entries);
    if (!status.ok()) return status;
    return read_nonnegative_u64(config, "io_granularity", context,
                                datapath.io_granularity);
}

} // namespace tutti::config::detail

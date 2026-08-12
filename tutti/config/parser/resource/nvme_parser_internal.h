#pragma once

#include <limits>

#include "tutti/config/parser/parser_internal.h"

namespace tutti::config::detail {

inline Result<NvmeSelection> parse_nvme_selection(
    const std::string& value, const std::string& path) {
    if (value == "allowed") {
        return Result<NvmeSelection>::Success(NvmeSelection::Allowed);
    }
    if (value == "explicit") {
        return Result<NvmeSelection>::Success(NvmeSelection::Explicit);
    }
    if (value == "striped") {
        return Result<NvmeSelection>::Success(NvmeSelection::Striped);
    }
    return Result<NvmeSelection>::Failure(parse_error(
        path + " must be allowed, explicit, or striped"));
}

inline Status parse_nvme_device_ids(
    const YAML::Node& node, const std::string& path,
    std::vector<std::int32_t>& device_ids) {
    if (!node) return Status::Ok();
    if (!node.IsSequence()) {
        return parse_error(path + " must be a sequence");
    }
    for (const YAML::Node& entry : node) {
        if (!entry.IsScalar()) {
            return parse_error(path + " entries must be int32 scalars");
        }
        std::int64_t parsed = 0;
        try {
            parsed = entry.as<std::int64_t>();
        } catch (const YAML::Exception&) {
            return parse_error(path +
                               " entries must be non-negative int32 values");
        }
        if (parsed < 0 || parsed > std::numeric_limits<std::int32_t>::max()) {
            return parse_error(path +
                               " entries must be non-negative int32 values");
        }
        device_ids.push_back(static_cast<std::int32_t>(parsed));
    }
    return Status::Ok();
}

} // namespace tutti::config::detail

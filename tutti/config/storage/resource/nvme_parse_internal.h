#pragma once

#include <limits>
#include <unordered_set>

#include "tutti/config/storage/parse_internal.h"

namespace tutti::config::detail {

inline Result<NvmeSelection> parse_nvme_selection(const std::string& value) {
    if (value == "allowed") {
        return Result<NvmeSelection>::Success(NvmeSelection::Allowed);
    }
    if (value == "explicit") {
        return Result<NvmeSelection>::Success(NvmeSelection::Explicit);
    }
    if (value == "striped") {
        return Result<NvmeSelection>::Success(NvmeSelection::Striped);
    }
    return Result<NvmeSelection>::Failure(
        invalid("selection must be allowed, explicit, or striped"));
}

inline const char* nvme_selection_name(NvmeSelection selection) {
    switch (selection) {
    case NvmeSelection::Allowed:
        return "allowed";
    case NvmeSelection::Explicit:
        return "explicit";
    case NvmeSelection::Striped:
        return "striped";
    }
    return "allowed";
}

inline Status parse_nvme_device_ids(
    const YAML::Node& node, const std::string& context,
    std::vector<std::int32_t>& device_ids) {
    if (!node) return Status::Ok();
    if (!node.IsSequence()) {
        return invalid(context + " must be a sequence");
    }
    std::unordered_set<std::int32_t> unique;
    for (const auto& entry : node) {
        const std::int64_t parsed = entry.as<std::int64_t>();
        if (parsed < 0 || parsed > std::numeric_limits<std::int32_t>::max()) {
            return invalid(
                context + " entries must be non-negative int32 values");
        }
        const auto device_id = static_cast<std::int32_t>(parsed);
        if (!unique.emplace(device_id).second) {
            return invalid(context + " must not contain duplicates");
        }
        device_ids.push_back(device_id);
    }
    return Status::Ok();
}

inline Status validate_nvme_selection(const AllocationSpec& allocation,
                                      const std::string& context) {
    if (allocation.selection == NvmeSelection::Allowed &&
        !allocation.device_ids.empty()) {
        return invalid(
            context + " selection=allowed requires empty device_ids");
    }
    if (allocation.selection == NvmeSelection::Explicit &&
        allocation.device_ids.size() != 1) {
        return invalid(
            context + " selection=explicit requires exactly one device_id");
    }
    if (allocation.selection == NvmeSelection::Striped &&
        allocation.device_ids.size() < 2) {
        return invalid(
            context + " selection=striped requires at least two device_ids");
    }
    return Status::Ok();
}

} // namespace tutti::config::detail

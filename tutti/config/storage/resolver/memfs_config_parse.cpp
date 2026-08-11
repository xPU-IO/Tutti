#include "tutti/config/storage/parse_internal.h"

namespace tutti::config::detail {

Status parse_memfs_resolver_config(const YAML::Node& config,
                                   const std::string& context,
                                   ResolverSpec&) {
    if (!config) return Status::Ok();
    return validate_keys(config, context, {});
}

} // namespace tutti::config::detail

#include "tutti/config/parser/parser_internal.h"

namespace tutti::config::detail {
Status parse_local_file_resolver(const YAML::Node& config,
                                 const std::string& path,
                                 ResolverSpec& resolver) {
    if (config) {
        Status status = validate_keys(config, path, {});
        if (!status.ok()) return status;
    }
    resolver.config = LocalFileResolverConfig{};
    return Status::Ok();
}
} // namespace tutti::config::detail

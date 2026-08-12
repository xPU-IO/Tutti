#include "tutti/config/parser/parser_internal.h"

namespace tutti::config::detail {
Status parse_memfs_datapath(const YAML::Node& config,
                            const std::string& path,
                            DataPathSpec& datapath) {
    if (config) {
        Status status = validate_keys(config, path, {});
        if (!status.ok()) return status;
    }
    datapath.config = MemfsDataPathConfig{};
    return Status::Ok();
}
} // namespace tutti::config::detail

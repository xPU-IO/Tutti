#include "tutti/config/parser/parser_internal.h"

namespace tutti::config::detail {
Status parse_ext4_local_nvme_backend(const YAML::Node& config,
                                     const std::string& path,
                                     BackendSpec& backend) {
    if (config) {
        Status status = validate_keys(config, path, {});
        if (!status.ok()) return status;
    }
    backend.config = Ext4LocalNvmeBackendConfig{};
    return Status::Ok();
}
} // namespace tutti::config::detail

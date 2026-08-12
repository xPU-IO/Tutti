#include "tutti/config/parser/parser_internal.h"

namespace tutti::config::detail {
Status parse_striped_local_nvme_backend(const YAML::Node& config,
                                        const std::string& path,
                                        BackendSpec& backend) {
    StripedLocalNvmeBackendConfig parsed;
    if (config) {
        Status status = validate_keys(config, path, {"stripe_unit"});
        if (!status.ok()) return status;
        status = read_optional_u64(config, "stripe_unit", path,
                                   parsed.stripe_unit);
        if (!status.ok()) return status;
    }
    backend.config = parsed;
    return Status::Ok();
}
} // namespace tutti::config::detail

#include "csrc/config/parser/parser_internal.h"

namespace tutti::config::detail {
Status parse_striped_local_nvme_backend(const YAML::Node& config,
                                        const std::string& path,
                                        BackendSpec& backend) {
    // 2026-09-22：条带化删除后本后端无可配置项（stripe_unit 已移除）；
    // 空 config（`config: {}` 或缺省）即合法形态，出现任何键都是配置错误。
    if (config && config.size() > 0) {
        return parse_error(path + " accepts no keys (stripe_unit was removed)");
    }
    backend.config = StripedLocalNvmeBackendConfig{};
    return Status::Ok();
}
} // namespace tutti::config::detail

#include "tutti/config/spec/spec_internal.h"

namespace tutti::config::detail {
Status validate_local_nvme_datapath(const DataPathSpec& spec,
                                    const std::string& path) {
    const auto* config =
        std::get_if<LocalNvmeDataPathConfig>(&spec.config);
    if (config == nullptr) {
        return invalid_spec(path + ".config does not match type local-nvme");
    }
    if (config->threads_per_block == 0 ||
        config->threads_per_block > 1024) {
        return invalid_spec(
            path + ".config.threads_per_block must be in [1, 1024]");
    }
    return Status::Ok();
}
} // namespace tutti::config::detail

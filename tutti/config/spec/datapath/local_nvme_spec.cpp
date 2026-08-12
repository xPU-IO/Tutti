#include "tutti/config/spec/spec_internal.h"

namespace tutti::config::detail {
Status validate_local_nvme_datapath(const DataPathSpec& spec,
                                    const std::string& path) {
    return std::holds_alternative<LocalNvmeDataPathConfig>(spec.config)
        ? Status::Ok()
        : invalid_spec(path + ".config does not match type local-nvme");
}
} // namespace tutti::config::detail

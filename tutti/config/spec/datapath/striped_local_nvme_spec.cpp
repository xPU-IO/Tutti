#include "tutti/config/spec/spec_internal.h"

namespace tutti::config::detail {
Status validate_striped_local_nvme_datapath(const DataPathSpec& spec,
                                            const std::string& path) {
    return std::holds_alternative<StripedLocalNvmeDataPathConfig>(spec.config)
        ? Status::Ok()
        : invalid_spec(path +
                       ".config does not match type striped-local-nvme");
}
} // namespace tutti::config::detail

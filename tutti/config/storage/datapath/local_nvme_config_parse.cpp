#include "tutti/config/storage/datapath/nvme_parse_internal.h"

namespace tutti::config::detail {

Status parse_local_nvme_datapath_config(const YAML::Node& config,
                                        const std::string& context,
                                        DataPathSpec& datapath) {
    return parse_nvme_datapath_tuning(config, context, datapath);
}

} // namespace tutti::config::detail

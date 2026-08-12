#include "tutti/config/parser/datapath/nvme_parser_internal.h"

namespace tutti::config::detail {
Status parse_local_nvme_datapath(const YAML::Node& config,
                                 const std::string& path,
                                 DataPathSpec& datapath) {
    LocalNvmeDataPathConfig parsed;
    Status status = parse_nvme_datapath_tuning(config, path, parsed);
    if (!status.ok()) return status;
    datapath.config = parsed;
    return Status::Ok();
}
} // namespace tutti::config::detail

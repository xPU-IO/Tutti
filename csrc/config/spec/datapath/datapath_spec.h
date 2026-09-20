#pragma once

#include <string>
#include <variant>

#include <tutti/config/spec/datapath/local_nvme_spec.h>
#include <tutti/config/spec/datapath/memfs_spec.h>
#include <tutti/config/spec/datapath/striped_local_nvme_spec.h>

namespace tutti::config {

using DataPathConfig = std::variant<LocalNvmeDataPathConfig,
                                    StripedLocalNvmeDataPathConfig,
                                    MemfsDataPathConfig>;

struct DataPathSpec {
    std::string id;
    std::string type;
    DataPathConfig config;
};

} // namespace tutti::config

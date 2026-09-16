#pragma once

#include <string>
#include <variant>

#include <tutti/config/spec/backend/ext4_local_nvme_spec.h>
#include <tutti/config/spec/backend/memfs_spec.h>
#include <tutti/config/spec/backend/striped_local_nvme_spec.h>

namespace tutti::config {

using BackendConfig = std::variant<Ext4LocalNvmeBackendConfig,
                                   StripedLocalNvmeBackendConfig,
                                   MemfsBackendConfig>;

struct BackendSpec {
    std::string id;
    std::string contract;
    std::string resolver;
    std::string datapath;
    std::string resource;
    BackendConfig config;
};

} // namespace tutti::config

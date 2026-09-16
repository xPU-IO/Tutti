#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include <tutti/config/tutti_runtime_spec.h>

namespace tutti::config::detail {

struct SpecContract {
    std::string_view name;
    std::string_view resolver_type;
    std::string_view resolver_scheme;
    std::string_view datapath_type;
    std::string_view resource_type;
    std::size_t minimum_cardinality = 0;
    std::size_t maximum_cardinality = 0;
};

Status invalid_spec(std::string message);
const SpecContract* find_spec_contract(std::string_view name);

Status validate_nvme_resource(const ResourceSpec& spec,
                              const std::string& path);
Status validate_memory_resource(const ResourceSpec& spec,
                                const std::string& path);
Status validate_local_file_resolver(const ResolverSpec& spec,
                                    const std::string& path);
Status validate_striped_file_resolver(const ResolverSpec& spec,
                                      const std::string& path);
Status validate_memfs_resolver(const ResolverSpec& spec,
                               const std::string& path);
Status validate_local_nvme_datapath(const DataPathSpec& spec,
                                    const std::string& path);
Status validate_striped_local_nvme_datapath(const DataPathSpec& spec,
                                            const std::string& path);
Status validate_memfs_datapath(const DataPathSpec& spec,
                               const std::string& path);
Status validate_ext4_local_nvme_backend(const ResourceSpec& resource,
                                        const BackendSpec& backend,
                                        const SpecContract& contract,
                                        const std::string& path);
Status validate_striped_local_nvme_backend(const ResourceSpec& resource,
                                           const BackendSpec& backend,
                                           const SpecContract& contract,
                                           const std::string& path);
Status validate_memfs_backend(const ResourceSpec& resource,
                              const BackendSpec& backend,
                              const SpecContract& contract,
                              const std::string& path);

} // namespace tutti::config::detail

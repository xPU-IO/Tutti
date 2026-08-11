#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>

#include <yaml-cpp/yaml.h>

#include "tutti/config/tutti_config.h"

namespace tutti::config::detail {

Status invalid(std::string message);

Status validate_keys(
    const YAML::Node& node, const std::string& context,
    std::initializer_list<const char*> allowed,
    std::initializer_list<const char*> required = {});

Status read_required_string(const YAML::Node& node, const char* key,
                            const std::string& context, std::string& value);

Status read_nonnegative_i32(const YAML::Node& node, const char* key,
                            const std::string& context, std::int32_t& value,
                            bool required);

Status read_nonnegative_u32(const YAML::Node& node, const char* key,
                            const std::string& context, std::uint32_t& value);

Status read_nonnegative_u64(const YAML::Node& node, const char* key,
                            const std::string& context, std::uint64_t& value,
                            bool required = false);

Status parse_nvme_resource(const YAML::Node& node,
                           const std::string& context,
                           ResourceSpec& resource);

Status parse_memory_resource(const YAML::Node& node,
                             const std::string& context,
                             ResourceSpec& resource);

Status parse_local_file_resolver_config(const YAML::Node& config,
                                        const std::string& context,
                                        ResolverSpec& resolver);

Status parse_striped_file_resolver_config(const YAML::Node& config,
                                          const std::string& context,
                                          ResolverSpec& resolver);

Status parse_memfs_resolver_config(const YAML::Node& config,
                                   const std::string& context,
                                   ResolverSpec& resolver);

Status parse_local_nvme_datapath_config(const YAML::Node& config,
                                        const std::string& context,
                                        DataPathSpec& datapath);

Status parse_striped_local_nvme_datapath_config(
    const YAML::Node& config, const std::string& context,
    DataPathSpec& datapath);

Status parse_memfs_datapath_config(const YAML::Node& config,
                                   const std::string& context,
                                   DataPathSpec& datapath);

Status parse_ext4_local_nvme_backend_config(const YAML::Node& config,
                                            const std::string& context,
                                            BackendSpec& backend);

Status validate_ext4_local_nvme_backend(const ResourceSpec& resource,
                                        const BackendSpec& backend,
                                        const StorageContract& contract);

Status parse_striped_local_nvme_backend_config(const YAML::Node& config,
                                               const std::string& context,
                                               BackendSpec& backend);

Status validate_striped_local_nvme_backend(const ResourceSpec& resource,
                                           const BackendSpec& backend,
                                           const StorageContract& contract);

Status parse_memfs_backend_config(const YAML::Node& config,
                                  const std::string& context,
                                  BackendSpec& backend);

Status validate_memfs_backend(const ResourceSpec& resource,
                              const BackendSpec& backend,
                              const StorageContract& contract);

Status parse_legacy_nvme(const YAML::Node& root, ParsedConfig& config);

CanonicalStorageConfig adapt_legacy_nvme(const ParsedConfig& config);

void apply_nvme_compatibility(const ResourceSpec& resource,
                              const DataPathSpec& datapath,
                              const BackendSpec& backend,
                              ParsedConfig& config);

} // namespace tutti::config::detail

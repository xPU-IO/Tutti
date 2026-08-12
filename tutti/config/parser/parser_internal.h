#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>

#include <yaml-cpp/yaml.h>

#include <tutti/config/tutti_runtime_spec.h>

namespace tutti::config::detail {

Status parse_error(std::string message);
Status validate_keys(const YAML::Node& node, const std::string& path,
                     std::initializer_list<const char*> allowed,
                     std::initializer_list<const char*> required = {});
Status read_required_string(const YAML::Node& node, const char* key,
                            const std::string& path, std::string& value);
Status read_optional_u32(const YAML::Node& node, const char* key,
                         const std::string& path, std::uint32_t& value);
Status read_optional_u64(const YAML::Node& node, const char* key,
                         const std::string& path, std::uint64_t& value);
Status read_required_u64(const YAML::Node& node, const char* key,
                         const std::string& path, std::uint64_t& value);

Status parse_nvme_resource(const YAML::Node& node, const std::string& path,
                           ResourceSpec& resource);
Status parse_memory_resource(const YAML::Node& node, const std::string& path,
                             ResourceSpec& resource);
Status parse_local_file_resolver(const YAML::Node& config,
                                 const std::string& path,
                                 ResolverSpec& resolver);
Status parse_striped_file_resolver(const YAML::Node& config,
                                   const std::string& path,
                                   ResolverSpec& resolver);
Status parse_memfs_resolver(const YAML::Node& config, const std::string& path,
                            ResolverSpec& resolver);
Status parse_local_nvme_datapath(const YAML::Node& config,
                                 const std::string& path,
                                 DataPathSpec& datapath);
Status parse_striped_local_nvme_datapath(const YAML::Node& config,
                                         const std::string& path,
                                         DataPathSpec& datapath);
Status parse_memfs_datapath(const YAML::Node& config,
                            const std::string& path,
                            DataPathSpec& datapath);
Status parse_ext4_local_nvme_backend(const YAML::Node& config,
                                     const std::string& path,
                                     BackendSpec& backend);
Status parse_striped_local_nvme_backend(const YAML::Node& config,
                                        const std::string& path,
                                        BackendSpec& backend);
Status parse_memfs_backend(const YAML::Node& config, const std::string& path,
                           BackendSpec& backend);

} // namespace tutti::config::detail

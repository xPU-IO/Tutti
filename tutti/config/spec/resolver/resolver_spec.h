#pragma once

#include <string>
#include <variant>

#include <tutti/config/spec/resolver/local_file_spec.h>
#include <tutti/config/spec/resolver/memfs_spec.h>
#include <tutti/config/spec/resolver/striped_file_spec.h>

namespace tutti::config {

using ResolverConfig = std::variant<LocalFileResolverConfig,
                                    StripedFileResolverConfig,
                                    MemfsResolverConfig>;

struct ResolverSpec {
    std::string id;
    std::string type;
    std::string scheme;
    ResolverConfig config;
};

} // namespace tutti::config

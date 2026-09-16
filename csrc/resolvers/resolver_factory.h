#pragma once

#include <memory>
#include <string>

#include <tutti/config/spec/backend/backend_spec.h>
#include <tutti/config/spec/resolver/resolver_spec.h>
#include <tutti/resource.h>
#include <tutti/spi/storage_target_resolver.h>

namespace tutti::resolvers {

struct ResolverCreateContext {
    const Resource& resource;
    const config::BackendSpec& relation;
    std::string data_path_key;
};

Result<std::unique_ptr<StorageTargetResolver>> create_resolver(
    const config::ResolverSpec& spec,
    const ResolverCreateContext& context);

} // namespace tutti::resolvers

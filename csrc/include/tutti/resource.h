#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <tutti/config/spec/resource/resource_spec.h>
#include <tutti/status.h>

namespace tutti {

enum class ResourceState {
    CREATED,
    INITIALIZED,
    SHUTTING_DOWN,
    STOPPED,
    FAILED,
};

struct ResourceCapabilities {
    std::string resource_type;
    bool provides_resolver_view = false;
    bool provides_datapath_view = false;
};

struct ResourceInfo {
    std::string id;
    std::string type;
    ResourceState state = ResourceState::CREATED;
};

class ResourceView {
public:
    virtual ~ResourceView() = default;
};

class Resource {
public:
    virtual ~Resource() = default;

    virtual const ResourceCapabilities& capabilities() const = 0;
    virtual Status initialize() = 0;
    virtual Status shutdown() = 0;
    virtual ResourceInfo info() const = 0;
    virtual Result<std::unique_ptr<const ResourceView>>
    get_resolver_view() const = 0;
    virtual Result<std::unique_ptr<const ResourceView>>
    get_datapath_view() const = 0;
};

namespace resources {

struct ResourceCreateContext {
    std::int32_t runtime_accel_id = -1;
};

Result<std::unique_ptr<Resource>> create_resource(
    const config::ResourceSpec& spec,
    const ResourceCreateContext& context);

} // namespace resources

} // namespace tutti

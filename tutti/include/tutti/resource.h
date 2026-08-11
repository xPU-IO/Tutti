#pragma once

#include <string>

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

class Resource {
public:
    virtual ~Resource() = default;

    virtual const ResourceCapabilities& capabilities() const = 0;
    virtual Status initialize() = 0;
    virtual Status shutdown() = 0;
    virtual ResourceInfo info() const = 0;
};

} // namespace tutti

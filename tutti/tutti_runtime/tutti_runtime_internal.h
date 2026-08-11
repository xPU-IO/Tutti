#pragma once

#include <tutti/tutti_runtime.h>

namespace tutti::config {

class TuttiRuntimeTestingAccess {
public:
    static const Resource* resource(const TuttiRuntime& runtime) noexcept {
        return runtime.resource_.get();
    }
};

} // namespace tutti::config

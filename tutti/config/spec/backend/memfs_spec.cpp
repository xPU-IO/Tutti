#include "tutti/config/spec/spec_internal.h"

namespace tutti::config::detail {

Status validate_memfs_backend(const ResourceSpec&,
                              const BackendSpec& backend,
                              const SpecContract&,
                              const std::string& path) {
    return std::holds_alternative<MemfsBackendConfig>(backend.config)
        ? Status::Ok()
        : invalid_spec(path + ".config does not match contract memfs");
}

} // namespace tutti::config::detail

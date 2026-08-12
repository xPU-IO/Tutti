#include "tutti/config/spec/spec_internal.h"

namespace tutti::config::detail {

Status validate_memory_resource(const ResourceSpec& spec,
                                const std::string& path) {
    const auto* config = std::get_if<MemoryResourceConfig>(&spec.config);
    if (config == nullptr) {
        return invalid_spec(path + ".config does not match type memory");
    }
    if (config->capacity_bytes == 0) {
        return invalid_spec(path +
                            ".config.capacity_bytes must be greater than zero");
    }
    return Status::Ok();
}

} // namespace tutti::config::detail

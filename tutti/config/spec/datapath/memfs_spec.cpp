#include "tutti/config/spec/spec_internal.h"

namespace tutti::config::detail {
Status validate_memfs_datapath(const DataPathSpec& spec,
                               const std::string& path) {
    return std::holds_alternative<MemfsDataPathConfig>(spec.config)
        ? Status::Ok()
        : invalid_spec(path + ".config does not match type memfs");
}
} // namespace tutti::config::detail

#include "tutti/config/spec/spec_internal.h"

namespace tutti::config::detail {
Status validate_memfs_resolver(const ResolverSpec& spec,
                               const std::string& path) {
    return std::holds_alternative<MemfsResolverConfig>(spec.config)
        ? Status::Ok()
        : invalid_spec(path + ".config does not match type memfs");
}
} // namespace tutti::config::detail

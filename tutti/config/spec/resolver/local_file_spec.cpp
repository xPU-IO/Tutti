#include "tutti/config/spec/spec_internal.h"

namespace tutti::config::detail {
Status validate_local_file_resolver(const ResolverSpec& spec,
                                    const std::string& path) {
    return std::holds_alternative<LocalFileResolverConfig>(spec.config)
        ? Status::Ok()
        : invalid_spec(path + ".config does not match type local-file");
}
} // namespace tutti::config::detail

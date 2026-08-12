#include "tutti/config/spec/spec_internal.h"

namespace tutti::config::detail {
Status validate_striped_file_resolver(const ResolverSpec& spec,
                                      const std::string& path) {
    return std::holds_alternative<StripedFileResolverConfig>(spec.config)
        ? Status::Ok()
        : invalid_spec(path + ".config does not match type striped-file");
}
} // namespace tutti::config::detail

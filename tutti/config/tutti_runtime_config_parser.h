#pragma once

#include <string>

#include <tutti/config/tutti_runtime_spec.h>

namespace tutti::config {

Result<TuttiRuntimeSpec> parse_tutti_runtime_config(const std::string& path);

} // namespace tutti::config

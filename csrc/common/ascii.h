#pragma once

// 共享 ASCII 大写化工具。
//
// 配置校验（config/spec/tutti_runtime_spec.cpp）与运行时创建
// （tutti_runtime/tutti_runtime_create.cpp）此前各有一份逐字相同的
// 实现；统一到这里，避免两份实现各自漂移。
//
// 只做字节级 ASCII 大写（std::toupper 配 unsigned char 转换），
// 不做 locale 折叠或 Unicode 处理。

#include <cctype>
#include <string>

namespace tutti::detail {

inline std::string upper_ascii(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(
            std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

} // namespace tutti::detail

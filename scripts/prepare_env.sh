#!/bin/bash
# Geminifs 一键环境准备脚本
# 目标：在多种 Linux 发行版（Debian/Ubuntu/RHEL/CentOS/TencentOS/Fedora/openSUSE/Arch）
#       上自动安装 protobuf/gRPC/uuid 等依赖，并按 CUDA 版本下载匹配的 libtorch。
# 兼容：x86_64 / aarch64（libtorch 仅在 x86_64 提供 GPU 包，aarch64 仅支持 CPU 包）。

set -eu
# 注意：不开启 pipefail，因为 install_pkg 中允许部分非致命失败被吞掉。

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- 运行环境探测 ----------------
ARCH="$(uname -m)"

# 计算 sudo 前缀：root 用户无需 sudo；非 root 但缺少 sudo 时给出明确错误。
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    echo "错误：当前为非 root 用户且未安装 sudo，无法继续安装系统依赖。" >&2
    echo "请使用 root 运行，或安装 sudo 后再试。" >&2
    exit 1
fi

# 解析命令行参数
FORCE_REINSTALL=0
# 默认使用全部 CPU 核数；nproc 不可用时回退到 4
if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
else
    JOBS=4
fi

show_help() {
    cat <<EOF
用法: $0 [选项]
  -f, --force         强制重新下载/解压 libtorch（即使已存在匹配版本）
  -j N, --jobs N      并行编译线程数（用于 vcpkg 等后续编译，默认 ${JOBS} = nproc）
  -j=N, --jobs=N      同上
  -h, --help          显示帮助
环境变量:
  CUDA_VER            手动指定 CUDA 版本，如 CUDA_VER=13.0
  LIBTORCH_URL        未在表中的 CUDA 版本时，手动指定 libtorch 下载链接
  VCPKG_ROOT          指定 vcpkg 安装路径（默认 \${PROJECT_ROOT}/third_pkgs/vcpkg）
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -f|--force)
            FORCE_REINSTALL=1
            shift
            ;;
        -j|--jobs)
            if [ -z "${2:-}" ] || [[ "$2" == -* ]]; then
                echo "错误：$1 需要一个整数参数。" >&2
                exit 1
            fi
            JOBS="$2"
            shift 2
            ;;
        -j=*|--jobs=*)
            JOBS="${1#*=}"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

# 校验 JOBS 是正整数
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：--jobs 参数无效：$JOBS（必须为正整数）" >&2
    exit 1
fi

# 让后续所有 make / vcpkg 子流程默认并发
export MAKEFLAGS="-j${JOBS}"
export VCPKG_MAX_CONCURRENCY="${JOBS}"
echo "并行编译线程数 (JOBS): ${JOBS}"

# 检测包管理器并识别发行版家族
detect_pkg_manager() {
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        DISTRO_ID="${ID:-unknown}"
        DISTRO_ID_LIKE="${ID_LIKE:-}"
    else
        DISTRO_ID="unknown"
        DISTRO_ID_LIKE=""
    fi

    if command -v apt-get >/dev/null 2>&1; then
        PKG_MANAGER="apt"
    elif command -v dnf >/dev/null 2>&1; then
        PKG_MANAGER="dnf"
    elif command -v yum >/dev/null 2>&1; then
        PKG_MANAGER="yum"
    elif command -v zypper >/dev/null 2>&1; then
        PKG_MANAGER="zypper"
    elif command -v pacman >/dev/null 2>&1; then
        PKG_MANAGER="pacman"
    else
        PKG_MANAGER="unknown"
    fi
    echo "检测到发行版: ${DISTRO_ID} (ID_LIKE=${DISTRO_ID_LIKE}), 包管理器: ${PKG_MANAGER}"
}

# 检查指定包是否已安装
pkg_is_installed() {
    local pkg="$1"
    case "$PKG_MANAGER" in
        apt)
            dpkg -s "$pkg" >/dev/null 2>&1
            ;;
        dnf|yum)
            rpm -q "$pkg" >/dev/null 2>&1
            ;;
        zypper)
            rpm -q "$pkg" >/dev/null 2>&1
            ;;
        pacman)
            pacman -Qi "$pkg" >/dev/null 2>&1
            ;;
        *)
            return 1
            ;;
    esac
}

# 安装包（自动按发行版选择名称）
# 第 4 个可选参数: "optional" 表示安装失败仅警告，不中断脚本
install_pkg() {
    local apt_name="$1"
    local rpm_name="$2"
    local pacman_name="$3"
    local optional="${4:-}"

    local target_name=""
    local install_cmd=""
    case "$PKG_MANAGER" in
        apt)
            target_name="$apt_name"
            install_cmd="$SUDO apt-get install -y $apt_name"
            ;;
        dnf)
            target_name="$rpm_name"
            install_cmd="$SUDO dnf install -y $rpm_name"
            ;;
        yum)
            target_name="$rpm_name"
            install_cmd="$SUDO yum install -y $rpm_name"
            ;;
        zypper)
            target_name="$rpm_name"
            install_cmd="$SUDO zypper install -y $rpm_name"
            ;;
        pacman)
            target_name="$pacman_name"
            install_cmd="$SUDO pacman -Sy --noconfirm $pacman_name"
            ;;
        *)
            echo "未识别的包管理器，请手动安装：$apt_name / $rpm_name"
            return 1
            ;;
    esac

    if pkg_is_installed "$target_name"; then
        echo "$target_name 已安装。"
        return 0
    fi

    echo "正在安装 $target_name ..."
    if [ "$PKG_MANAGER" = "apt" ]; then
        $SUDO apt-get update >/dev/null
    fi
    if eval "$install_cmd"; then
        return 0
    fi

    if [ "$optional" = "optional" ]; then
        echo "警告：$target_name 安装失败（该包在当前发行版可能不可用），将跳过；请关注后续依赖检查。"
        return 0
    fi
    return 1
}

detect_pkg_manager

echo "------------------------------------------------------------"
echo "  Geminifs 环境准备脚本"
echo "  项目根目录 : ${PROJECT_ROOT}"
echo "  架构       : ${ARCH}"
echo "  特权方式   : ${SUDO:-(root 直接执行)}"
echo "  并行数     : ${JOBS}"
echo "  发行版     : ${DISTRO_ID} (ID_LIKE=${DISTRO_ID_LIKE})"
echo "  包管理器   : ${PKG_MANAGER}"
echo "------------------------------------------------------------"

# 安装 uuid 开发库
#   Debian/Ubuntu : uuid-dev
#   RHEL/CentOS/TencentOS/Fedora/SUSE : libuuid-devel
#   Arch          : util-linux (自带 libuuid)
install_pkg "uuid-dev" "libuuid-devel" "util-linux"

# 安装 wget / unzip（后续步骤需要）
install_pkg "wget"  "wget"  "wget"
install_pkg "unzip" "unzip" "unzip"

# 安装 Protobuf 开发库 + protoc 编译器（CMakeLists.txt: find_package(Protobuf REQUIRED)）
#   Debian/Ubuntu : libprotobuf-dev + protobuf-compiler
#   RHEL/CentOS/TencentOS/Fedora/SUSE : protobuf-devel + protobuf-compiler
#   Arch          : protobuf
install_pkg "libprotobuf-dev"   "protobuf-devel"    "protobuf"
install_pkg "protobuf-compiler" "protobuf-compiler" "protobuf"

# 安装 gRPC 开发库 + grpc_cpp_plugin（CMakeLists.txt: find_package(gRPC REQUIRED)）
#   Debian/Ubuntu : libgrpc++-dev + protobuf-compiler-grpc
#   RHEL/CentOS/TencentOS/Fedora     : grpc-devel + grpc-plugins (EL8 系仓库均无)
#   SUSE          : grpc-devel
#   Arch          : grpc
# 系统包不可用时（典型场景：TencentOS 3 / EL8），fallback 到 vcpkg 编译安装。
install_pkg "libgrpc++-dev"          "grpc-devel"   "grpc" optional
install_pkg "protobuf-compiler-grpc" "grpc-plugins" "grpc" optional

# ----- 通过 vcpkg 提供 C++ 依赖（grpc / yaml-cpp 等） -----
# 触发逻辑：若系统未提供 gRPC（通过 pkg-config 与 grpc_cpp_plugin 检测），
# 则启动 vcpkg 来安装 grpc + yaml-cpp 等需要的 C++ 库。
# 之所以把 yaml-cpp 一并放进 vcpkg：
#   - RHEL/TencentOS 系统包仅 yaml-cpp 0.5.x，其 CMake config 缺少
#     RelWithDebInfo 配置的 IMPORTED_LOCATION，触发 CMP0111 警告刷屏。
#   - 由 vcpkg 统一安装可保证版本一致、CMake 集成干净。

# 判断系统是否提供了「CMake 可用」的 gRPC。
# 关键：CMakeLists.txt 使用 find_package(gRPC CONFIG REQUIRED)，它要求存在
#       gRPCConfig.cmake（CMake 包配置文件）。而某些发行版（如 Ubuntu 的
#       libgrpc++-dev）只随包提供 pkg-config（.pc）文件，并不提供 gRPCConfig.cmake。
# 因此「pkg-config 能找到 grpc++ + 存在 grpc_cpp_plugin」并不代表 CMake 能用 gRPC，
# 旧逻辑据此跳过 vcpkg，会导致 CMake 阶段报 "Could not find a package configuration
# file provided by gRPC"。这里改用 cmake 真实探测（最准确），失败时退化为在常见前缀
# 下查找 gRPCConfig.cmake。
system_has_cmake_grpc() {
    # 优先用 cmake 真实探测：覆盖 CMake 的全部默认搜索路径与 CMAKE_PREFIX_PATH。
    # 使用 LANGUAGES NONE 避免对编译器/构建工具的依赖。
    if command -v cmake >/dev/null 2>&1; then
        local probe_dir
        probe_dir="$(mktemp -d)"
        cat > "${probe_dir}/CMakeLists.txt" <<'CMEOF'
cmake_minimum_required(VERSION 3.16)
project(grpc_probe LANGUAGES NONE)
find_package(gRPC CONFIG REQUIRED)
CMEOF
        if cmake -S "${probe_dir}" -B "${probe_dir}/build" >/dev/null 2>&1; then
            rm -rf "${probe_dir}"
            return 0
        fi
        rm -rf "${probe_dir}"
        return 1
    fi

    # 退化检测（cmake 尚未安装时）：在常见安装前缀下查找 gRPCConfig.cmake。
    local d
    for d in /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /usr/local /usr; do
        [ -d "$d" ] || continue
        if find "$d" -maxdepth 4 \
            \( -name 'gRPCConfig.cmake' -o -name 'grpc-config.cmake' \) \
            2>/dev/null | grep -q .; then
            return 0
        fi
    done
    return 1
}

ensure_cpp_deps_via_vcpkg() {
    # 同时满足三者才认为系统 gRPC 可直接用于 CMake：
    #   1) pkg-config 能找到 grpc++（运行库/头文件）
    #   2) grpc_cpp_plugin 可用（代码生成）
    #   3) 存在 gRPCConfig.cmake（find_package(gRPC CONFIG) 需要）
    if command -v pkg-config >/dev/null 2>&1 \
        && pkg-config --exists grpc++ 2>/dev/null \
        && command -v grpc_cpp_plugin >/dev/null 2>&1 \
        && system_has_cmake_grpc; then
        echo "系统已提供 gRPC（grpc++ + grpc_cpp_plugin + gRPCConfig.cmake），跳过 vcpkg 安装。"
        return 0
    fi

    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists grpc++ 2>/dev/null; then
        echo "检测到系统 grpc++（pkg-config 可见），但缺少 CMake 包配置 gRPCConfig.cmake，"
        echo "find_package(gRPC CONFIG REQUIRED) 将无法找到它，回退到 vcpkg 安装方案。"
    else
        echo "系统未提供 gRPC，回退到 vcpkg 安装方案。"
    fi

    # 代理环境健康提示（vcpkg 经常因公司代理把 GitHub release 拦掉）
    if [ -n "${HTTPS_PROXY:-}" ] || [ -n "${https_proxy:-}" ]; then
        echo "提示：检测到 HTTPS 代理：${HTTPS_PROXY:-${https_proxy:-}}"
        echo "      如果 vcpkg 编译期出现 squid/HTML 错误页，多半是代理拦截了某个下载源；"
        echo "      可临时取消代理重试: unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy"
    fi

    # vcpkg 编译 grpc 需要的构建依赖
    # 说明：build-essential / gcc-c++ / base-devel 元包通常已包含 make 与 tar，
    #       因此不再单独安装 make / tar 以避免冗余。
    install_pkg "build-essential" "gcc-c++"     "base-devel"
    install_pkg "cmake"           "cmake"       "cmake"
    install_pkg "git"             "git"         "git"
    install_pkg "pkg-config"      "pkgconf-pkg-config" "pkgconf" optional
    install_pkg "curl"            "curl"        "curl"
    install_pkg "zip"             "zip"         "zip"
    # autotools / perl 是 grpc 的若干传递依赖（abseil/c-ares/openssl 等）需要
    install_pkg "autoconf"        "autoconf"    "autoconf" optional
    install_pkg "automake"        "automake"    "automake" optional
    install_pkg "libtool"         "libtool"     "libtool"  optional
    install_pkg "perl"            "perl"        "perl"     optional

    local vcpkg_root="${VCPKG_ROOT:-${PROJECT_ROOT}/third_pkgs/vcpkg}"
    if [ ! -d "$vcpkg_root/.git" ]; then
        echo "克隆 vcpkg 到: $vcpkg_root"
        git clone --depth 1 https://github.com/microsoft/vcpkg.git "$vcpkg_root" || {
            echo "vcpkg 克隆失败，请检查网络。"
            return 1
        }
    else
        echo "vcpkg 已存在: $vcpkg_root"
    fi

    if [ ! -x "$vcpkg_root/vcpkg" ]; then
        echo "bootstrap vcpkg ..."
        bash "$vcpkg_root/bootstrap-vcpkg.sh" -disableMetrics || {
            echo "vcpkg bootstrap 失败。"
            return 1
        }
    fi

    # 通过 vcpkg 安装一个包；已安装则跳过
    # 用法：vcpkg_install_pkg <port-name>
    vcpkg_install_pkg() {
        local port="$1"
        if "$vcpkg_root/vcpkg" list 2>/dev/null | grep -q "^${port}:"; then
            echo "vcpkg 已安装 ${port}。"
            return 0
        fi
        echo "通过 vcpkg 安装 ${port}（使用 ${JOBS} 个并发任务）..."
        "$vcpkg_root/vcpkg" install "${port}" || {
            echo "vcpkg 安装 ${port} 失败。"
            return 1
        }
    }

    # 需要通过 vcpkg 提供的 C++ 库
    #   grpc     : NVMeService（系统包通常不可用）
    #   yaml-cpp : 配置文件解析（系统包在 RHEL/TencentOS 上仅 0.5.x，
    #              缺少 RelWithDebInfo 的 IMPORTED_LOCATION，会产生大量 CMP0111 警告）
    local vcpkg_ports=("grpc" "yaml-cpp")
    for port in "${vcpkg_ports[@]}"; do
        vcpkg_install_pkg "$port" || return 1
    done

    # 二次校验：关键产物存在，避免代理/网络异常导致的"伪成功"
    local grpc_cmake_dir="${vcpkg_root}/installed/x64-linux/share/grpc"
    local grpc_plugin_bin="${vcpkg_root}/installed/x64-linux/tools/grpc/grpc_cpp_plugin"
    if [ ! -f "${grpc_cmake_dir}/gRPCConfig.cmake" ]; then
        # 兼容其它 triplet（如 arm64-linux）
        if ! find "${vcpkg_root}/installed" -maxdepth 4 -name 'gRPCConfig.cmake' 2>/dev/null | grep -q .; then
            echo "错误：vcpkg 报告安装成功，但找不到 gRPCConfig.cmake，疑似下载/编译异常。"
            echo "可执行 '${vcpkg_root}/vcpkg remove grpc --recurse' 后重试。"
            return 1
        fi
    fi
    if [ ! -x "$grpc_plugin_bin" ]; then
        if ! find "${vcpkg_root}/installed" -maxdepth 5 -name 'grpc_cpp_plugin' -type f -executable 2>/dev/null | grep -q .; then
            echo "错误：找不到 grpc_cpp_plugin 可执行文件。"
            return 1
        fi
    fi
    if ! find "${vcpkg_root}/installed" -maxdepth 4 -name 'yaml-cpp-config.cmake' 2>/dev/null | grep -q .; then
        echo "警告：未找到 yaml-cpp-config.cmake，可能 vcpkg 中的 yaml-cpp 安装异常。"
    fi
    echo "vcpkg 中 gRPC / yaml-cpp 安装校验通过。"

    VCPKG_TOOLCHAIN_FILE="${vcpkg_root}/scripts/buildsystems/vcpkg.cmake"
    cat <<EOF

==============================================================
gRPC 已通过 vcpkg 安装到: ${vcpkg_root}

推荐使用 CMake Presets（一行搞定）：

    cmake --preset default
    cmake --build --preset default

可用 preset：default / debug / release / system
查看完整列表：cmake --list-presets

如果你的 CMake < 3.21 不支持 presets，可手动指定 toolchain：

    cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=${VCPKG_TOOLCHAIN_FILE}
==============================================================
EOF
    return 0
}

# ----- 生成 CMakePresets.json -----
# 该文件是「按机器生成」的产物（已在 .gitignore 中忽略，不提交）：
#   - 直接内嵌本机 vcpkg toolchain 的绝对路径，避免依赖宏推导；
#   - 根据上一步实际采用的依赖来源（vcpkg / 系统包）裁剪 default 预设：
#       * 走了 vcpkg  -> default/debug/release 加载 vcpkg toolchain
#       * 用系统包    -> default/debug/release 不加 toolchain（直接用系统 gRPC）
#   - 始终额外提供 system 预设（不加 toolchain），便于在同机两种来源间切换。
# 个人化覆盖（自定义 VCPKG_ROOT、显式 SNVME_KERNEL_VERSION、其它 CUDA 路径等）
# 请写入 CMakeUserPresets.json（同样被忽略），它会自动继承本文件。
# libtorch 路径无需在此设置：CMakeLists.txt 会自动把 third_pkgs/libtorch 加入
# CMAKE_PREFIX_PATH。SNVME_KERNEL_VERSION 留空让 CMake 按 `uname -r` 自动选择。
generate_cmake_presets() {
    local presets_file="${PROJECT_ROOT}/CMakePresets.json"
    local toolchain="${VCPKG_TOOLCHAIN_FILE:-}"

    # 仅当本次确实通过 vcpkg 安装（toolchain 文件存在）时，default 才注入 toolchainFile。
    local tc_line=""
    if [ -n "$toolchain" ] && [ -f "$toolchain" ]; then
        tc_line="
      \"toolchainFile\": \"${toolchain}\","
        echo "生成 CMakePresets.json：default 预设使用 vcpkg toolchain：${toolchain}"
    else
        echo "生成 CMakePresets.json：未使用 vcpkg，default 预设依赖系统已安装的 gRPC/protobuf/yaml-cpp。"
    fi

    cat > "${presets_file}" <<EOF
{
  "version": 3,
  "cmakeMinimumRequired": { "major": 3, "minor": 21, "patch": 0 },
  "vendor": {
    "tutti.prepare_env": {
      "generated": true,
      "note": "Generated by scripts/prepare_env.sh; do not edit or commit. Re-run the script to regenerate. Put manual overrides in CMakeUserPresets.json."
    }
  },
  "configurePresets": [
    {
      "name": "base",
      "hidden": true,
      "binaryDir": "\${sourceDir}/build",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "RelWithDebInfo" }
    },
    {
      "name": "default",
      "displayName": "Default (RelWithDebInfo)",
      "description": "Primary preset for this machine, generated by prepare_env.sh.",
      "inherits": "base",${tc_line}
      "cacheVariables": {}
    },
    {
      "name": "debug",
      "displayName": "Debug",
      "inherits": "default",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Debug" }
    },
    {
      "name": "release",
      "displayName": "Release",
      "inherits": "default",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Release" }
    },
    {
      "name": "system",
      "displayName": "System packages (no vcpkg toolchain)",
      "description": "Use the host's CMake-config gRPC/protobuf/yaml-cpp. Requires gRPCConfig.cmake.",
      "inherits": "base",
      "cacheVariables": {}
    }
  ],
  "buildPresets": [
    { "name": "default", "configurePreset": "default", "jobs": ${JOBS} },
    { "name": "debug",   "configurePreset": "debug",   "jobs": ${JOBS} },
    { "name": "release", "configurePreset": "release", "jobs": ${JOBS} },
    { "name": "system",  "configurePreset": "system",  "jobs": ${JOBS} }
  ]
}
EOF
    echo "已写入: ${presets_file}（已在 .gitignore 忽略，按机器生成）"
}

ensure_cpp_deps_via_vcpkg || {
    echo "警告：通过 vcpkg 安装 C++ 依赖（grpc / yaml-cpp）失败，请参考 README 手工处理。"
}

# 依据上一步实际采用的依赖来源生成 CMakePresets.json
generate_cmake_presets

# 校验 protoc / grpc_cpp_plugin 是否真的可用
if ! command -v protoc >/dev/null 2>&1; then
    echo "警告：protoc 未在 PATH 中，CMake 阶段会失败。请确认 protobuf 编译器已正确安装。"
fi
if ! command -v grpc_cpp_plugin >/dev/null 2>&1; then
    if [ -n "${VCPKG_TOOLCHAIN_FILE:-}" ]; then
        echo "提示：grpc_cpp_plugin 通过 vcpkg 安装，CMake 会自动从 toolchain 找到，无需在 PATH 中。"
    else
        echo "警告：grpc_cpp_plugin 未在 PATH 中。若使用系统 grpc，请确认 grpc-plugins / protobuf-compiler-grpc 已安装。"
    fi
fi

# 检测 CUDA 版本（优先级：环境变量 CUDA_VER > nvcc > nvidia-smi）
# 输出形如 "12.8" / "13.0" 的字符串，赋值给 CUDA_VERSION
detect_cuda_version() {
    if [ -n "${CUDA_VER:-}" ]; then
        CUDA_VERSION="${CUDA_VER}"
        echo "使用环境变量指定的 CUDA 版本: ${CUDA_VERSION}"
        return 0
    fi

    local nvcc_bin=""
    if command -v nvcc >/dev/null 2>&1; then
        nvcc_bin="nvcc"
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        nvcc_bin="/usr/local/cuda/bin/nvcc"
    fi

    if [ -n "$nvcc_bin" ]; then
        # 形如: "Cuda compilation tools, release 12.8, V12.8.61"
        CUDA_VERSION="$($nvcc_bin --version 2>/dev/null \
            | grep -oE 'release [0-9]+\.[0-9]+' \
            | awk '{print $2}')"
    fi

    if [ -z "$CUDA_VERSION" ] && command -v nvidia-smi >/dev/null 2>&1; then
        # 形如: "CUDA Version: 13.0"
        CUDA_VERSION="$(nvidia-smi 2>/dev/null \
            | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' \
            | awk '{print $3}' | head -n1)"
    fi

    if [ -z "$CUDA_VERSION" ]; then
        echo "未能检测到 CUDA 版本（nvcc / nvidia-smi 都不可用）。"
        echo "请通过环境变量指定，例如: CUDA_VER=12.8 $0"
        return 1
    fi
    echo "检测到 CUDA 版本: ${CUDA_VERSION}"
}

# 按 CUDA 版本 + 架构选择对应的 libtorch 下载 URL
# 参考：https://pytorch.org/get-started/locally/ 与 https://download.pytorch.org/libtorch/
# 同时设置 LIBTORCH_BUILD_VERSION，形如 "2.12.0+cu130"，用于与本地 build-version 比对
#
# 注意：libtorch 官方仅在 x86_64 上提供 GPU 版（cu*）；
#   - aarch64 用户需自行编译 PyTorch from source 或使用 NVIDIA Jetson 提供的预编译包。
#   - 若需 CPU-only 版，请通过 LIBTORCH_URL 环境变量指定。
resolve_libtorch_url() {
    local major minor tag
    major="${CUDA_VERSION%%.*}"
    minor="${CUDA_VERSION#*.}"
    minor="${minor%%.*}"
    tag="cu${major}${minor}"

    # 架构兜底：非 x86_64 直接劝退使用环境变量
    if [ "$ARCH" != "x86_64" ] && [ -z "${LIBTORCH_URL:-}" ]; then
        echo "当前架构为 ${ARCH}，PyTorch 官方 libtorch 仅在 x86_64 上提供 GPU 预编译包。"
        echo "请通过环境变量 LIBTORCH_URL 指定适用于 ${ARCH} 的 libtorch 下载链接，"
        echo "或参考 https://pytorch.org/get-started/locally/ 自行编译。"
        return 1
    fi

    case "$tag" in
        cu130)
            # CUDA 13.0 - libtorch 2.12.0
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.12.0%2Bcu130.zip"
            LIBTORCH_BUILD_VERSION="2.12.0+cu130"
            ;;
        cu128)
            # CUDA 12.8 - libtorch 2.7.1 (cxx11 ABI)
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip"
            LIBTORCH_BUILD_VERSION="2.7.1+cu128"
            ;;
        cu126)
            # CUDA 12.6 - libtorch 2.6.0 (cxx11 ABI)
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip"
            LIBTORCH_BUILD_VERSION="2.6.0+cu126"
            ;;
        cu124)
            # CUDA 12.4 - libtorch 2.5.1 (cxx11 ABI)
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip"
            LIBTORCH_BUILD_VERSION="2.5.1+cu124"
            ;;
        cu121)
            # CUDA 12.1 - libtorch 2.4.1 (cxx11 ABI)
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu121.zip"
            LIBTORCH_BUILD_VERSION="2.4.1+cu121"
            ;;
        cu118)
            # CUDA 11.8 - libtorch 2.4.1 (cxx11 ABI)
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu118.zip"
            LIBTORCH_BUILD_VERSION="2.4.1+cu118"
            ;;
        *)
            echo "暂未为 CUDA ${CUDA_VERSION} (${tag}) 配置默认的 libtorch 下载链接。"
            echo "可手动设置 LIBTORCH_URL 环境变量后重试，或在脚本中补充对应分支。"
            if [ -n "${LIBTORCH_URL:-}" ]; then
                echo "使用环境变量指定的 LIBTORCH_URL: ${LIBTORCH_URL}"
                LIBTORCH_BUILD_VERSION=""
            else
                return 1
            fi
            ;;
    esac
    echo "对应 libtorch 下载地址: ${LIBTORCH_URL}"
}

detect_cuda_version || exit 1
resolve_libtorch_url || exit 1

# 定义变量
third_pkgs_dir="${PROJECT_ROOT}/third_pkgs"
libtorch_dir="${third_pkgs_dir}/libtorch"
libtorch_version_file="${libtorch_dir}/build-version"
url="${LIBTORCH_URL}"
zip_file_name="${third_pkgs_dir}/libtorch.zip"

# 创建 third_pkgs 目录
if [ ! -d "$third_pkgs_dir" ]; then
    mkdir -p "$third_pkgs_dir"
    echo "已创建目录: $third_pkgs_dir"
else
    echo "目录已存在: $third_pkgs_dir"
fi

# 已安装版本检测：依据 third_pkgs/libtorch/build-version 是否匹配目标版本
need_install_libtorch=1
if [ "$FORCE_REINSTALL" = "1" ]; then
    echo "已指定 --force，将强制重新下载 libtorch。"
elif [ -f "$libtorch_version_file" ]; then
    installed_ver="$(tr -d '[:space:]' < "$libtorch_version_file")"
    if [ -n "$LIBTORCH_BUILD_VERSION" ] && [ "$installed_ver" = "$LIBTORCH_BUILD_VERSION" ]; then
        echo "检测到已安装的 libtorch 版本 ${installed_ver}，与目标 ${LIBTORCH_BUILD_VERSION} 一致，跳过下载。"
        need_install_libtorch=0
    elif [ -z "$LIBTORCH_BUILD_VERSION" ]; then
        echo "已存在 libtorch (build-version=${installed_ver})，但当前未配置目标版本，跳过下载。"
        echo "如需替换，请使用 --force 强制重新下载。"
        need_install_libtorch=0
    else
        echo "检测到已安装的 libtorch 版本 ${installed_ver}，与目标 ${LIBTORCH_BUILD_VERSION} 不一致，将重新下载。"
    fi
elif [ -d "$libtorch_dir" ]; then
    echo "目录 ${libtorch_dir} 已存在但缺少 build-version 文件，将重新下载以确保版本一致。"
fi

if [ "$need_install_libtorch" = "1" ]; then
    # 清理可能存在的旧目录，避免解压冲突
    if [ -d "$libtorch_dir" ]; then
        echo "清理旧的 libtorch 目录: $libtorch_dir"
        rm -rf "$libtorch_dir"
    fi

    # 下载 ZIP 文件（优先 wget，回退到 curl）
    echo "开始下载文件: $url"
    if command -v wget >/dev/null 2>&1; then
        if ! wget -O "$zip_file_name" "$url"; then
            echo "wget 下载失败，请检查网络或URL是否正确。"
            exit 1
        fi
    elif command -v curl >/dev/null 2>&1; then
        if ! curl -fL -o "$zip_file_name" "$url"; then
            echo "curl 下载失败，请检查网络或URL是否正确。"
            exit 1
        fi
    else
        echo "未找到 wget / curl，无法下载 libtorch。请先安装其一后重试。"
        exit 1
    fi
    echo "文件已下载并保存为: $zip_file_name"

    # 解压 ZIP 文件
    if ! command -v unzip >/dev/null 2>&1; then
        echo "未找到 unzip，无法解压 libtorch。请先安装 unzip 后重试。"
        exit 1
    fi
    echo "正在解压文件..."
    if ! unzip -q "$zip_file_name" -d "$third_pkgs_dir"; then
        echo "解压失败，请检查ZIP文件是否完整。"
        exit 1
    fi
    echo "文件已成功解压到: $third_pkgs_dir"

    # 清理 ZIP 文件
    echo "正在清理临时文件..."
    rm -f "$zip_file_name"
fi

echo "操作完成！"
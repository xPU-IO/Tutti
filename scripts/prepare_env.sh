#!/bin/bash
# Tutti 一键环境准备脚本
# 目标：在多种 Linux 发行版（Debian/Ubuntu/RHEL/CentOS/TencentOS/Fedora/openSUSE/Arch）
#       上自动安装编译依赖；完整系统 CMake 依赖不可用时才回退 vcpkg，
#       也可通过 --force-vcpkg 强制使用。仓库共享的 default preset 不由脚本生成。

set -eu
# 注意：不开启 pipefail，因为 install_pkg 中允许部分非致命失败被吞掉。

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- 运行环境探测 ----------------
ARCH="$(uname -m)"

# 解析命令行参数
# 默认使用全部 CPU 核数；nproc 不可用时回退到 4
if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
else
    JOBS=4
fi
FORCE_VCPKG=0

show_help() {
    cat <<EOF
用法: $0 [选项]
  当前系统依赖或仓库已有 vcpkg 已满足时，脚本会直接退出，不执行安装、下载或编译。
  -j N, --jobs N      并行编译线程数（仅在需要 vcpkg 编译时使用，默认 ${JOBS} = nproc）
  -j=N, --jobs=N      同上
  --force-vcpkg       即使完整系统 CMake 依赖可用，也强制使用 vcpkg 的 C++ 依赖
  -h, --help          显示帮助
环境变量:
  VCPKG_ROOT          指定 vcpkg 安装路径（默认 \${PROJECT_ROOT}/third_pkgs/vcpkg）
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
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
        --force-vcpkg)
            FORCE_VCPKG=1
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

cmake_presets_supported() {
    local cmake_version
    local cmake_major
    local cmake_minor

    command -v cmake >/dev/null 2>&1 || return 1
    cmake_version="$(cmake --version | sed -n '1s/^cmake version //p')"
    cmake_major="${cmake_version%%.*}"
    cmake_minor="${cmake_version#*.}"
    cmake_minor="${cmake_minor%%.*}"

    [[ "${cmake_major}" =~ ^[0-9]+$ && "${cmake_minor}" =~ ^[0-9]+$ ]] || return 1
    [ "${cmake_major}" -gt 3 ] \
        || { [ "${cmake_major}" -eq 3 ] && [ "${cmake_minor}" -ge 21 ]; }
}

cmake_package_exists() {
    cmake --find-package \
        -DNAME="$1" \
        -DCOMPILER_ID=GNU \
        -DLANGUAGE=CXX \
        -DMODE=EXIST >/dev/null 2>&1
}

system_cpp_deps_available() {
    command -v protoc >/dev/null 2>&1 \
        && command -v grpc_cpp_plugin >/dev/null 2>&1 \
        && cmake_package_exists yaml-cpp \
        && cmake_package_exists gRPC \
        && cmake_package_exists Protobuf
}

vcpkg_cpp_deps_available() {
    local vcpkg_root="${PROJECT_ROOT}/third_pkgs/vcpkg"
    local vcpkg_prefix="${vcpkg_root}/installed/x64-linux"

    [ -x "${vcpkg_root}/vcpkg" ] \
        && [ -f "${vcpkg_prefix}/share/yaml-cpp/yaml-cpp-config.cmake" ] \
        && [ -f "${vcpkg_prefix}/share/grpc/gRPCConfig.cmake" ] \
        && [ -f "${vcpkg_prefix}/share/protobuf/protobuf-config.cmake" ] \
        && [ -x "${vcpkg_prefix}/tools/protobuf/protoc" ] \
        && [ -x "${vcpkg_prefix}/tools/grpc/grpc_cpp_plugin" ]
}

default_build_dependencies_available() {
    cmake_presets_supported \
        && command -v cc >/dev/null 2>&1 \
        && command -v c++ >/dev/null 2>&1 \
        && command -v make >/dev/null 2>&1 \
        && [ -f "${PROJECT_ROOT}/CMakePresets.json" ] \
        || return 1

    if [ "${FORCE_VCPKG}" -eq 1 ]; then
        if vcpkg_cpp_deps_available; then
            READY_CPP_DEPS_PROVIDER="vcpkg"
            return 0
        fi
        return 1
    fi

    if system_cpp_deps_available; then
        READY_CPP_DEPS_PROVIDER="system"
        return 0
    fi
    if vcpkg_cpp_deps_available; then
        READY_CPP_DEPS_PROVIDER="vcpkg"
        return 0
    fi
    return 1
}

READY_CPP_DEPS_PROVIDER=""
if default_build_dependencies_available; then
    echo "环境已满足（${READY_CPP_DEPS_PROVIDER} C++ 依赖）：跳过系统包安装和 vcpkg。"
    exit 0
fi

# 只有环境不完整时才需要 sudo 或包管理器。
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    echo "错误：当前为非 root 用户且未安装 sudo，无法继续安装系统依赖。" >&2
    echo "请使用 root 运行，或安装 sudo 后再试。" >&2
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
echo "  Tutti 环境准备脚本"
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

# 根构建直接需要的基础工具和库。
install_pkg "build-essential" "gcc-c++"              "base-devel"
install_pkg "cmake"           "cmake"                "cmake"
install_pkg "pkg-config"      "pkgconf-pkg-config"   "pkgconf"
# yaml-cpp 在 RHEL/TencentOS 上可能因 modular filtering 装不上
# （yaml-cpp-devel 被过滤），因此这里设为 optional。脚本随后确认完整系统 CMake
# 依赖；仅在系统路径不完整时才由 vcpkg 补齐。
install_pkg "libyaml-cpp-dev" "yaml-cpp-devel"       "yaml-cpp" optional
install_pkg "libunwind-dev"   "libunwind-devel"      "libunwind"

# CMakePresets.json version 3 requires CMake 3.21+.  Some older supported
# distro releases package an earlier CMake, so fail here with a focused error
# instead of generating a preset file that the local cmake cannot read.
require_cmake_presets_support() {
    local cmake_version
    local cmake_major
    local cmake_minor

    cmake_version="$(cmake --version | sed -n '1s/^cmake version //p')"
    cmake_major="${cmake_version%%.*}"
    cmake_minor="${cmake_version#*.}"
    cmake_minor="${cmake_minor%%.*}"

    if ! [[ "${cmake_major}" =~ ^[0-9]+$ && "${cmake_minor}" =~ ^[0-9]+$ ]]; then
        echo "错误：无法解析 CMake 版本：${cmake_version:-unknown}" >&2
        return 1
    fi
    if [ "${cmake_major}" -lt 3 ] \
        || { [ "${cmake_major}" -eq 3 ] && [ "${cmake_minor}" -lt 21 ]; }; then
        echo "错误：CMakePresets.json 需要 CMake >= 3.21，当前为 ${cmake_version}。" >&2
        echo "请升级 CMake 后重新运行本脚本。" >&2
        return 1
    fi
}

require_cmake_presets_support

# 安装 wget / unzip（vcpkg 等后续步骤需要）
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

# ----- C++ 依赖：优先使用系统 CMake package，必要时才回退 vcpkg -----
# system_cpp_deps_available 已在包管理器操作前定义，用于快速跳过完整环境。
ensure_cpp_deps_via_vcpkg() {
    if [ "${FORCE_VCPKG}" -ne 1 ] && system_cpp_deps_available; then
        CPP_DEPS_PROVIDER="system"
        echo "系统已提供 CMake 可发现的 yaml-cpp / gRPC / Protobuf 及代码生成工具，跳过 vcpkg 安装。"
        return 0
    fi

    if [ "${FORCE_VCPKG}" -eq 1 ]; then
        echo "已启用 --force-vcpkg：强制使用 vcpkg 的 grpc / yaml-cpp。"
    else
        echo "系统 CMake 依赖或代码生成工具不完整，回退到 vcpkg 安装方案。"
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

    VCPKG_SELECTED_ROOT="${vcpkg_root}"
    CPP_DEPS_PROVIDER="vcpkg"
    cat <<EOF

==============================================================
gRPC 已通过 vcpkg 安装到: ${vcpkg_root}

唯一硬件构建入口：

    cmake --preset default
    cmake --build --preset default --parallel 8
==============================================================
EOF
    return 0
}

CPP_DEPS_PROVIDER=""
VCPKG_SELECTED_ROOT=""
if ! ensure_cpp_deps_via_vcpkg; then
    echo "错误：无法准备根构建所需的 C++ 依赖（grpc / yaml-cpp）。" >&2
    echo "请检查上方安装日志、网络或代理配置后重试。" >&2
    exit 1
fi

verify_default_preset() {
    local presets_file="${PROJECT_ROOT}/CMakePresets.json"
    if [ ! -f "${presets_file}" ]; then
        echo "错误：缺少仓库默认 preset：${presets_file}" >&2
        return 1
    fi
    echo "使用唯一硬件 preset：${presets_file}（依赖提供方：${CPP_DEPS_PROVIDER}）"
}

verify_default_preset || exit 1

tsv_clean() {
    printf '%s' "$1" | tr '\t\r\n' '   '
}

write_dependency_row() {
    local output_file="$1"
    local name="$2"
    local version="$3"
    local provider="$4"
    local location="$5"

    printf '%s\t%s\t%s\t%s\n' \
        "$(tsv_clean "${name}")" \
        "$(tsv_clean "${version}")" \
        "$(tsv_clean "${provider}")" \
        "$(tsv_clean "${location}")" >>"${output_file}"
}

write_pkg_config_dependency() {
    local output_file="$1"
    local name="$2"
    local module="$3"
    local version="missing"
    local location="-"

    if command -v pkg-config >/dev/null 2>&1 \
        && pkg-config --exists "${module}" 2>/dev/null; then
        version="$(pkg-config --modversion "${module}")"
        location="$(pkg-config --variable=libdir "${module}")"
    fi
    write_dependency_row "${output_file}" "${name}" "${version}" "system" "${location}"
}

generate_dependency_manifest() {
    local manifest_file="${PROJECT_ROOT}/build_dependencies.tsv"
    local manifest_tmp
    local cmake_version
    local cxx_version
    local cuda_version="missing"
    local cuda_location="-"
    local kernel_version
    local kernel_location

    manifest_tmp="$(mktemp "${PROJECT_ROOT}/.build_dependencies.tsv.XXXXXX")" || return 1
    cmake_version="$(cmake --version | sed -n '1s/^cmake version //p')"
    cxx_version="$(c++ -dumpfullversion -dumpversion 2>/dev/null || c++ --version | head -n1)"
    kernel_version="$(uname -r)"
    kernel_location="/lib/modules/${kernel_version}/build"

    if command -v nvcc >/dev/null 2>&1; then
        cuda_version="$(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p' | tail -n1)"
        cuda_location="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    fi

    {
        echo "# Generated by scripts/prepare_env.sh; machine-local, do not commit."
        echo "# generated_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "# cpp_dependency_provider=${CPP_DEPS_PROVIDER}"
        printf 'name\tversion\tprovider\tlocation\n'
    } >"${manifest_tmp}"

    write_dependency_row "${manifest_tmp}" "cmake" "${cmake_version}" "system" \
        "$(readlink -f "$(command -v cmake)")"
    write_dependency_row "${manifest_tmp}" "cxx-compiler" "${cxx_version}" "system" \
        "$(readlink -f "$(command -v c++)")"
    write_dependency_row "${manifest_tmp}" "cuda-toolkit" "${cuda_version}" "system" \
        "${cuda_location}"
    write_dependency_row "${manifest_tmp}" "linux-kernel-headers" "${kernel_version}" "system" \
        "${kernel_location}"
    write_pkg_config_dependency "${manifest_tmp}" "libunwind" "libunwind"

    if [ "${CPP_DEPS_PROVIDER}" = "vcpkg" ]; then
        local vcpkg_version
        local spec
        local version
        local unused
        local package_name
        local triplet
        local install_prefix
        local package_location

        vcpkg_version="$("${VCPKG_SELECTED_ROOT}/vcpkg" version \
            | sed -n '1s/.*version //p')"
        write_dependency_row "${manifest_tmp}" "vcpkg" "${vcpkg_version}" "vcpkg" \
            "${VCPKG_SELECTED_ROOT}"

        while read -r spec version unused; do
            case "${spec}" in
                *\[*\]*) continue ;;
                *:*) ;;
                *) continue ;;
            esac
            package_name="${spec%%:*}"
            triplet="${spec#*:}"
            install_prefix="${VCPKG_SELECTED_ROOT}/installed/${triplet}"
            package_location="${install_prefix}/share/${package_name}"
            if [ ! -d "${package_location}" ]; then
                package_location="${install_prefix}"
            fi
            write_dependency_row "${manifest_tmp}" "${package_name}" "${version}" \
                "vcpkg:${triplet}" "${package_location}"
        done < <("${VCPKG_SELECTED_ROOT}/vcpkg" list)
    else
        write_pkg_config_dependency "${manifest_tmp}" "grpc" "grpc++"
        write_pkg_config_dependency "${manifest_tmp}" "protobuf" "protobuf"
        write_pkg_config_dependency "${manifest_tmp}" "yaml-cpp" "yaml-cpp"
    fi

    chmod 0644 -- "${manifest_tmp}" || return 1
    mv -f -- "${manifest_tmp}" "${manifest_file}"
    echo "已生成依赖清单：${manifest_file}"
}

generate_dependency_manifest || {
    echo "错误：生成 build_dependencies.tsv 失败。" >&2
    exit 1
}

echo "操作完成！可运行：cmake --preset default && cmake --build --preset default --parallel 8"

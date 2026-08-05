cmake_minimum_required (VERSION 3.18)

set (CMAKE_CUDA_ARCHITECTURES 90)
set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS on CACHE INTERNAL "" FORCE)
# set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
# set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
# set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

project (libnvm LANGUAGES CUDA C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
endif()


set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")

find_package (Threads REQUIRED)
find_package(CUDAToolkit 12.6 REQUIRED)
find_package(PkgConfig)

# Imported-target configuration fallback：
#   部分发行版（如 TencentOS/RHEL 自带的 yaml-cpp 等）
#   仅为 Release / Debug 提供 IMPORTED_LOCATION，不含 RelWithDebInfo / MinSizeRel。
#   不设置该映射时，CMake 会对每个链接到此 target 的目标都警告一次（CMP0111）。
#   下面让这两种构建类型依次尝试自动回退到 Release / NOCONFIG / 无后缀，消除刷屏告警。
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO RelWithDebInfo Release NOCONFIG "")
set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL     MinSizeRel     Release NOCONFIG "")

find_package(yaml-cpp REQUIRED)
# 兼容不同 yaml-cpp 包的 target 命名：
#   vcpkg / 新版 upstream : yaml-cpp::yaml-cpp
#   老的系统包（0.5.x）   : yaml-cpp (裸名 target) 或仅提供 ${YAML_CPP_LIBRARIES}
if (TARGET yaml-cpp::yaml-cpp)
    set(TUTTI_YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif (TARGET yaml-cpp)
    set(TUTTI_YAML_CPP_TARGET yaml-cpp)
else ()
    set(TUTTI_YAML_CPP_TARGET ${YAML_CPP_LIBRARIES})
endif ()
message(STATUS "yaml-cpp target: ${TUTTI_YAML_CPP_TARGET}")

pkg_check_modules(UNWIND REQUIRED libunwind)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "Find CUDA !")
else()
    message(FATAL_ERROR "Not found CUDA!")
endif()

if(CUDAToolkit_VERSION VERSION_LESS 12.6)
    message(FATAL_ERROR "CUDA version must be at least 12.6 (found: ${CUDAToolkit_VERSION})")
endif()

message(STATUS "Using CUDA ${CMAKE_CUDA_COMPILER_VERSION}")

# ---------------------------------------------------------------------------
# CUDA 13+ CCCL header location compatibility shim.
#
# Starting with CUDA 13.0 the C++ Core Libraries (libcudacxx + Thrust + CUB,
# collectively CCCL) were relocated from `<cuda_root>/include/cuda/...` to
# `<cuda_root>/targets/<triple>/include/cccl/cuda/...`. CUDA's own nvcc
# driver injects this path automatically when compiling .cu files, but
# host-only .cpp translation units (e.g. tutti/device_manager/nvme/libnvm/src/
# ctrl.cpp) compiled with g++ still need it on their include path or
# every `#include <cuda/atomic>` chain breaks at preprocess time.
#
# We add it as a *global* include dir because libnvm's headers are
# distributed across multiple targets (libnvm itself, nvmeservice,
# device_manager, io_engine), and every C++ TU that
# transitively pulls libnvm's nvm_types.h needs the same resolution.
#
# Pre-CUDA-13 toolkits do not have this directory; the if(EXISTS ...)
# guard keeps the shim a no-op there so older bring-up environments
# (e.g. CUDA 12.6 boxes) still build unchanged.
# ---------------------------------------------------------------------------
if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
    # CMake exposes the per-triple include dir as CUDAToolkit_INCLUDE_DIRS[0];
    # on x86_64 Linux that resolves to <root>/targets/x86_64-linux/include.
    # The CCCL relocation lives at the sibling `cccl/` subdir.
    set(_cccl_include "${CUDAToolkit_TARGET_DIR}/include/cccl")
    if (EXISTS "${_cccl_include}/cuda/atomic")
        message(STATUS "CUDA 13+ detected; adding CCCL include dir: ${_cccl_include}")
        include_directories(SYSTEM "${_cccl_include}")
    else ()
        message(WARNING
            "CUDA ${CUDAToolkit_VERSION} reports CCCL relocation but "
            "${_cccl_include}/cuda/atomic is missing. Host-only TUs that "
            "include <cuda/atomic> will fail. Check that "
            "cuda-cccl-${CUDAToolkit_VERSION_MAJOR}-0 is installed.")
    endif ()
endif ()

set (NVIDIA "" CACHE PATH "Path to Nvidia driver source")
set (KERNEL "/lib/modules/${CMAKE_SYSTEM_VERSION}/build" CACHE PATH "Path to kernel source, or module directory")

# set (nvidia_archs "90" CACHE STRING "NVIDIA compute architectures")
set (no_module false CACHE BOOL "Do not build kernel module")
set (no_cuda false CACHE BOOL "Do not build library with CUDA support")
set (no_cc false CACHE BOOL "Do not assume cache-coherent architecture")
set (BUILD_SHARED_LIBS true CACHE BOOL "Build shared library, set to false for static")


# Set global compiler flags
set (GPU_ARCHS "" CACHE INTERNAL "" FORCE)
foreach (arch ${nvidia_archs})
    string (CONCAT GPU_ARCHS ${GPU_ARCHS} " -gencode arch=compute_${arch},code=sm_${arch}")
endforeach ()

set (convert_warnings implicit-function-declaration incompatible-pointer-types int-conversion return-type unused-function unused-variable)
set (flags "")
foreach (flag ${convert_warnings})
    string (APPEND flags " -Werror=${flag}")
endforeach()


# C_FLAGS
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra ${flags}" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
# CXX_FLAGS
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++20" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG " CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -std=c++20" CACHE INTERNAL "" FORCE)
# CUDA_FLAGS
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -D__CUDA__ -D__CUDACC__ -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -rdc=true ${GPU_ARCHS}")
set (CUDA_NVCC_FLAGS_DEBUG "${CMAKE_NVCC_FLAGS_DEBUG} -DDEBUG" CACHE INTERNAL "" FORCE)
set (CUDA_NVCC_FLAGS_RELEASE "${CMAKE_NVCC_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)

# set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xlinker -rpath=${CMAKE_BINARY_DIR}/lib --expt-relaxed-constexpr --expt-extended-lambda  -lcudart ${GPU_ARCHS}")
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xlinker -rpath=\\$ORIGIN -Xlinker -rpath=${CMAKE_BINARY_DIR}/lib -rdc=true --expt-relaxed-constexpr --expt-extended-lambda -lineinfo ${GPU_ARCHS}")
set (CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)

if (no_cc)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CCUDA_NVCC_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
endif ()

# ---------------------------------------------------------------------------
# Phase 3 header hygiene: add the tutti/ subdirectory BEFORE the global
# libnvm include_directories below.  This ensures that contract targets
# (tutti_api, tutti_spi, tutti_cuda_like, tutti_types) and their consumer
# tests — all created inside tutti/CMakeLists.txt — do NOT inherit the
# libnvm private include paths.  Consumers linking only tutti_api must
# not be able to #include <nvm_types.h> or any other local-NVMe private
# header.  The global includes are still available to legacy root-scope
# targets (libnvm, nvmeservice, memory, device_manager, …) added below.
# ---------------------------------------------------------------------------
set(TUTTI_BUILD_HARDWARE_STACK OFF CACHE BOOL
    "Nested tutti/ hardware stack (forced OFF when included from repo root)" FORCE)
if(EXISTS "${PROJECT_SOURCE_DIR}/tutti/CMakeLists.txt")
    add_subdirectory(tutti)
    message(STATUS "Added tutti subdirectory (hardware stack disabled from root; placed before global libnvm includes for header hygiene)")
endif()

# Global libnvm include paths — available to root-scope legacy targets only.
# The tutti/ subdirectory (added above) does NOT inherit these.
include_directories ("${PROJECT_SOURCE_DIR}/tutti/device_manager/nvme/libnvm/src"
                     "${PROJECT_SOURCE_DIR}/tutti/device_manager/nvme/libnvm/include")

# Try to find Nvidia driver
if (CUDAToolkit_FOUND AND NOT no_cuda)

    # Try to find Nvidia headers (will choose the newest version)
    file (GLOB_RECURSE driver_paths "/usr/src/nvidia-*/nv-p2p.h")
    string (REPLACE "nv-p2p.h" "" driver_paths "${driver_paths}")
    if (driver_paths)
        list (REVERSE driver_paths)
    endif ()

    find_path (driver_include NAMES "nv-p2p.h" PATHS "${NVIDIA}" ${driver_paths})
    set (driver_dir "${driver_include}")

    # Find Nvidia driver symbols
    if (NOT EXISTS "${driver_dir}/Module.symvers")
        get_filename_component (driver_dir "${driver_dir}" DIRECTORY)
    endif ()

    if ("${driver_dir}" STREQUAL "")
        message(STATUS "No NVIDIA driver symbols found")
    else ()
        message(STATUS "Using NVIDIA driver found in ${driver_dir}")
    endif ()
endif ()


# Library source files
# libnvm source of truth: tutti/device_manager/nvme/libnvm/ (root tree copy removed)
set (libnvm_root "${PROJECT_SOURCE_DIR}/tutti/device_manager/nvme/libnvm/")
file (GLOB libnvm_source "${libnvm_root}/src/*.cpp")
file (GLOB libnvm_cuda_source "${libnvm_root}/src/*.cu")
file (GLOB libnvm_linux_source "${libnvm_root}/src/linux/*.cpp")
file (GLOB libnvm_include "${libnvm_root}/include/*.h")


# # Module source files
#
# snvme 内核模块按 kernel baseline 分目录维护：
#   tutti/device_manager/nvme/kernel_modules/snvme-<tag>/
#
# 命名规范：<tag> 必须是该 baseline 对应 `uname -r` 的前缀，例如：
#   snvme-5.15.0-public           -> 上游 5.15.0
#   snvme-5.4.241-1-tlinux4-0017  -> TencentOS 5.4.241-1-tlinux4-0017.x
#
# 选择优先级：
#   1) -DSNVME_KERNEL_VERSION=<tag>  用户显式指定
#   2) 否则按 `uname -r` 做最长前缀匹配自动选
set (SNVME_KERNEL_VERSION "" CACHE STRING
    "snvme baseline tag (= dir suffix). Leave empty to auto-detect from `uname -r`.")

# snvme kernel module source of truth: tutti/device_manager/nvme/kernel_modules/
set (snvme_root_dir "${PROJECT_SOURCE_DIR}/tutti/device_manager/nvme/kernel_modules")

# 收集所有合法 baseline tag（必须含 Makefile.in）
file (GLOB _snvme_cands RELATIVE "${snvme_root_dir}" "${snvme_root_dir}/snvme-*")
set (_snvme_tags "")
foreach (_c IN LISTS _snvme_cands)
    if (IS_DIRECTORY "${snvme_root_dir}/${_c}" AND EXISTS "${snvme_root_dir}/${_c}/Makefile.in")
        string (REGEX REPLACE "^snvme-" "" _tag "${_c}")
        list (APPEND _snvme_tags "${_tag}")
    endif ()
endforeach ()

if (NOT _snvme_tags)
    message (FATAL_ERROR "在 ${snvme_root_dir} 下未找到任何带 Makefile.in 的 snvme-* baseline 目录。")
endif ()

# 没显式指定则按 uname -r 前缀匹配
if (NOT SNVME_KERNEL_VERSION)
    execute_process (COMMAND uname -r
        OUTPUT_VARIABLE _uname_r OUTPUT_STRIP_TRAILING_WHITESPACE)
    message (STATUS "Detected running kernel: ${_uname_r}")
    set (_best_tag "")
    set (_best_len 0)
    foreach (_tag IN LISTS _snvme_tags)
        string (LENGTH "${_tag}" _tlen)
        string (LENGTH "${_uname_r}" _ulen)
        # 取 uname-r 中长度等于 _tlen 的前缀来比较
        if (_tlen LESS_EQUAL _ulen)
            string (SUBSTRING "${_uname_r}" 0 ${_tlen} _u_prefix)
            if ("${_u_prefix}" STREQUAL "${_tag}" AND _tlen GREATER _best_len)
                set (_best_tag "${_tag}")
                set (_best_len "${_tlen}")
            endif ()
        endif ()
    endforeach ()
    if (_best_tag)
        set (SNVME_KERNEL_VERSION "${_best_tag}")
        message (STATUS "Auto-selected snvme baseline by uname -r match: ${SNVME_KERNEL_VERSION}")
    else ()
        message (FATAL_ERROR
            "无法为当前 kernel '${_uname_r}' 自动匹配 snvme baseline。\n"
            "可用 baseline: ${_snvme_tags}\n"
            "请显式指定: -DSNVME_KERNEL_VERSION=<tag>")
    endif ()
endif ()

set (module_root "${snvme_root_dir}/snvme-${SNVME_KERNEL_VERSION}")
if (NOT EXISTS "${module_root}/Makefile.in")
    message (FATAL_ERROR
        "snvme baseline '${SNVME_KERNEL_VERSION}' not found at ${module_root} "
        "(missing Makefile.in). Available baselines: ${_snvme_tags}")
endif ()
message (STATUS "Using snvme kernel baseline: ${SNVME_KERNEL_VERSION} (${module_root})")
set (module_output "${CMAKE_BINARY_DIR}/module")
# for Makefile.in, do not delete
set (module_ccflags "-I${PROJECT_SOURCE_DIR}/tutti/device_manager/nvme/libnvm/include -I${driver_include}")
configure_file ("${module_root}/Makefile.in" "${module_output}/Makefile" @ONLY)
message (STATUS "Configuring module build in ${module_output}")

# The module Makefile recursively invokes the kernel build system.  Mark every
# CMake-launched make as jobserver-aware so a parallel parent build shares its
# token pool instead of warning and falling back to -j1.  JOB_SERVER_AWARE was
# added in CMake 3.28; with older Makefile generators, the literal $(MAKE)
# provides GNU Make's equivalent recursive-make marker.  Non-Makefile parent
# generators do not own a GNU Make jobserver, so they launch make normally.
find_program (SNVME_MAKE_EXECUTABLE NAMES gmake make REQUIRED)
set (_snvme_make_command "${SNVME_MAKE_EXECUTABLE}")
set (_snvme_jobserver_args "")
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    list (APPEND _snvme_jobserver_args JOB_SERVER_AWARE TRUE)
elseif (CMAKE_GENERATOR MATCHES "Makefiles")
    set (_snvme_make_command "$(MAKE)")
endif ()

add_custom_target(
    modules
    ALL
    COMMAND ${_snvme_make_command}
    WORKING_DIRECTORY ${module_output}
    COMMENT "Building kernel module"
    ${_snvme_jobserver_args}
)

add_custom_target(
    clean_modules
    COMMAND ${_snvme_make_command} clean
    WORKING_DIRECTORY ${module_output}
    COMMENT "Cleaning kernel module"
    ${_snvme_jobserver_args}
)

add_custom_target(
    insmod
    COMMAND sudo ${_snvme_make_command} insmod
    WORKING_DIRECTORY ${module_output}
    COMMENT "Inserting kernel module"
    ${_snvme_jobserver_args}
)

add_custom_target(
    rmmod
    COMMAND sudo ${_snvme_make_command} rmmod
    WORKING_DIRECTORY ${module_output}
    COMMENT "Removing kernel module"
    ${_snvme_jobserver_args}
)

# Benchmarks
set (benchmarks_root "${PROJECT_SOURCE_DIR}/benchmarks")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")


# Create NVME library target
# add_library(libnvm ${libnvm_source} ${libnvm_linux_source})
# CUDA_ADD_LIBRARY (libnvm ${libnvm_source} ${libnvm_linux_source}) # TODO: add check for OS
add_library(libnvm ${libnvm_source} ${libnvm_cuda_source} ${libnvm_linux_source})
set_target_properties(libnvm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(libnvm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories (libnvm PRIVATE "${libnvm_root}")
target_link_libraries(libnvm
    PUBLIC tutti_cuda_like
    PRIVATE CUDA::cudart Threads::Threads
)
message (STATUS "Configuring libnvm without SmartIO")


if (CUDA_FOUND AND NOT no_cuda)
    target_compile_definitions(libnvm PRIVATE _CUDA __CUDA__)
endif ()

# Set name of library
set_target_properties (libnvm PROPERTIES OUTPUT_NAME "nvm")
set_target_properties (libnvm PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
# Set install locations
install (FILES ${libnvm_include} DESTINATION "include")
install (TARGETS libnvm DESTINATION "lib")

# (tutti/ subdirectory added earlier, before the global libnvm includes,
#  to prevent private header propagation into contract targets.)

# memory/ — retired (Round 13 S3); cache concepts migrated to
# tutti/data_paths/local_nvme/metadata/ (Round 11 S2).

# The legacy monolithic GPU-file tree has been moved out of
# the repository (kept as an external backup); nothing here references it.

# NVMeService — tutti side is the single source of truth (Round 13 S3).
# backends/local/NVMeService/ deleted; root build compiles from tutti/ side.
set (nvmeservice_root "${PROJECT_SOURCE_DIR}/tutti/device_manager/nvme/nvmeservice")
file (GLOB nvmeservice_sources "${nvmeservice_root}/src/*.cpp" "${nvmeservice_root}/src/*.cu")

# ---------------------------------------------------------------------------
# gRPC discovery
#
# 推荐通过以下任一方式提供 Protobuf / gRPC：
#   1) vcpkg toolchain（跨发行版统一版本，推荐）
#        cmake --preset default
#      或
#        cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
#   2) 系统包（apt/dnf 等）：安装后 CMake 会自动从默认路径找到
#   3) 自定义安装位置：通过 -DCMAKE_PREFIX_PATH=<your-prefix> 提示 CMake
# ---------------------------------------------------------------------------
find_package(gRPC CONFIG REQUIRED)

if (DEFINED Protobuf_VERSION)
    message(STATUS "Using Protobuf: version=${Protobuf_VERSION}")
endif ()
if (TARGET protobuf::libprotobuf)
    message(STATUS "  - target: protobuf::libprotobuf available")
endif ()
message(STATUS "Found gRPC: target gRPC::grpc++ ready")

# protoc 可执行文件：优先使用 imported target（modern protobuf-config 提供）
if (TARGET protobuf::protoc)
    set(TUTTI_PROTOC_EXECUTABLE "$<TARGET_FILE:protobuf::protoc>")
elseif (Protobuf_PROTOC_EXECUTABLE)
    set(TUTTI_PROTOC_EXECUTABLE "${Protobuf_PROTOC_EXECUTABLE}")
else ()
    message(FATAL_ERROR "protoc not found. Please ensure protobuf is properly installed.")
endif ()

set(nvmeservice_proto "${nvmeservice_root}/src/nvmeservice.proto")
get_target_property(GRPC_CPP_PLUGIN_LOCATION gRPC::grpc_cpp_plugin LOCATION)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/nvmeservice.pb.cc"
           "${CMAKE_CURRENT_BINARY_DIR}/nvmeservice.pb.h"
           "${CMAKE_CURRENT_BINARY_DIR}/nvmeservice.grpc.pb.cc"
           "${CMAKE_CURRENT_BINARY_DIR}/nvmeservice.grpc.pb.h"
    COMMAND "${TUTTI_PROTOC_EXECUTABLE}"
    ARGS --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
         --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
         -I "${nvmeservice_root}/src"
         --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN_LOCATION}"
         "${nvmeservice_proto}"
    DEPENDS "${nvmeservice_proto}"
)

set(NVMESERVICE_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/nvmeservice.pb.cc")
set(NVMESERVICE_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/nvmeservice.pb.h")
set(NVMESERVICE_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/nvmeservice.grpc.pb.cc")
set(NVMESERVICE_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/nvmeservice.grpc.pb.h")

add_library(nvmeservice ${nvmeservice_sources} ${NVMESERVICE_PROTO_SRCS} ${NVMESERVICE_GRPC_SRCS})
target_include_directories(nvmeservice PUBLIC
    "${nvmeservice_root}/src"
    "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(nvmeservice PRIVATE libnvm CUDA::cudart Threads::Threads)
# 现代写法：通过 imported target 链接 protobuf + grpc，include 路径自动传播
if (TARGET protobuf::libprotobuf)
    target_link_libraries(nvmeservice PUBLIC gRPC::grpc++ protobuf::libprotobuf)
else ()
    # 极老的 FindProtobuf.cmake 兜底
    target_link_libraries(nvmeservice PUBLIC gRPC::grpc++ ${Protobuf_LIBRARIES})
    target_include_directories(nvmeservice PUBLIC ${Protobuf_INCLUDE_DIRS})
endif ()
target_link_libraries(nvmeservice PRIVATE ${TUTTI_YAML_CPP_TARGET})
set_target_properties(nvmeservice PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(nvmeservice PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(nvmeservice PROPERTIES OUTPUT_NAME "nvmeservice")
install (TARGETS nvmeservice DESTINATION "lib")

# tutti_daemon: production daemon (source migrated to tutti side, Round 13 S3).
# examples/ and backends/local/NVMeService/examples/ deleted; the daemon
# target is defined here for root build (standalone tutti/ defines it in
# tutti/device_manager/nvme/nvmeservice/examples/CMakeLists.txt).
add_executable(tutti_daemon "${nvmeservice_root}/examples/tutti_daemon.cpp")
target_link_libraries(tutti_daemon PRIVATE
    nvmeservice libnvm gRPC::grpc++ protobuf::libprotobuf Threads::Threads CUDA::cudart)
set_target_properties(tutti_daemon PROPERTIES
    BUILD_RPATH "$ORIGIN:$ORIGIN/../lib" INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib")
install(TARGETS tutti_daemon DESTINATION "bin")

# device_manager/ (root) — retired (Round 13 S3); control plane migrated to
# tutti/data_paths/local_nvme/control/ (Round 10 S2).

# Legacy trees deleted in Round 13 Session 2 (no production consumers):
#   nvme_storage/     — removed
#   block_storage/    — removed
#   io_engine/        — removed
#   coordinator/      — removed
#   adapters/kv_cache/— removed
# Their add_subdirectory calls and example/test wiring have been removed.
# Design docs archived in doc/history/.

# examples/ — retired (Round 13 S3); tutti_daemon migrated to
# tutti/device_manager/nvme/nvmeservice/examples/, smokes deleted with legacy trees.

# tests/ holds layer-specific smoke tests. Added after all layers
# and core libraries are declared.
if(EXISTS "${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
    message(STATUS "Added tests subdirectory")
endif()


# Macro for creating sample target
macro (make_sample target binary_name files)
    cuda_add_executable (${target} EXCLUDE_FROM_ALL ${files})
    
    add_dependencies (${target} libnvm)
    target_link_libraries (${target} PRIVATE libnvm)

    # TODO: Do we want to install samples or not?
    #set_target_properties (${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/samples")
    set_target_properties (${target} PROPERTIES OUTPUT_NAME "nvm-${binary_name}")

    list (APPEND sample_targets "${target}")
    set (sample_targets "${sample_targets}" PARENT_SCOPE)

    #install (TARGETS ${target} DESTINATION "bin")
endmacro ()

# Make target depend on pthreads
macro (set_multithread target)
    target_link_libraries (${target} PRIVATE Threads::Threads)
endmacro ()

# Set preferred ordering for sample targets
# Make benchmark target
# We assume all benchmarks are CUDA programs
macro (make_benchmark target binary_name files)
    if (CUDA_FOUND AND NOT no_cuda)
        cuda_add_executable (${target}) # Ugly bugly
        target_sources (${target} PRIVATE ${files})

        target_link_libraries (${target}  libnvm CUDA::cudart)
        set_target_properties (${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_target_properties (${target} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        set_target_properties (${target} PROPERTIES OUTPUT_NAME "nvm-${binary_name}")

        list (APPEND benchmark_targets "${target}")
        set (benchmark_targets "${benchmark_targets}" PARENT_SCOPE)
    endif ()
endmacro ()

macro (make_samples_choice ctarget sisci_target other_target)
    if (NOT no_smartio_samples AND sisci_include AND sisci_lib)
        add_custom_target (${ctarget} DEPENDS ${sisci_target})
        list (REMOVE_ITEM sample_targets "${other_target}")
        set (sample_targets "${sample_targets}" PARENT_SCOPE)
    else ()
        add_custom_target (${ctarget} DEPENDS ${other_target})
        list (REMOVE_ITEM sample_targets "${sisci_target}")
        set (sample_targets "${sample_targets}" PARENT_SCOPE)
    endif ()
endmacro ()


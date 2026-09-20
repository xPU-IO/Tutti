# snvme kernel-module targets for the default hardware build. They are tied to
# the running kernel and the NVIDIA P2P API.

set(TUTTI_P2P_BACKEND "" CACHE STRING
    "SNVMe peer-memory backend: nvidia|metax (empty selects from TUTTI_ACCELERATOR)")
if(NOT TUTTI_P2P_BACKEND)
    if(TUTTI_ACCELERATOR STREQUAL "CUDA")
        set(_snvme_default_backend nvidia)
    elseif(TUTTI_ACCELERATOR STREQUAL "MUSA" OR
           TUTTI_ACCELERATOR STREQUAL "MACA")
        set(_snvme_default_backend metax)
    else()
        message(FATAL_ERROR
            "No SNVMe P2P backend default exists for accelerator "
            "'${TUTTI_ACCELERATOR}'. Set -DTUTTI_P2P_BACKEND=<backend>.")
    endif()
    set(TUTTI_P2P_BACKEND "${_snvme_default_backend}" CACHE STRING
        "SNVMe peer-memory backend: nvidia|metax" FORCE)
endif()
string(TOLOWER "${TUTTI_P2P_BACKEND}" _snvme_backend)
if(NOT _snvme_backend MATCHES "^(nvidia|metax)$")
    message(FATAL_ERROR
        "TUTTI_P2P_BACKEND='${TUTTI_P2P_BACKEND}' is unsupported. "
        "Supported backends: nvidia, metax")
endif()
set(TUTTI_P2P_BACKEND "${_snvme_backend}" CACHE STRING
    "SNVMe peer-memory backend: nvidia|metax" FORCE)
set_property(CACHE TUTTI_P2P_BACKEND PROPERTY STRINGS nvidia metax)

set(SNVME_P2P_INCLUDE_DIR "" CACHE PATH
    "Directory containing the selected backend's P2P header")
set(NVIDIA "" CACHE PATH
    "Legacy NVIDIA nv-p2p.h search hint; prefer SNVME_P2P_INCLUDE_DIR")
set(SNVME_KERNEL_VERSION "" CACHE STRING
    "snvme baseline tag; leave empty to match the running kernel")

set(_snvme_root_dir
    "${TUTTI_SOURCE_DIR}/device_manager/nvme/kernel_modules")

# 唯一源码树：snvme/ 持有 snvme-private 代码的单副本，加上 baseline/<tag>/
# 下的各内核世代上游文件；kbuild Makefile 按运行内核自动选基线（见
# kernel_modules/PORTING.md）。历史上的 snvme-* 版本树已删除，这里不再有
# 目录探测或 legacy fallback。
set(_snvme_unified_dir "${_snvme_root_dir}/snvme")
if(NOT EXISTS "${_snvme_unified_dir}/Makefile.in")
    message(FATAL_ERROR
        "Unified snvme tree not found: ${_snvme_unified_dir}/Makefile.in")
endif()
set(module_root "${_snvme_unified_dir}")
if(SNVME_KERNEL_VERSION AND NOT SNVME_KERNEL_VERSION STREQUAL "unified")
    message(WARNING
        "SNVME_KERNEL_VERSION='${SNVME_KERNEL_VERSION}' is ignored by the "
        "unified snvme tree. Leave it unset; Makefile.in selects the "
        "running-kernel baseline."
    )
endif()
set(SNVME_KERNEL_VERSION "unified" CACHE STRING
    "snvme source layout selected for the module build" FORCE)

set(_snvme_backend_source
    "${module_root}/peer_memory/${TUTTI_P2P_BACKEND}.c")
if(NOT EXISTS "${_snvme_backend_source}")
    message(FATAL_ERROR
        "SNVMe backend '${TUTTI_P2P_BACKEND}' is not implemented for baseline "
        "'${SNVME_KERNEL_VERSION}': ${_snvme_backend_source} is missing")
endif()

if(TUTTI_P2P_BACKEND STREQUAL "nvidia")
    set(_snvme_backend_header "nv-p2p.h")
    file(GLOB_RECURSE _snvme_vendor_headers "/usr/src/nvidia-*/nv-p2p.h")
    # Vendored MIT-licensed nv-p2p.h (third_pkgs/nvidia/) is searched first so
    # the build is reproducible on hosts without a system NVIDIA driver source
    # tree (e.g. rpm-installed Open Kernel Module); /usr/src/nvidia-* and the
    # legacy NVIDIA hint remain as fallbacks.
    set(_snvme_vendor_hints
        "${TUTTI_REPOSITORY_ROOT}/third_pkgs/nvidia"
        "${NVIDIA}")
    foreach(_vendor_header IN LISTS _snvme_vendor_headers)
        get_filename_component(_vendor_hint "${_vendor_header}" DIRECTORY)
        list(APPEND _snvme_vendor_hints "${_vendor_hint}")
    endforeach()
else()
    set(_snvme_vendor_hints
        "${MUSA_INCLUDE_DIR}"
        "${MACA_INCLUDE_DIR}"
        "${MACA_ROOT}/include"
        "/usr/local/musa/include"
        "/opt/maca/include")
endif()

if(SNVME_P2P_INCLUDE_DIR)
    if(NOT EXISTS "${SNVME_P2P_INCLUDE_DIR}/${_snvme_backend_header}")
        message(FATAL_ERROR
            "SNVME_P2P_INCLUDE_DIR='${SNVME_P2P_INCLUDE_DIR}' does not contain "
            "${_snvme_backend_header} for backend '${TUTTI_P2P_BACKEND}'")
    endif()
    set(_snvme_p2p_include "${SNVME_P2P_INCLUDE_DIR}")
elseif(TUTTI_P2P_BACKEND STREQUAL "nvidia")
    find_path(_snvme_nvidia_include "${_snvme_backend_header}"
        PATHS ${_snvme_vendor_hints})
    set(_snvme_p2p_include "${_snvme_nvidia_include}")
else()
    find_path(_snvme_metax_include "${_snvme_backend_header}"
        PATHS ${_snvme_vendor_hints})
    set(_snvme_p2p_include "${_snvme_metax_include}")
endif()

if(NOT _snvme_p2p_include)
    message(FATAL_ERROR
        "${_snvme_backend_header} was not found for SNVMe backend "
        "'${TUTTI_P2P_BACKEND}'. Set -DSNVME_P2P_INCLUDE_DIR=<header-directory>.")
endif()

set(module_output "${CMAKE_BINARY_DIR}/module")
set(module_ccflags
    "-I${TUTTI_SOURCE_DIR}/device_manager/nvme/libnvm/include -I${_snvme_p2p_include}")
configure_file("${module_root}/Makefile.in" "${module_output}/Makefile" @ONLY)

find_program(SNVME_MAKE_EXECUTABLE NAMES gmake make REQUIRED)
set(_snvme_make_command "${SNVME_MAKE_EXECUTABLE}")
set(_snvme_jobserver_args "")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    list(APPEND _snvme_jobserver_args JOB_SERVER_AWARE TRUE)
elseif(CMAKE_GENERATOR MATCHES "Makefiles")
    set(_snvme_make_command "$(MAKE)")
endif()

add_custom_target(modules
    ALL
    COMMAND ${_snvme_make_command} TUTTI_P2P_BACKEND=${TUTTI_P2P_BACKEND}
    WORKING_DIRECTORY "${module_output}"
    COMMENT "Building snvme kernel modules (${SNVME_KERNEL_VERSION})"
    ${_snvme_jobserver_args})
add_custom_target(clean_modules
    COMMAND ${_snvme_make_command} TUTTI_P2P_BACKEND=${TUTTI_P2P_BACKEND} clean
    WORKING_DIRECTORY "${module_output}"
    ${_snvme_jobserver_args})
add_custom_target(insmod
    COMMAND ${_snvme_make_command} TUTTI_P2P_BACKEND=${TUTTI_P2P_BACKEND} insmod
    WORKING_DIRECTORY "${module_output}"
    ${_snvme_jobserver_args})
add_custom_target(rmmod
    COMMAND ${_snvme_make_command} TUTTI_P2P_BACKEND=${TUTTI_P2P_BACKEND} rmmod
    WORKING_DIRECTORY "${module_output}"
    ${_snvme_jobserver_args})

message(STATUS
    "Tutti: snvme module targets enabled: baseline=${SNVME_KERNEL_VERSION}, "
    "backend=${TUTTI_P2P_BACKEND}, include=${_snvme_p2p_include}")

# tutti/cmake/accelerators/MACA.cmake -- MACA profile (Metax MACA SDK)
#
# TEMPLATE — Metax must fill in the real toolchain/flags.
#
# Included by the root CMakeLists.txt when TUTTI_ACCELERATOR=MACA.
# Provides:
#   - TUTTI_BUILD_HARDWARE_STACK default ON
#   - tutti_configure_cuda_like() function (MACA variant)

# example for cu-bridge build with MACA SDK (Metax) and SNVMe kernel module
# #build_ko & insmod/rmmod
# cmake_maca --preset=maca-module --fresh -DSNVME_KERNEL_VERSION=5.15.0-public -DTUTTI_BUILD_HARDWARE_TESTS=ON
# cmake_maca --build build/maca-module --preset=maca-module --target modules
# cmake --build build/maca-module --target insmod
# cmake --build build/maca-module --target rmmod
#
# #build daemon
# cmake_maca --build build/maca-module/ --preset maca-module --target tutti_daemon -j8
#
# #build_lib
# cmake_maca --preset maca -DTUTTI_BUILD_HARDWARE_TESTS=ON -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
# cmake_maca  --build build/maca/ --preset maca --target tutti_layerwise_kv_overlap  -j8

set(TUTTI_BUILD_HARDWARE_STACK ON CACHE BOOL
    "Build hardware stack (accel/device_manager/backends/io_engine)")

# ---------------------------------------------------------------------------
# MACA SDK paths — ported from Mooncake (Apache 2.0)
#
# Mooncake's mooncake-common/common.cmake confirms:
#   root:    $MACA_HOME env var or /opt/maca
#   include: ${MACA_ROOT}/include
#   lib:     ${MACA_ROOT}/lib64 or ${MACA_ROOT}/lib
#   headers: mcr/maca.h, mcr/mc_runtime.h, mcr/mc_runtime_api.h
# Reference: third_pkgs/Mooncake/mooncake-common/common.cmake:246-284
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# tutti_configure_cuda_like(<target_name>)
# ---------------------------------------------------------------------------
function(tutti_configure_cuda_like target_name)
    # MACA SDK root: $MACA_HOME env var, or /opt/maca fallback (Mooncake-confirmed)
    if(NOT DEFINED MACA_ROOT OR MACA_ROOT STREQUAL "")
        if(DEFINED ENV{MACA_HOME} AND NOT "$ENV{MACA_HOME}" STREQUAL "")
            set(MACA_ROOT "$ENV{MACA_HOME}" CACHE PATH "MACA SDK root" FORCE)
        else()
            set(MACA_ROOT "/opt/maca" CACHE PATH "MACA SDK root" FORCE)
        endif()
    endif()

    set(MACA_INCLUDE_DIR
        "${MACA_ROOT}/include"
        "${MACA_ROOT}/mcr/include"
        CACHE STRING "MACA SDK include dir")

    if(EXISTS "${MACA_ROOT}/lib64")
        set(MACA_LIB_DIR "${MACA_ROOT}/lib64" CACHE PATH "MACA SDK lib dir" FORCE)
    else()
        set(MACA_LIB_DIR "${MACA_ROOT}/lib" CACHE PATH "MACA SDK lib dir" FORCE)
    endif()

    # MACA runtime libraries (Mooncake-confirmed: mcruntime mxc-runtime64 rt).
    # mcruntime provides the mc* runtime API (mcMalloc/mcStreamCreate/...);
    # mxc-runtime64 is the MXMACA runtime.  Override with -DMACA_RUNTIME_LIBS.
    set(MACA_RUNTIME_LIBS "mcruntime;mxc-runtime64;runtime_cu" CACHE STRING
        "MACA runtime libraries" FORCE)

    target_compile_definitions(${target_name} INTERFACE TUTTI_USE_MACA=1)

    target_include_directories(${target_name} INTERFACE
        ${MACA_INCLUDE_DIR}
        "${TUTTI_SOURCE_DIR}/include"
    )

    target_link_directories(${target_name} INTERFACE ${MACA_LIB_DIR})
    target_link_libraries(${target_name} INTERFACE
        ${MACA_RUNTIME_LIBS}
    )

    enable_language(CUDA)
    list(APPEND CMAKE_MODULE_PATH "${MACA_ROOT}/tools/cu-bridge/cmake_module/maca/")

    # Expose for downstream CMakeLists that link ${TUTTI_ACCEL_RUNTIME_LIBS}
    set(TUTTI_ACCEL_RUNTIME_LIBS "${MACA_RUNTIME_LIBS}" CACHE INTERNAL "MACA runtime lib")
    set(TUTTI_ACCEL_DRIVER_LIBS  "" CACHE INTERNAL "MACA driver lib")

    message(STATUS "Tutti: MACA profile — root=${MACA_ROOT} include=${MACA_INCLUDE_DIR} lib=${MACA_LIB_DIR} runtime=${MACA_RUNTIME_LIBS}")
endfunction()

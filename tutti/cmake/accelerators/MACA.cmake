# tutti/cmake/accelerators/MACA.cmake -- MACA profile (Metax MACA SDK)
#
# TEMPLATE — Metax must fill in the real toolchain/flags.
#
# Included by the root CMakeLists.txt when TUTTI_ACCELERATOR=MACA.
# Provides:
#   - TUTTI_BUILD_HARDWARE_STACK default ON
#   - tutti_configure_cuda_like() function (MACA variant)

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

    set(MACA_INCLUDE_DIR "${MACA_ROOT}/include" CACHE PATH "MACA SDK include dir")
    if(EXISTS "${MACA_ROOT}/lib64")
        set(MACA_LIB_DIR "${MACA_ROOT}/lib64" CACHE PATH "MACA SDK lib dir" FORCE)
    else()
        set(MACA_LIB_DIR "${MACA_ROOT}/lib" CACHE PATH "MACA SDK lib dir" FORCE)
    endif()

    target_compile_definitions(${target_name} INTERFACE TUTTI_USE_MACA=1)

    target_include_directories(${target_name} INTERFACE
        ${MACA_INCLUDE_DIR}
        "${TUTTI_SOURCE_DIR}/include"
    )

    target_link_directories(${target_name} INTERFACE ${MACA_LIB_DIR})

    # Expose for downstream CMakeLists that link ${TUTTI_ACCEL_RUNTIME_LIBS}
    # TODO(Metax): confirm the MACA runtime/driver library target names
    # (Mooncake does not link a named target — it uses link_directories +
    # the mc* symbols resolved by the compiler).  If MACA SDK ships named
    # libraries (e.g. mcrt), set them here.
    set(TUTTI_ACCEL_RUNTIME_LIBS "" CACHE INTERNAL "MACA runtime lib (Metax confirm)")
    set(TUTTI_ACCEL_DRIVER_LIBS  "" CACHE INTERNAL "MACA driver lib (Metax confirm)")

    message(STATUS "Tutti: MACA profile — root=${MACA_ROOT} include=${MACA_INCLUDE_DIR} lib=${MACA_LIB_DIR}")
endfunction()

# TODO(Metax): set MACA runtime/driver library target names.
# Empty by default — downstream CMakeLists.txt link ${TUTTI_ACCEL_RUNTIME_LIBS}
# / ${TUTTI_ACCEL_DRIVER_LIBS} (see CUDA.cmake for the CUDA equivalent).
set(TUTTI_ACCEL_RUNTIME_LIBS "" CACHE INTERNAL "Vendor runtime lib (MACA stub)")
set(TUTTI_ACCEL_DRIVER_LIBS  "" CACHE INTERNAL "Vendor driver lib (MACA stub)")

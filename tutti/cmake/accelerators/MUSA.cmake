# tutti/cmake/accelerators/MUSA.cmake -- MUSA profile (Metax MUSA SDK)
#
# Definitive implementation ported from Mooncake (Apache 2.0).
#
# Mooncake's mooncake-common/common.cmake confirms:
#   include path: /usr/local/musa/include
#   lib path:     /usr/local/musa/lib
#   libraries:    musa musart rt
# Reference: third_pkgs/Mooncake/mooncake-common/common.cmake:306-310
#            third_pkgs/Mooncake/mooncake-transfer-engine/src/CMakeLists.txt:85-87
# License: Apache 2.0
#
# Included by the root CMakeLists.txt when TUTTI_ACCELERATOR=MUSA.
# Provides:
#   - TUTTI_BUILD_HARDWARE_STACK default ON
#   - tutti_configure_cuda_like() function (MUSA variant)

set(TUTTI_BUILD_HARDWARE_STACK ON CACHE BOOL
    "Build hardware stack (accel/device_manager/backends/io_engine)")

# ---------------------------------------------------------------------------
# tutti_configure_cuda_like(<target_name>)
# ---------------------------------------------------------------------------
function(tutti_configure_cuda_like target_name)
    # MUSA SDK paths (Mooncake-confirmed)
    set(MUSA_INCLUDE_DIR "/usr/local/musa/include" CACHE PATH "MUSA SDK include dir")
    set(MUSA_LIB_DIR     "/usr/local/musa/lib"      CACHE PATH "MUSA SDK lib dir")

    target_compile_definitions(${target_name} INTERFACE TUTTI_USE_MUSA=1)

    target_include_directories(${target_name} INTERFACE
        ${MUSA_INCLUDE_DIR}
        "${TUTTI_SOURCE_DIR}/include"
    )

    # MUSA runtime + driver libraries (Mooncake-confirmed: musa musart rt)
    target_link_directories(${target_name} INTERFACE ${MUSA_LIB_DIR})
    target_link_libraries(${target_name} INTERFACE
        musa
        musart
    )

    # Expose for downstream CMakeLists that link ${TUTTI_ACCEL_RUNTIME_LIBS}
    set(TUTTI_ACCEL_RUNTIME_LIBS "musart" CACHE INTERNAL "MUSA runtime lib")
    set(TUTTI_ACCEL_DRIVER_LIBS  "musa"   CACHE INTERNAL "MUSA driver lib")

    message(STATUS "Tutti: MUSA profile — include=${MUSA_INCLUDE_DIR} lib=${MUSA_LIB_DIR}")
endfunction()

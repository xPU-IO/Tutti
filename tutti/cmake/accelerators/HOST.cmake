# tutti/cmake/accelerators/HOST.cmake -- HOST profile (hardware-free)
#
# Included by the root CMakeLists.txt when TUTTI_ACCELERATOR=HOST.
# Provides:
#   - TUTTI_BUILD_HARDWARE_STACK forced OFF
#   - tutti_configure_cuda_like() function (HOST variant)

# HOST profile never builds the hardware stack
if(DEFINED TUTTI_BUILD_HARDWARE_STACK AND TUTTI_BUILD_HARDWARE_STACK)
    message(STATUS
        "Tutti: HOST profile forces TUTTI_BUILD_HARDWARE_STACK=OFF "
        "(hardware stack not available without CUDA)")
endif()
set(TUTTI_BUILD_HARDWARE_STACK OFF CACHE BOOL
    "Build hardware stack (accel/device_manager/backends/io_engine)" FORCE)

# ---------------------------------------------------------------------------
# tutti_configure_cuda_like(<target_name>)
#
# Configures the given INTERFACE target with HOST usage requirements:
#   - TUTTI_USE_HOST=1 definition
#   - tutti/include directory
#   - No CUDA links
# ---------------------------------------------------------------------------
function(tutti_configure_cuda_like target_name)
    target_compile_definitions(${target_name} INTERFACE TUTTI_USE_HOST=1)

    target_include_directories(${target_name} INTERFACE
        "${TUTTI_SOURCE_DIR}/include"
    )
endfunction()

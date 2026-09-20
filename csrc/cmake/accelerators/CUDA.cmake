# tutti/cmake/accelerators/CUDA.cmake -- CUDA profile (NVIDIA-first)
#
# Included by the root CMakeLists.txt when TUTTI_ACCELERATOR=CUDA.
# Provides:
#   - TUTTI_BUILD_HARDWARE_STACK default ON
#   - tutti_configure_cuda_like() function

# CUDA profile defaults to full hardware stack
set(TUTTI_BUILD_HARDWARE_STACK ON CACHE BOOL
    "Build hardware stack (accel/device_manager/backends/io_engine)")

# Keep discovery at profile scope so CUDAToolkit_VERSION and include-path
# variables remain available to the rest of tutti/CMakeLists.txt.
find_package(CUDAToolkit 12.6 REQUIRED)

# ---------------------------------------------------------------------------
# tutti_configure_cuda_like(<target_name>)
#
# Configures the given INTERFACE target with CUDA usage requirements:
#   - TUTTI_USE_CUDA=1 definition
#   - tutti/include directory
#   - CUDA::cudart and CUDA::cuda_driver links
# ---------------------------------------------------------------------------
function(tutti_configure_cuda_like target_name)
    target_compile_definitions(${target_name} INTERFACE
        TUTTI_USE_CUDA=1
        TUTTI_COMPILED_ACCELERATOR_PROFILE=\"CUDA\"
        TUTTI_DEFAULT_ACCEL_ID=0
    )

    target_include_directories(${target_name} INTERFACE
        "${TUTTI_SOURCE_DIR}/include"
    )

    target_link_libraries(${target_name} INTERFACE
        CUDA::cudart
        CUDA::cuda_driver
    )
endfunction()

# Per-vendor runtime/driver libraries that downstream CMakeLists link via
# ${TUTTI_ACCEL_RUNTIME_LIBS} / ${TUTTI_ACCEL_DRIVER_LIBS}.  This decouples
# downstream targets from the vendor-specific target name (CUDA::cudart,
# MUSA equivalent, etc.) and lets MUSA/MACA profiles fill in their own libs
# without touching every CMakeLists.txt.
set(TUTTI_ACCEL_RUNTIME_LIBS "CUDA::cudart" CACHE INTERNAL "Vendor runtime lib")
set(TUTTI_ACCEL_DRIVER_LIBS  "CUDA::cuda_driver" CACHE INTERNAL "Vendor driver lib")

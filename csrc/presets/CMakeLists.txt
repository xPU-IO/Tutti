# tutti/presets/CMakeLists.txt
#
# Preset assembly layer — public API factory functions that construct
# StorageRuntime with the appropriate DataPath + resolver injected.

# CUDA device linking — needed because the linked libraries
# (tutti_local_nvme_datapath etc.) contain .cu device code.
add_library(tutti_presets STATIC
    local_nvme_preset.cpp
    compute_kernels.cu
)

# Tell CMake to do CUDA device linking on this target.
set_target_properties(tutti_presets PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(tutti_presets PUBLIC
    tutti_local_nvme_datapath
    tutti_striped_local_nvme_datapath
    tutti_striped_file_resolver
)

target_link_libraries(tutti_presets PRIVATE
    ${TUTTI_ACCEL_RUNTIME_LIBS}
)

target_include_directories(tutti_presets PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."          # for private headers (tutti/data_paths/...)
)

# CCCL headers (cuda/atomic, cuda/std/...) — needed by storage_runtime.h
# includes.  CUDA 13.0+ keeps them under targets/<arch>/include/cccl/, which
# is NOT part of CUDAToolkit_INCLUDE_DIRS.  Derive the path from
# CUDAToolkit_INCLUDE_DIRS so it works whether CUDA lives at the standard
# /usr/local/cuda or a vendored location (e.g. third_pkgs/nvidia/cuda-toolkit).
foreach(_tutti_inc_dir ${CUDAToolkit_INCLUDE_DIRS})
    get_filename_component(_tutti_cuda_root "${_tutti_inc_dir}/.." ABSOLUTE)
    file(GLOB _tutti_cccl_dirs "${_tutti_cuda_root}/targets/*/include/cccl")
    if(_tutti_cccl_dirs)
        target_include_directories(tutti_presets PRIVATE ${_tutti_cccl_dirs})
    endif()
endforeach()

set_target_properties(tutti_presets PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_compile_options(tutti_presets PRIVATE
    -Wall -Wextra
)

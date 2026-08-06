# tutti/data_paths/striped_local_nvme/CMakeLists.txt
#
# Private DataPath target: StripedDataPath -- single-kernel fused submission
# across N NVMe devices. Only built when TUTTI_BUILD_HARDWARE_STACK is ON
# (requires CUDA + libnvm), same gating as tutti_local_nvme_datapath.
#
# Depends on tutti_local_nvme_datapath for shared device primitives
# (nvme_submit_primitives.cuh), device_target.{h,cu}, nvme_queue_group.{h,cu},
# and prp_builder.h -- reused verbatim, not duplicated.

add_library(tutti_striped_local_nvme_datapath
    striped_data_path.cpp
    striped_arena.cpp
    fused_submit_kernel.cu
)

set_target_properties(tutti_striped_local_nvme_datapath PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "tutti_striped_local_nvme_datapath"
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

# Workspace root for <tutti/data_paths/...> private headers.
target_include_directories(tutti_striped_local_nvme_datapath PRIVATE
    ${TUTTI_REPOSITORY_ROOT}
)

target_link_libraries(tutti_striped_local_nvme_datapath
    PUBLIC
        tutti_local_nvme_datapath
        tutti_striped_local_nvme_binding
        tutti_striped_file_resolver
        libnvm
    PRIVATE
        ${TUTTI_ACCEL_RUNTIME_LIBS}
)

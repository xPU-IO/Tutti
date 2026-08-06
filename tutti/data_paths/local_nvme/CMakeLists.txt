# tutti/data_paths/local_nvme/CMakeLists.txt
#
# Private DataPath target: LocalNvmeDataPath with CUDA + libnvm.
# Only built when TUTTI_BUILD_HARDWARE_STACK is ON (requires CUDA + libnvm).
#
# Consumers link tutti_local_nvme_datapath to obtain:
#   - the compiled DataPath implementation (local_nvme_data_path.cpp + io/*.cu)
#   - libnvm include path (PUBLIC: the header includes <nvm_ctrl.h> etc.)
#   - tutti_local_file_binding + tutti_local_file_resolver (PUBLIC: header deps)
#   - tutti_spi (transitively via binding/resolver)
#
# ${TUTTI_ACCEL_RUNTIME_LIBS} is PRIVATE: only .cpp/.cu sources need it; the header
# uses void* for CUDA types.
#
# Does NOT propagate to tutti_api / tutti_spi / install public headers.

add_library(tutti_local_nvme_datapath
    local_nvme_data_path.cpp
    metadata/metadata_arena.cpp
    metadata/prp_page_cache.cpp
    metadata/desc_pool.cpp
    metadata/prp_buf_pool.cpp
    io/nvme_queue_group.cu
    io/device_target.cu
    io/submit_one.cu
)

set_target_properties(tutti_local_nvme_datapath PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "tutti_local_nvme_datapath"
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

# Workspace root for <tutti/data_paths/...> private headers.
# PRIVATE: only this target's sources need it.
target_include_directories(tutti_local_nvme_datapath PRIVATE
    ${TUTTI_REPOSITORY_ROOT}
)

target_link_libraries(tutti_local_nvme_datapath
    PUBLIC
        tutti_local_file_binding
        tutti_local_file_resolver
        libnvm
    PRIVATE
        ${TUTTI_ACCEL_RUNTIME_LIBS}
)

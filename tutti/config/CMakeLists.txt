# tutti/config/CMakeLists.txt
#
# Round 20 S1 — application config loader.
#
# Split into two libraries:
#   tutti_config_parse — pure host (yaml-cpp only), no CUDA.
#                         Used by unit tests.
#   tutti_config       — full loader (constructs DataPaths + Runtime).
#                         Needs tutti_local_nvme_datapath (CUDA).

# Pure-host parse library (no CUDA, no DataPath).
add_library(tutti_config_parse STATIC
    tutti_config_parse.cpp
    tutti_config_legacy_nvme_parse.cpp
    storage/resource/memory_config_parse.cpp
    storage/resource/nvme_config_parse.cpp
    storage/resolver/local_file_config_parse.cpp
    storage/resolver/memfs_config_parse.cpp
    storage/resolver/striped_file_config_parse.cpp
    storage/datapath/local_nvme_config_parse.cpp
    storage/datapath/memfs_config_parse.cpp
    storage/datapath/striped_local_nvme_config_parse.cpp
    storage/backend/ext4_local_nvme_config_parse.cpp
    storage/backend/memfs_config_parse.cpp
    storage/backend/striped_local_nvme_config_parse.cpp
)

target_include_directories(tutti_config_parse PUBLIC
    "${TUTTI_REPOSITORY_ROOT}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(tutti_config_parse PUBLIC
    tutti_api
    ${TUTTI_YAML_CPP_TARGET}
)

set_target_properties(tutti_config_parse PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

function(tutti_add_config_loader)
    tutti_enable_nvme_resource_grpc()

    add_library(tutti_config STATIC
        "${TUTTI_SOURCE_DIR}/config/tutti_config.cpp"
        "${TUTTI_SOURCE_DIR}/tutti_runtime/tutti_runtime.cpp"
    )

    target_include_directories(tutti_config PUBLIC
        "${TUTTI_REPOSITORY_ROOT}"
        "${TUTTI_SOURCE_DIR}/config"
    )

    target_link_libraries(tutti_config PUBLIC
        tutti_config_parse
        tutti_local_nvme_datapath
        tutti_striped_local_nvme_datapath
        tutti_local_file_resolver
        tutti_striped_file_resolver
        tutti_nvme_resource
        tutti_api
    )

    set_target_properties(tutti_config PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endfunction()

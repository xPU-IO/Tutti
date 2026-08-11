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
    add_library(tutti_config STATIC
        "${TUTTI_SOURCE_DIR}/config/tutti_config.cpp"
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
        tutti_api
    )

    if(TARGET nvmeservice)
        target_link_libraries(tutti_config PUBLIC nvmeservice)
        target_compile_definitions(tutti_config PRIVATE
            TUTTI_CONFIG_HAS_NVMESERVICE=1
        )
    endif()

    set_target_properties(tutti_config PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endfunction()

add_library(tutti_data_path_factory STATIC
    data_path_factory.cpp
)

target_include_directories(tutti_data_path_factory PUBLIC
    "${TUTTI_REPOSITORY_ROOT}"
)

target_link_libraries(tutti_data_path_factory PUBLIC
    tutti_runtime_spec
    tutti_resource
    tutti_spi
)

set_target_properties(tutti_data_path_factory PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

function(tutti_enable_data_path_factory_local_nvme)
    target_compile_definitions(tutti_data_path_factory PRIVATE
        TUTTI_DATA_PATH_FACTORY_HAS_LOCAL_NVME=1
    )
    target_link_libraries(tutti_data_path_factory PRIVATE
        tutti_local_nvme_datapath
        tutti_striped_local_nvme_datapath
    )
endfunction()

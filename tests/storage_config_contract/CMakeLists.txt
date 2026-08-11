cmake_minimum_required(VERSION 3.18)

add_executable(tutti_storage_config_contract_test
    storage_config_contract_test.cpp
)

target_link_libraries(tutti_storage_config_contract_test PRIVATE
    tutti_config_parse
    tutti_local_file_binding
    tutti_striped_local_nvme_binding
)

set_target_properties(tutti_storage_config_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_storage_config_contract_test PRIVATE -Wall -Wextra)

target_compile_definitions(tutti_storage_config_contract_test PRIVATE
    TUTTI_REPOSITORY_TUTTI_CONFIG="${TUTTI_REPOSITORY_ROOT}/config/tutti_config.yaml"
)

add_test(NAME tutti_storage_config_contract_test
         COMMAND tutti_storage_config_contract_test)

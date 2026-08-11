cmake_minimum_required(VERSION 3.18)

add_executable(tutti_runtime_resource_registry_contract_test
    runtime_resource_registry_contract_test.cpp
)

target_link_libraries(tutti_runtime_resource_registry_contract_test PRIVATE
    tutti_runtime
)

target_include_directories(tutti_runtime_resource_registry_contract_test PRIVATE
    ${TUTTI_REPOSITORY_ROOT}
)

set_target_properties(tutti_runtime_resource_registry_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_runtime_resource_registry_contract_test PRIVATE
    -Wall -Wextra
)

add_test(NAME tutti_runtime_resource_registry_contract_test
    COMMAND tutti_runtime_resource_registry_contract_test
)

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_resource_contract_test
    resource_contract_test.cpp
)

target_link_libraries(tutti_resource_contract_test PRIVATE
    tutti_resource
    tutti_nvme_resource
)

set_target_properties(tutti_resource_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_resource_contract_test PRIVATE -Wall -Wextra)

add_test(NAME tutti_resource_contract_test
         COMMAND tutti_resource_contract_test)

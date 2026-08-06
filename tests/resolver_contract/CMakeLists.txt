# tests/resolver_contract/CMakeLists.txt
#
# Hardware test: creates real files on NVMe mount, calls FS_IOC_FIEMAP.
# Needs TUTTI_BUILD_HARDWARE_TESTS=ON; labelled "hardware;resolver".

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_resolver_contract_test
    resolver_contract_test.cpp
)

target_link_libraries(tutti_resolver_contract_test PRIVATE
    tutti_local_file_resolver
)

set_target_properties(tutti_resolver_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_resolver_contract_test PRIVATE
    -Wall -Wextra
)

add_test(NAME tutti_resolver_contract_test
    COMMAND tutti_resolver_contract_test
)

set_tests_properties(tutti_resolver_contract_test PROPERTIES
    LABELS "hardware;resolver"
)

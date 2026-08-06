# tests/storage_target_resolver_contract/CMakeLists.txt
#
# Hardware-free: plain C++17, links only tutti_spi for SPI public headers.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_storage_target_resolver_contract_test
    storage_target_resolver_contract_test.cpp
)

target_link_libraries(tutti_storage_target_resolver_contract_test PRIVATE
    tutti_spi
)

set_target_properties(tutti_storage_target_resolver_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_storage_target_resolver_contract_test PRIVATE
    -Wall -Wextra
)

add_test(NAME tutti_storage_target_resolver_contract_test
    COMMAND tutti_storage_target_resolver_contract_test
)

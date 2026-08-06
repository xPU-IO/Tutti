# tests/memfs_sample_contract/CMakeLists.txt
#
# SAMPLE-ONLY contract test for the memfs extension.
# Hardware-free: plain C++17, links only tutti_spi + tutti_memfs_binding.

cmake_minimum_required(VERSION 3.18)

find_package(Threads REQUIRED)

add_executable(tutti_memfs_sample_contract_test
    memfs_sample_contract_test.cpp
)

target_link_libraries(tutti_memfs_sample_contract_test PRIVATE
    tutti_spi
    tutti_memfs_binding
    Threads::Threads
)

set_target_properties(tutti_memfs_sample_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_memfs_sample_contract_test PRIVATE
    -Wall -Wextra
)

add_test(NAME tutti_memfs_sample_contract_test
    COMMAND tutti_memfs_sample_contract_test
)

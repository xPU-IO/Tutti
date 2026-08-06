# tests/status_contract/CMakeLists.txt -- status.h contract test
#
# Hardware-free: plain C++17, links only tutti_api for public headers.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_status_contract_test
    status_contract_test.cpp
)

target_link_libraries(tutti_status_contract_test PRIVATE
    tutti_api
)

set_target_properties(tutti_status_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_status_contract_test PRIVATE -Wall -Wextra)

add_test(NAME tutti_status_contract_test COMMAND tutti_status_contract_test)

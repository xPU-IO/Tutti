# tests/payload_contract/CMakeLists.txt
#
# Hardware-free: plain C++17, links the ext4_local_nvme payload target.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_payload_contract_test
    payload_contract_test.cpp
)

target_link_libraries(tutti_payload_contract_test PRIVATE
    tutti_local_file_payload
)

set_target_properties(tutti_payload_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_payload_contract_test PRIVATE
    -Wall -Wextra
)

add_test(NAME tutti_payload_contract_test
    COMMAND tutti_payload_contract_test
)

# tests/local_nvme_datapath_contract/CMakeLists.txt
#
# Hardware test: needs GPU + snvme + libnvm.
# Gated by TUTTI_BUILD_HARDWARE_TESTS=ON; labelled "hardware;local_nvme".
#
# Links the local-NVMe DataPath and the LocalFileResolver used by this test.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_local_nvme_datapath_contract_test
    local_nvme_datapath_contract_test.cpp
)

target_link_libraries(tutti_local_nvme_datapath_contract_test PRIVATE
    tutti_local_nvme_datapath
)

set_target_properties(tutti_local_nvme_datapath_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_local_nvme_datapath_contract_test PRIVATE
    -Wall -Wextra
)

add_test(NAME tutti_local_nvme_datapath_contract_test
         COMMAND tutti_local_nvme_datapath_contract_test)

set_tests_properties(tutti_local_nvme_datapath_contract_test PROPERTIES
    LABELS "hardware;local_nvme"
)

message(STATUS "Configured tutti_local_nvme_datapath_contract_test (hardware;local_nvme)")

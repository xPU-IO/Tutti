# tests/striped_local_nvme_contract/CMakeLists.txt
#
# E2E striped StorageRuntime hardware contract test.
# Requires 2 or 4 NVMe devices + CUDA + snvme module + tutti_daemon.
# Test numbering starts at 82 (Round 15 S3's tests, not conflicting).

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_striped_local_nvme_contract_test
    striped_local_nvme_contract_test.cpp
    fill_helper.cu
)

target_link_libraries(tutti_striped_local_nvme_contract_test PRIVATE
    tutti_striped_local_nvme_datapath
    tutti_striped_file_resolver
    tutti_local_file_resolver
    tutti_local_file_binding
    tutti_spi
    libnvm
    Threads::Threads
)

set_target_properties(tutti_striped_local_nvme_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_striped_local_nvme_contract_test PRIVATE -Wall -Wextra)

add_test(NAME tutti_striped_local_nvme_contract_test
         COMMAND tutti_striped_local_nvme_contract_test)

set_tests_properties(tutti_striped_local_nvme_contract_test PROPERTIES
    LABELS "hardware;striped;runtime_e2e"
    SKIP_RETURN_CODE 77
)

message(STATUS "Configured tutti_striped_local_nvme_contract_test (hardware;striped;runtime_e2e)")

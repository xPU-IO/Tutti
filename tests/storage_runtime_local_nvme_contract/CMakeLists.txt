# tests/storage_runtime_local_nvme_contract/CMakeLists.txt
#
# Formal public-Runtime → LocalFileResolver → LocalNvmeDataPath E2E gate.
# Hardware test: needs GPU + snvme + libnvm + the NVMe mount.
# Gated by TUTTI_BUILD_HARDWARE_TESTS=ON (the subdir is only added then);
# labelled "hardware;local_nvme;runtime_e2e".
#
# The test links the private local-NVMe DataPath + resolver only to CONSTRUCT
# injected fixtures. All data-plane operations go through the public
# StorageRuntime API.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_storage_runtime_local_nvme_contract_test
    storage_runtime_local_nvme_contract_test.cpp
)

target_link_libraries(tutti_storage_runtime_local_nvme_contract_test PRIVATE
    tutti_local_nvme_datapath
)

set_target_properties(tutti_storage_runtime_local_nvme_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_storage_runtime_local_nvme_contract_test PRIVATE
    -Wall -Wextra
)

add_test(NAME tutti_storage_runtime_local_nvme_contract_test
         COMMAND tutti_storage_runtime_local_nvme_contract_test)

set_tests_properties(tutti_storage_runtime_local_nvme_contract_test PROPERTIES
    LABELS "hardware;local_nvme;runtime_e2e"
)

message(STATUS "Configured tutti_storage_runtime_local_nvme_contract_test (hardware;local_nvme;runtime_e2e)")

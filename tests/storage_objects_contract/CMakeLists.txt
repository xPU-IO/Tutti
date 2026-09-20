# tests/storage_objects_contract/CMakeLists.txt
#
# Contract test for the storage object layer's four units. Hardware-free: the
# only IO is an mmapped bitmap in a temporary directory. No CUDA, no NVMe.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_storage_objects_contract_test
    storage_objects_contract_test.cpp
)

target_link_libraries(tutti_storage_objects_contract_test PRIVATE
    tutti_storage_objects
)

set_target_properties(tutti_storage_objects_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_storage_objects_contract_test PRIVATE
    -Wall -Wextra
)

# Scratch files default to the build tree rather than /tmp: a Tutti host
# usually has a small root filesystem and large data mounts. TMPDIR still
# overrides at runtime.
target_compile_definitions(tutti_storage_objects_contract_test PRIVATE
    TUTTI_TEST_TMPDIR_DEFAULT="${CMAKE_BINARY_DIR}"
)

add_test(NAME tutti_storage_objects_contract_test
         COMMAND tutti_storage_objects_contract_test)

# tests/storage_object_store_e2e/CMakeLists.txt
#
# End-to-end test of ObjectStoreCore against a stub resolver. First test that
# exercises the layer as one object: reserve -> write payload -> commit ->
# restart -> recover, plus crash and corruption scenarios.
#
# Needs a filesystem supporting O_DIRECT; skips loudly otherwise. No CUDA, no
# NVMe hardware -- extent mapping is the real resolver's job and is covered by
# the hardware contract tests.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_storage_object_store_e2e_test
    storage_object_store_e2e_test.cpp
)

target_link_libraries(tutti_storage_object_store_e2e_test PRIVATE
    tutti_storage_objects
)

set_target_properties(tutti_storage_object_store_e2e_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_storage_object_store_e2e_test PRIVATE
    -Wall -Wextra
)

target_compile_definitions(tutti_storage_object_store_e2e_test PRIVATE
    TUTTI_TEST_TMPDIR_DEFAULT="${CMAKE_BINARY_DIR}"
)

add_test(NAME tutti_storage_object_store_e2e_test
         COMMAND tutti_storage_object_store_e2e_test)

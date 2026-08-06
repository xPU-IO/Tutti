# tests/header_hygiene/CMakeLists.txt -- Phase 3 header-hygiene consumer test
#
# Links ONLY tutti_api.  No additional include paths, no additional compile
# definitions.  The test TU uses __has_include to assert that local-NVMe
# private headers (libnvm, nvmeservice, snvme kernel, CUDA kernel) are NOT
# reachable, while public headers ARE reachable and usable.
#
# This is a hardware-free, compile+run test.  It is part of the default
# BUILD_TESTING suite.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_header_hygiene_test
    header_hygiene_test.cpp
)

# Link ONLY tutti_api — the public usage-requirements target.
# No other include paths or libraries are added.
target_link_libraries(tutti_header_hygiene_test PRIVATE
    tutti_api
)

set_target_properties(tutti_header_hygiene_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_header_hygiene_test PRIVATE -Wall -Wextra)

add_test(NAME tutti_header_hygiene_test COMMAND tutti_header_hygiene_test)

message(STATUS "Configured tutti_header_hygiene_test (Phase 3 header-hygiene consumer test)")

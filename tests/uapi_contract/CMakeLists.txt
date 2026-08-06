cmake_minimum_required(VERSION 3.18)

# Hardware-free UAPI contract test for tutti_snvme.h.
#
# This test verifies:
#   1. The shared UAPI header compiles cleanly from userspace C++.
#   2. static_asserts in the header lock struct sizes and field offsets
#      (compilation failure = test failure).
#   3. ABI version constant is the unique source of truth and has the
#      expected value.
#   4. Capability bits are non-overlapping and CAP_ALL includes all.
#   5. Struct sizes and field offsets match the documented LP64 layout.
#
# The test does NOT open /dev/ssnvme*, load kernel modules, or touch
# hardware.  It is a pure compile + constant/sizeof/offsetof check.

add_executable(tutti_uapi_contract_test
    uapi_contract_test.cpp
)

# The UAPI header lives at tutti/include/uapi/tutti_snvme.h.
# We add the path directly rather than linking libnvm (which pulls in
# CUDA dependencies) to keep this test truly hardware-free.
target_include_directories(tutti_uapi_contract_test PRIVATE
    ${TUTTI_SOURCE_DIR}/include/uapi
)

set_target_properties(tutti_uapi_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_uapi_contract_test PRIVATE -Wall -Wextra)

add_test(NAME tutti_uapi_contract_test COMMAND tutti_uapi_contract_test)

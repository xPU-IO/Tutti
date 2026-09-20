# csrc/payloads/memfs/CMakeLists.txt
#
# SAMPLE-ONLY payload package: memfs resolver + payload + DataPath.
#
# This is a community-extension sample (see doc/extending_tutti.md).
# It is NOT a production backend.  The entire sample is header-only and
# hardware-free: no CUDA, no libnvm, no kernel headers.
#
# Registration: the `add_subdirectory(payloads/memfs)` line in
# csrc/CMakeLists.txt enables both the library target and (under
# BUILD_TESTING) the contract test.  No other core CMake change is
# required.

# ---------------------------------------------------------------------------
# Header-only INTERFACE library
# ---------------------------------------------------------------------------

add_library(tutti_memfs_payload INTERFACE)

# Workspace root so that "csrc/payloads/memfs/*.h" and
# "csrc/resolvers/memfs/*.h" resolve.
target_include_directories(tutti_memfs_payload INTERFACE
    $<BUILD_INTERFACE:${TUTTI_REPOSITORY_ROOT}>
)

target_link_libraries(tutti_memfs_payload INTERFACE
    tutti_spi
)

# ---------------------------------------------------------------------------
# Contract test (self-contained; no additional top-level registration)
# ---------------------------------------------------------------------------

if(BUILD_TESTING)
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/memfs_sample_contract"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_memfs_sample_contract"
    )
    message(STATUS "Tutti: added tests/memfs_sample_contract (sample extension)")
endif()

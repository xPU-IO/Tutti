# tests/register_perf/CMakeLists.txt
#
# R19 S3b REQUIRED 3 — registration throughput microbenchmark.
# Hardware test: needs GPU + snvme + libnvm + the NVMe mount.
# Gated by TUTTI_BUILD_HARDWARE_TESTS=ON; labelled "hardware;local_nvme;perf".

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_register_perf
    register_perf.cpp
)

target_link_libraries(tutti_register_perf PRIVATE
    tutti_local_nvme_datapath
)

set_target_properties(tutti_register_perf PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_register_perf PRIVATE
    -Wall -Wextra
)

# Not registered as a ctest (it's a perf microbenchmark, not pass/fail).
# Invoked manually: ./bin/tutti_register_perf [N]

message(STATUS "Configured tutti_register_perf (hardware;local_nvme;perf)")

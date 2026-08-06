# tests/batch_open_perf/CMakeLists.txt
#
# Round 19 S1 — batch open performance microbenchmark.
# Hardware test: needs GPU + snvme + libnvm + the NVMe mount.
# Gated by TUTTI_BUILD_HARDWARE_TESTS=ON; labelled "hardware;local_nvme;perf".

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_batch_open_perf
    batch_open_perf.cpp
)

target_link_libraries(tutti_batch_open_perf PRIVATE
    tutti_local_nvme_datapath
)

set_target_properties(tutti_batch_open_perf PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_batch_open_perf PRIVATE
    -Wall -Wextra
)

# Not registered as a ctest (it's a perf microbenchmark, not pass/fail).
# Invoked manually: ./bin/tutti_batch_open_perf [N]

message(STATUS "Configured tutti_batch_open_perf (hardware;local_nvme;perf)")

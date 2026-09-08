# tutti/examples/layerwise_kv_overlap/CMakeLists.txt
#
# Layerwise KV-cache overlap workload -- TuttiRuntime port.
#
# HY3-shaped 128K-context request (80 layers, 512×256-token chunks,
# 90% prefix hit) with 3-stream layerwise pipeline:
#   read(L+1) ∥ SGEMM compute(L) ∥ write(L-1)
#
# The runtime is assembled from a Tutti YAML (--config); the user only passes
# daemon-published accelerator view directories (--directory). The default
# config below matches the 4-device striped YAML shipped next to this file.
#
# Manual hardware workload: needs GPU + snvme + libnvm + a running tutti_daemon.
# Gated by TUTTI_BUILD_HARDWARE_TESTS + TUTTI_FEATURE_LOCAL_NVME. It is not a
# CTest case because each invocation needs daemon-published --directory paths.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_layerwise_kv_overlap
    layerwise_kv_overlap.cpp
)

target_link_libraries(tutti_layerwise_kv_overlap PRIVATE
    tutti_runtime
    tutti_presets
    ${TUTTI_ACCEL_RUNTIME_LIBS}
)

target_compile_definitions(tutti_layerwise_kv_overlap PRIVATE
    TUTTI_LAYERWISE_DEFAULT_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/tutti_layerwise_striped.yaml"
)

set_target_properties(tutti_layerwise_kv_overlap PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_layerwise_kv_overlap PRIVATE
    -Wall -Wextra
)

message(STATUS "Configured tutti_layerwise_kv_overlap (manual hardware workload)")

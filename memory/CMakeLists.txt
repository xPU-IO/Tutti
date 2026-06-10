# memory/ -- Tutti memory subsystem (v0.1)
#
# Builds: libtutti_memory.so + memory_smoke
#
# - libtutti_memory: HostDeviceMemorySubsystem (CUDA + libnvm-dependent)
# - memory_smoke   : end-to-end smoke test against real libnvm + GPU
#
# Depends on:
#   libnvm  -- nvm_dma_map_data_*, nvm_controller_init_b3
#   CUDA    -- cudart

set(TUTTI_MEMORY_SOURCES
    src/host_device_memory_subsystem.cu
)

add_library(tutti_memory ${TUTTI_MEMORY_SOURCES})

target_include_directories(tutti_memory
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${PROJECT_SOURCE_DIR}/backends/local/nvme/libnvm/include"
)

target_link_libraries(tutti_memory
    PUBLIC
        libnvm
        CUDA::cudart
)

set_target_properties(tutti_memory PROPERTIES
    OUTPUT_NAME "tutti_memory"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH   "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

add_executable(memory_smoke
    test/memory_smoke.cu
)

target_link_libraries(memory_smoke
    PRIVATE
        tutti_memory
        tutti_device_manager
        libnvm
        CUDA::cudart
)

set_target_properties(memory_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH   "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

install(TARGETS tutti_memory DESTINATION "lib")
install(TARGETS memory_smoke DESTINATION "bin")
install(DIRECTORY include/ DESTINATION "include/tutti/memory"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")

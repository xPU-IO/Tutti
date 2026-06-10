# block_storage/ -- Tutti GpuFile (block) layer (R6).
#
# R6.1 shipped the interface + a pure resolve helper.
# R6.2 lands HostFsBackedBlockStorage + PersistentGpuFileLog +
#       a host-side end-to-end smoke.
# R6.3 lands the GPU acquire/release path + a GPU device-side
#       submit smoke (one kernel-thread-per-IO across shards).
#
# Builds:
#   tutti_block_storage              -- the runtime lib (static-style)
#   block_storage_resolve_test       -- R6.1 unit test for gpu_file_resolve
#   block_storage_smoke              -- R6.2 host-side end-to-end
#   block_storage_gpu_smoke          -- R6.3 GPU device-side end-to-end
#
# Layer dependency:
#   block_storage  -> nvme_storage   (INvmeStorage / NvmeFile)
#                  -> device_manager (Device)
#                  -> stdc++fs       (std::filesystem)
#                  -> pthread        (std::mutex)

set(TUTTI_BLOCK_STORAGE_SOURCES
    src/persistent_gpu_file_log.cpp
    src/host_fs_backed_block_storage.cpp
)

add_library(tutti_block_storage ${TUTTI_BLOCK_STORAGE_SOURCES})

target_include_directories(tutti_block_storage
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(tutti_block_storage
    PUBLIC
        tutti_nvme_storage          # INvmeStorage, NvmeFile
        tutti_device_manager        # Device
)

set_target_properties(tutti_block_storage PROPERTIES
    OUTPUT_NAME              "tutti_block_storage"
    BUILD_RPATH              "$ORIGIN"
    INSTALL_RPATH            "$ORIGIN"
    POSITION_INDEPENDENT_CODE ON
)

# ---------------------------------------------------------------------------
# R6.1 unit test: gpu_file_resolve correctness
# ---------------------------------------------------------------------------

add_executable(block_storage_resolve_test
    test/gpu_file_resolve_test.cpp
)

target_link_libraries(block_storage_resolve_test
    PRIVATE
        tutti_block_storage
)

set_target_properties(block_storage_resolve_test PROPERTIES
    BUILD_RPATH   "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

# ---------------------------------------------------------------------------
# R6.2 host-side end-to-end smoke
# ---------------------------------------------------------------------------

add_executable(block_storage_smoke
    test/block_storage_smoke.cu
)

target_link_libraries(block_storage_smoke
    PRIVATE
        tutti_block_storage
        tutti_nvme_storage
        tutti_device_manager
        libnvm
        CUDA::cudart
)

set_target_properties(block_storage_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH   "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

# ---------------------------------------------------------------------------
# R6.3 GPU device-side end-to-end smoke
#
# Verifies HostFsBackedBlockStorage::acquire_device_handle drives a
# GpuFile through R5b's submit_read_one with one kernel thread per
# IO and per-shard nvm_dma_map_data_device buffers.
# ---------------------------------------------------------------------------

add_executable(block_storage_gpu_smoke
    test/block_storage_gpu_smoke.cu
)

target_link_libraries(block_storage_gpu_smoke
    PRIVATE
        tutti_block_storage
        tutti_nvme_storage
        tutti_device_manager
        libnvm
        CUDA::cudart
)

set_target_properties(block_storage_gpu_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH   "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

# ---------------------------------------------------------------------------
# R6.4 prototype: 4-shard 8 GiB GpuFile bandwidth + IO/compute overlap
#
# Demonstrates aggregate read bandwidth across 4 NVMe drives through
# a single GpuFile, and verifies that an IO kernel and a compute
# kernel on separate streams overlap.  The smoke prototypes a small
# slice of R7's PRP-list build because tensor_size = 128 KiB > 2
# pages; that prototype goes away when memory/ lands.
# ---------------------------------------------------------------------------

add_executable(block_storage_gpu_bw_smoke
    test/block_storage_gpu_bw_smoke.cu
)

target_link_libraries(block_storage_gpu_bw_smoke
    PRIVATE
        tutti_block_storage
        tutti_nvme_storage
        tutti_device_manager
        libnvm
        CUDA::cudart
)

set_target_properties(block_storage_gpu_bw_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH   "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

install(TARGETS tutti_block_storage         DESTINATION "lib")
install(TARGETS block_storage_resolve_test  DESTINATION "bin")
install(TARGETS block_storage_smoke         DESTINATION "bin")
install(TARGETS block_storage_gpu_smoke     DESTINATION "bin")
install(TARGETS block_storage_gpu_bw_smoke  DESTINATION "bin")
install(DIRECTORY include/ DESTINATION "include/tutti/block_storage"
        FILES_MATCHING PATTERN "*.h")

# nvme_storage/ -- Tutti named-LBA-range storage layer (R5a + R5b)
#
# Builds:
#   libtutti_nvme_storage  -- the static-style runtime lib
#   nvme_storage_smoke     -- R5a host-side end-to-end smoke
#   nvme_storage_gpu_smoke -- R5b GPU device-side submit smoke
#
# - libtutti_nvme_storage: HostFsBackedNvmeStorage (mounts an ext4
#   filesystem on /dev/snvmeNn1, manages a flat directory of named
#   files via PersistentFileLog, exposes host-side blocking IO,
#   plus R5b acquire_device_handle / release_device_handle for GPU submit).
# - nvme_storage_smoke    : R5a end-to-end (1 + multi NVMe), CPU-side.
# - nvme_storage_gpu_smoke: R5b end-to-end (1 + multi NVMe), GPU-side
#                           submit_read_one verifying byte content.
#
# Depends on:
#   tutti_device_manager  -- for LocalNvmeDevice payload + the two
#                            registry impls used in smokes.
#   libnvm                -- pulled by R5b for ctrl.h Controller wrap +
#                            on-GPU submit helpers.
#   CUDA::cudart          -- R5b cudaMalloc / cudaMemcpy / cudaFree for
#                            NvmeFileDeviceHandle lifecycle.
#   stdc++fs              -- std::filesystem (older toolchains).
#   pthread               -- std::mutex.

set(TUTTI_NVME_STORAGE_SOURCES
    src/fiemap_helper.cpp
    src/persistent_file_log.cpp
    src/host_fs_backed_nvme_storage.cpp
    src/host_fs_backed_nvme_storage_device.cu     # R5b GPU bring-up
)

add_library(tutti_nvme_storage ${TUTTI_NVME_STORAGE_SOURCES})

target_include_directories(tutti_nvme_storage
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(tutti_nvme_storage
    PUBLIC
        tutti_device_manager
    PRIVATE
        libnvm                  # R5b: Controller (ctrl.h) for d_qps access
        CUDA::cudart            # R5b: cudaMalloc / cudaMemcpy / cudaFree
)

set_target_properties(tutti_nvme_storage PROPERTIES
    OUTPUT_NAME              "tutti_nvme_storage"
    BUILD_RPATH              "$ORIGIN"
    INSTALL_RPATH            "$ORIGIN"
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ---------------------------------------------------------------------------
# Smoke (R5a, host-side)
# ---------------------------------------------------------------------------

add_executable(nvme_storage_smoke
    test/nvme_storage_smoke.cu
)

target_link_libraries(nvme_storage_smoke
    PRIVATE
        tutti_nvme_storage
        tutti_device_manager
        libnvm
        CUDA::cudart
)

set_target_properties(nvme_storage_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH   "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

# ---------------------------------------------------------------------------
# Smoke (R5b, GPU device-side submit)
# ---------------------------------------------------------------------------

add_executable(nvme_storage_gpu_smoke
    test/nvme_storage_gpu_smoke.cu
)

target_link_libraries(nvme_storage_gpu_smoke
    PRIVATE
        tutti_nvme_storage
        tutti_device_manager
        libnvm
        CUDA::cudart
)

set_target_properties(nvme_storage_gpu_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH   "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

# ---------------------------------------------------------------------------
# Smoke (R5a.1, bulk-init perf + recovery)
# ---------------------------------------------------------------------------

add_executable(nvme_storage_bulk_smoke
    test/nvme_storage_bulk_smoke.cu
)

target_link_libraries(nvme_storage_bulk_smoke
    PRIVATE
        tutti_nvme_storage
        tutti_device_manager
        libnvm
        CUDA::cudart
)

set_target_properties(nvme_storage_bulk_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH   "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

install(TARGETS tutti_nvme_storage      DESTINATION "lib")
install(TARGETS nvme_storage_smoke      DESTINATION "bin")
install(TARGETS nvme_storage_gpu_smoke  DESTINATION "bin")
install(TARGETS nvme_storage_bulk_smoke DESTINATION "bin")
install(DIRECTORY include/ DESTINATION "include/tutti/nvme_storage"
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY include/ DESTINATION "include/tutti/nvme_storage"
        FILES_MATCHING PATTERN "*.cuh")

# tutti/CMakeLists.txt -- Tutti target graph
#
# The active graph is organized around public API/SPI contracts, accelerator
# profiles, target resolvers, bindings, and DataPath packages. The older
# L0-L6 directories remain behind TUTTI_BUILD_HARDWARE_STACK as migration
# references; they do not define the target architecture.
#
# Target baseline: Linux 5.15 and CUDA 12.6+ for the CUDA profile.

# This target graph is not a standalone CMake entry.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR
        "Configure Tutti from the repository root: "
        "cmake -S <repository-root> -B <build-dir>")
endif()

# Stable paths for every child CMake file.  This file is entered only through
# the repository-root CMakeLists.txt and intentionally does not call project().
get_filename_component(TUTTI_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
get_filename_component(TUTTI_REPOSITORY_ROOT "${TUTTI_SOURCE_DIR}/.." ABSOLUTE)

# ---------------------------------------------------------------------------
# Global settings (inherited from parent if included as subdirectory)
# ---------------------------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA architecture (default to Ada/Hopper, override with -DCMAKE_CUDA_ARCHITECTURES)
if(TUTTI_ACCELERATOR STREQUAL "CUDA")
    set(CMAKE_CUDA_STANDARD 17)
endif()

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------

# C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

# CUDA flags
if(TUTTI_ACCELERATOR STREQUAL "CUDA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr --expt-extended-lambda -rdc=true")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -DDEBUG")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O2")

    # Generate code for specified architectures
    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Feature switches (Phase 6)
#
# Each feature can be turned OFF to exclude its package sources and
# dependencies from configure/build.  Default ON preserves existing
# behavior.  When OFF, the corresponding subdirectories, find_package
# calls, and tests are skipped entirely — zero dependency participation.
#
# TUTTI_FEATURE_LOCAL_NVME
#   ON  (default): libnvm, nvmeservice, LocalNvmeDataPath, backends,
#                  io_engine, and all associated tests participate.
#   OFF:           The above are excluded.  accel (CUDA HAL) still
#                  builds if TUTTI_BUILD_HARDWARE_STACK is ON; only
#                  local-NVMe-specific packages exit.
#
# TUTTI_FEATURE_MEMFS_SAMPLE
#   ON  (default): memfs sample binding + its contract test participate.
#   OFF:           Sample is excluded (community contributor may disable).
# ---------------------------------------------------------------------------

option(TUTTI_FEATURE_LOCAL_NVME "Include local-NVMe DataPath and dependencies" ON)
option(TUTTI_FEATURE_MEMFS_SAMPLE "Include memfs sample extension" ON)
option(TUTTI_BUILD_KERNEL_MODULE
    "Configure snvme kernel-module build/load targets" OFF)

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
# The global include_directories for the project root (tutti/) was previously
# unconditional, which polluted Phase 0 contract targets (tutti_cuda_like,
# tutti_api, tutti_spi, tutti_types) and their consumers with the full tutti/
# tree.  It has been moved inside the TUTTI_BUILD_HARDWARE_STACK guard below
# so that only hardware-stack targets receive it.  Each hardware-stack layer
# and test already declares its own target_include_directories with
# $<BUILD_INTERFACE:${TUTTI_SOURCE_DIR}>, so the global directive is a
# safety net for any edge case not yet audited.

# ---------------------------------------------------------------------------
# Accelerator profile module
# ---------------------------------------------------------------------------

include("${TUTTI_SOURCE_DIR}/cmake/accelerators/${TUTTI_ACCELERATOR}.cmake")

if(TUTTI_BUILD_KERNEL_MODULE)
    if(TUTTI_ACCELERATOR STREQUAL "HOST")
        message(FATAL_ERROR
            "TUTTI_BUILD_KERNEL_MODULE requires a GPU P2P backend; "
            "use the CUDA, MUSA, or MACA accelerator profile")
    elseif(NOT TUTTI_BUILD_HARDWARE_STACK)
        message(FATAL_ERROR
            "TUTTI_BUILD_KERNEL_MODULE requires TUTTI_BUILD_HARDWARE_STACK=ON")
    elseif(NOT TUTTI_FEATURE_LOCAL_NVME)
        message(FATAL_ERROR
            "TUTTI_BUILD_KERNEL_MODULE requires TUTTI_FEATURE_LOCAL_NVME=ON")
    endif()
endif()

# ---------------------------------------------------------------------------
# tutti_cuda_like -- profile interface library
# ---------------------------------------------------------------------------

add_library(tutti_cuda_like INTERFACE)
tutti_configure_cuda_like(tutti_cuda_like)

# ---------------------------------------------------------------------------
# tutti_api -- public usage-requirements target
#
# Consumers link only tutti_api to obtain: public headers (tutti/include),
# profile definition (TUTTI_USE_<PROFILE>), and CUDA-like vendor requirements.
# No source files; no direct CUDA links — all inherited via tutti_cuda_like.
# ---------------------------------------------------------------------------

add_library(tutti_api INTERFACE)
target_include_directories(tutti_api INTERFACE
    $<BUILD_INTERFACE:${TUTTI_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tutti_api INTERFACE tutti_cuda_like)

# ---------------------------------------------------------------------------
# tutti_spi -- in-repo SPI usage-requirements target
#
# Consumers link only tutti_spi to obtain: the public include root
# (tutti/include, which now also hosts <tutti/spi/...>), and -- transitively
# via tutti_api -> tutti_cuda_like -- the TUTTI_USE_<PROFILE> macro. No source
# files; no direct CUDA links; no repeated profile definition. Not gated by
# TUTTI_BUILD_HARDWARE_STACK: the SPI headers are hardware-free source-level
# contracts available in every profile, like tutti_api. After the SPI-header
# migration the build-tree and install-tree layouts match (single include
# root), so tutti_spi and tutti_api share the same include interface; tutti_spi
# is retained as the explicit "I consume the in-repo SPI" declaration point
# and as a named member of the tutti_targets export set.
# ---------------------------------------------------------------------------

add_library(tutti_spi INTERFACE)
target_include_directories(tutti_spi INTERFACE
    $<BUILD_INTERFACE:${TUTTI_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tutti_spi INTERFACE tutti_api)

# ---------------------------------------------------------------------------
# Private package targets (new architecture)
#
# These are hardware-free header-only targets that provide the binding
# payload contract and the file resolver.  They do NOT propagate to
# tutti_api / tutti_spi / install public headers.
# ---------------------------------------------------------------------------

add_subdirectory(bindings/ext4_local_nvme)
add_subdirectory(resolvers/local_file)
add_subdirectory(bindings/striped_local_nvme)
add_subdirectory(resolvers/striped_file)

# SAMPLE-ONLY: community extension sample (doc/extending_tutti.md).
# The subdirectory is added inside BUILD_TESTING (below) because its
# contract test uses add_test(), which requires include(CTest) first.
# Gated by TUTTI_FEATURE_MEMFS_SAMPLE.

# ---------------------------------------------------------------------------
# Layer 0: Abstraction (header-only, no target)
# ---------------------------------------------------------------------------

# abstraction/ provides macros (TUTTI_DEVICE, TUTTI_LAUNCH_KERNEL, etc.)
# No library target needed - headers included directly

# ---------------------------------------------------------------------------
# Layer 0: Types (header-only, interface library)
# ---------------------------------------------------------------------------

add_library(tutti_types INTERFACE)
target_include_directories(tutti_types INTERFACE
    "${TUTTI_SOURCE_DIR}"
)
target_link_libraries(tutti_types INTERFACE tutti_cuda_like)

# ---------------------------------------------------------------------------
# Hardware stack (only when TUTTI_BUILD_HARDWARE_STACK is ON)
# ---------------------------------------------------------------------------

if(TUTTI_BUILD_HARDWARE_STACK)
    # Make tutti/ headers available to all hardware-stack targets.
    # Phase 0 contract targets (built in all profiles) do NOT receive this.
    include_directories("${TUTTI_SOURCE_DIR}")

    # Preset assembly layer (public API factory functions).
    add_subdirectory(presets)

    # The CUDA profile requires CUDAToolkit (including its minimum version)
    # in cmake/accelerators/CUDA.cmake. MUSA/MACA profiles supply their own
    # SDK through their profile modules.
    find_package(Threads REQUIRED)

    # gRPC: optional; when found, nvmeservice (daemon NVMe multi-process path) is
    # enabled and TUTTI_NVMESERVICE_ENABLED is defined on all targets that need it.
    # gRPC and yaml-cpp are local-NVMe-specific dependencies (nvmeservice config).
    # When TUTTI_FEATURE_LOCAL_NVME=OFF they are not looked up.
    if(TUTTI_FEATURE_LOCAL_NVME)
        find_package(gRPC CONFIG QUIET)
        if(gRPC_FOUND)
            message(STATUS "Tutti: gRPC found -- nvmeservice (daemon NVMe) enabled")
        else()
            message(STATUS "Tutti: gRPC NOT found -- daemon NVMe path disabled")
        endif()

        # yaml-cpp: used by Layer 2 (device_manager/nvme/nvmeservice) config parser.
        # Normalize the target name across packagings:
        #   vcpkg / new upstream : yaml-cpp::yaml-cpp
        #   old system pkg (0.5.x): bare "yaml-cpp" target, or only ${YAML_CPP_LIBRARIES}
        find_package(yaml-cpp REQUIRED)
        if(TARGET yaml-cpp::yaml-cpp)
            set(TUTTI_YAML_CPP_TARGET yaml-cpp::yaml-cpp)
        elseif(TARGET yaml-cpp)
            set(TUTTI_YAML_CPP_TARGET yaml-cpp)
        else()
            set(TUTTI_YAML_CPP_TARGET ${YAML_CPP_LIBRARIES})
        endif()
        message(STATUS "Tutti: yaml-cpp target: ${TUTTI_YAML_CPP_TARGET}")
    else()
        message(STATUS "Tutti: TUTTI_FEATURE_LOCAL_NVME=OFF -- gRPC/yaml-cpp/libnvm skipped")
    endif()

    if(TUTTI_ACCELERATOR STREQUAL "CUDA")
        message(STATUS "Tutti: Using CUDA ${CUDAToolkit_VERSION}")
        message(STATUS "Tutti: Target architectures: ${CMAKE_CUDA_ARCHITECTURES}")

        # CUDA 13.0+: CCCL headers (cuda/atomic, cuda/std/...) live under
        # targets/<arch>/include/cccl/ which is NOT in CUDAToolkit_INCLUDE_DIRS.
        # Add it for hardware stack targets (libnvm queue.cpp includes <cuda/atomic>).
        # Only needed when local-NVMe (libnvm) is in the build.
        if(TUTTI_FEATURE_LOCAL_NVME)
            foreach(_tutti_inc_dir ${CUDAToolkit_INCLUDE_DIRS})
                get_filename_component(_tutti_cuda_root "${_tutti_inc_dir}/.." ABSOLUTE)
                file(GLOB _tutti_cccl_dirs "${_tutti_cuda_root}/targets/*/include/cccl")
                if(_tutti_cccl_dirs)
                    include_directories(${_tutti_cccl_dirs})
                    message(STATUS "Tutti: CCCL include: ${_tutti_cccl_dirs}")
                endif()
            endforeach()
        endif()
    elseif(TUTTI_ACCELERATOR STREQUAL "MUSA" OR TUTTI_ACCELERATOR STREQUAL "MACA")
        message(STATUS "Tutti: Using ${TUTTI_ACCELERATOR} profile "
                       "(SDK paths configured via cmake/accelerators/${TUTTI_ACCELERATOR}.cmake)")
    endif()

    # ---------------------------------------------------------------------------
    # Local-NVMe-specific layers (gated by TUTTI_FEATURE_LOCAL_NVME)
    #
    # When OFF: libnvm, nvmeservice and LocalNvmeDataPath (and all their
    # sources/dependencies) exit configure/build entirely.
    # ---------------------------------------------------------------------------

    if(TUTTI_FEATURE_LOCAL_NVME)
        add_subdirectory(device_manager)   # libnvm + nvmeservice

        if(TUTTI_BUILD_KERNEL_MODULE)
            include("${TUTTI_SOURCE_DIR}/cmake/SNVMeModule.cmake")
        endif()

        # LocalNvmeDataPath package — needs libnvm + binding + resolver,
        # all already defined upstream.
        add_subdirectory(data_paths/local_nvme)

        # StripedDataPath package (Round 15 S5): single-kernel fused
        # multi-device submission. Needs tutti_local_nvme_datapath (shared
        # device primitives) + the striped binding/resolver already defined
        # above (lines ~170-171).
        add_subdirectory(data_paths/striped_local_nvme)

        # Round 20 S1: application config loader (yaml → RuntimeComponents).
        # Translates config/tutti_config.yaml into DataPath/Resolver
        # construction via the public API.
        add_subdirectory(config)
    else()
        message(STATUS "Tutti: TUTTI_FEATURE_LOCAL_NVME=OFF -- skipping device_manager, data_paths/local_nvme")
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests
#
# Set -DBUILD_TESTING=ON to include the test suite in the build. The root
# CMakeLists.txt owns include(CTest) so CTest discovers this subdirectory.
# Default is OFF so a plain `cmake -S . -B build` stays lean.
# ---------------------------------------------------------------------------

# Hardware tests (resolver on real FIEMAP mount, local-NVMe on real GPU+snvme)
# are OFF by default.  When ON, they are registered with CTest labels
# "hardware;resolver" and "hardware;local_nvme" so they can be filtered out
# of default CI runs.
option(TUTTI_BUILD_HARDWARE_TESTS "Register hardware-dependent tests" OFF)

if(BUILD_TESTING)
    # SAMPLE-ONLY: community extension sample (doc/extending_tutti.md).
    # Must live inside this block: add_test() only registers in directories
    # processed after include(CTest) has enabled testing.
    # Gated by TUTTI_FEATURE_MEMFS_SAMPLE.
    if(TUTTI_FEATURE_MEMFS_SAMPLE)
        add_subdirectory(bindings/memfs)
    endif()

    # ---- Hardware-free contract tests (all profiles) ----

    # Contract test (runs in all profiles: CUDA and HOST)
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/cuda_like"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_cuda_like"
    )
    message(STATUS "Tutti: added tests/cuda_like (contract test)")

    # Public API usage test (runs in all profiles: CUDA and HOST)
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/public_api"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_public_api"
    )
    message(STATUS "Tutti: added tests/public_api (public API usage test)")

    # SPI usage-requirements consumer test (runs in all profiles: CUDA and HOST)
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/spi_consumer"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_spi_consumer"
    )
    message(STATUS "Tutti: added tests/spi_consumer (SPI usage-requirements test)")

    # Runtime and DataPath SPI contracts are hardware-free and must remain
    # runnable under the HOST profile. Their fake components prove Runtime
    # routing does not require CUDA, libnvm, or a local-NVMe deployment.
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/data_path_contract"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_data_path_contract"
    )
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/storage_runtime_contract"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_storage_runtime_contract"
    )
    message(STATUS "Tutti: added tests/data_path_contract and tests/storage_runtime_contract")

    # Additional hardware-free contract tests (status, memory_types,
    # io_types, storage_target_resolver, binding).
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/status_contract"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_status_contract"
    )
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/memory_types_contract"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_memory_types_contract"
    )
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/io_types_contract"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_io_types_contract"
    )
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/storage_target_resolver_contract"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_storage_target_resolver_contract"
    )
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/binding_contract"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_binding_contract"
    )
    message(STATUS "Tutti: added hardware-free contract tests (status, memory_types, io_types, storage_target_resolver, binding)")

    # UAPI contract test (hardware-free; verifies shared tutti_snvme.h layout,
    # ABI version, capability bits, and static_assert locks).
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/uapi_contract"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_uapi_contract"
    )
    message(STATUS "Tutti: added tests/uapi_contract (hardware-free UAPI contract test)")

    # Striped resolver contract test (hardware-free; uses mock shard resolvers).
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/striped_resolver"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_striped_resolver"
    )
    message(STATUS "Tutti: added tests/striped_resolver (hardware-free striped resolver contract test)")

    # Phase 3 header-hygiene consumer test (hardware-free).
    # Links ONLY tutti_api; uses __has_include to assert that local-NVMe
    # private headers (libnvm, nvmeservice, snvme kernel, CUDA kernel) are
    # NOT reachable from the public include path.
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/header_hygiene"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_header_hygiene"
    )
    message(STATUS "Tutti: added tests/header_hygiene (Phase 3 header-hygiene consumer test)")

    # MockDataPath kit contract test (hardware-free; proves the kit itself
    # satisfies the DataPath SPI as a reference implementation).
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/mock_data_path_kit_contract"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_mock_data_path_kit_contract"
    )
    message(STATUS "Tutti: added tests/mock_data_path_kit_contract (MockDataPath kit self-contract test)")

    # Round 17 S1: MountManager host-side unit tests (hardware-free).
    # Tests config parsing, /proc scanning, mount/unmount lifecycle
    # with tmpfs, and force-exit flag.  No real NVMe devices needed.
    add_subdirectory(
        "${TUTTI_REPOSITORY_ROOT}/tests/mount_manager_contract"
        "${CMAKE_CURRENT_BINARY_DIR}/tests_mount_manager_contract"
    )
    message(STATUS "Tutti: added tests/mount_manager_contract (host;mount_manager)")

    # ---- Hardware tests (gated by TUTTI_BUILD_HARDWARE_TESTS) ----

    if(TUTTI_BUILD_HARDWARE_TESTS)
        # Resolver test: creates real files on NVMe mount, calls FIEMAP.
        # Links tutti_local_file_resolver (always built, hardware-free header-only).
        # Needs TUTTI_BUILD_HARDWARE_TESTS=ON; labelled "hardware;resolver".
        add_subdirectory(
            "${TUTTI_REPOSITORY_ROOT}/tests/resolver_contract"
            "${CMAKE_CURRENT_BINARY_DIR}/tests_resolver_contract"
        )
        message(STATUS "Tutti: added tests/resolver_contract (hardware;resolver)")

        # Local-NVMe DataPath tests: link tutti_local_nvme_datapath which is
        # only defined when TUTTI_FEATURE_LOCAL_NVME=ON.  When OFF, these
        # tests are not registered (they don't exist, not fail).
        if(TUTTI_FEATURE_LOCAL_NVME)
            # Local-NVMe DataPath test: needs GPU + snvme + libnvm.
            # Labelled "hardware;local_nvme".
            add_subdirectory(
                "${TUTTI_REPOSITORY_ROOT}/tests/local_nvme_datapath_contract"
                "${CMAKE_CURRENT_BINARY_DIR}/tests_local_nvme_datapath_contract"
            )
            message(STATUS "Tutti: added tests/local_nvme_datapath_contract (hardware;local_nvme)")

            # Public-Runtime → resolver → local-NVMe E2E acceptance gate.
            # Labelled "hardware;local_nvme;runtime_e2e".
            add_subdirectory(
                "${TUTTI_REPOSITORY_ROOT}/tests/storage_runtime_local_nvme_contract"
                "${CMAKE_CURRENT_BINARY_DIR}/tests_storage_runtime_local_nvme_contract"
            )
            message(STATUS "Tutti: added tests/storage_runtime_local_nvme_contract (hardware;local_nvme;runtime_e2e)")

            # StripedDataPath test: needs 2 GPU-attached NVMe devices + libnvm.
            # Labelled "hardware;striped;runtime_e2e".
            add_subdirectory(
                "${TUTTI_REPOSITORY_ROOT}/tests/striped_local_nvme_contract"
                "${CMAKE_CURRENT_BINARY_DIR}/tests_striped_local_nvme_contract"
            )
            message(STATUS "Tutti: added tests/striped_local_nvme_contract (hardware;striped;runtime_e2e)")

            # Round 19 S1: batch open performance microbenchmark.
            # Not a ctest (perf number, not pass/fail); manually invoked.
            add_subdirectory(
                "${TUTTI_REPOSITORY_ROOT}/tests/batch_open_perf"
                "${CMAKE_CURRENT_BINARY_DIR}/tests_batch_open_perf"
            )

            # Round 20 S2: layerwise KV-cache overlap example (moved to root examples/).
            # HY3-shaped 3-stream pipeline: read(L+1) || compute(L) || write(L-1).
            # Uses ONLY public StorageRuntime API + preset factory.
            add_subdirectory(
                "${TUTTI_REPOSITORY_ROOT}/examples/layerwise_kv_overlap"
                "${CMAKE_CURRENT_BINARY_DIR}/examples_layerwise_kv_overlap"
            )

            # Round 19 S3b: registration throughput microbenchmark.
            # Not a ctest (perf number, not pass/fail); manually invoked.
            add_subdirectory(
                "${TUTTI_REPOSITORY_ROOT}/tests/register_perf"
                "${CMAKE_CURRENT_BINARY_DIR}/tests_register_perf"
            )

            # Round 20 S1: tutti_config loader unit test (ctest).
            add_subdirectory(
                "${TUTTI_REPOSITORY_ROOT}/tests/config_loader"
                "${CMAKE_CURRENT_BINARY_DIR}/tests_config_loader"
            )
        else()
            message(STATUS "Tutti: TUTTI_FEATURE_LOCAL_NVME=OFF -- skipping local_nvme hardware tests")
        endif()
    endif()

endif()

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

# Install cuda_like profile headers
install(DIRECTORY "${TUTTI_SOURCE_DIR}/include/tutti/" DESTINATION include/tutti
        FILES_MATCHING PATTERN "*.h")

# Install tutti_cuda_like into the existing export set
install(TARGETS tutti_cuda_like
    EXPORT tutti_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install tutti_api into the existing export set
install(TARGETS tutti_api
    EXPORT tutti_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install tutti_spi into the existing export set
install(TARGETS tutti_spi
    EXPORT tutti_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Layer-specific headers installed by their own CMakeLists.txt

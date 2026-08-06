# tutti/device_manager/nvme/libnvm/CMakeLists.txt
# libnvm: User-space NVMe queue library wrapping snvme kernel module
#
# Layer 2 component: wraps snvme ioctls, manages queue memory, GPU DMA

cmake_minimum_required(VERSION 3.18)

# =============================================================================
# Source Collection
# =============================================================================

file(GLOB LIBNVM_SOURCES
    src/*.cpp
    src/linux/*.cpp
)

file(GLOB LIBNVM_CUDA_SOURCES
    src/*.cu
)

file(GLOB LIBNVM_PUBLIC_HEADERS
    include/*.h
)

# =============================================================================
# Library Target
# =============================================================================

add_library(libnvm SHARED ${LIBNVM_SOURCES} ${LIBNVM_CUDA_SOURCES})

set_target_properties(libnvm PROPERTIES
    OUTPUT_NAME "nvm"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

# =============================================================================
# Include Directories
# =============================================================================

target_include_directories(libnvm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${TUTTI_SOURCE_DIR}/include/uapi>
        $<INSTALL_INTERFACE:include/libnvm>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# =============================================================================
# Dependencies
# =============================================================================

target_link_libraries(libnvm
    PUBLIC
        tutti_cuda_like
        ${TUTTI_ACCEL_RUNTIME_LIBS}
        Threads::Threads
)

# =============================================================================
# Compile Definitions
# =============================================================================

if(CUDA_FOUND)
    target_compile_definitions(libnvm PRIVATE _CUDA __CUDA__)
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS libnvm
    EXPORT tutti_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${LIBNVM_PUBLIC_HEADERS}
    DESTINATION include/libnvm
)

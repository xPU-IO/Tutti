# Tutti has one canonical CMake entry: the repository root.
cmake_minimum_required(VERSION 3.18)

set(TUTTI_ACCELERATOR "CUDA" CACHE STRING
    "Tutti accelerator profile: CUDA|HOST|MUSA|MACA")
set_property(CACHE TUTTI_ACCELERATOR PROPERTY STRINGS CUDA HOST MUSA MACA)
string(TOUPPER "${TUTTI_ACCELERATOR}" TUTTI_ACCELERATOR)
if(NOT TUTTI_ACCELERATOR MATCHES "^(CUDA|HOST|MUSA|MACA)$")
    message(FATAL_ERROR
        "TUTTI_ACCELERATOR='${TUTTI_ACCELERATOR}' is not supported. "
        "Supported values: CUDA, HOST, MUSA, MACA "
        "(MUSA/MACA are framework stubs; see doc/gpu-porting-guide.md)")
endif()

# CMake initializes this value while enabling CUDA in project(), so the
# project default must be established first.  Explicit -D values still win.
if(TUTTI_ACCELERATOR STREQUAL "CUDA" AND
   NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING
        "CUDA architectures for Tutti")
endif()

# On hosts without nvcc on PATH or a standard /usr/local/cuda install, the
# CUDA toolkit may be vendored under third_pkgs/nvidia/cuda-toolkit (the whole
# third_pkgs/ tree is gitignored).  Prefer the standard location (CMake finds
# it natively); only fall back to the vendored toolkit when nvcc is otherwise
# undiscoverable, so the build works out-of-the-box without per-developer
# PATH/CUDAToolkit_ROOT setup.  Explicit -DCMAKE_CUDA_COMPILER still wins.
if(NOT TUTTI_ACCELERATOR STREQUAL "HOST" AND NOT CMAKE_CUDA_COMPILER)
    find_program(_tutti_nvcc nvcc)
    if(NOT _tutti_nvcc)
        set(_tutti_vendored_cuda
            "${CMAKE_CURRENT_SOURCE_DIR}/third_pkgs/nvidia/cuda-toolkit")
        if(EXISTS "${_tutti_vendored_cuda}/bin/nvcc")
            set(CMAKE_CUDA_COMPILER "${_tutti_vendored_cuda}/bin/nvcc"
                CACHE FILEPATH "CUDA compiler" FORCE)
            set(CUDAToolkit_ROOT "${_tutti_vendored_cuda}"
                CACHE PATH "CUDA toolkit root" FORCE)
        endif()
    endif()
endif()

if(TUTTI_ACCELERATOR STREQUAL "HOST")
    project(Tutti VERSION 0.1.1 LANGUAGES C CXX)
else()
    # CUDA is currently also the compiler integration used by the MUSA/MACA
    # framework stubs.  Their profile modules provide vendor-specific SDKs.
    project(Tutti VERSION 0.1.1 LANGUAGES C CXX CUDA)
endif()

# CTest must be enabled at the repository root so `ctest --test-dir <build>`
# discovers tests registered by the tutti subdirectory.
option(BUILD_TESTING "Build Tutti tests" ON)
if(BUILD_TESTING)
    include(CTest)
endif()

# The root owns project()/language initialization.  The target graph is kept
# in tutti/CMakeLists.txt as a regular subdirectory (not a standalone entry).

# GCC 8 keeps std::filesystem in a separate libstdc++fs; GCC 9+ folds it into
# libstdc++.  Detect and expose so filesystem-using targets link it explicitly.
set(TUTTI_FILESYSTEM_LIBS "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    set(TUTTI_FILESYSTEM_LIBS stdc++fs)
endif()

add_subdirectory(tutti)

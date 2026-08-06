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

if(TUTTI_ACCELERATOR STREQUAL "HOST")
    project(Tutti VERSION 0.1.1 LANGUAGES C CXX)
else()
    # CUDA is currently also the compiler integration used by the MUSA/MACA
    # framework stubs.  Their profile modules provide vendor-specific SDKs.
    project(Tutti VERSION 0.1.1 LANGUAGES C CXX CUDA)
endif()

# CTest must be enabled at the repository root so `ctest --test-dir <build>`
# discovers tests registered by the tutti subdirectory.
option(BUILD_TESTING "Build Tutti tests" OFF)
if(BUILD_TESTING)
    include(CTest)
endif()

# The root owns project()/language initialization.  The target graph is kept
# in tutti/CMakeLists.txt as a regular subdirectory (not a standalone entry).
add_subdirectory(tutti)

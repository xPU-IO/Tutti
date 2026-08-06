# tutti/device_manager/nvme/nvmeservice/CMakeLists.txt
# NVMeService: Cross-process daemon providing Level-1 queue arbitration
#
# Layer 2 component: gRPC server + client library

cmake_minimum_required(VERSION 3.18)

# =============================================================================
# Dependencies
# =============================================================================

find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf REQUIRED)

# Determine protoc executable
if(TARGET protobuf::protoc)
    set(PROTOC_EXECUTABLE "$<TARGET_FILE:protobuf::protoc>")
elseif(Protobuf_PROTOC_EXECUTABLE)
    set(PROTOC_EXECUTABLE "${Protobuf_PROTOC_EXECUTABLE}")
else()
    message(FATAL_ERROR "protoc not found for nvmeservice")
endif()

# gRPC C++ plugin
get_target_property(GRPC_CPP_PLUGIN_LOCATION gRPC::grpc_cpp_plugin LOCATION)

# =============================================================================
# Protobuf/gRPC Code Generation
# =============================================================================

set(NVMESERVICE_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/src/nvmeservice.proto")
set(PROTO_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

file(MAKE_DIRECTORY "${PROTO_OUT_DIR}")

add_custom_command(
    OUTPUT
        "${PROTO_OUT_DIR}/nvmeservice.pb.cc"
        "${PROTO_OUT_DIR}/nvmeservice.pb.h"
        "${PROTO_OUT_DIR}/nvmeservice.grpc.pb.cc"
        "${PROTO_OUT_DIR}/nvmeservice.grpc.pb.h"
    COMMAND "${PROTOC_EXECUTABLE}"
        --cpp_out "${PROTO_OUT_DIR}"
        --grpc_out "${PROTO_OUT_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
        --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN_LOCATION}"
        "${NVMESERVICE_PROTO}"
    DEPENDS "${NVMESERVICE_PROTO}"
    COMMENT "Generating protobuf/gRPC code for nvmeservice"
)

set(NVMESERVICE_PROTO_SRCS
    "${PROTO_OUT_DIR}/nvmeservice.pb.cc"
    "${PROTO_OUT_DIR}/nvmeservice.grpc.pb.cc"
)

set(NVMESERVICE_PROTO_HDRS
    "${PROTO_OUT_DIR}/nvmeservice.pb.h"
    "${PROTO_OUT_DIR}/nvmeservice.grpc.pb.h"
)

# =============================================================================
# Source Collection
# =============================================================================

file(GLOB NVMESERVICE_SOURCES
    src/*.cpp
    src/*.cu
)

# =============================================================================
# Library Target
# =============================================================================

add_library(nvmeservice SHARED
    ${NVMESERVICE_SOURCES}
    ${NVMESERVICE_PROTO_SRCS}
)

set_target_properties(nvmeservice PROPERTIES
    OUTPUT_NAME "nvmeservice"
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

target_include_directories(nvmeservice
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${PROTO_OUT_DIR}>
        $<INSTALL_INTERFACE:include/nvmeservice>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# =============================================================================
# Dependencies
# =============================================================================

target_link_libraries(nvmeservice
    PUBLIC
        gRPC::grpc++
        protobuf::libprotobuf
    PRIVATE
        libnvm
        ${TUTTI_ACCEL_RUNTIME_LIBS}
        Threads::Threads
        ${TUTTI_YAML_CPP_TARGET}
)

# =============================================================================
# Installation
# =============================================================================

install(TARGETS nvmeservice
    EXPORT tutti_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    src/nvmeservice_client.h
    src/nvmeservice_server.h
    src/nvmeservice_config.h
    src/nvmeservice_state.h
    ${NVMESERVICE_PROTO_HDRS}
    DESTINATION include/nvmeservice
)

# =============================================================================
# Examples Subdirectory
# =============================================================================

add_subdirectory(examples)

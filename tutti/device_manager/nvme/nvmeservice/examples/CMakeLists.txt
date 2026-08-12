# NVMeService examples: daemon + client smoke test.
#
# daemon: brings up libnvm controllers as the chrdev/bind owner and
#         serves gRPC for quota requests.
# client: runs the full L1 4b path -- gRPC Connect, libnvm
#         attach_client, create_group, add_user_queue, GPU IO smoke,
#         destroy_group, free_client, Disconnect.

add_executable(nvmeservice_daemon_example
    nvmeservice_daemon.cpp
)

target_link_libraries(nvmeservice_daemon_example PRIVATE
    nvmeservice
    libnvm
    ${TUTTI_YAML_CPP_TARGET}
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
    ${TUTTI_ACCEL_RUNTIME_LIBS}
)

set_target_properties(nvmeservice_daemon_example PROPERTIES
    OUTPUT_NAME   "nvmeservice_daemon"
    BUILD_RPATH   "$ORIGIN:$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
    CUDA_SEPARABLE_COMPILATION ON
)

# client: gRPC half in .cpp, CUDA / libnvm IO smoke in .cu (kept
# separate because nvcc trips over protobuf's C++17 inline-static
# enum traits if those headers go through its frontend).
add_executable(nvmeservice_client_example
    nvmeservice_client.cpp
    nvmeservice_client_io.cu
)

target_link_libraries(nvmeservice_client_example PRIVATE
    nvmeservice
    libnvm
    # nvmeservice_client_io.cu includes <tutti/cuda_like.h> (R18) — the
    # tutti_cuda_like INTERFACE carries TUTTI_USE_CUDA=1 + tutti/include.
    tutti_cuda_like
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
    ${TUTTI_ACCEL_RUNTIME_LIBS}
)

set_target_properties(nvmeservice_client_example PROPERTIES
    OUTPUT_NAME   "nvmeservice_client"
    BUILD_RPATH   "$ORIGIN:$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
    CUDA_SEPARABLE_COMPILATION ON
)

install(TARGETS nvmeservice_daemon_example nvmeservice_client_example
        DESTINATION "bin")

# ---- tutti_daemon (production) -------------------------------------------
# The owner-side daemon: brings up NVMe controllers via libnvm and
# serves gRPC so SERVICE_CLIENT Coordinators can attach.  Plain host C++
# (no CUDA device code).  Migrated from root examples/ in Round 13 S3.
add_executable(tutti_daemon tutti_daemon.cpp)
target_link_libraries(tutti_daemon PRIVATE
    nvmeservice
    libnvm
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
    ${TUTTI_ACCEL_RUNTIME_LIBS}
)
set_target_properties(tutti_daemon PROPERTIES
    BUILD_RPATH   "$ORIGIN:$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
)
install(TARGETS tutti_daemon DESTINATION "bin")

# Copy the canonical sys_config.yaml from the repo root next to the
# binaries so users can run them without figuring out a path.
if(EXISTS "${TUTTI_REPOSITORY_ROOT}/sys_config.yaml")
    configure_file("${TUTTI_REPOSITORY_ROOT}/sys_config.yaml"
                   "${CMAKE_BINARY_DIR}/bin/sys_config.yaml"
                   COPYONLY)
endif()

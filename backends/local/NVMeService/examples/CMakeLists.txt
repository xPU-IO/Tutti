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
    ${GEMINIFS_YAML_CPP_TARGET}
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
    CUDA::cudart
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
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
    CUDA::cudart
)

set_target_properties(nvmeservice_client_example PROPERTIES
    OUTPUT_NAME   "nvmeservice_client"
    BUILD_RPATH   "$ORIGIN:$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
    CUDA_SEPARABLE_COMPILATION ON
)

install(TARGETS nvmeservice_daemon_example nvmeservice_client_example
        DESTINATION "bin")

# Copy the canonical sys_config.yaml from the repo root next to the
# binaries so users can run them without figuring out a path.
configure_file("${CMAKE_SOURCE_DIR}/sys_config.yaml"
               "${CMAKE_BINARY_DIR}/bin/sys_config.yaml"
               COPYONLY)

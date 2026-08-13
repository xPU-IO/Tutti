cmake_minimum_required(VERSION 3.18)

add_executable(tutti_runtime_example
    tutti_runtime_example.cpp
)

add_executable(tutti_runtime_multi_accelerator_example
    tutti_runtime_multi_accelerator_example.cpp
)

target_link_libraries(tutti_runtime_example PRIVATE
    tutti_runtime
)

target_link_libraries(tutti_runtime_multi_accelerator_example PRIVATE
    tutti_runtime
    Threads::Threads
)

target_compile_definitions(tutti_runtime_multi_accelerator_example PRIVATE
    TUTTI_RUNTIME_MULTI_ACCEL_CONFIG_0=\"${CMAKE_CURRENT_SOURCE_DIR}/tutti_multi_accelerator_0.yaml\"
    TUTTI_RUNTIME_MULTI_ACCEL_CONFIG_1=\"${CMAKE_CURRENT_SOURCE_DIR}/tutti_multi_accelerator_1.yaml\"
)

target_compile_definitions(tutti_runtime_example PRIVATE
    TUTTI_RUNTIME_DEFAULT_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/tutti_local_nvme.yaml"
)

set_target_properties(tutti_runtime_example PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(tutti_runtime_multi_accelerator_example PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_runtime_example PRIVATE
    -Wall -Wextra
)

target_compile_options(tutti_runtime_multi_accelerator_example PRIVATE
    -Wall -Wextra
)

message(STATUS "Configured TuttiRuntime hardware examples")

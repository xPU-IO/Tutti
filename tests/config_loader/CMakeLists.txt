cmake_minimum_required(VERSION 3.18)

add_executable(tutti_config_loader_test
    config_loader_test.cpp
    config_loader_cuda_link.cu
)

target_link_libraries(tutti_config_loader_test PRIVATE
    tutti_config
    ${TUTTI_YAML_CPP_TARGET}
)

set_target_properties(tutti_config_loader_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_config_loader_test PRIVATE -Wall -Wextra)

add_test(NAME tutti_config_loader_test COMMAND tutti_config_loader_test)

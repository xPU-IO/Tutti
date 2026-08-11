cmake_minimum_required(VERSION 3.18)

add_executable(tutti_runtime_bundle_loader_contract_test
    runtime_bundle_loader_contract_test.cpp
    fill_helper.cu
)

target_link_libraries(tutti_runtime_bundle_loader_contract_test PRIVATE
    tutti_config
    nvmeservice
)

set_target_properties(tutti_runtime_bundle_loader_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_runtime_bundle_loader_contract_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
)

add_test(NAME tutti_runtime_bundle_loader_contract_test
         COMMAND tutti_runtime_bundle_loader_contract_test)

set_tests_properties(tutti_runtime_bundle_loader_contract_test PROPERTIES
    LABELS "hardware;loader_e2e;runtime_e2e"
    SKIP_RETURN_CODE 77
    TIMEOUT 300
)

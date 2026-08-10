cmake_minimum_required(VERSION 3.18)

add_executable(tutti_accelerator_device_guard_test
    accelerator_device_guard_test.cpp
)

target_link_libraries(tutti_accelerator_device_guard_test PRIVATE tutti_api)
target_include_directories(tutti_accelerator_device_guard_test PRIVATE
    ${TUTTI_REPOSITORY_ROOT}
)
set_target_properties(tutti_accelerator_device_guard_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_test(NAME tutti_accelerator_device_guard_test
         COMMAND tutti_accelerator_device_guard_test)

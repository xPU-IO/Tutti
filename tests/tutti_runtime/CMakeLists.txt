add_executable(tutti_tutti_runtime_test
    tutti_runtime_create_test.cpp
)
target_include_directories(tutti_tutti_runtime_test PRIVATE
    "${TUTTI_REPOSITORY_ROOT}"
)
target_link_libraries(tutti_tutti_runtime_test PRIVATE tutti_runtime)
set_target_properties(tutti_tutti_runtime_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
add_test(NAME tutti_tutti_runtime_test COMMAND tutti_tutti_runtime_test)
set_tests_properties(tutti_tutti_runtime_test PROPERTIES
    LABELS "host;tutti_runtime")

if(TUTTI_BUILD_HARDWARE_TESTS AND TUTTI_BUILD_HARDWARE_STACK AND
   TUTTI_FEATURE_LOCAL_NVME)
    add_executable(tutti_tutti_runtime_hardware_test
        tutti_runtime_hardware_test.cpp
    )
    target_link_libraries(tutti_tutti_runtime_hardware_test PRIVATE
        tutti_runtime
    )
    set_target_properties(tutti_tutti_runtime_hardware_test PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME tutti_tutti_runtime_hardware_test
             COMMAND tutti_tutti_runtime_hardware_test)
    set_tests_properties(tutti_tutti_runtime_hardware_test PROPERTIES
        LABELS "hardware;tutti_runtime;runtime_e2e"
        SKIP_RETURN_CODE 77
        TIMEOUT 300
    )
endif()

add_executable(tutti_config_spec_test tutti_runtime_spec_test.cpp)
target_link_libraries(tutti_config_spec_test PRIVATE tutti_runtime_spec)
set_target_properties(tutti_config_spec_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
add_test(NAME tutti_config_spec_test COMMAND tutti_config_spec_test)
set_tests_properties(tutti_config_spec_test PROPERTIES LABELS "host;config;spec")

add_executable(tutti_config_spec_debug_test tutti_runtime_spec_debug_test.cpp)
target_link_libraries(tutti_config_spec_debug_test PRIVATE tutti_runtime_spec)
set_target_properties(tutti_config_spec_debug_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
add_test(NAME tutti_config_spec_debug_test COMMAND tutti_config_spec_debug_test)
set_tests_properties(tutti_config_spec_debug_test PROPERTIES
    LABELS "host;config;spec")

add_executable(tutti_config_spec_headers_test
    tutti_runtime_spec_headers_test.cpp)
target_link_libraries(tutti_config_spec_headers_test PRIVATE tutti_api)
set_target_properties(tutti_config_spec_headers_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
add_test(NAME tutti_config_spec_headers_test
    COMMAND tutti_config_spec_headers_test)
set_tests_properties(tutti_config_spec_headers_test PROPERTIES
    LABELS "host;config;spec;headers")

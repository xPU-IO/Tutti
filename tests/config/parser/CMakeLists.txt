add_executable(tutti_config_parser_test
    tutti_runtime_config_parser_test.cpp
)
target_link_libraries(tutti_config_parser_test PRIVATE tutti_config_parser)
set_target_properties(tutti_config_parser_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
add_test(NAME tutti_config_parser_test COMMAND tutti_config_parser_test)
set_tests_properties(tutti_config_parser_test PROPERTIES
    LABELS "host;config;parser")

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_resolver_factory_test
    resolver_factory_test.cpp
)

target_include_directories(tutti_resolver_factory_test PRIVATE
    "${TUTTI_REPOSITORY_ROOT}"
)

target_link_libraries(tutti_resolver_factory_test PRIVATE
    tutti_resolver_factory
)

set_target_properties(tutti_resolver_factory_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_resolver_factory_test PRIVATE -Wall -Wextra)

add_test(NAME tutti_resolver_factory_test
         COMMAND tutti_resolver_factory_test)
set_tests_properties(tutti_resolver_factory_test PROPERTIES
    LABELS "host;resolver;factory"
)

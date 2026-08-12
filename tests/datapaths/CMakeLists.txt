cmake_minimum_required(VERSION 3.18)

add_executable(tutti_data_path_factory_test
    data_path_factory_test.cpp
)

target_include_directories(tutti_data_path_factory_test PRIVATE
    "${TUTTI_REPOSITORY_ROOT}"
)

target_link_libraries(tutti_data_path_factory_test PRIVATE
    tutti_data_path_factory
)

if(TUTTI_BUILD_HARDWARE_STACK AND TUTTI_FEATURE_LOCAL_NVME)
    target_compile_definitions(tutti_data_path_factory_test PRIVATE
        TUTTI_TEST_HAS_LOCAL_NVME=1
    )
    target_link_libraries(tutti_data_path_factory_test PRIVATE
        tutti_local_nvme_datapath
        tutti_striped_local_nvme_datapath
    )
endif()

set_target_properties(tutti_data_path_factory_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_data_path_factory_test PRIVATE -Wall -Wextra)

add_test(NAME tutti_data_path_factory_test
         COMMAND tutti_data_path_factory_test)
set_tests_properties(tutti_data_path_factory_test PROPERTIES
    LABELS "host;datapath;factory"
)

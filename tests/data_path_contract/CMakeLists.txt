# tests/data_path_contract/CMakeLists.txt
#
# Hardware-free: plain C++17, links only tutti_spi for DataPath SPI headers.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_data_path_contract_test
    data_path_contract_test.cpp
)

target_link_libraries(tutti_data_path_contract_test PRIVATE
    tutti_spi
)

target_include_directories(tutti_data_path_contract_test PRIVATE
    ${TUTTI_REPOSITORY_ROOT}
)

set_target_properties(tutti_data_path_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_data_path_contract_test PRIVATE
    -Wall -Wextra
)

add_test(NAME tutti_data_path_contract_test
         COMMAND tutti_data_path_contract_test)

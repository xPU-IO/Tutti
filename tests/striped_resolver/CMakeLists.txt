cmake_minimum_required(VERSION 3.18)

# Hardware-free contract test for StripedResolver + StripedLocalNvmePayload.
# Uses mock shard resolvers (no real block devices, no FIEMAP).

add_executable(tutti_striped_resolver_contract_test
    striped_resolver_contract_test.cpp
)

target_link_libraries(tutti_striped_resolver_contract_test PRIVATE
    tutti_striped_file_resolver
)

set_target_properties(tutti_striped_resolver_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_striped_resolver_contract_test PRIVATE -Wall -Wextra)

add_test(NAME tutti_striped_resolver_contract_test
    COMMAND tutti_striped_resolver_contract_test)

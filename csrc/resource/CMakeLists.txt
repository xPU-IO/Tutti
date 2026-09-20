add_subdirectory(nvme)
add_subdirectory(memory)

add_library(tutti_resource STATIC
    resource_factory.cpp
)

target_include_directories(tutti_resource PUBLIC
    "${TUTTI_REPOSITORY_ROOT}"
)

target_link_libraries(tutti_resource PUBLIC
    tutti_api
    tutti_nvme_resource
    tutti_memory_resource
)

set_target_properties(tutti_resource PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

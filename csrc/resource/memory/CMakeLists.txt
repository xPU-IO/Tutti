add_library(tutti_memory_resource STATIC
    memory_resource.cpp
)

target_include_directories(tutti_memory_resource PUBLIC
    "${TUTTI_REPOSITORY_ROOT}"
)

target_link_libraries(tutti_memory_resource PUBLIC
    tutti_api
)

set_target_properties(tutti_memory_resource PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

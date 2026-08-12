add_library(tutti_resolver_factory STATIC
    resolver_factory.cpp
)

target_include_directories(tutti_resolver_factory PUBLIC
    "${TUTTI_REPOSITORY_ROOT}"
)

target_link_libraries(tutti_resolver_factory PUBLIC
    tutti_runtime_spec
    tutti_resource
    tutti_local_file_resolver
    tutti_striped_file_resolver
    tutti_spi
)

set_target_properties(tutti_resolver_factory PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

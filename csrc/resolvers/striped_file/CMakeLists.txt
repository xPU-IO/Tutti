# tutti/resolvers/striped_file/CMakeLists.txt
#
# Private resolver target: header-only StripedResolver.
# Links tutti_local_file_resolver + striped binding.

add_library(tutti_striped_file_resolver INTERFACE)

target_include_directories(tutti_striped_file_resolver INTERFACE
    $<BUILD_INTERFACE:${TUTTI_REPOSITORY_ROOT}>
)

target_link_libraries(tutti_striped_file_resolver INTERFACE
    tutti_striped_local_nvme_binding
    tutti_local_file_resolver
    tutti_spi
)

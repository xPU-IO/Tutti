# tutti/resolvers/local_file/CMakeLists.txt
#
# Private resolver target: header-only LocalFileResolver using FS_IOC_FIEMAP.
#
# Consumers link tutti_local_file_resolver to obtain:
#   - workspace root include (for <tutti/resolvers/local_file/resolver.h>)
#   - tutti_local_file_binding (transitively, for the payload contract)
#   - tutti_spi (transitively, for public headers)
#
# Hardware-free at the header level (Linux FIEMAP headers are always available).
# Does NOT propagate to tutti_api / tutti_spi / install public headers.

add_library(tutti_local_file_resolver INTERFACE)

# Workspace root so that <tutti/resolvers/local_file/resolver.h> resolves.
target_include_directories(tutti_local_file_resolver INTERFACE
    $<BUILD_INTERFACE:${TUTTI_REPOSITORY_ROOT}>
)

target_link_libraries(tutti_local_file_resolver INTERFACE
    tutti_local_file_binding
    tutti_spi
)

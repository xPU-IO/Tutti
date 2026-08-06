# tutti/bindings/ext4_local_nvme/CMakeLists.txt
#
# Private binding target: header-only payload contract shared between
# the ext4 resolver and the local NVMe DataPath.
#
# Consumers link tutti_local_file_binding to obtain:
#   - workspace root include (for <tutti/bindings/ext4_local_nvme/binding.h>)
#   - tutti_spi (transitively, for <tutti/status.h> and <tutti/spi/...>)
#
# Hardware-free: no CUDA, no libnvm, no kernel headers.
# Does NOT propagate to tutti_api / tutti_spi / install public headers.

add_library(tutti_local_file_binding INTERFACE)

# Workspace root so that <tutti/bindings/ext4_local_nvme/binding.h> resolves.
target_include_directories(tutti_local_file_binding INTERFACE
    $<BUILD_INTERFACE:${TUTTI_REPOSITORY_ROOT}>
)

target_link_libraries(tutti_local_file_binding INTERFACE
    tutti_spi
)

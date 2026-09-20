# csrc/payloads/ext4_local_nvme/CMakeLists.txt
#
# Private payload target: header-only contract shared between
# the ext4 resolver and the local NVMe DataPath.
#
# Consumers link tutti_local_file_payload to obtain:
#   - workspace root include (for "csrc/payloads/ext4_local_nvme/payload.h")
#   - tutti_spi (transitively, for <tutti/status.h> and <tutti/spi/...>)
#
# Hardware-free: no CUDA, no libnvm, no kernel headers.
# Does NOT propagate to tutti_api / tutti_spi / install public headers.

add_library(tutti_local_file_payload INTERFACE)

# Workspace root so that "csrc/payloads/ext4_local_nvme/payload.h" resolves.
target_include_directories(tutti_local_file_payload INTERFACE
    $<BUILD_INTERFACE:${TUTTI_REPOSITORY_ROOT}>
)

target_link_libraries(tutti_local_file_payload INTERFACE
    tutti_spi
)

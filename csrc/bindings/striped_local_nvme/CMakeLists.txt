# tutti/bindings/striped_local_nvme/CMakeLists.txt
#
# Private binding target: header-only striped payload contract.
# Links tutti_spi + ext4_local_nvme binding (for sub-payload type).

add_library(tutti_striped_local_nvme_binding INTERFACE)

target_include_directories(tutti_striped_local_nvme_binding INTERFACE
    $<BUILD_INTERFACE:${TUTTI_REPOSITORY_ROOT}>
)

target_link_libraries(tutti_striped_local_nvme_binding INTERFACE
    tutti_local_file_binding
    tutti_spi
)

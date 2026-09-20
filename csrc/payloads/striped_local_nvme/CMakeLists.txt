# csrc/payloads/striped_local_nvme/CMakeLists.txt
#
# Private payload target: header-only striped payload contract.
# Links tutti_spi + the ext4_local_nvme payload (for the sub-payload type).

add_library(tutti_striped_local_nvme_payload INTERFACE)

target_include_directories(tutti_striped_local_nvme_payload INTERFACE
    $<BUILD_INTERFACE:${TUTTI_REPOSITORY_ROOT}>
)

target_link_libraries(tutti_striped_local_nvme_payload INTERFACE
    tutti_local_file_payload
    tutti_spi
)

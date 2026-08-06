# tests/mount_manager_contract/CMakeLists.txt
# Round 17 S1: host-side unit tests for MountManager.
#
# These tests do NOT touch the running daemon or real NVMe devices.
# They test:
#   1. config parsing (auto_mount field, unmount_retry block)
#   2. /proc parsing helpers (scan_holders against the test process itself)
#   3. is_mounted() against /proc/self/mountinfo
#   4. mount_one/unmount_all lifecycle using a loopback tmpfs (no real NVMe)

find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)

if(TARGET yaml-cpp::yaml-cpp)
    set(_tutti_mount_manager_yaml_target yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(_tutti_mount_manager_yaml_target yaml-cpp)
else()
    set(_tutti_mount_manager_yaml_target ${YAML_CPP_LIBRARIES})
endif()

add_executable(tutti_mount_manager_contract_test
    mount_manager_contract_test.cpp
    ${TUTTI_SOURCE_DIR}/device_manager/nvme/nvmeservice/src/mount_manager.cpp
    ${TUTTI_SOURCE_DIR}/device_manager/nvme/nvmeservice/src/nvmeservice_config.cpp
)

target_link_libraries(tutti_mount_manager_contract_test PRIVATE
    ${_tutti_mount_manager_yaml_target}
    Threads::Threads
)

target_include_directories(tutti_mount_manager_contract_test PRIVATE
    ${TUTTI_SOURCE_DIR}/device_manager/nvme/nvmeservice/src
    ${TUTTI_SOURCE_DIR}/device_manager/nvme/libnvm/include
)

set_target_properties(tutti_mount_manager_contract_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_test(NAME tutti_mount_manager_contract_test
         COMMAND tutti_mount_manager_contract_test)
set_tests_properties(tutti_mount_manager_contract_test PROPERTIES
    LABELS "host;mount_manager")

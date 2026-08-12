set(_tutti_config_public_headers
    tutti_runtime_config_parser.h
    tutti_runtime_spec.h
    spec/resource/resource_spec.h
    spec/resource/nvme_spec.h
    spec/resource/memory_spec.h
    spec/resolver/resolver_spec.h
    spec/resolver/local_file_spec.h
    spec/resolver/striped_file_spec.h
    spec/resolver/memfs_spec.h
    spec/datapath/datapath_spec.h
    spec/datapath/nvme_datapath_spec.h
    spec/datapath/local_nvme_spec.h
    spec/datapath/striped_local_nvme_spec.h
    spec/datapath/memfs_spec.h
    spec/backend/backend_spec.h
    spec/backend/ext4_local_nvme_spec.h
    spec/backend/striped_local_nvme_spec.h
    spec/backend/memfs_spec.h
)

# The public source headers live with their config implementation.  Mirror
# only that explicit set into a conventional build-tree include root so
# consumers can keep using <tutti/config/...> without exposing all of tutti/.
set(TUTTI_CONFIG_PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
foreach(_tutti_config_header IN LISTS _tutti_config_public_headers)
    set(_tutti_config_header_source
        "${CMAKE_CURRENT_SOURCE_DIR}/${_tutti_config_header}")
    set(_tutti_config_header_build
        "${TUTTI_CONFIG_PUBLIC_INCLUDE_DIR}/tutti/config/${_tutti_config_header}")
    get_filename_component(_tutti_config_header_build_dir
        "${_tutti_config_header_build}" DIRECTORY)
    file(MAKE_DIRECTORY "${_tutti_config_header_build_dir}")
    configure_file(
        "${_tutti_config_header_source}"
        "${_tutti_config_header_build}"
        COPYONLY
    )

    get_filename_component(_tutti_config_header_subdir
        "${_tutti_config_header}" DIRECTORY)
    install(FILES "${_tutti_config_header_source}"
        DESTINATION "include/tutti/config/${_tutti_config_header_subdir}")
endforeach()
set(TUTTI_CONFIG_PUBLIC_INCLUDE_DIR
    "${TUTTI_CONFIG_PUBLIC_INCLUDE_DIR}" PARENT_SCOPE)

add_library(tutti_runtime_spec STATIC
    spec/tutti_runtime_spec.cpp
    spec/resource/memory_spec.cpp
    spec/resource/nvme_spec.cpp
    spec/resolver/local_file_spec.cpp
    spec/resolver/memfs_spec.cpp
    spec/resolver/striped_file_spec.cpp
    spec/datapath/local_nvme_spec.cpp
    spec/datapath/memfs_spec.cpp
    spec/datapath/striped_local_nvme_spec.cpp
    spec/backend/ext4_local_nvme_spec.cpp
    spec/backend/memfs_spec.cpp
    spec/backend/striped_local_nvme_spec.cpp
)
target_include_directories(tutti_runtime_spec PUBLIC
    $<BUILD_INTERFACE:${TUTTI_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${TUTTI_CONFIG_PUBLIC_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(tutti_runtime_spec PRIVATE
    "${TUTTI_REPOSITORY_ROOT}"
)
if(TUTTI_ACCELERATOR STREQUAL "HOST")
    set(_tutti_spec_default_accel_id -1)
else()
    set(_tutti_spec_default_accel_id 0)
endif()
target_compile_definitions(tutti_runtime_spec PUBLIC
    TUTTI_COMPILED_ACCELERATOR_PROFILE="${TUTTI_ACCELERATOR}"
    TUTTI_DEFAULT_ACCEL_ID=${_tutti_spec_default_accel_id}
)
set_target_properties(tutti_runtime_spec PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_library(tutti_config_parser STATIC
    parser/tutti_runtime_config_parser.cpp
    parser/resource/memory_config_parser.cpp
    parser/resource/nvme_config_parser.cpp
    parser/resolver/local_file_config_parser.cpp
    parser/resolver/memfs_config_parser.cpp
    parser/resolver/striped_file_config_parser.cpp
    parser/datapath/local_nvme_config_parser.cpp
    parser/datapath/memfs_config_parser.cpp
    parser/datapath/striped_local_nvme_config_parser.cpp
    parser/backend/ext4_local_nvme_config_parser.cpp
    parser/backend/memfs_config_parser.cpp
    parser/backend/striped_local_nvme_config_parser.cpp
)
target_include_directories(tutti_config_parser PUBLIC
    $<BUILD_INTERFACE:${TUTTI_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(tutti_config_parser PRIVATE
    "${TUTTI_REPOSITORY_ROOT}"
)
target_link_libraries(tutti_config_parser PUBLIC
    tutti_runtime_spec
    ${TUTTI_YAML_CPP_TARGET}
)
set_target_properties(tutti_config_parser PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_library(tutti_runtime STATIC
    "${TUTTI_SOURCE_DIR}/tutti_runtime/tutti_runtime.cpp"
    "${TUTTI_SOURCE_DIR}/tutti_runtime/tutti_runtime_create.cpp"
)
target_include_directories(tutti_runtime PUBLIC
    $<BUILD_INTERFACE:${TUTTI_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(tutti_runtime PRIVATE
    "${TUTTI_REPOSITORY_ROOT}"
)
target_link_libraries(tutti_runtime PUBLIC
    tutti_config_parser
    tutti_runtime_spec
    tutti_resource
    tutti_api
)
target_link_libraries(tutti_runtime PRIVATE
    tutti_resolver_factory
    tutti_data_path_factory
)
set_target_properties(tutti_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

install(TARGETS tutti_runtime_spec tutti_config_parser tutti_runtime
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

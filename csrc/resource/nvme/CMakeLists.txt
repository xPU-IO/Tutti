add_library(tutti_nvme_resource STATIC
    nvme_resource.cpp
    nvme_resource_grpc.cpp
)

target_include_directories(tutti_nvme_resource PUBLIC
    "${TUTTI_REPOSITORY_ROOT}"
)

target_link_libraries(tutti_nvme_resource PUBLIC
    tutti_api
)

set_target_properties(tutti_nvme_resource PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

function(tutti_enable_nvme_resource_grpc)
    if(TARGET nvmeservice)
        target_link_libraries(tutti_nvme_resource PRIVATE nvmeservice)
        target_compile_definitions(tutti_nvme_resource PRIVATE
            TUTTI_NVME_RESOURCE_HAS_NVMESERVICE=1
        )
    endif()
endfunction()

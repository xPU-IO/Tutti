# csrc/storage_objects/CMakeLists.txt
#
# Storage object layer implementation. Private: consumers reach this only
# through <tutti/spi/storage_object_store.h>, and tests/header_hygiene asserts
# these headers stay unreachable as <tutti/...>.
#
# Pure host C++17 -- no CUDA, no device code. The four units here are
# independently testable without hardware:
#   object_header_codec   byte-level object header encode/decode
#   checkpoint_region     mirrored checkpoint containers
#   residency_bitmap      mmap cross-rank residency
#   space_allocator       slot-number allocation and reclamation states

add_library(tutti_storage_objects STATIC
    object_header_codec.cpp
    checkpoint_region.cpp
    residency_bitmap.cpp
    space_allocator.cpp
)

# Repo root on the include path so private headers are addressed as
# "csrc/storage_objects/...", which is what keeps them out of the public
# <tutti/...> namespace.
target_include_directories(tutti_storage_objects PUBLIC
    "${TUTTI_REPOSITORY_ROOT}"
)

target_link_libraries(tutti_storage_objects PUBLIC
    tutti_spi
)

set_target_properties(tutti_storage_objects PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_compile_options(tutti_storage_objects PRIVATE
    -Wall -Wextra
)

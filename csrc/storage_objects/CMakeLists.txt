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
    slot_placement_policy.cpp
    slot_media.cpp
    object_store_core.cpp
    store_factory.cpp
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

# Deliberately NOT linked against the resolvers. Resolution belongs to
# StorageRuntime, which consumes the slot URIs this layer produces; resolving
# here as well would resolve every object twice, and that path is
# open + fstat + fsync + FIEMAP plus a globally-serialised peer-memory DMA
# mapping. The absent dependency is the enforcement -- adding it would fail to
# link nothing, so it is asserted in tests/header_hygiene instead.

set_target_properties(tutti_storage_objects PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    # Consumed by the pybind11 extension (tutti_runtime._core), which is a
    # shared object: without PIC the linker rejects the archive's absolute
    # relocations.
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(tutti_storage_objects PRIVATE
    -Wall -Wextra
)

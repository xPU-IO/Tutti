#ifndef __TUTTI_IO_ENGINE_BACKEND_TYPE_H__
#define __TUTTI_IO_ENGINE_BACKEND_TYPE_H__

/**
 * backend_type.h -- enum tag for backend identification.
 *
 * Layer: io_engine (Backend SPI)
 *
 * Spec: doc/design/backend-spi.md §3.1
 *
 * Rule: do NOT reorder existing values when adding a new backend.
 *       New backends append at the end and bump the next free integer.
 *       Wire-format consumers (BufferDescriptor) rely on the numeric
 *       values being stable across versions and builds.
 */

#include <cstdint>

namespace tutti {

enum class BackendType : uint32_t {
    LOCAL_NVME = 0,
    RDMA       = 1,
    GDS        = 2,
    // New backends append here. Existing values must NEVER be reordered.
};

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_BACKEND_TYPE_H__

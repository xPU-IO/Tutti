#ifndef __TUTTI_IO_ENGINE_BUFFER_DESCRIPTOR_H__
#define __TUTTI_IO_ENGINE_BUFFER_DESCRIPTOR_H__

/**
 * buffer_descriptor.h -- backend-neutral wrapper for per-slice IO metadata.
 *
 * Layer: io_engine (Backend SPI)
 *
 * Spec: doc/design/backend-spi.md §3.2
 *
 * Role:
 *   - Memory Layer fills these via IBackendProvider::prepare_descriptors()
 *     when a tensor is registered. Each descriptor describes one IO-sized
 *     slice (one PRP / SGL segment for NVMe, one work-request region for
 *     RDMA, etc.).
 *   - IO Engine passes the array through to launch_batch_gpu_stream() /
 *     submit_batch_cpu_sync() unchanged.
 *   - Device-side kernels read the union member that matches their
 *     backend (host guarantees the discriminator before launch).
 *
 * Invariants:
 *   - The union payload MUST fit in raw[48]. Adding a new backend struct
 *     larger than 48 bytes is a versioning event (v0.2+).
 *   - `backend_type` is the discriminator; readers MUST check it before
 *     accessing a specific union member.
 *   - Upper layers treat the union as opaque -- they pass it through but
 *     never read backend-specific fields.
 */

#include <cstdint>

#include "backend_type.h"

namespace tutti {

/**
 * NVMe transfer kind. Distinguishes PRP-form transfers (legacy) from
 * SGL-form transfers (NVMe 1.2+). Numeric values:
 *   - 0..15 reserved for PRP family (PRP_SINGLE / DUAL / LIST occupy
 *     0/1/2 to stay numerically compatible with legacy `transfer_type`
 *     fields elsewhere in the codebase).
 *   - 16..31 reserved for SGL family.
 *   - 32+ reserved for future extensions (mixed mode, fused commands).
 */
enum class NVMeTransferKind : uint32_t {
    // PRP family ----------------------------------------------------------
    PRP_SINGLE     = 0,   // prp.prp1 = data page;          prp.prp2 = 0
    PRP_DUAL       = 1,   // prp.prp1 = page0;              prp.prp2 = page1
    PRP_LIST       = 2,   // prp.prp1 = first data page;    prp.prp2 = prp_list_page_ioaddr

    // SGL family ----------------------------------------------------------
    SGL_DATA_BLOCK = 16,  // sgl.addr = single SGL Data Block descriptor;     sgl.aux = 0
    SGL_LAST_SEG   = 17,  // sgl.addr = Last Segment descriptor list head;    sgl.aux = segment byte length
    SGL_SEG        = 18,  // sgl.addr = chained segment list;                 sgl.aux = segment byte length
};

/**
 * Per-slice description for the local NVMe backend.
 *
 * Layout (32 bytes):
 *   - 4-byte transfer kind discriminator
 *   - 4-byte byte length
 *   - 8-byte slice offset within the tensor
 *   - 16-byte addressing payload, interpreted per `kind` via the union
 *
 * The union keeps PRP and SGL paths textually distinct in device code:
 * a kernel branches on `kind` and reads only the matching member. No
 * code reads through the inactive member.
 */
struct NVMeBufferDesc {
    NVMeTransferKind kind;
    uint32_t         data_length;    // total IO size in bytes
    uint64_t         tensor_offset;  // slice offset within the tensor
    union {
        struct { uint64_t prp1; uint64_t prp2; } prp;
        struct { uint64_t addr; uint64_t aux;  } sgl;
    };
};

/**
 * Per-slice description for the RDMA backend. Reserved for future
 * implementation; present here so BufferDescriptor's union shape is
 * fixed across backends.
 */
struct RDMABufferDesc {
    uint64_t remote_addr;    // Remote virtual address
    uint64_t local_addr;     // Local GPU buffer address
    uint32_t rkey;           // Remote memory key
    uint32_t lkey;           // Local memory key
    uint32_t data_length;    // Transfer size in bytes
    uint32_t reserved;
};

/**
 * Tagged union the IO Engine carries through. Total payload size MUST
 * stay within raw[48] so descriptor arrays remain cacheline-friendly on
 * GPU. Currently:
 *   - NVMeBufferDesc = 32 bytes
 *   - RDMABufferDesc = 32 bytes
 *   - 48-byte budget leaves headroom for one more 16-byte field per
 *     backend before the format becomes a versioning event.
 */
struct BufferDescriptor {
    BackendType backend_type;
    uint32_t    reserved;
    union {
        NVMeBufferDesc nvme;
        RDMABufferDesc rdma;
        uint8_t        raw[48];
    };
};

struct MemoryRegion;  // memory/include/memory_region.h

/**
 * Contiguous batch of BufferDescriptor. Carries a MemoryRegion handle
 * that names the app-allocated, runtime-registered region the array
 * lives in. Pairs with IORequestBatch (see io_request.h); IORequest's
 * `descriptor` pointer typically points into this batch's `descs[]`.
 *
 * Memory ownership / residency policy is identical to IORequestBatch
 * (read its docstring). The two batches MAY share one MemoryRegion or
 * use two different regions; backends MUST handle both shapes.
 */
struct BufferDescriptorBatch {
    const BufferDescriptor* descs;
    uint32_t                count;
    const MemoryRegion*     region;   // app-registered home of `descs`
};

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_BUFFER_DESCRIPTOR_H__

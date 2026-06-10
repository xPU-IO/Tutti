#ifndef __TUTTI_IO_ENGINE_QUEUE_H__
#define __TUTTI_IO_ENGINE_QUEUE_H__

/**
 * queue.h -- queue abstraction with dual CPU/GPU pointer descriptor.
 *
 * Layer: io_engine (Backend SPI)
 *
 * Spec: doc/design/backend-spi.md §3.3
 *
 * Role:
 *   - IQueue is the runtime's portable handle on a single SQ/CQ pair,
 *     regardless of underlying transport (NVMe doorbell, io_uring,
 *     RDMA QP, mock for tests).
 *   - Each ring (SQ or CQ) exposes BOTH a CPU-accessible pointer and a
 *     GPU-accessible pointer. Either may be nullptr if that side isn't
 *     supported (e.g. io_uring sets dev_ptr = nullptr; NVMe with
 *     GPUDirect sets both).
 *   - capabilities is a QueueCap bitmask telling callers which
 *     enqueue/poll paths are valid for this instance.
 *
 * GPU dispatch:
 *   - GPU code does NOT call IQueue methods (no vtable on GPU). It
 *     reads `desc().sq.dev_ptr` / `desc().cq.dev_ptr` directly and
 *     calls backend-private device-side helpers.
 *   - Host code uses enqueue_cpu / poll_cq_cpu when the relevant cap
 *     bits are set; otherwise relies on the GPU pointers + a backend
 *     kernel.
 *
 * Lifecycle:
 *   - Created by IQueueProvider::create_queue(); destroyed by
 *     IQueueProvider::destroy_queue(). The QueueDesc handle is valid
 *     for the lifetime of the IQueue object.
 */

#include <cstdint>

#include "io_request.h"  // IORequest, IOCompletion (canonical home)

namespace tutti {

/**
 * Capability flags. AND with QueueDesc::capabilities to test support.
 * QUEUE_CAP_SHARED_MEM is the prerequisite for COOP submit mode.
 */
enum QueueCap : uint32_t {
    QUEUE_CAP_CPU_ENQUEUE = 1u << 0,   // CPU may write into SQ
    QUEUE_CAP_GPU_ENQUEUE = 1u << 1,   // GPU kernel may write into SQ
    QUEUE_CAP_CPU_POLL    = 1u << 2,   // CPU may read from CQ
    QUEUE_CAP_GPU_POLL    = 1u << 3,   // GPU kernel may read from CQ
    QUEUE_CAP_SHARED_MEM  = 1u << 4,   // SQ/CQ memory is visible to both sides
};

/**
 * Dual-pointer descriptor for one ring (SQ or CQ).
 * cpu_ptr and dev_ptr may point to the same pinned memory or to two
 * mappings of the same physical pages. Either may be nullptr.
 */
struct QueueRingDesc {
    void*    cpu_ptr;
    void*    dev_ptr;
    uint32_t depth;     // entries (must be power of 2)
    uint32_t entry_sz;  // bytes per entry
};

/**
 * Full queue descriptor, returned by IQueue::desc(). Stable for the
 * lifetime of the IQueue.
 */
struct QueueDesc {
    QueueRingDesc sq;
    QueueRingDesc cq;
    uint32_t      queue_index;   // index inside the provider's pool
    uint32_t      capabilities;  // bitmask of QueueCap
};

// Backend-neutral completion record (IOCompletion) lives in
// io_request.h alongside IORequest / IORequestBatch -- they form one
// request/response triple. CPU-side polling fills IOCompletion[];
// GPU-side polling reads backend-specific completion entry layouts
// directly via `desc().cq.dev_ptr`.

class IQueue {
public:
    virtual ~IQueue() = default;

    /// Stable descriptor for this queue's life. Reads cap flags / pointers.
    virtual const QueueDesc& desc() const = 0;

    /**
     * CPU-side enqueue. Writes up to `count` requests into the SQ.
     * Returns the number actually enqueued (may be < count if SQ full).
     * Only valid if capabilities & QUEUE_CAP_CPU_ENQUEUE.
     */
    virtual uint32_t enqueue_cpu(const IORequest* requests, uint32_t count) = 0;

    /**
     * CPU-side poll. Drains up to `max` completions into out[].
     * Returns the number written. Only valid if capabilities & QUEUE_CAP_CPU_POLL.
     */
    virtual uint32_t poll_cq_cpu(IOCompletion* out, uint32_t max) = 0;

    // GPU-side enqueue / poll have no virtual methods on purpose.
    // Backend kernels read desc().sq.dev_ptr / desc().cq.dev_ptr and call
    // backend-private inline device functions. See backend-spi.md §6.
};

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_QUEUE_H__

#ifndef __TUTTI_IO_ENGINE_BACKEND_PROVIDER_H__
#define __TUTTI_IO_ENGINE_BACKEND_PROVIDER_H__

/**
 * backend_provider.h -- the host-side SPI every backend implements.
 *
 * Layer: io_engine (Backend SPI). Builds on Memory Layer and produces
 *        work for Device Manager queues.
 *
 * Spec: doc/design/backend-spi.md §4.
 *
 * Role:
 *   - Single C++ interface the runtime core (Memory Layer / Device
 *     Manager / IO Engine) talks to. Every backend (local_nvme, RDMA,
 *     GDS, ...) provides one implementation.
 *   - Covers the four submission modes (BATCH_GPU_STREAM,
 *     BATCH_CPU_SYNC, BATCH_CPU_ASYNC, COOP) plus tensor descriptor
 *     preparation and queue acquisition.
 *
 * Memory contract:
 *   - Both batches passed to submission methods carry MemoryRegion*
 *     handles (see memory/include/memory_region.h). The two batches
 *     MAY share one region or sit in two different regions; backends
 *     MUST handle both shapes.
 *   - Backends never allocate batches or buffers themselves; the
 *     application allocates and the Memory Layer registers.
 *
 * GPU dispatch:
 *   - GPU kernels are NOT polymorphic. The host picks the backend and
 *     calls launch_batch_gpu_stream(); the backend's host-side method
 *     launches its OWN __global__ kernel directly. See backend-spi.md §6.
 *
 * v0.1 implementation requirements per backend:
 *   REQUIRED: prepare_descriptors, acquire_queue, release_queue,
 *             initialize, cleanup, launch_batch_gpu_stream,
 *             submit_batch_cpu_sync.
 *   OPTIONAL: submit_batch_cpu_async (may return nullptr),
 *             setup/drain/teardown_coop_channel (may return false).
 */

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "backend_type.h"

namespace tutti {

// Forward declarations -- consumers include the matching header where needed.
struct BufferDescriptor;     // buffer_descriptor.h
struct BufferDescriptorBatch;// buffer_descriptor.h
struct IORequest;            // io_request.h
struct IORequestBatch;       // io_request.h
struct CoopIOChannel;        // coop_channel.h
enum class CoopDirection : uint32_t;  // coop_channel.h
struct MemoryRegion;         // memory/include/memory_region.h
struct SubSliceInfo;         // currently in filesystems/.../geminifs_mem.h;
                             // a runtime-canonical home will be defined when
                             // the runtime/ object model is finalised (Slice 3).
class IQueue;                // queue.h
class IQueueProvider;        // queue_provider.h
class IOFuture;              // io_future.h

class IBackendProvider {
public:
    virtual ~IBackendProvider() = default;

    // ------------------------------------------------------------------
    // 4.1 Memory Layer interface
    // ------------------------------------------------------------------

    /**
     * Translate raw DMA IO addresses into backend-specific
     * BufferDescriptors. Called by the Memory Layer during tensor
     * registration.
     *
     * @param ioaddrs    Per-page DMA bus addresses obtained from libnvm
     *                   / ibverbs / GDS, length n_ioaddrs.
     * @param n_ioaddrs  Number of pages.
     * @param slices     Sub-slice layout, length n_slices.
     * @param n_slices   Number of slices == number of descriptors to produce.
     * @param out_descs  Caller-allocated output array, length n_slices.
     *                   Lives in the MemoryRegion the application chose for
     *                   the descriptor pool.
     * @return true on success, false on backend-side failure.
     */
    virtual bool prepare_descriptors(
        const uint64_t*      ioaddrs,
        std::size_t          n_ioaddrs,
        const SubSliceInfo*  slices,
        std::size_t          n_slices,
        BufferDescriptor*    out_descs) = 0;

    // ------------------------------------------------------------------
    // 4.2 Device Manager interface
    // ------------------------------------------------------------------

    /**
     * Acquire one queue. The returned IQueue* is valid until
     * release_queue() is called and MUST be thread-safe to acquire
     * concurrently. Returns nullptr if no queue can satisfy
     * `required_caps` (see queue.h for QueueCap bits).
     *
     * Examples:
     *   NVMe (default):  bits = CPU_ENQUEUE | GPU_ENQUEUE | CPU_POLL | GPU_POLL.
     *   io_uring:        bits = CPU_ENQUEUE | CPU_POLL.
     *   COOP-only:       bits = SHARED_MEM | (one or both ENQUEUE bits).
     */
    virtual IQueue* acquire_queue(uint32_t required_caps) = 0;

    /**
     * Return a queue to the pool. May be a no-op for stateless
     * round-robin pools.
     */
    virtual void release_queue(IQueue* queue) = 0;

    /**
     * Expose the underlying queue factory so a caller can introspect
     * or replace it (e.g. inject an io_uring-backed provider).
     * Returns nullptr if the backend pins its provider.
     */
    virtual IQueueProvider* queue_provider() = 0;

    // ------------------------------------------------------------------
    // 4.3 Lifecycle
    // ------------------------------------------------------------------

    /// Called once after queues + DMA contexts are ready. Finalises
    /// device-side init (registering pools, populating constant memory).
    virtual bool initialize() = 0;

    /// Called once at shutdown. Backend releases all device resources.
    virtual void cleanup() = 0;

    // ------------------------------------------------------------------
    // 4.4 IO Engine interface (four submission modes)
    //
    // All submission methods take BufferDescriptorBatch + IORequestBatch.
    // Each batch carries its own MemoryRegion* so the backend can
    // resolve DMA mapping, IPC import, etc. through the Memory Layer.
    // The two batches MAY share a region or use two different regions.
    // ------------------------------------------------------------------

    // ---- Mode 1: BATCH_GPU_STREAM (REQUIRED) -------------------------
    /**
     * Launch the backend's GPU IO kernel on `stream`. The backend
     * picks its own grid/block dimensions. Completion is CUDA
     * stream-ordered; the caller may chain further work via cuda
     * events or stream callbacks.
     *
     * Residency precondition: descs.region and requests.region MUST
     * expose a `device_ptr` accessible from `stream`'s CUDA context.
     *
     * @param queue      Must have GPU_ENQUEUE | GPU_POLL.
     * @param stream     CUDA stream the kernel is enqueued on.
     * @param descs      Descriptor batch (device-resident).
     * @param requests   Request batch (device-resident).
     * @param is_read    Direction (uniform across the batch).
     */
    virtual void launch_batch_gpu_stream(
        IQueue*                       queue,
        cudaStream_t                  stream,
        const BufferDescriptorBatch&  descs,
        const IORequestBatch&         requests,
        bool                          is_read) = 0;

    // ---- Mode 2: BATCH_CPU_SYNC (REQUIRED) ---------------------------
    /**
     * CPU prepares + submits + blocks until the batch completes.
     *
     * Residency precondition: both batches must expose `host_ptr`.
     *
     * @param queue      Must have CPU_ENQUEUE | CPU_POLL.
     * @return true if every request completed successfully.
     */
    virtual bool submit_batch_cpu_sync(
        IQueue*                       queue,
        const BufferDescriptorBatch&  descs,
        const IORequestBatch&         requests,
        bool                          is_read) = 0;

    // ---- Mode 3: BATCH_CPU_ASYNC (OPTIONAL) --------------------------
    /**
     * CPU submits, returns IOFuture* immediately. Caller owns and
     * must `delete` the future.
     *
     * Lifetime: both batches' MemoryRegions MUST stay registered until
     * `future->is_complete() == true`.
     *
     * @return nullptr if the backend does not implement async submit.
     *         Caller should fall back to BATCH_CPU_SYNC.
     */
    virtual IOFuture* submit_batch_cpu_async(
        IQueue*                       queue,
        const BufferDescriptorBatch&  descs,
        const IORequestBatch&         requests,
        bool                          is_read) = 0;

    // ---- Mode 4: COOP (OPTIONAL) -------------------------------------
    /**
     * Allocate a CoopIOChannel pinned in shared memory readable by
     * both CPU and GPU. The backing memory MUST come from the Memory
     * Layer (typically MANAGED or PINNED_HOST). Caller owns the
     * channel and MUST call teardown_coop_channel() to release it.
     *
     * @return false if the backend does not implement cooperative submit.
     */
    virtual bool setup_coop_channel(
        CoopIOChannel** out_channel,
        CoopDirection   direction,
        uint32_t        submit_depth,
        uint32_t        compl_depth) = 0;

    /**
     * CPU proxy: drain pending submissions from the channel's SQ and
     * forward them to the device. Used in GPU_PRODUCE_CPU_SUBMIT mode.
     * Polling-loop friendly; returns the number drained on this call.
     *
     * @param max_drain  Cap on entries processed this call. 0 = drain all.
     */
    virtual uint32_t drain_coop_channel(
        CoopIOChannel* channel,
        uint32_t       max_drain = 0) = 0;

    /**
     * Free a coop channel. Caller MUST ensure no GPU kernel is still
     * using it before calling this.
     */
    virtual void teardown_coop_channel(CoopIOChannel* channel) = 0;

    // ------------------------------------------------------------------
    // 4.5 Metadata
    // ------------------------------------------------------------------

    virtual BackendType  backend_type() const = 0;
    virtual const char*  backend_name() const = 0;
    virtual std::size_t  max_io_size()  const = 0;   // bytes
    virtual std::size_t  queue_depth()  const = 0;
    virtual std::size_t  queue_count()  const = 0;
};

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_BACKEND_PROVIDER_H__

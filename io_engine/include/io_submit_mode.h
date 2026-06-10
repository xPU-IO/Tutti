#ifndef __TUTTI_IO_ENGINE_IO_SUBMIT_MODE_H__
#define __TUTTI_IO_ENGINE_IO_SUBMIT_MODE_H__

/**
 * io_submit_mode.h -- the four submission modes the runtime exposes.
 *
 * Layer: io_engine (Backend SPI)
 *
 * Spec: doc/design/backend-spi.md §3.7 and §5.x.
 *
 * Mode rules at a glance:
 *
 *   BATCH_GPU_STREAM  -- CPU pre-stages descriptors; GPU kernel rings the
 *                        device doorbell directly. Completion is CUDA
 *                        stream-ordered. Primary high-throughput path.
 *                        REQUIRED for v0.1 backends.
 *
 *   BATCH_CPU_SYNC    -- CPU prepares + submits + blocks. No GPU kernel.
 *                        REQUIRED for v0.1 backends.
 *
 *   BATCH_CPU_ASYNC   -- CPU prepares + submits, returns IOFuture, polls
 *                        or waits later. Interface stable in v0.1;
 *                        backends MAY return nullptr if not implemented.
 *
 *   COOP              -- Shared SQ/CQ between CPU and GPU. Direction is
 *                        chosen per channel (see CoopDirection). Interface
 *                        stable; implementation deferred for local_nvme.
 *
 *   BATCH_GPU_ASYNC   -- Persistent kernel / CUDA Graphs driven IO.
 *                        OUT OF SCOPE for v0.1.
 *
 * Note: this is also the "tutti" pattern in code form. CPU and GPU each
 * play their own voice; COOP is when they share a stand.
 */

#include <cstdint>

namespace tutti {

enum class IOSubmitMode : uint32_t {
    BATCH_GPU_STREAM = 0,  // CPU stages descriptors; GPU kernel submits via doorbell
    BATCH_CPU_SYNC   = 1,  // CPU prepares + submits + blocks
    BATCH_CPU_ASYNC  = 2,  // CPU prepares + submits, returns IOFuture
    COOP             = 3,  // Shared SQ/CQ; direction set per channel
    BATCH_GPU_ASYNC  = 4,  // Persistent kernel / CUDA Graphs (out of scope v0.1)
};

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_IO_SUBMIT_MODE_H__

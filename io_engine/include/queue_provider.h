#ifndef __TUTTI_IO_ENGINE_QUEUE_PROVIDER_H__
#define __TUTTI_IO_ENGINE_QUEUE_PROVIDER_H__

/**
 * queue_provider.h -- IQueue factory; lets the caller swap queue
 *                     implementations independently of the backend.
 *
 * Layer: io_engine (Backend SPI)
 *
 * Spec: doc/design/backend-spi.md §3.3.1
 *
 * Why separate from IBackendProvider:
 *   - The same NVMe backend may want to run with libnvm-backed queues
 *     (default) OR with io_uring-backed queues (CPU-only path) OR with
 *     a mock queue (tests). Decoupling lets users inject the queue
 *     factory at construction time.
 *
 * Contract:
 *   - create_queue() honours capabilities: returns nullptr if the
 *     requested QueueCap bits cannot be satisfied.
 *   - destroy_queue() requires the queue to be quiesced (no in-flight IO).
 *   - active_count() is informational; backends may rate-limit on it.
 */

#include <cstdint>

namespace tutti {

class IQueue;

struct QueueConfig {
    uint32_t sq_depth;       // submission queue depth (power of 2)
    uint32_t cq_depth;       // completion queue depth (power of 2)
    uint32_t capabilities;   // required QueueCap bitmask
    int      device_id;      // CUDA device for GPU-visible allocation; -1 = CPU only
};

class IQueueProvider {
public:
    virtual ~IQueueProvider() = default;

    /// Allocate one queue. Returns nullptr if capabilities can't be satisfied.
    virtual IQueue* create_queue(const QueueConfig& config) = 0;

    /// Tear down a queue. Caller must ensure no in-flight IO on it.
    virtual void destroy_queue(IQueue* queue) = 0;

    /// Number of queues this provider currently has outstanding.
    virtual uint32_t active_count() const = 0;
};

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_QUEUE_PROVIDER_H__
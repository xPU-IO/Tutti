#ifndef __TUTTI_IO_ENGINE_IO_FUTURE_H__
#define __TUTTI_IO_ENGINE_IO_FUTURE_H__

/**
 * io_future.h -- async completion handle for CPU async submission.
 *
 * Layer: io_engine (Backend SPI)
 *
 * Spec: doc/design/backend-spi.md §3.5
 *
 * Role:
 *   - Returned by IBackendProvider::submit_batch_cpu_async(). The caller
 *     submits a batch and gets an IOFuture* immediately, so it can
 *     overlap CPU work with backend completion polling.
 *   - is_complete() / wait() poll the underlying queue's CQ.
 *   - cancel() is best-effort; already-submitted IOs may still complete
 *     after cancel returns.
 *
 * Ownership:
 *   - The backend allocates IOFuture; the caller owns the pointer and
 *     MUST `delete` it after use. Backends that don't support async
 *     submission return nullptr from submit_batch_cpu_async() and the
 *     caller should fall back to BATCH_CPU_SYNC.
 *
 * Buffer / descriptor lifetime contract:
 *   - The BufferDescriptor[] and IORequest[] arrays passed to
 *     submit_batch_cpu_async() MUST stay valid until is_complete()
 *     returns true. Caller responsibility, not backend's.
 */

#include <cstdint>

namespace tutti {

class IOFuture {
public:
    virtual ~IOFuture() = default;

    /**
     * Non-blocking probe. Returns true once every IO in the batch has
     * been written to the backend's CQ.
     */
    virtual bool is_complete() const = 0;

    /**
     * Block until is_complete() would return true, or until timeout_ms
     * elapses. timeout_ms == 0 blocks indefinitely. Returns true on
     * completion, false on timeout.
     */
    virtual bool wait(uint32_t timeout_ms = 0) = 0;

    /**
     * Best-effort cancel. Backend may not be able to recall a command
     * already at the device. After cancel(), is_complete() will still
     * return true once any in-flight commands drain.
     */
    virtual void cancel() = 0;

    /**
     * Total bytes the backend reports as transferred. Only meaningful
     * after is_complete() == true. May be less than the requested
     * total on partial-completion backends (rare for NVMe).
     */
    virtual uint64_t bytes_transferred() const = 0;
};

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_IO_FUTURE_H__

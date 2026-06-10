#ifndef __TUTTI_IO_ENGINE_COOP_CHANNEL_H__
#define __TUTTI_IO_ENGINE_COOP_CHANNEL_H__

/**
 * coop_channel.h -- shared-memory submit/complete channel between CPU and GPU.
 *
 * Layer: io_engine (Backend SPI)
 *
 * Spec: doc/design/backend-spi.md §3.6
 *
 * Role:
 *   - A CoopIOChannel wraps an IQueue with QUEUE_CAP_SHARED_MEM set:
 *     SQ and CQ memory is visible from both CPU and GPU at the same
 *     time. Producer and consumer roles are NOT fixed by the backend;
 *     CoopDirection picks who produces requests and who submits to the
 *     device for a given channel.
 *   - This is "tutti" submit mode: the same data structure serves both
 *     directions; the backend treats it as one ensemble where two
 *     sides hand off entries.
 *
 * Direction conventions:
 *   GPU_PRODUCE_CPU_SUBMIT
 *     - GPU kernel writes CoopSubmitEntry into sq.dev_ptr at sq_tail.
 *     - CPU proxy thread reads from sq.cpu_ptr at sq_head, submits to
 *       the device, writes the CoopCompletionEntry into cq.cpu_ptr.
 *     - GPU polls cq.dev_ptr for matching tickets.
 *     - Use case: GPU computes IO targets at runtime (GPUfs / FlashNeuron).
 *
 *   CPU_PRODUCE_GPU_SUBMIT
 *     - CPU writes CoopSubmitEntry into sq.cpu_ptr at sq_tail.
 *     - GPU kernel reads from sq.dev_ptr, submits via device queue
 *       doorbell, writes CoopCompletionEntry into cq.dev_ptr.
 *     - CPU polls cq.cpu_ptr.
 *     - Use case: CPU planner with GPU-driven device submission.
 *
 * Atomic counter contract (left to the backend / queue provider):
 *   - SQ head/tail counters MUST live in pinned memory accessible to
 *     both sides, and use cuda::atomic<uint32_t, thread_scope_system>.
 *   - Producer advances tail AFTER writing the entry (release).
 *   - Consumer reads ring[head % depth], processes, then advances head
 *     (release). CQ follows the same pattern with roles reversed.
 *   - These counters are NOT in this header; queue providers expose
 *     them via the IQueue's underlying ring memory.
 */

#include <cstdint>

#include "io_request.h"

namespace tutti {

class IQueue;

/**
 * SQ entry format for a coop channel. Must fit within
 * QueueDesc::sq.entry_sz bytes when the underlying IQueue is created.
 *
 * Sizing: 16-byte IORequest + 8-byte descriptor_idx + 8-byte ticket =
 * 32 bytes including padding. Backends MAY pad to a power-of-two for
 * cache alignment.
 */
struct CoopSubmitEntry {
    IORequest request;          // direction, offset, size, descriptor pointer
    uint64_t  descriptor_idx;   // index into the pre-registered descriptor array
    uint32_t  ticket;           // monotonic ID; CQ entry echoes this
    uint32_t  reserved;
};

/**
 * CQ entry format for a coop channel. The consumer side writes this
 * after the device reports completion; the producer side polls for
 * matching tickets.
 */
struct CoopCompletionEntry {
    uint32_t ticket;            // matches CoopSubmitEntry::ticket
    uint32_t status;            // 0 = success; backend-defined otherwise
    uint64_t bytes_done;
};

/**
 * Direction tag chosen at channel-setup time. Each channel is one
 * direction; bidirectional traffic is two channels.
 */
enum class CoopDirection : uint32_t {
    GPU_PRODUCE_CPU_SUBMIT = 0,   // GPUfs / FlashNeuron-style
    CPU_PRODUCE_GPU_SUBMIT = 1,   // CPU planner, GPU submits via doorbell
};

/**
 * The channel itself: an IQueue handle plus the chosen direction. The
 * IQueue is owned by the IBackendProvider's IQueueProvider; the channel
 * struct is just a typed view of it.
 */
struct CoopIOChannel {
    IQueue*       queue;        // must have QUEUE_CAP_SHARED_MEM
    CoopDirection direction;
};

} // namespace tutti

#endif // __TUTTI_IO_ENGINE_COOP_CHANNEL_H__

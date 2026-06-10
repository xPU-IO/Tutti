#ifndef __TUTTI_RUNTIME_DEVICE_H__
#define __TUTTI_RUNTIME_DEVICE_H__

/**
 * device.h -- runtime-level handle for one backend-managed device.
 *
 * Layer: Core Runtime (Roadmap.md §3 object model).
 *
 * Role:
 *   - A `Device` is the runtime-visible identity of one storage
 *     resource: an NVMe controller, an RDMA NIC + queue pair pool, a
 *     GDS adapter, a future RDMA-over-NVMe target, etc. Upper layers
 *     refer to devices via `device_id` and a `Device*` handle they
 *     obtain from the Runtime.
 *   - The Device is intentionally thin: identity, capabilities, and
 *     pointers to the backend pieces (IBackendProvider, IQueueProvider).
 *     Everything else (queue lifecycle, IO submission, mount paths,
 *     filesystems) lives behind the SPI or in adjacent runtime types.
 *
 * What's deliberately NOT here:
 *   - GPU bindings (those are Lease-time decisions; multiple GPUs
 *     may share one Device).
 *   - Filesystem mount paths (StorageTarget territory).
 *   - Per-queue scheduling state (Queue / IQueue territory).
 *
 * Lifetime:
 *   - The Runtime owns Device instances. Holders carry `Device*` and
 *     MUST NOT delete them. A Device's identity is stable for the
 *     life of the Runtime.
 */

#include <cstdint>
#include <string>

#include "capability_set.h"

// io_engine forward decls
#include "../../io_engine/include/backend_type.h"

namespace tutti {

class IBackendProvider;     // io_engine/include/backend_provider.h
class IQueueProvider;       // io_engine/include/queue_provider.h

/**
 * Runtime-level device descriptor. Field set is intentionally small;
 * extending requires bumping a version field on the Runtime.
 */
struct Device {
    // ---- Identity ----------------------------------------------------
    int32_t      device_id;        // dense index inside the Runtime
    BackendType  backend_type;     // which backend manages this device
    std::string  pci_addr;         // bus identifier (PCI for NVMe, IB GUID for RDMA, ...)
    std::string  display_name;     // human-readable, e.g. "Samsung 980 Pro @ 0000:50:00.0"

    // ---- Capabilities -----------------------------------------------
    CapabilitySet capabilities;

    // ---- Backend hooks ----------------------------------------------
    // Borrowed pointers; owned by the IBackendProvider implementation
    // that registered this Device with the Runtime.
    IBackendProvider* backend;
    IQueueProvider*   queues;

    // ---- Backend-private opaque payload -----------------------------
    // Backends may stash a private handle (e.g. a `Controller*` for
    // local_nvme, an `ibv_context*` for RDMA) so they can navigate
    // back from `Device` to their concrete state without a side table.
    // Upper layers MUST treat this as opaque.
    void* backend_private;
};

} // namespace tutti

#endif // __TUTTI_RUNTIME_DEVICE_H__

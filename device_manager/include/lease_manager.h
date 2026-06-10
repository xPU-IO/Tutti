#ifndef __TUTTI_DEVICE_MANAGER_LEASE_MANAGER_H__
#define __TUTTI_DEVICE_MANAGER_LEASE_MANAGER_H__

/**
 * lease_manager.h -- the Lease *lifecycle service*.
 *
 * Layer: Device Manager (Roadmap.md §3 layered architecture).
 *
 * Role:
 *   - The runtime-visible `Lease` value type lives in
 *     `runtime/include/lease.h`. This header defines the *service*
 *     that issues, refreshes, releases, and reaps those leases.
 *   - Splitting noun (Lease) from service (ILeaseManager) follows the
 *     "device_manager owns process-spanning resource management;
 *     runtime owns in-process request lifecycle" boundary. Anything
 *     that crosses processes (heartbeat RPC, PID-starttime reaper,
 *     daemon-side allocation table) lives behind this interface.
 *
 * Implementations:
 *   - `local_nvme`: NVMeService::ServiceState (in
 *     backends/local/NVMeService) is the first concrete implementer.
 *   - Future RDMA / bandwidth-quota daemons plug in here without
 *     touching upper layers.
 *
 * Why a separate header (vs reusing NVMeService's interface):
 *   - NVMeService::Allocation is a local_nvme implementation detail;
 *     ILeaseManager is the cross-backend contract. Keeping the
 *     contract in device_manager/ lets new backends implement leasing
 *     uniformly, and lets the runtime depend only on this header
 *     instead of on backend-specific types.
 *
 * Lifetime:
 *   - The lease manager owns the per-lease record and the per-lease
 *     payload pointer (`Lease::payload`) for the life of the lease.
 *   - Callers receive `Lease` values by copy and refresh / release
 *     them by `lease_id`.
 */

#include <cstdint>
#include <string>

#include "../../runtime/include/lease.h"   // struct Lease, LeaseKind

namespace tutti {

/**
 * The lease lifecycle interface. One ILeaseManager per resource pool
 * (e.g. one per NVMe controller queue pool, one per RDMA NIC QP pool).
 *
 * Concurrency:
 *   - Implementations MUST be safe to call from multiple threads.
 *   - For cross-process implementations, methods may block on RPC.
 *
 * Errors:
 *   - Methods return `false` and (optionally) populate `*error` on
 *     failure rather than throwing, to keep wire-protocol implementers
 *     honest about failure paths.
 */
class ILeaseManager {
public:
    virtual ~ILeaseManager() = default;

    /// Refresh the lease's heartbeat. Returns false (with error set
    /// if non-null) if the lease is unknown or already revoked.
    virtual bool heartbeat(const std::string& lease_id,
                            std::string*       error) = 0;

    /// Release a lease early. Returns false if the lease is unknown
    /// or `client_pid` doesn't match the recorded one (sanity check
    /// against accidental cross-process release).
    virtual bool release_lease(const std::string& lease_id,
                                uint32_t           client_pid,
                                std::string*       error) = 0;

    /// Whether `lease_id` is currently outstanding.
    virtual bool has_lease(const std::string& lease_id) const = 0;
};

} // namespace tutti

#endif // __TUTTI_DEVICE_MANAGER_LEASE_MANAGER_H__

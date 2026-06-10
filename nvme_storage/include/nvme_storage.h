#ifndef __TUTTI_NVME_STORAGE_NVME_STORAGE_H__
#define __TUTTI_NVME_STORAGE_NVME_STORAGE_H__

/**
 * nvme_storage.h -- the INvmeStorage interface.
 *
 * Layer: nvme_storage.  See doc/refactor/LegacyDecomposition.md §3.6.
 *
 * Role:
 *   - Owns named LBA ranges on top of one or more NVMe namespaces.
 *   - Knows how to mount / format / FIEMAP host filesystems on top
 *     of the snvme block device, but exposes only a flat "named
 *     file" API.  No filesystem-specific concept leaks out.
 *   - Provides host-side blocking IO (for bootstrap / metadata /
 *     tests) and -- in a follow-up R5b -- device-side submission
 *     primitives that io_engine kernels call inline.
 *
 * Layer boundary:
 *   - This header MUST NOT pull in libnvm headers; we want callers
 *     (block_storage) to consume the abstraction without leaking
 *     libnvm types.  Implementations are free to use libnvm
 *     internally.
 *   - Capacity / file directory metadata is persisted on the host
 *     filesystem (via PersistentFileLog).  That backing store is
 *     deliberately not exposed.
 *
 * Lifetime:
 *   - bootstrap() takes a list of Device*; for each device it
 *     mounts the host fs, prepares a directory, and reloads the
 *     PersistentFileLog.  Failure rolls back partial mounts.
 *   - shutdown() flushes everything and unmounts.  Idempotent.
 *
 * Multi-NVMe:
 *   - One INvmeStorage instance services multiple devices.
 *     Per-Device state is keyed off `Device*` everywhere.
 *
 * Threading:
 *   - Public methods are thread-safe; v0.1 uses a single std::mutex.
 *
 * Deferred to R5b: (DONE)
 *   - acquire_queue_pair / release_queue_pair: subsumed by giving
 *     out NvmeFileDeviceHandle that already references the
 *     controller's d_qps[] pool.  Host never touches a queue pair
 *     by handle.
 *   - __device__ submit_read_one / submit_write_one: in
 *     `nvme_storage_device.cuh`, callable from any kernel.
 *   - NvmeFileDeviceHandle: in `nvme_file_device_handle.h`,
 *     produced by acquire_device_handle() below.
 */

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <sys/types.h>     // ssize_t
#include <vector>

#include "nvme_file.h"

namespace tutti {

struct Device;
struct NvmeFileDeviceHandle;       // nvme_file_device_handle.h

class INvmeStorage {
public:
    virtual ~INvmeStorage() = default;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Mount + prepare the on-disk directory for every device.
    /// Returns false on any failure; partial state rolled back.
    /// MUST be called before any other method.
    virtual bool bootstrap(const std::vector<const Device*>& devices) = 0;

    /// Unmount everything and release internal state.  Safe to call
    /// twice.  Returns false if any unmount failed (state is still
    /// cleared on best-effort).
    virtual bool shutdown() = 0;

    // ------------------------------------------------------------------
    // Capacity (per device)
    // ------------------------------------------------------------------

    virtual uint64_t total_capacity   (const Device*) const = 0;
    virtual uint64_t available_capacity(const Device*) const = 0;

    /// Where this Device's host filesystem is mounted (so callers
    /// can place sidecar metadata under <mount>/.tutti/).  Returns
    /// empty string if `device` is not bootstrapped.  Used by
    /// block_storage (R6) to colocate gpu_file_log.bin with
    /// nvme_storage's file_log.bin under the same .tutti directory.
    virtual std::string mount_path(const Device*) const = 0;

    // ------------------------------------------------------------------
    // Directory
    // ------------------------------------------------------------------

    /// Create a named NvmeFile of `size_bytes` user-visible bytes on
    /// `device`.  Internally allocates `sizeof(NvmeFileHeader) +
    /// size_bytes` on the underlying host filesystem, embeds the
    /// header at byte 0, and stores the FIEMAP extent list in both
    /// the header and the persistent file log.
    /// Returns nullptr if a file with the same name already exists,
    /// or on allocation / FIEMAP failure.
    ///
    /// Bulk-init knobs (default = "single-file durable"):
    ///   sync_now    when true (default), fsync the host_fd twice
    ///               (after fallocate, after header pwrite).  Set
    ///               false to skip those two fsyncs -- the file's
    ///               extents and header bytes will still be in the
    ///               page cache but NOT yet on platter.  Caller
    ///               MUST follow up with `flush_metadata()` (which
    ///               internally syncfs() the mount) before relying
    ///               on durability.  Crash without that flush
    ///               leaves a ghost .bin which bootstrap reconcile
    ///               will unlink on the next start, so re-init must
    ///               be idempotent.
    ///   persist_now when true (default), the per-Device log is
    ///               rewritten and rename(2)'d after this call.  Set
    ///               false to keep the new entry only in memory; a
    ///               later `flush_metadata()` rewrites once for all
    ///               accumulated changes.  This collapses the
    ///               O(N^2) total bytes written across N create
    ///               calls into a single O(N) rewrite.
    ///
    /// Bulk-init pattern for, e.g., LMCache provisioning of millions
    /// of pre-allocated KV-shard files:
    /// @code
    /// for (auto& s : shards)
    ///     storage->create_file(dev, s.name, s.size,
    ///                          /*persist_now=*/false,
    ///                          /*sync_now=*/false);
    /// storage->flush_metadata(dev);   // one syncfs + one log rewrite
    /// @endcode
    virtual NvmeFile* create_file(const Device*  device,
                                   std::string_view name,
                                   uint64_t        size_bytes,
                                   bool            persist_now = true,
                                   bool            sync_now    = true) = 0;

    /// Flush all deferred metadata for `device`:
    ///   - syncfs() the mount so any data/extents written by
    ///     `create_file(..., sync_now=false)` since the last flush
    ///     reach the platter.
    ///   - rewrite + fsync + rename the PersistentFileLog so any
    ///     entries added by `create_file(..., persist_now=false)`
    ///     since the last flush become durable.
    /// No-op if nothing is pending; safe to call repeatedly.
    /// Returns false if either step fails (state still considered
    /// dirty, retry is OK).
    virtual bool      flush_metadata(const Device* device) = 0;

    /// Re-open an existing file (host fd reopened).  Returns nullptr
    /// if not found.
    virtual NvmeFile* open_file(const Device*    device,
                                 std::string_view name) = 0;

    /// fsync the host fd, close it, and persist the file log.  The
    /// NvmeFile pointer remains valid for re-open / list_files but
    /// its host_fd becomes < 0.
    virtual bool      close_file(NvmeFile* file) = 0;

    /// Remove the file from the directory + delete the underlying
    /// host file.  Returns false if not found or unlink fails.
    ///
    /// `persist_now` (default true) controls whether the per-Device
    /// log is rewritten before this call returns.  Set false for
    /// bulk deletion (e.g. fsck-style cleanup of millions of stale
    /// files), then call `flush_metadata(device)` once at the end
    /// to land all log changes in a single rewrite.  The on-disk
    /// `<name>.bin` file IS unlinked synchronously regardless of
    /// this flag -- only the log update is deferred.
    virtual bool      delete_file(NvmeFile* file,
                                  bool      persist_now = true) = 0;

    /// All currently-OPEN NvmeFiles for `device` (i.e. entries that
    /// have an active host_fd in this process).  This is NOT the same
    /// as "every file in the persistent directory": entries on disk
    /// that have never been opened in this process do NOT appear.
    /// Use `list_file_names()` for the directory-wide view.
    virtual std::vector<NvmeFile*> list_files(const Device*) const = 0;

    /// All file names known to this device's PersistentFileLog,
    /// regardless of open-status in this process.  Useful for bulk
    /// cleanup / fsck / migration: caller can `open_file(name)` each
    /// then `delete_file()` or whatever.  Order is the log's
    /// insertion order.  No host_fd is opened by this call.
    virtual std::vector<std::string> list_file_names(const Device*) const = 0;

    // ------------------------------------------------------------------
    // Host-side blocking IO (R5a)
    //
    // Useful for:
    //   - bootstrap / metadata writes that don't need GPU paths
    //   - the smoke test (verify byte content roundtrip)
    //   - cooperative CPU side of mixed CPU+GPU IO
    //
    // Implementation walks pread/pwrite over the host fd.  It is
    // NOT the high-throughput path -- that's the GPU device-side
    // submit (R5b).
    //
    // `byte_offset` is the LOGICAL offset (excludes the 4 KiB
    // NvmeFileHeader at byte 0).  The implementation adjusts.
    // ------------------------------------------------------------------

    virtual ssize_t read_blocking (NvmeFile* file, uint64_t byte_offset,
                                    void*        dst, size_t len) = 0;
    virtual ssize_t write_blocking(NvmeFile* file, uint64_t byte_offset,
                                    const void* src, size_t len) = 0;

    /// fsync the file (data + metadata).  Useful when callers want
    /// a flush point without closing.
    virtual bool    sync(NvmeFile* file) = 0;

    // ------------------------------------------------------------------
    // GPU device-side submit (R5b)
    //
    // These DO NOT open or create a file -- the file is already in the
    // directory (created via `create_file` / re-opened via `open_file`
    // above).  Acquire / release a GPU-side *view* (a small POD living
    // in GPU memory) that lets a kernel submit reads/writes against
    // the file's LBA extents through the NVMe queue group's d_qps.
    // Naming intentionally avoids "open" so this isn't confused with
    // the directory-level open_file.
    //
    // acquire_device_handle   cudaMalloc + cudaMemcpy a
    //                         NvmeFileDeviceHandle onto the file's
    //                         owning device's GPU.  The returned
    //                         pointer lives in GPU memory; pass it to
    //                         a kernel and call submit_read_one /
    //                         submit_write_one on it (see
    //                         nvme_storage_device.cuh).
    //
    //                         Returns nullptr if:
    //                           - the device has no NvmeQueueGroup
    //                             (its registry was opened with
    //                             build_queue_group=false).
    //                           - cudaMalloc / cudaMemcpy fails.
    //
    // release_device_handle   cudaFree the handle.  No-op on
    //                         nullptr.  Idempotent.
    //
    // Lifetime: the handle is valid as long as the underlying
    // NvmeFile is alive AND the device's queue_group is alive.  In
    // practice that means: don't outlive the storage subsystem
    // shutdown.
    //
    // R6 note: the unified "POSIX-style open that yields both host
    // and GPU views" lives at the block_storage layer; this acquire
    // pair is its building block.
    // ------------------------------------------------------------------

    virtual NvmeFileDeviceHandle* acquire_device_handle (NvmeFile* file)             = 0;
    virtual void                  release_device_handle(NvmeFileDeviceHandle* dh)   = 0;
};

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_NVME_STORAGE_H__

#ifndef __TUTTI_BLOCK_STORAGE_BLOCK_STORAGE_H__
#define __TUTTI_BLOCK_STORAGE_BLOCK_STORAGE_H__

/**
 * block_storage.h -- the IBlockStorage interface (R6).
 *
 * Layer: block_storage.  Sits *above* nvme_storage and *below*
 * io_engine.  See doc/refactor/LegacyDecomposition.md §3.4.
 *
 * Role:
 *   - Owns the "GpuFile" abstraction: a named logical container
 *     whose data plane is split across N NvmeFile shards (one per
 *     entry in `shard_placement`, currently 1:1 with Devices).
 *   - Persists GpuFile metadata (id, name, tensor_shape, shard
 *     names) per device in `<mount>/.tutti/gpu_file_log.bin`.
 *     Every device that hosts at least one shard carries a full
 *     copy of that log so a single-disk loss does not erase the
 *     directory view.
 *   - Provides bulk-init paths that delegate to nvme_storage's
 *     bulk-init flags (create_file persist_now=false, sync_now=false
 *     followed by flush_metadata) so creating millions of GpuFiles
 *     at LMCache provisioning time stays sub-minute.
 *   - Provides `acquire_device_handle` / `release_device_handle` to
 *     bring all N shard NvmeFiles to GPU together; the returned
 *     GpuFileHandle is what io_engine (R8) consumes when it
 *     pre-stages a kernel batch (one GPUIoContext per IO; each ctx
 *     references handle->d_shards_dev so the kernel reads it
 *     directly without chasing host pointers).
 *
 * Layer boundary:
 *   - This header MUST NOT pull in libnvm or CUDA headers; we want
 *     io_engine and smoke code to see a small, libnvm-free surface.
 *     (NvmeFileDeviceHandle is forward-declared; its full layout
 *     lives in nvme_storage/include/nvme_file_device_handle.h.)
 *   - The IO-submission hot path is NOT here.  This header only
 *     gives you metadata + acquired handles; the kernel-side
 *     translation/submit step lives in io_engine (R8) on top of
 *     gpu_file_resolve.h + nvme_storage_device.cuh.
 *
 * Lifetime:
 *   - bootstrap() takes a non-owning `INvmeStorage*` and the device
 *     list to participate in.  It loads each device's
 *     gpu_file_log.bin (a generation-number-arbitrated mirror) and
 *     reconciles it against nvme_storage's .bin directory.
 *   - shutdown() closes all open GpuFiles and persists the log on
 *     every device.  Idempotent.
 *
 * Multi-NVMe placement (v0.1):
 *   - GpuFileSpec.shard_placement explicitly enumerates which
 *     Device each shard lives on.  size() must equal
 *     tensor_shape[0].  v0.1 is 1:1 -- one shard per device, no
 *     two shards on the same device per file.  Higher-level
 *     placement policies (e.g. "balance shards across devices to
 *     fill the lowest-utilised pair first") are caller-side
 *     concerns above this layer.
 *
 * Threading:
 *   - Public methods are thread-safe; v0.1 uses a single std::mutex.
 *
 * The "block" name is historical (LegacyDecomposition.md §3.4).  In
 * the real world it always represents a GPU file; the surface
 * deliberately uses GpuFile / GpuFileHandle rather than Block /
 * BlockHandle.
 */

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tutti {

struct Device;                  // device_manager
struct NvmeFile;                // nvme_storage/include/nvme_file.h
struct NvmeFileDeviceHandle;    // nvme_storage/include/nvme_file_device_handle.h
class  INvmeStorage;            // nvme_storage/include/nvme_storage.h

using GpuFileId = uint32_t;

/// Hard upper bound on the number of shards (== tensor_shape[0])
/// per GpuFile.  Set to 4 because GPUs are typically PCIe x16 and
/// modern NVMe are PCIe x4, so four parallel queues already saturate
/// the GPU's host link.  Higher values may need scattergather changes
/// in io_engine.  Bumping this requires re-running every smoke and
/// extending OnDiskGpuFileEntry::per_shard_names accordingly (it is
/// length-prefixed in the on-disk format so this is mechanical, not
/// an ABI break -- but it IS a re-test-everything event).
constexpr uint32_t kGpuFileMaxShards = 4;

/// Description used to create a new GpuFile.
struct GpuFileSpec {
    /// Persistent directory name; unique within the storage.
    std::string_view name;

    /// Total user-visible bytes.  Must equal
    ///   tensor_shape[0] * tensor_shape[1] * tensor_shape[2].
    /// Stored as the authoritative size; tensor_shape is metadata
    /// for callers (LMCache, etc.) and the dimensional check.
    uint64_t total_size;

    /// Three-tuple describing the GpuFile content layout.  Tutti's
    /// hot path only reads tensor_shape[0] (== num_shards) and
    /// tensor_shape[2] (== tensor_size, the per-tensor interleave
    /// unit).  The middle dimension tensor_shape[1] is metadata
    /// only -- typical values are "layers per shard" for KV-cache,
    /// but Tutti does not interpret it.
    ///
    /// tensor_shape[0] = num_shards         (in [1, kGpuFileMaxShards])
    /// tensor_shape[1] = layers_per_shard   (>= 1, opaque to runtime)
    /// tensor_shape[2] = tensor_size_bytes  (must divide total_size)
    ///
    /// Note: tensor_size is the LOGICAL stripe / interleave unit at
    /// the GpuFile layer; it is NOT the NVMe-level IO size.  When
    /// tensor_size exceeds the device's MDTS (Max Data Transfer
    /// Size, typ. 128 KiB), `memory/` (R7) will fragment each
    /// tensor at register_tensor() time into ceil(tensor_size /
    /// MDTS) PRPMappingEntry sub-slices, and `io_engine/` (R8)
    /// will issue that many GPUIoContext entries per tensor at
    /// submit time.  This layer does not need to know about that
    /// fan-out: gpu_file_resolve() returns the shard's BASE byte
    /// offset for the tensor, and R8 adds each sub-slice's
    /// intra-tensor offset on top.
    uint32_t tensor_shape[3];

    /// Where each shard's backing NvmeFile lives.  Element i is the
    /// device for shard i.  size() must equal tensor_shape[0].
    /// v0.1: pointers must be distinct (1:1 shard-to-device).
    std::vector<const Device*> shard_placement;
};

/// Host-side runtime view of an open GpuFile.
///
/// Lifetime:
///   - Created by IBlockStorage::create_gpu_file / open_gpu_file.
///   - Borrowed only; callers MUST NOT delete -- the storage owns
///     it and reclaims on close_gpu_file / shutdown.
///   - shards[] are NvmeFile* borrowed from nvme_storage (same
///     reference-borrow rules: don't outlive INvmeStorage::shutdown).
struct GpuFile {
    GpuFileId   id;
    std::string name;

    uint64_t    total_size;
    uint32_t    tensor_shape[3];

    /// Borrowed NvmeFile* per shard, parallel to
    /// GpuFileSpec::shard_placement.  size() == tensor_shape[0].
    std::vector<NvmeFile*> shards;
};

/// "Acquired" GpuFile: every shard has been brought to GPU.
///
/// Allocated by IBlockStorage::acquire_device_handle.
/// Released by IBlockStorage::release_device_handle (which also
/// cudaFrees `d_shards_dev` and calls release_device_handle on each
/// underlying NvmeFile).
///
/// The struct itself lives in *host* memory.  io_engine consumes
/// it during host-side per-IO context staging (see
/// LegacyDecomposition.md §R8): for each IO, host code does
///   gpu_file_resolve(handle->tensor_size, handle->num_shards,
///                    byte_off, &si, &so);
///   ctx.d_shard         = handle->d_shards_host[si];
///   ctx.shard_byte_off  = so;
///   ctx.length          = handle->tensor_size;
/// then cudaMemcpyAsync the ctx[] array onto GPU and launches one
/// thread per IO.  The kernel never touches `d_shards_dev`
/// directly; it only dereferences a single
/// `NvmeFileDeviceHandle*` per ctx.  We still expose
/// `d_shards_dev` as a convenience for any future kernel that
/// wants to do its own resolve on-device with a single ctx
/// describing the whole file.
struct GpuFileHandle {
    /// Borrowed back-pointer.  Keeps name / total_size / shards
    /// reachable for diagnostics; do not free.
    GpuFile* file;

    /// Cached for fast access (same values as
    /// `file->tensor_shape[2]` and `file->tensor_shape[0]`).
    ///
    /// tensor_size here is the logical stripe unit, NOT MDTS.
    /// io_engine (R8) computes per-IO sub-slice sizes from R7's
    /// PRPMappingEntry tables, not from this field; this field's
    /// only role on the hot path is the `gpu_file_resolve`
    /// computation that picks a shard.
    uint32_t tensor_size;
    uint32_t num_shards;

    /// Host-side table of GPU-resident shard handles, in the same
    /// order as `file->shards`.  Element i was produced by
    /// INvmeStorage::acquire_device_handle(file->shards[i]) and
    /// MUST be released via the matching release at handle teardown.
    ///
    /// Each pointer itself lives in GPU memory (cudaMalloc'd by
    /// nvme_storage); do NOT dereference on host.  The vector that
    /// holds them is host-resident.
    ///
    /// Hot-path usage (mirrors the legacy gpu_controller.cu design):
    /// io_engine (R8) host-stages a per-IO context array; for each
    /// IO it resolves shard_idx via gpu_file_resolve() and copies
    /// d_shards_host[shard_idx] *into the context* before
    /// cudaMemcpy'ing the array to GPU.  The kernel therefore only
    /// dereferences a single NvmeFileDeviceHandle* per IO -- no
    /// indirection through a GPU-resident shard table.  This is both
    /// simpler and avoids an extra GPU load per IO compared with a
    /// `d_shards_dev[shard_idx]` lookup.
    std::vector<NvmeFileDeviceHandle*> d_shards_host;
};

class IBlockStorage {
public:
    virtual ~IBlockStorage() = default;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Bind to an already-bootstrapped INvmeStorage and reload the
    /// GpuFile directory from each device's gpu_file_log.bin.
    ///
    /// `storage` MUST outlive this IBlockStorage and MUST already
    /// have called bootstrap(devices) for every device passed here.
    /// `devices` is the subset that may host GpuFile shards; v0.1
    /// behavior is "every device hosts a copy of the gpu_file_log".
    ///
    /// Returns false on any failure; partial state rolled back.
    virtual bool bootstrap(INvmeStorage*                       storage,
                           const std::vector<const Device*>&   devices) = 0;

    /// Close all open GpuFiles + persist + flush.  Safe to call
    /// twice.  Returns false on best-effort partial failure.
    virtual bool shutdown() = 0;

    // ------------------------------------------------------------------
    // Directory
    // ------------------------------------------------------------------

    /// Create a new GpuFile.  Allocates one shard NvmeFile per
    /// entry in spec.shard_placement using the underlying
    /// INvmeStorage; spec.tensor_shape consistency is checked
    /// before any nvme_storage state is touched.
    ///
    /// Bulk-init knob (default = single-file durable):
    ///   persist_now == false defers the gpu_file_log rewrite AND
    ///   passes persist_now/sync_now=false through to every
    ///   underlying create_file.  Caller MUST follow up with
    ///   flush_metadata() so both layers' logs and the host fs
    ///   reach the platter.  Crash without that flush leaves
    ///   ghost shards which bootstrap reconcile will unlink.
    ///
    /// Returns nullptr on:
    ///   - invalid spec (tensor_shape inconsistency, shard count
    ///     out of range, duplicate device in shard_placement, etc.)
    ///   - name collision
    ///   - any underlying create_file failure (rolls back all
    ///     already-created shards).
    ///
    /// Bulk-init pattern for LMCache (millions of KV blocks):
    /// @code
    /// for (auto& s : specs)
    ///     bs->create_gpu_file(s, /*persist_now=*/false);
    /// bs->flush_metadata();
    /// @endcode
    virtual GpuFile* create_gpu_file(const GpuFileSpec& spec,
                                     bool persist_now = true) = 0;

    /// Re-open an existing GpuFile by name; opens every shard's
    /// NvmeFile (host fd reopened) via INvmeStorage::open_file.
    /// Returns nullptr if not found.
    virtual GpuFile* open_gpu_file(std::string_view name) = 0;

    /// Close the GpuFile: closes every shard via
    /// INvmeStorage::close_file, persists the gpu_file_log on
    /// every device that hosts a shard.  The GpuFile* remains
    /// valid for re-open via list_gpu_file_names + open_gpu_file
    /// but its shards' host_fds become < 0.
    virtual bool close_gpu_file(GpuFile* file) = 0;

    /// Remove the GpuFile from the directory + delete every shard.
    ///
    /// `persist_now` (default true) controls whether each device's
    /// gpu_file_log is rewritten before this call returns.  Set
    /// false for bulk deletion + finish with flush_metadata().
    /// The on-disk shard `.bin` files are unlinked synchronously
    /// regardless.
    virtual bool delete_gpu_file(GpuFile* file,
                                 bool persist_now = true) = 0;

    /// Flush all deferred metadata across all participating
    /// devices: rewrites every dirty gpu_file_log.bin AND calls
    /// INvmeStorage::flush_metadata on every device so any
    /// deferred shard creates / deletes also land.  Safe to call
    /// repeatedly.  Returns false if any sub-flush fails (state
    /// stays dirty; retry is OK).
    virtual bool flush_metadata() = 0;

    /// All GpuFile names known to the directory across all
    /// participating devices.  No host_fd is opened by this call.
    virtual std::vector<std::string> list_gpu_file_names() const = 0;

    /// All currently-OPEN GpuFiles.  Same semantics as
    /// INvmeStorage::list_files: only entries with at least one
    /// open shard show up; use list_gpu_file_names for the
    /// directory-wide view.
    virtual std::vector<GpuFile*> list_open_gpu_files() const = 0;

    // ------------------------------------------------------------------
    // GPU acquire (R6)
    //
    // These DO NOT open or create the GpuFile -- the file must
    // already be in the directory.  They produce a host-side
    // "acquired" handle that bundles per-shard NvmeFileDeviceHandle*
    // and a small GPU-resident pointer table (d_shards_dev).
    //
    // Naming intentionally avoids "open" so this isn't confused
    // with the directory-level open_gpu_file.
    //
    // Semantics:
    //   acquire_device_handle   for each shard NvmeFile, calls the
    //                           underlying INvmeStorage's
    //                           acquire_device_handle; cudaMallocs
    //                           a pointer table of size num_shards
    //                           on the file's GPU and copies the
    //                           per-shard handle pointers into it.
    //
    //                           Returns nullptr if any underlying
    //                           acquire fails (with rollback of the
    //                           ones already done).
    //
    //   release_device_handle   calls
    //                           INvmeStorage::release_device_handle
    //                           on every shard handle, cudaFrees
    //                           d_shards_dev, deletes the
    //                           GpuFileHandle.  No-op on nullptr.
    //                           Idempotent.
    //
    // Lifetime: the returned handle is valid as long as every
    // shard's NvmeFile is alive AND its device's queue_group is
    // alive.  In practice: don't outlive close_gpu_file or
    // INvmeStorage::shutdown.
    // ------------------------------------------------------------------

    virtual GpuFileHandle* acquire_device_handle (GpuFile* file)         = 0;
    virtual void           release_device_handle(GpuFileHandle* handle)  = 0;
};

} // namespace tutti

#endif // __TUTTI_BLOCK_STORAGE_BLOCK_STORAGE_H__

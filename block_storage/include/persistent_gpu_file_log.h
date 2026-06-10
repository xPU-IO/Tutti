#ifndef __TUTTI_BLOCK_STORAGE_PERSISTENT_GPU_FILE_LOG_H__
#define __TUTTI_BLOCK_STORAGE_PERSISTENT_GPU_FILE_LOG_H__

/**
 * persistent_gpu_file_log.h -- per-Device on-disk directory of GpuFiles.
 *
 * Layer: block_storage internal.  Each Device that participates in a
 * BlockStorage carries a *complete* copy of the GpuFile directory in
 *   <mount_path>/.tutti/gpu_file_log.bin
 * so the loss of any single drive does not erase the directory view.
 *
 * On bootstrap we read every device's copy and pick the one with the
 * highest `generation` as the source of truth (any laggers are
 * rewritten to match on the next flush_metadata).
 *
 * Layout (bin):
 *
 *   struct OnDiskHeader {
 *       u64 magic;            // 'TUTTGPU\0' little-endian
 *       u32 version;          // 1
 *       u32 entry_count;
 *       u64 generation;       // monotonic, incremented per persist
 *       u32 next_file_id;
 *       u32 reserved[5];
 *   };
 *
 *   struct OnDiskEntry {
 *       u32 file_id;
 *       u32 num_shards;       // <= kGpuFileMaxShards (4)
 *       u64 total_size;
 *       u32 tensor_shape[3];
 *       u32 reserved0;
 *       char name[128];
 *       i32 shard_device_ids[kGpuFileMaxShards];   // [4]
 *       u32 reserved1[4];
 *   };
 *
 * Shard NvmeFile names are derived deterministically as
 *   "<gpu_file_name>.s<shard_idx>"
 * so the on-disk entry does NOT need to store them.  This keeps the
 * entry small (~192 bytes) -- millions of entries fit in tens of MiB.
 *
 * Persistence model:
 *   - Whole-file rewrite on persist() (tmp -> fsync -> rename),
 *     same shape as nvme_storage's PersistentFileLog.  R5a.1's
 *     bulk-init pattern (deferred persist + single flush at end)
 *     applies: see HostFsBackedBlockStorage::flush_metadata.
 *
 * Threading:
 *   - Caller (HostFsBackedBlockStorage) holds an outer mutex.
 *     PersistentGpuFileLog itself is single-threaded by contract.
 */

#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "block_storage.h"   // kGpuFileMaxShards

namespace tutti {

class PersistentGpuFileLog {
public:
    struct Entry {
        uint32_t                 file_id;
        std::string              name;
        uint64_t                 total_size;
        uint32_t                 tensor_shape[3];
        uint32_t                 num_shards;
        // device_id of each shard (parallel to GpuFile::shards).
        // size() == num_shards.
        std::vector<int32_t>     shard_device_ids;
    };

    PersistentGpuFileLog();
    ~PersistentGpuFileLog();

    PersistentGpuFileLog(const PersistentGpuFileLog&)            = delete;
    PersistentGpuFileLog& operator=(const PersistentGpuFileLog&) = delete;

    /// Open <mount>/.tutti/gpu_file_log.bin.  ENOENT is treated as
    /// "fresh log".  Returns false on any other IO error.
    bool load_or_init(std::string log_path);

    /// Generation number read from disk on load, or 0 for a fresh
    /// log.  Used by HostFsBackedBlockStorage::bootstrap to pick the
    /// freshest copy across devices.
    uint64_t generation() const { return generation_; }

    /// Replace in-memory state with `other` and bump our generation
    /// to other.generation_ (used when bootstrap finds a staler
    /// copy on this device than peers).  The next persist() will
    /// also bump generation by 1.
    void overwrite_from(const PersistentGpuFileLog& other);

    const std::vector<Entry>& entries() const { return entries_; }

    /// Lookup by name.  Returns nullptr if not present.
    const Entry* find_by_name(const std::string& name) const;

    /// Allocate a fresh file_id (monotonic per log).
    uint32_t next_file_id();

    /// Add a fully-populated entry.  Returns false if `name`
    /// already exists.
    bool add(Entry e);

    /// Remove by file_id.  Returns false if not present.
    bool remove(uint32_t file_id);

    /// Persist current in-memory state.  Atomic via rename().
    /// Bumps generation by 1 on success.
    bool persist();

    std::size_t size() const { return entries_.size(); }

private:
    bool load_locked();
    bool persist_locked();

    std::string                                  log_path_;
    std::vector<Entry>                           entries_;
    std::unordered_map<std::string, std::size_t> name_to_index_;
    uint32_t                                     next_file_id_ = 1;
    uint64_t                                     generation_   = 0;
};

} // namespace tutti

#endif // __TUTTI_BLOCK_STORAGE_PERSISTENT_GPU_FILE_LOG_H__

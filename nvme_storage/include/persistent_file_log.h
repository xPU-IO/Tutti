#ifndef __TUTTI_NVME_STORAGE_PERSISTENT_FILE_LOG_H__
#define __TUTTI_NVME_STORAGE_PERSISTENT_FILE_LOG_H__

/**
 * persistent_file_log.h -- per-Device on-disk directory of NvmeFiles.
 *
 * Layer: nvme_storage internal.  Replaces legacy FileManager from
 * filesystems/ext4/libgeminifs/nvme_file.cpp.
 *
 * Role:
 *   - For one Device, persists the (name -> NvmeFile metadata)
 *     mapping so the daemon / runtime can recover the directory
 *     across restarts.
 *   - Stored as a single binary file under
 *     <mount_path>/.tutti/file_log.bin
 *
 * Layout (bin):
 *
 *   struct LogHeader {
 *       u64 magic;                     // 'TUTITLOG' little-endian
 *       u32 version;                   // 1
 *       u32 entry_count;
 *       // entries[entry_count] follow
 *   };
 *
 *   struct Entry {
 *       u64 file_id;
 *       u64 size_bytes;
 *       u32 num_extents;
 *       u32 reserved;
 *       char name[128];
 *       LbaExtent extents[num_extents];
 *   };
 *
 * Persistence model:
 *   - Append-on-create, edit-on-delete, but the WHOLE file is
 *     rewritten on every persist() call to keep recovery simple.
 *     This matches legacy semantics; can move to journaled appends
 *     later if write amplification becomes an issue.
 *
 * Threading:
 *   - Caller (HostFsBackedNvmeStorage) holds an outer mutex.
 *     PersistentFileLog itself is single-threaded by contract.
 */

#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "lba_extent.h"

namespace tutti {

class PersistentFileLog {
public:
    struct Entry {
        uint64_t              file_id;
        std::string           name;
        uint64_t              size_bytes;
        std::vector<LbaExtent> extents;
    };

    PersistentFileLog();
    ~PersistentFileLog();

    PersistentFileLog(const PersistentFileLog&)            = delete;
    PersistentFileLog& operator=(const PersistentFileLog&) = delete;

    /// Path to the .tutti/file_log.bin file we read/write.
    /// load_or_init returns false on IO error other than ENOENT
    /// (a missing file is treated as "fresh log, empty").
    bool load_or_init(std::string log_path);

    /// All entries currently in memory.  Order is insertion order.
    const std::vector<Entry>& entries() const { return entries_; }

    /// Lookup by name.  Returns nullptr if not present.
    const Entry* find_by_name(const std::string& name) const;

    /// Allocate a fresh file_id (monotonic per log).
    uint64_t next_file_id();

    /// Add or replace.  Caller passes a fully-populated Entry.
    /// Returns false if name already exists (use update for that).
    bool add(Entry e);

    /// Remove by file_id.  Returns false if not present.
    bool remove(uint64_t file_id);

    /// Persist current in-memory state to disk.  Atomic via
    /// rename(tmp -> final).  Returns false on IO error.
    bool persist();

    /// Number of entries.
    std::size_t size() const { return entries_.size(); }

private:
    bool load_locked();
    bool persist_locked();

    std::string                          log_path_;
    std::vector<Entry>                   entries_;
    std::unordered_map<std::string, std::size_t> name_to_index_;
    uint64_t                             next_file_id_ = 1;
};

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_PERSISTENT_FILE_LOG_H__

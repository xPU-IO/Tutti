#ifndef __TUTTI_BLOCK_STORAGE_HOST_FS_BACKED_BLOCK_STORAGE_H__
#define __TUTTI_BLOCK_STORAGE_HOST_FS_BACKED_BLOCK_STORAGE_H__

/**
 * host_fs_backed_block_storage.h -- v0.1 IBlockStorage impl.
 *
 * Layered on top of an INvmeStorage instance; per-Device GpuFile
 * directory persisted as `<mount>/.tutti/gpu_file_log.bin` using
 * PersistentGpuFileLog (one full copy on every participating
 * Device, generation-arbitrated at bootstrap).
 *
 * v0.1 limitations:
 *   - GPU acquire/release (R6.3) is implemented as a stub that
 *     returns nullptr.  R6.2 only ships host-side directory + IO
 *     plumbing.  Smokes that need GPU submit go via
 *     nvme_storage_gpu_smoke for now.
 *   - shard placement is strictly 1:1 (no two shards on the same
 *     device per file).
 */

#include "block_storage.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tutti {

class PersistentGpuFileLog;

class HostFsBackedBlockStorage : public IBlockStorage {
public:
    HostFsBackedBlockStorage();
    ~HostFsBackedBlockStorage() override;

    HostFsBackedBlockStorage(const HostFsBackedBlockStorage&)            = delete;
    HostFsBackedBlockStorage& operator=(const HostFsBackedBlockStorage&) = delete;

    bool bootstrap(INvmeStorage*                       storage,
                   const std::vector<const Device*>&   devices) override;
    bool shutdown() override;

    GpuFile* create_gpu_file(const GpuFileSpec& spec,
                             bool persist_now = true) override;
    GpuFile* open_gpu_file  (std::string_view name) override;
    bool     close_gpu_file (GpuFile* file)         override;
    bool     delete_gpu_file(GpuFile* file,
                             bool persist_now = true) override;
    bool     flush_metadata ()                      override;

    std::vector<std::string> list_gpu_file_names() const override;
    std::vector<GpuFile*>    list_open_gpu_files()  const override;

    GpuFileHandle* acquire_device_handle (GpuFile* file)         override;
    void           release_device_handle(GpuFileHandle* handle)  override;

private:
    // Per-Device state: one full mirror of the directory + the path
    // we read/write.  Every Device passed into bootstrap() gets one.
    struct PerDeviceState {
        const Device*                          device = nullptr;
        std::string                            log_path;     // <mount>/.tutti/gpu_file_log.bin
        std::unique_ptr<PersistentGpuFileLog>  log;
        bool                                   dirty = false; // pending persist
    };

    // ---- helpers ----
    PerDeviceState* find_state(const Device*) const;
    PerDeviceState* find_state(int32_t device_id) const;

    // Reconcile pass at bootstrap (nvme_storage R5a.0 analogue):
    //   - for each entry in the merged log, ensure every shard's
    //     <gpu_name>.s<i> NvmeFile exists on its device; if any is
    //     missing the entry is dropped (tombstone).
    //   - for every NvmeFile whose name matches "*.s<digit>" but
    //     does NOT correspond to a live entry, unlink it (ghost).
    bool reconcile_locked_();

    // Push the freshest log to every device that is staler.
    void redistribute_logs_locked_(uint64_t target_generation);

    // Build the deterministic shard NvmeFile name "<gpu_name>.s<i>".
    static std::string shard_name_(std::string_view gpu_name, uint32_t i);

    // Try to parse "*.s<digit>" -> (gpu_name, shard_idx).  Returns
    // false if the name doesn't match the convention.
    static bool parse_shard_name_(std::string_view nvme_name,
                                  std::string*     out_gpu_name,
                                  uint32_t*        out_shard_idx);

    // Validate spec.tensor_shape vs total_size + shard_placement.
    bool validate_spec_(const GpuFileSpec&) const;

    // Mark every device dirty (called after any successful
    // mutation; flush_metadata clears).
    void mark_all_dirty_locked_();

    // ---- state ----
    mutable std::mutex                              mtx_;
    INvmeStorage*                                   storage_ = nullptr;
    bool                                            is_open_ = false;

    // PerDeviceState in bootstrap order.  Indexed by device_id ->
    // looked up via id_to_state_; iteration goes in vector order.
    std::vector<std::unique_ptr<PerDeviceState>>    states_;

    // Live GpuFile* keyed by file_id.  We own the unique_ptrs.
    std::map<uint32_t, std::unique_ptr<GpuFile>>    files_;
};

} // namespace tutti

#endif // __TUTTI_BLOCK_STORAGE_HOST_FS_BACKED_BLOCK_STORAGE_H__

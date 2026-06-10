#ifndef __TUTTI_NVME_STORAGE_HOST_FS_BACKED_NVME_STORAGE_H__
#define __TUTTI_NVME_STORAGE_HOST_FS_BACKED_NVME_STORAGE_H__

/**
 * host_fs_backed_nvme_storage.h -- the v0.1 INvmeStorage impl.
 *
 * Layer: nvme_storage.
 *
 * Strategy:
 *   - For every Device passed to bootstrap(), the storage:
 *       1. Picks `<mount_root>/<snvme_minor_dir>` as the host
 *          mount point (e.g. /mnt/tutti/snvme0).
 *       2. If the underlying snvme block device is unformatted,
 *          mkfs.ext4 it (one-shot).
 *       3. mount(2) it read+write.
 *       4. mkdir(2) `<mount>/.tutti/`
 *       5. PersistentFileLog::load_or_init(<mount>/.tutti/file_log.bin)
 *
 *   - create_file:
 *       1. open(<mount>/.tutti/<name>.bin, O_CREAT|O_RDWR)
 *       2. fallocate(size + sizeof(NvmeFileHeader))
 *       3. fsync
 *       4. read_extents(fd) -> LbaExtent[]
 *       5. write NvmeFileHeader at byte 0
 *       6. fsync
 *       7. PersistentFileLog::add() + persist()
 *       8. return NvmeFile*
 *
 *   - read/write/sync go through the host fd (pread/pwrite/fsync).
 *     `byte_offset` is logical (relative to byte 0 of user data),
 *     so we add sizeof(NvmeFileHeader) before pread/pwrite.
 *
 *   - shutdown:
 *       1. fsync + close any still-open NvmeFiles.
 *       2. PersistentFileLog::persist() one last time.
 *       3. umount(2) every mount in reverse order.
 *
 * v0.1 limitations:
 *   - Hardcoded ext4 (mkfs.ext4 / mount type=ext4).  Other host
 *     filesystems work in principle (FIEMAP is generic VFS) but
 *     are not exercised here.
 *   - No quota; create_file just lets fallocate spit ENOSPC if
 *     the device fills up.
 *   - No file rename; delete + create.
 *   - No directory hierarchy; flat namespace under .tutti/.
 *   - Single std::mutex around all public methods.  Fine for
 *     bootstrap / smoke; revisit when block_storage starts
 *     hammering.
 */

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "nvme_storage.h"

namespace tutti {

class PersistentFileLog;

class HostFsBackedNvmeStorage : public INvmeStorage {
public:
    struct Config {
        // Where to mkdir per-device mount points.  The storage will
        // mkdir(`<root>/snvme<minor>`) and mount the block device on it.
        std::string mount_root = "/mnt/tutti";
        // Whether to mkfs.ext4 if the underlying block device has no
        // recognised filesystem.  Set false in production deployments
        // where ops pre-formats the disks.
        bool        auto_mkfs = true;
    };

    explicit HostFsBackedNvmeStorage(Config cfg);
    HostFsBackedNvmeStorage();              // uses Config{} defaults
    ~HostFsBackedNvmeStorage() override;

    HostFsBackedNvmeStorage(const HostFsBackedNvmeStorage&)            = delete;
    HostFsBackedNvmeStorage& operator=(const HostFsBackedNvmeStorage&) = delete;

    // ---- INvmeStorage --------------------------------------------------

    bool bootstrap(const std::vector<const Device*>& devices) override;
    bool shutdown() override;

    uint64_t total_capacity   (const Device*) const override;
    uint64_t available_capacity(const Device*) const override;
    std::string mount_path     (const Device*) const override;

    NvmeFile* create_file(const Device*  device,
                          std::string_view name,
                          uint64_t        size_bytes,
                          bool            persist_now = true,
                          bool            sync_now    = true) override;
    bool      flush_metadata(const Device* device)  override;
    NvmeFile* open_file(const Device*    device,
                         std::string_view name) override;
    bool      close_file(NvmeFile* file) override;
    bool      delete_file(NvmeFile* file, bool persist_now = true) override;
    std::vector<NvmeFile*>   list_files     (const Device*) const override;
    std::vector<std::string> list_file_names(const Device*) const override;

    ssize_t   read_blocking (NvmeFile*, uint64_t off, void*       dst, size_t len) override;
    ssize_t   write_blocking(NvmeFile*, uint64_t off, const void* src, size_t len) override;
    bool      sync(NvmeFile*) override;

    NvmeFileDeviceHandle* acquire_device_handle (NvmeFile* file)            override;
    void                  release_device_handle(NvmeFileDeviceHandle* dh)  override;

private:
    struct PerDeviceState {
        const Device*                       device = nullptr;
        std::string                         snvme_blk_path;   // /dev/snvmeNn1
        std::string                         mount_path;       // e.g. /mnt/tutti/snvme0
        std::unique_ptr<PersistentFileLog>  log;
        // Open NvmeFiles, keyed by file_id.  We own the unique_ptrs.
        std::map<uint64_t, std::unique_ptr<NvmeFile>> files;
        // Did THIS bootstrap() perform the mount(2)?  Used to decide
        // whether shutdown() should umount(2).
        bool                                we_mounted = false;

        // Bulk-init dirty flags (R5a.1):
        //   dirty_unsynced_files   set whenever create_file(sync_now=false)
        //                          accepted bytes/extents into the page
        //                          cache without an fsync.  Cleared by a
        //                          successful syncfs(2) inside
        //                          flush_metadata().
        //   dirty_unpersisted_log  set whenever create_file/delete_file
        //                          mutated the log without a subsequent
        //                          log.persist().  Cleared by a successful
        //                          log.persist() inside flush_metadata().
        // shutdown() also drains both flags (close_file fsyncs each
        // host_fd individually, and the final s.log->persist() lands
        // any deferred entries).
        bool                                dirty_unsynced_files   = false;
        bool                                dirty_unpersisted_log  = false;
    };

    PerDeviceState* find_state(const Device*);
    const PerDeviceState* find_state(const Device*) const;

    bool mkfs_if_needed_locked(const std::string& blk_path);
    bool mount_if_needed_locked(PerDeviceState& s);
    bool umount_locked(PerDeviceState& s);

    bool create_file_locked(PerDeviceState& s,
                            std::string_view name,
                            uint64_t        size_bytes,
                            bool            persist_now,
                            bool            sync_now,
                            NvmeFile**      out);

    // C0 reconcile (R5a.1): after PersistentFileLog::load_or_init,
    // walk <mount>/.tutti/ to find:
    //   - log entries whose <name>.bin is missing -> drop the entry
    //     (tombstone left by a delete_file that crashed between
    //      ::unlink and log.persist).
    //   - <name>.bin files with no corresponding log entry -> unlink
    //     them (ghost left by a create_file that crashed between
    //      file pwrite/fsync and log.persist).
    // If anything was changed, persist the log once.  Single-pass,
    // best-effort; logs warnings but never fails bootstrap.
    bool reconcile_locked_(PerDeviceState& s);

    // Helpers for snvme path conventions:  /dev/ssnvme<m> -> /dev/snvme<m>n1
    // and minor extraction for mount-point naming.
    static std::string blk_path_from_chrdev(const std::string& chrdev);
    static int         minor_from_chrdev (const std::string& chrdev);

    Config                          cfg_;
    mutable std::mutex              mtx_;
    std::vector<std::unique_ptr<PerDeviceState>> states_;   // device order
    bool                            booted_ = false;
};

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_HOST_FS_BACKED_NVME_STORAGE_H__

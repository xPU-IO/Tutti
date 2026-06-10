#ifndef __TUTTI_NVME_STORAGE_NVME_FILE_H__
#define __TUTTI_NVME_STORAGE_NVME_FILE_H__

/**
 * nvme_file.h -- the runtime-side NvmeFile handle.
 *
 * Layer: nvme_storage.  This is what callers (block_storage, smokes)
 * hold after a successful create_file / open_file.  It is *not* the
 * on-disk header (see nvme_file_header.h) -- that one lives at byte
 * 0..4095 of the file; this struct is the host-side runtime view.
 *
 * Lifetime:
 *   - Created and owned by INvmeStorage.
 *   - Destroyed via INvmeStorage::close_file() or delete_file().
 *   - Holders MUST NOT delete; the storage handles that.
 *
 * Multi-NVMe:
 *   - `device` points at the controller this file lives on.
 *     Multi-NVMe deployments have many NvmeFile instances per
 *     storage; each is bound to exactly one Device.
 *
 * What's NOT in here:
 *   - GPU-side mirror handle (NvmeFileDeviceHandle).  R5b adds it.
 *   - Per-file lease.  Block storage layer owns leasing.
 */

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "lba_extent.h"

namespace tutti {

struct Device;

using NvmeFileId = uint64_t;

struct NvmeFile {
    NvmeFileId            id;
    std::string           name;
    uint64_t              size_bytes;        // logical size (excludes header)
    const Device*         device;
    std::vector<LbaExtent> extents;          // physical LBAs of the host file
                                             //   incl. the 4 KiB header at extent[0]
    // ---- host-side bookkeeping ----
    int                   host_fd = -1;      // open fd for pread/pwrite/fsync;
                                             //   < 0 if file is closed.
    std::string           host_path;         // e.g. "/mnt/nvme1/.tutti/foo.bin"
    uint64_t              data_offset = 0;   // bytes; equals sizeof(NvmeFileHeader)
};

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_NVME_FILE_H__

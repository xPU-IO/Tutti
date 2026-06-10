#ifndef __TUTTI_NVME_STORAGE_FIEMAP_HELPER_H__
#define __TUTTI_NVME_STORAGE_FIEMAP_HELPER_H__

/**
 * fiemap_helper.h -- thin wrapper around the FS_IOC_FIEMAP ioctl.
 *
 * FS_IOC_FIEMAP is a Linux VFS ioctl supported by ext4, xfs, btrfs
 * and friends: given an open fd, it returns the file's physical
 * extent map in bytes.  We translate that to LbaExtent (in NVMe
 * block units) using the fs block size.
 *
 * Layer: nvme_storage internal helper.  Not a public API -- the
 * mapping is hidden behind INvmeStorage::create_file / open_file.
 *
 * Conditions for a clean extent capture:
 *   - The file must already be allocated on disk: call fallocate()
 *     and fsync() before fiemap.
 *   - The fs block size (e.g. 4 KiB ext4) must be a multiple of
 *     the NVMe LBA size, OR equal -- which is the typical case.
 *
 * Failure modes:
 *   - If FS_IOC_FIEMAP returns a hole or an unwritten extent
 *     (FIEMAP_EXTENT_UNWRITTEN), we treat it as a hard error.
 *     fallocate() is supposed to give us mapped extents; if it
 *     didn't, the file isn't truly contiguous on disk.
 *   - Encrypted / data-inline / unaligned extents are also rejected.
 *   - Caps at kNvmeFileHeaderMaxExtents extents (124).  Any file
 *     producing more than that on a freshly-fallocate'd ext4 should
 *     be reported and the caller asked to resize / use a less
 *     fragmented filesystem.
 */

#include <cstdint>
#include <string>
#include <vector>

#include "lba_extent.h"

namespace tutti {

struct FiemapResult {
    bool                     ok;
    std::vector<LbaExtent>   extents;        // in NVMe block units
    uint32_t                 fs_block_size;  // bytes, e.g. 4096
    std::string              error;          // populated if !ok
};

/// Read the file's extent map from `fd` and translate to NVMe block
/// units using `nvme_block_size_bytes` (typically 4096).
FiemapResult read_extents(int fd, uint32_t nvme_block_size_bytes);

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_FIEMAP_HELPER_H__

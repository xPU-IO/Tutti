/**
 * fiemap_helper.cpp -- FS_IOC_FIEMAP wrapper for ext4 / xfs / etc.
 */

#include "fiemap_helper.h"
#include "nvme_file_header.h"   // kNvmeFileHeaderMaxExtents

#include <linux/fiemap.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace tutti {

namespace {

// The kernel's struct fiemap is variable-length (extent[] array
// at the end).  We allocate a buffer sized for kMaxExtents queries
// in one shot.  Real-world fallocate'd files are rarely > 30 extents
// on ext4; 256 gives us plenty of slack.
constexpr uint32_t kFiemapMaxExtentsPerCall = 256;

uint32_t fs_block_size_for(int fd) {
    struct stat st{};
    if (fstat(fd, &st) != 0) return 0;
    // st_blksize is the filesystem's preferred IO size, which on
    // ext4/xfs/btrfs is the filesystem block size (typ. 4 KiB).
    return (uint32_t)st.st_blksize;
}

} // namespace

FiemapResult read_extents(int fd, uint32_t nvme_block_size_bytes) {
    FiemapResult res;
    res.ok = false;
    res.fs_block_size = 0;

    if (fd < 0) {
        res.error = "invalid fd";
        return res;
    }
    if (nvme_block_size_bytes == 0) {
        res.error = "nvme_block_size_bytes == 0";
        return res;
    }

    res.fs_block_size = fs_block_size_for(fd);
    if (res.fs_block_size == 0) {
        res.error = "fstat failed (errno " + std::to_string(errno) + ")";
        return res;
    }
    if (res.fs_block_size % nvme_block_size_bytes != 0) {
        res.error = "fs block size " + std::to_string(res.fs_block_size) +
                    " not multiple of NVMe block size " +
                    std::to_string(nvme_block_size_bytes);
        return res;
    }

    // Sync the file so the kernel emits stable extents (FS_IOC_FIEMAP
    // can otherwise return UNWRITTEN flags on freshly-fallocate'd files
    // until the journal commits).
    if (fsync(fd) != 0) {
        res.error = "fsync before fiemap: errno " + std::to_string(errno);
        return res;
    }

    // The kernel API expects a single buffer holding both the header
    // (struct fiemap) and the extent array.  Allocate inline.
    const size_t bytes = sizeof(struct fiemap) +
                         sizeof(struct fiemap_extent) * kFiemapMaxExtentsPerCall;
    std::vector<uint8_t> buf(bytes, 0);
    auto* fm = reinterpret_cast<struct fiemap*>(buf.data());

    // Loop in case the file has more extents than our buffer can hold.
    // Each call: ask for [logical_start, ~uint64_max), kernel returns
    // up to kFiemapMaxExtentsPerCall extents starting at the first one
    // that overlaps logical_start.
    uint64_t logical_cursor = 0;
    bool     done = false;
    while (!done) {
        std::memset(buf.data(), 0, bytes);
        fm->fm_start          = logical_cursor;
        fm->fm_length         = ~uint64_t(0) - logical_cursor;
        fm->fm_flags          = FIEMAP_FLAG_SYNC;
        fm->fm_extent_count   = kFiemapMaxExtentsPerCall;

        if (ioctl(fd, FS_IOC_FIEMAP, fm) != 0) {
            res.error = "FS_IOC_FIEMAP: errno " + std::to_string(errno);
            return res;
        }

        if (fm->fm_mapped_extents == 0) {
            // No more extents.
            break;
        }

        for (uint32_t i = 0; i < fm->fm_mapped_extents; ++i) {
            const struct fiemap_extent& ex = fm->fm_extents[i];

            // Reject anything that isn't a clean, on-disk extent.
            //
            // FIEMAP_EXTENT_UNWRITTEN is INTENTIONALLY *not* in this
            // mask: fallocate() on ext4 / xfs returns extents marked
            // UNWRITTEN until they are first written.  The physical
            // LBA range is already allocated and stable; ext4 simply
            // short-circuits reads to zero until a write touches the
            // block, at which point the kernel atomically clears the
            // UNWRITTEN flag in-place.  Treating UNWRITTEN as fatal
            // would mean every freshly-fallocate'd file is rejected,
            // which is exactly the opposite of what we want.
            const uint32_t bad_flags =
                FIEMAP_EXTENT_UNKNOWN |
                FIEMAP_EXTENT_DELALLOC |
                FIEMAP_EXTENT_ENCODED |
                FIEMAP_EXTENT_DATA_ENCRYPTED |
                FIEMAP_EXTENT_NOT_ALIGNED |
                FIEMAP_EXTENT_DATA_INLINE |
                FIEMAP_EXTENT_DATA_TAIL;
            if (ex.fe_flags & bad_flags) {
                char tmp[128];
                std::snprintf(tmp, sizeof(tmp),
                    "fiemap extent has bad flags 0x%x (logical=%llu len=%llu)",
                    (unsigned)ex.fe_flags,
                    (unsigned long long)ex.fe_logical,
                    (unsigned long long)ex.fe_length);
                res.error = tmp;
                return res;
            }

            if (ex.fe_physical % nvme_block_size_bytes != 0 ||
                ex.fe_length   % nvme_block_size_bytes != 0) {
                char tmp[128];
                std::snprintf(tmp, sizeof(tmp),
                    "extent not NVMe-block-aligned phys=%llu len=%llu",
                    (unsigned long long)ex.fe_physical,
                    (unsigned long long)ex.fe_length);
                res.error = tmp;
                return res;
            }

            LbaExtent out{};
            out.start_lba     = ex.fe_physical / nvme_block_size_bytes;
            out.length_blocks = ex.fe_length   / nvme_block_size_bytes;
            res.extents.push_back(out);

            if (res.extents.size() > kNvmeFileHeaderMaxExtents) {
                res.error = "too many extents (>" +
                            std::to_string(kNvmeFileHeaderMaxExtents) +
                            "); file too fragmented";
                return res;
            }

            // Last extent of the file?
            if (ex.fe_flags & FIEMAP_EXTENT_LAST) {
                done = true;
            }
            logical_cursor = ex.fe_logical + ex.fe_length;
        }

        // Defensive: if kernel filled the buffer to the brim and didn't
        // mark the last extent, loop again with the new cursor.
        if (!done && fm->fm_mapped_extents < kFiemapMaxExtentsPerCall) {
            // Got fewer than asked, no LAST flag -- treat as done.
            break;
        }
    }

    if (res.extents.empty()) {
        res.error = "fiemap returned 0 extents (file is empty / sparse?)";
        return res;
    }

    res.ok = true;
    return res;
}

} // namespace tutti

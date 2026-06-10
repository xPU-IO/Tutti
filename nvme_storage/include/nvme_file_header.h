#ifndef __TUTTI_NVME_STORAGE_NVME_FILE_HEADER_H__
#define __TUTTI_NVME_STORAGE_NVME_FILE_HEADER_H__

/**
 * nvme_file_header.h -- on-disk header embedded in every NvmeFile.
 *
 * Layer: nvme_storage.
 *
 * Why a per-file header:
 *   - The host fd can be closed and reopened across runs; we want
 *     every file to be self-describing without consulting an
 *     external table.
 *   - The GPU kernel resolves "NvmeFile + offset -> LBA" using the
 *     extent table that lives here; we copy it into device memory
 *     once at NvmeFile open time.
 *
 * Layout: a single 4 KiB block at the start of every NvmeFile.  This
 * block is *part of* the host-visible ext4 file (so fallocate'd byte
 * 0..4095 belongs to the header, byte 4096+ belongs to user data).
 * The GPU side computes effective LBA from
 *   logical_byte_offset + sizeof(header)
 * and walks `extents[]` to translate.
 *
 * Versioning:
 *   - `magic` and `version` together identify the layout.  Bump
 *     `version` for any incompatible change.
 *   - New fields go into `reserved[]`; alignment / size are fixed
 *     at exactly 4 KiB and must stay so (`static_assert` below).
 *
 * Lifetime:
 *   - Written once at create_file() time and never edited.  Re-open
 *     reads it back to recover extents + size + name without a
 *     side-table lookup.
 */

#include <cstdint>

#include "lba_extent.h"

namespace tutti {

// Spelled "TUTI" + "FH01" reversed for little-endian ascii readability.
inline constexpr uint64_t kNvmeFileHeaderMagic = 0x3130484654495455ULL;
inline constexpr uint32_t kNvmeFileHeaderVersion = 1;

// Cap on inline extents.  fallocate of contiguous space typically
// produces 1-30 extents; ext4 4 KiB block FS rarely exceeds 124 for
// any practical NvmeFile size.  Overflow handling is a follow-up.
inline constexpr uint32_t kNvmeFileHeaderMaxExtents = 124;

struct NvmeFileHeader {
    uint64_t  magic;                    //  8 B
    uint32_t  version;                  //  4 B
    uint32_t  fs_block_size;            //  4 B  e.g. 4096
    uint64_t  file_size_bytes;          //  8 B  logical size requested by caller
    uint64_t  file_id;                  //  8 B  unique within (storage, device)
    uint32_t  num_extents;              //  4 B
    uint32_t  reserved0;                //  4 B
    char      name[64];                 // 64 B  zero-terminated
    LbaExtent extents[kNvmeFileHeaderMaxExtents]; // 124 * 16 = 1984 B
    char      reserved1[4096
                         - 8 - 4 - 4 - 8 - 8 - 4 - 4 - 64
                         - (int)(sizeof(LbaExtent) * kNvmeFileHeaderMaxExtents)];
};

static_assert(sizeof(NvmeFileHeader) == 4096,
              "NvmeFileHeader must be exactly 4 KiB");

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_NVME_FILE_HEADER_H__

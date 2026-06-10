#ifndef __TUTTI_NVME_STORAGE_LBA_EXTENT_H__
#define __TUTTI_NVME_STORAGE_LBA_EXTENT_H__

/**
 * lba_extent.h -- a single contiguous LBA range owned by an NvmeFile.
 *
 * Layer: nvme_storage.  Used both on host and on device (the GPU
 * kernel side will mirror this struct to NvmeFileDeviceHandle).
 *
 * `start_lba`     first LBA of the extent (in NVMe block units).
 * `length_blocks` number of blocks in the extent (in NVMe block units).
 *
 * NVMe blocks are typically 4 KiB (matches ext4's logical block).
 * We stay in block units, not bytes, to match how libnvm + the NVMe
 * SQE submit path expresses them (NLB / SLBA).
 */

#include <cstdint>

namespace tutti {

struct LbaExtent {
    uint64_t start_lba;
    uint64_t length_blocks;
};

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_LBA_EXTENT_H__

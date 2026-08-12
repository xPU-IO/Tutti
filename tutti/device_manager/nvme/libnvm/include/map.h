#ifndef __NVM_INTERNAL_LINUX_MAP_H__
#define __NVM_INTERNAL_LINUX_MAP_H__
#ifdef __linux__

#include "ioctl.h"
#include "dma.h"


/*
 * What kind of memory are we mapping.
 */
enum mapping_type
{
    MAP_TYPE_CUDA   =   0x1,   // CUDA device memory
    MAP_TYPE_HOST   =   0x2,   // Host memory (RAM)
    MAP_TYPE_CUDA_QUEUE   =   0x3,   // CUDA device memory for IO QUEUE
    MAP_TYPE_API    =   0x4    // Allocated by the API (RAM)
};



/*
 * Mapping container
 *
 * group_id + map_kind (see ioctl.h):
 *
 *   group_id == 0 && map_kind == NVM_MAP_KIND_UNSPECIFIED
 *           Legacy fallback path.  Kernel registers the map against
 *           the controller-global host/device/device_queue list and
 *           ignores per-fd queue-group semantics.  This is what every
 *           legacy libnvm caller still gets, so behaviour is preserved
 *           bit-for-bit while we incrementally migrate callers.
 *
 *   group_id != 0 && map_kind == NVM_MAP_KIND_RING_SQ/RING_CQ
 *           Queue-group path.  Map is linked onto the group's map list
 *           and cascade-released by NVM_DESTROY_QUEUE_GROUP /
 *           fd close.  ioq_idx / is_cq are forced to -1 at ioctl
 *           submission time (new-mode API).
 *
 *   group_id == 0 && map_kind == NVM_MAP_KIND_DATA
 *           B6 fd-scoped DATA buffer.  Map is linked onto the per-fd
 *           data_maps list; survives queue-group destroy and is only
 *           released on fd close.  ioq_idx / is_cq are forced to -1.
 *
 * is_cq / ioq_idx remain the legacy bring-up (NVM_SET_IOQ_NUM) hints
 * and are only honoured when map_kind == UNSPECIFIED.
 */
struct ioctl_mapping
{
    enum mapping_type   type;       // What kind of memory
    void*               buffer;
    struct va_range     range;      // Memory range descriptor
    int                 is_cq;      // legacy mode only; -1 in explicit-map mode
    int                 ioq_idx;    // legacy mode only; -1 in explicit-map mode
    int                 n_entries;
    uint32_t            group_id;   // per-fd queue group id; 0 = legacy / DATA
    uint8_t             map_kind;   // enum nvm_map_kind (B6); 0 = UNSPECIFIED
};


#endif /* __linux__ */
#endif /* __NVM_INTERNAL_LINUX_MAP_H__ */

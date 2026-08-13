#include <nvm_types.h>
#include <nvm_util.h>
#include <nvm_dma.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include "lib_util.h"
#include "lib_ctrl.h"
#include "dma.h"
#include "map.h"
#include "dprintf.h"



static void remove_mapping_descriptor(struct ioctl_mapping* md)
{
    if (md->type == MAP_TYPE_API)
    {
        free((void*) md->buffer);
    }

    free(md);
}



static void release_mapping_descriptor(struct va_range* va)
{
    remove_mapping_descriptor(_nvm_container_of(va, struct ioctl_mapping, range));
}



static int create_mapping_descriptor(struct ioctl_mapping** handle, size_t page_size, enum mapping_type type, void* buffer, size_t size)
{
    size_t n_pages = NVM_PAGE_ALIGN(size, page_size) / page_size;
    if (n_pages == 0)
    {
        return EINVAL;
    }

    struct ioctl_mapping* md = (struct ioctl_mapping*) malloc(sizeof(struct ioctl_mapping));
    if (md == NULL)
    {
        dprintf("Failed to allocate mapping descriptor: %s\n", strerror(errno));
        return errno;
    }

    md->type = type;
    md->buffer = buffer;
    md->range.remote = false;
    md->range.vaddr = (volatile void*) buffer;
    md->range.page_size = page_size;
    md->range.n_pages = n_pages;
    // printf("create_mapping_descriptor page size is %u, page num is %u,size is %d\n",page_size,n_pages,size);
    md->is_cq = -1;
    md->ioq_idx = -1;
    md->n_entries = 0;
    /* Explicit-map defaults: legacy fallback path. Callers that want new-mode
     * behaviour overwrite group_id / map_kind on the returned descriptor
     * before calling _nvm_dma_init(). */
    md->group_id = 0;
    md->map_kind = NVM_MAP_KIND_UNSPECIFIED;
    *handle = md;
    return 0;
}



int nvm_dma_create(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, size_t size)
{
    void* buffer;
    struct ioctl_mapping* md;

    size = NVM_CTRL_ALIGN(ctrl, size);
    if (size == 0)
    {
        return EINVAL;
    }

    *handle = NULL;
    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = posix_memalign(&buffer, ctrl->page_size, size);
    if (err != 0)
    {
        dprintf("Failed to allocate page-aligned memory buffer: %s\n", strerror(err));
        return err;
    }

    err = create_mapping_descriptor(&md, ctrl->page_size, MAP_TYPE_API, buffer, size);
    if (err != 0)
    {
        free(buffer);
        return err;
    }

    err = _nvm_dma_init(handle, ctrl, &md->range, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }

    return 0;
}



int nvm_dma_map_host(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* vaddr, size_t size,int is_cq, int ioq_idx)
{
    struct ioctl_mapping* md;
    *handle = NULL;

    size = NVM_CTRL_ALIGN(ctrl, size);
    if (size == 0)
    {
        return EINVAL;
    }

    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = create_mapping_descriptor(&md, ctrl->page_size, MAP_TYPE_HOST, vaddr, size);
    if (err != 0)
    {
        return err;
    }
    md->is_cq = is_cq;
    md->ioq_idx = ioq_idx;
    err = _nvm_dma_init(handle, ctrl, &md->range, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }

    return 0;
}



int nvm_dma_map_device(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* devptr, size_t size)
{
    struct ioctl_mapping* md;
    *handle = NULL;

    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = create_mapping_descriptor(&md, 1ULL << 16, MAP_TYPE_CUDA, devptr, size);
    if (err != 0)
    {
        return err;
    }


    err = _nvm_dma_init(handle, ctrl, &md->range, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }

    return 0;
}


int nvm_dma_map_queue_device(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* devptr, size_t size,unsigned int is_cq, uint16_t qno)
{
    struct ioctl_mapping* md;
    *handle = NULL;

    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = create_mapping_descriptor(&md, 1ULL << 16, MAP_TYPE_CUDA_QUEUE, devptr, size);
    if (err != 0)
    {
        return err;
    }

    md->is_cq = (int)is_cq;
    md->ioq_idx = (int)qno;
    // printf("nvm_dma_map_queue_device cq is %d, idx is %d, cq is %u, qno is %u\n",md->is_cq,md->ioq_idx,is_cq,qno);
    err = _nvm_dma_init(handle, ctrl, &md->range, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }

    return 0;
}


/* ===================================================================
 * Explicit-intent DMA mapping API.
 *
 * All four follow the same shape:
 *   1) allocate ioctl_mapping descriptor (default: legacy fallback).
 *   2) set group_id + map_kind to the new-mode values.
 *   3) is_cq / ioq_idx stay at -1 (new-mode invariant: kernel does
 *      NOT use these in lookups when map_kind != UNSPECIFIED).
 *   4) hand off to _nvm_dma_init -> ioctl_map.
 * =================================================================== */

int nvm_dma_map_data_host(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* vaddr, size_t size)
{
    struct ioctl_mapping* md;
    *handle = NULL;

    size = NVM_CTRL_ALIGN(ctrl, size);
    if (size == 0)
    {
        return EINVAL;
    }
    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = create_mapping_descriptor(&md, ctrl->page_size, MAP_TYPE_HOST, vaddr, size);
    if (err != 0)
    {
        return err;
    }
    md->group_id = 0;                                /* DATA: group_id is ignored */
    md->map_kind = NVM_MAP_KIND_DATA;

    err = _nvm_dma_init(handle, ctrl, &md->range, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }
    return 0;
}


int nvm_dma_map_data_device(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* devptr, size_t size)
{
    struct ioctl_mapping* md;
    *handle = NULL;

    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = create_mapping_descriptor(&md, 1ULL << 16, MAP_TYPE_CUDA, devptr, size);
    if (err != 0)
    {
        return err;
    }
    md->group_id = 0;
    md->map_kind = NVM_MAP_KIND_DATA;

    err = _nvm_dma_init(handle, ctrl, &md->range, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }
    return 0;
}


int nvm_dma_map_ring_host(nvm_dma_t** handle, const nvm_ctrl_t* ctrl,
                          uint32_t group_id, void* vaddr, size_t size, int is_cq)
{
    struct ioctl_mapping* md;
    *handle = NULL;

    if (group_id == 0)
    {
        /* RING_SQ/CQ must be attached to a queue group; bail early
         * before the kernel rejects us with -EINVAL so the failure
         * mode is unambiguous. */
        return EINVAL;
    }
    size = NVM_CTRL_ALIGN(ctrl, size);
    if (size == 0)
    {
        return EINVAL;
    }
    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = create_mapping_descriptor(&md, ctrl->page_size, MAP_TYPE_HOST, vaddr, size);
    if (err != 0)
    {
        return err;
    }
    md->group_id = group_id;
    md->map_kind = is_cq ? NVM_MAP_KIND_RING_CQ : NVM_MAP_KIND_RING_SQ;

    err = _nvm_dma_init(handle, ctrl, &md->range, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }
    return 0;
}


int nvm_dma_map_ring_device(nvm_dma_t** handle, const nvm_ctrl_t* ctrl,
                            uint32_t group_id, void* devptr, size_t size, int is_cq)
{
    struct ioctl_mapping* md;
    *handle = NULL;

    if (group_id == 0)
    {
        return EINVAL;
    }
    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = create_mapping_descriptor(&md, 1ULL << 16, MAP_TYPE_CUDA, devptr, size);
    if (err != 0)
    {
        return err;
    }
    md->group_id = group_id;
    md->map_kind = is_cq ? NVM_MAP_KIND_RING_CQ : NVM_MAP_KIND_RING_SQ;

    err = _nvm_dma_init(handle, ctrl, &md->range, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }
    return 0;
}

#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_util.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <atomic>
#include "mutex.h"
#include "lib_ctrl.h"
#include "lib_util.h"
#include "regs.h"
#include "dprintf.h"
#include "ioctl.h"


/* Convenience defines */
#define encode_page_size(ps)    _nvm_b2log((ps) >> 12)
#define encode_entry_size(es)   _nvm_b2log(es)



/*
 * Helper function to allocate a handle container.
 */
static struct controller* create_handle(struct device* dev, const struct device_ops* ops, enum device_type type)
{
    int err;
    struct controller* handle;

    if (dev != NULL && (ops == NULL || ops->release_device == NULL))
    {
        dprintf("Inconsistent state, device operations is not set\n");
        return NULL;
    }

    handle = (struct controller*) malloc(sizeof(struct controller));
    if (handle == NULL)
    {
        dprintf("Failed to allocate controller handle: %s\n", strerror(errno));
        return NULL;
    }

    memset(&handle->handle, 0, sizeof(nvm_ctrl_t));
   
    err = _nvm_mutex_init(&handle->lock);
    if (err != 0)
    {
        free(handle);
        return NULL;
    }
    
    handle->count = 1;
    handle->device = dev;
    handle->type = type;
    if (ops != NULL)
    {
        handle->ops = *ops;
    }
    else
    {
        memset(&handle->ops, 0, sizeof(struct device_ops));
    }

    return handle;
}



static void remove_handle(struct controller* handle)
{
    int status;

    do
    {
        status = _nvm_mutex_free(&handle->lock);
    }
    while (status == EBUSY);
    
    free(handle);
}



/*
 * Take device reference.
 */
struct controller* _nvm_ctrl_get(const nvm_ctrl_t* ctrl)
{
    if (ctrl != NULL)
    {
        // This is technically undefined behaviour (casting away const),
        // but we are not modifying the handle itself, only the container.
        struct controller* controller = _nvm_container_of(ctrl, struct controller, handle);

        int err = _nvm_mutex_lock(&controller->lock);
        if (err != 0)
        {
            dprintf("Failed to take device reference lock: %s\n", strerror(err));
            return NULL;
        }

        // Increase reference count
        ++controller->count;

        _nvm_mutex_unlock(&controller->lock);

        return controller;
    }

    return NULL;
}



/*
 * Release device reference.
 */
void _nvm_ctrl_put(struct controller* controller)
{
    if (controller != NULL)
    {
        uint32_t count = 0;

        _nvm_mutex_lock(&controller->lock);
        count = --controller->count;
        if (count == 0)
        {
            if (controller->device != NULL)
            {
                controller->ops.release_device(controller->device);
            }

            controller->device = NULL;
        }
        _nvm_mutex_unlock(&controller->lock);

        if (count == 0)
        {
            remove_handle(controller);
        }
    }
}



int _nvm_ctrl_init(nvm_ctrl_t** handle, struct device* dev, const struct device_ops* ops, enum device_type type, volatile void* mm_ptr, size_t mm_size)
{
    struct controller* container;
    nvm_ctrl_t* ctrl;

    *handle = NULL;

    container = create_handle(dev, ops, type);
    if (container == NULL)
    {
        return ENOMEM;
    }

    ctrl = &container->handle;

    ctrl->mm_ptr = mm_ptr;
    ctrl->mm_size = mm_size;
    // Get the system page size
    size_t page_size = _nvm_host_page_size();
    if (page_size == 0)
    {
        remove_handle(container);
        return ENOMEM;
    }

    // Get the controller page size
    // uint8_t host_page_size = encode_page_size(page_size);

    // Set controller properties
    ctrl->page_size = page_size;
    ctrl->dstrd = CAP$DSTRD(ctrl->mm_ptr);
    ctrl->timeout = CAP$TO(ctrl->mm_ptr) * 500UL;
    ctrl->max_qs = CAP$MQES(ctrl->mm_ptr) + 1; // CAP.MQES is 0's based
    // printf("ctrl->dstrd is %u, ctrl->timeout is %lu, ctrl->max_qs is %u\n",ctrl->dstrd,ctrl->timeout,ctrl->max_qs);
    // printf("dstrd  is %u, ctrl->timeout is %lu, max qs is  %u\n",ctrl->dstrd,ctrl->timeout,ctrl->max_qs);
    ctrl->cq_num = 0;
    ctrl->sq_num = 0;

    *handle = ctrl;

    return 0;
}







/*
 * Release controller handle (OWNER-ONLY path).
 *
 * Unconditionally cascades through:
 *   nvm_queue_clear      -- removed: kernel has no NVM_CLEAR_IOQ_NUM handler
 *   nvm_device_unbind    -- SNVM_DEVICE_UNBIND ioctl: detaches snvme
 *                           from the PCI BDF.  System-wide effect:
 *                           the in-tree nvme driver gets the device
 *                           back, /dev/nvmeXnY reappears, etc.
 *   nvm_chrdev_remove    -- SNVM_CHRDEV_REMOVE ioctl: tears down
 *                           /dev/ssnvme<N>.  Affects every other
 *                           process that still has the chrdev open.
 *   _nvm_ctrl_put        -- closes fd_dev + fd_control on refcount=0
 *
 * MUST NOT be called from a client process that did NOT originally
 * issue SNVM_CHRDEV_CREATE + SNVM_DEVICE_BIND -- doing so would yank
 * the device out from under the owner and any sibling clients.
 *
 * Client processes (those built via nvm_ctrl_attach_client()) MUST
 * use nvm_ctrl_free_client() instead, which only drops the libnvm
 * refcount + closes the client's own fd; kernel snvm_dev_release
 * then cascade-cleans whatever groups / DATA maps were attached to
 * that fd, leaving the chrdev / PCI binding intact.
 */
void nvm_ctrl_free(nvm_ctrl_t* ctrl)
{
    if (ctrl != NULL)
    {
        if (ctrl->bar0_cuda_registered && ctrl->mm_ptr != NULL) {
            /* Paired with nvm_controller_init_gpu().  Host-only daemon
             * owners never enter the accelerator runtime here. */
            (void) cudaHostUnregister((void*) ctrl->mm_ptr);
            ctrl->bar0_cuda_registered = false;
        }
        /* nvm_queue_clear() removed: kernel never implemented
         * NVM_CLEAR_IOQ_NUM, so the ioctl always failed with -EINVAL.
         * Cleanup is handled by nvm_device_unbind() + _nvm_ctrl_put()
         * (fd close triggers kernel snvm_dev_release cascade).
         * Restore this call if a future kernel adds the handler. */
        nvm_device_unbind(ctrl);
        struct controller* container = _nvm_container_of(ctrl, struct controller, handle);
        nvm_chrdev_remove(container->device->fd_control, &ctrl->pdev_addr);
        _nvm_ctrl_put(container);
    }
}



int nvm_raw_ctrl_init(nvm_ctrl_t** ctrl)
{
    return _nvm_ctrl_init(ctrl, NULL, NULL, DEVICE_TYPE_UNKNOWN,NULL,0);
}

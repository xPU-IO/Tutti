#ifndef __linux__
#error "Must compile for Linux"
#endif

#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_ctrl.h>
#include <nvm_util.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdio.h>
#include "map.h"
#include "ioctl.h"
#include "lib_ctrl.h"
#include "dprintf.h"
#include <nvm_queue.h>
#include <nvm_error.h>






/*
 * Unmap controller memory and close file descriptor.
 *
 * fd_control may legitimately be -1 in the client-attach path
 * (nvm_ctrl_attach_client never opens /dev/snvm_control because
 * a client has no business calling CHRDEV_CREATE/REMOVE or
 * DEVICE_BIND/UNBIND -- those are owner-only).  Skip the close
 * in that case.
 */
static void release_device(struct device* dev)
{
    if (dev->fd_control >= 0) {
        close(dev->fd_control);
    }
    close(dev->fd_dev);
    free(dev);
}



/*
 * Call kernel module ioctl and map memory for DMA.
 */
static int ioctl_map(const struct device* dev, const struct va_range* va, uint64_t* ioaddrs)
{
    const struct ioctl_mapping* m = _nvm_container_of(va, struct ioctl_mapping, range);
    enum nvm_ioctl_type type;

    switch (m->type)
    {
        case MAP_TYPE_API:
        case MAP_TYPE_HOST:
            type = NVM_MAP_HOST_MEMORY;
            break;
        case MAP_TYPE_CUDA:
            type = NVM_MAP_DEVICE_MEMORY;
            break;
        case MAP_TYPE_CUDA_QUEUE:
            type = NVM_MAP_DEVICE_QUEUE_MEMORY;
            break;
        default:
            dprintf("Unknown memory type in map for device");
            return EINVAL;
    }

    /*
     * Layout matches struct nvm_ioctl_map exactly; the B6 `map_kind`
     * lives in the byte freed by splitting the old `reserved` u32, so
     * pre-B6 callers (m->map_kind == UNSPECIFIED) submit the same wire
     * bytes as before and the kernel takes the legacy fallback path.
     *
     * New-mode invariant (map_kind != UNSPECIFIED): ioq_idx and is_cq
     * are forced to -1 here.  The kernel honours map_kind alone for
     * RING_* / DATA discrimination; NVM_ADD_USER_QUEUE establishes
     * per-queue identity later via (group_id, sq_vaddr, cq_vaddr).
     */
    int wire_ioq_idx = m->ioq_idx;
    int wire_is_cq   = m->is_cq;
    if (m->map_kind != NVM_MAP_KIND_UNSPECIFIED)
    {
        wire_ioq_idx = -1;
        wire_is_cq   = -1;
    }

    struct nvm_ioctl_map request = {
        .vaddr_start = (uintptr_t) m->buffer,
        .n_pages     = va->n_pages,
        .ioaddrs     = (uint64_t)(uintptr_t) ioaddrs,
        .ioq_idx     = wire_ioq_idx,
        .is_cq       = wire_is_cq,
        .group_id    = m->group_id,
        .map_kind    = m->map_kind,
        .reserved0   = { 0, 0, 0 },
    };

    int err = ioctl(dev->fd_dev, type, &request);
    if (err < 0)
    {
        dprintf("Page mapping kernel request failed (ptr=%p, n_pages=%zu, group=%u, kind=%u): %s\n",
                m->buffer, va->n_pages,
                (unsigned)m->group_id, (unsigned)m->map_kind,
                strerror(errno));
        return errno;
    }

    return 0;
}
struct controller* ctrl_to_controller(nvm_ctrl_t* ctrl)
{
    struct controller* controll;
    controll = _nvm_container_of(ctrl, struct controller, handle);
    if(controll)
        return controll;
    else
        return NULL;
}



/*
 * Call kernel module ioctl and unmap memory.
 */
static void ioctl_unmap(const struct device* dev, const struct va_range* va)
{
    const struct ioctl_mapping* m = _nvm_container_of(va, struct ioctl_mapping, range);
    uint64_t addr = (uintptr_t) m->buffer;
    unsigned int unmap_type;
    if(m->type == MAP_TYPE_HOST)
        unmap_type = NVM_UNMAP_HOST_MEMORY;
    else if(m->type == MAP_TYPE_CUDA)
        unmap_type = NVM_UNMAP_DEVICE_MEMORY;
    else
    {
        printf("Page unmapping kernel type error,type is %u\n",m->type);
        return;
    }
    int err = ioctl(dev->fd_dev, unmap_type, &addr);
    if (err < 0)
    {
        dprintf("Page unmapping kernel request failed: %s\n", strerror(errno));
    }
}
static void ioctl_queue_unmap(const struct device* dev, const struct va_range* va)
{
    const struct ioctl_mapping* m = _nvm_container_of(va, struct ioctl_mapping, range);
    uint64_t addr = (uintptr_t) m->buffer;
    

    int err = ioctl(dev->fd_dev, NVM_UNMAP_DEVICE_QUEUE_MEMORY, &addr);
    if (err < 0)
    {
        dprintf("Page unmapping kernel request failed: %s\n", strerror(errno));
    }
}

int ioctl_get_dev_info(nvm_ctrl_t* ctrl, struct disk* d)
{
    struct nvm_ioctl_dev dev_info;
    struct controller* container;
    int err;
    container  = ctrl_to_controller(ctrl);
    if (container==NULL){
        printf("container error!\n");
        return -1;
    }

    memset(&dev_info, 0, sizeof(dev_info));
    err = ioctl(container->device->fd_dev, NVM_GET_DEV_INFO, &dev_info);
    if (err < 0){
        printf("ioctl_get_dev_info err is %d\n",err);
        return errno;
    }
    /* ABI handshake — fail-closed.
     *
     * The kernel reports its compiled-in TUTTI_SNVME_ABI_VERSION in
     * dev_info.abi_version.  If the kernel's version does not match
     * the userspace header's version, the struct layouts may differ
     * and proceeding would risk memory corruption.  Return ENODEV
     * (not EPERM/EINVAL) so the caller sees "device not compatible"
     * rather than a generic permission/format error.
     *
     * abi_version == 0 means the kernel predates the UAPI
     * consolidation (memset-zeroed reserved field); treat as
     * incompatible. */
    if (dev_info.abi_version != TUTTI_SNVME_ABI_VERSION) {
        nvm_error("ABI version mismatch: kernel=%u userspace=%u "
                  "(device fd=%d)",
                  (unsigned)dev_info.abi_version,
                  (unsigned)TUTTI_SNVME_ABI_VERSION,
                  container->device->fd_dev);
        return ENODEV;
    }
    /* Legacy fields (unchanged semantics).  Note nr_user_q reads back as
     * 0 in the queue-group flow (kernel sets it only when the legacy
     * NVM_SET_IOQ_NUM bring-up path is used); callers that need the
     * actual user-queue count should track it themselves from
     * NVM_ADD_USER_QUEUE results. */
    ctrl->start_cq_idx = dev_info.start_cq_idx;
    ctrl->nr_user_q = dev_info.nr_user_q;
    /* CAP.DSTRD is already in `dev_info.dstrd`; keep ctrl->dstrd as the
     * source of truth (overwrites whatever _nvm_ctrl_init read from
     * BAR0 -- the kernel value is authoritative per ioctl.h note). */
    ctrl->dstrd = dev_info.dstrd;

    /* Queue-group metadata. */
    ctrl->q_depth              = dev_info.q_depth;
    ctrl->bar0_size            = dev_info.bar0_size;
    ctrl->max_user_qid         = dev_info.max_user_qid;
    ctrl->max_queues_per_group = dev_info.max_queues_per_group;
    ctrl->sgl_supported        = dev_info.sgl_supported;

    /* dev_info.max_data_size is already in BYTES (CTRL.MDTS converted
     * by the kernel; see struct nvm_ioctl_dev comment in ioctl.h).
     * The legacy "* 512" here was a double-conversion bug that this
     * code only got away with on controllers whose MDTS happens to be
     * 0 or whose IO never exceeded the bogus inflated cap. */
    d->max_data_size = dev_info.max_data_size;
    // NVM_GET_DEV_INFO returns the namespace logical block size in bytes:
    // 1 << Identify Namespace LBA shift.  It is not the shift value itself.
    d->block_size = dev_info.block_size;
    memcpy(d->disk_name, dev_info.disk_name, DISK_NAME_LEN * sizeof(char));
    nvm_debug("Disk info: start cq idx=%u q_depth=%u max_user_qid=%u "
              "max_data_size=%zu block_size=%zu sgls=0x%x",
              ctrl->start_cq_idx, (unsigned)ctrl->q_depth,
              ctrl->max_user_qid, d->max_data_size, d->block_size,
              ctrl->sgl_supported);
    return 0;
}

static inline int ioctl_chrdev_helper(int fd_control, struct pci_device_addr *device_addr, enum snvm_ctrl_ioctl_type type){
    int err;

    if ((type != SNVM_CHRDEV_CREATE) && (type != SNVM_CHRDEV_REMOVE)){
        printf("ioctl_chrdev_helper type error!\n");
        return -1;
    }

    err = ioctl(fd_control, type, device_addr);
    if (err < 0){
        printf("ioctl_chrdev_helper err is %d\n",err);
        return errno;
    }
    return 0;
}

int nvm_chrdev_create(int fd_control, struct pci_device_addr *device_addr){
    return ioctl_chrdev_helper(fd_control, device_addr, SNVM_CHRDEV_CREATE);
}

int nvm_chrdev_remove(int fd_control, struct pci_device_addr *device_addr){
    return ioctl_chrdev_helper(fd_control, device_addr, SNVM_CHRDEV_REMOVE);
}


static inline int ioctl_device_helper(nvm_ctrl_t* ctrl, enum snvm_ctrl_ioctl_type type){
    struct controller* container;
    int err;

    container = ctrl_to_controller(ctrl);
    if(container==NULL)
    {
        printf("container error!\n");
        return -1;
    }

    if ((type != SNVM_DEVICE_BIND) && (type != SNVM_DEVICE_UNBIND)){
        printf("ioctl_device_helper type error!\n");
        return -1;
    }

    err = ioctl(container->device->fd_control, type, &(ctrl->pdev_addr));
    if (err < 0){
        printf("ioctl_device_helper err is %d\n",err);
        return errno;
    }
    return 0;
}

int nvm_device_bind(nvm_ctrl_t* ctrl){
    return ioctl_device_helper(ctrl, SNVM_DEVICE_BIND);
}

int nvm_device_unbind(nvm_ctrl_t* ctrl){
    return ioctl_device_helper(ctrl, SNVM_DEVICE_UNBIND);
}


static inline int ioctl_queue_helper(nvm_ctrl_t* ctrl, int arg, enum nvm_ioctl_type type)
{
    struct controller* container;
    int err;

    container = ctrl_to_controller(ctrl);
    if (container == NULL){
        nvm_error("container error!");
        return -1;
    }

    switch (type) {
        case NVM_SET_SHARE_REG: {
            /* SET_SHARE_REG still uses the legacy "pack into
             * nvm_ioctl_map.ioq_idx" pattern: it carries one bool
             * (the use_sreg flag), no per-ctrl knobs. */
            struct nvm_ioctl_map request;
            memset(&request, 0, sizeof(request));
            request.ioq_idx = arg;
            err = ioctl(container->device->fd_dev, type, &request);
            break;
        }
        case NVM_CLEAR_IOQ_NUM: {
            struct nvm_ioctl_dev request;
            memset(&request, 0, sizeof(request));
            err = ioctl(container->device->fd_dev, type, &request);
            break;
        }
        case NVM_SET_IOQ_NUM: {
            /* Convenience entry point: same intent as the legacy
             * "set total user IOQ count" call, but submitted through
             * the new struct nvm_ioctl_setup payload so kernel
             * unpacks a single ABI.  Callers that need to populate
             * cap_kernel_ioq / nr_write / nr_poll / groups should
             * use nvm_queue_setup() instead. */
            struct nvm_ioctl_setup setup;
            memset(&setup, 0, sizeof(setup));
            setup.ioq_num = (uint32_t)arg;
            setup.flags   = ctrl->on_host ? NVM_QUEUE_SETUP_F_ON_HOST : 0;
            /* cap_kernel_ioq = 0 -> kernel applies its default of
             * num_possible_cpus().  Callers that need a tighter cap
             * must use nvm_queue_setup() below. */
            err = ioctl(container->device->fd_dev, type, &setup);
            break;
        }
        default:
            return EINVAL;
    }

    if (err < 0){
        int saved_errno = errno;
        nvm_error("ioctl_queue_helper ioctl type=0x%x failed: errno=%d (%s)",
                  type, saved_errno, strerror(saved_errno));
        return saved_errno;
    }

    return 0;
}

int nvm_queue_set(nvm_ctrl_t* ctrl, int q_num){
    return ioctl_queue_helper(ctrl, q_num, NVM_SET_IOQ_NUM);
}

/*
 * Full-fidelity NVM_SET_IOQ_NUM entry point.  Callers that need to
 * pin the kernel-side IO-queue budget (cap_kernel_ioq) or partition
 * the user share across multiple GPUs (groups) must use this
 * instead of nvm_queue_set().
 *
 * setup must have setup->ioq_num populated; on_host comes from
 * ctrl->on_host (caller stays the source of truth) unless setup->
 * flags has already been set, in which case the caller-provided
 * flags win.
 */
int nvm_queue_setup(nvm_ctrl_t* ctrl, struct nvm_ioctl_setup* setup){
    struct controller* container;
    int err;

    if (ctrl == NULL || setup == NULL){
        return EINVAL;
    }
    container = ctrl_to_controller(ctrl);
    if (container == NULL){
        nvm_error("container error!");
        return -1;
    }
    if (setup->flags == 0 && ctrl->on_host){
        setup->flags |= NVM_QUEUE_SETUP_F_ON_HOST;
    }
    err = ioctl(container->device->fd_dev, NVM_SET_IOQ_NUM, setup);
    if (err < 0){
        printf("nvm_queue_setup err is %d\n", errno);
        return errno;
    }
    return 0;
}

int nvm_queue_clear(nvm_ctrl_t* ctrl){
    return ioctl_queue_helper(ctrl, 0, NVM_CLEAR_IOQ_NUM);
}
int nvm_queue_share(nvm_ctrl_t *ctrl){
    return ioctl_queue_helper(ctrl, 1, NVM_SET_SHARE_REG);
}

int nvm_ctrl_init(nvm_ctrl_t** ctrl, int snvme_c_fd, int snvme_d_fd)
{
    int err;
    struct device* dev;

    const struct device_ops ops = {
        .release_device = &release_device,
        .map_range = &ioctl_map,
        .unmap_range = &ioctl_unmap,
        .unmap_queue_range = &ioctl_queue_unmap,
    };

    *ctrl = NULL;

    dev = (struct device*) malloc(sizeof(struct device));
    if (dev == NULL)
    {
        dprintf("Failed to allocate device handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    dev->fd_control = -1;
    if (snvme_c_fd >= 0) {
        dev->fd_control = dup(snvme_c_fd);
        if (dev->fd_control < 0)
        {
            free(dev);
            dprintf("Could not duplicate file descriptor: %s\n", strerror(errno));
            return errno;
        }
    }

    dev->fd_dev = dup(snvme_d_fd);
    if (dev->fd_dev < 0)
    {
        if (dev->fd_control >= 0) close(dev->fd_control);
        free(dev);
        dprintf("Could not duplicate file descriptor: %s\n", strerror(errno));
        return errno;
    }


    if (dev->fd_control >= 0) {
        err = fcntl(dev->fd_control, F_SETFD, O_RDWR);
        if (err == -1)
        {
            close(dev->fd_control);
            close(dev->fd_dev);
            free(dev);
            dprintf("Failed to set file descriptor control: %s\n", strerror(errno));
            return errno;
        }
    }
    err = fcntl(dev->fd_dev, F_SETFD, O_RDWR);
    if (err == -1)
    {
        if (dev->fd_control >= 0) close(dev->fd_control);
        close(dev->fd_dev);
        free(dev);
        dprintf("Failed to set file descriptor control: %s\n", strerror(errno));
        return errno;
    }

    void* mm_ptr = mmap(NULL, NVM_CTRL_MEM_MINSIZE, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_FILE|MAP_LOCKED, dev->fd_dev, 0);
    if (mm_ptr == NULL)
    {
        if (dev->fd_control >= 0) close(dev->fd_control);
        close(dev->fd_dev);
        free(dev);
        dprintf("Failed to map device memory: %s\n", strerror(errno));
        return err;
    }    

    err = _nvm_ctrl_init(ctrl, dev, &ops, DEVICE_TYPE_IOCTL,mm_ptr,NVM_CTRL_MEM_MINSIZE);
    if (err != 0)
    {
        release_device(dev);
        return err;
    }
    return 0;
}

int nvm_device_init(nvm_ctrl_t* ctrl){
    int status;

    status = nvm_queue_share(ctrl);
    if (status != 0){
        nvm_error("Failed to set reg snvme");
        return EFAULT;
    }

    status = nvm_device_bind(ctrl);
    if (status != 0){
        nvm_error("Failed to bind device");
        return EFAULT;
    }
    sleep(3);
    return 0;
}

/* ===================================================================
 * Queue-group and fd-scoped thin ioctl wrappers.
 *
 * These are stateless 1:1 mappings to the kernel ioctls; the Controller
 * class (or any other consumer) decides when to issue them.  They live
 * here in linux/device.cpp because they all need fd_dev (or fd_control)
 * out of `struct controller`, which is a libnvm-internal type.
 *
 * Compared to the legacy nvm_queue_share / nvm_queue_set / nvm_device_init
 * triplet above, these do NOT push the controller through a hidden state
 * machine: the caller drives the sequence (CAP -> BIND -> GET_DEV_INFO
 * -> CREATE_QUEUE_GROUP -> ADD_USER_QUEUE) explicitly.
 * =================================================================== */

int nvm_set_kernel_ioq_cap_fd(int fd_dev, uint32_t cap)
{
    if (ioctl(fd_dev, NVM_SET_KERNEL_IOQ_CAP, &cap) < 0) {
        return errno;
    }
    return 0;
}

int nvm_set_kernel_ioq_cap(nvm_ctrl_t* ctrl, uint32_t cap)
{
    struct controller* container = ctrl_to_controller(ctrl);
    if (container == NULL) {
        return EINVAL;
    }
    return nvm_set_kernel_ioq_cap_fd(container->device->fd_dev, cap);
}

int nvm_create_group(nvm_ctrl_t* ctrl,
                     uint32_t* out_group_id,
                     uint32_t* out_max_queues)
{
    struct controller* container = ctrl_to_controller(ctrl);
    if (container == NULL) {
        return EINVAL;
    }
    struct nvm_ioctl_queue_group req;
    memset(&req, 0, sizeof(req));
    if (ioctl(container->device->fd_dev, NVM_CREATE_QUEUE_GROUP, &req) < 0) {
        return errno;
    }
    if (out_group_id)   *out_group_id   = req.group_id;
    if (out_max_queues) *out_max_queues = req.max_queues;
    return 0;
}

int nvm_destroy_group(nvm_ctrl_t* ctrl, uint32_t group_id)
{
    struct controller* container = ctrl_to_controller(ctrl);
    if (container == NULL) {
        return EINVAL;
    }
    if (ioctl(container->device->fd_dev,
              NVM_DESTROY_QUEUE_GROUP, &group_id) < 0) {
        return errno;
    }
    return 0;
}

int nvm_add_user_queue(nvm_ctrl_t* ctrl,
                       struct nvm_ioctl_add_user_queue* req)
{
    struct controller* container = ctrl_to_controller(ctrl);
    if (container == NULL || req == NULL) {
        return EINVAL;
    }
    if (ioctl(container->device->fd_dev,
              NVM_ADD_USER_QUEUE, req) < 0) {
        return errno;
    }
    return 0;
}

/*
 * Poll NVM_GET_DEV_INFO until nvme_scan_work has populated disk_name.
 *
 * SNVM_DEVICE_BIND returns as soon as the chrdev is callable, but the
 * kernel's nvme_reset_work + nvme_scan_work run asynchronously on
 * s_nvme_wq.  GET_DEV_INFO returns -ENODEV until those finish.
 *
 * Loops up to ~`timeout_ms` ms in 100 ms increments (same shape as the
 * smoke tests in backends/local/kernel_modules/test/).  Returns 0 on
 * success and writes the populated nvm_ioctl_dev into *out_info; returns
 * the last errno seen on timeout.
 */
int nvm_wait_dev_info(nvm_ctrl_t* ctrl,
                      struct nvm_ioctl_dev* out_info,
                      uint32_t timeout_ms)
{
    struct controller* container = ctrl_to_controller(ctrl);
    if (container == NULL || out_info == NULL) {
        return EINVAL;
    }
    int last_err = ENODEV;
    int iters = (int)(timeout_ms / 100);
    if (iters <= 0) iters = 1;
    for (int i = 0; i < iters; i++) {
        memset(out_info, 0, sizeof(*out_info));
        if (ioctl(container->device->fd_dev,
                  NVM_GET_DEV_INFO, out_info) == 0 &&
            out_info->disk_name[0] != '\0') {
            /* ABI handshake — fail-closed (same check as
             * ioctl_get_dev_info above).  If the kernel reports a
             * different ABI version, return ENODEV immediately
             * rather than continuing with potentially mismatched
             * struct layouts. */
            if (out_info->abi_version != TUTTI_SNVME_ABI_VERSION) {
                nvm_error("ABI version mismatch: kernel=%u "
                          "userspace=%u (device fd=%d)",
                          (unsigned)out_info->abi_version,
                          (unsigned)TUTTI_SNVME_ABI_VERSION,
                          container->device->fd_dev);
                return ENODEV;
            }
            return 0;
        }
        last_err = errno;
        usleep(100 * 1000);
    }
    return last_err;
}

/*
 * Shared owner bring-up implementation.
 *
 * Performs the full controller initialisation sequence in one call:
 *
 *   1. open(/dev/snvm_control)
 *   2. SNVM_CHRDEV_CREATE (assigns chrdev minor)
 *   3. open(/dev/ssnvme<minor>)
 *   4. NVM_SET_KERNEL_IOQ_CAP(kernel_ioq_cap) -- pre-bind; cap_kernel_ioq=0
 *      means "no cap, use upstream default of num_possible_cpus()"
 *   5. SNVM_DEVICE_BIND  (binds snvme to the target PCI device)
 *   6. NVM_GET_DEV_INFO poll loop, up to 10 s, until nvme_scan_work
 *      finishes and disk_name is populated.  Result populates the
 *      queue metadata on the ctrl handle (q_depth, bar0_size,
 *      max_user_qid, max_queues_per_group, sgl_supported) plus
 *      legacy fields (start_cq_idx, dstrd, max_data_size,
 *      block_size, disk_name).
 *   7. nvm_ctrl_init (wraps fds + BAR0 mmap into a controller struct)
 * Out-params:
 *   *ctrl       Populated controller handle on success.  Caller must
 *               eventually nvm_ctrl_free(*ctrl) (which closes fds
 *               and unbinds via nvm_device_unbind under the hood --
 *               legacy behaviour, to be split into owner/client
 *               roles in a follow-up commit).
 *   *out_disk   On success, copy of the post-probe disk metadata
 *               (block_size, max_data_size, ns_id, disk_name).
 *               May be NULL if caller does not need it.
 *
 * Returns 0 on success or a positive errno-style code on failure.
 */
static int nvm_controller_init_common(nvm_ctrl_t** ctrl,
                                      const char* snvme_control_path,
                                      const char* pci_addr,
                                      uint32_t kernel_ioq_cap,
                                      struct disk* out_disk)
{
    struct pci_device_addr device_addr, pdev_addr;
    char snvme_path[256];
    int snvme_d_fd = -1, snvme_c_fd = -1;
    int status;

    if (ctrl == NULL || snvme_control_path == NULL || pci_addr == NULL) {
        return EINVAL;
    }
    *ctrl = NULL;

    snvme_c_fd = open(snvme_control_path, O_RDWR | O_NONBLOCK);
    if (snvme_c_fd < 0) {
        nvm_error("Failed to open control descriptor: %s", snvme_control_path);
        return errno ? errno : EFAULT;
    }

    if (sscanf(pci_addr, "%x:%x:%x.%x",
               &device_addr.domain, &device_addr.bus,
               &device_addr.slot, &device_addr.func) != 4) {
        nvm_error("Failed to parse PCI address: %s", pci_addr);
        close(snvme_c_fd);
        return EINVAL;
    }
    pdev_addr = device_addr;   /* preserved across CHRDEV_CREATE clobber */

    status = nvm_chrdev_create(snvme_c_fd, &device_addr);
    if (status != 0) {
        nvm_error("SNVM_CHRDEV_CREATE failed (errno=%d)", status);
        close(snvme_c_fd);
        return status;
    }
    /* SNVM_CHRDEV_CREATE returns the chrdev minor in device_addr->domain.
     * Keep using pdev_addr for the original PCI address. */
    int minor_n = device_addr.domain;
    snprintf(snvme_path, sizeof(snvme_path), "/dev/ssnvme%d", minor_n);

    snvme_d_fd = open(snvme_path, O_RDWR | O_NONBLOCK);
    if (snvme_d_fd < 0) {
        int e = errno;
        nvm_error("Failed to open %s: errno=%d", snvme_path, e);
        /* Best-effort: undo the chrdev create so we don't leak a minor. */
        {
            struct pci_device_addr a = pdev_addr;
            (void) nvm_chrdev_remove(snvme_c_fd, &a);
        }
        close(snvme_c_fd);
        return e ? e : EFAULT;
    }

    /* Step 4: cap-only pre-bind hint.  cap=0 means "no cap, kernel uses
     * num_possible_cpus()" which is the same as not calling this ioctl. */
    if (kernel_ioq_cap != 0) {
        status = nvm_set_kernel_ioq_cap_fd(snvme_d_fd, kernel_ioq_cap);
        if (status != 0) {
            nvm_error("NVM_SET_KERNEL_IOQ_CAP(%u) failed errno=%d",
                      kernel_ioq_cap, status);
            goto fail_close_dev;
        }
    }

    /* Step 5: BIND.  Use a fresh copy of pdev_addr; the kernel will
     * write back into the struct we pass in (see kernel pci.c
     * snvm_chrdev_helper / snvm_rebind_driver). */
    {
        struct pci_device_addr a = pdev_addr;
        if (ioctl(snvme_c_fd, SNVM_DEVICE_BIND, &a) < 0) {
            status = errno;
            nvm_error("SNVM_DEVICE_BIND failed errno=%d", status);
            goto fail_close_dev;
        }
    }

    /* Step 7: wrap fds + BAR0 mmap into nvm_ctrl_t BEFORE we run the
     * GET_DEV_INFO poll loop -- the poll wrapper takes a nvm_ctrl_t. */
    status = nvm_ctrl_init(ctrl, snvme_c_fd, snvme_d_fd);
    if (status != 0) {
        nvm_error("nvm_ctrl_init failed errno=%d", status);
        goto fail_unbind;
    }
    (*ctrl)->on_host = 0;
    (*ctrl)->pdev_addr = pdev_addr;

    /* Step 6: wait for nvme_scan_work to populate disk_name and queue
     * metadata, then copy them onto the ctrl handle. */
    {
        struct nvm_ioctl_dev info;
        struct disk tmp_disk;
        if (out_disk == NULL) {
            out_disk = &tmp_disk;
        }
        memset(out_disk, 0, sizeof(*out_disk));
        status = nvm_wait_dev_info(*ctrl, &info, 10000 /* 10 s */);
        if (status != 0) {
            nvm_error("NVM_GET_DEV_INFO poll timed out errno=%d", status);
            goto fail_ctrl_free;
        }
        /* Apply to ctrl + caller's disk view, same shape as
         * ioctl_get_dev_info above. */
        (*ctrl)->start_cq_idx          = info.start_cq_idx;
        (*ctrl)->nr_user_q             = info.nr_user_q;
        (*ctrl)->dstrd                 = info.dstrd;
        (*ctrl)->q_depth               = info.q_depth;
        (*ctrl)->bar0_size             = info.bar0_size;
        (*ctrl)->max_user_qid          = info.max_user_qid;
        (*ctrl)->max_queues_per_group  = info.max_queues_per_group;
        (*ctrl)->sgl_supported         = info.sgl_supported;
        out_disk->max_data_size = info.max_data_size;  /* bytes already */
        out_disk->block_size    = info.block_size;
        out_disk->page_size     = (*ctrl)->page_size;
        memcpy(out_disk->disk_name, info.disk_name, DISK_NAME_LEN);
    }

    /* nvm_ctrl_init duped the originating fds into the controller. */
    close(snvme_c_fd);
    close(snvme_d_fd);
    snvme_c_fd = snvme_d_fd = -1;

    return 0;

fail_ctrl_free:
    /* Drop the ctrl ref without going through nvm_ctrl_free's unbind
     * cascade (we have NOT given the caller a usable handle).  This
     * uses the internal _nvm_ctrl_put symbol. */
    if (*ctrl != NULL) {
        extern void _nvm_ctrl_put(struct controller*);
        _nvm_ctrl_put(_nvm_container_of(*ctrl, struct controller, handle));
        *ctrl = NULL;
    }
fail_unbind:
    {
        struct pci_device_addr a = pdev_addr;
        (void) ioctl(snvme_c_fd, SNVM_DEVICE_UNBIND, &a);
    }
fail_close_dev:
    if (snvme_d_fd >= 0) close(snvme_d_fd);
    {
        struct pci_device_addr a = pdev_addr;
        (void) nvm_chrdev_remove(snvme_c_fd, &a);
    }
    if (snvme_c_fd >= 0) close(snvme_c_fd);
    return status ? status : EFAULT;
}

int nvm_controller_init_owner(nvm_ctrl_t** ctrl,
                              const char* snvme_control_path,
                              const char* pci_addr,
                              uint32_t kernel_ioq_cap,
                              struct disk* out_disk)
{
    return nvm_controller_init_common(ctrl, snvme_control_path, pci_addr,
                                      kernel_ioq_cap, out_disk);
}

int nvm_controller_init_gpu(nvm_ctrl_t** ctrl,
                            const char* snvme_control_path,
                            const char* pci_addr,
                            uint32_t kernel_ioq_cap,
                            struct disk* out_disk)
{
    int status = nvm_controller_init_common(
        ctrl, snvme_control_path, pci_addr, kernel_ioq_cap, out_disk);
    if (status != 0) {
        return status;
    }

    cudaError_t cerr = cudaHostRegister((void*) (*ctrl)->mm_ptr,
                                        NVM_CTRL_MEM_MINSIZE,
                                        cudaHostRegisterIoMemory);
    if (cerr != cudaSuccess) {
        nvm_error("cudaHostRegister(BAR0) failed: %s",
                  cudaGetErrorString(cerr));
        /* Registration is part of the GPU interface's contract.  Roll the
         * owner lifecycle back instead of returning a live, bound handle on
         * failure. */
        nvm_ctrl_free(*ctrl);
        *ctrl = NULL;
        return EFAULT;
    }
    (*ctrl)->bar0_cuda_registered = true;
    return 0;
}

/* ===================================================================
 * Owner / client role split (L1 Commit 4a) -- implementation.
 *
 * See nvm_ctrl.h for the API contract.
 *
 * The client path opens an existing /dev/ssnvme<N> chrdev (the owner
 * created it with SNVM_CHRDEV_CREATE earlier in another process),
 * mmaps its BAR0, registers the BAR0 host VA with CUDA, and wraps
 * fd_dev into a fresh struct controller.  fd_control is left at -1
 * inside struct device: a client has no business calling any of the
 * /dev/snvm_control-only ioctls (CHRDEV_CREATE/REMOVE, DEVICE_BIND/
 * UNBIND), and every queue-group or fd-scoped ioctl (CREATE/DESTROY_GROUP,
 * ADD_USER_QUEUE, MAP_HOST/DEVICE_MEMORY, GET_DEV_INFO, ...) goes
 * through fd_dev only.  release_device() tolerates fd_control == -1.
 *
 * The client path does NOT call:
 *   - SNVM_CHRDEV_CREATE  (the owner did this)
 *   - NVM_SET_KERNEL_IOQ_CAP  (pre-bind only; owner did it)
 *   - SNVM_DEVICE_BIND  (owner did this, single-driver-per-PCI invariant)
 *   - NVM_GET_DEV_INFO  (the disk metadata is not the client's concern;
 *                        the daemon already mounted the FS / told the
 *                        client what mount path / namespace_id / block_size
 *                        to use via RPC)
 *
 * What the client CAN do once attached:
 *   - nvm_create_group()       -- per-fd, kernel ignores other fds
 *   - nvm_dma_map_ring_*()     -- linked onto its own group->maps
 *   - nvm_dma_map_data_*()     -- linked onto its own data_maps list
 *   - nvm_add_user_queue()     -- kernel allocates user QID from pool
 *   - nvm_destroy_group()      -- cascades through Delete I/O CQ/SQ +
 *                                 RING_* map purge for its own group
 *   - close fd                 -- fd-cascade releases everything still
 *                                 attached to this fd.  This is the
 *                                 reaping path that fires when a
 *                                 client process dies unexpectedly.
 * =================================================================== */

int nvm_ctrl_attach_client(nvm_ctrl_t** ctrl,
                           const char* snvme_dev_path,
                           uint32_t bar0_size)
{
    if (ctrl == NULL || snvme_dev_path == NULL || bar0_size == 0) {
        return EINVAL;
    }
    *ctrl = NULL;

    /* Client does NOT open /dev/snvm_control: that fd is only used
     * for owner-only ioctls (CHRDEV_CREATE/REMOVE, DEVICE_BIND/
     * UNBIND).  All queue-group and fd-scoped ioctls (CREATE/DESTROY_GROUP,
     * ADD_USER_QUEUE, MAP_HOST/DEVICE_MEMORY, GET_DEV_INFO, ...) go
     * through fd_dev only.  We pass -1 to nvm_ctrl_init, which will
     * leave struct device::fd_control at -1 and skip the dup/fcntl
     * and the close-on-release for that field. */

    int fd_dev = open(snvme_dev_path, O_RDWR | O_NONBLOCK);
    if (fd_dev < 0) {
        return errno ? errno : EFAULT;
    }

    int rc = nvm_ctrl_init(ctrl, /*snvme_c_fd=*/-1, fd_dev);
    if (rc != 0) {
        close(fd_dev);
        return rc;
    }
    /* nvm_ctrl_init dup'd fd_dev; close ours. */
    close(fd_dev);

    (*ctrl)->on_host = 0;
    /* pdev_addr is unknown to a client (it never parsed a PCI BDF;
     * it only got a /dev/ssnvme<N> path).  Leave zero -- callers
     * that need it should use the daemon's RPC payload. */

    /* Register BAR0 with CUDA so doorbells become reachable from
     * GPU kernels via cudaHostGetDevicePointer.  Same code path
     * as nvm_controller_init_gpu() takes; failure here is fatal
     * because the only reason a client attaches in the first
     * place is to drive IO from the GPU. */
    cudaError_t cerr = cudaHostRegister(
        (void*) (*ctrl)->mm_ptr,
        NVM_CTRL_MEM_MINSIZE,
        cudaHostRegisterIoMemory);
    if (cerr != cudaSuccess) {
        nvm_error("nvm_ctrl_attach_client: cudaHostRegister(BAR0) failed: %s",
                  cudaGetErrorString(cerr));
        /* Drop the ctrl we just built. */
        extern void _nvm_ctrl_put(struct controller*);
        _nvm_ctrl_put(_nvm_container_of(*ctrl, struct controller, handle));
        *ctrl = NULL;
        return EFAULT;
    }
    (*ctrl)->bar0_cuda_registered = true;

    /* Note: bar0_size is currently not stored on nvm_ctrl_t (mm_size
     * was set to NVM_CTRL_MEM_MINSIZE inside nvm_ctrl_init).  The
     * passed-in size is reserved for future use (e.g. extended
     * doorbell windows) and validated only as non-zero above.
     * Callers that need to mmap MORE than NVM_CTRL_MEM_MINSIZE
     * should issue a separate mmap on a fresh fd; that's outside
     * libnvm's scope. */
    (void) bar0_size;
    return 0;
}

void nvm_ctrl_free_client(nvm_ctrl_t* ctrl)
{
    if (ctrl == NULL) {
        return;
    }
    /* Crucially: NO unbind / chrdev_remove.  Just unregister BAR0
     * with CUDA (matched to the cudaHostRegister in attach_client)
     * and drop the libnvm refcount; release_device closes fd_dev,
     * which triggers the kernel's snvm_dev_release cascade for
     * any groups / DATA maps still attached. */
    if (ctrl->bar0_cuda_registered && ctrl->mm_ptr != NULL) {
        /* Best-effort: ignore the return code.  If CUDA was already
         * shut down (process is exiting), unregister fails harmlessly. */
        (void) cudaHostUnregister((void*) ctrl->mm_ptr);
        ctrl->bar0_cuda_registered = false;
    }
    extern void _nvm_ctrl_put(struct controller*);
    _nvm_ctrl_put(_nvm_container_of(ctrl, struct controller, handle));
}

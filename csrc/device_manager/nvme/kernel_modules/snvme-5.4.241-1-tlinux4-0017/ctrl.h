#ifndef __LIBNVM_HELPER_CTRL_H__
#define __LIBNVM_HELPER_CTRL_H__

#include "list.h"
#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/mutex.h>           /* user_qid_lock for B3 */


/*
 * Per-controller queue-budget descriptor, populated by NVM_SET_IOQ_NUM.
 *
 * Mirrors the userspace ABI (struct nvm_ioctl_setup in libnvm/include/
 * ioctl.h) one-for-one so the ioctl handler is a single copy_from_user
 * into struct ctrl.  The fields are consumed at probe time by
 * snvm_rebind_driver's segment 6a hook (copies them onto struct
 * nvme_dev) and by s_nvme_setup_io_queues' kernel/user IOQ-split
 * reconciliation block.
 *
 * Fixed-size groups[] (NVM_MAX_QUEUE_GROUPS = 8) matches the
 * 8-GPU-per-node upper bound exposed via sys_config.yaml's
 * queue_groups list.
 */
#define SNVM_MAX_QUEUE_GROUPS  8

struct snvm_queue_group {
    u32 owner_id;     /* opaque tag; typically a GPU id. */
    u32 count;        /* (SQ+CQ) pair count in this group. */
    s32 numa_node;    /* doc-only hint; kernel does NOT enforce. */
    u32 reserved;     /* MBZ. */
};

struct snvm_queue_setup {
    u32 valid;        /* set to 1 once NVM_SET_KERNEL_IOQ_CAP has populated */
                      /* this block; 0 = use upstream defaults. */
    u32 ioq_num;      /* total user-side IOQ count. */
    u32 flags;        /* mirrors NVM_QUEUE_SETUP_F_* in uapi. */
    u32 cap_kernel_ioq; /* upper bound on kernel-side IOQ count */
                        /* asked from the controller.  0 means */
                        /* "use num_possible_cpus()". */
    u32 nr_write;     /* per-BDF write_queues override; 0 = module */
                      /* default. */
    u32 nr_poll;      /* ditto for poll_queues. */
    u32 nr_groups;    /* <= SNVM_MAX_QUEUE_GROUPS. */
    struct snvm_queue_group groups[SNVM_MAX_QUEUE_GROUPS];
};

/*
 * Represents an NVM controller.
 */
struct ctrl
{
    struct list_node    list;       /* Linked list head */
    struct pci_dev*     pdev;       /* Reference to physical PCI device */
    char                name[64];   /* Character device name */
    int                 number;     /* Controller number */
    dev_t               rdev;       /* Character device register */
    struct class*       cls;        /* Character device class */
    struct cdev         cdev;       /* Character device (separately allocated) */
    struct device*      chrdev;     /* Character device handle */
    struct nvme_dev *dev;
    /*
     * Full queue-budget snapshot from NVM_SET_IOQ_NUM.
     *   - "user called NVM_SET_IOQ_NUM" can be distinguished from
     *     "user never called it" via setup.valid.
     */
    struct snvm_queue_setup setup;

    /*
     * Per-controller user-QID pool (B3, NVM_ADD_USER_QUEUE).
     *
     * After bind completes, the QID space looks like:
     *   QID 0                : admin
     *   QID 1 ..  online_q-1 : kernel IOQs (managed by upstream)
     *   QID online_q .. nr-1 : user IOQ pool (managed here)
     *
     *   user_qid_first  = ndev->online_queues at first ADD call
     *   user_qid_last   = ndev->nr_allocated_queues - 1
     *   user_qid_bitmap : 1 bit per QID in [first, last], 1 = in use.
     *                     Lazily allocated in NVM_ADD_USER_QUEUE.
     *   user_qid_lock   : protects bitmap + first/last; nests INSIDE
     *                     own->groups_lock (the only way to reach
     *                     this state is via a per-fd ioctl).
     */
    unsigned long      *user_qid_bitmap;
    unsigned int        user_qid_first;
    unsigned int        user_qid_last;
    struct mutex        user_qid_lock;
};



/*
 * Acquire a controller reference.
 */
struct ctrl* ctrl_get(struct list* list, struct class* cls, struct pci_dev* pdev, int number);



/*
 * Release controller reference.
 */
void ctrl_put(struct ctrl* ctrl);



/*
 * Find controller device.
 */
struct ctrl* ctrl_find_by_pci_dev(const struct list* list, const struct pci_dev* pdev);



/*
 * Find controller reference.
 */
struct ctrl* ctrl_find_by_inode(const struct list* list, const struct inode* inode);



/*
 * Create character device and set up file operations.
 */
int ctrl_chrdev_create(struct ctrl* ctrl, 
                       dev_t first,
                       const struct file_operations* fops);



/*
 * Remove character device.
 */
void ctrl_chrdev_remove(struct ctrl* ctrl);



#endif /* __LIBNVM_HELPER_CTRL_H__ */

#include "ctrl.h"
#include "linux/idr.h"
#include "list.h"
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <asm/errno.h>



struct ctrl* ctrl_get(struct list* list, struct class* cls, struct pci_dev* pdev, int number)
{
    struct ctrl* ctrl = NULL;

    ctrl = kmalloc(sizeof(struct ctrl), GFP_KERNEL | GFP_NOWAIT);
    if (ctrl == NULL)
    {
        printk(KERN_CRIT "Failed to allocate controller reference\n");
        return ERR_PTR(-ENOMEM);
    }

    list_node_init(&ctrl->list);

    ctrl->pdev = pdev;
    ctrl->number = number;
    ctrl->rdev = 0;
    ctrl->cls = cls;
    ctrl->chrdev = NULL;
    /* B3 queue-budget snapshot + user-QID pool (Chunk A fields). */
    memset(&ctrl->setup, 0, sizeof(ctrl->setup));
    ctrl->user_qid_bitmap = NULL;
    ctrl->user_qid_first  = 0;
    ctrl->user_qid_last   = 0;
    mutex_init(&ctrl->user_qid_lock);
    // fixme: cdev name
    snprintf(ctrl->name, sizeof(ctrl->name), "%s%d", "ssnvme", ctrl->number);
    ctrl->name[sizeof(ctrl->name) - 1] = '\0';

    list_insert(list, &ctrl->list);

    return ctrl;
}
// // EXPORT_SYMBOL_GPL(ctrl_get);


void ctrl_put(struct ctrl* ctrl)
{
    if (ctrl != NULL)
    {
        list_remove(&ctrl->list);
        ctrl_chrdev_remove(ctrl);
        /*
         * ctrl->cdev is embedded in ctrl.  ctrl_chrdev_remove() calls
         * cdev_del() which unregisters the character device but does
         * NOT wait for in-flight references to drop.  The actual
         * kfree(ctrl) is deferred to ctrl_cdev_release(), which fires
         * when the last kobject reference to ctrl->cdev.kobj is
         * released (e.g. after all open file descriptors are closed)
         * and which first purges any VFS inode still attached to
         * ctrl->cdev.list (see the comment there).
         */
    }
}
// EXPORT_SYMBOL_GPL(ctrl_put);


struct ctrl* ctrl_find_by_pci_dev(const struct list* list, const struct pci_dev* pdev)
{
    const struct list_node* element = list_next(&list->head);
    struct ctrl* ctrl;

    while (element != NULL)
    {
        ctrl = container_of(element, struct ctrl, list);

        if (ctrl->pdev == pdev)
        {
            return ctrl;
        }

        element = list_next(element);
    }

    return NULL;
}
// EXPORT_SYMBOL_GPL(ctrl_find_by_pci_dev);


struct ctrl* ctrl_find_by_inode(const struct list* list, const struct inode* inode)
{
    const struct list_node* element = list_next(&list->head);
    struct ctrl* ctrl;

    while (element != NULL)
    {
        ctrl = container_of(element, struct ctrl, list);

        if (&ctrl->cdev == inode->i_cdev)
        {
            return ctrl;
        }

        element = list_next(element);
    }

    return NULL;
}
// EXPORT_SYMBOL_GPL(ctrl_find_by_inode);



/*
 * kobject release callback for ctrl->cdev.kobj.
 *
 * cdev is embedded in struct ctrl, so ctrl cannot be freed until
 * ALL kobject references to ctrl->cdev are dropped (including those
 * held by still-open file descriptors).  This callback is invoked
 * after cdev_del() + the final kobject_put(), guaranteeing that no
 * thread can access ctrl through the cdev or its kobj anymore --
 * with ONE exception, which is exactly the bug this handler must
 * defend against:
 *
 * VFS char-device inodes cache the cdev pointer in inode->i_cdev
 * and link themselves onto cdev->list through inode->i_devices
 * (chrdev_open).  That link is undone at inode eviction time
 * (cd_forget: list_del_init + i_cdev = NULL), which can run AFTER
 * the cdev kobject refcount already hit zero -- e.g. __fput does
 * cdev_put() first and dput() (-> inode eviction -> cd_forget)
 * afterwards.  kfree()ing ctrl while such an inode is still linked
 * makes the later cd_forget write into the freed cdev.list, which
 * shows up as SLUB "Poison overwritten" on the 16 bytes of
 * cdev.list (observed as ctrl+0xC0 corruption).
 *
 * The kernel's own ktype_cdev_default / cdev_dynamic_release
 * handlers solve this with cdev_purge(): before freeing they walk
 * cdev->list and detach every inode.  ctrl_chrdev_create() swaps in
 * this custom ktype, so the stock purge never runs; replicate it
 * here.  cdev_purge() itself is static and relies on the
 * unexported cdev_lock; replicating it WITHOUT the lock is safe at
 * this point because:
 *   - kobj refcount == 0 means no open file holds this cdev, and
 *     cdev_del() already unmapped us from cdev_map, so no NEW inode
 *     can attach (chrdev_open only attaches after a successful
 *     kobj_lookup);
 *   - a concurrent cd_forget() on a still-attached inode and our
 *     list_del_init() both write the same self-pointers, and
 *     inodes are RCU-freed (destroy_inode -> call_rcu), so any
 *     inode grabbed in this non-sleeping loop stays valid memory
 *     until we are done.
 */
static void ctrl_cdev_release(struct kobject *kobj)
{

    struct ctrl *ctrl = container_of(kobj, struct ctrl, cdev.kobj);
    printk(KERN_INFO "ctrl_cdev_release %s %p\n", ctrl->name, (void *)kobj);

    /*
     * Replicate cdev_purge(): detach every inode still hanging off
     * ctrl->cdev.list so its later eviction (cd_forget) cannot
     * write into the memory we are about to free.  After this loop,
     * any such inode's i_devices is self-pointing, so a future
     * list_del_init() only touches the inode itself.
     */
    while (!list_empty(&ctrl->cdev.list)) {
        struct inode *inode = container_of(ctrl->cdev.list.next,
                                           struct inode, i_devices);
        list_del_init(&inode->i_devices);
        inode->i_cdev = NULL;
    }

    kfree(ctrl->user_qid_bitmap);
    ctrl->user_qid_bitmap = NULL;
    mutex_destroy(&ctrl->user_qid_lock);
    kfree(ctrl);
}

static struct kobj_type ctrl_ktype = {.release = ctrl_cdev_release};

int ctrl_chrdev_create(struct ctrl* ctrl, dev_t first,
                       const struct file_operations* fops)
{
    int err;
    struct device* chrdev;

    if (ctrl->chrdev != NULL)
    {
        printk(KERN_WARNING "Character device is already created\n");
        return 0;
    }

    ctrl->rdev = MKDEV(MAJOR(first), ctrl->number);
    printk("nuo is %d\n", ctrl->rdev);

    cdev_init(&ctrl->cdev, fops);
    ctrl->cdev.kobj.ktype = &ctrl_ktype;

    err = cdev_add(&ctrl->cdev, ctrl->rdev, 1);
    if (err != 0) {
        printk(KERN_ERR "Failed to add cdev\n");
        /* cdev never mapped -> no fd can hold it, refcount == 1.
         * Take ownership of teardown; caller must NOT ctrl_put(). */
        list_remove(&ctrl->list);
        kobject_put(&ctrl->cdev.kobj);   /* -> ctrl_cdev_release -> kfree(ctrl) */
        return err;
    }

    chrdev = device_create(ctrl->cls, NULL, ctrl->rdev, NULL, ctrl->name);
    if (IS_ERR(chrdev)) {
        printk(KERN_ERR "Failed to create character device\n");
        err = PTR_ERR(chrdev);
        list_remove(&ctrl->list);
        cdev_del(&ctrl->cdev);           /* drops ref -> may kfree(ctrl) */
        return err;
    }

    ctrl->chrdev = chrdev;

    printk(KERN_INFO "Character device /dev/%s created (%d.%d)\n",
            ctrl->name, MAJOR(ctrl->rdev), MINOR(ctrl->rdev));

    return 0;
}
EXPORT_SYMBOL_GPL(ctrl_chrdev_create);


void ctrl_chrdev_remove(struct ctrl* ctrl)
{
    if (ctrl->chrdev != NULL)
    {
        /*
         * cdev_del() drops the base kobject reference that cdev_init()
         * acquired.  If all open file descriptors have already been
         * closed, the refcount will hit zero inside cdev_del() and
         * ctrl_cdev_release() will kfree(ctrl) synchronously.
         *
         * Save every field we need *after* cdev_del() into local
         * variables so we never touch a possibly-freed ctrl.
         */
        struct class *cls  = ctrl->cls;
        dev_t         rdev = ctrl->rdev;
        char          name[64];
        strscpy(name, ctrl->name, sizeof(name));

        // pci_dev_put(ctrl->pdev);
        printk("ctrl_chrdev_remove pci_dev_put\n");
        ctrl->chrdev = NULL;
        cdev_del(&ctrl->cdev);       /* may kfree(ctrl) here */
        device_destroy(cls, rdev);   /* use saved values */

        printk(KERN_DEBUG "Character device /dev/%s removed (%d.%d)\n",
                name, MAJOR(rdev), MINOR(rdev));
    }
}
EXPORT_SYMBOL_GPL(ctrl_chrdev_remove);

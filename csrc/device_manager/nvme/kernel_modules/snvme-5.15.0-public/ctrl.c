#define pr_fmt(fmt) "snvme: " fmt

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
        pr_crit("Failed to allocate controller reference\n");
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
         * released (e.g. after all open file descriptors are closed).
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
 * thread can access ctrl through the cdev or its kobj anymore.
 */
static void ctrl_cdev_release(struct kobject *kobj)
{

    struct ctrl *ctrl = container_of(kobj, struct ctrl, cdev.kobj);
    pr_info("ctrl_cdev_release %s %p\n", ctrl->name, (void *)kobj);
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
        pr_warn("Character device is already created\n");
        return 0;
    }

    ctrl->rdev = MKDEV(MAJOR(first), ctrl->number);
    pr_info("nuo is %d\n", ctrl->rdev);

    cdev_init(&ctrl->cdev, fops);
    ctrl->cdev.kobj.ktype = &ctrl_ktype;

    err = cdev_add(&ctrl->cdev, ctrl->rdev, 1);
    if (err != 0) {
        pr_err("Failed to add cdev\n");
        /* cdev never mapped -> no fd can hold it, refcount == 1.
         * Take ownership of teardown; caller must NOT ctrl_put(). */
        list_remove(&ctrl->list);
        kobject_put(&ctrl->cdev.kobj);   /* -> ctrl_cdev_release -> kfree(ctrl) */
        return err;
    }

    chrdev = device_create(ctrl->cls, NULL, ctrl->rdev, NULL, ctrl->name);
    if (IS_ERR(chrdev)) {
        pr_err("Failed to create character device\n");
        err = PTR_ERR(chrdev);
        list_remove(&ctrl->list);
        cdev_del(&ctrl->cdev);           /* drops ref -> may kfree(ctrl) */
        return err;
    }

    ctrl->chrdev = chrdev;

    pr_info("Character device /dev/%s created (%d.%d)\n",
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
        pr_info("ctrl_chrdev_remove pci_dev_put\n");
        ctrl->chrdev = NULL;
        cdev_del(&ctrl->cdev);       /* may kfree(ctrl) here */
        device_destroy(cls, rdev);   /* use saved values */

        pr_debug("Character device /dev/%s removed (%d.%d)\n",
                name, MAJOR(rdev), MINOR(rdev));
    }
}
EXPORT_SYMBOL_GPL(ctrl_chrdev_remove);

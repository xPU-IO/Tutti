// SPDX-License-Identifier: GPL-2.0
/*
 * snvm_control.c - module-level state: global registries, the singleton
 * /dev/snvm_control char device, PCI driver bind/unbind control plane.
 * Extracted from the 6.8 lineage pci.c; shared across baselines.
 */
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/idr.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/pci.h>
#include <linux/printk.h>
#include <linux/string.h>
#include <linux/workqueue.h>

#include "snvm_glue.h"
#include "snvm_dev_ioctl.h"
#include "snvm_qgroup.h"
#include "ctrl.h"
#include "map.h"
#include "nvme.h"
#include "compat.h"
#include "peer_memory.h"

/* ---------------- global registries + module params ---------------- */
static dev_t dev_first;
static DEFINE_IDA(snvm_chrdev_minor_ida);
dev_t  snvm_devno;
// static struct device snvm_dev; //snvme device
struct cdev snvm_cdev; //snvme cdev

static struct mutex snvm_control_lock;
static unsigned int snvm_registered;
/* Device class */
static struct class* dev_class;


/* List of controller devices */
struct list ctrl_list;


/* List of mapped host memory */
struct list host_list;


/* List of mapped device memory */
struct list device_list;

/* List of mapped device queue memory */
struct list device_queue_list;

/* Number of devices */
static int max_num_ctrls = 64;
module_param(max_num_ctrls, int, 0);
MODULE_PARM_DESC(max_num_ctrls, "Number of controller devices");

int nvme_num;
int gpu_num;

/* ---------------- /dev/ssnvme<N> creation / teardown lists ---------- */

int snvm_chrdev_create(struct pci_dev *pdev, unsigned int class){
	struct ctrl* ctrl = NULL;
	int minor, err;
	if (pdev->class != class) {
		printk("unexpected pci class mismatch, abort path find!\n");
		return -1;
	}
	minor = ida_simple_get(&snvm_chrdev_minor_ida, 0, 0, GFP_KERNEL);
	if (minor < 0)
		return minor;

	ctrl = ctrl_get(&ctrl_list, dev_class, pdev, minor);
	if (IS_ERR(ctrl)) {
		/* kmalloc failed: ctrl never entered ctrl_list and owns
		 * nothing, but the minor was already taken from the IDA. */
		ida_simple_remove(&snvm_chrdev_minor_ida, minor);
		return PTR_ERR(ctrl);
	}
	err = ctrl_chrdev_create(ctrl, dev_first, &snvm_dev_fops);
	if (err != 0) {
		/* ctrl_chrdev_create() already removed + freed ctrl on failure;
		* do NOT call ctrl_put(ctrl) here. */
		ida_simple_remove(&snvm_chrdev_minor_ida, minor);
		return err;
	}

	return 0;
}

/* ---------------- PCI driver registration (pointer-ized) ------------ */

unsigned long clear_ctrl_list(struct list* list)
{
    unsigned long i = 0;
    struct list_node* ptr = list_next(&list->head);
    struct ctrl* ctrl;

    while (ptr != NULL)
    {
        ctrl = container_of(ptr, struct ctrl, list);
		ctrl_put(ctrl);
        ++i;

        ptr = list_next(&list->head);
    }

    return i;
}

unsigned long clear_map_list(struct list* list)
{
    unsigned long i = 0;
    struct list_node* ptr = list_next(&list->head);
    struct map* map;

    while (ptr != NULL)
    {
        map = container_of(ptr, struct map, list);
        unmap_and_release(map);
        ++i;

        ptr = list_next(&list->head);
    }

    return i;
}

/* struct pci_driver snvme_driver lives in each baseline pci.c
 * (it names upstream static callbacks); the shared control plane
 * reaches it through snvm_pci_drv, set by snvm_pci_register(). */

static struct pci_driver *snvm_pci_drv;

static int snvm_register_driver(void)
{

	return pci_register_driver(snvm_pci_drv);
}

static void snvm_unregister_driver(void)
{
	pci_unregister_driver(snvm_pci_drv);
	flush_workqueue(s_nvme_wq);
}


static int register_driver(void){
	struct device_driver *dev_drv;
	int ret = 0;

	mutex_lock(&snvm_control_lock);
	dev_drv = driver_find(PCI_DRIVER_NAME, &pci_bus_type);
	if (!dev_drv && !snvm_registered){
		ret = snvm_register_driver();
		if (ret){
			printk("snvm register driver error\n");
		}else{
			snvm_registered = 1;
		}
	}
	mutex_unlock(&snvm_control_lock);
	return ret;
}

static int clean_driver(void){
	struct device_driver *dev_drv;
	unsigned long remaining = 0;

	//clear device mem map
	remaining = clear_map_list(&device_list);
	if (remaining){
		printk(KERN_NOTICE "%lu GPU memory mappings were still in use on unload\n", remaining);
	}

	//clear device io queue map
	remaining = clear_map_list(&device_queue_list);
	if (remaining){
		printk(KERN_NOTICE "%lu GPU memory mappings were still in use on unload\n", remaining);
	}

	remaining = clear_map_list(&host_list);
	if (remaining){
		printk(KERN_NOTICE "%lu host memory mappings were still in use on unload\n", remaining);
	}

	mutex_lock(&snvm_control_lock);
	dev_drv = driver_find(PCI_DRIVER_NAME, &pci_bus_type);
	if (dev_drv){
		snvm_unregister_driver();
	}
	mutex_unlock(&snvm_control_lock);
	return 0;
}

/* ---------------- bind/unbind control plane -------------------------- */
/*
 * SNVM_DEVICE_BIND: detach whatever driver owns the BDF (typically the
 * in-tree nvme driver), lazily register snvme on the PCI bus if not
 * already registered, then force-attach the device to snvme.
 *
 * driver_attach() (not device_attach): device_attach is bus->match()
 * based and picks the FIRST registered matching driver; on hosts where
 * the in-tree nvme.ko loaded before snvme.ko it silently rebinds the
 * device to nvme.ko.  driver_attach iterates the bus device list and,
 * for each device whose ->driver is NULL and whose id_table matches,
 * runs THIS driver's probe.  nvme_probe() has an explicit
 * ctrl_find_by_pci_dev(&ctrl_list, pdev) gate at the top, so only the
 * BDF the user pre-registered via SNVM_CHRDEV_CREATE actually probes.
 *
 * Race with udev autoprobe: after device_release_driver() the device is
 * briefly unbound and udev may rebind it to the in-tree driver.  We
 * close the window with a bounded retry loop; if udev wins three times
 * in a row the host has a misconfigured autoprobe rule and -EBUSY is
 * the honest answer.
 */
static int snvm_rebind_driver(struct pci_device_addr dev_addr)
{
	struct device_driver *dev_drv;
	struct pci_dev *pdev;
	int attempt;
	int ret = 0;

	pdev = TO_PCI_DEV(dev_addr);
	if (!pdev) {
		pr_err("pci_get_domain_bus_and_slot failed\n");
		return -ENODEV;
	}

	/* Detach any currently-bound driver so the slot is available to
	 * snvme.  pci_disable_device is the explicit counterpart of the
	 * implicit pci_enable_device that the *previous* driver (e.g.
	 * in-tree nvme) ran inside its probe; leaving it enabled would
	 * confuse snvme's own pci_enable_device on rebind.  */
	dev_drv = pdev->dev.driver;
	if (dev_drv && dev_drv->name) {
		if (pci_is_enabled(pdev)) {
			pci_disable_device(pdev);
			pr_info("(%s): disable device for bind new driver\n", __func__);
		}
		device_release_driver(&pdev->dev);
	}

	pr_info("binding nvme device to snvme: pci %x:%x:%x.%x\n",
		dev_addr.domain, dev_addr.bus, dev_addr.slot, dev_addr.func);

	if (register_driver()) {
		pr_err("register driver error\n");
		pci_dev_put(pdev);
		return -EFAULT;
	}

	/* Force-attach to snvme, fighting any udev autoprobe race. */
	for (attempt = 0; attempt < 3; attempt++) {
		dev_drv = pdev->dev.driver;
		if (dev_drv && dev_drv->name &&
		    strcmp(dev_drv->name, PCI_DRIVER_NAME) == 0) {
			ret = 0;
			break;
		}
		if (dev_drv) {
			pr_info("(%s): attempt %d: device autobound to '%s', releasing\n",
				__func__, attempt, dev_drv->name);
			if (pci_is_enabled(pdev))
				pci_disable_device(pdev);
			device_release_driver(&pdev->dev);
		}
		ret = driver_attach(&snvm_pci_drv->driver);
		if (ret) {
			pr_err("(%s): driver_attach failed: %d\n", __func__, ret);
			break;
		}
		/* driver_attach returns 0 regardless of whether ANY device
		 * was probed; check pdev->dev.driver to decide if we won
		 * the race on the target BDF. */
	}

	dev_drv = pdev->dev.driver;
	if (dev_drv && dev_drv->name &&
	    strcmp(dev_drv->name, PCI_DRIVER_NAME) == 0) {
		pr_info("device driver name: %s\n", dev_drv->name);
		ret = 0;
	} else {
		pr_err("(%s): failed to bind to %s after %d attempts (now: %s)\n",
		       __func__, PCI_DRIVER_NAME, attempt,
		       dev_drv ? dev_drv->name : "<none>");
		ret = -EBUSY;
	}

	pci_dev_put(pdev);
	return ret;
}

static int snvm_unbind_driver(struct pci_device_addr dev_addr){
	struct device_driver *dev_drv;
	struct pci_dev *pdev;

	pdev = TO_PCI_DEV(dev_addr);
	if (!pdev) {
		printk("(%s): pci_get_domain_bus_and_slot failed\n", __func__);
		return -EFAULT;
	}

	dev_drv = pdev->dev.driver;
	if (!dev_drv){
		printk("(%s): device do not have driver\n", __func__);
		return -EFAULT;
	}

	if (!dev_drv->name || strcmp(dev_drv->name, PCI_DRIVER_NAME) != 0){
		printk("(%s): device's driver is not %s\n", __func__, PCI_DRIVER_NAME);
		return -EFAULT;
	}

	if (pci_is_enabled(pdev)){
		pci_disable_device(pdev);
		printk("(%s): disable device for unbind driver\n", __func__);
	}
	pr_info("Unbinding device from driver %s\n", pdev->driver->name);
	device_release_driver(&pdev->dev);
	pci_dev_put(pdev);
	return 0;
}

static int snvm_chrdev_helper(struct pci_device_addr* dev_addr, int create){
	struct pci_device_addr pdev_addr;
	struct pci_dev *pdev;
	struct ctrl* ctrl;

	/*
	 * Idempotent default: a CHRDEV_CREATE for a BDF already created,
	 * or a CHRDEV_REMOVE for a BDF already gone, is treated as success.
	 * The original code initialised ret to -EFAULT here so those two
	 * fall-through cases silently returned errno=14 to userspace --
	 * which masquerades as a "Bad address" copy_from_user failure in
	 * diagnostic output and is what made snvme_smoke_gpu look like a
	 * pointer/CUDA bug.  See PORTING.md \xc2\xa77.3.1.
	 */
	int ret = 0;

	pdev_addr = *dev_addr;
	pdev = TO_PCI_DEV(pdev_addr);
	if (!pdev){
		printk("(%s): pci_get_domain_bus_and_slot failed\n", __func__);
		return -EFAULT;
	}

	ctrl = ctrl_find_by_pci_dev(&ctrl_list, pdev);
	if (create && !ctrl){ // create: register a fresh chrdev for this BDF
		ret = snvm_chrdev_create(pdev,PCI_CLASS_STORAGE_EXPRESS);
		if (!ret){
			ctrl = ctrl_find_by_pci_dev(&ctrl_list, pdev); // ctrl has been created by default
			memset(dev_addr, 0, sizeof(struct pci_device_addr));
			dev_addr->domain = ctrl->number;
		}
	} else if (create && ctrl){ // create: BDF already has a chrdev -- idempotent
		/*
		 * Userspace called CHRDEV_CREATE twice for the same BDF (typical
		 * shape: smoke test A finished and left its chrdev registered,
		 * smoke test B starts and calls CHRDEV_CREATE expecting to find
		 * out which /dev/ssnvme<N> to open).  Report the existing minor
		 * via dev_addr->domain (same protocol as the fresh-create path)
		 * and return success.  The cdev / class device / IDA minor are
		 * already in place from the previous CREATE -- nothing to do
		 * kernel-side.
		 */
		memset(dev_addr, 0, sizeof(struct pci_device_addr));
		dev_addr->domain = ctrl->number;
		ret = 0;
	} else if (!create && ctrl){ // remove: tear down the chrdev for this BDF
		/*
		 * Tear down order matters: ctrl_put() calls ctrl_chrdev_remove()
		 * which device_destroy()/cdev_del() uses ctrl->number to build
		 * MKDEV(). If we return the minor to the IDA pool *first*, a
		 * concurrent SNVM_CHRDEV_CREATE could pick up the same minor
		 * and race device_create() against our still-live cdev. Put
		 * first, then free the minor.
		 */
		int released_minor = ctrl->number;
		ctrl_put(ctrl);
		ida_simple_remove(&snvm_chrdev_minor_ida, released_minor);
		ret = 0;
	}
	/*
	 * (!create && !ctrl): remove on a BDF with no chrdev.  Idempotent;
	 * ret is already 0.
	 */

	pci_dev_put(pdev);
	return ret;
}

static long snvm_ioctl(struct file *file, unsigned int cmd,
			      unsigned long arg)
{
	struct pci_device_addr dev_addr;
	void __user *argp = (void __user *)arg;
	int ret;

	if (copy_from_user(&dev_addr, argp, sizeof(dev_addr))){
		printk("(%s): copy from user error\n", __func__);
		return -EFAULT;
	}

	switch (cmd){
		case SNVM_DEVICE_BIND:
			return snvm_rebind_driver(dev_addr);
		case SNVM_DEVICE_UNBIND:
			return snvm_unbind_driver(dev_addr);
		case SNVM_CHRDEV_CREATE:
			ret = snvm_chrdev_helper(&dev_addr, 1);
			if (!ret){
				ret = copy_to_user(argp, &dev_addr, sizeof(dev_addr));
			}
			return ret;
		case SNVM_CHRDEV_REMOVE:
			return snvm_chrdev_helper(&dev_addr, 0);
		default:
			return -ENOTTY;
	}
}

static const struct file_operations snvm_fops = {
	.owner	 = THIS_MODULE,
	.unlocked_ioctl	= snvm_ioctl,
};

/* ---------------- /dev/snvm_control singleton ------------------------ */
static char *get_snvme_mode(SNVM_DEVNODE_ARGS, umode_t *mode) {
    if (mode) *mode = 0666;
    return NULL;
}

static int snvm_cdev_init(void)
{
	int ret;
	struct device *device;

    mutex_init(&snvm_control_lock);

	dev_class = snvm_class_create(DRIVER_NAME);
	if (IS_ERR(dev_class)) {
		ret = PTR_ERR(dev_class);
		pr_err("failed to create class: %d\n", ret);
		return ret;
	}

	dev_class->devnode = get_snvme_mode;
	ret = alloc_chrdev_region(&dev_first, 0, max_num_ctrls, DRIVER_NAME);
	if (ret < 0) {
		pr_err("failed to allocate device numbers: %d\n", ret);
		goto destroy_subsys_class;
	}

	snvm_devno =  MKDEV(MAJOR(dev_first), max_num_ctrls);
	cdev_init(&snvm_cdev, &snvm_fops);
	snvm_cdev.owner = THIS_MODULE;
    ret = cdev_add(&snvm_cdev, snvm_devno, 1);
    if (ret < 0) {
        pr_err("failed to add cdev : %d\n", ret);
		goto err_unregister_chrdev;
    }

	// 创建设备文件
	device = device_create(dev_class, NULL, snvm_devno, NULL, "snvm_control");
	if (IS_ERR(device)) {
		ret = PTR_ERR(device);
		pr_err("failed to create device_create: %d\n", ret);
		goto destroy_cdev;
	}
	return 0;

destroy_cdev:
	cdev_del(&snvm_cdev);
err_unregister_chrdev:
	unregister_chrdev_region(dev_first, max_num_ctrls);
destroy_subsys_class:
	class_destroy(dev_class);
	return ret;
}

static void snvm_cdev_release(void)
{
	device_destroy(dev_class,snvm_devno);
	cdev_del(&snvm_cdev);
	unregister_chrdev_region(dev_first, max_num_ctrls);
	class_destroy(dev_class);
    mutex_destroy(&snvm_control_lock);
	ida_destroy(&snvm_chrdev_minor_ida);
	printk("snvme_helpers_cdev_release success!\n");
}

/*
 * snvm_pci_register - hand the baseline's struct pci_driver (which names
 * upstream static probe/remove/shutdown callbacks) to the shared control
 * plane.  Registration itself stays lazy (first SNVM_DEVICE_BIND).
 */
int snvm_pci_register(struct pci_driver *drv)
{
	snvm_pci_drv = drv;
	return 0;
}

int snvm_global_init(void)
{
	int ret;

	snvm_registered = 0;
	if (peer_memory_ops.init()) {
		pr_err("could not load peer_memory symbols\n");
		return -EOPNOTSUPP;
	}

	map_p2p_service_probe();

	list_init(&ctrl_list);
	list_init(&host_list);
	list_init(&device_list);
	list_init(&device_queue_list);

	ret = snvm_cdev_init();
	if (ret)
		return ret;
	return 0;
}

void snvm_global_exit(void)
{
	unsigned long leaked;

	/* Step 1: drain pinned map descriptors, stop accepting new probes and
	 * unbind any devices the snvme PCI driver owns (clean_driver: maps
	 * first -- a map's DMA unmap path dereferences its owning pdev,
	 * valid while ctrl lives -- then driver_find/unregister/flush). */
	clean_driver();

	/* Step 2: drop the Phoenix P2P service reference (maps drained above). */
	map_p2p_service_release();

	/* Step 3: drain orphaned /dev/ssnvme<N> ctrls before cdev_release so
	 * device_destroy/cdev_del inside ctrl_put see a valid dev_class. */
	leaked = clear_ctrl_list(&ctrl_list);
	if (leaked)
		pr_notice("drained %lu orphan ctrl(s) at unload (userspace forgot SNVM_CHRDEV_REMOVE)\n",
			  leaked);

	/* Step 4: singleton /dev/snvm_control + class. */
	snvm_cdev_release();

	/* Step 5: GPU/p2p notifier registration. */
	peer_memory_ops.exit();
}

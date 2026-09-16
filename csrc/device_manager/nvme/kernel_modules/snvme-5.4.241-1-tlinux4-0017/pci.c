// SPDX-License-Identifier: GPL-2.0
#define pr_fmt(fmt) "snvme: " fmt

/*
 * Modified NVM Express device driver -- snvme baseline for Linux 5.4
 * Copyright (c) 2011-2014, Intel Corporation.
 *
 * --------------------------------------------------------------------
 *  PORTING STATUS (snvme-5.4): COMPLETE
 *
 *  This file currently contains:
 *   - the upstream nvme/host/pci.c from kernel-5.4.241 (OpenCloudOS LTS
 *     and TencentOS 5.4.241-1-tlinux4-0017),
 *   - the minimal snvme symbol-rename pass (see snvme-rename.sed in
 *     this directory),
 *   - the full snvme-5.15 increment re-expressed against the 5.4
 *     struct nvme_dev and struct nvme_queue layout.
 *
 *  Per-segment status (relative to snvme-pci-5.15-incremental.diff,
 *  hunk numbering follows the order they appear in that diff):
 *    - segment 1-4 (struct nvme_dev field additions, MODULE_IMPORT_NS
 *      not needed in 5.4, sed renames)              : done
 *    - segment 5  (NVM_GET_DEV_INFO read-side)      : done
 *    - segment 6a (NVM_SET_IOQ_NUM on_host flag)    : done
 *    - segment 6b (nvme_setup_io_queues split)      : done
 *    - segment 6c (adapter_alloc_cq_user and _sq)   : done
 *    - segment 6d (nvme_create_user_queue + mix)    : done
 *    - segment 7  (user-IO-queue teardown in
 *      nvme_dev_disable; bug-for-bug DIVERGES from
 *      snvme-5.15, see snvme_disable_user_io_queues
 *      comment for the rationale)                   : done
 *    - modules 2-3 (the /dev/snvm_control device,
 *      the /dev/ssnvme<N> per-controller chrdev, the
 *      SNVM_xxx and NVM_xxx ioctl handlers,
 *      snvm_chrdev_create and snvm_chrdev_helper,
 *      snvm_rebind_driver, svm_mmap_registers, and
 *      the snvm_fops, snvm_dev_fops vtables)        : done
 *
 *  Bug fixes vs snvme-5.15 (all documented at the affected site, see
 *  PORTING.md section 7.3.1 for the regression-trap reference list):
 *    - snvm_dev_map_ioctl(): ret is initialised at declaration AND
 *      every success break sets ret = 0; in particular the
 *      NVM_UNMAP_xxx, NVM_SET_SHARE_REG and NVM_CLEAR_IOQ_NUM
 *      branches no longer leak stack garbage (trap #2).
 *    - NVM_MAP_HOST_MEMORY and NVM_MAP_DEVICE_QUEUE_MEMORY: a
 *      copy_to_user() failure on the IO-address writeback now rolls
 *      back ctrl->ioq_map_num, ctrl->cq_num and calls
 *      unmap_and_release(map) before returning -EFAULT.  The
 *      budget-overflow path also rolls back ioq_map_num before
 *      releasing (trap #4).
 *    - NVM_MAP_DEVICE_MEMORY: a copy_to_user() failure on the data
 *      path now calls unmap_and_release(map) instead of leaking the
 *      pinned p2p pages until module unload.
 *    - NVM_MAP_DEVICE_QUEUE_MEMORY: unmap_and_release(map) is
 *      called when ioq_idx<0 returns -EFAULT (5.15 leaked the
 *      fresh p2p mapping in this case).
 *    - nvme_create_user_queue(): ALL non-zero result values from
 *      adapter_alloc_sq_user() roll back the controller-side CQ
 *      (5.15 only rolled back for result > 0, leaking the CQ for
 *      the much more common transport / submit_sync failure case).
 *    - snvme_disable_user_io_queues() is a new teardown helper
 *      that tears the user-pinned QIDs down on every
 *      nvme_dev_disable() and reset cycle.  5.15 leaves them
 *      allocated on the controller and the next BIND fails with
 *      NVMe status 0x101 (Invalid Queue Identifier).
 *    - snvm_dev_fops now installs .open / .release hooks
 *      (snvm_dev_open / snvm_dev_release).  Upstream snvme-5.15.0
 *      and the original 5.4 port shipped this fops table without
 *      any release hook, so a userspace process dying between
 *      NVM_MAP_* and NVM_UNMAP_* leaked pinned host pages, NVIDIA
 *      p2p references and ctrl->ioq_map_num / cq_num counters.
 *      The observable failure mode on TencentOS 5.4.241 is:
 *        1. next SNVM_DEVICE_BIND logs "ctrl exist, ioq_num=N
 *           cq_num=M map_num=K" -- controller is reused dirty,
 *        2. nvidia.ko refcount accumulates -- "rmmod snvme"
 *           reports the module busy until reboot.
 *      The new release path uses map_purge_by_owner() (map.c) to
 *      walk host_list / device_list / device_queue_list and free
 *      every descriptor whose ->owner equals the task captured at
 *      .open time.  See PORTING.md section 7.3.1 for the full
 *      trap-list entry.
 *    - NVM_GET_DEV_INFO now flushes ctrl->scan_work and polls (5 s
 *      bounded) before declaring snvme_find_get_ns failure.
 *      Original code returned -EFAULT immediately and TencentOS
 *      5.4.241 logs "snvme_find_get_ns(nsid=1) failed" 3x on every
 *      BIND because snvme_start_ctrl enqueues nvme_scan_work on
 *      s_nvme_wq asynchronously: probe returns and the ssnvme
 *      cdev becomes callable before nsid=1 is list_add_tail'd to
 *      ctrl->namespaces.  This race is independent of the
 *      dirty-rebind path above and reproduces on a fresh module
 *      load.  See PORTING.md section 7.3.1 trap "NVM_GET_DEV_INFO
 *      vs nvme_scan_work" for the full reasoning.
 *    - snvm_rebind_driver: device_attach() replaced with
 *      driver_attach(&snvme_driver.driver) + bounded retry loop.
 *      device_attach is bus->match() based and picks the FIRST
 *      registered matching driver -- on TencentOS 5.4.241 the
 *      in-tree nvme.ko is loaded at boot, so it always wins,
 *      silently rebinding the device to nvme.ko (signature in
 *      dmesg: "nvme nvme0: pci function ..." + "135/0/0 default/
 *      read/poll queues" 3-tuple instead of snvme's 4-tuple).
 *      SNVM_DEVICE_UNBIND then returns -EFAULT and the smoke test
 *      reports "Bad address".  driver_attach() walks the bus once
 *      and force-binds the snvme driver to any matching unbound
 *      device -- the nvme_probe gate ctrl_find_by_pci_dev != NULL
 *      keeps it scoped to BDFs the user already CHRDEV_CREATEd.
 *      Reproducer: ./run_snvme_smoke.sh --gpu (no bind) then
 *      ./run_snvme_smoke.sh --gpu --bind on the same BDF; fails
 *      reliably at step 15 SNVM_DEVICE_UNBIND on the old code.
 *    - snvm_unbind_driver: -EFAULT replaced with -ENODEV / -EINVAL
 *      on the two failure branches.  -EFAULT means "bad userspace
 *      address" and misled callers into thinking they passed a
 *      wrong dev_addr; the actual semantics are "no driver to
 *      unbind" (-ENODEV) and "device owned by a different driver"
 *      (-EINVAL).
 *
 *  Co-existence (PORTING.md section 2): every "nvme%d..." literal in
 *  core.c, multipath.c, nvme.h and pci.c (sysfs device name, gendisk
 *  name, IRQ name) has been renamed to "snvme%d..." so that the
 *  in-tree nvme.ko and snvme.ko can each bind a different NVMe on
 *  the same host without /dev or /proc/interrupts collisions.
 *  snvme-rename.sed cannot touch these because they are string
 *  literals, not C identifiers.
 *
 *  Build: Makefile.in builds both snvme-core.ko and snvme.ko.  Both
 *  are required: libnvm opens /dev/snvm_control via snvme.ko and
 *  reads disk names produced by the namespace path in snvme-core.ko.
 *
 *  Reference: snvme-pci-5.15-incremental.diff in this directory is
 *  the unified diff between upstream nvme-5.15.0/pci.c and
 *  snvme-5.15.0-public/pci.c, i.e. the snvme increment that was
 *  re-expressed against the 5.4 layout to produce this file.  Keep
 *  it in tree as the canonical reference for the next baseline
 *  uplift.
 * --------------------------------------------------------------------
 */

#include <linux/aer.h>
#include <linux/async.h>
#include <linux/blkdev.h>
#include <linux/blk-mq.h>
#include <linux/blk-mq-pci.h>
#include <linux/cdev.h>
#include <linux/delay.h>          /* msleep() for NVM_GET_DEV_INFO scan-race wait */
#include <linux/dmi.h>
#include <linux/idr.h>
#include <linux/init.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/ioctl.h>           /* _IOC_TYPE() in snvm_dev_map_ioctl default */
#include <linux/list.h>            /* list_head / INIT_LIST_HEAD for queue groups */
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/once.h>
#include <linux/pci.h>
#include <linux/suspend.h>
#include <linux/t10-pi.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/io-64-nonatomic-lo-hi.h>
#include <linux/sed-opal.h>
#include <linux/pci-p2pdma.h>

/*
 * snvme intentionally drops the upstream NVMe tracepoint subsystem
 * (drivers/nvme/host/trace.{c,h}); see core.c for the full rationale.
 * The single trace_nvme_sq() call site below is no-op'd to match the
 * snvme-5.15.0 baseline.
 */
#include "nvme.h"

/* snvme helpers (kernel-version-agnostic, shared with snvme-5.15.0) */
#include "ctrl.h"
#include "list.h"
#include "map.h"
#include "ioctl.h"
#include "peer_memory.h"

/*
 * DRIVER_NAME identifies the snvme control/char-device subsystem in log
 * messages; PCI_DRIVER_NAME is the sysfs name under which the PCI core
 * binds this driver to NVMe devices -- it MUST differ from the stock
 * "nvme" so that `snvme` and the in-tree `nvme` driver can coexist and
 * libnvm's SNVM_DEVICE_BIND/UNBIND can driver_find() us by name.
 */
#define DRIVER_NAME         "libsnvm helper"
#define PCI_DRIVER_NAME     "snvme"

/* ------------------------------------------------------------------ *
 *  snvme: control-device + helper global state
 *
 *  These globals back /dev/snvm_control (singleton char device created
 *  at module init) and the per-controller /dev/ssnvme<domain> char
 *  devices created on demand via the SNVM_CHRDEV_CREATE ioctl.  They
 *  are populated by the snvme init/exit hooks and consumed by the
 *  ioctl dispatchers; both are ported in later segments.
 *
 *  Layout mirrors snvme-5.15.0/pci.c lines 48-72 so the SNVM_* / NVM_*
 *  ioctl handlers port across baselines with minimal edits.
 * ------------------------------------------------------------------ */
static dev_t            dev_first;                  /* base major for ssnvme%d minors */
static DEFINE_IDA(snvm_chrdev_minor_ida);           /* minor allocator for ssnvme%d   */
dev_t                   snvm_devno;                 /* devno of /dev/snvm_control     */
struct cdev             snvm_cdev;                  /* cdev for /dev/snvm_control     */

static struct mutex     snvm_control_lock;          /* serialises rebind / chrdev ops */
static unsigned int     snvm_registered;            /* 1 iff snvme PCI driver bound   */

/* sysfs class shared by /dev/snvm_control and /dev/ssnvme* devices. */
static struct class    *dev_class;

/* Per-controller handle list, keyed by pci_dev.  Populated by
 * snvm_chrdev_create(); walked by ctrl_find_by_pci_dev() / _by_inode().
 */
static struct list      ctrl_list;

/* Pinned host / device / device-queue DMA mapping lists.  Populated by
 * NVM_MAP_* ioctls, drained by NVM_UNMAP_* / module exit.
 */
static struct list      host_list;
static struct list      device_list;
static struct list      device_queue_list;

/*
 * Per-fd queue group support (NVM_CREATE_QUEUE_GROUP /
 * NVM_DESTROY_QUEUE_GROUP).
 *
 * snvm_queue_group_ida is the allocator for the opaque group_id
 * returned to userspace.  We bias allocations to >= 1 because
 * group_id 0 is reserved as an "invalid / no group" sentinel
 * (see ioctl.h struct nvm_ioctl_queue_group).  Lifetime: per-
 * module; ida_destroy() is paired with ida_destroy() of the
 * minor allocator in module exit (see pci.c snvm_helpers_release).
 *
 * A group's actual descriptor (struct snvm_qgroup, declared
 * below near snvm_dev_owner) is hung off the owning fd's
 * file->private_data.  This module-global IDA only owns the id
 * namespace, not the descriptors.  No global descriptor list is
 * needed: every group is reachable exactly once via its owning
 * fd, which guarantees fd-close cascade-cleanup is sufficient to
 * leak-free unwind.
 */
static DEFINE_IDA(snvm_queue_group_ida);

/* Upper bound on how many NVMe controllers snvme will manage
 * concurrently (= max number of /dev/ssnvme%d char devices).
 */
static int              max_num_ctrls = 64;
module_param(max_num_ctrls, int, 0);
MODULE_PARM_DESC(max_num_ctrls, "Number of controller devices");

/* Resolve a libnvm pci_device_addr (domain/bus/slot/func) to a
 * struct pci_dev* via the PCI core.  Used by SNVM_* ioctls that
 * address an NVMe device by BDF rather than by fd.
 */
#define TO_PCI_DEV(addr) \
	pci_get_domain_bus_and_slot((addr).domain, (addr).bus, \
				    PCI_DEVFN((addr).slot, (addr).func))

#define SQ_SIZE(q)	((q)->q_depth << (q)->sqes)
#define CQ_SIZE(q)	((q)->q_depth * sizeof(struct nvme_completion))

#define SGES_PER_PAGE	(PAGE_SIZE / sizeof(struct nvme_sgl_desc))

/*
 * These can be higher, but we need to ensure that any command doesn't
 * require an sg allocation that needs more than a page of data.
 */
#define NVME_MAX_KB_SZ	4096
#define NVME_MAX_SEGS	127

static int use_threaded_interrupts;
module_param(use_threaded_interrupts, int, 0);

static bool use_cmb_sqes = true;
module_param(use_cmb_sqes, bool, 0444);
MODULE_PARM_DESC(use_cmb_sqes, "use controller's memory buffer for I/O SQes");

static unsigned int max_host_mem_size_mb = 128;
module_param(max_host_mem_size_mb, uint, 0444);
MODULE_PARM_DESC(max_host_mem_size_mb,
	"Maximum Host Memory Buffer (HMB) size per controller (in MiB)");

static unsigned int sgl_threshold = SZ_32K;
module_param(sgl_threshold, uint, 0644);
MODULE_PARM_DESC(sgl_threshold,
		"Use SGLs when average request segment size is larger or equal to "
		"this size. Use 0 to disable SGLs.");

/*
 * snvme: no io_queue_depth module parameter.  The queue depth always
 * takes the controller's maximum: NVMe CAP.MQES + 1.  User IOQs created
 * via NVM_ADD_USER_QUEUE support multi-page (physically-fragmented) ring
 * memory, so the old single-page clamp (depth <= 64) no longer applies.
 * Applies to every user queue -- snvme does not support per-queue depth.
 */

static unsigned int write_queues;
module_param(write_queues, uint, 0644);
MODULE_PARM_DESC(write_queues,
	"Number of queues to use for writes. If not set, reads and writes "
	"will share a queue set.");

static unsigned int poll_queues;
module_param(poll_queues, uint, 0644);
MODULE_PARM_DESC(poll_queues, "Number of queues to use for polled IO.");

static unsigned int smp_affinity_enable = 1;
module_param(smp_affinity_enable, uint, 0644);
MODULE_PARM_DESC(smp_affinity_enable, "SMP affinity feature enable/disbale Default: enable(1)");

struct nvme_dev;
struct nvme_queue;

static void nvme_dev_disable(struct nvme_dev *dev, bool shutdown);
static bool __nvme_disable_io_queues(struct nvme_dev *dev, u8 opcode);

/*
 * Represents an NVM Express device.  Each nvme_dev is a PCI function.
 */
struct nvme_dev {
	struct nvme_queue *queues;
	struct blk_mq_tag_set tagset;
	struct blk_mq_tag_set admin_tagset;
	u32 __iomem *dbs;
	struct device *dev;
	struct dma_pool *prp_page_pool;
	struct dma_pool *prp_small_pool;
	unsigned online_queues;
	unsigned max_qid;
	unsigned io_queues[HCTX_MAX_TYPES];
	unsigned int num_vecs;
	int q_depth;
	int io_sqes;
	u32 db_stride;
	void __iomem *bar;
	unsigned long bar_mapped_size;
	struct work_struct remove_work;
	struct mutex shutdown_lock;
	bool subsystem;
	u64 cmb_size;
	bool cmb_use_sqes;
	u32 cmbsz;
	u32 cmbloc;
	struct nvme_ctrl ctrl;
	u32 last_ps;

	mempool_t *iod_mempool;

	/* shadow doorbell buffer support: */
	__le32 *dbbuf_dbs;
	dma_addr_t dbbuf_dbs_dma_addr;
	__le32 *dbbuf_eis;
	dma_addr_t dbbuf_eis_dma_addr;

	/* host memory buffer support: */
	u64 host_mem_size;
	u32 nr_host_mem_descs;
	dma_addr_t host_mem_descs_dma;
	struct nvme_host_mem_buf_desc *host_mem_descs;
	void **host_mem_desc_bufs;
	unsigned int nr_allocated_queues;
	unsigned int nr_write_queues;
	unsigned int nr_poll_queues;

	/* ------------------------------------------------------------ *
	 *  snvme: CPU/GPU IO-queue sharing extension fields.
	 *
	 *  Mirrored verbatim from snvme-5.15.0/pci.c so that segments
	 *  5 (NVM_GET_DEV_INFO ioctl read-side) and 6 (queue-share
	 *  hooks in nvme_setup_io_queues / nvme_alloc_queue / friends)
	 *  reference identical names across both kernel baselines.
	 *  None of these fields touch upstream nvme_dev semantics --
	 *  they record snvme-only state populated only when the user
	 *  has driven the ssnvme cdev's NVM_SET_IOQ_NUM /
	 *  NVM_SET_SHARE_REG ioctls.
	 *
	 *   user_start_qid       : first QID handed to userspace; queues
	 *                          [0, user_start_qid) remain kernel-owned
	 *                          (admin + kernel I/O), queues
	 *                          [user_start_qid, online_user_queues)
	 *                          are user-allocated.
	 *   online_user_queues   : count of QIDs currently online for user.
	 * ------------------------------------------------------------ */
	unsigned int online_user_queues;
	unsigned int user_start_qid;
	/*
	 * Optional caller-imposed cap on the kernel-side IO-queue count
	 * requested from the controller, populated at SNVM_DEVICE_BIND
	 * time from ctrl->setup.cap_kernel_ioq (which itself is set by
	 * the userspace NVM_SET_IOQ_NUM ioctl).
	 *
	 * Zero means "no override, fall back to num_possible_cpus()" --
	 * upstream nvme's behaviour.  A non-zero value lets the user
	 * shrink the kernel's IOQ ask so the controller's Set-Features
	 * grant has room left for the user-allocated share without
	 * triggering the fallback-to-dma_alloc_coherent branch in
	 * s_nvme_setup_io_queues(): typical use case is a NVMe whose
	 * MSI-X count is smaller than num_possible_cpus() on the host.
	 */
	unsigned int cap_kernel_ioq;
	/*
	 * snvme B3: controller-granted total IO queue count.  Written
	 * inside s_nvme_setup_io_queues() right after snvme_set_queue_count
	 * returns -- this is the authoritative upper bound on legal IOQ
	 * QIDs (1..ctrl_max_io_queues).  The kernel keeps the first
	 * (online_queues - 1) of those for its own use; the rest
	 * (online_queues..ctrl_max_io_queues) are handed to userspace
	 * via NVM_ADD_USER_QUEUE.
	 *
	 * Distinct from nr_allocated_queues, which is the snvme-side
	 * dev->queues[] capacity (admin + nvme_max_io_queues()).  On
	 * hosts where num_possible_cpus() > controller MSI-X count,
	 * nr_allocated_queues is significantly larger than what the
	 * controller will actually accept, and using nr_allocated_queues
	 * as the user QID upper bound triggers Invalid Queue Identifier
	 * (SC=0x4101) on Create I/O CQ.
	 */
	unsigned int ctrl_max_io_queues;
};

static inline unsigned int sq_idx(unsigned int qid, u32 stride)
{
	return qid * 2 * stride;
}

static inline unsigned int cq_idx(unsigned int qid, u32 stride)
{
	return (qid * 2 + 1) * stride;
}

static inline struct nvme_dev *to_nvme_dev(struct nvme_ctrl *ctrl)
{
	return container_of(ctrl, struct nvme_dev, ctrl);
}

/*
 * An NVM Express queue.  Each device has at least two (one for admin
 * commands and one for I/O commands).
 */
struct nvme_queue {
	struct nvme_dev *dev;
	spinlock_t sq_lock;
	void *sq_cmds;
	 /* only used for poll queues: */
	spinlock_t cq_poll_lock ____cacheline_aligned_in_smp;
	volatile struct nvme_completion *cqes;
	dma_addr_t sq_dma_addr;
	dma_addr_t cq_dma_addr;
	u32 __iomem *q_db;
	u16 q_depth;
	u16 cq_vector;
	u16 sq_tail;
	u16 last_sq_tail;
	u16 cq_head;
	u16 last_cq_head;
	u16 qid;
	u8 cq_phase;
	u8 sqes;
	unsigned long flags;
#define NVMEQ_ENABLED		0
#define NVMEQ_SQ_CMB		1
#define NVMEQ_DELETE_ERROR	2
#define NVMEQ_POLLED		3
	__le32 *dbbuf_sq_db;
	__le32 *dbbuf_cq_db;
	__le32 *dbbuf_sq_ei;
	__le32 *dbbuf_cq_ei;
	struct completion delete_done;
};

/*
 * The nvme_iod describes the data in an I/O.
 *
 * The sg pointer contains the list of PRP/SGL chunk allocations in addition
 * to the actual struct scatterlist.
 */
struct nvme_iod {
	struct nvme_request req;
	struct nvme_queue *nvmeq;
	bool use_sgl;
	int aborted;
	int npages;		/* In the PRP list. 0 means small pool in use */
	int nents;		/* Used in scatterlist */
	dma_addr_t first_dma;
	unsigned int dma_len;	/* length of single DMA segment mapping */
	dma_addr_t meta_dma;
	struct scatterlist *sg;
};

static inline unsigned int nvme_dbbuf_size(struct nvme_dev *dev)
{
	return dev->nr_allocated_queues * 8 * dev->db_stride;
}

static int nvme_dbbuf_dma_alloc(struct nvme_dev *dev)
{
	unsigned int mem_size = nvme_dbbuf_size(dev);

	if (dev->dbbuf_dbs)
		return 0;

	dev->dbbuf_dbs = dma_alloc_coherent(dev->dev, mem_size,
					    &dev->dbbuf_dbs_dma_addr,
					    GFP_KERNEL);
	if (!dev->dbbuf_dbs)
		return -ENOMEM;
	dev->dbbuf_eis = dma_alloc_coherent(dev->dev, mem_size,
					    &dev->dbbuf_eis_dma_addr,
					    GFP_KERNEL);
	if (!dev->dbbuf_eis) {
		dma_free_coherent(dev->dev, mem_size,
				  dev->dbbuf_dbs, dev->dbbuf_dbs_dma_addr);
		dev->dbbuf_dbs = NULL;
		return -ENOMEM;
	}

	return 0;
}

static void nvme_dbbuf_dma_free(struct nvme_dev *dev)
{
	unsigned int mem_size = nvme_dbbuf_size(dev);

	if (dev->dbbuf_dbs) {
		dma_free_coherent(dev->dev, mem_size,
				  dev->dbbuf_dbs, dev->dbbuf_dbs_dma_addr);
		dev->dbbuf_dbs = NULL;
	}
	if (dev->dbbuf_eis) {
		dma_free_coherent(dev->dev, mem_size,
				  dev->dbbuf_eis, dev->dbbuf_eis_dma_addr);
		dev->dbbuf_eis = NULL;
	}
}

static void nvme_dbbuf_init(struct nvme_dev *dev,
			    struct nvme_queue *nvmeq, int qid)
{
	if (!dev->dbbuf_dbs || !qid)
		return;

	nvmeq->dbbuf_sq_db = &dev->dbbuf_dbs[sq_idx(qid, dev->db_stride)];
	nvmeq->dbbuf_cq_db = &dev->dbbuf_dbs[cq_idx(qid, dev->db_stride)];
	nvmeq->dbbuf_sq_ei = &dev->dbbuf_eis[sq_idx(qid, dev->db_stride)];
	nvmeq->dbbuf_cq_ei = &dev->dbbuf_eis[cq_idx(qid, dev->db_stride)];
}

static void nvme_dbbuf_free(struct nvme_queue *nvmeq)
{
	if (!nvmeq->qid)
		return;

	nvmeq->dbbuf_sq_db = NULL;
	nvmeq->dbbuf_cq_db = NULL;
	nvmeq->dbbuf_sq_ei = NULL;
	nvmeq->dbbuf_cq_ei = NULL;
}

static void nvme_dbbuf_set(struct nvme_dev *dev)
{
	struct nvme_command c;
	unsigned int i;

	if (!dev->dbbuf_dbs)
		return;

	memset(&c, 0, sizeof(c));
	c.dbbuf.opcode = nvme_admin_dbbuf;
	c.dbbuf.prp1 = cpu_to_le64(dev->dbbuf_dbs_dma_addr);
	c.dbbuf.prp2 = cpu_to_le64(dev->dbbuf_eis_dma_addr);

	if (snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0)) {
		dev_warn(dev->ctrl.device, "unable to set dbbuf\n");
		/* Free memory and continue on */
		nvme_dbbuf_dma_free(dev);

		for (i = 1; i <= dev->online_queues; i++)
			nvme_dbbuf_free(&dev->queues[i]);
	}
}

static inline int nvme_dbbuf_need_event(u16 event_idx, u16 new_idx, u16 old)
{
	return (u16)(new_idx - event_idx - 1) < (u16)(new_idx - old);
}

/* Update dbbuf and return true if an MMIO is required */
static bool nvme_dbbuf_update_and_check_event(u16 value, __le32 *dbbuf_db,
					      volatile __le32 *dbbuf_ei)
{
	if (dbbuf_db) {
		u16 old_value, event_idx;

		/*
		 * Ensure that the queue is written before updating
		 * the doorbell in memory
		 */
		wmb();

		old_value = le32_to_cpu(*dbbuf_db);
		*dbbuf_db = cpu_to_le32(value);

		/*
		 * Ensure that the doorbell is updated before reading the event
		 * index from memory.  The controller needs to provide similar
		 * ordering to ensure the envent index is updated before reading
		 * the doorbell.
		 */
		mb();

		event_idx = le32_to_cpu(*dbbuf_ei);
		if (!nvme_dbbuf_need_event(event_idx, value, old_value))
			return false;
	}

	return true;
}

/*
 * Will slightly overestimate the number of pages needed.  This is OK
 * as it only leads to a small amount of wasted memory for the lifetime of
 * the I/O.
 */
static int nvme_npages(unsigned size, struct nvme_dev *dev)
{
	unsigned nprps = DIV_ROUND_UP(size + dev->ctrl.page_size,
				      dev->ctrl.page_size);
	return DIV_ROUND_UP(8 * nprps, PAGE_SIZE - 8);
}

/*
 * Calculates the number of pages needed for the SGL segments. For example a 4k
 * page can accommodate 256 SGL descriptors.
 */
static int nvme_pci_npages_sgl(unsigned int num_seg)
{
	return DIV_ROUND_UP(num_seg * sizeof(struct nvme_sgl_desc), PAGE_SIZE);
}

static unsigned int nvme_pci_iod_alloc_size(struct nvme_dev *dev,
		unsigned int size, unsigned int nseg, bool use_sgl)
{
	size_t alloc_size;

	if (use_sgl)
		alloc_size = sizeof(__le64 *) * nvme_pci_npages_sgl(nseg);
	else
		alloc_size = sizeof(__le64 *) * nvme_npages(size, dev);

	return alloc_size + sizeof(struct scatterlist) * nseg;
}

static int nvme_admin_init_hctx(struct blk_mq_hw_ctx *hctx, void *data,
				unsigned int hctx_idx)
{
	struct nvme_dev *dev = data;
	struct nvme_queue *nvmeq = &dev->queues[0];

	WARN(hctx_idx != 0, "snvme: unexpected admin hctx index %u\n",
	     hctx_idx);
	WARN(dev->admin_tagset.tags[0] != hctx->tags,
	     "snvme: admin tagset mismatch\n");

	hctx->driver_data = nvmeq;
	return 0;
}

static int nvme_init_hctx(struct blk_mq_hw_ctx *hctx, void *data,
			  unsigned int hctx_idx)
{
	struct nvme_dev *dev = data;
	struct nvme_queue *nvmeq = &dev->queues[hctx_idx + 1];

	WARN(dev->tagset.tags[hctx_idx] != hctx->tags,
	     "snvme: tagset mismatch for hctx %u\n", hctx_idx);
	hctx->driver_data = nvmeq;
	return 0;
}

static int nvme_init_request(struct blk_mq_tag_set *set, struct request *req,
		unsigned int hctx_idx, unsigned int numa_node)
{
	struct nvme_dev *dev = set->driver_data;
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	int queue_idx = (set == &dev->tagset) ? hctx_idx + 1 : 0;
	struct nvme_queue *nvmeq = &dev->queues[queue_idx];

	BUG_ON(!nvmeq);
	iod->nvmeq = nvmeq;

	nvme_req(req)->ctrl = &dev->ctrl;
	return 0;
}

static int queue_irq_offset(struct nvme_dev *dev)
{
	/* if we have more than 1 vec, admin queue offsets us by 1 */
	if (dev->num_vecs > 1)
		return 1;

	return 0;
}

static int nvme_pci_map_queues(struct blk_mq_tag_set *set)
{
	struct nvme_dev *dev = set->driver_data;
	int i, qoff, offset;

	offset = queue_irq_offset(dev);
	for (i = 0, qoff = 0; i < set->nr_maps; i++) {
		struct blk_mq_queue_map *map = &set->map[i];

		map->nr_queues = dev->io_queues[i];
		if (!map->nr_queues) {
			BUG_ON(i == HCTX_TYPE_DEFAULT);
			continue;
		}

		/*
		 * The poll queue(s) doesn't have an IRQ (and hence IRQ
		 * affinity), so use the regular blk-mq cpu mapping
		 */
		map->queue_offset = qoff;
		if (i != HCTX_TYPE_POLL && offset && smp_affinity_enable)
			blk_mq_pci_map_queues(map, to_pci_dev(dev->dev), offset);
		else
			blk_mq_map_queues(map);
		qoff += map->nr_queues;
		offset += map->nr_queues;
	}

	return 0;
}

/*
 * Write sq tail if we are asked to, or if the next command would wrap.
 */
static inline void nvme_write_sq_db(struct nvme_queue *nvmeq, bool write_sq)
{
	if (!write_sq) {
		u16 next_tail = nvmeq->sq_tail + 1;

		if (next_tail == nvmeq->q_depth)
			next_tail = 0;
		if (next_tail != nvmeq->last_sq_tail)
			return;
	}

	if (nvme_dbbuf_update_and_check_event(nvmeq->sq_tail,
			nvmeq->dbbuf_sq_db, nvmeq->dbbuf_sq_ei))
		writel(nvmeq->sq_tail, nvmeq->q_db);
	nvmeq->last_sq_tail = nvmeq->sq_tail;
}

/**
 * nvme_submit_cmd() - Copy a command into a queue and ring the doorbell
 * @nvmeq: The queue to use
 * @cmd: The command to send
 * @write_sq: whether to write to the SQ doorbell
 */
static void nvme_submit_cmd(struct nvme_queue *nvmeq, struct nvme_command *cmd,
			    bool write_sq)
{
	spin_lock(&nvmeq->sq_lock);
	memcpy(nvmeq->sq_cmds + (nvmeq->sq_tail << nvmeq->sqes),
	       cmd, sizeof(*cmd));
	if (++nvmeq->sq_tail == nvmeq->q_depth)
		nvmeq->sq_tail = 0;
	nvme_write_sq_db(nvmeq, write_sq);
	spin_unlock(&nvmeq->sq_lock);
}

static void nvme_commit_rqs(struct blk_mq_hw_ctx *hctx)
{
	struct nvme_queue *nvmeq = hctx->driver_data;

	spin_lock(&nvmeq->sq_lock);
	if (nvmeq->sq_tail != nvmeq->last_sq_tail)
		nvme_write_sq_db(nvmeq, true);
	spin_unlock(&nvmeq->sq_lock);
}

static void **nvme_pci_iod_list(struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	return (void **)(iod->sg + blk_rq_nr_phys_segments(req));
}

static inline bool nvme_pci_use_sgls(struct nvme_dev *dev, struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	int nseg = blk_rq_nr_phys_segments(req);
	unsigned int avg_seg_size;

	if (nseg == 0)
		return false;

	avg_seg_size = DIV_ROUND_UP(blk_rq_payload_bytes(req), nseg);

	if (!(dev->ctrl.sgls & ((1 << 0) | (1 << 1))))
		return false;
	if (!iod->nvmeq->qid)
		return false;
	if (!sgl_threshold || avg_seg_size < sgl_threshold)
		return false;
	return true;
}

static void nvme_free_prps(struct nvme_dev *dev, struct request *req)
{
	const int last_prp = dev->ctrl.page_size / sizeof(__le64) - 1;
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	dma_addr_t dma_addr = iod->first_dma;
	int i;

	for (i = 0; i < iod->npages; i++) {
		__le64 *prp_list = nvme_pci_iod_list(req)[i];
		dma_addr_t next_dma_addr = le64_to_cpu(prp_list[last_prp]);

		dma_pool_free(dev->prp_page_pool, prp_list, dma_addr);
		dma_addr = next_dma_addr;
	}

}

static void nvme_free_sgls(struct nvme_dev *dev, struct request *req)
{
	const int last_sg = SGES_PER_PAGE - 1;
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	dma_addr_t dma_addr = iod->first_dma;
	int i;

	for (i = 0; i < iod->npages; i++) {
		struct nvme_sgl_desc *sg_list = nvme_pci_iod_list(req)[i];
		dma_addr_t next_dma_addr = le64_to_cpu((sg_list[last_sg]).addr);

		dma_pool_free(dev->prp_page_pool, sg_list, dma_addr);
		dma_addr = next_dma_addr;
	}

}

static void nvme_unmap_sg(struct nvme_dev *dev, struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);

	if (is_pci_p2pdma_page(sg_page(iod->sg)))
		pci_p2pdma_unmap_sg(dev->dev, iod->sg, iod->nents,
				    rq_dma_dir(req));
	else
		dma_unmap_sg(dev->dev, iod->sg, iod->nents, rq_dma_dir(req));
}

static void nvme_unmap_data(struct nvme_dev *dev, struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);

	if (iod->dma_len) {
		dma_unmap_page(dev->dev, iod->first_dma, iod->dma_len,
			       rq_dma_dir(req));
		return;
	}

	WARN_ON_ONCE(!iod->nents);

	nvme_unmap_sg(dev, req);
	if (iod->npages == 0)
		dma_pool_free(dev->prp_small_pool, nvme_pci_iod_list(req)[0],
			      iod->first_dma);
	else if (iod->use_sgl)
		nvme_free_sgls(dev, req);
	else
		nvme_free_prps(dev, req);
	mempool_free(iod->sg, dev->iod_mempool);
}

static void nvme_print_sgl(struct scatterlist *sgl, int nents)
{
	int i;
	struct scatterlist *sg;

	for_each_sg(sgl, sg, nents, i) {
		dma_addr_t phys = sg_phys(sg);
		pr_warn("sg[%d] phys_addr:%pad offset:%d length:%d "
			"dma_address:%pad dma_length:%d\n",
			i, &phys, sg->offset, sg->length, &sg_dma_address(sg),
			sg_dma_len(sg));
	}
}

static blk_status_t nvme_pci_setup_prps(struct nvme_dev *dev,
		struct request *req, struct nvme_rw_command *cmnd)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct dma_pool *pool;
	int length = blk_rq_payload_bytes(req);
	struct scatterlist *sg = iod->sg;
	int dma_len = sg_dma_len(sg);
	u64 dma_addr = sg_dma_address(sg);
	u32 page_size = dev->ctrl.page_size;
	int offset = dma_addr & (page_size - 1);
	__le64 *prp_list;
	void **list = nvme_pci_iod_list(req);
	dma_addr_t prp_dma;
	int nprps, i;

	length -= (page_size - offset);
	if (length <= 0) {
		iod->first_dma = 0;
		goto done;
	}

	dma_len -= (page_size - offset);
	if (dma_len) {
		dma_addr += (page_size - offset);
	} else {
		sg = sg_next(sg);
		dma_addr = sg_dma_address(sg);
		dma_len = sg_dma_len(sg);
	}

	if (length <= page_size) {
		iod->first_dma = dma_addr;
		goto done;
	}

	nprps = DIV_ROUND_UP(length, page_size);
	if (nprps <= (256 / 8)) {
		pool = dev->prp_small_pool;
		iod->npages = 0;
	} else {
		pool = dev->prp_page_pool;
		iod->npages = 1;
	}

	prp_list = dma_pool_alloc(pool, GFP_ATOMIC, &prp_dma);
	if (!prp_list) {
		iod->first_dma = dma_addr;
		iod->npages = -1;
		return BLK_STS_RESOURCE;
	}
	list[0] = prp_list;
	iod->first_dma = prp_dma;
	i = 0;
	for (;;) {
		if (i == page_size >> 3) {
			__le64 *old_prp_list = prp_list;
			prp_list = dma_pool_alloc(pool, GFP_ATOMIC, &prp_dma);
			if (!prp_list)
				goto free_prps;
			list[iod->npages++] = prp_list;
			prp_list[0] = old_prp_list[i - 1];
			old_prp_list[i - 1] = cpu_to_le64(prp_dma);
			i = 1;
		}
		prp_list[i++] = cpu_to_le64(dma_addr);
		dma_len -= page_size;
		dma_addr += page_size;
		length -= page_size;
		if (length <= 0)
			break;
		if (dma_len > 0)
			continue;
		if (unlikely(dma_len < 0))
			goto bad_sgl;
		sg = sg_next(sg);
		dma_addr = sg_dma_address(sg);
		dma_len = sg_dma_len(sg);
	}
done:
	cmnd->dptr.prp1 = cpu_to_le64(sg_dma_address(iod->sg));
	cmnd->dptr.prp2 = cpu_to_le64(iod->first_dma);
	return BLK_STS_OK;
free_prps:
	nvme_free_prps(dev, req);
	return BLK_STS_RESOURCE;
bad_sgl:
	WARN(DO_ONCE(nvme_print_sgl, iod->sg, iod->nents),
			"snvme: Invalid SGL for payload:%d nents:%d\n",
			blk_rq_payload_bytes(req), iod->nents);
	return BLK_STS_IOERR;
}

static void nvme_pci_sgl_set_data(struct nvme_sgl_desc *sge,
		struct scatterlist *sg)
{
	sge->addr = cpu_to_le64(sg_dma_address(sg));
	sge->length = cpu_to_le32(sg_dma_len(sg));
	sge->type = NVME_SGL_FMT_DATA_DESC << 4;
}

static void nvme_pci_sgl_set_seg(struct nvme_sgl_desc *sge,
		dma_addr_t dma_addr, int entries)
{
	sge->addr = cpu_to_le64(dma_addr);
	if (entries < SGES_PER_PAGE) {
		sge->length = cpu_to_le32(entries * sizeof(*sge));
		sge->type = NVME_SGL_FMT_LAST_SEG_DESC << 4;
	} else {
		sge->length = cpu_to_le32(PAGE_SIZE);
		sge->type = NVME_SGL_FMT_SEG_DESC << 4;
	}
}

static blk_status_t nvme_pci_setup_sgls(struct nvme_dev *dev,
		struct request *req, struct nvme_rw_command *cmd, int entries)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct dma_pool *pool;
	struct nvme_sgl_desc *sg_list;
	struct scatterlist *sg = iod->sg;
	dma_addr_t sgl_dma;
	int i = 0;

	/* setting the transfer type as SGL */
	cmd->flags = NVME_CMD_SGL_METABUF;

	if (entries == 1) {
		nvme_pci_sgl_set_data(&cmd->dptr.sgl, sg);
		return BLK_STS_OK;
	}

	if (entries <= (256 / sizeof(struct nvme_sgl_desc))) {
		pool = dev->prp_small_pool;
		iod->npages = 0;
	} else {
		pool = dev->prp_page_pool;
		iod->npages = 1;
	}

	sg_list = dma_pool_alloc(pool, GFP_ATOMIC, &sgl_dma);
	if (!sg_list) {
		iod->npages = -1;
		return BLK_STS_RESOURCE;
	}

	nvme_pci_iod_list(req)[0] = sg_list;
	iod->first_dma = sgl_dma;

	nvme_pci_sgl_set_seg(&cmd->dptr.sgl, sgl_dma, entries);

	do {
		if (i == SGES_PER_PAGE) {
			struct nvme_sgl_desc *old_sg_desc = sg_list;
			struct nvme_sgl_desc *link = &old_sg_desc[i - 1];

			sg_list = dma_pool_alloc(pool, GFP_ATOMIC, &sgl_dma);
			if (!sg_list)
				goto free_sgls;

			i = 0;
			nvme_pci_iod_list(req)[iod->npages++] = sg_list;
			sg_list[i++] = *link;
			nvme_pci_sgl_set_seg(link, sgl_dma, entries);
		}

		nvme_pci_sgl_set_data(&sg_list[i++], sg);
		sg = sg_next(sg);
	} while (--entries > 0);

	return BLK_STS_OK;
free_sgls:
	nvme_free_sgls(dev, req);
	return BLK_STS_RESOURCE;
}

static blk_status_t nvme_setup_prp_simple(struct nvme_dev *dev,
		struct request *req, struct nvme_rw_command *cmnd,
		struct bio_vec *bv)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	unsigned int offset = bv->bv_offset & (dev->ctrl.page_size - 1);
	unsigned int first_prp_len = dev->ctrl.page_size - offset;

	iod->first_dma = dma_map_bvec(dev->dev, bv, rq_dma_dir(req), 0);
	if (dma_mapping_error(dev->dev, iod->first_dma))
		return BLK_STS_RESOURCE;
	iod->dma_len = bv->bv_len;

	cmnd->dptr.prp1 = cpu_to_le64(iod->first_dma);
	if (bv->bv_len > first_prp_len)
		cmnd->dptr.prp2 = cpu_to_le64(iod->first_dma + first_prp_len);
	return 0;
}

static blk_status_t nvme_setup_sgl_simple(struct nvme_dev *dev,
		struct request *req, struct nvme_rw_command *cmnd,
		struct bio_vec *bv)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);

	iod->first_dma = dma_map_bvec(dev->dev, bv, rq_dma_dir(req), 0);
	if (dma_mapping_error(dev->dev, iod->first_dma))
		return BLK_STS_RESOURCE;
	iod->dma_len = bv->bv_len;

	cmnd->flags = NVME_CMD_SGL_METABUF;
	cmnd->dptr.sgl.addr = cpu_to_le64(iod->first_dma);
	cmnd->dptr.sgl.length = cpu_to_le32(iod->dma_len);
	cmnd->dptr.sgl.type = NVME_SGL_FMT_DATA_DESC << 4;
	return 0;
}

static blk_status_t nvme_map_data(struct nvme_dev *dev, struct request *req,
		struct nvme_command *cmnd)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	blk_status_t ret = BLK_STS_RESOURCE;
	int nr_mapped;

	if (blk_rq_nr_phys_segments(req) == 1) {
		struct bio_vec bv = req_bvec(req);

		if (!is_pci_p2pdma_page(bv.bv_page)) {
			if (bv.bv_offset + bv.bv_len <= dev->ctrl.page_size * 2)
				return nvme_setup_prp_simple(dev, req,
							     &cmnd->rw, &bv);

			if (iod->nvmeq->qid && sgl_threshold &&
			    dev->ctrl.sgls & ((1 << 0) | (1 << 1)))
				return nvme_setup_sgl_simple(dev, req,
							     &cmnd->rw, &bv);
		}
	}

	iod->dma_len = 0;
	iod->sg = mempool_alloc(dev->iod_mempool, GFP_ATOMIC);
	if (!iod->sg)
		return BLK_STS_RESOURCE;
	sg_init_table(iod->sg, blk_rq_nr_phys_segments(req));
	iod->nents = blk_rq_map_sg(req->q, req, iod->sg);
	if (!iod->nents)
		goto out_free_sg;

	if (is_pci_p2pdma_page(sg_page(iod->sg)))
		nr_mapped = pci_p2pdma_map_sg_attrs(dev->dev, iod->sg,
				iod->nents, rq_dma_dir(req), DMA_ATTR_NO_WARN);
	else
		nr_mapped = dma_map_sg_attrs(dev->dev, iod->sg, iod->nents,
					     rq_dma_dir(req), DMA_ATTR_NO_WARN);
	if (!nr_mapped)
		goto out_free_sg;

	iod->use_sgl = nvme_pci_use_sgls(dev, req);
	if (iod->use_sgl)
		ret = nvme_pci_setup_sgls(dev, req, &cmnd->rw, nr_mapped);
	else
		ret = nvme_pci_setup_prps(dev, req, &cmnd->rw);
	if (ret != BLK_STS_OK)
		goto out_unmap_sg;
	return BLK_STS_OK;

out_unmap_sg:
	nvme_unmap_sg(dev, req);
out_free_sg:
	mempool_free(iod->sg, dev->iod_mempool);
	return ret;
}

static blk_status_t nvme_map_metadata(struct nvme_dev *dev, struct request *req,
		struct nvme_command *cmnd)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);

	iod->meta_dma = dma_map_bvec(dev->dev, rq_integrity_vec(req),
			rq_dma_dir(req), 0);
	if (dma_mapping_error(dev->dev, iod->meta_dma))
		return BLK_STS_IOERR;
	cmnd->rw.metadata = cpu_to_le64(iod->meta_dma);
	return 0;
}

/*
 * NOTE: ns is NULL when called on the admin queue.
 */
static blk_status_t nvme_queue_rq(struct blk_mq_hw_ctx *hctx,
			 const struct blk_mq_queue_data *bd)
{
	struct nvme_ns *ns = hctx->queue->queuedata;
	struct nvme_queue *nvmeq = hctx->driver_data;
	struct nvme_dev *dev = nvmeq->dev;
	struct request *req = bd->rq;
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct nvme_command cmnd;
	struct nvme_request *rq;
	blk_status_t ret;

	iod->aborted = 0;
	iod->npages = -1;
	iod->nents = 0;

	/*
	 * We should not need to do this, but we're still using this to
	 * ensure we can drain requests on a dying queue.
	 */
	if (unlikely(!test_bit(NVMEQ_ENABLED, &nvmeq->flags)))
		return BLK_STS_IOERR;

	ret = snvme_setup_cmd(ns, req, &cmnd);
	if (ret)
		return ret;

	rq = blk_mq_rq_to_pdu(req);
	rq->opcode = cmnd.common.opcode;
	
	if (blk_rq_nr_phys_segments(req)) {
		ret = nvme_map_data(dev, req, &cmnd);
		if (ret)
			goto out_free_cmd;
	}

	if (blk_integrity_rq(req)) {
		ret = nvme_map_metadata(dev, req, &cmnd);
		if (ret)
			goto out_unmap_data;
	}

	blk_mq_start_request(req);
	nvme_submit_cmd(nvmeq, &cmnd, bd->last);
	return BLK_STS_OK;
out_unmap_data:
	nvme_unmap_data(dev, req);
out_free_cmd:
	snvme_cleanup_cmd(req);
	return ret;
}

static void nvme_pci_complete_rq(struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct nvme_dev *dev = iod->nvmeq->dev;

	snvme_cleanup_cmd(req);
	if (blk_integrity_rq(req))
		dma_unmap_page(dev->dev, iod->meta_dma,
			       rq_integrity_vec(req)->bv_len, rq_data_dir(req));
	if (blk_rq_nr_phys_segments(req))
		nvme_unmap_data(dev, req);
	snvme_complete_rq(req);
}

/* We read the CQE phase first to check if the rest of the entry is valid */
static inline bool nvme_cqe_pending(struct nvme_queue *nvmeq)
{
	return (le16_to_cpu(nvmeq->cqes[nvmeq->cq_head].status) & 1) ==
			nvmeq->cq_phase;
}

static inline void nvme_ring_cq_doorbell(struct nvme_queue *nvmeq)
{
	u16 head = nvmeq->cq_head;

	if (nvme_dbbuf_update_and_check_event(head, nvmeq->dbbuf_cq_db,
					      nvmeq->dbbuf_cq_ei))
		writel(head, nvmeq->q_db + nvmeq->dev->db_stride);
}

static inline void nvme_eh_io_timeout(struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct nvme_ctrl *ctrl = &iod->nvmeq->dev->ctrl;
	struct nvme_request *rq = nvme_req(req);

	/* admin command error */
	if (req->q == ctrl->admin_q || req->q == ctrl->connect_q) {
		dev_warn_ratelimited(ctrl->device,
			 "Admin command timeout, CMD: %d(+%d)",
			 rq->opcode, rq->retries);
		return;
	}

	/* io command timeout */
	if (rq->opcode == nvme_cmd_write || rq->opcode == nvme_cmd_read)
		dev_warn_ratelimited(ctrl->device,
			 "I/O timeout, QID: %d, CMD: %d(+%d), Sector: %llu+%u",
			 iod->nvmeq->qid, rq->opcode, rq->retries,
			 (unsigned long long)blk_rq_pos(req),
			 blk_rq_sectors(req));
	else
		dev_warn_ratelimited(ctrl->device,
			 "I/O timeout, QID: %d, CMD: %d(+%d)",
			 iod->nvmeq->qid, rq->opcode, rq->retries);
}

static inline void nvme_eh_io_error(struct request *req, __le16 status)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct nvme_ctrl *ctrl = &iod->nvmeq->dev->ctrl;
	struct nvme_request *rq = nvme_req(req);

	if (status == NVME_SC_SUCCESS || rq->opcode == nvme_admin_abort_cmd)
		return;

	/* admin command error */
	if (req->q == ctrl->admin_q || req->q == ctrl->connect_q) {
		dev_warn_ratelimited(ctrl->device,
			 "Admin command error, CMD: %d(+%d), Status: 0x%x",
			 rq->opcode, rq->retries, status);
		return;
	}

	/* io command error */
	if (rq->opcode == nvme_cmd_write || rq->opcode == nvme_cmd_read)
		dev_warn_ratelimited(ctrl->device,
			 "I/O error, QID: %d, CMD: %d(+%d), Sector: %llu+%u, Status: 0x%x",
			 iod->nvmeq->qid, rq->opcode, rq->retries,
			 (unsigned long long)blk_rq_pos(req),
			 blk_rq_sectors(req), status);
	else
		dev_warn_ratelimited(ctrl->device,
			 "I/O error, QID: %d, CMD: %d(+%d), Status: 0x%x",
			 iod->nvmeq->qid, rq->opcode, rq->retries,
			 status);
}

static inline struct blk_mq_tags *nvme_queue_tagset(struct nvme_queue *nvmeq)
{
	if (!nvmeq->qid)
		return nvmeq->dev->admin_tagset.tags[0];
	return nvmeq->dev->tagset.tags[nvmeq->qid - 1];
}

static inline void nvme_handle_cqe(struct nvme_queue *nvmeq, u16 idx)
{
	volatile struct nvme_completion *cqe = &nvmeq->cqes[idx];
	struct request *req;

	/*
	 * AEN requests are special as they don't time out and can
	 * survive any kind of queue freeze and often don't respond to
	 * aborts.  We don't even bother to allocate a struct request
	 * for them but rather special case them here.
	 */
	if (unlikely(nvmeq->qid == 0 &&
			cqe->command_id >= NVME_AQ_BLK_MQ_DEPTH)) {
		snvme_complete_async_event(&nvmeq->dev->ctrl,
				cqe->status, &cqe->result);
		return;
	}

	req = blk_mq_tag_to_rq(nvme_queue_tagset(nvmeq), cqe->command_id);
	if (unlikely(!req)) {
		dev_warn(nvmeq->dev->ctrl.device,
			"invalid id %d completed on queue %d\n",
			cqe->command_id, le16_to_cpu(cqe->sq_id));
		return;
	}

	nvme_eh_io_error(req, le16_to_cpu(cqe->status) >> 1);
	/* trace_nvme_sq(req, cqe->sq_head, nvmeq->sq_tail): tracepoint dropped */
	nvme_end_request(req, cqe->status, cqe->result);
}

static void nvme_complete_cqes(struct nvme_queue *nvmeq, u16 start, u16 end)
{
	while (start != end) {
		nvme_handle_cqe(nvmeq, start);
		if (++start == nvmeq->q_depth)
			start = 0;
	}
}

static inline void nvme_update_cq_head(struct nvme_queue *nvmeq)
{
	if (nvmeq->cq_head == nvmeq->q_depth - 1) {
		nvmeq->cq_head = 0;
		nvmeq->cq_phase = !nvmeq->cq_phase;
	} else {
		nvmeq->cq_head++;
	}
}

static inline int nvme_process_cq(struct nvme_queue *nvmeq, u16 *start,
				  u16 *end, unsigned int tag)
{
	int found = 0;

	*start = nvmeq->cq_head;
	while (nvme_cqe_pending(nvmeq)) {
		if (tag == -1U || nvmeq->cqes[nvmeq->cq_head].command_id == tag)
			found++;
		nvme_update_cq_head(nvmeq);
	}
	*end = nvmeq->cq_head;

	if (*start != *end)
		nvme_ring_cq_doorbell(nvmeq);
	return found;
}

static irqreturn_t nvme_irq(int irq, void *data)
{
	struct nvme_queue *nvmeq = data;
	/* Round 16 S4: default IRQ_HANDLED instead of IRQ_NONE.  When GPU
	 * P2P polls user-queue CQs, the CQE is consumed before the kernel
	 * handler runs, so nvme_process_cq finds nothing.  Returning IRQ_NONE
	 * in that case triggers "nobody cared" → IRQ disable after ~100k
	 * spurious hits.  Returning IRQ_HANDLED silences the storm without
	 * skipping real CQE processing (nvme_process_cq still runs above). */
	irqreturn_t ret = IRQ_HANDLED;
	u16 start, end;

	/*
	 * The rmb/wmb pair ensures we see all updates from a previous run of
	 * the irq handler, even if that was on another CPU.
	 */
	rmb();
	if (nvmeq->cq_head != nvmeq->last_cq_head)
		ret = IRQ_HANDLED;
	nvme_process_cq(nvmeq, &start, &end, -1);
	nvmeq->last_cq_head = nvmeq->cq_head;
	wmb();

	if (start != end) {
		nvme_complete_cqes(nvmeq, start, end);
		return IRQ_HANDLED;
	}

	return ret;
}

static irqreturn_t nvme_irq_check(int irq, void *data)
{
	struct nvme_queue *nvmeq = data;
	if (nvme_cqe_pending(nvmeq))
		return IRQ_WAKE_THREAD;
	return IRQ_NONE;
}

/*
 * Poll for completions any queue, including those not dedicated to polling.
 * Can be called from any context.
 */
static int nvme_poll_irqdisable(struct nvme_queue *nvmeq, unsigned int tag)
{
	struct pci_dev *pdev = to_pci_dev(nvmeq->dev->dev);
	u16 start, end;
	int found;

	/*
	 * For a poll queue we need to protect against the polling thread
	 * using the CQ lock.  For normal interrupt driven threads we have
	 * to disable the interrupt to avoid racing with it.
	 */
	if (test_bit(NVMEQ_POLLED, &nvmeq->flags)) {
		spin_lock(&nvmeq->cq_poll_lock);
		found = nvme_process_cq(nvmeq, &start, &end, tag);
		spin_unlock(&nvmeq->cq_poll_lock);
	} else {
		disable_irq(pci_irq_vector(pdev, nvmeq->cq_vector));
		found = nvme_process_cq(nvmeq, &start, &end, tag);
		enable_irq(pci_irq_vector(pdev, nvmeq->cq_vector));
	}

	nvme_complete_cqes(nvmeq, start, end);
	return found;
}

static int nvme_poll(struct blk_mq_hw_ctx *hctx)
{
	struct nvme_queue *nvmeq = hctx->driver_data;
	u16 start, end;
	bool found;

	if (!nvme_cqe_pending(nvmeq))
		return 0;

	spin_lock(&nvmeq->cq_poll_lock);
	found = nvme_process_cq(nvmeq, &start, &end, -1);
	nvme_complete_cqes(nvmeq, start, end);
	spin_unlock(&nvmeq->cq_poll_lock);

	return found;
}

static void nvme_pci_submit_async_event(struct nvme_ctrl *ctrl)
{
	struct nvme_dev *dev = to_nvme_dev(ctrl);
	struct nvme_queue *nvmeq = &dev->queues[0];
	struct nvme_command c;

	memset(&c, 0, sizeof(c));
	c.common.opcode = nvme_admin_async_event;
	c.common.command_id = NVME_AQ_BLK_MQ_DEPTH;
	nvme_submit_cmd(nvmeq, &c, true);
}

static int adapter_delete_queue(struct nvme_dev *dev, u8 opcode, u16 id)
{
	struct nvme_command c;

	memset(&c, 0, sizeof(c));
	c.delete_queue.opcode = opcode;
	c.delete_queue.qid = cpu_to_le16(id);

	return snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
}

static int adapter_alloc_cq(struct nvme_dev *dev, u16 qid,
		struct nvme_queue *nvmeq, s16 vector)
{
	struct nvme_command c;
	int flags = NVME_QUEUE_PHYS_CONTIG;

	if (!test_bit(NVMEQ_POLLED, &nvmeq->flags))
		flags |= NVME_CQ_IRQ_ENABLED;

	/*
	 * Note: we (ab)use the fact that the prp fields survive if no data
	 * is attached to the request.
	 */
	memset(&c, 0, sizeof(c));
	c.create_cq.opcode = nvme_admin_create_cq;
	c.create_cq.prp1 = cpu_to_le64(nvmeq->cq_dma_addr);
	c.create_cq.cqid = cpu_to_le16(qid);
	c.create_cq.qsize = cpu_to_le16(nvmeq->q_depth - 1);
	c.create_cq.cq_flags = cpu_to_le16(flags);
	c.create_cq.irq_vector = cpu_to_le16(vector);

	return snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
}

static int adapter_alloc_sq(struct nvme_dev *dev, u16 qid,
						struct nvme_queue *nvmeq)
{
	struct nvme_ctrl *ctrl = &dev->ctrl;
	struct nvme_command c;
	int flags = NVME_QUEUE_PHYS_CONTIG;

	/*
	 * Some drives have a bug that auto-enables WRRU if MEDIUM isn't
	 * set. Since URGENT priority is zeroes, it makes all queues
	 * URGENT.
	 */
	if (ctrl->quirks & NVME_QUIRK_MEDIUM_PRIO_SQ)
		flags |= NVME_SQ_PRIO_MEDIUM;

	/*
	 * Note: we (ab)use the fact that the prp fields survive if no data
	 * is attached to the request.
	 */
	memset(&c, 0, sizeof(c));
	c.create_sq.opcode = nvme_admin_create_sq;
	c.create_sq.prp1 = cpu_to_le64(nvmeq->sq_dma_addr);
	c.create_sq.sqid = cpu_to_le16(qid);
	c.create_sq.qsize = cpu_to_le16(nvmeq->q_depth - 1);
	c.create_sq.sq_flags = cpu_to_le16(flags);
	c.create_sq.cqid = cpu_to_le16(qid);

	return snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
}

/*
 * adapter_alloc_{cq,sq}_user: snvme variants of adapter_alloc_{cq,sq}
 * for user-pinned IO queues (segment 6c).
 *
 * Differences vs upstream:
 *   - DMA address comes from the user-pinned page list (q_map->addrs[0])
 *     instead of nvmeq->{cq,sq}_dma_addr.
 *   - cq_flags omits NVME_CQ_IRQ_ENABLED -- user queues are polled from
 *     userspace / GPU; the kernel deliberately does not request an
 *     MSI-X vector for them.
 *   - irq_vector is therefore always 0 (the field is unused but the spec
 *     requires writing something).
 *   - q_depth comes from dev->q_depth (the controller-wide default)
 *     rather than nvmeq->q_depth -- there is no struct nvme_queue
 *     record for user queues.
 *
 * 5.4-vs-5.15 compatibility note: every API touched here
 * (struct nvme_command, snvme_submit_sync_cmd, the nvme_admin_create_*
 * opcodes, NVME_QUEUE_PHYS_CONTIG) is unchanged between baselines, so
 * the body is byte-for-byte equivalent to the snvme-5.15.0 originals.
 */
static int adapter_alloc_cq_user(struct nvme_dev *dev,
				 struct map *q_map, int qid)
{
	struct nvme_command c;
	int flags = NVME_QUEUE_PHYS_CONTIG;

	memset(&c, 0, sizeof(c));
	c.create_cq.opcode    = nvme_admin_create_cq;
	c.create_cq.prp1      = cpu_to_le64(q_map->addrs[0]);
	c.create_cq.cqid      = cpu_to_le16(qid);
	c.create_cq.qsize     = cpu_to_le16(dev->q_depth - 1);
	c.create_cq.cq_flags  = cpu_to_le16(flags);
	c.create_cq.irq_vector = cpu_to_le16(0);

	return snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
}

static int adapter_alloc_sq_user(struct nvme_dev *dev,
				 struct map *q_map, int qid)
{
	struct nvme_command c;
	int flags = NVME_QUEUE_PHYS_CONTIG;

	memset(&c, 0, sizeof(c));
	c.create_sq.opcode   = nvme_admin_create_sq;
	c.create_sq.prp1     = cpu_to_le64(q_map->addrs[0]);
	c.create_sq.sqid     = cpu_to_le16(qid);
	c.create_sq.qsize    = cpu_to_le16(dev->q_depth - 1);
	c.create_sq.sq_flags = cpu_to_le16(flags);
	c.create_sq.cqid     = cpu_to_le16(qid);

	return snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
}

static int adapter_delete_cq(struct nvme_dev *dev, u16 cqid)
{
	return adapter_delete_queue(dev, nvme_admin_delete_cq, cqid);
}

static int adapter_delete_sq(struct nvme_dev *dev, u16 sqid)
{
	return adapter_delete_queue(dev, nvme_admin_delete_sq, sqid);
}

static void abort_endio(struct request *req, blk_status_t error)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct nvme_queue *nvmeq = iod->nvmeq;

	dev_warn(nvmeq->dev->ctrl.device,
		 "Abort status: 0x%x", nvme_req(req)->status);
	atomic_inc(&nvmeq->dev->ctrl.abort_limit);
	blk_mq_free_request(req);
}

static bool nvme_should_reset(struct nvme_dev *dev, u32 csts)
{

	/* If true, indicates loss of adapter communication, possibly by a
	 * NVMe Subsystem reset.
	 */
	bool nssro = dev->subsystem && (csts & NVME_CSTS_NSSRO);

	/* If there is a reset/reinit ongoing, we shouldn't reset again. */
	switch (dev->ctrl.state) {
	case NVME_CTRL_RESETTING:
	case NVME_CTRL_CONNECTING:
		return false;
	default:
		break;
	}

	/* We shouldn't reset unless the controller is on fatal error state
	 * _or_ if we lost the communication with it.
	 */
	if (!(csts & NVME_CSTS_CFS) && !nssro)
		return false;

	return true;
}

static void nvme_warn_reset(struct nvme_dev *dev, u32 csts)
{
	/* Read a config register to help see what died. */
	u16 pci_status;
	int result;

	result = pci_read_config_word(to_pci_dev(dev->dev), PCI_STATUS,
				      &pci_status);
	if (result == PCIBIOS_SUCCESSFUL)
		dev_warn(dev->ctrl.device,
			 "controller is down; will reset: CSTS=0x%x, PCI_STATUS=0x%hx\n",
			 csts, pci_status);
	else
		dev_warn(dev->ctrl.device,
			 "controller is down; will reset: CSTS=0x%x, PCI_STATUS read failed (%d)\n",
			 csts, result);
}

static enum blk_eh_timer_return nvme_timeout(struct request *req, bool reserved)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct nvme_queue *nvmeq = iod->nvmeq;
	struct nvme_dev *dev = nvmeq->dev;
	struct request *abort_req;
	struct nvme_command cmd;
	u32 csts = readl(dev->bar + NVME_REG_CSTS);

	/* log error memset */
	nvme_eh_io_timeout(req);

	/* If PCI error recovery process is happening, we cannot reset or
	 * the recovery mechanism will surely fail.
	 */
	mb();
	if (pci_channel_offline(to_pci_dev(dev->dev)))
		return BLK_EH_RESET_TIMER;

	/*
	 * Reset immediately if the controller is failed
	 */
	if (nvme_should_reset(dev, csts)) {
		nvme_warn_reset(dev, csts);
		nvme_dev_disable(dev, false);
		snvme_reset_ctrl(&dev->ctrl);
		return BLK_EH_DONE;
	}

	/*
	 * Did we miss an interrupt?
	 */
	if (nvme_poll_irqdisable(nvmeq, req->tag)) {
		dev_warn(dev->ctrl.device,
			 "I/O %d QID %d timeout, completion polled\n",
			 req->tag, nvmeq->qid);
		return BLK_EH_DONE;
	}

	/*
	 * Shutdown immediately if controller times out while starting. The
	 * reset work will see the pci device disabled when it gets the forced
	 * cancellation error. All outstanding requests are completed on
	 * shutdown, so we return BLK_EH_DONE.
	 */
	switch (dev->ctrl.state) {
	case NVME_CTRL_CONNECTING:
		snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_DELETING);
		/* fall through */
	case NVME_CTRL_DELETING:
		dev_warn_ratelimited(dev->ctrl.device,
			 "I/O %d QID %d timeout, disable controller\n",
			 req->tag, nvmeq->qid);
		nvme_req(req)->flags |= NVME_REQ_CANCELLED;
		nvme_dev_disable(dev, true);
		return BLK_EH_DONE;
	case NVME_CTRL_RESETTING:
		return BLK_EH_RESET_TIMER;
	default:
		break;
	}

	/*
 	 * Shutdown the controller immediately and schedule a reset if the
 	 * command was already aborted once before and still hasn't been
 	 * returned to the driver, or if this is the admin queue.
	 */
	if (!nvmeq->qid || iod->aborted) {
		dev_warn(dev->ctrl.device,
			 "I/O %d QID %d timeout, reset controller\n",
			 req->tag, nvmeq->qid);
		nvme_req(req)->flags |= NVME_REQ_CANCELLED;
		nvme_dev_disable(dev, false);
		snvme_reset_ctrl(&dev->ctrl);

		return BLK_EH_DONE;
	}

	if (atomic_dec_return(&dev->ctrl.abort_limit) < 0) {
		atomic_inc(&dev->ctrl.abort_limit);
		return BLK_EH_RESET_TIMER;
	}
	iod->aborted = 1;

	memset(&cmd, 0, sizeof(cmd));
	cmd.abort.opcode = nvme_admin_abort_cmd;
	cmd.abort.cid = req->tag;
	cmd.abort.sqid = cpu_to_le16(nvmeq->qid);

	dev_warn(nvmeq->dev->ctrl.device,
		"I/O %d QID %d timeout, aborting\n",
		 req->tag, nvmeq->qid);

	abort_req = snvme_alloc_request(dev->ctrl.admin_q, &cmd,
			BLK_MQ_REQ_NOWAIT, NVME_QID_ANY);
	if (IS_ERR(abort_req)) {
		atomic_inc(&dev->ctrl.abort_limit);
		return BLK_EH_RESET_TIMER;
	}

	abort_req->timeout = ADMIN_TIMEOUT;
	abort_req->end_io_data = NULL;
	blk_execute_rq_nowait(abort_req->q, NULL, abort_req, 0, abort_endio);

	/*
	 * The aborted req will be completed on receiving the abort req.
	 * We enable the timer again. If hit twice, it'll cause a device reset,
	 * as the device then is in a faulty state.
	 */
	return BLK_EH_RESET_TIMER;
}

static void nvme_free_queue(struct nvme_queue *nvmeq)
{
	dma_free_coherent(nvmeq->dev->dev, CQ_SIZE(nvmeq),
				(void *)nvmeq->cqes, nvmeq->cq_dma_addr);
	if (!nvmeq->sq_cmds)
		return;

	if (test_and_clear_bit(NVMEQ_SQ_CMB, &nvmeq->flags)) {
		pci_free_p2pmem(to_pci_dev(nvmeq->dev->dev),
				nvmeq->sq_cmds, SQ_SIZE(nvmeq));
	} else {
		dma_free_coherent(nvmeq->dev->dev, SQ_SIZE(nvmeq),
				nvmeq->sq_cmds, nvmeq->sq_dma_addr);
	}
}

static void nvme_free_queues(struct nvme_dev *dev, int lowest)
{
	int i;

	for (i = dev->ctrl.queue_count - 1; i >= lowest; i--) {
		dev->ctrl.queue_count--;
		nvme_free_queue(&dev->queues[i]);
	}
}

/**
 * nvme_suspend_queue - put queue into suspended state
 * @nvmeq: queue to suspend
 */
static int nvme_suspend_queue(struct nvme_queue *nvmeq)
{
	if (!test_and_clear_bit(NVMEQ_ENABLED, &nvmeq->flags))
		return 1;

	/* ensure that nvme_queue_rq() sees NVMEQ_ENABLED cleared */
	mb();

	nvmeq->dev->online_queues--;
	if (!nvmeq->qid && nvmeq->dev->ctrl.admin_q)
		blk_mq_quiesce_queue(nvmeq->dev->ctrl.admin_q);
	if (!test_and_clear_bit(NVMEQ_POLLED, &nvmeq->flags))
		pci_free_irq(to_pci_dev(nvmeq->dev->dev), nvmeq->cq_vector, nvmeq);
	return 0;
}

static void nvme_suspend_io_queues(struct nvme_dev *dev)
{
	int i;

	for (i = dev->ctrl.queue_count - 1; i > 0; i--)
		nvme_suspend_queue(&dev->queues[i]);
}

static void nvme_disable_admin_queue(struct nvme_dev *dev, bool shutdown)
{
	struct nvme_queue *nvmeq = &dev->queues[0];

	if (shutdown)
		snvme_shutdown_ctrl(&dev->ctrl);
	else
		snvme_disable_ctrl(&dev->ctrl);

	nvme_poll_irqdisable(nvmeq, -1);
}

/*
 * Called only on a device that has been disabled and after all other threads
 * that can check this device's completion queues have synced. This is the
 * last chance for the driver to see a natural completion before
 * snvme_cancel_request() terminates all incomplete requests.
 */
static void nvme_reap_pending_cqes(struct nvme_dev *dev)
{
	u16 start, end;
	int i;

	for (i = dev->ctrl.queue_count - 1; i > 0; i--) {
		nvme_process_cq(&dev->queues[i], &start, &end, -1);
		nvme_complete_cqes(&dev->queues[i], start, end);
	}
}

static int nvme_cmb_qdepth(struct nvme_dev *dev, int nr_io_queues,
				int entry_size)
{
	int q_depth = dev->q_depth;
	unsigned q_size_aligned = roundup(q_depth * entry_size,
					  dev->ctrl.page_size);

	if (q_size_aligned * nr_io_queues > dev->cmb_size) {
		u64 mem_per_q = div_u64(dev->cmb_size, nr_io_queues);
		mem_per_q = round_down(mem_per_q, dev->ctrl.page_size);
		q_depth = div_u64(mem_per_q, entry_size);

		/*
		 * Ensure the reduced q_depth is above some threshold where it
		 * would be better to map queues in system memory with the
		 * original depth
		 */
		if (q_depth < 64)
			return -ENOMEM;
	}

	return q_depth;
}

static int nvme_alloc_sq_cmds(struct nvme_dev *dev, struct nvme_queue *nvmeq,
				int qid)
{
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	if (qid && dev->cmb_use_sqes && (dev->cmbsz & NVME_CMBSZ_SQS)) {
		nvmeq->sq_cmds = pci_alloc_p2pmem(pdev, SQ_SIZE(nvmeq));
		if (nvmeq->sq_cmds) {
			nvmeq->sq_dma_addr = pci_p2pmem_virt_to_bus(pdev,
							nvmeq->sq_cmds);
			if (nvmeq->sq_dma_addr) {
				set_bit(NVMEQ_SQ_CMB, &nvmeq->flags);
				return 0;
			}

			pci_free_p2pmem(pdev, nvmeq->sq_cmds, SQ_SIZE(nvmeq));
		}
	}

	nvmeq->sq_cmds = dma_alloc_coherent(dev->dev, SQ_SIZE(nvmeq),
				&nvmeq->sq_dma_addr, GFP_KERNEL);
	if (!nvmeq->sq_cmds)
		return -ENOMEM;
	return 0;
}

static int nvme_alloc_queue(struct nvme_dev *dev, int qid, int depth)
{
	struct nvme_queue *nvmeq = &dev->queues[qid];

	if (dev->ctrl.queue_count > qid)
		return 0;

	nvmeq->sqes = qid ? dev->io_sqes : NVME_ADM_SQES;
	nvmeq->q_depth = depth;
	nvmeq->cqes = dma_alloc_coherent(dev->dev, CQ_SIZE(nvmeq),
					 &nvmeq->cq_dma_addr, GFP_KERNEL);
	if (!nvmeq->cqes)
		goto free_nvmeq;

	if (nvme_alloc_sq_cmds(dev, nvmeq, qid))
		goto free_cqdma;

	nvmeq->dev = dev;
	spin_lock_init(&nvmeq->sq_lock);
	spin_lock_init(&nvmeq->cq_poll_lock);
	nvmeq->cq_head = 0;
	nvmeq->cq_phase = 1;
	nvmeq->q_db = &dev->dbs[qid * 2 * dev->db_stride];
	nvmeq->qid = qid;
	dev->ctrl.queue_count++;

	return 0;

 free_cqdma:
	dma_free_coherent(dev->dev, CQ_SIZE(nvmeq), (void *)nvmeq->cqes,
			  nvmeq->cq_dma_addr);
 free_nvmeq:
	return -ENOMEM;
}

static int queue_request_irq(struct nvme_queue *nvmeq)
{
	struct pci_dev *pdev = to_pci_dev(nvmeq->dev->dev);
	int nr = nvmeq->dev->ctrl.instance;

	/*
	 * PORTING.md section 2 / section 7.4 co-existence rule: IRQ description string
	 * MUST be "snvme%dq%d", not the upstream "nvme%dq%d", otherwise
	 * /proc/interrupts shows duplicate "nvme0q1" lines for the
	 * in-tree nvme.ko and snvme.ko bound NVMes and the two drivers
	 * become indistinguishable in debugging.  snvme-rename.sed only
	 * processes identifiers and does NOT touch this literal.
	 */
	if (use_threaded_interrupts) {
		return pci_request_irq(pdev, nvmeq->cq_vector, nvme_irq_check,
				nvme_irq, nvmeq, "snvme%dq%d", nr, nvmeq->qid);
	} else {
		return pci_request_irq(pdev, nvmeq->cq_vector, nvme_irq,
				NULL, nvmeq, "snvme%dq%d", nr, nvmeq->qid);
	}
}

static void nvme_init_queue(struct nvme_queue *nvmeq, u16 qid)
{
	struct nvme_dev *dev = nvmeq->dev;

	nvmeq->sq_tail = 0;
	nvmeq->last_sq_tail = 0;
	nvmeq->cq_head = 0;
	nvmeq->cq_phase = 1;
	nvmeq->q_db = &dev->dbs[qid * 2 * dev->db_stride];
	memset((void *)nvmeq->cqes, 0, CQ_SIZE(nvmeq));
	nvme_dbbuf_init(dev, nvmeq, qid);
	dev->online_queues++;
	wmb(); /* ensure the first interrupt sees the initialization */
}

static int nvme_create_queue(struct nvme_queue *nvmeq, int qid, bool polled)
{
	struct nvme_dev *dev = nvmeq->dev;
	int result;
	u16 vector = 0;

	clear_bit(NVMEQ_DELETE_ERROR, &nvmeq->flags);

	/*
	 * A queue's vector matches the queue identifier unless the controller
	 * has only one vector available.
	 */
	if (!polled)
		vector = dev->num_vecs == 1 ? 0 : qid;
	else
		set_bit(NVMEQ_POLLED, &nvmeq->flags);

	result = adapter_alloc_cq(dev, qid, nvmeq, vector);
	if (result)
		return result;

	result = adapter_alloc_sq(dev, qid, nvmeq);
	if (result < 0)
		return result;
	else if (result)
		goto release_cq;

	nvmeq->cq_vector = vector;
	nvme_init_queue(nvmeq, qid);

	if (!polled) {
		result = queue_request_irq(nvmeq);
		if (result < 0)
			goto release_sq;
	}

	set_bit(NVMEQ_ENABLED, &nvmeq->flags);
	return result;

release_sq:
	dev->online_queues--;
	adapter_delete_sq(dev, qid);
release_cq:
	adapter_delete_cq(dev, qid);
	return result;
}

static const struct blk_mq_ops nvme_mq_admin_ops = {
	.queue_rq	= nvme_queue_rq,
	.complete	= nvme_pci_complete_rq,
	.init_hctx	= nvme_admin_init_hctx,
	.init_request	= nvme_init_request,
	.timeout	= nvme_timeout,
};

static const struct blk_mq_ops nvme_mq_ops = {
	.queue_rq	= nvme_queue_rq,
	.complete	= nvme_pci_complete_rq,
	.commit_rqs	= nvme_commit_rqs,
	.init_hctx	= nvme_init_hctx,
	.init_request	= nvme_init_request,
	.map_queues	= nvme_pci_map_queues,
	.timeout	= nvme_timeout,
	.poll		= nvme_poll,
};

static void nvme_dev_remove_admin(struct nvme_dev *dev)
{
	if (dev->ctrl.admin_q && !blk_queue_dying(dev->ctrl.admin_q)) {
		/*
		 * If the controller was reset during removal, it's possible
		 * user requests may be waiting on a stopped queue. Start the
		 * queue to flush these to completion.
		 */
		blk_mq_unquiesce_queue(dev->ctrl.admin_q);
		blk_cleanup_queue(dev->ctrl.admin_q);
		blk_mq_free_tag_set(&dev->admin_tagset);
	}
}

static int nvme_alloc_admin_tags(struct nvme_dev *dev)
{
	if (!dev->ctrl.admin_q) {
		dev->admin_tagset.ops = &nvme_mq_admin_ops;
		dev->admin_tagset.nr_hw_queues = 1;

		dev->admin_tagset.queue_depth = NVME_AQ_MQ_TAG_DEPTH;
		dev->admin_tagset.timeout = ADMIN_TIMEOUT;
		dev->admin_tagset.numa_node = dev->ctrl.numa_node;
		dev->admin_tagset.cmd_size = sizeof(struct nvme_iod);
		dev->admin_tagset.flags = BLK_MQ_F_NO_SCHED;
		dev->admin_tagset.driver_data = dev;

		if (blk_mq_alloc_tag_set(&dev->admin_tagset))
			return -ENOMEM;
		dev->ctrl.admin_tagset = &dev->admin_tagset;

		dev->ctrl.admin_q = blk_mq_init_queue(&dev->admin_tagset);
		if (IS_ERR(dev->ctrl.admin_q)) {
			blk_mq_free_tag_set(&dev->admin_tagset);
			dev->ctrl.admin_q = NULL;
			return -ENOMEM;
		}
		if (!blk_get_queue(dev->ctrl.admin_q)) {
			nvme_dev_remove_admin(dev);
			dev->ctrl.admin_q = NULL;
			return -ENODEV;
		}
	} else
		blk_mq_unquiesce_queue(dev->ctrl.admin_q);

	return 0;
}

static unsigned long db_bar_size(struct nvme_dev *dev, unsigned nr_io_queues)
{
	return NVME_REG_DBS + ((nr_io_queues + 1) * 8 * dev->db_stride);
}

static int nvme_remap_bar(struct nvme_dev *dev, unsigned long size)
{
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	if (size <= dev->bar_mapped_size)
		return 0;
	if (size > pci_resource_len(pdev, 0))
		return -ENOMEM;
	if (dev->bar)
		iounmap(dev->bar);
	dev->bar = ioremap(pci_resource_start(pdev, 0), size);
	if (!dev->bar) {
		dev->bar_mapped_size = 0;
		return -ENOMEM;
	}
	dev->bar_mapped_size = size;
	dev->dbs = dev->bar + NVME_REG_DBS;

	return 0;
}

static int nvme_pci_configure_admin_queue(struct nvme_dev *dev)
{
	int result;
	u32 aqa;
	struct nvme_queue *nvmeq;

	result = nvme_remap_bar(dev, db_bar_size(dev, 0));
	if (result < 0)
		return result;

	dev->subsystem = readl(dev->bar + NVME_REG_VS) >= NVME_VS(1, 1, 0) ?
				NVME_CAP_NSSRC(dev->ctrl.cap) : 0;

	if (dev->subsystem &&
	    (readl(dev->bar + NVME_REG_CSTS) & NVME_CSTS_NSSRO))
		writel(NVME_CSTS_NSSRO, dev->bar + NVME_REG_CSTS);

	result = snvme_disable_ctrl(&dev->ctrl);
	if (result < 0)
		return result;

	result = nvme_alloc_queue(dev, 0, NVME_AQ_DEPTH);
	if (result)
		return result;

	dev->ctrl.numa_node = dev_to_node(dev->dev);

	nvmeq = &dev->queues[0];
	aqa = nvmeq->q_depth - 1;
	aqa |= aqa << 16;

	writel(aqa, dev->bar + NVME_REG_AQA);
	lo_hi_writeq(nvmeq->sq_dma_addr, dev->bar + NVME_REG_ASQ);
	lo_hi_writeq(nvmeq->cq_dma_addr, dev->bar + NVME_REG_ACQ);

	result = snvme_enable_ctrl(&dev->ctrl);
	if (result)
		return result;

	nvmeq->cq_vector = 0;
	nvme_init_queue(nvmeq, 0);
	result = queue_request_irq(nvmeq);
	if (result) {
		dev->online_queues--;
		return result;
	}

	set_bit(NVMEQ_ENABLED, &nvmeq->flags);
	return result;
}

static int nvme_create_io_queues(struct nvme_dev *dev)
{
	unsigned i, max, rw_queues;
	int ret = 0;

	for (i = dev->ctrl.queue_count; i <= dev->max_qid; i++) {
		if (nvme_alloc_queue(dev, i, dev->q_depth)) {
			ret = -ENOMEM;
			break;
		}
	}

	max = min(dev->max_qid, dev->ctrl.queue_count - 1);
	if (max != 1 && dev->io_queues[HCTX_TYPE_POLL]) {
		rw_queues = dev->io_queues[HCTX_TYPE_DEFAULT] +
				dev->io_queues[HCTX_TYPE_READ];
	} else {
		rw_queues = max;
	}

	for (i = dev->online_queues; i <= max; i++) {
		bool polled = i > rw_queues;

		ret = nvme_create_queue(&dev->queues[i], i, polled);
		if (ret)
			break;
	}

	/*
	 * Ignore failing Create SQ/CQ commands, we can continue with less
	 * than the desired amount of queues, and even a controller without
	 * I/O queues can still be used to issue admin commands.  This might
	 * be useful to upgrade a buggy firmware for example.
	 */
	return ret >= 0 ? 0 : ret;
}

static ssize_t nvme_cmb_show(struct device *dev,
			     struct device_attribute *attr,
			     char *buf)
{
	struct nvme_dev *ndev = to_nvme_dev(dev_get_drvdata(dev));

	return scnprintf(buf, PAGE_SIZE, "cmbloc : x%08x\ncmbsz  : x%08x\n",
		       ndev->cmbloc, ndev->cmbsz);
}
static DEVICE_ATTR(cmb, S_IRUGO, nvme_cmb_show, NULL);

static u64 nvme_cmb_size_unit(struct nvme_dev *dev)
{
	u8 szu = (dev->cmbsz >> NVME_CMBSZ_SZU_SHIFT) & NVME_CMBSZ_SZU_MASK;

	return 1ULL << (12 + 4 * szu);
}

static u32 nvme_cmb_size(struct nvme_dev *dev)
{
	return (dev->cmbsz >> NVME_CMBSZ_SZ_SHIFT) & NVME_CMBSZ_SZ_MASK;
}

static void nvme_map_cmb(struct nvme_dev *dev)
{
	u64 size, offset;
	resource_size_t bar_size;
	struct pci_dev *pdev = to_pci_dev(dev->dev);
	int bar;

	if (dev->cmb_size)
		return;

	dev->cmbsz = readl(dev->bar + NVME_REG_CMBSZ);
	if (!dev->cmbsz)
		return;
	dev->cmbloc = readl(dev->bar + NVME_REG_CMBLOC);

	size = nvme_cmb_size_unit(dev) * nvme_cmb_size(dev);
	offset = nvme_cmb_size_unit(dev) * NVME_CMB_OFST(dev->cmbloc);
	bar = NVME_CMB_BIR(dev->cmbloc);
	bar_size = pci_resource_len(pdev, bar);

	if (offset > bar_size)
		return;

	/*
	 * Controllers may support a CMB size larger than their BAR,
	 * for example, due to being behind a bridge. Reduce the CMB to
	 * the reported size of the BAR
	 */
	if (size > bar_size - offset)
		size = bar_size - offset;

	if (pci_p2pdma_add_resource(pdev, bar, size, offset)) {
		dev_warn(dev->ctrl.device,
			 "failed to register the CMB\n");
		return;
	}

	dev->cmb_size = size;
	dev->cmb_use_sqes = use_cmb_sqes && (dev->cmbsz & NVME_CMBSZ_SQS);

	if ((dev->cmbsz & (NVME_CMBSZ_WDS | NVME_CMBSZ_RDS)) ==
			(NVME_CMBSZ_WDS | NVME_CMBSZ_RDS))
		pci_p2pmem_publish(pdev, true);

	if (sysfs_add_file_to_group(&dev->ctrl.device->kobj,
				    &dev_attr_cmb.attr, NULL))
		dev_warn(dev->ctrl.device,
			 "failed to add sysfs attribute for CMB\n");
}

static inline void nvme_release_cmb(struct nvme_dev *dev)
{
	if (dev->cmb_size) {
		sysfs_remove_file_from_group(&dev->ctrl.device->kobj,
					     &dev_attr_cmb.attr, NULL);
		dev->cmb_size = 0;
	}
}

static int nvme_set_host_mem(struct nvme_dev *dev, u32 bits)
{
	u64 dma_addr = dev->host_mem_descs_dma;
	struct nvme_command c;
	int ret;

	memset(&c, 0, sizeof(c));
	c.features.opcode	= nvme_admin_set_features;
	c.features.fid		= cpu_to_le32(NVME_FEAT_HOST_MEM_BUF);
	c.features.dword11	= cpu_to_le32(bits);
	c.features.dword12	= cpu_to_le32(dev->host_mem_size >>
					      ilog2(dev->ctrl.page_size));
	c.features.dword13	= cpu_to_le32(lower_32_bits(dma_addr));
	c.features.dword14	= cpu_to_le32(upper_32_bits(dma_addr));
	c.features.dword15	= cpu_to_le32(dev->nr_host_mem_descs);

	ret = snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
	if (ret) {
		dev_warn(dev->ctrl.device,
			 "failed to set host mem (err %d, flags %#x).\n",
			 ret, bits);
	}
	return ret;
}

static void nvme_free_host_mem(struct nvme_dev *dev)
{
	int i;

	for (i = 0; i < dev->nr_host_mem_descs; i++) {
		struct nvme_host_mem_buf_desc *desc = &dev->host_mem_descs[i];
		size_t size = le32_to_cpu(desc->size) * dev->ctrl.page_size;

		dma_free_attrs(dev->dev, size, dev->host_mem_desc_bufs[i],
			       le64_to_cpu(desc->addr),
			       DMA_ATTR_NO_KERNEL_MAPPING | DMA_ATTR_NO_WARN);
	}

	kfree(dev->host_mem_desc_bufs);
	dev->host_mem_desc_bufs = NULL;
	dma_free_coherent(dev->dev,
			dev->nr_host_mem_descs * sizeof(*dev->host_mem_descs),
			dev->host_mem_descs, dev->host_mem_descs_dma);
	dev->host_mem_descs = NULL;
	dev->nr_host_mem_descs = 0;
}

static int __nvme_alloc_host_mem(struct nvme_dev *dev, u64 preferred,
		u32 chunk_size)
{
	struct nvme_host_mem_buf_desc *descs;
	u32 max_entries, len;
	dma_addr_t descs_dma;
	int i = 0;
	void **bufs;
	u64 size, tmp;

	tmp = (preferred + chunk_size - 1);
	do_div(tmp, chunk_size);
	max_entries = tmp;

	if (dev->ctrl.hmmaxd && dev->ctrl.hmmaxd < max_entries)
		max_entries = dev->ctrl.hmmaxd;

	descs = dma_alloc_coherent(dev->dev, max_entries * sizeof(*descs),
				   &descs_dma, GFP_KERNEL);
	if (!descs)
		goto out;

	bufs = kcalloc(max_entries, sizeof(*bufs), GFP_KERNEL);
	if (!bufs)
		goto out_free_descs;

	for (size = 0; size < preferred && i < max_entries; size += len) {
		dma_addr_t dma_addr;

		len = min_t(u64, chunk_size, preferred - size);
		bufs[i] = dma_alloc_attrs(dev->dev, len, &dma_addr, GFP_KERNEL,
				DMA_ATTR_NO_KERNEL_MAPPING | DMA_ATTR_NO_WARN);
		if (!bufs[i])
			break;

		descs[i].addr = cpu_to_le64(dma_addr);
		descs[i].size = cpu_to_le32(len / dev->ctrl.page_size);
		i++;
	}

	if (!size)
		goto out_free_bufs;

	dev->nr_host_mem_descs = i;
	dev->host_mem_size = size;
	dev->host_mem_descs = descs;
	dev->host_mem_descs_dma = descs_dma;
	dev->host_mem_desc_bufs = bufs;
	return 0;

out_free_bufs:
	while (--i >= 0) {
		size_t size = le32_to_cpu(descs[i].size) * dev->ctrl.page_size;

		dma_free_attrs(dev->dev, size, bufs[i],
			       le64_to_cpu(descs[i].addr),
			       DMA_ATTR_NO_KERNEL_MAPPING | DMA_ATTR_NO_WARN);
	}

	kfree(bufs);
out_free_descs:
	dma_free_coherent(dev->dev, max_entries * sizeof(*descs), descs,
			descs_dma);
out:
	dev->host_mem_descs = NULL;
	return -ENOMEM;
}

static int nvme_alloc_host_mem(struct nvme_dev *dev, u64 min, u64 preferred)
{
	u32 chunk_size;

	/* start big and work our way down */
	for (chunk_size = min_t(u64, preferred, PAGE_SIZE * MAX_ORDER_NR_PAGES);
	     chunk_size >= max_t(u32, dev->ctrl.hmminds * 4096, PAGE_SIZE * 2);
	     chunk_size /= 2) {
		if (!__nvme_alloc_host_mem(dev, preferred, chunk_size)) {
			if (!min || dev->host_mem_size >= min)
				return 0;
			nvme_free_host_mem(dev);
		}
	}

	return -ENOMEM;
}

static int nvme_setup_host_mem(struct nvme_dev *dev)
{
	u64 max = (u64)max_host_mem_size_mb * SZ_1M;
	u64 preferred = (u64)dev->ctrl.hmpre * 4096;
	u64 min = (u64)dev->ctrl.hmmin * 4096;
	u32 enable_bits = NVME_HOST_MEM_ENABLE;
	int ret;

	preferred = min(preferred, max);
	if (min > max) {
		dev_warn(dev->ctrl.device,
			"min host memory (%lld MiB) above limit (%d MiB).\n",
			min >> ilog2(SZ_1M), max_host_mem_size_mb);
		nvme_free_host_mem(dev);
		return 0;
	}

	/*
	 * If we already have a buffer allocated check if we can reuse it.
	 */
	if (dev->host_mem_descs) {
		if (dev->host_mem_size >= min)
			enable_bits |= NVME_HOST_MEM_RETURN;
		else
			nvme_free_host_mem(dev);
	}

	if (!dev->host_mem_descs) {
		if (nvme_alloc_host_mem(dev, min, preferred)) {
			dev_warn(dev->ctrl.device,
				"failed to allocate host memory buffer.\n");
			return 0; /* controller must work without HMB */
		}

		dev_info(dev->ctrl.device,
			"allocated %lld MiB host memory buffer.\n",
			dev->host_mem_size >> ilog2(SZ_1M));
	}

	ret = nvme_set_host_mem(dev, enable_bits);
	if (ret)
		nvme_free_host_mem(dev);
	return ret;
}

/*
 * nirqs is the number of interrupts available for write and read
 * queues. The core already reserved an interrupt for the admin queue.
 */
static void nvme_calc_irq_sets(struct irq_affinity *affd, unsigned int nrirqs)
{
	struct nvme_dev *dev = affd->priv;
	unsigned int nr_read_queues, nr_write_queues = dev->nr_write_queues;

	/*
	 * If there is no interupt available for queues, ensure that
	 * the default queue is set to 1. The affinity set size is
	 * also set to one, but the irq core ignores it for this case.
	 *
	 * If only one interrupt is available or 'write_queue' == 0, combine
	 * write and read queues.
	 *
	 * If 'write_queues' > 0, ensure it leaves room for at least one read
	 * queue.
	 */
	if (!nrirqs) {
		nrirqs = 1;
		nr_read_queues = 0;
	} else if (nrirqs == 1 || !nr_write_queues) {
		nr_read_queues = 0;
	} else if (nr_write_queues >= nrirqs) {
		nr_read_queues = 1;
	} else {
		nr_read_queues = nrirqs - nr_write_queues;
	}

	dev->io_queues[HCTX_TYPE_DEFAULT] = nrirqs - nr_read_queues;
	affd->set_size[HCTX_TYPE_DEFAULT] = nrirqs - nr_read_queues;
	dev->io_queues[HCTX_TYPE_READ] = nr_read_queues;
	affd->set_size[HCTX_TYPE_READ] = nr_read_queues;
	affd->nr_sets = nr_read_queues ? 2 : 1;
}

static int nvme_setup_irqs(struct nvme_dev *dev, unsigned int nr_io_queues)
{
	struct pci_dev *pdev = to_pci_dev(dev->dev);
	struct irq_affinity affd = {
		.pre_vectors	= 1,
		.calc_sets	= nvme_calc_irq_sets,
		.priv		= dev,
	};
	struct irq_affinity *p_affd = &affd;
	unsigned int irq_queues, this_p_queues;
	unsigned int flags = PCI_IRQ_ALL_TYPES;
	unsigned int affvecs;
	int nr_irqs;

	/*
	 * Poll queues don't need interrupts, but we need at least one IO
	 * queue left over for non-polled IO.
	 */
	this_p_queues = dev->nr_poll_queues;
	if (this_p_queues >= nr_io_queues) {
		this_p_queues = nr_io_queues - 1;
		irq_queues = 1;
	} else {
		irq_queues = nr_io_queues - this_p_queues + 1;
	}
	dev->io_queues[HCTX_TYPE_POLL] = this_p_queues;

	/* Initialize for the single interrupt case */
	dev->io_queues[HCTX_TYPE_DEFAULT] = 1;
	dev->io_queues[HCTX_TYPE_READ] = 0;

	/*
	 * Some Apple controllers require all queues to use the
	 * first vector.
	 */
	if (dev->ctrl.quirks & NVME_QUIRK_SINGLE_VECTOR)
		irq_queues = 1;

	if (smp_affinity_enable)
		flags |= PCI_IRQ_AFFINITY;
	else
		p_affd = NULL;

	nr_irqs = pci_alloc_irq_vectors_affinity(pdev, 1, irq_queues, flags, p_affd);

	/*
	 * When smp_affinity_enable is disabled, nvme_calc_irq_sets() is not
	 * called to initialize dev->io_queuesp[] when allocating drivers.
	 * So here we need to call nvme_calc_irq_sets() explicitly.
	 */
	if (nr_irqs > 0 && !smp_affinity_enable) {
		if (nr_irqs > affd.pre_vectors)
			affvecs = nr_irqs - affd.pre_vectors;
		else
			affvecs = 0;

		nvme_calc_irq_sets(&affd, affvecs);
	}

	return nr_irqs;
}

static void nvme_disable_io_queues(struct nvme_dev *dev)
{
	if (__nvme_disable_io_queues(dev, nvme_admin_delete_sq))
		__nvme_disable_io_queues(dev, nvme_admin_delete_cq);
}

/* ------------------------------------------------------------------ *
 *  snvme segment 7: user-IO-queue teardown.
 *
 *  Companion to nvme_create_user_queue() / nvme_create_io_queues_mix()
 *  (segments 6c/6d).  Those segments install user-pinned SQ/CQ rings on
 *  the controller for QIDs in [user_start_qid, user_start_qid +
 *  online_user_queues), but they do NOT allocate a struct nvme_queue
 *  for those QIDs and they do NOT bump dev->online_queues -- the user
 *  pages are owned entirely by libnvm/CUDA, the kernel only stamps the
 *  controller-side Create-IO-{CQ,SQ} commands and lets the user ring
 *  the doorbells from BAR0 directly.
 *
 *  Consequence for teardown: the standard nvme_disable_io_queues() /
 *  nvme_suspend_io_queues() pair walks dev->queues[1 .. online_queues)
 *  and is therefore *blind* to the user QIDs.  Without this helper,
 *  every nvme_dev_disable() / nvme_remove() / nvme_reset_work() cycle
 *  leaves the user QIDs allocated on the controller.  The next
 *  SNVM_DEVICE_BIND then re-issues Create-IO-CQ for the same QID and
 *  the controller responds with status code 0x101 (Invalid Queue
 *  Identifier) -- the symptom is a probe that "just fails" with no
 *  obvious reason in dmesg, because nvme_create_user_queue()'s
 *  pr_err() doesn't decode NVMe status codes.
 *
 *  Bug-for-bug honesty: snvme-5.15.0/pci.c has this same defect (see
 *  pci.c:2823-2875 in that tree -- nvme_dev_disable() is purely a sed
 *  rename of upstream and never learned about online_user_queues).
 *  We ARE diverging from 5.15.0 here, deliberately, because the
 *  symptom only manifests on the second BIND and 5.15.0's smoke tests
 *  all do a single BIND.  See PORTING.md section 7.3.1 for the broader rule
 *  ("re-audit teardown after every uplift").
 *
 *  Ordering: SQs first, then CQs (NVMe-spec rule: Delete-IO-CQ fails
 *  while any SQ still references the CQ).  We mirror the convention
 *  of __nvme_disable_io_queues(): walk highest QID downwards so that
 *  if any single delete fails (controller reset, device gone), we've
 *  at least released the higher QIDs and the next probe has a chance
 *  of reusing them.
 *
 *  Failure handling: every adapter_delete_{sq,cq}() failure is
 *  logged and otherwise ignored -- we are on the teardown path and
 *  there is no useful recovery to attempt.  The caller (nvme_dev_disable)
 *  proceeds to nvme_disable_io_queues() / nvme_disable_admin_queue()
 *  unconditionally.
 *
 *  Reset semantics: online_user_queues and user_start_qid are cleared
 *  here so that a subsequent reset_work() -> nvme_setup_io_queues() ->
 *  nvme_create_io_queues_mix() reruns segment 6d from a clean slate.
 *  Userspace is expected to re-issue NVM_MAP_* / NVM_SET_SHARE_REG
 *  before the next BIND; until that happens, nr_user_use_cq stays at
 *  the value the previous bring-up computed in
 *  nvme_setup_io_queues() (segment 6b), which is harmless because the
 *  loop in nvme_create_io_queues_mix() will simply find no maps and
 *  bail at the first iteration.
 *
 *  Concurrency: invoked exclusively from nvme_dev_disable() under
 *  dev->shutdown_lock, so there is no possible concurrent writer to
 *  online_user_queues / user_start_qid.
 */
static void snvme_disable_user_io_queues(struct nvme_dev *dev)
{
	unsigned int n = dev->online_user_queues;
	unsigned int start = dev->user_start_qid;
	int qid;
	int ret;

	if (!n || !start) {
		/*
		 * Either no user queues were ever brought up on this dev
		 * (the common case for kernel-only NVMe usage), or a
		 * previous teardown already cleared the counters.  Nothing
		 * to do.
		 */
		return;
	}

	/*
	 * Walk highest user QID downwards so a partial failure leaves
	 * the lower (older) QIDs intact for the next bring-up to retry.
	 */
	for (qid = (int)(start + n - 1); qid >= (int)start; qid--) {
		ret = adapter_delete_sq(dev, (u16)qid);
		if (ret)
			dev_warn(dev->ctrl.device,
				 "Delete-IO-SQ failed (user qid=%d): %d\n",
				 qid, ret);
	}
	for (qid = (int)(start + n - 1); qid >= (int)start; qid--) {
		ret = adapter_delete_cq(dev, (u16)qid);
		if (ret)
			dev_warn(dev->ctrl.device,
				 "Delete-IO-CQ failed (user qid=%d): %d\n",
				 qid, ret);
	}

	/*
	 * Reset the bookkeeping unconditionally.  Even if some Delete-IO
	 * commands above failed, from this driver's point of view those
	 * QIDs are no longer in use; if the controller still thinks
	 * otherwise, the next BIND will surface the conflict explicitly.
	 */
	dev->online_user_queues = 0;
	dev->user_start_qid = 0;
}

static unsigned int nvme_max_io_queues(struct nvme_dev *dev)
{
	return num_possible_cpus() + dev->nr_write_queues + dev->nr_poll_queues;
}

/*
 * s_nvme_setup_io_queues: snvme variant of upstream nvme_setup_io_queues.
 *
 * Renamed per PORTING.md section 3.1 because the function semantics have
 * diverged from upstream: when dev->use_user_allocated is set (segment
 * 6a populated this from ctrl->use_sreg), the queue count negotiation
 * is *biased upwards* by nr_user_allocated_cq so that there is room
 * for both the kernel-owned IO queues AND the user-owned ones the
 * caller already pinned via NVM_MAP_*; the post-Set-Features
 * reconciliation block then decides how many user queues survive
 * based on what the controller actually granted.
 *
 * The structural diff vs 5.4 upstream is intentionally minimal --
 * the lock / IRQ / remap-bar paths are kept verbatim so 5.4 reset
 * semantics aren't disturbed.  We do NOT pull in 5.15's
 * nvme_setup_io_queues_trylock / shutdown_lock churn: that is an
 * upstream-5.15 internal refactor unrelated to use_sreg, and merging
 * it into 5.4 would change the lock acquisition order against
 * nvme_dev_disable() / nvme_reset_work().
 */
static int s_nvme_setup_io_queues(struct nvme_dev *dev)
{
	struct nvme_queue *adminq = &dev->queues[0];
	struct pci_dev *pdev = to_pci_dev(dev->dev);
	unsigned int nr_io_queues;
	unsigned int kernel_target;
	unsigned long size;
	int result;

	/*
	 * Sample the module parameters once at reset time so that we have
	 * stable values to work with.  The per-BDF NVM_SET_IOQ_NUM
	 * overrides (if any) were already folded into dev->nr_write_queues
	 * / dev->nr_poll_queues during segment 6a's probe-time copy from
	 * struct ctrl, so just re-sample the upstream parameters when no
	 * override is in effect.
	 */
	if (!dev->cap_kernel_ioq) {
		dev->nr_write_queues = write_queues;
		dev->nr_poll_queues = poll_queues;
	}

	/*
	 * If tags are shared with admin queue (Apple bug), then
	 * make sure we only use one IO queue.
	 *
	 * Otherwise: stay byte-for-byte compatible with in-tree
	 * nvme-5.4 here -- ask the controller for nvme_max_io_queues()
	 * and let snvme_set_queue_count() write back the real grant.
	 * In-tree always has had the property that nr_io_queues
	 * coming out of Set-Features is the controller's
	 * authoritative IOQ ceiling, which we capture below as
	 * dev->ctrl_max_io_queues for the user QID pool.
	 *
	 * Any cap_kernel_ioq the user requested is applied AFTER
	 * the negotiation (further down): the kernel just stops
	 * creating IOQs once it has consumed `cap` of them, leaving
	 * QIDs [cap+1 .. ctrl_max_io_queues] free for
	 * NVM_ADD_USER_QUEUE.  This decouples "what the controller
	 * is willing to grant" (a hardware fact) from "what the
	 * kernel actually consumes" (a policy decision).
	 */
	if (dev->ctrl.quirks & NVME_QUIRK_SHARED_TAGS)
		nr_io_queues = 1;
	else
		nr_io_queues = min(nvme_max_io_queues(dev),
				   dev->nr_allocated_queues - 1);

	/*
	 * kernel_target: how many IOQs the kernel ultimately wants for
	 * itself.  Used by the B3 cap-only path after Set-Features to
	 * shrink nr_io_queues so the user QID pool gets the leftover
	 * range.  Default = nr_io_queues (no narrowing); cap_kernel_ioq=N
	 * narrows to min(N, ...).
	 */
	kernel_target = nr_io_queues;
	if (dev->cap_kernel_ioq && dev->cap_kernel_ioq < kernel_target)
		kernel_target = dev->cap_kernel_ioq;

	result = snvme_set_queue_count(&dev->ctrl, &nr_io_queues);

	/*
	 * snvme B3: record the controller-granted IOQ ceiling.  This
	 * is the authoritative bound for legal QID values used by
	 * NVM_ADD_USER_QUEUE; the user QID pool will be
	 * [online_queues..ctrl_max_io_queues].  Captured BEFORE any
	 * downstream code mutates nr_io_queues (the cap-shrink below)
	 * so the pool sizer always sees the controller's real
	 * grant -- not whatever value the kernel ends up consuming.
	 */
	if (result == 0)
		dev->ctrl_max_io_queues = nr_io_queues;

	/*
	 * B3 cap-only path: shrink the kernel-side consumption to
	 * cap_kernel_ioq AFTER the controller negotiation.  The
	 * controller already granted up to nr_io_queues, but we want
	 * QIDs [kernel_target+1 .. ctrl_max_io_queues] to remain
	 * unused by the kernel so NVM_ADD_USER_QUEUE can claim them.
	 *
	 * When cap_kernel_ioq is 0, this whole block is a no-op and
	 * nr_io_queues flows to setup_irqs/create_io_queues unchanged
	 * from the in-tree Set-Features grant.
	 */
	if (dev->cap_kernel_ioq &&
	    nr_io_queues > kernel_target) {
		pr_info("capping kernel-side IOQ count from %d to %u "
			"(ctrl_max=%u, user pool gets [%u..%u])\n",
			nr_io_queues, kernel_target,
			dev->ctrl_max_io_queues,
			kernel_target + 1, dev->ctrl_max_io_queues);
		nr_io_queues = kernel_target;
	}

	if (result < 0)
		return result;

	if (nr_io_queues == 0)
		return 0;

	clear_bit(NVMEQ_ENABLED, &adminq->flags);

	if (dev->cmb_use_sqes) {
		result = nvme_cmb_qdepth(dev, nr_io_queues,
				sizeof(struct nvme_command));
		if (result > 0)
			dev->q_depth = result;
		else
			dev->cmb_use_sqes = false;
	}

	do {
		size = db_bar_size(dev, nr_io_queues);
		result = nvme_remap_bar(dev, size);
		if (!result)
			break;
		if (!--nr_io_queues)
			return -ENOMEM;
	} while (1);
	adminq->q_db = dev->dbs;

 retry:
	/* Deregister the admin queue's interrupt */
	pci_free_irq(pdev, 0, adminq);

	/*
	 * If we enable msix early due to not intx, disable it again before
	 * setting up the full range we need.
	 */
	pci_free_irq_vectors(pdev);

	result = nvme_setup_irqs(dev, nr_io_queues);
	if (result <= 0)
		return -EIO;

	dev->num_vecs = result;
	result = max(result - 1, 1);
	dev->max_qid = result + dev->io_queues[HCTX_TYPE_POLL];

	/*
	 * Should investigate if there's a performance win from allocating
	 * more queues than interrupt vectors; it might allow the submission
	 * path to scale better, even if the receive path is limited by the
	 * number of interrupts.
	 */
	result = queue_request_irq(adminq);
	if (result)
		return result;
	set_bit(NVMEQ_ENABLED, &adminq->flags);

	result = nvme_create_io_queues(dev);
	if (result || dev->online_queues < 2)
		return result;

	if (dev->online_queues - 1 < dev->max_qid) {
		nr_io_queues = dev->online_queues - 1;
		nvme_disable_io_queues(dev);
		nvme_suspend_io_queues(dev);
		goto retry;
	}
	dev_info(dev->ctrl.device, "%d/%d/%d/%d default/read/poll/user queues\n",
					dev->io_queues[HCTX_TYPE_DEFAULT],
					dev->io_queues[HCTX_TYPE_READ],
				dev->io_queues[HCTX_TYPE_POLL],
				dev->online_user_queues);
	return 0;
}

static void nvme_del_queue_end(struct request *req, blk_status_t error)
{
	struct nvme_queue *nvmeq = req->end_io_data;

	blk_mq_free_request(req);
	complete(&nvmeq->delete_done);
}

static void nvme_del_cq_end(struct request *req, blk_status_t error)
{
	struct nvme_queue *nvmeq = req->end_io_data;

	if (error)
		set_bit(NVMEQ_DELETE_ERROR, &nvmeq->flags);

	nvme_del_queue_end(req, error);
}

static int nvme_delete_queue(struct nvme_queue *nvmeq, u8 opcode)
{
	struct request_queue *q = nvmeq->dev->ctrl.admin_q;
	struct request *req;
	struct nvme_command cmd;

	memset(&cmd, 0, sizeof(cmd));
	cmd.delete_queue.opcode = opcode;
	cmd.delete_queue.qid = cpu_to_le16(nvmeq->qid);

	req = snvme_alloc_request(q, &cmd, BLK_MQ_REQ_NOWAIT, NVME_QID_ANY);
	if (IS_ERR(req))
		return PTR_ERR(req);

	req->timeout = ADMIN_TIMEOUT;
	req->end_io_data = nvmeq;

	init_completion(&nvmeq->delete_done);
	blk_execute_rq_nowait(q, NULL, req, false,
			opcode == nvme_admin_delete_cq ?
				nvme_del_cq_end : nvme_del_queue_end);
	return 0;
}

static bool __nvme_disable_io_queues(struct nvme_dev *dev, u8 opcode)
{
	int nr_queues = dev->online_queues - 1, sent = 0;
	unsigned long timeout;

 retry:
	timeout = ADMIN_TIMEOUT;
	while (nr_queues > 0) {
		if (nvme_delete_queue(&dev->queues[nr_queues], opcode))
			break;
		nr_queues--;
		sent++;
	}
	while (sent) {
		struct nvme_queue *nvmeq = &dev->queues[nr_queues + sent];

		timeout = wait_for_completion_io_timeout(&nvmeq->delete_done,
				timeout);
		if (timeout == 0)
			return false;

		sent--;
		if (nr_queues)
			goto retry;
	}
	return true;
}

static void nvme_dev_add(struct nvme_dev *dev)
{
	int ret;

	if (!dev->ctrl.tagset) {
		dev->tagset.ops = &nvme_mq_ops;
		dev->tagset.nr_hw_queues = dev->online_queues - 1;
		dev->tagset.nr_maps = 2; /* default + read */
		if (dev->io_queues[HCTX_TYPE_POLL])
			dev->tagset.nr_maps++;
		dev->tagset.timeout = NVME_IO_TIMEOUT;
		dev->tagset.numa_node = dev->ctrl.numa_node;
		dev->tagset.queue_depth =
				min_t(int, dev->q_depth, BLK_MQ_MAX_DEPTH) - 1;
		dev->tagset.cmd_size = sizeof(struct nvme_iod);
		dev->tagset.flags = BLK_MQ_F_SHOULD_MERGE;
		dev->tagset.driver_data = dev;

		/*
		 * Some Apple controllers requires tags to be unique
		 * across admin and IO queue, so reserve the first 32
		 * tags of the IO queue.
		 */
		if (dev->ctrl.quirks & NVME_QUIRK_SHARED_TAGS)
			dev->tagset.reserved_tags = NVME_AQ_DEPTH;

		ret = blk_mq_alloc_tag_set(&dev->tagset);
		if (ret) {
			dev_warn(dev->ctrl.device,
				"IO queues tagset allocation failed %d\n", ret);
			return;
		}
		dev->ctrl.tagset = &dev->tagset;
	} else {
		blk_mq_update_nr_hw_queues(&dev->tagset, dev->online_queues - 1);

		/* Free previously allocated queues that are no longer usable */
		nvme_free_queues(dev, dev->online_queues);
	}

	nvme_dbbuf_set(dev);
}

static int nvme_pci_enable(struct nvme_dev *dev)
{
	int result = -ENOMEM;
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	if (pci_enable_device_mem(pdev))
		return result;

	pci_set_master(pdev);

	if (dma_set_mask_and_coherent(dev->dev, DMA_BIT_MASK(64)))
		goto disable;

	if (readl(dev->bar + NVME_REG_CSTS) == -1) {
		result = -ENODEV;
		goto disable;
	}

	/*
	 * Some devices and/or platforms don't advertise or work with INTx
	 * interrupts. Pre-enable a single MSIX or MSI vec for setup. We'll
	 * adjust this later.
	 */
	result = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_ALL_TYPES);
	if (result < 0)
		return result;

	dev->ctrl.cap = lo_hi_readq(dev->bar + NVME_REG_CAP);

	dev->q_depth = NVME_CAP_MQES(dev->ctrl.cap) + 1;
	dev->ctrl.sqsize = dev->q_depth - 1; /* 0's based queue depth */
	dev->db_stride = 1 << NVME_CAP_STRIDE(dev->ctrl.cap);
	dev->dbs = dev->bar + 4096;

	/*
	 * Some Apple controllers require a non-standard SQE size.
	 * Interestingly they also seem to ignore the CC:IOSQES register
	 * so we don't bother updating it here.
	 */
	if (dev->ctrl.quirks & NVME_QUIRK_128_BYTES_SQES)
		dev->io_sqes = 7;
	else
		dev->io_sqes = NVME_NVM_IOSQES;

	/*
	 * Temporary fix for the Apple controller found in the MacBook8,1 and
	 * some MacBook7,1 to avoid controller resets and data loss.
	 */
	if (pdev->vendor == PCI_VENDOR_ID_APPLE && pdev->device == 0x2001) {
		dev->q_depth = 2;
		dev_warn(dev->ctrl.device, "detected Apple NVMe controller, "
			"set queue depth=%u to work around controller resets\n",
			dev->q_depth);
	} else if (pdev->vendor == PCI_VENDOR_ID_SAMSUNG &&
		   (pdev->device == 0xa821 || pdev->device == 0xa822) &&
		   NVME_CAP_MQES(dev->ctrl.cap) == 0) {
		dev->q_depth = 64;
		dev_err(dev->ctrl.device, "detected PM1725 NVMe controller, "
                        "set queue depth=%u\n", dev->q_depth);
	}

	/*
	 * Controllers with the shared tags quirk need the IO queue to be
	 * big enough so that we get 32 tags for the admin queue
	 */
	if ((dev->ctrl.quirks & NVME_QUIRK_SHARED_TAGS) &&
	    (dev->q_depth < (NVME_AQ_DEPTH + 2))) {
		dev->q_depth = NVME_AQ_DEPTH + 2;
		dev_warn(dev->ctrl.device, "IO queue depth clamped to %d\n",
			 dev->q_depth);
	}


	nvme_map_cmb(dev);

	pci_enable_pcie_error_reporting(pdev);
	pci_save_state(pdev);
	return 0;

 disable:
	pci_disable_device(pdev);
	return result;
}

static void nvme_dev_unmap(struct nvme_dev *dev)
{
	if (dev->bar)
		iounmap(dev->bar);
	pci_release_mem_regions(to_pci_dev(dev->dev));
}

static void nvme_pci_disable(struct nvme_dev *dev)
{
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	pci_free_irq_vectors(pdev);

	if (pci_is_enabled(pdev)) {
		pci_disable_pcie_error_reporting(pdev);
		pci_disable_device(pdev);
	}
}

static void nvme_dev_disable(struct nvme_dev *dev, bool shutdown)
{
	bool dead = true, freeze = false;
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	mutex_lock(&dev->shutdown_lock);
	if (pci_is_enabled(pdev)) {
		u32 csts = readl(dev->bar + NVME_REG_CSTS);

		if (dev->ctrl.state == NVME_CTRL_LIVE ||
		    dev->ctrl.state == NVME_CTRL_RESETTING) {
			freeze = true;
			snvme_start_freeze(&dev->ctrl);
		}
		dead = !!((csts & NVME_CSTS_CFS) || !(csts & NVME_CSTS_RDY) ||
			pdev->error_state  != pci_channel_io_normal);
	}

	/*
	 * Give the controller a chance to complete all entered requests if
	 * doing a safe shutdown.
	 */
	if (!dead && shutdown && freeze)
		snvme_wait_freeze_timeout(&dev->ctrl, NVME_IO_TIMEOUT);

	snvme_stop_queues(&dev->ctrl);

	if (!dead && dev->ctrl.queue_count > 0) {
		/*
		 * snvme segment 7: tear down user-pinned IO queues BEFORE
		 * the kernel-owned ones.  Two reasons:
		 *  (a) Both paths use dev->ctrl.admin_q, but
		 *      nvme_disable_admin_queue(shutdown=true) further
		 *      down sends NVME_REG_CSTS shutdown notice that may
		 *      stall any subsequent admin command -- we want our
		 *      Delete-IO commands issued while admin is still
		 *      live.
		 *  (b) On the next probe / reset_work() the user QIDs
		 *      get re-allocated by nvme_create_io_queues_mix().
		 *      Releasing them here is what prevents the
		 *      "Invalid Queue Identifier" stall described in
		 *      snvme_disable_user_io_queues()'s comment.
		 *
		 * We do NOT gate this on `shutdown` -- both reset and
		 * remove paths must drop the user QIDs.  The function is
		 * a no-op when no user queues were ever brought up.
		 */
		snvme_disable_user_io_queues(dev);
		nvme_disable_io_queues(dev);
		nvme_disable_admin_queue(dev, shutdown);
	} else if (dev->online_user_queues) {
		/*
		 * Controller is dead (CFS set, !RDY, or pci_channel_io
		 * not normal): admin commands won't be acknowledged.  Just
		 * forget the user QIDs locally so the next probe starts
		 * from a clean dev->online_user_queues = 0 / user_start_qid
		 * = 0.  The controller will be reset (CC.EN cleared in
		 * nvme_disable_admin_queue path on the next bring-up),
		 * which implicitly drops every queue it knew about.
		 */
		dev->online_user_queues = 0;
		dev->user_start_qid = 0;
	}
	nvme_suspend_io_queues(dev);
	nvme_suspend_queue(&dev->queues[0]);
	nvme_pci_disable(dev);
	nvme_reap_pending_cqes(dev);

	blk_mq_tagset_busy_iter(&dev->tagset, snvme_cancel_request, &dev->ctrl);
	blk_mq_tagset_busy_iter(&dev->admin_tagset, snvme_cancel_request, &dev->ctrl);
	blk_mq_tagset_wait_completed_request(&dev->tagset);
	blk_mq_tagset_wait_completed_request(&dev->admin_tagset);

	/*
	 * The driver will not be starting up queues again if shutting down so
	 * must flush all entered requests to their failed completion to avoid
	 * deadlocking blk-mq hot-cpu notifier.
	 */
	if (shutdown) {
		snvme_start_queues(&dev->ctrl);
		if (dev->ctrl.admin_q && !blk_queue_dying(dev->ctrl.admin_q))
			blk_mq_unquiesce_queue(dev->ctrl.admin_q);
	}
	mutex_unlock(&dev->shutdown_lock);
}

static int nvme_disable_prepare_reset(struct nvme_dev *dev, bool shutdown)
{
	if (!snvme_wait_reset(&dev->ctrl))
		return -EBUSY;
	nvme_dev_disable(dev, shutdown);
	return 0;
}

static int nvme_setup_prp_pools(struct nvme_dev *dev)
{
	dev->prp_page_pool = dma_pool_create("prp list page", dev->dev,
						PAGE_SIZE, PAGE_SIZE, 0);
	if (!dev->prp_page_pool)
		return -ENOMEM;

	/* Optimisation for I/Os between 4k and 128k */
	dev->prp_small_pool = dma_pool_create("prp list 256", dev->dev,
						256, 256, 0);
	if (!dev->prp_small_pool) {
		dma_pool_destroy(dev->prp_page_pool);
		return -ENOMEM;
	}
	return 0;
}

static void nvme_release_prp_pools(struct nvme_dev *dev)
{
	dma_pool_destroy(dev->prp_page_pool);
	dma_pool_destroy(dev->prp_small_pool);
}

static void nvme_free_tagset(struct nvme_dev *dev)
{
	if (dev->tagset.tags)
		blk_mq_free_tag_set(&dev->tagset);
	dev->ctrl.tagset = NULL;
}

static void nvme_pci_free_ctrl(struct nvme_ctrl *ctrl)
{
	struct nvme_dev *dev = to_nvme_dev(ctrl);

	nvme_dbbuf_dma_free(dev);
	put_device(dev->dev);
	nvme_free_tagset(dev);
	if (dev->ctrl.admin_q)
		blk_put_queue(dev->ctrl.admin_q);
	kfree(dev->queues);
	free_opal_dev(dev->ctrl.opal_dev);
	mempool_destroy(dev->iod_mempool);
	kfree(dev);
}

static void nvme_remove_dead_ctrl(struct nvme_dev *dev)
{
	/*
	 * Set state to deleting now to avoid blocking snvme_wait_reset(), which
	 * may be holding this pci_dev's device lock.
	 */
	snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_DELETING);
	nvme_get_ctrl(&dev->ctrl);
	nvme_dev_disable(dev, false);
	snvme_kill_queues(&dev->ctrl);
	if (!queue_work(s_nvme_wq, &dev->remove_work))
		nvme_put_ctrl(&dev->ctrl);
}

static void nvme_reset_work(struct work_struct *work)
{
	struct nvme_dev *dev =
		container_of(work, struct nvme_dev, ctrl.reset_work);
	bool was_suspend = !!(dev->ctrl.ctrl_config & NVME_CC_SHN_NORMAL);
	int result;

	if (dev->ctrl.state != NVME_CTRL_RESETTING) {
		dev_warn(dev->ctrl.device, "ctrl state %d is not RESETTING\n",
			 dev->ctrl.state);
		result = -ENODEV;
		goto out;
	}

	/*
	 * If we're called to reset a live controller first shut it down before
	 * moving on.
	 */
	if (dev->ctrl.ctrl_config & NVME_CC_ENABLE)
		nvme_dev_disable(dev, false);
	snvme_sync_queues(&dev->ctrl);

	mutex_lock(&dev->shutdown_lock);
	result = nvme_pci_enable(dev);
	if (result)
		goto out_unlock;

	result = nvme_pci_configure_admin_queue(dev);
	if (result)
		goto out_unlock;

	result = nvme_alloc_admin_tags(dev);
	if (result)
		goto out_unlock;

	/*
	 * Limit the max command size to prevent iod->sg allocations going
	 * over a single page.
	 */
	dev->ctrl.max_hw_sectors = min_t(u32,
		NVME_MAX_KB_SZ << 1, dma_max_mapping_size(dev->dev) >> 9);
	dev->ctrl.max_segments = NVME_MAX_SEGS;

	/*
	 * Don't limit the IOMMU merged segment size.
	 */
	dma_set_max_seg_size(dev->dev, 0xffffffff);

	mutex_unlock(&dev->shutdown_lock);

	/*
	 * Introduce CONNECTING state from nvme-fc/rdma transports to mark the
	 * initializing procedure here.
	 */
	if (!snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_CONNECTING)) {
		dev_warn(dev->ctrl.device,
			"failed to mark controller CONNECTING\n");
		result = -EBUSY;
		goto out;
	}

	result = snvme_init_identify(&dev->ctrl);
	if (result)
		goto out;

	if (dev->ctrl.oacs & NVME_CTRL_OACS_SEC_SUPP) {
		if (!dev->ctrl.opal_dev)
			dev->ctrl.opal_dev =
				init_opal_dev(&dev->ctrl, &snvme_sec_submit);
		else if (was_suspend)
			opal_unlock_from_suspend(dev->ctrl.opal_dev);
	} else {
		free_opal_dev(dev->ctrl.opal_dev);
		dev->ctrl.opal_dev = NULL;
	}

	if (dev->ctrl.oacs & NVME_CTRL_OACS_DBBUF_SUPP) {
		result = nvme_dbbuf_dma_alloc(dev);
		if (result)
			dev_warn(dev->dev,
				 "unable to allocate dma for dbbuf\n");
	}

	if (dev->ctrl.hmpre) {
		result = nvme_setup_host_mem(dev);
		if (result < 0)
			goto out;
	}

	result = s_nvme_setup_io_queues(dev);
	if (result)
		goto out;

	/*
	 * Keep the controller around but remove all namespaces if we don't have
	 * any working I/O queue.
	 */
	if (dev->online_queues < 2) {
		dev_warn(dev->ctrl.device, "IO queues not created\n");
		snvme_kill_queues(&dev->ctrl);
		snvme_remove_namespaces(&dev->ctrl);
		nvme_free_tagset(dev);
	} else {
		snvme_start_queues(&dev->ctrl);
		snvme_wait_freeze(&dev->ctrl);
		nvme_dev_add(dev);
		snvme_unfreeze(&dev->ctrl);
	}

	/*
	 * If only admin queue live, keep it to do further investigation or
	 * recovery.
	 */
	if (!snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_LIVE)) {
		dev_warn(dev->ctrl.device,
			"failed to mark controller live state\n");
		result = -ENODEV;
		goto out;
	}

	snvme_start_ctrl(&dev->ctrl);
	return;

 out_unlock:
	mutex_unlock(&dev->shutdown_lock);
 out:
	if (result)
		dev_warn(dev->ctrl.device,
			 "Removing after probe failure status: %d\n", result);
	nvme_remove_dead_ctrl(dev);
}

static void nvme_remove_dead_ctrl_work(struct work_struct *work)
{
	struct nvme_dev *dev = container_of(work, struct nvme_dev, remove_work);
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	if (pci_get_drvdata(pdev))
		device_release_driver(&pdev->dev);
	nvme_put_ctrl(&dev->ctrl);
}

static int nvme_pci_reg_read32(struct nvme_ctrl *ctrl, u32 off, u32 *val)
{
	*val = readl(to_nvme_dev(ctrl)->bar + off);
	return 0;
}

static int nvme_pci_reg_write32(struct nvme_ctrl *ctrl, u32 off, u32 val)
{
	writel(val, to_nvme_dev(ctrl)->bar + off);
	return 0;
}

static int nvme_pci_reg_read64(struct nvme_ctrl *ctrl, u32 off, u64 *val)
{
	*val = lo_hi_readq(to_nvme_dev(ctrl)->bar + off);
	return 0;
}

static int nvme_pci_get_address(struct nvme_ctrl *ctrl, char *buf, int size)
{
	struct pci_dev *pdev = to_pci_dev(to_nvme_dev(ctrl)->dev);

	return snprintf(buf, size, "%s", dev_name(&pdev->dev));
}

static const struct nvme_ctrl_ops nvme_pci_ctrl_ops = {
	.name			= "pcie",
	.module			= THIS_MODULE,
	.flags			= NVME_F_METADATA_SUPPORTED |
				  NVME_F_PCI_P2PDMA,
	.reg_read32		= nvme_pci_reg_read32,
	.reg_write32		= nvme_pci_reg_write32,
	.reg_read64		= nvme_pci_reg_read64,
	.free_ctrl		= nvme_pci_free_ctrl,
	.submit_async_event	= nvme_pci_submit_async_event,
	.get_address		= nvme_pci_get_address,
};

static int nvme_dev_map(struct nvme_dev *dev)
{
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	/*
	 * PORTING.md section 2: pci_request_mem_regions() takes a name
	 * that ends up in /proc/iomem as the owner tag.  Per-BDF the PCI
	 * core enforces single-driver-binds-single-device, so we never
	 * race the in-tree nvme.ko on the same BAR, but if both drivers
	 * are loaded and bind different NVMes the operator gets two
	 * /proc/iomem lines both tagged "nvme" -- unambiguous only after
	 * grepping /sys.  Use "snvme" so the iomem owner string matches
	 * the PCI driver name (pci_driver.name == "snvme").
	 */
	if (pci_request_mem_regions(pdev, "snvme"))
		return -ENODEV;

	if (nvme_remap_bar(dev, NVME_REG_DBS + 4096))
		goto release;

	return 0;
  release:
	pci_release_mem_regions(pdev);
	return -ENODEV;
}

static unsigned long check_vendor_combination_bug(struct pci_dev *pdev)
{
	if (pdev->vendor == 0x144d && pdev->device == 0xa802) {
		/*
		 * Several Samsung devices seem to drop off the PCIe bus
		 * randomly when APST is on and uses the deepest sleep state.
		 * This has been observed on a Samsung "SM951 NVMe SAMSUNG
		 * 256GB", a "PM951 NVMe SAMSUNG 512GB", and a "Samsung SSD
		 * 950 PRO 256GB", but it seems to be restricted to two Dell
		 * laptops.
		 */
		if (dmi_match(DMI_SYS_VENDOR, "Dell Inc.") &&
		    (dmi_match(DMI_PRODUCT_NAME, "XPS 15 9550") ||
		     dmi_match(DMI_PRODUCT_NAME, "Precision 5510")))
			return NVME_QUIRK_NO_DEEPEST_PS;
	} else if (pdev->vendor == 0x144d && pdev->device == 0xa804) {
		/*
		 * Samsung SSD 960 EVO drops off the PCIe bus after system
		 * suspend on a Ryzen board, ASUS PRIME B350M-A, as well as
		 * within few minutes after bootup on a Coffee Lake board -
		 * ASUS PRIME Z370-A
		 */
		if (dmi_match(DMI_BOARD_VENDOR, "ASUSTeK COMPUTER INC.") &&
		    (dmi_match(DMI_BOARD_NAME, "PRIME B350M-A") ||
		     dmi_match(DMI_BOARD_NAME, "PRIME Z370-A")))
			return NVME_QUIRK_NO_APST;
	} else if ((pdev->vendor == 0x144d && (pdev->device == 0xa801 ||
		    pdev->device == 0xa808 || pdev->device == 0xa809)) ||
		   (pdev->vendor == 0x1e0f && pdev->device == 0x0001)) {
		/*
		 * Forcing to use host managed nvme power settings for
		 * lowest idle power with quick resume latency on
		 * Samsung and Toshiba SSDs based on suspend behavior
		 * on Coffee Lake board for LENOVO C640
		 */
		if ((dmi_match(DMI_BOARD_VENDOR, "LENOVO")) &&
		     dmi_match(DMI_BOARD_NAME, "LNVNB161216"))
			return NVME_QUIRK_SIMPLE_SUSPEND;
	}

	return 0;
}

static void nvme_async_probe(void *data, async_cookie_t cookie)
{
	struct nvme_dev *dev = data;

	flush_work(&dev->ctrl.reset_work);
	flush_work(&dev->ctrl.scan_work);
	nvme_put_ctrl(&dev->ctrl);
}

static int nvme_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
	int node, result = -ENOMEM;
	struct nvme_dev *dev;
	struct ctrl *ctrl;
	unsigned long quirks = id->driver_data;
	size_t alloc_size;

	/*
	 * Opt-in probe gate (PORTING.md section 7.3.1 #7 / segment 6a).
	 *
	 * pci_register_driver() triggers .probe() for *every* matching PCI
	 * device on the bus, including NVMes the user never asked snvme to
	 * touch.  Without this gate, the first SNVM_DEVICE_BIND ioctl would
	 * hijack every unbound NVMe on the host -- snvm_rebind_driver()
	 * detaches only the single target BDF, leaving the rest with no
	 * recovery path.
	 *
	 * Require an explicit per-BDF opt-in: a struct ctrl must already
	 * have been registered via SNVM_CHRDEV_CREATE (segment 4) before
	 * probe does anything.  If no ctrl record exists, return -ENODEV
	 * so the PCI core falls through to the next matching driver
	 * (typically in-tree nvme), leaving the device alone.
	 *
	 * This MUST stay at the top of probe().  Do NOT collapse it into
	 * the later `if (ctrl && ...)` check below: that one is the
	 * use_sreg pre-condition and is independent.
	 */
	ctrl = ctrl_find_by_pci_dev(&ctrl_list, pdev);
	if (ctrl == NULL) {
		dev_info(&pdev->dev,
			 "no ctrl registered for this BDF, skipping probe "
			 "(user must call SNVM_CHRDEV_CREATE first)\n");
		return -ENODEV;
	}

	node = dev_to_node(&pdev->dev);
	if (node == NUMA_NO_NODE)
		set_dev_node(&pdev->dev, first_memory_node);

	dev = kzalloc_node(sizeof(*dev), GFP_KERNEL, node);
	if (!dev)
		return -ENOMEM;

	/*
	 * Apply caller-supplied tunables from NVM_SET_IOQ_NUM.  These
	 * override the module-parameter defaults (write_queues /
	 * poll_queues) on a per-controller basis, and let the user pin
	 * the kernel-side IO-queue request below num_possible_cpus()
	 * when the controller cannot grant that many MSI-X vectors.
	 *
	 * setup.valid is the "ioctl was actually called" sentinel;
	 * unconditionally reading these fields when setup.valid == 0
	 * is harmless because the struct is kmalloc-zeroed inside
	 * ctrl_get(), but the explicit gate documents intent and
	 * prevents a future zero-meaning change from silently
	 * clamping cap_kernel_ioq to 0 = "no override".
	 */
	if (ctrl && ctrl->setup.valid) {
		if (ctrl->setup.nr_write)
			dev->nr_write_queues = ctrl->setup.nr_write;
		if (ctrl->setup.nr_poll)
			dev->nr_poll_queues  = ctrl->setup.nr_poll;
		dev->cap_kernel_ioq = ctrl->setup.cap_kernel_ioq;
	} else {
		dev->nr_write_queues = write_queues;
		dev->nr_poll_queues  = poll_queues;
		dev->cap_kernel_ioq  = 0;
	}

	dev->nr_allocated_queues = nvme_max_io_queues(dev) + 1;
	dev->queues = kcalloc_node(dev->nr_allocated_queues,
			sizeof(struct nvme_queue), GFP_KERNEL, node);
	if (!dev->queues)
		goto free;

	dev->dev = get_device(&pdev->dev);
	pci_set_drvdata(pdev, dev);

	result = nvme_dev_map(dev);
	if (result)
		goto put_pci;

	INIT_WORK(&dev->ctrl.reset_work, nvme_reset_work);
	INIT_WORK(&dev->remove_work, nvme_remove_dead_ctrl_work);
	mutex_init(&dev->shutdown_lock);

	result = nvme_setup_prp_pools(dev);
	if (result)
		goto unmap;

	quirks |= check_vendor_combination_bug(pdev);

	/*
	 * Double check that our mempool alloc size will cover the biggest
	 * command we support.
	 */
	alloc_size = nvme_pci_iod_alloc_size(dev, NVME_MAX_KB_SZ,
						NVME_MAX_SEGS, true);
	WARN_ON_ONCE(alloc_size > PAGE_SIZE);

	dev->iod_mempool = mempool_create_node(1, mempool_kmalloc,
						mempool_kfree,
						(void *) alloc_size,
						GFP_KERNEL, node);
	if (!dev->iod_mempool) {
		result = -ENOMEM;
		goto release_pools;
	}

	result = snvme_init_ctrl(&dev->ctrl, &pdev->dev, &nvme_pci_ctrl_ops,
			quirks);
	if (result)
		goto release_mempool;

	dev_info(dev->ctrl.device, "pci function %s\n", dev_name(&pdev->dev));

	snvme_reset_ctrl(&dev->ctrl);
	async_schedule(nvme_async_probe, dev);

	return 0;

 release_mempool:
	mempool_destroy(dev->iod_mempool);
 release_pools:
	nvme_release_prp_pools(dev);
 unmap:
	nvme_dev_unmap(dev);
 put_pci:
	put_device(dev->dev);
 free:
	kfree(dev->queues);
	kfree(dev);
	return result;
}

static void nvme_reset_prepare(struct pci_dev *pdev)
{
	struct nvme_dev *dev = pci_get_drvdata(pdev);

	/*
	 * We don't need to check the return value from waiting for the reset
	 * state as pci_dev device lock is held, making it impossible to race
	 * with ->remove().
	 */
	nvme_disable_prepare_reset(dev, false);
	snvme_sync_queues(&dev->ctrl);
}

static void nvme_reset_done(struct pci_dev *pdev)
{
	struct nvme_dev *dev = pci_get_drvdata(pdev);

	if (!snvme_try_sched_reset(&dev->ctrl))
		flush_work(&dev->ctrl.reset_work);
}

static void nvme_shutdown(struct pci_dev *pdev)
{
	struct nvme_dev *dev = pci_get_drvdata(pdev);
	nvme_disable_prepare_reset(dev, true);
}

/*
 * The driver's remove may be called on a device in a partially initialized
 * state. This function must not have any dependencies on the device state in
 * order to proceed.
 */
static void nvme_remove(struct pci_dev *pdev)
{
	struct nvme_dev *dev = pci_get_drvdata(pdev);

	/* output the hotplug nvme drive letter and BDF */
	dev_info(dev->ctrl.device, "remove pci function %s\n", dev_name(&pdev->dev));

	snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_DELETING);
	pci_set_drvdata(pdev, NULL);

	if (!pci_device_is_present(pdev)) {
		snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_DEAD);
		nvme_dev_disable(dev, true);
	}

	flush_work(&dev->ctrl.reset_work);
	snvme_stop_ctrl(&dev->ctrl);
	snvme_remove_namespaces(&dev->ctrl);
	nvme_dev_disable(dev, true);
	nvme_release_cmb(dev);
	nvme_free_host_mem(dev);
	nvme_dev_remove_admin(dev);
	nvme_free_queues(dev, 0);
	snvme_uninit_ctrl(&dev->ctrl);
	nvme_release_prp_pools(dev);
	nvme_dev_unmap(dev);
	nvme_put_ctrl(&dev->ctrl);
}

#ifdef CONFIG_PM_SLEEP
static int nvme_get_power_state(struct nvme_ctrl *ctrl, u32 *ps)
{
	return snvme_get_features(ctrl, NVME_FEAT_POWER_MGMT, 0, NULL, 0, ps);
}

static int nvme_set_power_state(struct nvme_ctrl *ctrl, u32 ps)
{
	return snvme_set_features(ctrl, NVME_FEAT_POWER_MGMT, ps, NULL, 0, NULL);
}

static int nvme_resume(struct device *dev)
{
	struct nvme_dev *ndev = pci_get_drvdata(to_pci_dev(dev));
	struct nvme_ctrl *ctrl = &ndev->ctrl;

	if (ndev->last_ps == U32_MAX ||
	    nvme_set_power_state(ctrl, ndev->last_ps) != 0)
		return snvme_try_sched_reset(&ndev->ctrl);
	return 0;
}

static int nvme_suspend(struct device *dev)
{
	struct pci_dev *pdev = to_pci_dev(dev);
	struct nvme_dev *ndev = pci_get_drvdata(pdev);
	struct nvme_ctrl *ctrl = &ndev->ctrl;
	int ret = -EBUSY;

	ndev->last_ps = U32_MAX;

	/*
	 * The platform does not remove power for a kernel managed suspend so
	 * use host managed nvme power settings for lowest idle power if
	 * possible. This should have quicker resume latency than a full device
	 * shutdown.  But if the firmware is involved after the suspend or the
	 * device does not support any non-default power states, shut down the
	 * device fully.
	 *
	 * If ASPM is not enabled for the device, shut down the device and allow
	 * the PCI bus layer to put it into D3 in order to take the PCIe link
	 * down, so as to allow the platform to achieve its minimum low-power
	 * state (which may not be possible if the link is up).
	 *
	 * If a host memory buffer is enabled, shut down the device as the NVMe
	 * specification allows the device to access the host memory buffer in
	 * host DRAM from all power states, but hosts will fail access to DRAM
	 * during S3.
	 */
	if (pm_suspend_via_firmware() || !ctrl->npss ||
	    !pcie_aspm_enabled(pdev) ||
	    ndev->nr_host_mem_descs ||
	    (ndev->ctrl.quirks & NVME_QUIRK_SIMPLE_SUSPEND))
		return nvme_disable_prepare_reset(ndev, true);

	snvme_start_freeze(ctrl);
	snvme_wait_freeze(ctrl);
	snvme_sync_queues(ctrl);

	if (ctrl->state != NVME_CTRL_LIVE)
		goto unfreeze;

	ret = nvme_get_power_state(ctrl, &ndev->last_ps);
	if (ret < 0)
		goto unfreeze;

	/*
	 * A saved state prevents pci pm from generically controlling the
	 * device's power. If we're using protocol specific settings, we don't
	 * want pci interfering.
	 */
	pci_save_state(pdev);

	ret = nvme_set_power_state(ctrl, ctrl->npss);
	if (ret < 0)
		goto unfreeze;

	if (ret) {
		/* discard the saved state */
		pci_load_saved_state(pdev, NULL);

		/*
		 * Clearing npss forces a controller reset on resume. The
		 * correct value will be resdicovered then.
		 */
		ret = nvme_disable_prepare_reset(ndev, true);
		ctrl->npss = 0;
	}
unfreeze:
	snvme_unfreeze(ctrl);
	return ret;
}

static int nvme_simple_suspend(struct device *dev)
{
	struct nvme_dev *ndev = pci_get_drvdata(to_pci_dev(dev));
	return nvme_disable_prepare_reset(ndev, true);
}

static int nvme_simple_resume(struct device *dev)
{
	struct pci_dev *pdev = to_pci_dev(dev);
	struct nvme_dev *ndev = pci_get_drvdata(pdev);

	return snvme_try_sched_reset(&ndev->ctrl);
}

static const struct dev_pm_ops nvme_dev_pm_ops = {
	.suspend	= nvme_suspend,
	.resume		= nvme_resume,
	.freeze		= nvme_simple_suspend,
	.thaw		= nvme_simple_resume,
	.poweroff	= nvme_simple_suspend,
	.restore	= nvme_simple_resume,
};
#endif /* CONFIG_PM_SLEEP */

static pci_ers_result_t nvme_error_detected(struct pci_dev *pdev,
						pci_channel_state_t state)
{
	struct nvme_dev *dev = pci_get_drvdata(pdev);

	/*
	 * A frozen channel requires a reset. When detected, this method will
	 * shutdown the controller to quiesce. The controller will be restarted
	 * after the slot reset through driver's slot_reset callback.
	 */
	switch (state) {
	case pci_channel_io_normal:
		return PCI_ERS_RESULT_CAN_RECOVER;
	case pci_channel_io_frozen:
		dev_warn(dev->ctrl.device,
			"frozen state error detected, reset controller\n");
		nvme_dev_disable(dev, false);
		return PCI_ERS_RESULT_NEED_RESET;
	case pci_channel_io_perm_failure:
		dev_warn(dev->ctrl.device,
			"failure state error detected, request disconnect\n");
		return PCI_ERS_RESULT_DISCONNECT;
	}
	return PCI_ERS_RESULT_NEED_RESET;
}

static pci_ers_result_t nvme_slot_reset(struct pci_dev *pdev)
{
	struct nvme_dev *dev = pci_get_drvdata(pdev);

	dev_info(dev->ctrl.device, "restart after slot reset\n");
	pci_restore_state(pdev);
	snvme_reset_ctrl(&dev->ctrl);
	return PCI_ERS_RESULT_RECOVERED;
}

static void nvme_error_resume(struct pci_dev *pdev)
{
	struct nvme_dev *dev = pci_get_drvdata(pdev);

	flush_work(&dev->ctrl.reset_work);
}

static const struct pci_error_handlers nvme_err_handler = {
	.error_detected	= nvme_error_detected,
	.slot_reset	= nvme_slot_reset,
	.resume		= nvme_error_resume,
	.reset_prepare	= nvme_reset_prepare,
	.reset_done	= nvme_reset_done,
};

static const struct pci_device_id nvme_id_table[] = {
	{ PCI_VDEVICE(INTEL, 0x0953),
		.driver_data = NVME_QUIRK_STRIPE_SIZE |
				NVME_QUIRK_DEALLOCATE_ZEROES, },
	{ PCI_VDEVICE(INTEL, 0x0a53),
		.driver_data = NVME_QUIRK_STRIPE_SIZE |
				NVME_QUIRK_DEALLOCATE_ZEROES, },
	{ PCI_VDEVICE(INTEL, 0x0a54),
		.driver_data = NVME_QUIRK_STRIPE_SIZE |
				NVME_QUIRK_DEALLOCATE_ZEROES, },
	{ PCI_VDEVICE(INTEL, 0x0a55),
		.driver_data = NVME_QUIRK_STRIPE_SIZE |
				NVME_QUIRK_DEALLOCATE_ZEROES, },
	{ PCI_VDEVICE(INTEL, 0xf1a5),	/* Intel 600P/P3100 */
		.driver_data = NVME_QUIRK_NO_DEEPEST_PS |
				NVME_QUIRK_MEDIUM_PRIO_SQ |
				NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_VDEVICE(INTEL, 0xf1a6),	/* Intel 760p/Pro 7600p */
		.driver_data = NVME_QUIRK_IGNORE_DEV_SUBNQN, },
	{ PCI_VDEVICE(INTEL, 0x5845),	/* Qemu emulated controller */
		.driver_data = NVME_QUIRK_IDENTIFY_CNS |
				NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x126f, 0x2263),	/* Silicon Motion unidentified */
		.driver_data = NVME_QUIRK_NO_NS_DESC_LIST, },
	{ PCI_DEVICE(0x1bb1, 0x0100),   /* Seagate Nytro Flash Storage */
		.driver_data = NVME_QUIRK_DELAY_BEFORE_CHK_RDY |
				NVME_QUIRK_NO_NS_DESC_LIST, },
	{ PCI_DEVICE(0x1c58, 0x0003),	/* HGST adapter */
		.driver_data = NVME_QUIRK_DELAY_BEFORE_CHK_RDY, },
	{ PCI_DEVICE(0x1c58, 0x0023),	/* WDC SN200 adapter */
		.driver_data = NVME_QUIRK_DELAY_BEFORE_CHK_RDY, },
	{ PCI_DEVICE(0x1c5f, 0x0540),	/* Memblaze Pblaze4 adapter */
		.driver_data = NVME_QUIRK_DELAY_BEFORE_CHK_RDY, },
	{ PCI_DEVICE(0x144d, 0xa821),   /* Samsung PM1725 */
		.driver_data = NVME_QUIRK_DELAY_BEFORE_CHK_RDY, },
	{ PCI_DEVICE(0x144d, 0xa822),   /* Samsung PM1725a */
		.driver_data = NVME_QUIRK_DELAY_BEFORE_CHK_RDY |
				NVME_QUIRK_DISABLE_WRITE_ZEROES|
				NVME_QUIRK_IGNORE_DEV_SUBNQN, },
	{ PCI_DEVICE(0x1987, 0x5016),	/* Phison E16 */
		.driver_data = NVME_QUIRK_IGNORE_DEV_SUBNQN, },
	{ PCI_DEVICE(0x1b4b, 0x1092),	/* Lexar 256 GB SSD */
		.driver_data = NVME_QUIRK_NO_NS_DESC_LIST |
				NVME_QUIRK_IGNORE_DEV_SUBNQN, },
	{ PCI_DEVICE(0x1d1d, 0x1f1f),	/* LighNVM qemu device */
		.driver_data = NVME_QUIRK_LIGHTNVM, },
	{ PCI_DEVICE(0x1d1d, 0x2807),	/* CNEX WL */
		.driver_data = NVME_QUIRK_LIGHTNVM, },
	{ PCI_DEVICE(0x1d1d, 0x2601),	/* CNEX Granby */
		.driver_data = NVME_QUIRK_LIGHTNVM, },
	{ PCI_DEVICE(0x10ec, 0x5762),   /* ADATA SX6000LNP */
		.driver_data = NVME_QUIRK_IGNORE_DEV_SUBNQN, },
	{ PCI_DEVICE(0x1cc1, 0x8201),   /* ADATA SX8200PNP 512GB */
		.driver_data = NVME_QUIRK_NO_DEEPEST_PS |
				NVME_QUIRK_IGNORE_DEV_SUBNQN, },
	{ PCI_DEVICE(0x1c5c, 0x1504),   /* SK Hynix PC400 */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x2646, 0x2263),   /* KINGSTON A2000 NVMe SSD  */
		.driver_data = NVME_QUIRK_NO_DEEPEST_PS, },
	{ PCI_DEVICE(PCI_VENDOR_ID_APPLE, 0x2001),
		.driver_data = NVME_QUIRK_SINGLE_VECTOR },
	{ PCI_DEVICE(PCI_VENDOR_ID_APPLE, 0x2003) },
	{ PCI_DEVICE(PCI_VENDOR_ID_APPLE, 0x2005),
		.driver_data = NVME_QUIRK_SINGLE_VECTOR |
				NVME_QUIRK_128_BYTES_SQES |
				NVME_QUIRK_SHARED_TAGS },

	{ PCI_DEVICE_CLASS(PCI_CLASS_STORAGE_EXPRESS, 0xffffff) },
	{ 0, }
};
MODULE_DEVICE_TABLE(pci, nvme_id_table);

static struct pci_driver snvme_driver = {
	.name		= PCI_DRIVER_NAME,
	.id_table	= nvme_id_table,
	.probe		= nvme_probe,
	.remove		= nvme_remove,
	.shutdown	= nvme_shutdown,
#ifdef CONFIG_PM_SLEEP
	.driver		= {
		.pm	= &nvme_dev_pm_ops,
	},
#endif
	.sriov_configure = pci_sriov_configure_simple,
	.err_handler	= &nvme_err_handler,
};

/* ------------------------------------------------------------------ *
 *  snvme: /dev/snvm_control char device
 *
 *  /dev/snvm_control is the singleton control entry point that libnvm
 *  opens to drive snvme as a whole (device BIND/UNBIND, ssnvme chrdev
 *  creation, PCI-distance queries).  Its ioctl handler -- snvm_ioctl
 *  below -- is a stub at this porting stage; segment 3 replaces the
 *  stub with the real SNVM_* dispatch.
 *
 *  Ported from snvme-5.15.0/pci.c lines ~4242-4349.  The body of
 *  snvm_cdev_init / snvm_cdev_release is identical across kernel
 *  baselines because it only touches baseline-agnostic cdev/class APIs
 *  that exist unchanged in 5.4.
 * ------------------------------------------------------------------ */

/*
 * Forward declaration of the /dev/snvm_control ioctl dispatcher.  The
 * real implementation lives further down (segment 3 of the staged
 * port) so it can call helpers (snvm_rebind_driver, snvm_chrdev_helper,
 * etc.) defined in that block; declaring it here lets snvm_fops below
 * take its address even though its body is not yet visible.
 */
static long snvm_ioctl(struct file *file, unsigned int cmd, unsigned long arg);

static const struct file_operations snvm_fops = {
	.owner		= THIS_MODULE,
	.unlocked_ioctl	= snvm_ioctl,
};

/*
 * devnode callback: make /dev/snvm_control and /dev/ssnvme* readable
 * and writable to non-root users (snvme requires a PID-scoped libnvm
 * client; filesystem perms are not the security boundary here).
 */
static char *get_snvme_mode(struct device *dev, umode_t *mode)
{
	if (mode)
		*mode = 0666;
	return NULL;
}

/*
 * snvm_cdev_init: run once at module load.  Sets up the shared
 * /dev/snvm_control cdev and reserves a pool of minor numbers for the
 * per-controller /dev/ssnvme%d devices that segment 4 will hand out.
 *
 * Layout choice mirrored from snvme-5.15.0/pci.c:
 *   - alloc_chrdev_region(&dev_first, 0, max_num_ctrls, DRIVER_NAME)
 *     reserves minors [0, max_num_ctrls) for ssnvme%d.
 *   - snvm_devno is placed at minor = max_num_ctrls (one past the
 *     reserved range) so that /dev/snvm_control cannot collide with
 *     any ssnvme%d minor.
 *
 * Failure semantics:
 *
 *   On any failure, all sub-resources allocated up to that point
 *   are released via the goto-chain below.  Two failure modes
 *   leave behind state that *cannot* be cleaned up by the module
 *   itself, and the operator must intervene before the next
 *   insmod attempt:
 *
 *     (a) class_create returns -EEXIST.  A previous module
 *         instance died after class_create succeeded but before
 *         class_destroy ran (e.g. partial init failure on an
 *         earlier load, or a forced unload that did not run
 *         module_exit).  /sys/class/"libsnvm helper"/ persists.
 *
 *     (b) device_create returns -EEXIST.  Same root cause as
 *         (a) but the residue is the /dev/snvm_control entry
 *         (sysfs + chrdev region).
 *
 *   Operator fix for both: reboot.  Removing the sysfs node by
 *   hand is fragile -- a partially-initialised class can hold
 *   internal kobject refs that surface as a BUG when re-inserted.
 *   We deliberately do NOT try to "rescue" by reusing the
 *   pre-existing kobject; sysfs ABI does not give us a clean way
 *   to do that, and silently adopting external state would mask
 *   the underlying instability.  Loud failure with explicit
 *   reboot guidance is the safer default.
 */
static int snvm_cdev_init(void)
{
	int ret;
	struct device *device;

	mutex_init(&snvm_control_lock);

	dev_class = class_create(THIS_MODULE, DRIVER_NAME);
	if (IS_ERR(dev_class)) {
		ret = PTR_ERR(dev_class);
		pr_err("class_create(\"%s\") failed: %d\n",
		       DRIVER_NAME, ret);
		if (ret == -EEXIST)
			pr_err("stale sysfs node /sys/class/\"%s\"/ "
			       "from a previous module instance.  "
			       "Reboot before retrying insmod.\n",
			       DRIVER_NAME);
		mutex_destroy(&snvm_control_lock);
		dev_class = NULL;
		return ret;
	}
	dev_class->devnode = get_snvme_mode;

	ret = alloc_chrdev_region(&dev_first, 0, max_num_ctrls, DRIVER_NAME);
	if (ret < 0) {
		pr_err("alloc_chrdev_region(%d minors) failed: %d\n",
		       max_num_ctrls, ret);
		goto destroy_class;
	}

	snvm_devno = MKDEV(MAJOR(dev_first), max_num_ctrls);
	cdev_init(&snvm_cdev, &snvm_fops);
	snvm_cdev.owner = THIS_MODULE;
	ret = cdev_add(&snvm_cdev, snvm_devno, 1);
	if (ret < 0) {
		pr_err("cdev_add failed: %d\n", ret);
		goto err_unregister_chrdev;
	}

	device = device_create(dev_class, NULL, snvm_devno, NULL, "snvm_control");
	if (IS_ERR(device)) {
		ret = PTR_ERR(device);
		pr_err("device_create(/dev/snvm_control) failed: %d\n",
		       ret);
		if (ret == -EEXIST)
			pr_err("stale /dev/snvm_control or sysfs "
			       "device node from a previous module "
			       "instance.  Reboot before retrying "
			       "insmod.\n");
		goto destroy_cdev;
	}
	return 0;

destroy_cdev:
	cdev_del(&snvm_cdev);
err_unregister_chrdev:
	unregister_chrdev_region(dev_first, max_num_ctrls);
destroy_class:
	class_destroy(dev_class);
	dev_class = NULL;
	mutex_destroy(&snvm_control_lock);
	return ret;
}

static void snvm_cdev_release(void)
{
	device_destroy(dev_class, snvm_devno);
	cdev_del(&snvm_cdev);
	unregister_chrdev_region(dev_first, max_num_ctrls);
	class_destroy(dev_class);
	mutex_destroy(&snvm_control_lock);
	ida_destroy(&snvm_chrdev_minor_ida);
	/*
	 * Drain the queue-group id namespace.  All per-fd group
	 * descriptors should already be gone by the time we get
	 * here (every owning fd has been closed before module exit,
	 * because module exit is rmmod-only and the chrdev cdev is
	 * about to be torn down).  ida_destroy is safe to call on
	 * an empty IDA, but a non-empty one would leak ints; in
	 * that pathological case the WARN inside ida_destroy will
	 * fire and we'll catch it in dmesg.
	 */
	ida_destroy(&snvm_queue_group_ida);
	pr_info("/dev/snvm_control released\n");
}

/* ------------------------------------------------------------------ *
 *  snvme: lazy PCI driver registration + SNVM_* ioctl dispatch
 *
 *  Design choice carried over from snvme-5.15.0: we do NOT call
 *  pci_register_driver() at module load time.  If we did, snvme would
 *  immediately bind every PCIe NVMe device on the system and yank them
 *  away from the in-tree nvme driver, breaking coexistence.  Instead,
 *  /dev/snvm_control's SNVM_DEVICE_BIND ioctl drives a *lazy* register
 *  that runs only the first time a userspace client (libnvm) asks
 *  snvme to take over a specific BDF; the snvm_registered flag from
 *  segment 1 guards against double-register.
 * ------------------------------------------------------------------ */

/*
 * Wrapper around pci_register_driver(): identical to upstream nvme's
 * nvme_init() body but as a callable from the lazy path.
 */
static int snvm_register_driver(void)
{
	BUILD_BUG_ON(sizeof(struct nvme_create_cq) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_create_sq) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_delete_queue) != 64);
	BUILD_BUG_ON(IRQ_AFFINITY_MAX_SETS < 2);
	return pci_register_driver(&snvme_driver);
}

static void snvm_unregister_driver(void)
{
	pci_unregister_driver(&snvme_driver);
	flush_workqueue(s_nvme_wq);
}

/*
 * register_driver: idempotent lazy registration entry.  Called from
 * snvm_rebind_driver() before the first device_attach.  Multiple
 * concurrent SNVM_DEVICE_BIND callers race here -- snvm_control_lock
 * + snvm_registered serialise the actual pci_register_driver() to a
 * single winner.
 */
static int register_driver(void)
{
	struct device_driver *dev_drv;
	int ret = 0;

	mutex_lock(&snvm_control_lock);
	dev_drv = driver_find(PCI_DRIVER_NAME, &pci_bus_type);
	if (!dev_drv && !snvm_registered) {
		ret = snvm_register_driver();
		if (ret) {
			pr_err("snvm_register_driver failed: %d\n", ret);
		} else {
			snvm_registered = 1;
		}
	}
	mutex_unlock(&snvm_control_lock);
	return ret;
}

/*
 * SNVM_DEVICE_BIND implementation: detach whatever driver currently
 * owns the PCI device addressed by `dev_addr` (typically the in-tree
 * nvme driver), lazily register snvme on the PCI bus if not already
 * registered, then force-attach the device to snvme.
 *
 * 5.4 vs 5.15 API note:
 *   snvme-5.15.0 calls device_driver_attach(&snvme_driver.driver,
 *   &pdev->dev) -- a 5.5+ helper that force-binds a specific dev to a
 *   specific driver in one device-locked critical section.
 *
 *   5.4 has neither device_driver_attach nor driver_probe_device on
 *   its exported surface.  The earlier 5.4 port used device_attach(),
 *   which is NOT equivalent: device_attach() is bus->match() based and
 *   picks the **first** registered matching driver.  On a system where
 *   the in-tree nvme.ko was loaded before snvme.ko (the universal
 *   case), the in-tree driver is the first match, so device_attach
 *   silently rebinds the device to nvme.ko -- the dmesg signature is:
 *
 *      snvme: binding nvme device to snvme: pci 0:8:0.0
 *      nvme nvme0: pci function 0000:08:00.0          <-- in-tree!
 *      nvme nvme0: 135/0/0 default/read/poll queues   <-- 3-tuple, not 4
 *
 *   The subsequent SNVM_DEVICE_UNBIND then fails with -EFAULT
 *   ("device's driver is not snvme").  TencentOS 5.4.241 reproduces
 *   this on every "GPU --bind" run that follows a previous "GPU
 *   no-bind" run (the no-bind run leaves the BDF with dev->driver ==
 *   NULL because device_release_driver was called but autoprobe
 *   didn't race in time, so the next BIND finds an unattached BDF
 *   and falls through to the device_attach trap).
 *
 *   5.4-compatible substitute: driver_attach(&snvme_driver.driver).
 *   driver_attach iterates the bus's device list and, for each device
 *   whose ->driver is NULL and whose id_table matches, runs that
 *   specific driver's probe.  Because nvme_probe() has an explicit
 *   `ctrl_find_by_pci_dev(&ctrl_list, pdev) != NULL` gate at the top
 *   (see segment 6a), it short-circuits to -ENODEV for every NVMe the
 *   user did not pre-register via SNVM_CHRDEV_CREATE -- only the BDF
 *   the user just declared intent for actually goes through probe.
 *
 *   Race with udev autoprobe: when device_release_driver(pdev) returns,
 *   the device is briefly unbound and the bus emits KOBJ_CHANGE.  udev
 *   may then echo the BDF into /sys/bus/pci/drivers/nvme/bind via its
 *   autoprobe rules, beating us to driver_attach.  We close that
 *   window by a bounded retry loop: after each driver_attach, if
 *   pdev->dev.driver isn't snvme, release it again and retry.  Three
 *   attempts is enough in practice; if udev wins three times in a row
 *   the host has a misconfigured autoprobe rule and -EBUSY is the
 *   honest answer.
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
	 * confuse snvme's own pci_enable_device on rebind.
	 */
	dev_drv = pdev->dev.driver;
	if (dev_drv && dev_drv->name) {
		if (pci_is_enabled(pdev)) {
			pci_disable_device(pdev);
			pr_info("(%s): disable device for bind new driver\n",
				__func__);
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

	/*
	 * Force-attach to snvme, fighting any udev autoprobe race.  See
	 * the function header above for the rationale of driver_attach
	 * over device_attach.
	 */
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
		ret = driver_attach(&snvme_driver.driver);
		if (ret) {
			pr_err("(%s): driver_attach failed: %d\n",
			       __func__, ret);
			break;
		}
		/* driver_attach returns 0 on success regardless of
		 * whether ANY device was probed; check pdev->dev.driver
		 * to decide if we won the race on the target BDF. */
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

/*
 * SNVM_DEVICE_UNBIND implementation: simply asks the bus to release
 * the snvme binding on the addressed PCI device.  After this returns,
 * udev / autoprobe may immediately re-bind the device to the in-tree
 * nvme driver again; that is the intended hand-back behaviour.
 */
static int snvm_unbind_driver(struct pci_device_addr dev_addr)
{
	struct device_driver *dev_drv;
	struct pci_dev *pdev;

	pdev = TO_PCI_DEV(dev_addr);
	if (!pdev) {
		pr_err("(%s): pci_get_domain_bus_and_slot failed\n", __func__);
		return -EFAULT;
	}

	dev_drv = pdev->dev.driver;
	if (!dev_drv) {
		/* Not an snvme error -- the device is already unbound.
		 * Honest answer is -ENODEV (no driver to unbind), not
		 * -EFAULT (which means "bad address" and misleads
		 * userspace into thinking the dev_addr arg was wrong).
		 */
		pr_err("(%s): device has no driver to unbind\n", __func__);
		pci_dev_put(pdev);
		return -ENODEV;
	}

	if (!dev_drv->name || strcmp(dev_drv->name, PCI_DRIVER_NAME) != 0) {
		/* Wrong owner.  This is the dmesg signature of the
		 * "GPU --bind after GPU no-bind" race fixed in
		 * snvm_rebind_driver above (see that function's header):
		 * if you still see this AFTER applying the
		 * driver_attach + retry fix, it means udev rebound the
		 * BDF to the in-tree nvme driver between BIND and
		 * UNBIND -- check /sys/bus/pci/drivers/nvme/bind logs.
		 *
		 * Return -EINVAL (caller's request is inconsistent with
		 * current kernel state), not -EFAULT.
		 */
		pr_err("(%s): device's driver is '%s', not '%s' -- refusing to unbind\n",
		       __func__, dev_drv->name, PCI_DRIVER_NAME);
		pci_dev_put(pdev);
		return -EINVAL;
	}

	if (pci_is_enabled(pdev)) {
		pci_disable_device(pdev);
		pr_info("(%s): disabled device prior to unbind\n", __func__);
	}
	pr_info("unbinding device from driver %s\n", dev_drv->name);
	device_release_driver(&pdev->dev);
	pci_dev_put(pdev);
	return 0;
}

/* ------------------------------------------------------------------ *
 *  snvme: /dev/ssnvme<minor> per-controller char device
 *
 *  Each NVMe device that libnvm has asked snvme to manage gets its own
 *  /dev/ssnvme<minor> entry, with two roles:
 *
 *    - mmap(fd) returns a userspace-writable mapping of the NVMe BAR0
 *      doorbell region; libnvm uses this for direct doorbell rings.
 *
 *    - ioctl(fd, NVM_*) implements the per-controller mapping /
 *      queue-share commands listed in libnvm/include/ioctl.h.  The
 *      real dispatcher is ported in segment 5; until then,
 *      snvm_dev_map_ioctl below returns -ENOTTY.
 *
 *  Ported from snvme-5.15.0/pci.c lines ~3925-3982.  All APIs used
 *  here (ida_simple_*, vm_iomap_memory, pci_resource_*, pgprot_*) are
 *  present in 5.4 with the same signatures, so this block is a
 *  mechanical copy.
 * ------------------------------------------------------------------ */

/*
 * /dev/ssnvme<minor> ioctl dispatcher.  Implements the libnvm
 * NVM_* ABI (see backends/local/nvme/libnvm/include/ioctl.h):
 *   NVM_MAP_HOST_MEMORY            pin user pages, return DMA addrs
 *   NVM_MAP_DEVICE_MEMORY          pin GPU pages (nvfs), return DMA addrs
 *   NVM_MAP_DEVICE_QUEUE_MEMORY    same, but for an IO-queue SQ/CQ/DB
 *   NVM_UNMAP_HOST_MEMORY          drop the host pin
 *   NVM_UNMAP_DEVICE_MEMORY        drop the GPU pin
 *   NVM_UNMAP_DEVICE_QUEUE_MEMORY  drop the GPU IO-queue pin
 *   NVM_SET_IOQ_NUM                userspace declares how many IOQs it wants
 *   NVM_SET_SHARE_REG              gate flag: register user-DMA on next probe
 *   NVM_GET_DEV_INFO               read disk_name / lba_shift / max_hw_sectors
 *   NVM_CLEAR_IOQ_NUM              reset accounting counters
 *
 * 5.4 vs 5.15 notes:
 *   - All map_*() / unmap_and_release() helpers come from map.c,
 *     which is kernel-version-agnostic.  No 5.15-only API used here.
 *   - NVM_GET_DEV_INFO uses snvme_find_get_ns / snvme_put_ns.  These
 *     are static in upstream nvme-5.4; snvme-5.4/core.c re-exports
 *     them (plain EXPORT_SYMBOL_GPL, since 5.4 has no SYMBOL_NS).
 *   - The "ns->disk->diskseq" printk that snvme-5.15 emits in
 *     NVM_GET_DEV_INFO is dropped: diskseq is a 5.10+ field and is
 *     diagnostic-only, no userspace contract relies on it.
 *   - Bug fix vs snvme-5.15: NVM_UNMAP_HOST_MEMORY now sets ret=0
 *     on the success path (snvme-5.15 left ret uninitialised, an
 *     obvious unrelated bug).  Same applies to NVM_SET_SHARE_REG.
 *   - Bug fix vs snvme-5.15: NVM_MAP_DEVICE_QUEUE_MEMORY now
 *     unmap_and_release()s `map` when ioq_idx<0 returns -EFAULT
 *     (snvme-5.15 leaked the fresh mapping).
 */

/* ------------------------------------------------------------------ *
 *  Per-fd queue-group machinery
 *
 *  These types and helpers are defined here (above
 *  snvm_dev_map_ioctl) so the NVM_CREATE_QUEUE_GROUP /
 *  NVM_DESTROY_QUEUE_GROUP cases can dereference snvm_dev_owner
 *  members directly.  open / release / fops vtable still live
 *  near the bottom of the file with the rest of the chrdev
 *  lifecycle code -- only the type *definitions* need to be in
 *  scope here.
 *
 *  See the rationale comment block above snvm_dev_open below for
 *  the full leak-on-crash story that motivates having a per-fd
 *  owner descriptor at all.
 * ------------------------------------------------------------------ */

struct snvm_dev_owner {
	struct ctrl		*ctrl;
	struct task_struct	*owner;

	/*
	 * Per-fd queue group list (NVM_CREATE_QUEUE_GROUP added them,
	 * NVM_DESTROY_QUEUE_GROUP / fd-close drains them).  Protected
	 * by groups_lock against concurrent ioctl threads on the same
	 * fd; release() runs after all ioctl handlers have returned
	 * (vfs guarantees fput happens after the last ref drops) so
	 * the lock is uncontended there but we still take it for
	 * lockdep cleanliness.
	 *
	 * We don't put the group descriptors in a global list because
	 * cascade-cleanup on fd-close needs only this fd's groups,
	 * and there's no cross-fd sharing of group_id (the IDA owns
	 * the namespace, descriptors are strictly per-fd).
	 */
	struct list_head	groups;       /* head of struct snvm_qgroup */
	struct mutex		groups_lock;  /* serialises group list mutation  */
	unsigned int		nr_groups;    /* current count, for cap check    */

	/*
	 * Per-fd data-buffer maps (B6, NVM_MAP_KIND_DATA).
	 *
	 * Maps registered with map_kind == NVM_MAP_KIND_DATA hang off
	 * THIS list, NOT off any snvm_qgroup.maps list.  That decouples
	 * the data-buffer DMA pool's lifetime from any single queue
	 * group's lifetime, which matches the common usage pattern:
	 *
	 *   open(/dev/ssnvme*)
	 *   NVM_MAP_HOST_MEMORY(kind=DATA, big DMA pool)         <-- once
	 *   loop:
	 *     NVM_CREATE_QUEUE_GROUP
	 *     NVM_MAP_HOST_MEMORY(kind=RING_SQ/RING_CQ, group=g)  <-- per group
	 *     NVM_ADD_USER_QUEUE
	 *     ... IO ...
	 *     NVM_DESTROY_QUEUE_GROUP                             <-- destroys
	 *                                                             rings; data
	 *                                                             pool keeps
	 *                                                             living
	 *   close(fd)                                             <-- finally
	 *                                                             releases the
	 *                                                             data pool
	 *
	 * Locking order is the same as for groups_lock: outermost
	 * lock on the fd, innermost lock everywhere else.  We never
	 * hold both data_maps_lock and groups_lock at the same time
	 * (the two lists hold disjoint maps so cross-list traversal
	 * is not needed).
	 */
	struct list_head	data_maps;
	struct mutex		data_maps_lock;
	unsigned int		nr_data_maps;
};

/*
 * Per-fd queue group descriptor.
 *
 * NB: This is the runtime per-fd container introduced for
 * NVM_CREATE_QUEUE_GROUP / NVM_DESTROY_QUEUE_GROUP.  Do NOT
 * confuse with `struct snvm_queue_group` in ctrl.h, which is the
 * legacy bind-time per-controller GPU partitioning descriptor used
 * by NVM_SET_IOQ_NUM.  The two solve different problems and have
 * different lifetimes; we deliberately use the suffix _qgroup to
 * keep the namespaces distinct.
 *
 * Step B1: this is just a container.  Steps B2-B6 will hang
 * registered maps and user IO queues off the per-group lists.
 *
 * Lifetime:
 *   - allocated by NVM_CREATE_QUEUE_GROUP, group_id assigned via
 *     ida_simple_get(&snvm_queue_group_ida, 1, 0, GFP_KERNEL).
 *   - released by NVM_DESTROY_QUEUE_GROUP or by fd-close cascade.
 *   - The IDA id is freed in destroy_qgroup_locked() AFTER all
 *     child resources are released, to avoid a window where the
 *     same group_id could be observed by two different
 *     descriptors (the IDA recycles aggressively).
 */
struct snvm_qgroup {
	struct list_head	link;       /* into snvm_dev_owner.groups */
	uint32_t		group_id;
	uint32_t		max_queues; /* hardcoded NVM_MAX_QUEUES_PER_GROUP for B1 */

	/*
	 * Per-group registered maps (B2).  Each entry is a struct map
	 * threaded by its group_link member.  Adding a map is done by
	 * NVM_MAP_HOST_MEMORY / NVM_MAP_DEVICE_MEMORY when the
	 * payload's group_id != 0; removing happens via NVM_UNMAP_*
	 * (vaddr lookup) or via destroy_qgroup_locked() during
	 * NVM_DESTROY_QUEUE_GROUP / fd-close cascade.
	 *
	 * Why a separate per-group list (rather than scanning the
	 * global host_list etc.)?  Two reasons:
	 *   1. Cascade cleanup is O(group_maps) instead of
	 *      O(global_maps); the global lists are intentionally
	 *      kept controller-wide because legacy NVM_SET_IOQ_NUM
	 *      paths still walk them.
	 *   2. Cross-group / cross-fd isolation: a map registered
	 *      under group A is not reachable via group B, even if
	 *      they happen to share a vaddr.
	 *
	 * No cap on nr_maps in B2; userspace can register
	 * arbitrarily many buffers (KVCache, page cache, ring
	 * buffers).  Future revisions should hook into RLIMIT_MEMLOCK
	 * or cgroup memory accounting to bound total pinned pages
	 * per fd.
	 */
	struct list_head	maps;
	unsigned int		nr_maps;

	/*
	 * Per-group user IO queues (B3, NVM_ADD_USER_QUEUE).
	 *
	 * Each slot pairs an SQ with a CQ on the controller.  The
	 * NVMe-controller-side state (Create I/O CQ + Create I/O SQ
	 * was issued, qid is committed) is reflected by
	 * queues[i].alive == 1.  destroy_qgroup_locked walks this
	 * array in reverse order issuing Delete I/O SQ + Delete I/O
	 * CQ (NVMe spec ordering: SQ before CQ) and freeing the qid
	 * back to ctrl->user_qid_bitmap.
	 *
	 * Layout choice -- inline array vs list:
	 *   - max_queues is a fixed compile-time cap, so the overhead is
	 *     bounded by NVM_MAX_QUEUES_PER_GROUP per group.
	 *   - inline array means destroy/cascade walk is cache-
	 *     friendly and we don't need yet another list_head
	 *     pair on struct map.
	 *
	 * Concurrency: protected by own->groups_lock at the qgroup
	 * level (the same mutex protecting maps[] and the group
	 * descriptor itself).  ctrl->user_qid_lock is taken inside
	 * own->groups_lock when the bitmap is mutated.
	 */
	struct snvm_user_queue {
		uint16_t qid;
		uint16_t alive;     /* 1 once Create I/O SQ committed */
		uint64_t sq_vaddr;  /* echoed back so destroy / recycle  */
		uint64_t cq_vaddr;  /* can recover the rings if needed   */
	} queues[NVM_MAX_QUEUES_PER_GROUP];
	unsigned int		cur_queues; /* number of slots currently alive */
};

/*
 * B3 user-QID pool management.
 *
 * Lazy-init the bitmap on the first allocation request: the pool
 * range [user_qid_first, user_qid_last] is only known once
 * nvme_probe has set ndev->online_queues / nr_allocated_queues,
 * which happens asynchronously after SNVM_DEVICE_BIND.  Doing it
 * eagerly at bind would require a probe-completion hook the
 * upstream driver doesn't expose; doing it lazily keeps the code
 * out of any reset/error path.
 *
 * Caller MUST hold ctrl->user_qid_lock.  ndev is the result of
 * pci_get_drvdata(ctrl->pdev) and must be non-NULL with admin_q
 * live (i.e. controller is bound and probe finished).
 */
static int snvm_user_qid_pool_init_locked(struct ctrl *ctrl,
					  struct nvme_dev *ndev)
{
	unsigned int first, last, count;
	unsigned long *bm;

	if (ctrl->user_qid_bitmap)
		return 0;     /* already initialised */

	if (!ndev || !ndev->online_queues || !ndev->nr_allocated_queues)
		return -ENODEV;

	/*
	 * online_queues counts admin + every kernel IOQ that finished
	 * Create I/O SQ; nr_allocated_queues is the total dev->queues[]
	 * capacity (admin + max_io_queues).  User QIDs occupy the gap.
	 *
	 * If the controller granted exactly num_possible_cpus() kernel
	 * IOQs, online_queues == nr_allocated_queues and there is no
	 * room for user queues -- treat that as -EBUSY so the caller
	 * surfaces a meaningful error to userspace.
	 */
	/*
	 * online_queues counts admin + every kernel IOQ that finished
	 * Create I/O SQ.  ctrl_max_io_queues is the authoritative
	 * controller-granted IOQ ceiling captured in s_nvme_setup_io_queues
	 * right after snvme_set_queue_count returned.
	 *
	 * The user QID pool occupies the gap between "first kernel
	 * unused QID" and "highest QID the controller will accept":
	 *   first = online_queues               (admin=0 + kernel IOQs)
	 *   last  = ctrl_max_io_queues          (granted ceiling)
	 *
	 * Why not nr_allocated_queues - 1?  On hosts where
	 * num_possible_cpus() exceeds the controller's MSI-X grant
	 * (e.g. 192-vCPU host + Intel DC SSD with MSI-X=136), the
	 * snvme-side dev->queues[] capacity is bigger than what the
	 * controller will actually accept; using nr_allocated_queues-1
	 * placed valid-looking QIDs in the pool that the controller
	 * then rejected with SC=0x4101 (Invalid Queue Identifier) at
	 * Create I/O CQ time.  ctrl_max_io_queues fixes this by
	 * surfacing the real controller ceiling to the pool sizer.
	 *
	 * If ctrl_max_io_queues is zero, probe never reached the
	 * negotiation step (or the build is older than this fix);
	 * fail loudly rather than fall back to the broken
	 * nr_allocated_queues-1 estimate.
	 */
	if (!ndev->ctrl_max_io_queues) {
		pr_warn("user QID pool: ctrl_max_io_queues=0 "
			"(probe did not complete the Set-Features negotiation?)\n");
		return -ENODEV;
	}

	first = ndev->online_queues;
	last  = ndev->ctrl_max_io_queues;
	if (first > last) {
		pr_warn("user QID pool empty (online=%u, ctrl_max=%u); "
			"controller refused to leave room for user IOQs.  "
			"Lower cap_kernel_ioq via NVM_SET_KERNEL_IOQ_CAP before bind, "
			"or attach to a controller with a larger MSI-X grant.\n",
			ndev->online_queues, ndev->ctrl_max_io_queues);
		return -EBUSY;
	}
	count = last - first + 1;

	bm = kcalloc(BITS_TO_LONGS(count), sizeof(unsigned long), GFP_KERNEL);
	if (!bm)
		return -ENOMEM;

	ctrl->user_qid_first  = first;
	ctrl->user_qid_last   = last;
	ctrl->user_qid_bitmap = bm;

	pr_info("user QID pool initialised: [%u..%u] (%u QIDs)\n",
		first, last, count);
	return 0;
}

/*
 * Allocate `nr` consecutive (per-call) user QIDs.  Not actually
 * required to be contiguous on the wire -- NVMe doesn't care --
 * but find_first_zero_bit + setting individually is plenty fast
 * for nr <= 16, so we just iterate.
 *
 * Caller MUST hold ctrl->user_qid_lock.  Returns the first QID
 * allocated (caller can deduce the rest in qids_out[]) or
 * -EAGAIN if the pool is full.  On failure, no bits are set.
 */
static int snvm_user_qid_alloc_locked(struct ctrl *ctrl,
				      unsigned int nr,
				      uint16_t *qids_out)
{
	unsigned int pool_size = ctrl->user_qid_last - ctrl->user_qid_first + 1;
	unsigned int i;
	unsigned int bit;

	for (i = 0; i < nr; i++) {
		bit = find_first_zero_bit(ctrl->user_qid_bitmap, pool_size);
		if (bit >= pool_size) {
			/* Pool exhausted; roll back the bits we already set. */
			while (i > 0) {
				--i;
				clear_bit(qids_out[i] - ctrl->user_qid_first,
					  ctrl->user_qid_bitmap);
			}
			return -EAGAIN;
		}
		set_bit(bit, ctrl->user_qid_bitmap);
		qids_out[i] = (uint16_t)(ctrl->user_qid_first + bit);
	}
	return 0;
}

/*
 * Release one previously-allocated user QID back to the pool.
 * Idempotent: calling on a never-allocated QID is a no-op (and
 * a WARN, since that indicates a bookkeeping bug).
 *
 * Caller MUST hold ctrl->user_qid_lock.
 */
static void snvm_user_qid_free_locked(struct ctrl *ctrl, uint16_t qid)
{
	unsigned int bit;

	if (qid < ctrl->user_qid_first || qid > ctrl->user_qid_last) {
		pr_warn("user_qid_free: qid %u outside pool [%u..%u]\n",
			qid, ctrl->user_qid_first, ctrl->user_qid_last);
		return;
	}
	bit = qid - ctrl->user_qid_first;
	if (!test_and_clear_bit(bit, ctrl->user_qid_bitmap))
		pr_warn("user_qid_free: qid %u was already free\n", qid);
}

/*
 * Forward declarations for adapter helpers used by user-queue
 * teardown.  Defined further up in the file (adapter_delete_cq /
 * adapter_delete_sq, currently around line 1597).  Re-declared
 * here so destroy_qgroup_locked, defined immediately below, can
 * reach them without re-ordering ~3000 lines.
 */
static int adapter_delete_cq(struct nvme_dev *dev, u16 cqid);
static int adapter_delete_sq(struct nvme_dev *dev, u16 sqid);

/*
 * Forward decl for snvm_ctrl_get_live_ndev (defined a few hundred
 * lines below, alongside find_qgroup_locked).  destroy_qgroup_locked
 * needs it to drain Delete I/O SQ/CQ admin commands while running
 * on cascade-cleanup paths that may race with unbind.
 */
static struct nvme_dev *snvm_ctrl_get_live_ndev(const struct ctrl *ctrl);


/*
 * Free a group descriptor and release its IDA id.  Caller must
 * hold own->groups_lock and must have already unlinked the group
 * from own->groups (or be in cascade cleanup where the list is
 * being walked-and-emptied).
 *
 * Order of operations (matters!):
 *
 *   1. Drain user queues (B3): for each alive (qid), issue
 *      Delete I/O SQ then Delete I/O CQ via the controller's
 *      admin queue.  NVMe 1.4 §5.4 requires SQ-before-CQ.
 *      Free the qid back to ctrl->user_qid_bitmap.  This MUST
 *      happen before maps are freed -- the rings the controller
 *      DMAs into are owned by maps[]; freeing them while the
 *      controller still thinks the SQ exists is a use-after-free
 *      from the DMA engine's perspective.
 *
 *   2. Drain maps (B2): unmap_and_release each one.  This frees
 *      pinned host pages / peer_memory refs, removes the map from
 *      both the global list and g->maps.
 *
 *   3. Release the group_id back to the IDA and kfree(g).
 *
 * Failure handling for step 1: NVMe Delete I/O SQ/CQ admin
 * commands almost never fail in practice (the only documented
 * failure modes are "queue not found", which is a kernel bug,
 * and timeout, which means the controller is stuck).  We log a
 * warning and continue rather than aborting the whole teardown
 * -- aborting would leave the group descriptor and its maps
 * leaked, which is strictly worse than a controller-side
 * residual SQ that the next bind will reset away.
 *
 * `ctrl` may be NULL if the caller knows the controller is gone
 * (e.g. final module exit).  In that case we skip the admin
 * commands and just reclaim the kernel-side state -- the
 * controller-side SQs will be reset on the next bind anyway.
 */
static void destroy_qgroup_locked(struct snvm_qgroup *g, struct ctrl *ctrl)
{
	struct map *m, *tmp_m;
	struct nvme_dev *ndev = NULL;
	unsigned int n_drained = 0;
	unsigned int n_queues = 0;
	unsigned int i;

	if (!g)
		return;

	/* ----- Step 1: drain user queues ----- */
	/*
	 * Resolve ndev defensively: we may be running on the
	 * fd-close cascade path AFTER SNVM_DEVICE_UNBIND already
	 * detached snvme from this BDF, in which case the in-tree
	 * nvme driver may have already rebound and reset the
	 * controller.  snvm_ctrl_get_live_ndev returns NULL for
	 * "not currently owned by snvme", and below we treat NULL
	 * as "skip the Delete I/O SQ/CQ admin step and just free
	 * host-side bookkeeping".  This keeps cleanup idempotent
	 * across unbind/rebind races.
	 */
	ndev = snvm_ctrl_get_live_ndev(ctrl);

	/*
	 * Walk in reverse just for symmetry with creation order;
	 * NVMe spec doesn't require any particular qid ordering as
	 * long as Delete-SQ precedes Delete-CQ for the same qid.
	 */
	for (i = NVM_MAX_QUEUES_PER_GROUP; i > 0; i--) {
		struct snvm_user_queue *uq = &g->queues[i - 1];

		if (!uq->alive)
			continue;

		if (ndev && ndev->ctrl.admin_q) {
			int rc;
			rc = adapter_delete_sq(ndev, uq->qid);
			if (rc)
				pr_warn("destroy_qgroup id=%u: "
					"Delete I/O SQ qid=%u failed: %d\n",
					g->group_id, uq->qid, rc);
			rc = adapter_delete_cq(ndev, uq->qid);
			if (rc)
				pr_warn("destroy_qgroup id=%u: "
					"Delete I/O CQ qid=%u failed: %d\n",
					g->group_id, uq->qid, rc);
		}

		if (ctrl) {
			mutex_lock(&ctrl->user_qid_lock);
			snvm_user_qid_free_locked(ctrl, uq->qid);
			mutex_unlock(&ctrl->user_qid_lock);
		}

		uq->alive = 0;
		n_queues++;
	}
	if (n_queues)
		pr_info("destroy_qgroup id=%u drained %u user queue(s)\n",
			g->group_id, n_queues);
	g->cur_queues = 0;

	/* ----- Step 2: drain maps ----- */
	list_for_each_entry_safe(m, tmp_m, &g->maps, group_link) {
		/*
		 * unmap_and_release() will list_del our group_link
		 * out as part of its global-list-and-group-list
		 * teardown, then free the page pins / nvidia p2p
		 * refs / etc.  We don't list_del here ourselves to
		 * keep the cleanup logic in one place.
		 */
		unmap_and_release(m);
		n_drained++;
	}
	if (n_drained)
		pr_info("destroy_qgroup id=%u drained %u map(s)\n",
			g->group_id, n_drained);

	g->nr_maps = 0;

	/* ----- Step 3: release group_id ----- */
	ida_simple_remove(&snvm_queue_group_ida, g->group_id);
	kfree(g);
}

/*
 * Resolve a per-controller "snvme owns this PCI device AND its
 * NVMe controller is fully initialised" check, returning the
 * struct nvme_dev * on success.
 *
 * Why this helper exists:
 *
 *   pci_get_drvdata(ctrl->pdev) is the obvious-looking way to
 *   reach the nvme_dev, but the in-tree `nvme` PCI driver ALSO
 *   stashes its struct nvme_dev there with a live admin_q.  If
 *   we use `pci_get_drvdata + admin_q` as the sole liveness
 *   check, an ioctl issued while the device is still owned by
 *   the in-tree driver would happily fall through and start
 *   issuing admin commands against a controller snvme does not
 *   own -- fighting the in-tree driver over IOQ resources, and
 *   in the worst case scribbling on its admin queue.
 *
 *   The single source of truth for "did SNVM_DEVICE_BIND succeed
 *   on this BDF" is the PCI core's pdev->dev.driver pointer: if
 *   it names PCI_DRIVER_NAME ("snvme") then probe ran here, and
 *   the drvdata field is owned by us.  Otherwise it's either NULL
 *   (no driver) or the in-tree nvme driver's.
 *
 * Returns:
 *   non-NULL  -- a struct nvme_dev * owned by snvme, admin_q live;
 *                safe to call adapter_alloc_*_user / etc.
 *   NULL      -- either the device is not bound to snvme, or it
 *                is bound but admin_q has not finished probe.
 *                Callers MUST surface -ENODEV in that case so
 *                userspace can poll (e.g. on probe race).
 *
 * No locking needed: pdev->dev.driver is stable for the duration
 * of one ioctl because BIND/UNBIND go through snvm_control_lock
 * via the control-plane ioctl path.
 */
static struct nvme_dev *snvm_ctrl_get_live_ndev(const struct ctrl *ctrl)
{
	struct device_driver *drv;
	struct nvme_dev *ndev;

	if (!ctrl || !ctrl->pdev)
		return NULL;

	drv = ctrl->pdev->dev.driver;
	if (!drv || !drv->name || strcmp(drv->name, PCI_DRIVER_NAME) != 0)
		return NULL;

	ndev = pci_get_drvdata(ctrl->pdev);
	if (!ndev || !ndev->ctrl.admin_q)
		return NULL;

	return ndev;
}

/*
 * Look up a queue group by id within a single fd's owner list.
 * Returns NULL if not found.  Caller MUST hold own->groups_lock.
 *
 * Cross-fd isolation is enforced here implicitly: groups are only
 * reachable from the owning fd's snvm_dev_owner.groups list, so a
 * group_id allocated by fd A is invisible to fd B's lookup.  The
 * IDA may recycle ids, but the descriptor identity is per-fd, so
 * a recycled id cannot be misused to alias someone else's group.
 */
static struct snvm_qgroup *find_qgroup_locked(struct snvm_dev_owner *own,
					      uint32_t group_id)
{
	struct snvm_qgroup *g;

	if (!own || group_id == 0)
		return NULL;
	list_for_each_entry(g, &own->groups, link) {
		if (g->group_id == group_id)
			return g;
	}
	return NULL;
}

static long snvm_dev_map_ioctl(struct file *file, unsigned int cmd,
			       unsigned long arg)
{
	struct ctrl *ctrl;
	struct map *map = NULL;
	struct nvm_ioctl_map request;
	struct nvm_ioctl_dev drequest;
	struct nvme_dev *ndev;
	struct nvme_ns *ns;
	void __user *argp = (void __user *)arg;
	u64 addr;
	int ret = 0;

	ctrl = ctrl_find_by_inode(&ctrl_list, file->f_inode);
	if (!ctrl) {
		pr_crit("unknown controller reference in snvm_dev_map_ioctl\n");
		return -EBADF;
	}

	switch (cmd) {
	case NVM_MAP_HOST_MEMORY:
		/*
		 * Pin user pages, hand back DMA addrs.  Three routings
		 * coexist (see ioctl.h struct nvm_ioctl_map and enum
		 * nvm_map_kind):
		 *
		 *   B6 map_kind == NVM_MAP_KIND_DATA
		 *                  fd-scoped data buffer.  Map is
		 *                  registered on own->data_maps; group_id
		 *                  is IGNORED for lifecycle.  Survives
		 *                  NVM_DESTROY_QUEUE_GROUP; reaped only
		 *                  on fd close (or by NVM_UNMAP_*).
		 *
		 *   B6 map_kind == NVM_MAP_KIND_RING_SQ / RING_CQ
		 *                  group-scoped ring buffer.  group_id
		 *                  MUST be non-zero.  Map is linked onto
		 *                  the per-fd group's maps list and is
		 *                  drained by NVM_DESTROY_QUEUE_GROUP /
		 *                  fd-close cascade.  NVM_ADD_USER_QUEUE
		 *                  enforces that pairs[i].sq_vaddr
		 *                  resolves to RING_SQ and pairs[i].
		 *                  cq_vaddr to RING_CQ.
		 *
		 *   map_kind == 0 (UNSPECIFIED)
		 *                  Pre-B6 caller.  Falls back to the B2
		 *                  semantics: group_id != 0 hangs the
		 *                  map on g->maps; group_id == 0 +
		 *                  ioq_idx >= 0 takes the legacy
		 *                  NVM_SET_IOQ_NUM accounting path.
		 */
		if (copy_from_user(&request, argp, sizeof(request)))
			return -EFAULT;
		if (request.reserved0[0] != 0 ||
		    request.reserved0[1] != 0 ||
		    request.reserved0[2] != 0)
			return -EINVAL;     /* MBZ; future compat */
		if (request.map_kind > NVM_MAP_KIND_DATA)
			return -EINVAL;     /* unknown kind */
		if ((request.map_kind == NVM_MAP_KIND_RING_SQ ||
		     request.map_kind == NVM_MAP_KIND_RING_CQ) &&
		    request.group_id == 0)
			return -EINVAL;     /* RING_* requires a group */

		map = map_userspace(&host_list, ctrl,
				    request.vaddr_start, request.n_pages);
		if (IS_ERR_OR_NULL(map))
			return map ? PTR_ERR(map) : -ENOMEM;

		map->kind = request.map_kind;

		if (request.map_kind == NVM_MAP_KIND_DATA) {
			/* B6: fd-scoped data buffer.  Always link onto
			 * own->data_maps; never onto g->maps.  group_id is
			 * accepted but not used for lifecycle decisions.   */
			struct snvm_dev_owner *own = file->private_data;

			if (!own) {
				unmap_and_release(map);
				return -ENODEV;
			}
			mutex_lock(&own->data_maps_lock);
			list_add_tail(&map->group_link, &own->data_maps);
			own->nr_data_maps++;
			mutex_unlock(&own->data_maps_lock);
		} else if (request.group_id != 0) {
			/* B2/B6: group-scoped attachment (ring or
			 * UNSPECIFIED-with-group).                      */
			struct snvm_dev_owner *own = file->private_data;
			struct snvm_qgroup *g;

			if (!own) {
				unmap_and_release(map);
				return -ENODEV;
			}
			mutex_lock(&own->groups_lock);
			g = find_qgroup_locked(own, request.group_id);
			if (!g) {
				mutex_unlock(&own->groups_lock);
				unmap_and_release(map);
				return -ENOENT;
			}
			map->group_id = request.group_id;
			list_add_tail(&map->group_link, &g->maps);
			g->nr_maps++;
			mutex_unlock(&own->groups_lock);
		}

		if (copy_to_user((void __user *)(uintptr_t)request.ioaddrs, map->addrs,
				 map->n_addrs * sizeof(uint64_t))) {
			/*
			 * PORTING.md section 7.3.1 trap #4: roll back every
			 * counter we bumped above AND release the
			 * mapping we just allocated.  For new-mode (group
			 * or data_maps) maps, unmap_and_release will
			 * list_del the group_link out so the per-list
			 * counter is the only thing we have to roll back
			 * manually.
			 */
			if (request.map_kind == NVM_MAP_KIND_DATA) {
				struct snvm_dev_owner *own = file->private_data;

				if (own) {
					mutex_lock(&own->data_maps_lock);
					if (own->nr_data_maps > 0)
						own->nr_data_maps--;
					mutex_unlock(&own->data_maps_lock);
				}
			} else if (request.group_id != 0) {
				struct snvm_dev_owner *own = file->private_data;
				struct snvm_qgroup *g;

				if (own) {
					mutex_lock(&own->groups_lock);
					g = find_qgroup_locked(own, request.group_id);
					if (g)
						g->nr_maps--;
					mutex_unlock(&own->groups_lock);
				}
			}
			unmap_and_release(map);
			return -EFAULT;
		}
		ret = 0;
		break;

	case NVM_MAP_DEVICE_MEMORY:
		/*
		 * Pin GPU pages (NVIDIA p2p) into device_list.  Same
		 * dual-mode semantics as NVM_MAP_HOST_MEMORY: nonzero
		 * group_id attaches the map to a per-fd group; zero
		 * keeps it on the controller-global list only.
		 *
		 * Note: the legacy NVM_MAP_DEVICE_MEMORY case did NOT
		 * touch ctrl->ioq_map_num / cq_num at all (data path
		 * only).  We preserve that: even with group_id == 0
		 * and ioq_idx >= 0 we just ignore the ioq tag here,
		 * matching the historical behaviour.  GPU queue ring
		 * registration still goes through
		 * NVM_MAP_DEVICE_QUEUE_MEMORY in legacy mode.
		 */
		if (copy_from_user(&request, argp, sizeof(request)))
			return -EFAULT;
		if (request.reserved0[0] != 0 ||
		    request.reserved0[1] != 0 ||
		    request.reserved0[2] != 0)
			return -EINVAL;
		if (request.map_kind > NVM_MAP_KIND_DATA)
			return -EINVAL;
		if ((request.map_kind == NVM_MAP_KIND_RING_SQ ||
		     request.map_kind == NVM_MAP_KIND_RING_CQ) &&
		    request.group_id == 0)
			return -EINVAL;

		map = map_device_memory(&device_list, ctrl,
					request.vaddr_start, request.n_pages,
					&ctrl_list);
		if (IS_ERR_OR_NULL(map))
			return map ? PTR_ERR(map) : -ENOMEM;

		map->kind = request.map_kind;

		if (request.map_kind == NVM_MAP_KIND_DATA) {
			/* B6: fd-scoped GPU data buffer.  Survives
			 * NVM_DESTROY_QUEUE_GROUP; reaped on fd close.   */
			struct snvm_dev_owner *own = file->private_data;

			if (!own) {
				unmap_and_release(map);
				return -ENODEV;
			}
			mutex_lock(&own->data_maps_lock);
			list_add_tail(&map->group_link, &own->data_maps);
			own->nr_data_maps++;
			mutex_unlock(&own->data_maps_lock);
		} else if (request.group_id != 0) {
			struct snvm_dev_owner *own = file->private_data;
			struct snvm_qgroup *g;

			if (!own) {
				unmap_and_release(map);
				return -ENODEV;
			}
			mutex_lock(&own->groups_lock);
			g = find_qgroup_locked(own, request.group_id);
			if (!g) {
				mutex_unlock(&own->groups_lock);
				unmap_and_release(map);
				return -ENOENT;
			}
			map->group_id = request.group_id;
			list_add_tail(&map->group_link, &g->maps);
			g->nr_maps++;
			mutex_unlock(&own->groups_lock);
		}

		if (copy_to_user((void __user *)(uintptr_t)request.ioaddrs, map->addrs,
				 map->n_addrs * sizeof(uint64_t))) {
			if (request.map_kind == NVM_MAP_KIND_DATA) {
				struct snvm_dev_owner *own = file->private_data;

				if (own) {
					mutex_lock(&own->data_maps_lock);
					if (own->nr_data_maps > 0)
						own->nr_data_maps--;
					mutex_unlock(&own->data_maps_lock);
				}
			} else if (request.group_id != 0) {
				struct snvm_dev_owner *own = file->private_data;
				struct snvm_qgroup *g;

				if (own) {
					mutex_lock(&own->groups_lock);
					g = find_qgroup_locked(own, request.group_id);
					if (g)
						g->nr_maps--;
					mutex_unlock(&own->groups_lock);
				}
			}
			unmap_and_release(map);
			return -EFAULT;
		}
		ret = 0;
		break;

	case NVM_UNMAP_HOST_MEMORY:
		if (copy_from_user(&addr, argp, sizeof(u64)))
			return -EFAULT;

		map = map_find(&host_list, addr);
		if (map) {
			/*
			 * Three concurrent attachment modes need different
			 * locking discipline before unmap_and_release runs
			 * list_del on map->group_link (the link member is
			 * shared between group_link-on-g->maps and
			 * group_link-on-own->data_maps depending on
			 * ->kind):
			 *
			 *   B6 NVM_MAP_KIND_DATA       own->data_maps_lock
			 *   B2 group-attached map      own->groups_lock
			 *   legacy / group_id == 0     no lock needed
			 */
			struct snvm_dev_owner *own = file->private_data;
			bool is_data = (map->kind == NVM_MAP_KIND_DATA);
			bool need_grp_lock =
				(!is_data && map->group_id != 0 && own);
			bool need_data_lock = (is_data && own);

			if (need_grp_lock)
				mutex_lock(&own->groups_lock);
			if (need_data_lock)
				mutex_lock(&own->data_maps_lock);

			if (need_data_lock) {
				if (own->nr_data_maps > 0)
					own->nr_data_maps--;
			} else if (map->group_id != 0 && own) {
				/*
				 * Decrement nr_maps before unmap_and_release
				 * (which list_dels group_link) so the count
				 * stays consistent throughout.  find_qgroup
				 * may return NULL if userspace destroyed the
				 * group between the map insertion and now;
				 * in that pathological case the map was
				 * already drained by destroy_qgroup_locked
				 * and we wouldn't be here.  Guard anyway.
				 */
				struct snvm_qgroup *g =
					find_qgroup_locked(own, map->group_id);
				if (g && g->nr_maps > 0)
					g->nr_maps--;
			}
			unmap_and_release(map);

			if (need_data_lock)
				mutex_unlock(&own->data_maps_lock);
			if (need_grp_lock)
				mutex_unlock(&own->groups_lock);
			ret = 0;
		} else {
			pr_warn("NVM_UNMAP_HOST_MEMORY: addr %llx not found\n", addr);
			ret = -EINVAL;
		}
		break;

	case NVM_UNMAP_DEVICE_MEMORY:
		if (copy_from_user(&addr, argp, sizeof(u64)))
			return -EFAULT;

		map = map_find(&device_list, addr);
		if (map) {
			struct snvm_dev_owner *own = file->private_data;
			bool is_data = (map->kind == NVM_MAP_KIND_DATA);
			bool need_grp_lock =
				(!is_data && map->group_id != 0 && own);
			bool need_data_lock = (is_data && own);

			if (need_grp_lock)
				mutex_lock(&own->groups_lock);
			if (need_data_lock)
				mutex_lock(&own->data_maps_lock);

			if (need_data_lock) {
				if (own->nr_data_maps > 0)
					own->nr_data_maps--;
			} else if (map->group_id != 0 && own) {
				struct snvm_qgroup *g =
					find_qgroup_locked(own, map->group_id);
				if (g && g->nr_maps > 0)
					g->nr_maps--;
			}
			unmap_and_release(map);

			if (need_data_lock)
				mutex_unlock(&own->data_maps_lock);
			if (need_grp_lock)
				mutex_unlock(&own->groups_lock);
			ret = 0;
		} else {
			pr_warn("NVM_UNMAP_DEVICE_MEMORY: addr %llx not found\n", addr);
			ret = -EINVAL;
		}
		break;

	case NVM_UNMAP_DEVICE_QUEUE_MEMORY:
		if (copy_from_user(&addr, argp, sizeof(u64)))
			return -EFAULT;

		map = map_find(&device_queue_list, addr);
		if (map) {
			unmap_and_release(map);
			ret = 0;
		} else {
			pr_warn("NVM_UNMAP_DEVICE_QUEUE_MEMORY: addr %llx not found\n", addr);
			ret = -EINVAL;
		}
		break;

	case NVM_GET_DEV_INFO:
		/* Read controller / namespace info back to userspace.
		 * Source data is split: the snvme-only fields
		 * (user_start_qid, db_stride, nr_user_use_cq) live in
		 * struct nvme_dev populated by segment 6's probe path;
		 * the namespace-derived fields (disk_name, lba_shift)
		 * are read live from the nvme_ns ref obtained via the
		 * cross-module snvme_find_get_ns export.
		 *
		 * Race fix (PORTING.md §7.3.1 trap "NVM_GET_DEV_INFO vs
		 * nvme_scan_work"): snvme_start_ctrl() enqueues
		 * nvme_scan_work asynchronously on s_nvme_wq; the
		 * cdev /dev/ssnvme<N> is already callable when the
		 * caller of SNVM_DEVICE_BIND returns, so userspace can
		 * legitimately reach this ioctl *before* the worker has
		 * scanned out nsid=1 and list_add_tail()'d it on
		 * ctrl->namespaces.  TencentOS 5.4.241 reproducibly
		 * logs "snvme_find_get_ns(nsid=1) failed" 3x at this
		 * window every BIND (see /var/log/messages May 18
		 * 16:11:12 and 19:21:46).
		 *
		 * Mitigation: flush_work(&ctrl->scan_work) synchronously
		 * waits for the in-flight scan to complete, then retry
		 * the lookup.  flush_work is documented to be safe even
		 * when the work was never queued (it returns false
		 * immediately).  We bound the total wait at 5 s to
		 * preserve userspace EFAULT semantics if the controller
		 * is genuinely broken (e.g. admin queue dead, scan
		 * never enqueued because state never reached
		 * NVME_CTRL_LIVE).
		 */
		/*
		 * Same liveness rule as the other admin-touching paths:
		 * if the device is still owned by the in-tree nvme
		 * driver, we MUST NOT return its drvdata as ours.  The
		 * B3 NVM_GET_DEV_INFO contract guarantees that fields
		 * like max_user_qid / max_queues_per_group are derived
		 * from snvme-internal state, so a pre-bind GET_DEV_INFO
		 * has no valid values to report.
		 */
		ndev = snvm_ctrl_get_live_ndev(ctrl);
		if (!ndev) {
			pr_debug("NVM_GET_DEV_INFO: controller not bound to snvme yet\n");
			return -ENODEV;
		}

		ns = snvme_find_get_ns(&ndev->ctrl, 1);
		if (!ns) {
			unsigned int wait_ms = 0;

			/*
			 * First try: synchronously wait for the in-flight
			 * scan_work to complete (no-op if it isn't queued).
			 * This handles the common case where
			 * snvme_start_ctrl already enqueued the scan but
			 * the worker hasn't run yet.
			 */
			flush_work(&ndev->ctrl.scan_work);
			ns = snvme_find_get_ns(&ndev->ctrl, 1);

			/*
			 * Fallback: scan may not have been enqueued yet
			 * (ctrl->state != LIVE at flush_work time), or
			 * scan ran but nsid=1 isn't ready (e.g. async
			 * controller cold-init).  Poll for up to 5 s.
			 */
			while (!ns && wait_ms < 5000) {
				msleep(50);
				wait_ms += 50;
				flush_work(&ndev->ctrl.scan_work);
				ns = snvme_find_get_ns(&ndev->ctrl, 1);
			}

			if (!ns) {
				pr_err("snvme_find_get_ns(nsid=1) failed after %u ms wait (state=%d)\n",
				       wait_ms, ndev->ctrl.state);
				return -EFAULT;
			}

			pr_info("NVM_GET_DEV_INFO: nsid=1 ready after %u ms scan wait\n",
				wait_ms);
		}

		memset(&drequest, 0, sizeof(drequest));
		memcpy(drequest.disk_name, ns->disk->disk_name,
		       DISK_NAME_LEN * sizeof(char));
		/*
		 * start_cq_idx: first QID available to user IOQs.
		 * Old path (NVM_SET_SHARE_REG -> probe -> mix) sets
		 * user_start_qid = online_queues at the end of mix.
		 * New path (no SET_SHARE_REG) leaves user_start_qid
		 * at 0; fall back to online_queues so userspace gets
		 * a consistent answer regardless of which flow brought
		 * the controller up.
		 */
		drequest.start_cq_idx  = ndev->user_start_qid
					 ? ndev->user_start_qid
					 : ndev->online_queues;
		drequest.dstrd         = ndev->db_stride;
		drequest.nr_user_q     = ndev->online_user_queues;
		drequest.block_size    = 1 << ns->lba_shift;
		/* CTRL.MDTS in BYTES.  ndev->ctrl.max_hw_sectors is the
		 * NVMe-block-layer internal in 512-byte sectors (regardless
		 * of LBA size); convert to bytes here so userspace gets a
		 * single, format-agnostic byte count -- matching the
		 * documented contract in include/ioctl.h ("CTRL.MDTS in
		 * bytes").  Pre-fix: userspace had to sometimes "* 512"
		 * itself depending on which version of libnvm was linked,
		 * which silently inflated/under-counted PRP_List sizing on
		 * 4 KiB-LBA controllers.  The 5.15 baseline must mirror
		 * this; both modules are in scope. */
		drequest.max_data_size = (size_t)ndev->ctrl.max_hw_sectors << 9;

		/*
		 * B3 fields.  These are the single source of truth for
		 * userspace ring sizing and QID allocation:
		 *
		 *   q_depth                NVMe CAP.MQES + 1 (controller maximum;
		 *                          no module-param clamp).  Applies to
		 *                          *every* user queue -- snvme does not
		 *                          support per-queue depth.
		 *   bar0_size              Full BAR0 region size; userspace
		 *                          mmaps up to this many bytes
		 *                          starting at offset 0 to reach
		 *                          all doorbell registers.
		 *   max_user_qid           Highest QID kernel will hand
		 *                          out via NVM_ADD_USER_QUEUE,
		 *                          inclusive.  User QID pool is
		 *                          [start_cq_idx, max_user_qid].
		 *   max_queues_per_group   Echoes the kernel-fixed cap
		 *                          (NVM_MAX_QUEUES_PER_GROUP)
		 *                          so userspace doesn't have to
		 *                          hardcode the value.
		 */
		drequest.q_depth             = (uint16_t)ndev->q_depth;
		drequest.bar0_size           = (uint32_t)pci_resource_len(ctrl->pdev, 0);
		/*
		 * B3 contract: max_user_qid is the inclusive top of the
		 * user QID pool, i.e. the controller-granted IOQ
		 * ceiling.  Userspace can rely on
		 *   start_cq_idx <= qid <= max_user_qid
		 * being safe to drive Create I/O SQ/CQ on.  We
		 * intentionally do NOT report nr_allocated_queues-1
		 * (the snvme dev->queues[] capacity), which on hosts
		 * with num_possible_cpus() > controller MSI-X grant is
		 * higher than what the controller will accept and would
		 * mis-tell userspace it has more headroom than reality.
		 */
		drequest.max_user_qid        = ndev->ctrl_max_io_queues;
		drequest.max_queues_per_group = NVM_MAX_QUEUES_PER_GROUP;
		/*
		 * Echo the Identify Controller SGLS dword captured by
		 * core.c during nvme_init_identify().  Userspace uses
		 * this to decide whether SGL data pointers are usable
		 * at all -- many SSDs are PRP-only and report sgls=0,
		 * in which case attempting CDW0.PSDT=1 returns SC=0x15
		 * (SGL Not Supported).
		 */
		drequest.sgl_supported       = (uint32_t)ndev->ctrl.sgls;

		/* ABI handshake: report the UAPI version and capability
		 * set this kernel was compiled with.  Userspace checks
		 * these in NVM_GET_DEV_INFO's return; mismatch =>
		 * fail-closed.  Old kernels (pre-UAPI-consolidation)
		 * report abi_version == 0 because memset zeroes the
		 * struct; userspace treats 0 as "unknown / legacy". */
		drequest.abi_version         = TUTTI_SNVME_ABI_VERSION;
		drequest.capabilities        = TUTTI_SNVME_CAP_ALL;

		snvme_put_ns(ns);

		if (copy_to_user(argp, &drequest, sizeof(struct nvm_ioctl_dev)))
			return -EFAULT;
		ret = 0;
		break;

	case NVM_RAW_ADMIN_CMD: {
		/*
		 * Pass-through admin SQE forwarder.  Used by userspace
		 * (NVMeService, smoke tests) to drive per-queue recycle
		 * (Delete + Create I/O SQ/CQ; NVMe 1.4 §5.4/§5.5) and
		 * any other admin command that snvme does not need a
		 * dedicated ioctl for.
		 *
		 * Restrictions:
		 *   - controller must be probed/bound: ctrl->pdev's
		 *     drvdata == valid struct nvme_dev with admin_q.
		 *     We surface -ENODEV otherwise so userspace knows
		 *     the bind step is missing.
		 *   - data-buffer admin commands are NOT supported in
		 *     this revision: we always hand snvme_submit_sync_cmd
		 *     buffer=NULL, bufflen=0.  Add a follow-up path if
		 *     Get Log Page / Set Features with payload is ever
		 *     needed (signal via reserved fields in the UAPI
		 *     struct so the _IOC_SIZE stays stable).
		 *
		 * We do NOT inspect the opcode -- this is deliberately
		 * a generic forwarder.  The caller is expected to be a
		 * privileged daemon that knows what it is sending.
		 */
		struct nvm_ioctl_raw_admin admin_req;
		struct nvme_command nvme_cmd;
		union nvme_result nvme_res;
		int admin_ret;

		if (copy_from_user(&admin_req, argp, sizeof(admin_req)))
			return -EFAULT;

		/*
		 * Same liveness rule as NVM_ADD_USER_QUEUE: only forward
		 * admin commands when snvme actually owns this BDF.
		 * pci_get_drvdata alone would happily return the in-tree
		 * nvme driver's nvme_dev pre-bind, which would be a
		 * cross-driver admin_q hijack.
		 */
		ndev = snvm_ctrl_get_live_ndev(ctrl);
		if (!ndev) {
			pr_warn("NVM_RAW_ADMIN_CMD on unbound controller (BDF=%04x:%02x:%02x.%x)\n",
				pci_domain_nr(ctrl->pdev->bus),
				ctrl->pdev->bus->number,
				PCI_SLOT(ctrl->pdev->devfn),
				PCI_FUNC(ctrl->pdev->devfn));
			return -ENODEV;
		}

		BUILD_BUG_ON(sizeof(admin_req.sqe) != sizeof(nvme_cmd));
		memcpy(&nvme_cmd, admin_req.sqe, sizeof(nvme_cmd));
		memset(&nvme_res, 0, sizeof(nvme_res));

		admin_ret = __snvme_submit_sync_cmd(ndev->ctrl.admin_q,
						   &nvme_cmd, &nvme_res,
						   NULL, 0,
						   0, NVME_QID_ANY, 0,
						   0, false);
		/*
		 * Per __snvme_submit_sync_cmd contract:
		 *   admin_ret == 0      -> success, result populated
		 *   admin_ret < 0       -> Linux errno; CQE never arrived
		 *   admin_ret > 0       -> NVMe spec status code
		 *                          (SC|SCT|...); CQE arrived but
		 *                          controller rejected the cmd
		 *
		 * We surface (>0) as ioctl success with nvme_status set
		 * so userspace can pattern-match on NVMe SC values
		 * (e.g. 0x01 "Invalid Command Opcode" for a controller
		 * that does not implement an opcode we sent).  Negative
		 * (transport) errors stay -errno.
		 */
		if (admin_ret < 0) {
			admin_req.nvme_status = 0;
			admin_req.result_dw0  = 0;
			admin_req.result_dw1  = 0;
			if (copy_to_user(argp, &admin_req, sizeof(admin_req)))
				return -EFAULT;
			return admin_ret;
		}
		admin_req.nvme_status = (uint16_t)(admin_ret & 0xFFFF);
		admin_req.result_dw0  = le32_to_cpu(nvme_res.u32);
		admin_req.result_dw1  = 0;     /* spec reserves DW1 for most admin cmds */

		if (copy_to_user(argp, &admin_req, sizeof(admin_req)))
			return -EFAULT;
		ret = 0;
		break;
	}

	case NVM_CREATE_QUEUE_GROUP:
	{
		/*
		 * Allocate a new per-fd queue group.  In B1 the group is
		 * just a kernel-side container; later steps will hang
		 * registered maps (B2) and user IO queues (B3) off of it.
		 *
		 * Why we don't require the controller to be bound:
		 * the group itself doesn't touch any NVMe state.  Bind
		 * status will be enforced when a child operation
		 * (NVM_ADD_USER_QUEUE) actually needs admin_q.  This
		 * matches the existing behaviour for NVM_MAP_HOST_MEMORY
		 * which is also bind-agnostic.
		 *
		 * Caps:
		 *   - per-fd:  NVM_MAX_GROUPS_PER_FD (default 1)
		 *   - per-group queue cap: NVM_MAX_QUEUES_PER_GROUP, echoed
		 *     back in payload.max_queues so userspace doesn't have
		 *     to hardcode the value.
		 */
		struct nvm_ioctl_queue_group req;
		struct snvm_dev_owner *own = file->private_data;
		struct snvm_qgroup *g;
		int new_id;

		if (!own)
			return -ENODEV;

		if (copy_from_user(&req, argp, sizeof(req)))
			return -EFAULT;
		if (req.flags != 0)
			return -EINVAL;
		/* MBZ check on reserved[]. */
		{
			size_t i;
			for (i = 0; i < ARRAY_SIZE(req.reserved); i++)
				if (req.reserved[i] != 0)
					return -EINVAL;
		}

		mutex_lock(&own->groups_lock);
		if (own->nr_groups >= NVM_MAX_GROUPS_PER_FD) {
			mutex_unlock(&own->groups_lock);
			return -EBUSY;
		}

		g = kzalloc(sizeof(*g), GFP_KERNEL);
		if (!g) {
			mutex_unlock(&own->groups_lock);
			return -ENOMEM;
		}

		/*
		 * IDA range starts at 1 -- group_id 0 is reserved as the
		 * "no group" sentinel for userspace.  ida_simple_get's
		 * (start, end) is [start, end), end=0 means "no upper
		 * bound", which gives us the full uint32_t range less id 0.
		 */
		new_id = ida_simple_get(&snvm_queue_group_ida, 1, 0, GFP_KERNEL);
		if (new_id < 0) {
			kfree(g);
			mutex_unlock(&own->groups_lock);
			return new_id;
		}

		g->group_id   = (uint32_t)new_id;
		g->max_queues = NVM_MAX_QUEUES_PER_GROUP;
		INIT_LIST_HEAD(&g->link);
		INIT_LIST_HEAD(&g->maps);
		g->nr_maps    = 0;
		list_add_tail(&g->link, &own->groups);
		own->nr_groups++;
		mutex_unlock(&own->groups_lock);

		req.group_id   = g->group_id;
		req.max_queues = g->max_queues;
		/* req.flags / req.reserved are already zeroed from input
		 * MBZ check; copy back unchanged. */
		if (copy_to_user(argp, &req, sizeof(req))) {
			/*
			 * Rollback: copy_to_user can fail if userspace
			 * unmapped the buffer between the copy_from and
			 * here.  Walk the group back out so the IDA
			 * doesn't leak an unreachable id.
			 */
			mutex_lock(&own->groups_lock);
			list_del(&g->link);
			own->nr_groups--;
			destroy_qgroup_locked(g, ctrl);
			mutex_unlock(&own->groups_lock);
			return -EFAULT;
		}

		pr_debug("NVM_CREATE_QUEUE_GROUP id=%u max_queues=%u pid=%d\n",
			 g->group_id, g->max_queues, current->pid);
		ret = 0;
		break;
	}

	case NVM_DESTROY_QUEUE_GROUP:
	{
		/*
		 * Explicit destroy.  Userspace passes the opaque
		 * group_id (uint32_t), we look it up in the per-fd
		 * group list (via find_qgroup_locked) and tear it
		 * down.  Cross-fd destroy is disallowed by
		 * construction: the group descriptor is only
		 * reachable via this fd's owner->groups list, so a
		 * foreign group_id is invisible and returns -ENOENT.
		 *
		 * Teardown is delegated to destroy_qgroup_locked,
		 * which since B2/B3 drains all attached user queues
		 * (Delete I/O SQ + Delete I/O CQ via the admin path)
		 * and registered maps before freeing the descriptor.
		 */
		uint32_t group_id;
		struct snvm_dev_owner *own = file->private_data;
		struct snvm_qgroup *g;
		bool found;

		if (!own)
			return -ENODEV;

		if (copy_from_user(&group_id, argp, sizeof(group_id)))
			return -EFAULT;
		if (group_id == 0)
			return -EINVAL;     /* sentinel value, never assigned */

		mutex_lock(&own->groups_lock);
		g = find_qgroup_locked(own, group_id);
		found = (g != NULL);
		if (found) {
			list_del(&g->link);
			own->nr_groups--;
			destroy_qgroup_locked(g, ctrl);
			g = NULL;     /* descriptor freed; null out to avoid use-after-free */
		}
		mutex_unlock(&own->groups_lock);

		if (!found) {
			pr_debug("NVM_DESTROY_QUEUE_GROUP id=%u not found on fd (pid=%d)\n",
				 group_id, current->pid);
			return -ENOENT;
		}

		pr_debug("NVM_DESTROY_QUEUE_GROUP id=%u pid=%d\n",
			 group_id, current->pid);
		ret = 0;
		break;
	}

	case NVM_ADD_USER_QUEUE:
	{
		/*
		 * Create a batch of (SQ, CQ) pairs against a queue group.
		 *
		 * The contract (see ioctl.h struct nvm_ioctl_add_user_queue
		 * for full text):
		 *   - controller must be bound (NVME_CTRL_LIVE),
		 *   - group_id must belong to this fd,
		 *   - 1 <= nr_pairs <= NVM_MAX_QUEUES_PER_GROUP,
		 *   - flags / reserved MBZ,
		 *   - each (sq_vaddr, cq_vaddr) resolves to a map already
		 *     registered against this group via NVM_MAP_HOST_MEMORY
		 *     / NVM_MAP_DEVICE_MEMORY (group_id != 0 path),
		 *   - cur_queues + nr_pairs <= max_queues_per_group,
		 *   - vaddrs unique within the call.
		 *
		 * On any failure mid-batch we Delete I/O SQ / CQ for every
		 * pair we already Created in this same call, free the QIDs
		 * back to the bitmap, and return the error.  The group is
		 * left exactly as the caller saw it before the ioctl.
		 */
		struct nvm_ioctl_add_user_queue *req;
		struct snvm_dev_owner *own = file->private_data;
		struct snvm_qgroup *g;
		struct nvme_dev *ndev;
		/*
		 * Zero-init defensively: snvm_user_qid_alloc_locked
		 * writes qids[0..nr_pairs-1] on success, and we never
		 * reach the rollback path unless alloc_locked succeeded
		 * (alloc_n stays 0 on alloc_locked failure).  But
		 * zeroing here keeps the invariant local and prevents
		 * a future refactor from accidentally reading stack
		 * garbage if some new error path lands here with
		 * alloc_n still > 0.
		 */
		uint16_t qids[NVM_MAX_QUEUES_PER_GROUP] = {0};
		struct map *sq_maps[NVM_MAX_QUEUES_PER_GROUP] = {NULL};
		struct map *cq_maps[NVM_MAX_QUEUES_PER_GROUP] = {NULL};
		unsigned int created = 0;     /* how many SQ+CQ pairs Create succeeded */
		unsigned int alloc_n  = 0;    /* how many qids allocated from bitmap   */
		unsigned int i, j;
		int rc;
		size_t k;

		if (!own)
			return -ENODEV;

		req = kzalloc(sizeof(*req), GFP_KERNEL);
		if (!req)
			return -ENOMEM;

		if (copy_from_user(req, argp, sizeof(*req))) {
			kfree(req);
			return -EFAULT;
		}

		/* Validate primitives. */
		if (req->flags != 0) {
			kfree(req);
			return -EINVAL;
		}
		for (k = 0; k < ARRAY_SIZE(req->reserved); k++) {
			if (req->reserved[k] != 0) {
				kfree(req);
				return -EINVAL;
			}
		}
		if (req->nr_pairs == 0 ||
		    req->nr_pairs > NVM_MAX_QUEUES_PER_GROUP) {
			kfree(req);
			return -EINVAL;
		}

		/* Detect duplicate vaddrs in the batch. */
		for (i = 0; i < req->nr_pairs; i++) {
			if (req->pairs[i].sq_vaddr == 0 ||
			    req->pairs[i].cq_vaddr == 0 ||
			    req->pairs[i].sq_vaddr == req->pairs[i].cq_vaddr) {
				kfree(req);
				return -EINVAL;
			}
			for (j = 0; j < i; j++) {
				if (req->pairs[i].sq_vaddr == req->pairs[j].sq_vaddr ||
				    req->pairs[i].cq_vaddr == req->pairs[j].cq_vaddr) {
					kfree(req);
					return -EINVAL;
				}
			}
		}

		/*
		 * Controller must be bound to snvme AND its admin_q
		 * must be live (probe finished).  snvm_ctrl_get_live_ndev
		 * checks pdev->dev.driver against PCI_DRIVER_NAME first
		 * to distinguish "owned by snvme" from "still owned by
		 * the in-tree nvme driver", which pci_get_drvdata
		 * cannot tell apart on its own.
		 */
		ndev = snvm_ctrl_get_live_ndev(ctrl);
		if (!ndev) {
			kfree(req);
			return -ENODEV;
		}

		mutex_lock(&own->groups_lock);

		g = find_qgroup_locked(own, req->group_id);
		if (!g) {
			mutex_unlock(&own->groups_lock);
			kfree(req);
			return -ENOENT;
		}
		if (g->cur_queues + req->nr_pairs > g->max_queues) {
			mutex_unlock(&own->groups_lock);
			kfree(req);
			return -EBUSY;
		}

		/*
		 * Resolve every (sq_vaddr, cq_vaddr) to a map living on
		 * g->maps.  We do this BEFORE allocating QIDs or sending
		 * any admin command so a bad lookup costs nothing on the
		 * controller side.
		 *
		 * The lookup is O(nr_pairs * group_maps) which is fine:
		 * both bounds are tiny in practice (16 * a few-dozen).
		 *
		 * GPU vs host page-size handling: map_userspace stores
		 * map->vaddr aligned to PAGE_SIZE (host 4 KiB), while
		 * map_device_memory stores it aligned to GPU_PAGE_SIZE
		 * (64 KiB).  We can't rely on a single mask covering
		 * both, so use the map's own page_size to compute the
		 * comparison mask -- this lets the same NVM_ADD_USER_QUEUE
		 * path serve both CPU smoke (host pages) and GPU smoke
		 * (NVM_MAP_DEVICE_MEMORY rings) without ABI churn.
		 */
		for (i = 0; i < req->nr_pairs; i++) {
			struct map *m_sq = NULL, *m_cq = NULL;
			struct map *cursor;

			list_for_each_entry(cursor, &g->maps, group_link) {
				u64 mask;

				/* page_size is 0 only on the create_descriptor
				 * stub before the type-specific helper runs;
				 * by the time the map is on g->maps the
				 * helper has set page_size to PAGE_SIZE
				 * (host) or GPU_PAGE_SIZE (device).  Default
				 * to host PAGE_SIZE for safety.            */
				mask = ~((cursor->page_size ?
					  (u64)cursor->page_size : (u64)PAGE_SIZE) - 1);

				/* B6: when the candidate map carries an
				 * explicit kind tag, only RING_SQ matches
				 * sq_vaddr and only RING_CQ matches
				 * cq_vaddr.  UNSPECIFIED (pre-B6 callers)
				 * still matches either slot, preserving
				 * back-compat for binaries that haven't
				 * been recompiled.  A DATA map MUST NOT
				 * match either slot -- accidentally using
				 * a data-buffer vaddr where a ring vaddr
				 * was meant would otherwise have the
				 * controller execute Create I/O SQ on the
				 * data buffer (silent corruption).      */
				if (cursor->kind == NVM_MAP_KIND_DATA)
					continue;

				if (cursor->vaddr == (req->pairs[i].sq_vaddr & mask) &&
				    (cursor->kind == 0 ||
				     cursor->kind == NVM_MAP_KIND_RING_SQ))
					m_sq = cursor;
				if (cursor->vaddr == (req->pairs[i].cq_vaddr & mask) &&
				    (cursor->kind == 0 ||
				     cursor->kind == NVM_MAP_KIND_RING_CQ))
					m_cq = cursor;
				if (m_sq && m_cq)
					break;
			}
			if (!m_sq || !m_cq) {
				mutex_unlock(&own->groups_lock);
				kfree(req);
				return -ENOENT;
			}
			sq_maps[i] = m_sq;
			cq_maps[i] = m_cq;
		}

		/*
		 * Allocate QIDs.  Lazy-init the pool on first use (this
		 * is the only ADD path that can be the very first ioctl
		 * after probe completes).
		 */
		mutex_lock(&ctrl->user_qid_lock);
		rc = snvm_user_qid_pool_init_locked(ctrl, ndev);
		if (rc) {
			mutex_unlock(&ctrl->user_qid_lock);
			mutex_unlock(&own->groups_lock);
			kfree(req);
			return rc;
		}
		rc = snvm_user_qid_alloc_locked(ctrl, req->nr_pairs, qids);
		mutex_unlock(&ctrl->user_qid_lock);
		if (rc) {
			mutex_unlock(&own->groups_lock);
			kfree(req);
			return rc;     /* -EAGAIN: pool full */
		}
		alloc_n = req->nr_pairs;

		/*
		 * Drive the controller: Create I/O CQ first, then Create
		 * I/O SQ (the SQ creation references the CQ by qid, so
		 * NVMe spec requires this order).  On failure, unwind:
		 * delete all (CQ, SQ) pairs we already created, free all
		 * QIDs we allocated.
		 */
		for (i = 0; i < req->nr_pairs; i++) {
			rc = adapter_alloc_cq_user(ndev, cq_maps[i], qids[i]);
			if (rc) {
				pr_warn("NVM_ADD_USER_QUEUE: Create I/O CQ qid=%u rc=%d\n",
					qids[i], rc);
				goto rollback_unlocked;
			}
			rc = adapter_alloc_sq_user(ndev, sq_maps[i], qids[i]);
			if (rc) {
				pr_warn("NVM_ADD_USER_QUEUE: Create I/O SQ qid=%u rc=%d\n",
					qids[i], rc);
				/*
				 * SQ failed but CQ for this i was already
				 * created -- delete it before unwinding the
				 * earlier pairs.  Account for this with
				 * created++ first so the rollback loop
				 * picks it up.
				 */
				adapter_delete_cq(ndev, qids[i]);
				goto rollback_unlocked;
			}
			created++;
		}

		/* All pairs created; commit them to the group descriptor. */
		for (i = 0; i < req->nr_pairs; i++) {
			struct snvm_user_queue *uq = &g->queues[g->cur_queues + i];
			uint16_t qid = qids[i];

			uq->qid       = qid;
			uq->alive     = 1;
			uq->sq_vaddr  = req->pairs[i].sq_vaddr;
			uq->cq_vaddr  = req->pairs[i].cq_vaddr;

			req->out_pairs[i].sq_doorbell_offset =
				(uint32_t)(NVME_REG_DBS + qid * 2 * ndev->db_stride * 4);
			req->out_pairs[i].cq_doorbell_offset =
				(uint32_t)(NVME_REG_DBS + (qid * 2 + 1) * ndev->db_stride * 4);
			req->out_pairs[i].qid = qid;
		}
		g->cur_queues += req->nr_pairs;

		mutex_unlock(&own->groups_lock);

		if (copy_to_user(argp, req, sizeof(*req))) {
			/*
			 * copy_to_user failed AFTER admin commands
			 * succeeded.  We must unwind the controller-side
			 * Create I/O SQ/CQ to keep snvme's view of the
			 * world consistent with the ioctl's failure
			 * return.  Take the locks again, walk the slots
			 * we just committed, and revert.
			 */
			mutex_lock(&own->groups_lock);
			for (i = 0; i < req->nr_pairs; i++) {
				struct snvm_user_queue *uq =
					&g->queues[g->cur_queues - req->nr_pairs + i];
				adapter_delete_sq(ndev, uq->qid);
				adapter_delete_cq(ndev, uq->qid);
				uq->alive = 0;
				mutex_lock(&ctrl->user_qid_lock);
				snvm_user_qid_free_locked(ctrl, uq->qid);
				mutex_unlock(&ctrl->user_qid_lock);
			}
			g->cur_queues -= req->nr_pairs;
			mutex_unlock(&own->groups_lock);
			kfree(req);
			return -EFAULT;
		}

		pr_info("NVM_ADD_USER_QUEUE group=%u created %u queue(s) (qids %u..%u)\n",
			req->group_id, req->nr_pairs,
			qids[0], qids[req->nr_pairs - 1]);
		kfree(req);
		ret = 0;
		break;

rollback_unlocked:
		/*
		 * `created` SQ+CQ pairs are committed on the controller;
		 * delete them in reverse order.  Then free ALL allocated
		 * QIDs (including the one whose Create CQ/SQ failed --
		 * snvm_user_qid_alloc_locked set its bit unconditionally).
		 */
		for (i = created; i > 0; i--) {
			adapter_delete_sq(ndev, qids[i - 1]);
			adapter_delete_cq(ndev, qids[i - 1]);
		}
		mutex_lock(&ctrl->user_qid_lock);
		for (i = 0; i < alloc_n; i++)
			snvm_user_qid_free_locked(ctrl, qids[i]);
		mutex_unlock(&ctrl->user_qid_lock);
		mutex_unlock(&own->groups_lock);
		kfree(req);
		return rc;
	}

	case NVM_SET_KERNEL_IOQ_CAP: {
		/*
		 * Cap-only update path: stash ctrl->setup.cap_kernel_ioq.
		 * Probe segment 6a copies ctrl->setup.cap_kernel_ioq into
		 * dev->cap_kernel_ioq at SNVM_DEVICE_BIND time, gated on
		 * ctrl->setup.valid -- so we set .valid here too.
		 *
		 * Must run pre-bind to have any effect.  We do not reject
		 * post-bind calls (the field write is still useful for the
		 * NEXT bind cycle if the user unbinds/rebinds), but log so
		 * a misordered userspace is diagnosable from dmesg.
		 */
		uint32_t cap;

		if (copy_from_user(&cap, argp, sizeof(cap)))
			return -EFAULT;

		ctrl->setup.cap_kernel_ioq = cap;
		ctrl->setup.valid          = 1;

		pr_info("NVM_SET_KERNEL_IOQ_CAP cap=%u\n", cap);
		ret = 0;
		break;
	}

	default:
		/*
		 * Unknown ioctl. Two sub-cases worth distinguishing:
		 *
		 *  (1) ioctl type byte == 'N' (0x4e): in-tree NVMe
		 *      uapi/linux/nvme_ioctl.h ioctls
		 *      (NVME_IOCTL_ADMIN_CMD = 0xc0484e41 etc.).  These
		 *      are issued by monitoring daemons that scan every
		 *      device-node that looks NVMe-ish: smartd, nvme-cli,
		 *      Tencent host-monitor's deviceQuery, etc.
		 *      /dev/ssnvme* deliberately does NOT implement the
		 *      in-tree NVMe ioctl ABI (libnvm is the only
		 *      sanctioned client; admin pass-through must go to
		 *      /dev/nvme%dn%d on the in-tree side).  Returning
		 *      -ENOTTY is the POSIX-correct answer ("this fd
		 *      does not support this ioctl") and tells smartd to
		 *      give up cleanly on this device.  Log at
		 *      pr_debug to avoid the 10-minute-cadence dmesg
		 *      flood that smartd otherwise produces on hosts
		 *      where the operator has not constrained smartd to
		 *      explicit devices.
		 *
		 *  (2) anything else: a genuinely unknown ioctl, almost
		 *      certainly a userspace bug.  Keep the loud
		 *      pr_notice + -EINVAL.
		 */
		if (_IOC_TYPE(cmd) == 'N') {
			pr_debug("rejecting in-tree NVMe ioctl 0x%x on /dev/ssnvme from pid %d (-ENOTTY)\n",
				 cmd, current->pid);
			ret = -ENOTTY;
		} else {
			pr_notice("unknown /dev/ssnvme ioctl 0x%x from pid %d\n",
				  cmd, current->pid);
			ret = -EINVAL;
		}
		break;
	}

	return ret;
}

/*
 * mmap callback for /dev/ssnvme<minor>: maps the controller's BAR0
 * register space (doorbell region included) into the calling process
 * so libnvm can ring doorbells from CPU userspace without an ioctl
 * round-trip.
 *
 * Per PORTING.md section 7.3.1: the NULL-guard MUST be `||`, not `&&`.
 * `ctrl_find_by_inode()` can legitimately return NULL when the caller
 * mmap()s an ssnvme cdev whose ctrl was already torn down by another
 * thread; the `&&` form would then dereference NULL.  This was a
 * regression in earlier snvme-5.15.0 revisions and has been fixed
 * upstream.
 */
static int svm_mmap_registers(struct file *file, struct vm_area_struct *vma)
{
	struct ctrl *ctrl = NULL;

	ctrl = ctrl_find_by_inode(&ctrl_list, file->f_inode);
	if (ctrl == NULL || ctrl->pdev == NULL) {
		pr_crit("unknown controller reference in svm_mmap_registers\n");
		return -EBADF;
	}

	if (vma->vm_end - vma->vm_start > pci_resource_len(ctrl->pdev, 0)) {
		pr_warn("invalid mmap range size\n");
		return -EINVAL;
	}
	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
	return vm_iomap_memory(vma, pci_resource_start(ctrl->pdev, 0),
			       vma->vm_end - vma->vm_start);
}

/*
 * /dev/ssnvme<minor> per-fd owner record.
 *
 * snvme-5.4 / 5.15 inherits an upstream design omission: snvm_dev_fops
 * had no .release hook, so when a userspace process exited without
 * issuing NVM_UNMAP_HOST_MEMORY / NVM_UNMAP_DEVICE_MEMORY /
 * NVM_UNMAP_DEVICE_QUEUE_MEMORY for every map it had created, the
 * pinned host pages and peer_memory references stayed live forever:
 *
 *   - host_list:         get_user_pages refs never put_page()d
 *   - device_list:       peer_memory get_pages refs never freed
 *   - device_queue_list: ditto, plus dma_mapping leaks
 *   - ctrl->ioq_map_num / ctrl->cq_num never decremented
 *
 * The fallout is exactly the cluster of symptoms reproducible against
 * baseline 5.4.241-1-tlinux4-0017:
 *   1. After a single test crash, the next SNVM_DEVICE_BIND sees
 *      "snvme: ctrl exist, ioq_num=N cq_num=M map_num=K" with stale
 *      counters -- the controller is reused dirty.
 *   2. probe completes asynchronously but the parent thread issues
 *      NVM_GET_DEV_INFO before nvme_scan_work has had a chance to
 *      attach nsid=1, hence "snvme_find_get_ns(nsid=1) failed" 3x.
 *   3. nvidia.ko refcnt accumulates and snvme.ko cannot be unloaded
 *      ("module in use"), which userspace experiences as "snvme
 *      cannot be released" / NVLink-Inband stalls.
 *
 * Fix: track per-fd owner task on open, and on release walk all three
 * map lists and unmap_and_release() every descriptor owned by that
 * task, while rolling back the per-ctrl IO-queue accounting counters
 * that NVM_MAP_* had bumped.  Combined with the existing
 * map->owner = current assignment in map.c::create_descriptor, this
 * makes the cleanup symmetric with NVM_UNMAP_* and idempotent against
 * userspace crashes.
 *
 * See PORTING.md section 7.3.1 trap #6 for the full reasoning.
 *
 * (`struct snvm_dev_owner`, `struct snvm_qgroup`, and
 * destroy_qgroup_locked() are defined above snvm_dev_map_ioctl --
 * the ioctl dispatcher needs them in scope to handle
 * NVM_CREATE_QUEUE_GROUP / NVM_DESTROY_QUEUE_GROUP.  The bodies
 * of snvm_dev_open / snvm_dev_release stay here with the rest of
 * the chrdev fops vtable.)
 */

static int snvm_dev_open(struct inode *inode, struct file *file)
{
	struct ctrl *ctrl;
	struct snvm_dev_owner *own;

	ctrl = ctrl_find_by_inode(&ctrl_list, inode);
	if (!ctrl) {
		pr_err("snvm_dev_open: no ctrl for inode\n");
		return -ENODEV;
	}

	own = kzalloc(sizeof(*own), GFP_KERNEL);
	if (!own)
		return -ENOMEM;

	own->ctrl  = ctrl;
	/*
	 * Pin the opener; release runs in the exit path of *some* task
	 * (could be a different thread group member, or even a forked
	 * child), so we cannot rely on `current` matching at .release
	 * time.  Stashing the opener gives map_purge_by_owner a stable
	 * key that matches what map.c::create_descriptor recorded.
	 */
	own->owner = current;
	INIT_LIST_HEAD(&own->groups);
	mutex_init(&own->groups_lock);
	own->nr_groups = 0;
	INIT_LIST_HEAD(&own->data_maps);
	mutex_init(&own->data_maps_lock);
	own->nr_data_maps = 0;
	file->private_data = own;
	return 0;
}

static int snvm_dev_release(struct inode *inode, struct file *file)
{
	struct snvm_dev_owner *own = file->private_data;
	struct ctrl *ctrl;
	struct task_struct *owner;
	struct snvm_qgroup *g, *tmp_g;
	unsigned long n_host = 0, n_dev = 0, n_devq = 0;
	unsigned int n_groups = 0;

	if (!own)
		return 0;

	ctrl  = own->ctrl;
	owner = own->owner;

	/*
	 * Pass 0: cascade-destroy any queue groups still attached to
	 * this fd.  Userspace may have crashed mid-flight, or simply
	 * closed the fd without calling NVM_DESTROY_QUEUE_GROUP --
	 * either way every group on this fd's list must be reaped or
	 * it leaks group_id bits in snvm_queue_group_ida.
	 *
	 * IMPORTANT ordering: groups are drained BEFORE the map
	 * passes below.  Steps B2/B3 will park user IO queues and
	 * pinned NVMe ring maps inside group descriptors; if the map
	 * lists were freed first, the Delete I/O SQ/CQ admin commands
	 * issued during group teardown would see ring physical
	 * addresses that have already been unmapped from the IOMMU,
	 * which the controller could DMA into freed pages.
	 *
	 * In B1 there are no NVMe-side resources yet, so this loop
	 * just frees the descriptors and returns the group_ids to
	 * the IDA.  The pr_info below tracks the count so smoke
	 * tests can grep dmesg to confirm cascade-cleanup ran.
	 */
	mutex_lock(&own->groups_lock);
	list_for_each_entry_safe(g, tmp_g, &own->groups, link) {
		list_del(&g->link);
		destroy_qgroup_locked(g, ctrl);
		n_groups++;
	}
	own->nr_groups = 0;
	mutex_unlock(&own->groups_lock);

	if (n_groups)
		pr_info("snvm_dev_release: cascade-destroyed %u orphan group(s) for pid=%d\n",
			n_groups, owner ? owner->pid : -1);

	/*
	 * Pass 0.5: cascade-release any fd-scoped DATA maps the
	 * owner registered with map_kind == NVM_MAP_KIND_DATA.  These
	 * are NOT attached to any snvm_qgroup, so the queue-group
	 * cascade above missed them; they live on own->data_maps and
	 * have to be reaped here on fd close.  unmap_and_release
	 * pulls each one off both the global list (host_list /
	 * device_list / device_queue_list, via map->list) and the
	 * data_maps list (via map->group_link, which we reuse for
	 * fd-scoped attachment in the same way snvm_qgroup.maps does
	 * for group-scoped attachment).
	 */
	{
		unsigned int n_data = 0;
		struct map *m, *tmp_m;

		mutex_lock(&own->data_maps_lock);
		list_for_each_entry_safe(m, tmp_m, &own->data_maps, group_link) {
			unmap_and_release(m);
			n_data++;
		}
		own->nr_data_maps = 0;
		mutex_unlock(&own->data_maps_lock);

		if (n_data)
			pr_info("snvm_dev_release: cascade-released %u DATA map(s) for pid=%d\n",
				n_data, owner ? owner->pid : -1);
	}

	/*
	 * Free every legacy (non-group, non-data) map the dying owner
	 * left on the global lists.  Group-attached and fd-scoped DATA
	 * maps were already reaped by the cascades above.
	 */
	n_host = map_purge_by_owner(&host_list,         owner);
	n_dev  = map_purge_by_owner(&device_list,       owner);
	n_devq = map_purge_by_owner(&device_queue_list, owner);

	if (n_host || n_dev || n_devq)
		pr_info("snvm_dev_release: reclaimed host=%lu dev=%lu devq=%lu for pid=%d\n",
			n_host, n_dev, n_devq,
			owner ? owner->pid : -1);

	mutex_destroy(&own->groups_lock);
	mutex_destroy(&own->data_maps_lock);
	kfree(own);
	file->private_data = NULL;
	return 0;
}

/*
 * file_operations for /dev/ssnvme<minor>.  Owned by ssnvme cdev created
 * inside snvm_chrdev_create() below via ctrl_chrdev_create().
 *
 * .open  records the owning task in file->private_data so .release can
 *        match map descriptors by ->owner irrespective of which thread
 *        ultimately invokes the close() / __fput().
 * .release reclaims any map left behind by an abnormal userspace exit.
 *          See snvm_dev_release above for the full rationale.
 */
static const struct file_operations snvm_dev_fops = {
	.owner		= THIS_MODULE,
	.open		= snvm_dev_open,
	.release	= snvm_dev_release,
	.unlocked_ioctl	= snvm_dev_map_ioctl,
	.mmap		= svm_mmap_registers,
};

/*
 * snvm_chrdev_create: real implementation (replaces the segment-3
 * stub).  Allocates an unused minor from snvm_chrdev_minor_ida,
 * builds a struct ctrl pinning the pci_dev, and registers the cdev
 * + sysfs entry as /dev/ssnvme<minor> via ctrl_chrdev_create().
 *
 * Caller (snvm_chrdev_helper) holds a pci_dev_get reference on pdev
 * and is responsible for pci_dev_put on failure paths above us; on
 * success the reference is transferred to ctrl->pdev for the lifetime
 * of the ssnvme cdev.
 */
static int snvm_chrdev_create(struct pci_dev *pdev, unsigned int class)
{
	struct ctrl *ctrl;
	int minor, err;

	if (pdev->class != class) {
		pr_err("unexpected pci class mismatch (got 0x%x, expected 0x%x)\n",
		       pdev->class, class);
		return -EINVAL;
	}

	minor = ida_simple_get(&snvm_chrdev_minor_ida, 0, 0, GFP_KERNEL);
	if (minor < 0)
		return minor;

	ctrl = ctrl_get(&ctrl_list, dev_class, pdev, minor);
	if (IS_ERR(ctrl)) {
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

/*
 * SNVM_CHRDEV_CREATE / SNVM_CHRDEV_REMOVE common path.  On create,
 * looks up (or creates) a struct ctrl for the addressed PCI device and
 * returns its assigned controller number to userspace via dev_addr.
 */
static int snvm_chrdev_helper(struct pci_device_addr *dev_addr, int create)
{
	struct pci_device_addr pdev_addr;
	struct pci_dev *pdev;
	struct ctrl *ctrl;
	/*
	 * Idempotent default (ported from snvme-5.15.0-public/pci.c, see
	 * that file's comment on this same function): a CHRDEV_CREATE for
	 * a BDF that already has a chrdev, or a CHRDEV_REMOVE for a BDF
	 * that has none, is success -- not an error.  This function used
	 * to init ret to -EFAULT, so BOTH of those idempotent cases fell
	 * through to `return -EFAULT` (errno=14, "Bad address") with no
	 * dev_addr/pdev-related fault having actually occurred.  Concretely
	 * this is what an owner process getting SIGKILL'd (instead of
	 * exiting via SIGINT/SIGTERM) reproduces: the killed process's
	 * struct ctrl for a BDF is still in ctrl_list (nothing tore it
	 * down), so the next daemon start's SNVM_CHRDEV_CREATE for that
	 * same BDF hit the missing `create && ctrl` case below and failed
	 * with errno=14 instead of just reporting the existing minor.
	 */
	int ret = 0;

	pdev_addr = *dev_addr;
	pdev = TO_PCI_DEV(pdev_addr);
	if (!pdev) {
		pr_err("(%s): pci_get_domain_bus_and_slot failed\n", __func__);
		return -EFAULT;
	}

	ctrl = ctrl_find_by_pci_dev(&ctrl_list, pdev);
	if (create && !ctrl) {
		ret = snvm_chrdev_create(pdev, PCI_CLASS_STORAGE_EXPRESS);
		if (!ret) {
			ctrl = ctrl_find_by_pci_dev(&ctrl_list, pdev);
			if (ctrl) {
				/* Tell userspace which ssnvme<domain> minor
				 * to open: libnvm reads dev_addr->domain
				 * back out and snprintf's "/dev/ssnvme%d".
				 */
				memset(dev_addr, 0, sizeof(struct pci_device_addr));
				dev_addr->domain = ctrl->number;
			}
		}
	} else if (create && ctrl) {
		/*
		 * create: BDF already has a chrdev -- idempotent.  Typical
		 * shape: a previous owner (smoke test, or a daemon that got
		 * SIGKILL'd before it could SNVM_CHRDEV_REMOVE) left this
		 * BDF's chrdev registered; report the existing minor via
		 * dev_addr->domain (same protocol as the fresh-create path)
		 * instead of failing.  Nothing to do kernel-side: the cdev /
		 * class device / IDA minor are already in place.
		 */
		memset(dev_addr, 0, sizeof(struct pci_device_addr));
		dev_addr->domain = ctrl->number;
		ret = 0;
	} else if (!create && ctrl) {
		/*
		 * PORTING.md section 7.3.1: ctrl_put() FIRST, ida_simple_remove()
		 * SECOND.  ctrl_put() runs device_destroy() / cdev_del()
		 * which decode the minor out of ctrl->number; freeing the
		 * minor first opens a race window where a concurrent
		 * SNVM_CHRDEV_CREATE picks up the same minor and clashes
		 * with our still-live cdev.  ctrl_put() also kfree(ctrl),
		 * so cache the minor in a local first.
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

/*
 * SNVM_CACULATE_PCIDISTANCE was an opcode-5 ioctl that returned
 * pci_p2pdma_distance_many() between two BDFs so libnvm could pick the
 * closest GPU-NVMe pair for queue placement.  It was REMOVED from the
 * UAPI (see backends/local/nvme/libnvm/include/ioctl.h) and the snvme-
 * 5.15.0 baseline already dropped both the kernel-side handler
 * (snvm_caculate_pci_distance) and the `struct pci_device_addr_pair`
 * type it used.  The 5.4 baseline -- which was forked from a snapshot
 * predating that cleanup -- still carried both, which broke the build
 * because the type and the SNVM_CACULATE_PCIDISTANCE constant no
 * longer exist in the shared UAPI header.
 *
 * Distance computation is now done entirely in userspace by libnvm
 * using sysfs (drivers/nvme/.../device/numa_node and analogous GPU
 * paths); see Controller::Controller() in libnvm/src/ctrl.cpp for the
 * replacement code path.  Per ioctl.h: opcode 5 must NOT be reused for
 * at least one release cycle so old userspace binaries don't silently
 * get a different command's result.
 */

/*
 * Real /dev/snvm_control ioctl dispatch.  Forward-declared near the
 * top of the file so snvm_fops can take its address before this
 * definition appears.
 */
static long snvm_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct pci_device_addr dev_addr;
	void __user *argp = (void __user *)arg;
	int ret;

	if (copy_from_user(&dev_addr, argp, sizeof(dev_addr))) {
		pr_err("(%s): copy_from_user error\n", __func__);
		return -EFAULT;
	}

	switch (cmd) {
	case SNVM_DEVICE_BIND:
		return snvm_rebind_driver(dev_addr);
	case SNVM_DEVICE_UNBIND:
		return snvm_unbind_driver(dev_addr);
	case SNVM_CHRDEV_CREATE:
		ret = snvm_chrdev_helper(&dev_addr, 1);
		if (!ret) {
			if (copy_to_user(argp, &dev_addr, sizeof(dev_addr)))
				return -EFAULT;
		}
		return ret;
	case SNVM_CHRDEV_REMOVE:
		return snvm_chrdev_helper(&dev_addr, 0);
	default:
		return -ENOTTY;
	}
}

static int __init nvme_init(void)
{
	int ret;

	BUILD_BUG_ON(sizeof(struct nvme_create_cq) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_create_sq) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_delete_queue) != 64);
	BUILD_BUG_ON(IRQ_AFFINITY_MAX_SETS < 2);

	snvm_registered = 0;

	/*
	 * Resolve peer_memory symbols from the selected GPU P2P backend. Failure
	 * here is fatal: GPU queue/memory mapping in map.c depends
	 * on these symbols, and without them libnvm's GPU-side registration
	 * path would silently crash.
	 */
	if (peer_memory_ops.init()) {
		pr_err("could not load peer_memory symbols\n");
		return -EOPNOTSUPP;
	}

	/*
	 * Probe the optional Phoenix P2P service once.  If phoenixfs is
	 * loaded, this caches its phxfs_p2p_* function pointers and holds
	 * a module reference for snvme's lifetime (so phoenixfs cannot be
	 * unloaded while snvme is loaded).  If phoenixfs is absent, GPU
	 * memory registration falls back to the native peer_memory path.
	 */
	map_p2p_service_probe();

	list_init(&ctrl_list);
	list_init(&host_list);
	list_init(&device_list);
	list_init(&device_queue_list);

	/*
	 * snvm_cdev_init MUST stay the last fallible step here.
	 * Its own internal goto-chain is the only cleanup path
	 * for the chrdev / sysfs resources it sets up; nvme_init's
	 * outer error path only knows how to undo peer_memory_ops.init.
	 *
	 * If you add a new fallible step AFTER snvm_cdev_init, you
	 * MUST also call snvm_cdev_release() in the new error label
	 * before falling through to err_p2p_exit -- otherwise a
	 * failure there will leak /dev/snvm_control + the sysfs
	 * class node, exactly the failure mode that requires a
	 * reboot to recover from.
	 */
	ret = snvm_cdev_init();
	if (ret)
		goto err_p2p_exit;

	/*
	 * Intentionally NOT calling pci_register_driver(&snvme_driver) here
	 * -- it is deferred to the first SNVM_DEVICE_BIND ioctl via
	 * register_driver().  See the design banner above the lazy block
	 * for the rationale.
	 */
	pr_info("module loaded successfully\n");
	return 0;

err_p2p_exit:
	peer_memory_ops.exit();
	return ret;
}

/*
 * clear_map_list -- drain a global map list at module-exit time.
 *
 * Mirrors snvme-5.15.0/pci.c::clear_map_list.  Each call to
 * unmap_and_release() does list_remove() on the descriptor it just
 * freed, so we must re-fetch list_next(&list->head) on every iteration
 * (saving a "next" pointer up-front would dereference a freed node).
 *
 * Returns the number of entries that were still alive on entry; a
 * non-zero return means userspace forgot to unmap something before
 * closing fds, and we plugged the leak on its behalf.
 */
static unsigned long clear_map_list(struct list *list)
{
	unsigned long freed = 0;
	struct list_node *element;

	if (list == NULL)
		return 0;

	element = list_next(&list->head);
	while (element != NULL) {
		struct map *map = container_of(element, struct map, list);

		unmap_and_release(map);
		++freed;
		/* head changed; restart from the new front */
		element = list_next(&list->head);
	}
	return freed;
}

/*
 * clear_ctrl_list -- drain ctrl_list at module-exit time.
 *
 * The /dev/ssnvme<N> chrdev is created by SNVM_CHRDEV_CREATE and would
 * normally be torn down by a matching SNVM_CHRDEV_REMOVE.  If userspace
 * forgets (or aborts mid-flight, as smoke tests sometimes do), the ctrl
 * stays on ctrl_list and its sysfs/cdev entries persist.  When rmmod
 * then runs snvm_cdev_release(), class_destroy() walks the class
 * children and trips over those orphaned devices, leaving the sysfs
 * tree wedged ("/sys/class/libsnvm helper/" survives) -- which on the
 * next insmod blows up class_create() with -EEXIST and the standard
 * advice is "reboot".
 *
 * Drain the list here BEFORE snvm_cdev_release() so device_destroy() /
 * cdev_del() (called inside ctrl_put -> ctrl_chrdev_remove) run while
 * dev_class is still valid.  ctrl_put removes the ctrl from the list
 * and kfrees it, so re-fetch list_next(&list->head) on every iteration.
 *
 * The minor IDA itself is released wholesale by ida_destroy() inside
 * snvm_cdev_release(); per-ctrl ida_simple_remove() is unnecessary
 * here.  ctrl does not hold an extra pci_dev reference, so no
 * pci_dev_put either.
 */
static unsigned long clear_ctrl_list(struct list *list)
{
	unsigned long freed = 0;
	struct list_node *element;

	if (list == NULL)
		return 0;

	element = list_next(&list->head);
	while (element != NULL) {
		struct ctrl *ctrl = container_of(element, struct ctrl, list);

		ctrl_put(ctrl);
		++freed;
		element = list_next(&list->head);
	}
	return freed;
}

static void __exit nvme_exit(void)
{
	unsigned long leaked;

	/*
	 * Step 1: stop accepting new probes / unbind any devices the
	 * snvme PCI driver currently owns.  After this returns no new
	 * ctrl can be registered and no in-flight probe is running.
	 */
	if (snvm_registered) {
		snvm_unregister_driver();
		snvm_registered = 0;
	}

	/*
	 * Step 2: drain pinned map descriptors.  These should already
	 * be empty if every fd closed cleanly (snvm_dev_release runs
	 * map_purge_by_owner on each fd), but a userspace crash can
	 * leave stragglers behind -- clean them up rather than leak
	 * pinned pages and trip the kernel's gup refcount WARN.
	 *
	 * Order matters: drain maps BEFORE ctrls because a map's DMA
	 * unmap path dereferences its owning pdev, which is still
	 * valid as long as the ctrl that pinned it is alive.
	 */
	leaked = clear_map_list(&host_list);
	if (leaked)
		pr_notice("%lu host memory mapping(s) leaked at unload\n", leaked);
	leaked = clear_map_list(&device_list);
	if (leaked)
		pr_notice("%lu device memory mapping(s) leaked at unload\n", leaked);
	leaked = clear_map_list(&device_queue_list);
	if (leaked)
		pr_notice("%lu device-queue mapping(s) leaked at unload\n", leaked);

	/*
	 * Step 3: drain orphaned /dev/ssnvme<N> ctrls.  Must run BEFORE
	 * snvm_cdev_release() so device_destroy() / cdev_del() inside
	 * ctrl_put -> ctrl_chrdev_remove see a still-valid dev_class.
	 * Without this, an unclean test exit leaves ssnvme0 wedged in
	 * sysfs and the next insmod fails with -EEXIST until reboot.
	 */
	leaked = clear_ctrl_list(&ctrl_list);
	if (leaked)
		pr_notice("drained %lu orphan ctrl(s) at unload (userspace forgot SNVM_CHRDEV_REMOVE)\n",
			  leaked);

	/* Step 4: tear down the singleton /dev/snvm_control + class. */
	snvm_cdev_release();

	/*
	 * Step 5: drop the Phoenix P2P service reference taken in
	 * nvme_init.  Safe here because every map (and thus every
	 * phxfs_p2p handle) has already been drained by the
	 * clear_map_list() calls in Step 2.
	 */
	map_p2p_service_release();

	/* Step 6: drop the GPU/p2p notifier registration. */
	peer_memory_ops.exit();
	pr_info("module unloaded successfully\n");
}

MODULE_AUTHOR("Matthew Wilcox <willy@linux.intel.com>");
MODULE_LICENSE("GPL");
MODULE_VERSION("1.0");
module_init(nvme_init);
module_exit(nvme_exit);

// SPDX-License-Identifier: GPL-2.0
/*
 * Modified NVM Express device driver
 *
 * PORTING STATUS (Linux 6.8):
 *   - based on the upstream v6.8 NVMe PCI lifecycle;
 *   - SNVMe queue groups, mappings, char devices, and opt-in bind gate ported;
 *   - GPU-consumed completion interrupts remain IRQ_HANDLED by design;
 *   - build-validated against Ubuntu 6.8.0-90-generic.
 */

#include <linux/acpi.h>
#include <linux/async.h>
#include <linux/blkdev.h>
#include <linux/blk-mq.h>
#include <linux/blk-mq-pci.h>
#include <linux/blk-integrity.h>
#include <linux/delay.h>
#include <linux/dmi.h>
#include <linux/init.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/kstrtox.h>
#include <linux/memremap.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/once.h>
#include <linux/pci.h>
#include <linux/suspend.h>
#include <linux/t10-pi.h>
#include <linux/types.h>
#include <linux/io-64-nonatomic-lo-hi.h>
#include <linux/io-64-nonatomic-hi-lo.h>
#include <linux/sed-opal.h>
#include <linux/pci-p2pdma.h>
#include <linux/device/driver.h>

#include "asm/string_64.h"
#include "linux/device.h"
#include "linux/idr.h"
#include "linux/printk.h"
#include "linux/uaccess.h"
#include "map.h"
#include "ioctl.h"
#include "nvme.h"
#include "list.h"
#include "ctrl.h"
#include "peer_memory.h"

#define DRIVER_NAME         "libsnvm helper"
#define PCI_DRIVER_NAME		"snvme"

MODULE_IMPORT_NS(NVME_TARGET_PASSTHRU);

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
static struct list ctrl_list;


/* List of mapped host memory */
static struct list host_list;


/* List of mapped device memory */
static struct list device_list;

/* List of mapped device queue memory */
static struct list device_queue_list;

/* Number of devices */
static int max_num_ctrls = 64;
module_param(max_num_ctrls, int, 0);
MODULE_PARM_DESC(max_num_ctrls, "Number of controller devices");

static int curr_ctrls = 0;

#define SQ_SIZE(q)	((q)->q_depth << (q)->sqes)
#define CQ_SIZE(q)	((q)->q_depth * sizeof(struct nvme_completion))
#define TO_PCI_DEV(addr) pci_get_domain_bus_and_slot(addr.domain, addr.bus, PCI_DEVFN(addr.slot, addr.func))

#define SGES_PER_PAGE	(NVME_CTRL_PAGE_SIZE / sizeof(struct nvme_sgl_desc))

int nvme_num;
int gpu_num;

/*
 * These can be higher, but we need to ensure that any command doesn't
 * require an sg allocation that needs more than a page of data.
 */
#define NVME_MAX_KB_SZ	8192
#define NVME_MAX_SEGS	128
#define NVME_MAX_NR_ALLOCATIONS	5

static int use_threaded_interrupts;
module_param(use_threaded_interrupts, int, 0444);

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

#define NVME_PCI_MIN_QUEUE_SIZE 2
#define NVME_PCI_MAX_QUEUE_SIZE 4095
static int io_queue_depth_set(const char *val, const struct kernel_param *kp);
static const struct kernel_param_ops io_queue_depth_ops = {
	.set = io_queue_depth_set,
	.get = param_get_uint,
};

static unsigned int io_queue_depth = 1024;
module_param_cb(io_queue_depth, &io_queue_depth_ops, &io_queue_depth, 0644);
MODULE_PARM_DESC(io_queue_depth, "set io queue depth, should >= 2 and < 4096");

static int io_queue_count_set(const char *val, const struct kernel_param *kp)
{
	unsigned int n;
	int ret;

	ret = kstrtouint(val, 10, &n);
	if (ret != 0 || n > num_possible_cpus())
		return -EINVAL;
	return param_set_uint(val, kp);
}

static const struct kernel_param_ops io_queue_count_ops = {
	.set = io_queue_count_set,
	.get = param_get_uint,
};

static unsigned int write_queues;
module_param_cb(write_queues, &io_queue_count_ops, &write_queues, 0644);
MODULE_PARM_DESC(write_queues,
	"Number of queues to use for writes. If not set, reads and writes "
	"will share a queue set.");

static unsigned int poll_queues;
module_param_cb(poll_queues, &io_queue_count_ops, &poll_queues, 0644);
MODULE_PARM_DESC(poll_queues, "Number of queues to use for polled IO.");

static bool noacpi;
module_param(noacpi, bool, 0444);
MODULE_PARM_DESC(noacpi, "disable acpi bios quirks");

struct nvme_dev;
struct nvme_queue;

static void nvme_dev_disable(struct nvme_dev *dev, bool shutdown);
static void nvme_delete_io_queues(struct nvme_dev *dev);
static void nvme_update_attrs(struct nvme_dev *dev);

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
	unsigned online_user_queues;
	unsigned user_start_qid;
	unsigned max_qid;
	unsigned io_queues[HCTX_MAX_TYPES];
	unsigned int num_vecs;
	u32 q_depth;
	int io_sqes;
	u32 db_stride;
	void __iomem *bar;
	unsigned long bar_mapped_size;
	struct mutex shutdown_lock;
	bool subsystem;
	u64 cmb_size;
	bool cmb_use_sqes;
	u32 cmbsz;
	u32 cmbloc;
	struct nvme_ctrl ctrl;
	u32 last_ps;
	bool hmb;

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
	/* Controller grant and the kernel/user queue budget split. */
	unsigned int ctrl_max_io_queues;
	unsigned int cap_kernel_ioq;
};

static int io_queue_depth_set(const char *val, const struct kernel_param *kp)
{
	return param_set_uint_minmax(val, kp, NVME_PCI_MIN_QUEUE_SIZE,
			NVME_PCI_MAX_QUEUE_SIZE);
}

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
	struct nvme_completion *cqes;
	dma_addr_t sq_dma_addr;
	dma_addr_t cq_dma_addr;
	u32 __iomem *q_db;
	u32 q_depth;
	u16 cq_vector;
	u16 sq_tail;
	u16 last_sq_tail;
	u16 cq_head;
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

union nvme_descriptor {
	struct nvme_sgl_desc	*sg_list;
	__le64			*prp_list;
};

/*
 * The nvme_iod describes the data in an I/O.
 *
 * The sg pointer contains the list of PRP/SGL chunk allocations in addition
 * to the actual struct scatterlist.
 */
struct nvme_iod {
	struct nvme_request req;
	struct nvme_command cmd;
	bool aborted;
	s8 nr_allocations;	/* PRP list pool allocations. 0 means small
				   pool in use */
	unsigned int dma_len;	/* length of single DMA segment mapping */
	dma_addr_t first_dma;
	dma_addr_t meta_dma;
	struct sg_table sgt;
	union nvme_descriptor list[NVME_MAX_NR_ALLOCATIONS];
};

static inline unsigned int nvme_dbbuf_size(struct nvme_dev *dev)
{
	return dev->nr_allocated_queues * 8 * dev->db_stride;
}

static void nvme_dbbuf_dma_alloc(struct nvme_dev *dev)
{
	unsigned int mem_size = nvme_dbbuf_size(dev);

	if (!(dev->ctrl.oacs & NVME_CTRL_OACS_DBBUF_SUPP))
		return;

	if (dev->dbbuf_dbs) {
		/*
		 * Clear the dbbuf memory so the driver doesn't observe stale
		 * values from the previous instantiation.
		 */
		memset(dev->dbbuf_dbs, 0, mem_size);
		memset(dev->dbbuf_eis, 0, mem_size);
		return;
	}

	dev->dbbuf_dbs = dma_alloc_coherent(dev->dev, mem_size,
					    &dev->dbbuf_dbs_dma_addr,
					    GFP_KERNEL);
	if (!dev->dbbuf_dbs)
		goto fail;
	dev->dbbuf_eis = dma_alloc_coherent(dev->dev, mem_size,
					    &dev->dbbuf_eis_dma_addr,
					    GFP_KERNEL);
	if (!dev->dbbuf_eis)
		goto fail_free_dbbuf_dbs;
	return;

fail_free_dbbuf_dbs:
	dma_free_coherent(dev->dev, mem_size, dev->dbbuf_dbs,
			  dev->dbbuf_dbs_dma_addr);
	dev->dbbuf_dbs = NULL;
fail:
	dev_warn(dev->dev, "unable to allocate dma for dbbuf\n");
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
	printk("nvme_dbbuf_init\n");
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
	struct nvme_command c = { };
	unsigned int i;

	if (!dev->dbbuf_dbs)
		return;

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
static int nvme_pci_npages_prp(void)
{
	unsigned max_bytes = (NVME_MAX_KB_SZ * 1024) + NVME_CTRL_PAGE_SIZE;
	unsigned nprps = DIV_ROUND_UP(max_bytes, NVME_CTRL_PAGE_SIZE);
	return DIV_ROUND_UP(8 * nprps, NVME_CTRL_PAGE_SIZE - 8);
}

static int nvme_admin_init_hctx(struct blk_mq_hw_ctx *hctx, void *data,
				unsigned int hctx_idx)
{
	struct nvme_dev *dev = to_nvme_dev(data);
	struct nvme_queue *nvmeq = &dev->queues[0];

	WARN_ON(hctx_idx != 0);
	WARN_ON(dev->admin_tagset.tags[0] != hctx->tags);

	hctx->driver_data = nvmeq;
	return 0;
}

static int nvme_init_hctx(struct blk_mq_hw_ctx *hctx, void *data,
			  unsigned int hctx_idx)
{
	struct nvme_dev *dev = to_nvme_dev(data);
	struct nvme_queue *nvmeq = &dev->queues[hctx_idx + 1];

	WARN_ON(dev->tagset.tags[hctx_idx] != hctx->tags);
	hctx->driver_data = nvmeq;
	return 0;
}

static int nvme_pci_init_request(struct blk_mq_tag_set *set,
		struct request *req, unsigned int hctx_idx,
		unsigned int numa_node)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);

	nvme_req(req)->ctrl = set->driver_data;
	nvme_req(req)->cmd = &iod->cmd;
	return 0;
}

static int queue_irq_offset(struct nvme_dev *dev)
{
	/* if we have more than 1 vec, admin queue offsets us by 1 */
	if (dev->num_vecs > 1)
		return 1;

	return 0;
}

static void nvme_pci_map_queues(struct blk_mq_tag_set *set)
{
	struct nvme_dev *dev = to_nvme_dev(set->driver_data);
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
		if (i != HCTX_TYPE_POLL && offset)
			blk_mq_pci_map_queues(map, to_pci_dev(dev->dev), offset);
		else
			blk_mq_map_queues(map);
		qoff += map->nr_queues;
		offset += map->nr_queues;
	}
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

static inline void nvme_sq_copy_cmd(struct nvme_queue *nvmeq,
				    struct nvme_command *cmd)
{
	memcpy(nvmeq->sq_cmds + (nvmeq->sq_tail << nvmeq->sqes),
		absolute_pointer(cmd), sizeof(*cmd));
	if (++nvmeq->sq_tail == nvmeq->q_depth)
		nvmeq->sq_tail = 0;
}

static void nvme_commit_rqs(struct blk_mq_hw_ctx *hctx)
{
	struct nvme_queue *nvmeq = hctx->driver_data;

	spin_lock(&nvmeq->sq_lock);
	if (nvmeq->sq_tail != nvmeq->last_sq_tail)
		nvme_write_sq_db(nvmeq, true);
	spin_unlock(&nvmeq->sq_lock);
}

static inline bool nvme_pci_use_sgls(struct nvme_dev *dev, struct request *req,
				     int nseg)
{
	struct nvme_queue *nvmeq = req->mq_hctx->driver_data;
	unsigned int avg_seg_size;

	avg_seg_size = DIV_ROUND_UP(blk_rq_payload_bytes(req), nseg);

	if (!nvme_ctrl_sgl_supported(&dev->ctrl))
		return false;
	if (!nvmeq->qid)
		return false;
	if (!sgl_threshold || avg_seg_size < sgl_threshold)
		return false;
	return true;
}

static void nvme_free_prps(struct nvme_dev *dev, struct request *req)
{
	const int last_prp = NVME_CTRL_PAGE_SIZE / sizeof(__le64) - 1;
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	dma_addr_t dma_addr = iod->first_dma;
	int i;

	for (i = 0; i < iod->nr_allocations; i++) {
		__le64 *prp_list = iod->list[i].prp_list;
		dma_addr_t next_dma_addr = le64_to_cpu(prp_list[last_prp]);

		dma_pool_free(dev->prp_page_pool, prp_list, dma_addr);
		dma_addr = next_dma_addr;
	}
}

static void nvme_unmap_data(struct nvme_dev *dev, struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);

	if (iod->dma_len) {
		dma_unmap_page(dev->dev, iod->first_dma, iod->dma_len,
			       rq_dma_dir(req));
		return;
	}

	WARN_ON_ONCE(!iod->sgt.nents);

	dma_unmap_sgtable(dev->dev, &iod->sgt, rq_dma_dir(req), 0);

	if (iod->nr_allocations == 0)
		dma_pool_free(dev->prp_small_pool, iod->list[0].sg_list,
			      iod->first_dma);
	else if (iod->nr_allocations == 1)
		dma_pool_free(dev->prp_page_pool, iod->list[0].sg_list,
			      iod->first_dma);
	else
		nvme_free_prps(dev, req);
	mempool_free(iod->sgt.sgl, dev->iod_mempool);
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
	struct scatterlist *sg = iod->sgt.sgl;
	int dma_len = sg_dma_len(sg);
	u64 dma_addr = sg_dma_address(sg);
	int offset = dma_addr & (NVME_CTRL_PAGE_SIZE - 1);
	__le64 *prp_list;
	dma_addr_t prp_dma;
	int nprps, i;

	length -= (NVME_CTRL_PAGE_SIZE - offset);
	if (length <= 0) {
		iod->first_dma = 0;
		goto done;
	}

	dma_len -= (NVME_CTRL_PAGE_SIZE - offset);
	if (dma_len) {
		dma_addr += (NVME_CTRL_PAGE_SIZE - offset);
	} else {
		sg = sg_next(sg);
		dma_addr = sg_dma_address(sg);
		dma_len = sg_dma_len(sg);
	}

	if (length <= NVME_CTRL_PAGE_SIZE) {
		iod->first_dma = dma_addr;
		goto done;
	}

	nprps = DIV_ROUND_UP(length, NVME_CTRL_PAGE_SIZE);
	if (nprps <= (256 / 8)) {
		pool = dev->prp_small_pool;
		iod->nr_allocations = 0;
	} else {
		pool = dev->prp_page_pool;
		iod->nr_allocations = 1;
	}

	prp_list = dma_pool_alloc(pool, GFP_ATOMIC, &prp_dma);
	if (!prp_list) {
		iod->nr_allocations = -1;
		return BLK_STS_RESOURCE;
	}
	iod->list[0].prp_list = prp_list;
	iod->first_dma = prp_dma;
	i = 0;
	for (;;) {
		if (i == NVME_CTRL_PAGE_SIZE >> 3) {
			__le64 *old_prp_list = prp_list;
			prp_list = dma_pool_alloc(pool, GFP_ATOMIC, &prp_dma);
			if (!prp_list)
				goto free_prps;
			iod->list[iod->nr_allocations++].prp_list = prp_list;
			prp_list[0] = old_prp_list[i - 1];
			old_prp_list[i - 1] = cpu_to_le64(prp_dma);
			i = 1;
		}
		prp_list[i++] = cpu_to_le64(dma_addr);
		dma_len -= NVME_CTRL_PAGE_SIZE;
		dma_addr += NVME_CTRL_PAGE_SIZE;
		length -= NVME_CTRL_PAGE_SIZE;
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
	cmnd->dptr.prp1 = cpu_to_le64(sg_dma_address(iod->sgt.sgl));
	cmnd->dptr.prp2 = cpu_to_le64(iod->first_dma);
	return BLK_STS_OK;
free_prps:
	nvme_free_prps(dev, req);
	return BLK_STS_RESOURCE;
bad_sgl:
	WARN(DO_ONCE(nvme_print_sgl, iod->sgt.sgl, iod->sgt.nents),
			"Invalid SGL for payload:%d nents:%d\n",
			blk_rq_payload_bytes(req), iod->sgt.nents);
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
	sge->length = cpu_to_le32(entries * sizeof(*sge));
	sge->type = NVME_SGL_FMT_LAST_SEG_DESC << 4;
}

static blk_status_t nvme_pci_setup_sgls(struct nvme_dev *dev,
		struct request *req, struct nvme_rw_command *cmd)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct dma_pool *pool;
	struct nvme_sgl_desc *sg_list;
	struct scatterlist *sg = iod->sgt.sgl;
	unsigned int entries = iod->sgt.nents;
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
		iod->nr_allocations = 0;
	} else {
		pool = dev->prp_page_pool;
		iod->nr_allocations = 1;
	}

	sg_list = dma_pool_alloc(pool, GFP_ATOMIC, &sgl_dma);
	if (!sg_list) {
		iod->nr_allocations = -1;
		return BLK_STS_RESOURCE;
	}

	iod->list[0].sg_list = sg_list;
	iod->first_dma = sgl_dma;

	nvme_pci_sgl_set_seg(&cmd->dptr.sgl, sgl_dma, entries);
	do {
		nvme_pci_sgl_set_data(&sg_list[i++], sg);
		sg = sg_next(sg);
	} while (--entries > 0);

	return BLK_STS_OK;
}

static blk_status_t nvme_setup_prp_simple(struct nvme_dev *dev,
		struct request *req, struct nvme_rw_command *cmnd,
		struct bio_vec *bv)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	unsigned int offset = bv->bv_offset & (NVME_CTRL_PAGE_SIZE - 1);
	unsigned int first_prp_len = NVME_CTRL_PAGE_SIZE - offset;

	iod->first_dma = dma_map_bvec(dev->dev, bv, rq_dma_dir(req), 0);
	if (dma_mapping_error(dev->dev, iod->first_dma))
		return BLK_STS_RESOURCE;
	iod->dma_len = bv->bv_len;

	cmnd->dptr.prp1 = cpu_to_le64(iod->first_dma);
	if (bv->bv_len > first_prp_len)
		cmnd->dptr.prp2 = cpu_to_le64(iod->first_dma + first_prp_len);
	else
		cmnd->dptr.prp2 = 0;
	return BLK_STS_OK;
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
	return BLK_STS_OK;
}

static blk_status_t nvme_map_data(struct nvme_dev *dev, struct request *req,
		struct nvme_command *cmnd)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	blk_status_t ret = BLK_STS_RESOURCE;
	int rc;

	if (blk_rq_nr_phys_segments(req) == 1) {
		struct nvme_queue *nvmeq = req->mq_hctx->driver_data;
		struct bio_vec bv = req_bvec(req);

		if (!is_pci_p2pdma_page(bv.bv_page)) {
			if (bv.bv_offset + bv.bv_len <= NVME_CTRL_PAGE_SIZE * 2)
				return nvme_setup_prp_simple(dev, req,
							     &cmnd->rw, &bv);

			if (nvmeq->qid && sgl_threshold &&
			    nvme_ctrl_sgl_supported(&dev->ctrl))
				return nvme_setup_sgl_simple(dev, req,
							     &cmnd->rw, &bv);
		}
	}

	iod->dma_len = 0;
	iod->sgt.sgl = mempool_alloc(dev->iod_mempool, GFP_ATOMIC);
	if (!iod->sgt.sgl)
		return BLK_STS_RESOURCE;
	sg_init_table(iod->sgt.sgl, blk_rq_nr_phys_segments(req));
	iod->sgt.orig_nents = blk_rq_map_sg(req->q, req, iod->sgt.sgl);
	if (!iod->sgt.orig_nents)
		goto out_free_sg;

	rc = dma_map_sgtable(dev->dev, &iod->sgt, rq_dma_dir(req),
			     DMA_ATTR_NO_WARN);
	if (rc) {
		if (rc == -EREMOTEIO)
			ret = BLK_STS_TARGET;
		goto out_free_sg;
	}

	if (nvme_pci_use_sgls(dev, req, iod->sgt.nents))
		ret = nvme_pci_setup_sgls(dev, req, &cmnd->rw);
	else
		ret = nvme_pci_setup_prps(dev, req, &cmnd->rw);
	if (ret != BLK_STS_OK)
		goto out_unmap_sg;
	return BLK_STS_OK;

out_unmap_sg:
	dma_unmap_sgtable(dev->dev, &iod->sgt, rq_dma_dir(req), 0);
out_free_sg:
	mempool_free(iod->sgt.sgl, dev->iod_mempool);
	return ret;
}

static blk_status_t nvme_map_metadata(struct nvme_dev *dev, struct request *req,
		struct nvme_command *cmnd)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct bio_vec bv = rq_integrity_vec(req);

	iod->meta_dma = dma_map_bvec(dev->dev, &bv,
			rq_dma_dir(req), 0);
	if (dma_mapping_error(dev->dev, iod->meta_dma))
		return BLK_STS_IOERR;
	cmnd->rw.metadata = cpu_to_le64(iod->meta_dma);
	return BLK_STS_OK;
}

static blk_status_t nvme_prep_rq(struct nvme_dev *dev, struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	blk_status_t ret;

	iod->aborted = false;
	iod->nr_allocations = -1;
	iod->sgt.nents = 0;

	ret = snvme_setup_cmd(req->q->queuedata, req);
	if (ret)
		return ret;

	if (blk_rq_nr_phys_segments(req)) {
		ret = nvme_map_data(dev, req, &iod->cmd);
		if (ret)
			goto out_free_cmd;
	}

	if (blk_integrity_rq(req)) {
		ret = nvme_map_metadata(dev, req, &iod->cmd);
		if (ret)
			goto out_unmap_data;
	}

	nvme_start_request(req);
	return BLK_STS_OK;
out_unmap_data:
	nvme_unmap_data(dev, req);
out_free_cmd:
	snvme_cleanup_cmd(req);
	return ret;
}

/*
 * NOTE: ns is NULL when called on the admin queue.
 */
static blk_status_t nvme_queue_rq(struct blk_mq_hw_ctx *hctx,
			 const struct blk_mq_queue_data *bd)
{
	struct nvme_queue *nvmeq = hctx->driver_data;
	struct nvme_dev *dev = nvmeq->dev;
	struct request *req = bd->rq;
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	blk_status_t ret;

	/*
	 * We should not need to do this, but we're still using this to
	 * ensure we can drain requests on a dying queue.
	 */
	if (unlikely(!test_bit(NVMEQ_ENABLED, &nvmeq->flags)))
		return BLK_STS_IOERR;

	if (unlikely(!nvme_check_ready(&dev->ctrl, req, true)))
		return snvme_fail_nonready_command(&dev->ctrl, req);

	ret = nvme_prep_rq(dev, req);
	if (unlikely(ret))
		return ret;
	spin_lock(&nvmeq->sq_lock);
	nvme_sq_copy_cmd(nvmeq, &iod->cmd);
	nvme_write_sq_db(nvmeq, bd->last);
	spin_unlock(&nvmeq->sq_lock);
	return BLK_STS_OK;
}

static void nvme_submit_cmds(struct nvme_queue *nvmeq, struct request **rqlist)
{
	spin_lock(&nvmeq->sq_lock);
	while (!rq_list_empty(*rqlist)) {
		struct request *req = rq_list_pop(rqlist);
		struct nvme_iod *iod = blk_mq_rq_to_pdu(req);

		nvme_sq_copy_cmd(nvmeq, &iod->cmd);
	}
	nvme_write_sq_db(nvmeq, true);
	spin_unlock(&nvmeq->sq_lock);
}

static bool nvme_prep_rq_batch(struct nvme_queue *nvmeq, struct request *req)
{
	/*
	 * We should not need to do this, but we're still using this to
	 * ensure we can drain requests on a dying queue.
	 */
	if (unlikely(!test_bit(NVMEQ_ENABLED, &nvmeq->flags)))
		return false;
	if (unlikely(!nvme_check_ready(&nvmeq->dev->ctrl, req, true)))
		return false;

	return nvme_prep_rq(nvmeq->dev, req) == BLK_STS_OK;
}

static void nvme_queue_rqs(struct request **rqlist)
{
	struct request *req, *next, *prev = NULL;
	struct request *requeue_list = NULL;

	rq_list_for_each_safe(rqlist, req, next) {
		struct nvme_queue *nvmeq = req->mq_hctx->driver_data;

		if (!nvme_prep_rq_batch(nvmeq, req)) {
			/* detach 'req' and add to remainder list */
			rq_list_move(rqlist, &requeue_list, req, prev);

			req = prev;
			if (!req)
				continue;
		}

		if (!next || req->mq_hctx != next->mq_hctx) {
			/* detach rest of list, and submit */
			req->rq_next = NULL;
			nvme_submit_cmds(nvmeq, rqlist);
			*rqlist = next;
			prev = NULL;
		} else
			prev = req;
	}

	*rqlist = requeue_list;
}

static __always_inline void nvme_pci_unmap_rq(struct request *req)
{
	struct nvme_queue *nvmeq = req->mq_hctx->driver_data;
	struct nvme_dev *dev = nvmeq->dev;

	if (blk_integrity_rq(req)) {
	        struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
		struct bio_vec bv = rq_integrity_vec(req);

		dma_unmap_page(dev->dev, iod->meta_dma,
			       bv.bv_len, rq_dma_dir(req));
	}

	if (blk_rq_nr_phys_segments(req))
		nvme_unmap_data(dev, req);
}

static void nvme_pci_complete_rq(struct request *req)
{
	nvme_pci_unmap_rq(req);
	snvme_complete_rq(req);
}

static void nvme_pci_complete_batch(struct io_comp_batch *iob)
{
	nvme_complete_batch(iob, nvme_pci_unmap_rq);
}

/* We read the CQE phase first to check if the rest of the entry is valid */
static inline bool nvme_cqe_pending(struct nvme_queue *nvmeq)
{
	struct nvme_completion *hcqe = &nvmeq->cqes[nvmeq->cq_head];

	return (le16_to_cpu(READ_ONCE(hcqe->status)) & 1) == nvmeq->cq_phase;
}

static inline void nvme_ring_cq_doorbell(struct nvme_queue *nvmeq)
{
	u16 head = nvmeq->cq_head;

	if (nvme_dbbuf_update_and_check_event(head, nvmeq->dbbuf_cq_db,
					      nvmeq->dbbuf_cq_ei))
		writel(head, nvmeq->q_db + nvmeq->dev->db_stride);
}

static inline struct blk_mq_tags *nvme_queue_tagset(struct nvme_queue *nvmeq)
{
	if (!nvmeq->qid)
		return nvmeq->dev->admin_tagset.tags[0];
	return nvmeq->dev->tagset.tags[nvmeq->qid - 1];
}

static inline void nvme_handle_cqe(struct nvme_queue *nvmeq,
				   struct io_comp_batch *iob, u16 idx)
{
	struct nvme_completion *cqe = &nvmeq->cqes[idx];
	__u16 command_id = READ_ONCE(cqe->command_id);
	struct request *req;

	/*
	 * AEN requests are special as they don't time out and can
	 * survive any kind of queue freeze and often don't respond to
	 * aborts.  We don't even bother to allocate a struct request
	 * for them but rather special case them here.
	 */
	if (unlikely(nvme_is_aen_req(nvmeq->qid, command_id))) {
		snvme_complete_async_event(&nvmeq->dev->ctrl,
				cqe->status, &cqe->result);
		return;
	}

	req = nvme_find_rq(nvme_queue_tagset(nvmeq), command_id);
	if (unlikely(!req)) {
		dev_warn(nvmeq->dev->ctrl.device,
			"invalid id %d completed on queue %d\n",
			command_id, le16_to_cpu(cqe->sq_id));
		return;
	}

	if (!nvme_try_complete_req(req, cqe->status, cqe->result) &&
	    !blk_mq_add_to_batch(req, iob, nvme_req(req)->status,
					nvme_pci_complete_batch))
		nvme_pci_complete_rq(req);
}

static inline void nvme_update_cq_head(struct nvme_queue *nvmeq)
{
	u32 tmp = nvmeq->cq_head + 1;

	if (tmp == nvmeq->q_depth) {
		nvmeq->cq_head = 0;
		nvmeq->cq_phase ^= 1;
	} else {
		nvmeq->cq_head = tmp;
	}
}

static inline int nvme_poll_cq(struct nvme_queue *nvmeq,
			       struct io_comp_batch *iob)
{
	int found = 0;

	while (nvme_cqe_pending(nvmeq)) {
		found++;
		/*
		 * load-load control dependency between phase and the rest of
		 * the cqe requires a full read memory barrier
		 */
		dma_rmb();
		nvme_handle_cqe(nvmeq, iob, nvmeq->cq_head);
		nvme_update_cq_head(nvmeq);
	}

	if (found)
		nvme_ring_cq_doorbell(nvmeq);
	return found;
}

static irqreturn_t nvme_irq(int irq, void *data)
{
	struct nvme_queue *nvmeq = data;
	DEFINE_IO_COMP_BATCH(iob);

	if (nvme_poll_cq(nvmeq, &iob)) {
		if (!rq_list_empty(iob.req_list))
			nvme_pci_complete_batch(&iob);
		return IRQ_HANDLED;
	}
	/* A GPU poller may already have consumed this queue's CQEs. */
	return IRQ_HANDLED;
}

static irqreturn_t nvme_irq_check(int irq, void *data)
{
	struct nvme_queue *nvmeq = data;

	if (nvme_cqe_pending(nvmeq))
		return IRQ_WAKE_THREAD;
	return IRQ_NONE;
}

/*
 * Poll for completions for any interrupt driven queue
 * Can be called from any context.
 */
static void nvme_poll_irqdisable(struct nvme_queue *nvmeq)
{
	struct pci_dev *pdev = to_pci_dev(nvmeq->dev->dev);

	WARN_ON_ONCE(test_bit(NVMEQ_POLLED, &nvmeq->flags));

	disable_irq(pci_irq_vector(pdev, nvmeq->cq_vector));
	nvme_poll_cq(nvmeq, NULL);
	enable_irq(pci_irq_vector(pdev, nvmeq->cq_vector));
}

static int nvme_poll(struct blk_mq_hw_ctx *hctx, struct io_comp_batch *iob)
{
	struct nvme_queue *nvmeq = hctx->driver_data;
	bool found;

	if (!nvme_cqe_pending(nvmeq))
		return 0;

	spin_lock(&nvmeq->cq_poll_lock);
	found = nvme_poll_cq(nvmeq, iob);
	spin_unlock(&nvmeq->cq_poll_lock);

	return found;
}

static void nvme_pci_submit_async_event(struct nvme_ctrl *ctrl)
{
	struct nvme_dev *dev = to_nvme_dev(ctrl);
	struct nvme_queue *nvmeq = &dev->queues[0];
	struct nvme_command c = { };

	c.common.opcode = nvme_admin_async_event;
	c.common.command_id = NVME_AQ_BLK_MQ_DEPTH;

	spin_lock(&nvmeq->sq_lock);
	nvme_sq_copy_cmd(nvmeq, &c);
	nvme_write_sq_db(nvmeq, true);
	spin_unlock(&nvmeq->sq_lock);
}

static int adapter_delete_queue(struct nvme_dev *dev, u8 opcode, u16 id)
{
	struct nvme_command c = { };

	c.delete_queue.opcode = opcode;
	c.delete_queue.qid = cpu_to_le16(id);

	return snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
}

static int adapter_alloc_cq(struct nvme_dev *dev, u16 qid,
		struct nvme_queue *nvmeq, s16 vector)
{
	struct nvme_command c = { };
	int flags = NVME_QUEUE_PHYS_CONTIG;

	if (!test_bit(NVMEQ_POLLED, &nvmeq->flags))
		flags |= NVME_CQ_IRQ_ENABLED;

	/*
	 * Note: we (ab)use the fact that the prp fields survive if no data
	 * is attached to the request.
	 */
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
	struct nvme_command c = { };
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
	c.create_sq.opcode = nvme_admin_create_sq;
	c.create_sq.prp1 = cpu_to_le64(nvmeq->sq_dma_addr);
	c.create_sq.sqid = cpu_to_le16(qid);
	c.create_sq.qsize = cpu_to_le16(nvmeq->q_depth - 1);
	c.create_sq.sq_flags = cpu_to_le16(flags);
	c.create_sq.cqid = cpu_to_le16(qid);
	// printk("adapter_alloc_sq q_depth is %u\n",nvmeq->q_depth);
	// printk("adapter_alloc_sq qid is,cq id %u\n",c.create_sq.sqid,c.create_sq.cqid);
	return snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
}

static int adapter_alloc_cq_user(struct nvme_dev *dev, struct map* q_map,int qid)
{
	struct nvme_command c = { };
	int flags = NVME_QUEUE_PHYS_CONTIG;

	/*
	 * Note: we (ab)use the fact that the prp fields survive if no data
	 * is attached to the request.
	 */
	c.create_cq.opcode = nvme_admin_create_cq;
	c.create_cq.prp1 = cpu_to_le64(q_map->addrs[0]);
	c.create_cq.cqid = cpu_to_le16(qid);
	c.create_cq.qsize = cpu_to_le16(dev->q_depth-1);
	c.create_cq.cq_flags = cpu_to_le16(flags);
	c.create_cq.irq_vector = cpu_to_le16(0);
	// printk("adapter_alloc_cq_user qid is %u, addr is %lx,q depth is %d,cq_flags is %u\n",qid,q_map->addrs[0],c.create_cq.qsize,c.create_cq.cq_flags);
	// printk("adapter_alloc_cq qid is %u\n",c.create_cq.cq_flags);
	return snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
}

static int adapter_alloc_sq_user(struct nvme_dev *dev, struct map* q_map,int qid)
{

	struct nvme_command c = { };
	int flags = NVME_QUEUE_PHYS_CONTIG;

	/*
	 * Some drives have a bug that auto-enables WRRU if MEDIUM isn't
	 * set. Since URGENT priority is zeroes, it makes all queues
	 * URGENT.
	 */

	/*
	 * Note: we (ab)use the fact that the prp fields survive if no data
	 * is attached to the request.
	 */
	c.create_sq.opcode = nvme_admin_create_sq;
	c.create_sq.prp1 = cpu_to_le64(q_map->addrs[0]);
	c.create_sq.sqid = cpu_to_le16(qid);
	c.create_sq.qsize = cpu_to_le16(dev->q_depth-1);
	c.create_sq.sq_flags = cpu_to_le16(flags);
	c.create_sq.cqid = cpu_to_le16(qid);
	// printk("adapter_alloc_sq_user qid is %u, addr is %lx,q depth is %d\n",qid,q_map->addrs[0],dev->q_depth - 1);
	// printk("adapter_alloc_sq_user qid is,cq id %u\n",c.create_sq.sqid,c.create_sq.cqid);
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

static enum rq_end_io_ret abort_endio(struct request *req, blk_status_t error)
{
	struct nvme_queue *nvmeq = req->mq_hctx->driver_data;

	dev_warn(nvmeq->dev->ctrl.device,
		 "Abort status: 0x%x", nvme_req(req)->status);
	atomic_inc(&nvmeq->dev->ctrl.abort_limit);
	blk_mq_free_request(req);
	return RQ_END_IO_NONE;
}

static bool nvme_should_reset(struct nvme_dev *dev, u32 csts)
{
	/* If true, indicates loss of adapter communication, possibly by a
	 * NVMe Subsystem reset.
	 */
	bool nssro = dev->subsystem && (csts & NVME_CSTS_NSSRO);

	/* If there is a reset/reinit ongoing, we shouldn't reset again. */
	switch (nvme_ctrl_state(&dev->ctrl)) {
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

	if (csts != ~0)
		return;

	dev_warn(dev->ctrl.device,
		 "Does your device have a faulty power saving mode enabled?\n");
	dev_warn(dev->ctrl.device,
		 "Try \"nvme_core.s_default_ps_max_latency_us=0 pcie_aspm=off\" and report a bug\n");
}

static enum blk_eh_timer_return nvme_timeout(struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct nvme_queue *nvmeq = req->mq_hctx->driver_data;
	struct nvme_dev *dev = nvmeq->dev;
	struct request *abort_req;
	struct nvme_command cmd = { };
	u32 csts = readl(dev->bar + NVME_REG_CSTS);
	u8 opcode;

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
		goto disable;
	}

	/*
	 * Did we miss an interrupt?
	 */
	if (test_bit(NVMEQ_POLLED, &nvmeq->flags))
		nvme_poll(req->mq_hctx, NULL);
	else
		nvme_poll_irqdisable(nvmeq);

	if (blk_mq_rq_state(req) != MQ_RQ_IN_FLIGHT) {
		dev_warn(dev->ctrl.device,
			 "I/O tag %d (%04x) QID %d timeout, completion polled\n",
			 req->tag, nvme_cid(req), nvmeq->qid);
		return BLK_EH_DONE;
	}

	/*
	 * Shutdown immediately if controller times out while starting. The
	 * reset work will see the pci device disabled when it gets the forced
	 * cancellation error. All outstanding requests are completed on
	 * shutdown, so we return BLK_EH_DONE.
	 */
	switch (nvme_ctrl_state(&dev->ctrl)) {
	case NVME_CTRL_CONNECTING:
		snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_DELETING);
		fallthrough;
	case NVME_CTRL_DELETING:
		dev_warn_ratelimited(dev->ctrl.device,
			 "I/O tag %d (%04x) QID %d timeout, disable controller\n",
			 req->tag, nvme_cid(req), nvmeq->qid);
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
	opcode = nvme_req(req)->cmd->common.opcode;
	if (!nvmeq->qid || iod->aborted) {
		dev_warn(dev->ctrl.device,
			 "I/O tag %d (%04x) opcode %#x (%s) QID %d timeout, reset controller\n",
			 req->tag, nvme_cid(req), opcode,
			 nvme_opcode_str(nvmeq->qid, opcode), nvmeq->qid);
		nvme_req(req)->flags |= NVME_REQ_CANCELLED;
		goto disable;
	}

	if (atomic_dec_return(&dev->ctrl.abort_limit) < 0) {
		atomic_inc(&dev->ctrl.abort_limit);
		return BLK_EH_RESET_TIMER;
	}
	iod->aborted = true;

	cmd.abort.opcode = nvme_admin_abort_cmd;
	cmd.abort.cid = nvme_cid(req);
	cmd.abort.sqid = cpu_to_le16(nvmeq->qid);

	dev_warn(nvmeq->dev->ctrl.device,
		 "I/O tag %d (%04x) opcode %#x (%s) QID %d timeout, aborting req_op:%s(%u) size:%u\n",
		 req->tag, nvme_cid(req), opcode, snvme_get_opcode_str(opcode),
		 nvmeq->qid, blk_op_str(req_op(req)), req_op(req),
		 blk_rq_bytes(req));

	abort_req = blk_mq_alloc_request(dev->ctrl.admin_q, nvme_req_op(&cmd),
					 BLK_MQ_REQ_NOWAIT);
	if (IS_ERR(abort_req)) {
		atomic_inc(&dev->ctrl.abort_limit);
		return BLK_EH_RESET_TIMER;
	}
	snvme_init_request(abort_req, &cmd);

	abort_req->end_io = abort_endio;
	abort_req->end_io_data = NULL;
	blk_execute_rq_nowait(abort_req, false);

	/*
	 * The aborted req will be completed on receiving the abort req.
	 * We enable the timer again. If hit twice, it'll cause a device reset,
	 * as the device then is in a faulty state.
	 */
	return BLK_EH_RESET_TIMER;

disable:
	if (!snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_RESETTING))
		return BLK_EH_DONE;

	nvme_dev_disable(dev, false);
	if (snvme_try_sched_reset(&dev->ctrl))
		snvme_unquiesce_io_queues(&dev->ctrl);
	return BLK_EH_DONE;
}

// pay attention on it
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

static void nvme_suspend_queue(struct nvme_dev *dev, unsigned int qid)
{
	struct nvme_queue *nvmeq = &dev->queues[qid];

	if (!test_and_clear_bit(NVMEQ_ENABLED, &nvmeq->flags))
		return;

	/* ensure that nvme_queue_rq() sees NVMEQ_ENABLED cleared */
	mb();

	nvmeq->dev->online_queues--;
	if (!nvmeq->qid && nvmeq->dev->ctrl.admin_q)
		snvme_quiesce_admin_queue(&nvmeq->dev->ctrl);
	if (!test_and_clear_bit(NVMEQ_POLLED, &nvmeq->flags))
		pci_free_irq(to_pci_dev(dev->dev), nvmeq->cq_vector, nvmeq);
}

static void nvme_suspend_io_queues(struct nvme_dev *dev)
{
	int i;

	for (i = dev->ctrl.queue_count - 1; i > 0; i--)
		nvme_suspend_queue(dev, i);
}

/*
 * Called only on a device that has been disabled and after all other threads
 * that can check this device's completion queues have synced, except
 * nvme_poll(). This is the last chance for the driver to see a natural
 * completion before snvme_cancel_request() terminates all incomplete requests.
 */
static void nvme_reap_pending_cqes(struct nvme_dev *dev)
{
	int i;

	for (i = dev->ctrl.queue_count - 1; i > 0; i--) {
		spin_lock(&dev->queues[i].cq_poll_lock);
		nvme_poll_cq(&dev->queues[i], NULL);
		spin_unlock(&dev->queues[i].cq_poll_lock);
	}
}

static int nvme_cmb_qdepth(struct nvme_dev *dev, int nr_io_queues,
				int entry_size)
{
	int q_depth = dev->q_depth;
	unsigned q_size_aligned = roundup(q_depth * entry_size,
					  NVME_CTRL_PAGE_SIZE);

	if (q_size_aligned * nr_io_queues > dev->cmb_size) {
		u64 mem_per_q = div_u64(dev->cmb_size, nr_io_queues);

		mem_per_q = round_down(mem_per_q, NVME_CTRL_PAGE_SIZE);
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

/*allocate sqes and cqes, and retuen its dma addr, and allocate the qid*/
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
	// printk("q id is %u, ab addr is %lx",qid,nvmeq->q_db);
	memset((void *)nvmeq->cqes, 0, CQ_SIZE(nvmeq));
	nvme_dbbuf_init(dev, nvmeq, qid);
	dev->online_queues++;
	wmb(); /* ensure the first interrupt sees the initialization */
}

/*
 * Try getting shutdown_lock while setting up IO queues.
 */
static int nvme_setup_io_queues_trylock(struct nvme_dev *dev)
{
	/*
	 * Give up if the lock is being held by nvme_dev_disable.
	 */
	if (!mutex_trylock(&dev->shutdown_lock))
		return -ENODEV;

	/*
	 * Controller is in wrong state, fail early.
	 */
	if (nvme_ctrl_state(&dev->ctrl) != NVME_CTRL_CONNECTING) {
		mutex_unlock(&dev->shutdown_lock);
		return -ENODEV;
	}

	return 0;
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
	if (result)
		goto release_cq;

	nvmeq->cq_vector = vector;

	result = nvme_setup_io_queues_trylock(dev);
	if (result)
		return result;
	nvme_init_queue(nvmeq, qid);
	if (!polled) {
		result = queue_request_irq(nvmeq);
		if (result < 0)
			goto release_sq;
	}

	set_bit(NVMEQ_ENABLED, &nvmeq->flags);
	mutex_unlock(&dev->shutdown_lock);
	return result;

release_sq:
	dev->online_queues--;
	mutex_unlock(&dev->shutdown_lock);
	adapter_delete_sq(dev, qid);
release_cq:
	adapter_delete_cq(dev, qid);
	return result;
}

static const struct blk_mq_ops nvme_mq_admin_ops = {
	.queue_rq	= nvme_queue_rq,
	.complete	= nvme_pci_complete_rq,
	.init_hctx	= nvme_admin_init_hctx,
	.init_request	= nvme_pci_init_request,
	.timeout	= nvme_timeout,
};

static const struct blk_mq_ops nvme_mq_ops = {
	.queue_rq	= nvme_queue_rq,
	.queue_rqs	= nvme_queue_rqs,
	.complete	= nvme_pci_complete_rq,
	.commit_rqs	= nvme_commit_rqs,
	.init_hctx	= nvme_init_hctx,
	.init_request	= nvme_pci_init_request,
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
		snvme_unquiesce_admin_queue(&dev->ctrl);
		snvme_remove_admin_tag_set(&dev->ctrl);
	}
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
	printk("nvme_remap_bar bar size is %llx\n", pci_resource_len(pdev, 0));
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

	/*
	 * If the device has been passed off to us in an enabled state, just
	 * clear the enabled bit.  The spec says we should set the 'shutdown
	 * notification bits', but doing so may cause the device to complete
	 * commands to the admin queue ... and we don't know what memory that
	 * might be pointing at!
	 */
	result = snvme_disable_ctrl(&dev->ctrl, false);
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

	if (NVME_CAP_CMBS(dev->ctrl.cap))
		writel(NVME_CMBMSC_CRE, dev->bar + NVME_REG_CMBMSC);

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
	 * Tell the controller about the host side address mapping the CMB,
	 * and enable CMB decoding for the NVMe 1.4+ scheme:
	 */
	if (NVME_CAP_CMBS(dev->ctrl.cap)) {
		hi_lo_writeq(NVME_CMBMSC_CRE | NVME_CMBMSC_CMSE |
			     (pci_bus_address(pdev, bar) + offset),
			     dev->bar + NVME_REG_CMBMSC);
	}

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

	nvme_update_attrs(dev);
}

static int nvme_set_host_mem(struct nvme_dev *dev, u32 bits)
{
	u32 host_mem_size = dev->host_mem_size >> NVME_CTRL_PAGE_SHIFT;
	u64 dma_addr = dev->host_mem_descs_dma;
	struct nvme_command c = { };
	int ret;

	c.features.opcode	= nvme_admin_set_features;
	c.features.fid		= cpu_to_le32(NVME_FEAT_HOST_MEM_BUF);
	c.features.dword11	= cpu_to_le32(bits);
	c.features.dword12	= cpu_to_le32(host_mem_size);
	c.features.dword13	= cpu_to_le32(lower_32_bits(dma_addr));
	c.features.dword14	= cpu_to_le32(upper_32_bits(dma_addr));
	c.features.dword15	= cpu_to_le32(dev->nr_host_mem_descs);

	ret = snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
	if (ret) {
		dev_warn(dev->ctrl.device,
			 "failed to set host mem (err %d, flags %#x).\n",
			 ret, bits);
	} else
		dev->hmb = bits & NVME_HOST_MEM_ENABLE;

	return ret;
}

static void nvme_free_host_mem(struct nvme_dev *dev)
{
	int i;

	for (i = 0; i < dev->nr_host_mem_descs; i++) {
		struct nvme_host_mem_buf_desc *desc = &dev->host_mem_descs[i];
		size_t size = le32_to_cpu(desc->size) * NVME_CTRL_PAGE_SIZE;

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
		descs[i].size = cpu_to_le32(len / NVME_CTRL_PAGE_SIZE);
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
		size_t size = le32_to_cpu(descs[i].size) * NVME_CTRL_PAGE_SIZE;

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
	u64 min_chunk = min_t(u64, preferred, PAGE_SIZE * MAX_ORDER_NR_PAGES);
	u64 hmminds = max_t(u32, dev->ctrl.hmminds * 4096, PAGE_SIZE * 2);
	u64 chunk_size;

	/* start big and work our way down */
	for (chunk_size = min_chunk; chunk_size >= hmminds; chunk_size /= 2) {
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

	if (!dev->ctrl.hmpre)
		return 0;

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

static ssize_t cmb_show(struct device *dev, struct device_attribute *attr,
		char *buf)
{
	struct nvme_dev *ndev = to_nvme_dev(dev_get_drvdata(dev));

	return sysfs_emit(buf, "cmbloc : x%08x\ncmbsz  : x%08x\n",
		       ndev->cmbloc, ndev->cmbsz);
}
static DEVICE_ATTR_RO(cmb);

static ssize_t cmbloc_show(struct device *dev, struct device_attribute *attr,
		char *buf)
{
	struct nvme_dev *ndev = to_nvme_dev(dev_get_drvdata(dev));

	return sysfs_emit(buf, "%u\n", ndev->cmbloc);
}
static DEVICE_ATTR_RO(cmbloc);

static ssize_t cmbsz_show(struct device *dev, struct device_attribute *attr,
		char *buf)
{
	struct nvme_dev *ndev = to_nvme_dev(dev_get_drvdata(dev));

	return sysfs_emit(buf, "%u\n", ndev->cmbsz);
}
static DEVICE_ATTR_RO(cmbsz);

static ssize_t hmb_show(struct device *dev, struct device_attribute *attr,
			char *buf)
{
	struct nvme_dev *ndev = to_nvme_dev(dev_get_drvdata(dev));

	return sysfs_emit(buf, "%d\n", ndev->hmb);
}

static ssize_t hmb_store(struct device *dev, struct device_attribute *attr,
			 const char *buf, size_t count)
{
	struct nvme_dev *ndev = to_nvme_dev(dev_get_drvdata(dev));
	bool new;
	int ret;

	if (kstrtobool(buf, &new) < 0)
		return -EINVAL;

	if (new == ndev->hmb)
		return count;

	if (new) {
		ret = nvme_setup_host_mem(ndev);
	} else {
		ret = nvme_set_host_mem(ndev, 0);
		if (!ret)
			nvme_free_host_mem(ndev);
	}

	if (ret < 0)
		return ret;

	return count;
}
static DEVICE_ATTR_RW(hmb);

static umode_t nvme_pci_attrs_are_visible(struct kobject *kobj,
		struct attribute *a, int n)
{
	struct nvme_ctrl *ctrl =
		dev_get_drvdata(container_of(kobj, struct device, kobj));
	struct nvme_dev *dev = to_nvme_dev(ctrl);

	if (a == &dev_attr_cmb.attr ||
	    a == &dev_attr_cmbloc.attr ||
	    a == &dev_attr_cmbsz.attr) {
		if (!dev->cmbsz)
			return 0;
	}
	if (a == &dev_attr_hmb.attr && !ctrl->hmpre)
		return 0;

	return a->mode;
}

static struct attribute *nvme_pci_attrs[] = {
	&dev_attr_cmb.attr,
	&dev_attr_cmbloc.attr,
	&dev_attr_cmbsz.attr,
	&dev_attr_hmb.attr,
	NULL,
};

static const struct attribute_group nvme_pci_dev_attrs_group = {
	.attrs		= nvme_pci_attrs,
	.is_visible	= nvme_pci_attrs_are_visible,
};

static const struct attribute_group *nvme_pci_dev_attr_groups[] = {
	&snvme_dev_attrs_group,
	&nvme_pci_dev_attrs_group,
	NULL,
};

static void nvme_update_attrs(struct nvme_dev *dev)
{
	sysfs_update_group(&dev->ctrl.device->kobj, &nvme_pci_dev_attrs_group);
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
	 * If there is no interrupt available for queues, ensure that
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
	unsigned int irq_queues, poll_queues;

	/*
	 * Poll queues don't need interrupts, but we need at least one I/O queue
	 * left over for non-polled I/O.
	 */
	poll_queues = min(dev->nr_poll_queues, nr_io_queues - 1);
	dev->io_queues[HCTX_TYPE_POLL] = poll_queues;

	/*
	 * Initialize for the single interrupt case, will be updated in
	 * nvme_calc_irq_sets().
	 */
	dev->io_queues[HCTX_TYPE_DEFAULT] = 1;
	dev->io_queues[HCTX_TYPE_READ] = 0;

	/*
	 * We need interrupts for the admin queue and each non-polled I/O queue,
	 * but some Apple controllers require all queues to use the first
	 * vector.
	 */
	irq_queues = 1;
	if (!(dev->ctrl.quirks & NVME_QUIRK_SINGLE_VECTOR))
		irq_queues += (nr_io_queues - poll_queues);
	return pci_alloc_irq_vectors_affinity(pdev, 1, irq_queues,
			      PCI_IRQ_ALL_TYPES | PCI_IRQ_AFFINITY, &affd);
}

static unsigned int nvme_max_io_queues(struct nvme_dev *dev)
{
	/*
	 * If tags are shared with admin queue (Apple bug), then
	 * make sure we only use one IO queue.
	 */
	if (dev->ctrl.quirks & NVME_QUIRK_SHARED_TAGS)
		return 1;
	return num_possible_cpus() + dev->nr_write_queues + dev->nr_poll_queues;
}

static int s_nvme_setup_io_queues(struct nvme_dev *dev)
{
	struct nvme_queue *adminq = &dev->queues[0];
	struct pci_dev *pdev = to_pci_dev(dev->dev);
	unsigned int nr_io_queues;
	unsigned long size;
	int result;
	/*
	 * Sample the module parameters once at reset time so that we have
	 * stable values to work with.
	 *
	 * Guard with !cap_kernel_ioq so that a B3 reset-rebind doesn't
	 * clobber the per-BDF nr_write/nr_poll overrides that nvme_probe
	 * copied from ctrl->setup.  When cap_kernel_ioq == 0 the user
	 * has not opted in to B3 budget management, so the legacy
	 * module-default sampling is correct.
	 */
	if (!dev->cap_kernel_ioq) {
		dev->nr_write_queues = write_queues;
		dev->nr_poll_queues = poll_queues;
	}

	nr_io_queues = dev->nr_allocated_queues - 1;

	result = snvme_set_queue_count(&dev->ctrl, &nr_io_queues);

	/*
	 * snvme B3: record the controller-granted IOQ ceiling.  This
	 * is the authoritative bound for legal QID values used by
	 * NVM_ADD_USER_QUEUE; the user QID pool will be
	 * [online_queues..ctrl_max_io_queues].  Captured BEFORE any
	 * downstream code mutates nr_io_queues (the cap-shrink below)
	 * so the pool sizer always sees the controller's real grant --
	 * not whatever value the kernel ends up consuming.
	 */
	if (result == 0)
		dev->ctrl_max_io_queues = nr_io_queues;

	if (result < 0)
		return result;

	if (nr_io_queues == 0)
		return 0;

	/*
	 * B3 cap-only path (PORTING.md \xc2\xa77.3.1 #11): shrink the
	 * kernel-side consumption to cap_kernel_ioq AFTER the
	 * controller negotiation.  The controller already granted up
	 * to nr_io_queues, but we want QIDs (cap_kernel_ioq+1 ..
	 * ctrl_max_io_queues] to remain unused by the kernel so
	 * NVM_ADD_USER_QUEUE can claim them.
	 *
	 * Guarded so it only fires when cap_kernel_ioq was set --
	 * otherwise the comparison is a no-op anyway.  Order
	 * matters: this must run AFTER nr_io_queues has been finalised
	 * by snvme_set_queue_count, and BEFORE nvme_setup_irqs
	 * allocates IRQ vectors based on the count (we want vectors
	 * sized for the kernel-only share, not the full grant).
	 */
	if (dev->cap_kernel_ioq &&
	    nr_io_queues > dev->cap_kernel_ioq) {
		pr_info("snvme: capping kernel-side IOQ count from %u to %u "
			"(ctrl_max=%u, user pool gets [%u..%u])\n",
			nr_io_queues, dev->cap_kernel_ioq,
			dev->ctrl_max_io_queues,
			dev->cap_kernel_ioq + 1, dev->ctrl_max_io_queues);
		nr_io_queues = dev->cap_kernel_ioq;
	}

	/*
	 * Free IRQ resources as soon as NVMEQ_ENABLED bit transitions
	 * from set to unset. If there is a window to it is truely freed,
	 * pci_free_irq_vectors() jumping into this window will crash.
	 * And take lock to avoid racing with pci_free_irq_vectors() in
	 * nvme_dev_disable() path.
	 */
	result = nvme_setup_io_queues_trylock(dev);
	if (result)
		return result;
	if (test_and_clear_bit(NVMEQ_ENABLED, &adminq->flags))
		pci_free_irq(pdev, 0, adminq);

	if (dev->cmb_use_sqes) {
		result = nvme_cmb_qdepth(dev, nr_io_queues,
				sizeof(struct nvme_command));
		if (result > 0) {
			dev->q_depth = result;
			dev->ctrl.sqsize = result - 1;
		} else {
			dev->cmb_use_sqes = false;
		}
	}

	do {
		size = db_bar_size(dev, nr_io_queues);
		result = nvme_remap_bar(dev, size);
		if (!result)
			break;
		if (!--nr_io_queues) {
			result = -ENOMEM;
			goto out_unlock;
		}
	} while (1);
	adminq->q_db = dev->dbs;

 retry:
	/* Deregister the admin queue's interrupt */
	if (test_and_clear_bit(NVMEQ_ENABLED, &adminq->flags))
		pci_free_irq(pdev, 0, adminq);

	/*
	 * If we enable msix early due to not intx, disable it again before
	 * setting up the full range we need.
	 */
	pci_free_irq_vectors(pdev);

	/*user definded queue dose need irq*/

	result = nvme_setup_irqs(dev, nr_io_queues);
	if (result <= 0) {
		result = -EIO;
		goto out_unlock;
	}

	dev->num_vecs = result;
	result = max(result - 1, 1);
	dev->max_qid = result + dev->io_queues[HCTX_TYPE_POLL];
	printk("s_nvme_setup_io_queues max_id is %d\n",dev->max_qid);
	printk("s_nvme_setup_io_queues online_queues is %d\n",dev->online_queues);
	/*
	 * Should investigate if there's a performance win from allocating
	 * more queues than interrupt vectors; it might allow the submission
	 * path to scale better, even if the receive path is limited by the
	 * number of interrupts.
	 */
	result = queue_request_irq(adminq);
	if (result)
		goto out_unlock;
	set_bit(NVMEQ_ENABLED, &adminq->flags);
	mutex_unlock(&dev->shutdown_lock);

	result = nvme_create_io_queues(dev);
	printk("create io queue finish max_id is %d, online_queues is %d\n",dev->max_qid,dev->online_queues);

	if (result || dev->online_queues < 2)
		return result;

	if (dev->online_queues - 1 < dev->max_qid) {
		nr_io_queues = dev->online_queues - 1;
		nvme_delete_io_queues(dev);
		result = nvme_setup_io_queues_trylock(dev);
		if (result)
			return result;
		nvme_suspend_io_queues(dev);
		goto retry;
	}
	dev_info(dev->ctrl.device, "%d/%d/%d default/read/poll queues\n",
					dev->io_queues[HCTX_TYPE_DEFAULT],
					dev->io_queues[HCTX_TYPE_READ],
					dev->io_queues[HCTX_TYPE_POLL]);
	return 0;
out_unlock:
	mutex_unlock(&dev->shutdown_lock);
	return result;
}

static enum rq_end_io_ret nvme_del_queue_end(struct request *req,
					     blk_status_t error)
{
	struct nvme_queue *nvmeq = req->end_io_data;

	blk_mq_free_request(req);
	complete(&nvmeq->delete_done);
	return RQ_END_IO_NONE;
}

static enum rq_end_io_ret nvme_del_cq_end(struct request *req,
					  blk_status_t error)
{
	struct nvme_queue *nvmeq = req->end_io_data;

	if (error)
		set_bit(NVMEQ_DELETE_ERROR, &nvmeq->flags);

	return nvme_del_queue_end(req, error);
}

static int nvme_delete_queue(struct nvme_queue *nvmeq, u8 opcode)
{
	struct request_queue *q = nvmeq->dev->ctrl.admin_q;
	struct request *req;
	struct nvme_command cmd = { };

	cmd.delete_queue.opcode = opcode;
	cmd.delete_queue.qid = cpu_to_le16(nvmeq->qid);

	req = blk_mq_alloc_request(q, nvme_req_op(&cmd), BLK_MQ_REQ_NOWAIT);
	if (IS_ERR(req))
		return PTR_ERR(req);
	snvme_init_request(req, &cmd);

	if (opcode == nvme_admin_delete_cq)
		req->end_io = nvme_del_cq_end;
	else
		req->end_io = nvme_del_queue_end;
	req->end_io_data = nvmeq;

	init_completion(&nvmeq->delete_done);
	blk_execute_rq_nowait(req, false);
	return 0;
}

static bool __nvme_delete_io_queues(struct nvme_dev *dev, u8 opcode)
{
	int nr_queues = dev->online_queues - 1, sent = 0;
	unsigned long timeout;

 retry:
	timeout = NVME_ADMIN_TIMEOUT;
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

static void nvme_delete_io_queues(struct nvme_dev *dev)
{
	if (__nvme_delete_io_queues(dev, nvme_admin_delete_sq))
		__nvme_delete_io_queues(dev, nvme_admin_delete_cq);
}

static unsigned int nvme_pci_nr_maps(struct nvme_dev *dev)
{
	if (dev->io_queues[HCTX_TYPE_POLL])
		return 3;
	if (dev->io_queues[HCTX_TYPE_READ])
		return 2;
	return 1;
}

static void nvme_pci_update_nr_queues(struct nvme_dev *dev)
{
	blk_mq_update_nr_hw_queues(&dev->tagset, dev->online_queues - 1);
	/* free previously allocated queues that are no longer usable */
	nvme_free_queues(dev, dev->online_queues);
}

static int nvme_pci_enable(struct nvme_dev *dev)
{
	int result = -ENOMEM;
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	if (pci_enable_device_mem(pdev))
		return result;

	pci_set_master(pdev);

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
		goto disable;

	dev->ctrl.cap = lo_hi_readq(dev->bar + NVME_REG_CAP);

	dev->q_depth = min_t(u32, NVME_CAP_MQES(dev->ctrl.cap) + 1,
				io_queue_depth);
	dev->db_stride = 1 << NVME_CAP_STRIDE(dev->ctrl.cap);

	dev->dbs = dev->bar + 4096;
	printk("nvme_pci_enable is dev->q_depth %u\n",dev->q_depth);
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
	dev->ctrl.sqsize = dev->q_depth - 1; /* 0's based queue depth */

	nvme_map_cmb(dev);

	pci_save_state(pdev);

	result = nvme_pci_configure_admin_queue(dev);
	if (result)
		goto free_irq;
	return result;

 free_irq:
	pci_free_irq_vectors(pdev);
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

static bool nvme_pci_ctrl_is_dead(struct nvme_dev *dev)
{
	struct pci_dev *pdev = to_pci_dev(dev->dev);
	u32 csts;

	if (!pci_is_enabled(pdev) || !pci_device_is_present(pdev))
		return true;
	if (pdev->error_state != pci_channel_io_normal)
		return true;

	csts = readl(dev->bar + NVME_REG_CSTS);
	return (csts & NVME_CSTS_CFS) || !(csts & NVME_CSTS_RDY);
}

static void nvme_dev_disable(struct nvme_dev *dev, bool shutdown)
{
	enum nvme_ctrl_state state = nvme_ctrl_state(&dev->ctrl);
	struct pci_dev *pdev = to_pci_dev(dev->dev);
	bool dead;

	mutex_lock(&dev->shutdown_lock);
	dead = nvme_pci_ctrl_is_dead(dev);
	if (state == NVME_CTRL_LIVE || state == NVME_CTRL_RESETTING) {
		if (pci_is_enabled(pdev))
			snvme_start_freeze(&dev->ctrl);
		/*
		 * Give the controller a chance to complete all entered requests
		 * if doing a safe shutdown.
		 */
		if (!dead && shutdown)
			snvme_wait_freeze_timeout(&dev->ctrl, NVME_IO_TIMEOUT);
	}

	snvme_quiesce_io_queues(&dev->ctrl);

	if (!dead && dev->ctrl.queue_count > 0) {
		nvme_delete_io_queues(dev);
		snvme_disable_ctrl(&dev->ctrl, shutdown);
		nvme_poll_irqdisable(&dev->queues[0]);
	}
	nvme_suspend_io_queues(dev);
	nvme_suspend_queue(dev, 0);
	pci_free_irq_vectors(pdev);
	if (pci_is_enabled(pdev))
		pci_disable_device(pdev);
	nvme_reap_pending_cqes(dev);

	snvme_cancel_tagset(&dev->ctrl);
	snvme_cancel_admin_tagset(&dev->ctrl);

	/*
	 * The driver will not be starting up queues again if shutting down so
	 * must flush all entered requests to their failed completion to avoid
	 * deadlocking blk-mq hot-cpu notifier.
	 */
	if (shutdown) {
		snvme_unquiesce_io_queues(&dev->ctrl);
		if (dev->ctrl.admin_q && !blk_queue_dying(dev->ctrl.admin_q))
			snvme_unquiesce_admin_queue(&dev->ctrl);
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
						NVME_CTRL_PAGE_SIZE,
						NVME_CTRL_PAGE_SIZE, 0);
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

static int nvme_pci_alloc_iod_mempool(struct nvme_dev *dev)
{
	size_t alloc_size = sizeof(struct scatterlist) * NVME_MAX_SEGS;

	dev->iod_mempool = mempool_create_node(1,
			mempool_kmalloc, mempool_kfree,
			(void *)alloc_size, GFP_KERNEL,
			dev_to_node(dev->dev));
	if (!dev->iod_mempool)
		return -ENOMEM;
	return 0;
}

static void nvme_free_tagset(struct nvme_dev *dev)
{
	if (dev->tagset.tags)
		snvme_remove_io_tag_set(&dev->ctrl);
	dev->ctrl.tagset = NULL;
}

/* pairs with nvme_pci_alloc_dev */
static void nvme_pci_free_ctrl(struct nvme_ctrl *ctrl)
{
	struct nvme_dev *dev = to_nvme_dev(ctrl);

	nvme_free_tagset(dev);
	put_device(dev->dev);
	kfree(dev->queues);
	kfree(dev);
}

static void nvme_reset_work(struct work_struct *work)
{
	struct nvme_dev *dev =
		container_of(work, struct nvme_dev, ctrl.reset_work);
	bool was_suspend = !!(dev->ctrl.ctrl_config & NVME_CC_SHN_NORMAL);
	int result;

	if (nvme_ctrl_state(&dev->ctrl) != NVME_CTRL_RESETTING) {
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
	snvme_unquiesce_admin_queue(&dev->ctrl);
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

	result = snvme_init_ctrl_finish(&dev->ctrl, was_suspend);
	if (result)
		goto out;

	nvme_dbbuf_dma_alloc(dev);

	result = nvme_setup_host_mem(dev);
	if (result < 0)
		goto out;

	result = s_nvme_setup_io_queues(dev);
	if (result)
		goto out;

	/*
	 * Freeze and update the number of I/O queues as thos might have
	 * changed.  If there are no I/O queues left after this reset, keep the
	 * controller around but remove all namespaces.
	 */
	if (dev->online_queues > 1) {
		nvme_dbbuf_set(dev);
		snvme_unquiesce_io_queues(&dev->ctrl);
		snvme_wait_freeze(&dev->ctrl);
		nvme_pci_update_nr_queues(dev);
		snvme_unfreeze(&dev->ctrl);
	} else {
		dev_warn(dev->ctrl.device, "IO queues lost\n");
		snvme_mark_namespaces_dead(&dev->ctrl);
		snvme_unquiesce_io_queues(&dev->ctrl);
		snvme_remove_namespaces(&dev->ctrl);
		nvme_free_tagset(dev);
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
	/*
	 * Set state to deleting now to avoid blocking snvme_wait_reset(), which
	 * may be holding this pci_dev's device lock.
	 */
	dev_warn(dev->ctrl.device, "Disabling device after reset failure: %d\n",
		 result);
	snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_DELETING);
	nvme_dev_disable(dev, true);
	snvme_sync_queues(&dev->ctrl);
	snvme_mark_namespaces_dead(&dev->ctrl);
	snvme_unquiesce_io_queues(&dev->ctrl);
	snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_DEAD);
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

	return snprintf(buf, size, "%s\n", dev_name(&pdev->dev));
}

static void nvme_pci_print_device_info(struct nvme_ctrl *ctrl)
{
	struct pci_dev *pdev = to_pci_dev(to_nvme_dev(ctrl)->dev);
	struct nvme_subsystem *subsys = ctrl->subsys;

	dev_err(ctrl->device,
		"VID:DID %04x:%04x model:%.*s firmware:%.*s\n",
		pdev->vendor, pdev->device,
		nvme_strlen(subsys->model, sizeof(subsys->model)),
		subsys->model, nvme_strlen(subsys->firmware_rev,
					   sizeof(subsys->firmware_rev)),
		subsys->firmware_rev);
}

static bool nvme_pci_supports_pci_p2pdma(struct nvme_ctrl *ctrl)
{
	struct nvme_dev *dev = to_nvme_dev(ctrl);

	return dma_pci_p2pdma_supported(dev->dev);
}

static const struct nvme_ctrl_ops nvme_pci_ctrl_ops = {
	.name			= "pcie",
	.module			= THIS_MODULE,
	.flags			= NVME_F_METADATA_SUPPORTED,
	.dev_attr_groups	= nvme_pci_dev_attr_groups,
	.reg_read32		= nvme_pci_reg_read32,
	.reg_write32		= nvme_pci_reg_write32,
	.reg_read64		= nvme_pci_reg_read64,
	.free_ctrl		= nvme_pci_free_ctrl,
	.submit_async_event	= nvme_pci_submit_async_event,
	.get_address		= nvme_pci_get_address,
	.print_device_info	= nvme_pci_print_device_info,
	.supports_pci_p2pdma	= nvme_pci_supports_pci_p2pdma,
};

static int nvme_dev_map(struct nvme_dev *dev)
{
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	if (pci_request_mem_regions(pdev, "nvme"))
		return -ENODEV;

	if (nvme_remap_bar(dev, NVME_REG_DBS + 4096*3))
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
	} else if (pdev->vendor == 0x2646 && (pdev->device == 0x2263 ||
		   pdev->device == 0x500f)) {
		/*
		 * Exclude some Kingston NV1 and A2000 devices from
		 * NVME_QUIRK_SIMPLE_SUSPEND. Do a full suspend to save a
		 * lot fo energy with s2idle sleep on some TUXEDO platforms.
		 */
		if (dmi_match(DMI_BOARD_NAME, "NS5X_NS7XAU") ||
		    dmi_match(DMI_BOARD_NAME, "NS5x_7xAU") ||
		    dmi_match(DMI_BOARD_NAME, "NS5x_7xPU") ||
		    dmi_match(DMI_BOARD_NAME, "PH4PRX1_PH6PRX1"))
			return NVME_QUIRK_FORCE_NO_SIMPLE_SUSPEND;
	}

	return 0;
}

static struct nvme_dev *nvme_pci_alloc_dev(struct pci_dev *pdev,
		const struct pci_device_id *id, struct ctrl *ctrl)
{
	unsigned long quirks = id->driver_data;
	int node = dev_to_node(&pdev->dev);
	struct nvme_dev *dev;
	int ret = -ENOMEM;

	dev = kzalloc_node(sizeof(*dev), GFP_KERNEL, node);
	if (!dev)
		return ERR_PTR(-ENOMEM);
	INIT_WORK(&dev->ctrl.reset_work, nvme_reset_work);
	mutex_init(&dev->shutdown_lock);

	/*
	 * B3 setup snapshot: copy ctrl->setup fields onto the per-dev
	 * shadow that s_nvme_setup_io_queues consumes.  Gated on
	 * ctrl->setup.valid because:
	 *   - valid==0 means userspace never issued NVM_SET_IOQ_NUM
	 *     /NVM_SET_KERNEL_IOQ_CAP, so keep upstream defaults;
	 *   - valid==1 means at least one of the new ioctls has run
	 *     and the snapshot is authoritative.
	 *
	 * cap_kernel_ioq=0 acts as "no override" inside
	 * s_nvme_setup_io_queues, so a setup that left it zero (a
	 * pure NVM_SET_IOQ_NUM without cap) still gets the upstream
	 * num_possible_cpus() default for the kernel side.
	 */
	if (ctrl && ctrl->setup.valid) {
		if (ctrl->setup.nr_write)
			dev->nr_write_queues = ctrl->setup.nr_write;
		else
			dev->nr_write_queues = write_queues;
		if (ctrl->setup.nr_poll)
			dev->nr_poll_queues = ctrl->setup.nr_poll;
		else
			dev->nr_poll_queues = poll_queues;
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
		goto out_free_dev;

	dev->dev = get_device(&pdev->dev);

	quirks |= check_vendor_combination_bug(pdev);
	if (!noacpi &&
	    !(quirks & NVME_QUIRK_FORCE_NO_SIMPLE_SUSPEND) &&
	    acpi_storage_d3(&pdev->dev)) {
		/*
		 * Some systems use a bios work around to ask for D3 on
		 * platforms that support kernel managed suspend.
		 */
		dev_info(&pdev->dev,
			 "platform quirk: setting simple suspend\n");
		quirks |= NVME_QUIRK_SIMPLE_SUSPEND;
	}
	ret = snvme_init_ctrl(&dev->ctrl, &pdev->dev, &nvme_pci_ctrl_ops,
			     quirks);
	if (ret)
		goto out_put_device;

	if (dev->ctrl.quirks & NVME_QUIRK_DMA_ADDRESS_BITS_48)
		dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(48));
	else
		dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
	dma_set_min_align_mask(&pdev->dev, NVME_CTRL_PAGE_SIZE - 1);
	dma_set_max_seg_size(&pdev->dev, 0xffffffff);

	/*
	 * Limit the max command size to prevent iod->sg allocations going
	 * over a single page.
	 */
	dev->ctrl.max_hw_sectors = min_t(u32,
		NVME_MAX_KB_SZ << 1, dma_opt_mapping_size(&pdev->dev) >> 9);
	dev->ctrl.max_segments = NVME_MAX_SEGS;

	/*
	 * There is no support for SGLs for metadata (yet), so we are limited to
	 * a single integrity segment for the separate metadata pointer.
	 */
	dev->ctrl.max_integrity_segments = 1;
	return dev;

out_put_device:
	put_device(dev->dev);
	kfree(dev->queues);
out_free_dev:
	kfree(dev);
	return ERR_PTR(ret);
}

static int nvme_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
	struct nvme_dev *dev;
	struct ctrl *ctrl;
	int result = -ENOMEM;

	/* Only bind controllers explicitly registered through SNVM_CHRDEV_CREATE. */
	ctrl = ctrl_find_by_pci_dev(&ctrl_list, pdev);
	if (!ctrl)
		return -ENODEV;

	dev = nvme_pci_alloc_dev(pdev, id, ctrl);
	if (IS_ERR(dev))
		return PTR_ERR(dev);

	result = nvme_dev_map(dev);
	if (result)
		goto out_uninit_ctrl;

	result = nvme_setup_prp_pools(dev);
	if (result)
		goto out_dev_unmap;

	result = nvme_pci_alloc_iod_mempool(dev);
	if (result)
		goto out_release_prp_pools;

	dev_info(dev->ctrl.device, "pci function %s\n", dev_name(&pdev->dev));

	result = nvme_pci_enable(dev);
	if (result)
		goto out_release_iod_mempool;

	result = snvme_alloc_admin_tag_set(&dev->ctrl, &dev->admin_tagset,
				&nvme_mq_admin_ops, sizeof(struct nvme_iod));
	if (result)
		goto out_disable;

	/*
	 * Mark the controller as connecting before sending admin commands to
	 * allow the timeout handler to do the right thing.
	 */
	if (!snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_CONNECTING)) {
		dev_warn(dev->ctrl.device,
			"failed to mark controller CONNECTING\n");
		result = -EBUSY;
		goto out_disable;
	}

	result = snvme_init_ctrl_finish(&dev->ctrl, false);
	if (result)
		goto out_disable;

	nvme_dbbuf_dma_alloc(dev);

	result = nvme_setup_host_mem(dev);
	if (result < 0)
		goto out_disable;

	result = s_nvme_setup_io_queues(dev);
	if (result)
		goto out_disable;

	if (dev->online_queues > 1) {
		snvme_alloc_io_tag_set(&dev->ctrl, &dev->tagset, &nvme_mq_ops,
				nvme_pci_nr_maps(dev), sizeof(struct nvme_iod));
		nvme_dbbuf_set(dev);
	}

	if (!dev->ctrl.tagset)
		dev_warn(dev->ctrl.device, "IO queues not created\n");

	if (!snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_LIVE)) {
		dev_warn(dev->ctrl.device,
			"failed to mark controller live state\n");
		result = -ENODEV;
		goto out_disable;
	}

	pci_set_drvdata(pdev, dev);

	snvme_start_ctrl(&dev->ctrl);
	nvme_put_ctrl(&dev->ctrl);
	flush_work(&dev->ctrl.scan_work);
	return 0;

out_disable:
	snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_DELETING);
	nvme_dev_disable(dev, true);
	nvme_free_host_mem(dev);
	nvme_dev_remove_admin(dev);
	nvme_dbbuf_dma_free(dev);
	nvme_free_queues(dev, 0);
out_release_iod_mempool:
	mempool_destroy(dev->iod_mempool);
out_release_prp_pools:
	nvme_release_prp_pools(dev);
out_dev_unmap:
	nvme_dev_unmap(dev);
out_uninit_ctrl:
	snvme_uninit_ctrl(&dev->ctrl);
	nvme_put_ctrl(&dev->ctrl);
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
	nvme_free_host_mem(dev);
	nvme_dev_remove_admin(dev);
	nvme_dbbuf_dma_free(dev);
	nvme_free_queues(dev, 0);
	mempool_destroy(dev->iod_mempool);
	nvme_release_prp_pools(dev);
	nvme_dev_unmap(dev);
	snvme_uninit_ctrl(&dev->ctrl);
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
		goto reset;
	if (ctrl->hmpre && nvme_setup_host_mem(ndev))
		goto reset;

	return 0;
reset:
	return snvme_try_sched_reset(ctrl);
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
	 */
	if (pm_suspend_via_firmware() || !ctrl->npss ||
	    !pcie_aspm_enabled(pdev) ||
	    (ndev->ctrl.quirks & NVME_QUIRK_SIMPLE_SUSPEND))
		return nvme_disable_prepare_reset(ndev, true);

	snvme_start_freeze(ctrl);
	snvme_wait_freeze(ctrl);
	snvme_sync_queues(ctrl);

	if (nvme_ctrl_state(ctrl) != NVME_CTRL_LIVE)
		goto unfreeze;

	/*
	 * Host memory access may not be successful in a system suspend state,
	 * but the specification allows the controller to access memory in a
	 * non-operational power state.
	 */
	if (ndev->hmb) {
		ret = nvme_set_host_mem(ndev, 0);
		if (ret < 0)
			goto unfreeze;
	}

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
		 * correct value will be rediscovered then.
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
		if (!snvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_RESETTING)) {
			nvme_dev_disable(dev, true);
			return PCI_ERS_RESULT_DISCONNECT;
		}
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
	if (!snvme_try_sched_reset(&dev->ctrl))
		snvme_unquiesce_io_queues(&dev->ctrl);
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
	{ PCI_VDEVICE(INTEL, 0x0953),	/* Intel 750/P3500/P3600/P3700 */
		.driver_data = NVME_QUIRK_STRIPE_SIZE |
				NVME_QUIRK_DEALLOCATE_ZEROES, },
	{ PCI_VDEVICE(INTEL, 0x0a53),	/* Intel P3520 */
		.driver_data = NVME_QUIRK_STRIPE_SIZE |
				NVME_QUIRK_DEALLOCATE_ZEROES, },
	{ PCI_VDEVICE(INTEL, 0x0a54),	/* Intel P4500/P4600 */
		.driver_data = NVME_QUIRK_STRIPE_SIZE |
				NVME_QUIRK_DEALLOCATE_ZEROES |
				NVME_QUIRK_IGNORE_DEV_SUBNQN |
				NVME_QUIRK_BOGUS_NID, },
	{ PCI_VDEVICE(INTEL, 0x0a55),	/* Dell Express Flash P4600 */
		.driver_data = NVME_QUIRK_STRIPE_SIZE |
				NVME_QUIRK_DEALLOCATE_ZEROES, },
	{ PCI_VDEVICE(INTEL, 0xf1a5),	/* Intel 600P/P3100 */
		.driver_data = NVME_QUIRK_NO_DEEPEST_PS |
				NVME_QUIRK_MEDIUM_PRIO_SQ |
				NVME_QUIRK_NO_TEMP_THRESH_CHANGE |
				NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_VDEVICE(INTEL, 0xf1a6),	/* Intel 760p/Pro 7600p */
		.driver_data = NVME_QUIRK_IGNORE_DEV_SUBNQN, },
	{ PCI_VDEVICE(INTEL, 0x5845),	/* Qemu emulated controller */
		.driver_data = NVME_QUIRK_IDENTIFY_CNS |
				NVME_QUIRK_DISABLE_WRITE_ZEROES |
				NVME_QUIRK_BOGUS_NID, },
	{ PCI_VDEVICE(REDHAT, 0x0010),	/* Qemu emulated controller */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x126f, 0x2263),	/* Silicon Motion unidentified */
		.driver_data = NVME_QUIRK_NO_NS_DESC_LIST |
				NVME_QUIRK_BOGUS_NID, },
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
	{ PCI_DEVICE(0x1987, 0x5012),	/* Phison E12 */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1987, 0x5016),	/* Phison E16 */
		.driver_data = NVME_QUIRK_IGNORE_DEV_SUBNQN |
				NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1987, 0x5019),  /* phison E19 */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x1987, 0x5021),   /* Phison E21 */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x1b4b, 0x1092),	/* Lexar 256 GB SSD */
		.driver_data = NVME_QUIRK_NO_NS_DESC_LIST |
				NVME_QUIRK_IGNORE_DEV_SUBNQN, },
	{ PCI_DEVICE(0x1cc1, 0x33f8),   /* ADATA IM2P33F8ABR1 1 TB */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x10ec, 0x5762),   /* ADATA SX6000LNP */
		.driver_data = NVME_QUIRK_IGNORE_DEV_SUBNQN |
				NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x10ec, 0x5763),  /* ADATA SX6000PNP */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1cc1, 0x8201),   /* ADATA SX8200PNP 512GB */
		.driver_data = NVME_QUIRK_NO_DEEPEST_PS |
				NVME_QUIRK_IGNORE_DEV_SUBNQN, },
	 { PCI_DEVICE(0x1344, 0x5407), /* Micron Technology Inc NVMe SSD */
		.driver_data = NVME_QUIRK_IGNORE_DEV_SUBNQN },
	 { PCI_DEVICE(0x1344, 0x6001),   /* Micron Nitro NVMe */
		 .driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1c5c, 0x1504),   /* SK Hynix PC400 */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x1c5c, 0x174a),   /* SK Hynix P31 SSD */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1c5c, 0x1D59),   /* SK Hynix BC901 */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x15b7, 0x2001),   /*  Sandisk Skyhawk */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x1d97, 0x2263),   /* SPCC */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x144d, 0xa80b),   /* Samsung PM9B1 256G and 512G */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES |
				NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x144d, 0xa809),   /* Samsung MZALQ256HBJD 256G */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x144d, 0xa802),   /* Samsung SM953 */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1cc4, 0x6303),   /* UMIS RPJTJ512MGE1QDY 512G */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x1cc4, 0x6302),   /* UMIS RPJTJ256MGE1QDY 256G */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x2646, 0x2262),   /* KINGSTON SKC2000 NVMe SSD */
		.driver_data = NVME_QUIRK_NO_DEEPEST_PS, },
	{ PCI_DEVICE(0x2646, 0x2263),   /* KINGSTON A2000 NVMe SSD  */
		.driver_data = NVME_QUIRK_NO_DEEPEST_PS, },
	{ PCI_DEVICE(0x2646, 0x5013),   /* Kingston KC3000, Kingston FURY Renegade */
		.driver_data = NVME_QUIRK_NO_SECONDARY_TEMP_THRESH, },
	{ PCI_DEVICE(0x2646, 0x5018),   /* KINGSTON OM8SFP4xxxxP OS21012 NVMe SSD */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x2646, 0x5016),   /* KINGSTON OM3PGP4xxxxP OS21011 NVMe SSD */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x2646, 0x501A),   /* KINGSTON OM8PGP4xxxxP OS21005 NVMe SSD */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x2646, 0x501B),   /* KINGSTON OM8PGP4xxxxQ OS21005 NVMe SSD */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x2646, 0x501E),   /* KINGSTON OM3PGP4xxxxQ OS21011 NVMe SSD */
		.driver_data = NVME_QUIRK_DISABLE_WRITE_ZEROES, },
	{ PCI_DEVICE(0x1f40, 0x1202),   /* Netac Technologies Co. NV3000 NVMe SSD */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1f40, 0x5236),   /* Netac Technologies Co. NV7000 NVMe SSD */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1e4B, 0x1001),   /* MAXIO MAP1001 */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1e4B, 0x1002),   /* MAXIO MAP1002 */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1e4B, 0x1202),   /* MAXIO MAP1202 */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1e4B, 0x1602),   /* MAXIO MAP1602 */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1cc1, 0x5350),   /* ADATA XPG GAMMIX S50 */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1dbe, 0x5236),   /* ADATA XPG GAMMIX S70 */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1e49, 0x0021),   /* ZHITAI TiPro5000 NVMe SSD */
		.driver_data = NVME_QUIRK_NO_DEEPEST_PS, },
	{ PCI_DEVICE(0x1e49, 0x0041),   /* ZHITAI TiPro7000 NVMe SSD */
		.driver_data = NVME_QUIRK_NO_DEEPEST_PS, },
	{ PCI_DEVICE(0xc0a9, 0x540a),   /* Crucial P2 */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1d97, 0x2263), /* Lexar NM610 */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1d97, 0x1d97), /* Lexar NM620 */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1d97, 0x2269), /* Lexar NM760 */
		.driver_data = NVME_QUIRK_BOGUS_NID |
				NVME_QUIRK_IGNORE_DEV_SUBNQN, },
	{ PCI_DEVICE(0x10ec, 0x5763), /* TEAMGROUP T-FORCE CARDEA ZERO Z330 SSD */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x1e4b, 0x1602), /* HS-SSD-FUTURE 2048G  */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(0x10ec, 0x5765), /* TEAMGROUP MP33 2TB SSD */
		.driver_data = NVME_QUIRK_BOGUS_NID, },
	{ PCI_DEVICE(PCI_VENDOR_ID_AMAZON, 0x0061),
		.driver_data = NVME_QUIRK_DMA_ADDRESS_BITS_48, },
	{ PCI_DEVICE(PCI_VENDOR_ID_AMAZON, 0x0065),
		.driver_data = NVME_QUIRK_DMA_ADDRESS_BITS_48, },
	{ PCI_DEVICE(PCI_VENDOR_ID_AMAZON, 0x8061),
		.driver_data = NVME_QUIRK_DMA_ADDRESS_BITS_48, },
	{ PCI_DEVICE(PCI_VENDOR_ID_AMAZON, 0xcd00),
		.driver_data = NVME_QUIRK_DMA_ADDRESS_BITS_48, },
	{ PCI_DEVICE(PCI_VENDOR_ID_AMAZON, 0xcd01),
		.driver_data = NVME_QUIRK_DMA_ADDRESS_BITS_48, },
	{ PCI_DEVICE(PCI_VENDOR_ID_AMAZON, 0xcd02),
		.driver_data = NVME_QUIRK_DMA_ADDRESS_BITS_48, },
	{ PCI_DEVICE(PCI_VENDOR_ID_APPLE, 0x2001),
		.driver_data = NVME_QUIRK_SINGLE_VECTOR },
	{ PCI_DEVICE(PCI_VENDOR_ID_APPLE, 0x2003) },
	{ PCI_DEVICE(PCI_VENDOR_ID_APPLE, 0x2005),
		.driver_data = NVME_QUIRK_SINGLE_VECTOR |
				NVME_QUIRK_128_BYTES_SQES |
				NVME_QUIRK_SHARED_TAGS |
				NVME_QUIRK_SKIP_CID_GEN |
				NVME_QUIRK_IDENTIFY_CNS },
	{ PCI_DEVICE_CLASS(PCI_CLASS_STORAGE_EXPRESS, 0xffffff) },
	{ 0, }
};
MODULE_DEVICE_TABLE(pci, nvme_id_table);

/*
 * Per-fd /dev/ssnvme<N> owner descriptor.
 *
 * The original snvme-5.15.0 fops table only carried .owner / .unlocked_ioctl
 * / .mmap -- no .open or .release.  That meant a userspace process dying
 * between NVM_MAP_* and NVM_UNMAP_* leaked:
 *   1. pinned host pages on the host_list,
 *   2. peer_memory get_pages references on the device_list /
 *      device_queue_list (rmmod snvme will then refuse with "module in
 *      use" until reboot).
 *
 * Capturing the opener at .open time (rather than reading `current` at
 * .release time) is critical: by the time __fput() invokes .release,
 * the task may have already exited (or be a different thread-group
 * member, or a forked child).  map.c::create_descriptor records
 * map->owner from `current` at the time of the NVM_MAP_* ioctl, so
 * matching that key at .release time requires we stash it at .open.
 *
 * See PORTING.md \xc2\xa77.3.1 trap "snvm_dev_fops MUST have .open + .release
 * hooks" for the full leak-on-crash motivation.
 */
struct snvm_dev_owner {
	struct ctrl		*ctrl;
	struct task_struct	*owner;

	/*
	 * Per-fd queue group list (NVM_CREATE_QUEUE_GROUP adds entries,
	 * NVM_DESTROY_QUEUE_GROUP and the fd-close cascade in
	 * snvm_dev_release drain them).  Protected by groups_lock
	 * against concurrent ioctl threads on the same fd; release()
	 * runs after all ioctl handlers have returned (vfs guarantees
	 * fput happens after the last fd ref drops) so the lock is
	 * uncontended there, but we still take it for lockdep
	 * cleanliness.
	 *
	 * Groups are not placed on any global list: cascade-cleanup on
	 * fd-close needs only this fd's groups, and there's no cross-fd
	 * sharing of group_id (the IDA owns the namespace, descriptors
	 * are strictly per-fd).
	 */
	struct list_head	groups;       /* head of struct snvm_qgroup */
	struct mutex		groups_lock;  /* serialises group list mutation */
	unsigned int		nr_groups;    /* current count, for cap check   */

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
 * NVM_CREATE_QUEUE_GROUP / NVM_DESTROY_QUEUE_GROUP.  Do NOT confuse
 * with struct snvm_queue_group in ctrl.h, which is the bind-time
 * per-controller GPU partitioning descriptor used by
 * NVM_SET_IOQ_NUM.  Different problems, different lifetimes; the
 * _qgroup suffix keeps the namespaces distinct.
 *
 * Fields (B1 + B2 + B3): link, group_id, max_queues, maps, nr_maps,
 * and (Chunk H) the inline queues[] array + cur_queues counter that
 * NVM_ADD_USER_QUEUE populates.
 *
 * Lifetime:
 *   - allocated by NVM_CREATE_QUEUE_GROUP, group_id assigned via
 *     ida_simple_get(&snvm_queue_group_ida, 1, 0, GFP_KERNEL).
 *   - released by NVM_DESTROY_QUEUE_GROUP or by the fd-close cascade
 *     in snvm_dev_release.
 */
struct snvm_qgroup {
	struct list_head	link;       /* into snvm_dev_owner.groups */
	uint32_t		group_id;
	uint32_t		max_queues; /* echoed NVM_MAX_QUEUES_PER_GROUP */

	/*
	 * Per-group registered maps (B2).  Each entry is a struct map
	 * threaded by its group_link member.  Adding a map is done by
	 * NVM_MAP_HOST_MEMORY / NVM_MAP_DEVICE_MEMORY when the payload's
	 * group_id != 0 (added in a future chunk); removing happens via
	 * NVM_UNMAP_* (vaddr lookup) or via destroy_qgroup_locked()
	 * during NVM_DESTROY_QUEUE_GROUP / fd-close cascade.
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

static DEFINE_IDA(snvm_queue_group_ida);

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
		pr_warn("snvme: user QID pool: ctrl_max_io_queues=0 "
			"(probe did not complete the Set-Features negotiation?)\n");
		return -ENODEV;
	}

	first = ndev->online_queues;
	last  = ndev->ctrl_max_io_queues;
	if (first > last) {
		pr_warn("snvme: user QID pool empty (online=%u, ctrl_max=%u); "
			"controller refused to leave room for user IOQs.  "
			"Lower cap_kernel_ioq via NVM_SET_IOQ_NUM before bind, "
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

	pr_info("snvme: user QID pool initialised: [%u..%u] (%u QIDs)\n",
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
		pr_warn("snvme: user_qid_free: qid %u outside pool [%u..%u]\n",
			qid, ctrl->user_qid_first, ctrl->user_qid_last);
		return;
	}
	bit = qid - ctrl->user_qid_first;
	if (!test_and_clear_bit(bit, ctrl->user_qid_bitmap))
		pr_warn("snvme: user_qid_free: qid %u was already free\n", qid);
}

/*
 * Forward decl for snvm_ctrl_get_live_ndev (defined below, between
 * destroy_qgroup_locked and find_qgroup_locked).  destroy_qgroup_locked
 * needs it (once H3 wires in the user-queue drain) to issue Delete
 * I/O SQ/CQ admin commands only while running against a controller
 * still bound to snvme -- the cascade-cleanup path may race with
 * unbind/rebind, in which case admin commands must be skipped.
 *
 * adapter_delete_sq / adapter_delete_cq are already defined above
 * in this file (see qid_release path further up), so they don't
 * need re-declaration here.
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
				pr_warn("snvme: destroy_qgroup id=%u: "
					"Delete I/O SQ qid=%u failed: %d\n",
					g->group_id, uq->qid, rc);
			rc = adapter_delete_cq(ndev, uq->qid);
			if (rc)
				pr_warn("snvme: destroy_qgroup id=%u: "
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
		pr_info("snvme: destroy_qgroup id=%u drained %u user queue(s)\n",
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
		pr_info("snvme: destroy_qgroup id=%u drained %u map(s)\n",
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

static long snvm_dev_map_ioctl(struct file* file, unsigned int cmd, unsigned long arg)
{
	int ret = 0;
    struct ctrl* ctrl = NULL;
    struct nvm_ioctl_map request;
    struct map* map = NULL;
	struct nvme_dev *ndev;
	struct nvme_ns *ns;
	struct nvm_ioctl_dev drequest;
	u64 addr;
    ctrl = ctrl_find_by_inode(&ctrl_list, file->f_inode);
    if (ctrl == NULL)
    {
        printk(KERN_CRIT "Unknown controller reference snvm_dev_map_ioctl\n");
        return -EBADF;
    }
    switch (cmd)
    {
        case NVM_MAP_HOST_MEMORY: // 将用户态地址pin住并得到dma地址返回用户态
		{
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
             *                  map on g->maps; group_id == 0 keeps
             *                  it on the controller-global list only.
             */
            if (copy_from_user(&request, (void __user*) arg, sizeof(request)))
            {
                return -EFAULT;
            }
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

            map = map_userspace(&host_list, ctrl, request.vaddr_start, request.n_pages);
            /*
             * Guard against ERR_PTR / NULL *before* any deref. See
             * PORTING.md §7.3.
             */
            if (IS_ERR_OR_NULL(map))
            {
                return IS_ERR(map) ? PTR_ERR(map) : -ENOMEM;
            }

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
                 * UNSPECIFIED-with-group). */
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

            if (copy_to_user((void __user*)(uintptr_t) request.ioaddrs, map->addrs,
                             map->n_addrs * sizeof(uint64_t)))
            {
                /*
                 * Roll back every counter we bumped above AND release
                 * the mapping.  For new-mode (group or data_maps)
                 * maps, unmap_and_release will list_del the
                 * group_link out so the per-list counter is the only
                 * thing to roll back manually.
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
		}

		case NVM_MAP_DEVICE_MEMORY: // 将用户态cuda malloc 分配的地址pin住并得到dma地址返回用户态
		{
			/*
			 * Pin GPU pages (NVIDIA p2p) into device_list.  Same
			 * dual-mode semantics as NVM_MAP_HOST_MEMORY: nonzero
			 * group_id attaches the map to a per-fd group; zero
			 * keeps it on the controller-global list only.
			 *
			 * Note: this is a data-path mapping only; GPU queue
			 * ring registration goes through the per-fd queue
			 * group (NVM_CREATE_QUEUE_GROUP +
			 * NVM_ADD_USER_QUEUE) flow.
			 */
			if (copy_from_user(&request, (void __user*) arg, sizeof(request)))
			{
				return -EFAULT;
			}
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

			map = map_device_memory(&device_list, ctrl, request.vaddr_start, request.n_pages, &ctrl_list);
			if (IS_ERR_OR_NULL(map))
			{
				return IS_ERR(map) ? PTR_ERR(map) : -ENOMEM;
			}

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

			if (copy_to_user((void __user*)(uintptr_t) request.ioaddrs, map->addrs,
			                 map->n_addrs * sizeof(uint64_t)))
			{
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
		}
        case NVM_UNMAP_HOST_MEMORY:
		{
            if (copy_from_user(&addr, (void __user*) arg, sizeof(u64)))
            {
                return -EFAULT;
            }

            map = map_find(&host_list, addr);
            if (map != NULL)
            {
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
                break;
            }
            ret = -EINVAL;
            printk(KERN_WARNING "NVM_UNMAP_HOST_MEMORY Mapping for address %llx not found\n", addr);
            break;
		}
        case NVM_UNMAP_DEVICE_MEMORY:
		{
            if (copy_from_user(&addr, (void __user*) arg, sizeof(u64)))
            {
                return -EFAULT;
            }

            map = map_find(&device_list, addr);
            if (map != NULL)
            {
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
                break;
            }
            ret = -EINVAL;
            printk(KERN_WARNING "NVM_UNMAP_DEVICE_MEMORY Mapping for address %llx not found\n", addr);
            break;
		}
        case NVM_UNMAP_DEVICE_QUEUE_MEMORY:
		{
            if (copy_from_user(&addr, (void __user*) arg, sizeof(u64)))
            {
                return -EFAULT;
            }

            map = map_find(&device_queue_list, addr);
            if (map != NULL)
            {
                unmap_and_release(map);
				ret = 0;
                break;
            }
            ret = -EINVAL;
            printk(KERN_WARNING "NVM_UNMAP_DEVICE_QUEUE_MEMORY Mapping for address %llx not found\n", addr);
            break;
		}
		case NVM_GET_DEV_INFO:
		{
			/*
			 * Race fix (PORTING.md \xc2\xa77.3.1 trap "NVM_GET_DEV_INFO vs
			 * nvme_scan_work"): snvme_start_ctrl() enqueues
			 * nvme_scan_work asynchronously on s_nvme_wq; the cdev
			 * /dev/ssnvme<N> is already callable when the caller of
			 * SNVM_DEVICE_BIND returns, so userspace can legitimately
			 * reach this ioctl *before* the worker has scanned out
			 * nsid=1 and list_add_tail()'d it on ctrl->namespaces.
			 *
			 * Mitigation: flush_work(&ctrl->scan_work) synchronously
			 * waits for the in-flight scan to complete, then retry the
			 * lookup.  flush_work is documented to be safe even when
			 * the work was never queued (it returns false immediately).
			 * Bound the total wait at 5 s to preserve userspace EFAULT
			 * semantics if the controller is genuinely broken (admin
			 * queue dead, scan never enqueued because state never
			 * reached NVME_CTRL_LIVE).
			 */
			unsigned int wait_ms;

			ndev = pci_get_drvdata(ctrl->pdev);
			if (ndev == NULL) {
				printk(" pci_get_drvdata error\n");
				return -EFAULT;
			}

			ns = snvme_find_get_ns(&ndev->ctrl, 1);
			if (!ns) {
				wait_ms = 0;
				flush_work(&ndev->ctrl.scan_work);
				ns = snvme_find_get_ns(&ndev->ctrl, 1);
				while (!ns && wait_ms < 5000) {
					msleep(50);
					wait_ms += 50;
					flush_work(&ndev->ctrl.scan_work);
					ns = snvme_find_get_ns(&ndev->ctrl, 1);
				}
				if (!ns) {
					pr_err("snvme: snvme_find_get_ns(nsid=1) failed after %u ms wait (state=%d)\n",
					       wait_ms, ndev->ctrl.state);
					return -EFAULT;
				}
				pr_info("snvme: NVM_GET_DEV_INFO: nsid=1 ready after %u ms scan wait\n",
					wait_ms);
			}

			/*
			 * Zero the response struct before populating so any
			 * fields the kernel doesn't fill in are deterministic
			 * (the prior code returned stack garbage in q_depth /
			 * max_user_qid / sgl_supported -- now MBZ).
			 */
			memset(&drequest, 0, sizeof(drequest));
			memcpy(drequest.disk_name, ns->disk->disk_name,
			       DISK_NAME_LEN * sizeof(char));
			/*
			 * start_cq_idx: first QID available to user IOQs.
			 * Old path (NVM_SET_SHARE_REG -> probe -> mix) sets
			 * user_start_qid = online_queues at the end of mix.
			 * New path (no SET_SHARE_REG) leaves user_start_qid at
			 * 0; fall back to online_queues so userspace gets a
			 * consistent answer regardless of which flow brought
			 * the controller up.
			 */
			drequest.start_cq_idx  = ndev->user_start_qid
						 ? ndev->user_start_qid
						 : ndev->online_queues;
			drequest.dstrd         = ndev->db_stride;
			drequest.nr_user_q     = ndev->online_user_queues;
			drequest.block_size    = 1 << ns->head->lba_shift;
			/* CTRL.MDTS in BYTES.  max_hw_sectors is the
			 * NVMe-block-layer internal in 512-byte sectors;
			 * convert here so userspace gets a single
			 * format-agnostic byte count.  See ioctl.h
			 * ("CTRL.MDTS in bytes") and the matching fix in
			 * the 5.4.241 baseline. */
			drequest.max_data_size = (size_t)ndev->ctrl.max_hw_sectors << 9;

			/*
			 * B3 fields.  These are the single source of truth for
			 * userspace ring sizing and QID allocation:
			 *
			 *   q_depth                NVMe CAP.MQES + 1, clamped by
			 *                          io_queue_depth module param.
			 *                          Applies to *every* user queue.
			 *   bar0_size              Full BAR0 region size; userspace
			 *                          mmaps up to this many bytes
			 *                          starting at offset 0 to reach
			 *                          all doorbell registers.
			 *   max_user_qid           Highest QID kernel will hand out
			 *                          via NVM_ADD_USER_QUEUE, inclusive.
			 *                          User QID pool is
			 *                          [start_cq_idx, max_user_qid].
			 *                          Sourced from ndev->ctrl_max_io_queues
			 *                          (the controller's authoritative Set
			 *                          Features grant captured in
			 *                          s_nvme_setup_io_queues, prior to any
			 *                          MSI-X clamping).  Using ndev->max_qid
			 *                          instead would under-report on
			 *                          MSI-X-limited hosts and silently
			 *                          shrink the user QID pool.
			 *   max_queues_per_group   Echoes the kernel-fixed cap
			 *                          (NVM_MAX_QUEUES_PER_GROUP) so
			 *                          userspace doesn't have to
			 *                          hardcode the value.
			 *   sgl_supported          Identify Controller SGLS dword;
			 *                          userspace uses this to decide
			 *                          whether CDW0.PSDT=1 is safe.
			 */
			drequest.q_depth              = (uint16_t)ndev->q_depth;
			drequest.bar0_size            = (uint32_t)pci_resource_len(ctrl->pdev, 0);
			drequest.max_user_qid         = ndev->ctrl_max_io_queues;
			drequest.max_queues_per_group = NVM_MAX_QUEUES_PER_GROUP;
			drequest.sgl_supported        = (uint32_t)ndev->ctrl.sgls;

			/* ABI handshake: report the UAPI version and capability
			 * set this kernel was compiled with.  Userspace checks
			 * these in NVM_GET_DEV_INFO's return; mismatch =>
			 * fail-closed.  Old kernels (pre-UAPI-consolidation)
			 * report abi_version == 0 because memset zeroes the
			 * struct; userspace treats 0 as "unknown / legacy". */
			drequest.abi_version          = TUTTI_SNVME_ABI_VERSION;
			drequest.capabilities         = TUTTI_SNVME_CAP_ALL;

			snvme_put_ns(ns);

			if (copy_to_user((void __user*) arg, &drequest,
					 sizeof(struct nvm_ioctl_dev)))
				return -EFAULT;

			ret = 0;
			break;
		}
		case NVM_CREATE_QUEUE_GROUP:
		{
			/*
			 * Allocate a new per-fd queue group.  In B1/B2 the
			 * group is a kernel-side container for maps; Chunk H
			 * will hang user IO queues off of it.
			 *
			 * We don't require the controller to be bound here:
			 * the group itself doesn't touch any NVMe state.
			 * Bind status will be enforced when a child operation
			 * (NVM_ADD_USER_QUEUE) actually needs admin_q.
			 *
			 * Caps:
			 *   per-fd:           NVM_MAX_GROUPS_PER_FD (default 1)
			 *   per-group queues: NVM_MAX_QUEUES_PER_GROUP, echoed
			 *                     back in payload.max_queues so
			 *                     userspace doesn't have to
			 *                     hardcode the value.
			 */
			struct nvm_ioctl_queue_group req;
			struct snvm_dev_owner *own = file->private_data;
			struct snvm_qgroup *g;
			int new_id;

			if (!own)
				return -ENODEV;

			if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
				return -EFAULT;
			if (req.flags != 0)
				return -EINVAL;
			{
				size_t k;
				for (k = 0; k < ARRAY_SIZE(req.reserved); k++)
					if (req.reserved[k] != 0)
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
			 * IDA range starts at 1 -- group_id 0 is reserved as
			 * the "no group" sentinel for userspace.
			 * ida_simple_get's (start, end) is [start, end);
			 * end=0 means "no upper bound", which gives us the
			 * full uint32_t range less id 0.
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
			if (copy_to_user((void __user *)arg, &req, sizeof(req))) {
				/*
				 * Rollback: tear the group back down so the
				 * caller's view (nothing exists) matches the
				 * kernel's view.
				 */
				mutex_lock(&own->groups_lock);
				list_del(&g->link);
				own->nr_groups--;
				destroy_qgroup_locked(g, ctrl);
				mutex_unlock(&own->groups_lock);
				return -EFAULT;
			}

			pr_debug("snvme: NVM_CREATE_QUEUE_GROUP id=%u max_queues=%u pid=%d\n",
				 g->group_id, g->max_queues, current->pid);
			ret = 0;
			break;
		}
		case NVM_DESTROY_QUEUE_GROUP:
		{
			/*
			 * Explicit destroy.  Userspace passes the opaque
			 * group_id (uint32_t); we look it up in the per-fd
			 * group list and tear it down.  Cross-fd destroy is
			 * disallowed by construction: the group descriptor
			 * is only reachable via this fd's owner->groups list,
			 * so a foreign group_id is invisible and returns
			 * -ENOENT.
			 */
			uint32_t group_id;
			struct snvm_dev_owner *own = file->private_data;
			struct snvm_qgroup *g;
			bool found;

			if (!own)
				return -ENODEV;

			if (copy_from_user(&group_id, (void __user *)arg, sizeof(group_id)))
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
				pr_debug("snvme: NVM_DESTROY_QUEUE_GROUP id=%u not found on fd (pid=%d)\n",
					 group_id, current->pid);
				return -ENOENT;
			}

			pr_debug("snvme: NVM_DESTROY_QUEUE_GROUP id=%u pid=%d\n",
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
			struct nvme_dev *uq_ndev;
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

			if (copy_from_user(req, (void __user *)arg, sizeof(*req))) {
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
			uq_ndev = snvm_ctrl_get_live_ndev(ctrl);
			if (!uq_ndev) {
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
			rc = snvm_user_qid_pool_init_locked(ctrl, uq_ndev);
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
				rc = adapter_alloc_cq_user(uq_ndev, cq_maps[i], qids[i]);
				if (rc) {
					pr_warn("snvme: NVM_ADD_USER_QUEUE: Create I/O CQ qid=%u rc=%d\n",
						qids[i], rc);
					goto rollback_unlocked;
				}
				rc = adapter_alloc_sq_user(uq_ndev, sq_maps[i], qids[i]);
				if (rc) {
					pr_warn("snvme: NVM_ADD_USER_QUEUE: Create I/O SQ qid=%u rc=%d\n",
						qids[i], rc);
					/*
					 * SQ failed but CQ for this i was already
					 * created -- delete it before unwinding the
					 * earlier pairs.  Account for this with
					 * created++ first so the rollback loop
					 * picks it up.
					 */
					adapter_delete_cq(uq_ndev, qids[i]);
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
					(uint32_t)(NVME_REG_DBS + qid * 2 * uq_ndev->db_stride * 4);
				req->out_pairs[i].cq_doorbell_offset =
					(uint32_t)(NVME_REG_DBS + (qid * 2 + 1) * uq_ndev->db_stride * 4);
				req->out_pairs[i].qid = qid;
			}
			g->cur_queues += req->nr_pairs;

			mutex_unlock(&own->groups_lock);

			if (copy_to_user((void __user *)arg, req, sizeof(*req))) {
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
					adapter_delete_sq(uq_ndev, uq->qid);
					adapter_delete_cq(uq_ndev, uq->qid);
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

			pr_info("snvme: NVM_ADD_USER_QUEUE group=%u created %u queue(s) (qids %u..%u)\n",
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
				adapter_delete_sq(uq_ndev, qids[i - 1]);
				adapter_delete_cq(uq_ndev, qids[i - 1]);
			}
			mutex_lock(&ctrl->user_qid_lock);
			for (i = 0; i < alloc_n; i++)
				snvm_user_qid_free_locked(ctrl, qids[i]);
			mutex_unlock(&ctrl->user_qid_lock);
			mutex_unlock(&own->groups_lock);
			kfree(req);
			return rc;
		}
		case NVM_SET_KERNEL_IOQ_CAP:
		{
			/*
			 * Cap-only update path: stash setup.cap_kernel_ioq.
			 * Probe segment 6a copies ctrl->setup.cap_kernel_ioq
			 * into dev->cap_kernel_ioq at SNVM_DEVICE_BIND time,
			 * gated on ctrl->setup.valid.
			 *
			 * Must run pre-bind to have any effect.  We do not
			 * reject post-bind calls (the field write is still
			 * useful for the NEXT bind cycle if the user
			 * unbinds/rebinds), but log so a misordered userspace
			 * is diagnosable from dmesg.
			 */
			uint32_t cap;

			if (copy_from_user(&cap, (void __user *)arg, sizeof(cap)))
				return -EFAULT;

			ctrl->setup.cap_kernel_ioq = cap;
			ctrl->setup.valid          = 1;

			pr_info("snvme: NVM_SET_KERNEL_IOQ_CAP cap=%u\n", cap);
			ret = 0;
			break;
		}
        default:
            printk(KERN_NOTICE "Unknown ioctl command from process %d: %u\n",
                    current->pid, cmd);
            ret = -EINVAL;
            break;
    }

    return ret;
}
static int svm_mmap_registers(struct file* file, struct vm_area_struct* vma)
{
	struct ctrl* ctrl = NULL;
    ctrl = ctrl_find_by_inode(&ctrl_list, file->f_inode);
    if (ctrl == NULL || ctrl->pdev == NULL)
    {
        printk(KERN_CRIT "Unknown controller reference svm_mmap_registers\n");
        return -EBADF;
    }

    if (vma->vm_end - vma->vm_start > pci_resource_len(ctrl->pdev, 0))
    {
        printk(KERN_WARNING "Invalid range size\n");
        return -EINVAL;
    }
	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
	return vm_iomap_memory(vma, pci_resource_start(ctrl->pdev, 0), vma->vm_end - vma->vm_start);

}

static int snvm_dev_open(struct inode *inode, struct file *file)
{
	struct ctrl *ctrl;
	struct snvm_dev_owner *own;

	ctrl = ctrl_find_by_inode(&ctrl_list, inode);
	if (!ctrl) {
		pr_err("snvme: snvm_dev_open: no ctrl for inode\n");
		return -ENODEV;
	}

	own = kzalloc(sizeof(*own), GFP_KERNEL);
	if (!own)
		return -ENOMEM;

	own->ctrl  = ctrl;
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
	 * IMPORTANT ordering: groups are drained BEFORE the map passes
	 * below.  Future chunks will park user IO queues and pinned
	 * NVMe ring maps inside group descriptors; if the global map
	 * lists were freed first, the Delete I/O SQ/CQ admin commands
	 * issued during group teardown (Chunk H) would see ring
	 * physical addresses that have already been unmapped from the
	 * IOMMU, which the controller could DMA into freed pages.
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
		pr_info("snvme: snvm_dev_release: cascade-destroyed %u orphan group(s) for pid=%d\n",
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
			pr_info("snvme: snvm_dev_release: cascade-released %u DATA map(s) for pid=%d\n",
				n_data, owner ? owner->pid : -1);
	}

	/* Free legacy (non-group) maps owned by the dying task. */
	n_host = map_purge_by_owner(&host_list,         owner);
	n_dev  = map_purge_by_owner(&device_list,       owner);
	n_devq = map_purge_by_owner(&device_queue_list, owner);

	if (n_host || n_dev || n_devq)
		pr_info("snvme: snvm_dev_release: reclaimed host=%lu dev=%lu devq=%lu for pid=%d\n",
			n_host, n_dev, n_devq,
			owner ? owner->pid : -1);

	mutex_destroy(&own->groups_lock);
	mutex_destroy(&own->data_maps_lock);
	kfree(own);
	file->private_data = NULL;
	return 0;
}

/* Define file operations for device file */
static const struct file_operations snvm_dev_fops =
{
    .owner = THIS_MODULE,
    .open = snvm_dev_open,
    .release = snvm_dev_release,
    .unlocked_ioctl = snvm_dev_map_ioctl,
    .mmap = svm_mmap_registers,
};

static int snvm_chrdev_create(struct pci_dev *pdev, unsigned int class){
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

static unsigned long clear_ctrl_list(struct list* list)
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

static unsigned long clear_map_list(struct list* list)
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

static struct pci_driver snvme_driver = {
	.name		= PCI_DRIVER_NAME,
	.id_table	= nvme_id_table,
	.probe		= nvme_probe,
	.remove		= nvme_remove,
	.shutdown	= nvme_shutdown,
	.driver		= {
		.probe_type	= PROBE_PREFER_ASYNCHRONOUS,
#ifdef CONFIG_PM_SLEEP
		.pm		= &nvme_dev_pm_ops,
#endif
	},
	.sriov_configure = pci_sriov_configure_simple,
	.err_handler	= &nvme_err_handler,
};

static int snvm_register_driver(void)
{
	BUILD_BUG_ON(sizeof(struct nvme_create_cq) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_create_sq) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_delete_queue) != 64);
	BUILD_BUG_ON(IRQ_AFFINITY_MAX_SETS < 2);
	BUILD_BUG_ON(NVME_MAX_SEGS > SGES_PER_PAGE);
	BUILD_BUG_ON(sizeof(struct scatterlist) * NVME_MAX_SEGS > PAGE_SIZE);
	BUILD_BUG_ON(nvme_pci_npages_prp() > NVME_MAX_NR_ALLOCATIONS);

	return pci_register_driver(&snvme_driver);
}

static void snvm_unregister_driver(void)
{
	pci_unregister_driver(&snvme_driver);
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

static int snvm_rebind_driver(struct pci_device_addr dev_addr){
	struct device_driver *dev_drv;
	struct pci_dev *pdev;
	int ret = 0;

	pdev = TO_PCI_DEV(dev_addr);

	if (!pdev) {
		printk("pci_get_domain_bus_and_slot failed\n");
		return -ENODEV;
	}
	// fixme
	dev_drv = pdev->dev.driver;
	if (dev_drv && dev_drv->name) { // whatever driver is binded to aviod automatic binding
		if (pci_is_enabled(pdev)){
			pci_disable_device(pdev);
			printk("(%s): disable device for bind new driver\n", __func__);
		}
		device_release_driver(&pdev->dev);
	}

	printk("start to bind nvme device to snvme: pci %x:%x:%x.%x\n",
														dev_addr.domain,
														dev_addr.bus,
														dev_addr.slot,
														dev_addr.func);

	// fixme: wait for new driver binding

	if (register_driver()){
		printk("register driver error\n");
		pci_dev_put(pdev);
		return -EFAULT;
	}

	// double check to avoid driver automatic binding
	dev_drv = pdev->dev.driver;
	if (dev_drv){
		printk("device driver name: %s\n", dev_drv->name);
	}
	if (!dev_drv){
		ret = device_driver_attach(&snvme_driver.driver, &pdev->dev);
		if (ret){
			printk("%s: device driver attach %d", __func__, ret);
		}
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

static char *get_snvme_mode(const struct device *dev, umode_t *mode) {
    if (mode) *mode = 0666;
    return NULL;
}

static int snvm_cdev_init(void)
{
	int ret;
	struct device *device;

    mutex_init(&snvm_control_lock);

	dev_class = class_create(DRIVER_NAME);
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


static int __init nvme_init(void)
{
    int ret;
    snvm_registered = 0;
	if(peer_memory_ops.init()) {
		printk("Could not load peer_memory symbols\n");
		ret = -EOPNOTSUPP;
		return ret;
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
    // Set up character device creation
    ret = snvm_cdev_init();
	if(ret)
		return ret;
    return 0;
}


static void __exit nvme_exit(void)
{
	int ret;
    unsigned long remaining = 0;


	//clear device mem map
    remaining = clear_map_list(&device_list);
    if (remaining != 0)
    {
        printk(KERN_NOTICE "%lu GPU memory mappings were still in use on unload\n", remaining);
    }
	//clear device io queue map
    remaining = clear_map_list(&device_queue_list);
    if (remaining != 0)
    {
        printk(KERN_NOTICE "%lu GPU memory mappings were still in use on unload\n", remaining);
    }

    remaining = clear_map_list(&host_list);
    if (remaining != 0)
    {
        printk(KERN_NOTICE "%lu host memory mappings were still in use on unload\n", remaining);
    }

	/*
	 * Drop the Phoenix P2P service reference taken in nvme_init.  Safe
	 * here because every map (and thus every phxfs_p2p handle) has
	 * already been drained by the clear_map_list() calls above.
	 */
	map_p2p_service_release();

	clean_driver();

	peer_memory_ops.exit();
	ret = clear_ctrl_list(&ctrl_list);
	if(ret!=curr_ctrls)
		printk("release ctrl error!, cur is %d, release %d",curr_ctrls,ret);
    snvm_cdev_release();

}

MODULE_AUTHOR("Shi Qiu <qiushijsxs@stu.xmu.edu.cn>");
MODULE_LICENSE("GPL");
MODULE_VERSION("1.0");
MODULE_DESCRIPTION("NVMe host PCIe transport driver");
module_init(nvme_init);
module_exit(nvme_exit);

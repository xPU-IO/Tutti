/* SPDX-License-Identifier: GPL-2.0 */
/*
 * snvm_ndev.h - struct nvme_dev definition for the 5.4-tlinux4 baseline.
 * Extracted verbatim from the baseline pci.c so shared translation
 * units (snvm_qgroup.c / snvm_dev_ioctl.c) can access its stable and
 * snvme-private fields.  Keep in sync with upstream + snvme hooks.
 */
#ifndef __SNVM_NDEV_H__
#define __SNVM_NDEV_H__

#include "nvme.h"

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

#endif /* __SNVM_NDEV_H__ */

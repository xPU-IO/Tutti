/* SPDX-License-Identifier: GPL-2.0 */
/*
 * snvm_ndev.h - struct nvme_dev definition for the 5.15 baseline.
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
	struct work_struct remove_work;
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
	u32 *dbbuf_dbs;
	dma_addr_t dbbuf_dbs_dma_addr;
	u32 *dbbuf_eis;
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
	bool attrs_added;
	/*
	 * B3 queue-budget fields, copied from ctrl->setup at nvme_probe
	 * time and consumed in s_nvme_setup_io_queues.
	 *
	 *   ctrl_max_io_queues   Captured from snvme_set_queue_count(),
	 *                        the controller's authoritative IOQ
	 *                        grant.  This is the upper bound on
	 *                        legal QIDs for NVM_ADD_USER_QUEUE
	 *                        (the user QID pool top).  NOT
	 *                        equivalent to dev->max_qid, which is
	 *                        the post-MSI-X kernel-used ceiling
	 *                        and can be smaller on hosts where
	 *                        MSI-X is the bottleneck.
	 *
	 *   cap_kernel_ioq       Mirror of ctrl->setup.cap_kernel_ioq.
	 *                        0 = no override (kernel takes
	 *                        num_possible_cpus() IOQs as usual);
	 *                        N>0 = kernel takes at most N IOQs,
	 *                        leaving the rest of the controller
	 *                        grant for NVM_ADD_USER_QUEUE.
	 */
	unsigned int ctrl_max_io_queues;
	unsigned int cap_kernel_ioq;
};

#endif /* __SNVM_NDEV_H__ */

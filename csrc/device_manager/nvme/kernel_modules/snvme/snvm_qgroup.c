// SPDX-License-Identifier: GPL-2.0
/*
 * snvm_qgroup.c - user-queue admin helpers, B3 user-QID pool, per-fd
 * queue-group lifecycle.  Extracted verbatim from the 6.8 lineage pci.c;
 * shared across baselines.
 */
#include <linux/idr.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/string.h>

#include "snvm_glue.h"
#include "snvm_qgroup.h"
#include "ctrl.h"
#include "map.h"
#include "nvme.h"
#include "compat.h"
#include "snvm_ndev.h"

static int adapter_delete_queue(struct nvme_dev *dev, u8 opcode, u16 id)
{
	struct nvme_command c = { };

	c.delete_queue.opcode = opcode;
	c.delete_queue.qid = cpu_to_le16(id);

	return snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
}

int adapter_alloc_cq_user(struct nvme_dev *dev, struct map* q_map,int qid)
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

int adapter_alloc_sq_user(struct nvme_dev *dev, struct map* q_map,int qid)
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

int adapter_delete_cq(struct nvme_dev *dev, u16 cqid)
{
	return adapter_delete_queue(dev, nvme_admin_delete_cq, cqid);
}

int adapter_delete_sq(struct nvme_dev *dev, u16 sqid)
{
	return adapter_delete_queue(dev, nvme_admin_delete_sq, sqid);
}

DEFINE_IDA(snvm_queue_group_ida);

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
int snvm_user_qid_pool_init_locked(struct ctrl *ctrl,
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
int snvm_user_qid_alloc_locked(struct ctrl *ctrl,
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
void snvm_user_qid_free_locked(struct ctrl *ctrl, uint16_t qid)
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
struct nvme_dev *snvm_ctrl_get_live_ndev(const struct ctrl *ctrl);

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
void destroy_qgroup_locked(struct snvm_qgroup *g, struct ctrl *ctrl)
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
struct nvme_dev *snvm_ctrl_get_live_ndev(const struct ctrl *ctrl)
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

struct snvm_qgroup *find_qgroup_locked(struct snvm_dev_owner *own,
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

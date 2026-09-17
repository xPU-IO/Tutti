/* SPDX-License-Identifier: GPL-2.0 */
/*
 * snvm_qgroup.h - per-fd owner/group descriptors, the B3 user-QID pool
 * and user-queue admin-command helpers.  Shared across baselines;
 * extracted from the 6.8 lineage.
 */
#ifndef __SNVM_QGROUP_H__
#define __SNVM_QGROUP_H__

#include <linux/fs.h>
#include <linux/list.h>
#include <linux/mutex.h>
#include <linux/sched.h>
#include <linux/types.h>

#include "ioctl.h"          /* NVM_MAX_QUEUES_PER_GROUP */

struct ctrl;
struct map;
struct nvme_dev;

/* admin-command helpers (snvm_qgroup.c) */
int adapter_delete_cq(struct nvme_dev *dev, u16 cqid);
int adapter_delete_sq(struct nvme_dev *dev, u16 sqid);
int adapter_alloc_cq_user(struct nvme_dev *dev, struct map *q_map, int qid);
int adapter_alloc_sq_user(struct nvme_dev *dev, struct map *q_map, int qid);

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

#include <linux/idr.h>
extern struct ida snvm_queue_group_ida;
int  snvm_user_qid_pool_init_locked(struct ctrl *ctrl, struct nvme_dev *ndev);
int  snvm_user_qid_alloc_locked(struct ctrl *ctrl, unsigned int nr, uint16_t *qids_out);
void snvm_user_qid_free_locked(struct ctrl *ctrl, uint16_t qid);

struct snvm_qgroup *find_qgroup_locked(struct snvm_dev_owner *own,
				       uint32_t group_id);
void destroy_qgroup_locked(struct snvm_qgroup *g, struct ctrl *ctrl);
struct nvme_dev *snvm_ctrl_get_live_ndev(const struct ctrl *ctrl);

#endif /* __SNVM_QGROUP_H__ */

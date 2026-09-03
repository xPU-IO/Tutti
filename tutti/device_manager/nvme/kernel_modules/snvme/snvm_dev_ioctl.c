// SPDX-License-Identifier: GPL-2.0
/*
 * snvm_dev_ioctl.c - /dev/ssnvme<N> per-fd ioctl surface and fops.
 * Extracted verbatim from the 6.8 lineage pci.c; shared across baselines.
 */
#include <linux/delay.h>
#include <linux/fs.h>
#include <linux/genhd.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/pci.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/uaccess.h>
#include <linux/nvme_ioctl.h>

#include "snvm_glue.h"
#include "snvm_qgroup.h"
#include "snvm_dev_ioctl.h"
#include "ctrl.h"
#include "map.h"
#include "nvme.h"
#include "ioctl.h"
#include "snvm_ndev.h"
#include "compat.h"

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
			drequest.block_size    = 1u << snvm_ns_lba_shift(ns);
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
            /*
             * Host monitoring tools use the standard NVMe controller
             * passthrough ABI for identify/log-health probes.  Reuse the
             * selected baseline's complete implementation (including
             * user-buffer mapping, permissions, timeout, and result
             * handling) rather than duplicating it in this dispatcher.
             */
            if (cmd == NVME_IOCTL_ADMIN_CMD ||
                cmd == NVME_IOCTL_ADMIN64_CMD) {
                struct nvme_dev *admin_ndev;

                admin_ndev = snvm_ctrl_get_live_ndev(ctrl);
                if (!admin_ndev)
                    return -ENODEV;
                return snvme_admin_ioctl(&admin_ndev->ctrl, cmd,
                                        (void __user *)arg,
                                        !!(file->f_mode & FMODE_WRITE));
            }

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
const struct file_operations snvm_dev_fops =
{
    .owner = THIS_MODULE,
    .open = snvm_dev_open,
    .release = snvm_dev_release,
    .unlocked_ioctl = snvm_dev_map_ioctl,
    .mmap = svm_mmap_registers,
};

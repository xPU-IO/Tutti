#ifndef __LIBNVM_HELPER_MAP_H__
#define __LIBNVM_HELPER_MAP_H__

#include "list.h"
#include <linux/types.h>
#include <linux/list.h>           /* struct list_head for group_link  */
#include <linux/mm_types.h>


/* Forward declaration */
struct ctrl;
struct map;


typedef void (*release)(struct map*);


/*
 * Describes a range of mapped memory.
 */
struct map
{
    struct list_node    list;           /* Linked list header (global) */
    struct task_struct* owner;          /* Owner of mapping */
    u64                 vaddr;          /* Starting virtual address */
    struct list*        ctrl_list;
    int                 n_entries;
    struct pci_dev*     pdev;           /* Reference to physical PCI device */
    unsigned long       page_size;      /* Logical page size */
    void*               data;           /* Custom data */
    release             release;        /* Custom callback for unmapping and releasing memory */
    /*
     * Per-fd queue-group attachment (B2) + map-kind tag (B6).
     *
     *   group_id == 0          legacy map; group_link is left as
     *                          an empty list head and the map is
     *                          reachable only via the global
     *                          host_list / device_list / etc.
     *   group_id != 0          map belongs to the per-fd group
     *                          identified by group_id; group_link
     *                          is threaded into that group's
     *                          snvm_qgroup::maps list.
     *
     *   kind                   B6 role tag.  See enum nvm_map_kind
     *                          in libnvm/include/ioctl.h.
     *                            UNSPECIFIED (0)  pre-B6 caller;
     *                                             legacy semantics.
     *                            RING_SQ / RING_CQ
     *                                             group-scoped
     *                                             ring; NVM_ADD_
     *                                             USER_QUEUE
     *                                             enforces match.
     *                            DATA             fd-scoped data
     *                                             buffer; lives on
     *                                             snvm_dev_owner.
     *                                             data_maps; NOT
     *                                             on g->maps; not
     *                                             cascaded by
     *                                             NVM_DESTROY_QUEUE
     *                                             _GROUP.
     *
     * group_link is initialised in create_descriptor (LIST_HEAD-init),
     * so it is safe to list_del() / list_empty() unconditionally
     * regardless of mode.
     */
    struct list_head    group_link;
    uint32_t            group_id;
    uint8_t             kind;           /* enum nvm_map_kind */
    uint8_t             reserved_pad[7];
    unsigned long       n_addrs;        /* Number of mapped pages */
    uint64_t            addrs[];        /* Bus addresses */
};



/*
 * Lock and map userspace pages for DMA.
 */
struct map* map_userspace(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages);



/*
 * Unmap and release memory.
 */
void unmap_and_release(struct map* map);




/*
 * Lock and map GPU device memory.
 */
struct map* map_device_memory(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages, struct list* ctrl_list);

struct map* map_device_ioqueue_memory(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages);


/*
 * Find memory mapping from vaddr and current task
 */
struct map* map_find(const struct list* list, u64 vaddr);

/*
 * snvme: purge every map whose ->owner matches the given task.
 *
 * Used by snvm_dev_fops.release to reclaim pinned host pages / nvidia
 * p2p references when a userspace process exits without issuing
 * NVM_UNMAP_*.  Returns number of descriptors freed.  Each unmap also
 * triggers map->release() (release_user_pages / peer_memory_ops.put_pages
 * / peer_memory_ops.free_dma_mapping), so this is safe to call from
 * file->release context where the process is on its way out.
 */
unsigned long map_purge_by_owner(struct list* list, struct task_struct* owner);

/*
 * Phoenix P2P service probe/release.  Called exactly once from the
 * snvme.ko module init/exit to resolve (and hold a reference to) the
 * phoenixfs phxfs_p2p_* symbols.  Holding the reference for snvme's
 * lifetime keeps phoenixfs pinned (cannot rmmod while snvme is loaded)
 * and avoids per-registration __symbol_get overhead.  See map.c.
 */
void map_p2p_service_probe(void);
void map_p2p_service_release(void);

#endif /* __LIBNVM_HELPER_MAP_H__ */

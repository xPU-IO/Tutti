#include "map.h"
#include "list.h"
#include "ctrl.h"
#include <linux/sched.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/slab.h>
#include <linux/mm_types.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/err.h>
#include <linux/module.h>
#include "peer_memory.h"
#include "compat.h"



struct gpu_region
{
    struct peer_page_table* pages;
    struct peer_dma_mapping** mappings;
    /*
     * Phoenix service path: when non-NULL, pages/mappings are unused
     * and cleanup goes through phx_deregister_fn instead of
     * peer_memory_ops.dma_unmap_pages + peer_memory_ops.put_pages.
     */
    void*                    phx_handle;
};



#define GPU_PAGE_SHIFT  16
#define GPU_PAGE_SIZE   (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_MASK   ~(GPU_PAGE_SIZE - 1)

uint32_t max_num_ctrls = 8;


/*
 * Phoenix P2P service (phoenixfs.ko) integration.
 *
 * When Phoenix has remapped a GPU's BAR via devm_memremap_pages
 * (MEMORY_DEVICE_PCI_P2PDMA), the peer-memory dma_map_pages fails for that
 * GPU.  Phoenix exports phxfs_p2p_register/phxfs_p2p_deregister which
 * pin the GPU pages and return the physical addresses directly as bus
 * addresses (valid under IOMMU=pt).  We resolve these symbols once at
 * module init and hold the reference for the lifetime of snvme -- this
 * both avoids the per-call __symbol_get overhead and guarantees that
 * phoenixfs cannot be unloaded while snvme is loaded.
 *
 * Load order: phoenixfs must be loaded before snvme for the service to
 * be picked up; otherwise snvme falls back to its own nvidia_p2p path
 * for its whole lifetime.
 */
struct phxfs_p2p_handle;
typedef int  (*phxfs_p2p_register_fn)(uint64_t, uint64_t,
                                      struct phxfs_p2p_handle**,
                                      const uint64_t**, uint32_t*);
typedef void (*phxfs_p2p_deregister_fn)(struct phxfs_p2p_handle*);

static phxfs_p2p_register_fn   phx_register_fn;
static phxfs_p2p_deregister_fn phx_deregister_fn;

/* Defined later in this file; used by force_release_gpu_memory's guard. */
void release_gpu_memory(struct map* map);

void map_p2p_service_probe(void)
{
    phx_register_fn   = (phxfs_p2p_register_fn)   __symbol_get("phxfs_p2p_register");
    phx_deregister_fn = (phxfs_p2p_deregister_fn) __symbol_get("phxfs_p2p_deregister");

    if (phx_register_fn != NULL && phx_deregister_fn != NULL)
    {
        printk(KERN_INFO "snvme: phoenix P2P service detected, using it for GPU memory registration\n");
    }
    else
    {
        /* Both symbols must be present as a pair; release any partial hold. */
        if (phx_register_fn != NULL)
        {
            __symbol_put("phxfs_p2p_register");
            phx_register_fn = NULL;
        }
        if (phx_deregister_fn != NULL)
        {
            __symbol_put("phxfs_p2p_deregister");
            phx_deregister_fn = NULL;
        }
    }
}

void map_p2p_service_release(void)
{
    if (phx_register_fn != NULL)
    {
        __symbol_put("phxfs_p2p_register");
        phx_register_fn = NULL;
    }
    if (phx_deregister_fn != NULL)
    {
        __symbol_put("phxfs_p2p_deregister");
        phx_deregister_fn = NULL;
    }
}


static struct map* create_descriptor(const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    unsigned long i;
    struct map* map = NULL;

    map = kvmalloc(struct_size(map, addrs, n_pages), GFP_KERNEL);
    if (map == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return ERR_PTR(-ENOMEM);
    }

    list_node_init(&map->list);
    /*
     * Initialise per-fd queue-group link as an empty self-loop so
     * that list_empty(&map->group_link) is true and list_del()
     * remains safe even when the map is never attached to any
     * group (legacy mode, group_id == 0).
     */
    INIT_LIST_HEAD(&map->group_link);
    map->group_id = 0;
    /*
     * Default the B6 map-kind tag to UNSPECIFIED.  Pre-B6 callers
     * (and every internal helper that builds a map without
     * touching map_kind) keep their existing semantics this way.
     * NVM_MAP_* dispatch in pci.c overrides ->kind based on the
     * caller-supplied request.map_kind.
     */
    map->kind = 0; /* NVM_MAP_KIND_UNSPECIFIED */
    memset(map->reserved_pad, 0, sizeof(map->reserved_pad));

    map->owner = current;
    map->vaddr = vaddr;
    map->pdev = ctrl->pdev;
    map->page_size = 0;
    map->data = NULL;
    map->release = NULL;
    map->n_addrs = n_pages;
    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = 0;
    }

    return map;
}



void unmap_and_release(struct map* map)
{
    list_remove(&map->list);

    /*
     * If this map is attached to a per-fd queue group (group_id !=
     * 0, group_link non-empty), splice it out of the group's
     * maps list.  Done unconditionally via list_del because
     * group_link was INIT_LIST_HEAD'd in create_descriptor: a
     * never-attached map's list_del is a no-op (next/prev point
     * at itself, list_del rewires them and that's that).  The
     * caller is expected to be holding own->groups_lock if the
     * map is on a group list -- pci.c snvm_dev_release / the
     * destroy ioctl handler / NVM_UNMAP_* paths all do.
     */
    list_del(&map->group_link);

    if (map->release != NULL && map->data != NULL)
    {
        map->release(map);
    }

    kvfree(map);
}



struct map* map_find(const struct list* list, u64 vaddr)
{
    const struct list_node* element = list_next(&list->head);
    struct map* map = NULL;

    while (element != NULL)
    {
        map = container_of(element, struct map, list);

        if (map->owner == current)
        {
            if (map->vaddr == (vaddr & PAGE_MASK) || map->vaddr == (vaddr & GPU_PAGE_MASK))
            {
                return map;
            }
        }

        element = list_next(element);
    }

    return NULL;
}

/*
 * snvme: walk `list` and unmap_and_release every descriptor whose
 * ->owner pointer equals `owner`.  Designed for snvm_dev_fops.release
 * cleanup when a userspace process dies without issuing NVM_UNMAP_*.
 *
 * Implementation note: unmap_and_release() does list_remove() on the
 * descriptor, so we must re-fetch list_next() from list->head on every
 * iteration -- saving a "next" pointer up front would dereference a
 * freed node on the next loop.
 */
unsigned long map_purge_by_owner(struct list* list, struct task_struct* owner)
{
    struct list_node* element;
    struct map* map;
    unsigned long freed = 0;

    if (list == NULL || owner == NULL)
        return 0;

    element = list_next(&list->head);
    while (element != NULL)
    {
        map = container_of(element, struct map, list);
        /*
         * Only reap legacy (non-group) maps here.
         *
         * Group-attached maps (group_id != 0) are owned per-fd
         * via the snvm_qgroup descriptor on file->private_data,
         * not per-task.  They are drained by
         * destroy_qgroup_locked() in pci.c during the fd's
         * release Pass 0 (which runs before this purge).
         *
         * If we walked group-attached maps here too, a process
         * holding multiple /dev/ssnvme<N> fds would have one
         * fd's release accidentally tear down maps registered
         * via a sibling fd, because all those fds share the same
         * task_struct as their map->owner.  That was the
         * symptom seen in the B2 smoke test (fd_d's release
         * reclaiming fd_a's group-attached map).
         */
        if (map->owner == owner && map->group_id == 0)
        {
            unmap_and_release(map);
            ++freed;
            /* head changed; restart from the new front */
            element = list_next(&list->head);
            continue;
        }
        element = list_next(element);
    }

    return freed;
}
EXPORT_SYMBOL_GPL(map_purge_by_owner);


static void release_user_pages(struct map* map)
{
    unsigned long i;
    struct page** pages;
    struct device* dev;

    dev = &map->pdev->dev;
    for (i = 0; i < map->n_addrs; ++i)
    {
        dma_unmap_page(dev, map->addrs[i], PAGE_SIZE, DMA_BIDIRECTIONAL);
    }

    pages = (struct page**) map->data;
    for (i = 0; i < map->n_addrs; ++i)
    {
        put_page(pages[i]);
    }

    kvfree(map->data);
    map->data = NULL;

    //printk(KERN_DEBUG "Released %lu host pages\n", map->n_addrs);
}



static long map_user_pages(struct map* map)
{
    unsigned long i;
    long retval;
    struct page** pages;
    struct device* dev;

    pages = (struct page**) kvcalloc(map->n_addrs, sizeof(struct page*), GFP_KERNEL);
    if (pages == NULL)
    {
        printk(KERN_CRIT "Failed to allocate page array\n");
        return -ENOMEM;
    }

    retval = compat_get_user_pages(map->vaddr, map->n_addrs, 1, pages);
    if (retval <= 0)
    {
        kfree(pages);
        printk(KERN_ERR "get_user_pages() failed: %ld\n", retval);
        return retval;
    }

    if (map->n_addrs != retval)
    {
        printk(KERN_WARNING "Requested %lu GPU pages, but only got %ld\n", map->n_addrs, retval);
    }
    map->n_addrs = retval;
    map->page_size = PAGE_SIZE;
    map->data = (void*) pages;
    map->release = release_user_pages;

    dev = &map->pdev->dev;
    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = dma_map_page(dev, pages[i], 0, PAGE_SIZE, DMA_BIDIRECTIONAL);

        retval = dma_mapping_error(dev, map->addrs[i]);
        if (retval != 0)
        {
            printk(KERN_ERR "Failed to map page for some reason\n");
            return retval;
        }
       // printk("map_user_page: device: %02x:%02x.%1x\tvaddr: %llx\ti: %lu\tdma_addr: %llx\n", map->pdev->bus->number, PCI_SLOT(map->pdev->devfn), PCI_FUNC(map->pdev->devfn), (uint64_t) map->vaddr, i, map->addrs[i]);
    }

    return 0;
}



struct map* map_userspace(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    long err;
    struct map* md;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    md->page_size = PAGE_SIZE;

    err = map_user_pages(md);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    //printk(KERN_DEBUG "Mapped %lu host pages starting at address %llx\n",
    //        md->n_addrs, md->vaddr);
    return md;
}




static void force_release_gpu_memory(struct map* map)
{
    struct gpu_region* gd = (struct gpu_region*) map->data;
    struct list* list = map->ctrl_list;

    /*
     * Defensive: snvme never registers this callback for Phoenix maps
     * (Phoenix's own free_cb handles force-release), so we should never
     * get here with a phx_handle set.  Route through release_gpu_memory
     * so the phoenixfs deregister still runs and the module refcount is
     * balanced.
     */
    if (gd != NULL && gd->phx_handle != NULL)
    {
        WARN_ON(1);
        release_gpu_memory(map);
        unmap_and_release(map);
        return;
    }

    if (gd != NULL)
    {
        if (gd->mappings != NULL)
        {
            const struct list_node* element = list_next(&list->head);
            struct ctrl* ctrl;

            uint32_t j = 0;
            while (element != NULL)
            {
                ctrl = container_of(element, struct ctrl, list);
                if (gd->mappings[j] != NULL)
                    peer_memory_ops.dma_unmap_pages(ctrl->pdev, gd->pages, gd->mappings[j++]);

                element = list_next(element);
            }
            kfree(gd->mappings);

        }

        if (gd->pages != NULL)
        {
            peer_memory_ops.free_page_table(gd->pages);
        }

        kfree(gd);
        map->data = NULL;

        printk(KERN_DEBUG "Nvidia driver forcefully reclaimed %lu GPU pages\n", map->n_addrs);
    }

    unmap_and_release(map);
}

static void force_release_gpu_ioqueue_memory(struct map* map)
{
    struct gpu_region* gd = (struct gpu_region*) map->data;

    if (gd != NULL)
    {
        if (gd->mappings != NULL)
        {
            if (gd->mappings[0] != NULL)
                peer_memory_ops.dma_unmap_pages(map->pdev, gd->pages, gd->mappings[0]);
            kfree(gd->mappings);

        }
        if (gd->pages != NULL)
        {
            peer_memory_ops.free_page_table(gd->pages);
        }
        kfree(gd);
        map->data = NULL;
        printk(KERN_DEBUG "Nvidia driver forcefully reclaimed %lu GPU pages\n", map->n_addrs);
    }

    unmap_and_release(map);
}


void release_gpu_memory(struct map* map)
{
    struct gpu_region* gd = (struct gpu_region*) map->data;
    struct list* list = map->ctrl_list;

    if (gd != NULL)
    {
        if (gd->phx_handle != NULL)
        {
            /*
             * Phoenix service path: delegate cleanup to Phoenix.  It
             * does the peer-memory put_pages (or skips it if the pages were
             * already force-reclaimed via its free_cb) + frees its
             * internal state + module_put.  phx_deregister_fn is
             * guaranteed non-NULL here because snvme holds a reference
             * to phoenixfs for its whole lifetime.
             */
            if (phx_deregister_fn != NULL)
                phx_deregister_fn(gd->phx_handle);
            gd->phx_handle = NULL;
            kfree(gd);
            map->data = NULL;
            return;
        }

        if (gd->mappings != NULL)
        {
            const struct list_node* element = list_next(&list->head);
            struct ctrl* ctrl;

            uint32_t j = 0;
            while (element != NULL)
            {
                ctrl = container_of(element, struct ctrl, list);
                if (gd->mappings[j] != NULL)
                    peer_memory_ops.dma_unmap_pages(ctrl->pdev, gd->pages, gd->mappings[j++]);

                element = list_next(element);
            }
            kfree(gd->mappings);

        }

        if (gd->pages != NULL)
        {
            peer_memory_ops.put_pages(0, 0, map->vaddr, gd->pages);
        }

        kfree(gd);
        map->data = NULL;

        //printk(KERN_DEBUG "Released %lu GPU pages\n", map->n_addrs);
    }
}


void release_gpu_ioqueue_memory(struct map* map)
{
    struct gpu_region* gd = (struct gpu_region*) map->data;


    if (gd != NULL)
    {
        if (gd->mappings != NULL)
        {

            if (gd->mappings[0] != NULL)
                peer_memory_ops.dma_unmap_pages(map->pdev, gd->pages, gd->mappings[0]);

            kfree(gd->mappings);

        }
        if (gd->pages != NULL)
        {
            peer_memory_ops.put_pages(0, 0, map->vaddr, gd->pages);
        }

        kfree(gd);
        map->data = NULL;
        //printk(KERN_DEBUG "Released %lu GPU pages\n", map->n_addrs);
    }
}



int map_gpu_memory(struct map* map, struct list* list)
{
    unsigned long i;
    uint32_t j;
    int err;
    struct gpu_region* gd;
    const struct list_node* element;
    struct ctrl* ctrl;

    gd = kmalloc(sizeof(struct gpu_region), GFP_KERNEL);
    if (gd == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return -ENOMEM;
    }

    gd->pages      = NULL;
    gd->mappings   = NULL;
    gd->phx_handle = NULL;

    map->page_size = GPU_PAGE_SIZE;
    map->data = gd;
    map->release = release_gpu_memory;

    /* ---- Try the Phoenix P2P service first (resolved once at init) ---- */
    if (phx_register_fn != NULL)
    {
        struct phxfs_p2p_handle* handle = NULL;
        const uint64_t* ioaddrs = NULL;
        uint32_t n = 0;

        err = phx_register_fn(map->vaddr, GPU_PAGE_SIZE * map->n_addrs,
                              &handle, &ioaddrs, &n);
        if (err == 0 && handle != NULL)
        {
            /*
             * Phoenix pinned the pages and returned bus addresses
             * directly (skipping the peer-memory dma_map_pages, which fails
             * once Phoenix has remapped the GPU BAR).  The handle owns
             * the lifecycle -- including the phoenixfs module refcount
             * -- until release_gpu_memory -> phx_deregister_fn.
             */
            gd->phx_handle = handle;
            map->n_addrs = n;
            for (i = 0; i < n; ++i)
                map->addrs[i] = ioaddrs[i];
            return 0;
        }
        /*
         * Registration failed for this GPU -- fall through to the
         * normal nvidia_p2p path.
         */
    }

    /* ---- Normal path: peer_memory get_pages + dma_map_pages ---- */
    gd->mappings = (struct peer_dma_mapping**)  kmalloc(sizeof(struct peer_dma_mapping*) * max_num_ctrls, GFP_KERNEL);

    if (gd->mappings == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        /* gd is freed by release_gpu_memory via unmap_and_release */
        return -ENOMEM;
    }
    for (j = 0; j < max_num_ctrls; j++)
        gd->mappings[j] = NULL;

    // get the io addr
    err = peer_memory_ops.get_pages(0, 0, map->vaddr, GPU_PAGE_SIZE * map->n_addrs, &gd->pages,
            (void (*)(void*)) force_release_gpu_memory, map);
    if (err != 0)
    {
        printk(KERN_ERR "peer_memory_ops.get_pages() failed: %d\n", err);
        return err;
    }

    element = list_next(&list->head);

    // create the map between each nvme device an GPU
    j = 0;
    while (element != NULL)
    {
        ctrl = container_of(element, struct ctrl, list);

        err = peer_memory_ops.dma_map_pages(ctrl->pdev, gd->pages, gd->mappings + j);
        if (err != 0)
        {
            //printk(KERN_ERR "peer_memory_ops.dma_map_pages() failed for nvme%u: %d\n", j-1, err);
            return err;
        }
        j++;
        //for (i = 0; i < map->n_addrs; ++i)
        //{

        //   printk("device: %u\ti: %lu\tpaddr: %llx\n", (j-1), i, (uint64_t)  gd->mappings[j-1]->dma_addresses[i]);
        //}
        if (j == 1) {
            for (i = 0; i < map->n_addrs; ++i)
            {
                map->addrs[i] = peer_memory_dm_addresses(gd->mappings[0])[i];
                //printk("++paddr: %llx\n", (uint64_t) map->addrs[i]);
            }
        }
        element = list_next(element);
    }




    if (map->n_addrs != peer_memory_pt_entries(gd->pages))
    {
        printk(KERN_WARNING "Requested %lu GPU pages, but only got %u\n", map->n_addrs, peer_memory_pt_entries(gd->pages));
    }

    map->n_addrs = peer_memory_pt_entries(gd->pages);

    //printk("vaddr: %llx\n", (uint64_t) map->vaddr);
//    for (j = 0; j < map->n_addrs; j++)
//        printk("\tpaddr: %llx\n", (uint64_t) map->addrs[j]);

    return 0;
}

int map_gpu_ioqueue_memory(struct map* map)
{
    unsigned long i;
    int err;
    struct gpu_region* gd;
    gd = kmalloc(sizeof(struct gpu_region), GFP_KERNEL);
    if (gd == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return -ENOMEM;
    }

    gd->mappings = (struct peer_dma_mapping**)  kmalloc(sizeof(struct peer_dma_mapping*) * 1, GFP_KERNEL);

    if (gd->mappings == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        kfree(gd);
        return -ENOMEM;
    }

    gd->pages = NULL;
    //gd->mappings = NULL;

    map->page_size = GPU_PAGE_SIZE;
    map->data = gd;
    map->release = release_gpu_ioqueue_memory;

    // get the io addr
    err = peer_memory_ops.get_pages(0, 0, map->vaddr, GPU_PAGE_SIZE * map->n_addrs, &gd->pages,
            (void (*)(void*)) force_release_gpu_ioqueue_memory, map);
    if (err != 0)
    {
        printk(KERN_ERR "peer_memory_ops.get_pages() failed: %d\n", err);
        return err;
    }

    err = peer_memory_ops.dma_map_pages(map->pdev, gd->pages, &gd->mappings[0]);
    if (err != 0)
    {
        //printk(KERN_ERR "peer_memory_ops.dma_map_pages() failed for nvme%u: %d\n", j-1, err);
        return err;
    }

    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = peer_memory_dm_addresses(gd->mappings[0])[i];
        //printk("++paddr: %llx\n", (uint64_t) map->addrs[i]);
    }


    if (map->n_addrs != peer_memory_pt_entries(gd->pages))
    {
        printk(KERN_WARNING "Requested %lu GPU pages, but only got %u\n", map->n_addrs, peer_memory_pt_entries(gd->pages));
    }

    map->n_addrs = peer_memory_pt_entries(gd->pages);

    //printk("vaddr: %llx\n", (uint64_t) map->vaddr);
//    for (j = 0; j < map->n_addrs; j++)
//        printk("\tpaddr: %llx\n", (uint64_t) map->addrs[j]);

    return 0;
}




struct map* map_device_memory(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages, struct list* ctrl_list)
{
    int err;
    struct map* md = NULL;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & GPU_PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    md->page_size = GPU_PAGE_SIZE;
    md->ctrl_list = ctrl_list;
    err = map_gpu_memory(md, ctrl_list);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    //printk(KERN_DEBUG "Mapped %lu GPU pages starting at address %llx\n",
    //        md->n_addrs, md->vaddr);
    return md;
}

struct map* map_device_ioqueue_memory(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    int err;
    struct map* md = NULL;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & GPU_PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }
    md->page_size = GPU_PAGE_SIZE;
    err = map_gpu_ioqueue_memory(md);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    //printk(KERN_DEBUG "Mapped %lu GPU pages starting at address %llx\n",
    //        md->n_addrs, md->vaddr);
    return md;
}

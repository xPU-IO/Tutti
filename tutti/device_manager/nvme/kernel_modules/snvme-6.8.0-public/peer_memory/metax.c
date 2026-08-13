/*
 * metax.c -- Metax P2P backend for peer_memory.
 *
 * Resolves metax_p2p_* symbols at module-load time via __symbol_get
 * (symmetric to the NVIDIA backend), so snvme.ko has no build-time
 * dependency on the Metax driver module -- only on its vendor-supplied
 * metax_p2p.h header.
 *
 * If the Metax driver is not loaded, peer_init() fails gracefully and
 * the module load fails with an error message (same behavior as the
 * NVIDIA path when nvidia_p2p_* symbols are missing).
 *
 * Compile this backend with: make TUTTI_P2P_BACKEND=metax
 *
 * Wrapper structs (opaque to the rest of the module):
 *   struct peer_page_table { void *handle; struct sg_table *sgt; };
 *   struct peer_dma_mapping { struct sg_table *sgt; };
 *
 * NVIDIA→Metax API mapping:
 *   nvidia_p2p_get_pages         → metax_p2p_acquire_mem + metax_p2p_get_mem
 *   nvidia_p2p_put_pages         → metax_p2p_put_mem + metax_p2p_release_mem
 *   nvidia_p2p_dma_map_pages     → bus addresses come from metax_p2p_get_mem's sgt
 *   nvidia_p2p_dma_unmap_pages   → metax_p2p_put_mem (no-op wrapper)
 *   nvidia_p2p_free_page_table   → metax_p2p_release_mem
 *   nvidia_p2p_free_dma_mapping  → kfree (wrapper only)
 *
 * SPDX-License-Identifier: GPL-2.0
 */
#define pr_fmt(fmt) "snvme: " fmt

#include <linux/module.h>
#include <linux/version.h>
#include <linux/mutex.h>
#include <linux/printk.h>
#include <linux/pci.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>

#include "peer_memory.h"

/* remove dependency for metax_p2p.h */


#define nvfs_err(FMT, ARGS...) pr_err("metax-fs: " FMT, ## ARGS)
#define nvfs_dbg(FMT, ARGS...)                               \
        do {                                                 \
                if (unlikely(nvfs_dbg_enabled))             \
                        pr_info("metax-fs: " FMT, ## ARGS); \
        } while (0)

int nvfs_dbg_enabled = 0;

struct peer_page_table {
	void *handle;             /* metax_p2p_acquire_mem handle */
	struct sg_table *sgt;     /* from metax_p2p_get_mem */
	uint32_t entries;         /* requested page count (length / page_size) */
	uint64_t length;
};

struct peer_dma_mapping {
	struct sg_table *sgt;     /* bus addresses (same sgt as page_table) */
	uint64_t *dma_addresses;  /* allocated array of bus addresses with offset applied */
};

/* Function-pointer typedefs (resolved at runtime via __symbol_get). */
typedef int (*metax_p2p_acquire_mem_t)(uint64_t, size_t, void **,
		int (*)(void *), void *);
typedef int (*metax_p2p_get_mem_t)(void *, struct sg_table **);
typedef int (*metax_p2p_put_mem_t)(void *, struct sg_table *);
typedef void (*metax_p2p_release_mem_t)(void *);
typedef uint32_t (*metax_p2p_get_page_size_t)(void *);

static metax_p2p_acquire_mem_t   metax_acquire_mem_p   = NULL;
static metax_p2p_get_mem_t       metax_get_mem_p       = NULL;
static metax_p2p_put_mem_t       metax_put_mem_p       = NULL;
static metax_p2p_release_mem_t   metax_release_mem_p   = NULL;
static metax_p2p_get_page_size_t metax_get_page_size_p = NULL;

static inline void metax_peer_put_symbols(void)
{
	if (metax_acquire_mem_p) {
		__symbol_put("metax_p2p_acquire_mem");
		metax_acquire_mem_p = NULL;
	}
	if (metax_get_mem_p) {
		__symbol_put("metax_p2p_get_mem");
		metax_get_mem_p = NULL;
	}
	if (metax_put_mem_p) {
		__symbol_put("metax_p2p_put_mem");
		metax_put_mem_p = NULL;
	}
	if (metax_release_mem_p) {
		__symbol_put("metax_p2p_release_mem");
		metax_release_mem_p = NULL;
	}
	if (metax_get_page_size_p) {
		__symbol_put("metax_p2p_get_page_size");
		metax_get_page_size_p = NULL;
	}
}

static int metax_peer_init(void)
{
#ifdef HAVE_MODULE_MUTEX
	mutex_lock(&module_mutex);
#endif
	if (!metax_acquire_mem_p)
		metax_acquire_mem_p = __symbol_get("metax_p2p_acquire_mem");
	if (!metax_get_mem_p)
		metax_get_mem_p = __symbol_get("metax_p2p_get_mem");
	if (!metax_put_mem_p)
		metax_put_mem_p = __symbol_get("metax_p2p_put_mem");
	if (!metax_release_mem_p)
		metax_release_mem_p = __symbol_get("metax_p2p_release_mem");
	if (!metax_get_page_size_p)
		metax_get_page_size_p = __symbol_get("metax_p2p_get_page_size");

	if (!metax_acquire_mem_p || !metax_get_mem_p || !metax_put_mem_p ||
	    !metax_release_mem_p || !metax_get_page_size_p) {
		nvfs_err("Metax P2P symbols not found (is metax driver loaded?)\n");
		goto error;
	}

#ifdef HAVE_MODULE_MUTEX
	mutex_unlock(&module_mutex);
#endif
	return 0;

error:
#ifdef HAVE_MODULE_MUTEX
	mutex_unlock(&module_mutex);
#endif
	metax_peer_put_symbols();
	return -1;
}

static void metax_peer_exit(void)
{
#ifdef HAVE_MODULE_MUTEX
	mutex_lock(&module_mutex);
#endif
	metax_peer_put_symbols();
#ifdef HAVE_MODULE_MUTEX
	mutex_unlock(&module_mutex);
#endif
}

static int metax_peer_get_pages(uint64_t p2p_token, uint32_t va_space,
			  uint64_t vaddr, uint64_t length,
			  struct peer_page_table **pt,
			  void (*free_cb)(void *data), void *data)
{
	void *handle = NULL;
	struct sg_table *sgt = NULL;
	struct peer_page_table *npt;
	int ret;

	if (!metax_acquire_mem_p || !metax_get_mem_p)
		return -ENOSYS;

	ret = metax_acquire_mem_p(vaddr, (size_t)length, &handle,
				   (int (*)(void *))free_cb, data);
	nvfs_dbg("metax_p2p_acquire_mem: virtual_address=0x%llx, length=0x%llx, handle=%p, pt=%p, ret=%d\n",
		 (unsigned long long)vaddr, (unsigned long long)length, handle, pt, ret);
	if (ret) {
		nvfs_err("metax_p2p_acquire_mem failed: %d\n", ret);
		return ret;
	}

	ret = metax_get_mem_p(handle, &sgt);
	nvfs_dbg("metax_p2p_get_mem: handle=%p, sgt=%p, nents=%d, err=%d\n", handle, sgt,
			(NULL != sgt ?sgt->nents: 0), ret);
	if (ret) {
		nvfs_err("metax_p2p_get_mem failed: %d\n", ret);
		metax_release_mem_p(handle);
		return ret;
	}

	npt = kzalloc(sizeof(*npt), GFP_KERNEL);
	if (!npt) {
		metax_put_mem_p(handle, sgt);
		metax_release_mem_p(handle);
		return -ENOMEM;
	}
	npt->handle = handle;
	npt->sgt = sgt;
	npt->length = length;

	*pt = npt;
	return 0;
}

static int metax_peer_put_pages(uint64_t p2p_token, uint32_t va_space,
			  uint64_t vaddr, struct peer_page_table *pt)
{
	if (!pt) return -EINVAL;
	if (metax_put_mem_p) {
		nvfs_dbg("metax_p2p_put_mem: handle=%p, sgt=%p\n", pt->handle, pt->sgt);
		metax_put_mem_p(pt->handle, pt->sgt);
	}

	if (metax_release_mem_p) {
		nvfs_dbg("metax_release_mem: handle=%p\n", pt->handle);
		metax_release_mem_p(pt->handle);
	}

	kfree(pt);
	return 0;
}

static int metax_peer_dma_map_pages(struct pci_dev *peer,
			      struct peer_page_table *pt,
			      struct peer_dma_mapping **dm,
				  uint32_t expect_page_size)
{
	struct peer_dma_mapping *ndm;
	struct scatterlist *sg;
	uint32_t expect_page_num;
	uint32_t i;
	uint32_t entry = 0;
	uint64_t *dma_addrs;

	if (!pt || !pt->sgt || !pt->sgt->sgl) return -EINVAL;

	ndm = kzalloc(sizeof(*ndm), GFP_KERNEL);
	if (!ndm) return -ENOMEM;

	expect_page_num = (pt->length)/ expect_page_size;
	dma_addrs = kmalloc(expect_page_num * sizeof(uint64_t), GFP_KERNEL);
	if (!dma_addrs) {
		kfree(ndm);
		return -ENOMEM;
	}

	for_each_sg(pt->sgt->sgl, sg, pt->sgt->nents, i) {
		uint64_t addr = sg_dma_address(sg);
		uint32_t len = sg_dma_len(sg);
		uint32_t pages = len / expect_page_size;
		uint32_t j;

		for (j = 0; j < pages; j++) {
			if (entry < expect_page_num) {
				dma_addrs[entry] = addr + j * expect_page_size;
				entry++;
			}
		}
	}

	ndm->sgt = pt->sgt;
	ndm->dma_addresses = dma_addrs;
	pt->entries = entry;
	*dm = ndm;
	return 0;
}

static int metax_peer_dma_unmap_pages(struct pci_dev *peer,
				struct peer_page_table *pt,
				struct peer_dma_mapping *dm)
{
	if (dm) {
		if (dm->dma_addresses)
			kfree(dm->dma_addresses);
		kfree(dm);
	}
	return 0;
}

static int metax_peer_free_dma_mapping(struct peer_dma_mapping *dm)
{
	if (dm) {
		if (dm->dma_addresses)
			kfree(dm->dma_addresses);
		kfree(dm);
	}
	return 0;
}

static int metax_peer_free_page_table(struct peer_page_table *pt)
{
	if (!pt) return -EINVAL;
	kfree(pt);
	return 0;
}

static uint32_t metax_pt_entries(const struct peer_page_table *pt)
{
	if (!pt || !pt->sgt) return 0;
	return pt->entries;
}

static const uint64_t *metax_dm_addresses(const struct peer_dma_mapping *dm)
{
	static uint64_t fallback = 0;
	if (!dm || !dm->dma_addresses) return &fallback;
	return dm->dma_addresses;
}

const struct peer_memory_ops peer_memory_ops = {
	.init              = metax_peer_init,
	.exit              = metax_peer_exit,
	.get_pages         = metax_peer_get_pages,
	.put_pages         = metax_peer_put_pages,
	.dma_map_pages     = metax_peer_dma_map_pages,
	.dma_unmap_pages   = metax_peer_dma_unmap_pages,
	.free_dma_mapping  = metax_peer_free_dma_mapping,
	.free_page_table   = metax_peer_free_page_table,
	.pt_entries        = metax_pt_entries,
	.dm_addresses      = metax_dm_addresses,
};

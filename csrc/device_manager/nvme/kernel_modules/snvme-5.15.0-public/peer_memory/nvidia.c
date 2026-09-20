/*
 * nvidia.c -- NVIDIA P2P backend for peer_memory.
 *
 * Resolves nvidia_p2p_* symbols at module-load time via __symbol_get
 * (not at link time), so snvme.ko has no build-time dependency on the
 * NVIDIA driver module -- only on its header (nv-p2p.h).
 *
 * Compile this backend with: make TUTTI_P2P_BACKEND=nvidia (CUDA default)
 *
 * SPDX-License-Identifier: GPL-2.0
 */
#define pr_fmt(fmt) "snvme: " fmt

#include <linux/module.h>
#include <linux/version.h>
#include <linux/mutex.h>
#include <linux/printk.h>
#include <linux/pci.h>

#include "peer_memory.h"

#if defined(__has_include)
#  if __has_include("nv-p2p.h")
#    include "nv-p2p.h"
#  else
#    error "nv-p2p.h not found: NVIDIA P2P headers are required to build" \
           " the nvidia backend. Ensure the NVIDIA driver source tree is" \
           " installed (e.g. /usr/src/nvidia-*/) and its path is passed" \
           " via the module ccflags (-I<driver_include>)."
#  endif
#else
#  include "nv-p2p.h"
#endif

/* Logging (shared across backends — each .c defines its own copy so
 * the backends are fully independent translation units). */
#define nvfs_err(FMT, ARGS...) pr_err("nvidia-fs: " FMT, ## ARGS)
#define nvfs_dbg(FMT, ARGS...)                               \
        do {                                                 \
                if (unlikely(nvfs_dbg_enabled))             \
                        pr_info("nvidia-fs: " FMT, ## ARGS); \
        } while (0)

int nvfs_dbg_enabled = 0;

/* NVIDIA function-pointer typedefs (resolved at runtime). */
typedef int (*nvidia_p2p_dma_unmap_pages_fptr)(struct pci_dev *,
		struct nvidia_p2p_page_table *,
		struct nvidia_p2p_dma_mapping *);
typedef int (*nvidia_p2p_get_pages_fptr)(uint64_t, uint32_t,
		uint64_t, uint64_t,
		struct nvidia_p2p_page_table **,
		void (*free_callback)(void *data), void *);
typedef int (*nvidia_p2p_put_pages_fptr)(uint64_t, uint32_t,
		uint64_t,
		struct nvidia_p2p_page_table *);
typedef int (*nvidia_p2p_dma_map_pages_fptr)(struct pci_dev *,
		struct nvidia_p2p_page_table *,
		struct nvidia_p2p_dma_mapping **);
typedef int (*nvidia_p2p_free_dma_mapping_fptr)(struct nvidia_p2p_dma_mapping *);
typedef int (*nvidia_p2p_free_page_table_fptr)(struct nvidia_p2p_page_table *);

static nvidia_p2p_dma_unmap_pages_fptr  nvidia_p2p_dma_unmap_pages_p = NULL;
static nvidia_p2p_get_pages_fptr        nvidia_p2p_get_pages_p = NULL;
static nvidia_p2p_put_pages_fptr        nvidia_p2p_put_pages_p = NULL;
static nvidia_p2p_dma_map_pages_fptr    nvidia_p2p_dma_map_pages_p = NULL;
static nvidia_p2p_free_dma_mapping_fptr nvidia_p2p_free_dma_mapping_p = NULL;
static nvidia_p2p_free_page_table_fptr  nvidia_p2p_free_page_table_p = NULL;

static inline void nvidia_peer_put_symbols(void)
{
	if (nvidia_p2p_dma_unmap_pages_p) {
		__symbol_put("nvidia_p2p_dma_unmap_pages");
		nvidia_p2p_dma_unmap_pages_p = NULL;
	}
	if (nvidia_p2p_get_pages_p) {
		__symbol_put("nvidia_p2p_get_pages");
		nvidia_p2p_get_pages_p = NULL;
	}
	if (nvidia_p2p_put_pages_p) {
		__symbol_put("nvidia_p2p_put_pages");
		nvidia_p2p_put_pages_p = NULL;
	}
	if (nvidia_p2p_dma_map_pages_p) {
		__symbol_put("nvidia_p2p_dma_map_pages");
		nvidia_p2p_dma_map_pages_p = NULL;
	}
	if (nvidia_p2p_free_dma_mapping_p) {
		__symbol_put("nvidia_p2p_free_dma_mapping");
		nvidia_p2p_free_dma_mapping_p = NULL;
	}
	if (nvidia_p2p_free_page_table_p) {
		__symbol_put("nvidia_p2p_free_page_table");
		nvidia_p2p_free_page_table_p = NULL;
	}
}

static int nvidia_peer_init(void)
{
#ifdef HAVE_MODULE_MUTEX
	mutex_lock(&module_mutex);
#endif
	if (nvidia_p2p_dma_unmap_pages_p == NULL) {
		nvidia_p2p_dma_unmap_pages_p = __symbol_get("nvidia_p2p_dma_unmap_pages");
		if (!nvidia_p2p_dma_unmap_pages_p) { nvfs_err("symbol nvidia_p2p_dma_unmap_pages not found\n"); goto error; }
	}
	if (nvidia_p2p_get_pages_p == NULL) {
		nvidia_p2p_get_pages_p = __symbol_get("nvidia_p2p_get_pages");
		if (!nvidia_p2p_get_pages_p) { nvfs_err("symbol nvidia_p2p_get_pages not found\n"); goto error; }
	}
	if (nvidia_p2p_put_pages_p == NULL) {
		nvidia_p2p_put_pages_p = __symbol_get("nvidia_p2p_put_pages");
		if (!nvidia_p2p_put_pages_p) { nvfs_err("symbol nvidia_p2p_put_pages not found\n"); goto error; }
	}
	if (nvidia_p2p_dma_map_pages_p == NULL) {
		nvidia_p2p_dma_map_pages_p = __symbol_get("nvidia_p2p_dma_map_pages");
		if (!nvidia_p2p_dma_map_pages_p) { nvfs_err("symbol nvidia_p2p_dma_map_pages not found\n"); goto error; }
	}
	if (nvidia_p2p_free_dma_mapping_p == NULL) {
		nvidia_p2p_free_dma_mapping_p = __symbol_get("nvidia_p2p_free_dma_mapping");
		if (!nvidia_p2p_free_dma_mapping_p) { nvfs_err("symbol nvidia_p2p_free_dma_mapping not found\n"); goto error; }
	}
	if (nvidia_p2p_free_page_table_p == NULL) {
		nvidia_p2p_free_page_table_p = __symbol_get("nvidia_p2p_free_page_table");
		if (!nvidia_p2p_free_page_table_p) { nvfs_err("symbol nvidia_p2p_free_page_table not found\n"); goto error; }
	}
#ifdef HAVE_MODULE_MUTEX
	mutex_unlock(&module_mutex);
#endif
	return 0;

error:
#ifdef HAVE_MODULE_MUTEX
	mutex_unlock(&module_mutex);
#endif
	nvidia_peer_put_symbols();
	return -1;
}

static void nvidia_peer_exit(void)
{
#ifdef HAVE_MODULE_MUTEX
	mutex_lock(&module_mutex);
#endif
	nvidia_peer_put_symbols();
#ifdef HAVE_MODULE_MUTEX
	mutex_unlock(&module_mutex);
#endif
}

static int nvidia_peer_get_pages(uint64_t p2p_token, uint32_t va_space,
			  uint64_t vaddr, uint64_t length,
			  struct peer_page_table **pt,
			  void (*free_cb)(void *data), void *data)
{
	struct nvidia_p2p_page_table *npt = NULL;
	int ret;
	if (!nvidia_p2p_get_pages_p)
		return -ENOMEM;
	ret = nvidia_p2p_get_pages_p(p2p_token, va_space, vaddr, length,
				     &npt, free_cb, data);
	if (ret == 0)
		*pt = (struct peer_page_table *)npt;
	return ret;
}

static int nvidia_peer_put_pages(uint64_t p2p_token, uint32_t va_space,
			  uint64_t vaddr, struct peer_page_table *pt)
{
	if (!nvidia_p2p_put_pages_p)
		return -ENOMEM;
	return nvidia_p2p_put_pages_p(p2p_token, va_space, vaddr,
				      (struct nvidia_p2p_page_table *)pt);
}

static int nvidia_peer_dma_map_pages(struct pci_dev *peer,
			      struct peer_page_table *pt,
			      struct peer_dma_mapping **dm,
				  uint32_t)
{
	struct nvidia_p2p_dma_mapping *ndm = NULL;
	int ret;
	if (!nvidia_p2p_dma_map_pages_p)
		return -ENOMEM;
	ret = nvidia_p2p_dma_map_pages_p(peer,
					 (struct nvidia_p2p_page_table *)pt,
					 &ndm);
	if (ret == 0)
		*dm = (struct peer_dma_mapping *)ndm;
	return ret;
}

static int nvidia_peer_dma_unmap_pages(struct pci_dev *peer,
				struct peer_page_table *pt,
				struct peer_dma_mapping *dm)
{
	if (!nvidia_p2p_dma_unmap_pages_p)
		return -ENOMEM;
	return nvidia_p2p_dma_unmap_pages_p(peer,
					    (struct nvidia_p2p_page_table *)pt,
					    (struct nvidia_p2p_dma_mapping *)dm);
}

static int nvidia_peer_free_dma_mapping(struct peer_dma_mapping *dm)
{
	if (!nvidia_p2p_free_dma_mapping_p)
		return -ENOMEM;
	return nvidia_p2p_free_dma_mapping_p((struct nvidia_p2p_dma_mapping *)dm);
}

static int nvidia_peer_free_page_table(struct peer_page_table *pt)
{
	if (!nvidia_p2p_free_page_table_p)
		return -ENOMEM;
	return nvidia_p2p_free_page_table_p((struct nvidia_p2p_page_table *)pt);
}

static uint32_t nvidia_pt_entries(const struct peer_page_table *pt)
{
	return ((const struct nvidia_p2p_page_table *)pt)->entries;
}

static const uint64_t *nvidia_dm_addresses(const struct peer_dma_mapping *dm)
{
	return ((const struct nvidia_p2p_dma_mapping *)dm)->dma_addresses;
}

const struct peer_memory_ops peer_memory_ops = {
	.init              = nvidia_peer_init,
	.exit              = nvidia_peer_exit,
	.get_pages         = nvidia_peer_get_pages,
	.put_pages         = nvidia_peer_put_pages,
	.dma_map_pages     = nvidia_peer_dma_map_pages,
	.dma_unmap_pages   = nvidia_peer_dma_unmap_pages,
	.free_dma_mapping  = nvidia_peer_free_dma_mapping,
	.free_page_table   = nvidia_peer_free_page_table,
	.pt_entries        = nvidia_pt_entries,
	.dm_addresses      = nvidia_dm_addresses,
};

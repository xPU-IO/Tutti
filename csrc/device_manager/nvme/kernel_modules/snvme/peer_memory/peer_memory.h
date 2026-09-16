/*
 * peer_memory.h -- Vendor-neutral GPU peer-memory abstraction.
 *
 * This header defines the function-pointer table through which the snvme
 * module interacts with ANY GPU vendor's P2P memory API.  The module body
 * (map.c, pci.c) calls only through peer_memory_ops and uses the opaque
 * types declared here; it never references vendor-specific symbols
 * (nvidia_p2p_*, metax_p2p_*) or includes vendor headers directly.
 *
 * Backends:
 *   nvidia (CUDA default)     — NVIDIA nvidia_p2p_* via __symbol_get
 *   metax  (MUSA/MACA default) — Metax metax_p2p_* via __symbol_get
 *
 * The backend is selected at compile time through the Kbuild variable
 * TUTTI_P2P_BACKEND=<name>. Only peer_memory/<name>.c is compiled into a
 * given snvme.ko.
 *
 * Vendor porting:
 *   To add a new backend, implement every member of peer_memory_ops
 *   (including the accessors) in peer_memory/<name>.c. See the existing
 *   NVIDIA / Metax files for the pattern. The opaque types
 *   peer_page_table / peer_dma_mapping can be typedef'd to whatever
 *   wrapper the backend needs (NVIDIA uses the raw vendor types; Metax
 *   uses a custom wrapper struct holding a handle + sg_table).
 *
 * SPDX-License-Identifier: GPL-2.0
 */
#ifndef SNVME_PEER_MEMORY_H
#define SNVME_PEER_MEMORY_H

#include <linux/types.h>
#include <linux/compiler.h>

/*
 * 跨厂商统一接口里，部分参数只被某些后端使用：例如 dma_map_pages 的第 4 个
 * 参数 expect_page_size 是 metax 后端用于按页大小拆分 DMA 地址的，而 NVIDIA
 * 的 nvidia_p2p_dma_map_pages 只有 3 个参数、用不到它。为保持 vendor-neutral
 * 的 peer_memory_ops 签名统一，不使用该参数的后端用本宏标记参数名——既满足老
 * 内核（gnu89）要求函数定义参数必须有名字，又抑制 -Wunused-parameter 告警。
 *
 * Some fields of the vendor-neutral ops table are only consumed by certain
 * backends: e.g. dma_map_pages' 4th arg expect_page_size is used by metax to
 * split DMA addresses per page, while NVIDIA's nvidia_p2p_dma_map_pages takes
 * only 3 args. Backends that don't use an arg tag it with this macro, which
 * both satisfies gnu89 (function-definition params must be named) and
 * suppresses -Wunused-parameter.
 */
#define SNVME_P2P_UNUSED __maybe_unused

struct pci_dev;

/* Opaque handles for GPU P2P page-table / dma-mapping objects.
 * The actual type is defined by the selected backend source;
 * the rest of the module only sees these forward declarations. */
struct peer_page_table;
struct peer_dma_mapping;

/*
 * peer_memory_ops -- function-pointer table for all GPU P2P operations.
 *
 * Every pin / unpin / dma-map / page-table / accessor operation goes
 * through this table.  Each backend (NVIDIA / Metax / future) provides
 * its own implementation of every member.
 */
struct peer_memory_ops {
	/*
	 * init: resolve vendor symbols / prepare backend.  Returns 0 on
	 *       success, -1 if setup fails (module load should fail).
	 * exit: release all resolved symbols / teardown backend.
	 */
	int  (*init)(void);
	void (*exit)(void);

	/*
	 * get_pages: pin a GPU virtual-address range and obtain the
	 *            page table.  free_cb is invoked by the GPU driver
	 *            if it force-reclaims the pages (e.g. on process
	 *            exit); data is passed back to free_cb.
	 */
	int  (*get_pages)(uint64_t p2p_token, uint32_t va_space,
			  uint64_t vaddr, uint64_t length,
			  struct peer_page_table **pt,
			  void (*free_cb)(void *data), void *data);

	/*
	 * put_pages: release a page table obtained via get_pages.
	 */
	int  (*put_pages)(uint64_t p2p_token, uint32_t va_space,
			  uint64_t vaddr, struct peer_page_table *pt);

	/*
	 * dma_map_pages: create a DMA mapping for a page table on a
	 *                given peer PCI device (NVMe controller).
	 * dma_unmap_pages: destroy a DMA mapping.
	 */
	int  (*dma_map_pages)(struct pci_dev *peer,
			      struct peer_page_table *pt,
			      struct peer_dma_mapping **dm,
				  uint32_t expect_page_size);
	int  (*dma_unmap_pages)(struct pci_dev *peer,
				struct peer_page_table *pt,
				struct peer_dma_mapping *dm);

	/*
	 * free_dma_mapping / free_page_table: free objects after all
	 * references are dropped (used in force-release paths).
	 */
	int  (*free_dma_mapping)(struct peer_dma_mapping *dm);
	int  (*free_page_table)(struct peer_page_table *pt);

	/*
	 * pt_entries: return the number of pinned pages in a page table.
	 * dm_addresses: return the array of DMA bus addresses in a mapping.
	 *
	 * These accessors are backend-specific because the field layout
	 * of peer_page_table / peer_dma_mapping differs per vendor.
	 */
	uint32_t        (*pt_entries)(const struct peer_page_table *pt);
	const uint64_t *(*dm_addresses)(const struct peer_dma_mapping *dm);
};

/* The single ops instance, defined by the selected backend source. */
extern const struct peer_memory_ops peer_memory_ops;

/*
 * Convenience inline wrappers for the accessors (call sites don't need
 * to write peer_memory_ops.pt_entries(pt) — they call peer_memory_pt_entries()).
 */
static inline uint32_t peer_memory_pt_entries(const struct peer_page_table *pt)
{
	return peer_memory_ops.pt_entries(pt);
}

static inline const uint64_t *peer_memory_dm_addresses(const struct peer_dma_mapping *dm)
{
	return peer_memory_ops.dm_addresses(dm);
}

#endif /* SNVME_PEER_MEMORY_H */

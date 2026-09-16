/*
 * compat.h -- Kernel-version compatibility shim for the snvme module.
 *
 * ALL LINUX_VERSION_CODE / KERNEL_VERSION conditional branches in the
 * snvme module live in this unit (compat.c).  The rest of the module
 * calls the stable wrappers declared here and never tests kernel
 * version numbers directly.
 *
 * Feature-probe macros (HAVE_*, selected by the Makefile via grep on
 * Module.symvers, not by version number) are documented here as the
 * canonical compat mechanism but may be consumed by other units:
 *
 *   HAVE_BLK_MARK_DISK_DEAD  -- target kernel exports blk_mark_disk_dead
 *                               (back-ported into some 5.15 point releases).
 *                               Consumed in pci.c; probed in Makefile.in.
 *   HAVE_MODULE_MUTEX        -- module_mutex is directly extern-able.
 *                               Consumed in peer_memory backend sources.
 *
 * SPDX-License-Identifier: GPL-2.0
 */
#ifndef SNVME_COMPAT_H
#define SNVME_COMPAT_H

#include <linux/types.h>

struct page;

/*
 * compat_get_user_pages -- pin a range of user-space pages for DMA.
 *
 * This wrapper absorbs the get_user_pages() signature changes across
 * kernel versions:
 *   <= 4.5.7 : get_user_pages(ts, mm, start, nr, write, force, pages, vmas)
 *   <= 4.8.17: get_user_pages(start, nr, write, force, pages, vmas)
 *   4.9-6.4  : get_user_pages(start, nr, gup_flags, pages, vmas)
 *   >= 6.5   : get_user_pages(start, nr, gup_flags, pages)
 *
 * @start:    starting user virtual address (page-aligned by caller).
 * @nr_pages: number of pages to pin.
 * @write:    non-zero requests write access (maps to FOLL_WRITE on >= 4.9).
 * @pages:    output array of pinned struct page pointers (caller-allocated).
 *
 * Returns the number of pages pinned (>= 0) or a negative errno.
 */
long compat_get_user_pages(unsigned long start, unsigned long nr_pages,
                           int write, struct page **pages);


/* ------------------------------------------------------------------ */
/* Version-compat shims consumed by the shared snvme private units.    */

#include <linux/device.h>
#include <linux/module.h>
#include <linux/version.h>

/*
 * class_create(): the owner argument (THIS_MODULE) was dropped in v6.4.
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 4, 0)
#define snvm_class_create(__name) class_create(THIS_MODULE, __name)
#else
#define snvm_class_create(__name) class_create(__name)
#endif

/*
 * struct class ->devnode callback: `struct device *` became
 * `const struct device *` in v6.1.
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 1, 0)
#define SNVM_DEVNODE_ARGS struct device *dev
#else
#define SNVM_DEVNODE_ARGS const struct device *dev
#endif


/*
 * lba_shift lived in struct nvme_ns until v6.6 and moved to
 * struct nvme_ns_head in v6.7 (upstream commit b3c6c4a788ed).
 * Macro form: struct nvme_ns only needs to be complete where the
 * macro is expanded, so compat.c itself never needs nvme.h.
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 7, 0)
#define snvm_ns_lba_shift(ns) ((ns)->head->lba_shift)
#else
#define snvm_ns_lba_shift(ns) ((ns)->lba_shift)
#endif

#endif /* SNVME_COMPAT_H */


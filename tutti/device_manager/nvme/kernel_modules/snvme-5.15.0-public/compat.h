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
 *   >= 4.9   : get_user_pages(start, nr, gup_flags, pages, vmas)
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

#endif /* SNVME_COMPAT_H */

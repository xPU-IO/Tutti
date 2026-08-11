/*
 * compat.c -- Kernel-version compatibility shim for the snvme module.
 *
 * This is the ONLY translation unit in the snvme module that contains
 * LINUX_VERSION_CODE / KERNEL_VERSION preprocessor conditionals.
 * Everything else calls compat_get_user_pages() and similar stable
 * wrappers.
 *
 * SPDX-License-Identifier: GPL-2.0
 */
#include "compat.h"

#include <linux/version.h>
#include <linux/mm.h>
#include <linux/sched.h>

long compat_get_user_pages(unsigned long start, unsigned long nr_pages,
                           int write, struct page **pages)
{
#if LINUX_VERSION_CODE <= KERNEL_VERSION(4, 5, 7)
#warning "Building for older kernel, not properly tested"
	return get_user_pages(current, current->mm, start, nr_pages,
			      write, 0, pages, NULL);
#elif LINUX_VERSION_CODE <= KERNEL_VERSION(4, 8, 17)
#warning "Building for older kernel, not properly tested"
	return get_user_pages(start, nr_pages, write, 0, pages, NULL);
#elif LINUX_VERSION_CODE < KERNEL_VERSION(6, 5, 0)
	return get_user_pages(start, nr_pages, write ? FOLL_WRITE : 0,
			      pages, NULL);
#else
	return get_user_pages(start, nr_pages, write ? FOLL_WRITE : 0, pages);
#endif
}

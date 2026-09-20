/*
 * libnvm/include/ioctl.h — thin wrapper.
 *
 * All snvme UAPI definitions (structs, ioctl command numbers, ABI
 * version/capability, static_asserts) now live in the shared header:
 *
 *   tutti/include/uapi/tutti_snvme.h
 *
 * This file remains as a compatibility shim so that existing
 *   #include "ioctl.h"
 * statements in both the kernel module (pci.c) and libnvm userspace
 * (device.cpp, ctrl.cpp, file.cpp) continue to work without changes.
 *
 * The kernel module finds this file via -I.../libnvm/include/ (set in
 * Makefile.in @module_ccflags@).  The relative #include below resolves
 * from THIS file's directory to the shared UAPI header, so both kernel
 * and userspace pull in the same physical tutti_snvme.h.
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#ifndef __NVM_INTERNAL_LINUX_IOCTL_H__
#define __NVM_INTERNAL_LINUX_IOCTL_H__

#include "../../../../include/uapi/tutti_snvme.h"

#endif /* __NVM_INTERNAL_LINUX_IOCTL_H__ */

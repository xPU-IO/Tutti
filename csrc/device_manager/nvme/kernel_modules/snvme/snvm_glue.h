/* SPDX-License-Identifier: GPL-2.0 */
/*
 * snvm_glue.h - shared globals/constants for the unified snvme tree.
 */
#ifndef __SNVM_GLUE_H__
#define __SNVM_GLUE_H__

#include <linux/pci.h>
#include "list.h"

#define DRIVER_NAME         "libsnvm helper"
#define PCI_DRIVER_NAME     "snvme"

#define TO_PCI_DEV(addr) \
    pci_get_domain_bus_and_slot((addr).domain, (addr).bus, \
                                PCI_DEVFN((addr).slot, (addr).func))

/* Global registries (snvm_control.c, init'd by snvm_global_init). */
extern struct list ctrl_list;
extern struct list host_list;
extern struct list device_list;
extern struct list device_queue_list;

struct pci_driver;
int  snvm_pci_register(struct pci_driver *drv);
int  snvm_chrdev_create(struct pci_dev *pdev, unsigned int class);
unsigned long clear_ctrl_list(struct list *list);
unsigned long clear_map_list(struct list *list);
int  snvm_global_init(void);
void snvm_global_exit(void);

#endif /* __SNVM_GLUE_H__ */

#ifndef __NVM_INTERNAL_FILE_H__
#define __NVM_INTERNAL_FILE_H__

#include "ioctl.h"

int Host_file_system_int(const char *device, const char *mountPoint);

int Host_file_system_exit(const char *mountPoint);

int get_pcie_addr(const char *dev_path, struct pci_device_addr* pdev_addr);

#endif 
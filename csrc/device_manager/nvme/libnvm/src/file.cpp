#include "ioctl.h"
#include <cerrno>
#include <cstdio>
#include <iostream>  
#include <cstdlib>  
#include <cstring>  
#include <filesystem>
#include <unistd.h>

void executeCommand(const char *command) {  
    std::cerr << "Executing: " << command << std::endl;  
    if (std::system(command) != 0) {  
        std::cerr << "Failed to execute command: " << command << std::endl;  
        exit(EXIT_FAILURE);  
    }  
}  
  
bool checkFileSystem(const char *device) {  
    char command[256];  
    snprintf(command, sizeof(command), "blkid %s | grep -q ext4", device);  
    return std::system(command) == 0;  
}  
  
void createFileSystem(const char *device) {  
    char command[256];  
    snprintf(command, sizeof(command), "mkfs.ext4 %s", device);  
    executeCommand(command);  
}  

void syncFileSystem(const char *device) {  
    char command[256];  
    snprintf(command, sizeof(command), "sync %s", device);  
    executeCommand(command);  
}  

bool mountDevice(const char *device, const char *mountPoint) {  
    std::error_code ec;
    if (!std::filesystem::exists(mountPoint, ec)) {
        if (!std::filesystem::create_directories(mountPoint, ec)) {
            std::cerr << "Failed to create mount point: " << mountPoint << std::endl;
            return false;
        }
    }

    char command[256];  
    snprintf(command, sizeof(command), "mount %s %s", device, mountPoint);  
    if (std::system(command) != 0) {  
        std::cerr << "Failed to mount device. Attempting to recreate filesystem..." << std::endl;  
        createFileSystem(device);  
        // Try mounting again after creating the filesystem  
        if (std::system(command) != 0) {  
            std::cerr << "Failed to mount device even after recreating filesystem." << std::endl;  
            return false;  
        }  
    }  
    return true;  
}  
  
bool umountDevice(const char *device) {  
    char command[256];  
    snprintf(command, sizeof(command), "umount %s", device);  
    return std::system(command) == 0;  
}
bool lumountDevice(const char *device) {  
    char command[256];  
    snprintf(command, sizeof(command), "umount %s", device);  
    return std::system(command) == 0;  
}

int Host_file_system_int(const char *device, const char *mountPoint)
{
    if (!checkFileSystem(device)) {  
        std::cerr << "Filesystem not detected. Creating ext4 filesystem..." << std::endl;  
        createFileSystem(device);  
    } 
    if (!mountDevice(device, mountPoint)) {  
        std::cerr << "Unable to mount the device. Exiting..." << std::endl;  
        return EXIT_FAILURE;  
    }  
    sleep(2);
    std::cerr << "Device mounted successfully at " << mountPoint << std::endl; 
    return 0; 
}

int Host_file_system_exit(const char *mountPoint)
{
    // 程序正常退出或异常捕获后尝试卸载设备
    int ret = -1;
    syncFileSystem(mountPoint);
    for (int i = 0; i < 3; ++i) {  
        if (umountDevice(mountPoint)) {  
            std::cerr << "Device umounted successfully." << std::endl;  
            ret = 0;
            break;  
            
        } else {  
            syncFileSystem(mountPoint);
            std::cerr << "Failed to umount the device. Attempt " << (i + 1) << ". Retrying..." << std::endl;  

            sleep(2); // 等待2秒  
        }  
    } 
    return ret;
}


int get_pcie_addr(const char *dev_path, struct pci_device_addr* pdev_addr){
    const char *prefix = "/dev/";
    size_t prefix_len = strlen(prefix);
    char dev_name[64];
    char command[256];
    FILE *fp;

    if (strncmp(dev_path, prefix, strlen(prefix)) == 0) {
        strncpy(dev_name, dev_path + prefix_len, sizeof(dev_name) - 1);
        
    } else {
        strncpy(dev_name, dev_path, sizeof(dev_name) - 1);
    }

    dev_name[sizeof(dev_name) - 1] = '\0';

    snprintf(command, sizeof(command), "cat /sys/class/block/%s/device/address", dev_name);
    fp = popen(command, "r");

    if (fp == NULL) {
        perror("popen");
        return -EFAULT;
    }
    memset(command, 0, sizeof(command));
    if (fgets(command, sizeof(command), fp) != NULL) {
        sscanf(command, "%x:%x:%x.%x", &pdev_addr->domain, &pdev_addr->bus, &pdev_addr->slot, &pdev_addr->func);
        printf("domain: %x, bus: %x, slot: %x, func: %x\n", pdev_addr->domain, pdev_addr->bus, pdev_addr->slot, pdev_addr->func);
    } else {
        pclose(fp);
        return -EFAULT;
    }
    return 0;
}
  

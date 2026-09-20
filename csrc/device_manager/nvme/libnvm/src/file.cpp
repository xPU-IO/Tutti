// Device mounting helpers for the legacy standalone Controller path.
//
// This file used to shell out to `blkid`/`mkfs.ext4`/`mount`/`umount`/`sync`
// through std::system(), and to format the device whenever a mount failed for
// any reason. Two properties made that unacceptable:
//
//   * A mount can fail for entirely ordinary reasons -- the device is busy,
//     the mount point is busy, permission is missing, the path is wrong, the
//     call raced another mounter. Every one of them was escalated into
//     mkfs.ext4, i.e. into destroying whatever the device held.
//
//   * Device paths were interpolated into a shell command unescaped. In a
//     process that runs as root by design, that is a command-injection
//     surface: a path containing shell metacharacters executes arbitrary
//     commands.
//
// So: no shell, no automatic formatting, no process termination from a
// library. Formatting is a deliberate administrative act and belongs to the
// operator running mkfs, not to a mount helper's error path.
//
// Note on reachability: nothing in-tree constructs the legacy Controller, and
// tutti_daemon uses MountManager instead. That does not make this harmless --
// ctrl.h is installed as a libnvm public header and this file is compiled into
// the shared library, so the API is published to external consumers.

#include "ioctl.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <string>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <unistd.h>

namespace {

// Offset of the ext2/3/4 superblock magic inside the device, and its value.
// The superblock starts at 1024; s_magic sits 0x38 bytes into it.
constexpr off_t kExtSuperblockMagicOffset = 1024 + 0x38;
constexpr unsigned kExtSuperblockMagic = 0xEF53;

// Reads the filesystem magic straight out of the device.
//
// Deliberately not statfs(2): statfs() on a device *node* reports the
// filesystem holding the node (devtmpfs), not the filesystem on the device.
// Reading the superblock is the only way to ask the device itself, and it
// needs no external tool.
bool read_device_magic(const char *device, unsigned *magic_out) {
    const int fd = ::open(device, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    unsigned char raw[2] = {0, 0};
    const ssize_t n = ::pread(fd, raw, sizeof(raw), kExtSuperblockMagicOffset);
    ::close(fd);
    if (n != static_cast<ssize_t>(sizeof(raw))) return false;
    // Little-endian on disk.
    *magic_out = static_cast<unsigned>(raw[0]) |
                 (static_cast<unsigned>(raw[1]) << 8);
    return true;
}

std::string errno_message(const char *what, const char *path) {
    std::string message = what;
    message += " (";
    message += path;
    message += "): ";
    message += std::strerror(errno);
    return message;
}

} // namespace

// Was: `blkid <device> | grep -q ext4`.
bool checkFileSystem(const char *device) {
    if (device == nullptr) return false;
    unsigned magic = 0;
    if (!read_device_magic(device, &magic)) return false;
    return magic == kExtSuperblockMagic;
}

// Was: a `sync <path>` shell call.
//
// A global sync(2) is used on purpose. The only in-tree caller passes a device
// node (see Controller::~Controller), and syncfs(2) on a device node would
// sync the filesystem holding the node rather than the device's contents --
// which would look like it worked while doing nothing useful.
void syncFileSystem(const char * /*path*/) {
    ::sync();
}

// Mounts the device. On failure it reports and returns false.
//
// It must NOT attempt to repair the situation by formatting. Every failure
// mode reachable here (busy, permissions, already mounted, wrong device) is
// either temporary or a misconfiguration, and none of them is evidence that
// the operator wants the device erased.
bool mountDevice(const char *device, const char *mountPoint) {
    if (device == nullptr || mountPoint == nullptr) return false;

    std::error_code ec;
    if (!std::filesystem::exists(mountPoint, ec)) {
        if (!std::filesystem::create_directories(mountPoint, ec)) {
            std::fprintf(stderr, "Failed to create mount point: %s\n",
                         mountPoint);
            return false;
        }
    }

    if (::mount(device, mountPoint, "ext4", 0, nullptr) != 0) {
        std::fprintf(stderr, "%s\n",
                     errno_message("mount failed", mountPoint).c_str());
        std::fprintf(stderr,
                     "Refusing to format the device: formatting is an explicit "
                     "administrative action, not a mount error fallback.\n");
        return false;
    }
    return true;
}

// Was: an `umount <path>` shell call.
bool umountDevice(const char *target) {
    if (target == nullptr) return false;
    if (::umount2(target, 0) == 0) return true;
    // EBUSY and EINVAL are the ordinary "not mounted" answers; the caller's
    // retry loop handles them, so no diagnostics here.
    return false;
}

// Was: `umount <path>` again, with a name that suggested something else.
bool lumountDevice(const char *target) {
    return umountDevice(target);
}

int Host_file_system_int(const char *device, const char *mountPoint)
{
    if (device == nullptr || mountPoint == nullptr) return EXIT_FAILURE;

    if (!checkFileSystem(device)) {
        // Previously formatted the device here. Now it stops: a device without
        // a recognised filesystem is a state the operator must resolve, and
        // guessing by writing a new one is how data gets lost.
        std::fprintf(stderr,
                     "No ext2/3/4 superblock on %s. Not creating a filesystem; "
                     "run mkfs explicitly if that is what you intend.\n",
                     device);
        return EXIT_FAILURE;
    }
    if (!mountDevice(device, mountPoint)) {
        std::fprintf(stderr, "Unable to mount the device. Exiting...\n");
        return EXIT_FAILURE;
    }
    sleep(2);
    std::fprintf(stderr, "Device mounted successfully at %s\n", mountPoint);
    return 0;
}

int Host_file_system_exit(const char *mountPoint)
{
    // Best-effort, retried. Never terminates the process: this runs from a
    // destructor, and a library that calls exit() from a destructor turns an
    // unmount failure into a crash of the whole host process.
    syncFileSystem(mountPoint);
    for (int i = 0; i < 3; ++i) {
        if (umountDevice(mountPoint)) {
            std::fprintf(stderr, "Device umounted successfully.\n");
            return 0;
        }
        syncFileSystem(mountPoint);
        std::fprintf(stderr, "Failed to umount the device. Attempt %d. Retrying...\n",
                     i + 1);
        sleep(2);
    }
    return -1;
}

// Was: popen("cat /sys/class/block/<dev>/device/address").
//
// Read directly. The device name comes from a caller-supplied path, so a shell
// here was an injection point as well as a needless process spawn on a path
// that runs at bring-up.
int get_pcie_addr(const char *dev_path, struct pci_device_addr* pdev_addr){
    if (dev_path == nullptr || pdev_addr == nullptr) return -EFAULT;

    const char *prefix = "/dev/";
    const size_t prefix_len = std::strlen(prefix);
    std::string dev_name = dev_path;
    if (dev_name.compare(0, prefix_len, prefix) == 0) {
        dev_name.erase(0, prefix_len);
    }
    if (dev_name.empty() || dev_name.find('/') != std::string::npos) {
        // A name with a separator could escape the sysfs directory we intend
        // to read, so refuse rather than build a path from it.
        return -EINVAL;
    }

    const std::string sysfs_path =
        "/sys/class/block/" + dev_name + "/device/address";
    const int fd = ::open(sysfs_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        std::perror("open pci address");
        return -EFAULT;
    }
    char buf[64] = {0};
    const ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
    ::close(fd);
    if (n <= 0) return -EFAULT;

    if (std::sscanf(buf, "%x:%x:%x.%x", &pdev_addr->domain, &pdev_addr->bus,
                    &pdev_addr->slot, &pdev_addr->func) != 4) {
        return -EFAULT;
    }
    std::printf("domain: %x, bus: %x, slot: %x, func: %x\n",
                pdev_addr->domain, pdev_addr->bus, pdev_addr->slot,
                pdev_addr->func);
    return 0;
}

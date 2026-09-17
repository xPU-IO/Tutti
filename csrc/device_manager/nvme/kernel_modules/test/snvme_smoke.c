/*
 * snvme_smoke.c -- Standalone end-to-end sanity test for the SNVMe kernel module.
 *
 * This program is intentionally self-contained:
 *   - no CUDA, no libnvm, no Tutti filesystem dependencies,
 *   - it links against nothing but libc,
 *   - it only includes the SNVMe UAPI header
 *     (backends/local/nvme/libnvm/include/ioctl.h).
 *
 * Two test modes are supported:
 *
 *   default ("UAPI smoke")   exercise every kernel↔user entry point that
 *                            does NOT trigger an NVMe controller probe.
 *                            Safe to run on any host with snvme loaded:
 *                            it does not touch the NVMe data path.
 *
 *   --bind   ("full bring-up") additionally runs SNVM_DEVICE_BIND, waits
 *                            for /dev/snvme<X>n<Y> (note: leading 's' --
 *                            SNVMe block devices are namespaced away from
 *                            the in-tree nvme driver), does a 512 B pread()
 *                            and tears the controller down. DESTRUCTIVE --
 *                            the in-tree nvme driver loses the device for
 *                            the duration of the test.
 *
 * UAPI-smoke steps (always run):
 *   [ 1] open /dev/snvm_control                                      (UAPI: chrdev factory exists)
 *   [ 2] SNVM_CHRDEV_CREATE(BDF)                                     (UAPI: chrdev create + minor returned)
 *   [ 3] open /dev/ssnvme<minor>                                     (UAPI: per-controller chrdev usable)
 *   [ 4] mmap(BAR0)                                                  (UAPI: BAR0 mapped, register reachable)
 *   [ 5] read NVMe CAP from BAR0                                     (sanity: register decoder agrees with spec)
 *   [ 6] NVM_SET_KERNEL_IOQ_CAP                                     (pre-bind budget hint, matches nvm_controller_init_owner)
 *   [F2] munmap(BAR0) + close(/dev/ssnvme<N>)                        (chrdev release path)
 *   [F3] SNVM_CHRDEV_REMOVE(BDF)                                     (release minor)
 *
 * NOTE: the legacy NVM_SET_IOQ_NUM / NVM_MAP_HOST_MEMORY(ring) /
 *       NVM_SET_SHARE_REG / NVM_CLEAR_IOQ_NUM pre-registration flow was
 *       retired from the module; the B3 queue-group path (see
 *       snvme_smoke_addq / snvme_smoke_io) is the current user-queue API.
 *
 * Additional steps when --bind is given (run BEFORE the cleanup tail):
 *   [B1] SNVM_DEVICE_BIND(BDF)                                       (kernel probe runs, /dev/snvme<X>n<Y> appears)
 *   [B2] NVM_GET_DEV_INFO                                            (returns disk_name, block_size...)
 *   [B3] open /dev/<disk_name>, pread 512 bytes                      (block device works)
 *   [B4] SNVM_DEVICE_UNBIND(BDF)
 *
 * Each step prints "[ OK ] step=..." on success or "[FAIL] step=... errno=N (...)"
 * and exits non-zero immediately on the first failure.
 *
 * Build:        make           (in this directory)
 * Invoke:       sudo ./snvme_smoke <PCI_BDF>
 *               sudo ./snvme_smoke --bind <PCI_BDF>
 *
 * Pre-conditions:
 *   - snvme-core.ko + snvme.ko loaded (insmod or via Makefile).
 *   - /dev/snvm_control exists (mode 0666).
 *   - The PCI device at <PCI_BDF> is a real NVMe SSD whose data you can lose
 *     when running with --bind. UAPI-smoke is safe even if the device is
 *     currently bound to the in-tree nvme driver.
 *
 * Exit codes:
 *   0  -- all steps passed; SNVMe is healthy on this kernel.
 *   1  -- usage error.
 *   2  -- a smoke step failed; see stderr for which one.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* The *only* SNVMe-specific header we need: the kernel/user UAPI. */
#include "ioctl.h"

/* ------------------------------------------------------------------ */
/* Logging helpers                                                    */
/* ------------------------------------------------------------------ */

static int g_step = 0;

static void step_ok(const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[ OK ] step=%-2d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void step_warn(const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[WARN] step=%-2d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void __attribute__((noreturn)) step_fail(int err, const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[FAIL] step=%-2d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, " errno=%d (%s)\n", err, err ? strerror(err) : "n/a");
    exit(2);
}

/* ------------------------------------------------------------------ */
/* Minimal NVMe register access: we only read CAP for sanity.          */
/* ------------------------------------------------------------------ */

#define NVME_REG_CAP    0x0000   /* 64-bit Controller Capabilities */

static uint64_t mmio_read64(volatile void* base, size_t off) {
    /*
     * x86 supports unaligned 64-bit MMIO loads; if you port this to
     * an arch that doesn't, switch to two readl()-equivalents.
     */
    volatile uint64_t* p = (volatile uint64_t*)((volatile char*)base + off);
    return *p;
}

/* ------------------------------------------------------------------ */
/* BDF parser                                                         */
/* ------------------------------------------------------------------ */

static int parse_bdf(const char* s, struct pci_device_addr* out) {
    /* Accept the canonical "DDDD:BB:DD.F" form, e.g. "0000:50:00.0". */
    return sscanf(s, "%x:%x:%x.%x",
                  &out->domain, &out->bus, &out->slot, &out->func) == 4 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Convenience: ioctl wrapper that turns -1 into errno-with-context.   */
/* ------------------------------------------------------------------ */

static int do_ioctl(int fd, unsigned long req, void* arg, const char* what) {
    int r = ioctl(fd, req, arg);
    if (r < 0) {
        int e = errno;
        fprintf(stderr, "ioctl(%s) failed: %s\n", what, strerror(e));
        errno = e;
    }
    return r;
}

/* ------------------------------------------------------------------ */
/* Argument parsing                                                   */
/* ------------------------------------------------------------------ */

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [--bind] <PCI_BDF>\n"
        "  e.g.: %s 0000:50:00.0           # UAPI-smoke only (safe)\n"
        "        %s --bind 0000:50:00.0    # full bring-up (destructive)\n",
        prog, prog, prog);
}

int main(int argc, char** argv) {
    int do_bind = 0;
    const char* bdf_str = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bind") == 0) {
            do_bind = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (bdf_str == NULL) {
            bdf_str = argv[i];
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (bdf_str == NULL) {
        usage(argv[0]);
        return 1;
    }

    struct pci_device_addr orig_bdf;
    if (parse_bdf(bdf_str, &orig_bdf) != 0) {
        fprintf(stderr, "Bad BDF: '%s' (expected DDDD:BB:DD.F)\n", bdf_str);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* [1] /dev/snvm_control                                              */
    /* ------------------------------------------------------------------ */
    int fd_ctl = open("/dev/snvm_control", O_RDWR | O_NONBLOCK);
    if (fd_ctl < 0)
        step_fail(errno, "open(/dev/snvm_control)");
    step_ok("open(/dev/snvm_control) fd=%d", fd_ctl);

    /* ------------------------------------------------------------------ */
    /* [2] SNVM_CHRDEV_CREATE                                             */
    /*     The kernel writes the allocated minor back into addr.domain.   */
    /* ------------------------------------------------------------------ */
    struct pci_device_addr addr = orig_bdf;
    if (do_ioctl(fd_ctl, SNVM_CHRDEV_CREATE, &addr, "SNVM_CHRDEV_CREATE") < 0)
        step_fail(errno, "SNVM_CHRDEV_CREATE %s", bdf_str);
    int minor_n = addr.domain;
    step_ok("SNVM_CHRDEV_CREATE minor=%d", minor_n);

    /* ------------------------------------------------------------------ */
    /* [3] /dev/ssnvme<N>                                                 */
    /* ------------------------------------------------------------------ */
    char dev_path[64];
    snprintf(dev_path, sizeof(dev_path), "/dev/ssnvme%d", minor_n);
    int fd_dev = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_dev < 0)
        step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd=%d", dev_path, fd_dev);

    /* ------------------------------------------------------------------ */
    /* [4] mmap BAR0                                                      */
    /*     8 KiB matches NVM_CTRL_MEM_MINSIZE used by libnvm; for any      */
    /*     stock NVMe controller BAR0 is at least 4 KiB (CAP + admin).     */
    /*                                                                    */
    /*     We intentionally do NOT pass MAP_LOCKED -- BAR0 is device      */
    /*     memory (vm_iomap_memory), not pageable, so MAP_LOCKED is       */
    /*     meaningless for it and only serves to fail under a tight       */
    /*     RLIMIT_MEMLOCK.                                                */
    /* ------------------------------------------------------------------ */
    const size_t bar0_size = 8192;
    void* bar0 = mmap(NULL, bar0_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_dev, 0);
    if (bar0 == MAP_FAILED)
        step_fail(errno, "mmap(BAR0, %zu)", bar0_size);
    step_ok("mmap(BAR0, %zu) -> %p", bar0_size, bar0);

    /* ------------------------------------------------------------------ */
    /* [5] Read CAP and decode dstrd / mqes for sanity.                    */
    /*                                                                    */
    /*     NOTE: if CAP reads as all-ones, the controller is most likely  */
    /*     in D3 / ASPM / disabled. That is NOT a SNVMe bug -- it just    */
    /*     means the device was never powered up by any driver. Warn and  */
    /*     continue; the rest of the UAPI-smoke path is still meaningful. */
    /*     all-zeros, on the other hand, means BAR0 is mapped to nothing  */
    /*     (pci_resource_start returns 0), which IS a real problem.        */
    /* ------------------------------------------------------------------ */
    uint64_t cap = mmio_read64(bar0, NVME_REG_CAP);
    uint32_t mqes = (uint32_t)(cap & 0xffff) + 1;     /* CAP.MQES is 0-based */
    uint32_t dstrd = (uint32_t)((cap >> 32) & 0xf);
    if (cap == 0)
        step_fail(EIO, "BAR0 CAP reads as all-zeros "
                       "(BAR not mapped or pci_resource_start==0)");
    if (cap == (uint64_t)-1)
        step_warn("BAR0 CAP=0xFFF..FF -- controller is probably powered down; "
                  "continuing UAPI-smoke. Bind to nvme/snvme first to test the full path.");
    else
        step_ok("BAR0 CAP=0x%016" PRIx64 " (mqes=%u, dstrd=%u)", cap, mqes, dstrd);

    /* ------------------------------------------------------------------ */
    /* [6] NVM_SET_KERNEL_IOQ_CAP -- the B3 pre-bind budget hint.          */
    /*                                                                    */
    /* This mirrors the current application startup path                 */
    /* (nvm_controller_init_owner in libnvm/src/linux/device.cpp, driven by */
    /* NVMeService's sys_config.yaml `kernel_ioq_cap`): declare an upper */
    /* bound on kernel-side IO queues so the remainder of the            */
    /* controller's grant is left for the NVM_ADD_USER_QUEUE user pool.  */
    /*                                                                    */
    /* The legacy NVM_SET_IOQ_NUM / NVM_MAP_HOST_MEMORY(ring) /          */
    /* NVM_SET_SHARE_REG pre-registration flow this test used to run was */
    /* retired from the module in favour of the B3 queue-group path      */
    /* (see snvme_smoke_addq / snvme_smoke_io for the full user-queue    */
    /* lifecycle); this UAPI smoke now stops at controller bring-up.     */
    /*                                                                    */
    /* Cap overridable via SNVME_TEST_KERNEL_IOQ_CAP for controllers     */
    /* whose total IOQ grant is smaller than the 32 default.            */
    /* ------------------------------------------------------------------ */
    {
        uint32_t cap = 32;
        const char *cap_env = getenv("SNVME_TEST_KERNEL_IOQ_CAP");
        if (cap_env != NULL && cap_env[0] != '\0') {
            char *end = NULL;
            unsigned long v = strtoul(cap_env, &end, 0);
            if (end == cap_env || *end != '\0' || v == 0 || v > UINT32_MAX)
                step_fail(0, "SNVME_TEST_KERNEL_IOQ_CAP=\"%s\" is not a positive "
                             "integer", cap_env);
            cap = (uint32_t)v;
        }
        if (do_ioctl(fd_dev, NVM_SET_KERNEL_IOQ_CAP, &cap, "NVM_SET_KERNEL_IOQ_CAP") < 0)
            step_fail(errno, "NVM_SET_KERNEL_IOQ_CAP cap=%u", cap);
        step_ok("NVM_SET_KERNEL_IOQ_CAP cap=%u%s", cap,
                cap_env ? " [from SNVME_TEST_KERNEL_IOQ_CAP]" : "");
    }

    /* ================================================================== */
    /*  --bind path: actually run s_nvme_probe and exercise the resulting */
    /*  /dev/snvme<X>n<Y> block device (note the leading 's' -- SNVMe     */
    /*  namespaces its disks away from the in-tree nvme driver; see       */
    /*  PORTING.md §2).                                                    */
    /*                                                                    */
    /*  This is exactly the bring-up nvm_controller_init_owner() performs: */
    /*  the NVM_SET_KERNEL_IOQ_CAP hint above, then BIND, then poll        */
    /*  NVM_GET_DEV_INFO until the async probe/scan completes.  User IO    */
    /*  queues are created afterwards via NVM_ADD_USER_QUEUE (see          */
    /*  snvme_smoke_addq / snvme_smoke_io); this smoke stops at bring-up.  */
    /* ================================================================== */
    char disk_name_buf[DISK_NAME_LEN + 1] = {0};

    if (do_bind) {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_BIND, &bdf, "SNVM_DEVICE_BIND") < 0)
            step_fail(errno,
                "SNVM_DEVICE_BIND %s -- the in-tree nvme driver may still own this device "
                "(see PORTING.md §5)",
                bdf_str);
        /*
         * s_nvme_probe() schedules an async worker for reset + scan.
         * Poll for NVM_GET_DEV_INFO to succeed instead of assuming a
         * hard-coded sleep is enough -- slow / power-managed drives
         * can easily take more than 3 s to come up.
         */
        struct nvm_ioctl_dev info;
        int ok = 0;
        for (int i = 0; i < 100; i++) {   /* up to ~10 s */
            memset(&info, 0, sizeof(info));
            if (ioctl(fd_dev, NVM_GET_DEV_INFO, &info) == 0 &&
                info.disk_name[0] != '\0') {
                ok = 1;
                break;
            }
            usleep(100 * 1000);
        }
        if (!ok)
            step_fail(errno, "NVM_GET_DEV_INFO did not succeed within 10s after bind "
                             "(probe may still be running; try --bind on a faster disk)");
        step_ok("SNVM_DEVICE_BIND %s (probe done)", bdf_str);

        /* `info` was already filled by the poll loop above. */
        memcpy(disk_name_buf, info.disk_name, DISK_NAME_LEN);
        step_ok("NVM_GET_DEV_INFO disk='%s' nr_user_q=%u block_size=%zu max_data_size=%zu",
                disk_name_buf, info.nr_user_q, info.block_size, info.max_data_size);

        char blk_path[DISK_NAME_LEN + 8];
        snprintf(blk_path, sizeof(blk_path), "/dev/%s", disk_name_buf);
        int fd_blk = open(blk_path, O_RDONLY);
        if (fd_blk < 0)
            step_fail(errno, "open(%s) -- block device did not appear", blk_path);
        char buf[512];
        ssize_t got = pread(fd_blk, buf, sizeof(buf), 0);
        if (got != (ssize_t)sizeof(buf))
            step_fail(errno, "pread(%s, 512) returned %zd", blk_path, got);
        close(fd_blk);
        step_ok("pread(%s, 512) ok", blk_path);

        bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_UNBIND, &bdf, "SNVM_DEVICE_UNBIND") < 0)
            step_fail(errno, "SNVM_DEVICE_UNBIND %s", bdf_str);
        step_ok("SNVM_DEVICE_UNBIND %s", bdf_str);
    }

    /* ------------------------------------------------------------------ */
    /* [F2] Release per-controller resources                               */
    /* ------------------------------------------------------------------ */
    if (munmap(bar0, bar0_size) < 0)
        step_fail(errno, "munmap(BAR0)");
    if (close(fd_dev) < 0)
        step_fail(errno, "close(%s)", dev_path);
    step_ok("munmap(BAR0) + close(%s)", dev_path);

    /* ------------------------------------------------------------------ */
    /* [F3] SNVM_CHRDEV_REMOVE -- release the per-controller minor.        */
    /* ------------------------------------------------------------------ */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_CHRDEV_REMOVE, &bdf,
                     "SNVM_CHRDEV_REMOVE") < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE %s", bdf_str);
    }
    step_ok("SNVM_CHRDEV_REMOVE %s", bdf_str);

    close(fd_ctl);
    fprintf(stderr, "\nAll %d steps passed. SNVMe is healthy%s.\n",
            g_step, do_bind ? " (full bring-up)" : " (UAPI-smoke)");
    return 0;
}

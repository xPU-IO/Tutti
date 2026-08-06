/*
 * snvme_ubind.c -- minimal owner-side reset helper.
 *
 * Issues SNVM_DEVICE_UNBIND then SNVM_CHRDEV_REMOVE for one BDF via
 * /dev/snvm_control, restoring a controller left in a half-bound state
 * (e.g. after a smoke test failed past SNVM_DEVICE_BIND but before its
 * teardown tail) back to a clean, unbound, chrdev-free state.
 *
 * The UNBIND-then-CHRDEV_REMOVE order matches the canonical teardown
 * tail of the smoke tests (see snvme_smoke.c / snvme_smoke_libnvm.c).
 * Both steps are best-effort: a missing bind or chrdev is warned about
 * (the matching state may already be clean) rather than treated as a
 * hard failure, so the helper is safe to run repeatedly / on a device
 * in any partial state.
 *
 * Build:  make snvme_ubind
 *     or  cc -O2 -I../../libnvm/include -o snvme_ubind snvme_ubind.c
 * Invoke: sudo ./snvme_ubind 0000:b1:00.0
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include "ioctl.h"

static int parse_bdf(const char* s, struct pci_device_addr* o) {
    return sscanf(s, "%x:%x:%x.%x",
                  &o->domain, &o->bus, &o->slot, &o->func) == 4 ? 0 : -1;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s DDDD:BB:DD.F\n", argv[0]);
        return 1;
    }
    struct pci_device_addr bdf;
    if (parse_bdf(argv[1], &bdf) != 0) {
        fprintf(stderr, "bad BDF '%s'\n", argv[1]);
        return 1;
    }
    int fd = open("/dev/snvm_control", O_RDWR);
    if (fd < 0) { perror("open /dev/snvm_control"); return 2; }

    struct pci_device_addr a = bdf;
    if (ioctl(fd, SNVM_DEVICE_UNBIND, &a) != 0)
        fprintf(stderr, "[warn] SNVM_DEVICE_UNBIND: %s (errno=%d) "
                "-- maybe already unbound\n", strerror(errno), errno);
    else
        printf("[ OK ] SNVM_DEVICE_UNBIND %s\n", argv[1]);

    a = bdf;
    if (ioctl(fd, SNVM_CHRDEV_REMOVE, &a) != 0)
        fprintf(stderr, "[warn] SNVM_CHRDEV_REMOVE: %s (errno=%d) "
                "-- maybe no chrdev\n", strerror(errno), errno);
    else
        printf("[ OK ] SNVM_CHRDEV_REMOVE %s\n", argv[1]);

    close(fd);
    return 0;
}

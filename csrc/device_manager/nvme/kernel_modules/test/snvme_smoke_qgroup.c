/*
 * snvme_smoke_qgroup.c -- Smoke test for NVM_CREATE_QUEUE_GROUP /
 * NVM_DESTROY_QUEUE_GROUP (queue-group plan, step B1).
 *
 * Purpose
 * -------
 *
 * B1 introduces the per-fd queue group container in snvme, but
 * still has *zero* NVMe-side resources attached to it -- no maps,
 * no user IO queues, no admin commands.  The point of this test
 * is therefore to exercise just the kernel container lifecycle:
 *
 *   - allocate a group and get an opaque group_id back,
 *   - destroy it explicitly,
 *   - allocate again and let the fd close cascade-clean it,
 *   - confirm cap enforcement (NVM_MAX_GROUPS_PER_FD),
 *   - confirm cross-fd isolation (group_id from fd A is not
 *     destroyable on fd B),
 *   - confirm error paths (group_id=0 is rejected; invalid id is
 *     -ENOENT).
 *
 * Crucially: this test does NOT call SNVM_DEVICE_BIND.  Group
 * lifecycle is bind-agnostic by design -- userspace can prepare
 * the group container before the controller is bound, and the
 * later NVM_ADD_USER_QUEUE call (B3) is the one that requires a
 * live admin_q.  Skipping bind here also keeps the test
 * non-destructive: it does not touch the in-tree nvme driver's
 * ownership of the BDF, so it is safe to run on a host where the
 * target NVMe namespace is mounted.
 *
 * After running, you can grep dmesg for:
 *
 *   "snvme: NVM_CREATE_QUEUE_GROUP id=N max_queues=16 pid=..."
 *   "snvme: NVM_DESTROY_QUEUE_GROUP id=N pid=..."
 *   "snvme: snvm_dev_release: cascade-destroyed K orphan group(s) ..."
 *
 * to confirm the kernel-side path executed.  These are pr_debug /
 * pr_info; you may need `dmesg -n 8` or similar.
 *
 * Build:    make snvme_smoke_qgroup        (parent Makefile)
 * Invoke:   sudo ./snvme_smoke_qgroup <PCI_BDF>
 *
 * Exit codes:
 *   0  -- all steps passed.
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
#include <unistd.h>

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
/* BDF parser                                                         */
/* ------------------------------------------------------------------ */

static int parse_bdf(const char* s, struct pci_device_addr* out) {
    return sscanf(s, "%x:%x:%x.%x",
                  &out->domain, &out->bus, &out->slot, &out->func) == 4 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Tiny ioctl wrapper.  We deliberately do NOT step_fail() inside     */
/* the wrapper -- some tests below use ioctl() directly because they */
/* expect a specific errno.                                           */
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
        "Usage: %s <PCI_BDF>\n"
        "  e.g.: %s 0000:50:00.0\n"
        "\n"
        "This test exercises the per-fd queue-group container only.\n"
        "It does NOT bind the controller, so it is safe to run on a\n"
        "host where the target NVMe device is mounted.\n",
        prog, prog);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(argv[0]);
        return 0;
    }
    const char* bdf_str = argv[1];

    struct pci_device_addr orig_bdf;
    if (parse_bdf(bdf_str, &orig_bdf) != 0) {
        fprintf(stderr, "Bad BDF: '%s' (expected DDDD:BB:DD.F)\n", bdf_str);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* [1] open /dev/snvm_control and create /dev/ssnvme<N> for our BDF.  */
    /*                                                                    */
    /* CHRDEV_CREATE returns the freshly allocated minor in addr.domain   */
    /* (mirroring SNVM_CHRDEV_REMOVE on the way out).                     */
    /* ------------------------------------------------------------------ */
    int fd_ctl = open("/dev/snvm_control", O_RDWR | O_NONBLOCK);
    if (fd_ctl < 0)
        step_fail(errno, "open(/dev/snvm_control)");
    step_ok("open(/dev/snvm_control) fd=%d", fd_ctl);

    struct pci_device_addr addr = orig_bdf;
    if (do_ioctl(fd_ctl, SNVM_CHRDEV_CREATE, &addr, "SNVM_CHRDEV_CREATE") < 0)
        step_fail(errno, "SNVM_CHRDEV_CREATE %s", bdf_str);
    int minor_n = addr.domain;
    step_ok("SNVM_CHRDEV_CREATE minor=%d", minor_n);

    char dev_path[64];
    snprintf(dev_path, sizeof(dev_path), "/dev/ssnvme%d", minor_n);

    /* ------------------------------------------------------------------ */
    /* [2] Open ssnvme fd #A.  This is the primary fd; we'll do most of */
    /* the lifecycle work on it.                                          */
    /* ------------------------------------------------------------------ */
    int fd_a = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_a < 0)
        step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd_a=%d", dev_path, fd_a);

    /* ------------------------------------------------------------------ */
    /* [3] NVM_CREATE_QUEUE_GROUP -- happy path.                          */
    /*                                                                    */
    /* Verifies:                                                          */
    /*   - kernel returns group_id != 0 (0 is the "no group" sentinel)   */
    /*   - kernel echoes max_queues = NVM_MAX_QUEUES_PER_GROUP            */
    /*   - flags / reserved untouched                                    */
    /* ------------------------------------------------------------------ */
    uint32_t group_a;
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_a, NVM_CREATE_QUEUE_GROUP, &req, "NVM_CREATE_QUEUE_GROUP") < 0)
            step_fail(errno, "NVM_CREATE_QUEUE_GROUP on fd_a");
        if (req.group_id == 0)
            step_fail(0, "NVM_CREATE_QUEUE_GROUP returned group_id=0 (sentinel)");
        if (req.max_queues != NVM_MAX_QUEUES_PER_GROUP)
            step_fail(0, "NVM_CREATE_QUEUE_GROUP max_queues=%u, expected %u",
                      req.max_queues, NVM_MAX_QUEUES_PER_GROUP);
        group_a = req.group_id;
        step_ok("NVM_CREATE_QUEUE_GROUP fd_a -> group_id=%u max_queues=%u",
                req.group_id, req.max_queues);
    }

    /* ------------------------------------------------------------------ */
    /* [4] NVM_CREATE_QUEUE_GROUP -- second call on same fd, expect      */
    /* -EBUSY (cap = NVM_MAX_GROUPS_PER_FD = 1 in B1).                   */
    /*                                                                    */
    /* This guards against a regression where the cap check is dropped   */
    /* or off-by-one'd.  If userspace ever needs >1 group/fd, the cap    */
    /* must be raised in the kernel AND in this assertion together.      */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        int r = ioctl(fd_a, NVM_CREATE_QUEUE_GROUP, &req);
        if (r == 0)
            step_fail(0,
                "NVM_CREATE_QUEUE_GROUP unexpectedly succeeded on second call "
                "(got group_id=%u); per-fd cap (%u) is not enforced",
                req.group_id, NVM_MAX_GROUPS_PER_FD);
        if (errno != EBUSY)
            step_fail(errno,
                "NVM_CREATE_QUEUE_GROUP second call returned errno=%d "
                "(expected EBUSY=%d)",
                errno, EBUSY);
        step_ok("per-fd group cap correctly returns -EBUSY on second create");
    }

    /* ------------------------------------------------------------------ */
    /* [5] NVM_DESTROY_QUEUE_GROUP -- happy path on fd_a's own group.    */
    /* ------------------------------------------------------------------ */
    {
        uint32_t gid = group_a;
        if (do_ioctl(fd_a, NVM_DESTROY_QUEUE_GROUP, &gid, "NVM_DESTROY_QUEUE_GROUP") < 0)
            step_fail(errno, "NVM_DESTROY_QUEUE_GROUP id=%u on fd_a", group_a);
        step_ok("NVM_DESTROY_QUEUE_GROUP id=%u on fd_a", group_a);
    }

    /* ------------------------------------------------------------------ */
    /* [6] NVM_DESTROY_QUEUE_GROUP -- destroying the same id twice must */
    /* return -ENOENT (the descriptor is gone after step 5).             */
    /* ------------------------------------------------------------------ */
    {
        uint32_t gid = group_a;
        int r = ioctl(fd_a, NVM_DESTROY_QUEUE_GROUP, &gid);
        if (r == 0)
            step_fail(0,
                "NVM_DESTROY_QUEUE_GROUP id=%u unexpectedly succeeded twice "
                "(double-free of group descriptor in kernel?)", group_a);
        if (errno != ENOENT)
            step_fail(errno, "second destroy returned errno=%d, expected ENOENT(%d)",
                      errno, ENOENT);
        step_ok("double NVM_DESTROY_QUEUE_GROUP correctly returns -ENOENT");
    }

    /* ------------------------------------------------------------------ */
    /* [7] NVM_DESTROY_QUEUE_GROUP -- group_id=0 is the sentinel and     */
    /* must be rejected with -EINVAL before any list lookup.             */
    /* ------------------------------------------------------------------ */
    {
        uint32_t gid = 0;
        int r = ioctl(fd_a, NVM_DESTROY_QUEUE_GROUP, &gid);
        if (r == 0)
            step_fail(0, "NVM_DESTROY_QUEUE_GROUP id=0 unexpectedly succeeded");
        if (errno != EINVAL)
            step_fail(errno, "NVM_DESTROY_QUEUE_GROUP id=0 returned errno=%d, "
                             "expected EINVAL(%d)", errno, EINVAL);
        step_ok("NVM_DESTROY_QUEUE_GROUP id=0 correctly returns -EINVAL");
    }

    /* ------------------------------------------------------------------ */
    /* [8] Cross-fd isolation: open a second fd, create a group on it,  */
    /* then try to destroy that group from fd_a -- must -ENOENT.         */
    /*                                                                    */
    /* This validates the design invariant that group descriptors are    */
    /* per-fd reachable only.  An adversarial process should not be     */
    /* able to scan group_ids and tear down a sibling's groups.         */
    /* ------------------------------------------------------------------ */
    int fd_b = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_b < 0)
        step_fail(errno, "open(%s) for fd_b", dev_path);
    step_ok("open(%s) fd_b=%d", dev_path, fd_b);

    uint32_t group_b;
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_b, NVM_CREATE_QUEUE_GROUP, &req, "NVM_CREATE_QUEUE_GROUP fd_b") < 0)
            step_fail(errno, "NVM_CREATE_QUEUE_GROUP on fd_b");
        if (req.group_id == 0)
            step_fail(0, "NVM_CREATE_QUEUE_GROUP returned group_id=0 on fd_b");
        /*
         * NB: req.group_id == group_a (i.e. the IDA recycled the
         * id we just freed in step 5) is *expected* and not a
         * bug.  Linux's IDA always allocates the lowest free id,
         * so after ida_simple_remove(1) the very next get() will
         * return 1 again.  Userspace MUST treat group_id as an
         * opaque cookie and never rely on it being unique across
         * the program's lifetime, only for the lifetime of the
         * fd that owns it.  Cross-fd isolation (step below) is
         * what actually guarantees safety.
         */
        group_b = req.group_id;
        step_ok("NVM_CREATE_QUEUE_GROUP fd_b -> group_id=%u%s",
                group_b,
                group_b == group_a ? " (IDA recycled freed id; expected)" : "");
    }
    {
        uint32_t gid = group_b;
        int r = ioctl(fd_a, NVM_DESTROY_QUEUE_GROUP, &gid);
        if (r == 0)
            step_fail(0,
                "fd_a managed to destroy fd_b's group %u -- cross-fd "
                "isolation is broken in the kernel",
                group_b);
        if (errno != ENOENT)
            step_fail(errno,
                "fd_a destroy of fd_b's group returned errno=%d, "
                "expected ENOENT(%d)", errno, ENOENT);
        step_ok("cross-fd destroy correctly blocked: fd_a -> fd_b's group %u "
                "returns -ENOENT", group_b);
    }

    /* ------------------------------------------------------------------ */
    /* [9] Cascade cleanup on fd close: leave fd_b's group alive and    */
    /* close fd_b.  The kernel must reap group_b automatically.         */
    /*                                                                    */
    /* We can't directly observe fd_b's groups list from userspace --   */
    /* the proof point is in dmesg ("snvme: snvm_dev_release:           */
    /* cascade-destroyed N orphan group(s)...").  To make this testable */
    /* in CI even without dmesg access, we follow up by re-opening      */
    /* /dev/ssnvme<N> and creating yet another group; if the IDA leaked */
    /* the id, we'd see ever-growing group_ids across runs (best-effort */
    /* signal, not deterministic).                                      */
    /* ------------------------------------------------------------------ */
    if (close(fd_b) < 0)
        step_fail(errno, "close(fd_b)");
    step_ok("close(fd_b) -- kernel must cascade-destroy group_id=%u "
            "(grep dmesg for 'cascade-destroyed 1 orphan group(s)')",
            group_b);

    int fd_c = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_c < 0)
        step_fail(errno, "open(%s) for fd_c", dev_path);
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_c, NVM_CREATE_QUEUE_GROUP, &req, "NVM_CREATE_QUEUE_GROUP fd_c") < 0)
            step_fail(errno, "NVM_CREATE_QUEUE_GROUP on fd_c after cascade");
        step_ok("post-cascade NVM_CREATE_QUEUE_GROUP fd_c -> group_id=%u "
                "(allocator still healthy)",
                req.group_id);
        /* Leave this group attached -- close(fd_c) should clean it. */
    }
    if (close(fd_c) < 0)
        step_fail(errno, "close(fd_c)");
    step_ok("close(fd_c) -- second cascade for the post-recovery group");

    /* ------------------------------------------------------------------ */
    /* [10] flags MBZ rejection: non-zero flags must be -EINVAL.         */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        req.flags = 0xdeadbeef;
        int r = ioctl(fd_a, NVM_CREATE_QUEUE_GROUP, &req);
        if (r == 0)
            step_fail(0,
                "NVM_CREATE_QUEUE_GROUP with flags=0x%x unexpectedly succeeded "
                "(group_id=%u); MBZ check missing",
                0xdeadbeef, req.group_id);
        if (errno != EINVAL)
            step_fail(errno,
                "NVM_CREATE_QUEUE_GROUP with flags!=0 returned errno=%d, "
                "expected EINVAL(%d)", errno, EINVAL);
        step_ok("flags MBZ check rejects non-zero with -EINVAL");
    }

    /* ================================================================== */
    /* B2: NVM_MAP_HOST_MEMORY with group_id                              */
    /*                                                                    */
    /* The next block creates a fresh group on fd_a, registers a host    */
    /* page against it via NVM_MAP_HOST_MEMORY (group_id != 0, new       */
    /* mode), then validates:                                             */
    /*   - the map IO addresses are returned correctly,                  */
    /*   - registering against a foreign group_id returns -ENOENT,       */
    /*   - registering with reserved!=0 is rejected with -EINVAL,        */
    /*   - destroying the group cascades through the map and releases   */
    /*     the pinned page (dmesg "destroy_qgroup id=N drained 1 map"). */
    /* ================================================================== */

    long psz = sysconf(_SC_PAGESIZE);
    if (psz <= 0)
        step_fail(errno, "sysconf(_SC_PAGESIZE)");

    /* ------------------------------------------------------------------ */
    /* [11] Create a new group on fd_a for the B2 map experiments.       */
    /* ------------------------------------------------------------------ */
    uint32_t group_d;
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_a, NVM_CREATE_QUEUE_GROUP, &req,
                     "NVM_CREATE_QUEUE_GROUP fd_a (B2 setup)") < 0)
            step_fail(errno, "NVM_CREATE_QUEUE_GROUP fd_a (B2 setup)");
        group_d = req.group_id;
        step_ok("B2 setup: NVM_CREATE_QUEUE_GROUP fd_a -> group_id=%u",
                group_d);
    }

    /* ------------------------------------------------------------------ */
    /* [12] mmap one page of host memory for the test buffer.            */
    /*                                                                    */
    /* No MAP_LOCKED: NVM_MAP_HOST_MEMORY pins the page kernel-side via  */
    /* get_user_pages_fast(), so a userspace mlock is redundant (and    */
    /* fails under low RLIMIT_MEMLOCK in containers anyway).             */
    /* ------------------------------------------------------------------ */
    void* host_buf = mmap(NULL, (size_t)psz, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (host_buf == MAP_FAILED)
        step_fail(errno, "mmap(host page)");
    memset(host_buf, 0, (size_t)psz);
    step_ok("mmap(host page) -> %p", host_buf);

    /* ------------------------------------------------------------------ */
    /* [13] NVM_MAP_HOST_MEMORY against group_d -- happy path.           */
    /*                                                                    */
    /* This validates the whole new-mode pipeline: kernel finds the      */
    /* group via find_qgroup_locked, attaches the map to g->maps,        */
    /* increments g->nr_maps, and returns the DMA addresses.             */
    /* ------------------------------------------------------------------ */
    uint64_t io_addr = 0;
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)host_buf;
        req.n_pages     = 1;
        req.ioaddrs     = (uint64_t)(uintptr_t)&io_addr;
        req.ioq_idx     = -1;        /* new mode: ioq_idx ignored */
        req.is_cq       = -1;
        req.group_id    = group_d;
        if (do_ioctl(fd_a, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(group_d)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY new-mode against group %u",
                      group_d);
        if (io_addr == 0)
            step_fail(0, "NVM_MAP_HOST_MEMORY returned ioaddr=0 (DMA mapping failed?)");
        step_ok("NVM_MAP_HOST_MEMORY new-mode group=%u vaddr=%p ioaddr=0x%016" PRIx64,
                group_d, host_buf, io_addr);
    }

    /* ------------------------------------------------------------------ */
    /* [14] NVM_MAP_HOST_MEMORY with reserved!=0 must -EINVAL.           */
    /*                                                                    */
    /* Guards forward-compat: if a future kernel adds new flags via the  */
    /* reserved field, an old userspace that happens to pass garbage     */
    /* there must fail loudly rather than silently misinterpret.         */
    /* ------------------------------------------------------------------ */
    {
        /* Use a separate buffer so the failure path doesn't accidentally
         * shadow the [13] mapping above. */
        void* probe_buf = mmap(NULL, (size_t)psz, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (probe_buf == MAP_FAILED)
            step_fail(errno, "mmap(probe page) for reserved-MBZ test");

        uint64_t throwaway_ioaddr = 0;
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)probe_buf;
        req.n_pages     = 1;
        req.ioaddrs     = (uint64_t)(uintptr_t)&throwaway_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = group_d;
        /* B6: `reserved` was renamed to `map_kind` (1 B) +
         * `reserved0[3]`.  Probe the MBZ check by setting one of
         * the reserved padding bytes; the kernel must still
         * reject the request with -EINVAL.                       */
        req.reserved0[1] = 0xab;
        int r = ioctl(fd_a, NVM_MAP_HOST_MEMORY, &req);
        if (r == 0)
            step_fail(0, "NVM_MAP_HOST_MEMORY with reserved0!=0 unexpectedly "
                         "succeeded -- MBZ check missing");
        if (errno != EINVAL)
            step_fail(errno, "NVM_MAP_HOST_MEMORY with reserved0!=0 errno=%d, "
                             "expected EINVAL(%d)", errno, EINVAL);
        munmap(probe_buf, (size_t)psz);
        step_ok("NVM_MAP_HOST_MEMORY reserved-MBZ check returns -EINVAL");
    }

    /* ------------------------------------------------------------------ */
    /* [15] NVM_MAP_HOST_MEMORY with bogus group_id (never allocated)    */
    /* must -ENOENT.                                                      */
    /*                                                                    */
    /* This validates that find_qgroup_locked rejects unknown ids        */
    /* without leaking a half-pinned page (the kernel must               */
    /* unmap_and_release on the find-failure path).  We can't directly  */
    /* observe the leak from userspace; the proof is "test reruns        */
    /* without exhausting RLIMIT_MEMLOCK across many invocations".       */
    /* ------------------------------------------------------------------ */
    {
        void* probe_buf = mmap(NULL, (size_t)psz, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (probe_buf == MAP_FAILED)
            step_fail(errno, "mmap(probe page) for bogus-group test");

        uint64_t throwaway_ioaddr = 0;
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)probe_buf;
        req.n_pages     = 1;
        req.ioaddrs     = (uint64_t)(uintptr_t)&throwaway_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0xdeadbeefU;   /* almost certainly never assigned */
        int r = ioctl(fd_a, NVM_MAP_HOST_MEMORY, &req);
        if (r == 0)
            step_fail(0, "NVM_MAP_HOST_MEMORY against unknown group_id=0x%x "
                         "unexpectedly succeeded", 0xdeadbeefU);
        if (errno != ENOENT)
            step_fail(errno, "NVM_MAP_HOST_MEMORY against unknown group_id "
                             "errno=%d, expected ENOENT(%d)", errno, ENOENT);
        munmap(probe_buf, (size_t)psz);
        step_ok("NVM_MAP_HOST_MEMORY against bogus group_id returns -ENOENT");
    }

    /* ------------------------------------------------------------------ */
    /* [16] Cross-fd: open fd_d, try to register a host page under       */
    /* fd_a's group_d.  Must -ENOENT (per-fd isolation).                  */
    /* ------------------------------------------------------------------ */
    {
        int fd_d = open(dev_path, O_RDWR | O_NONBLOCK);
        if (fd_d < 0)
            step_fail(errno, "open(%s) for fd_d", dev_path);

        void* probe_buf = mmap(NULL, (size_t)psz, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (probe_buf == MAP_FAILED)
            step_fail(errno, "mmap(probe page) for cross-fd test");

        uint64_t throwaway_ioaddr = 0;
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)probe_buf;
        req.n_pages     = 1;
        req.ioaddrs     = (uint64_t)(uintptr_t)&throwaway_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = group_d;     /* fd_a's group, used from fd_d */
        int r = ioctl(fd_d, NVM_MAP_HOST_MEMORY, &req);
        if (r == 0)
            step_fail(0,
                "fd_d managed to register a map under fd_a's group %u -- "
                "per-fd isolation broken", group_d);
        if (errno != ENOENT)
            step_fail(errno, "cross-fd map registration errno=%d, "
                             "expected ENOENT(%d)", errno, ENOENT);
        munmap(probe_buf, (size_t)psz);
        close(fd_d);
        step_ok("cross-fd NVM_MAP_HOST_MEMORY against another fd's group "
                "correctly returns -ENOENT");
    }

    /* ------------------------------------------------------------------ */
    /* [17] Destroy group_d -- the registered map MUST be drained        */
    /* without an explicit NVM_UNMAP_HOST_MEMORY.                         */
    /*                                                                    */
    /* Proof points:                                                      */
    /*   - dmesg shows "destroy_qgroup id=N drained 1 map(s)"             */
    /*   - Re-issuing NVM_UNMAP_HOST_MEMORY for the same vaddr now       */
    /*     returns -EINVAL (the map is gone from the global host_list   */
    /*     too).                                                          */
    /* ------------------------------------------------------------------ */
    {
        uint32_t gid = group_d;
        if (do_ioctl(fd_a, NVM_DESTROY_QUEUE_GROUP, &gid,
                     "NVM_DESTROY_QUEUE_GROUP id=group_d") < 0)
            step_fail(errno, "NVM_DESTROY_QUEUE_GROUP id=%u (B2 cascade)",
                      group_d);
        step_ok("NVM_DESTROY_QUEUE_GROUP id=%u cascades through 1 map "
                "(grep dmesg for 'destroy_qgroup id=%u drained 1 map')",
                group_d, group_d);
    }

    /* ------------------------------------------------------------------ */
    /* [18] Confirm the cascade actually freed the map: an explicit      */
    /* NVM_UNMAP_HOST_MEMORY for the same vaddr must now -EINVAL.        */
    /* ------------------------------------------------------------------ */
    {
        uint64_t vaddr = (uint64_t)(uintptr_t)host_buf;
        int r = ioctl(fd_a, NVM_UNMAP_HOST_MEMORY, &vaddr);
        if (r == 0)
            step_fail(0,
                "NVM_UNMAP_HOST_MEMORY succeeded after destroy_qgroup -- "
                "the cascade did NOT remove the map from the global list");
        if (errno != EINVAL)
            step_fail(errno, "expected EINVAL(%d) after cascade, got errno=%d",
                      EINVAL, errno);
        step_ok("post-cascade NVM_UNMAP_HOST_MEMORY -EINVAL confirms map "
                "was freed");
    }

    munmap(host_buf, (size_t)psz);

    /* ------------------------------------------------------------------ */
    /* [19] Cleanup chrdev so the test is rerunnable.                    */
    /*                                                                    */
    /* close(fd_a) must NOT see any leftover group (we destroyed group_a */
    /* in step 6 and group_d in step 17; steps 4/10 failed before        */
    /* allocation, so nothing was attached).  No "cascade-destroyed N"  */
    /* line should appear for fd_a in dmesg.                             */
    /* ------------------------------------------------------------------ */
    if (close(fd_a) < 0)
        step_fail(errno, "close(fd_a)");
    step_ok("close(fd_a)");

    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_CHRDEV_REMOVE, &bdf, "SNVM_CHRDEV_REMOVE") < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE %s", bdf_str);
        step_ok("SNVM_CHRDEV_REMOVE %s", bdf_str);
    }
    close(fd_ctl);

    fprintf(stderr, "\n=== snvme_smoke_qgroup: all %d steps passed ===\n", g_step);
    return 0;
}

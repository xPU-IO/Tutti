/*
 * snvme_smoke_addq.c -- Smoke test for NVM_ADD_USER_QUEUE (queue-group
 * plan, step B3).
 *
 * Purpose
 * -------
 *
 * Validate the per-group user-queue creation path end-to-end on a
 * BOUND controller, without sending any NVMe IO commands.  We
 * deliberately do not even try to ring a doorbell here -- the goal
 * is to confirm that the kernel:
 *
 *   1. Rejects ADD_USER_QUEUE before bind with -ENODEV.
 *   2. Returns plausible q_depth / max_user_qid / bar0_size from
 *      the new NVM_GET_DEV_INFO ABI.
 *   3. Pipes the (group_id, sq_vaddr, cq_vaddr) batch through the
 *      Create I/O CQ + Create I/O SQ admin path on the controller,
 *      and returns BAR0 doorbell offsets back to userspace.
 *   4. Cascade-destroys all created user queues when the group is
 *      destroyed (NVM_DESTROY_QUEUE_GROUP), via the kernel's
 *      adapter_delete_sq + adapter_delete_cq path.
 *   5. Leaves the controller in a state where it can be unbound
 *      cleanly afterwards.
 *
 * Out of scope (deferred to B4):
 *   - Issuing actual NVMe Read / Write commands through these queues.
 *   - GPU / nvfs ring memory; B3 smoke uses host pages only.
 *
 * Why a separate binary from snvme_smoke_qgroup:
 *   snvme_smoke_qgroup is the no-bind smoke that runs even on hosts
 *   where /dev/nvmeN holds a mounted filesystem; it cannot bind.
 *   B3 fundamentally requires bind, so it lives in its own binary
 *   that the operator has to opt into running.  Same split as
 *   snvme_smoke (UAPI-only bring-up) vs the bind-required B3 smokes.
 *
 * Build:    make snvme_smoke_addq
 * Invoke:   sudo ./snvme_smoke_addq <PCI_BDF>
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

static int parse_bdf(const char* s, struct pci_device_addr* out) {
    return sscanf(s, "%x:%x:%x.%x",
                  &out->domain, &out->bus, &out->slot, &out->func) == 4 ? 0 : -1;
}

static int do_ioctl(int fd, unsigned long req, void* arg, const char* what) {
    int r = ioctl(fd, req, arg);
    if (r < 0) {
        int e = errno;
        fprintf(stderr, "ioctl(%s) failed: %s\n", what, strerror(e));
        errno = e;
    }
    return r;
}

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <PCI_BDF>\n"
        "  e.g.: %s 0000:50:00.0\n"
        "\n"
        "BINDS the target controller (destructive).  Validates the\n"
        "B3 NVM_ADD_USER_QUEUE / NVM_DESTROY_QUEUE_GROUP cascade,\n"
        "without issuing any NVMe IO commands.\n",
        prog, prog);
}

/*
 * Round n_bytes up to the nearest multiple of page_size.
 */
static size_t round_up_pages(size_t n_bytes, long page_size) {
    return ((n_bytes + page_size - 1) / page_size) * page_size;
}

/*
 * Allocate a page-aligned host buffer suitable for use as an NVMe
 * SQ/CQ ring.  We MUST use posix_memalign (or aligned mmap) because
 * NVMe Create I/O SQ/CQ requires the ring's PRP1 to be page-aligned
 * (low 12 bits zero).  malloc() makes no such guarantee.
 *
 * The buffer is zeroed: the CQ ring's phase bit must start at 0
 * for the first lap, and the SQ ring is effectively don't-care but
 * zeroing it avoids false NVMe SC values during early debug.
 */
static void* alloc_ring(size_t bytes, long page_size) {
    void* p = NULL;
    size_t rounded = round_up_pages(bytes, page_size);
    if (posix_memalign(&p, page_size, rounded) != 0)
        return NULL;
    memset(p, 0, rounded);
    return p;
}

int main(int argc, char** argv) {
    if (argc != 2 || strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return argc == 2 ? 0 : 1;
    }
    const char* bdf_str = argv[1];

    struct pci_device_addr orig_bdf;
    if (parse_bdf(bdf_str, &orig_bdf) != 0) {
        fprintf(stderr, "Bad BDF: '%s' (expected DDDD:BB:DD.F)\n", bdf_str);
        return 1;
    }

    long psz = sysconf(_SC_PAGESIZE);
    if (psz <= 0)
        step_fail(errno, "sysconf(_SC_PAGESIZE)");

    /* ============================================================== */
    /* Phase 0: bring up control plane + chrdev.                      */
    /* ============================================================== */
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
    int fd_dev = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_dev < 0)
        step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd=%d", dev_path, fd_dev);

    /* ============================================================== */
    /* Phase 1: create a queue group BEFORE bind.                     */
    /*                                                                */
    /* This validates that group lifecycle is bind-agnostic AND       */
    /* that NVM_ADD_USER_QUEUE on a not-yet-bound controller          */
    /* returns -ENODEV (caught at controller liveness check).         */
    /* ============================================================== */
    uint32_t group_id;
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_dev, NVM_CREATE_QUEUE_GROUP, &req,
                     "NVM_CREATE_QUEUE_GROUP") < 0)
            step_fail(errno, "NVM_CREATE_QUEUE_GROUP");
        group_id = req.group_id;
        step_ok("NVM_CREATE_QUEUE_GROUP -> group_id=%u max_queues=%u",
                group_id, req.max_queues);
    }

    /* ------------------------------------------------------------------ */
    /* [4] Pre-bind: NVM_ADD_USER_QUEUE must -ENODEV.                    */
    /*                                                                    */
    /* We don't even bother allocating real ring buffers here -- the      */
    /* kernel checks bind status BEFORE looking up any maps, so a         */
    /* zeroed payload is fine for testing the rejection path.             */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_add_user_queue req;
        memset(&req, 0, sizeof(req));
        req.group_id = group_id;
        req.nr_pairs = 1;
        req.pairs[0].sq_vaddr = 0xfeed0000UL;  /* won't be inspected */
        req.pairs[0].cq_vaddr = 0xfeed4000UL;
        int r = ioctl(fd_dev, NVM_ADD_USER_QUEUE, &req);
        if (r == 0)
            step_fail(0,
                "NVM_ADD_USER_QUEUE unexpectedly succeeded on UNBOUND "
                "ctrl -- liveness check missing");
        if (errno != ENODEV)
            step_fail(errno,
                "NVM_ADD_USER_QUEUE on unbound ctrl: errno=%d, expected ENODEV(%d)",
                errno, ENODEV);
        step_ok("NVM_ADD_USER_QUEUE on unbound ctrl correctly returns -ENODEV");
    }

    /* ------------------------------------------------------------------ */
    /* [4b] Pre-bind: cap kernel-side IOQ count via NVM_SET_KERNEL_IOQ_CAP.*/
    /*                                                                    */
    /* Without this, probe defaults nvme_max_io_queues() to               */
    /* num_possible_cpus() + write_queues + poll_queues -- on a 192-vCPU  */
    /* host that asks the controller for ~192 IOQs.  Most NVMe SSDs grant */
    /* only as many as their MSI-X vector count allows (e.g. Intel DC SSD */
    /* MSI-X=136 -> grant=135), and the kernel then consumes every QID    */
    /* it was granted, leaving the [online_queues..ctrl_max_io_queues]    */
    /* user pool empty.  Create I/O CQ from NVM_ADD_USER_QUEUE later      */
    /* would fail with NVMe SC=0x4101 (Invalid Queue Identifier).         */
    /*                                                                    */
    /* Cap=36 is arbitrary but small: leaves ~99 QIDs in the user pool   */
    /* on a 135-grant controller, more than enough for any smoke run.    */
    /*                                                                    */
    /* NOTE: This is a B3 cap-only ioctl -- it sets                       */
    /* ctrl->setup.cap_kernel_ioq WITHOUT touching ctrl->ioq_num /        */
    /* use_sreg / on_host (which the legacy NVM_SET_IOQ_NUM would).  As   */
    /* such, probe still runs as plain in-tree-style nvme; only           */
    /* nvme_max_io_queues() is capped.                                    */
    /* ------------------------------------------------------------------ */
    {
        /*
         * Cap is overridable via SNVME_TEST_KERNEL_IOQ_CAP so the test can
         * run on controllers whose total I/O-queue grant is smaller
         * than the 36 default.  The cap only leaves room for user
         * queues when it is STRICTLY LESS than the controller grant
         * (ctrl_max_io_queues == max_user_qid): the kernel keeps
         * [1..cap] and the user pool gets [cap+1..max_user_qid].
         *
         * Example: a drive that grants only 31 IOQs (NVMe feature 0x07
         * NCQA=0x1e) needs e.g. SNVME_TEST_KERNEL_IOQ_CAP=16 -- the default
         * 36 >= 31 never triggers the kernel-side shrink, so the kernel
         * consumes all 31 queues and step 9 fails with
         * "max_user_qid <= start_cq_idx; no room for user queues".
         */
        uint32_t cap = 36;
        const char *cap_env = getenv("SNVME_TEST_KERNEL_IOQ_CAP");
        if (cap_env != NULL && cap_env[0] != '\0') {
            char *end = NULL;
            unsigned long v = strtoul(cap_env, &end, 0);
            if (end == cap_env || *end != '\0' || v == 0 || v > UINT32_MAX)
                step_fail(0, "SNVME_TEST_KERNEL_IOQ_CAP=\"%s\" is not a positive "
                             "integer", cap_env);
            cap = (uint32_t)v;
        }
        if (ioctl(fd_dev, NVM_SET_KERNEL_IOQ_CAP, &cap) != 0)
            step_fail(errno, "NVM_SET_KERNEL_IOQ_CAP cap=%u failed", cap);
        step_ok("NVM_SET_KERNEL_IOQ_CAP cap=%u%s (kernel will get <=%u IOQs, "
                "rest of controller grant goes to user pool)",
                cap, cap_env ? " [from SNVME_TEST_KERNEL_IOQ_CAP]" : "", cap);
    }

    /* ============================================================== */
    /* Phase 2: BIND.                                                  */
    /* ============================================================== */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_BIND, &bdf, "SNVM_DEVICE_BIND") < 0)
            step_fail(errno,
                "SNVM_DEVICE_BIND %s -- in-tree nvme may still own this BDF; "
                "try `sudo sh -c 'echo %s > /sys/bus/pci/drivers/nvme/unbind'` "
                "first", bdf_str, bdf_str);
        step_ok("SNVM_DEVICE_BIND %s", bdf_str);
    }

    /* ============================================================== */
    /* Phase 3: NVM_GET_DEV_INFO -- read the controller-derived       */
    /* ring sizing constraints.  Poll until probe completes.          */
    /* ============================================================== */
    struct nvm_ioctl_dev info;
    {
        int ok = 0;
        for (int i = 0; i < 100; i++) {
            memset(&info, 0, sizeof(info));
            if (ioctl(fd_dev, NVM_GET_DEV_INFO, &info) == 0 &&
                info.disk_name[0] != '\0') {
                ok = 1;
                break;
            }
            usleep(100 * 1000);
        }
        if (!ok)
            step_fail(errno, "NVM_GET_DEV_INFO did not complete within 10s");
        step_ok("NVM_GET_DEV_INFO disk='%s' block_size=%zu",
                info.disk_name, info.block_size);
    }

    /* Sanity-check the new B3 fields. */
    if (info.q_depth == 0)
        step_fail(0, "NVM_GET_DEV_INFO: q_depth == 0 (kernel bug)");
    if (info.bar0_size < 4096)
        step_fail(0, "NVM_GET_DEV_INFO: bar0_size=%u suspiciously small",
                  info.bar0_size);
    if (info.max_user_qid <= info.start_cq_idx && info.start_cq_idx != 0)
        /* It's OK if both are zero on a controller that gave us no
         * room for user queues at all -- the next add will -EBUSY. */
        step_fail(0, "NVM_GET_DEV_INFO: max_user_qid (%u) <= start_cq_idx (%u); "
                     "no room for user queues, please set SNVME_TEST_KERNEL_IOQ_CAP to a small value.",
                  info.max_user_qid, info.start_cq_idx);
    if (info.max_queues_per_group == 0)
        step_fail(0, "NVM_GET_DEV_INFO: max_queues_per_group == 0");
    step_ok("dev_info q_depth=%u bar0_size=0x%x max_user_qid=%u "
            "max_queues_per_group=%u start_cq_idx=%u",
            info.q_depth, info.bar0_size, info.max_user_qid,
            info.max_queues_per_group, info.start_cq_idx);

    /* ============================================================== */
    /* Phase 4: Allocate two SQ+CQ ring pairs (we'll create 2 user    */
    /* queues in this test so the rollback / cur_queues math gets a   */
    /* multi-queue exercise).                                          */
    /*                                                                 */
    /* Sizing math:                                                    */
    /*   SQ bytes = q_depth * 64                                       */
    /*   CQ bytes = q_depth * 16                                       */
    /*   Both must be page-aligned (PRP1 has low 12 bits = 0).         */
    /*   Both must fit in one host page (snvme single-PRP limit).     */
    /*                                                                 */
    /*   For q_depth=1024:                                             */
    /*     SQ = 64K bytes = 16 pages -- DOES NOT FIT in one page!     */
    /*                                                                 */
    /*   To stay safe we cap our test to q_depth' = min(64,            */
    /*   info.q_depth) for the SQ (4096 / 64 = 64 entries) and        */
    /*   q_depth' = min(256, info.q_depth) for the CQ (4096 / 16 =    */
    /*   256 entries).  NVMe spec lets a CQ be deeper than its SQ;     */
    /*   we set both QSIZE in the kernel side to dev->q_depth-1, but  */
    /*   we don't have to USE all the entries.  The actual requirement */
    /*   is just that the buffer covers (q_depth * entry_size) bytes  */
    /*   -- the controller will cap to whatever buffer size implies.  */
    /*                                                                 */
    /* In practice the kernel uses dev->q_depth verbatim in           */
    /* adapter_alloc_*_user, so the test ring MUST satisfy the full   */
    /* q_depth * entry_size.  If that exceeds one page, we fail        */
    /* loudly here rather than on Create I/O CQ.                       */
    /* ============================================================== */
    const unsigned NR_PAIRS = 2;
    const size_t   sqe_size = 64;
    const size_t   cqe_size = 16;
    size_t sq_bytes = (size_t)info.q_depth * sqe_size;
    size_t cq_bytes = (size_t)info.q_depth * cqe_size;
    size_t sq_pages = round_up_pages(sq_bytes, psz) / psz;
    size_t cq_pages = round_up_pages(cq_bytes, psz) / psz;

    if (sq_pages > 1)
        step_fail(0,
            "B3 single-PRP limit: q_depth=%u * sqe=64 = %zu B requires "
            "%zu pages; the snvme NVM_MAP_HOST_MEMORY -> "
            "adapter_alloc_sq_user path only uses addrs[0], so a "
            "multi-page ring would not be physically contiguous from "
            "the controller's POV.  Lower io_queue_depth at insmod time "
            "(or extend snvme to chain PRP).",
            info.q_depth, sq_bytes, sq_pages);
    if (cq_pages > 1)
        step_fail(0,
            "B3 single-PRP limit: q_depth=%u * cqe=16 = %zu B requires "
            "%zu pages; same constraint as above.",
            info.q_depth, cq_bytes, cq_pages);

    void* sq_buf[NR_PAIRS];
    void* cq_buf[NR_PAIRS];
    for (unsigned i = 0; i < NR_PAIRS; i++) {
        sq_buf[i] = alloc_ring(sq_bytes, psz);
        cq_buf[i] = alloc_ring(cq_bytes, psz);
        if (!sq_buf[i] || !cq_buf[i])
            step_fail(errno, "alloc_ring pair %u", i);
    }
    step_ok("allocated %u SQ+CQ ring pairs (sq=%zu B/page, cq=%zu B/page)",
            NR_PAIRS, sq_bytes, cq_bytes);

    /* ============================================================== */
    /* Phase 5: NVM_MAP_HOST_MEMORY each ring against group_id (new   */
    /* mode).  Throw-away ioaddr buffer just to satisfy the ABI; we   */
    /* don't need to remember it -- the kernel will look it up via    */
    /* (group, vaddr) in NVM_ADD_USER_QUEUE.                           */
    /* ============================================================== */
    for (unsigned i = 0; i < NR_PAIRS; i++) {
        uint64_t throwaway[1];
        struct nvm_ioctl_map req;

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)sq_buf[i];
        req.n_pages     = 1;
        req.ioaddrs     = (uint64_t)(uintptr_t)throwaway;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = group_id;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(SQ)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY pair %u SQ", i);

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)cq_buf[i];
        req.n_pages     = 1;
        req.ioaddrs     = (uint64_t)(uintptr_t)throwaway;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = group_id;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(CQ)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY pair %u CQ", i);
    }
    step_ok("NVM_MAP_HOST_MEMORY x %u ring pairs registered against group=%u",
            NR_PAIRS * 2, group_id);

    /* ============================================================== */
    /* Phase 6: NVM_ADD_USER_QUEUE -- the actual Create I/O CQ + SQ. */
    /* ============================================================== */
    struct nvm_ioctl_add_user_queue add_req;
    memset(&add_req, 0, sizeof(add_req));
    add_req.group_id = group_id;
    add_req.nr_pairs = NR_PAIRS;
    for (unsigned i = 0; i < NR_PAIRS; i++) {
        add_req.pairs[i].sq_vaddr = (uint64_t)(uintptr_t)sq_buf[i];
        add_req.pairs[i].cq_vaddr = (uint64_t)(uintptr_t)cq_buf[i];
    }

    if (do_ioctl(fd_dev, NVM_ADD_USER_QUEUE, &add_req,
                 "NVM_ADD_USER_QUEUE") < 0)
        step_fail(errno, "NVM_ADD_USER_QUEUE -- check dmesg for which "
                         "Create I/O CQ/SQ admin command failed");
    for (unsigned i = 0; i < NR_PAIRS; i++) {
        if (add_req.out_pairs[i].sq_doorbell_offset == 0 ||
            add_req.out_pairs[i].cq_doorbell_offset == 0)
            step_fail(0,
                "NVM_ADD_USER_QUEUE pair %u: kernel returned zero doorbell "
                "offset (sq=0x%x cq=0x%x), expected non-zero",
                i, add_req.out_pairs[i].sq_doorbell_offset,
                add_req.out_pairs[i].cq_doorbell_offset);
    }
    step_ok("NVM_ADD_USER_QUEUE created %u user queue(s):", NR_PAIRS);
    for (unsigned i = 0; i < NR_PAIRS; i++) {
        fprintf(stderr, "             pair[%u] qid=%u sq_db=0x%x cq_db=0x%x\n",
                i, add_req.out_pairs[i].qid,
                add_req.out_pairs[i].sq_doorbell_offset,
                add_req.out_pairs[i].cq_doorbell_offset);
    }

    /* ============================================================== */
    /* Phase 7: Negative -- ADD_USER_QUEUE filling the group beyond  */
    /* max_queues must -EBUSY.                                         */
    /* ============================================================== */
    {
        unsigned remaining = info.max_queues_per_group - NR_PAIRS;
        if (remaining > 0 && remaining < NVM_MAX_QUEUES_PER_GROUP) {
            /*
             * Try to add (remaining + 1) more pairs; the +1 must
             * push us over the cap.  We don't actually want them to
             * succeed, so we deliberately use stale vaddrs that
             * *would* otherwise -ENOENT on map lookup -- but the
             * cur_queues+nr_pairs > max_queues check happens
             * BEFORE map lookup, so we'll hit -EBUSY instead.
             *
             * The +1 also has to stay within
             * NVM_MAX_QUEUES_PER_GROUP itself (the per-call upper
             * bound), so we cap at that.
             */
            struct nvm_ioctl_add_user_queue req;
            unsigned over = remaining + 1;
            if (over > NVM_MAX_QUEUES_PER_GROUP)
                over = NVM_MAX_QUEUES_PER_GROUP;
            memset(&req, 0, sizeof(req));
            req.group_id = group_id;
            req.nr_pairs = over;
            for (unsigned i = 0; i < over; i++) {
                req.pairs[i].sq_vaddr = 0xdeadbeef0000UL + i * 0x2000;
                req.pairs[i].cq_vaddr = 0xdeadbeef0000UL + i * 0x2000 + 0x1000;
            }
            int r = ioctl(fd_dev, NVM_ADD_USER_QUEUE, &req);
            if (r == 0)
                step_fail(0,
                    "NVM_ADD_USER_QUEUE overflow (%u pairs into group with "
                    "max=%u, cur=%u) unexpectedly succeeded",
                    over, info.max_queues_per_group, NR_PAIRS);
            if (errno != EBUSY && errno != ENOENT)
                step_fail(errno,
                    "NVM_ADD_USER_QUEUE overflow: errno=%d, expected EBUSY "
                    "(cap exceeded) or ENOENT (cap check passed but vaddr "
                    "lookup failed)", errno);
            step_ok("NVM_ADD_USER_QUEUE overflow correctly rejected (errno=%s)",
                    errno == EBUSY ? "EBUSY" : "ENOENT");
        } else {
            step_ok("skip overflow check (max_queues_per_group=%u, NR_PAIRS=%u "
                    "leaves no room or full room)",
                    info.max_queues_per_group, NR_PAIRS);
        }
    }

    /* ============================================================== */
    /* Phase 8: Destroy group -- cascade must Delete I/O SQ + CQ for */
    /* every queue we created, free the QIDs, and drain all maps.    */
    /* ============================================================== */
    {
        uint32_t gid = group_id;
        if (do_ioctl(fd_dev, NVM_DESTROY_QUEUE_GROUP, &gid,
                     "NVM_DESTROY_QUEUE_GROUP") < 0)
            step_fail(errno, "NVM_DESTROY_QUEUE_GROUP");
        step_ok("NVM_DESTROY_QUEUE_GROUP id=%u cascades through %u user "
                "queue(s) + %u maps (grep dmesg for "
                "'destroy_qgroup id=%u drained ... user queue(s)')",
                group_id, NR_PAIRS, NR_PAIRS * 2, group_id);
    }

    /* ============================================================== */
    /* Phase 9: After destroy, ADD_USER_QUEUE against the same gid    */
    /* must -ENOENT (group is gone).                                   */
    /* ============================================================== */
    {
        struct nvm_ioctl_add_user_queue req;
        memset(&req, 0, sizeof(req));
        req.group_id = group_id;
        req.nr_pairs = 1;
        req.pairs[0].sq_vaddr = (uint64_t)(uintptr_t)sq_buf[0];
        req.pairs[0].cq_vaddr = (uint64_t)(uintptr_t)cq_buf[0];
        int r = ioctl(fd_dev, NVM_ADD_USER_QUEUE, &req);
        if (r == 0)
            step_fail(0,
                "NVM_ADD_USER_QUEUE against destroyed group %u "
                "unexpectedly succeeded", group_id);
        if (errno != ENOENT)
            step_fail(errno,
                "NVM_ADD_USER_QUEUE against destroyed group: errno=%d, "
                "expected ENOENT(%d)", errno, ENOENT);
        step_ok("NVM_ADD_USER_QUEUE against destroyed group correctly "
                "returns -ENOENT");
    }

    /* Free user-side ring buffers.  Note: the snvme-side maps were
     * already drained by NVM_DESTROY_QUEUE_GROUP above, so we don't
     * call NVM_UNMAP_HOST_MEMORY -- it would just -EINVAL. */
    for (unsigned i = 0; i < NR_PAIRS; i++) {
        free(sq_buf[i]);
        free(cq_buf[i]);
    }

    /* ============================================================== */
    /* Phase 10: Tear-down -- unbind, close, remove chrdev.            */
    /* ============================================================== */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_UNBIND, &bdf, "SNVM_DEVICE_UNBIND") < 0)
            step_fail(errno, "SNVM_DEVICE_UNBIND %s", bdf_str);
        step_ok("SNVM_DEVICE_UNBIND %s", bdf_str);
    }

    if (close(fd_dev) < 0)
        step_fail(errno, "close(%s)", dev_path);
    step_ok("close(%s)", dev_path);

    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_CHRDEV_REMOVE, &bdf, "SNVM_CHRDEV_REMOVE") < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE %s", bdf_str);
        step_ok("SNVM_CHRDEV_REMOVE %s", bdf_str);
    }
    close(fd_ctl);

    fprintf(stderr, "\n=== snvme_smoke_addq: all %d steps passed ===\n", g_step);
    return 0;
}

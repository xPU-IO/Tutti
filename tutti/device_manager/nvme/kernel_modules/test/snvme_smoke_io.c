/*
 * snvme_smoke_io.c -- End-to-end NVMe Read/Write IO smoke test on user
 * IO queues created via NVM_ADD_USER_QUEUE.
 *
 * Builds on snvme_smoke_addq's bring-up (B1+B2+B3 ioctl plumbing) by
 * actually issuing NVMe Read / Write commands on the resulting user
 * IOQs and verifying the round-trip data integrity.  Goal: prove that
 *
 *   1. The doorbell offset returned by NVM_ADD_USER_QUEUE is correct
 *      and writeable from user space (after BAR0 mmap).
 *   2. The user-side SQ ring placement (vaddr -> dma_addr from
 *      NVM_MAP_HOST_MEMORY) is what the controller reads SQEs from.
 *   3. The user-side CQ ring receives CQEs with the right phase bit
 *      and command_id echo.
 *   4. Data PRP1 (also via NVM_MAP_HOST_MEMORY) round-trips: a Write
 *      command lands the host-side bytes on the device, and a Read
 *      command on the same LBA brings them back unchanged.
 *   5. Multiple IOs per queue work (covers SQ tail wrap-around within
 *      q_depth, and CQ phase flips).
 *   6. Multiple queues coexist (qid=37 for writes, qid=38 for reads)
 *      without interfering with each other's CQE delivery.
 *
 * DESTRUCTIVE: writes to LBAs starting at TEST_LBA_BASE (default
 * 2621440 = 10 GiB / 4 KiB).  Caller has confirmed the target NVMe
 * is empty / disposable.
 *
 * Build:    make snvme_smoke_io
 * Invoke:   sudo ./snvme_smoke_io <PCI_BDF>
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
#include <sched.h>
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
/* NVMe spec constants (subset; we don't pull in the kernel's nvme.h) */
/* ------------------------------------------------------------------ */

#define NVME_OPC_FLUSH            0x00
#define NVME_OPC_WRITE            0x01
#define NVME_OPC_READ             0x02

#define NVME_SQE_SIZE             64u
#define NVME_CQE_SIZE             16u

/* Test parameters.  TEST_LBA_BASE = 10 GiB on a 4-KiB-LBA disk; well
 * past any partition table the user might add later.  Bumping this
 * up is fine; lower than ~1 GiB risks colliding with GPT/superblock
 * playgrounds that some platforms autocreate.                          */
#define TEST_LBA_BASE             2621440ULL   /* 10 GiB / 4 KiB */
#define TEST_NR_QUEUES            2u
#define TEST_NR_IO_PER_QUEUE      16u

/* Default opcode-specific dataword pattern.  Must be byte-stable so we
 * can also verify on big-endian, even though x86 is the only platform
 * we test on right now.                                                */
#define WRITE_PATTERN_BYTE(qid, ioidx) \
    ((uint8_t)(0xA5 ^ ((qid) & 0xff) ^ ((ioidx) & 0xff)))

/* ------------------------------------------------------------------ */
/* NVMe submission queue entry (Common Format, NVMe 1.4 figure 105).  */
/* We construct these by hand because libnvm's nvm_cmd_t is C++-       */
/* templated and would drag in CUDA headers; the smoke is libc-only.   */
/* ------------------------------------------------------------------ */

struct nvme_sqe {
    /* CDW0 */
    uint8_t  opcode;
    uint8_t  flags;
    uint16_t cid;
    /* CDW1 */
    uint32_t nsid;
    /* CDW2-3 */
    uint64_t rsvd_2_3;
    /* CDW4-5 */
    uint64_t metadata;
    /* CDW6-7 */
    uint64_t prp1;
    /* CDW8-9 */
    uint64_t prp2;
    /* CDW10-15: opcode-specific.  For Read/Write:
     *   CDW10-11 = SLBA (64-bit)
     *   CDW12    = NLB (low 16) | reserved | PRINFO | FUA | LR
     *   CDW13    = DSM
     *   CDW14    = ELBST/ILBRT
     *   CDW15    = ELBATM/ELBAT
     */
    uint32_t cdw10;
    uint32_t cdw11;
    uint32_t cdw12;
    uint32_t cdw13;
    uint32_t cdw14;
    uint32_t cdw15;
} __attribute__((packed));

_Static_assert(sizeof(struct nvme_sqe) == NVME_SQE_SIZE,
               "nvme_sqe must be exactly 64 bytes");

/* NVMe completion queue entry (NVMe 1.4 figure 39). */
struct nvme_cqe {
    uint32_t result;        /* DW0: command-specific */
    uint32_t rsvd;          /* DW1: reserved */
    uint16_t sq_head;       /* DW2 lo */
    uint16_t sq_id;         /* DW2 hi */
    uint16_t cid;           /* DW3 lo */
    uint16_t status;        /* DW3 hi: phase bit in [0], SC in [8:1], SCT in [11:9] */
} __attribute__((packed));

_Static_assert(sizeof(struct nvme_cqe) == NVME_CQE_SIZE,
               "nvme_cqe must be exactly 16 bytes");

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
        "  e.g.: %s 0000:08:00.0\n"
        "\n"
        "BINDS the target controller and writes/reads %llu LBAs starting at\n"
        "LBA %llu (=10 GiB on a 4 KiB block device).  DESTRUCTIVE.\n",
        prog, prog,
        (unsigned long long)(TEST_NR_QUEUES * TEST_NR_IO_PER_QUEUE),
        (unsigned long long)TEST_LBA_BASE);
}

/*
 * Round n_bytes up to the nearest multiple of page_size.
 */
static size_t round_up_pages(size_t n_bytes, long page_size) {
    return ((n_bytes + page_size - 1) / page_size) * page_size;
}

/*
 * Allocate a page-aligned host buffer suitable for use as an NVMe
 * SQ/CQ ring or PRP1 data page.
 */
static void* alloc_aligned(size_t bytes, long page_size) {
    void* p = NULL;
    size_t rounded = round_up_pages(bytes, page_size);
    if (posix_memalign(&p, page_size, rounded) != 0)
        return NULL;
    memset(p, 0, rounded);
    return p;
}

/* ------------------------------------------------------------------ */
/* Doorbell writes.  BAR0 is mapped WC/UC (snvme uses pgprot_noncached)*/
/* so a plain volatile store reaches the controller without caching;   */
/* we still issue an sfence first to flush any prior WC stores into    */
/* the SQ ring before the controller sees the new tail.                */
/* ------------------------------------------------------------------ */

static inline void mmio_writel(volatile uint32_t* addr, uint32_t value) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__ ("sfence" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
    *addr = value;
}

/* ------------------------------------------------------------------ */
/* Per-queue runtime state for the test driver.                       */
/* ------------------------------------------------------------------ */

struct test_queue {
    uint16_t            qid;
    uint16_t            q_depth;

    /* SQ ring (host vaddr; controller reads via PRP1 dma_addr we
     * registered with NVM_MAP_HOST_MEMORY).                         */
    struct nvme_sqe*    sq;
    uint16_t            sq_tail;

    /* CQ ring (host vaddr; controller writes here, we poll). */
    struct nvme_cqe*    cq;
    uint16_t            cq_head;
    uint8_t             cq_phase;   /* expected next phase bit (0 or 1) */

    /* Doorbell pointers in mmap'd BAR0 region. */
    volatile uint32_t*  sq_db;
    volatile uint32_t*  cq_db;

    /* Monotonic command id source.  We don't bother recycling; q_depth
     * is plenty.                                                      */
    uint16_t            next_cid;
};

/*
 * Submit one Read/Write SQE on this queue.  Caller has already
 * computed the data pointer (PRP1/PRP2 or SGL1) and packed it into
 * dptr0/dptr1 along with the right CDW0 PSDT bits in `flags`.
 *
 * For PRP-style commands (PSDT=0):
 *   dptr0 = PRP1 (may have page offset)
 *   dptr1 = PRP2 (page-aligned data page, OR PRP List page address,
 *           OR 0 when transfer fits in one page)
 *   flags = 0
 *
 * For SGL-style data block (PSDT=01b):
 *   dptr0 = low 64 bits of the 16-byte SGL Data Block descriptor
 *           (i.e. the descriptor's `address` field)
 *   dptr1 = high 64 bits (length:32 | reserved:24 | type:8)
 *   flags = NVME_CMD_SGL_METABUF (0x40, PSDT=01b in CDW0)
 */
static void tq_submit_rw(struct test_queue* q,
                         uint8_t opcode,
                         uint8_t flags,
                         uint32_t nsid,
                         uint64_t dptr0,
                         uint64_t dptr1,
                         uint64_t slba,
                         uint16_t nlb_zero_based,
                         uint16_t* cid_out) {
    struct nvme_sqe sqe;
    memset(&sqe, 0, sizeof(sqe));

    uint16_t cid = q->next_cid++;
    sqe.opcode = opcode;
    sqe.flags  = flags;
    sqe.cid    = cid;
    sqe.nsid   = nsid;
    sqe.prp1   = dptr0;
    sqe.prp2   = dptr1;
    sqe.cdw10  = (uint32_t)(slba & 0xffffffffu);
    sqe.cdw11  = (uint32_t)(slba >> 32);
    sqe.cdw12  = nlb_zero_based & 0xffffu;

    /* Copy SQE into the ring at sq_tail. */
    q->sq[q->sq_tail] = sqe;

    /* Advance tail (modular).  Ring the doorbell with the NEW tail
     * value -- NVMe doorbell semantics is "this is where I have not
     * yet written".                                                  */
    uint16_t new_tail = (uint16_t)((q->sq_tail + 1) % q->q_depth);
    q->sq_tail = new_tail;
    mmio_writel(q->sq_db, new_tail);

    if (cid_out) *cid_out = cid;
}

/* NVMe spec PSDT bits, in CDW0[15:14] -- expressed as raw flags byte
 * (CDW0[7:0] = opcode, CDW0[15:8] = flags).  PSDT lives in [15:14] of
 * CDW0, i.e. bits [7:6] of the flags byte.
 *   00b = PRP, 01b = SGL data block, 10b = SGL with metadata SGL.   */
#define NVME_FLAG_PSDT_PRP     (0u << 6)
#define NVME_FLAG_PSDT_SGL     (1u << 6)

/* SGL descriptor type / subtype field (high byte of the 16-byte
 * descriptor).  Spec figure 105.  We only build the simplest form:
 *   type = 0 (Data Block), subtype = 0 (Address).                  */
#define NVME_SGL_TYPE_DATA_BLOCK   (0x0u << 4)
#define NVME_SGL_SUBTYPE_ADDR      (0x0u)
#define NVME_SGL_DESC_BYTE15       (NVME_SGL_TYPE_DATA_BLOCK | \
                                    NVME_SGL_SUBTYPE_ADDR)

/*
 * Wait for any CQE on this queue's CQ.  Returns 0 on success and fills
 * *cqe_out; returns -ETIMEDOUT if the controller never wrote one.
 *
 * Polls in a tight CPU loop with a 5-second wall-clock timeout (enough
 * for any sane controller; misbehaviour shows as timeout rather than
 * smoke hang).
 */
static int tq_poll_one(struct test_queue* q,
                       struct nvme_cqe* cqe_out,
                       unsigned timeout_ms) {
    /* Crude polling timer: we don't call clock_gettime in the inner
     * loop; instead we count iterations and back off after N misses
     * by yielding the CPU.  At ~100 ns/iter that's 5e7 iters per 5 s.   */
    unsigned long long max_iters = (unsigned long long)timeout_ms * 100000ULL;
    unsigned long long i = 0;

    for (;;) {
        volatile struct nvme_cqe* slot = &q->cq[q->cq_head];
        uint16_t status = slot->status;
        uint8_t phase = status & 0x1;
        if (phase == q->cq_phase) {
            /* CQE valid.  Copy out before advancing the head, then
             * ring the CQ doorbell so the controller knows we
             * consumed it.                                          */
            *cqe_out = *(const struct nvme_cqe*)slot;

            uint16_t new_head = (uint16_t)((q->cq_head + 1) % q->q_depth);
            if (new_head == 0)
                q->cq_phase ^= 1;     /* wrap -> phase flips */
            q->cq_head = new_head;
            mmio_writel(q->cq_db, new_head);
            return 0;
        }
        if (++i > max_iters) {
            return -ETIMEDOUT;
        }
        /* Release on every 4096th iter; avoids hogging a core if the
         * controller is slow.                                       */
        if ((i & 0xfff) == 0)
            sched_yield();
    }
}

/* Pretty-print an NVMe status word.  Phase bit (bit 0) is masked out
 * since we already consumed it in tq_poll_one.                        */
static void format_status(uint16_t status, char* buf, size_t cap) {
    uint16_t s   = status >> 1;            /* SF + DNR + M */
    uint8_t  sc  = s & 0xff;
    uint8_t  sct = (s >> 8) & 0x7;
    snprintf(buf, cap, "0x%04x (SC=0x%02x SCT=0x%x)", status, sc, sct);
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
    int fd_dev = open(dev_path, O_RDWR);
    if (fd_dev < 0)
        step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd=%d", dev_path, fd_dev);

    /* ============================================================== */
    /* Phase 1: queue group + cap + bind + dev info.                  */
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
        step_ok("NVM_SET_KERNEL_IOQ_CAP cap=%u%s (rest of grant -> user pool)",
                cap, cap_env ? " [from SNVME_TEST_KERNEL_IOQ_CAP]" : "");
    }

    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_BIND, &bdf, "SNVM_DEVICE_BIND") < 0)
            step_fail(errno, "SNVM_DEVICE_BIND %s", bdf_str);
        step_ok("SNVM_DEVICE_BIND %s", bdf_str);
    }

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
        step_ok("NVM_GET_DEV_INFO disk='%s' block_size=%zu q_depth=%u "
                "start_cq_idx=%u max_user_qid=%u",
                info.disk_name, info.block_size, info.q_depth,
                info.start_cq_idx, info.max_user_qid);
    }

    if (info.block_size != 4096)
        step_fail(0,
            "smoke assumes a 4 KiB-LBA controller; this disk reports "
            "block_size=%zu.  Adjust TEST_LBA_BASE / read-buffer size "
            "in the source if you really want to test a 512 B disk.",
            info.block_size);

    /* ============================================================== */
    /* Phase 2: allocate SQ+CQ rings + data buffers, register all     */
    /* against the queue group via NVM_MAP_HOST_MEMORY.               */
    /*                                                                 */
    /* We allocate one shared write-source buffer (filled with the    */
    /* per-IO pattern before each Write) and one shared read-target   */
    /* buffer (zeroed before each Read, then verified).  Same buffer  */
    /* reused across IOs because the controller serialises with the   */
    /* CQE (we only submit one IO at a time per queue and wait for it */
    /* to complete before reusing the buffer).                        */
    /* ============================================================== */

    const size_t   sqe_size = NVME_SQE_SIZE;
    const size_t   cqe_size = NVME_CQE_SIZE;
    size_t sq_bytes = (size_t)info.q_depth * sqe_size;
    size_t cq_bytes = (size_t)info.q_depth * cqe_size;

    if (round_up_pages(sq_bytes, psz) > (size_t)psz)
        step_fail(0,
            "B3 single-PRP limit: SQ ring (q_depth=%u * %zu = %zu B) "
            "spans more than one host page; lower io_queue_depth.",
            info.q_depth, sqe_size, sq_bytes);
    if (round_up_pages(cq_bytes, psz) > (size_t)psz)
        step_fail(0,
            "B3 single-PRP limit: CQ ring (q_depth=%u * %zu = %zu B) "
            "spans more than one host page.",
            info.q_depth, cqe_size, cq_bytes);

    void* sq_buf[TEST_NR_QUEUES];
    void* cq_buf[TEST_NR_QUEUES];
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        sq_buf[i] = alloc_aligned(sq_bytes, psz);
        cq_buf[i] = alloc_aligned(cq_bytes, psz);
        if (!sq_buf[i] || !cq_buf[i])
            step_fail(errno, "alloc_aligned ring pair %u", i);
    }

    /* One Write-source page and one Read-target page (both 4 KiB,
     * one LBA's worth on this disk).                                */
    void* wbuf = alloc_aligned(info.block_size, psz);
    void* rbuf = alloc_aligned(info.block_size, psz);
    if (!wbuf || !rbuf)
        step_fail(errno, "alloc_aligned data buffers");
    step_ok("allocated %u SQ+CQ ring pairs + 2 data buffers (block=%zu)",
            TEST_NR_QUEUES, info.block_size);

    /* Register every ring + data buffer against the group.  We need
     * the ioaddr write-back for the data buffers (PRP1 below) but
     * not for the rings (the kernel resolves those internally
     * during NVM_ADD_USER_QUEUE).                                   */
    uint64_t wbuf_ioaddr = 0;
    uint64_t rbuf_ioaddr = 0;

    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        uint64_t throwaway[1];
        struct nvm_ioctl_map req;

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)sq_buf[i];
        req.n_pages     = 1;
        req.ioaddrs     = (uint64_t)(uintptr_t)throwaway;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = group_id;
        req.map_kind    = NVM_MAP_KIND_RING_SQ;
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
        req.map_kind    = NVM_MAP_KIND_RING_CQ;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(CQ)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY pair %u CQ", i);
    }

    /* Data buffers: capture ioaddr for PRP1 use later.  Registered
     * with kind=DATA + group_id=0 so they are fd-scoped (B6) -- they
     * survive NVM_DESTROY_QUEUE_GROUP and only get reaped on close().
     * This is what Phase 7 below actually verifies.                 */
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)wbuf;
        req.n_pages     = 1;
        req.ioaddrs     = (uint64_t)(uintptr_t)&wbuf_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(wbuf)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY wbuf");
    }
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)rbuf;
        req.n_pages     = 1;
        req.ioaddrs     = (uint64_t)(uintptr_t)&rbuf_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(rbuf)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY rbuf");
    }
    step_ok("NVM_MAP_HOST_MEMORY x %u rings + 2 data buffers (wbuf ioaddr=0x%llx, "
            "rbuf ioaddr=0x%llx)",
            TEST_NR_QUEUES * 2,
            (unsigned long long)wbuf_ioaddr,
            (unsigned long long)rbuf_ioaddr);

    /* ============================================================== */
    /* Phase 3: NVM_ADD_USER_QUEUE -- create queues and capture       */
    /* doorbell offsets.                                               */
    /* ============================================================== */
    struct nvm_ioctl_add_user_queue add_req;
    memset(&add_req, 0, sizeof(add_req));
    add_req.group_id = group_id;
    add_req.nr_pairs = TEST_NR_QUEUES;
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        add_req.pairs[i].sq_vaddr = (uint64_t)(uintptr_t)sq_buf[i];
        add_req.pairs[i].cq_vaddr = (uint64_t)(uintptr_t)cq_buf[i];
    }
    if (do_ioctl(fd_dev, NVM_ADD_USER_QUEUE, &add_req,
                 "NVM_ADD_USER_QUEUE") < 0)
        step_fail(errno, "NVM_ADD_USER_QUEUE -- check dmesg for which "
                         "Create I/O CQ/SQ admin command failed");

    step_ok("NVM_ADD_USER_QUEUE created %u user queue(s)", TEST_NR_QUEUES);
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        fprintf(stderr, "             pair[%u] qid=%u sq_db=0x%x cq_db=0x%x\n",
                i, add_req.out_pairs[i].qid,
                add_req.out_pairs[i].sq_doorbell_offset,
                add_req.out_pairs[i].cq_doorbell_offset);
    }

    /* ============================================================== */
    /* Phase 4: mmap BAR0 so we can ring the doorbells from user      */
    /* space.                                                          */
    /* ============================================================== */
    void* bar0 = mmap(NULL, info.bar0_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_dev, 0);
    if (bar0 == MAP_FAILED)
        step_fail(errno, "mmap BAR0 (%u bytes) on fd=%d", info.bar0_size, fd_dev);
    step_ok("mmap BAR0 size=0x%x at %p (snvme svm_mmap_registers)",
            info.bar0_size, bar0);

    /* Build per-queue runtime state. */
    struct test_queue Q[TEST_NR_QUEUES];
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        memset(&Q[i], 0, sizeof(Q[i]));
        Q[i].qid       = (uint16_t)add_req.out_pairs[i].qid;
        Q[i].q_depth   = info.q_depth;
        Q[i].sq        = (struct nvme_sqe*)sq_buf[i];
        Q[i].cq        = (struct nvme_cqe*)cq_buf[i];
        Q[i].sq_tail   = 0;
        Q[i].cq_head   = 0;
        Q[i].cq_phase  = 1;        /* CQ ring zeroed, controller XORs to 1 first lap */
        Q[i].sq_db     = (volatile uint32_t*)
                         ((char*)bar0 + add_req.out_pairs[i].sq_doorbell_offset);
        Q[i].cq_db     = (volatile uint32_t*)
                         ((char*)bar0 + add_req.out_pairs[i].cq_doorbell_offset);
        Q[i].next_cid  = 0;
    }

    /* ============================================================== */
    /* Phase 5: Sequential 1-LBA write+verify per queue, repeated     */
    /* TEST_NR_IO_PER_QUEUE times.  Each (queue, io) pair gets a      */
    /* unique LBA (no aliasing), and the byte pattern encodes both    */
    /* the qid and the iteration index so cross-queue/cross-io        */
    /* corruption is detectable from a single byte.                   */
    /*                                                                 */
    /* Layout:                                                         */
    /*   write queue (Q[0]): writes LBA = TEST_LBA_BASE + i            */
    /*                       for i in [0, TEST_NR_IO_PER_QUEUE)        */
    /*   read  queue (Q[1]): reads back the SAME LBA range             */
    /*                                                                 */
    /* This exercises:                                                 */
    /*   - sq_tail advancing 16 times (still within q_depth=64,        */
    /*     so no wrap-around -- separate test for that below)          */
    /*   - cq_phase staying at 1 the whole time                        */
    /*   - cross-queue isolation: Q[0]'s CQEs never appear in Q[1].   */
    /* ============================================================== */
    {
        const uint32_t nsid = 1;       /* snvme exposes ns 1 */
        const uint16_t nlb_zero_based = 0;  /* 1 LBA per IO */

        struct test_queue* qw = &Q[0];
        struct test_queue* qr = &Q[1];
        char status_buf[64];

        for (unsigned i = 0; i < TEST_NR_IO_PER_QUEUE; i++) {
            uint64_t lba = TEST_LBA_BASE + i;
            uint8_t  pat = WRITE_PATTERN_BYTE(qw->qid, i);

            /* Stamp the write buffer with this iteration's pattern. */
            memset(wbuf, pat, info.block_size);

            uint16_t cid_w;
            tq_submit_rw(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP, nsid,
                         wbuf_ioaddr, /*dptr1=*/0,
                         lba, nlb_zero_based, &cid_w);

            struct nvme_cqe cqe_w;
            int rc = tq_poll_one(qw, &cqe_w, /*timeout_ms=*/5000);
            if (rc != 0)
                step_fail(-rc, "Write IO %u (qid=%u, lba=%" PRIu64 "): "
                               "CQE poll timed out",
                          i, qw->qid, lba);
            if ((cqe_w.status >> 1) != 0) {
                format_status(cqe_w.status, status_buf, sizeof(status_buf));
                step_fail(0, "Write IO %u (qid=%u, lba=%" PRIu64
                             ") returned non-zero NVMe status %s",
                          i, qw->qid, lba, status_buf);
            }
            if (cqe_w.cid != cid_w)
                step_fail(0, "Write IO %u CQE.cid=%u, expected %u "
                             "(SQ/CQ command_id mismatch)",
                          i, cqe_w.cid, cid_w);

            /* Now read it back from the OTHER queue. */
            memset(rbuf, 0, info.block_size);

            uint16_t cid_r;
            tq_submit_rw(qr, NVME_OPC_READ, NVME_FLAG_PSDT_PRP, nsid,
                         rbuf_ioaddr, /*dptr1=*/0,
                         lba, nlb_zero_based, &cid_r);

            struct nvme_cqe cqe_r;
            rc = tq_poll_one(qr, &cqe_r, 5000);
            if (rc != 0)
                step_fail(-rc, "Read IO %u (qid=%u, lba=%" PRIu64 "): "
                               "CQE poll timed out",
                          i, qr->qid, lba);
            if ((cqe_r.status >> 1) != 0) {
                format_status(cqe_r.status, status_buf, sizeof(status_buf));
                step_fail(0, "Read IO %u (qid=%u, lba=%" PRIu64
                             ") returned non-zero NVMe status %s",
                          i, qr->qid, lba, status_buf);
            }
            if (cqe_r.cid != cid_r)
                step_fail(0, "Read IO %u CQE.cid=%u, expected %u",
                          i, cqe_r.cid, cid_r);

            /* Verify every byte. */
            uint8_t* rbytes = (uint8_t*)rbuf;
            for (size_t b = 0; b < info.block_size; b++) {
                if (rbytes[b] != pat) {
                    step_fail(0, "Read IO %u (lba=%" PRIu64 ") byte %zu = "
                                 "0x%02x, expected 0x%02x",
                              i, lba, b, rbytes[b], pat);
                }
            }
        }
        step_ok("write+verify x %u IOs across qid=%u (write) / qid=%u (read), "
                "LBA [%" PRIu64 "..%" PRIu64 "], 4 KiB each",
                TEST_NR_IO_PER_QUEUE, qw->qid, qr->qid,
                (uint64_t)TEST_LBA_BASE,
                (uint64_t)(TEST_LBA_BASE + TEST_NR_IO_PER_QUEUE - 1));
    }

    /* ============================================================== */
    /* Phase 5b: PRP1 + PRP2 (dual-PRP), 8 KiB IO = 2 host pages.     */
    /*                                                                 */
    /* NVMe PRP rules (1.4 spec, figure 11):                           */
    /*   PRP1 may have a page offset (low bits non-zero).              */
    /*   PRP2 MUST be page-aligned.                                    */
    /*   When the transfer crosses exactly one page boundary,          */
    /*   PRP1 = first-page-dma, PRP2 = second-page-dma.                */
    /*                                                                 */
    /* Buffer layout: 2 contiguous host pages, vaddr-aligned.  We      */
    /* register both pages in ONE NVM_MAP_HOST_MEMORY (n_pages=2),     */
    /* getting back ioaddrs[0] = page0_dma, ioaddrs[1] = page1_dma --  */
    /* the kernel pins each page individually so the dmas are NOT      */
    /* required to be physically contiguous.                           */
    /*                                                                 */
    /* Pattern: byte at offset b in the 8KiB write buffer =            */
    /*   0xA5 ^ qid ^ io_idx ^ (b >> 12)                               */
    /* so we can detect a page-swap bug (controller wrote page 0       */
    /* into PRP2 etc.) by spotting the wrong sub-byte at offset 0      */
    /* vs offset 4096.                                                 */
    /* ============================================================== */
    void* wbuf2 = alloc_aligned(2 * info.block_size, psz);
    void* rbuf2 = alloc_aligned(2 * info.block_size, psz);
    if (!wbuf2 || !rbuf2)
        step_fail(errno, "alloc_aligned 2-page buffers");

    uint64_t wbuf2_ioaddr[2] = {0, 0};
    uint64_t rbuf2_ioaddr[2] = {0, 0};
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)wbuf2;
        req.n_pages     = 2;
        req.ioaddrs     = (uint64_t)(uintptr_t)wbuf2_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(wbuf2 x 2)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY wbuf2");

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)rbuf2;
        req.n_pages     = 2;
        req.ioaddrs     = (uint64_t)(uintptr_t)rbuf2_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(rbuf2 x 2)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY rbuf2");
    }
    step_ok("Phase 5b: 2-page data buffers registered "
            "(wbuf2 ioaddrs=[0x%llx,0x%llx])",
            (unsigned long long)wbuf2_ioaddr[0],
            (unsigned long long)wbuf2_ioaddr[1]);

    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 1;     /* 2 LBAs per IO */
        const uint64_t LBA_PHASE_5B = TEST_LBA_BASE + 100;
        const unsigned NR_IO_5B = 8;
        const size_t io_bytes = 2 * info.block_size;

        struct test_queue* qw = &Q[0];
        struct test_queue* qr = &Q[1];
        char status_buf[64];

        for (unsigned i = 0; i < NR_IO_5B; i++) {
            uint64_t lba = LBA_PHASE_5B + 2u * i;       /* 2 LBA stride */
            uint8_t  pat = WRITE_PATTERN_BYTE(qw->qid, 100 + i);

            /* Stamp wbuf2 with sub-byte mixing per page. */
            uint8_t* wbytes = (uint8_t*)wbuf2;
            for (size_t b = 0; b < io_bytes; b++)
                wbytes[b] = (uint8_t)(pat ^ (uint8_t)(b >> 12));

            uint16_t cid_w;
            tq_submit_rw(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP, nsid,
                         wbuf2_ioaddr[0], wbuf2_ioaddr[1],
                         lba, nlb_zero_based, &cid_w);

            struct nvme_cqe cqe;
            int rc = tq_poll_one(qw, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "5b Write %u (qid=%u, lba=%" PRIu64 ") timeout",
                          i, qw->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "5b Write %u: NVMe %s", i, status_buf);
            }
            if (cqe.cid != cid_w)
                step_fail(0, "5b Write %u CQE.cid=%u, expected %u",
                          i, cqe.cid, cid_w);

            /* Read back. */
            memset(rbuf2, 0, io_bytes);
            uint16_t cid_r;
            tq_submit_rw(qr, NVME_OPC_READ, NVME_FLAG_PSDT_PRP, nsid,
                         rbuf2_ioaddr[0], rbuf2_ioaddr[1],
                         lba, nlb_zero_based, &cid_r);
            rc = tq_poll_one(qr, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "5b Read %u (qid=%u, lba=%" PRIu64 ") timeout",
                          i, qr->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "5b Read %u: NVMe %s", i, status_buf);
            }
            if (cqe.cid != cid_r)
                step_fail(0, "5b Read %u CQE.cid=%u, expected %u",
                          i, cqe.cid, cid_r);

            /* Verify byte-for-byte against the same per-page pattern. */
            uint8_t* rbytes = (uint8_t*)rbuf2;
            for (size_t b = 0; b < io_bytes; b++) {
                uint8_t expect = (uint8_t)(pat ^ (uint8_t)(b >> 12));
                if (rbytes[b] != expect)
                    step_fail(0, "5b IO %u byte %zu: got 0x%02x, "
                                 "expected 0x%02x (page index %zu)",
                              i, b, rbytes[b], expect, b >> 12);
            }
        }
        step_ok("Phase 5b: PRP1+PRP2 dual-PRP x %u IOs, 8 KiB each, "
                "LBA [%" PRIu64 "..%" PRIu64 "]",
                NR_IO_5B, LBA_PHASE_5B, LBA_PHASE_5B + 2u * (NR_IO_5B - 1) + 1);
    }

    /* ============================================================== */
    /* Phase 5c: PRP1 + PRP List, 16 KiB IO = 4 host pages.           */
    /*                                                                 */
    /* When transfer > 2 pages, NVMe spec requires:                    */
    /*   PRP1 = first-page-dma (with optional offset)                  */
    /*   PRP2 = PRP_LIST_PAGE_DMA (ABSOLUTELY page-aligned, low 12     */
    /*          bits zero)                                             */
    /*   PRP_LIST_PAGE[i] = page (i+1) dma_addr                        */
    /*                                                                 */
    /* For 4 data pages we need PRP_LIST entries [page1, page2, page3].*/
    /* The PRP List itself is one host page allocated for this purpose,*/
    /* registered with NVM_MAP_HOST_MEMORY so the kernel pins it and   */
    /* hands us the dma_addr to put in PRP2.                           */
    /* ============================================================== */
    const unsigned PRP_LIST_NR_DATA_PAGES = 4;     /* one PRP1 + 3 list entries */
    const unsigned PRP_LIST_ENTRIES = PRP_LIST_NR_DATA_PAGES - 1;
    void* wbuf4 = alloc_aligned(PRP_LIST_NR_DATA_PAGES * info.block_size, psz);
    void* rbuf4 = alloc_aligned(PRP_LIST_NR_DATA_PAGES * info.block_size, psz);
    void* prp_list_w = alloc_aligned(info.block_size, psz);   /* one page each */
    void* prp_list_r = alloc_aligned(info.block_size, psz);
    if (!wbuf4 || !rbuf4 || !prp_list_w || !prp_list_r)
        step_fail(errno, "alloc_aligned 4-page buffers + PRP lists");

    uint64_t wbuf4_ioaddr[PRP_LIST_NR_DATA_PAGES];
    uint64_t rbuf4_ioaddr[PRP_LIST_NR_DATA_PAGES];
    uint64_t prp_list_w_ioaddr = 0;
    uint64_t prp_list_r_ioaddr = 0;
    memset(wbuf4_ioaddr, 0, sizeof(wbuf4_ioaddr));
    memset(rbuf4_ioaddr, 0, sizeof(rbuf4_ioaddr));
    {
        struct nvm_ioctl_map req;

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)wbuf4;
        req.n_pages     = PRP_LIST_NR_DATA_PAGES;
        req.ioaddrs     = (uint64_t)(uintptr_t)wbuf4_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(wbuf4 x 4)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY wbuf4");

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)rbuf4;
        req.n_pages     = PRP_LIST_NR_DATA_PAGES;
        req.ioaddrs     = (uint64_t)(uintptr_t)rbuf4_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(rbuf4 x 4)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY rbuf4");

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)prp_list_w;
        req.n_pages     = 1;
        req.ioaddrs     = (uint64_t)(uintptr_t)&prp_list_w_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(prp_list_w)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY prp_list_w");

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)prp_list_r;
        req.n_pages     = 1;
        req.ioaddrs     = (uint64_t)(uintptr_t)&prp_list_r_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(prp_list_r)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY prp_list_r");
    }

    /* Populate PRP lists.  Entries are 8-byte little-endian dma_addrs;
     * entry [0] = data page 1 (NOT page 0 -- page 0 went into PRP1).  */
    {
        uint64_t* lw = (uint64_t*)prp_list_w;
        uint64_t* lr = (uint64_t*)prp_list_r;
        for (unsigned i = 0; i < PRP_LIST_ENTRIES; i++) {
            lw[i] = wbuf4_ioaddr[i + 1];
            lr[i] = rbuf4_ioaddr[i + 1];
        }
    }
    step_ok("Phase 5c: 4-page data buffers + PRP_List pages registered "
            "(prp_list_w ioaddr=0x%llx)",
            (unsigned long long)prp_list_w_ioaddr);

    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = (uint16_t)(PRP_LIST_NR_DATA_PAGES - 1);
        const uint64_t LBA_PHASE_5C = TEST_LBA_BASE + 200;
        const unsigned NR_IO_5C = 4;
        const size_t io_bytes = PRP_LIST_NR_DATA_PAGES * info.block_size;

        struct test_queue* qw = &Q[0];
        struct test_queue* qr = &Q[1];
        char status_buf[64];

        for (unsigned i = 0; i < NR_IO_5C; i++) {
            uint64_t lba = LBA_PHASE_5C + 4u * i;       /* 4 LBA stride */
            uint8_t  pat = WRITE_PATTERN_BYTE(qw->qid, 200 + i);

            /* Stamp wbuf4 -- still mix sub-byte by 4 KiB page index so
             * any cross-page DMA misorder shows up as a single byte. */
            uint8_t* wbytes = (uint8_t*)wbuf4;
            for (size_t b = 0; b < io_bytes; b++)
                wbytes[b] = (uint8_t)(pat ^ (uint8_t)(b >> 12));

            uint16_t cid_w;
            tq_submit_rw(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP, nsid,
                         wbuf4_ioaddr[0], prp_list_w_ioaddr,
                         lba, nlb_zero_based, &cid_w);

            struct nvme_cqe cqe;
            int rc = tq_poll_one(qw, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "5c Write %u (qid=%u, lba=%" PRIu64 ") timeout",
                          i, qw->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "5c Write %u: NVMe %s", i, status_buf);
            }

            /* Read back. */
            memset(rbuf4, 0, io_bytes);
            uint16_t cid_r;
            tq_submit_rw(qr, NVME_OPC_READ, NVME_FLAG_PSDT_PRP, nsid,
                         rbuf4_ioaddr[0], prp_list_r_ioaddr,
                         lba, nlb_zero_based, &cid_r);
            rc = tq_poll_one(qr, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "5c Read %u (qid=%u, lba=%" PRIu64 ") timeout",
                          i, qr->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "5c Read %u: NVMe %s", i, status_buf);
            }

            uint8_t* rbytes = (uint8_t*)rbuf4;
            for (size_t b = 0; b < io_bytes; b++) {
                uint8_t expect = (uint8_t)(pat ^ (uint8_t)(b >> 12));
                if (rbytes[b] != expect)
                    step_fail(0, "5c IO %u byte %zu: got 0x%02x, "
                                 "expected 0x%02x (page index %zu)",
                              i, b, rbytes[b], expect, b >> 12);
            }
        }
        step_ok("Phase 5c: PRP1 + PRP List x %u IOs, %u KiB each, "
                "LBA [%" PRIu64 "..%" PRIu64 "]",
                NR_IO_5C, (unsigned)(PRP_LIST_NR_DATA_PAGES * 4),
                LBA_PHASE_5C,
                LBA_PHASE_5C + 4u * (NR_IO_5C - 1) + 3);
    }

    /* ============================================================== */
    /* Phase 5d: SGL Data Block descriptor, conditional on the         */
    /* controller advertising SGL support in Identify Controller.     */
    /*                                                                 */
    /* SGL data block descriptor (NVMe 1.4 figure 105, Type=0):        */
    /*   bytes [0..7]   = address (data buffer dma_addr)               */
    /*   bytes [8..11]  = length (transfer size in bytes, little endian)*/
    /*   bytes [12..14] = reserved (zero)                              */
    /*   byte  [15]     = type (0x0) << 4 | subtype (0x0)              */
    /*                                                                 */
    /* Submitted by setting CDW0.PSDT = 01b (NVME_FLAG_PSDT_SGL) and   */
    /* packing the 16-byte descriptor into PRP1 (low 64 bits) +        */
    /* PRP2 (high 64 bits) of the SQE.                                 */
    /*                                                                 */
    /* Reuses the 4 KiB wbuf/rbuf from Phase 5 for simplicity -- this  */
    /* phase is just verifying that PSDT=01b is honoured for a 1-page  */
    /* transfer; multi-page SGL chains are out of scope for the smoke. */
    /* ============================================================== */
    if ((info.sgl_supported & 0x3) == 0) {
        step_ok("Phase 5d: SKIP -- controller advertises SGLS=0x%x, "
                "no SGL data block support (PRP-only controller)",
                info.sgl_supported);
    } else {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 0;             /* 1 LBA per IO */
        const uint64_t LBA_PHASE_5D = TEST_LBA_BASE + 300;
        const unsigned NR_IO_5D = 8;

        struct test_queue* qw = &Q[0];
        struct test_queue* qr = &Q[1];
        char status_buf[64];

        for (unsigned i = 0; i < NR_IO_5D; i++) {
            uint64_t lba = LBA_PHASE_5D + i;
            uint8_t pat = WRITE_PATTERN_BYTE(qw->qid, 300 + i);
            memset(wbuf, pat, info.block_size);

            /* Build the SGL data block descriptor in two 64-bit halves
             * that pack into the SQE's PRP1 (dptr0) and PRP2 (dptr1)
             * fields.  Spec: low 64 = address, high 64 has length in
             * the low 32 bits and the type/subtype byte at offset
             * 15 (i.e. high 8 bits of the dptr1 word).               */
            uint64_t sgl_addr = wbuf_ioaddr;
            uint64_t sgl_meta = ((uint64_t)info.block_size & 0xffffffffu)
                              | ((uint64_t)NVME_SGL_DESC_BYTE15 << 56);

            uint16_t cid_w;
            tq_submit_rw(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_SGL, nsid,
                         sgl_addr, sgl_meta,
                         lba, nlb_zero_based, &cid_w);

            struct nvme_cqe cqe;
            int rc = tq_poll_one(qw, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "5d Write %u (qid=%u, lba=%" PRIu64 ") timeout",
                          i, qw->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "5d Write %u: NVMe %s "
                             "(controller reported SGLS=0x%x but rejected "
                             "the descriptor; check PSDT bit packing)",
                          i, status_buf, info.sgl_supported);
            }

            /* Read back via SGL too. */
            memset(rbuf, 0, info.block_size);
            sgl_addr = rbuf_ioaddr;
            sgl_meta = ((uint64_t)info.block_size & 0xffffffffu)
                     | ((uint64_t)NVME_SGL_DESC_BYTE15 << 56);
            uint16_t cid_r;
            tq_submit_rw(qr, NVME_OPC_READ, NVME_FLAG_PSDT_SGL, nsid,
                         sgl_addr, sgl_meta,
                         lba, nlb_zero_based, &cid_r);
            rc = tq_poll_one(qr, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "5d Read %u (qid=%u, lba=%" PRIu64 ") timeout",
                          i, qr->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "5d Read %u: NVMe %s", i, status_buf);
            }

            uint8_t* rbytes = (uint8_t*)rbuf;
            for (size_t b = 0; b < info.block_size; b++) {
                if (rbytes[b] != pat)
                    step_fail(0, "5d IO %u byte %zu: got 0x%02x, "
                                 "expected 0x%02x",
                              i, b, rbytes[b], pat);
            }
        }
        step_ok("Phase 5d: SGL Data Block x %u IOs, 4 KiB each, "
                "LBA [%" PRIu64 "..%" PRIu64 "] (SGLS=0x%x)",
                NR_IO_5D, LBA_PHASE_5D, LBA_PHASE_5D + NR_IO_5D - 1,
                info.sgl_supported);
    }

    /* ============================================================== */
    /* Phase 6: SQ tail wrap test on qid=37.                          */
    /*                                                                 */
    /* q_depth=64 means SQ has 64 slots (sq_tail in [0..63]).  Phases */
    /* 5/5b/5c/5d advanced sq_tail by some amount on Q[0]; this phase */
    /* issues enough additional Writes that sq_tail definitely wraps  */
    /* past q_depth-1 back into [0..]; CQ phase on Q[0] also flips.   */
    /*                                                                 */
    /* LBA base bumped to TEST_LBA_BASE+1000 to stay clear of phases  */
    /* 5/5b/5c/5d (which use [0..15], 100, 200, 300 ranges).          */
    /* ============================================================== */
    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 0;
        struct test_queue* qw = &Q[0];
        char status_buf[64];

        /* Issue (q_depth + 8) total IOs from this phase regardless of
         * what previous phases left in sq_tail -- guarantees at least
         * one full SQ wrap and one CQ phase flip.                    */
        unsigned cnt = info.q_depth + 8u;
        const uint64_t LBA_PHASE_6 = TEST_LBA_BASE + 1000;

        for (unsigned i = 0; i < cnt; i++) {
            uint64_t lba = LBA_PHASE_6 + i;
            uint8_t pat = WRITE_PATTERN_BYTE(qw->qid, 1000u + i);
            memset(wbuf, pat, info.block_size);

            uint16_t cid;
            tq_submit_rw(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP, nsid,
                         wbuf_ioaddr, /*dptr1=*/0,
                         lba, nlb_zero_based, &cid);

            struct nvme_cqe cqe;
            int rc = tq_poll_one(qw, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "wrap Write %u (qid=%u, lba=%" PRIu64 ") "
                               "CQE poll timed out",
                          i, qw->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "wrap Write %u: NVMe status %s",
                          i, status_buf);
            }
        }
        step_ok("SQ-tail-wrap stress: %u sequential 1-LBA Writes on qid=%u "
                "(sq_tail wrapped past q_depth=%u, cq_phase flipped)",
                cnt, qw->qid, info.q_depth);
    }

    /* ============================================================== */
    /* Phase 7: B6 fd-scoped DATA buffer survives group destroy.       */
    /*                                                                 */
    /* End of Phase 6 -- destroy the queue group while every data      */
    /* buffer (wbuf, rbuf, wbuf2, ..., prp_list_*) was registered      */
    /* with map_kind=NVM_MAP_KIND_DATA + group_id=0.  By design those  */
    /* are linked onto own->data_maps, NOT g->maps, so the cascade     */
    /* should drain ONLY the 4 ring maps (TEST_NR_QUEUES*2) and leave  */
    /* every DATA descriptor still pinned + DMA-mapped.                */
    /*                                                                 */
    /* Then re-create a group, re-allocate fresh SQ/CQ rings (the      */
    /* NVMe spec requires Create I/O SQ/CQ to point at fresh rings;    */
    /* nothing here re-uses the previous ring memory), re-issue        */
    /* NVM_ADD_USER_QUEUE -- and run a 4 KiB write+read+verify on the  */
    /* SAME wbuf / rbuf the previous group was using.  If anything in  */
    /* the B6 plumbing is wrong (DATA map got cascade-destroyed; the   */
    /* IOMMU mapping was torn down; the fd-scoped list lost the        */
    /* descriptor), the read either fails with -EFAULT, comes back     */
    /* with an NVMe status error, or returns the wrong bytes.          */
    /*                                                                 */
    /* Note: BAR0 is still mmap'd (we didn't munmap above), which is   */
    /* fine because the doorbell offsets the new ADD_USER_QUEUE        */
    /* returns are absolute BAR0 byte offsets -- they index the same   */
    /* mapping cleanly.                                                */
    /* ============================================================== */
    {
        /* a. Destroy the original group.  Should drain only rings.   */
        uint32_t gid_old = group_id;
        if (do_ioctl(fd_dev, NVM_DESTROY_QUEUE_GROUP, &gid_old,
                     "NVM_DESTROY_QUEUE_GROUP(B6 first destroy)") < 0)
            step_fail(errno, "NVM_DESTROY_QUEUE_GROUP B6 first destroy");
        step_ok("Phase 7a: NVM_DESTROY_QUEUE_GROUP id=%u drains only %u "
                "ring map(s) (DATA buffers stay alive on own->data_maps)",
                group_id, TEST_NR_QUEUES * 2);

        /* b. Re-create a queue group on the same fd. */
        uint32_t group_id_b6 = 0;
        {
            struct nvm_ioctl_queue_group req;
            memset(&req, 0, sizeof(req));
            if (do_ioctl(fd_dev, NVM_CREATE_QUEUE_GROUP, &req,
                         "NVM_CREATE_QUEUE_GROUP(B6)") < 0)
                step_fail(errno, "NVM_CREATE_QUEUE_GROUP B6");
            group_id_b6 = req.group_id;
        }
        step_ok("Phase 7b: NVM_CREATE_QUEUE_GROUP -> group_id=%u (post-destroy)",
                group_id_b6);

        /* c. Re-allocate fresh rings + map them with kind=RING_*. */
        struct nvme_sqe* sq_buf_b6[TEST_NR_QUEUES];
        struct nvme_cqe* cq_buf_b6[TEST_NR_QUEUES];
        for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
            size_t sq_sz = (size_t)info.q_depth * sizeof(struct nvme_sqe);
            size_t cq_sz = (size_t)info.q_depth * sizeof(struct nvme_cqe);
            sq_buf_b6[i] = (struct nvme_sqe*)alloc_aligned(sq_sz, psz);
            cq_buf_b6[i] = (struct nvme_cqe*)alloc_aligned(cq_sz, psz);
            if (!sq_buf_b6[i] || !cq_buf_b6[i])
                step_fail(errno, "B6 alloc rings %u", i);
            memset(sq_buf_b6[i], 0, sq_sz);
            memset(cq_buf_b6[i], 0, cq_sz);
        }

        for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
            uint64_t throwaway[1];
            struct nvm_ioctl_map req;

            memset(&req, 0, sizeof(req));
            req.vaddr_start = (uint64_t)(uintptr_t)sq_buf_b6[i];
            req.n_pages     = 1;
            req.ioaddrs     = (uint64_t)(uintptr_t)throwaway;
            req.ioq_idx     = -1;
            req.is_cq       = -1;
            req.group_id    = group_id_b6;
            req.map_kind    = NVM_MAP_KIND_RING_SQ;
            if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                         "NVM_MAP_HOST_MEMORY(B6 SQ)") < 0)
                step_fail(errno, "NVM_MAP_HOST_MEMORY B6 SQ %u", i);

            memset(&req, 0, sizeof(req));
            req.vaddr_start = (uint64_t)(uintptr_t)cq_buf_b6[i];
            req.n_pages     = 1;
            req.ioaddrs     = (uint64_t)(uintptr_t)throwaway;
            req.ioq_idx     = -1;
            req.is_cq       = -1;
            req.group_id    = group_id_b6;
            req.map_kind    = NVM_MAP_KIND_RING_CQ;
            if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                         "NVM_MAP_HOST_MEMORY(B6 CQ)") < 0)
                step_fail(errno, "NVM_MAP_HOST_MEMORY B6 CQ %u", i);
        }
        step_ok("Phase 7c: NVM_MAP_HOST_MEMORY x %u fresh ring(s) registered "
                "against group_id=%u; existing DATA maps untouched",
                TEST_NR_QUEUES * 2, group_id_b6);

        /* d. Issue NVM_ADD_USER_QUEUE for the fresh rings. */
        struct nvm_ioctl_add_user_queue add_req;
        memset(&add_req, 0, sizeof(add_req));
        add_req.group_id = group_id_b6;
        add_req.nr_pairs = TEST_NR_QUEUES;
        for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
            add_req.pairs[i].sq_vaddr = (uint64_t)(uintptr_t)sq_buf_b6[i];
            add_req.pairs[i].cq_vaddr = (uint64_t)(uintptr_t)cq_buf_b6[i];
        }
        if (do_ioctl(fd_dev, NVM_ADD_USER_QUEUE, &add_req,
                     "NVM_ADD_USER_QUEUE(B6)") < 0)
            step_fail(errno, "NVM_ADD_USER_QUEUE B6");
        step_ok("Phase 7d: NVM_ADD_USER_QUEUE created %u user queue(s) "
                "(qids %u..%u)", TEST_NR_QUEUES,
                add_req.out_pairs[0].qid,
                add_req.out_pairs[TEST_NR_QUEUES - 1].qid);

        /* e. Build per-queue runtime state and run a 4 KiB W+R+verify
         *    using the SAME wbuf/rbuf that survived the group destroy.   */
        struct test_queue qb6_w;
        struct test_queue qb6_r;
        memset(&qb6_w, 0, sizeof(qb6_w));
        memset(&qb6_r, 0, sizeof(qb6_r));
        qb6_w.sq      = sq_buf_b6[0];
        qb6_w.cq      = cq_buf_b6[0];
        qb6_w.q_depth = info.q_depth;
        qb6_w.qid     = (uint16_t)add_req.out_pairs[0].qid;
        qb6_w.cq_phase = 1;
        qb6_w.sq_db   = (volatile uint32_t*)
            ((uint8_t*)bar0 + add_req.out_pairs[0].sq_doorbell_offset);
        qb6_w.cq_db   = (volatile uint32_t*)
            ((uint8_t*)bar0 + add_req.out_pairs[0].cq_doorbell_offset);
        qb6_r.sq      = sq_buf_b6[1 % TEST_NR_QUEUES];
        qb6_r.cq      = cq_buf_b6[1 % TEST_NR_QUEUES];
        qb6_r.q_depth = info.q_depth;
        qb6_r.qid     = (uint16_t)add_req.out_pairs[1 % TEST_NR_QUEUES].qid;
        qb6_r.cq_phase = 1;
        qb6_r.sq_db   = (volatile uint32_t*)
            ((uint8_t*)bar0 + add_req.out_pairs[1 % TEST_NR_QUEUES].sq_doorbell_offset);
        qb6_r.cq_db   = (volatile uint32_t*)
            ((uint8_t*)bar0 + add_req.out_pairs[1 % TEST_NR_QUEUES].cq_doorbell_offset);

        /* Stamp wbuf with a B6-specific pattern so we know we're not
         * reading something the previous group left there.            */
        const uint8_t b6_pat = (uint8_t)(0x5A ^ qb6_w.qid);
        memset(wbuf, 0, info.block_size);
        for (size_t b = 0; b < info.block_size; b++)
            ((uint8_t*)wbuf)[b] = b6_pat ^ (uint8_t)(b >> 12);

        const uint64_t b6_lba = TEST_LBA_BASE + 50000;
        char status_buf[64];
        struct nvme_cqe cqe;
        uint16_t cid_w;

        tq_submit_rw(&qb6_w, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP,
                     1, wbuf_ioaddr, 0, b6_lba, 0, &cid_w);
        int rc = tq_poll_one(&qb6_w, &cqe, 5000);
        if (rc) step_fail(-rc, "B6 Write poll rc=%d", rc);
        if ((cqe.status >> 1) != 0) {
            format_status(cqe.status, status_buf, sizeof(status_buf));
            step_fail(0, "B6 Write status %s", status_buf);
        }
        if (cqe.cid != cid_w)
            step_fail(0, "B6 Write CQE.cid=%u want %u", cqe.cid, cid_w);

        memset(rbuf, 0, info.block_size);
        uint16_t cid_r;
        tq_submit_rw(&qb6_r, NVME_OPC_READ, NVME_FLAG_PSDT_PRP,
                     1, rbuf_ioaddr, 0, b6_lba, 0, &cid_r);
        rc = tq_poll_one(&qb6_r, &cqe, 5000);
        if (rc) step_fail(-rc, "B6 Read poll rc=%d", rc);
        if ((cqe.status >> 1) != 0) {
            format_status(cqe.status, status_buf, sizeof(status_buf));
            step_fail(0, "B6 Read status %s", status_buf);
        }
        if (cqe.cid != cid_r)
            step_fail(0, "B6 Read CQE.cid=%u want %u", cqe.cid, cid_r);

        for (size_t b = 0; b < info.block_size; b++) {
            uint8_t want = b6_pat ^ (uint8_t)(b >> 12);
            if (((uint8_t*)rbuf)[b] != want)
                step_fail(0, "B6 readback mismatch at byte %zu: "
                             "got 0x%02x want 0x%02x (lba=%" PRIu64 ")",
                          b, ((uint8_t*)rbuf)[b], want, b6_lba);
        }
        step_ok("Phase 7e: 4 KiB W+R+verify on RECYCLED data buffers via "
                "post-destroy group (lba=%" PRIu64 ", qid_w=%u qid_r=%u)",
                b6_lba, qb6_w.qid, qb6_r.qid);

        /* f. Hand the fresh group back to Phase 8's destroy by
         *    overwriting group_id; Phase 8 will issue ONE final
         *    NVM_DESTROY_QUEUE_GROUP that picks up these new rings + 0
         *    data maps (DATA maps are already on own->data_maps and
         *    will be reaped at fd close instead).
         *
         *    The OLD rings (the sq_buf[i]/cq_buf[i] from Phase 2) are
         *    now orphaned at the user-space level: the kernel-side
         *    pinning was released by the Phase 7a DESTROY_QUEUE_GROUP,
         *    but the malloc'd memory itself is still ours to free.
         *    Drop it now before we lose the pointers.                 */
        for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
            free(sq_buf[i]);
            free(cq_buf[i]);
            sq_buf[i] = sq_buf_b6[i];
            cq_buf[i] = cq_buf_b6[i];
        }
        group_id = group_id_b6;
    }

    /* ============================================================== */
    /* Phase 8: Tear-down.                                            */
    /*                                                                 */
    /* munmap BAR0 first so the doorbell pointers in test_queue are    */
    /* invalidated before destroy_qgroup -- destroy_qgroup may issue  */
    /* admin commands that the kernel handles internally; we don't    */
    /* want to accidentally write a stale tail through the same       */
    /* pointer.  Then DESTROY_QUEUE_GROUP cascades through the        */
    /* Phase-7 fresh queue group's user queues + 4 ring maps; the     */
    /* DATA maps (wbuf/rbuf/wbuf2/rbuf2/wbuf4/rbuf4/prp_list_*) are    */
    /* released a moment later by close(fd_dev) via the per-fd        */
    /* data_maps cleanup in snvm_dev_release.                         */
    /* ============================================================== */
    if (munmap(bar0, info.bar0_size) < 0)
        step_fail(errno, "munmap BAR0");
    step_ok("munmap BAR0");

    {
        uint32_t gid = group_id;
        if (do_ioctl(fd_dev, NVM_DESTROY_QUEUE_GROUP, &gid,
                     "NVM_DESTROY_QUEUE_GROUP") < 0)
            step_fail(errno, "NVM_DESTROY_QUEUE_GROUP");
        /* Phase 7 only re-mapped 4 fresh ring buffers, all DATA maps
         * still live on own->data_maps and will be reaped at close.   */
        step_ok("NVM_DESTROY_QUEUE_GROUP id=%u cascades through %u user "
                "queue(s) + %u ring map(s) (8 DATA maps stay)",
                group_id, TEST_NR_QUEUES, TEST_NR_QUEUES * 2);
    }

    /* Free user-side ring + data buffers (the snvme-side maps are
     * already drained by NVM_DESTROY_QUEUE_GROUP).                    */
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        free(sq_buf[i]);
        free(cq_buf[i]);
    }
    free(wbuf);
    free(rbuf);
    free(wbuf2);
    free(rbuf2);
    free(wbuf4);
    free(rbuf4);
    free(prp_list_w);
    free(prp_list_r);

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

    fprintf(stderr, "\n=== snvme_smoke_io: all %d steps passed ===\n", g_step);
    return 0;
}

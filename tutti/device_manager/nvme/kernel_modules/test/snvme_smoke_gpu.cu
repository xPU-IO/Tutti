/*
 * snvme_smoke_gpu.cu -- GPU end-to-end NVMe IO smoke test on B3 user
 * IO queues, with rings AND data buffers placed in GPU memory and
 * SQE submission / CQE polling driven from CUDA kernels.
 *
 * Mirrors snvme_smoke_io.c (same Phase numbering and Tier coverage)
 * but with the GPU-resident counterparts:
 *
 *   * SQ / CQ rings: cudaMalloc'd, registered with snvme via
 *     NVM_MAP_DEVICE_MEMORY (group-scoped, B2 path).  64 KiB
 *     GPU-page alignment is enforced by allocating one full GPU page
 *     per ring -- the q_depth=64 SQ (4 KiB) and CQ (1 KiB) only fill
 *     the first NVMe page of that 64 KiB allocation; the rest is
 *     unused but kept registered so the controller's PRP1 lookup
 *     stays trivial.
 *
 *   * Data buffers: same.  The 4 KiB / 8 KiB / 16 KiB tier transfers
 *     all share a single 64 KiB GPU allocation per direction, sliced
 *     by NVMe-page (4 KiB) offset.  Lets the same allocation cover
 *     PRP1, PRP1+PRP2, and PRP1+PRP_List without re-registration.
 *
 *   * Doorbells: BAR0 is mmap()d on the CPU side, then registered
 *     with cudaHostRegister(IoMemory) and translated to a GPU device
 *     pointer via cudaHostGetDevicePointer.  Submission CUDA kernels
 *     ring the SQ doorbell via a volatile uint32_t store directly
 *     from the GPU.
 *
 *   * SQE submission: built on the GPU using a one-thread CUDA kernel
 *     that fills the next slot in the GPU-resident SQ ring, then
 *     issues __threadfence_system() and rings the doorbell.
 *
 *   * CQE polling: GPU kernel spins on the phase bit of the next CQ
 *     slot, then writes the CQ head doorbell to release credit back
 *     to the controller.  Polling is bounded by an iteration counter
 *     (similar to the CPU smoke) so a misbehaving controller times
 *     out instead of hanging.
 *
 *   * DYNAMIC ALLOC/FREE LOOP: the entire data-plane (queue group +
 *     user IO queues + GPU rings + GPU data buffers + GPU PRP_Lists)
 *     is built up and torn down N times in a row (--rounds N,
 *     default 4).  Between rounds we cudaFree the GPU allocations
 *     and call NVM_DESTROY_QUEUE_GROUP, which forces the controller
 *     to execute Delete I/O SQ + Delete I/O CQ for every user queue
 *     and snvme to release every NVM_MAP_DEVICE_MEMORY descriptor.
 *     The next round re-creates everything from scratch.  This
 *     verifies:
 *       1. snvme's user QID pool is reclaimed correctly across
 *          DESTROY_QUEUE_GROUP cycles (no leak after N rounds).
 *       2. The controller accepts Create I/O SQ/CQ a second/third/...
 *          time on the same controller bind without misbehaving.
 *       3. GPU rings registered via NVM_MAP_DEVICE_MEMORY can be
 *          allocated and released repeatedly without breaking the
 *          NVIDIA p2p get_pages/put_pages refcount.
 *
 * Pre-conditions:
 *   - snvme-core.ko + snvme.ko loaded.
 *   - The proprietary NVIDIA driver loaded AND nvfs_nvidia_p2p_init()
 *     succeeded at module-load time (snvme refuses to load otherwise).
 *   - At least one GPU visible to CUDA.
 *   - Root (PCI bind requires CAP_SYS_ADMIN).
 *
 * DESTRUCTIVE: writes to LBAs starting at TEST_LBA_BASE (default
 * 2621440 = 10 GiB / 4 KiB).  Each round uses a non-overlapping LBA
 * window so the verifier can run independently per round.
 *
 * Build:        make snvme_smoke_gpu
 * Invoke:       sudo ./snvme_smoke_gpu [--gpu N] [--rounds N] <PCI_BDF>
 *
 * Exit codes:
 *   0  -- all steps passed.
 *   1  -- usage error.
 *   2  -- a smoke step failed; see stderr for which one.
 */

#include <tutti/cuda_like.h>

#include <cerrno>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" {
#include "ioctl.h"
}

/* ------------------------------------------------------------------ */
/* NVMe spec constants.  Same definitions as snvme_smoke_io.c.        */
/* ------------------------------------------------------------------ */

#define NVME_OPC_WRITE              0x01u
#define NVME_OPC_READ               0x02u

#define NVME_SQE_SIZE               64u
#define NVME_CQE_SIZE               16u

#define TEST_LBA_BASE               2621440ULL   /* 10 GiB / 4 KiB */
#define TEST_NR_QUEUES              2u
#define TEST_NR_IO_PER_QUEUE        16u
#define TEST_DEFAULT_ROUNDS         4u
/* Per-round LBA window = 64 Ki LBAs = 256 MiB of 4 KiB sectors.
 * Rounds 0..N-1 occupy disjoint windows starting at TEST_LBA_BASE,
 * so a per-round verify never collides with another round.        */
#define TEST_LBA_PER_ROUND          0x10000ULL

/* GPU page size used by the kernel module
 * (snvme/map.c:GPU_PAGE_SHIFT=16).  Buffers passed to NVM_MAP_DEVICE_*
 * MUST be aligned to and sized in multiples of this.                  */
static constexpr size_t GPU_PAGE_SIZE = 1ULL << 16;     /* 64 KiB */

/* CDW0 PSDT bits (CDW0[15:14], appearing as bits [7:6] of the SQE
 * 'flags' byte).  PRP=00b, SGL data block=01b.                       */
#define NVME_FLAG_PSDT_PRP          (0u << 6)
#define NVME_FLAG_PSDT_SGL          (1u << 6)

/* SGL Data Block descriptor type|subtype byte.                        */
#define NVME_SGL_DESC_BYTE15        0x00u

/* Per-byte pattern is a function of (round, qid, ioidx) so a
 * cross-round byte mismatch is unambiguous in the failure log.    */
#define WRITE_PATTERN_BYTE(round, qid, ioidx) \
    ((uint8_t)(0xA5 ^ ((round) & 0xff) ^ ((qid) & 0xff) ^ ((ioidx) & 0xff)))

/* ------------------------------------------------------------------ */
/* Submission queue entry (Common Format, NVMe 1.4 figure 105).       */
/* Defined identically on host and device so __device__ kernels can   */
/* fill the same struct that the controller will then read via DMA.   */
/* ------------------------------------------------------------------ */

struct nvme_sqe {
    uint8_t  opcode;
    uint8_t  flags;
    uint16_t cid;
    uint32_t nsid;
    uint64_t rsvd_2_3;
    uint64_t metadata;
    uint64_t prp1;
    uint64_t prp2;
    uint32_t cdw10;
    uint32_t cdw11;
    uint32_t cdw12;
    uint32_t cdw13;
    uint32_t cdw14;
    uint32_t cdw15;
} __attribute__((packed));

static_assert(sizeof(nvme_sqe) == NVME_SQE_SIZE,
              "nvme_sqe must be exactly 64 bytes");

struct nvme_cqe {
    uint32_t result;
    uint32_t rsvd;
    uint16_t sq_head;
    uint16_t sq_id;
    uint16_t cid;
    uint16_t status;
} __attribute__((packed));

static_assert(sizeof(nvme_cqe) == NVME_CQE_SIZE,
              "nvme_cqe must be exactly 16 bytes");

/* ------------------------------------------------------------------ */
/* Logging helpers (host-side).                                       */
/* ------------------------------------------------------------------ */

static int g_step = 0;

static void step_ok(const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[ OK ] step=%-3d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void __attribute__((noreturn)) step_fail(int err, const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[FAIL] step=%-3d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, " errno=%d (%s)\n", err, err ? strerror(err) : "n/a");
    exit(2);
}

#define CUDA_OK(call_)                                                  \
    do {                                                                \
        cudaError_t _e = (call_);                                       \
        if (_e != cudaSuccess) {                                        \
            step_fail(0, "CUDA: %s -> %s", #call_, cudaGetErrorString(_e)); \
        }                                                               \
    } while (0)

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
        "Usage: %s [--gpu N] [--rounds N] <PCI_BDF>\n"
        "  e.g.: %s --gpu 0 --rounds 4 0000:08:00.0\n"
        "\n"
        "BINDS the target controller and writes/reads via GPU-resident\n"
        "rings + GPU-resident data buffers, repeated --rounds times\n"
        "(default %u) with full queue/buffer alloc-free between rounds.\n"
        "DESTRUCTIVE.\n",
        prog, prog, TEST_DEFAULT_ROUNDS);
}

static void format_status(uint16_t status, char* buf, size_t cap) {
    uint16_t s   = status >> 1;
    uint8_t  sc  = s & 0xff;
    uint8_t  sct = (s >> 8) & 0x7;
    snprintf(buf, cap, "0x%04x (SC=0x%02x SCT=0x%x)", status, sc, sct);
}

/* ------------------------------------------------------------------ */
/* Per-queue runtime state.  Lives partly in host memory (the         */
/* `*_dev` device pointers) and partly on the GPU (the rings + the    */
/* doorbell GPU VA).  Submission/poll kernels receive a copy of the   */
/* whole struct by value.                                             */
/* ------------------------------------------------------------------ */

struct test_queue_dev {
    nvme_sqe*           sq;             /* device VA */
    nvme_cqe*           cq;             /* device VA */
    volatile uint32_t*  sq_db;          /* GPU VA into BAR0 */
    volatile uint32_t*  cq_db;          /* GPU VA into BAR0 */
    uint16_t            q_depth;
    uint16_t            qid;
};

/* ------------------------------------------------------------------ */
/* GPU kernels for SQE submit / CQE poll / data fill / data verify.   */
/* ------------------------------------------------------------------ */

__global__ void k_submit_rw(test_queue_dev qd,
                            uint16_t* sq_tail_io,
                            uint16_t cid,
                            uint8_t opcode,
                            uint8_t flags,
                            uint32_t nsid,
                            uint64_t dptr0,
                            uint64_t dptr1,
                            uint64_t slba,
                            uint16_t nlb_zero_based) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint16_t tail = *sq_tail_io;
    nvme_sqe* slot = &qd.sq[tail];

    /* Zero the 64-byte slot through the same path the controller will
     * see (memset_d-style writes via CUDA's volatile semantics).     */
    uint8_t* p = (uint8_t*)slot;
    #pragma unroll
    for (int i = 0; i < (int)sizeof(nvme_sqe); i++) p[i] = 0;

    slot->opcode = opcode;
    slot->flags  = flags;
    slot->cid    = cid;
    slot->nsid   = nsid;
    slot->prp1   = dptr0;
    slot->prp2   = dptr1;
    slot->cdw10  = (uint32_t)(slba & 0xffffffffu);
    slot->cdw11  = (uint32_t)(slba >> 32);
    slot->cdw12  = (uint32_t)(nlb_zero_based & 0xffffu);

    /* Make the SQE bytes visible to the device DMA engine BEFORE we
     * ring the doorbell.                                            */
    __threadfence_system();

    uint16_t new_tail = (uint16_t)((tail + 1) % qd.q_depth);
    *qd.sq_db = new_tail;
    *sq_tail_io = new_tail;
}

/* Polls until either a CQE with the expected phase bit appears, or
 * `max_iters` iterations elapse without one (reported as
 * timed_out=1).  On success, copies the CQE out, advances cq_head,
 * flips cq_phase if the head wraps, and rings the CQ head doorbell.   */
__global__ void k_poll_one(test_queue_dev qd,
                           uint16_t* cq_head_io,
                           uint8_t* cq_phase_io,
                           nvme_cqe* out_cqe,
                           int* timed_out,
                           uint64_t max_iters) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint16_t head = *cq_head_io;
    uint8_t  expected = *cq_phase_io;

    uint64_t i = 0;
    for (;;) {
        volatile nvme_cqe* slot = &qd.cq[head];
        uint16_t status = slot->status;
        uint8_t phase = status & 0x1u;
        if (phase == expected) {
            /* Copy out the full 16-byte CQE before advancing. */
            nvme_cqe tmp;
            tmp.result  = slot->result;
            tmp.rsvd    = slot->rsvd;
            tmp.sq_head = slot->sq_head;
            tmp.sq_id   = slot->sq_id;
            tmp.cid     = slot->cid;
            tmp.status  = status;
            *out_cqe = tmp;

            uint16_t new_head = (uint16_t)((head + 1) % qd.q_depth);
            if (new_head == 0) expected ^= 1u;

            __threadfence_system();
            *qd.cq_db = new_head;
            *cq_head_io  = new_head;
            *cq_phase_io = expected;
            *timed_out = 0;
            return;
        }
        if (++i >= max_iters) {
            *timed_out = 1;
            return;
        }
    }
}

/* ------------------------------------------------------------------ */
/* GPU helpers: fill a buffer with a per-byte pattern; verify ditto.  */
/* ------------------------------------------------------------------ */

__global__ void k_fill_pattern(uint8_t* buf, size_t bytes, uint8_t pat) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bytes) return;
    buf[idx] = pat ^ (uint8_t)(idx >> 12);
}

__global__ void k_verify_pattern(const uint8_t* buf, size_t bytes,
                                 uint8_t pat, int* mismatch_idx) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bytes) return;
    uint8_t expect = pat ^ (uint8_t)(idx >> 12);
    if (buf[idx] != expect) {
        /* Race-tolerant: any thread can win, the first idx written
         * wins for the host-side error report.  We just need ANY
         * mismatching byte.                                          */
        atomicCAS(mismatch_idx, -1, (int)idx);
    }
}

/* ------------------------------------------------------------------ */
/* Host helpers around the submit / poll kernels.                     */
/* ------------------------------------------------------------------ */

struct queue_state {
    test_queue_dev      dev;
    /* host-side counters; we keep these in unified-pinned memory so
     * the kernels can read+update them in place without a roundtrip */
    uint16_t*           sq_tail_um;     /* unified memory, 1 element */
    uint16_t*           cq_head_um;     /* ditto */
    uint8_t*            cq_phase_um;    /* ditto */
    nvme_cqe*           out_cqe_um;     /* unified memory, 1 element */
    int*                timed_out_um;   /* ditto */
    uint16_t            next_cid;
};

static int submit_and_poll(queue_state& qs,
                           uint8_t opcode, uint8_t flags,
                           uint32_t nsid,
                           uint64_t dptr0, uint64_t dptr1,
                           uint64_t slba, uint16_t nlb_zero_based,
                           nvme_cqe* cqe_out, uint16_t* cid_out) {
    uint16_t cid = qs.next_cid++;
    if (cid_out) *cid_out = cid;

    k_submit_rw<<<1, 1>>>(qs.dev, qs.sq_tail_um, cid,
                          opcode, flags, nsid, dptr0, dptr1,
                          slba, nlb_zero_based);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "k_submit_rw launch: %s\n", cudaGetErrorString(e));
        return -EIO;
    }
    /* No explicit cudaDeviceSynchronize between submit and poll --
     * poll kernel will wait for SQE visibility implicitly through
     * the doorbell write.  But submit has to actually complete
     * before poll runs; the default stream serialises this. */

    /* 5-second budget at ~10 ns per iteration -> 5e8 iters; be
     * generous since GPU is slower than CPU on a tight spin loop. */
    constexpr uint64_t MAX_ITERS = 50000000ULL;
    k_poll_one<<<1, 1>>>(qs.dev, qs.cq_head_um, qs.cq_phase_um,
                         qs.out_cqe_um, qs.timed_out_um, MAX_ITERS);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "k_poll_one launch: %s\n", cudaGetErrorString(e));
        return -EIO;
    }
    CUDA_OK(cudaDeviceSynchronize());

    if (*qs.timed_out_um) return -ETIMEDOUT;
    *cqe_out = *qs.out_cqe_um;
    return 0;
}

/* ------------------------------------------------------------------ */
/* All per-round resources, allocated/freed by run_one_round().       */
/* B6: only the queue group + the SQ/CQ rings are per-round.  Data    */
/* buffers (wbuf / rbuf / prp_list_*) live in persistent_data_res     */
/* below and span every round on this fd.                             */
/* ------------------------------------------------------------------ */

struct round_resources {
    uint32_t    group_id;
    void*       sq_dev[TEST_NR_QUEUES];
    void*       cq_dev[TEST_NR_QUEUES];
    uint64_t    sq_ioaddr[TEST_NR_QUEUES];
    uint64_t    cq_ioaddr[TEST_NR_QUEUES];
    queue_state QS[TEST_NR_QUEUES];
};

/* ------------------------------------------------------------------ */
/* B6: persistent fd-scoped data resources.  Allocated once in main(),*/
/* registered with map_kind=NVM_MAP_KIND_DATA + group_id=0 so the     */
/* kernel parks them on snvm_dev_owner.data_maps and they survive    */
/* every NVM_DESTROY_QUEUE_GROUP cascade.  Released on close(fd_dev). */
/* ------------------------------------------------------------------ */

struct persistent_data_resources {
    void*       wbuf_dev;
    void*       rbuf_dev;
    void*       prp_list_w_dev;
    void*       prp_list_r_dev;
    uint64_t    wbuf_ioaddr;
    uint64_t    rbuf_ioaddr;
    uint64_t    prp_list_w_ioaddr;
    uint64_t    prp_list_r_ioaddr;
};

/* Run all the per-round IO phases (formerly Phase 2..10).  Caller
 * supplies a fresh `rr` (zeroed) and the controller-wide state
 * (fd_dev, info, bar0_gpu).  Returns 0 on success, exits on error.
 *
 * `round_idx` selects a unique LBA window plus the per-byte
 * pattern stripe.  `kernel_ioq_cap` drives NVM_SET_KERNEL_IOQ_CAP
 * for the very first round only -- subsequent rounds reuse the cap
 * already negotiated with the controller (snvme keeps it across
 * group destroy/create cycles within a single bind).             */
static void run_one_round(int fd_dev,
                          const struct nvm_ioctl_dev& info,
                          void* bar0_gpu,
                          unsigned round_idx,
                          round_resources& rr,
                          const persistent_data_resources& pdata) {
    /* ============================================================== */
    /* Phase R.1: queue group.                                        */
    /* ============================================================== */
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_dev, NVM_CREATE_QUEUE_GROUP, &req,
                     "NVM_CREATE_QUEUE_GROUP") < 0)
            step_fail(errno, "round=%u NVM_CREATE_QUEUE_GROUP", round_idx);
        rr.group_id = req.group_id;
        step_ok("round=%u NVM_CREATE_QUEUE_GROUP -> group_id=%u max_queues=%u",
                round_idx, rr.group_id, req.max_queues);
    }

    /* ============================================================== */
    /* Phase R.2: cudaMalloc rings only.  The data + PRP_List buffers */
    /* live in `pdata` and were registered ONCE in main() with        */
    /* map_kind=DATA + group_id=0; they outlive every per-round       */
    /* group and are reaped only at fd close.                         */
    /* ============================================================== */
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        CUDA_OK(cudaMalloc(&rr.sq_dev[i], GPU_PAGE_SIZE));
        CUDA_OK(cudaMalloc(&rr.cq_dev[i], GPU_PAGE_SIZE));
        CUDA_OK(cudaMemset(rr.sq_dev[i], 0, GPU_PAGE_SIZE));
        CUDA_OK(cudaMemset(rr.cq_dev[i], 0, GPU_PAGE_SIZE));
    }
    step_ok("round=%u cudaMalloc'd %u SQ + %u CQ GPU pages "
            "(%zu B each); reusing persistent wbuf/rbuf/prp_list",
            round_idx, TEST_NR_QUEUES, TEST_NR_QUEUES, GPU_PAGE_SIZE);

    /* ============================================================== */
    /* Phase R.3: NVM_MAP_DEVICE_MEMORY for the rings only, with      */
    /* explicit kind tags so NVM_ADD_USER_QUEUE will accept them and  */
    /* refuse any data-buffer vaddr in the same slot.                 */
    /* ============================================================== */
    auto map_ring = [&](void* gpu_va, unsigned long n_pages,
                        uint64_t* ioaddrs_out, uint8_t kind,
                        const char* what) {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)gpu_va;
        req.n_pages     = n_pages;
        req.ioaddrs     = (uint64_t)(uintptr_t)ioaddrs_out;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = rr.group_id;
        req.map_kind    = kind;
        if (do_ioctl(fd_dev, NVM_MAP_DEVICE_MEMORY, &req, what) < 0)
            step_fail(errno, "round=%u %s gpu_va=%p", round_idx, what, gpu_va);
    };

    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        map_ring(rr.sq_dev[i], 1, &rr.sq_ioaddr[i],
                 NVM_MAP_KIND_RING_SQ,
                 "NVM_MAP_DEVICE_MEMORY(SQ)");
        map_ring(rr.cq_dev[i], 1, &rr.cq_ioaddr[i],
                 NVM_MAP_KIND_RING_CQ,
                 "NVM_MAP_DEVICE_MEMORY(CQ)");
    }
    step_ok("round=%u NVM_MAP_DEVICE_MEMORY x %u ring(s) (RING_SQ/RING_CQ); "
            "data buffers (wbuf_ioaddr=0x%llx) come from persistent pool",
            round_idx, TEST_NR_QUEUES * 2,
            (unsigned long long)pdata.wbuf_ioaddr);

    /* ============================================================== */
    /* Phase R.4: NVM_ADD_USER_QUEUE.                                */
    /* ============================================================== */
    struct nvm_ioctl_add_user_queue add_req;
    memset(&add_req, 0, sizeof(add_req));
    add_req.group_id = rr.group_id;
    add_req.nr_pairs = TEST_NR_QUEUES;
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        add_req.pairs[i].sq_vaddr = (uint64_t)(uintptr_t)rr.sq_dev[i];
        add_req.pairs[i].cq_vaddr = (uint64_t)(uintptr_t)rr.cq_dev[i];
    }
    if (do_ioctl(fd_dev, NVM_ADD_USER_QUEUE, &add_req,
                 "NVM_ADD_USER_QUEUE") < 0)
        step_fail(errno, "round=%u NVM_ADD_USER_QUEUE", round_idx);
    step_ok("round=%u NVM_ADD_USER_QUEUE created %u user queue(s)",
            round_idx, TEST_NR_QUEUES);
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++)
        fprintf(stderr, "                pair[%u] qid=%u sq_db=0x%x cq_db=0x%x\n",
                i, add_req.out_pairs[i].qid,
                add_req.out_pairs[i].sq_doorbell_offset,
                add_req.out_pairs[i].cq_doorbell_offset);

    /* Build per-queue test_queue_dev structs (consumed by kernels).  */
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        rr.QS[i].dev.sq      = (nvme_sqe*)rr.sq_dev[i];
        rr.QS[i].dev.cq      = (nvme_cqe*)rr.cq_dev[i];
        rr.QS[i].dev.q_depth = info.q_depth;
        rr.QS[i].dev.qid     = (uint16_t)add_req.out_pairs[i].qid;
        rr.QS[i].dev.sq_db   = (volatile uint32_t*)
            ((char*)bar0_gpu + add_req.out_pairs[i].sq_doorbell_offset);
        rr.QS[i].dev.cq_db   = (volatile uint32_t*)
            ((char*)bar0_gpu + add_req.out_pairs[i].cq_doorbell_offset);

        /* Unified-memory counters / out-cqe so submit/poll kernels can
         * read+update them in place; avoids host<->device copies on
         * every IO.                                                  */
        CUDA_OK(cudaMallocManaged(&rr.QS[i].sq_tail_um,   sizeof(uint16_t)));
        CUDA_OK(cudaMallocManaged(&rr.QS[i].cq_head_um,   sizeof(uint16_t)));
        CUDA_OK(cudaMallocManaged(&rr.QS[i].cq_phase_um,  sizeof(uint8_t)));
        CUDA_OK(cudaMallocManaged(&rr.QS[i].out_cqe_um,   sizeof(nvme_cqe)));
        CUDA_OK(cudaMallocManaged(&rr.QS[i].timed_out_um, sizeof(int)));
        *rr.QS[i].sq_tail_um   = 0;
        *rr.QS[i].cq_head_um   = 0;
        *rr.QS[i].cq_phase_um  = 1;
        *rr.QS[i].timed_out_um = 0;
        rr.QS[i].next_cid = 0;
    }
    step_ok("round=%u per-queue device state ready (%u queues)",
            round_idx, TEST_NR_QUEUES);

    /* The data buffer's dma_addr lets us derive 4-KiB-grained NVMe
     * page addresses for tier 2/3.                                  */
    auto wpage = [&](unsigned npage) -> uint64_t {
        return pdata.wbuf_ioaddr + (uint64_t)npage * info.block_size;
    };
    auto rpage = [&](unsigned npage) -> uint64_t {
        return pdata.rbuf_ioaddr + (uint64_t)npage * info.block_size;
    };

    /* This round's LBA window starts here.  Each tier carves out a
     * disjoint sub-range.                                            */
    const uint64_t round_lba_base =
        TEST_LBA_BASE + (uint64_t)round_idx * TEST_LBA_PER_ROUND;

    /* ============================================================== */
    /* Phase R.5: Tier 1 -- 4 KiB IO, PRP1 only.                     */
    /* ============================================================== */
    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 0;
        const size_t io_bytes = info.block_size;
        queue_state& qw = rr.QS[0];
        queue_state& qr = rr.QS[1];
        char status_buf[64];

        int* mismatch_um = nullptr;
        CUDA_OK(cudaMallocManaged(&mismatch_um, sizeof(int)));

        for (unsigned i = 0; i < TEST_NR_IO_PER_QUEUE; i++) {
            uint64_t lba = round_lba_base + i;
            uint8_t pat  = WRITE_PATTERN_BYTE(round_idx, qw.dev.qid, i);

            int threads = 256, blocks = (int)((io_bytes + threads - 1) / threads);
            k_fill_pattern<<<blocks, threads>>>((uint8_t*)pdata.wbuf_dev,
                                                io_bytes, pat);
            CUDA_OK(cudaDeviceSynchronize());

            nvme_cqe cqe;
            uint16_t cid_w;
            int rc = submit_and_poll(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP,
                                     nsid, wpage(0), 0,
                                     lba, nlb_zero_based, &cqe, &cid_w);
            if (rc) step_fail(-rc, "round=%u T1 Write %u (qid=%u, lba=%" PRIu64 ")",
                              round_idx, i, qw.dev.qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T1 Write %u: NVMe %s",
                          round_idx, i, status_buf);
            }
            if (cqe.cid != cid_w)
                step_fail(0, "round=%u T1 Write %u CQE.cid=%u expected %u",
                          round_idx, i, cqe.cid, cid_w);

            CUDA_OK(cudaMemset(pdata.rbuf_dev, 0, io_bytes));
            uint16_t cid_r;
            rc = submit_and_poll(qr, NVME_OPC_READ, NVME_FLAG_PSDT_PRP,
                                 nsid, rpage(0), 0,
                                 lba, nlb_zero_based, &cqe, &cid_r);
            if (rc) step_fail(-rc, "round=%u T1 Read %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T1 Read %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            *mismatch_um = -1;
            k_verify_pattern<<<blocks, threads>>>((const uint8_t*)pdata.rbuf_dev,
                                                   io_bytes, pat, mismatch_um);
            CUDA_OK(cudaDeviceSynchronize());
            if (*mismatch_um != -1)
                step_fail(0, "round=%u T1 IO %u: byte %d mismatch (lba=%"
                          PRIu64 ")", round_idx, i, *mismatch_um, lba);
        }
        cudaFree(mismatch_um);
        step_ok("round=%u Tier 1 (PRP1, 4 KiB) write+verify x %u IOs, "
                "LBA [%" PRIu64 "..%" PRIu64 "]",
                round_idx, TEST_NR_IO_PER_QUEUE,
                round_lba_base,
                round_lba_base + TEST_NR_IO_PER_QUEUE - 1);
    }

    /* ============================================================== */
    /* Phase R.6: Tier 2 -- 8 KiB IO, PRP1 + PRP2.                   */
    /* ============================================================== */
    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 1;          /* 2 LBAs per IO */
        const size_t   io_bytes = 2 * info.block_size;
        const uint64_t LBA_BASE = round_lba_base + 100;
        const unsigned NR = 8;
        queue_state& qw = rr.QS[0];
        queue_state& qr = rr.QS[1];
        char status_buf[64];

        int* mismatch_um = nullptr;
        CUDA_OK(cudaMallocManaged(&mismatch_um, sizeof(int)));

        for (unsigned i = 0; i < NR; i++) {
            uint64_t lba = LBA_BASE + 2u * i;
            uint8_t  pat = WRITE_PATTERN_BYTE(round_idx, qw.dev.qid, 100 + i);

            int threads = 256, blocks = (int)((io_bytes + threads - 1) / threads);
            k_fill_pattern<<<blocks, threads>>>((uint8_t*)pdata.wbuf_dev,
                                                io_bytes, pat);
            CUDA_OK(cudaDeviceSynchronize());

            nvme_cqe cqe;
            uint16_t cid_w;
            int rc = submit_and_poll(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP,
                                     nsid, wpage(0), wpage(1),
                                     lba, nlb_zero_based, &cqe, &cid_w);
            if (rc) step_fail(-rc, "round=%u T2 Write %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T2 Write %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            CUDA_OK(cudaMemset(pdata.rbuf_dev, 0, io_bytes));
            uint16_t cid_r;
            rc = submit_and_poll(qr, NVME_OPC_READ, NVME_FLAG_PSDT_PRP,
                                 nsid, rpage(0), rpage(1),
                                 lba, nlb_zero_based, &cqe, &cid_r);
            if (rc) step_fail(-rc, "round=%u T2 Read %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T2 Read %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            *mismatch_um = -1;
            k_verify_pattern<<<blocks, threads>>>((const uint8_t*)pdata.rbuf_dev,
                                                   io_bytes, pat, mismatch_um);
            CUDA_OK(cudaDeviceSynchronize());
            if (*mismatch_um != -1)
                step_fail(0, "round=%u T2 IO %u: byte %d mismatch",
                          round_idx, i, *mismatch_um);
        }
        cudaFree(mismatch_um);
        step_ok("round=%u Tier 2 (PRP1+PRP2, 8 KiB) x %u IOs, LBA [%"
                PRIu64 "..%" PRIu64 "]",
                round_idx, NR, LBA_BASE, LBA_BASE + 2u * (NR - 1) + 1);
    }

    /* ============================================================== */
    /* Phase R.7: Tier 3 -- 16 KiB IO, PRP1 + PRP_List.              */
    /* ============================================================== */
    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 3;          /* 4 LBAs per IO */
        const size_t   io_bytes = 4 * info.block_size;
        const uint64_t LBA_BASE = round_lba_base + 200;
        const unsigned NR = 4;
        queue_state& qw = rr.QS[0];
        queue_state& qr = rr.QS[1];
        char status_buf[64];

        uint64_t prp_w_entries[3] = { wpage(1), wpage(2), wpage(3) };
        uint64_t prp_r_entries[3] = { rpage(1), rpage(2), rpage(3) };
        CUDA_OK(cudaMemcpy(pdata.prp_list_w_dev, prp_w_entries,
                           sizeof(prp_w_entries), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(pdata.prp_list_r_dev, prp_r_entries,
                           sizeof(prp_r_entries), cudaMemcpyHostToDevice));

        int* mismatch_um = nullptr;
        CUDA_OK(cudaMallocManaged(&mismatch_um, sizeof(int)));

        for (unsigned i = 0; i < NR; i++) {
            uint64_t lba = LBA_BASE + 4u * i;
            uint8_t  pat = WRITE_PATTERN_BYTE(round_idx, qw.dev.qid, 200 + i);

            int threads = 256, blocks = (int)((io_bytes + threads - 1) / threads);
            k_fill_pattern<<<blocks, threads>>>((uint8_t*)pdata.wbuf_dev,
                                                io_bytes, pat);
            CUDA_OK(cudaDeviceSynchronize());

            nvme_cqe cqe;
            uint16_t cid_w;
            int rc = submit_and_poll(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP,
                                     nsid, wpage(0), pdata.prp_list_w_ioaddr,
                                     lba, nlb_zero_based, &cqe, &cid_w);
            if (rc) step_fail(-rc, "round=%u T3 Write %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T3 Write %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            CUDA_OK(cudaMemset(pdata.rbuf_dev, 0, io_bytes));
            uint16_t cid_r;
            rc = submit_and_poll(qr, NVME_OPC_READ, NVME_FLAG_PSDT_PRP,
                                 nsid, rpage(0), pdata.prp_list_r_ioaddr,
                                 lba, nlb_zero_based, &cqe, &cid_r);
            if (rc) step_fail(-rc, "round=%u T3 Read %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T3 Read %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            *mismatch_um = -1;
            k_verify_pattern<<<blocks, threads>>>((const uint8_t*)pdata.rbuf_dev,
                                                   io_bytes, pat, mismatch_um);
            CUDA_OK(cudaDeviceSynchronize());
            if (*mismatch_um != -1)
                step_fail(0, "round=%u T3 IO %u: byte %d mismatch",
                          round_idx, i, *mismatch_um);
        }
        cudaFree(mismatch_um);
        step_ok("round=%u Tier 3 (PRP1+PRP_List, 16 KiB) x %u IOs, LBA [%"
                PRIu64 "..%" PRIu64 "]",
                round_idx, NR, LBA_BASE, LBA_BASE + 4u * (NR - 1) + 3);
    }

    /* ============================================================== */
    /* Phase R.8: Tier 4 -- SGL Data Block (skipped on PRP-only).    */
    /* ============================================================== */
    if ((info.sgl_supported & 0x3) == 0) {
        step_ok("round=%u Tier 4: SKIP -- controller advertises SGLS=0x%x "
                "(PRP-only)", round_idx, info.sgl_supported);
    } else {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 0;
        const size_t io_bytes = info.block_size;
        const uint64_t LBA_BASE = round_lba_base + 300;
        const unsigned NR = 8;
        queue_state& qw = rr.QS[0];
        queue_state& qr = rr.QS[1];
        char status_buf[64];

        int* mismatch_um = nullptr;
        CUDA_OK(cudaMallocManaged(&mismatch_um, sizeof(int)));

        for (unsigned i = 0; i < NR; i++) {
            uint64_t lba = LBA_BASE + i;
            uint8_t pat = WRITE_PATTERN_BYTE(round_idx, qw.dev.qid, 300 + i);

            int threads = 256, blocks = (int)((io_bytes + threads - 1) / threads);
            k_fill_pattern<<<blocks, threads>>>((uint8_t*)pdata.wbuf_dev,
                                                io_bytes, pat);
            CUDA_OK(cudaDeviceSynchronize());

            uint64_t sgl_addr_w = wpage(0);
            uint64_t sgl_meta_w = ((uint64_t)io_bytes & 0xffffffffu)
                                | ((uint64_t)NVME_SGL_DESC_BYTE15 << 56);
            nvme_cqe cqe;
            uint16_t cid_w;
            int rc = submit_and_poll(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_SGL,
                                     nsid, sgl_addr_w, sgl_meta_w,
                                     lba, nlb_zero_based, &cqe, &cid_w);
            if (rc) step_fail(-rc, "round=%u T4 Write %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T4 Write %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            CUDA_OK(cudaMemset(pdata.rbuf_dev, 0, io_bytes));
            uint64_t sgl_addr_r = rpage(0);
            uint64_t sgl_meta_r = ((uint64_t)io_bytes & 0xffffffffu)
                                | ((uint64_t)NVME_SGL_DESC_BYTE15 << 56);
            uint16_t cid_r;
            rc = submit_and_poll(qr, NVME_OPC_READ, NVME_FLAG_PSDT_SGL,
                                 nsid, sgl_addr_r, sgl_meta_r,
                                 lba, nlb_zero_based, &cqe, &cid_r);
            if (rc) step_fail(-rc, "round=%u T4 Read %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T4 Read %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            *mismatch_um = -1;
            k_verify_pattern<<<blocks, threads>>>((const uint8_t*)pdata.rbuf_dev,
                                                   io_bytes, pat, mismatch_um);
            CUDA_OK(cudaDeviceSynchronize());
            if (*mismatch_um != -1)
                step_fail(0, "round=%u T4 IO %u: byte %d mismatch",
                          round_idx, i, *mismatch_um);
        }
        cudaFree(mismatch_um);
        step_ok("round=%u Tier 4 (SGL Data Block, 4 KiB) x %u IOs, LBA [%"
                PRIu64 "..%" PRIu64 "]",
                round_idx, NR, LBA_BASE, LBA_BASE + NR - 1);
    }

    /* ============================================================== */
    /* Phase R.9: SQ-tail-wrap stress on QS[0].                      */
    /* ============================================================== */
    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 0;
        queue_state& qw = rr.QS[0];
        char status_buf[64];

        unsigned cnt = info.q_depth + 8u;
        const uint64_t LBA_BASE = round_lba_base + 1000;
        const size_t io_bytes = info.block_size;

        for (unsigned i = 0; i < cnt; i++) {
            uint64_t lba = LBA_BASE + i;
            uint8_t pat = WRITE_PATTERN_BYTE(round_idx, qw.dev.qid, 1000u + i);

            int threads = 256, blocks = (int)((io_bytes + threads - 1) / threads);
            k_fill_pattern<<<blocks, threads>>>((uint8_t*)pdata.wbuf_dev,
                                                io_bytes, pat);
            CUDA_OK(cudaDeviceSynchronize());

            nvme_cqe cqe;
            uint16_t cid;
            int rc = submit_and_poll(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP,
                                     nsid, wpage(0), 0,
                                     lba, nlb_zero_based, &cqe, &cid);
            if (rc) step_fail(-rc, "round=%u wrap Write %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u wrap Write %u: NVMe %s",
                          round_idx, i, status_buf);
            }
        }
        step_ok("round=%u SQ-tail-wrap: %u sequential Writes (sq wrapped "
                "past q_depth=%u, cq_phase flipped)",
                round_idx, cnt, info.q_depth);
    }
}

/* Tear down everything that run_one_round() built up.  In B6 this
 * is just the queue group + its 4 ring maps + the 2 user IO queues
 * the controller created.  Data buffers (wbuf / rbuf / prp_list_*)
 * stay alive across rounds and are reaped at fd close by
 * snvm_dev_release walking the per-fd data_maps list.  Order
 * matters: free per-queue UM bookkeeping FIRST, then
 * NVM_DESTROY_QUEUE_GROUP (cascades through user queues + the 4
 * ring maps), then cudaFree the GPU ring pages.                 */
static void teardown_one_round(int fd_dev, unsigned round_idx,
                               round_resources& rr) {
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        cudaFree(rr.QS[i].sq_tail_um);
        cudaFree(rr.QS[i].cq_head_um);
        cudaFree(rr.QS[i].cq_phase_um);
        cudaFree(rr.QS[i].out_cqe_um);
        cudaFree(rr.QS[i].timed_out_um);
    }

    {
        uint32_t gid = rr.group_id;
        if (do_ioctl(fd_dev, NVM_DESTROY_QUEUE_GROUP, &gid,
                     "NVM_DESTROY_QUEUE_GROUP") < 0)
            step_fail(errno, "round=%u NVM_DESTROY_QUEUE_GROUP", round_idx);
        step_ok("round=%u NVM_DESTROY_QUEUE_GROUP id=%u cascades through %u "
                "user queue(s) + %u ring map(s); data maps persist",
                round_idx, rr.group_id, TEST_NR_QUEUES,
                TEST_NR_QUEUES * 2);
    }

    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        cudaFree(rr.sq_dev[i]);
        cudaFree(rr.cq_dev[i]);
    }

    memset(&rr, 0, sizeof(rr));
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv) {
    int cuda_device = 0;
    unsigned nr_rounds = TEST_DEFAULT_ROUNDS;
    const char* bdf_str = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            cuda_device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) {
            int n = atoi(argv[++i]);
            if (n <= 0) { usage(argv[0]); return 1; }
            nr_rounds = (unsigned)n;
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        } else if (argv[i][0] != '-' && !bdf_str) {
            bdf_str = argv[i];
        } else {
            usage(argv[0]); return 1;
        }
    }
    if (!bdf_str) { usage(argv[0]); return 1; }

    struct pci_device_addr orig_bdf;
    if (parse_bdf(bdf_str, &orig_bdf) != 0) {
        fprintf(stderr, "Bad BDF: '%s' (expected DDDD:BB:DD.F)\n", bdf_str);
        return 1;
    }

    long psz = sysconf(_SC_PAGESIZE);
    if (psz <= 0) step_fail(errno, "sysconf(_SC_PAGESIZE)");

    CUDA_OK(cudaSetDevice(cuda_device));
    int n_dev = 0;
    CUDA_OK(cudaGetDeviceCount(&n_dev));
    if (cuda_device >= n_dev)
        step_fail(0, "--gpu %d invalid (CUDA sees %d devices)",
                  cuda_device, n_dev);
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, cuda_device));
    step_ok("CUDA setDevice(%d) name='%s' cap=%d.%d  rounds=%u",
            cuda_device, prop.name, prop.major, prop.minor, nr_rounds);

    /* ============================================================== */
    /* Phase 0: control plane + chrdev (lives across all rounds).    */
    /* ============================================================== */
    int fd_ctl = open("/dev/snvm_control", O_RDWR | O_NONBLOCK);
    if (fd_ctl < 0) step_fail(errno, "open(/dev/snvm_control)");
    step_ok("open(/dev/snvm_control) fd=%d", fd_ctl);

    struct pci_device_addr addr = orig_bdf;
    if (do_ioctl(fd_ctl, SNVM_CHRDEV_CREATE, &addr, "SNVM_CHRDEV_CREATE") < 0)
        step_fail(errno, "SNVM_CHRDEV_CREATE %s", bdf_str);
    int minor_n = addr.domain;
    step_ok("SNVM_CHRDEV_CREATE minor=%d", minor_n);

    char dev_path[64];
    snprintf(dev_path, sizeof(dev_path), "/dev/ssnvme%d", minor_n);
    int fd_dev = open(dev_path, O_RDWR);
    if (fd_dev < 0) step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd=%d", dev_path, fd_dev);

    /* ============================================================== */
    /* Phase 1: kernel ioq cap + bind + dev info.  These are bind-   */
    /* level state, NOT per-round; they persist across rounds.       */
    /* ============================================================== */
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
        step_ok("NVM_SET_KERNEL_IOQ_CAP cap=%u%s",
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
                info.disk_name[0] != '\0') { ok = 1; break; }
            usleep(100 * 1000);
        }
        if (!ok) step_fail(errno, "NVM_GET_DEV_INFO did not complete in 10s");
        step_ok("NVM_GET_DEV_INFO disk='%s' block_size=%zu q_depth=%u "
                "start_cq_idx=%u max_user_qid=%u sgls=0x%x",
                info.disk_name, info.block_size, info.q_depth,
                info.start_cq_idx, info.max_user_qid, info.sgl_supported);
    }
    if (info.block_size != 4096)
        step_fail(0, "smoke assumes 4 KiB-LBA controller; got %zu",
                  info.block_size);
    if ((size_t)info.q_depth * NVME_SQE_SIZE > GPU_PAGE_SIZE)
        step_fail(0, "GPU smoke: SQ ring (q_depth=%u * 64 = %zu B) "
                     "exceeds one GPU page (%zu B); lower io_queue_depth.",
                  info.q_depth,
                  (size_t)info.q_depth * NVME_SQE_SIZE,
                  GPU_PAGE_SIZE);

    /* ============================================================== */
    /* Phase 2: BAR0 mmap + cudaHostRegister (one-shot, lives across */
    /* rounds because doorbell offsets are stable per QID and QIDs   */
    /* are reused predictably by snvme's user_qid pool).              */
    /* ============================================================== */
    void* bar0_cpu = mmap(NULL, info.bar0_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd_dev, 0);
    if (bar0_cpu == MAP_FAILED)
        step_fail(errno, "mmap BAR0 (%u bytes)", info.bar0_size);
    CUDA_OK(cudaHostRegister(bar0_cpu, info.bar0_size,
                             cudaHostRegisterIoMemory));
    void* bar0_gpu = nullptr;
    CUDA_OK(cudaHostGetDevicePointer(&bar0_gpu, bar0_cpu, 0));
    step_ok("BAR0 mmap=%p gpu_va=%p (cudaHostRegister + GetDevicePointer)",
            bar0_cpu, bar0_gpu);

    /* ============================================================== */
    /* Phase 2b: persistent fd-scoped data buffers.  Allocated ONCE   */
    /* here and registered with map_kind=NVM_MAP_KIND_DATA +          */
    /* group_id=0; the kernel parks them on own->data_maps so they    */
    /* survive every per-round NVM_DESTROY_QUEUE_GROUP.  Reaped at    */
    /* fd close by snvm_dev_release.  This is the canonical B6       */
    /* usage pattern: one DMA pool spanning many short-lived queue   */
    /* groups, zero re-pinning per group.                             */
    /* ============================================================== */
    persistent_data_resources pdata;
    memset(&pdata, 0, sizeof(pdata));
    CUDA_OK(cudaMalloc(&pdata.wbuf_dev,       GPU_PAGE_SIZE));
    CUDA_OK(cudaMalloc(&pdata.rbuf_dev,       GPU_PAGE_SIZE));
    CUDA_OK(cudaMalloc(&pdata.prp_list_w_dev, GPU_PAGE_SIZE));
    CUDA_OK(cudaMalloc(&pdata.prp_list_r_dev, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(pdata.wbuf_dev,       0, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(pdata.rbuf_dev,       0, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(pdata.prp_list_w_dev, 0, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(pdata.prp_list_r_dev, 0, GPU_PAGE_SIZE));

    {
        auto map_data = [&](void* gpu_va, uint64_t* ioaddrs_out,
                            const char* what) {
            struct nvm_ioctl_map req;
            memset(&req, 0, sizeof(req));
            req.vaddr_start = (uint64_t)(uintptr_t)gpu_va;
            req.n_pages     = 1;
            req.ioaddrs     = (uint64_t)(uintptr_t)ioaddrs_out;
            req.ioq_idx     = -1;
            req.is_cq       = -1;
            req.group_id    = 0;                       /* fd-scoped */
            req.map_kind    = NVM_MAP_KIND_DATA;
            if (do_ioctl(fd_dev, NVM_MAP_DEVICE_MEMORY, &req, what) < 0)
                step_fail(errno, "%s gpu_va=%p", what, gpu_va);
        };
        map_data(pdata.wbuf_dev,       &pdata.wbuf_ioaddr,
                 "NVM_MAP_DEVICE_MEMORY(wbuf, DATA, fd-scoped)");
        map_data(pdata.rbuf_dev,       &pdata.rbuf_ioaddr,
                 "NVM_MAP_DEVICE_MEMORY(rbuf, DATA, fd-scoped)");
        map_data(pdata.prp_list_w_dev, &pdata.prp_list_w_ioaddr,
                 "NVM_MAP_DEVICE_MEMORY(prpl_w, DATA, fd-scoped)");
        map_data(pdata.prp_list_r_dev, &pdata.prp_list_r_ioaddr,
                 "NVM_MAP_DEVICE_MEMORY(prpl_r, DATA, fd-scoped)");
    }
    step_ok("persistent DATA maps registered (wbuf_ioaddr=0x%llx, "
            "rbuf_ioaddr=0x%llx); these survive every round destroy",
            (unsigned long long)pdata.wbuf_ioaddr,
            (unsigned long long)pdata.rbuf_ioaddr);

    /* ============================================================== */
    /* Phase 3+: rounds.  Each round builds its OWN queue group +    */
    /* fresh GPU rings only, runs all 4 tiers + wrap against the      */
    /* persistent data buffers above, then tears down the group +     */
    /* the 4 ring maps.  Data maps are untouched across rounds.       */
    /* scratch every round to exercise the snvme alloc/free path.   */
    /* ============================================================== */
    for (unsigned round = 0; round < nr_rounds; round++) {
        round_resources rr;
        memset(&rr, 0, sizeof(rr));

        fprintf(stderr, "\n===== ROUND %u / %u BEGIN =====\n",
                round + 1, nr_rounds);
        run_one_round(fd_dev, info, bar0_gpu, round, rr, pdata);
        teardown_one_round(fd_dev, round, rr);
        fprintf(stderr, "===== ROUND %u / %u END   =====\n",
                round + 1, nr_rounds);
    }

    /* ============================================================== */
    /* Final tear-down.                                               */
    /* ============================================================== */
    CUDA_OK(cudaHostUnregister(bar0_cpu));
    if (munmap(bar0_cpu, info.bar0_size) < 0)
        step_fail(errno, "munmap BAR0");
    step_ok("munmap BAR0 + cudaHostUnregister");

    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_UNBIND, &bdf, "SNVM_DEVICE_UNBIND") < 0)
            step_fail(errno, "SNVM_DEVICE_UNBIND");
        step_ok("SNVM_DEVICE_UNBIND %s", bdf_str);
    }
    if (close(fd_dev) < 0) step_fail(errno, "close(%s)", dev_path);
    step_ok("close(%s) -- snvm_dev_release cascades through %u DATA maps",
            dev_path, 4);

    /* cudaFree the persistent DATA buffers AFTER close(fd_dev) so the
     * snvm_dev_release path has already released the nvidia_p2p_get_pages
     * pin on them.  Doing it before close would leave the pages
     * referenced by snvme at fput() time, which is the leak the
     * snvm_dev_release hook is supposed to prevent in the first place
     * (see PORTING.md §7.3.1).                                       */
    CUDA_OK(cudaFree(pdata.wbuf_dev));
    CUDA_OK(cudaFree(pdata.rbuf_dev));
    CUDA_OK(cudaFree(pdata.prp_list_w_dev));
    CUDA_OK(cudaFree(pdata.prp_list_r_dev));

    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_CHRDEV_REMOVE, &bdf, "SNVM_CHRDEV_REMOVE") < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE");
        step_ok("SNVM_CHRDEV_REMOVE %s", bdf_str);
    }
    close(fd_ctl);

    fprintf(stderr, "\n=== snvme_smoke_gpu: all %d steps passed across "
            "%u round(s) ===\n", g_step, nr_rounds);
    return 0;
}

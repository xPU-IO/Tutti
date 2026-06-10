/**
 * nvmeservice_client_io.cu -- pure libnvm + CUDA IO smoke.
 *
 * Kept in its own translation unit so the gRPC / protobuf headers
 * never go through nvcc (their C++17 inline-static-variable traits
 * trip nvcc).  The .cpp half of the example fills in
 * nvmeservice_client_io_args, then calls run_nvmeservice_client_io.
 */

#include "nvmeservice_client_io.h"

#include <cuda_runtime.h>

#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include "ioctl.h"

#include <cerrno>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NVME_OPC_WRITE              0x01u
#define NVME_OPC_READ               0x02u
#define NVME_SQE_SIZE               64u
#define NVME_CQE_SIZE               16u
#define NVME_FLAG_PSDT_PRP          (0u << 6)

#define TEST_LBA_BASE               2621440ULL   /* 10 GiB / 4 KiB */
#define TEST_NR_IO                  4u

static constexpr size_t GPU_PAGE_SIZE = 1ULL << 16;

#define WRITE_PATTERN_BYTE(ioidx) \
    ((uint8_t)(0x4A ^ ((ioidx) & 0xff)))

struct nvme_sqe {
    uint8_t  opcode; uint8_t  flags; uint16_t cid; uint32_t nsid;
    uint64_t rsvd_2_3; uint64_t metadata; uint64_t prp1; uint64_t prp2;
    uint32_t cdw10, cdw11, cdw12, cdw13, cdw14, cdw15;
} __attribute__((packed));
static_assert(sizeof(nvme_sqe) == NVME_SQE_SIZE);

struct nvme_cqe {
    uint32_t result; uint32_t rsvd;
    uint16_t sq_head; uint16_t sq_id;
    uint16_t cid; uint16_t status;
} __attribute__((packed));
static_assert(sizeof(nvme_cqe) == NVME_CQE_SIZE);

struct test_queue_dev {
    nvme_sqe*           sq;
    nvme_cqe*           cq;
    volatile uint32_t*  sq_db;
    volatile uint32_t*  cq_db;
    uint16_t            q_depth;
    uint16_t            qid;
};

__global__ void k_submit_rw(test_queue_dev qd,
                            uint16_t* sq_tail_io,
                            uint16_t cid,
                            uint8_t opcode, uint8_t flags, uint32_t nsid,
                            uint64_t dptr0, uint64_t dptr1,
                            uint64_t slba, uint16_t nlb_zb) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint16_t tail = *sq_tail_io;
    nvme_sqe* slot = &qd.sq[tail];
    uint8_t* p = (uint8_t*)slot;
    #pragma unroll
    for (int i = 0; i < (int)sizeof(nvme_sqe); i++) p[i] = 0;
    slot->opcode = opcode; slot->flags = flags; slot->cid = cid;
    slot->nsid = nsid; slot->prp1 = dptr0; slot->prp2 = dptr1;
    slot->cdw10 = (uint32_t)(slba & 0xffffffffu);
    slot->cdw11 = (uint32_t)(slba >> 32);
    slot->cdw12 = (uint32_t)(nlb_zb & 0xffffu);
    __threadfence_system();
    uint16_t new_tail = (uint16_t)((tail + 1) % qd.q_depth);
    *qd.sq_db = new_tail;
    *sq_tail_io = new_tail;
}

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
        if ((status & 0x1u) == expected) {
            nvme_cqe tmp;
            tmp.result  = slot->result;  tmp.rsvd = slot->rsvd;
            tmp.sq_head = slot->sq_head; tmp.sq_id = slot->sq_id;
            tmp.cid = slot->cid;         tmp.status = status;
            *out_cqe = tmp;
            uint16_t new_head = (uint16_t)((head + 1) % qd.q_depth);
            if (new_head == 0) expected ^= 1u;
            __threadfence_system();
            *qd.cq_db = new_head;
            *cq_head_io = new_head;
            *cq_phase_io = expected;
            *timed_out = 0;
            return;
        }
        if (++i >= max_iters) { *timed_out = 1; return; }
    }
}

__global__ void k_fill(uint8_t* buf, size_t bytes, uint8_t pat) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bytes) buf[idx] = pat ^ (uint8_t)(idx >> 12);
}
__global__ void k_verify(const uint8_t* buf, size_t bytes,
                          uint8_t pat, int* mismatch_idx) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bytes) return;
    uint8_t expect = pat ^ (uint8_t)(idx >> 12);
    if (buf[idx] != expect) atomicCAS(mismatch_idx, -1, (int)idx);
}

/* ------------------------------------------------------------------ */
/* Logging helpers                                                    */
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
static int step_fail(int err, const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[FAIL] step=%-3d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, " errno=%d (%s)\n", err, err ? strerror(err) : "n/a");
    return err ? err : EIO;
}

#define CUDA_OK_OR(call_, label_)                                        \
    do {                                                                 \
        cudaError_t _e = (call_);                                        \
        if (_e != cudaSuccess) {                                         \
            rc = step_fail(0, "CUDA: %s -> %s",                          \
                           #call_, cudaGetErrorString(_e));              \
            goto label_;                                                 \
        }                                                                \
    } while (0)

extern "C"
int run_nvmeservice_client_io(const struct nvmeservice_client_io_args* a) {
    g_step = 0;
    int rc = 0;

    if (cudaSetDevice(a->cuda_dev) != cudaSuccess) {
        return step_fail(0, "cudaSetDevice(%d) failed", a->cuda_dev);
    }
    step_ok("cudaSetDevice(%d)", a->cuda_dev);

    nvm_ctrl_t* ctrl = nullptr;
    rc = nvm_ctrl_attach_client(&ctrl,
                                 a->snvme_dev_path,
                                 (uint32_t)a->bar0_size);
    if (rc != 0) return step_fail(rc, "nvm_ctrl_attach_client(%s)",
                                  a->snvme_dev_path);
    step_ok("nvm_ctrl_attach_client %s page=%u",
            a->snvme_dev_path, ctrl->page_size);

    void* bar0_gpu = nullptr;
    if (cudaHostGetDevicePointer(&bar0_gpu, (void*)ctrl->mm_ptr, 0) != cudaSuccess) {
        rc = step_fail(0, "cudaHostGetDevicePointer(BAR0)");
        goto out_ctrl;
    }

    {
    uint32_t gid = 0, max_q = 0;
    rc = nvm_create_group(ctrl, &gid, &max_q);
    if (rc != 0) { rc = step_fail(rc, "nvm_create_group"); goto out_ctrl; }
    step_ok("nvm_create_group gid=%u max_queues=%u granted=%d",
            gid, max_q, a->granted_queues);

    if (a->skip_io) {
        rc = nvm_destroy_group(ctrl, gid);
        if (rc != 0) (void) step_fail(rc, "nvm_destroy_group(skip-io)");
        else step_ok("nvm_destroy_group (skip-io path)");
        nvm_ctrl_free_client(ctrl);
        step_ok("nvm_ctrl_free_client (skip-io path)");
        return 0;
    }

    void* sq_dev = nullptr; void* cq_dev = nullptr;
    nvm_dma_t* dma_sq = nullptr; nvm_dma_t* dma_cq = nullptr;
    void* wbuf_dev = nullptr; void* rbuf_dev = nullptr;
    nvm_dma_t* dma_wbuf = nullptr; nvm_dma_t* dma_rbuf = nullptr;
    uint16_t* sq_tail_um = nullptr;  uint16_t* cq_head_um = nullptr;
    uint8_t*  cq_phase_um = nullptr; nvme_cqe* out_cqe_um = nullptr;
    int*      timed_out_um = nullptr; int* mismatch_um = nullptr;

    CUDA_OK_OR(cudaMalloc(&sq_dev, GPU_PAGE_SIZE), out_destroy);
    CUDA_OK_OR(cudaMalloc(&cq_dev, GPU_PAGE_SIZE), out_destroy);
    CUDA_OK_OR(cudaMemset(sq_dev, 0, GPU_PAGE_SIZE), out_destroy);
    CUDA_OK_OR(cudaMemset(cq_dev, 0, GPU_PAGE_SIZE), out_destroy);

    rc = nvm_dma_map_ring_device(&dma_sq, ctrl, gid, sq_dev, GPU_PAGE_SIZE, 0);
    if (rc != 0) { rc = step_fail(rc, "map_ring SQ"); goto out_destroy; }
    rc = nvm_dma_map_ring_device(&dma_cq, ctrl, gid, cq_dev, GPU_PAGE_SIZE, 1);
    if (rc != 0) { rc = step_fail(rc, "map_ring CQ"); goto out_destroy; }

    CUDA_OK_OR(cudaMalloc(&wbuf_dev, GPU_PAGE_SIZE), out_destroy);
    CUDA_OK_OR(cudaMalloc(&rbuf_dev, GPU_PAGE_SIZE), out_destroy);
    CUDA_OK_OR(cudaMemset(wbuf_dev, 0, GPU_PAGE_SIZE), out_destroy);
    CUDA_OK_OR(cudaMemset(rbuf_dev, 0, GPU_PAGE_SIZE), out_destroy);

    rc = nvm_dma_map_data_device(&dma_wbuf, ctrl, wbuf_dev, GPU_PAGE_SIZE);
    if (rc != 0) { rc = step_fail(rc, "map_data wbuf"); goto out_destroy; }
    rc = nvm_dma_map_data_device(&dma_rbuf, ctrl, rbuf_dev, GPU_PAGE_SIZE);
    if (rc != 0) { rc = step_fail(rc, "map_data rbuf"); goto out_destroy; }
    step_ok("mapped SQ/CQ + wbuf/rbuf");

    struct nvm_ioctl_add_user_queue add_req;
    std::memset(&add_req, 0, sizeof(add_req));
    add_req.group_id = gid;
    add_req.nr_pairs = 1;
    add_req.pairs[0].sq_vaddr = (uint64_t)(uintptr_t) sq_dev;
    add_req.pairs[0].cq_vaddr = (uint64_t)(uintptr_t) cq_dev;
    rc = nvm_add_user_queue(ctrl, &add_req);
    if (rc != 0) { rc = step_fail(rc, "nvm_add_user_queue"); goto out_destroy; }
    step_ok("nvm_add_user_queue qid=%u sq_db=0x%x cq_db=0x%x",
            add_req.out_pairs[0].qid,
            add_req.out_pairs[0].sq_doorbell_offset,
            add_req.out_pairs[0].cq_doorbell_offset);

    test_queue_dev qd;
    qd.sq = (nvme_sqe*) sq_dev; qd.cq = (nvme_cqe*) cq_dev;
    qd.q_depth = (uint16_t)a->queue_depth;
    qd.qid = (uint16_t) add_req.out_pairs[0].qid;
    qd.sq_db = (volatile uint32_t*) ((char*)bar0_gpu + add_req.out_pairs[0].sq_doorbell_offset);
    qd.cq_db = (volatile uint32_t*) ((char*)bar0_gpu + add_req.out_pairs[0].cq_doorbell_offset);

    CUDA_OK_OR(cudaMallocManaged(&sq_tail_um,   sizeof(uint16_t)), out_destroy);
    CUDA_OK_OR(cudaMallocManaged(&cq_head_um,   sizeof(uint16_t)), out_destroy);
    CUDA_OK_OR(cudaMallocManaged(&cq_phase_um,  sizeof(uint8_t)),  out_destroy);
    CUDA_OK_OR(cudaMallocManaged(&out_cqe_um,   sizeof(nvme_cqe)), out_destroy);
    CUDA_OK_OR(cudaMallocManaged(&timed_out_um, sizeof(int)),      out_destroy);
    CUDA_OK_OR(cudaMallocManaged(&mismatch_um,  sizeof(int)),      out_destroy);
    *sq_tail_um = 0; *cq_head_um = 0; *cq_phase_um = 1;

    {
    uint16_t next_cid = 0;
    uint32_t nsid = (a->namespace_id != 0) ? a->namespace_id : 1u;
    const uint32_t block = (a->blk_size != 0) ? a->blk_size : 4096u;
    char status_buf[64];
    for (unsigned i = 0; i < TEST_NR_IO; i++) {
        uint64_t lba = TEST_LBA_BASE + i;
        uint8_t pat = WRITE_PATTERN_BYTE(i);

        uint8_t* wslice = (uint8_t*)wbuf_dev + i * (size_t)block;
        uint8_t* rslice = (uint8_t*)rbuf_dev + i * (size_t)block;
        uint64_t wslice_io = (uint64_t)dma_wbuf->ioaddrs[0] + i * (uint64_t)block;
        uint64_t rslice_io = (uint64_t)dma_rbuf->ioaddrs[0] + i * (uint64_t)block;

        const int THR = 256;
        const int BLOCKS = ((int)block + THR - 1) / THR;

        k_fill<<<BLOCKS, THR>>>(wslice, block, pat);
        CUDA_OK_OR(cudaDeviceSynchronize(), out_destroy);

        uint16_t cid_w = next_cid++;
        k_submit_rw<<<1,1>>>(qd, sq_tail_um, cid_w,
                             NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP, nsid,
                             wslice_io, 0, lba, 0);
        k_poll_one<<<1,1>>>(qd, cq_head_um, cq_phase_um,
                            out_cqe_um, timed_out_um, 50000000ULL);
        CUDA_OK_OR(cudaDeviceSynchronize(), out_destroy);
        if (*timed_out_um) {
            rc = step_fail(ETIMEDOUT, "Write IO %u poll timeout", i);
            goto out_destroy;
        }
        if ((out_cqe_um->status >> 1) != 0) {
            std::snprintf(status_buf, sizeof(status_buf), "0x%04x", out_cqe_um->status);
            rc = step_fail(0, "Write IO %u status=%s", i, status_buf);
            goto out_destroy;
        }

        CUDA_OK_OR(cudaMemset(rslice, 0xff, block), out_destroy);
        uint16_t cid_r = next_cid++;
        k_submit_rw<<<1,1>>>(qd, sq_tail_um, cid_r,
                             NVME_OPC_READ, NVME_FLAG_PSDT_PRP, nsid,
                             rslice_io, 0, lba, 0);
        k_poll_one<<<1,1>>>(qd, cq_head_um, cq_phase_um,
                            out_cqe_um, timed_out_um, 50000000ULL);
        CUDA_OK_OR(cudaDeviceSynchronize(), out_destroy);
        if (*timed_out_um) {
            rc = step_fail(ETIMEDOUT, "Read IO %u poll timeout", i);
            goto out_destroy;
        }
        if ((out_cqe_um->status >> 1) != 0) {
            std::snprintf(status_buf, sizeof(status_buf), "0x%04x", out_cqe_um->status);
            rc = step_fail(0, "Read IO %u status=%s", i, status_buf);
            goto out_destroy;
        }

        *mismatch_um = -1;
        k_verify<<<BLOCKS, THR>>>(rslice, block, pat, mismatch_um);
        CUDA_OK_OR(cudaDeviceSynchronize(), out_destroy);
        if (*mismatch_um >= 0) {
            rc = step_fail(0, "IO %u (lba=%" PRIu64 ") mismatch at byte %d",
                           i, lba, *mismatch_um);
            goto out_destroy;
        }
    }
    step_ok("Write+Read+verify x %u IOs at LBA [%llu..%llu]",
            TEST_NR_IO,
            (unsigned long long)TEST_LBA_BASE,
            (unsigned long long)(TEST_LBA_BASE + TEST_NR_IO - 1));
    }

    rc = 0;

out_destroy:
    if (sq_tail_um)   cudaFree(sq_tail_um);
    if (cq_head_um)   cudaFree(cq_head_um);
    if (cq_phase_um)  cudaFree(cq_phase_um);
    if (out_cqe_um)   cudaFree(out_cqe_um);
    if (timed_out_um) cudaFree(timed_out_um);
    if (mismatch_um)  cudaFree(mismatch_um);
    if (dma_sq)   nvm_dma_unmap(dma_sq);
    if (dma_cq)   nvm_dma_unmap(dma_cq);
    if (sq_dev)   cudaFree(sq_dev);
    if (cq_dev)   cudaFree(cq_dev);
    {
        int drc = nvm_destroy_group(ctrl, gid);
        if (drc != 0) (void) step_fail(drc, "nvm_destroy_group");
        else step_ok("nvm_destroy_group gid=%u (rings cascade)", gid);
    }
    if (dma_wbuf) nvm_dma_unmap(dma_wbuf);
    if (dma_rbuf) nvm_dma_unmap(dma_rbuf);
    if (wbuf_dev) cudaFree(wbuf_dev);
    if (rbuf_dev) cudaFree(rbuf_dev);
    }

out_ctrl:
    nvm_ctrl_free_client(ctrl);
    if (rc == 0) step_ok("nvm_ctrl_free_client (no unbind, no chrdev_remove)");
    return rc;
}

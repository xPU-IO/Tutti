#ifndef __BENCHMARK_QUEUEPAIR_H__
#define __BENCHMARK_QUEUEPAIR_H__
// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <algorithm>
#include <cstdint>
#include "buffer.h"
#include "ctrl.h"
#include <tutti/cuda_like.h>
#include "nvm_types.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include "nvm_admin.h"
#include "nvm_queue.h"
#include "nvm_ctrl.h"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cmath>
#include "util.h"

#include <sys/ioctl.h>
#include <fcntl.h>
using error = std::runtime_error;
using std::string;


struct QueuePair
{
    uint32_t            pageSize;
    uint32_t            block_size;
    uint32_t            block_size_log;
    uint32_t            block_size_minus_1;
    uint32_t            nvmNamespace;
    //void*               prpList;
    //uint64_t*           prpListIoAddrs;
    nvm_queue_t         sq;
    nvm_queue_t         cq;
    uint16_t            qp_id;
    DmaPtr              sq_mem;
    DmaPtr              cq_mem;
    DmaPtr              prp_mem;
    BufferPtr           sq_tickets;
    //BufferPtr           sq_head_mark;
    BufferPtr           sq_tail_mark;
    BufferPtr           sq_cid;
    //BufferPtr           cq_tickets;
    BufferPtr           cq_head_mark;
    //BufferPtr           cq_tail_mark;
    BufferPtr           cq_pos_locks;
    BufferPtr             prp_list;
    //BufferPtr           cq_clean_cid;




#define MAX_SQ_ENTRIES_64K  (64*1024/64)
#define MAX_CQ_ENTRIES_64K  (64*1024/16)

    inline void init_gpu_specific_struct( const uint32_t cudaDevice) {
        this->sq_tickets = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        //this->sq_head_mark = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        this->sq_tail_mark = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        this->sq_cid = createBuffer(65536 * sizeof(padded_struct), cudaDevice);
        this->sq.tickets = (padded_struct*) this->sq_tickets.get();

        //this->sq.head_mark = (padded_struct*) this->sq_head_mark.get();
        this->sq.tail_mark = (padded_struct*) this->sq_tail_mark.get();
        this->sq.cid = (padded_struct*) this->sq_cid.get();
    //    std::cout << "init_gpu_specific: " << std::hex << this->sq.cid <<  std::endl;
        this->sq.qs_minus_1 = this->sq.qs - 1;
        this->sq.qs_log2 = (uint32_t) std::log2(this->sq.qs);


        //this->cq_tickets = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        this->cq_head_mark = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        //this->cq_tail_mark = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        //this->cq.tickets = (padded_struct*) this->cq_tickets.get();
        this->cq.head_mark = (padded_struct*) this->cq_head_mark.get();
        //this->cq.tail_mark = (padded_struct*) this->cq_tail_mark.get();
        this->cq.qs_minus_1 = this->cq.qs - 1;
        this->cq.qs_log2 = (uint32_t) std::log2(this->cq.qs);
        this->cq_pos_locks = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        this->cq.pos_locks = (padded_struct*) this->cq_pos_locks.get();

        //this->cq_clean_cid = createBuffer(this->cq.qs * sizeof(uint16_t), cudaDevice);
       // this->cq.clean_cid = (uint16_t*) this->cq_clean_cid.get();
    }



    inline QueuePair( const nvm_ctrl_t* ctrl, const uint32_t cudaDevice, const uint16_t qp_id, const uint64_t queueDepth)
    {
        //this->this = (QueuePairThis*) malloc(sizeof(QueuePairThis));


    //    std::cout << "HERE\n";
        uint64_t cap = ((volatile uint64_t*) ctrl->mm_ptr)[0];
        bool cqr = (cap & 0x0000000000010000) == 0x0000000000010000;
        //uint64_t sq_size = 16;
        //uint64_t cq_size = 16;

        uint64_t sq_size = (cqr) ?
            ((MAX_SQ_ENTRIES_64K <= ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) )) ? MAX_SQ_ENTRIES_64K :  ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) ) ) :
            ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) );
        uint64_t cq_size = (cqr) ?
            ((MAX_CQ_ENTRIES_64K <= ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) )) ? MAX_CQ_ENTRIES_64K :  ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) ) ) :
            ((((volatile uint16_t*) ctrl->mm_ptr)[0] + 1) );
        sq_size = std::min(queueDepth, sq_size);
        cq_size = std::min(queueDepth, cq_size);
 
        
        bool sq_need_prp = false;//(!cqr) || (sq_size > MAX_SQ_ENTRIES_64K);
        bool cq_need_prp = false;// (!cqr) || (cq_size > MAX_CQ_ENTRIES_64K);

        size_t sq_mem_size =  sq_size * sizeof(nvm_cmd_t) + sq_need_prp*(64*1024);
        size_t cq_mem_size =  cq_size * sizeof(nvm_cpl_t) + cq_need_prp*(64*1024);
        // printf("sq_size: %ld\tcq_size: %ld\n", sq_mem_size, cq_mem_size);
//        std::cout << sq_size << "\t" << sq_mem_size << std::endl;
        //size_t queueMemSize = ctrl.info.page_size * 2;
        //size_t prpListSize = ctrl.info.page_size * numThreads * (doubleBuffered + 1);
        //size_t prp_mem_size = sq_size * (4096) * 2;
//        std::cout << "Started creating DMA\n";
        // qmem->vaddr will be already a device pointer after the following call
        this->sq_mem = create_queue_Dma(ctrl, NVM_PAGE_ALIGN(sq_mem_size, 1UL << 16), cudaDevice,0,qp_id);
    //    std::cout << "Finished creating sq dma vaddr: " << this->sq_mem.get()->vaddr << "\tioaddr: " << std::hex<< this->sq_mem.get()->ioaddrs[0] << std::dec << std::endl;
        this->cq_mem = create_queue_Dma(ctrl, NVM_PAGE_ALIGN(cq_mem_size, 1UL << 16), cudaDevice,1,qp_id);
        //this->prp_mem = createDma(ctrl, NVM_PAGE_ALIGN(prp_mem_size, 1UL << 16), cudaDevice, adapter, segmentId);
    //    std::cout << "Finished creating cq dma vaddr: " << this->cq_mem.get()->vaddr << "\tioaddr: " << std::hex << this->cq_mem.get()->ioaddrs[0] << std::dec << std::endl;

        // Set members
        // this->pageSize = info.page_size;
        // this->block_size = ns.lba_data_size;
        // this->block_size_minus_1 = ns.lba_data_size-1;
        // this->block_size_log = std::log2(ns.lba_data_size);
        // this->nvmNamespace = ns.ns_id;

        this->qp_id = qp_id;



      //  std::cout << "before nvm_admin_cq_create\n";
        // Create completion queue
        // (nvm_aq_ref ref, nvm_queue_t* cq, uint16_t id, const nvm_dma_t* dma, size_t offset, size_t qs, bool need_prp = false)

        // std::cout << "after nvm_admin_cq_create\n";

        // Get a valid device pointer for CQ doorbell
        sq.qs = sq_size;
        cq.qs = cq_size;
        // printf("sq_size: %ld\tcq_size: %ld\n", sq.qs, cq.qs);
         // std::cout << "in preparequeuepair: " << std::hex << this->sq.cid << std::endl;
        init_gpu_specific_struct(cudaDevice);
        // std::cout << "in preparequeuepair: " << std::hex << this->sq.cid << std::endl;
        return;



    }

    /*
     * Queue-group QueuePair ctor.
     *
     * Successor to the legacy ctor above.  Differences:
     *
     *   - Allocates SQ + CQ ring memory via create_ring_Dma() against
     *     the supplied queue group_id, so the kernel registers them
     *     as RING_SQ / RING_CQ and links them onto g->maps.  This is
     *     the only ABI-correct way to feed rings into
     *     NVM_ADD_USER_QUEUE with the queue-group ABI.
     *   - Drops the (cqr, MQES)-derived sizing and uses ctrl->q_depth
     *     (NVM_GET_DEV_INFO authoritative value) clipped to queueDepth.
     *     This eliminates the BAR0-derived MQES read which the kernel
     *     ABI now lists as MUST-NOT-recompute.
     *   - Does NOT call init_gpu_specific_struct here; Controller::
     *     init_queues calls it explicitly after NVM_ADD_USER_QUEUE
     *     succeeds, so a per-pair allocation is paired with a
     *     per-pair Create I/O CQ + Create I/O SQ on the controller.
     *
     * The caller (Controller::init_queues) owns the rollback path:
     * if NVM_ADD_USER_QUEUE later fails, deleting this QueuePair will
     * drop the DmaPtrs which trigger nvm_dma_unmap.  Because the
     * group has not been destroyed yet that's a real unmap, not a
     * cascade no-op.
     */
    inline QueuePair(const nvm_ctrl_t* ctrl,
                     uint32_t          cudaDevice,
                     uint16_t          qp_id,
                     uint64_t          queueDepth,
                     uint32_t          group_id,
                     bool              defer_gpu_init)
    {
        if (group_id == 0) {
            throw error(string("QueuePair: group_id must be non-zero "
                               "(see nvm_dma_map_ring_device contract)"));
        }
        if (ctrl->q_depth == 0) {
            throw error(string("QueuePair: ctrl->q_depth == 0 -- did "
                               "controller owner bring-up / NVM_GET_DEV_INFO "
                               "run successfully?"));
        }

        const uint64_t hw_depth = (uint64_t) ctrl->q_depth;
        uint64_t sq_size = std::min<uint64_t>(queueDepth, hw_depth);
        uint64_t cq_size = std::min<uint64_t>(queueDepth, hw_depth);

        // Single-page contiguous ring requirement (NVMe spec PHYS_CONTIG):
        // q_depth=64 + SQE=64B = 4 KiB, q_depth=256 + CQE=16B = 4 KiB.
        // Round up to the GPU's 64 KiB granularity inside create_ring_Dma.
        size_t sq_mem_size = sq_size * sizeof(nvm_cmd_t);
        size_t cq_mem_size = cq_size * sizeof(nvm_cpl_t);

        this->sq_mem = create_ring_Dma(ctrl,
                                       NVM_PAGE_ALIGN(sq_mem_size, 1UL << 16),
                                       (int)cudaDevice, group_id,
                                       /*is_cq=*/0);
        this->cq_mem = create_ring_Dma(ctrl,
                                       NVM_PAGE_ALIGN(cq_mem_size, 1UL << 16),
                                       (int)cudaDevice, group_id,
                                       /*is_cq=*/1);

        this->qp_id = qp_id;
        this->sq.qs = (uint32_t) sq_size;
        this->cq.qs = (uint32_t) cq_size;

        if (!defer_gpu_init) {
            init_gpu_specific_struct(cudaDevice);
        }
    }

    // Default destructor: DmaPtr / BufferPtr members release their
    // GPU buffers via their own dtors.
    ~QueuePair() = default;

};

/*
 * Legacy bring-up tail.  Before the queue-group API, Controller::init_queues
 * pre-mapped all rings + run nvm_queue_set + nvm_device_init (BIND),
 * this function ran NVM_GET_DEV_INFO and called nvm_queue_clear() on
 * every QueuePair to derive doorbell host VAs from the controller's
 * dstrd via the SQ_DBL/CQ_DBL macros.
 *
 * In the queue-group flow the kernel returns BAR0-relative doorbell offsets
 * directly via NVM_ADD_USER_QUEUE.out_pairs[].sq/cq_doorbell_offset,
 * and Controller::init_queues writes those into nvm_queue_t.db
 * itself.  This function is kept only for binary compatibility with
 * any external caller; the Controller class no longer uses it.
 */
__attribute__((deprecated("Queue-group flow handles per-queue init inside "
                          "Controller::init_queues; use nvm_add_user_queue() "
                          "+ out_pairs[].*_doorbell_offset directly")))
inline int init_userioq_device(nvm_ctrl_t* ctrl, QueuePair** qps, struct disk* d)
{
    int err,i;

    err = ioctl_get_dev_info(ctrl, d);
    if(err)
    {
        return -1;
    }
    if(ctrl->nr_user_q > ctrl->cq_num)
    {
        return -1;
    }
    for (i = 0; i < ctrl->nr_user_q; i++){
        qps[i]->pageSize = ctrl->page_size;
        qps[i]->block_size = d->block_size;
        qps[i]->block_size_minus_1 = d->block_size -1;
        qps[i]->block_size_log = std::log2(d->block_size);
        qps[i]->nvmNamespace = d->ns_id;
        // printf("pageSize  is %u, block_size is %lu, block_size_minus_1 is %u, block_size_minus_1 is %u block_size_log is %u, nvmNamespace is %u\n",qps[i]->pageSize,qps[i]->block_size,qps[i]->block_size_minus_1,qps[i]->block_size_log,qps[i]->nvmNamespace);

        // clear cq
        nvm_queue_clear(&qps[i]->cq,ctrl,true,i+ctrl->start_cq_idx,qps[i]->cq.qs,1,qps[i]->cq_mem.get()->vaddr,qps[i]->cq_mem.get()->ioaddrs[0]);
    

        // clear sq
        nvm_queue_clear(&qps[i]->sq,ctrl,false,i+ctrl->start_cq_idx,qps[i]->sq.qs,1,qps[i]->sq_mem.get()->vaddr,qps[i]->sq_mem.get()->ioaddrs[0]);

    }
    return 0;
}



#endif

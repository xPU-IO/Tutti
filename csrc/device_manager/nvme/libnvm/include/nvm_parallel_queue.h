#ifndef __NVM_PARALLEL_QUEUE_H_
#define __NVM_PARALLEL_QUEUE_H_

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif


//#ifndef __CUDACC__
//#define __device__
//#define __host__
//#define __forceinline__ inline
//#endif

#include "host_util.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include <cuda/atomic>
#define LOCKED   1
#define UNLOCKED 0

__forceinline__ __device__ uint64_t get_id(uint64_t x, uint64_t y) {
    //return (x >> y);
    return (x >> y) * 2;  // (x/2^y) *2
}



inline __device__
uint16_t get_cid(nvm_queue_t* sq) {
    bool not_found = true;
    uint16_t id;

    do {
        id = sq->cid_ticket.fetch_add(1, cuda::memory_order_relaxed) & (65535);
        // id = sq->cid_ticket.fetch_add(1, cuda::memory_order_seq_cst) & (65535);
        //printf("in thread: %p\n", (void*) ((sq->cid)+id));
        uint64_t old = sq->cid[id].val.fetch_or(LOCKED, cuda::memory_order_acquire);
        not_found = old == LOCKED;
        //if (not_found)
        //       printf("still looking\n");
    } while (not_found);

    return id;

}

__forceinline__ __device__
void put_cid(nvm_queue_t* sq, uint16_t id) {
    sq->cid[id].val.store(UNLOCKED, cuda::memory_order_release);
}

__forceinline__ __device__
uint32_t move_tail(nvm_queue_t* q, uint32_t cur_tail) {
    uint32_t count = 0;




    bool pass = true;
    while (pass ) {
        //uint32_t count_copy = count;
        pass = (((cur_tail+count+1) & q->qs_minus_1) != (q->head.load(cuda::memory_order_relaxed) & q->qs_minus_1 ));
        if (pass) {
            pass = ((q->tail_mark[(cur_tail+count)&q->qs_minus_1].val.exchange(UNLOCKED, cuda::memory_order_relaxed)) == LOCKED);
            if (pass)
                count++;
        }

    }

    q->head_lock.fetch_add(1, cuda::memory_order_acq_rel);
    return (count);
}

__forceinline__ __device__
uint32_t move_head_cq(nvm_queue_t* q, uint32_t cur_head, nvm_queue_t* sq) {
    uint32_t count = 0;
    (void) sq;

    bool pass = true;
    //uint32_t old_head;
    while (pass) {
        uint32_t loc = (cur_head+count++)&q->qs_minus_1;
        pass = (q->head_mark[loc].val.exchange(UNLOCKED, cuda::memory_order_relaxed)) == LOCKED;
	//uint32_t cpl_entry = ((nvm_cpl_t*)q->vaddr)[loc].dword[3];
        //uint32_t cid = (cpl_entry & 0x0000ffff);
        //put_cid(sq, cid);
        /* if (pass) { */
        /*     uint32_t cpl_entry = ((nvm_cpl_t*)q->vaddr)[loc].dword[3]; */
        /*     uint32_t cid = (cpl_entry & 0x0000ffff); */
        /*     q->clean_cid[count-1] = cid; */
        /* } */
        // printf("cq_dequeue 5,cur_head is %u,loc is %u,pass is %u\n",cur_head,loc,pass);
    }
    count -= 1;
    if (count) {
        uint32_t loc_ = (cur_head + (count -1)) & q->qs_minus_1;
        uint32_t cpl_entry = ((nvm_cpl_t*)q->vaddr)[loc_].dword[2];
        uint16_t new_sq_head =  (cpl_entry & 0x0000ffff);
        uint32_t sq_move_count = 0;
        uint32_t cur_sq_head = sq->head.load(cuda::memory_order_relaxed);
        uint32_t loc = cur_sq_head & sq->qs_minus_1;
        // printf("+++new_sq_head: %llu\tcur_sq_head: %llu\tloc: %llu\tcpl_entry: %llx\n", (unsigned long long) new_sq_head, (unsigned long long) cur_sq_head, (unsigned long long) loc, (unsigned long long) cpl_entry);

        if (loc != new_sq_head) {
            for (; loc != new_sq_head; sq_move_count++, loc= ((loc+1)  & sq->qs_minus_1)) {
                //  printf("cq_dequeue 5\n");
                sq->tickets[loc].val.fetch_add(1, cuda::memory_order_relaxed);
            }
            // printf("---new_sq_head: %llu\tcur_sq_head: %llu\tloc: %llu\tsq_move_count: %llu\n", (unsigned long long) new_sq_head, (unsigned long long) cur_sq_head, (unsigned long long) loc, (unsigned long long) sq_move_count);

            sq->head.fetch_add(sq_move_count, cuda::memory_order_acq_rel);
        }
    }
    return (count);

}

__forceinline__ __device__
void clean_cids(nvm_queue_t* cq, nvm_queue_t* sq, uint32_t count) {
    for (size_t i  = 0; i < count; i++) {
        put_cid(sq, cq->clean_cid[i]);
    }
}

__forceinline__ __device__
uint32_t move_head_sq(nvm_queue_t* q, uint32_t cur_head) {
    uint32_t count = 0;
//    uint32_t cur_head = q->head.load(cuda::memory_order_acquire);

    bool pass = true;
    //uint32_t old_head;
    while (pass) {
//        count++;

        uint64_t loc = (cur_head + count)&q->qs_minus_1;
        pass = (q->head_mark[loc].val.exchange(UNLOCKED, cuda::memory_order_relaxed)) == LOCKED;
        if (pass) {
            //uint32_t old_cur_head = cur_head;
            //cur_head = q->head.fetch_add(1, cuda::memory_order_acq_rel);
            q->tickets[loc].val.fetch_add(1, cuda::memory_order_relaxed);

            //cur_head++;
            count++;


        }


    }
//    if (count)
//	q->head.fetch_add(count, cuda::memory_order_release);
    /* for (uint32_t i = 0; i < count; i++) */
    /*     q->tickets[(cur_head + i) & q->qs_minus_1].val.fetch_add(1, cuda::memory_order_release); */
    return (count);

}

typedef ulonglong4 copy_type;

__forceinline__ __device__
uint16_t sq_enqueue(nvm_queue_t* sq, nvm_cmd_t* cmd, cuda::atomic<uint64_t, cuda::thread_scope_device>* pc_tail =NULL, uint64_t * cur_pc_tail=NULL) {

    //uint32_t mask = __activemask();
    //uint32_t active_count = __popc(mask);
    //uint32_t leader = __ffs(mask) - 1;
    //uint32_t lane = lane_id();
    uint32_t ticket;
    ticket = sq->in_ticket.fetch_add(1, cuda::memory_order_relaxed);
    /* if (lane == leader) { */
    /*     ticket = sq->in_ticket.fetch_add(active_count, cuda::memory_order_acquire); */
    /* } */

    /* ticket = __shfl_sync(mask, ticket, leader); */
    /* ticket += __popc(mask & ((1 << lane) - 1)); */

    uint32_t pos = ticket & (sq->qs_minus_1);
    uint64_t id = get_id(ticket, sq->qs_log2);

    //uint64_t k = 0;
    unsigned int ns = 8;
    while ((sq->tickets[pos].val.load(cuda::memory_order_relaxed) != id) ) {
        /*if (k++ % 100 == 0)   {
            printf("tid: %llu\tpos: %llu\tticket: %llu\tid: %llu\ttickets_pos: %llu\tqueue_head: %llu\tqueue_tail: %llu\n",
                   (unsigned long long) threadIdx.x, (unsigned long long) pos,
                   (unsigned long long)ticket, (unsigned long long)id, (unsigned long long) (sq->tickets[pos].val.load(cuda::memory_order_acquire)),
                   (unsigned long long)(sq->head.load(cuda::memory_order_acquire) & (sq->qs_minus_1)), (unsigned long long)(sq->tail.load(cuda::memory_order_acquire) & (sq->qs_minus_1)));
                   }*/
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    }

    ns = 8;
    while ((sq->tickets[pos].val.load(cuda::memory_order_acquire) != id) ) {
        /*if (k++ % 100 == 0)   {
            printf("tid: %llu\tpos: %llu\tticket: %llu\tid: %llu\ttickets_pos: %llu\tqueue_head: %llu\tqueue_tail: %llu\n",
                   (unsigned long long) threadIdx.x, (unsigned long long) pos,
                   (unsigned long long)ticket, (unsigned long long)id, (unsigned long long) (sq->tickets[pos].val.load(cuda::memory_order_acquire)),
                   (unsigned long long)(sq->head.load(cuda::memory_order_acquire) & (sq->qs_minus_1)), (unsigned long long)(sq->tail.load(cuda::memory_order_acquire) & (sq->qs_minus_1)));
                   }*/
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    }

//    ulonglong4* queue_loc = ((ulonglong4*)(((nvm_cmd_t*)(sq->vaddr)) + pos));
//    ulonglong4* cmd_ = ((ulonglong4*)(cmd->dword));
//#pragma unroll
//    for (uint32_t i = 0; i < 64/sizeof(ulonglong4); i++) {
//        queue_loc[i] = cmd_[i];
//    }


/*     while (((pos+1) & sq->qs_minus_1) == (sq->head.load(cuda::memory_order_acquire) & (sq->qs_minus_1))) { */
/* #if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)) */
/*         __nanosleep(100); */
/* #endif */
/*     } */

    copy_type* queue_loc = ((copy_type*)(((nvm_cmd_t*)(sq->vaddr)) + pos));
    copy_type* cmd_ = ((copy_type*)(cmd->dword));

    //printf("+++tid: %llu\tcid: %llu\tsq_loc: %llx\tpos: %llu\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) (cmd->dword[0] >> 16), (unsigned long long) queue_loc, (unsigned long long) pos);

    //printf("sq->loc: %p\n", queue_loc);
    //queue_loc[0] =   *((ulonglong4*) (cmd->dword+0));
    //queue_loc[1] =   *((ulonglong4*) (cmd->dword+8));
    //queue_loc->dword[0] = cmd->dword[0];
    //queue_loc->dword[1] = cmd->dword[1];
    //queue_loc->dword[6] = cmd->dword[6];
    //queue_loc->dword[7] = cmd->dword[7];

    //*((ulonglong4*) (queue_loc->dword+8)) =   *((ulonglong4*) (cmd->dword+8));
    //queue_loc->dword[8] = cmd->dword[8];
    //queue_loc->dword[9] = cmd->dword[9];
    //queue_loc->dword[10] = cmd->dword[10];
    //queue_loc->dword[11] = cmd->dword[11];
    //queue_loc->dword[12] = cmd->dword[12];

#pragma unroll
    for (uint32_t i = 0; i < 64/sizeof(copy_type); i++) {
        queue_loc[i] = cmd_[i];
    }
    // printf("queue_loc is %lx\n", queue_loc);
    // for(k=0;k<16;k++ )
    // {
    //     printf ("sq %u\n",k);
    //     for(j=0;j<16;j++)
    //     {
    //         printf("dowrd %u is %u\t",j,((nvm_cmd_t*)sq->vaddr)[k].dword[j]);
    //         if(j>0 && j%8==0)
    //             printf ("\n");

    //     }
    //     printf ("\n");
    // }


    //uint32_t new_tail = pos;
    /*
    bool proceed = false;
    do {

        uint32_t cur_head = sq->head.load(cuda::memory_order_acquire) & (sq->qs_minus_1);

        uint32_t check = (cur_head - 1)  & (sq->qs_minus_1);
        //uint32_t cur_head_mod = cur_head & (sq->qs_minus_1);
        //uint32_t size = (cur_head > new_tail) ? (sq->qs - cur_head + new_tail) : (new_tail - cur_head);
        proceed = check != pos;
        printf("here pos: %llu\n", (unsigned long long) pos);
    } while(!proceed);
    */
    //sq->tickets[pos].val.store(id + 1, cuda::memory_order_release);
    if (pc_tail) {
        *cur_pc_tail = pc_tail->load(cuda::memory_order_relaxed);
    }
    sq->tail_mark[pos].val.store(LOCKED, cuda::memory_order_release);
    /*     while (((pos+1) & sq->qs_minus_1) == (sq->head.load(cuda::memory_order_acquire) & (sq->qs_minus_1))) { */
/* #if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)) */
/*         __nanosleep(100); */
/* #endif */
/*     } */
    bool cont = true;
    ns = 8;
    cont = sq->tail_mark[pos].val.load(cuda::memory_order_relaxed) == LOCKED;
    while(cont) {
        bool new_cont = sq->tail_lock.load(cuda::memory_order_relaxed) == LOCKED;
        if (!new_cont) {
            new_cont = sq->tail_lock.fetch_or(LOCKED, cuda::memory_order_acquire) == LOCKED;
            if(!new_cont) {
                uint32_t cur_tail = sq->tail.load(cuda::memory_order_relaxed);

                uint32_t tail_move_count = move_tail(sq, cur_tail);
                // printf("tail_move_count is %u,cur_tail is %u\n",tail_move_count,cur_tail);
                if (tail_move_count) {
                    uint32_t new_tail = cur_tail + tail_move_count;
                    uint32_t new_db = (new_tail) & (sq->qs_minus_1);
                    if (pc_tail) {
                        *cur_pc_tail = pc_tail->load(cuda::memory_order_acquire);
                    }
#if defined(TUTTI_USE_MACA)
                    __threadfence_system();
                    *(sq->db) = new_db;
#else
//                    *(sq->db) = new_db;
		    asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(sq->db),"r"(new_db) : "memory");
#endif
                    //sq->tail_copy.store(new_tail, cuda::memory_order_release);
//	            printf("wrote SQ_db: %llu\tcur_tail: %llu\tmove_count: %llu\tsq_tail: %llu\tsq_head: %llu\n", (unsigned long long) new_db, (unsigned long long) cur_tail, (unsigned long long) tail_move_count, (unsigned long long) (new_tail),  (unsigned long long)(sq->head.load(cuda::memory_order_acquire)));
                    sq->tail.store(new_tail, cuda::memory_order_release);
                    //cont = false;
                }
                sq->tail_lock.store(UNLOCKED, cuda::memory_order_release);
            }
        }
        cont = sq->tail_mark[pos].val.load(cuda::memory_order_relaxed) == LOCKED;
        if (cont) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
#endif
        }

    }



    sq->tickets[pos].val.fetch_add(1, cuda::memory_order_acq_rel);
    return pos;

}

__forceinline__ __device__
void sq_dequeue(nvm_queue_t* sq, uint16_t pos) {

    sq->head_mark[pos].val.store(LOCKED, cuda::memory_order_relaxed);
    bool cont = true;
    unsigned int ns = 8;
    cont = sq->head_mark[pos].val.load(cuda::memory_order_relaxed) == LOCKED;
    while (cont) {
            bool new_cont = sq->head_lock.exchange(LOCKED, cuda::memory_order_acquire) == LOCKED;
            if (!new_cont){
                uint32_t cur_head = sq->head.load(cuda::memory_order_relaxed);;

                uint32_t head_move_count = move_head_sq(sq, cur_head);
                //(void) head_move_count;
                if (head_move_count) {
                    sq->head.store(cur_head + head_move_count, cuda::memory_order_relaxed);
                    //for (uint16_t i = 0; i < head_move_count; i++)
                    //   sq->tickets[(cur_head+i) & sq->qs_minus_1].val.fetch_add(1, cuda::memory_order_release);
                    //cont = false;
  //              printf("sq cur_head: %llu\thead_move_count: %llu\tnew_head: %llu\n", (unsigned long long) cur_head, (unsigned long long) head_move_count, (unsigned long long) (cur_head+head_move_count));

                }

                /* if (head_move_count) { */
                /*     uint32_t new_head = cur_head + head_move_count; */
                /*     //printf("sq new_head: %llu\n", (unsigned long long) new_head); */
                /*     sq->head.store(new_head, cuda::memory_order_release); */
                /* } */
                sq->head_lock.store(UNLOCKED, cuda::memory_order_release);
            }
            cont = sq->head_mark[pos].val.load(cuda::memory_order_relaxed) == LOCKED;
            if (cont) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
                __nanosleep(ns);
                if (ns < 256) {
                    ns *= 2;
                }
#endif

            }
    }



}

__forceinline__ __device__
uint32_t cq_poll(nvm_queue_t* cq, uint16_t search_cid, uint32_t* loc_ = NULL, uint32_t* cq_head = NULL) {
    // uint64_t j = 0,k=0;
    unsigned int ns = 8;
    //uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("---tid: %llu\tcid: %llu\tcq_start: %llx\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) (search_cid), (uint64_t) cq->vaddr);

    while (true) {
        uint32_t head = cq->head.load(cuda::memory_order_relaxed);
        for (size_t i = 0; i < cq->qs_minus_1; i++) {
            uint32_t cur_head = head + i;
            bool search_phase = ((~(cur_head >> cq->qs_log2)) & 0x01);
            uint32_t loc = cur_head & (cq->qs_minus_1);
            uint32_t cpl_entry = ((nvm_cpl_t*)cq->vaddr)[loc].dword[3];
            // for(k=0;k<16;k++ )
            // {
            //     printf("dowrd0 is %u\t dowrd1 is %u\t dowrd2 is %u\t dowrd3 is %u\t",((nvm_cpl_t*)cq->vaddr)[k].dword[0],((nvm_cpl_t*)cq->vaddr)[k].dword[1],((nvm_cpl_t*)cq->vaddr)[k].dword[2],((nvm_cpl_t*)cq->vaddr)[k].dword[3]);
            //     printf ("\n");
            // }
            uint32_t cid = (cpl_entry & 0x0000ffff);
            bool phase = (cpl_entry & 0x00010000) >> 16;
//             if (j % 10000000 == 0)

                // printf("qs_log2: %llu\thead: %llu\tcur_head: %llu\tsearch_cid: %llu\tsearch_phase: %llu\tcq->loc: %p\tcq->qs: %llu\ti: %llu\tj: %llu\tcid: %llu\tphase:%llu\tmark: %llu\tcpl_entry is %llu\n",
                //         (unsigned long long) cq->qs_log2,
                //         (unsigned long long)head, (unsigned long long) cur_head, (unsigned long long) search_cid, (unsigned long long) search_phase, ((volatile nvm_cpl_t*)cq->vaddr)+loc,
                //         (unsigned long long) cq->qs, (unsigned long long) i, (unsigned long long) j, (unsigned long long) cid, (unsigned long long) phase,
                //         (unsigned long long) cq->head_mark[loc].val.load(cuda::memory_order_acquire),cpl_entry);

//            if ((cid == search_cid) && (phase == search_phase) && (cq->head_mark[loc].load(cuda::memory_order_acquire) == UNLOCKED)){
            // fail reason see 4.2.3.1 in NVM Express Base Specification
            if ((cid == search_cid) && (phase == search_phase)){

                 if ((cpl_entry >> 17) != 0) {
                    printf("NVM Error: %llx\tcid: %llu\n", (unsigned long long) (cpl_entry >> 17), (unsigned long long) search_cid);
                    assert(false);
                 }

                // *cq_head = head;
                // *loc_ = cur_head;
                return loc;
            }
            if (phase != search_phase)
                break;
            //__nanosleep(1000);
        }
        // j++;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
         __nanosleep(ns);
         if (ns < 256) {
             ns *= 2;
         }
#endif
#if defined(TUTTI_USE_MACA)
        __threadfence_system();
#endif
    }
}

/*
 * Bounded CQ poll: like cq_poll but returns NVM_CQ_TIMEOUT when the
 * poll budget (max_polls iterations) is exhausted without finding the
 * completion. This prevents the GPU kernel from spinning forever on a
 * dead controller / lost completion.
 *
 * Returns the CQ slot index on success, or NVM_CQ_TIMEOUT on timeout.
 * The caller must check for NVM_CQ_TIMEOUT and record a per-entry
 * failure instead of assuming success.
 */
#define NVM_CQ_TIMEOUT 0xFFFFFFFFu

__forceinline__ __device__
uint32_t cq_poll_bounded(nvm_queue_t* cq, uint16_t search_cid,
                          uint32_t max_polls)
{
    unsigned int ns = 8;
    uint32_t polls = 0;

    while (polls < max_polls) {
        uint32_t head = cq->head.load(cuda::memory_order_relaxed);
        for (size_t i = 0; i < cq->qs_minus_1; i++) {
            uint32_t cur_head = head + i;
            bool search_phase = ((~(cur_head >> cq->qs_log2)) & 0x01);
            uint32_t loc = cur_head & (cq->qs_minus_1);
            uint32_t cpl_entry = ((nvm_cpl_t*)cq->vaddr)[loc].dword[3];
            uint32_t cid = (cpl_entry & 0x0000ffff);
            bool phase = (cpl_entry & 0x00010000) >> 16;
            if ((cid == search_cid) && (phase == search_phase)){
                return loc;
            }
            if (phase != search_phase)
                break;
        }
        ++polls;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
         __nanosleep(ns);
         if (ns < 256) {
             ns *= 2;
         }
#endif
#if defined(TUTTI_USE_MACA)
        __threadfence_system();
#endif
    }
    return NVM_CQ_TIMEOUT;
}

__forceinline__ __device__
void cq_dequeue(nvm_queue_t* cq, uint16_t pos, nvm_queue_t* sq, uint32_t loc_ = 0, uint32_t cur_head_ = 0) {
    cq->tail.fetch_add(1, cuda::memory_order_acq_rel);

    unsigned int ns = 8;
    // unsigned int k=0;
    while ((cq->pos_locks[pos].val.load(cuda::memory_order_relaxed) != 0) ) {
        // if(k++ % 100 == 0)   {
        //     printf("cq_dequeue 1\n");
        // }
        /*if (k++ % 100 == 0)   {
            printf("tid: %llu\tpos: %llu\tticket: %llu\tid: %llu\ttickets_pos: %llu\tqueue_head: %llu\tqueue_tail: %llu\n",
                   (unsigned long long) threadIdx.x, (unsigned long long) pos,
                   (unsigned long long)ticket, (unsigned long long)id, (unsigned long long) (sq->tickets[pos].val.load(cuda::memory_order_acquire)),
                   (unsigned long long)(sq->head.load(cuda::memory_order_acquire) & (sq->qs_minus_1)), (unsigned long long)(sq->tail.load(cuda::memory_order_acquire) & (sq->qs_minus_1)));
                   }*/
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    }

    ns = 8;
    while ((cq->pos_locks[pos].val.fetch_or(1, cuda::memory_order_acquire) != 0) ) {
        // if(k++ % 100 == 0)   {
        //     printf("cq_dequeue 2\n");
        // }
        /*if (k++ % 100 == 0)   {
            printf("tid: %llu\tpos: %llu\tticket: %llu\tid: %llu\ttickets_pos: %llu\tqueue_head: %llu\tqueue_tail: %llu\n",
                   (unsigned long long) threadIdx.x, (unsigned long long) pos,
                   (unsigned long long)ticket, (unsigned long long)id, (unsigned long long) (sq->tickets[pos].val.load(cuda::memory_order_acquire)),
                   (unsigned long long)(sq->head.load(cuda::memory_order_acquire) & (sq->qs_minus_1)), (unsigned long long)(sq->tail.load(cuda::memory_order_acquire) & (sq->qs_minus_1)));
                   }*/
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    }

    //uint32_t pos = cq_poll(cq, cid);
    cq->head_mark[pos].val.store(LOCKED, cuda::memory_order_release);


    bool cont = true;
    ns = 8;
    cont = cq->head_mark[pos].val.load(cuda::memory_order_relaxed) == LOCKED;
    while (cont) {
            bool new_cont = cq->head_lock.fetch_or(LOCKED, cuda::memory_order_acquire) == LOCKED;
            // printf("cq_dequeue 3,cont is %u\n",new_cont);
            if (!new_cont) {
                uint32_t cur_head = cq->head.load(cuda::memory_order_relaxed);;
                uint32_t head_move_count = move_head_cq(cq, cur_head, sq);
                // printf("cq head_move_count: %llu\n", (unsigned long long) head_move_count);

                if (head_move_count) {
                    uint32_t new_head = cur_head + head_move_count;

                    uint32_t new_db = (new_head) & (cq->qs_minus_1);
#if defined(TUTTI_USE_MACA)
                    *(cq->db) = new_db;
#else
                    //*(cq->db) = new_db;
                    asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(cq->db),"r"(new_db) : "memory");
#endif
		    //cq->head_copy.store(new_head, cuda::memory_order_release);
//                    printf("wrote CQ_db: %llu\tcur_head: %llu\tmove_count: %llu\tcq_head: %llu\tcq_tail: %llu\n", (unsigned long long) new_db, (unsigned long long) cur_head, (unsigned long long) head_move_count, (unsigned long long) (new_head),  (unsigned long long)(cq->tail.load(cuda::memory_order_acquire)));
                    cq->head.store(new_head, cuda::memory_order_release);//axed);

                    //clean_cids(cq, sq, head_move_count);
                    //cont = false;
                }
                cq->head_lock.store(UNLOCKED, cuda::memory_order_release);
            }
            cont = cq->head_mark[pos].val.load(cuda::memory_order_relaxed) == LOCKED;
            if (cont) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
                __nanosleep(ns);
                if (ns < 256) {
                    ns *= 2;
                }
#endif
            }
    }


	uint64_t j = 0;
    uint32_t new_head = cq->head.load(cuda::memory_order_relaxed);
    ns = 8;
//    uint32_t cur_head_mod = cur_head_ & (cq->qs_minus_1);
    do {
        //      uint32_t new_head_mod = new_head & (cq->qs_minus_1);

        if (new_head > cur_head_) {
            if ((loc_ >= cur_head_) && (loc_ < new_head))
                break;


        }
        else if (new_head < cur_head_) {
            if ((loc_ >= cur_head_))
                break;
            if (loc_ < new_head)
                break;
        }

        j++;
        new_head = cq->head.load(cuda::memory_order_relaxed);
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    } while(true);

    cq->pos_locks[pos].val.store(0, cuda::memory_order_release);
}

//#ifndef __CUDACC__
//#undef __device__
//#undef __host__
//#undef __forceinline__
//#endif

#endif // __NVM_PARALLEL_QUEUE_H_

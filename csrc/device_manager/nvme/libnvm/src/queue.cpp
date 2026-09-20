#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include "regs.h"
#include "lib_util.h"
#include <cuda/atomic>



void nvm_queue_reset(nvm_queue_t* queue)
{
    queue->head = 0;
    queue->tail = 0;
    queue->last = 0;
    queue->phase = 1;
}



nvm_cpl_t* nvm_cq_dequeue_block(nvm_queue_t* cq, uint64_t timeout)
{
    uint64_t nsecs = timeout * 1000000UL;
    nvm_cpl_t* cpl = nvm_cq_dequeue(cq);

    while (cpl == NULL && nsecs > 0)
    {
        nsecs = _nvm_delay_remain(nsecs);
        cpl = nvm_cq_dequeue(cq);
    }

    return cpl;
}


#ifndef __NVM_AQ_H__
#define __NVM_AQ_H__
// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>






/*
 * Destroy admin queues and references.
 *
 * Send NVM abort command to controller and deallocate admin queues.
 *
 * After calling this function, all admin queue references are invalid.
 * This also means that remote references will no longer be valid.
 *
 * This function will also work for unbinding remote references.
 */
void nvm_aq_destroy(nvm_aq_ref ref);



//int nvm_tcp_rpc_enable(nvm_aq_ref ref, uint16_t port, nvm_rpc_cb_t filter, void* data);
//int nvm_tcp_rpc_disable(nvm_aq_ref ref, uint16_t port);






#endif /* #ifdef __NVM_AQ_H__ */

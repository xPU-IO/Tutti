#ifndef __NVM_DMA_H__
#define __NVM_DMA_H__
// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif




/*
 * Create DMA mapping descriptor from physical/bus addresses.
 *
 * Create a DMA mapping descriptor, describing a region of memory that is
 * accessible for the NVM controller. The caller must supply physical/bus  
 * addresses of physical memory pages, page size and total number of pages.
 * As the host's page size may differ from the controller's page size (MPS),
 * this function will calculate the necessary offsets into the actual memory
 * pages.
 *
 * While virtual memory is assumed to be continuous, the physical pages do not
 * need to be contiguous. Physical/bus addresses must be aligned to the 
 * controller's page size.
 *
 * Note: vaddr can be NULL.
 */
int nvm_dma_map(nvm_dma_t** map,                // Mapping descriptor reference
                const nvm_ctrl_t* ctrl,         // NVM controller reference
                void* vaddr,                    // Pointer to userspace memory (can be NULL if not required)
                size_t page_size,               // Physical page size
                size_t n_pages,                 // Number of pages to map
                const uint64_t* page_addrs);    // List of physical/bus addresses to the pages



/*
 * Create DMA mapping descriptor using offsets from a previously 
 * created DMA descriptor.
 */
int nvm_dma_remap(nvm_dma_t** new_map, const nvm_dma_t* other_map);



/*
 * Remove DMA mapping descriptor.
 *
 * Unmap DMA mappings (if necessary) and remove the descriptor.
 * This function destroys the descriptor.
 */
void nvm_dma_unmap(nvm_dma_t* map);



/*
 * Create DMA mapping descriptor from virtual address using the kernel module.
 * This function is similar to nvm_dma_map, except the user is not required
 * to pass physical/bus addresses. 
 *
 * Note: vaddr can not be NULL, and must be aligned to system page size.
 *
 * LEGACY API: kernel registers the map against the controller-
 * global host_list and `map_kind` defaults to NVM_MAP_KIND_UNSPECIFIED.
 * is_cq / ioq_idx are honoured by the legacy NVM_SET_IOQ_NUM bring-up
 * path only.  New callers should prefer:
 *   - nvm_dma_map_data_host()         (DATA buffer, fd-scoped)
 *   - nvm_dma_map_ring_host(group_id) (SQ/CQ ring, group-scoped)
 */
int nvm_dma_map_host(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* vaddr, size_t size,int is_cq, int ioq_idx);



//#if ( defined( __CUDA__ ) || defined( __CUDACC__ ) )

/*
 * Create DMA mapping descriptor from CUDA device pointer using the kernel
 * module. This function is similar to nvm_dma_map_host, except the memory
 * pointer must be a valid CUDA device pointer (see manual for 
 * cudaGetPointerAttributes).
 *
 * The controller handle must have been created using the kernel module.
 *
 * Note: vaddr can not be NULL, and must be aligned to GPU page size.
 *
 * LEGACY API.  See nvm_dma_map_host() above for the explicit-map successors:
 *   - nvm_dma_map_data_device()         (DATA buffer, fd-scoped)
 *   - nvm_dma_map_ring_device(group_id) (SQ/CQ ring, group-scoped)
 */
int nvm_dma_map_device(nvm_dma_t** map, const nvm_ctrl_t* ctrl, void* devptr, size_t size);

//#endif /* __CUDA__ */

/*
 * LEGACY API: registers a GPU IO-queue ring via the now-deprecated
 * NVM_MAP_DEVICE_QUEUE_MEMORY ioctl, which encoded is_cq / qno through the
 * struct nvm_ioctl_map.{ioq_idx,is_cq} fields.  New code registers rings
 * via NVM_MAP_DEVICE_MEMORY with map_kind == NVM_MAP_KIND_RING_{SQ,CQ} and a
 * non-zero group_id; see nvm_dma_map_ring_device() below.
 */
int nvm_dma_map_queue_device(nvm_dma_t** map, const nvm_ctrl_t* ctrl, void* devptr, size_t size,unsigned int is_cq, uint16_t qno);



/* ===================================================================
 * Explicit-intent DMA mapping API.
 *
 * These four entry points replace the overloaded legacy nvm_dma_map_*
 * functions above by encoding the map's role (DATA vs ring) and its
 * lifecycle scope (fd vs queue-group) at the API boundary instead of
 * inferring it from sentinel values in ioq_idx / is_cq.
 *
 *   data_*    -> kernel registers as NVM_MAP_KIND_DATA, fd-scoped.
 *                group_id is forced to 0 (kernel ignores it for DATA).
 *                Survives NVM_DESTROY_QUEUE_GROUP; only released when
 *                the underlying snvme fd is closed (cascade) or via
 *                NVM_UNMAP_HOST/DEVICE_MEMORY.
 *
 *   ring_*    -> kernel registers as NVM_MAP_KIND_RING_SQ or RING_CQ,
 *                attached to the supplied group_id.  Released by
 *                NVM_DESTROY_QUEUE_GROUP cascade (LIFO with the user
 *                queues created on top of it).
 *
 * Both families set struct nvm_ioctl_map.ioq_idx / is_cq to -1 at ioctl
 * time (the new-mode invariant: NVM_ADD_USER_QUEUE establishes per-queue
 * identity, not the map ioctl).
 *
 * Constraints (kernel-enforced):
 *   - ring_* requires group_id != 0; passing 0 will fail at the kernel
 *     side because RING_SQ/CQ must be attached to a group.
 *   - data_* requires the caller to keep the originating fd open for
 *     the entire lifetime of the buffer; closing the fd cascade-frees
 *     the map.
 * =================================================================== */

int nvm_dma_map_data_host(nvm_dma_t** handle,
                          const nvm_ctrl_t* ctrl,
                          void* vaddr,
                          size_t size);

int nvm_dma_map_data_device(nvm_dma_t** handle,
                            const nvm_ctrl_t* ctrl,
                            void* devptr,
                            size_t size);

int nvm_dma_map_ring_host(nvm_dma_t** handle,
                          const nvm_ctrl_t* ctrl,
                          uint32_t group_id,
                          void* vaddr,
                          size_t size,
                          int is_cq);

int nvm_dma_map_ring_device(nvm_dma_t** handle,
                            const nvm_ctrl_t* ctrl,
                            uint32_t group_id,
                            void* devptr,
                            size_t size,
                            int is_cq);



#if ( !defined( __CUDA__ ) && !defined( __CUDACC__ ) ) && ( defined (__unix__) )
/* 
 * Short-hand function for allocating a page aligned buffer and mapping it 
 * for the controller.
 *
 * Note: this function will not work if you are using the CUDA API
 */
int nvm_dma_create(nvm_dma_t** map, const nvm_ctrl_t* ctrl, size_t size);
#endif


#ifdef __cplusplus
}
#endif

#endif /* __NVM_DMA_H__ */

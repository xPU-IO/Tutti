#ifndef __NVM_TYPES_H__
#define __NVM_TYPES_H__
// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include "ioctl.h"
#include <stddef.h>
#include <stdint.h>

/*
 * CUDA-vs-C compilation gate.
 *
 * libnvm's queue / pad structs use `cuda::atomic<uint32_t, ...>` for
 * the GPU-side SQ/CQ producer-consumer locks.  Those types are only
 * available under nvcc; pure C consumers (host-only smokes, gRPC
 * services that just need ioctl plumbing, etc.) cannot include
 * <cuda/atomic>.
 *
 * To keep the public header reachable from both worlds we provide an
 * ABI-compatible C view: `cuda::atomic<uint32_t, ...>` is a 4-byte,
 * 4-byte-aligned scalar at the binary level, and `__align__(N)` is
 * nvcc-only syntactic sugar for `__attribute__((aligned(N)))`.  The
 * resulting C-visible structs have IDENTICAL sizeof / offsetof to the
 * CUDA versions, so userspace ioctl glue compiles either way and the
 * GPU side keeps full atomicity semantics where it matters.
 *
 * Callers that actually need to *operate* on the atomic fields must
 * still be compiled under nvcc -- this gate just makes the header
 * usable for binary-layout consumers (size, offset, ioctl pointers).
 */
#if defined(__CUDACC__)
  #include <tutti/cuda_like.h>
  typedef cuda::atomic<uint32_t, cuda::thread_scope_device> nvm_atomic_u32_dev_t;
  typedef cuda::atomic<uint32_t, cuda::thread_scope_system> nvm_atomic_u32_sys_t;
#else
  /* Plain-C fallback: same size / alignment as cuda::atomic<uint32_t>. */
  typedef uint32_t __attribute__((aligned(4))) nvm_atomic_u32_dev_t;
  typedef uint32_t __attribute__((aligned(4))) nvm_atomic_u32_sys_t;
  /* __align__(N) is an nvcc keyword; map it to the gcc/clang attribute
   * when seen from a plain-C compile so the rest of this header parses. */
  #ifndef __align__
    #define __align__(N) __attribute__((aligned(N)))
  #endif
#endif



/*
 * Device descriptor
 */
struct device
{
    int fd_control; /* ioctl file descriptor for snvme_control*/
    int fd_dev; /* ioctl file descriptor for snvme*/
    
};






/*
 * NVM admin queue-pair reference handle.
 *
 * As only a single process can be responsible of resetting the controller and
 * setting administration queues, this structure represents a remote handle to
 * that process. It is used as a descriptor for executing RPC calls to the 
 * remote process owning the admin queues.
 *
 * Note: This structure will be allocated by the API and needs to be released
 *       by the API.
 */
struct nvm_admin_reference;
typedef struct nvm_admin_reference* nvm_aq_ref;



/*
 * DMA mapping descriptor.
 *
 * This structure describes a region of memory that is accessible for the
 * NVM controller using DMA. The API assumes a continuous virtual memory
 * address, but the physical pages do not need to be contiguous.
 *
 * The structure contains a variably sized array of bus addresses that maps
 * to the physical memory pages. The user should therefore not create a local
 * instance of this descriptor, but rather rely on the API to allocate and
 * instantiate members.
 *
 * Note: Only page-aligned addresses are supported in NVM Express
 *
 * Note: This structure will be allocated by the API and needs to be released
 *       by the API.
 */
typedef struct
{
    void*                   vaddr;          // Virtual address to start of region (NB! can be NULL)
    int8_t                  local;          // Is this local memory
    int8_t                  contiguous;     // Is memory contiguous
    size_t                  page_size;      // Controller's page size (MPS)
    size_t                  n_ioaddrs;      // Number of MPS-sized pages
    uint64_t                ioaddrs[];      // Physical/IO addresses of the memory pages
}  nvm_dma_t;



typedef nvm_atomic_u32_dev_t padded_struct_pc;

#define CACHELINE_SIZE (128)

#define STATES_PER_CACHELINE (CACHELINE_SIZE/sizeof(padded_struct_pc))
/* typedef struct __align__(32) */
/* { */
/*     cuda::atomic<uint32_t, cuda::thread_scope_device>  val; */
/* //    uint8_t pad[32-4]; */
/* } __attribute__((aligned (32))) padded_struct_pc; */


typedef struct __align__(32)
{
    nvm_atomic_u32_dev_t  val;
    //uint8_t pad[32-8];
} __attribute__((aligned (32))) padded_struct;

/* typedef struct __align__(32) */
/* { */
/*     cuda::atomic<uint32_t, cuda::thread_scope_system>  val; */
/*     //uint8_t pad[32-8]; */
/* } __attribute__((aligned (32))) padded_struct_pc; */


/* typedef struct __align__(32) */
/* { */
/*     cuda::atomic<uint64_t, cuda::thread_scope_system>  val; */
/*     uint8_t pad[32-8]; */
/* } __attribute__((aligned (32))) padded_struct; */

/* 
 * NVM queue descriptor.
 *
 * This structure represents an NVM IO queue and holds information 
 * about memory addresses, queue entries as well as a memory mapped pointer to 
 * the device doorbell register. Maximum queue size is limited to a single 
 * page.
 *
 * Note: This descriptor represents both completion and submission queues.
 */
typedef struct
{
    nvm_atomic_u32_dev_t head_lock;
    uint8_t pad0[28];
    nvm_atomic_u32_dev_t tail_lock;
    uint8_t pad1[28];
    nvm_atomic_u32_dev_t head;
    uint8_t pad2[28];
    nvm_atomic_u32_dev_t tail;
    uint8_t pad3[28];
    nvm_atomic_u32_sys_t tail_copy;
    uint8_t pad4[28];
    nvm_atomic_u32_sys_t head_copy;
    uint8_t pad5[28];

    /* padded_struct<cuda::atomic<uint32_t, cuda::thread_scope_system>> head; */
    /* padded_struct<cuda::atomic<uint32_t, cuda::thread_scope_system>> tail; */
    nvm_atomic_u32_dev_t in_ticket;
    uint8_t pad6[28];
    nvm_atomic_u32_dev_t cid_ticket;
    //uint8_t pad7[28];
    padded_struct* tickets;

    padded_struct* head_mark;
    padded_struct* tail_mark;
    padded_struct* cid;
    padded_struct* pos_locks;

    uint16_t* clean_cid;
    uint32_t qs_minus_1;
    uint32_t qs_log2;
    uint16_t                no;             // Queue number (must be unique per SQ/CQ pair) 0 ~ max_user_num
    uint16_t                es;             // Queue entry size
    uint32_t                qs;             // Queue size (number of entries)
    //uint16_t                head;           // Queue's head pointer
    //uint16_t                tail;           // Queue's tail pointer
    int8_t                  phase;          // Current phase tag
    int8_t                  local;          // Is the queue allocated in local memory
    uint32_t                last;           // Used internally to check db writes
    volatile uint32_t*      db;             // Pointer to doorbell register (NB! write only)
    volatile void*          vaddr;          // Virtual address to start of queue memory
    uint64_t                ioaddr;         // Physical/IO address to start of queue memory
} nvm_queue_t;



/*
 * Convenience type for representing a single-page PRP list.
 */
typedef struct __align__(32)
{
    volatile void*          vaddr;          // Virtual address to memory page
    int16_t                 local;          // Indicates if the page is local memory
    size_t                  page_size;      // Page size
    uint64_t                ioaddr;         // Physical/IO address of memory page
} __attribute__((aligned (32))) nvm_prp_list_t;



/* 
 * NVM completion queue entry type (16 bytes) 
 */
typedef struct __align__(16) 
{
    volatile uint32_t                dword[4];       // The name DWORD is chosen to reflect the specification
} __attribute__((aligned (16))) nvm_cpl_t;



/* 
 * NVM command queue entry type (64 bytes) 
 */
typedef struct __align__(64) 
{
    uint32_t                dword[16];
} __attribute__((aligned (64))) nvm_cmd_t;



/*
 * Controller information structure.
 *
 * Holds information about an NVM controller retrieved from reading on-board
 * registers and running an IDENTIFY CONTROLLER admin command.
 */
struct nvm_ctrl_info
{
    uint32_t                nvme_version;   // NVM Express version number
    size_t                  page_size;      // Memory page size used by the controller (MPS)
    size_t                  db_stride;      // Doorbell stride (DSTRD)
    uint64_t                timeout;        // Controller timeout in milliseconds (TO)
    int                     contiguous;     // Contiguous queues required (CQR)
    uint16_t                max_entries;    // Maximum queue entries supported (MQES)
    uint8_t                 pci_vendor[4];  // PCI vendor and subsystem vendor identifier
    char                    serial_no[20];  // Serial number (NB! not null terminated)
    char                    model_no[40];   // Model number (NB! not null terminated)
    char                    firmware[8];    // Firmware revision
    size_t                  max_data_size;  // Maximum data transfer size (MDTS)
    size_t                  max_data_pages; // Maximum data transfer size (in controller pages)
    size_t                  cq_entry_size;  // CQ entry size (CQES)
    size_t                  sq_entry_size;  // SQ entry size (SQES)
    size_t                  max_out_cmds;   // Maximum outstanding commands (MAXCMD)
    size_t                  max_n_ns;       // Maximum number of namespaces (NN)
};



/*
 * Namespace information structure.
 *
 * Holds informaiton about an NVM namespace.
 */
struct nvm_ns_info
{
    uint32_t                ns_id;          // Namespace identifier
    size_t                  size;           // Size in logical blocks (NSZE)
    size_t                  capacity;       // Capacity in logical blocks (NCAP)
    size_t                  utilization;    // Utilization in logical blocks (NUSE)
    size_t                  lba_data_size;  // Logical block size (LBADS)
    size_t                  metadata_size;  // Metadata size (MS)
};

struct disk_info
{
    uint32_t                ns_id;          // Namespace identifier
    size_t                  size;           // Size in logical blocks (NSZE)
    size_t                  capacity;       // Capacity in logical blocks (NCAP)
    size_t                  utilization;    // Utilization in logical blocks (NUSE)
    size_t                  lba_data_size;  // Logical block size (LBADS)
    size_t                  metadata_size;  // Metadata size (MS)
};
/* Memory descriptor */
struct buffer
{
    void*                   buffer;
    nvm_dma_t*              dma;
};


/* Queue descriptor */
struct queue
{
    struct buffer           qmem;
    nvm_queue_t             queue;
    size_t                  counter;
};

/* 
 * NVM controller handle.
 *
 * Note: This structure will be allocated by the API and needs to be
 *       released by the API.
 */
typedef struct
{
    size_t                  page_size;      // Memory page size used by the controller (MPS)
    uint8_t                 dstrd;          // Doorbell stride (in encoded form)
    uint64_t                timeout;        // Controller timeout in milliseconds (TO)
    uint32_t                max_qs;         // Maximum queue entries supported (MQES)
    size_t                  mm_size;        // Size of memory-mapped region
    volatile void*          mm_ptr;         // Memory-mapped pointer to BAR0 of the physical device
    uint32_t                cq_num;         // Num of cq queues
    uint32_t                sq_num;         // Num of sq queues, the cq_idx is  sq_num/cq_num
    uint32_t                qs;             // queue size, number of entries in a queue
    uint32_t                nr_user_q;
    uint32_t                start_cq_idx;    
    uint32_t                on_host;        // 1 indicate qmem created on cpu, otherwise on GPU
    struct queue*           queues;   
    struct pci_device_addr  pdev_addr;

    /* === Queue metadata sourced from NVM_GET_DEV_INFO ===
     *
     * These mirror the like-named fields in struct nvm_ioctl_dev and are
     * the single source of truth for q_depth / user QID pool / BAR0 size
     * / SGL support, replacing the historical pattern of recomputing them
     * from CAP at libnvm bring-up time.  Populated by ioctl_get_dev_info()
     * after SNVM_DEVICE_BIND succeeds; zero before bind.
     */
    uint16_t                q_depth;             // applies to ALL user queues
    uint32_t                bar0_size;           // pci_resource_len(BAR0)
    uint32_t                max_user_qid;        // top of user QID pool
    uint32_t                max_queues_per_group;// echoes NVM_MAX_QUEUES_PER_GROUP
    uint32_t                sgl_supported;       // Identify Controller SGLS dword

    /* True only when this process registered mm_ptr with the accelerator
     * runtime.  Owner-only daemon handles leave this false; standalone GPU
     * owners and GPU clients set it after cudaHostRegister succeeds. */
    uint8_t                 bar0_cuda_registered;
} nvm_ctrl_t;

/* Disk descriptor */
struct disk
{
    size_t      page_size;
    size_t      max_data_size;
    uint32_t    ns_id;
    size_t      block_size;     // Namespace logical block size in bytes (1 << ns->lba_shift)
    char       disk_name[32];
};


//#ifndef __CUDACC__
//#undef __align__
//#endif

#endif /* __NVM_TYPES_H__ */

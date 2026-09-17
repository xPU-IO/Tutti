/*
 * tutti/include/uapi/tutti_snvme.h
 *
 * Shared UAPI header for the snvme kernel module and libnvm userspace.
 *
 * BOTH the kernel module (snvme-5.15.0-public/pci.c,
 * snvme-5.4.241-1-tlinux4-0017/pci.c)
 * AND userspace (libnvm/src/linux/device.cpp, etc.) include THIS FILE
 * as the single source of truth for all ioctl command numbers, struct
 * layouts, and ABI version/capability negotiation.
 *
 * Design rules (Roadmap Phase 3):
 *   - All integer types are fixed-width (uint8_t/uint16_t/uint32_t/uint64_t/int32_t).
 *     No long, size_t, or pointer-width-dependent types in any UAPI struct.
 *   - Every ioctl struct has _Static_assert locking sizeof and key field offsets.
 *   - TUTTI_SNVME_ABI_VERSION + capability bitmask are defined here; the kernel
 *     reports them via NVM_GET_DEV_INFO (in repurposed reserved fields), and
 *     userspace checks them fail-closed before proceeding.
 *   - 32/64-bit compat strategy is documented inline; the actual compat
 *     conversion layer is deferred to Session 4.
 *
 * Original definitions were migrated from libnvm/include/ioctl.h.
 * ioctl.h now contains only #include "tutti_snvme.h" for backward compat.
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#ifndef _TUTTI_SNVME_UAPI_H
#define _TUTTI_SNVME_UAPI_H

/* ===========================================================================
 * Type includes — kernel vs userspace
 * ===========================================================================
 *
 * In the kernel, <linux/types.h> provides uint8_t/uint16_t/uint32_t/uint64_t/
 * int32_t.  In userspace, <stdint.h> and <stddef.h> provide them.
 *
 * The _IOW/_IOR/_IOWR macros come from <asm/ioctl.h> (kernel) or
 * <asm-generic/ioctl.h> (userspace).
 */
#ifdef __KERNEL__
#  include <linux/types.h>
#  include <asm/ioctl.h>
#else
#  include <stdint.h>
#  include <stddef.h>
#  include <asm-generic/ioctl.h>
#endif

/* ===========================================================================
 * static_assert compatibility
 * ===========================================================================
 *
 * In C11, _Static_assert is the keyword.  In C++11+, static_assert is the
 * keyword.  GCC supports _Static_assert in C++ as an extension, but standard
 * C++17 mode does not.  Use the appropriate keyword per language.
 */
#ifdef __cplusplus
#  define TUTTI_SNVME_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#  define TUTTI_SNVME_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

/* ===========================================================================
 * ABI version and capability bitmask
 * ===========================================================================
 *
 * TUTTI_SNVME_ABI_VERSION is bumped whenever a struct layout changes or a
 * new ioctl command is added.  Both kernel and userspace include this
 * header, so at compile time they agree on the version.  At runtime, the
 * kernel reports its compiled-in version via NVM_GET_DEV_INFO
 * (nvm_ioctl_dev::abi_version), and userspace compares it against its own
 * TUTTI_SNVME_ABI_VERSION.  Mismatch => fail-closed (ENODEV/EPERM).
 *
 * Version history:
 *   0  — pre-UAPI consolidation (implicit; old kernels report 0 in the
 *        reserved field because memset zeroes it).
 *   1  — first consolidated UAPI header (this file).  Fixed-width types,
 *        static_assert layout locks, capability handshake.
 *   2  — raise the per-fd queue-group capacity from 16 to 32 queue pairs;
 *        this enlarges struct nvm_ioctl_add_user_queue and changes the
 *        _IOC_SIZE encoded in NVM_ADD_USER_QUEUE.
 */
#define TUTTI_SNVME_ABI_VERSION  2u

/*
 * Capability bitmask reported by the kernel in nvm_ioctl_dev::capabilities.
 * Userspace can check individual bits to decide whether a feature is
 * available on the running kernel without probing ioctls.
 *
 * A capability bit = 0 means the kernel does NOT support the feature.
 * Userspace MUST treat unknown capability bits as "unsupported" (forward-
 * compat: new userspace on old kernel).
 */
#define TUTTI_SNVME_CAP_QUEUE_GROUPS    (1u << 0)  /* NVM_CREATE_QUEUE_GROUP / NVM_DESTROY_QUEUE_GROUP */
#define TUTTI_SNVME_CAP_USER_QUEUES     (1u << 1)  /* NVM_ADD_USER_QUEUE */
#define TUTTI_SNVME_CAP_RAW_ADMIN       (1u << 2)  /* NVM_RAW_ADMIN_CMD */
#define TUTTI_SNVME_CAP_KERNEL_IOQ_CAP  (1u << 3)  /* NVM_SET_KERNEL_IOQ_CAP */
#define TUTTI_SNVME_CAP_MAP_KIND_TAG    (1u << 4)  /* B6 map_kind discrimination (RING_SQ/CQ/DATA) */

/*
 * Convenience: the capability set supported by THIS header version.
 * The kernel OR's in the bits it actually implements; userspace checks
 * that the bits it needs are present.
 */
#define TUTTI_SNVME_CAP_ALL  ( \
      TUTTI_SNVME_CAP_QUEUE_GROUPS    \
    | TUTTI_SNVME_CAP_USER_QUEUES     \
    | TUTTI_SNVME_CAP_RAW_ADMIN       \
    | TUTTI_SNVME_CAP_KERNEL_IOQ_CAP  \
    | TUTTI_SNVME_CAP_MAP_KIND_TAG    )

/* ===========================================================================
 * Shared constants
 * ===========================================================================
 */

#define NVM_IOCTL_TYPE          0x80
#define NVM_CTRL_IOCTL_TYPE     0x90
/* Backward compat: the original header had a typo (TYOE).  Keep the old
 * name as an alias so any out-of-tree code referencing it still compiles.
 * The ioctl command numbers depend on the value (0x90), not the name. */
#define NVM_CTRL_IOCTL_TYOE     NVM_CTRL_IOCTL_TYPE

#define DISK_NAME_LEN           32

#define DISK_NAME_COPY(dest, src) \
    memcpy(dest, src, DISK_NAME_LEN)

/* ===========================================================================
 * UAPI structs
 * ===========================================================================
 *
 * MIGRATION NOTE (size_t → uint64_t, int → int32_t, uint64_t* → uint64_t):
 *
 *   On LP64 (the ONLY platform this module supports — CUDA requires 64-bit),
 *   size_t == unsigned long == uint64_t (8 bytes), int == int32_t (4 bytes),
 *   and uint64_t* == void* == 8 bytes.  Therefore the binary layout of every
 *   struct is IDENTICAL before and after this change.  The _Static_assert
 *   below locks the LP64 layout; if any assertion fails the build stops,
 *   proving the migration broke something.
 *
 *   On ILP32 (32-bit, NOT supported), size_t would be 4 bytes and pointers
 *   would be 4 bytes, making the layout different.  This is a compat risk
 *   documented in the 32/64-bit compat strategy section below.  The actual
 *   compat translation is Session 4 work.
 */

/* -------------------------------------------------------------------------
 * enum nvm_map_kind — map_kind tag for nvm_ioctl_map (B6).
 * ------------------------------------------------------------------------- */
enum nvm_map_kind {
    NVM_MAP_KIND_UNSPECIFIED = 0,   /* legacy / pre-B6 binary       */
    NVM_MAP_KIND_RING_SQ     = 1,   /* user IO Submission Queue ring */
    NVM_MAP_KIND_RING_CQ     = 2,   /* user IO Completion Queue ring */
    NVM_MAP_KIND_DATA        = 3,   /* PRP / SGL data buffer         */
};

/* -------------------------------------------------------------------------
 * struct nvm_ioctl_map — memory map request.
 *
 * Used by NVM_MAP_HOST_MEMORY / NVM_MAP_DEVICE_MEMORY /
 * NVM_MAP_DEVICE_QUEUE_MEMORY.
 *
 * LP64 layout (locked by static_assert below):
 *   offset  0: vaddr_start  (uint64_t, 8)
 *   offset  8: n_pages      (uint64_t, 8)  [was size_t]
 *   offset 16: ioaddrs      (uint64_t, 8)  [was uint64_t*; user-space addr]
 *   offset 24: ioq_idx      (int32_t,  4)  [was int]
 *   offset 28: is_cq        (int32_t,  4)  [was int]
 *   offset 32: group_id     (uint32_t, 4)
 *   offset 36: map_kind     (uint8_t,  1)
 *   offset 37: reserved0[3] (uint8_t,  3)
 *   total: 40 bytes
 *
 * COMPAT NOTE: `ioaddrs` was originally `uint64_t*` (a pointer).  It is now
 * `uint64_t` representing the user-space address as an integer.  On LP64
 * this is the same size (8 bytes).  In 32-bit compat, this field IS 8 bytes
 * in both 32 and 64-bit ABIs (unlike a pointer which would shrink to 4).
 * This is actually a COMPAT IMPROVEMENT — the struct now has the same
 * layout in 32 and 64 bit for this field.
 * ------------------------------------------------------------------------- */
struct nvm_ioctl_map
{
    uint64_t    vaddr_start;
    uint64_t    n_pages;         /* was size_t; LP64: identical layout      */
    uint64_t    ioaddrs;         /* was uint64_t*; LP64: identical layout.
                                  * User-space address of the ioaddrs array.
                                  * Kernel casts via (void __user *)(uintptr_t). */
    int32_t     ioq_idx;         /* was int; legacy mode only; -1 in new mode */
    int32_t     is_cq;           /* was int; legacy mode only; -1 in new mode */
    uint32_t    group_id;        /* 0 = legacy; nonzero = new mode            */
    uint8_t     map_kind;        /* enum nvm_map_kind (B6)                    */
    uint8_t     reserved0[3];    /* MBZ; future extension                     */
};

TUTTI_SNVME_STATIC_ASSERT(sizeof(struct nvm_ioctl_map) == 40,
    "nvm_ioctl_map: size mismatch — LP64 layout broken");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_map, vaddr_start) == 0,
    "nvm_ioctl_map: vaddr_start offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_map, n_pages) == 8,
    "nvm_ioctl_map: n_pages offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_map, ioaddrs) == 16,
    "nvm_ioctl_map: ioaddrs offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_map, ioq_idx) == 24,
    "nvm_ioctl_map: ioq_idx offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_map, group_id) == 32,
    "nvm_ioctl_map: group_id offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_map, map_kind) == 36,
    "nvm_ioctl_map: map_kind offset");

/* -------------------------------------------------------------------------
 * struct nvm_ioctl_dev — per-controller info returned by NVM_GET_DEV_INFO.
 *
 * LP64 layout (locked by static_assert below):
 *   offset  0: nr_user_q           (uint32_t,  4)
 *   offset  4: start_cq_idx        (uint32_t,  4)
 *   offset  8: dstrd               (uint8_t,   1)
 *   offset  9: [7 bytes padding to align uint64_t]
 *   offset 16: max_data_size       (uint64_t,  8)  [was size_t]
 *   offset 24: block_size          (uint64_t,  8)  [was size_t]
 *   offset 32: disk_name[32]       (char,     32)
 *   offset 64: q_depth             (uint16_t,  2)
 *   offset 66: reserved0           (uint16_t,  2)
 *   offset 68: bar0_size           (uint32_t,  4)
 *   offset 72: max_user_qid        (uint32_t,  4)
 *   offset 76: max_queues_per_group(uint32_t,  4)
 *   offset 80: sgl_supported       (uint32_t,  4)
 *   offset 84: abi_version         (uint32_t,  4)  [was reserved1[0]]
 *   offset 88: capabilities        (uint32_t,  4)  [was reserved1[1]]
 *   offset 92: reserved1[3]        (uint32_t, 12)  [was reserved1[2..4]]
 *   total: 104 bytes
 *
 * ABI VERSION HANDSHAKE:
 *   The kernel populates abi_version with TUTTI_SNVME_ABI_VERSION and
 *   capabilities with the supported TUTTI_SNVME_CAP_* bits.  Userspace
 *   checks these after NVM_GET_DEV_INFO; mismatch => fail-closed.
 *
 *   Old kernels (pre-UAPI-consolidation) zero the entire struct via memset
 *   before populating, so abi_version reads back as 0.  Userspace treats
 *   abi_version == 0 as "unknown / legacy" and MUST fail-closed (the old
 *   kernel may have a different struct layout than the new header expects).
 * ------------------------------------------------------------------------- */
struct nvm_ioctl_dev
{
    /* === existing (semantics unchanged) === */
    uint32_t    nr_user_q;          /* legacy NVM_SET_IOQ_NUM result; 0 in new flow */
    uint32_t    start_cq_idx;       /* first QID available to user IOQs */
    uint8_t     dstrd;              /* CAP.DSTRD: doorbell stride exponent */
    uint64_t    max_data_size;      /* was size_t; CTRL.MDTS in bytes */
    uint64_t    block_size;         /* was size_t; Namespace logical block size in bytes: 1 << ns->lba_shift. */
    char        disk_name[DISK_NAME_LEN];

    /* === B3 fields === */
    uint16_t    q_depth;            /* NVMe CAP.MQES + 1, clamped */
    uint16_t    reserved0;          /* MBZ */
    uint32_t    bar0_size;          /* pci_resource_len(pdev, BAR0) */
    uint32_t    max_user_qid;       /* highest QID kernel hands out */
    uint32_t    max_queues_per_group;  /* echoes NVM_MAX_QUEUES_PER_GROUP */
    uint32_t    sgl_supported;      /* Identify Controller SGLS dword */

    /* === ABI handshake (repurposed from reserved1[0..1]) === */
    uint32_t    abi_version;        /* TUTTI_SNVME_ABI_VERSION; 0 = old kernel */
    uint32_t    capabilities;       /* TUTTI_SNVME_CAP_* bitmask */
    uint32_t    reserved1[3];       /* MBZ; future extension */
};

TUTTI_SNVME_STATIC_ASSERT(sizeof(struct nvm_ioctl_dev) == 104,
    "nvm_ioctl_dev: size mismatch — LP64 layout broken");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_dev, nr_user_q) == 0,
    "nvm_ioctl_dev: nr_user_q offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_dev, start_cq_idx) == 4,
    "nvm_ioctl_dev: start_cq_idx offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_dev, dstrd) == 8,
    "nvm_ioctl_dev: dstrd offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_dev, max_data_size) == 16,
    "nvm_ioctl_dev: max_data_size offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_dev, block_size) == 24,
    "nvm_ioctl_dev: block_size offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_dev, disk_name) == 32,
    "nvm_ioctl_dev: disk_name offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_dev, q_depth) == 64,
    "nvm_ioctl_dev: q_depth offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_dev, bar0_size) == 68,
    "nvm_ioctl_dev: bar0_size offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_dev, max_user_qid) == 72,
    "nvm_ioctl_dev: max_user_qid offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_dev, sgl_supported) == 80,
    "nvm_ioctl_dev: sgl_supported offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_dev, abi_version) == 84,
    "nvm_ioctl_dev: abi_version offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_dev, capabilities) == 88,
    "nvm_ioctl_dev: capabilities offset");

/* -------------------------------------------------------------------------
 * struct nvm_queue_group — per-owner partition descriptor for NVM_SET_IOQ_NUM.
 * ------------------------------------------------------------------------- */
struct nvm_queue_group {
    uint32_t    owner_id;    /* opaque tag, typically a GPU id.        */
    uint32_t    count;       /* SQ + CQ entry count for this group     */
    int32_t     numa_node;   /* doc-only hint; kernel does NOT enforce */
    uint32_t    reserved;    /* MBZ.                                   */
};

TUTTI_SNVME_STATIC_ASSERT(sizeof(struct nvm_queue_group) == 16,
    "nvm_queue_group: size mismatch");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_queue_group, owner_id) == 0,
    "nvm_queue_group: owner_id offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_queue_group, count) == 4,
    "nvm_queue_group: count offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_queue_group, numa_node) == 8,
    "nvm_queue_group: numa_node offset");

#define NVM_MAX_QUEUE_GROUPS  8

#define NVM_QUEUE_SETUP_F_ON_HOST   (1U << 0)

/* -------------------------------------------------------------------------
 * struct nvm_ioctl_setup — queue budget for NVM_SET_IOQ_NUM.
 * ------------------------------------------------------------------------- */
struct nvm_ioctl_setup
{
    uint32_t    ioq_num;        /* total user-side IOQ count             */
    uint32_t    flags;          /* NVM_QUEUE_SETUP_F_*.                  */
    uint32_t    cap_kernel_ioq; /* upper bound on kernel-side IOQ count  */
    uint32_t    nr_write;       /* per-BDF override of write_queues      */
    uint32_t    nr_poll;        /* per-BDF override of poll_queues       */
    uint32_t    nr_groups;      /* <= NVM_MAX_QUEUE_GROUPS; 0 = no split */
    uint32_t    reserved[2];    /* MBZ; future extension.                */
    struct nvm_queue_group  groups[NVM_MAX_QUEUE_GROUPS];
};

TUTTI_SNVME_STATIC_ASSERT(sizeof(struct nvm_ioctl_setup) == 160,
    "nvm_ioctl_setup: size mismatch");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_setup, ioq_num) == 0,
    "nvm_ioctl_setup: ioq_num offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_setup, cap_kernel_ioq) == 8,
    "nvm_ioctl_setup: cap_kernel_ioq offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_setup, nr_groups) == 20,
    "nvm_ioctl_setup: nr_groups offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_setup, groups) == 32,
    "nvm_ioctl_setup: groups offset");

/* -------------------------------------------------------------------------
 * struct pci_device_addr — PCI BDF for SNVM_DEVICE_BIND etc.
 *
 * All fields are int32_t (was int).  On LP64, int == int32_t (4 bytes);
 * layout is identical.  sscanf("%x") callers must cast to (unsigned int*)
 * or use SCNx32 format specifiers.
 * ------------------------------------------------------------------------- */
struct pci_device_addr {
    int32_t domain;
    int32_t bus;
    int32_t slot;
    int32_t func;
};

TUTTI_SNVME_STATIC_ASSERT(sizeof(struct pci_device_addr) == 16,
    "pci_device_addr: size mismatch");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct pci_device_addr, domain) == 0,
    "pci_device_addr: domain offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct pci_device_addr, bus) == 4,
    "pci_device_addr: bus offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct pci_device_addr, slot) == 8,
    "pci_device_addr: slot offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct pci_device_addr, func) == 12,
    "pci_device_addr: func offset");

/* -------------------------------------------------------------------------
 * struct nvm_ioctl_raw_admin — raw admin command pass-through.
 * ------------------------------------------------------------------------- */
struct nvm_ioctl_raw_admin {
    uint8_t     sqe[64];        /* in:  one NVMe admin SQE        */
    uint32_t    result_dw0;     /* out: CQE DW0                   */
    uint32_t    result_dw1;     /* out: CQE DW1                   */
    uint16_t    nvme_status;    /* out: CQE DW3[31:17]            */
    uint16_t    reserved0;      /* MBZ                            */
    uint32_t    reserved1[4];   /* MBZ; future expansion          */
};

TUTTI_SNVME_STATIC_ASSERT(sizeof(struct nvm_ioctl_raw_admin) == 92,
    "nvm_ioctl_raw_admin: size mismatch");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_raw_admin, sqe) == 0,
    "nvm_ioctl_raw_admin: sqe offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_raw_admin, result_dw0) == 64,
    "nvm_ioctl_raw_admin: result_dw0 offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_raw_admin, nvme_status) == 72,
    "nvm_ioctl_raw_admin: nvme_status offset");

/* -------------------------------------------------------------------------
 * struct nvm_ioctl_queue_group — per-fd queue group container.
 * ------------------------------------------------------------------------- */
#define NVM_MAX_GROUPS_PER_FD       1
#define NVM_MAX_QUEUES_PER_GROUP   32

struct nvm_ioctl_queue_group {
    uint32_t    group_id;       /* out: kernel-assigned, opaque   */
    uint32_t    flags;          /* MBZ; reserved for future       */
    uint32_t    max_queues;     /* out: per-group queue cap       */
    uint32_t    reserved[5];    /* MBZ                            */
};

TUTTI_SNVME_STATIC_ASSERT(sizeof(struct nvm_ioctl_queue_group) == 32,
    "nvm_ioctl_queue_group: size mismatch");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_queue_group, group_id) == 0,
    "nvm_ioctl_queue_group: group_id offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_queue_group, max_queues) == 8,
    "nvm_ioctl_queue_group: max_queues offset");

/* -------------------------------------------------------------------------
 * struct nvm_user_queue_pair_in / _out — per-queue-pair I/O for
 * NVM_ADD_USER_QUEUE.
 * ------------------------------------------------------------------------- */
struct nvm_user_queue_pair_in {
    uint64_t    sq_vaddr;       /* userspace VA of SQ ring */
    uint64_t    cq_vaddr;       /* userspace VA of CQ ring */
};

TUTTI_SNVME_STATIC_ASSERT(sizeof(struct nvm_user_queue_pair_in) == 16,
    "nvm_user_queue_pair_in: size mismatch");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_user_queue_pair_in, sq_vaddr) == 0,
    "nvm_user_queue_pair_in: sq_vaddr offset");

struct nvm_user_queue_pair_out {
    uint32_t    sq_doorbell_offset;     /* BAR0 byte offset for SQ tail dbl */
    uint32_t    cq_doorbell_offset;     /* BAR0 byte offset for CQ head dbl */
    uint32_t    qid;                    /* informational only */
    uint32_t    reserved;               /* MBZ */
};

TUTTI_SNVME_STATIC_ASSERT(sizeof(struct nvm_user_queue_pair_out) == 16,
    "nvm_user_queue_pair_out: size mismatch");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_user_queue_pair_out, sq_doorbell_offset) == 0,
    "nvm_user_queue_pair_out: sq_doorbell_offset offset");

/* -------------------------------------------------------------------------
 * struct nvm_ioctl_add_user_queue — NVM_ADD_USER_QUEUE payload.
 * ------------------------------------------------------------------------- */
struct nvm_ioctl_add_user_queue {
    /* in */
    uint32_t    group_id;
    uint32_t    nr_pairs;       /* 1..NVM_MAX_QUEUES_PER_GROUP */
    uint32_t    flags;          /* MBZ */
    uint32_t    reserved[5];    /* MBZ */
    struct nvm_user_queue_pair_in   pairs[NVM_MAX_QUEUES_PER_GROUP];

    /* out */
    struct nvm_user_queue_pair_out  out_pairs[NVM_MAX_QUEUES_PER_GROUP];
};

TUTTI_SNVME_STATIC_ASSERT(sizeof(struct nvm_ioctl_add_user_queue) == 1056,
    "nvm_ioctl_add_user_queue: size mismatch");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_add_user_queue, group_id) == 0,
    "nvm_ioctl_add_user_queue: group_id offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_add_user_queue, nr_pairs) == 4,
    "nvm_ioctl_add_user_queue: nr_pairs offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_add_user_queue, pairs) == 32,
    "nvm_ioctl_add_user_queue: pairs offset");
TUTTI_SNVME_STATIC_ASSERT(offsetof(struct nvm_ioctl_add_user_queue, out_pairs) == 544,
    "nvm_ioctl_add_user_queue: out_pairs offset");

/* ===========================================================================
 * Ioctl command numbers
 * ===========================================================================
 *
 * The ioctl type and command-number fields remain stable.  The complete
 * encoded value of NVM_ADD_USER_QUEUE changed in ABI version 2 because its
 * fixed arrays grew from 16 to 32 entries and _IOC_SIZE is part of the ioctl
 * value.  The ABI handshake rejects a stale kernel/userspace pair before
 * userspace attempts this command.
 */

/* NVM_* ioctls — issued on /dev/ssnvme<N> (fd_dev) */
enum nvm_ioctl_type {
    NVM_MAP_HOST_MEMORY             = _IOW(NVM_IOCTL_TYPE, 1, struct nvm_ioctl_map),
    NVM_MAP_DEVICE_MEMORY           = _IOW(NVM_IOCTL_TYPE, 2, struct nvm_ioctl_map),
    NVM_MAP_DEVICE_QUEUE_MEMORY     = _IOW(NVM_IOCTL_TYPE, 3, struct nvm_ioctl_map),
    NVM_UNMAP_HOST_MEMORY           = _IOW(NVM_IOCTL_TYPE, 4, uint64_t),
    NVM_UNMAP_DEVICE_MEMORY         = _IOW(NVM_IOCTL_TYPE, 5, uint64_t),
    NVM_UNMAP_DEVICE_QUEUE_MEMORY   = _IOW(NVM_IOCTL_TYPE, 6, uint64_t),
    NVM_SET_IOQ_NUM                 = _IOW(NVM_IOCTL_TYPE, 7, struct nvm_ioctl_setup),
    NVM_SET_SHARE_REG               = _IOW(NVM_IOCTL_TYPE, 8, struct nvm_ioctl_dev),
    NVM_GET_DEV_INFO                = _IOR(NVM_IOCTL_TYPE, 9, struct nvm_ioctl_dev),
    NVM_CLEAR_IOQ_NUM               = _IOW(NVM_IOCTL_TYPE, 10, struct nvm_ioctl_dev),
    NVM_RAW_ADMIN_CMD               = _IOWR(NVM_IOCTL_TYPE, 11, struct nvm_ioctl_raw_admin),
    NVM_CREATE_QUEUE_GROUP          = _IOWR(NVM_IOCTL_TYPE, 12, struct nvm_ioctl_queue_group),
    NVM_DESTROY_QUEUE_GROUP         = _IOW (NVM_IOCTL_TYPE, 13, uint32_t),
    NVM_ADD_USER_QUEUE              = _IOWR(NVM_IOCTL_TYPE, 14, struct nvm_ioctl_add_user_queue),
    NVM_SET_KERNEL_IOQ_CAP          = _IOW (NVM_IOCTL_TYPE, 15, uint32_t),
};

/* SNVM_* ioctls — issued on /dev/snvm_control (fd_control) */
enum snvm_ctrl_ioctl_type {
    SNVM_DEVICE_BIND                = _IOW(NVM_CTRL_IOCTL_TYPE, 1, struct pci_device_addr),
    SNVM_DEVICE_UNBIND              = _IOW(NVM_CTRL_IOCTL_TYPE, 2, struct pci_device_addr),
    SNVM_CHRDEV_CREATE              = _IOWR(NVM_CTRL_IOCTL_TYPE, 3, struct pci_device_addr),
    SNVM_CHRDEV_REMOVE              = _IOW(NVM_CTRL_IOCTL_TYPE, 4, struct pci_device_addr),
};

/* ===========================================================================
 * 32/64-bit compat strategy (documentation only — Session 4 implements)
 * ===========================================================================
 *
 * GOAL: a 32-bit userspace process must be able to call the snvme ioctls on
 * a 64-bit kernel (and vice versa) without silent data corruption.
 *
 * STRUCT LAYOUT ANALYSIS:
 *
 *   struct nvm_ioctl_map:
 *     All fields are now fixed-width.  `ioaddrs` (was uint64_t*) is now
 *     uint64_t — 8 bytes in BOTH 32 and 64 bit.  No pointer-width fields.
 *     => LAYOUT STABLE in 32/64 compat.  No compat_ptr needed.
 *
 *   struct nvm_ioctl_dev:
 *     All fields are now fixed-width.  `max_data_size` and `block_size`
 *     (were size_t) are now uint64_t — 8 bytes in BOTH 32 and 64 bit.
 *     No pointer-width fields.
 *     => LAYOUT STABLE in 32/64 compat.  No compat_ptr needed.
 *
 *   struct nvm_ioctl_setup, nvm_queue_group:
 *     All fields were already uint32_t/int32_t.  No changes needed.
 *     => LAYOUT STABLE.
 *
 *   struct pci_device_addr:
 *     All fields now int32_t (was int).  4 bytes in both ABIs.
 *     => LAYOUT STABLE.
 *
 *   struct nvm_ioctl_raw_admin:
 *     All fields fixed-width.  => LAYOUT STABLE.
 *
 *   struct nvm_ioctl_queue_group:
 *     All fields uint32_t.  => LAYOUT STABLE.
 *
 *   struct nvm_user_queue_pair_in / _out:
 *     All fields uint32_t/uint64_t.  => LAYOUT STABLE.
 *
 *   struct nvm_ioctl_add_user_queue:
 *     All fields fixed-width.  => LAYOUT STABLE.
 *
 * compat_ptr HANDLING:
 *
 *   No struct contains embedded user pointers (the `ioaddrs` field in
 *   nvm_ioctl_map is now a uint64_t integer, not a pointer).  Therefore
 *   no ioctl requires compat_ptr() translation.  The kernel's
 *   nvme_to_user_ptr() pattern (used in standard NVMe ioctl.c) is not
 *   needed for snvme-specific ioctls.
 *
 *   The kernel casts `ioaddrs` via (void __user *)(uintptr_t), which is
 *   safe because compat_uptr_t fits in uintptr_t on all supported arches.
 *
 * REMAINING COMPAT RISK (for Session 4):
 *
 *   1. NVM_SET_SHARE_REG uses struct nvm_ioctl_map as its ioctl arg but
 *      only populates ioq_idx.  The _IOC_SIZE is sizeof(nvm_ioctl_map)
 *      = 40 bytes, which is the same in 32 and 64 bit.  No issue.
 *
 *   2. NVM_UNMAP_* use uint64_t as the ioctl arg (a vaddr).  _IOC_SIZE
 *      is 8 bytes in both ABIs.  No issue.
 *
 *   3. The standard NVMe ioctls (NVME_IOCTL_*, handled in ioctl.c) have
 *      their own compat handling (COMPAT_FOR_U64_ALIGNMENT /
 *      nvme_user_io32).  These are NOT snvme-specific and are already
 *      handled by the upstream kernel compat infrastructure.  Session 4
 *      should verify this is wired up in snvme's compat_ioctl path.
 *
 *   4. A .compat_ioctl file_operations entry must be registered for
 *      snvm_dev_fops and snvm_fops so the kernel's compat syscall path
 *      routes 32-bit calls to the right handler.  This is Session 4 work.
 */

#endif /* _TUTTI_SNVME_UAPI_H */

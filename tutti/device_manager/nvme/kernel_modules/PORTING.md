# SNVMe Kernel Module — Porting Guide

> Audience: developers who want to port the **SNVMe** kernel module
> (`backends/local/kernel_modules/snvme/`) to a different upstream Linux
> kernel version (the current code base tracks the **Linux 5.15 LTS**
> NVMe driver tree). This document is a **prescriptive specification** of
> the modifications SNVMe applies on top of the stock `drivers/nvme/host`
> tree, the kernel/user-space contract it exposes, and the bring-up
> sequence the user-space side (`libnvm`) relies on.
>
> Read it together with:
>
> - `snvme/pci.c`          — SNVMe core (modified `nvme/host/pci.c`)
> - `snvme/ctrl.{h,c}`     — per-PCI controller bookkeeping + chrdev factory
> - `snvme/ioctl.c`        — original NVMe user ioctl path (unchanged)
> - `snvme/Makefile.in`    — out-of-tree build glue (configured by CMake)
> - `../nvme/libnvm/include/ioctl.h` — shared kernel/user UAPI header
> - `../nvme/libnvm/src/linux/device.cpp` — userspace counterpart
> - `../nvme/libnvm/include/ctrl.h` — `Controller` C++ wrapper (entry point)

---

## 1. What SNVMe is, in one paragraph

SNVMe is a fork of the in-tree Linux NVMe host driver that allows
**user-space and CUDA kernels to own NVMe IO submission/completion
queues directly**. The stock NVMe driver allocates SQ/CQ ring buffers
from `dma_alloc_coherent()` (kernel pages). SNVMe instead lets a user
process

1. allocate SQ/CQ memory itself — in **host-pinned memory** or in
   **GPU memory** (via `cudaMalloc` + `nvidia_p2p_get_pages`),
2. hand the IO addresses of those pages to the kernel through
   `ioctl(/dev/snvm_<N>)`,
3. ask the kernel to drive a normal NVMe `Identify`/`Create IO SQ`/
   `Create IO CQ` admin sequence using **the user-supplied memory** as
   the queue backing store,
4. and finally `mmap()` BAR0 of the device so the doorbell registers
   are accessible from CUDA kernels (via `cudaHostRegister`/
   `cudaHostGetDevicePointer`).

Once those steps complete, the same NVMe controller is simultaneously:

- a **regular block device** (`/dev/snvme<X>n<Y>` — note the leading
  `s`, set by the disk-name `sprintf("snvme%dn%d", ...)` in `core.c`
  / `multipath.c`; e.g. `/dev/snvme0n1`),
  mounted by the kernel like any other NVMe SSD, and
- a **direct-IO submission target** for user-space / GPU code, which
  rings the doorbells without round-tripping through the block layer.

This is the same idea behind BaM / GPUDirect Storage; SNVMe is the
in-house implementation that lives next to a stock `nvme` driver
without conflicting with it.

---

## 2. Co-existence with the in-tree `nvme` driver

> **Hard requirement.** SNVMe must coexist with the upstream `nvme`
> driver. The two modules differ only in which PCI device each one is
> currently bound to.

The upstream module provides the symbols `nvme_*` and registers a PCI
driver named `"nvme"`. To avoid clashes:

| Concern                     | Convention                                                |
| --------------------------- | --------------------------------------------------------- |
| Kernel module name          | `snvme.ko`, `snvme-core.ko`                               |
| `pci_driver.name`           | `"snvme"` (`PCI_DRIVER_NAME` in `pci.c`)                  |
| Internal symbol prefix      | `snvm_*` / `s_nvme_*` for anything SNVMe-specific         |
| Functions kept from upstream | rename with `s_` prefix when their semantics changed     |
| chrdev class name           | `"libsnvm helper"` (`DRIVER_NAME` in `pci.c`)             |
| Control device              | `/dev/snvm_control` (single-instance, factory)            |
| Per-controller chrdev       | `/dev/ssnvme<N>` — note the **double 's'** (see `DEV_NAME` / `ctrl_chrdev_create` in `ctrl.c`); used for BAR0 mmap + queue ioctls |
| Block device on success     | `/dev/snvme<X>n<Y>` — single 's', set by the `sprintf("snvme%dn%d", ...)` sites in `core.c` and `multipath.c` (e.g. `/dev/snvme0n1`) |

> **Naming gotcha.** SNVMe exposes **two** `/dev` objects per bound
> NVMe controller, and they look superficially similar:
>
> | Path                | Type      | Created by                                | fops / role                                  |
> | ------------------- | --------- | ----------------------------------------- | -------------------------------------------- |
> | `/dev/ssnvme<N>`    | char dev  | `SNVM_CHRDEV_CREATE` ioctl                | `snvm_dev_fops`: BAR0 mmap, queue ioctls     |
> | `/dev/snvme<X>n<Y>` | block dev | `s_nvme_probe()` → `device_add_disk()`    | regular block layer; what you actually mount |
>
> The names differ by **one letter** (`ssnvme` vs `snvme`). Both are
> namespaced away from the in-tree `nvme` driver: the upstream module
> would create `/dev/nvme<X>n<Y>`, never `/dev/snvme<X>n<Y>`.

When porting:

1. **Do not** rename the file operations vtables that `module_init`
   exports (`snvm_fops`, `snvm_dev_fops`); userspace finds them by
   device name.
2. The `snvme-core.ko` ↔ `snvme.ko` split mirrors the upstream
   `nvme-core.ko` ↔ `nvme.ko` split. Keep the layering — promoting
   helper code from `snvme-core` into `snvme` will cause symbol
   collisions on systems where the in-tree `nvme` is also loaded.
3. Stay strict on `EXPORT_SYMBOL_GPL` selection. If the in-tree
   `nvme-core.ko` has already exported a name SNVMe needs, do **not**
   re-export it from `snvme-core.ko`; rename the SNVMe one instead.
4. **`snvme-rename.sed` only rewrites C identifiers; it cannot touch
   `printf`-style format strings.** Every site that constructs a
   device, IRQ, workqueue, sysfs class or chrdev region name from a
   literal must be renamed by hand. The complete list — re-audit
   after every uplift:

   `core.c` (in `nvme_core_init` and friends):
   - `alloc_workqueue("snvme-wq", ...)`,
     `alloc_workqueue("snvme-reset-wq", ...)`,
     `alloc_workqueue("snvme-delete-wq", ...)`
     — workqueues are exposed under
     `/sys/devices/virtual/workqueue/` because of `WQ_SYSFS`; a
     duplicate name makes `alloc_workqueue()` fail with
     `kobject_add_internal failed for nvme-wq with -EEXIST` and
     `insmod` aborts in `nvme_core_init`.  This is the **first**
     symptom of a baseline that forgot the §2 string-literal
     audit.
   - `alloc_chrdev_region(..., "snvme")`
     — owner tag in `/proc/devices`; does not fail on duplicates,
     but the two modules end up sharing one line and udev rules
     that match on the chrdev name break.
   - `class_create(THIS_MODULE, "snvme")` and
     `class_create(THIS_MODULE, "snvme-subsystem")`
     — sysfs class names under `/sys/class/`; behavior on
     duplicates is kernel-version dependent (silent shared-pointer
     on some, `EEXIST` on others — never rely on either).
   - `dev_set_name(ctrl->device, "snvme%d", ...)` — per-controller
     sysfs name.
   - `dev_set_name(&subsys->dev, "snvme-subsys%d", ...)` — per-
     subsystem sysfs name (the class is already separate, so this
     is for grep-friendliness and uniform "snvme..." output, not
     a hard collision).

   `multipath.c`:
   - `sprintf(disk_name, "snvme%dn%d", ...)` (non-multipath fallback
     and multipath head),
     `sprintf(disk_name, "snvme%dc%dn%d", ...)` (hidden multipath
     leg).
     A leftover `"nvme%dn%d"` collides with the in-tree
     `/dev/nvme0n1`; `device_add_disk()` then fails and probe
     unwinds.

   `nvme.h`:
   - the non-multipath inline fallback for `nvme_set_disk_name()`:
     `sprintf(disk_name, "snvme%dn%d", ...)`.  Same hazard as
     above, only on kernels built without `CONFIG_NVME_MULTIPATH`.

   `pci.c`:
   - `pci_request_irq(..., "snvme%dq%d", ...)` — IRQ description
     string in `/proc/interrupts`; duplicates merely confuse
     debugging, no hard failure.
   - `pci_request_mem_regions(pdev, "snvme")` — `/proc/iomem`
     owner tag; same effect.

   **Do NOT** rename the NVMe wwid prefix used by the `wwid_show`
   sysfs attribute (`"nvme.%04x-..."` in `core.c`).  That prefix is
   part of the NVMe userspace contract (udev / multipath-tools
   matches on it); keep it byte-for-byte identical to upstream.

   Missing any one of the renamable sites above silently
   re-introduces a `/dev`, `/proc/interrupts` or workqueue-sysfs
   collision with the in-tree `nvme.ko`.  This regressed on the
   `snvme-5.4.241-1-tlinux4-0017` baseline initial port (the
   upstream-5.4 literals were carried verbatim) and was only
   caught by trying to `insmod` while `nvme.ko` was already loaded.

---

## 3. The two modifications, in detail

The SNVMe diff against upstream `drivers/nvme/host/` boils down to
**(A) renaming everything to `snvm_*`** and **(B) replacing the
implicit `nvme_probe()`-driven initialization with an explicit,
user-driven bring-up flow**. Section 3.1 covers (A); 3.2 covers (B).

### 3.1 Symbol renames (mechanical)

For every translation unit in `nvme/host/` that SNVMe carries
(`core.c`, `pci.c`, `ioctl.c`, `multipath.c`, `zns.c`, `hwmon.c`,
`fabrics.c`, `rdma.c`, `tcp.c`, `nvme.h`, …):

- Rename the PCI driver registration: `nvme_driver` → `snvme_driver`,
  `.name = "nvme"` → `.name = PCI_DRIVER_NAME` (= `"snvme"`).
- Rename `module_init`/`module_exit` entry points:
  `nvme_init` → keep the name (it is `static`) but make sure it's the
  **SNVMe** init, calling `snvm_cdev_init()` and `pci_register_driver(&snvme_driver)`.
- Rename module-wide globals that the in-tree module also defines
  (e.g. workqueues — `nvme_wq` becomes `s_nvme_wq`, the per-controller
  cache becomes the `ctrl_list` we maintain in `snvme/list.c`).
- Functions whose semantics change (queue allocation, queue pair
  allocation, IRQ setup) are prefixed with `s_` in this fork
  (`s_nvme_alloc_queue`, `s_nvme_setup_io_queues`, …). The upstream
  names are never re-defined in the SNVMe build.

When porting to a newer kernel, do the renames mechanically first,
then re-apply the surgical changes from §3.2.

### 3.2 User-driven bring-up: the **B3 queue-group flow**

> **Status note (2026-05-25, baseline `snvme-5.4.241-1-tlinux4-0017`).**
> SNVMe now supports **two coexisting** bring-up flows.  The legacy
> "B0" flow (`NVM_SET_IOQ_NUM` + `NVM_SET_SHARE_REG`, the
> all-or-nothing pre-bind queue-handover path described in earlier
> revisions of this document) is kept as a back-compat shim only.
> All new code, including every smoke test in §8 and the NVMeService
> daemon, uses the **B3 queue-group flow** described below.  When
> porting, both must keep working, but the B3 flow is the one that
> defines the data-plane and is what the §7.4 verification gate
> exercises end-to-end.

The intuition behind B3:

- **Stock NVMe**: bind PCI device → `nvme_probe()` → `nvme_alloc_queue()`
  uses `dma_alloc_coherent` for every SQ/CQ →
  `nvme_setup_io_queues()` is done before `nvme_probe()` returns.
  Number of IO queues = `num_possible_cpus()`, no headroom for any
  user-allocated share.
- **SNVMe B0** (legacy): PCI device is bound *after* user-space
  pre-registered every IO queue ring.  Probe consumed all of them
  in one shot (`use_sreg && ioq_num == ioq_map_num` gate).  This
  required userspace to know how many queues it wanted before bind
  and made dynamic add/remove impossible.
- **SNVMe B3** (current): PCI device is bound **first**, with the
  controller running plain in-tree-style probe (admin queue +
  kernel IO queues all from `dma_alloc_coherent`).  The kernel IOQ
  count is **capped** by `NVM_SET_KERNEL_IOQ_CAP` *before* bind so
  the controller's MSI-X grant has headroom for a user-side share.
  Then, while the controller is `NVME_CTRL_LIVE` and serving the
  block device normally, userspace can **dynamically allocate and
  release** user IO queues via three new ioctls:
  `NVM_CREATE_QUEUE_GROUP` → `NVM_MAP_*` → `NVM_ADD_USER_QUEUE` and
  the matching `NVM_DESTROY_QUEUE_GROUP` (cascades through every
  user queue + every map registered against the group).

The B3 state machine lives in **two** places:

1.  Per-controller `struct ctrl` (`snvme/ctrl.h`) — bind-level fields
    that survive across queue groups:

    ```c
    unsigned int ioq_num;            /* legacy B0; 0 in B3 flow                */
    unsigned int cq_num;             /* legacy B0; 0 in B3 flow                */
    unsigned int ioq_map_num;        /* legacy B0; 0 in B3 flow                */
    unsigned int use_sreg;           /* legacy B0 gate; 0 in B3 flow           */

    unsigned int ctrl_max_io_queues; /* B3: real controller grant after        */
                                     /* MSI-X negotiation; bound on user pool  */
    struct setup setup;              /* setup.cap_kernel_ioq written by        */
                                     /*   NVM_SET_KERNEL_IOQ_CAP               */

    /* Per-fd queue groups (one fd may own up to NVM_MAX_GROUPS_PER_FD).      */
    struct list_head groups;         /* protected by ctrl->groups_lock         */
    struct mutex     groups_lock;

    /* User QID pool (qids in [start_cq_idx, ctrl_max_io_queues]).            */
    DECLARE_BITMAP(user_qid_bitmap, NVME_MAX_USER_QIDS);
    struct mutex   user_qid_lock;
    unsigned int   user_qid_pool_initialised;
    ```

2.  Per-fd queue group (created on demand by `NVM_CREATE_QUEUE_GROUP`,
    file `snvme/qgroup.c`):

    ```c
    struct queue_group {
        uint32_t           group_id;          /* opaque to userspace          */
        struct nvme_ctrl  *ctrl;
        struct file       *owning_fd;
        struct list_head   maps;              /* every NVM_MAP_* of this gid  */
        struct list_head   queues;            /* every NVM_ADD_USER_QUEUE pair*/
        unsigned int       cur_queues;
        unsigned int       max_queues;        /* = NVM_MAX_QUEUES_PER_GROUP   */
    };
    ```

The B3 hook into the upstream probe path is intentionally minimal:
**no probe-time branch, no `use_sreg` check, no copy out of a map
list during probe.**  The only thing the kernel must do at probe
time is honor `setup.cap_kernel_ioq`:

```c
/* Inside nvme_setup_io_queues, AFTER MSI-X negotiation.            */
nr_io_queues = pci_alloc_irq_vectors_affinity(...);  /* upstream    */
dev->ctrl_max_io_queues = nr_io_queues;              /* B3: record  */

if (ctrl->setup.cap_kernel_ioq &&
    nr_io_queues > ctrl->setup.cap_kernel_ioq) {
    pr_info("snvme: capping kernel-side IOQ count from %u to %u "
            "(ctrl_max=%u, user pool gets [%u..%u])\n",
            nr_io_queues, ctrl->setup.cap_kernel_ioq,
            dev->ctrl_max_io_queues,
            ctrl->setup.cap_kernel_ioq + 1,
            dev->ctrl_max_io_queues);
    nr_io_queues = ctrl->setup.cap_kernel_ioq;
}
/* ... rest of upstream nvme_setup_io_queues runs unchanged.        */
```

That is the entire probe-side delta.  Everything else (Create I/O
SQ/CQ admin commands, queue tear-down via Delete I/O SQ/CQ, doorbell
indexing, MSI-X routing) is reused from upstream by issuing
**`NVM_RAW_ADMIN_CMD`-equivalent** admin SQEs from the kernel side
of `NVM_ADD_USER_QUEUE` / `NVM_DESTROY_QUEUE_GROUP`.

When porting to a newer kernel:

1.  **Find the post-MSI-X-negotiation point in `nvme_setup_io_queues`.**
    This is the only place the cap branch goes.  Don't apply the cap
    earlier (you'll skew the MSI-X allocation) or later (the kernel
    has already allocated tagsets sized for `nr_io_queues`).  In 5.4
    this is between `pci_alloc_irq_vectors_affinity()` and the
    `nvme_set_queue_count()` call; in 5.15+ the helper layout
    changed but the *logical* point is identical — between IRQ
    allocation and tagset sizing.

2.  **Add the `ctrl_max_io_queues` field to `struct nvme_dev` and
    record `nr_io_queues` into it BEFORE applying the cap.** This is
    the value the user QID pool's upper bound is derived from, NOT
    the post-cap `nr_io_queues`.  Skipping this step makes the user
    QID pool empty on hosts where the controller's MSI-X budget is
    smaller than `num_possible_cpus()` (every Intel DC SSD on a
    192-vCPU host hits this), and `NVM_ADD_USER_QUEUE` returns
    `-EAGAIN` from `snvm_user_qid_alloc_locked`.

3.  **Leave the admin queue path on `dma_alloc_coherent`.** Only IO
    queues created via `NVM_ADD_USER_QUEUE` use user-supplied
    memory; the kernel-side IO queues stay on coherent DMA, just
    capped.

4.  **Verify `nvme_dev_add()` / `nvme_alloc_admin_tags()` still produce
    `/dev/snvme<X>n<Y>`.** B3 keeps the disk visible for mount;
    losing it (e.g. by accidentally setting `nr_io_queues = 0` or
    by skipping `nvme_dev_add`) regresses the §2 namespace
    contract.

5.  **Carry the `cursor->page_size` mask in `NVM_ADD_USER_QUEUE`'s
    vaddr lookup.**  The lookup matches a (sq_vaddr, cq_vaddr) pair
    against the group's map list.  Host pages are 4 KiB,
    GPU/p2p pages are 64 KiB, so a single mask cannot cover both —
    use the map's own `page_size` to compute it (see §7.3.1 trap
    "vaddr mask must be page-size adaptive").

6.  **Keep the legacy `use_sreg` branch behind its existing
    `ctrl->ioq_num == ctrl->ioq_map_num && ctrl->use_sreg` gate.**
    B0 callers still exist (older NVMeService binaries during
    rolling upgrade); breaking them gains nothing.  Once both flows
    coexist, the rule "`NVM_SET_KERNEL_IOQ_CAP` is for B3,
    `NVM_SET_IOQ_NUM` is for B0" is what disambiguates which one a
    given `SNVM_DEVICE_BIND` activates.

---

## 4. UAPI surface (kernel ↔ user contract)

This is the public ABI. **Changes here break libnvm and any
NVMeService client.** Header: `backends/local/nvme/libnvm/include/ioctl.h`.

### 4.1 Device files

| Path                | Created by                            | fops                | Purpose                                   |
| ------------------- | ------------------------------------- | ------------------- | ----------------------------------------- |
| `/dev/snvm_control` | `snvm_cdev_init()` (module load)      | `snvm_fops`         | bind/unbind PCI devices, factory of chrdev |
| `/dev/ssnvme<N>`    | `SNVM_CHRDEV_CREATE` ioctl            | `snvm_dev_fops`     | per-controller: BAR0 mmap + queue ioctls   |

The minor `<N>` is allocated from `snvm_chrdev_minor_ida` and is
returned to user-space in `pci_device_addr.domain` of the `_IOWR`
result (see §4.3).

### 4.2 `/dev/snvm_control` ioctls

```c
enum snvm_ctrl_ioctl_type {
    SNVM_DEVICE_BIND        = _IOW(0x90, 1, struct pci_device_addr),
    SNVM_DEVICE_UNBIND      = _IOW(0x90, 2, struct pci_device_addr),
    SNVM_CHRDEV_CREATE      = _IOWR(0x90, 3, struct pci_device_addr),
    SNVM_CHRDEV_REMOVE      = _IOW(0x90, 4, struct pci_device_addr),
};
```

Semantics:

- **`SNVM_CHRDEV_CREATE` / `_REMOVE`**: idempotent; (un)registers
  `/dev/ssnvme<N>` for the BDF described by `pci_device_addr`.
  `_CREATE` writes the allocated minor back into `addr.domain` (this
  field is reused as an out-parameter — see `snvm_chrdev_create` in
  `pci.c`).
- **`SNVM_DEVICE_BIND`**: detaches whatever PCI driver currently owns
  the BDF (typically the in-tree `nvme`), registers `snvme_driver` if
  not already registered, and force-attaches it via
  `driver_attach(&snvme_driver.driver)` (NOT `device_attach()`; see
  §7.3.1).  Triggers `s_nvme_probe()`.  In the **B3 flow** probe
  honours `setup.cap_kernel_ioq` (see §3.2); in the **B0 flow**
  probe additionally honours the `use_sreg` flag and consumes
  pre-registered queue rings.  After this returns, the controller
  is `NVME_CTRL_LIVE` and `/dev/snvme<X>n<Y>` exists.
- **`SNVM_DEVICE_UNBIND`**: counterpart; only succeeds if the device
  is currently bound to `snvme`.

### 4.3 `/dev/ssnvme<N>` ioctls

```c
enum nvm_ioctl_type {
    /* memory map / unmap (host pages, GPU p2p pages, GPU queue pages) */
    NVM_MAP_HOST_MEMORY             = _IOW (0x80,  1, struct nvm_ioctl_map),
    NVM_MAP_DEVICE_MEMORY           = _IOW (0x80,  2, struct nvm_ioctl_map),
    NVM_MAP_DEVICE_QUEUE_MEMORY     = _IOW (0x80,  3, struct nvm_ioctl_map),
    NVM_UNMAP_HOST_MEMORY           = _IOW (0x80,  4, uint64_t),
    NVM_UNMAP_DEVICE_MEMORY         = _IOW (0x80,  5, uint64_t),
    NVM_UNMAP_DEVICE_QUEUE_MEMORY   = _IOW (0x80,  6, uint64_t),

    /* legacy B0 bring-up path (still supported for back-compat) */
    NVM_SET_IOQ_NUM                 = _IOW (0x80,  7, struct nvm_ioctl_setup),
    NVM_SET_SHARE_REG               = _IOW (0x80,  8, struct nvm_ioctl_dev),
    NVM_GET_DEV_INFO                = _IOR (0x80,  9, struct nvm_ioctl_dev),
    NVM_CLEAR_IOQ_NUM               = _IOW (0x80, 10, struct nvm_ioctl_dev),

    /* generic admin SQE forwarder (Delete + Create I/O SQ/CQ recycle,
     * Get Log Page, vendor admin, etc.) */
    NVM_RAW_ADMIN_CMD               = _IOWR(0x80, 11, struct nvm_ioctl_raw_admin),

    /* === B3 queue-group flow (the recommended path) === */
    NVM_CREATE_QUEUE_GROUP          = _IOWR(0x80, 12, struct nvm_ioctl_queue_group),
    NVM_DESTROY_QUEUE_GROUP         = _IOW (0x80, 13, uint32_t),
    NVM_ADD_USER_QUEUE              = _IOWR(0x80, 14, struct nvm_ioctl_add_user_queue),
    NVM_SET_KERNEL_IOQ_CAP          = _IOW (0x80, 15, uint32_t),
};
```

#### 4.3.1 `NVM_MAP_*` — pin user/GPU pages and report IO addresses

```c
enum nvm_map_kind {
    NVM_MAP_KIND_UNSPECIFIED = 0,   /* legacy / pre-B6 binary       */
    NVM_MAP_KIND_RING_SQ     = 1,   /* user IO Submission Queue ring */
    NVM_MAP_KIND_RING_CQ     = 2,   /* user IO Completion Queue ring */
    NVM_MAP_KIND_DATA        = 3,   /* PRP / SGL data buffer         */
};

struct nvm_ioctl_map {
    uint64_t  vaddr_start;   /* userspace VA of the buffer (page-aligned)   */
    size_t    n_pages;       /* host-page count for HOST, GPU-page count    */
                             /*   for DEVICE/DEVICE_QUEUE                   */
    uint64_t *ioaddrs;       /* OUT: kernel writes IO addresses here        */
    int       ioq_idx;       /* legacy B0: >=0 = queue ring, <0 = PRP/data  */
                             /*   B3+: pass -1                              */
    int       is_cq;         /* legacy B0: 1=CQ ring, 0=SQ ring; B3+: -1    */
    uint32_t  group_id;      /* B3: register against this queue group       */
                             /*   (0 = fd-scoped DATA / legacy)             */
    uint8_t   map_kind;      /* B6: enum nvm_map_kind                       */
    uint8_t   reserved0[3];  /* MBZ                                         */
};
```

The kernel pins the pages (via `get_user_pages_fast` for HOST, via
`nvidia_p2p_get_pages` for DEVICE/DEVICE_QUEUE), records the IO
addresses into the user-supplied `ioaddrs[]` array, and links the
resulting descriptor onto two lists; **which secondary list it goes
on depends on `map_kind`**:

| `map_kind`           | secondary list           | lifecycle                                   |
| -------------------- | ------------------------ | ------------------------------------------- |
| `RING_SQ` / `RING_CQ`| `g->maps` of `group_id`  | drained by `NVM_DESTROY_QUEUE_GROUP`        |
| `DATA`               | `own->data_maps` (per-fd)| reaped only on fd close (or `NVM_UNMAP_*`)  |
| `UNSPECIFIED` (= 0)  | `g->maps` if `group_id`  | back-compat; behaves as B2..B5              |
|                      | else legacy global only  |                                             |

Plus, in every case, the descriptor is also threaded onto the
per-controller `host_list` / `device_list` / `device_queue_list`
so legacy `NVM_UNMAP_*` `vaddr → map` lookups still work.

This decouples long-lived data buffers from short-lived queue
groups -- the common pattern of "one DMA pool, many groups" needs
no `NVM_UNMAP_*` between groups:

```
  open(/dev/ssnvme*)
  NVM_MAP_HOST_MEMORY(kind=DATA, big DMA pool, group_id=0)   <-- once
  loop:
      NVM_CREATE_QUEUE_GROUP -> g
      NVM_MAP_HOST_MEMORY(kind=RING_SQ/RING_CQ, group_id=g)   <-- per group
      NVM_ADD_USER_QUEUE
      ... IO ...
      NVM_DESTROY_QUEUE_GROUP                                 <-- destroys
                                                                  rings;
                                                                  data pool
                                                                  untouched
  close(fd)                                                    <-- finally
                                                                   releases the
                                                                   data pool
```

`map_kind` also lets `NVM_ADD_USER_QUEUE` reject mismatched
buffer roles up front (`-EINVAL`) instead of issuing
`Create I/O SQ` against a data buffer's DMA address and
silently corrupting controller state.  See §7.3.1 trap
"`map_kind` enforcement at `NVM_ADD_USER_QUEUE` lookup".

> **B3 alignment rule.** Host pages are 4 KiB (PAGE_SIZE);
> GPU pages are 64 KiB (`GPU_PAGE_SHIFT=16` in `snvme/map.c`).
> `vaddr_start` must be aligned to whichever of those applies.
> The kernel preserves both alignments by storing each map's own
> `page_size`; `NVM_ADD_USER_QUEUE`'s vaddr lookup masks with
> `cursor->page_size` so the same code path can resolve a host
> SQ vaddr in one byte and a GPU SQ vaddr in the next (see §7.3.1).

#### 4.3.2 `NVM_GET_DEV_INFO` — query controller capabilities

```c
struct nvm_ioctl_dev {
    /* legacy fields (semantics unchanged) */
    uint32_t nr_user_q;          /* B0: total user queue count;           */
                                 /* B3: 0 (queues are dynamic post-bind)  */
    uint32_t start_cq_idx;       /* first user-allocatable QID            */
                                 /*   = cap_kernel_ioq + 1 in B3 flow     */
    uint8_t  dstrd;              /* CAP.DSTRD doorbell stride exponent    */
    size_t   max_data_size;      /* CTRL.MDTS in bytes                    */
    size_t   block_size;         /* 1 << ns->lba_shift                    */
    char     disk_name[32];      /* e.g. "snvme0n1"                       */

    /* B3 additions */
    uint16_t q_depth;            /* NVMe CAP.MQES + 1, clamped (per-      */
                                 /*   queue depth; same for every user    */
                                 /*   queue; snvme does not support per-  */
                                 /*   queue depth at B3)                  */
    uint16_t reserved0;
    uint32_t bar0_size;          /* IORESOURCE_MEM #0 length (mmap arg)   */
    uint32_t max_user_qid;       /* upper bound of user QID pool          */
                                 /*   = ctrl_max_io_queues                */
    uint32_t sgl_supported;      /* echo of dev->ctrl.sgls (Identify      */
                                 /*   Controller offset 536, NVMe 1.4)    */
                                 /* bit 0:1 -- transport SGL supported    */
                                 /* bit 1   -- transport SGL with align   */
                                 /* bit 16  -- byte-aligned SGL           */
                                 /* bit 17  -- SGL bit-bucket descriptor  */
                                 /* bit 18  -- SGL MPTR descriptor        */
                                 /* bit 19  -- SGL larger than data xfer  */
                                 /* bit 20  -- transport SGL data block   */
                                 /* bit 21  -- keyed SGL data block       */
                                 /* sgls=0x0 means PRP-only controller    */
    uint32_t reserved1[5];       /* MBZ; future extension                  */
};
```

#### 4.3.3 `NVM_SET_KERNEL_IOQ_CAP` — split MSI-X budget pre-bind (B3)

```c
uint32_t cap = 36;
ioctl(fd, NVM_SET_KERNEL_IOQ_CAP, &cap);
```

Pre-bind, declares an upper bound on how many IO queues the kernel
side may consume from the controller's granted IOQ count.  Whatever
the controller actually grants above this cap becomes the **user
QID pool** that subsequent `NVM_ADD_USER_QUEUE` calls draw from.

This is a **cap-only** path:

- `ctrl->setup.cap_kernel_ioq` is set to the requested value;
- `ctrl->ioq_num`, `ctrl->cq_num`, `ctrl->use_sreg` stay at 0;
- the probe path runs as plain in-tree-style nvme.

Use this when userspace plans to allocate IOQs dynamically post-bind
via `NVM_ADD_USER_QUEUE`, instead of declaring them upfront.  The
cap MUST be issued *before* `SNVM_DEVICE_BIND`; calls after bind
update the field but have no probe to apply it to.

> **MSI-X budget rule.**  Pick `cap` such that
> `cap + Σ(post-bind queue groups)` ≤ controller's MSI-X grant.
> For a 192-vCPU host with an Intel DC SSD (MSI-X = 136), `cap=36`
> leaves `[37..135]` (99 QIDs) for user queue groups; the smoke
> tests use `cap=36`.

#### 4.3.4 `NVM_CREATE_QUEUE_GROUP` / `NVM_DESTROY_QUEUE_GROUP` (B3)

```c
struct nvm_ioctl_queue_group {
    uint32_t group_id;       /* OUT: kernel-assigned, opaque cookie  */
    uint32_t flags;          /* MBZ                                  */
    uint32_t max_queues;     /* OUT: per-group queue cap (= 16)      */
    uint32_t reserved[5];    /* MBZ                                  */
};
```

A "queue group" is the runtime resource container that subsequent
`NVM_ADD_USER_QUEUE` calls hang off of.  It is the kernel-side dual
of "one logical client" — typically one process binding to one GPU
on one NVMe controller.

Lifecycle / ownership:

- Created with `NVM_CREATE_QUEUE_GROUP` on an open `/dev/ssnvme<N>`
  fd.  The kernel returns an opaque, globally unique `group_id`.
  The group is bound to `(file, ctrl)`; **fd close cascades
  destroy automatically**.
- Destroyed with `NVM_DESTROY_QUEUE_GROUP(&group_id)` (the payload
  is just a `uint32_t`).  Cascade order: every (CQ, SQ) pair in
  `g->queues` LIFO → Delete I/O SQ → Delete I/O CQ → free QID;
  then every map in `g->maps` LIFO → `unmap_and_release()`.  At
  group-destroy time the kernel emits:

    ```
    snvme: destroy_qgroup id=N drained K user queue(s)
    snvme: destroy_qgroup id=N drained M map(s)
    ```

  Use these as the cleanup-correctness signature in dmesg.

- Up to `NVM_MAX_GROUPS_PER_FD` groups per fd (currently 1).
- `group_id == 0` is invalid (sentinel meaning "no group").

Why a separate ioctl from `NVM_SET_IOQ_NUM`: the legacy ioctl is a
**bind-time, per-controller** pre-declaration of the entire user
budget.  Queue groups are the **post-bind, per-fd** runtime path:
groups can be created and destroyed at any time after bind,
independently of each other.  The two paths coexist; legacy callers
see no behavioural change.

#### 4.3.5 `NVM_ADD_USER_QUEUE` — create user (SQ, CQ) pairs (B3)

```c
struct nvm_user_queue_pair_in {
    uint64_t sq_vaddr;       /* userspace VA of the SQ ring (already   */
                             /*   registered via NVM_MAP_* against     */
                             /*   the same group_id)                   */
    uint64_t cq_vaddr;       /* same, for the CQ ring                  */
};

struct nvm_user_queue_pair_out {
    uint32_t sq_doorbell_offset;   /* BAR0 byte offset for SQ tail dbl */
    uint32_t cq_doorbell_offset;   /* BAR0 byte offset for CQ head dbl */
    uint32_t qid;                  /* informational; not needed for    */
                                   /*   SQE submission (SQE has no    */
                                   /*   SQID field).  Useful for      */
                                   /*   dmesg correlation.             */
    uint32_t reserved;
};

struct nvm_ioctl_add_user_queue {
    /* in */
    uint32_t group_id;
    uint32_t nr_pairs;       /* 1..NVM_MAX_QUEUES_PER_GROUP (=16)      */
    uint32_t flags;          /* MBZ                                    */
    uint32_t reserved[5];    /* MBZ                                    */
    struct nvm_user_queue_pair_in  pairs[NVM_MAX_QUEUES_PER_GROUP];
    /* out: only pairs[0..nr_pairs-1] populated */
    struct nvm_user_queue_pair_out out_pairs[NVM_MAX_QUEUES_PER_GROUP];
};
```

Submits up to `NVM_MAX_QUEUES_PER_GROUP` (SQ, CQ) pairs in one
ioctl.  The kernel handles them as an **all-or-nothing batch**:
either every pair successfully gets a Create I/O CQ + Create I/O SQ
admin command through the controller, or none do (any pairs already
created in the same call are unwound via Delete I/O SQ + Delete I/O
CQ before the ioctl returns the error).

Add operations are incremental: a group may receive multiple
`NVM_ADD_USER_QUEUE` calls as long as
`cur_queues + nr_pairs <= max_queues`.

Pre-conditions enforced by the kernel:

1. Controller must be bound (`ctrl.state == NVME_CTRL_LIVE`).  The
   user QID pool is only populated once `nvme_probe` finishes
   allocating kernel IOQs — `ADD_USER_QUEUE` before bind returns
   `-ENODEV`.
2. `group_id` must belong to the calling fd (cross-fd usage is
   `-ENOENT`).
3. `nr_pairs` ∈ `[1, NVM_MAX_QUEUES_PER_GROUP]`.
4. `flags` / `reserved` MBZ.
5. Each `(sq_vaddr, cq_vaddr)` must resolve (under the map's own
   `page_size` mask) to exactly one map already registered against
   this group.  The kernel rejects with `-ENOENT` otherwise; rings
   registered against a different group (even on the same fd) are
   not visible.
6. `cur_queues + nr_pairs <= max_queues`.  Returns `-EBUSY` if the
   group is full.
7. No two pairs in one call may share the same sq_vaddr or cq_vaddr.
   `-EINVAL`.

Output: `out_pairs[i]` is populated for `i < nr_pairs` only;
trailing entries are left zero.

Doorbell offset arithmetic (consumed by user/GPU code):

```
sq_doorbell_offset = NVME_REG_DBS + ((qid * 2 + 0) << dstrd);
cq_doorbell_offset = NVME_REG_DBS + ((qid * 2 + 1) << dstrd);
```

Userspace adds these offsets to the `bar0_gpu` / `bar0_cpu` pointer
returned by `mmap()` and writes the new SQ tail / CQ head as a
plain `uint32_t` store.

#### 4.3.6 `NVM_RAW_ADMIN_CMD` — generic admin SQE forwarder

```c
struct nvm_ioctl_raw_admin {
    uint8_t  sqe[64];        /* in:  one NVMe admin SQE              */
    uint32_t result_dw0;     /* out: CQE DW0                         */
    uint32_t result_dw1;     /* out: CQE DW1                         */
    uint16_t nvme_status;    /* out: CQE DW3[31:17] (SC|SCT|...)     */
    uint16_t reserved0;      /* MBZ                                  */
    uint32_t reserved1[4];   /* MBZ; future expansion                */
};
```

Used to drive per-queue **recycle** (Delete + Create I/O SQ/CQ
without dropping the entire group), Abort, Get Log Page, vendor
admin commands, and any other admin-only command that snvme does
not need a dedicated ioctl for.  The kernel re-uses the CID by
handing the SQE to `snvme_submit_sync_cmd`, which manages its own
tag.

### 4.4 `mmap()` on `/dev/ssnvme<N>`

`snvm_dev_fops.mmap = svm_mmap_registers` (in `pci.c`) maps
**BAR0** of the bound NVMe controller into the calling process.
Userspace then uses `cudaHostRegister(..., cudaHostRegisterIoMemory)`
so CUDA kernels can read/write the doorbells directly — see
`cudaHostGetDevicePointer` for the GPU VA conversion.  Note that
`cudaHostRegisterIoMemory` requires a recent CUDA runtime / driver
(verified on H20 with CUDA 13.0).

This is the source of the `mm_ptr` value flowing through
`_nvm_ctrl_init()` and ultimately into each `QueuePair::sq.db` /
`cq.db`.  In B3 the same `mm_ptr` is reused across many
group-create / group-destroy cycles -- BAR0 mapping survives as
long as the fd stays open, even though every doorbell offset
inside it changes when queues are added or destroyed.

---

## 5. Bring-up sequences (canonical flows)

There are **two** canonical flows your port must keep working:

- **§5.1 B3 dynamic queue-group flow** (recommended; what every new
  caller, every smoke test, and every NVMeService client uses).
- **§5.2 B0 legacy pre-bind flow** (back-compat shim; older binaries
  built against pre-B3 libnvm).

### 5.1 B3 dynamic queue-group flow

This is what `snvme_smoke_io.c` (CPU) and `snvme_smoke_gpu.cu` (GPU)
exercise end-to-end, including a 4-round dynamic alloc/free
verification loop in the GPU smoke.  Reference implementations:

- CPU rings:  `backends/local/kernel_modules/test/snvme_smoke_io.c`
- GPU rings:  `backends/local/kernel_modules/test/snvme_smoke_gpu.cu`
- libnvm path: `backends/local/nvme/libnvm/src/...` (B3 wrapper)
- daemon:     `backends/local/NVMeService/src/nvmeservice_state.cu`

```
  USER (libnvm / smoke)                              KERNEL (snvme.ko)
  ---------------------                              -----------------

  --- Phase 0: chrdev factory ---
  open("/dev/snvm_control")
  SNVM_CHRDEV_CREATE(BDF)            ──ioctl──▶      allocate minor N
                                     ◀──return──     /dev/ssnvme<N> exists
                                                     addr.domain := N
  open("/dev/ssnvme<N>")  ⇒ fd_dev

  --- Phase 1: cap kernel queue share BEFORE bind ---
  uint32_t cap = 36;
  ioctl(fd_dev, NVM_SET_KERNEL_IOQ_CAP, &cap)
                                     ──ioctl──▶     setup.cap_kernel_ioq := 36

  --- Phase 2: bind (regular probe path with cap applied) ---
  SNVM_DEVICE_BIND(BDF)              ──ioctl──▶     pci_register_driver(snvme)
                                                    driver_attach(snvme_driver)
                                                    s_nvme_probe():
                                                      ↳ admin queue (kernel DMA)
                                                      ↳ MSI-X negotiation
                                                      ↳ ctrl_max_io_queues := grant
                                                      ↳ if cap < grant:
                                                          nr_io_queues = cap
                                                      ↳ kernel block-mq tagset
                                                      ↳ /dev/snvme<X>n<Y> appears
                                                      ↳ NVME_CTRL_LIVE
                                     ◀──return──    bind ok
                                                    dmesg:
                                                      "capping kernel-side IOQ
                                                       count from G to C
                                                       (ctrl_max=G, user pool
                                                       gets [C+1..G])"
                                                      "K/0/0/U default/read/poll
                                                       /user queues"

  --- Phase 3: query controller params ---
  NVM_GET_DEV_INFO                   ──ioctl──▶
                                     ◀──return──    fills q_depth, block_size,
                                                    start_cq_idx (=cap+1),
                                                    max_user_qid (=ctrl_max),
                                                    bar0_size, sgl_supported,
                                                    disk_name ("snvme0n1")

  --- Phase 4: per-process resource container ---
  NVM_CREATE_QUEUE_GROUP(&req)       ──ioctl──▶     allocate group_id,
                                                    hang on fd_dev's owner list
                                     ◀──return──    req.group_id, max_queues=16

  --- Phase 5: BAR0 for doorbells ---
  void *bar0 = mmap(fd_dev, NULL, bar0_size,
                    PROT_READ|PROT_WRITE, MAP_SHARED, 0);
  cudaHostRegister(bar0, bar0_size,
                   cudaHostRegisterIoMemory)        (only if GPU is the consumer)
  cudaHostGetDevicePointer(&bar0_gpu, bar0, 0)

  --- Phase 6: allocate ring buffers + data buffers ---
  for each (sq_buf[i], cq_buf[i]) in 0..nr_qp-1:
      posix_memalign(..., 4096, q_depth*64)   /* host SQ */
      posix_memalign(..., 4096, q_depth*16)   /* host CQ */
      OR
      cudaMalloc(&sq_buf[i], 65536)           /* GPU SQ, one GPU page */
      cudaMalloc(&cq_buf[i], 65536)           /* GPU CQ, one GPU page */
  posix_memalign / cudaMalloc → wbuf, rbuf, prp_list_w, prp_list_r

  --- Phase 7: register every buffer against the group ---
  for each buffer R:
      NVM_MAP_HOST_MEMORY {              ──ioctl──▶  pin pages (4 KiB),
        .vaddr_start = R, .n_pages = ceil(size/4K),                fill ioaddrs[],
        .ioq_idx = -1, .is_cq = -1, .group_id = group_id }         link onto g->maps
        OR
      NVM_MAP_DEVICE_MEMORY {            ──ioctl──▶  nvidia_p2p_get_pages
        ... .group_id = group_id }                                  (64 KiB pages),
                                                                    fill ioaddrs[],
                                                                    link onto g->maps

  --- Phase 8: create user IO queues for those rings ---
  struct nvm_ioctl_add_user_queue add_req = {
      .group_id = group_id, .nr_pairs = N,
      .pairs[i] = { .sq_vaddr = sq_buf[i], .cq_vaddr = cq_buf[i] },
  };
  NVM_ADD_USER_QUEUE(&add_req)         ──ioctl──▶   for each pair i:
                                                      lookup map by vaddr
                                                        (mask with map's
                                                         own page_size)
                                                      alloc qid from user pool
                                                      Create I/O CQ via admin q
                                                      Create I/O SQ via admin q
                                                    (atomic batch)
                                     ◀──return──   add_req.out_pairs[i]:
                                                     qid, sq_db_offset,
                                                     cq_db_offset

  --- Phase 9: data plane (CPU or GPU) ---
  /* user fills SQE in user-allocated SQ ring,
   * writes (uint32_t*)(bar0 + sq_db_offset) := new_tail,
   * polls cq[head].status phase bit,
   * writes (uint32_t*)(bar0 + cq_db_offset) := new_head.
   * GPU variant uses cudaHostGetDevicePointer(bar0_gpu)
   * + __threadfence_system() between SQE write and doorbell.        */

  --- Phase 10: dynamic teardown (anytime) ---
  NVM_DESTROY_QUEUE_GROUP(&group_id)  ──ioctl──▶   for each (sq, cq) in g LIFO:
                                                     Delete I/O SQ via admin q
                                                     Delete I/O CQ via admin q
                                                     free qid back to user pool
                                                   for each map in g LIFO:
                                                     unmap_and_release()
                                                     /* host: unpin pages */
                                                     /* GPU: nvidia_p2p_put_pages */
                                                   dmesg:
                                                     "destroy_qgroup id=N
                                                      drained K user queue(s)"
                                                     "destroy_qgroup id=N
                                                      drained M map(s)"

  /* The fd is still open; another CREATE_QUEUE_GROUP / MAP / ADD
   * cycle can run immediately, drawing the same QIDs back from
   * the pool.  The smoke `--rounds N` flag verifies this loop.   */

  --- Phase 11: final tear-down ---
  munmap(bar0, bar0_size)
  cudaHostUnregister(bar0)
  SNVM_DEVICE_UNBIND(BDF)             ──ioctl──▶   pci_unregister_driver(snvme)
                                                   /dev/snvme<X>n<Y> disappears
  close(fd_dev)
  SNVM_CHRDEV_REMOVE(BDF)             ──ioctl──▶   release minor N
  close(fd_ctl)
```

> **B3 invariants (your port MUST preserve these):**
>
> 1. `NVM_SET_KERNEL_IOQ_CAP` is **cap-only**.  It does not enable
>    `use_sreg`, does not pre-register any queue, does not require a
>    matching unmap.  Setting it to 0 (or never calling it) MUST
>    leave the probe path byte-for-byte identical to upstream.
>
> 2. `NVM_ADD_USER_QUEUE` MUST resolve `(sq_vaddr, cq_vaddr)`
>    against `g->maps` only — NOT against the global host_list /
>    device_list / device_queue_list.  This is the per-fd isolation
>    rule: one client's group cannot see another client's rings.
>
> 3. The vaddr lookup MUST mask each candidate vaddr with
>    `cursor->page_size`, NOT a single global `PAGE_MASK`.  Host
>    maps' page_size is `PAGE_SIZE` (4 KiB); device maps' page_size
>    is `GPU_PAGE_SIZE` (64 KiB).  See §7.3.1 trap "vaddr-mask must
>    be page-size adaptive".
>
> 4. `NVM_DESTROY_QUEUE_GROUP` MUST drain queues BEFORE maps
>    (the controller still has live SQEs writing to map'd memory
>    until Delete I/O SQ completes), and within each list MUST go
>    LIFO (a Create-CQ-then-Create-SQ atomic batch unwinds as
>    Delete-SQ-then-Delete-CQ).
>
> 5. fd close (`__fput()` → `snvm_dev_fops.release`) MUST cascade a
>    destroy across every group on the closing fd, with the same
>    Delete-I/O-{SQ,CQ}-then-unmap semantics as explicit destroy.
>    This is what makes `kill -9 <smoke>` safe.
>
> 6. `NVM_MAP_KIND_DATA` maps live on the per-fd `own->data_maps`
>    list, NOT on any `g->maps`, and MUST NOT be touched by
>    `NVM_DESTROY_QUEUE_GROUP` / `destroy_qgroup_locked`.  They
>    are reaped only by `snvm_dev_release` (fd close) or by an
>    explicit `NVM_UNMAP_*` from userspace.  This invariant is
>    what lets a long-lived data-buffer DMA pool span many
>    short-lived queue groups without a per-group re-pin (§4.3.1).

### 5.2 B0 legacy pre-bind flow (for back-compat only)

Older libnvm binaries pre-register every queue ring **before** bind
and let probe consume them in one shot.  This still works but is
not the recommended path; new code should use B3.

```
  open("/dev/snvm_control") ; SNVM_CHRDEV_CREATE ; open("/dev/ssnvme<N>")
  mmap BAR0 ; cudaHostRegister
  for each ring R:
      NVM_MAP_HOST_MEMORY(R, ioq_idx=i, is_cq=…)
                                                   /* group_id == 0 → */
                                                   /*   legacy lists  */
  NVM_SET_IOQ_NUM(...)                             /* ctrl->ioq_num   */
  NVM_SET_SHARE_REG(1)                             /* ctrl->use_sreg  */
  SNVM_DEVICE_BIND(BDF)                            /* probe consumes  */
                                                   /*   user rings if */
                                                   /*   ioq_num ==    */
                                                   /*   ioq_map_num   */
                                                   /*   && use_sreg   */
  NVM_GET_DEV_INFO ; mount disk ; ... IO ...
  NVM_CLEAR_IOQ_NUM ; SNVM_DEVICE_UNBIND ; SNVM_CHRDEV_REMOVE
```

> **Invariant.** `SNVM_DEVICE_BIND` in B0 must only be issued
> **after** `ioq_map_num == ioq_num` AND `use_sreg == 1`.  Failing
> this, the kernel still binds, but `s_nvme_probe()` falls back to
> the upstream `dma_alloc_coherent` path and the user's queue
> rings are silently ignored — symptom is "everything looks fine
> but the doorbells don't ring anything".

> **Coexistence rule.** B0 and B3 on the **same fd** are mutually
> exclusive: `use_sreg=1` makes probe consume rings, leaving no
> headroom for B3's `NVM_ADD_USER_QUEUE` to allocate from.  Use
> `NVM_SET_KERNEL_IOQ_CAP` xor `NVM_SET_IOQ_NUM` per fd, never
> both.

---

## 6. Build & install

`Makefile.in` is rendered by the repository-root CMake after it selects both
the kernel baseline and one peer-memory backend. CUDA defaults to `nvidia`;
MUSA/MACA default to `metax`. The equivalent explicit CUDA configuration is:

```bash
cmake --preset cuda-module --fresh \
  -DSNVME_KERNEL_VERSION=5.15.0-public \
  -DTUTTI_P2P_BACKEND=nvidia
cmake --build --preset cuda-module --target modules
```

To invoke the configured Kbuild wrapper directly, keep the same backend
selection:

```bash
cd build/cuda-module/module
make TUTTI_P2P_BACKEND=nvidia
sudo insmod snvme-core.ko
sudo insmod snvme.ko
ls -l /dev/snvm_control          # should appear, mode 0666
```

Module parameters:

| Param            | Default | Meaning                                 |
| ---------------- | ------- | --------------------------------------- |
| `max_num_ctrls`  | 64      | size of the chrdev minor pool           |

Unload:

```bash
sudo rmmod snvme
sudo rmmod snvme_core
```

`rmmod snvme` will:
- clear all outstanding host/device memory mappings,
- `pci_unregister_driver(snvme_driver)`, releasing every BDF it had
  bound, and
- destroy `/dev/snvm_control`.

Any process that still holds `/dev/ssnvme<N>` open will see further
ioctls fail with `-EBADF` because `ctrl_find_by_inode()` returns NULL.

### 6.1 Module signing on locked-down kernels

Deployment kernels that ship with `CONFIG_MODULE_SIG_FORCE=y` (e.g.
TencentOS Server 5.4.241-1-tlinux4-0017) reject every unsigned
module with `Loading of unsigned module is rejected` and
`insmod: ... Required key not available`.  Three observations
matter for porting:

1. `CONFIG_MODULE_SIG_FORCE=y` is a **compile-time** enforcement.
   It cannot be cleared at runtime via `sysctl
   kernel.modules_sig_enforce`, kernel cmdline `module.sig_enforce=0`,
   `insmod --force`, or a Secure Boot toggle — those knobs only
   apply to `CONFIG_MODULE_SIG_FORCE=n` kernels.  Confirm with:

   ```bash
   grep CONFIG_MODULE_SIG_FORCE /boot/config-$(uname -r)
   ```

2. Self-signing requires either
   `CONFIG_SECONDARY_TRUSTED_KEYRING=y` plus a writable secondary
   keyring (Secure Boot + MOK, or `keyctl add asymmetric` if
   integrity policy allows), OR access to the CA whose public
   half is baked into `.builtin_trusted_keys`.  On the TencentOS
   image above neither holds, so the only production path is the
   central signing service (kmod upload → signed kmod download
   → `insmod`).
3. For **active porting work** (editing snvme baselines, running
   the §7.4 verification gate), prefer a development host whose
   running kernel does not set `CONFIG_MODULE_SIG_FORCE=y` (any
   stock mainline kernel, the upstream `temp/kernel-5.4.241-1.0017.7`
   rebuilt with `CONFIG_MODULE_SIG_FORCE=n`, etc.).  The signing
   workflow is a deployment concern, not a porting concern, and
   trying to iterate on snvme with a "edit → build → upload →
   wait → download → insmod → dmesg" loop is impractical.

The same `scripts/sign-file` helper that ships with the kernel
build tree is used in all signing workflows:

```bash
/usr/src/kernels/$(uname -r)/scripts/sign-file \
    sha256 <priv_key.pem> <pub_key.x509> snvme-core.ko
/usr/src/kernels/$(uname -r)/scripts/sign-file \
    sha256 <priv_key.pem> <pub_key.x509> snvme.ko
```

---

## 7. Porting to a new kernel version

> **Reality check.** The in-tree `drivers/nvme/host/` is refactored
> **almost every LTS cycle** — allocator helpers get renamed, `struct
> nvme_dev` gains fields, admin-queue / tagset setup gets reshuffled,
> `blk_alloc_disk()` vs `blk_mq_alloc_disk()` switch in 5.14, queue
> limits in 6.0, mpath hashing in 6.6, ... A three-way `diff` between
> **old upstream / new upstream / SNVMe fork** is a *starting point*,
> not a complete recipe. What diff misses:
>
> - **Struct-layout surgery.** SNVMe injects ~6 fields into
>   `struct nvme_dev` (`nr_user_allocated_queues`,
>   `nr_user_use_cq`, `user_start_qid`, …). When upstream adds its own
>   fields, merge-three-way puts them in adjacent lines but cannot tell
>   you whether the resulting layout is still consistent with every
>   accessor (some are in inline helpers, some in admin path, some in
>   IO path).
> - **Semantic drift inside "unchanged" helpers.** A function whose
>   signature is identical between versions can acquire new implicit
>   preconditions (e.g. "caller must hold `ctrl->lock`", or "tagset
>   must already be live"). Diff stays silent; you debug at runtime.
> - **Helper refactors that move your hook point.** If upstream
>   replaces three `dma_alloc_coherent` call sites with a single
>   `nvme_alloc_queue_mem()`, the SNVMe `use_sreg` branch has to follow
>   the new split point — sometimes by hooking inside the new helper,
>   sometimes by not calling it at all.
> - **Lock-order / flow changes.** `nvme_reset_work` and `nvme_probe`
>   in particular are re-ordered every other release. A patch that
>   re-applies cleanly can still run with the wrong IRQ set up.
>
> Treat the steps below as the minimum; the **Phase 3 semantic audit**
> is the one people skip and regret.

### 7.1 Phase 1 — Mechanical merge (this is what `diff` gets you)

- [ ] **Anchor the old baseline.** SNVMe carries two reference
      baselines side-by-side under `backends/local/kernel_modules/`:
      `snvme-5.4.241-1-tlinux4-0017/` (the production TLinux
      kernel; upstream tag link is in the repo root `README.md`)
      and `snvme-5.15.0-public/` (cross-LTS reference, vanilla
      v5.15-class).  Pick whichever is closer to your target and
      treat it as the merge **left side**.  The corresponding
      stock upstream tree (without SNVMe modifications) is your
      merge **base**: for the TLinux baseline that lives under
      `temp/kernel-...` in this repo (see README); for the public
      baseline, check out `v5.15` of the upstream Linux tree.
- [ ] **Sync the new baseline.** Check out `drivers/nvme/host/`
      from the target kernel tag.  This is the merge **right side**.
- [ ] **3-way merge into SNVMe.** For each file SNVMe carries
      (`core.c`, `pci.c`, `ioctl.c`, `multipath.c`, `zns.c`,
      `hwmon.c`, `fabrics.c`, `rdma.c`, `tcp.c`, `nvme.h`), run a
      3-way merge (stock-old → stock-new → SNVMe-fork). **Never**
      rebase by just applying the `old→new` upstream patch on top
      of SNVMe — conflict resolution without the fork as the third
      input hides struct-layout bugs.
- [ ] **If you are uplifting from 5.4 to 5.15-class**, the repo
      already ships `snvme-5.4.241-1-tlinux4-0017/snvme-pci-5.15-incremental.diff`
      that captures every SNVMe-specific delta the 5.4 fork has on
      top of the 5.15 fork.  Use it as a sanity reference for which
      hunks must reappear after the 3-way merge — if a hunk in the
      diff has no analogue in your merged tree, that is a missed
      port.
- [ ] **Re-apply renames (§3.1)** on anything upstream added.
      `grep -n '\bnvme_[a-z_]*\(' snvme/*.c` — any new match that is
      also exported or referenced cross-module gets a `snvm_` /
      `s_nvme_` prefix.

### 7.2 Phase 2 — Structural surgery (diff cannot do this for you)

- [ ] **Audit `struct nvme_dev` layout.** Open `pci.c` side-by-side
      with the **new** upstream `nvme.h` definition. Every SNVMe-added
      field should go *after* all upstream fields (ABI doesn't matter
      inside the module, but bisecting crashes later is easier if
      SNVMe additions are in one contiguous block). Any upstream field
      that was removed must be removed from SNVMe access sites too.
      Confirm with `pahole snvme.ko` if you can.
- [ ] **Audit `struct ctrl`** (`snvme/ctrl.h`). This one is
      SNVMe-owned so upstream won't touch it, but verify
      `ctrl->pdev` lifetime matches the new probe/remove ordering
      (see §5 and §7.3).
- [ ] **Relocate the `use_sreg` branch (§3.2).** In the NEW upstream
      tree, find the call site that allocates SQ/CQ memory for IO
      queues. Call-graph to trace from:
      `nvme_probe → nvme_setup_io_queues → nvme_create_io_queues → nvme_alloc_queue → …dma_alloc_coherent…`
      Whatever that chain looks like in the target kernel, the
      `ctrl->use_sreg && ioq_num == ioq_map_num` test has to sit
      **immediately before** the `dma_alloc_coherent` it is replacing.
      Not one layer up, not one layer down.
- [ ] **Audit every `pci.c` branch that references user-allocated
      fields.** At time of writing these are
      `nr_user_allocated_{queues,cq,sq}`, `nr_user_use_{cq,sq}`,
      `user_start_qid`, `online_user_queues`, `max_qid`,
      `use_user_allocated`. If `nvme_setup_io_queues` changed how
      `nr_io_queues` is computed, each arithmetic site must be
      re-derived from first principles (locate the
      `nvme_set_queue_count` / `pci_alloc_irq_vectors_affinity`
      block in `s_nvme_setup_io_queues`).
- [ ] **Verify admin-queue path is untouched.** The user-pages branch
      MUST only apply to IO queues. If upstream merged admin+IO queue
      allocation, split them back out in SNVMe.
- [ ] **Verify block-device registration still uses
      `"snvme%dn%d"`** (every `sprintf` / `snprintf` of `disk_name`
      in `core.c` and `multipath.c`). Upstream
      naming of `disk->disk_name` has been touched by several
      releases; don't let a merge silently revert it to `"nvme..."` —
      that breaks the §2 namespace-separation guarantee.

### 7.3 Phase 3 — Semantic audit (this is where bugs actually live)

- [ ] **Re-read every upstream commit message** on `drivers/nvme/`
      between the old and new baseline. In practice:
      `git log --oneline v<OLD>..v<NEW> -- drivers/nvme/host/ | wc -l`
      — if this is more than ~50, budget at least a day. Flag any
      commit whose subject contains `lock`, `refcount`, `probe`,
      `reset`, `queue`, `tagset`, `irq`, `reinit`, or `remove` — these
      are the ones that silently change SNVMe's assumptions.
- [ ] **Lock-order check.** In the new kernel, walk `nvme_probe` and
      `nvme_reset_work` top-to-bottom and verify that every mutex /
      rw_sem SNVMe touches (admin_q, shutdown_lock, namespaces_rwsem,
      subsys_lock, `snvm_control_lock`) is still taken in the same
      order relative to each other. New kernels occasionally move
      `mutex_lock` calls across function boundaries.
- [ ] **Reset / live-migration path.** Run `nvme reset-controller
      /dev/snvme0n1` after a successful bind. If the reset path in
      the new kernel reallocates IO queues, the `use_sreg` flag
      must be re-honored — otherwise the reset silently falls back to
      `dma_alloc_coherent` and your SQ/CQ pointers on the GPU go
      stale. This is a common regression and **smoke tests will NOT
      catch it**; write a dedicated reset test.
- [ ] **IRQ affinity.** If `nvme_setup_irqs` /
      `pci_alloc_irq_vectors_affinity` changed, verify that the MSI-X
      vectors for user-allocated queues are still routed to the GPU's
      NUMA node (or at least not pinned to a CPU that disagrees with
      what `init_queues` assumed).
- [ ] **`nvidia_p2p_*` compatibility.** These are loaded from the
      proprietary NVIDIA driver and are sensitive to **driver
      version**, not kernel version. But the function-signature
      shims in `snvme/nvfs-p2p.c` + `nvfs-pci.{c,h}` are
      kernel-version-sensitive (they use `get_user_pages_fast`-family
      helpers whose signatures shift). Rebuild the NVIDIA driver
      against the new kernel first, then SNVMe on top.

#### 7.3.1 Known regression traps (re-audit these every uplift)

This is a list of bugs that have been found and fixed in SNVMe in the
past — they are easy to reintroduce during a 3-way merge because the
surrounding code changes but the **bug pattern** is invisible to diff.
Re-audit each one after §7.1.

> **Per-baseline status note.** Where a trap was specifically verified
> against a baseline different from `snvme-5.15.0-public`, the affected
> file calls it out at the patch site with a `PORTING.md §7.3.1` cross
> reference. The `snvme-5.4.241-1-tlinux4-0017` baseline was audited
> in full against this list and additionally fixes traps #4, the
> `NVM_MAP_DEVICE_MEMORY` `copy_to_user` leak, and the
> **`snvm_dev_fops` missing `.release`** hook (all three of which are
> still latent in `snvme-5.15.0-public`); see that directory's `pci.c`
> banner for the full bug-fix list. When uplifting to a new kernel,
> diff against `snvme-5.4.241-1-tlinux4-0017/pci.c` for the cleanest
> version of these fixes.

- **`svm_mmap_registers` null-check must be `||`, not `&&`.**
  (`snvme/pci.c`.) `ctrl_find_by_inode()` can legitimately return
  `NULL`; if it does, the `&&` form then dereferences it. A merge
  conflict in this function has historically lost the fix.
- **`snvm_dev_map_ioctl` `ret` must be initialised at declaration.**
  (`snvme/pci.c`.) Several `case` branches return through
  `ret = ...; break;`, but a few happy paths (`NVM_UNMAP_*`,
  `NVM_SET_SHARE_REG`, `NVM_CLEAR_IOQ_NUM`) used to fall through to
  the final `return ret;` without setting `ret`, leaking stack
  garbage as the ioctl return value. Keep `int ret = 0;` at the top
  AND set `ret = 0;` on every success `break;` — the redundancy is
  the point.
- **`NVM_MAP_*` must check `IS_ERR_OR_NULL(map)` *before* any
  dereference or counter bump.** (`snvme/pci.c`.) The tempting
  shape "bump `ioq_map_num` → check bound → write `map->ioq_idx`"
  oopses when `map_userspace()` / `map_device_ioqueue_memory()`
  return an `ERR_PTR`. When helpers are refactored in an uplift the
  check-after-deref pattern often creeps back. The correct order
  is: (1) call the mapper, (2) `IS_ERR_OR_NULL` guard, (3) bound
  check, (4) commit state, (5) `copy_to_user` with rollback-on-fail.
- **`ioq_map_num` counter must roll back on every failure path.**
  (`snvme/pci.c` `NVM_MAP_*` + `copy_to_user` error branches.)
  Failing to roll back poisons the `use_sreg` gate: subsequent
  `SNVM_DEVICE_BIND` sees `ioq_map_num > ioq_num` and silently falls
  back to `dma_alloc_coherent` (the exact Phase-3-class bug). The
  rollback set is: `ctrl->ioq_map_num--`, `ctrl->cq_num--` if
  `map->is_cq`, then `unmap_and_release(map)`. This applies to both
  the budget-overflow path (just after `ioq_map_num += 1`) AND the
  final `copy_to_user(request.ioaddrs, ...)` path. Fixed in
  `snvme-5.4.241-1-tlinux4-0017/pci.c` (`NVM_MAP_HOST_MEMORY` and
  `NVM_MAP_DEVICE_QUEUE_MEMORY` cases); still latent in
  `snvme-5.15.0-public/pci.c`.
- **`snvm_chrdev_helper(remove)` teardown order:
  `ctrl_put()` FIRST, `ida_simple_remove()` SECOND.** (`snvme/pci.c`.)
  `ctrl_put()` uses `ctrl->number` internally (`device_destroy()` /
  `cdev_del()` through the minor-encoded `dev_t`). Returning the
  minor to the IDA pool first opens a window where a concurrent
  `SNVM_CHRDEV_CREATE` picks up the same minor and races our still-
  live `cdev`. Same rule for `snvm_chrdev_create()`'s error unwind.
- **`NVM_MAP_DEVICE_QUEUE_MEMORY` must reject `ioq_idx < 0` *before*
  calling `map_device_ioqueue_memory()`.** Otherwise you pay the cost
  of `nvidia_p2p_get_pages()` (which is slow and can fail partially)
  only to throw the result away on the next line.
- **`NVM_MAP_DEVICE_MEMORY` (data path) must `unmap_and_release()`
  on `copy_to_user` failure.** Otherwise a crash in userspace between
  `ioctl()` and receiving the IO addresses leaks pinned GPU pages
  for the lifetime of the module. Fixed in
  `snvme-5.4.241-1-tlinux4-0017/pci.c`; still latent in
  `snvme-5.15.0-public/pci.c`.
- **`nvme_probe()` must gate on `ctrl_find_by_pci_dev(&ctrl_list, pdev) != NULL`
  at the very top, returning `-ENODEV` otherwise.** (`snvme/pci.c`.)
  `pci_register_driver(&snvme_driver)` inside `snvm_rebind_driver()`
  asks the PCI core to call `.probe()` for **every** matching NVMe on
  the host, not just the BDF the user passed to `SNVM_DEVICE_BIND`.
  Without the gate, the *first* `SNVM_DEVICE_BIND` on a multi-NVMe
  host hijacks every unbound NVMe it finds, even ones the in-tree
  `nvme` driver was supposed to own. Worse, those extra NVMes have
  no `ctrl` record, so they skip the `use_sreg` branch and come up
  on kernel-DMA queues — looking "fine" to probe but broken for
  SNVMe IO. Keep the gate at the top of `nvme_probe()`; do NOT
  collapse it into the later "`if (ctrl && ...)`" check.
- **User queue indices (`ioq_idx`) are 0-based.** `nvme_create_io_queues_mix()`
  (`pci.c`) walks the user queues with `count = 0; i = online_queues ..`
  and calls `map_find_by_pci_dev_and_idx(list, pdev, uqid=count, is_cq=1)`.
  If userspace registers its first SQ/CQ with `ioq_idx=1`, the lookup
  misses and probe dies with `map_find_by_pci_dev_and_idx cq error!`.
  libnvm gets this right (`queue.h` passes `qp_id` starting at 0);
  anything that talks to SNVMe directly (smoke tests, external tools)
  MUST start `ioq_idx` at 0 too.
- **`NVM_SET_IOQ_NUM` field-name landmine: `request.is_cq` is *not* a
  CQ flag here, it's the `on_host` flag.** (`pci.c` `NVM_SET_IOQ_NUM`
  case copies `request.is_cq` into `ctrl->on_host`.) Later, in
  `nvme_create_user_queue()`, `dev->queue_on_host` decides which list
  is searched:
  ```
  if (dev->queue_on_host) list = &host_list;
  else                    list = &device_queue_list;
  ```
  So a userspace program that pins SQ/CQ rings via `NVM_MAP_HOST_MEMORY`
  MUST pass `is_cq = 1` (= on_host=1) to `NVM_SET_IOQ_NUM`, otherwise
  probe silently looks in `device_queue_list`, fails to find the rings,
  and returns the same `map_find_by_pci_dev_and_idx cq error!` as the
  off-by-one trap above. The two failure modes look identical in dmesg
  but are independent. Same applies to `NVM_MAP_HOST_MEMORY` /
  `NVM_MAP_DEVICE_QUEUE_MEMORY` — pick the ioctl that matches your
  `on_host` decision.
- **`snvm_dev_fops` MUST have `.open` + `.release` hooks so that an
  abnormal userspace exit cannot leak host pins, GPU p2p references,
  or per-ctrl IO-queue accounting counters.** Upstream snvme-5.15.0
  and the original 5.4 port ship `snvm_dev_fops` with only `.owner +
  .unlocked_ioctl + .mmap` — there is no automatic cleanup if the
  process holding `/dev/ssnvme*` open dies between `NVM_MAP_*` and the
  matching `NVM_UNMAP_*`. Symptoms on TencentOS 5.4.241 (reproducible
  in `/var/log/messages`):

    1. Next `SNVM_DEVICE_BIND` after a test crash logs
       `snvme: ctrl exist, ioq_num=N cq_num=M map_num=K` — the
       controller is reused **dirty**, with counters carried over
       from the dead process.
    2. `nvidia.ko` refcount accumulates because nobody calls
       `nvidia_p2p_put_pages()`; eventually `rmmod snvme` says
       "module in use" forever and the box requires a reboot.

  Note: the `snvme: snvme_find_get_ns(nsid=1) failed` 3x log line
  often appears nearby in `/var/log/messages` but is a **separate**
  bug (the `NVM_GET_DEV_INFO` vs `nvme_scan_work` race documented in
  the next trap entry). The two are independently reproducible and
  must be fixed independently — the dirty-rebind path makes the
  scan race **more likely** by short-circuiting the probe-side
  delays, but the scan race exists on a fresh module load too.

  Fix (the `snvme-5.4.241-1-tlinux4-0017/` baseline; locate by
  symbol: `snvm_dev_open` / `snvm_dev_release` in `pci.c`,
  `map_purge_by_owner` in `map.c`, `struct snvm_dev_owner` in
  `pci.c` / `ctrl.h`):

  - `.open` allocates a `struct snvm_dev_owner { ctrl, owner }` and
    stashes it in `file->private_data`. Capturing the owner at open
    time (not at release time) is critical — by the time
    `__fput()` runs, `current` may be a different thread group
    member or a forked child, while `map->owner` was set to the
    process that issued the `NVM_MAP_*` ioctl.
  - `.release` walks `host_list / device_queue_list` once to compute
    the rollback deltas for `ctrl->ioq_map_num` and `ctrl->cq_num`,
    then calls `map_purge_by_owner(list, owner)` against all three
    map lists. Use **checked subtraction** for the counter rollback
    (a buggy userspace path can leave counters in a state where
    `rb_*` exceeds the current value; clamp to zero rather than
    underflow into UINT_MAX, which would then disable the `use_sreg`
    branch on the next bind).
  - The split `pass-1 count / pass-2 free` is mandatory because
    `unmap_and_release()` does `list_remove()` on the descriptor;
    saving a `next` pointer across the call would dereference a
    freed node. `map_purge_by_owner` re-fetches `list_next(&head)`
    after every free for the same reason.

  Re-audit rule: any uplift that touches `snvm_dev_fops`,
  `struct map`, or the `ioq_map_num` / `cq_num` accounting MUST
  re-verify that these two hooks still fire — a single
  `kill -9 <pid>` against the smoke test, immediately followed by
  `cat /sys/module/snvme/refcnt` and `lsof /dev/ssnvme0`, is the
  fastest manual probe.

- **`NVM_GET_DEV_INFO` MUST wait for `nvme_scan_work` to finish
  before returning `snvme_find_get_ns(nsid=1) failed`.** `pci.c`
  `snvme_start_ctrl()` -> `nvme_queue_scan()` -> `queue_work(s_nvme_wq,
  &ctrl->scan_work)` is asynchronous: the worker is the only code
  path that calls `nvme_alloc_ns()` and `list_add_tail(&ns->list,
  &ctrl->namespaces)`. `snvm_rebind_driver` finishes (and userspace
  gets back from `SNVM_DEVICE_BIND` -> `ioctl()`) at the moment
  `device_attach` returns, which is **before** `scan_work` has even
  started in many cases. Userspace then immediately issues
  `NVM_GET_DEV_INFO` and `snvme_find_get_ns` walks an empty
  `namespaces` list, returning NULL.

  Symptom (TencentOS 5.4.241, `/var/log/messages` 2026-05-18
  16:11:12 and 19:21:46): every BIND logs **exactly three**
  consecutive `snvme: snvme_find_get_ns(nsid=1) failed` lines —
  the "three" comes from libnvm's caller-side retry loop in
  `device.cpp`. The 3x retries finish well within the
  `scan_work` window, so all three observe an empty list.

  Fix (`snvme-5.4.241-1-tlinux4-0017/pci.c` `NVM_GET_DEV_INFO`
  case): on first lookup failure, call `flush_work(&ndev->ctrl.scan_work)`
  (no-op if the work was never queued — `flush_work` documents this)
  and retry; then if still NULL, poll with `msleep(50)` +
  `flush_work` for up to 5 s before returning `-EFAULT`. The bound
  preserves caller EFAULT semantics if the controller is actually
  broken (admin queue dead, state never reached `NVME_CTRL_LIVE`,
  etc.).

  Pitfall to avoid: do NOT "fix" this in userspace by adding more
  retry layers in libnvm. The kernel side has the synchronisation
  primitive (`flush_work`) and the access to `ctrl->scan_work`;
  userspace can only sleep blindly and hope, which is what created
  the 3-retries-but-all-too-fast pattern visible in the logs.

- **`snvm_rebind_driver` MUST use `driver_attach(&snvme_driver.driver)`,
  NOT `device_attach(&pdev->dev)`.** This is a 5.4-specific landmine.
  `device_driver_attach()` (used by snvme-5.15.0) does not exist on
  5.4, and the obvious substitute `device_attach()` has subtly wrong
  semantics for our use case: `device_attach` walks the device's bus
  callback `__device_attach`, which iterates *all matching drivers*
  and picks the **first** registered one. On a TencentOS host the
  in-tree `nvme.ko` is loaded at boot, so it is always the first
  match, and `device_attach()` silently rebinds the BDF to the
  in-tree driver. dmesg signature (with the old code):

  ```
  snvme: binding nvme device to snvme: pci 0:8:0.0
  nvme nvme0: pci function 0000:08:00.0          <-- nvme, not snvme!
  nvme nvme0: 135/0/0 default/read/poll queues   <-- 3-tuple
  ```

  Note the absence of the `snvme: ctrl exist, ioq_num=...` line that
  a successful snvme bind emits, and the absence of the
  `snvme: device driver name: snvme` confirmation line. The follow-up
  `SNVM_DEVICE_UNBIND` then trips the "device's driver is not snvme"
  branch (-EFAULT on the old code; now -EINVAL after a separate
  errno-cleanup fix).

  The reproducer that catches this every time:

  ```
  ./run_snvme_smoke.sh 0000:08:00.0 --gpu              # leaves BDF "loose"
  ./run_snvme_smoke.sh 0000:08:00.0 --gpu --bind       # fails at step 15
  ```

  Fix: replace `device_attach` with a bounded retry loop calling
  `driver_attach(&snvme_driver.driver)`. `driver_attach` walks the
  bus's device list and invokes the SPECIFIC driver's probe on every
  unbound matching device. The `nvme_probe()` per-BDF gate
  (`ctrl_find_by_pci_dev(&ctrl_list, pdev) != NULL`) ensures the
  effect is scoped to the BDF the user already CHRDEV_CREATEd; every
  other NVMe on the bus short-circuits to `-ENODEV` at the top of
  probe. The retry is needed because, between `device_release_driver`
  and `driver_attach`, udev's drivers_autoprobe rule may rebind the
  device to the in-tree nvme — three attempts is enough in practice;
  if udev wins three times in a row the host has a misconfigured
  autoprobe rule and `-EBUSY` is the honest answer.

  Verification: after a successful BIND the dmesg block should
  contain `snvme: device driver name: snvme` and `snvme snvme0: pci
  function ...` (note the doubled `s` in the device name); the
  follow-up `default/read/poll/user queues` line MUST be the
  4-tuple variant (`135/0/0/0`), not 3-tuple.

- **`NVM_ADD_USER_QUEUE` vaddr lookup MUST mask with each map's own
  `page_size`, NOT a single global `PAGE_MASK`.** (`snvme/pci.c`
  ADD_USER_QUEUE case.)  When B3 was first added the obvious-looking
  loop body

  ```c
  list_for_each_entry(cursor, &g->maps, group_link) {
      if (cursor->vaddr == (req->pairs[i].sq_vaddr & PAGE_MASK))
          m_sq = cursor;
      ...
  }
  ```

  worked for CPU smoke (host pages = 4 KiB → `PAGE_MASK = ~0xFFF`)
  but silently missed every GPU-allocated SQ/CQ ring registered via
  `NVM_MAP_DEVICE_MEMORY` (GPU pages = 64 KiB; the lower 16 bits of
  the GPU vaddr are not necessarily zero).  The lookup had to be
  page-size adaptive — `map_userspace` stores `map->page_size =
  PAGE_SIZE` while `map_device_memory` stores `map->page_size =
  GPU_PAGE_SIZE (= 64 KiB)`, so the correct shape is:

  ```c
  list_for_each_entry(cursor, &g->maps, group_link) {
      u64 mask = ~((cursor->page_size ?
                    (u64)cursor->page_size : (u64)PAGE_SIZE) - 1);
      if (cursor->vaddr == (req->pairs[i].sq_vaddr & mask))
          m_sq = cursor;
      if (cursor->vaddr == (req->pairs[i].cq_vaddr & mask))
          m_cq = cursor;
      if (m_sq && m_cq) break;
  }
  ```

  Symptom of regression: GPU `snvme_smoke_gpu` fails at `NVM_ADD_USER_QUEUE`
  with `errno=2 (No such file or directory)` and dmesg shows
  `snvme: NVM_ADD_USER_QUEUE: lookup miss for sq_vaddr=0x...`; CPU
  `snvme_smoke_io` continues to pass because it always allocates
  4 KiB-aligned host buffers.  Re-audit rule: any uplift that
  changes `struct map`, `map_userspace`, `map_device_memory`, or
  the alignment guarantees of `nvidia_p2p_get_pages` MUST re-verify
  this lookup.

- **`NVM_SET_KERNEL_IOQ_CAP` clamp MUST be applied AFTER MSI-X
  negotiation, not before.** (`snvme/pci.c` `s_nvme_setup_io_queues`.)
  The tempting shape

  ```c
  /* WRONG */
  unsigned int nr_io_queues = num_possible_cpus();
  if (ctrl->setup.cap_kernel_ioq && cap < nr_io_queues)
      nr_io_queues = ctrl->setup.cap_kernel_ioq;
  result = nvme_set_queue_count(&dev->ctrl, &nr_io_queues);
  ...
  result = pci_alloc_irq_vectors_affinity(...);
  ```

  asks the controller for `cap` queues instead of
  `num_possible_cpus()` and burns the rest of the controller's
  MSI-X budget needlessly.  The correct order is:

  ```c
  /* RIGHT */
  unsigned int nr_io_queues = num_possible_cpus();
  result = nvme_set_queue_count(&dev->ctrl, &nr_io_queues);
  ...
  result = pci_alloc_irq_vectors_affinity(...);
  dev->ctrl_max_io_queues = nr_io_queues;        /* what controller granted */
  if (ctrl->setup.cap_kernel_ioq &&
      ctrl->setup.cap_kernel_ioq < nr_io_queues)
      nr_io_queues = ctrl->setup.cap_kernel_ioq; /* user pool gets the rest */
  /* ... rest of setup_io_queues runs on the capped value ... */
  ```

  Two distinct things matter here: (1) the controller must be asked
  for its full grant first (so `ctrl_max_io_queues` is the real
  ceiling, not the cap), and (2) the cap must apply BEFORE
  tagset allocation (so the kernel block-mq layer sizes its tagset
  to the kernel-only share, not the controller grant).  Symptom of
  regression: `NVM_GET_DEV_INFO` reports `start_cq_idx=cap+1,
  max_user_qid=cap`, which makes the user QID pool empty and every
  `NVM_ADD_USER_QUEUE` returns `-EAGAIN`.

- **`NVM_GET_DEV_INFO` MUST populate `bar0_size`, `q_depth`,
  `max_user_qid`, AND `sgl_supported` for B3 callers to function.**
  (`snvme/pci.c` GET_DEV_INFO case.)  Adding fields at the tail of
  `struct nvm_ioctl_dev` after a kernel uplift is harmless ABI-wise
  (size grows; old userspace ignores the extra bytes), but the
  kernel side MUST actually fill them or B3 callers segfault on
  `mmap(BAR0_size=0)` and skip Tier 4 even on SGL-capable
  controllers.  Verify with: after `NVM_GET_DEV_INFO`, every field
  in §4.3.2 must be non-zero except `nr_user_q` (legacy, 0 in B3
  flow).

- **`map_kind` enforcement at `NVM_ADD_USER_QUEUE` lookup.**
  (`snvme/pci.c` ADD_USER_QUEUE case + `snvme/map.h` `struct map`.)
  Pre-B6 the lookup that resolves `(sq_vaddr, cq_vaddr)` against
  `g->maps` did **only** a vaddr-mask comparison, with nothing on
  the kernel side stopping userspace from passing a data-buffer
  vaddr where a ring vaddr was meant -- the kernel would then
  Create I/O SQ with PRP1 = data buffer's dma_addr, the controller
  would read garbage as SQEs, and the failure mode is silent
  corruption / controller hang rather than a clean `-EINVAL`.

  Fix shape: tag every `struct map` with `enum nvm_map_kind`
  (`RING_SQ`, `RING_CQ`, `DATA`, or `UNSPECIFIED` for pre-B6
  callers).  In the lookup, only `RING_SQ` matches `sq_vaddr` and
  only `RING_CQ` matches `cq_vaddr`; `DATA` maps are skipped
  outright; `UNSPECIFIED` (back-compat) matches either slot
  exactly as before.  Mismatch → `-EINVAL` up front.

  Also: `DATA` maps live on `own->data_maps` (per-fd), NOT on
  `g->maps`, so they survive `NVM_DESTROY_QUEUE_GROUP` and only
  get reaped on fd close.  This is the lifecycle decoupling
  documented in §4.3.1.  Re-audit rule: any uplift that touches
  the ADD_USER_QUEUE handler, `struct map`, or the
  `snvm_dev_open` / `_release` lifecycle MUST re-verify that
  `map_purge_by_owner` does not also walk `data_maps` (it would
  double-free; `snvm_dev_release` already drains `data_maps`
  separately) AND that destroy_qgroup_locked does NOT touch the
  fd-scoped `data_maps` list.

- **`nvm_ioctl_dev.max_data_size` units must be bytes, not 512 B
  sectors.**  (`snvme/pci.c::ioctl_get_dev_info` populator.)
  The kernel-internal `nvme_ctrl::max_hw_sectors` is in 512-byte
  sectors regardless of LBA size (NVMe block layer convention).
  Pre-fix the populator copied that field verbatim into
  `drequest.max_data_size`, but `include/ioctl.h` documents the
  field as **"CTRL.MDTS in bytes"**.  This silently asked every
  userspace consumer to either know the convention and `* 512`
  themselves, or to under-count PRP_List sizing on 4 KiB-LBA
  controllers.  libnvm flip-flopped on it across versions
  (Commit 1 of L1 even had to revert a `* 512` strip-out).

  Fix shape: convert at the source —
  `drequest.max_data_size = (size_t)ndev->ctrl.max_hw_sectors << 9;`
  Both `snvme-5.4.241-1-tlinux4-0017/pci.c` and
  `snvme-5.15.0-public/pci.c` carry the fix; userspace
  (`libnvm/src/linux/device.cpp::ioctl_get_dev_info` and
  `nvm_controller_init_b3`) trusts the byte value verbatim.
  Verify with: on a controller whose MDTS is X KiB,
  `disk.max_data_size` == X*1024.  On the H20 NVMe used as
  reference, MDTS = 128 KiB → `disk.max_data_size == 131072`,
  not 256.

None of these are detected by the smoke tests as written — the
smoke tests run the happy path. They are detected by (a) reading
this list during the merge, and (b) the **reset + stress** workload
described in §7.4.

### 7.4 Phase 4 — Verification (the mandatory gate)

The full verification gate runs four binaries.  All four must
return 0 against a freshly-loaded `snvme.ko` on a throw-away NVMe
SSD.  Kernel **MUST be rebuilt and `.ko` reloaded** between any
porting change and these runs; the smoke binaries embed
`_IOC_SIZE`-derived ioctl numbers and a stale .ko returns
`-ENOTTY` on otherwise-valid requests.

- [ ] **`snvme_smoke`** returns 0 (UAPI-only, no bind).  Exercises
      `SNVM_CHRDEV_CREATE` / `_REMOVE`, `NVM_MAP_HOST_MEMORY`,
      BAR0 mmap.  Necessary, not sufficient.
- [ ] **`snvme_smoke_gpu`** (no `--bind`, no `--rounds`) returns 0
      on a host with NVIDIA driver loaded.  Adds the
      `NVM_MAP_DEVICE_MEMORY` and p2p path; failures here typically
      live in `snvme/nvfs-p2p.c` or `nvfs-pci.{c,h}`, not the
      core driver.
- [ ] **`snvme_smoke_io`** (`--bind`, B3 flow, CPU rings) returns 0.
      23 phases including PRP1 / PRP1+PRP2 / PRP_List / SGL (auto-
      skipped on PRP-only controllers) + SQ-tail-wrap stress, with
      byte-by-byte data verification on every IO.  This is what
      catches the §7.3.1 vaddr-mask + cap-after-negotiation regressions
      on the CPU side.
- [ ] **`snvme_smoke_gpu --bind --rounds 4`** returns 0 on a host
      with NVIDIA driver loaded.  Repeats the `snvme_smoke_io`
      phase set 4 times with full GPU-resident rings and data
      buffers, allocating + freeing the entire (queue group, GPU
      pages, user IO queues) stack between rounds.  This is the
      authoritative test for:
        * dynamic alloc/free of GPU IO queues at runtime,
        * `NVM_DESTROY_QUEUE_GROUP` cascade through Delete I/O
          SQ/CQ + nvidia_p2p_put_pages,
        * user QID pool reclamation across rounds (a successful
          run shows the same QIDs being returned for round 0..N-1),
        * GPU SQE submission via `__threadfence_system()` ordering,
        * GPU CQE polling with phase-bit flips across SQ wraps.
      Expected dmesg signature is **N pairs** of
      `NVM_ADD_USER_QUEUE group=G created K queue(s) (qids ...)` +
      `destroy_qgroup id=G drained K user queue(s)` +
      `destroy_qgroup id=G drained M map(s)`, with no leak warnings.
- [ ] **Reset test.** After `snvme_smoke_gpu --bind` succeeds,
      issue `nvme reset-controller` against the resulting
      `/dev/snvmeXnY`.  It must either complete cleanly or fail
      loudly; **silent fallback to kernel-DMA queues for new
      `NVM_ADD_USER_QUEUE` calls is a bug** (the controller forgot
      every Create-I/O-{SQ,CQ} after reset, so the user QID pool
      MUST be torn down and rebuilt by the reset path).
- [ ] **Co-existence test.** With both `nvme.ko` and `snvme.ko`
      loaded, bind one NVMe to each, mount both, `fio` them
      simultaneously for ≥5 min.  The two drivers use **disjoint**
      block-device namespaces (in-tree → `/dev/nvme*`, SNVMe →
      `/dev/snvme*`), so collisions on the device-node side should
      not happen — verify that, and watch for kernel oopses or
      shared workqueue/IRQ name clashes (`s_nvme_wq` vs `nvme_wq`,
      etc.).
- [ ] **Stress test under reset.** Run the co-existence workload
      while looping `nvme reset-controller /dev/snvme0n1` every 30 s
      for 10 iterations.  If this stays clean, you've caught most
      Phase 3 semantic drift.

### 7.5 How far to go?

For a **patch-level** uplift (5.15.x → 5.15.y): Phase 1 + 4 usually
suffices. For a **minor-version** uplift (5.15 → 5.19): Phase 1–3
are all required. For a **cross-LTS** uplift (5.15 → 6.6): assume
Phase 2 and Phase 3 together are a week of work; do NOT skip the
reset test or `snvme_smoke_gpu --rounds`. If upstream merged a
large NVMe refactor (e.g. the `queue_limits` transition in 6.0,
or the `blk_mq_alloc_disk` switch in 5.14), plan for a rewrite of
the §3.2 hook from scratch, not a patch re-apply.

---

## 8. Sanity test programs

Four end-to-end tests, all self-contained (no Tutti filesystem,
no gRPC daemon).  They live at:

```
backends/local/kernel_modules/test/
├── snvme_smoke.c          # libc-only UAPI smoke (no bind)
├── snvme_smoke_gpu.cu     # B3 GPU end-to-end + dynamic alloc/free rounds
├── snvme_smoke_io.c       # B3 CPU end-to-end (PRP1/2/List/SGL + wrap)
├── snvme_smoke_qgroup.c   # B1 group create/destroy lifecycle
├── snvme_smoke_recycle.c  # B4 recycle (Delete + Create on the same QID)
├── snvme_smoke_addq.c     # B3 add/destroy lifecycle, host rings only
├── run_snvme_smoke.sh     # wrapper that auto-detects --gpu / --bind args
└── Makefile               # builds whichever binaries `nvcc` is available for
```

| Binary                  | Covers                                                                                               | Built when             |
| ----------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------- |
| `snvme_smoke`           | `NVM_MAP_HOST_MEMORY` + chrdev create/remove + BAR0 mmap (no bind)                                   | always                 |
| `snvme_smoke_qgroup`    | B1: `NVM_CREATE_QUEUE_GROUP` / `NVM_DESTROY_QUEUE_GROUP` lifecycle + fd-close cascade                 | always                 |
| `snvme_smoke_addq`      | B3 happy path: `NVM_SET_KERNEL_IOQ_CAP` + bind + `NVM_ADD_USER_QUEUE` (host rings)                  | always                 |
| `snvme_smoke_recycle`   | B4: `NVM_RAW_ADMIN_CMD` driving Delete + Create I/O SQ/CQ on the same QID                            | always                 |
| `snvme_smoke_io`        | B3 CPU end-to-end: PRP1 / PRP1+PRP2 / PRP_List / SGL (auto-skip if SGLS=0) + SQ-tail-wrap; 23 phases, byte-by-byte verify on every IO | always |
| `snvme_smoke_gpu`       | B3 GPU end-to-end: same Tier 1..4 + wrap as `_io`, but rings AND data buffers via `NVM_MAP_DEVICE_MEMORY`; supports `--rounds N` to repeat the entire alloc/free cycle | when `nvcc` is on `$PATH` |

`snvme_smoke_gpu` flags:

```
--gpu N        select CUDA device N (default 0)
--rounds N     run N full alloc/free cycles back-to-back (default 4)
<PCI_BDF>      target controller, e.g. 0000:08:00.0
```

A successful 4-round GPU smoke goes through **8 distinct queue
groups, 8 distinct allocations of every GPU page (rings + data +
PRP_List), and 8 distinct (Create + Delete) admin command pairs**
on the controller; if anything in the alloc/free lifecycle is
broken, the second round will fail (see the §7.3.1 trap entries
for symptoms).

Run via the wrapper:

```bash
cd backends/local/kernel_modules/test

# host (libc) UAPI smoke -- safe even on production hosts
./run_snvme_smoke.sh

# GPU UAPI smoke -- requires NVIDIA driver loaded, still safe (no bind)
./run_snvme_smoke.sh --gpu
./run_snvme_smoke.sh --gpu --gpu-id 1            # pick CUDA device 1

# full bring-up + B3 lifecycle on a throw-away NVMe (DESTRUCTIVE)
./run_snvme_smoke.sh --bind                      # B3 host-ring path
./run_snvme_smoke.sh --gpu --bind                # B3 GPU-ring path

# direct invocation of the post-bind binaries (--bind is implicit):
sudo ./snvme_smoke_io 0000:08:00.0
sudo ./snvme_smoke_gpu --gpu 0 --rounds 4 0000:08:00.0
```

Every binary exits with code `0` only when every UAPI step round-
trips cleanly.  Any failure prints `[FAIL] step=<N> ... errno=<E>`
and stops; the dmesg block for the same time window is the
authoritative source for the kernel-side cause.

> **Tip.** During a kernel uplift, run `snvme_smoke` first.  Only
> when it passes should you try `snvme_smoke_io` (CPU B3 path) and
> only after THAT passes should you try `snvme_smoke_gpu` (GPU B3
> path).  A failure in `snvme_smoke_gpu` after `_io` passes
> typically means the NVIDIA driver / `nvfs_nvidia_p2p_*` glue is
> broken, not the SNVMe core.

### 8.1 Build/run troubleshooting cheat sheet

| Symptom (where it surfaces) | Root cause | Fix |
|---|---|---|
| `nvcc fatal : Unsupported gpu architecture 'sm_XX'` at `make` time | The Makefile auto-detects CUDA_ARCH from the running GPU's compute capability via `nvidia-smi --query-gpu=compute_cap`.  Auto-detect fails (no GPU visible / driver not loaded) or the toolkit is too old/new for the detected arch (e.g. CUDA 13 dropped `sm_70`). | Pass `CUDA_ARCH` explicitly: `make CUDA_ARCH=sm_80` (A100, accepted by CUDA 11.0–13.x) or `make CUDA_ARCH=sm_90` (H100/H200/H20). |
| `[FAIL] step=1  cudaGetDeviceCount -> system not yet initialized` at runtime, with `nvidia-smi -L` listing GPUs just fine | NVSwitch-equipped multi-GPU host (HGX H100/H200/H20 boards expose `/dev/nvidia-nvswitch*`).  CUDA runtime refuses `cuInit()` until `nvidia-fabricmanager` finishes the NVLink topology bring-up.  `nvidia-smi -L` does NOT need fabricmanager and so does not catch this. | `sudo systemctl enable --now nvidia-fabricmanager`.  If the service is missing, install the package matching your driver exactly: `sudo dnf install nvidia-fabric-manager-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \| head -n1)`. |
| `[FAIL] step=1 cudaGetDeviceCount -> system not yet initialized` even after fabricmanager is active | NVLink Inband mode (H20 / H100 8-GPU NVL3 hosts): the GPU half of the NVLink handshake hasn't completed.  **Authoritative success signal** (verified on HGX H20 / driver 580.65.06): every GPU's "Fabric" block in `nvidia-smi -q` shows `State : Completed` and `Status : Success`.  Notes: (1) `GPU Fabric GUID : N/A` is **not** a failure indicator on this hardware -- some firmware/driver combos legitimately leave the GUID field N/A even on a healthy fabric.  (2) `Persistence-Mode = Disabled` is **also not** a failure indicator on a freshly rebooted host -- verified 2026-05-19 on HGX H20: PM Disabled across all 8 GPUs, Fabric all Completed, CUDA programs run fine.  PM only matters as a **recovery knob** when nvidia-uvm has been poisoned by a previous failed-cuInit / killed-CUDA-process refcount leak; in that case `nvidia-smi -pm 1` keeps the driver context resident long enough for fabricmanager's retry to complete.  Detect with: `nvidia-smi -q \| awk '/^    Fabric$/,/^$/' \| grep State` (every line should read "Completed"); `nvidia-smi --query-gpu=persistence_mode --format=csv,noheader` is informational only. | If Fabric is incomplete: `sudo nvidia-smi -pm 1 && sudo systemctl restart nvidia-fabricmanager`.  If that still fails, the nvidia kernel modules are likely in a poisoned half-state (typically caused by a previous CUDA process that crashed mid-cuInit and leaked a refcount into nvidia-uvm); the only known recovery is to kill every CUDA-touching process on the host, `rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia` (in that order), then reinstall the NVIDIA driver from its `.run` payload.  To make PM persistent across reboot (only useful as a defensive measure on hosts that are known to crash CUDA processes), install an `nvidia-persistenced.service` systemd unit -- the driver `.run` does not install one on TencentOS by default. |
| **Misleading symptom note**: `strings /lib64/libcuda.so.1 \| grep -E '^[0-9]+\\.[0-9]+\\.[0-9]+$'` is NOT a valid way to verify libcuda's own version on driver >= 575.x.  The integers it returns are the *compatibility table* (what older client drivers this libcuda accepts), not the library's own version.  A 580.65.06 libcuda legitimately shows `575.57.07` as its highest match.  Use the file's hash against the matching `.run` payload, or trust `nvidia-smi --query-gpu=driver_version`, which queries the kernel module directly. | -- | -- |
| `insmod: Required key not available` / `Loading of unsigned module is rejected` | Deployment kernel built with `CONFIG_MODULE_SIG_FORCE=y` (TencentOS 5.4.241-1-tlinux4-0017).  See section 6.1 above. | Sign through the deployment signing service, or develop on a kernel without `CONFIG_MODULE_SIG_FORCE=y`. |
| `kobject_add_internal failed for nvme-wq with -EEXIST` at `insmod snvme-core.ko` time | Leftover `"nvme-wq"` / `"nvme-reset-wq"` / `"nvme-delete-wq"` string literal not renamed to `"snvme-*"`; collides with in-tree `nvme-core.ko`.  Section 2 string-literal rename rule missed.  Re-audit using the checklist in section 2 item 4. | Apply the rename to every workqueue / chrdev region / class literal listed in section 2 item 4.  Re-`insmod`. |
| `map_find_by_pci_dev_and_idx cq error!` in dmesg during probe | Two distinct off-by-one bugs share this dmesg line: (a) user `ioq_idx` starts at 1 instead of 0 (section 7.3.1 trap #8); (b) userspace called `NVM_MAP_HOST_MEMORY` but passed `is_cq != 1` to `NVM_SET_IOQ_NUM` so the kernel searches the wrong queue list (trap #9). | Re-read PORTING.md section 7.3.1 traps #8 and #9; verify libnvm / smoke-test caller matches the on_host vs device_queue split. |
| `snvme: snvme_find_get_ns(nsid=1) failed` (exactly 3x per BIND) | `NVM_GET_DEV_INFO` ioctl races `nvme_scan_work`. Userspace gets back from `SNVM_DEVICE_BIND` at the moment `device_attach` returns, but `nvme_alloc_ns()` (the only path that puts `nsid=1` on `ctrl->namespaces`) runs asynchronously on `s_nvme_wq` after `snvme_start_ctrl()`. libnvm's 3-retry loop in `device.cpp` finishes inside the race window. **Independent of** any dirty-rebind / `.release` issue — reproduces on a fresh module load. Often *correlated* with a `snvme: ctrl exist, ioq_num=N cq_num=M map_num=K` line just above it (which is the separate dirty-rebind symptom of the `.release` bug). | The fix is on the **kernel side**, not userspace: `NVM_GET_DEV_INFO` must `flush_work(&ndev->ctrl.scan_work)` + bounded poll before declaring failure. See §7.3.1 trap "`NVM_GET_DEV_INFO` MUST wait for `nvme_scan_work`". Verify by grepping the BIND-time dmesg block for `NVM_GET_DEV_INFO: nsid=1 ready after N ms scan wait` (info log emitted on slow-path success). |
| `rmmod snvme` says `module is in use` long after every `/dev/ssnvme*` user has exited, with `lsmod` showing `Used by 0` but the refcount in `/sys/module/snvme/refcnt` non-zero | A process died holding pinned p2p / host pages, the original `snvm_dev_fops` had no `.release` hook, so the refs leaked into `nvidia.ko`. See §7.3.1 trap "`snvm_dev_fops` MUST have `.open` + `.release` hooks". | Reboot is the only safe recovery on a host without the `.release` fix applied. With the fix in place this should be impossible — open a bug if it recurs. |
| `rmmod snvme` says `module is in use` (`lsmod` refcnt e.g. `2`) after the OWNER daemon (holding `/dev/snvm_control`) got `SIGKILL`'d instead of exiting via SIGINT/SIGTERM, and `lsof /dev/snvm_control /dev/ssnvme*` shows nothing.  Manually re-issuing `SNVM_DEVICE_UNBIND` + `SNVM_CHRDEV_REMOVE` on a fresh `/dev/snvm_control` fd (`scripts/snvm_recover_addr.cpp`) does NOT clear the refcount either. | **Two independent things, don't conflate them.** (1) The Linux cdev framework auto-pairs `try_module_get`/`module_put` on every `open()`/fd-close of `/dev/snvm_control` and `/dev/ssnvme*`, regardless of whether the driver defines `.release` -- so a SIGKILL'd owner does NOT by itself leak `snvme`'s module refcount; `SNVM_DEVICE_UNBIND`/`SNVM_CHRDEV_REMOVE`'s implementations (`pci.c::snvm_unbind_driver` / `snvm_chrdev_helper`) never call `try_module_get`/`module_put` either, so "the ioctl reported success" tells you nothing about the module refcount. (2) The refcount that actually DOES survive an owner's death is `core.c::nvme_open`/`nvme_release` (the `/dev/snvme<N>n<Y>` **block device**, held for as long as it stays `mount(2)`'d -- see `nvme_storage/src/host_fs_backed_nvme_storage.cpp`'s `mount_if_needed_locked`/`umount_locked`, which is a CLIENT action, independent of the daemon's own lifecycle) and `core.c::nvme_dev_open`/`nvme_dev_release` (the RAW `/dev/snvme<N>` admin chardev, no leading `s`). Neither is touched by any owner-side UNBIND/CHRDEV_REMOVE ioctl. | Check `mount \| grep 'snvme[0-9]n'` and `lsof /dev/snvme[0-9]* \| grep -v ssnvme` (note: NOT `/dev/ssnvme*`) BEFORE assuming a kernel-side leak; `scripts/reset_snvme.sh` step 2b now does this automatically and `--force-cleanup`s the mount if found.  If both come back empty and `rmmod` still fails, that IS the case from the row above (§7.3.1's `.release` gap) and reboot is the only recovery. **Separately**: `snvm_chrdev_helper`'s tlinux-5.4 baseline was missing the 5.15 baseline's `create && ctrl` idempotent branch (`ret` stayed at its `-EFAULT` init value), so re-`SNVM_CHRDEV_CREATE`ing a BDF whose `struct ctrl` was left behind by a killed owner returned errno=14 instead of just reporting the existing minor -- this is what makes the daemon itself fail to restart even before you get to the rmmod question. Fixed in `pci.c` to match `snvme-5.15.0-public`. |
| `[FAIL] step=15 SNVM_DEVICE_UNBIND ... errno=14 (Bad address)` (or with the errno fix: `errno=22 (Invalid argument)`), dmesg shows `snvme: device's driver is '...nvme', not 'snvme'` and the BIND-time block contains `nvme nvme0: pci function ...` (in-tree, not snvme) | `snvm_rebind_driver` called the 5.4 helper `device_attach()` which picks the **first** registered matching driver. Since in-tree `nvme.ko` is loaded at boot it always wins, and the BIND silently rebinds to nvme.ko — the next UNBIND then refuses because the driver isn't snvme. Reproducible by running `--gpu` (no bind) immediately followed by `--gpu --bind` on the same BDF. See §7.3.1 trap "`snvm_rebind_driver` MUST use `driver_attach`". | Confirm `snvm_rebind_driver` uses `driver_attach(&snvme_driver.driver)` + bounded retry, not `device_attach(&pdev->dev)`. After the fix the BIND-time dmesg block must contain `snvme: device driver name: snvme` and the queue-summary line must be the 4-tuple `135/0/0/0 default/read/poll/user` variant. |
| `dmesg \| grep "queue squeeze"` after `SNVM_DEVICE_BIND` reports `queue squeeze: kernel=N user=M (controller granted ...)` and the GPU-direct path silently falls back to `dma_alloc_coherent` (smoke `nr_user_q=0` instead of the requested count) | The controller's MSI-X vector budget is smaller than the host's `num_possible_cpus()` (e.g. Intel DC SSD: MSI-X=136 on a 192-vCPU host).  The kernel's `nr_io_queues = num_possible_cpus()` ask consumed every vector, leaving zero room for the user-allocated share, so `s_nvme_setup_io_queues` falls back to host-pinned coherent DMA pages and the GPU-resident SQ/CQ rings never get bound.  Independent of `cap_kernel_ioq=0` (the upstream default).  Verified on HGX H20 + Intel DC SSD (MSI-X=136 vs 192 vCPUs): pre-fix smoke shows `nr_user_q=0`; with `kernel_ioq_cap` set the dmesg line becomes `queue split: kernel=K user=M`. | Set `nvmes[].queue_setup.kernel_ioq_cap` in `sys_config.yaml` so `kernel_ioq_cap + Σqueue_groups[].count <= total_queues` (all in QueuePair units; daemon validates).  Rule of thumb: pick a value low enough to leave the controller's MSI-X grant with room for the user share — typically the smoke-gpu reference value `32`, or whatever your `Identify Controller` MQES-derived per-queue grant supports after subtracting your user pair count.  See `sys_config.yaml`'s `queue_setup` block for the full schema and the "Queue budget tuning" subsection in the root `README.md` for the operator narrative. |

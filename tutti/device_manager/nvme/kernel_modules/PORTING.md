# snvme Kernel Module — Porting Guide

> This document explains how the `snvme` kernel module is structured, what
> it changes on top of upstream NVMe, and how to port it to a new kernel
> version, a new GPU vendor, or a new deployment. It is written for
> maintainers and integrators, not end users.

## 1. Repository layout

```text
tutti/device_manager/nvme/kernel_modules/
├── PORTING.md                       (this document)
├── test/                            kernel smoke-test suite (see §9)
├── snvme-5.4.241-1-tlinux4-0017/    Tencent tlinux4 production lineage
├── snvme-5.15.0-public/             upstream-ish public baseline
└── snvme-6.8.0-public/              Linux 6.8 public baseline
```

Each tree is a **full NVMe driver source tree** (the `nvme.ko` / `nvme-core.ko`
/ `nvme-fabrics` stack) with the snvme modifications applied. The trees
share an identical set of modified files (`compat.{c,h}`, `map.{c,h}`,
`ctrl.{c,h}`, `list.{c,h}`, `pci.c`, `peer_memory/`); tree-specific files
are only the parts that genuinely differ between the kernel versions
(see §7).

**Rule: a change to any shared file must be applied to ALL trees.**
There is no build-time sharing — each tree is self-contained.

## 2. Module pair

The driver builds two kernel modules:

| Module | Role |
|--------|------|
| `snvme-core.ko` | unmodified `nvme-core` (core.c + optional multipath). Provides the block layer, the register-level reset/quirks, and the core queue machinery. |
| `snvme.ko` | the modified PCI driver. Carries the `SNVM_`/`NVM_` ioctl surface, `/dev/snvm_control` + `/dev/ssnvme*` char devices, the CPU/GPU queue-share hooks, and the GPU P2P glue. |

`snvme.ko` is built from: `peer_memory/<backend>.o compat.o list.o ctrl.o
map.o pci.o` (see §6 for the full build).

## 3. What snvme changes on top of upstream NVMe

### 3.1 Queue model: user-visible queue groups

Upstream NVMe queues are opaque to userspace. snvme exposes them:

- Each controller has **32 queue pairs per queue group** (hard cap
  `NVM_MAX_QUEUES_PER_GROUP`); userspace requests beyond the cap are
  clamped to it.
- **Queue depth is kernel-authoritative.** The module parameter
  `io_queue_depth` fixes the depth at install time (default 1024 in the
  production Makefile; the kernel built-in default is conservative). The
  controller reports `q_depth = min(MQES+1, io_queue_depth)`; userspace
  rings must always follow the controller-reported depth. A userspace
  ring smaller than the controller's rings desynchronizes SQ wrap-around
  and CQ phase tracking — there is deliberately **no** userspace
  override.
- Userspace allocates queues via `NVM_ADD_USER_QUEUE` after mount; both
  CPU-side (daemon) and GPU-side queues are created dynamically through
  the batched ioctl surface. Queue objects are allocated by the kernel
  with the full installed depth regardless of what a client asks for.

> **5.4 tree only:** the 5.4 lineage retains a legacy user-queue
> negotiation path driven by `ctrl->use_sreg` + `nr_user_allocated_cq`:
> when user-allocated queues are pinned via `NVM_MAP_*` before
> Set-Features, the queue-count negotiation in `nvme_setup_io_queues` is
> biased upwards to leave room for both kernel-owned and user-owned
> queues. The 5.15 tree has no `use_sreg`; there the user queues sit
> purely on the queue-group path. Keep this divergence in mind when
> porting queue-count logic.

### 3.2 GPU P2P DMA

GPU memory is DMA-mapped to the NVMe controller through the vendor-neutral
`peer_memory/` layer (see §5). The block I/O path is standard NVMe; the
P2P glue only supplies `struct page *`/DMA addresses for GPU buffers at
registration time.

### 3.3 Char-device / ioctl surface

- `/dev/snvm_control` — module-wide operations (device enumeration,
  GPU attach bookkeeping).
- `/dev/ssnvme{N}` — one node per controller (`N` = daemon `device_id`).
  Node exists only after daemon bring-up and controller attach.
- All ioctl command numbers, argument structs, and the ABI version live
  in **one UAPI header**: `tutti/device_manager/nvme/libnvm/include/ioctl.h`.
  Kernel module and userspace library include the same file.

## 4. Kernel-version compatibility layer

The only translation unit allowed to test kernel version macros is
`compat.{c,h}`. Everything else calls the stable wrappers declared there
and never writes `#if LINUX_VERSION_CODE`.

- `compat_get_user_pages()` — pins a user range for DMA, hiding the
  `get_user_pages` signature changes across kernels.
- Feature-probe macros (`HAVE_*`) are selected by the Makefile via grep
  on `Module.symvers` (i.e. by *feature presence*, not by version number):
  - `HAVE_BLK_MARK_DISK_DEAD` — target kernel exports
    `blk_mark_disk_dead` (back-ported into some 5.15 point releases).
    Consumed in `pci.c`; probed in `Makefile.in`.
  - `HAVE_MODULE_MUTEX` — `module_mutex` is directly extern-able.
    Consumed by `peer_memory/` backend sources.

**Porting a new kernel:** put every new version-dependent call behind a
wrapper in `compat.c`, and every optional symbol behind a `HAVE_*` probe
in the Makefile. If you find yourself writing `KERNEL_VERSION` in any
file other than `compat.c`, stop and move it there.

## 5. GPU P2P backends (`peer_memory/`)

`peer_memory/` is the vendor-neutral P2P abstraction:

- The module body (`map.c`, `pci.c`) interacts with GPU memory **only**
  through the `peer_memory_ops` function-pointer table and two opaque
  types (`struct peer_page_table`, `struct peer_dma_mapping`). It never
  names a vendor symbol (`nvidia_p2p_*`, `metax_p2p_*`) and never
  includes a vendor header.
- One `.c` per vendor: `nvidia.c` (CUDA default), `metax.c` (MUSA/MACA
  default). The backend is selected at **compile time**:
  `make TUTTI_P2P_BACKEND=nvidia|metax` (default `nvidia`); only
  `peer_memory/<backend>.o` is linked into `snvme.ko`.
- Vendor symbols are resolved at runtime with `__symbol_get`, so the
  module loads even when the GPU driver stack is absent; failure
  surfaces fail-closed at first P2P use (never a silent fallback).

`peer_memory_ops` members (each backend implements all of them):

| Member | Role |
|--------|------|
| `init` / `exit` | resolve vendor symbols / tear the backend down |
| `get_pages` | pin a GPU VA range → page table (`free_cb` fires if the GPU driver force-reclaims, e.g. process exit) |
| `put_pages` | release a page table |
| `dma_map_pages` / `dma_unmap_pages` | create/destroy a DMA mapping on a peer PCI device (the NVMe controller) |
| `free_dma_mapping` / `free_page_table` | free objects once all references drop (force-release paths) |
| `pt_entries` / `dm_addresses` | accessors for pinned-page count and DMA bus addresses (field layouts differ per vendor) |

**Adding a new GPU vendor:** copy `nvidia.c` as a template, implement
every member of `peer_memory_ops` in `peer_memory/<name>.c`, typedef the
opaque types to whatever wrapper the vendor needs, and add the Makefile
branch for `TUTTI_P2P_BACKEND=<name>`. No other file in the module needs
to change.

> **P2P correctness notes (learned the hard way):**
> - In the NVIDIA force-release callback (`free_cb`), **do not** call
>   `nvidia_p2p_put_pages` from within the callback (it triggers a
>   put-pages warning storm on process exit). Perform the release in the
>   module's own free path instead.
> - The DMA-map of MMIO memory is subject to kernel `dma_map_resource`
>   semantics: RAM-like addresses (`pfn_valid`) are rejected. If a
>   deployment maps GPU BAR memory, prefer `NVreg_DmaRemapPeerMmio=0`
>   (fixed-offset remap) over asking the kernel to map pfn-valid ranges.

## 6. Building and installing

Per tree:

```bash
cd snvme-6.8.0-public               # or another matching baseline
make TUTTI_P2P_BACKEND=nvidia       # backend selector; default nvidia
make insmod IO_QDEPTH=1024          # insmod with io_queue_depth=1024
```

Details:

- The kbuild Makefile (`Makefile.in`, filled in by the repository build)
  sets `src`, `KERNEL_SRC` (= `/lib/modules/$(uname -r)/build`), the
  per-version `ccflags` (which include the NVIDIA driver SDK path
  `/usr/src/nvidia-*/nvidia`), and runs the `HAVE_*` probes against
  `Module.symvers`.
- The repository production build entry point is the root build:
  `cmake --build build --target libnvm tutti_daemon modules` → produces
  `build/module/snvme{,-core}.ko` and `build/bin/tutti_daemon`.
- **Runtime order is strict:** insmod modules → start `tutti_daemon` →
  mount. `/dev/ssnvme*` block devices exist only after daemon bring-up.
  Removing the module removes `/dev/snvme*` and `/dev/ssnvme*` together.

## 7. Kernel tree differences

| Aspect | 5.4.241 tlinux4 | 5.15.0-public | 6.8.0-public |
|--------|-----------------|---------------|--------------|
| `ioctl.c` | none — `NVME_IOCTL_*` lives in `core.c` | separate file | separate file |
| `zns.c` / `hwmon.c` | absent | present | present |
| Split `sysfs.c` / `pr.c` / `auth.c` | absent | absent | present |
| `snvme-rename.sed` | present (symbol-rename sync script) | absent | absent |
| `use_sreg` queue negotiation | present (§3.1) | absent | absent |
| `get_user_pages` wrapper | five-argument GUP | five-argument GUP | four-argument GUP |

Do **not** pull upstream-5.15 internal refactors (e.g. the
`nvme_setup_io_queues_trylock` / `shutdown_lock` churn) into 5.4: that
would change lock acquisition order against `nvme_dev_disable()` /
`nvme_reset_work()`.

## 8. Porting checklist — new kernel / new platform

1. **Layout diff:** copy the tree closest to your target (5.15 for
   ≥5.15, 5.4 lineage for tlinux4-class 5.4/5.10) and diff the top-level
   files against the target kernel's `drivers/nvme/host/` to pick up
   renames/additions (`ioctl.c`, `zns.c`, `hwmon.c`, …).
2. **Compatibility:** compile; route every version-dependent API into
   `compat.c`; add `HAVE_*` probes for optional symbols in `Makefile.in`.
   Common friction points: `get_user_pages` signature variants,
   `module_mutex` export status, `blk_mark_disk_dead` backports,
   `struct request` / `blk-mq` API drift.
3. **Interrupt semantics:** user CQs are polled by GPU threads, so the
   interrupt handler routinely finds a CQ already consumed. Return
   `IRQ_HANDLED` for that case — `IRQ_NONE` lets the kernel's spurious
   detector storm-disable the IRQ line. Keep this semantic on every
   port.
4. **Queue geometry:** confirm `NVM_MAX_QUEUES_PER_GROUP` (32) and the
   `io_queue_depth` parameter wiring survive the port; verify the
   controller-reported `q_depth` against a production-depth ring
   (e.g. with the `/tmp/probe_qdepth` style ioctl probe).
5. **P2P:** re-run the backend compile (`TUTTI_P2P_BACKEND`) against the
   vendor SDK path in `ccflags`; no stub should be needed for nvidia.
6. **UAPI stability:** do not change `libnvm/include/ioctl.h` struct
   layouts — the ABI is versioned and handshaken fail-closed between
   module and userspace (a mismatched pair is rejected with `ENODEV`,
   never silently mis-interpreted). Changing UAPI means a coordinated
   userspace rebuild.
7. **All-tree sync:** apply shared-file changes to every maintained tree and
   rebuild the affected baselines before committing.

## 9. Smoke tests (`test/`)

| File | Exercises |
|------|-----------|
| `snvme_smoke.c` | device enumeration, ioctl surface, ABI handshake |
| `snvme_smoke_io.c` | block I/O round-trip |
| `snvme_smoke_qgroup.c` | queue-group creation / multi-queue |
| `snvme_smoke_addq.c` | dynamic queue addition |
| `snvme_smoke_gpu.cu` | GPU-side queue submit via `cuda_like` |
| `snvme_ubind.c` | unbound/attach lifecycle edge cases |
| `run_snvme_smoke.sh` | driver for the suite |

The GPU smoke requires a mounted device and a live daemon (strict
insmod → daemon → mount order).

## 10. Development conventions

- Shared helper files (`compat.*`, `map.*`, `ctrl.*`, `list.*`,
  `peer_memory/`) should stay synchronized across maintained trees unless a
  kernel API difference is explicitly documented. `pci.c` is ported onto each
  kernel's native NVMe lifecycle and is therefore version-specific.
- The `PORTING STATUS` banner at the top of `pci.c` records per-segment
  port status — update it when you change port-relevant code.
- Kernel module changes land only in
  `tutti/device_manager/nvme/kernel_modules/`; the `third_pkgs/` mirror
  is read-only reference material.
- insmod / rmmod / daemon / mount are manual operator steps, not scripted
  by the build.

## 11. Function-level diff vs upstream 5.4

This section is a **line-by-line inventory of what the snvme 5.4 tree
changes on top of the pristine Tencent-tlinux4 5.4.241 NVMe host driver**
(reference source: `drivers/nvme/host` in the kernel-tlinux4-5.4.241
source tree). The 5.15 tree is structurally the same; its file-level
differences are listed in §7. Function lists below were extracted with
ctags from the two trees.

### 11.1 File-level summary

| Change | File(s) | Notes |
|--------|---------|-------|
| **Added** | `compat.{c,h}` | kernel-version compatibility layer (see §4) |
| **Added** | `ctrl.{c,h}` | per-controller character-device lifecycle (split from `pci.c`) |
| **Added** | `map.{c,h}` | GPU-memory DMA mapping core (the P2P `struct map` registry) |
| **Added** | `list.{c,h}` | intrusive doubly-linked list used by ctrl/map registries |
| **Added** | `peer_memory/` | vendor-neutral GPU P2P backends (see §5) |
| **Added** | `Makefile.in`, `snvme-rename.sed` | kbuild template; global symbol-rename script |
| **Deleted** | `fault_inject.c`, `fc.c`, `lightnvm.c` | fault injection / Fibre Channel / lightnvm not supported (lightnvm also gone upstream by 5.15) |
| **Modified** | `pci.c` | 3350 → 6763 lines: queue-sharing, ioctl surface, char devices, P2P hooks |
| **Modified** | `core.c` | 4396 → 4485 lines: 29 exported functions renamed `nvme_*` → `snvme_*`; 2 functions added |
| **Modified** | `nvme.h` | declarations + renamed-symbol externs + snvme-private fields |
| **Untouched** | `multipath.c`, `fabrics.{c,h}`, `rdma.c`, `tcp.c`, `trace.{c,h}`, `Kconfig` | zero function-level differences |

### 11.2 Symbol renames

**`nvme_*` → `snvme_*` (core.c, 29 exported functions).** Applied by
`snvme-rename.sed` so that the module's core API cannot collide with the
host kernel's own `nvme_*` symbols when both namespaces are visible.
Renamed functions:

`nvme_alloc_request` · `nvme_cancel_request` · `nvme_change_ctrl_state` ·
`nvme_cleanup_cmd` · `nvme_complete_async_event` · `nvme_complete_rq` ·
`nvme_disable_ctrl` · `nvme_enable_ctrl` · `nvme_find_get_ns` ·
`nvme_get_features` · `nvme_init_ctrl` · `nvme_init_identify` ·
`nvme_kill_queues` · `nvme_put_ns` · `nvme_remove_namespaces` ·
`nvme_reset_ctrl` · `nvme_sec_submit` (see below, added) ·
`nvme_set_features` · `nvme_set_queue_count` · `nvme_setup_cmd` ·
`nvme_shutdown_ctrl` · `nvme_start_ctrl` · `nvme_start_freeze` ·
`nvme_start_queues` · `nvme_stop_ctrl` · `nvme_stop_queues` ·
`nvme_submit_sync_cmd` · `nvme_sync_queues` · `nvme_try_sched_reset`

Each is a pure rename (same signature, same body; callers updated);
`nvme.h` declarations follow. **When porting, re-run `snvme-rename.sed`
after touching core.c, or the symbol split breaks the build.**

**`nvme_setup_io_queues` → `s_nvme_setup_io_queues` (pci.c, static).** The
upstream function is wrapped under the `s_` prefix so snvme can inject its
queue-share logic (see §11.3 entry below).

### 11.3 Added functions — `pci.c` (26)

| Function | Role |
|----------|------|
| `snvm_user_qid_pool_init_locked` | init the per-controller user queue-id pool (`ctrl->user_qid_first/last`) |
| `snvm_user_qid_alloc_locked` | carve `nr` queue ids out of the pool |
| `snvm_user_qid_free_locked` | return queue ids to the pool |
| `adapter_alloc_sq_user` | send an admin *Create SQ* to create a userspace SQ (`qsize = dev->q_depth - 1`, PHYS_CONTIG) |
| `adapter_alloc_cq_user` | send an admin *Create CQ* to create a userspace CQ (same depth semantics) |
| `find_qgroup_locked` | look up a queue group by qid under the group lock |
| `destroy_qgroup_locked` | tear down a queue group (and its queues) |
| `snvme_disable_user_io_queues` | disable user queues on controller reset/removal |
| `clear_ctrl_list` / `clear_map_list` | module-unload cleanup of the ctrl/map registries |
| `get_snvme_mode` | sysfs attribute mode callback (0666) |
| `snvm_register_driver` / `snvm_unregister_driver` | register/unregister the `nvme_driver` at module load/exit (with rebind support) |
| `snvm_rebind_driver` | `SNVM_DEVICE_BIND`: detach whatever driver owns the BDF and rebind it to snvme (`pci_get_domain_bus_and_slot`) |
| `snvm_unbind_driver` | `SNVM_DEVICE_UNBIND`: release the BDF from snvme |
| `snvm_cdev_init` / `snvm_cdev_release` | create/destroy the module-level control device `/dev/snvm_control` |
| `snvm_chrdev_create` / `snvm_chrdev_helper` | create the per-device `/dev/ssnvme{N}` (idempotent; tolerates a SIGKILLed owner's stale `struct ctrl`) |
| `snvm_dev_open` / `snvm_dev_release` | open/release for `/dev/ssnvme{N}` (binds file private data to the controller) |
| `snvm_ioctl` | control-plane ioctl dispatcher (`BIND`/`UNBIND`/`CHRDEV_CREATE`/`CHRDEV_REMOVE`) |
| `snvm_dev_map_ioctl` | `NVM_MAP_DEVICE_MEMORY` — authorize mapping controller memory (BAR/registers) to a client |
| `svm_mmap_registers` | `mmap` of the controller BAR0 register window (`pgprot_noncached`; the GPU doorbell aperture) |
| `snvm_ctrl_get_live_ndev` | return the controller's live block device (`ndev`) |
| `s_nvme_setup_io_queues` | the renamed `nvme_setup_io_queues`, extended: samples module params once per reset, folds per-BDF `NVM_SET_IOQ_NUM` overrides, applies the 5.4 `use_sreg` user-queue negotiation bias |

### 11.4 Added functions — `core.c` (2)

| Function | Role |
|----------|------|
| `__snvme_submit_sync_cmd` | generic synchronous admin-command submission (admin timeout, AEN wait, any-queue, arbitrary cmd payload) |
| `snvme_sec_submit` | synchronous Security Send/Recv wrapper (`spsp`/`secp`/len) on top of `__snvme_submit_sync_cmd` — used for security protocol commands such as TCG/OPAL |

### 11.5 Added files' functions

**`ctrl.c`** — per-controller cdev lifecycle:
`ctrl_get`/`ctrl_put` (refcount), `ctrl_find_by_pci_dev` (by BDF),
`ctrl_find_by_inode` (by char-device inode), `ctrl_cdev_release`,
`ctrl_chrdev_create`/`ctrl_chrdev_remove` (create/remove `/dev/ssnvme{N}`).

**`map.c`** — the GPU-memory DMA mapping registry (18 functions):
`create_descriptor` (allocate a `struct map` with inline `addrs[]`, self-loop
`group_link` so `list_del` stays safe when never attached),
`map_p2p_service_probe`/`map_p2p_service_release` (resolve/release the
vendor P2P service), `map_find` (lookup), `map_purge_by_owner` (drop all
mappings of one client), `map_user_pages`/`release_user_pages` (pin/release
a plain user range), `map_userspace` (map a user VA), `unmap_and_release`,
`map_gpu_memory`/`release_gpu_memory`/`force_release_gpu_memory` (pin,
release, and force-release a GPU buffer's DMA mapping — the
force-release path is used by the vendor's free callback),
`map_gpu_ioqueue_memory`/`release_gpu_ioqueue_memory` (GPU-side queue memory
mapping), `map_device_ioqueue_memory` (map a GPU-page-aligned range for a
controller's IO queues, GPU_PAGE_SIZE granularity),
`force_release_gpu_ioqueue_memory` (force-release counterpart).

**`list.c`** — minimal intrusive list: `list_init`, `list_insert`,
`list_remove`.

**`compat.c`** — `compat_get_user_pages` (version-neutral user-range pinning,
see §4).

### 11.6 Non-additive modifications to existing functions

Function-level ctags diffs miss behavioral edits inside unchanged
signatures. The important ones in `pci.c`/`core.c`:

- **`nvme_probe`** — extended with snvme initialization: controller
  registration in `ctrl_list`, user queue-id pool setup, map-list
  registration, char-device creation.
- **`nvme_irq`** — returns `IRQ_HANDLED` even when the GPU already
  consumed the CQ entries (the kernel would otherwise count them as
  spurious and storm-disable the IRQ line).
- **Queue creation** — SQ/CQ `qsize` is taken unconditionally as
  `dev->q_depth - 1` (no negotiation); the kernel allocates queue objects
  at full installed depth.
- **`core.c`** — the renamed functions' bodies are unchanged except where
  the `snvme_*` prefix propagates into calls; the ioctl surface and ABI
  handshake live in `libnvm/include/ioctl.h` (userspace side).

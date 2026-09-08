# Kernel Portability — the snvme module across Linux versions and GPU vendors

> How the `snvme` kernel module stays buildable across kernel lineages
> and GPU vendors, and the userspace/kernel contract that keeps the pair
> in lockstep.

## 1. Unified kernel tree

```text
tutti/device_manager/nvme/kernel_modules/
├── snvme/                           # active unified module tree
│   ├── baseline/5.4-tlinux4/
│   ├── baseline/5.10/
│   ├── baseline/5.15/
│   └── baseline/6.8/
└── snvme-*/                         # deprecated historical trees
```

CMake builds the active `snvme/` tree. Its Kbuild `Makefile.in` selects a
baseline from the running kernel; the versioned `snvme-*` trees remain only as
historical references. `third_pkgs/` is a read-only upstream mirror. Shared
logic lives once in `snvme/` (`map.c`, `peer_memory/`, `compat/`); kernel
version and vendor differences are isolated into two units:

| Unit | Sole owner of | Isolates |
|------|---------------|----------|
| `compat.{h,c}` | `LINUX_VERSION_CODE` / `KERNEL_VERSION` | kernel API drift (e.g. `get_user_pages` signature changes) — the only translation unit allowed to test kernel version macros |
| `peer_memory/` | vendor P2P headers (`nv-p2p.h` etc.) | GPU-vendor P2P mapping — one `.c` per vendor, symmetric |

Rule of thumb: `map.c` and everything above never `#if` on kernel
version and never name a GPU vendor symbol directly — they call
`compat_*` / `peer_memory_ops.*`.

## 2. The peer_memory backend layer (GPU P2P)

GPU page pinning and DMA mapping go through an ops table:

- **Opaque types** — `peer_page_table`, `peer_dma_mapping` hide the
  vendor's page-table representation from `map.c`; accessors replace
  direct `->entries` / `->dma_addresses` walks.
- **One backend per vendor** — `peer_memory/nvidia.c` wraps
  `nvidia_p2p_get_pages` / `nvidia_p2p_dma_map_pages`; `peer_memory/metax.c`
  is the symmetric Metax backend. Exactly one backend is selected at compile
  time through `TUTTI_P2P_BACKEND`; that backend resolves its vendor symbols
  dynamically (`__symbol_get`) during module initialization.
- **Build isolation** — only the backend `.c` includes vendor headers;
  CMake discovers the selected backend's header directory and passes it to
  Kbuild. CUDA defaults to `nvidia`; MUSA/MACA default to `metax`.

## 3. Userspace ABI handshake

The ioctl UAPI is versioned (`tutti/include/uapi/tutti_snvme.h`):

- `TUTTI_SNVME_ABI_VERSION` is negotiated at attach; a module older than
  the library's minimum is **rejected fail-closed** (`ENODEV`) — never a
  silent fallback to mismatched layouts.
- Consequence: swapping kernel modules requires rebuilding userspace
  (`libnvm`, `tutti_daemon`) against the matching headers. After configuring
  `MODULE_BUILD` with [`../getting-started.md`](../getting-started.md), run
  `cmake --build "$MODULE_BUILD" --target libnvm tutti_daemon modules` to
  produce the matched set together.
- UAPI structs are plain-C layout-stable; `libnvm` compiles its device
  headers under both nvcc and plain C via layout-identical fallbacks for
  the GPU-side atomic fields.

## 4. Queue geometry — controller maximum

- Queue depth always takes the **controller maximum**:
  `q_depth = NVMe CAP.MQES + 1` (no module parameter). Userspace rings always
  follow the controller-reported depth — there is no userspace override
  (a smaller userspace ring would desynchronize SQ wrap-around and CQ
  phase tracking against the controller's deeper rings).
- Queue count: up to **32 queue pairs per queue group**
  (`NVM_MAX_QUEUES_PER_GROUP`); userspace requests beyond that are
  clamped. Both CPU-side (daemon) and GPU-side queues are allocated
  dynamically after mount via batched ioctl.
- The kernel allocates queue objects with the full installed depth
  regardless of what any single client asks for.

## 5. Interrupt semantics under GPU-polled completions

User CQs are polled by GPU threads (phase-bit), so the kernel's interrupt
handler routinely finds a CQ already consumed by the GPU. The handler
reports `IRQ_HANDLED` for that case — returning `IRQ_NONE` would let the
kernel's spurious-interrupt detector storm-disable the IRQ line. This is
a deliberate semantic: GPU-consumed ≠ spurious.

## 6. Build & test entry points

- Production matched set: configure `MODULE_BUILD` through
  [`../getting-started.md`](../getting-started.md), then run
  `cmake --build "$MODULE_BUILD" --target libnvm tutti_daemon modules`.
  It produces `$MODULE_BUILD/module/snvme{,-core}.ko` and the matching CUDA
  userspace targets.
- Baseline matrix: each tree compiles against its own headers;
  cross-compiling a tree against a different lineage's headers is a
  known-incompatible configuration (NVMe core API drift), not a compat
  defect.
- Runtime order is strict: **insmod → daemon → mount** — the
  `/dev/snvme*` block devices exist only after daemon bring-up.

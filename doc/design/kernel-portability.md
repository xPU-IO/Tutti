# Kernel Portability — the snvme module across Linux versions and GPU vendors

> How the `snvme` kernel module stays buildable across kernel lineages
> and GPU vendors, and the userspace/kernel contract that keeps the pair
> in lockstep.

## 1. Two maintained kernel trees

```text
tutti/device_manager/nvme/kernel_modules/
├── snvme-5.4.241-1-tlinux4-0017/   # tlinux4 production lineage
└── snvme-5.15.0-public/            # upstream-ish public baseline
```

Both trees are built from this repository; `third_pkgs/` holds the
read-only upstream mirror for reference/diffing. Shared logic lives in
identical files across the two trees (`map.c`, `peer_memory/`,
`compat.c`); version/vendor differences are isolated into exactly two
units per tree:

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
  (`libnvm`, `tutti_daemon`) against the matching headers. The root
  production build (`cmake --build --preset cuda-module --target libnvm
  tutti_daemon modules`) produces the matched set together.
- UAPI structs are plain-C layout-stable; `libnvm` compiles its device
  headers under both nvcc and plain C via layout-identical fallbacks for
  the GPU-side atomic fields.

## 4. Queue geometry — fixed at install time

- Queue depth is a **module-install-time parameter**:
  `insmod snvme.ko io_queue_depth=1024` (production). The controller
  reports `q_depth = min(MQES+1, io_queue_depth)`; userspace rings always
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

- Production matched set: `cmake --build --preset cuda-module --target
  libnvm tutti_daemon modules` produces
  `build/cuda-module/module/snvme{,-core}.ko` plus the CUDA userspace targets.
- Baseline matrix: each tree compiles against its own headers;
  cross-compiling a tree against a different lineage's headers is a
  known-incompatible configuration (NVMe core API drift), not a compat
  defect.
- Runtime order is strict: **insmod → daemon → mount** — the
  `/dev/snvme*` block devices exist only after daemon bring-up.

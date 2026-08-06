# GPU Vendor Porting Guide (Tutti `cuda_like` Framework)

This document describes the three-layer GPU-vendor abstraction in Tutti
and the integration steps a new vendor (MUSA / MACA / future) must follow
to plug their device compiler + runtime SDK into the project.

## Audience

- **Tutti maintainers** — own the framework contracts.
- **Vendor port engineers** (e.g. Metax 沐曦) — fill in the per-vendor
  shim and compile their toolchain against Tutti's kernels.

## The three layers

```
┌─────────────────────────────────────────────────────────────┐
│ Layer A: Vendor selection framework                          │
│   cmake/accelerators/<VENDOR>.cmake     ← toolchain + libs   │
│   include/tutti/gpu_vendor/<vendor>.h   ← runtime API shim   │
│   include/tutti/cuda_like.h             ← profile selector  │
├─────────────────────────────────────────────────────────────┤
│ Layer B: Kernel primitives macro layer                       │
│   data_paths/.../io/tutti_gpu_primitives.cuh                │
│   ↑ device-only macros (TUTTI_DEVICE, TUTTI_THREADFENCE_*,   │
│     TUTTI_ATOMIC_ADD, ...) with semantic contracts           │
│   vendor fills: include/tutti/gpu_vendor/<vendor>_primitives.h│
├─────────────────────────────────────────────────────────────┤
│ Layer C: Direct includes → cuda_like                         │
│   All host/.cu/.cuh files include <tutti/cuda_like.h>       │
│   instead of <cuda_runtime.h> directly                       │
└─────────────────────────────────────────────────────────────┘
```

A single profile is selected at CMake configure time
(`-DTUTTI_ACCELERATOR=CUDA|HOST|MUSA|MACA`). The build is single-vendor
(no runtime coexistence). Each profile wires its own:

- Runtime API surface (cudaMalloc / cudaMemcpy / cudaStream_t / ...)
- Device compiler (nvcc / mcc / ...)
- Architecture flags
- Link libraries

## Layer A — Vendor selection framework

### `cmake/accelerators/<VENDOR>.cmake`

Each profile provides a `tutti_configure_cuda_like(<target_name>)`
function that:

1. Defines `TUTTI_USE_<VENDOR>=1` on the target.
2. Adds `include/` to the target's include path.
3. Links the vendor's runtime + driver libraries.

Each profile also sets two CMake variables that downstream CMakeLists
use to link the runtime/driver without naming the vendor-specific target:

```cmake
set(TUTTI_ACCEL_RUNTIME_LIBS "CUDA::cudart" CACHE INTERNAL "")
set(TUTTI_ACCEL_DRIVER_LIBS  "CUDA::cuda_driver" CACHE INTERNAL "")
```

Downstream targets link `${TUTTI_ACCEL_RUNTIME_LIBS}` /
`${TUTTI_ACCEL_DRIVER_LIBS}` (never `CUDA::cudart` directly) so vendor
ports don't touch every CMakeLists.txt.

**Vendor step A.1**: copy `cmake/accelerators/MUSA.cmake` (or `MACA.cmake`)
and replace the TODO markers:

```cmake
# TODO(Metax): locate MUSA SDK
find_path(MUSA_INCLUDE_DIR musa_runtime.h PATHS /opt/musa/include)
find_library(MUSA_RUNTIME_LIB  NAMES musart  PATHS /opt/musa/lib)
find_library(MUSA_DRIVER_LIB   NAME  musa     PATHS /opt/musa/lib)
target_include_directories(${target_name} INTERFACE ${MUSA_INCLUDE_DIR})
target_link_libraries(${target_name} INTERFACE ${MUSA_RUNTIME_LIB} ${MUSA_DRIVER_LIB})
set(TUTTI_ACCEL_RUNTIME_LIBS "${MUSA_RUNTIME_LIB}" CACHE INTERNAL "")
set(TUTTI_ACCEL_DRIVER_LIBS  "${MUSA_DRIVER_LIB}"  CACHE INTERNAL "")
```

If the vendor compiler is NOT nvcc, enable the device language in the
profile module:

```cmake
enable_language(MUSA)  # or whatever the toolchain exposes
```

… and adjust `project(... LANGUAGES ...)` in the root CMakeLists.txt
accordingly.

### `include/tutti/gpu_vendor/<vendor>.h`

A header that supplies the runtime API surface Tutti's host code calls.
This is the **host-side** shim — it does NOT define kernel-side primitives
(those live in Layer B).

Look at `include/tutti/gpu_vendor/host.h` for the full surface
(expected symbols: cudaMalloc, cudaFree, cudaMemcpy, cudaStreamCreate,
cudaEventCreate, cudaGetDeviceProperties, cudaPointerGetAttributes, …).

**Vendor step A.2**: replace the `#error` at the top of
`include/tutti/gpu_vendor/musa.h` with the real SDK `#include` and
define / typedef every symbol listed in the TODO skeleton.

### `include/tutti/cuda_like.h`

The profile selector — single header that downstream files include.
It dispatches based on `TUTTI_USE_<VENDOR>` and includes the matching
vendor header. Already wired; vendor does not modify this file.

## Layer B — Kernel primitives macro layer

File: `tutti/data_paths/local_nvme/io/tutti_gpu_primitives.cuh`

This file provides macros for the CUDA C++ language extensions used
by Tutti's IO kernels. Each macro has a **semantic contract** documented
in the file. The macros fall into groups:

| Group | Macros | Notes |
|-------|--------|-------|
| Function attrs | `TUTTI_DEVICE`, `TUTTI_HOST`, `TUTTI_GLOBAL`, `TUTTI_FORCEINLINE` | Mark function entry / linkage |
| Variable attrs | `TUTTI_SHARED` | Per-block shared memory |
| Built-ins | `TUTTI_THREAD_IDX_X`, `TUTTI_BLOCK_IDX_X`, `TUTTI_BLOCK_DIM_X` | Thread / block indices |
| **Fences** | `TUTTI_THREADFENCE_SYSTEM` (**CRITICAL**), `TUTTI_THREADFENCE`, `TUTTI_SYNC_THREADS` | Memory / control barriers — see contract |
| Atomics | `TUTTI_ATOMIC_ADD`, `TUTTI_ATOMIC_CAS` | Device-global scope |
| Warp | `TUTTI_SHFL_SYNC`, `TUTTI_BALLOT_SYNC` | Full-warp mask required |
| Timing | `TUTTI_NANOSLEEP`, `TUTTI_CLOCK`, `TUTTI_CLOCK64` | Best-effort |

### Why `TUTTI_THREADFENCE_SYSTEM` is the命根

NVMe doorbell writes must observe a precise memory-ordering contract:

1. CPU / GPU writes the SQE (submission queue entry) into the SQ ring.
2. The thread issues `TUTTI_THREADFENCE_SYSTEM` to flush all outstanding
   writes — including the SQE — to a point of **system-wide visibility**
   (GPU + host + peer PCIe / NVMe controller).
3. Then the thread writes the doorbell register (`sq.db`).

If the fence is missing or has weaker scope (device-only), the NVMe
controller may observe a stale doorbell value (wrote SQ tail = N, but
the SQE for slot N is not yet visible to the controller) → silent IO
corruption / hang.

The contract is documented in the header — every vendor port MUST
preserve it.

### Vendor step B.1: provide `<vendor>_primitives.h`

Create `tutti/include/tutti/gpu_vendor/musa_primitives.h` (or
`maca_primitives.h`) with the macro definitions for the vendor's device
compiler. Two cases:

**Case 1 — vendor compiler accepts CUDA-style keywords verbatim**
(the most common case for CUDA-like SDKs). Copy the CUDA block from
`tutti_gpu_primitives.cuh`:

```cpp
#pragma once
#define TUTTI_DEVICE              __device__
#define TUTTI_GLOBAL              __global__
#define TUTTI_FORCEINLINE         __forceinline__
#define TUTTI_THREAD_IDX_X        threadIdx.x
// ... (copy from tutti_gpu_primitives.cuh CUDA branch)
```

**Case 2 — vendor compiler uses different syntax**. Provide the MUSA /
MACA equivalents per the contract table in `tutti_gpu_primitives.cuh`.

`tutti_gpu_primitives.cuh` already uses `__has_include(...)` to pick up
your vendor header, so once the file exists the build will work.

### Kernel launch syntax (`<<<>>>`)

Tutti's host launchers (`submit_one.cu`, `fused_submit_kernel.cu`) use
CUDA's `kernel<<<blocks, threads, smem, stream>>>(args...)` triple-bracket
syntax. Every CUDA-like compiler we have surveyed (nvcc, mcc, …) accepts
this syntax. If the vendor's compiler does NOT, raise it on the Tutti
issue tracker — the launchers will be updated to use a
`TUTTI_LAUNCH_KERNEL` macro.

## Layer C — Direct includes

Every host / `.cu` / `.cuh` file in `tutti/` includes `<tutti/cuda_like.h>`
instead of `<cuda_runtime.h>` directly. The single source of truth for
"which vendor are we building against" is the `TUTTI_USE_<VENDOR>`
compile definition propagated by the `tutti_cuda_like` interface target.

### Files NOT modified (libnvm upstream headers)

The following headers under `tutti/device_manager/nvme/libnvm/include/`
are upstream libnvm sources and continue to `#include <cuda.h>` /
`#include <cuda_runtime.h>` / `#include <cuda/atomic>` directly:

- `buffer.h`, `event.h`, `queue.h`, `util.h` — `#include "cuda.h"`
- `nvm_util.h` — `#include <cuda.h>`, `#include <cuda_runtime.h>`
- `nvm_types.h` — `#include <cuda/atomic>`, `#include <cuda_runtime.h>`
  (under `#ifdef NVM_CUDA_ENABLED`)

Modifying these creates upstream divergence. Two options for the vendor:

1. **Patch the headers locally** (fork of libnvm) — straightforward,
   maintain a small diff.
2. **Provide a vendor shim header** that libnvm's `#include "cuda.h"`
   picks up via `-I` — e.g. create `<vendor_sdk>/include/cuda.h` that
   `#include`s the MUSA / MACA equivalent. Less invasive but requires
   the include path ordering to land the shim ahead of the system CUDA
   path.

The choice is the vendor's; document the decision in the vendor's
internal Tutti fork.

## Kernel module P2P adaptation (Metax)

The snvme kernel module (`tutti/device_manager/nvme/kernel_modules/`) uses
the `peer_memory_ops` function-pointer table in `peer_memory/peer_memory.h` as the
single point of contact with the GPU driver's P2P API.

### Current state

- **NVIDIA path** (CUDA default): `peer_memory/nvidia.c` resolves `nvidia_p2p_*`
  symbols at module-load time via `__symbol_get` and routes calls through
  the `peer_memory_ops` table. The module body (`map.c`, `pci.c`) never
  touches NVIDIA types directly.
- **Metax path** (MUSA/MACA default): `peer_memory/metax.c` includes
  `metax_p2p.h` (ported from ljye2023/Tutti PR #1) and resolves
  `metax_p2p_*` symbols at module-load time via `__symbol_get` (symmetric
  to the NVIDIA path). The `peer_memory_ops` implementation is complete;
  it works as soon as the Metax driver is loaded and exports the symbols.

### `metax_p2p.h` API (ported from Mooncake/Metax PR)

The vendor-supplied `metax_p2p.h` defines these functions. Tutti does not
currently vendor this header; provide its directory through
`SNVME_P2P_INCLUDE_DIR`.

| Function | Purpose | NVIDIA equivalent |
|----------|---------|-------------------|
| `metax_p2p_acquire_mem` | Pin a GPU VA range | `nvidia_p2p_get_pages` |
| `metax_p2p_get_mem` | Get bus addresses as sg_table | (part of dma_map_pages) |
| `metax_p2p_put_mem` | Unpin mem | `nvidia_p2p_dma_unmap_pages` |
| `metax_p2p_release_mem` | Release handle | `nvidia_p2p_put_pages` |
| `metax_p2p_get_pci_dev` | Get GPU's PCI device | (from page_table) |
| `metax_p2p_get_bus_offset` | Bus→CPU address offset | (implicit in dma_mapping) |
| `metax_p2p_get_page_size` | Page size of pinned region | (GPU_PAGE_SIZE) |

### Metax integration steps

1. Configure with `-DTUTTI_BUILD_KERNEL_MODULE=ON`,
   `-DTUTTI_P2P_BACKEND=metax`, and
   `-DSNVME_P2P_INCLUDE_DIR=<directory-containing-metax_p2p.h>`.
2. Load the Metax GPU driver (must export `metax_p2p_*` symbols to the
   kernel symbol namespace).
3. `insmod snvme.ko` — `peer_init()` resolves the symbols via
   `__symbol_get`; if the Metax driver is not loaded, `peer_init()`
   fails gracefully (module load fails with an error message, same as
   the NVIDIA path when `nvidia_p2p_*` symbols are missing).
4. No SNVMe common-source changes are needed — the backend implementation
   lives in `peer_memory/metax.c`.

The NVIDIA→Metax API mapping table is documented inline in
`peer_memory/metax.c`.

### Note on `map.c` signature differences

The PR (#1) also modifies `map.c`'s call to `nvfs_nvidia_p2p_dma_map_pages`
to add `page_size` and `n_addrs` parameters, and changes `pages->entries`
to `pages->virtual_entries`. These are **Metax driver API differences** —
in Tutti's architecture, `map.c` calls through the `peer_memory_ops`
function-pointer table, so the signature is fixed by
`peer_memory.h`. The Metax implementation of `peer_memory_ops` handles
the API differences internally; `map.c` is not modified.

## Verification

A vendor port is complete when:

1. **CUDA profile**: zero behavioral change — `ctest -LE "hardware"`
   passes 15/15 (the hardware-free contract suite).
2. **HOST profile**: `ctest -L host` passes (excluding the
   `mount_manager` test which depends on nvmeservice and is host-broken
   by design).
3. **Vendor profile**: `cmake -S . -B build -DTUTTI_ACCELERATOR=<VENDOR>`
   configures successfully with the vendor's toolchain, and the
   stub-removal `#error`s in `<vendor>.h` no longer fire.
4. **Vendor's device compiler** compiles the `__CUDACC__`-guarded section
   of `submit_one.cuh` / `fused_submit_kernel.cuh` /
   `nvme_submit_primitives.cuh` (these are the device code entry points).
5. **Hardware smoke**: run `tutti_layerwise_kv_overlap` (KV-cache
   pipeline simulator) end-to-end. If the port is sound, this exercises
   the kernel primitives + launch path.

## Reference: `host.h` as the canonical shim

The HOST profile (`include/tutti/gpu_vendor/host.h`) is the
**reference implementation** of the Layer A surface — it implements
the entire cuda_like API on top of `std::malloc` / `std::memcpy` /
`std::memset` with no GPU at all. Vendors should mirror its symbol
layout (same names, same signatures) so call sites compile unchanged.

## Quick start for Metax (MUSA / MACA)

**大部分工作已经从 Mooncake 搬过来并验证。** Metax 只需补齐 libnvm upstream 头的兼容层。

### 已从 Mooncake 搬过来的确定性实现

| 文件 | 来源 | 状态 |
|------|------|------|
| `tutti/include/tutti/gpu_vendor/musa.h` | Mooncake `gpu_vendor/musa.h`（Apache 2.0） | ✅ 完整 runtime API 宏映射（`cuda*` → `musa*`），去掉 Mooncake 特有的 `GPU_PREFIX` 和 IBGDA driver API |
| `tutti/include/tutti/gpu_vendor/maca.h` | Mooncake `gpu_vendor/maca.h`（Apache 2.0） | ✅ 完整 runtime API 宏映射（`cuda*` → `mc*`），含模板函数 `cudaHostGetDevicePointer` / `cudaFuncGetAttributes` |
| `tutti/include/tutti/gpu_vendor/musa_primitives.h` | Mooncake `musa_ops.cuh` 验证 | ✅ Case 1（mcc 接受 CUDA-style 关键字），含 Mooncake 验证过的已知 SDK 4.3.3 bug 注释（`atomicAdd_system` SelectionDAG bug、named barriers 不可用、grid sync 不可用——Tutti 的 kernel 都不使用这些有 bug 的特性） |
| `tutti/include/tutti/gpu_vendor/maca_primitives.h` | Mooncake `maca_ops.cuh` 验证 | ✅ Case 1（cu-bridge 接受 CUDA-like intrinsics），Mooncake 确认 `__threadfence_system` / `__ldg` / `__syncthreads` 可用 |
| `tutti/cmake/accelerators/MUSA.cmake` | Mooncake `common.cmake` 验证 | ✅ SDK 路径 `/usr/local/musa/include` + `/usr/local/musa/lib`，库名 `musa musart` |
| `tutti/cmake/accelerators/MACA.cmake` | Mooncake `common.cmake` 验证 | ✅ SDK root `$MACA_HOME` 或 `/opt/maca`，include `${MACA_ROOT}/include`，lib `${MACA_ROOT}/lib64` 或 `${MACA_ROOT}/lib` |

### Metax 剩余工作（仅 libnvm upstream 头兼容层）

libnvm upstream 头（`tutti/device_manager/nvme/libnvm/include/`）仍直引 `cuda.h` / `cuda_runtime.h` / `cuda/atomic`：

- `buffer.h`, `event.h`, `queue.h`, `util.h` — `#include "cuda.h"`
- `nvm_util.h` — `#include <cuda.h>`, `#include <cuda_runtime.h>`
- `nvm_types.h` — `#include <cuda/atomic>`, `#include <cuda_runtime.h>`（`__CUDACC__` guard 内）

Metax 选一种：

1. **Vendor shim `cuda.h`**（推荐，零 fork）：在 MUSA SDK 的 include 路径下放一个 `cuda.h` → `#include <musa.h>` + 宏映射。`cuda_runtime.h` 同理。`cuda/atomic` → 创建 `cuda/atomic` 头文件转发到 `musa/atomic`。这样 libnvm 头不改一行。

2. **Fork patch libnvm**：把 libnvm 头里的 `cuda.h` / `cuda_runtime.h` 改为 `#include <tutti/cuda_like.h>`，`cuda/atomic` 改为 vendor 等价物。维护一个小 diff。

方案 1 更轻量——Metax 的 SDK 已经提供了 `musa.h` / `musa_runtime.h`（Mooncake 验证），只需要补一个 `cuda/atomic` 转发头即可。

### 验证步骤

1. `cmake -S . -B build -DTUTTI_ACCELERATOR=MUSA`（configure 应该成功，无 WARNING）
2. `cmake --build --preset cuda --target tutti_layerwise_kv_overlap`（如果链接成功且跑通，框架级通过）
3. 跑 CUDA profile 回归确认零影响：`cmake --preset cuda --fresh && ctest --preset cuda -LE "hardware|mount_manager"`

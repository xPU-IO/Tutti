# Hardware & Host Environment

Tutti is a hardware/software co-designed system: it takes NVMe devices away from
the kernel driver, drives their queues from CUDA kernels, and DMAs into GPU
memory. Consequently a wrong *host* configuration does not degrade performance —
it makes the system non-functional, or destroys data. Read this before touching a
machine that is not already known-good.

## Prerequisites that are not negotiable

| Requirement | Why | How to verify |
|---|---|---|
| GPU and NVMe under the same PCIe switch | P2P DMA between them; crossing the root complex either fails or collapses bandwidth | `lspci -tv`, compare the switch ancestry of both BDFs |
| NVMe devices free for Tutti to claim | Tutti binds them itself; they must not be in use by the kernel `nvme` driver or hold a mounted filesystem you care about | `lsblk`, `lsof` on the mount points |
| IOMMU in passthrough | the GPU-BAR path assumes physical == bus address | `status` checks it (`iommu=pt` on the kernel cmdline) |
| Kernel headers matching the **running** kernel | out-of-tree modules | `uname -r` vs `/lib/modules/$(uname -r)/build` |
| CUDA toolkit + a CUDA-capable `torch` in the target interpreter | extensions link against both | `status` |

IOMMU is worth spelling out because its failure mode is silent: with an IOMMU
enabled but *not* in passthrough, physical ≠ bus address, and DMA lands at the
wrong place rather than returning an error. IOMMU fully disabled is also fine —
`phys == bus` holds either way.

The only prerequisite in this table `status` cannot check is the PCIe pairing,
because deciding which GPU should own which NVMe requires reading intent, not just
topology. Do it by hand on a new machine.

`MDTS` (maximum data transfer size) is **reported by the device, not configured**.
It bounds a single IO and therefore how many sub-IOs one logical transfer splits
into. Never hardcode a value derived from one machine — read it from the
controller. A contract test once encoded "MDTS ≥ 256 KiB" and failed on hardware
reporting 128 KiB, while the code under test was correct.

Likewise `queue_depth` is fixed by the kernel module at install time; user space
must read `ctrl->q_depth` (see invariant 5 in `SKILL.md`).

## `config/local/tutti_daemon.yaml` is the single source of truth

It declares, per NVMe: `pci_addr`, `backing_mount_path`, `auto_mount`, and which
accelerators may use it. This file — not a script, not an environment variable —
is what must match the physical machine.

Two failure shapes when it is wrong:

- **A `pci_addr` that does not exist** → daemon bring-up fails loudly. Harmless.
- **A `pci_addr` that exists but pairs a GPU with a topologically distant NVMe**
  → everything works and throughput silently collapses. Only `lspci -tv` catches
  this. Verify the pairing whenever the file is edited.

## Bring-up order is fixed

```
kernel modules  →  tutti_daemon  →  workload
```

The daemon is what creates `/dev/snvme*n1`, performs `auto_mount`, and serves
gRPC on :50051. Mounting before the daemon runs fails because the block devices
do not exist yet. `scripts/tutti-env.sh bootstrap` / `up` enforce this order;
prefer them over hand-typed `insmod` + `mount`.

## Kernel module / user space ABI handshake

The module and `libnvm` negotiate `TUTTI_SNVME_ABI_VERSION` through a shared
`uapi` header, and the check is **fail-closed** (`ENODEV`). Replacing the module
without rebuilding user space — or the reverse — produces a device-not-found
error that looks like missing hardware.

Use `scripts/tutti-env.sh modules rebuild`, which rebuilds both sides. Do not
`insmod` a `.ko` from an older build tree.

Module sources exist per kernel version under
`csrc/device_manager/nvme/kernel_modules/`. Modules are never signed by the
script: if load fails and `dmesg` reports "Required key not available", sign
`build/module/*.ko` per your site's procedure.

## Destructive operations — require explicit user confirmation

An agent must **not** perform any of the following on its own initiative. Each
has consequences outside the repository, and several are irreversible.

1. **`rmmod snvme` / `modules unload`** — every `/dev/snvme*` and `/dev/ssnvme*`
   disappears instantly. Any running workload dies; mounted filesystems on those
   devices must be unmounted first. Check for other users first:
   `pgrep -a -f 'vllm|tutti|python' ` and `mount | grep snvme`.

2. **Starting/stopping `tutti_daemon`, loading/unloading modules** — these are
   *host-level* actions with root privilege, not per-venv. They affect every
   process on the machine, including other people's jobs.

3. **Loading `phoenixfs` alongside `nvidia_peermem`** — structurally mutually
   exclusive. `phoenixfs` remaps the whole GPU BAR with `devm_memremap_pages`,
   giving those physical addresses a `struct page`; NVIDIA's peer-memory path
   then hits `dma_map_resource`, which refuses addresses backed by `struct page`
   ("Don't allow RAM to be mapped") and surfaces as `EIO` on RDMA memory
   registration. `status` reports this as a **warning**, not a failure — the
   local-NVMe path is unaffected, so it only blocks machines that also run RDMA.
   **Before any RDMA path, `rmmod phoenixfs`.** Use `--no-phoenix` to keep the
   script from loading it.

4. **Editing `pci_addr`** — once claimed, a device is written as raw LBAs.
   Pointing Tutti at the wrong disk destroys whatever is on it. Confirm with
   `lsblk` / `blkid` before changing the value.

5. **Deleting `build/`** — the deployed extensions' RUNPATH points into the build
   tree, so every `.so` on the machine breaks at import (`ImportError:
   libnvm.so`). Sources are unaffected, but recovery requires a rebuild. Never
   run `rm -rf` with a relative path from the repository root.

6. **`fsfreeze -f`** — if any dead process still holds a superblock reference,
   this blocks forever in `percpu_down_write` and hangs *all* writes to that
   filesystem, unrecoverable without a reboot. Do not use it to "make an
   experiment safe".

7. **Bulk deletes of KV pool directories** — may be intercepted by a safe-delete
   shim; use `shutil.rmtree` from Python. Better: an interrupted benchmark
   poisons its pool, so switch to a new pool root instead of cleaning the old one.

## Bringing up a new machine, in order

Do not skip ahead: each step's failure mode is only readable if the previous one
is known-good.

1. `lspci -tv` — record GPU↔NVMe switch pairing.
2. Write `config/local/tutti_daemon.yaml` to match it.
3. `scripts/tutti-env.sh bootstrap` (needs sudo for modules + daemon).
4. `scripts/tutti-env.sh status` — **must be entirely green** before anything
   else. A single red line makes every later symptom ambiguous.
5. `scripts/tutti-env.sh test cpp` — hardware contract tests; they write to the
   real devices and need their mount points to exist and be writable.
6. `scripts/tutti-env.sh test py` — no hardware needed.
7. One single-GPU, small-capacity benchmark run.
8. Only then multi-GPU / striped configurations, sizing capacity and queues per
   `benchmarking.md`.

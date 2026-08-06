# NVMeService

> Session broker for SNVMe (post L1 Commit 4b).
>
> Daemon owns the chrdev / bind / GPU-view symlinks; clients drive
> their own libnvm controllers and queue groups.  Kernel's user QID
> pool is the single source of truth for queue accounting.

---

## TL;DR

```
                                         /dev/snvm_control
                                                ▲
                                                │ owner-only ioctls
                                                │ (CHRDEV_CREATE, BIND, …)
   ┌────────────────────────────┐               │
   │  nvmeservice_daemon        │───────────────┘
   │   (one process, root)      │
   │                            │     gRPC :50051
   │   bring up /dev/ssnvmeN    │  ◀───────────────  ┌──────────────────────┐
   │   install GPU symlinks     │                    │ nvmeservice_client   │
   │   ACL + lease bookkeeping  │  Connect           │   = your app + libnvm│
   │                            │  ──────────────▶   │                      │
   └────────────────────────────┘                    │  open /dev/ssnvmeN   │
                                                     │  attach_client       │
                                  metadata only,     │  create_group        │
                                  no IPC handles     │  add_user_queue ×N   │
                                                     │  ┌────────────────┐  │
                                                     │  │ GPU IO kernel  │  │
                                                     │  └────────────────┘  │
                                                     │  destroy_group       │
                                                     │  free_client         │
                                                     └──────────────────────┘
```

The daemon never sees a SQ/CQ ring.  Everything between
`attach_client` and `free_client` lives entirely on the client's fd;
when that fd closes (graceful or crash), the kernel's
`snvm_dev_release` cascade reclaims every queue / map for that fd.

---

## Repository layout

```
backends/local/NVMeService/
├── NVMeService.md                       # this file
├── src/
│   ├── nvmeservice.proto                # gRPC contract (single source of truth)
│   ├── nvmeservice_config.{h,cpp}       # YAML schema + validator
│   ├── nvmeservice_state.{h,cu}         # DeviceState + Allocation table + reaper
│   ├── nvmeservice_server.{h,cpp}       # gRPC service impl
│   └── nvmeservice_client.{h,cpp}       # client-side helper library (libnvmeservice_client)
└── examples/
    ├── CMakeLists.txt
    ├── nvmeservice_daemon.cpp           # daemon entry point
    ├── nvmeservice_client.cpp           # client entry point (gRPC half)
    ├── nvmeservice_client_io.cu         # client entry point (CUDA + libnvm half)
    └── nvmeservice_client_io.h          # bridge between the two TUs
```

The client example is split into `.cpp` + `.cu` because nvcc trips
over protobuf's C++17 inline-static enum traits if those headers go
through its frontend.  Keep that split if you copy the pattern into
your own integration.

Build targets (CMake):

| Target                          | What it is                          |
|---------------------------------|--------------------------------------|
| `nvmeservice`                   | shared library: proto + state + server + client lib |
| `nvmeservice_daemon_example`    | the daemon binary (`build/<preset>/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_daemon`) |
| `nvmeservice_client_example`    | the reference client (`build/<preset>/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client`) |

---

## Quick start

Prerequisites:

- snvme kernel module loaded (`/dev/snvm_control` exists).
- Target NVMe is **NOT** mounted as a regular filesystem and not in
  use by `nvme0n1` etc. – snvme will rebind it.
- A working CUDA toolchain + at least one GPU.
- `/mnt/gpu0`, `/mnt/nvme0` writable (daemon installs symlinks here;
  paths are configurable in `sys_config.yaml`).

Build:

```bash
cd /path/to/Tutti
cmake --preset cuda-module
cmake --build --preset cuda-module \
    --target nvmeservice_daemon_example nvmeservice_client_example --parallel 8
```

Edit `build/cuda-module/bin/sys_config.yaml` (copied from the repository root
during configuration) so
`nvmes[].pci_addr` matches your card and `nvmes[].allowed_gpus`
lists the GPUs you intend to test from.

Terminal A (daemon):

```bash
sudo ./build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_daemon \
    --config ./build/cuda-module/bin/sys_config.yaml
```

Expected lines (excerpt):

```
nvmeservice: device=0 pci=0000:08:00.0 snvme=/dev/ssnvme0 ns=1 qdepth=64
             max_user_qid=135 max_q_per_grp=16 allowed_gpus={0}
NVMeService daemon listening on 127.0.0.1:50051 (port 50051)
Registered devices:
  device_id=0 ... max_user_qid=135 max_q/grp=16
      allowed: cuda_device=0 mount=/mnt/gpu0/ssnvme0
lease: heartbeat=10s timeout=30s
queue_pool: default=4 max=16
```

Terminal B (client):

```bash
./build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client \
    --list-only                                  # enumerate
./build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client \
    --device 0 --cuda 0 --count 4
```

Expected client output (8-step IO smoke):

```
[ OK ] step=1   cudaSetDevice(0)
[ OK ] step=2   nvm_ctrl_attach_client /dev/ssnvme0 page=4096
[ OK ] step=3   nvm_create_group gid=1 max_queues=16 granted=4
[ OK ] step=4   mapped SQ/CQ + wbuf/rbuf
[ OK ] step=5   nvm_add_user_queue qid=33 sq_db=0x1108 cq_db=0x110c
[ OK ] step=6   Write+Read+verify x 4 IOs at LBA [2621440..2621443]
[ OK ] step=7   nvm_destroy_group gid=1 (rings cascade)
[ OK ] step=8   nvm_ctrl_free_client (no unbind, no chrdev_remove)
```

Expected `dmesg`:

```
snvme: NVM_SET_KERNEL_IOQ_CAP cap=32
snvme: capping kernel-side IOQ count from 135 to 32 (user pool gets [33..135])
snvme: user QID pool initialised: [33..135] (103 QIDs)
snvme: NVM_ADD_USER_QUEUE group=1 created 1 queue(s) (qids 33..33)
snvme: destroy_qgroup id=1 drained 1 user queue(s)
snvme: destroy_qgroup id=1 drained 2 map(s)
```

> ⚠️ `--device 0 --cuda 0` is **destructive**: writes 4 KiB blocks at
> LBA 2621440..2621443 (10 GiB offset).  Use `--skip-io` for any
> non-scratch device.  See *CLI* below.

---

## YAML schema

Authoritative source: `nvmeservice_config.{h,cpp}` + `sys_config.yaml`.

```yaml
grpc:
  endpoint: "127.0.0.1:50051"            # required

gpus:
  - { id: 0, mount_path: "/mnt/gpu0" }   # at least one entry

nvmes:
  - pci_addr: "0000:08:00.0"             # required
    mount_path: "/mnt/nvme0"             # required, unique across all nvmes[]
    namespace_id: 1                      # default 1
    kernel_ioq_cap: 32                   # NVM_SET_KERNEL_IOQ_CAP hint, optional
    allowed_gpus: [0]                    # optional ACL; empty/missing = all gpus[].id

queue_pool:
  default_per_client: 4                  # ConnectRequest.num_queues == 0 -> use this
  max_per_client: 16                     # daemon clamp; kernel still enforces 16/group

lease:
  heartbeat_interval_sec: 10
  timeout_sec: 30
```

Validator rules (`config.cpp::validate_config`):

1. `gpus[].id` unique, `gpus[].mount_path` non-empty.
2. `nvmes[].pci_addr` / `mount_path` non-empty; `mount_path`s unique.
3. Every entry in `nvmes[].allowed_gpus` must reference an existing
   `gpus[].id`.
4. `queue_pool.max_per_client >= default_per_client`.
5. `lease.timeout_sec > heartbeat_interval_sec`.

Anything outside that list (`total_queues`, `queue_depth`,
`queue_groups[].count`, the entire `queue_setup` block) is **ignored** —
those were the pre-B3 design's knobs and have no daemon-side meaning.

---

## CLI reference

### `nvmeservice_daemon`

```
nvmeservice_daemon --config <path-to-sys_config.yaml>
```

- Must run as root (chrdev_create + bind + symlink install).
- SIGINT/SIGTERM does a clean shutdown: gRPC shutdown → reaper join →
  per-device `nvm_ctrl_free` (which still cascades unbind +
  chrdev_remove because the daemon is the *owner* of every chrdev).
- **Never `kill -9` this process.** `chrdev_remove` only runs from the
  graceful SIGINT/SIGTERM path above -- it is a `nvm_ctrl_free()`
  ioctl the daemon must issue itself, not something the kernel replays
  automatically on fd-close (unlike per-client queue/DMA state, which
  *is* reclaimed for free on fd-close -- see the reaper's "kernel
  fd-close already reclaimed the actual queues / DATA maps" log
  line). A `SIGKILL`'d daemon leaves its `struct ctrl` for each BDF
  still registered kernel-side, so the next `SNVM_CHRDEV_CREATE` for
  that BDF can fail (`errno=14` on an un-rebuilt/un-reloaded module --
  see PORTING.md's troubleshooting table for the `snvm_chrdev_helper`
  idempotent-create fix).
  **Important correction**: this is a *userspace state* problem, not
  a kernel module refcount leak -- the Linux cdev framework auto-pairs
  `try_module_get`/`module_put` on every open()/fd-close of
  `/dev/snvm_control` and `/dev/ssnvme*` regardless of signal, and
  neither `SNVM_DEVICE_UNBIND` nor `SNVM_CHRDEV_REMOVE` touches the
  module refcount at all.  If `rmmod snvme` says "Module snvme is in
  use" after a SIGKILL'd daemon, do NOT assume it's this; check for a
  mounted `/dev/snvme*n*` block device or an open raw `/dev/snvme<N>`
  admin chardev first (`scripts/reset_snvme.sh` step 2b does this
  automatically) -- those are the two refs that actually outlive the
  daemon and that no owner-side ioctl ever releases.
  See `scripts/reset_snvme.sh`, which sends SIGTERM + waits before
  ever escalating to SIGKILL.  A `(deleted)` `/proc/<pid>/exe` is a
  harmless rebuild artifact, NOT a sign the daemon is stale -- confirm
  it is actually unresponsive (dead PID, no recent log activity)
  before touching it.

### `nvmeservice_client`

```
nvmeservice_client [--endpoint host:port] [--list-only]
                   [--device N] [--cuda N] [--count N] [--hold S]
                   [--skip-io]
```

| Flag         | Default     | Notes |
|--------------|-------------|-------|
| `--endpoint` | `127.0.0.1:50051` | Override `grpc.endpoint`. |
| `--list-only`| off         | Just `ListDevices` + exit; no Connect. |
| `--device`   | 0           | `device_id` from `ListDevices`. |
| `--cuda`     | first allowed | Must be in `allowed_gpus[]`. |
| `--count`    | 4           | `num_queues` requested. |
| `--hold`     | 5           | Seconds to keep the session up before Disconnect. |
| `--skip-io`  | off         | Skip steps 5–7 (no `add_user_queue`, no GPU IO). Use for non-scratch devices and quick connectivity tests. |

---

## Lifecycle (sequence + invariants)

```
Daemon boot
  └─ for each nvmes[]:
       nvm_controller_init_b3()          → /dev/ssnvmeN, ctrl held forever
       install /mnt/gpu<G>/ssnvmeN  →  /mnt/nvmeM/GPU<G>

Client Connect
  Server::Connect:
     ▸ ACL check (cuda_device ∈ allowed_gpus)
     ▸ clamp num_queues:  min(req or default,  max_per_client,
                              kernel max_queues_per_group)
     ▸ allocations_[uuid] = { device_id, cuda_device, granted, pid, starttime, ts }
     ▸ return metadata + symlink + lease params

Client (own process)
  cudaSetDevice(cuda_device)
  nvm_ctrl_attach_client(/dev/ssnvmeN)        # opens its OWN fd, no /dev/snvm_control
  nvm_create_group(&gid, &max_q)              # group is fd-scoped
  cudaMalloc + nvm_dma_map_ring_device(SQ)    # RING_SQ map → group
  cudaMalloc + nvm_dma_map_ring_device(CQ)    # RING_CQ map → group
  cudaMalloc + nvm_dma_map_data_device(buf)   # DATA map → fd (NOT group)
  nvm_add_user_queue() × N                    # uses kernel user QID pool
  ... GPU IO ...
  nvm_destroy_group(gid)                      # drains queues, RING maps cascade
                                              # DATA maps SURVIVE
  nvm_ctrl_free_client()                      # closes fd → kernel cascades
                                              # DATA maps now reclaimed
  Disconnect RPC                              # daemon erases allocations_[uuid]

Crash path  (SIGKILL, segfault, OOM, …)
  fd auto-close → kernel snvm_dev_release:
     ▸ destroy every fd-owned group  (RING maps included)
     ▸ release every fd-scoped DATA map
  ~heartbeat-timeout later:
     reaper sees PID dead (or starttime mismatch) → drop allocations_[uuid]
     "kernel fd-close already reclaimed the actual queues / DATA maps"
```

Hard invariants you can rely on:

- A client **never** holds `/dev/snvm_control` open; only the daemon
  does.  See `nvm_ctrl_attach_client` in `libnvm/src/linux/device.cpp`.
- DATA maps are fd-scoped, *not* group-scoped — they survive
  `destroy_group`.  Verified in `dmesg`: `destroy_qgroup ... drained
  1 user queue(s) ... drained 2 map(s)` is exactly **2** (SQ + CQ),
  never more.
- The daemon does not maintain a queue-count ledger.  After
  `granted_queues` is returned it is policy guidance only; the
  authoritative numbers come from `NVM_GET_DEV_INFO`
  (`max_user_qid` / `max_queues_per_group`) plus runtime add/destroy.
- Kernel hard cap per fd: `NVM_MAX_QUEUES_PER_GROUP = 16`
  (`backends/local/kernel_modules/snvme/...`).  The daemon also
  clamps `max_per_client`; the *effective* cap is `min(both)`.

---

## In-process API (libnvmeservice_client)

If you embed NVMeService into a larger application (Tutti, your
own filesystem, an inference runtime, …), link `nvmeservice` and use:

```cpp
#include "nvmeservice_client.h"

nvmeservice::NvmeServiceClient client("127.0.0.1:50051");

// Optional: enumerate.
auto devs = client.list_devices();
for (const auto& d : devs) { /* d.device_id, d.allowed_gpus, … */ }

// Open a session (sends Connect; spawns a heartbeat thread).
auto sess = client.connect(/*device_id=*/0,
                           /*cuda_device=*/0,
                           /*num_queues=*/4);
if (!sess) { /* see stderr */ return -1; }

// sess->{snvme_dev_path, bar0_size, granted_queues, …}
// drive libnvm yourself:
nvm_ctrl_t* ctrl = nullptr;
int rc = nvm_ctrl_attach_client(&ctrl,
                                sess->snvme_dev_path.c_str(),
                                (uint32_t)sess->bar0_size);

uint32_t group_id = 0, max_q = 0;
nvm_create_group(ctrl, &group_id, &max_q);

// ... IO ...

nvm_destroy_group(ctrl, group_id);
nvm_ctrl_free_client(ctrl);

sess.reset();   // Disconnect RPC + heartbeat thread joins
                // when this is the last live session.
```

`Session` is move-only and its destructor sends `Disconnect`.  Don't
bypass it.

---

## Extension points

A short field guide for common modifications.

### "Add a new RPC"

1. Add the message + `rpc Foo(...) returns (...)` to
   `src/nvmeservice.proto`.
2. Re-run cmake (proto is GLOB'd; `cmake .` triggers regen).
3. Implement `NvmeServiceImpl::Foo` in `src/nvmeservice_server.cpp`,
   declare in `src/nvmeservice_server.h`.
4. (Optional) add a wrapper to `src/nvmeservice_client.{h,cpp}` so
   in-process callers don't have to deal with raw stubs.

### "Add a new field to ListDevices / Connect"

`DeviceInfo` and `ConnectResponse` already pass through
`ServiceState::list_devices` / `Connect_inner` — see
`server.cpp::populate_device_info` and `Connect`.  Add the field in:

1. `nvmeservice.proto` (use field numbers > 32 for new optional fields
   to avoid colliding with reserved-block reservations).
2. `state.h::DeviceState` if it's per-device static info.
3. `state.cu::init_device` to populate it.
4. `server.cpp` to copy DeviceState → proto.
5. `client.h::ClientDeviceInfo` + `client.cpp` to surface to embedders.

### "Tighten the ACL"

Add fields to `nvmeservice_config.h::NvmeConfig` (e.g.
`required_capability`, `min_kernel_version`, …), parse in
`config.cpp::parse_*`, validate in `validate_config`, and
gate `Connect` in `server.cpp::Connect` next to the existing
`allowed_gpus` check.

### "Make the IO smoke heavier"

`examples/nvmeservice_client_io.cu` is intentionally minimal: 1
queue, 4 sequential IOs, single GPU thread.  To stress it:

- Bump queue count: build a vector of `qid` from
  `nvm_add_user_queue`, distribute IO across queues, poll multiple
  CQs.  `granted_queues` already reflects the policy ceiling.
- Increase IO size beyond 1 PRP: see how
  `snvme_smoke_libnvm_io.cu` builds `prp_list_4k` /
  `prp_list_16k` (under `backends/local/nvme/test/`); copy that
  pattern.
- Multi-process: launch N copies of `nvmeservice_client` in
  parallel; each gets its own `allocation_id` + own `gid`.  Watch
  `dmesg`: each `add_user_queue` should see a fresh qid out of the
  pool, and `destroy_qgroup` should always report the matching
  drain count.

### "Add throughput / latency telemetry on the daemon"

The daemon never touches the IO data path, so anything queue-level
has to come from the client.  Two options:

- Client-side: timestamp around `nvm_dma_map_ring_device` /
  `add_user_queue`, periodically push counters back through a new
  RPC (or just log to stderr).
- Kernel-side: extend the snvme module's existing
  `print_reset_stats` / `nvm_admin_*` machinery; daemon can poll
  via a new "GetStats" RPC that calls into libnvm's `print_*` on
  the owner-side fd.

---

## Failure-mode cheatsheet

| Symptom | Likely cause | Where to look |
|---------|--------------|---------------|
| daemon: `Failed to bind to '127.0.0.1:50051'` | another daemon still up, or stale port | `lsof -i :50051`, `pkill -f nvmeservice_daemon`. |
| daemon: `nvm_controller_init_b3 failed: ENOENT` | snvme kmod not loaded | `lsmod \| grep snvme`, `modprobe snvme`. |
| daemon: `Failed to bring up NVMe ... EBUSY` | `nvme0n1` still bound to the stock driver | `nvme reset` / unmount. dmesg shows the racy unbind. |
| client: `Connect rejected: cuda_device=N not in allowed_gpus` | YAML `allowed_gpus` doesn't list this GPU | edit `sys_config.yaml`, restart daemon. |
| client: `nvm_ctrl_attach_client … EACCES` | perms on `/dev/ssnvmeN` (default root-only) | run client as root or chmod the chrdev. |
| client: `nvm_create_group … ENOSPC` | another fd already used all 16 group slots on the controller | check `/proc/<pid>/fd` for stale daemon/clients. |
| client: `nvm_add_user_queue … EBUSY` | per-fd cap (16) already reached; or `granted_queues` was clamped to 0 | reduce `num_queues`, or check `max_per_client` and `max_queues_per_group`. |
| `cudaSetDevice -> initialization error` in a child after fork | classic CUDA-fork-without-exec | fork **before** any `cuda*` call; see `snvme_smoke_libnvm_role.cu` for the reference handshake-via-pipe pattern. |
| dmesg: `cascade-released N DATA map(s)` after client exit but daemon still alive | normal: B6 fd-scoped DATA reclaim path | nothing to fix, this is the invariant working. |
| daemon log: `nvmeservice reaper: dropped lease device=… (pid=…)` | client crashed or lost heartbeat for `lease.timeout_sec` | check the client; the queues/DATA were already reclaimed at fd close. |

---

## Verification recipes

### A. Connect twice in a row (basic regression)

```
./nvmeservice_client --device 0 --cuda 0 --count 4
./nvmeservice_client --device 0 --cuda 0 --count 8
```

In dmesg, both runs should print exactly one `add_user_queue
group=N` and one matching `destroy_qgroup id=N drained 1 user
queue(s) ... drained 2 map(s)`.  No leftover queues between
invocations.

### B. ACL rejection

```
./nvmeservice_client --device 0 --cuda 99 --count 4 --skip-io
```

Daemon prints `Connect rejected: cuda_device=99 not in
allowed_gpus for device_id=0`; client returns non-zero.  No kernel
activity at all.

### C. Crash → reaper

```
./nvmeservice_client --device 0 --cuda 0 --count 4 --hold 600 --skip-io &
PID=$!
sleep 3
kill -9 $PID
```

Immediately in dmesg: `snvm_dev_release: cascade-released N DATA
map(s)`.  ~30 s later the daemon prints `reaper: dropped lease
device=0 cuda_device=0 (pid=…)`.  No kernel commands run from the
reaper — that's the point.

### D. Multi-client concurrency

Run N (≤ floor(99 / queues_per_client)) clients in parallel:

```
for i in 1 2 3 4; do
  ./nvmeservice_client --device 0 --cuda 0 --count 2 --skip-io &
done
wait
```

Each session gets distinct `qid` ranges drawn from the user QID pool
(`[33..135]` with `kernel_ioq_cap=32`).  Daemon's
`allocations_` table briefly contains all four entries, drains as
each Disconnects.

### E. Daemon survives a client storm

Tight loop:

```
for i in $(seq 1 200); do
  ./nvmeservice_client --device 0 --cuda 0 --count 4 --skip-io
done
```

`/dev/ssnvme0` stays alive, daemon log shows 200 Connect/Disconnect
pairs, no kernel `unbind` between iterations.

---

## Known limits / open follow-ups

These are tracked in repo-root `Todolist.md`; reproduced here for
context:

- `disk.ns_id` is not populated by `NVM_GET_DEV_INFO` — daemon
  forwards whatever YAML says.  When the kernel ioctl is extended to
  ship the namespace id, drop the YAML field.
- The daemon parses the chrdev minor out of `disk.disk_name` (e.g.
  `"snvme0n1"` → `0`) because the libnvm bring-up doesn't return it
  explicitly.  Cleaner once `nvm_controller_init_b3` exposes the
  minor directly.
- No telemetry RPC yet (see *Extension points / telemetry* above for
  the suggested shape).
- Trust model is cooperative: anyone who can `connect()` can issue
  IO at the LBA level.  Production deployments should use unix
  domain socket + uid check at minimum.

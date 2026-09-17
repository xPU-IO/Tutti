# Layer 2 (device_manager) Hardware Integration Test Report

**Date**: 2026-07-21
**Task**: Exercise the refactored Layer 2 runtime paths that the mock-only unit
test never covered, against a real NVMe device.

---

## Executive Summary

✅ **NVMeService daemon + client integration**: PASSED on real hardware
✅ **Build wiring fix**: `TUTTI_YAML_CPP_TARGET` added to layered `tutti/CMakeLists.txt`
✅ **Allocator unit test**: 8/8
✅ **Device restored** to stock `nvme` after testing

Test device: `0000:b1:00.0` (nvme1), scratch NVMe — Intel PCIe Data Center SSD.
GPU: NVIDIA L40S (cuda_device 0). Kernel: Linux 5.15.0-185.

---

## 1. Build wiring gap found and fixed

The Layer 2 refactor moved `nvmeservice` under
`device_manager/nvme/nvmeservice/`, but the layered build root
`tutti/CMakeLists.txt` never defined `TUTTI_YAML_CPP_TARGET` — that variable
existed only in the old repo-root `CMakeLists.txt`. Result: `libnvmeservice.a`
compiled, but the daemon/client examples failed to link:

```
undefined reference to `vtable for YAML::RepresentationException'
undefined reference to `vtable for YAML::BadSubscript'
...
```

**Fix**: added `find_package(yaml-cpp REQUIRED)` plus target-name
normalization to `tutti/CMakeLists.txt`, right after `find_package(Threads)`.
gRPC/protobuf resolve automatically via the vcpkg toolchain; only yaml-cpp
needed the explicit variable. After the fix both examples link and run.

---

## 2. Test procedure

1. Built `nvmeservice_daemon` + `nvmeservice_client` examples (`build_layer2`).
2. Loaded the snvme module (built with a build-dir-only
   `blk_set_queue_dying` → `blk_mark_disk_dead` 5.15.0-185 compat shim; repo
   source stays == main).
3. Ran the daemon with a config pointing at `0000:b1:00.0`
   (`kernel_ioq_cap: 16`, `allowed_gpus: [0]`).
4. Ran the client for the full Connect → attach → IO → teardown path, plus
   policy and reaper variants.
5. Tore down: daemon SIGTERM, `rmmod snvme snvme_core`, rebind device to
   stock `nvme`.

---

## 3. Results

### Daemon owner bring-up
`nvm_controller_init_owner` succeeded on `0000:b1:00.0`: created `/dev/ssnvme0`,
probed the controller, installed the GPU-view symlink.

```
device=0 pci=0000:b1:00.0 snvme=/dev/ssnvme0 ns=1 page=4096 blk=4096
qdepth=1024 dstrd=1 bar0=16384 max_user_qid=31 max_q/grp=16
allowed: cuda_device=0 mount=/mnt/gpu0/ssnvme0
```

### Client full IO path (Connect → attach → GPU IO → teardown)

| Step | Operation | Result |
|------|-----------|--------|
| 1 | `cudaSetDevice(0)` | ✅ |
| 2 | `nvm_ctrl_attach_client(/dev/ssnvme0)` page=4096 | ✅ |
| 3 | `nvm_create_group` gid=1 max_queues=16 granted=4 | ✅ |
| 4 | map SQ/CQ + wbuf/rbuf to GPU memory | ✅ |
| 5 | `nvm_add_user_queue` qid=17 (sq_db=0x1088 cq_db=0x108c) | ✅ |
| 6 | **Write+Read+verify x 4 IOs** at LBA [2621440..2621443] | ✅ |
| 7 | `nvm_destroy_group` gid=1 (rings cascade) | ✅ |
| 8 | `nvm_ctrl_free_client` (no unbind, no chrdev_remove) | ✅ |

Step 6 is real GPU-memory → NVMe write, read-back, and byte-compare.

### Daemon policy paths

| Case | Expected | Result |
|------|----------|--------|
| `--list-only` | list device, no Connect | ✅ |
| `--skip-io` (count 8) | attach/create/destroy, no IO | ✅ granted=8 |
| over-cap `--count 99` | clamp to kernel `max_queues_per_group`=16 | ✅ granted=16 |

### Lease reaper (dead-client reclamation)
Client took a lease then received SIGKILL. The daemon reaper detected the dead
PID (`kill(pid,0)` + `/proc/<pid>/stat` starttime) and dropped the lease:

```
nvmeservice reaper: dropped lease device=0 cuda_device=0 (pid=..., allocation_id=...);
kernel fd-close already reclaimed the actual queues / DATA maps
```

### Allocator unit test
`layer2_smoke_test` (mock-based `LocalNvmeVirtualRegistry`): **8/8 passed**.

### Teardown
Daemon exited cleanly (chrdev removed, device unbound); module unloaded;
`0000:b1:00.0` rebound to stock `nvme` as `nvme1`; `/dev/nvme1n1` live.

---

## 4. Remaining untested surface

- **`LocalNvmeVirtualRegistry`** (Level-2 in-process QP slicing) is still only
  covered by the mock unit test. No backend above Layer 2 exists yet to drive
  it against a real `NvmeQueueGroup`; hardware exercise is blocked until
  Layer 3 lands.
- Only **single-GPU / single-NVMe** topology was run (this host's affinity plus
  the one scratch device). Multi-device ACL enforcement is covered by config
  validation but not a live multi-NVMe run.

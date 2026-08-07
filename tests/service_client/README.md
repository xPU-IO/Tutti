# NVMeService client-mode attach smoke

This harness validates the historical `NVMeService` client-mode control-plane
attach lifecycle for GPU 0-3 and NVMe device_id 0-3. It does **not** execute
block read/write.

## NVMe device lifecycle (important)

The four target NVMe PCI devices (`0000:08/4b/57/63:00.0`) are managed
exclusively by Tutti. Their **steady state is UNBOUND** — they are not bound
to the kernel `nvme` driver when the daemon is not running. This is **normal**,
not a fault.

Block devices (`/dev/nvmeXn1` or `/dev/snvmeXn1`) only exist while the daemon
is running. The daemon's B3 process (chrdev create → PCI bind → probe) creates
them at startup and removes them on exit. Therefore:

- **Daemon not running → no block device → expected.**
- **Daemon running → block device exists → can be used for attach.**

The previous harness incorrectly treated missing block devices as a dry-run
error. This was a structural design flaw: dry-run runs *when the daemon is not
running*, so block devices are always absent. T-013 fixes this.

## T-013 fix

The dry-run block-device gate was changed from "must exist" to conditional:

- Block device **absent** → INFO (expected when daemon is not running).
- Block device **present** → must still be unmounted with empty holders;
  violation is a real error (BLOCKED).

Additionally, the dry-run now reports PCI driver binding status
(informational) and the generator is called with `python3 -B` to suppress
bytecode cache writes.

## T-003 corrections

Two errors from the T-003 harness are fixed:

1. **`mount_path` is not a filesystem mount target.** The raw NVMe
   `/dev/nvmeXn1` must **not** be mounted. `nvmes[].mount_path` is a work
   directory where the daemon creates `GPU<n>` sub-directories and GPU-view
   symlinks. The generator creates ordinary work directories under
   `.work/logs/<timestamp>/mount_work/` and verifies each block device is
   unmounted with empty holders.

2. **Client binary name corrected.** `nvmeservice_client_example` is the CMake
   target name; the real executable is `nvmeservice_client` (OUTPUT_NAME).
   The default `CLIENT_BIN` is now
   `build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client`
   (or set `CLIENT_BIN` to a different build preset's output).

## Dry-run vs execute

### Dry-run (default, no arguments)

Validates prerequisites that should hold when the daemon is **not** running:

- Four GPUs queryable via `nvidia-smi -i <id>`.
- Four PCI BDF sysfs paths exist; driver binding status reported (UNBOUND is
  expected and not an error).
- Four block devices: absent = INFO (expected); present = must be unmounted
  with empty holders.
- Daemon and client binaries executable.
- `127.0.0.1:50051` not occupied.
- `sudo -n true` works.
- Prints the planned commands.

**Zero side effects**: no daemon started, no `.work/` directory created, no
YAML generated, no PCI bind/unbind, no `/sys` writes.

Exit codes:
- `0` — all prerequisites met, prints `Dry-run result: PASS`.
- `1` — real error (binary missing, endpoint occupied, GPU unqueryable,
  mounted raw NVMe, etc.), prints `Dry-run result: BLOCKED`.

### Execute (`--execute`)

Generates config, starts daemon, runs `ListDevices` + four `--skip-io`
attach groups, SIGTERMs daemon. Before starting any client, it also asserts
that the daemon PID is absent from `nvidia-smi`'s compute-process list; this
guards the owner-only bring-up path against accidental CUDA initialization.
Creates a timestamped log directory under `.work/logs/`.

Exit codes:
- `0` — all attach groups and daemon clean exit passed.
- `1` — preflight failure or attach/daemon failure.

## `.work/` directory

`.work/` contains per-run artifacts (logs, configs, mount work directories).
It is listed in `tests/service_client/.gitignore` and can be safely deleted
at any time. `__pycache__/` is also ignored.

## Usage

```bash
cd /data/home/ryeqiu/Tutti/tests/service_client
./run_attach_smoke.sh             # dry-run, no side effects
./run_attach_smoke.sh --execute   # run the attach smoke
```

Without `--execute`, the script performs read-only prerequisite checks and
prints the planned commands. It does not start the daemon, create `.work/`,
or generate YAML.

With `--execute`, it creates a timestamped log directory, generates the config,
starts the daemon, runs `ListDevices` + four `--skip-io` attach groups, then
SIGTERMs the daemon.

## Overrides

```bash
DAEMON_BIN=/path/to/tutti_daemon \
CLIENT_BIN=/path/to/nvmeservice_client \
SUDO="sudo -n" \
ENDPOINT=127.0.0.1:50051 \
./run_attach_smoke.sh --execute
```

## Config generator

```bash
python3 generate_attach_config.py --output /path/to/attach_config.yaml
```

The generator:

- Checks four PCI sysfs paths exist;
- Checks four `/dev/nvmeXn1` exist;
- Verifies each `/dev/nvmeXn1` has **no** mount target (raw NVMe must not be mounted);
- Verifies `/sys/class/block/nvmeXn1/holders` is empty;
- Creates work directories under `<run_dir>/mount_work/`;
- Emits YAML with `heartbeat_interval_sec: 1`, `timeout_sec: 5`.

Uses only Python stdlib (no PyYAML).

## What `--skip-io` verifies

The client's `--skip-io` path executes:

```
cudaSetDevice
nvm_ctrl_attach_client
nvm_create_group
nvm_destroy_group
nvm_ctrl_free_client
```

It does **not** create user queue pairs, map SQ/CQ or data buffers, or execute
NVMe read/write. Results can only claim "non-IO attach/group lifecycle passed",
not "IO path verified".

## Heartbeat

Each attach client uses `--hold 2` with `heartbeat_interval_sec: 1`. The
harness treats any `lease revoked by daemon for allocation` notice as heartbeat
failure.

## Safety constraints

- All attach client commands use `--skip-io`; no block read/write.
- No mount/umount/format/partition operations.
- No `insmod`/`rmmod`/`modprobe` or kernel module modification.
- No writes to `/dev/nvme*n1`.
- No modifications to `/etc`, `/sys`, `/dev`.
- An occupied `127.0.0.1:50051` is reported as BLOCKED; existing daemons are
  not killed.
- No interactive sudo; `sudo -n` is required.
- Daemon is stopped with SIGTERM (20s wait); SIGKILL is last resort only.
- Daemon owner bring-up must not create a CUDA context; the execute path fails
  if the daemon appears in NVIDIA's compute-process list.

## Logs

Each `--execute` run creates:

```
tests/service_client/.work/logs/<YYYYmmdd-HHMMSS>/
├── attach_config.yaml
├── daemon.log
├── client_list.log
├── client_device_0_gpu_0.log
├── client_device_1_gpu_1.log
├── client_device_2_gpu_2.log
├── client_device_3_gpu_3.log
├── harness.log
└── mount_work/          # work directories (not filesystem mounts)
    ├── gpus/gpu0..3/
    └── nvmes/nvme0..3/
```

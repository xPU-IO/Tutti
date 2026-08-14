# TuttiRuntime hardware I/O example

This example uses a Tutti YAML as the application entry point. It creates a
`TuttiRuntime`, acquires local NVMe resources from `nvmeservice`, creates
scratch backing files on daemon-published accelerator views, and drives a
write and read through Tutti's local NVMe data path. The final result is a
byte-for-byte comparison in host memory.

Single-Runtime configurations:

- `tutti_local_nvme.yaml` acquires device 0 and runs the single-device local NVMe path.
- `tutti_striped.yaml` acquires devices 0 and 1 and runs the two-device
  striped local NVMe path with a 64 KiB stripe unit.

Both configurations request 32 queues per controller and use a 32-thread CUDA
submit block. The daemon and loaded `snvme` module must therefore grant at
least 32 queues per group. A smaller grant is rejected during Runtime creation
to prevent multiple GPU threads from concurrently using the same queue.

The multi-accelerator example consumes two additional Tutti configurations:

- `tutti_multi_accelerator_0.yaml` creates a Runtime on accelerator 0.
- `tutti_multi_accelerator_1.yaml` creates a Runtime on accelerator 1.

Both configurations explicitly acquire daemon device 0. Each allocation gets
its own 16-queue group and accelerator-specific filesystem view, so the two
Runtime instances can drive the same NVMe controller concurrently. These two
configs deliberately use 16 queues and a 16-thread submit block so two Runtime
instances can share 32 queues on the same controller. The single-Runtime
configs use the full 32-queue group supported by the current `snvme` UAPI.

The cross-mapping configurations exercise both daemon devices with the
accelerator-to-device association reversed:

- `tutti_cross_accelerator_0_device_1.yaml`: accelerator 0 uses device 1.
- `tutti_cross_accelerator_1_device_0.yaml`: accelerator 1 uses device 0.

The dual-Runtime striped configurations let both accelerators use both daemon
devices concurrently:

- `tutti_multi_accelerator_striped_0.yaml`: accelerator 0 stripes over `[0, 1]`.
- `tutti_multi_accelerator_striped_1.yaml`: accelerator 1 stripes over `[0, 1]`.

The scratch file is deleted after the run. Pass `--keep-file` to retain it.

## Build

From the repository root:

```bash
cmake --preset cuda-module
cmake --build --preset cuda-module --parallel 8 \
  --target tutti_daemon tutti_runtime_example \
  tutti_runtime_multi_accelerator_example
```

## Start the daemon

The host-local configuration used on this machine owns devices 0 and 1. The
example YAML explicitly acquires device 0 for accelerator 0.

```bash
sudo  env TUTTI_VERBOSE=1 \
  build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon \
  --config config/local/daemon_2disk.yaml 
```

Wait for the `tutti_daemon listening` line. Query the daemon and record the
accelerator `view_root` and owner-returned `chrdev` basename rather than
deriving either value from `device_id`:

```bash
build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client \
  --endpoint 127.0.0.1:50051 --list-only
```

The daemon publishes each usable directory as
`<view_root>/<chrdev-basename>`. With the local configuration and device 0 on
this machine, the returned values are `/mnt/snvme/gpu0` and `/dev/ssnvme0`, so
the published directory is `/mnt/snvme/gpu0/ssnvme0`.

## Run single-device I/O

Pass the actual accelerator view directory published for device 0. For
example, if the daemon published `/mnt/snvme/gpu0/ssnvme0`:

```bash
sudo  build/cuda-module/bin/tutti_runtime_example \
  --config examples/tutti_runtime/tutti_local_nvme.yaml \
  --directory /mnt/snvme/gpu0/ssnvme0 
```

`sudo` is needed with the daemon's default root-owned accelerator view
directories. It is unnecessary when the selected view is writable by the
calling user.

When the daemon grants all 32 requested queues, the expected final line is:

```text
TuttiRuntime local NVMe hardware I/O: PASS
```

## Run two-device striped I/O

The striped resolver expects one daemon-published view per acquired device.
Pass both directories in the same order as `device_ids` in
`tutti_striped.yaml`:

```bash
sudo  build/cuda-module/bin/tutti_runtime_example \
  --config examples/tutti_runtime/tutti_striped.yaml \
  --directory /mnt/snvme/gpu0/ssnvme0 \
  --directory /mnt/snvme/gpu0/ssnvme1 
```

The example creates one shard below each view's `striped/` directory. The
default 4 MiB request crosses 64 stripe boundaries, exercises both devices,
and is deleted from both views after verification. When both controllers grant
32 queues, the expected final line is:

```text
TuttiRuntime striped local NVMe hardware I/O: PASS
```

`--size` changes the transfer size and must be a multiple of 4096 bytes.
`--config` selects another Tutti application YAML. `--stripe-unit` must match
the striped backend's `config.stripe_unit`; it defaults to 65536 bytes.

## Run two Runtime instances on two accelerators

This command creates two Runtime instances in parallel. Each Runtime owns one
worker thread, one CUDA device buffer, one CUDA stream, and one independent
daemon allocation. Both YAML files select device 0, while their
`runtime.accel_id` values select accelerators 0 and 1. Pass the device-0 view
published below each accelerator root:

```bash
sudo  build/cuda-module/bin/tutti_runtime_multi_accelerator_example \
  --config-0 examples/tutti_runtime/tutti_multi_accelerator_0.yaml \
  --directory-0 /mnt/snvme/gpu0/ssnvme0 \
  --config-1 examples/tutti_runtime/tutti_multi_accelerator_1.yaml \
  --directory-1 /mnt/snvme/gpu1/ssnvme0 
```

The two YAML paths are compiled-in defaults, so `--config-0` and `--config-1`
may be omitted when running the repository build. `--size` changes each
Runtime's transfer size, must be a multiple of 4096 bytes, and must not exceed
the DataPath's reported single-I/O limit. `--keep-files` retains both scratch
files.

Each worker writes a different deterministic pattern, clears its GPU buffer,
reads the data back through local NVMe, and compares every byte in host memory.
The expected final lines are:

```text
[runtime 0] hardware write/read verification: PASS (... ms)
[runtime 1] hardware write/read verification: PASS (... ms)
Two TuttiRuntime instances: accelerator 0 -> NVMe devices [0], accelerator 1 -> NVMe devices [0]: PASS
```

## Run the cross-mapping validation

The same executable also accepts two configs that select different NVMe
devices. This command reverses the ordinal mapping: accelerator 0 drives
device 1 while accelerator 1 drives device 0. The directory paired with each
config must use that accelerator's view and the selected device's `chrdev`
basename:

```bash
sudo  build/cuda-module/bin/tutti_runtime_multi_accelerator_example \
  --config-0 examples/tutti_runtime/tutti_cross_accelerator_0_device_1.yaml \
  --directory-0 /mnt/snvme/gpu0/ssnvme1 \
  --config-1 examples/tutti_runtime/tutti_cross_accelerator_1_device_0.yaml \
  --directory-1 /mnt/snvme/gpu1/ssnvme0 
```

The expected summary is:

```text
Two TuttiRuntime instances: accelerator 0 -> NVMe devices [1], accelerator 1 -> NVMe devices [0]: PASS
```

## Run two striped Runtime instances

Both Runtime instances can independently stripe over both NVMe devices. Repeat
the worker's directory option in the same order as that YAML's `device_ids`.
The two workers use different logical filenames, so each Runtime creates its
own shard on each disk:

```bash
sudo  build/cuda-module/bin/tutti_runtime_multi_accelerator_example \
  --config-0 examples/tutti_runtime/tutti_multi_accelerator_striped_0.yaml \
  --directory-0 /mnt/snvme/gpu0/ssnvme0 \
  --directory-0 /mnt/snvme/gpu0/ssnvme1 \
  --config-1 examples/tutti_runtime/tutti_multi_accelerator_striped_1.yaml \
  --directory-1 /mnt/snvme/gpu1/ssnvme0 \
  --directory-1 /mnt/snvme/gpu1/ssnvme1 
```

With the default 4 MiB request and 64 KiB stripe unit, each logical file
crosses 64 stripe boundaries. The example prints the four backing shard paths
before I/O and verifies a different deterministic pattern for each Runtime.
The expected summary is:

```text
Two TuttiRuntime instances: accelerator 0 -> NVMe devices [0,1], accelerator 1 -> NVMe devices [0,1]: PASS
```

Stop the daemon with one `SIGINT` or `SIGTERM` so it can remove accelerator
views, unmount the filesystems, and release both controllers cleanly.

#!/usr/bin/env python3
"""Generate a validated, non-destructive NVMeService attach config.

This generator fixes two errors from the T-003 harness:

1. ``backing_mount_path`` is the daemon-owned filesystem mount and
   ``view_root`` is the accelerator-visible symlink root. The generated YAML
   uses the canonical explicit resource schema.

2. The real client binary is ``nvmeservice_client`` (OUTPUT_NAME), not the
   CMake target name ``nvmeservice_client_example``.  The generator itself
   does not resolve the binary, but the YAML it produces is consumed by the
   harness which uses the correct path.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Fixed hardware mapping (device_id, pci_bdf, allowed_gpu).
# Block device name is auto-detected: snvme driver creates "snvmeXn1" after
# re-bind, while initial module load may create "nvmeXn1". We try both.
DEVICES = (
    (0, "0000:08:00.0", 0),
    (1, "0000:4b:00.0", 1),
    (2, "0000:57:00.0", 2),
    (3, "0000:63:00.0", 3),
)
ENDPOINT = "127.0.0.1:50051"


def resolve_block_device(device_id: str) -> str:
    """Return the first existing block device path for the given device_id."""
    for name in ("nvme{}n1", "snvme{}n1"):
        path = "/dev/" + name.format(device_id)
        if Path(path).exists():
            return path
    return "/dev/nvme{}n1".format(device_id)  # default for error message


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_not_mounted(block_device: str) -> None:
    """Refuse if the raw NVMe block device has a filesystem mount target."""
    try:
        result = subprocess.run(
            ["findmnt", "-n", "-o", "TARGET", "--source", block_device],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"findmnt is not available: {exc}")

    targets = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if targets:
        raise SystemExit(
            f"{block_device} is mounted at {targets[0]}; "
            "raw NVMe must NOT be mounted for attach smoke"
        )


def check_holders_empty(block_device: str) -> None:
    """Refuse if the block device has open holders (dm, md, etc.)."""
    dev_name = os.path.basename(block_device)
    holders_dir = Path("/sys/class/block") / dev_name / "holders"
    if not holders_dir.is_dir():
        # Some kernels may not expose holders; treat as empty.
        return
    holders = [h.name for h in holders_dir.iterdir()] if holders_dir.exists() else []
    if holders:
        raise SystemExit(
            f"{block_device} has non-empty holders: {', '.join(holders)}"
        )


def quote_yaml(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a four-device NVMeService attach config."
    )
    parser.add_argument("--output", required=True, help="output YAML path")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    run_dir = output.parent

    if not run_dir.exists() or not run_dir.is_dir():
        fail(f"output directory does not exist: {run_dir}")

    # Validate hardware, check unmounted + empty holders.
    errors = []
    for device_id, bdf, _gpu_id in DEVICES:
        sysfs_path = Path("/sys/bus/pci/devices") / bdf
        if not sysfs_path.exists():
            errors.append(f"device={device_id}: PCI device missing: {sysfs_path}")
        block_device = resolve_block_device(device_id)
        if not Path(block_device).exists():
            # Block devices may not exist yet — the snvme driver creates them
            # only while the daemon is running (B3 bind+probe). Skip mount and
            # holder checks in that case; the daemon will create them safely.
            print(f"WARN: device={device_id}: block device {block_device} does not "
                  "exist yet (daemon will create via B3)", file=sys.stderr)
            continue
        try:
            check_not_mounted(block_device)
        except SystemExit as exc:
            errors.append(str(exc).replace("ERROR: ", ""))
            continue
        try:
            check_holders_empty(block_device)
        except SystemExit as exc:
            errors.append(str(exc).replace("ERROR: ", ""))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    # Create work directories (ordinary dirs, NOT filesystem mounts).
    gpus_work = run_dir / "mount_work" / "gpus"
    nvmes_work = run_dir / "mount_work" / "nvmes"
    for gpu_id in range(4):
        (gpus_work / f"gpu{gpu_id}").mkdir(parents=True, exist_ok=True)
    for device_id, _bdf, _gpu in DEVICES:
        (nvmes_work / f"nvme{device_id}").mkdir(parents=True, exist_ok=True)

    # Generate YAML using stdlib only (no PyYAML).
    lines = [
        "grpc:",
        f"  endpoint: {quote_yaml(ENDPOINT)}",
        "",
        "accelerators:",
    ]
    for gpu_id in range(4):
        gpu_mount = str(gpus_work / f"gpu{gpu_id}")
        lines.extend([
            f"  - accel_id: {gpu_id}",
            f"    view_root: {quote_yaml(gpu_mount)}",
        ])

    lines.extend(["", "nvmes:"])
    for device_id, bdf, gpu_id in DEVICES:
        nvme_mount = str(nvmes_work / f"nvme{device_id}")
        lines.extend([
            f"  - device_id: {device_id}",
            f"    pci_addr: {quote_yaml(bdf)}",
            f"    backing_mount_path: {quote_yaml(nvme_mount)}",
            "    namespace_id: 1",
            "    kernel_ioq_cap: 32",
            f"    allowed_accel_ids: [{gpu_id}]",
            "    auto_mount: true",
        ])

    lines.extend([
        "",
        "queue_pool:",
        "  default_per_client: 2",
        "  max_per_client: 4",
        "",
        "lease:",
        "  heartbeat_interval_sec: 1",
        "  timeout_sec: 5",
        "",
    ])

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

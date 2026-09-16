"""preset 设备字段推导：daemon 配置是硬件事实的单一来源。

daemon YAML 的 nvmes[] 按 device_id 给出 pci_addr / backing_mount_path /
namespace_id；preset 只写 device_id（local 为单数 device，striped 为
devices 列表），由此补全为运行时组装器要求显式给出的设备字段。

两种布局共用同一推导规则，避免 store 与 metadata 两处漂移；显式给出的
字段优先（preset 可覆盖 daemon），device_id 仅作查询键、不进运行时。
"""

from __future__ import annotations

from pathlib import Path


def derive_device_fields(preset: dict, yaml) -> dict:
    """daemon_config + device_id(s) → 补全设备字段的 preset 副本。

    无 daemon_config → 原样返回。striped 按 devices 列表逐个推导；
    local 按顶层 device_id + device 推导。函数幂等：已补全的设备条目
    再次调用不改变。
    """
    daemon_path = preset.get("daemon_config")
    if not daemon_path:
        return preset
    daemon = yaml.safe_load(Path(daemon_path).read_text())
    nvmes = {
        entry.get("device_id"): entry
        for entry in daemon.get("nvmes", [])
    }
    if preset.get("type") == "striped":
        derived = dict(preset)
        derived["devices"] = [
            _derive_one(dict(device), nvmes)
            for device in preset.get("devices") or []
        ]
        return derived
    device_id = preset.get("device_id")
    if device_id is None:
        raise RuntimeError("preset 携带 daemon_config 时必须同时给出 device_id")
    derived = dict(preset)
    derived["device"] = _derive_one(
        dict(preset.get("device") or {}, device_id=device_id), nvmes
    )
    return derived


def _derive_one(device: dict, nvmes: dict) -> dict:
    """单个设备条目：device_id → pci_bdf / mount_path / namespace / backing。"""
    device_id = device.pop("device_id", None)
    if device_id is None:
        return device  # 字段已显式给出（无 daemon 查询键）
    entry = nvmes.get(device_id)
    if entry is None:
        raise RuntimeError(f"daemon 配置无 device_id={device_id} 的 NVMe 条目")
    namespace_id = device.get("namespace_id", entry.get("namespace_id", 1))
    device.setdefault("pci_bdf", entry["pci_addr"])
    device.setdefault("mount_path", entry["backing_mount_path"])
    device.setdefault("namespace_id", namespace_id)
    device.setdefault("backing_device", f"/dev/snvme{device_id}n{namespace_id}")
    return device

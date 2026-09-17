"""真机 runtime 装配：preset 归一 → daemon 事实推导 → 绑定调用。

从 stores/tutti_nvme/store.py 搬出（评审意见：存储抽象与硬件装配混装）。
本模块只做"把配置变成 runtime 句柄"，不涉及任何 KV 语义；store.py 仅
保留存储抽象与数据面。函数体自原处逐字搬迁，行为不变。
"""

from __future__ import annotations

import os
from pathlib import Path

from tutti.storage.tutti_nvme.preset_derive import derive_device_fields


def normalize_preset(preset) -> dict:
    """递归归一 preset：字符串值恰为纯十进制整数时转 int。"""
    if isinstance(preset, dict):
        return {k: normalize_preset(v) for k, v in preset.items()}
    if isinstance(preset, list):
        return [normalize_preset(v) for v in preset]
    if isinstance(preset, str) and preset.strip().isdigit():
        return int(preset)
    return preset


def preset_mounts(preset):
    """Derive striped layout mounts from a striped preset when available."""
    if not isinstance(preset, dict):
        return None
    devices = preset.get("devices")
    if not isinstance(devices, (list, tuple)):
        return None
    mounts = []
    for device in devices:
        if not isinstance(device, dict) or not device.get("mount_path"):
            return None
        mounts.append(device["mount_path"])
    return mounts or None


def build_runtime(preset: dict):
    """按 preset dict 构造真机 runtime（daemon_config 推导与归一同环境变量路径）。"""
    import yaml

    if not isinstance(preset, dict):
        raise RuntimeError("preset 必须是映射")
    if "daemon_config" in preset:
        preset = derive_device_fields(preset, yaml)

    try:
        import tutti_runtime  # bindings 构建产物（需在 sys.path/PYTHONPATH）
    except ImportError as exc:
        raise RuntimeError(
            "tutti_runtime 绑定不可用：先构建 csrc/python/"
            "bindings/python 并将其加入 PYTHONPATH"
        ) from exc

    preset = dict(preset)
    preset_type = preset.pop("type", "local")
    preset.pop("daemon_config", None)  # 推导元键不进 runtime preset
    preset.pop("device_id", None)
    if preset_type == "striped":
        return tutti_runtime.make_striped_nvme_runtime(preset)
    if preset_type == "local":
        return tutti_runtime.make_local_nvme_runtime(preset)
    raise RuntimeError(f"未知 preset type：{preset_type}")


def build_runtime_from_env():
    """按 TUTTI_NVME_PRESET 构造真机 runtime（本包私有推导）。"""
    import yaml

    raw = os.environ.get("TUTTI_NVME_PRESET", "").strip()
    if not raw:
        raise RuntimeError(
            "runtime=None 需要 TUTTI_NVME_PRESET（yaml/json 内联或文件路径）"
        )
    text = Path(raw).read_text() if os.path.isfile(raw) else raw
    preset = yaml.safe_load(text)
    if not isinstance(preset, dict):
        raise RuntimeError("TUTTI_NVME_PRESET 解析结果必须是映射")
    return build_runtime(normalize_preset(preset))

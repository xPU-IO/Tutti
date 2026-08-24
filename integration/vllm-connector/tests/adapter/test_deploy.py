"""多副本占位符与 preset 直传的部署机制测试。"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from adapter.connector import TuttiConnectorV1, _expand_placeholders
from stores.tutti_nvme.store import _normalize_preset
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)


class TestLocalRankPlaceholder:
    """{LOCAL_RANK} 占位符：递归替换、缺省 0、非字符串值不动。"""

    def test_replace_in_nested_strings(self):
        spec = {
            "root": "/mnt/nvme{LOCAL_RANK}/pool",
            "num_chunks": 128,
            "preset": {
                "device_id": "{LOCAL_RANK}",
                "gpu_id": "{LOCAL_RANK}",
                "ssnvme_path": "/dev/ssnvme{LOCAL_RANK}",
            },
        }
        os.environ["LOCAL_RANK"] = "3"
        try:
            got = _expand_placeholders(spec)
        finally:
            del os.environ["LOCAL_RANK"]
        assert got["root"] == "/mnt/nvme3/pool"
        assert got["num_chunks"] == 128  # 数值不动
        assert got["preset"]["device_id"] == "3"
        assert got["preset"]["ssnvme_path"] == "/dev/ssnvme3"

    def test_defaults_to_zero(self):
        os.environ.pop("LOCAL_RANK", None)
        assert _expand_placeholders("/mnt/nvme{LOCAL_RANK}") == "/mnt/nvme0"

    def test_vllm_config_rank_takes_priority(self):
        """vllm_config.parallel_config.rank 优先于环境变量（V1 worker 无 env）。"""
        cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(rank=3),
        )
        os.environ["LOCAL_RANK"] = "1"
        try:
            assert _expand_placeholders("/mnt/nvme{LOCAL_RANK}", cfg) == "/mnt/nvme3"
        finally:
            del os.environ["LOCAL_RANK"]

    def test_env_used_when_config_absent(self):
        os.environ["LOCAL_RANK"] = "2"
        try:
            assert _expand_placeholders("/mnt/nvme{LOCAL_RANK}") == "/mnt/nvme2"
        finally:
            del os.environ["LOCAL_RANK"]

    def test_no_placeholder_untouched(self):
        spec = {"root": "/mnt/nvme0", "x": ["a", 1]}
        assert _expand_placeholders(spec) == spec

    def test_store_options_expanded_before_creation(self, monkeypatch):
        """engine 构造时 options 已按 LOCAL_RANK 展开。"""
        captured = {}

        import adapter.connector as conn_mod

        def fake_create_store(type_name, options):
            captured["type"] = type_name
            captured["options"] = options
            from stores.memory import MemoryKVStore

            return MemoryKVStore(segment_bytes=4096, num_chunks=4)

        monkeypatch.setattr(conn_mod, "create_store", fake_create_store)
        monkeypatch.setenv("LOCAL_RANK", "2")
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
                "chunk_tokens": 8,
                "chunk_kv_bytes": 12288,
                "max_chunks_per_wave": 4,
                "store": {"type": "tutti_nvme", "options": {
                    "root": "/mnt/nvme{LOCAL_RANK}/kv-pool",
                }},
            }),
            cache_config=SimpleNamespace(block_size=16),
        )
        connector = TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())
        connector.shutdown()
        assert captured["type"] == "tutti_nvme"
        assert captured["options"]["root"] == "/mnt/nvme2/kv-pool"


class TestPresetNormalization:
    """preset 归一：纯十进制数字字符串转 int，其余原样。"""

    def test_digit_strings_become_int(self):
        preset = {
            "device_id": "2",
            "gpu_id": "2",
            "num_queues": "8",
            "ssnvme_path": "/dev/ssnvme2",
            "device": {"namespace_id": "1", "block_size": "4096",
                       "mount_path": "/mnt/nvme2"},
        }
        got = _normalize_preset(preset)
        assert got["device_id"] == 2
        assert got["gpu_id"] == 2
        assert got["num_queues"] == 8
        assert got["ssnvme_path"] == "/dev/ssnvme2"  # 非纯数字不动
        assert got["device"]["namespace_id"] == 1
        assert got["device"]["block_size"] == 4096
        assert got["device"]["mount_path"] == "/mnt/nvme2"

    def test_plain_values_untouched(self):
        preset = {"gpu_id": 0, "type": "local", "stripe_unit": 262144}
        assert _normalize_preset(preset) == preset

    def test_lists_recursed(self):
        assert _normalize_preset(["1", "abc", 2]) == [1, "abc", 2]


class TestTuttiStorePresetParam:
    """TuttiKVStore 的 preset 构造参数：dict 保存、归一后待 open 使用。"""

    def test_preset_stored_normalized(self, tmp_path):
        from stores.tutti_nvme.store import TuttiKVStore

        store = TuttiKVStore(
            tmp_path / "pool", num_chunks=4, segment_bytes=4096,
            preset={"device_id": "1", "gpu_id": "1"},
        )
        assert store._preset == {"device_id": 1, "gpu_id": 1}

    def test_no_preset_stays_none(self, tmp_path):
        from stores.tutti_nvme.store import TuttiKVStore

        store = TuttiKVStore(tmp_path / "pool", num_chunks=4, segment_bytes=4096)
        assert store._preset is None

    def test_preset_direct_overrides_env(self, tmp_path, monkeypatch):
        """preset dict 给出时 open 不再读 TUTTI_NVME_PRESET。"""
        from stores.tutti_nvme import store as store_mod
        from stores.tutti_nvme.store import TuttiKVStore

        store = TuttiKVStore(
            tmp_path / "pool", num_chunks=4, segment_bytes=4096,
            preset={"device_id": "1"},
        )
        monkeypatch.setenv("TUTTI_NVME_PRESET", "/nonexistent/invalid.yaml")
        called = {}

        def fake_build(preset):
            called["preset"] = preset
            raise RuntimeError("stop-here")

        monkeypatch.setattr(store_mod, "_build_runtime", fake_build)
        with pytest.raises(RuntimeError, match="stop-here"):
            store.open()
        assert called["preset"] == {"device_id": 1}  # 归一化后直达

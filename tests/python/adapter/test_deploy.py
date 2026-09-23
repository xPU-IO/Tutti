"""多副本占位符与 preset 直传的部署机制测试。"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from tutti.integration.vllm.connector import (
    TuttiConnectorV1,
    _apply_capacity_bytes,
    _apply_device_groups,
    _deployment_rank,
    _expand_placeholders,
)
from tutti.storage.tutti_nvme.store import _normalize_preset
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
                "pci_bdf": "0000:4b:00.0",
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
        assert got["preset"]["pci_bdf"] == "0000:4b:00.0"

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

    def test_initialized_world_local_rank_is_authoritative(self, monkeypatch):
        """Worker config is shared; process-local world rank selects hardware."""
        import vllm.distributed.parallel_state as parallel_state

        monkeypatch.setattr(
            parallel_state,
            "get_world_group",
            lambda: SimpleNamespace(local_rank=2),
        )
        cfg = SimpleNamespace(parallel_config=SimpleNamespace(rank=0))
        assert _deployment_rank(cfg, worker=True) == "2"
        assert _expand_placeholders(
            {"device_id": "{LOCAL_RANK}", "gpu_id": "{LOCAL_RANK}"}, cfg,
            rank=_deployment_rank(cfg, worker=True),
        ) == {"device_id": "2", "gpu_id": "2"}

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

        import tutti.storage.metadata as metadata_mod

        def fake_create_store(type_name, options):
            captured["type"] = type_name
            captured["options"] = options
            from tutti.storage.memory import MemoryKVStore

            return MemoryKVStore(segment_bytes=4096, num_chunks=4)

        monkeypatch.setattr(metadata_mod, "create_metadata_store", fake_create_store)
        monkeypatch.setenv("LOCAL_RANK", "2")
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
                "chunk_tokens": 8,
                "chunk_kv_bytes": 12288,
                "max_chunks_per_wave": 4,
                "num_layers": 3,
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
        # 调度侧没有 bind 阶段，层宽必须构造时注入：metadata store 的 scan() 在
        # layer_span 未声明时 fail-closed 返回空，漏传会让"复用已有池"的进程
        # 永远恢复不到驻留项（命中静默归零，曾实测 B 请求退化为全量重算）。
        assert captured["options"]["layer_span"] == 3

    def test_metadata_store_applies_layer_span(self, tmp_path):
        """层宽注入后成为对象几何的一半；缺省保持未声明（fail-closed）。"""
        from tutti.storage.metadata import TuttiMetadataStore

        store = TuttiMetadataStore(
            root=str(tmp_path / "pool"),
            num_chunks=4,
            segment_bytes=4096,
            layer_span=80,
        )
        assert store.layer_span == 80

        bare = TuttiMetadataStore(
            root=str(tmp_path / "bare"),
            num_chunks=4,
            segment_bytes=4096,
        )
        bare.open()
        assert bare.layer_span is None
        assert bare.scan() == []

    def test_rank_local_nvme_and_gpu_are_expanded_together(self, monkeypatch):
        """TP rank selects matching mount, daemon device, and CUDA device."""
        captured = {}
        import tutti.storage.metadata as metadata_mod

        def fake_create_store(type_name, options):
            captured.update(options)
            from tutti.storage.memory import MemoryKVStore
            return MemoryKVStore(segment_bytes=4096, num_chunks=4)

        monkeypatch.setattr(metadata_mod, "create_metadata_store", fake_create_store)
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
                "chunk_tokens": 8,
                "chunk_kv_bytes": 12288,
                "max_chunks_per_wave": 4,
                "num_layers": 3,
                "store": {"type": "tutti_nvme", "options": {
                    "root": "/mnt/nvme{LOCAL_RANK}/pool",
                    "preset": {
                        "device_id": "{LOCAL_RANK}",
                        "gpu_id": "{LOCAL_RANK}",
                    },
                }},
            }),
            cache_config=SimpleNamespace(block_size=16),
            parallel_config=SimpleNamespace(
                rank=3, tensor_parallel_size=4,
                decode_context_parallel_size=1,
            ),
        )
        connector = TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())
        connector.shutdown()
        assert captured["root"] == "/mnt/nvme0/pool"
        assert captured["tp_size"] == 4
        assert [item["root"] for item in captured["rank_options"]] == [
            f"/mnt/nvme{rank}/pool" for rank in range(4)
        ]
        assert [item["preset"]["device_id"]
                for item in captured["rank_options"]] == [str(rank)
                                                          for rank in range(4)]
        assert [item["preset"]["gpu_id"]
                for item in captured["rank_options"]] == [str(rank)
                                                          for rank in range(4)]

    def test_device_groups_expand_per_rank_in_scheduler(self, monkeypatch):
        """scheduler 侧：8 卡 device_groups 按 rank 展开为各自盘组。"""
        captured = {}
        import tutti.storage.metadata as metadata_mod

        def fake_create_store(type_name, options):
            captured.update(options)
            from tutti.storage.memory import MemoryKVStore
            return MemoryKVStore(segment_bytes=4096, num_chunks=4)

        monkeypatch.setattr(metadata_mod, "create_metadata_store",
                            fake_create_store)
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
                "chunk_tokens": 8,
                "chunk_kv_bytes": 12288,
                "max_chunks_per_wave": 4,
                "num_layers": 3,
                "store": {"type": "tutti_nvme", "options": {
                    "root": "/mnt/nvme0/pool-rank{LOCAL_RANK}",
                    "layout": "striped",
                    "stripe_unit": 4096,
                    "preset": {
                        "type": "striped",
                        "device_groups": [[0, 1], [2, 3]],
                        "gpu_id": "{LOCAL_RANK}",
                    },
                }},
            }),
            cache_config=SimpleNamespace(block_size=16),
            parallel_config=SimpleNamespace(
                rank=4, tensor_parallel_size=8,
                decode_context_parallel_size=1,
            ),
        )
        connector = TuttiConnectorV1(cfg, KVConnectorRole.SCHEDULER, object())
        connector.shutdown()
        rank_devices = [
            [device["device_id"] for device in item["preset"]["devices"]]
            for item in captured["rank_options"]
        ]
        assert rank_devices == [[0, 1]] * 4 + [[2, 3]] * 4
        # TP8 的 8 个 root 必须两两不同（scheduler 侧强校验）
        assert len({item["root"]
                    for item in captured["rank_options"]}) == 8
        # device_groups 不应泄漏进运行时 preset
        assert all("device_groups" not in item["preset"]
                   for item in captured["rank_options"])


class TestPresetNormalization:
    """preset 归一：纯十进制数字字符串转 int，其余原样。"""

    def test_digit_strings_become_int(self):
        preset = {
            "device_id": "2",
            "gpu_id": "2",
            "num_queues": "8",
            "pci_bdf": "0000:4b:00.0",
            "device": {"namespace_id": "1", "block_size": "4096",
                       "mount_path": "/mnt/nvme2"},
        }
        got = _normalize_preset(preset)
        assert got["device_id"] == 2
        assert got["gpu_id"] == 2
        assert got["num_queues"] == 8
        assert got["pci_bdf"] == "0000:4b:00.0"  # 非纯数字不动
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
        from tutti.storage.tutti_nvme.store import TuttiKVStore

        store = TuttiKVStore(
            tmp_path / "pool", num_chunks=4, segment_bytes=4096,
            preset={"device_id": "1", "gpu_id": "1"},
        )
        assert store._preset == {"device_id": 1, "gpu_id": 1}

    def test_no_preset_stays_none(self, tmp_path):
        from tutti.storage.tutti_nvme.store import TuttiKVStore

        store = TuttiKVStore(tmp_path / "pool", num_chunks=4, segment_bytes=4096)
        assert store._preset is None

    def test_preset_direct_overrides_env(self, tmp_path, monkeypatch):
        """preset dict 给出时 open 不再读 TUTTI_NVME_PRESET。"""
        from tutti.storage.tutti_nvme import store as store_mod
        from tutti.storage.tutti_nvme.store import TuttiKVStore

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


class TestDeviceGroups:
    """device_groups：多卡共用盘组（8 卡 / 每 4 个 rank 一组 2 盘条带）。"""

    def test_groups_split_by_rank_span(self):
        options = {"preset": {"type": "striped",
                              "device_groups": [[0, 1], [2, 3]]}}
        for rank in range(4):
            got = _apply_device_groups(options, rank=str(rank), tp_size=8)
            assert [d["device_id"] for d in got["preset"]["devices"]] == [0, 1]
        for rank in range(4, 8):
            got = _apply_device_groups(options, rank=str(rank), tp_size=8)
            assert [d["device_id"] for d in got["preset"]["devices"]] == [2, 3]
        # 原 options 不被就地修改
        assert "device_groups" in options["preset"]
        assert "devices" not in options["preset"]

    def test_single_group_covers_all_ranks(self):
        options = {"preset": {"device_groups": [[0, 1]]}}
        got = _apply_device_groups(options, rank="7", tp_size=8)
        assert [d["device_id"] for d in got["preset"]["devices"]] == [0, 1]

    def test_absent_groups_passthrough(self):
        options = {"preset": {"type": "local", "device_id": "0"}}
        assert _apply_device_groups(options, rank="0", tp_size=8) is options

    def test_group_count_must_divide_tp_size(self):
        options = {"preset": {"device_groups": [[0], [1], [2]]}}
        with pytest.raises(ValueError, match="整除"):
            _apply_device_groups(options, rank="0", tp_size=8)

    def test_empty_groups_rejected(self):
        options = {"preset": {"device_groups": []}}
        with pytest.raises(ValueError, match="非空"):
            _apply_device_groups(options, rank="0", tp_size=8)


class TestCapacityBytesToChunks:
    """``capacity_bytes``（数据盘物理总量）→ ``num_chunks`` 的换算。

    方案 A 语义：用户按"这些盘一共占多少"给值；Python 层按放置几何换算成
    槽位数，底层仍只按槽位数建文件（不感知容量单位）。一个槽位是一个
    文件（整块 payload），slot 在盘间轮转。
    """

    # HY3 几何：80 层 × 128 KiB = 10 MiB/槽（一个文件）。
    SEGMENT_BYTES = 131072
    NUM_LAYERS = 80
    TI_BYTES = 1024 ** 4

    @staticmethod
    def _base_options():
        return {
            "preset": {
                "type": "striped",
                "gpu_id": "{LOCAL_RANK}",
                "device_groups": [[0, 1], [2, 3]],
            },
        }

    def _convert(self, options):
        return _apply_capacity_bytes(
            options, segment_bytes=self.SEGMENT_BYTES,
            num_layers=self.NUM_LAYERS,
        )

    def test_four_tib_fills_four_disks(self):
        """4 TiB → 419430 槽（每槽一个 10 MiB 文件，误差 < 一个槽位）。"""
        options = self._base_options()
        options["capacity_bytes"] = 4 * self.TI_BYTES
        got = self._convert(options)
        assert got["num_chunks"] == 419430
        used = got["num_chunks"] * 10 * 1024 * 1024
        assert used <= 4 * self.TI_BYTES
        assert 4 * self.TI_BYTES - used < 10 * 1024 * 1024

    def test_absent_key_leaves_options_untouched(self):
        got = self._convert(self._base_options())
        assert "num_chunks" not in got
        assert "capacity_bytes" not in got

    def test_mutually_exclusive_with_num_chunks(self):
        options = self._base_options()
        options["capacity_bytes"] = self.TI_BYTES
        options["num_chunks"] = 8
        with pytest.raises(ValueError, match="互斥"):
            self._convert(options)

    def test_rejects_non_positive(self):
        options = self._base_options()
        options["capacity_bytes"] = 0
        with pytest.raises(ValueError, match="正整数"):
            self._convert(options)

    def test_rejects_capacity_smaller_than_one_slot(self):
        options = self._base_options()
        options["capacity_bytes"] = 1024
        with pytest.raises(ValueError, match="小于单槽位"):
            self._convert(options)

    def test_rejects_without_disk_shape(self):
        """（已改语义）放置无几何可换算：capacity_bytes 不再需要盘形状。

        每 slot 一个文件（payload + 对象头），换算只依赖 payload 乘积；
        留下这条是为了钉住"不要求 device_groups/devices"这一行为变化。
        """
        got = self._convert({"capacity_bytes": self.TI_BYTES})
        assert got["num_chunks"] == (
            self.TI_BYTES // (self.SEGMENT_BYTES * self.NUM_LAYERS)
        )

    def test_single_disk_uses_whole_payload_per_slot(self):
        """单盘部署：每槽就是一个文件（10 MiB），与多盘同式。"""
        options = {
            "capacity_bytes": self.TI_BYTES,
            "preset": {"devices": [{"device_id": 0}]},
        }
        got = self._convert(options)
        assert got["num_chunks"] == (
            self.TI_BYTES // (self.SEGMENT_BYTES * self.NUM_LAYERS)
        )

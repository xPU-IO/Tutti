"""TuttiKVStore——tutti IO runtime 之上的层盲 KV 存储（02 §3.1/§3.1b）。

store 对 io_key 是纯映射（私有解读见 layout.py），数据面全部经由
runtime.submit 的 DMA 请求表达：put = memory→target（write），
get = target→memory（read）。**批是一等公民**：一次 put/get 对应
一批请求、一个 Completion；partial-commit 批次在内部窗口重发
（连续两轮零接受视为失败上抛）。

runtime 注入（测试/fake）或由 TUTTI_NVME_PRESET 构造（真机）：
环境变量值为 yaml/json 内联文本或文件路径；preset 可携带
daemon_config + device_id，设备事实（pci_bdf/mount_path/namespace_id）
从 daemon 配置推导——硬件信息单一来源，preset 只写用户级参数。
"""

from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path

from .layout import Layout, decode_io_key

#: KV IO 页基：register_buffer 粒度必须为其正倍数（对齐 NVMe/DMA 路径）。
_IO_PAGE_BYTES = 4096

#: Completion.wait 的轮询间隔（毫秒）。
_POLL_INTERVAL_MS = 100


def _buffer_info(buffer) -> tuple[int, int, str, int] | None:
    """提取 (地址, 字节数, memory 类型, accel_id)；无法定位稳定地址 → None。

    支持：torch tensor（data_ptr；CUDA → device）、bytearray、ctypes 数组。
    """
    data_ptr = getattr(buffer, "data_ptr", None)
    if callable(data_ptr):
        try:
            addr = int(data_ptr())
            if hasattr(buffer, "numel") and hasattr(buffer, "element_size"):
                size = int(buffer.numel()) * int(buffer.element_size())
            else:
                size = int(getattr(buffer, "nbytes", 0))
        except (TypeError, ValueError, RuntimeError):
            return None
        if addr == 0 or size <= 0:
            return None
        if getattr(buffer, "is_cuda", False):
            return addr, size, "device", int(buffer.get_device())
        return addr, size, "host", -1
    if isinstance(buffer, bytearray):
        if not buffer:
            return None
        keeper = (ctypes.c_char * len(buffer)).from_buffer(buffer)
        return ctypes.addressof(keeper), len(buffer), "host", -1
    try:
        addr = ctypes.addressof(buffer)
        size = ctypes.sizeof(buffer)
    except (TypeError, ValueError):
        return None
    if addr == 0 or size <= 0:
        return None
    return addr, size, "host", -1


class _TuttiCompletion:
    """一批 runtime IO 的完成句柄；settle 时统一 release_io 并回调。"""

    def __init__(self, runtime, handles, on_settled):
        self._runtime = runtime
        self._handles = list(handles)
        self._on_settled = on_settled
        self._settled = False
        self._failed = False

    def query(self) -> bool:
        if self._settled:
            return not self._failed
        for handle in self._handles:
            _, state = self._runtime.wait(handle, 0)
            if state == "FAILED":
                self._settle(ok=False)
                return False
            if state != "COMPLETED":
                return False
        self._settle(ok=True)
        return True

    def wait(self, timeout: float | None = None) -> None:
        if self._settled:
            if self._failed:
                raise RuntimeError("该 tutti IO 批已失败（FAILED）")
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        for handle in self._handles:
            while True:
                _, state = self._runtime.wait(handle, _POLL_INTERVAL_MS)
                if state == "COMPLETED":
                    break
                if state == "FAILED":
                    self._settle(ok=False)
                    raise RuntimeError(f"tutti IO 失败（io handle={handle}）")
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("等待 tutti IO 批超时")
        self._settle(ok=True)

    def _settle(self, ok: bool) -> None:
        if self._settled:
            return
        self._settled = True
        self._failed = not ok
        for handle in self._handles:
            try:
                self._runtime.release_io(handle)
            except Exception:
                pass  # release 尽力而为：shutdown 路径可能先行回收
        self._on_settled(ok)


class TuttiKVStore:
    """tutti runtime 之上的 KVStore SPI 实现（层盲，io_key 纯映射）。"""

    def __init__(self, root, num_chunks: int, segment_bytes: int, runtime=None):
        if num_chunks <= 0:
            raise ValueError(f"num_chunks 必须为正数，得到 {num_chunks}")
        if segment_bytes <= 0:
            raise ValueError(f"segment_bytes 必须为正数，得到 {segment_bytes}")
        self._root = Path(root)
        self._num_chunks = num_chunks
        self._segment_bytes = segment_bytes
        self._runtime = runtime
        self._own_runtime = runtime is None
        self._layout = Layout(self._root, segment_bytes)
        self._opened = False
        self._live: set[bytes] = set()
        self._buffers: dict[int, tuple[int, int]] = {}
        self._mem_cache: dict[tuple[int, int], int] = {}
        self._targets: dict[str, int] = {}
        self._keepers: list = []  # 持有 ctypes 视图防 GC
        self._next_buffer_id = 0
        self._accel_id = -1
        self._execution = "device"

    # ---------- 生命周期 ----------

    @property
    def capacity_chunks(self) -> int:
        return self._num_chunks

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("tutti store 已 open")
        if self._runtime is None:
            self._runtime = _build_runtime_from_env()
        self._layout.ensure_dirs()
        self._live = self._layout.scan()
        self._sync_execution_mode()
        self._opened = True

    def close(self) -> None:
        if not self._opened:
            return
        self._opened = False
        self._live = set()
        self._buffers = {}
        self._mem_cache = {}
        self._targets = {}
        self._keepers = []
        if self._own_runtime and self._runtime is not None:
            try:
                self._runtime.shutdown(5000)
            except Exception:
                pass
            self._runtime = None
        self._own_runtime = self._runtime is None

    def _sync_execution_mode(self) -> None:
        """由 runtime 能力推导 submit 的执行模式（device 优先，host-only 回退）。"""
        try:
            caps = self._runtime.caps()
            memories = caps.get("memory", [])
        except Exception:
            return
        if "device" in memories:
            self._execution = "device"
        elif "host" in memories:
            self._execution = "host"

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("tutti store 未 open")

    # ---------- SPI ----------

    def register_buffer(self, buffer, granularity: int) -> int | None:
        self._require_open()
        if granularity is None or granularity <= 0:
            return None
        if granularity % _IO_PAGE_BYTES != 0:
            return None
        info = _buffer_info(buffer)
        if info is None:
            return None
        addr, size, kind, accel_id = info
        mem_ticket = self._mem_cache.get((addr, size))
        if mem_ticket is None:
            try:
                mem_ticket = self._runtime.register_memory(
                    addr, size, kind, accel_id=accel_id, io_granularity=granularity
                )
            except Exception:
                return None
            self._mem_cache[(addr, size)] = mem_ticket
            self._keepers.append(buffer)
        self._next_buffer_id += 1
        self._buffers[self._next_buffer_id] = (addr, size)
        return self._next_buffer_id

    def put_batch(self, batch) -> _TuttiCompletion:
        self._require_open()
        entries = self._normalize(batch, require_live=False)
        io_keys = [io_key for io_key, _, _ in entries]
        self._layout.prepare_put(io_keys, self._num_chunks)  # 容量不足 → ValueError
        requests = []
        for io_key, buffer_id, offset in entries:
            chunk_id, layer = decode_io_key(io_key)
            uri = "file://" + str(self._layout.chunk_file(chunk_id).resolve())
            requests.append(
                (
                    self._target(uri),
                    layer * self._segment_bytes,
                    self._mem_for(buffer_id),
                    offset,
                    self._segment_bytes,
                    "write",
                )
            )
        handles = self._submit_retry(requests)
        return _TuttiCompletion(
            self._runtime, handles, lambda ok: self._on_put_settled(ok, io_keys)
        )

    def get_batch(self, batch) -> _TuttiCompletion:
        self._require_open()
        entries = self._normalize(batch, require_live=True)
        requests = []
        for io_key, buffer_id, offset in entries:
            chunk_id, layer = decode_io_key(io_key)
            uri = "file://" + str(self._layout.chunk_file(chunk_id).resolve())
            requests.append(
                (
                    self._target(uri),
                    layer * self._segment_bytes,
                    self._mem_for(buffer_id),
                    offset,
                    self._segment_bytes,
                    "read",
                )
            )
        handles = self._submit_retry(requests)
        return _TuttiCompletion(self._runtime, handles, lambda ok: None)

    def drop(self, keys) -> None:
        self._require_open()
        io_keys = []
        for key in keys:
            decode_io_key(key)  # 类型/非空校验
            io_keys.append(bytes(key))
        self._layout.drop(io_keys)
        self._live.difference_update(io_keys)

    def scan(self):
        self._require_open()
        return sorted(self._live)

    # ---------- 内部 ----------

    def _normalize(self, batch, require_live: bool):
        entries = []
        for item in batch:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError(f"批条目须为 (io_key, buffer_id, offset)，得到 {item!r}")
            io_key, buffer_id, offset = item
            decode_io_key(io_key)
            if buffer_id not in self._buffers:
                raise ValueError(f"未注册的 buffer id：{buffer_id}")
            if not isinstance(offset, int):
                raise ValueError(f"offset 须为 int，得到 {type(offset).__name__}")
            _, size = self._buffers[buffer_id]
            if offset < 0 or offset + self._segment_bytes > size:
                raise ValueError(
                    f"buffer 偏移越界：offset={offset} + 段长 "
                    f"{self._segment_bytes} > buffer {size} 字节"
                )
            if require_live and bytes(io_key) not in self._live:
                raise ValueError(f"get 未驻留 key：{bytes(io_key)!r}")
            entries.append((bytes(io_key), buffer_id, offset))
        return entries

    def _target(self, uri: str) -> int:
        ticket = self._targets.get(uri)
        if ticket is None:
            ticket = self._runtime.open_batch([uri])[0]
            self._targets[uri] = ticket
        return ticket

    def _mem_for(self, buffer_id: int) -> int:
        addr, size = self._buffers[buffer_id]
        return self._mem_cache[(addr, size)]

    def _on_put_settled(self, ok: bool, io_keys) -> None:
        """put 批 settle：数据确认落盘后才建层标记并更新在场集（崩溃安全）。"""
        if ok:
            self._layout.commit_layers(io_keys)
            self._live.update(io_keys)

    def _submit_retry(self, requests):
        """提交整批；partial-commit 的被拒请求窗口重发。

        连续两轮零接受（runtime 窗口彻底耗尽）→ RuntimeError。
        """
        handles = []
        pending = list(requests)
        all_rejected_rounds = 0
        while pending:
            result = self._runtime.submit(
                pending,
                accel_id=self._accel_id,
                stream=None,
                execution=self._execution,
            )
            if not result.status_ok or result.io_handle is None:
                raise RuntimeError(f"tutti submit 整批被拒：{result.status_msg}")
            handles.append(result.io_handle)
            rejected = list(result.rejected or [])
            if not rejected:
                break
            if len(rejected) == len(pending):
                all_rejected_rounds += 1
                if all_rejected_rounds >= 2:
                    raise RuntimeError(
                        "partial-commit 连续两轮零接受：runtime IO 窗口不可用"
                    )
            else:
                all_rejected_rounds = 0
            pending = [pending[i] for i in rejected]
        return handles


# ---------- 真机 runtime 构造（TUTTI_NVME_PRESET） ----------


def _build_runtime_from_env():
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

    if "daemon_config" in preset:
        preset = _derive_device_fields(preset, yaml)

    try:
        import tutti_runtime  # bindings 构建产物（需在 sys.path/PYTHONPATH）
    except ImportError as exc:
        raise RuntimeError(
            "tutti_runtime 绑定不可用：先构建 integration/vllm-connector/"
            "bindings/python 并将其加入 PYTHONPATH"
        ) from exc

    preset_type = preset.pop("type", "local")
    preset.pop("daemon_config", None)  # 推导元键不进 runtime preset
    preset.pop("device_id", None)
    if preset_type == "striped":
        return tutti_runtime.make_striped_nvme_runtime(preset)
    if preset_type == "local":
        return tutti_runtime.make_local_nvme_runtime(preset)
    raise RuntimeError(f"未知 preset type：{preset_type}")


def _derive_device_fields(preset: dict, yaml) -> dict:
    """daemon_config + device_id → 设备字段（硬件信息单一来源）。

    daemon 配置（yaml）的 nvmes 列表按 device_id 给出权威事实：
    pci_addr、backing_mount_path、namespace_id；preset 可显式覆盖。
    设备节点名（ssnvme/backing）按 tutti 命名约定回退，可显式给出。
    """
    daemon_path = preset.get("daemon_config")
    device_id = preset.get("device_id")
    if not daemon_path or device_id is None:
        raise RuntimeError("preset 携带 daemon_config 时必须同时给出 device_id")
    daemon = yaml.safe_load(Path(daemon_path).read_text())
    entry = next(
        (n for n in daemon.get("nvmes", []) if n.get("device_id") == device_id), None
    )
    if entry is None:
        raise RuntimeError(f"daemon 配置无 device_id={device_id} 的 NVMe 条目")
    device = dict(preset.get("device") or {})
    namespace_id = device.get("namespace_id", entry.get("namespace_id", 1))
    device.setdefault("pci_bdf", entry["pci_addr"])
    device.setdefault("mount_path", entry["backing_mount_path"])
    device.setdefault("namespace_id", namespace_id)
    device.setdefault("ssnvme_path", f"/dev/ssnvme{device_id}")
    device.setdefault(
        "backing_device", f"/dev/snvme{device_id}n{device.get('namespace_id', 1)}"
    )
    derived = dict(preset)
    derived["device"] = device
    return derived

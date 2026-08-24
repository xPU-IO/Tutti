"""TuttiKVStore——tutti IO runtime 之上的层盲 KV 存储。

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
import logging
import os
import time
from pathlib import Path

from .layout import Layout, decode_io_key
from .striped_layout import StripedLayout

#: 运行日志（池归属校验等部署问题的非静默说明）。
_LOG = logging.getLogger(__name__)

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

    def __init__(self, root, num_chunks: int, segment_bytes: int,
                 runtime=None, io_stream=None, preset=None,
                 layout="file_per_chunk", mounts=None, stripe_unit=None):
        """preset 为 dict 时优先于 TUTTI_NVME_PRESET 环境变量构造 runtime。

        preset 的字符串值恰为纯十进制整数时转为 int（配置占位符替换后
        的数字字符串由此归一，如 device_id / gpu_id）。

        ``layout="striped"`` 选择条带逻辑 target；其 ``mounts`` 与
        ``stripe_unit`` 仅作用于该布局，默认 file_per_chunk 不变。
        """
        if num_chunks <= 0:
            raise ValueError(f"num_chunks 必须为正数，得到 {num_chunks}")
        if segment_bytes <= 0:
            raise ValueError(f"segment_bytes 必须为正数，得到 {segment_bytes}")
        self._root = Path(root)
        self._num_chunks = num_chunks
        self._segment_bytes = segment_bytes
        self._runtime = runtime
        self._own_runtime = runtime is None
        self._preset = _normalize_preset(preset) if preset is not None else None
        self._key_namespace: bytes | None = None
        if layout in (None, "file_per_chunk", "file"):
            self._layout = Layout(self._root, segment_bytes)
        elif layout == "striped":
            if mounts is None:
                mounts = _preset_mounts(self._preset)
            if stripe_unit is None:
                raise ValueError("striped target 必须提供 stripe_unit")
            self._layout = StripedLayout(
                self._root, segment_bytes, mounts, stripe_unit
            )
        else:
            raise ValueError(f"未知 tutti_nvme layout：{layout!r}")
        self._opened = False
        self._live: set[bytes] = set()
        self._buffers: dict[int, tuple[int, int]] = {}
        self._mem_cache: dict[tuple[int, int], int] = {}
        self._targets: dict[str, int] = {}
        self._target_sizes: dict[str, int] = {}
        self._keepers: list = []  # 持有 ctypes 视图防 GC
        self._next_buffer_id = 0
        self._accel_id = -1
        # 'auto' 在 open() 惰性解析为专用 IO 流（见 _resolve_auto_stream）；
        # 其余取值（int 句柄 / None）原样使用。
        self._io_stream_raw = io_stream
        self._io_stream = None if io_stream == "auto" else io_stream
        self._io_stream_obj = None  # 专用流引用，防句柄悬空
        self._execution = "device"

    # ---------- 生命周期 ----------

    @property
    def capacity_chunks(self) -> int:
        return self._num_chunks

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("tutti store 已 open")
        if self._runtime is None:
            if self._preset is not None:
                self._runtime = _build_runtime(self._preset)
            else:
                self._runtime = _build_runtime_from_env()
        self._layout.ensure_dirs()
        self._live = self._layout.scan()
        # 池归属校验：manifest 与命名空间不一致 → 空池语义（miss），
        # 禁止静默复用异构数据（不同模型/几何的旧池）。
        if self._key_namespace is not None:
            if not self._layout.check_namespace(self._key_namespace):
                _LOG.warning(
                    "池 %s 的命名空间 manifest 与当前配置不一致——按空池"
                    "处理（不读旧数据）；如需腾挪请人工清理",
                    self._root,
                )
                self._live = set()
        if self._io_stream_raw == "auto":
            self._resolve_auto_stream()
        self._sync_execution_mode()
        self._opened = True

    def _resolve_auto_stream(self) -> None:
        """在 runtime 加速器上建专用 IO 流并取其句柄。

        默认流句柄为 0，与绑定的空指针语义冲突（submit 会当作未给流
        拒绝），故 'auto' 一律建专用流；落在 preset 的 gpu_id 设备上
        （runtime 校验流须属于自身加速器）。open 仅发生在 worker 进程，
        此处 CUDA 必已初始化。
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("io_stream='auto' 需要 CUDA 可用")
        accel = 0
        if isinstance(self._preset, dict):
            raw = self._preset.get("gpu_id", 0)
            try:
                accel = int(raw)
            except (TypeError, ValueError):
                accel = 0
        self._io_stream_obj = torch.cuda.Stream(device=f"cuda:{accel}")
        self._io_stream = int(self._io_stream_obj.cuda_stream)

    def close(self) -> None:
        if not self._opened:
            return
        self._opened = False
        self._live = set()
        self._buffers = {}
        self._mem_cache = {}
        self._targets = {}
        self._target_sizes = {}
        self._keepers = []
        if self._own_runtime and self._runtime is not None:
            try:
                self._runtime.shutdown(5000)
            except Exception:
                pass
            self._runtime = None
        self._own_runtime = self._runtime is None

    def _sync_execution_mode(self) -> None:
        """推导 submit 执行模式：无 io_stream → host 路径；有则按 runtime
        能力（device 优先，host-only 回退）。"""
        if self._io_stream is None:
            self._execution = "host"
            return
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
        """注册批量 IO 的内存缓冲，返回 store 内部 buffer_id。

        具体注册的是哪块内存：engine.bind 传入的 staging 环窗缓冲——
        一段连续 device 显存（或 host 内存），按"槽"等分，布局为
        slots × segment_bytes，槽 i 占据
        [i×segment_bytes, (i+1)×segment_bytes)。本方法的 buffer 即
        该段内存的起始视图；size 取自对象属性（numel×element_size）。

        granularity：单条批量 IO 的字节粒度 = 层段宽（segment_bytes）。
        DMA 以此为对齐与切分单位；必须是 _IO_PAGE_BYTES 的正倍数。

        返回 buffer_id：store 局部编号，映射到底层 (addr, size)。
        之后 put_batch/get_batch 的每条 IO 以
        (io_key, buffer_id, offset) 寻址，offset 即槽内字节偏移。

        语义：同一 (addr, size) 只向 runtime 注册一次（ticket 复用）；
        注册期间持有 buffer 引用防止回收；注册失败 → None（调用方
        视为该缓冲不可用于 DMA）。
        """
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
        self._invalidate_stale_targets(entries)
        requests = []
        for io_key, buffer_id, offset in entries:
            chunk_id, layer = decode_io_key(io_key)
            uri = self._layout.target_uri(chunk_id)
            requests.append(
                (
                    self._target(uri, self._layout.target_size(chunk_id)),
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
        self._invalidate_stale_targets(entries)
        requests = []
        for io_key, buffer_id, offset in entries:
            chunk_id, layer = decode_io_key(io_key)
            uri = self._layout.target_uri(chunk_id)
            requests.append(
                (
                    self._target(uri, self._layout.target_size(chunk_id)),
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
        chunk_ids = set()
        for key in keys:
            chunk_id, _ = decode_io_key(key)  # 类型/非空校验
            io_keys.append(bytes(key))
            chunk_ids.add(chunk_id)
        self._layout.drop(io_keys)
        self._live.difference_update(io_keys)
        # Dropping the last layer unlinks the backing file(s).  Do not reuse
        # a runtime target ticket that still owns the old unlinked inode when
        # the same chunk is written again.
        for chunk_id in chunk_ids:
            uri = self._layout.target_uri(chunk_id)
            self._targets.pop(uri, None)
            self._target_sizes.pop(uri, None)

    def scan(self):
        self._require_open()
        return sorted(self._live)

    def has(self, io_key) -> bool:
        """存活查询（O(1)）：读取侧跳过无数据层（混合注意力模型的
        线性注意力层从未落盘，属正常状态而非错误）。"""
        return io_key in self._live

    def set_key_namespace(self, namespace: bytes) -> None:
        """声明 key 命名空间（engine 构造期注入，open 前生效）。

        用于池归属 manifest 校验：不透明字节串，本层不解读字段。
        """
        if self._opened:
            raise RuntimeError("命名空间须在 open 之前注入")
        self._key_namespace = bytes(namespace)

    def set_layer_span(self, num_layers: int) -> None:
        """声明层宽（bind 后由引擎注入）：数据文件按层宽全尺寸预分配。"""
        self._layout.set_layer_span(num_layers)

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

    def _target(self, uri: str, target_size: int | None = None) -> int:
        ticket = self._targets.get(uri)
        if ticket is None:
            ticket = self._runtime.open_batch([uri])[0]
            self._targets[uri] = ticket
            if target_size is not None:
                self._target_sizes[uri] = target_size
            elif uri.startswith("file://"):
                try:
                    self._target_sizes[uri] = os.stat(uri[len("file://"):]).st_size
                except OSError:
                    self._target_sizes[uri] = -1
            else:
                self._target_sizes[uri] = -1
        return ticket

    def _invalidate_stale_targets(self, entries) -> None:
        """按当前文件尺寸失效过期的 target 票据。

        数据文件随写入逐层增长，runtime 票据在开票时解析文件尺寸——
        文件增长后旧票据的 target 尺寸过期，写更高层段会被
        OUT_OF_RANGE 拒绝。组批前按 stat 对账，尺寸变化即弃票重开
        （旧票据由 runtime 在 shutdown 统一回收，无逐票释放接口）。
        """
        for io_key, _, _ in entries:
            chunk_id, _ = decode_io_key(io_key)
            uri = self._layout.target_uri(chunk_id)
            if uri not in self._targets:
                continue
            size_now = self._layout.target_size(chunk_id)
            if size_now != self._target_sizes.get(uri):
                self._targets.pop(uri, None)

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
                stream=self._io_stream,
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


def _normalize_preset(preset) -> dict:
    """递归归一 preset：字符串值恰为纯十进制整数时转 int。"""
    if isinstance(preset, dict):
        return {k: _normalize_preset(v) for k, v in preset.items()}
    if isinstance(preset, list):
        return [_normalize_preset(v) for v in preset]
    if isinstance(preset, str) and preset.strip().isdigit():
        return int(preset)
    return preset


def _preset_mounts(preset):
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


def _build_runtime(preset: dict):
    """按 preset dict 构造真机 runtime（daemon_config 推导与归一同环境变量路径）。"""
    import yaml

    if not isinstance(preset, dict):
        raise RuntimeError("preset 必须是映射")
    if "daemon_config" in preset:
        preset = _derive_device_fields(preset, yaml)

    try:
        import tutti_runtime  # bindings 构建产物（需在 sys.path/PYTHONPATH）
    except ImportError as exc:
        raise RuntimeError(
            "tutti_runtime 绑定不可用：先构建 integration/vllm-connector/"
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
    return _build_runtime(_normalize_preset(preset))


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

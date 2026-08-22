"""worker 侧逐层编排：把 vLLM 的层回调翻译为引擎执行态调用。

读取按层流水（等待第 L 层时预取第 L+1 层）；写入逐层发起并以在途
句柄上限背压；staging 环窗显存在此分配并注入引擎。
"""

from __future__ import annotations

import torch

from engine.staging import RingWindow


class WorkerImpl:
    """worker 角色的执行编排器。

    契约：
    - 构造注入引擎与在途句柄上限（max_in_flight_layers，0 = 不限）。
    - 显存对象经两个登记回调进入（单块跨层池或逐层映射，二选一）；
      首个执行回调触发惰性绑定（分配 staging 环窗显存并接入引擎）。
    - 读取流程：start_load_kv 组批并预取第 0 层；wait_for_layer_load
      等待当前层后预取下一层；末层等待返回即释放读保护。
    - 写入流程：save_kv_layer 逐层发起一批；wait_for_save 等待全部
      完成并结算写入（在途句柄超限时先行等待最旧一批）。
    - 近似视图未遂（读取保护失败）的块经 get_block_ids_with_load_errors
      上报，由上层重算兜底。
    """

    def __init__(self, engine, max_in_flight_layers: int = 0):
        """engine 为编排核心实例；max_in_flight_layers 为负 → ValueError。"""
        if not isinstance(max_in_flight_layers, int) or max_in_flight_layers < 0:
            raise ValueError(
                f"max_in_flight_layers 须为非负整数，got {max_in_flight_layers!r}"
            )
        self._engine = engine
        self._max_in_flight = max_in_flight_layers
        # configure 注入
        self._chunk_tokens = 0
        self._chunk_kv_bytes = 0
        self._max_chunks_per_wave = 0
        self._block_size = 16
        # 显存对象登记
        self._kv_caches: dict | None = None
        self._cross_pool = None
        self._layer_names: list[str] = []
        # 绑定态
        self._num_layers = 0
        self._bound = False
        self._window: RingWindow | None = None
        # 本步计划与编排状态
        self._metadata = None
        self._load_keys: list[bytes] = []
        self._load_first_blocks: list[int] = []
        self._load_handles: dict[int, object] = {}
        self._pinned = False
        self._load_error_blocks: set[int] = set()
        self._call_seq = 0
        self._save_keys: list[bytes] | None = None
        self._save_first_blocks: list[int] = []
        self._save_inflight: list = []

    # ---- 配置与登记 ----

    def configure(
        self,
        chunk_tokens: int,
        chunk_kv_bytes: int,
        max_chunks_per_wave: int,
        block_size: int,
    ) -> None:
        """注入编排所需的几何参数（由挂载点从配置读出）。"""
        self._chunk_tokens = chunk_tokens
        self._chunk_kv_bytes = chunk_kv_bytes
        self._max_chunks_per_wave = max_chunks_per_wave
        self._block_size = block_size

    def register_kv_caches(self, kv_caches: dict) -> None:
        """登记逐层显存对象映射（层名顺序即层序）。"""
        self._kv_caches = dict(kv_caches)
        self._layer_names = list(kv_caches.keys())

    def register_cross_layers_kv_cache(self, kv_cache, attn_backend) -> None:
        """登记单块跨层显存对象（第 0 维为层数）。"""
        self._cross_pool = kv_cache

    def set_metadata(self, metadata) -> None:
        """接收本步传输计划。"""
        self._metadata = metadata

    # ---- 读取编排 ----

    def start_load_kv(self, forward_context=None, **kwargs) -> None:
        """组读取批并预取第 0 层。"""
        self._ensure_bound()
        self._load_keys = []
        self._load_first_blocks = []
        self._load_handles = {}
        self._call_seq = 0
        self._pinned = False
        keys: list[bytes] = []
        first_blocks: list[int] = []
        for meta in getattr(self._metadata, "requests", []) or []:
            if meta.load_tokens <= 0:
                continue
            count = meta.load_tokens // self._chunk_tokens
            req_keys, _ = self._engine.hash_keys(meta.token_ids)
            req_keys = req_keys[:count]
            try:
                self._engine.pin(req_keys)
            except KeyError:
                # 近似视图未遂：该区间块上报重算
                for i in range(count):
                    pos = i * self._chunk_tokens
                    span = min(self._chunk_tokens, len(meta.block_ids) * self._block_size - pos)
                    if span <= 0:
                        break
                    self._load_error_blocks.update(
                        meta.block_ids[pos // self._block_size: (pos + span) // self._block_size + 1]
                    )
                continue
            keys.extend(req_keys)
            first_blocks.extend(self._first_blocks(meta, count))
        if keys:
            self._load_keys = keys
            self._load_first_blocks = first_blocks
            self._pinned = True
            self._start_load_layer(0)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """等待指定层读取完成，并预取下一层。"""
        idx = self._resolve_layer(layer_name)
        handle = self._load_handles.pop(idx, None)
        if handle is not None:
            handle.wait()
        nxt = idx + 1
        if nxt < self._num_layers and self._load_keys and nxt not in self._load_handles:
            self._start_load_layer(nxt)
        if idx == self._num_layers - 1 and self._pinned:
            self._engine.unpin(self._load_keys)
            self._pinned = False

    # ---- 写入编排 ----

    def save_kv_layer(self, layer_name: str, kv_layer=None, attn_metadata=None, **kwargs) -> None:
        """发起指定层的写入批（本步首个写入层时组批）。"""
        self._ensure_bound()
        idx = self._resolve_layer(layer_name)
        if self._save_keys is None:
            keys: list[bytes] = []
            first_blocks: list[int] = []
            for meta in getattr(self._metadata, "requests", []) or []:
                if meta.save_chunk_count <= 0:
                    continue
                req_keys, _ = self._engine.hash_keys(meta.token_ids)
                start = meta.save_chunk_start
                end = start + meta.save_chunk_count
                keys.extend(req_keys[start:end])
                first_blocks.extend(self._first_blocks(meta, end)[start:end])
            self._save_keys = keys
            self._save_first_blocks = first_blocks
            if not keys:
                return
        if not self._save_keys:
            return
        handle = self._engine.store_layer(
            self._save_keys, idx, self._save_first_blocks
        )
        self._save_inflight.append(handle)
        if self._max_in_flight and len(self._save_inflight) > self._max_in_flight:
            self._save_inflight.pop(0).wait()

    def wait_for_save(self) -> None:
        """等待全部写入完成并结算。"""
        for handle in self._save_inflight:
            handle.wait()
        self._save_inflight = []
        if self._save_keys:
            self._engine.confirm_store(self._save_keys, ok=True)
        self._save_keys = None
        self._save_first_blocks = []
        self._call_seq = 0

    # ---- 状态与收尾 ----

    @property
    def window(self) -> RingWindow | None:
        """绑定的 staging 环窗（未绑定为 None）。"""
        return self._window

    def get_block_ids_with_load_errors(self) -> set[int]:
        """上报读取未遂的块集合（读取后清空）。"""
        errors = self._load_error_blocks
        self._load_error_blocks = set()
        return errors

    def shutdown(self) -> None:
        """等待在途并关闭引擎（幂等由引擎保证）。"""
        self._engine.close()

    # ---- 内部 ----

    def _start_load_layer(self, layer_idx: int) -> None:
        """发起一层的读取批。"""
        self._load_handles[layer_idx] = self._engine.load_layer(
            self._load_keys, layer_idx, self._load_first_blocks
        )

    def _first_blocks(self, meta, count: int) -> list[int]:
        """计算前 count 个 chunk 各自首 token 所在块号。"""
        blocks = meta.block_ids
        per_chunk = [blocks[(i * self._chunk_tokens) // self._block_size] for i in range(count)]
        return per_chunk

    def _resolve_layer(self, layer_name: str) -> int:
        """把层名解析为层序号。

        逐层登记模式用登记序；单块跨层模式（无层名登记）按本步内
        的调用序号轮转。
        """
        if self._kv_caches:
            try:
                return self._layer_names.index(layer_name)
            except ValueError:
                return 0
        idx = self._call_seq
        self._call_seq += 1
        return idx % max(self._num_layers, 1)

    def _ensure_bound(self) -> None:
        """惰性绑定：分配 staging 环窗显存并接入引擎（恰一次）。"""
        if self._bound:
            return
        if self._cross_pool is not None:
            num_layers = int(self._cross_pool.shape[0])
            device = self._cross_pool.device
        elif self._kv_caches:
            num_layers = len(self._kv_caches)
            first = next(iter(self._kv_caches.values()))
            device = getattr(first, "device", "cpu")
        else:
            raise RuntimeError("尚未登记任何显存对象，无法绑定")
        if num_layers <= 0:
            raise RuntimeError("登记的层数为空，无法绑定")
        if self._chunk_kv_bytes % num_layers != 0:
            raise ValueError("chunk_kv_bytes 不能按层数整分")
        segment_bytes = self._chunk_kv_bytes // num_layers
        slots = 2 * self._max_chunks_per_wave
        staging = torch.empty(
            slots * segment_bytes, dtype=torch.uint8, device=device
        )
        # 存储侧按缓冲协议登记：本机驻留内存用共享底层存储的数组视图，
        # 加速侧内存保持原对象（由实现按地址登记）。
        raw = staging.numpy() if staging.device.type == "cpu" else staging
        window = RingWindow(raw, slots, segment_bytes)
        blocks_per_chunk = -(-self._chunk_tokens // self._block_size)
        self._engine.bind(self._kv_caches or {}, window, num_layers, blocks_per_chunk)
        self._num_layers = num_layers
        self._window = window
        self._bound = True

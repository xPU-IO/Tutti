"""KVEngine：计划态转发语义索引，执行态编排 staging 环窗、store 与传输路径。"""

from __future__ import annotations

from typing import Sequence

from engine.staging import RingWindow
from engine.transfer import StagedTransfer, select_transfer
from index.chunk_index import (
    ChunkIndex,
    StorePlan,
    chunk_key_of,
    derive_io_key,
    layer_of,
)

# 冷启动恢复时 io_key 分组的来源宽度（chunk key 16 字节 + 层编号 2 字节）
_IO_KEY_BYTES = 18


class _PostCompletion:
    """底层完成句柄 wait 返回后执行收尾动作的完成句柄。

    契约：wait 先等底层完成再执行收尾动作（恰一次）；query 只反映
    底层完成状态，收尾动作由 wait 触发。
    """

    __slots__ = ("_inner", "_after", "_done")

    def __init__(self, inner, after):
        """inner 为底层完成句柄，after 为无参收尾可调用。"""
        self._inner = inner
        self._after = after
        self._done = False

    def wait(self) -> None:
        """阻塞至底层完成并执行收尾动作（恰一次）。"""
        self._inner.wait()
        if not self._done:
            self._done = True
            self._after()

    def query(self) -> bool:
        """非阻塞查询底层是否完成。"""
        return self._inner.query()


class KVEngine:
    """编排核心：构造注入 store，语义索引自建并做冷启动恢复。

    config 契约（键全部与硬件无关）：
    - chunk_tokens：每 chunk 的 token 数（正整数）。
    - chunk_kv_bytes：单 chunk 的 KV 字节数（正整数）。
    - max_chunks_per_wave：单波最大 chunk 数（正整数）。
    - gather_fn / scatter_fn：可选搬运钩子（缺省 None，搬运为 no-op），
      语义见传输路径。

    构造期：打开 store、建立语义索引、对 store 的存活枚举按 chunk key
    分组；层数定案（bind）后才把层完整的 chunk 灌入索引（见 io_key
    线格式约定）。
    """

    def __init__(self, config: dict, store):
        """config 见类契约；store 为 KVStore 实现。参数非法 → ValueError。"""
        self._chunk_tokens = _positive_int(config, "chunk_tokens")
        self._chunk_kv_bytes = _positive_int(config, "chunk_kv_bytes")
        self._max_chunks_per_wave = _positive_int(config, "max_chunks_per_wave")
        for name in ("gather_fn", "scatter_fn"):
            fn = config.get(name)
            if fn is not None and not callable(fn):
                raise ValueError(f"config[{name!r}] 须为可调用或 None，got {fn!r}")
        self._config = dict(config)
        self._store = store
        self._closed = False
        store.open()
        self._index = ChunkIndex(store.capacity_chunks, self._chunk_tokens)
        # 冷启动分组：层数定案前暂存；层集合不完整的 chunk 视为缺失。
        self._scan_groups = _group_scan(store)
        self._restored = False
        # 恢复层集的连续性：混合注意力模型（部分层无 KV 语义，如
        # Hy3 偶数层）恢复层集不连续——前缀恢复会跳过无数据层的
        # 状态建立，输出错误；非连续时命中查询安全降级为 0。
        self._contiguous_restore = True
        # 执行态（bind 后可用）
        self._num_layers: int | None = None
        self._segment_bytes: int | None = None
        self._window: RingWindow | None = None
        self._transfer: StagedTransfer | None = None
        self._staging_buffer_id: int | None = None
        self._inflight: list = []

    # ---- 计划态 ----

    def lookup_prefix(self, token_ids: Sequence[int]) -> int:
        """查询前缀命中的 token 数（转发语义索引）。

        恢复层集非连续（混合注意力模型）时恒为 0：无数据层的推理
        状态无法从前缀恢复，命中报告会导致输出错误（安全降级，
        见 _contiguous_restore）。
        """
        self._require_open()
        if not self._contiguous_restore:
            return 0
        return self._index.lookup_prefix(token_ids)

    def hash_keys(
        self,
        token_ids: Sequence[int],
        start: int = 0,
        parent: bytes = b"",
    ) -> tuple[list[bytes], bytes]:
        """把 token 序列折叠为 chunk key 链（转发语义索引）。"""
        self._require_open()
        return self._index.hash_keys(token_ids, start, parent)

    def plan_store(self, keys) -> StorePlan | None:
        """受理写入计划并预留容量（转发语义索引）。

        层数已定案时，驱逐的 chunk 展开为其全部层的 io_key 交给
        store 执行删除；层数未定案（bind 之前）时不执行数据面删除，
        驱逐善后由层数定案后的权威进程结算。
        """
        self._require_open()
        plan = self._index.plan_store(keys)
        if plan is None:
            return None
        if plan.evicted_keys and self._num_layers is not None:
            self._store.drop(_expand_io_keys(plan.evicted_keys, self._num_layers))
        return plan

    def confirm_store(self, keys, ok: bool = True) -> None:
        """结算写入计划（转发语义索引）。"""
        self._require_open()
        self._index.confirm_store(keys, ok)

    def pin(self, keys) -> None:
        """对一批 chunk key 加读保护；任一未驻留 → KeyError。"""
        self._require_open()
        self._index.pin(keys)

    def unpin(self, keys) -> None:
        """解除读保护；无保护计数的 key → KeyError。"""
        self._require_open()
        self._index.unpin(keys)

    # ---- 执行态 ----

    def bind(
        self,
        kv_caches: dict,
        window: RingWindow,
        num_layers: int,
        blocks_per_chunk: int,
        *,
        gather_fn=None,
        scatter_fn=None,
    ) -> None:
        """绑定执行态：层数定案、选定传输路径、staging 环窗接入 store。

        kv_caches 为布局侧的层缓存映射（staged 路径不使用）；window 为
        部署层注入的环形窗口；num_layers、blocks_per_chunk 为正整数。
        gather_fn / scatter_fn 为可选搬运钩子，给出时优先于 config 中
        的同名项（部署层在绑定期按池几何构造后注入）。
        重复 bind、chunk_kv_bytes 不能按层数整分、窗口几何不匹配、
        staging 环窗被 store 拒收 → RuntimeError / ValueError。
        """
        self._require_open()
        if self._window is not None:
            raise RuntimeError("bind 恰允许一次")
        if not _is_int(num_layers) or num_layers <= 0:
            raise ValueError(f"num_layers 须为正整数，got {num_layers!r}")
        if not _is_int(blocks_per_chunk) or blocks_per_chunk <= 0:
            raise ValueError(f"blocks_per_chunk 须为正整数，got {blocks_per_chunk!r}")
        if not isinstance(window, RingWindow):
            raise ValueError(f"window 须为 RingWindow 实例，got {window!r}")
        for name, fn in (("gather_fn", gather_fn), ("scatter_fn", scatter_fn)):
            if fn is not None and not callable(fn):
                raise ValueError(f"{name} 须为可调用或 None，got {fn!r}")
        if self._chunk_kv_bytes % num_layers != 0:
            raise ValueError(
                f"chunk_kv_bytes({self._chunk_kv_bytes}) 不能被 "
                f"num_layers({num_layers}) 整分"
            )
        segment_bytes = self._chunk_kv_bytes // num_layers
        if window.segment_bytes != segment_bytes:
            raise ValueError(
                f"窗口槽宽 {window.segment_bytes} 与层段宽 {segment_bytes} 不匹配"
            )
        if window.capacity_per_wave < self._max_chunks_per_wave:
            raise ValueError(
                f"窗口单波容量 {window.capacity_per_wave} 小于 "
                f"max_chunks_per_wave({self._max_chunks_per_wave})"
            )
        self._num_layers = num_layers
        self._segment_bytes = segment_bytes
        self._window = window
        config = dict(self._config)
        if gather_fn is not None:
            config["gather_fn"] = gather_fn
        if scatter_fn is not None:
            config["scatter_fn"] = scatter_fn
        self._transfer = select_transfer(kv_caches, self._store, config)
        buffer_id = self._store.register_buffer(window.buffer, segment_bytes)
        if buffer_id is None:
            raise RuntimeError(
                f"staging 环窗被 store 拒收（granularity={segment_bytes}）"
            )
        self._staging_buffer_id = buffer_id
        self._deferred_restore()

    def load_layer(self, keys, layer_idx: int, dst_first_blocks):
        """发起一批读取：一层 × N chunk，持久化 → staging 槽 → 目的侧。

        返回完成句柄；wait 返回即源侧搬运（scatter）已执行。批内
        chunk 数超过单波容量、层号越界、未 bind → ValueError /
        RuntimeError。store 侧未知 key 的异常原样上抛——混合注意力
        模型的无 KV 层（如 Hy3 偶数层）经存活探测跳过：store 提供
        has 查询时整层缺数据的读取降级为即时完成（scatter 无事可做，
        该层槽内容不被引用）。
        """
        keys = self._prepare_layer_call(keys, layer_idx)
        has = getattr(self._store, "has", None)
        if has is not None:
            keys = [k for k in keys if has(derive_io_key(k, layer_idx))]
            if not keys:
                return _ImmediateHandle()
        wave, slots = self._window.acquire(len(keys))
        batch = [
            (derive_io_key(k, layer_idx), self._staging_buffer_id, self._window.slot_offset(s))
            for k, s in zip(keys, slots)
        ]
        completion = self._store.get_batch(batch)
        return self._settle(wave, keys, layer_idx, dst_first_blocks, slots, completion, is_load=True)

    def store_layer(self, keys, layer_idx: int, src_first_blocks):
        """发起一批写入：一层 × N chunk，源侧 → staging 槽 → 持久化。

        返回完成句柄；源侧搬运（gather）在提交前已执行。其余契约同
        load_layer。
        """
        keys = self._prepare_layer_call(keys, layer_idx)
        wave, slots = self._window.acquire(len(keys))
        self._transfer.gather(keys, layer_idx, src_first_blocks, slots)
        batch = [
            (derive_io_key(k, layer_idx), self._staging_buffer_id, self._window.slot_offset(s))
            for k, s in zip(keys, slots)
        ]
        completion = self._store.put_batch(batch)
        return self._settle(wave, keys, layer_idx, src_first_blocks, slots, completion, is_load=False)

    def wait_idle(self) -> None:
        """等待全部在途批次完成。"""
        inflight = self._inflight
        self._inflight = []
        for completion in inflight:
            completion.wait()

    def close(self) -> None:
        """收尾：等待在途批次并关闭 store；幂等。"""
        if self._closed:
            return
        self._closed = True
        try:
            self.wait_idle()
        finally:
            self._store.close()

    # ---- 内部 ----

    def _prepare_layer_call(self, keys, layer_idx: int) -> list[bytes]:
        """校验执行态前置条件，返回 key 列表。"""
        self._require_open()
        if self._window is None:
            raise RuntimeError("执行态方法须在 bind 之后调用")
        if not _is_int(layer_idx) or not 0 <= layer_idx < self._num_layers:
            raise ValueError(
                f"layer_idx 须在 [0, {self._num_layers}) 内，got {layer_idx!r}"
            )
        keys = list(keys)
        if not keys:
            raise ValueError("keys 不能为空")
        if len(keys) > self._max_chunks_per_wave:
            raise ValueError(
                f"批内 chunk 数 {len(keys)} 超过单波容量 {self._max_chunks_per_wave}"
            )
        return keys

    def _settle(
        self,
        wave: int,
        keys: list[bytes],
        layer_idx: int,
        first_blocks,
        slots: list[int],
        completion,
        is_load: bool,
    ):
        """登记波次完成事件与在途句柄；load 方向追加目的侧搬运。"""
        handle = completion
        if is_load:
            handle = _PostCompletion(
                completion,
                lambda: self._transfer.scatter(keys, layer_idx, first_blocks, slots),
            )
        self._window.complete(wave, handle)
        self._inflight.append(handle)
        return handle

    def sync_from_store(self) -> None:
        """从盘上持久层枚举增量灌入索引（幂等，可重复调用）。

        多副本部署下索引属主与命中查询方可能分属不同进程（worker
        落盘、调度侧查询），查询方以盘上标记为准对账——仅扫持久层
        元数据目录，不触碰数据面。完整性基准取扫描结果中的多数
        层集（混合注意力模型仅部分层槽有数据，见 _deferred_restore）。
        """
        self._require_open()
        groups = _group_scan(self._store)
        if not groups:
            return
        counts: dict[frozenset, int] = {}
        for layers in groups.values():
            key = frozenset(layers)
            counts[key] = counts.get(key, 0) + 1
        expected = max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        full_keys = [
            chunk_key for chunk_key, layers in groups.items()
            if frozenset(layers) == expected
        ]
        self._index.restore(full_keys)
        self._scan_groups = groups
        self._restored = True
        # 层集连续性：排序后恰为区间 [min, max] 全体才连续（混合
        # 注意力模型的恢复层集有洞，如 Hy3 的 {1,3,…,79}）。
        ordered = sorted(expected)
        self._contiguous_restore = ordered == list(
            range(ordered[0], ordered[-1] + 1)
        )

    def _deferred_restore(self) -> None:
        """层数定案后执行冷启动灌入：层完整的 chunk 才驻留。

        完整性基准取扫描结果中的多数层集——混合注意力模型（如
        Hy3：奇数层全注意力落盘、偶数层线性注意力无 KV）仅部分
        层槽有数据，按 range(num_layers) 全量判定会整体判缺；
        多数层集对均匀模型等价于全量，对混合模型取真实层集，
        孤立的异构/残缺 chunk 被排除。
        """
        if self._restored:
            return
        self.sync_from_store()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("engine 已 close")


class _ImmediateHandle:
    """即时完成句柄：wait 立即返回，query 恒为真（无数据层的读取）。"""

    def wait(self) -> None:
        return None

    def query(self) -> bool:
        return True


def _group_scan(store) -> dict[bytes, set[int]]:
    """把 store 的存活枚举按 chunk key 分组为层集合；线格式非法的条目忽略。"""
    groups: dict[bytes, set[int]] = {}
    for io_key in store.scan():
        if not isinstance(io_key, (bytes, bytearray)) or len(io_key) != _IO_KEY_BYTES:
            continue
        chunk_key = chunk_key_of(bytes(io_key))
        groups.setdefault(chunk_key, set()).add(layer_of(bytes(io_key)))
    return groups


def _expand_io_keys(chunk_keys, num_layers: int) -> list[bytes]:
    """把一批 chunk key 展开为全部层的 io_key。"""
    return [
        derive_io_key(chunk_key, layer)
        for chunk_key in chunk_keys
        for layer in range(num_layers)
    ]


def _positive_int(config: dict, key: str) -> int:
    """从 config 读取正整数键；缺失或非法 → ValueError。"""
    value = config.get(key)
    if not _is_int(value) or value <= 0:
        raise ValueError(f"config[{key!r}] 须为正整数，got {value!r}")
    return value


def _is_int(value) -> bool:
    """判断是否为真 int（排除 bool）。"""
    return isinstance(value, int) and not isinstance(value, bool)

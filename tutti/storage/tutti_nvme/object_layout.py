"""对象层支撑的 KV 布局：Python 侧不做任何文件系统操作。

`Layout` / `StripedLayout` 的公开面（store.py 与调度侧只依赖这些方法）在这里
全部由 C++ ``StorageObjectStore`` 承担：

    槽位分配  → reserve()          对象有效性 → commit()（每对象一次）
    槽位路径  → placement.uri      崩溃恢复   → recover()
    容量      → usage()

与旧实现的关键语义差异（评审定案）：

* **按对象提交，不按层**。一个 chunk 的全部层段共用一个对象，``commit`` 只写
  一次对象头；某层没写完在结构上就等于"整个对象未提交"，读侧永远看不到半截
  chunk。Python 侧只记 ``{chunk: 已写层集合}``，齐了才调一次 commit。
* **槽位路径稳定且不含 chunk 身份**（``<root>/slots/<slot>.obj`` 或
  ``<mount>/striped/<slot>.shard<i>``），因此票据缓存可以按 URI 长驻，分配与
  回收都不需要 rename。
* **零 marker 文件**。旧实现的 ``meta/<chunk>.<layer>.ok`` 与 manifest JSON 全
  部消失，冷启动驻留集合来自 ``recover()``。

层宽（``num_layers`` → ``segment_count``）在建布局时未知，因此对象的 open 推迟
到 :meth:`set_layer_span`（引擎在此之后立刻 ``_deferred_restore``，见
``engine/core.py``）；在那之前的 :meth:`scan` 返回空集。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

# 挂在 tutti 树下的专属通道（connector 会为该树挂 handler 放行 INFO）。
_PRECREATE_LOG = logging.getLogger("tutti.precreate")

# 后台预建的节奏。batch 决定一次调用最多认领多少槽位（存储锁只在认领与发布
# 时短暂持有，所以真正决定"写路径最多等多久"的是单槽 IO，而非 batch）；
# idle 是没有可做的工作时的轮询间隔。
_PRECREATE_BATCH_SLOTS = 64
_PRECREATE_IDLE_SLEEP_S = 0.5
# 认领在飞时的重试间隔：这个状态最长持续一次单槽 IO（~44ms），只需让出一
# 点时间，避免四个线程一起空转白烧四个核。
_PRECREATE_FLIGHT_SLEEP_S = 0.002
# 需求驱动增长的默认就绪余量（槽位）。写路径永不等预建，余量的作用只是把
# "增长跟不上需求"的概率压低：按 44ms/槽的单线程速度，1024 槽约 45s 的追赶
# 余量，足够吸收突发。
_MIN_PRECREATE_HEADROOM_SLOTS = 1024

# 磁盘空间护栏：预建线程在把盘写满之前必须停手。故障形态（2026-09-22 事故）：
# 容量配置超过物理盘（8TiB/rank × 8 rank = 64TiB 需求 > 4×5.8TB 盘），预建
# 线程一路补到 ENOSPC——既写坏池子（半截文件），又让写路径退回"自己建槽"
# 慢路径，前向线程被拖垮。护栏把增长压回"可用空间之内"：空间不足时暂停
# 增长（写路径继续按非阻塞裁剪契约拒绝缺槽写），空间恢复后自动继续。
# 门限可用 TUTTI_PRECREATE_MIN_FREE_BYTES 覆盖（测试用）。
_PRECREATE_MIN_FREE_BYTES_ENV = "TUTTI_PRECREATE_MIN_FREE_BYTES"
_PRECREATE_DEFAULT_MIN_FREE_BYTES = 32 * 1024 ** 3  # 32 GiB
_PRECREATE_SPACE_RECHECK_S = 5.0
# CAUGHT_UP 是多线程共享的信号（每个线程各打一行会形成风暴），节流到这条
# 间隔内只由最先追平的线程播报一次。
_PRECREATE_CAUGHT_UP_LOG_INTERVAL_S = 10.0

from tutti.index.chunk_index import decode_io_key as _decode
from tutti.storage.object_store import (
    ObjectPlacement,
    ObjectStore,
    SCHEME_LOCAL_NVME_FILE,
    SCHEME_STRIPED_NVME_FILE,
)

__all__ = ["ObjectLayout"]

class ObjectLayout:
    """由对象层驱动的布局。

    ``mounts`` 是数据盘目录；一个 slot 是一个文件，slot 号在 mounts 间
    轮转（单 mount 即单文件布局）。``devices`` 是给 runtime resolver 用的
    设备事实（controller/namespace），对象层只用其中的 ``mount_path``。
    """

    def __init__(
        self,
        root: str | os.PathLike,
        segment_bytes: int,
        *,
        mounts=None,
        devices=None,
        capacity_chunks: int = 0,
        prewarm_chunks: int = 0,
        rank_id: int = 0,
        rank_count: int = 1,
        background_reclaim: bool = True,
        warmup_probe_only: bool = False,
        namespace: bytes | str | None = None,
    ):
        self._root = Path(root)
        self._segment_bytes = int(segment_bytes)
        self._mounts = [str(m) for m in (mounts if mounts else [self._root])]
        self._devices = list(devices or [])
        # 容量/预热按 chunk 计（对象层按 slot_bytes 折算为字节），避免上层
        # 重复推导对象几何——对象头 4096B 由本模块在打开时加进去。
        self._capacity_chunks = int(capacity_chunks or 0)
        self._prewarm_chunks = int(prewarm_chunks or 0)
        self._rank_id = int(rank_id)
        self._rank_count = int(rank_count)
        self._background_reclaim = bool(background_reclaim)
        # open 期预热口径：True = 只走查已存在的槽位、不建缺失的（配后台预建）；
        # False = 旧同步语义（把缺失的预热槽位建出来）。
        self._warmup_probe_only = bool(warmup_probe_only)
        # 后台预建线程的停止信号（None = 未启动），见 start_background_precreate。
        self._precreate_stop: threading.Event | None = None
        # CAUGHT_UP 日志节流（多线程共享一个时间戳，见 _PRECREATE_CAUGHT_UP_LOG_INTERVAL_S）。
        self._precreate_caught_up_log_at = 0.0
        self._namespace = (
            namespace.encode("utf-8") if isinstance(namespace, str) else namespace
        )

        self._layer_span: int | None = None
        self._store: ObjectStore | None = None
        self._committed: set[bytes] = set()
        self._reserved: dict[bytes, ObjectPlacement] = {}
        self._written_layers: dict[bytes, set[int]] = {}

    # ---------- 几何 / 生命周期 ----------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def mounts(self) -> tuple[str, ...]:
        """数据盘挂载点（单文件布局时即 root 本身）。"""
        return tuple(self._mounts)

    @property
    def segment_bytes(self) -> int:
        return self._segment_bytes

    @property
    def layer_span(self) -> int | None:
        return self._layer_span

    def set_namespace(self, namespace) -> None:
        """声明 key 命名空间（打开前生效）；不一致时对象层 fail-closed。"""
        if self._store is not None:
            raise RuntimeError("命名空间须在打开对象层之前声明")
        self._namespace = (
            namespace.encode("utf-8") if isinstance(namespace, str) else namespace
        )

    def set_layer_span(self, num_layers: int) -> None:
        """定层宽并打开对象层（此后才可能做冷启动恢复）。"""
        num_layers = int(num_layers)
        if num_layers <= 0:
            raise ValueError(f"层数须为正整数，got {num_layers}")
        if self._layer_span is not None and self._layer_span != num_layers:
            raise ValueError(
                f"层宽已定案 {self._layer_span}，不能改为 {num_layers}"
            )
        self._layer_span = num_layers
        if self._store is None:
            self._open_store()

    def _config(self) -> dict:
        assert self._layer_span is not None
        devices = []
        if self._devices:
            for index, device in enumerate(self._devices):
                entry = dict(device)
                entry.setdefault(
                    "mount_path",
                    self._mounts[index] if index < len(self._mounts) else self._mounts[0],
                )
                devices.append(entry)
        else:
            devices = [{"mount_path": mount} for mount in self._mounts]
        prewarm_chunks = min(self._prewarm_chunks, self._capacity_chunks)
        return {
            "scheme": (
                SCHEME_STRIPED_NVME_FILE
                if len(self._mounts) > 1 else SCHEME_LOCAL_NVME_FILE
            ),
            "uri": str(self._root),
            # 容量/预热按槽位声明：每槽位字节数由对象层按几何算。
            "capacity_slots": self._capacity_chunks,
            "prewarm_slots": prewarm_chunks,
            "warmup_probe_only": self._warmup_probe_only,
            "segment_bytes": self._segment_bytes,
            "segment_count": self._layer_span,
            "namespace_fingerprint": self._namespace or b"",
            "devices": devices,
            "background_reclaim": self._background_reclaim,
            "rank_id": self._rank_id,
            "rank_count": self._rank_count,
        }

    def _open_store(self) -> None:
        store = ObjectStore(self._config())
        # 命名空间/几何不一致时对象层 fail-closed（不覆盖旧数据），异常直接上抛：
        # 异构池必须由人处理，不能被当成"空池"静默复用。
        store.open()
        self._store = store
        self._committed = store.recover()

    def ensure_dirs(self) -> None:
        """对象层自行物化目录与槽位文件，这里无事可做。"""

    def close_object_pool(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
        self._committed.clear()
        self._reserved.clear()
        self._written_layers.clear()

    def attach_object_pool(self, pool) -> None:
        """兼容旧装配签名：对象布局不需要外部池。"""

    def check_namespace(self, namespace) -> bool:
        """命名空间校验已下沉到对象层（open 时 fail-closed）。"""
        return True

    # ---------- 恢复 ----------

    def ready_slots(self) -> list[tuple[str, int]]:
        """已物化槽位的 ``(uri, generation)`` 列表（绑定期预热用）。

        冷池与热池都非空：对象层在 open() 时已把预热槽位物化，恢复出的槽位
        同样在盘上。绑定期拿这些 URI 先把运行时票据开好，请求路径就只剩内存
        查找——首轮 resolve + peer-memory 注册的几百毫秒因此移出请求路径。
        """
        store = self._store
        if store is None:
            return []
        out: list[tuple[str, int]] = []
        for slot in range(store.ready_slots()):
            uri = store.slot_uri(slot)
            if uri:
                out.append((uri, store.slot_generation(slot)))
        return out

    @property
    def slot_payload_bytes(self) -> int:
        """每个对象的 payload 字节数（各槽位一致，用于预热票据的尺寸校验）。"""
        return self._segment_bytes * int(self._layer_span or 0)

    # ---------- 后台预建（大容量冷启动与按需增长都不在关键路径上）----------

    def _mounts_free_bytes(self) -> int | None:
        """所有数据盘挂载点的可用空间合计；任一 statvfs 失败返回 None。

        None 表示护栏失效（调用方按"不拦"处理）：护栏是防呆，不是正确性前提，
        探测不到空间时不该把预建卡死。
        """
        total = 0
        for mount in self._mounts:
            try:
                st = os.statvfs(mount)
            except OSError:
                return None
            total += st.f_bavail * st.f_frsize
        return total

    def _min_free_bytes(self) -> int:
        """护栏门限：默认 min(32 GiB, 挂载点总容量的 5%)。

        绝对门限保护大盘（32 GiB ≈ 5.8 TB 盘的 0.55%，足够写路径回旋）；
        百分比上限保护小盘与测试环境（20 GiB 的盘不该因为凑不够 32 GiB 而
        永远不能预建）。``TUTTI_PRECREATE_MIN_FREE_BYTES`` 直接覆盖门限
        （测试用；设 0 表示关闭护栏）。
        """
        raw = os.environ.get(_PRECREATE_MIN_FREE_BYTES_ENV)
        if raw is not None:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass
        total = 0
        for mount in self._mounts:
            try:
                st = os.statvfs(mount)
            except OSError:
                total = 0
                break
            total += st.f_blocks * st.f_frsize
        if total <= 0:
            return _PRECREATE_DEFAULT_MIN_FREE_BYTES
        return min(_PRECREATE_DEFAULT_MIN_FREE_BYTES, total * 5 // 100)

    def start_background_precreate(self, threads: int = 4,
                                   headroom: int = 0) -> bool:
        """后台预建槽位（只补就绪余量，跟随分配前沿），写路径不等它。

        术语（与 SPI 头部一致）：**预热**是 open 期同步做掉的那一份；
        **预建**是把槽位文件建成"尺寸正确、内容为零"的可用状态（~44ms/槽）；
        **就绪**表示已预建、可被分配。

        与"复用"的关系（决定这批文件的来历，也决定这里要不要干活）：
          * 容量 ≤ 盘上已有的槽位 → open 的走查已把它们记成就绪，本方法
            不动任何文件（复用立即生效，零 IO、零异步）；
          * 容量 > 盘上已有的槽位 → 差量在这里异步补，但**只为分配前沿
            之前的一段就绪余量**铺文件：容量是用户指定的上限，不是要立刻
            实体化的目标。已经存在的那部分由 C++ 的幂等探测跳过，所以
            "复用"与"增长"合在同一条前沿上，不重复劳动。
        就绪余量 headroom 的语义：增长总是从低到高、跟随分配前沿，所以先
        满足的正是即将被分配到的槽位。曾有一个"一路补到容量上限"的模式
        （full_capacity），被删除：盘占用 ∝ 容量而非工作集、几十万次零写
        与在线 KV 争带宽，且容量配错时会把盘写满（2026-09-22 事故）。

        为什么必须异步：预建一个槽位是 create + 写实零 + fsync（~44ms/槽，见
        space_allocator 的注释），放在哪一端都是灾难——
          * 放在 open 期：10 TB 级容量要几十分钟才起得来；
          * 放在写路径（C++ reserve 的按需预建）：每个新槽位把前向线程卡
            44ms，正好打在 GPU 计算/IO 流水线上。

        因此契约拆成两半（见 storage_object_store.h 的异步增长一节）：
          * open 只**走查**（probe，不写）调用方声明的初始槽位，把已经存在
            的那些记成就绪——复用不花 IO，也不进异步；
          * 本方法打开 ``set_precreate_on_write(False)``——写路径拿不到
            就绪槽位时**拒绝并要求调用方裁剪**，与容量耗尽同一契约，且
            自愈：后台追上后同一笔写就会成功；
          * 增长由这里的后台线程驱动，容量上限仍然生效。

        失败自愈：后台出错时立刻把按需预建交还写路径——宁慢，不静默停摆
        （池子停止增长是看不见的容量故障）。
        幂等：可安全重复启动。
        返回 True 表示已启动（或无需启动）。
        """
        if self._precreate_stop is not None:
            return True
        store = self._store
        step = getattr(store, "precreate_step", None)
        if not callable(step):
            # 原生扩展还没有异步预建接口：维持写路径按需预建（旧行为）。
            return False
        capacity = int(self._capacity_chunks)
        # keep = 分配前沿之前要保持就绪的槽位数。容量只是上限，不在这里
        # 铺实体（见方法头注释：full_capacity 已删除）。
        keep = max(1, int(headroom or _MIN_PRECREATE_HEADROOM_SLOTS))
        try:
            store.set_precreate_on_write(False)
        except Exception as exc:
            _PRECREATE_LOG.warning(
                "BACKGROUND_PRECREATE_DISABLED err=%r", exc
            )
            return False
        stop = threading.Event()
        self._precreate_stop = stop

        def run(index: int) -> None:
            worked = False
            space_low = False
            while not stop.is_set():
                # 磁盘空间护栏（见模块头常量注释）：写路径永不因它阻塞——
                # 空间不足时这里只是停止增长，缺槽写仍走非阻塞裁剪契约。
                free = self._mounts_free_bytes()
                floor = self._min_free_bytes()
                if free is not None and free < floor:
                    if not space_low:
                        space_low = True
                        _PRECREATE_LOG.warning(
                            "BACKGROUND_PRECREATE_SPACE_LOW free=%dB floor=%dB "
                            "precreated=%d target=%d；暂停增长，空间恢复后继续",
                            free, floor, store.precreated_slots(),
                            store.precreate_target(),
                        )
                    stop.wait(_PRECREATE_SPACE_RECHECK_S)
                    continue
                if space_low:
                    space_low = False
                    _PRECREATE_LOG.info(
                        "BACKGROUND_PRECREATE_SPACE_RESUMED free=%dB floor=%dB",
                        free, floor,
                    )
                try:
                    made = step(_PRECREATE_BATCH_SLOTS, keep)
                except Exception as exc:
                    _PRECREATE_LOG.warning(
                        "BACKGROUND_PRECREATE_FAILED thread=%d err=%r；"
                        "把按需预建交还写路径",
                        index, exc,
                    )
                    stop.set()
                    try:
                        store.set_precreate_on_write(True)
                    except Exception:
                        pass
                    return
                if made > 0:
                    # 有活就接着干，不在批次之间空等——追赶分配前沿时
                    # 这条循环是热路径，睡眠会把它拖慢到 1/500。
                    worked = True
                    continue
                # 返回 0 有两种可能：真无事可做（已到 target），或这一次的
                # 认领都还在飞（别人刚认领走了）。只有前者该让出 CPU，
                # 后者必须立刻重试，否则多线程会退化成 2 次/秒。
                if store.precreated_slots() < store.precreate_target():
                    # 让出极短时间再重试：这个状态最长持续一次单槽 IO
                    # （~44ms），但四个线程一起空转会白烧四个核。
                    stop.wait(_PRECREATE_FLIGHT_SLEEP_S)
                    continue
                if worked:
                    # 从"有活"回到"无事可做"：这是外部唯一能观测到预建
                    # 已补齐的信号（线程名不进 /proc，CPU 也测不出来）。
                    # 四个线程会各自经历一次同样的转变，节流成一条。
                    worked = False
                    now = time.monotonic()
                    if (now - self._precreate_caught_up_log_at
                            >= _PRECREATE_CAUGHT_UP_LOG_INTERVAL_S):
                        self._precreate_caught_up_log_at = now
                        _PRECREATE_LOG.info(
                            "BACKGROUND_PRECREATE_CAUGHT_UP precreated=%d "
                            "target=%d capacity=%d",
                            store.precreated_slots(), store.precreate_target(),
                            capacity,
                        )
                # 没有可做的工作（分配前沿还没推过来，或已到容量上限）：
                # 让出 CPU，等下一次需求把它叫醒。
                stop.wait(_PRECREATE_IDLE_SLEEP_S)

        workers = max(1, min(int(threads), 32))
        for index in range(workers):
            threading.Thread(
                target=run, args=(index,),
                name=f"tutti-precreate-{index}", daemon=True,
            ).start()
        _PRECREATE_LOG.info(
            "BACKGROUND_PRECREATE_START reused=%d capacity=%d "
            "headroom=%d threads=%d batch=%d",
            store.precreated_slots(), capacity,
            keep, workers, _PRECREATE_BATCH_SLOTS,
        )
        return True

    def stop_background_precreate(self) -> None:
        """请求后台预建停止（close 时调用；线程是 daemon，不阻塞退出）。

        同时把按需预建交还写路径：增长线程没了，池子若还在被写入，就必须
        退回"自己建槽"的慢路径，而不是永远拒绝。
        """
        if self._precreate_stop is not None:
            self._precreate_stop.set()
            self._precreate_stop = None
        restore = getattr(self._store, "set_precreate_on_write", None)
        if callable(restore):
            try:
                restore(True)
            except Exception:
                pass

    def committed_chunks(self) -> set[bytes]:
        """已提交（= 层齐全）的 chunk 集合；层宽未定案时为空集。"""
        return set(self._committed)

    def checkpoint(self) -> None:
        """把对象层内存状态（已提交 key 集合）落到检查点。

        close() 会自动落盘；显式调用用于"希望下一次冷启动立刻看到"的时点
        （例如回收/驱逐之后）。
        """
        if self._store is not None:
            self._store.checkpoint()

    def scan(self) -> set[bytes]:
        """已提交对象的 io_key 集合。

        对象有效 ⇔ 全部段都写过，因此每个已提交 chunk 展开成 layer_span
        个 io_key——与旧实现"全层标记齐备"的判定等价。
        """
        span = self._layer_span or 0
        keys: set[bytes] = set()
        for chunk in self._committed:
            if len(chunk) == 16:
                # 标准 chunk：对象有效 ⇒ 全部层段都可读
                keys.update(
                    chunk + layer.to_bytes(2, "little")
                    for layer in range(span)
                )
            else:
                # 通用短 key（通用 KV 契约）：key 本身没有层后缀，形态即原样
                keys.add(chunk)
        return keys

    # ---------- 写路径 ----------

    def prepare_put(self, io_keys, capacity_chunks: int):
        """预留对象，返回 ``(admitted, rejected_count)``。

        ``admitted`` 形如 ``{io_key: (chunk, layer)}``，只含**受理**的 key；
        容量耗尽是稳态而非故障（对象层契约：部分受理、绝不阻塞），因此这里
        不抛错——调用方按返回的集合裁剪本批写入即可，缓存未命中不该让产生它
        的请求失败。
        """
        decoded = {}
        chunks: list[bytes] = []
        for io_key in io_keys:
            chunk_id, layer = _decode(io_key)
            decoded[bytes(io_key)] = (chunk_id, layer)
            if chunk_id not in chunks:
                chunks.append(chunk_id)
        placements, rejected = self._store_required().reserve(chunks)
        self._reserved.update(placements)
        admitted = {
            io_key: value for io_key, value in decoded.items()
            if value[0] in placements
        }
        return admitted, int(rejected)

    def commit_layers(self, io_keys) -> set[bytes]:
        """逐层记账；某 chunk 的层写齐后按对象提交一次。

        返回**本次新提交**的 chunk 集合——调用方据此把"对象有效"的那批
        层记为驻留（对象有效 ⇔ 全部段都写过）。
        """
        pending: dict[bytes, set[int]] = {}
        for io_key in io_keys:
            chunk_id, layer = _decode(io_key)
            pending.setdefault(chunk_id, set()).add(layer)
        ready: list[bytes] = []
        span = self._layer_span
        for chunk_id, layers in pending.items():
            written = self._written_layers.setdefault(chunk_id, set())
            written |= layers
            # 层宽未知（未 set_layer_span）时不允许提交：全对象语义要求
            # "所有段都写过"才有效，无法判定就必须等。
            if span is not None and len(written) >= span:
                ready.append(chunk_id)
        if not ready:
            return set()
        self._store_required().commit(ready)
        for chunk_id in ready:
            self._committed.add(chunk_id)
            self._written_layers.pop(chunk_id, None)
        return set(ready)

    def abort_uncommitted(self, chunk_ids) -> None:
        """丢弃未提交的预留（已提交的对象不受影响）。"""
        pending = [
            chunk for chunk in dict.fromkeys(bytes(c) for c in chunk_ids)
            if chunk not in self._committed
        ]
        if pending:
            self._store_required().abort(pending)
        for chunk in pending:
            self._reserved.pop(chunk, None)
            self._written_layers.pop(chunk, None)

    def release_chunks(self, chunk_ids) -> int:
        chunks = list(dict.fromkeys(bytes(c) for c in chunk_ids))
        released = self._store_required().release(chunks)
        for chunk in chunks:
            self._committed.discard(chunk)
            self._reserved.pop(chunk, None)
            self._written_layers.pop(chunk, None)
        return released

    def drop(self, io_keys) -> None:
        self.release_chunks(_decode(c)[0] for c in io_keys)

    def releasable_chunks(self, io_keys) -> set[bytes]:
        """这些 key 里可回收的 chunk：只回收已提交（层齐全）的对象。"""
        return {
            chunk for chunk in (_decode(c)[0] for c in io_keys)
            if chunk in self._committed
        }

    # ---------- 定位 ----------

    def is_committed(self, chunk_id: bytes) -> bool:
        """对象是否有效（全部段都写过）。半截对象一律不可读。"""
        return bytes(chunk_id) in self._committed

    def _placement(self, chunk_id: bytes) -> ObjectPlacement | None:
        placement = self._reserved.get(chunk_id)
        if placement is not None:
            return placement
        store = self._store
        if store is None:
            return None
        placement = store.placement(bytes(chunk_id))
        if placement is not None:
            self._reserved[chunk_id] = placement
        return placement

    def target_uri(self, chunk_id: bytes) -> str:
        placement = self._placement(chunk_id)
        if placement is None:
            raise KeyError(f"chunk 未预留：{bytes(chunk_id)!r}")
        return placement.uri

    def reserved_uris(self, chunk_ids) -> list[str]:
        """已预留 chunk 的 URI 列表；未预留的静默跳过（保序去重）。

        给清理路径（Store::abort_chunks）用：一个 chunk 可能已被驱逐/回收，
        此时它不在预留表里、也没有可清理的缓存目标——这不是异常。target_uri
        与 target_offset 保持 fail-fast，供真正要求"必须已预留"的调用方用。
        """
        uris = []
        for chunk_id in dict.fromkeys(bytes(c) for c in chunk_ids):
            placement = self._placement(chunk_id)
            if placement is not None:
                uris.append(placement.uri)
        return uris

    def target_offset(self, chunk_id: bytes) -> int:
        """段 0 在对象逻辑地址空间中的起点（对象头之后）。"""
        placement = self._placement(chunk_id)
        if placement is None:
            raise KeyError(f"chunk 未预留：{bytes(chunk_id)!r}")
        return int(placement.offset)

    def target_size(self, chunk_id: bytes) -> int:
        placement = self._placement(chunk_id)
        return int(placement.payload_bytes) if placement is not None else 0

    def target_generation(self, chunk_id: bytes) -> int:
        placement = self._placement(chunk_id)
        return int(placement.generation) if placement is not None else 0

    def object_pool_snapshot(self) -> dict | None:
        if self._store is None:
            return None
        snapshot = self._store.usage()
        snapshot["committed_chunks"] = len(self._committed)
        return snapshot

    # ---------- 内部 ----------

    def _store_required(self) -> ObjectStore:
        if self._store is None:
            raise RuntimeError(
                "对象层未打开：先 set_layer_span(num_layers)（层宽定案后才能定"
                "义对象几何）"
            )
        return self._store




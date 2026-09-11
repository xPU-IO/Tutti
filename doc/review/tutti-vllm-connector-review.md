# Tutti vLLM Connector 实现 Review

> 范围：对比 legacy geminifs 接入（`third_pkgs/tutti-legacy/`）与新版 Tutti connector
> （`integration/vllm-connector/`），并对照最新 vLLM（`third_pkgs/vllm/`）的
> `KVConnectorBase_V1` 契约，评估设计合理性与缺失项。
>
> 基线：commit `8ffd81f`（vLLM KV connector baseline）。
>
> 结论先行：**两代实现的结构高度同源、逻辑大体一致，方向正确；但当前实现存在
> 若干阻断性问题（跨进程索引不同步、metadata 不可序列化、缺 CUDA graph 适配、
> staging 落地为 host 内存导致 GPU-direct 倒退），尚不能直接跑通真实 vLLM。**

---

## 1. 两代实现逻辑一致性对比

结论：**结构同源、语义一致**。新实现是 legacy 的"四层结构重排 + staging 解耦"，
不是推倒重来。下表逐项核对：

| 维度 | legacy（geminifs） | 新 Tutti | 一致性 |
|---|---|---|---|
| 存储文件 | 1 chunk = 1 GPUFile，每盘 4 物理文件条带化，64KB 对齐；`GPUFileDesc{file_id, block_size, tensor_shape, CompactNVMeMapping}` GPU-resident | 1 chunk = 1 文件（`chunk://local/<i>`），`LocalStoreBackend` 一 chunk 一文件，层段 offset = `layer_idx × segment_bytes` packed | ✅ 同为"chunk=文件"；条带化改由 `striped://` resolver 承担（backend 私有） |
| 索引 | 逻辑索引（CPU dict）+ 物理索引（GPU CompactNVMeMapping）；LRU evictor | `ChunkIndex`（链式 blake2b 哈希 key→path + LRU + pin + pending_store） | ✅ 逻辑索引对齐；物理索引（PRP/FIEMAP）下沉到 runtime |
| 哈希体系 | `CacheEngineKey.chunk_hash` | 父链式 `H_i = blake2b(H_{i-1} ‖ tokens_i)`（D-007） | ✅ 同构（链式滚动前缀指纹） |
| 命中查询 | `lookup_client.lookup()` 跨进程 RPC（ZMQ/Redis lookup server） | `engine.lookup_prefix()` **进程内** | ❌ **关键差异，见 C1** |
| 分配 | `gpu_open_file()` → GPUFileMetadata + LRU evictor | `plan_store()` → `ChunkIndex.allocate()`（LRU 驱逐 + pin 保护） | ✅ 对齐 |
| 读写调度 | worker 逐层回调：`start_load_kv` 预取第 0 层，`wait_for_layer_load(L)` 发 L+1，`save_kv_layer(L)` 落盘 | `WorkerImpl` 同样逐层编排 + 波次切分 + 在途句柄背压 | ✅ 对齐 |
| 数据搬运 | **GPU-direct**：`register_tensor_with_gpu()` 直接注册 paged KV 池给 DMA，`geminifs_batched_read/write` GPU 端直接读写盘 | staging（`register_memory`）→ 盘，再 `scatter/gather` kernel 搬 staging↔paged | ⚠️ **见 C4/C5** |
| overlap | StreamController（green context SM 分区：load 16SM / save 8SM），逐层 prefetch 重叠 | 逐层预取 + 环形窗口 + 背压；`io_stream ∥ compute` | ✅ 设计对齐，但真接线未完成 |
| debug/nsys | `_infinikv_nvtx_annotate`（domain="infinikv"，函数名 hash 分色）+ `nvtx.start_range` | 设计有 `nvtx_utils.py`，**代码缺失** | ❌ **见 M5** |

**结论**：逻辑一致，legacy 的验证性资产（逐层回调、slot_mapping、RequestTracker、
chunk 边界对齐、skip_leading_tokens、full-hit 减 1 等）都被忠实平移了。差异集中在
两个点：**跨进程索引同步被删掉了**（C1），**直接 GPU-direct 变成了 staging 中转**
（C4/C5）。

---

## 2. staging buffer 设计（新引入的核心差异）

用户说明："tutti 先读到 staging buffer，然后通过 lmcache 类似的 load and reshape
kernel 拷贝到 GPU 内部"。

这一设计的**动机是正确的**：

- legacy 的 block_size 必须等于 chunk_size（`init_infinikv_engine` 里
  `chunk_size = cache_config.block_size` 强制对齐），无法适配 vLLM 动态 block size；
- staging + `single_layer_kv_transfer`（`slot_mapping` 驱动的 paged↔packed 重排）
  正是 LMCache 验证过的"块大小无关"方案，`mem_kernels.cu` 已平移到位。

但当前落地有三个问题（C4/C5/M1 详述）：

1. staging 用 `ctypes.create_string_buffer`（**pageable host 内存**），不是 GPU/pinned
   内存——这与设计文档 §5.4"数据进出只经 gather/scatter kernel（HBM 域 μs 级）"
   相矛盾，且丢失了 legacy 的 GPU-direct 优势；
2. gather/scatter 仍是 `_noop_*` 桩（只记录调用），真 kernel 未接线；
3. `mem_kernels.cu::get_kernel_ptr` 对 CPU tensor 要求 pinned（`cudaHostGetDevicePointer`），
   但 `create_string_buffer` 不是 pinned，直接接线会失败。

---

## 3. 阻断性问题（Critical）

### C1 — 跨进程索引不同步，load 路径整体失效（最严重）

legacy 的关键机制是 **lookup server**：

- worker 的 `GeminiFSBackend.dict` 持有真实索引，起 `LookupServerInterface` 对外服务；
- scheduler 的 `get_num_new_matched_tokens` 通过 `LookupClientFactory.create_lookup_client`
  的 `lookup()` **RPC 查询 worker 索引**；
- `insert_key/remove/close` 同步维护（`KVAdmitMsg/KVEvictMsg`、`batched_remove`）。

新 Tutti 把它改成了**进程内查询**：

```python
# connector.py:438（scheduler 角色）
num_external_hit_tokens = self.engine.lookup_prefix(token_ids)
```

但 vLLM v1 里 **scheduler 与 worker 是两个独立进程**（`KVConnectorRole.SCHEDULER` /
`WORKER` 的分离正是为此）。scheduler 侧 `TuttiEngine` 的 `ChunkIndex` 与 worker 侧的
`ChunkIndex` 是两份独立内存：

- 真正落盘在 worker：`store_layer → complete_store` 只更新 **worker 的** `ChunkIndex.stored`；
- scheduler 侧 `ChunkIndex.stored` **永远为空**：
  - `lookup_prefix` 只查 `self.stored`，永不命中 → `get_num_new_matched_tokens` 恒返回 0；
  - `plan_load` 对空索引 `pin()` 会抛 `KeyError` → 即便命中也无法 plan。

**后果**：外部 KV 命中（load）完全失效，只剩 save（best-effort 写盘），且 scheduler
永远不知道哪些 chunk 已落盘，LRU 决策与 pin 保护全部失效。

设计文档 §4.4 明确写"`get_num_new_matched_tokens → engine.lookup_prefix（进程内，无 RPC）`"，
这是**架构性错误**——除非部署形态是单进程共享（当前 vLLM 不支持 connector 跨进程共享内存）。

> 修复方向：恢复一个跨进程索引同步层（最小方案：scheduler 侧不查本地，改为
> RPC/共享内存/文件 sidecar 读 worker 索引；或把"命中查询"整体移到 worker 侧由
> worker 汇报）。legacy 的 lookup server 是可复用的参考实现。

### C2 — metadata 无法序列化（scheduler→worker IPC 会失败）

`TuttiConnectorMetadata` 是 **普通 class**（connector.py:314-317），既不是
`@dataclass` 也不是 `KVConnectorMetadata` 子类：

```python
class TuttiConnectorMetadata:   # ❌ 非 dataclass、非 KVConnectorMetadata 子类
    def __init__(self) -> None:
        self.requests: list[ReqMeta] = []
```

而新 vLLM 的 `SchedulerOutput.kv_connector_metadata: KVConnectorMetadata | None`
（vllm/v1/core/sched/output.py:255）经 `MsgpackEncoder` 序列化跨进程传输。
`vllm/v1/serial_utils.py` 对未知对象直接 `raise TypeError`（除非设
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` 回退 pickle）。因此：

1. **`TuttiConnectorMetadata` 本身**：msgspec 不认识普通 class → 序列化抛异常。
2. **`ReqMeta` 字段含 `torch.Tensor`**（`token_ids/block_ids/slot_mapping`，
   connector.py:173-177）：即便改成 dataclass，tensor 字段会走 `enc_hook` 的
   `_encode_tensor`（返回 `(dtype, shape, data)` 三元组），decode 侧未必能正确重建。

**对比 LMCache 的做法**：`LMCacheConnectorMetadata` 是
`@dataclass class LMCacheConnectorMetadata(KVConnectorMetadata)`，且 `ReqMeta.token_ids`
被**刻意改成了 `list[int]`**（源码注释 `# torch.Tensor` 还留着），正是为了规避
tensor 进 metadata 的序列化问题。

> 修复方向：`TuttiConnectorMetadata` 改为 `@dataclass(KVConnectorMetadata)`；
> `ReqMeta` 中 `token_ids/block_ids/slot_mapping` 改为 `list[int]`（worker 侧再
> `torch.tensor()` 化），或提供显式 `enc_hook/dec_hook`。

### C3 — 缺少 `requires_piecewise_for_cudagraph` 覆盖

新 vLLM `base.py:630-649` 提供 `requires_piecewise_for_cudagraph` classmethod，
文档明确：**使用逐层异步回调（`wait_for_layer_load`/`save_kv_layer`）的 connector
必须覆盖返回 True**，否则 CUDA graph 捕获/回放时会跳过这些回调造成数据竞争。

legacy fork 当年就加了这一项（设计文档 §1.2 明确列出 `requires_piecewise_for_cudagraph`）。
新 Tutti 的 `TuttiConnectorV1` **没有覆盖**（全仓搜索无此方法），默认返回 False。
在开启 CUDA graph（默认生产配置）时，逐层 load/store 会被错误捕获。

> 修复：覆盖 `requires_piecewise_for_cudagraph` 返回 True。

### C4 — staging 落地为 host 内存，GPU-direct 优势丢失

设计文档 §2b.2 与 §5.4 描述的目标是 GPU-direct（"io_stream 下发 IO kernel，GPU 自闭环"、
"vllm paged 池不注册，数据进出只经 gather/scatter kernel，HBM 域 μs 级"）。

但当前实现：

```python
# core.py:316
staging = ctypes.create_string_buffer(num_slots * segment_bytes)  # pageable host 内存
...
# local_store_backend.py
self._mem_ticket = self._runtime.register_memory(staging_addr, ..., "host", ...)
```

staging 是 **host 内存**。若按此接线，数据路径变成：

```
NVMe →(DMA)→ host staging →(GPU kernel 读 host)→ paged GPU
```

即 **host bounce**：相比 legacy 的 `register_tensor_with_gpu`（paged GPU 池直接 DMA），
多了一次 PCIe 往返，有效带宽减半。这与"最大化 KV 性能、GPU-direct"的目标相悖。

> 需要澄清：staging 应当落在 **GPU 显存（或 pinned host 内存 + runtime 的 DEVICE
> 注册）**，gather/scatter 是 GPU↔GPU（HBM 域）kernel，才保留 GPU-direct 收益。
> 若 staging 必须放 host，需在文档里明确接受该带宽代价并重新测算。

### C5 — gather/scatter 是 noop 桩，端到端搬运尚未接通

`core.py:113-118` 的 `_noop_gather/_noop_scatter` 只 `append` 一条日志；默认
`gather_fn/scatter_fn` 即这两个桩。`mem_kernels.cu` 的 `single_layer_kv_transfer`
虽已平移（T-117），但**未作为 gather/scatter 钩子注入**。

当前端到端：staging 与 paged GPU 之间**没有真正的数据拷贝**，load/store 只在
backend 与 staging 之间搬运字节。T-117 的"集成接线"是未完成项。

此外，`single_layer_kv_transfer` 需要 `slot_mapping` 参数（token→slot 映射），而
当前 `WorkerImpl._chunk_first_blocks` 只从 `block_ids` 算首块号、没有构建
`slot_mapping` 传给 kernel——接线时需补齐这段。

---

## 4. 兼容性问题（Major）

### M1 — `chunk_tokens % block_size == 0` 约束仍在

`worker.py` 强制 `chunk_tokens % block_size == 0`。这并未完全实现"兼容
vLLM 动态 block size"的目标——虽然 chunk(256) 通常是 block(16/32/128) 的整数倍，
但严格说 staging 解耦的价值正是"chunk 与 block 完全无关"。当前仍耦合。

> 评估：多数场景可接受（整数倍约束很弱），但若要真正支持任意动态 block，需让
> gather/scatter 处理 chunk 尾部跨 block 边界的情况，并放开该断言。

### M2 — `update_state_after_alloc` 忽略 `blocks` 参数

`connector.py` 注释称"blocks（新版 KVCacheBlocks）暂不使用——CoW pending
copy 语义由 worker 侧接线时处理"。新 vLLM 里 `blocks` 是 `KVCacheBlocks`（CoW
copy-on-write 语义），预占/CoW 场景下忽略它可能导致块号不一致。需在 T-114 接线时
补上 CoW 处理。

### M3 — 未覆盖 `handle_preemptions` / `prefer_cross_layer_blocks`

- `handle_preemptions`（base.py:306）：预占请求在被覆盖前需保存 KV。legacy 有
  `withdraw_tasks` 回收未落盘任务。新实现只在 `build_connector_meta` 处理了
  `resumed_req_ids` 的恢复，worker 侧无预占保存——best-effort 可接受，但会造成
  预占时未落盘 KV 丢失（下次 miss）。
- `prefer_cross_layer_blocks`（base.py:185）：legacy 用单一 cross-layer tensor 注册
  （`register_cross_layers_kv_cache`，规避 per-layer DMA 注册破坏 CUDA 内存池）。
  新实现用 per-layer `register_kv_caches`。因 staging 是唯一注册对象，本可接受，
  但需确认与 gather/scatter 的 per-layer paged 布局对齐。

### M4 — 索引不持久、无并发去重、无容量上限联动

- 设计文档 §5.3 明确"索引不持久（R1 接受重启冷缓存）"——可接受，但需在
  review 层面记录为已知限制。
- `ChunkIndex` 无跨进程锁（见 C1），单进程内无锁直接操作 `OrderedDict`，若未来
  scheduler 侧多线程并发查/写需加锁。

### M5 — nsys/NVTX 标记缺失

legacy 的 `_infinikv_nvtx_annotate`（`utils.py`，domain="infinikv"，函数名
hash 分色）铺满 adapter/engine，配合 `nvtx.start_range`。新设计文档 §4.1 列了
`adapter/nvtx_utils.py`，但**该文件不存在**，`connector.py/worker.py` 也无任何
nvtx 装饰器。debug 手段从 legacy 到新实现是**缺失**的，需补齐（`nvtx_utils.py`
+ 关键回调/engine 方法装饰）。

---

## 5. 缺失清单（还缺哪些）

按优先级排序：

| # | 缺失项 | 对应 | 严重度 |
|---|---|---|---|
| 1 | 跨进程索引同步（lookup server / RPC / 共享内存） | C1 | 🔴 阻断 |
| 2 | metadata 可序列化改造（dataclass + list 字段） | C2 | 🔴 阻断 |
| 3 | `requires_piecewise_for_cudagraph` = True | C3 | 🔴 阻断 |
| 4 | staging 落到 GPU/pinned + runtime DEVICE 注册 | C4 | 🔴 阻断 |
| 5 | gather/scatter 真接线（含 slot_mapping 构建） | C5/M1 | 🔴 阻断 |
| 6 | `nvtx_utils.py` + NVTX 装饰 | M5 | 🟡 高 |
| 7 | `handle_preemptions`（预占保存） | M3 | 🟡 高 |
| 8 | CoW（`update_state_after_alloc` 的 blocks） | M2 | 🟡 高 |
| 9 | e2e 测试（真实 vLLM + LocalStoreBackend + 真 kernel） | — | 🟡 高 |
| 10 | perf/overlap 测试（nsys 剖面，验证带宽与重叠） | — | 🟢 中 |
| 11 | 索引持久化（V2）与 fd 收敛（LocalStoreBackend 私有） | M4 | 🟢 中 |

---

## 6. 针对最新 vLLM 的设计合理性评估

**总体判断：四层结构（adapter/engine/backend/kernels）+ 外部 connector 零侵入
路线，是对的，且与最新 vLLM 完全兼容。** 依据：

1. `KVConnectorBase_V1` 的逐层回调（`start_load_kv / wait_for_layer_load /
   save_kv_layer / wait_for_save / get_finished`）在最新 vLLM 中**原生存在且未变**，
   Tutti 的逐层编排与之严丝合缝。
2. 外部包零侵入注册（`factory.py` 的 `kv_connector_module_path`）为
   `TuttiConnectorV1` 提供现成挂载点，无需 fork vLLM。
3. `get_num_new_matched_tokens` 已正确适配新签名（返回 `tuple[int|None, bool]`），
   `update_state_after_alloc(request, blocks, num_external_tokens)` 签名正确，
   `build_connector_meta` 遍历 `finished/new/cached` 三类请求对齐了
   `CachedRequestData`（`req_ids/resumed_req_ids/new_block_ids/all_token_ids`）新结构。

**但不合理/需修正的点**：

- **C1 是方向性错误**：最新 vLLM 明确把 connector 拆成 SCHEDULER/WORKER 双角色跨进程，
  设计文档却写"进程内无 RPC"，属于对 vLLM 进程模型的误判。这是最需要先纠偏的。
- **staging 从 GPU-direct 退化为 host-bounce**：违背 Tutti 的核心卖点，需在设计层
  重新确认 staging 的内存位置。
- 若干新 vLLM 抽象未接入（piecewise CUDA graph、CoW、HMA、preemption），属于
  "能跑起来但生产环境会踩坑"的缺口。

**结论**：设计骨架合理、可落地；但当前处于"结构完成、端到端未通"状态，5 个
阻断性问题解决前无法在真实最新 vLLM 上跑通，且其中 C1（跨进程同步）与 C4
（staging 内存位置）属于需要在设计层重新决策、而非单纯补代码的问题。

---

## 7. 建议（行动顺序）

1. **先纠偏 C1**：在设计文档中明确跨进程索引同步方案（恢复 lookup server 或等价的
   RPC/共享内存），并落地实现——这是 load 路径成立的前提。
2. **C4 设计决策**：确定 staging 内存位置（GPU vs pinned host），并同步修改
   `core.py:bind`、`LocalStoreBackend.bind_staging` 与 `mem_kernels.cu` 的
   `get_kernel_ptr` 路径，确保三者一致。
3. **C2 + C3**：metadata dataclass 化 + list 字段；补 `requires_piecewise_for_cudagraph`。
4. **C5**：把 `single_layer_kv_transfer` 接成 gather/scatter 钩子，补 slot_mapping 构建。
5. **补齐 M 类**：nvtx 标记、preemption、CoW、e2e/perf 测试。

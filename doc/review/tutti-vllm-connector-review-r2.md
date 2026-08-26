# Tutti vLLM Connector 综合复审（重构后）

日期：2026-08-24
范围：`integration/vllm-connector/` 当前实现、权威结构设计
`doc/intergration/redesign/02-structure-and-naming.md`、legacy
`third_pkgs/tutti-legacy/`、vLLM V1 `KVConnectorBase_V1`，以及
`orchestra/BOARD.md` 和未破问题现场记录。
本次只做静态代码和文档评审，未启动程序、未运行测试。

## 结论摘要

当前重构后的代码已经从“结构草图”进入“逐层 staged pipeline 可运行”的阶段：

- `index` 已保持纯语义，链式前缀哈希、LRU、pin、pending-store 和容量计划都与
  legacy 的逻辑索引同源。
- 双进程索引已经按裁决改为 worker 权威、scheduler 近似、查询前
  `sync_from_store()` 对账；旧 review 中“scheduler 永远查不到 worker 数据”的
  C1 已不再是当前实现的准确描述。
- metadata 已改为 `@dataclass` 的 `KVConnectorMetadata` 子类，并使用 Python 基本类型列表；
  `requires_piecewise_for_cudagraph()` 也已覆盖为 `True`。旧 C2/C3 已修复。
- CUDA staging 和真正的 gather/scatter hook 已接线；旧 C4/C5 的“必然 host bounce /
  noop 搬运”结论不能直接套用到当前代码。

当前仍有两个独立层面的关键问题：

1. **正确性层面**：重启后真实外部命中乱码尚未被 vLLM 协议边界定位。数据面三组
   probe 已证明 Tutti runtime、文件层、块表重映射和跨卡 scatter 基本无罪，最可疑的
   是 external-computed token 对应的 block table/chunk 映射。
2. **性能/调度层面**：当前实现只能做“当前层等待后提交下一层”的单层 lookahead，
   不能复现 legacy `batched_get()` 的“forward 前提交所有层 IO、再用 stream/event
   与计算重叠”的 all-layer DAG。根因不是 `load_kv_async=False`，而是有界双半窗、
   host-polled IO completion，以及 scatter 只在 `wait_for_layer_load()` 中触发。

## 分级意见

### 阻断

#### B1. 重启后真实外部命中乱码：当前最可能是 vLLM block-table 语义边界

`get_num_new_matched_tokens()` 返回外部命中 token 后，scheduler 在
`update_state_after_alloc()` 中记录加载区间；worker 再按 metadata 的 block table
为每个 chunk 生成目的映射：

- scheduler 记录 `num_external_tokens` 和起点：
  `integration/vllm-connector/adapter/connector.py:331-344`；
- worker 按 `first_chunk/n_chunks` 切 token key 与 block table：
  `integration/vllm-connector/adapter/worker.py:308-327`；
- 每个 chunk 的块表按固定 `blocks_per_chunk` 切片：
  `integration/vllm-connector/adapter/worker.py:432-438`。

现场记录明确显示：重启后真实盘读会乱码，而单卡、异表、双卡立即消费三组 probe
均逐字节通过，覆盖了 TuttiKVStore、文件 offset、非连续块表、跨卡 staging 和
scatter 写回范围（`orchestra/results/layer-resolve.cross-card-scatter-suspect.md:16-25`）。
因此首要核对项是 vLLM 为 external-computed tokens 分配的
`block_ids` 是否满足“第 `c` 个 chunk 对应 `block_ids[c * blocks_per_chunk:]`”这一
隐含假设，尤其是共享前缀、局部命中、CoW 和 `num_computed_tokens` 截断场景。

这是当前唯一仍直接影响“真外部命中输出正确性”的阻断项。应在真实 vLLM 服务中记录：
`load_tokens`、`load_start_token`、`block_ids` 长度、每 chunk 块数、共享前缀边界和
scatter 实际写回块集合，不能只用绕过 scheduler 的 engine probe 代替。

#### B2. 若性能目标是 legacy all-layer 图，当前 engine/store SPI 不具备该能力

当前读取编排明确是逐层 lookahead：

- `start_load_kv()` 只提交第 0 层：`integration/vllm-connector/adapter/worker.py:293-334`；
- `wait_for_layer_load()` 等待当前层后才提交下一层：
  `integration/vllm-connector/adapter/worker.py:336-347`；
- `KVEngine.load_layer()` 是“一层 × N chunk”并占用一个 ring wave：
  `integration/vllm-connector/engine/core.py:264-278`。

RingWindow 只有两个半窗，wave 2 在复用 wave 0 前等待旧句柄：
`integration/vllm-connector/engine/staging.py:58-68`。因此把
`_start_load_layer()` 改成启动所有层，会在第三个 wave 阻塞；如果不阻塞，就必须
为所有层保留 staging 数据，破坏当前有界 HBM 设计。

这不是 vLLM V1 契约禁止的。V1 明确允许 `start_load_kv()` 在 forward 前启动异步加载，
`wait_for_layer_load()` 逐层保证数据可用（`third_pkgs/vllm/vllm/distributed/
kv_transfer/kv_connector/v1/base.py:313-343`）。因此这是当前实现能力不足，而不是
`load_kv_async=False` 的契约问题。

### 主要

#### M1. 当前 overlap 是 `IO(L+1) || compute(L)`，不是 legacy 的 all-layer overlap

vLLM attention wrapper 在每层计算入口调用 wait，返回后才执行该层计算：
`third_pkgs/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:50-57`。
当前 `_PostCompletion.wait()` 会先 host-wait Tutti IO，再执行 scatter 并同步其 event：
`integration/vllm-connector/engine/core.py:21-47`。所以实际时间线为：

```text
submit IO(L0)
wait L0 -> scatter L0 -> submit IO(L1) -> compute L0
wait L1 -> scatter L1 -> submit IO(L2) -> compute L1
```

legacy 的 all-layer 版本则在一次 forward 前遍历所有层提交 DMA，直接写 paged KV
目的地址，然后由计算流逐层 wait event；对应代码为
`third_pkgs/tutti-legacy/engine/pkg/v1/storage_backend/geminifs_backend.py:334-372`。
legacy 也有逐层 lookahead 版本（同文件 `:411-455`），当前实现与后者一致。

#### M2. Tutti IO completion 仍是 host-polled opaque handle，不能直接形成 CUDA DAG

Python binding 只暴露 opaque io ticket 和 `wait()` 状态查询：
`integration/vllm-connector/bindings/python/src/_core.cpp:225-244`。Store 侧按轮询间隔
调用 runtime wait，并在 host 侧 release：
`integration/vllm-connector/stores/tutti_nvme/store.py:71-122`。

submit 虽然带有 CUDA stream 句柄（`store.py:467-480`），但当前 Python API 没有返回
可供 compute stream `wait_event` 的 IO-ready event。因此仅增加“预提交所有 IO”的
循环不能得到 legacy 的非阻塞 stream/event 顺序，反而会在 host wait 或 ring 回绕处
串行化。

#### M3. scatter 事件链已修复 wave 覆盖保护，但仍不是前置异步 scatter

当前 scatter hook 以 `sync=False` 发起目的侧搬运，并在当前设备流记录 CUDA event：
`integration/vllm-connector/adapter/worker.py:125-136`；组合句柄交给 RingWindow
保护槽位，满足“DMA 完成且消费方读完后才能复用”的 B1 裁决。

这解决了旧版“wave 回绕覆盖未消费 staging”的正确性问题，但 scatter 的调用时机仍
绑定在 `handle.wait()`。也就是说，它是 layer gate 的一部分，而不是 IO 完成后立即在
独立 transfer stream 上排队的后台动作。要实现 all-layer 或更深 lookahead，需要把
“IO 完成通知”和“scatter 提交”拆开，并让 `wait_for_layer_load()` 只向计算流插入
event wait。

#### M4. `max_chunks_per_wave` 目前是硬失败，不是自动 wave splitting

`KVEngine._prepare_layer_call()` 对单层 chunk 数超过 wave 容量直接抛错：
`integration/vllm-connector/engine/core.py:315-331`。worker 当前把一个请求计划的
全部加载 chunk 传给每一层，因此长前缀可能在 engine 入口失败，而不是拆成多个波次。
legacy 的批量提交也受 runtime 窗口约束，但其调用层已经有 windowed submit/retry
语义；当前 connector 需要在 engine 或 worker 层显式切 wave，并把每个 wave 的
block-table 子区间保持一致。

#### M5. 分层纪律仍有硬件语义泄漏

`ChunkIndex` 本身是纯逻辑组件，明确“不接触数据本体、单线程使用”：
`integration/vllm-connector/index/chunk_index.py:1-5`。`KVEngine` 主要编排 store SPI、
ring 和 transfer hook，整体边界合理。

但 adapter/worker 仍直接决定 storage factory、设备和硬件对齐细节：

- adapter 直接 `create_store()`：`integration/vllm-connector/adapter/connector.py:106-136`；
- worker 将 staging 固定分配到 `cuda:0`，并在该层做 64 KiB 对齐：
  `integration/vllm-connector/adapter/worker.py:512-545`；
- gather 为确保跨卡提交前数据可见，调用全设备同步：
  `integration/vllm-connector/adapter/worker.py:140-161`。

这使 adapter/worker 不再是纯 vLLM 协议翻译层，也限制了 TP rank 到本卡 runtime 的
直接路径。设备选择、对齐和 runtime 能力应由部署层/store binding 提供；跨流正确性
应使用 event 链，不能把全局 `torch.cuda.synchronize()` 作为长期方案。

#### M6. worker→scheduler 回报链为空不是当前 sync-load 的契约违例，但限制失败治理

当前 `load_kv_async=False`，`get_finished()` 返回空集合：
`integration/vllm-connector/adapter/connector.py:286-292`。同时没有实现
`build_connector_worker_meta()` / `update_connector_output()`，但 V1 基类对这两个
方法提供了可选默认实现（`third_pkgs/vllm/vllm/distributed/kv_transfer/
kv_connector/v1/base.py:450-458,558-566`）。

V1 对同步加载明确允许在发生错误的 forward 中直接通过
`get_block_ids_with_load_errors()` 报告失败块（`base.py:396-414`）；当前 connector
也保留了独立错误集合路径（`worker.py:414-448`）。因此“零实现”本身不是当前
`load_kv_async=False` 的硬性违约，也不是 S1 的首要根因。

但它意味着：未来一旦引入跨 step 异步、后台 scatter 或 all-layer completion，现有
回报链无法向 scheduler 传递请求完成、失败和释放时机，必须补齐 worker metadata/output
契约，不能继续依赖空实现。

### 次要

#### S1. 双进程索引对账是安全降级，不是原子共享索引

当前 scheduler 查询前调用 `sync_from_store()`，再执行本地 `lookup_prefix()`：
`integration/vllm-connector/adapter/connector.py:301-329`；engine 的对账语义也明确
“持久层完整层组缺失即 miss，不触碰数据面”：
`integration/vllm-connector/engine/core.py:361-368`。

这符合已裁决的“worker 权威 + scheduler 近似 + miss 降级”方案，修复了旧 review C1
的进程内索引假设。但它不是原子 RPC：worker 刚完成写入而 scheduler 尚未扫描到时，
会短暂 miss；这应当降级为重算，不得把近似视图当作数据面权威。对账扫描的频率和
元数据目录开销也需要 perf 数据。

#### S2. 当前 staged path 仍不等价于 legacy GPU-direct

CUDA 池场景现在确实分配 device staging，并注册为 runtime device memory：
`integration/vllm-connector/adapter/worker.py:527-545`，旧版“必然 pageable host
staging”的 C4 已修正。

但 `select_transfer()` 当前恒返回 `StagedTransfer`：
`integration/vllm-connector/engine/transfer.py:49-55`，NVMe DMA 目标仍是 staging，
随后再由 transfer kernel 写入 paged KV。它尚未提供 legacy 那种直接把 paged KV 作为
DMA 目的地址的 `DirectTransfer`。这是性能路径差异，不应再描述为“数据不搬运”；
当前 hooks 已实际接线。

#### S3. gather 的全设备同步会压低并发收益

`PagedTransferHooks._transfer()` 在 `sync=True` 时调用
`torch.cuda.synchronize()`（`worker.py:140-161`）。这可以保证 staging 写入后再提交
device IO，但会等待该进程设备上的所有工作，可能把无关 compute、通信和其他请求
一起串住。应改成 source stream record + IO stream wait，或由 runtime 接受 CUDA event
依赖。

#### S4. NVTX/perf 观测仍不足以证明 legacy overlap

legacy 有 load/store stream 和 NVTX 标记；当前实现虽然有独立 IO stream 配置
（`stores/tutti_nvme/store.py:204-229`），但没有看到覆盖 adapter、每层 IO、scatter、
compute gate 的统一 trace 标记。没有 nsys 时间线，不能仅凭“submit 返回异步”断言
IO/compute 已经 overlap。

## 与 legacy geminifs 的逻辑一致性

| 维度 | legacy geminifs | 当前 Tutti connector | 判断 |
|---|---|---|---|
| chunk 文件语义 | 一个 chunk 对应一个 GPUFile，层数据位于固定层段 | 一个 chunk 对应一个 target/file，`layer_idx × segment_bytes` 定位 | 一致，当前 store 仍层盲，层号由 io_key 编入 |
| 前缀哈希 | `CacheEngineKey.chunk_hash` 的滚动前缀身份 | `blake2b(parent + token bytes)`，namespace 可注入 | 语义一致；字节编码和 namespace 是新实现的明确化 |
| LRU | 逻辑索引 + evictor，命中刷新 | `resident/pending/pin` + OrderedDict LRU | 一致；当前 worker 权威、scheduler 近似 |
| pin | load/store 期间保护待使用 chunk，避免驱逐 | `pin()` / `unpin()`，wave 和请求边界释放 | 一致；真实 vLLM block 映射仍待 B1 核验 |
| 层完整性 | 读请求按层目标，缺层不能组成完整 KV | `io_key = chunk_key + layer_idx`，对账要求层集合完整 | 一致，且新实现显式拒绝部分层命中 |
| 写入确认 | 写完成后更新索引/元数据 | `put` completion settle 后 commit layer/live | 一致，崩溃安全语义更明确 |
| 逐层流水 | `layerwise_batch_get()` 等待当前后提交下一层 | `wait_for_layer_load()` 等待当前后提交下一层 | 一致 |
| all-layer 预提交 | `batched_get()` 可遍历所有层并在两个 load stream 上发起 | 当前双半窗 staged path 不支持安全 all-layer | **不一致，且是有依据的重排限制，不是静默偏离** |
| 数据目标 | GPU-direct 直接写 paged KV | staged NVMe → device staging → scatter kernel → paged KV | 性能路径不一致；语义可一致，需 DirectTransfer 才恢复 legacy 路径 |
| 双进程索引 | lookup server/RPC 是权威查询 | worker 权威 + scheduler `sync_from_store()` 近似 | 架构不同但符合当前裁决；miss 安全降级 |

总体判断：chunk、链式前缀、LRU/pin、层完整性和逐层流水是有依据的重排；当前
主要静默偏离集中在“GPU-direct 变 staged”和“all-layer 变单层 lookahead”，二者
影响性能/调度能力，不应混同为存储语义错误。

## 并发语义与 `load_kv_async=False`

当前选型成立的理由：

- scheduler 返回 `(external_tokens, False)`，请求直接进入本步 forward，而不是进入
  跨 step 的 Deferred/WAITING 状态；实现位置为
  `integration/vllm-connector/adapter/connector.py:301-329`。
- worker 只在当前 forward 的层入口按需等待，staging 由双半窗界定，HBM 占用不随
  “所有等待请求 × 所有层”线性增长。
- 这正好避开 LMCache async bulk 在并发大于 55 时 Deferred 请求持有大量 HBM，导致
  DMA 带宽从约 2.5 GB/s 下降到约 380 MB/s 的事故模式；该裁决已记录在
  `orchestra/BOARD.md:72-77`。

反例和边界：

- 单个超长前缀仍可能在 `max_chunks_per_wave` 处硬失败，而不是平滑退化为多个 wave；
- 多个请求共用一个 IO stream 时，host polling、文件 target 打开和 runtime in-flight
  窗口可能形成队头阻塞；
- 如果未来把所有层都预提交到 staging，`load_kv_async=False` 仍然不能阻止 HBM 被
  预取占满。正确做法是 K-lookahead/bytes admission，而不是改变 scheduler flag。

因此建议保持 `load_kv_async=False`，先把“有界逐层流水”做成明确的 K-lookahead，
不要退回无界 async bulk。

## legacy 风格 overlap 的可行实现路径

### 当前可做到的程度

当前代码可以稳定表达：

```text
IO(L+1) 与 compute(L) 重叠
```

前提是 Tutti runtime 的 device submit 真正异步，且 scatter event 能保护 ring 槽位。
当前 ring 保护和 scatter event 已具备，但 host wait 仍位于每层 gate 之前。

### 推荐的增量路径：有界 K-lookahead

1. 在 `start_load_kv()` 预提交前 K 层，每层拥有自己的 staging lease；K 由 staging
   bytes、runtime in-flight capacity 和请求准入共同限制。
2. 为 Tutti completion 增加完成通知或后台 poller；IO 完成后在 transfer stream
   自动发起该层 scatter，并记录 `ready[layer]` CUDA event。
3. `wait_for_layer_load(layer_name)` 只向 compute stream 插入
   `wait_event(ready[layer])`，不再 host-synchronize scatter。
4. RingWindow 的释放条件保持“底层 IO 完成 AND scatter event 完成”，不能只看 IO handle。
5. 为每个 wave 保存 key/block-table 子区间，避免长前缀切 wave 时重新解释 chunk 映射。

### 若必须复现 legacy all-layer 图

需要下列至少一项底层能力：

- `DirectTransfer`：runtime 直接把每层 paged KV 目的地址注册为 device memory，返回
  可与 CUDA stream 建立依赖的 completion event；或
- 每层持久 device buffer：所有层 IO 在 forward 前写入稳定的 per-layer device buffer，
  scatter/reshape kernel 再由计算流按 event 消费。

仅在 adapter 中循环调用 `engine.load_layer()` 不够：当前 ring 会回绕等待，当前
completion 只有 host polling，当前 scatter 又在 wait 中才提交。

读写不必先拆成两套调度器。一个 read IO stream 也能验证 all-layer read + compute
overlap；读写分流是吞吐优化项，不是形成正确 event DAG 的必要条件。

## “重启后真外部命中乱码”根因排序

1. **S1：external-computed block table 与 connector 的 chunk 映射假设不一致。**
   最符合“probe 全部 PASS、只有真实 scheduler 命中失败”的差集；重点检查共享前缀、
   CoW、`num_computed_tokens` 边界和 `block_ids` 的 chunk 分组。
2. **S2：computed 边界/尾部 partial chunk 的 KV 衔接。**
   现场 `61 × 256 + 101` 的尾部会重算；若外部命中边界与本地 computed 状态衔接错，
   可能出现前几个 token 正确后发散。非确定性使“错位读取”仍更优先。
3. **S1.5：异步 D2D scatter 与 wave 回绕覆盖。**
   这是合理的理论候选，但当前组合完成句柄和 RingWindow 已加入保护，且双卡/立即消费
   probe 通过，因此低于 S1/S2。仍需在真实 forward stream、两波回绕和多请求负载下做
   nsys + 事件状态核对。
4. **第四可能：跨请求 metadata/块表生命周期或 scheduler 近似索引时序。**
   例如 scheduler 在 worker 写入结算前后对同一 prefix 生成不同的近似视图，或
   `scheduled_cached_reqs` 的 tracker 增量与本步 `num_scheduled_tokens` 不一致。该类
   问题需要把 request id、metadata 快照和 worker 实际消费快照关联起来；目前没有直接
   数据证明。

结论：现阶段不应再把“盘上数据损坏”或“runtime read 返回乱码”作为首要假设。应先
   证明真实 vLLM 的 block allocation、chunk boundary 和 layer-wise destination mapping。

## 建议执行顺序

1. 在真实重启命中请求上 dump S1 所需的 metadata/block-table 映射，并用同一快照离线
   重放 `worker.start_load_kv()`；这是正确性门禁。
2. 给 engine 增加 wave splitting，先消除 `max_chunks_per_wave` 长前缀硬失败。
3. 将 gather 的全设备同步改成显式 stream/event 依赖，并补齐 IO completion →
   scatter → ready event 的可观测时间线。
4. 实现 K-lookahead，保持 `load_kv_async=False`，以 bytes/in-flight 为准入上限。
5. 只有 perf/nsys 证明 staged path 不够时，再实现 DirectTransfer；不要为追求
   all-layer 图而无界扩大 HBM staging。
6. 补充真实 vLLM + 重启 + 共享前缀/CoW/partial-tail 的回归用例，以及 nsys overlap
   和高并发带宽曲线。

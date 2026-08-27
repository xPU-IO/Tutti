# Tutti vLLM Connector 架构与实现静态评审

> 评审日期：2026-08-25
> 评审范围：`integration/vllm-connector`、Tutti runtime/local NVMe data path、当前 vendored vLLM，以及 `third_pkgs/infinikv-for-ae`。
> 方法：初稿来自只读代码和 Git 历史分析；本次修订又对照了 `hy3-final-r5.nsys-rep`、当前 vendored vLLM `897ff4f39`、真机验证摘要和当前工作树。
> 目标：核对历史决策与当前实现，逐层梳理 Python 到 GPU 线程的调用路径，比较 InfiniKV，并给出实现“计算 kernel 与存储 kernel 按层 overlap”的修改方案。

> 修订说明（2026-08-25）：初稿把完整 `StepPlan/GpuDagExecutor` 写成了取消逐层 host 调度的近似前置条件，复杂度过高。当前优先方案改为独立的静态 read/write staging bank：同一 stream 内依靠天然顺序完成 `read -> scatter` 或 `gather -> write`，跨 compute stream 只桥接 layer-ready/compute-done fence。完整 step DAG 保留为后续优化，而不是第一阶段准入条件。

> 三张并行审计卡验收结论：静态 bank 的几何和 event 顺序设计通过，但 Runtime 当前不能保证 80 层全量预 enqueue；`max_in_flight_operations=4` 是 DataPath op admission，不是无限 stream queue。Nsight 2026.4.1 已确认 H20/GH100 metric set 可采 PCIe RX/TX，storage plugin 可采 local NVMe Read/Write B/s；r5 没有这些数据是因为采集参数关闭了它们。

> 后续实现验收：Store 双 stream routing 已接通 production binding capability，binding tests 10/10、store contracts 65/65；`io_granularity`/prebuilt fast path 已接通，StorageRuntime contracts 41/41、local-NVMe 真机 contracts 774/774。正常 staging-shaped READ/WRITE 的 dynamic descriptor count 为 0，跨 FIEMAP extent 的短 fragment 明确回退 dynamic。

> 2026-08-26 设计卡验收：当前按 vLLM/LMCache 的既有同步加载失败契约实现，不考虑 CUDA Graph。每层 read/CQ 成功后才 scatter；失败时记录 `invalid_block_ids`、禁止失败层及后续 lookahead scatter/save，但不把确认的 I/O 失败作为裸异常抛出。eager forward 可能继续执行，scheduler 丢弃受影响请求的本步输出，并按 `kv_load_failure_policy=recompute|fail` 重算或终止。`STREAM-ARENA-DESIGN-06` 条件通过：per-stream workspace 与 compact completion方向正确，但 `build[2] + feeder` 只表示80个逻辑 op admission，不等于80个 CUDA kernel 已预排。`SPLIT-BANK-06` 已通过；`STRUCTURED-COMPLETION-07` 已通过；`REMOVE-GPU-PRP-10` 已删除local/striped GPU PRP backing，cache miss/exhausted统一使用可增长host-pinned pool。仍需限制released terminal result的长期缓存。

> 2026-08-26 真机闭环：修复preset/LocalNvmeDataPath旧静态库ABI错配、rank-local RuntimeConfig.accel_id、完整4 KiB host PRP page、logical block/driver MDTS descriptor切分，以及Runtime capability驱动的submit前completion回压。local真机contract 179/179，32x256 KiB write/read byte-exact，80层x32 chunk在窗口4下完成，Python 253/253。Hy3 TP4 `hy3-recompute-real-10f`完成A-cold与B-80pct，B外部命中6400 tokens。随后 scheduler 已拆成 `SchedulerMetadataIndex` 和 metadata-only store；worker 才构造 StorageRuntime/CUDA/NVMe data plane。

> 2026-08-27 feeder闭环：一次性诊断证明Hy3的物理KV层与callback均为完整顺序0..79；旧的“callback缺层”推断不成立。停滞点是最后read callback 79尚未进入DataPath gate wait，原因是step提交时立即启动的whole-op watcher与per-layer gate竞争Runtime registry lock。当前已建立`KVCacheConfig.kv_cache_groups[*].layer_names`到物理ordinal的显式映射，feeder计划按callback数构造；whole-op watcher延迟到全部callback发布后启动，Runtime gate wait改为lock内单次非阻塞probe。subset/乱序/重复/缺失callback均有fail-safe合同。真实Hy3 TP4 A/B和Nsight通过，B命中6400 tokens并正常退出；compute/read/write为stream 19/31/35，每rank read/write feeder kernel为40/80，无永久gate。仍保持eager，不引入CUDA Graph。

## 1. 总体结论

当前重构的**长期分层方向优于历史实现和 InfiniKV**：

1. `adapter -> engine -> index -> transfer -> store -> Tutti runtime/data path` 的职责边界比历史的 `adapter -> engine -> backend -> kernel` 更清楚。
2. 当前实现把语义键、放置、传输、持久化拆开，并引入多槽 `RingWindow`、LMCache 风格 load/reshape kernel 和可选 direct transfer，分层方向正确；但当前同一个 ring 同时服务 read/write，不是最终资源模型。
3. staging 不是临时 fallback。对 `block_size=16`、MLA、linear attention 等单 block 字节数小、地址或长度不能满足 4 KiB/64 KiB 约束的布局，staging 是主数据路径：先把多个小 block gather/reshape 成连续、对齐的大 segment，再做 I/O；load 方向相反。
4. InfiniKV 已经具备更直接的 paged KV GPU I/O 和注册期 PRP/DMA 上下文，适合尺寸足够大且满足对齐的 block，但 metadata、失败传播、生命周期和当前 vLLM layout 兼容性明显更弱。
5. 当前 Tutti 实现仍然是**逐层 Python 回调推进的双路径 pipeline**。load 可预取前 K 层，staged 路径也可在独立 I/O stream 上运行，但 `wait_for_layer_load()` 仍负责等待当前层并提交第 L+K 层，`save_kv_layer()` 仍逐层提交，尚未达到一次提交完整 compute + I/O DAG。
6. 历史 `Target_arch.md` 的“一次描述、一次 H2D、GPU 内调度”目标仍有价值，但不是第一阶段所必需。先把所有 read/scatter 排到固定 read stream、把 write 放到独立 write stream，即可删除热路径的动态槽位 `acquire()`；完整一次 H2D/step DAG 可以后续再做。
7. 路径选择必须使用实际 I/O 字节数和能力约束，而不能只判断 token block 数是 16、128 还是 256。决定量至少包括 `page_size_bytes`、单次 extent 长度、地址/offset/length 对齐、descriptor 数量、是否需要 reshape，以及 data path 的最小高效 I/O 大小。
8. 当前不存在一个独立的“Tutti 物理块”层。staged 数据从 `block_id -> canonical layer segment -> target byte offset -> FIEMAP extent -> namespace LBA`；内存从 `staging byte offset -> 64 KiB GPU pin granule -> controller MPS ioaddrs/PRP`。namespace LBA 以下的 NAND 物理映射属于 SSD FTL，代码不可见。

因此，建议保留 staging 但把当前共享 `RingWindow` 改成**静态、互不重叠的 read/write bank**。第一阶段用 CUDA stream 顺序和 layer fence 消除 host wait/动态 `acquire()`；第二阶段再让 staged/direct 按 I/O 几何自动选择，并视 profiling 结果决定是否需要 `StepPlan`、一次 H2D 和 GPU timeline。

## 2. 历史决策文档的定位

### 2.1 文档来源

`integration/vllm-connector/Target_arch.md` 与历史提交 `8ffd81f` 中的 `ARCHITECTURE.md` 内容一致；该历史文件在提交 `666c622` 被删除。当前更接近重构事实来源的是：

- `doc/intergration/redesign/02-structure-and-naming.md`
- `integration/vllm-connector/adapter/connector.py`
- `integration/vllm-connector/engine/core.py`
- `integration/vllm-connector/stores/tutti_nvme/store.py`

因此 `Target_arch.md` 应被视为**历史目标和性能意图**，不能当成当前代码的精确说明。

### 2.2 历史目标中仍然正确的部分

历史文档的核心性能目标是：

1. 一次构造全部层、全部 token/block 的 I/O 描述符。
2. 一次或常数次 H2D，把描述符、地址和依赖关系送到 GPU。
3. GPU 侧执行 I/O kernel、scatter/gather kernel，并通过 event/fence 与计算流建立依赖。
4. CPU 不在每层 I/O 完成后轮询或重新提交下一层工作。
5. 通过固定 ring/pool 避免热路径分配和注册。

这些方向与最终目标一致，尤其适合实现：

```text
read[L+1]  || compute[L] || write[L-1]
```

### 2.3 历史文档内部的冲突

`Target_arch.md` 同时描述了两种不同执行模型：

```text
模型 A：start_load_kv() 一次提交全部 N x M 描述符和所有层 I/O
模型 B：start_load_kv() 只提交 layer 0，wait_for_layer_load(L) 再提交 L+1
```

二者不能同时成立：

- 模型 A 才能做到 Python 提交次数与层数无关。
- 模型 B 仍然有每层 Python 控制路径，只是把同步等待改成流水提交。

最终性能目标仍偏向模型 A，但不要求第一版就实现通用 DAG executor。最小落地可以在 `start_load_kv()` 的一次 C++/pybind 调用中，把各层 `read -> scatter -> record ready[L]` 按顺序排到 read stream；vLLM 的逐层 callback 只让 compute stream 等待 `ready[L]`。write 可先由逐层 callback 在 `compute_done[L]` 之后排到独立 write stream，再逐步升级为一次预描述。

### 2.4 历史文档中过强的表述

- “CPU O(1)”不严谨。可以做到 Python 调用次数和 H2D 次数为 O(1)，但构造 N 层、M 个 token/block 的描述符总工作量仍是 O(NM)，除非描述符可由规则在 GPU 上生成。
- 历史文档描述 16 种 layout；当前 transfer discovery 和 native kernel 实际只接通 5 种旧格式。
- 历史文档要求注册期预建 PRP；当前 Tutti connector 的 `io_granularity` 没有完整传到 data path，导致 local NVMe 的预建分支在该路径下不可达。
- 当前 worker 已实现一块连续 staging allocation 内的多槽 `RingWindow`。槽数为 `2 * max_chunks_per_wave * min(lookahead_k, num_layers)`，能够表达双半窗和 K 层预取；仍未实现的是把这些波次一次性编译、提交为 GPU DAG。

## 3. 历史、当前重构和 InfiniKV 的分层对比

| 层次 | 历史设计 | 当前 Tutti connector | InfiniKV | 实际效果 |
|---|---|---|---|---|
| vLLM 接口 | connector callback | `TuttiConnector` | `InfiniKVConnector` | 都受 vLLM per-layer hook 约束 |
| 请求语义 | backend 生成 opaque chunk/path | deterministic `chunk_key` + layer I/O key | block/file metadata | 当前 Tutti 最清晰 |
| metadata | 与 backend/文件耦合 | `ChunkIndex` 独立于 store | 进程内 dict + backend lookup side effect | 当前 Tutti 更适合持久化和多后端 |
| 执行编排 | engine/backend | `WorkerImpl` + `KVEngine` + `RingWindow` | `InfiniKVEngine` + vLLM adapter | 当前 Tutti 支持 K 层预取，但仍由 per-layer callback 推进 |
| layout | 声称支持 16 种 | LMCache kernel 支持 5 种格式，另有 cross-layer interleaved 索引拷贝 | 面向旧式显式 K/V layout | Tutti 已覆盖 MLA 和一种 cross-layer 形态，仍需按真实 stride 验证 |
| 内存注册 | 目标为固定 ring + PRP | 已有 host-pinned `PrpBufPool`/`PrpPageCache` 和 GPU descriptor pool，但 connector granularity 丢失 | DMA context + granule mappings + PRP | Tutti 的 PRP backing 已修好；当前缺口是 fast-path 选路和 paged KV pool 注册 |
| GPU I/O | 目标为 batch kernel | local NVMe submit/progress kernel | `nvme_batch_xfer_kernel` | 两者均有 GPU I/O kernel |
| 完成语义 | event/fence | Python watcher + completion/event | host callback + stream/event | Tutti 的 active `wait()` 路径安全，但 `query()` 对最终 scatter 完成语义过早；InfiniKV 状态粒度不足 |
| staged reshape I/O | 固定 ring/pool | 多槽 `RingWindow` + LMCache 风格 gather/scatter/reshape | 非主路径 | 小块、非对齐、MLA/linear attention 下 Tutti 设计更合理 |
| direct paged I/O | 历史目标 | capability probe 已有，generic runtime 尚未提供 backend | 已实现 explicit K/V paged read/write | 大且对齐的 I/O 上 InfiniKV 暂时领先 |
| 路径选择 | 未完整定义 | bind 期 `direct_transfer` 布尔开关 | 固定 direct | 目标应按实际 I/O 字节 extent 自动选择 |
| 一次提交完整 DAG | 目标 | 未实现 | 未实现 | 三者都未达到最终目标 |

## 4. 当前 Tutti connector 的实际调用路径

### 4.1 Scheduler 侧：查询、分配和 metadata

#### vLLM 调度入口

```text
Scheduler.schedule()
  -> kv_connector.get_num_new_matched_tokens(request, block_aligned_local)
  -> TuttiConnectorV1.get_num_new_matched_tokens()
  -> KVEngine.sync_from_store()
  -> KVEngine.lookup_prefix(token_ids)
  -> ChunkIndex.lookup_prefix(...)

Scheduler.schedule()
  -> block_manager.allocate_slots(..., num_external_computed_tokens)
  -> kv_connector.update_state_after_alloc(request, blocks, ...)
  -> TuttiConnectorV1.update_state_after_alloc()
  -> 保存 load_start_token / load_tokens / live request
  -> Scheduler._update_after_schedule()
  -> request.num_computed_tokens += num_scheduled_tokens

Scheduler.schedule()
  -> kv_connector.build_connector_meta(scheduler_output)
  -> TuttiConnectorV1.build_connector_meta()
  -> 更新 RequestTracker 的 token_ids/block_ids
  -> 构造 TuttiConnectorMetadata
  -> save 候选执行 plan_store()/optimistic confirm_store()
```

关键位置：

- `third_pkgs/vllm/vllm/v1/core/sched/scheduler.py:821`
- `third_pkgs/vllm/vllm/v1/core/sched/scheduler.py:1033`
- `third_pkgs/vllm/vllm/v1/core/sched/scheduler.py:1061`
- `third_pkgs/vllm/vllm/v1/core/sched/scheduler.py:1317`
- `integration/vllm-connector/adapter/connector.py:399`
- `integration/vllm-connector/adapter/connector.py:429`
- `integration/vllm-connector/adapter/connector.py:444`

#### 这一层抽象的效果

- vLLM scheduler 只需要知道可复用 token 数、外部 KV 对应的 block table 和请求 metadata。
- `ChunkIndex` 隐藏了实际文件、offset 和 store placement。
- 当前 `TuttiConnectorV1.__init__()` 在 role 分支后惰性构造：scheduler 使用
  `SchedulerMetadataIndex` 和只访问 manifest/marker 的 metadata store；worker
  才导入 `WorkerImpl`、`KVEngine` 和 data-plane store registry。scheduler
  不再创建 CUDA stream、StorageRuntime、GPU registration 或 NVMe queue group。

### 4.2 Worker 初始化和 KV pool 注册

#### 调用路径

```text
GPUModelRunner 初始化/捕获 KV cache
  -> kv_connector.register_kv_caches(kv_caches)
  -> TuttiConnectorV1.register_kv_caches()
  -> WorkerImpl.register_kv_caches()
  -> WorkerImpl._ensure_bound()
  -> 根据 page_size_bytes/chunk_tokens 取得 segment_bytes
  -> 超量分配并切出 64 KiB 基址对齐的 staging allocation
  -> RingWindow(buffer, num_slots, segment_bytes)
  -> PagedTransferHooks + discover_engine_format()
  -> KVEngine.bind(kv_caches, window, ...)
  -> select_transfer()
     -> direct: store.create_direct_transfer() + register_paged_caches()
     -> staged: TuttiKVStore.register_buffer(window.buffer, segment_bytes)
  -> pybind Runtime.register_memory(...)
  -> StorageRuntime::register_memory(...)
  -> IDataPath::register_memory(...)
  -> LocalNvmeDataPath::register_memory(...)
```

关键位置：

- `third_pkgs/vllm/vllm/v1/worker/gpu_model_runner.py:7818`
- `integration/vllm-connector/adapter/connector.py:354`
- `integration/vllm-connector/adapter/worker.py:313`
- `integration/vllm-connector/adapter/worker.py:561`
- `integration/vllm-connector/engine/staging.py:6`
- `integration/vllm-connector/engine/transfer.py:119`
- `integration/vllm-connector/engine/core.py:230`
- `integration/vllm-connector/stores/tutti_nvme/store.py:341`
- `tutti/include/tutti/storage_runtime.h:434`
- `tutti/data_paths/local_nvme/local_nvme_data_path.cpp:983`

#### 当前注册内容

connector/store 层注册的是：

- tensor base address
- byte length
- memory kind/device
- runtime memory handle
- staging buffer 与 KV pool 的 layout 信息

local NVMe data path 本来还可以在注册时：

- map DMA pages
- 按 I/O granularity 切分 span
- 构造长期稳定的 PRP/descriptor 模板

但 `MemoryView::io_granularity` 在 runtime 的 `MemoryEntry` 中没有保存，后续 `registration_for_()` 重建 view 时也没有恢复：

- `tutti/include/tutti/memory_types.h:69`
- `tutti/include/tutti/storage_runtime.h:1079`
- `tutti/include/tutti/storage_runtime.h:1433`
- `tutti/data_paths/local_nvme/local_nvme_data_path.cpp:1026`

结果是 connector 路径下 local NVMe 的注册期 descriptor/PRP 预建能力没有真正生效。

### 4.3 Load：Python 到 GPU I/O/scatter 的完整路径

#### vLLM worker 入口

```text
KVConnectorModelRunnerMixin.execute_model()
  -> connector.start_load_kv(forward_context, **kwargs)
  -> TuttiConnectorV1.start_load_kv()
  -> WorkerImpl.start_load_kv()
  -> _ensure_bound()
  -> _prefetch_load_layers(0)
  -> 首次提交 [0, min(K, num_layers))
  -> _start_load_layer(layer_idx)
  -> KVEngine.load_layer(keys, layer_idx, block_tables)
```

关键位置：

- `third_pkgs/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:78`
- `integration/vllm-connector/adapter/connector.py:368`
- `integration/vllm-connector/adapter/worker.py:346`
- `integration/vllm-connector/adapter/worker.py:493`
- `integration/vllm-connector/engine/core.py:320`

#### Engine/store/runtime 路径

```text
KVEngine.load_layer()
  -> direct 时委托 DirectTransfer.load_layer()
  -> staged 时按 max_chunks_per_wave 切 wave
  -> RingWindow.acquire()/发生槽位覆盖时等待旧 wave
  -> 为每个 layer/chunk 构造 store get request
  -> TuttiKVStore.get_batch()
  -> Runtime.get_batch()/submit
  -> StorageRuntime::submit_with_backpressure_()
  -> LocalNvmeDataPath::submit()
  -> 动态构造 host descriptor
  -> H2D descriptor copy
  -> submit_one/native NVMe submit kernel
  -> GPU thread 提交 NVMe command
  -> progress kernel/thread 读取 CQ/status
  -> D2H status / runtime completion
  -> Python _TuttiCompletion watcher
  -> _PostCompletion.wait()
  -> transfer.scatter()
  -> mem_kernels.cu copy/scatter kernel
  -> CUDA event synchronize
```

关键位置：

- `integration/vllm-connector/engine/core.py:320`
- `integration/vllm-connector/engine/staging.py:70`
- `integration/vllm-connector/stores/tutti_nvme/store.py:431`
- `integration/vllm-connector/stores/tutti_nvme/store.py:73`
- `tutti/data_paths/local_nvme/local_nvme_data_path.cpp:1664`
- `tutti/data_paths/local_nvme/local_nvme_data_path.cpp:1807`
- `tutti/data_paths/local_nvme/local_nvme_data_path.cpp:2024`
- `tutti/data_paths/local_nvme/local_nvme_data_path.cpp:2386`
- `tutti/data_paths/local_nvme/io/submit_one.cuh:103`
- `integration/vllm-connector/engine/core.py:22`
- `integration/vllm-connector/transfer/csrc/mem_kernels.cu:242`

#### 每层计算前 callback

```text
attention layer forward decorator
  -> connector.wait_for_layer_load(layer_name)
  -> WorkerImpl.wait_for_layer_load()
  -> 对该层 completion.wait()
  -> host 等待 I/O
  -> launch scatter
  -> scatter 完成后该层 KV 可消费
  -> 提交 layer_idx + lookahead_k 的下一次预取
  -> attention compute kernel 才继续提交
```

关键位置：

- `third_pkgs/vllm/vllm/v1/attention/layer.py` 对应 connector decorator
- `third_pkgs/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:51`
- `integration/vllm-connector/adapter/connector.py:372`
- `integration/vllm-connector/adapter/worker.py:394`
- `integration/vllm-connector/engine/core.py:22`

#### 实际效果

这里不是 GPU-only 调度。`_PostCompletion.wait()` 由 Python 主机线程等待 I/O，随后发 scatter，再同步 event。计算流无法只靠预置 stream dependency 自动继续，CPU 仍位于每层关键路径。

### 4.4 Save：attention 后回调到 GPU I/O

```text
attention layer forward decorator 完成计算
  -> connector.save_kv_layer(layer_name, kv_layer, ...)
  -> WorkerImpl.save_kv_layer()
  -> transfer.gather(kv_pool -> staging)
  -> KVEngine.store_layer()
  -> TuttiKVStore.prepare_put()/put_batch()
  -> marker/file prepare，可能 zero-fill + fsync
  -> Runtime.put_batch()/submit
  -> LocalNvmeDataPath::submit()
  -> H2D descriptor
  -> GPU NVMe submit kernel
  -> NVMe CQ/progress
  -> Python completion watcher

model execution 尾部
  -> connector.wait_for_save()
  -> KVEngine.settle()
  -> 等待 outstanding completion
  -> ChunkIndex commit/marker 更新
```

关键位置：

- `third_pkgs/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:57`
- `integration/vllm-connector/adapter/connector.py:376`
- `integration/vllm-connector/adapter/worker.py:415`
- `integration/vllm-connector/engine/core.py:363`
- `integration/vllm-connector/stores/tutti_nvme/layout.py:160`
- `integration/vllm-connector/stores/tutti_nvme/store.py:477`
- `integration/vllm-connector/engine/core.py:457`

#### 实际效果

- gather 和 store 可使用独立 stream，但每层仍由 Python callback 发起。
- marker/file prepare 位于热路径，可能包含真实 zero-fill 和 `fsync`，会显著放大首写延迟。
- 当前实现不是“一次提交所有层 write I/O”；只是逐层边计算边提交。

### 4.5 GPU 最底层线程的工作

#### Transfer kernel

`integration/vllm-connector/transfer/csrc/mem_kernels.cu:242` 的当前策略是：

- 一个 CUDA block 处理一个 token。
- block 内线程数约为 `min(heads * qwords, 128)`。
- 每个线程按 64-bit word 搬运。
- native dispatch 在 `mem_kernels.cu:258` 只接通 5 种旧 layout。

它负责 KV pool 与 contiguous staging 之间的 scatter/gather，不负责 NVMe 命令提交。

#### Local NVMe submit kernel

`tutti/data_paths/local_nvme/io/submit_one.cuh:103` 的 GPU 线程：

- 每个 entry/thread 读取描述符。
- 选择 NVMe queue。
- 填写并提交 SQ entry。
- 后续 progress 路径检查 completion/status 并聚合 batch 状态。

`tutti/data_paths/local_nvme/io/nvme_submit_primitives.cuh:70` 的 queue 选择没有使用 `blockIdx`，但初稿据此推断“多 block 会压到少数 queue”过强。当前真机 preset 是 `threads_per_block=16`、`num_queues=16`，每个 block 都把 16 个 thread 均匀映射到 16 个 queue；底层 `nvm_parallel_queue.h` 又通过 atomic ticket 支持同一 queue 的并发提交者。这不是当前已确认的正确性 bug。若未来 `num_queues > threads_per_block`，确实会有部分 queue 永远不用，应作为可观测的吞吐优化项验证，而不是 P0。

### 4.6 当前实际时间线

当前近似时间线为：

```text
CPU: start_load(all requests) -> submit read layers [0, K)
CPU: wait layer 0 I/O -> launch scatter 0 -> submit read layer K -> submit compute 0
CPU: save layer 0 -> launch gather/store 0
CPU: wait layer 1 I/O -> launch scatter 1 -> submit read layer K+1 -> submit compute 1
CPU: save layer 1 -> launch gather/store 1
...
CPU: wait_for_save()
```

它已经能维持 K 层 read lookahead，并通过 ring slot 在一定窗口内重叠 read、reshape、compute 和 write；但窗口仍由每层 Python callback 补交，load 侧 `handle.wait()` 仍位于关键路径，所以还不是 GPU 自主流水。

### 4.7 `hy3-final-r5` 的运行证据与采集边界

`/data2/ryeqiu/tutti-profile/reports/hy3-final-r5.nsys-rep` 对当前路径给出了运行证据：

- workload 明确记录 `K=1`；四个 TP worker 合计 320 次 read launch，即每个 rank 80 层各一次。除 layer 0 外，layer 1..79 都位于 `tutti.request.prefetch_load|layer=N`，证明后续 read 是由逐层 callback 补交，不是 step 开始时一次排队。
- 四个 TP worker 合计 1600 次 write launch。它们来自多次 forward/save step，每个 step 内仍由 `save_kv_layer()` 按层提交。
- rank 0 上已经观察到 read kernel 与 FlashAttention compute overlap；因此问题不是“完全没有 overlap”，而是 host wait、每层 submit 和共享 ring 限制了流水深度及可预测性。
- ranks 1-3 的模型 compute 在 device 1-3，而当前 Tutti I/O kernel 都落在 device 0。这是独立的设备放置缺陷，不能用 rank 0 的 overlap 结论代表其余 rank。

该 report **不能**回答 PCIe 带宽、GPU 利用率采样或 NVMe 吞吐利用率，因为采集命令显式使用了 `--gpu-metrics-devices=none --sample=none --cpuctxsw=none --gpuctxsw=false`，也没有开启 storage metrics。r5 使用的是 Nsight Systems 2025.3.2；本机已有的 2026.4.1 提供 `--gpu-metrics-devices` 和实验性的 `--storage-metrics`，但具体 PCIe counter 是否出现在当前 GPU metric set、storage metrics 所需权限仍须单独验证。SQLite 中只有 CUDA/NVTX activity，没有 GPU metrics/storage metrics 表。下一次性能验收应单独做 metrics capture，避免把低开销 timeline capture 与系统指标 capture 混成一个巨大 report。

## 5. 当前内存注册和 layout 管理

### 5.1 当前 vLLM KV pool 的真实 layout

当前 vendored vLLM 使用 packed content，而不是旧式显式 K/V 维度：

- FlashAttention：`third_pkgs/vllm/vllm/v1/attention/backends/flash_attn.py:184`
- FlashInfer：`third_pkgs/vllm/vllm/v1/attention/backends/flashinfer.py:468`

cross-layer pool 的构造在：

- `third_pkgs/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:205`

其物理语义大致是：

```text
NHD: [block, layer, token, head, packed(2 * head_dim)]
HND: [block, head, layer, token, packed(2 * head_dim)]
```

单层 view 通常会去掉 layer 维，形成 packed 4D tensor。

### 5.2 当前 connector 已支持的 reshape 与剩余 layout 缺口

`WorkerImpl._ensure_bound()`、`PagedTransferHooks` 和 transfer discovery 已经实现三类 staging reshape：

- 3D MLA `[num_blocks, block_size, state_size]`。
- 5D 显式 K/V 的 NHD/HND 形式。
- 专用 cross-layer interleaved `[block, layer, block_token, K/V, channel]`，逐层 view 为 `[block, block_token, K/V, channel]`。

但当前 vendored vLLM 的 FlashAttention/FlashInfer 已把 K/V pack 到最后一个 content 维，cross-layer allocation 的真实物理形态是：

```text
NHD: [block, layer, block_token, head, packed(2 * head_dim)]
HND: [block, head, layer, block_token, packed(2 * head_dim)]
```

当前 worker 仍有两个固定假设：

- `WorkerImpl._layer_view()` 固定用 `pool[:, layer_idx]`，即假定 layer 总在 axis 1。
- interleaved 判型要求逐层 4D view 的 `shape[2] == 2`，即假定存在显式 K/V axis。

相关位置：

- `integration/vllm-connector/adapter/worker.py:561`
- `integration/vllm-connector/adapter/worker.py:633`
- `integration/vllm-connector/adapter/worker.py:659`
- `integration/vllm-connector/transfer/tutti_kv_transfer/_discovery.py:59`
- `integration/vllm-connector/transfer/csrc/mem_kernels.cu:258`
- `third_pkgs/vllm/vllm/v1/attention/backends/flash_attn.py:184`
- `third_pkgs/vllm/vllm/v1/attention/backends/flash_attn.py:197`

这会产生两个问题：

1. NHD 的逐层 packed 4D view 通常不满足 `shape[2] == 2`，随后又不能进入只接受 3D/5D 的 discovery，因而绑定失败。若 `num_kv_heads == 2`，它会被误判为显式 K/V interleaved；当前 `copy_/index_put_` 恰好按相同的连续字节 shape 搬运时不必然损坏数据，Hy3 r5 就走通了这一形状巧合，但这不是稳定的 layout 契约，不能据此宣称已支持 packed NHD。
2. HND 的 layer 在 axis 2，当前 axis 1 假设会把 head 当 layer，层数、地址和跨度均错误。
3. MLA 3D path 已有，但 linear attention 或多 cache group 的 state layout 还没有通用 descriptor；当前 connector 明确只接受一个 KV cache group。

现有测试仍主要构造旧 3D/5D tensor：

- `integration/vllm-connector/tests/adapter/test_adapter.py:466`
- `integration/vllm-connector/tests/adapter/test_adapter.py:740`
- `integration/vllm-connector/tests/transfer/test_transfer.py:140`

这些测试覆盖了旧 3D/5D 和测试自定义的显式 K/V cross pool，但没有覆盖当前 vendored vLLM 的真实 packed NHD/HND cross-layer allocation。staging ring 的设计是正确的，缺口位于 registration/layout descriptor 和 reshape kernel 的真实 packed-layout 接入。

### 5.3 目标中的注册期产物

注册阶段不应只记录一个 enum。应生成稳定的 `LayoutDescriptor`：

```text
LayoutDescriptor
  base_ptr
  allocation_bytes
  dtype / element_bytes
  logical_axes: block, layer, token, head, packed_kv
  sizes[]
  strides_bytes[]
  layer_axis
  token_axis
  block_stride_bytes
  layer_stride_bytes
  direct_io_spans[]
  staging_scatter_plan[]
  memory_registration_handle
  device_prp_table / descriptor_template
```

效果：

- 不再通过 rank 和固定 axis 猜 layout。
- 同一套 descriptor 可支持 NHD、HND、MLA 和未来 layout。
- direct 模式可直接生成每层/每 block 的 device span。
- staged 模式可预编译 scatter/gather 参数，step 内只填 block id 和 token range。

### 5.4 Staging ring 的设计目的

staging ring 不是为了绕过尚未实现的 direct backend，而是解决 paged KV 与存储 I/O 几何不匹配的问题。

#### 小 KV block 为什么不适合直接 I/O

vLLM 的 `block_size` 是 token 数，不是字节数。每层、每个 vLLM block 的实际字节数来自 cache spec：

```text
block_bytes = kv_cache_spec.page_size_bytes
```

普通 MHA/GQA 中它通常等价于：

```text
block_size_tokens * 2(K/V) * num_kv_heads * head_dim * element_bytes
```

MLA、linear attention 或其他 state cache 不能安全套用这个公式，必须直接使用 vLLM cache spec 给出的 `page_size_bytes`、真实 shape 和 stride。

当 `block_size=16` 或模型每 token KV/state 很小时，direct path 会遇到：

1. 单 block 长度不足 4 KiB，无法形成合法的 NVMe/direct-I/O 命令。
2. 长度虽达到 4 KiB，但 GPU DMA 映射或基址要求 64 KiB 粒度。
3. 多个 paged block 在地址上离散，一层请求变成大量小 descriptor，IOPS、doorbell 和 CQ 压力远高于有效带宽收益。
4. vLLM pool layout 与落盘 token-major layout不同，直接读写不能完成 K/V 拆合、MLA packing 或 linear-attention state reshape。
5. 为每个小 extent 构造注册、PRP 和 metadata，会让注册表和 device descriptor 表异常复杂。

staging path 把多个 vLLM block 合并成一个连续 layer segment：

```text
blocks_per_chunk = chunk_tokens / block_size
segment_bytes = blocks_per_chunk * page_size_bytes
```

当前几何推导已经位于：

- `integration/vllm-connector/adapter/connector.py:62`
- `integration/vllm-connector/adapter/connector.py:123`

这样只注册长期稳定、64 KiB 基址对齐的 staging allocation；每次存储 I/O 面向连续且至少 4 KiB 对齐的 `segment_bytes`，paged KV 与 segment 之间由 D2D gather/scatter/reshape kernel 转换。

#### 当前 staging 实现已经具备的部分

```text
WorkerImpl._ensure_bound()
  -> segment_bytes = chunk_kv_bytes / num_layers
  -> window_layers = min(lookahead_k, num_layers)
  -> num_slots = 2 * max_chunks_per_wave * window_layers
  -> CUDA allocation 额外预留 64 KiB
  -> 切出 64 KiB 对齐的 staging view
  -> RingWindow(...)
  -> PagedTransferHooks(...)
  -> KVEngine.bind(...)
```

关键位置：

- `integration/vllm-connector/adapter/worker.py:561`
- `integration/vllm-connector/adapter/worker.py:591`
- `integration/vllm-connector/adapter/worker.py:592`
- `integration/vllm-connector/adapter/worker.py:609`
- `integration/vllm-connector/adapter/worker.py:627`
- `integration/vllm-connector/engine/staging.py:6`

`RingWindow` 已经不是单槽：

- 最少是两个半窗。
- K 大于 1 时，窗口能容纳多层预取波次。
- `acquire()` 只在即将覆盖仍占用的旧槽时等待旧 completion。
- `complete()` 把一个 wave 的最终消费 event 与槽位绑定。

`PagedTransferHooks` 已经负责 paged pool 与 staging segment 的两端变换：

- 常规 3D/5D/MLA layout 走从 LMCache 移植的 `single_layer_kv_transfer`。
- cross-layer interleaved layout 走 block table 索引 copy。
- staging 在 CUDA 上时，源码里的 `D2H/H2D` 是历史命名，物理操作实际是 GPU 内 D2D。

关键位置：

- `integration/vllm-connector/adapter/worker.py:48`
- `integration/vllm-connector/adapter/worker.py:127`
- `integration/vllm-connector/transfer/csrc/mem_kernels.cu:1`
- `integration/vllm-connector/transfer/csrc/mem_kernels.cu:59`
- `integration/vllm-connector/transfer/csrc/mem_kernels.cu:146`

#### Ring slot 的正确生命周期

load slot：

```text
FREE
  -> READING_IO
  -> READ_DONE
  -> SCATTER_RESHAPE
  -> KV_READY
  -> FREE
```

save slot：

```text
FREE
  -> GATHER_RESHAPE
  -> WRITE_READY
  -> WRITING_IO
  -> WRITE_DONE
  -> FREE
```

slot 不需要跨越整层 compute 生命周期：load slot 在 scatter 完成后即可释放，save 使用另一个空闲 slot。为了避免 read 和 write 互相造成 head-of-line blocking，最终实现应使用两个 ring，或在一个 allocation 内静态划分 read/write slot pool。

每个 slot 的复用 fence 必须是该方向的最终消费者：

- load 是 scatter/reshape 完成 event，不是 NVMe read completion。
- save 是 NVMe write completion，不是 gather/reshape 完成 event。

### 5.5 按实际 I/O 大小选择 staged/direct

当前 `select_transfer()` 的行为仍是 bind 期布尔选择：

```text
direct_transfer=true 且 store 提供 create_direct_transfer
  -> DirectTransfer
否则
  -> StagedTransfer
```

位置：`integration/vllm-connector/engine/transfer.py:119`。

这只能做部署 capability probe，不能表达“block 16 走 staging，block 128/256 在 I/O 足够大时走 direct”的设计。正确选择应拆成两层。

#### 第一层：bind/registration 期能力判断

输入：

- `page_size_bytes`
- KV allocation 的 base、size、shape、stride、layer axis
- data path 的 address/offset/length alignment
- DMA registration granularity
- direct paged descriptor capability
- layout 是否能由 direct descriptor 完整表达
- staging gather/scatter/reshape kernel capability

输出不是一个固定 bool，而是：

```text
TransferCapabilities
  direct_eligible
  staged_eligible
  direct_registration_handle
  staging_registration_handle
  io_alignment_bytes
  dma_alignment_bytes
  min_efficient_io_bytes
  max_descriptors_per_batch
```

对于可能混合使用两条路径的模型，KV pool 和静态 staging bank 都在 bind 期一次注册，step 热路径不再注册内存。

#### 第二层：step/extent 期 cost-based 选择

每个 layer/chunk run 根据真实 extent 选择：

```text
Direct 必要条件：
  direct_eligible
  device address 满足 direct address alignment
  file offset 满足 storage alignment
  每个 extent length 满足 storage/DMA length alignment
  layout 无需 direct backend 不支持的 reshape
  descriptor_count <= backend limit

在满足必要条件后：
  io_extent_bytes >= min_efficient_io_bytes
  且 direct_cost <= staged_cost
    -> DIRECT
  否则
    -> STAGED
```

初版不必实现复杂在线模型，可使用可观测、可配置阈值：

```text
direct_cost = descriptor_count * direct_submit_cost
              + io_bytes / direct_bandwidth

staged_cost = 2 * io_bytes / d2d_bandwidth
              + coalesced_io_count * staged_submit_cost
              + io_bytes / storage_bandwidth
```

其中 `2 * io_bytes / d2d_bandwidth` 表示 save/load 两个方向各自只发生一次 D2D；对单次 operation 实际只取对应的一次 copy 成本。

#### 自动求 staging segment 几何

设：

- `B = page_size_bytes`，即每层一个 vLLM block 的字节数。
- `A = required_length_alignment`，至少包含 4 KiB I/O 长度约束；若 backend 还要求 64 KiB 长度，则取两者的最小公倍数。
- `M = min_efficient_io_bytes`，是性能阈值，不等同于合法性对齐。

先求满足长度对齐的最小 block 数：

```text
aligned_blocks = A / gcd(A, B)
aligned_segment = aligned_blocks * B
performance_multiple = ceil(M / aligned_segment)

blocks_per_chunk = aligned_blocks * max(1, performance_multiple)
chunk_tokens = blocks_per_chunk * block_size
segment_bytes = blocks_per_chunk * B
```

同时需要独立满足 staging allocation 的 base alignment。当前 local path 中：

- NVMe/store command 长度粒度是 4 KiB：`integration/vllm-connector/stores/tutti_nvme/store.py:31`、`integration/vllm-connector/stores/tutti_nvme/store.py:364`。
- device staging 基址按 64 KiB 对齐：`integration/vllm-connector/adapter/worker.py:609`。
- 64 KiB 是当前 GPU DMA 注册约束，不应被误写成所有后端统一的 I/O 性能阈值。

#### block 16、128、256 的准确解释

- `block_size=16` 常常产生小 extent，因此通常选择 staged；但若某模型每 token KV 很大，单 block 已达到高效 I/O 尺寸并满足对齐，也可以 direct。
- `block_size=128/256` 通常更容易达到 direct 条件；但 MLA/linear attention 的每 token state 可能较小，仍必须按 `page_size_bytes` 判断。
- 同一个模型中，完整大 run 可 direct，尾部或非对齐 run 可 staged。路径是 plan/extent 属性，不应永久绑定为 engine 全局属性。

#### 对 metadata 的约束

staged 和 direct 必须共享相同的语义 chunk key 和 durable commit 规则。路径选择只改变物理执行计划，不改变命中语义：

```text
ChunkPlan
  chunk_key
  layer_idx
  token_range
  logical_bytes
  disk_object/offset
  transfer_path: STAGED | DIRECT
  direct_extents[]
  staging_slot/wave
  reshape_plan
```

这样可以在不同模型、不同层类型或同一步不同 extent 间切换路径，而不制造两套 metadata/index。

### 5.6 `block`、`page`、`segment` 和物理块的准确含义

当前代码中至少存在七种不同粒度。它们不能都叫 page，也不能互相直接替换：

| 名称 | 单位 | 当前代码中的准确含义 | 不是 |
|---|---:|---|---|
| vLLM `block_size` | token | KV allocator 的一个逻辑 block 容纳多少 token | 字节数、文件系统块、LBA |
| vLLM `block_id` | 索引 | block table 中指向 GPU KV pool 页的 allocator 索引 | NVMe LBA、磁盘 block id |
| vLLM `page_size_bytes` | byte/layer/block | 一层中一个 vLLM block 的完整缓存字节数 | OS page、CUDA page、NVMe page |
| Tutti semantic chunk | token/key | 连续 `chunk_tokens` 个 token 的哈希和缓存命中单位 | 一次 NVMe command |
| Tutti layer segment | byte/chunk/layer | 一个 semantic chunk 在一层的 canonical staging/落盘字节串 | 整个 chunk 的所有层 |
| staging slot | byte range | ring 中一个恰好容纳一个 layer segment 的连续显存区间 | vLLM block |
| Runtime target offset | byte | file target 或 striped virtual target 内的逻辑字节偏移 | 物理设备偏移 |
| FIEMAP extent | byte range | 文件逻辑区间到 block device 字节区间的一段连续映射 | 固定大小 page |
| NVMe namespace LBA | logical block | NVMe command 的 namespace 逻辑块地址 | SSD NAND 物理页 |
| controller MPS/PRP page | byte page | PRP 地址表的页粒度，来自 `ctrl->page_size` | namespace LBA 大小、vLLM page |
| GPU pin granule | byte page | `snvme` 注册 GPU memory 时使用的 64 KiB 页粒度 | I/O 必须一次传 64 KiB |

代码依据：

- vLLM 明确定义 `block_size` 是 token 数，`page_size_bytes` 是含 `block_size` 个 token 的页字节数：`third_pkgs/vllm/vllm/v1/kv_cache_interface.py:147`、`third_pkgs/vllm/vllm/v1/kv_cache_interface.py:150`。
- Tutti public `IoRequest`、resolver payload 和 DataPath request 均使用字节 offset/length，没有公开“tutti physical block id”：`tutti/include/tutti/io_types.h:76`、`tutti/bindings/ext4_local_nvme/binding.h:53`。
- namespace block size 在 `NamespaceIdentity::block_size` 中；controller MPS 在 `nvm_dma_t::page_size` 中，是两套独立参数：`tutti/bindings/ext4_local_nvme/binding.h:88`、`tutti/device_manager/nvme/libnvm/include/nvm_types.h:98`。
- GPU 显存 pin 使用 64 KiB，但 `ioaddrs[]` 被 libnvm 展开成 controller MPS 粒度：`tutti/device_manager/nvme/libnvm/src/linux/dma.cpp:253`、`tutti/device_manager/nvme/libnvm/src/dma.cpp:127`。

最底层可从静态代码确认的“物理”边界是 **NVMe namespace byte offset/LBA**。FIEMAP 的 `fe_physical` 也是 Linux block-device 地址，不是 NAND 地址。SSD controller 的 FTL 如何把 LBA 映射到 NAND channel/die/page/erase block 对 host 不可见，因此本仓库代码无法给出“LBA -> NAND 物理页”的映射；任何把 `fe_physical` 称为 NAND 物理地址的文档都是错误的。

另外，resolver 中变量 `fs_block_size` 实际取自 `stat.st_blksize`：`tutti/resolvers/local_file/resolver.h:356`。在当前 ext4 部署它通常为 4 KiB，但 API 语义更接近文件 I/O 首选块大小；不能据此推导 SSD 的 NAND page。

### 5.7 从 vLLM cache spec 到 segment 的字节几何

定义：

```text
T  = spec.block_size                  # token / vLLM block
B  = per-layer spec.page_size_bytes   # byte / vLLM block / layer
C  = chunk_tokens                     # token / semantic chunk
P  = C / T                            # vLLM blocks / semantic chunk
S  = P * B                            # byte / layer segment
N  = num_layers
F  = N * S                            # byte / chunk file 的有效 payload
R  = number_of_ring_slots
M  = R * S                            # staging view 的逻辑字节数
```

当前 `_resolve_geometry()` 强制 `C % T == 0`，取 cache group 内每层 child spec 的 `page_size_bytes`，再要求所有层的 `B` 相同：`integration/vllm-connector/adapter/connector.py:99`、`integration/vllm-connector/adapter/connector.py:105`、`integration/vllm-connector/adapter/connector.py:116`。最终计算位于 `integration/vllm-connector/adapter/connector.py:122`。

`UniformTypeKVCacheSpecs.page_size_bytes` 是多个 child layer 页大小之和：`third_pkgs/vllm/vllm/v1/kv_cache_interface.py:796`。connector 没有误用这个 aggregate；存在 `kv_cache_specs` 时，它按 `layer_name` 读取 child spec。否则才使用 group spec 的 `page_size_bytes`。

不同 cache 类型的 `B`：

```text
普通 dense attention:
  B_unpadded = num_heads * storage_block_size
               * (head_size_k + head_size_v) * dtype_bytes
  B = page_size_padded or B_unpadded

普通 MHA/GQA 且 head_size_v == head_size_k:
  B = T * num_kv_heads * 2 * head_size * dtype_bytes

MLA:
  head_size_v = 0
  storage_block_size = T / compress_ratio
  B_real = num_heads * storage_block_size * head_size * dtype_bytes
  B = round_up(B_real, alignment)  # 仅配置 alignment 时

Mamba/linear state 类:
  B_real = sum(prod(state_shape_i) * dtype_bytes_i)
  B = page_size_padded or B_real
```

代码位置：`third_pkgs/vllm/vllm/v1/kv_cache_interface.py:237`、`third_pkgs/vllm/vllm/v1/kv_cache_interface.py:243`、`third_pkgs/vllm/vllm/v1/kv_cache_interface.py:250`、`third_pkgs/vllm/vllm/v1/kv_cache_interface.py:372`、`third_pkgs/vllm/vllm/v1/kv_cache_interface.py:381`、`third_pkgs/vllm/vllm/v1/kv_cache_interface.py:667`。

因此 `page_size_bytes` 的正确描述是：**vLLM allocator 的一个 block，在一层 cache 中占用的 storage bytes**。backend 还可能把一个 allocator block 拆成多个 kernel block；`gpu_model_runner` 用 `kernel_num_blocks = num_blocks * (spec.block_size / kernel_block_size)` 形成 attention backend view：`third_pkgs/vllm/vllm/v1/worker/gpu_model_runner.py:7542`。kernel block 是 attention kernel 的 shape 细节，不改变 allocator block 的 `B` 和落盘 `S`。

`block_id` 到 GPU raw allocation 的字节地址需要再区分两种配置。对 layer `L`、allocator block `b`、页内字节 `u`：

```text
独立 per-layer allocation:
  gpu_addr(L,b,u)
    = layer_base[L] + b * layer_block_stride_bytes[L] + u

  block-first contiguous 且无额外 padding 时：
    layer_block_stride_bytes[L] == B[L]

cross-layer packed allocation:
  gpu_addr(L,b,u)
    = shared_base
      + b * packed_block_stride_bytes
      + layer_offset_bytes[L]
      + u
```

packed layout 由 `KVCacheTensor.offset` 和 `KVCacheTensor.block_stride` 明确定义：`third_pkgs/vllm/vllm/v1/kv_cache_interface.py:928`。planner 在一个 block slab 内按 layer `page_size_bytes` 累加 `layer_offset`，`block_stride` 是一个 allocator block 的全部 layer storage bytes：`third_pkgs/vllm/vllm/v1/core/kv_cache_utils.py:1249`。worker 只分配一次 shared backing，并用 `raw.view(-1, block_stride)[:, offset:offset+page_bytes]` 产生 layer view：`third_pkgs/vllm/vllm/v1/worker/gpu_model_runner.py:7467`、`third_pkgs/vllm/vllm/v1/worker/gpu/attn_utils.py:286`。

这里的 `u` 不能盲目取 `[0,B)` 当作有效 tensor payload：当 `page_size_padded` 存在时，`B` 包含页尾 padding，backend view 只覆盖 `real_page_size_bytes`，相邻 block 的实际 stride 才是 `B`。direct descriptor 必须区分 payload span 与 padding span；staged reshape 也必须定义 padding 的 canonical 值。

### 5.8 vLLM block 到 canonical segment、文件 offset 的调用路径

完整 staged save 路径的地址变换是：

```text
TuttiConnectorV1.register_kv_caches()/register_kv_cache()
  -> WorkerImpl 保存真实 KV pool view

首次 start_load_kv()/save_kv_layer()
  -> WorkerImpl._ensure_bound()
  -> segment_bytes = chunk_kv_bytes / num_layers = S
  -> slots = 2 * max_chunks_per_wave * min(lookahead_k, N) = R
  -> 分配并切出 staging[0:M]
  -> RingWindow.slot_offset(slot) = slot * S
  -> PagedTransferHooks(..., segment_bytes=S, chunk_tokens=C, block_size=T)
  -> KVEngine.bind()
  -> TuttiKVStore.register_buffer(staging, granularity=S)

save 某层 L、某 chunk q:
  -> WorkerImpl.save_kv_layer()
  -> KVEngine.store_layer(keys, L, src_first_blocks)
  -> RingWindow.acquire() 得到 slot r
  -> PagedTransferHooks.gather()
  -> _slot_mapping(block_ids[q])
       = flatten(block_id * T + [0..T-1])
  -> single_layer_transfer()/batched_layer_transfer()
       或 interleaved index copy
  -> staging memory range [r*S, (r+1)*S)
  -> derive_io_key(chunk_key[q], L)
  -> TuttiKVStore.put_batch(io_key, buffer_id, r*S)
  -> Layout.prepare_put() 创建/扩展 chunk file
  -> Runtime request:
       target_offset = L*S
       memory_offset = r*S
       length        = S
       direction     = write
```

load 是严格逆路径：Runtime 从 chunk file 的 `[L*S,(L+1)*S)` 读到 slot `[r*S,(r+1)*S)`，随后 `PagedTransferHooks.scatter()` 根据该请求的 block table 把 token 行写回任意离散 `block_id`。

关键位置：

- staging 和 slot 几何：`integration/vllm-connector/adapter/worker.py:591`、`integration/vllm-connector/adapter/worker.py:595`、`integration/vllm-connector/engine/staging.py:66`。
- block table 到逐 token slot mapping：`integration/vllm-connector/adapter/worker.py:203`。
- canonical staging shape：普通 cache 为 `[chunk_tokens, 2, content]`，MLA 为 `[chunk_tokens, state]`：`integration/vllm-connector/adapter/worker.py:115`。
- engine 形成 `(io_key, buffer_id, slot*S)`：`integration/vllm-connector/engine/core.py:341`、`integration/vllm-connector/engine/core.py:401`。
- store 形成六元组：`integration/vllm-connector/stores/tutti_nvme/store.py:431`、`integration/vllm-connector/stores/tutti_nvme/store.py:457`。

这里有一个容易忽略但非常重要的语义：对当前已接通的 unpadded 3D/5D/interleaved 路径，落盘 segment 是 **按 semantic chunk 内 token 顺序归一化后的 canonical 字节串**，不是简单把 GPU 上任意 `block_id` 的 raw storage 按物理地址拼接。block table 只参与 gather/scatter，不写入数据文件；同一 token chunk 下次可以加载到完全不同的 vLLM block ids。`page_size_padded` 的 canonical padding 目前没有被正确建模，见第 5.14 节。

file-per-chunk 的有效布局是：

```text
chunks/<chunk_key>.bin

[ layer 0: S bytes ]
[ layer 1: S bytes ]
...
[ layer N-1: S bytes ]

layer L target offset = L * S
有效 payload F        = N * S
```

`io_key = 16-byte chunk_key || 2-byte little-endian layer_idx`，marker 文件是 `meta/<io_key>.ok`：`integration/vllm-connector/index/chunk_index.py:41`、`integration/vllm-connector/stores/tutti_nvme/layout.py:98`。marker 管层级提交状态，data file 管一个 semantic chunk 的全部层。

`capacity_chunks` 也是 semantic chunk/file 的数量，不是 block 数或字节数。单盘 nominal payload capacity 为：

```text
capacity_payload_bytes = capacity_chunks * N * S
```

实际磁盘占用还包含文件系统 allocation rounding、目录/marker metadata 和 striped padding。

### 5.9 file target 从逻辑字节到 namespace LBA 的完整路径

file-per-chunk 的 target 打开路径：

```text
TuttiKVStore._target("file://...")
  -> Python Runtime.open_batch()
  -> StorageRuntime::open_batch()/open()
  -> LocalFileResolver::resolve()
       open(O_RDONLY | O_DIRECT)
       fstat() regular file
       校验 file st_dev == configured backing block st_rdev
       fsync(fd)
       ioctl(FS_IOC_FIEMAP)
       拒绝 UNKNOWN/DELALLOC/UNWRITTEN/SHARED/... extent
       Extent {
         logical_offset = fe_logical,
         device_offset  = fe_physical + namespace_base_bytes,
         length         = fe_length
       }
       校验 extents 从 0 连续覆盖完整 file_size
  -> ResolvedTarget(Ext4LocalNvmePayload, fd lease)
  -> LocalNvmeDataPath::open()
       byte extent -> LbaExtent {
         start_lba           = device_offset / namespace_block_size,
         length_blocks       = length / namespace_block_size,
         logical_offset_bytes= logical_offset
       }
       构造 DeviceTargetHandle
       inline 上传前 8 个 extent；更多 extent 放 GPU overflow buffer
```

resolver 与 DataPath 位置：`tutti/resolvers/local_file/resolver.h:220`、`tutti/resolvers/local_file/resolver.h:297`、`tutti/resolvers/local_file/resolver.h:467`、`tutti/data_paths/local_nvme/local_nvme_data_path.cpp:685`、`tutti/data_paths/local_nvme/local_nvme_data_path.cpp:738`。

对 file target 逻辑 offset `x`，若它位于 extent `e`：

```text
e.logical_offset <= x < e.logical_offset + e.length

namespace_byte(x)
  = e.device_offset + (x - e.logical_offset)

SLBA(x)
  = namespace_byte(x) / namespace_block_size
```

一条长度 `n` 的 NVMe command 要求 `x`、`n` 都按 namespace block size 对齐，并且整个 `[x,x+n)` 位于同一 FIEMAP extent。command 的逻辑块数为：

```text
n_blocks = n / namespace_block_size
```

`nvm_cmd_rw_blks()` 最终把 `n_blocks - 1` 编入 NVMe SQE 的 NLB 字段：`tutti/device_manager/nvme/libnvm/include/nvm_cmd.h:92`。

动态 submit 的 host fan-out 先执行：

```text
sub_io = min(remaining, effective_MDTS, current_extent_remaining)
```

再为每个 sub-I/O 生成一个 `DeviceSubmitEntry`：`tutti/data_paths/local_nvme/local_nvme_data_path.cpp:1415`。GPU 上一个 thread 消费一个 entry：

```text
submit_one_kernel(thread i)
  -> AddressDescriptor {prp1, prp2, data_length}
  -> submit_read_one()/submit_write_one()
  -> resolve_lba(DeviceTargetHandle, target_offset, data_length)
  -> 遍历 GPU extent table
  -> starting_lba + n_blocks
  -> acquire_queue()
  -> nvm_cmd_header()/nvm_cmd_data_ptr()/nvm_cmd_rw_blks()
  -> sq_enqueue() / doorbell
  -> cq_poll_bounded()
  -> EntryCompletionStatus
```

位置：`tutti/data_paths/local_nvme/io/submit_one.cuh:103`、`tutti/data_paths/local_nvme/io/nvme_submit_primitives.cuh:118`、`tutti/data_paths/local_nvme/io/nvme_submit_primitives.cuh:201`。

GPU `resolve_lba()` 没有保存每个 extent 的 `logical_offset_bytes`，而是用前面 extent 的 `length_blocks` 累加 logical cursor。这在当前 resolver 强制“从 0 开始、无洞、无重叠”的前提下与上述公式等价；如果以后允许 sparse target，这个 device handle 必须扩展，不能继续依赖累加。

当前 preset 把 `namespace_base_bytes` 固定传 0：`tutti/presets/local_nvme_preset.cpp:114`。所以当前安全部署假设是 filesystem 直接位于 namespace/whole-disk 映射上。如果 backing path 是 partition 而 FIEMAP `fe_physical` 相对 partition，必须把 partition start byte 正确传入；现有 preset API 没有接通这个配置。

### 5.10 striped target 的两级映射

设：

```text
D   = shard 数
U   = stripe_unit bytes
rot = chunk_key 前 4 bytes 的 little-endian 值 mod D
x   = striped target 逻辑 byte offset
```

第一层把 virtual target offset 映射到 shard：

```text
stripe_index = floor(x / U)
shard        = (stripe_index + rot) mod D
shard_offset = floor(x / (U * D)) * U + (x mod U)
```

第二层在该 shard 的 file target 内执行上一节的 FIEMAP 映射：

```text
shard_namespace_byte
  = shard_extent.device_offset
    + (shard_offset - shard_extent.logical_offset)

SLBA = shard_namespace_byte / namespace_block_size
```

host 在 `StripedDataPath::submit()` 中按以下三个边界共同切分：

```text
sub_io = min(
  remaining,
  stripe_unit_remaining,
  effective_MDTS,
  current_shard_extent_remaining,
)
```

然后一次 H2D 上传所有 shard entries，一次 `fused_submit_kernel` launch；GPU 仍是一个 thread 处理一个切分后的 entry，并在对应 shard 的 `DeviceTargetHandle` 上调用同一个 `resolve_lba()`：`tutti/data_paths/striped_local_nvme/striped_data_path.cpp:1151`、`tutti/data_paths/striped_local_nvme/fused_submit_kernel.cuh:95`。

striped backing file 的容量不是简单 `F / D`。`StripedLayout.prepare_put()` 为保证所有 shard 都有完整 stripe round，计算：

```text
logical_need       = N * S
rounds             = ceil(logical_need / (D * U))
physical_per_shard = max(U, rounds * U)
allocated_per_chunk= D * physical_per_shard
visible_target_size= D * floor(min_shard_size / U) * U
```

位置：`integration/vllm-connector/stores/tutti_nvme/striped_layout.py:177`。因此最后一轮 padding 可能没有 semantic payload，但属于 target 可见范围；实际 layer I/O 仍只访问 `[0,F)`。

### 5.11 staging memory 到 NVMe PRP 的注册和寻址

当前预期调用路径：

```text
TuttiKVStore.register_buffer(staging, granularity=S)
  -> _buffer_info() 得到 base/address、logical size M、kind、accel id
  -> Python Runtime.register_memory(..., io_granularity=S)
  -> pybind RuntimeWrapper::register_memory()
  -> MemoryView {address, size=M, kind=DEVICE, io_granularity=S}
  -> StorageRuntime::register_memory()
       建 MemoryHandle/MemoryEntry
       注意：此时尚未调用具体 DataPath 注册

第一次对某 DataPath/registration domain submit
  -> StorageRuntime::registration_for_()
  -> DataPathMemoryView
  -> LocalNvmeDataPath::register_memory()
       cudaPointerGetAttributes()
       校验 base % 64KiB == 0
       nvm_dma_map_data_device(base, M)
       snvme 以 64KiB GPU pin granule 注册
       libnvm 展开为 controller MPS 粒度 ioaddrs[]
       [预期] io_granularity=S 时预建 AddressDescriptor[]/PRP list
```

位置：`integration/vllm-connector/stores/tutti_nvme/store.py:341`、`integration/vllm-connector/bindings/python/src/_core.cpp:105`、`tutti/include/tutti/storage_runtime.h:434`、`tutti/include/tutti/storage_runtime.h:1433`、`tutti/data_paths/local_nvme/local_nvme_data_path.cpp:920`。

内存侧三种粒度的关系是：

```text
GPU pin granule = 64 KiB
  约束 registration base；libnvm 会把 registration size 向上覆盖整页

controller MPS = ctrl->page_size，常见 4 KiB
  决定 nvm_dma_t.ioaddrs[]、PRP1/PRP2 和 PRP-list entry 的页粒度

namespace block size = configured NamespaceIdentity.block_size，常见 512B/4KiB
  决定 target offset、length 和 NVMe SLBA/NLB 的合法粒度
```

对 staging slot `r`：

```text
memory_offset = r * S
start_prp_page = memory_offset / controller_MPS
pages_in_io = ceil(sub_io / controller_MPS)

1 page : PRP1 = ioaddrs[start_prp_page], PRP2 = 0
2 pages: PRP1 = ioaddrs[start_prp_page], PRP2 = ioaddrs[start_prp_page + 1]
>2 pages:
  PRP1 = first data page IOVA
  PRP2 = PRP-list page IOVA
  PRP-list = remaining data page IOVAs
```

动态路径在 host 构造 `AddressDescriptor`；PRP list来自host-pinned、DMA-mapped `PrpPageCache`，cache disabled/miss/exhausted时由可增长`PrpBufPool`提供。NVMe controller直接读取host IOVA。旧`MetadataArena` GPU PRP pool和PRP H2D fallback已经删除；entry/descriptor仍需H2D后launch I/O kernel。

注册期预建路径已经接通：Python只传逻辑block大小`S`作为`io_granularity`；C++从驱动`ioctl_get_dev_info()`读取MDTS，保持`bytes_per_slice=S`并计算`ios_per_slice=ceil(S/MDTS)`。每个LIST sub-I/O预建一个完整4 KiB host PRP page，page entries指向GPU data page IOVA，descriptor的`prp2`指向host page IOVA。submit按`slice_idx/sub_idx`取表；extent截短时只对该fragment回退dynamic。`StorageRuntime::MemoryEntry`已保存并恢复granularity，`d_prp_gpu`和256B子页打包均已删除。

“GPU 自己调度 I/O”的当前准确含义也应限定：CPU 仍负责 resolver、extent/MDTS 切分、PRP/entry 构造和 H2D；GPU kernel 中一个 thread 负责一个 sub-I/O entry 的 LBA resolve、SQE enqueue 和 CQ poll。它还不是 GPU 根据一个高层 segment/DAG 自己生成全部 sub-I/O。

### 5.12 两个具体几何示例

**示例 A：普通 GQA，单 vLLM block 已是 64 KiB。**

假设：

```text
T=16 tokens
num_kv_heads=8
head_size_k=head_size_v=128
dtype=FP16/BF16=2 bytes
N=32 layers
C=256 tokens
```

则：

```text
B = 16 * 8 * (128 + 128) * 2 = 65,536 B = 64 KiB
P = 256 / 16 = 16 blocks/chunk
S = 16 * 64 KiB = 1 MiB/layer segment
F = 32 * 1 MiB = 32 MiB/chunk file

layer 7 target_offset = 7 MiB
slot 3 memory_offset  = 3 MiB
request length        = 1 MiB
```

若 MDTS 为 128 KiB 且该 layer segment 不遇到更短的 FIEMAP extent，DataPath 会切成 8 个 GPU I/O entries。每个 entry 在 4 KiB namespace block 下是 32 blocks，SQE NLB 字段写 31。

这个模型下一个 vLLM block 已经是 64 KiB，direct 很可能有价值；但还要检查 raw packed layout 是否与 canonical disk layout一致、block 地址是否连续、descriptor 数是否可接受。

**示例 B：MLA 单 block 是 18 KiB，必须聚合。**

假设无 compression/padding：

```text
T=16 tokens
latent head_size=576
num_heads=1
dtype=2 bytes
```

则：

```text
B = 16 * 576 * 2 = 18,432 B = 18 KiB
```

单 block 不是 4 KiB 倍数，不能作为当前 store 的一条 I/O。按 4 KiB 合法性求最小聚合数：

```text
P_align = 4096 / gcd(4096, 18432) = 2
C       = 2 * 16 = 32 tokens
S       = 2 * 18 KiB = 36 KiB = 9 * 4 KiB
```

36 KiB 已合法，但可能仍小于高效 I/O 阈值。如果希望 `S` 同时为 64 KiB 的倍数，最小 `P` 是 32，得到 `C=512 tokens`、`S=576 KiB`。这说明 staging chunk size 应由 `B`、合法 alignment 和性能阈值共同求出，而不是固定看到 `block_size=16` 就采用某个 magic size。

当前实现尚未自动做这个推导。`_resolve_geometry()` 只校验 `chunk_tokens % block_size == 0`，而 store 到注册时才拒绝 `S % 4096 != 0`。所以示例 B 若仍配置 `C=16`，会由 `register_buffer()` 返回 `None`，随后 engine 只抛出包含 granularity 的泛化 `RuntimeError`，并没有自动合并到两个 block，也没有保留 store 拒绝的具体 capability 原因。

### 5.13 与 InfiniKV 的内存/文件/LBA 映射对照

| 映射层 | 当前 Tutti staged | 当前 Tutti direct 目标 | InfiniKV |
|---|---|---|---|
| vLLM pool | block table -> gather/scatter | paged span descriptor，尚未贯通 | 直接注册旧式 K/V paged tensor |
| I/O 粒度 | 一个 canonical layer segment `S` | 大且对齐的 block run | `gpu_file_shape[2]`/per-block granularity |
| GPU memory map | 一块 64 KiB base-aligned staging allocation | 应注册整个 KV pool/layout spans | 每个 tensor 建 `geminifs_dma`，granule pointer 映射到同一 DMA context |
| PRP | host-pinned `PrpBufPool`/`PrpPageCache`；prebuilt和dynamic均无GPU PRP backing | 已接通并删除GPU fallback | `performDMASlicing()` + `initializePRPEntries()` + GPU PRP mappings |
| disk object | 一个 semantic chunk file，层 offset=`L*S` | 应共享相同 canonical object | 一个 GPU file，shape 近似 `[K/V, layer, kv_size]` |
| file -> device | resolver FIEMAP payload，host 按 extent 切分，GPU 再 resolve | 同左 | GPU `NVMe_File::__get_nvmeofst()` 直接遍历 file extents |
| striped | host stripe/extent/MDTS 切分，一次 fused kernel | 同左 | 自身 controller/file abstraction，不具备 Tutti 的同等 target SPI |
| metadata | 16B chunk key + 18B layer key + durable marker | 应复用同一语义 | 进程 dict + GPUFileMetadata，恢复/发布较弱 |

InfiniKV 的 memory registration 调用链已经在第 7.2 节列出。其关键效果是整个 paged KV tensor 的 direct mapping。Tutti 自己已经完成 host-pinned PRP pool/cache 和注册期 descriptor build；被 runtime granularity 阻断的是 connector 选中该 fast path，而不是底层 PRP backing 能力。真正仍可借鉴的是直接注册 paged KV pool及其 span mapping。

InfiniKV 的 file mapping 更直接：

```text
GeminiFSBackend:
  kv_size = chunk_tokens * num_heads * head_size * dtype_bytes
  gpu_file_shape = [1 or 2, num_layers, kv_size]
  per_file_size = kv_dim * num_layers * kv_size

GPU NVMe_File::nvme_xfer(file_offset, nbytes):
  -> __get_nvmeofst(file_offset)
  -> 遍历 geminiFS_hdr.extents
  -> fe_physical + offset_in_extent
  -> starting_lba = nvme_offset >> block_size_log
  -> n_blocks = nbytes >> block_size_log
  -> issue_nvme_cmd()/poll()
```

位置：`third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/storage_backend/geminifs_backend.py:94`、`third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/include/file.cuh:67`、`third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/include/file.cuh:115`。

InfiniKV 的直接映射有两个 Tutti 不应复制的问题：

1. `NVMe_File::nvme_xfer()` 只用起始 `file_offset` 找 extent，没有验证 `[file_offset,file_offset+nbytes)` 是否跨 extent，也没有切分。若文件逻辑连续但物理 extent 不连续，命令会从第一个 extent 的 LBA 连续越界，读写错误物理区域。当前 Tutti dynamic path 会按 extent boundary 切分；Tutti local 预建分支则存在同类问题。
2. InfiniKV 使用 raw K/V tensor/page 的 direct layout，缺少 Tutti staged path 的 canonical token-major reshape。对于当前 packed 4D/HND/cross-layer vLLM pool，不能只换地址公式；必须先定义 disk canonical layout，并仅在 raw spans 与它字节同构时 direct。

### 5.14 当前映射链路的可见 bug 与必须修改项

1. **P0：staging segment 没有自动对齐求解。** `_resolve_geometry()` 只保证 token block 整数倍，不保证 `S` 是 Runtime alignment 的倍数；store 在更晚的 `register_buffer()` 才硬编码按 4 KiB 拒绝，engine 只得到泛化失败。应在 bind 前根据 cache spec 和 Runtime capabilities 自动求 `P/C/S`，或给出包含具体 capability 的早期配置错误。
2. **P0：page-level padding 被当成 per-token bytes。** `PagedTransferHooks` 用 `row_bytes=S/C=B/T` 建 token-major staging shape。若 padded `B` 不能被 `T` 整除，聚合更多 block 也无法修复，bind 直接失败；若能整除，LMCache kernel 只写真实 head/state bytes，额外 staging columns 没有初始化，save 会把 `torch.empty` 中的旧数据作为 padding 落盘。应把 `real_page_size_bytes`、`page_size_bytes` 和 per-block padding 分开描述；gather 显式 zero-fill canonical padding，scatter 忽略它，不能把 page padding平均摊到 token row。
3. **已修复：`io_granularity`贯通StorageRuntime。** Python/store传入逻辑block大小`S`，`MemoryEntry`保存并在惰性DataPath注册时恢复。
4. **已修复：local prebuilt与FIEMAP/MDTS切分协同。** 预建表只描述memory PRP；submit仍按driver MDTS和target extent生成entry，只有sub-I/O形状与预建descriptor一致时走fast path，extent短fragment回退dynamic。
5. **P0：DataPath 声明的 memory alignment 与 PRP 寻址实现不一致。** capability 把 memory alignment 设为 namespace `block_size_`，但动态路径用 `start_page = memory_offset / ctrl->page_size` 且丢弃页内 remainder。若 namespace block=512B、MPS=4KiB，512B 对齐的 memory offset 会通过校验，却从错误的 4 KiB 页首 DMA。当前 connector 的 `S % 4096 == 0` 在常见 MPS=4KiB 时掩盖了问题。应把 memory offset alignment 声明为 MPS，或正确给 PRP1 加页内 offset并按 NVMe PRP 规则计算跨页数。
6. **P1：staging backing allocation 没有显式覆盖 DMA size rounding。** worker 只分配 `M + 64KiB` 来移动 base；`nvm_dma_map_data_device()` 又把注册 size `M` 向上取整到 64 KiB pin pages。`alignment_skip + round_up(M,64KiB)` 可能大于 `M + 64KiB`，静态 contract 不能保证映射尾部仍属于原 tensor allocation。应分配 `round_up(M,64KiB) + 2*64KiB`，对齐后保留完整 rounded span，并把 logical usable size 与 registered backing size 分开记录。
7. **P1：4 KiB 被硬编码成模糊的 `_IO_PAGE_BYTES`。** 它目前同时承担 connector length alignment 和 stripe unit alignment，但没有来自 DataPath capabilities；不能代表 GPU pin page、controller MPS 或所有 namespace LBA。应改名为 capability-derived `required_io_alignment_bytes = lcm(target_alignment, memory_alignment, length_alignment)`，并显式记录每个来源。
8. **P1：namespace block size 依赖配置值，没有和硬件 identify 结果做显式一致性校验。** preset 同时把同一个 `device.block_size` 注入 resolver 和 DataPath，所以软件内部会自洽；但 queue 的 `disk_info.block_size` 也被该配置覆盖。配置错误会直接产生错误 SLBA/NLB。应从 controller/namespace identify 获取权威 LBA size，再验证配置。
9. **P1：partition base 没有从 preset 接通。** resolver 支持 `namespace_base_bytes`，但 local/striped preset 都传 0。非 whole-namespace filesystem 需要可靠计算 partition start，否则 `fe_physical -> namespace byte` 映射错误。
10. **P1：target ticket 只按 URI/size 失效，未验证 extent generation。** fd lease 能保持 inode 存活，但不能证明 ext4 extent 永不被 defrag、reflink、truncate 或外部 writer 改变。direct NVMe I/O 又绕过 filesystem page cache。部署必须禁止外部修改，并建议保存 inode/device/extent signature，在提交或重开时校验；长期方案应使用预分配、不可移动的 extent pool 或 raw namespace allocator。
11. **P1：direct 与 staged 的 canonical disk bytes 尚未定义成正式 ABI。** 当前 staged segment 是 token-major reshape，future direct 若直接写 raw packed vLLM page，二者可能对同一 chunk key 产生不同字节布局。namespace manifest 必须包含 layout version/spec hash，direct eligibility 必须证明 raw span 与 canonical layout同构，否则仍走 staged。
12. **P1：当前几何只支持一个 cache group、所有层相同 `B`。** MLA/linear hybrid 模型可能有多个 group 或不同 state page。目标 metadata 应使用 `group_id/layer_id -> B/S/offset` 表，而不是固定 `offset=L*S`；文件 header/manifest 需携带 per-layer offset table 或按 group 分对象。

建议新增一个统一、不可歧义的注册产物：

```text
StorageGeometry
  cache_group_id
  tokens_per_vllm_block
  bytes_per_vllm_page_per_layer
  blocks_per_semantic_chunk
  bytes_per_layer_segment
  canonical_layout_id
  layer_offsets[]
  target_alignment_bytes
  namespace_lba_bytes
  controller_mps_bytes
  gpu_pin_granule_bytes
  registered_base
  registered_bytes_rounded
  usable_bytes
  extent_signature/generation
```

这个对象应同时供 path selector、staging allocator、store layout、Runtime registration 和 StepPlan 编译使用，避免每层各自重新解释 `page`、`block` 和 `granularity`。

## 6. 当前 metadata 管理

### 6.1 语义键与层 I/O 键

`integration/vllm-connector/index/chunk_index.py:15` 定义 deterministic chunk key，语义上把 token prefix/chunk 与物理文件分离。每层 store I/O 再基于 chunk key 派生 layer key。

优点：

- 相同 token chunk 可稳定命中。
- `ChunkIndex` 可以独立管理 `ABSENT/WRITING/RESIDENT/EVICTING` 等状态。
- store 可以更换而不改变 scheduler 语义。

风险：

- namespace 没有显式编码 rank；当前依赖不同 root/device 做物理隔离。若多个 rank 配置到相同 root，会发生键冲突。
- engine cache key 没有覆盖全部 store/options/num_layers 配置，错误复用 engine 的风险较高。

### 6.2 Load metadata 生命周期

```text
scheduler lookup prefix
  -> 返回 external token count
  -> allocate_slots
  -> update_state_after_alloc 保存 block ids/token interval
  -> build_connector_meta 生成 worker metadata
  -> worker start_load_kv 消费 metadata
  -> scheduler 提前 confirm_loaded/更新 index
```

问题在于 scheduler 的 `confirm_loaded()` 是乐观更新，发生在 worker I/O 真正成功之前。若 worker load 失败，scheduler 仍可能把不存在或损坏的数据视为 resident。

### 6.3 Store metadata 生命周期

```text
save_kv_layer
  -> prepare marker/target
  -> index 状态进入 WRITING
  -> 每层 put completion
  -> KVEngine.settle()
  -> 所有层成功后 commit marker/index RESIDENT
```

设计上应保证“全部层成功才发布”。当前实现有以下破坏原子性的路径：

- `plan_store()` 可能在 replacement 完成前先丢弃旧物理数据：`integration/vllm-connector/engine/core.py:193`。
- `sync()` 对从未真正写入的 optimistic entry 清理不完整：`integration/vllm-connector/engine/core.py:489`。
- I/O 失败更多表现为 worker 异常，没有可靠反馈到 scheduler invalid block 和 metadata rollback。

## 7. InfiniKV 的逐层调用路径

### 7.1 顶层接口与角色拆分

```text
vLLM KV connector API
  -> third_pkgs/infinikv-for-ae/vllm-infinikv/vllm/distributed/kv_transfer/kv_connector/v1/infinikv_connector.py
  -> third_pkgs/infinikv-for-ae/infinikv/infinikv/integration/vllm/vllm_v1_adapter.py
  -> InfiniKVEngine
  -> GeminiFSBackend
  -> torch custom op
  -> pybind/CUDA
  -> GeminiFS
  -> GPUController
```

顶层 wrapper：

- 强制使用 piecewise CUDA graph 模式：`third_pkgs/infinikv-for-ae/vllm-infinikv/vllm/distributed/kv_transfer/kv_connector/v1/infinikv_connector.py:36`。
- 依据 64 KiB 对 block 数做对齐：`third_pkgs/infinikv-for-ae/vllm-infinikv/vllm/distributed/kv_transfer/kv_connector/v1/infinikv_connector.py:60`。
- scheduler 只创建 lookup client；worker 才创建 engine、server、stream：`third_pkgs/infinikv-for-ae/infinikv/infinikv/integration/vllm/vllm_v1_adapter.py:538`。

这一点比当前 Tutti 更符合 scheduler metadata-only、worker data-plane-only 的目标。

### 7.2 InfiniKV 内存注册完整路径

```text
gpu_model_runner 根据 prefer_cross_layer_blocks=True
  -> InfiniKVConnector.register_cross_layers_kv_cache()
  -> vllm_v1_adapter.register_cross_layers_kv_cache()
  -> InfiniKVEngine.register_cross_layers_kv_cache()
  -> GeminiFSBackend.register_single_tensor()
  -> _custom_ops.register_tensor_with_gpu()
  -> geminifs_pybind.cu binding
  -> GeminiFS::registerTensor()
  -> GPUController::registerTensor()
  -> GPUController::createDMAContext()
  -> granule pointer map
  -> DMA page mapping
  -> PRP 初始化
  -> mapping/descriptor copy 到 GPU
```

关键位置：

- `third_pkgs/infinikv-for-ae/vllm-infinikv/vllm/distributed/kv_transfer/kv_connector/v1/infinikv_connector.py:137`
- `third_pkgs/infinikv-for-ae/infinikv/infinikv/integration/vllm/vllm_v1_adapter.py:793`
- `third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/infinikv_engine.py:110`
- `third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/storage_backend/geminifs_backend.py:188`
- `third_pkgs/infinikv-for-ae/infinikv/infinikv/_custom_ops.py:57`
- `third_pkgs/infinikv-for-ae/infinikv/csrc/geminifs_pybind.cu:106`
- `third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/geminifs.cu:142`
- `third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/gpu_controller.cu:347`
- `third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/gpu_controller.cu:886`

#### 这一层抽象的效果

InfiniKV 当前活跃路径把整个 cross-layer GPU tensor 一次注册，再按 `gpu_file_shape[2]` granularity 建立 DMA granule 和 PRP/IO mapping。后续 paged read/write 可以直接引用共享 KV pool，不必先经过 contiguous staging buffer。这是它相对当前 Tutti 最值得借鉴的部分；逐层 tensor 的 batch registration 仍保留 Python 调用，但对应 pybind binding 被注释，不是可用主路径。

### 7.3 InfiniKV load 路径

```text
vLLM execute_model
  -> InfiniKVConnector.start_load_kv()
  -> vllm_v1_adapter.start_load_kv()
  -> prepare_load_kv()
  -> 构造 IOSpec / block mappings / per-layer metadata

attention layer 前
  -> wait_for_layer_load(layer)
  -> 当前 CUDA stream 上 retrieve/load 下一层
  -> InfiniKVEngine.load_kv_cache()/prepare load
  -> GeminiFSBackend.batched_read()
  -> _custom_ops.geminifs_batched_read()
  -> pybind
  -> GeminiFS::geminifs_batched_xfer()
  -> 构造 GPUIoContext
  -> H2D mappings/context
  -> nvme_batch_xfer_kernel<<<blocks, 32>>>()
  -> 每个 GPU thread 处理一个 mapping
  -> file.cuh/helper.cuh 提交 NVMe command
  -> host callback/completion
```

关键位置：

- `third_pkgs/infinikv-for-ae/infinikv/infinikv/integration/vllm/vllm_v1_adapter.py:803`
- `third_pkgs/infinikv-for-ae/infinikv/infinikv/integration/vllm/vllm_v1_adapter.py:826`
- `third_pkgs/infinikv-for-ae/infinikv/infinikv/integration/vllm/vllm_v1_adapter.py:1021`
- `third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/storage_backend/geminifs_backend.py:249`
- `third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/geminifs.cu:269`
- `third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/gpu_controller.cu:1626`
- `third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/include/file.cuh:115`
- `third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/include/helper.cuh:15`

#### 实际效果

InfiniKV 的现有 explicit K/V 数据面是 direct paged I/O，不经过 Tutti 的 staging reshape；这对尺寸足够大、已对齐的 extent 有优势，但不能解决小 block 合并和任意 layout reshape。其 active 路径仍在每层 callback 中发起/等待，并且使用当前 stream，仍然不能形成一次提交的多层 DAG。

### 7.4 InfiniKV save 路径

预期路径为：

```text
start/prepare save metadata
  -> attention layer 后 save_kv_layer()
  -> GeminiFSBackend.batched_write()
  -> geminifs_batched_xfer(write)
  -> nvme_batch_xfer_kernel
```

但当前静态代码中，成功准备 save 后没有把 stream 写入：

```text
connector_metadata.save_stream = stream
```

`save_kv_layer()` 又以 `save_stream is None` 作为直接返回条件。因此当前主路径的 save 很可能整体被禁用：

- `third_pkgs/infinikv-for-ae/infinikv/infinikv/integration/vllm/vllm_v1_adapter.py:969`
- `third_pkgs/infinikv-for-ae/infinikv/infinikv/integration/vllm/vllm_v1_adapter.py:1074`

文件中另有 `launch_io()` 异步路径，但没有发现有效调用者：

- `third_pkgs/infinikv-for-ae/infinikv/infinikv/integration/vllm/vllm_v1_adapter.py:1098`
- `third_pkgs/infinikv-for-ae/infinikv/infinikv/integration/vllm/vllm_v1_adapter.py:1268`

### 7.5 InfiniKV metadata

InfiniKV 的 metadata 主要是：

- request tracker 中的 block/token 状态。
- IOSpec 和 GPUFileMetadata。
- backend 内进程级字典和 lookup side effect。
- 文件名/offset 到 GPU page mapping。

关键问题：

- dict snapshot restore 被注释，重启恢复不完整：`third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/storage_backend/geminifs_backend.py:89`。
- engine 的 `close()` 会委托 backend，但 backend `close()` 仍留下 `TODO: Free GeminiFS`；`destroy()` 也缺少生命周期单元测试：`third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/storage_backend/geminifs_backend.py:647`、`third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/infinikv_engine.py:575`。
- lookup 的 pin 语义向下传递，但 backend `contains()` 没有真正执行 pin：`third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/infinikv_engine.py:408`、`third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/storage_backend/geminifs_backend.py:225`。
- preemption 处理有显式 FIXME：`third_pkgs/infinikv-for-ae/infinikv/infinikv/integration/vllm/vllm_v1_adapter.py:258`。
- metadata 不具备当前 Tutti `ChunkIndex` 那样清晰的发布/回滚状态机。

### 7.6 InfiniKV 和当前 vLLM layout 不兼容

InfiniKV 自带的 vendored vLLM 使用旧式显式 K/V 维：

- FlashAttention：`third_pkgs/infinikv-for-ae/vllm-infinikv/vllm/v1/attention/backends/flash_attn.py:112`
- FlashInfer：`third_pkgs/infinikv-for-ae/vllm-infinikv/vllm/v1/attention/backends/flashinfer.py:309`

形态接近：

```text
FlashAttention: [2, block, token, head, dim]
FlashInfer:      [block, 2, token, head, dim]
```

当前 Tutti vendored vLLM 是 packed 4D/cross-layer pool，因此 InfiniKV 的 layout 和 binding 代码不能原样移植。可借鉴的是 DMA registration、direct paged descriptor 和 PRP 生命周期，而不是 tensor axis 假设。

## 8. 采用当前重构而不是直接复制 InfiniKV 的原因

| 决策点 | 选择当前 Tutti 分层的原因 | 从 InfiniKV 借鉴的部分 |
|---|---|---|
| 语义与物理解耦 | `ChunkIndex`、store、transfer 分层更适合多 backend | 不复制其进程内 dict 作为唯一真相 |
| metadata 原子性 | 可建立 `WRITING -> RESIDENT` journal 状态机 | 借鉴 direct mapping，不借鉴 lookup side effect |
| 后端可替换 | Tutti runtime/data path SPI 已抽象 local NVMe | InfiniKV engine 硬编码 GeminiFS，不适合长期扩展 |
| 当前 vLLM 适配 | 应基于真实 packed layout 编译 descriptor | InfiniKV 面向旧 explicit K/V layout |
| direct I/O | 当前 Tutti 接口尚未贯通 | 借鉴注册期 DMA map + PRP + paged read/write |
| 错误传播 | Tutti runtime 有 per-entry/status 聚合基础 | InfiniKV GPU kernel 缺少可靠 per-entry status |
| 生命周期 | 当前可在 runtime/store 层定义 handle ownership | InfiniKV unregister/clear 会留下 mapping 清理问题 |

结论不是“当前实现已经优于 InfiniKV”，而是：

- 当前 Tutti 的**抽象基础更适合作为最终架构**。
- InfiniKV 的**direct GPU I/O 数据面成熟度更高**，但优势范围是满足 direct eligibility 的大且对齐的 extent。
- 最合理路径是把 InfiniKV 的注册期和 paged I/O 思路作为 Tutti 的 direct 分支，同时保留 staging ring 处理小块、非对齐和 reshape，而不是替换当前 index/store/engine 分层。

## 9. 当前 Tutti 的缺陷与修复状态

以下结论基于当前工作树。原始评审后已经修复的项目单独标为“已修复”，其余问题可以从静态调用顺序或接口契约确认；性能影响仍需后续运行验证。

### 已修复（原 P0-1）：load token 区间反推错误

当前 scheduler/worker 已改为显式传递区间：

- `update_state_after_alloc()` 在 vLLM `_update_after_schedule()` 增加本步 token 数之前，保存当前 `request.num_computed_tokens` 为 `load_start_token`。
- `build_connector_meta()` 同时传递 `load_start_token` 和 `load_tokens`。
- worker 直接消费 `[load_start_token, load_start_token + load_tokens)`，不再用 `computed - external` 反推。

位置：

- `third_pkgs/vllm/vllm/v1/core/sched/scheduler.py:1061`
- `third_pkgs/vllm/vllm/v1/core/sched/scheduler.py:1379`
- `integration/vllm-connector/adapter/connector.py:429`
- `integration/vllm-connector/adapter/connector.py:440`
- `integration/vllm-connector/adapter/connector.py:492`
- `integration/vllm-connector/adapter/worker.py:346`

剩余风险已经收敛到下一项的区间对齐和 cap 顺序，不再是起点使用旧值的问题。

### P0-2：对齐后再 cap，可能得到不可加载的 token 数

`get_num_new_matched_tokens()` 先在 `integration/vllm-connector/adapter/connector.py:417` 做 chunk 对齐，随后又在 `integration/vllm-connector/adapter/connector.py:420` 用 `max_tokens_per_load` 截断。如果该上限不是 `chunk_tokens` 的整数倍，最终承诺给 vLLM 的 external token 数不再按 chunk 对齐；worker `_load_chunk_span()` 只取完整 chunk，实际加载量可能少于 scheduler 承诺量。

修改：cap、block availability、chunk alignment 必须在一个函数中求交集，最终 metadata 只输出合法完整区间；尾块需显式 partial descriptor。

### P0-3：block table 长度缺少精确校验

当前路径主要检查 block table 是否存在，没有验证它是否覆盖最终 load token 区间。地址生成可能读取不足的 block id。

修改：编译 plan 时计算 required blocks，并检查每个 KV group/layer 的 block table 长度；不足时返回 invalid blocks，不进入 native kernel。

### P0-4：packed 4D/cross-layer NHD 仍未通用支持；HND 已通过 required-layout fail-fast

证据见第 5 节。NHD packed 4D 仍可能 unsupported；当前 connector 已 override `get_required_kvcache_layout()` 返回 `NHD`，因此不会再接受 HND 后把 head axis 当 layer。剩余问题是对真实 packed NHD 建立稳定 descriptor/native path，而不是继续猜 axis。

修改：去掉 rank-based 五格式推断，改为真实 sizes/strides/semantic axes 的 descriptor；新增 packed 4D 和 cross-layer NHD/HND native path。

### 已修复（保守边界，原 P0-4a）：DCP block/slot mapping 在支持前 fail-fast

当前几何按物理 `spec.block_size` 计算 `blocks_per_chunk`，worker 又用连续的 `block_id * block_size + offset` 生成 slot。vLLM 在 `decode_context_parallel_size > 1` 时，scheduler block size 是物理 block size 乘 DCP world size，worker slot 还依赖 DCP rank 和 `cp_kv_cache_interleave_size`。当前实现既没有这种映射，也没有把 DCP size/rank/interleave 放进持久 namespace，可能静默保存或恢复错误 token 的 KV。

位置：

- `integration/vllm-connector/adapter/connector.py:123`
- `integration/vllm-connector/adapter/worker.py:203`
- `third_pkgs/vllm/vllm/v1/core/single_type_kv_cache_manager.py:73`
- `third_pkgs/vllm/vllm/v1/worker/block_table.py:214`

当前已在 connector 构造、调用 `_engine_for()` 打开 store 之前拒绝 `decode_context_parallel_size != 1`。这消除了静默错误映射；完整支持仍需复用 vendored offloading connector 的 rank/interleave-aware canonical mapping，并升级 namespace schema。

### P0-4b：HMA/multi-group 未支持，Qwen 不能作为 Tutti 数据面验证模型

`TuttiConnectorV1` 不实现 `SupportsHMA`，`_resolve_geometry()` 又明确拒绝多个 cache group；`_flatten_blocks()` 对嵌套 block table 只取第一个 group。Qwen3.8-27B 在当前 vLLM 下形成 4 个 group（3 个 GDN/Mamba group 和 1 个 full-attention group），因此不能通过关闭 HMA 合并成单组，也不能用当前 connector 做真实 NVMe load/save 验证。

位置：

- `integration/vllm-connector/adapter/connector.py:46`
- `integration/vllm-connector/adapter/connector.py:85`
- `third_pkgs/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:85`

修复必须保留 `tuple[list[int], ...]` 的多组 block tables，按 group/spec 注册 raw page，并实现 all-groups completion/commit。完成前，`/data2/qwen` 只适合 `--without-tutti` 的模型和 profiler 生命周期迭代；真正 connector 数据面仍需使用受支持的单组模型。

### P0-5：`_submit_retry()` 把合法 partial commit 当失败

Tutti runtime 合约允许 status 非 OK 但返回有效 handle/部分已提交；调用方应继续 drain handle 并根据 committed count 重试剩余项。当前 store 在这种情况下直接 raise。

位置：

- `integration/vllm-connector/stores/tutti_nvme/store.py:590`
- `tutti/include/tutti/storage_runtime.h:861`
- `tutti/include/tutti/storage_runtime.h:1879`

效果：backpressure 下可能把已提交 I/O 视为全失败，既丢 completion，又可能重复提交或泄漏资源。

pybind 已返回 `io_handle + rejected indices + status`，不必新增 `committed_count` 才能修复。正确行为是：只要 `io_handle` 存在，就先保存并最终 drain/release 该 handle；再按 `rejected` 只重试未提交项。只有 `io_handle is None` 的 all-rejected 才按零接受处理。

### P1（未来异步化前必须修复）：`_PostCompletion.query()` 可在 scatter 完成前返回 true

`query()` 主要透传 inner I/O completion；inner 完成时 scatter 可能尚未 launch 或 event 尚未完成。

位置：`integration/vllm-connector/engine/core.py:22`。

效果：completion 的 `query()` 契约会提前报告完成，上层或未来异步路径若依据 query 发布 `KV_READY`，可能在 scatter 尚未执行时消费 KV。当前槽位覆盖和 layer callback 都使用 `wait()`，所以这不是已复现的当前 P0；但静态 bank/fence 改造不能继承该语义。

修改：completion 状态机至少区分 `IO_DONE -> COPY_SUBMITTED -> COPY_DONE`，`query()` 只有在最终 device fence 完成后才返回 true。

### P0-7：scheduler 乐观发布 store residency，worker 失败后不回滚

位置：

- `integration/vllm-connector/adapter/connector.py:506`
- `integration/vllm-connector/adapter/connector.py:516`
- `integration/vllm-connector/engine/core.py:489`

效果：后续请求可能命中实际不存在或只写入部分层的 chunk。

修改：scheduler 只创建 pending lease；worker 全部层成功后通过 completion journal/RPC 发布 `RESIDENT`。失败写 `FAILED/ABSENT` 并清除 lease。

### 已修复（原 P1 角色隔离）：scheduler metadata-only

`TuttiConnector` 现在在 role 判断后选择独立构造路径。

当前 scheduler 仅持有 `ChunkIndex`、容量 reservation、manifest/marker 文件扫描
状态；不会调用 runtime factory，也不暴露 buffer/I/O/stream 方法。worker 路径
仍使用原 `KVEngine + TuttiKVStore` 数据面。测试覆盖 runtime factory 0 调用、
角色对象不共享，以及 79/80 层 marker 仍 miss、补齐全部层后 6400-token prefix
可见。

### P0-9：I/O 失败没有可靠转成 vLLM invalid blocks

当前异常更多在 worker 线程 raise；scheduler 的 invalid block 更新存在但 connector 没有完整接通。save 失败也缺少完整 marker/index rollback。

相关 vLLM 位置：

- `third_pkgs/vllm/vllm/v1/core/sched/scheduler.py:1761`
- `third_pkgs/vllm/vllm/v1/core/sched/scheduler.py:2934`

修改：device/per-entry status -> worker step result -> scheduler invalidation/rollback 必须形成闭环，不能只依赖 Python exception。

### P0-10：replacement 可能先删除旧数据再完成新写入

位置：`integration/vllm-connector/engine/core.py:193`。

效果：新写失败时，旧的有效副本也已经丢失。

修改：使用新 generation 临时目标，全部层成功后原子切换 marker/index，再异步回收旧 generation。

## 10. 当前 Tutti 的其他高风险问题

### P1：初始化、生命周期和并发

- `_config_ready` 用 `is not None` 判断，但构造字段初始化为 `0`，从函数本身看会把未配置状态误判为 ready。正常 vLLM 构造路径会在 cache registration 前同步调用 `configure()`，所以当前生产可达性低；仍应改成正整数检查，避免测试替身或未来时序变化触发零几何绑定。
- `_engine_for()` 的二级 cache key 没有 store path/options，但它先按 `vllm_config` 对象身份分桶，正常 factory 生命周期又不会原地修改 extra config。初稿把它写成当前错误复用风险过强；保留为配置不可变性 hardening，不作为现有 regression。
- worker staging 和 runtime preset 硬编码/归一到 `cuda:0`。r5 已确认 ranks 1-3 的模型 compute 在 device 1-3，而 I/O kernel 在 device 0；这是当前真实的设备放置和跨卡流量问题。
- worker 目前只有一套同时服务 read/write 的 `RingWindow`。第一阶段应改成独立静态 read/write bank：单 bank 依赖同 stream 顺序安全复用，双 read bank 才用于 overlap `read[L+1]` 与 `scatter[L]`。connector 同时仍只支持单 KV cache group。
- `_TuttiCompletion._finish()` 在 release/callback 前先标 settled，其他线程可能观察到半完成状态：`integration/vllm-connector/stores/tutti_nvme/store.py:171`。
- 每 batch 创建一个 Python watcher 线程，负载高时线程数和调度开销不可控。
- binding 缺少成对的 unregister/target close/query/fence API，长期服务有资源泄漏风险。
- scheduler shutdown 没有明确关闭已缓存 engine。
- request state 清理路径不完整，abort/preemption 下可能保留 stale metadata。

### P1：store 和 runtime

- `prepare_put()` 首写可能在热路径 zero-fill 整个目标并 `fsync`：`integration/vllm-connector/stores/tutti_nvme/layout.py:160`。
- store capability 协商期待的字段与 pybind 返回的 numeric capabilities 不完全一致：`integration/vllm-connector/stores/tutti_nvme/store.py:319`、`integration/vllm-connector/bindings/python/src/_core.cpp:71`。
- direct mode API 已有 capability probe，但 generic runtime 没有 `create_direct_transfer`，现有 Tutti NVMe store 返回空能力：`integration/vllm-connector/stores/tutti_nvme/store.py:384`。
- runtime 已保存并恢复 `io_granularity`；connector staging注册命中prebuilt路径。
- local NVMe host descriptor buffer 使用 pageable allocation，H2D 和 host lifetime/性能不稳定：`tutti/data_paths/local_nvme/local_nvme_data_path.cpp:1742`。
- `_resolve_geometry()` 不根据 `page_size_bytes` 自动求 4 KiB 对齐 segment；MLA/linear state 的小 block 可能直到 `register_buffer()` 才被拒绝，见第 5.12、5.14 节。
- `page_size_padded` 被 `row_bytes=segment_bytes/chunk_tokens` 当作 per-token bytes；可能 bind 失败，或把未初始化 staging padding 落盘，见第 5.7、5.14 节。
- local NVMe prebuilt descriptor只描述memory PRP；target仍按FIEMAP extent/driver MDTS切分，短fragment回退dynamic。
- DataPath 对外声明 `memory_alignment_bytes=block_size_`，但 PRP page index 丢弃 MPS 页内 offset；namespace LBA 小于 controller MPS 时可寻址错误，见第 5.14 节。
- staging registration 的 64 KiB size rounding、preset 固定 `namespace_base_bytes=0`、配置 LBA size 未和硬件权威值校验，均需在 direct/staged 双路径上线前修复。

### P1：语义和命名

- adapter 被文档描述为 pure translation，但实际承担 engine cache、namespace、min/max/alignment 和 store 创建。
- namespace 不显式包含 rank，依赖配置隔离。
- `if keys` 一类重复条件和分支表明 request metadata 状态机仍未收敛。
- staged 和 direct abstraction 存在，但当前只有 staged 真正可用；selector 也只读全局 `direct_transfer` 开关，尚未按 `io_extent_bytes` 自动决策。

## 11. InfiniKV 的已确认缺陷与风险

### 已确认缺陷

1. Save 主路径没有设置 `save_stream`，`save_kv_layer()` 会直接返回，见第 7.4 节。
2. `createDMAContext()` 的 granularity 校验写成 `tensor_size > tensor_size`，条件恒 false：`third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/gpu_controller.cu:891`。
3. batch registration 的 pybind 路径被注释/未完整接通：`third_pkgs/infinikv-for-ae/infinikv/csrc/geminifs_pybind.cu:109`。
4. `unregisterTensorMemory()` 删除 context 时没有调用已存在的 `removeAllMappings()`，`geminifs_dma` 析构也不负责释放映射/DMA 资源：`third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/gpu_controller.cu:412`、`third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/include/geminifs_mem.h:99`。
5. MLA bindings 被注释；custom op 捕获异常后返回 `False`，backend 调用者又忽略该返回值：`third_pkgs/infinikv-for-ae/infinikv/csrc/geminifs_pybind.cu:91`、`third_pkgs/infinikv-for-ae/infinikv/infinikv/_custom_ops.py:191`、`third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/storage_backend/geminifs_backend.py:440`。
6. `insert_key` 构造 `GPUFileMetadata` 的参数与 dataclass 定义不一致：`third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/storage_backend/geminifs_backend.py:548`、`third_pkgs/infinikv-for-ae/infinikv/infinikv/utils.py:48`。
7. `get_finished()`/save wait 等接口仍返回空或 no-op：`third_pkgs/infinikv-for-ae/infinikv/infinikv/integration/vllm/vllm_v1_adapter.py:1141`。
8. `NVMe_File::nvme_xfer()` 只按起始 file offset 选择 extent，没有按 extent end 切分或校验整个 I/O；跨非连续 FIEMAP extent 时会把后半段发到错误 LBA：`third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/include/file.cuh:67`、`third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/include/file.cuh:115`。
9. backend `close()` 没有释放 GeminiFS/GPU DMA context，代码中仍有明确 TODO：`third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/storage_backend/geminifs_backend.py:647`。

### 高风险 GPU I/O 问题

- queue 选择公式看不到 `blockIdx`，多 block 时可能集中到同一 queue：`third_pkgs/infinikv-for-ae/infinikv/csrc/GeminiFS/libgeminifs/include/helper.cuh:15`。
- kernel 一线程处理一个 mapping，但缺少可靠 per-entry completion status。
- 安全检查失败时 GPU thread 可直接 return，host 仍可能收到整体成功。
- CQ status 没有形成完整的错误解释和 block invalidation。
- launch error check 和同步代码有被注释的部分。
- Python/backend 多处忽略底层 bool 返回值。
- file offset 到 LBA 的 device 映射缺少 extent boundary fan-out，并默认 `fe_physical` 可直接换算 namespace LBA；partition base 和 extent relocation 约束不清晰。

### 架构限制

- `InfiniKVEngine` 构造时硬编码 GeminiFS backend：`third_pkgs/infinikv-for-ae/infinikv/infinikv/v1/infinikv_engine.py:87`。
- active load 仍在逐层 callback 中使用当前 stream，计算和 I/O 容易串行。
- 没有 durable、原子、可恢复的 metadata journal。
- 面向旧 vLLM explicit K/V layout，无法直接用于当前 packed cross-layer pool。

## 12. 目标架构

### 12.1 必须保留的抽象边界

```text
vLLM adapter
  只翻译 request/block table/layout registration

SchedulerIndexClient
  只做 prefix lookup、lease、commit/rollback/invalidate

KVEngine
  编译 step plan；不拥有 vLLM 语义

GpuDagExecutor
  一次提交 read/copy/compute dependency/write dependency

TransferPlan
  描述 staged scatter/gather 或 direct spans

StorageGeometry
  统一 block/page/segment、canonical layout、layer offset、
  namespace LBA、controller MPS、GPU pin granule 和 extent generation

Store/Runtime
  管理 target、batch I/O、completion、backpressure

DataPath
  注册 DMA、预建 PRP、提交 GPU I/O kernel、解释 CQ/status
```

adapter 不应再创建 store 或缓存多种 engine；scheduler 不应打开 CUDA/runtime。

### 12.2 第一落地点：静态 read/write staging bank

在当前单 execution-step 数据面内，不需要动态 `RingWindow.acquire()` 才能保证复用安全。先把资源固定为：

```text
read_staging  owned by read_stream
write_staging owned by write_stream
compute       owned by vLLM compute_stream
```

read stream 顺序：

```text
read L -> scatter L -> record ready[L]
read L+1 -> scatter L+1 -> record ready[L+1]
```

同一 stream 天然有序，因此一个 read bank 可以在 `scatter L` 完成后安全复用；若要让下一层 read 与当前 scatter 并行，再静态配置两个 read bank，按 layer/wave 轮转。compute callback 只做 `compute_stream.wait_event(ready[L])`，不 host wait、不提交下一层 read。

write stream 顺序：

```text
record compute_done[L]
write_stream wait compute_done[L]
gather L -> write L
```

一个 write bank 可以靠 write stream 顺序安全复用。read/write bank 必须物理分离，否则两个 stream 仍可能覆盖同一 staging。这里的 bank 容量按“本 step 所有请求聚合后的最大 chunk wave”计算，不依赖 vLLM 一次只有一个请求；vLLM 可以在同一 step 批处理多个请求。

不能把当前 `acquire()` 直接删除：当前 scatter 是在 host `wait()` 后才 launch，read/write 也共用一个 ring。必须先把 scatter 排进 read stream、拆分 write stream/bank，并建立跨 compute stream 的 fence。普通 CUDA event 对“尚未 record 的未来 event”能否提前 wait 还必须按实际 CUDA 契约验证；若不能，write 第一版保留逐层 record-and-enqueue，read 仍可一次预排。

Runtime `max_in_flight_operations`/MetadataArena 和 handle release仍限制一次全量预排80层。当前实现不再尝试无限预排：worker自动读取Runtime窗口，在下一层submit前wait/release最老write completion；窗口4已完成80层真机写和Hy3 TP4。若目标改为一次性全量enqueue，仍需stream-ordered sequence submission或per-stream workspace。

#### `max_in_flight_operations` 的资源账与调参边界

它不是 KV cache/HBM 配额，而是每个 DataPath 同时保留的 I/O operation 数。当前 local/striped 实现把 arena 配成 `2 * max_in_flight_operations` 个 slot：一半给 IN_FLIGHT，一半给已 terminal 但 public handle 尚未 release 的 operation。每个 slot 预分配：

```text
DeviceSubmitEntry[ max_batch_entries ]       32 B/entry
EntryCompletionStatus[ max_batch_entries ]    8 B/entry
AddressDescriptor[ max_batch_entries ]       24 B/entry
CUDA event + allocator metadata              small
```

注册期prebuilt path的`PrpBufPool`和动态`PrpPageCache`都是host-pinned、DMA-mapped内存；cache miss/exhausted也从可增长host pool取得lease。NVMe controller直接读取其IOVA，不占HBM。`io_granularity`已贯通，正常connector staging READ/WRITE命中prebuilt path；local/striped初始化和submit的GPU PRP allocation counter均为0。旧的约128 MiB/rank GPU PRP预留已经删除，不能再纳入当前资源公式。

当前 4 是实现的保守窗口，不是硬件固定值。上层已从Runtime capability自动取值，并在达到窗口时回收最老completion，因此80层逐层save不再需要把旋钮放大。调到8/16只能改变允许保留的arena lease数量，仍应以HBM、arena init、NVMe queue occupancy和tail latency实测为依据；完整全层预排仍需要sequence lease/bulk handle，而不是无限放大该值。

上述是**当前实现成本**，不是目标架构的必要成本。第一性拆分应是：

```text
queued_ops_per_stream       # 已 enqueue，尚未执行，可以很多
executing_ops_per_device    # 真正占 SM/NVMe queues，由 stream/queue 限制，通常很少
unreaped_completion_records # 已提交但 host 尚未回收，只需小状态
```

当前 `max_in_flight_operations` 把三者绑定在“一 op 一整个 worst-case arena slot”上，因而过于保守。对单 read stream，同一时刻只有队首 I/O kernel 执行；同一个 GPU entry/descriptor/status workspace 可以按 stream 顺序执行 `H2D[i] -> kernel[i] -> compact_status[i] -> H2D[i+1]` 后复用。每个 queued op 只需独立的小 completion record、target/key 生命周期和错误归属，不应复制 `max_batch_entries` 大小的 workspace。

同理，`max_batch_entries=4096` 是一个 op fan-out 后的最坏上限，不是 r5 每层实际 entry 数。r5 SQLite 的 `submit_one_kernel` launch geometry 提供了直接证据：read launch 是 `gridX=13, blockX=16`，覆盖 204 个 chunk entries；最大的 write launch 是 `gridX=16, blockX=16`，上限 256 entries。其余 launch 是 `gridX=4`。当前硬件 contract 又确认 effective MDTS 是 2 MiB，因此该 workload 的 256 KiB layer segment 不发生 MDTS 二次切分。r5 实际最大量级不超过 256，配置 4096 至少过量 16 倍。目标实现应按实际 plan/stream workspace capacity 配置，并为 extent fragmentation 留显式余量，而不是用 `queued_ops * 4096` 做资源账。

按 stream 复用 status workspace 时必须增加一个步骤：在覆盖 `EntryCompletionStatus[]` 之前，将本 op 的 per-entry CQ/error 状态聚合到独立的小 `CompactCompletionRecord`。否则下一 op 的 memset/kernel 会覆盖前一 op 尚未被 host读取的错误。可选实现是 `aggregate_status_kernel -> compact device record -> event`；每个 queued op 只保留几十字节 compact record和一个 fence，不保留整份 entry/status array。

更重要的是 load 错误门控：CUDA event只表达“完成”，不表达“成功”。若预排 `read -> scatter -> ready`，NVMe失败时 scatter/ready仍会放行 compute，造成错误 KV 被消费。第一版必须选择并写清失败策略：

1. 保留逐层 host status gate，只优化 descriptor/stream和 lookahead；正确但不能完全去掉host关键路径。
2. device-side status guard，失败时 fail-stop 整个 CUDA execution/context；适合当前 `kv_load_failure_policy=fail`，但不能做 request级 recompute。
3. timeline + host recovery protocol：失败不发布 success timeline，host invalidates blocks并显式解除/替换等待；最完整也最复杂。

在选定错误门控前，不得仅凭 same-stream ordering 把 `wait_for_layer_load()` 改成无条件 event wait。

当前代码的另一个预 enqueue 障碍是 `h_entries`、`h_dynamic_descs` 和临时 PRP page 使用 pageable `std::vector`；代码明确注明相应 `cudaMemcpyAsync` 会阻塞 host 直到 copy 完成。即使 arena 改成按 stream，如果 host source 不改为 pinned/immutable plan，submission loop 仍无法在前序 kernel 后面快速排长队。静态 staging 路径应优先修通注册期 host-pinned PRP/immutable descriptor；动态 descriptor 则使用 per-stream pinned host workspace或 device-side plan expansion。

当前 production local DataPath 的 CUDA 构造路径实际声明 `supports_multi_stream=true`、`max_concurrent_streams=2`，并已有双 stream contract；初始 skeleton 构造器的 `false` 不能代表真机配置。真正的上层缺口是 `TuttiKVStore` 只有一个 `_io_stream`，get/put 共用它，尚未把 read/write stream 分开注入。因此“一条 read stream + 一条 write stream”主要需要 store/engine stream routing，而不是修改 local DataPath capability。

从第一性原理看，`max_in_flight_operations` 也不应被解释为“stream 中能 enqueue 的工作数”。同一 stream 上的 kernel 会按序执行；可以把多个 op 排进 stream，但每个 op 当前仍持有独立的 entry/status/descriptor/event lease，直到 DataPath progress 观察到 terminal。现有限制是实现的 lifetime/admission 设计，不是 CUDA stream 的物理限制。若改为按 stream 分配一个可复用 workspace，并保证 `submit[i] -> fence[i] -> workspace reuse` 的顺序，就可以用很少的 arena slot 支持更长的 stream queue。

### 12.3 可选的 Step 级 API

建议形成两个核心接口：

```text
StepPlan KVEngine.compile_step(StepMetadata metadata)
LayerFenceSet GpuDagExecutor.submit_step(const StepPlan& plan)
```

`StepMetadata` 至少包含：

- request id / generation
- 精确的 load token interval 和 save token interval
- 每个 KV group 的 block table
- registered layout id
- layer range
- chunk key / layer object key
- read/write policy
- failure policy

`StepPlan` 是长期注册模板加本 step 小量动态参数的组合：

- device descriptor ranges
- per-layer read/write entry offsets
- copy/scatter/gather arguments
- compute dependency slots
- device status array
- commit token/generation

### 12.4 大块、对齐 extent 的 Direct GPU DAG

```text
read_io[L]
  -> kv_ready[L]
  -> compute[L]
  -> compute_done[L]
  -> write_io[L]

并行窗口：
read_io[L+1] || compute[L] || write_io[L-1]
```

满足 direct eligibility 的 extent 不需要 staging copy。read/write descriptor 直接引用已注册 KV pool 的 paged spans。block token 数较大只会提高满足条件的概率，最终仍以实际 extent 字节数、对齐、layout 和 descriptor 数量为准。

### 12.5 小块、非对齐或需 reshape 的 Staged GPU DAG

```text
read_io[L]
  -> staging_ready[L]
  -> scatter[L]
  -> kv_ready[L]
  -> compute[L]
  -> compute_done[L]
  -> gather[L]
  -> write_io[L]
```

staging 是小块和 reshape 场景的正式 fast path，不是失败兜底。优先使用上一节的独立静态 bank；若保留 ring，load slot 复用必须依赖 scatter/reshape 最终 device fence，save slot 复用必须依赖 write completion，而不是中间 I/O/copy completion。

### 12.6 Eager 按层流水的准确边界

当前 vLLM 以 eager、逐层 callback 为主，不需要把 compute 改造成 CUDA Graph。目标是由 connector 在已有 callback 边界内组织依赖：

```text
read metadata[L] -> read IO[L] -> transfer/scatter[L]
                                      -> compute[L]
read metadata[L+1] -> read IO[L+1]      -> compute[L+1]
compute[L] -> gather/write[L]
```

`start_load_kv()` 负责建立当前请求的 read 计划和有限 lookahead；`wait_for_layer_load(L)` 只等待/桥接 layer L 的 read/transfer 完成；`save_kv_layer(L)` 在 compute callback 后把 gather/write 排入 write stream。这里的“自动执行”是 CUDA stream 中已经排队的 operation 按顺序执行，不是 device 端动态 launch compute。失败门控必须在当前层 compute 前完成，不能把 `read_done` event 当成 `read_success`。

不把 CUDA Graph、persistent dispatcher 或 device timeline semaphore 作为当前实现前置条件。只有 eager 按层流水已正确、且 profiling 证明 Python callback/submit 开销成为主要瓶颈时，才另开性能设计卡。

### 12.7 Stream 和资源规划

最小流集合：

```text
compute_stream
read_io_stream
write_io_stream
copy_stream       # staged only，可与 read/write 共享后再优化
```

当前 local NVMe capability 声明最多 2 个 stream，因此第一阶段可用：

- 一个 read I/O stream。
- 一个 write I/O stream。
- compute stream 由 vLLM 管理。
- staged copy 先挂 read/write stream，后续扩展 capability。

为了避免 I/O kernel 抢占 attention，可继续使用 CUDA Green Context/SM partition 或严格限制 I/O kernel blocks/threads。目标是可控 overlap，不是无限并行。

增加 stream 只能增加独立流水的并发度，不能绕过共享资源上限。每新增一条
stream，至少要有独立的可写 workspace、completion/fence 状态和明确的 queue
ownership；同时还会竞争同一 GPU 的 SM/DRAM、同一 NVMe controller 的 queue/CQ、
PCIe 链路和 DataPath progress。当前 eager vLLM 单 step 的依赖链只要求一条
read stream 和一条 write stream；继续增加 read stream 只有在存在多个独立
request/step、后端 queue 足够且实测 read/scatter 或跨请求 IO 能并行时才有收益。
“stream 数增加”不等于“`max_in_flight_operations` 增加”，也不等于 NVMe
带宽线性增加。首版固定双 stream，后续以 Nsight timeline、PCIe RX/TX、NVMe
Read/Write B/s 和 compute slowdown 共同决定是否扩展。

### 12.8 Metadata 和失败语义

```text
ABSENT
  -> RESERVED(generation, lease)
  -> WRITING(all layers pending)
  -> COMMITTING(marker/journal)
  -> RESIDENT

任意失败：
  -> FAILED/ABSENT
  -> invalidate affected block range
  -> 回收新 generation
  -> 保留旧 generation
```

约束：

- 只有全部层 write 成功才发布 resident。
- load 的任意 layer 失败都必须阻止该 request 使用对应 external KV。
- device status array 保存 per-entry NVMe/CUDA 状态。
- worker 通过单一 progress thread 聚合状态并通知 scheduler。
- scheduler 不根据“计划提交”推断“已经成功”。

## 13. 推荐实施顺序

### 阶段 0：先恢复正确性

1. 保留已修复的精确 load interval，修复 `max_tokens_per_load` cap 后失去 chunk 对齐的问题，并增加 block table 精确长度校验。
2. 在 store open 前拒绝 DCP>1；完整支持前把该部署边界写入配置与测试。
3. override required layout 为 NHD，或真正按 `attn_backend` stride descriptor 支持 HND；补真实 packed 4D 测试。
4. Qwen/HMA 保持明确 fail-fast，不允许 `_flatten_blocks()` 静默取首组；HMA 作为独立实现阶段。
5. 修复 partial commit；`_PostCompletion.query()` 在转为异步 fence 前修复最终完成语义。
6. 建立 load failure -> invalid blocks、save failure -> rollback 闭环，并去掉错误的 optimistic residency。
7. 保留已修复的prebuilt extent/MDTS切分；继续修复MPS memory offset、64 KiB registration backing rounding和padded page canonical layout。

### 阶段 1：静态双 bank，先消除逐层 read host wait

1. 分配、注册互不重叠的 read bank 和 write bank；不再让一个 `RingWindow` 同时服务两个 stream。
2. read stream 排列 `read -> scatter -> ready event`；`wait_for_layer_load()` 只桥接 compute stream fence。
3. write 第一版在 `save_kv_layer()` record compute-done 后排 `gather -> write` 到 write stream，不追求一次预提交未来 event。
4. 当前write按Runtime窗口做submit前回压；若仍要求所有read launch一次预排，增加sequence submission，而不是放大窗口。
5. 用 r5 同形 workload 验证 read launch 从逐层散布变成启动阶段排队，并比较 HBM、TTFT、read/compute/write overlap。

### 阶段 2：固定注册产物和生命周期

1. 引入通用 `LayoutDescriptor` 和 `StorageGeometry`，不再基于 rank/axis 猜 layout。
2. runtime `MemoryEntry` 持久保存 `io_granularity`。
3. 补齐 binding 的 unregister、close、query/fence。
4. 保持host-pinned prebuilt/dynamic PRP路径和target extent/MDTS正确切分；GPU PRP allocation counter必须恒为0。
5. 替换 per-batch Python watcher 为 worker 级 progress engine。
6. 把 file zero-fill/fsync 移出 step 热路径，并为 extent map 建 generation/signature 校验。

### 阶段 3：形成按字节自动选路的双路径

1. 保留并完善静态 staged read/write bank、packed reshape kernel；仅在确需多 wave 并发时引入静态槽轮转。
2. 将 DataPath paged SPI 接入 StorageRuntime，并暴露到 pybind/store/engine。
3. 用当前 vLLM packed layout 生成 direct spans 和 staged reshape plan。
4. 引入 `TransferCapabilities + IoPathPolicy`，按实际 bytes/alignment/descriptor cost 选择，不设置全局固定优先路径。
5. 允许同一 step 的不同 extent 使用不同路径，但共享 semantic key、disk layout 和 commit journal。
6. per-entry status、CQ status 和 retry 语义统一。
7. 冻结 canonical segment ABI 和 layout version；仅在 raw KV span 与 ABI 字节同构时允许 direct。

### 阶段 4（当前不实施）：CUDA Graph/全设备 DAG

当前 eager 目标不实施该阶段。若未来 profiling 证明逐层 Python submit 成本不可接受，再单独设计 `StepPlan`/sequence API；不得把它作为当前验收条件。

### 阶段 5（未来可选）：vLLM compute graph 联动

1. 使用 piecewise CUDA Graph 捕获每层 compute。
2. 把 `kv_ready[L]` 和 `compute_done[L]` 接入 graph edge。
3. step 开始时一次 instantiate/launch 整体 graph 或常数个 graph wave。
4. 去掉每层 Python callback 的关键路径职责。

## 14. 验收条件

### 正确性

- 覆盖 packed 4D 单层和 cross-layer NHD/HND 的地址映射测试。
- 覆盖 DCP>1 fail-fast；完整支持后再覆盖 DCP rank/interleave canonical mapping 和 namespace 隔离。
- 覆盖 multi-group/HMA block table 不被 flatten；HMA 未启用时 Qwen 必须在 store/CUDA 初始化前明确失败。
- 覆盖 MHA/GQA、MLA 和 linear-attention state spec 的 `page_size_bytes -> path` 决策测试。
- 覆盖 `real_page_size_bytes != page_size_bytes`，断言 page padding 不被摊入 token row、不会包含未初始化显存。
- 覆盖 block 16/128/256，但断言依据是实际 `io_extent_bytes`、对齐和 descriptor 数，而不是 token block 数本身。
- 覆盖 `block_size -> page_size_bytes -> segment -> layer file offset -> FIEMAP extent -> namespace SLBA/NLB` 的公式级 golden cases。
- 覆盖 namespace LBA 为 512B/4KiB、controller MPS 为 4KiB、segment 跨 MDTS/extent/stripe unit 的组合；每个 sub-I/O 都命中正确 target 和 memory byte range。
- 覆盖 staging logical size 非 64 KiB 倍数时，rounded DMA registration 不越过 backing allocation。
- staged/direct 对同一 canonical layout 的 on-disk bytes 必须一致；layout version/spec hash 不同不得互相命中。
- 任意 local/external prefix 组合都产生精确、不重叠、不漏 token 的 load interval。
- block table 不足、尾块、preemption、abort 有显式行为。
- 任意 layer read/write 失败都不会发布错误 residency。
- replacement 失败保留旧 generation。
- partial commit 不重复提交已 committed entry。

### 路径选择

- 小于最小高效 I/O、非 4 KiB 长度对齐或需要 reshape 的 extent 稳定选择 staged。
- 大且满足 address/offset/length alignment 的 extent 在 direct capability 存在时选择 direct。
- 同一 logical chunk 分别经 staged/direct round trip 后字节和 token 语义完全一致。
- path decision、阈值、拒绝 direct 的具体原因可观测，不能静默回退。
- KV pool 和 read/write staging bank 都只在 bind/registration 期注册，step 热路径无内存注册。
- path selector 的 capability 输出分别展示 target alignment、namespace LBA、controller MPS、GPU pin granule 和 minimum efficient bytes，不能只显示一个含糊 `page_size`。

### 执行路径

- eager 按层阶段：`wait_for_layer_load(L)` 在当前层 callback 中检查本层 CQ；成功才 scatter。失败时不上屏该 KV，记录 invalid blocks并停止后续外部 load/save；vLLM可以继续本次 eager forward，但 scheduler必须丢弃受影响输出并按policy处理。
- eager 按层阶段：允许 `start_load_kv()` 预取有限 lookahead 层；后续层仍可由 callback 补交，直到 sequence API 单独验收完成。
- eager 按层阶段：`save_kv_layer(L)` 在 compute callback 后 record compute-done，并将 gather/write 排入 write stream。
- read/write staging bank 永不互相覆盖；单 bank 复用由所属 stream 的最终 scatter/write 顺序保证，双 bank 复用由明确 fence 保证。

### GPU overlap

- timeline/trace 中可见 `read[L+1]`、`compute[L]`、`write[L-1]` 同时在不同 stream 执行。
- I/O kernel 有受控的 SM/queue 占用，不显著拖慢 compute。
- 每个 I/O entry 都有可追踪 status，NVMe CQ 错误可到达 scheduler invalidation。
- timeline capture 与 system-metrics capture 分开验收；后者开启 GPU metrics 和 storage metrics，报告当前硬件实际提供的 PCIe、GPU SM/DRAM 和目标 NVMe throughput/IOPS。若 GPU metric set 不提供 PCIe RX/TX，必须明确记录并用 DCGM/NVML/系统级计数器补充，不能伪造 Nsight 指标。未开启相关采集选项时不得从 `.nsys-rep` 推断链路利用率。

### 生命周期

- register/unregister、target open/close、engine shutdown 成对。
- scheduler 进程不创建 CUDA stream、不注册 GPU memory、不打开 data path target。
- worker 长时间运行时 watcher thread、memory mapping、DMA context 和 file handle 数量稳定。

## 15. 最终判断

历史文档给出的“批量描述符 + GPU 自主调度”是长期性能方向，但当前目标明确采用 eager 按层流水，不考虑 CUDA Graph。最直接的执行模型是：read metadata/IO 预取有限 lookahead；`wait_for_layer_load(L)` 完成当前层 CQ gate、scatter 和 compute stream fence；`save_kv_layer(L)` 在计算后把 gather/write 排到独立 write stream。这样先保证 read/transfer/compute/write 的层间流水和失败安全，再根据 profiling 决定是否需要 sequence API。

InfiniKV 仍证明了 direct paged KV registration 对大且对齐 extent 的价值，但其 metadata、错误状态、资源回收和 layout 不能直接复制。Tutti 的 host-pinned PRP backing和staged prebuilt fast path已经完成，不应重做；下一步应保留staged路径处理小块/reshape，再补direct paged span。之后按实际需要引入 `LayoutDescriptor + IoPathPolicy + StepPlan + durable commit journal`。此外，DCP、HND、HMA/multi-group和系统metrics仍必须作为独立准入项，不能被Hy3 TP4的一次成功smoke覆盖。scheduler角色隔离已由metadata-only实现关闭，仍需真机queue-group计数复验。

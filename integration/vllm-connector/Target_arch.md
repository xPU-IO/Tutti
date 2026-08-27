# Tutti vLLM Connector 目标架构与当前实现

> 状态：当前权威架构文档，更新于 2026-08-26。
> 适用版本：Tutti 当前工作树；vendored vLLM `897ff4f39`。
> 当前执行目标：vLLM eager 模式下的逐层 read / transfer / compute / write
> 流水，不以 CUDA Graph、persistent dispatcher 或整步 GPU DAG 为前置条件。

本文同时描述当前已经接通的真实对象、调用路径和错误语义，部署必须
fail-fast 的边界，以及后续优化不可破坏的契约。历史设想只在与本文一致时
有效。

## 1. 目标与非目标

### 1.1 当前目标

- scheduler 只消费 prefix 命中数、block table 和 connector metadata。
- worker 在 attention 的既有逐层 callback 中编排 KV I/O。
- read、write 使用不同 CUDA stream 和不同 staging bank。
- 当前层 read 在 Runtime/NVMe CQ 确认成功后才允许 scatter。
- gather 发生在当前 compute stream；write stream 等 gather event 后提交 I/O。
- 维持有限 `lookahead_k`，形成：

```text
read/transfer[L+1] || compute[L] || write[L-1]
```

- 几何从 vLLM `KVCacheConfig` 推导，不手写模型层数和 KV 字节数。
- PRP list 使用 host-pinned、DMA-mapped 内存。
- 加载失败遵循 vLLM/LMCache 的 `invalid_block_ids` 契约，由
  `kv_load_failure_policy` 决定重算或终止。
- local/striped DataPath、structured completion 和生命周期语义保持对称。

### 1.2 当前非目标

- 不要求把 vLLM compute kernel 捕获成 CUDA Graph。
- 不要求一次调用预排全部层的 compute。
- 不把 `cudaEvent` 完成等同于 NVMe I/O 成功。
- 不用增加 stream 数掩盖错误的 workspace/lifetime 设计。
- SupportsHMA 完成前不支持 Qwen3.8-27B 的 Tutti 数据面。
- canonical DCP mapping 完成前不支持 DCP > 1。
- 当前不承诺 direct paged KV I/O；生产路径是 staged transfer。

## 2. 系统层次

```text
vLLM Scheduler
  Scheduler.schedule / Scheduler.update_from_output
          |
          v
TuttiConnectorV1 (scheduler role)
  prefix lookup / request tracking / connector metadata / store reservation
          |
          | TuttiConnectorMetadata
          v
TuttiConnectorV1 (worker role)
  WorkerImpl: step state, K-lookahead, layer callbacks, failure block reporting
          |
          v
KVEngine
  ChunkIndex + read/write RingWindow + transfer selector
          |                         |
          | paged <-> staging       | layer-segment byte I/O
          v                         v
PagedTransferHooks             TuttiKVStore
  gather/scatter                layout/marker/target/buffer/stream/completion
          |                         |
          +-----------+-------------+
                      v
Python binding: tutti_runtime
  open/register/submit/wait_result/release/shutdown
                      |
                      v
StorageRuntime
  public handles, partial commit, lazy registration, progress, terminal result
                      |
                      v
Resolver -> DataPath SPI
  LocalFileResolver / StripedResolver
  LocalNvmeDataPath / StripedDataPath
                      |
                      v
GPU submit kernel -> NVMe SQ/CQ -> registered payload memory

tutti_daemon
  device/queue/resource control plane; not the Python layer scheduling loop
```

### 2.1 代码归属

| 层 | 当前路径 | 职责 |
|---|---|---|
| vLLM adapter | `adapter/connector.py` | 双角色 API、scheduler 翻译、metadata |
| worker orchestration | `adapter/worker.py` | K-lookahead、逐层 wait/save、bank 生命周期 |
| semantic engine | `engine/core.py` | ChunkIndex、layer I/O、transfer、completion |
| staging | `engine/staging.py` | 有界槽位和复用保护 |
| transfer | `engine/transfer.py`, `transfer/` | paged KV 与 canonical segment 搬运 |
| semantic index | `index/chunk_index.py` | chunk hash、LRU、pin、pending/resident |
| Tutti NVMe store | `stores/tutti_nvme/` | layout、marker、runtime bridge |
| Python binding | `bindings/python/src/_core.cpp` | C++ handle 和 structured result 映射 |
| Runtime | `tutti/include/tutti/storage_runtime.h` | backend-neutral I/O runtime |
| DataPath | `tutti/data_paths/` | DMA、descriptor、GPU NVMe kernel |
| device/resource | `tutti/device_manager/`, daemon | controller、queue、resource 生命周期 |

### 2.2 抽象边界

- adapter 理解 vLLM request、token、block table 和 cache group。
- engine 理解 semantic chunk、layer、staging slot 和 transfer layout。
- store 理解 `io_key`、target URI、buffer ID、byte offset/length。
- Runtime 只理解 public target/memory/I/O handle 和 byte-range request。
- DataPath 理解 controller、LBA、PRP、queue、stream 和 CQ status。
- SSD FTL 以下的 NAND 映射不属于本系统可见范围。

上层不得从文件路径或 LBA 反推 token 语义；DataPath 不得理解模型层名。

## 3. 调度对象与所有权

### 3.1 Scheduler 对象

```text
_RequestTracker
  req_id / token_ids / block_ids / saved_tokens

_pending_loads[req_id] -> external token count
_load_starts[req_id]   -> exact load interval start
_live_requests[req_id] -> current vLLM Request

TuttiConnectorMetadata
  requests: list[_ReqMeta]

_ReqMeta
  req_id / token_ids / block_ids
  load_start_token / load_tokens
  save_chunk_start / save_chunk_count
```

当前 metadata 是单 KV cache group 结构。多组不能用 `_flatten_blocks()` 只取
第一组；HMA 必须升级为 `tuple[list[int], ...]` 并携带 group identity。

### 3.2 Worker step 状态

```text
load:
  _load_keys / _load_block_tables / _load_handles[layer]
  _pinned / _load_failed / _load_error_blocks

save:
  _save_keys / _save_block_tables / _save_inflight

binding:
  _read_window / _write_window / _num_layers
  _layer_names / _cross_pool
```

新 metadata 到达前，上一 step 必须 finalize 或 abort。abort 不取消已经下发的
NVMe 命令；它 drain completion，并禁止后续 scatter/save。

### 3.3 Engine、store 与 handle

- scheduler 持有 `SchedulerMetadataIndex` 和 metadata-only store；它只扫描
  manifest/marker、维护 `ChunkIndex` 和容量 reservation。
- worker 的 `KVEngine` 持有 `ChunkIndex`、data-plane store、read/write bank、
  transfer 和 inflight。
- 同一进程、同一 `vllm_config` 可经 `_ENGINE_CACHE` 共享 engine；多进程实例
  通过持久 marker 对账。
- `TuttiKVStore` 持有 runtime、target/memory tickets、buffer IDs、read/write
  CUDA stream owner 和文件 layout。
- public I/O handle 必须 terminal 后才能 release。
- scheduler role 不导入/构造 worker data plane，不创建 StorageRuntime、CUDA
  stream、GPU registration 或 NVMe queue group；worker role 保持完整数据面。

## 4. 几何、布局与准入

### 4.1 权威几何

```text
T = kv_cache_spec.block_size                 tokens / vLLM block
B = child_spec.page_size_bytes               bytes / layer / vLLM block
C = chunk_tokens                             tokens / semantic chunk
P = C / T                                    blocks / semantic chunk
S = P * B                                    bytes / layer segment
N = number of layers
F = N * S                                    payload bytes / chunk object
```

要求：

- `C % T == 0`；
- 当前仅一个 cache group；
- 组内所有层 `B` 相同；
- 显式 `num_layers/chunk_kv_bytes/segment_bytes` 必须与推导值一致；
- segment 满足 store/DataPath 的 address/offset/length alignment。

`num_layers` 和 `chunk_kv_bytes` 不是部署 magic number，不能硬编码 80 层或
20 MiB。

### 4.2 布局准入

- connector 要求 vLLM 使用 `NHD` cross-layer layout。
- HND 当前不准入，避免把 head axis 当 layer axis。
- DCP > 1 在打开 store 前 fail-fast。
- 多 cache group在 geometry 阶段 fail-fast；当前不继承 `SupportsHMA`。
- Qwen3.8-27B 是 3 个 GDN/Mamba group + 1 个 full-attention group，只能用于
  `--without-tutti` 的生命周期验证。
- Hy3 单组模型是当前真实 Tutti 数据面验收模型。

### 4.3 Padded page

`page_size_bytes` 可能包含 block-tail padding，`real_page_size_bytes` 才是有效
payload。当前 canonical segment尚未完整区分两者。完成前，不满足现有 reshape
契约的 padded spec必须 fail-fast；后续 ABI必须确定性处理padding并版本化。

## 5. 初始化与注册

```text
TuttiConnectorV1.__init__
  -> _resolve_geometry(extra, kv_cache_config)
  -> reject DCP > 1 / multi-group / non-uniform pages
  -> scheduler: SchedulerMetadataIndex -> metadata store -> scan markers
  -> worker: create_store -> KVEngine -> data-plane store.open

vLLM KV cache allocation
  -> register_kv_caches or register_cross_layers_kv_cache
  -> WorkerImpl._ensure_bound
  -> determine layer view / transfer format
  -> allocate one 64 KiB-aligned staging allocation
  -> split read bank + write bank
  -> KVEngine.bind
  -> store.register_buffer(segment_bytes)
  -> Runtime.register_memory(io_granularity=segment_bytes)
  -> lazy DataPath registration on first target submit
```

key namespace至少编码模型、KV dtype、TP、chunk bytes、chunk tokens和格式版本。
未来 DCP/HMA/PP/layout支持必须扩充namespace。

### 5.1 Split bank 几何

```text
W = max_chunks_per_wave
K = min(lookahead_k, num_layers)
bank_slots  = W * K
total_slots = 2 * bank_slots

read  slots = [0, bank_slots)
write slots = [bank_slots, 2 * bank_slots)
```

总 staging HBM 与拆分前相同。两个 `RingWindow` 共享 backing，但物理
`slot_base`区间不得重叠。store只注册一次原始allocation。

## 6. Scheduler 调用路径

### 6.1 Prefix lookup

```text
Scheduler.schedule
  -> get_num_new_matched_tokens
  -> engine.sync_from_store
  -> ChunkIndex.lookup_prefix
  -> clamp remaining prompt
  -> chunk align
  -> min_retrieve_tokens / max_tokens_per_load
```

最终返回仍须按 `chunk_tokens` 对齐；load cap不能制造partial chunk。

### 6.2 Allocation 与 metadata

```text
update_state_after_alloc
  -> record load_start_token before scheduler advances computed tokens
  -> record external load_tokens

build_connector_meta
  -> update RequestTracker snapshots
  -> compute exact save chunk interval
  -> plan_store capacity reservation
  -> DO NOT publish resident
  -> emit TuttiConnectorMetadata
```

scheduler只做reservation。worker全部写成功后才`confirm_store(ok=True)`；失败
必须`confirm_store(ok=False)`。

TP部署还要求 all-rank commit：scheduler 为每个 chunk 下发同一逻辑
generation；每个worker在本rank全部层完成后写rank-local commit record。scheduler
同时扫描所有rank独立root，只有rank集合、层marker、namespace、slot bytes、
object-pool generation和逻辑generation全部一致时才恢复resident。任一rank缺失、
失败或generation不一致都使整个chunk miss；restart只清理不完整commit record，
不删除其他rank payload/marker。

## 7. Eager 逐层执行编排

### 7.1 总时间线

```text
start_load_kv:
  build load keys/block tables -> pin -> submit read layers [0, K)

for layer L:
  wait_for_layer_load(L)
    wait Runtime/NVMe CQ result
    success -> scatter on read stream -> wait scatter fence
    failure -> no scatter; report invalid blocks; drain lookahead consumers
    success path submits read L+K

  vLLM compute layer L on compute stream

  save_kv_layer(L)
    gather paged KV on current compute stream
    record gather event
    write stream waits gather event
    submit NVMe write on write stream

step exit:
  wait_for_save
    drain save handles and both banks
    confirm_store(success/failure)
    release pins and step state
```

### 7.2 Load 依赖

```text
read stream:
  descriptor H2D -> submit kernel -> CQ poll

host gate:
  Runtime progress -> structured terminal result

success only:
  read stream scatter -> scatter event
  wait_for_layer_load returns -> vLLM submits compute L
```

`read_done_event` 不等于 NVMe success。success来自`TuttiBatchResult.ok`。失败后：

- 本层不scatter；
- lookahead completion仍drain/release但不scatter；
- 不再补交新layer；
- 本step后续external save跳过。

### 7.3 Save 依赖

```text
compute stream: compute L -> gather L -> gather_event
write stream:   wait gather_event -> descriptor H2D -> write kernel -> CQ
```

不能在write stream直接发gather而不等待compute。bank隔离只解决内存覆盖，不
自动建立跨stream依赖。

### 7.4 Stream 与并发

当前固定：vLLM compute stream、Tutti read stream、Tutti write stream。

增加stream可以增加独立流水并发，但每条stream需要独立可写workspace、completion
和queue ownership，并竞争GPU SM/DRAM、NVMe SQ/CQ、PCIe和progress。当前单
eager step不需要更多stream；只有真实trace证明多request/step可并行且资源未饱和
时才扩展。

## 8. 存储布局与地址映射

### 8.1 Key 与文件布局

```text
chunk_key = chained hash(token chunk, namespace)
io_key    = chunk_key[16 B] || layer_idx[2 B little-endian]

chunks/<chunk_key>.bin
layer L offset = L * segment_bytes
payload size   = num_layers * segment_bytes
```

`ChunkIndex`管semantic chunk；store以layer `io_key`管marker。scheduler只有在
期望层集合完整时恢复resident。

### 8.2 Striped mapping

virtual offset先按stripe unit映射shard，再经FIEMAP映射namespace LBA。fan-out
必须同时遵守stripe boundary、extent boundary、MDTS和alignment。

### 8.3 Host-pinned PRP 路径

当前主路径已经使用CPU pinned PRP：

- registration-time `PrpBufPool`：host-pinned、DMA-mapped、可增长；
- dynamic `PrpPageCache`：host-pinned、DMA-mapped；
- GPU `DescPool`只保存`AddressDescriptor`；
- controller直接读取host PRP list IOVA。

`io_granularity`已贯通；正常staged READ/WRITE真机contract命中prebuilt
descriptor，dynamic count为0。

这里`io_granularity`只表示Python staging中的逻辑block/slice大小`S`。生产
preset不接收MDTS；LocalNvmeDataPath attach后通过`ioctl_get_dev_info()`读取
`dev_info.max_data_size`，得到hardware/effective MDTS，并在C++中计算：

```text
bytes_per_slice = S
ios_per_slice   = ceil(S / effective_MDTS)
sub_io_bytes    = min(slice_remaining, effective_MDTS, extent_remaining)
```

每个LIST sub-I/O使用一个完整、controller-page对齐的host PRP page。page内
entry依次写入该sub-I/O第2..N个GPU data page IOVA；GPU
`AddressDescriptor.prp2`只保存这个host PRP page的IOVA。submit按
`slice_idx * ios_per_slice + sub_idx`引用预建descriptor；若FIEMAP extent提前
截断sub-I/O，则该fragment回退dynamic descriptor。不存在256B PRP子页打包，
也不存在128 KiB人为上限。

旧的`MetadataArena`/`StripedArena` GPU PRP backing和cache-exhausted H2D
fallback已经删除。cache disabled/miss/exhausted时由可增长的host-pinned
`PrpBufPool`提供lease；GPU只保留entry/status/`AddressDescriptor`等kernel metadata。
timeout只保留controller仍可能访问的host PRP/payload/target lease，不再永久消耗
整个GPU arena slot。

### 8.4 Rank-local 文件尺寸与对象预分配

当前`set_layer_span(num_layers)`只确定rank-local chunk文件的最终逻辑尺寸。文件
并未在模型初始化时创建；第一次`prepare_put()`仍同步真实写零扩展到
`num_layers * segment_bytes`并`fsync`。这保证FIEMAP有真实extent，但属于首请求
热路径分配，不是存储对象池预分配。

Hy3有8个KV heads，TP4时每rank持有2个不同KV heads，因此payload不重复，但几何
相同：每rank每chunk均为`80 * 256 KiB = 20 MiB`。重复的是chunk key、token元数据
和marker目录结构，不是KV payload。

目标对象池按rank-local geometry建立：初始化时真实写零创建固定大小slot文件，
分配chunk时把free slot原子rename到content-key名字；evict后在所有target handle
关闭后rename回free pool。slot inode/FIEMAP extent保持稳定，请求热路径不再扩文件
或fsync。不同rank即使未来PP/HMA导致文件尺寸不同，也各自使用本rank manifest和
slot size，不要求全局文件等长。scheduler发布公共命中前仍需all-rank commit仲裁。

对象池必须接通Tutti public Runtime生命周期，而不是只改Python文件名：

```text
allocate batch:
  ChunkIndex selects unpinned LRU victims
  wait victim IO terminal
  Runtime.close(TargetHandle) / close_batch
  rename victim chunk file -> free slot
  rename free slots -> new chunk-key files
  Runtime.open_batch(all missing URIs) exactly once
  cache returned TargetHandles by URI + slot generation
```

当前binding虽暴露`open_batch`，但store逐URI调用`open_batch([uri])`，没有真正批量
打开；binding也尚未暴露target close，Python pop ticket不会释放C++ TargetHandle。
对象池实现必须先补`close_target/close_batch`，并禁止旧handle跨slot generation
访问已复用inode。

pool采用`initial_slots/low_watermark/high_watermark/max_slots`。初始化同步创建
`initial_slots`；free slots降到low watermark时后台真实写零补到high watermark，
不超过max slots。请求先通过LRU回收已分配slot；仍不足时只允许有界等待后台allocator
或返回结构化`RESOURCE_EXHAUSTED`，不得在请求线程回退到同步扩文件/fsync。

## 9. Structured completion

### 9.1 C++

```text
IoFailureKind:
  NONE / RESOLVE_LBA / CQ_TIMEOUT / NVME_CQ_ERROR /
  CUDA_QUERY_ERROR / STATUS_D2H_ERROR / UNKNOWN

IoFailureScope:
  NONE / REQUEST_INDICES / WHOLE_OPERATION

IoCompletionDetail:
  confirmed_bytes / timeout_seen / first_failed_entry /
  failure_kind / raw_cq_status / failure_scope / failed_request_indices
```

local/striped必须对称填充。无法安全反向映射时扩大为`WHOLE_OPERATION`，不能
伪造不完整bitmap。

### 9.2 Python

binding暴露`WaitResult`；store聚合`TuttiTerminalResult`和`TuttiBatchResult`。
旧`wait()`失败仍抛异常；engine load gate使用`wait_result()`。

partial commit规则：

- overall status可以non-OK，但有效handle表示部分I/O已发出；
- handle必须恰好drain/release一次；
- 只重试rejected indices；
- all-rejected无handle时才作为零接受背压。

### 9.3 Result 生命周期

terminal detail必须在release前复制。当前Runtime `released_results_`和pybind
`terminal_results_`允许release后查询，但仍需有界LRU或consume/forget语义；
长期服务不能永久保留每个I/O结果。

## 10. 加载失败与 vLLM 回退

### 10.1 vLLM/LMCache 标准契约

自动回退不是“connector抛异常”：

```text
connector records invalid block IDs
  -> get_block_ids_with_load_errors returns and clears them
  -> KVConnectorOutput.invalid_block_ids
  -> Scheduler._handle_invalid_blocks
  -> discard affected request output for this step
  -> recompute: truncate valid prefix and reschedule
     fail: finish request with error
```

LMCache也是通过`_invalid_block_ids`接入，不依靠裸异常。
`kv_load_failure_policy`默认是`fail`；显式设置`recompute`才自动重算。

### 10.2 Tutti 当前行为

确认的Runtime/CQ失败：

- `_PostCompletion`不scatter；
- worker把失败wave或whole operation对应block加入`_load_error_blocks`；
- drain已预取I/O但禁止consumer callback；
- 停止补交后续external load和save；
- `LoadGateError`转换为invalid IDs，不作为裸异常逃出forward；
- 编程/配置错误仍抛出。

当前遵循vLLM post-forward recovery：eager forward可能继续运行，但scheduler丢弃
受影响输出，再重算或终止。若要求失败层后完全不compute，需要扩展vLLM
model-runner abort协议，不属于当前LMCache标准路径。

### 10.3 HMA限制

vendored scheduler的invalid-block recovery仍强制单group解构。Qwen HMA失败恢复
不能复用该路径，必须先实现group-aware共同有效前缀和全group回退。

## 11. Save一致性

```text
ABSENT -> WRITING(reserved) -> RESIDENT
                   |
                   +-> failure -> ABSENT
```

- scheduler只`plan_store()`，不得乐观发布resident；
- worker全部save completion成功后才confirm；
- 任一层失败继续drain，然后`confirm_store(ok=False)`；
- store先完成数据I/O再建layer marker；
- scheduler只恢复完整层集合；
- failure step不执行后续save。

已知未完成项：replacement可能在新generation成功前删除旧数据。目标是新
generation完整commit后原子切换，再回收旧generation。

## 12. 背压与资源

```text
scheduler: max_tokens_per_load
worker:    lookahead_k / max_in_flight_layers
engine:    read/write RingWindow reuse
Runtime:   partial commit / DataPath admission
```

`max_batch_entries=4096`是fan-out最坏上限，不是实际每层entry数。r5实际read
204、write最多256 entries，hardware MDTS 2 MiB，segment 256 KiB。

`max_in_flight_operations`当前把queued、executing、terminal-unreleased绑定为
“一op一arena slot”，是实现限制，不是stream物理限制。当前worker在未显式
配置`max_in_flight_layers`时自动读取Runtime capability；达到窗口前，在下一层
submit之前wait/release最老write completion。同一write stream保持FIFO，80层
可用窗口4稳定推进，不靠放大arena。目标模型仍是：

```text
queued ops per stream       may be many
executing ops per stream    stream ordered
completion per op           compact result
workspace per stream        reusable entry/status/descriptor arrays
```

## 13. 可观测性与 profiling

### 13.1 NVTX颜色

| 方向 | 颜色 | ARGB |
|---|---|---|
| read/load/scatter | cyan | `0xFF00B8D9` |
| write/save/store/gather | orange | `0xFFFF8C00` |
| wait/mixed | gray | `0xFF8D99A6` |

C++外层range带`|op=read/write`，内层保留旧marker
`tutti.local_nvme.io_kernel`/`tutti.striped_nvme.io_kernel`。

### 13.2 Nsight采集

- timeline：CUDA + NVTX，关闭高体积sampling；
- metrics：Nsight Systems 2026.4.1，GH100 GPU metrics从1 kHz起步，storage
  metrics只采目标`snvme0n1`；
- H20/GH100可提供PCIe RX/TX占峰值百分比；
- local storage plugin提供Read/Write B/s，IOPS用diskstats/iostat/eBPF；
- DCGM与Nsight counters冲突时采集前暂停、结束后恢复。

### 13.3 真机验收

- `/data2/tencent/Hy3-FP8`、TP4、eager、单group；
- A cold save；reset local APC但保留Tutti；B external load；
- 容错验收显式设置`kv_load_failure_policy=recompute`；
- Qwen只用于`--without-tutti`生命周期迭代。

2026-08-26真机结果（`hy3-recompute-real-10f`）：A-cold 8192 prompt tokens、
4 completion tokens，2.669s；reset本地prefix后B命中Tutti 6400 tokens，
4 completion tokens，1.209s；四个TP worker按rank-local GPU/NVMe运行并正常退出。

### 13.4 `hy3-visible-r1` 调度证据与未完成项

Nsight SQLite证明三条CUDA stream在每个rank上实际分离：Transformer compute为
`streamId=19`，read I/O/scatter为`31`，write I/O为`35`。read stream中的
`index_elementwise`/`vectorized_elementwise`是scatter数据搬运，不是模型计算；
GUI按NVTX projection显示时可能看起来在同一组。

当前已接通step feeder：K=2时每个rank的B read为40个window kernel，A/B write
合计80个window kernel；每个step只有一个Runtime/DataPath handle。逐层callback
只消费/publish gate，whole-op completion在全部callback结束后才开始观察。

目标eager实现不使用CUDA Graph，而使用两个step级设备任务：

```text
start_load_kv once:
  submit all-layer read descriptors
  one read kernel updates layer_done[L] / layer_status[L]

wait_for_layer_load(L):
  compute stream device-waits layer_done[L]
  guarded scatter(L)
  return without host Runtime.wait

start/save step once:
  launch one low-occupancy write feeder

save_kv_layer(L):
  gather into layer-owned staging
  publish write_ready[L]

write feeder:
  wait write_ready[L] -> issue layer L writes -> compact status

step end:
  harvest one read handle + one write handle
```

read/write staging必须保证同时在途layer物理隔离，不能让未scatter/未write的
数据被后续层覆盖。默认资源模型是每个stream固定`K`个layer slot，并由device-side
`ready/consumed`握手控制复用；read/write feeder复用slot前等待consumer完成，不走
host `acquire/wait`。Hy3按实际25/32 chunks、256 KiB segment、`K=2`计算，read约
12.5 MiB/rank，write约16 MiB/rank。若要求所有80层read完全同时发出，也可选择
layer-owned全量staging，代价约500 MiB/rank；必须作为显式高内存模式，而非默认。

### 13.5 真实 Hy3 feeder callback/gate准入

2026-08-27诊断确认Hy3的物理KV层与实际callback都是完整顺序0..79；旧结论中
“callback缺层”并非根因。真正停滞发生在最后read callback 79：DataPath已经记录
final event，但step构造时立即启动的whole-op completion watcher与per-layer gate
API竞争Runtime registry lock，callback无法进入`wait_feeder_layer()`。

当前修复：

- 从`KVCacheConfig.kv_cache_groups[*].layer_names`建立callback ordinal到物理KV
  ordinal的显式映射；feeder layer plan按callback数构造，不盲改物理层数。
- 无callback的物理层不创建gate；支持多group前仍在store构造前fail-fast。
- 重复callback幂等；乱序callback立即drain/fail-fast；finalize发现缺失callback
  时drain并fail-closed，不留下永久gate。
- step whole-op watcher延迟到全部callback gate发布后启动。
- Runtime每次只在registry lock内做一次非阻塞gate probe；PENDING时释放lock后
  再等待，避免长gate wait阻塞progress/signal。

真实Hy3 TP4 eager A/B及Nsight已通过：B命中6400 tokens并正常退出；每rank
compute/read/write分别位于stream 19/31/35，read 40、write 80个feeder kernel，
39/40 read windows同时与compute和write overlap。

这里的“step-level”仅表示一次Runtime/DataPath提交和一个arena lease，不表示一次
GPU kernel发出全部80层I/O。默认`K=2`复用staging bank，host feeder仍分40个
window依次启动read kernel；每层read完成后的scatter/reshape由vLLM callback在
当前compute stream 19上发出。因此Nsight中NVMe read kernel位于独立stream 31，
但read数据搬运kernel会与模型计算共同出现在stream 19。若目标是所有read I/O
一次性下发并让scatter也脱离compute stream，需要独立的全量read staging或更大的
K，以及read-copy stream到compute stream的逐层event桥接；当前实现尚未达到该目标。

## 14. 当前状态

| 能力 | 状态 |
|---|---|
| KVCacheConfig自动几何 | 已实现 |
| DCP>1 fail-fast | 已实现 |
| required NHD | 已实现 |
| single-group staged transfer | 已实现 |
| split read/write bank | 已实现 |
| local/striped双stream routing | 已实现 |
| host-pinned prebuilt PRP | 已实现并真机命中 |
| structured completion | 已实现 |
| partial-commit正确drain/retry | 已实现 |
| invalid-block error reporting | 已实现单group路径 |
| save完成后发布resident | 已实现 |
| NVTX read/write颜色 | 已实现 |
| HMA/Qwen多group | 未实现，fail-fast |
| DCP canonical mapping | 未实现，fail-fast |
| packed NHD通用descriptor | 部分实现，需真实shape覆盖 |
| padded page canonical ABI | 未实现 |
| GPU PRP fallback清理 | 已实现并通过local/striped真机contract |
| logical block + driver MDTS切分 | 已实现，256KiB/4MiB真机回归通过 |
| Runtime窗口自动回压 | 已实现，真实Hy3 K=2 feeder通过 |
| per-stream compact workspace | 设计完成，未实现 |
| terminal result有界生命周期 | 未实现 |
| replacement generation原子切换 | 未实现 |
| rank-local I/O device placement | 已实现并通过Hy3 TP4复验 |
| scheduler metadata-only role | 已实现，runtime factory 单测为 0 调用 |
| TP all-rank commit gate | 已实现，任一rank缺失/异代时整chunk fail-closed |
| step-level feeder | 已实现；synthetic/local/真实Hy3 TP4/Nsight均通过 |
| feeder callback ordinal map | 已实现；subset/乱序/重复/缺失合同覆盖 |

## 15. 验收条件

### 15.1 正确性

- read失败不scatter，invalid IDs在同一forward输出上报。
- `recompute`丢弃失败step输出并重调度；`fail`终止请求。
- 编程错误不会被误吞成KV miss。
- write失败不创建完整marker、不发布resident。
- partial commit不重复提交accepted request。
- read/write bank物理区间不重叠，总staging HBM不增加。
- gather严格发生在compute后，write严格发生在gather后。
- local/striped failure detail对称。

### 15.2 性能

- trace可见`read/transfer[L+1] || compute[L] || write[L-1]`。
- read/write使用不同stream，compute使用vLLM当前stream。
- routine path无step内memory registration。
- staged path命中prebuilt descriptor，GPU PRP allocation count为0。
- I/O kernel占用不显著拖慢attention compute。

### 15.3 生命周期与部署

- register/unregister、open/close、submit/release、shutdown成对。
- abort/preemption drain已下发I/O并禁止consumer callback。
- terminal result cache、watcher、target、DMA mapping长期稳定。
- timeout只隔离仍可能被controller访问的资源。
- TP rank I/O落在对应local GPU，不集中到device0。
- 测试后daemon、GPU memory、NVMe queue和file handle稳定。

## 16. 推荐实施顺序

1. 给Runtime/pybind release后result增加有界LRU或consume语义。
2. 用真实Hy3 TP4验证split bank、双stream、rank-local device和recompute。
3. 修复packed NHD/padded page canonical测试缺口。
4. 实现HMA/multi-group，再启用Qwen Tutti数据面。
5. profiling证明必要后，再把per-op arena改为per-stream compact workspace。

当前不以CUDA Graph为实施阶段。eager逐层流水正确、可恢复、可观测之后，再根据
CPU submit开销和GPU/PCIe/NVMe利用率决定是否需要更激进的sequence API。

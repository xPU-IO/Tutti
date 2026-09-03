# Tutti vLLM Connector 目标架构与当前实现

> 状态：当前权威架构文档，更新于 2026-09-03。
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
- 目标主路径把 vLLM KV page 直接注册给 Runtime，read、write 不经过 staging。
- read、write 使用不同 CUDA stream；direct read 完成事件直接放行当前层 compute，
  direct write stream 等当前层 compute event 后提交 I/O。
- staging 保留为布局、对齐、padding 或 backend capability 不满足时的兜底路径；
  仅兜底路径分配不同的 read/write staging bank 并执行 scatter/gather。
- direct read 使用 layer callback 交错 enqueue：`start_load_kv` 只提交第 0 层，
  layer L 的 compute 已 enqueue 后再提交 read L+1。staged fallback 维持有限
  `lookahead_k`。两者都要形成：

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
- direct paged KV I/O 尚未完成真机准入；在完成前，当前生产实现仍走 staged
  transfer。该事实是迁移状态，不是目标架构。

### 1.0 当前安全闸（READ-SEQUENCE-AND-GATHER-FUSION-19）

当前工作树中曾尝试把 K-window ready/release 门控下沉到 C++ DataPath，
并使用 device-side future flag 维持 sequence。该方案已确认存在
进程级 GPU/NVMe stream hang 风险：write 可能在 Python `signal_step_layer()`
之前等待未来 ready，read reuse 可能等待后续 compute/scatter 发布的 release；
普通 abort/release 不能取消已经排入 CUDA stream 的 wait。故在架构修正前：

- 禁止对真实 `/dev/ssnvme*`、CUDA device buffer 执行整步预排或 future wait；
  当前只允许上层把已物化的当前层/窗口 flatten 成普通 byte-range
  `DataPathRequest[]` 后提交，窗口推进和 staging 复用由 Python host 编排；
- 不以增加 arena、`max_in_flight_operations` 或超时杀进程作为修复；
- 生产准入只保留已经验证的普通 `submit()`、host-pinned PRP、driver MDTS
  切分和 structured completion；
- 如未来需要 sequence API，必须先定义 immutable plan、ownership、
  terminal-before-release 和可证明的 abort/timeout 协议，再恢复真机验收；
- packed gather/scatter 应先作为独立 native transfer 优化验收，不与 NVMe
  feeder gate 同时改动。

本机近期多次 reboot 的 journal 原因是外部 `Power key pressed` 的干净关机，
尚未证明由 Tutti 直接触发；但 feeder hang 已由代码和 smoke 直接确认，必须
按高风险路径隔离处理。

### 1.3 Read pipeline safety baseline (STAGED-READ-PIPELINE-18)

The active read path is legacy-compatible layer orchestration. Worker/KVEngine
submits the first bounded lookahead using ordinary `get_batch()` operations;
each layer callback waits only for its own completion, enqueues scatter on the
read-copy stream, bridges `scatter_done[L]` to compute, then submits the next
layer. The read/write RingWindows remain separate and bounded. No lower layer
knows callback ordinals, windows, ready/release flags, or future events.

The old whole-plan future-event experiment is not part of the production
interface. A later layer can reuse a bank only after the upper layer observes
the prior completion/scatter event. Write uses one ordinary `put_batch()` per
layer after the compute/gather fence.

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
  ChunkIndex + transfer selector + completion
          |                         |
          | direct (target)         | staged (fallback)
          | registered KV pages     | RingWindow + gather/scatter
          v                         v
TuttiKVStore direct backend     PagedTransferHooks
  block table -> byte ranges      paged <-> staging
          |                         |
          +-----------+-------------+
                      |
                 TuttiKVStore
          layout/marker/target/buffer/stream/completion
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
| semantic engine | `engine/core.py` | ChunkIndex、layer I/O、路径选择、completion |
| direct backend | `stores/tutti_nvme/store.py` | KV pool 注册、block table 到 byte-range 映射 |
| staging fallback | `engine/staging.py` | 不满足 direct 准入时的有界槽位和复用保护 |
| transfer | `engine/transfer.py`, `transfer/` | direct/staged 选择与 fallback 搬运 |
| semantic index | `index/chunk_index.py` | chunk hash、LRU、pin、pending/resident |
| Tutti NVMe store | `stores/tutti_nvme/` | layout、marker、runtime bridge |
| Python binding | `bindings/python/src/_core.cpp` | C++ handle 和 structured result 映射 |
| Runtime | `tutti/include/tutti/storage_runtime.h` | backend-neutral I/O runtime |
| DataPath | `tutti/data_paths/` | DMA、descriptor、GPU NVMe kernel |
| device/resource | `tutti/device_manager/`, daemon | controller、queue、resource 生命周期 |

### 2.2 抽象边界

- adapter 理解 vLLM request、token、block table 和 cache group。
- engine 理解 semantic chunk、layer 和 transfer mode；仅 staged mode 理解 staging
  slot，direct 的物理 byte offset 由 store backend 从已登记布局和 block table 推导。
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
- worker 的 `KVEngine` 持有 `ChunkIndex`、data-plane store、transfer 和 inflight；
  仅 staged fallback 持有 read/write bank。
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

direct 路径还要求从实际 tensor shape/stride 推导以下几何，不得假设层数或固定
维序：

```text
pool_base       = cross_layers_kv_cache.data_ptr()
block_stride    = stride(block_axis) * element_size
layer_stride    = stride(layer_axis) * element_size
page_bytes      = B
memory_offset   = block_id * block_stride + layer_id * layer_stride
target_offset   = layer_id * S + block_ordinal_in_chunk * page_bytes
length          = page_bytes
```

对 Hy3 的 NHD cross-layer `[num_blocks, num_layers, block_size, 2, kv]`，一个
物理 block 的某层 page 连续。`block_size=128/256` 会把单次 direct extent 放大到
128/256 KiB；Runtime/DataPath 仍根据真实 controller MDTS 和 FIEMAP extent 在底层
切分，Python 不传 MDTS。

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
  -> determine actual tensor shape / stride / page bytes / alignment
  -> KVEngine.bind selects path
     -> direct eligible:
        register the KV pool once
        Runtime.register_memory(io_granularity=0)
        no staging allocation and no gather/scatter
     -> otherwise staged fallback:
        allocate one 64 KiB-aligned staging allocation
        split read bank + write bank
        store.register_buffer(segment_bytes)
        Runtime.register_memory(io_granularity=segment_bytes)
  -> lazy DataPath registration on first target submit
```

key namespace至少编码模型、KV dtype、TP、chunk bytes、chunk tokens和格式版本。
未来 DCP/HMA/PP/layout支持必须扩充namespace。

### 5.1 Direct 准入与回退

direct 是 capability 和实际 extent 驱动的默认候选，不是只看一个全局 bool：

- 单 cache group、uniform attention page，且 block 轴和 layer 轴可从实际 stride
  无歧义识别；
- 每个 `(block_id, layer_id)` page 是连续内存，`page_bytes` 与 target offset、
  request length 满足 DataPath alignment；
- pool 注册基址满足 snvme 当前 64 KiB GPU page 要求，注册范围覆盖所有 byte
  offset；PyTorch allocator 不提供此契约时必须回退，不能假定偶然对齐；
- `C % T == 0`，每个 chunk 的 block table 长度为 `C/T`；block id 可以离散，
  backend 将它们展开为同一 layer batch 内的多个 byte-range request；
- padded page、HMA/GDN、多 group、非连续/未知布局在各自 canonical ABI 完成前
  fail-closed 或回退 staged。

任何 direct 注册或准入失败都必须发生在首个真实 I/O 前。普通模式记录原因并
回退 staged；strict/profile 模式直接失败，防止误把 staged trace 当 direct。

整池 direct 注册初期必须使用 `io_granularity=0`。当前 prebuilt 实现会按
`registered_bytes / io_granularity` 为整段注册内存建立 descriptor，并为每个
大于两个 controller page 的 slice 保留完整 4 KiB host PRP page。十几 GiB KV pool
若按 64/128/256 KiB page 全量 prebuild，会把资源放大到全部 KV 容量；这与只按
在途 I/O 消耗资源的目标冲突。`io_granularity=0` 复用现有 dynamic descriptor 和
host-pinned PRP cache/pool，资源随已 enqueue 的 page request 数增长。后续如需稀疏
按需 prebuild，必须先单独设计和量化，不能在 Python 层假装整池 prebuild 免费。

当前 nonblocking direct 实现中，Python 可能在 GPU progress 前遍历完全部 callback，
因此 operation capacity 保守覆盖全部 read/write layer。`max_batch_entries` 则只需覆盖
单层展开后的 sub-I/O 数，两者是不同维度：

```text
required_in_flight_ops >= 2 * num_layers
required_batch_entries >= chunks_in_layer * blocks_per_chunk * MDTS_split_factor
```

因此 direct profile 应同时提高 `max_in_flight_operations`、按真实单层上界收紧
`max_batch_entries`，避免 arena 以“很深 × 很宽”同时过量预留；生产值在 trace 和
长期稳定性验证前不写死。

### 5.2 Split bank 几何（仅 staged fallback）

```text
W = max_chunks_per_wave
K = min(lookahead_k, num_layers)
bank_slots  = W * K
total_slots = 2 * bank_slots

read  slots = [0, bank_slots)
write slots = [bank_slots, 2 * bank_slots)
```

staged fallback 的总 staging HBM 与拆分前相同。两个 `RingWindow` 共享 backing，但物理
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
  build load keys/block tables -> pin -> direct submit read layer 0 -> return
  (staged fallback keeps the bounded [0, K) window)

for layer L:
  wait_for_layer_load(L)
    direct -> compute stream waits fence_event[L]; no host completion gate
    staged success -> scatter on read-copy stream -> compute waits scatter fence
    direct failure -> forward may finish, but step output is discarded and request recomputes
    staged failure -> no scatter; report invalid blocks and drain submitted I/O

  vLLM compute layer L on compute stream

  save_kv_layer(L)
    record compute fence_event[L]
    direct: submit read layer L+1 and record its fence, if any
    direct: write stream waits fence_event[L] -> submit KV page writes
    staged: gather paged KV -> gather event -> write stream waits -> submit writes

step exit:
  wait_for_save
    drain save handles and both banks
    confirm_store(success/failure)
    release pins and step state
```

### 7.2 Load 依赖

```text
read stream:
  enqueue layer batches in order:
    descriptor H2D -> submit kernel -> CQ poll -> fence_event[L]

direct consumer:
  compute stream wait_event(fence_event[L]) -> compute L

staged host gate:
  Runtime progress -> structured terminal result
  staged: read completion -> read-copy scatter -> scatter event -> compute L
```

`fence_event[L]` 只建立 GPU 可见性和 stream 顺序，不表示 NVMe success。direct 中
每个 `(block_id, layer_id)` 都是独立 KV page，不存在 staging 槽覆盖或复用。为了
避免全层 Python submit 阻塞首个 compute enqueue，`start_load_kv` 只提交 read 0；
after-layer callback 执行时 compute L 已 enqueue，再提交 read L+1。这样 read L+1
的 host preparation/I/O 才能与 compute L overlap。direct read 的 structured failure
由 completion watcher异步收集，不作为逐层host gate。完整证据和状态机见
`doc/Tutticonnector/README.md`。失败后：

- direct 已排入的 compute 可以继续，但本 step 输出必须丢弃，受影响 request 的
  block IDs按现有connector合同上报并整请求重算；
- staged 本层不scatter，后续external save跳过；
- 两种路径都必须 drain/release 已提交 completion；失败不发布 resident marker。

### 7.3 Save 依赖

```text
direct:
  compute stream: compute L -> fence_event[L]
  write stream:   wait fence_event[L] -> direct KV page write kernel -> CQ

staged fallback:
  compute stream: compute L -> gather L -> gather_event
  write stream:   wait gather_event -> staging write kernel -> CQ
```

direct 省掉 gather，但不能省掉 compute 到 write 的 event。host callback 发生在
compute kernel enqueue 之后，不等于另一个 CUDA stream 已获得依赖；必须在当前
compute stream record `fence_event[L]`，再由 write stream wait。staged 的 bank 隔离
同样只解决内存覆盖，不自动建立跨 stream 依赖。

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

staged 路径的 `io_granularity` 是层段大小 `S`；direct 整池注册初期使用 0，按
在途请求动态构建 descriptor/PRP。该参数不表示 MDTS。生产 preset 不接收 MDTS；
LocalNvmeDataPath attach后通过`ioctl_get_dev_info()`读取
`dev_info.max_data_size`，得到hardware/effective MDTS，并在C++中计算：

```text
bytes_per_slice = staged prebuilt ? S : current direct request length B
ios_per_slice   = ceil(bytes_per_slice / effective_MDTS)
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

TP failure barrier不传播allocator产生的raw block ID，更不能把离散ID压缩成
`[min, max]`。每个rank在现有TP CPU process group上以固定数值tensor交换：

```text
step generation
+ repeated(request ordinal, first failed chunk ordinal, affected chunk count)
```

所有rank合并逻辑区间后，使用本rank同一request的block table重建invalid IDs。
因此离散block ID、同一步多request和多rank同时失败都不会误伤无关block。callback
只记录失败且不scatter；旧forward完成全部模型collective后，在connector finalize中
执行一次failure collective、各rank drain，再执行failure-only drain barrier。

async scheduler可能已在失败step之后预留依赖其sample token的后续step。当前失败
sample在worker bookkeeping前通过discard mask撤销；若已有下游placeholder，则使用
既有preempt/stale-output协议清空完整request执行状态，并由scheduler权威
`all_token_ids`重建worker persistent batch。只减少一个placeholder不足以恢复语义。
speculative decode的deferred connector finalize还必须同步撤销accepted/sample token、
draft token、`num_tokens_no_spec`和generator offset。TP collective超时会把该process
group视为poisoned并要求worker/process-group重启，不在同一group上假装继续服务。

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

历史`hy3-visible-r1`报告中只有三条CUDA stream：Transformer compute为
`streamId=19`，read I/O为`31`，write I/O为`35`；scatter/reshape实际回落在
compute stream 19。NVTX projection不能替代CUDA stream ID判断。

当前已删除底层 step feeder：上层按 K 窗口构造普通 byte-range batch，每次
DataPath submit 只包含当前已物化层/窗口；DataPath 不接收 layer plan 或 gate。

当前eager实现不使用CUDA Graph。一次普通 submit 只表示当前层/窗口的 operation
handle 和 arena lease，不表示一个 GPU kernel 覆盖整步：

```text
start_load_kv:
  Worker/KVEngine builds current layer/window byte-range descriptors
  submit one ordinary batch; the next window is admitted by host/event ordering

wait_for_layer_load(L):
  host waits for the current ordinary completion
  read-copy stream enqueues scatter(L) and records scatter_done[L]
  compute stream wait_event(scatter_done[L])
  read-copy stream publishes release[L] after scatter

save_kv_layer(L):
  gather current layer into staging, then submit current layer's ordinary write batch
```

read/write staging必须保证同时在途layer物理隔离，不能让未scatter/未write的
数据被后续层覆盖。默认资源模型是每个stream固定`K`个layer slot。read方向由
`Worker/KVEngine` 预先构造层计划，先提交一个`K`层窗口；后续窗口只在前一窗口
的 read-copy scatter/release 已完成后由 host feeder 提交。每个窗口内部由一个
普通 DataPath submit 发一个当前窗口 kernel，DataPath 不排 future wait，也不负责
推进后续窗口。write 方向遵循 legacy：当前 compute stream 完成后，逐层在 write
stream 提交当前层的 gather+write，不能把未 ready 的后续层交给 C++ feeder。

Hy3按实际25/32 chunks、256 KiB segment、`K=2`计算，read约12.5 MiB/rank，write
约16 MiB/rank；窗口 submit 不增加全层 HBM，只保留固定 read/write bank。窗口
推进使用已完成的 host/runtime completion 和现有 stream event，不依赖 future wait。

### 13.4.1 Legacy 连续 enqueue 对照

`examples/layerwise_kv_overlap/layerwise_kv_overlap.cpp`中的 legacy 路径在 host
循环中按层执行：先在 read stream 提交`read(L+1)`并记录`er[L+1]`，compute stream
等待`er[L]`后执行 compute(L)并记录`ec[L]`，write stream等待`ec[L-1]`后提交
`write(L-1)`。同一层的所有 chunk 在一个 DataPath batch 内聚合；层间依赖仍由
host/stream event 编排。当前 A 路径保持这个边界：`Worker/KVEngine` 管理层序、
窗口和 staging 生命周期，C++ DataPath 只提交当前窗口/当前层并回报 completion。

Nsight CUDA activity用于独立测量 kernel 实际执行间隔；host submit 连续不等于
NVMe kernel 同时执行。

### 13.5 真实 Hy3 逐层准入

2026-08-27诊断确认Hy3的物理KV层与实际callback都是完整顺序0..79；旧结论中
“callback缺层”并非根因。此前的底层 feeder/gate 方案因下沉了 layer 调度和生命周期而删除；当前由
Worker/KVEngine 逐层提交普通 batch。

当前约束：

- callback ordinal 到物理 KV ordinal 的映射只存在于 Worker/KVEngine。
- 重复/乱序/缺失 callback 在上层处理并 fail-closed，不在 Runtime 中轮询 layer flag。

历史Hy3 TP4 eager A/B及Nsight证明了B命中6400 tokens和四 stream 布局。
READ-PREENQUEUE-16后的新实测显示，stream分离不等于GPU interval
重叠：read stream 31的40个window在每个GPU上先连续执行，FlashAttention随后才
开始，read/FlashAttn interval overlap为0；write/FlashAttn只有少量偶发重叠。
当前H20/580.105.08对“先`cudaStreamWaitEvent`、后`cudaEventRecord`”不提供可用
future-event依赖；不得把这种调用序列当作K=2 bank fence。保持K=2与完整pre-enqueue
时，稳定安全的重叠需要全量layer staging或拆分submit-only/CQ-progress ABI。

single submit与single kernel必须严格区分：Hy3 80个callback在`K=2`下仍是40个
read window kernel，不是一个kernel发出全部80层I/O。一个submit只减少host端
operation/descriptor/arena生命周期，不消除有界window调度。

read-copy修复后每rank有四类stream：compute、NVMe read、scatter/copy和NVMe
write。scatter在独立read-copy stream enqueue；`scatter_done[L]`让compute只等待
当前层，同时release kernel排在同一copy stream的scatter之后，保证K=2 bank不
提前覆盖。理论上下一层read/scatter可与`compute[L]`和`write[L-1]`并行，但
IO-COMPUTE-OVERLAP-17实测的单block CQ-poll feeder与Hy3首层调度没有产生这种
GPU interval overlap。该路径禁止`event.synchronize()`/`cudaStreamSynchronize()`
作为正常依赖；若需稳定重叠，下一步应拆分submit-only与poll阶段或引入经验证的
full-layer staging，而不是只改变stream颜色。

2026-09-03 的 `write22-batched-final-20260903.nsys-rep` 进一步确认当前 staged
实现已将 read、read-copy、write 与 compute 分流，并把 cross-layer scatter 从
逐 chunk 小 kernel 合并到逐层 batch；write 与主 GEMM/MoE 已有实际区间重叠，read
与主要 compute 仍无稳定重叠。该结果也暴露出 staging 本身的额外 gather/scatter
kernel 和 HBM 占用。因此下一阶段改为 direct-first：注册 cross-layer KV pool，按
block table 直接生成 byte-range I/O；仅不满足 5.1 准入条件时走现有 staged 路径。

## 14. 当前状态

| 能力 | 状态 |
|---|---|
| KVCacheConfig自动几何 | 已实现 |
| DCP>1 fail-fast | 已实现 |
| required NHD | 已实现 |
| single-group staged transfer | 已实现 |
| single-group direct KV page transfer | Python backend、event桥和memory lifecycle已实现；待真机准入 |
| direct-first / staged-fallback selector | 已实现；按layout/alignment/capability在首个I/O前选择 |
| split read/write bank | 已实现 |
| local/striped双stream routing | 已实现 |
| 独立read-copy stream与scatter event桥接 | 已实现并通过真实Hy3/Nsight准入 |
| staged read plan 全量 pre-enqueue（K=2 future-event fences） | 已回退；生产路径为 host-window submit，禁止 future wait |
| direct rolling read orchestration | 设计完成，待把当前 `_ReadPlan._submit_all()` 改为 start-R0/after-layer-Rnext |
| 实际 read/compute GPU interval overlap | staged 当前实测为0；direct rolling实现后重新验收 |
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
| Runtime窗口自动回压 | 已实现为上层 host/window completion，底层无 feeder |
| per-stream compact workspace | 设计完成，未实现 |
| terminal result有界生命周期 | 未实现 |
| replacement generation原子切换 | 未实现 |
| rank-local I/O device placement | 已实现并通过Hy3 TP4复验 |
| scheduler metadata-only role | 已实现，runtime factory 单测为 0 调用 |
| TP all-rank commit gate | 已实现，任一rank缺失/异代时整chunk fail-closed |
| step-level feeder | 已删除；上层逐层/窗口构造普通 `DataPathRequest[]` |
| callback ordinal map | 已实现于 Worker/KVEngine；subset/乱序/重复/缺失在上层处理 |
| TP load failure logical consensus | 已实现；按request/chunk传播并在各rank重建离散block ID |
| async failure semantic rollback | 已实现；non-spec与spec deferred-finalize合同覆盖 |

## 15. 验收条件

### 15.1 正确性

- staged read失败不scatter；direct read失败允许已排入的compute继续，但不得提交
  本step输出或把失败数据标记为resident。invalid IDs在同一forward输出上报。
- `recompute`丢弃失败step输出并整请求重调度；`fail`终止请求。direct 不增加逐层
  host completion gate。
- 编程错误不会被误吞成KV miss。
- write失败不创建完整marker、不发布resident。
- partial commit不重复提交accepted request。
- read/write bank物理区间不重叠，总staging HBM不增加。
- direct read 不经过 scatter，direct write 不经过 gather；读取页必须与原 KV page
  byte-for-byte 一致。
- direct write严格发生在对应层compute fence后；staged gather严格发生在compute
  后，write严格发生在gather后。
- direct准入失败在首个I/O前确定性回退；strict模式必须报告具体不满足的条件。
- local/striped failure detail对称。

### 15.2 性能

- trace可见`read/transfer[L+1] || compute[L] || write[L-1]`。
- read/write使用不同stream，compute使用vLLM当前stream。
- routine path无step内memory registration。
- direct trace无staging allocation、gather/scatter和read-copy stream kernel；KV pool
  每个rank只注册一次。
- block size 64/128/256 的离散block table round-trip均正确；128/256真机用于比较
  IOPS、吞吐、TTFT和KV容量，不能只因单次I/O更大就默认更快。
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

1. 提交已经完成的 single-group cross-layer direct backend、memory unregister、
   `2*num_layers` capacity 和 staged fallback 基线。
2. 按 `doc/Tutticonnector/README.md` 把 direct read 改为 start-R0、after-layer-Rnext，
   保留 per-layer event 与整请求 recompute，不修改 Runtime/DataPath。
3. 用小规模合成测试验证精确 enqueue 顺序，再用真实Hy3 TP4比较 block size
   64/128/256 和 Nsight timeline。
4. 若且仅若真实 KV pool 基址不能满足 64 KiB 注册约束，再评审“aligned containing
   allocation + logical base offset”的最小 Runtime API；不得在 DataPath 下沉 layer
   或 block-table 语义。
5. 修复packed NHD/padded page canonical测试缺口。
6. 实现HMA/multi-group，再启用Qwen Tutti数据面。

当前不以CUDA Graph为实施阶段。eager逐层流水正确、可恢复、可观测之后，再根据
CPU submit开销和GPU/PCIe/NVMe利用率决定是否需要更激进的sequence API。

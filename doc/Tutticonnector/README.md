# Tutti vLLM Connector 调度编排

> 状态：调度编排专项文档，更新于 2026-09-03。
> 范围：vLLM eager 模式、single-group direct KV page I/O，以及 staged fallback。
> 本文只定义上层调度、CUDA stream/event 和 Runtime submit 的关系。模型 layer、
> block table、callback ordinal 等语义不得下沉到 Tutti Runtime/DataPath/kernel。

## 1. 结论

legacy Tutti 能稳定形成 overlap，关键不是 stream 数量，也不是一次提交了多少层，
而是 **host enqueue 顺序**：

```text
read[0] ready
enqueue compute[0]
enqueue/write or wait write[-1]
submit/wait read[1] while compute[0] is already running
enqueue compute[1]
submit/wait write[0] and read[2] while compute[1] is already running
...
```

当前 staged connector 虽然已经把 read、read-copy、compute、write 分到不同 CUDA
stream，但在 `start_load_kv()` 内用 Python 循环提交全部层 read/scatter。该函数不
返回，vLLM 就不能 enqueue 第一层 compute。实测该 host 区间约 288--303 ms；在这段
时间内 GPU 已经把大部分或全部 read 执行完，因此 timeline 表现为：

```text
all read -> all/most scatter -> compute
```

这不是 CUDA stream 依赖错误，而是 compute 的 enqueue 时间太晚。

direct I/O 解决了 staging HBM、gather/scatter 小 kernel 和槽复用问题。提交
`DIRECT-ROLLING-ORCH-24` 已把 direct 从 `_ReadPlan._submit_all()` 分离为独立 rolling
plan；staged 仍保持原计划。该实现已通过 fake-runtime/adapter 合同，尚待真实 Hy3
和 Nsight Systems 证明 GPU interval overlap。

## 2. 三份 legacy 证据必须区分

### 2.1 可复现的权威参考负载

`examples/layerwise_kv_overlap/layerwise_kv_overlap.cpp` 的 Phase G 是当前最明确的
工作参考。该逻辑最早来自提交 `26f1f7ef`，后续 TuttiRuntime 移植仍保留相同时序。

它具有以下特征：

- K/V tensor 直接注册，NVMe DMA 不经过 staging；
- `s_r`、`s_c`、`s_w` 三条独立 CUDA stream；
- 每层所有 chunk 聚合成一个 Runtime/DataPath batch；
- `er[L]` 表示 read stream 上 layer L 的工作已完成；
- `ec[L]` 表示 compute stream 上 layer L 的计算已完成；
- host 的 `windowed_submit_wait()` 会等待并 release 当前 I/O operation。

Phase G 的实际顺序是：

```text
prefetch and finish read[0]

for L:
  compute stream waits er[L]
  enqueue compute[L]
  record ec[L]

  write stream waits ec[L-1]
  submit and host-wait write[L-1]

  submit and host-wait read[L+1]
  record er[L+1]
```

即使 host 正在等待 I/O，`compute[L]` 已经先排入 `s_c`，所以 GPU 可以并行执行。
legacy 的低 operation window 也来自这里：每次 submit 后都 wait/release，并没有在
Runtime 中同时保留 80 个 read 和 80 个 write operation。

### 2.2 legacy InfiniKV/GeminiFS connector

`third_pkgs/tutti-legacy/engine/pkg/v1/storage_backend/geminifs_backend.py` 同时保留了两种
读取方式：

- `batched_get()`：遍历所有层，并在两个 load stream 上提交；最后 synchronize；
- `layerwise_batch_get()`：当前 compute stream 等 load stream，然后在 load stream
  发起下一层读取。

后者体现了 rolling prefetch 的设计意图。但当前 checkout 的 legacy vLLM adapter
存在多处实验残留：一些 dedicated `load_stream/save_stream` 参数被注释后换成
`torch.cuda.current_stream()`，save 又可能被 `save_stream is None` 直接跳过。因此它
只能用于理解意图，不能单独作为“当前代码一定能 overlap”的运行证据。

### 2.3 旧文档中的 all-layer DAG

历史文档经常把 `batched_get()` 描述为 all-layer overlap。需要严格区分：

- 全部 read 已经 enqueue 到 GPU，不等于 compute 也已经 enqueue；
- 多条 stream 不会自动改变 host API 的调用顺序；
- 若提交 80 层本身耗时接近全部 read 执行时间，GPU 会在 compute 出现前清空 read
  队列；
- 真正的一次性 GPU DAG 需要低开销 sequence submission，或者在同一次调用中返回
  可逐层消费的 fence。当前 Python/Runtime 接口没有这个能力，也不应把 layer 语义
  下沉到 DataPath 来临时实现。

## 3. 当前 connector 为什么没有 read/compute overlap

### 3.1 当前 host 调用链

```text
vLLM forward entry
  -> WorkerImpl.start_load_kv()
     -> KVEngine.start_read_plan()
        -> _ReadPlan.__init__()
           -> _submit_all()
              for every layer:
                Runtime.submit(read layer)
                record read event
                staged only: enqueue scatter and scatter event
  <- start_load_kv returns

vLLM layer loop starts here
  -> wait_for_layer_load(0)
  -> compute layer 0
```

`_submit_all()` 完成前，vLLM 的 layer loop 尚未开始。这是当前 read/compute 不重叠的
第一原因。

### 3.2 staged 路径进一步放大 launch phase

staged 模式每层还包含：

```text
read stream:      metadata -> NVMe read -> read_done[L]
read-copy stream: wait read_done[L] -> scatter/index kernels -> fence[L]
compute stream:   wait fence[L] -> compute[L]
```

cross-layer scatter 虽已从逐 chunk 合并成逐层 batch，仍会产生 cat、div、remainder、
index 等 helper kernel。它们延长 `start_load_kv()`，也让 Nsight 中的数据搬运活动看起来
与计算混杂。

### 3.3 stream 实际没有共用

`write22-batched-final-20260903` 的 CUDA activity 显示：

| 工作 | CUDA streamId |
|---|---:|
| vLLM compute | 19、27 |
| NVMe read kernel | 31 |
| staged scatter/copy | 35 |
| NVMe write kernel | 39 |

所以当前问题不是同一个 CUDA stream 上发生了 read、write 和 compute。NVTX range 是
host thread 范围，Nsight 的 range projection 也可能跨多条 stream；不能只看 range
所在行判断 kernel stream。direct 路径完成后，stream 35 及其 helper kernel 应完全
消失，只保留 read、compute、write 三类 stream。

### 3.4 为什么 write 已经能 overlap

`save_kv_layer(L)` 在 layer L 的 compute kernel 已经 enqueue 后才被调用：

```text
compute stream: compute[L] -> compute_done[L]
write stream:   wait compute_done[L] -> write[L]
```

因此 write 的 host submit 虽然逐层发生，但 compute 已经在 GPU 队列中。这个顺序与
legacy 成功路径一致，所以 profile 已能观察到 write 与 GEMM/MoE overlap。

## 4. direct 路径的目标编排

### 4.1 只保留真实依赖

direct 中每个 `(block_id, layer_id)` 对应独立 KV page，没有 staging slot，也没有
覆盖或复用关系。所需依赖只有：

```text
read[L]    -> compute[L]
compute[L] -> write[L]
```

不需要：

- staging acquire/release；
- read-copy stream；
- scatter/gather；
- slot release flag；
- device-side ready/release gate；
- future-event feeder；
- 在 DataPath 中理解 layer。

### 4.2 利用 vLLM 前后层 callback 交错 enqueue

为了保证 compute 不再被全层 read 的 host 提交阶段挡住，目标时序为：

```text
start_load_kv:
  prepare request keys/block tables/targets
  submit direct read[0] on read stream
  record read_ready[0]
  return to vLLM immediately

wait_for_layer_load(L), before layer compute:
  compute stream waits read_ready[L]
  no Runtime submit
  no host completion wait
  return

vLLM:
  enqueue compute[L]

save_kv_layer(L), after layer compute enqueue:
  record compute_done[L] on current compute stream
  submit direct read[L+1] on read stream, if any
  record read_ready[L+1]
  write stream waits compute_done[L]
  submit direct write[L] on write stream, if this step saves data
  return
```

这里把 read advance 放在 after-layer callback，而不是 before-layer callback，原因是
`Runtime.submit(read[L+1])` 可能有显著 host 开销。after-layer callback 执行时
`compute[L]` 已经 enqueue，host 准备下一层 read 的时间才有机会被计算覆盖。

内部可以把 `save_kv_layer()` 视为 vLLM 提供的 after-layer hook。推进 read 必须独立于
“本请求是否需要保存”：即使没有 save plan，该 callback 也要提交下一层 read。

### 4.3 目标时间线

```text
read stream:    R0 -------- R1 -------- R2 -------- R3 --------
                     |           |           |
                     v           v           v
compute stream:      C0 -------- C1 -------- C2 -------- C3 ----
                       |           |           |
                       v           v           v
write stream:          W0 -------- W1 -------- W2 -------- W3 --
```

event 只连接竖向真实依赖。read stream 内按层自然有序，write stream 内也按层自然有序。

### 4.4 同层 batch 与 kernel 数量

每个 layer 的所有 chunk/block 先在 Python 中展开为一个 byte-range request batch，再
调用一次 Runtime.submit：

```text
read kernel count  ~= loaded_layers
write kernel count ~= saved_layers
scatter/gather     = 0
```

DataPath 仍可按 FIEMAP extent、stripe boundary 和 controller MDTS 将 request 展开为
sub-I/O entry，但一个 layer 不应退化成每 chunk 一个 kernel。

## 5. operation capacity 的两种模式

### 5.1 当前 nonblocking direct 模式

CUDA kernel enqueue 是异步的。即使 read 按 after-layer callback 推进，Python 仍可能
很快遍历完 80 层，在 GPU 把早期 operation 标记 terminal 前排入大部分 read/write。
因此当前每层一个 public Runtime operation 的实现需要保守容量：

```text
max_in_flight_operations >= 2 * num_layers
```

Hy3 的 80 层 profile 可用 192。这里的 160/192 是当前 public operation/arena 生命周期
造成的 admission 要求，不是 CUDA stream 的物理并发度。

### 5.2 legacy bounded 模式

reference workload 每次 submit 后 host wait/release，所以只需要很小的 operation
window。它仍能 overlap，是因为 compute 已经先 enqueue，再进入 I/O wait。

如果后续必须把 `max_in_flight_operations` 降回 4，可选择：

- 在 after-layer 阶段等待/回收最老 operation，让等待被已 enqueue 的 compute 覆盖；
- 使用独立 host progress/feeder 维持有界 K-window；
- 未来实现严格限定为 stream-owned workspace 的 sequence submit。

这些是资源优化，不应再次引入 device future gate，也不应让 DataPath 获得 layer 语义。

## 6. 错误处理

direct read 的不同 layer/page 地址彼此独立，不存在 staging 覆盖风险。正常 callback
不做 per-layer host completion 检查：

- `read_ready[L]` 只表达 CUDA stream 顺序和内存可见性，不表达 I/O 成功；
- completion watcher 异步收集 structured result；
- 某个 read 失败后，已经 enqueue 的 compute 可以继续，以维持 eager/TP 执行；
- `wait_for_save()` 在 forward 结果交还 scheduler 前 drain completion 并执行 TP failure
  consensus；
- 失败 step 的输出被丢弃，受影响请求按 `invalid_block_ids` 整请求重算；
- 失败数据不得发布 resident/rank commit marker。

`kv_load_failure_policy=recompute` 才启用自动重算；`fail` 仍终止请求。编程错误、地址
越界和不支持的布局不能伪装成普通 cache miss。

## 7. 当前实现与目标的差异

| 项目 | 当前 direct 实现 | 目标 |
|---|---|---|
| KV 数据路径 | direct page byte-range | 保持 |
| staging/gather/scatter | direct 已不需要 | 保持为 0 |
| read 提交位置 | direct 已实现 start-R0/after-layer-Rnext | 真机确认 host 区间和 GPU overlap |
| compute wait | per-layer `fence_event` | 保持 |
| write 依赖 | compute event -> write stream | 保持 |
| host completion gate | callback 无 wait | 保持 |
| operation capacity | `2 * num_layers` | 第一版保持；后续再做有界优化 |
| failure | watcher + step-end recompute | 保持并补 rolling failure 测试 |

该状态机已经实现于 Worker/KVEngine，未修改 direct 地址映射、Runtime/DataPath 或
CUDA I/O kernel。下一阶段只做真机准入；如 timeline 不符合本文，不得直接修改底层。

## 8. 状态机约束

direct rolling plan 至少维护：

```text
physical_layers
handles[layer]
read_ready_events[layer]
next_read_to_submit
waited_callbacks
advanced_callbacks
terminal_failure
```

必须满足：

- layer 0 恰好提交一次；
- after-layer L 恰好提交 L+1 一次；
- duplicate callback 幂等；
- callback ordinal 与 physical layer 不一致时 fail-closed；
- 最后一层不提交越界 read；
- 没有 save keys 时仍推进 read；
- read failure 后不发布 resident，abort/finalize drain 所有已提交 handle；
- 新 step 开始前旧 plan 已 finalize；
- staged fallback 保持现有窗口和复用合同，不混用 direct rolling 状态。

## 9. Nsight Systems 验收

### 9.1 必须观察到

- `start_load_kv` 只覆盖 metadata/target preparation 和 layer 0 read，不再覆盖 80 层
  submit；
- `read[L+1]` 的 CUDA interval 与 `compute[L]` 存在稳定重叠；
- write 与后续 compute 继续 overlap；
- read、compute、write 是三个不同 CUDA stream ID；
- direct timeline 中没有 read-copy stream 上的 cat/div/remainder/index/scatter kernel；
- 每 rank 每个参与层最多一个 read I/O kernel、一个 write I/O kernel；
- normal path 没有 `event.synchronize()`、`cudaStreamSynchronize()` 或逐层 host
  `completion.wait()`；
- request B 的 external token 命中、输出和 recompute 行为正确。

### 9.2 不可作为证据

- NVTX range 的显示行或颜色不能证明 CUDA stream identity；
- host submit 已返回不能证明 NVMe 成功；
- read/write kernel 数量正确不能单独证明 overlap；
- 只看整个 request wall time不能定位 layer pipeline。

验收应同时查询 CUDA kernel `streamId`、各层 event/interval overlap 和 structured
completion 日志。

## 10. 实施顺序

1. 完成并提交 direct memory lifecycle、地址映射和 `2*N` capacity 基线。
2. 已将 direct `_ReadPlan._submit_all()` 改为 start-R0 + after-layer-Rnext。
3. 已用 fake runtime 测试精确 host enqueue 顺序，并禁止 callback host wait。
4. 使用小规模合成 CUDA/runtime 测试检查 stream/event，不加载大模型。
5. 最后只运行一次 Hy3 TP4 A/B 和 Nsight Systems 2026.4.1 准入。
6. direct 单组通过后，再单独设计 HMA/GDN 多 group；不能用 Qwen HMA 验收本阶段
   single-group direct 调度。

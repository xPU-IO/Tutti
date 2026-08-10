# 多加速器与单设备 StorageRuntime 设计

> **状态**：目标设计（business-first）
>
> 本文定义 Tutti 在一台机器上同时服务多个加速器和多块 NVMe 的业务模型。
> 它描述的是希望稳定下来的运行时边界，不等同于当前代码已经全部实现的行为。
> 实现细节以本文的身份和生命周期语义为准；严格的拓扑、DMA 和 I/O 探测可以
> 分阶段补齐。

## 1. 决策摘要

Tutti 的加速器后端在**编译期**选择，一个二进制只包含一种后端（例如 CUDA、
MUSA 或 MACA）。运行时只选择该后端中的一个 `accel_id`：

```text
一个 Tutti 二进制       -> 一个编译期 accelerator backend
一个 StorageRuntime      -> 一个 accel_id
一个 Runtime             -> 零个或多个同一 accel_id 上的 DataPath
多个 accelerator device  -> 创建多个 StorageRuntime
一个 daemon              -> 服务整台机器的所有 accelerator/NVMe 组合
```

因此，应用不在运行时切换 CUDA/MUSA/MACA，也不把多个 GPU 展开到同一个
`StorageRuntime`。需要同时使用 GPU 0 和 GPU 1 时，在同一进程中创建 Runtime 0
和 Runtime 1；两个 Runtime 可以并发提交。

一个 Runtime 通过 daemon 请求 NVMe **切片（lease）**。切片是一个客户端对
controller/namespace 的访问和队列配额，不是自动创建的 LBA 分区。一个 Runtime
可以拿到多个切片并组成 striped DataPath；多个 Runtime 也可以在 daemon 的队列
预算允许时共享同一 controller。

## 2. 目标与非目标

### 2.1 目标

- daemon 启动一次即可管理机器上全部配置的 GPU 关联和 NVMe controller。
- 应用按 Runtime 的 `accel_id` 获取允许使用的 NVMe 集合和实际设备路径。
- Runtime、DataPath、stream、workspace、memory registration 和 DMA mapping
  的 accelerator 归属一致。
- 单设备 Runtime 的 API 简单、可诊断，跨 GPU 部署通过多个 Runtime 组合完成。
- daemon 不创建 CUDA/MUSA/MACA context，也不参与每个 I/O 的数据路径。
- 资源分配失败在创建或 attach 阶段可见；数据路径不会静默切换到另一块盘或另一
  个 GPU。

### 2.2 非目标

- 一个 Runtime 在运行中迁移到另一个 accelerator device。
- 根据 YAML 在运行时切换 accelerator vendor/profile。
- 一个 batch 横跨多个 Runtime 或多个 accelerator device。
- daemon 持有 client 的 SQ/CQ、GPU stream 或 device pointer。
- 第一阶段就实现跨 GPU memory、跨 GPU stream 或自动 P2P 路由。
- 把 `available` 误解为已经完成一次真实 GPU I/O 的证明。

## 3. 术语与身份

必须区分下面各类身份。daemon 的 `DeviceInfo.device_id` 保留为 NVMe 资源的
本地编号；真正的加速器设备统一使用短名 `accel_id`，两者不能混用。

| 名称 | 所属 | 含义 | 稳定性 |
| --- | --- | --- | --- |
| `accel_id` | 编译期后端/Runtime | GPU、MUSA 或 MACA 设备编号 | 本机进程内有效 |
| `device_id` | daemon | daemon 管理的 NVMe controller 编号 | daemon 会话内有效 |
| `pci_bdf`、序列号、namespace | 物理设备 | controller/namespace 身份 | 机器级事实 |
| `allocation_id` | daemon/client 会话 | 一次 NVMe 切片租约 | 租约生命周期内有效 |
| `Runtime` handle ID | 应用进程 | Runtime 内的 target/memory/I/O 句柄 | 进程和 Runtime 生命周期内有效 |

`/dev/ssnvmeN`、`/dev/snvmeNn1` 和 mount/view path 都是 daemon 或 kernel
返回的**事实**，不是由 `accel_id`、YAML 数组序号或字符串拼接推导的身份。应用
配置可以保存 BDF 等稳定事实，数
据路径实际打开的 path 必须来自 daemon 返回的
allocation metadata。

## 4. 主机拓扑与责任边界

```text
                         control plane
  GPU 0  ───────────────┐                 ┌───────────────┐
  GPU 1  ───────────────┼── client RPC ──▶│ tutti_daemon  │
                         │                 │ owns all NVMe │
  NVMe 0 ────────────────┤                 │ ACL + leases  │
  NVMe 1 ────────────────┘                 └──────┬────────┘
                                                  │ metadata/path
                         data plane              ▼
  Runtime(accel_id=0) ────────────── client libnvm + snvme ── NVMe slices
  Runtime(accel_id=1) ────────────── client libnvm + snvme ── NVMe slices
```

### 4.1 daemon

daemon 是主机级资源 broker，负责：

1. 枚举并 bring-up 所有配置的 NVMe controller/namespace。
2. 保持 chrdev、bind、mount/view 和 controller 的生命周期。
3. 按 accelerator ACL、队列配额和 lease 状态分配切片。
4. 通过 RPC 返回真实的 chrdev、block、mount/view、BDF、namespace 和队列元数据。
5. 在 client 断开、心跳超时或进程死亡时回收 lease 记录。

daemon 不负责创建 GPU context、不 mmap client 的 ring、不提交 I/O。client 的
fd 关闭仍由内核回收 client-owned queue、ring 和 DMA map；daemon 只负责自身拥有
的 controller 状态。

### 4.2 StorageRuntime

Runtime 是应用进程内的单设备控制面和数据面入口，负责：

- 固定并暴露 `accel_id`；
- 向 daemon 获取一个或多个 NVMe allocation；
- 用 allocation metadata 构造 resolver、binding 和 DataPath；
- 管理 target、memory、I/O handle 以及 shutdown 顺序；
- 在 accelerator 调用前切换到 Runtime 的设备，并在调用后恢复线程原设备。

Runtime 不拥有全局 NVMe 设备表，也不把其他 Runtime 的句柄、memory 或 stream
混入自己的 submit。

## 5. 编译期后端与 Runtime 身份

编译配置选择唯一后端，例如 `-DTUTTI_ACCELERATOR=CUDA`。建议把编译结果导出
为只读常量 `TUTTI_COMPILED_ACCELERATOR_PROFILE`，并将公共配置改为：

```cpp
struct RuntimeConfig {
    std::int32_t accel_id = 0;
    std::uint64_t max_terminal_results = 64;
    std::string profile_name = TUTTI_COMPILED_ACCELERATOR_PROFILE;
};
```

语义如下：

- `accel_id >= 0` 表示绑定一个编译后端中的 accelerator device。
- `accel_id == -1` 只表示 host-only Runtime；它不是“自动选择任意 GPU”。
- `accel_id` 在 `StorageRuntime::create()` 成功后不可修改。
- 创建阶段校验 `0 <= accel_id < compiled_backend_device_count`；host-only
  Runtime 不需要 accelerator device。
- `profile_name` 是构建信息和诊断信息。配置中若指定 profile，只做与编译结果
  的一致性检查；不匹配直接报错，不触发运行时后端切换。
- Runtime 身份只需要不可变的 `accel_id`（以及普通的内部 handle/runtime ID）。
  `profile_name` 不参与 resolver、DataPath 或 NVMe 路由。

## 6. DataPath 绑定规则

通用 SPI 应暴露 accelerator 绑定信息，而不是只暴露一个容易误读的
`supports_multi_gpu`：

```cpp
struct DataPathCapabilities {
    // -1: host-only; >= 0: fixed accelerator device
    std::int32_t bound_accel_id = -1;
    bool supports_multi_device = false;
    bool supports_cross_device = false;
    // existing capability fields remain unchanged
};
```

规则：

1. `bound_accel_id >= 0` 的 DataPath 固定使用该 accelerator device。
2. `bound_accel_id == RuntimeConfig.accel_id` 才能挂载到 accelerator Runtime。
3. 同一个 Runtime 可以挂载多个 DataPath，但不能出现两个不同的
   `bound_accel_id`。
4. `bound_accel_id == -1` 的 host-only DataPath 不占用 accelerator 资源，只能做
   host execution。MVP 允许它挂载到 accelerator Runtime；`accel_id == -1` 的
   host-only Runtime 则只能挂载 host-only DataPath。
5. `supports_multi_device` 是 DataPath 自身的能力描述，不能绕过 Runtime 的单
   accelerator 边界，也不能使一个 submit 跨 GPU。
6. `supports_cross_device` 预留给未来跨 accelerator 数据路径。当前实现必须保持
   `false`，除非另有明确的跨设备内存和同步协议。
7. striped DataPath 可以管理多个 NVMe slice，但其执行 accelerator 仍只有一个，
   即 `bound_accel_id` 不变。多盘不等于多 GPU。

目标 DataPath 必须显式提供 `bound_accel_id`。缺少绑定信息或出现设备冲突时，
Runtime 在创建阶段拒绝；不能通过“未知设备”把 DataPath 挂到任意 Runtime。

## 7. daemon 资源模型与 RPC

### 7.1 配置模型

daemon 配置描述全机事实和 ACL，不为每个 GPU 启动独立 daemon：

```yaml
accelerators:
  - { accel_id: 0, mount_path: "/mnt/gpu0" }
  - { accel_id: 1, mount_path: "/mnt/gpu1" }

nvmes:
  - device_id: 0
    pci_addr: "0000:31:00.0"
    namespace_id: 1
    mount_path: "/mnt/nvme0"
    allowed_accel_ids: [0, 1]
  - device_id: 1
    pci_addr: "0000:52:00.0"
    namespace_id: 1
    mount_path: "/mnt/nvme1"
    allowed_accel_ids: [1]
```

`allowed_accel_ids` 是“可以申请”的 ACL，不是已经验证过 GPU DMA/P2P/I/O 的结论。
空 ACL 的默认值仍可表示所有已枚举 accelerator，但必须在 daemon 日志和 RPC
中明确展开后的集合。

### 7.2 逻辑 RPC

协议直接定义 NVMe 资源发现和切片分配：

```text
ListNvmeResources()
  -> 所有 daemon device_id 及其真实 path、BDF、namespace、队列能力、ACL

AcquireNvmeSlices(accel_id, selection, queue_budget)
  -> 一个 allocation_id + 一个或多个 NvmeSlice

NvmeSlice
  -> device_id, pci_bdf, chrdev_path, block_path, mount_path/view,
     namespace metadata, queue quota, lease parameters

Release(allocation_id)
```

`selection` 可以是显式 daemon `device_id` 列表、按 ACL 自动选择，或一个
striped group。实现可以在内部对每个 NVMe 分别建立 lease，但对 Runtime 暴露的
结果必须是一个 allocation 列表，且每个条目都带有明确的 daemon `device_id` 和
请求的 `accel_id`。

### 7.3 分配与共享

- daemon 只分配配置 ACL 允许的 slice，并依据队列池做配额控制。
- 默认允许多个 Runtime 共享一个 NVMe controller；独占模式作为部署策略，而非
  Runtime 身份的一部分。
- 一次 lease 只对应一个 client Runtime。Runtime 关闭时释放全部 allocation；
  不允许把 allocation 转移给另一个 Runtime。
- daemon 返回的 path 是本次 allocation 的事实。应用不得用
  `accel_id` 或数组序号拼接 `/dev/ssnvmeN`、`/dev/snvmeNn1`。

## 8. 应用配置与 Runtime 创建

应用配置直接选择一个 Runtime accelerator：

```yaml
accelerator:
  profile: CUDA                 # 可省略，必须与编译 profile 一致

runtime:
  accel_id: 1

nvme:
  selection: allowed             # allowed | explicit | striped
  device_ids: [0, 1]             # daemon device_id；explicit/striped 时可指定
  queues_per_controller: 4
```

加载器的职责是：

1. 读取 `runtime.accel_id`，创建一个 `RuntimeConfig`。
2. 只向 daemon 请求该 `accel_id` 允许的 NVMe slice。
3. 按返回的真实 metadata 创建一个或多个同 `accel_id` DataPath。
4. 发现 DataPath 绑定了其他 accelerator 时在 `create()` 失败；不要把所有
   `allowed_accel_ids` 展开成一个 Runtime。
5. 上层需要多 GPU 时重复执行上述流程，形成 Runtime 集合：

```cpp
RuntimeConfig cfg0;
cfg0.accel_id = 0;
RuntimeConfig cfg1;
cfg1.accel_id = 1;
auto r0 = create_runtime(cfg0, daemon, policy_for_gpu0);
auto r1 = create_runtime(cfg1, daemon, policy_for_gpu1);
// r0 和 r1 独立提交，可在同一进程并发运行。
```

配置文件中的 NVMe 数组顺序不是身份，也不决定 chrdev minor。若需要持久拓扑，
使用 BDF/序列号/namespace 作为事实键；daemon 启动时可执行 BDF、chrdev、block
三方校验，不一致则不发布 allocation。

## 9. 内存、stream 与 submit 语义

### 9.1 内存

业务层只允许以下清晰语义：

| MemoryKind | 归属 | MVP 行为 |
| --- | --- | --- |
| `HOST` / `PINNED_HOST` | `accel_id = -1` | 可注册到 host-only 或 accelerator Runtime，由 DataPath 决定是否支持 |
| `DEVICE` | Runtime 的 `accel_id` | 显式指定其他 `accel_id` 时拒绝；未指定时解析为 Runtime 的 `accel_id` |
| `MANAGED` | 由编译期 backend 管理，但仍绑定一个 Runtime 的 `accel_id` | 若 backend 不能保证归属，返回 `UNSUPPORTED`，不伪装成 host allocation |

`register_memory()` 可以接受外部 memory 的未指定设备（`expected_accel_id < 0`），
但在第一次被 device DataPath 注册时必须解析到 Runtime 的 `accel_id`。
显式冲突不应延迟到 I/O 完成后才暴露。

当前阶段不要求所有 accelerator 都实现 `allocate_memory(DEVICE/MANAGED)`；尚未
支持时返回 `UNSUPPORTED` 比调用 `malloc()` 后标记成 device memory 更安全。

### 9.2 submit

- `DEVICE_EXECUTION` 的 context、stream、target 和 memory 必须属于同一 Runtime
  `accel_id`；一个 batch 天然只对应一个 accelerator device。
- `context.accel_id == -1` 可解释为“使用 Runtime 的 `accel_id`”；显式不同值
  立即拒绝。
- `HOST_EXECUTION` 可用于 host-only DataPath；不能借此让 device DataPath 跨设备。
- Runtime 层做第一层归属检查，DataPath 保留第二层检查。stream 所属设备如果
  后端 API 可查询则 fail-closed；无法查询时把它作为调用方前置条件并记录诊断，
  不把该探针变成 daemon 启动的硬前置条件。
- submit/open/register/close/shutdown 的每条 accelerator 调用路径都使用保存、
  切换、恢复 current device 的 RAII guard；后台线程也必须显式设置目标设备。

### 9.3 句柄隔离

Target、Memory、IO handle 携带 Runtime 的内部身份和 generation。跨 Runtime
传入句柄一律失败；句柄不能因为两个 Runtime 使用相同的 `accel_id` 或相同的
NVMe slice 而互通。

## 10. `available` 的业务语义

RPC 应区分三个层次：

1. **allowed**：配置 ACL 允许该 accelerator 请求。
2. **available**：daemon 已 bring-up，且当前有可分配的 queue/lease 资源。
3. **validated_available**：独立 client-side probe 已完成 attach、ring/data map
   和最小真实 I/O。

MVP 的 daemon 可以只提供前两项，不创建 CUDA context，也不承诺第三项。应用在
`Acquire` 或 DataPath initialize 失败时应获得明确错误，并可释放 allocation 后
尝试其他允许的 NVMe；绝不静默改用错误路径。需要调度“可用 GPU/NVMe”时，由
单独的 probe worker 上报 `validated_available`，而不是让 daemon 本身进入 GPU
运行时。

## 11. 生命周期

```text
编译期选择 backend
        │
daemon 启动：枚举全部 NVMe，建立 controller/path，开始 RPC
        │
Runtime::create(accel_id)
        │  profile/accel_id 基本校验
        ▼
Acquire slices(accel_id, policy)
        │  返回 allocation metadata
        ▼
构造并 initialize 同 accel_id DataPath
        │
open/register/submit/wait/progress
        │  每次 accelerator 调用设置并恢复 current device
        ▼
drain I/O -> close targets -> unregister memory -> shutdown DataPath
        -> Release allocations -> Runtime STOPPED
```

初始化失败必须释放已经取得的 allocation 和已经初始化的 DataPath。Runtime
shutdown 不停止 daemon，不解绑其他 Runtime 正在使用的 controller。

## 12. 分阶段实现策略

### 12.1 MVP：先满足业务拓扑

以下项目是业务闭环的最小集合：

- `RuntimeConfig.accel_id` 和 immutable `bound_accel_id`。
- loader 按一个 `runtime.accel_id` 过滤和创建 DataPath；多 GPU 由多个 Runtime
  实例完成。
- daemon 一次管理全机 NVMe，Acquire 返回真实 path 和 allocation。
- 一个 Runtime 的多个 NVMe DataPath/striped DataPath 共用一个 accelerator device。
- Runtime/handle 不跨 `accel_id`；显式冲突在 create/register/submit 返回错误。
- host-only DataPath 的挂载策略按第 6 节执行；host-only Runtime 不使用
  accelerator DataPath。
- daemon 的 `available` 不等于 I/O 验证，应用可在 attach 初始化阶段得到失败。

DataPath 必须在创建时提供唯一的 `bound_accel_id`；它不能同时被两个不同
`accel_id` 的 Runtime 使用。

### 12.2 生产前的正确性增强

这些项目不改变业务模型，但在宣称多盘多 GPU 稳定可用前必须完成：

1. 保存 `libnvm` 实际获得的 chrdev minor/path；block path 使用 ioctl 返回的
   `disk_name`，禁止从 block 名推导 chrdev。
2. GPU DMA map、地址和释放逻辑只绑定当前 controller；5.15 与 5.4 分支保持一致。
3. 配置加载器注册一个 resolver/DataPath key，不为每个派生设备重复注册同一 key。
4. attach、BAR0/ring/data map 前设置目标 accelerator device；所有路径恢复原
   current device。
5. 增加冷启动、stale minor、双 Runtime 并发、非目标 current-GPU、不同
   IOMMU/P2P 拓扑测试，真实 I/O 不能只用 `--skip-io`。

### 12.3 后续能力

- 独立 probe worker 和 `validated_available` 状态。
- 跨 accelerator DataPath、peer memory 和 stream 归属查询。
- MANAGED memory 的明确迁移/一致性协议。
- allocation 的容量配额、优先级、NUMA affinity 和故障转移策略。

## 13. 验收场景

设计落地后，至少应能证明：

1. Runtime 0 和 Runtime 1 在同一进程内分别获得不同 GPU 的 NVMe slice 并发读写。
2. 一个 Runtime 绑定 GPU 1 时，只能看到允许 GPU 1 的 NVMe；不能把 GPU 0 的
   DataPath 注入进来。
3. 一个 Runtime 注入两个同 GPU 的 NVMe DataPath 成功；注入 GPU 0/GPU 1 混合
   DataPath 在创建阶段失败。
4. striped DataPath 可横跨多个 NVMe，但其 kernel、stream、workspace 和 DMA
   mapping 全部属于一个 accelerator device。
5. 当前线程先设置到另一 GPU 时，Runtime 0 的 initialize、submit、progress 和
   shutdown 仍使用 GPU 0，并恢复调用前的 current device。
6. stale minor、错误 block path 或 BDF 不一致时，不向 client 发布可用 allocation。
7. daemon 启动和 `ListNvmeResources` 不创建 CUDA context。
8. 跨 Runtime 的 target/memory/IO handle、错误 device memory 和错误 context
   在任何 allocation、DMA map、kernel launch 或 doorbell 之前被拒绝。

## 14. 与现有文档的关系

- daemon bring-up、module 顺序和 lease 的现状见
  [`doc/tutti_daemon.md`](../tutti_daemon.md) 和
  [`tutti/device_manager/nvme/nvmeservice/NVMeService.md`](../../tutti/device_manager/nvme/nvmeservice/NVMeService.md)。
- DataPath SPI 的通用生命周期见 [`backend-spi.md`](backend-spi.md)。
- local-NVMe 和 striped 的硬件 contract test 见
  [`doc/local_nvme_contract_tests.md`](../local_nvme_contract_tests.md)。

这些文档可以继续记录当前实现和测试命令；本文是多加速器部署时应遵循的目标
业务契约。

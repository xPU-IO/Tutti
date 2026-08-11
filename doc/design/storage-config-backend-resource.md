# Storage config、Backend 装配与 Resource 分离设计

> **状态**：目标设计，尚未实现
>
> 本文定义应用侧 Tutti config 的目标结构，以及 loader 在创建
> `TuttiRuntime` 时对 resolver、payload contract、DataPath 和 storage
> resource 的装配与生命周期语义。本文不改变 daemon 的部署配置，也不改变
> `StorageRuntime` 当前按 URI scheme 和 DataPath key 路由的核心契约。

## 1. 决策摘要

应用 config 只保留三个一级配置：

```text
accelerator
runtime
storage
```

`storage` 下声明四组带显式 ID 的对象：

```text
resources[]
resolvers[]
datapaths[]
backends[]
```

其中：

- `resource` 描述一次逻辑资源申请，不描述 resolver 或 DataPath 的实现参数；
- `resolver` 只描述名称解析行为，不包含设备路径和 controller 身份；
- `datapath` 只描述 I/O 实现和调优参数，不包含要使用的具体设备；
- `backend` 通过 ID 显式引用一个 resolver、一个 DataPath 和一个 resource，并声明
  resolver/DataPath 之间使用的 payload contract；
- 当前产品 loader 对一个 `TuttiRuntime` 只允许一个 backend，数组结构为后续
  多 backend 和实例共享保留表达能力；
- config 中的 backend 会在 loader 中被解析和实例化，进入 `StorageRuntime` 后仍然
  展开为 `scheme -> resolver` 和 `DataPath key -> DataPath` 两个 registry；
- resource 是 `TuttiRuntime` 的一等运行时对象，按 resource ID 注册和管理；
- `TuttiRuntime` 使用独立的 `tutti_runtime.h/.cpp` 实现，config loader 不再承载其
  生命周期和 registry 管理代码；
- 每个运行时 resource 都是一个具体 `Resource` 子类对象，使用与 DataPath 类似的继承
  接口；
- `NvmeResource` 作为具体 resource 实现，拥有 NVMe 专用 client、allocation、resource
  metadata 和 NVMe lease/release 状态；
- 公共 `Resource` 接口不包含统一的 ResourceClient 抽象，`TuttiRuntime` 通过 resource
  registry 拥有各类 ResourceInstance；
- DataPath 只拥有从 resource 建立的 controller、queue、workspace 等数据面私有
  状态，`StorageRuntime` 不直接管理 resource allocation。

## 2. 当前实现事实

当前 [`tutti_config.cpp`](../../tutti/config/tutti_config.cpp) 已经按下面的顺序创建
Runtime：

```text
解析扁平 config
  -> RuntimeResourceClient::acquire_nvme_slices()
  -> RuntimeNvmeAllocation
  -> 从同一个 slice/slices 构造 resolver 和 DataPath
  -> 生成 RuntimeComponents
  -> StorageRuntime::create()
```

单盘模式下，同一个 `RuntimeNvmeSlice` 的字段被拆给两个消费者：

- `LocalFileResolver` 消费 `pci_bdf`、`namespace_id`、
  `logical_block_size` 和 `block_path`；
- `LocalNvmeDataPath` 消费 `chrdev_path`、`bar0_size`、`accel_id`、
  `granted_queues`、`namespace_id`、`logical_block_size` 和
  `max_data_size`。

因此，NVMe 信息不是只属于 DataPath。它描述的是 resolver 和 DataPath 共同指向的
底层 namespace/controller，只是两个组件使用不同的只读视图。

`StorageRuntime` 本身不保存 backend/group。它接收
[`RuntimeComponents`](../../tutti/include/tutti/storage_runtime.h) 后建立：

```text
resolvers_[URI scheme] = resolver pointer
data_paths_[recommended DataPath key] = DataPath pointer
```

resolver 返回的 `ResolvedTarget::recommended_data_path_key()` 在 `open()` 时连接这两张
表。payload type ID 和 API version 由 DataPath 在读取 pair-private payload 时校验。

当前 [`TuttiRuntime`](../../tutti/config/tutti_config.h) 已经同时持有
`StorageRuntime`、resolver、DataPath、`RuntimeResourceClient` 和 allocation metadata，
但 resource 仍被拆成下面几个单例字段：

```text
resource_client
allocation_id
allocation_slices
allocation_released
```

当前 `TuttiRuntime` 也没有独立的实现文件：类型声明位于
[`tutti_config.h`](../../tutti/config/tutti_config.h)，析构和 `shutdown()` 实现位于
[`tutti_config.cpp`](../../tutti/config/tutti_config.cpp)。这使 Runtime 所有权和生命周期
逻辑与 config 解析、资源申请及组件装配逻辑耦合在同一个 loader 模块文件中。

这种结构没有 resource ID、类型和独立状态，并隐含一个 `TuttiRuntime` 只能管理一个
allocation。目标设计将这些字段收敛为一等 `ResourceInstance`，由 `TuttiRuntime`
通过 resource ID registry 统一拥有和管理。

## 3. 术语

### 3.1 Backend

一个 backend 是一个可实例化、可向 Runtime 提供完整存储能力的逻辑组合：

```text
BackendInstance
  = ResolverInstance
  + payload contract
  + DataPathInstance
  + ResourceInstance
```

backend 回答的是：

> 本次运行使用哪个 resolver、哪个 DataPath、哪次资源申请，以及它们遵守哪种协议。

backend ID 是 config 和诊断身份，不是 `StorageRuntime` 的路由 key。

### 3.2 Payload contract

payload contract 是 resolver 输出与 DataPath 输入之间的类型和版本契约。现有 contract
位于 `tutti/bindings/<name>/binding.h`，定义：

- resolver type ID；
- payload type ID；
- payload API version；
- recommended DataPath key；
- payload C++ 类型；
- `make_resolved_target()` 和 `view_payload()`。

contract 是类型/协议级定义，一个 contract 可以被多个 backend 实例复用。backend
则是实例/部署级关系。配置不直接暴露 payload type ID 或 API version，而是使用稳定
的逻辑 contract 名称，由 loader 映射到编译进二进制的具体 contract 版本。

### 3.3 ResourceSpec、ResourceInstance、Resolver 与 DataPath 私有状态

必须区分资源本身，以及 resolver 和 DataPath 基于资源建立的两类不同私有状态：

| 层次 | 含义 | 典型内容 | 所有者 |
| --- | --- | --- | --- |
| `ResourceSpec` | 声明需要什么资源 | provider、selection、device IDs、请求的队列配额 | config |
| `ResourceInstance` | 一个具体 `Resource` 子类的运行时实例，表示实际获得或创建的资源 | 实现私有状态（例如 NVMe client/session 与 allocation，或 Memfs backing）、只读 metadata、生命周期状态 | `TuttiRuntime` 的 resource registry |
| Resolver 私有状态 | 如何在 host 侧解释资源并解析逻辑 target | URI scheme、namespace/backing identity、view/path 规则、FIEMAP 参数、shard resolver 集合、stripe layout 参数 | Resolver |
| DataPath 私有状态 | 如何使用资源执行 I/O | controller attach、queue group、PRP/cache/workspace | DataPath |

resource 在配置和所有权上独立，不意味着 DataPath 不处理资源。准确边界是：

```text
具体 Resource 实现拥有自己所需的实现私有状态、使用设备的权利和 lease
Resolver 拥有基于该资源进行名称解析和物理映射的 host-side 状态
DataPath 拥有使用该设备执行 I/O 的私有运行状态
```

其中 provider client 不是 Resource 的公共组成部分。只有需要访问外部 provider 的具体
实现才持有自己的专用 client；例如 `NvmeResource` 持有 `NvmeResourceClient`，而
`MemfsResource` 不需要 client。

Resolver 私有状态与 DataPath 私有状态的区别是：

| 对比项 | Resolver 私有状态 | DataPath 私有状态 |
| --- | --- | --- |
| 主要职责 | URI/name 到逻辑 target 和物理 extent 的解析 | 将 resolver target 降低为实际 I/O 并提交、推进和完成请求 |
| 执行位置 | 主要在 host，可能执行 stat、FIEMAP 和路径校验 | accelerator/transport 数据面，涉及 controller、queue 和 device workspace |
| 典型输入 | scheme、view path、backing block identity、namespace identity、stripe layout | chrdev path、BAR size、granted queues、硬件 MDTS、block size、DataPath tuning |
| 典型输出/状态 | `ResolvedTarget` payload、logical size、owner lease、extent/shard 映射 | `DataPathTarget`、registered memory、in-flight operation 和 completion state |
| 不应拥有 | allocation lease、controller queue、GPU workspace | URI namespace 规则、FIEMAP 解析规则和 resolver 的 payload 生成逻辑 |

Resolver 可以复制或缓存 ResourceInstance 中自己需要的只读字段，但不拥有 resource
lease，也不能自行选择或申请另一个 resource。Local resolver 的 backing device identity
和 namespace 信息属于这种 resolver view；Striped resolver 的 shard resolver 集合和
stripe mapping 状态也属于 resolver 私有状态。它们描述“如何解释一个资源”，不等于
资源本身。

在 target 生命周期中，resolver 返回的 `ResolvedTarget` 还会携带 payload owner lease。
该 lease 保证本次 target 解析结果所需的 host-side 对象在 DataPath 使用期间有效；它
不是 ResourceInstance 的 allocation lease，也不允许 resolver 越过 `TuttiRuntime` 的
资源边界管理 controller 或 queue。

本文中的 storage resource 也不同于 DataPath SPI 中当前为空的
`tutti::ResourceProvider`。不得为了传递 NVMe 路径或 allocation metadata，把
transport-specific 字段加入公共 `ResourceProvider`。这些字段应由 backend factory
以 backend-private 类型或构造参数注入。

### 3.4 `TuttiRuntime` 与 `StorageRuntime`

本文区分两个不同层次的 Runtime：

| 类型 | 职责 | 是否拥有 resource |
| --- | --- | --- |
| `TuttiRuntime` | 应用侧完整 Runtime，拥有资源、组件、backend 关系和关闭流程 | 是，按 resource ID 拥有 ResourceInstance |
| `StorageRuntime` | 核心 I/O Runtime，管理 target、memory、I/O，并按 scheme/key 路由 | 否，只借用 resolver/DataPath 指针 |

目标设计不再引入额外的 `OwnedRuntimeBundle` 概念。`TuttiRuntime` 本身就是完整的
所有权聚合对象，`StorageRuntime` 是其中负责 I/O 路由的一个核心成员。

Resource 的公共继承接口可以与 DataPath 放在同一扩展层，但不应把 resource provider
协议强行统一：

```text
Resource                       DataPath
  -> capabilities()              -> capabilities()
  -> initialize()                -> initialize(config, ...)
  -> shutdown()                  -> shutdown(timeout)
  -> info()                      -> target/memory/op lifecycle
```

`Resource` 基类只定义生命周期、能力和只读信息。具体 Resource 的配置、获取协议和
可选的 provider client 由实现自己的 factory/类处理。这样 `NvmeResource` 可以依赖
`NvmeResourceClient`，而 `MemfsResource` 可以直接创建进程内 backing，不需要为了
满足统一接口而引入无意义的 client。

## 4. 目标配置结构

### 4.1 Local NVMe 完整示例

```yaml
accelerator:
  profile: CUDA

runtime:
  accel_id: 0

storage:
  resources:
    - id: nvme-local-0
      type: nvme
      provider:
        type: nvme-service
        endpoint: "127.0.0.1:50051"
      allocation:
        selection: explicit
        device_ids: [0]
        queues_per_controller: 4

  resolvers:
    - id: file-resolver-0
      type: local-file
      scheme: file
      config: {}

  datapaths:
    - id: local-nvme-datapath-0
      type: local-nvme
      config:
        handle_cache_capacity: 0
        prp_cache_capacity: 0
        handle_cache_l2_capacity: 0
        max_in_flight_operations: 0
        max_batch_entries: 0
        io_granularity: 0

  backends:
    - id: model-storage
      contract: ext4-local-nvme
      resolver: file-resolver-0
      datapath: local-nvme-datapath-0
      resource: nvme-local-0
      config: {}
```

### 4.2 Striped NVMe 示例

Striped backend 仍然只对应一个逻辑 resource。该 resource 的一次 allocation 返回
多个有序 slice，而不是让 backend 独立引用多块叶子设备：

```yaml
storage:
  resources:
    - id: nvme-striped-0
      type: nvme
      provider:
        type: nvme-service
        endpoint: "127.0.0.1:50051"
      allocation:
        selection: striped
        device_ids: [0, 1]
        queues_per_controller: 4

  resolvers:
    - id: striped-resolver-0
      type: striped-file
      scheme: striped
      config: {}

  datapaths:
    - id: striped-datapath-0
      type: striped-local-nvme
      config:
        handle_cache_capacity: 0
        prp_cache_capacity: 0
        max_in_flight_operations: 0
        max_batch_entries: 0

  backends:
    - id: striped-storage
      contract: striped-local-nvme
      resolver: striped-resolver-0
      datapath: striped-datapath-0
      resource: nvme-striped-0
      config:
        stripe_unit: 65536
```

`stripe_unit` 决定 resolver 和 DataPath 共同遵守的数据布局，因此属于 backend/contract
配置，不属于物理 NVMe resource，也不属于 DataPath 私有调优。

### 4.3 Memfs 完整示例

Memfs 不需要外部 provider 或 NVMe client。`MemfsResource` 负责创建并拥有进程内
backing；resolver 和 DataPath 只通过 resource-derived view 使用它：

```yaml
accelerator:
  profile: HOST

runtime:
  accel_id: -1

storage:
  resources:
    - id: memfs-memory-0
      type: memory
      config:
        capacity_bytes: 1073741824

  resolvers:
    - id: memfs-resolver-0
      type: memfs
      scheme: memfs
      config: {}

  datapaths:
    - id: memfs-datapath-0
      type: memfs
      config: {}

  backends:
    - id: memfs-storage
      contract: memfs
      resolver: memfs-resolver-0
      datapath: memfs-datapath-0
      resource: memfs-memory-0
      config: {}
```

该配置表示一个容量为 1 GiB 的逻辑 memory resource，以及一组明确绑定到该 resource
的 `memfs` resolver/DataPath。`memfs` 的 DataPath key、payload type ID、API version
和 resolver type ID 仍由 binding contract 固定，不能通过 config ID 改写。

当前 sample resolver 使用 `memfs://<bytes>` URI，并在 resolve 时创建 backing；这是
sample 的现状，不是目标 resource 所有权模型。目标实现中，`MemfsResource` 应在
`initialize()` 时创建 backing，resolver 根据 URI 选择或校验其中的逻辑 view，URI 请求的
大小不得超过 `capacity_bytes`。如果第一阶段不支持 backing 分片，则 URI 大小必须等于
该 resource 的容量。

## 5. 各配置对象的责任

### 5.1 `storage.resources[]`

每个 resource 条目表示一次逻辑资源申请。一个 NVMe resource 可以解析为一个 slice，
也可以在 striped 模式下解析为一个包含多个 slice 的原子 allocation。

ResourceSpec 可以包含：

- config ID；
- resource 类型；
- resource provider 类型和 endpoint；
- selection 策略；
- 显式 device ID 列表；
- 请求的每 controller 队列配额；
- resource-specific request config，例如 memory resource 的 `capacity_bytes`；
- lease/heartbeat 等控制面策略，若后续对应用开放。

ResourceSpec 不得包含 daemon 在运行时返回的事实：

- `allocation_id`；
- `pci_bdf`；
- `chrdev_path`、`block_path`；
- `backing_mount_path`、`view_path`；
- `bar0_size`、硬件 MDTS、logical block size；
- 实际 `granted_queues`；
- controller/queue/workspace 指针或句柄。

这些字段属于 ResourceInstance 或 DataPath 或 Resolver 私有状态。

### 5.2 `storage.resolvers[]`

resolver 配置描述 URI/name 到逻辑 target 的解析行为：

- config ID；
- resolver 实现类型；
- URI scheme；
- 与解析算法有关、但不标识具体 resource 的参数。

具体 block path、PCI BDF、namespace identity 和 block size 由 backend factory 从
ResourceInstance 注入，不在 resolver config 中重复。

### 5.3 `storage.datapaths[]`

DataPath 配置描述 I/O 实现和调优：

- config ID；
- DataPath 实现类型；
- cache capacity；
- max in-flight、batch limit、I/O granularity；
- 其他不标识具体 resource 的数据面策略。

`chrdev_path`、BAR size、namespace ID、硬件 MDTS 和实际 queue grant 不属于 DataPath
config。DataPath 在实例化时获得这些运行时事实，并在 `initialize()` 中 attach
controller、创建 queue group 和 workspace。

队列相关字段必须按语义拆分：

```text
queues_per_controller  -> ResourceSpec，表示向资源服务申请的配额
granted_queues         -> ResourceInstance，表示资源服务实际授予的配额
max_in_flight          -> DataPathSpec，表示 DataPath 的提交容量策略
queue group/queue ptr  -> DataPath 私有运行状态
```

### 5.4 `storage.backends[]`

backend 条目只通过 ID 引用其他配置对象，不复制它们的内容：

```text
backend.id
backend.contract
backend.resolver -> resolver.id
backend.datapath -> datapath.id
backend.resource -> resource.id
backend.config   -> contract/backend 共享语义
```

backend config 只容纳 resolver 和 DataPath 必须共同遵守的语义，例如 striped 数据
布局。不能把它作为无法分类字段的兜底容器。

## 6. ID、scheme、DataPath key 与 contract

以下身份不能混用：

| 身份 | 示例 | 用途 |
| --- | --- | --- |
| resolver config ID | `file-resolver-0` | config 引用、实例复用、诊断；只在 loader/backend 装配层有效 |
| resolver implementation type | `local-file` | 选择 resolver factory 和实现参数校验；不参与 URI 路由 |
| URI scheme | `file` | `StorageRuntime` 查找 resolver；同一 Runtime 内必须唯一 |
| resolved resolver type ID | `ext4-extent-resolver-v1` | 写入 `ResolvedTarget` 的来源元数据，用于 contract/诊断校验；不是 resolver registry key |
| backend ID | `model-storage` | 标识一次逻辑 backend 装配 |
| logical contract | `ext4-local-nvme` | loader 选择兼容性规则和 backend factory |
| payload type ID | `ext4-local-nvme-payload-v1` | DataPath 校验 resolver payload ABI |
| DataPath key | `local-nvme-ext4` | `ResolvedTarget` 到 DataPath 的 Runtime 路由 |

这里的 resolver 身份必须按层次区分：

```text
resolver config ID
  -> loader 找到一个具体 resolver 配置实例
resolver implementation type
  -> loader 选择如何构造该实例
URI scheme
  -> StorageRuntime 在 open(uri) 时找到该实例
resolved resolver type ID
  -> resolver 写入 ResolvedTarget，标识实际产生 payload 的实现契约
```

例如，`file-resolver-0` 可以是 config ID，`local-file` 是实现类型，`file` 是 URI
scheme，而 `ext4-extent-resolver-v1` 是 resolver 生成 `ResolvedTarget` 时记录的类型
元数据。四者不能互换：修改 config ID 不应改变 URI 路由；修改 scheme 会改变应用 URI
和 Runtime registry；修改 implementation/type ID 则属于 factory 或 contract 兼容性
变化。

DataPath key 由 payload contract 定义，不应直接使用 `datapath.id`，也不应允许用户在
YAML 中任意改写。否则 config 对象命名会进入 resolver/DataPath 的协议兼容面。

同理，config 只指定逻辑 contract，具体 payload type ID 和 API version 由编译进
二进制的 binding package 决定。loader 必须拒绝二进制不支持的 contract。

## 7. Contract 兼容矩阵

当前三组实现的目标校验矩阵如下：

| Contract | Resolver type | DataPath type | Resource type | Resource cardinality | Runtime key |
| --- | --- | --- | --- | --- | --- |
| `ext4-local-nvme` | `local-file` | `local-nvme` | `nvme` | 1 slice | `local-nvme-ext4` |
| `striped-local-nvme` | `striped-file` | `striped-local-nvme` | `nvme` | 2 个或更多有序 slices | `striped-local-nvme` |
| `memfs` | `memfs` | `memfs` | `memory` | 1 个逻辑 backing | `memfs` |

Memfs 当前是 sample backend，不要求第一阶段产品 loader 实现其 config factory；但
schema 和验证模型不应假定 resource 必然是 NVMe。

backend 不能任意组合三个 ID。例如 `memfs` resolver 与 `local-nvme` DataPath 即使
都存在于配置中，也必须因为 contract 不兼容而在任何 resource acquisition 之前
失败。

resource allocation 返回的 slice 数量也不能反向决定 backend 类型。loader 必须先
根据 backend 的显式 contract 选择 factory，再校验 ResourceInstance cardinality。
例如，`ext4-local-nvme` 收到多个 slices 必须失败，不能静默改成 striped backend。

## 8. Loader 的解析与实例化流程

loader 分为静态解析、资源解析和实例装配三个阶段。

### 8.1 静态解析

```text
读取 YAML
  -> 建立 resource/resolver/datapath/backend ID 表
  -> 检查 ID 唯一
  -> 解析 backend 引用
  -> 检查 contract/type 兼容矩阵
  -> 检查当前产品约束
```

该阶段不得连接 daemon、打开设备或初始化 DataPath。

### 8.2 资源解析

对于被 backend 引用的每个 ResourceSpec：

```text
ResourceFactory 根据 resource type 创建具体 Resource 实现
  -> 具体 factory/构造函数接收对应的 typed ResourceSpec
       NvmeResourceFactory: 创建或注入 NvmeResourceClient
       MemfsResourceFactory: 不创建 client
  -> Resource::initialize()
       NvmeResource: 使用 NvmeResourceClient 查询 snapshot 并请求 allocation
       MemfsResource: 创建进程内 backing，不需要 provider client
  -> 校验返回的 resource metadata
  -> 初始化成功的具体 Resource 对象成为 ResourceInstance
  -> TuttiRuntime.resources_[resource ID] = unique_ptr<Resource>
  -> TuttiRuntime.resource_initialization_order.push_back(resource ID)
```

同一个 resource ID 在一次 `TuttiRuntime` 创建中只能解析一次。后续 backend 引用必须
指向 `TuttiRuntime` 中的同一个 ResourceInstance，不能静默重复申请。

`NvmeResource` 必须在 Acquire 成功后立即接管 `NvmeResourceClient` 和 allocation。即使
后续 metadata 校验或 backend factory 失败，它也必须能够执行幂等 Release，避免存在
一段 allocation 已经成功但没有一等所有者的异常窗口。该要求属于 NVMe 具体实现；不使用
外部 provider 的 `MemfsResource` 只需接管自己创建的 backing。

### 8.3 Backend 实例装配

backend factory 根据 contract 和实现类型，把 ResourceInstance 转换为两个只读视图：

```text
                    concrete Resource instance
                         /              \
              ResolverResourceView   DataPathResourceView
                       |                    |
                       v                    v
                ResolverInstance      DataPathInstance
                         \              /
                          payload contract
```

这里的转换由具体 resource 类型对应的 backend factory 完成。公共 `Resource` 只提供
能力和诊断信息；factory 先校验 `Resource::capabilities()`，再按 contract 选择类型化
路径。例如 NVMe factory 要求 registry 中的对象实际是 `NvmeResource`，并调用它提供的
只读 `resolver_view()` 和 `datapath_view()` 构造输入；这些方法是 NVMe-specific API，
不是 `Resource` 基类的虚函数。这样 allocation、路径和 provider client 仍封装在
`NvmeResource` 内部，factory 只能获得已经授权的只读视图。

Local NVMe 的视图至少包含：

| ResourceInstance 字段 | Resolver view | DataPath view |
| --- | --- | --- |
| PCI BDF | 是 | 诊断或实现私有校验 |
| block path | 是 | 否 |
| chrdev path、BAR size | 否 | 是 |
| namespace ID、logical block size | 是 | 是 |
| hardware max data size | 否 | 是 |
| granted queues、accel ID | 否 | 是 |
| backing/view path | 路径约束所需 | 否 |

Striped NVMe 不能把 ResourceInstance 压缩成一个设备视图。它仍然对应一次逻辑
allocation，但视图内部必须保留有序的 shard 列表。顶层结构仍然是：

```text
one ResourceInstance/allocation
  -> N ordered shard ResourceViews
  -> N LocalFileResolver instances inside one StripedResolver
  -> one StripedDataPath containing N device descriptors
```

Striped NVMe 的资源视图至少包含：

| ResourceInstance 内容 | Striped resolver view | Striped DataPath view |
| --- | --- | --- |
| `allocation_id` | 不直接消费；由对应 ResourceInstance 持有 | 不直接消费；由对应 ResourceInstance 持有和释放 |
| slice 顺序与 `device_id` | 定义 shard index；必须保持 URI 解析、payload 和 stripe mapping 的顺序一致 | 定义 device descriptor 和 fused submit 的设备顺序 |
| 每个 slice 的 `pci_bdf` | 创建对应的 `LocalFileResolver` 并校验 backing device identity | 可作为 device/controller 诊断事实，不作为 I/O 路由 key |
| 每个 slice 的 `block_path` | 创建对应的 backing device view，执行 stat/FIEMAP 和 extent 校验 | 不需要直接消费 |
| 每个 slice 的 `chrdev_path`、`bar0_size` | 不需要直接消费 | 创建对应 controller client 和 queue group |
| 每个 slice 的 `namespace_id`、`logical_block_size` | 生成每个 shard 的 ext4-local payload | 填充每个 device descriptor，并校验所有 shard 一致 |
| 每个 slice 的 `accel_id`、`granted_queues` | 不直接消费；backend factory 在构造前校验归属 | 绑定对应 accelerator 并创建实际 queue group |
| 每个 slice 的 `max_data_size` | 不需要直接消费 | 计算整个 striped DataPath 的有效 MDTS，通常取所有 shard 的最小值 |
| `backing_mount_path`、`view_path` | 约束每个 shard 的文件可见性和路径解析 | 不需要直接消费 |

因此，Striped resolver view 和 DataPath view 都是“按 shard 索引的视图”，但它们的
索引来源相同、消费内容不同：

```text
shard i
  ResourceSlice[i]
    -> ResolverResourceView[i]
    -> LocalFileResolver[i]
    -> ext4-local-nvme shard payload[i]

ResourceSlice[i]
  -> DataPathResourceView[i]
  -> StripedDataPath device descriptor[i]
```

外层 `StripedResolver` 将 N 个 shard resolver 的结果封装成一个
`striped-local-nvme` payload；外层 `StripedDataPath` 再按同一个 shard 顺序消费这些
payload。`stripe_unit` 和布局公式属于 backend/contract 共享语义，shard rotation
属于单个 resolved target 的 payload 语义；两者都不属于某一个物理 slice。

Striped backend 装配时至少必须校验：

- allocation 返回的 slice 数量满足 contract，且顺序与请求一致；
- resolver shard 数、DataPath device descriptor 数和 slice 数完全一致；
- 每个 slice 的 `accel_id` 与 Runtime 一致；
- 所有 slice 的 logical block size 满足 contract 的一致性要求；
- resolver 生成的 shard payload 顺序与 DataPath descriptor 顺序一致；
- 任一 slice 的资源视图、路径校验或 controller attach 失败时，整个 backend 创建失败；
- 多个 shard 仍然只生成一个顶层 resolver scheme 和一个 DataPath key，不展开成多个
  Runtime 顶层组件。

这些 view 是 backend-private 的构造输入，不进入公共 resolver/DataPath SPI。

### 8.4 展开到 `TuttiRuntime` 与 `StorageRuntime`

完成 ResourceInstance 和 backend 装配后，loader 同时更新 `TuttiRuntime` 的资源/组件
所有权 registry，并生成现有 `RuntimeComponents`：

```text
TuttiRuntime.resources_[resource.id] = unique_ptr<Resource>
TuttiRuntime.resolvers_[resolver.id] = owned Resolver
TuttiRuntime.data_paths_[datapath.id] = owned DataPath
TuttiRuntime.backends_[backend.id] = BackendInstance{resource, resolver, datapath, contract}

RuntimeComponents.resolvers += {resolver.scheme, resolver pointer}
RuntimeComponents.data_paths += {contract.DataPathKey, DataPath pointer, config}
```

#### 8.4.1 Resource 的展开路径

storage resource 会作为一个独立的 `resource_id -> unique_ptr<Resource>` registry 注册进
`TuttiRuntime`，但不会注册进 `StorageRuntime`。它在 backend factory 阶段被投影到两个
组件，并由对应的具体 Resource 实例持有自身资源状态：

```text
ResourceSpec(resource ID)
  -> concrete Resource instance
       ├─ ResolverResourceView
       │    -> Resolver 私有状态
       │    └─ Resolver 生成 ResolvedTarget payload/target lease
       ├─ DataPathResourceView
       │    -> DataPath 私有状态
       │    └─ DataPath 创建 controller/queue/workspace
       └─ Resource 实例继续由 TuttiRuntime.resources_[resource ID] 持有
```

因此，resource 的“展开”是实例构造和依赖注入，不是 Runtime 路由注册：

- resolver 只保留自己解析所需的 resource-derived host-side 状态；
- DataPath 只保留自己执行 I/O 所需的 device/transport 状态；
- `NvmeResource` 保存 `NvmeResourceClient`、allocation、`allocation_id` 和 release
  状态；`MemfsResource` 只保存进程内 backing，不需要 client；
- `TuttiRuntime.resources_` 按 resource ID 提供资源的生命周期、诊断和 backend 引用；
- `TargetHandle` 不携带 resource ID，`TargetEntry` 也不建立 resource registry；打开
  target 后只保存 `ResolvedTarget`、选中的 `DataPath*` 和 DataPath 私有 target。

当前 `StorageRuntime` 中的 `resources_` / `RuntimeComponents::resources` 是公共
DataPath SPI 的通用 `ResourceProvider` 指针，不是 storage resource allocation，也
不应被用来传递 NVMe slice、路径、allocation ID 或 lease。storage resource 已经在
loader/backend factory 阶段完成投影后，Runtime 只接收 resolver 和 DataPath 的借用指针。

#### 8.4.2 Runtime 路由

`StorageRuntime` 不按 backend ID 或 resource ID 路由。运行时路径保持：

```text
URI scheme
  -> Resolver
  -> ResolvedTarget.recommended_data_path_key
  -> DataPath
```

#### 8.4.3 `TuttiRuntime` backend manifest

为了诊断和生命周期管理，`TuttiRuntime` 应额外保存只读的 backend 装配 manifest：

```text
backend ID -> resolver ID + datapath ID + resource ID + contract
```

该 manifest 属于 `TuttiRuntime` 管理层，不参与 I/O 热路径。

## 9. 所有权与关闭顺序

目标 `TuttiRuntime` 的逻辑结构为：

```text
TuttiRuntime
  -> resource ID -> unique_ptr<Resource>
       -> NvmeResource
            -> owns NvmeResourceClient
            -> owns RuntimeNvmeAllocation / lease
       -> MemfsResource
            -> owns in-process backing; no client
  -> resolver ID -> owned Resolver
  -> datapath ID -> owned DataPath
  -> backend ID -> BackendInstance / read-only manifest
  -> StorageRuntime
```

`StorageRuntime` 继续只借用 resolver 和 DataPath 指针。`TuttiRuntime` 必须保证这些
对象和 ResourceInstance 存活到 Runtime shutdown 完成。

目标实现必须为 `TuttiRuntime` 使用独立文件：

```text
tutti/config/tutti_runtime.h
  -> 声明 TuttiRuntime、只读 manifest/访问接口和 ownership registry

tutti/config/tutti_runtime.cpp
  -> 实现析构、shutdown、状态转换、逆序资源释放和失败清理

tutti/config/tutti_config.h/.cpp
  -> 只保留 config model、loader options、解析/校验和实例装配入口
```

`tutti_config.cpp` 可以创建并填充 `TuttiRuntime`，但不得继续承载它的析构、shutdown 或
registry 管理实现。该文件拆分首先用于建立 loader 与 Runtime 生命周期的模块边界，不
要求把 `TuttiRuntime` 的内部 registry 暴露为公共 API。

### 9.1 `Resource` 继承接口与具体实现

Resource 使用与 DataPath 相同的“公共基类 + 具体实现”方式。`ResourceInstance` 不是另一层
通用 wrapper，而是对一个已经实例化的具体 `Resource` 对象的语义称呼。最小接口只包含
能力、生命周期和只读诊断：

```cpp
enum class ResourceState {
    CREATED,
    INITIALIZED,
    SHUTTING_DOWN,
    STOPPED,
    FAILED,
};

enum class NvmeLeaseState {
    NONE,
    ACQUIRED,
    RELEASING,
    RELEASED,
};

class Resource {
public:
    virtual ~Resource() = default;

    virtual const ResourceCapabilities& capabilities() const = 0;
    virtual Status initialize() = 0;
    virtual Status shutdown() = 0;
    virtual ResourceInfo info() const = 0;
};

class NvmeResource final : public Resource {
public:
    const ResourceCapabilities& capabilities() const override;
    Status initialize() override;
    Status shutdown() override;
    ResourceInfo info() const override;

    // NVMe-specific, read-only construction views; not part of Resource.
    NvmeResolverResourceView resolver_view() const;
    NvmeDataPathResourceView datapath_view() const;

private:
    std::unique_ptr<NvmeResourceClient> client_;
    RuntimeNvmeAllocation allocation_;
    ResourceState state_ = ResourceState::CREATED;
    NvmeLeaseState lease_state_ = NvmeLeaseState::NONE;
};
```

`ResourceCapabilities` 描述实现类型及可供 backend factory 校验的稳定能力；
`ResourceInfo` 包含 config ID、类型、通用生命周期状态等只读诊断信息，但不暴露可修改
的 allocation payload。`initialize()` 成功后状态变为 `INITIALIZED`；`shutdown()`
幂等释放资源并进入 `STOPPED`。backend 是否引用该对象是 `TuttiRuntime` 的装配关系，
不应通过公共 Resource 状态中的 `BOUND` 表达。析构可以进行 best-effort shutdown，但
需要向调用者报告错误的正常关闭必须显式调用 `shutdown()`。

`NvmeLeaseState` 是 `NvmeResource` 的实现私有状态，用于保证 Acquire/Release 至多一次；
它不是所有 Resource 都必须实现的公共状态机。`MemfsResource` 可以使用与 backing 创建、
销毁相符的内部状态，无需模拟 NVMe lease。

公共接口故意不包含 `client()`、`acquire_nvme_slices()`、`release(allocation_id)` 等
方法，也不定义统一的 `ResourceClient` 基类。这些操作只对 NVMe resource provider 有
意义，应由 `NvmeResourceClient` 在 `NvmeResource` 内部完成。如果需要 gRPC/测试替换，
可以在 NVMe resource 模块内部保留 `NvmeResourceClient` 接口及其实现；这不构成所有
Resource 的公共 client 抽象。

第一阶段 NVMe 实现的逻辑所有权为：

```text
NvmeResource : Resource
  -> resource ID/type/state
  -> unique NvmeResourceClient
  -> RuntimeNvmeAllocation
       -> allocation_id
       -> ordered slices
  -> private NvmeLeaseState / idempotent release guard
```

对 NVMe 而言，`NvmeResourceClient`、allocation 和 release channel 必须属于同一个
`NvmeResource`，不能分别保存在 `TuttiRuntime` 的平行字段中。其他 Resource 子类只
保存其语义所需的状态；例如 `MemfsResource` 只拥有 backing memory，不需要伪造 client
或 allocation ID。

如果未来多个 `NvmeResource` 需要复用同一 endpoint 的连接，可以在 NVMe 实现内部引入
共享 `NvmeProviderSession`，由 `NvmeResourceClient` 或 NVMe 专用 factory 管理。不能为
了这一优化把 client/session 提升为公共 `Resource` 抽象，也不能改变“每个
`NvmeResource` 独立拥有并释放自己的 allocation”这一语义。

### 9.2 `TuttiRuntime` registry 与访问约束

`TuttiRuntime` 应按 config ID 拥有资源和组件，而不是继续暴露可任意修改的 public
vectors/fields。backend 引用解析后保存实际对象引用：

```text
resources_[resource ID] -> unique_ptr<Resource>
resource_initialization_order -> resource IDs in successful initialization order
resolvers_[resolver ID] -> owned Resolver
data_paths_[datapath ID] -> owned DataPath
backends_[backend ID]    -> {contract, resource*, resolver*, datapath*}
```

对外只暴露只读的 `ResourceInfo` 和 backend manifest。调用方不能修改 allocation
slices、替换 `NvmeResourceClient`，或者在 backend/DataPath 仍存活时单独 shutdown
resource。

### 9.3 关闭与失败回滚

正常关闭顺序：

```text
阻止新 I/O 并 drain
  -> StorageRuntime::shutdown()
  -> 销毁 StorageRuntime
  -> 清除 BackendInstance 中对 resource/resolver/DataPath 的非 owning 引用
  -> 销毁 resolver/DataPath instances
  -> 按 initialization 顺序逆序调用 Resource::shutdown()
  -> 具体 Resource 销毁实现私有状态
       NvmeResource: NvmeResourceClient 和 allocation metadata
       MemfsResource: in-process backing
  -> TuttiRuntime 进入 STOPPED
```

任一创建步骤失败时，按已完成步骤的逆序回滚。Acquire 成功后的 allocation metadata
校验、factory、DataPath initialize 或 Runtime create 失败，都必须 Release 对应
allocation，且每个 allocation 只能 Release 一次。对于非 NVMe Resource，回滚其具体
实现已经创建的状态，不套用 NVMe client/allocation 语义。

## 10. 共享与基数

第一阶段产品约束：

- 一个 `TuttiRuntime` 拥有一个 `StorageRuntime`，并只启用一个 backend；
- 一个 backend 引用一个 resolver、一个 DataPath 和一个逻辑 resource；
- local backend 的 resource 必须解析为一个 slice；
- striped backend 的 resource 必须以一次原子 allocation 解析为两个或更多有序
  slices；
- 所有声明必须从唯一启用的 backend 可达，未引用配置视为错误；
- 不允许两个独立 DataPath 实例消费同一个 ResourceInstance；
- backend/DataPath 仍引用 resource 时，单独调用 `Resource::shutdown()` 必须返回 `BUSY`。

最后一条不是永久架构限制。它避免两个 DataPath 各自使用完整 `granted_queues`，造成
queue 配额超用。未来若需要共享，应先引入 ResourceInstance 子 lease 或明确的 queue
配额切分。

后续阶段可以支持：

- 不同 scheme 的多个 resolver 共享同一个兼容 DataPath 和 resource；
- 一个 Runtime 装配多个 backend；
- 多个 backend 引用同一个 resolver/DataPath 实例；
- resource provider 支持 NVMe 之外的资源类型。

即使开放这些能力，仍必须满足当前 `StorageRuntime` 的约束：启用的 resolver scheme
唯一，注册的 DataPath key 唯一。同一 scheme 多 resolver 需要 composite resolver
或新的核心路由语义，不能只靠 config 表达解决。

## 11. 校验规则

loader 必须 fail-closed，并按以下顺序校验。

### 11.1 结构与引用

- 四组 ID 在各自命名空间内非空且唯一；
- backend 引用的 resolver、DataPath 和 resource ID 必须存在；
- 当前阶段 `backends.size() == 1`；
- accelerator Runtime 必须声明 backend；`runtime.accel_id == -1` 的 HOST Runtime
  可以不声明 storage 对象；
- 当前阶段不允许未引用声明；
- 新 schema 与 legacy schema 不得在同一文件中混用。

### 11.2 类型与 contract

- contract 必须由当前二进制支持；
- resolver、DataPath 和 resource type 必须匹配 contract 矩阵；
- resolver scheme 必须符合实现能力，且所有启用 scheme 唯一；
- contract 导出的 DataPath key 在 Runtime 内唯一；
- backend 共享配置满足 contract 约束，例如 stripe unit 非零且对齐。

### 11.3 ResourceSpec

- `allowed` 要求空 `device_ids`；
- `explicit` 要求恰好一个 device ID；
- `striped` 要求至少两个无重复 device ID；
- requested queues 非负；
- provider 类型和 endpoint 有效；
- resource selection 与 backend contract 的 cardinality 一致。

### 11.4 ResourceInstance

- `Resource::initialize()` 成功后通用生命周期状态为 `INITIALIZED`；
- capabilities、实现类型和 backend contract 一致；
- `ResourceInfo` 中的 ID、类型和状态与 registry entry 一致；
- 不要求不依赖外部 provider 的 Resource 子类持有 client 或 allocation ID。

对 `NvmeResource` 额外校验：

- allocation ID 非空；
- `NvmeResourceClient` 非空，并且能够释放该 allocation；
- slice 数量和顺序符合请求；
- slice 的 `accel_id` 与 Runtime 一致；
- ACL 包含 Runtime accelerator；
- 必需路径和 namespace metadata 完整；
- striped slices 的 block size 和 contract 要求一致；
- 实际 granted queues 不超过 lease，且足以创建 DataPath。

所有结构、引用和 contract 错误必须在 Acquire 之前发现。

## 12. Legacy 配置迁移

现有字段到目标 schema 的映射如下：

| 现有字段 | 目标位置 |
| --- | --- |
| `accelerator.profile` | 保持不变 |
| `runtime.accel_id` | 保持不变 |
| `nvme_service.endpoint` | `storage.resources[].provider.endpoint` |
| `nvme.selection` | `storage.resources[].allocation.selection` |
| `nvme.device_ids` | `storage.resources[].allocation.device_ids` |
| `nvme.queues_per_controller` | `storage.resources[].allocation.queues_per_controller` |
| `local_nvme.handle_cache_capacity`、batch/cache 等调优 | `storage.datapaths[].config` |
| `local_nvme.num_user_queues` | 移除；统一使用 resource allocation 的队列申请和实际 grant |
| `nvme.stripe_unit` / `storage.default_stripe_unit` | striped `storage.backends[].config.stripe_unit` |
| `storage.backend` | 展开为 resolver、DataPath、resource 和 backend 四个声明 |
| `local_nvme_config` | 从应用 config 移除，部署事实由 daemon 管理 |

当前 loader 中的以下字段不再属于 `TuttiRuntime` 本身，而是迁移到唯一的
`NvmeResource`。这是 NVMe 专用迁移，不表示公共 `Resource` 拥有 client 或 allocation
字段：

| 当前字段 | 目标字段 |
| --- | --- |
| `TuttiRuntime::resource_client` | `NvmeResource::client_`（类型降级为 `NvmeResourceClient`） |
| `TuttiRuntime::allocation_id` | `NvmeResource::allocation_.allocation_id` |
| `TuttiRuntime::allocation_slices` | `NvmeResource::allocation_.slices` |
| `TuttiRuntime::allocation_released_` | `NvmeResource::lease_state_` / release guard |

过渡实现可以把纯 legacy 文件转换成同一份 canonical parsed model，但必须满足：

- canonical schema 是文档和示例中的唯一推荐形式；
- legacy 和 canonical 字段混用时报错，不做优先级猜测；
- legacy 转换发生在静态解析阶段，后续资源申请和装配只消费 canonical model；
- 兼容入口有明确废弃计划，不能长期维护两套 loader 逻辑。

## 13. 实现阶段建议

本文只定义设计，后续实现建议分为六步：

1. 新增纯解析数据结构和 schema 校验，保持 loader 行为不变。
2. 新增独立的 `tutti/config/tutti_runtime.h` 和 `tutti_runtime.cpp`，将现有
   `TuttiRuntime` 声明、析构和 `shutdown()` 从 `tutti_config.*` 迁出；本步骤只调整文件
   所有权，不改变 Runtime 行为。
3. 新增 `Resource` 基类和 `NvmeResource : public Resource`，由 resource factory 按
   resource type 创建具体实现；将当前 `RuntimeResourceClient` 重命名并降级为
   `NvmeResourceClient`，只由 `NvmeResource` 持有。
4. 引入 ResourceSpec 到 ResourceInstance 的解析层，把 legacy 字段转换移到入口。
5. 引入 backend factory 和只读 resource views，替换当前散列的构造函数参数装配。
6. 在 `TuttiRuntime` 中实现 `unique_ptr<Resource>` registry、backend/resource manifest，
   并补齐失败回滚和共享约束测试。

每一步都应保持以下 contract tests：

- Local 与 Striped loader 仍只向当前产品 Runtime 注入一组顶层 resolver/DataPath；
- `StorageRuntime` 原有多 scheme、多 DataPath key 能力不收缩；
- duplicate scheme/key 继续 fail-closed；
- Acquire 后任一失败路径只 Release 一次；
- resolver 和 DataPath 从同一个 ResourceInstance 获得一致的 namespace identity；
- DataPath shutdown 和 resource Release 顺序符合本文定义。

## 14. 非目标

- 不让 `StorageRuntime` 直接解析 YAML 或连接 resource daemon；
- 不让 resolver 或 DataPath 按 config ID 自行查找其他组件；
- 不把 backend ID 变成 I/O 热路径路由 key；
- 不把 NVMe/protobuf 类型加入公共 resolver/DataPath SPI；
- 不允许 DataPath 自行选择未由 ResourceSpec 和 lease 授权的设备；
- 不在本设计中增加同 scheme 多 resolver 的动态选择；
- 不把 daemon 部署事实重新复制进应用 config。

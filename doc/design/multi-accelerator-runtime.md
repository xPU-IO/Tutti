# Tutti 多加速器、多盘与 NVMe daemon 设计

> **状态**：当前实现设计收束
>
> 本文基于当前生产代码描述一台主机上多个 accelerator、多个 NVMe、多个
> `TuttiRuntime` 的资源分配关系，以及 NVMe daemon 的配置、接口和生命周期。

## 1. 范围与相关文档

本文覆盖：

- 一个进程如何用多个 `TuttiRuntime` 实例服务多个 accelerator；
- daemon 如何拥有多块 NVMe，并通过 ACL、queue reservation 和 lease 将资源授予
  Runtime；
- `allowed`、`explicit` 和 `striped` 三种多盘选择模式；
- daemon 的配置、启动、mount、accelerator view 发布和关闭流程；
- daemon 的 canonical/legacy gRPC 接口，以及每个接口的实际调用组件；
- Runtime 侧 `NvmeResource` 与 daemon 的交界，以及 DataPath attach 的边界。

[`tutti-runtime-assembly.md`](tutti-runtime-assembly.md) 已经描述 YAML parser、spec
验证、Resource/Resolver/DataPath factory 和 `StorageRuntime` 装配。本文不重复这些
细节，只描述 NVMe Resource 以外的主机级控制面，以及它与装配流程相接的位置。

本文不深入：

- Resolver 和 DataPath factory 返回实例后的解析或 I/O 算法；
- `StorageRuntime` 的 target、memory、submit、wait、progress 等接口；
- libnvm queue、DMA mapping 和 command submission 的内部实现；
- 测试专用注入接口或测试样例。

## 2. 当前系统模型

当前实现有四条基础约束：

1. 一个 Tutti 二进制只包含一种编译期 accelerator backend，例如 CUDA、MUSA 或
   MACA；配置不能在运行时切换 vendor。
2. 一个 `TuttiRuntime` 及其内部 `StorageRuntime` 固定绑定一个 `accel_id`。
3. 同时使用多个 accelerator 时，应用创建多个相互独立的 `TuttiRuntime`。每个
   Runtime 有自己的 Resource lease、Resolver、DataPath 和 `StorageRuntime`。
4. 一个 Runtime 可以取得一个 NVMe slice，也可以通过一次 striped allocation 取得
   多个有序 slice。多盘不会改变该 Runtime 只有一个 `accel_id` 的事实。

```text
application process
  |
  +-- TuttiRuntime(accel_id=0)
  |     +-- NvmeResource ---- allocation A ---- NVMe 0 [reserves 16 queues]
  |     +-- Resolver/DataPath/StorageRuntime for accelerator 0
  |
  +-- TuttiRuntime(accel_id=1)
        +-- NvmeResource ---- allocation B ---- NVMe 0 [reserves 16 queues]
        +-- Resolver/DataPath/StorageRuntime for accelerator 1

                         control plane
  NvmeResource --------------------------------------> tutti_daemon
                                                        owns NVMe 0..N

                         data plane
  DataPath ---------------- client libnvm attach ----------------> NVMe
```

daemon 是主机级 owner 和资源 broker。它不创建 accelerator context，不持有 Runtime
的 stream/device pointer，也不代理每次 I/O。`StorageRuntime` 完全不访问 daemon。

Runtime 创建链中的实际调用层次为：

```text
TuttiRuntime
  -> NvmeResource
     -> GrpcNvmeResourceClient
        -> nvmeservice::NvmeServiceClient
           -> gRPC NvmeService
```

## 3. 身份模型

| 身份 | 含义 | 来源 | 使用者 |
| --- | --- | --- | --- |
| `accel_id` | 编译期 backend 中的 accelerator ordinal | daemon 配置和 Runtime spec | Runtime 绑定、NVMe ACL、view 选择 |
| `device_id` | daemon 本地 NVMe resource ID | daemon 配置中的显式字段 | list/acquire、Runtime allocation spec |
| PCI BDF | 物理 controller 身份 | daemon 配置，bring-up 后核对 | owner bring-up、诊断和 grant metadata |
| `allocation_id` | 一次逻辑资源租约 | daemon 随机生成 | heartbeat、release、reaper |
| chrdev/block/view path | 本次 bring-up 和 mount 的路径事实 | libnvm owner 结果和 daemon 发布 | DataPath attach、Resolver 文件视图 |

`accel_id` 和 `device_id` 是不同命名空间，数值相同不表示默认配对关系。例如
accelerator 0 可以申请 `device_id: 1`，只要该 NVMe 的 `allowed_accel_ids` 包含 0。

daemon、Runtime 和 accelerator backend 必须对 `accel_id` ordinal 使用同一部署映射。
当前没有 accelerator UUID 或可见设备重映射协议。legacy RPC 中的 `cuda_device` 和
`allowed_gpus` 实际承载同一种 ordinal，但名称只为线协议兼容而保留。

`device_id` 也不是 YAML 数组下标，不能用它拼接 `/dev/ssnvmeN`、block path 或 view
path。客户端必须使用 allocation 返回的实际路径。

## 4. 多 accelerator Runtime 模型

应用为每个 accelerator 分别调用 `TuttiRuntime::create()`。每份 Runtime YAML 都包含
自己的 `runtime.accel_id`、daemon endpoint 和 NVMe allocation 请求。不同 Runtime
之间没有共享的 Tutti handle、memory registration、stream 或 I/O batch；并行协调由
应用负责。

多个 Runtime 可以使用：

- 不同 accelerator 和不同 NVMe；
- 不同 accelerator 和同一个 NVMe；
- 不同 accelerator，并分别 stripe 到同一组 NVMe。

第二、三种情况不是 controller 的独占分配。daemon 按 allocation 预留 queue budget，
而每个 DataPath 作为 libnvm client 创建自己的 queue group 和 user queues。只要 ACL、
published view 和剩余 queue capacity 均满足，请求就可以共享 controller。

一个 Runtime 当前只绑定一个 accelerator。跨 accelerator submit、一个 batch 横跨多个
Runtime、运行中的 accelerator 迁移，以及 daemon 侧 I/O 调度都不在当前实现中。

## 5. daemon 配置模型

生产入口使用 `nvmeservice::parse_config_file()` 读取 daemon YAML。canonical 配置可参考
`config/local/daemon_2disk.yaml`、`config/local/daemon_4disk.yaml` 和
`config/local_nvme_config.yaml`。

```yaml
grpc:
  endpoint: "127.0.0.1:50051"

accelerators:
  - accel_id: 0
    view_root: "/mnt/snvme/gpu0"
  - accel_id: 1
    view_root: "/mnt/snvme/gpu1"

nvmes:
  - device_id: 0
    pci_addr: "0000:41:00.0"
    backing_mount_path: "/mnt/snvme/nvme1"
    namespace_id: 1
    kernel_ioq_cap: 32
    allowed_accel_ids: [0, 1]
    auto_mount: true

queue_pool:
  default_per_client: 16
  max_per_client: 32

lease:
  heartbeat_interval_sec: 10
  timeout_sec: 30

unmount_retry:
  interval_ms: 1000
  max: 30
```

### 5.1 字段职责

| 配置 | 含义 |
| --- | --- |
| `grpc.endpoint` | daemon 监听地址，也是 Runtime resource provider 的 endpoint |
| `accelerators[].accel_id` | daemon 可识别的 accelerator ordinal |
| `accelerators[].view_root` | 该 accelerator 的文件系统 view 根目录 |
| `nvmes[].device_id` | 显式、唯一的 daemon 本地 NVMe ID |
| `nvmes[].pci_addr` | owner bring-up 使用的 canonical PCI BDF |
| `nvmes[].backing_mount_path` | owner 返回 block device 的 mount 目标 |
| `nvmes[].namespace_id` | grant 暴露的 namespace ID |
| `nvmes[].kernel_ioq_cap` | bind 前传给内核的 I/O queue-pair cap hint；0 表示不设置 |
| `nvmes[].allowed_accel_ids` | acquire ACL，同时决定为哪些 accelerator 发布 view |
| `nvmes[].auto_mount` | daemon 是否负责 mount，并在正常关闭时负责 unmount |
| `queue_pool.*` | 单个 allocation、单个 controller 的默认值和上限 |
| `lease.*` | client 心跳周期和 daemon 回收超时 |
| `unmount_retry.*` | daemon-owned mount 遇到 `EBUSY` 时的重试策略 |

省略或显式给出空 `allowed_accel_ids` 时，parser 会将其展开为全部已配置的
`accel_id`，并对 ACL 排序。配置验证还保证 ID、BDF 和路径唯一，ACL 只引用已声明
accelerator，queue policy 合法，且 heartbeat interval 小于 timeout。

parser 暂时仍接受完整的 legacy schema：`gpus[].id/mount_path`、
`nvmes[].mount_path/allowed_gpus`。legacy NVMe 的 `device_id` 由数组下标补出并产生
迁移诊断。canonical 和 legacy 字段不能在同一份配置中混用；新部署应只使用
canonical schema。

## 6. daemon 架构和启动流程

生产 daemon 入口为
`tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon.cpp`。其启动顺序如下：

```text
parse + validate daemon YAML
  -> construct ServiceState
     -> owner bring-up every configured NVMe
     -> validate returned device metadata
     -> validate uniform namespace logical block size
  -> mount actual block_path when auto_mount=true
  -> publish accelerator-specific filesystem views
  -> start lease reaper
  -> bind and start gRPC server
```

### 6.1 NVMe owner bring-up

`ServiceState` 对每个 NVMe 调用 owner bring-up。配置只提供 BDF、namespace 和策略；
以下事实来自 libnvm/controller，而不是配置推导：

- chrdev minor、chrdev path、disk name 和 block path；
- BAR0 size、DSTRD、page size 和 queue depth；
- user QID range、kernel I/O queue 数和 `max_queues_per_group`；
- MDTS 对应的最大单次传输大小；
- namespace logical block size。

daemon 核对 configured/observed BDF、chrdev 和 block device 关联。单盘 bring-up 失败时，
该 resource 保留在 snapshot 中但不可分配，并携带 `diagnostic`。多个 ready namespace
的 logical block size 不一致时，当前实现将相关资源置为不可用；这保证 striped factory
不会收到不同的 block size。

controller 的 queue capacity 来自内核报告的 user QID 区间，不来自 YAML 中的
`queue_pool`。`queue_pool` 只定义每个 client allocation 的准入策略。

### 6.2 mount 和 accelerator view

owner bring-up 完成后，daemon 才知道真实 `block_path`。`auto_mount: true` 时，
`MountManager` 将该路径以 ext4 mount 到 `backing_mount_path`。目标如果已经被 operator
mount，daemon 使用它但不取得 unmount 所有权；`auto_mount: false` 也要求 operator
预先准备好 mount，才能发布 view。

每个 `(device_id, allowed accel_id)` 的 view 结构为：

```text
<backing_mount_path>/ACCEL<accel_id>
        ^
        | symlink target
<accelerator.view_root>/<owner-returned chrdev basename>
```

目录和 symlink 在 mount 后创建，路径记录在 resource snapshot 中。申请结果直接返回
该 `view_path`。客户端不得根据 `device_id` 猜测 symlink basename。

`ListNvmeResources.available` 当前表示：owner control ready、至少有一个 view 已发布，
且 controller 仍有 queue capacity。它不证明特定 `accel_id` 的 view 一定存在，也不
证明真实 accelerator I/O 已经成功；`AcquireNvmeSlices` 会再次检查请求 accelerator
的 ACL 和 view。

## 7. daemon 接口及调用方

gRPC contract 位于 `nvmeservice.proto`。接口分为 Tutti Runtime 使用的 canonical API
和只为旧 client 保留的 legacy API。

### 7.1 Canonical RPC

| RPC | 请求/返回的核心数据 | 当前调用组件 | 在 Runtime 创建链中的用途 |
| --- | --- | --- | --- |
| `ListAccelerators` | 返回 `accel_id`、`view_root` | `NvmeResource::initialize()` -> `GrpcNvmeResourceClient` -> `NvmeServiceClient`；独立的 `nvmeservice_client` CLI | 确认 Runtime 的 `accel_id` 被 daemon 声明，且 view root 非空 |
| `ListNvmeResources` | 返回 ID、ACL、owner metadata、queue snapshot、`available` 和诊断 | `NvmeResource::initialize()`，经同一 adapter/client；独立的 `nvmeservice_client` CLI | 在 acquire 前检查目标 resource 是否存在、ACL 是否允许及 snapshot 是否可用 |
| `AcquireNvmeSlices` | 输入 accelerator、selection、device IDs、每盘 queue 请求和 client PID；返回一个 allocation 及有序 slices | `NvmeResource::initialize()`，经同一 adapter/client；`nvmeservice_client` CLI | 原子选择设备、预留 queue budget，并取得 factory 所需的运行期路径和 controller metadata |
| `Heartbeat` | allocation ID 的双向 stream；可返回 lease notice | `NvmeServiceClient` 内部后台线程 | 刷新 canonical allocation 和 legacy session 共用的 lease |
| `Release` | 输入 allocation ID；返回 success/`already_released`/error | canonical `NvmeServiceClient::Allocation` 析构；上层由 `NvmeResource::shutdown()` 触发 | 一次释放 allocation 中全部 controller reservation |

`TuttiRuntime` 不直接构造 RPC，也不依赖 protobuf 类型。`NvmeResourceClient` 是
Resource 内部的抽象边界，`GrpcNvmeResourceClient` 负责将 nvmeservice client 结构转换
为 `RuntimeNvmeAllocation`。

服务端目前将 selection、ACL 或容量等业务拒绝写入 response 的 `error_message`，同时
返回 gRPC OK；transport failure 才表现为非 OK gRPC status。Runtime adapter 当前会将
多数 acquire 失败收束为较粗粒度的 `NOT_READY`，所以 daemon 端原始错误仍是定位资源
拒绝的重要信息。

### 7.2 Legacy compatibility RPC

| RPC | 调用组件 | 当前行为 |
| --- | --- | --- |
| `ListDevices` | `NvmeServiceClient::list_devices()` 及 legacy/example client | 将 canonical resource snapshot 投影为 `DeviceInfo/AllowedGpu` |
| `Connect` | `NvmeServiceClient::connect()` 及 legacy/example client | 转换为 single-device、explicit canonical acquire，共享同一 ACL 和 reservation ledger |
| `Disconnect` | legacy `NvmeServiceClient::Session` 析构 | 校验 client PID 后调用统一 release 路径 |
| `Heartbeat` | canonical `Allocation` 和 legacy `Session` 共用 | 刷新相同的 allocation record |

`TuttiRuntime` 创建链不使用 `ListDevices`、`Connect` 或 `Disconnect`。新组件不应继续
扩展 legacy 的 `cuda_device`/`allowed_gpus` 命名。

### 7.3 C++ client 的 lease 行为

`NvmeServiceClient::acquire_nvme_slices()` 返回 move-only ownership 语义的
`unique_ptr<Allocation>`。一个 `Allocation` 包含一个 `allocation_id` 和一个或多个
有序 `ClientNvmeSlice`：

- allocation 加入 client 的 live session map 后，client 启动 heartbeat thread；
- heartbeat thread 对全部 live allocation 使用最短 interval；当前实现还把实际周期上限
  固定为 10 秒，所以配置返回更长 interval 时会更频繁地发送；
- `Allocation` 析构时先从 heartbeat map 移除，再发送 `Release`；
- `GrpcNvmeResourceClient` 将 Allocation 保存在以 allocation ID 为 key 的 map 中；
- `NvmeResource::shutdown()` 调用 adapter `release()`，erase 该对象，从而触发 RAII
  `Release`。

因此 canonical lease 的真正 owner 是 Resource 内部 client 对象，不是
`StorageRuntime`。client 必须比其 `Allocation` 存活更久；当前 adapter 的成员销毁顺序
满足这一约束。

## 8. 多盘选择、queue reservation 与共享

### 8.1 Selection 语义

| 模式 | `device_ids` 约束 | daemon 选择规则 | slice 顺序 |
| --- | --- | --- | --- |
| `allowed` | 必须为空 | 按 `device_id` 升序，选择第一个 ACL、view 和 queue capacity 均满足的 resource | 单个 slice |
| `explicit` | 恰好一个 ID | 必须使用该 resource，不自动回退 | 请求中的唯一 ID |
| `striped` | 至少两个互不重复的 ID | 所有指定 resource 都必须满足条件 | 严格保留请求顺序 |

Runtime spec 对 cardinality 做静态检查；daemon 再基于实时 ACL、view 和容量检查动态
可行性。`NvmeResource` 收到 grant 后还会验证 single/striped slice 数量、striped 顺序、
每个 slice 的 `accel_id`、ACL、路径和 controller metadata，失败时立即 release。

### 8.2 Queue grant

每个 selected controller 独立计算 grant：

```text
requested > 0 ? requested : queue_pool.default_per_client
  -> min(queue_pool.max_per_client)
  -> min(controller.max_queues_per_group)
  -> 必须 <= controller_queue_capacity - reserved_queues
```

daemon 可能把请求 clamp 为更小的 grant。之后 DataPath factory 仍会根据自身并发配置
检查 grant 是否足够，例如 submit threads 不能超过已授予 queues。daemon 的 reservation
是准入账本；client 真正创建 queue group/user queues 时，内核仍进行最终约束。

### 8.3 原子性与 controller 共享

设备选择、所有 grant 校验、reservation 增加和 allocation 插入在同一个 state mutex
下完成。striped request 只有在全部 controller 均可满足时才增加账本；任一盘失败不会
留下部分 reservation，插入异常也会回滚此前增加的全部计数。

一个 allocation 保存完整的 `{device_id, queues}` reservation 列表。`Release`、legacy
`Disconnect` 和 lease reaper 最终都调用 `release_locked()`，先校验整份账本，再一次性
退还 allocation 的所有 controller reservation。

同一 controller 可以同时服务多个 Runtime，条件是每个 Runtime 的 `accel_id` 都通过
ACL，并且总 reservation 不超过 controller capacity。daemon 不为 allocation 划分 LBA
范围；不同 Runtime 的文件命名和数据隔离由应用、Resolver 及 backing filesystem 负责。

## 9. Runtime 与 DataPath 的集成边界

NVMe Resource 初始化成功后，把同一个 grant 投影为两个最小视图：

| 消费者 | Resource view 中的主要数据 |
| --- | --- |
| Resolver factory | `device_id`、BDF、block/backing/view path、namespace、logical block size |
| DataPath factory | `device_id`、BDF、`accel_id`、chrdev path、namespace、block size、BAR0、MDTS、granted queues |

local allocation 产生一个 slice，striped allocation 产生多个有序 slice。Resolver 和
DataPath factory 如何将这些视图构造成具体实例，见
[`tutti-runtime-assembly.md`](tutti-runtime-assembly.md)。

`StorageRuntime::create()` 初始化 DataPath 时，local/striped NVMe DataPath 才作为 client
对每个 slice 的真实 chrdev 执行 `nvm_ctrl_attach_client`，并创建自己的 queue group 和
user queues。这里是 daemon control plane 与 Runtime data plane 的分界：

```text
daemon owner bring-up + grant metadata + queue reservation
                              |
                              v
DataPath client attach + own queue group + actual Runtime I/O
```

daemon 不创建或持有这些 client queue group。反过来，DataPath 不负责 ACL、全局分配、
mount 或 lease 回收。

## 10. 生命周期与失败处理

### 10.1 Runtime allocation 生命周期

```text
NvmeResource::initialize()
  -> list + acquire
  -> validate grant
  -> Resolver/DataPath factory
  -> StorageRuntime::create() and DataPath initialize
  ... Runtime active; client heartbeats allocation ...
TuttiRuntime::shutdown()
  -> StorageRuntime shutdown and destruction
  -> Resolver/DataPath destruction
  -> NvmeResource::shutdown()
  -> Allocation destruction -> Release RPC
```

acquire 后任意 grant 校验、factory 或 `StorageRuntime::create()` 失败，TuttiRuntime 的
创建回滚最终都会 shutdown Resource 并 release allocation。正常关闭也先停止数据面，
再释放 daemon lease，避免 queue/metadata 仍被 DataPath 使用时先退还 reservation。

### 10.2 Lease 回收

allocation 可以通过四条路径结束：

1. canonical client 显式 `Release`；
2. legacy client `Disconnect`；
3. 超过 `lease.timeout_sec` 未刷新 heartbeat；
4. reaper 发现 client PID 消失，或 `/proc/<pid>/stat` starttime 改变，说明 PID 已复用。

reaper 以 `max(1s, heartbeat_interval/2)` 为周期检查。四条路径最终使用同一个完整
release helper。`Release` 对已经释放过的 ID 返回 `already_released`，对从未出现过的
ID 返回错误。

heartbeat/lease 只管理 daemon reservation record。client 异常退出后的 fd、queue 和
mapping 回收仍依赖内核的 fd cascade；daemon 不远程销毁另一个进程的数据面对象。

### 10.3 daemon 关闭

第一个 `SIGINT`/`SIGTERM` 触发：

```text
stop accepting/drain gRPC
  -> stop lease reaper
  -> remove accelerator view symlinks and owned ACCEL directories
  -> unmount daemon-owned mounts
  -> ServiceState destruction frees owner controllers
```

unmount 遇到 `EBUSY` 时，`MountManager` 扫描 `/proc` 中的 fd、cwd 和 mappings，输出
holder，并按配置重试。第二个 signal 请求立即退出，busy mount 可能保留给 operator
处理。daemon 不 unmount 启动前已经存在、因而不归它所有的 mount。

当前关闭流程不实现“等待所有 allocation 主动 release”协议。部署方应先停止 Runtime
clients，再停止 daemon；否则 client attach、文件句柄或 mmap 可能使 unmount 进入
`EBUSY` 重试。

## 11. 当前多 accelerator 示例

以下配置来自 `examples/tutti_runtime`，它们说明的是资源关系，不要求 `device_id` 和
`accel_id` 同号。

### 11.1 两个 Runtime 共享一块盘

```text
tutti_multi_accelerator_0.yaml: accel 0 -> explicit device 0 -> 16 queues
tutti_multi_accelerator_1.yaml: accel 1 -> explicit device 0 -> 16 queues
```

两个 Runtime 各有一个 allocation 和一个 accelerator-specific view。若 device 0 的
capacity 至少为 32，且 ACL 包含 0、1，两份 reservation 可以并存。

### 11.2 交叉映射

```text
tutti_cross_accelerator_0_device_1.yaml: accel 0 -> device 1
tutti_cross_accelerator_1_device_0.yaml: accel 1 -> device 0
```

该组合说明 daemon 按 ACL 和显式 ID 分配，不使用 ordinal 相等作为隐式拓扑规则。

### 11.3 两个 Runtime 共享两盘 striped 组合

```text
tutti_multi_accelerator_striped_0.yaml: accel 0 -> devices [0, 1]
tutti_multi_accelerator_striped_1.yaml: accel 1 -> devices [0, 1]
```

每个 Runtime 获得一个逻辑 allocation，其中有两个按 `[0, 1]` 排序的 slice。每份
allocation 在两块盘上各预留 16 queues，因此两块 controller 都需要至少 32 的可用
capacity 才能让两个 Runtime 同时创建成功。

这些例子都由应用创建两个 `TuttiRuntime`，而不是由一个 `StorageRuntime` 管理两个
accelerator。

## 12. 限制与实现漂移

- daemon 使用 insecure gRPC channel，没有认证、授权令牌或 allocation ownership
  credential；当前 trust model 是同机或受控网络中的 cooperative process。
- `accel_id` 是 ordinal，不是稳定硬件身份。daemon、client 和 backend 的设备可见性
  配置必须一致。
- `available` 是控制面 snapshot，不是 accelerator I/O health check；list 与 acquire
  之间状态也可能变化，最终以 acquire 结果为准。
- queue reservation ledger 只代表 daemon 准入；实际 client queue 创建仍可能因内核或
  attach 错误失败，随后由 Runtime 回滚 release。
- canonical C++ client 目前用析构执行 `Release` 并把 RPC 错误写到 stderr；
  `GrpcNvmeResourceClient::release()` 本身不把该 RPC 结果返回给 `NvmeResource`，因此
  Runtime shutdown 的 `Status` 不能证明 daemon 已确认释放。失联 reservation 仍由
  heartbeat timeout/PID reaper 回收。
- released allocation ID 集合用于实现幂等 `Release`，当前进程生命周期内不会淘汰。
- daemon config parser 仍保留 legacy schema，proto 仍暴露 CUDA 命名的 legacy RPC；
  这些都不是新 Runtime 集成入口。
- `nvmeservice.proto` 和生产 `tutti_daemon.cpp` 文件头仍写着 daemon 不维护 shadow
  ledger。这些注释已经过时：当前 `ServiceState` 明确维护 `reserved_queues` 和每个
  allocation 的 reservation 列表，list、acquire、release 和 reaper 都使用该账本。

## 13. 边界结论

当前设计可以收束为以下规则：

1. accelerator vendor 在编译期固定；一个 Runtime 固定一个 `accel_id`，多 accelerator
   由多个 Runtime 实例实现。
2. daemon 独占 owner-side controller 生命周期，并发布实际 device metadata 和文件系统
   view；Runtime 不推导设备路径。
3. daemon 用 ACL、实时 queue reservation 和 lease 管理共享，不参与 Runtime I/O。
4. `allowed` 选择一盘，`explicit` 固定一盘，`striped` 原子取得至少两盘；一个 striped
   allocation 的所有 slice 一起 release。
5. `NvmeResource` 是 Runtime 与 daemon 的唯一集成边界，负责 grant 校验和 lease 所有权。
6. Resolver factory 只消费文件视图，DataPath factory 只消费 attach 所需 metadata；
   `StorageRuntime` 不理解 daemon 或 allocation。
7. 关闭顺序必须先停止 Runtime data plane，再销毁组件，最后 release Resource lease。

## 14. 代码索引

| 主题 | 代码位置 |
| --- | --- |
| gRPC contract | `tutti/device_manager/nvme/nvmeservice/src/nvmeservice.proto` |
| C++ client、Allocation/Session RAII 和 heartbeat | `tutti/device_manager/nvme/nvmeservice/src/nvmeservice_client.h/.cpp` |
| gRPC server adapter | `tutti/device_manager/nvme/nvmeservice/src/nvmeservice_server.h/.cpp` |
| selection、reservation、lease 和 owner state | `tutti/device_manager/nvme/nvmeservice/src/nvmeservice_state.h/.cpp` |
| daemon 配置 parser 和 validator | `tutti/device_manager/nvme/nvmeservice/src/nvmeservice_config.h/.cpp` |
| mount/unmount ownership | `tutti/device_manager/nvme/nvmeservice/src/mount_manager.h/.cpp` |
| 生产 daemon 入口 | `tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon.cpp` |
| Runtime gRPC adapter | `tutti/resource/nvme/nvme_resource_grpc.cpp` |
| Runtime NVMe Resource | `tutti/resource/nvme/nvme_resource.cpp` |
| TuttiRuntime 创建和关闭 | `tutti/tutti_runtime/tutti_runtime_create.cpp`、`tutti/tutti_runtime/tutti_runtime.cpp` |
| Runtime 配置和多 accelerator 示例 | `examples/tutti_runtime/*.yaml`、`examples/tutti_runtime/tutti_runtime_multi_accelerator_example.cpp` |

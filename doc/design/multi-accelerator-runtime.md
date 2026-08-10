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
StorageRuntime 核心      -> 多个不同 scheme 的 Resolver + 多个不同 key 的 DataPath
本阶段产品 loader/factory -> 每个 Runtime 只装配一组顶层 Resolver + DataPath
多个 accelerator device  -> 创建多个 StorageRuntime
一个 daemon              -> 服务整台机器的所有 accelerator/NVMe 组合
```

因此，应用不在运行时切换 CUDA/MUSA/MACA，也不把多个 GPU 展开到同一个
`StorageRuntime`。需要同时使用 GPU 0 和 GPU 1 时，在同一进程中创建 Runtime 0
和 Runtime 1；两个 Runtime 可以并发提交。

产品 loader/factory 通过 daemon 为一个 Runtime bundle 请求 NVMe **切片
（lease）**。切片是一个客户端对 controller/namespace 的访问和队列配额，不是
自动创建的 LBA 分区。一个 bundle 持有一个逻辑 allocation；allocation 可以包含
多个 slice，本阶段 loader 把多个 slice 组成一个 striped DataPath，不为每个 slice
分别注册顶层 resolver 或 DataPath。多个 bundle 可以在 daemon 的队列预算允许时
共享同一 controller。这里限制的是本次产品装配，不是 `StorageRuntime` 公共装配
契约的固有限制。

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
- 保持 `StorageRuntime` 已有的多 scheme resolver、多 key DataPath 路由能力不变；
  本阶段产品 loader 对一个 Runtime 只装配一个顶层 resolver 和 DataPath，多盘
  能力封装在 striped resolver/DataPath 内部。

### 2.2 非目标

- 一个 Runtime 在运行中迁移到另一个 accelerator device。
- 根据 YAML 在运行时切换 accelerator vendor/profile。
- 一个 batch 横跨多个 Runtime 或多个 accelerator device。
- daemon 持有 client 的 SQ/CQ、GPU stream 或 device pointer。
- 第一阶段就实现跨 GPU memory、跨 GPU stream 或自动 P2P 路由。
- 把 `available` 误解为已经完成一次真实 GPU I/O 的证明。
- 扩展 `StorageRuntime` 以支持同一 URI scheme 下的多个 resolver 或动态 resolver
  选择。
- 让本阶段产品 loader 同时装配多个独立的顶层 resolver/DataPath 组合。
- 删除或收缩 `StorageRuntime` 当前已有的多 scheme、多 key 路由能力。

## 3. 术语与身份

必须区分下面各类身份。daemon 的 `NvmeResource.device_id` 保留为 NVMe 资源的
本地编号；真正的加速器设备统一使用短名 `accel_id`，两者不能混用。

| 名称 | 所属 | 含义 | 稳定性 |
| --- | --- | --- | --- |
| `accel_id` | 编译期后端/Runtime | 部署约定并映射到 backend ordinal 的 accelerator 编号 | Runtime bundle/daemon 部署内有效 |
| `device_id` | daemon | daemon 管理的 NVMe controller 编号 | daemon 会话内有效 |
| `pci_bdf`、序列号、namespace | 物理设备 | controller/namespace 身份 | 机器级事实 |
| `allocation_id` | daemon/client 会话 | 一次 NVMe 切片租约 | 租约生命周期内有效 |
| `Runtime` handle ID | 应用进程 | Runtime 内的 target/memory/I/O 句柄 | 进程和 Runtime 生命周期内有效 |

目标 YAML、通用 C++ API 和新 RPC 中，只能使用 `accel_id` 表示 accelerator。
`gpu_id`、`cuda_device`，以及用 `device_id` 表示 accelerator 的字段，均视为当前
实现的 legacy/vendor-specific 命名；只能保留在有期限的兼容入口或 CUDA 私有实现
内部，不能进入新通用契约。`device_id` 在新 daemon RPC 中只表示 NVMe resource。

`allowed_accel_ids` 只有在 daemon 和所有 client 对 `accel_id` 使用同一部署映射时
才有意义。`accel_id` 直接使用编译 backend 提供的 accelerator ordinal；本设计不
引入 UUID/stable identity，也不提供 `CUDA_VISIBLE_DEVICES` 或同类重映射兼容层。
部署侧必须保证 daemon、client 和 Runtime 使用相同的 ordinal 顺序。`pci_bdf` 可以
作为诊断元数据返回，但不参与 Runtime 创建或归属判定。

`/dev/ssnvmeN`、`/dev/snvmeNn1`、NVMe 的 `backing_mount_path` 和按 accelerator
生成的 `view_path` 都是 daemon 或 kernel
返回的**事实**，不是由 `accel_id`、YAML 数组序号或字符串拼接推导的身份。应用
配置可以保存 BDF 等稳定事实，数据路径实际打开的 path 必须来自 daemon 返回的
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
2. 保持 chrdev、bind、`backing_mount_path`、`view_path` 和 controller 的生命周期。
3. 按 accelerator ACL、队列配额和 lease 状态分配切片。
4. 通过 RPC 返回真实的 chrdev、block、backing mount、view、BDF、namespace 和
   队列元数据。
5. 在 client 断开、心跳超时或进程死亡时回收 lease 记录。

daemon 不负责创建 GPU context、不 mmap client 的 ring、不提交 I/O。client 的
fd 关闭仍由内核回收 client-owned queue、ring 和 DMA map；daemon 只负责自身拥有
的 controller 状态。

### 4.2 StorageRuntime

`StorageRuntime` 是应用进程内的单设备控制面和数据面入口，负责：

- 固定并暴露 `accel_id`；
- 接收 factory 已构造好的 resolver 和 DataPath bindings；
- 管理 target、memory、I/O handle、DataPath 初始化以及 shutdown 顺序；
- 在 accelerator 调用前切换到 Runtime 的设备，并在调用后恢复线程原设备。

`StorageRuntime` 不连接 daemon，不拥有 allocation/lease 或全局 NVMe 设备表，也
不把其他 Runtime 的句柄、memory 或 stream 混入自己的 submit。

### 4.3 当前能力、架构契约与本阶段装配边界

“一个 Runtime 只允许一个 resolver/DataPath”不是当前 `StorageRuntime` 核心行为，
也不是既有架构文档定义的通用契约。三者的准确关系如下：

| 层次 | 当前实现 | 既有架构文档 | 本次选择 |
| --- | --- | --- | --- |
| resolver 路由 | `RuntimeComponents::resolvers` 是 vector，Runtime 按唯一 scheme 存入 map | URI 按 scheme 路由到 resolver | 保持不变；不新增同 scheme 多 resolver |
| DataPath 路由 | `RuntimeComponents::data_paths` 是 vector，Runtime 按唯一 key 存入 map，并初始化多个 DataPath | resolved target 按 recommended key 路由，batch 按 DataPath 分组 | 保持不变；所有 device DataPath 必须属于同一 `accel_id` |
| contract test | test 83 已装配两个 scheme 和两个 key，并验证跨 DataPath batch 分组 | 与按 scheme/key 路由的描述一致 | 必须作为回归测试保留 |
| 当前 loader | 对每个 spec 重复注册固定 `file` scheme 和 `local-nvme-ext4` key，第二项会被 Runtime 判为 duplicate | 不代表核心架构限制 | 改为只产生一个顶层 Local 或 Striped 组合 |

对应实现和架构依据见：

- [`storage_runtime.h`](../../tutti/include/tutti/storage_runtime.h) 中的
  `RuntimeComponents`、`initialize_components_()`、resolver/DataPath map 和
  `initialized_data_paths_`；
- [`storage_runtime_contract_test.cpp`](../../tests/storage_runtime_contract/storage_runtime_contract_test.cpp)
  的 test 83；
- [`system-architecture.md`](../architecture/system-architecture.md) 的
  StorageRuntime 职责，以及 [`backend-spi.md`](backend-spi.md) 的 Runtime routing
  and grouping；
- [`tutti_config.cpp`](../../tutti/config/tutti_config.cpp) 当前固定 scheme/key 的
  loader 代码。

因此，本任务既不扩展也不收缩核心路由契约。产品 loader/factory 只对一个 Runtime
注入一组顶层 resolver 和 DataPath：

| 模式 | daemon slice 数量 | Resolver | DataPath |
| --- | ---: | --- | --- |
| `allowed` | 1 | `LocalFileResolver` | `LocalNvmeDataPath` |
| `explicit` | 1 | `LocalFileResolver` | `LocalNvmeDataPath` |
| `striped` | 2 或更多 | `StripedResolver` | `StripedDataPath` |

`StripedResolver` 内部可以持有多个 shard resolver，`StripedDataPath` 内部可以持有
多个 controller/queue group；它们对 `StorageRuntime` 仍表现为唯一的一组组件。
本任务不增加同 scheme 多 resolver 试探或动态选择，也不让产品 loader 装配多个
独立 local-NVMe DataPath；其他调用方仍可继续使用核心已有的多 scheme、多 key
能力。

### 4.4 loader/factory 与 Runtime bundle

产品 loader/factory 负责 daemon 交互和产品装配，返回的 owned Runtime bundle
必须共同拥有 daemon client、一个逻辑 allocation、resolver/DataPath 实例和
`StorageRuntime`，并显式实现关闭顺序。它负责：

1. 解析应用配置并向 daemon 执行一次 `AcquireNvmeSlices`；
2. 从 allocation metadata 构造一组顶层 Local 或 Striped 组件；
3. 将 bindings 注入 `StorageRuntime::create()`；
4. 创建失败时按逆序回滚并 `Release(allocation_id)`；
5. 正常关闭时调用一次 `StorageRuntime::shutdown()`（由它停止已初始化的
   DataPath），随后销毁 Runtime 和组件，最后 Release。

这一区分保证 daemon RPC 不进入 `StorageRuntime` 公共 API，也使 allocation 生命周期
不会被误认为 Runtime 核心路由能力的一部分。

## 5. 编译期后端与 Runtime 身份

编译配置选择唯一后端，例如 `-DTUTTI_ACCELERATOR=CUDA`。建议把编译结果导出
为只读常量 `TUTTI_COMPILED_ACCELERATOR_PROFILE`，并将公共配置改为：

```cpp
struct RuntimeConfig {
    std::int32_t accel_id = TUTTI_DEFAULT_ACCEL_ID;
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
- `TUTTI_DEFAULT_ACCEL_ID` 在 accelerator profile 中为 `0`，在 HOST profile 中为
  `-1`。这样默认构造不会把 HOST profile 伪装成 accelerator device 0。

## 6. DataPath 绑定规则

通用 SPI 必须新增 accelerator 绑定信息；本任务不依赖或重新解释现有的
`supports_multi_gpu`/`supports_cross_device`：

```cpp
struct DataPathCapabilities {
    // -1: host-only; >= 0: fixed accelerator device
    std::int32_t bound_accel_id = -1;
    // existing capability fields remain unchanged
};
```

规则：

1. `bound_accel_id >= 0` 的 DataPath 固定使用该 accelerator device。
2. `bound_accel_id == RuntimeConfig.accel_id` 才能挂载到 accelerator Runtime。
3. 核心允许一个 Runtime 挂载多个 DataPath；每个支持 device execution 的
   DataPath 都必须满足 `bound_accel_id == RuntimeConfig.accel_id`。本阶段产品
   loader 只挂载一个顶层 DataPath。
4. `bound_accel_id == -1` 的 host-only DataPath 不占用 accelerator 资源，只能做
   host execution。MVP 允许它挂载到 accelerator Runtime；`accel_id == -1` 的
   host-only Runtime 则只能挂载 host-only DataPath。
5. 本任务装配的 Local/Striped DataPath 继续令 `supports_multi_gpu` 和
   `supports_cross_device` 为 `false`；若以后需要 vendor-neutral rename，应作为
   独立 SPI 变更定义清楚“多个 accelerator”与“跨 accelerator submit”的区别。
6. striped DataPath 可以管理多个 NVMe slice，但其执行 accelerator 仍只有一个，
   即 `bound_accel_id` 不变。多盘不等于多 GPU。

特别是，不能把 `supports_multi_gpu` 重命名成 `supports_multi_device` 后用于表示
“聚合多个 NVMe”。前者描述 accelerator 执行拓扑，后者会与 storage device 数量
混淆；striped 的多盘事实由其 slice/device descriptor 表达。

目标 DataPath 必须显式提供 `bound_accel_id`。缺少绑定信息或出现设备冲突时，
Runtime 在创建阶段拒绝；不能通过“未知设备”把 DataPath 挂到任意 Runtime。

## 7. daemon 资源模型与 RPC

### 7.1 配置模型

daemon 配置描述全机事实和 ACL，不为每个 GPU 启动独立 daemon：

```yaml
accelerators:
  - { accel_id: 0, view_root: "/mnt/gpu0" }
  - { accel_id: 1, view_root: "/mnt/gpu1" }

nvmes:
  - device_id: 0
    pci_addr: "0000:31:00.0"
    namespace_id: 1
    backing_mount_path: "/mnt/nvme0"
    allowed_accel_ids: [0, 1]
  - device_id: 1
    pci_addr: "0000:52:00.0"
    namespace_id: 1
    backing_mount_path: "/mnt/nvme1"
    allowed_accel_ids: [1]
```

`allowed_accel_ids` 是“可以申请”的 ACL，不是已经验证过 GPU DMA/P2P/I/O 的结论。
空 ACL 的默认值仍可表示所有已枚举 accelerator，但必须在 daemon 日志和 RPC
中明确展开后的集合。其数值必须引用 `accelerators[].accel_id`，并满足第 3 节的
deployment mapping 前置条件。

目标 schema 中，`accelerators`、`accelerators[].accel_id`、
`accelerators[].view_root`、
`nvmes[].allowed_accel_ids`、`nvmes[].backing_mount_path` 和显式
`nvmes[].device_id` 都是 canonical 字段。`device_id` 必须配置且唯一，反转
`nvmes` 数组顺序不能改变资源身份。YAML 保留 `pci_addr` 名称，它必须保存规范化
PCI BDF，并映射到新 RPC 的 `pci_bdf`。

### 7.2 配置与接口命名迁移

当前实现与目标契约的迁移关系如下：

| 当前 legacy 名称/行为 | 目标名称/行为 | 迁移规则 |
| --- | --- | --- |
| `gpus[].id` | `accelerators[].accel_id` | checked-in YAML、parser model、错误信息和测试一起修改 |
| accelerator 身份映射 | 直接使用 `accelerators[].accel_id` ordinal | daemon、client、Runtime 必须使用相同 backend ordinal；不做 UUID 或重映射校验 |
| `gpus[].mount_path` | `accelerators[].view_root` | 明确它是 per-accelerator view 的根目录 |
| `nvmes[].allowed_gpus` | `nvmes[].allowed_accel_ids` | 目标配置和新 RPC 只暴露新名称 |
| `nvmes[].mount_path` | `nvmes[].backing_mount_path` | 明确它是真实 namespace 文件系统的 backing mount |
| `nvmes` 数组下标充当 `device_id` | 显式 `nvmes[].device_id` | legacy 输入可临时按下标补齐并告警，目标输入必须显式配置 |
| 通用 C++/RPC 中的 `gpu_id`、`cuda_device` 或 accelerator 含义的 `device_id` | `accel_id` | 新 API 只使用 `accel_id`；CUDA 私有代码可以保留 `cuda_device` |

迁移采用一个有期限的兼容窗口：parser 可以在一个过渡版本接受 legacy YAML，
但必须输出 deprecation warning；同一文件同时出现新旧字段时直接报错，不能静默
选择其一。仓库内配置、测试 fixture 和用户文档在 parser 改动的同一阶段全部切换
到 canonical schema，生产 gate 前移除 legacy YAML alias。旧
`ListDevices/Connect/Disconnect` RPC adapter 可以暂时保留原 protobuf 字段名以
维持 wire/source compatibility，但内部模型和所有新 RPC 必须立即规范化为
`accel_id`/`allowed_accel_ids`。

当前 [`config/local/daemon_2disk.yaml`](../../config/local/daemon_2disk.yaml) 仍是
legacy `gpus`/`allowed_gpus` schema；它是迁移输入，不是本文目标 schema 的例外。
进入目标 daemon integration 和 E2E 验证前必须完成转换。

### 7.3 逻辑 RPC

协议直接定义 NVMe 资源发现和切片分配：

```text
ListAccelerators()
  -> daemon 配置的 accel_id 和 view_root

ListNvmeResources()
  -> 所有 daemon device_id 及其真实 path、BDF、namespace、队列能力、ACL

AcquireNvmeSlices(accel_id, selection, queue_budget)
  -> 一个 allocation_id + 一个或多个 NvmeSlice

NvmeSlice
  -> device_id, pci_bdf, chrdev_path, block_path, backing_mount_path,
     view_path, namespace metadata, queue quota, lease parameters

Release(allocation_id)
```

daemon 不调用 accelerator API；`ListAccelerators` 只返回配置事实。loader 和 daemon
直接按 `accel_id` ordinal 做归属检查，`pci_bdf` 仅用于诊断。

`selection` 可以是显式 daemon `device_id` 列表、按 ACL 自动选择，或一个
striped group。MVP 对三种模式定义确定语义：

- `allowed`：daemon 从 ACL 允许且当前可分配的资源中按 `device_id` 升序选择第一
  个 slice；后续如引入 NUMA/优先级策略必须另行版本化。
- `explicit`：请求必须且只能包含一个 `device_id`；成功返回一个 slice。
- `striped`：请求包含两个或更多 `device_id`；全部 slice 一次成功或全部失败。

daemon 可以在内部为 striped 的每个 NVMe 保存子 lease，但对 Runtime bundle 必须
返回一个逻辑 `allocation_id` 和一个有序 slice 列表。每个条目都携带 daemon
`device_id`、请求的 `accel_id` 和真实 metadata，列表顺序同时定义 striped shard
顺序。任何一个 slice 的 ACL、路径或队列预算不满足时，不得保留部分 allocation。

`backing_mount_path` 是 daemon 挂载真实 namespace 文件系统的位置；`view_path` 是
本次 `accel_id` 对应的 client-visible view。resolver 只能使用返回的 `view_path`，
不得从任一 mount root 自行拼接路径。

### 7.4 分配与共享

- daemon 只分配配置 ACL 允许的 slice，并依据队列池做配额控制。
- striped allocation 采用 all-or-nothing 预留；校验全部资源后在同一临界区扣减
  各 controller 的队列预算，失败时不产生 allocation。
- daemon ledger 是 daemon-mediated client 的 admission control；kernel user-QID
  pool 仍是实际建队列的最终权威。client attach/add-queue 失败时必须 Release，
  不能把 ledger reservation 解释为队列已经创建成功。
- 默认允许多个 Runtime 共享一个 NVMe controller；独占模式作为部署策略，而非
  Runtime 身份的一部分。
- 一次 allocation 只对应一个 owned Runtime bundle。bundle 关闭时释放该
  allocation；不允许把 allocation 转移给另一个 bundle。
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
  device_ids: []                 # explicit 恰好一个；striped 至少两个
  queues_per_controller: 4
```

loader/factory 的职责是：

1. 读取 `runtime.accel_id`，创建一个 `RuntimeConfig`，并从编译 backend 查询
   `accel_id`。
2. 用 `ListAccelerators` 校验请求的 `accel_id` 在部署映射中存在，再向 daemon 请求
   该 `accel_id` 允许的 NVMe slice，并持有返回的一个逻辑 allocation。
3. 按 selection 创建唯一的一组组件：单 slice 创建 local 组件，多 slice 创建
   striped 组件。
4. 只向 Runtime 注入一个 resolver binding 和一个 DataPath binding；不得把每个
   `allowed_accel_ids` 或每个 striped shard 展开成 Runtime 级组件。
5. 发现 DataPath 绑定了其他 accelerator 时在 `create()` 失败；不要把所有
   `allowed_accel_ids` 展开成一个 Runtime。
6. 上层需要多 accelerator 时重复执行上述流程，形成 Runtime bundle 集合：

```cpp
RuntimeConfig cfg0;
cfg0.accel_id = 0;
RuntimeConfig cfg1;
cfg1.accel_id = 1;
auto r0 = create_runtime_bundle(cfg0, daemon, policy_for_accel0);
auto r1 = create_runtime_bundle(cfg1, daemon, policy_for_accel1);
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

目标通用 API 将当前 accelerator 含义的字段统一为 `accel_id`：
`DeviceInfo.accel_id`、`DeviceCapabilities.accel_id`、`MemorySpec.accel_id`、
`MemoryInfo.accel_id`、`DataPathMemoryView.expected_accel_id` 和
`HostSubmitContext.accel_id`。这些通用类型是否另行重命名不影响本任务，但其
accelerator 身份字段不能继续叫 `device_id`。

`register_memory()` 可以接受外部 memory 的未指定 accelerator
（`expected_accel_id < 0`），但在第一次被 device DataPath 注册时必须解析到
Runtime 的 `accel_id`。
显式冲突不应延迟到 I/O 完成后才暴露。

当前阶段不要求所有 accelerator 都实现 `allocate_memory(DEVICE/MANAGED)`；尚未
支持时返回 `UNSUPPORTED` 比调用 `malloc()` 后标记成 device memory 更安全。

### 9.2 submit

- `DEVICE_EXECUTION` 的 context、stream、target 和 memory 必须属于同一 Runtime
  `accel_id`；一个 batch 天然只对应一个 accelerator device。
- `HostSubmitContext.accel_id == -1` 可解释为“使用 Runtime 的 `accel_id`”；显式不同值
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
loader/factory 解析 config(accel_id)
        │  profile/accel_id 基本校验
        ▼
AcquireNvmeSlices(accel_id, policy)
        │  返回一个 allocation_id + 有序 slice metadata
        ▼
构造同 accel_id 的 Local/Striped 组件并 StorageRuntime::create()
        │  单盘为 Local；多盘为一个 Striped DataPath
        │
open/register/submit/wait/progress
        │  每次 accelerator 调用设置并恢复 current device
        ▼
drain I/O -> close targets -> unregister memory -> StorageRuntime::shutdown()
        -> Runtime STOPPED -> 销毁组件 -> Release(allocation_id)
```

任一步初始化失败都由 owned bundle 按逆序清理组件并释放 allocation。
`StorageRuntime::shutdown()` 不连接或停止 daemon，也不解绑其他 bundle 正在使用的
controller。

## 12. 分阶段实现策略

每个阶段都应保持可独立合入、可回退，并在进入下一阶段前满足对应 exit gate。
实现顺序不能从 loader 直接开始；必须先固定 Runtime/DataPath 身份，否则 loader
仍会把错误设备带到 DMA map 或 kernel launch 才暴露。

### 12.0 当前实现差距

| 区域 | 当前实现 | 本次目标 |
| --- | --- | --- |
| Runtime 身份 | `RuntimeConfig` 没有 `accel_id`，profile/device discovery 仍是 stub | 不可变 `accel_id`、编译 profile、一创建一设备 |
| 通用身份命名 | accelerator 在 public API 中仍使用 `device_id`/`expected_device_id`，daemon/RPC 使用 `cuda_device` | 通用 API、新 RPC 和 target YAML 统一为 `accel_id` |
| DataPath SPI | 只有 `supports_multi_gpu`/`supports_cross_device`，没有强制 accelerator 绑定 | 新增唯一 `bound_accel_id`；不重新解释现有 capability |
| daemon schema | `gpus[].id/mount_path`、`allowed_gpus`、含混的 NVMe `mount_path`、数组下标作为 `device_id` | `accelerators[].accel_id/view_root`、`allowed_accel_ids`、`backing_mount_path`、显式 `nvmes[].device_id` |
| loader | 从 daemon YAML 展开 `allowed_gpus`、按数组序号合成 chrdev；多项 spec 会重复注册固定 scheme/key 并在 Runtime 创建时失败 | 从 Acquire metadata 构造唯一顶层 Local 或 Striped 组件，不改变核心路由能力 |
| daemon RPC | `Connect` 一次只分配一个 device；返回单 session，尚无 striped 原子 allocation | 三种 selection、一个逻辑 allocation、N 个 slice |
| queue policy | 只 clamp 单 client 请求，没有 reservation ledger | daemon admission ledger + kernel 最终权威 |
| path 身份 | owner bring-up 已取得实际 minor/disk name，但上层仍有重建或拼接 | chrdev/block/BDF 全部作为 bring-up/RPC 事实传递 |
| accelerator 调用 | device 切换分散，部分 helper 设置后不恢复，部分释放路径依赖线程当前 device | Runtime/DataPath 双层 guard 和完整路径审计 |
| owned lifetime | loader bundle 的默认成员析构顺序不能表达 Runtime shutdown 后再销毁组件/Release | 显式事务回滚和 shutdown/Release 顺序 |

`StorageRuntime` 当前已经支持不同 scheme 的多个 resolver 和不同 key 的多个
DataPath，并有跨 DataPath batch contract test。本任务不新增核心路由能力，也不
删除该能力；只把产品 loader/factory 的输出限定为一组顶层 binding。

### 12.1 阶段 0：冻结契约与建立基线

实现范围：

1. 冻结两层契约：`StorageRuntime` 核心继续支持多 scheme/multi-key 路由；本阶段
   产品 loader 采用“一 Runtime、一 accelerator、一组顶层 resolver/DataPath”。
2. 固定 `allowed`、`explicit`、`striped` 的 slice 数量和原子性语义。
3. 冻结 canonical schema 和迁移规则：目标字段是 `accelerators[].accel_id`、
   `accelerators[].view_root`、`nvmes[].allowed_accel_ids`、
   `nvmes[].backing_mount_path` 和显式 `nvmes[].device_id`；当前
   `gpus`/`allowed_gpus` 只作为待迁移的 legacy baseline，不能出现在新增配置或
   新接口中。
4. 记录当前 host/CUDA contract test、daemon list/connect 和三类 local-NVMe
   hardware contract test 的基线结果；其中必须保留并运行 StorageRuntime test 83，
   证明本任务没有误删跨 DataPath 分组能力。
5. 修正文档中 `daemon_2_disk.yaml` 与实际
   `config/local/daemon_2disk.yaml` 的文件名、BDF 和 mount path 漂移；测试参数
   应优先来自 daemon RPC，而不是手工复制 YAML 数组顺序。

Exit gate：不改业务行为，现有无硬件 contract test 全绿；双盘 daemon 能列出两
个 NVMe resource，且 RPC metadata 与配置 BDF 一致。

### 12.2 阶段 1：Runtime 和 DataPath 设备身份

实现范围：

1. 在构建系统导出只读的 `TUTTI_COMPILED_ACCELERATOR_PROFILE`，避免
   `RuntimeConfig.profile_name` 继续硬编码为 `host`。
2. 为 `RuntimeConfig` 增加不可变的 `accel_id`；`create()` 在任何 DataPath
   initialize 之前校验 profile 和 device count。
3. 将通用 API 中表示 accelerator 的 `device_id`/`expected_device_id` 字段迁移为
   `accel_id`/`expected_accel_id`，覆盖 accelerator info/capabilities、memory view
   和 `HostSubmitContext`；CUDA 私有实现中的 `cuda_device` 不要求在本阶段改名。
4. 为 `DataPathCapabilities` 只增加 `bound_accel_id`；Local 和 Striped DataPath 在
   构造时写入唯一绑定，现有 `supports_multi_gpu`/`supports_cross_device` 保持
   `false`，不改成 storage-device capability。
5. Runtime 创建时先完成无副作用 preflight：组件非空、每个 device DataPath 的
   `bound_accel_id` 都与 Runtime 匹配、host-only 规则满足，然后才 initialize；
   不在核心层增加 binding 数量限制，无组件的 contract-test stub 模式不受影响。
6. `query_cuda_like_profile()`、设备发现和诊断信息返回编译 profile、真实 device
   count 和可选 BDF 诊断信息；不再固定返回一个 stub device。Runtime 归属只依赖
   backend ordinal，不因 backend 无法提供 UUID/stable identity 而失败。
7. `allocate_memory(DEVICE/MANAGED)` 若尚未实现真正的 backend allocation，明确
   返回 `UNSUPPORTED`；不得继续用 `malloc()` 产生伪 device memory。

验证范围：

- 有效 `accel_id=0/1` 创建成功；负数、越界和 profile 不匹配在 initialize 前失败。
- accelerator Runtime + 同 `bound_accel_id` DataPath 成功；绑定冲突和 host-only
  反向挂载失败。
- 现有多 scheme、multi-key contract test 继续成功，跨 DataPath batch 仍按
  DataPath 分组；重复 scheme 或重复 key 继续返回 `INVALID_ARGUMENT`。
- 多个同 `accel_id` DataPath 创建成功；任一 device DataPath 绑定其他 accelerator
  时，整个 create 在 initialize 前失败。
- generic public headers 和新增测试中不再出现 accelerator 含义的 `device_id`；
  daemon NVMe `device_id` 与 Runtime `accel_id` 的类型/字段不会互换。
- backend 返回的 `accel_id` 与编译 backend ordinal 一致；daemon deployment mapping
  的 ordinal 校验在阶段 4 完成，BDF 只用于诊断。
- 跨 Runtime 的 target、memory、I/O handle 保持 `NOT_FOUND`/`INVALID_ARGUMENT`，
  即使两个 Runtime 的 `accel_id` 相同也不能互用。

Exit gate：所有身份冲突都在 attach、DMA map、kernel launch 和 doorbell 之前失败。

### 12.3 阶段 2：accelerator current-device 与归属检查

实现范围：

1. 提供 backend-neutral 的 RAII device guard：保存线程原 device、切换到
   Runtime/DataPath 的 `accel_id`。guard 创建失败立即返回；正常路径在返回前显式
   restore 并传播失败，析构函数只做 no-throw best-effort 恢复和诊断兜底。
2. Runtime 在 initialize、open、close、register/unregister、submit、progress、
   release 和 shutdown 的 DataPath 调用边界使用 guard。
3. Local/Striped DataPath 保留第二层 guard，使直接使用 DataPath 的硬件 contract
   test 也具备相同语义。
4. 清理内部遗漏路径，包括 queue-group 构造/析构、device-target build/free、
   arena/cache、event query、DMA map/unmap；不能只在 kernel launch 前切换设备。
5. `DEVICE` memory 显式 `expected_accel_id` 与 Runtime 不同时立即拒绝；未指定时
   解析为 Runtime `accel_id`。CUDA 后端可查询 pointer/stream 归属时执行第二次
   fail-closed 校验；其他后端无法查询时记录清晰的调用前置条件。
6. `HostSubmitContext.accel_id == -1` 解析为 Runtime `accel_id`；显式不同值拒绝。

验证范围：

- 在线程 current device 预先设为 GPU 1 时调用 Runtime 0 的完整生命周期，每次
  调用结束后仍为 GPU 1。
- GPU 0 Runtime 拒绝 GPU 1 pointer、GPU 1 stream 和显式 `accel_id=1` context。
- 两个线程分别使用 Runtime 0/1，current device 不互相污染。
- 对 initialize、submit、progress 和 shutdown 分别注入 backend API 失败，确认
  guard 不泄漏错误 device 状态。

Exit gate：不依赖调用方预先 `cudaSetDevice()`；每条 accelerator 路径都能证明
使用目标 device 并恢复调用前状态。

### 12.4 阶段 3：daemon resource 与原子 slice allocation

实现范围：

1. 迁移 daemon YAML/parser/internal model：`gpus[].id` 改为
   `accelerators[].accel_id`，`nvmes[].allowed_gpus` 改为
   `nvmes[].allowed_accel_ids`，新增 `accelerators[].accel_id`，将 GPU
   `mount_path` 改为 `view_root`、NVMe `mount_path` 改为 `backing_mount_path`，并
   要求显式、唯一的 `nvmes[].device_id`。
2. 过渡 parser 对 legacy alias 只保留一个版本：读取时告警，新旧字段同时出现时
   失败；将 `config/local/daemon_2disk.yaml`、daemon config tests、生成脚本和相关
   用户文档在同一改动中切换到 canonical schema。
3. 在保留旧 `ListDevices/Connect/Disconnect` 兼容入口的同时，增加或演进为
   accelerator-neutral 的
   `ListAccelerators/ListNvmeResources/AcquireNvmeSlices/Release`。
4. 新 RPC 和内部 model 使用 `accel_id`/`allowed_accel_ids`；旧 RPC adapter 的
   `cuda_device`/`allowed_gpus` 只在兼容边界转换，不能继续向 loader 泄漏。通用
   client CLI 使用 `--accel`；`--cuda` 只允许作为旧 client 的兼容参数。
5. `NvmeSlice` 返回实际 `chrdev_path`、ioctl `disk_name` 对应的 `block_path`、BDF、
   namespace、block/page size、BAR0、`backing_mount_path`、`view_path`、ACL、queue
   quota 和 lease 参数。
6. 扩展 `libnvm` owner bring-up 返回创建时实际取得的 chrdev path/minor；daemon
   不再从 YAML 数组序号推导 chrdev，也不从 `device_id` 拼 block path。
7. 引入 per-controller queue reservation ledger。`Acquire` 在同一锁范围内校验
   ACL、available 和所有 slice 的预算；striped 请求全量预留或全量失败。
8. `Release`、client disconnect、heartbeat timeout 和 PID/starttime reaper 都按
   allocation 一次性退还全部 slice 预算；重复 Release 保持幂等或返回明确状态。
9. daemon 只做控制面，不调用 accelerator runtime，不创建 accelerator context。

验证范围：

- 纯状态单测覆盖三种 selection、无效/重复 `device_id`、ACL 拒绝、预算不足、
  striped 中途失败无残留、Release/reaper 退款和并发 Acquire。
- schema contract 覆盖 canonical 配置成功、legacy-only 配置告警且规范化、混用新旧
  字段失败、缺失/重复 `device_id` 失败，以及 `allowed_accel_ids` 引用不存在的
  `accel_id` 失败。
- accelerator ordinal contract 覆盖有效/无效 `accel_id`、越界拒绝，以及 daemon/client
  使用同一 ordinal 映射的要求；不测试 UUID/BDF 不匹配或 `CUDA_VISIBLE_DEVICES` 重映射。
- gRPC client/server contract 覆盖 metadata 字段完整性和旧 RPC 兼容。
- 反转 YAML NVMe 顺序后 `device_id` 和 allocation 选择不变；制造 stale minor 后，
  返回 path 仍来自实际 bring-up 结果；BDF/chrdev/block 三方不一致时资源不进入
  `available`。
- `view_path` 必须属于请求 `accel_id` 对应的 configured view root；resolver 不使用
  `backing_mount_path` 或自行拼接路径。
- daemon 启动、list 和 acquire 前后均不出现在 accelerator compute-context 列表中。

Exit gate：Runtime 构造所需设备事实全部来自 allocation metadata；任何代码都不再
用 `accel_id`、`device_id` 或数组下标推导数据路径；仓库内 daemon 配置和目标文档
均使用 `accelerators`/`accel_id`/`view_root`/`allowed_accel_ids`/
`backing_mount_path`/显式 `device_id`。

### 12.5 阶段 4：allocation 驱动的单组装配 loader

实现范围：

1. 应用 loader 解析 `accelerator.profile`、`runtime.accel_id`、daemon endpoint、
   `nvme.selection/device_ids/queues_per_controller`；查询 backend device count 并
   确认请求的 ordinal 存在。loader 不再读取 daemon YAML，只消费新 RPC 的
   `allowed_accel_ids` 和 allocation metadata，也不合成 `/dev/ssnvmeN`。
2. `allowed/explicit` 构造唯一的 `LocalFileResolver + LocalNvmeDataPath`；
   `striped` 构造唯一的 `StripedResolver + StripedDataPath`。striped 的多个 shard
   resolver/DataPath device descriptor 只存在于组件内部。
3. loader 对 Runtime 只注册一个 resolver scheme 和一个 DataPath key；这是
   loader/factory 的产品约束，不是 `StorageRuntime::create()` 的新限制。本阶段
   不新增 composite resolver 或由 loader 使用多 key 路由。
4. 将 gRPC 生成类型转换为 Tutti 内部的只读 slice metadata，避免 DataPath SPI
   直接依赖 protobuf 类型。
5. 修正 owned bundle 的销毁顺序：drain I/O、Runtime close/unregister、调用一次
   `StorageRuntime::shutdown()`、销毁 Runtime、销毁 resolver/DataPath、Release
   allocation、销毁 daemon client。不能重复 shutdown DataPath，也不能依赖当前
   成员声明的反向析构顺序。
6. 创建失败保持事务性：Acquire 成功后任一 resolver/DataPath/Runtime 初始化失败，
   都清理已初始化对象并 Release；不得留下心跳线程或 queue reservation。
7. 提供可注入的 resource-client seam，使 loader 单测不需要 root、gRPC daemon、
   NVMe 或 GPU。

验证范围：

- fake client 分别返回 1 个和 2 个 slice，断言 loader 输出始终只有一个顶层
  resolver 和一个顶层 DataPath，2 个 slice 时类型为 striped。
- 同一测试套件继续运行 StorageRuntime 的多 scheme/multi-key contract test，区分
  “loader 选择使用一组”和“核心只能接受一组”。
- `explicit` 零个/多个 device ID、`striped` 少于两个、返回 slice 顺序或 accel_id
  不一致均 fail-closed。
- `ListAccelerators` 返回的 accelerator ordinal 不存在或不在 ACL 中时，不执行 Acquire
  或 DataPath initialize。
- 对 Acquire、第二个 slice、DataPath initialize、Runtime create 和 shutdown
  逐点注入失败，断言 Release 次数、顺序和 allocation_id 正确。
- daemon 不可达时 accelerator Runtime 创建失败；不再 fallback 到
  `/dev/ssnvme0`。

Exit gate：配置 loader 能从 daemon metadata 事务性创建一个单盘或 striped
Runtime，并保证失败与 shutdown 都释放 allocation。

### 12.6 阶段 5：单 Runtime 的单盘与 striped 实机闭环

先在一个 accelerator 上闭环，再引入两个 Runtime，避免把多盘问题和多 GPU
问题同时调试：

1. GPU 0 + NVMe 0：Local Runtime 完成 scratch file 读写。
2. GPU 0 + NVMe 1：只修改 daemon `device_id`，路径完全来自 allocation。
3. GPU 0 + NVMe 0/1：一个 Striped Runtime 完成跨 stripe boundary 读写、mixed
   batch、restart persistence。
4. GPU 1 重复上述单盘和 striped 场景，证明 DataPath 不含 GPU 0 假设。
5. 每个场景以另一 GPU 作为线程 current device 进入 initialize/submit/progress/
   shutdown，并检查恢复。

Exit gate：两个 NVMe 均可分别服务 GPU 0/1；一个 Runtime 通过唯一的 striped
resolver/DataPath 使用双盘并完成真实 I/O。

### 12.7 阶段 6：双 Runtime、双 accelerator 并发

实现和验证场景：

1. 同一进程创建 Runtime bundle 0 和 Runtime bundle 1；各 bundle 分别持有自己的
   daemon allocation、StorageRuntime、stream、workspace、memory registration 和
   handle registry。
2. Runtime 0 使用 NVMe 0、Runtime 1 使用 NVMe 1，并行读写不同 scratch file。
3. 两个 Runtime 共享同一 controller，各自申请较小 queue budget，并行 I/O 后
   独立 shutdown；关闭任一 Runtime 不影响另一方。
4. 两个 Runtime 各自持有一个双盘 striped allocation，并行执行不同 pattern，
   验证每个 controller 的 queue ledger 和每个 GPU 的 DMA mapping 隔离。
5. 交叉传递 target、memory、I/O handle、pointer、stream 和 context，确认在任何
   DataPath 副作用前拒绝。
6. 让其中一个 Runtime 初始化失败或进程异常退出，确认另一个 Runtime 和 daemon
   controller 生命周期不受影响，失败方 allocation 最终被释放或回收。

Exit gate：同进程双 Runtime 可持续并发真实 I/O，资源与故障隔离满足第 13 节矩阵。

### 12.8 阶段 7：生产前正确性加固

1. GPU DMA map、地址记录和释放逻辑严格绑定当前 controller；审计并对齐 snvme
   5.15 与 5.4 分支。
2. 增加冷启动、daemon 重启、stale minor、错误 block path/BDF、client SIGKILL、
   heartbeat timeout、accelerator visible-order remap、IOMMU on/off 和不同 P2P
   拓扑测试。
3. 为 allocation、Runtime、DataPath、controller、accel_id 和 current-device guard
   增加结构化诊断；日志不得混淆 daemon `device_id` 与 accelerator `accel_id`。
4. 压测 queue exhaustion/reuse、反复 create/shutdown、双 Runtime 长稳、并发
   register/unregister 和 timeout 后保守资源保留。
5. 所有宣称 `validated_available` 的组合都必须经过 client-side 最小真实 I/O；
   daemon `available` 本身仍不升级为该语义。

### 12.9 后续能力

以下能力明确不属于本次实现：

- 同一 scheme 下的多个 resolver 和动态 resolver 选择；这需要扩展核心路由语义。
- 产品 loader/factory 同时装配多个独立的顶层 resolver/DataPath 组合；核心已有
  多 scheme/multi-key 能力，但本次产品装配不使用。
- 跨 accelerator DataPath、peer memory、跨 GPU stream 和一个 batch 跨 Runtime。
- MANAGED memory 的迁移/一致性协议。
- 独立 probe worker、容量配额、优先级、NUMA affinity 和自动故障转移。

## 13. 验收场景

### 13.1 目标双盘验收配置

本机硬件验收使用迁移后的 `config/local/daemon_2disk.yaml`。当前仓库版本仍使用
legacy `gpus`/`allowed_gpus` 且没有显式 `device_id`；阶段 3 必须先将其转换为
`accelerators`/`accel_id`/`view_root`/`allowed_accel_ids`/
`backing_mount_path`/显式 `device_id`，以下验收不得把 legacy schema 视为目标行为：

- accelerator：GPU 0、GPU 1；当前机器均为 NVIDIA H100 PCIe。
- 两个 accelerator 直接使用 backend ordinal `accel_id=0/1`；BDF 可记录为诊断信息，
  不参与身份匹配。
- NVMe `device_id=0`：BDF `0000:41:00.0`，`backing_mount_path`
  `/mnt/snvme/nvme1`。
- NVMe `device_id=1`：BDF `0000:44:00.0`，`backing_mount_path`
  `/mnt/snvme/nvme2`。
- 两个 NVMe 的 `allowed_accel_ids` 都是 `[0, 1]`。
- daemon queue policy 为每 client 默认/最大 32，但 client 仍受 kernel
  `max_queues_per_group` 限制。双 Runtime 首轮验证建议每 controller 只申请 4 个
  queue，确认共享与回收后再提升压力。

因为当前硬件配置对 GPU 0/1 都放行，按盘 ACL 不对称的拒绝场景应先用纯状态单测
覆盖；实机可用不存在的 `accel_id` 验证 ACL/设备拒绝。若必须验证 GPU 0/1 的
不对称 ACL，应使用单独测试 YAML 重启 daemon，不修改本文件的常用双盘配置。

striped 测试开始前必须从 RPC 确认两个 namespace 的 block size 相同；不能仅凭
YAML 假设为 4 KiB。

### 13.2 分层验证矩阵

| 层次 | 是否需要 root/GPU/NVMe | 覆盖内容 | 失败定位 |
| --- | --- | --- | --- |
| Runtime/SPI contract | 否 | accel/profile、多 binding 路由回归、全量 DataPath binding、handle 隔离 | Runtime/SPI |
| daemon schema/allocator unit | 否 | canonical/legacy 迁移、ACL、selection、queue ledger、原子 striped、reaper | daemon config/state |
| loader fake-client | 否 | metadata 驱动的单组顶层装配、事务回滚、析构顺序 | loader/lifetime |
| CUDA device contract | GPU | guard、pointer/stream 归属、双线程 current device | accelerator 层 |
| daemon integration | root + NVMe | bring-up、真实 path、RPC、lease/queue 回收 | daemon/libnvm |
| local/striped I/O | root + GPU + NVMe | DMA map、queue、kernel、真实数据正确性 | DataPath/kernel |
| multi-Runtime E2E | 全部 | 双 GPU 并发、共享、故障和资源隔离 | 业务闭环 |

无硬件层必须先全绿；硬件测试应从单 Runtime 单盘开始，再到单 Runtime striped，
最后执行双 Runtime。不得用 `--skip-io` 代替最终验收。

### 13.3 建议的实机执行流程

构建现有硬件 contract test 和计划新增的多 Runtime contract test：

```bash
cmake --preset cuda --fresh -DTUTTI_BUILD_HARDWARE_TESTS=ON
cmake --build --preset cuda --parallel 8 --target \
  tutti_local_nvme_datapath_contract_test \
  tutti_storage_runtime_local_nvme_contract_test \
  tutti_striped_local_nvme_contract_test \
  tutti_multi_accelerator_runtime_contract_test
```

在独立终端启动 daemon。sudo 密码从受保护文件输入，不打印到日志：

```bash
sudo -S env TUTTI_VERBOSE=1 \
  build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon \
  --config config/local/daemon_2disk.yaml < ~/.passwd/1
```

首先只做资源发现，确认 accelerator `accel_id`/`view_root` 映射，以及两个 NVMe
BDF、真实 chrdev/block/backing/view path、ACL、block size、queue 能力和
`available`：

```bash
build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client \
  --endpoint 127.0.0.1:50051 --list-only
```

新增的 E2E 测试建议直接使用 daemon RPC，不再要求人工拼接 `--nvme` 字符串：

```bash
build/cuda/bin/tutti_multi_accelerator_runtime_contract_test \
  --endpoint 127.0.0.1:50051 \
  --accel0 0 --accel1 1 \
  --device0 0 --device1 1 \
  --queues-per-controller 4
```

测试只允许读写各 mount 下唯一命名的 scratch file；不得对裸 namespace 做破坏性
覆盖。测试前后记录 daemon allocation/queue ledger、client exit status 和 kernel
group/map 回收日志。daemon 启动、list 和 acquire 前后还应检查 accelerator
compute-process 列表，证明 daemon 未创建 context。

### 13.4 必须通过的业务场景

| ID | 场景 | 预期结果 |
| --- | --- | --- |
| A1 | Runtime 0 + Local NVMe 0 | 创建、真实读写、shutdown、Release 成功 |
| A2 | Runtime 1 + Local NVMe 1 | 不含 GPU 0/path-index 假设，真实读写成功 |
| A3 | Runtime 0 + Striped NVMe 0/1 | loader 仅装配一组顶层组件，跨 stripe 数据正确 |
| A4 | Runtime 1 + Striped NVMe 0/1 | 所有 workspace、DMA map、kernel 属于 GPU 1 |
| B1 | Runtime 0/1 分盘并发 | 两组 handle/stream/allocation 隔离，数据均正确 |
| B2 | Runtime 0/1 共享一个 NVMe | queue 预算分别预留和退还，任一 shutdown 不影响另一方 |
| B3 | Runtime 0/1 各自双盘 striped | 双 controller、双 GPU 并发，无交叉 pattern |
| C1 | current device 预设为非目标 GPU | 每个 API 使用目标 GPU，返回后恢复原 device |
| C2 | 错误 handle/pointer/stream/context | 在 DMA map、kernel、doorbell 前拒绝 |
| C3 | striped 第二 slice 失败 | Acquire/初始化全量回滚，无部分 lease 或 DataPath |
| C4 | client 异常退出/heartbeat timeout | kernel fd 回收实际资源，daemon 最终退还 ledger |
| C5 | stale path 或 BDF 不一致 | resource 不发布为 available，Runtime 创建失败 |
| C6 | daemon boot/list/acquire | daemon 不创建 accelerator context |

### 13.5 发布门槛

- **MVP gate**：A1、A2、A3、B1、C1、C2、C3、C6 通过。
- **双盘双 GPU gate**：A4、B2、B3 通过，连续 create/shutdown 100 轮无 lease 或
  queue 泄漏。
- **生产 gate**：C4、C5、5.15/5.4 一致性、长稳和不同 IOMMU/P2P 拓扑通过。

任何阶段若只能证明 daemon `allowed/available` 而没有真实 I/O，不得标记为
`validated_available` 或宣称多 accelerator 数据路径已完成。

## 14. 主要代码落点

- Runtime 身份、所有 DataPath 的 binding preflight 和句柄/内存校验：
  [`tutti/include/tutti/storage_runtime.h`](../../tutti/include/tutti/storage_runtime.h)。
- DataPath 绑定能力：
  [`tutti/include/tutti/spi/data_path.h`](../../tutti/include/tutti/spi/data_path.h)。
- Local/Striped 的 device guard、DMA 和真实 I/O：
  [`tutti/data_paths/local_nvme`](../../tutti/data_paths/local_nvme) 与
  [`tutti/data_paths/striped_local_nvme`](../../tutti/data_paths/striped_local_nvme)。
- daemon RPC、resource state、lease 和 queue ledger：
  [`tutti/device_manager/nvme/nvmeservice`](../../tutti/device_manager/nvme/nvmeservice)。
- daemon canonical schema、legacy alias 迁移和 checked-in 双盘配置：
  [`nvmeservice_config.cpp`](../../tutti/device_manager/nvme/nvmeservice/src/nvmeservice_config.cpp)
  与 [`daemon_2disk.yaml`](../../config/local/daemon_2disk.yaml)。
- allocation 驱动的单组顶层装配和 owned lifetime：
  [`tutti/config`](../../tutti/config)。
- 无硬件、硬件和新增双 Runtime contract test：
  [`tests`](../../tests)。

这些改动不要求修改 resolver SPI 或 `StorageRuntime` 的多 binding 路由语义；
Local 和 Striped resolver 继续作为产品 loader 的两种互斥装配模式，其他调用方
已有的多 scheme/multi-key 组合继续有效。

## 15. 与现有文档的关系

- daemon bring-up、module 顺序和 lease 的现状见
  [`doc/tutti_daemon.md`](../tutti_daemon.md) 和
  [`tutti/device_manager/nvme/nvmeservice/NVMeService.md`](../../tutti/device_manager/nvme/nvmeservice/NVMeService.md)。
- DataPath SPI 的通用生命周期见 [`backend-spi.md`](backend-spi.md)。
- local-NVMe 和 striped 的硬件 contract test 见
  [`doc/local_nvme_contract_tests.md`](../local_nvme_contract_tests.md)。

这些文档当前仍记录 legacy `gpus`/`allowed_gpus`/`cuda_device` 接口，可以在阶段 0
用于建立现状基线，但不能作为目标命名的依据。阶段 3 修改 schema/RPC 时必须同步
更新 `doc/tutti_daemon.md`、`NVMeService.md`、配置样例、生成脚本和相关 contract
test；不能让本文已经使用 `allowed_accel_ids`，而可执行验收文档继续要求
`allowed_gpus`。

# Tutti Runtime 配置、资源与装配流程

> **状态**：当前实现设计收束
>
> 本文基于当前生产代码，描述 `TuttiRuntime` 从读取 YAML 配置，到创建并持有
> `StorageRuntime` 的完整流程。配置示例以 `examples/tutti_runtime` 为准。

## 1. 范围

本文覆盖：

- `tutti/config/parser` 如何将 YAML 翻译为 `tutti/config/spec` 定义的
  `TuttiRuntimeSpec`；
- `tutti/config/spec` 如何提供默认值并验证静态配置；
- `TuttiRuntime` 如何消费已验证的 spec，创建和初始化 Resource；
- NVMe Resource 如何与 daemon 交互、持有 granted allocation，并分别向 Resolver
  factory 和 DataPath factory 提供运行期资源视图；
- Resolver 和 DataPath 的 factory 接口、输入来源和输出所有权；
- `TuttiRuntime` 如何构造 `RuntimeComponents`，调用 `StorageRuntime::create()`；
- `TuttiRuntime`、`StorageRuntime` 和各组件的所有权、失败回滚及销毁顺序。

本文不覆盖：

- daemon 自身的初始化、配置和内部实现；
- Resolver 和 DataPath factory 返回实例后的内部算法；
- `StorageRuntime` 内部 registry、target、memory、submit、wait、progress 等 I/O
  接口和实现；
- DataPath 的实际 I/O 初始化细节。本文只确认该初始化由谁触发，以及失败时由谁回滚。

## 2. 总体模型

配置到 Runtime 的主链路如下：

```text
YAML file
  |
  | parse_tutti_runtime_config(path)
  v
TuttiRuntimeSpec（缺省字段已经取得 spec 中定义的默认值）
  |
  | TuttiRuntimeSpec::validate()
  v
静态有效的装配图
  |
  | Resource factory + Resource::initialize()
  v
已授予的运行期 Resource
  |
  +-------------------------+
  |                         |
  | resolver resource view  | datapath resource view
  v                         v
Resolver factory        DataPath factory
  |                         |
  | resolver instance       | instance + initialize config
  +-------------+-----------+
                |
                | RuntimeComponents（借用组件指针）
                v
       StorageRuntime::create()
                |
                v
       RUNNING TuttiRuntime
```

该模型中有三个必须分开的数据层次：

| 数据层次 | 典型内容 | 产生者 | 生命周期 |
| --- | --- | --- | --- |
| 期望配置 | accelerator、resource selection、请求队列数、组件 ID、类型、引用关系、调优参数 | YAML/parser，或调用方直接构造 spec | 创建期 value object |
| 运行期事实 | allocation ID、PCI BDF、字符/块设备路径、namespace、BAR0、MDTS、实际 granted queues | daemon，经 NVMe Resource 获取和验证 | Resource lease 生命周期 |
| 装配结果 | Resolver/DataPath 实例、路由 scheme、DataPath key、`DataPathConfig` | 两个 factory 和 `TuttiRuntime` | `TuttiRuntime`/`StorageRuntime` 生命周期 |

静态 spec 不保存 daemon grant，daemon grant 也不回写 spec。二者只在 Resource 和组件
factory 的边界上汇合。

## 3. 模块职责总览

| 模块 | 主要输入 | 主要输出 | 核心职责 | 不负责 |
| --- | --- | --- | --- | --- |
| Config parser | YAML 路径和 YAML 节点 | `Result<TuttiRuntimeSpec>` | YAML 加载、结构检查、字段转换、具体 config variant 构造 | 完整 spec 语义、硬件检查、对象创建 |
| Config spec | C++ spec value | `Status`、debug string | 默认值、字段约束、引用和拓扑校验、contract 校验 | YAML、RPC、硬件探测、运行期对象 |
| Resource factory | `ResourceSpec`、`ResourceCreateContext` | 未初始化的 `Resource` | 按 config variant 选择 concrete Resource | 初始化顺序和 Runtime 所有权 |
| NVMe Resource | NVMe 请求、runtime accel ID、daemon client | 初始化后的 allocation、两类 Resource view | daemon 发现/申请、grant 验证、lease 释放、信息投影 | Resolver/DataPath 构造、StorageRuntime 装配 |
| Resolver factory | `ResolverSpec`、Resource、`BackendSpec`、DataPath key | `StorageTargetResolver` | 校验关系，消费 resolver view，创建对应 Resolver | Resolver 的 Runtime 使用、对象长期所有权 |
| DataPath factory | `DataPathSpec`、Resource、`BackendSpec`、runtime accel ID | `CreatedDataPath` | 校验关系，消费 datapath view，创建 DataPath 和初始化配置 | 调用 `DataPath::initialize()`、对象长期所有权 |
| TuttiRuntime | config path 或 `TuttiRuntimeSpec` | 完整的 `TuttiRuntime` | 编排、环境校验、组件所有权、StorageRuntime 创建和整体销毁 | 重复实现 parser/spec/factory 规则 |
| StorageRuntime | `RuntimeConfig`、`RuntimeComponents` | `StorageRuntime` | 接收路由绑定并初始化 DataPath；销毁时停止 DataPath | 拥有 Resource、Resolver、DataPath 或理解 concrete backend spec |

`BackendSpec` 是装配关系，不对应一个运行期 backend 对象。它在创建期把一个
Resource、一个 Resolver 和一个 DataPath 连接起来，并携带三者共享的关系参数，例如
striped backend 的 `stripe_unit`。

## 4. Config parser 边界

### 4.1 公共入口

Parser 对外只提供一个入口：

```cpp
Result<TuttiRuntimeSpec> parse_tutti_runtime_config(
    const std::string& path);
```

实现位于 `tutti/config/parser/tutti_runtime_config_parser.cpp`。`YAML::Node`、yaml-cpp
异常和 YAML 路径错误都留在 parser 私有实现中，不进入 spec、Resource 或 Runtime
接口。

`TuttiRuntime::create(config_path)` 先调用该入口，成功后将返回的 spec 交给
`TuttiRuntime::create(spec)`。Parser 自身不调用 `spec.validate()`，所以其成功语义是
“YAML 已经可以完整翻译为 C++ spec”，而不是“该 spec 已经可以创建 Runtime”。完整
语义验证由 `TuttiRuntime::create(spec)` 统一触发。

### 4.2 Parser 负责的检查

Parser 负责所有依赖 YAML 表示的信息：

- 文件存在且可由 yaml-cpp 加载；
- root、`storage`、单个声明和 `config` 等节点是预期的 mapping/sequence/scalar；
- required field 存在，unknown field 被拒绝；
- YAML 标量可以转换到目标 C++ 整数范围；
- `selection` 等字符串可以翻译成 spec enum；
- 根据 `type` 或 `contract` 构造正确的 `std::variant` 分支；
- 错误消息携带类似 `storage.resources[0].allocation` 的 YAML 路径。

Parser 不判断以下语义：

- ID 是否重复、backend 引用是否存在；
- Resource、Resolver 和 DataPath 类型能否组成一个合法 backend；
- scheme 是否重复；
- accelerator 是否真实存在；
- daemon 是否能满足资源请求；
- factory 或具体组件是否在当前 build 中可用。

### 4.3 分发边界

顶层 parser 解析公共字段后，按声明类型分发到私有 parser：

| YAML 声明 | 分发键 | 目标 config variant |
| --- | --- | --- |
| Resource | `type: nvme` | `NvmeResourceConfig` |
| Resource | `type: memory` | `MemoryResourceConfig` |
| Resolver | `type: local-file` | `LocalFileResolverConfig` |
| Resolver | `type: striped-file` | `StripedFileResolverConfig` |
| Resolver | `type: memfs` | `MemfsResolverConfig` |
| DataPath | `type: local-nvme` | `LocalNvmeDataPathConfig` |
| DataPath | `type: striped-local-nvme` | `StripedLocalNvmeDataPathConfig` |
| DataPath | `type: memfs` | `MemfsDataPathConfig` |
| Backend relation | `contract: ext4-local-nvme` | `Ext4LocalNvmeBackendConfig` |
| Backend relation | `contract: striped-local-nvme` | `StripedLocalNvmeBackendConfig` |
| Backend relation | `contract: memfs` | `MemfsBackendConfig` |

这种分发让 YAML 解析逻辑与 spec 类型一一对应。新增组件类型时，应同时增加其 spec
config、私有 parser、spec validator 和 factory 分支，而不是在 `TuttiRuntime` 中解析
组件字段。

## 5. Config spec 边界

### 5.1 Spec 是唯一静态配置模型

顶层结构为：

```cpp
struct TuttiRuntimeSpec {
    AcceleratorSpec accelerator;
    RuntimeSpec runtime;
    StorageSpec storage;

    Status validate() const;
    Result<std::string> to_debug_string() const;
};
```

`StorageSpec` 分别保存 `ResourceSpec`、`ResolverSpec`、`DataPathSpec` 和
`BackendSpec` 数组。前三者是组件声明，`BackendSpec` 通过字符串 ID 引用它们：

```text
BackendSpec
  resource  ----> ResourceSpec.id
  resolver  ----> ResolverSpec.id
  datapath  ----> DataPathSpec.id
  contract  ----> 合法的类型、scheme、cardinality 组合
  config    ----> 关系自身的参数
```

Spec 是普通 C++ value object。调用方可以绕过 YAML 直接构造，因此所有静态语义必须由
`TuttiRuntimeSpec::validate()` 独立保证，不能只依赖 parser 的防御性检查。

### 5.2 默认值归属

默认值定义在 spec header 的成员初始化器中，parser 对 optional field 只在 YAML 出现时
覆盖它。这保证 YAML 和程序化构造共享同一套默认值。

当前主要默认值如下：

| 字段 | 默认值来源 | 当前默认值 |
| --- | --- | --- |
| `accelerator.profile` | `AcceleratorSpec` | 编译期 `TUTTI_COMPILED_ACCELERATOR_PROFILE` |
| `runtime.accel_id` | `RuntimeSpec` | 编译期 `TUTTI_DEFAULT_ACCEL_ID`；HOST 为 `-1`，加速器 profile 为 `0` |
| `NvmeAllocationSpec.selection` | NVMe resource spec | `Allowed`；当前 YAML parser 仍要求显式提供该字段，因此该默认值主要服务程序化构造 |
| `NvmeDataPathTuning.threads_per_block` | NVMe DataPath spec | `16` |
| NVMe DataPath 其余 tuning | NVMe DataPath spec | `0`；具体语义由 factory/concrete DataPath 解释 |
| striped `stripe_unit` | striped backend spec | `512 KiB` |

`provider.endpoint`、`queues_per_controller`、memory `capacity_bytes` 等字段没有可用的业务
默认值；parser 将其设为 required，spec validator 也会拒绝空值或零值。

例如 `examples/tutti_runtime/tutti_striped.yaml` 显式设置 `stripe_unit: 65536`；如果省略
该 optional field，parser 构造 `StripedLocalNvmeBackendConfig` 时会保留 spec 中的
`512 KiB` 默认值。

### 5.3 `validate()` 的职责

`TuttiRuntimeSpec::validate()` 是静态配置语义的权威入口，当前依次保证：

1. accelerator profile 只能是 HOST、CUDA、MUSA 或 MACA，且与 `accel_id` 的 host/
   accelerator 语义一致；
2. 每个声明的 ID、type、scheme、variant 和具体字段有效；
3. 同类声明 ID 唯一；
4. 每个 backend 引用的 Resource、Resolver、DataPath 都存在；
5. backend contract 与 Resource type、Resolver type/scheme、DataPath type 和资源数量匹配；
6. Resolver scheme 格式有效且全局唯一；
7. 一个 Resource 不会被两个不同 DataPath 消费；
8. 每个声明都可以从某个 backend 到达，不允许未使用声明；
9. 当前产品约束为恰好一个 backend。

“恰好一个 backend”加上“所有声明可达”意味着当前一个 `TuttiRuntimeSpec` 实际上只会
装配一个 Resource、一个 Resolver 和一个 DataPath。数据结构仍使用 vector，保留了未来
扩展装配图的表达能力，但当前 validator 不允许多 backend Runtime。

当前 contract 矩阵为：

| Contract | Resource | Resolver | Scheme | DataPath | Resource cardinality |
| --- | --- | --- | --- | --- | --- |
| `ext4-local-nvme` | `nvme` | `local-file` | `file` | `local-nvme` | 1 |
| `striped-local-nvme` | `nvme` | `striped-file` | `striped` | `striped-local-nvme` | 至少 2 |
| `memfs` | `memory` | `memfs` | `memfs` | `memfs` | 1 |

其中 striped contract 还要求 `stripe_unit` 非零且按 4096 字节对齐。

这些检查只基于配置即可完成。daemon 实际返回几个 slice、每个 slice 的队列数和设备
元数据是否有效，属于 Resource 初始化后的动态校验。

### 5.4 Debug 输出

`to_debug_string()` 先再次调用 `validate()`，再输出包含默认值在内的确定性配置文本。
`TuttiRuntimeCreateOptions::spec_debug_logger` 可在创建任何 Resource 之前接收这份文本。
因此日志表达的是将被消费的 effective spec，而不是原始 YAML 文本。

## 6. TuttiRuntime 创建入口与环境校验

公共入口有两种：

```cpp
static Result<std::unique_ptr<TuttiRuntime>> create(
    const std::string& config_path,
    TuttiRuntimeCreateOptions options = {});

static Result<std::unique_ptr<TuttiRuntime>> create(
    config::TuttiRuntimeSpec spec,
    TuttiRuntimeCreateOptions options = {});
```

两条路径最终都进入同一个 `create_with_options_()`。创建顺序为：

1. `spec.validate()`；
2. 生成并可选记录 spec debug string；
3. 校验 Runtime 环境；
4. 创建并初始化所有 Resource；
5. 按 `BackendSpec` 解析关系，分别调用 Resolver 和 DataPath factory；
6. 将实例注册到 `TuttiRuntime` 的 ownership registry，同时构造
   `RuntimeComponents`；
7. 调用 `StorageRuntime::create(RuntimeConfig, RuntimeComponents)`；
8. 保存 `StorageRuntime`，将 `TuttiRuntime` 状态置为 `RUNNING`。

环境校验不同于 spec 校验。它比较 `accelerator.profile` 和实际编译 profile，并在
`accel_id != -1` 时查询当前 accelerator backend 的 device count，拒绝超出范围的
device ID。这些结果依赖当前 build 和机器，不能放入 host-only spec validator。

## 7. Resource 边界

### 7.1 通用接口

Resource 抽象提供生命周期、诊断信息和两个面向消费者的窄视图：

```cpp
class Resource {
public:
    virtual const ResourceCapabilities& capabilities() const = 0;
    virtual Status initialize() = 0;
    virtual Status shutdown() = 0;
    virtual ResourceInfo info() const = 0;
    virtual Result<std::unique_ptr<const ResourceView>>
        get_resolver_view() const = 0;
    virtual Result<std::unique_ptr<const ResourceView>>
        get_datapath_view() const = 0;
};
```

通用创建入口为：

```cpp
Result<std::unique_ptr<Resource>> create_resource(
    const config::ResourceSpec& spec,
    const ResourceCreateContext& context);
```

输入来源如下：

| 输入 | 来源 | 含义 |
| --- | --- | --- |
| `ResourceSpec` | `spec.storage.resources` | Resource ID、type 和具体资源请求 |
| `ResourceCreateContext.runtime_accel_id` | `spec.runtime.accel_id` | 该 Resource 归属的 Runtime accelerator |

`resource_factory.cpp` 按 `ResourceSpec.config` 的 variant 分发到 NVMe 或 memory
Resource。factory 只构造 `CREATED` 状态对象；`TuttiRuntime` 随后调用
`Resource::initialize()`，并验证 Resource ID/type/capability/state 与 spec 一致，最后才
接管所有权。

这里的 `tutti::Resource` 是 TuttiRuntime 装配层的资源抽象，不是
`RuntimeComponents::resources` 所使用的 `ResourceProvider` SPI。前者负责兑现 daemon
资源并为两个组件 factory 提供 view；后者是 `StorageRuntime` 调用
`DataPath::initialize()` 时传入的通用 provider。当前 `TuttiRuntime` 不设置
`RuntimeComponents::resources`，StorageRuntime 会使用其内部默认 provider。换言之，
NVMe daemon grant 在 DataPath factory 构造实例时已经完成消费，不会继续穿透到
StorageRuntime。

### 7.2 NVMe Resource 的静态请求

以 `examples/tutti_runtime/tutti_local_nvme.yaml` 为例，静态 NVMe 请求只包含：

- daemon endpoint；
- selection mode：`allowed`、`explicit` 或 `striped`；
- explicit/striped 模式下期望的 daemon device ID；
- 每个 controller 请求的 queue 数；
- `runtime.accel_id`，由顶层 Runtime 配置提供。

PCI BDF、设备节点路径、mount/view 路径、namespace、block size、BAR0 size、MDTS 和实际
grant queue 数都不属于 YAML。

### 7.3 NVMe Resource 初始化与 daemon grant

NVMe Resource 的 concrete factory 使用 `provider.endpoint` 创建
`NvmeResourceClient`，但真正的 daemon 交互发生在 `NvmeResource::initialize()`：

```text
list_accelerators()
  -> 确认 runtime.accel_id 存在并有 view_root

list_nvme_resources()
  -> 确认请求设备存在、可用且 ACL 允许该 accelerator

acquire_nvme_slices(accel_id, selection, device_ids, queues)
  -> RuntimeNvmeAllocation { allocation_id, slices[] }

validate_allocation_metadata()
  -> ResourceState::INITIALIZED
```

每个 granted slice 包含：

```text
device_id, accel_id, allowed_accel_ids,
pci_bdf, chrdev_path, block_path,
backing_mount_path, view_path,
namespace_id, logical_block_size,
bar0_size, max_data_size, granted_queues
```

Resource 会校验 allocation ID、slice 数量/顺序、accelerator 和 ACL、所有必要路径、
namespace、block size、BAR0、MDTS、queue grant，以及多 slice block size 一致性。
如果 grant 已取得但后续验证失败，Resource 立即 release allocation。

daemon client 将 allocation handle 保存在 client 内部；`release(allocation_id)` 通过销毁
对应 handle 释放 lease。`NvmeResource::shutdown()` 和析构函数最终都会走该释放路径。

### 7.4 面向 factory 的两种投影

同一份 daemon grant 不直接暴露给所有消费者。NVMe Resource 投影成两种 view：

| View | 提供字段 | 消费者 |
| --- | --- | --- |
| `NvmeResolverResourceView` | device ID、PCI BDF、block path、mount/view path、namespace、logical block size | Resolver factory |
| `NvmeDataPathResourceView` | device ID、PCI BDF、accel ID、character device path、namespace、logical block size、BAR0、MDTS、granted queues | DataPath factory |

这两个 view 只能在 Resource 为 `INITIALIZED` 时取得，并以 value snapshot 的形式返回。
Resolver 无法通过其 view 取得 BAR0 或字符设备，DataPath 也不依赖文件系统 mount
信息。该隔离是 Resource 模块的主要边界，而不仅是类型转换便利。

## 8. Resolver factory 边界

### 8.1 接口和输入来源

```cpp
struct ResolverCreateContext {
    const Resource& resource;
    const config::BackendSpec& relation;
    std::string data_path_key;
};

Result<std::unique_ptr<StorageTargetResolver>> create_resolver(
    const config::ResolverSpec& spec,
    const ResolverCreateContext& context);
```

| 输入 | 来源 | 用途 |
| --- | --- | --- |
| `ResolverSpec` | backend 的 `resolver` ID 在 `storage.resolvers` 中解析 | 选择 Resolver 类型和其静态 config |
| `Resource&` | backend 的 `resource` ID 在已初始化 Resource registry 中解析 | 获取 resolver resource view |
| `BackendSpec` | 当前装配关系 | 校验 contract，并读取 `stripe_unit` 等关系参数 |
| `data_path_key` | 被引用的 `DataPathSpec.id` | 写入 Resolver 产生的 target，使其能路由到同一个 DataPath binding |

Factory 会防御性确认 relation 确实引用传入的 Resolver、Resource 和 DataPath key，再按
`ResolverSpec.config` variant 分发。它通过通用 `Resource::get_resolver_view()` 取得
snapshot，并 `dynamic_cast` 到 concrete view；类型不匹配即创建失败。

### 8.2 NVMe Resolver factory 消费的数据

local-file 组合要求一个 NVMe slice，构造 Resolver 时消费：

- `pci_bdf`；
- `namespace_id`；
- `logical_block_size`；
- `block_path`；
- `DataPathSpec.id` 形成的 DataPath key。

striped-file 组合要求至少两个 slice。Factory 为每个 slice 构造 local-file shard，再用：

- 所有 shard Resolver；
- `BackendSpec.config.stripe_unit`；
- `DataPathSpec.id` 形成的 DataPath key；

构造 striped Resolver。

Factory 的输出是未被 `StorageRuntime` 拥有的 `unique_ptr`。`TuttiRuntime` 将其注册到
`resolvers_` registry，并向 `RuntimeComponents` 只提供：

```cpp
ResolverBinding{
    .scheme = ResolverSpec.scheme,
    .resolver = borrowed_pointer,
};
```

## 9. DataPath factory 边界

### 9.1 接口和输入来源

```cpp
struct DataPathCreateContext {
    const Resource& resource;
    const config::BackendSpec& relation;
    std::int32_t runtime_accel_id = -1;
};

struct CreatedDataPath {
    std::unique_ptr<DataPath> instance;
    DataPathConfig initialize_config;
};

Result<CreatedDataPath> create_data_path(
    const config::DataPathSpec& spec,
    const DataPathCreateContext& context);
```

| 输入 | 来源 | 用途 |
| --- | --- | --- |
| `DataPathSpec` | backend 的 `datapath` ID 在 `storage.datapaths` 中解析 | 选择 concrete DataPath 并提供 tuning |
| `Resource&` | 已初始化 Resource registry | 获取 datapath resource view |
| `BackendSpec` | 当前装配关系 | 校验 DataPath 与 contract/Resource 的连接 |
| `runtime_accel_id` | `spec.runtime.accel_id` | 创建 context 中保留 Runtime 身份；当前 factory 尚未直接读取该字段，NVMe concrete binding 来自已验证 slice view |

与 Resolver factory 一样，DataPath factory 先防御性检查 relation，再调用
`Resource::get_datapath_view()` 并检查 concrete view 类型。

### 9.2 NVMe DataPath factory 消费的数据

local-nvme 组合要求一个 slice。构造 DataPath 时，factory 合并：

- spec tuning：cache capacity、threads per block、in-flight/batch 上限等；
- daemon grant：character device path、BAR0 size、accel ID、granted queues、namespace、
  logical block size、MDTS、PCI BDF。

striped-local-nvme 组合要求至少两个 slice。Factory 为每个 slice 构造 device descriptor，
并取所有 slice `max_data_size` 的最小值作为 effective MDTS。`threads_per_block` 不能超过
任何 slice 的实际 granted queues。

因此，诸如“请求 32 个 queue，但 daemon 只 grant 16 个，而 tuning 要求 32 个线程”的
冲突无法仅靠 spec 判断，应由 DataPath factory 在静态 tuning 与动态 grant 汇合处拒绝。

### 9.3 Factory 不初始化 DataPath

DataPath factory 返回 `CreatedDataPath`，其中：

- `instance` 由 `TuttiRuntime` 持有；
- `initialize_config` 随 `DataPathBinding` 传给 `StorageRuntime`。

Factory 不调用 `DataPath::initialize()`。该调用发生在
`StorageRuntime::create(RuntimeConfig, RuntimeComponents)` 内。这条边界保证
`StorageRuntime` 在注册完整路由后，以统一方式执行 DataPath 生命周期初始化和失败回滚。

## 10. 装配为 StorageRuntime

### 10.1 Backend relation 的解析

通过 spec validation 后，`TuttiRuntime` 可以把每个 `BackendSpec` 当作静态有效的装配
关系。当前实现按 ID 查找三个声明和已初始化 Resource：

```text
BackendSpec relation
  +-- relation.resource -> ResourceSpec + initialized Resource
  +-- relation.resolver -> ResolverSpec
  +-- relation.datapath -> DataPathSpec
```

随后先创建 Resolver，再创建 DataPath，并把两者存入各自 registry。这里不创建
`BackendInstance`；relation 完成 factory 参数传递和 binding key 对齐后即完成使命。

最关键的不变量是：

```text
BackendSpec.datapath
    == ResolverCreateContext.data_path_key
    == DataPathBinding.key
```

Resolver 解析 target 后给出的 recommended DataPath key，必然命中同一 relation 所绑定
的 DataPath 实例。

### 10.2 提交边界

`TuttiRuntime` 最终构造以下等价数据。实际代码使用 C++17 成员赋值和
`push_back()`：

```cpp
RuntimeConfig runtime_config;
runtime_config.accel_id = spec.runtime.accel_id;
runtime_config.profile_name = spec.accelerator.profile;

RuntimeComponents components;
components.resolvers.push_back(
    {resolver_spec.scheme, resolver_borrowed_pointer});
components.data_paths.push_back(
    {data_path_spec.id,
     data_path_borrowed_pointer,
     created_data_path.initialize_config});
```

然后调用：

```cpp
StorageRuntime::create(
    std::move(runtime_config),
    std::move(components));
```

`RuntimeComponents` 是 in-process assembly contract，只包含 SPI 指针和路由信息，不包含
concrete NVMe、filesystem 或 backend spec 类型。其 `resources` 字段在此保持
`nullptr`；它不接收也不拥有 `tutti::Resource`。

`RuntimeConfig.max_terminal_results` 没有对应的 TuttiRuntime spec 字段，因此此处保留
StorageRuntime 自身的默认值 `64`。这也说明 `TuttiRuntimeSpec` 目前只向
StorageRuntime 下传 accelerator 身份，不镜像其所有内部配置。

### 10.3 StorageRuntime 创建边界

本文只依赖 `StorageRuntime::create()` 的以下契约：

- `StorageRuntime` 不拥有 Resolver 或 DataPath，调用方必须让它们至少存活到
  `StorageRuntime::shutdown()` 完成；
- create 先校验 `RuntimeConfig` 和 binding 完整性/唯一性；
- create 检查 DataPath accelerator binding 与 Runtime 一致；
- create 使用每个 binding 的 `DataPathConfig` 调用 `DataPath::initialize()`；
- 如果一个 DataPath 初始化失败，create shutdown 已初始化的 DataPath，并且不返回
  `StorageRuntime` 实例；
- create 成功后，Resolver/DataPath 指针由 `StorageRuntime` 借用，实际对象继续由
  `TuttiRuntime` 持有。

StorageRuntime 如何使用 Resolver 打开 target，以及如何向 DataPath 提交 I/O，不在本文
范围内。

## 11. 所有权与生命周期

### 11.1 成功创建后的所有权

```text
application
  owns unique_ptr<TuttiRuntime>
        |
        +-- owns unique_ptr<StorageRuntime>
        +-- owns Resource registry
        +-- owns Resolver registry
        +-- owns DataPath registry

StorageRuntime
  borrows Resolver* and DataPath*

Resolver/DataPath instances
  were constructed from Resource view snapshots

Resource
  owns daemon client and active allocation lease
```

`TuttiRuntime::storage_runtime()` 返回 borrowed pointer。调用方不能单独销毁它，也不能让
它超出 `TuttiRuntime` 生命周期。

### 11.2 创建失败回滚

创建过程具有事务语义：任何阶段失败都不向调用方返回半初始化 `TuttiRuntime`。

- Resource factory 或 initialize 失败：当前 Resource 被 shutdown；此前已注册 Resource
  由临时 `TuttiRuntime` 析构逆序 shutdown；
- NVMe allocation 已取得但 grant 校验失败：NVMe Resource 当场 release；
- Resolver/DataPath factory 或 registry 失败：已创建实例由 `unique_ptr` 或 registry
  释放，已初始化 Resource 随 Runtime 回滚；
- `StorageRuntime::create()` 中 DataPath 初始化失败：StorageRuntime 先 shutdown 已初始化
  DataPath，随后临时 `TuttiRuntime` 销毁组件和 Resource；
- Resource、Resolver、DataPath 和 StorageRuntime factory 调用抛出的异常都会在
  `TuttiRuntime` 创建边界转换为 `Status`。

### 11.3 正常 shutdown 和销毁顺序

`TuttiRuntime::~TuttiRuntime()` 调用幂等的 `shutdown()`。当前顺序为：

```text
1. StorageRuntime::shutdown(0)
2. destroy StorageRuntime
3. destroy Resolver instances（逆注册顺序）
4. destroy DataPath instances（逆注册顺序）
5. Resource::shutdown()（逆初始化顺序）
6. destroy Resource；NVMe lease/client 在此之前已经释放
7. TuttiRuntime state = STOPPED
```

即使某一步返回错误，`TuttiRuntime` 也记录第一个错误并继续后续清理，避免 Resource
lease 因前序清理失败而无人释放。必须先销毁 `StorageRuntime`，因为它借用 Resolver 和
DataPath；必须最后 shutdown Resource，因为两个 concrete 组件都是由该 Resource 的
grant 构造出来的。

`StorageRuntime::shutdown(0)` 是非阻塞 drain：仍有 in-flight I/O 时可能返回 TIMEOUT。
当前 `TuttiRuntime::shutdown()` 即使收到 TIMEOUT，也会继续销毁 StorageRuntime、组件并
释放 Resource lease。因此“调用 shutdown”本身并不等于“已经完成 drain”；应用必须在
销毁 `TuttiRuntime` 前结束并回收全部 I/O，再显式检查 `TuttiRuntime::shutdown()` 的
返回值。否则借用组件和底层设备资源可能在未完成操作仍存在时被释放。这是当前创建/
销毁边界的前置条件，本文不扩展其 I/O drain 策略。

## 12. 两个 NVMe 示例的端到端数据流

### 12.1 Local NVMe

`examples/tutti_runtime/tutti_local_nvme.yaml` 声明：

```text
Resource:  example-nvme / nvme / explicit device 0 / request 32 queues
Resolver:  example-file-resolver / local-file / scheme=file
DataPath:  example-local-nvme-datapath / local-nvme / threads=32
Backend:   ext4-local-nvme，引用以上三个 ID
```

创建时的数据流为：

```text
runtime.accel_id = 0
  + ResourceSpec request
  -> daemon acquire
  -> one granted NVMe slice
       | resolver view: BDF, block path, NSID, block size
       |   -> LocalFileResolver(data_path_key =
       |                        "example-local-nvme-datapath")
       |
       | datapath view: chrdev, BAR0, accel, queues, NSID, block size, MDTS, BDF
       |   + DataPathSpec tuning
       |   -> LocalNvmeDataPath + DataPathConfig{"local_nvme"}
       v
RuntimeComponents:
  resolver["file"] -> resolver instance
  datapath["example-local-nvme-datapath"] -> datapath instance
```

### 12.2 Striped Local NVMe

`examples/tutti_runtime/tutti_striped.yaml` 声明两个 device、striped Resolver/DataPath 和
`stripe_unit: 65536`。它与 local 流程的差异只在 factory 内：

- NVMe Resource 要求 daemon 按请求顺序返回两个 slice；
- Resolver factory 为两个 slice 构造 file shard，并用 backend 的 stripe unit 构造
  `StripedResolver`；
- DataPath factory 为两个 slice 构造 device descriptor，以最小 MDTS 作为共同上限，
  构造 `StripedDataPath`；
- binding key 使用 YAML 中的 DataPath ID，scheme 使用 `striped`。

TuttiRuntime 的编排、所有权、StorageRuntime 创建和销毁流程与 local 模式完全相同。

## 13. 边界结论

当前实现可以收束为以下规则：

1. Parser 只负责 YAML 到 spec 的忠实、类型安全翻译；
2. Spec 定义所有静态字段和默认值，并独立保证整个装配图合法；
3. Resource 将静态资源请求兑现为动态 grant，并独占外部 lease 生命周期；
4. Resource 分别向 Resolver 和 DataPath factory 暴露最小必要的运行期视图；
5. Resolver/DataPath factory 是静态 spec 与动态 grant 的汇合点，只负责实例化；
6. `BackendSpec` 是关系，不是运行期对象；
7. `TuttiRuntime` 是唯一编排者和所有者，负责把 owned components 转换为
   `StorageRuntime` 借用的 bindings；
8. `StorageRuntime` 的边界从 `create(RuntimeConfig, RuntimeComponents)` 开始，在
   `shutdown()` 和对象销毁后结束；
9. 销毁严格遵循 StorageRuntime、组件、Resource/lease 的依赖顺序。

## 14. 代码索引

| 主题 | 代码位置 |
| --- | --- |
| TuttiRuntime 公共 API | `tutti/include/tutti/tutti_runtime.h` |
| 创建和装配编排 | `tutti/tutti_runtime/tutti_runtime_create.cpp` |
| shutdown 和 ownership registry | `tutti/tutti_runtime/tutti_runtime.cpp` |
| Parser 公共 API | `tutti/config/tutti_runtime_config_parser.h` |
| Parser 顶层分发 | `tutti/config/parser/tutti_runtime_config_parser.cpp` |
| Spec 顶层结构 | `tutti/config/tutti_runtime_spec.h` |
| Spec 验证 | `tutti/config/spec/tutti_runtime_spec.cpp` |
| Resource SPI 和 factory | `tutti/include/tutti/resource.h`、`tutti/resource/resource_factory.cpp` |
| NVMe Resource 和 daemon adapter | `tutti/resource/nvme/nvme_resource.cpp`、`tutti/resource/nvme/nvme_resource_grpc.cpp` |
| Resolver factory | `tutti/resolvers/resolver_factory.h`、`tutti/resolvers/resolver_factory.cpp` |
| DataPath factory | `tutti/data_paths/data_path_factory.h`、`tutti/data_paths/data_path_factory.cpp` |
| StorageRuntime 创建/销毁契约 | `tutti/include/tutti/storage_runtime.h` |
| 配置示例 | `examples/tutti_runtime/*.yaml` |

你的定位是清楚的：`config/spec` 定义并验证装配关系；`TuttiRuntime` 将已验证配置实例化为 Resource、Resolver、DataPath，再交给 `StorageRuntime`。配置中的 `BackendSpec` 是装配关系，不对应任何运行期 backend 对象。

**对 config/spec 的判断**

现有 `TuttiRuntimeSpec::validate()` 已覆盖装配前所需的静态约束：

| Spec 提供的信息 | 当前校验 |
| --- | --- |
| Resource/Resolver/DataPath 的 ID 和具体配置 | ID、type、variant 一致性 |
| backend 对三个组件的引用 | 引用必须存在 |
| 三种组件能否组成合法关系 | `SpecContract` 校验类型组合 |
| URI 路由 | resolver scheme 合法且唯一 |
| Resource 分片数量 | local 单分片、striped 至少双分片 |
| Resource 与 DataPath 关系 | Resource 不被不同 DataPath 消费 |
| 声明可达性 | 所有组件均被 backend 关系引用 |
| 当前产品规模 | 恰好一个 backend 关系 |
| 组合参数 | `stripe_unit` 等关系属性得到校验 |

因此，`TuttiRuntime` 在 `spec.validate()` 成功后，可以直接将 spec 当作静态有效的装配图，不再维护另一套 runtime contract 表。

`BackendSpec` 中的 `stripe_unit` 也可以保留。按当前语义，它描述 striped resolver 和 striped DataPath 之间共同遵循的布局关系，作为“边的配置”是成立的。

**各模块需要提供的能力**

| 模块 | 提供给 TuttiRuntime 的内容 |
| --- | --- |
| `config/spec` | 已验证的组件声明、ID 引用、类型组合和关系配置 |
| Resource | 按 `ResourceSpec` 创建实例、生命周期管理、供 resolver/DataPath 创建使用的类型化资源能力 |
| Resolver | 按 `ResolverSpec + 关系上下文` 创建 `StorageTargetResolver` |
| DataPath | 按 `DataPathSpec + 关系上下文` 创建 `DataPath` 和 `DataPathConfig` |
| StorageRuntime | 接受完成组装的 `RuntimeComponents`，初始化 DataPath 并执行 I/O |
| TuttiRuntime | 建索引、创建组件、建立路由、管理所有权、组装 StorageRuntime、控制关闭顺序 |

## Resource 提供什么

Resource 继续统一提供生命周期：

```cpp
class Resource {
public:
    virtual Status initialize() = 0;
    virtual Status shutdown() = 0;
    virtual ResourceInfo info() const = 0;
};
```

同时，各 Resource 类型向对应组件 factory 提供窄的类型化能力。例如 NVMe：

```cpp
class NvmeResolverResourceProvider {
public:
    virtual Result<NvmeResolverResourceView> resolver_view() const = 0;
};

class NvmeDataPathResourceProvider {
public:
    virtual Result<NvmeDataPathResourceView> datapath_view() const = 0;
};
```

`NvmeResource` 实现这两个接口。具体 resolver/DataPath factory 查询自己需要的接口，`TuttiRuntime` 只负责把 `Resource&` 传给它们。

Memory Resource 则应提供内存容量或分配能力，供 memfs resolver/DataPath 使用。这样 `MemoryResourceConfig::capacity_bytes` 会对应真实运行期资源，而不是仅用于配置校验。

资源创建入口按 `ResourceSpec` 组织：

```cpp
Result<std::unique_ptr<Resource>> create_resource(
    const config::ResourceSpec& spec,
    const ResourceCreateContext& context);
```

它内部可以使用 `std::visit(spec.config)` 分派 NVMe 和 Memory Resource。

## Resolver 提供什么

Resolver 模块提供独立创建入口：

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

各实现使用的信息如下：

| Resolver 类型 | 创建输入 |
| --- | --- |
| `local-file` | `ResolverSpec`、NVMe resolver view、关联 DataPath key |
| `striped-file` | N 个 NVMe resolver slices、关系中的 `stripe_unit`、关联 DataPath key |
| `memfs` | Memory Resource 能力、关联 DataPath key |

关键点是 resolver 必须把关系中引用的 DataPath 实例 ID 写入：

```cpp
ResolvedTarget::recommended_data_path_key()
```

因此 binding 的 `make_resolved_target()` 需要接受运行期 `data_path_key`，而不是始终写死 `"local-nvme"`、`"striped-local-nvme"` 或 `"memfs"`。

## DataPath 提供什么

DataPath 模块同样提供独立创建入口：

```cpp
struct DataPathCreateContext {
    const Resource& resource;
    const config::BackendSpec& relation;
    std::int32_t runtime_accel_id;
};

struct CreatedDataPath {
    std::unique_ptr<DataPath> instance;
    DataPathConfig initialize_config;
};

Result<CreatedDataPath> create_data_path(
    const config::DataPathSpec& spec,
    const DataPathCreateContext& context);
```

| DataPath 类型 | 创建输入 |
| --- | --- |
| `local-nvme` | spec 中的 tuning、单个 NVMe DataPath slice、accelerator ID |
| `striped-local-nvme` | spec 中的 tuning、N 个 slices、accelerator ID |
| `memfs` | Memory Resource 能力和 memfs 配置 |

Resource slice 数量、namespace 一致性、MDTS、accelerator 等运行期事实，由对应组件 factory 在构造时校验。这些校验属于具体组件的创建过程。

DataPath 注册到 `StorageRuntime` 时使用实例 ID：

```cpp
DataPathBinding{
    .key = datapath_spec.id,
    .data_path = instance,
    .config = initialize_config,
};
```

于是配置关系得到真正落实：

```
BackendSpec.datapath
    == Resolver 生成的 recommended_data_path_key
    == RuntimeComponents 中的 DataPathBinding.key
```

## TuttiRuntime 如何组织

建议将创建过程明确分成六个阶段：

```
1. parse config
2. spec.validate()
3. 建立 ResourceSpec / ResolverSpec / DataPathSpec 的 ID 索引
4. 创建并初始化 Resource
5. 根据 BackendSpec 关系分别创建 Resolver 和 DataPath
6. 生成 RuntimeComponents 并创建 StorageRuntime
```

内部装配关系只需一个临时视图：

```cpp
struct AssemblyRelation {
    const config::BackendSpec& relation;
    const config::ResourceSpec& resource_spec;
    const config::ResolverSpec& resolver_spec;
    const config::DataPathSpec& datapath_spec;
    Resource& resource;
};
```

它只在 `create()` 过程中存在，不成为运行期对象。

核心伪代码：

```cpp
spec.validate();

create_and_initialize_resources(spec.storage.resources);

for (const BackendSpec& relation : spec.storage.backends) {
    AssemblyRelation assembly = resolve_relation(relation);

    auto resolver = create_resolver(
        assembly.resolver_spec,
        {assembly.resource, relation, assembly.datapath_spec.id});

    auto datapath = create_data_path(
        assembly.datapath_spec,
        {assembly.resource, relation, spec.runtime.accel_id, cache});

    own_resolver(assembly.resolver_spec.id, std::move(resolver));
    own_datapath(assembly.datapath_spec.id, std::move(datapath.instance));

    components.resolvers.push_back({
        assembly.resolver_spec.scheme,
        resolver_pointer,
    });
    components.data_paths.push_back({
        assembly.datapath_spec.id,
        datapath_pointer,
        datapath.initialize_config,
    });
}

storage_runtime = StorageRuntime::create(runtime_config, components);
```

## TuttiRuntime 的目标内部状态

```cpp
std::unique_ptr<StorageRuntime> runtime_;

std::unordered_map<std::string, std::unique_ptr<Resource>> resources_;
std::unordered_map<std::string,
                   std::unique_ptr<StorageTargetResolver>> resolvers_;
std::unordered_map<std::string, std::unique_ptr<DataPath>> datapaths_;

std::vector<std::string> resource_initialization_order_;
std::vector<std::string> resolver_creation_order_;
std::vector<std::string> datapath_creation_order_;
```

运行期状态由实际拥有的组件组成。配置中的 `BackendSpec` 在创建阶段完成关系解析后即可结束其使命；如果需要诊断，可以保留完整的 validated spec，而不是创建 `BackendInstance`。

**当前代码的迁移对应关系**

| 当前实现 | 新实现归属 |
| --- | --- |
| `default_resource_factory()` | `create_resource()` |
| `create_local_backend()` 中 resolver 部分 | `create_resolver(local-file)` |
| `create_local_backend()` 中 DataPath 部分 | `create_data_path(local-nvme)` |
| `create_striped_backend()` 中 resolver 部分 | `create_resolver(striped-file)` |
| `create_striped_backend()` 中 DataPath 部分 | `create_data_path(striped-local-nvme)` |
| `BackendFactoryContext` | 创建期 `AssemblyRelation` 和两个组件 context |
| `BackendFactoryProduct` | resolver 与 DataPath 的独立创建结果 |
| `RuntimeBackendRegistration` | 由现有 spec contract 校验和各组件 config variant 分派替代 |
| `BackendShardProjection` | 留在 NVMe 组件 factory 内部 |
| `BackendInstance/backends_` | validated spec 加三个实际组件 registry |
| `BackendManifest` | validated spec 中的关系信息 |

最终边界是：`config/spec` 决定连接是否合法，Resource 提供创建材料，Resolver 和 DataPath 各自负责实例化，`TuttiRuntime` 将它们连接并持有，`StorageRuntime` 只消费已经完成的 `RuntimeComponents`。

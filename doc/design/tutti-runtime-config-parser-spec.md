# TuttiRuntime config parser、spec 与创建流程设计

> **状态**：目标设计，尚未实现
>
> 本文定义应用配置从 YAML 到 `TuttiRuntime` 的目标边界、目录结构、校验职责和
> 创建流程。本文聚焦 `tutti/config` 与 `tutti/tutti_runtime` 的拆分，不改变
> `StorageRuntime` 的公共路由和 I/O 契约，也不改变当前环境变量覆盖的插入位置与
> 优先级。

## 1. 决策摘要

配置到 Runtime 的主流程固定为：

```text
YAML file
  -> config parser
  -> TuttiRuntimeSpec（字段默认值已填充）
  -> TuttiRuntimeSpec::validate()
  -> validated TuttiRuntimeSpec
  -> TuttiRuntime::create() runtime preflight
  -> Resource / Resolver / DataPath / Backend assembly
  -> StorageRuntime::create()
  -> RUNNING TuttiRuntime
```

核心决策如下：

| 主题 | 决策 |
| --- | --- |
| config 的唯一职责 | 将 YAML 忠实翻译成不依赖 YAML 的 `TuttiRuntimeSpec`，并完成 YAML 语法和结构检查 |
| spec 的职责 | 定义字段和默认值，校验单字段领域规则、跨字段关系和完整配置拓扑 |
| Runtime 的职责 | 调用 parser 和 spec validation，校验运行环境，获取资源，创建组件并管理生命周期 |
| Spec 名称 | 统一使用 `TuttiRuntimeSpec`，明确它由 `TuttiRuntime` 消费，不与 `StorageRuntime::RuntimeConfig` 混淆 |
| TuttiRuntime 创建入口 | 使用与 `StorageRuntime::create()` 一致的静态 factory 风格，不允许公开构造半初始化 Runtime |
| Backend 数量 | 当前 `TuttiRuntimeSpec` 只允许一个 backend；该限制属于 spec 语义，不属于 parser 语法 |
| Resource 消费关系 | 同一 Resource 不能由不同 DataPath 消费；该限制由 spec 统一校验 |
| Backend-specific 配置 | 每个 backend/resource/resolver/datapath 的开发者负责增加对应 spec 类型、parser 和 spec validator |
| Spec 调试输出 | spec 提供确定性的 debug string；成功校验后打印实际进入 Runtime builder 的 spec |
| 环境变量 | 本阶段不移动、不删除、不改变现有环境变量插入点和优先级；它是 effective spec 之外的临时 overlay |
| 迁移策略 | 直接进行破坏性修改，不保留旧 API、旧 namespace、旧 target、兼容 wrapper 或类型 alias |
| 命名约束 | 新实现和新测试禁止使用 `canonical` 一类表示“另一套配置形态”的命名；系统中只有 `TuttiRuntimeSpec` 这一套配置模型 |
| 测试迁移 | 删除因旧 loader、类型和 namespace 契约失效的测试，在 `tests/config` 下按 parser、spec、tutti_runtime 重建测试 |

## 2. 目标与非目标

### 2.1 目标

| 目标 | 验收含义 |
| --- | --- |
| config host-only | parser 和 spec 不依赖 CUDA/MUSA/MACA、GRPC、Resource 实现或具体 DataPath/Resolver 类 |
| YAML 隔离 | `YAML::Node` 和 yaml-cpp 只出现在 `tutti/config/parser` 的私有实现中 |
| Spec 自校验 | 程序化构造的 `TuttiRuntimeSpec` 不经过 YAML，也能执行与配置拓扑一致的完整校验 |
| 单一创建入口 | 返回给调用方的 `TuttiRuntime` 必须已完成 Resource、Backend 和 `StorageRuntime` 初始化 |
| 事务创建 | 任一步初始化失败都按依赖关系逆序回滚，不返回半初始化对象 |
| 可调试 | Runtime 创建前可以输出包含默认值和 backend-specific 字段的规范化 spec |
| 可扩展 | 新增 backend/resource/resolver/datapath 时，特定逻辑落在对应私有目录，不扩张公共 parser/spec 实现 |

### 2.2 非目标

| 非目标 | 本阶段处理 |
| --- | --- |
| 重构环境变量和 programmatic override 的优先级 | 不处理，保持当前行为和插入点 |
| 支持一个 `TuttiRuntime` 的多个 backend | 不处理，spec 继续要求恰好一个 backend |
| 修改 `StorageRuntime` 的多 scheme、多 DataPath key 能力 | 不修改 |
| 修改 daemon 配置或 RPC | 不修改 |
| 把运行时分配结果写回静态 spec | 不执行；allocation ID、设备路径、BDF 和 lease 不属于配置 spec |
| 在 spec 中保存 YAML 源节点或 YAML 文本 | 禁止 |
| 保持旧 config loader API 或测试兼容 | 不处理；旧入口、旧 target 和依赖旧边界的测试直接删除 |

### 2.3 破坏性修改原则

本设计不设置兼容迁移期。实现完成时，源码树、公共 API、构建 target 和测试只表达新
边界。

| 项目 | 决策 |
| --- | --- |
| 旧 `load_tutti_config()` | 直接删除，由 `TuttiRuntime::create()` 取代 |
| 旧 `ParsedConfig` 和相关 config aggregate 名称 | 直接删除，由 `TuttiRuntimeSpec` 取代 |
| `config::TuttiRuntime` | 直接删除，类型只存在于 `tutti::TuttiRuntime` |
| `TuttiRuntimeAssemblyAccess` | 创建逻辑进入 `TuttiRuntime::create()` 后直接删除 |
| 旧 CMake target | 删除并更新所有内部消费者，不提供转发 target 或 alias target |
| 旧 include path | 删除并更新调用方，不保留 forwarding header |
| 旧测试 | 依赖旧 API、旧 namespace、旧 loader 职责或旧错误层次的测试目录和 target 直接删除 |
| 仍然有效的业务场景 | 在新测试目录中按新边界重新编写；不以保留旧测试源码为目标 |

`canonical` 及同类名称通常暗示旧、新两套配置模型并存，因此不得出现在本次新增的
类型、函数、文件、target、测试名称或错误消息中。描述配置时直接使用
`TuttiRuntimeSpec`、`StorageSpec` 或具体的 Resource/Resolver/DataPath/Backend spec
名称。

## 3. 分层职责

### 3.1 总体边界

| 层次 | 输入 | 输出 | 允许的行为 | 禁止的行为 |
| --- | --- | --- | --- | --- |
| Parser | YAML 文件 | `TuttiRuntimeSpec` | 文件读取、节点遍历、类型转换、required/unknown field 检查、结构和表示范围检查 | Runtime 拓扑决策、硬件探测、factory lookup、对象创建 |
| Spec | `TuttiRuntimeSpec` | `Status`、debug string | 默认值、领域校验、引用和拓扑校验、backend-specific 语义校验 | YAML 访问、环境变量读取、RPC、CUDA、具体组件创建 |
| TuttiRuntime create | path 或 spec | owned `TuttiRuntime` | 调用 parser/validator、运行环境 preflight、资源和组件创建、事务回滚 | 重新实现 YAML 语法规则或复制一份 spec validator |
| StorageRuntime create | `RuntimeConfig` + borrowed components | owned `StorageRuntime` | 校验核心 Runtime config、注册路由、初始化 DataPath | 拥有 `TuttiRuntime` Resource 或 concrete backend 配置 |

### 3.2 Parser 与 Spec 校验的分界

Parser 可以尽早返回带 YAML 路径的错误，但 `TuttiRuntimeSpec::validate()` 是配置语义的
最终权威。程序化构造的 spec 必须得到相同的语义结果。

| 检查项 | Parser | Spec validator | 说明 |
| --- | :---: | :---: | --- |
| 根节点是 mapping | 是 | 否 | YAML 表示规则 |
| `resources` 是 sequence | 是 | 否 | YAML 表示规则 |
| 标量能转换为 `uint64_t` | 是 | 否 | YAML 类型和 C++ 表示范围 |
| required field 是否出现 | 是 | 可防御性检查 | Parser 提供准确的源路径错误，spec 防止程序化构造绕过 |
| unknown YAML field | 是 | 否 | 只有 parser 知道原始字段集合 |
| enum 字符串能映射到 spec enum | 是 | 否 | 无法翻译的字符串是 parse error |
| ID 非空和长度合法 | 可提前检查 | 是 | 属于 spec 字段契约，spec 必须独立保证 |
| ID 唯一、引用存在 | 否 | 是 | 完整对象图语义 |
| 恰好一个 backend | 否 | 是 | 当前产品 Runtime 拓扑规则 |
| 所有声明都可从 backend 到达 | 否 | 是 | 完整对象图语义 |
| Resource 不能被不同 DataPath 消费 | 否 | 是 | Runtime 所有权拓扑规则 |
| resolver scheme 唯一 | 否 | 是 | Runtime 路由约束 |
| backend 的类型组合匹配 contract | 否 | 是 | Spec contract 语义 |
| striped `stripe_unit` 对齐 | 可检查表示类型 | 是 | Backend-specific 领域语义 |
| contract 导出的 DataPath key 是否冲突 | 否 | 否 | Key 属于 binding ABI，由 Runtime registration 检查 |
| backend factory 是否编译可用 | 否 | 否 | Runtime backend registry 负责 |
| accelerator device 是否存在 | 否 | 否 | Runtime platform preflight 负责 |

## 4. 目标目录结构

### 4.1 实现目录

`tutti/tutti_runtime` 保持当前目录。`tutti/config` 收敛为 `parser` 和 `spec` 两个
实现子目录：

```text
tutti/
  config/
    CMakeLists.txt
    tutti_runtime_spec.h
    parser/
      tutti_runtime_config_parser.cpp
      parser_internal.h
      resource/
        memory_config_parser.cpp
        nvme_config_parser.cpp
        nvme_parser_internal.h
      resolver/
        local_file_config_parser.cpp
        memfs_config_parser.cpp
        striped_file_config_parser.cpp
      datapath/
        local_nvme_config_parser.cpp
        memfs_config_parser.cpp
        striped_local_nvme_config_parser.cpp
        nvme_parser_internal.h
      backend/
        ext4_local_nvme_config_parser.cpp
        memfs_config_parser.cpp
        striped_local_nvme_config_parser.cpp
    spec/
      tutti_runtime_spec.cpp
      spec_internal.h
      resource/
        resource_spec.h
        memory_spec.h
        nvme_spec.h
        memory_spec.cpp
        nvme_spec.cpp
      resolver/
        resolver_spec.h
        local_file_spec.h
        memfs_spec.h
        striped_file_spec.h
        local_file_spec.cpp
        memfs_spec.cpp
        striped_file_spec.cpp
      datapath/
        datapath_spec.h
        nvme_datapath_spec.h
        local_nvme_spec.h
        memfs_spec.h
        striped_local_nvme_spec.h
        local_nvme_spec.cpp
        memfs_spec.cpp
        striped_local_nvme_spec.cpp
      backend/
        backend_spec.h
        ext4_local_nvme_spec.h
        memfs_spec.h
        striped_local_nvme_spec.h
        ext4_local_nvme_spec.cpp
        memfs_spec.cpp
        striped_local_nvme_spec.cpp
  tutti_runtime/
    tutti_runtime.cpp
    tutti_runtime_create.cpp
    tutti_runtime_internal.h
    backend_factory.h
    backend_factory.cpp
```

目录所有权如下：

| 路径 | 公有/公共逻辑 | 特定私有逻辑 |
| --- | --- | --- |
| `config/parser` | YAML 加载、根节点分发、公共读值和错误上下文 helper | 各 resource/resolver/datapath/backend 的 YAML 字段映射 |
| `config/spec` | 公开 value header 定义各类 Spec 的字段和默认值；`TuttiRuntimeSpec::validate()` 调度、ID index、引用和拓扑校验、debug formatter | 各 resource/resolver/datapath/backend 的领域校验实现 |
| `tutti_runtime` | `TuttiRuntime::create()`、ownership registry、shutdown 和 rollback | concrete backend factory、Resource factory 和运行时 contract registration |

### 4.2 公共头文件

公开 Spec value header 与 config 实现共置于 `tutti/config`；对外 include
名和安装布局仍保持 `<tutti/config/...>`：

```text
tutti/config/
  tutti_runtime_spec.h
  spec/
    resource/*.h
    resolver/*.h
    datapath/*.h
    backend/*.h

tutti/include/tutti/config/
  tutti_runtime_config_parser.h

tutti/include/tutti/
  tutti_runtime.h
```

| 头文件 | 公开内容 | 不得公开的内容 |
| --- | --- | --- |
| `tutti_runtime_config_parser.h` | `parse_tutti_runtime_config(path)` | `YAML::Node`、parser helper、backend parser function |
| `tutti/config/tutti_runtime_spec.h` | `TuttiRuntimeSpec`、顶层 Spec、`validate()`、`to_debug_string()` | yaml-cpp、concrete Runtime component、allocation metadata |
| `tutti/config/spec/<kind>/*.h` | 各 resource/resolver/datapath/backend 的 aggregate 与 specific value type、枚举、字段默认值 | parser helper、validator 实现、运行时分配结果 |
| `tutti_runtime.h` | `TuttiRuntime::create()`、查询和 shutdown API | parser internal、factory product internal、test hook |

Backend-specific spec 类型是 Runtime factory 需要读取的稳定 value contract，因此
放在 `tutti/config/spec/backend` 的公开 value header 中，并由
`tutti_runtime_spec.h` 间接包含。Backend-specific 的解析和校验函数仍是
`config/parser/*`、`config/spec/*` 下的私有实现。Build tree 仅镜像上述公开头，
安装后它们位于 `include/tutti/config/...`。

### 4.3 命名空间

| 类型或函数 | 目标命名空间 |
| --- | --- |
| `TuttiRuntimeSpec` | `tutti::config` |
| `parse_tutti_runtime_config()` | `tutti::config` |
| parser/spec internal helper | `tutti::config::detail` |
| `TuttiRuntime` | `tutti` |
| Runtime backend factory/registration | `tutti::tutti_runtime` |
| Runtime test access | `tutti::testing` |

## 5. `TuttiRuntimeSpec` 模型

### 5.1 顶层模型

```cpp
namespace tutti::config {

struct TuttiRuntimeSpec {
    AcceleratorSpec accelerator;
    RuntimeSpec runtime;
    StorageSpec storage;

    Status validate() const;
    Result<std::string> to_debug_string() const;
};

} // namespace tutti::config
```

| 顶层字段 | 含义 | 默认值归属 |
| --- | --- | --- |
| `accelerator` | 编译期 accelerator profile 的声明 | `AcceleratorSpec` 字段初始化器 |
| `runtime` | 本次 `TuttiRuntime` 的身份和基础参数 | `RuntimeSpec` 字段初始化器 |
| `storage` | Resource、Resolver、DataPath 和 Backend 声明图 | 各具体 spec 类型 |

默认值必须是配置契约的一部分，并满足以下约束：

| 规则 | 决策 |
| --- | --- |
| 无 YAML 依赖 | 默认值由 C++ spec 类型定义，不从 parser helper 产生 |
| 单一来源 | parser、validator 和 Runtime factory 不分别复制默认常量 |
| 可见性 | `to_debug_string()` 必须打印已填充的默认值 |
| 可校验 | 默认值与显式 YAML 值走相同 validator |
| 运行环境值 | 设备数量、allocation metadata 等不是 spec 默认值 |

### 5.2 Backend-specific 配置

通用 `BackendSpec` 不再平铺只属于某个 backend 的字段。目标模型使用 typed config：

```cpp
struct Ext4LocalNvmeBackendConfig {};

struct StripedLocalNvmeBackendConfig {
    std::uint64_t stripe_unit = kDefaultStripeUnit;
};

struct MemfsBackendConfig {};

using BackendConfig = std::variant<
    Ext4LocalNvmeBackendConfig,
    StripedLocalNvmeBackendConfig,
    MemfsBackendConfig>;

struct BackendSpec {
    std::string id;
    std::string contract;
    std::string resolver;
    std::string datapath;
    std::string resource;
    BackendConfig config;
};
```

| 场景 | Parser 行为 | Spec 行为 | Runtime 行为 |
| --- | --- | --- | --- |
| striped YAML 含合法 `stripe_unit` | 填入 `StripedLocalNvmeBackendConfig` | 校验非零和对齐 | factory 消费最终值 |
| striped YAML 未写 `stripe_unit` | 保留 spec 默认值 | 校验默认值 | factory 消费默认值 |
| ext4 YAML 写入 `stripe_unit` | unknown field，parse 失败 | 不执行 | 不执行 |
| contract 与 variant 类型不一致 | 完成可表示字段翻译 | validation 失败 | 不执行 |
| spec 合法但 factory 未编译 | parse 成功 | validation 成功 | 返回 `UNSUPPORTED` |

Resource、Resolver 和 DataPath 使用相同原则：公共声明保存 ID、type 和引用关系，只有
某个实现需要的字段放入对应 typed config。新增实现时，不得向所有类型共享的 struct
平铺一个只被单个实现使用的字段。

### 5.3 Spec contract 与 Runtime registration

当前同时包含静态配置规则和运行时 ABI 信息的 contract registry 拆为两层：

| 层次 | 典型字段 | 所属目录 | 失败类型 |
| --- | --- | --- | --- |
| Spec contract | contract name、Resource/Resolver/DataPath type 组合、cardinality、specific validator | `config/spec/backend` | `INVALID_ARGUMENT` |
| Runtime registration | factory、resolver/payload type ID、payload API version、DataPath key、编译实现状态 | `tutti_runtime` | `UNSUPPORTED` 或运行时错误 |

已知、可表达且语义合法的 config 不应因为当前二进制缺少 factory 而 parse 或 spec
validation 失败。实现可用性只由 Runtime registration 判断。

## 6. Spec validation

### 6.1 校验阶段

`TuttiRuntimeSpec::validate()` 按固定顺序执行，确保错误稳定且易于定位：

| 顺序 | 阶段 | 主要检查 |
| ---: | --- | --- |
| 1 | 顶层字段 | profile/runtime 字段的领域范围、必需顶层结构 |
| 2 | 单对象校验 | 每个 Resource/Resolver/DataPath/Backend 的 ID、type 和 specific config |
| 3 | ID index | 各声明组内部 ID 唯一 |
| 4 | 引用解析 | backend 引用的 resource/resolver/datapath 全部存在 |
| 5 | Spec contract | 类型组合、selection/cardinality、backend-specific 组合 |
| 6 | 路由拓扑 | resolver scheme 等 spec 显式声明的 Runtime 路由身份无冲突 |
| 7 | 所有权拓扑 | Resource 不被不同 DataPath 消费 |
| 8 | 可达性 | 所有声明均由 backend 引用 |
| 9 | 产品约束 | backend 数量恰好为 1 |

### 6.2 校验结果与索引

| 决策 | 说明 |
| --- | --- |
| `validate()` 不修改 spec | 默认值在构造时已经存在，校验保持 const 和可重复 |
| `validate()` 不缓存裸指针 index | 避免 spec move 后产生悬空指针 |
| Runtime 可以在校验后建立临时 lookup table | lookup 只用于创建过程，不重复实现有效性规则 |
| Runtime 保留底层防御性检查 | `StorageRuntime` 和 registry 仍拒绝重复 route/null component，但错误不应由正常 spec 触发 |
| 错误包含逻辑路径 | 例如 `storage.backends[0].resource`，不依赖 YAML node 生命周期 |

## 7. Spec 调试输出

### 7.1 API 与调用时机

Spec 提供不依赖日志框架和 YAML 的确定性格式化函数：

```cpp
Result<std::string> TuttiRuntimeSpec::to_debug_string() const;
```

| 设计项 | 决策 |
| --- | --- |
| 校验要求 | `to_debug_string()` 只对有效 spec 返回字符串；内部可调用 `validate()`，无效时返回相同语义错误 |
| Runtime 调用时机 | `TuttiRuntime::create(spec)` 完成显式 `validate()` 后、产生 Resource/RPC 副作用前 |
| 日志职责 | Spec 只返回字符串，不直接写 stdout/stderr；Runtime 或应用 logger 决定是否输出 |
| 输出内容 | 所有顶层字段、默认值、数组顺序、ID 引用和 backend-specific config |
| 输出顺序 | 固定为 accelerator、runtime、resources、resolvers、datapaths、backends；数组保持 spec 顺序 |
| 输出格式 | 稳定的 key-path 文本或等价的规范化文本，不使用 yaml-cpp emitter |
| 敏感信息 | credential/token 等字段必须 redact；普通 endpoint 可按当前契约输出 |
| 运行时事实 | allocation ID、slice path、BDF、lease 和 factory pointer 不进入 spec 输出 |

示例输出：

```text
accelerator.profile = "CUDA"
runtime.accel_id = 0
storage.resources[0].id = "nvme-local-0"
storage.resources[0].type = "nvme"
storage.resources[0].allocation.selection = "explicit"
storage.resources[0].allocation.device_ids = [0]
storage.resources[0].allocation.queues_per_controller = 4
storage.resolvers[0].id = "file-resolver-0"
storage.resolvers[0].type = "local-file"
storage.resolvers[0].scheme = "file"
storage.datapaths[0].id = "local-nvme-datapath-0"
storage.datapaths[0].type = "local-nvme"
storage.datapaths[0].config.handle_cache_capacity = 0
storage.backends[0].id = "local-backend-0"
storage.backends[0].contract = "ext4-local-nvme"
storage.backends[0].resolver = "file-resolver-0"
storage.backends[0].datapath = "local-nvme-datapath-0"
storage.backends[0].resource = "nvme-local-0"
```

### 7.2 环境变量的阶段性例外

现有 cache 环境变量和 programmatic override 仍在当前 assembly 点位解析。本阶段不把
这些值迁入 `TuttiRuntimeSpec`，也不改变其优先级。

因此，本阶段的诊断输出分为两部分：

| 输出 | 内容 | 所有者 |
| --- | --- | --- |
| Validated spec dump | YAML 显式字段 + spec 默认值，即进入 Runtime builder 的结构化配置 | `TuttiRuntimeSpec::to_debug_string()` |
| Existing effective cache overlay | 当前环境变量和 programmatic override 解析后的 cache 值 | 保持现有 Runtime assembly 点位 |

Runtime 必须先打印 validated spec，再在现有插入点继续处理 effective cache overlay。不得把
overlay 值伪装为 spec 自身字段。后续环境变量设计完成后，可以形成单独的 effective spec
或 effective runtime options，再扩展统一打印；本阶段不提前改变该边界。

## 8. `TuttiRuntime::create()`

### 8.1 API

```cpp
namespace tutti {

class TuttiRuntime {
public:
    static Result<std::unique_ptr<TuttiRuntime>> create(
        const std::string& config_path,
        TuttiRuntimeCreateOptions options = {});

    static Result<std::unique_ptr<TuttiRuntime>> create(
        config::TuttiRuntimeSpec spec,
        TuttiRuntimeCreateOptions options = {});

    Status shutdown();

private:
    TuttiRuntime();
};

} // namespace tutti
```

| 重载 | 用途 | 固定行为 |
| --- | --- | --- |
| `create(path, options)` | YAML 应用入口 | 调用 parser 后转发给 spec 重载 |
| `create(spec, options)` | 程序化配置、测试和嵌入场景 | 总是调用 `spec.validate()`，不能绕过校验 |

### 8.2 创建状态机

| 状态 | 对外可见 | 允许的内容 |
| --- | :---: | --- |
| `INITIALIZING` | 否 | 已创建的部分 Resource/Resolver/DataPath，用于失败回滚 |
| `RUNNING` | 是 | Resource、Backend 和 `StorageRuntime` 全部就绪 |
| `SHUTTING_DOWN` | 是 | shutdown 正在按顺序释放对象 |
| `STOPPED` | 是 | shutdown 已完成 |

公开默认构造函数必须删除或改为 private，避免产生 state 为 `RUNNING` 但
`storage_runtime() == nullptr` 的半初始化对象。

### 8.3 创建顺序

| 顺序 | 操作 | 是否允许外部副作用 | 失败处理 |
| ---: | --- | :---: | --- |
| 1 | parse YAML（path 重载） | 否 | 返回 parse error |
| 2 | `TuttiRuntimeSpec::validate()` | 否 | 返回 spec error |
| 3 | 生成并记录 spec debug string | 否 | formatter 失败则不继续创建 |
| 4 | accelerator/profile Runtime preflight | 只读设备查询 | 不创建 Resource |
| 5 | 保持现有点位解析 effective cache overlay | 读取环境 | 保持当前错误/默认行为 |
| 6 | 创建并 initialize Resource | 是，可能 RPC Acquire | 关闭当前和此前 Resource |
| 7 | backend factory 创建 Resolver/DataPath | 是，构造实现对象 | 销毁部分组件并释放 Resource |
| 8 | 注册 backend manifest 和 owned component | 否 | 逆序销毁 |
| 9 | `StorageRuntime::create()` | 是，初始化 DataPath | `StorageRuntime` 自身回滚，aggregate 继续逆序清理 |
| 10 | commit `TuttiRuntime` 为 `RUNNING` | 否 | 返回完整 owned Runtime |

### 8.4 所有权与关闭顺序

`StorageRuntime` 借用 Resolver/DataPath 指针，`TuttiRuntime` 拥有所有组件。因此关闭
顺序保持：

```text
StorageRuntime::shutdown()
  -> destroy StorageRuntime
  -> destroy Resolver instances
  -> destroy DataPath instances
  -> Resource::shutdown() in reverse initialization order
  -> destroy Resource instances
```

| 对象 | 所有者 | 生命周期约束 |
| --- | --- | --- |
| `StorageRuntime` | `TuttiRuntime` | 必须先于 borrowed components 销毁 |
| Resolver | `TuttiRuntime` | 至少存活到 `StorageRuntime` 销毁完成 |
| DataPath | `TuttiRuntime` | 至少存活到 `StorageRuntime::shutdown()` 完成 |
| Resource | `TuttiRuntime` | 必须晚于依赖它的 component 和 `StorageRuntime` 释放 |
| Backend manifest | `TuttiRuntime` | 只保存诊断身份和 owned object 的非 owning 关联 |

## 9. 构建目标与依赖

目标依赖方向如下：

```text
tutti_runtime_spec
  <- tutti_config_parser (+ yaml-cpp)
  <- tutti_runtime

tutti_runtime
  -> tutti_config_parser       // create(path)
  -> runtime backend/resource implementations
  -> StorageRuntime API
```

| Target | 源文件 | 允许依赖 | 禁止依赖 |
| --- | --- | --- | --- |
| `tutti_runtime_spec` | `config/spec/**` | `tutti_api` value/status headers | yaml-cpp、cuda-like、GRPC、concrete component |
| `tutti_config_parser` | `config/parser/**` | `tutti_runtime_spec`、yaml-cpp | Runtime implementation、Resource factory、DataPath/Resolver concrete type |
| `tutti_runtime` | `tutti_runtime/**` | parser、spec、StorageRuntime、已启用的 backend/resource implementation | YAML node/private parser helper |

构建系统必须提供 header hygiene 或依赖审计，证明 `tutti_runtime_spec` 可以在纯 HOST、无
yaml-cpp include 的翻译单元中单独编译。

### 9.1 测试目录与 CMake 聚合

新的 config 相关测试统一放在 `test`，目录层次与生产代码职责对应：

```text
tests/
  config/
    CMakeLists.txt
    parser/
      CMakeLists.txt
      tutti_runtime_config_parser_test.cpp
    spec/
      CMakeLists.txt
      tutti_runtime_spec_test.cpp
      tutti_runtime_spec_debug_test.cpp
  tutti_runtime/
    CMakeLists.txt
    tutti_runtime_create_test.cpp
    tutti_runtime_lifecycle_test.cpp
    tutti_runtime_hardware_test.cpp
```

| 目录 | 测试对象 | 允许链接的主要 target | 不应覆盖的内容 |
| --- | --- | --- | --- |
| `tests/config/parser` | YAML 到 `TuttiRuntimeSpec` 的翻译 | `tutti_config_parser` | spec 拓扑语义、Runtime factory、硬件 |
| `tests/config/spec` | 默认值、`validate()`、debug string | `tutti_runtime_spec` | YAML node、Resource Acquire、具体 DataPath 初始化 |
| `tests/tutti_runtime` | `TuttiRuntime::create()`、factory、rollback、shutdown、硬件闭环 | `tutti_runtime` 及测试需要的具体实现 | parser 私有 helper、旧 loader API |

CMake 从现有测试聚合位置添加 `tests/config`，`tests/tutti_runtime`。

| CMake 规则 | 决策 |
| --- | --- |
| 旧目录接入 | 从 `tutti/CMakeLists.txt` 删除 config loader、storage config、runtime bundle 和 runtime resource registry 的逐目录 `add_subdirectory()` |
| 新目录接入 | 顶层接入 `tests/config`，`tests/tutti_runtime`，具体测试由其内部 CMake 管理 |
| 测试 target 名称 | 使用 `tutti_config_parser_test`、`tutti_config_spec_test`、`tutti_tutti_runtime_test` 等新名称 |
| 旧 target 名称 | 直接删除，不提供 alias target，不维护旧 CTest 名称 |
| Profile 条件 | HOST parser/spec 始终构建；需要 concrete backend 或硬件的 Runtime target 在 `tests/tutti_runtime/CMakeLists.txt` 内按 profile 条件添加 |
| 测试依赖方向 | parser 测试不能链接 Runtime；spec 测试不能链接 parser；Runtime 测试可以链接 parser/spec 的公开 target |

## 10. 当前文件迁移

| 当前文件/内容 | 目标位置 | 处理 |
| --- | --- | --- |
| `config/tutti_config_parse.cpp` 的 YAML 主流程 | `config/parser/tutti_runtime_config_parser.cpp` | 只保留 parse 和字段填充 |
| `config/storage/*/*_config_parse.cpp` | `config/parser/<kind>/*_config_parser.cpp` | 保持实现类型私有分层 |
| 现有整体 storage 配置校验逻辑 | `config/spec/tutti_runtime_spec.cpp` | 改为 `TuttiRuntimeSpec::validate()` 公共调度，不保留旧函数名称 |
| backend-specific validator | `config/spec/backend/*_spec.cpp` | 与相应 typed config 同步维护 |
| Resource/Resolver/DataPath specific validator | `config/spec/<kind>/*_spec.cpp` | 不再放在 parser 文件中 |
| `config/storage/backend/contract_registry.cpp` | spec contract + Runtime registration 两处 | 拆分静态语义与运行时 ABI/factory |
| `config/backend_factory.*` | `tutti_runtime/backend_factory.*` | 从 config 移出 |
| `config/tutti_config.cpp` 的 assembly | `tutti_runtime/tutti_runtime_create.cpp` | 成为 `TuttiRuntime::create()` |
| `config/tutti_config_internal.h` 的 factory options | `tutti_runtime/tutti_runtime_internal.h` | 测试 dependency injection 跟随 Runtime |
| `resolve_cache_config()` | 暂时保持当前实现和调用点 | 本阶段不改环境变量行为 |
| public `load_tutti_config()` | `TuttiRuntime::create(path)` | 直接删除旧入口，不提供 deprecated wrapper |
| `config::TuttiRuntime` | `tutti::TuttiRuntime` | 直接修改 namespace，不提供 alias |

### 10.1 旧测试处置

旧测试不作为兼容资产迁移。测试源码与新边界冲突时直接删除，再按新模块的公开契约
重写必要场景。

| 当前测试目录或 case | 处置 | 新归属 |
| --- | --- | --- |
| `tests/storage_config_contract` | 删除整个旧目录和 target | YAML 翻译进入 `tests/config/parser`；spec contract 进入 `tests/config/spec` |
| `tests/config_loader` | 删除整个旧目录和 target | parser case、spec case、Runtime create/rollback case 分别进入三个新子目录 |
| `tests/runtime_bundle_loader_contract` | 删除整个旧目录和 target | Runtime owned bundle 和硬件闭环进入 `tests/config/tutti_runtime` |
| `tests/runtime_resource_registry_contract` | 删除整个旧目录和 target | Resource registry/lifecycle 进入 `tests/config/tutti_runtime` |
| 其他共享测试中的 `load_tutti_config()` case | 删除受影响 case | 仅在新 Runtime 测试确有业务价值时重写为 `TuttiRuntime::create()` case |
| 旧 header hygiene include case | 删除旧 include/API 断言 | 按新 parser/spec/runtime 公共头重新增加依赖隔离断言 |

新测试不包含任何兼容性断言，也不保留旧错误消息、旧 namespace、旧测试 fixture 或旧
target 名称。必须保留的是业务不变量，例如 spec 拒绝非法拓扑、创建失败释放
Resource、关闭顺序正确和真实 I/O 数据正确性，而不是旧 loader 的调用形式。

## 11. 新增类型的开发者契约

新增 backend/resource/resolver/datapath 时，必须同时完成对应层次，不能只修改 parser
或只修改 Runtime factory。

| 扩展类型 | Spec value | Parser private | Spec validator private | Runtime factory/registration | 测试 |
| --- | :---: | :---: | :---: | :---: | :---: |
| Resource | 必须 | 必须 | 必须 | 必须 | 必须 |
| Resolver | 必须 | 必须 | 必须 | 必须 | 必须 |
| DataPath | 必须 | 必须 | 必须 | 必须 | 必须 |
| Backend | 必须 | 必须 | 必须 | 必须 | 必须 |

每个扩展应满足：

| 规则 | 要求 |
| --- | --- |
| 私有字段隔离 | 实现特定字段放在 typed config，不加入无关类型共享字段 |
| 默认值单一来源 | 默认值定义在 specific spec value 中 |
| Parser 对称 | 只把 YAML 映射到 spec，不执行 factory 或实现可用性检查 |
| Validator 对称 | 显式值和默认值使用同一 specific validator |
| Runtime 可用性独立 | Spec 合法不代表 factory 已编译；缺失实现返回 `UNSUPPORTED` |
| Debug 完整性 | 新字段必须加入 spec debug formatter 和 golden test |

## 12. 验证矩阵

| 测试层 | 目录 | 依赖 | 覆盖内容 |
| --- | --- | --- | --- |
| Parser syntax | `tests/config/parser` | HOST + yaml-cpp | YAML 节点类型、required/unknown field、标量转换、backend-specific 字段映射 |
| Spec common | `tests/config/spec` | HOST，无 yaml-cpp runtime 使用 | 默认值、ID/引用、exactly-one-backend、可达性、Resource/DataPath 消费关系 |
| Spec specific | `tests/config/spec` | HOST | NVMe selection、striped cardinality、stripe unit、各 typed config 组合 |
| Spec programmatic | `tests/config/spec` | HOST | 手工构造 spec 后 `validate()`，证明校验不依赖 parser |
| Spec debug | `tests/config/spec` | HOST | 默认值可见、输出顺序稳定、specific 字段完整、无效 spec 拒绝输出、敏感字段脱敏 |
| Runtime create fake | `tests/tutti_runtime` | HOST | parse -> validate -> factory -> `StorageRuntime::create()`、失败回滚和 shutdown 顺序 |
| Runtime contract | `tests/tutti_runtime` | HOST/accelerator profile | profile/device preflight、route binding 和 component lifecycle |
| Hardware E2E | `tests/tutti_runtime` | accelerator + NVMe | config path 创建真实 `TuttiRuntime`、I/O、shutdown 和 lease release |
| Build hygiene | `tests/config/parser`、`tests/config/spec` | HOST | spec 不包含 yaml/CUDA/GRPC；parser 不链接 concrete backend |

必须保留的关键负例：

| ID | 输入 | 预期失败层 |
| --- | --- | --- |
| P1 | `storage.resources` 不是 sequence | Parser |
| P2 | striped backend YAML 含错误类型 `stripe_unit` | Parser |
| S1 | 两个 backend | Spec |
| S2 | backend 引用不存在的 Resource | Spec |
| S3 | 同一 Resource 被两个不同 DataPath 消费 | Spec |
| S4 | contract 与 backend typed config 不匹配 | Spec |
| S5 | striped selection 只有一个 device | Spec |
| R1 | spec 合法但 factory 未注册 | Runtime，`UNSUPPORTED` |
| R2 | accel ID 超出设备数量 | Runtime preflight |
| R3 | Resource Acquire 成功后 DataPath 创建失败 | Runtime rollback，并释放 allocation |

## 13. 实施阶段

| 阶段 | 主要工作 | 行为约束 |
| ---: | --- | --- |
| 1 | 建立 `config/parser`、`config/spec` 目录和新 build target，迁移现有 parse 文件 | 不改变合法 YAML 行为，不保留旧 target |
| 2 | 引入 `TuttiRuntimeSpec`，迁移默认值、整体配置校验和 specific validators | Parser 不再执行整体拓扑校验，不保留旧配置类型和名称 |
| 3 | 实现 `to_debug_string()` 和 golden tests | 不引入 yaml-cpp emitter，不读取环境变量 |
| 4 | 在当前 `tutti/tutti_runtime` 中实现两个 `TuttiRuntime::create()` 重载 | 保持现有 Resource、Backend、StorageRuntime 创建和回滚顺序 |
| 5 | 移动 backend factory、Runtime options 和 namespace，删除 AssemblyAccess | `TuttiRuntime` 只能通过 factory 创建 |
| 6 | 删除旧测试目录和 CMake 接入，在 `tests/config` 下建立 parser/spec/tutti_runtime 测试 | 直接删除 `load_tutti_config()`；保持环境变量插入点和优先级不变 |
| 后续 | 单独设计 effective config/environment overlay | 不与本次 parser/spec 拆分混合 |

## 14. 与现有设计文档的关系

| 文档 | 关系 |
| --- | --- |
| `storage-config-backend-resource.md` | 保留其 storage schema、Resource/Backend 语义和生命周期设计；本文取代其中由 config loader 承担 Runtime assembly 的旧边界描述 |
| `multi-accelerator-runtime.md` | 保留一个 Runtime 对应一个 `accel_id`、资源归属和设备校验语义；其中 loader/factory 表述按本文解释为 `TuttiRuntime::create()` |
| `backend-spi.md` | 不修改 Resolver/DataPath/Binding SPI；本文只定义这些实现如何从 validated spec 被创建 |

若现有文档中的 `config loader` 与本文冲突，以本文为准：config parser 只翻译 YAML，
spec 负责静态配置语义，`TuttiRuntime::create()` 负责运行时装配和生命周期。

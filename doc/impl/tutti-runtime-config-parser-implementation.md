# TuttiRuntime Config Parser / Spec / Runtime 重构实现总结

## 1. 实现范围与结论

本文记录
[`tutti-runtime-config-parser-spec.md`](../design/tutti-runtime-config-parser-spec.md)
所定义目标设计的当前实现结果。对应实现提交为：

```text
1f8501a refactor config parser and runtime assembly
```

本次重构已经完成设计要求的破坏性迁移：应用 YAML 只被 parser 翻译为
`TuttiRuntimeSpec`；配置语义由 spec 独立校验；运行环境预检、Resource 获取、backend
装配、`StorageRuntime` 创建和生命周期统一由 `tutti::TuttiRuntime::create()` 管理。旧
loader、旧配置 aggregate、旧 namespace、旧 target 和兼容 wrapper 均已删除。

| 设计目标 | 当前实现 | 状态 |
| --- | --- | --- |
| Config host-only | parser/spec 不包含 CUDA、MUSA、MACA、gRPC 或具体组件依赖 | 完成 |
| YAML 隔离 | `YAML::Node` 只出现在 `tutti/config/parser` 私有实现 | 完成 |
| Spec 自校验 | 程序化构造的 `TuttiRuntimeSpec` 可独立执行完整语义校验 | 完成 |
| 单一创建入口 | 公开入口收敛为 `TuttiRuntime::create(path/spec)` | 完成 |
| 事务创建 | Resource、component、StorageRuntime 失败均由 owned aggregate 逆序清理 | 完成 |
| 可调试 | 有效 spec 可生成确定、包含默认值的 debug string | 完成 |
| 静态 contract 与运行时注册拆分 | spec contract 在 `config/spec`；ABI/factory 注册在 `tutti_runtime` | 完成 |
| 旧 API 直接删除 | 不保留 loader、类型、namespace、header、target alias | 完成 |
| 新测试分层 | parser、spec、Runtime 和 gated hardware E2E 分目录覆盖 | 完成 |

## 2. 实现前后流程对比

### 2.1 端到端主流程

| 阶段 | 重构前 | 重构后 | 变化效果 |
| ---: | --- | --- | --- |
| 1. 应用入口 | 调用 `load_tutti_config(path)` | 调用 `TuttiRuntime::create(path)` | 创建入口归属 Runtime，不再由 config 层装配对象 |
| 2. YAML 解析 | `parse_tutti_config()` 同时生成 `ParsedConfig` 和 storage aggregate | `parse_tutti_runtime_config()` 只生成 `TuttiRuntimeSpec` | Parser 只负责忠实翻译 YAML |
| 3. 默认值 | parser、旧配置结构和装配代码间分散 | specific spec value 的字段初始化器是唯一来源 | YAML 与程序化构造共享相同默认值 |
| 4. 配置校验 | parser 内同时做结构、引用、contract 和拓扑校验 | parser 检查 YAML 表示；`TuttiRuntimeSpec::validate()` 检查全部语义 | YAML 语法与产品配置语义明确分层 |
| 5. 中间模型 | `ParsedConfig` 内含 `CanonicalStorageConfig` 等旧名称 | 系统中只有 `TuttiRuntimeSpec` | 不再存在平行配置形态 |
| 6. 调试输出 | 没有稳定的 validated spec formatter | `to_debug_string()` 输出固定顺序和已填充默认值 | Runtime 副作用前即可记录真实 builder 输入 |
| 7. 环境预检 | loader 内校验 profile/device | `TuttiRuntime::create(spec)` 在 Resource 前预检 | 运行环境规则归属 Runtime |
| 8. Cache overlay | loader assembly 中解析 | 仍在 Runtime assembly 的原点位解析 | 优先级与行为保持不变 |
| 9. Resource 获取 | config loader 创建并注册 Resource | Runtime 创建、initialize 并取得 ownership | 失败由同一 aggregate 回滚 |
| 10. Backend 创建 | factory 位于 `tutti/config` | factory/ABI registration 位于 `tutti/tutti_runtime` | config 层不再依赖具体 Runtime 组件 |
| 11. StorageRuntime | loader 构造并塞入可半初始化的 aggregate | `StorageRuntime::create()` 成功后才 commit `RUNNING` | 公开调用方无法得到半初始化 Runtime |
| 12. 关闭 | aggregate 通过 public/内部 assembly seam 管理对象 | `TuttiRuntime::shutdown()` 固定逆序释放 | ownership 和回滚顺序由一个类维护 |

重构前的实际职责混合流程可以概括为：

```text
YAML
  -> parse_tutti_config()
  -> ParsedConfig + CanonicalStorageConfig
  -> parser 内拓扑/contract 校验
  -> load_tutti_config()
  -> config 层创建 Resource、Resolver、DataPath、Backend
  -> 通过 assembly access 拼装 TuttiRuntime
  -> StorageRuntime
```

重构后的流程为：

```text
YAML file
  -> parse_tutti_runtime_config()
  -> TuttiRuntimeSpec（默认值已填充）
  -> TuttiRuntimeSpec::validate()
  -> TuttiRuntimeSpec::to_debug_string()
  -> TuttiRuntime runtime preflight
  -> effective cache overlay
  -> Resource initialize
  -> Resolver / DataPath / Backend assembly
  -> StorageRuntime::create()
  -> RUNNING TuttiRuntime
```

### 2.2 失败路径对比

| 失败场景 | 重构前失败位置 | 重构后失败位置 | 副作用边界 |
| --- | --- | --- | --- |
| YAML 根节点/字段类型错误 | parser | parser | 无 Resource/RPC 副作用 |
| unknown YAML field | parser | parser | 无 Resource/RPC 副作用 |
| duplicate ID、缺失引用 | parser 的整体校验 | spec validator 第 3/4 阶段 | 无 Resource/RPC 副作用 |
| contract/type 组合错误 | parser/contract registry | spec validator 第 5 阶段 | 无 Resource/RPC 副作用 |
| 多 backend 或不可达声明 | parser 整体验证 | spec validator 第 8/9 阶段 | 无 Resource/RPC 副作用 |
| profile 与当前 binary 不匹配 | loader | Runtime preflight | Resource 创建前失败 |
| accelerator ID 越界 | loader/runtime 混合路径 | Runtime preflight，返回 `NOT_FOUND` | Resource 创建前失败 |
| 已知 contract 但 factory 未编译 | 可能与 config contract 混合 | Runtime registry，返回 `UNSUPPORTED` | 已获取 Resource 由 aggregate 释放 |
| Resource initialize 失败 | loader 局部清理 | 当前 Resource shutdown，aggregate 清理此前 Resource | 事务回滚 |
| DataPath initialize 失败 | loader/StorageRuntime 分担 | `StorageRuntime` 回滚 DataPath，aggregate 再销毁 component、释放 Resource | 事务回滚 |
| Runtime 创建成功 | 可产生半初始化 aggregate 的内部路径较多 | 只有完整装配后状态改为 `RUNNING` | 调用方只得到完整对象 |

## 3. 模块职责与依赖变化

### 3.1 分层职责

| 模块 | 重构前职责 | 重构后职责 | 明确禁止的依赖/行为 |
| --- | --- | --- | --- |
| `config/parser` | YAML 翻译并参与完整拓扑校验 | 文件加载、节点遍历、required/unknown field、类型与表示范围检查 | factory lookup、硬件查询、引用/拓扑决策、对象创建 |
| `config/spec` | 没有独立模块，语义散落在 parser 和 contract registry | 默认值、单对象领域校验、引用、contract、路由、ownership、可达性、产品约束、debug formatter | YAML、环境变量、RPC、CUDA、具体组件 |
| `tutti_runtime` | 主要保存 aggregate 生命周期，assembly 在 config loader | preflight、cache overlay、Resource/Backend/StorageRuntime 装配、ownership、rollback、shutdown | YAML 私有节点和 parser helper |
| Backend contract | 静态配置规则与 Runtime ABI 信息共处 registry | spec contract 与 Runtime registration 分离 | Spec 不判断 factory 是否编译可用 |
| `StorageRuntime` | 由 loader 创建并借用 component | 保持既有路由/I/O 契约，初始化所借用 DataPath | 不拥有 TuttiRuntime Resource 或静态 backend spec |

### 3.2 Target 依赖

| Target | 重构前 | 重构后 | 主要依赖边界 |
| --- | --- | --- | --- |
| 配置值模型 | 旧值模型混在 config target/header | `tutti_runtime_spec` | 仅公共 status/value header；无 yaml-cpp/CUDA/gRPC |
| YAML parser | `tutti_config_parse`/旧 config target | `tutti_config_parser` | `tutti_runtime_spec` + yaml-cpp |
| Runtime assembly | `tutti_config` 与 loader | `tutti_runtime` | parser/spec、Resource、StorageRuntime、已启用 backend |
| Backend factory | `tutti/config/backend_factory.*` | `tutti/tutti_runtime/backend_factory.*` | 位于运行时实现层，不再属于 config |

当前目标关系为：

```text
tutti_runtime_spec
  <- tutti_config_parser (+ yaml-cpp)
  <- tutti_runtime

tutti_runtime
  -> tutti_config_parser
  -> tutti_runtime_spec
  -> Resource / Resolver / DataPath implementations
  -> StorageRuntime API
```

## 4. 数据模型和公共 API 对比

### 4.1 配置模型

| 主题 | 重构前 | 重构后 |
| --- | --- | --- |
| 顶层配置类型 | `ParsedConfig` | `tutti::config::TuttiRuntimeSpec` |
| storage aggregate | `CanonicalStorageConfig` | `StorageSpec` |
| profile/runtime 字段 | `ParsedConfig` 字段 | `AcceleratorSpec` / `RuntimeSpec` |
| specific 字段 | 部分平铺在共享 struct | `std::variant` typed config |
| 默认值来源 | parser/旧结构/装配逻辑可能各自处理 | specific spec 类型字段初始化器 |
| YAML 生命周期 | parser 内部及旧 aggregate 语义耦合 | spec 不保存 `YAML::Node` 或 YAML 文本 |
| 运行时事实 | 需要防止混入静态 aggregate | 明确不进入 spec：allocation ID、BDF、设备路径、lease、queue grant |

四类声明都使用 typed config：

| 声明 | 公共身份字段 | Specific config variants |
| --- | --- | --- |
| Resource | `id`, `type` | `NvmeResourceConfig`, `MemoryResourceConfig` |
| Resolver | `id`, `type`, `scheme` | `LocalFileResolverConfig`, `StripedFileResolverConfig`, `MemfsResolverConfig` |
| DataPath | `id`, `type` | `LocalNvmeDataPathConfig`, `StripedLocalNvmeDataPathConfig`, `MemfsDataPathConfig` |
| Backend | `id`, `contract`, 三类引用 | `Ext4LocalNvmeBackendConfig`, `StripedLocalNvmeBackendConfig`, `MemfsBackendConfig` |

例如 `stripe_unit` 只存在于 `StripedLocalNvmeBackendConfig`。striped YAML 未声明时使用
spec 默认值 512 KiB；ext4 YAML 声明同名字段会在 parser 层作为 unknown field 拒绝。

### 4.2 公共 API

| 用途 | 重构前 API | 重构后 API |
| --- | --- | --- |
| 解析 YAML | `parse_tutti_config(path)` | `parse_tutti_runtime_config(path)` |
| 创建应用 Runtime | `load_tutti_config(path)` | `TuttiRuntime::create(path, options)` |
| 程序化创建 | 依赖 loader/internal options | `TuttiRuntime::create(TuttiRuntimeSpec, options)` |
| Runtime namespace | `tutti::config::TuttiRuntime` | `tutti::TuttiRuntime` |
| 默认构造 | 公开，可先产生 aggregate | private，只允许 factory 创建 |
| 调试配置 | 无稳定公共 formatter | `TuttiRuntimeSpec::to_debug_string()` |
| 测试/装配 seam | `TuttiRuntimeAssemblyAccess` 等旧类型 | repository-private `tutti::testing::TuttiRuntimeTestAccess` |

公开头文件收敛为：

| Header | 公开内容 |
| --- | --- |
| `tutti/config/tutti_runtime_spec.h` | 顶层 spec value types、`validate()`、`to_debug_string()` |
| `tutti/config/spec/<kind>/*.h` | 各组件 aggregate/specific value types 及其字段默认值 |
| `tutti/config/tutti_runtime_config_parser.h` | `parse_tutti_runtime_config(path)` |
| `tutti/include/tutti/tutti_runtime.h` | 两个 `create()` 重载、只读查询、state、`shutdown()` |

## 5. Spec 校验实现

`TuttiRuntimeSpec::validate()` 是 YAML 与程序化构造 spec 的统一语义权威。实现保持
`const`，不修改 spec，也不缓存会在 move 后悬空的裸指针 index。

| 阶段 | 当前检查 | 典型错误 |
| ---: | --- | --- |
| 1 | profile 枚举、`accel_id` 范围及二者一致性 | HOST 配置使用非 `-1` accelerator ID |
| 2 | 每个声明的 ID/type/specific config 领域规则 | variant 与 type 不匹配、NVMe selection 非法 |
| 3 | 每组声明内部 ID 唯一 | 两个 Resource 使用同一 ID |
| 4 | backend 的 Resource/Resolver/DataPath 引用存在 | backend 引用不存在的 Resource |
| 5 | 静态 backend contract、type/scheme/cardinality/specific config | striped contract 只有一个 device |
| 6 | resolver scheme 格式和 route 唯一性 | 两个 resolver 声明同一 scheme |
| 7 | 同一 Resource 不被不同 DataPath 消费 | 两个 backend 将同一 Resource 绑定到不同 DataPath |
| 8 | 所有声明均可从 backend 到达 | 孤立 Resource/Resolver/DataPath |
| 9 | 当前产品恰好一个 backend | 零个或多个 backend |

`to_debug_string()` 先调用相同的 `validate()`。有效输出固定按 accelerator、runtime、
resources、resolvers、datapaths、backends 排列，数组保持声明顺序；输出包含 specific
字段和默认值，不包含 allocation、factory pointer 或 lease 等运行时事实。

## 6. Runtime 创建与生命周期实现

### 6.1 创建状态机

| State | 创建/关闭阶段 | 对外语义 |
| --- | --- | --- |
| `INITIALIZING` | Runtime 内部持有部分对象，尚未 commit | 不会返回给调用方 |
| `RUNNING` | Resource、component、backend manifest、StorageRuntime 全部就绪 | `create()` 唯一成功返回状态 |
| `SHUTTING_DOWN` | 正在释放借用关系和 owned objects | shutdown 过程状态 |
| `STOPPED` | 释放完成 | 重复 `shutdown()` 返回成功 |

### 6.2 创建顺序与回滚

| 顺序 | 实现动作 | 失败处理 |
| ---: | --- | --- |
| 1 | path overload 调用 parser | 返回 parse error |
| 2 | spec overload 无条件调用 `validate()` | 返回 spec error |
| 3 | 生成并通过可选 logger 记录 debug string | logger/formatter 异常转换为 `INTERNAL` |
| 4 | 校验编译 profile 和设备数量 | Resource 前返回错误；R2 为 `NOT_FOUND` |
| 5 | 解析 effective cache overlay | 保持既有 programmatic > spec > env > default |
| 6 | 逐个创建并 initialize Resource | 当前和已注册 Resource 均执行 shutdown |
| 7 | 查 Runtime backend registration 并创建 Resolver/DataPath | 缺 factory 返回 `UNSUPPORTED`；aggregate 析构回滚 |
| 8 | 校验 factory product 并注册 owned component/manifest | 容器异常转换为 `INTERNAL`，owned objects 回滚 |
| 9 | 调用 `StorageRuntime::create()` 初始化 DataPath | StorageRuntime 自身回滚，aggregate 继续释放 component/Resource |
| 10 | 保存 StorageRuntime 并设为 `RUNNING` | 返回完整 `unique_ptr<TuttiRuntime>` |

### 6.3 Ownership 与关闭顺序

| 对象 | 所有者 | 关闭约束 |
| --- | --- | --- |
| `StorageRuntime` | `TuttiRuntime` | 最先 shutdown/destroy，停止使用借用 component |
| Resolver | `TuttiRuntime` | 在 StorageRuntime 销毁后逆序销毁 |
| DataPath | `TuttiRuntime` | 在 StorageRuntime shutdown 后逆序销毁 |
| Resource | `TuttiRuntime` | 最后按 initialize 逆序 shutdown 并销毁 |
| Backend manifest | `TuttiRuntime` | 只保存身份及非 owning 关联 |

实际关闭顺序固定为：

```text
StorageRuntime::shutdown()
  -> destroy StorageRuntime
  -> clear backend borrowed relations
  -> destroy Resolver instances in reverse order
  -> destroy DataPath instances in reverse order
  -> Resource::shutdown() in reverse initialization order
  -> destroy Resource instances
  -> STOPPED
```

## 7. 代码与目录差异

### 7.1 生产代码迁移

| 重构前文件/目录 | 重构后文件/目录 | 代码差异 |
| --- | --- | --- |
| `tutti/include/tutti/config/storage_config.h` | `tutti/config/tutti_runtime_spec.h` + `tutti/config/spec/<kind>/*.h` | 删除旧 storage aggregate，顶层模型与组件 value type 按所属目录拆分 |
| `tutti/include/tutti/config/tutti_config.h` | `tutti_runtime_config_parser.h` + `tutti_runtime.h` | parser API 与 Runtime factory API 分离 |
| `tutti/config/tutti_config_parse.cpp` | `tutti/config/parser/tutti_runtime_config_parser.cpp` | 只保留 YAML 主流程、字段分发和表示检查 |
| `tutti/config/storage/*/*_config_parse.cpp` | `tutti/config/parser/<kind>/*_config_parser.cpp` | specific YAML mapping 按类型归档 |
| parser 内整体 storage 校验 | `tutti/config/spec/tutti_runtime_spec.cpp` | 引入固定九阶段语义校验 |
| parser/旧 contract 中 specific 规则 | `tutti/config/spec/<kind>/*_spec.cpp` | specific validator 与 specific value 对称 |
| `tutti/config/storage/backend/contract_registry.cpp` | `config/spec/backend/*` + `tutti_runtime/backend_factory.*` | 静态语义与运行时 ABI/factory 拆分 |
| `tutti/config/backend_factory.*` | `tutti/tutti_runtime/backend_factory.*` | factory 离开 config ownership 边界 |
| `tutti/config/tutti_config.cpp` | `tutti/tutti_runtime/tutti_runtime_create.cpp` | loader assembly 改为 Runtime static factory |
| `tutti/config/tutti_config_internal.h` | `tutti/tutti_runtime/tutti_runtime_internal.h` | 注入 seam 跟随 Runtime，并删除 assembly access |
| `tutti/tutti_runtime/tutti_runtime.cpp` 中旧 namespace/lifecycle | 同文件新实现 | namespace 改为 `tutti`，加入 `INITIALIZING`、事务 adoption 和固定回滚 |

### 7.2 删除的兼容表面

| 删除项 | 当前替代 | 兼容策略 |
| --- | --- | --- |
| `load_tutti_config()` | `TuttiRuntime::create(path)` | 不保留 wrapper |
| `parse_tutti_config()` | `parse_tutti_runtime_config()` | 不保留 alias |
| `ParsedConfig` | `TuttiRuntimeSpec` | 不保留转换层 |
| `CanonicalStorageConfig` | `StorageSpec` | 不保留平行模型 |
| `config::TuttiRuntime` | `tutti::TuttiRuntime` | 不保留 namespace alias |
| `TuttiRuntimeAssemblyAccess` | Runtime private assembly | 直接删除 |
| `tutti_config_parse`、`tutti_config` targets | `tutti_runtime_spec`、`tutti_config_parser`、`tutti_runtime` | 不提供 CMake alias target |

提交统计为 81 个文件变化、3381 行新增、5323 行删除。大部分删除来自旧 loader、旧配置
模型和依赖旧边界的测试；新代码按 parser/spec/runtime 三层重建。

## 8. 测试迁移与覆盖

### 8.1 测试目录对比

| 重构前测试 | 处理 | 重构后测试 |
| --- | --- | --- |
| `tests/storage_config_contract` | 整目录删除 | `tests/config/parser` + `tests/config/spec` |
| `tests/config_loader` | 整目录删除 | `tests/config/*` + `tests/tutti_runtime/tutti_runtime_create_test.cpp` |
| `tests/runtime_resource_registry_contract` | 整目录删除 | Runtime create/lifecycle fake tests |
| `tests/runtime_bundle_loader_contract` | 整目录删除 | `tutti_runtime_hardware_test.cpp`，只使用新 public entry/API |
| 旧 header hygiene config include | 删除旧 include 断言 | 新 parser/spec/runtime public header 与 private seam 隔离断言 |

### 8.2 当前验证矩阵

| 测试层 | Target | 主要覆盖 |
| --- | --- | --- |
| Parser | `tutti_config_parser_test` | P1、P2、required/unknown field、typed mapping、默认 stripe、parser 不做 duplicate ID 拓扑判断 |
| Spec common/specific | `tutti_config_spec_test` | S1-S5、programmatic memfs、profile、引用、ownership、cardinality、variant contract |
| Spec debug | `tutti_config_spec_debug_test` | 固定 golden、默认值、specific 字段、无效 spec 拒绝输出 |
| Runtime fake | `tutti_tutti_runtime_test` | path overload、debug-before-Resource、R1、R2、R3、rollback、state、shutdown 顺序和幂等 |
| Hardware E2E | `tutti_tutti_runtime_hardware_test` | `create(config_path)`、公开 open/register/submit/wait/release/close、byte-exact、重复 shutdown、lease reacquire |
| Header hygiene | `tutti_header_hygiene_test` | 新 public headers 可用，Runtime/parser private 实现头不可公开到达 |

关键负例的落点为：

| ID | 输入 | 当前失败层与结果 |
| --- | --- | --- |
| P1 | `storage.resources` 不是 sequence | Parser，`INVALID_ARGUMENT` |
| P2 | striped `stripe_unit` 不是 `uint64_t` | Parser，错误包含完整 YAML path |
| S1 | 两个 backend | Spec 第 9 阶段 |
| S2 | backend 引用不存在的 Resource | Spec 第 4 阶段 |
| S3 | 同一 Resource 被不同 DataPath 消费 | Spec 第 7 阶段 |
| S4 | contract 与 typed config 不匹配 | Spec 第 5 阶段 |
| S5 | striped selection 只有一个 device | Spec specific validator |
| R1 | spec 合法但 factory 未注册 | Runtime，`UNSUPPORTED`，Resource 回滚 |
| R2 | accelerator ID 等于设备数量 | Runtime preflight，`NOT_FOUND`，Resource 未创建 |
| R3 | Resource 成功后 DataPath initialize 失败 | Runtime/StorageRuntime 回滚，component 销毁且 Resource shutdown |

## 9. 验证结果

### 9.1 HOST

HOST 配置使用：

```bash
cmake -S . -B /tmp/tutti-config-refactor-host \
  -DTUTTI_ACCELERATOR=HOST \
  -DBUILD_TESTING=ON \
  -DTUTTI_BUILD_HARDWARE_STACK=OFF \
  -DTUTTI_FEATURE_LOCAL_NVME=OFF
```

| 验证 | 结果 | 摘要 |
| --- | --- | --- |
| 新 parser/spec/runtime targets 构建 | PASS | 四个新测试 target 均成功编译链接 |
| 新模块测试 | PASS | 4/4 |
| 排除既有 mount manager 后的 HOST 回归 | PASS | 21/21 |
| `tutti_runtime_spec` 依赖审计 | PASS | 无 yaml-cpp、CUDA、gRPC usage requirement |
| HOST 完整 build | 非本次问题导致失败 | 唯一失败为 `mount_manager_contract` 的 yaml-cpp undefined reference |

`mount_manager_contract` 失败符号为
`YAML::detail::node_data::convert_to_map(...)`。该 target 和实现未由本次重构修改；全部本次
新增 target 已在失败前完成构建和测试。

### 9.2 CUDA

| 验证 | 结果 | 摘要 |
| --- | --- | --- |
| `tutti_runtime` concrete NVMe registration 编译 | PASS | local/striped factory 路径成功链接 |
| parser/spec/runtime 测试 | PASS | 4/4，无可用 GPU 时 fake Runtime 仍独立运行 |
| hardware E2E target 编译 | PASS | 使用公开 Runtime/StorageRuntime API |
| hardware E2E 执行 | SKIP | 未提供 `--config`/`--uri` 且当前无真实 GPU/NVMe/daemon 部署，退出码 77 |

硬件测试不会在缺少部署时把环境不足报告为 correctness PASS。具备环境后可使用：

```bash
build/cuda/bin/tutti_tutti_runtime_hardware_test \
  --config /path/to/tutti-runtime.yaml \
  --uri file:///path/to/preallocated-test-file \
  --size 4194304
```

目标文件必须适合所配置 DataPath 的对齐和 direct I/O 要求；测试会覆盖写、清空 device
buffer、读回、byte-exact 比较、关闭、重复 shutdown，以及相同逻辑 Resource 的再次创建。

### 9.3 静态审计

| 检查 | 结果 |
| --- | --- |
| `git diff --check` | PASS |
| 新代码中旧 loader/type/header 名称扫描 | PASS，无残留 |
| 新 config/runtime/test 中 `canonical` 命名扫描 | PASS，无残留 |
| `YAML::Node` 隔离检查 | PASS，只在 parser 私有实现 |
| 工作区状态（实现提交后） | clean |

## 10. 保持不变的行为

| 行为 | 当前结论 |
| --- | --- |
| `StorageRuntime` 路由和公开 I/O contract | 未修改 |
| 一个 `TuttiRuntime` 只允许一个 backend | 保持，现由 spec 第 9 阶段统一检查 |
| 同一 Resource 不被不同 DataPath 消费 | 保持，现由 spec 第 7 阶段统一检查 |
| Cache 优先级 | programmatic override > DataPath spec > environment > default |
| 环境变量插入点 | 仍在 Runtime assembly 的 Resource 创建前 |
| daemon 配置/RPC/protobuf | 未修改 |
| allocation ID、BDF、path、lease 的来源 | 仍只来自运行时 Resource allocation，不进入静态 spec |

## 11. 当前限制与后续验证

| 项目 | 当前状态 | 后续动作 |
| --- | --- | --- |
| 真实 GPU/NVMe I/O | target 已实现并编译，当前环境未执行 | 在有效 daemon、GPU 和预分配文件环境运行 gated hardware E2E |
| `mount_manager_contract` HOST 全量链接 | 既有 yaml-cpp ABI/库选择问题 | 独立修复该 target 的 yaml-cpp include/library 一致性 |
| 多 backend Runtime | 非本阶段目标 | 若产品需要，必须同时修改 spec 产品约束和 Runtime assembly |
| 环境变量进入 effective spec | 非本阶段目标 | 后续单独设计 effective runtime options，不回填静态 spec |

综上，设计中 parser、spec 与 Runtime 的职责拆分和破坏性 API 迁移已经落地；当前剩余工作
是部署依赖型硬件验收和与本重构无关的 mount manager 链接环境问题，不存在旧 config
loader 的兼容路径或第二套静态配置模型。

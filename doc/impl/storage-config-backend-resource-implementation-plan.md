# Storage Config、Backend 与 Resource 分阶段实现和验证计划

## 1. 文档状态

- **状态**：实施路线设计，尚未开始本设计的代码迁移。
- **依据**：[`doc/design/storage-config-backend-resource.md`](../design/storage-config-backend-resource.md)。
- **适用分支基线**：当前 `fix/multi-gpu` 工作树。
- **本地实机配置**：`config/local/daemon_2_disk.yaml`，两块 NVMe，daemon endpoint
  `127.0.0.1:50051`。
- **推进方式**：一次只实施一个阶段；每阶段完成代码、测试和
  `doc/impl/storage-config-backend-resource-phase-N.md`，形成独立提交后再进入下一阶段。

本文件只定义先后顺序和验收规则，不替代目标设计，也不在本阶段修改 loader、daemon
或应用配置。

## 2. 目标和边界

### 2.1 要达到的结果

应用侧配置最终只保留三个一级对象：

```yaml
accelerator: {}
runtime: {}
storage:
  resources: []
  resolvers: []
  datapaths: []
  backends: []
```

loader 的职责收敛为：

```text
YAML
  -> canonical model + 静态校验
  -> ResourceSpec -> ResourceInstance
  -> backend factory + typed resource views
  -> RuntimeComponents
  -> TuttiRuntime(resources/backends/components) + StorageRuntime
```

其中：

- `TuttiRuntime` 按 resource ID 拥有 ResourceInstance、组件和 backend manifest；
- `StorageRuntime` 仍只按 URI scheme 和 DataPath key 路由，不知道 YAML、backend ID 或
  allocation lease；
- `Resource` 公共接口只表达能力、生命周期和只读信息；NVMe client、allocation 和
  release 状态只属于 `NvmeResource`；
- resolver 只保留名称解析所需的 host-side 状态，DataPath 只保留 controller、queue、
  workspace 等数据面状态；两者不能自行申请、释放或替换 resource；
- 当前产品仍限制一个 `TuttiRuntime` 只有一个 backend，但数据结构保留数组表达能力。

### 2.2 明确不在本计划内的事项

- 不修改 daemon 的 YAML schema、protobuf/gRPC RPC 或 `config/local/daemon_2_disk.yaml`；
- 不把 NVMe path、PCI BDF、allocation ID 或 lease 字段加入公共
  `ResourceProvider`、resolver SPI 或 DataPath SPI；
- 不改变 `StorageRuntime` 现有多 scheme、多 DataPath key 的核心能力；
- 不在本设计中实现同 scheme 多 resolver 动态选择、多个 backend 的产品行为、Resource
  子 lease 或跨 Runtime 共享 queue 配额；
- 不以性能阈值替代 correctness、生命周期和资源回收验收。

## 3. 当前实现事实和迁移约束

当前 loader 仍是 legacy 扁平配置：

- [`tutti/config/tutti_config_parse.cpp`](../../tutti/config/tutti_config_parse.cpp) 解析
  `nvme_service`、`nvme`、`storage.backend` 和 `local_nvme`；
- [`tutti/config/tutti_config.cpp`](../../tutti/config/tutti_config.cpp) 直接创建
  `LocalFileResolver`/`StripedResolver` 与 Local/Striped DataPath，并在同一文件中实现
  `TuttiRuntime` 析构、shutdown 和 allocation release；
- [`tutti/config/tutti_config.h`](../../tutti/config/tutti_config.h) 的
  `TuttiRuntime` 仍公开 `resource_client`、`allocation_id`、`allocation_slices`、
  `datapaths`、`resolvers` 等平行字段；
- `RuntimeComponents` 的 `resources` 是 DataPath SPI 的通用 `ResourceProvider*`，不是
  storage resource allocation，迁移时不能复用其传递 NVMe metadata；
- [`tests/config_loader/config_loader_test.cpp`](../../tests/config_loader/config_loader_test.cpp)
  已有 fake client，能验证 acquire/release 次数和 runtime factory 失败回滚；
- [`tests/runtime_bundle_loader_contract/runtime_bundle_loader_contract_test.cpp`](../../tests/runtime_bundle_loader_contract/runtime_bundle_loader_contract_test.cpp)
  仍直接读取 `TuttiRuntime::allocation_slices` 来创建 scratch file、核对 RPC metadata
  和重启持久性。它必须先迁移到只读诊断 seam，才能删除平行 public 字段；
- `config/local/daemon_2_disk.yaml` 只由 daemon 使用。应用 loader 必须继续消费
  `AcquireNvmeSlices` 返回的实际 `chrdev_path`、`block_path`、`view_path`、BAR0、MDTS、
  namespace 和 granted queues，不能从 device ID、数组顺序或 daemon YAML 推导路径。

迁移期间允许保留兼容适配，但所有新代码只能消费 canonical model；legacy 解析只能在
入口一次性转换，legacy 与 canonical 字段混用必须 fail-closed。

## 4. 不变量和验收语言

所有阶段都使用下面的不变量。文档中的“通过”必须有测试输出或可复现的检查证据，不以
“构建成功”代替行为验证。

### 4.1 静态阶段不变量

1. 四组对象各自的 ID 非空且唯一，backend 引用必须存在。
2. 当前产品 `backends.size() == 1`；启用的 resolver scheme 和导出的 DataPath key
   唯一；所有声明必须从唯一 backend 可达。
3. contract/type/resource cardinality 组合必须符合兼容矩阵；例如
   `ext4-local-nvme` 只能是 `local-file + local-nvme + nvme + 1 slice`，
   `striped-local-nvme` 必须是有序的至少 2 slices。
4. `allowed`/`explicit`/`striped` 的 device ID 规则、queues、stripe unit 和 backend
   config 在任何 daemon RPC 或设备初始化前校验。
5. 静态错误不得触发 `list_*`、`acquire_*`、DataPath constructor 或
   `StorageRuntime::create()`。

### 4.2 所有权和失败回滚不变量

1. 一个 resource ID 在一次 Runtime 创建中最多 Acquire 一次；多个 backend 引用时复用
   同一个 ResourceInstance（当前产品先拒绝多个独立消费者）。
2. Acquire 成功后 allocation 立即有唯一 owning `NvmeResource`；之后 metadata 校验、
   resolver/DataPath factory、DataPath initialize 或 Runtime create 任一步失败，都必须
   Release 恰好一次。
3. 正常关闭顺序固定为：drain/`StorageRuntime::shutdown()` -> 销毁 runtime -> 清除
   backend 非 owning 引用 -> resolver/DataPath -> 按 resource 初始化逆序
   `Resource::shutdown()`。
4. `TuttiRuntime::shutdown()` 幂等；析构是 best-effort 兜底，不得二次 Release；正常
   shutdown 返回第一个错误，但仍尝试清理所有已拥有对象。
5. backend/DataPath 仍引用 resource 时，直接 shutdown resource 返回 `BUSY`；最终关闭
   不允许存在悬空 view、controller、queue 或 allocation lease。

### 4.3 硬件事实不变量

实机测试必须逐项证明 RPC 返回值被使用，而不是由配置推导：

```text
pci_bdf, chrdev_path, block_path, backing_mount_path, view_path
namespace_id, logical_block_size, bar0_size, max_data_size, granted_queues
```

Striped 测试还必须证明 slice 顺序同时用于 resolver shard 和 DataPath device descriptor，
并且一次 allocation 的 shutdown 会同时归还两块盘的 queue reservation。

## 5. 阶段总览

| 阶段 | 主题 | 主要产物 | 主要验证 | 完成后的能力 |
|---|---|---|---|---|
| P0 | 基线和观测面冻结 | 基线记录、测试 seam 清单 | HOST/CUDA 无硬件 + 双盘 daemon 当前 loader | 知道迁移前行为和资源清理基线 |
| P1 | canonical 配置模型和静态解析 | typed config model、严格 schema、legacy adapter | 纯 host parser contract | 静态错误在 Acquire 前失败 |
| P2 | `TuttiRuntime` 文件和所有权拆分 | `tutti_runtime.h/.cpp`、兼容生命周期 | loader lifecycle/failure tests | Runtime 生命周期脱离 parser 文件 |
| P3 | Resource SPI 和 `NvmeResource` | `Resource`、`NvmeResource`、NVMe 专用 client | Resource/fake-client contract | allocation 有一等 owning instance |
| P4 | Resource graph 装配 | Resource registry、初始化/回滚顺序 | acquire 次数、cardinality、失败注入 | canonical ResourceSpec 可实例化 |
| P5 | Backend factory 和 typed views | contract matrix、resolver/DataPath factories、manifest | local/striped 组合与视图 contract | 不再散列传递 NVMe 字段 |
| P6 | 迁移、封装和回归收口 | canonical 示例、测试访问迁移、删除 legacy 字段 | HOST/CUDA 全量非硬件 | 目标 loader 成为默认路径 |
| P7 | daemon 双盘实机闭环 | 实施记录、清理证据 | explicit/striped I/O、restart、ledger | 真实 allocation/Release 验收 |

阶段之间有依赖：`P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7`。P0 是所有阶段的入口；如果
某阶段 exit gate 未通过，不进入下一阶段，也不以临时绕过代码宣称完成。

## 6. 分阶段实施和验证

### P0：建立基线和观测面

**实施范围**

- 不改产品行为；确认当前 branch、工作树和 build preset，记录现有修改，不覆盖用户的
  dirty changes。
- 为后续测试固定三类观测：fake client 的 acquire/release ledger、Runtime factory
  注入失败、daemon `ListNvmeResources` 的 reservation ledger。
- 盘点所有直接依赖 `TuttiRuntime` public allocation 字段的调用点，优先准备只读测试
  seam 的迁移清单，不在本阶段删除字段。

**验证步骤**

```bash
cmake --build build/host --parallel 8
ctest --test-dir build/host --output-on-failure -j 8
cmake --build build/cuda --parallel 8
ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8
```

具备硬件前提时，再按第 7 节启动 daemon，记录两盘初始 `reserved_queues`、
`available_queues`、BDF、mount、device node 和 accelerator view。运行现有
`tutti_runtime_bundle_loader_contract_test` 后确认 ledger 回到基线，再单次 SIGINT/SIGTERM
停止 daemon，并确认 mount、view symlink、chrdev/block node 已清理。

**Exit gate**

- HOST/CUDA 非硬件回归全绿；
- 当前 loader 的单盘、striped（如环境可用）测试结果有记录；
- daemon 测试前后 reservation ledger 相同，且没有遗留进程或 allocation；
- 形成 `doc/impl/storage-config-backend-resource-phase-0.md` 后才能开始 P1。

### P1：canonical config model 和静态 schema

**代码范围**

- 在 `tutti/config` 中增加纯值类型：`ResourceSpec`、`ProviderSpec`、`AllocationSpec`、
  `ResolverSpec`、`DataPathSpec`、`BackendSpec`、`CanonicalStorageConfig`；字段只表达
  声明，不放 allocation/path/controller 事实。
- 重写 `tutti_config_parse.cpp` 为“读取 -> canonical model -> 静态校验”；legacy
  `nvme_service`/`nvme`/`storage.backend`/`local_nvme` 只由一个 adapter 转换。
- canonical 和 legacy 字段同文件出现时立即报 `INVALID_ARGUMENT`；不做字段优先级猜测。
- 增加逻辑 contract registry，至少注册 `ext4-local-nvme` 和
  `striped-local-nvme`；`memfs` 只保留 schema/兼容性入口，若产品 factory 尚未实现则
  以明确 `UNSUPPORTED` 失败。
- 保留 `ParsedConfig` 的兼容返回或提供适配层，避免 P1 同时改变 Runtime 装配。

**必须覆盖的 parser case**

- 缺 ID、重复 ID、悬空引用、重复 scheme、重复 contract DataPath key；
- 0 个或多个 backend、未引用声明、未知 contract/type、非法 scheme；
- `allowed` 携带 device ID、`explicit` 非 1 个、`striped` 少于 2 个或有重复 ID；
- queues 为负、stripe unit 为 0/未对齐、canonical/legacy 混用；
- 合法 local canonical、合法 striped canonical、HOST 无 storage 的配置。

**验证**

- 扩展 `tutti_config_loader_test` 或新增纯 host `tutti_storage_config_contract_test`；
- 每个静态失败 case 都注入一个计数 client，断言 acquire/list 均为 0；
- HOST 和 CUDA 两个 build 都运行 parser test，确认 parser 不依赖 CUDA 或 daemon。

**Exit gate**：canonical model 的合法/非法矩阵通过；静态错误在任何资源申请前失败；
legacy 合法配置仍行为等价；不修改 daemon 配置和现有 `StorageRuntime` contract。

### P2：独立 `TuttiRuntime` 生命周期模块

**代码范围**

- 新增 `tutti/config/tutti_runtime.h` 和 `tutti/config/tutti_runtime.cpp`，迁出
  `TuttiRuntime` 声明、析构、`shutdown()` 和阶段性 registry 操作。
- P2 先保持旧 loader 的装配结果不变；允许保留兼容 public 字段，但由新 Runtime 模块
  统一清理，禁止 `tutti_config.cpp` 再实现生命周期逻辑。
- 把 shutdown 的 first-error 传播、幂等标志和 reverse cleanup 变成可测试的显式状态机。
- 为硬件 E2E 提供临时只读 inspection seam（返回 copy 或 const view），禁止返回可修改
  allocation/client 所有权；最终 API 形态在 P6 收口。

**验证**

- 现有 `tutti_config_loader_test`：正常 shutdown 两次、析构兜底、runtime factory
  失败、constructor 抛异常、release 失败仍完成剩余清理；
- 增加一个 fake `StorageRuntime` 记录 shutdown 调用，断言 runtime 先于 DataPath/Resolver
  和 resource 清理；
- `git diff --check`、HOST/CUDA 编译，确认 include 依赖没有把 CUDA 引入 parse library。

**Exit gate**：Runtime 行为无回归，生命周期实现已从 config loader 文件分离，任何失败
路径都能通过测试证明“不泄漏、不二次 Release”。

### P3：Resource SPI 与 `NvmeResource`

**代码范围**

- 在 config/resource 模块增加公共最小 `Resource` 接口以及 `ResourceState`、
  `ResourceCapabilities`、`ResourceInfo`；不修改公共 `ResourceProvider`。
- 将 `RuntimeResourceClient` 降级并重命名为 NVMe 模块内部的 `NvmeResourceClient`；
  gRPC 实现和 fake 实现都只服务 `NvmeResource`。
- 增加 `NvmeResource : Resource`：拥有 typed spec、专用 client、allocation metadata、
  `NvmeLeaseState`；`initialize()` Acquire，成功后立即接管 allocation；`shutdown()`
  幂等 Release。
- 增加 NVMe-specific `resolver_view()`/`datapath_view()`，只返回构造所需的只读副本或
  const view，不把这些方法放进 `Resource` 基类。
- 规定 `NvmeResource` 状态转移：`CREATED -> INITIALIZED -> SHUTTING_DOWN -> STOPPED`；
  初始化或释放异常进入 `FAILED`，仍禁止第二次释放。

**验证**

- 新增纯 host `tutti_resource_contract_test`，用 fake NVMe client 覆盖 Acquire 成功、
  Acquire 失败、空 allocation ID、缺 metadata、Release 失败、析构兜底、重复 shutdown。
- 验证 allocation 成功后在后续任意 factory 失败路径仍能找到唯一 owner；client 和
  allocation 不再存在于 `TuttiRuntime` 平行字段。
- 验证 `Resource::info()` 只读快照与 registry ID/type/state 一致；不能通过它修改 slice
  或 lease。

**Exit gate**：单独的 Resource contract 全绿；NVMe allocation 的唯一所有者变为
`NvmeResource`；公共接口没有 transport-specific 字段；daemon 尚未连接的 HOST test
仍可运行。

### P4：ResourceSpec 到 ResourceInstance 的解析层

**代码范围**

- loader 先建立四组 ID 表和 backend reachability，再为被引用的 ResourceSpec 调用
  `ResourceFactory`；每个 resource ID 只初始化一次。
- 资源初始化顺序写入 `resource_initialization_order`；失败按已成功初始化顺序逆序回滚。
- 根据 selection 校验 allocation cardinality；local 必须 1 slice，striped 必须请求数
  量且顺序完全一致；所有 slice 的 accelerator、ACL、namespace、block size 和必需
  metadata 必须有效。
- 未引用资源、重复消费同一 resource 的独立 DataPath 和 resource shutdown 时机按当前
  产品约束拒绝，错误发生在后续 Acquire 前（能静态判断的部分）或刚收到 allocation
  后（依赖运行时 metadata 的部分）。
- `TuttiRuntime` 增加私有 `resources_[id]` registry 和只读访问接口；backend manifest
  先保存 ID/contract/指针关系的内部表示。

**故障注入验证**

| 注入点 | 期望 |
|---|---|
| 第二个 resource snapshot 缺失 | 第一个已成功 resource 逆序 shutdown；每个 allocation 一次 Release |
| allocation ID 为空/片数错误/顺序错误 | Acquire 后立即 Release 一次；不创建 DataPath |
| slice ACL/accel/block-size 不符 | Acquire 后 Release 一次；错误明确 |
| 第二个 ResourceFactory 初始化失败 | 已初始化 resource 全部逆序释放 |
| backend 引用同一 resource 两次 | fail-closed，不重复 Acquire |

**Exit gate**：Resource registry、初始化顺序和失败回滚有单元测试；fake client ledger
在所有错误路径回到零；现有 loader 的单 backend 行为仍可由兼容 adapter 产生。

### P5：Backend factory、contract 矩阵与 typed resource views

**代码范围**

- 增加 backend factory registry，将逻辑 contract 映射到 resolver type、DataPath type、
  resource type、cardinality、payload contract 和推荐 DataPath key；config 不直接指定
  payload type ID/API version/key。
- `ext4-local-nvme` factory 从一个 `NvmeResource` 的 resolver view 创建一个
  `LocalFileResolver`，从 datapath view 创建一个 `LocalNvmeDataPath`；
  `striped-local-nvme` factory 保留有序 shard views，创建 N 个 local shard resolver、
  一个 `StripedResolver` 和一个含 N 个 descriptor 的 `StripedDataPath`。
- `stripe_unit` 等 resolver/DataPath 共享语义只从 backend config 注入；controller path、
  BAR0、MDTS、granted queues 等只能来自 resource view。
- factory 在构造、resolver payload contract、DataPath initialize 和 Runtime create 之间
  逐级校验；任一 shard 失败回滚整个 backend。
- `TuttiRuntime` 保存只读 backend manifest：`backend_id -> contract/resolver_id/
  datapath_id/resource_id`；`StorageRuntime` 仍只接收 scheme/key 两组 binding。

**验证**

- local/striped 正向 contract：只发布一个顶层 scheme 和一个 DataPath key；resolver 和
  DataPath 消费的 namespace identity、slice 顺序一致；striped MDTS 取各 shard 最小值。
- 负向 contract：memfs/local-nvme 混搭、local + 多片 allocation、striped + 单片、未知
  contract、错误 payload version、重复 scheme/key 均在 DataPath initialize 前失败。
- 注入 DataPath initialize/runtime factory 失败，断言全部 resource Release 一次且没有
  悬空 resolver/DataPath。
- 保留并运行现有 `storage_runtime_contract` 的多 scheme/multi-key case，证明核心路由
  能力没有收缩。

**Exit gate**：backend 关系显式、contract 兼容性 fail-closed、所有 transport 字段只
  通过 typed view 注入；loader 不再依赖 `allocation_slices.size()` 反向猜 backend 类型。

### P6：canonical 迁移、API 封装和回归收口

**代码范围**

- 将 [`config/tutti_config.yaml`](../../config/tutti_config.yaml) 和应用配置文档更新为
  canonical 示例；legacy adapter 仍可读取旧文件，但输出废弃诊断和迁移说明。
- 将 loader E2E 从 `allocation_slices` 等 public 平行字段迁移到只读
  `ResourceInfo`/backend manifest 或明确的 test-only immutable inspection seam；删除
  `TuttiRuntime::resource_client`、`allocation_id`、`allocation_slices` 等可绕过 registry
  的字段。
- `TuttiRuntime` 内部只保留 owning registry；外部不能单独销毁 resource、替换 client、
  修改 allocation 或让 DataPath 脱离 backend 关系。
- 为 public headers、CMake target 和安装列表补齐新文件；parse-only library 继续保持
  无 CUDA/daemon 依赖。
- 记录 legacy 配置移除的版本/期限；canonical 与 legacy 混用测试必须固定为失败。

**验证**

```bash
cmake --build build/host --parallel 8
ctest --test-dir build/host --output-on-failure -j 8
cmake --build build/cuda --parallel 8
ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8
git diff --check
```

额外检查：

- `tutti_config_loader_test`、`tutti_resource_contract_test`、backend contract 全绿；
- `tutti_runtime_bundle_loader_contract_test` 编译时不再依赖已删除的平行字段；
- `StorageRuntime`、resolver、DataPath 公共头中没有 NVMe service/protobuf 类型；
- 失败后 `TuttiRuntime` 不暴露半初始化 registry，重复 shutdown/destructor 均幂等。

**Exit gate**：canonical loader 成为默认装配路径；legacy 只在明确兼容入口存在；HOST
和 CUDA 非硬件集合全绿；没有 config、daemon 或生成物的无关改动。

### P7：daemon 双盘实机闭环

P7 只在 P6 通过后进行。实机验证使用用户指定的两盘配置，应用临时 canonical YAML
不得写入仓库，也不得包含任何设备路径。

**准备和启动 daemon**

```bash
# 先确认没有遗留 daemon；若已有实例，先按既有清理流程停止并确认 50051 释放。
sudo -S env TUTTI_VERBOSE=1 \
  build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon \
  --config config/local/daemon_2_disk.yaml < ~/.passwd/1
```

实际路径以当前 build 产物为准；启动前后不得把密码放入命令行、YAML、日志或实施文档。
用 `nvmeservice_client --endpoint 127.0.0.1:50051 --list-only` 保存基线：两个 accelerator、
两个 device、BDF、实际 chrdev/block/mount、namespace/LBA、BAR0/MDTS、available queues、
以及 `reserved_queues`。

**必须执行的场景**

| 场景 | 配置 | 验收 |
|---|---|---|
| A0 | accelerator 0 + explicit device 0 | Local write/read/byte-exact；只增加 device 0 queue reservation |
| A1 | accelerator 0 + explicit device 1 | Local write/read/byte-exact；验证跨 ACL 允许的设备 |
| A2 | accelerator 0 + striped `[0,1]` | 跨 64 KiB 边界、mixed batch、重启后读取；一次 allocation 同时占用两盘 |
| B0 | accelerator 1 + explicit device 0 | 与 A0 同样的 I/O 和回收 |
| B1 | accelerator 1 + explicit device 1 | 与 A1 同样的 I/O 和回收 |
| B2 | accelerator 1 + striped `[0,1]` | 与 A2 同样的跨 shard、重启和回收 |

若机器没有两个可用 accelerator 或两块盘不可用，测试必须返回明确 `SKIP`，不能把缺失
硬件报告成通过；若只验证单盘，则不得宣称 P7 完成。

**每个场景的检查顺序**

1. 记录 daemon snapshot；调用 canonical `load_tutti_config()`，断言静态 config 不含
   path/BDF/lease 字段。
2. 通过只读 ResourceInfo/test inspection seam 读取 allocation metadata，并逐项与独立
   `ListNvmeResources` snapshot 对比；检查 view path 与 backing mount 位于同一文件系统。
3. 在返回的 view path 创建唯一 scratch file（root client；普通用户权限不足时记录为
   部署前提，不修改 daemon 权限）；使用 public `StorageRuntime` 完成
   `open/register_memory/submit/progress/query/wait/close/unregister_memory` 和 byte
   exact 校验。
4. Striped 场景覆盖 stripe 边界、多个 shard 的 batch、write -> shutdown -> reload ->
   read persistence，并断言 resolver shard 顺序等于 DataPath descriptor 顺序。
5. 调用 `TuttiRuntime::shutdown()` 两次；轮询 daemon snapshot，断言所有选中 device 的
   reservation 和 available queues 精确回到基线。

**daemon 停止和清理**

- 测试完成后只发送一次 Ctrl-C/SIGTERM，等待 `tutti_daemon exited cleanly`；不要使用
  `kill -9`，除非进入单独的应急恢复流程；
- 检查 `findmnt` 中两处 mount 已卸载、GPU view symlink 已删除、`/dev/ssnvme*` 和
  `/dev/snvme*n1` 已清理、50051 已释放；
- 删除临时 YAML、scratch file 和临时目录，确认没有测试进程、未释放 allocation 或
  queue group；
- 将实际 metadata、命令、场景结果、ledger 前后值、清理结果和残余风险写入
  `doc/impl/storage-config-backend-resource-phase-7.md`，不写入密码。

**P7 exit gate**

- A0/A1/A2/B0/B1/B2 在硬件可用时全部 correctness PASS；
- 每次 shutdown 后 ledger 精确恢复，striped allocation 两盘一次性释放；
- daemon clean shutdown，无遗留 mount/device/view/process；
- HOST/CUDA 非硬件回归在实机改动后再次通过；
- 只在上述条件全部满足时宣称目标设计完成。性能数字单独记录为观察值，不作为本
  resource ownership 设计的唯一通过条件。

## 7. 验证矩阵和证据格式

每个阶段实施记录统一包含：基线 commit、改动范围、测试命令、原始计数、失败/skip
原因、清理状态、残余风险和 exit gate 判定。

| 层次 | 命令/target | 目的 | 必须证明 |
|---|---|---|---|
| 纯解析 | `tutti_config_loader_test` / `tutti_storage_config_contract_test` | schema、引用、legacy adapter | 静态失败无 RPC/Acquire |
| Resource | `tutti_resource_contract_test` | state、client、allocation ownership | Release 至多一次、失败可回滚 |
| Backend | backend factory contract + existing resolver/DataPath tests | contract/view/cardinality | local/striped 组合正确、错误 fail-closed |
| Runtime | `tutti_storage_runtime_contract_test` | 核心路由和生命周期回归 | multi scheme/key 能力未收缩 |
| CUDA 非硬件 | `ctest --test-dir build/cuda -LE hardware` | include/link/current-device 回归 | 全部通过 |
| HOST 非硬件 | `ctest --test-dir build/host` | parser/resource/Runtime portability | 全部通过 |
| daemon 控制面 | `nvmeservice_client --list-only` | allocation metadata 和 ledger | BDF/path/queue facts 一致 |
| Local 实机 | `tutti_local_nvme_datapath_contract_test`、`tutti_storage_runtime_local_nvme_contract_test` | 单盘 I/O | DMA、读写、关闭和回收 |
| Striped 实机 | `tutti_striped_local_nvme_contract_test` + loader E2E | 多 shard I/O | 顺序、边界、持久化、一次 Release |

故障注入至少记录以下计数：`list_calls`、`acquire_calls`、`release_calls`、每个
`Resource::initialize/shutdown` 调用次数、DataPath initialize/shutdown 次数，以及 daemon
每个 device 的 `reserved_queues/available_queues`。

## 8. 风险、暂停条件和处理顺序

- **静态 schema 与现有 legacy 文件冲突**：先保留 adapter，修正测试/示例后再收紧；不能
  让 canonical 和 legacy 在同一文件中按“优先级”共存。
- **ResourceInfo 与硬件 E2E 观测需求冲突**：优先使用返回 copy 的 immutable
  diagnostic/test seam；不得为测试重新公开可写 allocation 或让测试直接持有 client。
- **DataPath initialize 在 GPU/NVMe 上失败**：先验证 Resource 已接管 allocation，再修
  回滚；不能临时跳过 Release 继续测试。
- **daemon ledger 未回零、mount 忙或 device node 残留**：停止后续场景，收集 holder、
  PID、fd、mount 和日志；按 daemon 文档单次优雅停止/重试处理，不使用破坏性 reset。
- **硬件环境不满足**：允许明确 `SKIP`，但只能把对应阶段标记为“未验证”，不能伪造
  PASS，也不能用性能数字替代缺失的 correctness 验证。
- **与用户已有 dirty changes 冲突**：只在本计划涉及文件上协作；无关修改保持原样；若
  目标文件的变化使实现无法安全继续，暂停并报告具体冲突。

## 9. 最终完成定义

本设计只有在以下条件同时满足时才算完成：

1. canonical config 已成为唯一推荐、默认装配路径，legacy 仅由有期限的 adapter 支持；
2. `TuttiRuntime` 通过 resource ID registry 拥有具体 Resource，NVMe allocation/client
   不再以平行字段存在；
3. backend factory 通过 contract 和 typed resource views 装配 resolver/DataPath，
   `StorageRuntime` 的 scheme/key 路由契约没有收缩；
4. 所有静态错误、factory/runtime 失败和 shutdown 失败路径都有回滚/幂等测试；
5. HOST/CUDA 非硬件回归全绿；
6. 在双盘 daemon 上完成可复现的单盘、striped、重启、ledger 回收和 clean shutdown
   验证，实测 metadata 与 RPC 一致；
7. 每个阶段都有独立实施记录和 commit，最终文档明确记录任何未解决的性能或并发残余
   风险。

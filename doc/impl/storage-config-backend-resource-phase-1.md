# Storage Config、Backend 与 Resource 阶段 1 实施记录

## 1. 阶段范围与结论

本阶段实现 `doc/impl/storage-config-backend-resource-implementation-plan.md` 的 P1：
canonical storage 值模型、严格静态 schema、contract registry 和 legacy 单次 adapter。
阶段基线 commit 为 `2e880ef`，分支为 `fix/multi-gpu`，测试日期为 2026-08-11 UTC。

阶段结论：P1 exit gate 通过。合法 local/striped canonical 配置可转换到现有
`ParsedConfig` 兼容装配字段；legacy 合法配置保持行为等价；19 类 canonical 静态错误
均在 client factory、list RPC、Acquire 或 DataPath 构造之前失败。HOST/CUDA parser
contract 和完整非硬件回归全绿，daemon schema、RPC、配置和 `StorageRuntime` contract
均未修改。

## 2. 计划项与本次实现对比

| P1 计划项 | 本次实现 | 结果 |
| --- | --- | --- |
| 增加纯值 canonical model | 新增 `ProviderSpec`、`AllocationSpec`、`ResourceSpec`、`ResolverSpec`、`DataPathSpec`、`BackendSpec`、`CanonicalStorageConfig` | 完成 |
| 值模型不包含运行时事实 | model 只保存 provider endpoint、selection、device IDs、请求 queues、调优和 backend 关系；无 allocation ID、path、BDF、BAR0、MDTS、controller 或 lease | 完成 |
| 读取后统一静态校验 | parser 先识别 canonical/legacy，再生成 canonical graph，统一执行 ID、引用、scheme、reachability、contract、cardinality 和 backend config 校验 | 完成 |
| canonical/legacy 混用 fail-closed | canonical 四数组与任意 legacy storage/root 字段同时出现即 `INVALID_ARGUMENT` | 完成 |
| 逻辑 contract registry | 注册 `ext4-local-nvme`、`striped-local-nvme` 和 schema-only `memfs`，固定 resolver/DataPath/resource type、cardinality 和 Runtime key | 完成 |
| memfs 未实现时明确失败 | 完成 schema/type/引用校验后返回 `UNSUPPORTED`，不伪装为 NVMe 或隐式选择 factory | 完成 |
| 保留 `ParsedConfig` 兼容层 | `ParsedConfig` 新增 syntax/canonical model；canonical 和 legacy 都填充旧 loader 所需的只读兼容字段，P1 不改 runtime assembly | 完成 |
| HOST/CUDA 纯 parser target | `tutti_config_parse` 提升为所有 profile 构建；新增 `tutti_storage_config_contract_test`，只链接 parse library | 完成 |
| 静态失败零 RPC | CUDA full loader 使用计数 fake client 验证 19 类错误的 list/acquire/release 全为 0 | 完成 |

## 3. 业务逻辑变化

### 3.1 Canonical 配置路径

canonical 应用配置只接受 `accelerator`、`runtime`、`storage` 三个一级字段；
`storage` 必须显式包含 `resources`、`resolvers`、`datapaths` 和 `backends` 四个数组。
唯一例外是 `accelerator.profile=HOST` 且 `runtime.accel_id=-1` 的无 storage 配置。

静态校验现在在任何资源观测或申请之前执行：

- 每组 ID 必须非空且组内唯一；backend 引用必须存在；
- 当前产品必须恰好一个 backend，所有声明必须从该 backend 可达；
- resolver scheme 必须为小写合法 URI scheme 且 Runtime 内唯一；
- contract 固定 resolver type、DataPath type、resource type、cardinality 和 DataPath key，
  用户不能通过 YAML 改写 payload type/API version/Runtime key；
- `allowed` 不得带 device ID，`explicit` 必须恰好一个 ID，`striped` 必须至少两个有序
  且不重复的 ID；requested queues 必须为非负 int32；
- striped backend 的 `stripe_unit` 必须非零且按 4096 bytes 对齐；非 striped contract
  不允许声明 `stripe_unit`；
- canonical nested mapping 使用严格字段白名单，未知字段立即失败。

### 3.2 Legacy 兼容路径

legacy `nvme_service`、`nvme`、`storage.backend`、`local_nvme` 和
`local_nvme_config` 只在入口解析一次，并转换为带稳定 synthetic ID 的 canonical graph。
legacy flat `ParsedConfig` 字段继续保留给当前 loader；P1 不改变 allocation、resolver、
DataPath 或 `StorageRuntime::create()` 的装配顺序。

合法 legacy single/striped loader 行为保持不变。`storage.backend=rdma` 仍返回
`UNSUPPORTED`；未知 legacy backend 现在 fail-closed，不再被现有 loader 静默当作
local NVMe。

### 3.3 Build 行为

`yaml-cpp` 和 `tutti_config_parse` 从 local-NVMe hardware stack 内移到通用 target graph，
因此 HOST profile 也编译并运行相同 parser。full `tutti_config` loader 仍只在 hardware
stack + local-NVMe feature 启用时创建，未把 CUDA、gRPC 或 daemon 依赖带入 HOST parser
测试。

## 4. 文件变化

| 文件 | 变化 |
| --- | --- |
| `tutti/config/storage_config.h` | 新增 canonical 值模型、syntax 标识和 contract descriptor/lookup |
| `tutti/config/tutti_config.h` | 复用新 `NvmeSelection` 定义，并在 `ParsedConfig` 中保存 syntax 与 canonical model |
| `tutti/config/tutti_config_parse.cpp` | 重写为 syntax 识别、strict canonical parse、legacy adapter、graph/contract 静态校验和兼容字段投影 |
| `tutti/config/CMakeLists.txt` | parse library 始终可构建，full loader 由 hardware stack 在依赖就绪后创建 |
| `tutti/CMakeLists.txt` | 通用查找 `yaml-cpp`、注册 parser target 和全 profile parser contract test |
| `tests/storage_config_contract/CMakeLists.txt` | 新增硬件无关 parser contract target |
| `tests/storage_config_contract/storage_config_contract_test.cpp` | 新增合法矩阵、非法矩阵、legacy adapter、HOST no-storage 和 registry 测试 |
| `tests/config_loader/config_loader_test.cpp` | 新增 canonical local/striped assembly 正向测试及 19 类静态失败零 RPC 测试 |
| `doc/impl/storage-config-backend-resource-phase-1.md` | 新增本阶段实施和验证记录 |

本阶段未修改 daemon YAML/protobuf/gRPC、`config/local/daemon_2_disk.yaml`、resolver/DataPath
SPI、binding payload contract、`StorageRuntime` 路由或现有 TuttiRuntime 所有权字段。

## 5. Parser 验证矩阵

### 5.1 正向

| 场景 | 结果 | 证明内容 |
| --- | --- | --- |
| canonical local NVMe | PASS | graph 各 1 个对象；contract 为 `ext4-local-nvme`；endpoint、selection、tuning 投影到兼容字段 |
| canonical striped NVMe | PASS | contract 为 `striped-local-nvme`；device order `[1,0]` 保持；stripe unit 为 65536 |
| legacy striped NVMe | PASS | syntax 标记 legacy；一次转换得到 striped canonical backend；endpoint/tuning 保持 |
| HOST 无 storage | PASS | `HOST/-1` 返回空且 `present=false` 的 canonical storage |
| canonical local full loader | PASS | list accelerator/resource 各 1 次、Acquire 1 次、Release 1 次；发布 `file`/`local-nvme-ext4` |
| canonical striped full loader | PASS | Acquire/Release 各 1 次；发布 `striped`/`striped-local-nvme` |

### 5.2 负向

parser contract 覆盖并拒绝：缺 ID、重复 ID、悬空引用、重复 scheme、重复 contract
DataPath key、0 backend、多 backend、未引用声明、未知 contract/type、非法 scheme、
`allowed` 带 ID、`explicit` 0/多个 ID、`striped` 少于 2 个/重复 ID、负 queues、
stripe unit 为 0/未对齐、canonical/legacy 混用、unknown canonical field、contract type
混搭和 memfs factory 未实现。

CUDA full loader 对其中 19 类 canonical 静态错误逐例注入计数 client，结果均为：

```text
list_accelerators_calls = 0
list_resources_calls    = 0
acquire_calls           = 0
release_calls           = 0
```

`memfs` 先通过 registry schema/type 兼容检查，再以 `UNSUPPORTED` 返回；未知或混搭配置
返回 `INVALID_ARGUMENT`。

## 6. 非硬件验证结果

| 命令/target | 结果 | 原始结果摘要 |
| --- | --- | --- |
| HOST `tutti_storage_config_contract_test` | PASS | 88 checks、0 failures |
| CUDA `tutti_storage_config_contract_test` | PASS | 88 checks、0 failures |
| CUDA `tutti_config_loader_test` | PASS | 124 passed、0 failed |
| `cmake --build build/host --parallel 8` | PASS | 包含通用 `tutti_config_parse` 和新 parser test |
| `ctest --test-dir build/host --output-on-failure -j 8` | PASS | 18/18，通过时间 0.26 秒 |
| `cmake --build build/cuda --parallel 8` | PASS | full loader、runtime bundle E2E binary 和所有非硬件 target 构建成功 |
| `ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8` | PASS | 20/20，通过时间 0.49 秒 |
| `git diff --check` | PASS | 无 whitespace error |

## 7. 基于硬件的验证结果

### 7.1 环境

- accelerator：1 块 NVIDIA L40S，`accel_id=0`；
- daemon：`config/local/daemon_2_disk.yaml`，endpoint `127.0.0.1:50051`；
- device 0：BDF `0000:b1:00.0`，chrdev `/dev/ssnvme0`，block
  `/dev/snvme0n1`，backing `/mnt/nvme0`，LBA 4096，BAR0 16384，available 23；
- device 1：BDF `0000:e3:00.0`，chrdev `/dev/ssnvme1`，block
  `/dev/snvme1n1`，backing `/mnt/nvme1`，LBA 4096，BAR0 16384，available 72。

### 7.2 结果

| 场景 | 结果 | 证据 |
| --- | --- | --- |
| explicit device 0、4 queues | PASS | RPC 返回 `/dev/ssnvme0`、`/dev/snvme0n1`、view `/mnt/gpu0/ssnvme0`、granted 4；attach/group/destroy/detach 均成功 |
| striped `[0,1]`、每盘 4 queues | PASS | 单个 allocation 返回 2 个有序 slice；两盘均完成 attach/group/destroy/detach |
| 前后 reservation ledger | PASS | device 0 为 `reserved=0 available=23`；device 1 为 `reserved=0 available=72` |
| 完整 loader hardware E2E | SKIP | 返回 77；现有测试要求两个 accelerator，本机只有一个 L40S |
| daemon clean shutdown | PASS | 单次 SIGTERM，日志 `tutti_daemon exited cleanly`；无 mount、view、device node、50051 listener 或 daemon 进程残留 |

P1 的 parser/static validation 本身不访问硬件；硬件回归用于证明 build graph 和兼容
loader 变更未影响已有 daemon allocation/control path。由于双 accelerator 前置条件不
满足，本阶段没有把完整 loader I/O 场景报告为通过。

## 8. Exit Gate 与后续约束

- canonical 合法/非法矩阵通过，静态错误在 client factory/list/Acquire 前失败；
- legacy 合法 loader 行为等价，runtime assembly 和 `StorageRuntime` contract 不变；
- HOST/CUDA parser 使用同一源码和测试矩阵，HOST 不需要 CUDA 或 daemon；
- 本阶段保留 `ParsedConfig` flat compatibility 字段和 `TuttiRuntime` public allocation
  字段，按计划分别留给 P2 生命周期拆分和 P6 API 收口；
- P2 必须继续以 `canonical_storage` 为静态输入，不得重新解析或混用 legacy YAML 字段。

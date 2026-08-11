# Storage Config、Backend 与 Resource 阶段 3 实施记录

## 1. 阶段范围与结论

本阶段实现
`doc/impl/storage-config-backend-resource-implementation-plan.md` 的 P3：增加最小公共
`Resource` SPI，将 NVMe provider client、allocation metadata 和 Release guard 收敛到
`NvmeResource`，并使 config loader 只通过 NVMe-specific 只读 view 装配 resolver 和
DataPath。

阶段基线为 P2 commit `36b0082`，分支为 `fix/multi-gpu`，测试日期为 2026-08-11 UTC。
按本阶段实施期间的目录约束，具体 resource 实现统一位于
`tutti/resource/<resource-type>/`；NVMe 实现位于 `tutti/resource/nvme/`，未放回
config parser 目录。

P3 exit gate 通过：纯主机 Resource contract、CUDA loader contract、HOST/CUDA 非硬件
回归全部通过；`NvmeResource` 的私有 PImpl 成为 NVMe client 和 allocation 的唯一
owner；公共 `Resource`、config loader options、NVMe 公共头和 `TuttiRuntime` 均不声明
client、allocation、lease state、inspection 或 transport 接口。NVMe allocation 观测只保留
在 resource internal test access 中。使用
`config/local/daemon_2_disk.yaml` 的单 accelerator canonical loader 实机测试完成两盘
local I/O 和 striped write/reload/read，共 224 项检查、0 失败，每次 shutdown 后两盘
reservation ledger 均精确回到基线。默认双 accelerator 验收模式因本机仅一块 L40S 仍按
约定返回 `SKIP(77)`，未将缺失的第二块 accelerator 报告为通过。

## 2. 计划项与本次实现对比

| P3 计划项 | 本次实现 | 结果 |
| --- | --- | --- |
| 公共最小 `Resource` 接口 | 新增 `ResourceState`、`ResourceCapabilities`、`ResourceInfo` 和 `Resource`；接口只含 capability、initialize、shutdown、info | 完成 |
| 不修改 `ResourceProvider` | 未改 DataPath SPI 的 `ResourceProvider` 或 `RuntimeComponents::resources` | 完成 |
| NVMe 私有状态收敛 | 删除 `RuntimeResourceClient`；`NvmeResourceClient`、`RuntimeNvmeAllocation`、`NvmeLeaseState` 和 `NvmeResourceInspection` 只在 `nvme_resource_internal.h` 中声明，config/Runtime/public headers 不可见 | 完成 |
| 按具体类型组织 Resource | 新增 `tutti/resource/nvme/`，集中 NVMe spec、client、view、gRPC adapter 和生命周期实现 | 完成 |
| `NvmeResource` 接管 allocation | `initialize()` 内部完成 snapshot、Acquire、立即记录 lease，再校验 metadata；任何 Acquire 后失败都 Release 一次 | 完成 |
| NVMe typed read-only views | `resolver_view()` 与 `datapath_view()` 返回分离的值拷贝，只含各自构造需要的字段 | 完成 |
| 状态机与 Release guard | 通用状态使用 `CREATED/INITIALIZED/SHUTTING_DOWN/STOPPED/FAILED`；lease 使用 `NONE/ACQUIRED/RELEASING/RELEASED` | 完成 |
| allocation/client 不再平行存储 | 删除 `TuttiRuntime::resource_client`、`allocation_id` 和 `allocation_slices`；Runtime 私有拥有一个阶段性 `Resource` | 完成 |
| 只读诊断 | 产品 API 只保留通用 `ResourceInfo` 和 NVMe typed views；allocation/lease snapshot 仅由 NVMe internal testing access 返回 copy，`TuttiRuntime` 不感知该类型 | 完成 |
| Resource contract | 新增硬件无关 `tutti_resource_contract_test`，覆盖成功、Acquire 失败、空 ID、缺 metadata、Release 状态/异常、析构和幂等性 | 完成 |
| 通用 loader 注入 seam | `LoadTuttiConfigOptions` 只注入 `ResourceSpec -> Resource` factory；fake client 只能通过 NVMe internal test access 注入 | 完成 |
| 后续 factory 失败回滚 | loader runtime factory 失败测试继续证明 Acquire 后 Release 恰好一次，allocation owner 始终存在于回滚链中 | 完成 |

P3 没有提前实现 P4 的 resource ID registry、多 Resource 初始化顺序或跨 Resource 逆序
回滚。`TuttiRuntime` 本阶段私有拥有一个 `Resource`，保持当前单 backend 产品约束；P4 再
将其扩展为按 ID 的 registry。

## 3. 业务逻辑变化

### 3.1 NVMe allocation ownership

迁移前的 ownership 是 `TuttiRuntime` 中三个可独立修改的平行字段：client、allocation ID
和 slices。迁移后，loader 的资源路径为：

```text
ResourceSpec + runtime accel_id
  -> NvmeResourceSpec
  -> NVMe module factory
       -> private NvmeResourceClient
       -> NvmeResource(private PImpl)
  -> NvmeResource::initialize()
       -> ListAccelerators / ListNvmeResources
       -> AcquireNvmeSlices
       -> lease = ACQUIRED，立即拥有 allocation
       -> 校验 allocation ID 和 slice 基础 metadata
  -> resolver_view() + datapath_view()
  -> TuttiRuntime::adopt_resource()
  -> resolver/DataPath/StorageRuntime factory
```

Acquire 返回成功后，allocation 立即写入 `NvmeResource`，lease 进入 `ACQUIRED`。空
allocation ID、空 slices、accelerator/ACL 不符，或 BDF、chrdev、block、backing、view、
namespace、LBA、BAR0、MDTS、granted queues 任一必需 metadata 缺失，都会在返回错误前占用
唯一 Release 槽。Release 返回错误或抛异常时，Resource 进入 `FAILED`，client 被销毁，后续
shutdown/析构不重试同一个 allocation。

Acquire 本身失败时没有 lease，因此不调用 Release。重复 `initialize()` 返回 `BUSY` 且
不会二次 Acquire；正常 shutdown 和析构兜底共用同一 at-most-once release guard。

### 3.2 公共接口与 NVMe 私有接口

公共 `Resource` 只暴露 config ID、逻辑 type、通用状态和稳定 capabilities。它不暴露
endpoint、client、allocation ID、slice、PCI BDF 或设备路径，也没有通用
`ResourceClient` 抽象。NVMe 公共头只声明 typed spec、typed views 和 `NvmeResource`
factory；allocation/slice、lease state、inspection、provider snapshot、
`NvmeResourceClient`、client factory 和 fake/inspection 测试入口只存在于同目录的 private
internal header。`NvmeResource` 通过 PImpl 隐藏 client、allocation 和 release guard，config
loader 与 `TuttiRuntime` 只能获得通用 `Resource` ownership。

`NvmeResource` 的两个具体 view 是独立值拷贝：

| view | 字段用途 |
| --- | --- |
| resolver view | ordered device ID、PCI BDF、block/backing/view path、namespace、logical block size |
| DataPath view | ordered device ID、accel ID、chrdev、namespace、logical block size、BAR0、MDTS、granted queues |

loader 不再读取或拆分 allocation。allocation ID、slice cardinality/order、accelerator、
ACL、必需 metadata、跨 slice block size 和 granted queue 上限均在 `NvmeResource`
初始化期间校验；Local 和 striped resolver/DataPath 只消费对应 typed view。修改测试取得的
view 或 `ResourceInfo` 不会修改 `NvmeResource` 内部 metadata 或 lease。

### 3.3 Runtime shutdown 和诊断

关闭顺序保持 P2 的 first-error 和幂等语义，但最后一步从直接 provider Release 改为通用
Resource shutdown：

```text
StorageRuntime shutdown/destroy
  -> resolver destroy
  -> DataPath destroy
  -> Resource::shutdown()
  -> TuttiRuntime STOPPED
```

`TuttiRuntime` 不再保存 NVMe client/allocation 平行字段，也不包含 NVMe dynamic cast、
lease state 或 allocation inspection。硬件 E2E 通过 `TuttiRuntimeTestingAccess` 取得通用
`Resource` borrowed pointer，再在测试代码中显式进入 NVMe internal testing access；该路径
不属于产品 API，不能取得 client 或重新 Release。

### 3.4 Hardware E2E 测试入口

`tutti_runtime_bundle_loader_contract_test` 的临时配置改为真正的 canonical
`storage.resources/resolvers/datapaths/backends`，不再以 legacy flat YAML 冒充 canonical。
新增显式 `--single-accelerator` 模式，使只有一块 accelerator 的机器可以执行 A0 范围内的
两盘 local 和 striped correctness 测试；默认模式仍要求两块 accelerator，P7 的完整双
accelerator gate 没有放宽。

hardware E2E 直接使用 `NvmeServiceClient` 观察 daemon ledger，因此测试 target 现在显式
链接 `nvmeservice`，不再依赖 `tutti_config` 意外传递 transport include/link dependency。

## 4. 文件变化

| 文件 | 变化 |
| --- | --- |
| `tutti/include/tutti/resource.h` | 新增硬件无关公共 Resource SPI、通用状态、capabilities 和 info snapshot |
| `tutti/resource/CMakeLists.txt` | 新增按具体 resource 类型进入子目录的构建入口 |
| `tutti/resource/nvme/CMakeLists.txt` | 新增 `tutti_nvme_resource` target；仅在 nvmeservice target 存在时启用 gRPC adapter |
| `tutti/resource/nvme/nvme_resource.h` | 新增 NVMe typed spec、resolver/DataPath view、PImpl `NvmeResource` 和 production factory；不声明 client、allocation、lease 或 inspection |
| `tutti/resource/nvme/nvme_resource_internal.h` | 私有声明 allocation/slice、lease state、inspection、provider snapshot、`NvmeResourceClient`、client factory 和 test access |
| `tutti/resource/nvme/nvme_resource.cpp` | 实现私有 PImpl、snapshot/Acquire、metadata 校验、只读 view、状态机、Release guard 和析构兜底 |
| `tutti/resource/nvme/nvme_resource_grpc.cpp` | 将原 config loader 中的 gRPC adapter 和 allocation RAII map 迁入 NVMe resource 模块 |
| `tutti/include/tutti/tutti_runtime.h` | 删除 client/allocation 平行字段及 NVMe allocation/inspection 类型；增加私有 Resource ownership、adopt 和只读 info 接口 |
| `tutti/tutti_runtime/tutti_runtime.cpp` | shutdown 改为组件销毁后调用 `Resource::shutdown()`；不再包含或 dynamic cast 具体 NVMe Resource |
| `tutti/tutti_runtime/tutti_runtime_internal.h` | 增加仅供测试使用的通用 Resource borrowed access，不包含 NVMe 类型 |
| `tutti/config/tutti_config.h` | loader 注入 seam 改为通用 `ResourceSpec -> Resource` factory，不包含 NVMe client 头或类型 |
| `tutti/config/tutti_config.cpp` | 删除 gRPC/client、snapshot RPC 和 allocation inspection；经 NVMe module factory 创建后仅持有通用 Resource ownership，使用 typed views 装配组件 |
| `tutti/config/CMakeLists.txt` | full loader 链接 `tutti_nvme_resource`，不再公开传递 nvmeservice dependency |
| `tutti/CMakeLists.txt` | 注册 resource 模块和全 profile Resource contract test |
| `tests/resource_contract/CMakeLists.txt` | 新增硬件无关 Resource contract target |
| `tests/resource_contract/resource_contract_test.cpp` | 新增 36 项 Resource/NVMe ownership、状态、view 和失败回滚检查 |
| `tests/config_loader/config_loader_test.cpp` | fake client 只经 NVMe internal test access 构造 Resource；config seam 仅返回通用 Resource；通过 info/state、外部 client 计数和 shutdown 顺序断言生命周期 |
| `tests/runtime_bundle_loader_contract/CMakeLists.txt` | 显式链接测试直接使用的 `nvmeservice` |
| `tests/runtime_bundle_loader_contract/runtime_bundle_loader_contract_test.cpp` | allocation 观测迁移到 NVMe internal test access，产品 Runtime 不暴露 inspection；临时 YAML 改 canonical；增加单 accelerator 实机模式 |
| `doc/impl/storage-config-backend-resource-phase-3.md` | 新增本阶段实现、业务变化、文件变化和验证证据 |

本阶段未修改 daemon YAML schema、protobuf/gRPC RPC、
`config/local/daemon_2_disk.yaml`、`StorageRuntime` scheme/key 路由、resolver/DataPath SPI 或
公共 `ResourceProvider`。

## 5. 软件验证结果

### 5.1 Resource 和 loader 定向验证

| target/场景 | 结果 | 原始结果摘要 |
| --- | --- | --- |
| HOST `tutti_resource_contract_test` | PASS | `passed: 36`, `failed: 0` |
| CUDA `tutti_resource_contract_test` | PASS | `passed: 36`, `failed: 0` |
| CUDA `tutti_config_loader_test` | PASS | `passed: 147`, `failed: 0` |
| runtime factory 注入失败 | PASS | Acquire 1 次，Resource owner 回滚时 Release 1 次 |
| Acquire 失败 | PASS | Acquire 1 次，Release 0 次，Resource 状态 `FAILED` |
| 空 allocation ID | PASS | initialize 失败，Release 尝试 1 次，析构不重复 |
| 缺 slice metadata | PASS | initialize 失败，Acquire 后 Release 1 次 |
| Release status/exception | PASS | 分别传播 `DEVICE_ERROR`/转换为 `INTERNAL`，Release 均只调用 1 次 |
| 正常 shutdown/析构兜底 | PASS | 显式两次 shutdown 或仅析构均只 Release 1 次 |
| private state boundary | PASS | public/include、config 和 Runtime 源码检索不到 NVMe allocation、lease 或 inspection；internal snapshot 修改不影响 Resource 状态 |

### 5.2 完整非硬件回归

| 命令 | 结果 | 原始结果摘要 |
| --- | --- | --- |
| `cmake --build build/host --parallel 8` | PASS | 新 Resource target、parser 和全部 HOST tests 构建成功 |
| `ctest --test-dir build/host --output-on-failure -j 8` | PASS | 19/19，通过时间 0.24 秒 |
| `cmake --build build/cuda --parallel 8` | PASS | config loader、Resource、daemon 和 hardware E2E target 全部构建成功 |
| `ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8` | PASS | 21/21，通过时间 0.48 秒 |
| `git diff --check` | PASS | 无 whitespace error |

完整 CUDA 构建首次发现 hardware E2E 直接包含 `nvmeservice_client.h`，但依赖此前由
`tutti_config` 偶然传递。测试 target 改为显式链接 `nvmeservice` 后，完整构建和回归通过；
产品 target 没有重新公开 transport dependency。

## 6. 基于硬件的验证结果

### 6.1 环境与基线

- daemon 配置：`config/local/daemon_2_disk.yaml`，endpoint `127.0.0.1:50051`；
- accelerator：1 块 NVIDIA L40S，仅 `accel_id=0`，view root `/mnt/gpu0`；
- device 0：BDF `0000:b1:00.0`，chrdev `/dev/ssnvme0`，block
  `/dev/snvme0n1`，backing `/mnt/nvme0`，view `/mnt/gpu0/ssnvme0`，namespace 1，
  logical block 4096，BAR0 16384，MDTS 131072，capacity/available queues 23；
- device 1：BDF `0000:e3:00.0`，chrdev `/dev/ssnvme1`，block
  `/dev/snvme1n1`，backing `/mnt/nvme1`，view `/mnt/gpu0/ssnvme1`，namespace 1，
  logical block 4096，BAR0 16384，MDTS 131072，capacity/available queues 72；
- 初始 ledger：device 0 `reserved=0 available=23`，device 1
  `reserved=0 available=72`。

mount/view 由 root 管理且普通用户不可写，测试按实施计划使用用户授权的 sudo 运行；密码
未写入命令参数、日志或本文，也未修改 daemon 权限。

### 6.2 Canonical loader Resource/I/O 场景

命令使用 `tutti_runtime_bundle_loader_contract_test --single-accelerator --accel0 0
--device0 0 --device1 1 --queues 4`。所有临时应用配置均使用 canonical storage graph，
不包含 BDF、device path、allocation ID 或 lease 字段。

| 场景 | 结果 | Resource metadata、I/O 与 ledger 证据 |
| --- | --- | --- |
| explicit device 0、4 queues | PASS | NVMe internal test snapshot 返回 device 0 的实际 BDF/chrdev/block/backing/view/ns/LBA/BAR0/MDTS/grant；local write/read byte-exact；运行中 ledger `4/19, 0/72`，shutdown 后回到 `0/23, 0/72` |
| explicit device 1、4 queues | PASS | 返回 device 1 的全部实际 metadata；local write/read byte-exact；运行中 ledger `0/23, 4/68`，shutdown 后回到 `0/23, 0/72` |
| striped `[0,1]`、每盘 4 queues | PASS | 一次 allocation 返回有序 device 0、1 slices，resolver shard/DataPath descriptor 使用同一顺序；跨 stripe write 后 shutdown，再次 allocation/reload/read 验证持久性；两次运行中均为 `4/19, 4/68`，每次均同时回到 `0/23, 0/72` |
| E2E 总结果 | PASS | `phase5 checks=224 failures=0 result=PASS` |
| 默认双 accelerator 模式 | SKIP | exit 77，明确输出 requested accelerator set unavailable；本机只有 accelerator 0 |

该结果证明 loader 使用 daemon RPC 返回的 path、BDF、namespace、LBA、BAR0、MDTS 和
granted queues，而非由 device ID 或 daemon YAML 推导。Striped write 和 restart-read 使用
两次不同 allocation ID，均由一个 `NvmeResource` 同时释放两盘 queue reservation。

### 6.3 daemon 与文件清理

测试结束后的独立 list snapshot 与初始 ledger 完全一致。随后仅发送一次 Ctrl-C，daemon
进程正常退出。清理检查结果：

- 无 `tutti_daemon`/`nvmeservice_daemon` 进程和 50051 listener；
- 无 `/mnt/nvme0`、`/mnt/nvme1` 或 accelerator view mount；
- `/mnt/gpu0/ssnvme0`、`/mnt/gpu0/ssnvme1` view symlink 已删除；
- `/dev/ssnvme0`、`/dev/ssnvme1`、`/dev/snvme0n1`、`/dev/snvme1n1` 已删除；
- 无 `tutti_phase5_*` scratch file 或临时 config directory 残留。

## 7. Exit Gate 与后续约束

- 独立 Resource contract 在 HOST/CUDA 均通过，daemon 未连接的测试可运行；
- NVMe client、allocation、lease 和 Release channel 的唯一 owner 是 `NvmeResource`
  私有 PImpl；client 类型不出现在 config 或公共头；
- `NvmeLeaseState`、`NvmeResourceInspection` 和 allocation/slice 类型只存在于 NVMe
  internal header；`TuttiRuntime` 不包含 NVMe 类型或具体 Resource dynamic cast；
- `TuttiRuntime` 不再有 client/allocation 平行字段，正常/失败路径通过同一 Resource
  ownership cleanup；
- 公共 `Resource` 没有 transport-specific 字段，NVMe views 是具体实现 API 和值拷贝；
- canonical loader 和现有 local/striped `StorageRuntime` 路由行为无回归；
- 单 accelerator 双盘实机 correctness 和 ledger 回收通过，默认双 accelerator gate 明确
  SKIP，不扩大为 P7 完成。

P4 必须在此基础上把 `TuttiRuntime` 的单 Resource ownership 扩展为私有
`resource_id -> unique_ptr<Resource>` registry，记录初始化顺序，并在 Acquire 前完成
resource/backend contract cardinality 等静态图校验。allocation 返回后的 slice
cardinality/order、跨 slice block-size 和 granted queue 校验已由 `NvmeResource` 私有完成。
P4 不得重新引入 Runtime 平行 allocation 字段，也不得让 backend/resolver/DataPath 直接持有
或替换 `NvmeResourceClient`。

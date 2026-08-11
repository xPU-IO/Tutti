# Storage Config、Backend 与 Resource 阶段 5 实施记录

## 1. 阶段范围与结论

本阶段实现
`doc/impl/storage-config-backend-resource-implementation-plan.md` 的 P5：把 P4 loader 中
按字符串分支的 local/striped 组件装配迁入显式 backend factory registry；contract registry
同时固定 resolver type、scheme、DataPath type、Resource type、cardinality、resolver type ID、
payload type ID/API version 和推荐 DataPath key。配置仍只声明逻辑 contract，不接受 payload
identity、API version 或 Runtime key。

阶段基线为 P4 commit `96abdf4`，分支为 `fix/multi-gpu`，测试日期为 2026-08-11 UTC。

P5 exit gate 通过：`ext4-local-nvme` 和 `striped-local-nvme` 都由 registry 选择具体 factory，
只从 `NvmeResource` 的 resolver/DataPath typed views 注入路径、PCI、namespace、BAR0、MDTS、
accelerator 和 granted queues；striped factory 保留 allocation 的有序 shard views，并把最小
shard MDTS 注入唯一的 `StripedDataPath`。factory product 在注册组件和调用
`StorageRuntime::create()` 前再次与 contract 和实际 Resource views 逐 shard 对照；错误 payload
version、cardinality/order/namespace 不一致、DataPath initialize 失败和 Runtime factory 失败都
fail-closed，并由 `TuttiRuntime` 回滚唯一 Resource lease。

HOST 20/20、CUDA 非硬件 22/22 回归通过；两个 profile 的现有
`storage_runtime_contract` 均保持 40/40，多 scheme/multi-key 核心路由能力没有收缩。使用
`config/local/daemon_2_disk.yaml` 的单 accelerator 双盘 local/striped 实机闭环通过 224 项检查，
最终 queue ledger、daemon、端口、mount、view、device node 和 scratch 文件均回到基线。

## 2. 计划项与本次实现对比

| P5 计划项 | 本次实现 | 结果 |
| --- | --- | --- |
| backend factory registry | 新增两项 registry，逻辑 contract 分别映射到 local 与 striped factory；未实现的 `memfs` 没有 product factory | 完成 |
| 完整 contract 矩阵 | contract entry 固定 resolver type/scheme、DataPath type、Resource type、cardinality、resolver type ID、payload type/version 和推荐 key | 完成 |
| local typed views | factory 要求实际对象为 `NvmeResource`，从单 slice resolver view 创建 `LocalFileResolver`，从 DataPath view 创建 `LocalNvmeDataPath` | 完成 |
| striped typed views | factory 保留有序 N shard views，创建 N 个 local shard resolver、一个 `StripedResolver`、N 个 descriptor 和一个 `StripedDataPath` | 完成 |
| 共享语义来源 | `stripe_unit` 只从 `BackendSpec` 注入 resolver/product contract；不从 Resource slice 或 DataPath config 推导 | 完成 |
| transport 字段来源 | chrdev、PCI、BAR0、namespace、block size、MDTS、accel 和 granted queues 只从 `NvmeDataPathResourceView` 注入；resolver 路径/namespace 只从 resolver view 注入 | 完成 |
| 逐级 fail-closed | factory context、typed views、product payload contract、projection、DataPath capabilities、Runtime factory 返回值逐级校验 | 完成 |
| 失败回滚 | payload mismatch、DataPath initialize 和 Runtime create 注入均证明 resolver/DataPath 析构且 allocation Release 恰好一次 | 完成 |
| backend manifest | 继续使用 P4 的只读 `backend_id -> contract/resolver_id/datapath_id/resource_id` manifest；失败不返回半初始化 Runtime | 完成 |
| Runtime 路由边界 | `RuntimeComponents` 仍只包含一个顶层 scheme binding 和一个 contract key binding，不注册 Resource 或 backend ID | 完成 |

## 3. 业务逻辑变化

### 3.1 Contract 与 factory registry

迁移后的静态和实例装配路径为：

```text
backend.contract
  -> StorageContract
       resolver type + supported scheme
       DataPath type + Resource type
       cardinality
       resolver type ID + payload type ID/API version
       recommended DataPath key
  -> BackendFactoryRegistration
       ext4-local-nvme factory
       striped-local-nvme factory
```

`tutti_config_parse.cpp` 不再保存一份局部 contract 数组。唯一 contract registry 直接引用
binding package 的 identity 常量，配置对象 ID 不会进入 payload ABI 或 Runtime key。parser 在
ResourceFactory/Acquire 前拒绝未知 contract、memfs/local-nvme 混搭、不支持的 scheme、错误
selection/cardinality、重复 scheme 和重复 contract key。

### 3.2 Local backend

`ext4-local-nvme` factory 的输入是一个已初始化且 capability 匹配的 `NvmeResource`：

```text
NvmeResource
  -> resolver_view[0]
       pci_bdf + block_path + namespace + block size
       -> LocalFileResolver
  -> datapath_view[0]
       pci_bdf + chrdev + BAR0 + namespace + block size
       + MDTS + accel + granted queues
       -> LocalNvmeDataPath
```

factory 先证明两个 view 都只有一个 slice，且 device ID、PCI、namespace 和 block size 一致。
`LocalNvmeDataPath::open()` 现在还会把 resolver payload 的 PCI、namespace ID 和 block size 与
factory 注入的 DataPath resource identity 再次比较，防止一个合法 payload 被错误 controller
实例消费。

### 3.3 Striped backend

`striped-local-nvme` factory 不压缩 ResourceInstance：

```text
one NvmeResource allocation
  -> ordered resolver views [0..N)
       -> N LocalFileResolver
       -> one StripedResolver(stripe_unit)
  -> ordered DataPath views [0..N)
       -> N DeviceDescriptor
       -> one StripedDataPath(min(shard MDTS))
```

factory 和 product validator 都逐项检查两组 view 的 device ID、PCI、namespace、block size 与
请求顺序；`effective_max_data_size` 必须等于所有 DataPath shard 的最小 MDTS。每个 striped
descriptor 保存期望 PCI/namespace/block-size，`StripedDataPath` 在建立 shard handle 时再次
与 resolver 的 ext4 shard payload 对照。`stripe_unit` 只来自 backend config，并由外层
`StripedResolver` 写入 top-level payload；Runtime 仍只看到一个 `striped` scheme 和一个
`striped-local-nvme` key。

### 3.4 装配、manifest 与失败边界

loader 当前顺序为：

```text
canonical static validation
  -> lookup StorageContract + BackendFactoryRegistration
  -> ResourceFactory/initialize/adopt by resource ID
  -> invoke backend factory with typed Resource
  -> validate payload contract and actual Resource projections
  -> register owned resolver and DataPath
  -> register read-only BackendManifest
  -> StorageRuntime::create(scheme/key bindings)
```

factory 返回 null 组件、错误 scheme/key、错误 resolver/payload identity 或 API version、错误
shard count/order/namespace、非最小 MDTS、丢失/多余 `stripe_unit`、错误 accelerator binding
均在 DataPath initialize 前失败。Runtime factory 抛异常或返回 success/null 也被转换为明确
失败。任何 Acquire 之后的失败都由局部 `TuttiRuntime` 析构按既有关闭顺序清除 backend
引用、销毁 resolver/DataPath、最后 Release Resource；不会返回可访问的半初始化 manifest。

## 4. 文件变化

| 文件 | 变化 |
| --- | --- |
| `tutti/config/storage/backend/contract_registry.cpp` | 新增唯一 contract registry，identity/version/key 直接取 binding constants |
| `tutti/config/storage_config.h` | 扩展 `StorageContract` 的 scheme、resolver type ID、payload type/version 字段 |
| `tutti/config/backend_factory.h` | 定义 loader-private factory context、product/projection、registration 和校验入口 |
| `tutti/config/backend_factory.cpp` | 实现 local/striped registry factories、typed view 对照、MDTS/stripe/cardinality/payload contract 校验 |
| `tutti/config/tutti_config.cpp` | 删除 `add_local_components()`/`add_striped_components()` 字符串分支，改为 registry factory product 装配；统一 Runtime factory 异常/null 校验 |
| `tutti/config/tutti_config.h` | 增加 backend factory 故障注入 seam，生产默认使用 registry |
| `tutti/config/tutti_config_parse.cpp` | contract lookup 迁到唯一 registry，并校验 resolver 实现支持的 scheme |
| `tutti/resource/nvme/nvme_resource.h/.cpp` | DataPath typed slice view 增加来自 allocation 的 PCI identity |
| `tutti/data_paths/local_nvme/local_nvme_data_path.h/.cpp` | 保存 factory 注入的 PCI，并在 open 时校验 payload PCI/namespace/block-size |
| `tutti/data_paths/striped_local_nvme/striped_data_path.h/.cpp` | descriptor 保存 PCI，并逐 shard 校验 ext4 payload namespace identity |
| `tutti/presets/local_nvme_preset.cpp` | programmatic striped preset 同步注入 descriptor PCI identity |
| `tutti/config/CMakeLists.txt` | parse target加入 contract registry，full loader target 加入 backend factory |
| `tests/storage_config_contract/storage_config_contract_test.cpp` | 核对两组 NVMe contract 的完整 compiled identity/cardinality/key 矩阵 |
| `tests/resource_contract/resource_contract_test.cpp` | 核对 resolver/DataPath typed views 的 PCI/namespace/block-size 一致性 |
| `tests/config_loader/config_loader_test.cpp` | 增加真实 factory 类型、顶层 binding、ordered projections、最小 MDTS、错误 payload version 和两类失败回滚测试 |
| `tests/striped_local_nvme_contract/striped_local_nvme_contract_test.cpp` | 硬件 descriptor fixture 同步注入 PCI identity |
| `doc/impl/storage-config-backend-resource-phase-5.md` | 本阶段实施与软硬件验证记录 |

本阶段未修改 daemon YAML/schema、protobuf/gRPC、allocation RPC、
`config/local/daemon_2_disk.yaml`、公共 `ResourceProvider`、`StorageRuntime` 路由模型或
canonical 配置结构；没有把 path、BAR0、MDTS、queue、payload ID/version/key 加入 YAML。

## 5. Contract 与故障注入结果

| 场景 | 结果 | Acquire/initialize/cleanup 证据 |
| --- | --- | --- |
| canonical local 正向 | PASS | 一个 `LocalFileResolver`、一个 `LocalNvmeDataPath`、一个 `file` binding、一个 `local-nvme-ext4` binding；resolver/DataPath projection identity 一致 |
| canonical striped 正向 | PASS | 两个有序 shard resolver、一个 `StripedResolver`、两个有序 descriptor、一个顶层 scheme/key；测试 shard MDTS `262144/65536` 得到有效 MDTS `65536` |
| memfs/local-nvme 混搭 | PASS | parser contract matrix 拒绝，ResourceFactory/list/Acquire/Release 和 Runtime factory 均 0 |
| local request/return 多片 | PASS | 静态 selection 拒绝或 Acquire 后 factory cardinality 拒绝；后者 Acquire 1、Release 1、Runtime factory 0 |
| striped request/return 单片 | PASS | 静态 selection 拒绝或 Acquire 后 Resource/factory cardinality 拒绝；后者 Acquire 1、Release 1、Runtime factory 0 |
| 未知 contract | PASS | parser 在 ResourceFactory/Acquire 前拒绝 |
| 错误 payload API version | PASS | Acquire 1、Release 1、DataPath initialize 0、Runtime factory 0；临时 resolver/DataPath 各析构 1 次 |
| 重复 scheme/key | PASS | parser 在 ResourceFactory/Acquire 和 DataPath initialize 前拒绝 |
| DataPath initialize 注入失败 | PASS | initialize 1；resolver/DataPath 各析构 1；Resource Acquire 1、Release 1；没有 Runtime 或悬空组件 |
| Runtime factory 注入失败 | PASS | Runtime factory 1；resolver/DataPath 各析构 1；Resource Release 1；没有返回 manifest/Runtime |
| Runtime factory success/null | PASS | loader 显式拒绝，不把 null Runtime 作为成功结果 |

## 6. 软件验证结果

### 6.1 定向 contract

| target | 结果 | 原始摘要 |
| --- | --- | --- |
| CUDA `tutti_config_loader_test` | PASS | `passed: 189`, `failed: 0` |
| HOST/CUDA `tutti_storage_config_contract_test` | PASS | `storage config checks=124 failures=0` |
| HOST/CUDA `tutti_resource_contract_test` | PASS | `passed: 40`, `failed: 0` |
| HOST/CUDA `tutti_storage_runtime_contract_test` | PASS | `All 40 storage runtime contract tests passed.`；包含既有 multi-scheme/multi-key case |
| `git diff --check` | PASS | 无 whitespace error |

### 6.2 完整非硬件回归

| 命令 | 结果 | 原始摘要 |
| --- | --- | --- |
| `cmake --build build/host --parallel 8` | PASS | parser、Resource、Runtime registry 和全部 HOST targets 构建成功 |
| `ctest --test-dir build/host --output-on-failure -j 8` | PASS | 20/20，0 failed |
| `cmake --build build/cuda --parallel 8` | PASS | factory、loader、daemon、硬件 E2E 和全部 CUDA targets 构建成功 |
| `ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8` | PASS | 22/22，0 failed |

## 7. 基于硬件的验证结果

### 7.1 环境与基线

- daemon 配置：`config/local/daemon_2_disk.yaml`，endpoint `127.0.0.1:50051`；
- accelerator：1 块 NVIDIA L40S，`accel_id=0`，PCI `0000:4B:00.0`，view root
  `/mnt/gpu0`；
- device 0：BDF `0000:b1:00.0`，chrdev `/dev/ssnvme0`，block
  `/dev/snvme0n1`，backing `/mnt/nvme0`，namespace 1，logical block 4096，BAR0 16384，
  MDTS 131072，capacity/available queues 23；
- device 1：BDF `0000:e3:00.0`，chrdev `/dev/ssnvme1`，block
  `/dev/snvme1n1`，backing `/mnt/nvme1`，namespace 1，logical block 4096，BAR0 16384，
  MDTS 131072，capacity/available queues 72；
- 测试前无 daemon、50051 listener、NVMe mount、view symlink 或 snvme device node；
- 初始 ledger：device 0 `reserved=0 available=23`，device 1
  `reserved=0 available=72`。

daemon 和 E2E 按用户授权使用 sudo；密码没有写入命令参数、测试输出或本文，也没有修改
daemon 配置、设备权限或 mount 策略。机器只有一个 accelerator，因此按既有测试的
`--single-accelerator` 模式验证 P5 所需的双盘 local/striped backend；这不是 P7 的双
accelerator A/B 全矩阵声明。

### 7.2 Canonical loader I/O 与 backend factory

执行：

```bash
build/cuda/bin/tutti_runtime_bundle_loader_contract_test \
  --single-accelerator --accel0 0 --device0 0 --device1 1 --queues 4
```

测试生成的应用配置全部使用 canonical logical contract 与 ID 引用，不包含 BDF、device
path、payload type/version、DataPath key、allocation ID 或 lease 字段。

| 场景 | 结果 | metadata、I/O 与 ledger 证据 |
| --- | --- | --- |
| explicit device 0、4 queues | PASS | typed views 返回 device 0 的实际 PCI/path/ns/LBA/BAR0/MDTS/grant；local write/read byte-exact；运行中 `4/19, 0/72`，shutdown 后 `0/23, 0/72` |
| explicit device 1、4 queues | PASS | typed views 返回 device 1 的实际 metadata；local write/read byte-exact；运行中 `0/23, 4/68`，shutdown 后 `0/23, 0/72` |
| striped `[0,1]` write | PASS | 一次 allocation 按 device 0、1 返回两个有序 slice；两盘同时 `4/19, 4/68`；跨 64 KiB stripe 与 mixed batch byte-exact；shutdown 后同时回到基线 |
| striped restart-read | PASS | 新 allocation 保持 device 0、1 resolver/DataPath 顺序，读取前次写入 byte-exact；shutdown 后再次同时回到基线 |
| E2E 汇总 | PASS | `phase5 checks=224 failures=0 result=PASS` |
| 最终 ledger | PASS | device 0 `reserved=0 available=23`；device 1 `reserved=0 available=72` |

该结果同时验证新增 DataPath namespace identity 检查没有拒绝真实 daemon metadata；PCI、
namespace 和 block size 来自同一个 Resource slice 的两类 typed view。两块实盘 MDTS 都为
131072，因此实机 striped 有效 MDTS 为 131072；不同 shard MDTS 的最小值逻辑由硬件无关
factory test 使用 `262144/65536 -> 65536` 单独覆盖。

### 7.3 daemon 与文件清理

E2E 后独立 `ListNvmeResources` snapshot 与初始 ledger 完全一致。随后只向已确认 PID 发送
一次 SIGTERM，daemon 输出：

```text
Shutting down...
mount_manager: unmounted /mnt/nvme0
mount_manager: unmounted /mnt/nvme1
tutti_daemon exited cleanly.
```

最终独立检查结果：

- 无 `tutti_daemon` 或 loader E2E 进程，无 50051 listener；
- 无 `/mnt/nvme0`、`/mnt/nvme1` 或 accelerator view mount；
- `/mnt/gpu0/ssnvme0`、`/mnt/gpu0/ssnvme1` view symlink 不存在；
- `/dev/ssnvme0`、`/dev/ssnvme1`、`/dev/snvme0n1`、`/dev/snvme1n1` 不存在；
- 无 `tutti_phase5_*` scratch file 或 `tutti_cfg_*` 临时配置残留。

## 8. Exit Gate 与 P6 边界

- backend 关系由 logical contract registry、factory registration 和只读 manifest 显式表达；
- contract/type/scheme/cardinality/payload identity/version/key 均 fail-closed；
- controller path、PCI、BAR0、namespace、MDTS、accelerator、granted queues 和 resolver path
  只经具体 `NvmeResource` typed views 注入；
- loader 不读取 public allocation slices，也不根据 allocation slice 数量选择 backend；它先按
  `backend.contract` 选择 factory，再校验 typed view cardinality；
- local/striped 只向 `StorageRuntime` 发布一个顶层 scheme 和一个 DataPath key；核心 Runtime
  的多 scheme/multi-key 能力保持不变；
- factory、DataPath initialize、Runtime create 任一步失败均无悬空 resolver/DataPath，
  allocation Release 恰好一次；
- `TuttiRuntime` manifest 仍是值快照，包含 backend ID、contract 和三个配置引用 ID，不参与
  I/O 热路径。

P6 再迁移 canonical 默认示例、收口兼容 inspection/public 字段和安装/API 表面。本阶段没有
提前删除 legacy adapter，也没有实现 `MemfsResource` 或 memfs product factory。

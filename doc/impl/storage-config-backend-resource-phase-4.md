# Storage Config、Backend 与 Resource 阶段 4 实施记录

## 1. 阶段范围与结论

本阶段实现
`doc/impl/storage-config-backend-resource-implementation-plan.md` 的 P4：把 canonical
`ResourceSpec` 经 ResourceFactory 解析为由 `TuttiRuntime` 私有 ID registry 拥有的
`ResourceInstance`，记录成功初始化顺序，并在失败或正常关闭时按该顺序逆序 shutdown。
Runtime 同时保存 backend 的 ID/contract 与 Resource/Resolver/DataPath 非 owning 指针关系，
对外只返回 `ResourceInfo` 和 `BackendManifest` 值快照。

阶段基线为 P3 commit `e430392`，分支为 `fix/multi-gpu`，测试日期为 2026-08-11 UTC。

P4 exit gate 通过：Runtime registry、初始化顺序、逆序回滚、first-error 和幂等性有
HOST/CUDA 硬件无关测试；canonical/legacy loader 的 allocation/cardinality/metadata 失败路径
都证明 fake client ledger 回到零；重复 Resource 消费在 ResourceFactory 和 Acquire 前拒绝；
HOST 20/20、CUDA 非硬件 22/22 回归通过；使用
`config/local/daemon_2_disk.yaml` 的两盘 local/striped 实机测试通过 224 项检查，最终 queue
ledger、进程、端口、mount、view、device node 和 scratch 文件均回到测试前基线。

本阶段没有增加 `resource_graph.*` 文件、`ResourceGraph` 类型、独立 graph library 或
`instantiate_resource_graph`/`validate_resource_graph` 系列入口。四组 ID 表和 reachability
只是在现有 loader 内部一次性建立的局部装配数据；P1 parser 仍负责 YAML schema 的首层
fail-closed 校验，P4 loader 只做实例化边界需要的防御性检查。

当前产品约束仍是一个 Runtime 只启用一个 backend，因此产品入口一次只会初始化一个被引用
的 logical Resource。Runtime registry 本身支持多个 ID，跨 Resource 的初始化顺序和失败回滚
由硬件无关 registry contract 验证；本阶段没有放宽多 backend 产品行为，也没有提前实现 P5
backend factory。

## 2. 计划项与本次实现对比

| P4 计划项 | 本次实现 | 结果 |
| --- | --- | --- |
| loader 建立四组 ID 表和 reachability | `tutti_config.cpp` 内部建立 resource/resolver/datapath/backend 四张局部表，校验悬空引用、contract type、全量可达性和独立 DataPath 重复消费 | 完成 |
| 只解析被引用 ResourceSpec | loader 按 backend 引用查找 ResourceSpec；registry 已存在相同 ID 时复用，不再次调用 factory/initialize | 完成 |
| 记录初始化顺序 | Resource 只有在 factory identity、capabilities、`initialize()` 和 `INITIALIZED` 状态全部通过后才进入 registry，并追加 `resource_initialization_order_` | 完成 |
| 失败逆序回滚 | loader 失败返回时 `TuttiRuntime` RAII cleanup 按成功初始化顺序逆序 shutdown；当前失败 Resource 由局部 owner 先执行幂等 cleanup | 完成 |
| selection/allocation cardinality | Acquire 前按 contract 校验请求 cardinality；Acquire 后 NVMe 校验 local 恰好 1 slice、striped 数量等于请求且 device 顺序完全一致 | 完成 |
| slice metadata | 对 accelerator、ACL、必需路径、namespace、logical block size、BAR0、MDTS、granted queues 和跨 slice block size 分别校验并返回明确错误 | 完成 |
| 未引用与重复消费 | 未引用声明继续由 parser 和 loader 双层拒绝；同一 Resource 被不同 DataPath 消费时在 ResourceFactory 前失败 | 完成 |
| Resource shutdown 时机 | Runtime 内部受控单 Resource shutdown 在 backend 指针仍存活时返回 `BUSY`；最终 shutdown 先清除 backend 指针和组件，再释放 Resource | 完成 |
| Runtime 私有 registry | 单一 `resource_` 替换为 `resource ID -> unique_ptr<Resource>`；公开查询按 ID 返回值快照，并保留单 Resource 兼容查询 | 完成 |
| backend manifest | Runtime 保存 backend ID、contract、三个配置引用 ID 和实际 Resource/Resolver/DataPath 非 owning 指针；公开 manifest 为副本 | 完成 |
| legacy adapter 行为 | legacy 配置仍先转换为 canonical storage，随后与 canonical 配置共用同一 ID lookup、ResourceFactory、registry、manifest 和组件装配路径 | 完成 |

## 3. 业务逻辑变化

### 3.1 ResourceSpec 解析与唯一 ownership

迁移前 loader 直接读取 `canonical_storage.resources.front()`，创建并初始化一个 Resource，
再放入 Runtime 单例字段。迁移后路径为：

```text
Parsed canonical storage
  -> loader 局部建立四组 ID table
  -> backend 引用、reachability、contract cardinality、重复消费校验
  -> 按 backend 引用的 resource ID：
       ResourceFactory(spec, accel_id)
       -> 校验 ResourceInfo/capabilities/CREATED
       -> Resource::initialize()
       -> 校验 identity/type/INITIALIZED
       -> TuttiRuntime.resources_[id] = unique_ptr<Resource>
       -> resource_initialization_order_.push_back(id)
  -> 按 backend resource ID 取得同一个只读 Resource 指针
```

ResourceFactory 返回 null、错误、异常、错误 identity/type/capabilities/state 或 initialize 异常
时均 fail-closed。Acquire 成功后的 allocation 仍立即由 P3 `NvmeResource` 唯一拥有；P4 没有
把 client、allocation ID 或 slices 重新暴露给 Runtime 或 backend。

### 3.2 Contract 驱动的 cardinality

loader 不再根据 allocation 返回 `slices.size()` 猜测 local 或 striped backend，而是先读取
显式 `backend.contract`：

- `ext4-local-nvme` 只接受一个 slice；
- `striped-local-nvme` 只接受 striped selection，返回数量必须等于请求的 device ID 数量，
  且 `NvmeResource` 已逐项证明 slice device ID 顺序与请求完全一致；
- allocation 返回多个 slice 不会把 local backend 静默升级成 striped；
- allocation 返回一个 slice也不会把 striped backend 降级成 local。

具体 resolver/DataPath factory 仍是 P3 的兼容装配函数；P5 再把它们迁移到 contract registry
驱动的 backend factory。本阶段只修正 Resource cardinality 的选择依据，不提前引入 P5 实体。

### 3.3 Metadata 失败与 Release

`NvmeResource` 在 Acquire 后逐项校验：

```text
allocation_id
slice count and ordered device IDs
slice accel_id and allowed_accel_ids ACL
pci_bdf / chrdev / block / backing / view paths
namespace_id / logical_block_size
bar0_size / max_data_size / granted_queues
uniform block size and granted queue upper bound
```

这些错误均在 resolver/DataPath 构造之前返回。Acquire 已成功时先消耗唯一 Release 槽，再把
错误返回 loader；析构和 Runtime 回滚不会二次 Release。accelerator/ACL/namespace/block
size/path/BAR0/MDTS/queues 不再共用一个笼统的“incomplete metadata”错误，而是返回可定位字段
类别的诊断。

### 3.4 Registry、manifest 与关闭顺序

正常关闭顺序为：

```text
StorageRuntime shutdown/destroy
  -> 清除 BackendInstance 非 owning 指针
  -> resolver destroy
  -> DataPath destroy
  -> resource_initialization_order_ 逆序 Resource::shutdown()
  -> TuttiRuntime STOPPED
```

任一 Resource shutdown 失败时返回逆序清理遇到的第一个错误，但仍继续 shutdown 更早初始化
的所有 Resource。Runtime 到达 `STOPPED` 后重复 shutdown 不会再次 Release。关闭后
`ResourceInfo` 和 `BackendManifest` 值快照仍可用于诊断，内部 backend 指针已经清空。

## 4. 文件变化

| 文件 | 变化 |
| --- | --- |
| `tutti/include/tutti/tutti_runtime.h` | 增加私有 Resource registry、初始化顺序、BackendInstance、只读 ResourceInfo/BackendManifest API；删除单 Resource owner |
| `tutti/tutti_runtime/tutti_runtime.cpp` | 实现 ID 注册/查询、backend 关系校验、绑定期间 `BUSY`、逆序 Resource shutdown、first-error 和值快照 |
| `tutti/tutti_runtime/tutti_runtime_internal.h` | 增加 loader assembly 的受控 registry/manifest 入口和 registry contract 的只读测试入口 |
| `tutti/config/tutti_config.cpp` | 在现有 loader 内建立四组 ID 表和 reachability；实现 ResourceFactory identity/state 防御、按 ID 一次初始化、contract cardinality 和 manifest 登记 |
| `tutti/config/tutti_config.h` | 显式包含 canonical storage model，消除此前经 Runtime 头的传递 include 依赖 |
| `tutti/config/tutti_config_parse.cpp` | 在 generic backend 数量错误前明确拒绝同一 Resource 被独立 DataPath 重复消费 |
| `tutti/resource/nvme/nvme_resource.cpp` | 校验 accelerator snapshot `view_root`，把 slice accelerator/ACL/path/namespace/LBA/BAR0/MDTS/queue 失败拆为明确诊断 |
| `tutti/config/CMakeLists.txt` | 将 Runtime 生命周期实现拆为所有 profile 可链接的 `tutti_runtime` target；full loader 链接该 target |
| `tests/runtime_resource_registry_contract/*` | 新增硬件无关 registry、初始化顺序、第二 Resource snapshot/factory 失败、逆序清理、重复 ID 和 shutdown first-error 测试 |
| `tests/config_loader/config_loader_test.cpp` | 增加 registry/manifest/`BUSY`、factory 计数、allocation/cardinality/order/ACL/accel/namespace/block-size/path/snapshot 故障和零 ledger 断言 |
| `tests/resource_contract/resource_contract_test.cpp` | 增加 provider snapshot 失败和 accelerator snapshot metadata 测试 |
| `tutti/CMakeLists.txt` | 在 HOST/CUDA 全 profile 注册 Runtime Resource registry contract |
| `doc/impl/storage-config-backend-resource-phase-4.md` | 记录本阶段实现、业务变化、文件变化和软硬件验证证据 |

本阶段未修改 daemon YAML schema、protobuf/gRPC RPC、
`config/local/daemon_2_disk.yaml`、`StorageRuntime` scheme/key 路由、resolver/DataPath SPI、公共
`ResourceProvider` 或 NVMe allocation/view 数据结构。

## 5. 故障注入结果

| 注入点 | 结果 | ledger/装配证据 |
| --- | --- | --- |
| 第二 Resource provider snapshot 缺失 | PASS | registry contract 中第一个已初始化 Resource 在失败返回时 shutdown 1 次；NVMe `ListNvmeResources` 失败发生在 Acquire 前，第二项 Acquire/Release 均 0，最终 live allocation 为 0 |
| 第二个 ResourceFactory 失败 | PASS | registry contract 中只有第一个成功 ID 进入初始化顺序；失败返回后的 Runtime cleanup 对第一个 shutdown 1 次，ledger 回到 0 |
| allocation ID 为空 | PASS | Acquire 1、Release 1、Runtime factory 0；错误包含 `allocation_id` |
| local slice 数错误 | PASS | Acquire 1、Release 1、Runtime factory 0；错误明确要求 exactly one slice |
| striped slice 数错误 | PASS | Acquire 1、Release 1、Runtime factory 0；错误明确为 slice count mismatch |
| striped slice 顺序错误 | PASS | Acquire 1、Release 1、Runtime factory 0；错误包含 order mismatch |
| slice accelerator 不符 | PASS | Acquire 1、Release 1、Runtime factory 0；错误包含 `accel_id` mismatch |
| slice ACL 不符 | PASS | Acquire 1、Release 1、Runtime factory 0；错误包含 ACL excludes Runtime accelerator |
| namespace 无效 | PASS | Acquire 1、Release 1、Runtime factory 0；错误定位 `namespace_id` |
| striped block size 不一致 | PASS | Acquire 1、Release 1、Runtime factory 0；错误定位 inconsistent logical block sizes |
| 必需 path metadata 缺失 | PASS | Acquire 1、Release 1、Runtime factory 0；错误定位 required path metadata |
| ResourceFactory 自身失败 | PASS | factory 1；provider list/Acquire/Release 和 Runtime factory 全部 0 |
| backend 重复消费同一 Resource | PASS | parser 返回 independent datapaths 明确错误；ResourceFactory/list/Acquire/Release 全部 0，不会重复 Acquire |
| backend 绑定期间单独 shutdown Resource | PASS | 返回 `BUSY`，Release 仍为 0；最终 Runtime shutdown 时 Release 1 次 |
| 后初始化 Resource shutdown 失败 | PASS | 返回 `DEVICE_ERROR`，仍继续 shutdown 更早 Resource；最终 fake ledger 为 0 |

“Runtime factory 0”结合 loader 源码顺序证明错误在 resolver/DataPath/StorageRuntime 装配入口前
返回；所有 Acquire 后的 fake client 场景均满足 `acquire_calls == release_calls`，所有静态或
snapshot/factory 前置错误均满足两者同时为 0。

## 6. 软件验证结果

### 6.1 定向 contract

| target | 结果 | 原始摘要 |
| --- | --- | --- |
| HOST/CUDA `tutti_runtime_resource_registry_contract_test` | PASS | `passed: 28`, `failed: 0` |
| HOST/CUDA `tutti_resource_contract_test` | PASS | `passed: 40`, `failed: 0` |
| HOST/CUDA `tutti_storage_config_contract_test` | PASS | `storage config checks=124 failures=0` |
| CUDA `tutti_config_loader_test` | PASS | `passed: 178`, `failed: 0` |
| `git diff --check` | PASS | 无 whitespace error |
| 禁止抽象检索 | PASS | 产品和测试源码不存在 `resource_graph`、`ResourceGraph`、`instantiate_resource_graph` 或 `validate_resource_graph` |

### 6.2 完整非硬件回归

| 命令 | 结果 | 原始摘要 |
| --- | --- | --- |
| `cmake --build build/host --parallel 8` | PASS | Runtime registry、Resource、parser 和全部 HOST targets 构建成功 |
| `ctest --test-dir build/host --output-on-failure -j 8` | PASS | 20/20，0 failed，0.26 秒 |
| `cmake --build build/cuda --parallel 8` | PASS | loader、daemon、Resource、hardware E2E 和全部 CUDA targets 构建成功 |
| `ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8` | PASS | 22/22，0 failed，0.56 秒 |

重新配置后的 HOST/CUDA CTest 清单包含
`tutti_runtime_resource_registry_contract_test`，不再包含被回退实现遗留的
`tutti_resource_graph_contract_test`。

## 7. 基于硬件的验证结果

### 7.1 环境与基线

- daemon 配置：`config/local/daemon_2_disk.yaml`，endpoint `127.0.0.1:50051`；
- accelerator：1 块 NVIDIA L40S，`accel_id=0`，PCI `0000:4B:00.0`；
- device 0：BDF `0000:b1:00.0`，chrdev `/dev/ssnvme0`，block
  `/dev/snvme0n1`，backing `/mnt/nvme0`，view `/mnt/gpu0/ssnvme0`，namespace 1，
  logical block 4096，BAR0 16384，MDTS 131072，capacity/available queues 23；
- device 1：BDF `0000:e3:00.0`，chrdev `/dev/ssnvme1`，block
  `/dev/snvme1n1`，backing `/mnt/nvme1`，view `/mnt/gpu0/ssnvme1`，namespace 1，
  logical block 4096，BAR0 16384，MDTS 131072，capacity/available queues 72；
- 测试前无 daemon、50051 listener、NVMe mount、view symlink 或 snvme device node；
- 初始 ledger：device 0 `reserved=0 available=23`，device 1
  `reserved=0 available=72`。

daemon 和 E2E 按用户授权使用 sudo；密码没有写入命令参数、测试输出或本文，也没有修改
daemon 配置、设备权限或 mount 策略。

### 7.2 Canonical loader I/O 与 Resource registry

执行：

```bash
build/cuda/bin/tutti_runtime_bundle_loader_contract_test \
  --single-accelerator --accel0 0 --device0 0 --device1 1 --queues 4
```

测试生成的应用配置全部使用 canonical storage ID 引用，不包含 BDF、device path、
allocation ID 或 lease 字段。

| 场景 | 结果 | metadata、I/O 与 ledger 证据 |
| --- | --- | --- |
| explicit device 0、4 queues | PASS | Resource registry 按 canonical ID 拥有 allocation；RPC metadata 与上表 device 0 完全一致；local write/read byte-exact；运行中 `4/19, 0/72`，shutdown 后 `0/23, 0/72` |
| explicit device 1、4 queues | PASS | RPC metadata 与 device 1 完全一致；local write/read byte-exact；运行中 `0/23, 4/68`，shutdown 后 `0/23, 0/72` |
| striped `[0,1]` write | PASS | 一次 allocation 按 device 0、1 顺序返回两个 slice；两盘同时为 `4/19, 4/68`；跨 stripe 和 mixed batch byte-exact；shutdown 后同时回到基线 |
| striped restart-read | PASS | 新 allocation ID 保持 device 0、1 顺序，读取前一次写入的数据 byte-exact；shutdown 后两盘再次同时回到基线 |
| E2E 汇总 | PASS | `phase5 checks=224 failures=0 result=PASS` |
| 最终 ledger | PASS | device 0 `reserved=0 available=23`；device 1 `reserved=0 available=72` |

该结果证明 Resource registry/manifest 改动没有改变 P3 typed view 的硬件事实来源：PCI BDF、
chrdev、block/backing/view path、namespace、LBA、BAR0、MDTS 和实际 granted queues 全部来自
daemon snapshot/allocation RPC，而不是由 ResourceSpec、device ID 或 YAML 数组位置推导。

### 7.3 daemon 与文件清理

E2E 返回后向实际 `tutti_daemon` PID 发送一次 SIGTERM。daemon 日志文件只落下启动行，没有
落下 clean-exit 文本，因此本阶段不把日志文本作为关闭证据；随后进行的独立系统检查结果为：

- 无 `tutti_daemon` 进程和 50051 listener；
- 无 `/mnt/nvme0`、`/mnt/nvme1` 或 accelerator view mount；
- `/mnt/gpu0/ssnvme0`、`/mnt/gpu0/ssnvme1` view symlink 不存在；
- `/dev/ssnvme0`、`/dev/ssnvme1`、`/dev/snvme0n1`、`/dev/snvme1n1` 不存在；
- 无 `tutti_phase5_*` scratch file 或 `/tmp/tutti_phase5_*` 临时配置目录；
- 本阶段生成的临时 daemon/test 日志已删除。

## 8. Exit Gate 与 P5 边界

- Runtime 私有 Resource registry、初始化顺序、逆序 cleanup、first-error 和幂等性均有
  HOST/CUDA 单元测试；
- loader 为每个引用 ID 最多调用一次 ResourceFactory/initialize，失败路径由 Runtime owner
  逆序释放；
- 所有静态图错误在 ResourceFactory/Acquire 前失败，所有 runtime metadata 错误在 Acquire 后
  立即 Release，fake ledger 均回到零；
- Runtime 对外只提供 ResourceInfo/BackendManifest 值快照，不暴露 mutable Resource、client、
  allocation 或 slices；
- backend 绑定期间拒绝单独 Resource shutdown，最终关闭先断开 backend 指针和组件；
- canonical 与 legacy 单 backend loader、StorageRuntime route 和双盘实机 I/O 无回归；
- 本阶段没有引入独立 Resource graph 实体或提前实现 backend factory。

P5 应在此基础上把现有 `add_local_components()`/`add_striped_components()` 兼容装配迁入
contract 驱动的 backend factory，并从 registry 中的具体 Resource typed views 创建
resolver/DataPath。P5 不得绕过 ID registry 重复 Acquire，也不得让 backend/DataPath 持有
NVMe client 或 allocation owner。

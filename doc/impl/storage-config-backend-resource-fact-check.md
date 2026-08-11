# Storage Config / Backend / Resource 全实现事实校验

## 1. 校验范围与最终结论

本文件对照
[`storage-config-backend-resource.md`](../design/storage-config-backend-resource.md)
校验 P0–P7 的实际实现，不以阶段文档中的结论代替源码、diff 和测试事实。

- 修改前基线：`f5e82d6`；
- P0–P7 实现末端：`6b790ee`；
- 审计日期：2026-08-11 UTC；
- 基线到 P7 共 64 个文件变化，`7774 insertions(+), 1257 deletions(-)`；
- `config/local/daemon_2_disk.yaml`、daemon protobuf/RPC 和 public
  `StorageRuntime`/resolver/DataPath SPI 不在该 diff 中。

最终事实结论分为两部分：

1. **代码与非硬件合同符合设计。** canonical config、Resource ownership、backend
   contract factory、typed resource views、TuttiRuntime owning registry、关闭顺序和 public
   API 边界均已落地；未发现与设计冲突的新增产品实体。
2. **整个目标设计仍不能宣称完成。** P7 严格 A0–B2 矩阵因当前只有 accelerator 0 而
   返回 SKIP(77)，B0/B1/B2 未执行。accelerator 0 的双盘 A0/A1/A2、ledger 和 clean
   shutdown 已 PASS，但不满足设计规定的最终 hardware exit gate。

## 2. 实现提交与阶段事实

| 阶段 | 提交 | 已实现事实 | Gate 事实 |
| --- | --- | --- | --- |
| P0 | `2e880ef` | 冻结基线、fake ledger/failure 观测，修复测试条件变量丢通知 | PASS；完整双 accelerator E2E 已明确 SKIP |
| P1 | `d167385`、`b0ffadb` | canonical typed model、模块化 parser、结构/引用/contract/selection 静态校验 | PASS |
| P2 | `36b0082` | `TuttiRuntime` 独立生命周期对象、逆序 cleanup 和只读 inspection 基础 | PASS |
| P3 | `e430392` | public `Resource`、private `NvmeResourceClient`、NVMe allocation ownership/state machine | PASS |
| P4 | `96abdf4` | ResourceSpec -> ResourceInstance、Runtime resource registry、失败回滚 | PASS |
| P5 | `7458702` | contract registry、backend factory、typed resolver/DataPath views 和 namespace identity 校验 | PASS |
| P6 | `6924d78` | canonical-only 默认入口、legacy application schema 删除、public/install/API 收口 | PASS |
| P7 | `6b790ee` | A0–B2 硬件 gate、完整 metadata/ledger/shard-order/cleanup 验收实现 | **SKIP/未完成：缺 accelerator 1** |

P7 的代码提交说明是 `add phase 7 hardware validation`，没有使用 `complete`，与实际 gate
一致。

## 3. 修改前后事实对比

### 3.1 Application config

| 修改前 `f5e82d6` | 修改后 `6b790ee` | 设计判断 |
| --- | --- | --- |
| `ParsedConfig` 保存 `storage_backend`、`nvme_service_endpoint`、selection、device IDs、stripe/cache 等扁平字段 | `ParsedConfig` 只保存 `CanonicalStorageConfig`、profile 和 runtime accelerator | 符合 canonical graph |
| application config 可引用 `local_nvme_config`，并存在从数组顺序推导 device path 的 legacy helper | application parser 不接受 `local_nvme_config`；路径、BDF、lease 均不在 canonical model | 符合部署事实只属于 daemon |
| legacy flat schema 是正常入口 | legacy-only 和 canonical/legacy mixed 均在 parse 阶段报 `legacy Tutti config schema was removed in P6` | 符合最终目标态；比过渡兼容建议更严格 |
| backend 类型由扁平字符串和 allocation slice 数量参与分支 | backend 通过 ID 引用 resolver/DataPath/resource，并由 logical contract 选择 factory | 符合显式 backend 关系 |

当前 canonical 值模型只包含：

```text
storage.resources[]  -> ResourceSpec/provider/allocation
storage.resolvers[]  -> ResolverSpec/type/scheme
storage.datapaths[]  -> DataPathSpec/type/tuning
storage.backends[]   -> BackendSpec/contract/three IDs/shared config
```

它没有 BDF、chrdev、block、mount、view、allocation ID、granted queue 或 lease 字段。
P7 进一步把该事实固定为真实临时 YAML 的运行时断言。

### 3.2 Resource 与 allocation ownership

| 修改前 | 修改后 | 设计判断 |
| --- | --- | --- |
| public loader header 暴露 `RuntimeResourceClient`、`RuntimeNvmeAllocation`、`RuntimeNvmeSlice` | public 只暴露通用 `Resource`、`ResourceInfo`、`ResourceCapabilities` | 符合公共抽象边界 |
| `TuttiRuntime` 直接拥有 resource client、allocation ID 和 slices | `NvmeResource` 私有拥有 `NvmeResourceClient`、allocation 和 lease state | 符合 allocation owner 唯一性 |
| loader 在 Acquire 后自行校验并装配 | `NvmeResource::initialize()` 先检查 provider snapshot，再 Acquire、校验 metadata，失败时自行 Release | 符合 Resource lifecycle |
| Runtime shutdown 直接调用 client Release | Runtime 只逆序调用 `Resource::shutdown()`；`NvmeResource` guard 保证 Release 至多一次 | 符合 ownership/lifecycle |

`NvmeResource` 的 allocation metadata 校验实际覆盖 allocation ID、cardinality/order、
accelerator、ACL、必需路径、namespace、LBA、BAR0、MDTS、grant 和 striped block-size 一致性。
产品 public API 不返回 allocation/client；硬件 E2E 只通过 repository-internal immutable copy
读取这些值。

### 3.3 Backend 装配

| 修改前 | 修改后 | 设计判断 |
| --- | --- | --- |
| loader 直接构造 Local/Striped resolver 和 DataPath | contract registry 先定位实现，再由 backend factory 构造产品 | 符合 contract-driven assembly |
| backend 选择与具体 allocation slice 数量耦合 | contract 先决定 factory；factory 再校验 typed view cardinality | 符合先 contract、后实例事实 |
| resolver/DataPath 参数分别从 loader 散列传入 | 同一 `NvmeResource` 产生 resolver view 和 DataPath view | 符合同一 ResourceInstance 投影 |
| shard identity 没有统一 product 校验 | factory 对 resolver/DataPath projection、请求顺序、payload identity/version、MDTS 最小值逐项 fail-closed | 符合 namespace/shard 一致性 |

当前已实现的产品 factory 只有 `ext4-local-nvme` 和 `striped-local-nvme`。memfs contract/parser
保留为 closed-but-extensible schema 项并明确 `implemented=false`；不存在新增
`MemfsResource`，loader 在 factory/provider 前返回 unsupported。这符合设计的非目标，未
把示例扩展点误实现成新的 Resource 实体。

### 3.4 TuttiRuntime 与 StorageRuntime

修改前 `TuttiRuntime` 是 public struct，公开 `runtime`、resolver/DataPath vectors、route
vectors、client 和 allocation。修改后它是封装 class：

```text
TuttiRuntime
  owns StorageRuntime
  owns Resource registry by config ID
  owns resolver registry by config ID
  owns DataPath registry by config ID
  owns backend manifest + borrowed internal relations by config ID
```

public 只保留借用式 `storage_runtime()`、`ResourceInfo` copy、`BackendManifest` copy、state
和幂等 `shutdown()`。backend ID 只用于 config/诊断，不进入 I/O 热路径。

`StorageRuntime` 的核心 source header 相对基线无变化，仍按 URI scheme 查 resolver、按
resolved target 的 DataPath key 查 DataPath。没有新增 backend ID/resource ID 路由，也
没有让 `StorageRuntime` 解析 YAML 或连接 daemon。设计要求的 multi-scheme/multi-key
核心能力没有收缩；当前 application loader 的单 backend 限制在 config/assembly 层执行。

### 3.5 Shutdown 与失败回滚

当前正常关闭顺序与设计一致：

```text
StorageRuntime::shutdown/destroy
  -> 清空 backend borrowed relations
  -> resolver 逆注册顺序销毁
  -> DataPath 逆注册顺序销毁
  -> Resource 逆 initialize 顺序 shutdown
```

Runtime 记录第一个 shutdown error，但继续清理更早的 Resource。NVMe Resource 以
`ACQUIRED -> RELEASING -> RELEASED` guard 防止二次 Release；重复 Runtime shutdown 和析构
fallback 均幂等。fake tests 覆盖第二 Resource snapshot/factory 失败、backend factory
失败、DataPath initialize 失败、Runtime create 失败和 shutdown error，不暴露半初始化
registry。

P7 在每次真实 allocation 后显式调用两次 `TuttiRuntime::shutdown()`，并通过独立 daemon
snapshot 证明已执行场景的 ledger 精确恢复。

## 4. 设计逐项符合性矩阵

| 设计要求 | 源码/测试事实 | 判定 |
| --- | --- | --- |
| 四组 config ID 非空、各自唯一、引用存在 | parser contract 和 loader preflight 覆盖 duplicate/missing/unreferenced | 符合 |
| 当前只启用一个 backend | parse 和 full loader 均要求 `backends.size() == 1` | 符合 |
| scheme/DataPath key 唯一 | static validation、Runtime registry 和 duplicate tests 三层 fail-closed | 符合 |
| allowed/explicit/striped selection 约束 | parser 在任何 ResourceFactory/RPC 前验证 device ID cardinality/order | 符合 |
| 静态错误不得 List/Acquire | fake call counters 固定为 0 | 符合 |
| Resource 是 application-side allocation owner | `NvmeResource` 持有 client/allocation/lease；Runtime 只持 `unique_ptr<Resource>` | 符合 |
| ResourceInfo 不暴露 mutable allocation | public snapshot 只有 ID/type/state | 符合 |
| resolver/DataPath 从同一 ResourceInstance 获取事实 | typed views 从同一 immutable allocation copy 生成 | 符合 |
| backend contract 校验 type/cardinality/payload/key | registry + product validator + contract tests | 符合 |
| StorageRuntime 不感知 YAML/backend/resource/daemon | public core header无相关类型，source diff 无路由改造 | 符合 |
| backend ID 不成为热路径 key | 仅 `BackendManifest` 和 Runtime internal relation 使用 | 符合 |
| public SPI 不包含 NVMe/protobuf 类型 | header hygiene 和源码扫描均无命中 | 符合 |
| application config 不复制设备路径 | canonical model/默认 config/P7 临时 YAML 均无路径字段 | 符合 |
| legacy 仅作为明确迁移路径或删除 | P6 选择直接删除并提供固定迁移诊断，无双 loader | 符合最终目标态 |
| 不实现额外 MemfsResource | 没有该 concrete Resource；unsupported fail-closed | 符合非目标 |
| P7 A0–B2 双 accelerator 闭环 | A0/A1/A2 辅助 PASS；B0/B1/B2 SKIP | **未满足** |

## 5. 验证事实

### 5.1 非硬件

最终 P7 修改后的回归结果：

| 验证 | 结果 |
| --- | --- |
| HOST full build | PASS |
| HOST ctest | 20/20 PASS |
| CUDA full build | PASS |
| CUDA `-LE hardware` | 22/22 PASS |
| parser/Resource/backend/Runtime registry/header hygiene | 全部 PASS |
| `git diff --check` | PASS |

CUDA build 仍有未触及的 striped contract test misleading-indentation/unused-variable warning；
它们不是本实现引入的失败，也不影响上述 22 个非硬件测试结果。

### 5.2 实机

指定 daemon 配置提供一块 NVIDIA L40S 和两块 NVMe。两盘 baseline 分别为
`reserved/available = 0/23`、`0/72`，ACL 都是 `[0]`。

- 严格 `--accel0 0 --accel1 1`：SKIP(77)，未发起 allocation；
- `--single-accelerator` 辅助：A0/A1/A2 共 `246 checks, 0 failures`；
- A0 只改变 device 0 ledger，A1 只改变 device 1 ledger；
- A2 单 allocation 同时改变两盘，跨 64 KiB、mixed batch、reload persistence 和实际
  resolver/DataPath shard order均 PASS；
- 每次双 shutdown 后 ledger 回到 `0/23, 0/72`；
- 单次 SIGTERM 后 daemon clean exit，mount、view、device node、port、process、scratch 和
  temp YAML 全部无残留。

这些事实证明 accelerator 0 的两盘实现正确，但不能推导 accelerator 1 correctness。

## 6. 偏差、边界与最终判定

### 6.1 有意差异

设计允许过渡期保留 legacy adapter，但不要求永久兼容。P6 按当时实施要求直接删除
application legacy schema，没有 grace period。该行为改变了修改前的配置兼容性，但没有
违反最终 canonical-only ownership 设计；README、错误诊断和 P6 文档均明确记录。

设计文字使用“ResourceFactory”描述职责；实现使用 private default factory function 加
internal injection seam，没有增加 public factory class。其调用顺序、失败语义和 ownership
与设计相同，也避免引入不需要的 public 实体。

P7 为实际 shard-order 证明增加的两个 access point 都是 repository-internal/test-only；
它们不进入 install API，不改变 ownership，也不让应用绕过 Resource allocation。

### 6.2 未发现的错误扩展

本次审计未发现以下设计禁止项：

- 没有让 `StorageRuntime` 解析 YAML、连接 daemon 或按 backend/resource ID 路由；
- 没有把 nvmeservice/protobuf/allocation 类型加入 public Runtime/resolver/DataPath SPI；
- 没有让 DataPath 自选 ResourceSpec/lease 之外的设备；
- 没有增加同 scheme 多 resolver 的隐式动态选择；
- 没有把 daemon 的 BDF/path/mount 复制回 application config；
- 没有实现 `MemfsResource` 或额外 product factory。

### 6.3 最终判定

**代码实现符合设计，P0–P6 完成；P7 验收工具完成但硬件 exit gate 未完成。因此整个实现
当前状态是“设计实现已落地、最终双 accelerator 事实验证待完成”，不是“目标设计完成”。**

完成条件仍然只有一个：在具有 accelerator 0 和 1、两盘均允许 `[0,1]` 的 daemon 配置上
运行严格 P7 入口，使 A0/A1/A2/B0/B1/B2 全部 correctness PASS，并再次证明每次 ledger
恢复和 daemon 零残留。性能数字不是该 ownership 设计的替代 gate。

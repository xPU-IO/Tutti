# Storage Config / Backend / Resource P6 实现记录

## 1. 阶段范围与结论

本阶段实现
[`storage-config-backend-resource-implementation-plan.md`](storage-config-backend-resource-implementation-plan.md)
的 P6，并按本次要求采用比原计划更激进的应用配置迁移策略：Tutti application config
只保留 canonical schema，不保留 legacy adapter。daemon config 不属于本次迁移范围，
`config/local/daemon_2_disk.yaml` 及 daemon 自身的 schema 兼容逻辑均未修改。

P6 exit gate 通过：默认应用示例和 loader 都以 canonical graph 为唯一入口；legacy-only
与 canonical/legacy 混用均在静态解析阶段失败；`TuttiRuntime` 的 StorageRuntime、
resolver、DataPath、Resource 和 backend 关系不再通过可变 public 字段暴露；HOST 全量、
CUDA 非硬件全量以及单 accelerator 双盘 loader 实机闭环均通过。

## 2. 计划项与本次实现对比

| P6 要求 | 本次实现 | 状态 |
| --- | --- | --- |
| 默认示例迁移到 canonical | `config/tutti_config.yaml` 改为完整的 resource/resolver/datapath/backend ID graph | 完成 |
| legacy 迁移策略 | 按本次要求删除 adapter、legacy parser、扁平兼容投影和 topology derive helper；旧字段只保留 fail-closed 识别与迁移诊断 | 完成，强于原计划 |
| canonical/legacy 混用固定失败 | parse-only 和 full loader 合同均覆盖 legacy-only 与 mixed 文件，失败发生在 ResourceFactory/RPC 前 | 完成 |
| 删除 Runtime 平行 public 字段 | 删除 public StorageRuntime/component vectors、route vectors、register/set hook API 和单 Resource 兼容 lookup；只保留借用式 `storage_runtime()` 与只读 ID snapshot | 完成 |
| Runtime 只保留 owning registry | resolver、DataPath、Resource 按 canonical ID 私有拥有；backend 保存只读 manifest 和内部 borrowed relation | 完成 |
| E2E 迁移 inspection | loader E2E 用 `ResourceInfo`、`BackendManifest` 和 NVMe internal immutable copy seam；不读取 Runtime allocation/client 字段 | 完成 |
| public header/CMake/install | canonical config model/loader header 移入 public include tree；parse/runtime/config library 加入 install；内部 factory/injection seam 不进入 public include | 完成 |
| parse-only 无 CUDA/daemon 依赖 | HOST profile 可单独构建并运行 `tutti_config_parse` 及 parser contract；public include 不传播 workspace/private NVMe headers | 完成 |
| legacy 移除期限 | 应用 legacy schema 截止为 P6，日期 2026-08-11，无 grace period；README 和错误诊断均记录 | 完成 |

## 3. 业务逻辑变化

### 3.1 Application config 成为 canonical-only

解析路径收敛为：

```text
YAML
  -> canonical root whitelist
  -> resources/resolvers/datapaths/backends typed model
  -> reference/contract/cardinality/reachability validation
  -> ResourceFactory + backend factory + Runtime registry
```

以下应用字段不再解析或转换：`gpu`、`nvme_service`、`nvme`、`local_nvme`、
`local_nvme_config`、`storage.backend`、`storage.default_stripe_unit`。任一字段出现都会返回
`INVALID_ARGUMENT` 和 `legacy Tutti config schema was removed in P6` 迁移提示。该失败在
backend device count、ResourceFactory、daemon list/acquire 和 StorageRuntime factory 前发生。

`ParsedConfig` 不再保存 syntax tag 或 endpoint、selection、device IDs、cache、stripe 等
扁平副本。cache override 直接通过唯一 backend 引用找到 canonical `DataPathSpec`，继续
保持 programmatic > config > test-only env > default 的优先级。

### 3.2 TuttiRuntime API 与 ownership 收口

公开 API 只提供：

- 借用式 `storage_runtime()`，调用方不能替换或转移其 ownership；
- 按 resource ID 返回 copy 的 `resource_info(id)` / `resource_infos()`；
- 按 backend ID 返回 copy 的 `backend_manifest(id)` / `backend_manifests()`；
- 幂等 `shutdown()` 和只读 lifecycle state。

内部 ownership 为：

```text
TuttiRuntime
  -> runtime_: unique_ptr<StorageRuntime>
  -> resources_[resource ID]: unique_ptr<Resource>
  -> resolvers_[resolver ID]: unique_ptr<StorageTargetResolver>
  -> datapaths_[datapath ID]: unique_ptr<DataPath>
  -> backends_[backend ID]: manifest + internal borrowed relations
```

component registration、StorageRuntime adoption、lifecycle hook 和单 Resource shutdown 只在
repository-internal assembly/testing header 中存在，public include target 无法包含这些 seam。
外部不能单独 shutdown Resource、替换 provider client、修改 allocation，或从 Runtime
取走 resolver/DataPath ownership。

关闭顺序保持：StorageRuntime shutdown/destroy -> backend borrowed relation 置空 -> resolver
逆序销毁 -> DataPath 逆序销毁 -> Resource 按 initialize 顺序逆序 shutdown。第二次
`shutdown()` 和析构 fallback 均不会重复 Release。

### 3.3 Public/install 边界

`storage_config.h` 和 `tutti_config.h` 移至 `tutti/include/tutti/config/`，与
`tutti_runtime.h`、`resource.h` 一起由统一 public include install 规则发布。
`tutti_config_parse`、`tutti_runtime` 和 CUDA graph 中的 `tutti_config` 均安装到 `lib/`。

`backend_factory.h`、`tutti_config_internal.h`、`tutti_runtime_internal.h` 和 NVMe Resource
实现继续是 repository-private。header hygiene 使用只链接 `tutti_api` 的 consumer 固定
验证这些头不可达，公共 `StorageRuntime`、resolver SPI、DataPath SPI 中也没有
NVMe service、protobuf、gRPC 或 allocation 类型。

## 4. 文件变化

| 文件 | 变化 |
| --- | --- |
| `config/tutti_config.yaml` | legacy flat 示例替换为 canonical local NVMe graph；不包含 daemon 部署事实 |
| `README.md` | 修正 daemon/application config 边界，记录 P6 legacy 截止日期和迁移行为 |
| `tutti/include/tutti/config/storage_config.h` | canonical public 值模型迁入安装 include tree，删除 `ConfigSyntax` |
| `tutti/include/tutti/config/tutti_config.h` | public loader/parse API 只暴露 canonical `ParsedConfig` 和 cache override |
| `tutti/config/storage_config.h` | 删除旧 source-tree public header 位置 |
| `tutti/config/tutti_config.h` | 删除带 full-loader test seam 和 legacy 扁平字段的旧 public header |
| `tutti/config/tutti_config_internal.h` | 新增 repository-internal factory/lifecycle injection seam |
| `tutti/config/tutti_config_legacy_nvme_parse.cpp` | 删除 legacy parser、adapter、compatibility projection 和 daemon topology derive helper |
| `tutti/config/tutti_config_parse.cpp` | 唯一 canonical parse 路径、legacy removal diagnostic、canonical cache lookup |
| `tutti/config/storage/parse_internal.h` | 删除 legacy adapter 声明 |
| `tutti/include/tutti/tutti_runtime.h` | public aggregate 改为封装 class；组件、route 和 StorageRuntime ownership 全部私有 |
| `tutti/tutti_runtime/tutti_runtime.cpp` | resolver/DataPath ID owning registry、私有 StorageRuntime adoption、逆序 cleanup |
| `tutti/tutti_runtime/tutti_runtime_internal.h` | assembly/testing access 迁移到内部 seam |
| `tutti/config/tutti_config.cpp` | loader 只通过内部 assembly access 注册 canonical ID graph |
| `tutti/config/CMakeLists.txt` | 移除 legacy source；阻止 workspace include 泄漏；补齐 build/install interface 和 library install |
| `tests/storage_config_contract/*` | 删除 legacy 成功用例，新增 legacy-only/mixed 固定失败和 canonical-only value assertions |
| `tests/config_loader/config_loader_test.cpp` | 所有 loader 成功/失败用例迁为 canonical；public route/vector 断言迁为 manifest/ResourceInfo |
| `tests/runtime_bundle_loader_contract/runtime_bundle_loader_contract_test.cpp` | I/O 经借用式 `storage_runtime()`；装配断言改为 immutable resource/backend inspection |
| `tests/runtime_resource_registry_contract/runtime_resource_registry_contract_test.cpp` | 删除无 ID 的单 Resource 兼容 lookup 依赖 |
| `tests/header_hygiene/header_hygiene_test.cpp` | 固定 public canonical headers 可达、factory/internal/NVMe implementation headers 不可达 |

daemon YAML、daemon parser/RPC、protobuf、kernel module、DataPath SPI、resolver SPI 和
`StorageRuntime` 路由实现均未修改。

## 5. 验证结果

### 5.1 计划要求的回归

| 命令 | 结果 | 摘要 |
| --- | --- | --- |
| `cmake --build build/host --parallel 8` | PASS | canonical parse、Resource、private Runtime registry 和全部 HOST targets 构建成功 |
| `ctest --test-dir build/host --output-on-failure -j 8` | PASS | 20/20，0 failed |
| `cmake --build build/cuda --parallel 8` | PASS | config loader、backend、daemon、硬件测试目标及全部 CUDA targets 构建成功 |
| `ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8` | PASS | 22/22，0 failed |
| `ctest --test-dir build/cuda -R '^tutti_config_loader_test$' --output-on-failure` | PASS | fake provider loader contract 通过，canonical static failures 均为 0 RPC |
| `git diff --check` | PASS | 无 whitespace error |

重点合同：

- `tutti_storage_config_contract_test`：PASS；
- `tutti_resource_contract_test`：HOST/CUDA PASS；
- `tutti_runtime_resource_registry_contract_test`：HOST/CUDA PASS；
- `tutti_config_loader_test`：PASS；
- `tutti_runtime_bundle_loader_contract_test`：编译成功，不依赖已删除 public parallel fields；
- lifecycle failure injection：factory、payload、DataPath initialize、Runtime create 失败后
  Resource Release 一次，resolver/DataPath 均销毁；
- repeated shutdown/destructor：PASS，无二次 Release。

### 5.2 Install 与依赖边界

HOST 和 CUDA build 分别安装到临时 staging prefix 后检查：

- `include/tutti/config/storage_config.h`、`include/tutti/config/tutti_config.h`、
  `include/tutti/tutti_runtime.h` 均存在；
- `lib/libtutti_config_parse.a`、`lib/libtutti_runtime.a` 均存在；
- CUDA staging 额外存在 `lib/libtutti_config.a`；
- staging 目录检查后已删除，无安装生成物进入工作树；
- HOST profile 能独立构建并运行 parser contract，证明 parse-only target 不需要 CUDA、
  nvmeservice 或 daemon。

## 6. 基于硬件的验证结果

### 6.1 环境

- daemon 配置：`config/local/daemon_2_disk.yaml`，未修改；
- accelerator：NVIDIA L40S，`accel_id=0`，PCI `0000:4b:00.0`；
- device 0：`0000:b1:00.0`，namespace 1，LBA 4096，BAR0 16384，MDTS 131072，
  baseline `reserved=0 available=23`；
- device 1：`0000:e3:00.0`，namespace 1，LBA 4096，BAR0 16384，MDTS 131072，
  baseline `reserved=0 available=72`。

daemon 启动后 mount/view 由 root 创建为 `0755`。普通用户第一次 dry run 因无权在 view
目录创建 scratch，在真正 I/O 前失败；该次所有 allocation 均正确 Release，最终 ledger
仍为 `0/23, 0/72`。随后按用户授权使用 sudo 执行正式硬件用例，未修改目录权限或 daemon
配置。

### 6.2 Canonical loader 双盘闭环

执行：

```bash
sudo build/cuda/bin/tutti_runtime_bundle_loader_contract_test \
  --single-accelerator --accel0 0 --device0 0 --device1 1 --queues 4
```

| 场景 | 结果 | 证据 |
| --- | --- | --- |
| local device 0 | PASS | canonical manifest/resource inspection；运行中 `4/19, 0/72`；write/read byte-exact；shutdown 回到基线 |
| local device 1 | PASS | canonical manifest/resource inspection；运行中 `0/23, 4/68`；write/read byte-exact；shutdown 回到基线 |
| striped `[0,1]` write | PASS | 单 allocation 返回有序双 slice；运行中 `4/19, 4/68`；跨 stripe 与 mixed batch byte-exact |
| striped restart-read | PASS | 新 allocation 保持 shard 顺序，读取前次内容 byte-exact；shutdown 回到基线 |
| 汇总 | PASS | `phase5 checks=224 failures=0 result=PASS`（测试程序沿用既有 phase5 输出名） |
| 最终 ledger | PASS | device 0 `0/23`，device 1 `0/72` |

### 6.3 清理检查

测试结束后只向已确认的唯一 `tutti_daemon` PID 发送 SIGTERM。最终检查：

- 无 `tutti_daemon` 或 loader E2E 进程，无 50051 listener；
- 无 `/mnt/nvme0`、`/mnt/nvme1` mount；
- 无 `/mnt/gpu0/ssnvme0`、`/mnt/gpu0/ssnvme1` view symlink；
- 无 `/dev/ssnvme0`、`/dev/ssnvme1`、`/dev/snvme0n1`、`/dev/snvme1n1`；
- 无 `tutti_phase5_*` scratch、临时 application config 或 install staging 残留。

## 7. Exit Gate

- canonical loader 是应用配置唯一解析和默认装配路径；
- legacy application config 已删除，不存在兼容入口或第二套 loader；
- legacy-only 与 mixed 文件稳定 fail-closed，且在任何 provider/RPC 前失败；
- Runtime owning registries 和 backend relation 私有，产品 public API 不暴露 allocation、
  client 或 component ownership；
- public/install/CMake 表面完整，parse-only target 保持硬件无关；
- HOST、CUDA 非硬件回归和双盘硬件闭环全绿；
- daemon config、daemon 代码与生成物无无关改动。

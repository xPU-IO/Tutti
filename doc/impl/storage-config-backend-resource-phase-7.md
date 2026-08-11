# Storage Config / Backend / Resource P7 实现记录

## 1. 阶段范围与结论

本阶段实现
[`storage-config-backend-resource-implementation-plan.md`](storage-config-backend-resource-implementation-plan.md)
的 P7 daemon 双盘实机验收能力，并使用用户指定的
`config/local/daemon_2_disk.yaml` 完成当前硬件允许的验证。

P7 测试实现完成，但 **P7 exit gate 未通过，阶段结果为 SKIP/未完成**。当前主机和指定
daemon 配置都只有 accelerator 0；严格 A0/A1/A2/B0/B1/B2 入口以退出码 77 返回
`SKIP: requested accelerator set is unavailable`。因此本文只把 accelerator 0 上的双盘
A0/A1/A2 结果记录为辅助 PASS，不把缺失的 B0/B1/B2 报告为通过，也不宣称目标设计已
完成实机闭环。

## 2. 计划项与本次实现对比

| P7 要求 | 本次实现 | 状态 |
| --- | --- | --- |
| 两 accelerator、两 NVMe 前置条件 | 同时检查 CUDA device count、daemon accelerator ID、两盘 available 状态和每盘 ACL；任一不足返回 77 | 完成；本机结果 SKIP |
| A0/A1/A2/B0/B1/B2 固定场景 | 场景名修正为 accelerator 0 的 A0/A1/A2 和 accelerator 1 的 B0/B1/B2，不再使用 A1–A6 | 完成 |
| 临时 canonical YAML 只含逻辑事实 | 每次 load 前调用 `parse_tutti_config()`，核对 canonical graph、selection、device IDs 和 queues；同时断言无 path/BDF/allocation/lease 字段及 daemon 路径值 | 完成 |
| 独立 daemon snapshot | 输出 accelerator view root 及两盘 BDF、chrdev、block、backing、namespace、LBA、BAR0、MDTS、capacity/reserved/available queues、ACL | 完成 |
| allocation metadata 对照 | 通过 NVMe immutable inspection copy 与独立 snapshot 逐字段对照；检查 lease state、ResourceInfo、BackendManifest、view root 和同文件系统 | 完成 |
| public Runtime I/O | 通过 `storage_runtime()` 执行 open、register、submit、query、wait、release、close、unregister 和 byte-exact 校验 | 完成 |
| striped 边界、mixed batch、重载 | 覆盖 64 KiB 边界、三请求 mixed batch、write -> shutdown -> reload -> read persistence | 完成 |
| resolver/DataPath shard 顺序 | repository-internal seam 只借用 backend 组件；实际 resolve 后逐 shard 对照 resolver payload namespace、DataPath descriptor 和 Resource slice | 完成 |
| shutdown 幂等和 ledger 回收 | 每次 allocation 显式调用两次 `TuttiRuntime::shutdown()`，轮询所有设备并要求 ledger 精确回到场景前基线 | 完成 |
| daemon clean shutdown | 对确认的唯一 PID 只发送一次 SIGTERM，等待 clean exit，并检查 mount/view/device/process/port/temp 残留 | 完成 |
| 完整双 accelerator correctness | 当前只有一块 accelerator，B0/B1/B2 无法执行 | **SKIP，exit gate 未通过** |

`StorageRuntime` 当前没有独立 public `progress()` 方法；其 public `query()` 和 `wait()`
按既有 contract 驱动 bounded DataPath progress。本阶段没有为了测试新增第二套 Runtime API，
而是继续通过 `submit -> query -> wait` 验证相同 progress 路径。

## 3. 业务逻辑变化

产品配置、daemon、Resource 分配、backend factory 和 I/O 路由逻辑均未改变。本阶段变化
仅用于硬件验收和 repository-internal 只读观测：

1. 硬件测试在任何 allocation 前验证精确 accelerator ID、两盘状态和 ACL。双 accelerator
   不可用时直接 SKIP，不退化为单 accelerator PASS。
2. 每个场景从独立 `ListNvmeResources` snapshot 开始；allocation 期间只允许所选盘的
   `reserved_queues` 增加，请求结束后所有盘必须精确恢复。
3. 临时 YAML 解析结果必须是一个 resource/resolver/datapath/backend canonical graph，且
   YAML 文本不得复制 daemon 的 BDF、chrdev、block 或 backing path。
4. allocation inspection 必须处于 `ACQUIRED`，slice 顺序、accelerator、ACL、namespace、
   LBA、BAR0、MDTS、grant 和路径必须与 daemon snapshot 一致。
5. striped resolver 的实际 immutable payload 和 striped DataPath 的 retained descriptor
   按 PCI、namespace、block size、chrdev 逐项对照，证明两条装配路径没有 shard 重排。
6. `TuttiRuntime::shutdown()` 显式执行两次，第二次必须成功且不能重复 Release。

新增 seam 不转移 ownership：`TuttiRuntimeTestingAccess` 只返回 Runtime 已拥有的 backend
resolver/DataPath 借用指针；striped DataPath accessor 只返回 descriptor 的 const reference。
这些入口位于 repository-internal header，不进入安装 public API。

## 4. 文件变化

| 文件 | 变化 |
| --- | --- |
| `tests/runtime_bundle_loader_contract/runtime_bundle_loader_contract_test.cpp` | P7 前置检查、A0–B2 命名、完整 snapshot、静态 config 断言、metadata/ACL/view 校验、actual shard-order inspection、双 shutdown、phase7 scratch/output 和明确 SKIP |
| `tutti/tutti_runtime/tutti_runtime_internal.h` | 增加按 backend ID 借用 resolver 和 const DataPath 的 internal testing access |
| `tutti/data_paths/striped_local_nvme/striped_data_path.h` | 增加 test-only const descriptor view，用于实机 shard 顺序对照 |
| `doc/impl/storage-config-backend-resource-phase-7.md` | 本阶段实现、业务变化、文件变化、硬件结果、清理证据和 exit gate |

未修改 `config/local/daemon_2_disk.yaml`、daemon schema/protobuf/RPC、kernel module、
`StorageRuntime` public API、resolver/DataPath SPI、canonical schema 或默认应用配置。临时
canonical YAML 仅在 `/tmp` 下生成并由测试析构清理，没有进入工作树。

## 5. 非硬件验证结果

| 命令 | 结果 | 原始摘要 |
| --- | --- | --- |
| `cmake --build build/host --parallel 8` | PASS | 全部 HOST targets 构建成功 |
| `ctest --test-dir build/host --output-on-failure -j 8` | PASS | 20/20，0 failed |
| `cmake --build build/cuda --parallel 8` | PASS | daemon、loader E2E 和全部 CUDA targets 构建成功 |
| `ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8` | PASS | 22/22，0 failed |
| `cmake --build build/cuda --target tutti_runtime_bundle_loader_contract_test --parallel 8` | PASS | 新 inspection 和 P7 E2E 编译、链接成功 |
| `git diff --check` | PASS | 无 whitespace error |

CUDA 全量构建仍报告 `striped_local_nvme_contract_test.cpp` 中既有的 misleading-indentation
和 unused-variable warning；构建及测试成功，本阶段未修改这些无关代码。

## 6. 基于硬件的验证结果

### 6.1 环境与 daemon 基线

- 日期：2026-08-11 UTC；endpoint：`127.0.0.1:50051`；
- accelerator：仅 1 块 NVIDIA L40S，`accel_id=0`，PCI `0000:4B:00.0`，view root
  `/mnt/gpu0`；不存在 accelerator 1；
- device 0：BDF `0000:b1:00.0`，chrdev `/dev/ssnvme0`，block
  `/dev/snvme0n1`，backing `/mnt/nvme0`，namespace 1，LBA 4096，BAR0 16384，
  MDTS 131072，queue capacity 23，baseline `reserved=0 available=23`，ACL `[0]`；
- device 1：BDF `0000:e3:00.0`，chrdev `/dev/ssnvme1`，block
  `/dev/snvme1n1`，backing `/mnt/nvme1`，namespace 1，LBA 4096，BAR0 16384，
  MDTS 131072，queue capacity 72，baseline `reserved=0 available=72`，ACL `[0]`。

daemon 使用指定配置和用户授权的 sudo 启动；凭据没有出现在命令参数、YAML、测试日志或
本文中。测试由 root client 执行，未修改 daemon 创建的目录权限、设备权限或 mount 策略。

### 6.2 严格 P7 矩阵

执行严格入口：

```bash
sudo build/cuda/bin/tutti_runtime_bundle_loader_contract_test \
  --accel0 0 --accel1 1 --device0 0 --device1 1 --queues 4
```

结果：`SKIP: requested accelerator set is unavailable`，退出码 77。测试在生成临时 YAML、
发起 RPC allocation 或创建 scratch file 前退出。

| 场景 | 结果 | 原因 |
| --- | --- | --- |
| A0：accelerator 0 + device 0 | 未由严格矩阵执行 | 整体前置条件在场景前返回 SKIP；下节有单 accelerator 辅助 PASS |
| A1：accelerator 0 + device 1 | 未由严格矩阵执行 | 同上；下节有单 accelerator 辅助 PASS |
| A2：accelerator 0 + striped `[0,1]` | 未由严格矩阵执行 | 同上；下节有单 accelerator 辅助 PASS |
| B0：accelerator 1 + device 0 | SKIP | CUDA 和 daemon 均无 accelerator 1，device ACL 也只有 `[0]` |
| B1：accelerator 1 + device 1 | SKIP | 同上 |
| B2：accelerator 1 + striped `[0,1]` | SKIP | 同上 |

### 6.3 单 accelerator 双盘辅助闭环

为验证本阶段新增断言及 accelerator 0 的真实 I/O，额外执行：

```bash
sudo build/cuda/bin/tutti_runtime_bundle_loader_contract_test \
  --single-accelerator --accel0 0 --device0 0 --device1 1 --queues 4
```

最终结果为 `phase7 checks=246 failures=0 result=PASS`。

| 场景 | correctness 与 metadata | allocation 期间 ledger | 两次 shutdown 后 |
| --- | --- | --- | --- |
| A0 explicit device 0 | canonical-only；slice 与 device 0 snapshot 全字段一致；Local write/read byte-exact | device 0 `4/19`，device 1 `0/72` | `0/23, 0/72` |
| A1 explicit device 1 | canonical-only；跨 ACL 允许的 device 1；Local write/read byte-exact | device 0 `0/23`，device 1 `4/68` | `0/23, 0/72` |
| A2 striped write | 单 allocation 返回 device 0、1 有序 slices；resolver payload 与 DataPath descriptor 同序；跨 64 KiB 和 mixed batch byte-exact | device 0 `4/19`，device 1 `4/68` | 两盘一次性回到 `0/23, 0/72` |
| A2 restart-read | 新 allocation/slices 顺序不变；重载后读取前次内容 byte-exact | device 0 `4/19`，device 1 `4/68` | 两盘再次回到 `0/23, 0/72` |

每个 load 前均重新保存 daemon snapshot 并验证 ledger 等于全局基线；每个 allocation 的
ID 非空，restart 使用不同 allocation，inspection lease state 为 `ACQUIRED`。每个 view
path 都位于对应 accelerator view root，且与 backing mount 的 `st_dev` 相同。

第一次辅助 dry run 报告 `checks=246 failures=12`：新增 static-YAML 检查错误地要求
`istream::eof()` 必须置位，导致四次 config 断言失败，并使复用累计 `ok` 的三个摘要断言
连带失败；实际 allocation、I/O 和每次 ledger 回收没有泄漏。实现改为验证文件成功打开且
无 `badbit`，同时将 Local/manifest 摘要改为独立结果；重新构建后的上述最终运行 0 failure。

### 6.4 daemon 停止与残留检查

正式测试后独立 `ListNvmeResources` snapshot 仍为 device 0 `reserved=0 available=23`、
device 1 `reserved=0 available=72`。随后只向已确认的唯一 daemon PID 发送一次 SIGTERM；
daemon 输出：

```text
Shutting down...
mount_manager: unmounted /mnt/nvme0
mount_manager: unmounted /mnt/nvme1
tutti_daemon exited cleanly.
```

最终检查全部 PASS：

- 无 `tutti_daemon` 或 loader E2E 进程，50051 已释放；
- `/mnt/nvme0`、`/mnt/nvme1` 均未挂载；
- `/mnt/gpu0/ssnvme0`、`/mnt/gpu0/ssnvme1` view symlink 已删除；
- `/dev/ssnvme0`、`/dev/ssnvme1`、`/dev/snvme0n1`、`/dev/snvme1n1` 已删除；
- 无 `tutti_phase7_*` scratch file 或临时 config 目录；
- 无未释放 allocation 或 queue reservation（停止前最终 ledger 与基线精确相同）。

## 7. Exit Gate 与残余风险

| Gate | 判定 |
| --- | --- |
| A0/A1/A2 correctness | 辅助模式 PASS；严格双 accelerator 矩阵在场景前 SKIP |
| B0/B1/B2 correctness | **SKIP，未验证** |
| 每次 shutdown ledger 精确恢复 | 已执行的 A0/A1/A2 PASS |
| striped 两盘原子 allocation/release | accelerator 0 PASS；accelerator 1 未验证 |
| daemon clean shutdown 和零残留 | PASS |
| HOST/CUDA 非硬件回归 | PASS |
| P7 总体 | **未通过，不得宣称目标设计完成** |

唯一阻塞项是第二块可用 accelerator 及其 daemon view/两盘 ACL。获得该硬件并在 daemon
配置中声明 accelerator 1、允许两盘 ACL `[0,1]` 后，应直接重跑严格入口；只有 A0–B2
全部 PASS 且再次完成零残留清理，才可把 P7 和整个目标设计标记为完成。

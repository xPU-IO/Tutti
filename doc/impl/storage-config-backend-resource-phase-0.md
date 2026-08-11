# Storage Config、Backend 与 Resource 阶段 0 实施记录

## 1. 阶段范围与结论

本阶段基于 `doc/impl/storage-config-backend-resource-implementation-plan.md` 的 P0
建立迁移前基线和测试观测面。基线 commit 为 `f5e82d6`，分支为
`fix/multi-gpu`，开始实施时工作树干净；测试日期为 2026-08-11 UTC。

阶段结论：P0 exit gate 通过。产品 loader、daemon schema、RPC 和
`StorageRuntime` 业务契约均未改变；HOST/CUDA 非硬件回归全绿，fake client 已覆盖
Acquire/Release 和 runtime factory 失败回滚。双盘 daemon 的单盘与 striped
allocation、controller attach、queue group 创建/销毁和 reservation 回收均通过。
完整 loader 硬件 E2E 要求两个 accelerator，本机只有一个 accelerator，因此按测试
契约返回 `SKIP(77)`，未将缺失硬件报告为通过。

## 2. 本次实现修改

| 计划项 | 本次实现 | 结果 |
| --- | --- | --- |
| 冻结分支、工作树和 build 基线 | 记录 `f5e82d6`、干净工作树、现有 `build/host` 与 `build/cuda` | 完成 |
| 固定 fake client ledger | 确认 `list_accelerators_calls`、`list_resources_calls`、`acquire_calls`、`release_calls` 均有测试观测 | 完成 |
| 固定 runtime factory 失败注入 | 确认 Acquire 后 factory 失败触发一次 Release，Acquire 失败不 Release，静态 preflight 失败不 Acquire | 完成 |
| 盘点 public allocation 字段 | 记录 `runtime_bundle_loader_contract_test.cpp` 对 `allocation_id`、`allocation_slices` 的直接依赖 | 完成 |
| 稳定并发测试观测面 | 修复 `MockDataPath::unblock_progress()` 的条件变量丢通知竞态，并以 100 次压力复跑验证 | 完成 |
| 核对本机 daemon 配置 | 将实施计划中的路径修正为实际存在且由用户指定的 `config/local/daemon_2_disk.yaml` | 完成 |

## 3. 业务逻辑变化

产品业务逻辑无变化。本阶段唯一代码行为变化位于测试工具：

- `MockDataPath::unblock_progress()` 现在持有与条件变量配套的 mutex 后清除原子阻塞
  标志，再执行 `notify_all()`，避免 waiter 在 predicate 检查和休眠之间丢失通知；
- `test_progress_serialization()` 不再在 mutex 外重复清除阻塞标志，并删除未使用的
  `done` 原子变量；
- 修改只影响硬件无关 contract test 的确定性，不改变 `DataPath` SPI 或
  `StorageRuntime` 的产品实现。

问题在未修改基线上可重复表现为 `tutti_storage_runtime_contract_test` 卡在 test 30：
一个 query 线程等待 `block_progress_cv_`，另一个 query 线程等待同一 DataPath 的
progress serialization mutex，主线程等待 join。修复后同一二进制连续运行 100 次均通过。

## 4. 文件变化

| 文件 | 变化 |
| --- | --- |
| `tutti/testing/mock_data_path.h` | 以条件变量 mutex 保护 unblock 状态转换，消除丢通知窗口 |
| `tests/storage_runtime_contract/storage_runtime_contract_test.cpp` | 统一通过 `unblock_progress()` 解锁，删除冗余状态 |
| `doc/impl/storage-config-backend-resource-implementation-plan.md` | 修正四处本机 daemon 配置文件名 |
| `doc/impl/storage-config-backend-resource-phase-0.md` | 新增本阶段实施、业务变化、文件变化和测试证据 |

本阶段未修改 `config/local/daemon_2_disk.yaml`、daemon schema/protobuf/RPC、loader
装配逻辑或 `StorageRuntime` 路由逻辑。

## 5. 非硬件验证结果

| 命令/场景 | 结果 | 原始结果摘要 |
| --- | --- | --- |
| `cmake --build build/host --parallel 8` | PASS | HOST 全量构建成功 |
| `ctest --test-dir build/host --output-on-failure -j 8` | PASS | 17/17，通过时间 0.25 秒 |
| 连续运行 `tutti_storage_runtime_contract_test` 100 次 | PASS | 100/100，无超时或失败 |
| `cmake --build build/cuda --parallel 8` | PASS | CUDA 12.8.93、架构 89，全量构建成功 |
| `ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8` | PASS | 19/19，通过时间 0.48 秒 |
| fake single-slice loader | PASS | Acquire 1 次、shutdown Release 1 次、析构不重复 Release |
| fake striped loader | PASS | 有序 2 slices，Acquire/Release 各 1 次 |
| fake runtime factory 失败 | PASS | Acquire 后 Release 恰好 1 次 |
| fake preflight/acquire 失败 | PASS | 缺 accelerator 时 Acquire/Release 均 0；Acquire 失败时 Release 0 |

首次 HOST 并行运行暴露了上述既有条件变量竞态；在测试 seam 修复前，单独复跑亦可
超时。该问题不是本阶段改动引入，因此没有把初次卡住计为产品回归；修复后完整 HOST
回归和 100 次压力复跑均通过。

## 6. 基于硬件的验证结果

### 6.1 环境和 daemon 基线

- accelerator：1 块 NVIDIA L40S，`accel_id=0`；
- daemon 配置：`config/local/daemon_2_disk.yaml`；endpoint：`127.0.0.1:50051`；
- NVMe 0：BDF `0000:b1:00.0`、`/dev/ssnvme0`、`/dev/snvme0n1`、
  backing `/mnt/nvme0`、namespace 1、LBA 4096、BAR0 16384、可用 user queues 23；
- NVMe 1：BDF `0000:e3:00.0`、`/dev/ssnvme1`、`/dev/snvme1n1`、
  backing `/mnt/nvme1`、namespace 1、LBA 4096、BAR0 16384、可用 user queues 72；
- 两块盘均允许 `accel_id=0`，初始 ledger 均为 `reserved=0`。

### 6.2 分配、attach 与回收

| 场景 | 结果 | metadata/ledger 证据 |
| --- | --- | --- |
| explicit device 0、4 queues | PASS | 返回 device 0、`/dev/ssnvme0`、`/dev/snvme0n1`、view `/mnt/gpu0/ssnvme0`、granted 4；attach/group/destroy/detach 全部成功 |
| explicit device 1、4 queues | PASS | 返回 device 1、`/dev/ssnvme1`、`/dev/snvme1n1`、view `/mnt/gpu0/ssnvme1`、granted 4；attach/group/destroy/detach 全部成功 |
| striped `[0,1]`、每盘 4 queues | PASS | 单个 allocation 返回 2 个有序 slice；两盘分别完成 attach/group/destroy/detach |
| 每个场景后的 snapshot | PASS | device 0 回到 `reserved=0 available=23`；device 1 回到 `reserved=0 available=72` |
| 完整 loader E2E | SKIP | `tutti_runtime_bundle_loader_contract_test` 返回 77：测试要求两个 accelerator，本机只有一个 |
| daemon 停止和清理 | PASS | 单次 SIGTERM；日志为 `tutti_daemon exited cleanly`；两处 mount、view symlink、device node、50051 listener 和 daemon 进程均无残留 |

硬件场景使用真实 RPC 返回的 device ID、chrdev、block path、view path 和 granted queues，
未从应用配置推导这些字段。完整 loader 的单盘/striped I/O 未在本机执行，原因仅为现有
E2E 对双 accelerator 的硬件前置条件；P0 对应项明确记录为 SKIP，不扩大为 PASS。

## 7. P1 输入约束

- P1 新 parser test 必须继续使用计数 client 证明所有静态错误的 list/acquire 均为 0；
- `ParsedConfig` 和 legacy loader 装配保持兼容，P1 不删除 public allocation 字段；
- `runtime_bundle_loader_contract_test.cpp` 仍直接依赖 `allocation_id`、
  `allocation_slices`、`resolver_schemes` 和 `data_path_keys`，这些依赖留待 P2/P6 迁移；
- 本机缺少第二个 accelerator，P1 硬件回归只能验证单 accelerator 控制面，完整双
  accelerator loader E2E 必须继续明确 SKIP。

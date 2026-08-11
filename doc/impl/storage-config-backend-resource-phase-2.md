# Storage Config、Backend 与 Resource 阶段 2 实施记录

## 1. 阶段范围与结论

本阶段实现
`doc/impl/storage-config-backend-resource-implementation-plan.md` 的 P2：将
`TuttiRuntime` 的声明、组件 ownership、shutdown、失败清理和诊断 seam 从 config loader
中拆出，保持 P1 的 canonical/legacy 解析和旧 loader 装配结果不变。

阶段基线为 P1 模块化 parser commit `b0ffadb`，分支为 `fix/multi-gpu`，测试日期为
2026-08-11（Asia/Tokyo）。P2 exit gate 通过：生命周期实现已不再位于
`tutti/config/tutti_config.cpp`，正常关闭、析构兜底、runtime factory 失败、shutdown
错误、release 错误和重复 shutdown 均有测试证据；HOST/CUDA 非硬件回归全绿。实机 daemon
双盘控制面和 I/O 回归通过，完整 loader E2E 因本机只有一块 accelerator 按约定返回
`SKIP`，没有将其计为通过。

## 2. 计划项与本次实现对比

| P2 计划项 | 本次实现 | 结果 |
| --- | --- | --- |
| 新增独立 Runtime 头/源文件 | 新增 `tutti/include/tutti/tutti_runtime.h` 和 `tutti/tutti_runtime/tutti_runtime.cpp` | 完成 |
| 迁出 `TuttiRuntime` 声明和析构/shutdown | `tutti_config.h` 只包含 Runtime 公共头；`tutti_config.cpp` 不再定义生命周期 | 完成 |
| 保持旧 loader 装配结果 | local/striped 仍构造相同 resolver、DataPath、route key 和 `RuntimeComponents` | 完成 |
| 显式 shutdown 状态机 | `RUNNING -> SHUTTING_DOWN -> STOPPED`；STOPPED 再次 shutdown 返回 OK | 完成 |
| first-error 传播 | 记录 runtime shutdown、release 或异常中的第一个错误，随后仍执行所有清理 | 完成 |
| reverse cleanup | 先 shutdown/destroy `StorageRuntime`，再逆序销毁 DataPath/resolver，最后尝试 release allocation | 完成 |
| ownership registry 操作 | `register_datapath()` / `register_resolver()` 负责转移 ownership 和登记 route key；shutdown 后拒绝新登记 | 完成 |
| 失败路径不二次 Release | release 尝试后立即占用 release guard；析构和重复 shutdown 不会再次调用 provider | 完成 |
| 临时只读 inspection seam | `inspection()`/`inspect()` 返回 allocation、slice、route 和 lifecycle state 的副本，不返回 ownership | 完成 |
| 可测试的 runtime/failure 注入 | `LoadTuttiConfigOptions` 支持 runtime shutdown hook 和 stage observer，仅供 contract test | 完成 |

## 3. 业务逻辑变化

### 3.1 Runtime ownership 边界

`TuttiRuntime` 继续是应用侧 aggregate，仍拥有 `StorageRuntime`、DataPath、resolver、
resource client 和兼容 allocation 字段；本阶段没有提前删除 P6 才收口的 public 平行
字段。变化是 ownership 相关操作现在只经过 Runtime 模块：loader 在创建组件后调用
`register_datapath()` / `register_resolver()`，返回的裸指针只作为
`RuntimeComponents` 的借用绑定，不再由 loader 直接维护 owning vector 和 route 诊断表。

登记只能发生在 `RUNNING` 状态；shutdown 开始后传入的组件被拒绝并由调用方的
`unique_ptr` 回收。登记 route key 的异常会回滚刚加入的 owning vector，避免出现“组件已
拥有但 route 未登记”的半状态。

### 3.2 关闭顺序和状态

`TuttiRuntime::shutdown()` 现在使用显式状态和 first-error accumulator：

```text
RUNNING
  -> SHUTTING_DOWN
       -> StorageRuntime::shutdown(0) / injected hook
       -> destroy StorageRuntime
       -> destroy resolvers (reverse construction order)
       -> destroy DataPaths (reverse construction order)
       -> release allocation at most once
       -> destroy resource client
  -> STOPPED
```

即使 runtime shutdown 返回 `TIMEOUT`、provider release 返回错误，仍会继续销毁组件并尝试
释放资源；返回值是最早出现的错误。shutdown hook 或 release 实现抛出异常时转换为
`INTERNAL`，不会阻断剩余清理。析构只调用同一个幂等 shutdown 路径，显式 shutdown 后的
析构不产生第二次 Release。

route 字符串保留为只读诊断快照，owning DataPath/resolver 对象在对应 stage 已销毁；这
保持了旧测试读取 route key 的兼容性，同时不留下可用的组件指针。

### 3.3 只读诊断

`TuttiRuntimeInspection` 是值拷贝，包含 lifecycle state、allocation ID、slice metadata、
resolver scheme、DataPath key 和 release 标志。修改返回值不会修改 runtime 内部对象，
也不能替换 client 或 allocation ownership。P6 收口时可将现有 public 字段迁移到此 seam
或 ResourceInfo/backend manifest，而不需要再次改动关闭实现。

## 4. 文件变化

| 文件 | 变化 |
| --- | --- |
| `tutti/include/tutti/tutti_runtime.h` | 新增 Runtime 公共生命周期类型、NVMe 兼容值类型、inspection、登记 API 和测试 seam |
| `tutti/tutti_runtime/tutti_runtime.cpp` | 新增析构、状态机、first-error 清理、reverse ownership cleanup、登记回滚和 inspection copy 实现 |
| `tutti/config/tutti_config.h` | 删除 Runtime/NVMe 运行时类型的重复声明，包含新公共头；增加 lifecycle 注入 options |
| `tutti/config/tutti_config.cpp` | 删除 TuttiRuntime 析构/shutdown 实现；local/striped 装配改用 Runtime 登记 API；传递 lifecycle seam |
| `tutti/config/CMakeLists.txt` | 将 `tutti_runtime/tutti_runtime.cpp` 加入 full loader target；parse target 仍无 CUDA/daemon 依赖 |
| `tests/config_loader/config_loader_test.cpp` | 增加 state/inspection copy、析构兜底、stage 顺序、runtime shutdown first-error、release error 和幂等性测试 |
| `doc/impl/storage-config-backend-resource-phase-2.md` | 新增本阶段实施和验证记录 |

本阶段未修改 canonical schema、legacy adapter、daemon YAML/protobuf/gRPC、
`StorageRuntime` 路由契约、resolver/DataPath SPI 或 `config/local/daemon_2_disk.yaml`。

## 5. 软件验证结果

| 命令/target | 结果 | 原始结果摘要 |
| --- | --- | --- |
| `cmake --build build/host --parallel 8` | PASS | parser target 和所有 HOST 测试构建成功 |
| `cmake --build build/cuda --parallel 8` | PASS | full loader、runtime bundle E2E 和 daemon 相关 target 构建成功 |
| `build/cuda/bin/tutti_config_loader_test` | PASS | `passed: 147`, `failed: 0` |
| `ctest --test-dir build/host --output-on-failure -j 8` | PASS | 18/18，通过时间 0.26 秒 |
| `ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8` | PASS | 20/20，通过时间 0.59 秒 |
| `git diff --check` | PASS | 无 whitespace error |

生命周期用例具体证明：正常 shutdown 两次只 release 一次；不显式 shutdown 的析构会
释放一次；stage 顺序为 runtime shutdown/destroy、DataPath、resolver、allocation release、
complete；注入的 runtime `TIMEOUT` 或 provider release `DEVICE_ERROR` 会返回且不阻止
剩余清理；inspection 返回值是独立副本。

## 6. 基于硬件的验证结果

### 6.1 环境和 daemon 基线

- daemon 配置：`config/local/daemon_2_disk.yaml`；endpoint `127.0.0.1:50051`；
  daemon 以 `sudo` 启动，密码未写入命令、日志或本文；
- accelerator：仅发现 `accel_id=0`，view root `/mnt/gpu0`；
- device 0：BDF `0000:b1:00.0`，chrdev `/dev/ssnvme0`，block
  `/dev/snvme0n1`，backing `/mnt/nvme0`，namespace `1`，logical block `4096`，BAR0
  `16384`，available queues `23`；
- device 1：BDF `0000:e3:00.0`，chrdev `/dev/ssnvme1`，block
  `/dev/snvme1n1`，backing `/mnt/nvme1`，namespace `1`，logical block `4096`，BAR0
  `16384`，available queues `72`；
- 初始及所有 allocation 后 `ListNvmeResources` ledger 均为：device 0
  `reserved=0 available_queues=23`，device 1 `reserved=0 available_queues=72`。

### 6.2 allocation/I/O 场景

| 场景 | 结果 | 证据 |
| --- | --- | --- |
| explicit device 0, 4 queues, `--skip-io` | PASS | attach `/dev/ssnvme0`、create group grant 4、destroy/free；返回 device 0 的真实 path/view/grant |
| explicit device 1, 4 queues | PASS | attach/create group；`Write+Read+verify x 4 IOs`；destroy/free；返回 `/dev/ssnvme1`、`/dev/snvme1n1`、`/mnt/gpu0/ssnvme1`、grant 4 |
| striped `[0,1]`, 4 queues/controller | PASS | 两个有序 slice 分别 attach；每个 shard `Write+Read+verify x 4 IOs`；均 destroy/free；返回两盘真实 metadata |
| allocation 后 reservation ledger | PASS | 两次单盘和一次 striped 完成后均精确回到 device 0 `0/23`、device 1 `0/72` |
| `tutti_runtime_bundle_loader_contract_test` | SKIP | exit 77，测试要求两个 accelerator，本机 daemon 只发现 accelerator 0；未将缺少前置条件报告为 I/O PASS |

这些实机场景直接验证 daemon 返回的 BDF、chrdev/block、view/backing、namespace、LBA、
BAR0 和 granted queue 可用且 allocation 清理完成；P2 新增的 inspection seam 和
`TuttiRuntime` first-error 状态机由无硬件 fake-client contract 覆盖，未用设备 ID/数组顺序
推导运行时路径。

### 6.3 daemon 清理

测试结束只发送一次 Ctrl-C/SIGTERM，daemon 输出 `tutti_daemon exited cleanly`，并输出
两处 `mount_manager: unmounted /mnt/nvme0`、`/mnt/nvme1`。随后检查无 daemon/nvmeservice
进程、无 50051 listener、无 `/mnt/nvme[01]` mount，以及 `/dev/ssnvme0`、`/dev/ssnvme1`、
`/dev/snvme0n1`、`/dev/snvme1n1` 均已移除。

## 7. Exit gate 与后续约束

- Runtime 生命周期代码已从 config loader 文件分离，loader 只负责解析和装配；
- shutdown 状态、first-error、reverse cleanup、析构兜底和 at-most-once release 均有
  可复现测试；
- inspection 仅返回值拷贝，暂不作为最终 Resource API；
- 兼容 public allocation/client 字段仍保留，P3/P4 将把 allocation ownership 迁入
  `NvmeResource` 和 resource registry；
- P3 不得把 `RuntimeResourceClient` 提升为公共 Resource SPI，也不得绕过本阶段 Runtime
  cleanup；P6 再删除兼容平行字段。

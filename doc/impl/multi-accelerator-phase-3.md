# 多加速器 Runtime 阶段 3 实施记录

## 基线与范围

- 分支：`fix/multi-gpu`。
- 实施起点：`63662f8`（`docs: phase3 handoff`），阶段 2 基线为
  `61af360`。
- 本记录只覆盖阶段 3 的 daemon resource、bring-up metadata、RPC compatibility、
  原子 slice allocation 和回收；没有进入阶段 4 loader/Runtime bundle。

## 实施内容

- `nvmeservice_config` 以 `accelerators[].accel_id/view_root`、显式
  `nvmes[].device_id`、`backing_mount_path`、`allowed_accel_ids` 为 canonical
  model；PCI BDF 在 parser 中规范化并校验。完整 legacy 文件只保留一版读取窗口，
  通过 `ConfigDiagnostics` 发出可断言 deprecation warning；顶层、accelerator entry
  和 NVMe entry 的新旧字段混用均失败。
- libnvm owner bring-up 新增 `nvm_owner_bringup_result`，直接传递实际 chrdev
  minor/path、ioctl `disk_name` 和 `/dev/<disk_name>` block path。daemon 做 BDF、字符
  设备 minor、block path 和 sysfs BDF 关联的 fail-closed 校验。
- `ServiceState` 使用显式 `device_id` map，并提供可注入 resource snapshot seam。
  `allowed`、`explicit`、`striped` 在一个锁临界区中完成 ACL、view、availability、
  queue budget 校验和全量 reservation；striped 对外只有一个 allocation ID。
- `Release`、旧 `Disconnect`、heartbeat timeout、dead/reused PID reaper 共享同一
  锁内退款 helper；重复 Release 幂等成功，不会重复增加预算。daemon 不调用
  accelerator runtime。
- 新增 `ListAccelerators`、`ListNvmeResources`、`AcquireNvmeSlices`、`Release`，
  保留旧 `ListDevices/Connect/Disconnect/Heartbeat` protobuf 方法和字段编号。CLI
  默认使用 `--accel`，`--cuda` 仅走兼容入口；配置、生成器和 NVMeService/daemon
  文档迁移到 canonical schema。
- 新增 `nvmeservice_phase3_contract`（无 root/GPU/NVMe/gRPC daemon 依赖的 fake
  resource state seam），覆盖 parser、metadata validation、selection、ACL、预算、
  striped 原子性、Release/reaper、并发 ledger、数组反转不变性、view root fail-closed、
  新旧 RPC metadata/字段编号和 daemon no-context 控制面路径。

## 无硬件验证

实际命令与结果：

```text
cmake --preset host
cmake --build --preset host --parallel 8
ctest --test-dir build/host --output-on-failure -j 8
  17/17 passed

cmake --preset cuda
cmake --build --preset cuda --parallel 8
ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8
  19/19 passed

cmake --preset cuda-module
cmake --build --preset cuda-module --parallel 8 --target nvmeservice tutti_daemon nvmeservice_client_example
cmake --build --preset cuda-module --parallel 8
ctest --test-dir build/cuda-module -LE hardware --output-on-failure -j 8
  19/19 passed

build/cuda/tutti/device_manager/nvme/nvmeservice/nvmeservice_phase3_contract_test
  nvmeservice phase 3 contract: PASS (83 checks)
```

首次 HOST 并行回归遇到既有 `test_progress_serialization` mock 条件变量卡死；线程
栈在 `StorageRuntime::drive_progress_unlocked_`/`MockDataPath::progress`，与本阶段
改动无交集。终止该次进程后单独复跑 40/40，并在最终完整 HOST 回归中 17/17 通过。

## 双盘实机验证

测试前 GPU compute process 列表记录为 6 个既有 LMServe Python 进程；没有终止、暂停
或干预这些进程。用 `config/local/daemon_2disk.yaml` 启动：

```text
sudo -S env TUTTI_VERBOSE=1 build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon --config config/local/daemon_2disk.yaml < ~/.passwd/1
```

owner/RPC 事实：

| device_id | pci_bdf | chrdev | block | backing mount | capacity | block/page | queue depth | BAR0 |
|---:|---|---|---|---|---:|---|---:|---:|
| 0 | `0000:41:00.0` | `/dev/ssnvme0` (minor 0) | `/dev/snvme0n1` | `/mnt/snvme/nvme1` | 96 | 4096/4096 | 1024 | 32768 |
| 1 | `0000:44:00.0` | `/dev/ssnvme1` (minor 1) | `/dev/snvme1n1` | `/mnt/snvme/nvme2` | 96 | 4096/4096 | 1024 | 32768 |

两盘均报告 `max_queues_per_group=16`；view 发布为
`/mnt/snvme/gpu{0,1}/ssnvme{0,1}`，目标 accelerator 与 backing `ACCEL<n>` 目录
对应正确。

- accel 0/1 `allowed`：各 1 slice，真实 GPU write/read/verify 4 IO，通过。
- accel 0 explicit device 1、accel 1 explicit device 0：各 1 slice，真实 I/O 通过。
- accel 0 striped `[0,1]`：一个 allocation、两个按请求顺序 slice；两盘各完成
  LBA `2621440..2621443` 的 write/read/verify，通过后一次 Release。
- 同一 NVMe 并发 6 个 allocation，各请求 16 queues，ledger 达到 `reserved=96`；
  第 7 个请求明确返回 `controller queue budget is insufficient`。释放后资源回到
  `reserved=0/available_queues=96`，另一盘不受影响。
- client SIGTERM 异常退出后，reaper 在约 7 秒内将 2 queues 精确退回 0；旧
  `--cuda` Connect/attach/Disconnect compatibility smoke 通过。
- list/acquire 前后 daemon 未出现在 GPU compute process 列表；外部 LMServe 进程
  期间发生 PID/显存变化，但原有服务未被 daemon 干预。

单次 Ctrl-C 后 daemon 输出 clean exit；复核结果：`:50051` 不监听，两个 BDF 均为
unbound，chrdev/block 节点、两个 mount、四个 GPU view symlink、allocation、queue
reservation 和测试进程均无残留。GPU 1 的外部 compute 服务仍在运行。

## 残余风险

- 阶段 2 记录的同一 `LocalNvmeDataPath` 双线程 `ops_` 遍历 SIGSEGV 风险仍未扩展
  修复；本阶段 fake allocator/RPC 测试不触及该路径。
- daemon `available` 仍表示 bring-up/mount/view/queue admission 可用，不是
  `validated_available`；真实 I/O probe 继续由 client/后续 loader 负责。
- 阶段 4 allocation-driven loader、owned Runtime bundle 和 `tutti/config` RPC 消费
  尚未实施，按要求在此停止。

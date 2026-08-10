# 多加速器 Runtime 阶段 5 实施记录

## 基线与范围

- 分支：`fix/multi-gpu`。
- 实施起点：`e96ec66`（`docs: add multi-accelerator phase 5 handoff`），阶段 4
  基线为 `a493603`。
- 本阶段只实现并验证单个 allocation-driven Runtime bundle 的 Local/Striped 实机
  I/O 闭环；没有实现双 Runtime 并发、跨 Runtime 负例或阶段 6 的共享资源矩阵。
- daemon 使用用户指定的 `config/local/daemon_2disk.yaml`，应用配置由测试在 `/tmp`
  临时生成，不修改仓库内 `config/tutti_config.yaml`，也不包含任何设备路径。

## 实施内容

- 新增 `tutti_runtime_bundle_loader_contract_test` 硬件 contract target。测试只通过
  `load_tutti_config()` 获取 owned `TuttiRuntime`，再使用 public
  `StorageRuntime` API 完成 `open/register_memory/submit/query(wait-progress)/wait/
  release_io/close/unregister_memory/shutdown`。
- 测试为 GPU 0/1 分别生成 explicit device 0、explicit device 1 和 striped
  `[0,1]` 配置。两个 explicit 配置除 `device_id` 外保持一致；配置中不出现
  chrdev、block、backing 或 view path。
- Local 场景在 allocation 返回的 `view_path` 下创建唯一 scratch file，完成真实
  write/read/byte verify。
- Striped 场景在两个 allocation `view_path` 下创建同名 shard，覆盖 64 KiB stripe
  边界两侧的 8 KiB I/O、包含三个 accepted request 的 mixed batch，以及完整
  `TuttiRuntime` shutdown/Release 后重新 load/open/read 的 persistence。
- 每次调用 StorageRuntime API 前把调用线程 current device 设为另一张 GPU，并在 API
  返回后断言仍为原 caller device。初始化由 `load_tutti_config()` 内部的
  `StorageRuntime::create()` 覆盖；`query()` 作为 public progress observation 覆盖。
- 测试把 bundle 保存的 slice metadata 与独立 `ListNvmeResources` 快照逐项核对，记录
  owner/RPC 返回的 PCI BDF、chrdev、block、backing/view path、namespace、LBA、
  BAR0、MDTS 和 queue grant；同时断言 view 与 backing 位于同一实际文件系统。
- 每次 load 后检查选中 controller 的 reservation 精确增加 4，bundle shutdown 后
  轮询并断言所有 controller 的 ledger 精确回到场景基线。striped 的单个
  allocation ID 同时预留并一次 Release 两盘。
- 测试断言 Local bundle 只发布 `file + local-nvme-ext4` 一组顶层 binding，Striped
  bundle 只发布 `striped + striped-local-nvme` 一组顶层 binding。
- 修正既有 `tutti_striped_local_nvme_contract_test` 的测试 target 链接设置，启用 CUDA
  separable compilation 并由 CUDA linker 收尾；否则 handoff 要求构建该 target 时会
  缺少 `__cudaRegisterLinkedBinary`。未修改其测试业务逻辑。

## 硬件环境与资源事实

验证日期为 2026-08-10，CUDA toolkit 为 12.8.93，kernel 为 5.15.0，snvme 模块已
加载。两张 accelerator 均为 NVIDIA H100 PCIe：

| accel_id | daemon view root | caller current device |
|---:|---|---:|
| 0 | `/mnt/snvme/gpu0` | 1 |
| 1 | `/mnt/snvme/gpu1` | 0 |

daemon owner 与 `ListNvmeResources` 返回的资源事实：

| device_id | PCI BDF | chrdev | block | backing mount | namespace | LBA/page | BAR0 | MDTS | queues |
|---:|---|---|---|---|---:|---|---:|---:|---|
| 0 | `0000:41:00.0` | `/dev/ssnvme0` | `/dev/snvme0n1` | `/mnt/snvme/nvme1` | 1 | 4096/4096 | 32768 | 1048576 | capacity 96, max/group 16 |
| 1 | `0000:44:00.0` | `/dev/ssnvme1` | `/dev/snvme1n1` | `/mnt/snvme/nvme2` | 1 | 4096/4096 | 32768 | 1048576 | capacity 96, max/group 16 |

实际 allocation view path 由 accelerator 和 resource 组合返回：

| accel_id | device 0 view | device 1 view |
|---:|---|---|
| 0 | `/mnt/snvme/gpu0/ssnvme0` | `/mnt/snvme/gpu0/ssnvme1` |
| 1 | `/mnt/snvme/gpu1/ssnvme0` | `/mnt/snvme/gpu1/ssnvme1` |

每个 slice 的 `granted_queues=4`。测试没有从上述 ID、数组顺序或配置推导任何 path。

## 构建与无硬件回归

执行：

```text
cmake --preset cuda
cmake --build --preset cuda --parallel 8 --target \
  tutti_daemon nvmeservice_client_example tutti_config_loader_test \
  tutti_storage_runtime_contract_test \
  tutti_striped_local_nvme_contract_test \
  tutti_runtime_bundle_loader_contract_test
```

全部 target 构建成功。handoff 指定的最小回归：

```text
cmake --build --preset cuda --parallel 8 --target \
  tutti_config_loader_test \
  tutti_storage_runtime_contract_test \
  nvmeservice_phase3_contract_test
ctest --test-dir build/cuda --output-on-failure -R \
  'tutti_config_loader_test|tutti_storage_runtime_contract_test|nvmeservice_phase3_contract'

3/3 passed
```

额外运行完整非硬件 CUDA 集合：

```text
ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8

19/19 passed
```

本阶段只新增硬件测试/CMake 注册，没有修改 HOST profile 可见头文件或产品实现，因此
没有额外运行 HOST 全量构建。

## 实机执行与结果

daemon 启动和资源基线命令：

```text
sudo -S env TUTTI_VERBOSE=1 \
  build/cuda/tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon \
  --config config/local/daemon_2disk.yaml < ~/.passwd/1

build/cuda/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client \
  --endpoint 127.0.0.1:50051 --list-only
```

初始两盘均为 `reserved=0, available_queues=96, available=true`。daemon 创建的
accelerator backing 子目录为 `root:root 0755`，所以权威 I/O run 按现有部署权限约束
使用 root client：

```text
sudo -S env build/cuda/bin/tutti_runtime_bundle_loader_contract_test \
  --endpoint 127.0.0.1:50051 \
  --accel0 0 --accel1 1 --device0 0 --device1 1 --queues 4 \
  < ~/.passwd/1

phase5 checks=447 failures=0 result=PASS
```

场景结果：

| ID | 场景 | I/O 与生命周期 | ledger |
|---|---|---|---|
| A1 | accel 0 + explicit device 0 | Local write/read/verify PASS | device 0: `0/96 -> 4/92 -> 0/96` |
| A2 | accel 0 + explicit device 1 | Local write/read/verify PASS | device 1: `0/96 -> 4/92 -> 0/96` |
| A3 | accel 0 + striped `[0,1]` | boundary/mixed/restart PASS | both: `0/96 -> 4/92 -> 0/96`, repeated for restart |
| A4 | accel 1 + explicit device 0 | Local write/read/verify PASS | device 0: `0/96 -> 4/92 -> 0/96` |
| A5 | accel 1 + explicit device 1 | Local write/read/verify PASS | device 1: `0/96 -> 4/92 -> 0/96` |
| A6 | accel 1 + striped `[0,1]` | boundary/mixed/restart PASS | both: `0/96 -> 4/92 -> 0/96`, repeated for restart |

测试实际生成并在成功后删除的路径：

```text
/tmp/tutti_phase5_4w0skc/A1.yaml ... A6.yaml
/mnt/snvme/gpu0/ssnvme0/tutti_phase5_2612260_A1.bin
/mnt/snvme/gpu0/ssnvme1/tutti_phase5_2612260_A2.bin
/mnt/snvme/gpu0/ssnvme0/striped/tutti_phase5_2612260_A3.shard0
/mnt/snvme/gpu0/ssnvme1/striped/tutti_phase5_2612260_A3.shard1
/mnt/snvme/gpu1/ssnvme0/tutti_phase5_2612260_A4.bin
/mnt/snvme/gpu1/ssnvme1/tutti_phase5_2612260_A5.bin
/mnt/snvme/gpu1/ssnvme0/striped/tutti_phase5_2612260_A6.shard0
/mnt/snvme/gpu1/ssnvme1/striped/tutti_phase5_2612260_A6.shard1
```

GPU 1 在测试前已有 6 个外部 Python compute process。daemon boot、`list-only` 和
Acquire 前的 `nvidia-smi --query-compute-apps` 快照与 daemon 启动前一致，没有
`tutti_daemon`；测试退出后快照再次回到相同的 6 个外部进程，证明 daemon 没有创建
accelerator context。

## 清理状态

- 最终 `ListNvmeResources`：两盘 `reserved=0, available_queues=96`。
- 未发现 `tutti_phase5_*` scratch、临时配置目录或测试进程。
- daemon 收到 Ctrl-C 后报告两盘 unmounted，并以 `tutti_daemon exited cleanly`
  退出。
- daemon 退出后两处 mount、四个 accelerator view symlink、chrdev 和 block 节点均已
  清理；snvme module 保持加载，符合本机既有测试环境约定。

## 残余风险

- 本阶段是严格顺序的单 Runtime 验证，没有覆盖双 Runtime 并发、共享 controller、
  双 striped allocation 并发或跨 Runtime handle/pointer/stream/context；这些属于
  阶段 6。
- daemon 创建的 accelerator backing 子目录默认不可由普通用户写，当前硬件 E2E
  需要 root client，或由部署方预先配置 owner/group/ACL。非特权诊断能完成
  Acquire/Release，但不能创建 scratch；该诊断不计入本阶段 I/O 验收。
- 阶段 2/3 已记录的 Local DataPath `ops_` 并发风险未在本阶段触发或扩修；本阶段只
  覆盖单 Runtime 顺序调用。
- GPU 1 与外部工作负载共享，可能影响性能数据；阶段 5 不作性能结论，真实数据正确性
  和资源回收不受影响。

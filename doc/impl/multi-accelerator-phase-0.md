# 多加速器 Runtime 阶段 0 实施记录

## 范围

本阶段冻结并建立 `doc/design/multi-accelerator-runtime.md` 第 12.1 节要求的基线，
不修改业务行为：

- `StorageRuntime` 核心继续支持多 scheme resolver 和 multi-key DataPath；产品
  loader 的目标装配边界为一个 Runtime、一个 accelerator、一组顶层组件。
- `allowed` 选择一个 slice，`explicit` 必须选择一个 slice，`striped` 必须选择
  两个或更多 slice 且原子成功或失败。
- canonical daemon schema 采用 `accelerators`、`accel_id`、`view_root`、
  `allowed_accel_ids`、`backing_mount_path` 和显式 `device_id`；阶段 3
  迁移前，现有 `gpus`/`allowed_gpus` 仅作为 legacy baseline。
- 修正 local-NVMe 测试文档中的双盘配置文件名和已漂移的 BDF、mount、BAR0；测试
  参数优先取 daemon RPC metadata。

基线代码：`41b8f95`（`fix/multi-gpu`）。测试日期：2026-08-10 UTC。

## 环境

- GPU 0：NVIDIA H100 PCIe，backend ordinal `accel_id=0`，BDF `0000:4b:00.0`。
- GPU 1：NVIDIA H100 PCIe，backend ordinal `accel_id=1`，BDF `0000:4c:00.0`。
- daemon 配置：`config/local/daemon_2disk.yaml`。
- daemon NVMe 0：BDF `0000:41:00.0`，`/dev/ssnvme0`，
  `/dev/snvme0n1`，backing mount `/mnt/snvme/nvme1`。
- daemon NVMe 1：BDF `0000:44:00.0`，`/dev/ssnvme1`，
  `/dev/snvme1n1`，backing mount `/mnt/snvme/nvme2`。
- 两个 namespace 的 block size 均为 4096，BAR0 size 均为 32768；两个 GPU 均在
  两块盘的 legacy ACL 中。

## 验证结果

| 验证 | 结果 | 证据摘要 |
| --- | --- | --- |
| HOST configure/build | PASS | `cmake --preset host --fresh`；完整构建成功 |
| HOST 无硬件 contract | PASS | `ctest --preset host`，16/16；包含 StorageRuntime 历史 test 83 的跨 DataPath 分组回归 |
| CUDA configure/build | PASS | CUDA 12.8.93、sm_90；完整构建成功 |
| CUDA 无硬件 contract | PASS | `ctest --test-dir build/cuda -LE hardware ...`，17/17 |
| daemon bring-up/list | PASS | 两个 resource 的 RPC BDF、chrdev、namespace、block size 与配置/bring-up 一致 |
| daemon Connect/Disconnect | PASS | device 0/GPU 0 与 device 1/GPU 1 均以 4 queues 完成 attach、建组、销毁和 detach |
| daemon accelerator context | PASS | bring-up/list 后 `nvidia-smi --query-compute-apps` 无 daemon 进程 |
| LocalNvmeDataPath hardware | PASS | GPU 0、双盘参数；818 passed、0 failed，包含真实 DMA 与读写校验 |
| StorageRuntime local-NVMe | PASS | GPU 0、NVMe 1；156 passed、0 failed，SINGLE/DUAL/LIST 与 batch byte-exact |
| Striped local-NVMe correctness | PASS | 两次均 50 个正确性检查通过，包含跨 shard、分布、partial commit、restart persistence |
| Striped performance assertion | BASELINE FAIL | 两次 speedup 均为 1.16x，低于既有 1.3x 阈值；正确性未失败 |

第一次 HOST 并行 `ctest` 中，`tutti_storage_runtime_contract_test` 曾在 60 秒超时；
该二进制直接运行、单独 `ctest` 和随后完整 16-test 复跑均在 0.01 秒通过，因此记录
为一次不可复现的基线调度异常。

第一次以非 root 用户运行 LocalNvmeDataPath hardware test 时，scratch directory
创建被 `/mnt/snvme/nvme1/GPU0` 权限拒绝。按现有硬件测试的 root 前提使用 sudo
重跑后 818/818 通过；失败发生在数据路径初始化之前。

## 阶段结论

阶段 0 exit gate 通过：本阶段未改变业务行为，HOST/CUDA 无硬件 contract 全绿，
双盘 daemon 可列出两个 NVMe resource，RPC BDF 与配置一致。Striped 性能阈值是
已记录的基线缺口，不属于阶段 0 exit gate，也不能据此宣称 `validated_available`；
后续阶段仍以真实数据正确性测试作为硬件闭环条件。

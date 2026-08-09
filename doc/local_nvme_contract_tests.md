# Tutti local-NVMe hardware contract tests

本文记录三个 CUDA userspace local-NVMe contract test 的运行方法。运行测试前，
必须先启动 `tutti_daemon`，并使用与 daemon 配置一致的设备参数。

## 1. 运行前提

### 1.1 构建测试

硬件测试由 `TUTTI_BUILD_HARDWARE_TESTS` 控制。首次配置或切换配置后执行：

```bash
cmake --preset cuda --fresh -DTUTTI_BUILD_HARDWARE_TESTS=ON
cmake --build --preset cuda --target \
  tutti_local_nvme_datapath_contract_test \
  tutti_storage_runtime_local_nvme_contract_test \
  tutti_striped_local_nvme_contract_test --parallel 8
```

二进制位于 `build/cuda/bin/`。

### 1.2 启动 daemon

```bash
sudo env TUTTI_VERBOSE=1 \
  build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon \
  --config config/local/daemon_2_disk.yaml
```

### 1.3 将 daemon 配置转换为 `--nvme`

测试不会读取 YAML；通过 `--nvme` 显式传入设备表，格式为：

```text
ssnvme_path,pci_bdf,backing_device,mount_path[,block_size[,bar0_size[,namespace_id]]]
```

字段依次为 daemon 报告的 `snvme`、`pci_addr`、namespace 块设备、真实
`mount_path`、`blk`、`bar0` 和 YAML 的 `namespace_id`。GPU view（例如
`/mnt/gpu0/ssnvme0`）不是 `mount_path`。配置数组顺序决定 `device_id`，
`allowed_gpus` 必须包含测试命令的 `--gpu`。

当前 `config/local/daemon_2_disk.yaml`（两个 namespace 均为 4 KiB）对应：

```bash
NVME0='/dev/ssnvme0,0000:b1:00.0,/dev/snvme0n1,/mnt/nvme0,4096,16384,1'
NVME1='/dev/ssnvme1,0000:e3:00.0,/dev/snvme1n1,/mnt/nvme1,4096,16384,1'
```

如果 daemon 使用其他 YAML，必须同步修改这些字段。`block_size` 是 namespace
LBA 大小；Striped 测试选择的 namespace 必须一致。

## 2. Test

### 2.1 LocalNvmeDataPath contract

覆盖 LocalNvmeDataPath 生命周期、内存/DMA、SINGLE/DUAL/LIST I/O、错误处理、并发、
缓存和多设备场景：

```bash
build/cuda/bin/tutti_local_nvme_datapath_contract_test \
  --gpu 0 \
  --device-index 0 \
  --nvme "$NVME0" \
  --nvme "$NVME1"
```

重复 `--nvme` 会替换内置设备表（最多四项）；`--device-index` 选择 primary entry。

#### Queue pair 要求

测试中的生产 `LocalNvmeDataPath` 固定请求 `kNumQueues=16`，因此每个使用的
controller 需要 `user_io_qps >= 16` 且 `max_q_per_grp >= 16`。

### 2.2 StorageRuntime local-NVMe contract

该测试只接受一个 `--nvme`，通过 public `StorageRuntime` API 验证 open、内存、
SINGLE/DUAL/LIST、batch、并发、超时和 capacity 行为。

```bash
build/cuda/bin/tutti_storage_runtime_local_nvme_contract_test \
  --gpu 0 \
  --nvme "$NVME1"
```

每个 runtime 申请 `16` 个 QP；`assembly/open` 会同时创建第二个 `dp2` 验证
mismatched key。因此同一 controller 需要 `max_q_per_grp >= 16` 且至少有
`32` 个 user QP。

请求 payload 固定为 4 KiB；`block_size` 只负责 LBA 换算。

### 2.3 Striped local-NVMe contract

该测试验证 striped resolver/DataPath 的跨盘读写、分片、重启持久性和 mixed batch：

```bash
build/cuda/bin/tutti_striped_local_nvme_contract_test \
  --devices 2 \
  --gpu 0 \
  --nvme "$NVME0" \
  --nvme "$NVME1"
```

`--devices` 只接受 `2` 或 `4`；缺少设备、mount 或 CUDA runtime 时返回 skip（77）。
所有选中的 namespace 必须有相同 `block_size`，且 `allowed_gpus` 包含 `--gpu 0`；
每个 selected controller 也要有至少 16 个 user QP。

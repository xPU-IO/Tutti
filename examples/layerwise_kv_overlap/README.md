# examples/layerwise_kv_overlap

## 这是什么

`layerwise_kv_overlap` 是 Tutti 的标准 KV-cache 参考负载示例。它模拟 HY3-shaped 128K-context 请求（80 层，512 × 256-token chunks，90% prefix hit），采用 3-stream layerwise pipeline：

```
read(L+1) ∥ SGEMM compute(L) ∥ write(L-1)
```

每个 chunk 的 K/V tensor（512 KiB）独立注册到 DataPath，NVMe DMA 直接读写 GPU tensor——无 scratch buffer、无 D2D bounce。

## 前置环境

**严格顺序**：内核模块 → `tutti_daemon` → 挂载——块设备由 daemon bring-up 后才创建，先 mount 会找不到设备。

### 1. 编译内核模块

```bash
cd /path/to/Tutti
cmake --preset cuda-module --fresh \
    -DSNVME_KERNEL_VERSION=5.15.0-public \
    -DTUTTI_BUILD_HARDWARE_TESTS=ON
cmake --build --preset cuda-module --target modules --parallel 8
```

产物：`build/cuda-module/module/snvme.ko`、`build/cuda-module/module/snvme-core.ko`

### 2. 加载内核模块

```bash
sudo insmod build/cuda-module/module/snvme-core.ko
sudo insmod build/cuda-module/module/snvme.ko io_queue_depth=1024
```

> **注意**：必须先加载 `snvme-core.ko`（依赖模块），再加载 `snvme.ko`。`io_queue_depth=1024` 是必需的——默认 64 会导致 striped 模式大规模 "wait failed"（见 [FAQ](../../doc/build_and_test.md#faq)）。

### 3. 编译并启动 tutti_daemon

```bash
cmake --build --preset cuda-module --target tutti_daemon --parallel 8
sudo ./build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon \
    --config sys_config.yaml &
```

daemon 启动后会：
- bind 到 NVMe 控制器（`/dev/ssnvme0-3`）
- 创建块设备 `/dev/snvme0n1` ~ `/dev/snvme3n1`
- 按 `sys_config.yaml` 的 `auto_mount: true` 自动挂载到 `/mnt/nvme0-3`（ext4）

### 4. 确认挂载

```bash
lsblk | grep snvme
# 应看到 4 个 snvme 块设备挂载到 /mnt/nvme0-3
```

如果 daemon 没有自动挂载（`auto_mount: false` 或失败），手动挂载：

```bash
sudo mount /dev/snvme0n1 /mnt/nvme0
sudo mount /dev/snvme1n1 /mnt/nvme1
sudo mount /dev/snvme2n1 /mnt/nvme2
sudo mount /dev/snvme3n1 /mnt/nvme3
```

### 5. 关闭

```bash
sudo killall tutti_daemon     # 或 kill <pid>
sudo umount /mnt/nvme0-3
sudo rmmod snvme snvme-core
```

详见 [`doc/build_and_test.md`](../../doc/build_and_test.md)。

## 编译

```bash
cmake --preset cuda --fresh -DTUTTI_BUILD_HARDWARE_TESTS=ON
cmake --build --preset cuda --target tutti_layerwise_kv_overlap --parallel 8
```

产物：`build/cuda/bin/tutti_layerwise_kv_overlap`

## 运行

### 默认模式（striped）

```bash
sudo ./build/cuda/bin/tutti_layerwise_kv_overlap --striped
```

不传 `--striped` 时同样默认使用 striped 模式和源码内置的 4 盘配置。
通过可重复的 `--nvme` 参数可以整体覆盖内置设备列表：

```bash
sudo ./build/cuda/bin/tutti_layerwise_kv_overlap --striped \
  --nvme /dev/ssnvme0,0000:b1:00.0,/dev/snvme0n1,/mnt/nvme0 \
  --nvme /dev/ssnvme1,0000:e3:00.0,/dev/snvme1n1,/mnt/nvme1
```

每个 `--nvme` 值依次包含 `ssnvme_path,pci_bdf,backing_device,mount_path`。
第一次出现 `--nvme` 时会清空内置列表，后续参数继续追加设备。striped 模式
要求最终设备数是不小于 2 的幂，例如 2、4 或 8。
程序会根据 chunk 和 shard 数量自动提升进程的 `RLIMIT_NOFILE` 软上限；不会修改
系统全局配置。如果硬上限不足，程序会在创建文件前报告所需的最小值并退出。

### 单盘模式

```bash
sudo ./build/cuda/bin/tutti_layerwise_kv_overlap --single
```

single 模式只使用设备列表中的第一项。覆盖设备时，应同时通过 `--data-dir`
指定该设备文件系统上的数据目录。

### 可选参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--layers` | 80 | 层数 |
| `--ctx-tokens` | 131072 | 上下文 token 数 |
| `--chunk-tokens` | 256 | 每层 chunk 的 token 数 |
| `--hit-pct` | 90 | prefix hit 百分比 |
| `--tensor-kb` | 512 | 每 K/V tensor 大小（KiB）；同时作为 stripe unit，参数值必须是 4 的倍数 |
| `--compute-us` | 0 | 每次 compute 模拟延迟（μs） |
| `--gemm-n` | 1024 | SGEMM 矩阵维度 |
| `--data-dir` | /mnt/nvme0/GPU0 | 单盘模式数据目录 |
| `--nvme` | 内置 4 盘配置 | 覆盖 NVMe 设备；格式为 `ssnvme_path,pci_bdf,backing_device,mount_path`，可重复 |
| `--striped` | 默认启用 | 使用 striped 模式；设备数必须是不小于 2 的幂 |
| `--single` | | 使用单盘模式 |
| `--no-verify` | | 跳过逐字节校验 |

## 预期输出

以下以双盘 `--nvme` 覆盖命令为例：

```
[ OK ] cudaSetDevice(0)
[ OK ] StorageRuntime created (StripedDataPath, N=2)
[ OK ] Phase A (striped): 512 targets x 2 shards (X.X GB) in X.XXs
[ OK ] Phase B: 460 hit + 52 miss chunks, 1024 tensors registered
[ OK ] Phase C: opened 512 targets (striped)
[INFO] Layer  0: read   X.XX ms (XXXX MB/s)  write  X.XX ms (XXXX MB/s)
[INFO] Layer  1: read   X.XX ms (XXXX MB/s)  write  X.XX ms (XXXX MB/s)
...
[INFO] Phase G: READ  XX.XX ms total, XXXX.X GB/s avg
[INFO] Phase G: WRITE XX.XX ms total, XXXX.X GB/s avg
[ OK ] Phase H: 26/26 tensors verified
```

**参考带宽**：4-disk striped 模式 ~25 GB/s（READ）。

## 作为 ctest 运行

```bash
ctest -R tutti_layerwise_kv_overlap
```

Labels: `hardware;local_nvme;layerwise_overlap`

## 公共 API

此示例只使用公共 API：
- `<tutti/storage_runtime.h>` — StorageRuntime
- `<tutti/io_types.h>` — IoRequest, IoDirection 等
- `<tutti/memory_types.h>` — MemoryView, MemoryHandle 等
- `<tutti/presets/local_nvme.h>` — preset 工厂函数（make_local_nvme_runtime / make_striped_nvme_runtime）
- `<tutti/cuda_like.h>` — CUDA 运行时抽象

不引用任何私有 DataPath 或 resolver 头文件。

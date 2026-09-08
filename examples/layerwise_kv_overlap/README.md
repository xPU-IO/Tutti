# examples/layerwise_kv_overlap

## 这是什么

`layerwise_kv_overlap` 是 Tutti 的标准 KV-cache 参考负载。它模拟 HY3-shaped
128K-context 请求（80 层，512 × 256-token chunks，90% prefix hit），采用
3-stream layerwise pipeline：

```
read(L+1) ∥ SGEMM compute(L) ∥ write(L-1)
```

每个 chunk 的 K/V tensor（512 KiB）独立注册到 DataPath，NVMe DMA 直接读写
GPU tensor——无 scratch buffer、无 D2D bounce。

## TuttiRuntime 模式

本示例基于 **TuttiRuntime**：runtime（设备、队列、数据路径）由 Tutti YAML
（`--config`）组装，部署事实（PCI BDF、chrdev、挂载点、队列配额）全部来自
`tutti_daemon`。用户只传 daemon 发布的 **view 目录**（`--directory`），不再
需要 `--nvme ssnvme_path,pci_bdf,...` 之类的设备细节。

随示例附带两个 YAML（字段含义见文件内注释）：

| YAML | 模式 | 设备 | 队列/盘 |
| --- | --- | --- | --- |
| `tutti_layerwise_striped.yaml`（默认） | 4 盘 striped | daemon device 0-3 | 32 |
| `tutti_layerwise_local.yaml` | 单盘 | daemon device 0 | 16 |

## 前置环境

**严格顺序**：snvme 内核模块（已签名加载）→ `tutti_daemon` 运行中 → 记下
daemon 发布的 view 目录。设 `TUTTI_BUILD` 为按
[`doc/getting-started.md`](../../doc/getting-started.md) 配置的 CUDA module build
目录后，可查询实际发布路径：

```bash
lsmod | grep snvme                                    # snvme + snvme_core
"$TUTTI_BUILD/bin/nvmeservice_client" \
  --endpoint 127.0.0.1:50051 --list-only              # 查 view_root 与 chrdev
# 不要从 device_id 或 /dev 编号猜测 view 路径
```

## 编译

本目标依赖 `TUTTI_BUILD_HARDWARE_TESTS=ON`。完整的 vcpkg 预检、CMake configure 和
module 编译步骤见 [`doc/getting-started.md`](../../doc/getting-started.md) 第 6、7 节；
完成 configure 后执行：

```bash
cmake --build "$TUTTI_BUILD" --target tutti_layerwise_kv_overlap --parallel "$JOBS"
```

产物：`$TUTTI_BUILD/bin/tutti_layerwise_kv_overlap`

## 运行

### striped（4 盘，默认）

`--directory` 顺序须与 YAML 的 `device_ids` 一致：

```bash
sudo "$TUTTI_BUILD/bin/tutti_layerwise_kv_overlap" --striped \
  --directory /mnt/gpu0/ssnvme0 \
  --directory /mnt/gpu0/ssnvme1 \
  --directory /mnt/gpu0/ssnvme2 \
  --directory /mnt/gpu0/ssnvme3
```

### 单盘

```bash
sudo "$TUTTI_BUILD/bin/tutti_layerwise_kv_overlap" --single \
  --config examples/layerwise_kv_overlap/tutti_layerwise_local.yaml \
  --directory /mnt/gpu0/ssnvme0
```

### 可选参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--directory` | **必填** | daemon 发布的 view 目录，可重复；striped 要求 2 的幂个 |
| `--config` | `tutti_layerwise_striped.yaml` | Tutti YAML 路径 |
| `--layers` | 80 | 层数 |
| `--ctx-tokens` | 131072 | 上下文 token 数 |
| `--chunk-tokens` | 256 | 每层 chunk 的 token 数 |
| `--hit-pct` | 90 | prefix hit 百分比 |
| `--tensor-kb` | 512 | 每 K/V tensor 大小（KiB）；须与 YAML `stripe_unit` 一致 |
| `--compute-us` | 0（自动校准） | 每次 compute 模拟延迟 |
| `--gemm-n` | 1024 | SGEMM 矩阵维度 |
| `--striped` / `--single` | striped 默认 | 多盘 / 单盘 |
| `--no-verify` | — | 跳过逐字节校验 |

程序会按需自动提升进程 `RLIMIT_NOFILE` 软上限，不修改系统全局配置。

## 预期输出（4 盘 striped 实测）

```
[ OK ] cudaSetDevice(0) via TuttiRuntime (examples/layerwise_kv_overlap/tutti_layerwise_striped.yaml)
[ OK ] StorageRuntime created (StripedDataPath, N=4)
[ OK ] Phase A (striped): 512 targets x 4 shards (40.0 GB) in X.XXs
[ OK ] Phase B: 460 hit + 52 miss chunks, 1024 tensors registered
[ OK ] Phase C: opened 512 targets (striped)
[ OK ] Phase E: pre-wrote 460 chunks x 80 layers (35.94 GB) in X.XXs
[INFO] rq1 L9   read 482.3MB/21.2ms=22.8GB/s write 54.5MB/38.4ms=1.4GB/s
...
[ OK ] SIM TOTAL: 2 req wall=9.4s | READ 77.18GB=23.0GB/s | WRITE 8.72GB=1.4GB/s | overlap 40%
[ OK ] Phase H: verified 26 samples, all correct

=== layerwise_kv_overlap: PASSED ===
```

**参考带宽**（8×H20 + 4×PM9A1 实测）：striped 读 ~23 GB/s（文档口径 ~25 GB/s）、
单盘读 ~6.4 GB/s。

## 调整设备/队列

改 YAML 即可，不用重编译（`--config` 指向改后的文件）。关键约束：

- `device_ids`：daemon YAML `nvmes[].device_id`；striped 个数须为 2 的幂
- `queues_per_controller` ≤ daemon `queue_pool.max_per_client`
- `threads_per_block` 可大于实际获批的队列数（会打告警并按取模共享队列，
  并行队列本身支持多线程同队列提交；不会报错）
- `stripe_unit` = 工作负载 tensor 大小（`--tensor-kb` × 1024）

## 手动硬件 workload

该程序需要 daemon 实际发布的一个或多个 `--directory`，因此不是无参数 CTest。
按上面的命令在完成模块、daemon 和挂载 bring-up 后手动运行；普通回归仍使用
`ctest --test-dir "$TUTTI_BUILD" -LE hardware`。

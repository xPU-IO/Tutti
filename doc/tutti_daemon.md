# tutti_daemon 启动与部署指南

本文说明在 CUDA 主机上配置和启动已经编译好的 `tutti_daemon`。编译统一遵循
[`getting-started.md`](getting-started.md) 的唯一 `default` preset。示例使用一个 GPU
和一个 NVMe namespace，重点说明以下对象之间的关系：

- 物理 NVMe 的 PCI 地址，例如 `0000:31:00.0`；
- daemon bring-up 后生成的 SNVMe 字符设备和块设备；
- 挂载真实 ext4 文件系统的 NVMe 目录；
- 暴露给指定 GPU 的目录和软链接。

文中的 `0000:31:00.0`、`/dev/nvme0n1`、GPU 0 和挂载路径都只是示例。
必须先在目标主机上确认设备身份，不能直接照抄。

> **数据安全警告**
>
> `mkfs.ext4` 会破坏目标 namespace 上的现有文件系统和数据。执行格式化前，
> 必须同时核对 PCI BDF、型号、序列号、容量、系统盘关系和当前挂载状态。
> 如果不能确定设备是允许清空的数据盘，请停止，不要执行格式化命令。

## 1. 启动流程概览

加载 snvme kernel module 是启动 `tutti_daemon` 的**前置条件**，不属于 daemon
自身的启动流程。运行 daemon 前应已经满足：

- `snvme-core.ko` 和 `snvme.ko` 已加载，`/dev/snvm_control` 存在；
- 目标 NVMe namespace 已准备为 ext4；
- 本机 YAML 中已经填写正确的 PCI BDF、GPU 和挂载路径。

满足这些前置条件后，`tutti_daemon` 自身的启动顺序如下：

```text
读取并校验 YAML
        │
        ▼
ServiceState bring-up
  chrdev_create → kernel IOQ cap → bind → probe
        │
        ├── /dev/ssnvme<N>       字符设备，供 libnvm client 使用
        └── /dev/snvme<N>n<NSID> 块设备，供 ext4 mount 使用
        │
        ▼
检查每个 namespace 报告的 logical block size
  多盘必须统一；不统一时在 mount/gRPC 前拒绝启动
  非 4 KiB 的统一值输出 WARNING（当前 striped 假设仍为 4 KiB）
        │
        ▼
MountManager 创建 `backing_mount_path` 并挂载 ext4
        │
        ▼
确认文件系统已经挂载
        │
        ▼
在真实 NVMe 文件系统内创建 `ACCEL<accel_id>` 目录
并在对应 `view_root` 下发布软链接
        │
        ▼
启动 reaper 和 gRPC server
```

## 2. 环境准备

建议从项目根目录执行本文命令：

```bash
cd /path/to/Tutti
```

需要以下基础环境：

- 已按 [`getting-started.md`](getting-started.md) 成功编译的默认硬件 build；
- 与目标 GPU 驱动匹配的 CUDA toolkit；
- 当前内核对应的 headers/devel 包；
- `lspci`、`lsblk`、`findmnt`、`blkid`；
- 可选的 `nvme-cli`，用于查看 NVMe 型号、序列号和 namespace；
- `e2fsprogs`，仅在需要执行 `mkfs.ext4` 时使用。

默认 preset 工作流要求 CMake 3.21 以上。

## 3. 查找目标 NVMe 及其 PCI 地址

使用项目提供的 PCI 拓扑脚本发现 GPU、NVMe 和 PCI BDF：

```bash
sudo bash scripts/pci_topology_check.sh
```

脚本会直接打印：

- GPU index 和 GPU BDF；
- 每块 NVMe 的完整 BDF；
- NVMe 当前由标准 `nvme` 驱动管理时对应的 `/dev/nvme...` namespace；
- 每个 GPU/NVMe 组合的拓扑距离。

示例：

```text
Found GPU 0: 0000:4b:00.0
Found NVMe: 0000:31:00.0 -> N/A

NVMe \ GPU                      GPU0
N/A (31:00.0)                   1
```

距离含义为：`0` 表示相同 PCIe switch/root complex，`1` 表示相同 NUMA node，
`2` 表示跨 NUMA。通常优先选择距离较小的 NVMe，并据此填写 `allowed_accel_ids`。

结果同时保存在：

```text
/mnt/sys_GPU_NVMe_topology.json
```

JSON 中的 `nvme_bdf` 用于填写 `nvmes[].pci_addr`，`gpu_index` 用于填写
`accelerators[].accel_id` 和 `allowed_accel_ids`。如果 `nvme_device` 为 `N/A`，表示
该设备当前没有绑定到标准 `nvme` 驱动；BDF 和拓扑结果仍然有效，下一章使用项目
bind 脚本后即可获得 `/dev/nvme...` 设备。

配置文件中的 PCI 地址必须使用带 domain 的完整形式 `DDDD:BB:DD.F`，例如
`0000:31:00.0`。

## 4. 检查或创建 ext4 文件系统

检查或重建文件系统前，应先停止 `tutti_daemon` 并确认目标盘没有挂载。使用
上一章得到的 PCI BDF，将目标设备绑定到标准 Linux `nvme` 驱动：

```bash
sudo bash scripts/bind_nvme_device.sh 0000:31:00.0
```

脚本会校验 PCI 设备类型、切换驱动并输出注册后的 NVMe controller。然后找出
它对应的 namespace，例如 `/dev/nvme0n1`，检查文件系统类型：

```bash
lsblk -f
sudo blkid /dev/nvme0n1
```

如果 `FSTYPE`/`TYPE` 为 `ext4`，通过一次临时 mount 验证：

```bash
sudo mkdir -p /mnt/tutti-ext4-check
sudo mount -t ext4 /dev/nvme0n1 /mnt/tutti-ext4-check
findmnt /mnt/tutti-ext4-check
sudo umount /mnt/tutti-ext4-check
```

mount 成功且 `findmnt` 显示 `ext4`，说明该 namespace 可以交给 daemon 使用。
当前 daemon 挂载整个 namespace，因此文件系统应建立在 `/dev/nvme0n1`，而不是
`/dev/nvme0n1p1` 分区。

### 可选：创建新的 ext4 文件系统

如果检查结果不是 ext4，并且确认该 namespace 允许被清空，可以使用 `mkfs`
重建 ext4 文件系统：

```bash
sudo mkfs.ext4 -F -L tutti-nvme0 /dev/nvme0n1
```

> **警告：**`mkfs.ext4` 会重建文件系统并清除原有数据。必须确认设备名来自
> `bind_nvme_device.sh` 所绑定的目标 BDF，并确认它不是系统盘或业务盘。

完成后重复上一节的 `blkid` 和临时 mount 测试；卸载测试目录后再启动 daemon。

## 5. 已构建模块与 daemon 前置条件

按 [`getting-started.md`](getting-started.md) 完成默认构建后，产物固定在 `build/`：

```bash
cd /path/to/Tutti
ls -lh build/module/snvme-core.ko build/module/snvme.ko
test -x build/bin/tutti_daemon
```

启动 daemon 前，模块必须已经按站点签名和加载流程安装，且
`/dev/snvm_control` 已存在。`insmod`、`rmmod`、NVMe 接管和 reload 会改变运行中
内核状态，因此不属于普通构建命令；需要修改 driver 时，阅读
[`advanced-build.md`](advanced-build.md)。

## 6. 创建本机 YAML 配置

本机配置应放在 `config/local/` 下。该目录中的 `*.yaml` 和 `*.yml` 已被
Git 忽略，适合保存不同机器的 PCI、GPU 和挂载路径配置。

可以从仓库模板复制：

```bash
mkdir -p config/local
cp config/local_nvme_config.yaml config/local/tutti_daemon.yaml
```

然后删除不属于本机的设备项，并只编辑模板已有的 canonical 字段。不要新建历史
`gpus`、`mount_path` 或 `allowed_gpus` 字段，也不要从数组顺序推导设备编号。

| 字段 | 含义 |
| --- | --- |
| `accelerators[].accel_id` | 编译后端的 accelerator ordinal，例如 CUDA device index |
| `accelerators[].view_root` | 对应 accelerator 的 view 根目录，不是真实磁盘挂载点 |
| `nvmes[].device_id` | 显式、唯一的 daemon NVMe 资源 ID |
| `nvmes[].pci_addr` | 目标 NVMe 的完整 PCI BDF，也是最重要的设备身份 |
| `nvmes[].backing_mount_path` | daemon 挂载 owner 返回 block device 的 ext4 目录 |
| `namespace_id` | NVMe namespace ID，通常为 1 |
| `kernel_ioq_cap` | bind 前的内核 IO queue cap 提示；0 表示交给内核 |
| `allowed_accel_ids` | 可获取该 NVMe 资源并获得 view 的 accelerator ID ACL |
| `auto_mount` | `true` 表示 daemon 挂载并在退出时卸载 |
| `queue_pool` / `lease` / `unmount_retry` | 每 client 队列、租约和卸载重试策略 |

`device_id` 和 `accel_id` 是显式身份，数组顺序不参与 `/dev`、block device 或 view
路径推导。daemon 通过 owner bring-up 返回实际 `chrdev_path` 和 `block_path`；客户端
必须使用 RPC 返回的 view 路径，不能拼接 `/dev/ssnvmeN` 或 `/mnt/gpuN/ssnvmeN`。

## 7. 使用指定 YAML 启动 daemon

建议启用详细日志，以便观察 bring-up、mount 和 shutdown：

```bash
sudo env TUTTI_VERBOSE=1 \
  build/bin/tutti_daemon \
  --config config/local/tutti_daemon.yaml
```

启动过程需要 root 权限，因为 daemon 要执行 controller bind、创建字符设备、
挂载文件系统和创建系统目录。

### 7.1 启动成功标志

详细日志中的关键成功信息类似：

```text
nvmeservice: device=0 pci=0000:31:00.0 snvme=/dev/ssnvme0 ns=1 ...
mount_manager: mounted /dev/snvme0n1 at /mnt/nvme0 (owned)
tutti_daemon listening on 127.0.0.1:50051 (port 50051)
Owned devices:
  device_id=0 pci=0000:31:00.0 snvme=/dev/ssnvme0 ns=1 block_size=4096 io_qp_limit=64 kernel_io_qps=32 user_io_qps=32 max_user_qid=64 max_q_per_grp=32
```

`tutti_daemon listening` 只说明 gRPC 已启动。完整成功还必须确认 mount 和 GPU view，
因为 mount 失败时 daemon 会打印 warning 并继续启动 gRPC。启动日志中不应出现：

```text
warning: auto-mount ... failed
warning: ... is not mounted; GPU views ... will not be published
```

### 7.2 队列日志字段

以下字段均以 I/O queue pair 为单位，不包含 QID 0 的 admin queue：

| 字段 | 含义 |
| --- | --- |
| `io_qp_limit` | 控制器协商得到的 I/O QP 总上限 |
| `kernel_io_qps` | kernel 实际占用的 I/O QP 数量 |
| `user_io_qps` | 启动时划给 user QID pool 的容量，不是实时剩余量 |
| `max_user_qid` | `NVM_ADD_USER_QUEUE` 可分配的最大 QID 编号（包含），不是 QP 数量 |
| `max_q_per_grp` | 单个 client fd/queue group 最多可创建的 user QP 数量 |

总量关系为 `kernel_io_qps + user_io_qps = io_qp_limit`。单个 client 的
实际上限还会受到 `queue_pool.max_per_client` 和 `max_q_per_grp` 的共同限制。

<details>
<summary>QID 编号关系示例</summary>

当 `start_cq_idx=33`、`max_user_qid=64` 时，QID 1～32 属于 kernel，
QID 33～64 属于 user：

```text
kernel_io_qps = start_cq_idx - 1 = 32
user_io_qps   = max_user_qid - start_cq_idx + 1 = 32
io_qp_limit   = max_user_qid = 64
```

`io_qp_limit` 与 `max_user_qid` 数值相同，是因为 I/O QID 从 1 连续编号；
前者表示 QP 总数量，后者表示最大 QID 编号。

</details>

### 7.3 Namespace block size

日志中的 `block_size`/`blk_size` 是 namespace logical block size，即
`1 << lba_shift` bytes；`blk_size_log` 是对应的 LBA shift。它不是 ext4 block
size、physical block size 或 controller-wide 属性，并且无需在 YAML 中重复配置。

- 多盘值不一致（例如同时出现 `4096` 和 `512`）时，daemon 会在挂载前退出；
- 所有盘一致但不是 4 KiB 时，daemon 会输出 `WARNING` 后继续启动，但不满足当前
  striped workload 的 4 KiB 假设。

### 7.4 启动后检查

在另一个终端先从 daemon 获取**实际**资源和 view 路径：

```bash
ls -l /dev/snvm_control
build/bin/nvmeservice_client \
  --endpoint 127.0.0.1:50051 --list-only
ss -ltn | grep ':50051'
```

对每一个返回的 resource，记录 `<block-path>`、`<backing-mount-path>` 和
`<view-path>`，再检查：

```bash
findmnt -T "<backing-mount-path>/ACCEL<accel-id>"
readlink -f "<view-path>"
```

两者必须解析到同一个 `ACCEL<accel-id>` 目录。不要从 `device_id`、YAML 数组顺序或
`/dev` ordinal 猜测这些路径。

## 8. 启动后生成的设备和目录

假设 device `D` 的 ACL 包含 accelerator `A`，daemon 通过 owner bring-up 返回字符
设备路径和块设备路径，并依照 canonical YAML 创建以下对象：

| 对象 | 类型 | 是否承载真实 NVMe 数据 | 作用 |
| --- | --- | ---: | --- |
| `/dev/snvm_control` | module control 字符设备 | 否 | daemon 执行 create/bind 等 owner 操作 |
| `<chrdev-path>` | per-controller 字符设备 | 否 | client/libnvm attach、队列和映射操作 |
| `<block-path>` | namespace 块设备 | 是 | ext4 的 mount source |
| `<backing-mount-path>` | ext4 mount root | 是 | `nvmes[].backing_mount_path`，真实磁盘根目录 |
| `<backing-mount-path>/ACCEL<A>` | ext4 内普通目录 | 是 | accelerator `A` 的真实数据目录 |
| `<view-root>` | accelerator view root | 否 | `accelerators[].view_root` |
| `<view-path>` | 软链接 | 间接指向真实数据 | RPC 返回的 accelerator view |

目录关系为：

```text
YAML nvmes[].device_id = D
  ├─ <chrdev-path>                   client 字符设备
  └─ <block-path>                    namespace 块设备
      └─ mount ext4 at <backing-mount-path>
          └─ ACCEL<A>                真实 NVMe 目录
              ▲
              └── <view-path>        <view-root> 下的软链接
```

如果 `allowed_accel_ids: [0, 2]`，daemon 会在同一个真实 NVMe 文件系统内创建
`ACCEL0` 和 `ACCEL2`，并分别从 accelerator 0、accelerator 2 的 view root 发布软链接。

daemon 只创建 `ACCEL<N>` 目录，不会自动创建 `resolver_test` 等测试目录。应用必须
使用 RPC 返回的 `<view-path>`；数据最终位于对应 backing mount 的 `ACCEL<N>` 目录。

## 9. 优雅停止

在前台按一次 `Ctrl-C`，或从另一个终端发送一次 `SIGTERM`：

```bash
sudo kill -TERM <tutti_daemon_pid>
```

正常关闭顺序为：

```text
停止 gRPC
  → 停止 reaper
  → 删除 accelerator view 软链接
  → 删除空的 ACCEL<N> 目录
  → 卸载 daemon 自己挂载的 ext4
  → 释放 controller，移除 per-controller 设备节点
```

如果 `ACCEL<N>` 内仍有业务文件，daemon 只会尝试 `rmdir`，不会递归删除数据；
目录会保留在 NVMe 文件系统内。退出后使用启动时 RPC 返回的路径检查：

```bash
findmnt "<backing-mount-path>"
test ! -L "<view-path>"
ls -l /dev/snvm_control
```

当卸载因 holder 返回 `EBUSY` 时，daemon 会报告相关 PID、fd、maps 或 cwd，并按
`unmount_retry` 重试。第二次发送信号会强制结束重试并留下挂载；除非处于明确的
应急恢复流程，否则不要发送第二次信号，更不要使用 `kill -9`。

## 10. 资源分配补充

第 6 节和 `config/local_nvme_config.yaml` 是唯一的 daemon YAML schema。`device_id`
和 `accel_id` 都是显式、唯一的身份；数组顺序不参与 `/dev`、block device 或 view
路径推导。daemon 通过 libnvm owner bring-up 返回实际 `chrdev_path`/minor 和
`block_path`，并在 BDF、设备节点、mount/view 校验失败时将资源保持为不可用。

控制面客户端应先调用 `ListAccelerators` 和 `ListNvmeResources`，再使用
`AcquireNvmeSlices`。请求支持 allowed、explicit 和按请求顺序的 striped selection；
一个 striped 请求返回一个 `allocation_id` 和全部 slices。所有 slices 的 queue
预算在一个临界区内原子预留，`Release`、旧 `Disconnect`、heartbeat timeout 以及
PID/starttime reaper 都经过同一回收路径。

新 client 入口使用 `--accel`；`--cuda` 仅保留给旧 `Connect` 兼容路径，不能绕过 ACL
或 queue ledger。daemon 的 list/acquire 路径不调用 accelerator runtime，也不创建
compute context。最终 hardware gate 必须使用 owner/RPC 返回的实际路径完成 scratch
区域 write/read/verify；`--skip-io` 只能用于 attach 诊断，不能证明
`validated_available`。

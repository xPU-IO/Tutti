# tutti_daemon 启动与部署指南

本文说明如何在 CUDA 主机上编译、配置并启动 `tutti_daemon`。示例使用
`cuda-module` preset、一个 GPU 和一个 NVMe namespace，重点说明以下对象之间
的关系：

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
MountManager 创建 mount_path 并挂载 ext4
        │
        ▼
确认文件系统已经挂载
        │
        ▼
在真实 NVMe 文件系统内创建 GPU<N> 目录
并在 GPU view root 下发布软链接
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

- CMake 3.21 或更新版本；
- 与目标 GPU 驱动匹配的 CUDA toolkit；
- 当前内核对应的 headers/devel 包；
- `lspci`、`lsblk`、`findmnt`、`blkid`；
- 可选的 `nvme-cli`，用于查看 NVMe 型号、序列号和 namespace；
- `e2fsprogs`，仅在需要执行 `mkfs.ext4` 时使用。

依赖和 preset 的完整准备方式见 [build_and_test.md](build_and_test.md)。

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
`2` 表示跨 NUMA。通常优先选择距离较小的 NVMe，并据此填写 `allowed_gpus`。

结果同时保存在：

```text
/mnt/sys_GPU_NVMe_topology.json
```

JSON 中的 `nvme_bdf` 用于填写 `nvmes[].pci_addr`，`gpu_index` 用于填写
`gpus[].id` 和 `allowed_gpus`。如果 `nvme_device` 为 `N/A`，表示该设备当前没有
绑定到标准 `nvme` 驱动；BDF 和拓扑结果仍然有效，下一章使用项目 bind 脚本后
即可获得 `/dev/nvme...` 设备。

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

## 5. cuda-module 构建和 module 前置条件

使用 `cuda-module` preset 构建 snvme module 和生产 daemon：

```bash
cmake --preset cuda-module
cmake --build --preset cuda-module \
  --target modules tutti_daemon --parallel 8
```

在启动 daemon 前安装 module：

```bash
cmake --build --preset cuda-module --target insmod
```

`cuda-module` 的环境准备、内核 baseline、P2P backend、module 产物和 reload
流程见 [build_and_test.md](build_and_test.md)。本章只要求 daemon 启动前已经
完成 `insmod`，并存在 `/dev/snvm_control`。

## 6. 创建本机 YAML 配置

本机配置应放在 `config/local/` 下。该目录中的 `*.yaml` 和 `*.yml` 已被
Git 忽略，适合保存不同机器的 PCI、GPU 和挂载路径配置。

可以从仓库模板复制：

```bash
cp config/local_nvme_config.yaml config/local/tutti_daemon.yaml
```

然后删除不属于本机的设备项并修改配置。单 GPU、单 NVMe 示例：

```yaml
grpc:
  endpoint: "127.0.0.1:50051"

gpus:
  - id: 0
    mount_path: "/mnt/gpu0"

nvmes:
  - pci_addr: "0000:31:00.0"
    mount_path: "/mnt/nvme0"
    namespace_id: 1
    kernel_ioq_cap: 32
    allowed_gpus: [0]
    auto_mount: true

queue_pool:
  default_per_client: 32
  max_per_client: 32

lease:
  heartbeat_interval_sec: 10
  timeout_sec: 30

unmount_retry:
  interval_ms: 1000
  max: 30
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `grpc.endpoint` | daemon 的 gRPC 监听地址 |
| `gpus[].id` | CUDA device index |
| `gpus[].mount_path` | 该 GPU 的视图根目录，不是真实磁盘挂载点 |
| `nvmes[].pci_addr` | 目标 NVMe 的完整 PCI BDF，也是最重要的设备身份 |
| `nvmes[].mount_path` | 真实 NVMe ext4 文件系统的挂载点 |
| `namespace_id` | NVMe namespace ID，通常为 1 |
| `kernel_ioq_cap` | bind 前设置的内核 IO queue cap；0 表示使用内核默认值 |
| `allowed_gpus` | 可以连接并获得该 NVMe 视图的 GPU ID；省略或为空表示所有已配置 GPU |
| `auto_mount` | `true` 表示 daemon 挂载并在退出时卸载；通常应保持为 `true` |
| `unmount_retry` | 退出时遇到 `EBUSY` 的重试间隔和次数 |

`nvmes` 数组的顺序决定 daemon 的 `device_id`。例如第一个条目为
`device_id=0`；在 `namespace_id: 1` 时，daemon 会尝试挂载
`/dev/snvme0n1`。第二个条目对应 `device_id=1` 和 `/dev/snvme1n1`。

## 7. 使用指定 YAML 启动 daemon

建议启用详细日志，以便观察 bring-up、mount 和 shutdown：

```bash
sudo env TUTTI_VERBOSE=1 \
  ./build/cuda-module/tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon \
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
  device_id=0 pci=0000:31:00.0 snvme=/dev/ssnvme0 ns=1 ...
```

`tutti_daemon listening` 只说明 gRPC 已启动。完整成功还必须确认 mount 和 GPU view，
因为 mount 失败时 daemon 会打印 warning 并继续启动 gRPC。启动日志中不应出现：

```text
warning: auto-mount ... failed
warning: ... is not mounted; GPU views ... will not be published
```

在另一个终端执行：

```bash
ls -l /dev/snvm_control /dev/ssnvme0 /dev/snvme0n1
findmnt -no SOURCE,FSTYPE,TARGET /mnt/nvme0
test -d /mnt/nvme0/GPU0
readlink -f /mnt/gpu0/ssnvme0
ss -ltn | grep ':50051'
```

单盘示例的期望关系是：

```text
/dev/snvme0n1 ext4 /mnt/nvme0
/mnt/gpu0/ssnvme0 -> /mnt/nvme0/GPU0
```

也可以直接证明某个路径位于哪个文件系统：

```bash
findmnt -T /mnt/nvme0/GPU0
findmnt -T /mnt/gpu0/ssnvme0
```

## 8. 启动后生成的设备和目录

假设 YAML 中第一个 NVMe 允许 GPU 0 使用，启动后对象含义如下：

| 对象 | 类型 | 是否承载真实 NVMe 数据 | 作用 |
| --- | --- | ---: | --- |
| `/dev/snvm_control` | module control 字符设备 | 否 | daemon 执行 create/bind 等 owner 操作 |
| `/dev/ssnvme0` | per-controller 字符设备 | 否 | client/libnvm attach、队列和映射操作 |
| `/dev/snvme0n1` | namespace 块设备 | 是 | ext4 的 mount source |
| `/mnt/nvme0` | ext4 mount root | 是 | `nvmes[0].mount_path`，真实磁盘根目录 |
| `/mnt/nvme0/GPU0` | ext4 内的普通目录 | 是 | GPU 0 对应的真实数据目录 |
| `/mnt/gpu0` | GPU view root | 否 | `gpus[id=0].mount_path`，用于组织软链接 |
| `/mnt/gpu0/ssnvme0` | 软链接 | 间接指向真实数据 | 指向 `/mnt/nvme0/GPU0` |

目录关系为：

```text
PCI 0000:31:00.0
  └─ YAML nvmes[0] / daemon device_id=0
      ├─ /dev/ssnvme0                 client 字符设备
      └─ /dev/snvme0n1                namespace 块设备
          └─ mount ext4 at /mnt/nvme0 真实 NVMe 文件系统
              └─ GPU0                 真实 NVMe 目录
                  ▲
                  └── /mnt/gpu0/ssnvme0 软链接
```

如果 `allowed_gpus: [0, 2]`，daemon 会在同一个真实 NVMe 文件系统内创建
`GPU0` 和 `GPU2`，并分别从 GPU 0、GPU 2 的 view root 发布软链接。

daemon 只创建 `GPU<N>` 目录，不会自动创建 `resolver_test` 等测试目录。
应用写入 `/mnt/gpu0/ssnvme0/...` 时，软链接最终解析到
`/mnt/nvme0/GPU0/...`，数据实际位于该 NVMe 的 ext4 文件系统上。

## 9. 优雅停止

在前台按一次 `Ctrl-C`，或从另一个终端发送一次 `SIGTERM`：

```bash
sudo kill -TERM <tutti_daemon_pid>
```

正常关闭顺序为：

```text
停止 gRPC
  → 停止 reaper
  → 删除 GPU view 软链接
  → 删除空的 GPU<N> 目录
  → 卸载 daemon 自己挂载的 ext4
  → 释放 controller，移除 per-controller 设备节点
```

如果 `GPU<N>` 内仍有业务文件，daemon 只会尝试 `rmdir`，不会递归删除数据；
目录会保留在 NVMe 文件系统内。退出后可检查：

```bash
findmnt /mnt/nvme0
test ! -L /mnt/gpu0/ssnvme0
ls -l /dev/ssnvme0 /dev/snvme0n1
```

当卸载因 holder 返回 `EBUSY` 时，daemon 会报告相关 PID、fd、maps 或 cwd，并按
`unmount_retry` 重试。第二次发送信号会强制结束重试并留下挂载；除非处于明确的
应急恢复流程，否则不要发送第二次信号，更不要使用 `kill -9`。

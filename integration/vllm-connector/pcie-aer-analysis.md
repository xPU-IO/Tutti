# Tutti NVMe-GPU 直通场景 PCIe AER 分析

> 分析日期：2026-08-26
> 范围：本机 H20、PEX890xx、Samsung NVMe、snvme/Tutti local NVMe data path
> 方法：只读取 `dmesg`、PCI sysfs、`lspci`、`nvidia-smi` 和源码；未运行 I/O、GPU 压测或故障注入。

## 1. 结论

这组日志不是“NVMe 返回了一个可纠正错误”，而是 GPU 所在 PCIe link 报告了可纠正的物理层/数据链路层错误：

- GPU endpoint `49:00.0`：`RxErr + BadTLP`，接收端、Physical Layer。
- 部分同类事件的上游 PEX890xx downstream port：`Timeout`，发送端、Data Link Layer。
- `49:00.0` 的 Lane Error Status 已记录 lane 4、5、7、10、15。
- 当前只有 corrected counter 增长；fatal/non-fatal counter 均为 0，没有同时间窗口的 NVIDIA Xid、NVMe timeout/reset 或 I/O failure。

因此当前事件本身已被 PCIe 硬件恢复，不能据此认定发生数据丢失；但多张 GPU、多条 lane 重复出现同类错误，不应视为正常噪声。它通常意味着 Gen5 GPU link、retimer、switch downstream port、板级走线或相应 firmware preset 的链路裕量不足，高带宽或特定 P2P traffic pattern 只是触发条件。

软件不能通过修改 PRP/LBA/queue 算法修复 Physical Layer `RxErr`。软件能够做的是：

1. 精确定位触发条件并监控错误增量。
2. 降低并发 burst，减少触发概率。
3. 错误持续增长时停止 GPU-direct，切换到真正的 host-pinned bounce path。
4. 在运维层将受影响链路降到 PCIe Gen4，或更新 BIOS/BMC/PEX switch/retimer/GPU VBIOS/NVMe firmware。

其中 firmware 更新或 Gen4 downtrain 可能真正消除错误；Tutti 限流和 fallback 只能缓解或隔离，不能证明硬件已修复。

## 2. 本机证据

### 2.1 设备与链路

`49:00.0` 的实际设备是 NVIDIA H20：

| 项目 | 当前值 |
|---|---:|
| GPU | NVIDIA H20，`10de:2329` |
| GPU BDF | `0000:49:00.0` |
| GPU link | PCIe 5.0，32 GT/s，x16 |
| GPU link capability | Gen5 x16，支持 retimer |
| NVMe BDF | `0000:4b:00.0` |
| NVMe | Samsung `144d:a80a`，PM9A1/PM9A3/980PRO controller family |
| NVMe link | PCIe 4.0，16 GT/s，x4 |
| PCIe switch | Broadcom/LSI PEX890xx Gen5，`1000:c030` |
| NUMA node | 0 |

拓扑为：

```text
Intel root port 46:01.0 (Gen5 x16)
  -> PEX890xx upstream 47:00.0
     -> PEX890xx downstream 48:00.0
        -> H20 49:00.0 (Gen5 x16)
     -> sibling downstream port
        -> NVMe 4b:00.0 (Gen4 x4, snvme)
```

GPU 和 NVMe 位于同一个 PCIe switch 下，没有跨 CPU root complex。这是 GPUDirect/P2P 的优选路径，拓扑选择本身没有明显错误。

### 2.2 AER 和 lane 计数

本机 PCI sysfs 当前累计值：

| GPU | `RxErr` | `BadTLP` | Lane Error Status |
|---|---:|---:|---|
| `06:00.0` | 3 | 3 | lane 9、10、14 |
| `49:00.0` | 5 | 5 | lane 4、5、7、10、15 |
| `56:00.0` | 3 | 3 | lane 2、5、11 |
| `62:00.0` | 7 | 7 | lane 2、3、6、8、10 |
| `85:00.0` | 0 | 0 | 0 |
| `c4:00.0` | 0 | 0 | 0 |
| `d0:00.0` | 0 | 0 | 0 |
| `dd:00.0` | 0 | 0 | 0 |

前四张 GPU 正好对应当前绑定到 snvme 的四组 GPU/NVMe；后四组当前走普通 NVMe driver，未参与这批直通运行。因此可以说“直通 workload 与错误相关”，但还不能说“NVMe I/O 命令直接导致错误”。必须在后四组执行同样 workload，或者对前四组做 host-bounce/Gen4 A/B，才能区分 workload、fabric group 和 firmware preset。

`48:00.0` 的 corrected `Timeout` 当前累计为 2；`49:00.0` 的 fatal/non-fatal AER 均为 0。`nvidia-smi` 显示 GPU 仍保持 Gen5 x16，PCIe replay counter 为 0。这里应以 PCI AER sysfs 为准：NVML replay counter 为 0 并不能否定 endpoint 已上报的 `RxErr/BadTLP`。

### 2.3 时间线不支持“这次 NVMe command 写坏了链路”

用户给出的事件附近，本机日志顺序是：

```text
147030.139 - 147030.287  四次 memmap_init_zone_device
147065.073               49:00.0 corrected RxErr + BadTLP
147091.409 - 147091.413  四组 snvme user queue 创建
```

即 AER 比该轮 user queue 创建早约 26 秒。此前最后一个 queue group 已在 `146968s` 销毁。事件更靠近 GPU memory registration/mapping 阶段，而不是这轮 NVMe GPU submit 阶段。

这不能排除系统中还有其他 GPU PCIe traffic，但说明仅凭该时间点不能把根因归到 Tutti 的某个 NVMe read/write command。

## 3. 为什么不像 PRP、LBA 或 SQE 软件错误

软件描述符错误通常产生不同症状：

| 软件错误 | 更常见的可见结果 |
|---|---|
| PRP 指向错误但仍可访问的地址 | 数据写错位置、校验失败，不一定产生 AER |
| PRP 指向不可访问地址 | Unsupported Request/Completer Abort、IOMMU fault、NVMe timeout 或 Xid |
| SLBA/NLB 错误 | NVMe completion status、越界或错误文件数据 |
| SQ/CQ phase 或 queue depth 不一致 | CQ poll timeout、command timeout、重复/丢 completion |
| command length 未满足 MDTS/alignment | NVMe status error，或 Tutti submit 前拒绝 |

本次日志明确标记：

```text
aer_layer=Physical Layer
aer_agent=Receiver ID
RxErr
BadTLP
LaneErrStat: multiple lanes
```

NVMe controller 和 switch 负责生成 PCIe TLP；Tutti 只能填写 SQE、PRP 和 doorbell，不能从 CUDA kernel 构造电气层 symbol。错误的软件地址可能触发 Transaction Layer 错误，但不能解释多 lane Physical Layer receiver error。

因此 PRP/LBA 正确性仍然必须测试，但它是另一条问题链，不能用来解释或修复本次 `RxErr`。

## 4. 当前 Tutti 对该问题的实际能力

### 4.1 GPU staging 不能绕开故障链路

`integration/vllm-connector/adapter/worker.py:745` 在 CUDA device 上分配 staging ring。NVMe 无论直接 DMA 到 KV pool，还是先 DMA 到 GPU staging，最终都要经过相同的 switch -> GPU PCIe receiver。

所以当前两条路径是：

```text
DIRECT: NVMe -> GPU KV pool
STAGED: NVMe -> GPU staging -> D2D reshape -> GPU KV pool
```

两者都属于 GPU-direct。STAGED 能解决小 block、对齐和 layout reshape，不能作为 PCIe AER fallback。

真正绕开 P2P receiver path 的降级路径应为：

```text
HOST_BOUNCE load:
NVMe -> pinned host buffer -> cudaMemcpyAsync(H2D) -> GPU staging/KV pool

HOST_BOUNCE save:
GPU KV pool/staging -> cudaMemcpyAsync(D2H) -> pinned host buffer -> NVMe
```

这仍会使用 CPU root port 到 GPU 的普通 DMA，但不再让 NVMe peer 直接访问 GPU BAR/memory mapping；它可以用来区分 P2P-specific platform 问题，也可以在链路健康下降时保住服务。

### 4.2 local NVMe 目前没有 host-bounce data path

当前源码明确声明和拒绝 host I/O：

- `tutti/data_paths/local_nvme/local_nvme_data_path.cpp:98`：host memory registration 可工作，但 host I/O 未实现。
- `tutti/data_paths/local_nvme/local_nvme_data_path.cpp:99`：`supports_host_memory=false`。
- `tutti/data_paths/local_nvme/local_nvme_data_path.cpp:1435`：非 DEVICE memory submit 返回 `UNSUPPORTED`。

因此 connector 即使分配了 pinned host staging，也无法通过当前 local NVMe GPU data path 发 I/O。HOST_BOUNCE 需要新增 CPU/NVMe submission path，或委托 Linux `io_uring`/pread/pwrite，再接 CUDA async copy。

### 4.3 当前参数不是链路限流器

默认 local preset：

```text
num_queues = 16
threads_per_block = 16
max_batch_entries = 4096
max_in_flight_operations = 4
```

`max_batch_entries` 是 workspace/fan-out 容量，不会把一个实际 204/256-entry batch 自动切成小 wave。`max_in_flight_operations` 是 op admission，也不限制一个 kernel 内同时活跃的 submit threads。

当前 queue 映射公式不使用 `blockIdx`，但在 16 threads/16 queues 配置下，每个 CUDA block 的 16 个线程恰好覆盖全部 16 个 queue；不同 block 会共享 queue，底层 parallel queue 使用 atomic ticket。因此它不是本次 AER 的已确认根因。需要优化的是每个 endpoint 的实际 issue burst，而不是简单改一个 queue index 公式。

### 4.4 当前没有链路健康反馈

源码中没有读取以下信息：

- `/sys/bus/pci/devices/<GPU BDF>/aer_dev_correctable`
- GPU upstream/downstream port 的 corrected `Timeout`
- fatal/non-fatal AER counter
- 当前 link speed/width 和 Lane Error Status
- Xid/NVMe timeout 与 Tutti batch id 的关联

所以 runtime 无法知道某个 P2P path 正在发生 corrected recovery，也无法自动 throttle、quarantine 或 fallback。

## 5. 推荐的软件修改

### P0：增加只读 PCIe health guard

在 worker/runtime 初始化时，为每个 target 固化：

```text
gpu_bdf
nvme_bdf
gpu_downstream_port_bdf
switch_upstream_bdf
numa_node
current_link_speed/width
```

新增低频 host monitor，不放在每层热路径：

```text
PciePathHealthMonitor
  -> snapshot sysfs AER counters
  -> 每 1 秒或每个 step group 读取增量
  -> 关联 Tutti op/batch id、READ/WRITE、bytes、entry_count、streams
  -> 发布 HEALTHY / DEGRADED / QUARANTINED
```

策略必须可配置，建议初始语义为：

- 任一新 corrected event：记录 warning 和完整 path，不判当前 I/O 数据失败。
- corrected rate 连续超过阈值：降低 issue window 和 in-flight op。
- endpoint `RxErr/BadTLP` 与 upstream `Timeout` 同窗口增长：标记 DEGRADED，优先切 HOST_BOUNCE。
- 任一 non-fatal/fatal AER、link width/speed 意外下降、Xid、NVMe timeout/reset：停止该 target 新提交，quiesce 已发命令并 QUARANTINE。
- 只有完整重置并重新基线后才恢复 GPU-direct，不能仅清 status bit 后自动恢复。

Linux sysfs counter 是累计值，应保存 delta，不要依赖清零。`TOTAL_ERR_COR` 表示事件计数；一次事件可以同时设置 `RxErr` 和 `BadTLP`，不要把各 bit counter 相加后当事件数。

### P0：增加真实 issue window

新增与容量参数分离的配置：

```text
max_issue_entries_per_wave
max_issue_bytes_per_wave
max_active_waves_per_endpoint
inter_wave_delay_or_credit_policy
```

第一版可在 host plan 编译时把一个大 descriptor batch 切成 16/32/64-entry 子 batch，并保持同 stream 顺序。长期版本可由 GPU executor 使用 endpoint credit，在 GPU 上逐 wave 发命令。

诊断期建议从以下保守配置开始，而不是直接改永久默认值：

```text
max_issue_entries_per_wave = 16
max_active_waves_per_endpoint = 1
max_in_flight_operations = 1
read/write 不同时冲击同一 GPU endpoint
四个 rank 的首次 registration/submit 错峰
```

如果 AER rate 随 wave size 明显变化，说明 traffic burst 是触发器；这仍不等于软件是根因。

### P1：实现 HOST_BOUNCE fallback

建议把路径策略扩展为三态：

```text
DIRECT_GPU       大且对齐，NVMe <-> KV pool
STAGED_GPU       小/非对齐/需 reshape，NVMe <-> GPU staging
HOST_BOUNCE      PCIe path degraded，NVMe <-> host pinned <-> GPU
```

HOST_BOUNCE 必须满足：

- host buffer 长期 pin/register，不在每层注册。
- load 使用 NVMe completion -> H2D event -> compute wait。
- save 使用 D2H event -> NVMe submit -> durable completion。
- 切换前停止新 GPU-direct command，并等待旧 queue/CQ 完成。
- metadata commit 只依赖最终目标路径完成，切换不改变 key/segment/file offset。
- host pool exhaustion 必须 backpressure，不能静默回到不健康的 GPU-direct。

### P1：增加数据校验和 failure attribution

corrected AER 按协议不应丢数据，但研发期应为 sample batch 增加端到端 checksum：

```text
canonical segment checksum before save
file/NVMe readback checksum
load reshape 后 logical KV checksum
```

每条 completion 应保留：

```text
target BDF + GPU BDF + op id + direction + SLBA/NLB
memory registration generation + PRP template generation
queue id + command id + NVMe status
AER delta before/after batch window
```

这样可以把“链路发生纠错”和“某个 segment 数据错误”分开，避免仅凭同一秒日志做因果判断。

### P2：把 topology 作为路径选择条件

当前 49/4b 已是同 switch，本项不会修复当前错误，但应防止未来跨 root P2P：

- DIRECT/STAGED_GPU 只允许同 root complex，并优先同 switch。
- 跨 root、ACS redirect 或 IOMMU 非 identity mapping 时使用 HOST_BOUNCE。
- 将 BDF 和实际 route 写入 capability report，不能只配置 `gpu_id` 和 `/dev/ssnvmeN`。

## 6. 推荐的运维/硬件处理顺序

### 6.1 先更新完整 firmware stack

本机平台信息：

```text
System: Tencent XG262
BIOS: American Megatrends 1.10.06
BIOS date: 2024-07-15
Kernel: 5.4.241-1-tlinux4-0017.7
NVIDIA driver: 580.105.08
```

向整机厂/OEM 获取与 H20、PEX890xx 和该 riser/retimer 拓扑匹配的验证版本，至少核对：

1. SBIOS/BMC/CPLD。
2. Broadcom PEX890xx switch firmware/preset。
3. PCIe Gen5 retimer firmware 和 equalization preset。
4. H20 VBIOS/InfoROM 与 NVIDIA driver 组合。
5. Samsung NVMe firmware。

不能直接套用 DGX 的 firmware 文件，但 NVIDIA 的公开 firmware release notes 多次包含“通过 PCIe switch/retimer preset 修复 AER”的案例，说明这类问题确实可能由 platform firmware 解决。

### 6.2 用 Gen4 x16 做最有价值的 A/B

在维护窗口通过 OEM BIOS/BMC/PCIe port 配置，把受影响 GPU downstream link 从 Gen5 x16 降为 Gen4 x16；不要在生产运行中用 `setpci` 强制 retrain。

这台机器每张 GPU 对应的 NVMe 只有 Gen4 x4。理论链路带宽上，GPU Gen4 x16 仍明显高于单盘 Gen4 x4，因此这个 workaround 很可能不限制单盘直通上限，却能大幅增加 GPU link 的信号裕量。

判断：

- Gen4 下错误停止：优先处理 Gen5 retimer/switch/board preset，不要继续靠应用限流掩盖。
- Gen4 下仍增长：检查 riser、connector、GPU、switch port，并做部件交换定位。
- 错误跟随 GPU：GPU/link module/VBIOS 方向。
- 错误留在 slot/port：riser/retimer/switch/主板方向。

### 6.3 不要通过屏蔽 AER“解决”

不建议使用以下方式作为修复：

```text
pci=noaer
修改 Correctable Error Mask 屏蔽 RxErr/BadTLP
关闭 GHES/APEI 日志
只降低 kernel log level
```

这些操作只隐藏告警和计数，既不减少重传/恢复开销，也不能阻止 corrected error 升级为 uncorrectable。当前日志来自 APEI/GHES firmware-first 路径，更应从 OEM firmware 和物理链路处理。

## 7. 最小 A/B 验证矩阵

不需要一开始运行完整 vLLM。应在维护环境按下表逐项进行，并只比较测试前后 counter delta：

| 变量 | A | B | 用途 |
|---|---|---|---|
| 数据路径 | HOST_BOUNCE | GPU-direct | 判断是否 P2P-specific |
| GPU link | Gen4 x16 | Gen5 x16 | 判断 Gen5 margin/retimer |
| I/O 方向 | NVMe -> GPU | GPU -> NVMe | 判断 receiver traffic pattern |
| issue wave | 16 entries | 64/256 entries | 判断 burst 敏感性 |
| active ops | 1 | 4 | 判断并行压力 |
| fabric group | 前四组 | 后四组 | 判断错误跟 workload 还是 slot group |
| 部件 | 原 slot/GPU | 交换 GPU 或 riser | 判断错误跟随关系 |

每轮至少采集：

```bash
cat /sys/bus/pci/devices/0000:49:00.0/aer_dev_correctable
cat /sys/bus/pci/devices/0000:49:00.0/aer_dev_nonfatal
cat /sys/bus/pci/devices/0000:49:00.0/aer_dev_fatal
cat /sys/bus/pci/devices/0000:48:00.0/aer_dev_correctable
lspci -s 49:00.0 -vv
nvidia-smi -q -i 1
dmesg | tail -n 300
```

这些读取不会清空 counter。测试报告必须记录开始/结束时间、GPU/NVMe BDF、方向、总字节、wave size、queue 数和 link speed/width。

## 8. 判定标准

当前状态可定义为 `DEGRADED_CLEAN`：

- 有 corrected Physical/Data Link errors。
- 没有 fatal/non-fatal、Xid、NVMe reset/timeout。
- link 未 downtrain，仍为 Gen5 x16。
- 暂无证据表明数据损坏。

它不要求立即中止当前已完成请求，但不应直接进入长期生产验收。生产标准至少应是：

1. 固定 workload 窗口内 AER delta 为 0，或低于 OEM 明确给出的允许阈值。
2. 无 upstream Timeout、无 Lane Error 新增、无 Xid/NVMe timeout。
3. 端到端数据校验通过。
4. Gen5/Gen4、HOST_BOUNCE/GPU-direct A/B 已能解释错误跟随关系。
5. Tutti 有可观测的 throttle/quarantine/fallback，而不是继续无限提交。

## 9. 参考资料

- Linux PCIe AER：corrected error 由硬件恢复，不要求软件 recovery，但应保留统计与监控：<https://docs.kernel.org/next/PCI/pcieaer-howto.html>
- NVIDIA GPUDirect RDMA：P2P 最优路径、root complex/IOMMU 和 platform chipset 限制：<https://docs.nvidia.com/cuda/gpudirect-rdma/>
- NVIDIA GPUDirect Storage troubleshooting：peer affinity、跨 root traffic 和 GDS 诊断：<https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/>
- NVIDIA DCGM PCIe diagnostics：link downtrain、recovery/replay/correctable error threshold：<https://docs.nvidia.com/datacenter/dcgm/latest/reference/diagnostics/plugins/pcie.html>
- NVIDIA firmware release 示例：PCIe switch/retimer firmware/preset 可修复 AER，但具体版本必须使用本机 OEM 包：<https://docs.nvidia.com/dgx/dgxa100-fw-container-release-notes/ver-20-11_3.html>

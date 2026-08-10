# 多加速器 Runtime 阶段 2 实施记录

## 基线与范围

- 分支：`fix/multi-gpu`
- 阶段 2基线 HEAD：`45c6976`（`docs: add multi-accelerator phase 2 handoff`）
- 阶段 1实现：`1eed707`
- 实施依据：`doc/impl/multi-accelerator-phase-2-handoff.md` 和设计文档第 12.3 节
- 测试日期：2026-08-10 UTC

本阶段只实现 accelerator current-device 与归属检查，未进入阶段 3；daemon schema、
protobuf/gRPC RPC 和 NVMeService 契约没有修改。

## 实现摘要

### DeviceGuard

新增 `tutti/include/tutti/accelerator_device_guard.h`。它采用 CUDA-like backend
抽象，保存调用线程的 current accelerator，进入时切换到目标 ordinal，正常路径要求
显式 `restore()` 并返回恢复错误，析构函数只做 `noexcept` best-effort 兜底。
`accel_id=-1` 是 OptionalDeviceGuard 风格 no-op；HOST profile 对任意目标都保持可观测
的 no-op。测试 seam 对应 PyTorch `DeviceGuardImplInterface` 的最小 get/set 形状，
但项目没有引入 libtorch、C10 或其他外部依赖。

### Runtime 与 DataPath 边界

- `StorageRuntime` 对 DataPath 的 initialize、open、close、registration_domain、
  register/unregister、submit、progress、query、release、shutdown 使用 guard。
- Local/Striped DataPath 的公共 SPI 方法保留第二层 guard，内部逻辑拆到 `*_impl_`，
  因而直接调用 DataPath 也不依赖调用线程预先选择正确 GPU。
- queue group、device target、arena/cache、event、DMA map/unmap、kernel launch 和
  doorbell 路径由这些边界 guard 兜底；析构清理路径同样先选择绑定 accelerator。
- Runtime 校验 DEVICE/MANAGED pointer ordinal、CUDA stream ordinal、memory/context
  accelerator 和 `HostSubmitContext.accel_id`。未填写的 `accel_id=-1` 解析为 Runtime
  accelerator；显式冲突在 DMA map、kernel launch 或 doorbell 前拒绝。HOST execution
  保留既有 host ordinal 兼容行为。
- submit 的 guard 恢复失败会拒绝该组、回滚 Runtime inflight credit，并尝试释放
  DataPath 已返回的 opaque operation，避免设备侧操作泄漏。

## 验证结果

| 验证 | 结果 | 命令/证据 |
| --- | --- | --- |
| HOST 构建 | PASS | `cmake --build build/host --parallel 8` |
| HOST 非硬件 CTest | PASS | `ctest --test-dir build/host --output-on-failure -j 8`，17/17 |
| CUDA 构建 | PASS | `cmake --build build/cuda --parallel 8` |
| CUDA 非硬件 CTest | PASS | `ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8`，18/18 |
| DeviceGuard 直接测试 | PASS | `build/host/bin/tutti_accelerator_device_guard_test`、`build/cuda/bin/tutti_accelerator_device_guard_test` |
| Local/Striped/Runtime hardware contract 编译 | PASS | `tutti_local_nvme_datapath_contract_test`、`tutti_storage_runtime_local_nvme_contract_test`、`tutti_striped_local_nvme_contract_test` |
| 格式检查 | PASS | `git diff --check` |

CUDA 测试覆盖了两个 backend ordinal（0/1）、双线程 current-device 隔离、成功/错误/
early-return/异常清理恢复、guard 切换与恢复失败诊断，以及错误 pointer/stream/context
归属拒绝。Local/Striped hardware contract 还编译并包含了错误 caller current device 下
直接 initialize/shutdown 的恢复探针。

本机可见两张 NVIDIA H100 PCIe：ordinal 0 的 BDF 为
`00000000:4B:00.0`，ordinal 1 的 BDF 为 `00000000:4C:00.0`。GPU 1 上已有其他
进程，验证期间没有终止或干预它们。

## 实机 NVMe 验证

使用 `config/local/daemon_2disk.yaml` 启动了支持 `auto_mount` 生命周期的
`tutti_daemon`（密码通过 `sudo -S` 的受保护 stdin 提供，未进入命令行或日志）。启动
后 RPC `ListDevices` 返回并实际使用了以下 metadata：

| device | BDF | chrdev | backing block | mount | blk/bar0 | queue limits |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `0000:41:00.0` | `/dev/ssnvme0` | `/dev/snvme0n1` | `/mnt/snvme/nvme1` | `4096/32768` | `user_io_qps=96`, `max_q/grp=16` |
| 1 | `0000:44:00.0` | `/dev/ssnvme1` | `/dev/snvme1n1` | `/mnt/snvme/nvme2` | `4096/32768` | `user_io_qps=96`, `max_q/grp=16` |

真实 I/O contract 结果（均以 root 运行，避免 mount 下 root-owned GPU view 目录的
权限干扰）如下：

| contract | 结果 | 证据 |
| --- | --- | --- |
| `tutti_local_nvme_datapath_contract_test` | PASS（复跑） | `824 passed, 0 failed`；包含第 58 项双线程 race 和双盘 78--81 场景 |
| `tutti_storage_runtime_local_nvme_contract_test` | PASS | `156 passed, 0 failed`；SINGLE/DUAL/LIST、跨 segment、batch、超时和双 host thread |
| `tutti_striped_local_nvme_contract_test` | PASS | `57 passed, 0 failed`；双盘跨 shard、持久化重启；双盘 READ speedup `1.57x` |

Local contract 的首轮运行在第 58 项同一 `LocalNvmeDataPath` 的并发
`submit/query/release` 测试中以 SIGSEGV 退出（栈顶为 `progress_impl_` 遍历共享
`ops_`）；行缓冲复跑完整通过。该 DataPath 在阶段 2基线中没有并发锁，故这是仍需后续
线程安全工作跟进的非确定性残余风险；阶段 2只引入 device guard，没有扩大为并发模型
重构。

测试结束后发送单次 SIGINT，daemon 输出 `tutti_daemon exited cleanly`，并确认：

- `/mnt/snvme/nvme1`、`/mnt/snvme/nvme2` 已卸载；
- `/dev/ssnvme*`、`/dev/snvme*` 和 `:50051` 均已消失；
- 两个配置 BDF 均恢复为 unbound，`snvme`/`snvme_core` 模块仍正常加载；
- 测试 client fd 和 queue/map 均由内核回收；
- GPU 1 上验证前已有的 Python compute processes 未被终止或干预，daemon 未创建
  额外 compute process。

阶段 2没有实现跨 accelerator submit/P2P、MANAGED memory 协议或 Runtime accelerator
allocation；MUSA/MACA 仍依赖现有 cuda-like shim，缺少可移植 stream-owner 查询时按
既定策略由调用者保证 stream 创建于 Runtime accelerator。上述项目属于后续阶段或
backend 专项工作，不影响本阶段的 HOST/CUDA contract 结论。

## 阶段 2出口

阶段 2实现、测试和本文档在独立提交中完成。完成后停止，不实施阶段 3；后续由新会话
继续 daemon schema/RPC 工作。

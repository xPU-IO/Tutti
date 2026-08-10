# 多加速器 Runtime 阶段 1 实施记录

## 范围

本阶段完成 `doc/design/multi-accelerator-runtime.md` 第 12.2 节定义的 Runtime 和
DataPath accelerator 身份基础，基线提交为 `807b49e`。

- 构建 profile 导出 `TUTTI_COMPILED_ACCELERATOR_PROFILE` 和
  `TUTTI_DEFAULT_ACCEL_ID`。
- `RuntimeConfig` 增加 `accel_id`；Runtime 创建时在任何 DataPath initialize 前校验
  编译 profile、backend device count 和 accelerator ordinal。
- 通用 API 中 accelerator 含义的 `device_id`/`expected_device_id` 迁移为
  `accel_id`/`expected_accel_id`。
- `DataPathCapabilities` 增加 `bound_accel_id`；Local/Striped DataPath 写入固定绑定，
  Runtime 通过无副作用 preflight 拒绝绑定冲突。
- backend discovery 返回真实 device count、名称、显存和可选 PCI BDF；BDF 仅用于
  诊断。
- 未实现 backend allocation 前，`allocate_memory(DEVICE/MANAGED)` 明确返回
  `UNSUPPORTED`，不再用 host `malloc()` 伪装 accelerator memory。
- memory 注册和 submit context 增加 Runtime accelerator 归属检查；跨 Runtime handle
  隔离保持原契约。

## 身份决策

根据阶段 1 实施期间的设计修订，accelerator 身份直接使用编译 backend 提供的
ordinal：

- `accel_id=0/1/...` 就是 backend ordinal，不引入 UUID 或 `stable_id`。
- 不提供 `CUDA_VISIBLE_DEVICES` 或同类 ordinal 重映射兼容层。
- daemon、client 和 Runtime 的部署必须使用相同 ordinal 顺序。
- PCI BDF 可以出现在发现结果和日志中，但不参与 Runtime 创建、DataPath 绑定或归属
  判定。

该决策已同步到设计文档和阶段 0 实施记录。实现中没有 accelerator UUID 查询或
公开 UUID 字段。

## 实现摘要

### Runtime

- `StorageRuntime::create()` 首先执行 profile/ordinal 校验；CUDA/MACA/MUSA 使用
  backend device count，HOST profile 使用 host-only `accel_id=-1`。
- `query_cuda_like_profile()` 和 `list_devices()` 不再返回单个 stub device。
- `RuntimeComponents` 在 initialize 前检查空 binding、重复 scheme/key、非法
  `bound_accel_id` 和 Runtime/DataPath 绑定冲突；失败不会初始化任何 DataPath。
- 同一 `accel_id` 的多个 DataPath 仍可创建，没有收缩已有的多 scheme/multi-key
  核心路由能力。

### API 与 DataPath

- `HostSubmitContext`、`MemoryView`、`DataPathMemoryView`、`DeviceInfo`、
  `DeviceCapabilities`、`MemorySpec` 和 `MemoryInfo` 使用 `accel_id` 语义。
- Local/Striped DataPath 的能力和 submit 二次检查使用 `bound_accel_id`/
  `ctx.accel_id`；daemon 的 NVMe `device_id` 保持独立含义。
- contract fixture、memfs、mock DataPath 和相关测试调用点已同步迁移。

## 验证结果

测试日期：2026-08-10 UTC。CUDA 机器提供两个 NVIDIA H100 PCIe，backend ordinal
分别为 0 和 1；BDF `0000:4b:00.0`、`0000:4c:00.0` 仅作为诊断记录。

| 验证 | 结果 | 证据摘要 |
| --- | --- | --- |
| HOST StorageRuntime contract | PASS | `build/host/bin/tutti_storage_runtime_contract_test`，40/40 |
| CUDA StorageRuntime contract | PASS | `build/cuda/bin/tutti_storage_runtime_contract_test`，40/40；显式枚举并创建 ordinal 0/1 |
| HOST 无硬件 CTest | PASS | `ctest --test-dir build/host --output-on-failure -j 8`，16/16 |
| CUDA 非 hardware label CTest | PASS | `ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8`，16/16 |
| CUDA DataPath/daemon 编译 | PASS | Local、Striped、StorageRuntime contract、daemon、nvmeservice 和 client example 均构建成功 |
| 格式检查 | PASS | `git diff --check` 无错误 |

阶段 1 新增的 contract 覆盖 profile/ordinal 校验、真实 device count 和 ordinal
枚举、initialize 前 binding preflight、同 accelerator 多 DataPath，以及
DEVICE/MANAGED allocation 不伪造。已有 StorageRuntime test 83 继续通过，证明跨
DataPath batch 分组能力没有被删除。

本阶段没有修改 daemon schema、NVMe bring-up、DMA map 或 kernel 路径，因此没有
重复运行阶段 0 的 daemon/local-NVMe/striped 实机 I/O。阶段 2 修改 current-device
边界后必须重新运行相应 CUDA contract，并按风险决定是否进入真实 NVMe I/O。

## Exit Gate

阶段 1 exit gate 通过：Runtime profile、ordinal 和 DataPath binding 冲突都在
DataPath initialize、DMA map、kernel launch 和 doorbell 前失败；HOST/CUDA 回归全绿。

## 下一阶段交接

下一会话只实施阶段 2：accelerator current-device 与归属检查。开始前应读取：

1. `doc/design/multi-accelerator-runtime.md` 第 12.3 节。
2. 本文档，确认阶段 1 已完成的 ordinal-only 身份契约和测试基线。
3. `config/local/daemon_2disk.yaml`，仅在阶段 2 验证需要 daemon/NVMe 时使用。

阶段 2 的核心任务是 backend-neutral current-device RAII guard、Runtime/DataPath 全调用
边界的保存/切换/恢复、pointer/stream/memory/context 归属校验，以及双线程/双 GPU
验证。不得重新引入 UUID、`stable_id` 或 `CUDA_VISIBLE_DEVICES` 兼容逻辑，也不得把
daemon NVMe `device_id` 当作 accelerator identity。

阶段 2 完成并验证通过后，必须创建独立 git commit 和
`doc/impl/multi-accelerator-phase-2.md`；然后停止继续实现，由用户创建新的会话接力
阶段 3。后续每个阶段都遵循同样的“单阶段实现、验证、文档、提交、换会话”流程。

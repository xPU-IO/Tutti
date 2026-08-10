# 多加速器 Runtime 阶段 2 接力说明

## 接力起点

- 工作分支：`fix/multi-gpu`。
- 阶段 0 提交：`807b49e`。
- 阶段 1 提交：`1eed707`（`runtime: establish accelerator identity contract`）。
- 阶段 1 实施与验证记录：`doc/impl/multi-accelerator-phase-1.md`。
- 总体设计：`doc/design/multi-accelerator-runtime.md`，阶段 2 对应第 12.3 节。

新会话开始后先确认 `git status` 干净、`HEAD` 包含 `1eed707`，再实施阶段 2。不要
修改或重做已经完成的阶段 0/1。

## 已完成契约

阶段 1 已完成 Runtime/DataPath accelerator identity 基础：

- `RuntimeConfig.accel_id`、编译 profile 和 backend device count 校验。
- 通用 API 的 accelerator 字段统一为 `accel_id`/`expected_accel_id`。
- `DataPathCapabilities.bound_accel_id` 和 initialize 前 binding preflight。
- Local/Striped DataPath 固定 accelerator binding。
- 真实 backend ordinal 0/1 发现和创建验证。
- DEVICE/MANAGED runtime allocation 尚未实现时返回 `UNSUPPORTED`。
- HOST/CUDA StorageRuntime contract 各 40/40，完整 CTest 各 16/16。

accelerator identity 直接使用编译 backend ordinal。不得引入 UUID、`stable_id` 或
`CUDA_VISIBLE_DEVICES` 兼容逻辑；PCI BDF 只允许作为诊断元数据。daemon 的
`device_id` 只表示 NVMe resource，不能用作 accelerator identity。

## 阶段 2 目标

只实施设计第 12.3 节“accelerator current-device 与归属检查”：

1. 增加 backend-neutral RAII device guard，保存线程原 device，切换到目标
   `accel_id`，正常路径显式恢复并传播错误，析构仅作 no-throw best-effort 兜底。
2. 在 Runtime 调用 DataPath 的 initialize、open、close、register/unregister、submit、
   progress、release 和 shutdown 边界使用 guard。
3. Local/Striped DataPath 保留第二层 guard，保证直接调用 DataPath 也正确。
4. 审计 queue group、device target、arena/cache、event、DMA map/unmap 和析构路径；
   accelerator API 不得依赖调用线程预先设置正确 current device。
5. 校验 DEVICE memory、pointer、stream 和 `HostSubmitContext.accel_id` 对 Runtime 的
   归属。`accel_id=-1` 按设计解析为 Runtime accelerator；显式冲突立即拒绝。
6. 保持 HOST profile 和 MACA/MUSA 编译接口可用，优先复用现有 cuda-like abstraction。

不要提前实现阶段 3 daemon schema/RPC，也不要扩展为跨 accelerator submit、P2P 或
MANAGED memory 协议。

## 验证要求

每次验证先运行无硬件层，再运行 CUDA device contract：

```bash
cmake --build build/host --parallel 8
ctest --test-dir build/host --output-on-failure -j 8

cmake --build build/cuda --parallel 8
ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8
```

阶段 2 必须新增测试覆盖：

- current device 在成功、错误、early return 和异常清理路径后恢复。
- Runtime/DataPath 直接调用均可在错误的 caller current device 下工作。
- `accel_id=0/1`、双线程分别绑定不同 GPU，互不污染线程 current device。
- pointer/stream/memory/context 属于错误 accelerator 时在 DMA map、kernel launch 和
  doorbell 前失败。
- guard 切换或恢复失败有确定状态和诊断；HOST guard 是可验证的 no-op。
- 阶段 1 的 40 项 StorageRuntime contract、test 83 和 Local/Striped 既有 contract
  不回归。

本机有两个 NVIDIA H100 PCIe，backend ordinal 为 0/1。需要 daemon/NVMe 验证时使用
`config/local/daemon_2disk.yaml`；sudo 密码可从 `~/.passwd/1` 输入，禁止打印或写入
日志。硬件和测试命令、结果、未覆盖风险必须记录在
`doc/impl/multi-accelerator-phase-2.md`。

## 完成规则

阶段 2 验证通过后：

1. 提交阶段 2 实现、测试和 `doc/impl/multi-accelerator-phase-2.md`，使用独立 git
   commit。
2. 报告 commit hash、验证结果和残余风险。
3. 停止，不进入阶段 3。

用户会在阶段 2 完成后再创建一个新会话接力阶段 3。后续每个阶段都继续执行
“单阶段实现、验证、实施文档、独立提交、停止并换新会话”的流程。

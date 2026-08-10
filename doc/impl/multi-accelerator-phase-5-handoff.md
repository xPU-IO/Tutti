# 多加速器 Runtime 阶段 5 接力说明

## 接力起点

- 工作分支：`fix/multi-gpu`。
- 阶段 0 提交：`807b49e`。
- 阶段 1 提交：`1eed707`（`runtime: establish accelerator identity contract`）。
- 阶段 2 提交：`61af360`（`runtime: implement multi-accelerator phase 2 device guards`）。
- 阶段 3 提交：`1cee2b7`（`daemon: implement multi-accelerator phase 3 allocation`）。
- 阶段 4 提交：`a493603`（`runtime: implement phase 4 allocation loader`）。
- 阶段 3 实施与验证记录：`doc/impl/multi-accelerator-phase-3.md`。
- 阶段 4 实施与验证记录：`doc/impl/multi-accelerator-phase-4.md`。
- 总体设计：`doc/design/multi-accelerator-runtime.md`，阶段 5 对应第 12.6 节；实机
  流程还应同时阅读第 11、13.2--13.4 节。

新会话开始后先运行 `git status --short --branch` 和 `git log -1 --oneline`。工作树
应当干净。保留本文，不要修改或重做已经完成的阶段 0--4。

## 已完成契约

阶段 0--4 已经固定并实现以下前置契约：

- accelerator identity 统一为 `accel_id`；daemon 的 `device_id` 只表示 NVMe
  resource。
- 一个 `StorageRuntime` 固定一个 `accel_id`；Local/Striped DataPath 固定绑定同一
  accelerator，创建前执行 binding preflight。
- Runtime 和 DataPath 的 accelerator API 边界使用 backend-neutral guard，设置目标
  device 并恢复 caller current device。
- pointer、stream、memory、context 和 Runtime accelerator 的归属冲突会在 DMA map、
  kernel launch 或 doorbell 之前失败。
- daemon canonical schema 为 `accelerators[].accel_id/view_root`、
  `nvmes[].device_id/backing_mount_path/allowed_accel_ids`；完整 legacy-only schema
  仍在迁移窗口内可读，但新旧字段混用失败。
- daemon 提供 `ListAccelerators`、`ListNvmeResources`、`AcquireNvmeSlices`、
  `Release`，并保留旧 `ListDevices/Connect/Disconnect/Heartbeat` wire/source
  compatibility。
- daemon allocation 支持 `allowed`、`explicit`、`striped`；一次 striped allocation
  对外只有一个 `allocation_id`，内部包含多个有序 slice，并通过统一 Release/reaper
  回收全部 queue reservation。
- daemon 不创建 accelerator runtime context；`available` 表示 bring-up/mount/view/
  queue admission 可用，不等于已经完成真实 I/O 的 `validated_available`。
- `tutti/config` loader 已消费 canonical 配置和新 RPC metadata：解析
  `accelerator.profile`、`runtime.accel_id`、`nvme_service.endpoint`、
  `nvme.selection/device_ids/queues_per_controller/stripe_unit`，并通过可注入
  `RuntimeResourceClient` seam 获取 allocation。
- loader 不再从 daemon YAML、数组序号、`accel_id` 或 `device_id` 推导数据路径；
  单 slice 构造唯一 `LocalFileResolver + LocalNvmeDataPath`，多 slice 构造唯一
  `StripedResolver + StripedDataPath`。
- `TuttiRuntime` owned bundle 显式管理 `StorageRuntime`、顶层组件、resource client
  和 allocation 生命周期；创建失败与 shutdown/destructor 路径应 Release 一次。

阶段 4 留下的明确下一步是：用真实 daemon allocation metadata 通过
`load_tutti_config()` 跑通单 Runtime 的单盘和 striped 真实 I/O 闭环。本阶段只做
单 Runtime 实机闭环，不进入阶段 6 的双 Runtime、双 accelerator 并发矩阵。

## 当前实现差距

当前实现已经有 fake-client loader contract 和 daemon allocation smoke，但还缺少阶段
5 必须证明的实机闭环：

- 还没有一个硬件 contract test 从 `load_tutti_config()` 创建 `TuttiRuntime` bundle，
  再通过 public `StorageRuntime` API 执行 open/register/submit/wait/close/unregister/
  shutdown。
- 阶段 4 的双盘 daemon 验证只跑了 `nvmeservice_client --skip-io` allocation/attach
  smoke；它不能替代 Local/Striped DataPath 的真实 kernel I/O correctness。
- 现有硬件 contract 多数仍由测试代码手工构造 resolver/DataPath，尚未覆盖“loader
  从 daemon slice metadata 装配组件”的完整路径。
- `config/tutti_config.yaml` 现在是单个配置文件；阶段 5 需要为不同 accelerator、
  explicit device 和 striped selection 生成或临时写入测试配置，避免人工修改仓库内
  示例配置。
- 阶段 4 的 `TuttiRuntime::shutdown()` 调用 `StorageRuntime::shutdown(0)`；阶段 5
  必须在真实 I/O 完成、memory unregister、target close 后显式 shutdown，避免把
  drain 超时当作成功。
- 阶段 2/3 记录的 Local DataPath `ops_` 并发风险仍存在。本阶段按单 Runtime 顺序
  场景推进，除非真实 I/O 闭环直接触发且无法绕开，否则不要在阶段 5 扩大修复范围。

## 阶段 5 目标

只实施设计第 12.6 节“单 Runtime 的单盘与 striped 实机闭环”：

1. 新增或扩展硬件 contract test，使其通过 `load_tutti_config()` 创建一个 owned
   Runtime bundle，而不是手工拼 resolver/DataPath。
2. 在 GPU 0 + NVMe 0 上，用 `explicit` selection 完成 Local Runtime scratch file
   write/read/verify。
3. 在 GPU 0 + NVMe 1 上，只修改配置中的 daemon `device_id`，路径完全来自
   allocation metadata，同样完成 Local Runtime scratch file write/read/verify。
4. 在 GPU 0 + NVMe 0/1 上，用 `striped` selection 创建一个 Striped Runtime，完成
   跨 stripe boundary、mixed batch 和 restart persistence 的真实 I/O 验证。
5. 在 GPU 1 上重复上述单盘和 striped 场景，证明 loader/DataPath 不含 GPU 0 假设。
6. 每个场景以另一个 GPU 作为调用线程 current device 进入 initialize/open/register/
   submit/progress/wait/close/unregister/shutdown，并检查返回后恢复原 current device。
7. 每个场景都验证 daemon allocation ledger：Acquire 后 queue reservation 增加，
   bundle shutdown/destructor 后 reservation 精确归零；striped 一次 Release 归还两个
   controller reservation。
8. 记录真实 owner-returned `chrdev_path`、`block_path`、`backing_mount_path`、
   `view_path`、BAR0、logical block size、MDTS 和 queue grant；测试不得依赖推导路径。
9. 提供阶段 5 实施记录 `doc/impl/multi-accelerator-phase-5.md`，写明硬件环境、命令、
   测试结果、daemon 清理状态和残余风险。

## 明确非目标

阶段 5 不实施以下内容：

- 不实现同进程双 Runtime、双 accelerator 并发；这是阶段 6。
- 不实现 Runtime 0/1 分盘并发、共享同一 NVMe 并发、双 striped allocation 并发、
  跨 Runtime handle/pointer/stream/context 负例矩阵；这些属于阶段 6。
- 不删除 daemon legacy YAML/RPC compatibility。
- 不把 daemon RPC 放进 `StorageRuntime` 公共 API；allocation 生命周期仍属于
  `TuttiRuntime` owned bundle。
- 不新增调度器、NUMA 策略、自动故障转移、P2P 拓扑选择、probe worker 或
  `validated_available` daemon 语义。
- 不把 `--skip-io` 当作阶段 5 correctness 证据。`--skip-io` 只能用于 daemon attach
  诊断。
- 不对裸 namespace 做破坏性覆盖；所有 I/O 只写测试专属 scratch file 或 fixture。
- 不做大规模性能调优、长稳压测或队列耗尽压力；只保留必要的 correctness 和清理
  断言。

## 冻结的阶段 5 配置与测试语义

阶段 5 的测试配置必须走阶段 4 canonical loader path：

```yaml
accelerator:
  profile: "CUDA"

runtime:
  accel_id: 0

nvme_service:
  endpoint: "127.0.0.1:50051"

nvme:
  selection: "explicit"        # allowed | explicit | striped
  device_ids: [0]              # explicit 恰好 1 个；striped 至少 2 个
  queues_per_controller: 4
  stripe_unit: 65536
```

规则：

- 测试可以在临时目录生成多份 `tutti_config.yaml`，但不要修改仓库内
  `config/tutti_config.yaml` 来切换场景。
- `accelerator.profile` 必须与编译 profile 一致；阶段 5 实机验证使用 CUDA build。
- `runtime.accel_id` 只表示 accelerator ordinal；`nvme.device_ids` 只表示 daemon
  NVMe resource ID。
- explicit 单盘场景只改变 `runtime.accel_id` 或 `nvme.device_ids`，不能修改任何
  `/dev` path。
- striped 场景必须设置 `selection: striped`、`device_ids: [0, 1]` 和明确
  `stripe_unit`；测试需覆盖跨 stripe boundary 的读写。
- 测试 target URI 应使用 loader 注册的唯一顶层 scheme：Local 用 `file://...`，
  striped 用 `striped://...`。URI 中的路径/挂载点必须与现有 resolver 约定一致，并
  来自 daemon view/backing 语义或测试 fixture，不从 `device_id` 推导 controller path。
- 每个测试场景都要显式 close target、unregister memory、wait/release terminal I/O，
  再调用 `TuttiRuntime::shutdown()`；析构路径只作为兜底或单独负例。
- 若当前 device 被预设为非目标 GPU，测试必须记录并断言 API 返回后恢复；如果机器
  少于两个 GPU，相关 current-device 恢复场景应 fail with skip code，而不是伪通过。

## 主要代码落点

- loader hardware/E2E contract test：建议新增
  `tests/runtime_bundle_loader_contract/`，或在 `tests/config_loader/` 下新增单独的
  hardware target。不要把需要 root/GPU/NVMe 的测试混入纯 fake-client loader test。
- 配置测试 helper：可新增小型临时 YAML writer，生成 explicit/striped、accel 0/1、
  device 0/1 组合。
- Runtime bundle 使用入口：`tutti/config/tutti_config.{h,cpp}`。
- 现有 Local E2E 可复用参考：
  `tests/storage_runtime_local_nvme_contract/storage_runtime_local_nvme_contract_test.cpp`。
- 现有 Striped E2E 可复用参考：
  `tests/striped_local_nvme_contract/striped_local_nvme_contract_test.cpp`。
- daemon/client 资源检查可复用：
  `tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client.cpp` 或
  `tutti/device_manager/nvme/nvmeservice/src/nvmeservice_client.h`。
- CMake：`tutti/CMakeLists.txt`、新增测试目录 `CMakeLists.txt`、必要时
  `tutti/config/CMakeLists.txt`。
- 阶段实施记录：新增 `doc/impl/multi-accelerator-phase-5.md`。

## 建议实施顺序

1. 先新增硬件测试 target 骨架，接收 `--config` 或直接接收 endpoint/accel/device 参数
   后生成临时 `tutti_config.yaml`。测试默认用 skip return code 表示硬件条件不足。
2. 复用现有 Local Runtime contract 的 scratch file 准备、aligned host/device buffer、
   write/read/verify helper，但 Runtime 必须由 `load_tutti_config()` 创建。
3. 实现 GPU 0 + NVMe 0 explicit 单盘场景，确认 loader-created Local resolver/DataPath
   能完成真实 I/O，并在 shutdown 后 daemon reservation 回到 0。
4. 实现 GPU 0 + NVMe 1 explicit 单盘场景，只改 `device_id`，用断言证明实际 chrdev/
   block path 来自 allocation metadata。
5. 实现 GPU 0 + NVMe 0/1 striped 场景，覆盖至少一个跨 stripe boundary 的 write/read
   和 mixed batch；shutdown 后两个 controller 的 reservation 都回到 0。
6. 在 GPU 1 上重复 explicit device 0、explicit device 1 和 striped `[0,1]` 场景。
   如果设备数不足两个，清晰 skip GPU 1 子场景。
7. 为每个场景加 current-device 恢复检查：调用前设置为非目标 GPU，调用后断言恢复。
8. 增加 daemon ledger/resource snapshot helper，记录每个场景前、Acquire 后、shutdown
   后的 `reserved_queues/available_queues`。
9. 跑最小实机矩阵，修正 loader/DataPath/resolver 的真实 metadata 缝合问题；不要顺手
   扩展到双 Runtime。
10. 补 `doc/impl/multi-accelerator-phase-5.md`，运行 `git diff --check`，提交阶段 5。

## 无硬件验证要求

阶段 5 是实机阶段，但仍必须保证无硬件层不回退：

```bash
cmake --preset cuda
cmake --build --preset cuda --parallel 8 --target \
  tutti_config_loader_test \
  tutti_storage_runtime_contract_test \
  nvmeservice_phase3_contract_test
ctest --test-dir build/cuda --output-on-failure -R \
  'tutti_config_loader_test|tutti_storage_runtime_contract_test|nvmeservice_phase3_contract'
```

如改动触及公共 Runtime/SPI 或 daemon parser/state，再运行：

```bash
ctest --test-dir build/cuda -LE hardware --output-on-failure -j 8
```

若改动影响 HOST profile 可见头文件，也要跑：

```bash
cmake --preset host
cmake --build --preset host --parallel 8
ctest --test-dir build/host --output-on-failure -j 8
```

## 必跑实机验证

使用用户指定的 daemon 配置：

```bash
cmake --preset cuda
cmake --build --preset cuda --parallel 8 --target \
  tutti_daemon \
  nvmeservice_client_example \
  tutti_config_loader_test \
  tutti_storage_runtime_contract_test \
  tutti_striped_local_nvme_contract_test \
  <新增阶段5测试target>
```

启动 daemon：

```bash
sudo -S env TUTTI_VERBOSE=1 \
  build/cuda/tutti/device_manager/nvme/nvmeservice/examples/tutti_daemon \
  --config config/local/daemon_2disk.yaml < ~/.passwd/1
```

资源发现基线：

```bash
build/cuda/tutti/device_manager/nvme/nvmeservice/examples/nvmeservice_client \
  --endpoint 127.0.0.1:50051 --list-only
```

阶段 5 测试至少覆盖：

| ID | 场景 | 配置 | 预期结果 |
| --- | --- | --- | --- |
| A1 | Runtime 0 + Local NVMe 0 | `accel_id=0`, `explicit [0]` | loader 创建、真实 write/read/verify、shutdown、Release 成功 |
| A2 | Runtime 0 + Local NVMe 1 | `accel_id=0`, `explicit [1]` | 只改 daemon `device_id`，路径来自 allocation，真实 I/O 成功 |
| A3 | Runtime 0 + Striped NVMe 0/1 | `accel_id=0`, `striped [0,1]` | 一个顶层 striped resolver/DataPath，跨 stripe 数据正确，一次 Release 归还双盘 |
| A4 | Runtime 1 + Local NVMe 0 | `accel_id=1`, `explicit [0]` | GPU 1 上无 GPU 0 假设，真实 I/O 成功 |
| A5 | Runtime 1 + Local NVMe 1 | `accel_id=1`, `explicit [1]` | 只改 device ID，真实 I/O 成功 |
| A6 | Runtime 1 + Striped NVMe 0/1 | `accel_id=1`, `striped [0,1]` | workspace、DMA map、kernel 属于 GPU 1，数据正确 |
| C1 | current device 恢复 | 每个 A 场景前设置为非目标 GPU | API 返回后恢复调用方 current device |
| C6 | daemon no-context | daemon boot/list/acquire 前后检查 compute process | daemon 不创建 accelerator context |

测试结束后必须确认：

- daemon `ListNvmeResources` 中两个 controller 的 `reserved_queues=0`，
  `available_queues` 回到场景前值。
- 测试 scratch file 和临时配置文件已清理，或路径记录在实施文档中。
- daemon Ctrl-C clean exit，mount 和 view symlink 状态符合 daemon 既有清理契约。
- 不存在遗留测试进程、未释放 allocation 或长时间占用的 queue group。

## 完成规则

阶段 5 完成前必须满足：

1. 新增 loader-driven 单 Runtime 硬件 contract test，真实使用 `load_tutti_config()`
   和 public `StorageRuntime` API 完成单盘与 striped I/O。
2. GPU 0 和 GPU 1 都分别通过 explicit 单盘和 striped 双盘场景；如果环境 GPU 数不足，
   GPU 1 场景必须明确 skip 并在实施记录中说明，不能伪造通过。
3. 每个场景都证明 path、BAR0、namespace、block size、MDTS 和 queue grant 来自
   allocation metadata，不从配置或 ID 推导。
4. 每个场景都验证 shutdown/Release 后 daemon ledger 回到基线；striped 一次
   allocation 的 Release 归还所有 slice reservation。
5. current-device 恢复检查覆盖 initialize/open/register/submit/progress/wait/
   close/unregister/shutdown 中阶段 5 实际调用的路径。
6. 无硬件 contract 回归通过；实机测试命令、daemon list output、allocation metadata、
   queue ledger、scratch path、清理状态和残余风险写入
   `doc/impl/multi-accelerator-phase-5.md`。
7. 运行 `git diff --check`，确认没有生成物、日志、密码、临时 YAML 或无关改动。
8. 将阶段 5 实现、测试和实施记录作为独立 git commit 提交；建议提交说明为
   `runtime: validate multi-accelerator phase 5 loader e2e`。
9. 报告 commit hash、验证结果和残余风险，然后停止，不实施阶段 6。

阶段 5 的 exit gate 是：同一个 Runtime bundle 能通过 allocation-driven loader 在 GPU
0/1 上分别使用 NVMe 0、NVMe 1 和 striped NVMe 0/1 完成真实 I/O；所有资源在 shutdown
后准确释放，daemon 不创建 accelerator context，且产品 loader 仍只装配一组顶层
resolver/DataPath。
